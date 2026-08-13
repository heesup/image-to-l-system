import os
import sys
import time
import json
import math
import subprocess
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Add repository root to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser, OrganNode3D
from diffusion_based.models.legacy.helios_xml_writer_track_a import write_organ_nodes_to_xml
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer
from diffusion_based.models.legacy.graph_diffuser_3d_track_a import PlantGraphDiffuser3D
from diffusion_based.eval.visualize_diffusion_3d import sample_reverse_diffusion_3d


def setup_display_env() -> dict:
    """Ensure DISPLAY is configured correctly for headless GPU nodes."""
    env = os.environ.copy()
    import socket
    hostname = socket.gethostname()
    if "DISPLAY" not in env or not env["DISPLAY"]:
        if hostname.startswith("gpu-") or "farm" in hostname or torch.cuda.is_available():
            env["DISPLAY"] = ":1.0"
            print(f"[DISPLAY LOGIC] Host '{hostname}' detected -> export DISPLAY=:1.0")
    return env


def compute_ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM metric between two RGB numpy images in [0, 1]."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        return float(ssim_fn(img1, img2, channel_axis=-1, data_range=1.0))
    except Exception as e:
        mse = float(np.mean((img1 - img2) ** 2))
        return float(max(0.0, 1.0 - 5.0 * mse))


def compute_silhouette_iou(img1: np.ndarray, img2: np.ndarray, thresh: float = 0.1) -> float:
    """Compute silhouette IoU for non-background plant regions."""
    if img1.shape[-1] == 4:
        mask1 = img1[..., 3] > 0.05
    else:
        mask1 = np.linalg.norm(img1, axis=-1) > thresh

    if img2.shape[-1] == 4:
        mask2 = img2[..., 3] > 0.05
    else:
        mask2 = np.linalg.norm(img2, axis=-1) > thresh

    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter / union) if union > 0 else 1.0


def step1_generate_groundtruth_data(output_dir: str, dap: int = 1, seed: int = 42) -> dict:
    """Step 1: Generate initial GT plant XML & image using C++ Helios binary with --focus-plant."""
    print("\n=======================================================")
    print(f"STEP 1: Generating Ground Truth Data (DAP={dap}, Seed={seed})")
    print("=======================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    main_binary = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "main"
    )
    base_params_file = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "params.json"
    )
    
    assert os.path.exists(main_binary), f"Main C++ binary not found at {main_binary}"
    assert os.path.exists(base_params_file), f"Params JSON not found at {base_params_file}"
    
    with open(base_params_file, "r") as f:
        params = json.load(f)
        
    params.setdefault("camera", {}).setdefault("positioning", {})["azimuth_angle"] = 0.0
    params["camera"]["positioning"]["camera_height"] = 1.0
    params["camera"]["positioning"]["focusing_plants"] = True
    params.setdefault("metadata", {})["dap"] = int(dap)
    params["metadata"].pop("DAP", None)
    
    name = f"dap{dap}_groundtruth"
    tmp_params_path = os.path.join(output_dir, f"{name}_params.json")
    with open(tmp_params_path, "w") as f:
        json.dump(params, f, indent=2)
        
    cmd = [
        main_binary,
        "--renderer", "vis",
        "--save-xml",
        "--focus-plant",
        "-n", name,
        "--dap", str(dap),
        "-s", str(seed),
        "--output", output_dir,
        "-f", tmp_params_path,
    ]
    
    env = setup_display_env()
    print(f"Executing C++ Helios binary with --focus-plant: {' '.join(cmd)}")
    build_dir = os.path.dirname(main_binary)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=build_dir, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    print(f"C++ binary finished in {elapsed:.2f}s (returncode={res.returncode})")
    expected_img = os.path.join(output_dir, f"{name}_0000_vis.jpeg")
    if not os.path.exists(expected_img):
        alt_img = os.path.join(output_dir, f"{name}_0000.jpeg")
        if os.path.exists(alt_img):
            expected_img = alt_img
            
    expected_xml = os.path.join(output_dir, f"{name}_0000_plant_0000.xml")
    assert os.path.exists(expected_img), f"GT image not found at {expected_img}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert os.path.exists(expected_xml), f"GT XML not found at {expected_xml}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    
    print(f"[PASS] GT Image: {expected_img}")
    print(f"[PASS] GT XML:   {expected_xml}")
    
    return {
        "gt_img_path": expected_img,
        "gt_xml_path": expected_xml,
        "output_dir": output_dir,
        "dap": dap,
        "seed": seed,
    }


def step2_diffusion_proposal_and_refinement(
    gt_info: dict,
    output_dir: str,
    image_size: int = 256,
    num_iters: int = 200,
    lr: float = 0.01,
) -> dict:
    """Step 2: Propose 3D 15D graph nodes via Diffusion Model, then refine via Differentiable Renderer."""
    print("\n=======================================================")
    print("STEP 2: Diffusion Model 3D Prior Proposal & Differentiable Renderer Refinement")
    print("=======================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Parse GT XML
    parser = HeliosXMLParser(gt_info["gt_xml_path"])
    parser.parse()
    gt_organ_nodes = parser.get_all_organ_nodes()
    print(f"Parsed {len(gt_organ_nodes)} organ nodes from GT XML")
    
    gt_nodes_np = np.stack([n.to_15d() for n in gt_organ_nodes], axis=0)
    gt_nodes_tensor = torch.tensor(gt_nodes_np, dtype=torch.float32, device=device).unsqueeze(0)
    parents_tensor = torch.tensor([n.parent_idx for n in gt_organ_nodes], dtype=torch.long, device=device).unsqueeze(0)
    
    # 2. Render target input image with Differentiable Renderer
    rasterizer = HeliosGeometryRasterizer(image_size=image_size).to(device)
    renderer = DifferentiableHeliosRenderer(rasterizer).to(device)
    
    with torch.no_grad():
        target_rgba = renderer(
            gt_nodes_tensor,
            parents=parents_tensor,
            focus_plant=True,
            background="ground",
        )
    target_rgb_tensor = target_rgba[:, :3].detach()
    target_rgb_np = target_rgb_tensor[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    target_img_path = os.path.join(output_dir, "differentiable_target_input.png")
    Image.fromarray((target_rgb_np * 255).astype(np.uint8)).save(target_img_path)
    print(f"[PASS] Differentiable Target Input Image saved to: {target_img_path}")
    
    # 3. Diffusion Model 3D Prior proposal
    print("\n[DIFFUSION PRIOR] Running 3D Graph Diffuser reverse sampling...")
    max_nodes = 256
    diffuser_model = PlantGraphDiffuser3D(max_nodes=max_nodes, node_dim=15).to(device)
    
    checkpoint_path = os.path.join(repo_root, "diffusion_based", "checkpoints", "diffusion_model_3d.pt")
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                diffuser_model.load_state_dict(state["model_state_dict"])
            elif "model" in state:
                diffuser_model.load_state_dict(state["model"])
            else:
                diffuser_model.load_state_dict(state)
        else:
            diffuser_model.load_state_dict(state)
        print(f"[DIFFUSION MODEL] Loaded weights from: {checkpoint_path}")
    else:
        print("[DIFFUSION MODEL] No pre-trained checkpoint found; sampling from 3D Graph Diffuser prior initialization")
        
    diffuser_model.eval()
    
    # Run DDPM Reverse Diffusion Sampling
    img_pil = Image.open(gt_info["gt_img_path"]).convert("RGB").resize((image_size, image_size))
    img_t = torch.tensor(np.array(img_pil, dtype=np.float32)/255.0).permute(2, 0, 1).to(device)
    camera_pose = torch.tensor([0.0, 1.0], dtype=torch.float32).to(device)
    dap_t = torch.tensor([gt_info["dap"] / 90.0], dtype=torch.float32).to(device)
    
    results = sample_reverse_diffusion_3d(
        diffuser_model, img_t, camera_pose=camera_pose, dap=dap_t, steps=50
    )
    
    step_last = results["step_last"]
    diff_nodes_np = results["snapshots"][step_last]["nodes"][:len(gt_organ_nodes)]
    diff_nodes_tensor = torch.tensor(diff_nodes_np, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Render Diffusion Prior prediction
    with torch.no_grad():
        diff_pred_rgba = renderer(
            diff_nodes_tensor,
            parents=parents_tensor,
            focus_plant=True,
            background="ground",
        )
    diff_pred_np = diff_pred_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    # 4. Refine Diffusion prediction via Multi-Scale Differentiable Renderer Backpropagation
    print(f"\n[FINE-TUNING] Refining Diffusion 3D nodes via Backpropagation with Multi-Scale Softness Annealing ({num_iters} iters, lr={lr})...")
    nodes_param = diff_nodes_tensor.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([nodes_param], lr=lr)
    target_alpha_tensor = target_rgba[:, 3:4].detach()
    
    t0 = time.time()
    for i in range(1, num_iters + 1):
        optimizer.zero_grad()
        # Anneal leaf_sigma from 0.04 (wide basin) down to 0.002 (sharp pixel fitting)
        sigma_cur = 0.04 * (1.0 - i / num_iters) + 0.002 * (i / num_iters)
        pred_rgba = renderer(
            nodes_param,
            parents=parents_tensor,
            focus_plant=True,
            background="ground",
            leaf_sigma=sigma_cur,
        )
        loss_rgb = F.l1_loss(pred_rgba[:, :3], target_rgb_tensor)
        loss_sil = F.l1_loss(pred_rgba[:, 3:4], target_alpha_tensor)
        reg_len = F.relu(-nodes_param[:, :, 3]).mean() * 10.0
        reg_rad = F.relu(-nodes_param[:, :, 4]).mean() * 10.0
        
        loss = loss_rgb + 2.0 * loss_sil + reg_len + reg_rad
        loss.backward()
        
        optimizer.step()
        with torch.no_grad():
            nodes_param[:, :, 3].clamp_(min=0.005)
            nodes_param[:, :, 4].clamp_(min=0.001)
            nodes_param[:, :, 14].clamp_(min=0.0)
            
        if i == 1 or i % 50 == 0 or i == num_iters:
            print(f"Iter {i:03d}/{num_iters:03d} | Loss: {loss.item():.5f} (RGB: {loss_rgb.item():.5f}, Sil: {loss_sil.item():.5f}, Sigma: {sigma_cur:.4f})")
            
    print(f"[PASS] Multi-scale refinement finished in {time.time()-t0:.2f}s")
    
    with torch.no_grad():
        opt_pred_rgba = renderer(
            nodes_param,
            parents=parents_tensor,
            focus_plant=True,
            background="ground",
        )
    opt_pred_np = opt_pred_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    return {
        "gt_organ_nodes": gt_organ_nodes,
        "target_rgb_np": target_rgb_np,
        "diff_pred_np": diff_pred_np,
        "opt_pred_np": opt_pred_np,
        "opt_nodes_tensor": nodes_param.detach(),
        "parents_tensor": parents_tensor,
    }


def step3_rerender_diffusion_xml(gt_info: dict, opt_info: dict, output_dir: str) -> dict:
    """Step 3: Export fine-tuned Diffusion 15D nodes to XML and re-render with C++ Helios (--focus-plant)."""
    print("\n=======================================================")
    print("STEP 3: Re-rendering Fine-tuned XML via C++ Helios Binary with --focus-plant")
    print("=======================================================")
    
    dap = gt_info["dap"]
    gt_organ_nodes = opt_info["gt_organ_nodes"]
    opt_nodes_np = opt_info["opt_nodes_tensor"][0].cpu().numpy()
    
    opt_organ_nodes = []
    for orig_node, opt_vec in zip(gt_organ_nodes, opt_nodes_np):
        new_node = OrganNode3D.from_15d(opt_vec)
        new_node.organ_type = orig_node.organ_type
        new_node.parent_idx = orig_node.parent_idx
        new_node.shoot_id = orig_node.shoot_id
        new_node.phytomer_idx = orig_node.phytomer_idx
        opt_organ_nodes.append(new_node)
        
    opt_xml_path = os.path.abspath(os.path.join(output_dir, f"dap{dap}_diffusion_rerendered.xml"))
    write_organ_nodes_to_xml(opt_organ_nodes, opt_xml_path, plant_age=dap)
    print(f"[PASS] Exported fine-tuned XML to: {opt_xml_path}")
    
    main_binary = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "main"
    )
    base_params_file = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "params.json"
    )
    with open(base_params_file, "r") as f:
        params = json.load(f)
        
    params.setdefault("camera", {}).setdefault("positioning", {})["azimuth_angle"] = 0.0
    params["camera"]["positioning"]["camera_height"] = 1.0
    params["camera"]["positioning"]["focusing_plants"] = True
    params.setdefault("metadata", {})["dap"] = int(dap)
    
    params.setdefault("field", {}).setdefault("plots", [{}])[0].setdefault("plants", [{}])[0]["xml"] = opt_xml_path
    
    rerender_name = f"dap{dap}_diffusion_rerendered"
    tmp_params_path = os.path.join(output_dir, f"{rerender_name}_params.json")
    with open(tmp_params_path, "w") as f:
        json.dump(params, f, indent=2)
        
    cmd = [
        main_binary,
        "--renderer", "vis",
        "--focus-plant",
        "-n", rerender_name,
        "--dap", str(dap),
        "-s", str(gt_info["seed"]),
        "--output", output_dir,
        "-f", tmp_params_path,
    ]
    
    env = setup_display_env()
    print(f"Executing C++ Helios binary for re-render (--focus-plant): {' '.join(cmd)}")
    build_dir = os.path.dirname(main_binary)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=build_dir, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    print(f"C++ Re-render finished in {elapsed:.2f}s (returncode={res.returncode})")
    rerender_img_path = os.path.join(output_dir, f"{rerender_name}_0000_vis.jpeg")
    assert os.path.exists(rerender_img_path), f"Re-rendered image not found at {rerender_img_path}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    print(f"[PASS] Re-rendered C++ Image: {rerender_img_path}")
    
    # Compute quantitative metrics
    target_np = opt_info["target_rgb_np"]
    diff_pred_np = opt_info["diff_pred_np"]
    opt_pred_np = opt_info["opt_pred_np"]
    
    rerender_pil = Image.open(rerender_img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    rerender_np = np.array(rerender_pil, dtype=np.float32) / 255.0
    
    metrics = {
        "diff_mae": float(np.mean(np.abs(diff_pred_np - target_np))),
        "diff_ssim": compute_ssim_numpy(diff_pred_np, target_np),
        "diff_iou": compute_silhouette_iou(diff_pred_np, target_np),
        "opt_mae": float(np.mean(np.abs(opt_pred_np - target_np))),
        "opt_ssim": compute_ssim_numpy(opt_pred_np, target_np),
        "opt_iou": compute_silhouette_iou(opt_pred_np, target_np),
        "cpp_mae": float(np.mean(np.abs(rerender_np - target_np))),
        "cpp_ssim": compute_ssim_numpy(rerender_np, target_np),
        "cpp_iou": compute_silhouette_iou(rerender_np, target_np),
    }
    
    print("\n-------------------------------------------------------")
    print("QUANTITATIVE COMPARISON SUMMARY (Diffusion + Renderer vs Target)")
    print("-------------------------------------------------------")
    print(f"1. Diffusion Model 3D Prior:       MAE={metrics['diff_mae']:.4f}, SSIM={metrics['diff_ssim']:.4f}, IoU={metrics['diff_iou']:.4f}")
    print(f"2. Diff Renderer Fine-tuned:        MAE={metrics['opt_mae']:.4f}, SSIM={metrics['opt_ssim']:.4f}, IoU={metrics['opt_iou']:.4f}")
    print(f"3. C++ Helios Re-rendered (Focus): MAE={metrics['cpp_mae']:.4f}, SSIM={metrics['cpp_ssim']:.4f}, IoU={metrics['cpp_iou']:.4f}")
    print("-------------------------------------------------------")
    
    # 4-Panel Comparison Figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(target_np)
    axes[0].set_title("1. Target Input Image\n(Differentiable Render from GT)", fontsize=10, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(diff_pred_np)
    axes[1].set_title(f"2. Diffusion 3D Prior\nMAE={metrics['diff_mae']:.4f} | SSIM={metrics['diff_ssim']:.4f}", fontsize=10, fontweight="bold", color="purple")
    axes[1].axis("off")
    
    axes[2].imshow(opt_pred_np)
    axes[2].set_title(f"3. Diff Renderer Fine-tuned\nMAE={metrics['opt_mae']:.4f} | SSIM={metrics['opt_ssim']:.4f}", fontsize=10, fontweight="bold", color="navy")
    axes[2].axis("off")
    
    axes[3].imshow(rerender_np)
    axes[3].set_title(f"4. C++ Helios Re-rendered\n(--focus-plant from XML)\nMAE={metrics['cpp_mae']:.4f} | SSIM={metrics['cpp_ssim']:.4f}", fontsize=10, fontweight="bold", color="darkgreen")
    axes[3].axis("off")
    
    plt.tight_layout()
    comp_figure_path = os.path.join(output_dir, "comparison_diffusion_pipeline.png")
    plt.savefig(comp_figure_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PASS] Saved 4-panel comparison figure to: {comp_figure_path}")
    
    return {
        "opt_xml_path": opt_xml_path,
        "rerender_img_path": rerender_img_path,
        "comp_figure_path": comp_figure_path,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Diffusion + Differentiable Renderer Integrated Test")
    parser.add_argument("--dap", type=int, default=1, help="Days after planting (e.g. 1, 10)")
    args = parser.parse_args()
    
    dap = args.dap
    output_dir = os.path.join(repo_root, "notebooks", f"output_dap{dap}_diffusion_test")
    
    gt_info = step1_generate_groundtruth_data(output_dir=output_dir, dap=dap, seed=42)
    opt_info = step2_diffusion_proposal_and_refinement(gt_info=gt_info, output_dir=output_dir, image_size=256, num_iters=200)
    final_info = step3_rerender_diffusion_xml(gt_info=gt_info, opt_info=opt_info, output_dir=output_dir)
    
    print("\n=======================================================")
    print(f"DIFFUSION + DIFFERENTIABLE RENDERER DAP {dap} TEST COMPLETED!")
    print("=======================================================")


if __name__ == "__main__":
    main()
