"""Run 10 DAP plant generation, Differentiable Renderer target rendering, 15D node optimization with numerical stability monitoring, and C++ Helios re-rendering."""

import os
import sys
import json
import time
import socket
import tempfile
import subprocess
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# Ensure repo root is in python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser, OrganNode3D
from diffusion_based.models.legacy.helios_xml_writer_track_a import write_organ_nodes_to_xml
from diffusion_based.models.legacy.helios_geometry_track_a import nodes_to_geometry_torch
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer


def setup_display_env() -> dict:
    """Set DISPLAY environment variable if hostname starts with gpu- or DISPLAY is missing."""
    env = os.environ.copy()
    hostname = socket.gethostname()
    if hostname.startswith("gpu-") or "gpu" in hostname or "DISPLAY" not in env:
        env["DISPLAY"] = ":1.0"
        print(f"[DISPLAY LOGIC] Host '{hostname}' detected -> export DISPLAY=:1.0")
    else:
        print(f"[DISPLAY LOGIC] Host '{hostname}', DISPLAY={env.get('DISPLAY')}")
    return env


def compute_ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity (SSIM) between two RGB images (H, W, 3) in [0, 1]."""
    try:
        from skimage.metrics import structural_similarity as ssim
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=win_size))
    except Exception as e:
        print(f"Warning: skimage SSIM failed ({e}), returning fallback MSE-based similarity")
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


def step1_generate_groundtruth_data(output_dir: str, dap: int = 10, seed: int = 42) -> dict:
    """Step 1: Generate initial 10 DAP plant XML & image using C++ Helios binary (-n dap10_groundtruth)."""
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
    
    # Customize params.json for ground-truth generation
    with open(base_params_file, "r") as f:
        params = json.load(f)
        
    params.setdefault("camera", {}).setdefault("positioning", {})["azimuth_angle"] = 0.0
    params["camera"]["positioning"]["camera_height"] = 5.0
    params["camera"]["positioning"]["focusing_plants"] = True
    params.setdefault("metadata", {})["dap"] = int(dap)
    params["metadata"].pop("DAP", None)
    
    # Requirement 2: Unique output name prefix
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
    print(f"Executing C++ Helios binary: {' '.join(cmd)}")
    build_dir = os.path.dirname(main_binary)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=build_dir, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    print(f"C++ binary finished in {elapsed:.2f}s (returncode={res.returncode})")
    
    gt_img_path = os.path.join(output_dir, f"{name}_0000_vis.jpeg")
    if not os.path.exists(gt_img_path):
        alt_img = os.path.join(output_dir, f"{name}_0000.jpeg")
        if os.path.exists(alt_img):
            gt_img_path = alt_img
            
    gt_xml_path = os.path.join(output_dir, f"{name}_0000_plant_0000.xml")
    
    assert os.path.exists(gt_img_path), f"GT Image not found at {gt_img_path}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert os.path.exists(gt_xml_path), f"GT XML not found at {gt_xml_path}"
    
    print(f"[PASS] GT Image: {gt_img_path}")
    print(f"[PASS] GT XML:   {gt_xml_path}")
    
    return {
        "gt_img_path": gt_img_path,
        "gt_xml_path": gt_xml_path,
        "name": name,
        "dap": dap,
        "seed": seed,
    }


def step2_render_differentiable_target_and_optimize(
    gt_info: dict,
    output_dir: str,
    image_size: int = 256,
    num_iters: int = 200,
    lr: float = 0.01,
    noise_scale: float = 0.05,
) -> dict:
    print("\n=======================================================")
    print("STEP 2: Render Differentiable Target & Optimize 15D Array")
    print("=======================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Parse GT XML to 15D organ nodes
    parser = HeliosXMLParser(gt_info["gt_xml_path"])
    parser.parse()
    gt_organ_nodes = parser.get_all_organ_nodes()
    print(f"Parsed {len(gt_organ_nodes)} organ nodes from GT XML")
    
    gt_nodes_np = np.stack([n.to_15d() for n in gt_organ_nodes], axis=0)
    gt_nodes_tensor = torch.tensor(gt_nodes_np, dtype=torch.float32, device=device).unsqueeze(0) # (1, N, 15)
    parents_tensor = torch.tensor([n.parent_idx for n in gt_organ_nodes], dtype=torch.long, device=device).unsqueeze(0)
    
    # Setup differentiable renderer
    rasterizer = HeliosGeometryRasterizer(image_size=image_size).to(device)
    renderer = DifferentiableHeliosRenderer(rasterizer).to(device)
    
    # 2. Render target input image directly using Differentiable Renderer (Requirement 3)
    with torch.no_grad():
        target_rgba = renderer(
            gt_nodes_tensor,
            parents=parents_tensor,
            camera_height=5.0,
            focus_plant=True,
            background="ground",
        )
    
    target_rgb_tensor = target_rgba[:, :3].detach() # (1, 3, H, W)
    target_rgb_np = target_rgb_tensor[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) # (H, W, 3)
    
    # Save target input image
    target_img_path = os.path.join(output_dir, "differentiable_target_input.png")
    Image.fromarray((target_rgb_np * 255).astype(np.uint8)).save(target_img_path)
    print(f"[PASS] Differentiable Target Input Image created & saved to: {target_img_path}")
    
    # 3. Perturb parameters for optimization testing
    torch.manual_seed(42)
    init_nodes_tensor = gt_nodes_tensor.clone()
    noise = torch.randn_like(init_nodes_tensor[:, :, :8]) * noise_scale
    init_nodes_tensor[:, :, :8] += noise
    
    nodes_param = init_nodes_tensor.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([nodes_param], lr=lr)
    
    history = {
        "loss": [],
        "loss_rgb": [],
        "grad_norm": [],
        "nan_count": 0,
        "inf_count": 0,
        "param_err": [],
    }
    
    print(f"Starting optimization against Differentiable Target Input for {num_iters} iterations (lr={lr}, noise={noise_scale})...")
    t0 = time.time()
    
    for i in range(1, num_iters + 1):
        optimizer.zero_grad()
        
        pred_rgba = renderer(
            nodes_param,
            parents=parents_tensor,
            focus_plant=True,
            background="ground",
        )
        
        # Loss between predicted render and target differentiable input image
        loss_rgb = F.l1_loss(pred_rgba[:, :3], target_rgb_tensor)
        reg_len = F.relu(-nodes_param[:, :, 3]).mean() * 10.0
        reg_rad = F.relu(-nodes_param[:, :, 4]).mean() * 10.0
        
        loss = loss_rgb + reg_len + reg_rad
        loss.backward()
        
        # Stability diagnostics
        grad = nodes_param.grad
        has_nan = torch.isnan(grad).any().item() or torch.isnan(pred_rgba).any().item()
        has_inf = torch.isinf(grad).any().item() or torch.isinf(pred_rgba).any().item()
        
        if has_nan:
            history["nan_count"] += 1
        if has_inf:
            history["inf_count"] += 1
            
        grad_norm = grad.norm().item() if grad is not None else 0.0
        param_err = (nodes_param.detach() - gt_nodes_tensor).abs().mean().item()
        
        history["loss"].append(loss.item())
        history["loss_rgb"].append(loss_rgb.item())
        history["grad_norm"].append(grad_norm)
        history["param_err"].append(param_err)
        
        optimizer.step()
        
        # Physical constraints projection
        with torch.no_grad():
            nodes_param[:, :, 3].clamp_(min=0.005)
            nodes_param[:, :, 4].clamp_(min=0.001)
            nodes_param[:, :, 14].clamp_(min=0.0)
            
        if i == 1 or i % 50 == 0 or i == num_iters:
            print(f"Iter {i:03d}/{num_iters:03d} | Loss: {loss.item():.5f} (RGB: {loss_rgb.item():.5f}) | GradNorm: {grad_norm:.4f} | ParamErr: {param_err:.5f}")
            
    print(f"Optimization finished in {time.time()-t0:.2f}s | NaNs: {history['nan_count']} | Infs: {history['inf_count']}")
    
    with torch.no_grad():
        final_pred_rgba = renderer(
            nodes_param,
            parents=parents_tensor,
            focus_plant=True,
            background="ground",
        )
    opt_pred_rgb = final_pred_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    # Save diagnostics plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["loss"], label="Total Loss", color="crimson")
    axes[0].plot(history["loss_rgb"], label="RGB L1 Loss", color="navy", linestyle="--")
    axes[0].set_title("Loss Trajectory (vs Target Diff Render)")
    axes[0].set_xlabel("Iteration")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(history["grad_norm"], color="darkgreen")
    axes[1].set_title("Gradient Norm (Numerical Stability)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history["param_err"], color="purple")
    axes[2].set_title("15D Node Parameter MAE vs GT")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    diag_plot_path = os.path.join(output_dir, "optimization_diagnostics.png")
    plt.savefig(diag_plot_path, dpi=150)
    plt.close()
    
    return {
        "gt_organ_nodes": gt_organ_nodes,
        "opt_nodes_tensor": nodes_param.detach(),
        "parents_tensor": parents_tensor,
        "target_rgb_np": target_rgb_np,
        "opt_pred_rgb": opt_pred_rgb,
        "history": history,
        "diag_plot_path": diag_plot_path,
    }


def step3_rerender_optimized_xml(
    gt_info: dict,
    opt_info: dict,
    output_dir: str,
) -> dict:
    """Step 3: Export optimized 15D nodes to XML and re-render using C++ Helios binary (-n dap10_rerendered)."""
    print("\n=======================================================")
    print("STEP 3: Re-rendering Optimized XML via C++ Helios Binary")
    print("=======================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    gt_organ_nodes = opt_info["gt_organ_nodes"]
    opt_nodes_np = opt_info["opt_nodes_tensor"][0].cpu().numpy()
    
    # Update OrganNode3D objects
    opt_organ_nodes = []
    for orig_node, opt_vec in zip(gt_organ_nodes, opt_nodes_np):
        new_node = OrganNode3D.from_15d(opt_vec)
        new_node.parent_idx = orig_node.parent_idx
        new_node.shoot_id = orig_node.shoot_id
        new_node.phytomer_idx = orig_node.phytomer_idx
    dap = gt_info["dap"]
    opt_xml_path = os.path.abspath(os.path.join(output_dir, f"dap{dap}_rerendered.xml"))
    write_organ_nodes_to_xml(opt_organ_nodes, opt_xml_path, plant_age=dap)
    print(f"[PASS] Exported optimized XML to: {opt_xml_path}")
    
    # Requirement 2: Pass XML file path inside params.json so C++ binary loads it properly
    main_binary = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "main"
    )
    base_params_file = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "params.json"
    )
    with open(base_params_file, "r") as f:
        params = json.load(f)
        
    params.setdefault("camera", {}).setdefault("positioning", {})["azimuth_angle"] = 0.0
    params["camera"]["positioning"]["camera_height"] = 5.0
    params["camera"]["positioning"]["focusing_plants"] = True
    params.setdefault("metadata", {})["dap"] = int(gt_info["dap"])
    
    # CRITICAL: Point params.json field plot plant[0] to the optimized XML file
    params.setdefault("field", {}).setdefault("plots", [{}])[0].setdefault("plants", [{}])[0]["xml"] = opt_xml_path
    
    # Requirement 2: Unique output name prefix for re-render
    rerender_name = f"dap{dap}_rerendered"
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
    
    # Requirement 1: DISPLAY environment setup
    env = setup_display_env()
        
    print(f"Executing C++ Helios binary for re-render: {' '.join(cmd)}")
    build_dir = os.path.dirname(main_binary)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=build_dir, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    print(f"C++ Re-render finished in {elapsed:.2f}s (returncode={res.returncode})")
    
    rerender_img_path = os.path.join(output_dir, f"{rerender_name}_0000_vis.jpeg")
    if not os.path.exists(rerender_img_path):
        alt_img = os.path.join(output_dir, f"{rerender_name}_0000.jpeg")
        if os.path.exists(alt_img):
            rerender_img_path = alt_img
            
    assert os.path.exists(rerender_img_path), f"Re-rendered image not found at {rerender_img_path}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    print(f"[PASS] Re-rendered C++ Image: {rerender_img_path}")
    
    # -------------------------------------------------------------
    # 3-Panel Visual & Quantitative Evaluation
    # -------------------------------------------------------------
    target_np = opt_info["target_rgb_np"]
    opt_pred_np = opt_info["opt_pred_rgb"]
    
    rerender_pil = Image.open(rerender_img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    rerender_np = np.array(rerender_pil, dtype=np.float32) / 255.0
    
    # Compute metrics against Differentiable Target Input Image
    mae_pred = float(np.mean(np.abs(opt_pred_np - target_np)))
    mse_pred = float(np.mean((opt_pred_np - target_np) ** 2))
    ssim_pred = compute_ssim_numpy(opt_pred_np, target_np)
    iou_pred = compute_silhouette_iou(opt_pred_np, target_np)
    
    mae_rerender = float(np.mean(np.abs(rerender_np - target_np)))
    mse_rerender = float(np.mean((rerender_np - target_np) ** 2))
    ssim_rerender = compute_ssim_numpy(rerender_np, target_np)
    iou_rerender = compute_silhouette_iou(rerender_np, target_np)
    
    print("\n-------------------------------------------------------")
    print("QUANTITATIVE COMPARISON SUMMARY (vs Target Diff Render)")
    print("-------------------------------------------------------")
    print(f"Optimized Differentiable Render: MAE={mae_pred:.4f}, MSE={mse_pred:.4f}, SSIM={ssim_pred:.4f}, IoU={iou_pred:.4f}")
    print(f"C++ Helios Re-rendered Image:    MAE={mae_rerender:.4f}, MSE={mse_rerender:.4f}, SSIM={ssim_rerender:.4f}, IoU={iou_rerender:.4f}")
    print("-------------------------------------------------------")
    
    # Plot 3-panel comparison figure with full quantitative metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].imshow(target_np)
    axes[0].set_title("Input Target Image\n(Differentiable Render from GT XML)", fontsize=11, fontweight="bold", pad=10)
    axes[0].axis("off")
    
    axes[1].imshow(opt_pred_np)
    axes[1].set_title(
        f"Optimized Differentiable Render\n"
        f"MAE: {mae_pred:.4f} | MSE: {mse_pred:.4f}\n"
        f"SSIM: {ssim_pred:.4f} | IoU: {iou_pred:.4f}",
        fontsize=10, fontweight="bold", pad=10, color="navy"
    )
    axes[1].axis("off")
    
    axes[2].imshow(rerender_np)
    axes[2].set_title(
        f"C++ Helios Re-rendered Image\n"
        f"MAE: {mae_rerender:.4f} | MSE: {mse_rerender:.4f}\n"
        f"SSIM: {ssim_rerender:.4f} | IoU: {iou_rerender:.4f}",
        fontsize=10, fontweight="bold", pad=10, color="darkgreen"
    )
    axes[2].axis("off")
    
    plt.tight_layout()
    comp_figure_path = os.path.join(output_dir, "comparison_3panel.png")
    plt.savefig(comp_figure_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PASS] Updated and saved 3-panel comparison figure to: {comp_figure_path}")
    
    return {
        "opt_xml_path": opt_xml_path,
        "rerender_img_path": rerender_img_path,
        "comp_figure_path": comp_figure_path,
        "metrics": {
            "pred_mae": mae_pred,
            "pred_ssim": ssim_pred,
            "pred_iou": iou_pred,
            "rerender_mae": mae_rerender,
            "rerender_ssim": ssim_rerender,
            "rerender_iou": iou_rerender,
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Differentiable Renderer Stability Test")
    parser.add_argument("--dap", type=int, default=1, help="Days after planting (e.g., 1, 10)")
    args = parser.parse_args()
    
    dap = args.dap
    output_dir = os.path.join(repo_root, "notebooks", f"output_dap{dap}_stability")
    
    # Step 1: Generate Ground Truth Data
    gt_info = step1_generate_groundtruth_data(output_dir=output_dir, dap=dap, seed=42)
    
    # Step 2 & Requirement 3: Render Diff Target Image & Optimize perturbed 15D nodes
    opt_info = step2_render_differentiable_target_and_optimize(
        gt_info=gt_info,
        output_dir=output_dir,
        image_size=256,
        num_iters=200,
        lr=0.01,
        noise_scale=0.05,
    )
    
    # Step 3: Re-render XML using C++ Helios binary
    final_info = step3_rerender_optimized_xml(
        gt_info=gt_info,
        opt_info=opt_info,
        output_dir=output_dir,
    )
    
    print("\n=======================================================")
    print(f"DAP {dap} PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
