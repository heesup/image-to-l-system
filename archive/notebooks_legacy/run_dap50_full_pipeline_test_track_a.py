"""DAP 50 End-to-End Pipeline Test: Data Generation -> Model Training -> DDPM Inference & Evaluation.

Performs:
1. Ground truth DAP 50 data generation using C++ Helios binary (main) with camera_height=5.0 and --focus-plant across multiple seeds.
2. 15D Graph Diffusion Model Training (PlantGraphDiffuser3D) on DAP 50 dataset.
3. DDPM Reverse Diffusion Inference sampling for DAP 50, converting predicted 15D graph to XML and re-rendering via C++ Helios binary for quantitative evaluation (SSIM, MAE, IoU).
"""

import os
import sys
import json
import time
import socket
import subprocess
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure repo root is in python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser, OrganNode3D
from diffusion_based.models.legacy.helios_xml_writer_track_a import write_organ_nodes_to_xml
from diffusion_based.models.legacy.helios_geometry_track_a import nodes_to_geometry_torch
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer
from diffusion_based.models.legacy.graph_diffuser_3d_track_a import PlantGraphDiffuser3D
from diffusion_based.training.train_diffusion_3d import DDPMScheduler, compute_losses


def setup_display_env() -> dict:
    """Set DISPLAY environment variable for Helios renderer."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATASET GENERATION (DAP 50)
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_generate_dap50_dataset(output_dir: str, seeds: list = [42, 43, 44, 45, 46]) -> list:
    """Generate DAP 50 dataset using C++ Helios binary with camera_height=5.0 and --focus-plant."""
    print("\n=======================================================")
    print(f"PHASE 1: Generating DAP 50 Dataset ({len(seeds)} samples)")
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
    build_dir = os.path.dirname(main_binary)
    env = setup_display_env()

    samples = []
    for s_idx, seed in enumerate(seeds):
        name = f"dap50_seed{seed:02d}"
        with open(base_params_file, "r") as f:
            params = json.load(f)
            
        params.setdefault("camera", {}).setdefault("positioning", {})["azimuth_angle"] = 0.0
        params["camera"]["positioning"]["camera_height"] = 5.0
        params["camera"]["positioning"]["focusing_plants"] = True
        params.setdefault("metadata", {})["dap"] = 50
        params["metadata"].pop("DAP", None)
        
        tmp_params_path = os.path.join(output_dir, f"{name}_params.json")
        with open(tmp_params_path, "w") as f:
            json.dump(params, f, indent=2)
            
        cmd = [
            main_binary,
            "--renderer", "vis",
            "--save-xml",
            "--focus-plant",
            "-n", name,
            "--dap", "50",
            "-s", str(seed),
            "--output", output_dir,
            "-f", tmp_params_path,
        ]
        
        t0 = time.time()
        res = subprocess.run(cmd, cwd=build_dir, env=env, capture_output=True, text=True)
        elapsed = time.time() - t0
        
        img_path = os.path.join(output_dir, f"{name}_0000_vis.jpeg")
        if not os.path.exists(img_path):
            alt = os.path.join(output_dir, f"{name}_0000.jpeg")
            if os.path.exists(alt):
                img_path = alt
                
        xml_path = os.path.join(output_dir, f"{name}_0000_plant_0000.xml")
        
        assert os.path.exists(img_path), f"Sample {name} image failed!\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
        assert os.path.exists(xml_path), f"Sample {name} XML failed!"
        
        print(f"[{s_idx+1}/{len(seeds)}] Generated GT sample: {name} ({elapsed:.2f}s)")
        samples.append({
            "name": name,
            "seed": seed,
            "img_path": img_path,
            "xml_path": xml_path,
            "params_path": tmp_params_path,
        })
        
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: MODEL TRAINING (DAP 50)
# ═══════════════════════════════════════════════════════════════════════════════

class DAP50Dataset(Dataset):
    """PyTorch Dataset for DAP 50 samples."""
    def __init__(self, samples: list, max_nodes: int = 256, image_size: int = 256):
        self.samples = samples
        self.max_nodes = max_nodes
        self.image_size = image_size
        
        self.data = []
        for s in samples:
            parser = HeliosXMLParser(s["xml_path"])
            parser.parse()
            organ_nodes = parser.get_all_organ_nodes()
            
            # Load and preprocess image
            img_pil = Image.open(s["img_path"]).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
            # Normalize with ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_np - mean) / std
            img_tensor = torch.tensor(img_norm).permute(2, 0, 1)  # (3, H, W)
            
            # 15D nodes matrix & parent indices
            N = len(organ_nodes)
            nodes_mat = np.zeros((max_nodes, 15), dtype=np.float32)
            parents_mat = np.full(max_nodes, -1, dtype=np.int64)
            existence_mat = np.zeros(max_nodes, dtype=np.float32)
            
            n_cap = min(N, max_nodes)
            for i in range(n_cap):
                nodes_mat[i] = organ_nodes[i].to_15d()
                parents_mat[i] = min(organ_nodes[i].parent_idx, n_cap - 1)
                existence_mat[i] = organ_nodes[i].existence
                
            self.data.append({
                "image": img_tensor,
                "nodes": torch.tensor(nodes_mat, dtype=torch.float32),
                "parents": torch.tensor(parents_mat, dtype=torch.long),
                "existence": torch.tensor(existence_mat, dtype=torch.float32),
                "num_nodes": n_cap,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def phase2_train_dap50_diffusion(
    samples: list,
    save_path: str,
    num_epochs: int = 100,
    lr: float = 3e-4,
    max_nodes: int = 256,
) -> PlantGraphDiffuser3D:
    """Train PlantGraphDiffuser3D model on DAP 50 dataset."""
    print("\n=======================================================")
    print(f"PHASE 2: Training 3D Graph Diffusion Model on DAP 50 ({num_epochs} epochs)")
    print("=======================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = DAP50Dataset(samples, max_nodes=max_nodes)
    dataloader = DataLoader(dataset, batch_size=len(samples), shuffle=True)
    
    model = PlantGraphDiffuser3D(max_nodes=max_nodes, node_dim=15, embed_dim=256, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = DDPMScheduler(timesteps=1000)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.train()
    
    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        for batch in dataloader:
            images = batch["image"].to(device)         # (B, 3, H, W)
            gt_nodes = batch["nodes"].to(device)       # (B, N, 15)
            gt_parents = batch["parents"].to(device)   # (B, N)
            gt_existence = batch["existence"].to(device) # (B, N)
            
            B = images.shape[0]
            timesteps = scheduler.sample_timesteps(B, device)
            
            # Add noise to 15D node features
            noisy_nodes, noise = scheduler.add_noise(gt_nodes, timesteps)
            
            # Dummy adj matrix & camera pose
            gt_adj = torch.zeros(B, max_nodes, max_nodes, device=device)
            camera_poses = torch.zeros(B, 2, device=device)
            dap_tensor = torch.full((B, 1), 50.0, device=device)
            
            outputs = model(
                noisy_nodes,
                gt_existence.unsqueeze(-1),
                timesteps,
                images,
                camera_poses=camera_poses,
                dap=dap_tensor,
            )
            
            loss, metrics = compute_losses(
                outputs,
                gt_nodes,
                gt_existence,
                gt_parents,
                gt_adj,
                noisy_nodes,
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if epoch % 20 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"Epoch [{epoch:03d}/{num_epochs}] | Total Loss: {avg_loss:.4f} | Coord MSE: {metrics['coord']:.4f} | Exist BCE: {metrics['existence']:.4f} | Parent CE: {metrics['parent']:.4f}")

    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.2f}s")
    
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, save_path)
    print(f"[PASS] Saved DAP 50 diffusion model checkpoint to: {save_path}")
    
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: INFERENCE & EVALUATION (DAP 50)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def phase3_inference_and_evaluation(
    model: PlantGraphDiffuser3D,
    gt_sample: dict,
    output_dir: str,
    image_size: int = 256,
    num_ddpm_steps: int = 50,
) -> dict:
    """Run DDPM inference, convert 15D graph to XML, re-render via C++ Helios, and calculate quantitative metrics."""
    print("\n=======================================================")
    print("PHASE 3: DDPM Reverse Diffusion Inference & Evaluation")
    print("=======================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    
    # Load GT sample image
    img_pil = Image.open(gt_sample["img_path"]).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_np - mean) / std
    img_tensor = torch.tensor(img_norm, device=device).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # 1. DDPM Reverse Diffusion Sampling
    scheduler = DDPMScheduler(timesteps=1000)
    step_indices = torch.linspace(999, 0, num_ddpm_steps).long().to(device)
    
    max_nodes = model.max_nodes
    x_t = torch.randn(1, max_nodes, 15, device=device)
    e_t = torch.ones(1, max_nodes, 1, device=device)
    camera_poses = torch.zeros(1, 2, device=device)
    dap_tensor = torch.full((1, 1), 50.0, device=device)
    
    print(f"Sampling DAP 50 15D graph using {num_ddpm_steps} DDPM reverse steps...")
    for t_idx, t in enumerate(step_indices):
        t_batch = torch.tensor([t], device=device).long()
        outputs = model(x_t, e_t, t_batch, img_tensor, camera_poses=camera_poses, dap=dap_tensor)
        pred_x0 = torch.clamp(outputs["pred_x0"], 0.0, 1.0)
        
        # DDPM step update
        if t_idx < len(step_indices) - 1:
            alpha_t = scheduler.alphas_cumprod[t].to(device)
            alpha_prev = scheduler.alphas_cumprod[step_indices[t_idx + 1]].to(device)
            beta_t = 1.0 - alpha_t / alpha_prev
            noise = torch.randn_like(x_t)
            x_t = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1.0 - alpha_prev) * noise
        else:
            x_t = pred_x0
            
    final_pred_nodes = pred_x0[0]                       # (N, 15)
    pred_existence = torch.sigmoid(outputs["pred_existence_logits"])[0] # (N,)
    
    # Sparse parent index extraction
    parent_candidates = outputs["pred_parent_candidates"][0] # (N, k)
    parent_logits = outputs["pred_parent_logits"][0]         # (N, k)
    best_k = torch.argmax(parent_logits, dim=-1)             # (N,)
    pred_parents = torch.gather(parent_candidates, 1, best_k.unsqueeze(-1)).squeeze(-1) # (N,)
    
    # 2. Parse 15D predicted nodes into OrganNode3D objects
    exist_prob = pred_existence.cpu().numpy()
    
    # Adaptive threshold + Top-K guarantee (ensure active nodes are extracted)
    th = float(np.percentile(exist_prob, 30))  # Top 70% highest confidence nodes
    th = min(th, 0.3)
    active_mask = (exist_prob >= th)
    if active_mask.sum() < 30:
        top_k_indices = np.argsort(exist_prob)[-100:]
        active_mask = np.zeros(max_nodes, dtype=bool)
        active_mask[top_k_indices] = True

    pred_nodes_np = final_pred_nodes.cpu().numpy()
    parents_np = pred_parents.cpu().numpy()
    
    organ_nodes = []
    node_idx_map = {}
    for i in range(max_nodes):
        if active_mask[i]:
            node_idx_map[i] = len(organ_nodes)
            node = OrganNode3D.from_15d(pred_nodes_np[i])
            node.existence = 1.0  # Mark active for rendering
            organ_nodes.append(node)
            
    # Remap parent indices
    for i, orig_idx in enumerate(sorted(node_idx_map.keys())):
        p_orig = int(parents_np[orig_idx])
        organ_nodes[i].parent_idx = node_idx_map.get(p_orig, 0)
        
    if organ_nodes:
        organ_nodes[0].parent_idx = 0

    # Reconstruct exact shoot/phytomer tree hierarchy from parent graph
    N_nodes = len(organ_nodes)
    if N_nodes > 0:
        next_shoot_id = 1
        organ_nodes[0].shoot_id = 0
        organ_nodes[0].phytomer_idx = 0
        
        for i in range(N_nodes):
            p_idx = organ_nodes[i].parent_idx
            p_node = organ_nodes[p_idx] if (0 <= p_idx < N_nodes and p_idx != i) else None
            
            if i == 0 or p_node is None:
                organ_nodes[i].shoot_id = 0
                organ_nodes[i].phytomer_idx = 0
                continue
                
            if organ_nodes[i].organ_type == OrganNode3D.INTERNODE:
                if p_node.organ_type == OrganNode3D.INTERNODE:
                    # Continuation of parent's shoot
                    organ_nodes[i].shoot_id = p_node.shoot_id
                    organ_nodes[i].phytomer_idx = p_node.phytomer_idx + 1
                else:
                    # Branching off petiole/leaf -> New secondary shoot!
                    organ_nodes[i].shoot_id = next_shoot_id
                    next_shoot_id += 1
                    organ_nodes[i].phytomer_idx = 0
            else:
                # Petiole, Leaf, Bud -> Same shoot & phytomer as parent
                organ_nodes[i].shoot_id = p_node.shoot_id
                organ_nodes[i].phytomer_idx = p_node.phytomer_idx

    shoots_count = len(set(n.shoot_id for n in organ_nodes))
    phytomers_count = len(set((n.shoot_id, n.phytomer_idx) for n in organ_nodes))
    print(f"Extracted {len(organ_nodes)} active 15D organ nodes across {shoots_count} shoots and {phytomers_count} phytomers for DAP 50 inference.")

    # Export predicted 15D graph to XML
    pred_xml_path = os.path.join(output_dir, "dap50_inference_pred.xml")
    write_organ_nodes_to_xml(organ_nodes, pred_xml_path, plant_age=50.0)
    print(f"[PASS] Saved inferred DAP 50 XML to: {pred_xml_path}")
    
    # 3. Render Differentiable Prediction vs Target Differentiable Render
    from diffusion_based.models.legacy.helios_geometry_track_a import build_helios_geometry_from_nodes
    rasterizer = HeliosGeometryRasterizer(image_size=image_size).to(device)
    geom_pred = build_helios_geometry_from_nodes(organ_nodes)
    diff_pred_rgba = rasterizer.render_numpy_geometry(
        geom_pred.tubes, geom_pred.leaflets, geom_pred.ellipsoids,
        camera_height=5.0, focus_plant=True, background="ground"
    )
        
    if isinstance(diff_pred_rgba, torch.Tensor):
        diff_pred_rgb = diff_pred_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    else:
        if diff_pred_rgba.shape[0] == 4:
            diff_pred_rgb = np.transpose(diff_pred_rgba[:3], (1, 2, 0)).clip(0, 1)
        else:
            diff_pred_rgb = diff_pred_rgba[:, :, :3].clip(0, 1)
    
    # 4. C++ Helios Re-rendering of Predicted XML
    main_binary = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "main"
    )
    base_params_file = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "params.json"
    )
    build_dir = os.path.dirname(main_binary)
    env = setup_display_env()

    with open(base_params_file, "r") as f:
        params = json.load(f)
        
    params.setdefault("camera", {}).setdefault("positioning", {})["azimuth_angle"] = 0.0
    params["camera"]["positioning"]["camera_height"] = 5.0
    params["camera"]["positioning"]["focusing_plants"] = True
    params.setdefault("metadata", {})["dap"] = 50
    params.setdefault("field", {}).setdefault("plots", [{}])[0].setdefault("plants", [{}])[0]["xml"] = pred_xml_path
    
    rerender_name = "dap50_inference_rerendered"
    tmp_params_path = os.path.join(output_dir, f"{rerender_name}_params.json")
    with open(tmp_params_path, "w") as f:
        json.dump(params, f, indent=2)
        
    cmd = [
        main_binary,
        "--renderer", "vis",
        "--focus-plant",
        "-n", rerender_name,
        "--dap", "50",
        "-s", "42",
        "--output", output_dir,
        "-f", tmp_params_path,
    ]
    
    print(f"Executing C++ Helios re-render: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=build_dir, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(f"C++ Re-render finished in {elapsed:.2f}s (returncode={res.returncode})")
    
    rerender_img_path = os.path.join(output_dir, f"{rerender_name}_0000_vis.jpeg")
    if not os.path.exists(rerender_img_path):
        alt = os.path.join(output_dir, f"{rerender_name}_0000.jpeg")
        if os.path.exists(alt):
            rerender_img_path = alt
            
    assert os.path.exists(rerender_img_path), f"C++ Re-render image failed!\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    
    # Compute quantitative evaluation metrics against GT Image
    gt_rgb = img_np
    pred_diff_rgb = diff_pred_rgb
    rerender_pil = Image.open(rerender_img_path).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    rerender_rgb = np.array(rerender_pil, dtype=np.float32) / 255.0
    
    mae_diff = float(np.mean(np.abs(pred_diff_rgb - gt_rgb)))
    mse_diff = float(np.mean((pred_diff_rgb - gt_rgb) ** 2))
    ssim_diff = compute_ssim_numpy(pred_diff_rgb, gt_rgb)
    iou_diff = compute_silhouette_iou(pred_diff_rgb, gt_rgb)
    
    mae_cpp = float(np.mean(np.abs(rerender_rgb - gt_rgb)))
    mse_cpp = float(np.mean((rerender_rgb - gt_rgb) ** 2))
    ssim_cpp = compute_ssim_numpy(rerender_rgb, gt_rgb)
    iou_cpp = compute_silhouette_iou(rerender_rgb, gt_rgb)
    
    print("\n-------------------------------------------------------")
    print("DAP 50 QUANTITATIVE EVALUATION SUMMARY (vs GT Image)")
    print("-------------------------------------------------------")
    print(f"Predicted Differentiable Render: MAE={mae_diff:.4f}, MSE={mse_diff:.4f}, SSIM={ssim_diff:.4f}, IoU={iou_diff:.4f}")
    print(f"C++ Helios Re-rendered Image:    MAE={mae_cpp:.4f}, MSE={mse_cpp:.4f}, SSIM={ssim_cpp:.4f}, IoU={iou_cpp:.4f}")
    print("-------------------------------------------------------")
    
    # 5. Save 3-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth Image (DAP 50)\n(C++ Helios Target)", fontsize=11, fontweight="bold", pad=10)
    axes[0].axis("off")
    
    axes[1].imshow(pred_diff_rgb)
    axes[1].set_title(
        f"Inferred 15D Diff Render\n"
        f"MAE: {mae_diff:.4f} | MSE: {mse_diff:.4f}\n"
        f"SSIM: {ssim_diff:.4f} | IoU: {iou_diff:.4f}",
        fontsize=10, fontweight="bold", pad=10, color="navy"
    )
    axes[1].axis("off")
    
    axes[2].imshow(rerender_rgb)
    axes[2].set_title(
        f"Inferred C++ Helios Re-render\n"
        f"MAE: {mae_cpp:.4f} | MSE: {mse_cpp:.4f}\n"
        f"SSIM: {ssim_cpp:.4f} | IoU: {iou_cpp:.4f}",
        fontsize=10, fontweight="bold", pad=10, color="darkgreen"
    )
    axes[2].axis("off")
    
    plt.tight_layout()
    comp_fig_path = os.path.join(output_dir, "comparison_3panel_dap50.png")
    plt.savefig(comp_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PASS] Saved 3-panel comparison figure to: {comp_fig_path}")
    
    return {
        "pred_xml_path": pred_xml_path,
        "rerender_img_path": rerender_img_path,
        "comp_fig_path": comp_fig_path,
        "metrics_diff": {"mae": mae_diff, "mse": mse_diff, "ssim": ssim_diff, "iou": iou_diff},
        "metrics_cpp": {"mae": mae_cpp, "mse": mse_cpp, "ssim": ssim_cpp, "iou": iou_cpp},
    }


def main():
    output_dir = os.path.join(repo_root, "notebooks", "output_dap50_pipeline")
    save_path = os.path.join(repo_root, "diffusion_based", "checkpoints", "diffusion_3d_dap50.pt")
    
    # Phase 1: Generate Dataset (10 seeds for DAP 50 mature architecture)
    seeds = list(range(42, 52))
    samples = phase1_generate_dap50_dataset(output_dir=output_dir, seeds=seeds)
    
    # Phase 2: Model Training
    model = phase2_train_dap50_diffusion(samples=samples, save_path=save_path, num_epochs=120, lr=3e-4, max_nodes=256)
    
    # Phase 3: Inference & Evaluation
    eval_results = phase3_inference_and_evaluation(model=model, gt_sample=samples[0], output_dir=output_dir)
    
    print("\n=======================================================")
    print("DAP 50 END-TO-END PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
