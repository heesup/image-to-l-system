"""
Backpropagation-based inverse rendering demo on a simple DAP 10 cowpea plant.

Optimizes global organ-scale multipliers and per-node existence so the
rendered image matches the Helios GT.  DAP 10 is a small plant, making this an
easy-to-converge demonstration of backpropagation through the renderer.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    T_COL_LENGTH,
    T_COL_RADIUS,
    T_COL_SCALE,
    T_COL_PITCH,
    T_COL_CURVATURE,
    T_COL_CURRENT_LEAF_SCALE_FACTOR,
    T_COL_ORGAN_TYPE,
    T_COL_EXISTENCE,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


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


def main():
    output_dir = os.path.join(repo_root, "diffusion_based", "eval", "output")
    os.makedirs(output_dir, exist_ok=True)
    source_xml = os.path.join(
        repo_root,
        "Digital-Crops",
        "projects",
        "syntheticdata_generation",
        "build",
        "output",
        "dap10_gt_0000_plant_0000.xml",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Running DAP 10 backpropagation-based inverse rendering on device: {device}")

    # 1. Target OrganArray & Image from GT Helios XML (typed 40D layout)
    organ_array_gt = PlantOrganArray.from_xml_file_typed(source_xml)
    organ_array_gt.tensor = organ_array_gt.tensor.to(device)

    renderer = HeliosPyTorchRenderer(image_size=128)
    with torch.no_grad():
        target_rgb = renderer.render_organ_array(
            organ_array_gt,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=1.0,
            background="black",
            device=device,
            differentiable=True,
            focus_plant=True,
        )  # (3, H, W)
    target_rgb_np = target_rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
    target_mask = (target_rgb.sum(0) > 0.05).float().detach()

    # 2. Small-plant initialization: all organs present but scaled down
    base_tensor = organ_array_gt.tensor.clone().detach()
    organ_type = base_tensor[:, T_COL_ORGAN_TYPE].long()
    is_internode = (organ_type == ORGAN_INTERNODE)
    is_petiole = (organ_type == ORGAN_PETIOLE)
    is_leaf = (organ_type == ORGAN_LEAF)

    init_tensor = base_tensor.clone()
    init_tensor[is_internode, T_COL_LENGTH] *= 0.45
    init_tensor[is_internode, T_COL_RADIUS] *= 0.45
    init_tensor[is_petiole, T_COL_LENGTH] *= 0.40
    init_tensor[is_petiole, T_COL_RADIUS] *= 0.40
    init_tensor[is_petiole, T_COL_PITCH] *= 0.80
    init_tensor[is_petiole, T_COL_CURVATURE] *= 0.40
    init_tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= 0.40
    init_tensor[is_leaf, T_COL_SCALE] *= 0.40
    # Existence: all present, with slight uncertainty
    init_tensor[:, T_COL_EXISTENCE] = 1.0

    # Optimizable parameters
    leaf_logit = torch.tensor(np.log(0.45 / (1.5 - 0.45)), device=device, requires_grad=True, dtype=torch.float32)
    stem_logit = torch.tensor(np.log(0.45 / (1.5 - 0.45)), device=device, requires_grad=True, dtype=torch.float32)
    petiole_logit = torch.tensor(np.log(0.40 / (1.5 - 0.40)), device=device, requires_grad=True, dtype=torch.float32)
    opt_existence = init_tensor[:, T_COL_EXISTENCE].clone().detach().requires_grad_(True)

    optimizer = optim.Adam([leaf_logit, stem_logit, petiole_logit, opt_existence], lr=0.03)

    history_images = []
    history_losses = []
    history_ssim = []
    history_existence_sum = []

    num_steps = 60
    print("\n=======================================================")
    print(f"STARTING DAP 10 BACKPROPAGATION-BASED INVERSE RENDERING ({num_steps} Steps)")
    print("=======================================================")

    def get_scales():
        return (
            torch.sigmoid(leaf_logit) * 1.5,
            torch.sigmoid(stem_logit) * 1.5,
            torch.sigmoid(petiole_logit) * 1.5,
        )

    def build_opt_array():
        leaf_scale, stem_scale, petiole_scale = get_scales()

        opt_tensor = base_tensor.clone()
        opt_tensor[is_internode, T_COL_LENGTH] *= stem_scale
        opt_tensor[is_internode, T_COL_RADIUS] *= stem_scale
        opt_tensor[is_petiole, T_COL_LENGTH] *= petiole_scale
        opt_tensor[is_petiole, T_COL_RADIUS] *= petiole_scale
        opt_tensor[is_petiole, T_COL_PITCH] *= (petiole_scale * 0.5 + 0.5)
        opt_tensor[is_petiole, T_COL_CURVATURE] *= petiole_scale
        opt_tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= leaf_scale
        opt_tensor[is_leaf, T_COL_SCALE] *= leaf_scale
        opt_tensor[:, T_COL_EXISTENCE] = torch.sigmoid(opt_existence)

        return PlantOrganArray(opt_tensor, raw_metadata=[])

    t0 = time.time()
    for step in range(num_steps + 1):
        optimizer.zero_grad()
        organ_array_opt = build_opt_array()

        rendered_rgb = renderer.render_organ_array(
            organ_array_opt,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=1.0,
            background="black",
            device=device,
            differentiable=True,
            focus_plant=True,
        )  # (3, H, W)

        # Masked RGB MSE + silhouette BCE
        rendered_mask = (rendered_rgb.sum(0) > 0.05).float()
        loss_rgb = F.mse_loss(rendered_rgb * target_mask.unsqueeze(0), target_rgb * target_mask.unsqueeze(0))
        loss_sil = F.binary_cross_entropy(rendered_mask, target_mask)
        # Pull existence toward 1.0 so we recover all target organs
        existence_pull = 0.1 * F.mse_loss(organ_array_opt.existence, torch.ones_like(organ_array_opt.existence))
        total_loss = loss_rgb + 2.0 * loss_sil + existence_pull

        if step < num_steps:
            total_loss.backward()
            optimizer.step()

        cur_rgb_np = rendered_rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        ssim_val = compute_ssim_numpy(cur_rgb_np, target_rgb_np)
        existence_sum = organ_array_opt.existence.sum().item()

        history_losses.append(total_loss.item())
        history_ssim.append(ssim_val)
        history_existence_sum.append(existence_sum)

        if step in [0, 15, 30, 45, 60]:
            history_images.append((step, cur_rgb_np, total_loss.item(), ssim_val))
            leaf_scale, stem_scale, petiole_scale = get_scales()
            print(f"Step {step:02d}/{num_steps:02d} | Loss: {total_loss.item():.6f} | SSIM: {ssim_val:.4f} | "
                  f"existence sum: {existence_sum:.1f} | "
                  f"leaf={leaf_scale.item():.3f} stem={stem_scale.item():.3f} petiole={petiole_scale.item():.3f}")

    print(f"Optimization finished in {time.time() - t0:.2f}s!")

    # Save figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="black")
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")

    axes[0, 0].imshow(target_rgb_np)
    axes[0, 0].set_title("Target Helios GT Image\n(DAP 10)", color="white", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    for idx, (step_num, img, loss_v, ssim_v) in enumerate(history_images):
        if idx < 3:
            ax = axes[0, idx + 1]
        else:
            ax = axes[1, 0]
        ax.imshow(img)
        ax.set_title(f"Step {step_num:02d}\nLoss={loss_v:.4f} | SSIM={ssim_v:.4f}", color="cyan", fontsize=12, fontweight="bold")
        ax.axis("off")

    axes[1, 1].plot(history_losses, color="crimson", linewidth=2.5)
    axes[1, 1].set_title("Loss Convergence Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Optimization Step", color="white")
    axes[1, 1].set_ylabel("Loss", color="crimson")
    axes[1, 1].tick_params(colors="white")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    axes[1, 2].plot(history_ssim, color="springgreen", linewidth=2.5)
    axes[1, 2].set_title("SSIM Progression Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Optimization Step", color="white")
    axes[1, 2].set_ylabel("SSIM", color="springgreen")
    axes[1, 2].tick_params(colors="white")
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)

    final_diff = np.abs(history_images[-1][1] - target_rgb_np)
    im = axes[1, 3].imshow(final_diff.mean(axis=-1), cmap="inferno", vmin=0.0, vmax=0.2)
    axes[1, 3].set_title(f"Final Diff Map\nMAE={np.mean(final_diff):.5f}", color="gold", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    opt_fig_path = os.path.join(output_dir, "dap10_organ_array_backprop_demo.png")
    plt.savefig(opt_fig_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved DAP 10 backpropagation demo figure to: {opt_fig_path}")

    metrics = {
        "initial_loss": history_losses[0],
        "final_loss": history_losses[-1],
        "initial_ssim": history_ssim[0],
        "final_ssim": history_ssim[-1],
        "initial_existence_sum": history_existence_sum[0],
        "final_existence_sum": history_existence_sum[-1],
        "num_steps": num_steps,
        "device": str(device),
    }
    with open(os.path.join(output_dir, "dap10_organ_array_backprop_demo_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
