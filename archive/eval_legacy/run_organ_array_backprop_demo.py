"""
Backpropagation-based single-image inverse rendering demo.

Given a Helios XML plant, renders a target image, then optimizes both
cardinality (existence embedded in the PlantOrganArray tensor) and global
organ-scale multipliers of a randomly initialized PlantOrganArray so that the
rendered image matches the target.

This is **direct backpropagation through the renderer**, not diffusion.
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
    COL_INODE_LEN,
    COL_INODE_RAD,
    COL_PET0_LEN,
    COL_PET0_RAD,
    COL_PET0_PITCH,
    COL_PET0_CURV,
    COL_PET0_LEAF_SCALE,
    COL_PET0_L0_SCALE,
    COL_PET0_L1_SCALE,
    COL_PET0_L2_SCALE,
    COL_EXISTENCE,
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


def initialize_reasonable(
    template: PlantOrganArray,
    seed: int = 42,
    leaf_scale: float = 0.55,
    stem_scale: float = 0.65,
    petiole_scale: float = 0.55,
    existence_rate: float = 0.5,
) -> PlantOrganArray:
    """
    Create a biologically reasonable sparse initialization.

    - ``existence`` (last column of the tensor) is initialized mostly active
      with a random dropout, giving the optimizer room to prune organs.
    - Continuous geometry is derived from the GT template but scaled down like a
      younger / smaller plant.
    """
    device = template.tensor.device
    rng = torch.Generator(device=device).manual_seed(seed)

    init_tensor = template.tensor.clone()

    # Random existence: continuous in [0, 1], biased toward the activation rate
    existence = torch.rand(template.num_nodes, device=device, generator=rng)
    existence = (existence < existence_rate).float()
    existence = existence * (0.7 + 0.3 * torch.rand(template.num_nodes, device=device, generator=rng))
    init_tensor[:, COL_EXISTENCE] = existence

    # Scale geometry down uniformly
    init_tensor[:, COL_INODE_LEN] *= stem_scale
    init_tensor[:, COL_INODE_RAD] *= stem_scale
    init_tensor[:, COL_PET0_LEN] *= petiole_scale
    init_tensor[:, COL_PET0_RAD] *= petiole_scale
    init_tensor[:, COL_PET0_PITCH] *= (petiole_scale * 0.5 + 0.5)
    init_tensor[:, COL_PET0_CURV] *= petiole_scale
    init_tensor[:, COL_PET0_LEAF_SCALE] *= leaf_scale
    init_tensor[:, COL_PET0_L0_SCALE] *= leaf_scale
    init_tensor[:, COL_PET0_L1_SCALE] *= leaf_scale
    init_tensor[:, COL_PET0_L2_SCALE] *= leaf_scale

    return PlantOrganArray(init_tensor, raw_metadata=template.raw_metadata)


def main():
    output_dir = os.path.join(repo_root, "diffusion_based", "eval", "output")
    os.makedirs(output_dir, exist_ok=True)
    source_xml = os.path.join(
        repo_root,
        "notebooks",
        "output_dap30_verification",
        "dap30_gt_seed42_0000_plant_0000.xml",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Running backpropagation-based inverse rendering on device: {device}")

    # 1. Target OrganArray & Image from GT Helios XML
    organ_array_gt = PlantOrganArray.from_xml_file(source_xml)
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

    # 2. Random but biologically reasonable initialization
    organ_array_init = initialize_reasonable(
        organ_array_gt,
        seed=42,
        leaf_scale=0.55,
        stem_scale=0.65,
        petiole_scale=0.55,
        existence_rate=0.5,
    )
    organ_array_init.tensor = organ_array_init.tensor.to(device)

    # Optimizable tensor: clone and make differentiable
    # We keep base geometry columns fixed by masking, and optimize selected
    # geometry columns + existence.
    opt_tensor = organ_array_init.tensor.clone().detach().requires_grad_(True)

    # Mask: 1 = optimizable, 0 = fixed at GT
    opt_mask = torch.zeros(organ_array_gt.tensor.shape[1], device=device)
    opt_channels = [
        COL_INODE_LEN,
        COL_INODE_RAD,
        COL_PET0_LEN,
        COL_PET0_RAD,
        COL_PET0_PITCH,
        COL_PET0_CURV,
        COL_PET0_LEAF_SCALE,
        COL_PET0_L0_SCALE,
        COL_PET0_L1_SCALE,
        COL_PET0_L2_SCALE,
        COL_EXISTENCE,
    ]
    for c in opt_channels:
        opt_mask[c] = 1.0

    # Global scale multipliers (constrained via sigmoid * 1.5)
    leaf_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    stem_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    petiole_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    base_tensor = organ_array_gt.tensor.clone().detach()
    base_metadata = organ_array_gt.raw_metadata

    optimizer = optim.Adam([opt_tensor, leaf_logit, stem_logit, petiole_logit], lr=0.03)

    history_images = []
    history_losses = []
    history_ssim = []
    history_existence_sum = []

    num_steps = 60
    print("\n=======================================================")
    print(f"STARTING BACKPROPAGATION-BASED INVERSE RENDERING ({num_steps} Steps)")
    print("=======================================================")

    def get_scales():
        return (
            torch.sigmoid(leaf_logit) * 1.5,
            torch.sigmoid(stem_logit) * 1.5,
            torch.sigmoid(petiole_logit) * 1.5,
        )

    def build_opt_array():
        leaf_scale, stem_scale, petiole_scale = get_scales()

        # Apply global scale multipliers to the base template (not to the
        # per-node opt_tensor directly), then overwrite selected columns with
        # the per-node optimized values.
        scaled_base = base_tensor.clone()
        scaled_base[:, COL_INODE_LEN] *= stem_scale
        scaled_base[:, COL_INODE_RAD] *= stem_scale
        scaled_base[:, COL_PET0_LEN] *= petiole_scale
        scaled_base[:, COL_PET0_RAD] *= petiole_scale
        scaled_base[:, COL_PET0_PITCH] *= (petiole_scale * 0.5 + 0.5)
        scaled_base[:, COL_PET0_CURV] *= petiole_scale
        scaled_base[:, COL_PET0_LEAF_SCALE] *= leaf_scale
        scaled_base[:, COL_PET0_L0_SCALE] *= leaf_scale
        scaled_base[:, COL_PET0_L1_SCALE] *= leaf_scale
        scaled_base[:, COL_PET0_L2_SCALE] *= leaf_scale

        # Replace optimizable columns with opt_tensor values
        final_tensor = scaled_base * (1.0 - opt_mask) + opt_tensor * opt_mask
        # Clamp existence to [0, 1] via sigmoid (last column)
        final_tensor[:, COL_EXISTENCE] = torch.sigmoid(final_tensor[:, COL_EXISTENCE])

        return PlantOrganArray(final_tensor, raw_metadata=base_metadata)

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

        # Image reconstruction loss
        loss_rgb = F.l1_loss(rendered_rgb, target_rgb) + F.mse_loss(rendered_rgb, target_rgb)
        # Mild sparsity prior on existence (encourage parsimonious plant)
        existence = organ_array_opt.existence
        sparsity = 0.005 * existence.mean()
        # Parameter regularizer towards 1.0 to avoid collapse
        leaf_scale, stem_scale, petiole_scale = get_scales()
        scale_reg = 0.002 * (
            F.mse_loss(leaf_scale, torch.tensor(1.0, device=device)) +
            F.mse_loss(stem_scale, torch.tensor(1.0, device=device)) +
            F.mse_loss(petiole_scale, torch.tensor(1.0, device=device))
        )
        total_loss = loss_rgb + sparsity + scale_reg

        if step < num_steps:
            total_loss.backward()
            optimizer.step()

        cur_rgb_np = rendered_rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        ssim_val = compute_ssim_numpy(cur_rgb_np, target_rgb_np)
        existence_sum = existence.sum().item()

        history_losses.append(total_loss.item())
        history_ssim.append(ssim_val)
        history_existence_sum.append(existence_sum)

        if step in [0, 15, 30, 45, 60]:
            history_images.append((step, cur_rgb_np, total_loss.item(), ssim_val))
            print(f"Step {step:02d}/{num_steps:02d} | Loss: {total_loss.item():.6f} | SSIM: {ssim_val:.4f} | "
                  f"existence sum: {existence_sum:.1f} | "
                  f"leaf={leaf_scale.item():.3f} stem={stem_scale.item():.3f} petiole={petiole_scale.item():.3f}")

    print(f"Optimization finished in {time.time() - t0:.2f}s!")

    # 3. Save visualization progression figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="black")
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")

    # Target reference image
    axes[0, 0].imshow(target_rgb_np)
    axes[0, 0].set_title("Target Helios GT Image\n(Seed=42)", color="white", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Progression steps
    for idx, (step_num, img, loss_v, ssim_v) in enumerate(history_images):
        if idx < 3:
            ax = axes[0, idx + 1]
        else:
            ax = axes[1, 0]
        ax.imshow(img)
        ax.set_title(f"Step {step_num:02d}\nLoss={loss_v:.4f} | SSIM={ssim_v:.4f}", color="cyan", fontsize=12, fontweight="bold")
        ax.axis("off")

    # Loss curve
    axes[1, 1].plot(history_losses, color="crimson", linewidth=2.5)
    axes[1, 1].set_title("Loss Convergence Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Optimization Step", color="white")
    axes[1, 1].set_ylabel("Loss", color="crimson")
    axes[1, 1].tick_params(colors="white")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    # SSIM curve
    axes[1, 2].plot(history_ssim, color="springgreen", linewidth=2.5)
    axes[1, 2].set_title("SSIM Progression Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Optimization Step", color="white")
    axes[1, 2].set_ylabel("SSIM", color="springgreen")
    axes[1, 2].tick_params(colors="white")
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)

    # Final Pixel Diff Map
    final_diff = np.abs(history_images[-1][1] - target_rgb_np)
    im = axes[1, 3].imshow(final_diff.mean(axis=-1), cmap="inferno", vmin=0.0, vmax=0.2)
    axes[1, 3].set_title(f"Final Diff Map\nMAE={np.mean(final_diff):.5f}", color="gold", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    opt_fig_path = os.path.join(output_dir, "organ_array_backprop_demo.png")
    plt.savefig(opt_fig_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved backpropagation demo figure to: {opt_fig_path}")

    # Save metrics JSON
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
    with open(os.path.join(output_dir, "organ_array_backprop_demo_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
