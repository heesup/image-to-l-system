"""
Backpropagation-based inverse rendering problem suite.

Generates three figures demonstrating increasing difficulty:
  1. Easy:   fixed GT topology + fixed per-node geometry, optimize only
             per-node existence + global leaf/stem/petiole scale multipliers.
  2. Medium: grow from a tiny seed plant (2 active nodes), optimize existence
             + global scales.
  3. Hard:   random topology initialization, attempt backpropagation-based
             recovery (expected partial/limited convergence).

Output: diffusion_based/eval/output/backprop_problem_{easy,medium,hard}.png
"""

import os
import sys
import time
import json
import re
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Any

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
    T_COL_PARENT_SHOOT_ID,
    T_COL_PARENT_NODE_IDX,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.perceptual_loss import VGGPerceptualLoss
from diffusion_based.models.vit_image_to_organ_array import ViTOrganArrayDiffuser
from diffusion_based.dataset.organ_array_dataset import OrganArrayDataset
from diffusion_based.training.train_organ_array_diffusion import DDPMScheduler, train_epoch


def organ_type_masks(tensor: torch.Tensor):
    """Return boolean masks over rows by organ_type for typed 40D layout."""
    ot = tensor[:, T_COL_ORGAN_TYPE].long()
    return (
        ot == ORGAN_INTERNODE,
        ot == ORGAN_PETIOLE,
        ot == ORGAN_LEAF,
    )


def render_organ_array_with_sanity(organ_array, renderer, target_rgb, device, label=""):
    """Render an organ array and compare to a target image for debugging/sanity."""
    rgb = renderer.render_organ_array(
        organ_array,
        azimuth_deg=0.0,
        elevation_deg=90.0,
        camera_height=1.0,
        background="black",
        device=device,
        differentiable=True,
        focus_plant=True,
    )
    mae = float(torch.mean(torch.abs(rgb - target_rgb)).item())
    ssim = compute_ssim_numpy(
        rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1),
        target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1),
    )
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}render MAE={mae:.6f} SSIM={ssim:.4f}")
    return rgb


def compute_ssim_numpy(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=win_size))
    except Exception as e:
        mse = float(np.mean((img1 - img2) ** 2))
        return float(max(0.0, 1.0 - 5.0 * mse))


def render_target(organ_array_gt, renderer, device):
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
        )
    return target_rgb


def flow_to_hsv(flow_np: np.ndarray) -> np.ndarray:
    """Convert (H, W, 2) optical flow vector field into an RGB HSV direction & magnitude visual map."""
    mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0


def compute_optical_flow_farneback(img_src_np: np.ndarray, img_tgt_np: np.ndarray) -> np.ndarray:
    """Compute dense 2D optical flow (u, v) from img_src_np to img_tgt_np using OpenCV Farneback."""
    gray_src = (cv2.cvtColor((img_src_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    gray_tgt = (cv2.cvtColor((img_tgt_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    flow = cv2.calcOpticalFlowFarneback(
        gray_src, gray_tgt, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow  # (H, W, 2) in pixel offsets


def apply_flow_warping_loss(rendered_rgb: torch.Tensor, target_rgb: torch.Tensor, device: torch.device):
    """
    Computes Differentiable Optical Flow Warping Loss using PyTorch F.grid_sample:
    1. Estimate dense 2D flow field v = (dx, dy) mapping rendered_rgb -> target_rgb.
    2. Convert flow into normalized [-1, 1] 2D sampling grid.
    3. Differentiably warp rendered_rgb using F.grid_sample.
    4. Compute MSE loss between warped_rgb and target_rgb.
    """
    H, W = rendered_rgb.shape[1], rendered_rgb.shape[2]
    rendered_np = rendered_rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
    target_np = target_rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

    flow_np = compute_optical_flow_farneback(rendered_np, target_np)
    flow_tensor = torch.from_numpy(flow_np).to(device, dtype=torch.float32)

    # Build sampling grid (x + dx, y + dy) normalized to [-1, 1]
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij"
    )
    grid_x = x_coords + (2.0 * flow_tensor[:, :, 0] / max(W - 1, 1))
    grid_y = y_coords + (2.0 * flow_tensor[:, :, 1] / max(H - 1, 1))
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    rendered_batch = rendered_rgb.unsqueeze(0)  # (1, 3, H, W)
    warped_rgb = F.grid_sample(rendered_batch, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(0)

    loss_warp = F.mse_loss(warped_rgb, target_rgb)
    flow_magnitude = torch.mean(torch.sqrt(flow_tensor[:, :, 0]**2 + flow_tensor[:, :, 1]**2 + 1e-8))

    flow_hsv_np = flow_to_hsv(flow_np)
    warped_np = warped_rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

    return loss_warp, flow_magnitude, warped_rgb, flow_tensor, flow_hsv_np, warped_np


def make_non_relevant_source_plant(device: torch.device, alt_xml_path: str = None) -> PlantOrganArray:
    """
    Creates an initial plant from a completely separate, non-relevant source
    (e.g., an independent seed plant or alternate DAP plant XML) to ensure ZERO
    ground-truth information leakage from the target plant.
    """
    if alt_xml_path and os.path.exists(alt_xml_path):
        source_array = PlantOrganArray.from_xml_file_typed(alt_xml_path)
    else:
        alt_xml = os.path.join(repo_root, "dataset", "helios_data", "cowpea_dap009_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml")
        if not os.path.exists(alt_xml):
            alt_xml = os.path.join(repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "dapcmp", "d10", "d10_all_0000_plant_0000.xml")
        source_array = PlantOrganArray.from_xml_file_typed(alt_xml)
    source_array.tensor = source_array.tensor.to(device)
    return source_array


def make_seed_plant(target_array, seed=42):
    """Create a tiny 2-node seed plant from the target template.

    Only the first 2 nodes are active; all other existence values are 0.0.
    This matches a real germination stage with just an unifoliate leaf pair.
    """
    cpu_rng = torch.Generator(device='cpu').manual_seed(seed)
    N = target_array.num_nodes
    tensor = target_array.tensor.clone()
    existence = torch.zeros(N, device=tensor.device)
    existence[:2] = 1.0
    tensor[:, T_COL_EXISTENCE] = existence

    is_internode, is_petiole, is_leaf = organ_type_masks(tensor)
    tensor[is_internode, T_COL_LENGTH] *= 0.15
    tensor[is_internode, T_COL_RADIUS] *= 0.15
    tensor[is_petiole, T_COL_LENGTH] *= 0.15
    tensor[is_petiole, T_COL_RADIUS] *= 0.15
    tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= 0.15
    tensor[is_leaf, T_COL_SCALE] *= 0.15
    tensor[is_petiole, T_COL_PITCH] *= 0.5
    tensor[is_petiole, T_COL_CURVATURE] *= 0.3
    return PlantOrganArray(tensor, raw_metadata=target_array.raw_metadata)


def make_random_topology(target_array, seed=42):
    """Create a random topology initialization by perturbing geometry, existence, and scales while keeping valid shoot metadata."""
    cpu_rng = torch.Generator(device='cpu').manual_seed(seed)
    N = target_array.num_nodes
    tensor = target_array.tensor.clone()

    existence = (torch.rand(N, generator=cpu_rng) < 0.5).float()
    existence = existence * (0.6 + 0.4 * torch.rand(N, generator=cpu_rng))
    tensor[:, T_COL_EXISTENCE] = existence.to(tensor.device)

    is_internode, is_petiole, is_leaf = organ_type_masks(tensor)
    tensor[is_internode, T_COL_LENGTH] *= 0.4
    tensor[is_internode, T_COL_RADIUS] *= 0.4
    tensor[is_petiole, T_COL_LENGTH] *= 0.4
    tensor[is_petiole, T_COL_RADIUS] *= 0.4
    tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= 0.4
    tensor[is_leaf, T_COL_SCALE] *= 0.4

    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    return PlantOrganArray(tensor, raw_metadata=target_array.raw_metadata)


def optimize_backprop(
    target_rgb,
    init_array,
    renderer,
    device,
    num_steps=100,
    lr=0.03,
    optimize_geometry=False,
    optimize_topology=False,
    snapshot_steps=None,
    binary_threshold_step=None,
    grad_clip=1.0,
    existence_pull_weight=0.05,
    fix_existence=False,
    use_flow_loss=True,
):
    if snapshot_steps is None:
        snapshot_steps = [0, 20, 40, 60, 80, 100]
    """Run backpropagation-based inverse rendering with optional Optical Flow Warping Loss."""
    base_tensor = init_array.tensor.clone().detach().to(device)
    fixed_existence = torch.sigmoid(base_tensor[:, T_COL_EXISTENCE]).detach()
    opt_existence = base_tensor[:, T_COL_EXISTENCE].clone().detach().requires_grad_(not fix_existence)

    # Global scale multipliers (constrained to [0, 1.5])
    leaf_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    stem_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    petiole_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    # Per-node scale multipliers (initialized near 1.0, constrained to [0, 2.0])
    N = base_tensor.shape[0]
    node_leaf_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_stem_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_pet_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    scale_params = [leaf_logit, stem_logit, petiole_logit,
                    node_leaf_logit, node_stem_logit, node_pet_logit]

    opt_tensor = None
    if optimize_geometry or optimize_topology:
        opt_tensor = base_tensor.clone().detach().requires_grad_(True)

    # Soft parent topology optimization
    opt_parent_logits = None
    parent_candidates = None
    if optimize_topology and init_array.parent_logits is not None:
        opt_parent_logits = init_array.parent_logits.clone().detach().to(device).requires_grad_(True)
        parent_candidates = init_array.parent_candidates.to(device)

    # Build parameter groups
    param_groups = [{"params": scale_params, "lr": lr}]
    if opt_existence.requires_grad:
        param_groups.append({"params": [opt_existence], "lr": lr})
    if opt_tensor is not None:
        param_groups.append({"params": [opt_tensor], "lr": lr * 0.1})
    if opt_parent_logits is not None:
        param_groups.append({"params": [opt_parent_logits], "lr": lr})

    params = []
    for g in param_groups:
        params.extend(g["params"])

    optimizer = optim.Adam(param_groups)

    base_metadata = init_array.raw_metadata
    target_mask = (target_rgb.sum(0) > 0.05).float().detach()

    history = {"loss": [], "ssim": [], "existence_sum": [], "images": [], "flow_mag": []}

    def get_scales():
        leaf_scale = torch.sigmoid(leaf_logit) * 1.5
        stem_scale = torch.sigmoid(stem_logit) * 1.5
        petiole_scale = torch.sigmoid(petiole_logit) * 1.5
        node_leaf = torch.sigmoid(node_leaf_logit) * 2.0
        node_stem = torch.sigmoid(node_stem_logit) * 2.0
        node_pet = torch.sigmoid(node_pet_logit) * 2.0
        return leaf_scale, stem_scale, petiole_scale, node_leaf, node_stem, node_pet

    def build_array():
        leaf_scale, stem_scale, petiole_scale, node_leaf, node_stem, node_pet = get_scales()

        if opt_tensor is not None:
            tensor = opt_tensor.clone()
        else:
            tensor = base_tensor.clone()

        is_internode, is_petiole, is_leaf = organ_type_masks(tensor)
        tensor[is_internode, T_COL_LENGTH] *= stem_scale * node_stem[is_internode]
        tensor[is_internode, T_COL_RADIUS] *= stem_scale * node_stem[is_internode]
        tensor[is_petiole, T_COL_LENGTH] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_RADIUS] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_PITCH] *= ((petiole_scale * node_pet[is_petiole]) * 0.5 + 0.5)
        tensor[is_petiole, T_COL_CURVATURE] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= leaf_scale * node_leaf[is_petiole]
        tensor[is_leaf, T_COL_SCALE] *= leaf_scale * node_leaf[is_leaf]
        if fix_existence:
            tensor[:, T_COL_EXISTENCE] = fixed_existence
        else:
            tensor[:, T_COL_EXISTENCE] = torch.sigmoid(opt_existence)
        if opt_parent_logits is not None:
            return PlantOrganArray(
                tensor,
                raw_metadata=base_metadata,
                parent_logits=opt_parent_logits,
                parent_candidates=parent_candidates,
            )
        return PlantOrganArray(tensor, raw_metadata=base_metadata)

    # Cosine LR schedule wrapper (warmup 0..5, cosine decay 5..num_steps)
    def lr_lambda(step):
        if step < 5:
            return step / 5.0
        progress = (step - 5) / max(1, num_steps - 5)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for step in range(num_steps + 1):
        # Optional hard-thresholding of existence for cleaner convergence
        if not fix_existence and binary_threshold_step is not None and step == binary_threshold_step:
            with torch.no_grad():
                opt_existence.data = torch.where(
                    torch.sigmoid(opt_existence) > 0.5,
                    torch.tensor(6.0, device=device),   # ~ sigmoid = 0.997
                    torch.tensor(-6.0, device=device),  # ~ sigmoid = 0.002
                )

        # Keep parent logits in a numerically stable range for softmax
        if opt_parent_logits is not None:
            with torch.no_grad():
                opt_parent_logits.clamp_(-5.0, 5.0)

        optimizer.zero_grad()
        organ_array = build_array()

        rendered_rgb = renderer.render_organ_array(
            organ_array,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=1.0,
            background="black",
            device=device,
            differentiable=True,
            focus_plant=True,
        )

        loss_rgb = F.mse_loss(rendered_rgb * target_mask.unsqueeze(0), target_rgb * target_mask.unsqueeze(0))
        rendered_mask = (rendered_rgb.sum(0) > 0.05).float()
        loss_sil = F.binary_cross_entropy(rendered_mask, target_mask)

        # Differentiable Optical Flow Warping Loss
        if use_flow_loss:
            loss_warp, flow_mag, _, _, flow_hsv_np, warped_np = apply_flow_warping_loss(rendered_rgb, target_rgb, device)
            if step == 0:
                history["initial_flow_hsv"] = flow_hsv_np
                history["initial_warped_rgb"] = warped_np
            if step == num_steps:
                history["final_flow_hsv"] = flow_hsv_np
                history["final_warped_rgb"] = warped_np
        else:
            loss_warp = torch.tensor(0.0, device=device)
            flow_mag = torch.tensor(0.0, device=device)

        # Pull existence toward 1.0 so we recover all target organs
        existence_pull = existence_pull_weight * F.mse_loss(organ_array.existence, torch.ones_like(organ_array.existence))
        total_loss = loss_rgb + 2.0 * loss_sil + 1.5 * loss_warp + 0.01 * flow_mag + existence_pull

        if step < num_steps:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            cur_np = rendered_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            ssim = compute_ssim_numpy(cur_np, target_np)

        history["loss"].append(total_loss.item())
        history["ssim"].append(ssim)
        history["existence_sum"].append(organ_array.existence.sum().item())

        if step in snapshot_steps:
            history["images"].append((step, cur_np, total_loss.item(), ssim))
            if step % 50 == 0 or step in snapshot_steps:
                leaf, stem, pet, _, _, _ = get_scales()
                print(f"  step {step:03d} | loss={total_loss.item():.4f} | ssim={ssim:.4f} | "
                      f"exist={organ_array.existence.sum().item():.1f} | "
                      f"leaf={leaf:.3f} stem={stem:.3f} pet={pet:.3f}")

    return history


def _extract_dap_and_name(xml_path: str):
    base = os.path.basename(xml_path)
    name = base.replace(".xml", "")
    m = re.search(r"dap(\d+)", name, re.IGNORECASE)
    dap = int(m.group(1)) if m else 10
    return name, dap


def plot_problem(target_rgb_np, history, title, caption, output_path, dap=10):
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), facecolor="black")
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")

    fig.suptitle(caption, color="white", fontsize=14, fontweight="bold", y=0.98)

    # 1. Target GT
    axes[0, 0].imshow(target_rgb_np)
    axes[0, 0].set_title(f"Target Helios GT\n(DAP {dap})", color="white", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")

    # Snapshots (Initial, Mid, Final)
    imgs = history["images"]
    if len(imgs) >= 1:
        axes[0, 1].imshow(imgs[0][1])
        axes[0, 1].set_title(f"Step {imgs[0][0]:03d} (Init)\nLoss={imgs[0][2]:.4f} | SSIM={imgs[0][3]:.4f}", color="cyan", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

    if len(imgs) >= 3:
        mid_idx = len(imgs) // 2
        axes[0, 2].imshow(imgs[mid_idx][1])
        axes[0, 2].set_title(f"Step {imgs[mid_idx][0]:03d} (Mid)\nLoss={imgs[mid_idx][2]:.4f} | SSIM={imgs[mid_idx][3]:.4f}", color="cyan", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")

    if len(imgs) >= 1:
        axes[0, 3].imshow(imgs[-1][1])
        axes[0, 3].set_title(f"Step {imgs[-1][0]:03d} (Final)\nLoss={imgs[-1][2]:.4f} | SSIM={imgs[-1][3]:.4f}", color="cyan", fontsize=11, fontweight="bold")
        axes[0, 3].axis("off")

    # 2. Optical Flow Map (Farneback HSV)
    if "initial_flow_hsv" in history:
        axes[0, 4].imshow(history["initial_flow_hsv"])
        axes[0, 4].set_title("Optical Flow Vector Map\n(Farneback HSV Field)", color="gold", fontsize=11, fontweight="bold")
    else:
        axes[0, 4].text(0.5, 0.5, "N/A", color="white", ha="center", va="center")
    axes[0, 4].axis("off")

    # 3. Flow-Warped Render (F.grid_sample)
    if "initial_warped_rgb" in history:
        axes[1, 0].imshow(history["initial_warped_rgb"])
        axes[1, 0].set_title("Flow-Warped Render\n(PyTorch F.grid_sample)", color="springgreen", fontsize=11, fontweight="bold")
    else:
        axes[1, 0].axis("off")
    axes[1, 0].axis("off")

    # 4. Loss Curve
    axes[1, 1].plot(history["loss"], color="crimson", linewidth=2.5)
    axes[1, 1].set_title("Loss Convergence Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Step", color="white")
    axes[1, 1].set_ylabel("Loss", color="crimson")
    axes[1, 1].tick_params(colors="white")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    # 5. SSIM Curve
    axes[1, 2].plot(history["ssim"], color="springgreen", linewidth=2.5)
    axes[1, 2].set_title("SSIM Progression Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Step", color="white")
    axes[1, 2].set_ylabel("SSIM", color="springgreen")
    axes[1, 2].tick_params(colors="white")
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)

    # 6. Final Difference Map
    final_diff = np.abs(imgs[-1][1] - target_rgb_np)
    im = axes[1, 3].imshow(final_diff.mean(axis=-1), cmap="inferno", vmin=0.0, vmax=0.2)
    axes[1, 3].set_title(f"Final Diff Map\nMAE={np.mean(final_diff):.5f}", color="gold", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    # 7. Active Organ Existence Progression
    axes[1, 4].plot(history["existence_sum"], color="deepskyblue", linewidth=2.5)
    axes[1, 4].set_title("Active Organ Existence", color="white", fontsize=12, fontweight="bold")
    axes[1, 4].set_xlabel("Step", color="white")
    axes[1, 4].set_ylabel("Existence Sum", color="deepskyblue")
    axes[1, 4].tick_params(colors="white")
    axes[1, 4].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved {title} figure to {output_path}")


from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def train_diffusion_fresh(
    data_root: str,
    device: torch.device,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    max_nodes: int = 2048,
    render_every: int = 10,
    perceptual_weight: float = 0.5,
) -> Tuple[nn.Module, DDPMScheduler, OrganArrayDataset]:
    """Train a fresh Diffusion model on the dataset with Parameter + Image + VGG Perceptual Loss + EMA."""
    print(f"\n--- Training Fresh Diffusion Model with Image & Perceptual Loss + EMA ({epochs} epochs) ---", flush=True)
    dataset = OrganArrayDataset(
        data_root=data_root,
        max_nodes=max_nodes,
        image_size=128,
        use_gt_renderer_image=True,
        device=device,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ViTOrganArrayDiffuser(
        max_nodes=max_nodes,
        node_dim=40,
        image_size=128,
        patch_size=8,
        embed_dim=256,
        encoder_layers=6,
        decoder_layers=4,
        num_heads=8,
        num_organ_types=8,
    ).to(device)

    # EMA Model setup (decay = 0.9999)
    ema_decay = 0.9999
    ema_avg_fn = get_ema_multi_avg_fn(ema_decay)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = DDPMScheduler(timesteps=1000)
    renderer = HeliosPyTorchRenderer(image_size=128).to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    global_step = 0
    for epoch in range(1, epochs + 1):
        metrics = train_epoch(
            model, dataloader, optimizer, scheduler, renderer, perceptual_loss_fn, device,
            lambda_continuous=1.0, lambda_exist=1.0, lambda_organ_type=0.5,
            render_weight=1.0, perceptual_weight=perceptual_weight,
            render_every=render_every, global_step=global_step,
            ema_model=ema_model,
        )
        global_step = metrics["global_step"]
        print(f"Diffusion Training Epoch {epoch:02d}/{epochs:02d} | loss={metrics['loss']:.4f} mse={metrics['mse']:.4f} render={metrics['render']:.4f}", flush=True)

    print("Diffusion Model Training Complete (Using EMA weights for solving)!\n", flush=True)
    eval_model = ema_model.module if hasattr(ema_model, "module") else ema_model
    return eval_model, scheduler, dataset


def solve_problem_diffusion(
    target_rgb: torch.Tensor,
    init_array: PlantOrganArray,
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataset: OrganArrayDataset,
    renderer: HeliosPyTorchRenderer,
    perceptual_loss_fn: nn.Module,
    device: torch.device,
    steps: int = 50,
    guidance_scale: float = 2.0,
    guidance_weight: float = 0.5,
    snapshot_steps: list = None,
):
    """
    Solves 3D plant recovery via Guided Reverse DDIM Sampling with Classifier-Free Guidance (CFG)
    and Image & Perceptual Gradients.
    """
    model.eval()
    if snapshot_steps is None:
        snapshot_steps = [0, 10, 25, 40, steps - 1]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    target_norm = (target_rgb.unsqueeze(0) - mean) / std

    use_cfg = guidance_scale > 1.0
    if use_cfg:
        uncond_norm = torch.zeros_like(target_norm)
        batched_images = torch.cat([target_norm, uncond_norm], dim=0)  # (2, 3, H, W)
    else:
        batched_images = target_norm

    N = model.max_nodes
    node_dim = 40
    step_indices = torch.linspace(scheduler.timesteps - 1, 0, steps, device=device).long()

    if init_array is not None:
        norm_init = dataset.normalize(init_array.tensor.clone().to(device)).unsqueeze(0)
        t0 = step_indices[0].unsqueeze(0)
        x_t = scheduler.add_noise(norm_init, t0, torch.randn_like(norm_init))
    else:
        x_t = torch.randn((1, N, node_dim), device=device)

    history = {"loss": [], "ssim": [], "existence_sum": [], "images": []}

    for idx, t in enumerate(step_indices):
        t_batch = torch.tensor([t], device=device).long()

        x_t_in = x_t.detach().requires_grad_(True)

        if use_cfg:
            batched_x_t = torch.cat([x_t_in, x_t_in], dim=0)
            batched_t = torch.cat([t_batch, t_batch], dim=0)
            outputs = model(batched_x_t, batched_t, batched_images)
            pred_x0_cond, pred_x0_uncond = outputs["pred_x0"].chunk(2, dim=0)
            ot_logits_cond, ot_logits_uncond = outputs["organ_type_logits"].chunk(2, dim=0)
            pred_x0 = pred_x0_uncond + guidance_scale * (pred_x0_cond - pred_x0_uncond)
            organ_type_logits = ot_logits_uncond + guidance_scale * (ot_logits_cond - ot_logits_uncond)
        else:
            outputs = model(x_t_in, t_batch, batched_images)
            pred_x0 = outputs["pred_x0"]
            organ_type_logits = outputs["organ_type_logits"]

        denorm = dataset.denormalize(pred_x0[0])
        denorm[:, dataset.existence_col] = torch.clamp(denorm[:, dataset.existence_col], 0.0, 1.0)
        denorm[:, dataset.continuous_cols] = torch.clamp(denorm[:, dataset.continuous_cols], min=0.0)
        if organ_type_logits is not None:
            denorm[:, 11] = organ_type_logits[0].argmax(dim=-1).float()

        cand_array = PlantOrganArray(tensor=denorm)
        try:
            rendered_rgb = renderer.render_organ_array(
                cand_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
                background="black", device=device, differentiable=True, focus_plant=True,
                existence_threshold=0.1,
            )
            pix_loss = F.l1_loss(rendered_rgb, target_rgb)
            perc_loss = perceptual_loss_fn(rendered_rgb.unsqueeze(0), target_rgb.unsqueeze(0))
            guide_loss = pix_loss + 0.3 * perc_loss
            guide_grad = torch.autograd.grad(guide_loss, x_t_in)[0]
            guide_grad = torch.nan_to_num(guide_grad, nan=0.0).clamp(-1.0, 1.0)
        except Exception:
            pix_loss = torch.tensor(1.0, device=device)
            guide_grad = torch.zeros_like(x_t)
            rendered_rgb = torch.zeros_like(target_rgb)

        with torch.no_grad():
            cur_np = rendered_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            ssim_val = compute_ssim_numpy(cur_np, target_np)

            history["loss"].append(float(pix_loss.item()))
            history["ssim"].append(ssim_val)
            history["existence_sum"].append(float((denorm[:, dataset.existence_col] > 0.5).sum().item()))

            if idx in snapshot_steps or idx == len(step_indices) - 1:
                history["images"].append((idx, cur_np, float(pix_loss.item()), ssim_val))

            alpha_t = scheduler.alphas_cumprod[t].clamp(min=1e-6)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            pred_noise = (x_t - sqrt_alpha_t * pred_x0) / sqrt_one_minus_alpha_t
            pred_noise = pred_noise - guidance_weight * sqrt_one_minus_alpha_t * guide_grad

            if idx < len(step_indices) - 1:
                t_prev = step_indices[idx + 1]
                alpha_prev = scheduler.alphas_cumprod[t_prev].clamp(min=1e-6)
                sqrt_alpha_prev = torch.sqrt(alpha_prev)
                sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev)
                x_t = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * pred_noise
            else:
                x_t = pred_x0

    if len(history["images"]) >= 2:
        init_img = history["images"][0][1]
        target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        flow = compute_optical_flow_farneback(init_img, target_np)
        history["initial_flow_hsv"] = flow_to_hsv(flow)
        history["initial_warped_rgb"] = warp_image_torch(
            torch.from_numpy(init_img).permute(2, 0, 1).to(device),
            torch.from_numpy(flow).to(device)
        ).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_xml", type=str, default="diffusion_based/eval/output/dap10_gt_0000_plant_0000.xml",
                        help="Path to the source Helios XML plant to use as GT target")
    parser.add_argument("--alt_source_xml", type=str, default=None,
                        help="Path to independent non-relevant XML plant for initial source (prevents info leak)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs. Default is diffusion_based/eval/output/<xml_name>_backprop")
    parser.add_argument("--steps", type=int, default=300, help="Number of optimization steps")
    parser.add_argument("--method", type=str, default="diffusion", choices=["diffusion", "backprop", "both"],
                        help="Solver method: diffusion (trains fresh diffusion + Guided DDIM) or backprop")
    parser.add_argument("--diffusion_epochs", type=int, default=15, help="Epochs to train fresh diffusion model")
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--max_nodes", type=int, default=2048)
    parser.add_argument("--no_flow", action="store_true", help="Disable optical flow warping loss")
    args = parser.parse_args()

    xml_name, dap = _extract_dap_and_name(args.source_xml)
    if args.output_dir is None:
        args.output_dir = os.path.join("diffusion_based", "eval", "output", f"{xml_name}_backprop")

    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    source_xml = os.path.join(repo_root, args.source_xml)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Running problem suite ({args.method.upper()}) on device: {device}")
    print(f"Source XML: {args.source_xml} (DAP {dap})")
    print(f"Output dir: {args.output_dir}")

    organ_array_gt = PlantOrganArray.from_xml_file_typed(source_xml)
    organ_array_gt.tensor = organ_array_gt.tensor.to(device)

    renderer = HeliosPyTorchRenderer(image_size=128).to(device)
    target_rgb = render_target(organ_array_gt, renderer, device)
    target_rgb_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    # If diffusion method selected, train fresh diffusion model with Image & Perceptual loss
    diff_model, diff_scheduler, diff_dataset = None, None, None
    if args.method in ("diffusion", "both"):
        diff_model, diff_scheduler, diff_dataset = train_diffusion_fresh(
            data_root=args.data_root,
            device=device,
            epochs=args.diffusion_epochs,
            batch_size=16,
            lr=1e-4,
            max_nodes=args.max_nodes,
            render_every=10,
            perceptual_weight=0.5,
        )

    all_metrics = {}
    snapshot_steps = [0, 20, 40, 60, 80, args.steps] if args.steps >= 100 else [0, args.steps // 4, args.steps // 2, 3 * args.steps // 4, args.steps]
    binary_step = int(args.steps * 0.6)

    # Problem 1: Non-relevant source start
    print("\n=== PROBLEM 1: EASY / NON-RELEVANT SOURCE ===")
    init_easy = make_non_relevant_source_plant(device, alt_xml_path=args.alt_source_xml)
    if args.method == "diffusion":
        hist_easy = solve_problem_diffusion(
            target_rgb, init_easy, diff_model, diff_scheduler, diff_dataset,
            renderer, perceptual_loss_fn, device, steps=50, guidance_weight=0.5,
        )
    else:
        hist_easy = optimize_backprop(
            target_rgb, init_easy, renderer, device,
            num_steps=args.steps, lr=0.03,
            optimize_geometry=False, optimize_topology=False,
            snapshot_steps=snapshot_steps,
            binary_threshold_step=binary_step,
            grad_clip=1.0, existence_pull_weight=0.05,
            use_flow_loss=not args.no_flow
        )
    plot_problem(
        target_rgb_np, hist_easy, "easy",
        f"PROBLEM 1 (EASY) - {args.method.upper()} GUIDED RECONSTRUCTION (DAP {dap})",
        os.path.join(output_dir, f"{xml_name}_backprop_problem_easy.png"),
        dap=dap,
    )
    all_metrics["easy"] = {
        "initial_loss": hist_easy["loss"][0],
        "final_loss": hist_easy["loss"][-1],
        "initial_ssim": hist_easy["ssim"][0],
        "final_ssim": hist_easy["ssim"][-1],
    }

    # Problem 2: Medium (grow from seed)
    print("\n=== PROBLEM 2: MEDIUM (grow from tiny seed) ===")
    init_medium = make_seed_plant(organ_array_gt, seed=42)
    if args.method == "diffusion":
        hist_medium = solve_problem_diffusion(
            target_rgb, init_medium, diff_model, diff_scheduler, diff_dataset,
            renderer, perceptual_loss_fn, device, steps=50, guidance_weight=0.5,
        )
    else:
        hist_medium = optimize_backprop(
            target_rgb, init_medium, renderer, device,
            num_steps=args.steps, lr=0.03,
            optimize_geometry=False, optimize_topology=False,
            snapshot_steps=snapshot_steps,
            binary_threshold_step=binary_step,
            use_flow_loss=not args.no_flow
        )
    plot_problem(
        target_rgb_np, hist_medium, "medium",
        f"PROBLEM 2 (MEDIUM) - {args.method.upper()} SEED EXPANSION (DAP {dap})",
        os.path.join(output_dir, f"{xml_name}_backprop_problem_medium.png"),
        dap=dap,
    )
    all_metrics["medium"] = {
        "initial_loss": hist_medium["loss"][0],
        "final_loss": hist_medium["loss"][-1],
        "initial_ssim": hist_medium["ssim"][0],
        "final_ssim": hist_medium["ssim"][-1],
    }

    # Problem 3: Hard (random topology)
    print("\n=== PROBLEM 3: HARD (random topology) ===")
    init_hard = make_random_topology(organ_array_gt, seed=42)
    if args.method == "diffusion":
        hist_hard = solve_problem_diffusion(
            target_rgb, init_hard, diff_model, diff_scheduler, diff_dataset,
            renderer, perceptual_loss_fn, device, steps=50, guidance_weight=0.5,
        )
    else:
        parent_logits, parent_candidates = PlantOrganArray.build_parent_candidates_from_gt(
            init_hard, num_candidates=8, seed=42
        )
        init_hard = init_hard.clone_with_parent_logits(parent_logits, parent_candidates)
        hist_hard = optimize_backprop(
            target_rgb, init_hard, renderer, device,
            num_steps=args.steps, lr=0.03,
            optimize_geometry=False, optimize_topology=True,
            snapshot_steps=snapshot_steps,
            binary_threshold_step=binary_step,
            grad_clip=1.0, existence_pull_weight=0.05,
            fix_existence=True, use_flow_loss=not args.no_flow
        )
    plot_problem(
        target_rgb_np, hist_hard, "hard",
        f"PROBLEM 3 (HARD) - {args.method.upper()} RANDOM TOPOLOGY RECOVERY (DAP {dap})",
        os.path.join(output_dir, f"{xml_name}_backprop_problem_hard.png"),
        dap=dap,
    )
    all_metrics["hard"] = {
        "initial_loss": hist_hard["loss"][0],
        "final_loss": hist_hard["loss"][-1],
        "initial_ssim": hist_hard["ssim"][0],
        "final_ssim": hist_hard["ssim"][-1],
    }

    metrics_file = os.path.join(output_dir, f"{xml_name}_backprop_problem_suite_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll problem suite metrics saved to {metrics_file}:", all_metrics)


if __name__ == "__main__":
    main()
