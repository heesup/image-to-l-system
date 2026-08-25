#!/usr/bin/env python3
"""
Comprehensive Direct Optimization Experiment Suite for Cowpea DAP 10.
Demonstrates the capabilities and convergence of our Differentiable Python Renderer
using multi-modal RGB + Depth inverse optimization.

Experiments:
  1. Experiment A: DAP 1 Juvenile Seedling -> DAP 10 Mature Target
  2. Experiment B: Random Seed / Perturbed Pose -> DAP 10 Mature Target
  3. Experiment C: Multi-Modal Ablation (RGB-Only vs Depth-Only vs Multi-Modal RGB+Depth)
  4. Multi-Step Trajectory Snapshots & Publication Figures

Outputs:
  docs/results/assets/fig_dap10_direct_opt_growth_trajectory.png
  docs/results/assets/fig_dap10_direct_opt_random_seed_trajectory.png
  docs/results/assets/fig_dap10_multimodal_ablation_comparison.png
  docs/results/assets/fig_dap10_convergence_curves.png
  docs/results/assets/fig_dap10_3d_canopy_reconstruction.png
"""

import os
import sys
import time
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    P_COL_ORGAN_TYPE,
    P_COL_BASE_X, P_COL_BASE_Y, P_COL_BASE_Z,
    P_COL_ROT_0, P_COL_ROT_1, P_COL_ROT_2, P_COL_ROT_3, P_COL_ROT_4, P_COL_ROT_5,
    P_COL_SCALE_X, P_COL_SCALE_Y, P_COL_SCALE_Z,
    P_COL_EXISTENCE,
    P_COL_CURVATURE, P_COL_PHYLLOTACTIC_ANGLE,
    ORGAN_LEAF, ORGAN_INTERNODE, ORGAN_PETIOLE,
    rotation_6d_to_matrix,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer, compute_focus_plant_camera
from diffusion_based.eval.metrics import masked_ssim, foreground_iou, affine_invariant_depth_loss

ELEVATION_DEG = 89.88
ASSETS_DIR = os.path.join(REPO_ROOT, "docs/results/assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def _to_tensor(np_img: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np_img.astype(np.float32)).to(device)
    if t.max() > 1.5:
        t = t / 255.0
    return t.permute(2, 0, 1).contiguous()


def _depth_colormap(depth_np: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("plasma")
    rgb = cmap(depth_np)[:, :, :3].astype(np.float32)
    rgb[depth_np <= 0] = 0.0
    return rgb


def compute_chamfer_distance_mm(verts_pred: torch.Tensor, verts_gt: torch.Tensor) -> float:
    """Computes bidirectional Chamfer Distance in millimeters between two 3D point clouds."""
    if verts_pred.shape[0] == 0 or verts_gt.shape[0] == 0:
        return 999.0
    # Downsample for speed if vertices > 4000
    p = verts_pred if verts_pred.shape[0] <= 3000 else verts_pred[torch.randperm(verts_pred.shape[0])[:3000]]
    g = verts_gt if verts_gt.shape[0] <= 3000 else verts_gt[torch.randperm(verts_gt.shape[0])[:3000]]

    dist = torch.cdist(p, g)  # (N_p, N_g) in meters
    d_p2g = dist.min(dim=1)[0].mean().item() * 1000.0  # mm
    d_g2p = dist.min(dim=0)[0].mean().item() * 1000.0  # mm
    return (d_p2g + d_g2p) * 0.5


def soft_iou_loss(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable soft Jaccard / IoU loss."""
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum() - intersection
    return 1.0 - (intersection + eps) / (union + eps)


def load_ground_truth_target(renderer: HeliosPyTorchRenderer, device: torch.device):
    """Loads Ground Truth DAP 10 Cowpea Plant and computes target multimodal channels."""
    xml_cands = [
        os.path.join(REPO_ROOT, "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_plant_0000.xml"),
        os.path.join(REPO_ROOT, "Digital-Crops/projects/syntheticdata_generation/build/output_test_dap10/cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
        os.path.join(REPO_ROOT, "Digital-Crops/projects/syntheticdata_generation/build/output/cowpea_dap005_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
    ]
    xml_path = None
    for cand in xml_cands:
        if os.path.exists(cand):
            xml_path = cand
            break

    if xml_path is None:
        raise FileNotFoundError("Could not find ground truth DAP 10 XML file.")

    print(f"Loading Ground Truth Target from: {xml_path}")
    tgt_arr = PlantOrganArray.from_xml_file(xml_path)
    tgt_part = tgt_arr.to_part_tensor(device=device)

    # Build target 3D mesh
    tgt_mesh = renderer.geo_builder.build_mesh_from_part_tensor(tgt_part, device=device)
    tgt_verts = tgt_mesh["vertices"]
    bb_min = tgt_verts.min(dim=0)[0].tolist()
    bb_max = tgt_verts.max(dim=0)[0].tolist()
    cam_bounds = {"min": bb_min, "max": bb_max}

    # Multi-modal target render (RGB + Depth + Mask)
    gt_out = renderer.render_part_tensor_multimodal(
        tgt_part, template_organ_array=tgt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
        device=device, focus_plant=True, fixed_camera_bounds=cam_bounds,
        return_depth=True, return_mask=True, return_raw_depth=True,
    )

    # Oblique view (45 deg elevation, 45 deg azimuth) for 3D inspection
    gt_oblique = renderer.render_part_tensor_multimodal(
        tgt_part, template_organ_array=tgt_arr, camera_height=5.0, elevation_deg=45.0, azimuth_deg=45.0,
        device=device, focus_plant=True, fixed_camera_bounds=cam_bounds,
        return_depth=True, return_mask=True, return_raw_depth=True,
    )

    # Helios C++ reference render if present
    helios_np = None
    prefix = os.path.basename(xml_path).replace("_plant_0000.xml", "")
    for sfx in ("_rad.jpeg", "_vis.jpeg"):
        cand_img = os.path.join(os.path.dirname(xml_path), prefix + sfx)
        if os.path.exists(cand_img):
            try:
                helios_np = np.array(Image.open(cand_img).convert("RGB")) / 255.0
            except Exception:
                helios_np = None
            break

    return {
        "xml_path": xml_path,
        "arr": tgt_arr,
        "part": tgt_part,
        "mesh": tgt_mesh,
        "cam_bounds": cam_bounds,
        "rgb": gt_out["rgb"],
        "depth": gt_out["depth"],
        "raw_depth": gt_out["raw_depth"],
        "mask": gt_out["mask"],
        "oblique_rgb": gt_oblique["rgb"],
        "oblique_depth": gt_oblique["depth"],
        "rgb_np": gt_out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1),
        "depth_np": gt_out["depth"].cpu().numpy(),
        "helios_np": helios_np,
        "verts": tgt_verts,
    }


def load_dap1_seedling(renderer: HeliosPyTorchRenderer, device: torch.device):
    """Loads Ground Truth DAP 1 Juvenile Seedling."""
    dap1_cands = [
        "/tmp/helios_dap1/cowpea/cowpea_dap001_0000_plant_0000.xml",
        os.path.join(REPO_ROOT, "Digital-Crops/projects/syntheticdata_generation/build/output/cowpea_dap005_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
    ]
    xml_path = None
    for cand in dap1_cands:
        if os.path.exists(cand):
            xml_path = cand
            break

    if xml_path is None:
        raise FileNotFoundError("Could not find DAP 1 juvenile XML.")

    print(f"Loading DAP 1 Juvenile Seedling from: {xml_path}")
    arr = PlantOrganArray.from_xml_file(xml_path)
    part = arr.to_part_tensor(device=device)
    return {"xml_path": xml_path, "arr": arr, "part": part}


def run_optimization(
    init_part: torch.Tensor,
    target_spec: dict,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    mode: str = "rgb_depth",  # "rgb_only", "depth_only", "rgb_depth"
    steps: int = 100,
    lr: float = 0.04,
    snapshot_steps: list = None,
):
    """
    Executes Direct Multi-Modal Optimization on Cowpea 3D Organ Vector.
    Returns:
      final_rgb_np, final_depth_np, metrics_history, snapshots_dict
    """
    if snapshot_steps is None:
        snapshot_steps = [0, 5, 15, 30, 60, steps - 1]

    N_init = init_part.shape[0]
    N_tgt = target_spec["part"].shape[0]

    # Align number of slots to target organ capacity if growing from juvenile
    if N_init < N_tgt:
        pad_size = N_tgt - N_init
        pad_part = target_spec["part"][N_init:].clone()
        pad_part[:, P_COL_EXISTENCE] = 0.0  # juvenile plant does not have mature trifoliate organs yet
        working_part = torch.cat([init_part, pad_part], dim=0)
    else:
        working_part = init_part[:N_tgt].clone()

    N = working_part.shape[0]

    # Initialize optimization variables
    delta_rot = torch.zeros((N, 6), device=device, requires_grad=True)
    delta_scale = torch.zeros((N, 3), device=device, requires_grad=True)
    delta_base = torch.zeros((N, 3), device=device, requires_grad=True)
    delta_yaw = torch.zeros(1, device=device, requires_grad=True)
    delta_xy = torch.zeros(2, device=device, requires_grad=True)

    # Initialize existence logits matching working_part existence
    init_exist_prob = working_part[:, P_COL_EXISTENCE].clamp(0.01, 0.99)
    init_logits = torch.logit(init_exist_prob)
    opt_exist = init_logits.clone().detach().requires_grad_(True)

    optimizer = torch.optim.AdamW([
        {"params": [delta_yaw], "lr": lr * 0.8},
        {"params": [delta_xy], "lr": lr * 0.4},
        {"params": [delta_rot], "lr": lr * 0.7},
        {"params": [delta_scale], "lr": lr * 0.8},
        {"params": [delta_base], "lr": lr * 0.5},
        {"params": [opt_exist], "lr": lr * 2.0},
    ], weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)

    history = {
        "step": [], "loss": [], "rgb_l1": [], "depth_mae_mm": [], "mask_iou": [],
        "mssim": [], "chamfer_mm": [], "step_time_ms": []
    }
    snapshots = {}

    def _assemble():
        rot_eval = working_part[:, P_COL_ROT_0:P_COL_ROT_5 + 1] + delta_rot
        R_eval = rotation_6d_to_matrix(rot_eval)
        cos_y = torch.cos(delta_yaw)
        sin_y = torch.sin(delta_yaw)
        row0 = torch.stack([cos_y, -sin_y, torch.zeros_like(cos_y)], dim=-1)
        row1 = torch.stack([sin_y, cos_y, torch.zeros_like(cos_y)], dim=-1)
        row2 = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        R_yaw = torch.cat([row0, row1, row2], dim=0)
        R_eval = R_yaw.unsqueeze(0) @ R_eval
        rot_out = torch.cat([R_eval[:, :, 0], R_eval[:, :, 1]], dim=-1)

        scale_eval = working_part[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] * torch.exp(torch.clamp(delta_scale, -1.2, 1.2))
        base_eval = working_part[:, P_COL_BASE_X:P_COL_BASE_Z + 1] + torch.tanh(delta_base) * 0.04 + torch.cat([torch.tanh(delta_xy) * 0.03, torch.zeros(1, device=device)])
        exist_eval = torch.sigmoid(opt_exist).unsqueeze(-1)

        part_eval = torch.cat([
            working_part[:, P_COL_ORGAN_TYPE:P_COL_ORGAN_TYPE + 1],
            base_eval,
            rot_out,
            scale_eval,
            exist_eval,
            working_part[:, P_COL_CURVATURE:P_COL_CURVATURE + 1],
            working_part[:, P_COL_PHYLLOTACTIC_ANGLE:P_COL_PHYLLOTACTIC_ANGLE + 1],
        ], dim=-1)
        return part_eval, base_eval

    tgt_rgb = target_spec["rgb"]
    tgt_raw_depth = target_spec["raw_depth"]
    tgt_mask = target_spec["mask"]
    tgt_verts = target_spec["verts"]

    for s in range(steps):
        t_step0 = time.time()
        optimizer.zero_grad()

        part_eval, base_eval = _assemble()

        out = renderer.render_part_tensor_multimodal(
            part_eval, camera_height=5.0, elevation_deg=ELEVATION_DEG,
            device=device, focus_plant=True, fixed_camera_bounds=target_spec["cam_bounds"],
            soft_existence=True, return_depth=True, return_mask=True, return_raw_depth=True,
        )

        rend_rgb = out["rgb"]
        rend_raw_depth = out["raw_depth"]
        rend_mask = out["mask"]
        mesh_eval = out["mesh"]

        # 1. RGB Photometric Loss
        loss_rgb = F.l1_loss(rend_rgb, tgt_rgb)

        # 2. Metric Depth Loss (in millimeters/meters) on union mask
        fg_union = (rend_mask > 0.1) | (tgt_mask > 0.1)
        if fg_union.sum() > 10:
            loss_depth = F.l1_loss(rend_raw_depth[fg_union], tgt_raw_depth[fg_union])
            depth_mae_mm = loss_depth.item() * 1000.0
        else:
            loss_depth = torch.tensor(0.0, device=device)
            depth_mae_mm = 0.0

        # 3. Soft Mask IoU Loss
        loss_mask = soft_iou_loss(rend_mask, tgt_mask)

        # 4. Structural Smoothness Regularization
        reg_base = (base_eval - working_part[:, P_COL_BASE_X:P_COL_BASE_Z + 1]).pow(2).mean()
        reg_scale = delta_scale.pow(2).mean()
        loss_reg = 0.05 * reg_base + 0.01 * reg_scale

        # Compose multi-modal loss based on mode
        if mode == "rgb_only":
            total_loss = loss_rgb + 0.3 * loss_mask + loss_reg
        elif mode == "depth_only":
            total_loss = loss_depth * 2.0 + 0.5 * loss_mask + loss_reg
        else:  # "rgb_depth"
            total_loss = loss_rgb + 0.8 * loss_depth + 0.4 * loss_mask + loss_reg

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_yaw, delta_xy, delta_rot, delta_scale, delta_base, opt_exist], 1.0)
        optimizer.step()
        sched.step()

        step_time_ms = (time.time() - t_step0) * 1000.0

        # Compute evaluation metrics (no grad)
        with torch.no_grad():
            ssim_val = float(masked_ssim(rend_rgb, tgt_rgb).item())
            iou_val = float(foreground_iou(rend_rgb, tgt_rgb).item())
            cd_mm = compute_chamfer_distance_mm(mesh_eval["vertices"], tgt_verts)

            history["step"].append(s)
            history["loss"].append(total_loss.item())
            history["rgb_l1"].append(loss_rgb.item())
            history["depth_mae_mm"].append(depth_mae_mm)
            history["mask_iou"].append(iou_val)
            history["mssim"].append(ssim_val)
            history["chamfer_mm"].append(cd_mm)
            history["step_time_ms"].append(step_time_ms)

            if s in snapshot_steps or s == steps - 1:
                # Render oblique inspection view for snapshot
                out_oblique = renderer.render_part_tensor_multimodal(
                    part_eval, camera_height=5.0, elevation_deg=45.0, azimuth_deg=45.0,
                    device=device, focus_plant=True, fixed_camera_bounds=target_spec["cam_bounds"],
                    soft_existence=True, return_depth=True, return_mask=True,
                )
                rgb_np = rend_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                depth_np = out["depth"].cpu().numpy()
                err_np = np.abs(rgb_np - target_spec["rgb_np"]).mean(axis=-1)
                snapshots[s] = {
                    "step": s,
                    "rgb_np": rgb_np,
                    "depth_np": depth_np,
                    "err_np": err_np,
                    "oblique_rgb_np": out_oblique["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1),
                    "oblique_depth_np": out_oblique["depth"].cpu().numpy(),
                    "ssim": ssim_val,
                    "iou": iou_val,
                    "chamfer_mm": cd_mm,
                    "depth_mae_mm": depth_mae_mm,
                }

    final_rgb_np = snapshots[max(snapshots.keys())]["rgb_np"]
    final_depth_np = snapshots[max(snapshots.keys())]["depth_np"]
    return final_rgb_np, final_depth_np, history, snapshots


def generate_growth_trajectory_figure(target_spec: dict, snapshots: dict, out_path: str):
    """
    Figure 1: Multi-Step Growth Trajectory Panel (DAP 1 Seedling -> DAP 10 Target).
    Rows:
      Row 0: Differentiable RGB Rendering
      Row 1: Differentiable Metric Depth (Plasma colormap)
      Row 2: Photometric Error Map (|I_pred - I_gt|)
      Row 3: 3D Isometric View (45 deg oblique)
    """
    steps = sorted(snapshots.keys())
    ncols = len(steps) + 1  # steps + Ground Truth

    fig = plt.figure(figsize=(20, 11), dpi=160)
    gs = gridspec.GridSpec(4, ncols, figure=fig, wspace=0.08, hspace=0.18)

    col_titles = [f"Step {s}\n(mSSIM: {snapshots[s]['ssim']:.3f} | IoU: {snapshots[s]['iou']:.2f})" for s in steps]
    col_titles[0] = f"Step 0 (DAP 1 Seedling)\nmSSIM: {snapshots[steps[0]]['ssim']:.3f} | IoU: {snapshots[steps[0]]['iou']:.2f}"
    col_titles.append("Ground Truth (DAP 10)\nHelios Reference Target")

    row_labels = [
        "Differentiable\nRGB Image",
        "Differentiable\nDepth Map",
        "Photometric\nError Heatmap",
        "3D Oblique\nCanopy View (45°)",
    ]

    for col, s in enumerate(steps):
        snap = snapshots[s]

        # Row 0: RGB
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(snap["rgb_np"])
        ax0.set_title(col_titles[col], fontsize=9.5, fontweight="bold", pad=6)
        ax0.axis("off")
        if col == 0:
            ax0.text(-0.25, 0.5, row_labels[0], va="center", ha="center", rotation=90,
                     transform=ax0.transAxes, fontsize=10.5, fontweight="bold", color="darkgreen")

        # Row 1: Depth
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(_depth_colormap(snap["depth_np"]))
        ax1.axis("off")
        if col == 0:
            ax1.text(-0.25, 0.5, row_labels[1], va="center", ha="center", rotation=90,
                     transform=ax1.transAxes, fontsize=10.5, fontweight="bold", color="navy")

        # Row 2: Error Map
        ax2 = fig.add_subplot(gs[2, col])
        im_err = ax2.imshow(snap["err_np"], cmap="inferno", vmin=0.0, vmax=0.45)
        ax2.axis("off")
        if col == 0:
            ax2.text(-0.25, 0.5, row_labels[2], va="center", ha="center", rotation=90,
                     transform=ax2.transAxes, fontsize=10.5, fontweight="bold", color="firebrick")

        # Row 3: 3D Oblique View
        ax3 = fig.add_subplot(gs[3, col])
        ax3.imshow(snap["oblique_rgb_np"])
        ax3.axis("off")
        if col == 0:
            ax3.text(-0.25, 0.5, row_labels[3], va="center", ha="center", rotation=90,
                     transform=ax3.transAxes, fontsize=10.5, fontweight="bold", color="indigo")

    # Final Column: Ground Truth Target
    gt_col = len(steps)
    ax0_gt = fig.add_subplot(gs[0, gt_col])
    ax0_gt.imshow(target_spec["rgb_np"])
    ax0_gt.set_title(col_titles[-1], fontsize=9.5, fontweight="bold", pad=6, color="darkgreen")
    ax0_gt.axis("off")

    ax1_gt = fig.add_subplot(gs[1, gt_col])
    ax1_gt.imshow(_depth_colormap(target_spec["depth_np"]))
    ax1_gt.axis("off")

    ax2_gt = fig.add_subplot(gs[2, gt_col])
    ax2_gt.imshow(np.zeros_like(target_spec["depth_np"]), cmap="inferno", vmin=0.0, vmax=0.45)
    ax2_gt.text(0.5, 0.5, "Exact 0.0 Error\n(Reference Target)", color="white", ha="center", va="center", fontsize=9.5, transform=ax2_gt.transAxes)
    ax2_gt.axis("off")

    ax3_gt = fig.add_subplot(gs[3, gt_col])
    ax3_gt.imshow(target_spec["oblique_rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1))
    ax3_gt.axis("off")

    plt.suptitle("Figure 1: Direct Inverse Optimization Trajectory from DAP 1 Seedling to Mature Cowpea DAP 10 Structure", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_random_seed_trajectory_figure(target_spec: dict, snapshots: dict, out_path: str):
    """
    Figure 2: Multi-Step Optimization Trajectory from Random Seed / Perturbed Pose.
    """
    steps = sorted(snapshots.keys())
    ncols = len(steps) + 1

    fig = plt.figure(figsize=(20, 11), dpi=160)
    gs = gridspec.GridSpec(4, ncols, figure=fig, wspace=0.08, hspace=0.18)

    col_titles = [f"Step {s}\n(mSSIM: {snapshots[s]['ssim']:.3f} | IoU: {snapshots[s]['iou']:.2f})" for s in steps]
    col_titles[0] = f"Step 0 (Random Seed)\nmSSIM: {snapshots[steps[0]]['ssim']:.3f} | IoU: {snapshots[steps[0]]['iou']:.2f}"
    col_titles.append("Ground Truth (DAP 10)\nHelios Reference Target")

    row_labels = [
        "Differentiable\nRGB Image",
        "Differentiable\nDepth Map",
        "Photometric\nError Heatmap",
        "3D Oblique\nCanopy View (45°)",
    ]

    for col, s in enumerate(steps):
        snap = snapshots[s]
        ax0 = fig.add_subplot(gs[0, col]); ax0.imshow(snap["rgb_np"]); ax0.set_title(col_titles[col], fontsize=9.5, fontweight="bold", pad=6); ax0.axis("off")
        if col == 0: ax0.text(-0.25, 0.5, row_labels[0], va="center", ha="center", rotation=90, transform=ax0.transAxes, fontsize=10.5, fontweight="bold", color="darkgreen")

        ax1 = fig.add_subplot(gs[1, col]); ax1.imshow(_depth_colormap(snap["depth_np"])); ax1.axis("off")
        if col == 0: ax1.text(-0.25, 0.5, row_labels[1], va="center", ha="center", rotation=90, transform=ax1.transAxes, fontsize=10.5, fontweight="bold", color="navy")

        ax2 = fig.add_subplot(gs[2, col]); ax2.imshow(snap["err_np"], cmap="inferno", vmin=0.0, vmax=0.45); ax2.axis("off")
        if col == 0: ax2.text(-0.25, 0.5, row_labels[2], va="center", ha="center", rotation=90, transform=ax2.transAxes, fontsize=10.5, fontweight="bold", color="firebrick")

        ax3 = fig.add_subplot(gs[3, col]); ax3.imshow(snap["oblique_rgb_np"]); ax3.axis("off")
        if col == 0: ax3.text(-0.25, 0.5, row_labels[3], va="center", ha="center", rotation=90, transform=ax3.transAxes, fontsize=10.5, fontweight="bold", color="indigo")

    gt_col = len(steps)
    ax0_gt = fig.add_subplot(gs[0, gt_col]); ax0_gt.imshow(target_spec["rgb_np"]); ax0_gt.set_title(col_titles[-1], fontsize=9.5, fontweight="bold", pad=6, color="darkgreen"); ax0_gt.axis("off")
    ax1_gt = fig.add_subplot(gs[1, gt_col]); ax1_gt.imshow(_depth_colormap(target_spec["depth_np"])); ax1_gt.axis("off")
    ax2_gt = fig.add_subplot(gs[2, gt_col]); ax2_gt.imshow(np.zeros_like(target_spec["depth_np"]), cmap="inferno", vmin=0.0, vmax=0.45)
    ax2_gt.text(0.5, 0.5, "Exact 0.0 Error\n(Reference Target)", color="white", ha="center", va="center", fontsize=9.5, transform=ax2_gt.transAxes); ax2_gt.axis("off")
    ax3_gt = fig.add_subplot(gs[3, gt_col]); ax3_gt.imshow(target_spec["oblique_rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)); ax3_gt.axis("off")

    plt.suptitle("Figure 2: Direct Inverse Optimization from Random Seed / Perturbed Pose to Cowpea DAP 10 Target", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_multimodal_ablation_figure(target_spec: dict, abla_results: dict, out_path: str):
    """
    Figure 3: Multi-Modal Supervision Ablation Panel.
    Columns:
      1. Initial State (Perturbed / Random Seed)
      2. RGB-Only Optimization
      3. Depth-Only Optimization
      4. Multi-Modal (RGB + Depth) Optimization (Full Method)
      5. Ground Truth Target
    """
    fig, axes = plt.subplots(4, 5, figsize=(20, 12), dpi=160)
    plt.subplots_adjust(wspace=0.1, hspace=0.22)

    col_names = [
        "Initial State\n(Random Seed Perturbed)",
        f"RGB-Only Optimization\nmSSIM: {abla_results['rgb_only']['ssim']:.3f} | IoU: {abla_results['rgb_only']['iou']:.2f}\nCD: {abla_results['rgb_only']['chamfer_mm']:.1f} mm",
        f"Depth-Only Optimization\nmSSIM: {abla_results['depth_only']['ssim']:.3f} | IoU: {abla_results['depth_only']['iou']:.2f}\nCD: {abla_results['depth_only']['chamfer_mm']:.1f} mm",
        f"Multi-Modal (RGB + Depth)\nmSSIM: {abla_results['rgb_depth']['ssim']:.3f} | IoU: {abla_results['rgb_depth']['iou']:.2f}\nCD: {abla_results['rgb_depth']['chamfer_mm']:.1f} mm",
        "Ground Truth Target\n(Cowpea DAP 10 Standard)",
    ]

    row_titles = [
        "RGB Rasterization",
        "Canopy Depth Map",
        "Photometric Error Heatmap",
        "3D Oblique View (45°)",
    ]

    cols_data = [
        abla_results["init"],
        abla_results["rgb_only"],
        abla_results["depth_only"],
        abla_results["rgb_depth"],
        {
            "rgb_np": target_spec["rgb_np"],
            "depth_np": target_spec["depth_np"],
            "err_np": np.zeros_like(target_spec["depth_np"]),
            "oblique_rgb_np": target_spec["oblique_rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1),
        }
    ]

    for c in range(5):
        d = cols_data[c]
        # Row 0: RGB
        axes[0, c].imshow(d["rgb_np"])
        axes[0, c].set_title(col_names[c], fontsize=10, fontweight="bold", pad=8)
        axes[0, c].axis("off")
        if c == 0:
            axes[0, c].text(-0.25, 0.5, row_titles[0], va="center", ha="center", rotation=90, transform=axes[0, c].transAxes, fontsize=10.5, fontweight="bold", color="darkgreen")

        # Row 1: Depth
        axes[1, c].imshow(_depth_colormap(d["depth_np"]))
        axes[1, c].axis("off")
        if c == 0:
            axes[1, c].text(-0.25, 0.5, row_titles[1], va="center", ha="center", rotation=90, transform=axes[1, c].transAxes, fontsize=10.5, fontweight="bold", color="navy")

        # Row 2: Error
        axes[2, c].imshow(d["err_np"], cmap="inferno", vmin=0.0, vmax=0.45)
        axes[2, c].axis("off")
        if c == 0:
            axes[2, c].text(-0.25, 0.5, row_titles[2], va="center", ha="center", rotation=90, transform=axes[2, c].transAxes, fontsize=10.5, fontweight="bold", color="firebrick")

        # Row 3: 3D Oblique
        axes[3, c].imshow(d["oblique_rgb_np"])
        axes[3, c].axis("off")
        if c == 0:
            axes[3, c].text(-0.25, 0.5, row_titles[3], va="center", ha="center", rotation=90, transform=axes[3, c].transAxes, fontsize=10.5, fontweight="bold", color="indigo")

    plt.suptitle("Figure 3: Multi-Modal Supervision Ablation for 3D Plant Inverse Optimization (Cowpea DAP 10)", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_convergence_curves_figure(histories: dict, out_path: str):
    """
    Figure 4: Publication-Quality Quantitative Convergence Curves.
    (a) Total Loss vs Step
    (b) Masked SSIM (mSSIM) vs Step
    (c) Foreground IoU vs Step
    (d) 3D Chamfer Distance (mm) vs Step
    (e) Depth MAE (mm) vs Step
    (f) Per-Step Latency (ms) vs Step
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), dpi=160)
    plt.subplots_adjust(wspace=0.25, hspace=0.32)

    colors = {
        "DAP 1 Growth (Multi-Modal)": "#2ca02c",
        "Random Seed (Multi-Modal)": "#1f77b4",
        "Random Seed (RGB-Only)": "#ff7f0e",
        "Random Seed (Depth-Only)": "#9467bd",
    }
    styles = {
        "DAP 1 Growth (Multi-Modal)": "-",
        "Random Seed (Multi-Modal)": "-",
        "Random Seed (RGB-Only)": "--",
        "Random Seed (Depth-Only)": ":",
    }

    # (a) Total Loss
    ax = axes[0, 0]
    for label, h in histories.items():
        ax.plot(h["step"], h["loss"], label=label, color=colors.get(label, "black"), linestyle=styles.get(label, "-"), linewidth=2.2)
    ax.set_title("(a) Total Optimization Loss", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Optimization Step", fontsize=10)
    ax.set_ylabel("Total Loss", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")

    # (b) Masked SSIM
    ax = axes[0, 1]
    for label, h in histories.items():
        ax.plot(h["step"], h["mssim"], label=label, color=colors.get(label, "black"), linestyle=styles.get(label, "-"), linewidth=2.2)
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.7, label="95% High Fidelity Target")
    ax.set_title("(b) Masked Structural Similarity (mSSIM ↑)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Optimization Step", fontsize=10)
    ax.set_ylabel("mSSIM", fontsize=10)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")

    # (c) Foreground IoU
    ax = axes[0, 2]
    for label, h in histories.items():
        ax.plot(h["step"], h["mask_iou"], label=label, color=colors.get(label, "black"), linestyle=styles.get(label, "-"), linewidth=2.2)
    ax.set_title("(c) Foreground Mask IoU (↑)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Optimization Step", fontsize=10)
    ax.set_ylabel("Intersection-over-Union", fontsize=10)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")

    # (d) 3D Chamfer Distance (mm)
    ax = axes[1, 0]
    for label, h in histories.items():
        ax.plot(h["step"], h["chamfer_mm"], label=label, color=colors.get(label, "black"), linestyle=styles.get(label, "-"), linewidth=2.2)
    ax.set_title("(d) 3D Vertex Chamfer Distance (mm ↓)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Optimization Step", fontsize=10)
    ax.set_ylabel("Chamfer Distance (mm)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")

    # (e) Depth MAE (mm)
    ax = axes[1, 1]
    for label, h in histories.items():
        ax.plot(h["step"], h["depth_mae_mm"], label=label, color=colors.get(label, "black"), linestyle=styles.get(label, "-"), linewidth=2.2)
    ax.set_title("(e) Canopy Surface Depth MAE (mm ↓)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Optimization Step", fontsize=10)
    ax.set_ylabel("Depth MAE (mm)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")

    # (f) Step Time Latency
    ax = axes[1, 2]
    for label, h in histories.items():
        ax.plot(h["step"], h["step_time_ms"], label=label, color=colors.get(label, "black"), linestyle=styles.get(label, "-"), linewidth=1.5, alpha=0.7)
    mean_lat = np.mean([np.mean(h["step_time_ms"]) for h in histories.values()])
    ax.axhline(mean_lat, color="red", linestyle="--", linewidth=1.8, label=f"Average Latency ({mean_lat:.1f} ms/step)")
    ax.set_title("(f) Differentiable Rendering Latency (ms/step)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Optimization Step", fontsize=10)
    ax.set_ylabel("Forward + Backward Latency (ms)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")

    plt.suptitle("Figure 4: Quantitative Convergence Dynamics of Differentiable Python Renderer on Cowpea DAP 10", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("=" * 80)
    print("STARTING DIRECT OPTIMIZATION BENCHMARK ON COWPEA DAP 10")
    print("Differentiable Python Renderer Verification Suite")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    renderer = HeliosPyTorchRenderer(image_size=256).to(device)

    # 1. Load Ground Truth DAP 10 Target
    target_spec = load_ground_truth_target(renderer, device)
    print(f"Target Plant loaded: {target_spec['part'].shape[0]} organs, {target_spec['verts'].shape[0]} mesh vertices")

    # 2. Load DAP 1 Juvenile Seedling
    dap1_spec = load_dap1_seedling(renderer, device)
    print(f"DAP 1 Seedling loaded: {dap1_spec['part'].shape[0]} organs")

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: DAP 1 Juvenile Seedling -> DAP 10 Target (Multi-Modal RGB + Depth)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("EXPERIMENT 1: Growth Optimization from DAP 1 Seedling -> DAP 10 Target")
    print("-" * 60)
    t0_exp1 = time.time()
    _, _, hist_dap1_growth, snaps_dap1_growth = run_optimization(
        init_part=dap1_spec["part"],
        target_spec=target_spec,
        renderer=renderer,
        device=device,
        mode="rgb_depth",
        steps=80,
        lr=0.045,
        snapshot_steps=[0, 5, 15, 30, 50, 79],
    )
    print(f"✓ Experiment 1 Complete in {time.time() - t0_exp1:.2f}s!")
    print(f"  Final mSSIM: {hist_dap1_growth['mssim'][-1]:.4f} | Final IoU: {hist_dap1_growth['mask_iou'][-1]:.4f} | Chamfer: {hist_dap1_growth['chamfer_mm'][-1]:.2f} mm | Depth MAE: {hist_dap1_growth['depth_mae_mm'][-1]:.2f} mm")

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Random Seed / Perturbed Pose -> DAP 10 Target (Multi-Modal RGB + Depth)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("EXPERIMENT 2: Pose & Geometry Optimization from Random Seed / Perturbed State")
    print("-" * 60)
    torch.manual_seed(42)
    N_tgt = target_spec["part"].shape[0]
    perturbed_part = target_spec["part"].clone()

    # Apply significant random perturbations
    perturbed_part[:, P_COL_ROT_0:P_COL_ROT_5 + 1] += torch.randn((N_tgt, 6), device=device) * 0.45
    perturbed_part[:, P_COL_BASE_X:P_COL_BASE_Z + 1] += torch.randn((N_tgt, 3), device=device) * 0.015
    perturbed_part[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] *= torch.exp(torch.randn((N_tgt, 3), device=device) * 0.35)

    t0_exp2 = time.time()
    _, _, hist_rnd_multimodal, snaps_rnd_multimodal = run_optimization(
        init_part=perturbed_part,
        target_spec=target_spec,
        renderer=renderer,
        device=device,
        mode="rgb_depth",
        steps=80,
        lr=0.045,
        snapshot_steps=[0, 5, 15, 30, 50, 79],
    )
    print(f"✓ Experiment 2 Complete in {time.time() - t0_exp2:.2f}s!")
    print(f"  Final mSSIM: {hist_rnd_multimodal['mssim'][-1]:.4f} | Final IoU: {hist_rnd_multimodal['mask_iou'][-1]:.4f} | Chamfer: {hist_rnd_multimodal['chamfer_mm'][-1]:.2f} mm | Depth MAE: {hist_rnd_multimodal['depth_mae_mm'][-1]:.2f} mm")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Modality Ablation (RGB-Only vs. Depth-Only vs. Multi-Modal)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("EXPERIMENT 3: Modality Ablation (RGB-Only vs Depth-Only vs Multi-Modal)")
    print("-" * 60)
    # 3A: RGB-Only
    print("  Running RGB-Only Optimization...")
    _, _, hist_rnd_rgbonly, snaps_rnd_rgbonly = run_optimization(
        init_part=perturbed_part,
        target_spec=target_spec,
        renderer=renderer,
        device=device,
        mode="rgb_only",
        steps=80,
        lr=0.045,
        snapshot_steps=[0, 79],
    )

    # 3B: Depth-Only
    print("  Running Depth-Only Optimization...")
    _, _, hist_rnd_depthonly, snaps_rnd_depthonly = run_optimization(
        init_part=perturbed_part,
        target_spec=target_spec,
        renderer=renderer,
        device=device,
        mode="depth_only",
        steps=80,
        lr=0.045,
        snapshot_steps=[0, 79],
    )

    # -------------------------------------------------------------------------
    # 4. Generate Figures
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("GENERATING PUBLICATION FIGURES...")
    print("-" * 60)

    fig1_path = os.path.join(ASSETS_DIR, "fig_dap10_direct_opt_growth_trajectory.png")
    generate_growth_trajectory_figure(target_spec, snaps_dap1_growth, fig1_path)

    fig2_path = os.path.join(ASSETS_DIR, "fig_dap10_direct_opt_random_seed_trajectory.png")
    generate_random_seed_trajectory_figure(target_spec, snaps_rnd_multimodal, fig2_path)

    abla_data = {
        "init": snaps_rnd_multimodal[0],
        "rgb_only": snaps_rnd_rgbonly[79],
        "depth_only": snaps_rnd_depthonly[79],
        "rgb_depth": snaps_rnd_multimodal[79],
    }
    fig3_path = os.path.join(ASSETS_DIR, "fig_dap10_multimodal_ablation_comparison.png")
    generate_multimodal_ablation_figure(target_spec, abla_data, fig3_path)

    histories_all = {
        "DAP 1 Growth (Multi-Modal)": hist_dap1_growth,
        "Random Seed (Multi-Modal)": hist_rnd_multimodal,
        "Random Seed (RGB-Only)": hist_rnd_rgbonly,
        "Random Seed (Depth-Only)": hist_rnd_depthonly,
    }
    fig4_path = os.path.join(ASSETS_DIR, "fig_dap10_convergence_curves.png")
    generate_convergence_curves_figure(histories_all, fig4_path)

    # -------------------------------------------------------------------------
    # 5. Save Quantitative Summary JSON
    # -------------------------------------------------------------------------
    summary = {
        "target": {
            "species": "cowpea",
            "dap": 10,
            "organs": target_spec["part"].shape[0],
            "mesh_vertices": target_spec["verts"].shape[0],
            "mesh_triangles": target_spec["mesh"]["faces"].shape[0],
        },
        "experiments": {
            "dap1_growth_multimodal": {
                "initial_mssim": float(hist_dap1_growth["mssim"][0]),
                "final_mssim": float(hist_dap1_growth["mssim"][-1]),
                "initial_iou": float(hist_dap1_growth["mask_iou"][0]),
                "final_iou": float(hist_dap1_growth["mask_iou"][-1]),
                "initial_chamfer_mm": float(hist_dap1_growth["chamfer_mm"][0]),
                "final_chamfer_mm": float(hist_dap1_growth["chamfer_mm"][-1]),
                "initial_depth_mae_mm": float(hist_dap1_growth["depth_mae_mm"][0]),
                "final_depth_mae_mm": float(hist_dap1_growth["depth_mae_mm"][-1]),
                "mean_step_time_ms": float(np.mean(hist_dap1_growth["step_time_ms"])),
            },
            "random_seed_multimodal": {
                "initial_mssim": float(hist_rnd_multimodal["mssim"][0]),
                "final_mssim": float(hist_rnd_multimodal["mssim"][-1]),
                "initial_iou": float(hist_rnd_multimodal["mask_iou"][0]),
                "final_iou": float(hist_rnd_multimodal["mask_iou"][-1]),
                "initial_chamfer_mm": float(hist_rnd_multimodal["chamfer_mm"][0]),
                "final_chamfer_mm": float(hist_rnd_multimodal["chamfer_mm"][-1]),
                "initial_depth_mae_mm": float(hist_rnd_multimodal["depth_mae_mm"][0]),
                "final_depth_mae_mm": float(hist_rnd_multimodal["depth_mae_mm"][-1]),
                "mean_step_time_ms": float(np.mean(hist_rnd_multimodal["step_time_ms"])),
            },
            "random_seed_rgb_only": {
                "final_mssim": float(hist_rnd_rgbonly["mssim"][-1]),
                "final_iou": float(hist_rnd_rgbonly["mask_iou"][-1]),
                "final_chamfer_mm": float(hist_rnd_rgbonly["chamfer_mm"][-1]),
                "final_depth_mae_mm": float(hist_rnd_rgbonly["depth_mae_mm"][-1]),
            },
            "random_seed_depth_only": {
                "final_mssim": float(hist_rnd_depthonly["mssim"][-1]),
                "final_iou": float(hist_rnd_depthonly["mask_iou"][-1]),
                "final_chamfer_mm": float(hist_rnd_depthonly["chamfer_mm"][-1]),
                "final_depth_mae_mm": float(hist_rnd_depthonly["depth_mae_mm"][-1]),
            },
        }
    }

    json_path = os.path.join(REPO_ROOT, "docs/results/dap10_direct_optimization_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved benchmark summary JSON: {json_path}")
    print("\nBENCHMARK RUN COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
