"""
Generate honest 16D-only report figures (no aspirational learned-method numbers).

Outputs:
  docs/results/assets/fig3_direct_opt_multi_dap.png
  docs/results/assets/fig4_direct_opt_strategies.png
  docs/results/assets/fig5_direct_opt_ablation.png
  docs/results/assets/fig6_real_loss_convergence.png
  docs/results/assets/fig7_botanical_3d_canopy_metrics.png

All numbers come from actual 16D direct optimization runs against ground-truth
Helios renderings. No ViT/decoder/diffusion checkpoints are loaded.
"""

import os
import sys
import time
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_LEAF,
    P_COL_ORGAN_TYPE,
    P_COL_BASE_X, P_COL_BASE_Y, P_COL_BASE_Z,
    P_COL_ROT_0, P_COL_ROT_5,
    P_COL_SCALE_X, P_COL_SCALE_Y, P_COL_SCALE_Z,
    P_COL_EXISTENCE,
    P_COL_CURVATURE, P_COL_PHYLLOTACTIC_ANGLE,
    rotation_6d_to_matrix,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.eval.metrics import (
    masked_ssim, foreground_iou, affine_invariant_depth_loss,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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


def run_direct_opt(
    init_array: PlantOrganArray,
    target_rgb: torch.Tensor,
    target_raw_depth: torch.Tensor,
    target_mask: torch.Tensor,
    cam_bounds: Tuple,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    mode: str = "rgb_depth",
    steps: int = 35,
    lr: float = 0.04,
    target_leaf_bases: Optional[torch.Tensor] = None,
    chamfer_weight: float = 0.0,
    empty_start: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    16D part-centric direct optimization.  Modes:
      "rgb"      : RGB L1 only
      "rgb_depth": RGB L1 + affine-invariant raw depth L1
      "rgb_depth_chamfer": + optional Chamfer leaf-base loss
    If empty_start=True, begin from an empty plant (all existence=0) and grow
    organs via soft-existence rendering.
    Returns (rgb_np, depth_np, metrics).
    """
    part_init = init_array.to_part_tensor(device=device)
    N = part_init.shape[0]
    if empty_start:
        part_init = part_init.clone()
        part_init[:, P_COL_EXISTENCE] = 0.0

    init_center = part_init[:, P_COL_BASE_X:P_COL_BASE_Z + 1].mean(dim=0, keepdim=True)
    canopy_radius = float((part_init[:, P_COL_BASE_X:P_COL_BASE_Z + 1] - init_center).norm(dim=-1).max().item()) + 0.01
    max_local_shift = canopy_radius * 0.5

    # Global rigid transform (yaw + 2D translation) + per-organ residuals.
    delta_yaw = torch.zeros(1, device=device, requires_grad=True)
    delta_xy = torch.zeros(2, device=device, requires_grad=True)
    delta_rot_6d = torch.zeros((N, 6), device=device, requires_grad=True)
    delta_scale = torch.zeros((N, 3), device=device, requires_grad=True)
    delta_base = torch.zeros((N, 3), device=device, requires_grad=True)
    # Initialize existence logit so sigmoid(logit) = stored existence.
    # (The stored tensor holds existence directly in [0,1], not a logit.)
    exist_init = part_init[:, P_COL_EXISTENCE].clamp(1e-3, 1 - 1e-3).clone().detach()
    opt_exist = torch.logit(exist_init).requires_grad_(True)

    optimizer = torch.optim.AdamW([
        {"params": [delta_yaw], "lr": lr * 1.5},
        {"params": [delta_xy], "lr": lr * 1.0},
        {"params": [delta_rot_6d], "lr": lr * 0.8},
        {"params": [delta_scale], "lr": lr * 0.8},
        {"params": [delta_base], "lr": lr * 0.4},
        {"params": [opt_exist], "lr": lr * 0.6},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)

    leaf_mask = part_init[:, P_COL_ORGAN_TYPE].long() == ORGAN_LEAF
    history = {"loss": [], "rgb_loss": [], "depth_loss": [], "ssim": [], "time": []}
    t_start = time.time()

    def _assemble():
        rot_6d_eval = part_init[:, P_COL_ROT_0:P_COL_ROT_5 + 1] + delta_rot_6d
        R_eval = rotation_6d_to_matrix(rot_6d_eval)
        cos_y, sin_y = torch.cos(delta_yaw), torch.sin(delta_yaw)
        R_global_yaw = torch.eye(3, device=device)
        R_global_yaw[0, 0] = cos_y; R_global_yaw[0, 1] = -sin_y
        R_global_yaw[1, 0] = sin_y; R_global_yaw[1, 1] = cos_y
        R_eval = R_global_yaw.unsqueeze(0) @ R_eval
        rot_6d_out = torch.cat([R_eval[:, :, 0], R_eval[:, :, 1]], dim=-1)
        scale_eval = part_init[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] * torch.exp(torch.clamp(delta_scale, -0.6, 0.6) * 0.5)
        bounded_shift = torch.tanh(delta_base) * max_local_shift
        bases_eval = part_init[:, P_COL_BASE_X:P_COL_BASE_Z + 1] + bounded_shift
        # Apply global translation in the canopy plane.
        bases_eval = bases_eval + torch.cat([delta_xy, torch.zeros(1, device=device)])
        return rot_6d_out, scale_eval, bases_eval

    for s in range(steps):
        optimizer.zero_grad()
        rot_6d_out, scale_eval, bases_eval = _assemble()
        # With empty_start we want a true zero baseline with a strong gradient,
        # so use the raw existence value; otherwise keep it in [0,1] via sigmoid.
        exist_eval = (opt_exist if empty_start else torch.sigmoid(opt_exist)).unsqueeze(-1)
        # New layout: [Existence, OrganType, Base, Rot6D, Scale, Curv, Phyllo]
        part_eval = torch.cat([
            exist_eval,
            part_init[:, P_COL_ORGAN_TYPE:P_COL_ORGAN_TYPE + 1],
            bases_eval, rot_6d_out, scale_eval,
            part_init[:, P_COL_CURVATURE:P_COL_CURVATURE + 1],
            part_init[:, P_COL_PHYLLOTACTIC_ANGLE:P_COL_PHYLLOTACTIC_ANGLE + 1],
        ], dim=-1)

        out = renderer.render_part_tensor_multimodal(
            part_eval, template_organ_array=init_array, camera_height=5.0, elevation_deg=ELEVATION_DEG,
            device=device, focus_plant=True, use_kinematics_tree=False,
            fixed_camera_bounds=cam_bounds, return_depth=False, return_mask=True,
            return_organ_masks=False, return_raw_depth=True, soft_existence=empty_start,
        )
        rend = out["rgb"]
        pred_raw_depth = out["raw_depth"]
        pred_mask = out["mask"]

        loss_rgb = F.l1_loss(rend, target_rgb)
        loss = loss_rgb
        loss_depth = torch.tensor(0.0, device=device)
        if "depth" in mode:
            # Use the intersection of pred & target foreground so the affine
            # depth loss is not skewed by target-only pixels where pred=0.
            fg_inter = pred_mask & target_mask
            loss_depth = affine_invariant_depth_loss(pred_raw_depth, target_raw_depth, mask=fg_inter)
            loss = loss + 0.5 * loss_depth
        # Structural regularization: keep organ bases from drifting far from the init
        # (prevents the optimizer from scattering organs and destroying the plant).
        reg = (bases_eval - part_init[:, P_COL_BASE_X:P_COL_BASE_Z + 1]).pow(2).mean()
        loss = loss + 0.5 * reg

        if "chamfer" in mode and chamfer_weight > 0.0 and target_leaf_bases is not None:
            src_leaf_bases = bases_eval[leaf_mask]
            d = torch.cdist(target_leaf_bases, src_leaf_bases)
            loss = loss + chamfer_weight * d.min(dim=1)[0].mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_yaw, delta_xy, delta_rot_6d, delta_scale, delta_base, opt_exist], 1.0)
        optimizer.step()
        sched.step()

        with torch.no_grad():
            cur_np = rend.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            tgt_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            history["loss"].append(float(loss.item()))
            history["rgb_loss"].append(float(loss_rgb.item()))
            history["depth_loss"].append(float(loss_depth.item()))
            history["ssim"].append(float(masked_ssim(_to_tensor(cur_np, device), _to_tensor(tgt_np, device)).item()))
            history["time"].append(time.time() - t_start)

    with torch.no_grad():
        rot_6d_out, scale_eval, bases_eval = _assemble()
        exist_eval = (opt_exist if empty_start else torch.sigmoid(opt_exist)).unsqueeze(-1)
        part_final = torch.cat([
            exist_eval,
            part_init[:, P_COL_ORGAN_TYPE:P_COL_ORGAN_TYPE + 1],
            bases_eval, rot_6d_out, scale_eval,
            part_init[:, P_COL_CURVATURE:P_COL_CURVATURE + 1],
            part_init[:, P_COL_PHYLLOTACTIC_ANGLE:P_COL_PHYLLOTACTIC_ANGLE + 1],
        ], dim=-1)
        out_final = renderer.render_part_tensor_multimodal(
            part_final, template_organ_array=init_array, camera_height=5.0, elevation_deg=ELEVATION_DEG,
            device=device, focus_plant=True, use_kinematics_tree=False,
            fixed_camera_bounds=cam_bounds, return_depth=True, return_mask=False,
            return_organ_masks=False, soft_existence=empty_start,
        )
        rgb_np = out_final["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        depth_np = out_final["depth"].cpu().numpy()

    final_ssim = float(masked_ssim(_to_tensor(rgb_np, device), _to_tensor(target_rgb.permute(1,2,0).cpu().numpy(), device)).item())
    final_iou = float(foreground_iou(_to_tensor(rgb_np, device), _to_tensor(target_rgb.permute(1,2,0).cpu().numpy(), device)).item())
    metrics = {"ssim": final_ssim, "iou": final_iou, "history": history}
    return rgb_np, depth_np, metrics


ELEVATION_DEG = 89.88  # matches Helios radiation camera

def _load_target_init_pair(renderer, device, tgt_rel, init_rel):
    tgt_arr = PlantOrganArray.from_xml_file(os.path.join(repo_root, tgt_rel))
    tgt_part = tgt_arr.to_part_tensor(device=device)

    # Load the original Helios C++ render for the target (seed00).
    helios_np = None
    tgt_xml = os.path.join(repo_root, tgt_rel)
    tgt_prefix = os.path.basename(tgt_xml).replace("_plant_0000.xml", "")
    from PIL import Image
    for suffix in ("_rad.jpeg", "_vis.jpeg"):
        cand = os.path.join(os.path.dirname(tgt_xml), tgt_prefix + suffix)
        if os.path.exists(cand):
            try:
                helios_np = np.array(Image.open(cand).convert("RGB")) / 255.0
            except Exception:
                helios_np = None
            break

    tgt_mesh = renderer.geo_builder.build_mesh_from_part_array(
        tgt_part, template_organ_array=tgt_arr, device=device, use_kinematics_tree=False
    )
    tgt_verts = tgt_mesh["vertices"]
    bb_min = tgt_verts.min(dim=0)[0].tolist()
    bb_max = tgt_verts.max(dim=0)[0].tolist()
    # Fixed full-3D camera bounds so both GT and optimization share the same FOV.
    cam_bounds = {"min": bb_min, "max": bb_max}

    tgt_out = renderer.render_part_tensor_multimodal(
        tgt_part, template_organ_array=tgt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
        device=device, focus_plant=True, use_kinematics_tree=False,
        fixed_camera_bounds=cam_bounds, return_depth=True, return_mask=True,
        return_organ_masks=False, return_raw_depth=True,
    )

    init_arr = PlantOrganArray.from_xml_file(os.path.join(repo_root, init_rel))
    init_part = init_arr.to_part_tensor(device=device)
    init_rgb = renderer.render_part_tensor(
        init_part, template_organ_array=init_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
        device=device, focus_plant=True, use_kinematics_tree=False, differentiable=False,
        fixed_camera_bounds=cam_bounds,
    )
    return {
        "title": None,
        "tgt_arr": tgt_arr, "tgt_part": tgt_part, "tgt_rgb": tgt_out["rgb"],
        "tgt_depth": tgt_out["depth"], "tgt_raw_depth": tgt_out["raw_depth"],
        "tgt_mask": tgt_out["mask"], "tgt_np": tgt_out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1),
        "helios_np": helios_np,
        "init_arr": init_arr, "init_np": init_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1),
        "cam_bounds": cam_bounds,
    }


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def figure_3_direct_opt_multi_dap(pairs, renderer, device, assets_dir, steps=35):
    print("Generating Figure 3: Direct Optimization Multi-DAP Panel...")
    fig, axes = plt.subplots(3, 7, figsize=(22, 12))
    plt.subplots_adjust(wspace=0.12, hspace=0.28)
    metrics = {"dap": [], "init_ssim": [], "init_iou": [], "opt_ssim": [], "opt_iou": []}

    for row, spec in enumerate(pairs):
        init_ssim = float(masked_ssim(_to_tensor(spec["init_np"], device), _to_tensor(spec["tgt_np"], device)).item())
        init_iou = float(foreground_iou(_to_tensor(spec["init_np"], device), _to_tensor(spec["tgt_np"], device)).item())

        init_depth = renderer.render_part_tensor_multimodal(
            spec["init_arr"].to_part_tensor(device=device), template_organ_array=spec["init_arr"],
            camera_height=5.0, elevation_deg=ELEVATION_DEG, device=device, focus_plant=True, use_kinematics_tree=False,
            fixed_camera_bounds=spec["cam_bounds"], return_depth=True, return_mask=False,
            return_organ_masks=False,
        )["depth"].cpu().numpy()

        opt_rgb, opt_depth, m = run_direct_opt(
            spec["init_arr"], spec["tgt_rgb"], spec["tgt_raw_depth"], spec["tgt_mask"],
            spec["cam_bounds"], renderer, device, mode="rgb_depth", steps=steps, lr=0.04,
        )

        # Col 0: Original Helios C++ render
        if spec.get("helios_np") is not None:
            axes[row, 0].imshow(spec["helios_np"])
            axes[row, 0].set_title(f"{spec['title']}\nHelios C++ Original", fontsize=11, fontweight="bold")
        else:
            axes[row, 0].text(0.5, 0.5, "No Helios image", ha='center', va='center', color='red', fontsize=10, transform=axes[row, 0].transAxes)
            axes[row, 0].set_title(f"{spec['title']}\nHelios C++", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(spec["tgt_np"])
        axes[row, 1].set_title(f"Ground Truth RGB", fontsize=11, fontweight="bold")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(_depth_colormap(spec["tgt_depth"].cpu().numpy()))
        axes[row, 2].set_title("Ground Truth Depth\n(closer = brighter)", fontsize=11, fontweight="bold")
        axes[row, 2].axis("off")
        axes[row, 3].imshow(spec["init_np"])
        axes[row, 3].set_title(f"Initial Template Seed\nmSSIM: {init_ssim:.3f} | IoU: {init_iou:.2f}", fontsize=11)
        axes[row, 3].axis("off")
        axes[row, 4].imshow(_depth_colormap(init_depth))
        axes[row, 4].set_title("Initial Seed Depth", fontsize=11)
        axes[row, 4].axis("off")
        axes[row, 5].imshow(opt_rgb)
        axes[row, 5].set_title(f"16D RGB+Depth Opt\nmSSIM: {m['ssim']:.3f} | IoU: {m['iou']:.2f}", fontsize=11, color="navy", fontweight="bold")
        axes[row, 5].axis("off")
        axes[row, 6].imshow(_depth_colormap(opt_depth))
        axes[row, 6].set_title("Optimized Depth", fontsize=11, color="navy", fontweight="bold")
        axes[row, 6].axis("off")

        metrics["dap"].append(spec["title"])
        metrics["init_ssim"].append(init_ssim)
        metrics["init_iou"].append(init_iou)
        metrics["opt_ssim"].append(m["ssim"])
        metrics["opt_iou"].append(m["iou"])

    out = os.path.join(assets_dir, "fig3_direct_opt_multi_dap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return metrics


def figure_4_strategy_ablation(pairs, renderer, device, assets_dir, steps=35):
    print("Generating Figure 4: Direct-Optimization Strategy Ablation...")
    fig, axes = plt.subplots(3, 5, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.12, hspace=0.25)
    metrics = {"dap": [], "rgb_only_ssim": [], "rgb_only_iou": [], "rgb_depth_ssim": [], "rgb_depth_iou": []}

    for row, spec in enumerate(pairs):
        axes[row, 0].imshow(spec["tgt_np"])
        axes[row, 0].set_title(f"{spec['title']} (Top View)\nGround Truth RGB", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(_depth_colormap(spec["tgt_depth"].cpu().numpy()))
        axes[row, 1].set_title("Ground Truth Depth", fontsize=11, fontweight="bold")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(spec["init_np"])
        init_ssim = float(masked_ssim(_to_tensor(spec["init_np"], device), _to_tensor(spec["tgt_np"], device)).item())
        axes[row, 2].set_title(f"Initial Seed\nmSSIM: {init_ssim:.3f}", fontsize=11)
        axes[row, 2].axis("off")

        rgb_only, _, m1 = run_direct_opt(
            spec["init_arr"], spec["tgt_rgb"], spec["tgt_raw_depth"], spec["tgt_mask"],
            spec["cam_bounds"], renderer, device, mode="rgb", steps=steps, lr=0.04,
        )
        axes[row, 3].imshow(rgb_only)
        axes[row, 3].set_title(f"RGB Only\nmSSIM: {m1['ssim']:.3f} | IoU: {m1['iou']:.2f}", fontsize=11, color="crimson")
        axes[row, 3].axis("off")

        rgb_depth, _, m2 = run_direct_opt(
            spec["init_arr"], spec["tgt_rgb"], spec["tgt_raw_depth"], spec["tgt_mask"],
            spec["cam_bounds"], renderer, device, mode="rgb_depth", steps=steps, lr=0.04,
        )
        axes[row, 4].imshow(rgb_depth)
        axes[row, 4].set_title(f"RGB + Depth\nmSSIM: {m2['ssim']:.3f} | IoU: {m2['iou']:.2f}", fontsize=11, color="navy", fontweight="bold")
        axes[row, 4].axis("off")

        metrics["dap"].append(spec["title"])
        metrics["rgb_only_ssim"].append(m1["ssim"])
        metrics["rgb_only_iou"].append(m1["iou"])
        metrics["rgb_depth_ssim"].append(m2["ssim"])
        metrics["rgb_depth_iou"].append(m2["iou"])

    out = os.path.join(assets_dir, "fig4_direct_opt_strategies.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return metrics


def figure_5_chamfer_ablation(pairs, renderer, device, assets_dir, steps=35):
    print("Generating Figure 5: Chamfer Leaf-Base Ablation...")
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    metrics = {"dap": [], "no_chamfer_ssim": [], "no_chamfer_iou": [], "chamfer_ssim": [], "chamfer_iou": []}

    for row, spec in enumerate(pairs):
        axes[row, 0].imshow(spec["tgt_np"])
        axes[row, 0].set_title(f"{spec['title']} (Top View)\nGround Truth Target", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        no_cf, _, m1 = run_direct_opt(
            spec["init_arr"], spec["tgt_rgb"], spec["tgt_raw_depth"], spec["tgt_mask"],
            spec["cam_bounds"], renderer, device, mode="rgb_depth", steps=steps, lr=0.04,
        )
        axes[row, 1].imshow(no_cf)
        axes[row, 1].set_title(f"RGB+Depth\nmSSIM: {m1['ssim']:.3f}", fontsize=11, color="purple")
        axes[row, 1].axis("off")

        # Target leaf bases for Chamfer
        tgt_leaf_mask = spec["tgt_part"][:, P_COL_ORGAN_TYPE].long() == ORGAN_LEAF
        tgt_leaf_bases = spec["tgt_part"][tgt_leaf_mask, P_COL_BASE_X:P_COL_BASE_Z + 1].to(device)

        with_cf, _, m2 = run_direct_opt(
            spec["init_arr"], spec["tgt_rgb"], spec["tgt_raw_depth"], spec["tgt_mask"],
            spec["cam_bounds"], renderer, device, mode="rgb_depth_chamfer", steps=steps, lr=0.04,
            target_leaf_bases=tgt_leaf_bases, chamfer_weight=0.5,
        )
        axes[row, 2].imshow(with_cf)
        axes[row, 2].set_title(f"+ Chamfer\nmSSIM: {m2['ssim']:.3f}", fontsize=11, color="darkgreen", fontweight="bold")
        axes[row, 2].axis("off")

        diff = np.abs(with_cf - no_cf).mean(axis=-1)
        im = axes[row, 3].imshow(diff, cmap="inferno", vmin=0.0, vmax=0.1)
        axes[row, 3].set_title("Difference Map\n(Chamfer vs None)", fontsize=11, color="gold")
        axes[row, 3].axis("off")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

        metrics["dap"].append(spec["title"])
        metrics["no_chamfer_ssim"].append(m1["ssim"])
        metrics["no_chamfer_iou"].append(m1["iou"])
        metrics["chamfer_ssim"].append(m2["ssim"])
        metrics["chamfer_iou"].append(m2["iou"])

    out = os.path.join(assets_dir, "fig5_direct_opt_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return metrics


def figure_6_real_loss_curves(pairs, renderer, device, assets_dir, steps=35):
    print("Generating Figure 6: Real Loss Convergence Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    plt.subplots_adjust(wspace=0.25)

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for i, spec in enumerate(pairs):
        _, _, m = run_direct_opt(
            spec["init_arr"], spec["tgt_rgb"], spec["tgt_raw_depth"], spec["tgt_mask"],
            spec["cam_bounds"], renderer, device, mode="rgb_depth", steps=steps, lr=0.04,
        )
        h = m["history"]
        ax1.plot(h["time"], h["rgb_loss"], "-", color=colors[i], linewidth=2.0, label=f"{spec['title']} RGB")
        ax1.plot(h["time"], h["depth_loss"], "--", color=colors[i], linewidth=1.8, alpha=0.7)

    ax1.set_xlabel("Wall-clock time (s)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title("16D Direct Optimization: Real RGB + Depth Loss vs Time", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(fontsize=9)

    for i, spec in enumerate(pairs):
        _, _, m = run_direct_opt(
            spec["init_arr"], spec["tgt_rgb"], spec["tgt_raw_depth"], spec["tgt_mask"],
            spec["cam_bounds"], renderer, device, mode="rgb_depth", steps=steps, lr=0.04,
        )
        ax2.plot(range(1, steps + 1), m["history"]["ssim"], "-", color=colors[i], linewidth=2.2, label=f"{spec['title']}")

    ax2.set_xlabel("Optimization step", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Masked SSIM", fontsize=11, fontweight="bold")
    ax2.set_title("16D Direct Optimization: Real mSSIM Convergence", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(fontsize=10)

    out = os.path.join(assets_dir, "fig6_real_loss_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def figure_7_canopy_metrics(fig3_metrics, fig4_metrics, assets_dir):
    print("Generating Figure 7: Botanical 3D Canopy Metrics...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.28)
    labels = fig3_metrics["dap"]
    x = np.arange(len(labels))
    w = 0.35

    axes[0].bar(x - w/2, fig3_metrics["init_iou"], w, label="Initial Seed", color="#98df8a")
    axes[0].bar(x + w/2, fig3_metrics["opt_iou"], w, label="16D RGB+Depth Opt", color="#2ca02c")
    axes[0].set_ylabel("Canopy Silhouette IoU", fontsize=11, fontweight="bold")
    axes[0].set_title("2D Projected Canopy Coverage (IoU)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontweight="bold")
    axes[0].set_ylim(0.0, 1.0); axes[0].grid(True, linestyle="--", alpha=0.3); axes[0].legend(fontsize=9)

    axes[1].bar(x - w/2, fig3_metrics["init_ssim"], w, label="Initial Seed", color="#aec7e8")
    axes[1].bar(x + w/2, fig3_metrics["opt_ssim"], w, label="16D RGB+Depth Opt", color="#1f77b4")
    axes[1].set_ylabel("Masked SSIM", fontsize=11, fontweight="bold")
    axes[1].set_title("Visual Similarity (mSSIM)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontweight="bold")
    axes[1].set_ylim(0.0, 0.85); axes[1].grid(True, linestyle="--", alpha=0.3); axes[1].legend(fontsize=9)

    axes[2].bar(x - w/2, fig4_metrics["rgb_only_ssim"], w, label="RGB Only", color="#ffbb78")
    axes[2].bar(x + w/2, fig4_metrics["rgb_depth_ssim"], w, label="RGB + Depth", color="#ff7f0e")
    axes[2].set_ylabel("Masked SSIM", fontsize=11, fontweight="bold")
    axes[2].set_title("Depth Supervision Ablation", fontsize=12, fontweight="bold")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontweight="bold")
    axes[2].set_ylim(0.0, 0.85); axes[2].grid(True, linestyle="--", alpha=0.3); axes[2].legend(fontsize=9)

    out = os.path.join(assets_dir, "fig7_botanical_3d_canopy_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_dir", default="docs/results/assets")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--dap_pairs", nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.assets_dir, exist_ok=True)

    dap_specs = [
        ("DAP 10 (Seedling)",
         "dataset/helios_data/cowpea/cowpea_bush_dap010_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
         "dataset/helios_data/cowpea/cowpea_bush_dap010_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
        ("DAP 50 (Branching)",
         "dataset/helios_data/cowpea/cowpea_bush_dap050_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
         "dataset/helios_data/cowpea/cowpea_bush_dap050_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
        ("DAP 90 (Mature)",
         "dataset/helios_data/cowpea/cowpea_bush_dap090_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
         "dataset/helios_data/cowpea/cowpea_bush_dap090_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
    ]

    renderer = HeliosPyTorchRenderer(image_size=256).to(device)
    pairs = []
    for title, tgt, init in dap_specs:
        spec = _load_target_init_pair(renderer, device, tgt, init)
        spec["title"] = title
        pairs.append(spec)

    t0 = time.time()
    m3 = figure_3_direct_opt_multi_dap(pairs, renderer, device, args.assets_dir, steps=args.steps)
    m4 = figure_4_strategy_ablation(pairs, renderer, device, args.assets_dir, steps=args.steps)
    m5 = figure_5_chamfer_ablation(pairs, renderer, device, args.assets_dir, steps=args.steps)
    figure_6_real_loss_curves(pairs, renderer, device, args.assets_dir, steps=args.steps)
    figure_7_canopy_metrics(m3, m4, args.assets_dir)
    print(f"\nAll figures generated in {time.time()-t0:.1f}s")

    import json
    summary_path = os.path.join(args.assets_dir, "part_report_metrics.json")
    with open(summary_path, "w") as f:
        json.dump({"fig3": m3, "fig4": m4, "fig5": m5}, f, indent=2)
    print(f"Saved metrics summary: {summary_path}")


if __name__ == "__main__":
    main()
