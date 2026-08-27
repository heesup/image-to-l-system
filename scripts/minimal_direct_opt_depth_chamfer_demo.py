#!/usr/bin/env python3
"""
Minimal Direct Optimization Demo: 3D Chamfer Pulls a Leaf Outward

A focused, self-contained example that verifies the 3D Chamfer gradient on the
simplest possible plant template: one internode + one petiole + one leaf.

Demonstrates:
  - The initial petiole pitch is 10° (leaf nearly upright/flat).
  - The target petiole pitch is 60°.
  - Direct optimization of only the petiole pitch via the 3D vertex Chamfer
    distance rotates the leaf outward until it matches the target location.

Outputs:
  - docs/results/assets/minimal_direct_opt_depth_chamfer_demo.png
  - docs/results/assets/minimal_direct_opt_depth_chamfer_demo.json
"""

import os
import sys
import json
import math
import numpy as np
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF,
    T_COL_BASE_X, T_COL_BASE_Y, T_COL_BASE_Z,
    T_COL_ORGAN_TYPE, T_COL_LENGTH, T_COL_RADIUS,
    T_COL_SCALE, T_COL_PITCH, T_COL_YAW, T_COL_ROLL,
    T_COL_EXISTENCE,
)
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.eval.metrics import affine_invariant_depth_loss

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
ASSETS_DIR = os.path.join(REPO_ROOT, "docs", "results", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
FIG_PATH = os.path.join(ASSETS_DIR, "minimal_direct_opt_depth_chamfer_demo.png")
JSON_PATH = os.path.join(ASSETS_DIR, "minimal_direct_opt_depth_chamfer_demo.json")

ELEVATION_TOP = 89.88
ELEVATION_SIDE = 20.0
AZIMUTH_SIDE = 0.0
ELEVATION_OBLIQUE = 45.0
AZIMUTH_OBLIQUE = 45.0
CAM_HEIGHT = 4.0
IMG_SIZE = 256

DEFAULT_INTERNODE_RAD = 0.015
DEFAULT_PETIOLE_RAD = 0.008


# -----------------------------------------------------------------------------
# Minimal 3-organ typed plant factory
# -----------------------------------------------------------------------------
def make_minimal_plant(
    internode_len: float = 0.25,
    petiole_pitch_deg: float = 60.0,
    petiole_len: float = 0.18,
    leaf_scale: float = 0.22,
    leaf_pitch_deg: float = 15.0,
    leaf_yaw_deg: float = 10.0,
    leaf_roll_deg: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> PlantOrganArray:
    """
    Builds a typed (3, 40) PlantOrganArray with one internode, one petiole,
    and one leaf.

    NOTE: The current HeliosPyTorchGeometry builder derives petiole/leaf base
    positions from the internode length and petiole pitch/yaw, not from the
    T_COL_BASE_* columns. Therefore the demo optimizes the geometry-relevant
    continuous parameters (internode_len, petiole_pitch) directly.
    """
    t = torch.zeros((3, 40), dtype=torch.float32, device=device)

    # Row 0: internode (vertical stem segment).
    # Tiny non-zero pitch avoids a degenerate rotation when the internode axis
    # would otherwise be exactly +z.
    t[0, T_COL_ORGAN_TYPE] = ORGAN_INTERNODE
    t[0, T_COL_BASE_X:T_COL_BASE_Z + 1] = torch.tensor([0.0, 0.0, 0.0], device=device)
    t[0, T_COL_LENGTH] = float(internode_len)
    t[0, T_COL_RADIUS] = 0.02
    t[0, T_COL_SCALE] = 1.0
    t[0, T_COL_PITCH] = 1.0
    t[0, T_COL_YAW] = 0.0
    t[0, T_COL_ROLL] = 0.0
    t[0, T_COL_EXISTENCE] = 1.0

    # Row 1: petiole.
    t[1, T_COL_ORGAN_TYPE] = ORGAN_PETIOLE
    t[1, T_COL_BASE_X:T_COL_BASE_Z + 1] = torch.tensor([0.0, 0.0, 0.0], device=device)
    t[1, T_COL_LENGTH] = float(petiole_len)
    t[1, T_COL_RADIUS] = 0.01
    t[1, T_COL_SCALE] = 1.0
    t[1, T_COL_PITCH] = float(petiole_pitch_deg)
    t[1, T_COL_YAW] = 0.0
    t[1, T_COL_ROLL] = 0.0
    t[1, T_COL_EXISTENCE] = 1.0

    # Row 2: leaf.
    t[2, T_COL_ORGAN_TYPE] = ORGAN_LEAF
    t[2, T_COL_BASE_X:T_COL_BASE_Z + 1] = torch.tensor([0.0, 0.0, 0.0], device=device)
    t[2, T_COL_LENGTH] = float(leaf_scale)
    t[2, T_COL_RADIUS] = float(leaf_scale) * 0.5
    t[2, T_COL_SCALE] = 1.0
    t[2, T_COL_PITCH] = float(leaf_pitch_deg)
    t[2, T_COL_YAW] = float(leaf_yaw_deg)
    t[2, T_COL_ROLL] = float(leaf_roll_deg)
    t[2, T_COL_EXISTENCE] = 1.0

    return PlantOrganArray(t)


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------
def get_fixed_bounds(arr: PlantOrganArray, device: torch.device) -> Dict[str, Any]:
    """Compute static camera bounds from the target plant."""
    mesh = HeliosPlantGeometryBuilder().build_mesh_from_part_tensor(arr.to_part_tensor(device=device), device=device)
    verts = mesh["vertices"]
    bb_min = verts.min(dim=0)[0].tolist()
    bb_max = verts.max(dim=0)[0].tolist()
    return {"min": bb_min, "max": bb_max}


def render_plant(
    renderer: HeliosPyTorchRenderer,
    arr: PlantOrganArray,
    device: torch.device,
    elevation_deg: float = ELEVATION_TOP,
    azimuth_deg: float = 0.0,
    fixed_bounds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render RGB+depth (top or oblique) for a typed organ array."""
    mesh = renderer.geo_builder.build_mesh_from_part_tensor(arr.to_part_tensor(device=device), device=device)
    rgbd = renderer.forward(
        mesh,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        camera_height=CAM_HEIGHT,
        focus_plant=True,
        include_depth=True,
        fixed_camera_bounds=fixed_bounds,
    )
    rgb = rgbd[:3]
    depth = rgbd[3]
    return {
        "rgb": rgb,
        "depth": depth,
        "mask": (depth > 1e-4).float(),
        "rgb_np": rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1),
        "depth_np": depth.detach().cpu().numpy(),
        "verts": mesh["vertices"],
        "faces": mesh["faces"],
        "organ_types": mesh.get("organ_types"),
    }


def get_leaf_centroid(verts: np.ndarray, organ_types: np.ndarray) -> Optional[np.ndarray]:
    """Return the centroid of leaf vertices (organ type == 2) in world coordinates."""
    if organ_types is None:
        return None
    leaf_mask = organ_types == 2
    if leaf_mask.sum() == 0:
        return None
    return verts[leaf_mask].mean(axis=0)


def get_leaf_screen_centroid(
    renderer: HeliosPyTorchRenderer,
    arr: PlantOrganArray,
    device: torch.device,
    elevation_deg: float,
    azimuth_deg: float,
    fixed_bounds: Optional[Dict[str, Any]],
) -> Optional[Tuple[float, float]]:
    """Return (row, col) centroid of leaf pixels in the rendered organ-type buffer."""
    mesh = renderer.geo_builder.build_mesh_from_part_tensor(arr.to_part_tensor(device=device), device=device)
    type_buf = renderer.render_organ_type_buffer(
        mesh,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        camera_height=CAM_HEIGHT,
        focus_plant=True,
        fixed_camera_bounds=fixed_bounds,
        image_size=IMG_SIZE,
    )
    type_buf_np = type_buf.detach().cpu().numpy()
    leaf_mask = type_buf_np == 2
    if leaf_mask.sum() == 0:
        return None
    rows, cols = np.where(leaf_mask)
    return (float(rows.mean()), float(cols.mean()))


def plot_3d_side_overlay(ax, target_verts: np.ndarray, current_verts: np.ndarray,
                         target_organ_types: Optional[np.ndarray] = None,
                         current_organ_types: Optional[np.ndarray] = None,
                         xlim: Optional[Tuple[float, float]] = None,
                         zlim: Optional[Tuple[float, float]] = None,
                         draw_gradient: bool = False):
    """
    Plot target (red) and current (blue) vertex point clouds in the x-z plane.
    Optionally draw an arrow from the current leaf centroid toward the target leaf
    centroid to visualize the Chamfer gradient direction.
    """
    ax.scatter(target_verts[:, 0] * 100.0, target_verts[:, 2] * 100.0,
               c="red", s=3, alpha=0.28, label="Target", rasterized=True)
    ax.scatter(current_verts[:, 0] * 100.0, current_verts[:, 2] * 100.0,
               c="blue", s=3, alpha=0.60, label="Current", rasterized=True)
    ax.set_xlabel("X (cm)", fontsize=9)
    ax.set_ylabel("Z (cm)", fontsize=9)
    ax.tick_params(labelsize=8)
    if xlim is not None:
        ax.set_xlim(xlim)
    if zlim is not None:
        ax.set_ylim(zlim)

    if draw_gradient:
        c_cent = get_leaf_centroid(current_verts, current_organ_types)
        t_cent = get_leaf_centroid(target_verts, target_organ_types)
        if c_cent is not None and t_cent is not None:
            dx = (t_cent[0] - c_cent[0]) * 100.0
            dz = (t_cent[2] - c_cent[2]) * 100.0
            ax.annotate(
                "",
                xy=(c_cent[0] * 100.0 + dx * 0.95, c_cent[2] * 100.0 + dz * 0.95),
                xytext=(c_cent[0] * 100.0, c_cent[2] * 100.0),
                arrowprops=dict(arrowstyle="-|>", color="darkgreen", lw=2.2,
                                connectionstyle="arc3,rad=0.05"),
            )
            ax.text(
                0.98, 0.98, "Chamfer\ngradient",
                transform=ax.transAxes,
                fontsize=8, color="darkgreen", fontweight="bold",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="darkgreen", alpha=0.85),
            )

    ax.legend(fontsize=8, loc="upper right")


def compute_depth_error_mm(pred_depth_np: np.ndarray, target_depth_np: np.ndarray) -> np.ndarray:
    """Per-pixel absolute depth error in mm within the foreground union."""
    err = np.abs(pred_depth_np - target_depth_np) * 1000.0  # mm
    mask = (pred_depth_np > 1e-4) | (target_depth_np > 1e-4)
    err[~mask] = np.nan
    return err


# -----------------------------------------------------------------------------
# Loss helpers
# -----------------------------------------------------------------------------
def l1_depth_loss(pred_depth: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
    """Raw L1 depth loss over foreground union (meters)."""
    mask = (pred_depth > 1e-4) | (target_depth > 1e-4)
    if mask.sum() < 1:
        return torch.tensor(0.0, device=pred_depth.device, dtype=pred_depth.dtype)
    return F.l1_loss(pred_depth[mask], target_depth[mask])


def chamfer_distance_mm(pred_verts: torch.Tensor, target_verts: torch.Tensor) -> torch.Tensor:
    """
    Bidirectional Chamfer distance in millimeters. For the small 3-organ meshes
    we can compute the full distance matrix deterministically.
    """
    if pred_verts.shape[0] == 0 or target_verts.shape[0] == 0:
        return torch.tensor(999.0, device=pred_verts.device, dtype=pred_verts.dtype)
    n_max = 4000
    p = pred_verts[:: max(1, pred_verts.shape[0] // n_max)][:n_max]
    g = target_verts[:: max(1, target_verts.shape[0] // n_max)][:n_max]
    dist = torch.cdist(p, g)  # (N_p, N_g) meters
    d_p2g = dist.min(dim=1)[0].mean()
    d_g2p = dist.min(dim=0)[0].mean()
    return (d_p2g + d_g2p) * 0.5 * 1000.0  # mm


# -----------------------------------------------------------------------------
# Generic direct optimizer
# -----------------------------------------------------------------------------
def optimize(
    init_arr: PlantOrganArray,
    target_arr: PlantOrganArray,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    param_spec: Dict[str, bool],
    loss_spec: Dict[str, float],
    steps: int = 150,
    lr: float = 0.01,
    petiole_pitch_lr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Optimize a subset of typed-plant parameters to match the target.

    param_spec: flags for `internode_len` and `petiole_pitch`.
    loss_spec: weights for `l1_depth`, `affine_depth`, `chamfer`.
    """
    target_top = render_plant(renderer, target_arr, device, elevation_deg=ELEVATION_TOP)
    fixed_bounds = get_fixed_bounds(target_arr, device)
    target_oblique = render_plant(
        renderer, target_arr, device,
        elevation_deg=ELEVATION_OBLIQUE, azimuth_deg=AZIMUTH_OBLIQUE,
        fixed_bounds=fixed_bounds,
    )
    target_verts = target_top["verts"].detach()

    base_t = init_arr.tensor.detach().clone()

    # Build per-parameter optimization variables.
    opt_params = []
    param_meta = []  # (row, col, init_value, name)

    if param_spec.get("internode_len"):
        v = torch.zeros(1, device=device, requires_grad=True)
        opt_params.append(v)
        param_meta.append((0, T_COL_LENGTH, base_t[0, T_COL_LENGTH].item(), "internode_len"))

    if param_spec.get("petiole_pitch"):
        v = torch.zeros(1, device=device, requires_grad=True)
        opt_params.append(v)
        param_meta.append((1, T_COL_PITCH, base_t[1, T_COL_PITCH].item(), "petiole_pitch"))

    # Different LRs for length (meters) vs angle (degrees).
    if len(param_meta) == 2:
        lrs = [lr]
        if petiole_pitch_lr is not None:
            lrs = [lr, petiole_pitch_lr]
        else:
            lrs = [lr, lr * 40.0]
        param_groups = [{"params": [p], "lr": lr_i} for p, lr_i in zip(opt_params, lrs)]
    else:
        param_groups = [{"params": opt_params, "lr": lr}]

    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)

    history = {
        "step": [],
        "total_loss": [],
        "l1_depth_mm": [],
        "affine_depth": [],
        "chamfer_mm": [],
        "internode_len": [],
        "petiole_pitch": [],
    }

    best_loss = float("inf")
    best_snap = None

    for s in range(steps):
        optimizer.zero_grad()

        t_eval = base_t.clone()
        for v, (row, col, init_val, name) in zip(opt_params, param_meta):
            raw = init_val + v
            if name == "internode_len":
                t_eval[row, col] = raw.clamp(min=1e-3)
            elif name == "petiole_pitch":
                t_eval[row, col] = raw.clamp(min=0.0, max=89.0)
            else:
                t_eval[row, col] = raw

        arr_eval = PlantOrganArray(t_eval)
        top = render_plant(renderer, arr_eval, device, elevation_deg=ELEVATION_TOP, fixed_bounds=fixed_bounds)
        oblique = render_plant(
            renderer, arr_eval, device,
            elevation_deg=ELEVATION_OBLIQUE, azimuth_deg=AZIMUTH_OBLIQUE,
            fixed_bounds=fixed_bounds,
        )

        pred_depth = top["depth"]
        target_depth = target_top["depth"]
        pred_verts = top["verts"]

        loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        l1_d = torch.tensor(0.0, device=device, dtype=torch.float32)
        aff_d = torch.tensor(0.0, device=device, dtype=torch.float32)
        chamf = torch.tensor(0.0, device=device, dtype=torch.float32)

        if loss_spec.get("l1_depth", 0.0) > 0:
            l1_d = l1_depth_loss(pred_depth, target_depth)
            loss = loss + loss_spec["l1_depth"] * l1_d

        if loss_spec.get("affine_depth", 0.0) > 0:
            mask = (pred_depth > 1e-4) | (target_depth > 1e-4)
            aff_d = affine_invariant_depth_loss(pred_depth, target_depth, mask=mask)
            loss = loss + loss_spec["affine_depth"] * aff_d

        if loss_spec.get("chamfer", 0.0) > 0:
            chamf = chamfer_distance_mm(pred_verts, target_verts)
            loss = loss + loss_spec["chamfer"] * (chamf / 1000.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            l1_mm = l1_d.item() * 1000.0
            chamf_val = chamf.item()
            history["step"].append(s)
            history["total_loss"].append(loss.item())
            history["l1_depth_mm"].append(l1_mm)
            history["affine_depth"].append(aff_d.item())
            history["chamfer_mm"].append(chamf_val)
            history["internode_len"].append(t_eval[0, T_COL_LENGTH].item())
            history["petiole_pitch"].append(t_eval[1, T_COL_PITCH].item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_snap = {
                    "step": s,
                    "rgb_np": top["rgb_np"],
                    "depth_np": top["depth_np"],
                    "oblique_rgb_np": oblique["rgb_np"],
                    "oblique_depth_np": oblique["depth_np"],
                    "verts": pred_verts.detach().cpu().numpy(),
                    "l1_depth_mm": l1_mm,
                    "affine_depth": aff_d.item(),
                    "chamfer_mm": chamf_val,
                }

    # Final render using best parameters.
    t_final = base_t.clone()
    for v, (row, col, init_val, name) in zip(opt_params, param_meta):
        raw = init_val + v.detach()
        if name == "internode_len":
            t_final[row, col] = raw.clamp(min=1e-3)
        elif name == "petiole_pitch":
            t_final[row, col] = raw.clamp(min=0.0, max=89.0)
        else:
            t_final[row, col] = raw
    final_arr = PlantOrganArray(t_final)

    final_top = render_plant(renderer, final_arr, device, elevation_deg=ELEVATION_TOP, fixed_bounds=fixed_bounds)
    final_oblique = render_plant(
        renderer, final_arr, device,
        elevation_deg=ELEVATION_OBLIQUE, azimuth_deg=AZIMUTH_OBLIQUE,
        fixed_bounds=fixed_bounds,
    )

    return {
        "history": history,
        "final_arr": final_arr,
        "final_top": final_top,
        "final_oblique": final_oblique,
        "best_snap": best_snap,
        "param_meta": param_meta,
        "fixed_bounds": fixed_bounds,
        "target_top": target_top,
        "target_oblique": target_oblique,
    }


# -----------------------------------------------------------------------------
# Figure generation
# -----------------------------------------------------------------------------
def _depth_colormap(depth_np: np.ndarray, vmax_cm: Optional[float] = None) -> Tuple[np.ndarray, float]:
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#111111")
    d_cm = np.ma.masked_where(depth_np <= 1e-4, depth_np * 100.0)
    if vmax_cm is None:
        vmax_cm = max(5.0, float(np.ceil(d_cm.max())))
    rgb = cmap((d_cm.filled(np.nan) / vmax_cm).clip(0, 1))[:, :, :3]
    rgb[d_cm.mask] = 0.0
    return rgb, vmax_cm





def generate_figure(results: Dict[str, Dict[str, Any]], out_path: str):
    """Compact, publication-friendly figure for the B2 leaf-outward experiment."""
    key = "chamfer_lateral"
    r = results[key]
    h = r["history"]

    # Shared canopy-height color scale.
    all_depths = [
        r["target_top"]["depth_np"],
        r["init_top"]["depth_np"],
        r["final_top"]["depth_np"],
    ]
    vmax_cm = max(5.0, float(np.ceil(max(d.max() for d in all_depths) * 100.0)))

    # Compact 2x6 layout: five image columns + one full-height metrics column.
    fig = plt.figure(figsize=(22, 10), dpi=180)
    outer = gridspec.GridSpec(
        2, 6, figure=fig,
        hspace=0.30, wspace=0.22,
        left=0.04, right=0.965, top=0.92, bottom=0.08,
        width_ratios=[1, 1, 1, 1, 1, 1.15],
    )

    # Common side-view limits from the target.
    target_verts_np = r["target_top"]["verts"].detach().cpu().numpy()
    target_org_np = r["target_top"]["organ_types"].detach().cpu().numpy()
    xs = target_verts_np[:, 0] * 100.0
    zs = target_verts_np[:, 2] * 100.0
    x_margin = (xs.max() - xs.min()) * 0.15
    z_margin = (zs.max() - zs.min()) * 0.15
    side_xlim = (xs.min() - x_margin, xs.max() + x_margin)
    side_zlim = (max(0.0, zs.min() - z_margin), zs.max() + z_margin)

    def _title(ax, text, color="black"):
        ax.set_title(text, fontsize=10, color=color, fontweight="bold", pad=4)

    def _add_screen_arrow(ax, src_cent, dst_cent, color="darkgreen"):
        """Draw an arrow from src to dst in image (pixel) coordinates."""
        if src_cent is None or dst_cent is None:
            return
        dy = dst_cent[0] - src_cent[0]
        dx = dst_cent[1] - src_cent[1]
        pix_dist = math.hypot(dx, dy)
        if pix_dist < 5.0:
            return
        ax.annotate(
            "",
            xy=(src_cent[1] + dx * 0.95, src_cent[0] + dy * 0.95),
            xytext=(src_cent[1], src_cent[0]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                            connectionstyle="arc3,rad=0.05"),
        )
        ax.text(
            0.03, 0.97, "pull\noutward",
            transform=ax.transAxes,
            fontsize=7, color=color, fontweight="bold",
            ha="left", va="top",
        )

    # -------------------------------------------------------------------------
    # Row 0: targets and initial state
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(outer[0, 0])
    ax.imshow(r["target_top"]["rgb_np"])
    _title(ax, "Target RGB\n(90\u00b0 top-down)")
    ax.axis("off")

    ax_depth = fig.add_subplot(outer[0, 1])
    dimg, _ = _depth_colormap(r["target_top"]["depth_np"], vmax_cm)
    im_depth = ax_depth.imshow(dimg)
    _title(ax_depth, "Target Canopy\nHeight (cm)")
    ax_depth.axis("off")

    ax = fig.add_subplot(outer[0, 2])
    ax.imshow(r["target_side"]["rgb_np"])
    _title(ax, "Target Side View\n(20\u00b0 elevation)")
    ax.axis("off")

    ax = fig.add_subplot(outer[0, 3])
    ax.imshow(r["init_top"]["rgb_np"])
    _title(ax, "Initial RGB\n(90\u00b0)")
    ax.axis("off")

    ax = fig.add_subplot(outer[0, 4])
    ax.imshow(r["init_side"]["rgb_np"])
    _title(ax, "Initial Side View")
    ax.axis("off")
    _add_screen_arrow(ax, r["init_leaf_side_cent"], r["target_leaf_side_cent"])

    # -------------------------------------------------------------------------
    # Row 1: optimized state + 3D overlays + zoomed inset panel
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(outer[1, 0])
    ax.imshow(r["final_top"]["rgb_np"])
    _title(ax, "Optimized RGB\n(90\u00b0)", color="navy")
    ax.axis("off")

    ax = fig.add_subplot(outer[1, 1])
    ax.imshow(r["final_side"]["rgb_np"])
    _title(ax, "Optimized Side View", color="navy")
    ax.axis("off")
    _add_screen_arrow(ax, r["final_leaf_side_cent"], r["target_leaf_side_cent"])

    ax = fig.add_subplot(outer[1, 2])
    plot_3d_side_overlay(
        ax,
        r["target_top"]["verts"].detach().cpu().numpy(),
        r["init_top"]["verts"].detach().cpu().numpy(),
        target_organ_types=target_org_np,
        current_organ_types=r["init_top"]["organ_types"].detach().cpu().numpy(),
        xlim=side_xlim, zlim=side_zlim,
        draw_gradient=True,
    )
    _title(ax, "Initial 3D Overlay (x-z)")
    ax.set_aspect("equal", adjustable="box")

    ax = fig.add_subplot(outer[1, 3])
    plot_3d_side_overlay(
        ax,
        r["target_top"]["verts"].detach().cpu().numpy(),
        r["final_top"]["verts"].detach().cpu().numpy(),
        target_organ_types=target_org_np,
        current_organ_types=r["final_top"]["organ_types"].detach().cpu().numpy(),
        xlim=side_xlim, zlim=side_zlim,
        draw_gradient=True,
    )
    _title(ax, "Optimized 3D Overlay (x-z)", color="navy")
    ax.set_aspect("equal", adjustable="box")

    # Dedicated zoomed panel: leaf-tip region.
    ax_zoom = fig.add_subplot(outer[1, 4])
    plot_3d_side_overlay(
        ax_zoom,
        r["target_top"]["verts"].detach().cpu().numpy(),
        r["final_top"]["verts"].detach().cpu().numpy(),
        target_organ_types=target_org_np,
        current_organ_types=r["final_top"]["organ_types"].detach().cpu().numpy(),
        xlim=(side_xlim[0] * 0.35, side_xlim[1] * 0.35),
        zlim=(side_zlim[1] * 0.55, side_zlim[1] * 0.95),
        draw_gradient=False,
    )
    _title(ax_zoom, "Zoom: Leaf Tip", color="navy")
    ax_zoom.set_xlabel("X (cm)", fontsize=9)
    ax_zoom.set_ylabel("Z (cm)", fontsize=9)
    ax_zoom.tick_params(labelsize=8)
    ax_zoom.set_aspect("equal", adjustable="box")

    # -------------------------------------------------------------------------
    # Right-hand metrics + parameter trajectory + summary (spans both rows).
    # -------------------------------------------------------------------------
    gs_metrics = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[:, 5], hspace=0.25,
        height_ratios=[1, 1, 0.7]
    )

    ax_m = fig.add_subplot(gs_metrics[0, 0])
    steps = np.array(h["step"])
    ax_m.plot(steps, h["chamfer_mm"], label="Chamfer (mm)", color="#d62728", linewidth=2)
    ax_m.set_xlabel("Optimization step", fontsize=9)
    ax_m.set_ylabel("Chamfer distance (mm)", fontsize=9)
    ax_m.set_title("3D Chamfer Distance", fontsize=10, fontweight="bold")
    ax_m.grid(True, alpha=0.3)
    ax_m.legend(fontsize=8, loc="upper right")

    ax_p = fig.add_subplot(gs_metrics[1, 0])
    ax_p.plot(steps, h["petiole_pitch"], label="Petiole pitch (\u00b0)", color="#9467bd", linewidth=2)
    ax_p.axhline(60.0, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="Target 60\u00b0")
    ax_p.set_xlabel("Optimization step", fontsize=9)
    ax_p.set_ylabel("Petiole pitch (\u00b0)", fontsize=9)
    ax_p.set_title("Parameter Trajectory", fontsize=10, fontweight="bold")
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(fontsize=8, loc="lower right")

    ax_s = fig.add_subplot(gs_metrics[2, 0])
    ax_s.axis("off")
    summary_text = (
        f"Petiole pitch: {h['petiole_pitch'][0]:.1f}\u00b0 \u2192 {h['petiole_pitch'][-1]:.1f}\u00b0\n"
        f"3D Chamfer: {h['chamfer_mm'][0]:.1f} mm \u2192 {h['chamfer_mm'][-1]:.2f} mm\n"
        f"Optimization steps: {len(h['step'])} | LR: 0.5\u00b0/step"
    )
    ax_s.text(
        0.5, 0.5, summary_text,
        transform=ax_s.transAxes,
        fontsize=10, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#aaaaaa", alpha=0.9),
    )

    # Attach a colorbar to the target depth panel.
    sm_depth = plt.cm.ScalarMappable(cmap=plt.get_cmap("magma"), norm=plt.Normalize(vmin=0.0, vmax=vmax_cm))
    sm_depth.set_array([])
    cbar_depth = fig.colorbar(sm_depth, ax=ax_depth, fraction=0.046, pad=0.04, shrink=0.7)
    cbar_depth.set_label("Canopy Height (cm)", fontsize=9, fontweight="bold")
    cbar_depth.ax.tick_params(labelsize=8)

    # Caption
    caption = (
        "B2 \u2013 Chamfer gradient pulls a leaf outward. "
        "Template: one internode + one petiole + one leaf. "
        "The initial petiole pitch is 10\u00b0 (leaf nearly upright/flat); the target pitch is 60\u00b0. "
        "Only the petiole pitch is optimized via the 3D vertex Chamfer distance. "
        "Green arrows show the Chamfer gradient direction on rendered side views and 3D x-z overlays. "
        "The inset zooms into the leaf-tip region after convergence."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc", edgecolor="#999999", alpha=0.95))

    fig.suptitle(
        "Direct Optimization: 3D Chamfer Gradient Pulls a Leaf Outward",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE).to(device)

    # Target plant: one internode + one petiole + one leaf.
    target = make_minimal_plant(
        internode_len=0.25,
        petiole_pitch_deg=60.0,
        petiole_len=0.18,
        leaf_scale=0.22,
        leaf_pitch_deg=15.0,
        leaf_yaw_deg=10.0,
        device=device,
    )
    print(f"Target plant: {target.num_nodes} organs")

    # -------------------------------------------------------------------------
    # B2. Chamfer gradient pulls the leaf outward (lateral / 3D location)
    # -------------------------------------------------------------------------
    print("\nExperiment B2: Chamfer pulls leaf outward")
    init_b2 = make_minimal_plant(internode_len=0.25, petiole_pitch_deg=10.0, device=device)
    res_b2 = optimize(
        init_b2, target, renderer, device,
        param_spec={"petiole_pitch": True},
        loss_spec={"l1_depth": 0.0, "affine_depth": 0.0, "chamfer": 1.0},
        steps=200,
        lr=0.5,
    )
    h = res_b2["history"]
    print(f"  Initial petiole pitch: {h['petiole_pitch'][0]:.1f}°  ->  Final: {h['petiole_pitch'][-1]:.1f}°")
    print(f"  Initial Chamfer: {h['chamfer_mm'][0]:.2f} mm  ->  Final: {h['chamfer_mm'][-1]:.2f} mm")

    # -------------------------------------------------------------------------
    # Render initial/final side views for figure
    # -------------------------------------------------------------------------
    fixed_bounds = get_fixed_bounds(target, device)
    results = {"chamfer_lateral": res_b2}
    res_b2["init_top"] = render_plant(
        renderer, init_b2, device, elevation_deg=ELEVATION_TOP, fixed_bounds=fixed_bounds
    )
    res_b2["init_side"] = render_plant(
        renderer, init_b2, device,
        elevation_deg=ELEVATION_SIDE, azimuth_deg=AZIMUTH_SIDE,
        fixed_bounds=fixed_bounds,
    )
    res_b2["target_side"] = render_plant(
        renderer, target, device,
        elevation_deg=ELEVATION_SIDE, azimuth_deg=AZIMUTH_SIDE,
        fixed_bounds=fixed_bounds,
    )
    res_b2["final_side"] = render_plant(
        renderer, res_b2["final_arr"], device,
        elevation_deg=ELEVATION_SIDE, azimuth_deg=AZIMUTH_SIDE,
        fixed_bounds=fixed_bounds,
    )

    # Leaf centroids in the rendered side-view images for gradient arrows.
    res_b2["target_leaf_side_cent"] = get_leaf_screen_centroid(
        renderer, target, device,
        elevation_deg=ELEVATION_SIDE, azimuth_deg=AZIMUTH_SIDE,
        fixed_bounds=fixed_bounds,
    )
    res_b2["init_leaf_side_cent"] = get_leaf_screen_centroid(
        renderer, init_b2, device,
        elevation_deg=ELEVATION_SIDE, azimuth_deg=AZIMUTH_SIDE,
        fixed_bounds=fixed_bounds,
    )
    res_b2["final_leaf_side_cent"] = get_leaf_screen_centroid(
        renderer, res_b2["final_arr"], device,
        elevation_deg=ELEVATION_SIDE, azimuth_deg=AZIMUTH_SIDE,
        fixed_bounds=fixed_bounds,
    )

    # -------------------------------------------------------------------------
    # Build figure
    # -------------------------------------------------------------------------
    generate_figure(results, FIG_PATH)

    # -------------------------------------------------------------------------
    # Save JSON summary
    # -------------------------------------------------------------------------
    summary = {
        "experiment": "chamfer_lateral_leaf_outward",
        "description": "3D Chamfer-only direct optimization of petiole pitch on a 3-organ plant.",
        "initial": {
            "internode_len_m": float(h["internode_len"][0]),
            "petiole_pitch_deg": float(h["petiole_pitch"][0]),
            "petiole_len_m": float(init_b2.tensor[1, T_COL_LENGTH].item()),
            "leaf_scale_m": float(init_b2.tensor[2, T_COL_LENGTH].item()),
            "chamfer_mm": float(h["chamfer_mm"][0]) if h["chamfer_mm"][0] > 0 else None,
        },
        "final": {
            "internode_len_m": float(h["internode_len"][-1]),
            "petiole_pitch_deg": float(h["petiole_pitch"][-1]),
            "chamfer_mm": float(h["chamfer_mm"][-1]) if h["chamfer_mm"][-1] > 0 else None,
        },
        "target": {
            "internode_len_m": 0.25,
            "petiole_pitch_deg": 60.0,
        },
        "trajectory": {
            "step": [int(x) for x in h["step"]],
            "internode_len_cm": [float(v * 100.0) for v in h["internode_len"]],
            "petiole_pitch_deg": [float(v) for v in h["petiole_pitch"]],
            "chamfer_mm": [float(v) for v in h["chamfer_mm"]],
        },
    }

    with open(JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics JSON: {JSON_PATH}")
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
