"""
Differentiable Renderer Self-Consistency Evaluation for Hierarchical Flow Matching.

Validates 3D-to-2D consistency by:
  1. Generating 3D plant organ geometries via 2nd-order Heun ODE integration.
  2. Converting predicted part arrays to 3D surface meshes.
  3. Differentiably rasterizing top-down drone views via HeliosPyTorchRenderer (RGB + CHM Depth).
  4. Evaluating Silhouette IoU, Canopy Height Map (CHM) Depth MAE, and Peak Height Consistency
     against both 3D ground truth and input drone imagery.
  5. Generating diagnostic multi-view panels and logging to W&B.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plant_recon.dataset.part_array_dataset import (
    BASE_SCALE,
    SCALE_SCALE,
    CURV_SCALE,
    EMPTY_IDX,
    FM_OT_END,
    FM_BASE_START,
    P_COL_ORGAN_TYPE,
    P_COL_BASE_X,
    P_COL_BASE_Z,
    P_COL_ROT_0,
    P_COL_ROT_5,
    P_COL_SCALE_X,
    P_COL_SCALE_Z,
)
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from plant_recon.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def decode_predictions_to_part_tensor(
    pred_geom: torch.Tensor,       # (N, 13): [base(3)*20, rot6d(6), scale(3)*50, curv(1)/60]
    pred_cls: torch.Tensor,        # (N,): organ class integers 0..12
    slot_active: torch.Tensor,     # (N,): binary 0/1 activity mask
    device: torch.device,
) -> torch.Tensor:
    """Decodes normalized 13D geometry and classes into canonical (M_active, 14) part tensor."""
    active_mask = (slot_active > 0.5) & (pred_cls > EMPTY_IDX)
    active_indices = torch.nonzero(active_mask, as_tuple=True)[0]

    if len(active_indices) == 0:
        return torch.empty((0, 14), dtype=torch.float32, device=device)

    sub_geom = pred_geom[active_indices]  # (M_active, 13)
    sub_cls = pred_cls[active_indices]    # (M_active,)

    # Un-normalize physical units
    base_xyz = sub_geom[:, 0:3] / BASE_SCALE            # (M_active, 3) in meters
    rot6d = sub_geom[:, 3:9]                             # (M_active, 6)
    scale_xyz = sub_geom[:, 9:12] / SCALE_SCALE          # (M_active, 3) in meters
    curv_val = (sub_geom[:, 12] / CURV_SCALE).unsqueeze(-1)  # (M_active, 1) in deg/m
    cls_col = sub_cls.float().unsqueeze(-1)              # (M_active, 1)
    exist_col = torch.ones_like(cls_col)                 # (M_active, 1)

    # 14D part layout: [type(1), base(3), rot6d(6), scale(3), curv(1)]
    part_tensor = torch.cat([cls_col, base_xyz, rot6d, scale_xyz, curv_val], dim=-1)
    return part_tensor


def rot6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """Converts 6D continuous rotation representation to 3x3 rotation matrix via Gram-Schmidt."""
    u = rot_6d[:, :3]
    v = rot_6d[:, 3:6]
    u_norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp(min=1e-6)
    r1 = u / u_norm
    dot = (r1 * v).sum(dim=-1, keepdim=True)
    v_ortho = v - dot * r1
    v_norm = torch.linalg.norm(v_ortho, dim=-1, keepdim=True).clamp(min=1e-6)
    r2 = v_ortho / v_norm
    r3 = torch.linalg.cross(r1, r2)
    return torch.stack([r1, r2, r3], dim=-1)


def build_stem_parts_from_phytomer_nodes(
    phytomer_pos: torch.Tensor,                    # (K, 3) in meters
    phytomer_exist: torch.Tensor,                  # (K,) in [0, 1]
    phytomer_scale: Optional[torch.Tensor] = None, # (K, 3) in FM units [L_pet, r_pet, ...]
    stem_radius: float = 0.0035,                 # meters fallback default (3.5mm radius = 7mm diameter)
    exist_thresh: float = 0.25,
    petiole_to_stem_ratio: float = 1.8,          # Option C: stem radius derived from petiole scale
) -> torch.Tensor:
    """Builds continuous botanical stem tubes (Internode, type 3) directly from Stage 2 phytomer nodes.
    
    Creates a directed growth skeleton tree from ground collar (0,0,0) through all active nodes,
    connecting each node to its nearest lower-height parent node.
    
    Option C:
    When phytomer_scale is provided (Petiole scale row in FM units [L_pet, r_pet, ...]),
    the stem segment radius is dynamically derived from the petiole radius of the parent
    node (r_stem = r_pet * petiole_to_stem_ratio), bounded between [1.2mm, 10.0mm],
    naturally reflecting whole-plant ontogeny and acropetal tapering.
    
    Returns:
        (M_stems, 14) part tensor matching HeliosPlantGeometryBuilder layout:
        [type(1)=3, base(3), rot6d(6), scale(3)=[L, r, r], curv(1)=0]
    """
    device = phytomer_pos.device
    mask = phytomer_exist > exist_thresh
    valid_pos = phytomer_pos[mask]  # (N, 3)
    if valid_pos.shape[0] == 0:
        return torch.empty((0, 14), dtype=torch.float32, device=device)

    # Compute node-level stem radii if phytomer_scale is provided (Option C)
    node_radii = None
    if phytomer_scale is not None:
        valid_scale = phytomer_scale[mask]  # (N, 3)
        # Scale row col 1 is petiole radius in FM units (meter * 50.0)
        pet_r_m = valid_scale[:, 1] / 50.0
        # Allometric scaling: stem radius is ~1.8x petiole radius, physically clamped to [1.2mm, 10mm]
        node_radii = (pet_r_m * petiole_to_stem_ratio).clamp(min=0.0012, max=0.010)

    # Add ground collar origin (0, 0, 0)
    origin = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=valid_pos.dtype)
    all_nodes = torch.cat([origin, valid_pos], dim=0)  # (N+1, 3)

    if node_radii is not None:
        # Collar origin connects to main stem base; its radius reflects the plant's maximum stem girth
        origin_r = node_radii.max().unsqueeze(0) if node_radii.numel() > 0 else torch.tensor([stem_radius], device=device, dtype=valid_pos.dtype)
        all_radii = torch.cat([origin_r, node_radii], dim=0)
    else:
        all_radii = None

    # Sort nodes by height (Z-axis ascending)
    z_order = torch.argsort(all_nodes[:, 2])
    sorted_nodes = all_nodes[z_order]
    sorted_radii = all_radii[z_order] if all_radii is not None else None

    # Directed nearest lower-height parent tree
    segments = []
    for i in range(1, sorted_nodes.shape[0]):
        child = sorted_nodes[i]
        parents = sorted_nodes[:i]
        dists = torch.linalg.norm(parents - child.unsqueeze(0), dim=-1)
        parent_idx = torch.argmin(dists)
        parent = parents[parent_idx]

        vec = child - parent
        L = torch.linalg.norm(vec).item()
        if L < 0.0001:  # skip near-zero degenerate segments (<0.1mm)
            continue

        # Forward axis Y along stem direction
        fwd = vec / (L + 1e-8)
        # Construct orthogonal coordinate frame
        z_ref = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=valid_pos.dtype)
        cross_val = torch.linalg.cross(fwd, z_ref)
        if torch.linalg.norm(cross_val) < 1e-4:
            x_ref = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=valid_pos.dtype)
            x_axis = F.normalize(torch.linalg.cross(fwd, x_ref), dim=-1)
        else:
            x_axis = F.normalize(cross_val, dim=-1)
        z_axis = F.normalize(torch.linalg.cross(x_axis, fwd), dim=-1)
        R = torch.stack([x_axis, fwd, z_axis], dim=-1)  # (3, 3)
        rot6d = torch.cat([R[:, 0], R[:, 1]], dim=-1)   # 6D Zhou continuous rotation

        # Determine segment radius from parent node girth (Option C) or constant fallback
        if sorted_radii is not None:
            seg_r = float(sorted_radii[parent_idx].item())
        else:
            seg_r = float(stem_radius)

        row = torch.zeros(14, device=device, dtype=torch.float32)
        row[0] = 3.0                                  # type 3 = internode
        row[1:4] = parent                             # base position
        row[4:10] = rot6d                             # rotation 6d
        row[10] = float(L)                            # length
        row[11] = seg_r                               # radius X
        row[12] = seg_r                               # radius Y
        row[13] = 0.0                                 # curvature
        segments.append(row)

    if len(segments) == 0:
        return torch.empty((0, 14), dtype=torch.float32, device=device)
    return torch.stack(segments, dim=0)


def extract_stem_segments(parts: Optional[torch.Tensor]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extracts true botanical stem tube segments (base [cm], tip [cm]) from internodes (type 3)."""
    if parts is None or len(parts) == 0:
        return []
    internode_mask = (parts[:, 0] == 3)
    inode_parts = parts[internode_mask]
    if len(inode_parts) == 0:
        return []
    R = rot6d_to_matrix(inode_parts[:, 4:10])  # (M, 3, 3)
    bases = (inode_parts[:, 1:4] * 100.0).cpu().numpy()  # [cm]
    lengths = (inode_parts[:, 10] * 100.0).cpu().numpy()  # [cm]
    fwds = R[:, :, 1].cpu().numpy()  # forward axis (Y-axis along stem)
    tips = bases + fwds * lengths[:, None]
    return list(zip(bases, tips))


def make_2x2_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """Tiled 2x2 grid of RGB multi-scale zoom pyramid (1x, 2x, 4x, 8x)."""
    if img_tensor.shape[0] < 12:
        return (img_tensor[:3] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    rgbs = []
    for s in range(4):
        rgb = (img_tensor[s * 4 : s * 4 + 3] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        rgbs.append(rgb)
    top = np.concatenate([rgbs[0], rgbs[1]], axis=1)
    bot = np.concatenate([rgbs[2], rgbs[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    mid_y = grid.shape[0] // 2
    mid_x = grid.shape[1] // 2
    grid[mid_y - 1 : mid_y + 1, :, :] = 1.0
    grid[:, mid_x - 1 : mid_x + 1, :] = 1.0
    return grid


def make_2x2_depth(img_tensor: torch.Tensor, cmap_name: str = "viridis") -> np.ndarray:
    """Tiled 2x2 grid of Depth multi-scale zoom pyramid (1x, 2x, 4x, 8x) in viridis."""
    cmap = plt.get_cmap(cmap_name)
    if img_tensor.shape[0] < 4:
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        return np.zeros((h, w, 3), dtype=np.float32)
    elif img_tensor.shape[0] < 16:
        d = img_tensor[3].cpu().numpy()
        norm_d = np.clip(d / max(0.15, float(d.max())), 0.0, 1.0)
        return cmap(norm_d)[:, :, :3]
    depths = [img_tensor[s * 4 + 3].cpu().numpy() for s in range(4)]
    vmax = max(0.15, max(float(d.max()) for d in depths))
    colored = []
    for d in depths:
        norm_d = np.clip(d / vmax, 0.0, 1.0)
        colored.append(cmap(norm_d)[:, :, :3])
    top = np.concatenate([colored[0], colored[1]], axis=1)
    bot = np.concatenate([colored[2], colored[3]], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    mid_y = grid.shape[0] // 2
    mid_x = grid.shape[1] // 2
    grid[mid_y - 1 : mid_y + 1, :, :] = 1.0
    grid[:, mid_x - 1 : mid_x + 1, :] = 1.0
    return grid


@torch.no_grad()
def evaluate_self_consistency_batch(
    model,
    val_batch: Dict[str, Any],
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    epoch: int,
    output_dir: str = "outputs/eval",
    num_samples_to_plot: int = 3,
    vae: Optional[torch.nn.Module] = None,
    phytomer_vae: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """Runs end-to-end self-consistency check on validation batch.

    Computes:
      - val/silhouette_iou
      - val/depth_mae_meters
      - val/dice_loss (training-loss parity, all panel samples)
    Saves diagnostic panel to outputs/eval/hierarchical_self_consistency_epoch_{epoch:03d}.png
    """
    was_training = model.training
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model

    images = val_batch["image"].to(device)  # (B, 4, H, W) or (B, 16, H, W)
    daps = val_batch.get("dap", None)
    if daps is not None:
        daps = daps.to(device)

    B = images.shape[0]
    # Pure Autonomous Inference: pass daps=None so model determines phytomer capacity
    # 100% autonomously from its own predicted phytomer count (pred_num_phytomers)
    sample_out = raw_model.sample_ode(images=images, daps=None, num_steps=20, vae=vae, phytomer_vae=phytomer_vae)

    pred_geoms = sample_out["pred_geometry"]  # (B, N, 13) or (B, N, 16)
    pred_classes = sample_out["pred_cls"]     # (B, N)
    slot_actives = sample_out["slot_active"]  # (B, N)

    ious = []
    depth_maes = []
    peak_height_diffs = []
    dice_losses = []
    panels_data = []

    os.makedirs(output_dir, exist_ok=True)

    # Plot samples SPREAD across the DAP range (sorted set -> first, last, and
    # evenly spaced in between), not just the first `num_samples_to_plot` —
    # the eval set is DAP-stratified and sorted, so taking the head would always
    # show only the youngest plants.
    plot_idx = np.linspace(0, B - 1, min(B, num_samples_to_plot)).round().astype(int)
    for b in sorted(set(plot_idx.tolist())):
        # 1. Decode predicted lateral phytomer parts (petiole, leaflets, repro)
        if "part_14d" in sample_out:
            pred_14d_b = sample_out["part_14d"][b]
            act_mask = (slot_actives[b] > 0.5) & (pred_classes[b] > EMPTY_IDX)
            lateral_parts = pred_14d_b[act_mask]
        else:
            lateral_parts = decode_predictions_to_part_tensor(
                pred_geoms[b], pred_classes[b], slot_actives[b], device=device
            )

        # Build continuous stem tubes directly from Stage 2 phytomer node scaffold (Option C)
        node_scale_b = sample_out.get("phytomer_scale")
        node_scale_b = node_scale_b[b] if node_scale_b is not None else None
        stem_parts = build_stem_parts_from_phytomer_nodes(
            phytomer_pos=sample_out["phytomer_pos"][b],
            phytomer_exist=sample_out["phytomer_existence"][b],
            phytomer_scale=node_scale_b,
        )
        if stem_parts.shape[0] > 0 and lateral_parts.shape[0] > 0:
            active_parts = torch.cat([stem_parts, lateral_parts], dim=0)
        elif stem_parts.shape[0] > 0:
            active_parts = stem_parts
        else:
            active_parts = lateral_parts

        # 2. Extract Input Drone Image (Channels 0:3 = RGB [-1, 1], Channel 3 = CHM Depth)
        input_img = images[b]
        input_rgb = (input_img[:3] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        input_depth = input_img[3].clamp(min=0.0).cpu().numpy() if input_img.shape[0] >= 4 else np.zeros_like(input_rgb[:, :, 0])

        dap_val = int(daps[b].item()) if daps is not None else 0
        eval_zoom = 8.0 if dap_val <= 15 else 1.0

        # 3. Differentiably render ground-truth 3D label mesh (3D Ground Truth Depth)
        gt_3d_rgb = input_rgb
        gt_3d_depth = input_depth
        gt_point_cloud = None
        gt_parts = None
        gt_mesh = None
        if "nodes" in val_batch:
            try:
                gt_nodes_b = val_batch["nodes"][b].to(device)
                gt_types = gt_nodes_b[:, :FM_OT_END].argmax(dim=-1)
                gt_exist = val_batch["existence_mask"][b].to(device)
                gt_parts = decode_predictions_to_part_tensor(
                    gt_nodes_b[:, FM_BASE_START:], gt_types, gt_exist, device=device
                )
                if gt_parts.shape[0] > 0:
                    gt_mesh = renderer.geo_builder.build_mesh_from_part_tensor(gt_parts, device=device)
                    if "vertices" in gt_mesh and gt_mesh["vertices"].shape[0] > 0:
                        v_np = (gt_mesh["vertices"].detach().cpu().numpy() * 100.0)  # [cm]
                        sub_n = min(len(v_np), 800)
                        sub_idx = np.random.choice(len(v_np), sub_n, replace=False)
                        gt_point_cloud = v_np[sub_idx]

                    gt_rendered_rgbd = renderer.forward(
                        gt_mesh,
                        azimuth_deg=0.0,
                        elevation_deg=90.0,
                        camera_height=5.0,
                        background="ground",
                        focus_plant=False,
                        include_depth=True,
                        zoom_factor=eval_zoom,
                        reference_window_size=1.2,
                    )
                    gt_3d_rgb = gt_rendered_rgbd[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    gt_3d_depth = gt_rendered_rgbd[3].clamp(min=0.0).cpu().numpy()
            except Exception:
                pass

        # 4. Differentiably render predicted 3D plant mesh
        pred_point_cloud = None
        if active_parts.shape[0] > 0:
            try:
                mesh = renderer.geo_builder.build_mesh_from_part_tensor(active_parts, device=device)
                if "vertices" in mesh and mesh["vertices"].shape[0] > 0:
                    v_pred_np = (mesh["vertices"].detach().cpu().numpy() * 100.0)  # [cm]
                    sub_n_p = min(len(v_pred_np), 800)
                    sub_idx_p = np.random.choice(len(v_pred_np), sub_n_p, replace=False)
                    pred_point_cloud = v_pred_np[sub_idx_p]

                rendered_rgbd = renderer.forward(
                    mesh,
                    azimuth_deg=0.0,
                    elevation_deg=90.0,
                    camera_height=5.0,
                    background="ground",
                    focus_plant=False,
                    include_depth=True,
                    zoom_factor=eval_zoom,
                    reference_window_size=1.2,
                )
                pred_rgb = rendered_rgbd[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                pred_depth = rendered_rgbd[3].clamp(min=0.0).cpu().numpy()
            except Exception as e:
                pred_rgb = np.zeros_like(input_rgb)
                pred_depth = np.zeros_like(input_depth)
        else:
            pred_rgb = np.zeros_like(input_rgb)
            pred_depth = np.zeros_like(input_depth)

        # 5. Compute Self-Consistency Metrics against 3D Ground Truth Label Depth
        pred_mask = (pred_depth > 0.005)
        gt_mask = (gt_3d_depth > 0.005)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = float(intersection / max(union, 1))

        # Depth MAE across 3D ground truth footprint
        canopy_mask = np.logical_or(pred_mask, gt_mask)
        if canopy_mask.sum() > 0:
            depth_mae_3d_gt = float(np.abs(pred_depth[canopy_mask] - gt_3d_depth[canopy_mask]).mean())
        else:
            depth_mae_3d_gt = 0.0

        peak_err = float(abs(pred_depth.max() - gt_3d_depth.max()))

        ious.append(iou)
        depth_maes.append(depth_mae_3d_gt)
        peak_height_diffs.append(peak_err)

        # 5b. Self-Consistency Cosine Color & Silhouette Dice (training-loss parity:
        # computed on the SAME normalization & mask definitions as the training
        # photometric losses, over ALL evaluated panel samples at full render).
        try:
            # GT tensors: dataset cache stores RGB in [-1, 1], depth in meters.
            gt_rgb_t = input_img[:3].to(device).float()
            gt_depth_t = input_img[3].to(device).float().clamp(min=0.0)
            pred_rgb_t = rendered_rgbd[:3].clamp(0.0, 1.0)  # renderer outputs [0, 1]
            pred_rgb_t = (pred_rgb_t - 0.5) / 0.5           # -> [-1, 1] like dataset GT
            pred_depth_t = rendered_rgbd[3].clamp(min=0.0)

            pred_soft = torch.sigmoid((pred_depth_t - 0.005) * 100.0)
            gt_soft = (gt_depth_t > 0.005).float()
            inter = (pred_soft * gt_soft).sum()
            denom = pred_soft.sum() + gt_soft.sum()
            dice_loss = float(1.0 - (2.0 * inter + 1e-4) / (denom + 1e-4))
        except Exception:
            dice_loss = 1.0
        dice_losses.append(dice_loss)

        # Oblique 3D Real Mesh Composite (Cyan: GT, Amber: Pred, Blend: Overlap)
        comp_img = None
        try:
            # Lifecycle-adaptive zoom for optimal 3D visual framing
            if dap_val <= 10:
                eval_zoom_obl = 8.0
            elif dap_val <= 20:
                eval_zoom_obl = 4.0
            elif dap_val <= 35:
                eval_zoom_obl = 2.2
            else:
                eval_zoom_obl = 1.3

            rend_gt_obl = None
            if gt_mesh is not None and "vertices" in gt_mesh and gt_mesh["vertices"].shape[0] > 0:
                rend_gt_obl = renderer.forward(
                    gt_mesh, elevation_deg=30.0, azimuth_deg=30.0, background="black", focus_plant=False, include_depth=False, zoom_factor=eval_zoom_obl, reference_window_size=1.2
                )

            rend_pred_obl = None
            if active_parts.shape[0] > 0 and "vertices" in mesh and mesh["vertices"].shape[0] > 0:
                rend_pred_obl = renderer.forward(
                    mesh, elevation_deg=30.0, azimuth_deg=30.0, background="black", focus_plant=False, include_depth=False, zoom_factor=eval_zoom_obl, reference_window_size=1.2
                )

            def boost_lum(lum, thresh=0.02):
                mask = lum > thresh
                if not mask.any():
                    return lum
                val = lum.clone()
                peak = val[mask].max()
                val_norm = torch.clamp(val / (peak + 1e-4), 0.0, 1.0)
                val_boosted = torch.pow(val_norm, 0.65) * 0.9 + 0.1
                return torch.where(mask, val_boosted, torch.zeros_like(lum))

            # Print style (Heesup, 2026-09-15): composite on WHITE, GT in blue, prediction in vermillion,
            # overlap in slate (Okabe-Ito palette), shaded by the render's luminance.
            c_gt = torch.tensor([0.0, 0.447, 0.698], device=device)[:, None, None]
            c_pred = torch.tensor([0.835, 0.369, 0.0], device=device)[:, None, None]
            c_both = torch.tensor([0.30, 0.30, 0.36], device=device)[:, None, None]
            white = torch.ones(3, 1, 1, device=device)

            def _paint(lum, color):
                # white -> color as luminance grows; keeps the leaf shading visible in print
                m = (lum > 0.03).float()
                shade = 0.55 + 0.45 * lum.clamp(0, 1)
                return m * color * shade + (1 - m) * white

            if rend_gt_obl is not None and rend_pred_obl is not None:
                gt_lum = boost_lum(rend_gt_obl[:3].mean(dim=0, keepdim=True))
                pred_lum = boost_lum(rend_pred_obl[:3].mean(dim=0, keepdim=True))
                gt_m = gt_lum > 0.03
                pred_m = pred_lum > 0.03
                comp_tensor = white.expand(3, gt_lum.shape[1], gt_lum.shape[2]).clone()
                comp_tensor = torch.where(gt_m & ~pred_m, _paint(gt_lum, c_gt), comp_tensor)
                comp_tensor = torch.where(pred_m & ~gt_m, _paint(pred_lum, c_pred), comp_tensor)
                comp_tensor = torch.where(gt_m & pred_m, _paint(torch.max(gt_lum, pred_lum), c_both), comp_tensor)
                comp_img = comp_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            elif rend_gt_obl is not None:
                comp_img = _paint(boost_lum(rend_gt_obl[:3].mean(dim=0, keepdim=True)), c_gt).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            elif rend_pred_obl is not None:
                comp_img = _paint(boost_lum(rend_pred_obl[:3].mean(dim=0, keepdim=True)), c_pred).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        except Exception:
            comp_img = None

        dap_val = int(daps[b].item()) if daps is not None else 0

        # Extract Helios Ray-traced JPEG reference if available
        jpeg_path = None
        if "jpeg" in val_batch and val_batch["jpeg"] is not None:
            jpegs = val_batch["jpeg"]
            if isinstance(jpegs, (list, tuple)) and b < len(jpegs) and jpegs[b]:
                jpeg_path = jpegs[b]
        if not jpeg_path and "prefix" in val_batch and val_batch["prefix"] is not None:
            prefixes = val_batch["prefix"]
            if isinstance(prefixes, (list, tuple)) and b < len(prefixes) and prefixes[b]:
                for sfx in ("_rad.jpeg", "_vis.jpeg"):
                    candidate = os.path.join("dataset/helios_data/cowpea", f"{prefixes[b]}{sfx}")
                    if os.path.exists(candidate):
                        jpeg_path = candidate
                        break

        im_helios = None
        if jpeg_path and os.path.exists(jpeg_path):
            try:
                im_helios = Image.open(jpeg_path).convert("RGB")
            except Exception:
                im_helios = None

        # Extract Ground-Truth & Predicted 3D Botanical Nodes
        gt_nodes = np.zeros((0, 3), dtype=np.float32)
        if "nodes" in val_batch:
            try:
                gt_nodes_raw = val_batch["nodes"][b]
                gt_types_b = gt_nodes_raw[:, :FM_OT_END].argmax(dim=-1).cpu().numpy()
                gt_exist_b = val_batch["existence_mask"][b].cpu().numpy() if "existence_mask" in val_batch else np.ones_like(gt_types_b)
                act_gt_mask = (gt_types_b > EMPTY_IDX) & (gt_exist_b > 0.5)

                if act_gt_mask.any():
                    gt_pos_metric = (gt_nodes_raw[act_gt_mask, FM_BASE_START:FM_BASE_START+3] / BASE_SCALE).cpu().numpy()
                    act_types_sub = gt_types_b[act_gt_mask]

                    # Physical Nodes: Petiole bases (type 4) + standalone internodes (type 3)
                    pet_idx = np.where(act_types_sub == 4)[0]
                    if len(pet_idx) > 0:
                        gt_nodes = gt_pos_metric[pet_idx]
                        in_idx = np.where(act_types_sub == 3)[0]
                        if len(in_idx) > 0:
                            dists = np.linalg.norm(gt_pos_metric[in_idx, None, :] - gt_nodes[None, :, :], axis=-1).min(axis=1)
                            standalone = in_idx[dists > 0.04]
                            if len(standalone) > 0:
                                gt_nodes = np.concatenate([gt_nodes, gt_pos_metric[standalone]], axis=0)
                    else:
                        in_idx = np.where(act_types_sub == 3)[0]
                        gt_nodes = gt_pos_metric[in_idx] if len(in_idx) > 0 else gt_pos_metric[:16]
            except Exception:
                gt_nodes = np.zeros((0, 3), dtype=np.float32)

        # Predicted 3D Nodes
        pred_nodes = np.zeros((0, 3), dtype=np.float32)
        node_rmse_cm = 0.0
        if "phytomer_pos" in sample_out:
            try:
                pred_pos_raw = (sample_out["phytomer_pos"][b] / BASE_SCALE).cpu().numpy()
                pred_exist_raw = sample_out["phytomer_existence"][b].cpu().numpy() if "phytomer_existence" in sample_out else np.ones(len(pred_pos_raw))

                k_target = max(len(gt_nodes), 8)
                active_node_mask = pred_exist_raw > 0.35
                if active_node_mask.sum() >= 4:
                    pred_nodes = pred_pos_raw[active_node_mask]
                else:
                    top_k_idx = np.argsort(-pred_exist_raw)[:k_target]
                    pred_nodes = pred_pos_raw[top_k_idx]

                if len(pred_nodes) > 0 and len(gt_nodes) > 0:
                    dists = np.linalg.norm(pred_nodes[:, None, :] - gt_nodes[None, :, :], axis=-1).min(axis=1)
                    node_rmse_cm = float(np.sqrt(np.mean(dists ** 2)) * 100.0)
            except Exception:
                pass

        # Extract True Botanical Stem Segments (Internode tubes)
        pred_stem_segments = extract_stem_segments(active_parts)
        gt_stem_segments = []
        if gt_parts is not None and gt_parts.shape[0] > 0:
            try:
                gt_stem_segments = extract_stem_segments(gt_parts)
            except Exception:
                gt_stem_segments = []

        pred_num_phy = float(sample_out["pred_num_phytomers"][b].item()) if ("pred_num_phytomers" in sample_out and sample_out["pred_num_phytomers"] is not None) else None

        panels_data.append({
            "dap": dap_val,
            "im_helios": im_helios,
            "input_img": input_img.cpu(),
            "gt_3d_depth": gt_3d_depth,
            "pred_rgb": pred_rgb,
            "pred_depth": pred_depth,
            "iou": iou,
            "depth_mae": depth_mae_3d_gt,
            "active_organs": int(active_parts.shape[0]),
            "gt_nodes": gt_nodes,
            "pred_nodes": pred_nodes,
            "node_rmse_cm": node_rmse_cm,
            "gt_point_cloud": gt_point_cloud,
            "pred_point_cloud": pred_point_cloud,
            "pred_num_phy": pred_num_phy,
            "gt_num_phy": len(gt_nodes),
            "gt_stem_segments": gt_stem_segments,
            "pred_stem_segments": pred_stem_segments,
            "comp_img": comp_img,
        })

    # 6. Build Diagnostic Visualization Figure (7 Columns: Ref + RGB + Depth + Pred RGB + Pred Depth + Real Solid Mesh + 3D Botanical Skeleton & Nodes)
    if panels_data:
        n_rows = len(panels_data)
        # Academic print style (white background, one sans-serif family, dark text, muted palette).
        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                             "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "text.color": "#111111",
                             "axes.labelcolor": "#111111", "xtick.color": "#444444", "ytick.color": "#444444"})
        fig = plt.figure(figsize=(24.5, 3.5 * n_rows), facecolor="white")
        gs = fig.add_gridspec(n_rows, 7, wspace=0.14, hspace=0.25, left=0.03, right=0.97, top=0.84, bottom=0.04)

        mean_node_rmse = np.mean([d["node_rmse_cm"] for d in panels_data])
        fig.suptitle(
            f"Hierarchical Flow Matching: 3D Reconstruction, Real Mesh & Botanical Skeleton (Epoch {epoch:03d})\n"
            f"Mean silhouette IoU {np.mean(ious)*100:.1f}%, depth MAE {np.mean(depth_maes)*100:.2f} cm, node RMSE {mean_node_rmse:.1f} cm",
            fontsize=13, fontweight="normal", color="#111111", y=0.96
        )

        col_titles = [
            "(a) Helios ray-traced reference\n(not used in training)",
            "(b) Input RGB pyramid\n(1×, 2× / 4×, 8×)",
            "(c) Input canopy height pyramid\n(1×, 2× / 4×, 8×)",
            "(d) Predicted plant, top view",
            "(e) Predicted canopy height",
            "(f) Oblique overlay\n(blue: GT, orange: prediction)",
            "(g) Stem trees and nodes",
        ]

        for row_idx, data in enumerate(panels_data):
            vmax_d = max(0.3, float(data["gt_3d_depth"].max()))

            # Create 2D subplots for Columns 0 to 5
            axes_row = []
            for col_idx in range(6):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx], fontsize=9.5, fontweight="normal", color="#111111", pad=8)
                axes_row.append(ax)

            # Col 0: Helios Raytrace Reference
            if data["im_helios"] is not None:
                axes_row[0].imshow(data["im_helios"])
            else:
                axes_row[0].text(0.5, 0.5, "Reference\nN/A", color="#666666", ha="center", va="center")
            axes_row[0].set_ylabel(f"DAP {data['dap']}\n({data['active_organs']} organs)", fontsize=10, color="#111111", labelpad=8)
            axes_row[0].set_xticks([])
            axes_row[0].set_yticks([])

            # Col 1: Drone RGB 2x2 Pyramid
            rgb_2x2 = make_2x2_rgb(data["input_img"])
            axes_row[1].imshow(rgb_2x2)
            axes_row[1].set_xticks([])
            axes_row[1].set_yticks([])

            # Col 2: Drone Depth 2x2 Pyramid
            depth_2x2 = make_2x2_depth(data["input_img"])
            axes_row[2].imshow(depth_2x2)
            axes_row[2].set_xticks([])
            axes_row[2].set_yticks([])

            # Col 3: Pred 3D Mesh
            axes_row[3].imshow(data["pred_rgb"])
            phy_str = f" | Phy: {data['pred_num_phy']:.1f}/{data['gt_num_phy']}" if data.get("pred_num_phy") is not None else ""
            axes_row[3].set_xlabel(f"IoU {data['iou']*100:.1f}%{phy_str}", fontsize=9, color="#111111")
            axes_row[3].set_xticks([])
            axes_row[3].set_yticks([])

            # Col 4: Pred 3D Depth
            axes_row[4].imshow(data["pred_depth"], cmap="viridis", vmin=0.0, vmax=vmax_d)
            axes_row[4].set_xlabel(f"Depth MAE {data['depth_mae']*100:.2f} cm", fontsize=8.5, color="#111111")
            axes_row[4].set_xticks([])
            axes_row[4].set_yticks([])

            # Col 5: Oblique 3D Real Mesh Composite (Cyan: GT, Amber: Pred, Blend: Overlap)
            if data.get("comp_img") is not None:
                axes_row[5].imshow(data["comp_img"])
                axes_row[5].set_xlabel("overlap in slate", fontsize=8.5, color="#111111")
            else:
                axes_row[5].text(0.5, 0.5, "Mesh N/A", color="#666666", ha="center", va="center")
            axes_row[5].set_xticks([])
            axes_row[5].set_yticks([])
            for spine in axes_row[5].spines.values():
                spine.set_visible(False)

            # Col 6: 3D Botanical Skeleton & Nodes (Matplotlib 3D)
            ax6 = fig.add_subplot(gs[row_idx, 6], projection="3d")
            ax6.set_facecolor("white")
            ax6.view_init(elev=24, azim=-55)
            if row_idx == 0:
                ax6.set_title(col_titles[6], fontsize=9.5, fontweight="normal", color="#111111", pad=8)

            # Draw GT stem segments (internode tubes)
            gt_segs = data.get("gt_stem_segments", [])
            for i, (b_pos, tp_pos) in enumerate(gt_segs):
                lbl = "GT Stem Tree" if i == 0 else ""
                ax6.plot([b_pos[0], tp_pos[0]], [b_pos[1], tp_pos[1]], [b_pos[2], tp_pos[2]],
                         color="#0072B2", linestyle="-", linewidth=2.2, alpha=0.95, zorder=10, label=lbl)

            # Draw Pred stem segments (internode tubes)
            pred_segs = data.get("pred_stem_segments", [])
            for i, (b_pos, tp_pos) in enumerate(pred_segs):
                lbl = "Pred Stem Tree" if i == 0 else ""
                ax6.plot([b_pos[0], tp_pos[0]], [b_pos[1], tp_pos[1]], [b_pos[2], tp_pos[2]],
                         color="#D55E00", linestyle="--", linewidth=1.8, alpha=0.95, zorder=11, label=lbl)

            # Draw GT Nodes
            if len(data["gt_nodes"]) > 0:
                ax6.scatter(data["gt_nodes"][:, 0] * 100, data["gt_nodes"][:, 1] * 100, data["gt_nodes"][:, 2] * 100,
                            c="#0072B2", s=26, edgecolors="black", linewidth=0.5, zorder=12,
                            label=f"GT Node (N={len(data['gt_nodes'])})")

            # Draw Pred Nodes
            if len(data["pred_nodes"]) > 0:
                ax6.scatter(data["pred_nodes"][:, 0] * 100, data["pred_nodes"][:, 1] * 100, data["pred_nodes"][:, 2] * 100,
                            c="#D55E00", marker="s", s=22, edgecolors="black", linewidth=0.5, zorder=13,
                            label=f"Pred Node (K={len(data['pred_nodes'])})")

            ax6.tick_params(colors="#444444", labelsize=6.5, pad=0.5)
            for pane in [ax6.xaxis.pane, ax6.yaxis.pane, ax6.zaxis.pane]:
                pane.set_facecolor("white")
                pane.set_edgecolor("#cccccc")

            ax6.set_xlabel("x (cm)", fontsize=7, color="#111111", labelpad=-3)
            ax6.set_ylabel("y (cm)", fontsize=7, color="#111111", labelpad=-3)
            ax6.set_zlabel("z (cm)", fontsize=7, color="#111111", labelpad=-3)

            rmse_str = f"node RMSE {data['node_rmse_cm']:.1f} cm"
            ax6.text2D(0.05, 0.88, rmse_str, transform=ax6.transAxes, color="#111111", fontsize=8)
            if row_idx == 0:
                ax6.legend(loc="upper right", fontsize=6.0, facecolor="white", edgecolor="#999999", framealpha=0.9)

        panel_path = os.path.join(output_dir, f"hierarchical_self_consistency_epoch_{epoch:03d}.png")
        fig.savefig(panel_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # Log to W&B if active (including interactive 3D Point Cloud object)
        if WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {
                "val/silhouette_iou": float(np.mean(ious)),
                "val/depth_mae": float(np.mean(depth_maes)),
                "val/peak_height_error": float(np.mean(peak_height_diffs)),
                "val/node_rmse_cm": float(mean_node_rmse),
                "val/self_consistency_panel": wandb.Image(panel_path),
                "val/dice_loss": float(np.mean(dice_losses)) if dice_losses else 0.0,
            }
            try:
                first_data = panels_data[0]
                pts_list = []
                if first_data.get("gt_point_cloud") is not None:
                    gt_pc = first_data["gt_point_cloud"]
                    gt_rgb = np.tile(np.array([16, 185, 129]), (len(gt_pc), 1))
                    pts_list.append(np.hstack([gt_pc, gt_rgb]))
                if len(first_data["gt_nodes"]) > 0:
                    gt_n_cm = first_data["gt_nodes"] * 100.0
                    gt_n_rgb = np.tile(np.array([0, 229, 255]), (len(gt_n_cm), 1))
                    pts_list.append(np.hstack([gt_n_cm, gt_n_rgb]))
                if len(first_data["pred_nodes"]) > 0:
                    pred_n_cm = first_data["pred_nodes"] * 100.0
                    pred_n_rgb = np.tile(np.array([244, 63, 94]), (len(pred_n_cm), 1))
                    pts_list.append(np.hstack([pred_n_cm, pred_n_rgb]))
                if pts_list:
                    all_pts_colored = np.vstack(pts_list)
                    log_dict["val/interactive_3d_point_cloud"] = wandb.Object3D(all_pts_colored)
            except Exception:
                pass

            wandb.log(log_dict, step=epoch)

    # Restore training mode so the in-loop eval never leaves the DDP-wrapped
    # model in eval() state for subsequent training steps (Job 38237555 deadlock).
    if was_training:
        model.train()

    return {
        "val/silhouette_iou": float(np.mean(ious)) if ious else 0.0,
        "val/depth_mae": float(np.mean(depth_maes)) if depth_maes else 0.0,
        "val/peak_height_error": float(np.mean(peak_height_diffs)) if peak_height_diffs else 0.0,
        "val/dice_loss": float(np.mean(dice_losses)) if dice_losses else 0.0,
        "val/node_rmse_cm": float(np.mean([d["node_rmse_cm"] for d in panels_data])) if panels_data else 0.0,
        "silhouette_iou": float(np.mean(ious)) if ious else 0.0,
        "depth_mae": float(np.mean(depth_maes)) if depth_maes else 0.0,
        "peak_height_error": float(np.mean(peak_height_diffs)) if peak_height_diffs else 0.0,
        "dice_loss": float(np.mean(dice_losses)) if dice_losses else 0.0,
    }
