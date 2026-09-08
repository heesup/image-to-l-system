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

from diffusion_based.dataset.part_array_dataset import (
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
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder

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
    output_dir: str = "docs/results/assets",
    num_samples_to_plot: int = 3,
    vae: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """Runs end-to-end self-consistency check on validation batch.

    Computes:
      - val/silhouette_iou
      - val/depth_mae_meters
      - val/peak_height_error_meters
    Saves diagnostic panel to docs/results/assets/hierarchical_self_consistency_epoch_{epoch:03d}.png
    """
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model

    images = val_batch["image"].to(device)  # (B, 4, H, W) or (B, 16, H, W)
    daps = val_batch.get("dap", None)
    if daps is not None:
        daps = daps.to(device)

    B = images.shape[0]
    sample_out = raw_model.sample_ode(images=images, daps=daps, num_steps=20, vae=vae)

    pred_geoms = sample_out["pred_geometry"]  # (B, N, 13) or (B, N, 16)
    pred_classes = sample_out["pred_cls"]     # (B, N)
    slot_actives = sample_out["slot_active"]  # (B, N)

    ious = []
    depth_maes = []
    peak_height_diffs = []
    panels_data = []

    os.makedirs(output_dir, exist_ok=True)

    for b in range(min(B, num_samples_to_plot)):
        # 1. Decode predicted parts
        if "part_14d" in sample_out:
            pred_14d_b = sample_out["part_14d"][b]
            act_mask = (slot_actives[b] > 0.5) & (pred_classes[b] > EMPTY_IDX)
            active_parts = pred_14d_b[act_mask]
        else:
            active_parts = decode_predictions_to_part_tensor(
                pred_geoms[b], pred_classes[b], slot_actives[b], device=device
            )

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

        # Oblique 3D Real Mesh Composite (Cyan: GT, Amber: Pred, Blend: Overlap)
        comp_img = None
        try:
            rend_gt_obl = None
            if gt_mesh is not None and "vertices" in gt_mesh and gt_mesh["vertices"].shape[0] > 0:
                rend_gt_obl = renderer.forward(
                    gt_mesh, elevation_deg=30.0, azimuth_deg=30.0, background="black", focus_plant=False, include_depth=False, zoom_factor=eval_zoom, reference_window_size=1.2
                )

            rend_pred_obl = None
            if active_parts.shape[0] > 0 and "vertices" in mesh and mesh["vertices"].shape[0] > 0:
                rend_pred_obl = renderer.forward(
                    mesh, elevation_deg=30.0, azimuth_deg=30.0, background="black", focus_plant=False, include_depth=False, zoom_factor=eval_zoom, reference_window_size=1.2
                )

            if rend_gt_obl is not None and rend_pred_obl is not None:
                gt_lum = rend_gt_obl[:3].mean(dim=0, keepdim=True)
                pred_lum = rend_pred_obl[:3].mean(dim=0, keepdim=True)
                cyan = torch.tensor([0.0, 0.9, 1.0], device=device)[:, None, None]
                amber = torch.tensor([1.0, 0.55, 0.0], device=device)[:, None, None]
                gt_colored = gt_lum * cyan
                pred_colored = pred_lum * amber
                overlap_mask = (gt_lum > 0.08) & (pred_lum > 0.08)
                comp_tensor = gt_colored + pred_colored
                comp_tensor = torch.where(overlap_mask, torch.tensor([0.95, 0.85, 0.3], device=device)[:, None, None] * torch.max(gt_lum, pred_lum), comp_tensor)
                comp_img = comp_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            elif rend_gt_obl is not None:
                gt_lum = rend_gt_obl[:3].mean(dim=0, keepdim=True)
                cyan = torch.tensor([0.0, 0.9, 1.0], device=device)[:, None, None]
                comp_img = (gt_lum * cyan).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            elif rend_pred_obl is not None:
                pred_lum = rend_pred_obl[:3].mean(dim=0, keepdim=True)
                amber = torch.tensor([1.0, 0.55, 0.0], device=device)[:, None, None]
                comp_img = (pred_lum * amber).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
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
        if "anchor_pos" in sample_out:
            try:
                pred_pos_raw = (sample_out["anchor_pos"][b] / BASE_SCALE).cpu().numpy()
                pred_exist_raw = sample_out["anchor_existence"][b].cpu().numpy() if "anchor_existence" in sample_out else np.ones(len(pred_pos_raw))

                k_target = max(len(gt_nodes), 8)
                active_anc_mask = pred_exist_raw > 0.35
                if active_anc_mask.sum() >= 4:
                    pred_nodes = pred_pos_raw[active_anc_mask]
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
        fig = plt.figure(figsize=(24.5, 3.5 * n_rows), facecolor="#12151a")
        gs = fig.add_gridspec(n_rows, 7, wspace=0.14, hspace=0.25, left=0.03, right=0.97, top=0.84, bottom=0.04)

        mean_node_rmse = np.mean([d["node_rmse_cm"] for d in panels_data])
        fig.suptitle(
            f"Hierarchical Flow Matching: 3D Reconstruction, Real Mesh & Botanical Skeleton (Epoch {epoch:03d})\n"
            f"Mean Silhouette IoU: {np.mean(ious)*100:.1f}% | Depth MAE: {np.mean(depth_maes)*100:.2f} cm | Node RMSE: {mean_node_rmse:.1f} cm",
            fontsize=15, fontweight="bold", color="#f0f4f8", y=0.96
        )

        col_titles = [
            "0. Helios Raytrace\n(Reference - Not Trained)",
            "1. Drone RGB Pyramid\n(2x2: 1x, 2x / 4x, 8x)",
            "2. Drone Depth Pyramid\n(2x2: 1x, 2x / 4x, 8x)",
            "3. Pred 3D Mesh\n(Top-Down Reconstruction)",
            "4. Pred 3D Depth\n(Canopy Height CHM)",
            "5. Real 3D Solid Mesh\n(Oblique 30° Composite)",
            "6. 3D Botanical Skeleton\n(Stem Trees & Nodes)",
        ]

        for row_idx, data in enumerate(panels_data):
            vmax_d = max(0.3, float(data["gt_3d_depth"].max()))

            # Create 2D subplots for Columns 0 to 5
            axes_row = []
            for col_idx in range(6):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                if row_idx == 0:
                    color = "#fbbf24" if col_idx == 0 else ("#38bdf8" if col_idx in (1, 2) else ("#34d399" if col_idx in (3, 4) else "#00e5ff"))
                    ax.set_title(col_titles[col_idx], fontsize=9.5, fontweight="bold", color=color, pad=8)
                axes_row.append(ax)

            # Col 0: Helios Raytrace Reference
            if data["im_helios"] is not None:
                axes_row[0].imshow(data["im_helios"])
            else:
                axes_row[0].text(0.5, 0.5, "Reference\nN/A", color="#94a3b8", ha="center", va="center")
            axes_row[0].set_ylabel(f"DAP {data['dap']}\n({data['active_organs']} organs)", fontsize=11, color="#fbbf24", labelpad=8)
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
            axes_row[3].set_xlabel(f"IoU: {data['iou']*100:.1f}%{phy_str}", fontsize=9, color="#4ade80")
            axes_row[3].set_xticks([])
            axes_row[3].set_yticks([])

            # Col 4: Pred 3D Depth
            axes_row[4].imshow(data["pred_depth"], cmap="viridis", vmin=0.0, vmax=vmax_d)
            axes_row[4].set_xlabel(f"Depth MAE: {data['depth_mae']*100:.2f} cm", fontsize=8.5, color="#38bdf8")
            axes_row[4].set_xticks([])
            axes_row[4].set_yticks([])

            # Col 5: Oblique 3D Real Mesh Composite (Cyan: GT, Amber: Pred, Blend: Overlap)
            if data.get("comp_img") is not None:
                axes_row[5].imshow(data["comp_img"])
                axes_row[5].set_xlabel("Real 3D Mesh Composite\n(Cyan: GT, Amber: Pred)", fontsize=8.5, color="#00e5ff")
            else:
                axes_row[5].text(0.5, 0.5, "Mesh N/A", color="#94a3b8", ha="center", va="center")
            axes_row[5].set_xticks([])
            axes_row[5].set_yticks([])
            for spine in axes_row[5].spines.values():
                spine.set_visible(False)

            # Col 6: 3D Botanical Skeleton & Nodes (Matplotlib 3D)
            ax6 = fig.add_subplot(gs[row_idx, 6], projection="3d")
            ax6.set_facecolor("#181c24")
            ax6.view_init(elev=24, azim=-55)
            if row_idx == 0:
                ax6.set_title(col_titles[6], fontsize=9.5, fontweight="bold", color="#e879f9", pad=8)

            # Draw GT stem segments (internode tubes)
            gt_segs = data.get("gt_stem_segments", [])
            for i, (b_pos, tp_pos) in enumerate(gt_segs):
                lbl = "GT Stem Tree" if i == 0 else ""
                ax6.plot([b_pos[0], tp_pos[0]], [b_pos[1], tp_pos[1]], [b_pos[2], tp_pos[2]],
                         color="#00e5ff", linestyle="-", linewidth=3.2, alpha=0.95, zorder=10, label=lbl)

            # Draw Pred stem segments (internode tubes)
            pred_segs = data.get("pred_stem_segments", [])
            for i, (b_pos, tp_pos) in enumerate(pred_segs):
                lbl = "Pred Stem Tree" if i == 0 else ""
                ax6.plot([b_pos[0], tp_pos[0]], [b_pos[1], tp_pos[1]], [b_pos[2], tp_pos[2]],
                         color="#f43f5e", linestyle="--", linewidth=2.4, alpha=0.90, zorder=11, label=lbl)

            # Draw GT Nodes
            if len(data["gt_nodes"]) > 0:
                ax6.scatter(data["gt_nodes"][:, 0] * 100, data["gt_nodes"][:, 1] * 100, data["gt_nodes"][:, 2] * 100,
                            c="#00e5ff", s=32, edgecolors="white", linewidth=0.9, zorder=12,
                            label=f"GT Node (N={len(data['gt_nodes'])})")

            # Draw Pred Nodes
            if len(data["pred_nodes"]) > 0:
                ax6.scatter(data["pred_nodes"][:, 0] * 100, data["pred_nodes"][:, 1] * 100, data["pred_nodes"][:, 2] * 100,
                            c="#f43f5e", marker="s", s=28, edgecolors="black", linewidth=0.8, zorder=13,
                            label=f"Pred Node (K={len(data['pred_nodes'])})")

            ax6.tick_params(colors="#94a3b8", labelsize=6.5, pad=0.5)
            for pane in [ax6.xaxis.pane, ax6.yaxis.pane, ax6.zaxis.pane]:
                pane.set_facecolor("#1e2330")
                pane.set_edgecolor("#334155")

            ax6.set_xlabel("X [cm]", fontsize=7, color="#94a3b8", labelpad=-3)
            ax6.set_ylabel("Y [cm]", fontsize=7, color="#94a3b8", labelpad=-3)
            ax6.set_zlabel("Z [cm]", fontsize=7, color="#94a3b8", labelpad=-3)

            rmse_str = f"Node RMSE: {data['node_rmse_cm']:.1f} cm"
            ax6.text2D(0.05, 0.88, rmse_str, transform=ax6.transAxes, color="#e879f9", fontsize=8, fontweight="bold")
            if row_idx == 0:
                ax6.legend(loc="upper right", fontsize=6.0, facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", framealpha=0.8)

        panel_path = os.path.join(output_dir, f"hierarchical_self_consistency_epoch_{epoch:03d}.png")
        fig.savefig(panel_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # Log to W&B if active (including interactive 3D Point Cloud object)
        if WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {
                "val/silhouette_iou": float(np.mean(ious)),
                "val/depth_mae": float(np.mean(depth_maes)),
                "val/peak_height_error": float(np.mean(peak_height_diffs)),
                "val/node_rmse_cm": float(mean_node_rmse),
                "val/self_consistency_panel": wandb.Image(panel_path),
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

    return {
        "val/silhouette_iou": float(np.mean(ious)) if ious else 0.0,
        "val/depth_mae": float(np.mean(depth_maes)) if depth_maes else 0.0,
        "val/peak_height_error": float(np.mean(peak_height_diffs)) if peak_height_diffs else 0.0,
        "val/node_rmse_cm": float(np.mean([d["node_rmse_cm"] for d in panels_data])) if panels_data else 0.0,
        "silhouette_iou": float(np.mean(ious)) if ious else 0.0,
        "depth_mae": float(np.mean(depth_maes)) if depth_maes else 0.0,
        "peak_height_error": float(np.mean(peak_height_diffs)) if peak_height_diffs else 0.0,
    }
