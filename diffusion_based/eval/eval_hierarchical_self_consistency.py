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
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def make_2x2_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """Tiled 2x2 grid of RGB multi-scale zoom pyramid (1x, 2x, 4x, 8x)."""
    if img_tensor.shape[0] < 16:
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
    if img_tensor.shape[0] < 16:
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
        if active_parts.shape[0] > 0:
            try:
                mesh = renderer.geo_builder.build_mesh_from_part_tensor(active_parts, device=device)
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

        dap_val = int(daps[b].item()) if daps is not None else 0

        # Extract Helios Ray-traced JPEG reference if available
        jpeg_path = None
        if "jpeg" in val_batch and val_batch["jpeg"] is not None:
            jpegs = val_batch["jpeg"]
            if isinstance(jpegs, (list, tuple)) and b < len(jpegs):
                jpeg_path = jpegs[b]
        elif "prefix" in val_batch and val_batch["prefix"] is not None:
            prefixes = val_batch["prefix"]
            if isinstance(prefixes, (list, tuple)) and b < len(prefixes):
                candidate = os.path.join("dataset/helios_data/cowpea", f"{prefixes[b]}_rad.jpeg")
                if os.path.exists(candidate):
                    jpeg_path = candidate

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
        })

    # 6. Build Diagnostic Visualization Figure (7 Columns: Helios Ref + 2x2 RGB + 2x2 Depth + 3D Preds & Error + 3D Nodes)
    if panels_data:
        n_rows = len(panels_data)
        fig, axes = plt.subplots(n_rows, 7, figsize=(23.5, 3.4 * n_rows), facecolor="#12151a")
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        mean_node_rmse = np.mean([d["node_rmse_cm"] for d in panels_data])
        plt.subplots_adjust(wspace=0.12, hspace=0.25, left=0.03, right=0.97, top=0.82, bottom=0.04)
        fig.suptitle(
            f"Hierarchical Flow Matching: 3D Reconstruction & Botanical Node Estimation (Epoch {epoch:03d})\n"
            f"Mean Silhouette IoU: {np.mean(ious)*100:.1f}% | Depth MAE: {np.mean(depth_maes)*100:.2f} cm | Node RMSE: {mean_node_rmse:.1f} cm",
            fontsize=15, fontweight="bold", color="#f0f4f8", y=0.96
        )

        col_titles = [
            "0. Helios Raytrace\n(Reference - Not Trained)",
            "1. Drone RGB Pyramid\n(2x2: 1x, 2x / 4x, 8x)",
            "2. Drone Depth Pyramid\n(2x2: 1x, 2x / 4x, 8x)",
            "3. Pred 3D Mesh\n(Reconstruction)",
            "4. Pred 3D Depth\n(CHM Height)",
            "5. Depth Error Heatmap\n|Pred - GT 3D|",
            "6. 3D Skeleton Nodes\n(GT Cyan vs Pred Magenta)",
        ]

        for col_idx, title in enumerate(col_titles):
            if col_idx == 0:
                color = "#fbbf24"
            elif col_idx in (1, 2):
                color = "#38bdf8"
            elif col_idx == 6:
                color = "#f43f5e"
            else:
                color = "#34d399"
            axes[0, col_idx].set_title(title, fontsize=10, fontweight="bold", color=color, pad=8)

        for row_idx, data in enumerate(panels_data):
            vmax_d = max(0.3, float(data["gt_3d_depth"].max()))

            # Col 0: Helios Raytrace Reference
            if data["im_helios"] is not None:
                axes[row_idx, 0].imshow(data["im_helios"])
            else:
                axes[row_idx, 0].text(0.5, 0.5, "Reference\nN/A", color="#94a3b8", ha="center", va="center")
            axes[row_idx, 0].set_ylabel(f"DAP {data['dap']}\n({data['active_organs']} organs)", fontsize=11, color="#fbbf24", labelpad=8)
            axes[row_idx, 0].set_xticks([])
            axes[row_idx, 0].set_yticks([])

            # Col 1: Drone RGB 2x2 Pyramid
            rgb_2x2 = make_2x2_rgb(data["input_img"])
            axes[row_idx, 1].imshow(rgb_2x2)
            axes[row_idx, 1].set_xticks([])
            axes[row_idx, 1].set_yticks([])

            # Col 2: Drone Depth 2x2 Pyramid
            depth_2x2 = make_2x2_depth(data["input_img"])
            axes[row_idx, 2].imshow(depth_2x2)
            axes[row_idx, 2].set_xticks([])
            axes[row_idx, 2].set_yticks([])

            # Col 3: Pred 3D Mesh
            axes[row_idx, 3].imshow(data["pred_rgb"])
            axes[row_idx, 3].set_xlabel(f"IoU: {data['iou']*100:.1f}%", fontsize=10, color="#4ade80")
            axes[row_idx, 3].set_xticks([])
            axes[row_idx, 3].set_yticks([])

            # Col 4: Pred 3D Depth
            axes[row_idx, 4].imshow(data["pred_depth"], cmap="viridis", vmin=0.0, vmax=vmax_d)
            axes[row_idx, 4].axis("off")

            # Col 5: Depth Error Heatmap
            diff = np.abs(data["pred_depth"] - data["gt_3d_depth"])
            axes[row_idx, 5].imshow(diff, cmap="magma", vmin=0.0, vmax=0.20)
            axes[row_idx, 5].set_xlabel(f"Depth MAE: {data['depth_mae']*100:.2f} cm", fontsize=10, color="#f87171")
            axes[row_idx, 5].axis("off")

            # Col 6: 3D Botanical Nodes (Front Elevation X-Z View)
            ax_node = axes[row_idx, 6]
            ax_node.set_facecolor("#181c24")
            ax_node.axhline(0, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.6, label="Soil Line (Z=0)")

            # Ground Truth stem spine & nodes (Cyan circles)
            if len(data["gt_nodes"]) > 0:
                sort_z = np.argsort(data["gt_nodes"][:, 2])
                if len(data["gt_nodes"]) > 1:
                    ax_node.plot(
                        data["gt_nodes"][sort_z, 0] * 100, data["gt_nodes"][sort_z, 2] * 100,
                        color="#00e5ff", linestyle="-", linewidth=2.8, alpha=0.8, zorder=5,
                        label="GT Stem Spine"
                    )
                ax_node.scatter(
                    data["gt_nodes"][:, 0] * 100, data["gt_nodes"][:, 2] * 100,
                    c="#00e5ff", s=130, edgecolors="white", linewidth=1.8,
                    label=f"GT Node (N={len(data['gt_nodes'])})", zorder=6
                )
                # Label highest GT botanical node (Apex)
                apex_idx = sort_z[-1]
                apex_x_cm = data["gt_nodes"][apex_idx, 0] * 100
                apex_z_cm = data["gt_nodes"][apex_idx, 2] * 100
                ax_node.text(
                    apex_x_cm - 0.25, apex_z_cm + 0.12,
                    f"GT Top ({apex_z_cm:.1f}cm)",
                    color="#38bdf8", fontsize=8, fontweight="bold", ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#00e5ff", alpha=0.85),
                    zorder=8
                )

            # Predicted 3D nodes (Magenta diamonds)
            if len(data["pred_nodes"]) > 0:
                ax_node.scatter(
                    data["pred_nodes"][:, 0] * 100, data["pred_nodes"][:, 2] * 100,
                    c="#f43f5e", marker="D", s=85, edgecolors="white", linewidth=1.4,
                    label=f"Pred Node (K={len(data['pred_nodes'])})", zorder=7
                )

            # Draw displacement error vectors
            if len(data["gt_nodes"]) > 0 and len(data["pred_nodes"]) > 0:
                for p in data["pred_nodes"]:
                    dists = np.linalg.norm(data["gt_nodes"] - p, axis=1)
                    g_near = data["gt_nodes"][np.argmin(dists)]
                    ax_node.plot(
                        [g_near[0] * 100, p[0] * 100], [g_near[2] * 100, p[2] * 100],
                        color="#94a3b8", linestyle=":", linewidth=1.1, alpha=0.7, zorder=4
                    )

            # Symmetrically center X-axis around 0 so plant stem sits prominently in the center
            all_x = []
            all_z = []
            if len(data["gt_nodes"]) > 0:
                all_x.append(data["gt_nodes"][:, 0] * 100)
                all_z.append(data["gt_nodes"][:, 2] * 100)
            if len(data["pred_nodes"]) > 0:
                all_x.append(data["pred_nodes"][:, 0] * 100)
                all_z.append(data["pred_nodes"][:, 2] * 100)

            if all_x:
                all_x_arr = np.concatenate(all_x)
                all_z_arr = np.concatenate(all_z)
                x_limit = max(float(np.abs(all_x_arr).max() * 1.3), 2.0)
                z_min = min(-1.0, float(all_z_arr.min() * 1.15))
                z_max = max(1.5, float(all_z_arr.max() * 1.35))
                ax_node.set_xlim(-x_limit, x_limit)
                ax_node.set_ylim(z_min, z_max)

            ax_node.set_xlabel(f"X Width [cm]\nRMSE: {data['node_rmse_cm']:.1f} cm", color="#38bdf8", fontsize=9, fontweight="bold")
            ax_node.set_ylabel("Z Height [cm]", color="#cbd5e1", fontsize=9)
            ax_node.tick_params(colors="#94a3b8", labelsize=8)
            ax_node.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
            ax_node.legend(facecolor="#12151a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=7.5, loc="upper right")

        panel_path = os.path.join(output_dir, f"hierarchical_self_consistency_epoch_{epoch:03d}.png")
        fig.savefig(panel_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        # Log to W&B if active
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "val/silhouette_iou": float(np.mean(ious)),
                "val/depth_mae": float(np.mean(depth_maes)),
                "val/peak_height_error": float(np.mean(peak_height_diffs)),
                "val/node_rmse_cm": float(mean_node_rmse),
                "val/self_consistency_panel": wandb.Image(panel_path),
            }, step=epoch)

    return {
        "val/silhouette_iou": float(np.mean(ious)) if ious else 0.0,
        "val/depth_mae": float(np.mean(depth_maes)) if depth_maes else 0.0,
        "val/peak_height_error": float(np.mean(peak_height_diffs)) if peak_height_diffs else 0.0,
        "val/node_rmse_cm": float(np.mean([d["node_rmse_cm"] for d in panels_data])) if panels_data else 0.0,
        "silhouette_iou": float(np.mean(ious)) if ious else 0.0,
        "depth_mae": float(np.mean(depth_maes)) if depth_maes else 0.0,
        "peak_height_error": float(np.mean(peak_height_diffs)) if peak_height_diffs else 0.0,
    }
