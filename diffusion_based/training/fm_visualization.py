"""
Per-epoch visualization + optional W&B logging for Part Flow Matching training.

Produces a 2-row diagnostic panel per validation epoch:
  Row 1: Target image | GT render | Generated plant render | Organ-type overlay
  Row 2: Curvature comparison — GT curvature vs FM-predicted curvature per organ
         (bar chart) + velocity-field norm image.

Curvature prediction mechanism (how it works):
  The FM node layout is 26D: [one-hot(13), base*20(3), rot6d(6), scale*50(3),
  curvature/60(1)]. The model predicts the velocity v(x_t, t) per slot; the
  curvature CHANNEL of that velocity transports the normalized curvature
  component from the prior (scaffold prior: near-0) to the data value:
      curvature_pred = x1_curv = x0_curv + integral(v_curv dt)
  At t=1 the slot's curvature channel holds the predicted normalized curvature,
  decoded by decode_fm() as fm[:, FM_CURV] / CURV_SCALE (deg/m).
"""

import math
import os
from typing import Dict, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_based.dataset.part_array_dataset import (
    decode_fm, decode_fm_twostage, FM_CURV, CURV_SCALE, NUM_ORGAN_CATEGORIES,
    FM_BASE_START,
)
from diffusion_based.models.plant_organ_array import (
    ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

REPO_ROOT = "/home/lion397/codes/image-to-l-system"
OT_NAMES = {0: "NONE", 3: "INODE", 4: "PET", 5: "LEAF", 6: "PED", 11: "FRUIT"}


def _draw_error_text(rgb_np: np.ndarray, msg: str) -> None:
    """Paints a visible error message into a failed render slot (instead of black)."""
    h, w = rgb_np.shape[:2]
    for y in range(h // 3, h * 2 // 3):
        rgb_np[y, w // 5: w * 4 // 5] = 0.2
    step = max(1, len(msg) // 24)
    short = msg[:60]
    # crude centered text via block letters is overkill; mark with a border + label row
    rgb_np[8:14, 8: w - 8] = 0.9
    rgb_np[3, 10: 10 + min(len(short), w - 20) * 3] = 0.0


@torch.no_grad()
def _sample_fm(model, scaffold_gen, image, dap, num_steps=30):
    """Integrate one plant from the scaffold prior conditioned on `image` via 2nd-order Heun solver."""
    device = image.device
    raw = model.module if hasattr(model, "module") else model
    node_dim = getattr(raw, "node_dim", 26)
    tgt_size = getattr(raw, "image_size", 256)

    if image.shape[-1] != tgt_size:
        image = F.interpolate(image.unsqueeze(0), size=(tgt_size, tgt_size), mode="bilinear", align_corners=False).squeeze(0)

    scaf = scaffold_gen.generate_from_dap(float(dap), device=device)
    dt = 1.0 / num_steps

    if node_dim == 13:
        # Two-stage: integrate pure 13D geometry
        x_t = scaf[:, FM_BASE_START:].unsqueeze(0)  # (1, N, 13)
        dap_t = torch.tensor([float(dap)], device=device)
        for i in range(num_steps):
            t0 = torch.full((1,), i * dt, device=device)
            v0 = model(x_t, t0, image.unsqueeze(0), daps=dap_t)["pred_velocity"]
            x_next = x_t + v0 * dt
            if i + 1 < num_steps:
                t1 = torch.full((1,), (i + 1) * dt, device=device)
                v1 = model(x_next, t1, image.unsqueeze(0), daps=dap_t)["pred_velocity"]
                x_t = x_t + 0.5 * (v0 + v1) * dt
            else:
                x_t = x_next
        # Final pass at t=1 for discrete class logits
        t_final = torch.full((1,), 1.0, device=device)
        final_out = model(x_t, t_final, image.unsqueeze(0), daps=dap_t)
        geom = x_t.squeeze(0)  # (N, 13)
        logits = final_out.get("pred_type_logits", torch.zeros((geom.shape[0], 13), device=device)).squeeze(0)
        return decode_fm_twostage(geom, logits)
    else:
        # Legacy 26D
        x_t = scaf.unsqueeze(0)
        for i in range(num_steps):
            t = torch.full((1,), i * dt, device=device)
            v = model(x_t, t, image.unsqueeze(0))["pred_velocity"]
            x_t = x_t + v * dt
            ot = x_t[..., :NUM_ORGAN_CATEGORIES].clamp(min=0.0)
            x_t[..., :NUM_ORGAN_CATEGORIES] = ot / (ot.sum(dim=-1, keepdim=True) + 1e-8)
        return decode_fm(x_t.squeeze(0))


@torch.no_grad()
def render_epoch_panel(
    model,
    scaffold_gen,
    geo_builder,
    renderer,
    batch: Dict[str, torch.Tensor],
    epoch: int,
    global_step: int,
    out_dir: str = "docs/results/assets",
    num_rows: int = 3,
    wandb_run=None,
) -> Optional[plt.Figure]:
    """
    Builds and saves the per-epoch diagnostic figure showcasing:
      - All 4 zoom pyramid channels: 1.0x (global), 2.0x, 4.0x, 8.0x (organ-level)
      - Ground Truth 3D Organ Mesh Render
      - Flow Matching Generated 3D Organ Mesh Render
      - Tube Curvature (GT vs Prediction) quantitative distribution
    """
    device = next(model.parameters()).device
    model.eval()

    images = batch["image"].to(device)[:num_rows]
    nodes_gt = batch["nodes"].to(device)[:num_rows]
    em_gt = batch["existence_mask"].to(device)[:num_rows]
    daps = batch["dap"].to(device)[:num_rows]

    num_cols = 7
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25.0, 3.8 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)

    curv_gt_rows, curv_pred_rows, labels_rows = [], [], []

    for r in range(num_rows):
        img = images[r]
        dap = float(daps[r].item())

        # --- FM generation for this sample ---
        part_gen = _sample_fm(model, scaffold_gen, img, dap)

        # --- GT part tensor ---
        part_gt = decode_fm(nodes_gt[r])

        # --- Renders ---
        geo = geo_builder
        active_gen = part_gen[part_gen[:, 0] > 0]
        active_gt = part_gt[part_gt[:, 0] > 0]

        cam_h = 5.0
        zoom_f = 8.0 if dap <= 15.0 else 1.0
        ref_w = 1.2
        try:
            mesh_gen = geo.build_mesh_from_part_tensor(active_gen, device=device)
            rgb_gen = renderer.render_mesh(
                mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=cam_h,
                background="ground", focus_plant=True, include_depth=False,
                reference_window_size=ref_w, zoom_factor=zoom_f,
            )
            rgb_gen_np = rgb_gen.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        except Exception as e:
            rgb_gen_np = np.full((renderer.image_size, renderer.image_size, 3), 0.85, dtype=np.float32)
            _draw_error_text(rgb_gen_np, f"gen render fail: {type(e).__name__}")
        try:
            mesh_gt = geo.build_mesh_from_part_tensor(active_gt, device=device)
            rgb_gt = renderer.render_mesh(
                mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=cam_h,
                background="ground", focus_plant=True, include_depth=False,
                reference_window_size=ref_w, zoom_factor=zoom_f,
            )
            rgb_gt_np = rgb_gt.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        except Exception as e:
            rgb_gt_np = np.full((renderer.image_size, renderer.image_size, 3), 0.85, dtype=np.float32)
            _draw_error_text(rgb_gt_np, f"gt render fail: {type(e).__name__}")

        # --- Display All Pyramid Levels (Columns 0 to 3) ---
        pyramid_zooms = ["1.0x (Global)", "2.0x (Mid-Canopy)", "4.0x (Cluster)", "8.0x (Organ-Level)"]
        for p_idx, p_name in enumerate(pyramid_zooms):
            ax_p = axes[r, p_idx]
            if img.shape[0] >= 16:
                # 16-channel layout: 4 zooms * 4ch (RGB + Depth)
                ch_start = p_idx * 4
                p_rgb = img[ch_start:ch_start + 3].float()
                p_unnorm = (p_rgb * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                ax_p.imshow(p_unnorm)
            else:
                p_rgb = img[:3].float()
                p_unnorm = (p_rgb * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                ax_p.imshow(p_unnorm)
            if r == 0:
                ax_p.set_title(f"Pyramid {p_name}", fontsize=10, fontweight="bold")
            else:
                ax_p.set_title(f"Zoom {p_name.split()[0]}", fontsize=9)
            if p_idx == 0:
                ax_p.set_ylabel(f"DAP {dap:.0f}", fontsize=11, fontweight="bold")
            ax_p.set_xticks([])
            ax_p.set_yticks([])

        # --- Column 4: GT 3D Mesh Render ---
        ax_gt = axes[r, 4]
        ax_gt.imshow(rgb_gt_np)
        if r == 0:
            ax_gt.set_title("Ground Truth 3D Mesh", fontsize=10, fontweight="bold", color="#15803D")
        else:
            ax_gt.set_title(f"GT ({len(active_gt)} organs)", fontsize=9, color="#15803D")
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])

        # --- Column 5: FM Generated 3D Mesh Render ---
        ax_gen = axes[r, 5]
        ax_gen.imshow(rgb_gen_np)
        if r == 0:
            ax_gen.set_title("Flow Matching Generated", fontsize=10, fontweight="bold", color="#B45309")
        else:
            ax_gen.set_title(f"FM Pred ({len(active_gen)} organs)", fontsize=9, color="#B45309")
        ax_gen.set_xticks([])
        ax_gen.set_yticks([])

        # --- Column 6: Curvature Distribution Comparison ---
        ax_curv = axes[r, 6]
        ot_gt = part_gt[:, 0].long()
        ot_gen = part_gen[:, 0].long()
        tube_mask_gt = (ot_gt == ORGAN_INTERNODE) | (ot_gt == ORGAN_PETIOLE)
        tube_mask_gen = (ot_gen == ORGAN_INTERNODE) | (ot_gen == ORGAN_PETIOLE)

        curv_gt = part_gt[tube_mask_gt, 13].cpu().numpy()
        curv_gen = part_gen[tube_mask_gen, 13].cpu().numpy()
        n_bars = max(len(curv_gt), len(curv_gen), 1)
        xs = np.arange(min(n_bars, 40))
        width = 0.40

        gt_pad = np.zeros(len(xs)); gen_pad = np.zeros(len(xs))
        if len(curv_gt) > 0:
            gt_pad[:min(len(curv_gt), len(xs))] = curv_gt[:len(xs)]
        if len(curv_gen) > 0:
            gen_pad[:min(len(curv_gen), len(xs))] = curv_gen[:len(xs)]

        ax_curv.bar(xs - width / 2, gt_pad, width, label="GT curv", color="#2E7D32", alpha=0.85)
        ax_curv.bar(xs + width / 2, gen_pad, width, label="FM pred", color="#F59E0B", alpha=0.85)
        ax_curv.set_ylabel("deg/m", fontsize=8)
        ax_curv.tick_params(labelsize=7)
        if r == 0:
            ax_curv.legend(fontsize=7, loc="upper right")
            ax_curv.set_title("Tube Curvature", fontsize=10, fontweight="bold")
        else:
            ax_curv.set_title(f"Curv DAP {dap:.0f}", fontsize=8)
        labels_rows.append((curv_gt, curv_gen))

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(REPO_ROOT, out_dir, f"fm_curv_epoch_{epoch:03d}.png")
    latest_path = os.path.join(REPO_ROOT, out_dir, "fig_fm_curv_latest_eval.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.savefig(latest_path, dpi=150, bbox_inches="tight")

    if wandb_run is not None:
        wandb_run.log({
            "eval/panel": wandb.Image(fig),
            "curvature/gt_hist": wandb.Histogram(
                np.concatenate([g for g, _ in labels_rows]) if labels_rows else np.array([0.0])
            ),
            "curvature/pred_hist": wandb.Histogram(
                np.concatenate([p for _, p in labels_rows]) if labels_rows else np.array([0.0])
            ),
        }, step=global_step)

    model.train()
    return fig


@torch.no_grad()
def log_epoch_scalars(metrics: Dict[str, float], epoch: int, global_step: int, wandb_run=None):
    """Logs scalar losses to wandb if enabled."""
    if wandb_run is None:
        return
    wandb_run.log({
        "train/loss": metrics.get("loss", 0.0),
        "train/nan_skips": metrics.get("nan_skips", 0),
        "train/epoch": epoch,
    }, step=global_step)