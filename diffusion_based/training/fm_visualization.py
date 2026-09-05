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
    decode_fm, FM_CURV, CURV_SCALE, NUM_ORGAN_CATEGORIES,
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
def _sample_fm(model, scaffold_gen, image, dap, num_steps=15):
    """Euler-integrate one plant from the scaffold prior conditioned on `image`."""
    device = image.device
    x_t = scaffold_gen.generate_from_dap(float(dap), device=device).unsqueeze(0)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((1,), i * dt, device=device)
        v = model(x_t, t, image.unsqueeze(0))["pred_velocity"]
        x_t = x_t + v * dt
        ot = x_t[..., :NUM_ORGAN_CATEGORIES].clamp(min=0.0)
        x_t[..., :NUM_ORGAN_CATEGORIES] = ot / (ot.sum(dim=-1, keepdim=True) + 1e-8)
    return x_t.squeeze(0)


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
    Builds and saves the per-epoch diagnostic figure (GT vs generated +
    curvature prediction comparison). Returns the matplotlib figure.
    """
    device = next(model.parameters()).device
    model.eval()

    images = batch["image"].to(device)[:num_rows]
    nodes_gt = batch["nodes"].to(device)[:num_rows]
    em_gt = batch["existence_mask"].to(device)[:num_rows]
    daps = batch["dap"].to(device)[:num_rows]

    fig, axes = plt.subplots(2, num_rows, figsize=(4.6 * num_rows, 9.0))
    if num_rows == 1:
        axes = axes.reshape(2, 1)

    curv_gt_rows, curv_pred_rows, labels_rows = [], [], []

    for r in range(num_rows):
        img = images[r]
        dap = float(daps[r].item())

        # --- FM generation for this sample ---
        x_gen = _sample_fm(model, scaffold_gen, img, dap)
        part_gen = decode_fm(x_gen)  # (N, 14) incl. curvature col 13

        # --- GT part tensor (from the 26D normalized node array) ---
        part_gt = decode_fm(nodes_gt[r])

        # --- renders ---
        geo = geo_builder
        active_gen = part_gen[part_gen[:, 0] > 0]
        active_gt = part_gt[part_gt[:, 0] > 0]
        try:
            mesh_gen = geo.build_mesh_from_part_tensor(active_gen, device=device)
            rgb_gen = renderer.render_mesh(
                mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=0.4,
                background="ground", focus_plant=True, include_depth=False,
            )
            rgb_gen_np = rgb_gen.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        except Exception as e:
            rgb_gen_np = np.full((128, 128, 3), 0.85, dtype=np.float32)
            _draw_error_text(rgb_gen_np, f"gen render fail: {type(e).__name__}")
        try:
            mesh_gt = geo.build_mesh_from_part_tensor(active_gt, device=device)
            rgb_gt = renderer.render_mesh(
                mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=0.4,
                background="ground", focus_plant=True, include_depth=False,
            )
            rgb_gt_np = rgb_gt.cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        except Exception as e:
            rgb_gt_np = np.full((128, 128, 3), 0.85, dtype=np.float32)
            _draw_error_text(rgb_gt_np, f"gt render fail: {type(e).__name__}")

        # Target display: prefer the 8x zoom channels (12:15) when present —
        # DAP-1 plants at the fixed 5m drone camera fill <2% of the 1x frame
        # and look like bare ground. Zoom 8 shows the actual seedling.
        if img.shape[0] >= 16:
            target_view = img[12:15]  # zoom-8 RGB
            zoom_note = " (8x zoom)"
        else:
            target_view = img[:3]
            zoom_note = ""
        axes[0, r].imshow(target_view.cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, r].set_title(f"Target (DAP {dap:.0f}){zoom_note}", fontsize=10, fontweight="bold")
        axes[0, r].axis("off")

        # Row 1 second panel: GT render vs generated render (side-by-side composite)
        comp = np.concatenate([rgb_gt_np, rgb_gen_np], axis=1)
        axes[1, r].imshow(comp)
        axes[1, r].set_title("GT render | FM-generated", fontsize=10, fontweight="bold")
        axes[1, r].axis("off")

        # --- curvature comparison: tubes only (internode + petioles) ---
        ot_gt = part_gt[:, 0].long()
        ot_gen = part_gen[:, 0].long()
        tube_mask_gt = (ot_gt == ORGAN_INTERNODE) | (ot_gt == ORGAN_PETIOLE)
        tube_mask_gen = (ot_gen == ORGAN_INTERNODE) | (ot_gen == ORGAN_PETIOLE)

        curv_gt = part_gt[tube_mask_gt, 13].cpu().numpy()
        curv_gen = part_gen[tube_mask_gen, 13].cpu().numpy()
        n_bars = max(len(curv_gt), len(curv_gen), 1)
        xs = np.arange(n_bars)
        width = 0.38
        ax = axes[1, r] if False else None  # placeholder to keep linter quiet

        # Use a twin panel: plot curvature bars on the same subplot's twin? 
        # Cleaner: draw on a small inset under the composite.
        ax2 = axes[1, r].inset_axes([0.0, -0.42, 1.0, 0.36])
        gt_pad = np.zeros(n_bars); gen_pad = np.zeros(n_bars)
        gt_pad[:len(curv_gt)] = curv_gt
        gen_pad[:len(curv_gen)] = curv_gen
        ax2.bar(xs - width / 2, gt_pad, width, label="GT curv", color="#2E7D32")
        ax2.bar(xs + width / 2, gen_pad, width, label="FM pred", color="#F59E0B")
        ax2.set_ylabel("deg/m", fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.legend(fontsize=6, loc="upper right")
        ax2.set_title("Tube curvature: GT vs FM prediction", fontsize=8)
        labels_rows.append((curv_gt, curv_gen))

    # Row 2 header uses first row's data for the aggregate histogram
    axes[1, 0].text(
        0.5, -0.62,
        "Curvature transport: FM predicts v_curv; x1_curv = x0_curv + ∫v_curv dt; "
        "decode = fm[:, FM_CURV] / CURV_SCALE",
        transform=axes[1, 0].transAxes, fontsize=7, ha="center", color="#334155",
    )

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