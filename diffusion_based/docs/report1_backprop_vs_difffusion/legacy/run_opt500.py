"""
500-step DAP-30 direct backpropagation inverse rendering optimization.
Self-contained: inlines SSIM so nothing can block.
Saves:
  - output_dap30_verification/opt500_metrics.json
  - output_dap30_verification/opt500_figure.png
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from diffusion_based.models.legacy.helios_geometry_track_a import HeliosPlantGeometryTorch
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer

output_dir = os.path.join(repo_root, "notebooks", "output_dap30_verification")
xml_path   = os.path.join(output_dir, "dap30_gt_seed42_0000_plant_0000.xml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
assert os.path.exists(xml_path), f"XML not found: {xml_path}"


# ---------- fast SSIM (pure numpy, no skimage) ----------
def _ssim_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Simple sliding-window SSIM approximation (no blocking)."""
    C1, C2 = 0.01**2, 0.03**2
    mu_a = a.mean(); mu_b = b.mean()
    sig_a = a.var(); sig_b = b.var()
    sig_ab = float(np.mean((a - mu_a) * (b - mu_b)))
    num = (2*mu_a*mu_b + C1) * (2*sig_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sig_a + sig_b + C2)
    return float(num / (den + 1e-12))


# 1. Ground-truth target
rasterizer = HeliosGeometryRasterizer(image_size=128).to(device)
geom_gt = HeliosPlantGeometryTorch.from_xml(xml_path, device=device)
with torch.no_grad():
    target_rgba = rasterizer(geom_gt, focus_plant=True, background="black")
target_rgb = target_rgba[0, :3].permute(1, 2, 0).detach()      # (H,W,3) cuda tensor
target_rgb_np = target_rgb.cpu().numpy().clip(0, 1)

# 2. Perturbed initial geometry
geom_opt = HeliosPlantGeometryTorch.from_xml(xml_path, device=device)
geom_opt.leaf_verts_base = nn.Parameter(geom_opt.leaf_verts_base.clone())
geom_opt.tube_verts_base = nn.Parameter(geom_opt.tube_verts_base.clone())
geom_opt.ell_centers      = nn.Parameter(geom_opt.ell_centers.clone())

torch.manual_seed(123)
geom_opt.leaf_scales.data.uniform_(0.2, 2.0)
geom_opt.tube_scales.data.uniform_(0.2, 2.0)
if geom_opt.leaf_verts_base.numel() > 0:
    geom_opt.leaf_verts_base.data += torch.randn_like(geom_opt.leaf_verts_base) * 0.25
if geom_opt.tube_verts_base.numel() > 0:
    geom_opt.tube_verts_base.data += torch.randn_like(geom_opt.tube_verts_base) * 0.25
if geom_opt.ell_centers.numel() > 0:
    geom_opt.ell_centers.data += torch.randn_like(geom_opt.ell_centers) * 0.25
if geom_opt.leaf_existence.numel() > 0:
    geom_opt.leaf_existence.data.uniform_(0.0, 1.0)
if geom_opt.tube_existence.numel() > 0:
    geom_opt.tube_existence.data.uniform_(0.0, 1.0)
if geom_opt.bud_existence.numel() > 0:
    geom_opt.bud_existence.data.uniform_(0.0, 1.0)

# 3. Optimizer + LR scheduler
base_lr   = 0.03
optimizer = optim.Adam(geom_opt.parameters(), lr=base_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250, 450], gamma=0.5)

num_steps      = 500
snapshot_steps = {0, 50, 100, 250, 500}
log_interval   = 50

history_losses, history_ssim, history_lrs = [], [], []
snapshots = {}   # step -> np.ndarray (H,W,3)

print(f"Starting 500-step optimization …", flush=True)
t0 = time.time()

for step in range(num_steps + 1):
    optimizer.zero_grad()
    rendered_rgba = rasterizer(geom_opt, focus_plant=True, background="black")
    rendered_rgb  = rendered_rgba[0, :3].permute(1, 2, 0)   # (H,W,3)

    loss_rgb   = F.l1_loss(rendered_rgb, target_rgb) + F.mse_loss(rendered_rgb, target_rgb)
    loss_alpha = F.mse_loss(rendered_rgba[0, 3], target_rgba[0, 3])
    total_loss = loss_rgb + 2.0 * loss_alpha

    if step < num_steps:
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        cur_np = rendered_rgb.detach().cpu().numpy().clip(0, 1)

    ssim_val = _ssim_fast(cur_np, target_rgb_np)
    history_losses.append(float(total_loss.item()))
    history_ssim.append(float(ssim_val))
    history_lrs.append(float(optimizer.param_groups[0]["lr"]))

    if step in snapshot_steps:
        snapshots[step] = cur_np

    if step % log_interval == 0 or step == num_steps:
        mae = float(np.mean(np.abs(cur_np - target_rgb_np)))
        print(f"Step {step:03d}/{num_steps} | Loss: {total_loss.item():.6f} | SSIM: {ssim_val:.4f} | MAE: {mae:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

total_time = time.time() - t0
print(f"Finished in {total_time:.1f}s", flush=True)

# 4. Save metrics JSON
metrics = {
    "num_steps":      num_steps,
    "total_time_sec": total_time,
    "device":         str(device),
    "snapshot_steps": sorted(snapshot_steps),
    "losses":  history_losses,
    "ssims":   history_ssim,
    "lrs":     history_lrs,
    "step_metrics": {
        str(s): {
            "loss": history_losses[s],
            "ssim": history_ssim[s],
            "mae":  float(np.mean(np.abs(snapshots[s] - target_rgb_np))) if s in snapshots else None,
            "lr":   history_lrs[s],
        }
        for s in sorted(snapshot_steps)
    },
}
json_path = os.path.join(output_dir, "opt500_metrics.json")
with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved metrics → {json_path}", flush=True)

# 5. Build 2-row figure
fig = plt.figure(figsize=(20, 9), dpi=200)
fig.patch.set_facecolor("#0f111a")
gs  = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25)

def _imax(ax, img, title, color="white"):
    ax.imshow(img); ax.axis("off"); ax.set_facecolor("#0f111a")
    ax.set_title(title, color=color, fontsize=10, fontweight="bold")

_imax(fig.add_subplot(gs[0, 0]), target_rgb_np,
      "Ground Truth Target\n(DAP 30, Seed 42)", "white")

for col, st in enumerate([0, 50, 100]):
    l, s, m = metrics["step_metrics"][str(st)]["loss"], \
              metrics["step_metrics"][str(st)]["ssim"], \
              metrics["step_metrics"][str(st)]["mae"]
    _imax(fig.add_subplot(gs[0, col+1]), snapshots[st],
          f"Step {st:03d}\nLoss={l:.4f}  SSIM={s:.4f}  MAE={m:.5f}", "#38ef7d")

for col, st in enumerate([250, 500]):
    l, s, m = metrics["step_metrics"][str(st)]["loss"], \
              metrics["step_metrics"][str(st)]["ssim"], \
              metrics["step_metrics"][str(st)]["mae"]
    _imax(fig.add_subplot(gs[1, col]), snapshots[st],
          f"Step {st:03d}\nLoss={l:.4f}  SSIM={s:.4f}  MAE={m:.5f}",
          "#11998e" if st == 500 else "#38ef7d")

# Loss curve
ax_loss = fig.add_subplot(gs[1, 2])
ax_loss.set_facecolor("#1a1d29")
ax_loss.plot(history_losses, color="#ff4b5c", lw=2, label="Loss")
for ms in [100, 250, 450]:
    ax_loss.axvline(ms, color="yellow", ls=":", alpha=0.5)
ax_loss.set_title("Loss Convergence Curve", color="white", fontsize=11, fontweight="bold")
ax_loss.set_xlabel("Step", color="white", fontsize=9)
ax_loss.set_ylabel("Loss", color="#ff4b5c", fontsize=9)
ax_loss.tick_params(colors="white"); ax_loss.grid(True, color="#333344", ls="--", alpha=0.5)

ax_ssim2 = ax_loss.twinx()
ax_ssim2.plot(history_ssim, color="#00d2ff", lw=2, ls="-.")
ax_ssim2.set_ylabel("SSIM", color="#00d2ff", fontsize=9)
ax_ssim2.tick_params(colors="white")

# Pixel diff heatmap
ax_diff = fig.add_subplot(gs[1, 3])
diff_map = np.abs(snapshots[500] - target_rgb_np).mean(axis=-1)
im = ax_diff.imshow(diff_map, cmap="inferno", vmin=0, vmax=0.25)
final_mae = float(np.mean(diff_map))
ax_diff.set_title(f"Final Diff Map (Step 500)\nMAE={final_mae:.5f}", color="#ffdd59", fontsize=10, fontweight="bold")
ax_diff.axis("off")
cb = fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors="white")

fig.suptitle("Direct Backpropagation Inverse Optimization — DAP 30 Plant (500 Steps)",
             color="white", fontsize=14, fontweight="bold", y=0.99)

fig_path = os.path.join(output_dir, "opt500_figure.png")
plt.savefig(fig_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"Saved figure → {fig_path}", flush=True)
