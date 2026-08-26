"""
Single-image inverse rendering optimization via 18D OrganNode3D nodes.

Pipeline:
  XML → parser.get_all_organ_nodes() → (N, 18) nodes tensor  ← nn.Parameter
      → DifferentiableHeliosRenderer                         ← fully differentiable
      → loss.backward() → nodes_opt.grad (N, 18)
      → optimizer.step() updates nodes_opt
  (after optimization)
  nodes_opt → write_organ_nodes_to_xml() → optimized XML

This replaces the deprecated HeliosPlantGeometryTorch.from_xml() direct-geom path.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser, OrganNode3D
from diffusion_based.models.legacy.helios_xml_writer_track_a import write_organ_nodes_to_xml
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer


# ──────────────────────────────────────────────────────────────────
# Fast SSIM (no skimage dependency — avoids blocking)
# ──────────────────────────────────────────────────────────────────
def _fast_ssim(a: np.ndarray, b: np.ndarray) -> float:
    C1, C2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.var(), b.var()
    sig_ab = float(np.mean((a - mu_a) * (b - mu_b)))
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sig_a + sig_b + C2)
    return float(num / (den + 1e-12))


# ──────────────────────────────────────────────────────────────────
# Reconstruct OrganNode3D from optimized 18D vector
# ──────────────────────────────────────────────────────────────────
def _apply_optimized_vec(orig_node: OrganNode3D, vec: np.ndarray) -> OrganNode3D:
    """Copy orig_node and update its fields from an optimized 18D vector."""
    import copy
    node = copy.copy(orig_node)
    node.xyz         = vec[:3].tolist()
    node.length      = float(np.clip(vec[3], 1e-4, 5.0))
    node.radius      = float(np.clip(vec[4], 1e-4, 0.5))
    node.dir_xyz     = (vec[5:8] / (np.linalg.norm(vec[5:8]) + 1e-8)).tolist()
    # organ_onehot [8:14]: keep original organ type (don't let optimizer flip organ identity)
    node.existence   = float(np.clip(vec[16], 0.0, 1.0))
    node.head_radius = float(np.clip(vec[17], 0.0, 0.03))
    return node


def main():
    output_dir = os.path.join(repo_root, "notebooks", "output_dap30_verification")
    os.makedirs(output_dir, exist_ok=True)
    xml_path = os.path.join(output_dir, "dap30_gt_seed42_0000_plant_0000.xml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Running 18D inverse rendering optimization on device: {device}", flush=True)

    # ── 1. Parse GT XML → 19D nodes tensor ───────────────────────
    parser = HeliosXMLParser(xml_path)
    parser.parse()
    gt_organ_nodes = parser.get_all_organ_nodes()
    N = len(gt_organ_nodes)
    print(f"Parsed {N} organ nodes from GT XML", flush=True)

    gt_nodes_np = np.stack([n.to_vec() for n in gt_organ_nodes], axis=0)  # (N, 19)

    # ── 2. Renderer and GT target image ──────────────────────────
    IMAGE_SIZE = 64  # smaller to avoid OOM with 455 organ nodes
    rasterizer    = HeliosGeometryRasterizer(image_size=IMAGE_SIZE).to(device)
    diff_renderer = DifferentiableHeliosRenderer(rasterizer).to(device)

    # Render GT using 19D path
    gt_nodes_t = torch.tensor(gt_nodes_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 19)
    with torch.no_grad():
        target_rgba = diff_renderer(gt_nodes_t, focus_plant=True, background="black")
    target_rgb   = target_rgba[0, :3].permute(1, 2, 0).clip(0, 1)   # (H, W, 3) cuda
    tar_rgb_np   = target_rgb.cpu().numpy()

    # ── 3. Perturbed 19D nodes as nn.Parameter ───────────────────
    torch.manual_seed(123)
    perturb = gt_nodes_np.copy().astype(np.float32)
    # Perturb positions [0:3], length [3], radius [4], direction [5:8]
    perturb[:, :8]  += np.random.randn(N, 8).astype(np.float32) * 0.15
    # Perturb existence [16] to simulate unknown organ visibility
    perturb[:, 16]  = np.random.uniform(0.0, 1.0, N).astype(np.float32)
    # Keep organ one-hot [8:14], shoot_id [14], phytomer [15], parent_idx [18] fixed (structural)

    nodes_opt = nn.Parameter(torch.tensor(perturb, device=device))  # (N, 19)

    # ── 4. Optimizer ──────────────────────────────────────────────
    base_lr   = 0.01
    optimizer = optim.Adam([nodes_opt], lr=base_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250, 450], gamma=0.5)

    num_steps      = 500
    snapshot_steps = {0, 50, 100, 250, 500}
    log_interval   = 50
    history_losses, history_ssim, history_lrs = [], [], []
    history_images = []  # (step, np_img, loss, ssim)

    print(f"\n{'='*55}")
    print(f"STARTING 19D INVERSE RENDERING OPTIMIZATION ({num_steps} Steps)")
    print(f"  Parameter: nodes_opt  shape={nodes_opt.shape}  ({nodes_opt.numel()} scalars)")
    print(f"{'='*55}", flush=True)

    t0 = time.time()
    for step in range(num_steps + 1):
        optimizer.zero_grad()

        # Forward: (N,19) → DifferentiableHeliosRenderer → (B,4,H,W)
        rendered_rgba = diff_renderer(
            nodes_opt.unsqueeze(0),
            focus_plant=True, background="black",
        )
        rendered_rgb = rendered_rgba[0, :3].permute(1, 2, 0)  # (H,W,3)

        loss_rgb   = F.l1_loss(rendered_rgb, target_rgb) + F.mse_loss(rendered_rgb, target_rgb)
        loss_alpha = F.mse_loss(rendered_rgba[0, 3], target_rgba[0, 3])
        total_loss = loss_rgb + 2.0 * loss_alpha

        if step < num_steps:
            total_loss.backward()
            # Freeze organ one-hot [8:14], shoot_id [14], phytomer_idx [15], parent_idx [18]
            # so optimizer cannot change graph structure or organ identity
            if nodes_opt.grad is not None:
                nodes_opt.grad[:, 8:16] = 0.0
                nodes_opt.grad[:, 18] = 0.0
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            cur_np = rendered_rgb.detach().cpu().numpy().clip(0, 1)

        ssim_val = _fast_ssim(cur_np, tar_rgb_np)
        history_losses.append(float(total_loss.item()))
        history_ssim.append(float(ssim_val))
        history_lrs.append(float(optimizer.param_groups[0]["lr"]))

        if step in snapshot_steps:
            history_images.append((step, cur_np, float(total_loss.item()), ssim_val))

        if step % log_interval == 0 or step == num_steps:
            mae = float(np.mean(np.abs(cur_np - tar_rgb_np)))
            print(
                f"Step {step:03d}/{num_steps} | Loss: {total_loss.item():.6f} | "
                f"SSIM: {ssim_val:.4f} | MAE: {mae:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

    total_time = time.time() - t0
    print(f"\nOptimization finished in {total_time:.1f}s", flush=True)

    # ── 5. Save optimized XML via write_organ_nodes_to_xml ────────
    opt_nodes_np = nodes_opt.detach().cpu().numpy()  # (N, 18)
    opt_organ_nodes = [
        _apply_optimized_vec(orig, opt_nodes_np[i])
        for i, orig in enumerate(gt_organ_nodes)
    ]
    optimized_xml = os.path.join(output_dir, "dap30_18d_optimized.xml")
    write_organ_nodes_to_xml(opt_organ_nodes, optimized_xml, plant_age=30)
    print(f"Saved optimized XML → {optimized_xml}", flush=True)

    # ── 6. Save metrics JSON ──────────────────────────────────────
    metrics = {
        "pipeline":        "XML → 18D nodes → DifferentiableHeliosRenderer",
        "num_steps":       num_steps,
        "total_time_sec":  total_time,
        "device":          str(device),
        "num_organ_nodes": N,
        "num_parameters":  int(nodes_opt.numel()),
        "final_loss":      history_losses[-1],
        "final_ssim":      history_ssim[-1],
        "losses":          history_losses,
        "ssims":           history_ssim,
        "learning_rates":  history_lrs,
        "snapshot_steps":  sorted(snapshot_steps),
    }
    metrics_path = os.path.join(output_dir, "18d_inverse_optimization_500_steps_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}", flush=True)

    # ── 7. Visualization figure ───────────────────────────────────
    BG = "#0d0f1a"
    fig = plt.figure(figsize=(20, 9), dpi=160, facecolor=BG)
    gs  = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25)

    def _iax(pos, img, title, color):
        ax = fig.add_subplot(pos)
        ax.imshow(img); ax.axis("off"); ax.set_facecolor(BG)
        ax.set_title(title, color=color, fontsize=10, fontweight="bold")
        return ax

    _iax(gs[0, 0], tar_rgb_np,
         "GT Target (18D Render)\nDAP 30, Seed=42", "white")

    for col, (step_n, img, loss_v, ssim_v) in enumerate(history_images[:3]):
        mae_v = float(np.mean(np.abs(img - tar_rgb_np)))
        _iax(gs[0, col + 1], img,
             f"Step {step_n:03d}\nLoss={loss_v:.4f}  SSIM={ssim_v:.4f}  MAE={mae_v:.5f}",
             "#38ef7d")

    for col, (step_n, img, loss_v, ssim_v) in enumerate(history_images[3:5]):
        mae_v = float(np.mean(np.abs(img - tar_rgb_np)))
        clr   = "#11998e" if step_n == 500 else "#38ef7d"
        _iax(gs[1, col], img,
             f"Step {step_n:03d}\nLoss={loss_v:.4f}  SSIM={ssim_v:.4f}  MAE={mae_v:.5f}", clr)

    # Loss + SSIM dual-axis
    ax_loss = fig.add_subplot(gs[1, 2])
    ax_loss.set_facecolor("#1a1d29")
    ax_loss.plot(history_losses, color="#ff4b5c", lw=2, label="Loss")
    for ms in [100, 250, 450]:
        ax_loss.axvline(ms, color="yellow", ls=":", alpha=0.5, lw=1)
    ax_loss.set_title("Loss / SSIM Convergence", color="white", fontsize=10, fontweight="bold")
    ax_loss.set_xlabel("Step", color="white", fontsize=9)
    ax_loss.set_ylabel("Loss", color="#ff4b5c", fontsize=9)
    ax_loss.tick_params(colors="white")
    ax_loss.grid(True, color="#333344", ls="--", alpha=0.5)
    ax2 = ax_loss.twinx()
    ax2.plot(history_ssim, color="#00d2ff", lw=2, ls="-.")
    ax2.set_ylabel("SSIM", color="#00d2ff", fontsize=9)
    ax2.tick_params(colors="white")

    # Pixel diff heatmap
    ax_diff = fig.add_subplot(gs[1, 3])
    diff_map = np.abs(history_images[-1][1] - tar_rgb_np).mean(axis=-1)
    im = ax_diff.imshow(diff_map, cmap="inferno", vmin=0, vmax=0.25)
    final_mae = float(np.mean(diff_map))
    ax_diff.set_title(f"Final Diff Map (Step 500)\nMAE={final_mae:.5f}", color="#ffdd59", fontsize=10, fontweight="bold")
    ax_diff.axis("off")
    cb = fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors="white")

    fig.suptitle(
        "18D Inverse Optimization: XML → (N,18) nodes_opt → DifferentiableHeliosRenderer → loss.backward()",
        color="white", fontsize=13, fontweight="bold", y=1.0,
    )

    fig_path = os.path.join(output_dir, "18d_inverse_optimization_500_steps.png")
    plt.savefig(fig_path, facecolor=BG, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"Saved figure → {fig_path}", flush=True)


if __name__ == "__main__":
    main()
