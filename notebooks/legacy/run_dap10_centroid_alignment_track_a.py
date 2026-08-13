#!/usr/bin/env python
"""Reproduce DAP 10 plant mask centroid realignment figure."""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import shift

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion_based.models.legacy.helios_geometry_track_a import HeliosPlantGeometryTorch
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer

out_dir = os.path.join(os.path.dirname(__file__), "output_dap_benchmark")
os.makedirs(out_dir, exist_ok=True)
xml_path = os.path.join(out_dir, "dap10_gt_0000_plant_0000.xml")
gt_img_path = os.path.join(out_dir, "dap10_gt_0000_rad.jpeg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
geom_torch = HeliosPlantGeometryTorch.from_xml(xml_path, device=device)

with torch.no_grad():
    torch_rgba = rasterizer(geom_torch, focus_plant=True, background="black")
torch_rgb = torch_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
torch_alpha = torch_rgba[0, 3].cpu().numpy()

gt_img = Image.open(gt_img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
gt_np = np.array(gt_img, dtype=np.float32) / 255.0
gt_mask = np.linalg.norm(gt_np - np.array([0.61, 0.58, 0.55]), axis=-1) > 0.1
pred_mask = torch_alpha > 0.05

def iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return inter / max(union, 1)

def centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (mask.shape[1] // 2, mask.shape[0] // 2)
    return (xs.mean(), ys.mean())

# Centroid alignment
gt_cx, gt_cy = centroid(gt_mask)
pred_cx, pred_cy = centroid(pred_mask)
cdx = int(round(gt_cx - pred_cx))
cdy = int(round(gt_cy - pred_cy))
centroid_aligned = shift(pred_mask.astype(np.float32), (cdy, cdx), order=0, mode="constant", cval=0) > 0.5
centroid_iou = iou(gt_mask, centroid_aligned)

# Optimal translation search
best = {"iou": -1, "dy": 0, "dx": 0}
for dy in range(-40, 41):
    for dx in range(-40, 41):
        shifted = shift(pred_mask.astype(np.float32), (dy, dx), order=0, mode="constant", cval=0) > 0.5
        val = iou(gt_mask, shifted)
        if val > best["iou"]:
            best = {"iou": val, "dy": dy, "dx": dx}
optimal_aligned = shift(pred_mask.astype(np.float32), (best["dy"], best["dx"]), order=0, mode="constant", cval=0) > 0.5
optimal_iou = best["iou"]

raw_iou = iou(gt_mask, pred_mask)
print(f"Raw IoU: {raw_iou:.4f}")
print(f"Centroid aligned: dy={cdy}, dx={cdx}, IoU={centroid_iou:.4f}")
print(f"Optimal aligned: dy={best['dy']}, dx={best['dx']}, IoU={optimal_iou:.4f}")

# Figure
fig, axes = plt.subplots(1, 5, figsize=(25, 5), facecolor="#0a0a0a")
for ax in axes:
    ax.set_facecolor("#0a0a0a")

axes[0].imshow(gt_np)
axes[0].set_title("1. Ground Truth Helios Image", color="white")
axes[0].axis("off")

axes[1].imshow(torch_rgb)
axes[1].set_title(f"2. Raw PyTorch Diff Renderer\n(IoU: {raw_iou:.4f})", color="white")
axes[1].axis("off")

overlay_c = np.zeros((256, 256, 3))
overlay_c[gt_mask] = [0, 0, 1]
overlay_c[centroid_aligned] = [0, 1, 0]
overlay_c[gt_mask & centroid_aligned] = [1, 1, 1]
axes[2].imshow(overlay_c)
axes[2].set_title(f"3. Centroid Aligned Overlay\n(Shift: dy={cdy}, dx={cdx} | IoU: {centroid_iou:.4f})", color="white")
axes[2].axis("off")

overlay_o = np.zeros((256, 256, 3))
overlay_o[gt_mask] = [0, 0, 1]
overlay_o[optimal_aligned] = [0, 1, 0]
overlay_o[gt_mask & optimal_aligned] = [1, 1, 1]
axes[3].imshow(overlay_o)
axes[3].set_title(f"4. Optimal Aligned Overlay\n(Shift: dy={best['dy']}, dx={best['dx']} | IoU: {optimal_iou:.4f})", color="white")
axes[3].axis("off")

axes[4].imshow(optimal_aligned, cmap="gray")
axes[4].set_title("5. Optimal Aligned PyTorch Mask", color="white")
axes[4].axis("off")

fig.suptitle(
    f"DAP 10 Plant Mask Realignment (Improved)\nRaw IoU: {raw_iou:.4f} -> Centroid Aligned: {centroid_iou:.4f} -> Optimal Aligned: {optimal_iou:.4f}",
    color="white", fontsize=14, fontweight="bold"
)
plt.tight_layout()
fig_path = os.path.join(out_dir, "dap10_centroid_aligned_pixel_match.png")
plt.savefig(fig_path, dpi=150, facecolor="#0a0a0a", bbox_inches="tight")
print(f"Saved: {fig_path}")
