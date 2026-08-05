import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# skimage for SSIM (installed via pip in this env)
from skimage.metrics import structural_similarity as ssim

# Ensure repo root is on path
repo_root = "/home/lion397/codes/image-to-l-system"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_geometry import (
    build_helios_geometry_from_xml,
    nodes_to_geometry,
    nodes_to_geometry_torch,
)
from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer
from diffusion_based.models.differentiable_pipeline import DifferentiableHeliosRenderer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
# Using the test_dap10 dataset because the main output/plot_0000_vis.jpeg is all-black
OUTPUT_DIR = "/home/lion397/codes/image-to-l-system/Digital-Crops/projects/syntheticdata_generation/build/output"
XML_FILE = os.path.join(OUTPUT_DIR, "plot_0000_plant_0000.xml")
REF_IMAGE = os.path.join(OUTPUT_DIR, "plot_0000_vis.jpeg")

# Camera parameters (matching RENDER_ALIGNMENT_DEBUG.md)
CAMERA_HEIGHT = 1.0
AZIMUTH_DEG = 0.0
DISTANCE_FROM_CENTER = 0.0
IMAGE_SIZE = 256

assert os.path.exists(XML_FILE), f"XML not found: {XML_FILE}"
assert os.path.exists(REF_IMAGE), f"Ref image not found: {REF_IMAGE}"
print(f"XML file: {XML_FILE}")
print(f"Ref image: {REF_IMAGE}")
ref_img = Image.open(REF_IMAGE).convert("RGB")
ref_np = np.array(ref_img, dtype=np.float32) / 255.0
print(f"Reference image size: {ref_img.size}, dtype: {ref_np.dtype}, range: [{ref_np.min():.3f}, {ref_np.max():.3f}]")

# Resize to match render size (256x256) for fair comparison
ref_img_256 = ref_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
ref_np_256 = np.array(ref_img_256, dtype=np.float32) / 255.0
print(f"Resized reference: {ref_img_256.size}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rasterizer = HeliosGeometryRasterizer(image_size=IMAGE_SIZE).to(device)
print(f"Rasterizer on device: {device}")
parser = HeliosXMLParser(XML_FILE)
parser.parse()
organ_nodes = parser.get_all_organ_nodes()
print(f"Parsed {len(organ_nodes)} organ nodes from XML")

# Build 15D node tensor
nodes_tensor = torch.stack([
    torch.tensor(n.to_15d(), dtype=torch.float32) for n in organ_nodes
]).unsqueeze(0).to(device)
parents = torch.tensor([n.parent_idx for n in organ_nodes], dtype=torch.long).unsqueeze(0).to(device)

print(f"nodes_tensor shape: {nodes_tensor.shape}")   # (1, N, 15)
print(f"parents shape: {parents.shape}")             # (1, N)

# --------------------------------------------------------------
# CRITICAL: nodes_to_geometry is NOT fully differentiable
# It calls .detach().cpu().numpy() internally.
# See diffusion_based/models/helios_geometry.py:849-922
# --------------------------------------------------------------
with torch.no_grad():
    t0 = time.time()
    tubes_15d, leaflets_15d, ellipsoids_15d = nodes_to_geometry(nodes_tensor, parents)
    nodes_geom_time = time.time() - t0
    print(f"nodes_to_geometry elapsed: {nodes_geom_time:.3f}s")
    print(f"  tubes: {len(tubes_15d[0])}, leaflets: {len(leaflets_15d[0])}, ellipsoids: {len(ellipsoids_15d[0])}")

    # Render with black background
    img_15d_black_rgba = rasterizer.render_numpy_geometry(
        tubes_15d[0], leaflets_15d[0], ellipsoids_15d[0],
        camera_height=CAMERA_HEIGHT,
        distance_from_center=DISTANCE_FROM_CENTER,
        azimuth_deg=AZIMUTH_DEG,
        focus_plant=True,
        background=None,
    )

    # Render with ground background
    img_15d_ground_rgba = rasterizer.render_numpy_geometry(
        tubes_15d[0], leaflets_15d[0], ellipsoids_15d[0],
        camera_height=CAMERA_HEIGHT,
        distance_from_center=DISTANCE_FROM_CENTER,
        azimuth_deg=AZIMUTH_DEG,
        focus_plant=True,
        background="ground",
    )

# Strip alpha channel for RGB comparison
img_15d_black = img_15d_black_rgba[..., :3]
img_15d_ground = img_15d_ground_rgba[..., :3]

print(f"15D black bg shape: {img_15d_black.shape}, range: [{img_15d_black.min():.3f}, {img_15d_black.max():.3f}]")
print(f"15D ground bg shape: {img_15d_ground.shape}, range: [{img_15d_ground.min():.3f}, {img_15d_ground.max():.3f}]")
geom = build_helios_geometry_from_xml(XML_FILE)
print(f"XML geometry: tubes={len(geom.tubes)}, leaflets={len(geom.leaflets)}, ellipsoids={len(geom.ellipsoids)}")

with torch.no_grad():
    # Render with black background
    img_xml_black_rgba = rasterizer.render_numpy_geometry(
        geom.tubes, geom.leaflets, geom.ellipsoids,
        camera_height=CAMERA_HEIGHT,
        distance_from_center=DISTANCE_FROM_CENTER,
        azimuth_deg=AZIMUTH_DEG,
        focus_plant=True,
        background=None,
    )

    # Render with ground background
    img_xml_ground_rgba = rasterizer.render_numpy_geometry(
        geom.tubes, geom.leaflets, geom.ellipsoids,
        camera_height=CAMERA_HEIGHT,
        distance_from_center=DISTANCE_FROM_CENTER,
        azimuth_deg=AZIMUTH_DEG,
        focus_plant=True,
        background="ground",
    )

# Strip alpha channel for RGB comparison
img_xml_black = img_xml_black_rgba[..., :3]
img_xml_ground = img_xml_ground_rgba[..., :3]

print(f"XML black bg shape: {img_xml_black.shape}, range: [{img_xml_black.min():.3f}, {img_xml_black.max():.3f}]")
print(f"XML ground bg shape: {img_xml_ground.shape}, range: [{img_xml_ground.min():.3f}, {img_xml_ground.max():.3f}]")

# ------------------------------------------------------------------
# NEW: 15D Torch Renderer (fully differentiable)
# ------------------------------------------------------------------
print("--- Running differentiable torch renderer ---")
diff_renderer = DifferentiableHeliosRenderer(rasterizer).to(device)

with torch.no_grad():
    img_torch_rgba = diff_renderer(
        nodes_tensor,
        parents,
        camera_height=CAMERA_HEIGHT,
        distance_from_center=DISTANCE_FROM_CENTER,
        azimuth_deg=AZIMUTH_DEG,
        focus_plant=True,
        background=None,
    )

# Convert from (B, 4, H, W) to (H, W, 4) numpy
img_torch_np = img_torch_rgba[0].permute(1, 2, 0).detach().cpu().numpy()
img_torch_black = img_torch_np[..., :3]

print(f"Torch black bg shape: {img_torch_black.shape}, range: [{img_torch_black.min():.3f}, {img_torch_black.max():.3f}]")

def compute_metrics(ref, pred, multichannel=True):
    """Compute MAE and SSIM between two [0,1] RGB images."""
    mae = float(np.mean(np.abs(ref - pred)))
    # SSIM expects images in range [0, 1]
    ssim_val = ssim(ref, pred, channel_axis=-1 if multichannel else None,
                    data_range=1.0)
    return mae, ssim_val

def compute_mask_iou(ref, pred, ref_thresh=0.65, pred_thresh=0.08):
    """
    Compute IoU of plant masks.
    Helios ref plant region ≈ pixels darker than ref_thresh in grayscale.
    Python render plant region ≈ pixels brighter than pred_thresh in grayscale (black bg).
    """
    ref_gray = ref.mean(axis=-1)
    pred_gray = pred.mean(axis=-1)
    
    ref_mask = ref_gray < ref_thresh      # darker = plant
    pred_mask = pred_gray > pred_thresh   # brighter = plant (black bg)
    
    inter = np.logical_and(ref_mask, pred_mask).sum()
    union = np.logical_or(ref_mask, pred_mask).sum()
    iou = float(inter) / float(union) if union > 0 else 0.0
    precision = float(inter) / float(pred_mask.sum()) if pred_mask.sum() > 0 else 0.0
    recall = float(inter) / float(ref_mask.sum()) if ref_mask.sum() > 0 else 0.0
    return iou, precision, recall

print("Metrics utilities ready.")
results = {}

# Black background variants
mae, sm = compute_metrics(ref_np_256, img_15d_black)
iou, prec, rec = compute_mask_iou(ref_np_256, img_15d_black)
results["15D black bg"] = {"MAE": mae, "SSIM": sm, "IoU": iou, "Precision": prec, "Recall": rec}

mae, sm = compute_metrics(ref_np_256, img_xml_black)
iou, prec, rec = compute_mask_iou(ref_np_256, img_xml_black)
results["XML black bg"] = {"MAE": mae, "SSIM": sm, "IoU": iou, "Precision": prec, "Recall": rec}

# Ground background variants
mae, sm = compute_metrics(ref_np_256, img_15d_ground)
iou2, prec2, rec2 = compute_mask_iou(ref_np_256, img_15d_ground)
results["15D ground bg"] = {"MAE": mae, "SSIM": sm, "IoU": iou2, "Precision": prec2, "Recall": rec2}

mae, sm = compute_metrics(ref_np_256, img_xml_ground)
iou2, prec2, rec2 = compute_mask_iou(ref_np_256, img_xml_ground)
results["XML ground bg"] = {"MAE": mae, "SSIM": sm, "IoU": iou2, "Precision": prec2, "Recall": rec2}

# Torch renderer (black bg)
mae, sm = compute_metrics(ref_np_256, img_torch_black)
iou_t, prec_t, rec_t = compute_mask_iou(ref_np_256, img_torch_black)
results["15D torch black bg"] = {"MAE": mae, "SSIM": sm, "IoU": iou_t, "Precision": prec_t, "Recall": rec_t}

print("─" * 70)
print(f"{'Variant':<20} {'MAE':>8} {'SSIM':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8}")
print("─" * 70)
for name, vals in results.items():
    print(f"{name:<20} {vals['MAE']:>8.3f} {vals['SSIM']:>8.3f} {vals['IoU']:>8.3f} {vals['Precision']:>10.3f} {vals['Recall']:>8.3f}")
print("─" * 70)

print("\nExpected values from RENDER_ALIGNMENT_DEBUG.md (different input):")
print("  black bg   : MAE=0.319, SSIM=0.286")
print("  ground bg  : MAE=0.237, SSIM=0.439")
print("  XML render : MAE=0.319, SSIM=0.286")
print("  15D nodes  : MAE=0.295, SSIM=0.304")
fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))

titles = [
    "Helios ref",
    f"black bg\nMAE={results['XML black bg']['MAE']:.3f}, SSIM={results['XML black bg']['SSIM']:.3f}",
    f"ground bg\nMAE={results['XML ground bg']['MAE']:.3f}, SSIM={results['XML ground bg']['SSIM']:.3f}",
    f"XML render\nMAE={results['XML black bg']['MAE']:.3f}, SSIM={results['XML black bg']['SSIM']:.3f}",
    f"15D nodes\nMAE={results['15D black bg']['MAE']:.3f}, SSIM={results['15D black bg']['SSIM']:.3f}",
    f"15D torch\nMAE={results['15D torch black bg']['MAE']:.3f}, SSIM={results['15D torch black bg']['SSIM']:.3f}",
]
images = [ref_np_256, img_xml_black, img_xml_ground, img_xml_black, img_15d_black, img_torch_black]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(title, fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(repo_root, "notebooks", "render_comparison_5panel.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/render_comparison_5panel.png")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

axes[0].imshow(np.clip(ref_np_256, 0, 1))
axes[0].set_title("Helios ref", fontsize=12)
axes[0].axis("off")

axes[1].imshow(np.clip(img_15d_black, 0, 1))
axes[1].set_title(f"15D nodes (black bg)\nMAE={results['15D black bg']['MAE']:.3f}, SSIM={results['15D black bg']['SSIM']:.3f}", fontsize=11)
axes[1].axis("off")

axes[2].imshow(np.clip(img_15d_ground, 0, 1))
axes[2].set_title(f"15D nodes (ground bg)\nMAE={results['15D ground bg']['MAE']:.3f}, SSIM={results['15D ground bg']['SSIM']:.3f}", fontsize=11)
axes[2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(repo_root, "notebooks", "background_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/background_comparison.png")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

diff_black = np.abs(ref_np_256 - img_xml_black).mean(axis=-1)
diff_ground = np.abs(ref_np_256 - img_xml_ground).mean(axis=-1)
diff_15d_black = np.abs(ref_np_256 - img_15d_black).mean(axis=-1)
diff_15d_ground = np.abs(ref_np_256 - img_15d_ground).mean(axis=-1)

vmax = max(diff_black.max(), diff_ground.max(), diff_15d_black.max(), diff_15d_ground.max())

axes[0].imshow(diff_black, cmap="hot", vmin=0, vmax=vmax)
axes[0].set_title("XML black bg diff", fontsize=11)
axes[0].axis("off")

axes[1].imshow(diff_ground, cmap="hot", vmin=0, vmax=vmax)
axes[1].set_title("XML ground bg diff", fontsize=11)
axes[1].axis("off")

axes[2].imshow(diff_15d_black, cmap="hot", vmin=0, vmax=vmax)
axes[2].set_title("15D black bg diff", fontsize=11)
axes[2].axis("off")

axes[3].imshow(diff_15d_ground, cmap="hot", vmin=0, vmax=vmax)
axes[3].set_title("15D ground bg diff", fontsize=11)
axes[3].axis("off")

# Shared colorbar
cbar = fig.colorbar(axes[3].images[0], ax=axes, orientation="horizontal", fraction=0.04, pad=0.05)
cbar.set_label("Absolute difference (mean over RGB)", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(repo_root, "notebooks", "diff_maps.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/diff_maps.png")
ref_gray = ref_np_256.mean(axis=-1)
ref_mask = ref_gray < 0.65

xml_black_gray = img_xml_black.mean(axis=-1)
xml_black_mask = xml_black_gray > 0.08

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Row 1: masks
axes[0, 0].imshow(ref_mask, cmap="gray")
axes[0, 0].set_title("Helios ref mask (dark < 0.65)", fontsize=11)
axes[0, 0].axis("off")

axes[0, 1].imshow(xml_black_mask, cmap="gray")
axes[0, 1].set_title("XML render mask (bright > 0.08)", fontsize=11)
axes[0, 1].axis("off")

inter = np.logical_and(ref_mask, xml_black_mask)
union = np.logical_or(ref_mask, xml_black_mask)
axes[0, 2].imshow(inter, cmap="Greens", vmin=0, vmax=1)
axes[0, 2].set_title(f"Intersection (IoU={results['XML black bg']['IoU']:.3f})", fontsize=11)
axes[0, 2].axis("off")

# Row 2: overlay on original
overlay = ref_np_256.copy()
overlay[~ref_mask] = [1.0, 0.0, 0.0]   # red: ref only
overlay[xml_black_mask & ~ref_mask] = [0.0, 0.0, 1.0]  # blue: pred only
overlay[inter] = [0.0, 1.0, 0.0]       # green: both

axes[1, 0].imshow(np.clip(overlay, 0, 1))
axes[1, 0].set_title("Overlay: G=both, R=ref, B=pred", fontsize=11)
axes[1, 0].axis("off")

# Same for 15D
nodes_black_gray = img_15d_black.mean(axis=-1)
nodes_black_mask = nodes_black_gray > 0.08
inter_15d = np.logical_and(ref_mask, nodes_black_mask)
union_15d = np.logical_or(ref_mask, nodes_black_mask)

axes[1, 1].imshow(nodes_black_mask, cmap="gray")
axes[1, 1].set_title("15D nodes mask (bright > 0.08)", fontsize=11)
axes[1, 1].axis("off")

overlay_15d = ref_np_256.copy()
overlay_15d[~ref_mask] = [1.0, 0.0, 0.0]
overlay_15d[nodes_black_mask & ~ref_mask] = [0.0, 0.0, 1.0]
overlay_15d[inter_15d] = [0.0, 1.0, 0.0]

axes[1, 2].imshow(np.clip(overlay_15d, 0, 1))
axes[1, 2].set_title(f"15D overlay (IoU={results['15D black bg']['IoU']:.3f})", fontsize=11)
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(repo_root, "notebooks", "mask_metrics.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/mask_metrics.png")
azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for ax, az in zip(axes, azimuths):
    with torch.no_grad():
        img_az_rgba = rasterizer.render_numpy_geometry(
            geom.tubes, geom.leaflets, geom.ellipsoids,
            camera_height=CAMERA_HEIGHT,
            distance_from_center=DISTANCE_FROM_CENTER,
            azimuth_deg=float(az),
            focus_plant=True,
            background="ground",
        )
    img_az = img_az_rgba[..., :3]
    ax.imshow(np.clip(img_az, 0, 1))
    ax.set_title(f"azimuth={az}°", fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(repo_root, "notebooks", "azimuth_grid.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Saved: notebooks/azimuth_grid.png")
import pandas as pd

df = pd.DataFrame.from_dict(results, orient="index")
df = df[["MAE", "SSIM", "IoU", "Precision", "Recall"]]
print(df.round(3).to_string())

print("\n--- Key observations ---")
print(f"1. Ground background reduces MAE from {results['XML black bg']['MAE']:.3f} → {results['XML ground bg']['MAE']:.3f}")
print(f"   and raises SSIM from {results['XML black bg']['SSIM']:.3f} → {results['XML ground bg']['SSIM']:.3f}")
print(f"2. Plant mask IoU ≈ {results['XML black bg']['IoU']:.3f} (structurally aligned)")
print(f"3. 15D nodes render is slightly closer than XML render:")
print(f"   15D MAE={results['15D black bg']['MAE']:.3f} vs XML MAE={results['XML black bg']['MAE']:.3f}")
print(f"4. nodes_to_geometry is NOT differentiable (uses .detach().cpu().numpy())")
# Print the exact signature and first few lines
import inspect
src = inspect.getsource(nodes_to_geometry)
print(src[:1200])
