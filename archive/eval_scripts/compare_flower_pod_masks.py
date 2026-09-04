"""
Compare Flower and Pod masks between Helios C++ Ground Truth and PyTorch 17D Differentiable Renderer.
Extracts per-organ masks, calculates IoU, Dice, Precision, Recall, and produces comparison figure.
"""
import os
import sys
import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512  # High resolution for precise mask analysis

EXACT_GT_DIR = os.path.join(repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "output", "exact_gt_renders")
ASSETS_DIR = os.path.join(repo_root, "docs", "results", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

xml_path = os.path.join(EXACT_GT_DIR, "rad_dap090_0000_plant_0000.xml")
helios_rgb_path = os.path.join(EXACT_GT_DIR, "rad_dap090_0000_rad.jpeg")
helios_masks_path = os.path.join(EXACT_GT_DIR, "rad_dap090_0000_masks.json")
helios_cam_path = os.path.join(EXACT_GT_DIR, "rad_dap090_0000_camera.json")
helios_cam_legacy_path = os.path.join(EXACT_GT_DIR, "rad_dap090_0000_camera_params.json")

print("Loading DAP 90 plant XML and Helios Ground Truth...")
arr = PlantOrganArray.from_xml_file(xml_path)
renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE).to(DEVICE)
mesh = renderer.geo_builder.build_mesh_from_part_tensor(arr.to_part_tensor(device=DEVICE), device=DEVICE)

# Read exact camera parameters. The Helios exact-GT renders use focus-plant auto-FOV
# (matching compute_focus_plant_camera), so we only force a fixed HFOV if an explicit
# legacy *_camera_params.json file is present.
cam_h = 5.0
cam_el = 90.0
cam_hfov = None
if os.path.exists(helios_cam_path):
    with open(helios_cam_path, "r") as f:
        cam_data = json.load(f)
    cam_h = float(cam_data.get("acquisition_properties", {}).get("camera_height_m", 5.0))
    cam_el = float(cam_data.get("acquisition_properties", {}).get("camera_angle_deg", 90.0))
if os.path.exists(helios_cam_legacy_path):
    with open(helios_cam_legacy_path, "r") as f:
        cam_data = json.load(f)
    f_len = cam_data.get("camera_properties", {}).get("focal_length", 50.0)
    s_w = cam_data.get("camera_properties", {}).get("sensor_width", 35.0)
    cam_hfov = 2.0 * math.degrees(math.atan((s_w * 0.5) / max(f_len, 1e-3)))

# Extract Helios GT masks
coco_data = json.load(open(helios_masks_path))
src_w = int(coco_data["images"][0].get("width", 720))
src_h = int(coco_data["images"][0].get("height", 720))
sx = IMG_SIZE / src_w
sy = IMG_SIZE / src_h

# Categories: 0: internode, 1: petiole, 2: leaf, 3: floral_bud, 4: flower, 5: pod
gt_masks = {cat_id: np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool) for cat_id in range(6)}

for ann in coco_data["annotations"]:
    cat = ann["category_id"]
    if cat not in gt_masks:
        continue
    seg = ann["segmentation"]
    if not isinstance(seg, list):
        continue
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in seg:
        pts = []
        for i in range(0, len(poly), 2):
            pts.append((poly[i] * sx, poly[i + 1] * sy))
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)
    gt_masks[cat] = gt_masks[cat] | (np.array(canvas) > 0)

# Render PyTorch 17D type buffer
type_t = renderer.render_organ_type_buffer(
    mesh,
    azimuth_deg=0.0,
    elevation_deg=cam_el,
    camera_height=cam_h,
    focus_plant=(cam_hfov is None),
    hfov_override_deg=cam_hfov,
    image_size=IMG_SIZE,
)
type_buf = type_t.detach().cpu().numpy()

# PyTorch organ types: 0: stem, 1: petiole, 2: leaf, 3: peduncle, 4: flower, 5: pod
pt_masks = {
    3: (type_buf == 3),  # peduncle / floral bud
    4: (type_buf == 4),  # flower
    5: (type_buf == 5),  # pod
}

print("\n--- Quantitative Mask Overlap Statistics (DAP 90) ---")
for organ_id, name in [(4, "Flower"), (5, "Pod"), (3, "Peduncle / Bud")]:
    gt_m = gt_masks[organ_id]
    pt_m = pt_masks[organ_id]
    intersection = np.logical_and(gt_m, pt_m).sum()
    union = np.logical_or(gt_m, pt_m).sum()
    iou = intersection / max(union, 1)
    dice = 2.0 * intersection / max(gt_m.sum() + pt_m.sum(), 1)
    print(f"[{name}]")
    print(f"  Helios GT Pixels: {gt_m.sum()} ({100.0 * gt_m.sum() / (IMG_SIZE*IMG_SIZE):.2f}%)")
    print(f"  PyTorch 17D Pixels: {pt_m.sum()} ({100.0 * pt_m.sum() / (IMG_SIZE*IMG_SIZE):.2f}%)")
    print(f"  Intersection: {intersection}")
    print(f"  IoU: {iou:.4f}, Dice: {dice:.4f}")

# Generate detailed comparison plot
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.patch.set_facecolor("#12131C")
plt.subplots_adjust(wspace=0.08, hspace=0.15)

rows = [
    ("Flower (Category 4)", 4, "#FFD700"),
    ("Pod / Fruit (Category 5)", 5, "#DAA520"),
    ("Peduncle / Bud (Category 3)", 3, "#BDB76B"),
]

for r_idx, (title, cat_id, color_hex) in enumerate(rows):
    gt_m = gt_masks[cat_id]
    pt_m = pt_masks[cat_id]
    
    # Col 0: Helios GT
    ax = axes[r_idx, 0]
    ax.set_facecolor("#0d0d1a")
    rgb_gt = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    r = int(color_hex[1:3], 16) / 255.0
    g = int(color_hex[3:5], 16) / 255.0
    b = int(color_hex[5:7], 16) / 255.0
    rgb_gt[gt_m] = [r, g, b]
    ax.imshow(rgb_gt)
    ax.axis("off")
    ax.set_title(f"Helios GT {title}\n({gt_m.sum()} px)", fontsize=11, color="#ff9999", fontweight="bold")
    
    # Col 1: PyTorch 17D
    ax = axes[r_idx, 1]
    ax.set_facecolor("#0d0d1a")
    rgb_pt = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    rgb_pt[pt_m] = [r, g, b]
    ax.imshow(rgb_pt)
    ax.axis("off")
    ax.set_title(f"PyTorch 17D {title}\n({pt_m.sum()} px)", fontsize=11, color="#70d6ff", fontweight="bold")
    
    # Col 2: Overlap / Diff
    ax = axes[r_idx, 2]
    ax.set_facecolor("#0d0d1a")
    diff_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    # Green = True Positive (overlap)
    diff_rgb[gt_m & pt_m] = [0.1, 0.9, 0.2]
    # Red = False Positive (PyTorch only)
    diff_rgb[pt_m & ~gt_m] = [0.9, 0.2, 0.2]
    # Blue = False Negative (Helios GT only)
    diff_rgb[gt_m & ~pt_m] = [0.2, 0.4, 0.9]
    ax.imshow(diff_rgb)
    ax.axis("off")
    intersection = np.logical_and(gt_m, pt_m).sum()
    union = np.logical_or(gt_m, pt_m).sum()
    iou = intersection / max(union, 1)
    ax.set_title(f"Overlap (TP=Green, FP=Red, FN=Blue)\nIoU: {iou:.3f}", fontsize=11, color="#c3a6e0", fontweight="bold")
    
    # Col 3: Overlay on Plant RGB
    ax = axes[r_idx, 3]
    ax.set_facecolor("#0d0d1a")
    if os.path.exists(helios_rgb_path):
        base_img = np.array(Image.open(helios_rgb_path).resize((IMG_SIZE, IMG_SIZE))) / 255.0
        overlay = base_img.copy() * 0.4
        overlay[gt_m] = overlay[gt_m] + np.array([0.0, 0.6, 0.0]) # GT in green
        overlay[pt_m] = overlay[pt_m] + np.array([0.6, 0.0, 0.6]) # PT in magenta
        ax.imshow(np.clip(overlay, 0, 1))
    ax.axis("off")
    ax.set_title(f"Overlay on RGB\n(Green=GT, Magenta=PT)", fontsize=11, color="#ffd166", fontweight="bold")

fig.suptitle("Flower, Pod, and Peduncle Mask Comparison: Helios C++ GT vs PyTorch 17D (DAP 90)", fontsize=14, fontweight="bold", color="white", y=0.99)

out_path = os.path.join(ASSETS_DIR, "fig_flower_pod_mask_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
print("DONE")
