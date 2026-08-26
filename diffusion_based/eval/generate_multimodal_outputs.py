"""
Visualize render_multimodal() outputs: RGB, Depth, Foreground Mask, Organ-Type Map.
Produces fig8_multimodal_depth_mask.png using the canonical 16D (26D) Part Tensor and PyTorch Differentiable Renderer.
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

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
ASSETS_DIR = os.path.join(repo_root, "docs", "results", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

EXACT_GT_DIR = os.path.join(repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "output", "exact_gt_renders")

PLANTS = [
    ("DAP 10\n(Seedling)",
     os.path.join(EXACT_GT_DIR, "rad_dap010_0000_plant_0000.xml"),
     os.path.join(EXACT_GT_DIR, "rad_dap010_0000_rad.jpeg"),
     os.path.join(EXACT_GT_DIR, "rad_dap010_0000_masks.json"),
     os.path.join(EXACT_GT_DIR, "rad_dap010_0000_camera.json")),
    ("DAP 50\n(Branching)",
     os.path.join(EXACT_GT_DIR, "rad_dap050_0000_plant_0000.xml"),
     os.path.join(EXACT_GT_DIR, "rad_dap050_0000_rad.jpeg"),
     os.path.join(EXACT_GT_DIR, "rad_dap050_0000_masks.json"),
     os.path.join(EXACT_GT_DIR, "rad_dap050_0000_camera.json")),
    ("DAP 90\n(Mature)",
     os.path.join(EXACT_GT_DIR, "rad_dap090_0000_plant_0000.xml"),
     os.path.join(EXACT_GT_DIR, "rad_dap090_0000_rad.jpeg"),
     os.path.join(EXACT_GT_DIR, "rad_dap090_0000_masks.json"),
     os.path.join(EXACT_GT_DIR, "rad_dap090_0000_camera.json")),
]

# Helios COCO organ category -> display color
HELIOS_CAT_COLORS = {
    0: (0.55, 0.27, 0.07),   # internode
    1: (0.42, 0.56, 0.14),   # petiole
    2: (0.13, 0.56, 0.13),   # leaf
    3: (0.74, 0.72, 0.42),   # floral_bud
    4: (1.00, 0.84, 0.00),   # flower
    5: (0.85, 0.65, 0.13),   # pod
}

# Mesh organ type (OT_*) -> color and label
ORGAN_META = {
    0: ("#8B4513", "Stem/Internode"),
    1: ("#6B8E23", "Petiole"),
    2: ("#228B22", "Leaf"),
    3: ("#BDB76B", "Peduncle"),
    4: ("#FFD700", "Flower"),
    5: ("#DAA520", "Pod/Fruit"),
}

renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE).to(DEVICE)


def depth_to_canopy_height_cm(depth_tensor):
    """
    Convert depth tensor (Z_world in physical meters) to masked Canopy Height in cm.
    Taller/higher canopy elements (larger Z) have larger cm values.
    """
    d = depth_tensor.detach().cpu().numpy()
    fg_mask = (d > 1e-4)
    if not fg_mask.any():
        return np.ma.masked_all_like(d), 0.0, 0.0

    d_cm = d * 100.0
    min_h_cm = float(d_cm[fg_mask].min())
    max_h_cm = float(d_cm[fg_mask].max())

    masked_d_cm = np.ma.masked_where(~fg_mask, d_cm)
    return masked_d_cm, min_h_cm, max_h_cm


def helios_organ_map(masks_path, H, W):
    """Rasterize Helios COCO polygon masks into a single organ-type color image with clean white background."""
    data = json.load(open(masks_path))
    src_w = int(data["images"][0].get("width", 720))
    src_h = int(data["images"][0].get("height", 720))
    sx = W / src_w
    sy = H / src_h
    cat_masks = {cat["id"]: np.zeros((H, W), dtype=np.uint8) for cat in data["categories"]}
    for ann in data["annotations"]:
        cat = ann["category_id"]
        seg = ann["segmentation"]
        if not isinstance(seg, list):
            continue
        canvas = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(canvas)
        for poly in seg:
            pts = []
            for i in range(0, len(poly), 2):
                pts.append((poly[i] * sx, poly[i + 1] * sy))
            if len(pts) >= 3:
                draw.polygon(pts, fill=1)
        cat_masks[cat] = np.maximum(cat_masks[cat], np.array(canvas))
    rgb = np.ones((H, W, 3), dtype=np.float32)  # pure white background matching GT
    for cat_id, color in HELIOS_CAT_COLORS.items():
        rgb[cat_masks[cat_id] > 0] = color
    return rgb


def organ_buffer_to_rgb(type_buf: torch.Tensor, H: int, W: int) -> np.ndarray:
    """Map organ type buffer (-1=bg, 0=stem, 1=petiole, 2=leaf, ...) to color image with clean white background."""
    buf_np = type_buf.detach().cpu().numpy()
    rgb = np.ones((H, W, 3), dtype=np.float32)  # pure white background
    for ot_id, (hex_col, _) in ORGAN_META.items():
        r = int(hex_col[1:3], 16) / 255.0
        g = int(hex_col[3:5], 16) / 255.0
        b = int(hex_col[5:7], 16) / 255.0
        mask = (buf_np == ot_id)
        rgb[mask, 0] = r
        rgb[mask, 1] = g
        rgb[mask, 2] = b
    return rgb


# -----------------------------------------------------------------------
# Figure 8: Multi-Modal Output Visualization
# -----------------------------------------------------------------------
print("Generating Figure 8: Multi-Modal Render Outputs (RGB / Depth / Mask / Organ Map)...")

n_rows = len(PLANTS)
n_cols = 6  # Helios GT | Helios Organ Map | RGB | Depth | FG Mask | Organ Type

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4.2 * n_rows))
fig.patch.set_facecolor("#12131C")
plt.subplots_adjust(wspace=0.06, hspace=0.18)

col_titles = [
    "Helios C++\nRadiation GT",
    "Helios C++\nOrgan Map GT",
    "PyTorch 16D (26D)\nRGB Render",
    "Canopy Height (CHM)\n(taller = brighter)",
    "PyTorch 16D (26D)\nForeground Mask",
    "PyTorch 16D (26D)\nOrgan-Type Map"
]
col_colors = ["#ff9999", "#ff9999", "#70d6ff", "#c3a6e0", "#88d8c0", "#ffd166"]

for col, (title, color) in enumerate(zip(col_titles, col_colors)):
    axes[0, col].set_title(title, fontsize=12, fontweight="bold", color=color, pad=8)

for row, (label, xml_path, helios_path, helios_masks_path, helios_cam_path) in enumerate(PLANTS):
    ax_row = axes[row]

    if not os.path.exists(xml_path):
        print(f"Warning: {xml_path} missing")
        continue

    arr = PlantOrganArray.from_xml_file(xml_path)
    mesh = renderer.geo_builder.build_mesh_from_organ_array(arr, device=DEVICE, species="cowpea")

    # Read exact camera height/elevation from the camera JSON. The exact-GT renders
    # use focus-plant auto-FOV (Helios fits the plant bbox into the frame), so we
    # deliberately ignore the recorded telephoto focal_length and use focus_plant=True.
    cam_h = 5.0
    cam_el = 90.0
    cam_hfov = None
    if os.path.exists(helios_cam_path):
        with open(helios_cam_path, "r") as f:
            cam_data = json.load(f)
        cam_h = float(cam_data.get("acquisition_properties", {}).get("camera_height_m", 5.0))
        cam_el = float(cam_data.get("acquisition_properties", {}).get("camera_angle_deg", 90.0))

    # --- Col 0: Helios C++ GT ---
    ax = ax_row[0]
    if os.path.exists(helios_path):
        helios_img = np.array(Image.open(helios_path).convert("RGB")) / 255.0
        h, w = helios_img.shape[:2]
        if h != w:
            min_dim = min(h, w)
            y0 = (h - min_dim) // 2
            x0 = (w - min_dim) // 2
            helios_img = helios_img[y0:y0+min_dim, x0:x0+min_dim]
        if helios_img.shape[0] != IMG_SIZE:
            from PIL import Image as PILImage
            helios_img = np.array(PILImage.fromarray((helios_img * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)) / 255.0
        ax.imshow(helios_img)
    else:
        ax.text(0.5, 0.5, "No Helios GT", ha='center', va='center', color='red', fontsize=10, transform=ax.transAxes)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")
    ax.set_ylabel(label, fontsize=12, color="#e0e0e0", rotation=0,
                  labelpad=65, va='center', fontweight='bold')

    # --- Col 1: Helios C++ radiation organ-type map ---
    ax = ax_row[1]
    if os.path.exists(helios_masks_path):
        helios_organ = helios_organ_map(helios_masks_path, IMG_SIZE, IMG_SIZE)
        ax.imshow(helios_organ)
    else:
        ax.text(0.5, 0.5, "No Helios masks", ha='center', va='center', color='red', fontsize=10, transform=ax.transAxes)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")

    # Render multimodal PyTorch outputs via unified forward(..., include_depth=True) with exact camera FoV
    rgbd_t = renderer.forward(
        mesh,
        azimuth_deg=0.0,
        elevation_deg=cam_el,
        camera_height=cam_h,
        background="ground",
        differentiable=False,
        focus_plant=(cam_hfov is None),
        hfov_override_deg=cam_hfov,
        image_size=IMG_SIZE,
        include_depth=True,
    )
    rgb_t = rgbd_t[:3]
    depth_t = rgbd_t[3]
    type_t = renderer.render_organ_type_buffer(
        mesh,
        azimuth_deg=0.0,
        elevation_deg=cam_el,
        camera_height=cam_h,
        focus_plant=(cam_hfov is None),
        hfov_override_deg=cam_hfov,
        image_size=IMG_SIZE,
    )

    H, W = IMG_SIZE, IMG_SIZE
    rgb_np = rgb_t.permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
    masked_h_cm, min_h_cm, max_h_cm = depth_to_canopy_height_cm(depth_t)
    mask_np = (type_t >= 0).detach().cpu().numpy()
    organ_np = organ_buffer_to_rgb(type_t, H, W)

    fg_pct = 100.0 * mask_np.sum() / (H * W)

    # --- Col 2: RGB ---
    ax = ax_row[2]
    ax.imshow(rgb_np)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")
    ax.text(0.03, 0.03, f"N={arr.num_nodes} organs", transform=ax.transAxes,
            fontsize=9, color='white', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # --- Col 3: Depth / Canopy Height Model ---
    ax = ax_row[3]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#0d0d1a")
    vmax_cm = max(1.0, float(math.ceil(max_h_cm)))
    im_depth = ax.imshow(masked_h_cm, cmap=cmap, vmin=0.0, vmax=vmax_cm)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")

    cbar = plt.colorbar(im_depth, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (cm)", fontsize=9, fontweight="bold", color="#c3a6e0")
    cbar.ax.tick_params(labelsize=8, colors="#e0e0e0")
    cbar.outline.set_edgecolor("#555555")

    ax.text(0.03, 0.03, f"Height: 0–{max_h_cm:.1f} cm", transform=ax.transAxes,
            fontsize=9, color='#ffffff', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # --- Col 4: Foreground Mask ---
    ax = ax_row[4]
    mask_display = np.stack([mask_np * 0.4, mask_np * 0.9, mask_np * 0.6], axis=-1)
    ax.imshow(mask_display)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")
    ax.text(0.03, 0.03, f"FG: {fg_pct:.1f}%", transform=ax.transAxes,
            fontsize=9, color='#88d8c0', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # --- Col 5: Organ-Type Map ---
    ax = ax_row[5]
    ax.imshow(organ_np)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")
    # Mini legend
    present_types = np.unique(type_t.detach().cpu().numpy())
    present_types = [t for t in present_types if t >= 0]
    patches = []
    for ot in sorted(present_types):
        meta = ORGAN_META.get(int(ot), ("#AAAAAA", f"Type {ot}"))
        patches.append(mpatches.Patch(color=meta[0], label=meta[1]))
    if patches:
        ax.legend(handles=patches, loc='lower right',
                  fontsize=7, framealpha=0.85,
                  facecolor='#1a1a2e', labelcolor='white',
                  edgecolor='#444', ncol=1)

# Overall title
fig.suptitle(
    "Figure 8: Helios C++ Raytrace vs 16D / 26D PyTorch Differentiable Multi-Modal Outputs (RGB · Canopy Height · Mask · Semantic Map)",
    fontsize=13, fontweight='bold', color='white', y=0.995
)

out_path = os.path.join(ASSETS_DIR, "fig8_multimodal_depth_mask.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
print("DONE")
