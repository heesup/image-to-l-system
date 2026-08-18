"""
Visualize render_multimodal() outputs: RGB, Depth, Foreground Mask, Organ-Type Map.
Produces fig8_multimodal_depth_mask.png for the benchmark report.
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE, ORGAN_PETIOLE,
    ORGAN_LEAF, ORGAN_BUD, ORGAN_PEDUNCLE, ORGAN_FLOWER_OPEN, ORGAN_FRUIT, ORGAN_FLOWER_CLOSED,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
ASSETS_DIR = os.path.join(repo_root, "docs", "results", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

PLANTS = [
    ("DAP 10\n(Seedling)",
     "dataset/helios_data/cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
    ("DAP 50\n(Branching)",
     "dataset/helios_data/cowpea_dap050_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
    ("DAP 90\n(Mature)",
     "dataset/helios_data/cowpea_dap090_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
]

# Organ type → display color & label
# NOTE: render_multimodal() keys organ_masks by the MESH organ-type convention
# (helios_pytorch_geometry.py OT_* constants), NOT the 14D part-tensor convention.
#   OT_STEM=0, OT_PETIOLE=1, OT_LEAF=2, OT_PEDUNCLE=3, OT_FLOWER=4, OT_FRUIT=5
ORGAN_META = {
    0: ("#8B4513", "Stem/Internode"),
    1: ("#6B8E23", "Petiole"),
    2: ("#228B22", "Leaf"),
    3: ("#BDB76B", "Peduncle"),
    4: ("#FFD700", "Flower"),
    5: ("#DAA520", "Pod/Fruit"),
}

renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE)

def load_plant(xml_path):
    full = os.path.join(repo_root, xml_path)
    if not os.path.exists(full):
        return None, None
    arr = PlantOrganArray.from_xml_file_typed(full)
    p14 = arr.to_part_tensor_14d(device=DEVICE)
    return arr, p14

def organ_type_colormap(organ_masks, H, W, device):
    """Compose a color image from per-organ-type boolean masks."""
    rgb = np.ones((H, W, 3), dtype=np.float32)  # white background
    for ot_id, mask in organ_masks.items():
        color_hex = ORGAN_META.get(ot_id, ("#AAAAAA", "Unknown"))[0]
        r = int(color_hex[1:3], 16) / 255.0
        g = int(color_hex[3:5], 16) / 255.0
        b = int(color_hex[5:7], 16) / 255.0
        m = mask.cpu().numpy()
        rgb[m, 0] = r
        rgb[m, 1] = g
        rgb[m, 2] = b
    return rgb

def depth_colormap(depth_tensor):
    """Apply plasma colormap to depth map, black where depth == 0."""
    d = depth_tensor.cpu().numpy()
    cmap = plt.get_cmap("plasma")
    rgb = cmap(d)[:, :, :3].astype(np.float32)
    rgb[d == 0] = 0  # black background where no plant
    return rgb

# -----------------------------------------------------------------------
# Figure 8: Multi-Modal Output Visualization
# -----------------------------------------------------------------------
print("Generating Figure 8: Multi-Modal Render Outputs (RGB / Depth / Mask / Organ Map)...")

n_rows = len(PLANTS)
n_cols = 4  # RGB | Depth | FG Mask | Organ Type

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
fig.patch.set_facecolor("#1a1a2e")
plt.subplots_adjust(wspace=0.05, hspace=0.18)

col_titles = ["RGB Render", "Depth Map\n(closer = brighter)", "Foreground Mask", "Organ-Type Map"]
col_colors = ["#e0e0e0", "#c3a6e0", "#88d8c0", "#f7c59f"]

for col, (title, color) in enumerate(zip(col_titles, col_colors)):
    axes[0, col].set_title(title, fontsize=13, fontweight="bold", color=color, pad=8)

for row, (label, xml_path) in enumerate(PLANTS):
    arr, p14 = load_plant(xml_path)
    ax_row = axes[row]

    if arr is None or p14 is None:
        for ax in ax_row:
            ax.text(0.5, 0.5, "File not found", ha='center', va='center',
                    color='red', fontsize=10, transform=ax.transAxes)
            ax.set_facecolor("#1a1a2e")
            ax.axis("off")
        ax_row[0].set_ylabel(label, fontsize=12, color="white", rotation=0,
                             labelpad=60, va='center')
        continue

    # Render multimodal
    out = renderer.render_part_tensor_14d_multimodal(
        p14,
        template_organ_array=arr,
        azimuth_deg=0.0,
        elevation_deg=90.0,
        camera_height=5.0,
        device=DEVICE,
        return_depth=True,
        return_mask=True,
        return_organ_masks=True,
    )

    H, W = IMG_SIZE, IMG_SIZE
    rgb_np    = out["rgb"].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    depth_np  = depth_colormap(out["depth"])
    mask_np   = out["mask"].cpu().numpy()
    organ_np  = organ_type_colormap(out["organ_masks"], H, W, DEVICE)

    # Compute stats for annotation
    fg_pct    = 100.0 * mask_np.sum() / (H * W)
    depth_raw = out["depth"].cpu().numpy()
    d_min     = float(depth_raw[depth_raw > 0].min()) if (depth_raw > 0).any() else 0
    d_max     = float(depth_raw.max())

    # --- Col 0: RGB ---
    ax = ax_row[0]
    ax.imshow(rgb_np)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")
    ax.set_ylabel(label, fontsize=12, color="#e0e0e0", rotation=0,
                  labelpad=65, va='center', fontweight='bold')
    organ_count = p14.shape[0]
    ax.text(0.02, 0.02, f"N={organ_count} organs", transform=ax.transAxes,
            fontsize=9, color='white', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # --- Col 1: Depth ---
    ax = ax_row[1]
    ax.imshow(depth_np)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")
    ax.text(0.02, 0.02, f"NDC-z: {d_min:.2f}–{d_max:.2f}", transform=ax.transAxes,
            fontsize=9, color='#c3a6e0', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # --- Col 2: Foreground Mask ---
    ax = ax_row[2]
    mask_display = np.stack([mask_np * 0.4, mask_np * 0.9, mask_np * 0.6], axis=-1)
    ax.imshow(mask_display)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")
    ax.text(0.02, 0.02, f"FG: {fg_pct:.1f}%", transform=ax.transAxes,
            fontsize=9, color='#88d8c0', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # --- Col 3: Organ-Type Map ---
    ax = ax_row[3]
    ax.imshow(organ_np)
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")
    # Mini legend for present organ types
    present_types = sorted(out["organ_masks"].keys())
    patches = []
    for ot in present_types:
        meta = ORGAN_META.get(ot, ("#AAAAAA", f"Type {ot}"))
        patches.append(mpatches.Patch(color=meta[0], label=meta[1]))
    if patches:
        leg = ax.legend(handles=patches, loc='lower right',
                        fontsize=7, framealpha=0.85,
                        facecolor='#1a1a2e', labelcolor='white',
                        edgecolor='#444', ncol=1)

# Overall title
fig.suptitle(
    "Figure 8: render_multimodal() Output — RGB · Depth · Foreground Mask · Organ-Type Map",
    fontsize=14, fontweight='bold', color='white', y=0.995
)

out_path = os.path.join(ASSETS_DIR, "fig8_multimodal_depth_mask.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
print("DONE")
