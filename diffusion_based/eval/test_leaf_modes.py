"""
Test and compare GenericLeafPrototype vs High-Res OBJ Leaf Mesh rendering in PyTorch Differentiable Renderer.
Generates side-by-side visual and quantitative comparison across DAP 10, 50, and 90.
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
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512

EXACT_GT_DIR = os.path.join(repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "output", "exact_gt_renders")
ASSETS_DIR = os.path.join(repo_root, "docs", "results", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

test_stages = [
    ("DAP 10 (Seedling)", "rad_dap010_0000"),
    ("DAP 50 (Branching)", "rad_dap050_0000"),
    ("DAP 90 (Mature)", "rad_dap090_0000"),
]

renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE).to(DEVICE)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.patch.set_facecolor("#12131C")
plt.subplots_adjust(wspace=0.08, hspace=0.15)

col_titles = [
    "Helios C++ Raytrace GT\n(GenericLeafPrototype)",
    "PyTorch 40D Render\n(leaf_mode='generic')",
    "PyTorch 40D Render\n(leaf_mode='obj')",
    "Organ Map Diff\n(Generic=Mint, OBJ=Magenta, Both=White)"
]

for col_idx, ct in enumerate(col_titles):
    axes[0, col_idx].annotate(
        ct,
        xy=(0.5, 1.15),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#70d6ff" if col_idx > 0 else "#ff9999",
    )

for row_idx, (stage_name, prefix) in enumerate(test_stages):
    xml_path = os.path.join(EXACT_GT_DIR, f"{prefix}_plant_0000.xml")
    gt_rgb_path = os.path.join(EXACT_GT_DIR, f"{prefix}_rad.jpeg")
    cam_path = os.path.join(EXACT_GT_DIR, f"{prefix}_camera_params.json")

    arr = PlantOrganArray.from_xml_file(xml_path)

    # Read camera parameters
    cam_h = 5.0
    cam_el = 90.0
    cam_hfov = None
    if os.path.exists(cam_path):
        with open(cam_path, "r") as f:
            cam_data = json.load(f)
        f_len = cam_data.get("camera_properties", {}).get("focal_length", 50.0)
        s_w = cam_data.get("camera_properties", {}).get("sensor_width", 35.0)
        cam_h = float(cam_data.get("acquisition_properties", {}).get("camera_height_m", 5.0))
        cam_el = float(cam_data.get("acquisition_properties", {}).get("camera_angle_deg", 90.0))
        cam_hfov = 2.0 * math.degrees(math.atan((s_w * 0.5) / max(f_len, 1e-3)))

    # Col 0: Helios GT Raytrace
    ax0 = axes[row_idx, 0]
    ax0.set_facecolor("#0d0d1a")
    if os.path.exists(gt_rgb_path):
        gt_img = Image.open(gt_rgb_path).resize((IMG_SIZE, IMG_SIZE))
        ax0.imshow(gt_img)
    ax0.axis("off")
    ax0.set_ylabel(stage_name, color="white", fontsize=12, fontweight="bold", labelpad=10)

    # Col 1: leaf_mode="generic"
    mesh_generic = renderer.geo_builder.build_mesh_from_organ_array(arr, device=DEVICE, species="cowpea", leaf_mode="generic")
    img_generic_t = renderer.render_mesh(
        mesh_generic,
        azimuth_deg=0.0,
        elevation_deg=cam_el,
        camera_height=cam_h,
        focus_plant=(cam_hfov is None),
        hfov_override_deg=cam_hfov,
        image_size=IMG_SIZE,
    )
    img_generic = img_generic_t.permute(1, 2, 0).detach().cpu().numpy()
    type_generic_t = renderer.render_organ_type_buffer(
        mesh_generic,
        azimuth_deg=0.0,
        elevation_deg=cam_el,
        camera_height=cam_h,
        focus_plant=(cam_hfov is None),
        hfov_override_deg=cam_hfov,
        image_size=IMG_SIZE,
    )
    mask_generic = (type_generic_t.detach().cpu().numpy() >= 0) & (type_generic_t.detach().cpu().numpy() != 255)

    ax1 = axes[row_idx, 1]
    ax1.set_facecolor("#0d0d1a")
    ax1.imshow(np.clip(img_generic, 0, 1))
    ax1.axis("off")
    ax1.text(0.03, 0.05, f"V={mesh_generic['vertices'].shape[0]}, F={mesh_generic['faces'].shape[0]}", transform=ax1.transAxes, color="white", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # Col 2: leaf_mode="obj"
    mesh_obj = renderer.geo_builder.build_mesh_from_organ_array(arr, device=DEVICE, species="cowpea", leaf_mode="obj")
    img_obj_t = renderer.render_mesh(
        mesh_obj,
        azimuth_deg=0.0,
        elevation_deg=cam_el,
        camera_height=cam_h,
        focus_plant=(cam_hfov is None),
        hfov_override_deg=cam_hfov,
        image_size=IMG_SIZE,
    )
    img_obj = img_obj_t.permute(1, 2, 0).detach().cpu().numpy()
    type_obj_t = renderer.render_organ_type_buffer(
        mesh_obj,
        azimuth_deg=0.0,
        elevation_deg=cam_el,
        camera_height=cam_h,
        focus_plant=(cam_hfov is None),
        hfov_override_deg=cam_hfov,
        image_size=IMG_SIZE,
    )
    mask_obj = (type_obj_t.detach().cpu().numpy() >= 0) & (type_obj_t.detach().cpu().numpy() != 255)

    ax2 = axes[row_idx, 2]
    ax2.set_facecolor("#0d0d1a")
    ax2.imshow(np.clip(img_obj, 0, 1))
    ax2.axis("off")
    ax2.text(0.03, 0.05, f"V={mesh_obj['vertices'].shape[0]}, F={mesh_obj['faces'].shape[0]}", transform=ax2.transAxes, color="white", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

    # Col 3: Difference between Generic and OBJ
    ax3 = axes[row_idx, 3]
    ax3.set_facecolor("#0d0d1a")
    diff_map = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    # Both active -> white
    diff_map[mask_generic & mask_obj] = [0.9, 0.9, 0.9]
    # Generic only -> mint
    diff_map[mask_generic & ~mask_obj] = [0.1, 0.9, 0.5]
    # OBJ only -> magenta
    diff_map[~mask_generic & mask_obj] = [0.9, 0.1, 0.7]
    ax3.imshow(diff_map)
    ax3.axis("off")
    
    inter = np.logical_and(mask_generic, mask_obj).sum()
    union = np.logical_or(mask_generic, mask_obj).sum()
    iou = inter / max(union, 1)
    ax3.text(0.03, 0.05, f"Mask IoU: {iou:.3f}", transform=ax3.transAxes, color="white", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))

fig.suptitle("Helios Leaf Mode Benchmark: GenericLeafPrototype vs High-Res OBJ Mesh", fontsize=15, fontweight="bold", color="white", y=0.98)

out_path = os.path.join(ASSETS_DIR, "fig_leaf_mode_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
print("DONE")
