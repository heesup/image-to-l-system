"""
Evaluation & Comparison of Multi-Modal Per-Organ Masks and Depth between Helios C++ Ground Truth,
Reconstructed XML Helios Raytrace, and PyTorch 13D Differentiable Renderer across DAP 10, 50, 90.

Produces:
1. Per-class IoU for Internode, Petiole, Leaf, Peduncle, Flower, Fruit.
2. Depth map comparison (Helios raytraced depth vs PyTorch CHM depth).
3. High-resolution multi-modal comparison figure:
   docs/results/assets/fig10_helios_per_organ_mask_comparison.png
"""

import os
import sys
import math
import json
import subprocess
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.part_assembly_to_xml import assemble_part_tensor_to_xml

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
OUTPUT_DIR = os.path.join(REPO_ROOT, "docs/results/assets")
SCRATCH_DIR = "/tmp/helios_organ_mask_eval"

TEST_PLANTS = [
    ("DAP 10 (Seedling)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_plant_0000.xml", "dap010"),
    ("DAP 50 (Branching)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_plant_0000.xml", "dap050"),
    ("DAP 90 (Fruiting)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_plant_0000.xml", "dap090"),
]

# Color palette for 6 semantic organ classes
ORGAN_CLASSES = ["Internode", "Petiole", "Leaf", "Peduncle", "Flower", "Fruit"]
ORGAN_COLORS = [
    np.array([120, 80, 40], dtype=np.uint8),    # 0: Internode (Brown/Woody)
    np.array([180, 220, 100], dtype=np.uint8),  # 1: Petiole (Light Yellow-Green)
    np.array([40, 160, 50], dtype=np.uint8),    # 2: Leaf (Green)
    np.array([210, 190, 70], dtype=np.uint8),   # 3: Peduncle (Olive/Yellowish Stem)
    np.array([255, 220, 0], dtype=np.uint8),    # 4: Flower (Bright Yellow)
    np.array([160, 50, 180], dtype=np.uint8),   # 5: Fruit (Purple/Burgundy)
]


def render_helios_full(xml_path: str, name_prefix: str, species: str = "cowpea") -> Dict[str, Any]:
    """Invokes Helios C++ binary to render radiation RGB, Depth, and COCO masks."""
    out_dir = os.path.join(SCRATCH_DIR, name_prefix)
    os.makedirs(out_dir, exist_ok=True)
    build_dir = os.path.join(REPO_ROOT, "Digital-Crops/projects/syntheticdata_generation/build")
    cfg_file = os.path.join(REPO_ROOT, f"Digital-Crops/projects/syntheticdata_generation/configs/params_{species}.json")
    if not os.path.exists(cfg_file):
        cfg_file = os.path.join(build_dir, "params.json")

    cmd = [
        "./main",
        "--renderer", "radiation",
        "--input-xml", os.path.abspath(xml_path),
        "--output", out_dir,
        "-n", name_prefix,
        "--focus-plant",
        "--depth", "true",
        "-f", cfg_file,
    ]
    try:
        subprocess.run(cmd, cwd=build_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print(f"Warning: Helios C++ run failed for {xml_path}: {e}")

    # Locate outputs (either in out_dir/species/ or out_dir/)
    def _find_file(pattern):
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith(pattern):
                    return os.path.join(root, f)
        return None

    rgb_file = _find_file("_rad.jpeg")
    depth_file = _find_file("_depth.jpeg")
    masks_file = _find_file("_masks.json")

    rgb_np = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    depth_np = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    mask_map = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32) - 1

    if rgb_file and os.path.exists(rgb_file):
        img = np.array(Image.open(rgb_file).convert("RGB")) / 255.0
        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)) / 255.0
        rgb_np = img

    if depth_file and os.path.exists(depth_file):
        d_img = np.array(Image.open(depth_file).convert("L")) / 255.0
        if d_img.shape[0] != IMG_SIZE or d_img.shape[1] != IMG_SIZE:
            d_img = np.array(Image.fromarray((d_img * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) / 255.0
        depth_np = d_img

    if masks_file and os.path.exists(masks_file):
        try:
            import cv2
            with open(masks_file, "r") as f:
                coco = json.load(f)
            orig_h = coco.get("images", [{}])[0].get("height", IMG_SIZE)
            orig_w = coco.get("images", [{}])[0].get("width", IMG_SIZE)
            temp_map = np.zeros((orig_h, orig_w), dtype=np.int32) - 1
            for ann in coco.get("annotations", []):
                cid = ann.get("category_id", -1)
                for seg in ann.get("segmentation", []):
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(temp_map, [pts], int(cid))
            if temp_map.shape[0] != IMG_SIZE or temp_map.shape[1] != IMG_SIZE:
                temp_map = cv2.resize(temp_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            mask_map = temp_map
        except Exception as e:
            print(f"Warning parsing COCO masks from {masks_file}: {e}")

    return {
        "rgb": rgb_np,
        "depth": depth_np,
        "mask_map": mask_map,
    }


def rasterize_semantic_color(mask_map: np.ndarray) -> np.ndarray:
    """Converts a semantic category ID map into an RGB visualization."""
    H, W = mask_map.shape[:2]
    rgb = np.zeros((H, W, 3), dtype=np.uint8) + 245  # Clean off-white background
    for cid in range(len(ORGAN_CLASSES)):
        sel = (mask_map == cid)
        if sel.any():
            rgb[sel] = ORGAN_COLORS[cid]
    return rgb / 255.0


def compute_iou_per_class(mask_a: np.ndarray, mask_b: np.ndarray) -> Dict[str, float]:
    """Compute per-class Intersection over Union."""
    ious = {}
    for cid, name in enumerate(ORGAN_CLASSES):
        a = (mask_a == cid)
        b = (mask_b == cid)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            ious[name] = float("nan")
        else:
            ious[name] = float(inter / union)
    return ious


def main():
    print("=" * 80)
    print("RUNNING MULTI-MODAL PER-ORGAN MASK & DEPTH ROUNDTRIP EVALUATION")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    geo_builder = HeliosPlantGeometryBuilder()
    renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE).to(DEVICE)

    num_rows = len(TEST_PLANTS)
    # Columns:
    # 0: Helios GT Raytrace RGB
    # 1: Helios GT Semantic Organ Mask
    # 2: Helios GT Raytraced Depth
    # 3: Helios Recon Raytrace RGB (from 13D XML)
    # 4: Helios Recon Semantic Organ Mask (from 13D XML)
    # 5: Helios Recon Raytraced Depth (from 13D XML)
    # 6: PyTorch 13D Direct Render RGB
    fig, axes = plt.subplots(num_rows, 7, figsize=(32, 4.8 * num_rows), facecolor="#0a0a14")
    plt.subplots_adjust(wspace=0.03, hspace=0.08, left=0.06, right=0.98, top=0.93, bottom=0.04)

    col_titles = [
        "Helios GT\nRaytrace RGB",
        "Helios GT\nOrgan Mask (COCO)",
        "Helios GT\nRaytrace Depth",
        "Helios Reconstructed\nRaytrace RGB (14D XML)",
        "Helios Reconstructed\nOrgan Mask (14D XML)",
        "Helios Reconstructed\nRaytrace Depth (14D XML)",
        "PyTorch 14D\nDirect Differentiable"
    ]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", color="#7ee8fa", pad=12)

    for row_idx, (label, orig_xml_rel, tag) in enumerate(TEST_PLANTS):
        orig_xml_path = os.path.join(REPO_ROOT, orig_xml_rel)
        if not os.path.exists(orig_xml_path):
            print(f"Skipping {label} (missing: {orig_xml_path})")
            continue

        print(f"\n--- Processing [{label}] ---")
        # 1. Load Original XML & extract 14D Part Tensor
        arr = PlantOrganArray.from_xml_file(orig_xml_path)
        part_tensor = arr.to_part_tensor(device=DEVICE)

        # 2. Render PyTorch 14D Direct
        mesh_14d = geo_builder.build_mesh_from_part_tensor(part_tensor, device=DEVICE)
        rgbd_14d = renderer.forward(
            mesh_14d,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=5.0,
            background="ground",
            focus_plant=True,
            image_size=IMG_SIZE,
            include_depth=True
        )
        rgb_pytorch_np = rgbd_14d[:3].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()

        # 3. Export 14D Part Tensor -> Helios XML via Analytical IK
        recon_xml_str = assemble_part_tensor_to_xml(part_tensor.cpu(), existence_threshold=0.5)
        recon_xml_path = os.path.join(SCRATCH_DIR, f"recon_{tag}.xml")
        with open(recon_xml_path, "w", encoding="utf-8") as f:
            f.write(recon_xml_str)

        # 4. Render Helios GT
        print("  Rendering Helios GT (RGB + Depth + COCO Masks)...")
        helios_gt = render_helios_full(orig_xml_path, f"gt_{tag}")

        # 5. Render Helios Reconstructed XML
        print("  Rendering Helios Reconstructed XML (RGB + Depth + COCO Masks)...")
        helios_recon = render_helios_full(recon_xml_path, f"recon_{tag}")

        # 6. Compute Quantitative Metrics
        ious = compute_iou_per_class(helios_gt["mask_map"], helios_recon["mask_map"])
        valid_ious = [v for v in ious.values() if not math.isnan(v)]
        m_iou = np.mean(valid_ious) if valid_ious else 0.0

        # Mask overlap (all foreground)
        fg_gt = (helios_gt["mask_map"] >= 0)
        fg_recon = (helios_recon["mask_map"] >= 0)
        fg_iou = np.logical_and(fg_gt, fg_recon).sum() / max(1, np.logical_or(fg_gt, fg_recon).sum())

        # Depth MSE on foreground
        d_gt = helios_gt["depth"]
        d_recon = helios_recon["depth"]
        d_mask = np.logical_and(fg_gt, fg_recon)
        depth_mse = float(np.mean((d_gt[d_mask] - d_recon[d_mask]) ** 2)) if d_mask.any() else 0.0
        depth_psnr = -10.0 * math.log10(max(depth_mse, 1e-8))

        print(f"  Results for {label}:")
        print(f"    - Foreground Mask IoU: {fg_iou*100:.1f}%")
        print(f"    - Mean Organ Class IoU: {m_iou*100:.1f}%")
        for cname, v in ious.items():
            if not math.isnan(v):
                print(f"      * {cname:<10s}: {v*100:.1f}%")
        print(f"    - Depth Map PSNR: {depth_psnr:.2f} dB (MSE: {depth_mse:.6f})")

        # 7. Plotting
        ax_row = axes[row_idx]

        # Col 0: Helios GT RGB
        ax = ax_row[0]
        ax.imshow(helios_gt["rgb"])
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.set_ylabel(label, fontsize=12, fontweight="bold", color="#f0f0f0", rotation=0, labelpad=70, va="center")
        ax.text(0.03, 0.03, "Helios GT", transform=ax.transAxes, fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 1: Helios GT Semantic Organ Mask
        ax = ax_row[1]
        ax.imshow(rasterize_semantic_color(helios_gt["mask_map"]))
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, "GT Organ Mask", transform=ax.transAxes, fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 2: Helios GT Depth
        ax = ax_row[2]
        ax.imshow(helios_gt["depth"], cmap="magma")
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, "GT Depth", transform=ax.transAxes, fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 3: Helios Recon RGB
        ax = ax_row[3]
        ax.imshow(helios_recon["rgb"])
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, f"14D XML Recon\nIoU: {fg_iou*100:.1f}%", transform=ax.transAxes, fontsize=9, color="#7ee8fa",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 4: Helios Recon Organ Mask
        ax = ax_row[4]
        ax.imshow(rasterize_semantic_color(helios_recon["mask_map"]))
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, f"Recon Mask\nmIoU: {m_iou*100:.1f}%", transform=ax.transAxes, fontsize=9, color="#7ee8fa",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 5: Helios Recon Depth
        ax = ax_row[5]
        ax.imshow(helios_recon["depth"], cmap="magma")
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, f"Recon Depth\nPSNR: {depth_psnr:.1f} dB", transform=ax.transAxes, fontsize=9, color="#ffd166",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 6: PyTorch 14D Direct
        ax = ax_row[6]
        ax.imshow(rgb_pytorch_np)
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, "PyTorch 14D (Direct)", transform=ax.transAxes, fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

    # Add legend for semantic organ classes at bottom
    patches = [mpatches.Patch(color=ORGAN_COLORS[i]/255.0, label=ORGAN_CLASSES[i]) for i in range(len(ORGAN_CLASSES))]
    fig.legend(handles=patches, loc="lower center", ncol=len(ORGAN_CLASSES), fontsize=12,
               facecolor="#151525", edgecolor="#444466", labelcolor="white", bbox_to_anchor=(0.52, 0.005))

    save_path = os.path.join(OUTPUT_DIR, "fig10_helios_per_organ_mask_comparison.png")
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSuccessfully generated and saved per-organ comparison figure:\n  -> {save_path}")


if __name__ == "__main__":
    main()
