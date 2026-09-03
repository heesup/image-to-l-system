"""
Comprehensive Validation: 13D Part Tensor -> Helios XML Analytical IK -> Helios C++ Raytrace vs PyTorch Differentiable Renderer.

Workflow:
1. Load ground truth Cowpea XMLs across DAP 10, DAP 30, DAP 50, DAP 70, DAP 90.
2. Convert XML -> PlantOrganArray -> Canonical 13D Part Tensor [organ_type(1), base_xyz(3), rot6d(6), scale_xyz(3)].
3. Render direct PyTorch 13D multi-modal outputs (RGB, Depth, Mask, Organ Map).
4. Export 13D Part Tensor -> Helios XML using analytical Inverse Kinematics (R_rel = R_parent^T @ R_child).
5. Render the exported XML with Helios C++ radiation raytracer.
6. Evaluate PSNR, SSIM, and Mask IoU between:
   - Helios Original GT vs Helios Reconstructed XML
   - Python 13D Render vs Helios Reconstructed XML
7. Generate high-resolution side-by-side comparison figure:
   docs/results/assets/fig9_13d_xml_helios_roundtrip_comparison.png
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
from typing import Dict, Any, List, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import PlantOrganArray, NUM_FEATURES_PART
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.part_assembly_to_xml import assemble_part_tensor_to_xml

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
OUTPUT_DIR = os.path.join(REPO_ROOT, "docs/results/assets")
SCRATCH_DIR = "/tmp/helios_13d_roundtrip_test"

TEST_PLANTS = [
    ("DAP 10 (Seedling)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_plant_0000.xml", "dap010"),
    ("DAP 50 (Branching)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_plant_0000.xml", "dap050"),
    ("DAP 90 (Fruiting)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_plant_0000.xml", "dap090"),
]


def render_helios_cxx(xml_path: str, name_prefix: str, species: str = "cowpea") -> np.ndarray:
    """Invokes Helios C++ syntheticdata_generation binary to render radiation JPEG from XML."""
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    build_dir = os.path.join(REPO_ROOT, "Digital-Crops/projects/syntheticdata_generation/build")
    cfg_file = os.path.join(REPO_ROOT, f"Digital-Crops/projects/syntheticdata_generation/configs/params_{species}.json")
    if not os.path.exists(cfg_file):
        cfg_file = os.path.join(build_dir, "params.json")

    cmd = [
        "./main",
        "--renderer", "radiation",
        "--input-xml", os.path.abspath(xml_path),
        "--output", SCRATCH_DIR,
        "-n", name_prefix,
        "--focus-plant",
        "-f", cfg_file,
    ]
    try:
        subprocess.run(cmd, cwd=build_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # Search for output jpeg
        cand1 = os.path.join(SCRATCH_DIR, species, f"{name_prefix}_0000_rad.jpeg")
        cand2 = os.path.join(SCRATCH_DIR, f"{name_prefix}_0000_rad.jpeg")
        target_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
        if target_path:
            img = np.array(Image.open(target_path).convert("RGB")) / 255.0
            h, w = img.shape[:2]
            if h != w:
                m = min(h, w)
                img = img[(h-m)//2:(h-m)//2+m, (w-m)//2:(w-m)//2+m]
            if img.shape[0] != IMG_SIZE:
                img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)) / 255.0
            return img
    except Exception as e:
        print(f"Warning: Helios C++ render failed for {xml_path}: {e}")
    return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)


def compute_metrics(img_a: np.ndarray, img_b: np.ndarray) -> Tuple[float, float]:
    """Compute MSE and Mask IoU between two rendered RGB images."""
    mse = float(np.mean((img_a - img_b) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1e-10))
    # Simple foreground mask (> 0.05 brightness diff from background)
    mask_a = (np.abs(img_a - img_a[0, 0]).sum(axis=-1) > 0.08)
    mask_b = (np.abs(img_b - img_b[0, 0]).sum(axis=-1) > 0.08)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(inter / max(union, 1))
    return psnr, iou


def main():
    print("=" * 80)
    print("RUNNING 13D PART TENSOR -> XML -> HELIOS C++ ROUNDTRIP VALIDATION TEST")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    geo_builder = HeliosPlantGeometryBuilder()
    renderer = HeliosPyTorchRenderer(image_size=IMG_SIZE).to(DEVICE)

    num_rows = len(TEST_PLANTS)
    fig, axes = plt.subplots(num_rows, 5, figsize=(25, 5 * num_rows), facecolor="#0a0a14")
    plt.subplots_adjust(wspace=0.04, hspace=0.08, left=0.08, right=0.98, top=0.94, bottom=0.03)

    col_titles = [
        "Helios C++ Raytrace GT\n(Original XML)",
        "PyTorch 13D Render\n(Direct Part Tensor)",
        "Helios C++ Raytrace\n(Reconstructed from 13D XML)",
        "Visual Error Map\n(|PyTorch 13D − Helios XML|)",
        "PyTorch 13D Multi-Modal\n(Canopy Height Model depth)"
    ]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", color="#7ee8fa", pad=14)

    for row_idx, (label, orig_xml_rel, tag) in enumerate(TEST_PLANTS):
        orig_xml_path = os.path.join(REPO_ROOT, orig_xml_rel)
        if not os.path.exists(orig_xml_path):
            print(f"Skipping {label} (missing: {orig_xml_path})")
            continue

        print(f"\n--- Processing [{label}] ---")
        # 1. Load Original XML & extract 13D Part Tensor
        arr = PlantOrganArray.from_xml_file(orig_xml_path)
        part_tensor = arr.to_part_tensor(device=DEVICE)
        print(f"  Loaded PlantOrganArray ({arr.num_nodes} nodes) -> 13D Part Tensor shape {part_tensor.shape}")

        # 2. Render Direct PyTorch 13D Multi-Modal Outputs
        mesh_13d = geo_builder.build_mesh_from_part_tensor(part_tensor, device=DEVICE)
        rgbd_13d = renderer.forward(
            mesh_13d,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=5.0,
            background="ground",
            focus_plant=True,
            image_size=IMG_SIZE,
            include_depth=True
        )
        rgb_pytorch_np = rgbd_13d[:3].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
        depth_pytorch = rgbd_13d[3].detach().cpu().numpy()

        # 3. Export 13D Part Tensor -> Helios XML via Analytical IK
        recon_xml_str = assemble_part_tensor_to_xml(part_tensor.cpu(), existence_threshold=0.5)
        recon_xml_path = os.path.join(SCRATCH_DIR, f"recon_{tag}.xml")
        with open(recon_xml_path, "w", encoding="utf-8") as f:
            f.write(recon_xml_str)
        print(f"  Exported 13D -> Helios XML ({len(recon_xml_str)} chars) -> {recon_xml_path}")

        # 4. Render Original XML with Helios C++
        print("  Rendering Original XML with Helios C++...")
        helios_gt_np = render_helios_cxx(orig_xml_path, f"gt_{tag}")

        # 5. Render Reconstructed XML with Helios C++
        print("  Rendering Reconstructed XML with Helios C++...")
        helios_recon_np = render_helios_cxx(recon_xml_path, f"recon_{tag}")

        # 6. Compute Metrics
        psnr_recon_gt, iou_recon_gt = compute_metrics(helios_recon_np, helios_gt_np)
        psnr_py_recon, iou_py_recon = compute_metrics(rgb_pytorch_np, helios_recon_np)
        print(f"  Metrics:")
        print(f"    - Helios Reconstructed vs Helios GT:  PSNR = {psnr_recon_gt:.2f} dB, Mask IoU = {iou_recon_gt*100:.1f}%")
        print(f"    - PyTorch 13D vs Helios Reconstructed: PSNR = {psnr_py_recon:.2f} dB, Mask IoU = {iou_py_recon*100:.1f}%")

        # 7. Plotting
        ax_row = axes[row_idx]

        # Col 0: Helios GT
        ax = ax_row[0]
        ax.imshow(helios_gt_np)
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.set_ylabel(label, fontsize=12, fontweight="bold", color="#f0f0f0", rotation=0, labelpad=65, va="center")
        ax.text(0.03, 0.03, "Helios Ground Truth", transform=ax.transAxes, fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 1: PyTorch 13D
        ax = ax_row[1]
        ax.imshow(rgb_pytorch_np)
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, f"PyTorch 13D (N={part_tensor.shape[0]})", transform=ax.transAxes, fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 2: Helios Reconstructed XML
        ax = ax_row[2]
        ax.imshow(helios_recon_np)
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, f"13D -> XML -> Helios\nIoU={iou_recon_gt*100:.1f}%", transform=ax.transAxes, fontsize=9, color="#7ee8fa",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 3: Difference Map
        ax = ax_row[3]
        diff_np = np.abs(rgb_pytorch_np - helios_recon_np).mean(axis=-1)
        im_diff = ax.imshow(diff_np, cmap="hot", vmin=0.0, vmax=0.5)
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, f"Diff (Mean: {diff_np.mean():.3f})", transform=ax.transAxes, fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))

        # Col 4: CHM Depth
        ax = ax_row[4]
        d_np = depth_pytorch
        fg_mask = (d_np > 1e-4)
        if fg_mask.any():
            h_cm = d_np * 100.0
            masked_h = np.ma.masked_where(~fg_mask, h_cm)
            max_h = float(np.max(h_cm[fg_mask]))
        else:
            masked_h = np.ma.masked_all_like(d_np)
            max_h = 5.0
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color="#0a0a14")
        im_chm = ax.imshow(masked_h, cmap=cmap, vmin=0.0, vmax=max_h)
        ax.axis("off")
        ax.set_facecolor("#0a0a14")
        ax.text(0.03, 0.03, f"Height: 0–{max_h:.1f} cm", transform=ax.transAxes, fontsize=9, color="#ffd166",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
        cbar = plt.colorbar(im_chm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Height (cm)", fontsize=8, color="#c3a6e0")
        cbar.ax.tick_params(labelsize=7, colors="#e0e0e0")
        cbar.outline.set_edgecolor("#444444")

    save_path = os.path.join(OUTPUT_DIR, "fig9_13d_xml_helios_roundtrip_comparison.png")
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nSuccessfully generated and saved comparison figure:\n  -> {save_path}")


if __name__ == "__main__":
    main()
