#!/usr/bin/env python3
"""
Generates a comprehensive, publication-quality Dataset Card visualization
for the newly generated Cowpea synthetic dataset.

Displays representative samples across growth stages (DAP 1 to 100), showing:
  - Drone top-down radiation RGB view (what the model sees)
  - 3D Organ mesh geometry (ground truth organ parts)
  - Canopy Height Map (CHM / Depth)
  - Phenotypic metrics (organ count, canopy diameter, height)
"""

import os
import sys
import glob
import re

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

def main():
    repo_root = "/home/lion397/codes/image-to-l-system"
    data_dir = os.path.join(repo_root, "dataset", "helios_data", "cowpea")
    cache_dir = os.path.join(repo_root, "dataset", "cache", "cowpea_curv26")
    output_card_path = os.path.join(repo_root, "dataset", "helios_data", "dataset_card_cowpea.png")
    artifact_card_path = "/home/lion397/.gemini/antigravity-ide/brain/50a7f983-db89-458f-a5c9-595340ecf41a/dataset_card_cowpea.png"

    # Selected target DAPs across the lifespan
    target_daps = [2, 10, 25, 50, 75, 95]
    samples_to_plot = []

    for td in target_daps:
        pattern = os.path.join(data_dir, f"*_dap{td:03d}_*0000_rad.jpeg")
        matches = sorted(glob.glob(pattern))
        if not matches:
            # fallback to closest available DAP
            all_rads = sorted(glob.glob(os.path.join(data_dir, "*0000_rad.jpeg")))
            matches = [min(all_rads, key=lambda p: abs(int(re.search(r"dap(\d+)", p).group(1)) - td))]
        rad_path = matches[0]
        prefix = os.path.basename(rad_path).replace("_0000_rad.jpeg", "")
        xml_path = os.path.join(data_dir, f"{prefix}_0000_plant_0000.xml")
        pt_path = os.path.join(cache_dir, f"{prefix}.pt")
        samples_to_plot.append({
            "dap": td,
            "prefix": prefix,
            "rad_path": rad_path,
            "xml_path": xml_path,
            "pt_path": pt_path
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geo_builder = HeliosPlantGeometryBuilder()
    renderer = HeliosPyTorchRenderer(image_size=256, device=device)

    ncols = len(samples_to_plot)
    nrows = 3  # Row 0: Drone RGB, Row 1: 3D Organ Mesh, Row 2: Canopy Height Map
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 11.5), facecolor="#14181c")

    plt.subplots_adjust(wspace=0.08, hspace=0.18, left=0.04, right=0.98, top=0.88, bottom=0.04)

    fig.suptitle("Synthetic Cowpea Multimodal Dataset Card\nHigh-Fidelity Phenotyping Across Lifespan (DAP 1 – 100)",
                 fontsize=16, fontweight="bold", color="#f0f4f8", y=0.96)

    row_labels = ["Drone Top-down RGB", "3D Organ Mesh (GT)", "Canopy Height Map (CHM)"]

    for col, s in enumerate(samples_to_plot):
        # 1. Load drone RGB
        rgb_img = Image.open(s["rad_path"]).convert("RGB")
        axes[0, col].imshow(rgb_img)
        axes[0, col].set_title(f"DAP {s['dap']}", fontsize=13, fontweight="bold", color="#38bdf8", pad=8)
        axes[0, col].axis("off")

        # 2. Render 3D Organ Mesh from XML
        total_organs = 0
        canopy_w = 0.0
        canopy_h = 0.0
        try:
            arr = PlantOrganArray.from_xml_file(s["xml_path"])
            part_tensor = arr.to_part_tensor(device=device)
            total_organs = part_tensor.shape[0]
            active_parts = part_tensor[part_tensor[:, 0] > 0]
            
            # calculate bounding box
            xyz = active_parts[:, 1:4]
            if len(xyz) > 0:
                canopy_w = float((xyz[:, 0].max() - xyz[:, 0].min() + xyz[:, 1].max() - xyz[:, 1].min()).item() * 0.5)
                canopy_h = float((xyz[:, 2].max() - xyz[:, 2].min()).item())

            mesh = renderer.geo_builder.build_mesh_from_part_tensor(active_parts, device=device)
            zoom = 8.0 if s["dap"] <= 15.0 else 1.0
            with torch.no_grad():
                rgbd = renderer.render_mesh(
                    mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                    background="ground", focus_plant=True, include_depth=True,
                    reference_window_size=1.2, zoom_factor=zoom
                )
            mesh_np = rgbd[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            chm_np = rgbd[3].clamp(min=0.0).cpu().numpy()
            
            axes[1, col].imshow(mesh_np)
            axes[2, col].imshow(chm_np, cmap="viridis")
        except Exception as e:
            axes[1, col].imshow(np.zeros((256, 256, 3)))
            axes[1, col].text(128, 128, f"Render fail: {e}", color="red", ha="center", fontsize=8)
            axes[2, col].imshow(np.zeros((256, 256)), cmap="viridis")
        axes[1, col].axis("off")
        axes[2, col].axis("off")

        # Annotation under column
        annot_text = f"Organs: {total_organs}\nWidth: {canopy_w:.2f}m | H: {canopy_h:.2f}m"
        axes[2, col].text(0.5, -0.18, annot_text, transform=axes[2, col].transAxes,
                          ha="center", va="top", fontsize=10, color="#94a3b8", fontweight="medium")

    # Add row labels on the left
    for row_idx, rlabel in enumerate(row_labels):
        axes[row_idx, 0].text(-0.15, 0.5, rlabel, transform=axes[row_idx, 0].transAxes,
                              rotation=90, va="center", ha="right", fontsize=11, fontweight="bold", color="#e2e8f0")

    plt.savefig(output_card_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(artifact_card_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Dataset card successfully saved to {output_card_path} and {artifact_card_path}")

if __name__ == "__main__":
    main()
