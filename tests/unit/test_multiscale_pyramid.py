"""
Verification of Progressive Multi-Scale Zoom Pyramid (1x -> 2x -> 4x -> 8x).
Tests on Canonical Unifoliate Seedling (14cm) and DAP 50 Branching Plant (60cm).
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    print("=" * 70)
    print("TESTING PROGRESSIVE MULTI-SCALE ZOOM PYRAMID (1x, 2x, 4x, 8x)")
    print("=" * 70)

    geo_builder = HeliosPlantGeometryBuilder()
    renderer = HeliosPyTorchRenderer(image_size=256).to(DEVICE)
    scales = [1.0, 2.0, 4.0, 8.0]
    ref_window = 1.2  # 1.2m x 1.2m global canopy window

    # 1. Canonical Unifoliate Seedling
    target_14d_path = "scratch/target_unifoliate_14d.pt"
    if os.path.exists(target_14d_path):
        pt_seedling = torch.load(target_14d_path, map_location=DEVICE)
    else:
        from scratch.make_target_unifoliate import create_canonical_unifoliate_gt
        pt_seedling = create_canonical_unifoliate_gt(device=DEVICE)

    mesh_seedling = geo_builder.build_mesh_from_part_tensor(pt_seedling, device=DEVICE)
    pyramid_seedling = renderer.render_multiscale_pyramid(
        mesh_seedling,
        scales=scales,
        reference_window_size=ref_window,
        camera_height=5.0,
        background="ground",
        include_depth=True,
    )

    # 2. DAP 50 Branching Plant
    dap50_xml = "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_plant_0000.xml"
    if os.path.exists(dap50_xml):
        arr50 = PlantOrganArray.from_xml_file(dap50_xml)
        pt_dap50 = arr50.to_part_tensor(device=DEVICE)
        mesh_dap50 = geo_builder.build_mesh_from_part_tensor(pt_dap50, device=DEVICE)
        pyramid_dap50 = renderer.render_multiscale_pyramid(
            mesh_dap50,
            scales=scales,
            reference_window_size=ref_window,
            camera_height=5.0,
            background="ground",
            include_depth=True,
        )
    else:
        pyramid_dap50 = None

    # Visualization
    num_rows = 2 if pyramid_dap50 is not None else 1
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4.2 * num_rows), facecolor="#0a0a14")
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    window_labels = {
        1.0: "1.0x Global (1.2m x 1.2m)\n[Metric Scale Preserved]",
        2.0: "2.0x Sub-Canopy (0.6m x 0.6m)\n[Canopy Context]",
        4.0: "4.0x Plant-Level (0.3m x 0.3m)\n[Branch Alignment]",
        8.0: "8.0x Organ-Level (0.15m x 0.15m)\n[Dense Seedling Gradients]",
    }

    # Row 0: Seedling
    for col, s in enumerate(scales):
        rgbd = pyramid_seedling[s]
        rgb = rgbd[:3].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
        ax = axes[0, col]
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(window_labels[s], fontsize=11, fontweight="bold", color="#7ee8fa", pad=10)
        if col == 0:
            ax.set_ylabel("DAP 10 Seedling\n(Width = 14 cm)", fontsize=12, fontweight="bold", color="white", rotation=0, labelpad=80, va="center")

    # Row 1: DAP 50
    if pyramid_dap50 is not None:
        for col, s in enumerate(scales):
            rgbd = pyramid_dap50[s]
            rgb = rgbd[:3].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
            ax = axes[1, col]
            ax.imshow(rgb)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel("DAP 50 Branching\n(Width = 60 cm)", fontsize=12, fontweight="bold", color="white", rotation=0, labelpad=80, va="center")

    plt.tight_layout()
    out_dir = "docs/results/assets"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig13_progressive_multiscale_pyramid.png")
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Multi-scale pyramid verification saved to:\n  -> {out_path}")

if __name__ == "__main__":
    main()
