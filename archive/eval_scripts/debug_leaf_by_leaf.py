"""
Step-by-Step Sketchbook Debugging Grid for Plant Organ Array (DAP 10).
Renders plant organs incrementally (Stem only -> Stem + 1 Leaf Group -> Stem + 2 Leaf Groups -> ... -> Full Canopy)
and saves a multi-panel grid figure to `diffusion_based/eval/output/dap10_leaf_by_leaf_debug.png`.
Accelerated via CUDA GPU with Helios --focus-plant math.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def run_leaf_by_leaf_sketchbook_debug(
    xml_path: str = "Digital-Crops/projects/syntheticdata_generation/build/output/dap10_gt_0000_plant_0000.xml",
    output_dir: str = "diffusion_based/eval/output"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading Organ Array Tensor from {xml_path}...")
    organ_array = PlantOrganArray.from_xml_file(xml_path)
    N_rows = organ_array.tensor.shape[0]
    print(f"Organ Array shape: {organ_array.tensor.shape}")

    builder = HeliosPlantGeometryBuilder()
    renderer = HeliosPyTorchRenderer(image_size=512).to(device)
    renderer.geo_builder = builder

    # Render Incremental Frames (0 leaves, 1 leaf, 2 leaves, ..., 11 leaves)
    max_frames = 12
    cols = 4
    rows = (max_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for step in range(max_frames):
        max_l = None if step == max_frames - 1 else step

        mesh_dict = builder.build_mesh_from_organ_array(organ_array, device=device, max_leaves=max_l)
        img_t = renderer.forward(mesh_dict, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="ground", focus_plant=True)

        rgb_np = img_t.permute(1, 2, 0).cpu().numpy()

        title = "Full Canopy" if max_l is None else f"Step {step}: {step} Leaf Groups"
        axes[step].imshow(rgb_np)
        axes[step].set_title(title, fontsize=12, fontweight='bold')
        axes[step].axis("off")

    for step in range(max_frames, len(axes)):
        axes[step].axis("off")

    plt.suptitle(f"DAP 10 Plant Incremental Leaf-by-Leaf Sketchbook Debug Grid ({N_rows} phytomer nodes)", fontsize=16, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "dap10_leaf_by_leaf_debug.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n[SUCCESS] Saved DAP 10 Leaf-by-Leaf Sketchbook Debug Grid to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="Digital-Crops/projects/syntheticdata_generation/build/output/dap10_gt_0000_plant_0000.xml")
    parser.add_argument("--output-dir", default="diffusion_based/eval/output")
    args = parser.parse_args()

    run_leaf_by_leaf_sketchbook_debug(args.xml, args.output_dir)
