"""
Visual Demonstration & Verification of Soft Existence Alpha Rendering ("fading/ghost").

Renders a plant organ at various existence values:
  - existence = 0.1 (faint ghost)
  - existence = 0.35 (semi-translucent)
  - existence = 0.70 (mostly opaque)
  - existence = 1.0 (crisp ground truth)

Saves diagnostic visualization to docs/results/assets/test_soft_existence_visual.png.
"""

import os
import torch
import matplotlib.pyplot as plt

from diffusion_based.models.plant_organ_array import ORGAN_LEAF
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def run_soft_existence_demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SoftExistence] Running on {device}")

    geo_builder = HeliosPlantGeometryBuilder()
    renderer = HeliosPyTorchRenderer(image_size=256)

    # 1 Leaf Canonical 14D Part Tensor
    part_14d = torch.zeros((1, 14), dtype=torch.float32, device=device)
    part_14d[0, 0] = float(ORGAN_LEAF)
    part_14d[0, 1:4] = torch.tensor([0.0, 0.0, 0.05], device=device)
    part_14d[0, 4:10] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
    part_14d[0, 10:13] = torch.tensor([0.12, 0.12, 0.12], device=device)

    exist_levels = [0.10, 0.35, 0.70, 1.00]
    renders = []

    for e_val in exist_levels:
        e_tensor = torch.tensor([e_val], dtype=torch.float32, device=device)
        mesh_dict = geo_builder.build_mesh_from_part_tensor(part_14d, existence=e_tensor, device=device)
        rgbd = renderer(
            mesh_dict,
            azimuth_deg=45.0,
            elevation_deg=60.0,
            camera_height=0.8,
            include_depth=True,
            differentiable=True,
        )
        img_np = rgbd[:3].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        renders.append(img_np)

    os.makedirs("docs/results/assets", exist_ok=True)
    out_png = "docs/results/assets/test_soft_existence_visual.png"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    titles = [
        "existence = 0.10\n(Faint Ghost / Noise)",
        "existence = 0.35\n(Semi-Translucent)",
        "existence = 0.70\n(Solidifying)",
        "existence = 1.00\n(Crisp XML GT)",
    ]

    for ax, img, title in zip(axes, renders, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Soft Existence Continuous Alpha Rendering (Translucent -> Opaque) for Diffusion & Opt", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"[SoftExistence] Visualization saved to {out_png}")


if __name__ == "__main__":
    run_soft_existence_demo()
