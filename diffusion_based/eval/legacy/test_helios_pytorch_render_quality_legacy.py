"""
Quantitative & Visual Evaluation Script for PyTorch Helios Renderer using PlantOrganArray Tensor.
Compares PyTorch plant renders directly against Helios C++ visualizer reference images (_vis.jpeg).
Saves output figure directly to project folder `diffusion_based/eval/output`.
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes basic SSIM between two RGB images in [0, 1]."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_val)


def extract_plant_mask(img_np: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    """Extracts plant binary mask (True where pixel is not background).

    DEPRECATED: For Helios visualizer reference images the color-based
    thresholding is unreliable. Use the organ-type buffer mask instead.
    """
    gray = img_np.mean(axis=-1)
    mask = gray < threshold
    return mask


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def evaluate_render_quality(
    xml_path: str,
    helios_img_path: str,
    output_dir: str = "diffusion_based/eval/output",
    azimuth_deg: float = 0.0,
    elevation_deg: float = 90.0,
    use_generic_leaves: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading XML into PlantOrganArray Tensor: {xml_path}")
    organ_array = PlantOrganArray.from_xml_file(xml_path)
    print(f"Organ Array Tensor shape: {organ_array.tensor.shape}")

    print(f"Rendering plant with PyTorch Helios Renderer (Device: {device}, Elevation: Top View {elevation_deg}°)...")
    geo_builder = HeliosPlantGeometryBuilder(use_generic_leaves=use_generic_leaves, leaf_scale_factor=1.0, tube_radial_subdivisions=6)
    renderer = HeliosPyTorchRenderer(image_size=256)
    renderer.geo_builder = geo_builder

    rendered_tensor = renderer.render_organ_array(
        organ_array, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, camera_height=1.0, background="ground", device=device
    ) # (3, 256, 256)

    pred_np = rendered_tensor.permute(1, 2, 0).cpu().numpy() # (256, 256, 3)

    # Exact organ-type masks from the PyTorch rasterizer (1=stem/petiole, 2=leaf)
    mesh_dict = renderer.geo_builder.build_mesh_from_organ_array(organ_array, device=device)
    organ_type_buffer = renderer.render_organ_type_buffer(
        mesh_dict, azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, camera_height=1.0, focus_plant=True
    )
    leaf_mask_pred = (organ_type_buffer == 2).cpu().numpy()
    plant_mask_pred = (organ_type_buffer > 0).cpu().numpy()

    # Load Reference Image
    print(f"Loading Helios reference image: {helios_img_path}")
    ref_pil = Image.open(helios_img_path).convert("RGB").resize((256, 256))
    ref_np = np.array(ref_pil, dtype=np.float32) / 255.0

    # Quantitative Metrics
    mae = float(np.abs(pred_np - ref_np).mean())
    mse = float(((pred_np - ref_np) ** 2).mean())
    ssim_score = compute_ssim(pred_np, ref_np)

    # Plant mask from the organ-type buffer only; no color thresholding on
    # the Helios visualizer image because its background/ground colors vary.
    iou_score = compute_iou(plant_mask_pred, leaf_mask_pred)

    print("\n" + "=" * 50)
    print("   PYTORCH ORGAN ARRAY HELIOS RENDER QUALITY METRICS   ")
    print("=" * 50)
    print(f"  MAE (Mean Absolute Error) : {mae:.4f}")
    print(f"  MSE (Mean Squared Error)  : {mse:.4f}")
    print(f"  SSIM (Structural Similarity): {ssim_score:.4f}")
    print(f"  Plant vs Leaf Mask IoU  : {iou_score:.4f}")
    print("=" * 50 + "\n")

    # Generate Visual Side-by-Side Comparison (5 columns)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(ref_np)
    axes[0].set_title("Helios Reference C++")
    axes[0].axis("off")

    axes[1].imshow(pred_np)
    axes[1].set_title(f"PyTorch Organ Array (Top {int(elevation_deg)}°, Generic:{use_generic_leaves})")
    axes[1].axis("off")

    diff_map = np.abs(pred_np - ref_np).mean(axis=-1)
    im3 = axes[2].imshow(diff_map, cmap="hot", vmin=0, vmax=0.5)
    axes[2].set_title(f"Abs Diff Map (MAE={mae:.3f})")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(leaf_mask_pred, cmap="gray")
    axes[3].set_title("PyTorch Leaf Mask")
    axes[3].axis("off")

    mask_comp = np.zeros((256, 256, 3))
    mask_comp[plant_mask_pred, 0] = 1.0 # Red = Plant mask
    mask_comp[leaf_mask_pred, 1] = 1.0 # Green = Leaf mask
    axes[4].imshow(mask_comp)
    axes[4].set_title(f"Mask Overlay (Plant vs Leaf IoU={iou_score:.3f})")
    axes[4].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "render_quality_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved visual comparison figure to project folder: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default="/home/lion397/codes/image-to-l-system/Digital-Crops/projects/syntheticdata_generation/build/output/cowpea_dap005_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"
    )
    parser.add_argument(
        "--helios-img",
        default="/home/lion397/codes/image-to-l-system/Digital-Crops/projects/syntheticdata_generation/build/output/cowpea_dap005_seed00_caz000_h1.0_se045_saz180_0000_vis.jpeg"
    )
    parser.add_argument("--output-dir", default="diffusion_based/eval/output")
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--elevation", type=float, default=90.0)
    parser.add_argument("--generic-leaves", action="store_true", default=False)
    args = parser.parse_args()

    evaluate_render_quality(args.xml, args.helios_img, args.output_dir, azimuth_deg=args.azimuth, elevation_deg=args.elevation, use_generic_leaves=args.generic_leaves)
