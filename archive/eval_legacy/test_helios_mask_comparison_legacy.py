"""
Leaf Mask Comparison Script comparing Helios C++ radiation/visualizer output against PyTorch Organ Array render mask.
Calculates IoU, Dice coefficient, and saves visual comparison figure to project folder `diffusion_based/eval/output/mask_comparison.png`.
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def extract_leaf_mask(img_np: np.ndarray, bg_color: np.ndarray = np.array([0.72, 0.62, 0.50])) -> np.ndarray:
    """Extracts binary leaf mask by checking color difference from background or green thresholding."""
    diff_from_bg = np.linalg.norm(img_np - bg_color, axis=-1)
    mask = diff_from_bg > 0.12
    return mask


def evaluate_leaf_mask_comparison(
    xml_path: str,
    helios_img_path: str,
    output_dir: str = "diffusion_based/eval/output"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading Organ Array Tensor from {xml_path}...")
    organ_array = PlantOrganArray.from_xml_file(xml_path)

    print("Rendering PyTorch Organ Array in Top View (Elevation: 90°)...")
    renderer = HeliosPyTorchRenderer(image_size=256)
    pytorch_render_t = renderer.render_organ_array(
        organ_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="ground", device=device
    )

    pytorch_rgb = pytorch_render_t.permute(1, 2, 0).cpu().numpy()

    # Load Helios Reference Image
    print(f"Loading Helios Reference Image: {helios_img_path}")
    ref_pil = Image.open(helios_img_path).convert("RGB").resize((256, 256))
    helios_rgb = np.array(ref_pil, dtype=np.float32) / 255.0

    # Extract Masks
    pytorch_mask = extract_leaf_mask(pytorch_rgb, bg_color=np.array([0.72, 0.62, 0.50]))
    helios_mask = (helios_rgb.mean(axis=-1) < 0.85)

    intersection = (pytorch_mask & helios_mask).sum()
    union = (pytorch_mask | helios_mask).sum()
    iou = float(intersection / union) if union > 0 else 1.0
    dice = float(2 * intersection / (pytorch_mask.sum() + helios_mask.sum())) if (pytorch_mask.sum() + helios_mask.sum()) > 0 else 1.0

    print("\n" + "=" * 50)
    print("   HELIOS VS PYTORCH ORGAN ARRAY LEAF MASK EVALUATION   ")
    print("=" * 50)
    print(f"  PyTorch Mask Pixel Count  : {pytorch_mask.sum()}")
    print(f"  Helios Mask Pixel Count   : {helios_mask.sum()}")
    print(f"  Leaf Mask IoU             : {iou:.4f}")
    print(f"  Leaf Mask Dice Coefficient: {dice:.4f}")
    print("=" * 50 + "\n")

    # Plot Side-by-Side Comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(helios_rgb)
    axes[0].set_title("Helios Reference C++")
    axes[0].axis("off")

    axes[1].imshow(pytorch_rgb)
    axes[1].set_title("PyTorch Organ Array (Top View)")
    axes[1].axis("off")

    axes[2].imshow(helios_mask, cmap="gray")
    axes[2].set_title("Helios Leaf Mask")
    axes[2].axis("off")

    axes[3].imshow(pytorch_mask, cmap="gray")
    axes[3].set_title(f"PyTorch Leaf Mask (IoU={iou:.3f})")
    axes[3].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "mask_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved leaf mask comparison figure to project folder: {save_path}")


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
    args = parser.parse_args()

    evaluate_leaf_mask_comparison(args.xml, args.helios_img, args.output_dir)
