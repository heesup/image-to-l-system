"""
Verification script for the typed (N, 40) PlantOrganArray refactor.

Loads a Helios XML, builds both legacy (N, 94) and typed (N, 40) PlantOrganArray
representations, renders them through the PyTorch differentiable renderer, and
produces a 5-panel comparison figure:

  1. Helios C++ GT render (from existing _vis.jpeg)
  2. PyTorch typed-organ-array RGB render
  3. Helios GT leaf mask
  4. PyTorch typed leaf mask
  5. Overlay: typed leaf mask vs legacy leaf mask

Also prints pixel-match metrics (MAE, SSIM, IoU) between legacy and typed renders.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two RGB images in [0, 1]."""
    try:
        from skimage.metrics import structural_similarity as ssim
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=win_size))
    except Exception as e:
        print(f"Warning: skimage SSIM failed ({e}), using fallback MSE-based similarity")
        mse = float(np.mean((img1 - img2) ** 2))
        return float(max(0.0, 1.0 - 5.0 * mse))


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = float((mask1 & mask2).sum())
    union = float((mask1 | mask2).sum())
    return intersection / union if union > 0 else 1.0


def load_gt_image_and_mask(xml_path: str, image_size: int = 256):
    """Load the matching Helios C++ GT render and derive a leaf mask."""
    prefix = os.path.basename(xml_path).split("_plant_")[0]
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    vis_path = os.path.join(xml_dir, f"{prefix}_vis.jpeg")

    if not os.path.exists(vis_path):
        return None, None

    img = Image.open(vis_path).convert("RGB").resize((image_size, image_size))
    img_np = np.array(img).astype(np.float32) / 255.0

    # Approximate leaf mask: green-ish pixels that are not ground brown.
    # The Helios GT ground is a brownish pattern; leaves are green.
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    green = (g > 0.35) & (g > r + 0.05) & (g > b + 0.05)
    not_ground = ~((r > 0.5) & (g > 0.4) & (b < 0.35) & (r - b > 0.15))
    leaf_mask = green & not_ground
    return img_np, leaf_mask


def render_organ_array(organ_array: PlantOrganArray, renderer: HeliosPyTorchRenderer,
                       device: torch.device, image_size: int = 256):
    """Render an organ array and return RGB numpy image and leaf mask."""
    rgb_t = renderer.render_organ_array(
        organ_array,
        azimuth_deg=0.0,
        elevation_deg=90.0,
        camera_height=1.0,
        background="ground",
        device=device,
        differentiable=False,
        focus_plant=True,
        existence_threshold=0.5,
    )
    # PyTorch renderer output is (3, H, W), row-0 = bottom.
    rgb_np = rgb_t.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

    type_buffer = renderer.render_organ_type_buffer(
        renderer.geo_builder.build_mesh_from_organ_array(organ_array, device=device),
        azimuth_deg=0.0,
        elevation_deg=90.0,
        camera_height=1.0,
        focus_plant=True,
        image_size=image_size,
    )
    # Organ type 2 = Leaf
    leaf_mask = (type_buffer.detach().cpu().numpy() == 2)
    return rgb_np, leaf_mask


def main():
    output_dir = os.path.join(repo_root, "diffusion_based", "eval", "output")
    os.makedirs(output_dir, exist_ok=True)

    xml_path = os.path.join(
        repo_root,
        "Digital-Crops",
        "projects",
        "syntheticdata_generation",
        "build",
        "output",
        "dap10_gt_0000_plant_0000.xml",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 256
    renderer = HeliosPyTorchRenderer(image_size=image_size)

    # Load legacy and typed representations
    legacy = PlantOrganArray.from_xml_file(xml_path)
    legacy.tensor = legacy.tensor.to(device)

    typed = PlantOrganArray.from_xml_file_typed(xml_path)
    typed.tensor = typed.tensor.to(device)

    # Render both
    legacy_rgb, legacy_leaf = render_organ_array(legacy, renderer, device, image_size)
    typed_rgb, typed_leaf = render_organ_array(typed, renderer, device, image_size)

    # Metrics
    mae = float(np.mean(np.abs(legacy_rgb - typed_rgb)))
    ssim = compute_ssim(legacy_rgb, typed_rgb)
    iou = compute_iou(legacy_leaf, typed_leaf)

    print(f"Legacy render shape: {legacy_rgb.shape}, typed render shape: {typed_rgb.shape}")
    print(f"Legacy leaf mask pixels: {legacy_leaf.sum()}, typed leaf mask pixels: {typed_leaf.sum()}")
    print(f"MAE (legacy vs typed RGB): {mae:.6f}")
    print(f"SSIM (legacy vs typed RGB): {ssim:.6f}")
    print(f"IoU (legacy vs typed leaf mask): {iou:.6f}")

    # Load Helios GT image and mask
    gt_rgb, gt_leaf = load_gt_image_and_mask(xml_path, image_size)
    if gt_rgb is None:
        print("Warning: no Helios GT _vis.jpeg found; using typed render as GT proxy")
        gt_rgb = typed_rgb.copy()
        gt_leaf = typed_leaf.copy()

    # 5-panel figure
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    axes[0].imshow(gt_rgb)
    axes[0].set_title("Helios C++ Ray-Traced", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(typed_rgb)
    axes[1].set_title("PyTorch Typed Organ Array", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(gt_leaf, cmap="gray")
    axes[2].set_title("Helios GT Leaf Mask", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(typed_leaf, cmap="gray")
    axes[3].set_title("PyTorch Typed Leaf Mask", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # Overlay: red = legacy only, green = typed only, yellow = both
    overlay = np.zeros((image_size, image_size, 3), dtype=np.float32)
    overlay[..., 0] = legacy_leaf.astype(np.float32)  # red
    overlay[..., 1] = typed_leaf.astype(np.float32)   # green
    both = legacy_leaf & typed_leaf
    overlay[..., 0] = np.where(both, 1.0, overlay[..., 0])
    overlay[..., 1] = np.where(both, 1.0, overlay[..., 1])
    axes[4].imshow(overlay)
    axes[4].set_title("Mask Overlay (R=legacy, G=typed, Y=both)", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    plt.suptitle(
        f"Typed PlantOrganArray Render Verification | MAE={mae:.5f} | SSIM={ssim:.4f} | IoU={iou:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "typed_organ_array_render_verification.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved verification figure to: {out_path}")

    # Assertions
    assert mae < 1e-3, f"Render MAE too large: {mae}"
    assert iou > 0.99, f"Leaf mask IoU too low: {iou}"
    print("\nSUCCESS: typed PlantOrganArray renders identically to legacy layout.")


if __name__ == "__main__":
    main()
