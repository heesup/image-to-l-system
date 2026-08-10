import os
import sys
import json
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer
from diffusion_based.models.differentiable_pipeline import DifferentiableHeliosRenderer
from notebooks.run_differentiable_renderer_stability_test import (
    step1_generate_groundtruth_data,
    compute_ssim_numpy,
    compute_silhouette_iou,
)


def main():
    output_dir = os.path.join(repo_root, "notebooks", "output_dap1_verification")
    
    # 1. Generate Day 1 GT plant via C++ Helios binary
    gt_info = step1_generate_groundtruth_data(output_dir=output_dir, dap=1, seed=42)
    
    cpp_img_path = gt_info["gt_image_path"]
    xml_path = gt_info["gt_xml_path"]
    
    # Load C++ Helios JPEG image
    cpp_pil = Image.open(cpp_img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    cpp_np = np.array(cpp_pil, dtype=np.float32) / 255.0
    
    # 2. Render 15D nodes parsed from Day 1 GT XML via PyTorch Differentiable Renderer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = HeliosXMLParser(xml_path)
    parser.parse()
    gt_organ_nodes = parser.get_all_organ_nodes()
    
    gt_nodes_np = np.stack([n.to_15d() for n in gt_organ_nodes], axis=0)
    gt_nodes_tensor = torch.tensor(gt_nodes_np, dtype=torch.float32, device=device).unsqueeze(0)
    parents_tensor = torch.tensor([n.parent_idx for n in gt_organ_nodes], dtype=torch.long, device=device).unsqueeze(0)
    
    rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
    renderer = DifferentiableHeliosRenderer(rasterizer).to(device)
    
    with torch.no_grad():
        torch_rgba = renderer(
            gt_nodes_tensor,
            parents=parents_tensor,
            focus_plant=True,
            background="ground",
        )
    torch_rgb_np = torch_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    # 3. Quantitative Comparison
    mae = float(np.mean(np.abs(torch_rgb_np - cpp_np)))
    mse = float(np.mean((torch_rgb_np - cpp_np) ** 2))
    ssim = compute_ssim_numpy(torch_rgb_np, cpp_np)
    iou = compute_silhouette_iou(torch_rgb_np, cpp_np)
    
    print("\n=======================================================")
    print("DAY 1 PLANT: HELIOS C++ RENDER vs 15D TORCH RENDER COMPARISON")
    print("=======================================================")
    print(f"C++ Helios Image:   {cpp_img_path}")
    print(f"15D Torch Image:     Rendered from {len(gt_organ_nodes)} organ nodes")
    print(f"MAE:            {mae:.4f}")
    print(f"MSE:            {mse:.4f}")
    print(f"SSIM:           {ssim:.4f}")
    print(f"Silhouette IoU: {iou:.4f}")
    print("=======================================================")
    
    # 4. Generate Side-by-Side & Difference Map Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cpp_np)
    axes[0].set_title("1. C++ Helios Rendering\n(OpenGL / JPEG output)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(torch_rgb_np)
    axes[1].set_title("2. 15D PyTorch Rendering\n(Differentiable Helios Renderer)", fontsize=11, fontweight="bold", color="navy")
    axes[1].axis("off")
    
    diff_map = np.abs(torch_rgb_np - cpp_np)
    im = axes[2].imshow(diff_map)
    axes[2].set_title(f"3. Absolute Pixel Difference Map\nMAE={mae:.4f} | SSIM={ssim:.4f} | IoU={iou:.4f}", fontsize=11, fontweight="bold", color="darkred")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    comp_fig_path = os.path.join(output_dir, "day1_helios_vs_torch_comparison.png")
    plt.savefig(comp_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PASS] Saved Day 1 comparison figure to: {comp_fig_path}")


if __name__ == "__main__":
    main()
