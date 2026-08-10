import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.helios_geometry import build_helios_geometry_from_xml, DifferentiableHeliosXMLRenderer
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer
from notebooks.run_differentiable_renderer_stability_test import compute_ssim_numpy

def main():
    output_dir = os.path.join(repo_root, "notebooks", "output_dap30_verification")
    os.makedirs(output_dir, exist_ok=True)
    xml_path = os.path.join(output_dir, "dap30_gt_seed42_0000_plant_0000.xml")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Running inverse rendering optimization on device: {device}")
    
    # 1. Target Geometry & Image from GT Helios XML
    rasterizer = HeliosGeometryRasterizer(image_size=128).to(device)
    geom_gt = HeliosPlantGeometryTorch.from_xml(xml_path, device=device)
    
    xml_renderer = DifferentiableHeliosXMLRenderer(rasterizer).to(device)
    with torch.no_grad():
        target_rgba = xml_renderer(geom_gt, focus_plant=True, background="black")
    target_rgb = target_rgba[0, :3].permute(1, 2, 0).clip(0, 1)  # (128, 128, 3)
    
    # 2. Differentiable Plant Graph Parameterization via HeliosPlantGeometryTorch
    geom_opt = HeliosPlantGeometryTorch.from_xml(xml_path, device=device)
    geom_opt.leaf_scales.data.fill_(0.6)
    geom_opt.tube_scales.data.fill_(0.7)
    
    optimizer = optim.Adam(geom_opt.parameters(), lr=0.03)
    
    history_images = []
    history_losses = []
    history_ssim = []
    
    num_steps = 60
    print("\n=======================================================")
    print(f"STARTING SINGLE IMAGE INVERSE RENDERING OPTIMIZATION ({num_steps} Steps)")
    print("=======================================================")
    
    t0 = time.time()
    for step in range(num_steps + 1):
        optimizer.zero_grad()
        rendered_rgba = xml_renderer(geom_opt, focus_plant=True, background="black")
        rendered_rgb = rendered_rgba[0, :3].permute(1, 2, 0)
        
        # Loss: L1 + MSE + Alpha silhouette
        loss_rgb = F.l1_loss(rendered_rgb, target_rgb) + F.mse_loss(rendered_rgb, target_rgb)
        loss_alpha = F.mse_loss(rendered_rgba[0, 3], target_rgba[0, 3])
        total_loss = loss_rgb + 2.0 * loss_alpha
        
        if step < num_steps:
            total_loss.backward()
            optimizer.step()
            
        cur_rgb_np = rendered_rgb.detach().cpu().numpy().clip(0, 1)
        tar_rgb_np = target_rgb.cpu().numpy().clip(0, 1)
        ssim_val = compute_ssim_numpy(cur_rgb_np, tar_rgb_np)
        
        history_losses.append(total_loss.item())
        history_ssim.append(ssim_val)
        
        if step in [0, 15, 30, 45, 60]:
            history_images.append((step, cur_rgb_np, total_loss.item(), ssim_val))
            print(f"Step {step:02d}/{num_steps:02d} | Loss: {total_loss.item():.6f} | SSIM: {ssim_val:.4f}")
            
    print(f"Optimization finished in {time.time() - t0:.2f}s!")
    
    # 3. Save Visualization Progression Figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="black")
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")
            
    # Target reference image
    axes[0, 0].imshow(tar_rgb_np)
    axes[0, 0].set_title("Target Helios GT Image\n(Seed=42)", color="white", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    
    # Progression steps
    for idx, (step_num, img, loss_v, ssim_v) in enumerate(history_images):
        if idx < 3:
            ax = axes[0, idx + 1]
        else:
            ax = axes[1, 0]
        ax.imshow(img)
        ax.set_title(f"Step {step_num:02d}\nLoss={loss_v:.4f} | SSIM={ssim_v:.4f}", color="cyan", fontsize=12, fontweight="bold")
        ax.axis("off")
        
    # Loss curve
    axes[1, 1].plot(history_losses, color="crimson", linewidth=2.5)
    axes[1, 1].set_title("Loss Convergence Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Optimization Step", color="white")
    axes[1, 1].set_ylabel("Loss", color="crimson")
    axes[1, 1].tick_params(colors="white")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)
    
    # SSIM curve
    axes[1, 2].plot(history_ssim, color="springgreen", linewidth=2.5)
    axes[1, 2].set_title("SSIM Progression Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Optimization Step", color="white")
    axes[1, 2].set_ylabel("SSIM", color="springgreen")
    axes[1, 2].tick_params(colors="white")
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)
    
    # Final Pixel Diff Map
    final_diff = np.abs(history_images[-1][1] - tar_rgb_np)
    im = axes[1, 3].imshow(final_diff.mean(axis=-1), cmap="inferno", vmin=0.0, vmax=0.2)
    axes[1, 3].set_title(f"Final Diff Map\nMAE={np.mean(final_diff):.5f}", color="gold", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    opt_fig_path = os.path.join(output_dir, "diffusion_inverse_optimization_demo.png")
    plt.savefig(opt_fig_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved optimization demo figure to: {opt_fig_path}")

if __name__ == "__main__":
    main()
