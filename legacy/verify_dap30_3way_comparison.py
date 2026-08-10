import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.helios_geometry import build_helios_geometry_from_xml
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer
from diffusion_based.models.differentiable_pipeline import DifferentiableHeliosRenderer
from notebooks.run_differentiable_renderer_stability_test import (
    step1_generate_groundtruth_data,
    compute_ssim_numpy,
    compute_silhouette_iou,
)


def main():
    output_dir = os.path.join(repo_root, "notebooks", "output_dap30_verification")
    
    # 1. Generate DAP 30 GT plant via C++ Helios binary
    print("=======================================================")
    print("STEP 1: Generating Ground Truth Data (DAP=30, Seed=42)")
    print("=======================================================")
    gt_info = step1_generate_groundtruth_data(output_dir=output_dir, dap=30, seed=42)
    
    cpp_img_path = gt_info["gt_image_path"]
    xml_path = gt_info["gt_xml_path"]
    
    # Load C++ Helios JPEG image
    cpp_pil = Image.open(cpp_img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    cpp_np = np.array(cpp_pil, dtype=np.float32) / 255.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
    
    # 2. Python-based XML Renderer (build_helios_geometry_from_xml)
    print("\n[PYTHON XML RENDERER] Building geometry from XML...")
    geom = build_helios_geometry_from_xml(xml_path)
    with torch.no_grad():
        py_xml_rgba = rasterizer.render_numpy_geometry(
            geom.tubes, geom.leaflets, geom.ellipsoids,
            focus_plant=True,
            background="ground",
        )
    py_xml_np = py_xml_rgba[..., :3].clip(0, 1)
    
    # 3. 15D PyTorch Differentiable Renderer (15D nodes parsed from XML)
    print("\n[15D PYTORCH RENDERER] Parsing 15D organ nodes...")
    parser = HeliosXMLParser(xml_path)
    parser.parse()
    gt_organ_nodes = parser.get_all_organ_nodes()
    
    gt_nodes_np = np.stack([n.to_15d() for n in gt_organ_nodes], axis=0)
    gt_nodes_tensor = torch.tensor(gt_nodes_np, dtype=torch.float32, device=device).unsqueeze(0)
    parents_tensor = torch.tensor([n.parent_idx for n in gt_organ_nodes], dtype=torch.long, device=device).unsqueeze(0)
    
    renderer = DifferentiableHeliosRenderer(rasterizer).to(device)
    with torch.no_grad():
        torch_15d_rgba = renderer(
            gt_nodes_tensor,
            parents=parents_tensor,
            focus_plant=True,
            background="ground",
        )
    torch_15d_np = torch_15d_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    # 4. Metrics
    mask_py = py_xml_rgba[..., 3] > 0.1
    mask_15d = torch_15d_rgba[0, 3].cpu().numpy() > 0.1

    py_masked = np.where(mask_py[..., None], py_xml_np, 0.0)
    torch_15d_masked = np.where(mask_15d[..., None], torch_15d_np, 0.0)
    cpp_masked_py = np.where(mask_py[..., None], cpp_np, 0.0)
    cpp_masked_15d = np.where(mask_15d[..., None], cpp_np, 0.0)

    mae_py_xml = float(np.mean(np.abs(py_xml_np - cpp_np)))
    ssim_py_xml = compute_ssim_numpy(py_xml_np, cpp_np)
    plant_ssim_py_xml = compute_ssim_numpy(py_masked, cpp_masked_py)
    iou_py_xml = compute_silhouette_iou(py_xml_np, cpp_np)
    
    mae_15d = float(np.mean(np.abs(torch_15d_np - cpp_np)))
    ssim_15d = compute_ssim_numpy(torch_15d_np, cpp_np)
    plant_ssim_15d = compute_ssim_numpy(torch_15d_masked, cpp_masked_15d)
    iou_15d = compute_silhouette_iou(torch_15d_np, cpp_np)
    
    print("\n=======================================================")
    print("DAP 30 PLANT 3-WAY RENDERER COMPARISON SUMMARY")
    print("=======================================================")
    print("1. C++ Helios Render:       Reference OpenGL JPEG")
    print(f"2. Python XML Render:      Plant-SSIM={plant_ssim_py_xml:.4f}, IoU={iou_py_xml:.4f} (Full-SSIM={ssim_py_xml:.4f})")
    print(f"3. 15D PyTorch Render:     Plant-SSIM={plant_ssim_15d:.4f}, IoU={iou_15d:.4f} (Full-SSIM={ssim_15d:.4f})")
    print("=======================================================")
    
    # 5. Generate 3-Way + Diff 4-Panel Figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(cpp_np)
    axes[0].set_title("1. C++ Helios OpenGL\n(Ground Truth Binary)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(py_xml_np)
    axes[1].set_title(f"2. Python XML Renderer\nPlant-SSIM={plant_ssim_py_xml:.3f} | IoU={iou_py_xml:.3f}", fontsize=11, fontweight="bold", color="darkgreen")
    axes[1].axis("off")
    
    axes[2].imshow(torch_15d_np)
    axes[2].set_title(f"3. 15D PyTorch Renderer\nPlant-SSIM={plant_ssim_15d:.3f} | IoU={iou_15d:.3f}", fontsize=11, fontweight="bold", color="navy")
    axes[2].axis("off")
    
    diff_map = np.abs(py_xml_np - cpp_np)
    im = axes[3].imshow(diff_map)
    axes[3].set_title(f"4. Python XML Diff Map\nMAE={mae_py_xml:.4f}", fontsize=11, fontweight="bold", color="darkred")
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    comp_fig_path = os.path.join(output_dir, "dap30_3way_comparison.png")
    plt.savefig(comp_fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PASS] Saved DAP 30 3-way comparison figure to: {comp_fig_path}")


if __name__ == "__main__":
    main()
