import os
import sys
import shutil
import json
import time
import socket
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser
from diffusion_based.models.legacy.helios_geometry_track_a import build_helios_geometry_from_xml, HeliosPlantGeometryTorch, DifferentiableHeliosXMLRenderer
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer
from notebooks.run_differentiable_renderer_stability_test import (
    setup_display_env,
    compute_ssim_numpy,
    compute_silhouette_iou,
)


def generate_gt_sample_for_seed(output_dir: str, dap: int = 100, seed: int = 42) -> dict:
    """Generate ground truth C++ Helios image and XML for a specific seed."""
    main_binary = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "main"
    )
    base_params_file = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "params.json"
    )
    
    assert os.path.exists(main_binary), f"Main C++ binary not found at {main_binary}"
    assert os.path.exists(base_params_file), f"Params JSON not found at {base_params_file}"
    build_dir = os.path.dirname(main_binary)
    env = setup_display_env()
    
    name = f"dap{dap}_gt_seed{seed}"
    with open(base_params_file, "r") as f:
        params = json.load(f)
        
    params.setdefault("camera", {}).setdefault("positioning", {})["azimuth_angle"] = 0.0
    params["camera"]["positioning"]["camera_height"] = 5.0
    params["camera"]["positioning"]["focusing_plants"] = True
    params.setdefault("metadata", {})["dap"] = int(dap)
    params["metadata"].pop("DAP", None)
    
    tmp_params_path = os.path.join(output_dir, f"{name}_params.json")
    with open(tmp_params_path, "w") as f:
        json.dump(params, f, indent=2)
        
    cmd = [
        main_binary,
        "--renderer", "vis",
        "--save-xml",
        "--focus-plant",
        "-n", name,
        "--dap", str(dap),
        "-s", str(seed),
        "--output", output_dir,
        "-f", tmp_params_path,
    ]
    
    t0 = time.time()
    res = subprocess.run(cmd, cwd=build_dir, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    img_path = os.path.join(output_dir, f"{name}_0000_vis.jpeg")
    if not os.path.exists(img_path):
        alt = os.path.join(output_dir, f"{name}_0000.jpeg")
        if os.path.exists(alt):
            img_path = alt
            
    xml_path = os.path.join(output_dir, f"{name}_0000_plant_0000.xml")
    assert os.path.exists(img_path), f"Seed {seed} GT image failed!\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert os.path.exists(xml_path), f"Seed {seed} GT XML failed!"
    
    return {
        "seed": seed,
        "gt_image_path": img_path,
        "gt_xml_path": xml_path,
        "elapsed": elapsed,
    }


def verify_seed(seed: int, output_dir: str, device: torch.device, rasterizer: HeliosGeometryRasterizer):
    """Run 3-way renderer comparison for a specific seed."""
    print(f"\n--- Testing Seed {seed} ---")
    gt_info = generate_gt_sample_for_seed(output_dir=output_dir, dap=100, seed=seed)
    
    cpp_img_path = gt_info["gt_image_path"]
    xml_path = gt_info["gt_xml_path"]
    
    # Load C++ Helios OpenGL reference image
    cpp_pil = Image.open(cpp_img_path).convert("RGB").resize((256, 256), Image.LANCZOS)
    cpp_np = np.array(cpp_pil, dtype=np.float32) / 255.0
    
    # 2. HeliosPlantGeometryTorch & Direct PyTorch Differentiable Rasterizer (on BLACK background)
    geom_torch = HeliosPlantGeometryTorch.from_xml(xml_path, device=device)
    with torch.no_grad():
        torch_15d_rgba = rasterizer(
            geom_torch,
            focus_plant=True,
            background="black",
        )
    torch_15d_np = torch_15d_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    # 3. Metrics & Direct C++ Helios vs PyTorch Diff Comparison on Black Background
    mask_cpp = np.linalg.norm(cpp_np - np.array([0.61, 0.58, 0.55]), axis=-1) > 0.1
    mask_15d = torch_15d_rgba[0, 3].cpu().numpy() > 0.05
    
    diff_map = np.abs(cpp_np - torch_15d_np)
    mae_diff = float(np.mean(diff_map))
    ssim_diff = compute_ssim_numpy(cpp_np, torch_15d_np)
    
    intersection = np.logical_and(mask_cpp, mask_15d).sum()
    union = np.logical_or(mask_cpp, mask_15d).sum()
    iou_diff = float(intersection / max(union, 1))

    print(f"Seed {seed:04d} Results (Black Background):")
    print(f"  C++ Helios vs PyTorch Diff: SSIM={ssim_diff:.4f}, IoU={iou_diff:.4f}, MAE={mae_diff:.4f}")

    # Generate 3-panel comparison figure: 1. GT Reference | 2. PyTorch Diff Renderer | 3. Pixel Diff Map
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="black")
    for ax in axes:
        ax.set_facecolor("black")
        
    axes[0].imshow(cpp_np)
    axes[0].set_title(f"1. C++ Helios Reference GT\n(Seed={seed})", fontsize=11, fontweight="bold", color="white")
    axes[0].axis("off")
    
    axes[1].imshow(torch_15d_np)
    axes[1].set_title(f"2. PyTorch Diff Renderer\nSSIM={ssim_diff:.3f} | IoU={iou_diff:.3f}", fontsize=11, fontweight="bold", color="cyan")
    axes[1].axis("off")
    
    im = axes[2].imshow(diff_map.mean(axis=-1), cmap="inferno", vmin=0.0, vmax=0.3)
    axes[2].set_title(f"3. C++ vs PyTorch Diff Map\nMAE={mae_diff:.4f}", fontsize=11, fontweight="bold", color="crimson")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    comp_fig_path = os.path.join(output_dir, f"dap100_3way_comparison_seed{seed}.png")
    plt.savefig(comp_fig_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    
    # Save main file if seed==42 to update dap100_3way_comparison.png
    if seed == 42:
        main_fig_path = os.path.join(output_dir, "dap100_3way_comparison.png")
        shutil.copyfile(comp_fig_path, main_fig_path)
        
    return {
        "seed": seed,
        "ssim_diff": ssim_diff,
        "iou_diff": iou_diff,
        "mae_diff": mae_diff,
    }


def main():
    output_dir = os.path.join(repo_root, "notebooks", "output_dap100_verification")
    os.makedirs(output_dir, exist_ok=True)
    
    seeds = [42, 43, 44, 45, 100]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
    
    results = []
    print("=======================================================")
    print(f"STARTING MULTI-SEED DAP 100 RENDERER VERIFICATION ({len(seeds)} seeds)")
    print(f"Seeds: {seeds}")
    print("=======================================================")
    
    for s in seeds:
        res = verify_seed(s, output_dir=output_dir, device=device, rasterizer=rasterizer)
        results.append(res)
        
    print("\n=========================================================================")
    print("MULTI-SEED DAP 100 RENDERER VERIFICATION (XML vs PyTorch Diff on Black BG)")
    print("=========================================================================")
    print(f"{'Seed':<6} | {'XML vs Diff SSIM':<18} | {'XML vs Diff Alpha IoU':<20} | {'XML vs Diff MAE':<15}")
    print("-" * 68)
    for r in results:
        print(f"{r['seed']:<6} | {r['ssim_diff']:<18.4f} | {r['iou_diff']:<20.4f} | {r['mae_diff']:<15.4f}")
        
    ssims = [r['ssim_diff'] for r in results]
    ious = [r['iou_diff'] for r in results]
    maes = [r['mae_diff'] for r in results]
    
    print("-" * 68)
    print(f"MEAN   | {np.mean(ssims):<18.4f} | {np.mean(ious):<20.4f} | {np.mean(maes):<15.4f}")
    print(f"STD    | {np.std(ssims):<18.4f} | {np.std(ious):<20.4f} | {np.std(maes):<15.4f}")
    print("=========================================================================")


if __name__ == "__main__":
    main()
