import os
import sys
import time
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
from diffusion_based.models.helios_geometry import HeliosPlantGeometryTorch, DifferentiableHeliosXMLRenderer
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer
from notebooks.run_differentiable_renderer_stability_test import setup_display_env, compute_ssim_numpy, compute_silhouette_iou

def generate_gt_sample(output_dir: str, dap: int, seed: int = 42) -> dict:
    """Generate C++ Helios GT image and XML for given DAP."""
    main_binary = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "main"
    )
    base_params_file = os.path.join(
        repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "params.json"
    )
    
    assert os.path.exists(main_binary), f"Main C++ binary not found at {main_binary}"
    assert os.path.exists(base_params_file), f"Params JSON not found at {base_params_file}"
    
    with open(base_params_file, "r") as f:
        params = json.load(f)
        
    params.setdefault("camera", {}).setdefault("positioning", {})["azimuth_angle"] = 0.0
    params["camera"]["positioning"]["camera_height"] = 1.0
    params["camera"]["positioning"]["focusing_plants"] = True
    params.setdefault("environment", {}).setdefault("soil", {})["use_obj_ground"] = False
    params.setdefault("metadata", {})["dap"] = int(dap)
    params["metadata"].pop("DAP", None)
    params["seed"] = int(seed)
    
    prefix = f"dap{dap}_gt"
    params_file = os.path.join(output_dir, f"{prefix}_params.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
        
    env = setup_display_env()
    cmd = [
        main_binary,
        "--renderer", "vis",
        "--save-xml",
        "--focus-plant",
        "--dap", str(dap),
        "--output-dir", output_dir,
        "--params-file", params_file,
        "--name", prefix,
    ]
    
    print(f"\n[C++ HELIOS] Generating DAP {dap} (Seed={seed})...")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=os.path.dirname(main_binary), env=env, capture_output=True, text=True)
    cpp_time_ms = (time.time() - t0) * 1000.0
    
    if res.returncode != 0:
        print(f"Error running C++ binary for DAP {dap}: {res.stderr}")
        raise RuntimeError(res.stderr)
        
    output_build_dir = os.path.join(os.path.dirname(main_binary), "output")
    xml_path = os.path.join(output_build_dir, f"{prefix}_0000_plant_0000.xml")
    if not os.path.exists(xml_path):
        xml_path = os.path.join(output_dir, f"{prefix}_0000_plant_0000.xml")

    img_path = os.path.join(output_build_dir, f"{prefix}_0000_vis.jpeg")
    if not os.path.exists(img_path):
        img_path = os.path.join(output_dir, f"{prefix}_0000_vis.jpeg")
        
    return {
        "dap": dap,
        "xml_path": xml_path,
        "img_path": img_path,
        "cpp_time_ms": cpp_time_ms,
    }

def main():
    output_dir = os.path.join(repo_root, "notebooks", "output_dap_benchmark")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Multi-DAP (10, 30, 50) Renderer Benchmark on device: {device}")
    
    rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
    
    daps = [10, 30, 50]
    seed = 42
    results = []
    
    for dap in daps:
        gt_info = generate_gt_sample(output_dir, dap=dap, seed=seed)
        
        # Load C++ Helios image
        cpp_pil = Image.open(gt_info["img_path"]).convert("RGB").resize((256, 256), Image.LANCZOS)
        cpp_np = np.array(cpp_pil, dtype=np.float32) / 255.0
        
        # Direct PyTorch Rasterizer call via rasterizer(geom_torch)
        t_parse_start = time.time()
        geom_torch = HeliosPlantGeometryTorch.from_xml(gt_info["xml_path"], device=device)
        t_parse_ms = (time.time() - t_parse_start) * 1000.0
        
        # Warmup
        with torch.no_grad():
            _ = rasterizer(geom_torch, focus_plant=True, background="black")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        t_render_start = time.time()
        with torch.no_grad():
            torch_rgba = rasterizer(geom_torch, focus_plant=True, background="black")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_time_ms = (time.time() - t_render_start) * 1000.0
        
        torch_rgb = torch_rgba[0, :3].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        torch_alpha = torch_rgba[0, 3].cpu().numpy()
        
        # Compute Metrics
        ssim_val = compute_ssim_numpy(cpp_np, torch_rgb)
        
        mask_cpp = np.linalg.norm(cpp_np - np.array([0.61, 0.58, 0.55]), axis=-1) > 0.1
        mask_torch = torch_alpha > 0.05
        inter = np.logical_and(mask_cpp, mask_torch).sum()
        union = np.logical_or(mask_cpp, mask_torch).sum()
        iou_val = float(inter / max(union, 1))
        
        # Plant mask MAE on black background
        diff_map = np.abs(cpp_np - torch_rgb)
        mae_val = float(np.mean(diff_map))
        
        results.append({
            "dap": dap,
            "cpp_np": cpp_np,
            "torch_rgb": torch_rgb,
            "diff_map": diff_map,
            "cpp_time_ms": gt_info["cpp_time_ms"],
            "torch_parse_ms": t_parse_ms,
            "torch_time_ms": torch_time_ms,
            "ssim": ssim_val,
            "iou": iou_val,
            "mae": mae_val,
            "n_tubes": geom_torch.tube_verts_base.shape[0],
            "n_leaflets": geom_torch.leaf_verts_base.shape[0],
        })
        
        print(f"DAP {dap:02d} | C++ Helios: {gt_info['cpp_time_ms']:.1f}ms | PyTorch Render: {torch_time_ms:.2f}ms (Parse: {t_parse_ms:.1f}ms)")
        print(f"       | SSIM: {ssim_val:.4f} | Alpha IoU: {iou_val:.4f} | MAE: {mae_val:.4f}")

    # Create 3-Row x 4-Column Comparison Grid Figure
    fig, axes = plt.subplots(3, 4, figsize=(22, 16), facecolor="black")
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")
            
    for row_idx, res in enumerate(results):
        dap = res["dap"]
        
        # Col 0: C++ Helios Reference GT
        axes[row_idx, 0].imshow(res["cpp_np"])
        axes[row_idx, 0].set_title(f"DAP {dap} - C++ Helios GT\n(Time: {res['cpp_time_ms']:.0f} ms)", color="white", fontsize=13, fontweight="bold")
        axes[row_idx, 0].axis("off")
        
        # Col 1: PyTorch Differentiable Renderer (HeliosPlantGeometryTorch)
        axes[row_idx, 1].imshow(res["torch_rgb"])
        axes[row_idx, 1].set_title(f"DAP {dap} - PyTorch Diff Renderer\n(Render: {res['torch_time_ms']:.2f} ms)", color="cyan", fontsize=13, fontweight="bold")
        axes[row_idx, 1].axis("off")
        
        # Col 2: Pixel Difference Map
        im = axes[row_idx, 2].imshow(res["diff_map"].mean(axis=-1), cmap="inferno", vmin=0.0, vmax=0.3)
        axes[row_idx, 2].set_title(f"DAP {dap} - Difference Map\n(MAE = {res['mae']:.5f})", color="gold", fontsize=13, fontweight="bold")
        axes[row_idx, 2].axis("off")
        plt.colorbar(im, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
        
        # Col 3: Benchmark & Structure Summary Box
        summary_text = (
            f"DAP {dap} Benchmark Summary\n"
            f"----------------------------------------\n"
            f"• Plant Scale   : {res['n_tubes']} Stems | {res['n_leaflets']} Leaflets\n"
            f"• C++ Helios    : {res['cpp_time_ms']:.1f} ms\n"
            f"• PyTorch Parse : {res['torch_parse_ms']:.1f} ms\n"
            f"• PyTorch Render: {res['torch_time_ms']:.2f} ms\n"
            f"• Speedup Factor: {res['cpp_time_ms'] / max(res['torch_time_ms'], 0.01):.1f}x Faster!\n"
            f"----------------------------------------\n"
            f"• SSIM Similarity : {res['ssim']:.4f}\n"
            f"• Alpha IoU Match : {res['iou']:.4f}\n"
            f"• Pixel MAE Loss  : {res['mae']:.5f}"
        )
        axes[row_idx, 3].text(
            0.05, 0.5, summary_text, color="springgreen", fontsize=11, family="monospace",
            verticalalignment="center", bbox=dict(boxstyle="round,pad=0.8", facecolor="#111111", edgecolor="springgreen", alpha=0.9)
        )
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    bench_fig_path = os.path.join(output_dir, "dap_10_30_50_renderer_benchmark.png")
    plt.savefig(bench_fig_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"\nSaved Multi-DAP Renderer Benchmark Figure to: {bench_fig_path}")

if __name__ == "__main__":
    main()
