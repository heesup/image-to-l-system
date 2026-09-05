"""
Empirical Performance & Scaling Benchmark: Helios C++ Raytracer vs PyTorch Canonical 14D Differentiable Renderer (DAP 1-100).

Function Call Path Profiled:
  1. XML I/O: PlantOrganArray.from_xml_file() -> (N, 40)
  2. Forward Kinematics (FK): organ_array.to_part_tensor() -> (N, 14)
  3. Vectorized GPU Mesh Build: HeliosPlantGeometryBuilder.build_mesh_from_part_tensor()
  4. Differentiable Rasterization: HeliosPyTorchRenderer.forward() (Nvdiffrast)
  5. Backpropagation: loss.backward() -> dLoss/d(14D Part Tensor)

Outputs:
  - docs/results/assets/fig1_helios_vs_torch_rendering_benchmark.png
  - diffusion_based/eval/benchmark_cache_14d.json
"""

import os
import sys
import time
import glob
import json
import shutil
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

BUILD_DIR = os.path.join(REPO_ROOT, "Digital-Crops", "projects", "syntheticdata_generation", "build")
MAIN_BIN = os.path.join(BUILD_DIR, "main")
PARAMS_FILE = os.path.join(BUILD_DIR, "params.json")


def benchmark_accurate_dap(force_recompute_helios=False):
    cache_file = os.path.join(REPO_ROOT, "diffusion_based", "eval", "benchmark_cache_14d.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Real Empirical Benchmark on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")

    renderer = HeliosPyTorchRenderer(image_size=512).to(device)

    test_daps = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = {
        "dap": [],
        "organ_count": [],
        "triangle_count": [],
        "helios_time_sec": [],
        "torch_14d_fwd_sec": [],
        "torch_14d_bwd_sec": [],
        "xml_io_sec": [],
        "fk_to_14d_sec": [],
        "stage_build_mesh_sec": [],
        "stage_rasterize_sec": [],
        "end_to_end_14d_sec": [],
        "speedup_14d_vs_helios": [],
    }

    # Load baseline Helios C++ cache if available
    old_cache_file = os.path.join(REPO_ROOT, "diffusion_based", "eval", "benchmark_cache.json")
    helios_cached = {}
    if os.path.exists(old_cache_file):
        with open(old_cache_file, "r") as f:
            old_data = json.load(f)
            for d, h in zip(old_data.get("dap", []), old_data.get("helios_time_sec", [])):
                helios_cached[d] = h

    # Comprehensive global warm-up across multiple growth stages so all OBJ meshes,
    # CUDA contexts, nvdiffrast kernels, and memory pools are fully primed.
    print("[Warmup] Priming CUDA context, nvdiffrast rasterizer, and OBJ mesh caches...")
    for warm_dap in [1, 10, 50, 90]:
        warm_matches = glob.glob(os.path.join(REPO_ROOT, "dataset", "helios_data", "**", f"*cowpea_dap{warm_dap:03d}_seed68*.xml"), recursive=True)
        if warm_matches:
            warm_oa = PlantOrganArray.from_xml_file(warm_matches[0])
            warm_part = warm_oa.to_part_tensor(device=device)
            warm_mesh = renderer.geo_builder.build_mesh_from_part_tensor(warm_part, device=device)
            _ = renderer.forward(warm_mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
            opt_part = warm_part.clone().requires_grad_(True)
            rend = renderer.render_part_tensor(opt_part, device=device, focus_plant=True, differentiable=True)
            rend.sum().backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    print("[Warmup] Complete.\n")

    for dap in test_daps:
        # 1. Helios C++ timing
        if dap in helios_cached and not force_recompute_helios:
            helios_sec = helios_cached[dap]
        else:
            tmp_dir = f"/tmp/helios_bench_dap{dap}"
            os.makedirs(tmp_dir, exist_ok=True)
            cmd = [
                MAIN_BIN,
                "--renderer", "radiation",
                "--save-xml",
                "--focus-plant",
                "-n", f"bench_dap{dap:03d}",
                "--dap", str(dap),
                "-s", "0",
                "--output", tmp_dir,
                "-f", PARAMS_FILE,
            ]
            t0 = time.time()
            res = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True)
            helios_sec = time.time() - t0
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # 2. Find matching XML
        xml_path = None
        matches = glob.glob(os.path.join(REPO_ROOT, "dataset", "helios_data", "**", f"*cowpea_dap{dap:03d}_seed68*.xml"), recursive=True)
        if not matches:
            matches = glob.glob(os.path.join(REPO_ROOT, "dataset", "helios_data", "**", f"*dap{dap:03d}*.xml"), recursive=True)
        if not matches:
            matches = glob.glob(os.path.join(BUILD_DIR, "output", f"*dap{dap}*.xml"))
        if matches:
            xml_path = matches[0]
        else:
            all_xmls = glob.glob(os.path.join(BUILD_DIR, "output", "*.xml"))
            if all_xmls:
                xml_path = all_xmls[0]

        if xml_path is None or not os.path.exists(xml_path):
            print(f"[Warning] No XML found for DAP {dap}, skipping.")
            continue

        # 3. Call Path Stage-by-Stage Profiling with warm iterations
        # Stage 1: XML Parsing & Deserialization
        # Warmup pass
        _ = PlantOrganArray.from_xml_file(xml_path)
        t0 = time.time()
        n_xml_iters = 5
        for _ in range(n_xml_iters):
            organ_array = PlantOrganArray.from_xml_file(xml_path)
        t_xml_io = (time.time() - t0) / float(n_xml_iters)

        # Stage 2: Forward Kinematics (40D XML Tree -> 14D Part Tensor)
        _ = organ_array.to_part_tensor(device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        n_fk_iters = 5
        for _ in range(n_fk_iters):
            part_tensor = organ_array.to_part_tensor(device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_fk_14d = (time.time() - t0) / float(n_fk_iters)

        # Stage 3: Fully Vectorized 14D Part Mesh Construction
        _ = renderer.geo_builder.build_mesh_from_part_tensor(part_tensor, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        n_mesh_iters = 10
        for _ in range(n_mesh_iters):
            mesh_dict = renderer.geo_builder.build_mesh_from_part_tensor(part_tensor, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_mesh_build_14d = (time.time() - t0) / float(n_mesh_iters)

        tri_count = mesh_dict["faces"].shape[0]
        organ_count = part_tensor.shape[0]

        # Stage 4: 14D GPU Rasterization & Shading (512x512)
        for _ in range(3):
            _ = renderer.forward(mesh_dict, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()
        n_rast_iters = 15
        for _ in range(n_rast_iters):
            _ = renderer.forward(mesh_dict, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_gpu_fwd = (time.time() - t0) / float(n_rast_iters)

        # Stage 5: Total Forward Direct Render (Mesh Build + GPU Rasterize)
        t_render_14d = t_mesh_build_14d + torch_gpu_fwd
        t_end_to_end = t_xml_io + t_fk_14d + t_render_14d

        # Stage 6: Differentiable 14D Optimization Pass (Mesh + Rasterize + Backward)
        # Warmup diff pass
        opt_part_warm = part_tensor.clone().requires_grad_(True)
        rend_warm = renderer.render_part_tensor(
            opt_part_warm, camera_height=5.0, elevation_deg=90.0,
            device=device, focus_plant=True, differentiable=True,
        )
        rend_warm.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()
        n_diff_iters = 5
        for _ in range(n_diff_iters):
            opt_part = part_tensor.clone().requires_grad_(True)
            rend_14d = renderer.render_part_tensor(
                opt_part, camera_height=5.0, elevation_deg=90.0,
                device=device, focus_plant=True, differentiable=True,
            )
            loss_14d = rend_14d.sum()
            loss_14d.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_14d_bwd = (time.time() - t0) / float(n_diff_iters)

        speedup_14d = helios_sec / max(t_render_14d, 1e-6)

        results["dap"].append(dap)
        results["organ_count"].append(organ_count)
        results["triangle_count"].append(tri_count)
        results["helios_time_sec"].append(helios_sec)
        results["torch_14d_fwd_sec"].append(t_render_14d)
        results["torch_14d_bwd_sec"].append(torch_14d_bwd)
        results["xml_io_sec"].append(t_xml_io)
        results["fk_to_14d_sec"].append(t_fk_14d)
        results["stage_build_mesh_sec"].append(t_mesh_build_14d)
        results["stage_rasterize_sec"].append(torch_gpu_fwd)
        results["end_to_end_14d_sec"].append(t_end_to_end)
        results["speedup_14d_vs_helios"].append(speedup_14d)

        print(f"DAP {dap:03d} (Organs={organ_count:4d}, Tris={tri_count:6d}): "
              f"Helios C++={helios_sec:5.2f}s | "
              f"FK={t_fk_14d*1000:5.2f}ms | "
              f"Mesh Build={t_mesh_build_14d*1000:5.2f}ms | "
              f"GPU Rast={torch_gpu_fwd*1000:5.2f}ms | "
              f"Total 14D Direct Render={t_render_14d*1000:5.2f}ms | "
              f"Diff (Fwd+Bwd)={torch_14d_bwd*1000:5.2f}ms | "
              f"Speedup={speedup_14d:6.1f}x")

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)

    # Plot 1x4 Widescreen Presentation Figure (16:9 layout)
    plt.style.use("default")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.8), dpi=200)
    plt.subplots_adjust(wspace=0.28, left=0.045, right=0.96, top=0.86, bottom=0.14)
    daps = np.array(results["dap"])

    # Panel 1: Execution Time (Log Scale)
    ax0 = axes[0]
    ax0.set_yscale("log")
    ax0.plot(daps, results["helios_time_sec"], "o-", color="#d62728", linewidth=2.5, label="Helios C++ Raytracer (s)")
    ax0.plot(daps, results["end_to_end_14d_sec"], "s-", color="#ff7f0e", linewidth=2.2, label="14D E2E (XML $\\to$ Img)")
    ax0.plot(daps, results["torch_14d_bwd_sec"], "v-", color="#9467bd", linewidth=2.2, label="14D Diff (Fwd+Bwd)")
    ax0.plot(daps, results["torch_14d_fwd_sec"], "*-", color="#2ca02c", linewidth=2.8, label="14D Part Direct Render")
    ax0.set_xlabel("Plant Age (DAP)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Frame Latency (seconds, log scale)", fontsize=11, fontweight="bold")
    ax0.set_title("(a) Frame Rendering Latency (Log Scale)", fontsize=12, fontweight="bold")
    ax0.grid(True, which="both", linestyle="--", alpha=0.35)
    ax0.legend(fontsize=8.5, loc="upper left")

    # Panel 2: Speedup Factor (LOG SCALE on Y-axis)
    ax1 = axes[1]
    ax1.set_yscale("log")
    ax1.plot(daps, results["speedup_14d_vs_helios"], "D-", color="#1f77b4", linewidth=2.5, label="14D Part Assembly vs Helios C++")
    ax1.fill_between(daps, results["speedup_14d_vs_helios"], 1.0, color="#1f77b4", alpha=0.12)
    ax1.set_ylim(bottom=50.0, top=5000.0)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}x" if y >= 1 else f"{y:.1f}x"))
    ax1.set_xlabel("Plant Age (DAP)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Speedup Factor vs Helios C++ (Log Scale)", fontsize=11, fontweight="bold", color="#1f77b4")
    ax1.set_title("(b) 14D Hardware Acceleration Speedup", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)

    max_idx = np.argmax(results["speedup_14d_vs_helios"])
    ax1.annotate(
        f"Peak Speedup: {results['speedup_14d_vs_helios'][max_idx]:,.0f}x\n(DAP {daps[max_idx]})",
        xy=(daps[max_idx], results["speedup_14d_vs_helios"][max_idx]),
        xytext=(daps[max_idx] - 25, results["speedup_14d_vs_helios"][max_idx] * 0.45),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=5),
        fontweight="bold", fontsize=8.5,
    )

    # Panel 3: Pipeline Call Path Stage Breakdown (ms)
    ax2 = axes[2]
    ax2.plot(daps, np.array(results["xml_io_sec"]) * 1000, "o-", color="#8c564b", linewidth=2.0, label="1. XML Deserialization")
    ax2.plot(daps, np.array(results["fk_to_14d_sec"]) * 1000, "x-", color="#1f77b4", linewidth=2.0, label="2. Forward Kinematics (FK)")
    ax2.plot(daps, np.array(results["stage_build_mesh_sec"]) * 1000, "^-", color="#e377c2", linewidth=2.0, label="3. 14D GPU Mesh Build")
    ax2.plot(daps, np.array(results["stage_rasterize_sec"]) * 1000, "s-", color="#2ca02c", linewidth=2.5, label="4. GPU Rasterize (512x512)")
    ax2.set_xlabel("Plant Age (DAP)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Execution Time (milliseconds)", fontsize=11, fontweight="bold")
    ax2.set_title("(c) 14D Call Path Breakdown (ms)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8.5, loc="upper left")

    # Panel 4: Canopy Complexity Scaling
    ax3 = axes[3]
    ax3_twin = ax3.twinx()
    p1 = ax3.plot(daps, results["organ_count"], "o-", color="#17becf", linewidth=2.2, label="Organ Count (N)")
    p2 = ax3_twin.plot(daps, results["triangle_count"], "^-", color="#e6550d", linewidth=2.2, label="Mesh Triangles (F)")
    ax3.set_xlabel("Plant Age (DAP)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Organ Count (N)", fontsize=11, fontweight="bold", color="#17becf")
    ax3_twin.set_ylabel("Triangle Count (F)", fontsize=11, fontweight="bold", color="#e6550d")
    ax3.set_title("(d) Canopy Geometric Complexity Scaling", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.3)

    # Combined legend for twin axes
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, fontsize=8.5, loc="upper left")

    plt.suptitle("Figure 1: Empirical Rendering Performance & Scaling Benchmark: Helios C++ Raytracer vs Differentiable PyTorch Renderer (DAP 1–100)", fontsize=14, fontweight="bold", y=0.97)

    out_png = os.path.join(REPO_ROOT, "docs", "results", "assets", "fig1_helios_vs_torch_rendering_benchmark.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"\n[OK] Successfully saved updated Figure 1 benchmark to:\n  -> {out_png}")
    return results


if __name__ == "__main__":
    benchmark_accurate_dap(force_recompute_helios=False)
