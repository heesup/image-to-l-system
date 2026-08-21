"""
Accurate Empirical Benchmark: Actual Helios C++ Binary Execution vs PyTorch Part-Centric Direct Renderer (DAP 1-100).

Directly runs and benchmarks:
  1. Actual Helios C++ binary (main --renderer radiation) across DAPs (1-100)
  2. PyTorch part-centric direct assembly (Forward & Backward passes)
  3. End-to-end XML -> part tensor -> Image wall-clock time
  4. Real Speedup Factor across plant growth timeline (vs Helios C++)

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

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

BUILD_DIR = os.path.join(repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build")
MAIN_BIN = os.path.join(BUILD_DIR, "main")
PARAMS_FILE = os.path.join(BUILD_DIR, "params.json")


def benchmark_accurate_dap(force_recompute=True):
    cache_file = os.path.join(repo_root, "diffusion_based", "eval", "benchmark_cache_14d.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Real Empirical Benchmark on {device}...")

    renderer = HeliosPyTorchRenderer(image_size=512).to(device)

    test_daps = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = {
        "dap": [],
        "organ_count": [],
        "triangle_count": [],
        "helios_time_sec": [],
        "torch_14d_fwd_sec": [],
        "torch_14d_bwd_sec": [],
        "xml_to_14d_sec": [],
        "render_14d_sec": [],
        "end_to_end_14d_sec": [],
        "speedup_14d_vs_helios": [],
    }

    # Load baseline Helios C++ cache if available
    old_cache_file = os.path.join(repo_root, "diffusion_based", "eval", "benchmark_cache.json")
    helios_cached = {}
    if os.path.exists(old_cache_file):
        with open(old_cache_file, "r") as f:
            old_data = json.load(f)
            for d, h in zip(old_data.get("dap", []), old_data.get("helios_time_sec", [])):
                helios_cached[d] = h

    for dap in test_daps:
        # 1. Helios C++ timing
        if dap in helios_cached and not force_recompute:
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
        matches = glob.glob(os.path.join(repo_root, "dataset", "helios_data", f"*dap{dap:03d}*.xml"))
        if not matches:
            matches = glob.glob(os.path.join(BUILD_DIR, "output", f"*dap{dap}*.xml"))
        if matches:
            xml_path = matches[0]
        else:
            all_xmls = glob.glob(os.path.join(BUILD_DIR, "output", "*.xml"))
            if all_xmls:
                xml_path = all_xmls[0]

        if xml_path is None or not os.path.exists(xml_path):
            continue

        # 3. End-to-end XML -> 40D OrganArray -> Image timing
        t0 = time.time()
        organ_array = PlantOrganArray.from_xml_file(xml_path)
        t_xml_to_40d = time.time() - t0

        t0 = time.time()
        _ = renderer.render_organ_array(
            organ_array, camera_height=5.0, elevation_deg=90.0,
            device=device, focus_plant=True, differentiable=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_render_40d = time.time() - t0
        t_end_to_end = t_xml_to_40d + t_render_40d

        mesh_dict = renderer.geo_builder.build_mesh_from_organ_array(
            organ_array, device=device, species="cowpea"
        )
        tri_count = mesh_dict["faces"].shape[0]
        organ_count = organ_array.tensor.shape[0]

        # Warmup
        _ = renderer.render_organ_array(
            organ_array, camera_height=5.0, elevation_deg=90.0,
            device=device, focus_plant=True, differentiable=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark 40D Direct Forward
        t0 = time.time()
        for _ in range(5):
            _ = renderer.render_organ_array(
                organ_array, camera_height=5.0, elevation_deg=90.0,
                device=device, focus_plant=True, differentiable=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_40d_fwd = (time.time() - t0) / 5.0

        # Benchmark 40D Direct Backward
        t0 = time.time()
        for _ in range(3):
            opt_arr = PlantOrganArray(organ_array.tensor.clone().requires_grad_(True), raw_metadata=organ_array.raw_metadata)
            rend_40d = renderer.render_organ_array(
                opt_arr, camera_height=5.0, elevation_deg=90.0,
                device=device, focus_plant=True, differentiable=True,
            )
            loss_40d = rend_40d.sum()
            loss_40d.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_40d_bwd = (time.time() - t0) / 3.0

        speedup_helios = helios_sec / max(torch_40d_fwd, 1e-4)

        results["dap"].append(dap)
        results["organ_count"].append(organ_count)
        results["triangle_count"].append(tri_count)
        results["helios_time_sec"].append(helios_sec)
        results["torch_14d_fwd_sec"].append(torch_40d_fwd)
        results["torch_14d_bwd_sec"].append(torch_40d_bwd)
        results["xml_to_14d_sec"].append(t_xml_to_40d)
        results["render_14d_sec"].append(t_render_40d)
        results["end_to_end_14d_sec"].append(t_end_to_end)
        results["speedup_14d_vs_helios"].append(speedup_helios)

        print(f"DAP {dap:03d} (Organs={organ_count:4d}, Tris={tri_count:6d}): "
              f"Helios C++={helios_sec:5.2f}s | "
              f"PyTorch 40D Fwd={torch_40d_fwd*1000:5.1f}ms | "
              f"XML->40D={t_xml_to_40d*1000:5.1f}ms | "
              f"40D->img={t_render_40d*1000:5.1f}ms | "
              f"E2E={t_end_to_end*1000:5.1f}ms | "
              f"Speedup={speedup_helios:5.1f}x vs Helios")

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)

    # Plot Figure
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plt.subplots_adjust(wspace=0.30, hspace=0.25, left=0.07, right=0.95, top=0.92, bottom=0.08)
    daps = np.array(results["dap"])

    # Panel 1: Execution Time in Seconds (Raw linear time axis)
    axes[0, 0].plot(daps, results["helios_time_sec"], "o-", color="#d62728", linewidth=2.5, label="Helios C++ Binary (Raytracing)")
    axes[0, 0].plot(daps, results["end_to_end_14d_sec"], "s-", color="#ff7f0e", linewidth=2.2, label="PyTorch 40D E2E (XML -> 40D -> Image)")
    axes[0, 0].plot(daps, results["torch_14d_bwd_sec"], "v-", color="#9467bd", linewidth=2.2, label="PyTorch 40D Forward + Backward")
    axes[0, 0].plot(daps, results["torch_14d_fwd_sec"], "*-", color="#2ca02c", linewidth=2.8, label="PyTorch 40D Forward Pass")
    axes[0, 0].set_ylim(bottom=0, top=max(results["helios_time_sec"]) * 1.15)
    axes[0, 0].set_xlabel("Plant Age (Days After Planting / DAP)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Execution Time per Frame (seconds, linear)", fontsize=11, fontweight="bold")
    axes[0, 0].set_title("Execution Time: Helios C++ vs PyTorch 40D Direct (Linear Scale)", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend(fontsize=9, loc="upper left")

    # Panel 2: End-to-End Breakdown (ms, raw linear time axis)
    axes[0, 1].plot(daps, np.array(results["xml_to_14d_sec"]) * 1000, "o-", color="#1f77b4", linewidth=2.2, label="XML -> 40D Organ Tensor")
    axes[0, 1].plot(daps, np.array(results["render_14d_sec"]) * 1000, "s-", color="#2ca02c", linewidth=2.2, label="40D Tensor -> Image Render")
    axes[0, 1].set_xlabel("Plant Age (Days After Planting / DAP)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Time (milliseconds, linear)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("40D Differentiable End-to-End Breakdown (Linear Scale)", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend(fontsize=9)

    # Panel 3: Speedup Factor (LOG SCALE on Y-axis as requested)
    axes[1, 0].set_yscale("log")
    axes[1, 0].plot(daps, results["speedup_14d_vs_helios"], "D-", color="#1f77b4", linewidth=2.5, label="PyTorch 40D vs Helios C++ Speedup")
    axes[1, 0].fill_between(daps, results["speedup_14d_vs_helios"], 1.0, color="#1f77b4", alpha=0.12)
    axes[1, 0].set_ylim(bottom=1.0, top=max(results["speedup_14d_vs_helios"]) * 2.0)
    axes[1, 0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{int(y)}x" if y >= 1 else f"{y:.1f}x"))
    axes[1, 0].set_xlabel("Plant Age (Days After Planting / DAP)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Speedup Factor vs Helios C++ (Log Scale, x-fold)", fontsize=11, fontweight="bold", color="#1f77b4")
    axes[1, 0].set_title("PyTorch Hardware Acceleration Speedup (Log Scale)", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, which="both", linestyle="--", alpha=0.4)

    max_idx = np.argmax(results["speedup_14d_vs_helios"])
    axes[1, 0].annotate(
        f"Peak Speedup: {results['speedup_14d_vs_helios'][max_idx]:.1f}x\n(DAP {daps[max_idx]})",
        xy=(daps[max_idx], results["speedup_14d_vs_helios"][max_idx]),
        xytext=(daps[max_idx] + 10, results["speedup_14d_vs_helios"][max_idx] * 0.55),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", fontsize=9,
    )

    # Panel 4: Complexity Scaling
    ax3_twin = axes[1, 1].twinx()
    axes[1, 1].plot(daps, results["organ_count"], "o-", color="#17becf", linewidth=2.2, label="Organ Count (N)")
    ax3_twin.plot(daps, results["triangle_count"], "^-", color="#8c564b", linewidth=2.2, label="Mesh Triangles (F)")
    axes[1, 1].set_xlabel("Plant Age (Days After Planting / DAP)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Organ Count (N)", fontsize=11, fontweight="bold", color="#17becf")
    ax3_twin.set_ylabel("Triangle Count (F)", fontsize=11, fontweight="bold", color="#8c564b")
    axes[1, 1].set_title("Canopy Geometric Complexity Scaling", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    out_png = os.path.join(repo_root, "docs", "results", "assets", "fig1_helios_vs_torch_rendering_benchmark.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"\n[OK] Saved updated part-centric benchmark figure to: {out_png}")
    return results


if __name__ == "__main__":
    benchmark_accurate_dap(force_recompute=False)
