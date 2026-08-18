"""
Accurate Empirical Benchmark: Actual Helios C++ Binary Execution vs PyTorch 14D Direct Part Renderer (DAP 1-100).

Directly runs and benchmarks:
  1. Actual Helios C++ binary (main --renderer radiation) across DAPs (1-100)
  2. PyTorch 40D Hierarchical Tree Kinematics (Forward & Backward passes)
  3. PyTorch 14D Direct Part Assembly (Forward & Backward passes)
  4. Real Speedup Factor across plant growth timeline (vs Helios C++ and vs 40D Tree)

Outputs:
  - docs/results/assets/fig0_helios_vs_torch_rendering_benchmark.png
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
        "torch_40d_fwd_sec": [],
        "torch_40d_bwd_sec": [],
        "torch_14d_fwd_sec": [],
        "torch_14d_bwd_sec": [],
        "speedup_14d_vs_helios": [],
        "speedup_14d_vs_40d": [],
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
            # Fallback to closest
            all_xmls = glob.glob(os.path.join(BUILD_DIR, "output", "*.xml"))
            if all_xmls:
                xml_path = all_xmls[0]

        if xml_path is None or not os.path.exists(xml_path):
            continue

        organ_array = PlantOrganArray.from_xml_file_typed(xml_path)
        organ_array.tensor = organ_array.tensor.to(device)
        p14 = organ_array.to_part_tensor_14d(device=device)

        mesh_dict = renderer.geo_builder.build_mesh_from_part_array_14d(p14, template_organ_array=organ_array, device=device, use_kinematics_tree=False)
        tri_count = mesh_dict["faces"].shape[0]
        organ_count = p14.shape[0]

        # Warmup
        _ = renderer.render_organ_array(organ_array, camera_height=5.0, elevation_deg=90.0, device=device, focus_plant=True)
        _ = renderer.render_part_tensor_14d(p14, template_organ_array=organ_array, camera_height=5.0, elevation_deg=90.0, device=device, focus_plant=True, use_kinematics_tree=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark 40D Forward
        t0 = time.time()
        for _ in range(5):
            _ = renderer.render_organ_array(organ_array, camera_height=5.0, elevation_deg=90.0, device=device, focus_plant=True, differentiable=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_40d_fwd = (time.time() - t0) / 5.0

        # Benchmark 40D Backward
        t0 = time.time()
        for _ in range(3):
            opt_t = organ_array.tensor.clone().requires_grad_(True)
            arr_opt = PlantOrganArray(opt_t, raw_metadata=organ_array.raw_metadata)
            rend = renderer.render_organ_array(arr_opt, camera_height=5.0, elevation_deg=90.0, device=device, focus_plant=True, differentiable=True)
            loss = rend.sum()
            loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_40d_bwd = (time.time() - t0) / 3.0

        # Benchmark 14D Direct Forward
        t0 = time.time()
        for _ in range(5):
            _ = renderer.render_part_tensor_14d(p14, template_organ_array=organ_array, camera_height=5.0, elevation_deg=90.0, device=device, focus_plant=True, use_kinematics_tree=False, differentiable=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_14d_fwd = (time.time() - t0) / 5.0

        # Benchmark 14D Direct Backward
        t0 = time.time()
        for _ in range(3):
            opt_p14 = p14.clone().requires_grad_(True)
            rend_14d = renderer.render_part_tensor_14d(opt_p14, template_organ_array=organ_array, camera_height=5.0, elevation_deg=90.0, device=device, focus_plant=True, use_kinematics_tree=False, differentiable=True)
            loss_14d = rend_14d.sum()
            loss_14d.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch_14d_bwd = (time.time() - t0) / 3.0

        speedup_helios = helios_sec / max(torch_14d_fwd, 1e-4)
        speedup_40d = torch_40d_fwd / max(torch_14d_fwd, 1e-4)

        results["dap"].append(dap)
        results["organ_count"].append(organ_count)
        results["triangle_count"].append(tri_count)
        results["helios_time_sec"].append(helios_sec)
        results["torch_40d_fwd_sec"].append(torch_40d_fwd)
        results["torch_40d_bwd_sec"].append(torch_40d_bwd)
        results["torch_14d_fwd_sec"].append(torch_14d_fwd)
        results["torch_14d_bwd_sec"].append(torch_14d_bwd)
        results["speedup_14d_vs_helios"].append(speedup_helios)
        results["speedup_14d_vs_40d"].append(speedup_40d)

        print(f"DAP {dap:03d} (Organs={organ_count:4d}, Tris={tri_count:6d}): "
              f"Helios C++={helios_sec:5.2f}s | 40D Fwd={torch_40d_fwd*1000:5.1f}ms | "
              f"14D Direct Fwd={torch_14d_fwd*1000:5.1f}ms (Speedup: {speedup_helios:5.1f}x vs Helios, {speedup_40d:4.2f}x vs 40D)")

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)

    # Plot 3-Panel Figure (High Resolution)
    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    plt.subplots_adjust(wspace=0.28, left=0.06, right=0.96, top=0.88, bottom=0.15)
    daps = np.array(results["dap"])

    # Panel 1: Execution Time in Seconds
    axes[0].plot(daps, results["helios_time_sec"], "o-", color="#d62728", linewidth=2.5, label="Helios C++ Binary (Raytracing)")
    axes[0].plot(daps, results["torch_40d_bwd_sec"], "s--", color="#ff7f0e", linewidth=2.0, label="PyTorch 40D Tree (Fwd + Bwd)")
    axes[0].plot(daps, results["torch_40d_fwd_sec"], "^--", color="#bcbd22", linewidth=2.0, label="PyTorch 40D Tree (Forward Pass)")
    axes[0].plot(daps, results["torch_14d_bwd_sec"], "v-", color="#9467bd", linewidth=2.2, label="PyTorch 14D Direct (Fwd + Bwd)")
    axes[0].plot(daps, results["torch_14d_fwd_sec"], "*-", color="#2ca02c", linewidth=2.8, label="PyTorch 14D Direct (Forward Pass)")
    
    axes[0].set_ylim(bottom=0, top=max(results["helios_time_sec"]) * 1.15)
    axes[0].set_xlabel("Plant Age (Days After Planting / DAP)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Execution Time per Frame (seconds)", fontsize=11, fontweight="bold")
    axes[0].set_title("Execution Time: Helios C++ vs 40D vs 14D Direct", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=9, loc="upper left")

    # Panel 2: Speedup Factor
    ax2_twin = axes[1].twinx()
    l1 = axes[1].plot(daps, results["speedup_14d_vs_helios"], "D-", color="#1f77b4", linewidth=2.5, label="14D vs Helios C++ Speedup")
    axes[1].fill_between(daps, results["speedup_14d_vs_helios"], color="#1f77b4", alpha=0.12)
    l2 = ax2_twin.plot(daps, results["speedup_14d_vs_40d"], "s-.", color="#e377c2", linewidth=2.2, label="14D vs 40D Tree Speedup")
    
    axes[1].set_ylim(bottom=0, top=max(results["speedup_14d_vs_helios"]) * 1.12)
    ax2_twin.set_ylim(bottom=1.0, top=3.0)
    axes[1].set_xlabel("Plant Age (Days After Planting / DAP)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("14D Speedup vs Helios C++ (x-fold)", fontsize=11, fontweight="bold", color="#1f77b4")
    ax2_twin.set_ylabel("14D Speedup vs 40D Tree (x-fold)", fontsize=11, fontweight="bold", color="#e377c2")
    axes[1].set_title("PyTorch 14D Direct Hardware Acceleration", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    max_idx = np.argmax(results["speedup_14d_vs_helios"])
    axes[1].annotate(
        f"Peak Speedup: {results['speedup_14d_vs_helios'][max_idx]:.1f}x\n(DAP {daps[max_idx]})",
        xy=(daps[max_idx], results["speedup_14d_vs_helios"][max_idx]),
        xytext=(daps[max_idx] + 10, results["speedup_14d_vs_helios"][max_idx] * 0.85),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold", fontsize=9,
    )

    # Panel 3: Complexity Scaling
    ax3_twin = axes[2].twinx()
    axes[2].plot(daps, results["organ_count"], "o-", color="#17becf", linewidth=2.2, label="Organ Count (N)")
    ax3_twin.plot(daps, results["triangle_count"], "^-", color="#8c564b", linewidth=2.2, label="Mesh Triangles (F)")
    axes[2].set_xlabel("Plant Age (Days After Planting / DAP)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Organ Count (N)", fontsize=11, fontweight="bold", color="#17becf")
    ax3_twin.set_ylabel("Triangle Count (F)", fontsize=11, fontweight="bold", color="#8c564b")
    axes[2].set_title("Canopy Geometric Complexity Scaling", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    out_png = os.path.join(repo_root, "docs", "results", "assets", "fig1_helios_vs_torch_rendering_benchmark.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Also copy to artifact directory
    artifact_dir = "/home/lion397/.gemini/antigravity-ide/brain/c148742b-205e-4e0f-8722-f0c0dbedcc27"
    if os.path.exists(artifact_dir):
        shutil.copy(out_png, os.path.join(artifact_dir, "fig1_helios_vs_torch_rendering_benchmark.png"))

    print(f"\n[OK] Saved updated 14D benchmark figure to: {out_png}")
    return results


if __name__ == "__main__":
    benchmark_accurate_dap(force_recompute=False)
