"""
Multi-seed DAP 30 comparison panel.
Reads 5 Helios radiation outputs (different seeds) from a single directory and
renders a 5-row x 5-column figure:
  Row = seed
  Col = Helios RGB | PyTorch RGB | Helios GT Leaf Mask | PyTorch Leaf Mask | Overlay
"""

import os
import re
import time
import subprocess
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.eval.test_helios_coco_mask_comparison import (
    decode_helios_coco_leaf_mask,
)


def time_helios_render(base_dir: str, seed: int, params_file: str = "../params.json") -> float:
    """Time a single Helios C++ radiation render for the given seed."""
    build_dir = "/home/lion397/codes/image-to-l-system/Digital-Crops/projects/syntheticdata_generation/build"
    cmd = [
        "/usr/bin/time", "-f", "%e",
        os.path.join(build_dir, "./main"),
        "--renderer", "radiation",
        "-f", params_file,
        "--dap", "30",
        "--focus-plant",
        "--seed", str(seed),
        "--output", "output_rad_dap30",
        "-n", f"seed{seed}_time",
    ]
    try:
        result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True, timeout=600)
        stderr = result.stderr
        # last line from /usr/bin/time -f %e
        lines = [line.strip() for line in stderr.strip().splitlines() if line.strip()]
        if lines:
            return float(lines[-1])
    except Exception as e:
        print(f"Helios timing failed for seed {seed}: {e}")
    return -1.0


def compute_iou_dice(mask1: np.ndarray, mask2: np.ndarray):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    iou = float(intersection / union) if union > 0 else 1.0
    dice = float(2 * intersection / (mask1.sum() + mask2.sum())) if (mask1.sum() + mask2.sum()) > 0 else 1.0
    return iou, dice


def process_seed(
    base_dir: str,
    seed: int,
    output_dir: str,
    use_generic_leaves: bool = False,
    time_helios: bool = True
):
    prefix = f"seed{seed}_0000"
    xml_path = os.path.join(base_dir, f"{prefix}_plant_0000.xml")
    json_path = os.path.join(base_dir, f"{prefix}_masks.json")
    rad_path = os.path.join(base_dir, f"{prefix}_rad.jpeg")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Time PyTorch path
    t0 = time.time()
    organ_array = PlantOrganArray.from_xml_file(xml_path)
    t_load = time.time() - t0

    helios_gt_mask = decode_helios_coco_leaf_mask(json_path)
    H, W = helios_gt_mask.shape

    builder = HeliosPlantGeometryBuilder(use_generic_leaves=use_generic_leaves, leaf_scale_factor=1.0, tube_radial_subdivisions=6)
    renderer = HeliosPyTorchRenderer(image_size=W)
    renderer.geo_builder = builder

    t0 = time.time()
    mesh_dict = renderer.geo_builder.build_mesh_from_organ_array(organ_array, device=device)
    t_mesh = time.time() - t0

    t0 = time.time()
    pytorch_render_t = renderer.forward(
        mesh_dict,
        azimuth_deg=0.0,
        elevation_deg=90.0,
        camera_height=5.0,
        background="ground",
        focus_plant=True,
    )
    t_render = time.time() - t0

    t0 = time.time()
    organ_type_buffer = renderer.render_organ_type_buffer(
        mesh_dict,
        azimuth_deg=0.0,
        elevation_deg=90.0,
        camera_height=5.0,
        focus_plant=True,
    )
    t_mask = time.time() - t0

    pytorch_rgb = pytorch_render_t.permute(1, 2, 0).cpu().numpy()
    pytorch_leaf_mask = (organ_type_buffer == 2).cpu().numpy()

    pytorch_total = t_load + t_mesh + t_render + t_mask
    pytorch_no_build = t_load + t_render + t_mask  # if mesh reused

    iou, dice = compute_iou_dice(pytorch_leaf_mask, helios_gt_mask)

    rad_rgb = np.array(Image.open(rad_path).convert("RGB"), dtype=np.float32) / 255.0

    # Time Helios C++ render once if requested
    helios_time = -1.0
    if time_helios:
        helios_time = time_helios_render(base_dir, seed)

    return {
        'rad_rgb': rad_rgb,
        'pytorch_rgb': pytorch_rgb,
        'helios_mask': helios_gt_mask,
        'pytorch_mask': pytorch_leaf_mask,
        'iou': iou,
        'dice': dice,
        'seed': seed,
        'helios_time': helios_time,
        'pytorch_total_time': pytorch_total,
        'pytorch_mesh_time': t_mesh,
        'pytorch_render_time': t_render,
        'pytorch_mask_time': t_mask,
    }


def build_panel(results: list, output_path: str):
    n_rows = len(results)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    col_titles = [
        "Helios C++ Ray-Traced",
        "PyTorch Organ Array",
        "Helios GT Leaf Mask",
        "PyTorch Leaf Mask",
        "Mask Overlay",
    ]

    for r_idx, res in enumerate(results):
        axes[r_idx, 0].imshow(res['rad_rgb'])
        axes[r_idx, 1].imshow(res['pytorch_rgb'])
        axes[r_idx, 2].imshow(res['helios_mask'], cmap="gray")
        axes[r_idx, 3].imshow(res['pytorch_mask'], cmap="gray")

        mask_comp = np.zeros((*res['helios_mask'].shape, 3))
        mask_comp[res['helios_mask'], 0] = 1.0
        mask_comp[res['pytorch_mask'], 1] = 1.0
        axes[r_idx, 4].imshow(mask_comp)
        axes[r_idx, 4].set_title(f"IoU={res['iou']:.3f}, Dice={res['dice']:.3f}")

        # Row label with timing info
        h_t = res['helios_time']
        p_t = res['pytorch_total_time']
        h_str = f"{h_t:.1f}s" if h_t >= 0 else "N/A"
        row_label = (
            f"Seed {res['seed']}\n"
            f"Helios: {h_str}\n"
            f"PyTorch: {p_t:.1f}s\n"
            f"(mesh {res['pytorch_mesh_time']:.1f}s, render {res['pytorch_render_time']:.2f}s, mask {res['pytorch_mask_time']:.2f}s)"
        )
        axes[r_idx, 0].set_ylabel(row_label, rotation=0, fontsize=9, labelpad=80, va='center')

        for c_idx in range(n_cols):
            axes[r_idx, c_idx].axis("off")
            if r_idx == 0:
                axes[r_idx, c_idx].set_title(col_titles[c_idx])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved 5x5 multi-seed panel to: {output_path}")


def build_timing_figure(results: list, output_path: str):
    """Create a timing comparison bar chart and speedup table."""
    n = len(results)
    seeds = [r['seed'] for r in results]
    helios_t = [r['helios_time'] for r in results]
    pytorch_total_t = [r['pytorch_total_time'] for r in results]
    pytorch_mesh_t = [r['pytorch_mesh_time'] for r in results]
    pytorch_render_t = [r['pytorch_render_time'] for r in results]
    pytorch_mask_t = [r['pytorch_mask_time'] for r in results]

    speedups = [h / p if p > 0 and h > 0 else 0.0 for h, p in zip(helios_t, pytorch_total_t)]
    mean_speedup = float(np.mean([s for s in speedups if s > 0]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Bar chart: total time comparison
    x = np.arange(n)
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, helios_t, width, label='Helios C++', color='steelblue')
    ax.bar(x + width/2, pytorch_total_t, width, label='PyTorch total', color='coral')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Time (s)')
    ax.set_title('Total Rendering Time Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Stacked PyTorch breakdown
    ax2 = axes[1]
    bottom = np.zeros(n)
    ax2.bar(x, pytorch_mesh_t, width, label='mesh build', bottom=bottom, color='seagreen')
    bottom += np.array(pytorch_mesh_t)
    ax2.bar(x, pytorch_render_t, width, label='render', bottom=bottom, color='coral')
    bottom += np.array(pytorch_render_t)
    ax2.bar(x, pytorch_mask_t, width, label='mask', bottom=bottom, color='mediumpurple')
    ax2.set_xlabel('Seed')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('PyTorch Time Breakdown')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in seeds])
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved timing analysis figure to: {output_path}")

    # Console table
    print("\n" + "=" * 80)
    print("   DAP 30 RENDERING TIME ANALYSIS (seconds)")
    print("=" * 80)
    print(f"{'Seed':>6} {'Helios C++':>12} {'PyTorch Total':>15} {'Mesh':>10} {'Render':>10} {'Mask':>10} {'Speedup':>10}")
    print("-" * 80)
    for r, h, sp in zip(results, helios_t, speedups):
        print(f"{r['seed']:>6} {h:>12.2f} {r['pytorch_total_time']:>15.2f} "
              f"{r['pytorch_mesh_time']:>10.2f} {r['pytorch_render_time']:>10.2f} "
              f"{r['pytorch_mask_time']:>10.2f} {sp:>10.2f}x")
    print("-" * 80)
    print(f"{'Mean':>6} {float(np.mean(helios_t)):>12.2f} {float(np.mean(pytorch_total_t)):>15.2f} "
          f"{float(np.mean(pytorch_mesh_t)):>10.2f} {float(np.mean(pytorch_render_t)):>10.2f} "
          f"{float(np.mean(pytorch_mask_t)):>10.2f} {mean_speedup:>10.2f}x")
    print("=" * 80 + "\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="Digital-Crops/projects/syntheticdata_generation/build/output_rad_dap30")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", default="diffusion_based/eval/output")
    parser.add_argument("--generic-leaves", action="store_true", default=False)
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        print(f"\n=== Processing seed {seed} ===")
        # Time Helios C++ for every seed now that we have a fast enough benchmark mode
        time_helios = True
        res = process_seed(args.base_dir, seed, args.output_dir, use_generic_leaves=args.generic_leaves, time_helios=time_helios)
        results.append(res)
        print(f"Seed {seed}: Helios GT={res['helios_mask'].sum()}, PyTorch={res['pytorch_mask'].sum()}, IoU={res['iou']:.4f}, Dice={res['dice']:.4f}")
        print(f"  Helios render: {res['helios_time']:.2f}s, PyTorch total: {res['pytorch_total_time']:.2f}s (mesh {res['pytorch_mesh_time']:.2f}s, render {res['pytorch_render_time']:.2f}s, mask {res['pytorch_mask_time']:.2f}s)")

    output_path = os.path.join(args.output_dir, "dap30_seed_panel.png")
    build_panel(results, output_path)

    timing_output_path = os.path.join(args.output_dir, "dap30_timing_analysis.png")
    build_timing_figure(results, timing_output_path)

    # Summary metrics
    print("\n" + "=" * 60)
    print("   DAP 30 MULTI-SEED LEAF MASK COMPARISON SUMMARY")
    print("=" * 60)
    for res in results:
        print(f"  Seed {res['seed']}: IoU={res['iou']:.4f}, Dice={res['dice']:.4f}")
    mean_iou = float(np.mean([r['iou'] for r in results]))
    mean_dice = float(np.mean([r['dice'] for r in results]))
    print(f"  Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
