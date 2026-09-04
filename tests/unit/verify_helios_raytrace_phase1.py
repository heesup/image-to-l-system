"""
Verify Helios C++ raytracing on exported XML files from Phase 1.
Renders ground_truth.xml, method_1.xml, method_2.xml, and method_3.xml.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from diffusion_based.eval.eval_13d_xml_organ_masks import render_helios_full

def main():
    xml_dir = "scratch/xml_outputs"
    methods = [
        ("ground_truth", "Ground Truth"),
        ("method_1", "Method 1 (ICP)"),
        ("method_2", "Method 2 (Diff Renderer)"),
        ("method_3", "Method 3 (Flow Matching)")
    ]

    results = {}
    for prefix, label in methods:
        xml_path = os.path.join(xml_dir, f"{prefix}.xml")
        print(f"\n[Helios-Raytrace] Rendering {label} ({xml_path})...")
        res = render_helios_full(xml_path, f"phase1_raytrace_{prefix}")
        results[prefix] = res
        rgb = res['rgb']
        depth = res['depth']
        mask = res['mask_map']
        valid_pixels = np.sum(mask >= 0) if mask is not None else 0
        print(f"  -> RGB shape: {rgb.shape}, max: {rgb.max():.3f}, valid plant pixels: {valid_pixels}")

    # Compute raytraced IoU against GT raytrace
    gt_mask = results["ground_truth"]["mask_map"] >= 0
    print("\n" + "="*70)
    print("HELIOS C++ RAYTRACE VERIFICATION RESULTS")
    print("="*70)
    for prefix, label in methods:
        m = results[prefix]["mask_map"] >= 0
        inter = np.logical_and(m, gt_mask).sum()
        union = np.logical_or(m, gt_mask).sum()
        iou = (inter / max(union, 1)) * 100.0
        print(f"{label:35s} | Helios Raytraced Mask IoU: {iou:6.2f}%")
    print("="*70)

    # Save a diagnostic visual grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, (prefix, label) in enumerate(methods):
        axes[0, col].imshow(results[prefix]["rgb"])
        axes[0, col].set_title(f"{label}\nHelios C++ RGB", fontsize=11, fontweight='bold')
        axes[0, col].axis('off')

        axes[1, col].imshow(results[prefix]["depth"], cmap='viridis')
        axes[1, col].set_title(f"{label}\nHelios C++ Depth", fontsize=11, fontweight='bold')
        axes[1, col].axis('off')

    plt.tight_layout()
    out_png = "docs/results/assets/helios_cpp_raytrace_verification.png"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Helios-Raytrace] Visual verification grid saved to {out_png}")

if __name__ == "__main__":
    main()
