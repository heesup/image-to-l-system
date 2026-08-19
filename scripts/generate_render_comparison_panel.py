#!/usr/bin/env python3
"""
Multi-Species Rendering Comparison Benchmark & Visualizer.
Compares Helios C++ OptiX Raytracer vs. PyTorch Differentiable Renderer across multiple plant species.
Generates 4-column side-by-side visual panels with execution time benchmarks:
  Col 1: Helios Renderer (RGB)
  Col 2: Helios Renderer Mask
  Col 3: Python PyTorch Renderer (RGB)
  Col 4: Python PyTorch Renderer Mask
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder, SPECIES_CONFIG

ORGAN_COLORS = {
    "background": (20, 20, 24),
    "leaf": (45, 180, 60),
    "stem": (180, 120, 50),
    "petiole": (130, 175, 45),
    "peduncle": (105, 145, 55),
    "flower": (245, 220, 40),
    "fruit": (230, 85, 40),
}

HELIOS_CAT_MAP = {
    0: "stem",
    1: "petiole",
    2: "leaf",
    3: "peduncle",
    4: "flower",
    5: "fruit",
}


def decode_helios_mask(json_path: str, H: int = 720, W: int = 720) -> np.ndarray:
    """Decodes Helios COCO masks.json into a colored RGB semantic mask image."""
    mask_img = np.full((H, W, 3), ORGAN_COLORS["background"], dtype=np.uint8)
    if not os.path.exists(json_path):
        return mask_img

    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # Decode polygons in category order (background to foreground)
    order = [0, 1, 3, 2, 4, 5]
    anns_by_cat = {c: [] for c in order}
    for ann in coco_data.get("annotations", []):
        cat_id = ann.get("category_id", -1)
        if cat_id in anns_by_cat:
            anns_by_cat[cat_id].append(ann)

    for cat_id in order:
        cat_name = HELIOS_CAT_MAP.get(cat_id, "leaf")
        color = ORGAN_COLORS.get(cat_name, (50, 180, 50))
        for ann in anns_by_cat[cat_id]:
            for poly in ann.get("segmentation", []):
                if len(poly) >= 6:
                    temp_img = Image.new("L", (W, H), 0)
                    ImageDraw.Draw(temp_img).polygon(poly, outline=1, fill=1)
                    poly_mask = np.array(temp_img, dtype=bool)
                    mask_img[poly_mask] = color

    return mask_img


def decode_pytorch_mask(multimodal_out: dict, H: int = 512, W: int = 512) -> np.ndarray:
    """Decodes PyTorch renderer multimodal organ masks into a colored RGB semantic mask."""
    mask_img = np.full((H, W, 3), ORGAN_COLORS["background"], dtype=np.uint8)
    organ_masks = multimodal_out.get("organ_masks", {})

    id_to_color = {
        0: ORGAN_COLORS["stem"],
        1: ORGAN_COLORS["petiole"],
        2: ORGAN_COLORS["leaf"],
        3: ORGAN_COLORS["peduncle"],
        4: ORGAN_COLORS["flower"],
        5: ORGAN_COLORS["fruit"],
    }

    if organ_masks:
        for ot_id in [0, 1, 3, 2, 4, 5]:
            if ot_id in organ_masks:
                m = organ_masks[ot_id].cpu().numpy().astype(bool)
                if m.shape[0] != H or m.shape[1] != W:
                    m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
                mask_img[m] = id_to_color[ot_id]
    else:
        fg_mask = multimodal_out.get("mask", None)
        if fg_mask is not None:
            m = fg_mask.cpu().numpy().astype(bool)
            if m.shape[0] != H or m.shape[1] != W:
                m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
            mask_img[m] = ORGAN_COLORS["leaf"]

    return mask_img


def generate_helios_sample(species: str, genotype: str, dap: int, seed: int, out_dir: str, build_dir: str, prefix: str):
    """Executes Helios C++ main with species-specific configuration."""
    main_bin = os.path.join(build_dir, "main")
    config_path = os.path.abspath(os.path.join(build_dir, f"../configs/params_{species}.json"))

    cmd = [
        main_bin,
        "--renderer", "radiation",
        "--save-xml",
        "--focus-plant",
        "--plant-type", species,
        "--genotype", genotype,
        "--dap", str(dap),
        "-s", str(seed),
        "-n", prefix,
        "--output", out_dir,
    ]
    if os.path.exists(config_path):
        cmd.extend(["-f", config_path])

    print(f"[HELIOS] Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=build_dir, check=True)
    dt = time.perf_counter() - t0
    return dt


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.path.join(repo_root, "Digital-Crops/projects/syntheticdata_generation/build")
    demo_dir = "/tmp/multispecies_demo"
    assets_dir = os.path.join(repo_root, "docs/results/assets")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device for PyTorch Renderer: {device}")

    renderer = HeliosPyTorchRenderer(image_size=512)

    species_list = [
        {
            "species": "cowpea",
            "genotype": "bush",
            "dap": 25,
            "seed": 12,
            "display_name": "Cowpea (Vigna unguiculata)",
            "prefix": "cowpea_demo",
        },
        {
            "species": "bean",
            "genotype": "erect",
            "dap": 22,
            "seed": 12,
            "display_name": "Common Bean (Phaseolus vulgaris)",
            "prefix": "bean_demo",
        },
        {
            "species": "sorghum",
            "genotype": "tall",
            "dap": 28,
            "seed": 12,
            "display_name": "Sorghum (Sorghum bicolor)",
            "prefix": "sorghum_demo",
        },
    ]

    results = []
    print("\n--- Generating & Benchmarking Species-Specific Plant Models ---")

    for s_info in species_list:
        sp = s_info["species"]
        gt = s_info["genotype"]
        dap = s_info["dap"]
        seed = s_info["seed"]
        prefix = s_info["prefix"]
        sp_dir = os.path.join(demo_dir, sp)
        os.makedirs(sp_dir, exist_ok=True)

        rad_path = os.path.join(sp_dir, f"{prefix}_0000_rad.jpeg")
        masks_path = os.path.join(sp_dir, f"{prefix}_0000_masks.json")
        xml_path = os.path.join(sp_dir, f"{prefix}_0000_plant_0000.xml")
        if not os.path.exists(xml_path):
            xml_path = os.path.join(sp_dir, f"{prefix}_plant_0000.xml")

        # Generate fresh sample with new binary & species configs
        print(f"\n[GENERATING] {s_info['display_name']} (DAP {dap}, {gt})...")
        h_time_sec = generate_helios_sample(sp, gt, dap, seed, demo_dir, build_dir, prefix)

        # Refresh xml_path if needed
        if not os.path.exists(xml_path):
            xml_path = os.path.join(sp_dir, f"{prefix}_0000_plant_0000.xml")
        if not os.path.exists(xml_path):
            xml_path = os.path.join(sp_dir, f"{prefix}_plant_0000.xml")

        # Load Helios RGB & Mask
        helios_rgb_img = Image.open(rad_path).convert("RGB")
        W_h, H_h = helios_rgb_img.size
        helios_rgb = np.array(helios_rgb_img)
        helios_mask = decode_helios_mask(masks_path, H=H_h, W=W_h)

        # Load XML to Part Tensor
        organ_array = PlantOrganArray.from_xml_file(xml_path)
        part_tensor = organ_array.to_part_tensor().to(device)

        # Warm up PyTorch renderer
        _ = renderer.render_part_tensor_multimodal(part_tensor, species=sp, device=device, focus_plant=True)

        # Benchmark PyTorch renderer (20 iterations)
        num_iters = 20
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            py_out = renderer.render_part_tensor_multimodal(part_tensor, species=sp, device=device, focus_plant=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
        py_time_sec = (time.perf_counter() - t0) / num_iters

        # Convert PyTorch outputs
        py_rgb_t = py_out["rgb"].permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
        py_rgb = (py_rgb_t * 255).astype(np.uint8)
        py_mask = decode_pytorch_mask(py_out, H=py_rgb.shape[0], W=py_rgb.shape[1])

        speedup = h_time_sec / max(py_time_sec, 1e-6)

        print(f"[{sp.upper()}] {s_info['display_name']}:")
        print(f"  - Plant Organs: {part_tensor.shape[0]}")
        print(f"  - Helios OptiX Raytracer: {h_time_sec:.3f} s")
        print(f"  - PyTorch Differentiable Renderer: {py_time_sec*1000.0:.2f} ms ({speedup:.0f}× speedup)")

        results.append({
            "species": sp,
            "genotype": gt,
            "dap": dap,
            "display_name": s_info["display_name"],
            "num_organs": part_tensor.shape[0],
            "helios_rgb": helios_rgb,
            "helios_mask": helios_mask,
            "helios_time_sec": h_time_sec,
            "py_rgb": py_rgb,
            "py_mask": py_mask,
            "py_time_sec": py_time_sec,
            "speedup": speedup,
        })

    # -------------------------------------------------------------
    # Render Combined 4-Column Multi-Species Grid Figure
    # -------------------------------------------------------------
    plt.style.use("dark_background")
    num_rows = len(results)
    fig, axes = plt.subplots(num_rows, 4, figsize=(18, 5.0 * num_rows), dpi=220)

    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    col_headers = [
        "Col 1: Helios Renderer (RGB)",
        "Col 2: Helios Renderer Mask",
        "Col 3: Python PyTorch Renderer (RGB)",
        "Col 4: Python PyTorch Renderer Mask",
    ]

    for row_idx, res in enumerate(results):
        imgs = [res["helios_rgb"], res["helios_mask"], res["py_rgb"], res["py_mask"]]
        h_time_str = f"Helios Raytracer: {res['helios_time_sec']:.2f}s"
        p_time_str = f"PyTorch Renderer: {res['py_time_sec']*1000.0:.1f}ms ({res['speedup']:.0f}× speedup)"

        subtitles = [
            h_time_str,
            "COCO Ray-Cast Mask",
            p_time_str,
            "Differentiable Organ Mask",
        ]

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs[col_idx])
            ax.axis("off")

            # Column main title (top row only)
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=13, fontweight="bold", pad=12, color="#4ecdc4")

            # Subtitle with timing
            sub_col = "#ffda79" if "s" in subtitles[col_idx] and "Helios" in subtitles[col_idx] else ("#70a1ff" if "speedup" in subtitles[col_idx] else "#ced6e0")
            ax.text(
                0.5, -0.06, subtitles[col_idx],
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=11,
                fontweight="semibold",
                color=sub_col,
            )

        # Row label on the left
        row_title = f"{res['display_name']}\n(DAP {res['dap']}, {res['num_organs']} organs, {res['genotype'].capitalize()})"
        axes[row_idx, 0].text(
            -0.12, 0.5, row_title,
            transform=axes[row_idx, 0].transAxes,
            ha="right", va="center",
            fontsize=12, fontweight="bold",
            color="#f1f2f6",
            rotation=90,
        )

    plt.suptitle(
        "Multi-Species Plant Architecture: Helios C++ Raytracer vs. PyTorch Differentiable Renderer",
        fontsize=16,
        fontweight="bold",
        y=0.995,
        color="#ffffff",
    )
    plt.tight_layout(rect=[0.06, 0.03, 0.98, 0.97])

    out_panel_path = os.path.join(assets_dir, "multi_species_render_comparison.png")
    fig.savefig(out_panel_path, bbox_inches="tight", dpi=220, facecolor="#18191f")
    plt.close(fig)
    print(f"\n[SUCCESS] Combined multi-species comparison panel saved: {out_panel_path}")

    # Also save single species panels
    for res in results:
        sp = res["species"]
        fig_s, axes_s = plt.subplots(1, 4, figsize=(18, 5.0), dpi=220)
        imgs = [res["helios_rgb"], res["helios_mask"], res["py_rgb"], res["py_mask"]]
        h_time_str = f"Helios Raytracer: {res['helios_time_sec']:.2f}s"
        p_time_str = f"PyTorch Renderer: {res['py_time_sec']*1000.0:.1f}ms ({res['speedup']:.0f}× speedup)"
        subtitles = [h_time_str, "COCO Ray-Cast Mask", p_time_str, "Differentiable Organ Mask"]

        for col_idx in range(4):
            ax = axes_s[col_idx]
            ax.imshow(imgs[col_idx])
            ax.axis("off")
            ax.set_title(col_headers[col_idx], fontsize=12, fontweight="bold", pad=10, color="#4ecdc4")
            sub_col = "#ffda79" if "s" in subtitles[col_idx] and "Helios" in subtitles[col_idx] else ("#70a1ff" if "speedup" in subtitles[col_idx] else "#ced6e0")
            ax.text(0.5, -0.07, subtitles[col_idx], transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="semibold", color=sub_col)

        plt.suptitle(f"{res['display_name']} (DAP {res['dap']}, {res['genotype'].capitalize()}) — Rendering & Mask Comparison", fontsize=14, fontweight="bold", y=1.02, color="#ffffff")
        plt.tight_layout()
        single_path = os.path.join(assets_dir, f"{sp}_render_comparison.png")
        fig_s.savefig(single_path, bbox_inches="tight", dpi=220, facecolor="#18191f")
        plt.close(fig_s)
        print(f"  ✓ Saved single species panel: {single_path}")


if __name__ == "__main__":
    main()
