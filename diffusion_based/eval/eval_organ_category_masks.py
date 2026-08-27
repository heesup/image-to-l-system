"""
Per-organ-category mask comparison between Helios C++ COCO ground truth and
the PyTorch organ-array renderer.

Category mapping (mesh OT id == COCO category id):
  1 petiole | 2 leaf | 3 floral_bud (=C++ peduncle objects) | 4 flower | 5 pod

Two camera modes are compared:
  auto : HFOV auto-fit from the PyTorch mesh bbox (previous behavior)
  exact: HFOV read from the Helios camera.json (focal_length / sensor_width)

Usage:
  python diffusion_based/eval/eval_organ_category_masks.py \
      --samples "dataset/helios_data/cowpea_dap080_seed00*,dataset/helios_data/cowpea_dap100_seed01*"
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

CATEGORIES = {
    1: "petiole",
    2: "leaf",
    3: "floral_bud",
    4: "flower",
    5: "pod",
}


def decode_coco_masks(json_path: str) -> dict:
    """Decode per-category binary masks from a Helios COCO masks.json."""
    with open(json_path, "r") as f:
        data = json.load(f)
    img_info = data["images"][0]
    W, H = img_info["width"], img_info["height"]
    masks = {cid: np.zeros((H, W), dtype=bool) for cid in CATEGORIES}
    for ann in data["annotations"]:
        cid = ann["category_id"]
        if cid not in masks:
            continue
        for poly in ann["segmentation"]:
            img = Image.new("L", (W, H), 0)
            ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
            masks[cid] |= np.array(img, dtype=bool)
    return masks


def hfov_from_camera_json(camera_json_path: str) -> float:
    """Exact HFOV (deg) from Helios camera.json intrinsics."""
    with open(camera_json_path, "r") as f:
        cam = json.load(f)
    props = cam["camera_properties"]
    focal = float(props["focal_length"])        # mm
    sensor_w = float(props["sensor_width"])      # mm
    return float(np.degrees(2.0 * np.arctan(sensor_w / (2.0 * focal))))


def iou(m1: np.ndarray, m2: np.ndarray) -> float:
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(m1, m2).sum() / union)


def evaluate_sample(prefix: str, builder: HeliosPlantGeometryBuilder,
                    renderer: HeliosPyTorchRenderer, device, mode: str,
                    output_dir: str, save_figure: bool = True) -> dict:
    xml_path = f"{prefix}_plant_0000.xml"
    masks_json = f"{prefix}_masks.json"
    camera_json = f"{prefix}_camera.json"
    rad_path = f"{prefix}_rad.jpeg"

    gt_masks = decode_coco_masks(masks_json)
    H, W = gt_masks[2].shape

    organ_array = PlantOrganArray.from_xml_file(xml_path)
    mesh_dict = builder.build_mesh_from_part_tensor(organ_array.to_part_tensor(device=device), device=device)

    hfov = None
    if mode == "exact":
        hfov = hfov_from_camera_json(camera_json)

    organ_buffer = renderer.render_organ_type_buffer(
        mesh_dict, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
        focus_plant=True, image_size=W, hfov_override_deg=hfov,
    ).cpu().numpy()

    results = {}
    for cid, name in CATEGORIES.items():
        pred = organ_buffer == cid
        results[name] = iou(pred, gt_masks[cid])

    if save_figure:
        os.makedirs(output_dir, exist_ok=True)
        rad_rgb = np.array(Image.open(rad_path).convert("RGB"), dtype=np.float32) / 255.0
        pred_masks = {cid: (organ_buffer == cid) for cid in CATEGORIES}
        fig, axes = plt.subplots(3, len(CATEGORIES), figsize=(3.2 * len(CATEGORIES), 10))
        for col, (cid, name) in enumerate(CATEGORIES.items()):
            axes[0, col].imshow(gt_masks[cid], cmap="gray")
            axes[0, col].set_title(f"GT {name}\n{gt_masks[cid].sum()} px", fontsize=10)
            axes[1, col].imshow(pred_masks[cid], cmap="gray")
            axes[1, col].set_title(f"Pred {name}\n{pred_masks[cid].sum()} px", fontsize=10)
            comp = np.zeros((H, W, 3))
            comp[gt_masks[cid], 0] = 1.0
            comp[pred_masks[cid], 1] = 1.0
            axes[2, col].imshow(comp)
            axes[2, col].set_title(f"IoU={results[name]:.3f}", fontsize=10)
            for r in range(3):
                axes[r, col].axis("off")
        fig.suptitle(f"{os.path.basename(prefix)} ({mode} FOV)", fontsize=12)
        plt.tight_layout()
        name_tag = os.path.basename(prefix)
        out_png = os.path.join(output_dir, f"{name_tag}_{mode}_categories.png")
        plt.savefig(out_png, dpi=130)
        plt.close()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str,
                        default="dataset/helios_data/cowpea_dap080_seed00*,dataset/helios_data/cowpea_dap100_seed00*")
    parser.add_argument("--output-dir", default="diffusion_based/eval/output/category_masks")
    parser.add_argument("--modes", type=str, default="auto,exact")
    parser.add_argument("--image-size", type=int, default=720)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    builder = HeliosPlantGeometryBuilder(use_generic_leaves=False, leaf_scale_factor=1.0,
                                         tube_radial_subdivisions=6)
    renderer = HeliosPyTorchRenderer(image_size=args.image_size)
    renderer.geo_builder = builder

    prefixes = []
    for pattern in args.samples.split(","):
        prefixes.extend(sorted(glob.glob(pattern.rstrip("_"))))
    prefixes = sorted(set(p.replace("_plant_0000.xml", "") for p in prefixes))
    print(f"Evaluating {len(prefixes)} samples, modes={args.modes}")

    all_results = {}
    for prefix in prefixes:
        if not os.path.exists(f"{prefix}_masks.json"):
            print(f"[skip] no masks.json for {prefix}")
            continue
        all_results[prefix] = {}
        for mode in args.modes.split(","):
            res = evaluate_sample(prefix, builder, renderer, device, mode,
                                  args.output_dir, save_figure=not args.no_figures)
            all_results[prefix][mode] = res
            pretty = " ".join(f"{k}={v:.3f}" for k, v in res.items())
            print(f"  [{mode:>5}] {os.path.basename(prefix)}: {pretty}")

    print("\n" + "=" * 84)
    print(f"{'sample':>40} {'mode':>6} " + " ".join(f"{n:>11}" for n in CATEGORIES.values()))
    print("-" * 84)
    for prefix, modes in all_results.items():
        for mode, res in modes.items():
            print(f"{os.path.basename(prefix):>40} {mode:>6} " +
                  " ".join(f"{res[n]:>11.3f}" for n in CATEGORIES.values()))
    print("=" * 84)

    for mode in args.modes.split(","):
        vals = [m[mode] for m in all_results.values() if mode in m]
        if vals:
            means = {n: float(np.mean([v[n] for v in vals])) for n in CATEGORIES.values()}
            print(f"MEAN [{mode}]: " + " ".join(f"{n}={v:.3f}" for n, v in means.items()))

    out_json = os.path.join(args.output_dir, "category_mask_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved summary to {out_json}")


if __name__ == "__main__":
    main()