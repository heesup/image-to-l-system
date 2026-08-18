"""
Precompute and cache 14D part tensors for all XML samples to disk.

The 14D part tensor is extracted via forward kinematics (to_part_tensor),
which currently builds the full mesh (tube meshes + leaf OBJ loading) — ~1-13s
per sample. This script computes each tensor ONCE and caches it to a .pt file,
so the training dataset can load them instantly.

Usage:
    python diffusion_based/dataset/precompute_part_tensors.py \
        --data_root dataset/helios_data \
        --cache_dir dataset/helios_data_14d_cache
"""

import os
import sys
import glob
import argparse
import time

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--cache_dir", type=str, default="dataset/helios_data_14d_cache")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    xml_paths = sorted(glob.glob(os.path.join(args.data_root, "*_plant_*.xml")))
    if args.end is not None:
        xml_paths = xml_paths[args.start:args.end]
    else:
        xml_paths = xml_paths[args.start:]

    print(f"Precomputing 14D tensors + images for {len(xml_paths)} samples -> {args.cache_dir}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
    renderer = HeliosPyTorchRenderer(image_size=args.image_size).to(device)

    done = 0
    t0 = time.time()
    for i, xml_path in enumerate(xml_paths):
        prefix = os.path.basename(xml_path).split("_plant_")[0]
        cache_path = os.path.join(args.cache_dir, f"{prefix}.pt")
        img_path = os.path.join(args.cache_dir, f"{prefix}_img.pt")
        if os.path.exists(cache_path) and os.path.exists(img_path):
            done += 1
            continue
        try:
            arr = PlantOrganArray.from_xml_file_typed(xml_path)
            p14 = arr.to_part_tensor(device=torch.device("cpu"))
            torch.save(p14, cache_path)
            # Render from the 14D tensor directly (fast: 0.1-0.8s) instead of
            # re-building the full mesh (slow: 20-40s).
            with torch.no_grad():
                rgb = renderer.render_part_tensor(
                    p14.to(device), template_organ_array=arr, camera_height=1.0,
                    elevation_deg=90.0, device=device, focus_plant=True,
                    use_kinematics_tree=False, differentiable=False,
                )
            torch.save(rgb.cpu(), img_path)
            done += 1
        except Exception as e:
            print(f"  FAIL {prefix}: {e}", flush=True)

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-6)
            print(f"  {i+1}/{len(xml_paths)} done ({rate:.2f} samples/s, ETA {(len(xml_paths)-i-1)/max(rate,1e-6):.0f}s)", flush=True)

    print(f"Done. {done} samples cached in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
