"""
High-Throughput Tensor Generator: two output modes + progressive zoom pyramid.

Modes (--mode):
  shard  - packed shards of `--shard-size` samples (for distributed training pipelines
           that read .pt lists). Image layout follows --pyramid (below).
  cache  - one .pt PER SAMPLE (PartArrayDataset fast path). This replaces the former
           scripts/cache_dataset_tensors.py, which has been deleted.

Pyramid option (--pyramid):
  none     - (4,  H, W)        RGB(3, [-1,1]) + CHM depth(1, meters)
  concat   - (16, H, W)        the 4-channel image rendered at zoom 1x, 2x, 4x, 8x,
                               concatenated along the channel dimension:
       [ z1_rgb(3) | z1_depth(1) | z2_rgb(3) | z2_depth(1) |
         z4_rgb(3) | z4_depth(1) | z8_rgb(3) | z8_depth(1) ]
    All zoom levels share the same canvas: zoom k crops the central 1/k window of
    the field and upsamples it back to (H, W), so channels are pixel-aligned and
    can be concatenated losslessly. The model receives progressively finer detail
    (stem/leaf facets at 8x) without extra resolution.

Data selection (--species):
  cowpea  - only dataset/helios_data/cowpea (default)
  bean    - only bean
  all     - both

Crop-named caches: with --crop-suffix, cache mode writes to
  <cache-root>/<crop>_curv26/   (e.g. cowpea_curv26/)
so multiple crops never mix.
"""

import os
import sys
import glob
import math
import random
import argparse
import re
import time
from typing import List, Dict, Any, Tuple, Optional

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0+PTX"

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray, NUM_FEATURES_PART, ORGAN_NONE, P_COL_ORGAN_TYPE,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.dataset.part_array_dataset import (
    encode_fm, FM_NODE_DIM, NUM_ORGAN_CATEGORIES, EMPTY_IDX,
)

PYRAMID_ZOOMS = [1.0, 2.0, 4.0, 8.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Plant Tensor Generator (shard / per-sample cache)")
    parser.add_argument("--mode", type=str, default="cache", choices=["shard", "cache"],
                        help="shard: packed shards; cache: one .pt per sample (PartArrayDataset fast path)")
    parser.add_argument("--species", type=str, default="cowpea", choices=["cowpea", "bean", "all"],
                        help="Which crop's XML models to generate from (default: cowpea only)")
    parser.add_argument("--data-root", type=str, default="dataset/helios_data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override the output directory (default depends on mode + --crop-suffix)")
    parser.add_argument("--crop-suffix", action="store_true", default=True,
                        help="Append crop name to the output dir to keep crops separate (e.g. cowpea_curv26)")
    parser.add_argument("--pyramid", type=str, default="concat", choices=["none", "concat"],
                        help="none: single 4-ch image; concat: 4 channels x 4 zoom levels = 16-ch channels")
    parser.add_argument("--total-samples", type=int, default=100000)
    parser.add_argument("--num-workers", type=int, default=20, help="Parallel SLURM workers")
    parser.add_argument("--worker-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-slots", type=int, default=4096)
    parser.add_argument("--max-templates", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parsed = parser.parse_args()
    crop = "all" if parsed.species == "all" else parsed.species
    if parsed.output_dir is None:
        if parsed.mode == "cache":
            base = os.path.join(repo_root, "dataset", "cache")
            parsed.output_dir = os.path.join(base, f"{crop}_curv26") if parsed.crop_suffix else base
        else:
            parsed.output_dir = os.path.join(parsed.data_root, f"{crop}_shard_curv26")
    return parsed


def load_species_xml_samples(data_root: str, species: str) -> List[Dict[str, Any]]:
    """Discovers plant XML files for the given crop (or both with 'all')."""
    if species == "all":
        search_path = os.path.join(data_root, "**", "*_plant_*.xml")
    else:
        search_path = os.path.join(data_root, species, "**", "*_plant_*.xml")

    xml_files = sorted(glob.glob(search_path, recursive=True))
    if not xml_files:
        flat = os.path.join(data_root, species, "*_plant_*.xml") if species != "all" else os.path.join(data_root, "*_plant_*.xml")
        xml_files = sorted(glob.glob(flat))

    samples = []
    for x in xml_files:
        bn = os.path.basename(x)
        dap = 30.0
        m = re.search(r"dap(\d+)", bn)
        if m:
            dap = float(m.group(1))
        samples.append({"xml": x, "dap": dap, "filename": bn})
    return samples


def encode_sample(
    arr: PlantOrganArray,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    max_slots: int,
    use_pyramid: bool,
    dap: float,
) -> Optional[Dict[str, Any]]:
    """
    Renders one XML into a training sample dict (nodes + image).
    Image channels: 4 (RGB[-1,1] + CHM meters) per zoom level; 16 channels total
    when --pyramid concat (zooms 1,2,4,8 concatenated along channels).
    """
    part_13d = arr.to_part_tensor()
    num_nodes = min(part_13d.shape[0], max_slots)

    nodes_fm = torch.zeros((max_slots, FM_NODE_DIM))
    nodes_fm[:num_nodes] = encode_fm(part_13d[:num_nodes])
    nodes_fm[num_nodes:, EMPTY_IDX] = 1.0  # EMPTY padding rows

    mesh = renderer.geo_builder.build_mesh_from_part_tensor(
        arr.to_part_tensor(device=device), device=device)

    zooms = PYRAMID_ZOOMS if use_pyramid else [1.0]
    channel_imgs = []
    for zoom in zooms:
        with torch.no_grad():
            rgbd = renderer.render_mesh(
                mesh,
                azimuth_deg=0.0,
                elevation_deg=90.0,
                camera_height=5.0,
                background="ground",
                focus_plant=True,
                include_depth=True,
                zoom_factor=float(zoom),
                reference_window_size=1.2,
            )  # (4, H, W)
        rgb_norm = (rgbd[:3].clamp(0.0, 1.0) - 0.5) / 0.5
        depth_ch = rgbd[3:4].clamp(min=0.0)
        channel_imgs.append(torch.cat([rgb_norm, depth_ch], dim=0))

    image = torch.cat(channel_imgs, dim=0)  # (4*len(zooms), H, W)
    return {
        "nodes": nodes_fm.cpu(),
        "dap": torch.tensor(dap, dtype=torch.float32),
        "num_organs": torch.tensor(num_nodes, dtype=torch.long),
        "existence_mask": (part_13d[:num_nodes, P_COL_ORGAN_TYPE] > ORGAN_NONE).cpu(),
        "image": image.half().cpu(),
        "num_zooms": len(zooms),
        "zooms": zooms,
    }


def generate_cache(
    species: str,
    data_root: str,
    output_dir: str,
    total_samples: int,
    num_workers: int,
    worker_id: int,
    image_size: int,
    max_slots: int,
    max_templates: int,
    device_str: str,
    use_pyramid: bool,
):
    """One .pt per sample, written to <output_dir>/<prefix>.pt (PartArrayDataset fast path)."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)
    all_xml = load_species_xml_samples(data_root, species)
    if not all_xml:
        raise FileNotFoundError(f"No XML plant models for species '{species}' in {data_root}")

    per_worker = (len(all_xml) + num_workers - 1) // num_workers
    lo = worker_id * per_worker
    hi = min(lo + per_worker, len(all_xml))
    shard_slice = all_xml[lo:hi]
    print(f"[Cache worker {worker_id}/{num_workers}] {lo} -> {hi} ({hi - lo} samples) -> {output_dir}")

    renderer = HeliosPyTorchRenderer(image_size=image_size, device=device)
    t0 = time.time()
    ok = 0
    for k, s_info in enumerate(shard_slice):
        prefix = os.path.basename(s_info["xml"]).split("_plant_")[0]
        out_pt = os.path.join(output_dir, f"{prefix}.pt")
        if os.path.exists(out_pt):
            ok += 1
            continue
        try:
            arr = PlantOrganArray.from_xml_file(s_info["xml"])
            sample = encode_sample(arr, renderer, device, max_slots, use_pyramid, dap=s_info["dap"])
            if sample is None:
                continue
            sample["dap"] = torch.tensor(s_info["dap"], dtype=torch.float32)
            sample["prefix"] = prefix
            sample["xml"] = s_info["xml"]
            torch.save(sample, out_pt)
            ok += 1
        except Exception:
            continue
        if (k + 1) % 500 == 0:
            rate = (k + 1) / max(time.time() - t0, 1e-3)
            print(f"  {k + 1}/{hi - lo} cached | {rate:.1f} samples/s", flush=True)

    el = time.time() - t0
    print(f"[Cache worker {worker_id}] ✅ {ok}/{hi - lo} samples cached in {el:.1f}s "
          f"({ok / max(el, 1e-3):.1f}/s) -> {output_dir}")


def generate_shards_packed(
    species: str,
    data_root: str,
    output_dir: str,
    total_samples: int,
    num_workers: int,
    worker_id: int,
    shard_size: int,
    image_size: int,
    max_slots: int,
    max_templates: int,
    device_str: str,
    use_pyramid: bool,
):
    """Packed shards (legacy shard layout), each image 4 or 16 channels."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)
    all_xml_samples = load_species_xml_samples(data_root, species)
    if not all_xml_samples:
        raise FileNotFoundError(f"No XML plant models found in '{data_root}' for species '{species}'.")

    samples_per_worker = (total_samples + num_workers - 1) // num_workers
    start_idx = worker_id * samples_per_worker
    end_idx = min(start_idx + samples_per_worker, total_samples)

    print(f"[Shard worker {worker_id:02d}/{num_workers:02d}] Range [{start_idx:,} -> {end_idx:,}) -> {output_dir}")
    renderer = HeliosPyTorchRenderer(image_size=image_size, device=device)

    worker_templates = random.sample(all_xml_samples, min(len(all_xml_samples), max_templates))
    cached_templates = []
    for s_info in worker_templates:
        try:
            arr = PlantOrganArray.from_xml_file(s_info["xml"])
            part_13d = arr.to_part_tensor()
            num_nodes = min(part_13d.shape[0], max_slots)
            nodes_fm = encode_fm(part_13d[:num_nodes])
            if num_nodes < max_slots:
                nodes_fm = torch.cat([nodes_fm, torch.ones((max_slots - num_nodes, FM_NODE_DIM))], dim=0)
                nodes_fm[num_nodes:, EMPTY_IDX] = 1.0
                nodes_fm[num_nodes:, :EMPTY_IDX] = 0.0
                nodes_fm[num_nodes:, EMPTY_IDX + 1:] = 0.0
            mesh = renderer.geo_builder.build_mesh_from_part_tensor(
                arr.to_part_tensor(device=device), device=device)
            cached_templates.append({
                "mesh": mesh,
                "nodes_fm": nodes_fm.cpu(),
                "dap": float(s_info["dap"]),
                "num_organs": num_nodes,
                "existence_mask": (part_13d[:num_nodes, P_COL_ORGAN_TYPE] > ORGAN_NONE).cpu(),
            })
        except Exception:
            continue

    if not cached_templates:
        raise RuntimeError(f"[Shard worker {worker_id}] No valid XML templates loaded!")

    num_shards_expected = (samples_per_worker + shard_size - 1) // shard_size
    generated = 0
    t0 = time.time()
    for s_id in range(num_shards_expected):
        shard_filename = os.path.join(output_dir, f"shard_w{worker_id:02d}_s{s_id:04d}.pt")
        if os.path.exists(shard_filename):
            continue
        shard_samples = []
        for i in range(shard_size):
            global_idx = start_idx + s_id * shard_size + i
            if global_idx >= end_idx:
                break
            tmpl = random.choice(cached_templates)
            saz = random.uniform(0.0, 360.0)
            sel = random.uniform(30.0, 85.0)
            sel_rad, saz_rad = math.radians(sel), math.radians(saz)
            light_dir = torch.tensor([
                math.cos(sel_rad) * math.sin(saz_rad),
                -math.cos(sel_rad) * math.cos(saz_rad),
                math.sin(sel_rad)], dtype=torch.float32, device=device)
            try:
                with torch.no_grad():
                    per_zoom = []
                    for zoom in (PYRAMID_ZOOMS if use_pyramid else [1.0]):
                        rgbd = renderer.render_mesh(
                            tmpl["mesh"], azimuth_deg=0.0, elevation_deg=90.0,
                            camera_height=5.0, background="ground", focus_plant=True,
                            include_depth=True, zoom_factor=float(zoom),
                            reference_window_size=1.2,
                        )
                        rgb_norm = (rgbd[:3].clamp(0.0, 1.0) - 0.5) / 0.5
                        per_zoom.append(torch.cat([rgb_norm, rgbd[3:4].clamp(min=0.0)], dim=0))
                    img = torch.cat(per_zoom, dim=0)  # (4*zooms, H, W)
                shard_samples.append({
                    "image": img.half().cpu(),
                    "nodes": tmpl["nodes_fm"],
                    "dap": torch.tensor(tmpl["dap"], dtype=torch.float32),
                    "camera_view": torch.tensor([0.0, 90.0, 5.0], dtype=torch.float32),
                    "sun_view": torch.tensor([saz, sel], dtype=torch.float32),
                    "num_organs": torch.tensor(tmpl["num_organs"], dtype=torch.long),
                    "existence_mask": tmpl["existence_mask"],
                    "sample_id": global_idx,
                })
            except Exception:
                continue
        if shard_samples:
            torch.save(shard_samples, shard_filename)
            generated += 1
            if (s_id + 1) % 5 == 0 or (s_id + 1) == num_shards_expected:
                rate = (generated * shard_size) / max(time.time() - t0, 1e-3)
                print(f"[Shard worker {worker_id:02d}] Saved shard {s_id:04d}/{num_shards_expected:04d} "
                      f"({len(shard_samples)}) | {rate:.1f}/s", flush=True)

    print(f"[Shard worker {worker_id:02d}] ✅ {generated} shards in {time.time() - t0:.1f}s.")


def main():
    args = parse_args()
    use_pyramid = (args.pyramid == "concat")
    if args.mode == "cache":
        generate_cache(
            species=args.species, data_root=args.data_root, output_dir=args.output_dir,
            total_samples=args.total_samples, num_workers=args.num_workers,
            worker_id=args.worker_id, image_size=args.image_size, max_slots=args.max_slots,
            max_templates=args.max_templates, device_str=args.device, use_pyramid=use_pyramid,
        )
    else:
        generate_shards_packed(
            species=args.species, data_root=args.data_root, output_dir=args.output_dir,
            total_samples=args.total_samples, num_workers=args.num_workers,
            worker_id=args.worker_id, shard_size=args.shard_size, image_size=args.image_size,
            max_slots=args.max_slots, max_templates=args.max_templates, device_str=args.device,
            use_pyramid=use_pyramid,
        )


if __name__ == "__main__":
    main()