"""
High-Throughput Multi-Species Tensor Shard Generator (Ultra-Fast In-Memory Preloaded Edition).
Supports distributed SLURM Job Arrays (--array=0-39) and standalone GPU rendering.
Caches base 3D plant meshes in GPU/RAM memory to eliminate XML disk parsing bottlenecks.
Outputs compressed tensor shards (.pt) to destination cache directory at peak GPU throughput.
"""

import os
import sys
import glob
import math
import random
import argparse
import time
from typing import List, Dict, Any, Tuple, Optional

# Ensure multi-architecture CUDA support before loading PyTorch / nvdiffrast
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

from diffusion_based.models.plant_organ_array import PlantOrganArray, NUM_FEATURES_PART
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.dataset.part_array_dataset import (
    ORGAN_CATEGORIES, EMPTY_IDX, FM_NODE_DIM, FM_OT_END,
    FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END,
    FM_SCALE_START, FM_SCALE_END, FM_CURV_IDX, FM_PHYLLO_IDX,
    BASE_SCALE, SCALE_SCALE, CURVATURE_SCALE, PHYLLOTACTIC_SCALE
)


def parse_args():
    parser = argparse.ArgumentParser(description="High-Throughput Plant Dataset Tensor Shard Generator")
    parser.add_argument("--species", type=str, default="cowpea", help="Plant species (e.g. cowpea, bean, sorghum, all)")
    parser.add_argument("--data-root", type=str, default="dataset/helios_data", help="Root directory containing species XMLs")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination cache directory (default: <data-root>/<species>_shard)")
    parser.add_argument("--total-samples", type=int, default=100000, help="Total dataset size across all workers")
    parser.add_argument("--num-workers", type=int, default=20, help="Total number of parallel SLURM workers")
    parser.add_argument("--worker-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), help="Worker ID (0 to num_workers-1)")
    parser.add_argument("--shard-size", type=int, default=100, help="Samples per saved shard file")
    parser.add_argument("--image-size", type=int, default=512, help="Rendered RGB image resolution (default: 512)")
    parser.add_argument("--max-slots", type=int, default=4096, help="Max slots per plant tensor")
    parser.add_argument("--max-templates", type=int, default=50, help="Max base XML templates to pre-load per worker")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parsed = parser.parse_args()
    if parsed.output_dir is None:
        parsed.output_dir = os.path.join(parsed.data_root, f"{parsed.species}_shard")
    return parsed


def load_species_xml_samples(data_root: str = "dataset/helios_data", species: str = "cowpea") -> List[Dict[str, Any]]:
    """Discovers all plant XML files for specified species (or all species)."""
    if species == "all":
        search_path = os.path.join(data_root, "**", "*_plant_*.xml")
    else:
        search_path = os.path.join(data_root, species, "**", "*_plant_*.xml")
    
    xml_files = sorted(glob.glob(search_path, recursive=True))
    if not xml_files:
        # Fallback flat search
        flat_path = os.path.join(data_root, species, "*_plant_*.xml") if species != "all" else os.path.join(data_root, "*_plant_*.xml")
        xml_files = sorted(glob.glob(flat_path))

    samples = []
    for x in xml_files:
        bn = os.path.basename(x)
        dap = 30.0
        if "dap" in bn:
            try:
                import re
                m = re.search(r"dap(\d+)", bn)
                if m:
                    dap = float(m.group(1))
            except Exception:
                pass
        samples.append({"xml": x, "dap": dap, "filename": bn})
    return samples


def generate_shards(
    species: str = "cowpea",
    data_root: str = "dataset/helios_data",
    output_dir: Optional[str] = None,
    total_samples: int = 100000,
    num_workers: int = 1,
    worker_id: int = 0,
    shard_size: int = 100,
    image_size: int = 128,
    max_slots: int = 4096,
    max_templates: int = 50,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Ultra-fast in-memory cached GPU sharding pipeline."""
    if output_dir is None:
        output_dir = os.path.join(data_root, f"{species}_shard")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    samples_per_worker = total_samples // num_workers
    start_idx = worker_id * samples_per_worker
    end_idx = start_idx + samples_per_worker
    
    print(f"================================================================")
    print(f"🚀 High-Throughput Plant Sharding Worker {worker_id:02d}/{num_workers} ({species})")
    print(f"Target Range: [{start_idx:,} -> {end_idx:,}] ({samples_per_worker:,} samples)")
    print(f"Device: {device_str} | Output: {output_dir}")
    print(f"================================================================")

    base_samples = load_species_xml_samples(data_root=data_root, species=species)
    if not base_samples:
        raise RuntimeError(f"No base plant XML samples found in {data_root}/{species}!")

    print(f"Found {len(base_samples)} unique base XML templates for '{species}'.")

    device = torch.device(device_str)
    renderer = HeliosPyTorchRenderer(image_size=image_size).to(device)

    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    random.seed(1000 + worker_id)
    torch.manual_seed(1000 + worker_id)

    # Subdivide base templates evenly across workers if num_workers > 1
    if num_workers > 1 and len(base_samples) > samples_per_worker:
        worker_templates = base_samples[worker_id::num_workers]
    else:
        worker_templates = base_samples

    # Cap templates per worker to max_templates to ensure < 2s startup
    if len(worker_templates) > max_templates:
        step = max(1, len(worker_templates) // max_templates)
        worker_templates = worker_templates[::step][:max_templates]

    # 1. Pre-load and Cache 3D Meshes & 26D Tensors in Memory to saturate GPU throughput
    t0_preload = time.time()
    print(f"[Worker {worker_id:02d}] Pre-loading {len(worker_templates)} XML templates into GPU/RAM cache...")
    cached_templates = []
    for s_info in worker_templates:
        try:
            arr = PlantOrganArray.from_xml_file(s_info["xml"])
            part_16d = arr.to_part_tensor()  # (N, 16)
            num_nodes = min(part_16d.shape[0], max_slots)

            # Pre-encode 26D normalized Flow Matching vector
            nodes_26d = torch.zeros((max_slots, FM_NODE_DIM), dtype=torch.float32)
            ot = part_16d[:num_nodes, 0].long()
            exist = part_16d[:num_nodes, 13]

            for i, cat in enumerate(ORGAN_CATEGORIES):
                mask = (ot == cat) & (exist > 0.5)
                nodes_26d[:num_nodes][mask, i] = 1.0
            nodes_26d[:num_nodes][exist <= 0.5, EMPTY_IDX] = 1.0
            if num_nodes < max_slots:
                nodes_26d[num_nodes:, EMPTY_IDX] = 1.0

            act = exist > 0.5
            nodes_26d[:num_nodes][act, FM_BASE_START:FM_BASE_END] = part_16d[:num_nodes][act, 1:4] * BASE_SCALE
            nodes_26d[:num_nodes][act, FM_ROT_START:FM_ROT_END] = part_16d[:num_nodes][act, 4:10]
            nodes_26d[:num_nodes][act, FM_SCALE_START:FM_SCALE_END] = part_16d[:num_nodes][act, 10:13] * SCALE_SCALE
            nodes_26d[:num_nodes][act, FM_CURV_IDX] = part_16d[:num_nodes][act, 14] / CURVATURE_SCALE
            nodes_26d[:num_nodes][act, FM_PHYLLO_IDX] = part_16d[:num_nodes][act, 15] / PHYLLOTACTIC_SCALE

            # Pre-build GPU Mesh
            mesh = renderer.geo_builder.build_mesh_from_part_tensor(arr.to_part_tensor(device=device), device=device)

            cached_templates.append({
                "mesh": mesh,
                "nodes_26d": nodes_26d.cpu(),
                "dap": float(s_info["dap"]),
                "num_organs": num_nodes,
                "existence_mask": (exist > 0.5).cpu(),
            })
        except Exception as e:
            continue

    if not cached_templates:
        raise RuntimeError(f"[Worker {worker_id:02d}] Failed to pre-load any valid XML templates!")

    preload_time = time.time() - t0_preload
    print(f"[Worker {worker_id:02d}] ✓ Preloaded {len(cached_templates)} templates in {preload_time:.2f}s.")

    num_shards_expected = (samples_per_worker + shard_size - 1) // shard_size
    print(f"[Worker {worker_id:02d}] Generating {num_shards_expected} shards ({samples_per_worker:,} samples)...")

    generated_shards = 0
    t0_render = time.time()
    for s_id in range(num_shards_expected):
        shard_filename = os.path.join(
            output_dir,
            f"shard_w{worker_id:02d}_s{s_id:04d}.pt"
        )
        if os.path.exists(shard_filename):
            print(f"[Worker {worker_id:02d}] Shard {s_id:04d} already exists. Skipping.")
            continue

        shard_samples = []
        for i in range(shard_size):
            global_idx = start_idx + s_id * shard_size + i
            if global_idx >= end_idx:
                break

            tmpl = random.choice(cached_templates)

            # Drone Orthophoto Top-View (Fixed 5.0m Height, 90.0° Nadir, Fixed 0.0° North Azimuth)
            caz = 0.0                         # Fixed 0.0° North-Aligned Azimuth (Orthophoto Standard)
            cel = 90.0                        # Fixed 90.0° Nadir Top-Down View
            cam_h = 5.0                       # Fixed 5.0m Drone Altitude
            saz = random.uniform(0.0, 360.0)  # Sun azimuth
            sel = random.uniform(30.0, 85.0)  # Sun elevation

            # Convert sun angles to light direction vector
            sel_rad = math.radians(sel)
            saz_rad = math.radians(saz)
            lx = math.cos(sel_rad) * math.sin(saz_rad)
            ly = -math.cos(sel_rad) * math.cos(saz_rad)
            lz = math.sin(sel_rad)
            light_dir = torch.tensor([lx, ly, lz], dtype=torch.float32, device=device)

            try:
                with torch.no_grad():
                    # 4-Channel Single-Pass Rendering (RGB 3ch + Metric Canopy Depth 1ch)
                    rgbd = renderer.render_mesh(
                        tmpl["mesh"],
                        azimuth_deg=caz,
                        elevation_deg=cel,
                        camera_height=cam_h,
                        background="ground",
                        focus_plant=True,
                        include_depth=True,
                    )  # (4, H, W)

                    # RGB normalized to [-1, 1], Depth in meters
                    rgb_norm = (rgbd[:3].clamp(0.0, 1.0) - 0.5) / 0.5
                    depth_ch = rgbd[3:4].clamp(min=0.0)
                    img_4d = torch.cat([rgb_norm, depth_ch], dim=0)  # (4, H, W)

                data = {
                    "image": img_4d.half().cpu(),
                    "nodes": tmpl["nodes_26d"],
                    "dap": torch.tensor(tmpl["dap"], dtype=torch.float32),
                    "camera_view": torch.tensor([caz, cel, cam_h], dtype=torch.float32),
                    "sun_view": torch.tensor([saz, sel], dtype=torch.float32),
                    "num_organs": torch.tensor(tmpl["num_organs"], dtype=torch.long),
                    "existence_mask": tmpl["existence_mask"],
                    "sample_id": global_idx,
                }
                shard_samples.append(data)
            except Exception as e:
                continue

        if shard_samples:
            torch.save(shard_samples, shard_filename)
            generated_shards += 1
            if (s_id + 1) % 5 == 0 or (s_id + 1) == num_shards_expected:
                elapsed = time.time() - t0_render
                rate = (generated_shards * shard_size) / max(elapsed, 1e-3)
                print(f"[Worker {worker_id:02d}] Saved shard {s_id:04d}/{num_shards_expected:04d} ({len(shard_samples)} samples) | {rate:.1f} samples/sec -> {shard_filename}")

    total_time = time.time() - t0_render
    print(f"[Worker {worker_id:02d}] ✅ Completed {generated_shards} shards in {total_time:.2f}s ({((generated_shards * shard_size) / max(total_time, 1e-3)):.1f} samples/sec).")


def main():
    args = parse_args()
    generate_shards(
        species=args.species,
        data_root=args.data_root,
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
        shard_size=args.shard_size,
        image_size=args.image_size,
        max_slots=args.max_slots,
        max_templates=args.max_templates,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
