"""
High-Throughput Multi-Species Tensor Shard Generator.
Supports distributed SLURM Job Arrays (--array=0-39) and standalone GPU rendering.
Converts raw Helios XMLs into pre-rendered RGB images + 26D Flow Matching Node Tensors.
Outputs compressed tensor shards (.pt) to destination cache directory.
"""

import os
import sys
import glob
import math
import random
import argparse
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
    parser = argparse.ArgumentParser(description="Generate Plant Dataset Tensor Shards")
    parser.add_argument("--species", type=str, default="cowpea", help="Plant species (e.g. cowpea, bean, sorghum, all)")
    parser.add_argument("--data-root", type=str, default="dataset/helios_data", help="Root directory containing species XMLs")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination cache directory (default: <data-root>/<species>_shard)")
    parser.add_argument("--total-samples", type=int, default=100000, help="Total dataset size across all workers")
    parser.add_argument("--num-workers", type=int, default=20, help="Total number of parallel SLURM workers")
    parser.add_argument("--worker-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), help="Worker ID (0 to num_workers-1)")
    parser.add_argument("--shard-size", type=int, default=100, help="Samples per saved shard file")
    parser.add_argument("--image-size", type=int, default=128, help="Rendered RGB image resolution")
    parser.add_argument("--max-slots", type=int, default=4096, help="Max slots per plant tensor")
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


def augment_and_render_sample(
    sample_info: Dict[str, Any],
    renderer: HeliosPyTorchRenderer,
    image_transform: transforms.Compose,
    max_slots: int = 512,
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Loads base plant geometry, applies random camera and light viewpoint augmentation,
    and returns encoded 26D node tensor + normalized image tensor.
    """
    arr = PlantOrganArray.from_xml_file(sample_info["xml"])
    part_16d = arr.to_part_tensor()  # (N, 16)
    
    # 1. Random Camera & Lighting Viewpoints
    caz = random.uniform(0.0, 360.0)
    cel = random.uniform(30.0, 90.0)  # Elevation 30° to 90° (top-down)
    cam_h = random.uniform(0.9, 2.2)
    saz = random.uniform(0.0, 360.0)
    sel = random.uniform(20.0, 75.0)

    # Convert sun angles to light direction vector
    sel_rad = math.radians(sel)
    saz_rad = math.radians(saz)
    lx = math.cos(sel_rad) * math.sin(saz_rad)
    ly = -math.cos(sel_rad) * math.cos(saz_rad)
    lz = math.sin(sel_rad)
    light_dir = torch.tensor([lx, ly, lz], dtype=torch.float32, device=device)

    # Render on GPU
    mesh = renderer.geo_builder.build_mesh_from_organ_array(arr, device=device)
    rgb = renderer(
        mesh,
        azimuth_deg=caz,
        elevation_deg=cel,
        camera_height=cam_h,
        background="ground",
        light_dir=light_dir,
        focus_plant=True
    )  # (3, H, W)
    
    # Normalize RGB
    img_pil = transforms.ToPILImage()(rgb.clamp(0.0, 1.0).cpu())
    img_t = image_transform(img_pil)

    # 2. Encode 16D Part Tensor -> 26D Normalized Flow Matching Vector
    nodes_26d = torch.zeros((max_slots, FM_NODE_DIM), dtype=torch.float32)
    num_nodes = min(part_16d.shape[0], max_slots)

    ot = part_16d[:num_nodes, 0].long()  # organ_type
    exist = part_16d[:num_nodes, 13]    # existence

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

    return {
        "image": img_t.cpu(),
        "nodes": nodes_26d.cpu(),
        "dap": torch.tensor(sample_info["dap"], dtype=torch.float32),
        "camera_view": torch.tensor([caz, cel, cam_h], dtype=torch.float32),
        "sun_view": torch.tensor([saz, sel], dtype=torch.float32),
        "num_organs": torch.tensor(num_nodes, dtype=torch.long),
        "existence_mask": (exist > 0.5).cpu(),
    }


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
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Core sharding generation function callable from scripts or training modules."""
    if output_dir is None:
        output_dir = os.path.join(data_root, f"{species}_shard")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    samples_per_worker = total_samples // num_workers
    start_idx = worker_id * samples_per_worker
    end_idx = start_idx + samples_per_worker
    
    print(f"================================================================")
    print(f"Plant Dataset Sharding Worker {worker_id:02d}/{num_workers} ({species})")
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

    num_shards_expected = (samples_per_worker + shard_size - 1) // shard_size
    print(f"[Worker {worker_id:02d}] Generating {num_shards_expected} shards ({samples_per_worker:,} samples)...")

    generated_shards = 0
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
            base_sample = random.choice(base_samples)
            try:
                with torch.no_grad():
                    data = augment_and_render_sample(
                        base_sample,
                        renderer=renderer,
                        image_transform=img_transform,
                        max_slots=max_slots,
                        device=device
                    )
                data["sample_id"] = global_idx
                shard_samples.append(data)
            except Exception as e:
                print(f"[Worker {worker_id:02d}] Error rendering sample: {e}", file=sys.stderr)
                continue

        if shard_samples:
            torch.save(shard_samples, shard_filename)
            generated_shards += 1
            print(f"[Worker {worker_id:02d}] Saved shard {s_id:04d} ({len(shard_samples)} samples) -> {shard_filename}")

    print(f"[Worker {worker_id:02d}] Completed. Generated {generated_shards} new shards.")


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
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
