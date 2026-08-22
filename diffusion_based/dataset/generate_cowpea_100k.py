"""
High-Throughput Cowpea 100K Dataset Generator.
Supports distributed SLURM Job Arrays (--array=0-39) and multi-threaded GPU rendering.
Outputs compressed tensor shards to dataset/cache_cowpea_100k/
"""

import os
import sys
import glob
import math
import random
import argparse
from typing import List, Dict, Any, Tuple

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
    parser = argparse.ArgumentParser(description="Generate Cowpea 100K Dataset")
    parser.add_argument("--total-samples", type=int, default=100000, help="Total dataset size across all workers")
    parser.add_argument("--num-workers", type=int, default=40, help="Total number of parallel SLURM workers")
    parser.add_argument("--worker-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), help="Worker ID (0 to num_workers-1)")
    parser.add_argument("--output-dir", type=str, default="dataset/cache_cowpea_100k", help="Destination cache directory")
    parser.add_argument("--shard-size", type=int, default=100, help="Samples per saved shard file")
    parser.add_argument("--image-size", type=int, default=128, help="Rendered RGB image resolution")
    parser.add_argument("--max-slots", type=int, default=512, help="Max slots per plant tensor")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_base_cowpea_samples(data_root: str = "dataset/helios_data/cowpea") -> List[Dict[str, Any]]:
    xml_files = sorted(glob.glob(os.path.join(data_root, "**", "*_plant_*.xml"), recursive=True))
    if not xml_files:
        xml_files = sorted(glob.glob(os.path.join(data_root, "*_plant_*.xml")))
    
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
        samples.append({"xml": x, "dap": dap})
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
    part_16d = arr.to_part_tensor() # (N, 16)
    
    # 1. Random Camera & Lighting Viewpoints
    caz = random.uniform(0.0, 360.0)
    cel = random.uniform(30.0, 90.0) # Elevation 30° to 90° (top-down)
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
    ) # (3, H, W)
    
    # Normalize RGB
    img_pil = transforms.ToPILImage()(rgb.clamp(0.0, 1.0).cpu())
    img_t = image_transform(img_pil)

    # 2. Encode 16D Part Tensor -> 26D Normalized Flow Matching Vector
    nodes_26d = torch.zeros((max_slots, FM_NODE_DIM), dtype=torch.float32)
    num_nodes = min(part_16d.shape[0], max_slots)

    ot = part_16d[:num_nodes, 0].long() # organ_type
    exist = part_16d[:num_nodes, 13] # existence

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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    samples_per_worker = args.total_samples // args.num_workers
    start_idx = args.worker_id * samples_per_worker
    end_idx = start_idx + samples_per_worker
    
    print(f"================================================================")
    print(f"Cowpea 100K Dataset Worker {args.worker_id:02d}/{args.num_workers}")
    print(f"Target Range: [{start_idx:,} -> {end_idx:,}] ({samples_per_worker:,} samples)")
    print(f"Device: {args.device} | Output: {args.output_dir}")
    print(f"================================================================")

    base_samples = load_base_cowpea_samples()
    if not base_samples:
        print("Error: No base Cowpea samples found in dataset/helios_data/cowpea!")
        sys.exit(1)

    device = torch.device(args.device)
    renderer = HeliosPyTorchRenderer(image_size=args.image_size).to(device)

    img_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    current_shard = []
    shard_id = 0

    random.seed(1000 + args.worker_id)
    torch.manual_seed(1000 + args.worker_id)

    for i in range(samples_per_worker):
        global_idx = start_idx + i
        base_sample = random.choice(base_samples)
        
        try:
            with torch.no_grad():
                data = augment_and_render_sample(
                    base_sample,
                    renderer=renderer,
                    image_transform=img_transform,
                    max_slots=args.max_slots,
                    device=device
                )
            data["sample_id"] = global_idx
            current_shard.append(data)
        except Exception as e:
            continue

        if len(current_shard) >= args.shard_size or (i == samples_per_worker - 1 and current_shard):
            shard_filename = os.path.join(
                args.output_dir,
                f"shard_w{args.worker_id:02d}_s{shard_id:04d}.pt"
            )
            torch.save(current_shard, shard_filename)
            current_shard = []
            shard_id += 1
            if shard_id % 5 == 0 or (i == samples_per_worker - 1):
                progress = (i + 1) / samples_per_worker * 100.0
                print(f"[Worker {args.worker_id:02d}] Progress: {i+1:,}/{samples_per_worker:,} ({progress:.1f}%) | Shards Saved: {shard_id}")

    print(f"Worker {args.worker_id:02d} Finished Successfully! Generated {samples_per_worker:,} samples in {shard_id} shards.")


if __name__ == "__main__":
    main()
