"""
High-Speed Multiprocess Dataset Tensor Pre-Cacher.

Iterates over all XML/JPEG pairs across all plant species in dataset/helios_data,
pre-parses them into normalized 16D/26D PyTorch tensors and resized image tensors,
and saves them to dataset/cache/ for instant, 0-overhead training.
"""

import os
import sys
import glob
import time
from multiprocessing import Pool, cpu_count
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.dataset.part_array_dataset import PartArrayDataset, FM_NODE_DIM, EMPTY_IDX

IMAGE_SIZE = 128
MAX_NODES = 512
CACHE_DIR = os.path.join(REPO_ROOT, "dataset/cache")


def process_single_sample(args):
    xml_path, jpeg_path, prefix, cache_dir = args
    out_pt = os.path.join(cache_dir, f"{prefix}.pt")

    try:
        # 1. Parse XML to PlantOrganArray
        gt_arr = PlantOrganArray.from_xml_file(xml_path)
        part = gt_arr.to_part_tensor(device=torch.device("cpu"))
        N = min(part.shape[0], MAX_NODES)

        # 2. Dummy dataset instance helper for encode_fm
        ds_helper = PartArrayDataset(data_root="dataset/helios_data", max_nodes=MAX_NODES, image_size=IMAGE_SIZE)
        nodes = torch.zeros((MAX_NODES, FM_NODE_DIM), dtype=torch.float32)
        nodes[:N] = ds_helper.encode_fm(part[:N])
        existence_mask = (nodes[:, EMPTY_IDX] < 0.5).float()

        # 3. Load & preprocess JPEG image
        with Image.open(jpeg_path) as pil_img:
            pil_rgb = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            arr_img = np.array(pil_rgb, dtype=np.float32) / 255.0
            rgb_t = torch.from_numpy(arr_img).permute(2, 0, 1)
            image_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(rgb_t)

        # 4. Save combined sample dict
        torch.save({
            "image": image_tensor,
            "nodes": nodes,
            "existence_mask": existence_mask,
            "num_nodes": torch.tensor(N, dtype=torch.long),
            "prefix": prefix,
            "xml": xml_path,
        }, out_pt)
        return True
    except Exception as e:
        return False


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Scanning dataset in {os.path.join(REPO_ROOT, 'dataset/helios_data')}...")

    xml_paths = sorted(glob.glob(os.path.join(REPO_ROOT, "dataset/helios_data", "**", "*_plant_*.xml"), recursive=True))
    xml_paths = [p for p in xml_paths if "/_tmp_" not in p and "_tmp_" not in os.path.basename(p)]

    tasks = []
    for xml_path in xml_paths:
        prefix = os.path.basename(xml_path).split("_plant_")[0]
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        jpeg_path = ""
        for suffix in ("_vis.jpeg", "_rad.jpeg"):
            cand = os.path.join(xml_dir, f"{prefix}{suffix}")
            if os.path.exists(cand):
                jpeg_path = cand
                break
        if jpeg_path and os.path.exists(jpeg_path):
            tasks.append((xml_path, jpeg_path, prefix, CACHE_DIR))

    print(f"Found {len(tasks)} valid XML/JPEG pairs to cache.")
    workers = min(32, cpu_count())
    print(f"Starting parallel caching with {workers} CPU workers...")

    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(process_single_sample, tasks)

    success_count = sum(results)
    elapsed = time.time() - t0
    print(f"\nSuccessfully cached {success_count} / {len(tasks)} samples in {elapsed:.2f}s ({success_count / max(elapsed, 0.01):.1f} samples/s) to {CACHE_DIR}")


if __name__ == "__main__":
    main()
