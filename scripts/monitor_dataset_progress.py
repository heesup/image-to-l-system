"""
Dataset Generation Progress Monitor.

Tracks real-time generation progress across DAP 1..100 and seeds 0..99:
  - Total completed samples (XML, JPEG, Masks, Params)
  - Completion percentage towards target (10,000 samples)
  - SLURM job queue status
"""

import os
import glob
import subprocess
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_dir = os.path.join(repo_root, "dataset", "helios_data")


def check_progress():
    xml_files = glob.glob(os.path.join(dataset_dir, "*_plant_0000.xml"))
    jpeg_files = glob.glob(os.path.join(dataset_dir, "*_rad.jpeg"))
    mask_files = glob.glob(os.path.join(dataset_dir, "*_masks.json"))

    total_target = 10000
    completed = min(len(xml_files), len(jpeg_files))
    pct = (completed / total_target) * 100.0

    print("=" * 60)
    print("Cowpea 10,000 Dataset Batch Generation Progress")
    print("=" * 60)
    print(f"Target Samples:     {total_target:,} (DAP 1-100 × 100 Seeds)")
    print(f"Completed Samples:  {completed:,} ({pct:.1f}%)")
    print(f"  - XML Files:      {len(xml_files):,}")
    print(f"  - JPEG Renders:   {len(jpeg_files):,}")
    print(f"  - Mask JSONs:     {len(mask_files):,}")
    print(f"Remaining Samples:  {max(0, total_target - completed):,}")
    print("-" * 60)

    # Check SLURM queue
    try:
        res = subprocess.run(["squeue", "-u", os.environ.get("USER", "lion397")], capture_output=True, text=True)
        lines = [l for l in res.stdout.strip().split("\n") if "cowpea" in l]
        running = sum(1 for l in lines if " R " in l)
        pending = sum(1 for l in lines if " PD " in l)
        print(f"SLURM Jobs Status:  {len(lines)} total ({running} Running, {pending} Pending/Queued)")
    except Exception as e:
        print(f"SLURM check: {e}")

    print("=" * 60)


if __name__ == "__main__":
    check_progress()
