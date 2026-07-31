"""Batch generate Helios synthetic plant image + XML pairs for model training.

Example (quick test):
    python -m dataset.generate_helios_dataset --quick

Example (full DAP sweep with 5 seeds):
    python -m dataset.generate_helios_dataset --dap-start 5 --dap-end 60 --dap-step 5 --seeds 5
"""

import os
import sys
import argparse
import subprocess
import multiprocessing
from typing import List, Tuple
from tqdm import tqdm


def generate_one(args: Tuple[str, int, int, str]) -> Tuple[int, int, bool, str]:
    """Generate a single Helios sample.

    Args:
        args: (main_binary, dap, seed, output_dir)
    Returns:
        (dap, seed, success, message)
    """
    main_binary, dap, seed, output_dir = args
    name = f"cowpea_dap{dap:03d}_seed{seed:02d}"
    build_dir = os.path.dirname(main_binary)
    cmd = [
        main_binary,
        "--radiation", "false",
        "--renderer", "vis",
        "--vis",
        "--focus-plant",
        "-n", name,
        "--dap", str(dap),
        "-s", str(seed),
        "--output", output_dir,
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=1800,
        )
        success = result.returncode == 0 and "Saved Visualizer image" in result.stdout
        return dap, seed, success, result.stdout
    except subprocess.TimeoutExpired:
        return dap, seed, False, "Timeout"
    except Exception as e:
        return dap, seed, False, str(e)


def build_job_list(dap_start: int, dap_end: int, dap_step: int,
                   seeds: int, main_binary: str, output_dir: str) -> List[Tuple[str, int, int, str]]:
    jobs = []
    for dap in range(dap_start, dap_end + 1, dap_step):
        for seed in range(seeds):
            jobs.append((main_binary, dap, seed, output_dir))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: DAP 5,10,15 x seed 0,1")
    parser.add_argument("--dap-start", type=int, default=5)
    parser.add_argument("--dap-end", type=int, default=60)
    parser.add_argument("--dap-step", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds per DAP")
    parser.add_argument("--main-binary", type=str,
                        default="Digital-Crops/projects/syntheticdata_generation/build/main")
    parser.add_argument("--output-dir", type=str,
                        default="Digital-Crops/projects/syntheticdata_generation/build/output")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel processes; default = 4")
    parser.add_argument("--resume", action="store_true",
                        help="Skip existing outputs")
    args = parser.parse_args()

    main_binary = os.path.abspath(args.main_binary)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(main_binary):
        print(f"ERROR: main binary not found at {main_binary}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    if args.quick:
        dap_start, dap_end, dap_step = 5, 15, 5
        seeds = 2
    else:
        dap_start, dap_end, dap_step = args.dap_start, args.dap_end, args.dap_step
        seeds = args.seeds

    jobs = build_job_list(dap_start, dap_end, dap_step, seeds, main_binary, output_dir)

    if args.resume:
        existing_prefixes = {
            f.replace("_vis.jpeg", "")
            for f in os.listdir(output_dir)
            if f.endswith("_vis.jpeg")
        }
        filtered = []
        for main_binary, dap, seed, output_dir in jobs:
            name = f"cowpea_dap{dap:03d}_seed{seed:02d}"
            if name not in existing_prefixes:
                filtered.append((main_binary, dap, seed, output_dir))
        jobs = filtered
        print(f"Resume mode: {len(jobs)} jobs remaining after skipping existing files")

    print(f"Generating {len(jobs)} Helios samples in {output_dir}")
    print(f"DAP range: {dap_start}..{dap_end} step {dap_step}, seeds per DAP: {seeds}")
    print(f"Parallel workers: {args.workers if args.workers else 'auto (CPU count)'}")

    successes = 0
    failures = []

    with multiprocessing.Pool(processes=args.workers) as pool:
        for dap, seed, success, msg in tqdm(
            pool.imap_unordered(generate_one, jobs),
            total=len(jobs),
            desc="Generating Helios samples"
        ):
            if success:
                successes += 1
            else:
                failures.append((dap, seed, msg[-200:]))

    print(f"\nDone. Successes: {successes}/{len(jobs)}")
    if failures:
        print(f"Failures: {len(failures)}")
        for dap, seed, msg in failures[:10]:
            print(f"  DAP {dap} seed {seed}: {msg.strip()}")


if __name__ == "__main__":
    main()
