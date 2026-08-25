"""
Multi-Species & Genotype/Phenotype Helios Dataset Generator.

Generates plant datasets for multiple species (cowpea, bean, sorghum, soybean, maize, etc.)
with subfolder separation and systematic genotype/phenotype variations.

Each sample produces:
    <output_dir>/<species>/<name>_0000_rad.jpeg   (radiation RGB render)
    <output_dir>/<species>/<name>_0000_plant_0000.xml
    <output_dir>/<species>/<name>_0000_masks.json
    <output_dir>/<species>/<name>_0000_camera.json
    <output_dir>/<species>/<name>_0000_params.json
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_DIR = os.path.join(REPO_ROOT, "Digital-Crops", "projects", "syntheticdata_generation", "build")
MAIN_BIN = os.path.join(BUILD_DIR, "main")
BASE_PARAMS = os.path.join(BUILD_DIR, "params.json")

# Default genotype archetypes per species family
GENOTYPE_PRESETS = {
    "cowpea": ["bush", "spreading", "vine", "random"],
    "bean": ["bush", "spreading", "vine", "random"],
    "soybean": ["bush", "spreading", "random"],
    "sorghum": ["dwarf", "tall", "random"],
    "maize": ["dwarf", "tall", "random"],
}


def _sample_name(plant_type: str, dap: int, seed: int, genotype: str = "") -> str:
    gt_suffix = f"_{genotype}" if genotype and genotype != "random" else ""
    return f"{plant_type}{gt_suffix}_dap{dap:03d}_seed{seed:02d}_caz000_h1.0_se045_saz180"


def _complete(target_dir: str, name: str) -> bool:
    return os.path.exists(os.path.join(target_dir, f"{name}_0000_rad.jpeg")) and \
        os.path.exists(os.path.join(target_dir, f"{name}_0000_plant_0000.xml"))


def render_one(job_args):
    plant_type, genotype, dap, seed, output_dir, params_file, job_id, overwrite, renderer = job_args
    species_dir = os.path.join(output_dir, plant_type)
    name = _sample_name(plant_type, dap, seed, genotype)
    
    if not overwrite and _complete(species_dir, name):
        return {"plant_type": plant_type, "genotype": genotype, "dap": dap, "seed": seed, "status": "skip", "elapsed": 0.0}

    # Isolated temporary rendering folder per worker
    tmp_dir = os.path.join(output_dir, f"_tmp_{os.getpid()}_{plant_type}_{dap}_{seed}")
    os.makedirs(tmp_dir, exist_ok=True)

    # Resolve species-specific config if available
    sp_cfg = os.path.join(REPO_ROOT, "Digital-Crops", "projects", "syntheticdata_generation", "configs", f"params_{plant_type}.json")
    if os.path.exists(sp_cfg):
        params_file = sp_cfg

    cmd = [
        MAIN_BIN,
        "--renderer", renderer,
        "--save-xml",
        "--focus-plant",
        "--ground-occlusion", "false",
        "--plant-type", plant_type,
        "-n", name,
        "--dap", str(dap),
        "-s", str(seed),
        "--output", tmp_dir,
        "-f", params_file,
    ]
    if genotype and genotype != "random":
        cmd.extend(["--genotype", genotype])

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        status = "fail"
        
        # main routes into tmp_dir/plant_type/
        gen_dir = os.path.join(tmp_dir, plant_type)
        if not os.path.exists(gen_dir):
            gen_dir = tmp_dir

        if result.returncode == 0:
            rad = os.path.join(gen_dir, f"{name}_0000_rad.jpeg")
            xml = os.path.join(gen_dir, f"{name}_0000_plant_0000.xml")
            if (renderer == "vis" or os.path.exists(rad)) and os.path.exists(xml):
                os.makedirs(species_dir, exist_ok=True)
                for suffix in ("_0000_rad.jpeg", "_0000_vis.jpeg", "_0000_plant_0000.xml",
                               "_0000_masks.json", "_0000_camera.json", "_0000_params.json", "_0000_boxes.txt"):
                    src = os.path.join(gen_dir, f"{name}{suffix}")
                    if os.path.exists(src):
                        shutil.move(src, os.path.join(species_dir, f"{name}{suffix}"))
                status = "ok"

        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        if status == "ok":
            return {"plant_type": plant_type, "genotype": genotype, "dap": dap, "seed": seed, "status": "ok", "elapsed": elapsed}
        return {"plant_type": plant_type, "genotype": genotype, "dap": dap, "seed": seed, "status": "fail", "elapsed": elapsed,
                "stderr": (result.stderr or "")[-500:]}
    except Exception as e:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        return {"plant_type": plant_type, "genotype": genotype, "dap": dap, "seed": seed, "status": "error", "elapsed": time.time() - t0,
                "stderr": str(e)[-500:]}


def main():
    parser = argparse.ArgumentParser(description="Generate multi-species plant dataset with Helios radiation renderer.")
    parser.add_argument("--plant-types", "--species", type=str, default="cowpea",
                        help="Comma-separated plant types (e.g. cowpea,bean,sorghum)")
    parser.add_argument("--genotypes", type=str, default="random",
                        help="Comma-separated genotypes (e.g. bush,spreading,vine,dwarf,tall,random) or 'all'")
    parser.add_argument("--dap-min", type=int, default=1)
    parser.add_argument("--dap-max", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds per dap (0..seeds-1)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "dataset", "helios_data"))
    parser.add_argument("--renderer", type=str, default="radiation", choices=["radiation", "vis", "all"])
    parser.add_argument("--workers", type=int, default=2,
                        help="number of concurrent Helios subprocesses")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing generated samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.abspath(args.output_dir)
    assert os.path.exists(MAIN_BIN), f"main binary not found: {MAIN_BIN}"
    assert os.path.exists(BASE_PARAMS), f"params.json not found: {BASE_PARAMS}"

    plant_types = [p.strip().lower() for p in args.plant_types.split(",") if p.strip()]
    
    jobs = []
    for plant_type in plant_types:
        if args.genotypes == "all":
            gts = GENOTYPE_PRESETS.get(plant_type, ["random"])
        else:
            gts = [g.strip().lower() for g in args.genotypes.split(",") if g.strip()]

        for gt in gts:
            for dap in range(args.dap_min, args.dap_max + 1):
                for seed in range(args.seeds):
                    jobs.append((plant_type, gt, dap, seed, args.output_dir, BASE_PARAMS, len(jobs), args.overwrite, args.renderer))

    total = len(jobs)
    print("=" * 65)
    print(f"HELIOS MULTI-SPECIES DATASET GENERATOR")
    print("=" * 65)
    print(f"Plant Types:   {plant_types}")
    print(f"Genotypes:     {args.genotypes}")
    print(f"DAP Range:     {args.dap_min} to {args.dap_max}")
    print(f"Seeds / DAP:   {args.seeds}")
    print(f"Total Samples: {total}")
    print(f"Output Root:   {args.output_dir}")
    print(f"Renderer:      {args.renderer}")
    print(f"Workers:       {args.workers}")
    print("=" * 65)

    done = 0
    stats = {"ok": 0, "fail": 0, "skip": 0, "error": 0}
    t_start = time.time()

    pending = list(jobs)
    retries_left = {j: args.max_retries for j in jobs}
    active_futures = {}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        while pending and len(active_futures) < args.workers * 2:
            job = pending.pop(0)
            fut = ex.submit(render_one, job)
            active_futures[fut] = job

        while active_futures:
            done_futs = []
            for fut in as_completed(active_futures):
                done_futs.append(fut)
                break

            for fut in done_futs:
                job = active_futures.pop(fut)
                res = fut.result()
                done += 1
                status = res["status"]
                stats[status] = stats.get(status, 0) + 1

                species_dir = os.path.join(args.output_dir, res["plant_type"])
                log_path = os.path.join(species_dir, "_generation_log.jsonl")
                os.makedirs(species_dir, exist_ok=True)

                if status == "ok":
                    print(f"[{done}/{total}] {res['plant_type'].upper()}:{res['genotype']} dap{res['dap']:03d}_seed{res['seed']:02d} OK "
                          f"({res['elapsed']:.1f}s)", flush=True)
                elif status == "skip":
                    print(f"[{done}/{total}] {res['plant_type'].upper()}:{res['genotype']} dap{res['dap']:03d}_seed{res['seed']:02d} skip", flush=True)
                else:
                    print(f"[{done}/{total}] {res['plant_type'].upper()}:{res['genotype']} dap{res['dap']:03d}_seed{res['seed']:02d} "
                          f"{status}: {res.get('stderr', '')[-200:]}", flush=True)
                    if retries_left[job] > 0:
                        retries_left[job] -= 1
                        pending.append(job)

                with open(log_path, "a") as f:
                    f.write(json.dumps({"plant_type": res["plant_type"], "genotype": res["genotype"],
                                        "dap": res["dap"], "seed": res["seed"],
                                        "status": status, "elapsed": res["elapsed"]}) + "\n")

                while pending and len(active_futures) < args.workers * 2:
                    next_job = pending.pop(0)
                    new_fut = ex.submit(render_one, next_job)
                    active_futures[new_fut] = next_job

                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-6)
                remaining = len(pending) + len(active_futures)
                eta = remaining / max(rate, 1e-6)
                if done % 10 == 0 or not active_futures:
                    print(f"  Progress: {done}/{total} done, {stats['ok']} ok, {stats['skip']} skip, "
                          f"{stats['fail'] + stats['error']} fail, ETA {eta/60:.1f} min", flush=True)

    print("\n" + "=" * 65)
    print(f"DATASET GENERATION COMPLETE: {stats}")
    print(f"Total Wall Time: {(time.time() - t_start)/60:.1f} min")
    print(f"Output: {args.output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
