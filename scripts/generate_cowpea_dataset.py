"""
Generate a large cowpea dataset with Helios radiation renders.

For each (dap, seed) in the requested range, runs the Helios `main` binary with
`--renderer radiation --focus-plant` and the camera/sun configuration that matches
the existing `dataset/helios_data` convention (camera height 1.0, azimuth 0,
sun elevation 45, sun azimuth 180). Each sample produces:
    <name>_0000_rad.jpeg   (radiation RGB render)
    <name>_0000_plant_0000.xml
    <name>_0000_masks.json
    <name>_0000_params.json

Supports --workers > 1 (each worker runs its own subprocess) and is resumable:
already-complete samples are skipped.
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


def _sample_name(dap: int, seed: int) -> str:
    return f"cowpea_dap{dap:03d}_seed{seed:02d}_caz000_h1.0_se045_saz180"


def _complete(output_dir: str, name: str) -> bool:
    return os.path.exists(os.path.join(output_dir, f"{name}_0000_rad.jpeg")) and \
        os.path.exists(os.path.join(output_dir, f"{name}_0000_plant_0000.xml"))


def render_one(args):
    dap, seed, output_dir, params_file, job_id = args
    name = _sample_name(dap, seed)
    if _complete(output_dir, name):
        return {"dap": dap, "seed": seed, "status": "skip", "elapsed": 0.0}

    # Each worker renders into its own temporary directory to avoid
    # filename collisions (camera_RGB_00000.jpeg etc.) between concurrent
    # subprocesses, then moves the completed artifacts into output_dir.
    tmp_dir = os.path.join(output_dir, f"_tmp_{job_id}")
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        MAIN_BIN,
        "--renderer", "radiation",
        "--save-xml",
        "--focus-plant",
        "-n", name,
        "--dap", str(dap),
        "-s", str(seed),
        "--output", tmp_dir,
        "-f", params_file,
    ]
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        status = "fail"
        if result.returncode == 0:
            rad = os.path.join(tmp_dir, f"{name}_0000_rad.jpeg")
            xml = os.path.join(tmp_dir, f"{name}_0000_plant_0000.xml")
            if os.path.exists(rad) and os.path.exists(xml):
                for suffix in ("_0000_rad.jpeg", "_0000_plant_0000.xml",
                               "_0000_masks.json", "_0000_camera.json", "_0000_params.json"):
                    src = os.path.join(tmp_dir, f"{name}{suffix}")
                    if os.path.exists(src):
                        shutil.move(src, os.path.join(output_dir, f"{name}{suffix}"))
                status = "ok"
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        if status == "ok":
            return {"dap": dap, "seed": seed, "status": "ok", "elapsed": elapsed}
        return {"dap": dap, "seed": seed, "status": "fail", "elapsed": elapsed,
                "stderr": (result.stderr or "")[-500:]}
    except Exception as e:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        return {"dap": dap, "seed": seed, "status": "error", "elapsed": time.time() - t0,
                "stderr": str(e)[-500:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dap-min", type=int, default=1)
    parser.add_argument("--dap-max", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds per dap (0..seeds-1)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "dataset", "helios_data"))
    parser.add_argument("--workers", type=int, default=2,
                        help="number of concurrent Helios subprocesses")
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.abspath(args.output_dir)
    assert os.path.exists(MAIN_BIN), f"main binary not found: {MAIN_BIN}"
    assert os.path.exists(BASE_PARAMS), f"params.json not found: {BASE_PARAMS}"

    # A single params.json (the base template) is shared by all runs; each `main`
    # invocation applies --dap/--seed overrides on top of it.
    params_file = BASE_PARAMS

    jobs = []
    for dap in range(args.dap_min, args.dap_max + 1):
        for seed in range(args.seeds):
            jobs.append((dap, seed, args.output_dir, params_file, len(jobs)))

    total = len(jobs)
    print(f"Total samples to generate: {total} ({args.dap_min}-{args.dap_max}, {args.seeds} seeds)")
    print(f"Output dir: {args.output_dir}")
    print(f"Workers: {args.workers}")

    done = 0
    ok = 0
    fail = 0
    t_start = time.time()
    stats = {"ok": 0, "fail": 0, "skip": 0, "error": 0}
    log_path = os.path.join(args.output_dir, "_generation_log.jsonl")

    pending = list(jobs)
    retries_left = {j: args.max_retries for j in jobs}
    # Map job->worker assignment: keep it simple, submit in batches.
    attempt = 0
    while pending:
        attempt += 1
        batch = pending[:args.workers]
        pending = pending[args.workers:]

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(render_one, j) for j in batch]
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                stats[res["status"]] = stats.get(res["status"], 0) + 1
                if res["status"] == "ok":
                    ok += 1
                    print(f"[{done}/{total}] dap{res['dap']:03d}_seed{res['seed']:02d} OK "
                          f"({res['elapsed']:.1f}s)", flush=True)
                elif res["status"] == "skip":
                    print(f"[{done}/{total}] dap{res['dap']:03d}_seed{res['seed']:02d} skip "
                          f"(exists)", flush=True)
                else:
                    fail += 1
                    print(f"[{done}/{total}] dap{res['dap']:03d}_seed{res['seed']:02d} "
                          f"{res['status']}: {res.get('stderr', '')[-200:]}", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"dap": res["dap"], "seed": res["seed"],
                                        "status": res["status"], "elapsed": res["elapsed"]}) + "\n")

        # Requeue failed jobs up to max_retries (check by file existence)
        new_pending = []
        for j in batch:
            dap, seed, od, pf, jid = j
            name = _sample_name(dap, seed)
            if not _complete(od, name):
                retries_left[j] -= 1
                if retries_left[j] >= 0:
                    new_pending.append(j)
        pending = new_pending + pending

        elapsed = time.time() - t_start
        rate = done / max(elapsed, 1e-6)
        remaining = len(pending)
        eta = remaining / max(rate, 1e-6)
        print(f"  progress: {done}/{total} done, {stats['ok']} ok, {stats['skip']} skip, "
              f"{fail} fail, ETA {eta/60:.1f} min", flush=True)

    print("\n" + "=" * 60)
    print(f"DATASET GENERATION COMPLETE: {stats}")
    print(f"Total wall time: {(time.time() - t_start)/60:.1f} min")
    print(f"Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()