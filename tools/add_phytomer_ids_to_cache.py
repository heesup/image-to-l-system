"""
Adds XML phytomer membership (shoot_id, phytomer_idx) to every cache sample.

The 26D FM cache drops the typed 40D topology columns, so packet builders had
to re-cluster organs by nearest-center — which mis-assigns ~77% of mature-plant
leaflets to the wrong phytomer. This script re-reads each sample's XML, extracts
the per-organ (shoot_id, phytomer_idx) membership (row order is preserved:
cache nodes[i] == 40D row i), and stores it as `phytomer_ids` (N, 2) int64.

Usage:
    python tools/add_phytomer_ids_to_cache.py [--cache-dir ...] [--workers 8]
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    T_COL_SHOOT_ID,
    T_COL_PHYTOMER_IDX,
    T_COL_ORGAN_TYPE,
    ORGAN_ROOT_META,
    ORGAN_SHOOT_META,
)


def process_one(path: str) -> tuple:
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        if "phytomer_ids" in d:
            return path, "skip"
        xml = d.get("xml")
        if not xml or not os.path.exists(xml):
            return path, "no-xml"
        arr = PlantOrganArray.from_xml_file(xml)
        t = arr.tensor  # (N, 40), row order == cache nodes order
        n = int(d["num_organs"].item())
        ids = torch.zeros(n, 2, dtype=torch.int64)
        for i in range(n):
            ot = int(t[i, T_COL_ORGAN_TYPE].item())
            if ot in (ORGAN_ROOT_META, ORGAN_SHOOT_META):
                ids[i, 0] = -1
                ids[i, 1] = -1
            else:
                ids[i, 0] = int(t[i, T_COL_SHOOT_ID].item())
                ids[i, 1] = int(t[i, T_COL_PHYTOMER_IDX].item())
        d["phytomer_ids"] = ids
        torch.save(d, path)
        return path, "ok"
    except Exception as e:
        return path, f"err:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
    print(f"{len(files)} cache files")
    t0 = time.time()
    ok = skip = noxml = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, f) for f in files]
        for i, fut in enumerate(as_completed(futs)):
            path, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status == "no-xml":
                noxml += 1
            else:
                err += 1
            if (i + 1) % 2000 == 0:
                print(f"  [{i+1}/{len(files)}] ok={ok} skip={skip} noxml={noxml} err={err} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"Done: ok={ok} skip={skip} noxml={noxml} err={err} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
