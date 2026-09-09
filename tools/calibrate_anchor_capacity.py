"""Calibrates the DAP -> anchor capacity logistic curve from the cache dataset.

Samples GT phytomer cluster counts per DAP bucket (matcher-consistent: petiole
bases + standalone internodes, mirroring HierarchicalBotanicalMatcher), fits a
logistic curve to the rolling-max-smoothed p97.5 envelope, and prints constants
for ANCHOR_CURVE_{L,K,X0,N0} in diffusion_based/models/hierarchical_part_flow_matching.py.

Usage (from workspace root):
    /home/lion397/.conda/envs/digital-crops/bin/python tools/calibrate_anchor_capacity.py \
        --cache_dir dataset/cache/cowpea_curv26 --samples_per_dap 25
"""

import argparse
import glob
import math
import re
import statistics
from collections import defaultdict

import numpy as np
import torch


def phytomer_cluster_count(nodes: torch.Tensor, standalone_threshold: float = 0.8) -> int:
    """Counts GT phytomer clusters the same way HierarchicalBotanicalMatcher does."""
    ot = nodes[:, :13].argmax(-1)
    pos3 = nodes[:, 13:16]
    pet = ot == 4
    intern = ot == 3
    if pet.any():
        centers = pos3[pet]
        if intern.any():
            dmin = torch.cdist(pos3[intern], centers).min(dim=1).values
            standalone = torch.zeros_like(intern)
            standalone[intern] = dmin > standalone_threshold
            if standalone.any():
                centers = torch.cat([centers, pos3[torch.nonzero(standalone).squeeze(-1)]], dim=0)
    elif intern.any():
        centers = pos3[intern]
    else:
        act = ot > 0
        centers = pos3[act][:16] if act.any() else torch.zeros((0, 3))
    return centers.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--samples_per_dap", type=int, default=300)
    parser.add_argument("--samples_per_dap_young", type=int, default=100, help="Sample count for DAP < 40 buckets")
    parser.add_argument("--fit_target", type=str, default="max", choices=["max", "p975"],
                        help="Fit envelope: 'max' = rolling-max of observed sample maxima (100%% max coverage), 'p975' = rolling-max of p97.5")
    parser.add_argument("--margin", type=float, default=1.10)
    parser.add_argument("--flat", type=float, default=6.0)
    parser.add_argument("--max_anchors", type=int, default=512)
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.cache_dir}/*.pt"))
    by_dap = defaultdict(list)
    for f in files:
        m = re.search(r"dap(\d+)", f)
        if m:
            by_dap[int(m.group(1))].append(f)

    rng = np.random.RandomState(0)
    daps, means, p975s, maxs = [], [], [], []
    for d in sorted(by_dap.keys()):
        flist = by_dap[d]
        n_target = args.samples_per_dap if d >= 40 else args.samples_per_dap_young
        n = min(len(flist), n_target)
        picks = [flist[i] for i in rng.choice(len(flist), n, replace=False)]
        counts = []
        for f in picks:
            try:
                data = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not (isinstance(data, dict) and "nodes" in data):
                continue
            counts.append(phytomer_cluster_count(data["nodes"]))
        if counts:
            daps.append(d)
            means.append(statistics.mean(counts))
            p975s.append(float(np.quantile(counts, 0.975)))
            maxs.append(max(counts))
            print(f"DAP {d:3d}: mean={means[-1]:7.1f} p97.5={p975s[-1]:7.1f} max={maxs[-1]:6d} (n={len(counts)})")

    daps_a = np.array(daps, dtype=float)
    p975_a = np.array(p975s)
    max_a = np.array(maxs)
    mean_a = np.array(means)

    # Rolling-max smoothing (window +/- 4 DAP) to suppress sparse-bucket noise
    w = 4
    if args.fit_target == "max":
        target_a = np.array([max_a[max(0, i - w):i + w + 1].max() for i in range(len(daps_a))])
        fit_label = "rolling-max of observed sample maxima"
    else:
        target_a = np.array([p975_a[max(0, i - w):i + w + 1].max() for i in range(len(daps_a))])
        fit_label = "rolling-max of p97.5"

    def logistic(d, L, k, x0, n0):
        return L / (1.0 + np.exp(-k * (d - x0))) + n0

    from scipy.optimize import curve_fit
    popt, _ = curve_fit(
        logistic, daps_a, target_a,
        p0=[300.0, 0.15, 30.0, 2.0],
        bounds=([50.0, 0.01, 5.0, -20.0], [600.0, 1.0, 100.0, 20.0]),
        maxfev=200000,
    )
    L, k, x0, n0 = popt
    pred = logistic(daps_a, L, k, x0, n0)
    cap = np.minimum(np.ceil(np.maximum(pred, 0) * args.margin + args.flat), args.max_anchors)

    print(f"\n=== FITTED CONSTANTS (fit target: {fit_label}) ===")
    print(f"ANCHOR_CURVE_L   = {L:.1f}")
    print(f"ANCHOR_CURVE_K   = {k:.4f}")
    print(f"ANCHOR_CURVE_X0  = {x0:.1f}")
    print(f"ANCHOR_CURVE_N0  = {n0:.2f}")
    print(f"ANCHOR_MARGIN    = {args.margin}")
    print(f"ANCHOR_MARGIN_FLAT = {args.flat}")
    cov_p975 = float(np.mean(cap >= p975_a))
    cov_max = float(np.mean(cap >= max_a))
    print(f"coverage: p97.5 {cov_p975*100:.0f}% | sample-max {cov_max*100:.0f}%")
    print(f"avg capacity anchors: {cap.mean():.0f} (fine {cap.mean()*8:.0f} slots)")

    tiers = [8, 16, 32, 64, 128, 256, args.max_anchors]
    old_caps = []
    for d in daps_a:
        old = 8.0 * 2.0 ** (d / 8.5)
        tier = next((t for t in tiers if math.ceil(old) <= t), args.max_anchors)
        old_caps.append(tier)
    old_caps = np.array(old_caps)
    print(f"avg old-tier anchors: {old_caps.mean():.0f} (fine {old_caps.mean()*8:.0f} slots)")
    print(f"avg fine-slot saving vs old: {(1 - cap.mean() / old_caps.mean())*100:.0f}%")


if __name__ == "__main__":
    main()