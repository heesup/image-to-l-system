"""Silhouette metrics beyond IoU: is a high score a matching OUTLINE, or just a blob over the centre?

IoU compares filled areas, so a compact mass covering the plant's middle scores well while missing every
leaf gap and every arm of the canopy. Three metrics that do not:

  boundary F   precision/recall of the silhouette's BOUNDARY pixels within a tolerance (the DAVIS-style
               measure). Rewards getting the outline right, not the bulk.
  solidity     mask area / its convex hull area. A real canopy is full of gaps and concavities and scores
               LOW; a blob scores near 1. Compared as |pred - gt|, so matching the GT's own value is what
               counts, not being high or low.
  perimeter    boundary length relative to the GT's. A blob of the right area has a much shorter outline.
  hull ratio   the mask's convex hull area over the GT's. This is SPREAD: >1 means the prediction reaches
               further than the plant does, <1 means it is too compact. Distinct from solidity, which is
               each mask's own area/hull and says how gappy it is. The 2026-09-07/08 lineage scored 4.35x
               here (wildly over-spread) while the 2026-09-15/16 phytomer runs sat at 0.5-0.7x (too
               compact), so a value near 1.0 is what "the right size canopy" looks like.

Run on the renders saved by eval_test_time_refinement.py --save_renders (gt / before / after per plant).
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def mask_of(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (g > r * 1.02) & (g > b * 1.02)


def boundary(m):
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)


def boundary_f(pred, gt, tol=2):
    bp, bg = boundary(pred), boundary(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    dg = ndimage.distance_transform_edt(~bg)
    dp = ndimage.distance_transform_edt(~bp)
    prec = (dg[bp] <= tol).mean()
    rec = (dp[bg] <= tol).mean()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def hull_area(m):
    """Convex hull area of a mask in pixels (ConvexHull.volume is the area in 2D)."""
    if m.sum() < 8:
        return float("nan")
    from scipy.spatial import ConvexHull
    try:
        return float(ConvexHull(np.argwhere(m)).volume)
    except Exception:
        return float("nan")


def solidity(m):
    h = hull_area(m)
    if not np.isfinite(h):
        return float("nan")
    return float(m.sum() / max(h, 1e-6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--out", default="outputs/logs/20260917/silhouette_shape_metrics.json")
    a = ap.parse_args()

    rows = []
    for mp in sorted(glob.glob(os.path.join(a.renders, "*_meta.json"))):
        meta = json.load(open(mp))
        idx = meta["index"]
        paths = {k: os.path.join(a.renders, f"{idx}_{k}.png") for k in ("gt", "before", "after")}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        m = {k: mask_of(p) for k, p in paths.items()}
        gt = m["gt"]
        row = {"index": idx, "dap": meta["dap"], "sol_gt": solidity(gt),
               "per_gt": int(boundary(gt).sum()), "hull_gt": hull_area(gt)}
        for k in ("before", "after"):
            inter = (m[k] & gt).sum(); union = (m[k] | gt).sum()
            row[f"iou_{k}"] = float(inter / max(union, 1))
            row[f"bf_{k}"] = boundary_f(m[k], gt, a.tol)
            row[f"sol_{k}"] = solidity(m[k])
            row[f"per_{k}"] = int(boundary(m[k]).sum())
            row[f"hull_{k}"] = hull_area(m[k])
        rows.append(row)

    def agg(sel, name):
        if not sel:
            return
        f = lambda k: np.nanmean([r[k] for r in sel])
        print(f"{name:<12}{len(sel):>4}"
              f"{f('iou_before')*100:>9.1f}{f('iou_after')*100:>9.1f}"
              f"{f('bf_before')*100:>11.1f}{f('bf_after')*100:>11.1f}"
              f"{f('sol_gt'):>9.3f}{f('sol_before'):>10.3f}{f('sol_after'):>10.3f}"
              f"{f('per_before')/max(f('per_gt'),1):>11.2f}{f('per_after')/max(f('per_gt'),1):>10.2f}"
              f"{f('hull_before')/max(f('hull_gt'),1):>10.2f}{f('hull_after')/max(f('hull_gt'),1):>9.2f}")

    print(f"{'plants':<12}{'n':>4}{'IoU raw':>9}{'IoU ref':>9}{'bF raw':>11}{'bF ref':>11}"
          f"{'sol GT':>9}{'sol raw':>10}{'sol ref':>10}{'per raw':>11}{'per ref':>10}"
          f"{'hull raw':>10}{'hull ref':>9}")
    agg([r for r in rows if r["dap"] <= 15], "DAP<=15")
    agg([r for r in rows if 16 <= r["dap"] <= 45], "16-45")
    agg([r for r in rows if 46 <= r["dap"] <= 75], "46-75")
    agg([r for r in rows if r["dap"] > 75], ">75")
    agg(rows, "all")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"renders": a.renders, "tol": a.tol, "rows": rows}, open(a.out, "w"), indent=1)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
