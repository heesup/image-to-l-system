#!/usr/bin/env python
"""Before/after test-time refinement figure (print style): rows = plants; columns = Helios reference, GT render,
prediction as sampled, prediction after refinement against the input CHM. Reads the folder written by
eval_test_time_refinement.py --save_renders."""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="", help="comma-separated indices (row order); default: by DAP")
    a = ap.parse_args()
    metas = [json.load(open(f)) for f in glob.glob(os.path.join(a.root, "*_meta.json"))]
    if a.order:
        want = [int(x) for x in a.order.split(",")]; metas = sorted(metas, key=lambda m: want.index(m["index"]))
    else:
        metas = sorted(metas, key=lambda m: m["dap"])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 9, "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "text.color": "#111111"})
    cols = [("jpeg", "(a) Helios ray-traced input"), ("gt", "(b) Ground-truth render"), ("before", "(c) Prediction as sampled"), ("after", "(d) After test-time refinement")]
    fig, axes = plt.subplots(len(metas), 4, figsize=(10.4, 2.7 * len(metas)), facecolor="white"); axes = np.atleast_2d(axes)
    for r, m in enumerate(metas):
        for c, (tag, title) in enumerate(cols):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if tag == "jpeg":
                p = m.get("jpeg")
                if p and os.path.exists(p):
                    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
                    ax.imshow(im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((256, 256)))
                ax.set_ylabel(f"DAP {m['dap']}", fontsize=9)
            else:
                p = os.path.join(a.root, f"{m['index']}_{tag}.png")
                if os.path.exists(p): ax.imshow(Image.open(p))
                if tag == "before": ax.set_xlabel(f"IoU {m['iou_before']*100:.1f}%", fontsize=9)
                if tag == "after": ax.set_xlabel(f"IoU {m['iou_after']*100:.1f}%", fontsize=9)
            if r == 0: ax.set_title(title, fontsize=9.5)
    b = np.mean([m["iou_before"] for m in metas]) * 100; c_ = np.mean([m["iou_after"] for m in metas]) * 100
    fig.suptitle(f"Test-time refinement against the input canopy height map (40 Adam steps, no ground truth): mean IoU {b:.1f}% → {c_:.1f}%", fontsize=10, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white"); print("saved", a.out)


if __name__ == "__main__":
    main()
