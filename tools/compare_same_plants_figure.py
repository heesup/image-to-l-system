#!/usr/bin/env python
"""Same-plant comparison grid (academic print style): rows = plants, columns = reference image, GT render and one
column per model, each cell labelled with its strict-protocol IoU. Reads the per-plant folders written by the
render scripts (meta.json + <tag>.png per model)."""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with one sub-folder per plant index")
    ap.add_argument("--indices", required=True, help="comma-separated plant indices (row order)")
    ap.add_argument("--models", required=True, help="comma-separated tag=title pairs, e.g. optionB_ep125=Option B (9/8)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    idxs = [int(x) for x in a.indices.split(",")]
    models = [m.split("=", 1) for m in a.models.split(",")]
    cols = [("jpeg", "(a) Helios ray-traced reference"), ("gt", "(b) Ground-truth render")] + [
        (tag, f"({chr(ord('c') + k)}) {title}") for k, (tag, title) in enumerate(models)]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 9, "axes.edgecolor": "#444444", "axes.linewidth": 0.6, "text.color": "#111111"})
    fig, axes = plt.subplots(len(idxs), len(cols), figsize=(2.6 * len(cols), 2.7 * len(idxs)), facecolor="white")
    axes = np.atleast_2d(axes)
    means = {tag: [] for tag, _ in models}
    for r, i in enumerate(idxs):
        d = os.path.join(a.root, str(i)); meta = json.load(open(os.path.join(d, "meta.json")))
        for c, (tag, title) in enumerate(cols):
            ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if tag == "jpeg":
                p = meta.get("jpeg")
                if p and os.path.exists(p):
                    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
                    ax.imshow(im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((256, 256)))
                ax.set_ylabel(f"DAP {meta['dap']}\n{meta['n_organs']} organs", fontsize=9)
            else:
                p = os.path.join(d, tag + ".png")
                if os.path.exists(p):
                    ax.imshow(Image.open(p))
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", color="#666666")
                if tag in meta.get("iou", {}):
                    iou = meta["iou"][tag]; means[tag].append(iou)
                    ax.set_xlabel(f"IoU {iou*100:.1f}%", fontsize=9)
            if r == 0:
                ax.set_title(title, fontsize=9.5)
    sub = ", ".join(f"{title}: {np.mean(means[tag])*100:.1f}%" for tag, title in models if means[tag])
    fig.suptitle((a.title + "\n" if a.title else "") + f"Mean silhouette IoU over {len(idxs)} plants — {sub}", fontsize=10.5, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white")
    print("saved", a.out, "|", sub)


if __name__ == "__main__":
    main()
