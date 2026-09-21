"""Figure: the rasterised render barely separates four architectures that raytraced separates by 22 points."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
    "axes.linewidth": 0.6, "axes.edgecolor": "#444444", "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

L = "outputs/logs/20260919"
BLUE, ORANGE, GREY = "#1f4e79", "#d1761c", "#888888"

def m(ns):
    return float(np.mean([100 * np.mean([x["iou_after"] for x in json.load(open(f"{L}/{n}.json"))["rows"]])
                          for n in ns]))

rows = [
    ("relative\nno aug", ["s_base", "rep_base1", "rep_base2", "rep_base3"], ["base_hel1", "base_hel2"]),
    ("relative\n+ aug", ["v10aug_flat1", "v10aug_flat2"], ["v10aug_hel1", "v10aug_hel2"]),
    ("absolute\nno aug", ["mg_def1", "mg_def2", "mg_def3", "mg_def4", "mg_def5"], ["ab_hel1b", "ab_hel2b"]),
    ("absolute\n+ aug", ["ag_def1", "ag_def2", "ag_def3"], ["ag_hel1b", "ag_hel2b"]),
]
lab = [r[0] for r in rows]
ras = np.array([m(r[1]) for r in rows])
ray = np.array([m(r[2]) for r in rows])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})

x = np.arange(len(rows)); w = 0.38
ax1.bar(x - w/2, ras, w, color=GREY, edgecolor="none", label="rasterised", zorder=3)
ax1.bar(x + w/2, ray, w, color=BLUE, edgecolor="none", label="raytraced", zorder=3)
for k in range(len(rows)):
    ax1.plot([x[k] + w/2, x[k] + w/2], [ray[k], ras[k]], color=ORANGE, lw=1.2, zorder=4)
    ax1.text(x[k] + w/2 + 0.06, (ray[k] + ras[k]) / 2, f"{ray[k]-ras[k]:+.1f}",
             fontsize=7, color=ORANGE, va="center")
ax1.set_xticks(x); ax1.set_xticklabels(lab)
ax1.set_ylabel("refined silhouette IoU (%)"); ax1.set_ylim(0, 100)
ax1.legend(loc="lower left", frameon=True, framealpha=1, edgecolor="#bbbbbb", fancybox=False)
ax1.set_title("(a) the appearance gap per architecture", loc="left", fontsize=8.5)

for k, (v, col, nm) in enumerate(((ras, GREY, "rasterised"), (ray, BLUE, "raytraced"))):
    ax2.scatter(np.full(len(v), k), v, s=30, color=col, zorder=3, clip_on=False)
    ax2.vlines(k, v.min(), v.max(), color=col, lw=1.4, zorder=2)
    dx, ha = (-0.16, "right") if k == 0 else (0.16, "left")
    ax2.text(k + dx, (v.min() + v.max()) / 2, f"spread\n{v.max()-v.min():.1f} pts",
             fontsize=7.5, color=col, va="center", ha=ha)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["rasterised", "raytraced"])
ax2.set_xlim(-0.75, 1.6); ax2.set_ylim(40, 80)
ax2.set_ylabel("refined IoU of the 4 models (%)")
ax2.set_title("(b) what each protocol can tell apart", loc="left", fontsize=8.5)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd"); ax.set_axisbelow(True)
fig.tight_layout()
out = "docs/experiments/20260919-rasterised-renders-do-not-discriminate/assets/rasterised_vs_raytraced.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
