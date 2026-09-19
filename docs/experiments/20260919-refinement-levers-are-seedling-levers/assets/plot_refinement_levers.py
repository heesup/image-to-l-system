"""Figure: every refinement lever in the 2026-09-19 sweep is a seedling lever.

(a) the reg_latent ladder, seedlings vs mature, showing the monotone prior effect
(b) all nine runs as DAP<=15 vs DAP>15, showing mature plants are flat
"""
import json, os
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

LOG = "outputs/logs/20260919"
BLUE, ORANGE = "#1f4e79", "#d1761c"

def band(name):
    rows = json.load(open(f"{LOG}/{name}.json"))["rows"]
    lo = [r for r in rows if r["dap"] <= 15]
    hi = [r for r in rows if r["dap"] > 15]
    f = lambda s, k: 100 * np.mean([r[k] for r in s])
    return f(lo, "iou_after"), f(hi, "iou_after")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.9))

# (a) prior ladder
priors = [0.0, 0.25, 0.5, 2.0]
runs = ["s_pr0", "s_pr025", "s_base", "s_pr2"]
lo = [band(r)[0] for r in runs]
hi = [band(r)[1] for r in runs]
x = np.arange(len(priors))
ax1.plot(x, lo, "o-", color=ORANGE, lw=1.4, ms=5, label="seedlings (DAP $\\leq$ 15, n=4)")
ax1.plot(x, hi, "s-", color=BLUE, lw=1.4, ms=4.5, label="mature (DAP > 15, n=16)")
ax1.set_xticks(x); ax1.set_xticklabels([f"{p:g}" for p in priors])
ax1.set_xlabel("latent prior weight  (--reg_latent)")
ax1.set_ylabel("refined silhouette IoU (%)")
ax1.set_ylim(0, 100)
ax1.annotate("default", xy=(2, lo[2]), xytext=(2, lo[2] + 13), ha="center", color="#444444",
             arrowprops=dict(arrowstyle="->", lw=0.6, color="#444444"))
ax1.legend(loc="lower left", frameon=True, framealpha=1, edgecolor="#bbbbbb", fancybox=False)
ax1.set_title("(a) stronger prior, worse seedlings", loc="left", fontsize=8.5)

# (b) all runs
allruns = [("s_base", "default"), ("s_res256", "256 px"), ("s_res384", "384 px"),
           ("s_pr0", "prior 0"), ("s_pr025", "prior 0.25"), ("s_pr2", "prior 2.0"),
           ("s_noreg", "no priors"), ("s_n8", "8 starts"), ("s_pr0_n8", "prior 0\n+ 8 starts")]
lo2 = [band(r)[0] for r, _ in allruns]
hi2 = [band(r)[1] for r, _ in allruns]
x2 = np.arange(len(allruns))
w = 0.38
ax2.bar(x2 - w/2, lo2, w, color=ORANGE, edgecolor="none", label="seedlings (DAP $\\leq$ 15)")
ax2.bar(x2 + w/2, hi2, w, color=BLUE, edgecolor="none", label="mature (DAP > 15)")
ax2.set_xticks(x2); ax2.set_xticklabels([l for _, l in allruns], rotation=45, ha="right")
ax2.set_ylabel("refined silhouette IoU (%)")
ax2.set_ylim(0, 100)
ax2.axhspan(min(hi2), max(hi2), color=BLUE, alpha=0.10, zorder=0)
ax2.legend(loc="upper center", ncol=2, frameon=True, framealpha=1, edgecolor="#bbbbbb",
           fancybox=False, bbox_to_anchor=(0.5, 1.0))
ax2.set_title("(b) mature plants are flat; all variance is in seedlings", loc="left", fontsize=8.5)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

fig.tight_layout()
out = "docs/experiments/20260919-refinement-levers-are-seedling-levers/refinement_levers.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
