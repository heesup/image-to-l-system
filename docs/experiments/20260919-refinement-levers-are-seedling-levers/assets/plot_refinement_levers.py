"""Figure: the refinement noise floor, and which levers clear it.

(a) four replicates each of the default and --reg_latent 0: the run-to-run spread of the 20-plant
    mean, and the gap between configs measured against it
(b) every config in the sweep against the replicated default mean, with the +-2 SD noise band that
    a single-run difference has to clear to mean anything
(c) the seedling prior ladder re-measured on 24 plants after the n=4 version failed to replicate
"""
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
BLUE, ORANGE, GREY = "#1f4e79", "#d1761c", "#777777"

def mean_iou(n):
    return 100 * np.mean([x["iou_after"] for x in json.load(open(f"{L}/{n}.json"))["rows"]])

base = np.array([mean_iou(n) for n in ("s_base", "rep_base1", "rep_base2", "rep_base3")])
pr0 = np.array([mean_iou(n) for n in ("s_pr0", "rep_pr01", "rep_pr02", "rep_pr03")])
sd = np.sqrt((base.var(ddof=1) + pr0.var(ddof=1)) / 2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.3, 2.75),
                                    gridspec_kw={"width_ratios": [1, 1.5, 1]})

# (a) replicates
for k, (v, col, lab) in enumerate(((base, BLUE, "default\n(reg_latent 0.5)"),
                                   (pr0, ORANGE, "reg_latent 0"))):
    ax1.scatter(np.full(len(v), k) + np.linspace(-.07, .07, len(v)), v, s=22, color=col,
                zorder=3, clip_on=False)
    ax1.hlines(v.mean(), k - .22, k + .22, color=col, lw=1.6, zorder=4)
    ax1.text(k, v.mean() + 0.55, f"{v.mean():.2f}", ha="center", fontsize=7.5, color=col)
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["default\n(0.5)", "reg_latent 0"])
ax1.set_xlim(-.45, 1.45); ax1.set_ylabel("refined IoU, 20-plant mean (%)")
ax1.set_title(f"(a) 4 runs each\nnoise SD {sd:.2f}", loc="left", fontsize=8.5)

# (b) every config vs the replicated default, with the noise band
cfgs = [("8 starts", 74.3), ("prior 0 + 8 starts", 74.6), ("prior 0", pr0.mean()),
        ("no priors", 72.5), ("prior 0.25", 71.9), ("prior 2.0", 70.2),
        ("384 px", 69.8), ("256 px", 69.2)]
lab = [c[0] for c in cfgs]
val = np.array([c[1] for c in cfgs]) - base.mean()
y = np.arange(len(cfgs))[::-1]
cols = [ORANGE if abs(v) > 2 * sd else GREY for v in val]
ax2.barh(y, val, 0.62, color=cols, edgecolor="none", zorder=3)
ax2.axvspan(-2 * sd, 2 * sd, color=GREY, alpha=0.20, zorder=1,
            label=f"$\\pm$2 SD noise ({2*sd:.1f} pts)")
ax2.axvline(0, color="#444444", lw=0.7, zorder=2)
ax2.set_yticks(y); ax2.set_yticklabels(lab)
ax2.set_xlabel("change vs replicated default (points)")
ax2.legend(loc="upper left", frameon=True, framealpha=1, edgecolor="#bbbbbb", fancybox=False)
ax2.set_title("(b) orange clears the noise floor, grey does not", loc="left", fontsize=8.5)

# (c) seedling ladder, n=24
sdl = [("2.0", "sd_pr2"), ("0.5", "sd_pr05"), ("0", "sd_pr0")]
sv = [mean_iou(n) for _, n in sdl]
ax3.plot(range(3), sv, "o-", color=ORANGE, lw=1.4, ms=5)
for k, v in enumerate(sv):
    ax3.text(k, v + 1.6, f"{v:.1f}", ha="center", fontsize=7.5, color=ORANGE)
ax3.set_xticks(range(3)); ax3.set_xticklabels([s for s, _ in sdl])
ax3.set_xlim(-.35, 2.35); ax3.set_ylim(0, 50)
ax3.set_xlabel("latent prior weight")
ax3.set_ylabel("refined IoU (%)")
ax3.set_title("(c) 24 seedlings\n(n=4 gave +18, not +5.8)", loc="left", fontsize=8.5)

for ax in (ax1, ax2, ax3):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
ax1.grid(axis="y", lw=0.4, color="#dddddd")
ax2.grid(axis="x", lw=0.4, color="#dddddd")
ax3.grid(axis="y", lw=0.4, color="#dddddd")

fig.tight_layout()
out = "docs/experiments/20260919-refinement-levers-are-seedling-levers/assets/refinement_levers.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
