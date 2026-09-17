"""Compare two eval_test_time_refinement.py result files that differ only in the RGB source of the input
(flat cache render vs Helios raytraced re-render, see render_helios_eval_crops.py): the appearance-gap
readout of docs/ongoing/20260916_sim_to_real_assessment_and_plan.md §4.

Prints and writes (<out_dir>/compare.md) the per-plant table and DAP-bucket means of raw strict P,
refined P, active nodes and the DAP probe's error, and draws a paired per-plant figure
(<out_dir>/<fig>.png, 300 dpi, white academic style): (a) raw P, (b) refined P, (c) predicted vs true DAP.
"""
import argparse
import json
import os

import numpy as np

SERIES = {"flat": ("#2a78d6", "flat cache render"), "helios": ("#eb6834", "Helios raytraced render")}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BUCKETS = [("DAP <= 15", lambda d: d <= 15), ("16-45", lambda d: 16 <= d <= 45),
           ("46-75", lambda d: 46 <= d <= 75), ("> 75", lambda d: d > 75), ("all", lambda d: True)]


def load(path):
    rows = json.load(open(path))["rows"]
    return {r["index"]: r for r in rows}


def bucket_table(flat, hel):
    idx = sorted(set(flat) & set(hel), key=lambda i: flat[i]["dap"])
    dap = np.array([flat[i]["dap"] for i in idx], dtype=float)
    cols = {}
    for name, rows in (("flat", flat), ("helios", hel)):
        cols[name] = {
            "raw": np.array([rows[i]["iou_before"] * 100 for i in idx]),
            "ref": np.array([rows[i]["iou_after"] * 100 for i in idx]),
            "nodes": np.array([rows[i]["n_nodes_before"] for i in idx], dtype=float),
            "dap_err": np.array([abs(rows[i].get("pred_dap", np.nan) - rows[i]["dap"]) for i in idx]),
        }
    gt_nodes = np.array([flat[i].get("n_phytomers_gt", np.nan) for i in idx], dtype=float)
    lines = ["| plants | n | raw P flat | raw P Helios | refined flat | refined Helios | nodes flat / Helios / GT | DAP error flat / Helios |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    for name, pred in BUCKETS:
        m = np.array([pred(d) for d in dap])
        if m.sum() == 0:
            continue
        f, h = cols["flat"], cols["helios"]
        lines.append(f"| {name} | {int(m.sum())} | {f['raw'][m].mean():.1f} | {h['raw'][m].mean():.1f} | "
                     f"{f['ref'][m].mean():.1f} | {h['ref'][m].mean():.1f} | "
                     f"{f['nodes'][m].mean():.1f} / {h['nodes'][m].mean():.1f} / {np.nanmean(gt_nodes[m]):.1f} | "
                     f"{np.nanmean(f['dap_err'][m]):.1f} / {np.nanmean(h['dap_err'][m]):.1f} |")
    per = ["", "| index | DAP | raw flat | raw Helios | refined flat | refined Helios | nodes flat / Helios / GT | pred DAP flat / Helios |",
           "|---:|---:|---:|---:|---:|---:|---|---|"]
    for k, i in enumerate(idx):
        f, h = flat[i], hel[i]
        per.append(f"| {i} | {f['dap']} | {f['iou_before'] * 100:.1f} | {h['iou_before'] * 100:.1f} | "
                   f"{f['iou_after'] * 100:.1f} | {h['iou_after'] * 100:.1f} | "
                   f"{f['n_nodes_before']} / {h['n_nodes_before']} / {int(gt_nodes[k]) if not np.isnan(gt_nodes[k]) else '-'} | "
                   f"{f.get('pred_dap', float('nan')):.1f} / {h.get('pred_dap', float('nan')):.1f} |")
    return idx, dap, cols, "\n".join(lines + per)


def _style_axis(ax):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color(AXIS)
    ax.tick_params(width=0.5, color=AXIS, labelcolor=INK2, labelsize=7, length=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def figure(idx, dap, cols, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.5), gridspec_kw={"width_ratios": [1.25, 1.25, 1.0]})
    x = np.arange(len(idx))
    for ax, key, title in zip(axes[:2], ("raw", "ref"), ("(a) Strict P before refinement", "(b) Strict P after refinement")):
        _style_axis(ax)
        f, h = cols["flat"][key], cols["helios"][key]
        ax.vlines(x, np.minimum(f, h), np.maximum(f, h), color=AXIS, linewidth=0.8, zorder=1)
        for name, y in (("flat", f), ("helios", h)):
            ax.plot(x, y, linestyle="none", marker="o", markersize=5, markerfacecolor=SERIES[name][0],
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(d)}" for d in dap], rotation=90, fontsize=6)
        ax.set_xlabel("plant (labelled by DAP)", color=INK2)
        ax.set_ylabel("silhouette IoU (%)", color=INK2)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.text(0.02, 0.96, f"mean flat {f.mean():.1f}, Helios {h.mean():.1f}", transform=ax.transAxes,
                fontsize=7, color=INK, va="top")
    ax = axes[2]
    _style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    lim = max(100.0, float(np.nanmax([np.nanmax(c["pred"]) for c in cols.values()])), float(dap.max())) + 5.0
    ax.plot([0, lim], [0, lim], color=AXIS, linewidth=0.8, zorder=1)
    for name in ("flat", "helios"):
        ax.plot(dap, cols[name]["pred"], linestyle="none", marker="o", markersize=5, markerfacecolor=SERIES[name][0],
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("true DAP", color=INK2)
    ax.set_ylabel("predicted DAP (Stage 1 probe)", color=INK2)
    ax.set_title("(c) DAP probe", fontsize=8, color=INK, loc="left")
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=5, markerfacecolor=SERIES[n][0],
                      markeredgecolor="white", label=SERIES[n][1]) for n in ("flat", "helios")]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=7, frameon=True,
                   edgecolor=GRID, framealpha=1.0, borderpad=0.4, handletextpad=0.3)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--helios", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig", default="appearance_gap_flat_vs_helios")
    a = ap.parse_args()
    flat, hel = load(a.flat), load(a.helios)
    idx, dap, cols, table = bucket_table(flat, hel)
    for name, rows in (("flat", flat), ("helios", hel)):
        cols[name]["pred"] = np.array([rows[i].get("pred_dap", np.nan) for i in idx], dtype=float)
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, "compare.md"), "w").write(table + "\n")
    print(table)
    fig_path = os.path.join(a.out_dir, a.fig + ".png")
    figure(idx, dap, cols, fig_path)
    print(f"figure -> {fig_path}")


if __name__ == "__main__":
    main()
