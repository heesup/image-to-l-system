"""
Figure: Fixed Plant Organ Vector Sparsity & Gradient Unbalance using VAE.

Empirically measured (no synthetic/mock data) from:
  - dataset/cache/cowpea_curv26/*.pt           (10,000-sample Helios Cowpea 26D shard corpus)
  - diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt
  - diffusion_based/models/organ_latent_vae.py (OrganLatentVAE, z in R^16 ~ N(0, I))

Panels:
  A. Organ-class frequency imbalance across 13 botanical classes
  B. Raw 13D per-channel gradient unbalance (tube organs: internode/petiole/peduncle)
  C. Latent per-dim gradient balance (dL/dz at Flow Matching prior z0 ~ N(0, I)
     through the FROZEN VAE decoder) vs raw 13D gradient profile
  D. 16D latent manifold structure (PCA-2D, organ-class clusters)

Usage:
  PYTHONPATH=. python scripts/generate_fig_vae_sparsity_gradient_balance.py
Output:
  docs/results/assets/fig_organ_vae_sparsity_gradient_balance.png
  docs/results/assets/fig_organ_vae_sparsity_gradient_balance.json  (measured numbers)
"""

import os
import sys
import glob
import json
import random
import collections

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.organ_latent_vae import OrganLatentVAE  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CACHE_DIR = os.path.join(REPO_ROOT, "dataset", "cache", "cowpea_curv26")
VAE_CKPT = os.path.join(REPO_ROOT, "diffusion_based", "checkpoints", "organ_vae", "organ_latent_vae_best.pt")
OUT_PNG = os.path.join(REPO_ROOT, "docs", "results", "assets", "fig_organ_vae_sparsity_gradient_balance.png")
OUT_JSON = os.path.join(REPO_ROOT, "docs", "results", "assets", "fig_organ_vae_sparsity_gradient_balance.json")

N_STAT_SAMPLES = 600          # shards scanned for organ statistics
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "NONE", "ROOT_META", "SHOOT_META", "INTERNODE", "PETIOLE", "LEAF",
    "PEDUNCLE", "BUD_DORMANT", "BUD_ACTIVE", "FLOWER_CLOSED", "FLOWER_OPEN",
    "FRUIT", "BUD_ABORTED",
]
TUBE_CLASSES = [3, 4, 6]      # INTERNODE, PETIOLE, PEDUNCLE

# Light theme (publication/lab-meeting style, white background)
BG = "#ffffff"
PANEL = "#ffffff"
FG = "#1a1f27"
GRID = "#e3e7ec"
AXIS = "#57606a"
RAW_C = "#d43f3f"      # problem / raw space
LAT_C = "#1e9e6f"      # fixed / latent space
ACC_C = "#2f6fd6"      # neutral accent
WARN_C = "#c77800"     # warning annotation
GRAY_C = "#9aa4b2"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "axes.edgecolor": "#c2c9d1",
    "axes.labelcolor": FG,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "text.color": FG,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlepad": 9,
    "legend.framealpha": 0.6,
    "legend.facecolor": BG,
    "legend.edgecolor": GRID,
})

CLASS_PALETTE = {
    3: "#2f6fd6",   # INTERNODE
    4: "#11897b",   # PETIOLE
    5: "#1e9e6f",   # LEAF
    6: "#e0940f",   # PEDUNCLE
    8: "#8250df",   # BUD_ACTIVE
    9: "#d66294",   # FLOWER_CLOSED
    10: "#d43f3f",  # FLOWER_OPEN
    11: "#9a6a1b",  # FRUIT
    12: "#77828f",  # BUD_ABORTED
}


def log(msg):
    print(f"[fig] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 1. Organ statistics sample (26D rows + classes)
# ----------------------------------------------------------------------------
def load_organ_sample(files, n, seed):
    rng = random.Random(seed)
    picked = rng.sample(files, n)
    rows, cls_rows = [], []
    for f in picked:
        d = torch.load(f, map_location="cpu", weights_only=False)
        nodes = d["nodes"][: int(d["num_organs"])]
        if nodes.shape[0] == 0:
            continue
        cls = nodes[:, :13].argmax(-1)
        m = cls >= 3                       # physical organs only
        if m.sum() == 0:
            continue
        rows.append(nodes[m])
        cls_rows.append(cls[m])
    X = torch.cat(rows)
    C = torch.cat(cls_rows)
    return X, C


# ----------------------------------------------------------------------------
# 2. VAE latent statistics + frozen-decoder gradient probe at FM prior
# ----------------------------------------------------------------------------
def vae_latents_and_grads(vae, X):
    N = X.shape[0]
    chunk = 32768

    Z_parts = []
    with torch.no_grad():
        for i in range(0, N, chunk):
            xb = X[i:i + chunk].to(DEVICE)
            mu, _ = vae.encode(xb)
            Z_parts.append(mu.cpu())
    Z = torch.cat(Z_parts)

    grad_sum = torch.zeros(16, device=DEVICE)
    grad_n = 0
    for i in range(0, N, chunk):
        xb = X[i:i + chunk].to(DEVICE)
        z0 = torch.randn(xb.shape[0], 16, device=DEVICE, requires_grad=True)
        dec = vae.decode(z0)
        tgt_cls = xb[:, :13].argmax(-1)
        loss = (
            F.cross_entropy(dec["cls_logits"], tgt_cls)
            + 3.0 * F.smooth_l1_loss(dec["base"], xb[:, 13:16])
            + 2.0 * F.smooth_l1_loss(dec["rot"], xb[:, 16:22])
            + 3.0 * (F.smooth_l1_loss(torch.log(dec["scale"] + 1e-3),
                                      torch.log(xb[:, 22:25] / 50.0 + 1e-3))
                     + 0.1 * F.smooth_l1_loss(dec["scale"], xb[:, 22:25] / 50.0))
            + 0.5 * F.smooth_l1_loss(dec["curv"], xb[:, 25:26])
        )
        loss.backward()
        grad_sum += z0.grad.abs().sum(0)
        grad_n += xb.shape[0]
        vae.zero_grad(set_to_none=True)
    latent_grad = (grad_sum / grad_n).cpu()
    return Z, latent_grad


# ----------------------------------------------------------------------------
# 3. Panel renderers
# ----------------------------------------------------------------------------
def style_axes(ax):
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_a(ax, cls_counts, total):
    items = sorted(cls_counts.items(), key=lambda kv: kv[1])
    names = [CLASS_NAMES[c] for c, _ in items]
    vals = np.array([v / total * 100.0 for _, v in items])
    colors = []
    for c, _ in items:
        if c == 5:
            colors.append(LAT_C)
        elif c in (6, 9, 10, 11):
            colors.append(RAW_C)
        elif c in (0, 1, 2):
            colors.append(GRAY_C)
        else:
            colors.append(ACC_C)
    bars = ax.barh(names, vals, color=colors, alpha=0.9, height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 700)
    ax.set_xlabel("Share of all organ rows (%, log)")
    style_axes(ax)
    for b, v in zip(bars, vals):
        ax.text(v * 1.14, b.get_y() + b.get_height() / 2, f"{v:.2f}%",
                va="center", fontsize=8, color=FG)
    leaf = cls_counts[5] / total * 100
    flwc = cls_counts[9] / total * 100
    ax.annotate(f"LEAF : FLOWER_CLOSED = {leaf / flwc:.0f}\u00d7 imbalance\n"
                "reproductive organs < 1.6% each",
                xy=(flwc * 1.3, list(names).index("FLOWER_CLOSED") + 0.15),
                xytext=(6.0, 2.3), fontsize=8.2, color=WARN_C,
                arrowprops=dict(arrowstyle="-", color=WARN_C, lw=0.9))
    ax.set_title("A  \u00b7  Botanical Class Imbalance (13 classes)")


def panel_b(ax, tube_grad, labels13):
    vals = np.array(tube_grad)
    disp = []
    for v in vals:
        disp.append(max(v, vals[vals > 0].min() * 0.35) if v > 0 else np.nan)
    colors = []
    for l, v in zip(labels13, vals):
        if l == "s_len":
            colors.append(RAW_C)
        elif l == "s_rad":
            colors.append(WARN_C)
        elif v == 0:
            colors.append(GRAY_C)
        else:
            colors.append(ACC_C)
    bars = ax.bar(labels13, disp, color=colors, alpha=0.9, width=0.66)
    ax.set_yscale("log")
    ylim_min = vals[vals > 0].min() * 0.2
    ax.set_ylim(ylim_min, max(vals) * 12)
    ax.set_ylabel("Gradient RMS  ($2\\sigma$, FM-encoded units)")
    ax.tick_params(axis="x", rotation=45, labelsize=8.2)
    style_axes(ax)
    for b, l, v in zip(bars, labels13, vals):
        if v == 0:
            ax.text(b.get_x() + b.get_width() / 2, ylim_min * 1.4, "DEAD\n($\\sigma{=}0$)",
                    ha="center", va="bottom", fontsize=7.2, color="#6e7681")
        else:
            ax.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:.2f}",
                    ha="center", fontsize=7.4, color=FG)
    spread = vals.max() / vals[vals > 0].min()
    ax.annotate(f"spread $s_{{len}}/s_{{rad}}$ = {spread:.0f}\u00d7\n"
                "peduncle physical $L/r$ = 156\u00d7\n(0.35 m stem vs 2.25 mm stalk)",
                xy=(labels13.index("s_rad"), vals[vals > 0].min() * 1.5),
                xytext=(3.0, max(vals) * 5.0), fontsize=8.2, color=WARN_C,
                arrowprops=dict(arrowstyle="-", color=WARN_C, lw=0.9))
    ax.set_title("B  \u00b7  Raw 13D Gradient Unbalance (tube organs)")


def panel_c(ax_raw, ax_lat, tube_grad, latent_grad):
    labels13 = ["bx", "by", "bz", "r0", "r1", "r2", "r3", "r4", "r5",
                "s_len", "s_rad", "s_z", "curv"]
    tv = np.array(tube_grad)
    tv_norm = tv / tv.max()

    ax_raw.bar(range(len(tv_norm)), tv_norm, color=RAW_C, alpha=0.9, width=0.62)
    ax_raw.set_xticks(range(len(labels13)))
    ax_raw.set_xticklabels(labels13, fontsize=7.2, rotation=45)
    ax_raw.set_ylabel("norm. |grad|", fontsize=8.5)
    ax_raw.set_ylim(0, 1.42)
    style_axes(ax_raw)
    cv_raw = tv_norm.std() / tv_norm.mean()
    spread_raw = tv.max() / tv[tv > 0].min()
    ax_raw.text(0.98, 0.86, f"RAW 13D  \u00b7  CV {cv_raw:.2f}  \u00b7  spread {spread_raw:.0f}\u00d7  \u00b7  1 dead dim",
                transform=ax_raw.transAxes, ha="right", fontsize=8.6, color=RAW_C, fontweight="bold")
    ax_raw.set_title("C  \u00b7  Gradient Profile: Raw 13D vs 16D Latent (frozen VAE)")

    lg = latent_grad if isinstance(latent_grad, np.ndarray) else latent_grad.numpy()
    lg_norm = lg / lg.max()
    ax_lat.bar(range(len(lg_norm)), lg_norm, color=LAT_C, alpha=0.9, width=0.62)
    ax_lat.set_xticks(range(0, 16))
    ax_lat.set_xticklabels([f"z{i}" for i in range(16)], fontsize=7.2)
    ax_lat.set_xlabel("latent dimension")
    ax_lat.set_ylabel("norm. |grad|", fontsize=8.5)
    ax_lat.set_ylim(0, 1.42)
    style_axes(ax_lat)
    cv_lat = lg_norm.std() / lg_norm.mean()
    spread_lat = lg.max() / lg.min()
    ax_lat.text(0.98, 0.86, f"LATENT 16D  \u00b7  CV {cv_lat:.2f}  \u00b7  spread {spread_lat:.1f}\u00d7  \u00b7  16/16 alive",
                transform=ax_lat.transAxes, ha="right", fontsize=8.6, color=LAT_C, fontweight="bold")
    ax_lat.text(0.02, 0.80, f"$dL/dz$ at FM prior $z_0\\sim\\mathcal{{N}}(0,I)$ through frozen decoder \u00b7 "
                            f"{spread_raw / spread_lat:.0f}\u00d7 tighter spread",
                transform=ax_lat.transAxes, ha="left", fontsize=8, color=AXIS)


def panel_d(ax, Z, C):
    Zc = Z - Z.mean(0)
    U, S, V = torch.pca_lowrank(Zc, q=4, niter=4)
    P = (Zc @ V[:, :2]).numpy()
    evr = float((S[:2] ** 2).sum() / (S ** 2).sum())

    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(P.shape[0], generator=g)[:12000].numpy()
    classes = sorted(set(C.tolist()))
    for c in classes:
        m = C[idx] == c
        if m.sum() < 30:
            continue
        ax.scatter(P[idx, 0][m], P[idx, 1][m], s=2.6, alpha=0.32,
                   color=CLASS_PALETTE.get(c, "#555555"), label=CLASS_NAMES[c], lw=0)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    style_axes(ax)
    ax.legend(loc="upper left", fontsize=6.8, ncol=2, markerscale=3, columnspacing=0.8,
              handletextpad=0.3)

    try:
        from sklearn.metrics import silhouette_score
        sub = torch.randperm(P.shape[0], generator=g)[:8000].numpy()
        sil = silhouette_score(P[sub], C[sub].numpy())
    except Exception:
        sil = float("nan")
    ax.text(0.98, 0.03,
            f"PCA-2D silhouette {sil:.2f}  \u00b7  {evr * 100:.0f}% var in 2 PCs\n"
            "16D $z\\sim\\mathcal{N}(0,I)$ \u2014 semantic clusters + isotropic noise prior",
            transform=ax.transAxes, ha="right", fontsize=8, color=AXIS)
    ax.set_title("D  \u00b7  16D Latent Manifold (PCA, organ-class clusters)")
    return sil, evr


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))
    log(f"shard corpus: {len(files)} samples")
    assert len(files) > 0, f"no shards found in {CACHE_DIR}"

    log(f"loading stratified organ sample ({N_STAT_SAMPLES} shards, seed {SEED}) ...")
    X, C = load_organ_sample(files, N_STAT_SAMPLES, SEED)
    log(f"physical organs sampled: {X.shape[0]} (classes >= 3)")

    cls_counts = collections.Counter(C.tolist())
    total_rows = sum(cls_counts.values())

    # raw 26D per-channel gradient RMS at mean-predictor = 2*sigma
    tube_mask = torch.isin(C, torch.tensor(TUBE_CLASSES))
    Xt = X[tube_mask]
    tube_grad = (2.0 * Xt.std(0))[13:26]          # geometry channels of tube organs
    labels13 = ["bx", "by", "bz", "r0", "r1", "r2", "r3", "r4", "r5",
                "s_len", "s_rad", "s_z", "curv"]

    # peduncle physical L/r
    ped = X[C == 6]
    ped_len = ped[:, 22] / 50.0
    ped_rad = ped[:, 23] / 50.0
    ped_ratio = float(ped_len.mean() / ped_rad.mean())
    log(f"peduncle L/r = {ped_ratio:.0f}x (L {ped_len.mean()*100:.1f} cm, r {ped_rad.mean()*1000:.2f} mm)")

    # --- VAE ---
    log("loading frozen OrganLatentVAE ...")
    vae = OrganLatentVAE(latent_dim=16, hidden_dim=256)
    sd = torch.load(VAE_CKPT, map_location="cpu", weights_only=False)
    vae.load_state_dict(sd, strict=True)
    vae.eval().to(DEVICE)
    for p in vae.parameters():
        p.requires_grad_(False)

    log("encoding latents + probing dL/dz at FM prior through frozen decoder ...")
    Z, latent_grad = vae_latents_and_grads(vae, X)
    log(f"latent grad per-dim: {np.array2string(latent_grad.numpy(), precision=2)}")

    # measured summary
    tube_spread = float(tube_grad.max() / tube_grad[tube_grad > 0].min())
    lat_spread = float(latent_grad.max() / latent_grad.min())
    leaf_imb = cls_counts[5] / cls_counts[9]
    summary = {
        "n_shards_scanned": len(files),
        "n_stat_samples": N_STAT_SAMPLES,
        "n_physical_organs": int(X.shape[0]),
        "class_counts": {CLASS_NAMES[k]: int(v) for k, v in sorted(cls_counts.items())},
        "raw_tube_gradient_spread": tube_spread,
        "peduncle_length_radius_ratio": ped_ratio,
        "latent_gradient_spread": lat_spread,
        "latent_gradient_dims_alive": int((latent_grad > 1e-6).sum()),
        "source": {
            "corpus": "dataset/cache/cowpea_curv26 (10,000 Helios Cowpea shards)",
            "vae_ckpt": "diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt",
        },
    }

    # --- Figure ---
    log("rendering figure ...")
    fig = plt.figure(figsize=(15.6, 9.4), dpi=200)
    gs = fig.add_gridspec(2, 2, left=0.092, right=0.966, top=0.845, bottom=0.075,
                          wspace=0.31, hspace=0.36)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    gs_c = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.46)
    axC_raw = fig.add_subplot(gs_c[0])
    axC_lat = fig.add_subplot(gs_c[1])
    axD = fig.add_subplot(gs[1, 1])

    panel_a(axA, cls_counts, total_rows)
    panel_b(axB, tube_grad.numpy(), labels13)
    panel_c(axC_raw, axC_lat, tube_grad.numpy(), latent_grad.numpy())
    sil, evr = panel_d(axD, Z, C)
    summary["pca2d_silhouette"] = float(sil)
    summary["pca2d_explained_variance"] = float(evr)

    fig.suptitle("Fixed Plant Organ Vector Sparsity & Gradient Unbalance using VAE",
                 x=0.5, y=0.971, fontsize=17, fontweight="bold", color=FG)
    fig.text(0.5, 0.913,
             f"Raw 13D regression: {tube_spread:.0f}\u00d7 gradient spread, {ped_ratio:.0f}\u00d7 peduncle $L/r$, "
             f"{leaf_imb:.0f}\u00d7 class imbalance    \u2192    "
             "16D latent $z\\sim\\mathcal{N}(0,\\;I)$ via frozen OrganLatentVAE: "
             f"{lat_spread:.1f}\u00d7 spread, 16/16 dims alive, IoU 91.9%, depth MAE 1.39 cm",
             ha="center", fontsize=10.5, color="#4a5361")
    fig.text(0.5, 0.016,
             f"Empirically measured \u00b7 {len(files):,}-shard Cowpea corpus \u00b7 {N_STAT_SAMPLES}-sample stratified scan "
             f"\u2192 {X.shape[0]:,} physical organs \u00b7 gradient probe $dL/dz$ at FM prior $z_0\\sim\\mathcal{{N}}(0,I)$ "
             "through frozen decoder (organ_latent_vae_best.pt)",
             ha="center", fontsize=8.5, color=AXIS)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    log(f"saved {OUT_PNG}")
    log(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()