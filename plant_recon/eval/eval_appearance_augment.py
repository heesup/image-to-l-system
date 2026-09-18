"""Validate the appearance augmentation against the measured sim-to-real gap.

For the 20 strict-protocol eval plants there are now pixels of the SAME plant at the SAME framing in
two domains: the flat cache render the network trains on, and the Helios raytraced re-render that cost
it 10 points of raw strict P (render_helios_eval_crops.py). This script asks whether
plant_recon/dataset/appearance_augment.py moves the flat render toward the raytraced one.

Outputs (--out_dir):
  appearance_augment_panel.png   flat | augmented x2 | Helios, per plant and zoom level
  appearance_augment_stats.json  DINOv2 feature distances between the domains (flat / augmented / Helios / real)

The distance is computed on the frozen DINOv2 CLS+mean-patch features the model itself reads
(dinov2_vits14 at 224 px): per-plant cosine distance to that plant's Helios render, and a
Frechet distance between the domain feature distributions. Runs on CPU by default -- it is small, and
a local training run must not share the GPU (2026-09-16: three stalls from stacking work on one card).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from plant_recon.dataset.appearance_augment import AugmentConfig, SoilBank, augment_image16, ZOOMS  # noqa: E402


def load_pairs(cache_dir, rgb_dir, limit):
    """[(prefix, cached 16ch image, helios 12ch RGB)] for every eval plant that has a Helios render."""
    out = []
    for f in sorted(glob.glob(os.path.join(rgb_dir, "*_helios_rgb.pt")))[:limit]:
        prefix = Path(f).name.replace("_helios_rgb.pt", "")
        cp = os.path.join(cache_dir, prefix + ".pt")
        if not os.path.exists(cp):
            continue
        d = torch.load(cp, map_location="cpu", weights_only=False)
        img = d["image"].float()
        hel = torch.load(f, map_location="cpu", weights_only=True).float()
        if img.shape[-1] != hel.shape[-1]:
            img = F.interpolate(img.unsqueeze(0), size=hel.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        out.append((prefix, img, hel))
    return out


def rgb_levels(img, is16):
    """[(3, S, S)] in [0,1] per zoom level, from a 16-channel cached image or a 12-channel RGB stack."""
    return [((img[4 * l:4 * l + 3] if is16 else img[3 * l:3 * l + 3]) * 0.5 + 0.5).clamp(0, 1) for l in range(len(ZOOMS))]


@torch.no_grad()
def features(backbone, imgs, device):
    """DINOv2 CLS + mean patch token, L2-normalized, for a list of (3, S, S) tensors in [0,1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    out = []
    for i in range(0, len(imgs), 8):
        b = torch.stack([(im - mean) / std for im in imgs[i:i + 8]]).to(device)
        b = F.interpolate(b, size=(224, 224), mode="bilinear", align_corners=False)
        f = backbone.forward_features(b)
        v = torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], -1)
        out.append(F.normalize(v, dim=-1).cpu())
    return torch.cat(out)


def mmd2(a, b):
    """Unbiased MMD^2 with an RBF kernel (median-heuristic bandwidth) between two feature sets.

    A Frechet/FID-style distance needs far more samples than feature dimensions to estimate a covariance;
    here one domain has 20 members against 768 dimensions, so its covariance would be rank-deficient and
    the number meaningless. MMD is consistent at this sample size. 0 means the two sets are
    indistinguishable under the kernel; larger is further apart.
    """
    x = torch.cat([a, b])
    d2 = torch.cdist(x, x) ** 2
    n = a.shape[0]
    med = d2[d2 > 0].median().clamp(min=1e-8)
    k = torch.exp(-d2 / med)
    kaa, kbb, kab = k[:n, :n], k[n:, n:], k[:n, n:]
    na, nb = n, b.shape[0]
    t1 = (kaa.sum() - kaa.diag().sum()) / (na * (na - 1))
    t2 = (kbb.sum() - kbb.diag().sum()) / (nb * (nb - 1))
    return float(t1 + t2 - 2 * kab.mean())


def panel(pairs, augs, path, n_plants=4, levels=(0, 2)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "figure.facecolor": "white"})
    sel = list(range(0, len(pairs), max(1, len(pairs) // n_plants)))[:n_plants]
    rows = len(sel) * len(levels)
    fig, axes = plt.subplots(rows, 4, figsize=(7.2, 1.85 * rows))
    axes = np.atleast_2d(axes)
    r = 0
    for si in sel:
        prefix, img, hel = pairs[si]
        for li in levels:
            cols = [rgb_levels(img, True)[li], rgb_levels(augs[si][0], True)[li],
                    rgb_levels(augs[si][1], True)[li], rgb_levels(hel, False)[li]]
            titles = ["flat cache render", "augmented: sunlight", "augmented: rover lamps", "Helios raytraced"]
            for c, (ax, im, t) in enumerate(zip(axes[r], cols, titles)):
                ax.imshow(im.permute(1, 2, 0).numpy(), interpolation="nearest")
                if r == 0:
                    ax.set_title(t, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_linewidth(0.5); s_.set_color("0.4")
                if c == 0:
                    ax.set_ylabel(f"{prefix.split('_')[1]}\nzoom {int(ZOOMS[li])}x", fontsize=7)
            r += 1
    fig.suptitle("Appearance augmentation vs the raytraced target (same plant, same framing)", fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print("figure ->", path)


def real_crops(frames_glob, weights, limit_frames, out_px, plot_width_m=1.3, window_m=1.2, margins=None):
    """Real plant crops at the SAME physical framing as the cache: a fixed `window_m` ground window
    centred on each detected plant, not 1.2x its bounding box.

    This is the framing correction of the assessment §1.3 -- `build_pyramid_16ch` sizes the zoom-1x window
    from the detector box, while the training cache uses a fixed 1.2 m window, so a real plant reaches the
    network several times too large. Cutting the comparison crops the cache's way keeps this measurement
    about pixels alone.
    """
    from PIL import Image
    from use_cases.real_world.dataset.real_plant_crop_utils import (DEFAULT_ROVER_MARGINS, crop_rover_margins,
                                                                     detect_plants)
    margins = margins or DEFAULT_ROVER_MARGINS
    out = []
    for fp in sorted(glob.glob(frames_glob))[:limit_frames]:
        im = crop_rover_margins(Image.open(fp).convert("RGB"), margins)
        px_per_m = im.width / plot_width_m
        win = int(round(window_m * px_per_m))
        for det in detect_plants(im, weights, conf=0.25):
            cx, cy = det.center
            x0, y0 = int(round(cx - win / 2)), int(round(cy - win / 2))
            patch = Image.new("RGB", (win, win), (0, 0, 0))
            src = im.crop((max(0, x0), max(0, y0), min(im.width, x0 + win), min(im.height, y0 + win)))
            patch.paste(src, (max(0, -x0), max(0, -y0)))
            t = torch.from_numpy(np.asarray(patch.resize((out_px, out_px), Image.BILINEAR), dtype=np.float32) / 255.0)
            out.append(t.permute(2, 0, 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default="dataset/cache/cowpea_curv26")
    ap.add_argument("--rgb_dir", default="outputs/logs/20260916/helios_eval_crops/rgb")
    ap.add_argument("--soil_bank", default="dataset/soil_bank_agml")
    ap.add_argument("--out_dir", default="outputs/logs/20260917/appearance_augment")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--draws", type=int, default=4, help="augmentation draws per plant for the statistics")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_features", action="store_true")
    ap.add_argument("--real_frames", default="use_cases/real_world/data/agml_gemini_plant_detection_2022/test/images/*.jpg",
                    help="real frames to cut comparison crops from ('' disables the real-domain column)")
    ap.add_argument("--detector_weights", default="outputs/logs/20260916/agml_plant_detector/weights/best.pt")
    ap.add_argument("--real_frame_limit", type=int, default=40)
    a = ap.parse_args()

    pairs = load_pairs(a.cache_dir, a.rgb_dir, a.limit)
    print(f"{len(pairs)} plants with both a cached render and a Helios render")
    soil = SoilBank(a.soil_bank)
    print(f"soil bank: {len(soil.patches)} patches at {soil.px_per_m:.0f} px/m")
    # one draw of each lighting regime, so the panel shows both rather than two random draws of whichever
    # the coin landed on (the real captures mix sunlight and the rover's own lamps).
    cfg_sun, cfg_rover = AugmentConfig(p_rover=0.0), AugmentConfig(p_rover=1.0)
    augs = []
    for i, (prefix, img, _) in enumerate(pairs):
        g = torch.Generator().manual_seed(1000 + i)
        draws = [augment_image16(img, soil, cfg_sun, g), augment_image16(img, soil, cfg_rover, g)]
        while len(draws) < a.draws:      # extra draws for the statistics alternate between the regimes
            draws.append(augment_image16(img, soil, cfg_sun if len(draws) % 2 == 0 else cfg_rover, g))
        augs.append(draws)
    os.makedirs(a.out_dir, exist_ok=True)
    panel(pairs, augs, os.path.join(a.out_dir, "appearance_augment_panel.png"))

    if a.no_features:
        return
    dev = torch.device(a.device)
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True, verbose=False).to(dev).eval()
    real = []
    if a.real_frames and os.path.exists(a.detector_weights):
        S = pairs[0][1].shape[-1]
        real = real_crops(a.real_frames, a.detector_weights, a.real_frame_limit, S)
        print(f"{len(real)} real plant crops at the cache's fixed 1.2 m window", flush=True)
    stats = {}
    for li in range(len(ZOOMS)):
        flat = [rgb_levels(im, True)[li] for _, im, _ in pairs]
        hel = [rgb_levels(h, False)[li] for _, _, h in pairs]
        aug = [rgb_levels(a_, True)[li] for lst in augs for a_ in lst]
        f_flat, f_hel, f_aug = (features(backbone, x, dev) for x in (flat, hel, aug))
        # per-plant: distance to that plant's own Helios render (augmentation draws averaged)
        d_flat = float((1 - (f_flat * f_hel).sum(-1)).mean())
        f_aug_r = f_aug.reshape(len(pairs), a.draws, -1)
        d_aug = float((1 - (f_aug_r * f_hel.unsqueeze(1)).sum(-1)).mean())
        stats[f"zoom{int(ZOOMS[li])}x"] = {
            "cosine_flat_vs_helios": round(d_flat, 4),
            "cosine_augmented_vs_helios": round(d_aug, 4),
            "mmd_flat_vs_helios": round(mmd2(f_flat, f_hel), 4),
            "mmd_augmented_vs_helios": round(mmd2(f_aug, f_hel), 4),
        }
        s = stats[f"zoom{int(ZOOMS[li])}x"]
        if real:
            # the decision-relevant direction: Helios is only a proxy, the real photos are the target.
            # Each zoom level is the concentric sub-window of the same real crop, matching how the cache
            # builds its pyramid, so the comparison is like-for-like at every level.
            S = real[0].shape[-1]
            half = max(8, int(round(S / ZOOMS[li] / 2)))
            c0 = S // 2
            r_lvl = [F.interpolate(t[:, c0 - half:c0 + half, c0 - half:c0 + half].unsqueeze(0),
                                   size=(S, S), mode="bilinear", align_corners=False).squeeze(0) for t in real]
            f_r = features(backbone, r_lvl, dev)
            s["mmd_flat_vs_real"] = round(mmd2(f_flat, f_r), 4)
            s["mmd_augmented_vs_real"] = round(mmd2(f_aug, f_r), 4)
            s["mmd_helios_vs_real"] = round(mmd2(f_hel, f_r), 4)
        print(f"zoom {int(ZOOMS[li])}x  cosine to Helios: flat {s['cosine_flat_vs_helios']:.4f} -> augmented "
              f"{s['cosine_augmented_vs_helios']:.4f} | MMD to Helios: {s['mmd_flat_vs_helios']:.4f} -> "
              f"{s['mmd_augmented_vs_helios']:.4f}"
              + (f" | MMD to REAL: flat {s['mmd_flat_vs_real']:.4f} -> augmented {s['mmd_augmented_vs_real']:.4f}"
                 f" (Helios {s['mmd_helios_vs_real']:.4f})" if "mmd_flat_vs_real" in s else ""), flush=True)
    json.dump({"n_plants": len(pairs), "draws": a.draws, "stats": stats},
              open(os.path.join(a.out_dir, "appearance_augment_stats.json"), "w"), indent=1)
    print("stats ->", os.path.join(a.out_dir, "appearance_augment_stats.json"))


if __name__ == "__main__":
    main()
