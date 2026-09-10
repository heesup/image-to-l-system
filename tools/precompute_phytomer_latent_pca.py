"""Precompute PhytomerVAE-64 latent cloud + PCA projections for the GUI.

Builds canonical 8-slot phytomer packets from the cache dataset using EXACT
XML phytomer membership (cache field `phytomer_ids`), encodes them with the
frozen accepted PhytomerVAE (phytomer_vae_xml, use_rot_branch=True), fits
PCA(64 -> 3), and saves everything the visualizer needs:

    /home/lion397/codes/image-to-l-system/dataset/cache/phytomer_gui_cache/
        z.pt            (P, 64)  mu-encoded latents
        pca.pt          fitted sklearn PCA object
        proj2d.pt       (P, 2)  PCA 2D projection
        proj3d.pt       (P, 3)  PCA 3D projection
        presence.pt     (P, 8)  bool slot mask
        refs.pt         (P, 6)  absolute reference-frame rot6d per packet
        centers.pt      (P, 3)  absolute cluster centers (metres)
        packets.pt      (P, 8, 26) relative packets (for re-anchoring)
        dap.pt          (P,)    DAP label per packet
        meta.json       config + per-dim percentile ranges for sliders

Usage (workspace root):
    .../bin/python tools/precompute_phytomer_latent_pca.py \
        --ckpt diffusion_based/checkpoints/phytomer_vae_xml/phytomer_vae_64d_best.pt \
        --out /tmp/opencode/phytomer_gui_cache
"""

import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.decomposition import PCA

from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.dataset.phytomer_packets import build_phytomer_packets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="diffusion_based/checkpoints/phytomer_vae_xml/phytomer_vae_64d_best.pt")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out", type=str, default="dataset/cache/phytomer_gui_cache")
    parser.add_argument("--cache-dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Cap the number of cache files (0 = all). For smoke tests.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. Build XML-phytomer packets from cache files (with DAP labels) ----
    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.max_files > 0:
        files = files[: args.max_files]
    print(f"Building packets from {len(files):,} cache files (XML clustering)...")

    all_packets, all_presence, all_centers, all_refs, all_dap = [], [], [], [], []
    stats: dict = {}
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            data = torch.load(f, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if not (isinstance(data, dict) and "nodes" in data):
            continue
        packets, presence, centers, refs = build_phytomer_packets(
            data["nodes"], data.get("existence_mask"),
            phytomer_ids=data.get("phytomer_ids"), drop_stats=stats)
        if packets.shape[0] == 0:
            continue
        all_packets.append(packets)
        all_presence.append(presence)
        all_centers.append(centers)
        all_refs.append(refs)
        dap = float(data.get("dap", -1.0))
        all_dap.extend([dap] * packets.shape[0])
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{len(files)}] packets={sum(p.shape[0] for p in all_packets):,} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    packets = torch.cat(all_packets, dim=0)
    presence = torch.cat(all_presence, dim=0)
    centers = torch.cat(all_centers, dim=0)
    refs = torch.cat(all_refs, dim=0)
    dap = torch.tensor(all_dap, dtype=torch.float32)
    print(f"Extracted {packets.shape[0]:,} packets from {len(files):,} files in "
          f"{time.time()-t0:.0f}s (drop rate "
          f"{stats.get('dropped_organs', 0) / max(stats.get('total_organs', 1), 1) * 100:.2f}%)")

    # ---- 2. Encode with the frozen XML VAE (mu encoding, no sampling) ----
    model = PhytomerVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {args.ckpt}")

    z_list = []
    bs = 8192
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, packets.shape[0], bs):
            p = packets[i : i + bs].to(device)
            pr = presence[i : i + bs].to(device)
            mu, _ = model.encode(model.pack_input(p, pr))
            z_list.append(mu.cpu())
    z = torch.cat(z_list, dim=0)
    print(f"Encoded {z.shape[0]:,} latents in {time.time() - t0:.0f}s "
          f"(|z| mean {z.norm(dim=-1).mean().item():.2f}, std {z.std().item():.2f})")

    # ---- 3. PCA fit on the full latent cloud ----
    zn = z.numpy()
    pca = PCA(n_components=3, whiten=False)
    proj3d = pca.fit_transform(zn)
    proj2d = proj3d[:, :2]
    evr = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1 {evr[0]*100:.1f}% PC2 {evr[1]*100:.1f}% PC3 {evr[2]*100:.1f}%")

    # Per-dim percentile ranges for slider bounds (0.5% .. 99.5%).
    lo = np.percentile(zn, 0.5, axis=0)
    hi = np.percentile(zn, 99.5, axis=0)
    meta = {
        "n_packets": int(z.shape[0]),
        "latent_dim": args.latent_dim,
        "pca_evr": [float(e) for e in evr],
        "z_mean": [float(m) for m in zn.mean(axis=0)],
        "z_std": [float(s) for s in zn.std(axis=0)],
        "slider_lo": [float(v) for v in lo],
        "slider_hi": [float(v) for v in hi],
        "ckpt": args.ckpt,
        "cache_dir": args.cache_dir,
        "clustering": "xml_phytomer_ids",
    }

    os.makedirs(args.out, exist_ok=True)
    torch.save(z, os.path.join(args.out, "z.pt"))
    torch.save(pca, os.path.join(args.out, "pca.pt"))
    torch.save(torch.from_numpy(proj2d).float(), os.path.join(args.out, "proj2d.pt"))
    torch.save(torch.from_numpy(proj3d).float(), os.path.join(args.out, "proj3d.pt"))
    torch.save(presence, os.path.join(args.out, "presence.pt"))
    torch.save(refs, os.path.join(args.out, "refs.pt"))
    torch.save(centers, os.path.join(args.out, "centers.pt"))
    torch.save(packets, os.path.join(args.out, "packets.pt"))
    torch.save(dap, os.path.join(args.out, "dap.pt"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved cache to {args.out} (dap range {dap.min().item():.0f}-{dap.max().item():.0f})")
    print("Done.")


if __name__ == "__main__":
    main()
