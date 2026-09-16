"""Precompute PhytomerVAE latent cloud + PCA projections for the GUI.

Two modes:

1. Full build (default): builds canonical 10-slot phytomer packets from the
   cache dataset using EXACT XML phytomer membership (cache field
   `phytomer_ids`), encodes them with the frozen accepted PhytomerVAE
   (phytomer_vae_v9_tl_rw4_20k, 128D hybrid latent).

2. Fast path (--from-pkt-cache): reuses packets stored in the pkt cache
   (dataset/cache/cowpea_curv26_pkt_v9/, terminal-last packets, pkt_version
   7). Latents are ALWAYS re-encoded on the fly with --ckpt: since
   2026-09-14 the pkt caches are VAE-independent and any `latent` field they
   still carry comes from whichever VAE generated the cache, which is NOT
   guaranteed to be the VAE you load. No packet rebuild; DAP labels are
   parsed from filenames (cowpea_dapDDD_...).

Both save everything the visualizer needs:

    dataset/cache/phytomer_gui_cache/
        z.pt            (P, D)  mu-encoded latents
        pca.pt          fitted sklearn PCA object
        proj2d.pt       (P, 2)  PCA 2D projection
        proj3d.pt       (P, 3)  PCA 3D projection
        presence.pt     (P, 10)  bool slot mask
        refs.pt         (P, 6)  absolute reference-frame rot6d per packet
        centers.pt      (P, 3)  absolute cluster centers (metres)
        packets.pt      (P, 10, 26) relative packets (for placing packets back into the world frame)
        dap.pt          (P,)    DAP label per packet
        meta.json       config + per-dim percentile ranges for sliders

Usage (workspace root):
    .../bin/python tools/precompute_phytomer_latent_pca.py \
        --ckpt diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/phytomer_vae_128d_best.pt \
        --out dataset/cache/phytomer_gui_cache

    .../bin/python tools/precompute_phytomer_latent_pca.py --from-pkt-cache \
        --pkt-dir dataset/cache/cowpea_curv26_pkt_v9 \
        --ckpt diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/phytomer_vae_128d_best.pt \
        --max-packets 250000 \
        --out dataset/cache/phytomer_gui_cache
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.decomposition import PCA

from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.dataset.phytomer_packets import build_phytomer_packets

_DAP_RE = re.compile(r"cowpea_dap(\d+)_")


def _dap_from_filename(path: str) -> float:
    m = _DAP_RE.search(os.path.basename(path))
    return float(int(m.group(1))) if m else -1.0


def _load_phytomer_vae(args, device):
    model = PhytomerVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {args.ckpt}")
    return model


def _encode_packets(model, packets, presence, device, bs=8192):
    """Mu-encode (no sampling) packet rows with the frozen VAE."""
    z_list = []
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
    return z


def _load_from_pkt_cache(args, device):
    """Concat packets from the pkt cache + encode latents on the fly with --ckpt."""
    files = sorted(glob.glob(os.path.join(args.pkt_dir, "*.pt")))
    rng = random.Random(args.seed)
    rng.shuffle(files)
    print(f"Loading pkt cache: {len(files):,} files, cap {args.max_packets:,} packets...")

    model = _load_phytomer_vae(args, device)

    all_packets, all_presence, all_centers, all_refs, all_dap, all_z = [], [], [], [], [], []
    n_packets = 0
    t0 = time.time()
    for i, f in enumerate(files):
        if n_packets >= args.max_packets:
            break
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not (isinstance(data, dict) and "packets" in data):
            continue
        n = data["packets"].shape[0]
        if n == 0:
            continue
        take = min(n, args.max_packets - n_packets)
        all_packets.append(data["packets"][:take].float())
        all_presence.append(data["presence"][:take])
        all_centers.append(data["centers"][:take].float())
        all_refs.append(data["refs"][:take].float())
        dap = _dap_from_filename(f)
        all_dap.extend([dap] * take)
        n_packets += take
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{len(files)}] packets={n_packets:,} elapsed={time.time()-t0:.0f}s",
                  flush=True)

    packets = torch.cat(all_packets, dim=0)
    presence = torch.cat(all_presence, dim=0)
    centers = torch.cat(all_centers, dim=0)
    refs = torch.cat(all_refs, dim=0)
    dap = torch.tensor(all_dap, dtype=torch.float32)
    z = _encode_packets(model, packets, presence, device)
    source = {"source": "pkt_cache", "pkt_dir": args.pkt_dir,
              "clustering": "xml_phytomer_ids", "latent": "on_the_fly_mu_encode"}
    return packets, presence, centers, refs, dap, z, source


def _build_and_encode(args, device):
    """Build XML-phytomer packets from cache files and encode with the VAE."""
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

    # Encode with the frozen VAE (mu encoding, no sampling).
    model = _load_phytomer_vae(args, device)
    z = _encode_packets(model, packets, presence, device)
    source = {"source": "cache_rebuild", "cache_dir": args.cache_dir,
              "clustering": "xml_phytomer_ids"}
    return packets, presence, centers, refs, dap, z, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/phytomer_vae_128d_best.pt")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out", type=str, default="dataset/cache/phytomer_gui_cache")
    parser.add_argument("--cache-dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--from-pkt-cache", action="store_true",
                        help="Reuse packets stored in the pkt cache (skips packet "
                             "rebuild) and encode latents on the fly with --ckpt.")
    parser.add_argument("--pkt-dir", type=str, default="dataset/cache/cowpea_curv26_pkt_v9")
    parser.add_argument("--max-packets", type=int, default=250000,
                        help="Cap packets in --from-pkt-cache mode (seed-shuffled).")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Cap the number of cache files (0 = all). For smoke tests.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # The VAE's slot semantics depend on the packet convention it was trained
    # on: `_tl` checkpoints expect terminal-last packets. Must be set before
    # build_phytomer_packets runs (same convention as the eval scripts).
    os.environ.setdefault(
        "PHYTOMER_TERMINAL_LAST",
        "1" if "_tl" in os.path.basename(os.path.dirname(args.ckpt)) else "0")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (PHYTOMER_TERMINAL_LAST={os.environ['PHYTOMER_TERMINAL_LAST']})")

    if args.from_pkt_cache:
        packets, presence, centers, refs, dap, z, source = _load_from_pkt_cache(args, device)
    else:
        packets, presence, centers, refs, dap, z, source = _build_and_encode(args, device)

    # PCA fit on the latent cloud (3D for scatter + 16D for coarse sliders).
    zn = z.numpy()
    pca = PCA(n_components=3, whiten=False)
    proj3d = pca.fit_transform(zn)
    proj2d = proj3d[:, :2]
    evr = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1 {evr[0]*100:.1f}% PC2 {evr[1]*100:.1f}% PC3 {evr[2]*100:.1f}%")
    pca16 = PCA(n_components=min(16, zn.shape[1]), whiten=False).fit(zn)
    print(f"PCA16 explained variance: {pca16.explained_variance_ratio_.sum()*100:.1f}%")

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
        "seed": args.seed,
    }
    meta.update(source)

    os.makedirs(args.out, exist_ok=True)
    torch.save(z, os.path.join(args.out, "z.pt"))
    torch.save(pca, os.path.join(args.out, "pca.pt"))
    import pickle
    with open(os.path.join(args.out, "pca16.pt"), "wb") as f:
        pickle.dump(pca16, f)
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
