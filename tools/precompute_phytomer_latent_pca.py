"""Precompute PhytomerVAE-64 latent cloud + PCA projections for the GUI.

Encodes the packet cache (/tmp/opencode/phytomer_packets_4k_rel.pt) with the
frozen accepted PhytomerVAE (phytomer_vae_relative_d, use_rot_branch=True),
fits PCA(64 -> 3), and saves everything the visualizer needs:

    /tmp/opencode/phytomer_gui_cache/
        z.pt            (P, 64)  mu-encoded latents
        pca.pt          fitted sklearn PCA object
        proj2d.pt       (P, 2)  PCA 2D projection
        proj3d.pt       (P, 3)  PCA 3D projection
        presence.pt     (P, 8)  bool slot mask
        refs.pt         (P, 6)  absolute reference-frame rot6d per packet
        centers.pt      (P, 3)  absolute cluster centers (metres)
        packets.pt      (P, 8, 26) relative packets (for re-anchoring)
        dap.pt          (P,)    DAP label per packet (optional, --with-dap)
        meta.json       config + per-dim percentile ranges for sliders

Usage (workspace root):
    .../bin/python tools/precompute_phytomer_latent_pca.py \
        --packet-cache /tmp/opencode/phytomer_packets_4k_rel.pt \
        --ckpt diffusion_based/checkpoints/phytomer_vae_relative_d/phytomer_vae_64d_best.pt \
        --out /tmp/opencode/phytomer_gui_cache
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.decomposition import PCA

from diffusion_based.models.phytomer_vae import PhytomerVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-cache", type=str,
                        default="/tmp/opencode/phytomer_packets_4k_rel.pt")
    parser.add_argument("--ckpt", type=str,
                        default="diffusion_based/checkpoints/phytomer_vae_structural/phytomer_vae_64d_best.pt")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--out", type=str, default="/tmp/opencode/phytomer_gui_cache")
    parser.add_argument("--max-packets", type=int, default=0,
                        help="Cap the number of packets (0 = all). For smoke tests.")
    parser.add_argument("--with-dap", action="store_true",
                        help="Also extract DAP labels by re-reading cache files "
                             "(requires --cache-dir; slower, one-time).")
    parser.add_argument("--cache-dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    blob = torch.load(args.packet_cache, map_location="cpu", weights_only=False)
    packets = blob["packets"]
    presence = blob["presence"]
    refs = blob["reference_rots"]
    if args.max_packets > 0:
        packets = packets[: args.max_packets]
        presence = presence[: args.max_packets]
        refs = refs[: args.max_packets]
    # The structural VAE strips base columns before encoding (pack_input drops
    # them internally), so full packets are fine as-is.
    print(f"Packets: {packets.shape[0]:,} x {packets.shape[1]} slots x {packets.shape[2]}D")

    model = PhytomerVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {args.ckpt}")

    # Encode in batches (mu encoding, no sampling).
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

    # PCA fit on the full latent cloud.
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
        "packet_cache": args.packet_cache,
    }

    os.makedirs(args.out, exist_ok=True)
    torch.save(z, os.path.join(args.out, "z.pt"))
    torch.save(pca, os.path.join(args.out, "pca.pt"))
    torch.save(torch.from_numpy(proj2d).float(), os.path.join(args.out, "proj2d.pt"))
    torch.save(torch.from_numpy(proj3d).float(), os.path.join(args.out, "proj3d.pt"))
    torch.save(presence, os.path.join(args.out, "presence.pt"))
    torch.save(refs, os.path.join(args.out, "refs.pt"))
    torch.save(packets, os.path.join(args.out, "packets.pt"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved cache to {args.out}")

    if args.with_dap:
        import glob
        import random
        from diffusion_based.dataset.phytomer_packets import build_phytomer_packets

        files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
        rng = random.Random(0)
        rng.shuffle(files)
        dap_list = []
        n_packets = 0
        t0 = time.time()
        for f in files:
            if n_packets >= z.shape[0]:
                break
            try:
                data = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not (isinstance(data, dict) and "nodes" in data):
                continue
            pk, pr, _, _ = build_phytomer_packets(
                data["nodes"], data.get("existence_mask"),
                phytomer_ids=data.get("phytomer_ids"))
            if pk.shape[0] == 0:
                continue
            dap = float(data.get("dap", -1.0))
            dap_list.extend([dap] * pk.shape[0])
            n_packets += pk.shape[0]
            if (n_packets // 10000) != ((n_packets - pk.shape[0]) // 10000):
                print(f"  dap labels: {n_packets:,}/{z.shape[0]:,} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        dap_t = torch.tensor(dap_list[: z.shape[0]], dtype=torch.float32)
        torch.save(dap_t, os.path.join(args.out, "dap.pt"))
        print(f"Saved DAP labels ({dap_t.shape[0]:,}, "
              f"range {dap_t.min().item():.0f}-{dap_t.max().item():.0f})")

    print("Done.")


if __name__ == "__main__":
    main()
