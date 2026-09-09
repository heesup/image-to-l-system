"""Offline Hungarian relabeling of phytomer packets (Exp F protocol).

Problem: canonical within-role ordering (bottom->top z) is unstable for
near-coincident organs (leaflet base z-gap median 1.4mm; ordering agreement
43.5% ~ random), forcing the VAE to fit a multimodal target with a unimodal
posterior (blurred rotations + heavy-tailed bases). In-loop Hungarian (Exp E)
fixes assignment but flickers (targets move every step -> plateau at rot 0.036).

Solution: compute the optimal assignment ONCE, offline, using a converged model
(Exp D), then train fresh on the relabeled packets with the standard FAST
canonical loss (no in-loop Hungarian). Fixed targets (no flicker) +
optimal assignment (no canonical bias). Relabeling cost includes rotation so
ambiguous organs match by full similarity, not just position.

Performance architecture: forward all packets on GPU in bulk, transfer ALL
predictions to CPU ONCE, then pure-numpy per-packet loop (no per-packet GPU
dispatches or syncs — those made the naive version take hours).

Usage:
    .../bin/python tools/relabel_packets_hungarian.py \
        --packet-cache /tmp/opencode/phytomer_packets_4k.pt \
        --checkpoint diffusion_based/checkpoints/phytomer_vae_expd/phytomer_vae_64d_best.pt \
        --out-cache /tmp/opencode/phytomer_packets_relabeled.pt
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from diffusion_based.models.phytomer_vae import PhytomerVAE


ROLE_RANGES = [(0, 1), (1, 2), (2, 5), (5, 6), (6, 8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-cache", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-cache", type=str, required=True)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--rot-branch", action="store_true", default=True)
    parser.add_argument("--no-rot-branch", dest="rot_branch", action="store_false")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--w-pos", type=float, default=1.0)
    parser.add_argument("--w-scale", type=float, default=0.5)
    parser.add_argument("--w-cls", type=float, default=1.0)
    parser.add_argument("--w-rot", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.packet_cache, map_location="cpu", weights_only=False)
    packets, presence = blob["packets"], blob["presence"]
    print(f"Loaded {packets.shape[0]:,} packets.")

    vae = PhytomerVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    vae.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    vae.eval()

    relabeled = torch.zeros_like(packets)
    relabeled[:, :, 0] = 1.0  # NONE one-hot default
    relabeled_pres = torch.zeros_like(presence, dtype=torch.bool)
    n_changed = 0
    n_ambiguous = 0

    # Bulk forward on GPU, then transfer EVERYTHING to CPU once.
    # All assignment math below is numpy (no per-packet GPU dispatches/syncs).
    all_pred_base, all_pred_scale, all_pred_cls, all_pred_rot = [], [], [], []
    with torch.no_grad():
        for i in range(0, packets.shape[0], args.batch_size):
            p = packets[i:i + args.batch_size].to(device)
            pr = presence[i:i + args.batch_size].to(device)
            out = vae(p, pr, use_rot_branch=args.rot_branch)
            all_pred_base.append(out["base"].cpu())
            all_pred_scale.append(out["scale"].cpu())
            all_pred_cls.append(out["cls_logits"].argmax(-1).cpu())
            all_pred_rot.append(out["rot"].cpu())
    import numpy as np
    pred_base = torch.cat(all_pred_base, dim=0).numpy()    # (P, 8, 3)
    pred_scale = torch.cat(all_pred_scale, dim=0).numpy()  # (P, 8, 3)
    pred_cls = torch.cat(all_pred_cls, dim=0).numpy()      # (P, 8)
    pred_rot = torch.cat(all_pred_rot, dim=0).numpy()      # (P, 8, 6)
    tgt_base = packets[:, :, 13:16].numpy()
    tgt_scale = packets[:, :, 22:25].numpy()
    tgt_rot = packets[:, :, 16:22].numpy()
    tgt_cls = packets[:, :, :13].argmax(-1).numpy()
    pres_np = presence.numpy()

    def rot_dist(a6, b6):
        # Angular distance between FORWARD axes (col 1 = tube long axis).
        # Used for assignment on cylindrical roles (stem/petiole/peduncle);
        # leaflets/repro use full 6D Frobenius via full-matrix cosine.
        na = np.linalg.norm(a6[..., 3:6], axis=-1, keepdims=True).clip(min=1e-8)
        nb = np.linalg.norm(b6[..., 3:6], axis=-1, keepdims=True).clip(min=1e-8)
        cosang = ((a6[..., 3:6] / na) * (b6[..., 3:6] / nb)).sum(-1).clip(-1.0, 1.0)
        return np.arccos(cosang)

    def full_rot_dist(a6, b6):
        # Full SO(3) distance (Frobenius cosine), for leaflet/repro assignment.
        import numpy.linalg as la
        Ra = Rmat_np(a6)
        Rb = Rmat_np(b6)
        c = ((Ra * Rb).sum(axis=(-2, -1)) - 1.0) / 2.0
        return np.arccos(np.clip(c, -1.0, 1.0))

    def Rmat_np(a6):
        x = a6[..., 0:3] / np.linalg.norm(a6[..., 0:3], axis=-1, keepdims=True).clip(min=1e-8)
        z = np.cross(x, a6[..., 3:6])
        z = z / np.linalg.norm(z, axis=-1, keepdims=True).clip(min=1e-8)
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=-2)

    P = packets.shape[0]
    for i in range(P):
        for (lo, hi) in ROLE_RANGES:
            gt_idx = [s for s in range(lo, hi) if pres_np[i, s]]
            if not gt_idx:
                continue
            slots = list(range(lo, hi))
            if len(gt_idx) == 1:
                # Unambiguous: canonical-first placement (no LSA).
                relabeled[i, lo] = packets[i, gt_idx[0]]
                relabeled_pres[i, lo] = True
                continue
            n_ambiguous += 1
            n_s = len(slots)
            # Assignment distance per role type:
            #   stem/petiole/peduncle (cylindrical): forward-axis (col 1) angle
            #   leaflets/repro (meshes): full SO(3) distance
            use_axis = (lo, hi) in ((0, 1), (1, 2), (5, 6))
            dist_fn = rot_dist if use_axis else full_rot_dist
            cb = np.zeros((n_s, len(gt_idx)))
            for r, s in enumerate(slots):
                db = np.abs(pred_base[i, s] - tgt_base[i, gt_idx]).sum(-1)
                ds = np.abs(pred_scale[i, s] - tgt_scale[i, gt_idx]).sum(-1)
                dc = (pred_cls[i, s] != np.array(tgt_cls[i, gt_idx])).astype(float)
                dr = np.array([dist_fn(pred_rot[i, s], tgt_rot[i, g])
                               for g in gt_idx])
                cb[r] = db + args.w_scale * ds + args.w_cls * dc + args.w_rot * dr
            r_idx, c_idx = linear_sum_assignment(cb)
            for r, c in zip(r_idx.tolist(), c_idx.tolist()):
                relabeled[i, slots[r]] = packets[i, gt_idx[c]]
                relabeled_pres[i, slots[r]] = True
            if (r_idx.tolist() != list(range(len(r_idx)))
                    or c_idx.tolist() != list(range(len(c_idx)))):
                n_changed += 1
        if (i + 1) % 20000 == 0:
            print(f"  relabeled {i + 1}/{P}", flush=True)

    print(f"Ambiguous roles: {n_ambiguous}; "
          f"assignment differed from canonical in {n_changed} cases.")
    out_blob = {"packets": relabeled, "presence": relabeled_pres,
                "drop_stats": blob.get("drop_stats", {})}
    torch.save(out_blob, args.out_cache)
    print(f"Saved relabeled packets to {args.out_cache}")


if __name__ == "__main__":
    main()
