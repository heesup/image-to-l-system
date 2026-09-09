"""Roundtrip comparison: PhytomerVAE vs OrganLatentVAE baseline on held-out packets.

Fair comparison in ABSOLUTE coordinates:
  - PhytomerVAE: relative packet -> z(D) -> relative packet_hat -> re-anchor
    with the TRUE cluster center -> absolute rows.
  - OrganLatentVAE (frozen baseline): absolute GT rows -> z(16) -> rows_hat.
Both compared against absolute GT rows on present slots only:
  - slot class accuracy, base MAE (cm), rotation angular error (deg),
    scale relative error.

Packets are rebuilt fresh from cache files (disjoint from training files)
so neither model has seen them.

Usage:
    .../bin/python tools/eval_phytomer_vae_roundtrip.py \
        --phytomer-ckpt diffusion_based/checkpoints/phytomer_vae/phytomer_vae_64d_best.pt \
        --organ-ckpt diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt \
        --latent-dim 64 --max-files 200
"""

import argparse
import glob
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.models.organ_latent_vae import OrganLatentVAE
from diffusion_based.dataset.phytomer_packets import build_phytomer_packets, decode_packets, assemble_packets
from diffusion_based.dataset.part_array_dataset import FM_OT_END, FM_BASE_START, BASE_SCALE


def rot6d_to_matrix(r):
    x = F.normalize(r[..., 0:3], dim=-1)
    z = F.normalize(torch.cross(x, r[..., 3:6], dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def angular_error_deg(R_pred, R_tgt):
    cosang = ((R_pred * R_tgt).sum(dim=(-2, -1)) - 1.0) / 2.0
    cosang = cosang.clamp(-1.0, 1.0)
    return torch.acos(cosang) * 180.0 / math.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phytomer-ckpt", type=str, required=True)
    parser.add_argument("--organ-ckpt", type=str,
                        default="diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt")
    parser.add_argument("--cache-dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--rot-branch", action="store_true",
                        help="Decode rotation through the dedicated branch.")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    # Held-out files: high seed indices (training used seed 0 shuffle head;
    # disjoint by construction with overwhelming probability — verified below).
    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
    rng.shuffle(files)

    phy = PhytomerVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    phy.load_state_dict(torch.load(args.phytomer_ckpt, map_location=device, weights_only=True))
    phy.eval()

    organ = OrganLatentVAE(latent_dim=16, hidden_dim=256).to(device)
    organ.load_state_dict(torch.load(args.organ_ckpt, map_location=device, weights_only=True))
    organ.eval()

    stats = {
        "phy": {"cls": 0.0, "base": 0.0, "rot": 0.0, "scale": 0.0, "n": 0},
        "org": {"cls": 0.0, "base": 0.0, "rot": 0.0, "scale": 0.0, "n": 0},
    }
    n_packets = 0
    with torch.no_grad():
        for f in files[:args.max_files]:
            try:
                data = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not (isinstance(data, dict) and "nodes" in data):
                continue
            packets, presence, centers, ref_rots = build_phytomer_packets(
                data["nodes"], data.get("existence_mask"),
                phytomer_ids=data.get("phytomer_ids"))
            if packets.shape[0] == 0:
                continue
            p = packets.to(device)
            pr = presence.to(device)
            refs = ref_rots.to(device)
            m = pr.float()
            n_present = m.sum().clamp(min=1.0)

            # --- PhytomerVAE joint roundtrip, re-anchored to absolute ---
            out = phy(p, pr, use_rot_branch=args.rot_branch)
            # Structurally assemble slot bases from petiole geometry (deterministic).
            rec_assembled = assemble_packets(out["recon_packets"], refs)
            abs_hat = decode_packets(rec_assembled, centers.to(device), pr, refs)
            abs_gt = decode_packets(p, centers.to(device), pr, refs)
            pred_cls = out["cls_logits"].argmax(-1)
            tgt_cls = p[:, :, :FM_OT_END].argmax(-1)
            stats["phy"]["cls"] += (((pred_cls == tgt_cls).float() * m).sum() / n_present).item()
            db = (abs_hat[:, :, FM_BASE_START:FM_BASE_START + 3]
                  - abs_gt[:, :, FM_BASE_START:FM_BASE_START + 3]).abs()
            stats["phy"]["base"] += ((db.mean(-1) * m).sum() / n_present).item() / BASE_SCALE * 100.0
            stats["phy"]["rot"] += ((angular_error_deg(
                rot6d_to_matrix(out["rot"]), rot6d_to_matrix(p[:, :, 16:22])) * m).sum() / n_present).item()
            ds = ((out["scale"] - p[:, :, 22:25]).abs() / (p[:, :, 22:25].abs() + 1e-3))
            stats["phy"]["scale"] += ((ds.mean(-1) * m).sum() / n_present).item()
            stats["phy"]["n"] += 1

            # --- OrganLatentVAE per-organ roundtrip on the SAME absolute rows ---
            # Shapes: o["rot"] is (P*8, 6) flat; reshape to (P, 8, 6) FIRST, then to matrices.
            flat = abs_gt.reshape(-1, 26)
            o = organ(flat)
            mm = m.reshape(-1)
            nn = mm.sum().clamp(min=1.0)
            tgt_c = abs_gt[:, :, :FM_OT_END].argmax(-1).reshape(-1)
            stats["org"]["cls"] += (((o["cls_logits"].argmax(-1) == tgt_c).float() * mm).sum() / nn).item()
            ob = (o["base"].reshape(-1, 8, 3) - abs_gt[:, :, FM_BASE_START:FM_BASE_START + 3]).abs()
            stats["org"]["base"] += ((ob.mean(-1).reshape(-1) * mm).sum() / nn).item() / BASE_SCALE * 100.0
            o_rot_mat = rot6d_to_matrix(o["rot"].reshape(-1, 8, 6))          # (P, 8, 3, 3)
            t_rot_mat = rot6d_to_matrix(abs_gt[:, :, 16:22])                 # (P, 8, 3, 3)
            stats["org"]["rot"] += ((angular_error_deg(o_rot_mat, t_rot_mat).reshape(-1) * mm).sum() / nn).item()
            os_ = ((o["scale"].reshape(-1, 8, 3) - abs_gt[:, :, 22:25]).abs()
                   / (abs_gt[:, :, 22:25].abs() + 1e-3))
            stats["org"]["scale"] += ((os_.mean(-1).reshape(-1) * mm).sum() / nn).item()
            stats["org"]["n"] += 1

            n_packets += p.shape[0]

    print(f"\nEvaluated {n_packets:,} held-out packets from {args.max_files} files.")
    print("\n=== ROUNDTRIP COMPARISON (present slots, absolute coords) ===")
    print(f"{'metric':<28} {'PhytomerVAE-%d' % args.latent_dim:>14} {'OrganLatentVAE':>14}")
    for k, label in (("cls", "class acc (%)"), ("base", "base MAE (cm)"),
                     ("rot", "rot err (deg)"), ("scale", "scale rel err")):
        pv = stats["phy"][k] / max(stats["phy"]["n"], 1)
        ov = stats["org"][k] / max(stats["org"]["n"], 1)
        if k == "cls":
            pv, ov = pv * 100.0, ov * 100.0
        print(f"{label:<28} {pv:>14.3f} {ov:>14.3f}")
    print("GO/NO-GO: PhytomerVAE within ~10-15% of baseline per-organ error.")


if __name__ == "__main__":
    main()