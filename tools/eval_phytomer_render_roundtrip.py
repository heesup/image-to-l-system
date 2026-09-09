"""Render roundtrip comparison: does PhytomerVAE angle error matter for the task?

Reconstructs full plants three ways and compares silhouette IoU / depth MAE
against GT renders:
  1. GT 14D part rows (reference)
  2. OrganLatentVAE per-organ roundtrip (baseline)
  3. PhytomerVAE joint roundtrip (packets -> z -> packets -> re-anchor)

If (3) ≈ (2) on IoU/depth, the 7-8 deg rotation gap is task-irrelevant and the
integration gate passes on render fidelity.

Usage:
    .../bin/python tools/eval_phytomer_render_roundtrip.py \
        --phytomer-ckpt .../phytomer_vae_64d_best.pt \
        --organ-ckpt .../organ_latent_vae_best.pt \
        --max-plants 12 --seed 7
"""

import argparse
import glob
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.models.organ_latent_vae import OrganLatentVAE
from diffusion_based.dataset.phytomer_packets import build_phytomer_packets, decode_packets, assemble_packets
from diffusion_based.dataset.part_array_dataset import decode_fm
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def render_silhouette_depth(renderer, part_14d, device):
    mesh = renderer.geo_builder.build_mesh_from_part_tensor(part_14d, device=device)
    out = renderer.forward(
        mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
        background="ground", focus_plant=True, include_depth=True,
        zoom_factor=1.0, reference_window_size=1.2,
    )
    rgb = out[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    depth = out[3].clamp(min=0.0).cpu().numpy()
    return rgb, depth


def metrics_vs_gt(pred_depth, gt_depth):
    pred_mask = pred_depth > 0.005
    gt_mask = gt_depth > 0.005
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = float(inter / max(union, 1)) * 100.0
    canopy = np.logical_or(pred_mask, gt_mask)
    dmae = float(np.abs(pred_depth[canopy] - gt_depth[canopy]).mean()) * 100.0 if canopy.any() else 0.0
    return iou, dmae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phytomer-ckpt", type=str, required=True)
    parser.add_argument("--organ-ckpt", type=str,
                        default="diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt")
    parser.add_argument("--cache-dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--rot-branch", action="store_true")
    parser.add_argument("--max-plants", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    files = rng.sample(sorted(glob.glob(os.path.join(args.cache_dir, "*.pt"))), args.max_plants)

    phy = PhytomerVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    phy.load_state_dict(torch.load(args.phytomer_ckpt, map_location=device, weights_only=True))
    phy.eval()
    organ = OrganLatentVAE(latent_dim=16, hidden_dim=256).to(device)
    organ.load_state_dict(torch.load(args.organ_ckpt, map_location=device, weights_only=True))
    organ.eval()
    renderer = HeliosPyTorchRenderer(image_size=256).to(device)

    res = {"phy_iou": [], "phy_dmae": [], "org_iou": [], "org_dmae": []}
    with torch.no_grad():
        for f in files:
            try:
                data = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            nodes = data["nodes"]
            packets, presence, centers, ref_rots = build_phytomer_packets(
                nodes, data.get("existence_mask"),
                phytomer_ids=data.get("phytomer_ids"))
            if packets.shape[0] == 0:
                continue
            dap = float(data.get("dap", -1))
            P = packets.shape[0]
            p = packets.to(device)
            pr = presence.to(device)
            refs = ref_rots.to(device)

            # GT 14D rows on the PACKETED organ set (fair comparison set for both).
            abs_gt = decode_packets(p, centers.to(device), pr, refs).reshape(P, 8, 26)
            gt_14d = decode_fm(abs_gt.reshape(-1, 26)[pr.reshape(-1)])
            _, gt_depth = render_silhouette_depth(renderer, gt_14d, device)

            # PhytomerVAE roundtrip -> absolute 14D rows
            out = phy(p, pr, use_rot_branch=args.rot_branch)
            rec_assembled = assemble_packets(out["recon_packets"], refs)
            abs_rows = decode_packets(rec_assembled, centers.to(device), pr, refs)
            keep = out["cls_logits"].argmax(-1).reshape(-1) > 0
            phy_14d = decode_fm(abs_rows.reshape(-1, 26)[keep])
            _, phy_depth = render_silhouette_depth(renderer, phy_14d, device)
            pi, pd = metrics_vs_gt(phy_depth, gt_depth)
            res["phy_iou"].append(pi)
            res["phy_dmae"].append(pd)

            # OrganLatentVAE roundtrip on the same absolute rows
            flat = nodes[act].to(device)
            o = organ(flat)
            org_14d = decode_fm(o["recon_26d"])
            _, org_depth = render_silhouette_depth(renderer, org_14d, device)
            oi, od = metrics_vs_gt(org_depth, gt_depth)
            res["org_iou"].append(oi)
            res["org_dmae"].append(od)

            print(f"DAP {dap:5.0f} | phy IoU {pi:5.1f}% dmae {pd:5.2f}cm | "
                  f"org IoU {oi:5.1f}% dmae {od:5.2f}cm")

    print("\n=== RENDER ROUNDTRIP (vs GT mesh) ===")
    print(f"PhytomerVAE: mean IoU {np.mean(res['phy_iou']):.1f}% | depth MAE {np.mean(res['phy_dmae']):.2f} cm")
    print(f"OrganVAE:    mean IoU {np.mean(res['org_iou']):.1f}% | depth MAE {np.mean(res['org_dmae']):.2f} cm")


if __name__ == "__main__":
    main()
