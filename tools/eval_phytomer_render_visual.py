"""Renders a roundtrip comparison figure: GT vs PhytomerVAE vs OrganLatentVAE.

For N sample plants (diverse DAP), builds a figure with rows = plants,
columns = [GT RGB | GT Depth | PhytomerVAE RGB | PhytomerVAE Depth | OrganVAE RGB | OrganVAE Depth],
each annotated with the reconstruction IoU vs GT.

Accepted checkpoint = PhytomerVAE (Option 1: ~6 deg rotation, compute win).
Baseline = frozen OrganLatentVAE (per-organ 16D).

Usage:
    .../bin/python tools/eval_phytomer_render_visual.py \
        --phytomer-ckpt diffusion_based/checkpoints/phytomer_vae_expd/phytomer_vae_64d_best.pt \
        --rot-branch --max-plants 4 --seed 3 --out docs/results/assets/fig_phytomer_vae_roundtrip.png
"""

import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.models.organ_latent_vae import OrganLatentVAE
from diffusion_based.dataset.phytomer_packets import build_phytomer_packets, decode_packets, assemble_packets
from diffusion_based.dataset.part_array_dataset import decode_fm, FM_OT_END
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def render_rgbd(renderer, part_14d, device):
    mesh = renderer.geo_builder.build_mesh_from_part_tensor(part_14d, device=device)
    out = renderer.forward(
        mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
        background="ground", focus_plant=True, include_depth=True,
        zoom_factor=1.0, reference_window_size=1.2,
    )
    rgb = out[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    depth = out[3].clamp(min=0.0).cpu().numpy()
    return rgb, depth


def iou_vs(gt_depth, pred_depth):
    a = gt_depth > 0.005
    b = pred_depth > 0.005
    return float((a & b).sum() / max((a | b).sum(), 1)) * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phytomer-ckpt", type=str, required=True)
    parser.add_argument("--organ-ckpt", type=str,
                        default="diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt")
    parser.add_argument("--cache-dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--rot-branch", action="store_true", default=True)
    parser.add_argument("--no-rot-branch", dest="rot_branch", action="store_false")
    parser.add_argument("--max-plants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", type=str,
                        default="docs/results/assets/fig_phytomer_vae_roundtrip.png")
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

    rows = []
    with torch.no_grad():
        for f in files:
            try:
                data = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not (isinstance(data, dict) and "nodes" in data):
                continue
            dap = float(data.get("dap", -1))
            packets, presence, centers, ref_rots = build_phytomer_packets(
                data["nodes"], data.get("existence_mask"),
                phytomer_ids=data.get("phytomer_ids"))
            if packets.shape[0] == 0:
                continue
            P = packets.shape[0]
            p = packets.to(device)
            pr = presence.to(device)
            refs = ref_rots.to(device)

            # GT on the PACKETED organ set (fair comparison set for both models).
            abs_gt = decode_packets(p, centers.to(device), pr, refs).reshape(P, 8, 26)
            gt_14d = decode_fm(abs_gt.reshape(-1, 26)[pr.reshape(-1)])
            gt_rgb, gt_depth = render_rgbd(renderer, gt_14d, device)

            # PhytomerVAE joint roundtrip.
            out = phy(p, pr, use_rot_branch=args.rot_branch)
            keep = out["cls_logits"].argmax(-1).reshape(-1) > 0
            rec_assembled = assemble_packets(out["recon_packets"], refs)
            abs_hat = decode_packets(rec_assembled, centers.to(device), pr, refs).reshape(P, 8, 26)
            phy_14d = decode_fm(abs_hat.reshape(-1, 26)[keep])
            phy_rgb, phy_depth = render_rgbd(renderer, phy_14d, device)

            # OrganLatentVAE per-organ roundtrip on the same GT rows.
            o = organ(abs_gt.reshape(-1, 26))
            org_14d = decode_fm(o["recon_26d"])
            org_rgb, org_depth = render_rgbd(renderer, org_14d, device)

            rows.append((dap, gt_rgb, gt_depth, phy_rgb, phy_depth,
                         org_rgb, org_depth,
                         iou_vs(gt_depth, phy_depth), iou_vs(gt_depth, org_depth)))
            print(f"DAP {dap:5.0f} | phy IoU {rows[-1][7]:5.1f}% | org IoU {rows[-1][8]:5.1f}%")

    if not rows:
        print("No plants rendered.")
        return 1

    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(24, 4.2 * n))
    fig.patch.set_facecolor("#10131a")
    if n == 1:
        axes = axes[None, :]
    cols = ["GT RGB", "GT Depth", "Phytomer RGB", "Phytomer Depth",
            "OrganVAE RGB", "OrganVAE Depth"]
    for c, t in enumerate(cols):
        axes[0, c].set_title(t, color="#ced4da", fontsize=12, fontweight="bold")
    for r, (dap, gt_rgb, gt_depth, phy_rgb, phy_depth, org_rgb, org_depth,
            pi, oi) in enumerate(rows):
        axes[r, 0].imshow(gt_rgb); axes[r, 0].axis("off")
        axes[r, 1].imshow(gt_depth, cmap="viridis", vmin=0, vmax=0.6); axes[r, 1].axis("off")
        axes[r, 2].imshow(phy_rgb)
        axes[r, 2].set_title(f"IoU {pi:.1f}%", color="#ffd166", fontsize=10)
        axes[r, 2].axis("off")
        axes[r, 3].imshow(phy_depth, cmap="viridis", vmin=0, vmax=0.6); axes[r, 3].axis("off")
        axes[r, 4].imshow(org_rgb)
        axes[r, 4].set_title(f"IoU {oi:.1f}%", color="#06d6a0", fontsize=10)
        axes[r, 4].axis("off")
        axes[r, 5].imshow(org_depth, cmap="viridis", vmin=0, vmax=0.6); axes[r, 5].axis("off")
        axes[r, 0].set_ylabel(f"DAP {dap:.0f}", color="#ced4da", fontsize=12, fontweight="bold")
    fig.suptitle(
        "PhytomerVAE-64 Roundtrip (Option 1: accepted, ~6\u00b0 rotation) vs OrganLatentVAE baseline\n"
        "Top-view; IoU vs GT mesh on the packeted organ set",
        color="white", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved figure to {args.out}")

    print("\n=== SUMMARY ===")
    print(f"PhytomerVAE-64: mean IoU {np.mean([r[7] for r in rows]):.1f}%")
    print(f"OrganLatentVAE: mean IoU {np.mean([r[8] for r in rows]):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
