"""
Which of the 64 latent dimensions actually carry information, and which ones
visibly change the decoded shape?

The GUI ranks slider dimensions by std(mu), which cannot tell a collapsed
dimension (KL ~ 0, posterior variance ~ 1) from one that is merely
low-variance but used. This script measures both the information-theoretic
answer (per-dimension KL to the prior) and the causal one (perturb a single
dimension, decode, and measure how far the organs actually move).

Writes per-dimension KL to <checkpoint_dir>/latent_usage.json so
precompute_phytomer_latent_pca.py and the GUI can rank sliders by it.

Sampling note: the packet cache is alphabetically DAP-sorted, so taking the
first N files yields only DAP001 seedlings. Files are sampled at random.
"""

import os
import sys
import glob
import json
import random
import argparse

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.dataset.phytomer_packets import (
    rot6d_to_matrix, FM_ROT_START, FM_ROT_END, FM_SCALE_START, FM_SCALE_END,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT = os.path.join(
    REPO_ROOT, "diffusion_based/checkpoints/phytomer_vae_v8/phytomer_vae_128d_best.pt")
DEFAULT_PKT_DIR = os.path.join(REPO_ROOT, "dataset/cache/cowpea_curv26_pkt")


def load_packets(pkt_dir: str, max_packets: int, seed: int):
    files = sorted(glob.glob(os.path.join(pkt_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"no packet files under {pkt_dir}")
    random.Random(seed).shuffle(files)
    packets, presence, n = [], [], 0
    for f in files:
        if n >= max_packets:
            break
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not (isinstance(d, dict) and "packets" in d):
            continue
        take = min(d["packets"].shape[0], max_packets - n)
        if take <= 0:
            continue
        packets.append(d["packets"][:take].float())
        presence.append(d["presence"][:take])
        n += take
    return torch.cat(packets), torch.cat(presence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae_checkpoint", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--pkt_dir", type=str, default=DEFAULT_PKT_DIR)
    ap.add_argument("--max_packets", type=int, default=60000)
    ap.add_argument("--sens_packets", type=int, default=2000,
                    help="packets used for the per-dimension perturbation test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_write", action="store_true",
                    help="skip writing latent_usage.json next to the checkpoint")
    args = ap.parse_args()

    vae = PhytomerVAE(latent_dim=128, hidden_dim=256).to(DEVICE)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=DEVICE, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    D = vae.latent_dim

    packets, presence = load_packets(args.pkt_dir, args.max_packets, args.seed)
    print("=" * 78)
    print("PHYTOMER VAE LATENT USAGE")
    print(f"checkpoint : {args.vae_checkpoint}")
    print(f"packets    : {packets.shape[0]:,} (randomly sampled across DAPs)")
    print("=" * 78)

    mus, logvars = [], []
    with torch.no_grad():
        for i in range(0, packets.shape[0], 8192):
            p = packets[i:i + 8192].to(DEVICE)
            pr = presence[i:i + 8192].to(DEVICE)
            mu, logvar = vae.encode(vae.pack_input(p, pr))
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
    mu = torch.cat(mus)
    logvar = torch.cat(logvars)

    # Per-dimension KL to N(0, I), averaged over packets.
    kl_d = (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)).mean(dim=0)
    order = torch.argsort(kl_d, descending=True)
    cum = kl_d[order].cumsum(0) / kl_d.sum()

    print(f"\nTotal KL: {kl_d.sum():.2f} nats/packet")
    for thr in (0.01, 0.1, 1.0):
        print(f"  dims with KL > {thr:<5}: {(kl_d > thr).sum().item():3d} / {D}")
    for frac in (0.5, 0.8, 0.9, 0.95, 0.99):
        print(f"  dims carrying {int(frac*100):2d}% of KL: "
              f"{int((cum < frac).sum().item()) + 1:3d} / {D}")

    std_mu = mu.std(dim=0)
    post_var = logvar.exp().mean(dim=0)
    print(f"\nstd(mu)      : min {std_mu.min():.3f}  median {std_mu.median():.3f}  max {std_mu.max():.3f}")
    print(f"E[post var]  : min {post_var.min():.3f}  median {post_var.median():.3f}  max {post_var.max():.3f}"
          "   (~1.0 = collapsed to the prior)")
    print(f"aggregate mu : std {mu.std():.3f}  max|mu| {mu.abs().max():.2f}"
          "   (the flow stage assumes N(0,I) — see plan 2.3)")

    # Causal sensitivity: move one dimension by +2 sigma and see how far the
    # decoded organs actually move. std(mu) cannot answer this.
    ns = min(args.sens_packets, mu.shape[0])
    z0 = mu[:ns].to(DEVICE)
    with torch.no_grad():
        base = vae.decode(z0)
        R0 = rot6d_to_matrix(base["recon_packets"][:, :, FM_ROT_START:FM_ROT_END])
        s0 = base["recon_packets"][:, :, FM_SCALE_START:FM_SCALE_END]
        rot_sens = torch.zeros(D)
        scale_sens = torch.zeros(D)
        for d in range(D):
            zp = z0.clone()
            zp[:, d] += 2.0 * std_mu[d].to(DEVICE)
            out = vae.decode(zp)
            R1 = rot6d_to_matrix(out["recon_packets"][:, :, FM_ROT_START:FM_ROT_END])
            rel = R0.transpose(-1, -2) @ R1
            ang = torch.rad2deg(torch.acos(
                (((rel.diagonal(dim1=-2, dim2=-1).sum(-1)) - 1) / 2).clamp(-1, 1)))
            rot_sens[d] = ang.mean()
            s1 = out["recon_packets"][:, :, FM_SCALE_START:FM_SCALE_END]
            scale_sens[d] = ((s1 - s0).abs() / s0.abs().clamp(min=1e-3)).mean()

    sens_order = torch.argsort(rot_sens, descending=True)
    print(f"\nPer-dimension effect of a +2σ nudge ({ns:,} packets):")
    print(f"  {'rank':>4} {'dim':>4} {'KL':>8} {'std(mu)':>9} {'E[var]':>8} "
          f"{'Δrot(deg)':>10} {'Δscale':>8}")
    for r, d in enumerate(sens_order[:15].tolist()):
        print(f"  {r:>4} {d:>4} {kl_d[d]:>8.3f} {std_mu[d]:>9.3f} {post_var[d]:>8.3f} "
              f"{rot_sens[d]:>10.2f} {scale_sens[d]:>8.3f}")
    print("  ...")
    for r, d in enumerate(sens_order[-5:].tolist(), start=D - 5):
        print(f"  {r:>4} {d:>4} {kl_d[d]:>8.3f} {std_mu[d]:>9.3f} {post_var[d]:>8.3f} "
              f"{rot_sens[d]:>10.2f} {scale_sens[d]:>8.3f}")

    dead = (rot_sens < 0.5) & (scale_sens < 0.01)
    print(f"\ndims moving nothing (<0.5 deg and <1% scale): {int(dead.sum())} / {D}")
    print("KL rank vs sensitivity rank agreement (Spearman): "
          f"{torch.corrcoef(torch.stack([kl_d.argsort().argsort().float(), rot_sens.argsort().argsort().float()]))[0,1]:.3f}")

    if not args.no_write:
        out_path = os.path.join(os.path.dirname(args.vae_checkpoint), "latent_usage.json")
        with open(out_path, "w") as f:
            json.dump({
                "checkpoint": args.vae_checkpoint,
                "n_packets": int(packets.shape[0]),
                "kl_per_dim": kl_d.tolist(),
                "std_mu_per_dim": std_mu.tolist(),
                "posterior_var_per_dim": post_var.tolist(),
                "rot_sensitivity_deg_per_dim": rot_sens.tolist(),
                "scale_sensitivity_per_dim": scale_sens.tolist(),
                "mu_mean_per_dim": mu.mean(dim=0).tolist(),
            }, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
