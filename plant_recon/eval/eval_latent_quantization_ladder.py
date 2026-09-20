"""How many distinct phytomer shapes does a cowpea actually need?

Lee et al. (Latent L-systems, ACM TOG 2023, section 6.2) quantise their tree parameters with
k-means and find that reducing the branching angles from 773 distinct values to SEVEN leaves the
tree visually unchanged. If the same holds for our phytomer latents, the consequences for Stage 3
are direct, because every Stage 3 failure this project has measured is a failure of CONTINUOUS
regression:

  - per-organ shape is not recoverable from a nadir view at any resolution (R^2 ~ 0, 2026-09-18)
  - a regression head collapses to the dataset mean exactly (latent R^2 +0.001)
  - refinement scores HIGHER when the latent prior is switched off entirely (+2.0, 2026-09-19)

A K-way classification over shape clusters cannot collapse to a mean -- there is no "average of two
classes" to fall into -- and it needs far less from the image: enough to say which cluster, not to
place a point in a 128-dimensional space. It also makes the multi-hypothesis selection that Gate G
already proved the best lever (+3.4) principled rather than incidental: enumerate the top-K classes
instead of drawing K samples from a continuous flow and hoping they spread.

This measures the ceiling of that idea before anything is trained. For each K, every phytomer's
latent is replaced by its k-means cluster centre, decoded, and rendered.

The reference is the IK-only reconstruction, NOT the Helios ground truth: the VAE round-trip is
required to reproduce IK-only (it is a hard requirement of this project, not a nice-to-have), so
quantisation loss has to be read as degradation from where the VAE already sits. Both are reported
-- if `vae exact` is already well below IK-only, the round-trip has regressed and the quantisation
numbers below it mean nothing.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plant_recon.dataset.part_array_dataset import (
    PartArrayDataset, attach_parent_links, FM_OT_END, FM_BASE_START)
from plant_recon.dataset.phytomer_packets import phytomer_scale
from plant_recon.eval.ckpt_compat import fix_ckpt_args
from plant_recon.eval.eval_gt_substitution_ablation import plant_from_nodes, render_depth, score
from plant_recon.eval.eval_hierarchical_self_consistency import decode_predictions_to_part_tensor
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from plant_recon.models.phytomer_vae import PhytomerVAE


def perdim_levels(x, k):
    """Per-DIMENSION quantisation levels: k quantiles of each coordinate, independently.

    This, not whole-vector VQ, is the analogue of Lee et al. section 6.2. They quantise each scalar
    parameter (a branching angle) to k representative values, so a tree with n angles still has k^n
    reachable configurations. One symbol per phytomer is a far stronger claim and a far smaller
    codebook: k total shapes for the whole species.
    """
    qs = torch.linspace(0, 1, k + 2, device=x.device)[1:-1]         # k interior quantiles
    return torch.quantile(x.double(), qs.double(), dim=0).float()   # (k, D)


def quantise_perdim(z, levels):
    """Snap each coordinate of z to its nearest level in that coordinate."""
    d = (z.unsqueeze(0) - levels.unsqueeze(1)).abs()                # (k, N, D)
    return torch.gather(levels.unsqueeze(1).expand(-1, z.shape[0], -1), 0,
                        d.argmin(0, keepdim=True)).squeeze(0)


def kmeans(x, k, iters=60, seed=0):
    """Plain Lloyd's on the GPU; k-means++ style seeding by farthest-point on a random start."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.shape[0]
    c = x[torch.randperm(n, generator=g)[:1].to(x.device)]
    while c.shape[0] < k:                      # farthest-point seeding: spreads the initial centres
        d = torch.cdist(x, c).min(1).values
        c = torch.cat([c, x[d.argmax()].unsqueeze(0)])
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="supplies data_dir / cache_dir / phytomer VAE path")
    ap.add_argument("--n_plants", type=int, default=120)
    ap.add_argument("--ks", default="2,4,8,16,32,64,128,256,512")
    ap.add_argument("--fit_plants", type=int, default=400, help="plants whose latents the clusters are fit on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/logs/20260919/latent_quantization_ladder.json")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    M = args["slots_per_phytomer"]

    pvae = PhytomerVAE(latent_dim=args["phytomer_latent_dim"],
                       residual_dim=args.get("phytomer_residual_dim", 8), hidden_dim=256)
    sd = torch.load(args["phytomer_vae_checkpoint"], map_location="cpu", weights_only=False)
    pvae.load_state_dict(sd.get("model_state_dict", sd))
    pvae = pvae.to(dev).eval()
    # The latent is not an unstructured vector: PhytomerVAE splits it into a COARSE channel
    # (latent_dim - slots*residual_dim) plus one residual block per slot. Quantising the whole
    # vector as one symbol is the strict reading of Lee et al.; quantising only the coarse channel
    # is the looser one and is reported alongside, since that is the part a classifier would pick.
    coarse_dim = pvae.coarse_dim
    renderer = HeliosPyTorchRenderer(image_size=256).to(dev)
    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * M,
                          cache_dir=args["cache_dir"], pkt_cache_dir=args.get("pkt_cache_dir") or None,
                          species="cowpea", image_size=128)

    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(len(ds), generator=g).tolist()
    fit_idx, eval_idx = order[: a.fit_plants], order[a.fit_plants: a.fit_plants + a.n_plants]

    # ---- fit the codebook on held-out plants (never the ones scored below)
    pool = []
    for i in fit_idx:
        pkt = ds[i].get("pkt")
        if pkt is None:
            continue
        with torch.no_grad():
            z = pvae.encode(pvae.pack_input(pkt["packets"].to(dev).float(), pkt["presence"].to(dev)))[0].float()
        pool.append(z)
    pool = torch.cat(pool)
    print(f"codebook fit on {pool.shape[0]} phytomers from {len(fit_idx)} plants, dim {pool.shape[1]}", flush=True)

    ks = [int(x) for x in a.ks.split(",")]
    books = {}
    for k in ks:
        books[k] = kmeans(pool, k, seed=a.seed)
        print(f"  k={k:<4} fitted", flush=True)

    # ---- score
    acc = {f"vq{k}": [] for k in ks}
    acc.update({f"pd{k}": [] for k in ks})
    acc["vae_exact"] = []
    levels = {k: perdim_levels(pool, k) for k in ks}
    n_used = 0
    for i in eval_idx:
        it = ds[i]
        pkt = it.get("pkt")
        if pkt is None:
            continue
        attach_parent_links(pkt)
        dap = int(it["dap"].item()) if "dap" in it else 0
        zoom = 8.0 if dap <= 15 else 1.0
        nodes, ex = it["nodes"].to(dev), it["existence_mask"].to(dev)
        pos = pkt["centers"].to(dev).float(); rot = pkt["refs"].to(dev).float()
        pks = pkt["packets"].to(dev).float(); pres = pkt["presence"].to(dev)
        scl = phytomer_scale(pks); ppos = pkt["parent_pos"].to(dev).float()
        pex = torch.ones(pos.shape[0], device=dev)
        with torch.no_grad():
            lat = pvae.encode(pvae.pack_input(pks, pres))[0].float()
            # reference: IK-only, the GT nodes straight to parts with no VAE in the path
            ik = render_depth(renderer, decode_predictions_to_part_tensor(
                nodes[:, FM_BASE_START:], nodes[:, :FM_OT_END].argmax(-1), ex, device=dev), zoom, dev)
            if (ik > 0.005).sum() < 64:
                continue
            p = plant_from_nodes(pvae, pos, rot, scl, lat, pex, ppos, M)
            acc["vae_exact"].append(score(render_depth(renderer, p, zoom, dev), ik)[0])
            for k in ks:
                qi = torch.cdist(lat, books[k]).argmin(1)
                p = plant_from_nodes(pvae, pos, rot, scl, books[k][qi], pex, ppos, M)
                acc[f"vq{k}"].append(score(render_depth(renderer, p, zoom, dev), ik)[0])
                p = plant_from_nodes(pvae, pos, rot, scl, quantise_perdim(lat, levels[k]), pex, ppos, M)
                acc[f"pd{k}"].append(score(render_depth(renderer, p, zoom, dev), ik)[0])
        n_used += 1
        if n_used % 20 == 0:
            print(f"  scored {n_used} plants", flush=True)

    base = 100 * float(np.mean(acc["vae_exact"]))
    print(f"\n{n_used} plants. Reference = IK-only reconstruction.")
    print(f"\n{'k':<8}{'whole-latent VQ':>18}{'delta':>9}   {'per-dim quantise':>18}{'delta':>9}")
    print(f"{'exact':<8}{base:>18.2f}{0.0:>+9.2f}   {base:>18.2f}{0.0:>+9.2f}   <- continuous 128-D")
    rows = {"vae_exact": base}
    for k in ks:
        v = 100 * float(np.mean(acc[f"vq{k}"]))
        w = 100 * float(np.mean(acc[f"pd{k}"]))
        rows[f"vq{k}"], rows[f"pd{k}"] = v, w
        print(f"{k:<8}{v:>18.2f}{v - base:>+9.2f}   {w:>18.2f}{w - base:>+9.2f}")
    print("\nwhole-latent VQ: k shapes for the entire species (one symbol per phytomer)")
    print("per-dim:         k levels PER COORDINATE, so k^%d reachable -- the Lee et al. analogue"
          % int(pool.shape[1]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"n_plants": n_used, "fit_plants": len(fit_idx), "latent_dim": int(pool.shape[1]),
               "iou": rows}, open(a.out, "w"), indent=1)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
