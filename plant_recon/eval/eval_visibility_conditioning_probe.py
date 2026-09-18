"""Does hiding the occluded organs change what the image can tell us about them?

Stage 3's per-node latent sits at the dataset mean (R^2 -0.064, and +0.001 even with ground-truth
node positions). Two explanations have very different fixes:

  (a) the conditioning is too coarse -- one DINOv2 token spans 75 mm and covers 7-15 phytomers, so
      the feature is shared and the regression target is their average;
  (b) most of the targets are unanswerable -- 53.4% of ground-truth organs contribute ZERO pixels
      (measured 2026-09-18 from the rasterizer's triangle ids), so for the occluded majority the
      conditionally-correct answer IS the mean and their gradient drags the shared feature there.

If (b) dominates, restricting the probe to VISIBLE organs should lift R^2 sharply, and a
visibility-weighted loss is worth building. If R^2 stays flat on visible organs too, the limit is
(a) and no amount of loss masking rescues it -- finer conditioning is the answer instead.

Three targets, because masking is defensible for some and not others:
  latent     the organ's shape. Unobservable when occluded -- the natural thing to mask.
  position   where the organ is. Still matters when occluded: it holds up the canopy and occludes
             other organs, so masking it could break the scaffold, which is currently the healthiest
             part of the model. Measured rather than assumed.
  existence  is there an organ here at all, against sampled empty locations.

Also reports RETRIEVAL accuracy alongside R^2. R^2 asks whether the conditional MEAN is recoverable;
retrieval asks whether the feature can pick its own organ out of the other organs of the same plant
-- the question a contrastive (InfoNCE) alignment between ViT tokens and phytomer latents would
actually optimise. The two can disagree: if a token is equally compatible with two very different
shapes, the conditional mean is their average and R^2 is ~0 while the feature still carries real
information. That gap is precisely the case where a contrastive objective beats regression.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
import torch.nn.functional as F

from plant_recon.dataset.part_array_dataset import PartArrayDataset, FM_OT_END, FM_BASE_START
from plant_recon.eval.ckpt_compat import fix_ckpt_args
from plant_recon.eval.eval_gt_substitution_ablation import decode_predictions_to_part_tensor
from plant_recon.eval.eval_latent_conditioning_ceiling import (
    CACHE_CAMERA_HEIGHT, CACHE_WINDOW_M, load_backbone, project, ridge_r2, token_features,
)
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer, compute_focus_plant_camera
from plant_recon.models.organ_latent_vae import OrganLatentVAE


def ridge_fit(xtr, ytr, alpha=1.0):
    x = torch.cat([xtr, torch.ones(len(xtr), 1)], 1).double()
    reg = alpha * torch.eye(x.shape[1], dtype=torch.float64)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(x.T @ x + reg, x.T @ ytr.double())


def retrieval_top1(xtr, ytr, xte, yte, plant_id, alpha=1.0):
    """Within each plant, can a node's feature pick its OWN target out of that plant's other nodes?

    Measured through a learned linear map, not by comparing raw spaces: DINOv2 features and VAE
    latents share no geometry, so a direct cosine between them is meaningless (an earlier version of
    this function did that and scored below chance, which is how the mistake surfaced). The map is
    fitted on the training plants, then each test node's PREDICTED target is ranked against the actual
    targets of the other nodes of the same plant.

    This is the quantity a contrastive (InfoNCE) alignment optimises, and it can succeed where R^2
    fails: if a token is equally compatible with two very different shapes, the conditional mean is
    their average -- R^2 ~ 0 -- while the feature still discriminates against a third shape. Chance is
    1/n for each plant, averaged and reported alongside.
    """
    w = ridge_fit(xtr, ytr, alpha)
    pred = (torch.cat([xte, torch.ones(len(xte), 1)], 1).double() @ w).float()
    pf = F.normalize(pred - pred.mean(0), dim=1)
    tf = F.normalize(yte - yte.mean(0), dim=1)
    hits = tot = 0
    chance = []
    for p in plant_id.unique():
        m = plant_id == p
        n = int(m.sum())
        if n < 2:
            continue
        sim = pf[m] @ tf[m].T
        hits += int((sim.argmax(1) == torch.arange(n)).sum())
        tot += n
        chance.append(1.0 / n)
    return (hits / max(tot, 1)), float(np.mean(chance)) if chance else float("nan"), tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="supplies data_dir / cache_dir / organ VAE paths")
    ap.add_argument("--n_plants", type=int, default=80)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--render_px", type=int, default=512, help="render used for visibility and token sampling")
    ap.add_argument("--backbone", default="dinov2_vits14@224")
    ap.add_argument("--max_nodes_per_plant", type=int, default=150)
    ap.add_argument("--min_visible_px", type=int, default=1, help="an organ counts as visible at or above this")
    ap.add_argument("--out", default="outputs/logs/20260918/visibility_probe.json")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))

    ovae = OrganLatentVAE(latent_dim=args["node_dim"], hidden_dim=256)
    sd = torch.load(args["organ_vae_checkpoint"], map_location="cpu", weights_only=False)
    ovae.load_state_dict(sd.get("model_state_dict", sd))
    ovae = ovae.to(dev).eval()

    backbone, kind, px = load_backbone(a.backbone, dev)
    renderer = HeliosPyTorchRenderer(image_size=a.render_px).to(dev)
    renderer.collect_part_visibility = True

    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * args["slots_per_phytomer"],
                          cache_dir=args["cache_dir"], species="cowpea", image_size=128)
    g = torch.Generator().manual_seed(0)
    picks = torch.randperm(len(ds), generator=g)[: a.n_plants].tolist()

    FEAT, LAT, POS, VIS, PID, DAP = [], [], [], [], [], []
    for pi, idx in enumerate(picks):
        it = ds[idx]
        nodes = it["nodes"].to(dev)
        ex = it["existence_mask"].to(dev)
        types = nodes[:, :FM_OT_END].argmax(dim=-1)
        parts = decode_predictions_to_part_tensor(nodes[:, FM_BASE_START:], types, ex, device=dev)
        if parts.shape[0] < 4:
            continue
        mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=dev)
        v = mesh["vertices"]
        centre = 0.5 * (v.min(0).values + v.max(0).values)
        view, proj, _ = compute_focus_plant_camera(v, None, azimuth_deg=0.0, elevation_deg=90.0,
                                                   camera_height=CACHE_CAMERA_HEIGHT, focus_plant=False,
                                                   reference_window_size=CACHE_WINDOW_M, zoom_factor=1.0,
                                                   center_override=centre)
        with torch.no_grad():
            rgbd = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0,
                                    camera_height=CACHE_CAMERA_HEIGHT, background="ground",
                                    focus_plant=False, include_depth=True, image_size=a.render_px,
                                    zoom_factor=1.0, reference_window_size=CACHE_WINDOW_M,
                                    center_override=centre)
        vis = renderer.last_part_visibility
        n = parts.shape[0]
        if vis.numel() < n:
            vis = torch.cat([vis, vis.new_zeros(n - vis.numel())])
        vis = vis[:n].cpu()

        # The organ VAE encodes the raw 26D node vector ([13 class | 3 base | 6 rot | 3 scale | 1 curv]),
        # which is what training feeds it -- not the decoded 14D part tensor.
        act = ex > 0.5
        nodes_act = nodes[act].float()
        if nodes_act.shape[0] != parts.shape[0]:
            continue  # decode dropped or merged organs; alignment with visibility would be wrong
        with torch.no_grad():
            lat = ovae.encode(nodes_act)
            if isinstance(lat, (tuple, list)):
                lat = lat[0]
        base = nodes_act[:, FM_BASE_START:FM_BASE_START + 3]
        uv = project(base, view, proj, a.render_px)
        uvn = torch.stack([uv[:, 0] / a.render_px * 2 - 1, uv[:, 1] / a.render_px * 2 - 1], -1)
        rgb = rgbd[:3]  # spatial_map adds the batch dim itself
        with torch.no_grad():
            feat = token_features(backbone, kind, rgb, uvn, px, dev)

        keep = torch.arange(n)[: a.max_nodes_per_plant]
        FEAT.append(feat[keep]); LAT.append(lat[keep].float().cpu()); POS.append(base[keep].float().cpu())
        VIS.append(vis[keep]); PID.append(torch.full((len(keep),), pi)); DAP.append(int(it["dap"].item()))
        if (pi + 1) % 20 == 0:
            print(f"  {pi+1}/{len(picks)} plants", flush=True)

    X = torch.cat(FEAT); Y_lat = torch.cat(LAT); Y_pos = torch.cat(POS)
    V = torch.cat(VIS); P = torch.cat(PID)
    visible = V >= a.min_visible_px
    print(f"\n{len(X)} organs from {int(P.max())+1} plants | visible {int(visible.sum())} "
          f"({100*float(visible.float().mean()):.1f}%), occluded {int((~visible).sum())}")

    plants = P.unique()
    n_te = max(1, int(len(plants) * a.test_frac))
    te_plants = set(plants[-n_te:].tolist())
    te = torch.tensor([int(p) in te_plants for p in P])

    res = {}
    print(f"\n{'target':<10}{'subset':<12}{'n train':>9}{'n test':>8}{'held-out R^2':>15}{'retrieval top1':>16}{'chance':>9}")
    for tname, Y in (("latent", Y_lat), ("position", Y_pos)):
        for sname, sel in (("all", torch.ones_like(visible)), ("visible", visible), ("occluded", ~visible)):
            tr = (~te) & sel
            ts = te & sel
            if int(tr.sum()) < 50 or int(ts.sum()) < 20:
                continue
            r2, _ = ridge_r2(X[tr], Y[tr], X[ts], Y[ts])
            acc, ch, ntot = retrieval_top1(X[tr], Y[tr], X[ts], Y[ts], P[ts])
            res[f"{tname}_{sname}"] = {"r2": round(r2, 4), "retrieval_top1": round(acc, 4),
                                       "chance": round(ch, 4), "n_train": int(tr.sum()), "n_test": int(ts.sum())}
            print(f"{tname:<10}{sname:<12}{int(tr.sum()):>9}{int(ts.sum()):>8}{r2:>+15.4f}{acc:>16.3f}{ch:>9.3f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
