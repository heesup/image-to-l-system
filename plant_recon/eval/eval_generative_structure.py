"""Does the relative or the absolute layout produce more plausible plant STRUCTURE?

Every comparison in this project so far scores reconstruction -- "did it match this plant?" -- via
silhouette IoU. That says nothing about whether a model is a good generative prior over plant
structure, which is the question the two Stage 3 layouts actually differ on.

The relative layout carries `[dpos(3) | roll(2) | scale(3) | latent]` and derives each node's forward
axis from the resolved chain, so a sampled organ CANNOT point away from the branch it sits on -- the
constraint is in the coordinate system. The absolute layout carries
`[pos(3) | rot6d(6) | scale(3) | latent]`, consults no parent, and gets the same botany back only as
a soft loss (internode length + forward-axis agreement). This measures what that difference costs.

Three statistics per sampled plant, each compared against the ground-truth distribution:

  forward-axis deviation   angle between a node's own forward axis (rot6d's x column) and the
                           direction from its parent to it. Zero by construction under the relative
                           layout; free under the absolute one. This is the sharpest test -- it
                           quantifies how often an absolute sample puts an organ off its branch.
  internode length         distance from a node to its parent. The relative layout predicts this
                           quantity directly; the absolute layout has to arrive at it.
  live node count          how many phytomers the sample activates, against the GT plant's.

Sampling is seeded per (plant, draw) exactly as eval_test_time_refinement does, so draws are
reproducible and comparable across checkpoints.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plant_recon.dataset.part_array_dataset import PartArrayDataset, attach_parent_links
from plant_recon.eval.ckpt_compat import fix_ckpt_args
from plant_recon.models.hierarchical_part_flow_matching import (
    HierarchicalPartFlowMatchingModel, reconstruct_phytomer_rot)
from plant_recon.models.organ_latent_vae import OrganLatentVAE
from plant_recon.models.phytomer_vae import PhytomerVAE
from plant_recon.models.plant_organ_array import NUM_ORGAN_TYPES


def axis_deviation_deg(rot6d, pos, parent, has_par):
    """Angle between each node's own forward axis and the parent->node direction, in degrees."""
    if has_par.sum() == 0:
        return np.array([])
    fwd = F.normalize(rot6d[has_par][..., 0:3], dim=-1, eps=1e-6)
    d = F.normalize(pos[has_par] - parent[has_par], dim=-1, eps=1e-6)
    cos = (fwd * d).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).cpu().numpy()


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    m = HierarchicalPartFlowMatchingModel(
        max_phytomers=a["max_phytomers"], slots_per_phytomer=a["slots_per_phytomer"], node_dim=a["node_dim"],
        num_classes=NUM_ORGAN_TYPES, image_size=128, patch_size=8, embed_dim=a["embed_dim"],
        vit_layers=a["vit_layers"], vit_heads=a["vit_heads"], coarse_layers=a["coarse_layers"],
        fine_layers=a["fine_layers"], flow_granularity=a["flow_granularity"],
        phytomer_latent_dim=a["phytomer_latent_dim"], backbone=a["backbone"], freeze_backbone=True,
        init_phytomer_count=a.get("init_phytomer_count", 50.0),
        stage3_geometry=bool(a.get("stage3_geometry", False)),
        stage3_absolute=bool(a.get("stage3_absolute", False)), multizoom=bool(a.get("multizoom", False)),
        node_token_window=int(a.get("node_token_window", 1)),
        stage3_regression=bool(a.get("stage3_regression", False)),
        use_depth=bool(a.get("use_depth", False))).to(dev).eval()
    m.load_state_dict(ck["model_state_dict"], strict=False)
    return m, a, bool(a.get("stage3_absolute", False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True, help="comma list of name=path")
    ap.add_argument("--eval_set", default="outputs/checkpoints/hierarchical_fm_v9/eval_set.json")
    ap.add_argument("--draws", type=int, default=4, help="samples per plant per checkpoint")
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/logs/20260920/generative_structure.json")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    specs = [c.split("=", 1) for c in a.checkpoints.split(",")]
    _, a0, _ = load(specs[0][1], dev)
    ovae = OrganLatentVAE(latent_dim=a0["node_dim"], hidden_dim=256)
    ovae.load_state_dict(torch.load(a0["organ_vae_checkpoint"], map_location="cpu", weights_only=False).get(
        "model_state_dict", torch.load(a0["organ_vae_checkpoint"], map_location="cpu", weights_only=False)))
    ovae = ovae.to(dev).eval()
    pvae = PhytomerVAE(latent_dim=a0["phytomer_latent_dim"],
                       residual_dim=a0.get("phytomer_residual_dim", 8), hidden_dim=256)
    pvae.load_state_dict(torch.load(a0["phytomer_vae_checkpoint"], map_location="cpu", weights_only=False).get(
        "model_state_dict", torch.load(a0["phytomer_vae_checkpoint"], map_location="cpu", weights_only=False)))
    pvae = pvae.to(dev).eval()

    ds = PartArrayDataset(data_root=a0["data_dir"], max_nodes=a0["max_phytomers"] * a0["slots_per_phytomer"],
                          cache_dir=a0["cache_dir"], pkt_cache_dir=a0.get("pkt_cache_dir") or None,
                          species="cowpea", image_size=128)
    by_prefix = {s["prefix"]: i for i, s in enumerate(ds.samples)}
    idxs = [by_prefix[s["prefix"]] for s in json.load(open(a.eval_set))["samples"] if s["prefix"] in by_prefix]

    # ---- ground truth reference
    gt_len, gt_dev, gt_n = [], [], []
    for i in idxs:
        pkt = ds[i].get("pkt")
        if pkt is None:
            continue
        attach_parent_links(pkt)
        pos = pkt["centers"].to(dev).float(); par = pkt["parent_pos"].to(dev).float()
        refs = pkt["refs"].to(dev).float()
        has = torch.isfinite(par).all(-1) & (par.abs().sum(-1) > 0)
        gt_len += (pos[has] - par[has]).norm(dim=-1).cpu().numpy().tolist()
        gt_dev += axis_deviation_deg(refs, pos, par, has).tolist()
        gt_n.append(pos.shape[0])
    print(f"GT: {len(gt_len)} internodes over {len(gt_n)} plants", flush=True)

    res = {"gt": {"internode_cm": [float(np.mean(gt_len) * 100), float(np.std(gt_len) * 100)],
                  "axis_dev_deg": [float(np.mean(gt_dev)), float(np.median(gt_dev))],
                  "nodes": float(np.mean(gt_n))}}

    for name, path in specs:
        model, args, s3abs = load(path, dev)
        s_len, s_dev, s_n = [], [], []
        for i in idxs:
            it = ds[i]
            images = it["image"].unsqueeze(0).to(dev)
            for d in range(a.draws):
                torch.manual_seed(a.sample_seed * 100003 + i * 101 + d)
                with torch.no_grad():
                    tok = model.image_encoder(images)
                    clue = model.probe_pred_dap(tok)
                    co0 = model.coarse_stage(tok, capacity_mode="pred_phyto", pred_dap=clue)
                    co = model.coarse_stage(tok, active_k=int(co0["active_k"]), pred_dap=clue)
                    so = model.sample_ode(images=images, daps=None, num_steps=20, vae=ovae, phytomer_vae=pvae)
                pos = so["phytomer_pos"][0].float()
                ex = (so["phytomer_existence"][0].float() > 0.5)
                roll = so["phytomer_roll"][0].float()
                ordn = co["phytomer_ordinal"][0].float(); base = co["phytomer_base_logits"][0].float()
                # LAYOUT SPLIT, as in eval_test_time_refinement: under the absolute layout `phytomer_roll`
                # carries the node's own rot6d, and the chain is used only for parent positions.
                if s3abs:
                    dummy = torch.zeros(roll.shape[0], 2, device=dev); dummy[:, 0] = 1.0
                    _, par = reconstruct_phytomer_rot(pos.unsqueeze(0), dummy.unsqueeze(0),
                                                      ordn.unsqueeze(0), base.unsqueeze(0), exist=ex.float().unsqueeze(0))
                    rot = roll
                else:
                    rot, par = reconstruct_phytomer_rot(pos.unsqueeze(0), roll.unsqueeze(0),
                                                        ordn.unsqueeze(0), base.unsqueeze(0), exist=ex.float().unsqueeze(0))
                    rot = rot[0].float()
                par = par[0].float()
                has = ex & torch.isfinite(par).all(-1) & (par.abs().sum(-1) > 0)
                if has.sum() == 0:
                    continue
                s_len += (pos[has] - par[has]).norm(dim=-1).cpu().numpy().tolist()
                s_dev += axis_deviation_deg(rot, pos, par, has).tolist()
                s_n.append(int(ex.sum()))
        res[name] = {"internode_cm": [float(np.mean(s_len) * 100), float(np.std(s_len) * 100)],
                     "axis_dev_deg": [float(np.mean(s_dev)), float(np.median(s_dev))],
                     "nodes": float(np.mean(s_n)), "n_internodes": len(s_len)}
        print(f"  {name}: {len(s_len)} internodes sampled", flush=True)

    print(f"\n{'model':<22}{'internode cm':>18}{'axis dev deg':>20}{'nodes':>8}")
    print(f"{'':<22}{'mean (sd)':>18}{'mean / median':>20}")
    for k in ["gt"] + [n for n, _ in specs]:
        r = res[k]
        print(f"{k:<22}{r['internode_cm'][0]:>10.2f} ({r['internode_cm'][1]:.2f}){r['axis_dev_deg'][0]:>13.1f} /{r['axis_dev_deg'][1]:>6.1f}{r['nodes']:>8.1f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
