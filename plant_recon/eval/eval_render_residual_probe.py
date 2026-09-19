"""Does the render residual carry the correction, before any of it is trained?

The render-feedback corrector (RenderFeedbackCorrector) is built on a claim: the residual between the
model's own render and the input canopy height map is per-NODE information that the image tokens are not
(one 7.5 cm token holds 7-15 phytomers, so a patch cannot say which of them is wrong, while "the render
is too low HERE" can). Test-time refinement is worth +35 IoU by using exactly that residual.

This probes the claim without training anything. For each eval plant: sample cold, render, take the
residual; then run the refinement that is known to work and record the correction it applied to every
node. A ridge probe then asks whether that correction is predictable from the residual read at the node's
own projected position, scored on HELD-OUT plants. If it is not, a learned corrector cannot work either
and the idea should be dropped before it costs GPU hours.

Baselines in the same table:
  residual   the residual patch around the node  (the claim)
  state      the node's own position/scale, no image at all  (what is trivially predictable)
  both       the two together
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from plant_recon.dataset.part_array_dataset import PartArrayDataset, FM_BASE_START, FM_OT_END
from plant_recon.dataset.phytomer_topology import chain_phytomers
from plant_recon.eval.ckpt_compat import fix_ckpt_args
from plant_recon.eval.eval_gt_substitution_ablation import plant_from_nodes
from plant_recon.eval.eval_hierarchical_self_consistency import decode_predictions_to_part_tensor
from plant_recon.eval.eval_latent_conditioning_ceiling import ridge_r2
from plant_recon.eval.eval_test_time_refinement import refine_plant
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from plant_recon.models.hierarchical_part_flow_matching import (HierarchicalPartFlowMatchingModel,
                                                                 reconstruct_phytomer_rot)
from plant_recon.models.organ_latent_vae import OrganLatentVAE
from plant_recon.models.phytomer_vae import PhytomerVAE
from plant_recon.models.plant_organ_array import NUM_ORGAN_TYPES

WINDOW_M = 1.2


def _norm(x):
    return (x - x.mean()) / x.std().clamp(min=1e-4)


def patches(res_map, pos, half=4):
    """(K, (2*half+1)^2) residual window around each node's projected position."""
    S = res_map.shape[-1]
    u = (pos[:, 0] / (WINDOW_M * 0.5)).clamp(-1, 1)
    v = -(pos[:, 1] / (WINDOW_M * 0.5)).clamp(-1, 1)
    px = ((u * 0.5 + 0.5) * (S - 1)).round().long().clamp(half, S - 1 - half)
    py = ((v * 0.5 + 0.5) * (S - 1)).round().long().clamp(half, S - 1 - half)
    out = []
    for i in range(pos.shape[0]):
        out.append(res_map[py[i] - half:py[i] + half + 1, px[i] - half:px[i] + half + 1].reshape(-1))
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_set", default="outputs/checkpoints/hierarchical_fm_v9/eval_set.json")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--half", type=int, default=4, help="residual window half-width in pixels")
    ap.add_argument("--out", default="outputs/logs/20260917/render_residual_probe.json")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    M = args["slots_per_phytomer"]
    model = HierarchicalPartFlowMatchingModel(
        max_phytomers=args["max_phytomers"], slots_per_phytomer=M, node_dim=args["node_dim"],
        num_classes=NUM_ORGAN_TYPES, image_size=128, patch_size=8, embed_dim=args["embed_dim"],
        vit_layers=args["vit_layers"], vit_heads=args["vit_heads"], coarse_layers=args["coarse_layers"],
        fine_layers=args["fine_layers"], flow_granularity=args["flow_granularity"],
        phytomer_latent_dim=args["phytomer_latent_dim"], backbone=args["backbone"], freeze_backbone=True,
        init_phytomer_count=args.get("init_phytomer_count", 50.0),
        stage3_geometry=bool(args.get("stage3_geometry", False)),
        stage3_absolute=bool(args.get("stage3_absolute", False)), multizoom=bool(args.get("multizoom", False)),
        node_token_window=int(args.get("node_token_window", 1)), use_depth=bool(args.get("use_depth", False))).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=False); model.eval()
    pvae = PhytomerVAE(latent_dim=args["phytomer_latent_dim"], residual_dim=args.get("phytomer_residual_dim", 8),
                       hidden_dim=256).to(dev).eval()
    pvae.load_state_dict(torch.load(args["phytomer_vae_checkpoint"], map_location=dev, weights_only=True))
    for p_ in pvae.parameters():
        p_.requires_grad_(False)
    ovae = OrganLatentVAE(latent_dim=args["node_dim"], hidden_dim=256).to(dev).eval()
    sd = torch.load(args["organ_vae_checkpoint"], map_location=dev, weights_only=False)
    ovae.load_state_dict(sd.get("model_state_dict", sd))
    renderer = HeliosPyTorchRenderer(image_size=128).to(dev)
    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * M,
                          cache_dir=args["cache_dir"], pkt_cache_dir=args.get("pkt_cache_dir") or None,
                          species="cowpea", image_size=128)
    idxs = json.load(open(a.eval_set))["indices"]
    ns = SimpleNamespace(steps=a.steps, lr_pos=3e-3, lr_scale=2e-2, lr_latent=2e-2, lr_roll=2e-2, lr_exist=3e-2,
                         reg_scale=5.0, reg_latent=0.5, reg_exist=0.3, keep_best=True, target_zooms="1,2,4,8",
                         plant_centered=False, recompute_rot=False, input_camera=True, lr_schedule="const",
                         refine_px=128)
    rows = {"residual": [], "state": [], "dpos": [], "plant": []}
    for n, i in enumerate(idxs):
        it = ds[i]
        images = it["image"].unsqueeze(0).to(dev)
        nodes = it["nodes"].to(dev); exg = it["existence_mask"].to(dev)
        with torch.no_grad():
            gt_parts = decode_predictions_to_part_tensor(nodes[:, FM_BASE_START:], nodes[:, :FM_OT_END].argmax(-1), exg, device=dev)
            gv = renderer.geo_builder.build_mesh_from_part_tensor(gt_parts, device=dev)["vertices"]
            gt_center = 0.5 * (gv.min(0).values + gv.max(0).values) if gv.shape[0] else None
            tok = model.image_encoder(images); clue = model.probe_pred_dap(tok)
            co0 = model.coarse_stage(tok, capacity_mode="pred_phyto", pred_dap=clue)
            co = model.coarse_stage(tok, active_k=int(co0["active_k"]), pred_dap=clue)
            so = model.sample_ode(images=images, daps=None, num_steps=20, vae=ovae, phytomer_vae=pvae)
            pos0 = so["phytomer_pos"][0].float(); ex = (so["phytomer_existence"][0].float() > 0.5).float()
            roll = so["phytomer_roll"][0].float(); lat0 = so["pred_latent"][0].float()
            scale0 = (so.get("phytomer_scale") if so.get("phytomer_scale") is not None else co.get("phytomer_scale"))[0].float()
            ordn = co["phytomer_ordinal"][0].float(); base = co["phytomer_base_logits"][0].float()
            rot, par = reconstruct_phytomer_rot(pos0.unsqueeze(0), roll.unsqueeze(0), ordn.unsqueeze(0),
                                                 base.unsqueeze(0), exist=ex.unsqueeze(0))
            rot, par = rot[0].float(), par[0].float()
            parent_idx, _, _ = chain_phytomers(pos0, ordinal=ordn, is_base=(base > 0).float(), exist=ex)
            parts0 = plant_from_nodes(pvae, pos0, rot, scale0, lat0, ex, par, M)
            if parts0.shape[0] == 0:
                continue
            mesh0 = renderer.geo_builder.build_mesh_from_part_tensor(parts0, device=dev)
            pred0 = renderer.render_batched([mesh0], azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                                            differentiable=False, image_size=128, zoom_factor=1.0,
                                            reference_window_size=WINDOW_M,
                                            centers=([gt_center] if gt_center is not None else None))[0, 3]
            residual = (_norm(pred0) - _norm(images[0, 3])).float()
        live = ex > 0.5
        if int(live.sum()) < 3:
            continue
        pos_f, scale_f, lat_f, *_ = refine_plant(renderer, pvae, M, images, gt_center, pos0, rot, par, roll,
                                                 scale0, lat0, ex, parent_idx, live, parent_idx >= 0,
                                                 {"pos", "scale", "latent"}, ns,
                                                 exist_prob0=so["phytomer_existence"][0].float())
        d = (pos_f - pos0)[live].detach().cpu()
        rows["residual"].append(patches(residual, pos0[live], a.half).cpu())
        rows["state"].append(torch.cat([pos0[live], scale0[live]], -1).detach().cpu())
        rows["dpos"].append(d)
        rows["plant"].append(torch.full((int(live.sum()),), n))
        print(f"  plant {n + 1}/{len(idxs)}: {int(live.sum())} nodes, refinement moved {d.norm(dim=-1).mean() * 100:.1f} cm", flush=True)

    X = {k: torch.cat(v) for k, v in rows.items() if k in ("residual", "state")}
    X["both"] = torch.cat([X["residual"], X["state"]], -1)
    y = torch.cat(rows["dpos"]); plant = torch.cat(rows["plant"])
    plants = plant.unique(); test = set(plants[-max(1, len(plants) // 4):].tolist())
    te = torch.tensor([int(p) in test for p in plant.tolist()])
    print(f"\n{len(y)} nodes, {len(plants)} plants | train {int((~te).sum())} / held-out {int(te.sum())} nodes")
    print(f"refinement's own correction: mean {y.norm(dim=-1).mean() * 100:.2f} cm, std per axis {y.std(0).mul(100).tolist()}")
    res = {}
    for name in ("state", "residual", "both"):
        r2, alpha = ridge_r2(X[name][~te], y[~te], X[name][te], y[te])
        res[name] = {"r2": round(r2, 4), "dim": int(X[name].shape[1]), "alpha": alpha}
        print(f"  {name:<9} dim {X[name].shape[1]:>4}  held-out R^2 on refinement's correction: {r2:+.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"checkpoint": a.checkpoint, "n_nodes": int(len(y)), "results": res}, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
