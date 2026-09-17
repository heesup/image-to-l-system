"""GT-substitution ablation: which predicted quantity costs the silhouette?

On the run's fixed eval set, every predicted node is matched to a ground-truth
phytomer (the training matcher), and the plant is re-rendered with ONE quantity
of the matched nodes replaced by its ground truth:

    P        everything predicted (the self-consistency panel's plant)
    pos      node position <- GT centre            (chain + rotation recomputed)
    topo     ordinal / base <- GT depth            (chain recomputed, positions predicted)
    rot      node rotation <- GT reference rotation
    scale    node scale <- GT petiole scale row
    latent   packet latent <- VAE encode of the GT packet
    prune    unmatched predicted nodes removed     (spurious nodes only)
    ALL      pos + topo + rot + scale + latent <- GT (existence predicted)

against the GT render (same renderer, same camera). The gap P -> ALL is what the
model's per-node predictions cost; ALL -> 100% is what the node SET costs
(missing / spurious nodes). Decides where modelling capacity should go
(design doc §2.1: geometry into Stage 3, or render gradients into the latent).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from plant_recon.dataset.part_array_dataset import (
    PartArrayDataset, attach_parent_links, decode_fm, BASE_SCALE, FM_OT_END, FM_BASE_START, FM_BASE_END)
from plant_recon.dataset.phytomer_packets import assemble_packets, denormalize_packet_scales, phytomer_scale
from plant_recon.models.hierarchical_part_flow_matching import (
    HierarchicalPartFlowMatchingModel, reconstruct_phytomer_rot, apply_ref_for_flow)
from plant_recon.models.phytomer_vae import PhytomerVAE
from plant_recon.models.organ_latent_vae import OrganLatentVAE
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from plant_recon.models.plant_organ_array import NUM_ORGAN_TYPES
from plant_recon.training.hierarchical_hungarian_matcher import HierarchicalBotanicalMatcher
from plant_recon.eval.eval_hierarchical_self_consistency import decode_predictions_to_part_tensor
from plant_recon.dataset.phytomer_roll import encode_roll
from plant_recon.dataset.phytomer_packets import rot6d_to_matrix as _rot6d_to_matrix

EMPTY_IDX = 0


def render_depth(renderer, parts, zoom, device, size=256):
    if parts.shape[0] == 0:
        return torch.zeros(size, size, device=device)
    mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=device)
    if mesh["vertices"].shape[0] == 0:
        return torch.zeros(size, size, device=device)
    rgbd = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="ground",
                            focus_plant=False, include_depth=True, image_size=size, zoom_factor=zoom,
                            reference_window_size=1.2)
    return rgbd[3].clamp(min=0.0)


def score(pred_depth, gt_depth):
    pm, gm = pred_depth > 0.005, gt_depth > 0.005
    inter, union = (pm & gm).sum().item(), (pm | gm).sum().item()
    canopy = pm | gm
    mae = float((pred_depth - gt_depth).abs()[canopy].mean()) if canopy.any() else 0.0
    return inter / max(union, 1), mae


def plant_from_nodes(phytomer_vae, pos, rot, scale, latent, exist, parent_pos, M, soft_exist=False):
    """The training render block's decode: latent -> packets -> assembled 14D parts.

    soft_exist=False (default): `exist` is a hard 0/1 (or thresholded) per-phytomer mask; organs of non-existent
    phytomers are dropped from the returned row set entirely (the K*M -> row-count reduction has no gradient
    through `exist`). Every other caller of this function uses this mode, unchanged.

    soft_exist=True: `exist` is a CONTINUOUS per-phytomer probability in [0, 1] (e.g. the model's pre-threshold
    sigmoid output, or a value being optimised). No row is dropped by existence (only cls>0 still filters); instead
    a matching per-row alpha tensor is returned for build_mesh_from_part_tensor(existence=alpha) -- the renderer's
    own soft-existence alpha channel (helios_pytorch_geometry.build_mesh_from_part_tensor's `existence` argument,
    already alpha-composited against the background for both rgb and depth in helios_pytorch_renderer, and already
    used this way -- normally with existence.detach() for training stability -- by the training render block).
    That is the differentiable, physically-meaningful way to fade an organ in or out: it blends toward the
    background depth/silhouette at that pixel, not a `scale *= exist_prob` shrink (tried first, 2026-09-16,
    then dropped: shrinking is a real geometric change with its own depth footprint, not a fade, and does not
    correctly handle occlusion between overlapping organs the way alpha compositing does).
    Returns (parts, alpha) instead of parts alone when soft_exist=True.
    """
    K = pos.shape[0]
    out = phytomer_vae.decode(latent)
    recon = denormalize_packet_scales(out["recon_packets"], scale)
    pk = assemble_packets(recon, rot, parent_pos=parent_pos, centers=pos)
    abs_pk = apply_ref_for_flow(pk.reshape(1, K, M, 26), pos.unsqueeze(0), rot.unsqueeze(0))[0]   # (K, M, 26)
    cls = out["cls_logits"].reshape(K, M, 13).argmax(-1)
    if soft_exist:
        keep = cls > 0
        parts = decode_fm(abs_pk.reshape(-1, 26)[keep.reshape(-1)])
        alpha = exist.unsqueeze(-1).expand(K, M).reshape(-1)[keep.reshape(-1)]
        return parts, alpha
    keep = (cls > 0) & (exist > 0.5).unsqueeze(-1)
    return decode_fm(abs_pk.reshape(-1, 26)[keep.reshape(-1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--eval_set", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--teacher_force", action="store_true",
                    help="Condition Stage 3 on the GT node geometry of matched nodes when sampling (the "
                         "--stage3_gt_nodes arm; defaults to the checkpoint's own setting).")
    ap.add_argument("--no_teacher_force", action="store_true",
                    help="Sample with Stage 2's own nodes even for a --stage3_gt_nodes checkpoint (the deployable protocol).")
    ap.add_argument("--tag", default="", help="Suffix for the output JSON name (e.g. 'tf' / 'notf').")
    ap.add_argument("--refine", default="", help="comma list of variants to ALSO run through test-time refinement (eval_test_time_refinement.refine_plant, "
                                                  "default settings: input camera, prior 5/0.5, four zoom targets, keep_best); reported as '<variant>+refine'")
    ap.add_argument("--refine_steps", type=int, default=40)
    ap.add_argument("--self_cond_passes", type=int, default=1,
                    help="Sample N times, feeding each pass's refined nodes (pos/roll/scale) back as Stage 3's conditioning "
                         "for the next (2-pass self-conditioning; needs a --stage3_geometry checkpoint to change anything).")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    ckpt_path = a.checkpoint or sorted(glob.glob("outputs/checkpoints/hierarchical_fm_v9/hierarchical_fm_epoch_*.pt"))[-1]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    print(f"checkpoint {ckpt_path} (epoch {ck.get('epoch')})")
    model = HierarchicalPartFlowMatchingModel(
        max_phytomers=args["max_phytomers"], slots_per_phytomer=args["slots_per_phytomer"], node_dim=args["node_dim"],
        num_classes=NUM_ORGAN_TYPES, image_size=128, patch_size=8, embed_dim=args["embed_dim"], vit_layers=args["vit_layers"],
        vit_heads=args["vit_heads"], coarse_layers=args["coarse_layers"], fine_layers=args["fine_layers"],
        flow_granularity=args["flow_granularity"], phytomer_latent_dim=args["phytomer_latent_dim"], backbone=args["backbone"],
        freeze_backbone=True, init_phytomer_count=args.get("init_phytomer_count", 50.0),
        stage3_geometry=bool(args.get("stage3_geometry", False)), multizoom=bool(args.get("multizoom", False)), node_token_window=int(args.get("node_token_window", 1))).to(dev)
    print(f"  stage3_geometry: {bool(args.get('stage3_geometry', False))} (flow width {model.flow_dim})")
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    if missing or unexpected:
        print("  state dict: missing", missing[:5], "unexpected", unexpected[:5])
    model.eval()
    M = args["slots_per_phytomer"]
    pvae = PhytomerVAE(latent_dim=args["phytomer_latent_dim"], residual_dim=args.get("phytomer_residual_dim", 8), hidden_dim=256).to(dev).eval()
    pvae.load_state_dict(torch.load(args["phytomer_vae_checkpoint"], map_location=dev, weights_only=True))
    ovae = None
    ov_path = args.get("organ_vae_checkpoint", "outputs/checkpoints/organ_vae/organ_latent_vae_best.pt")
    if ov_path and os.path.exists(ov_path):
        ovae = OrganLatentVAE(latent_dim=args["node_dim"], hidden_dim=256).to(dev).eval()
        ovae.load_state_dict(torch.load(ov_path, map_location=dev, weights_only=False).get("model_state_dict", torch.load(ov_path, map_location=dev, weights_only=False)))
    renderer = HeliosPyTorchRenderer(image_size=256).to(dev)
    matcher = HierarchicalBotanicalMatcher(matcher_type="greedy", slots_per_phytomer=M)
    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * M, cache_dir=args["cache_dir"],
                          pkt_cache_dir=args.get("pkt_cache_dir") or None, species="cowpea", image_size=128)
    es_path = a.eval_set or os.path.join(os.path.dirname(ckpt_path), "eval_set.json")
    idxs = json.load(open(es_path))["indices"]
    out_dir = a.out_dir or os.path.dirname(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)

    refine_list = [x for x in a.refine.split(",") if x]

    refine_ns = None

    if refine_list:

        from types import SimpleNamespace

        from plant_recon.eval.eval_test_time_refinement import refine_plant

        refine_ns = SimpleNamespace(steps=a.refine_steps, lr_pos=3e-3, lr_scale=2e-2, lr_latent=2e-2, lr_roll=2e-2, lr_exist=3e-2, reg_scale=5.0, reg_latent=0.5,

                                    reg_exist=1.0, keep_best=True, target_zooms="1,2,4,8", plant_centered=False, recompute_rot=False)

    variants = ["P", "pos", "topo", "rot", "scale", "latent", "prune",
                "pos+topo", "pos+rot", "pos+scale", "pos+latent",
                "ALL-pos", "ALL-topo", "ALL-rot", "ALL-scale", "ALL-latent", "ALL",
                "meanlat", "ALL-latent+meanlat"]
    FULL = {"pos", "topo", "rot", "scale", "latent"}
    teacher_force = (a.teacher_force or bool(args.get("stage3_gt_nodes", False))) and not a.no_teacher_force
    print(f"  teacher forcing (GT nodes as Stage 3 conditioning): {teacher_force}")
    # Mean GT latent over the eval set: a latent that carries no image information at all.
    # 'meanlat' = predicted geometry + mean latent; 'ALL-latent+meanlat' = GT geometry + mean latent,
    # to be read against 'ALL-latent' (GT geometry + predicted latent).
    lat_sum, lat_n = None, 0
    with torch.no_grad():
        for i in idxs:
            pk = ds[i].get("pkt")
            if pk is None: continue
            l_ = pvae.encode(pvae.pack_input(pk["packets"].to(dev).float(), pk["presence"].to(dev)))[0].float()
            lat_sum = l_.sum(0) if lat_sum is None else lat_sum + l_.sum(0); lat_n += l_.shape[0]
    mean_lat = lat_sum / max(lat_n, 1)
    lat_err = {"pred": [], "mean": [], "pred_std": [], "gt_std": []}

    def subset(v):
        if v == "P": return set()
        if v == "ALL": return set(FULL)
        if v == "ALL-latent+meanlat": return (FULL - {"latent"}) | {"meanlat"}
        if v.startswith("ALL-"): return FULL - {v[4:]}
        return set(v.split("+"))
    rows = []
    for i in idxs:
        it = ds[i]
        images = it["image"].unsqueeze(0).to(dev)
        dap = int(it["dap"].item()) if "dap" in it else 0
        zoom = 8.0 if dap <= 15 else 1.0
        nodes = it["nodes"].to(dev); exist_gt = it["existence_mask"].to(dev)
        pkt = it.get("pkt")
        if pkt is None:
            print(f"  sample {i}: no packet cache, skipped"); continue
        attach_parent_links(pkt)
        g_pos = pkt["centers"].to(dev).float(); g_refs = pkt["refs"].to(dev).float()
        g_packets = pkt["packets"].to(dev).float(); g_pres = pkt["presence"].to(dev)
        g_scale = phytomer_scale(g_packets); g_depth = pkt["depth"].to(dev).float()
        with torch.no_grad():
            g_lat = pvae.encode(pvae.pack_input(g_packets, g_pres))[0].float()
            # GT reference render
            gt_types = nodes[:, :FM_OT_END].argmax(-1)
            gt_parts = decode_predictions_to_part_tensor(nodes[:, FM_BASE_START:], gt_types, exist_gt, device=dev)
            gt_depth_img = render_depth(renderer, gt_parts, zoom, dev)
            # prediction (Stage 2 first: the match is made on its positions; with teacher forcing the
            # sampler is then conditioned on the GT geometry of the matched nodes)
            image_tokens = model.image_encoder(images)
            clue = model.probe_pred_dap(image_tokens)
            co0 = model.coarse_stage(image_tokens, capacity_mode="pred_phyto", pred_dap=clue)
            K = int(co0["active_k"])
            co = model.coarse_stage(image_tokens, active_k=K, pred_dap=clue)
            override = None
            if teacher_force:
                p_pos0 = co["phytomer_pos"][0].float(); p_ex0 = (torch.sigmoid(co["phytomer_logits"]).squeeze(-1) * co["soft_margin_weights"])[0].float() if "soft_margin_weights" in co else torch.sigmoid(co["phytomer_logits"]).squeeze(-1)[0].float()
                labels0 = nodes[:, :FM_OT_END].argmax(-1); positions0 = nodes[:, FM_BASE_START:FM_BASE_END] / BASE_SCALE
                valid0 = (exist_gt > 0.5) & (labels0 > 0)
                m0 = matcher(pred_phytomer_pos=p_pos0.unsqueeze(0), pred_phytomer_logits=torch.logit(p_ex0.clamp(1e-4, 1 - 1e-4)).view(1, K, 1),
                             pred_fine_geom=torch.zeros(1, K * M, 16, device=dev), pred_fine_logits=torch.zeros(1, K * M, 13, device=dev),
                             tgt_geoms=[torch.zeros(0, 16, device=dev)], tgt_labels=[labels0], tgt_positions=None,
                             per_sample_max_phytomers=torch.tensor([K], device=dev), skip_fine=True,
                             tgt_labels_padded=labels0.unsqueeze(0), tgt_positions_padded=positions0.unsqueeze(0), tgt_valid=valid0.unsqueeze(0))[0]
                src0 = m0["phytomer_src_idx"]; j0 = torch.cdist(m0["phytomer_tgt_pos"], g_pos).argmin(dim=1) if src0.numel() > 0 else torch.zeros(0, dtype=torch.long, device=dev)
                # same forward axis as the training roll target: parent -> child, world up for the root
                g_parent = torch.nan_to_num(pkt["parent_pos"].to(dev).float())
                g_d = g_pos - g_parent
                g_fwd = torch.where((g_d.norm(dim=-1, keepdim=True) > 1e-6), torch.nn.functional.normalize(g_d, dim=-1),
                                    torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(g_d))
                g_roll = encode_roll(_rot6d_to_matrix(g_refs), g_fwd)
                o_pos, o_roll, o_scale = p_pos0.clone(), co["phytomer_roll"][0].float().clone(), co["phytomer_scale"][0].float().clone()
                o_pos[src0] = g_pos[j0]; o_roll[src0] = g_roll[j0]; o_scale[src0] = g_scale[j0]
                override = {"pos": o_pos.unsqueeze(0), "roll": o_roll.unsqueeze(0), "scale": o_scale.unsqueeze(0)}
            for _pass in range(max(a.self_cond_passes, 1) - 1):
                so_prev = model.sample_ode(images=images, daps=None, num_steps=a.num_steps, vae=ovae, phytomer_vae=pvae, cond_override=override)
                override = {"pos": so_prev["phytomer_pos"].float(), "roll": so_prev["phytomer_roll"].float(),
                            "scale": (so_prev.get("phytomer_scale") if so_prev.get("phytomer_scale") is not None else co.get("phytomer_scale")).float()}
            so = model.sample_ode(images=images, daps=None, num_steps=a.num_steps, vae=ovae, phytomer_vae=pvae, cond_override=override)
            # with --stage3_geometry these are Stage 3's refined values (sample_ode returns them under the usual keys)
            p_pos = so["phytomer_pos"][0].float(); p_exist = so["phytomer_existence"][0].float()
            p_roll = so["phytomer_roll"][0].float(); p_ord = co["phytomer_ordinal"][0].float(); p_base = co["phytomer_base_logits"][0].float()
            p_scale = (so.get("phytomer_scale") if so.get("phytomer_scale") is not None else co.get("phytomer_scale"))[0].float()
            p_lat = so["pred_latent"][0].float()
            # match predicted nodes to GT phytomers (as training does), then to GT packets by centre
            labels = nodes[:, :FM_OT_END].argmax(-1); positions = nodes[:, FM_BASE_START:FM_BASE_END] / BASE_SCALE
            valid = (exist_gt > 0.5) & (labels > 0)
            m = matcher(pred_phytomer_pos=p_pos.unsqueeze(0), pred_phytomer_logits=torch.logit(p_exist.clamp(1e-4, 1 - 1e-4)).view(1, K, 1),
                        pred_fine_geom=torch.zeros(1, K * M, 16, device=dev), pred_fine_logits=torch.zeros(1, K * M, 13, device=dev),
                        tgt_geoms=[torch.zeros(0, 16, device=dev)], tgt_labels=[labels], tgt_positions=None,
                        per_sample_max_phytomers=torch.tensor([K], device=dev), skip_fine=True,
                        tgt_labels_padded=labels.unsqueeze(0), tgt_positions_padded=positions.unsqueeze(0), tgt_valid=valid.unsqueeze(0))[0]
            src = m["phytomer_src_idx"]; tgt_pos = m["phytomer_tgt_pos"]
            j = torch.cdist(tgt_pos, g_pos).argmin(dim=1) if src.numel() > 0 else torch.zeros(0, dtype=torch.long, device=dev)
            if src.numel() > 0:
                # how much image information the sampled latent carries: its error against the GT latent of the
                # matched phytomer, next to the error of the dataset-mean latent (a latent with no information)
                pl = so["phytomer_latent"][0].float()[src]; gl = g_lat[j]
                lat_err["pred"].append(((pl - gl) ** 2).sum(-1)); lat_err["mean"].append(((mean_lat - gl) ** 2).sum(-1))
                lat_err["pred_std"].append(pl.std(0)); lat_err["gt_std"].append(gl.std(0))
            matched = torch.zeros(K, dtype=torch.bool, device=dev); matched[src] = True

            def build_vars(sub):
                pos, roll, ordn, base, scale, lat, ex = p_pos.clone(), p_roll.clone(), p_ord.clone(), p_base.clone(), p_scale.clone(), p_lat.clone(), p_exist.clone()
                if "pos" in sub: pos[src] = g_pos[j]
                if "topo" in sub: ordn[src] = g_depth[j]; base[src] = torch.where(g_depth[j] == 0, 8.0, -8.0)
                if "scale" in sub: scale[src] = g_scale[j]
                if "latent" in sub: lat[src] = g_lat[j]
                if "meanlat" in sub: lat[:] = mean_lat
                if "prune" in sub: ex = torch.where(matched, ex, torch.zeros_like(ex))
                rot, par = reconstruct_phytomer_rot(pos.unsqueeze(0), roll.unsqueeze(0), ordn.unsqueeze(0), base.unsqueeze(0), exist=(ex > 0.5).float().unsqueeze(0))
                rot, par = rot[0].float(), par[0].float()
                if "rot" in sub: rot[src] = g_refs[j]
                return pos, rot, scale, lat, ex, par, roll

            def build(sub):
                pos, rot, scale, lat, ex, par, _ = build_vars(sub)
                return plant_from_nodes(pvae, pos, rot, scale, lat, ex, par, M)

            row = {"index": i, "dap": dap, "K": K, "n_active": int((p_exist > 0.5).sum()), "n_gt": int(g_pos.shape[0]), "n_matched": int(src.numel())}
            for v in variants:
                sub = subset(v)
                parts = build(sub)
                iou, mae = score(render_depth(renderer, parts, zoom, dev), gt_depth_img)
                row[v] = iou; row[v + "_mae"] = mae
            if refine_list:
                gt_center = None
                if gt_parts.shape[0] > 0:
                    gv = renderer.geo_builder.build_mesh_from_part_tensor(gt_parts, device=dev)["vertices"]
                    gt_center = 0.5 * (gv.min(0).values + gv.max(0).values) if gv.shape[0] > 0 else None
                for v in refine_list:
                    pos_, rot_, scale_, lat_, ex_, par_, roll_ = build_vars(subset(v))
                    with torch.enable_grad():
                        pos_, scale_, lat_, _, rot_, par_, _ = refine_plant(renderer, pvae, M, images, gt_center, pos_, rot_, par_, roll_, scale_, lat_, ex_,
                                                                          None, None, None, {"pos", "scale", "latent"}, refine_ns)
                    parts = plant_from_nodes(pvae, pos_, rot_, scale_, lat_, ex_, par_, M)
                    iou, mae = score(render_depth(renderer, parts, zoom, dev), gt_depth_img)
                    row[v + "+refine"] = iou; row[v + "+refine_mae"] = mae
            rows.append(row)
            print(f"  DAP {dap:>3} K {K:>3} active {row['n_active']:>3} gt {row['n_gt']:>3} matched {row['n_matched']:>3} | " +
                  " ".join(f"{v} {row[v]*100:5.1f}" for v in ("P", "pos", "ALL")), flush=True)

    print("\n=== mean silhouette IoU (%) / depth MAE (cm) over", len(rows), "samples")
    print(f"{'variant':<11}{'IoU':>7}{'MAE':>7}   {'young<=15':>10}{'mid':>7}{'old>60':>8}")
    summary = {}
    for v in list(variants) + [v_ + "+refine" for v_ in refine_list]:
        ious = np.array([r[v] for r in rows]); maes = np.array([r[v + "_mae"] for r in rows]); daps = np.array([r["dap"] for r in rows])
        yb, mb, ob = daps <= 15, (daps > 15) & (daps <= 60), daps > 60
        summary[v] = {"iou": float(ious.mean()), "mae_cm": float(maes.mean() * 100),
                      "iou_young": float(ious[yb].mean()) if yb.any() else None, "iou_mid": float(ious[mb].mean()) if mb.any() else None,
                      "iou_old": float(ious[ob].mean()) if ob.any() else None}
        f = lambda x: f"{x*100:6.1f}" if x is not None else "   n/a"
        print(f"{v:<11}{ious.mean()*100:7.1f}{maes.mean()*100:7.2f}   {f(summary[v]['iou_young']):>10}{f(summary[v]['iou_mid']):>7}{f(summary[v]['iou_old']):>8}")
    if lat_err["pred"]:
        e_p = torch.cat(lat_err["pred"]).mean().sqrt().item(); e_m = torch.cat(lat_err["mean"]).mean().sqrt().item()
        s_p = torch.stack(lat_err["pred_std"]).mean().item(); s_g = torch.stack(lat_err["gt_std"]).mean().item()
        summary["_latent"] = {"rmse_pred": e_p, "rmse_mean": e_m, "r2_vs_mean": 1.0 - (e_p / max(e_m, 1e-9)) ** 2,
                              "pred_std": s_p, "gt_std": s_g}
        print(f"latent (matched nodes, 128D): RMSE pred vs GT {e_p:.3f} | mean-latent vs GT {e_m:.3f} | "
              f"R^2 over the mean {summary['_latent']['r2_vs_mean']:.3f} | per-dim std pred {s_p:.3f} / GT {s_g:.3f}")
    print(f"node set: predicted active / GT / matched = {np.mean([r['n_active'] for r in rows]):.1f} / {np.mean([r['n_gt'] for r in rows]):.1f} / {np.mean([r['n_matched'] for r in rows]):.1f}")
    tag = f"epoch{int(ck.get('epoch', 0)):03d}" + (f"_{a.tag}" if a.tag else "")
    with open(os.path.join(out_dir, f"gt_substitution_{tag}.json"), "w") as f:
        json.dump({"checkpoint": ckpt_path, "summary": summary, "rows": rows}, f, indent=2)
    print("saved", os.path.join(out_dir, f"gt_substitution_{tag}.json"))


if __name__ == "__main__":
    main()
