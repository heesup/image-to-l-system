"""Diffusion-Posterior-Sampling-style render guidance DURING the Stage 3 ODE (2026-09-16), as opposed to
eval_test_time_refinement.py's post-hoc Adam refinement of the FINAL sampled output. At every one of the 20 Heun
steps, this extrapolates the current x1 estimate (x + v*(1-t), the standard rectified-flow endpoint estimate),
decodes+renders it, and subtracts the gradient of a render-vs-input loss from the network's own velocity before
the ODE step is taken -- so the trajectory itself is steered toward the input image as it is generated, using
sample_ode's `guidance_fn` hook. The trained weights are never touched (see that hook's docstring for how this
differs from an unrolled, MAML-style training extension that WOULD touch them).

Reads the same checkpoint / eval set as eval_test_time_refinement.py; prints per-plant IoU with guidance on vs off
and the mean. Nothing is trained or saved.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from plant_recon.dataset.part_array_dataset import PartArrayDataset, FM_OT_END, FM_BASE_START
from plant_recon.models.hierarchical_part_flow_matching import (
    HierarchicalPartFlowMatchingModel, split_flow_state, geometry_from_flow, reconstruct_phytomer_rot,
)
from plant_recon.models.phytomer_vae import PhytomerVAE
from plant_recon.models.organ_latent_vae import OrganLatentVAE
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from plant_recon.models.plant_organ_array import NUM_ORGAN_TYPES
from plant_recon.dataset.phytomer_roll import roll_to_matrix
from plant_recon.dataset.phytomer_packets import matrix_to_rot6d
from plant_recon.dataset.phytomer_topology import chain_phytomers
from plant_recon.eval.eval_hierarchical_self_consistency import decode_predictions_to_part_tensor
from plant_recon.eval.eval_gt_substitution_ablation import plant_from_nodes, render_depth, score
from plant_recon.eval.eval_test_time_refinement import refine_plant
from plant_recon.eval.ckpt_compat import fix_ckpt_args

_ZI = {1.0: 3, 2.0: 7, 4.0: 11, 8.0: 15}


def _rot6d_from_pos_roll_parent(pos, roll, parent_pos):
    """Node rotation the same way reconstruct_phytomer_rot does, but from a (possibly still-noisy, mid-ODE)
    position/parent-position pair directly instead of re-running chain_phytomers: forward axis = unit
    (pos - parent), falling back to world-+Z where parent is NaN (the root, or an inactive slot) or the two
    coincide. Good enough for guidance -- topology itself is not being re-solved mid-ODE, only used to orient
    the mesh for the render loss -- and exact for the final materialisation once positions have converged."""
    d = pos - torch.nan_to_num(parent_pos, nan=0.0)
    n = d.norm(dim=-1, keepdim=True)
    up = torch.zeros_like(d); up[..., 2] = 1.0
    fwd = torch.where(n > 1e-6, d / n.clamp(min=1e-6), up)
    R = roll_to_matrix(fwd, roll)
    return matrix_to_rot6d(R)


def make_render_guidance(renderer, pvae, M, images, gt_center, target_zooms, guide_scale, D):
    """Returns a sample_ode(guidance_fn=...) callback closed over one plant's input/target/camera."""
    tgt = {float(z): images[0, _ZI[float(z)]] for z in target_zooms if images.shape[1] > _ZI[float(z)]}

    def guidance_fn(model, x, v1, t, dt, step, forward_kwargs, chain_parent_pos):
        with torch.enable_grad():
            x_g = x.detach().requires_grad_(True)
            t_tensor = torch.full((x_g.shape[0],), t, device=x_g.device)
            out = model.fine_stage(noisy_flow=x_g, timesteps=t_tensor, **forward_kwargs)
            v1_g = out["pred_velocity"]
            x1_hat = x_g + v1_g * (1.0 - t)                      # rectified-flow endpoint estimate at this step
            geom_block, lat_n = split_flow_state(x1_hat, D)
            lat = model.denormalize_latent(lat_n) if model.latent_norm_active() else lat_n
            if geom_block is not None:
                pos, roll, scale = geometry_from_flow(geom_block, chain_parent_pos)
            else:
                pos, roll, scale = forward_kwargs["phytomer_pos"], forward_kwargs["phytomer_roll"], forward_kwargs.get("phytomer_scale")
            exist = torch.ones(pos.shape[1], device=x.device)     # existence is Stage 2's alone here, not re-derived mid-ODE
            rot = _rot6d_from_pos_roll_parent(pos[0], roll[0], chain_parent_pos[0])
            parts = plant_from_nodes(pvae, pos[0], rot, scale[0], lat[0], exist, torch.nan_to_num(chain_parent_pos[0]), M)
            if parts.shape[0] == 0:
                return v1
            mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=x.device)
            loss = torch.zeros((), device=x.device)
            for z, t_img in tgt.items():
                pred = renderer.render_batched([mesh], azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, differentiable=True,
                                               image_size=128, zoom_factor=z, reference_window_size=1.2,
                                               centers=([gt_center] if gt_center is not None else None))[0, 3]
                canopy = (t_img > 0.005) | (pred > 0.005)
                l1 = F.smooth_l1_loss(pred, t_img, beta=0.02, reduction="none")
                loss_d = (l1 * canopy).sum() / canopy.sum().clamp(min=1)
                pm = torch.sigmoid((pred - 0.005) * 100.0); gm = (t_img > 0.005).float()
                dice = 1.0 - (2.0 * (pm * gm).sum() + 1e-4) / (pm.sum() + gm.sum() + 1e-4)
                loss = loss + 0.5 * loss_d + 1.0 * dice
            grad = torch.autograd.grad(loss, x_g, retain_graph=False)[0]
        return v1 - guide_scale * torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    return guidance_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_set", default="outputs/checkpoints/hierarchical_fm_v9/eval_set.json")
    ap.add_argument("--only", default="", help="comma-separated plant indices (default: the whole eval set)")
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--guide_scale", type=float, default=2.0)
    ap.add_argument("--guide_start_frac", type=float, default=0.0, help="skip guidance before this fraction of the ODE (t < guide_start_frac)")
    ap.add_argument("--target_zooms", default="1,2,4,8")
    ap.add_argument("--then_refine", action="store_true",
                    help="also run the standard post-hoc refinement (eval_test_time_refinement.refine_plant, "
                         "default settings: pos/scale/latent, input camera, reg 5/0.5, 40 steps) on top of the "
                         "GUIDED sample, to test whether a better-guided starting point beats plain-sample-then-"
                         "refine (2026-09-16 best: strict P 35.8 -> 67.6 on 10% v10)")
    ap.add_argument("--refine_steps", type=int, default=40)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    model = HierarchicalPartFlowMatchingModel(
        max_phytomers=args["max_phytomers"], slots_per_phytomer=args["slots_per_phytomer"], node_dim=args["node_dim"],
        num_classes=NUM_ORGAN_TYPES, image_size=128, patch_size=8, embed_dim=args["embed_dim"], vit_layers=args["vit_layers"],
        vit_heads=args["vit_heads"], coarse_layers=args["coarse_layers"], fine_layers=args["fine_layers"],
        flow_granularity=args["flow_granularity"], phytomer_latent_dim=args["phytomer_latent_dim"], backbone=args["backbone"],
        freeze_backbone=True, init_phytomer_count=args.get("init_phytomer_count", 50.0),
        stage3_geometry=bool(args.get("stage3_geometry", False)), multizoom=bool(args.get("multizoom", False)),
        node_token_window=int(args.get("node_token_window", 1)), stage3_regression=bool(args.get("stage3_regression", False)),
        use_depth=bool(args.get("use_depth", False))).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=False); model.eval()
    D = args["phytomer_latent_dim"]; M = args["slots_per_phytomer"]
    pvae = PhytomerVAE(latent_dim=D, residual_dim=args.get("phytomer_residual_dim", 8), hidden_dim=256).to(dev).eval()
    pvae.load_state_dict(torch.load(args["phytomer_vae_checkpoint"], map_location=dev, weights_only=True))
    for p_ in pvae.parameters():
        p_.requires_grad_(False)
    ovae = OrganLatentVAE(latent_dim=args["node_dim"], hidden_dim=256).to(dev).eval()
    sd = torch.load(args["organ_vae_checkpoint"], map_location=dev, weights_only=False); ovae.load_state_dict(sd.get("model_state_dict", sd))
    renderer = HeliosPyTorchRenderer(image_size=256).to(dev)
    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * M, cache_dir=args["cache_dir"],
                          pkt_cache_dir=args.get("pkt_cache_dir") or None, species="cowpea", image_size=128)
    idxs = json.load(open(a.eval_set))["indices"]
    if a.only:
        idxs = [int(x) for x in a.only.split(",")]
    zooms = [float(z) for z in a.target_zooms.split(",")]
    rows = []
    for i in idxs:
        it = ds[i]; images = it["image"].unsqueeze(0).to(dev); dap = int(it["dap"].item()); zoom = 8.0 if dap <= 15 else 1.0
        nodes = it["nodes"].to(dev); exist_gt = it["existence_mask"].to(dev)
        gt_parts = decode_predictions_to_part_tensor(nodes[:, FM_BASE_START:], nodes[:, :FM_OT_END].argmax(-1), exist_gt, device=dev)
        gt_depth = render_depth(renderer, gt_parts, zoom, dev)
        gt_center = None
        if gt_parts.shape[0] > 0:
            gv = renderer.geo_builder.build_mesh_from_part_tensor(gt_parts, device=dev)["vertices"]
            gt_center = 0.5 * (gv.min(0).values + gv.max(0).values) if gv.shape[0] > 0 else None
        with torch.no_grad():
            tok = model.image_encoder(images); clue = model.probe_pred_dap(tok)
            co0 = model.coarse_stage(tok, capacity_mode="pred_phyto", pred_dap=clue); K = int(co0["active_k"])
            co = model.coarse_stage(tok, active_k=K, pred_dap=clue)
            ordn = co["phytomer_ordinal"][0].float(); base = co["phytomer_base_logits"][0].float()

            so_plain = model.sample_ode(images=images, daps=None, num_steps=a.num_steps, vae=ovae, phytomer_vae=pvae)
            pos0 = so_plain["phytomer_pos"][0].float(); exist0 = so_plain["phytomer_existence"][0].float()
            roll0 = so_plain["phytomer_roll"][0].float(); scale0 = so_plain.get("phytomer_scale")
            scale0 = scale0[0].float() if scale0 is not None else torch.ones_like(pos0)
            lat0 = so_plain["pred_latent"][0].float()
            rot0, par0 = reconstruct_phytomer_rot(pos0.unsqueeze(0), roll0.unsqueeze(0), ordn.unsqueeze(0), base.unsqueeze(0),
                                                  exist=(exist0 > 0.5).float().unsqueeze(0))
            rot0, par0 = rot0[0].float(), par0[0].float()
            parts0 = plant_from_nodes(pvae, pos0, rot0, scale0, lat0, exist0, par0, M)
            iou0, _ = score(render_depth(renderer, parts0, zoom, dev), gt_depth)

        guidance = make_render_guidance(renderer, pvae, M, images, gt_center, zooms, a.guide_scale, D)

        def gated(model, x, v1, t, dt, step, forward_kwargs, chain_parent_pos):
            if t < a.guide_start_frac:
                return v1
            return guidance(model, x, v1, t, dt, step, forward_kwargs, chain_parent_pos)

        so_g = model.sample_ode(images=images, daps=None, num_steps=a.num_steps, vae=ovae, phytomer_vae=pvae, guidance_fn=gated)
        with torch.no_grad():
            pos1 = so_g["phytomer_pos"][0].float(); exist1 = so_g["phytomer_existence"][0].float()
            roll1 = so_g["phytomer_roll"][0].float()
            scale1 = so_g.get("phytomer_scale"); scale1 = scale1[0].float() if scale1 is not None else torch.ones_like(pos1)
            lat1 = so_g["pred_latent"][0].float()
            rot1, par1 = reconstruct_phytomer_rot(pos1.unsqueeze(0), roll1.unsqueeze(0), ordn.unsqueeze(0)[:, :pos1.shape[0]],
                                                  base.unsqueeze(0)[:, :pos1.shape[0]], exist=(exist1 > 0.5).float().unsqueeze(0))
            rot1, par1 = rot1[0].float(), par1[0].float()
            parts1 = plant_from_nodes(pvae, pos1, rot1, scale1, lat1, exist1, par1, M)
            iou1, _ = score(render_depth(renderer, parts1, zoom, dev), gt_depth)
        moved = float((pos1 - pos0[: pos1.shape[0]]).norm(dim=-1).mean() * 100) if pos1.shape[0] == pos0.shape[0] else float("nan")
        row = {"index": i, "dap": dap, "iou_plain": iou0, "iou_guided": iou1}
        extra = ""
        if a.then_refine:
            from types import SimpleNamespace
            parent_idx, _, _ = chain_phytomers(pos1, ordinal=ordn[: pos1.shape[0]], is_base=(base[: pos1.shape[0]] > 0).float(),
                                               exist=(exist1 > 0.5).float())
            live_m = exist1 > 0.5; has_par_m = parent_idx >= 0
            refine_ns = SimpleNamespace(steps=a.refine_steps, lr_pos=3e-3, lr_scale=2e-2, lr_latent=2e-2, lr_roll=2e-2, lr_exist=3e-2,
                                        reg_scale=5.0, reg_latent=0.5, reg_exist=0.3, keep_best=True, target_zooms=",".join(str(z) for z in zooms),
                                        plant_centered=False, recompute_rot=False)
            pos2, scale2, lat2, _, rot2, par2, _, exist2 = refine_plant(renderer, pvae, M, images, gt_center, pos1, rot1, par1, roll1, scale1, lat1,
                                                                 exist1, parent_idx, live_m, has_par_m, {"pos", "scale", "latent"}, refine_ns)
            with torch.no_grad():
                parts2 = plant_from_nodes(pvae, pos2, rot2, scale2, lat2, exist2, par2, M)
                iou2, _ = score(render_depth(renderer, parts2, zoom, dev), gt_depth)
            row["iou_guided_then_refined"] = iou2
            extra = f" -> refined {iou2*100:5.1f}"
        rows.append(row)
        print(f"idx {i:6d} DAP {dap:3d} IoU {iou0*100:5.1f} -> {iou1*100:5.1f}{extra}  (nodes moved {moved:.1f} cm)", flush=True)
    b = np.array([r["iou_plain"] for r in rows]); c = np.array([r["iou_guided"] for r in rows]); d = np.array([r["dap"] for r in rows])
    print(f"MEAN over {len(rows)} plants: {b.mean()*100:.1f} -> {c.mean()*100:.1f}  (DAP>15: {b[d>15].mean()*100:.1f} -> {c[d>15].mean()*100:.1f}) | "
          f"guide_scale={a.guide_scale} steps={a.num_steps}")
    if a.then_refine:
        e = np.array([r["iou_guided_then_refined"] for r in rows])
        print(f"MEAN guided->refined: {e.mean()*100:.1f}  (DAP>15: {e[d>15].mean()*100:.1f}) | refine_steps={a.refine_steps}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"checkpoint": a.checkpoint, "guide_scale": a.guide_scale, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
