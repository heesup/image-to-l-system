"""Test-time refinement (analysis-by-synthesis): after sampling a plant, optimise its node positions, scales and
phytomer latents for a few Adam steps against the INPUT canopy height map with the training render loss
(depth smooth-L1 + silhouette dice at zoom 1x/2x), then score under the strict 256 px protocol.

Reads the same checkpoint / eval set as eval_gt_substitution_ablation.py; prints per-plant IoU before -> after and
the mean, for the variable sets given by --opt (e.g. pos,scale,latent). Nothing is trained or saved."""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from diffusion_based.dataset.part_array_dataset import PartArrayDataset, attach_parent_links, FM_OT_END, FM_BASE_START
from diffusion_based.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel, reconstruct_phytomer_rot
from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.models.organ_latent_vae import OrganLatentVAE
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.plant_organ_array import NUM_ORGAN_TYPES
from diffusion_based.eval.eval_hierarchical_self_consistency import decode_predictions_to_part_tensor
from diffusion_based.eval.eval_gt_substitution_ablation import plant_from_nodes, render_depth, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_set", default="diffusion_based/checkpoints/hierarchical_fm_v9/eval_set.json")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--opt", default="pos,scale,latent", help="comma list from pos,scale,latent")
    ap.add_argument("--lr_pos", type=float, default=3e-3)
    ap.add_argument("--lr_scale", type=float, default=2e-2)
    ap.add_argument("--lr_latent", type=float, default=2e-2)
    ap.add_argument("--out", default="")
    ap.add_argument("--plant_centered", action="store_true",
                    help="Render the prediction with the camera centred on its own bounding box (focus_plant), the frame "
                         "the cached input CHM was rendered in (plant-bbox-centred), instead of the fixed origin window. "
                         "Measured 2026-09-15: the cached CHM matches a bbox-centred GT render at 73-89% IoU but the "
                         "origin-window one at only 41-76%.")
    ap.add_argument("--input_camera", action="store_true", default=True,
                    help="render the prediction in the camera frame the cached input was rendered in (GT plant bbox centre); default on")
    ap.add_argument("--origin_camera", action="store_true", help="use the origin-centred window instead of --input_camera (the pre-2026-09-15 behaviour)")
    ap.add_argument("--save_renders", default="", help="folder: save gt / before / after top-view renders (256 px PNG) per plant")
    ap.add_argument("--only", default="", help="comma-separated plant indices to process (default: the whole eval set)")
    ap.add_argument("--target_zooms", default="1,2,4,8", help="comma list of cache zoom levels (1,2,4,8) whose depth channels are the refinement targets; the 4x/8x levels add signal for plants that are only a few pixels wide at 1x/2x")
    ap.add_argument("--reg_scale", type=float, default=5.0, help="penalty weight on (scale - sampled scale)^2, keeps leaves from inflating to fill the silhouette")
    ap.add_argument("--reg_latent", type=float, default=0.5, help="penalty weight on mean (latent - sampled latent)^2")
    ap.add_argument("--keep_best", action="store_true",
                    help="Return the variables of the step with the lowest INPUT loss (model selection on the input only), "
                         "not the last step -- guards against the divergent plants.")
    a = ap.parse_args()
    if a.origin_camera:
        a.input_camera = False
    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False); args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    model = HierarchicalPartFlowMatchingModel(
        max_phytomers=args["max_phytomers"], slots_per_phytomer=args["slots_per_phytomer"], node_dim=args["node_dim"],
        num_classes=NUM_ORGAN_TYPES, image_size=128, patch_size=8, embed_dim=args["embed_dim"], vit_layers=args["vit_layers"],
        vit_heads=args["vit_heads"], coarse_layers=args["coarse_layers"], fine_layers=args["fine_layers"],
        flow_granularity=args["flow_granularity"], phytomer_latent_dim=args["phytomer_latent_dim"], backbone=args["backbone"],
        freeze_backbone=True, init_phytomer_count=args.get("init_phytomer_count", 50.0),
        stage3_geometry=bool(args.get("stage3_geometry", False)), multizoom=bool(args.get("multizoom", False)),
        node_token_window=int(args.get("node_token_window", 1))).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=False); model.eval()
    M = args["slots_per_phytomer"]
    pvae = PhytomerVAE(latent_dim=args["phytomer_latent_dim"], residual_dim=args.get("phytomer_residual_dim", 8), hidden_dim=256).to(dev).eval()
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
    opt_set = set(a.opt.split(","))
    rows = []
    for i in idxs:
        it = ds[i]; images = it["image"].unsqueeze(0).to(dev); dap = int(it["dap"].item()); zoom = 8.0 if dap <= 15 else 1.0
        nodes = it["nodes"].to(dev); exist_gt = it["existence_mask"].to(dev)
        with torch.no_grad():
            gt_parts = decode_predictions_to_part_tensor(nodes[:, FM_BASE_START:], nodes[:, :FM_OT_END].argmax(-1), exist_gt, device=dev)
            gt_depth = render_depth(renderer, gt_parts, zoom, dev)
            gt_center = None
            if a.input_camera and gt_parts.shape[0] > 0:
                gv = renderer.geo_builder.build_mesh_from_part_tensor(gt_parts, device=dev)["vertices"]
                gt_center = 0.5 * (gv.min(0).values + gv.max(0).values) if gv.shape[0] > 0 else None
            tok = model.image_encoder(images); clue = model.probe_pred_dap(tok)
            co0 = model.coarse_stage(tok, capacity_mode="pred_phyto", pred_dap=clue); K = int(co0["active_k"]); co = model.coarse_stage(tok, active_k=K, pred_dap=clue)
            so = model.sample_ode(images=images, daps=None, num_steps=20, vae=ovae, phytomer_vae=pvae)
            pos0 = so["phytomer_pos"][0].float(); exist = so["phytomer_existence"][0].float(); roll = so["phytomer_roll"][0].float()
            ordn = co["phytomer_ordinal"][0].float(); base = co["phytomer_base_logits"][0].float()
            scale0 = (so.get("phytomer_scale") if so.get("phytomer_scale") is not None else co.get("phytomer_scale"))[0].float(); lat0 = so["pred_latent"][0].float()
            rot, par = reconstruct_phytomer_rot(pos0.unsqueeze(0), roll.unsqueeze(0), ordn.unsqueeze(0), base.unsqueeze(0), exist=(exist > 0.5).float().unsqueeze(0))
            rot, par = rot[0].float(), par[0].float()
            parts0 = plant_from_nodes(pvae, pos0, rot, scale0, lat0, exist, par, M)
            iou0, _ = score(render_depth(renderer, parts0, zoom, dev), gt_depth)
        # variables
        pos = pos0.clone().requires_grad_("pos" in opt_set); scale = scale0.clone().requires_grad_("scale" in opt_set); lat = lat0.clone().requires_grad_("latent" in opt_set)
        params = [{"params": [pos], "lr": a.lr_pos}] if "pos" in opt_set else []
        if "scale" in opt_set: params.append({"params": [scale], "lr": a.lr_scale})
        if "latent" in opt_set: params.append({"params": [lat], "lr": a.lr_latent})
        opt = torch.optim.Adam(params)
        # input CHM at the two training zooms (cache channels 3 and 7)
        _zi = {1.0: 3, 2.0: 7, 4.0: 11, 8.0: 15}
        tgt = {float(z): images[0, _zi[float(z)]] for z in a.target_zooms.split(",") if images.shape[1] > _zi[float(z)]}
        best = (float("inf"), pos0.clone(), scale0.clone(), lat0.clone())
        for step in range(a.steps + (1 if a.keep_best else 0)):
            opt.zero_grad()
            parts = plant_from_nodes(pvae, pos, rot, scale, lat, exist, par, M)
            if parts.shape[0] == 0:
                break
            mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=dev)
            loss = torch.zeros((), device=dev)
            for z, t in tgt.items():
                if a.plant_centered:
                    pred = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="ground",
                                            focus_plant=True, include_depth=True, differentiable=True, image_size=128,
                                            zoom_factor=z, reference_window_size=1.2)[3]
                else:
                    pred = renderer.render_batched([mesh], azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, differentiable=True,
                                                   image_size=128, zoom_factor=z, reference_window_size=1.2,
                                                   centers=([gt_center] if gt_center is not None else None))[0, 3]
                canopy = (t > 0.005) | (pred > 0.005)
                l1 = F.smooth_l1_loss(pred, t, beta=0.02, reduction="none")
                loss_d = (l1 * canopy).sum() / canopy.sum().clamp(min=1)
                pm = torch.sigmoid((pred - 0.005) * 100.0); gm = (t > 0.005).float()
                dice = 1.0 - (2.0 * (pm * gm).sum() + 1e-4) / (pm.sum() + gm.sum() + 1e-4)
                loss = loss + 0.5 * loss_d + 1.0 * dice
            if a.reg_scale > 0:
                loss = loss + a.reg_scale * ((scale - scale0) ** 2).mean()
            if a.reg_latent > 0:
                loss = loss + a.reg_latent * ((lat - lat0) ** 2).mean()
            if a.keep_best and float(loss) < best[0]:
                best = (float(loss), pos.detach().clone(), scale.detach().clone(), lat.detach().clone())
            if step == a.steps:
                break
            loss.backward(); opt.step()
        if a.keep_best:
            with torch.no_grad():
                pos, scale, lat = best[1], best[2], best[3]
        with torch.no_grad():
            parts1 = plant_from_nodes(pvae, pos, rot, scale, lat, exist, par, M)
            iou1, _ = score(render_depth(renderer, parts1, zoom, dev), gt_depth)
            moved = float((pos - pos0).norm(dim=-1).mean() * 100)
            if a.save_renders:
                from PIL import Image as _Image
                os.makedirs(a.save_renders, exist_ok=True)
                for name_, parts_ in (("gt", gt_parts), ("before", parts0), ("after", parts1)):
                    if parts_.shape[0] == 0:
                        continue
                    mesh_ = renderer.geo_builder.build_mesh_from_part_tensor(parts_, device=dev)
                    rgbd_ = renderer.forward(mesh_, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="ground",
                                             focus_plant=False, include_depth=True, image_size=256, zoom_factor=zoom, reference_window_size=1.2)
                    _Image.fromarray((rgbd_[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
                        os.path.join(a.save_renders, f"{i}_{name_}.png"))
                json.dump({"index": i, "dap": dap, "iou_before": iou0, "iou_after": iou1, "jpeg": ds.samples[i].get("jpeg")},
                          open(os.path.join(a.save_renders, f"{i}_meta.json"), "w"))
        rows.append({"index": i, "dap": dap, "iou_before": iou0, "iou_after": iou1, "mean_node_move_cm": moved})
        print(f"idx {i:6d} DAP {dap:3d} IoU {iou0*100:5.1f} -> {iou1*100:5.1f}  (nodes moved {moved:.1f} cm)", flush=True)
    b = np.array([r["iou_before"] for r in rows]); c = np.array([r["iou_after"] for r in rows]); d = np.array([r["dap"] for r in rows])
    print(f"MEAN over {len(rows)} plants: {b.mean()*100:.1f} -> {c.mean()*100:.1f}  (DAP>15: {b[d>15].mean()*100:.1f} -> {c[d>15].mean()*100:.1f}) | opt={a.opt} steps={a.steps}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump({"checkpoint": a.checkpoint, "opt": a.opt, "steps": a.steps, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
