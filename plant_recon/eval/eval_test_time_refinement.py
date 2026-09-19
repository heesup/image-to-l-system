"""Test-time refinement (analysis-by-synthesis): after sampling a plant, optimise its node positions, scales and
phytomer latents for a few Adam steps against the INPUT canopy height map with the training render loss
(depth smooth-L1 + silhouette dice at zoom 1x/2x), then score under the strict 256 px protocol.

Reads the same checkpoint / eval set as eval_gt_substitution_ablation.py; prints per-plant IoU before -> after and
the mean, for the variable sets given by --opt (e.g. pos,scale,latent). Nothing is trained or saved."""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from plant_recon.dataset.part_array_dataset import PartArrayDataset, attach_parent_links, FM_OT_END, FM_BASE_START
from plant_recon.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel, reconstruct_phytomer_rot
from plant_recon.dataset.phytomer_topology import chain_phytomers
from plant_recon.dataset.phytomer_roll import derive_forward, roll_to_matrix
from plant_recon.dataset.phytomer_packets import matrix_to_rot6d
from plant_recon.models.phytomer_vae import PhytomerVAE
from plant_recon.models.organ_latent_vae import OrganLatentVAE
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from plant_recon.models.plant_organ_array import NUM_ORGAN_TYPES
from plant_recon.eval.eval_hierarchical_self_consistency import decode_predictions_to_part_tensor
from plant_recon.eval.eval_gt_substitution_ablation import plant_from_nodes, render_depth, score
from plant_recon.eval.ckpt_compat import fix_ckpt_args


def refine_plant(renderer, pvae, M, images, gt_center, pos0, rot, par, roll, scale0, lat0, exist, parent_idx, live_m, has_par_m, opt_set, a, exist_prob0=None):
    """Test-time refinement of one plant: Adam on node positions / scales / latents / existence / roll against the
    INPUT canopy height map channels (no ground truth), rendered in the input's camera frame.

    "roll" in opt_set optimises the twist angle around a FOREVER-FROZEN forward axis (derived once from pos0, the
    same fixed axis+parent_pos the default pos/scale/latent-only refinement already uses) -- NOT `--recompute_rot`'s
    axis, which re-derives the forward axis from the CURRENT (moving) node positions every step and measured worse
    even with pos/scale/latent alone (2026-09-16: 33.5 -> 60.0 vs the frozen default's 67.6): re-deriving the axis
    from a position that is itself being optimised compounds into a rough, unstable loss landscape. Freezing the
    axis (and parent_pos, exactly as the default already does for position) removes that coupling so roll is a
    small, independent search around a fixed frame. `--recompute_rot` is kept only as a slower legacy comparison
    and is ignored whenever "roll" is requested.

    "exist" in opt_set optimises each node's existence continuously: a sigmoid-parameterised probability (init from
    `exist_prob0`, the model's own PRE-threshold sigmoid output -- not the >0.5 hard mask in `exist`, which cannot
    be corrected once baked in, see 2026-09-15's --exist_thresh finding) is passed as a per-organ ALPHA to
    build_mesh_from_part_tensor(existence=...) -- the renderer's own soft-existence channel, already alpha-composited
    against the background for both rgb and depth (helios_pytorch_renderer), and already used this way (normally
    with existence.detach()) by the training render block. A suppressed (false-positive) node fades toward the
    background depth/silhouette at its pixels and a promoted (false-negative, previously masked out) node fades in
    -- both fully differentiable through the render loss. This replaces two things: the boolean `cls>0 & exist>0.5`
    row gate inside plant_from_nodes (no gradient, and can only ever narrow the node set, never revive a masked-off
    one -- plant_from_nodes(soft_exist=True) instead keeps every cls>0 row and returns a matching per-row alpha),
    and a first attempt (2026-09-16, dropped) that multiplied the node's SCALE by the probability instead: that
    is a real geometry change with its own depth footprint, not a fade, and does not handle occlusion between
    overlapping organs the way alpha compositing does. A quadratic penalty toward `exist_prob0` (a.reg_exist)
    keeps this from turning on every candidate slot.

    `a` carries steps, lr_*, reg_*, keep_best, target_zooms, plant_centered, recompute_rot (argparse namespace or
    any object with those attributes). Returns (pos, scale, lat, roll, rot, parent_pos, final_input_loss, exist)."""
    dev = images.device
    # variables
    # .detach() before .clone(): pos0/scale0/lat0/roll are always meant as plain values to start the search
    # from, but a caller may hand in a tensor that is still attached to an active autograd graph (e.g.
    # eval_guided_sampling.py's guided sample_ode() output, read out mid-graph) -- requires_grad_() below
    # would fail on a non-leaf tensor otherwise ("you can only change requires_grad flags of leaf variables").
    pos = pos0.detach().clone().requires_grad_("pos" in opt_set); scale = scale0.detach().clone().requires_grad_("scale" in opt_set)
    lat = lat0.detach().clone().requires_grad_("latent" in opt_set)
    roll_v = roll.detach().clone().requires_grad_("roll" in opt_set)
    fwd0 = derive_forward(pos0, parent_idx) if ("roll" in opt_set or a.recompute_rot) else None

    exist_logit = None
    if "exist" in opt_set:
        assert exist_prob0 is not None, "exist in --opt needs exist_prob0 (the model's pre-threshold existence probability)"
        exist_logit = torch.logit(exist_prob0.detach().clamp(1e-3, 1 - 1e-3)).clone().requires_grad_(True)

    def _rot_par(pos_c, roll_c):
        if "roll" in opt_set:
            R = roll_to_matrix(fwd0, roll_c)   # frozen axis (derived once from pos0), never re-derived from moving pos_c
            return matrix_to_rot6d(R), par     # parent_pos frozen too, matching the default pos-only path
        if a.recompute_rot:
            R = roll_to_matrix(derive_forward(pos_c, parent_idx), roll_c)
            par_c = torch.where(has_par_m.unsqueeze(-1), pos_c[parent_idx.clamp(min=0)],
                                torch.where(live_m.unsqueeze(-1), torch.zeros_like(pos_c), pos_c))
            return matrix_to_rot6d(R), par_c
        return rot, par
    params = [{"params": [pos], "lr": a.lr_pos}] if "pos" in opt_set else []
    if "scale" in opt_set: params.append({"params": [scale], "lr": a.lr_scale})
    if "latent" in opt_set: params.append({"params": [lat], "lr": a.lr_latent})
    if "roll" in opt_set: params.append({"params": [roll_v], "lr": a.lr_roll})
    if "exist" in opt_set: params.append({"params": [exist_logit], "lr": a.lr_exist})
    opt = torch.optim.Adam(params)
    # Cosine decay of every group's lr to --lr_final_frac of its initial value. With the constant lr the
    # search keeps taking full-size steps after it has essentially arrived, so the last steps jitter around
    # the optimum instead of settling into it; that jitter is why --keep_best was needed to avoid ending on
    # a worse step than the best one seen. Matters more the longer the run.
    lr0 = [g["lr"] for g in opt.param_groups]
    def _set_lr(step):
        if getattr(a, "lr_schedule", "const") != "cosine":
            return
        f = getattr(a, "lr_final_frac", 0.05)
        t = min(1.0, step / max(1, a.steps))
        m = f + (1.0 - f) * 0.5 * (1.0 + math.cos(math.pi * t))
        for g, l0 in zip(opt.param_groups, lr0):
            g["lr"] = l0 * m
    refine_px = int(getattr(a, "refine_px", 128))
    # input CHM at the two training zooms (cache channels 3 and 7)
    _zi = {1.0: 3, 2.0: 7, 4.0: 11, 8.0: 15}
    hires = getattr(a, "_hires_tgt", None)      # (4, P, P) CHM pyramid re-rendered by tools/render_hires_targets.py
    if hires is not None:
        _zl = [1.0, 2.0, 4.0, 8.0]
        tgt = {float(z): hires[_zl.index(float(z))].to(dev).float() for z in a.target_zooms.split(",")}
    else:
        tgt = {float(z): images[0, _zi[float(z)]] for z in a.target_zooms.split(",") if images.shape[1] > _zi[float(z)]}
    if refine_px != next(iter(tgt.values())).shape[-1]:
        # the cached input CHM is stored at the dataset's image_size; match the render grid so the loss is
        # pixel-aligned. Upsampling the target does not add information -- it lets the finer render be
        # compared without resampling the PREDICTION, which would blur the gradient it carries.
        tgt = {z: F.interpolate(t.view(1, 1, *t.shape), size=(refine_px, refine_px), mode="bilinear",
                                align_corners=False)[0, 0] for z, t in tgt.items()}
    best = (float("inf"), pos0.clone(), scale0.clone(), lat0.clone(), roll.clone(), (exist_logit.detach().clone() if exist_logit is not None else None))
    for step in range(a.steps + (1 if a.keep_best else 0)):
        opt.zero_grad()
        rot_c, par_c = _rot_par(pos, roll_v)
        if exist_logit is not None:
            exist_p = torch.sigmoid(exist_logit)
            parts, alpha = plant_from_nodes(pvae, pos, rot_c, scale, lat, exist_p, par_c, M, soft_exist=True)
        else:
            exist_p = None; alpha = None
            parts = plant_from_nodes(pvae, pos, rot_c, scale, lat, exist, par_c, M)
        if parts.shape[0] == 0:
            break
        mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, existence=alpha, device=dev)
        loss = torch.zeros((), device=dev)
        for z, t in tgt.items():
            if a.plant_centered:
                pred = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="ground",
                                        focus_plant=True, include_depth=True, differentiable=True, image_size=refine_px,
                                        zoom_factor=z, reference_window_size=1.2)[3]
            else:
                pred = renderer.render_batched([mesh], azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, differentiable=True,
                                               image_size=refine_px, zoom_factor=z, reference_window_size=1.2,
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
        if exist_p is not None and a.reg_exist > 0:
            loss = loss + a.reg_exist * ((exist_p - exist_prob0) ** 2).mean()
        if a.keep_best and float(loss) < best[0]:
            best = (float(loss), pos.detach().clone(), scale.detach().clone(), lat.detach().clone(), roll_v.detach().clone(),
                    (exist_logit.detach().clone() if exist_logit is not None else None))
        if step == a.steps:
            break
        loss.backward(); _set_lr(step); opt.step()
        # Absolute realized-scale ceilings (FM units, packet convention), applied after every step, on top of the
        # soft reg_scale penalty. Mirrors use_cases/real_world/eval/run_approach2_refine.py: the multiplicative
        # scale bound is on the WRONG thing (s_a), and a leaf's area grows quadratically with scale, so the soft
        # penalty alone cannot stop inflation against a loose target. Absolute metres-like ceilings stop far-OOD
        # growth regardless of how wrong the cold start already was. 0 = off.
        _smax_len = float(getattr(a, "scale_abs_max_len", 0.0))
        _smax_rad = float(getattr(a, "scale_abs_max_rad", 0.0))
        if _smax_len > 0 or _smax_rad > 0:
            with torch.no_grad():
                if _smax_len > 0:
                    scale[..., 0].clamp_(min=0.25, max=_smax_len)
                if _smax_rad > 0:
                    scale[..., 1].clamp_(min=0.025, max=_smax_rad)
    if a.keep_best:
        with torch.no_grad():
            pos, scale, lat, roll_v = best[1], best[2], best[3], best[4]
            if exist_logit is not None:
                exist_logit = best[5]
    with torch.no_grad():
        rot_c, par_c = _rot_par(pos, roll_v)
        if exist_logit is not None:
            # Final materialisation is a crisp in/out decision, but the threshold must match what the search
            # optimised against. The old hardcoded >0.5 ignored --exist_thresh, so a node that settled at 0.49
            # (with --exist_thresh 0.3) still vanished from the scored plant: the 2026-09-17 "continuous
            # existence costs 1.8 points" finding is exactly this mismatch, not the idea itself.
            _thr = float(getattr(a, "exist_thresh", 0.5))
            exist = (torch.sigmoid(exist_logit) > _thr).float()
            # scale is returned unmodified -- alpha only faded the RENDER during the search, it never touched shape
    final_loss = best[0] if a.keep_best else float(loss)
    return pos.detach(), scale.detach(), lat.detach(), roll_v.detach(), rot_c, par_c, final_loss, exist.detach() if torch.is_tensor(exist) else exist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_set", default="outputs/checkpoints/hierarchical_fm_v9/eval_set.json")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--opt", default="pos,scale,latent", help="comma list from pos,scale,latent,roll,exist (roll/exist: see refine_plant docstring)")
    ap.add_argument("--lr_pos", type=float, default=3e-3)
    ap.add_argument("--lr_scale", type=float, default=2e-2)
    ap.add_argument("--lr_latent", type=float, default=2e-2)
    ap.add_argument("--out", default="")
    ap.add_argument("--plant_centered", action="store_true",
                    help="Render the prediction with the camera centred on its own bounding box (focus_plant), the frame "
                         "the cached input CHM was rendered in (plant-bbox-centred), instead of the fixed origin window. "
                         "Measured 2026-09-15: the cached CHM matches a bbox-centred GT render at 73-89%% IoU but the "
                         "origin-window one at only 41-76%%.")
    ap.add_argument("--input_camera", action="store_true", default=True,
                    help="render the prediction in the camera frame the cached input was rendered in (GT plant bbox centre); default on")
    ap.add_argument("--origin_camera", action="store_true", help="use the origin-centred window instead of --input_camera (the pre-2026-09-15 behaviour)")
    ap.add_argument("--save_renders", default="", help="folder: save gt / before / after top-view renders (256 px PNG) per plant")
    ap.add_argument("--only", default="", help="comma-separated plant indices to process (default: the whole eval set)")
    ap.add_argument("--target_zooms", default="1,2,4,8", help="comma list of cache zoom levels (1,2,4,8) whose depth channels are the refinement targets; the 4x/8x levels add signal for plants that are only a few pixels wide at 1x/2x")
    ap.add_argument("--exist_thresh", type=float, default=0.5, help="node existence threshold applied to the sampled existence probabilities (default 0.5; the trained models activate ~60 of 73 GT nodes at 0.5, so a lower threshold adds nodes for the refinement to place)")
    ap.add_argument("--multi_start", action="store_true", help="refine twice, from the sampled latent and from the DAP-spanning mean latent, and keep the start with the lower final input loss")
    ap.add_argument("--n_starts", type=int, default=1,
                    help="Multi-hypothesis render-and-select: draw this many plants from the flow (each a different "
                         "latent draw), refine each against the INPUT, keep the start with the lowest final input "
                         "loss. 1 = the old single-sample behaviour. The flow is the only readout with realistic "
                         "spread, so selection over its samples is how the ill-posed per-organ shape is answered "
                         "without a second viewpoint (20260918-stage3-conditioning-settled §6).")
    ap.add_argument("--init_mean_latent", action="store_true", help="start the refinement from the training-set mean latent (the model's latent_mu buffer) instead of the flow-sampled latent; on full data the sampled latent scored 6-7 points below the mean latent")
    ap.add_argument("--recompute_rot", action="store_true", help="LEGACY, underperforms: re-derive each node's rotation from the CURRENT (moving) parent->node segment every step, instead of the frozen axis refine_plant uses by default and for --opt roll. Ignored whenever roll is in --opt.")
    ap.add_argument("--lr_roll", type=float, default=2e-2)
    ap.add_argument("--lr_exist", type=float, default=3e-2)
    ap.add_argument("--reg_exist", type=float, default=0.3, help="penalty weight on (existence_prob - sampled existence_prob)^2, keeps refinement from turning on every candidate slot")
    ap.add_argument("--reg_scale", type=float, default=5.0, help="penalty weight on (scale - sampled scale)^2, keeps leaves from inflating to fill the silhouette")
    ap.add_argument("--reg_latent", type=float, default=0.5, help="penalty weight on mean (latent - sampled latent)^2")
    ap.add_argument("--scale_abs_max_len", type=float, default=0.0,
                    help="Hard ABSOLUTE ceiling on scale's length component (FM units, packet convention), applied "
                         "after every refinement step; 0 = off. Bounds the REALIZED organ size in metres-like units "
                         "rather than the multiplicative s_a bound, so a leaf cannot inflate regardless of how small "
                         "the cold start was (mirrors run_approach2_refine.py; the GT training ceiling is ~4.0).")
    ap.add_argument("--scale_abs_max_rad", type=float, default=0.0,
                    help="Same as --scale_abs_max_len for the radius component (live range ~[-0.11, 0.18] FM).")
    ap.add_argument("--target_dir", default="",
                    help="Folder of <prefix>_chm<PX>.pt CHM pyramids re-rendered at high resolution "
                         "(tools/render_hires_targets.py). Replaces the cached CHM as the refinement target, which "
                         "is the only way a render finer than the cache's 256 px can show a real gain.")
    ap.add_argument("--input_px", type=int, default=128,
                    help="Resolution the cached input is loaded at -- which is what the refinement's CHM TARGET is. "
                         "The cache holds 256 px on disk, so the default 128 discards half of it for free. Raising "
                         "the RENDER (--refine_px) past the target only sharpens the prediction against a blurry "
                         "target; raising this is what adds real information. The backbone is unaffected either way "
                         "(it resizes to 224), except that 256 reaches it as a downsample instead of an upsample.")
    ap.add_argument("--lr_schedule", default="const", choices=["const", "cosine"],
                    help="Decay every optimised variable's learning rate along the run (cosine, to --lr_final_frac). "
                         "The constant default keeps taking full-size steps after the search has arrived.")
    ap.add_argument("--lr_final_frac", type=float, default=0.05)
    ap.add_argument("--refine_px", type=int, default=128,
                    help="Render resolution INSIDE the refinement loop (the strict protocol scores at 256 either "
                         "way). At 128 px a 1.2 m window is 9.4 mm per pixel, so a cowpea leaflet is ~4 px and the "
                         "gradient is dominated by the global silhouette; 256 halves that.")
    ap.add_argument("--rgb_override_dir", default="",
                    help="folder of <prefix>_helios_rgb.pt files ((12, S, S): level-major RGB in the cache's [-1, 1] convention, written by "
                         "render_helios_eval_crops.py). When given, the 12 RGB planes of each plant's 16-channel input are replaced by them and "
                         "the CHM planes are kept, so the same checkpoint is scored on the same plants with only the pixels changed "
                         "(appearance-gap measurement, docs/engineering/20260916-sim-to-real-assessment-and-plan/20260916-sim-to-real-assessment-and-plan.md §4)")
    ap.add_argument("--keep_best", action="store_true",
                    help="Return the variables of the step with the lowest INPUT loss (model selection on the input only), "
                         "not the last step -- guards against the divergent plants.")
    a = ap.parse_args()
    if a.origin_camera:
        a.input_camera = False
    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))
    model = HierarchicalPartFlowMatchingModel(
        max_phytomers=args["max_phytomers"], slots_per_phytomer=args["slots_per_phytomer"], node_dim=args["node_dim"],
        num_classes=NUM_ORGAN_TYPES, image_size=128, patch_size=8, embed_dim=args["embed_dim"], vit_layers=args["vit_layers"],
        vit_heads=args["vit_heads"], coarse_layers=args["coarse_layers"], fine_layers=args["fine_layers"],
        flow_granularity=args["flow_granularity"], phytomer_latent_dim=args["phytomer_latent_dim"], backbone=args["backbone"],
        freeze_backbone=True, init_phytomer_count=args.get("init_phytomer_count", 50.0),
        stage3_geometry=bool(args.get("stage3_geometry", False)),
        stage3_absolute=bool(args.get("stage3_absolute", False)), multizoom=bool(args.get("multizoom", False)),
        node_token_window=int(args.get("node_token_window", 1)), stage3_regression=bool(args.get("stage3_regression", False)),
        use_depth=bool(args.get("use_depth", False))).to(dev)
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
                          pkt_cache_dir=args.get("pkt_cache_dir") or None, species="cowpea", image_size=a.input_px)
    # Resolve the eval set by PREFIX, not by the stored "indices".
    #
    # Two reasons. A holdout_set.json has no "indices" at all -- held-out plants are excluded from the
    # training subset, so they have no position in its index space -- and without this the script died with
    # KeyError: 'indices'. More quietly, the stored indices of an eval_set.json are relative to the SUBSET the
    # run trained on (MAX_TRAIN_SAMPLES), while `ds` here is built over the full data_dir: the same 20 plants
    # are indices 148/263/1089... in a 10k-sample run and 1354/2616/10833... in a full-data one, so indexing
    # the full dataset with subset indices silently evaluates 20 different plants (2026-09-18).
    # Prefixes are the stable identity across both files and both index spaces.
    _es = json.load(open(a.eval_set))
    _by_prefix = {smp["prefix"]: i for i, smp in enumerate(ds.samples)}
    _wanted = [smp["prefix"] for smp in _es["samples"]]
    _missing = [q for q in _wanted if q not in _by_prefix]
    if _missing:
        raise SystemExit(f"{len(_missing)} of {len(_wanted)} eval prefixes are absent from {args['data_dir']}, "
                         f"first: {_missing[0]}")
    idxs = [_by_prefix[q] for q in _wanted]
    if "indices" in _es and idxs != _es["indices"]:
        print(f"note: resolved {len(idxs)} plants by prefix; the stored indices differ (they are relative to "
              f"that run's training subset, not to the full dataset)", flush=True)
    if a.only:
        idxs = [int(x) for x in a.only.split(",")]
    opt_set = set(a.opt.split(","))
    rows = []
    mean_lat = None
    if a.init_mean_latent or a.multi_start:
        # Mean GT phytomer latent over 300 random training plants (eval plants excluded), spanning all growth
        # stages. NOT the model's latent_mu buffer: that one is gathered from the first plants in dataset order
        # (one growth stage), and starting from it scored 9-12% before refinement against 34-36% for a
        # DAP-spanning mean (2026-09-15).
        g = torch.Generator().manual_seed(0); pool = [j for j in torch.randperm(len(ds), generator=g).tolist() if j not in set(idxs)][:300]
        lat_sum, lat_n = None, 0
        with torch.no_grad():
            for j in pool:
                pk = ds[j].get("pkt")
                if pk is None: continue
                l_ = pvae.encode(pvae.pack_input(pk["packets"].to(dev).float(), pk["presence"].to(dev)))[0].float()
                lat_sum = l_.sum(0) if lat_sum is None else lat_sum + l_.sum(0); lat_n += l_.shape[0]
        mean_lat = lat_sum / max(lat_n, 1)
        print(f"init_mean_latent: mean over {lat_n} phytomers from {len(pool)} training plants", flush=True)
    for i in idxs:
        it = ds[i]; images = it["image"].unsqueeze(0).to(dev); dap = int(it["dap"].item()); zoom = 8.0 if dap <= 15 else 1.0
        if a.rgb_override_dir:
            # appearance-gap measurement: Helios raytraced RGB in place of the flat cache render, CHM planes untouched
            ov = torch.load(os.path.join(a.rgb_override_dir, f"{ds.samples[i]['prefix']}_helios_rgb.pt"), map_location="cpu", weights_only=True).float()
            if ov.shape[-1] != images.shape[-1]:
                ov = F.interpolate(ov.unsqueeze(0), size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
            for l_ in range(4):
                images[0, 4 * l_:4 * l_ + 3] = ov[3 * l_:3 * l_ + 3].to(dev)
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
        a._hires_tgt = None
        if a.target_dir:
            import glob as _glob
            _m = _glob.glob(os.path.join(a.target_dir, f"{ds.samples[i]['prefix']}_chm*.pt"))
            if _m:
                a._hires_tgt = torch.load(_m[0], map_location="cpu", weights_only=True)

        # Multi-hypothesis render-and-select (--n_starts, 2026-09-18 Phase G): the flow is a PROPOSAL
        # distribution -- its wrong-but-right-variance latents are worth ~8 IoU under GT geometry and it is
        # the only readout with realistic spread (20260918-stage3-conditioning-settled §6). So sample N
        # plants from it, refine each against the INPUT (never GT), and keep the start whose refined render
        # fits the input best. Selection is on refine_plant's final INPUT loss, so the deployable path
        # stays GT-free. --multi_start adds one further start from the DAP-spanning mean latent, on the
        # selected sample's own scaffold (latent swapped only).
        best_run = None
        for _s in range(max(1, a.n_starts)):
            with torch.no_grad():
                so = model.sample_ode(images=images, daps=None, num_steps=20, vae=ovae, phytomer_vae=pvae)
                pos0 = so["phytomer_pos"][0].float(); exist_prob0 = so["phytomer_existence"][0].float(); roll = so["phytomer_roll"][0].float()
                exist = (exist_prob0 > a.exist_thresh).float()
                ordn = co["phytomer_ordinal"][0].float(); base = co["phytomer_base_logits"][0].float()
                scale0 = (so.get("phytomer_scale") if so.get("phytomer_scale") is not None else co.get("phytomer_scale"))[0].float(); lat0 = so["pred_latent"][0].float()
                if a.init_mean_latent:
                    lat0 = mean_lat.reshape(1, -1).expand_as(lat0).clone()
                rot, par = reconstruct_phytomer_rot(pos0.unsqueeze(0), roll.unsqueeze(0), ordn.unsqueeze(0), base.unsqueeze(0), exist=(exist > 0.5).float().unsqueeze(0))
                rot, par = rot[0].float(), par[0].float()
                # topology chained once from the sampled positions (discrete, non-differentiable); the rotation and
                # parent position can then be re-derived differentiably from the current pos / roll (--recompute_rot)
                parent_idx, _, _ = chain_phytomers(pos0, ordinal=ordn, is_base=(base > 0).float(), exist=(exist > 0.5).float())
                live_m = exist > 0.5; has_par_m = parent_idx >= 0
                parts0 = plant_from_nodes(pvae, pos0, rot, scale0, lat0, exist, par, M)
                iou0, _ = score(render_depth(renderer, parts0, zoom, dev), gt_depth)
            pos, scale, lat, roll_v, rot_c, par_c, loss_a, exist_after = refine_plant(renderer, pvae, M, images, gt_center, pos0, rot, par, roll, scale0, lat0, exist, parent_idx, live_m, has_par_m, opt_set, a, exist_prob0=exist_prob0)
            if best_run is None or loss_a < best_run["loss"]:
                best_run = {"loss": loss_a, "pos": pos, "scale": scale, "lat": lat, "roll": roll_v, "rot": rot_c, "par": par_c,
                            "exist_after": exist_after, "pos0": pos0, "exist": exist, "iou0": iou0, "start": _s,
                            "rot0": rot, "par0": par, "roll0": roll, "scale0": scale0, "parent_idx": parent_idx,
                            "exist_prob0": exist_prob0, "lat0": lat0, "parts0": parts0}
        if a.multi_start:
            # extra start from the DAP-spanning mean latent on the SELECTED sample's scaffold (everything
            # except the latent is the selected start's own sampled state); keep whichever ends with the
            # lower final INPUT loss. Same semantics as the pre-Phase-G behaviour at n_starts=1.
            lat_m = mean_lat.reshape(1, -1).expand_as(best_run["lat0"]).clone()
            out_b = refine_plant(renderer, pvae, M, images, gt_center, best_run["pos0"], best_run["rot0"], best_run["par0"],
                                 best_run["roll0"], best_run["scale0"], lat_m, best_run["exist"], best_run["parent_idx"],
                                 best_run["exist"] > 0.5, best_run["parent_idx"] >= 0, opt_set, a,
                                 exist_prob0=best_run["exist_prob0"])
            if out_b[6] < best_run["loss"]:
                best_run.update({"loss": out_b[6], "pos": out_b[0], "scale": out_b[1], "lat": out_b[2], "roll": out_b[3],
                                 "rot": out_b[4], "par": out_b[5], "exist_after": out_b[7], "start": -1})
        loss_a = best_run["loss"]; pos = best_run["pos"]; scale = best_run["scale"]; lat = best_run["lat"]; roll_v = best_run["roll"]
        rot_c = best_run["rot"]; par_c = best_run["par"]; exist_after = best_run["exist_after"]
        pos0 = best_run["pos0"]; exist = best_run["exist"]; iou0 = best_run["iou0"]; sel = best_run["start"]
        parts0 = best_run["parts0"]
        with torch.no_grad():
            parts1 = plant_from_nodes(pvae, pos, rot_c, scale, lat, exist_after, par_c, M)
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
        n_before, n_after = int((exist > 0.5).sum()), int((exist_after > 0.5).sum())
        pred_dap = float(clue.reshape(-1)[0].item())
        pid_ = it.get("phytomer_ids")   # (N, 2) (shoot_id, phytomer_idx) per organ row, -1 padded
        n_gt_phyto = int(torch.unique(pid_[(exist_gt.cpu() > 0.5) & (pid_[:, 0] >= 0)], dim=0).shape[0]) if pid_ is not None else -1
        rows.append({"index": i, "dap": dap, "iou_before": iou0, "iou_after": iou1, "mean_node_move_cm": moved,
                    "n_nodes_before": n_before, "n_nodes_after": n_after, "n_phytomers_gt": n_gt_phyto,
                    "n_organs_gt": int(exist_gt.sum().item()), "pred_dap": pred_dap, "sel_start": sel})
        extra = f"  nodes {n_before:>3d} -> {n_after:>3d}" if "exist" in opt_set else f"  nodes {n_before:>3d}"
        extra += f"  start {sel}/{max(1, a.n_starts)}" if a.n_starts > 1 or a.multi_start else ""
        print(f"idx {i:6d} DAP {dap:3d} (pred {pred_dap:5.1f}) IoU {iou0*100:5.1f} -> {iou1*100:5.1f}  (nodes moved {moved:.1f} cm){extra}", flush=True)
    b = np.array([r["iou_before"] for r in rows]); c = np.array([r["iou_after"] for r in rows]); d = np.array([r["dap"] for r in rows])
    print(f"MEAN over {len(rows)} plants: {b.mean()*100:.1f} -> {c.mean()*100:.1f}  (DAP>15: {b[d>15].mean()*100:.1f} -> {c[d>15].mean()*100:.1f}) | opt={a.opt} steps={a.steps}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump({"checkpoint": a.checkpoint, "opt": a.opt, "steps": a.steps, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
