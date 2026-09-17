"""Approach 2 (DAP-conditioned test-time refinement): start from Approach 1's cold sample
(real_world/eval/run_approach1_cold.py), then refine node positions/scales/phytomer latents
for a few AdamW steps against the REAL crop's own pseudo-CHM depth + detector segmentation
mask, using the differentiable HeliosPyTorchRenderer — the analysis-by-synthesis loop already
proven on synthetic inputs in plant_recon/eval/eval_test_time_refinement.py, extended here
to a real photo. Differences from that script (see the plan, docs/ongoing 2026-09-15 notes on
test-time refinement):
  - no GT anywhere (no gt_parts/gt_depth/iou scoring) — a real photo has no ground truth.
  - optimizer is AdamW (small weight_decay) instead of plain Adam, per the user's request; the
    loop's own reg_scale/reg_latent quadratic priors are kept (found necessary in the synthetic
    experiments: without them, leaves inflate into flat polygons to fill the silhouette).
  - target is the real crop's own depth-anything pseudo-CHM (real_world/dataset/depth_anything_calib.py)
    at zooms 1/2/4/8, gated by --no_depth_loss (pseudo-depth has no ground-truth anchor — see
    the plan's open risks); the silhouette Dice target is the detector's own segmentation mask
    (real_world/dataset/real_plant_crop_utils.py::build_mask_pyramid) when available, falling
    back to depth-threshold silhouette like the synthetic script otherwise.
  - camera is always focus_plant=True at camera_height=1.5 (the rover's real height) — the
    synthetic script's --input_camera/gt_center concept (a cached GT-bbox-centred camera) has
    no analog for a real photo.
  - --scale_clip_mult (hard multiplicative bound on scale, default 1.5) and --reg_pos (quadratic
    position prior, default 20.0) were added 2026-09-15 after real-image runs reproduced the
    synthetic "canvas inflation" failure at a MUCH larger magnitude than reg_scale/reg_latent
    alone (even at 8x-40x the synthetic reg_scale) could stop. Root cause, diagnosed by
    elimination (isolating to zoom=1 only, silhouette-only loss, stronger reg_scale — all still
    inflated): the cold sample's coverage of the real canopy is so much smaller than on
    synthetic (in-distribution) images that Dice loss's incentive to grow area overwhelms a
    SOFT quadratic penalty on raw scale, because denormalize_packet_scales()
    (plant_recon/dataset/phytomer_packets.py) multiplies every organ in a phytomer's packet
    by the SAME scale row linearly, so leaf area grows quadratically with scale while the
    synthetic script's un-bounded position (no reg_pos existed there either) is equally free to
    spread. A HARD bound (clip scale after each step, not just penalize it) plus a position
    prior fixed it on the plants tested so far — see docs/experiments/20260915-real-image-first-test/20260915-real-image-first-test.md.
"""
import argparse
import datetime
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from plant_recon.dataset.phytomer_roll import derive_forward, roll_to_matrix
from plant_recon.dataset.phytomer_packets import matrix_to_rot6d
from plant_recon.eval.eval_gt_substitution_ablation import plant_from_nodes
from real_world.dataset.dap_from_timestamp import dap_from_filename
from real_world.dataset.real_field_dataset import RealFieldPlantDataset
from real_world.eval.run_approach1_cold import load_pipeline, sample_cold
from real_world.eval.viz_utils import colorize_depth, colorize_mask

_ZOOM_TO_CHANNEL = {1.0: 3, 2.0: 7, 4.0: 11, 8.0: 15}


def refine_one(model, pvae, renderer, item, pos0, rot, scale0, lat0, exist, par, parent_idx, M,
                dev, steps, opt_set, lrs, target_zooms, reg_scale, reg_latent, reg_pos, no_depth_loss,
                recompute_rot, keep_best, scale_clip_mult=0.0, scale_abs_max_len=0.0, scale_abs_max_rad=0.0,
                return_state=False):
    live_m = exist > 0.5
    has_par_m = parent_idx >= 0
    pos = pos0.clone().requires_grad_("pos" in opt_set)
    scale = scale0.clone().requires_grad_("scale" in opt_set)
    lat = lat0.clone().requires_grad_("latent" in opt_set)
    roll0 = torch.zeros(pos0.shape[0], device=dev)  # rot is already resolved; roll refinement needs recompute_rot
    roll_v = roll0.clone().requires_grad_("roll" in opt_set)
    if "roll" in opt_set and not recompute_rot:
        raise SystemExit("roll in --opt needs --recompute_rot")

    def _rot_par(pos_c, roll_c):
        if not recompute_rot:
            return rot, par
        R = roll_to_matrix(derive_forward(pos_c, parent_idx), roll_c)
        par_c = torch.where(has_par_m.unsqueeze(-1), pos_c[parent_idx.clamp(min=0)],
                             torch.where(live_m.unsqueeze(-1), torch.zeros_like(pos_c), pos_c))
        return matrix_to_rot6d(R), par_c

    params = []
    if "pos" in opt_set: params.append({"params": [pos], "lr": lrs["pos"]})
    if "scale" in opt_set: params.append({"params": [scale], "lr": lrs["scale"]})
    if "latent" in opt_set: params.append({"params": [lat], "lr": lrs["latent"]})
    if "roll" in opt_set: params.append({"params": [roll_v], "lr": lrs["roll"]})
    opt = torch.optim.AdamW(params, weight_decay=lrs["weight_decay"])

    image = item["image"].to(dev)  # (16, S, S)
    depth_tgt = {z: image[_ZOOM_TO_CHANNEL[z]] for z in target_zooms if not no_depth_loss}
    mask_pyr = item["mask_pyramid"]
    mask_tgt = {z: mask_pyr[zi].to(dev) for zi, z in enumerate([1.0, 2.0, 4.0, 8.0]) if z in target_zooms} if mask_pyr is not None else None

    best = (float("inf"), pos0.clone(), scale0.clone(), lat0.clone(), float("inf"))
    for step in range(steps + (1 if keep_best else 0)):
        opt.zero_grad()
        rot_c, par_c = _rot_par(pos, roll_v)
        parts = plant_from_nodes(pvae, pos, rot_c, scale, lat, exist, par_c, M)
        if parts.shape[0] == 0:
            break
        mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=dev)
        loss = torch.zeros((), device=dev)
        for z in target_zooms:
            pred = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.5, background="ground",
                                     focus_plant=True, include_depth=True, differentiable=True, image_size=image.shape[-1],
                                     zoom_factor=z, reference_window_size=1.2)[3]
            if z in depth_tgt:
                t = depth_tgt[z]
                canopy = (t > 0.005) | (pred > 0.005)
                l1 = F.smooth_l1_loss(pred, t, beta=0.02, reduction="none")
                loss = loss + 0.5 * (l1 * canopy).sum() / canopy.sum().clamp(min=1)
            pm = torch.sigmoid((pred - 0.005) * 100.0)
            gm = mask_tgt[z] if mask_tgt is not None and z in mask_tgt else (depth_tgt.get(z, pred) > 0.005).float()
            dice = 1.0 - (2.0 * (pm * gm).sum() + 1e-4) / (pm.sum() + gm.sum() + 1e-4)
            loss = loss + 1.0 * dice
        data_loss = float(loss)  # depth + dice only, BEFORE the plausibility priors below — the
        # actual "how well does this match the depth/mask target" number, tracked separately so
        # a capped run's target-matching quality can be compared against an uncapped run's.
        if reg_scale > 0:
            loss = loss + reg_scale * ((scale - scale0) ** 2).mean()
        if reg_latent > 0:
            loss = loss + reg_latent * ((lat - lat0) ** 2).mean()
        if reg_pos > 0:
            loss = loss + reg_pos * ((pos - pos0) ** 2).sum(-1).mean()
        if keep_best and float(loss) < best[0]:
            best = (float(loss), pos.detach().clone(), scale.detach().clone(), lat.detach().clone(), data_loss)
        if step == steps:
            break
        loss.backward(); opt.step()
        if scale_clip_mult > 0:
            with torch.no_grad():
                scale.clamp_(min=scale0 * (1.0 / scale_clip_mult), max=scale0 * scale_clip_mult)
        if scale_abs_max_len > 0 or scale_abs_max_rad > 0:
            # 2026-09-16: on the AgML (box-only, no seg mask) source, scale_clip_mult=1.5 -- even 1.1 -- was
            # not enough on 5/6 test plants: it multiplies whatever the (sometimes severely undersized) cold
            # start already was, so a MULTIPLICATIVE bound on a very wrong scale can still land visually large.
            # These are ABSOLUTE ceilings (FM units, same as scale0/the packet's own scale field), applied on
            # top, independent of how small the cold start was. Calibrated by directly measuring scale0's three
            # components ([length, radius, unused]) on live (exist>0.5) organs across 20 AgML cold-start plants:
            # length in [0.48, 4.12] FM (matches the GT training ceiling of ~4.0 almost exactly), radius in
            # [-0.11, 0.18] FM -- an order of magnitude smaller, so one scalar cap across both was wrong (the
            # v1 scaffolding's mistake). The wide, sign-flipping range seen in an earlier informal check
            # (-2.3 to 2.6) turned out to come from non-existent/padding slots, not live organs, once measured
            # per-component with a live mask -- so a per-component ceiling, plus reusing phytomer_scale()'s own
            # magnitude floors (0.25 length / 0.025 radius, plant_recon/dataset/phytomer_packets.py) as a
            # floor here too, guards against Adam pushing a live organ's scale negative (a degenerate/mirrored
            # mesh, a second plausible inflation contributor) without touching the "unused" 3rd column, which
            # stays tightly clustered near 1.0 regardless (not the inflation channel).
            with torch.no_grad():
                if scale_abs_max_len > 0:
                    scale[..., 0].clamp_(min=0.25, max=scale_abs_max_len)
                if scale_abs_max_rad > 0:
                    scale[..., 1].clamp_(min=0.025, max=scale_abs_max_rad)
    if keep_best:
        with torch.no_grad():
            pos, scale, lat = best[1], best[2], best[3]
    with torch.no_grad():
        rot_c, par_c = _rot_par(pos, roll_v)
        parts1 = plant_from_nodes(pvae, pos, rot_c, scale, lat, exist, par_c, M)
    moved = float((pos - pos0).norm(dim=-1).mean() * 100)
    if return_state:
        # the optimized state itself, for callers that need to re-decode it differently -- e.g. an
        # XML export, which needs the shoot structure plant_from_nodes flattens away
        # (real_world/dataset/helios_cold_start.part_tensor_with_shoots)
        return parts1, moved, best[4], (pos.detach(), rot_c.detach(), scale.detach(), lat.detach(),
                                         exist.detach(), par_c.detach())
    return parts1, moved, best[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs/checkpoints/hierarchical_fm_v10_cam/hierarchical_fm_epoch_085.pt"))
    ap.add_argument("--detector_weights", default=str(REPO_ROOT / "outputs/logs/20260915/real_plant_detector/weights/best.pt"))
    ap.add_argument("--images", default=str(REPO_ROOT / "real_world/data/roboflow_t4_plant_weed_seg/1/test/images/*.jpg"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--opt", default="pos,scale,latent")
    ap.add_argument("--lr_pos", type=float, default=3e-3)
    ap.add_argument("--lr_scale", type=float, default=2e-2)
    ap.add_argument("--lr_latent", type=float, default=2e-2)
    ap.add_argument("--lr_roll", type=float, default=2e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--reg_scale", type=float, default=5.0)
    ap.add_argument("--reg_latent", type=float, default=0.5)
    ap.add_argument("--scale_abs_max_len", type=float, default=5.0,
                    help="Hard ABSOLUTE ceiling on scale's length component (FM units, packet convention), "
                         "applied after --scale_clip_mult every step; 0 = off. Unlike the multiplicative clip, "
                         "this does not scale with how wrong the cold start already was -- added 2026-09-16 "
                         "after scale_clip_mult alone (even at 1.1x) failed to stop inflation on 5/6 AgML test "
                         "plants. Calibrated 2026-09-16 by measuring scale0 on live (exist>0.5) organs across "
                         "20 AgML cold-start plants: length ranged [0.48, 4.12] FM units, matching the training "
                         "GT ceiling (~4.0) almost exactly, so 5.0 gives headroom without allowing far-OOD growth.")
    ap.add_argument("--scale_abs_max_rad", type=float, default=0.3,
                    help="Same as --scale_abs_max_len for the radius component -- kept separate because radius's "
                         "live range ([-0.11, 0.18] FM units, 2026-09-16 measurement) is an order of magnitude "
                         "smaller than length's; one scalar cap across both components (the original v1 "
                         "scaffolding) was wrong for this reason.")
    ap.add_argument("--scale_clip_mult", type=float, default=1.5,
                     help="hard multiplicative bound: after each step, clamp scale into "
                          "[sampled_scale/mult, sampled_scale*mult]; 0 = off. 2026-09-15 finding: "
                          "denormalize_packet_scales() multiplies EVERY organ in a phytomer's packet "
                          "(internode/petiole/leaflets/peduncle/repro) by the SAME scale row linearly, "
                          "so leaf area grows quadratically with scale while reg_scale's quadratic "
                          "penalty on the raw value could not hold it back even at 8x/40x the synthetic "
                          "default -- a hard bound is a more robust fix than tuning the soft penalty. "
                          "1.5 (default) resolved the flat-plate failure on the plants tested so far; "
                          "raise it if refinement looks too conservative once more plants are checked.")
    ap.add_argument("--reg_pos", type=float, default=20.0,
                     help="penalty weight on (pos - sampled pos)^2 (meters^2); 0 disables it, matching "
                          "the synthetic eval_test_time_refinement.py, which never needed one in-distribution. "
                          "On real images (2026-09-15 finding) neither reg_scale nor the mask-vs-depth target "
                          "choice stops inflation alone -- position is completely unconstrained by default "
                          "and free to spread with scale to close the coverage gap; combined with "
                          "--scale_clip_mult this default (20) was needed on top of the clamp alone to stop "
                          "individual organs tiling the frame edge-to-edge.")
    ap.add_argument("--target_zooms", default="1,2,4,8")
    ap.add_argument("--no_depth_loss", action="store_true", help="silhouette-only Dice; use when the pseudo-depth calibration looks unreliable for this scene")
    ap.add_argument("--recompute_rot", action="store_true")
    ap.add_argument("--keep_best", action="store_true", default=True)
    ap.add_argument("--planted_date", default="", help="YYYY-MM-DD; see run_approach1_cold.py's --planted_date docstring "
                     "-- overrides Stage 1's self-predicted DAP clue with the timestamp-derived one for both the cold "
                     "sample and (implicitly, since refinement starts from it) this script.")
    ap.add_argument("--out", default=str(REPO_ROOT / "use_cases/real_world/eval/output/approach2_refine"))
    a = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, pvae, ovae, renderer, M = load_pipeline(a.checkpoint, dev)
    planted_date = datetime.date.fromisoformat(a.planted_date) if a.planted_date else None

    paths = sorted(glob.glob(a.images))
    ds = RealFieldPlantDataset(paths, a.detector_weights, conf=a.conf)
    os.makedirs(a.out, exist_ok=True)
    opt_set = set(a.opt.split(","))
    target_zooms = [float(z) for z in a.target_zooms.split(",")]
    lrs = {"pos": a.lr_pos, "scale": a.lr_scale, "latent": a.lr_latent, "roll": a.lr_roll, "weight_decay": a.weight_decay}

    n = min(a.limit, len(ds))
    rows = []
    for i in range(n):
        it = ds[i]
        images = it["image"].unsqueeze(0).to(dev)
        ts_dap = dap_from_filename(it["jpeg"], planted_date) if planted_date else None
        daps_override = torch.tensor([float(ts_dap)], device=dev) if ts_dap is not None else None
        parts0, pred_dap, state = sample_cold(model, pvae, ovae, images, dev, M, daps=daps_override)
        pos0, rot, scale0, lat0, exist, par, parent_idx = state
        if parts0.shape[0] == 0:
            print(f"[{i+1}/{n}] {it['prefix']}: empty sample, skipping"); continue
        parts1, moved, data_loss = refine_one(model, pvae, renderer, it, pos0, rot, scale0, lat0, exist, par, parent_idx, M, dev,
                                    a.steps, opt_set, lrs, target_zooms, a.reg_scale, a.reg_latent, a.reg_pos, a.no_depth_loss,
                                    a.recompute_rot, a.keep_best, a.scale_clip_mult, a.scale_abs_max_len, a.scale_abs_max_rad)
        print(f"[{i+1}/{n}] {it['prefix']}: pred_dap={pred_dap:.1f} nodes moved {moved:.1f} cm  data_loss={data_loss:.3f}")

        for tag, parts in (("before", parts0), ("after", parts1)):
            if parts.shape[0] == 0:
                continue
            mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=dev)
            with torch.no_grad():
                rgbd = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.5, background="ground",
                                         focus_plant=True, include_depth=True, image_size=256, zoom_factor=1.0,
                                         reference_window_size=1.2)
            png = (rgbd[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(png).save(os.path.join(a.out, f"{it['prefix']}_{tag}.png"))
            Image.fromarray(colorize_depth(rgbd[3].cpu().numpy())).save(os.path.join(a.out, f"{it['prefix']}_{tag}_depth.png"))
        crop_rgb = ((it["image"][:3] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(crop_rgb).save(os.path.join(a.out, f"{it['prefix']}_input.png"))
        Image.fromarray(colorize_depth(it["image"][3].numpy())).save(os.path.join(a.out, f"{it['prefix']}_input_depth.png"))
        if it["mask_pyramid"] is not None:
            # zoom index 0 = 1x — the detector's own segmentation mask, the actual Dice-loss
            # foreground target above (mask_tgt in refine_one, when available)
            Image.fromarray(colorize_mask(it["mask_pyramid"][0].numpy())).save(os.path.join(a.out, f"{it['prefix']}_input_mask.png"))
        rows.append({"prefix": it["prefix"], "jpeg": it["jpeg"], "pred_dap": pred_dap, "timestamp_dap": ts_dap, "nodes_moved_cm": moved, "data_loss": data_loss})

    json.dump({"checkpoint": a.checkpoint, "opt": a.opt, "steps": a.steps, "rows": rows}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(f"Done. {len(rows)} plants -> {a.out}")


if __name__ == "__main__":
    main()
