"""Approach 1 (cold generation): run the trained 4-stage model forward on a real per-plant crop
with NO optimization — a direct test of how well the pipeline generalizes to real images.
Mirrors the checkpoint-loading and initial-sampling code in
plant_recon/eval/eval_test_time_refinement.py (lines 62-110) exactly, minus everything
GT/scoring-related (there is no ground truth for a real photo) and minus the optimization loop
(that is Approach 2, real_world/eval/run_approach2_refine.py, which continues from here).
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from plant_recon.eval.eval_gt_substitution_ablation import plant_from_nodes
from plant_recon.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel, reconstruct_phytomer_rot, compute_matryoshka_slice
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from plant_recon.models.organ_latent_vae import OrganLatentVAE
from plant_recon.models.phytomer_vae import PhytomerVAE
from plant_recon.models.plant_organ_array import NUM_ORGAN_TYPES
from plant_recon.dataset.phytomer_topology import chain_phytomers
from real_world.dataset.dap_from_timestamp import dap_from_filename
from real_world.dataset.real_field_dataset import RealFieldPlantDataset
from real_world.eval.viz_utils import colorize_depth, colorize_mask


def load_pipeline(checkpoint: str, device: torch.device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    model = HierarchicalPartFlowMatchingModel(
        max_phytomers=args["max_phytomers"], slots_per_phytomer=args["slots_per_phytomer"], node_dim=args["node_dim"],
        num_classes=NUM_ORGAN_TYPES, image_size=128, patch_size=8, embed_dim=args["embed_dim"], vit_layers=args["vit_layers"],
        vit_heads=args["vit_heads"], coarse_layers=args["coarse_layers"], fine_layers=args["fine_layers"],
        flow_granularity=args["flow_granularity"], phytomer_latent_dim=args["phytomer_latent_dim"], backbone=args["backbone"],
        freeze_backbone=True, init_phytomer_count=args.get("init_phytomer_count", 50.0),
        stage3_geometry=bool(args.get("stage3_geometry", False)), multizoom=bool(args.get("multizoom", False)),
        node_token_window=int(args.get("node_token_window", 1))).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=False); model.eval()
    M = args["slots_per_phytomer"]
    pvae = PhytomerVAE(latent_dim=args["phytomer_latent_dim"], residual_dim=args.get("phytomer_residual_dim", 8), hidden_dim=256).to(device).eval()
    pvae.load_state_dict(torch.load(args["phytomer_vae_checkpoint"], map_location=device, weights_only=True))
    for p_ in pvae.parameters():
        p_.requires_grad_(False)
    ovae = OrganLatentVAE(latent_dim=args["node_dim"], hidden_dim=256).to(device).eval()
    sd = torch.load(args["organ_vae_checkpoint"], map_location=device, weights_only=False)
    ovae.load_state_dict(sd.get("model_state_dict", sd))
    renderer = HeliosPyTorchRenderer(image_size=256).to(device)
    return model, pvae, ovae, renderer, M


def sample_cold(model, pvae, ovae, images: torch.Tensor, device: torch.device, M: int,
                 daps: "torch.Tensor | None" = None):
    """images: (1, 16, 128, 128). Returns (parts, pred_dap, state) — parts is a part-tensor
    plant ready for renderer.geo_builder.build_mesh_from_part_tensor / XML export, pred_dap is
    Stage 1's self-predicted DAP (flag this alongside the render: the DAP head was trained
    only on Helios-simulated appearances and may be unreliable on a real photo), and state is
    (pos0, rot, scale0, lat0, exist, par, parent_idx) for Approach 2's refinement loop to
    continue from — parent_idx is the SAME chain_phytomers() call this function already made
    (from Stage 2's own ordinal/base, not a placeholder), so a caller doing --recompute_rot
    topology lookups gets the model's actual predicted topology, not a dummy one.

    daps: optional (1,) tensor overriding Stage 1's self-predicted DAP. Per
    hierarchical_part_flow_matching.py's sample_ode (~line 1798-1811): when given, this fully
    REPLACES the "clue" fed into Stage 2/3's conditioning (not just a capacity floor via
    max(k_pred, k_dap) — the capacity floor also applies, but active_k can only WIDEN, never
    shrink, below Stage 1's self-prediction). 2026-09-15 finding: on this real dataset Stage 1's
    self-predicted DAP is essentially uncorrelated with the true calendar DAP (computed from the
    filename timestamp + a known planting date, real_world/dataset/dap_from_timestamp.py) and is
    higher in 7/8 tested cases — so capacity never needs widening here, but the conditioning
    clue itself is very likely wrong regardless, which this argument corrects independently."""
    with torch.no_grad():
        tok = model.image_encoder(images)
        pred_dap = float(model.probe_pred_dap(tok).item())
        so = model.sample_ode(images=images, daps=daps, num_steps=20, vae=ovae, phytomer_vae=pvae)
        pos0 = so["phytomer_pos"][0].float(); exist = so["phytomer_existence"][0].float(); roll = so["phytomer_roll"][0].float()
        dap_clue = daps if daps is not None else model.probe_pred_dap(tok)
        co0 = model.coarse_stage(model.image_encoder(images), capacity_mode="pred_phyto", pred_dap=model.probe_pred_dap(tok))
        k_dap = int(compute_matryoshka_slice(dap=daps, max_phytomers=model.max_phytomers)) if daps is not None else 0
        K = max(int(co0["active_k"]), k_dap); co = model.coarse_stage(tok, active_k=K, pred_dap=dap_clue)
        ordn = co["phytomer_ordinal"][0].float(); base = co["phytomer_base_logits"][0].float()
        scale0 = (so.get("phytomer_scale") if so.get("phytomer_scale") is not None else co.get("phytomer_scale"))[0].float()
        lat0 = so["pred_latent"][0].float()
        rot, par = reconstruct_phytomer_rot(pos0.unsqueeze(0), roll.unsqueeze(0), ordn.unsqueeze(0), base.unsqueeze(0),
                                             exist=(exist > 0.5).float().unsqueeze(0))
        rot, par = rot[0].float(), par[0].float()
        parent_idx, _, _ = chain_phytomers(pos0, ordinal=ordn, is_base=(base > 0).float(), exist=(exist > 0.5).float())
        parts = plant_from_nodes(pvae, pos0, rot, scale0, lat0, exist, par, M)
    return parts, pred_dap, (pos0, rot, scale0, lat0, exist, par, parent_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "outputs/checkpoints/hierarchical_fm_v10_cam/hierarchical_fm_epoch_085.pt"))
    ap.add_argument("--detector_weights", default=str(REPO_ROOT / "outputs/logs/20260915/real_plant_detector/weights/best.pt"))
    ap.add_argument("--images", default=str(REPO_ROOT / "real_world/data/roboflow_t4_plant_weed_seg/1/test/images/*.jpg"))
    ap.add_argument("--limit", type=int, default=10, help="max detected plants to process")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--planted_date", default="", help="YYYY-MM-DD; if given, overrides Stage 1's self-predicted "
                     "DAP with one computed from each image's filename timestamp (real_world/dataset/dap_from_timestamp.py). "
                     "2026-09-15 finding: on this dataset Stage 1's self-predicted DAP is essentially uncorrelated with "
                     "the true calendar DAP and higher in 7/8 tested cases, so this rarely changes organ COUNT (capacity "
                     "can only widen, never shrink, below Stage 1's self-prediction) -- it corrects the DAP conditioning "
                     "clue itself, which is a real but separate effect from capacity; kept available since it's the "
                     "correct age signal even when it doesn't change the output much.")
    ap.add_argument("--out", default=str(REPO_ROOT / "use_cases/real_world/eval/output/approach1_cold"))
    a = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, pvae, ovae, renderer, M = load_pipeline(a.checkpoint, dev)
    planted_date = datetime.date.fromisoformat(a.planted_date) if a.planted_date else None

    paths = sorted(glob.glob(a.images))
    if not paths:
        raise SystemExit(f"No images matched {a.images}")
    ds = RealFieldPlantDataset(paths, a.detector_weights, conf=a.conf)
    os.makedirs(a.out, exist_ok=True)

    n = min(a.limit, len(ds))
    rows = []
    for i in range(n):
        it = ds[i]
        images = it["image"].unsqueeze(0).to(dev)
        ts_dap = dap_from_filename(it["jpeg"], planted_date) if planted_date else None
        daps_override = torch.tensor([float(ts_dap)], device=dev) if ts_dap is not None else None
        parts, pred_dap, _ = sample_cold(model, pvae, ovae, images, dev, M, daps=daps_override)
        ts_str = f" timestamp_dap={ts_dap}" if ts_dap is not None else ""
        print(f"[{i+1}/{n}] {it['prefix']}: pred_dap={pred_dap:.1f}{ts_str} num_organs={parts.shape[0]}")
        if parts.shape[0] == 0:
            rows.append({"prefix": it["prefix"], "jpeg": it["jpeg"], "pred_dap": pred_dap, "timestamp_dap": ts_dap, "num_organs": 0})
            continue
        mesh = renderer.geo_builder.build_mesh_from_part_tensor(parts, device=dev)
        with torch.no_grad():
            rgbd = renderer.forward(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.5, background="ground",
                                     focus_plant=True, include_depth=True, image_size=256, zoom_factor=1.0,
                                     reference_window_size=1.2)
        render_png = (rgbd[:3].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(render_png).save(os.path.join(a.out, f"{it['prefix']}_render.png"))
        Image.fromarray(colorize_depth(rgbd[3].cpu().numpy())).save(os.path.join(a.out, f"{it['prefix']}_render_depth.png"))
        # save the real crop's own zoom-1x RGB + pseudo-CHM depth channels for side-by-side comparison
        crop_rgb = ((it["image"][:3] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(crop_rgb).save(os.path.join(a.out, f"{it['prefix']}_input.png"))
        Image.fromarray(colorize_depth(it["image"][3].numpy())).save(os.path.join(a.out, f"{it['prefix']}_input_depth.png"))
        if it["mask_pyramid"] is not None:
            # zoom index 0 = 1x, matching the RGB/depth panels above — this is the detector's own
            # segmentation mask, the actual Dice-loss foreground target in run_approach2_refine.py
            Image.fromarray(colorize_mask(it["mask_pyramid"][0].numpy())).save(os.path.join(a.out, f"{it['prefix']}_input_mask.png"))
        rows.append({"prefix": it["prefix"], "jpeg": it["jpeg"], "pred_dap": pred_dap, "timestamp_dap": ts_dap, "num_organs": int(parts.shape[0])})

    json.dump({"checkpoint": a.checkpoint, "rows": rows}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(f"Done. {len(rows)} plants -> {a.out}")


if __name__ == "__main__":
    main()
