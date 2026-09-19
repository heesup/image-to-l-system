"""Dump the per-plant silhouette masks a checkpoint produces, for the shape metrics.

Why this exists rather than reusing eval_test_time_refinement.py --save_renders: that script decodes
through PhytomerVAE and so only works for `flow_granularity=phytomer`; on an organ-granularity
checkpoint it dies in PhytomerVAE.decode with a reshape error, because organ mode carries a 16D
OrganLatentVAE latent instead. The self-consistency evaluator already handles both modes -- it is what
rendered both A/B runs' figures -- so this simply drives it with `dump_masks_dir` set and writes the
masks the reported IoU is actually computed from.

The output layout (`<i>_gt.png`, `<i>_before.png`, `<i>_after.png`, `<i>_meta.json`) is what
eval_silhouette_shape_metrics.py consumes. There is no refinement here, so `after` == `before` and only
the "raw" columns of that report are meaningful.

    python plant_recon/eval/dump_eval_masks.py \
        --checkpoint outputs/checkpoints/ab_organ/hierarchical_fm_epoch_060.pt \
        --eval_set  outputs/checkpoints/ab_organ/holdout_set.json \
        --out       outputs/logs/20260918/rescore/organ_holdout
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from plant_recon.dataset.part_array_dataset import PartArrayDataset
from plant_recon.eval import eval_hierarchical_self_consistency as ehsc
from plant_recon.eval.ckpt_compat import fix_ckpt_args
from plant_recon.models.hierarchical_part_flow_matching import HierarchicalPartFlowMatchingModel
from plant_recon.models.organ_latent_vae import OrganLatentVAE
from plant_recon.models.phytomer_vae import PhytomerVAE
from plant_recon.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from plant_recon.training.train_hierarchical_flow_matching import collate_eval_set
from plant_recon.models.plant_organ_array import NUM_ORGAN_TYPES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval_set", required=True, help="an eval_set.json or holdout_set.json")
    ap.add_argument("--out", required=True, help="folder for the dumped masks")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = fix_ckpt_args(ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"]))

    model = HierarchicalPartFlowMatchingModel(
        max_phytomers=args["max_phytomers"], slots_per_phytomer=args["slots_per_phytomer"],
        node_dim=args["node_dim"], num_classes=NUM_ORGAN_TYPES, image_size=128, patch_size=8,
        embed_dim=args["embed_dim"], vit_layers=args["vit_layers"], vit_heads=args["vit_heads"],
        coarse_layers=args["coarse_layers"], fine_layers=args["fine_layers"],
        flow_granularity=args["flow_granularity"], phytomer_latent_dim=args["phytomer_latent_dim"],
        backbone=args["backbone"], freeze_backbone=True,
        init_phytomer_count=args.get("init_phytomer_count", 50.0),
        stage3_geometry=bool(args.get("stage3_geometry", False)),
        multizoom=bool(args.get("multizoom", False)),
        node_token_window=int(args.get("node_token_window", 1)),
        stage3_regression=bool(args.get("stage3_regression", False)),
        use_depth=bool(args.get("use_depth", False)),
    ).to(dev)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()

    pvae = PhytomerVAE(latent_dim=args["phytomer_latent_dim"],
                       residual_dim=args.get("phytomer_residual_dim", 8), hidden_dim=256).to(dev).eval()
    pvae.load_state_dict(torch.load(args["phytomer_vae_checkpoint"], map_location=dev, weights_only=True))

    # Organ granularity decodes each organ through the 16D OrganLatentVAE; without it the evaluator
    # produces empty geometry and reports 0.0% IoU (which is how this was first caught).
    ovae = None
    ovc = args.get("organ_vae_checkpoint")
    if ovc and os.path.exists(ovc):
        ovae = OrganLatentVAE(latent_dim=args["node_dim"], hidden_dim=256)
        sd = torch.load(ovc, map_location="cpu", weights_only=False)
        ovae.load_state_dict(sd.get("model_state_dict", sd))
        ovae = ovae.to(dev).eval()
    elif args["flow_granularity"] == "organ":
        raise SystemExit(f"organ-granularity checkpoint but organ_vae_checkpoint is missing: {ovc!r}")

    ds = PartArrayDataset(data_root=args["data_dir"], max_nodes=args["max_phytomers"] * args["slots_per_phytomer"],
                          cache_dir=args["cache_dir"], pkt_cache_dir=args.get("pkt_cache_dir") or None,
                          species="cowpea", image_size=128)

    # Resolve by prefix: a holdout set carries no indices, and an eval set's indices are relative to the
    # training subset rather than to the full dataset (see eval_test_time_refinement.py for the same note).
    by_prefix = {smp["prefix"]: i for i, smp in enumerate(ds.samples)}
    wanted = [smp["prefix"] for smp in json.load(open(a.eval_set))["samples"]]
    missing = [q for q in wanted if q not in by_prefix]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(wanted)} prefixes absent from {args['data_dir']}; first: {missing[0]}")
    idxs = [by_prefix[q] for q in wanted]

    batch = collate_eval_set(ds, idxs, dev)
    renderer = HeliosPyTorchRenderer(image_size=256).to(dev)
    os.makedirs(a.out, exist_ok=True)
    m = ehsc.evaluate_self_consistency_batch(
        model=model, val_batch=batch, renderer=renderer, device=dev, epoch=0,
        output_dir=a.out, num_samples_to_plot=0, vae=ovae, phytomer_vae=pvae,
        dump_masks_dir=a.out,
    )
    print(f"{os.path.basename(a.out)}: n={len(idxs)}  IoU {m['silhouette_iou']*100:.1f}%  "
          f"depth MAE {m['depth_mae']*100:.2f} cm  -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
