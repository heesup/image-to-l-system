"""
Evaluation for the ViT Image -> PlantOrganArray inverse rendering model.

For each input image:
  - predict the (N, 40) organ array with the ViT model
  - render the predicted array through HeliosPyTorchRenderer
  - compare against the target image (MAE / SSIM / silhouette IoU)
  - compare the predicted organ array against GT (node count, type accuracy,
    existence IoU, continuous-channel MAE on active nodes)

Outputs a 4-panel figure + metrics JSON per sample.
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.vit_image_to_organ_array import ViTImageToOrganArray
from diffusion_based.dataset.organ_array_dataset import OrganArrayDataset
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.plant_organ_array import PlantOrganArray, T_COL_ORGAN_TYPE, T_COL_EXISTENCE


def compute_ssim_numpy(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=win_size))
    except Exception:
        mse = float(np.mean((img1 - img2) ** 2))
        return float(max(0.0, 1.0 - 5.0 * mse))


def compute_silhouette_iou(img1, img2, thresh=0.05):
    m1 = (img1.mean(axis=-1) > thresh)
    m2 = (img2.mean(axis=-1) > thresh)
    inter = float(np.logical_and(m1, m2).sum())
    union = float(np.logical_or(m1, m2).sum())
    return inter / union if union > 0 else (1.0 if inter > 0 else 0.0)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def predict_organ_array(model, image: torch.Tensor, dataset: OrganArrayDataset):
    """image: (3, H, W) normalized tensor. Returns PlantOrganArray.

    Post-processing mirrors the training-time prediction_to_organ_array so
    predictions stay renderable (round the organ_type column, sigmoid the
    existence channel)."""
    model.eval()
    outputs = model(image.unsqueeze(0))
    denorm = dataset.denormalize(outputs["pred_x0"][0])
    denorm[:, dataset.continuous_cols] = torch.clamp(denorm[:, dataset.continuous_cols], min=0.0)
    denorm[:, T_COL_EXISTENCE] = torch.sigmoid(outputs["existence_logits"][0])
    denorm[:, T_COL_ORGAN_TYPE] = torch.clamp(torch.round(denorm[:, T_COL_ORGAN_TYPE]), 0, 7)
    denorm[:, T_COL_EXISTENCE] = (denorm[:, T_COL_EXISTENCE] > 0.5).float()
    return PlantOrganArray(tensor=denorm.cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="diffusion_based/checkpoints/vit_backprop_vit_render.pt")
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--pattern", type=str, default="*seed09*",
                        help="Glob pattern for the evaluation (holdout) samples")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_nodes", type=int, default=256)
    parser.add_argument("--limit", type=int, default=5, help="max number of samples to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    image_size = ckpt_args.get("image_size", args.image_size)
    patch_size = ckpt_args.get("patch_size", 8)
    embed_dim = ckpt_args.get("embed_dim", 256)
    encoder_layers = ckpt_args.get("encoder_layers", 6)
    decoder_layers = ckpt_args.get("decoder_layers", 4)
    max_nodes = ckpt_args.get("max_nodes", args.max_nodes)
    print(f"Using image_size={image_size}, patch={patch_size}, embed={embed_dim}")

    model = ViTImageToOrganArray(
        max_nodes=max_nodes, node_dim=40, image_size=image_size, patch_size=patch_size,
        embed_dim=embed_dim, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        num_heads=8, num_organ_types=8,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch')})")

    dataset = OrganArrayDataset(
        data_root=args.data_root, max_nodes=max_nodes, image_size=image_size,
        include_globs=[args.pattern] if args.pattern else None,
    )
    if len(dataset) == 0:
        print(f"No samples matched pattern={args.pattern} in {args.data_root}")
        return
    print(f"Evaluating {min(len(dataset), args.limit)} samples from {len(dataset)} matched")

    renderer = HeliosPyTorchRenderer(image_size=image_size).to(device)
    if args.output_dir is None:
        args.output_dir = os.path.join("diffusion_based", "eval", "output", "vit_backprop_eval")
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}
    idxs = list(range(len(dataset)))[:args.limit]
    for i in idxs:
        item = dataset[i]
        image_t = item["image"].to(device)
        prefix = os.path.basename(item["xml_path"]).replace("_plant_0000.xml", "")
        print(f"\n=== {prefix} ===")

        pred_array = predict_organ_array(model, image_t, dataset)

        gt_organ_array = PlantOrganArray.from_xml_file_typed(item["xml_path"])
        gt_tensor = gt_organ_array.tensor
        gt_tensor = torch.nan_to_num(gt_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        gt_exist = (gt_tensor[:, T_COL_EXISTENCE] > 0.5)
        pred_tensor = pred_array.tensor
        pred_exist = (pred_tensor[:, T_COL_EXISTENCE] > 0.5)
        n_gt = int(gt_exist.sum().item())
        n_pred = int(pred_exist.sum().item())

        # Align lengths: GT may be shorter than max_nodes-padded predictions.
        align = min(gt_tensor.shape[0], pred_tensor.shape[0])
        gt_tensor = gt_tensor[:align]
        pred_tensor = pred_tensor[:align]
        gt_exist = gt_exist[:align]
        pred_exist = pred_exist[:align]

        # Organ type accuracy over active rows (min of GT/pred rows compared by index)
        n_cmp = min(n_gt, n_pred)
        if n_cmp > 0:
            gt_ot = gt_tensor[:n_cmp, T_COL_ORGAN_TYPE].long()
            pred_ot = pred_tensor[:n_cmp, T_COL_ORGAN_TYPE].long()
            type_acc = float((gt_ot == pred_ot).float().mean().item())
        else:
            type_acc = 0.0

        exist_iou = float((gt_exist & pred_exist).sum().item() /
                          max(int((gt_exist | pred_exist).sum().item()), 1))

        # Continuous MAE on active rows
        if n_cmp > 0:
            cont = dataset.continuous_cols
            mae = float(torch.abs(pred_tensor[:n_cmp, cont] - gt_tensor[:n_cmp, cont]).mean().item())
        else:
            mae = float("inf")

        input_np = denormalize_image(image_t)

        # Render prediction and compare to target image
        try:
            pred_rgb_t = renderer.render_organ_array(
                pred_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
                background="ground", device=device, differentiable=False, focus_plant=True,
                existence_threshold=0.5,
            )
            pred_np = pred_rgb_t.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            img_mae = float(np.mean(np.abs(pred_np - input_np)))
            img_ssim = compute_ssim_numpy(pred_np, input_np)
            img_iou = compute_silhouette_iou(pred_np, input_np)
            render_ok = True
        except Exception as e:
            print(f"  [warn] prediction render failed: {e}")
            pred_np = np.zeros_like(input_np)
            img_mae = float("nan")
            img_ssim = 0.0
            img_iou = 0.0
            render_ok = False

        metrics = {
            "n_gt": n_gt, "n_pred": n_pred, "type_acc": type_acc,
            "exist_iou": exist_iou, "cont_mae": mae,
            "img_mae": img_mae, "img_ssim": img_ssim, "img_iou": img_iou,
            "render_ok": render_ok,
        }
        all_metrics[prefix] = metrics
        print(f"  nodes GT={n_gt} pred={n_pred} | type_acc={type_acc:.3f} exist_iou={exist_iou:.3f} "
              f"cont_mae={mae:.4f}\n  image MAE={img_mae:.4f} SSIM={img_ssim:.4f} IoU={img_iou:.4f}")

        # 4-panel figure: input | pred render | GT render | organ-type mask overlay
        gt_rgb_t = renderer.render_organ_array(
            gt_organ_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
            background="ground", device=device, differentiable=False, focus_plant=True,
            existence_threshold=0.5,
        )
        gt_np = gt_rgb_t.permute(1, 2, 0).cpu().numpy().clip(0, 1)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(input_np)
        axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[1].imshow(pred_np)
        axes[1].set_title(f"ViT Predicted Render\nMAE={img_mae:.4f} SSIM={img_ssim:.4f}",
                          fontsize=11, fontweight="bold")
        axes[2].imshow(gt_np)
        axes[2].set_title(f"GT Render\n{n_gt} nodes", fontsize=11, fontweight="bold")
        # organ-type mask of prediction
        with torch.no_grad():
            mask = renderer.render_organ_type_buffer(
                renderer.geo_builder.build_mesh_from_organ_array(pred_array, device=device),
                azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, focus_plant=True,
            ).cpu().numpy()
        axes[3].imshow(mask, cmap="tab10", vmin=0, vmax=7)
        axes[3].set_title(f"Pred Organ-Type Mask\n{n_pred} active", fontsize=11, fontweight="bold")
        for ax in axes:
            ax.axis("off")
        out_png = os.path.join(args.output_dir, f"{prefix}_vit_eval.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved {out_png}")

    summary = {
        "checkpoint": args.checkpoint,
        "mean_type_acc": float(np.mean([m["type_acc"] for m in all_metrics.values()])),
        "mean_exist_iou": float(np.mean([m["exist_iou"] for m in all_metrics.values()])),
        "mean_img_mae": float(np.mean([m["img_mae"] for m in all_metrics.values()])),
        "mean_img_ssim": float(np.mean([m["img_ssim"] for m in all_metrics.values()])),
        "mean_img_iou": float(np.mean([m["img_iou"] for m in all_metrics.values()])),
        "per_sample": all_metrics,
    }
    out_json = os.path.join(args.output_dir, "vit_backprop_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()