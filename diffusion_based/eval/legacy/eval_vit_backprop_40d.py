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
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.legacy.vit_image_to_organ_array_40d import ViTImageToOrganArray
from diffusion_based.dataset.legacy.organ_array_dataset_40d import OrganArrayDataset
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
    """image: (3, H, W) normalized tensor. Returns PlantOrganArray."""
    model.eval()
    outputs = model(image.unsqueeze(0))
    denorm = dataset.denormalize(outputs["pred_x0"][0])
    denorm[:, dataset.continuous_cols] = torch.clamp(denorm[:, dataset.continuous_cols], min=0.0)
    
    if "organ_type_logits" in outputs:
        denorm[:, T_COL_ORGAN_TYPE] = outputs["organ_type_logits"][0].argmax(dim=-1).float()
    else:
        denorm[:, T_COL_ORGAN_TYPE] = torch.clamp(torch.round(denorm[:, T_COL_ORGAN_TYPE]), 0, 7)

    exist_prob = torch.sigmoid(outputs["existence_logits"][0])
    denorm[:, T_COL_EXISTENCE] = exist_prob
    return PlantOrganArray(tensor=denorm.cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="diffusion_based/checkpoints/vit_backprop_vit_render.pt")
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--pattern", type=str, default="*seed09*",
                        help="Glob pattern for the evaluation (holdout) samples")
    parser.add_argument("--use-gt-renderer-image", action="store_true", default=True,
                        help="Use GT PyTorch render as input image")
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

    dataset = OrganArrayDataset(
        data_root=args.data_root,
        max_nodes=args.max_nodes,
        image_size=image_size,
        use_gt_renderer_image=args.use_gt_renderer_image,
        device=device,
        include_globs=[g.strip() for g in args.pattern.split(",")],
    )

    if args.output_dir is None:
        args.output_dir = os.path.join("diffusion_based", "eval", "output", "vit_backprop_eval")
    os.makedirs(args.output_dir, exist_ok=True)

    model = ViTImageToOrganArray(
        max_nodes=args.max_nodes, node_dim=40,
        image_size=image_size, patch_size=patch_size,
        embed_dim=embed_dim, encoder_layers=encoder_layers,
        decoder_layers=decoder_layers, num_heads=8, num_organ_types=8,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    renderer = HeliosPyTorchRenderer(image_size=image_size).to(device)

    if args.limit and args.limit < len(dataset.samples):
        indices = np.linspace(0, len(dataset.samples) - 1, args.limit, dtype=int)
        samples_to_eval = [dataset.samples[i] for i in indices]
    else:
        samples_to_eval = dataset.samples[:args.limit]
    print(f"Evaluating {len(samples_to_eval)} samples spanning DAPs from {len(dataset)} matched\n")

    all_metrics = {}

    for sample in samples_to_eval:
        prefix = os.path.basename(sample["xml"]).split("_plant_")[0]
        print(f"=== {prefix} ===")

        # Load ground truth organ array
        gt_organ_array = PlantOrganArray.from_xml_file_typed(sample["xml"])
        gt_tensor = gt_organ_array.tensor.to(device)
        n_gt = int((gt_organ_array.existence > 0.1).sum().item())

        # Render Ground Truth via differentiable PyTorch renderer
        with torch.no_grad():
            gt_rgb_t = renderer.render_organ_array(
                gt_organ_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
                background="black", device=device, differentiable=False, focus_plant=True,
                existence_threshold=0.1,
            )
        gt_np = gt_rgb_t.permute(1, 2, 0).cpu().numpy().clip(0, 1)

        # Input image: use GT PyTorch render directly for clean domain matching
        if args.use_gt_renderer_image:
            image_t = dataset._transform_tensor(gt_rgb_t).to(device)
            input_np = gt_np
        else:
            image_path = sample["jpeg"]
            raw_img = Image.open(image_path).convert("RGB")
            image_t = dataset.transform(raw_img).to(device)
            input_np = denormalize_image(image_t)

        # Model prediction
        pred_array = predict_organ_array(model, image_t, dataset)
        pred_tensor = pred_array.tensor.to(device)
        n_pred = int((pred_array.existence > 0.1).sum().item())

        # Type accuracy on active GT rows
        n_cmp = min(gt_tensor.shape[0], pred_tensor.shape[0])
        gt_types = gt_tensor[:n_cmp, T_COL_ORGAN_TYPE].long()
        pred_types = pred_tensor[:n_cmp, T_COL_ORGAN_TYPE].long()
        gt_active = (gt_tensor[:n_cmp, T_COL_EXISTENCE] > 0.5)
        if gt_active.sum() > 0:
            type_acc = float((gt_types[gt_active] == pred_types[gt_active]).float().mean().item())
        else:
            type_acc = 1.0

        # Existence IoU
        pred_exist_mask = (pred_tensor[:n_cmp, T_COL_EXISTENCE] > 0.1)
        gt_exist_mask = (gt_tensor[:n_cmp, T_COL_EXISTENCE] > 0.1)
        inter = float((pred_exist_mask & gt_exist_mask).sum().item())
        union = float((pred_exist_mask | gt_exist_mask).sum().item())
        exist_iou = inter / union if union > 0 else 1.0

        # Continuous MAE on active rows
        if n_cmp > 0:
            cont = dataset.continuous_cols
            mae = float(torch.abs(pred_tensor[:n_cmp, cont] - gt_tensor[:n_cmp, cont]).mean().item())
        else:
            mae = float("inf")

        # Render prediction and compare to target image
        try:
            pred_rgb_t = renderer.render_organ_array(
                pred_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
                background="black", device=device, differentiable=False, focus_plant=True,
                existence_threshold=0.1,
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

        # 4-panel figure with spacious layout so top titles are never clipped
        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        
        axes[0].imshow(input_np)
        axes[0].set_title(f"Input Image (GT PyTorch Render)\n{n_gt} nodes", fontsize=12, fontweight="bold", pad=12)
        
        axes[1].imshow(pred_np)
        axes[1].set_title(f"ViT Predicted Render\nMAE={img_mae:.4f} | SSIM={img_ssim:.4f}",
                          fontsize=12, fontweight="bold", pad=12)
        
        axes[2].imshow(gt_np)
        axes[2].set_title(f"GT Render\n{n_gt} active nodes", fontsize=12, fontweight="bold", pad=12)
        
        # organ-type mask of prediction
        try:
            with torch.no_grad():
                mask = renderer.render_organ_type_buffer(
                    renderer.geo_builder.build_mesh_from_organ_array(pred_array, device=device),
                    azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, focus_plant=True,
                ).cpu().numpy()
            axes[3].imshow(mask, cmap="tab10", vmin=0, vmax=7)
            axes[3].set_title(f"Pred Organ-Type Mask\n{n_pred} active organs", fontsize=12, fontweight="bold", pad=12)
        except Exception as e:
            axes[3].text(0.5, 0.5, f"Render err:\n{e}", ha="center", va="center", color="red", fontsize=8)
            axes[3].set_title(f"Pred Mask (Error)", fontsize=12, fontweight="bold", pad=12)
            
        for ax in axes:
            ax.axis("off")
            
        plt.subplots_adjust(top=0.82, bottom=0.06, left=0.03, right=0.97, wspace=0.15)
        out_png = os.path.join(args.output_dir, f"{prefix}_vit_eval.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
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