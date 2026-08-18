"""
Training for the ViT Image -> PlantOrganArray inverse rendering model.

Loss terms:
  1. organ-array supervised loss (masked MSE + existence BCE + organ-type CE)
  2. image-space loss: render the predicted organ array through the
     differentiable HeliosPyTorchRenderer and compare to the target image.

Two training modes:
  --train-rendering  : include the differentiable render image loss (slower).
  default (no flag)  : organ-array supervised loss only (fast warm-up).
"""

import os
import sys
import argparse
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.legacy.vit_image_to_organ_array_40d import ViTImageToOrganArray
from diffusion_based.dataset.legacy.organ_array_dataset_40d import OrganArrayDataset
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.plant_organ_array import PlantOrganArray


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prediction_to_organ_array(pred_x0: torch.Tensor, existence_logits: torch.Tensor,
                              dataset: OrganArrayDataset) -> PlantOrganArray:
    """Denormalize a (1, N, 40) prediction into a PlantOrganArray for rendering."""
    assert pred_x0.shape[0] == 1, "rendering helper supports batch_size=1"
    denorm = dataset.denormalize(pred_x0[0])
    existence_col = dataset.existence_col
    exist_prob = torch.sigmoid(existence_logits[0]) if existence_logits is not None else torch.sigmoid(
        denorm[:, existence_col])
    denorm[:, existence_col] = exist_prob
    denorm[:, dataset.continuous_cols] = torch.clamp(denorm[:, dataset.continuous_cols], min=0.0)
    denorm[:, 11] = torch.round(denorm[:, 11]).clamp(0, 7)
    return PlantOrganArray(tensor=denorm.cpu())


def render_loss(
    pred_x0: torch.Tensor,
    existence_logits: torch.Tensor,
    target_image: torch.Tensor,
    dataset: OrganArrayDataset,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
) -> torch.Tensor:
    """Render predicted organ arrays (differentiable) and MSE against target image."""
    B = pred_x0.shape[0]
    losses = []
    for b in range(B):
        organ_array = prediction_to_organ_array(pred_x0[b:b + 1], existence_logits[b:b + 1], dataset)
        try:
            rendered = renderer.render_organ_array(
                organ_array,
                azimuth_deg=0.0,
                elevation_deg=90.0,
                camera_height=1.0,
                background="ground",
                device=device,
                differentiable=True,
                focus_plant=True,
                existence_threshold=0.5,
            )
        except Exception:
            rendered = torch.zeros_like(target_image[b])
        losses.append(F.mse_loss(rendered, target_image[b]))
    return torch.stack(losses).mean()


def supervised_loss(
    pred_x0: torch.Tensor,
    organ_type_logits: torch.Tensor,
    existence_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    dataset: OrganArrayDataset,
    device: torch.device,
    lambda_continuous: float = 1.0,
    lambda_exist: float = 1.0,
    lambda_organ_type: float = 0.5,
    exist_pos_weight: float = 10.0,
    channel_weights: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    nodes = batch["nodes"].to(device)
    existence_gt = batch["existence_mask"].to(device)
    row_relevance = batch["row_relevance"].to(device)
    existence_col = dataset.existence_col
    continuous_cols = dataset.continuous_cols

    active_mask = existence_gt.unsqueeze(-1).float()
    relevance = row_relevance[:, :, continuous_cols].float()
    weight_map = relevance * active_mask
    diff = pred_x0[:, :, continuous_cols] - nodes[:, :, continuous_cols]
    if channel_weights is not None:
        cw = channel_weights.to(device).view(1, 1, -1)
        mse_loss = ((diff ** 2) * weight_map * cw).sum() / max(weight_map.sum(), 1.0)
    else:
        mse_loss = ((diff ** 2) * weight_map).sum() / max(weight_map.sum(), 1.0)

    pos_weight = torch.tensor(exist_pos_weight, device=device)
    existence_loss = F.binary_cross_entropy_with_logits(existence_logits, existence_gt, pos_weight=pos_weight)

    organ_type_gt = nodes[:, :, 11].long().clamp(0, 7)
    ce = F.cross_entropy(
        organ_type_logits.reshape(-1, organ_type_logits.shape[-1]),
        organ_type_gt.reshape(-1),
        reduction="none",
    )
    organ_type_loss = (ce.view_as(existence_gt) * existence_gt).sum() / max(existence_gt.sum(), 1.0)

    total = (
        lambda_continuous * mse_loss
        + lambda_exist * existence_loss
        + lambda_organ_type * organ_type_loss
    )
    return {
        "loss": total,
        "mse": mse_loss,
        "exist": existence_loss,
        "organ_type": organ_type_loss,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    use_rendering: bool = True,
    render_every: int = 1,
    render_weight: float = 1.0,
    lambda_continuous: float = 1.0,
    lambda_exist: float = 1.0,
    lambda_organ_type: float = 0.5,
    exist_pos_weight: float = 10.0,
    channel_weights: torch.Tensor = None,
    global_step: int = 0,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "exist": 0.0, "organ_type": 0.0, "render": 0.0}
    count = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        B = images.shape[0]
        outputs = model(images)
        sup = supervised_loss(
            outputs["pred_x0"], outputs["organ_type_logits"], outputs["existence_logits"],
            batch, dataloader.dataset, device,
            lambda_continuous=lambda_continuous, lambda_exist=lambda_exist,
            lambda_organ_type=lambda_organ_type, exist_pos_weight=exist_pos_weight,
            channel_weights=channel_weights,
        )
        loss = sup["loss"]

        render_term = torch.tensor(0.0, device=device)
        if use_rendering and (render_every <= 0 or global_step % render_every == 0):
            render_term = render_loss(
                outputs["pred_x0"], outputs["existence_logits"], images,
                dataloader.dataset, renderer, device,
            )
            loss = loss + render_weight * render_term

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        totals["loss"] += loss.item() * B
        totals["mse"] += sup["mse"].item() * B
        totals["exist"] += sup["exist"].item() * B
        totals["organ_type"] += sup["organ_type"].item() * B
        totals["render"] += render_term.item() * B
        count += B

    return {k: v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, val_loader, device, lambda_continuous=1.0, lambda_exist=1.0,
             lambda_organ_type=0.5, exist_pos_weight=10.0, channel_weights=None) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "exist": 0.0, "organ_type": 0.0}
    count = 0
    for batch in val_loader:
        images = batch["image"].to(device)
        B = images.shape[0]
        outputs = model(images)
        sup = supervised_loss(
            outputs["pred_x0"], outputs["organ_type_logits"], outputs["existence_logits"],
            batch, val_loader.dataset, device,
            lambda_continuous=lambda_continuous, lambda_exist=lambda_exist,
            lambda_organ_type=lambda_organ_type, exist_pos_weight=exist_pos_weight,
            channel_weights=channel_weights,
        )
        for k in totals:
            totals[k] += sup[k].item() * B
        count += B
    model.train()
    return {k: v / max(count, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--max_nodes", type=int, default=2048)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-rendering", action="store_true",
                        help="Include differentiable-render image loss in training")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--render-weight", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="diffusion_based/checkpoints")
    parser.add_argument("--use-gt-renderer-image", action="store_true", default=True,
                        help="Render GT directly via PyTorch renderer for training input")
    parser.add_argument("--val_pattern", type=str, default=None,
                        help="Comma-separated basename globs held out for validation, e.g. '*seed09*'")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from a checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    val_globs = [g.strip() for g in args.val_pattern.split(",")] if args.val_pattern else []
    dataset = OrganArrayDataset(
        data_root=args.data_root,
        max_nodes=args.max_nodes,
        image_size=args.image_size,
        use_gt_renderer_image=args.use_gt_renderer_image,
        device=device,
        exclude_globs=val_globs if val_globs else None,
    )
    print(f"Train samples: {len(dataset)}")
    val_dataset = None
    if val_globs:
        val_dataset = OrganArrayDataset(
            data_root=args.data_root, max_nodes=args.max_nodes, image_size=args.image_size,
            use_gt_renderer_image=args.use_gt_renderer_image,
            device=device,
            include_globs=val_globs,
        )
        print(f"Val samples: {len(val_dataset)}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_dataset else None

    model = ViTImageToOrganArray(
        max_nodes=args.max_nodes, node_dim=40,
        image_size=args.image_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers, num_heads=8, num_organ_types=8,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.checkpoint} (epoch {start_epoch})")

    renderer = HeliosPyTorchRenderer(image_size=args.image_size).to(device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    channel_weights = None
    n_cont = len(dataset.continuous_cols)
    # Slight upweighting of structure-critical columns (length, radius, scale)
    w = torch.ones(n_cont, dtype=torch.float32)
    channel_weights = w

    global_step = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        metrics = train_epoch(
            model, dataloader, optimizer, renderer, device,
            use_rendering=args.train_rendering,
            render_every=args.render_every,
            render_weight=args.render_weight,
            channel_weights=channel_weights,
            global_step=global_step,
        )
        global_step += len(dataset)
        print(f"Epoch {epoch:03d} | loss={metrics['loss']:.4f} mse={metrics['mse']:.4f} "
              f"exist={metrics['exist']:.4f} ot={metrics['organ_type']:.4f} "
              f"render={metrics['render']:.4f}")

        if val_loader is not None:
            v = validate(model, val_loader, device, channel_weights=channel_weights)
            print(f"          VAL  | loss={v['loss']:.4f} mse={v['mse']:.4f} "
                  f"exist={v['exist']:.4f} ot={v['organ_type']:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            tag = "vit_render" if args.train_rendering else "vit"
            ckpt_path = os.path.join(args.checkpoint_dir, f"vit_backprop_{tag}_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved {ckpt_path}")

    tag = "vit_render" if args.train_rendering else "vit"
    final_path = os.path.join(args.checkpoint_dir, f"vit_backprop_{tag}.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()