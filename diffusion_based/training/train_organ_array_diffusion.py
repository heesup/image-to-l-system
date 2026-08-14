"""
Training script for 40D typed PlantOrganArray Image-to-Graph Diffusion.

Combines:
  - Organ-type masked continuous MSE on the normalized 40D typed organ array
    tensor (only columns relevant to each row's organ_type contribute)
  - Existence BCE on channel 39
  - Categorical CE on organ_type (channel 11)
  - Optional periodic render reconstruction loss via HeliosPyTorchRenderer
  - Optional image-space augmentation (photometric jitter only)
  - Train/val split over the samples in data_root
"""

import os
import sys
import math
import argparse
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.organ_array_diffuser import PlantOrganArrayDiffuser
from diffusion_based.models.vit_image_to_organ_array import ViTOrganArrayDiffuser
from diffusion_based.dataset.organ_array_dataset import OrganArrayDataset
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.plant_organ_array import PlantOrganArray
class DDPMScheduler:
    """Simple DDPM noise schedule."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        acp = self.alphas_cumprod.to(x0.device)[t].view(-1, 1, 1)
        sqrt_acp = torch.sqrt(acp)
        sqrt_omc = torch.sqrt(1.0 - acp)
        return sqrt_acp * x0 + sqrt_omc * noise


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prediction_to_organ_array(pred_x0: torch.Tensor, dataset: OrganArrayDataset,
                              existence_logits: torch.Tensor = None) -> PlantOrganArray:
    """Denormalize model prediction and build PlantOrganArray. B must be 1."""
    assert pred_x0.shape[0] == 1, "rendering helper supports batch_size=1"
    denorm = dataset.denormalize(pred_x0[0])
    existence_col = dataset.existence_col
    # existence channel is the last column
    if existence_logits is not None:
        denorm[:, existence_col] = torch.sigmoid(existence_logits[0])
    else:
        denorm[:, existence_col] = torch.sigmoid(denorm[:, existence_col])
    # Clamp physical parameters to sensible non-negative ranges to avoid rendering failures
    denorm[:, dataset.continuous_cols] = torch.clamp(denorm[:, dataset.continuous_cols], min=0.0)
    # Round the categorical organ_type column (11) to the nearest valid class.
    denorm[:, 11] = torch.round(denorm[:, 11]).clamp(0, 7)
    return PlantOrganArray(tensor=denorm.cpu())


def render_loss(
    pred_x0: torch.Tensor,
    target_image: torch.Tensor,
    dataset: OrganArrayDataset,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render predicted organ arrays and compute MSE against target image."""
    B = pred_x0.shape[0]
    losses = []
    rendered_images = []

    for b in range(B):
        organ_array = prediction_to_organ_array(pred_x0[b:b + 1], dataset)
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
        rendered_images.append(rendered)

    loss = torch.stack(losses).mean()
    rendered_batch = torch.stack(rendered_images)
    return loss, rendered_batch


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: DDPMScheduler,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    lambda_continuous: float = 1.0,
    lambda_exist: float = 1.0,
    lambda_organ_type: float = 0.5,
    exist_pos_weight: float = 10.0,
    channel_weights: torch.Tensor = None,
    render_weight: float = 1.0,
    render_every: int = 25,
    global_step: int = 0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_exist = 0.0
    total_organ_type = 0.0
    total_render = 0.0
    count = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        nodes = batch["nodes"].to(device)  # (B, N, node_dim), normalized
        existence_gt = batch["existence_mask"].to(device)  # (B, N)
        row_relevance = batch["row_relevance"].to(device)  # (B, N, node_dim)
        dataset = dataloader.dataset
        existence_col = dataset.existence_col
        continuous_cols = dataset.continuous_cols

        B = images.shape[0]
        N = nodes.shape[1]

        t = torch.randint(0, scheduler.timesteps, (B,), device=device).long()

        # Add noise to normalized organ array
        noise = torch.randn_like(nodes)
        noisy_nodes = scheduler.add_noise(nodes, t, noise)

        outputs = model(noisy_nodes, t, images)
        pred_x0 = outputs["pred_x0"]
        organ_type_logits = outputs["organ_type_logits"]

        # Masked continuous channels MSE: only active nodes AND only columns
        # that are relevant to each row's organ_type contribute. Optional
        # per-channel weighting boosts structural vs. perturbation channels.
        active_mask = existence_gt.unsqueeze(-1).float()  # (B, N, 1)
        relevance = row_relevance[:, :, continuous_cols].float()  # (B, N, n_cont)
        weight_map = relevance * active_mask  # (B, N, n_cont)
        continuous_diff = pred_x0[:, :, continuous_cols] - nodes[:, :, continuous_cols]
        if channel_weights is not None:
            channel_weights = channel_weights.to(device).view(1, 1, -1)
            weighted_diff = (continuous_diff ** 2) * weight_map * channel_weights
        else:
            weighted_diff = (continuous_diff ** 2) * weight_map
        mse_loss = weighted_diff.sum() / max(weight_map.sum(), 1.0)

        # Existence BCE (last column) with positive weighting because positives are sparse
        pred_existence_logit = pred_x0[:, :, existence_col]
        pos_weight = torch.tensor(exist_pos_weight, device=device)
        existence_loss = F.binary_cross_entropy_with_logits(
            pred_existence_logit, existence_gt, pos_weight=pos_weight
        )

        # Categorical CE for organ_type (column 11) on active nodes
        organ_type_gt = nodes[:, :, 11].long().clamp(0, model.num_organ_types - 1)
        organ_type_loss = F.cross_entropy(
            organ_type_logits.reshape(-1, model.num_organ_types),
            organ_type_gt.reshape(-1),
            reduction="none",
        )
        organ_type_loss = (organ_type_loss.view_as(existence_gt) * existence_gt).sum() / max(
            existence_gt.sum(), 1.0
        )

        loss = (
            lambda_continuous * mse_loss
            + lambda_exist * existence_loss
            + lambda_organ_type * organ_type_loss
        )

        render_rec_loss = torch.tensor(0.0, device=device)
        if render_every > 0 and global_step % render_every == 0:
            render_rec_loss, _ = render_loss(pred_x0, images, dataloader.dataset, renderer, device)
            loss = loss + render_weight * render_rec_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        total_loss += loss.item() * B
        total_mse += mse_loss.item() * B
        total_exist += existence_loss.item() * B
        total_organ_type += organ_type_loss.item() * B
        if render_every > 0:
            total_render += render_rec_loss.item() * B
        count += B

    return {
        "loss": total_loss / max(count, 1),
        "mse": total_mse / max(count, 1),
        "exist": total_exist / max(count, 1),
        "organ_type": total_organ_type / max(count, 1),
        "render": total_render / max(count, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    scheduler: DDPMScheduler,
    device: torch.device,
    lambda_continuous: float = 1.0,
    lambda_exist: float = 1.0,
    lambda_organ_type: float = 0.5,
    exist_pos_weight: float = 10.0,
    channel_weights: torch.Tensor = None,
) -> Dict[str, float]:
    """Evaluate the same loss terms (minus render) on the held-out set."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_exist = 0.0
    total_organ_type = 0.0
    count = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        nodes = batch["nodes"].to(device)
        existence_gt = batch["existence_mask"].to(device)
        row_relevance = batch["row_relevance"].to(device)
        dataset = val_loader.dataset
        existence_col = dataset.existence_col
        continuous_cols = dataset.continuous_cols

        B = images.shape[0]
        t = torch.randint(0, scheduler.timesteps, (B,), device=device).long()
        noise = torch.randn_like(nodes)
        noisy_nodes = scheduler.add_noise(nodes, t, noise)
        outputs = model(noisy_nodes, t, images)
        pred_x0 = outputs["pred_x0"]
        organ_type_logits = outputs["organ_type_logits"]

        active_mask = existence_gt.unsqueeze(-1).float()
        relevance = row_relevance[:, :, continuous_cols].float()
        weight_map = relevance * active_mask
        continuous_diff = pred_x0[:, :, continuous_cols] - nodes[:, :, continuous_cols]
        if channel_weights is not None:
            channel_weights = channel_weights.to(device).view(1, 1, -1)
            weighted_diff = (continuous_diff ** 2) * weight_map * channel_weights
        else:
            weighted_diff = (continuous_diff ** 2) * weight_map
        mse_loss = weighted_diff.sum() / max(weight_map.sum(), 1.0)

        pos_weight = torch.tensor(exist_pos_weight, device=device)
        existence_loss = F.binary_cross_entropy_with_logits(
            pred_x0[:, :, existence_col], existence_gt, pos_weight=pos_weight
        )

        organ_type_gt = nodes[:, :, 11].long().clamp(0, model.num_organ_types - 1)
        organ_type_loss = F.cross_entropy(
            organ_type_logits.reshape(-1, model.num_organ_types),
            organ_type_gt.reshape(-1),
            reduction="none",
        )
        organ_type_loss = (organ_type_loss.view_as(existence_gt) * existence_gt).sum() / max(
            existence_gt.sum(), 1.0
        )

        loss = (
            lambda_continuous * mse_loss
            + lambda_exist * existence_loss
            + lambda_organ_type * organ_type_loss
        )

        total_loss += loss.item() * B
        total_mse += mse_loss.item() * B
        total_exist += existence_loss.item() * B
        total_organ_type += organ_type_loss.item() * B
        count += B

    model.train()
    return {
        "loss": total_loss / max(count, 1),
        "mse": total_mse / max(count, 1),
        "exist": total_exist / max(count, 1),
        "organ_type": total_organ_type / max(count, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--single_xml", type=str, default=None,
                        help="Train on one XML only for fast sanity check")
    parser.add_argument("--max_nodes", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true",
                        help="Enable image-space photometric augmentation")
    parser.add_argument("--percentile", type=float, default=1.0,
                        help="Percentile clip for normalization stats (0 disables)")
    parser.add_argument("--lambda_continuous", type=float, default=1.0)
    parser.add_argument("--lambda_exist", type=float, default=1.0)
    parser.add_argument("--lambda_organ_type", type=float, default=0.5)
    parser.add_argument("--exist_pos_weight", type=float, default=10.0)
    parser.add_argument("--channel_weights", type=str, default=None,
                        help="Comma-separated per-channel weights for continuous MSE")
    parser.add_argument("--render_every", type=int, default=25,
                        help="Run render loss every N steps (0 disables)")
    parser.add_argument("--render_weight", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="diffusion_based/checkpoints")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "vit"],
                        help="backbone: resnet (PlantOrganArrayDiffuser) or vit (ViTOrganArrayDiffuser)")
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--val_pattern", type=str, default=None,
                        help="Comma-separated basename globs held out for validation, e.g. '*seed02*'")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    val_globs = [g.strip() for g in args.val_pattern.split(",")] if args.val_pattern else []
    if args.single_xml is not None:
        dataset = OrganArrayDataset(
            data_root=args.data_root,
            max_nodes=args.max_nodes,
            image_size=args.image_size,
            single_xml_path=args.single_xml,
            augment=args.augment,
            percentile=args.percentile,
        )
        val_dataset = None
        print(f"Dataset size: {len(dataset)}")
    else:
        dataset = OrganArrayDataset(
            data_root=args.data_root,
            max_nodes=args.max_nodes,
            image_size=args.image_size,
            augment=args.augment,
            percentile=args.percentile,
            exclude_globs=val_globs,
        )
        val_dataset = None
        if val_globs:
            val_dataset = OrganArrayDataset(
                data_root=args.data_root,
                max_nodes=args.max_nodes,
                image_size=args.image_size,
                augment=False,
                percentile=args.percentile,
                include_globs=val_globs,
            )
        print(f"Train dataset size: {len(dataset)}")
        if val_dataset is not None:
            print(f"Val dataset size: {len(val_dataset)}")
    print(f"Channel min range: [{dataset.min_vals.min():.3f}, {dataset.min_vals.max():.3f}]")
    print(f"Channel range range: [{dataset.max_vals.min():.3f}, {dataset.max_vals.max():.3f}]")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    if args.model == "vit":
        model = ViTOrganArrayDiffuser(
            max_nodes=args.max_nodes,
            node_dim=40,
            image_size=args.image_size,
            patch_size=args.patch_size,
            embed_dim=256,
            encoder_layers=args.encoder_layers,
            decoder_layers=4,
            num_heads=8,
            num_organ_types=8,
        ).to(device)
    else:
        model = PlantOrganArrayDiffuser(
            max_nodes=args.max_nodes,
            node_dim=40,
            embed_dim=256,
            num_layers=4,
            num_organ_types=8,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = DDPMScheduler(timesteps=args.timesteps)
    renderer = HeliosPyTorchRenderer(image_size=args.image_size).to(device)

    channel_weights = None
    if args.channel_weights:
        vals = [float(v) for v in args.channel_weights.split(",")]
        assert len(vals) == len(dataset.continuous_cols), \
            f"channel_weights length {len(vals)} != continuous cols {len(dataset.continuous_cols)}"
        channel_weights = torch.tensor(vals, dtype=torch.float32)
        print("Per-channel continuous weights:", channel_weights.tolist())

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model, dataloader, optimizer, scheduler, renderer, device,
            lambda_continuous=args.lambda_continuous,
            lambda_exist=args.lambda_exist,
            lambda_organ_type=args.lambda_organ_type,
            exist_pos_weight=args.exist_pos_weight,
            channel_weights=channel_weights,
            render_weight=args.render_weight,
            render_every=args.render_every,
            global_step=global_step,
        )
        global_step += len(dataset)
        print(
            f"Epoch {epoch:03d} | loss={metrics['loss']:.4f} "
            f"mse={metrics['mse']:.4f} exist={metrics['exist']:.4f} "
            f"organ_type={metrics['organ_type']:.4f} render={metrics['render']:.4f}"
        )

        if val_loader is not None:
            val_metrics = validate(
                model, val_loader, scheduler, device,
                lambda_continuous=args.lambda_continuous,
                lambda_exist=args.lambda_exist,
                lambda_organ_type=args.lambda_organ_type,
                exist_pos_weight=args.exist_pos_weight,
                channel_weights=channel_weights,
            )
            print(
                f"           VAL  | loss={val_metrics['loss']:.4f} "
                f"mse={val_metrics['mse']:.4f} exist={val_metrics['exist']:.4f} "
                f"organ_type={val_metrics['organ_type']:.4f}"
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"organ_array_diffuser_norm_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "organ_array_diffuser_norm.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
