"""
Training script for 40D typed PlantOrganArray Image-to-Graph Diffusion.

Combines:
  - DDPM-style denoising MSE on the normalized 40D typed organ array tensor
  - Existence BCE on channel 39
  - Optional render reconstruction loss via HeliosPyTorchRenderer
"""

import os
import sys
import math
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.organ_array_diffuser import PlantOrganArrayDiffuser
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


def prediction_to_organ_array(pred_x0: torch.Tensor, dataset: OrganArrayDataset) -> PlantOrganArray:
    """Denormalize model prediction and build PlantOrganArray. B must be 1."""
    assert pred_x0.shape[0] == 1, "rendering helper supports batch_size=1"
    denorm = dataset.denormalize(pred_x0[0])
    existence_col = dataset.existence_col
    # existence channel is the last column
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
    render_weight: float,
    use_render_loss: bool,
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
        dataset = dataloader.dataset
        node_dim = dataset.node_dim
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

        # Masked continuous channels MSE: only active nodes contribute.
        # Excludes the categorical organ_type column (11) and existence from MSE.
        active_mask = existence_gt.unsqueeze(-1)  # (B, N, 1)
        continuous_diff = pred_x0[:, :, continuous_cols] - nodes[:, :, continuous_cols]
        mse_loss = (continuous_diff ** 2 * active_mask).sum() / max(active_mask.sum(), 1.0)

        # Existence BCE (last column) with positive weighting because positives are sparse
        pred_existence_logit = pred_x0[:, :, existence_col]
        pos_weight = torch.tensor(10.0, device=device)
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

        loss = mse_loss + existence_loss + organ_type_loss

        render_rec_loss = torch.tensor(0.0, device=device)
        if use_render_loss:
            render_rec_loss, _ = render_loss(pred_x0, images, dataloader.dataset, renderer, device)
            loss = loss + render_weight * render_rec_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        total_mse += mse_loss.item() * B
        total_exist += existence_loss.item() * B
        total_organ_type += organ_type_loss.item() * B
        if use_render_loss:
            total_render += render_rec_loss.item() * B
        count += B

    return {
        "loss": total_loss / max(count, 1),
        "mse": total_mse / max(count, 1),
        "exist": total_exist / max(count, 1),
        "organ_type": total_organ_type / max(count, 1),
        "render": total_render / max(count, 1),
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
    parser.add_argument("--use_render_loss", action="store_true")
    parser.add_argument("--render_weight", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=str, default="diffusion_based/checkpoints")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    dataset = OrganArrayDataset(
        data_root=args.data_root,
        max_nodes=args.max_nodes,
        image_size=args.image_size,
        single_xml_path=args.single_xml,
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Channel min range: [{dataset.min_vals.min():.3f}, {dataset.min_vals.max():.3f}]")
    print(f"Channel range range: [{dataset.max_vals.min():.3f}, {dataset.max_vals.max():.3f}]")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

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

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            model, dataloader, optimizer, scheduler, renderer, device,
            render_weight=args.render_weight,
            use_render_loss=args.use_render_loss,
        )
        print(
            f"Epoch {epoch:03d} | loss={metrics['loss']:.4f} "
            f"mse={metrics['mse']:.4f} exist={metrics['exist']:.4f} "
            f"organ_type={metrics['organ_type']:.4f} render={metrics['render']:.4f}"
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
