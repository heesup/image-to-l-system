"""Training script for 3D Plant Graph Diffusion with Differentiable Rendering (15D).

Trains PlantGraphDiffuser3D with:
- Helios dataset (image + XML pairs)
- 15D organ-typed node representation
- Multi-view camera conditioning
- Differentiable renderer for render-in-the-loop
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diffusion_based.models.graph_diffuser_3d import PlantGraphDiffuser3D
from diffusion_based.models.differentiable_renderer_3d import DifferentiablePlantRenderer3D
from diffusion_based.dataset.helios_dataset import HeliosPlantDataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class DDPMScheduler:
    """Linear DDPM Noise Scheduler."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x0.device
        noise = torch.randn_like(x0)

        alphas_cumprod = self.alphas_cumprod.to(device)
        sqrt_alpha = torch.sqrt(alphas_cumprod[timesteps])[:, None, None]
        sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod[timesteps])[:, None, None]

        xt = sqrt_alpha * x0 + sqrt_one_minus * noise
        return xt, noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device).long()


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    gt_nodes: torch.Tensor,
    gt_existence: torch.Tensor,
    gt_parents: torch.Tensor,
    gt_adj: torch.Tensor,
    noisy_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-objective training losses for 15D organ-typed graph diffusion."""

    pred_x0 = outputs["pred_x0"]
    pred_existence_logits = outputs["pred_existence_logits"]
    pred_parent_logits = outputs["pred_parent_logits"]
    pred_parent_candidates = outputs["pred_parent_candidates"]
    pred_organ_type_logits = outputs.get("pred_organ_type_logits", None)
    pred_node_noise = outputs["pred_node_noise"]

    # 1. Node coordinate MSE (3D position accuracy)
    loss_coord = F.mse_loss(pred_x0[:, :, :3], gt_nodes[:, :, :3])

    # 2. Full 15D node attribute MSE
    loss_x0 = F.mse_loss(pred_x0, gt_nodes)

    # 3. Existence confidence BCE
    existence_target = (gt_existence > 0).float()
    pos_weight = torch.tensor([5.0], device=gt_nodes.device)
    loss_existence = F.binary_cross_entropy_with_logits(
        pred_existence_logits, existence_target, pos_weight=pos_weight
    )

    # 4. Parent cross-entropy over sparse k-NN candidates
    B, N = gt_parents.shape
    k = pred_parent_candidates.shape[-1]

    # Build mask: which candidates match the true parent
    parent_candidates = pred_parent_candidates  # (B, N, k)
    gt_parents_exp = gt_parents.unsqueeze(-1).expand(-1, -1, k)
    candidate_match = (parent_candidates == gt_parents_exp)  # (B, N, k)

    # For cross-entropy over k candidates, target is the index within candidates
    target_idx = torch.argmax(candidate_match.float(), dim=-1)  # (B, N)
    valid = candidate_match.any(dim=-1)  # (B, N)

    if valid.sum() > 0:
        loss_parent_all = F.cross_entropy(
            pred_parent_logits.view(-1, k),
            target_idx.view(-1),
            reduction='none'
        )
        loss_parent = (loss_parent_all * valid.view(-1).float()).sum() / (valid.sum().float() + 1e-8)
    else:
        loss_parent = torch.tensor(0.0, device=gt_nodes.device)

    # 5. Joint Snap Loss (tip-to-base connection in 3D)
    base = pred_x0[:, :, :3]
    pitch_rad = pred_x0[:, :, 5] * math.pi / 180.0
    yaw_rad = pred_x0[:, :, 6] * math.pi / 180.0
    length = pred_x0[:, :, 3]
    dir_x = torch.cos(pitch_rad) * torch.cos(yaw_rad)
    dir_y = torch.cos(pitch_rad) * torch.sin(yaw_rad)
    dir_z = torch.sin(pitch_rad)
    tip = base + length.unsqueeze(-1) * torch.stack([dir_x, dir_y, dir_z], dim=-1)

    diff = tip.unsqueeze(2) - base.unsqueeze(1)
    dist_sq = (diff ** 2).sum(dim=-1)
    loss_snap = (dist_sq * gt_adj).sum() / (gt_adj.sum() + 1e-5)

    # 6. Organ type classification (optional)
    loss_organ_type = torch.tensor(0.0, device=gt_nodes.device)
    if pred_organ_type_logits is not None:
        organ_target = torch.argmax(gt_nodes[:, :, 8:12], dim=-1)
        loss_organ_type = F.cross_entropy(
            pred_organ_type_logits.view(-1, 4),
            organ_target.view(-1),
            reduction='mean'
        )

    # 7. Noise prediction loss (for DDPM training)
    gt_noise = noisy_nodes - gt_nodes
    loss_noise = F.mse_loss(pred_node_noise, gt_noise)

    # Combined weighted loss
    loss = (
        10.0 * loss_coord +
        loss_x0 +
        0.5 * loss_existence +
        0.5 * loss_parent +
        0.2 * loss_snap +
        0.2 * loss_organ_type +
        loss_noise
    )

    metrics = {
        "coord": loss_coord.item(),
        "x0": loss_x0.item(),
        "existence": loss_existence.item(),
        "parent": loss_parent.item(),
        "snap": loss_snap.item(),
        "organ": loss_organ_type.item(),
        "noise": loss_noise.item(),
    }

    return loss, metrics


def train_3d_diffusion(
    data_dir: str = "Digital-Crops/projects/syntheticdata_generation/build/output",
    num_epochs: int = 500,
    batch_size: int = 4,
    lr: float = 3e-4,
    save_path: str = "diffusion_based/checkpoints/diffusion_3d_15d.pt",
    render_loss_weight: float = 0.0,  # Set > 0 to enable render-in-the-loop
):
    device = get_device()
    print(f"Training 15D 3D Plant Diffusion on device: {device}")

    max_nodes = 256
    node_dim = 15

    # Dataset
    dataset = HeliosPlantDataset(
        data_dir=data_dir,
        image_size=256,
        max_nodes=max_nodes,
        node_dim=node_dim,
    )

    if len(dataset) == 0:
        print(f"ERROR: No samples found in {data_dir}")
        print("Please run dataset generation first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset size: {len(dataset)} samples")

    # Model (15D)
    model = PlantGraphDiffuser3D(
        max_nodes=max_nodes,
        node_dim=node_dim,
        embed_dim=256,
        num_layers=4,
        k_nearest=16,
    ).to(device)

    # Differentiable renderer
    renderer = None
    if render_loss_weight > 0:
        renderer = DifferentiablePlantRenderer3D(image_size=256).to(device)
        print(f"Render-in-the-loop enabled (weight={render_loss_weight})")

    scheduler = DDPMScheduler(timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Training loop
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in dataloader:
            images = batch["image"].to(device)
            gt_nodes = batch["nodes"].to(device)
            gt_existence = batch["existence"].to(device)
            gt_adj = batch["adj"].to(device)
            gt_parents = batch["parents"].to(device)
            cam_az_norm = batch["cam_az"].to(device)
            sun_elev = batch["sun_elev"].to(device)
            sun_az = batch["sun_az"].to(device)
            dap = batch["dap"].to(device)

            B = gt_nodes.shape[0]

            # Sample timesteps and add noise
            timesteps = scheduler.sample_timesteps(B, device)
            noisy_nodes, noise = scheduler.add_noise(gt_nodes, timesteps)

            # Camera pose: [azimuth_norm, elevation_norm]
            camera_pose = torch.stack([
                cam_az_norm,
                torch.full((B,), 0.0, device=device),  # elevation = 0 after normalization
            ], dim=1)

            # Forward pass
            noisy_existence = gt_existence.unsqueeze(-1)
            outputs = model(
                noisy_nodes, noisy_existence, timesteps, images,
                camera_poses=camera_pose,
                dap=dap.unsqueeze(-1)
            )

            # Compute losses
            loss, metrics = compute_losses(
                outputs, gt_nodes, gt_existence, gt_parents, gt_adj,
                noisy_nodes
            )

            # Optional render loss
            if renderer is not None and render_loss_weight > 0:
                pred_nodes = outputs["pred_x0"]
                pred_existence = torch.sigmoid(outputs["pred_existence_logits"])

                # Camera azimuth in degrees for renderer
                cam_az_deg = (cam_az_norm + 1.0) * 180.0

                rendered = renderer(
                    pred_nodes,
                    parent_indices=gt_parents,
                    cam_azimuth_deg=cam_az_deg[0].item(),  # same for batch
                    focus_plant=True,
                    camera_params={
                        'camera_height': 1.0,
                        'ground_width': 1.5,
                        'sun_elevation_deg': sun_elev[0].item() * 90.0,
                        'sun_azimuth_deg': sun_az[0].item() * 360.0,
                    },
                    background="ground",
                )

                # Denormalize target image back to [0, 1]
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                target_rgb = images * std + mean
                target_rgb = torch.clamp(target_rgb, 0.0, 1.0)

                loss_render = F.mse_loss(rendered, target_rgb)
                loss = loss + render_loss_weight * loss_render
                metrics["render"] = loss_render.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

        lr_scheduler.step()

        # Logging
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if epoch % 10 == 0 or epoch == 1:
            log_msg = (f"Epoch [{epoch:03d}/{num_epochs}] - Loss: {avg_loss:.4f} "
                       f"Coord={metrics['coord']:.4f} X0={metrics['x0']:.4f} "
                       f"Exist={metrics['existence']:.4f} Parent={metrics['parent']:.4f} "
                       f"Snap={metrics['snap']:.4f} Organ={metrics['organ']:.4f} "
                       f"Noise={metrics['noise']:.4f}")
            if "render" in metrics:
                log_msg += f" Render={metrics['render']:.4f}"
            print(log_msg)

        # Save checkpoint
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path.replace('.pt', f'_epoch{epoch}.pt'))

    # Final save
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, save_path)
    print(f"Saved final model to '{save_path}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="Digital-Crops/projects/syntheticdata_generation/build/output")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", type=str,
                        default="diffusion_based/checkpoints/diffusion_3d_15d.pt")
    parser.add_argument("--render-loss", type=float, default=0.0,
                        help="Enable render-in-the-loop loss weight")
    args = parser.parse_args()

    train_3d_diffusion(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        render_loss_weight=args.render_loss,
    )
