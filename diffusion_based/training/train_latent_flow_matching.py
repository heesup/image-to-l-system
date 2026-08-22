"""
Training Script for 512D Latent Flow Matching (LFM).

Trains LatentFlowMatchingModel to predict straight velocity paths transporting
Gaussian prior z_0 ~ N(0, I) to continuous 512D plant organ latents z_1 in R^{N x 512},
conditioned on input RGB plant images.
"""

import os
import sys
import argparse
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.latent_flow_matching import LatentFlowMatchingModel
from diffusion_based.models.plant_vae import PlantOrganVAE
from diffusion_based.dataset.plant_latent_dataset import PlantLatentDataset, collate_latent_flow_batch


class LatentFlowScheduler:
    """
    Optimal Transport Conditional Flow Matching (OT-CFM) scheduler for continuous 512D latent space.
    """
    def __init__(self, sigma_min: float = 0.0):
        self.sigma_min = sigma_min

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Samples continuous time t ~ U[0, 1]."""
        return torch.rand(batch_size, device=device)

    def sample_zt(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Straight-line interpolation z_t = (1 - t) * z_0 + t * z_1."""
        t_view = t.view(-1, 1, 1)
        return (1.0 - t_view) * z0 + t_view * z1

    def velocity_target(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """Constant optimal transport velocity v = z_1 - z_0."""
        return z1 - z0

    @torch.no_grad()
    def sample_euler(
        self,
        model: nn.Module,
        images: torch.Tensor,
        num_organs: int = 1200,
        num_steps: int = 15,
        device: torch.device = None,
        z0: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Euler ODE integration from t=0 (Noise) to t=1 (Latent Plant Manifold).
        """
        B = images.shape[0]
        if device is None:
            device = images.device
        if z0 is None:
            z0 = torch.randn((B, num_organs, 512), device=device)

        z_t = z0
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            out = model(z_t, t, images)
            v_pred = out["pred_velocity"]
            z_t = z_t + v_pred * dt

        return z_t


def train_latent_flow_matching(
    dataset_dir: str = "dataset/helios_data",
    vae_ckpt: str = "diffusion_based/checkpoints/plant_organ_vae_best.pt",
    checkpoint_dir: str = "diffusion_based/checkpoints",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 2e-4,
    embed_dim: int = 512,
    encoder_layers: int = 6,
    decoder_layers: int = 6,
    max_samples: int = 1000,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[INFO] Initializing 512D Latent Flow Matching Training on {device}...")

    # 1. Load Pretrained PlantOrganVAE
    vae = PlantOrganVAE(latent_dim=512, hidden_dim=512).to(device)
    if os.path.exists(vae_ckpt):
        ckpt = torch.load(vae_ckpt, map_location=device)
        vae.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Successfully loaded pretrained VAE from {vae_ckpt} (Val Loss: {ckpt.get('val_loss', 0.0):.4f})")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # 2. Dataset & Dataloader
    dataset = PlantLatentDataset(
        data_root=dataset_dir,
        image_size=128,
        vae_model=vae,
        device=device,
        max_samples=max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_latent_flow_batch,
        num_workers=0,
    )
    print(f"[INFO] Loaded {len(dataset)} paired (Image, 512D Latent) samples.")

    # 3. Model & Scheduler
    model = LatentFlowMatchingModel(
        latent_dim=512,
        image_size=128,
        embed_dim=embed_dim,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        num_heads=8,
    ).to(device)

    scheduler = LatentFlowScheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float("inf")
    best_ckpt_path = os.path.join(checkpoint_dir, "latent_flow_matching_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        total_batches = len(dataloader)

        for batch in dataloader:
            images = batch["images"].to(device)       # (B, 3, 128, 128)
            z1 = batch["latents"].to(device)          # (B, N, 512)
            masks = batch["masks"].to(device)         # (B, N)
            B, N, D = z1.shape

            # Sample Prior Noise z_0 and Time t
            z0 = torch.randn_like(z1)
            t = scheduler.sample_time(B, device)
            zt = scheduler.sample_zt(z0, z1, t)
            v_target = scheduler.velocity_target(z0, z1)

            # Invert mask for Transformer key_padding_mask (True = padding)
            key_padding_mask = ~masks

            optimizer.zero_grad()
            out = model(zt, t, images, key_padding_mask=key_padding_mask)
            v_pred = out["pred_velocity"]

            # Masked Optimal Transport Velocity Loss (computed only on active organs)
            mask_expanded = masks.unsqueeze(-1).expand_as(v_pred)
            loss = F.mse_loss(v_pred[mask_expanded], v_target[mask_expanded])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        lr_scheduler.step()
        epoch_time = time.time() - t0
        avg_loss = running_loss / max(total_batches, 1)

        print(f"Epoch {epoch:02d}/{epochs:02d} ({epoch_time:4.1f}s) | "
              f"Flow MSE Loss: {avg_loss:.6f} | LR: {lr_scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "embed_dim": embed_dim,
                "latent_dim": 512,
            }, best_ckpt_path)
            print(f"  --> Saved Best Checkpoint to {best_ckpt_path} (Loss: {best_loss:.6f})")

    print(f"\n[OK] Latent Flow Matching Training Complete. Best Checkpoint saved to: {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()

    train_latent_flow_matching(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
    )
