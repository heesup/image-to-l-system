"""
Train Pure Transformer Global Plant VAE.
"""

import os
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.training.train_plant_global_and_shoot_vae import (
    FastPlantArrayDataset,
    collate_plant_batch,
)
from diffusion_based.models.plant_pure_transformer_vae import (
    PlantPureTransformerVAE,
    compute_pure_transformer_vae_loss,
)


def train(epochs: int = 30, batch_size: int = 16, lr: float = 3e-4, max_samples: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=======================================================")
    print(f"  Training PURE TRANSFORMER Plant VAE on {device}")
    print(f"=======================================================")

    dataset = FastPlantArrayDataset(
        data_root="dataset/helios_data",
        max_samples=max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_plant_batch)

    model = PlantPureTransformerVAE(
        latent_dim=512, hidden_dim=512, ffn_dim=2048,
        encoder_layers=6, decoder_layers=6, num_heads=8, dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Pure Transformer VAE Total Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs("diffusion_based/checkpoints", exist_ok=True)
    best_loss = float("inf")
    save_path = "diffusion_based/checkpoints/plant_pure_transformer_vae_best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_geom = 0.0
        total_cls = 0.0
        total_ang = 0.0
        n_batches = 0

        t0 = time.time()
        for batch in dataloader:
            x = batch["tensor"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            losses = compute_pure_transformer_vae_loss(model, x, mask=mask, beta=1e-4)
            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_geom += losses["loss_geom"].item()
            total_cls += losses["loss_cls"].item()
            total_ang += losses["loss_angle"].item()
            n_batches += 1

        scheduler.step()
        ep_time = time.time() - t0
        avg_loss = total_loss / n_batches
        avg_geom = total_geom / n_batches
        avg_cls = total_cls / n_batches
        avg_ang = total_ang / n_batches

        print(f"Epoch {ep:02d}/{epochs:02d} ({ep_time:4.1f}s) | Loss: {avg_loss:6.2f} | Geom: {avg_geom:.4f} | Cls: {avg_cls:.4f} | Ang: {avg_ang:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
            }, save_path)
            print(f"  --> Saved Best PURE TRANSFORMER Checkpoint to: {save_path}")

    print(f"\n[OK] Completed training Pure Transformer VAE. Checkpoint: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_samples=args.max_samples)
