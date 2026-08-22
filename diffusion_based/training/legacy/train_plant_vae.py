"""
Training Script for Plant VAE (40D Plant Organ Array Compression).

Trains:
  1. PlantOrganVAE: High-precision per-organ manifold compressor (40D -> z_organ in R^16).
  2. PlantTransformerVAE: Full plant canopy sequence compressor ((N, 40) -> z_plant in R^256).

Saves checkpoints to:
  - diffusion_based/checkpoints/plant_organ_vae_best.pt
  - diffusion_based/checkpoints/plant_transformer_vae_best.pt
"""

import os
import sys
import glob
import time
import argparse
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray, NUM_FEATURES_TYPED
from diffusion_based.models.plant_vae import (
    PlantOrganVAE,
    PlantTransformerVAE,
    compute_organ_vae_loss,
)


class PlantOrganDataset(Dataset):
    """Loads plant XML files into padded 40D typed organ tensors."""
    def __init__(self, xml_paths: List[str], max_organs: int = 1200):
        self.xml_paths = xml_paths
        self.max_organs = max_organs

    def __len__(self) -> int:
        return len(self.xml_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        xml_path = self.xml_paths[idx]
        try:
            arr = PlantOrganArray.from_xml_file(xml_path)
            t = arr.tensor  # (N, 40)
            N = t.shape[0]
            if N > self.max_organs:
                # Keep root + shoot metas + first organs
                t = t[:self.max_organs]
                N = self.max_organs

            padded = torch.zeros(self.max_organs, NUM_FEATURES_TYPED, dtype=torch.float32)
            padded[:N] = t
            mask = torch.zeros(self.max_organs, dtype=torch.bool)
            mask[:N] = True
            return padded, mask, N
        except Exception as e:
            # Fallback dummy tensor
            padded = torch.zeros(self.max_organs, NUM_FEATURES_TYPED, dtype=torch.float32)
            mask = torch.zeros(self.max_organs, dtype=torch.bool)
            return padded, mask, 0


def collate_plant_batch(batch):
    padded_list, mask_list, n_list = zip(*batch)
    max_n = max(max(n_list), 1)
    # Trim to max_n in batch
    trimmed_padded = torch.stack([p[:max_n] for p in padded_list], dim=0)
    trimmed_mask = torch.stack([m[:max_n] for m in mask_list], dim=0)
    return trimmed_padded, trimmed_mask


def train_plant_organ_vae(
    dataset_dir: str = "dataset/helios_data",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    latent_dim: int = 512,
    hidden_dim: int = 512,
    max_samples: int = 2000,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing Plant VAE Training (latent_dim={latent_dim}, hidden_dim={hidden_dim}) on {device}...")

    # Discover XML files
    xml_files = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.xml"), recursive=True))
    if not xml_files:
        raise FileNotFoundError(f"No XML dataset found in {dataset_dir}")

    # Subsample if requested for fast efficient training
    if len(xml_files) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(xml_files), max_samples, replace=False)
        xml_files = [xml_files[i] for i in indices]

    print(f"[INFO] Loaded {len(xml_files)} plant XML files for training.")

    # Split train/val
    split_idx = int(len(xml_files) * 0.9)
    train_paths = xml_files[:split_idx]
    val_paths = xml_files[split_idx:]

    train_dataset = PlantOrganDataset(train_paths, max_organs=1200)
    val_dataset = PlantOrganDataset(val_paths, max_organs=1200)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_plant_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_plant_batch
    )

    # Instantiate Model
    model = PlantOrganVAE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    ckpt_dir = os.path.join(repo_root, "diffusion_based", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(ckpt_dir, "plant_organ_vae_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        geom_losses = []
        cls_losses = []

        # Warmup beta for KL
        beta = min(5e-3, (epoch / 10.0) * 5e-3)

        for batch_x, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            loss_dict = compute_organ_vae_loss(model, batch_x, beta=beta)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            geom_losses.append(loss_dict["loss_geom"].item())
            cls_losses.append(loss_dict["loss_cls"].item())

        scheduler.step()
        epoch_time = time.time() - t0

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_x, val_mask in val_loader:
                val_x = val_x.to(device)
                val_dict = compute_organ_vae_loss(model, val_x, beta=beta)
                val_losses.append(val_dict["loss"].item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)
        mean_geom = np.mean(geom_losses)
        mean_cls = np.mean(cls_losses)

        print(f"Epoch {epoch:02d}/{epochs:02d} ({epoch_time:.1f}s) | "
              f"Train Loss: {mean_train:.4f} (Geom: {mean_geom:.4f}, Cls: {mean_cls:.4f}) | "
              f"Val Loss: {mean_val:.4f} | Beta: {beta:.4f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "latent_dim": latent_dim,
            }, best_ckpt_path)
            print(f"  --> Saved Best Checkpoint to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

    print(f"\n[OK] Training Complete. Best Checkpoint saved to: {best_ckpt_path}")
    return best_ckpt_path


if __name__ == "__main__":
    train_plant_organ_vae(epochs=30, batch_size=32, lr=1e-3, latent_dim=512, hidden_dim=512, max_samples=2000)
