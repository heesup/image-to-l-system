"""
Training Script: Canonical Cowpea DiT Flow Matching (DAP 010 - DAP 035).

Trains the Canonical Botanical Slot-Ordered DiT Model with dynamic variable-length
collation, DAP age conditioning, and balanced focal geometry loss.
"""

import os
import sys
import random
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.dataset.canonical_cowpea_dataset import CanonicalCowpeaDataset, canonical_collate_fn
from diffusion_based.dataset.part_array_dataset import EMPTY_IDX
from diffusion_based.models.canonical_cowpea_dit import CanonicalCowpeaDiTModel
from diffusion_based.training.flow_matching import FlowMatchingScheduler, FM_OT_END, FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END, FM_SCALE_START, FM_SCALE_END


def train_canonical_cowpea(
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 3e-4,
    data_root: str = "dataset/helios_data/cowpea",
    ckpt_dir: str = "diffusion_based/checkpoints/fm",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Starting Canonical Cowpea DiT Flow Matching Training on {device} (Epochs: {epochs}, Batch: {batch_size})...")

    # 1. Dataset with Canonical Botanical Slot Ordering (DAP 10 - 35)
    dataset = CanonicalCowpeaDataset(
        data_root=data_root,
        min_dap=5.0,
        max_dap=35.0,
        image_size=128,
        max_slots=512,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=canonical_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Scaled-up DiT Model with DAP conditioning and variable-length support
    model = CanonicalCowpeaDiTModel(
        max_slots=512,
        node_dim=26,
        image_size=128,
        patch_size=8,
        embed_dim=384,
        encoder_layers=8,
        decoder_layers=6,
        num_heads=12,
        dropout=0.05,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    fm_scheduler = FlowMatchingScheduler()
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float("inf")
    best_ckpt = os.path.join(ckpt_dir, "canonical_cowpea_dit_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            images = batch["image"].to(device)                 # (B, 3, 128, 128)
            daps = batch["dap"].to(device)                     # (B,)
            nodes_gt = batch["nodes"].to(device)               # (B, N_batch, 26)
            existence_mask = batch["existence_mask"].to(device)# (B, N_batch) bool
            key_pad_mask = batch["key_padding_mask"].to(device)# (B, N_batch) bool
            gt_counts = batch["num_organs"].to(device).float() # (B,)

            B, N_b, D = nodes_gt.shape

            # 1. PURE GAUSSIAN NOISE PRIOR x_0 ~ N(0, I) (ZERO LEAKAGE)
            x_0 = torch.randn((B, N_b, D), device=device)
            x_1 = nodes_gt

            # 2. Continuous Timestep Sampling
            t = fm_scheduler.sample_time(B, device)

            # 3. Straight Path Interpolation & Target Velocity
            x_t = fm_scheduler.sample_xt(x_0, x_1, t)
            v_target = fm_scheduler.velocity_target(x_0, x_1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    noisy_slots=x_t,
                    timesteps=t,
                    images=images,
                    daps=daps,
                    key_padding_mask=key_pad_mask,
                )
                v_pred = outputs["pred_velocity"]
                pred_count = outputs["pred_count"].squeeze(-1)

                # Organ count auxiliary loss
                loss_count = F.l1_loss(pred_count, gt_counts) * 0.1

                # Active organ mask
                active_mask = existence_mask.unsqueeze(-1).float()  # (B, N_batch, 1)
                num_active = max(active_mask.sum(), 1.0)

                # Category classification loss (One-Hot)
                diff_cat = (v_pred[:, :, :EMPTY_IDX] - v_target[:, :, :EMPTY_IDX]) ** 2
                loss_cat = (diff_cat * active_mask).sum() / (num_active * EMPTY_IDX)

                # Existence loss
                loss_exist = F.mse_loss(v_pred[:, :, EMPTY_IDX], v_target[:, :, EMPTY_IDX])

                # Position, Rotation, Scale Geometry Losses
                diff_base = (v_pred[:, :, FM_BASE_START:FM_BASE_END] - v_target[:, :, FM_BASE_START:FM_BASE_END]) ** 2
                loss_base = (diff_base * active_mask).sum() / (num_active * 3.0)

                diff_rot = (v_pred[:, :, FM_ROT_START:FM_ROT_END] - v_target[:, :, FM_ROT_START:FM_ROT_END]) ** 2
                loss_rot = (diff_rot * active_mask).sum() / (num_active * 6.0)

                diff_scale = (v_pred[:, :, FM_SCALE_START:FM_SCALE_END] - v_target[:, :, FM_SCALE_START:FM_SCALE_END]) ** 2
                loss_scale = (diff_scale * active_mask).sum() / (num_active * 3.0)

                # Total Balanced Loss
                loss = (
                    loss_cat * 5.0 +
                    loss_exist * 2.0 +
                    loss_base * 4.0 +
                    loss_rot * 4.0 +
                    loss_scale * 5.0 +
                    loss_count
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * B
            count += B

        lr_scheduler.step()
        avg_loss = total_loss / count
        print(f"Epoch {epoch:02d}/{epochs:02d} | Velocity Loss: {avg_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, best_ckpt)

    print(f"\nTraining Complete! Best Canonical Model Saved to: {best_ckpt} (Best Loss: {best_loss:.4f})")
    return best_ckpt


if __name__ == "__main__":
    train_canonical_cowpea(epochs=40, batch_size=32, lr=3e-4)
