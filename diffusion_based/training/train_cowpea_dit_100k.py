"""
Large-Scale Training Pipeline for 150M Parameter Cowpea DiT Flow Matching Model.
Trained on 100K synthetic Cowpea dataset shards with Mixed Precision & Variable-Length Collation.
"""

import os
import sys
import math
import argparse
from typing import Optional

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from diffusion_based.dataset.cowpea_shard_dataset import CowpeaShardDataset, cowpea_collate_fn
from diffusion_based.models.canonical_cowpea_dit_large import CanonicalCowpeaDiTLargeModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT-Large on Cowpea 100K Dataset")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--cache-dir", type=str, default="dataset/cache_cowpea_100k")
    parser.add_argument("--save-dir", type=str, default="diffusion_based/checkpoints/fm")
    parser.add_argument("--save-name", type=str, default="cowpea_dit_large_150m.pt")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"=================================================================")
    print(f"Starting 150M DiT-Large Cowpea Flow Matching Training on {device}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"=================================================================")

    dataset = CowpeaShardDataset(cache_dir=args.cache_dir, fallback_cache_dir="dataset/cache")
    if len(dataset) == 0:
        print("Error: No training shards or cache files found!")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cowpea_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    model = CanonicalCowpeaDiTLargeModel(
        max_slots=512,
        node_dim=26,
        image_size=128,
        patch_size=8,
        embed_dim=768,
        encoder_layers=16,
        decoder_layers=12,
        num_heads=16,
        dropout=0.05,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized: {total_params / 1e6:.2f}M Trainable Parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    
    total_steps = len(loader) * args.epochs
    warmup_steps = len(loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float("inf")
    save_path = os.path.join(args.save_dir, args.save_name)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            daps = batch["daps"].to(device, non_blocking=True)
            x1 = batch["nodes"].to(device, non_blocking=True)
            exist_mask = batch["existence_masks"].to(device, non_blocking=True)
            k_mask = batch["key_padding_masks"].to(device, non_blocking=True)
            gt_counts = batch["num_organs"].float().to(device, non_blocking=True)

            B, N, D = x1.shape
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=device)

            t_expand = t.view(B, 1, 1)
            x_t = (1.0 - t_expand) * x0 + t_expand * x1
            u_t = x1 - x0

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                out = model(x_t, t, images, daps, key_padding_mask=k_mask)
                v_pred = out["pred_velocity"]
                pred_counts = out["pred_count"]

                # Active slot velocity loss
                active_weights = torch.where(exist_mask.unsqueeze(-1) > 0.5, 1.0, 0.15)
                loss_v = (active_weights * (v_pred - u_t) ** 2).mean()
                
                # Count regression loss
                loss_count = 0.01 * F.mse_loss(pred_counts, gt_counts)
                loss = loss_v + loss_count

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        cur_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {cur_lr:.6e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, save_path)

    print(f"Training Complete! Saved best DiT-Large model to {save_path} (Best Loss: {best_loss:.4f})")


if __name__ == "__main__":
    train()
