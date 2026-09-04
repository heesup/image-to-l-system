"""
Training Script for Stage-1 Conditional Scaffold Predictor.

Trains the visual encoder (RGB / Depth -> DAP & Canopy Bounds) on the Helios dataset
to output developmental bounding parameters (DAP, Radius, Height, Leaf Scale, Active Nodes)
for dynamic 3D Botanical Scaffold generation.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.dataset.part_array_dataset import (
    PartArrayDataset,
    EMPTY_IDX,
    P_COL_BASE_X,
    P_COL_BASE_Z,
    P_COL_SCALE_X,
    P_COL_SCALE_Z,
    P_COL_EXISTENCE,
    FM_BASE_START,
    FM_BASE_END,
    FM_SCALE_START,
    FM_SCALE_END,
    BASE_SCALE,
    SCALE_SCALE,
)
from diffusion_based.models.conditional_scaffold_predictor import ConditionalScaffoldPredictor


def main():
    parser = argparse.ArgumentParser(description="Train Stage-1 Conditional Scaffold Predictor")
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--cache_dir", type=str, default="dataset/cache")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="diffusion_based/checkpoints/fm")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Stage-1 Scaffold Predictor on {device}...")

    dataset = PartArrayDataset(
        data_root=os.path.join(REPO_ROOT, args.data_root),
        max_nodes=512,
        cache_dir=os.path.join(REPO_ROOT, args.cache_dir) if os.path.isdir(args.cache_dir) else None,
        device=device,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ConditionalScaffoldPredictor(in_channels=3, embed_dim=256, max_nodes=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)  # (B, 3, 128, 128)
            nodes = batch["nodes"].to(device)    # (B, N, 26)
            daps = batch["dap"].to(device).unsqueeze(-1)  # (B, 1)

            B = images.shape[0]

            # Compute Ground Truth Targets from nodes
            bases_m = nodes[:, :, FM_BASE_START:FM_BASE_END] / BASE_SCALE
            scales_m = nodes[:, :, FM_SCALE_START:FM_SCALE_END] / SCALE_SCALE
            is_active = (nodes[:, :, EMPTY_IDX] < 0.5).float()  # (B, N)

            gt_active_count = is_active.sum(dim=-1, keepdim=True).clamp(min=4.0)
            
            # Radii and Height
            r_xy = torch.linalg.norm(bases_m[:, :, :2], dim=-1) * is_active
            gt_radius = r_xy.max(dim=-1, keepdim=True)[0].clamp(min=0.08, max=1.2)

            z_pos = bases_m[:, :, 2] * is_active
            gt_height = z_pos.max(dim=-1, keepdim=True)[0].clamp(min=0.08, max=1.2)

            # Leaf scale
            leaf_mask = (nodes[:, :, 4] > 0.5) & (is_active > 0.5)  # ORGAN_LEAF = 4
            num_leaves = leaf_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            leaf_s = (scales_m[:, :, 0] * leaf_mask.float()).sum(dim=-1, keepdim=True) / num_leaves
            gt_leaf_scale = leaf_s.clamp(min=0.02, max=0.15)

            # Predict
            preds = model(images)

            loss_dap = F.mse_loss(preds["pred_dap"] / 100.0, daps / 100.0)
            loss_r = F.mse_loss(preds["pred_radius"], gt_radius)
            loss_h = F.mse_loss(preds["pred_height"], gt_height)
            loss_ls = F.mse_loss(preds["pred_leaf_scale"], gt_leaf_scale) * 10.0
            loss_cnt = F.mse_loss(preds["pred_active_count"] / 512.0, gt_active_count / 512.0)

            loss = loss_dap + 2.0 * loss_r + 2.0 * loss_h + loss_ls + 2.0 * loss_cnt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            count += B

        scheduler.step()
        avg_loss = total_loss / count
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} (DAP Loss: {loss_dap.item():.4f}, R Loss: {loss_r.item():.4f}, H Loss: {loss_h.item():.4f})", flush=True)

    ckpt_path = os.path.join(args.checkpoint_dir, "scaffold_predictor.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Successfully saved Stage-1 Scaffold Predictor checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
