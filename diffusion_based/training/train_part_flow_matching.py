"""
Training script for Part-Centric Flow Matching.

Trains a ViT + transformer decoder to predict the velocity field that transports
a Gaussian prior to the part tensor, conditioned on a rendered plant image.

Loss: masked MSE between predicted velocity and the constant velocity target
      v = x1 - x0, masked to active organ slots (existence > 0).
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
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.dataset.part_array_dataset import (
    PartArrayDataset, FM_OT_END, EMPTY_IDX, FM_BASE_START, FM_NODE_DIM,
)
from diffusion_based.models.part_flow_matching import PartFlowMatchingModel
from diffusion_based.training.flow_matching import FlowMatchingScheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: FlowMatchingScheduler,
    device: torch.device,
    ema_model: AveragedModel = None,
    global_step: int = 0,
    empty_prior: bool = False,
    epoch: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    count = 0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        nodes = batch["nodes"].to(device)  # (B, N, 16) normalized
        existence_mask = batch["existence_mask"].to(device)  # (B, N)

        B = images.shape[0]

        # Sample time and prior
        t = scheduler.sample_time(B, device)
        if empty_prior:
            # True Zero Plant Array Prior:
            # All 3D bases, rotations, scales, curvature, phyllotaxis = 0, and all slots are EMPTY.
            # No template coordinates are used.
            x0 = torch.zeros_like(nodes)
            x0[:, :, EMPTY_IDX] = 1.0
        else:
            x0 = torch.randn_like(nodes)  # Standard Gaussian Noise Prior

        x1 = nodes  # Ground truth 26D normalized organ array
        x_t = scheduler.sample_xt(x0, x1, t)
        v_target = scheduler.velocity_target(x0, x1)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(x_t, t, images)
            pred_velocity = outputs["pred_velocity"]

            # Category velocity loss (across all slots: learning active organ types vs empty)
            loss_cat = F.mse_loss(pred_velocity[:, :, :FM_OT_END], v_target[:, :, :FM_OT_END])

            # Geometry velocity loss (masked to active organ slots: base, rot, scale, curv, phyllo)
            active_mask = existence_mask.unsqueeze(-1).float()  # (B, N, 1)
            diff_geom = (pred_velocity[:, :, FM_BASE_START:] - v_target[:, :, FM_BASE_START:]) ** 2
            loss_geom = (diff_geom * active_mask).sum() / max(active_mask.sum() * (FM_NODE_DIM - FM_BASE_START), 1.0)

            loss = loss_cat + 2.0 * loss_geom

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        if ema_model is not None:
            ema_model.update_parameters(model)

        total_loss += loss.item() * B
        count += B

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
            print(f"  [Epoch {epoch:02d}] Step {batch_idx+1:03d}/{num_batches:03d} | Loss: {loss.item():.4f} (Cat: {loss_cat.item():.4f}, Geom: {loss_geom.item():.4f})", flush=True)

    return {"loss": total_loss / max(count, 1), "global_step": global_step}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--max_nodes", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="diffusion_based/checkpoints/fm")
    parser.add_argument("--val_pattern", type=str, default=None,
                        help="Comma-separated basename globs held out for validation, e.g. '*seed00*'")
    parser.add_argument("--cache_dir", type=str, default="dataset/cache",
                        help="Directory of precomputed part tensors + images (skip if absent)")
    parser.add_argument("--empty_prior", action="store_true",
                        help="Start flow-matching from an empty plant (existence=0) instead of Gaussian noise")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    val_globs = [g.strip() for g in args.val_pattern.split(",")] if args.val_pattern else []
    dataset = PartArrayDataset(
        data_root=args.data_root,
        max_nodes=args.max_nodes,
        image_size=args.image_size,
        device=device,
        use_gt_renderer_image=True,
        exclude_globs=val_globs if val_globs else None,
        cache_dir=args.cache_dir if os.path.isdir(args.cache_dir) else None,
    )
    print(f"Train dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = PartFlowMatchingModel(
        max_nodes=args.max_nodes,
        node_dim=dataset.node_dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=8,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = FlowMatchingScheduler()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, dataloader, optimizer, scheduler, device, ema_model, global_step,
                              empty_prior=args.empty_prior, epoch=epoch)
        global_step = metrics["global_step"]
        print(f"Epoch {epoch:03d} | loss={metrics['loss']:.4f}", flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"part_flow_matching_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}", flush=True)

    final_path = os.path.join(args.checkpoint_dir, "part_flow_matching.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
