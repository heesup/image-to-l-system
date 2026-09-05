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
from torch.utils.data.distributed import DistributedSampler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.distributed as dist

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.dataset.part_array_dataset import (
    PartArrayDataset, FM_OT_END, EMPTY_IDX, FM_NODE_DIM,
    FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END,
    FM_SCALE_START, FM_SCALE_END, FM_CURV,
)
from diffusion_based.models.botanical_scaffold import BotanicalScaffoldGenerator
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.part_flow_matching import PartFlowMatchingModel
from diffusion_based.training.flow_matching import FlowMatchingScheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_ddp() -> tuple:
    """
    Initializes torchrun-based DDP when launched with torchrun (LOCAL_RANK set).
    Returns (ddp_enabled, local_rank, world_size). No-op for plain python launches.
    """
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, dist.get_world_size()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: FlowMatchingScheduler,
    device: torch.device,
    ema_model: AveragedModel = None,
    global_step: int = 0,
    prior_type: str = "scaffold",
    scaffold_gen: BotanicalScaffoldGenerator = None,
    epoch: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    count = 0
    nan_skips = 0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        nodes = batch["nodes"].to(device)  # (B, N, 26) normalized
        existence_mask = batch["existence_mask"].to(device)  # (B, N)

        B = images.shape[0]

        if prior_type == "scaffold" and scaffold_gen is not None:
            # 3D Developmental Botanical Scaffold Prior conditioned on sample age (DAP)
            daps = batch.get("dap", None)
            if daps is not None:
                x0_list = []
                for b in range(B):
                    dap_b = float(daps[b].item())
                    x0_list.append(scaffold_gen.generate_from_dap(dap_b, device=device))
                x0 = torch.stack(x0_list, dim=0)
                base_noise = torch.randn((B, nodes.shape[1], 3), device=device) * 0.015
                x0[:, :, FM_BASE_START:FM_BASE_END] += base_noise
            else:
                x0 = scaffold_gen.sample_prior(B, device=device, noise_std=0.015)
        elif prior_type == "empty":
            # Zero Plant Array Prior
            x0 = torch.zeros_like(nodes)
            x0[:, :, EMPTY_IDX] = 1.0
        else:
            # Standard Gaussian Noise Prior
            x0 = torch.randn_like(nodes)

        x1 = nodes  # Ground truth 26D normalized organ array
        t = scheduler.sample_time(B, device)
        x_t = scheduler.sample_xt(x0, x1, t)
        v_target = scheduler.velocity_target(x0, x1)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(x_t, t, images)
        # Loss computed OUTSIDE autocast in fp32: bf16 square error on the
        # scale block (targets up to 100.0 = 2m * SCALE_SCALE) overflows to
        # inf -> NaN losses poison Adam moments and kill training (observed
        # step-25 NaN collapse).
        pred_velocity = outputs["pred_velocity"].float()
        v_target_f = v_target.float()

        active_mask = existence_mask.unsqueeze(-1).float()  # (B, N, 1)
        num_active = max(active_mask.sum(), 1.0)

        # Balanced category loss: active organ classification + existence velocity
        # BUG FIX 2026-09-04: the slice was `:EMPTY_IDX` where EMPTY_IDX = 0 →
        # empty tensor → NaN loss since the beginning. The one-hot category
        # block is cols 0:FM_OT_END (13 classes incl. ORGAN_NONE at col 0).
        diff_cat_active = (pred_velocity[:, :, :FM_OT_END] - v_target_f[:, :, :FM_OT_END]) ** 2
        loss_cat_active = (diff_cat_active * active_mask).sum() / (num_active * FM_OT_END)
        loss_empty = F.mse_loss(pred_velocity[:, :, EMPTY_IDX], v_target_f[:, :, EMPTY_IDX])
        loss_cat = loss_cat_active * 4.0 + loss_empty * 1.0

        # Geometry velocity losses (masked to active organ slots)
        diff_base = (pred_velocity[:, :, FM_BASE_START:FM_BASE_END] - v_target_f[:, :, FM_BASE_START:FM_BASE_END]) ** 2
        loss_base = (diff_base * active_mask).sum() / (num_active * 3.0)

        diff_rot = (pred_velocity[:, :, FM_ROT_START:FM_ROT_END] - v_target_f[:, :, FM_ROT_START:FM_ROT_END]) ** 2
        loss_rot = (diff_rot * active_mask).sum() / (num_active * 6.0)

        diff_scale = (pred_velocity[:, :, FM_SCALE_START:FM_SCALE_END] - v_target_f[:, :, FM_SCALE_START:FM_SCALE_END]) ** 2
        loss_scale = (diff_scale * active_mask).sum() / (num_active * 3.0)

        # Curvature velocity (26D col 25 = FM_CURV, deg/m * CURV_SCALE):
        # NEEDED — a dedicated term is not redundant. Analysis (2026-09-04):
        # in a UNIFIED 26-col MSE, curvature would receive 0.00% of the
        # gradient signal because the shard-normalized scale block has std
        # ~70.9 vs curvature ~0.61 (≈115x overweighted the other way; scale
        # alone = 97.28% of total variance). The trainer therefore balances
        # per-block explicitly: each block term is a per-slot-per-col mean,
        # and curvature has NO coverage elsewhere — loss_inactive_geom only
        # regularizes INACTIVE slots (no active_mask), so without loss_curv
        # active-slot curvature would receive zero dedicated gradient.
        # Weight 1.0 gives curvature 1/7 of the loss budget.
        diff_curv = (pred_velocity[:, :, FM_CURV] - v_target_f[:, :, FM_CURV]) ** 2
        loss_curv = (diff_curv.squeeze(-1) * active_mask.squeeze(-1)).sum() / num_active

        # Inactive slot velocity penalty: ensure unactivated slots remain smooth
        loss_inactive_geom = (pred_velocity[:, :, FM_BASE_START:] ** 2 * (1.0 - active_mask)).mean()

        loss = loss_cat + 1.0 * loss_base + 1.0 * loss_rot + 3.0 * loss_scale + 1.0 * loss_curv + 0.3 * loss_inactive_geom

        # NaN guard: skip poisoned batches (bf16 overflow or bad scaffold prior)
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            global_step += 1
            nan_skips += 1
            continue

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
            print(f"  [Epoch {epoch:02d}] Step {batch_idx+1:03d}/{num_batches:03d} | Loss: {loss.item():.4f} (Cat: {loss_cat.item():.4f}, Base: {loss_base.item():.4f}, Scale: {loss_scale.item():.4f}, Curv: {loss_curv.item():.4f} | NaN skips: {nan_skips})", flush=True)

    return {"loss": total_loss / max(count, 1), "global_step": global_step, "nan_skips": nan_skips}


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
    parser.add_argument("--prior_type", type=str, default="scaffold", choices=["scaffold", "empty", "gaussian"],
                        help="Prior distribution: scaffold (3D botanical lattice), empty (zero plant array), gaussian (standard noise)")
    parser.add_argument("--empty_prior", action="store_true",
                        help="Backward-compatibility alias for --prior_type empty")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="DataLoader workers per rank (CPU-bound dataset needs coverage)")
    parser.add_argument("--vis_every", type=int, default=1,
                        help="Render the diagnostic panel every N epochs (0 = never)")
    parser.add_argument("--vis_samples", type=int, default=3,
                        help="Number of plants in the diagnostic panel")
    parser.add_argument("--use-wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging (losses + panel)")
    parser.add_argument("--wandb-project", type=str, default="part-flow-matching")
    parser.add_argument("--wandb-group", type=str, default="fm-curv26")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    if args.empty_prior:
        args.prior_type = "empty"

    ddp_enabled, local_rank, world_size = setup_ddp()
    set_seed(args.seed + local_rank)  # decorrelate per-rank shuffling
    device = get_device()
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank == 0:
        print(f"Using device: {device} | DDP: {ddp_enabled} | world_size: {world_size}")
        print(f"Flow Matching Prior type: {args.prior_type}")

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

    scaffold_gen = BotanicalScaffoldGenerator(max_nodes=args.max_nodes) if args.prior_type == "scaffold" else None

    sampler = DistributedSampler(dataset, shuffle=True) if ddp_enabled else None
    # GPU utilization fix: the dataset __getitem__ is CPU-bound (XML parse +
    # FK evaluation + jpeg decode per sample). num_workers must cover it;
    # prefetch keeps the GPU fed. Batch sized to fill VRAM (A100/6000_ada 48GB
    # -> 9.3M model at 128px: batch 256 uses ~20GB).
    def fm_collate(batch):
        """
        Custom collate: crops nodes/existence to max_nodes per-sample (cache stores
        up to 4096 wide) so torch.stack works, and keys the batch dict to what
        train_epoch expects (image/nodes/existence_mask/dap).
        """
        import torch as _t
        max_nodes = args.max_nodes
        images, nodes, masks, daps = [], [], [], []
        for b in batch:
            img = b["image"]
            if img.shape[-1] != args.image_size:
                img = F.interpolate(img.unsqueeze(0), size=(args.image_size, args.image_size),
                                    mode="bilinear", align_corners=False).squeeze(0)
            images.append(img)
            n = b["nodes"][:max_nodes]
            m = b.get("existence_mask", None)
            if m is None:
                m = (n[:, EMPTY_IDX] < 0.5).float()
            m = m[:max_nodes]
            if m.shape[0] < max_nodes:
                m = _t.cat([m, _t.zeros(max_nodes - m.shape[0])], dim=0)
            if n.shape[0] < max_nodes:
                pad = _t.zeros((max_nodes - n.shape[0], n.shape[1]))
                pad[:, EMPTY_IDX] = 1.0
                n = _t.cat([n, pad], dim=0)
            nodes.append(n)
            masks.append(m.bool())
            daps.append(b["dap"] if isinstance(b["dap"], _t.Tensor) else _t.tensor(float(b["dap"])))
        return {
            "image": _t.stack(images, dim=0).float(),
            "nodes": _t.stack(nodes, dim=0).float(),
            "existence_mask": _t.stack(masks, dim=0),
            "dap": _t.stack(daps, dim=0).float(),
        }

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=fm_collate,
    )

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
    if ddp_enabled:
        # use find_unused_parameters: scaffold prior generator varies active-slot count
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
    raw_model = model.module if ddp_enabled else model
    if rank == 0:
        print(f"Model params: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f}M")

    ema_model = AveragedModel(raw_model, multi_avg_fn=get_ema_multi_avg_fn(0.9999)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = FlowMatchingScheduler()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Optional W&B — rank 0 ONLY (each DDP rank previously opened its own run,
    # producing 2 duplicate runs on the dashboard with 2 GPUs)
    wandb_run = None
    if args.use_wandb and rank == 0:
        try:
            from diffusion_based.training.fm_visualization import WANDB_AVAILABLE
            if WANDB_AVAILABLE:
                import wandb as _wandb
                wandb_run = _wandb.init(
                    project=args.wandb_project, group=args.wandb_group,
                    name=args.wandb_name or f"fm-curv26-e{args.epochs}",
                    config=vars(args),
                )
                print(f"W&B run initialized: {wandb_run.name}")
            else:
                print("wandb not installed; skipping W&B logging")
        except Exception as e:
            print(f"W&B init failed ({e}); continuing without it")

    # Fixed validation batch for the diagnostic panel (deterministic samples)
    vis_batch = None
    if args.vis_every > 0:
        from torch.utils.data import Subset
        # DAP-diverse vis samples: stride across the dataset so the panel shows
        # early/late growth stages instead of only DAP-1 (first samples are sorted by DAP)
        n_vis = max(args.vis_samples, 1)
        vis_idx = [int(i * (len(dataset) - 1) / max(n_vis - 1, 1)) for i in range(n_vis)]
        vis_loader = DataLoader(
            Subset(dataset, vis_idx),
            batch_size=max(args.vis_samples, 1), shuffle=False, num_workers=0,
            collate_fn=fm_collate,
        )
        vis_batch = next(iter(vis_loader))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if ddp_enabled:
            sampler.set_epoch(epoch)  # decorrelate shuffling across epochs
        metrics = train_epoch(
            model, dataloader, optimizer, scheduler, device, ema_model, global_step,
            prior_type=args.prior_type, scaffold_gen=scaffold_gen, epoch=epoch
        )
        global_step = metrics["global_step"]
        nan_s = metrics.get("nan_skips", 0)
        if rank == 0:
            print(f"Epoch {epoch:03d} | loss={metrics['loss']:.4f} | NaN skips: {nan_s}", flush=True)

        if wandb_run is not None and rank == 0:
            from diffusion_based.training.fm_visualization import log_epoch_scalars
            log_epoch_scalars(metrics, epoch, global_step, wandb_run)

        # Diagnostic panel (renders + curvature comparison) - rank 0 only
        if rank == 0 and args.vis_every > 0 and vis_batch is not None and (
            epoch % args.vis_every == 0 or epoch == args.epochs
        ):
            try:
                from diffusion_based.training.fm_visualization import render_epoch_panel
                render_epoch_panel(
                    ema_model, scaffold_gen, geo_builder=HeliosPlantGeometryBuilder(),
                    renderer=HeliosPyTorchRenderer(image_size=args.image_size),
                    batch=vis_batch, epoch=epoch, global_step=global_step,
                    out_dir="docs/results/assets",
                    num_rows=args.vis_samples, wandb_run=wandb_run,
                )
                print(f"Visualization panel saved (epoch {epoch:03d})", flush=True)
            except Exception as e:
                print(f"Visualization failed: {e}", flush=True)

        if rank == 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            ckpt_path = os.path.join(args.checkpoint_dir, f"part_flow_matching_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}", flush=True)

    final_path = os.path.join(args.checkpoint_dir, "part_flow_matching.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": raw_model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, final_path)
    if rank == 0:
        print(f"Training complete. Final checkpoint: {final_path}")
    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
