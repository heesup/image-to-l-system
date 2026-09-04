import os
import sys
import glob
import math
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from diffusion_based.dataset.cowpea_shard_dataset import CowpeaShardDataset, cowpea_collate_fn
from diffusion_based.models.canonical_cowpea_dit_large import CanonicalCowpeaDiTLargeModel
from diffusion_based.models.plant_organ_array import PlantOrganArray, NUM_FEATURES_PART
from diffusion_based.models.part_assembly_to_xml import PartAssemblyToXMLConverter
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.dataset.part_array_dataset import (
    ORGAN_CATEGORIES, EMPTY_IDX, FM_NODE_DIM, FM_OT_END,
    FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END,
    FM_SCALE_START, FM_SCALE_END, FM_CURV_IDX, FM_PHYLLO_IDX
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT-Large on Cowpea 100K Dataset")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU micro-step")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps (effective batch size = batch_size * grad_accum_steps)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--cache-dir", type=str, default="dataset/helios_data/cowpea_shard")
    parser.add_argument("--data-root", type=str, default="dataset/helios_data/cowpea")
    parser.add_argument("--save-dir", type=str, default="diffusion_based/checkpoints/fm")
    parser.add_argument("--save-name", type=str, default="cowpea_dit_large_150m.pt")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cowpea-dit-generation", help="W&B Project name")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B Experiment Group (default: YYYY-MM-DD_cowpea_100k)")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B Run Display Name")
    parser.add_argument("--eval-every", type=int, default=2, help="Epoch interval to log 3D visual reconstruction debug images")
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_fixed_eval_samples(data_root: str) -> List[Dict[str, Any]]:
    """Loads canonical test cases across early, mid, and late growth stages."""
    target_daps = [10, 35, 70]
    samples = []
    for d in target_daps:
        xmls = sorted(glob.glob(os.path.join(data_root, f"*dap0{d}*_plant_0000.xml")))
        if not xmls:
            xmls = sorted(glob.glob(os.path.join(data_root, f"*dap{d}*_plant_0000.xml")))
        if not xmls:
            continue
        xml_path = xmls[0]
        prefix = os.path.basename(xml_path).split("_plant_0000.xml")[0]
        img_path = os.path.join(data_root, f"{prefix}_rad.jpeg")
        if os.path.exists(img_path):
            samples.append({
                "name": f"DAP {d:02d}",
                "dap": float(d),
                "xml": xml_path,
                "img": img_path,
            })
    return samples


def render_and_log_debug_images(
    model: nn.Module,
    eval_samples: List[Dict[str, Any]],
    renderer: HeliosPyTorchRenderer,
    assembler: PartAssemblyToXMLConverter,
    device: torch.device,
    epoch: int,
    global_step: int,
):
    if not eval_samples:
        return
    model.eval()
    fig, axes = plt.subplots(len(eval_samples), 4, figsize=(16, 3.8 * len(eval_samples)))
    if len(eval_samples) == 1:
        axes = np.expand_dims(axes, 0)
    fig.patch.set_facecolor("#080C14")

    for row_idx, sc in enumerate(eval_samples):
        arr_gt = PlantOrganArray.from_xml_file(sc["xml"])
        mesh_gt = renderer.geo_builder.build_mesh_from_part_tensor(arr_gt.to_part_tensor(device=device), device=device)
        rgb_gt = renderer(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
        rgb_gt_np = rgb_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        pil_img = Image.open(sc["img"]).convert("RGB").resize((128, 128))
        img_np = np.array(pil_img) / 255.0
        img_t = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        dap_t = torch.tensor([sc["dap"]], dtype=torch.float32, device=device)

        gt_count = arr_gt.tensor.shape[0]
        n_slots = max(64, int(gt_count * 1.15))

        with torch.no_grad():
            torch.manual_seed(100 + row_idx)
            x_t = torch.randn((1, n_slots, 26), device=device)
            num_steps = 30
            dt = 1.0 / num_steps
            for s in range(num_steps):
                t_val = torch.full((1,), s * dt, device=device)
                out = model(x_t, t_val, img_t, dap_t)
                x_t = x_t + out["pred_velocity"] * dt
            x_gen = x_t.squeeze(0)

        # High-fidelity Part Tensor Decoding
        ot_probs = torch.softmax(x_gen[:, :FM_OT_END], dim=-1)
        empty_prob = ot_probs[:, EMPTY_IDX]
        exist_prob = 1.0 - empty_prob

        ot_probs_real = x_gen[:, :EMPTY_IDX]
        ot_idx = ot_probs_real.argmax(dim=-1)
        raw_ot = torch.tensor([ORGAN_CATEGORIES[min(i.item(), len(ORGAN_CATEGORIES)-1)] for i in ot_idx], device=device).float()

        part_16d = torch.zeros((n_slots, NUM_FEATURES_PART), device=device)
        part_16d[:, P_COL_ORGAN_TYPE] = raw_ot
        part_16d[:, P_COL_EXISTENCE] = exist_prob.clamp(0.0, 1.0)
        part_16d[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = x_gen[:, FM_BASE_START:FM_BASE_END] / 20.0
        part_16d[:, P_COL_ROT_0:P_COL_ROT_5 + 1] = x_gen[:, FM_ROT_START:FM_ROT_END]
        part_16d[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] = F.softplus(x_gen[:, FM_SCALE_START:FM_SCALE_END]).clamp(min=0.001) / 50.0
        part_16d[:, P_COL_CURVATURE] = x_gen[:, FM_CURV_IDX] * 100.0
        part_16d[:, P_COL_PHYLLOTACTIC_ANGLE] = x_gen[:, FM_PHYLLO_IDX] * 180.0

        try:
            mesh_gen = renderer.geo_builder.build_mesh_from_part_tensor(part_16d, device=device, existence_threshold=0.30)
            rgb_gen = renderer(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
            rgb_gen_np = rgb_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            depth_gen = renderer.render_depth(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
            depth_np = depth_gen.detach().cpu().numpy()
            d_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
        except Exception:
            rgb_gen_np = np.zeros((128, 128, 3))
            d_norm = np.zeros((128, 128))

        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title(f"{sc['name']} Input RGB", color="white", fontsize=11)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(rgb_gt_np)
        axes[row_idx, 1].set_title(f"GT 3D Mesh ({gt_count} organs)", color="#4ADE80", fontsize=11)
        axes[row_idx, 1].axis("off")

        active_n = int((exist_prob >= 0.30).sum().item())
        axes[row_idx, 2].imshow(rgb_gen_np)
        axes[row_idx, 2].set_title(f"Generated 3D ({active_n} organs)", color="#60A5FA", fontsize=11)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(d_norm, cmap="plasma")
        axes[row_idx, 3].set_title("Generated 3D Depth", color="#F472B6", fontsize=11)
        axes[row_idx, 3].axis("off")

    plt.tight_layout()
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "eval/visual_reconstructions": wandb.Image(fig),
            "epoch": epoch
        }, step=global_step)
    plt.close(fig)
    model.train()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    group_name = args.wandb_group or f"{date_str}_cowpea_100k"
    run_name = args.wandb_name or f"[{date_str} #{timestamp[:4]}] DiT-Large_bs{args.batch_size * args.grad_accum_steps}_lr{args.lr}"

    print(f"=================================================================")
    print(f"Starting 232M DiT-Large Cowpea Flow Matching Training on {device}")
    print(f"Epochs: {args.epochs} | Effective Batch Size: {args.batch_size * args.grad_accum_steps} | LR: {args.lr}")
    print(f"W&B Logging: {'Enabled (' + run_name + ')' if args.use_wandb and WANDB_AVAILABLE else 'Disabled'}")
    print(f"=================================================================")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            group=group_name,
            name=run_name,
            tags=["cowpea", "dit-large", "flow-matching", "4096-slots", "26d", "100k"],
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size * args.grad_accum_steps,
                "micro_batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "lr": args.lr,
                "embed_dim": 768,
                "encoder_layers": 16,
                "decoder_layers": 12,
                "num_heads": 16,
                "max_slots": 4096,
                "node_dim": 26,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )

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
        max_slots=4096,
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
    
    total_steps = (len(loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = (len(loader) // args.grad_accum_steps) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda')

    # Visual rendering setup for debug logging
    renderer = HeliosPyTorchRenderer(image_size=256).to(device)
    assembler = PartAssemblyToXMLConverter()
    eval_samples = load_fixed_eval_samples(args.data_root)

    best_loss = float("inf")
    save_path = os.path.join(args.save_dir, args.save_name)
    accum_steps = args.grad_accum_steps
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_v_loss = 0.0
        epoch_c_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader):
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

            with torch.amp.autocast('cuda'):
                out = model(x_t, t, images, daps, key_padding_mask=k_mask)
                v_pred = out["pred_velocity"]
                pred_counts = out["pred_count"]

                # Active slot velocity loss
                active_weights = torch.where(exist_mask.unsqueeze(-1) > 0.5, 1.0, 0.15)
                loss_v = (active_weights * (v_pred - u_t) ** 2).mean()
                
                # Count regression loss
                loss_count = 0.01 * F.mse_loss(pred_counts.squeeze(), gt_counts.squeeze())
                loss = (loss_v + loss_count) / accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            raw_step_loss = (loss_v + loss_count).item()
            epoch_loss += raw_step_loss
            epoch_v_loss += loss_v.item()
            epoch_c_loss += loss_count.item()
            num_batches += 1

            # Log to console & W&B every 50 optimizer steps
            if (batch_idx + 1) % (accum_steps * 25) == 0 or (batch_idx + 1) == len(loader):
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:02d} [{batch_idx+1:05d}/{len(loader):05d}] | Loss: {raw_step_loss:.4f} (v: {loss_v.item():.4f}, count: {loss_count.item():.4f}) | Nodes: {N:4d} | LR: {cur_lr:.4e}", flush=True)

                if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "train/step_loss": raw_step_loss,
                        "train/velocity_loss": loss_v.item(),
                        "train/count_loss": loss_count.item(),
                        "train/learning_rate": cur_lr,
                        "train/max_nodes_in_batch": N,
                    }, step=global_step)

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_v_loss = epoch_v_loss / max(num_batches, 1)
        avg_c_loss = epoch_c_loss / max(num_batches, 1)
        cur_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{args.epochs} | Avg Epoch Loss: {avg_loss:.4f} (v: {avg_v_loss:.4f}, c: {avg_c_loss:.4f}) | LR: {cur_lr:.6e}")

        if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "epoch/loss": avg_loss,
                "epoch/velocity_loss": avg_v_loss,
                "epoch/count_loss": avg_c_loss,
                "epoch/epoch": epoch,
            }, step=global_step)

        # Render & Log 3D Reconstructed Visual Debug Images
        if epoch % args.eval_every == 0 or epoch == 1:
            print(f"Rendering 3D visual reconstruction debug images for Epoch {epoch}...")
            render_and_log_debug_images(model, eval_samples, renderer, assembler, device, epoch, global_step)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, save_path)
            if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"eval/best_loss": best_loss}, step=global_step)

    print(f"Training Complete! Saved best DiT-Large model to {save_path} (Best Loss: {best_loss:.4f})")
    if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    train()
