"""
High-Performance Distributed Data Parallel (DDP) Training Pipeline for 232M Cowpea DiT-Large.
Optimized for 2x / 4x NVIDIA H100 80GB SXM5/PCIe with BFloat16 Mixed Precision & NCCL NVLink.
"""

import os
import sys
import glob
import math
import json
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
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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
    parser = argparse.ArgumentParser(description="Train DiT-Large on Cowpea 100K Dataset with Multi-GPU DDP")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size per GPU")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate for multi-GPU")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--cache-dir", type=str, default="dataset/helios_data/cowpea_shard")
    parser.add_argument("--data-root", type=str, default="dataset/helios_data/cowpea")
    parser.add_argument("--save-dir", type=str, default="diffusion_based/checkpoints/fm")
    parser.add_argument("--save-name", type=str, default="cowpea_dit_large_h100_ddp.pt")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cowpea-dit-generation", help="W&B Project name")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B Experiment Group")
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
        cam_path = os.path.join(data_root, f"{prefix}_camera.json")
        if os.path.exists(img_path):
            samples.append({
                "name": f"DAP {d:02d}",
                "dap": float(d),
                "xml": xml_path,
                "img": img_path,
                "cam": cam_path,
            })
    return samples


def render_and_log_debug_images(
    raw_model: nn.Module,
    eval_samples: List[Dict[str, Any]],
    renderer: HeliosPyTorchRenderer,
    assembler: PartAssemblyToXMLConverter,
    device: torch.device,
    epoch: int,
    global_step: int,
):
    if not eval_samples:
        return
    raw_model.eval()
    fig, axes = plt.subplots(len(eval_samples), 6, figsize=(22, 3.8 * len(eval_samples)))
    if len(eval_samples) == 1:
        axes = np.expand_dims(axes, 0)
    fig.patch.set_facecolor("#080C14")

    for row_idx, sc in enumerate(eval_samples):
        # Extract exact camera configuration matching Helios C++
        cam_hfov = None
        cam_h = 5.0
        cam_el = 90.0
        if os.path.exists(sc.get("cam", "")):
            with open(sc["cam"], "r") as f:
                cam_data = json.load(f)
            f_len = cam_data.get("camera_properties", {}).get("focal_length", 50.0)
            s_w = cam_data.get("camera_properties", {}).get("sensor_width", 35.0)
            cam_h = float(cam_data.get("acquisition_properties", {}).get("camera_height_m", 5.0))
            cam_el = float(cam_data.get("acquisition_properties", {}).get("camera_angle_deg", 90.0))
            cam_hfov = 2.0 * math.degrees(math.atan((s_w * 0.5) / max(f_len, 1e-3)))

        # 1. Ground Truth 3D Mesh & Differentiable Renderings (Col 1, 2, 3)
        arr_gt = PlantOrganArray.from_xml_file(sc["xml"])
        mesh_gt = renderer.geo_builder.build_mesh_from_organ_array(arr_gt, device=device)
        rgb_gt = renderer.render_mesh(
            mesh_gt, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
            background="white", focus_plant=(cam_hfov is None), hfov_override_deg=cam_hfov, image_size=128
        )
        rgb_gt_np = rgb_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        depth_gt = renderer.render_depth(
            mesh_gt, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
            focus_plant=(cam_hfov is None), hfov_override_deg=cam_hfov, image_size=128
        )
        depth_gt_np = depth_gt.detach().cpu().numpy()
        d_gt_norm = (depth_gt_np - depth_gt_np.min()) / (depth_gt_np.max() - depth_gt_np.min() + 1e-6)

        seg_gt = renderer.render_organ_segmentation(
            mesh_gt, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
            focus_plant=(cam_hfov is None), hfov_override_deg=cam_hfov, image_size=128
        )
        seg_gt_np = seg_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        # 2. Input Image & Model Inference
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
                out = raw_model(x_t, t_val, img_t, dap_t)
                x_t = x_t + out["pred_velocity"] * dt
            x_gen = x_t.squeeze(0)

        ot_probs = torch.softmax(x_gen[:, :FM_OT_END], dim=-1)
        exist_prob = 1.0 - ot_probs[:, EMPTY_IDX]
        active_n = int((exist_prob >= 0.30).sum().item())

        # 3. Generated 3D Mesh & Differentiable Renderings (Col 4, 5, 6)
        try:
            mesh_gen = renderer.geo_builder.build_mesh_from_part_tensor(x_gen, device=device, existence_threshold=0.30)
            rgb_gen = renderer.render_mesh(
                mesh_gen, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
                background="white", focus_plant=(cam_hfov is None), hfov_override_deg=cam_hfov, image_size=128
            )
            rgb_gen_np = rgb_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

            depth_gen = renderer.render_depth(
                mesh_gen, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
                focus_plant=(cam_hfov is None), hfov_override_deg=cam_hfov, image_size=128
            )
            depth_gen_np = depth_gen.detach().cpu().numpy()
            d_gen_norm = (depth_gen_np - depth_gen_np.min()) / (depth_gen_np.max() - depth_gen_np.min() + 1e-6)

            seg_gen = renderer.render_organ_segmentation(
                mesh_gen, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
                focus_plant=(cam_hfov is None), hfov_override_deg=cam_hfov, image_size=128
            )
            seg_gen_np = seg_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        except Exception as e:
            rgb_gen_np = np.zeros((128, 128, 3))
            d_gen_norm = np.zeros((128, 128))
            seg_gen_np = np.zeros((128, 128, 3))

        # Col 1: PyTorch Differentiable RGB Input (GT)
        axes[row_idx, 0].imshow(rgb_gt_np)
        axes[row_idx, 0].set_title(f"{sc['name']}\nDiff RGB Input ({gt_count})", color="#4ADE80", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 2: PyTorch Differentiable Depth Input (GT)
        axes[row_idx, 1].imshow(d_gt_norm, cmap="plasma")
        axes[row_idx, 1].set_title("Diff Depth Input", color="#22D3EE", fontsize=10, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 3: PyTorch Differentiable Organ Segmentation Mask Input (GT)
        axes[row_idx, 2].imshow(seg_gt_np)
        axes[row_idx, 2].set_title("Diff Organ Seg Input", color="#A78BFA", fontsize=10, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # Col 4: Generated PyTorch Differentiable RGB
        axes[row_idx, 3].imshow(rgb_gen_np)
        axes[row_idx, 3].set_title(f"Gen Diff RGB\n({active_n} organs)", color="#60A5FA", fontsize=10, fontweight="bold")
        axes[row_idx, 3].axis("off")

        # Col 5: Generated PyTorch Differentiable Depth
        axes[row_idx, 4].imshow(d_gen_norm, cmap="plasma")
        axes[row_idx, 4].set_title("Gen Diff Depth", color="#F472B6", fontsize=10, fontweight="bold")
        axes[row_idx, 4].axis("off")

        # Col 6: Generated PyTorch Differentiable Organ Segmentation Mask
        axes[row_idx, 5].imshow(seg_gen_np)
        axes[row_idx, 5].set_title("Gen Diff Organ Seg", color="#FB923C", fontsize=10, fontweight="bold")
        axes[row_idx, 5].axis("off")

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.2)

    plt.tight_layout()
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "eval/visual_reconstructions": wandb.Image(fig),
            "epoch": epoch
        }, step=global_step)
    plt.close(fig)
    raw_model.train()


def train():
    args = parse_args()

    # DDP Initialization
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = (rank == 0)
    effective_batch_size = args.batch_size * args.grad_accum_steps * world_size

    if is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M%S")
        group_name = args.wandb_group or f"{date_str}_cowpea_100k_h100_ddp"
        run_name = args.wandb_name or f"[{date_str} #{timestamp[:4]}] DiT-Large_2xH100_bs{effective_batch_size}_lr{args.lr}"

        print(f"=================================================================")
        print(f"🚀 Starting 232M DiT-Large Cowpea Flow Matching on {world_size}x H100 (DDP)")
        print(f"Epochs: {args.epochs} | Effective Batch: {effective_batch_size} (Micro: {args.batch_size}, Accum: {args.grad_accum_steps}) | LR: {args.lr}")
        print(f"W&B Logging: {'Enabled (' + run_name + ')' if args.use_wandb and WANDB_AVAILABLE else 'Disabled'}")
        print(f"=================================================================")

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                group=group_name,
                name=run_name,
                tags=["cowpea", "dit-large", "flow-matching", "h100", "ddp", "4096-slots", "26d"],
                config={
                    "epochs": args.epochs,
                    "effective_batch_size": effective_batch_size,
                    "micro_batch_size_per_gpu": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "num_gpus": world_size,
                    "lr": args.lr,
                    "embed_dim": 768,
                    "encoder_layers": 16,
                    "decoder_layers": 12,
                    "num_heads": 16,
                    "max_slots": 4096,
                    "node_dim": 26,
                    "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                }
            )

    # Dataset & Distributed Sampler
    dataset = CowpeaShardDataset(cache_dir=args.cache_dir, fallback_cache_dir="dataset/cache")
    if len(dataset) == 0:
        if is_main_process:
            print("Error: No training shards or cache files found!")
        sys.exit(1)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if is_distributed else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=cowpea_collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Build Model & Wrap in DDP
    base_model = CanonicalCowpeaDiTLargeModel(
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

    if is_distributed:
        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = base_model

    if is_main_process:
        total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"Model Initialized: {total_params / 1e6:.2f}M Trainable Parameters on {world_size} GPUs.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    
    total_steps = (len(loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = (len(loader) // args.grad_accum_steps) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Visual rendering setup
    renderer = HeliosPyTorchRenderer(image_size=128).to(device)
    assembler = PartAssemblyToXMLConverter()
    eval_samples = load_fixed_eval_samples(args.data_root) if is_main_process else []

    best_loss = float("inf")
    save_path = os.path.join(args.save_dir, args.save_name)
    accum_steps = args.grad_accum_steps
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if is_distributed:
            sampler.set_epoch(epoch)

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

            # DDP Gradient Accumulation with no_sync()
            is_accumulating = (batch_idx + 1) % accum_steps != 0 and (batch_idx + 1) != len(loader)
            sync_context = model.no_sync() if is_accumulating and is_distributed else torch.amp.autocast('cuda', enabled=False)

            with sync_context:
                # Native BFloat16 mixed precision on H100 (Optimal Speed & Precision)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x_t, t, images, daps, key_padding_mask=k_mask)
                    v_pred = out["pred_velocity"]
                    pred_counts = out["pred_count"]

                    active_weights = torch.where(exist_mask.unsqueeze(-1) > 0.5, 1.0, 0.15)
                    loss_v = (active_weights * (v_pred.float() - u_t) ** 2).mean()
                    loss_count = 0.5 * F.smooth_l1_loss(pred_counts.squeeze() / 100.0, gt_counts.squeeze() / 100.0)

                    # Every N-step Differentiable Depth Regularization (Hybrid Multimodal Backprop)
                    loss_render = torch.tensor(0.0, device=device)
                    if (global_step + 1) % 50 == 0:
                        try:
                            # 1-step endpoint prediction: x1_hat = x_t + (1 - t) * v_pred
                            x1_hat = x_t[0] + (1.0 - t[0]) * v_pred[0].float()
                            ot_p = torch.softmax(x1_hat[:, :FM_OT_END], dim=-1)
                            exist_p = 1.0 - ot_p[:, EMPTY_IDX]
                            ot_idx = x1_hat[:, :EMPTY_IDX].argmax(dim=-1)
                            raw_ot = torch.tensor([ORGAN_CATEGORIES[min(i.item(), len(ORGAN_CATEGORIES)-1)] for i in ot_idx], device=device).float()

                            part_hat = torch.zeros((x1_hat.shape[0], 16), device=device)
                            part_hat[:, 0] = raw_ot
                            part_hat[:, 1] = exist_p.clamp(0.0, 1.0)
                            part_hat[:, 2:5] = x1_hat[:, FM_BASE_START:FM_BASE_END] / 20.0
                            part_hat[:, 5:11] = x1_hat[:, FM_ROT_START:FM_ROT_END]
                            part_hat[:, 11:14] = F.softplus(x1_hat[:, FM_SCALE_START:FM_SCALE_END]).clamp(min=0.001) / 50.0
                            part_hat[:, 14] = x1_hat[:, FM_CURV_IDX] * 100.0
                            part_hat[:, 15] = x1_hat[:, FM_PHYLLO_IDX] * 180.0

                            mesh_hat = renderer.geo_builder.build_mesh_from_part_tensor(part_hat, device=device, existence_threshold=0.30)
                            if mesh_hat['vertices'].shape[0] > 0:
                                depth_pred = renderer.render_depth(mesh_hat, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
                                loss_render = 0.02 * depth_pred.clamp(min=0.0, max=1.0).mean()
                        except Exception:
                            pass

                    loss = (loss_v + loss_count + loss_render) / accum_steps

                loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            raw_step_loss = (loss_v + loss_count + loss_render).item()
            epoch_loss += raw_step_loss
            epoch_v_loss += loss_v.item()
            epoch_c_loss += loss_count.item()
            num_batches += 1

            # Log to console & W&B every 25 optimizer steps on rank 0
            if is_main_process and ((batch_idx + 1) % (accum_steps * 25) == 0 or (batch_idx + 1) == len(loader)):
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch:02d} [{batch_idx+1:05d}/{len(loader):05d}] | Step Loss: {raw_step_loss:.4f} (v: {loss_v.item():.4f}, count: {loss_count.item():.4f}) | Nodes: {N:4d} | LR: {cur_lr:.4e}", flush=True)

                if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "train/step_loss": raw_step_loss,
                        "train/velocity_loss": loss_v.item(),
                        "train/count_loss": loss_count.item(),
                        "train/learning_rate": cur_lr,
                        "train/max_nodes_in_batch": N,
                    }, step=global_step)

        # Average loss across workers
        loss_tensor = torch.tensor([epoch_loss, epoch_v_loss, epoch_c_loss, float(num_batches)], device=device)
        if is_distributed:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        tot_loss, tot_v, tot_c, tot_batches = loss_tensor.tolist()

        avg_loss = tot_loss / max(tot_batches, 1)
        avg_v_loss = tot_v / max(tot_batches, 1)
        avg_c_loss = tot_c / max(tot_batches, 1)

        if is_main_process:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"\n🌟 Epoch {epoch:02d}/{args.epochs} Complete | Avg Loss: {avg_loss:.4f} (v: {avg_v_loss:.4f}, c: {avg_c_loss:.4f}) | LR: {cur_lr:.6e}\n")

            if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "epoch/loss": avg_loss,
                    "epoch/velocity_loss": avg_v_loss,
                    "epoch/count_loss": avg_c_loss,
                    "epoch/epoch": epoch,
                }, step=global_step)

            # Render 3D debug visualizations
            if epoch % args.eval_every == 0 or epoch == 1:
                print(f"Rendering 3D visual reconstruction debug images for Epoch {epoch}...")
                raw_model = model.module if is_distributed else model
                render_and_log_debug_images(raw_model, eval_samples, renderer, assembler, device, epoch, global_step)

            if avg_loss < best_loss:
                best_loss = avg_loss
                raw_model = model.module if is_distributed else model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "world_size": world_size,
                    "effective_batch_size": effective_batch_size,
                }, save_path)
                if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({"eval/best_loss": best_loss}, step=global_step)
                print(f"  ✓ Saved new best checkpoint -> {save_path} (Loss: {best_loss:.4f})")

        if is_distributed:
            dist.barrier()

    if is_main_process:
        print(f"🎉 Training Complete! Saved best DiT-Large model to {save_path} (Best Loss: {best_loss:.4f})")
        if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
