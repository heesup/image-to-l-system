"""
Distributed Multi-GPU (2x / 4x NVIDIA H100) Training Pipeline for VLM-Scaffold-DiT.
Combines Pretrained DINOv3 Vision Backbone, Macro Botanical Scaffold Prior,
and Cross-Attention Bridge Flow Matching with BFloat16 Mixed Precision & NCCL NVLink.
"""

import os
import sys
import glob
import math
import json
import signal
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
from diffusion_based.models.vlm_scaffold_dit import VLMScaffoldDiTModel
from diffusion_based.models.plant_organ_array import PlantOrganArray, NUM_FEATURES_PART
from diffusion_based.models.part_assembly_to_xml import PartAssemblyToXMLConverter
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.dataset.part_array_dataset import (
    ORGAN_CATEGORIES, EMPTY_IDX, FM_NODE_DIM, FM_OT_END,
    FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END,
    FM_SCALE_START, FM_SCALE_END, FM_CURV_IDX, FM_PHYLLO_IDX,
    decode_fm,
)

ORGAN_LEGEND_MAP = {
    0: ("#8B4513", "Stem/Internode"),
    1: ("#E68026", "Petiole"),
    2: ("#228B22", "Leaf"),
    3: ("#9ACD32", "Peduncle"),
    4: ("#FFD700", "Flower"),
    5: ("#E63333", "Pod/Fruit"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM-Scaffold-DiT on Cowpea 100K with Multi-GPU DDP")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size per GPU")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps per GPU")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate for multi-GPU")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--cache-dir", type=str, default="dataset/helios_data/cowpea_shard")
    parser.add_argument("--data-root", type=str, default="dataset/helios_data/cowpea")
    parser.add_argument("--save-dir", type=str, default="diffusion_based/checkpoints/fm")
    parser.add_argument("--save-name", type=str, default="cowpea_vlm_scaffold_dit_h100_ddp.pt")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cowpea-vlm-scaffold-dit", help="W&B Project name")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B Experiment Group")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B Run Display Name")
    parser.add_argument("--eval-every", type=int, default=2, help="Epoch interval to log 6-column visual evaluation images")
    parser.add_argument("--max-slots", type=int, default=4096, help="Maximum slot capacity")
    parser.add_argument("--embed-dim", type=int, default=768, help="DiT latent embed dimension")
    parser.add_argument("--decoder-layers", type=int, default=12, help="DiT Transformer decoder layers")
    parser.add_argument("--num-heads", type=int, default=12, help="DiT attention heads")
    parser.add_argument("--cond-drop-prob", type=float, default=0.10, help="Classifier-Free Guidance condition dropout probability")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="CFG inference guidance scale")
    parser.add_argument("--render-loss-weight", type=float, default=0.15, help="Differentiable 2D photometric rendering loss weight")
    parser.add_argument("--helios-roundtrip", action="store_true", default=False, help="Enable 17D->XML->Helios round-trip validation column in eval debug images")
    parser.add_argument("--noise-sigma", type=float, default=0.05, help="Scaffold bridge perturbation sigma")
    parser.add_argument("--freeze-vision", action="store_true", default=False, help="Freeze pretrained DINOv3 backbone")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest checkpoint if exists")
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def depth_to_chm_rgb(depth_tensor: torch.Tensor, far_plane: float = 20.0) -> Tuple[np.ndarray, float]:
    """Convert depth buffer (distance to camera) to Canopy Height Model (CHM) colormap.
    Taller plant parts are closer to camera (smaller depth) -> mapped to brighter yellow/orange in plasma.
    Ground/empty space -> pure pitch black background (0, 0, 0).
    """
    d = depth_tensor.detach().cpu().numpy().squeeze()
    fg_mask = (d < (far_plane - 0.5)) & (d > 0.01)
    if not np.any(fg_mask):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.float32), 0.0

    d_min = d[fg_mask].min()
    d_max = d[fg_mask].max()
    canopy_h_cm = (d_max - d_min) * 100.0  # height in cm

    d_norm = np.zeros_like(d)
    if d_max > d_min:
        d_norm[fg_mask] = (d_max - d[fg_mask]) / (d_max - d_min)  # invert: closer (taller) = 1.0 (brighter)
    else:
        d_norm[fg_mask] = 1.0

    cmap = plt.get_cmap("plasma")
    rgb = cmap(d_norm)[:, :, :3].astype(np.float32)
    rgb[~fg_mask] = 0.0  # pure black background
    return rgb, canopy_h_cm


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
    raw_model: VLMScaffoldDiTModel,
    eval_samples: List[Dict[str, Any]],
    renderer: HeliosPyTorchRenderer,
    assembler: PartAssemblyToXMLConverter,
    device: torch.device,
    epoch: int,
    global_step: int,
    helios_roundtrip: bool = False,
):
    if not eval_samples:
        return
    raw_model.eval()
    n_cols = 7 if helios_roundtrip else 6
    fig, axes = plt.subplots(len(eval_samples), n_cols, figsize=(28 if helios_roundtrip else 24, 4.2 * len(eval_samples)))
    if len(eval_samples) == 1:
        axes = np.expand_dims(axes, 0)
    fig.patch.set_facecolor("#080C14")

    for row_idx, sc in enumerate(eval_samples):
        # 1. Ground Truth 3D Mesh & Differentiable Renderings (Col 1, 2, 3)
        arr_gt = PlantOrganArray.from_xml_file(sc["xml"])
        mesh_gt = renderer.geo_builder.build_mesh_from_part_tensor(arr_gt.to_part_tensor(device=device), device=device)
        rgb_gt = renderer.render_mesh(
            mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
            background="white", focus_plant=True
        )
        rgb_gt_np = rgb_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        depth_gt = renderer.render_depth(
            mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
            focus_plant=True
        )
        depth_gt_rgb, h_gt_cm = depth_to_chm_rgb(depth_gt)

        seg_gt = renderer.render_organ_segmentation(
            mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
            focus_plant=True
        )
        seg_gt_np = seg_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        # 2. Input Image & Model Bridge Inference
        pil_img = Image.open(sc["img"]).convert("RGB").resize((512, 512))
        img_np = np.array(pil_img) / 255.0
        img_t = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            sample_out = raw_model.sample_plant(img_t, num_steps=15, guidance_scale=2.0, device=device)
            x_gen = sample_out["x_gen"].squeeze(0)
            pred_dap = sample_out["pred_dap"].squeeze().item() if sample_out["pred_dap"].numel() == 1 else sample_out["pred_dap"][0].item()
            pred_h = sample_out["pred_height"].squeeze().item() if sample_out["pred_height"].numel() == 1 else sample_out["pred_height"][0].item()

        ot_probs = torch.softmax(x_gen[:, :FM_OT_END], dim=-1)
        exist_prob = 1.0 - ot_probs[:, EMPTY_IDX]
        active_n = int((exist_prob >= 0.30).sum().item())

        # 3. Generated 3D Mesh & Differentiable Renderings (Col 4, 5, 6)
        try:
            mesh_gen = renderer.geo_builder.build_mesh_from_part_tensor(x_gen, device=device, existence_threshold=0.30)
            rgb_gen = renderer.render_mesh(
                mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                background="white", focus_plant=True
            )
            rgb_gen_np = rgb_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

            depth_gen = renderer.render_depth(
                mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                focus_plant=True
            )
            depth_gen_rgb, h_gen_cm = depth_to_chm_rgb(depth_gen)

            seg_gen = renderer.render_organ_segmentation(
                mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                focus_plant=True
            )
            seg_gen_np = seg_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            vert_count = mesh_gen["vertices"].shape[0]
        except Exception:
            rgb_gen_np = np.zeros((512, 512, 3), dtype=np.float32)
            depth_gen_rgb = np.zeros((512, 512, 3), dtype=np.float32)
            seg_gen_np = np.zeros((512, 512, 3), dtype=np.float32)
            h_gen_cm = 0.0
            vert_count = 0

        # Plot 6 columns
        # Col 1: PyTorch Differentiable RGB Input (GT)
        axes[row_idx, 0].imshow(rgb_gt_np)
        axes[row_idx, 0].set_title(f"{sc['name']}\nDiff RGB Input ({arr_gt.num_nodes} organs)", color="#4ADE80", fontsize=10, fontweight="bold")
        axes[row_idx, 0].text(0.03, 0.03, f"N={arr_gt.num_nodes}", transform=axes[row_idx, 0].transAxes, fontsize=8, color='white', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        axes[row_idx, 0].axis("off")

        # Col 2: PyTorch Differentiable Canopy Height (CHM) Input (GT)
        axes[row_idx, 1].imshow(depth_gt_rgb)
        axes[row_idx, 1].set_title("Canopy Height (CHM)\n(taller = brighter)", color="#22D3EE", fontsize=10, fontweight="bold")
        axes[row_idx, 1].text(0.03, 0.03, f"Height: 0–{h_gt_cm:.1f} cm", transform=axes[row_idx, 1].transAxes, fontsize=8, color='#22D3EE', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        axes[row_idx, 1].axis("off")

        # Col 3: PyTorch Differentiable Organ Segmentation Mask Input (GT)
        axes[row_idx, 2].imshow(seg_gt_np)
        axes[row_idx, 2].set_title("Diff Organ Seg Input", color="#A78BFA", fontsize=10, fontweight="bold")
        patches = [mpatches.Patch(color=c, label=l) for ot, (c, l) in ORGAN_LEGEND_MAP.items()]
        axes[row_idx, 2].legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.85, facecolor='#0D1117', labelcolor='white', edgecolor='#334155', ncol=1)
        axes[row_idx, 2].axis("off")

        # Col 4: Generated PyTorch Differentiable RGB
        axes[row_idx, 3].imshow(rgb_gen_np)
        axes[row_idx, 3].set_title(f"Gen Diff RGB\n(DAP: {pred_dap:.1f} | H: {pred_h*100:.1f}cm)", color="#60A5FA", fontsize=10, fontweight="bold")
        axes[row_idx, 3].text(0.03, 0.03, f"N={active_n} ({vert_count}v)", transform=axes[row_idx, 3].transAxes, fontsize=8, color='white', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        axes[row_idx, 3].axis("off")

        # Col 5: Generated PyTorch Differentiable Canopy Height (CHM)
        axes[row_idx, 4].imshow(depth_gen_rgb)
        axes[row_idx, 4].set_title("Gen Canopy Height\n(taller = brighter)", color="#F472B6", fontsize=10, fontweight="bold")
        axes[row_idx, 4].text(0.03, 0.03, f"Height: 0–{h_gen_cm:.1f} cm", transform=axes[row_idx, 4].transAxes, fontsize=8, color='#F472B6', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        axes[row_idx, 4].axis("off")

        # Col 6: Generated PyTorch Differentiable Organ Segmentation Mask
        axes[row_idx, 5].imshow(seg_gen_np)
        axes[row_idx, 5].set_title("Gen Diff Organ Seg", color="#FB923C", fontsize=10, fontweight="bold")
        axes[row_idx, 5].legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.85, facecolor='#0D1117', labelcolor='white', edgecolor='#334155', ncol=1)
        axes[row_idx, 5].axis("off")

        # Col 7: 17D -> XML -> Helios Round-trip (re-render generated 17D via Helios XML)
        if helios_roundtrip:
            try:
                # Decode 26D FM model output -> canonical 17D part tensor
                part_gen_17d = decode_fm(x_gen)
                # Serialize 17D -> Helios XML
                xml_rt = assembler.convert_to_xml_string(part_gen_17d)
                # Re-parse XML -> 17D to verify round-trip self-consistency
                arr_rt = PlantOrganArray.from_xml_string(xml_rt)
                part_rt = arr_rt.to_part_tensor(device=device)
                rt_max_diff = float(torch.abs(part_gen_17d - part_rt).max().item())
                # Render the round-tripped 17D mesh (same camera as GT)
                mesh_rt = renderer.geo_builder.build_mesh_from_part_tensor(part_rt, device=device, existence_threshold=0.30)
                rgb_rt = renderer.render_mesh(
                    mesh_rt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                    background="white", focus_plant=True
                )
                rgb_rt_np = rgb_rt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
                rt_ok = True
            except Exception:
                rgb_rt_np = np.zeros((512, 512, 3), dtype=np.float32)
                rt_max_diff = float("nan")
                rt_ok = False
            axes[row_idx, 6].imshow(rgb_rt_np)
            axes[row_idx, 6].set_title("17D→XML→Helios RT\n(round-trip)", color="#FDE047", fontsize=10, fontweight="bold")
            axes[row_idx, 6].text(0.03, 0.03, f"Δ={rt_max_diff:.4f}" if rt_ok else "FAIL", transform=axes[row_idx, 6].transAxes, fontsize=8, color='white', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            axes[row_idx, 6].axis("off")

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.2)

    plt.tight_layout()
    # Save to local repository assets for offline/local inspection
    save_fig_path = os.path.join(repo_root, "docs", "results", "assets", f"eval_vlm_scaffold_epoch_{epoch:03d}.png")
    latest_fig_path = os.path.join(repo_root, "docs", "results", "assets", "fig_vlm_scaffold_latest_eval.png")
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(latest_fig_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "eval/visual_reconstructions": wandb.Image(fig),
        }, step=global_step)
    plt.close()
    print(f"  ✓ Saved local evaluation figures to: {latest_fig_path} & {save_fig_path}")


def main():
    args = parse_args()

    # Initialize Distributed Processing Group
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = (local_rank == 0)
    effective_batch_size = args.batch_size * args.grad_accum_steps * world_size

    if is_main_process:
        print("\n" + "="*80)
        print("🚀 INITIALIZING VLM-SCAFFOLD-DiT DDP TRAINING PIPELINE (H100 NVLINK)")
        print("="*80)
        print(f"  • Date & Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  • World Size:            {world_size} GPUs")
        print(f"  • Micro-Batch Size:      {args.batch_size} per GPU")
        print(f"  • Grad Accumulation:     {args.grad_accum_steps} steps")
        print(f"  • Global Batch Size:     {effective_batch_size}")
        print(f"  • Learning Rate:         {args.lr:.2e}")
        print(f"  • Max Slot Capacity:     {args.max_slots}")
        print(f"  • DiT Embed Dim:         {args.embed_dim}")
        print(f"  • Vision Backbone:       DINOv3 ViT-B/14 (Frozen: {args.freeze_vision})")
        print(f"  • Shard Cache Directory: {args.cache_dir}")
        print(f"  • Checkpoint Directory:  {args.save_dir}")
        print("="*80 + "\n")

        os.makedirs(args.save_dir, exist_ok=True)
        if args.use_wandb and WANDB_AVAILABLE:
            run_name = args.wandb_name or f"vlm_scaffold_dit_2xh100_b{effective_batch_size}_{datetime.now().strftime('%m%d_%H%M')}"
            wandb.init(
                project=args.wandb_project,
                group=args.wandb_group or "vlm-scaffold-dit-scale",
                name=run_name,
                config=vars(args),
            )

    # 1. Dataset & DataLoader Initialization
    dataset = CowpeaShardDataset(
        cache_dir=args.cache_dir,
        fallback_cache_dir="dataset/cache"
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        drop_last=True,
    ) if is_distributed else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=cowpea_collate_fn,
        drop_last=True,
    )

    if is_main_process:
        print(f"  ✓ Loaded {len(dataset):,} plant records ({len(loader)} batches/epoch per GPU)")

    # 2. Model Instantiation & DDP Wrapping
    model = VLMScaffoldDiTModel(
        max_slots=args.max_slots,
        embed_dim=args.embed_dim,
        in_channels=4,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        pretrained=True,
        freeze_vision_backbone=args.freeze_vision,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if is_main_process:
        print(f"  ✓ Model Size: {total_params / 1e6:.1f}M params ({trainable_params / 1e6:.1f}M trainable)")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 3. Optimizer & Learning Rate Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.95),
    )

    total_steps = (len(loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = (len(loader) // args.grad_accum_steps) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Visual rendering setup
    renderer = HeliosPyTorchRenderer(image_size=512).to(device)
    assembler = PartAssemblyToXMLConverter()
    eval_samples = load_fixed_eval_samples(args.data_root) if is_main_process else []

    start_epoch = 1
    best_loss = float("inf")
    save_path = os.path.join(args.save_dir, args.save_name)
    accum_steps = args.grad_accum_steps
    global_step = 0

    if args.resume and os.path.exists(save_path):
        if is_main_process:
            print(f"🔄 Resuming checkpoint from: {save_path}")
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        raw_model = model.module if is_distributed else model
        if "model_state_dict" in checkpoint:
            raw_model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            raw_model.load_state_dict(checkpoint["model"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "loss" in checkpoint:
            best_loss = checkpoint["loss"]
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        if is_main_process:
            print(f"  ✓ Resumed successfully from Epoch {start_epoch-1} (Best Loss: {best_loss:.4f}, Global Step: {global_step})")

    for epoch in range(start_epoch, args.epochs + 1):
        if is_distributed:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_v_loss = 0.0
        epoch_m_loss = 0.0
        epoch_r_loss = 0.0
        epoch_r_rgb_loss = 0.0
        epoch_r_dep_loss = 0.0
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

            # Macro trait ground truths
            raw_model = model.module if is_distributed else model

            # DDP Gradient Accumulation with no_sync()
            is_accumulating = (batch_idx + 1) % accum_steps != 0 and (batch_idx + 1) != len(loader)
            sync_context = model.no_sync() if is_accumulating and is_distributed else torch.amp.autocast('cuda', enabled=False)

            with sync_context:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # 1. Sample standard Gaussian Prior x_0 ~ N(0, I)
                    x0 = torch.randn_like(x1)

                    # 2. Sample continuous flow matching timestep & displacement
                    t = torch.rand(B, device=device)
                    t_expand = t.view(B, 1, 1)
                    x_t = (1.0 - t_expand) * x0 + t_expand * x1
                    u_t = x1 - x0

                    # 3. Pure Single-Stage MM-DiT Forward Pass with CFG
                    out = model(
                        x_t=x_t,
                        t=t,
                        img=images,
                        key_padding_mask=k_mask,
                        cond_drop_prob=args.cond_drop_prob,
                    )
                    v_pred = out["pred_velocity"]

                    # Active organ weighted velocity loss
                    active_weights = torch.where(exist_mask.unsqueeze(-1) > 0.5, 1.0, 0.15)
                    loss_v = (active_weights * (v_pred.float() - u_t) ** 2).mean()

                    # Macro regression loss (DAP, Count, Height, Radius)
                    loss_dap = F.smooth_l1_loss(out["pred_dap"] / 100.0, daps / 100.0)
                    loss_count = F.smooth_l1_loss(out["pred_active_count"] / 100.0, gt_counts / 100.0)
                    loss_h = F.smooth_l1_loss(out["pred_height"], torch.clamp(daps / 100.0 * 0.7 + 0.1, 0.08, 0.85))
                    loss_macro = 0.5 * loss_dap + 0.3 * loss_count + 0.2 * loss_h

                    # 4. Differentiable Photometric & Geometric 4-Channel Rendering Supervision on Endpoint x_1_hat
                    loss_render = torch.tensor(0.0, device=device)
                    loss_render_rgb = torch.tensor(0.0, device=device)
                    loss_render_depth = torch.tensor(0.0, device=device)
                    if args.render_loss_weight > 0.0 and renderer is not None:
                        # Select sample for differentiable 4D rendering loss
                        r_idx = torch.randint(0, B, (1,)).item()
                        x1_hat_sample = out["x_1_hat"][r_idx]
                        try:
                            # 1. Render Predicted 3D Mesh (4-Channel: RGB 3ch + Canopy Depth 1ch)
                            pred_mesh = renderer.geo_builder.build_mesh_from_part_tensor(
                                x1_hat_sample, device=device, existence_threshold=0.25
                            )
                            if pred_mesh['vertices'].shape[0] > 0 and pred_mesh['faces'].shape[0] > 0:
                                with torch.amp.autocast('cuda', enabled=False):
                                    pred_rgbd = renderer.render_mesh(
                                        pred_mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                                        background="white", focus_plant=True, include_depth=True
                                    )
                                    # Target 4D from Dataset (RGB in [-1, 1], Depth in meters)
                                    target_4d = images[r_idx].float()
                                    if pred_rgbd.shape[-1] != target_4d.shape[-1]:
                                        target_4d = F.interpolate(
                                            target_4d.unsqueeze(0), size=pred_rgbd.shape[-2:], mode='bilinear', align_corners=False
                                        ).squeeze(0)
                                    
                                    target_rgb = target_4d[:3] * 0.5 + 0.5  # de-normalize to [0, 1]
                                    target_depth = target_4d[3:]
                                    
                                    loss_render_rgb = F.l1_loss(pred_rgbd[:3], target_rgb)
                                    loss_render_depth = F.l1_loss(pred_rgbd[3:], target_depth)
                                    loss_render = loss_render_rgb + 0.5 * loss_render_depth
                        except Exception:
                            pass

                    # Pure Single-Stage Joint Loss
                    loss = (loss_v + 0.5 * loss_macro + args.render_loss_weight * loss_render) / accum_steps

                loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            raw_step_loss = (loss_v + 0.5 * loss_macro + args.render_loss_weight * loss_render).item()
            epoch_loss += raw_step_loss
            epoch_v_loss += loss_v.item()
            epoch_m_loss += loss_macro.item()
            epoch_r_loss += loss_render.item()
            epoch_r_rgb_loss += loss_render_rgb.item()
            epoch_r_dep_loss += loss_render_depth.item()
            num_batches += 1

            # Log to console & W&B every 25 optimizer steps on rank 0
            if is_main_process and ((batch_idx + 1) % (accum_steps * 25) == 0 or (batch_idx + 1) == len(loader)):
                cur_lr = scheduler.get_last_lr()[0]
                dap_err = abs(out['pred_dap'][0].item() - daps[0].item())
                cnt_err = abs(out['pred_active_count'][0].item() - gt_counts[0].item())
                h_pred = out['pred_height'][0].item() * 100.0  # cm
                vram_gb = torch.cuda.memory_allocated(device) / 1e9
                vram_max_gb = torch.cuda.max_memory_allocated(device) / 1e9
                pct = (batch_idx + 1) / len(loader) * 100.0
                print(
                    f"Epoch {epoch:02d} [{batch_idx+1:05d}/{len(loader):05d}] ({pct:4.1f}%) | "
                    f"Step Loss: {raw_step_loss:.4f} (v: {loss_v.item():.4f}, macro: {loss_macro.item():.4f}, "
                    f"r_rgb: {loss_render_rgb.item():.4f}, r_dep: {loss_render_depth.item():.4f}) | "
                    f"DAP_err: {dap_err:4.1f}d | Cnt_err: {cnt_err:3.0f} | H_pred: {h_pred:4.1f}cm | "
                    f"Nodes: {N:4d} | VRAM: {vram_gb:.1f}/{vram_max_gb:.1f}GB | LR: {cur_lr:.4e}",
                    flush=True
                )

                if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "train/step_loss": raw_step_loss,
                        "train/velocity_loss": loss_v.item(),
                        "train/macro_loss": loss_macro.item(),
                        "train/render_loss": loss_render.item(),
                        "train/render_loss_rgb": loss_render_rgb.item(),
                        "train/render_loss_depth": loss_render_depth.item(),
                        "train/dap_loss": loss_dap.item(),
                        "train/count_loss": loss_count.item(),
                        "train/dap_error_days": dap_err,
                        "train/count_error": cnt_err,
                        "train/learning_rate": cur_lr,
                        "train/max_nodes_in_batch": N,
                        "train/vram_allocated_gb": vram_gb,
                        "train/vram_max_gb": vram_max_gb,
                    }, step=global_step)

        # Average loss across workers
        loss_tensor = torch.tensor([epoch_loss, epoch_v_loss, epoch_m_loss, epoch_r_loss, epoch_r_rgb_loss, epoch_r_dep_loss, float(num_batches)], device=device)
        if is_distributed:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        tot_loss, tot_v, tot_m, tot_r, tot_r_rgb, tot_r_dep, tot_batches = loss_tensor.tolist()

        avg_loss = tot_loss / max(tot_batches, 1)
        avg_v_loss = tot_v / max(tot_batches, 1)
        avg_m_loss = tot_m / max(tot_batches, 1)
        avg_r_loss = tot_r / max(tot_batches, 1)
        avg_r_rgb_loss = tot_r_rgb / max(tot_batches, 1)
        avg_r_dep_loss = tot_r_dep / max(tot_batches, 1)

        if is_main_process:
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"\n🌟 Epoch {epoch:02d}/{args.epochs} Complete | Avg Loss: {avg_loss:.4f} "
                f"(v: {avg_v_loss:.4f}, macro: {avg_m_loss:.4f}, r_rgb: {avg_r_rgb_loss:.4f}, r_dep: {avg_r_dep_loss:.4f}) | "
                f"LR: {cur_lr:.6e}\n",
                flush=True
            )

            if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "epoch/loss": avg_loss,
                    "epoch/velocity_loss": avg_v_loss,
                    "epoch/macro_loss": avg_m_loss,
                    "epoch/render_loss": avg_r_loss,
                    "epoch/render_loss_rgb": avg_r_rgb_loss,
                    "epoch/render_loss_depth": avg_r_dep_loss,
                    "epoch/epoch": epoch,
                }, step=global_step)

            # Render 6-column 3D debug visualizations
            if epoch % args.eval_every == 0 or epoch == 1:
                print(f"Rendering 6-column 3D visual reconstruction debug images for Epoch {epoch}...")
                render_and_log_debug_images(raw_model, eval_samples, renderer, assembler, device, epoch, global_step, helios_roundtrip=args.helios_roundtrip)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
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
        print(f"🎉 Training Complete! Saved best VLM-Scaffold-DiT model to {save_path} (Best Loss: {best_loss:.4f})")
        if args.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
