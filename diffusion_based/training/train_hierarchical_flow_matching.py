"""
Distributed Training Script for Hierarchical Matryoshka Botanical Flow Matching.

Jointly trains:
  Stage 1: Coarse Skeletal Transformer (3D anchors, existence logits, aux DAP)
  Stage 2: Fine Botanical Flow Matching Decoder (local organ vector field, discrete classes)
"""

import os
import sys
import argparse
import math
import time
from typing import Dict, List, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate

import wandb

from diffusion_based.dataset.part_array_dataset import (
    PartArrayDataset,
    EMPTY_IDX,
    FM_OT_END,
    FM_BASE_START,
    GEOM_NODE_DIM,
    NUM_ORGAN_TYPES,
    BASE_SCALE,
    SCALE_SCALE,
    CURV_SCALE,
)
from diffusion_based.models.plant_organ_array import (
    ORGAN_SHOOT_META,
)
from diffusion_based.training.flow_matching import FlowMatchingScheduler
from diffusion_based.models.hierarchical_part_flow_matching import (
    HierarchicalPartFlowMatchingModel,
    compute_matryoshka_slice,
)
from diffusion_based.training.hierarchical_hungarian_matcher import (
    HierarchicalBotanicalMatcher,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.organ_latent_vae import OrganLatentVAE
import importlib
import diffusion_based.eval.eval_hierarchical_self_consistency as ehsc


def forward_backward_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: FlowMatchingScheduler,
    matcher: HierarchicalBotanicalMatcher,
    device: torch.device,
    slots_per_anchor: int,
    renderer: Optional[HeliosPyTorchRenderer] = None,
    vae: Optional[nn.Module] = None,
    depth_loss_weight: float = 0.5,
    color_loss_weight: float = 0.2,
    silhouette_loss_weight: float = 2.0,
    render_ratio: float = 1.0,
    render_sub_batch: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    images = batch["image"].to(device)
    nodes = batch["nodes"].to(device)  # (B, N_max, 26)
    existence_mask = batch["existence_mask"].to(device)
    daps = batch.get("dap", None)
    if daps is not None:
        daps = daps.to(device)

    B = images.shape[0]

    # Extract GT labels and sub-slice for Matryoshka capacity
    type_labels = nodes[:, :, :FM_OT_END].argmax(dim=-1)  # (B, N_max)
    raw_model = model.module if hasattr(model, "module") else model
    active_k = compute_matryoshka_slice(daps, max_anchors=raw_model.max_anchors)
    active_fine = active_k * slots_per_anchor

    nodes_sub = nodes[:, :active_fine]
    type_labels_sub = type_labels[:, :active_fine]
    existence_mask_sub = existence_mask[:, :active_fine]
    act_mask_sub = (type_labels_sub > ORGAN_SHOOT_META) & (existence_mask_sub > 0.5)

    node_dim = raw_model.node_dim if hasattr(raw_model, "node_dim") else 16

    # For 16D Latent Mode: compute target latents z_1 via frozen VAE encoder
    if vae is not None:
        tgt_z1 = torch.zeros(B, active_fine, node_dim, device=device)
        if act_mask_sub.any():
            with torch.no_grad():
                flat_act_nodes = nodes_sub[act_mask_sub]
                tgt_z1[act_mask_sub] = vae.encode(flat_act_nodes)[0]
    else:
        tgt_z1 = nodes_sub[:, :, FM_BASE_START:]

    # Flow Matching Prior in 16D: Standard Gaussian N(0, I)
    z_0 = torch.randn(B, active_fine, node_dim, device=device)
    t = scheduler.sample_time(B, device)
    z_t = scheduler.sample_xt(z_0, tgt_z1, t)

    optimizer.zero_grad()

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            noisy_fine_nodes=z_t,
            timesteps=t,
            images=images,
            daps=daps,
        )

    pred_anchor_pos = outputs["pred_anchor_pos"].float()              # (B, K, 3)
    pred_anchor_logits = outputs["pred_anchor_logits"].float()          # (B, K, 1)
    pred_velocity = outputs["pred_velocity"].float()                  # (B, active_fine, node_dim)
    pred_fine_exist_logits = outputs["pred_fine_exist_logits"].float()  # (B, active_fine, 1)
    pred_dap = outputs["pred_dap"].float()
    pred_num_phytomers = outputs.get("pred_num_phytomers")
    soft_margin_weights = outputs.get("soft_margin_weights")

    # 1-Step Analytical Clean Prediction in 16D latent space: z_hat_1 = z_t + (1 - t) * v_pred
    t_b = t.view(B, 1, 1)
    clean_z1 = z_t + (1.0 - t_b) * pred_velocity

    # Prepare GT targets for hierarchical matching
    tgt_geoms_list = []
    tgt_labels_list = []
    tgt_positions_list = []
    for b in range(B):
        act_b = act_mask_sub[b]
        tgt_geoms_list.append(tgt_z1[b, act_b])
        tgt_labels_list.append(type_labels_sub[b, act_b])
        tgt_positions_list.append(nodes_sub[b, act_b, FM_BASE_START : FM_BASE_START + 3] / BASE_SCALE)

    # Hierarchical Bipartite Matching in 16D Latent Space with Soft Margin Modulation
    matches = matcher(
        pred_anchor_pos=pred_anchor_pos,
        pred_anchor_logits=pred_anchor_logits,
        pred_fine_geom=clean_z1,
        pred_fine_logits=pred_fine_exist_logits,
        tgt_geoms=tgt_geoms_list,
        tgt_labels=tgt_labels_list,
        tgt_positions=tgt_positions_list,
        soft_margin_weights=soft_margin_weights,
    )

    # Extract GT phytomer counts from matcher clusters
    gt_phy_counts = torch.tensor(
        [m["num_gt_phytomers"] for m in matches],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)  # (B, 1)

    # Loss Computation
    loss_anchor_pos_acc = torch.tensor(0.0, device=device)
    loss_anchor_exist_acc = torch.tensor(0.0, device=device)
    loss_fine_vel_acc = torch.tensor(0.0, device=device)
    loss_fine_exist_acc = torch.tensor(0.0, device=device)

    total_matched_anc = 0
    total_matched_fine = 0
    correct_cls = 0

    for b in range(B):
        m_b = matches[b]
        anc_src = m_b["anchor_src_idx"]
        anc_tgt = m_b["anchor_tgt_idx"]
        fine_src = m_b["fine_src_idx"]
        fine_tgt = m_b["fine_tgt_idx"]

        num_anc_m = len(anc_src)
        total_matched_anc += num_anc_m
        num_fine_m = len(fine_src)
        total_matched_fine += num_fine_m

        # Stage 1 Anchor Losses
        if num_anc_m > 0:
            exist_targets = torch.zeros(active_k, 1, device=device)
            exist_targets[anc_src] = 1.0
            pos_weight_anc = torch.tensor([8.0], device=device)
            loss_anchor_exist_acc += F.binary_cross_entropy_with_logits(
                pred_anchor_logits[b], exist_targets, pos_weight=pos_weight_anc
            )

            if "anchor_tgt_pos" in m_b and len(m_b["anchor_tgt_pos"]) > 0:
                p_anc = pred_anchor_pos[b, anc_src]
                t_anc = m_b["anchor_tgt_pos"]
                loss_anchor_pos_acc += F.smooth_l1_loss(p_anc, t_anc, reduction="sum")

        # Stage 2 Fine Losses
        exist_targets_fine = torch.zeros(active_fine, 1, device=device)
        if num_fine_m > 0:
            exist_targets_fine[fine_src] = 1.0

            p_v = pred_velocity[b, fine_src]
            t_geom = tgt_geoms_list[b][fine_tgt]
            t_v = t_geom - z_0[b, fine_src]
            loss_fine_vel_acc += F.mse_loss(p_v, t_v, reduction="sum") / float(node_dim)

            # VAE classification accuracy check on matched slots
            if vae is not None:
                with torch.no_grad():
                    pred_cls_m = vae.decode(clean_z1[b, fine_src])["cls_logits"].argmax(dim=-1)
                    t_cls = tgt_labels_list[b][fine_tgt]
                    correct_cls += (pred_cls_m == t_cls).sum().item()

        # Idle slot velocity damping: gently regularize velocity of unmatched slots towards 0
        # so unassigned reproductive / spare slots do not drift into empty space
        all_fine_idx = torch.arange(active_fine, device=device)
        if num_fine_m > 0:
            idle_mask = torch.ones(active_fine, dtype=torch.bool, device=device)
            idle_mask[fine_src] = False
            idle_src = all_fine_idx[idle_mask]
        else:
            idle_src = all_fine_idx
        if len(idle_src) > 0:
            p_v_idle = pred_velocity[b, idle_src]
            loss_fine_vel_acc += 0.05 * (p_v_idle ** 2).sum() / float(node_dim)

        # Fine slot existence loss across all fine slots (BCE with pos_weight=12.0)
        pos_weight_fine = torch.tensor([12.0], device=device)
        loss_fine_exist_acc += F.binary_cross_entropy_with_logits(
            pred_fine_exist_logits[b], exist_targets_fine, pos_weight=pos_weight_fine
        )

    norm_m = max(total_matched_fine, 1)
    norm_anc = max(total_matched_anc, 1)

    loss_anchor_pos = loss_anchor_pos_acc / norm_anc
    loss_anchor_exist = loss_anchor_exist_acc / max(B, 1)
    loss_fine_vel = loss_fine_vel_acc / norm_m
    loss_fine_exist = loss_fine_exist_acc / max(B, 1)

    # Stage 1: Macro Losses (Phytomer Count & Plant Age DAP)
    loss_phy_count = torch.tensor(0.0, device=device)
    if pred_num_phytomers is not None:
        loss_phy_count = F.smooth_l1_loss(pred_num_phytomers.float(), gt_phy_counts)

    loss_dap = torch.tensor(0.0, device=device)
    if daps is not None and pred_dap is not None:
        loss_dap = F.smooth_l1_loss(pred_dap.squeeze(-1), daps.float()) * 0.05

    # In-Loop Differentiable Optical Grounding (1-Step Clean Prediction -> 4-Scale Multi-Scale Pyramid Depth + Silhouette Dice + Cosine Color)
    loss_depth = torch.tensor(0.0, device=device)
    loss_cos = torch.tensor(0.0, device=device)
    loss_dice = torch.tensor(0.0, device=device)

    if renderer is not None and (depth_loss_weight > 0.0 or color_loss_weight > 0.0 or silhouette_loss_weight > 0.0):
        if render_sub_batch is not None and render_sub_batch > 0:
            n_render = min(B, render_sub_batch)
        else:
            n_render = max(1, int(round(B * max(0.0, min(1.0, render_ratio))))) if render_ratio > 0.0 else 0

        if n_render > 0:
            if n_render < B:
                render_indices = torch.randperm(B, device=device)[:n_render]
            else:
                render_indices = torch.arange(B, device=device)

            loss_depth_acc = torch.tensor(0.0, device=device)
            loss_cos_acc = torch.tensor(0.0, device=device)
            loss_dice_acc = torch.tensor(0.0, device=device)

            pyramid_scales = [1.0, 2.0, 4.0, 8.0]
            num_scales = len(pyramid_scales)

            for b_idx in range(n_render):
                real_b = render_indices[b_idx].item()
                if vae is not None:
                    # Differentiably decode 16D latents into 14D part tensor via frozen VAE!
                    part_14d, probs = vae.decode_to_part_tensor(clean_z1[real_b])
                    exist_b = torch.sigmoid(pred_fine_exist_logits[real_b]).squeeze(-1)
                else:
                    base_xyz = clean_z1[real_b, :, 0:3] / BASE_SCALE
                    rot6d = clean_z1[real_b, :, 3:9]
                    scale_xyz = clean_z1[real_b, :, 9:12].clamp(min=1e-4) / SCALE_SCALE
                    curv_val = clean_z1[real_b, :, 12:13] / CURV_SCALE
                    cls_val = torch.ones((active_fine, 1), device=device) * 5.0
                    exist_b = torch.sigmoid(pred_fine_exist_logits[real_b]).squeeze(-1)
                    part_14d = torch.cat([cls_val, base_xyz, rot6d, scale_xyz, curv_val], dim=-1)
                    probs = None

                # Differentiable mesh assembly with organ_probs attached for autograd backprop
                mesh_dict = renderer.geo_builder.build_mesh_from_part_tensor(
                    part_14d, existence=exist_b, organ_probs=probs, device=device
                )

                # Multi-scale pyramid rendering (differentiable across all 4 zoom levels: 1.0x, 2.0x, 4.0x, 8.0x)
                pred_pyramid = renderer.render_multiscale_pyramid(
                    mesh_dict,
                    scales=pyramid_scales,
                    azimuth_deg=0.0,
                    elevation_deg=90.0,
                    camera_height=5.0,
                    include_depth=True,
                    differentiable=True,
                    image_size=128,
                    reference_window_size=1.2,
                )

                sample_loss_depth = torch.tensor(0.0, device=device)
                sample_loss_cos = torch.tensor(0.0, device=device)
                sample_loss_dice = torch.tensor(0.0, device=device)

                for k, s in enumerate(pyramid_scales):
                    gt_rgb = images[real_b, 4 * k : 4 * k + 3]
                    gt_depth = images[real_b, 4 * k + 3]

                    pred_rgbd_s = pred_pyramid[float(s)]
                    pred_rgb_s = pred_rgbd_s[:3]
                    pred_depth_s = pred_rgbd_s[3]

                    # 1. Masked Canopy CHM Depth Loss (Evaluated strictly on plant footprint)
                    canopy_mask = (gt_depth > 0.005) | (pred_depth_s > 0.005)
                    if canopy_mask.sum() > 0:
                        loss_d_s = F.smooth_l1_loss(pred_depth_s[canopy_mask], gt_depth[canopy_mask], beta=0.02)
                    else:
                        loss_d_s = F.smooth_l1_loss(pred_depth_s, gt_depth, beta=0.02)
                    sample_loss_depth = sample_loss_depth + loss_d_s

                    # 2. Pixel-Wise Cosine Similarity Color Loss (Lighting & Shadow Invariant)
                    eps = 1e-6
                    dot_product = (pred_rgb_s * gt_rgb).sum(dim=0)
                    norm_pred = torch.sqrt((pred_rgb_s ** 2).sum(dim=0) + eps)
                    norm_gt = torch.sqrt((gt_rgb ** 2).sum(dim=0) + eps)
                    cos_sim = (dot_product / (norm_pred * norm_gt)).clamp(-1.0, 1.0)

                    if canopy_mask.sum() > 0:
                        loss_c_s = (1.0 - cos_sim[canopy_mask]).mean()
                    else:
                        loss_c_s = (1.0 - cos_sim).mean()
                    sample_loss_cos = sample_loss_cos + loss_c_s

                    # 3. Top-View Silhouette Soft Dice Loss (Forces branch lobes and gap accuracy)
                    pred_mask_soft = torch.sigmoid((pred_depth_s - 0.005) * 100.0)
                    gt_mask = (gt_depth > 0.005).float()
                    intersection = (pred_mask_soft * gt_mask).sum()
                    denominator = pred_mask_soft.sum() + gt_mask.sum()
                    loss_dice_s = 1.0 - (2.0 * intersection + 1e-4) / (denominator + 1e-4)
                    sample_loss_dice = sample_loss_dice + loss_dice_s

                loss_depth_acc = loss_depth_acc + (sample_loss_depth / num_scales)
                loss_cos_acc = loss_cos_acc + (sample_loss_cos / num_scales)
                loss_dice_acc = loss_dice_acc + (sample_loss_dice / num_scales)

            loss_depth = loss_depth_acc / max(n_render, 1)
            loss_cos = loss_cos_acc / max(n_render, 1)
            loss_dice = loss_dice_acc / max(n_render, 1)

    # Composite Loss (Macro Prior + 3D Node Scaffold + Intra-Phytomer Flow Matching + Photometric)
    loss = (
        4.0 * loss_anchor_pos
        + 1.0 * loss_anchor_exist
        + 2.0 * loss_fine_vel
        + 1.0 * loss_fine_exist
        + 0.5 * loss_phy_count
        + loss_dap
        + depth_loss_weight * loss_depth
        + color_loss_weight * loss_cos
        + silhouette_loss_weight * loss_dice
    )

    if torch.isnan(loss) or torch.isinf(loss):
        return None

    loss.backward()

    cls_accuracy = (correct_cls / norm_m) if norm_m > 0 else 0.0

    return {
        "loss": loss.item(),
        "anchor_pos_loss": loss_anchor_pos.item(),
        "anchor_exist_loss": loss_anchor_exist.item(),
        "phy_count_loss": loss_phy_count.item(),
        "pred_phy_mean": pred_num_phytomers.mean().item() if pred_num_phytomers is not None else 0.0,
        "gt_phy_mean": gt_phy_counts.mean().item(),
        "dap_loss": loss_dap.item(),
        "fine_vel_loss": loss_fine_vel.item(),
        "fine_exist_loss": loss_fine_exist.item(),
        "dense_depth_loss": loss_depth.item(),
        "cos_color_loss": loss_cos.item(),
        "silhouette_dice_loss": loss_dice.item(),
        "cls_acc": cls_accuracy,
    }


def probe_optimal_batch_size(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: FlowMatchingScheduler,
    matcher: HierarchicalBotanicalMatcher,
    dataset,
    device: torch.device,
    target_vram_ratio: float = 0.90,
    min_batch: int = 8,
    max_batch: int = 112,
    rank: int = 0,
    renderer: Optional[HeliosPyTorchRenderer] = None,
    vae: Optional[nn.Module] = None,
    depth_loss_weight: float = 0.5,
    color_loss_weight: float = 0.2,
    silhouette_loss_weight: float = 2.0,
    render_ratio: float = 1.0,
    render_sub_batch: Optional[int] = None,
) -> int:
    """
    Directly measures base model/optimizer memory and per-sample activation memory on this GPU
    to determine the optimal batch size achieving target_vram_ratio (e.g. 90%).
    Calibrated against worst-case mature plants (DAP=100, 4,096 slots) to guarantee zero OOM.
    """
    if device.type != "cuda":
        return min_batch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_allocated_bytes = torch.cuda.memory_allocated(device)
    total_vram_bytes = torch.cuda.get_device_properties(device).total_memory

    raw_model = model.module if hasattr(model, "module") else model
    slots_per_anchor = raw_model.slots_per_anchor
    max_fine_slots = raw_model.max_anchors * slots_per_anchor

    probe_n = min(min_batch, len(dataset))
    probe_items = [dataset[i] for i in range(probe_n)]
    probe_batch = {
        k: torch.stack([item[k] for item in probe_items])
        for k in ["image", "nodes", "existence_mask", "dap"]
        if k in probe_items[0]
    }

    # Calibrate on worst-case mature plants (DAP=100, full slots) to ensure safe batch size
    if "dap" in probe_batch:
        probe_batch["dap"] = torch.full_like(probe_batch["dap"], 100.0)
    if "nodes" in probe_batch and probe_batch["nodes"].shape[1] < max_fine_slots:
        cur_n = probe_batch["nodes"].shape[1]
        pad_len = max_fine_slots - cur_n
        probe_batch["nodes"] = F.pad(probe_batch["nodes"], (0, 0, 0, pad_len))
        probe_batch["existence_mask"] = F.pad(probe_batch["existence_mask"], (0, pad_len))

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    _ = forward_backward_step(
        model=model,
        batch=probe_batch,
        optimizer=optimizer,
        scheduler=scheduler,
        matcher=matcher,
        device=device,
        slots_per_anchor=slots_per_anchor,
        renderer=renderer,
        vae=vae,
        depth_loss_weight=depth_loss_weight,
        color_loss_weight=color_loss_weight,
        silhouette_loss_weight=silhouette_loss_weight,
        render_ratio=render_ratio,
        render_sub_batch=render_sub_batch,
    )

    peak_probe_bytes = torch.cuda.max_memory_allocated(device)
    activation_bytes = peak_probe_bytes - base_allocated_bytes
    bytes_per_sample = max(activation_bytes / probe_n, 1024 * 1024)

    optimizer.zero_grad(set_to_none=True)
    del probe_batch, probe_items
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    target_vram_bytes = total_vram_bytes * target_vram_ratio
    available_bytes = target_vram_bytes - base_allocated_bytes
    if available_bytes <= 0:
        optimal_batch = min_batch
    else:
        calculated = int(available_bytes / bytes_per_sample)
        # Snap down to nearest multiple of 8 for Tensor Core efficiency
        optimal_batch = (calculated // 8) * 8
        optimal_batch = max(min_batch, min(max_batch, optimal_batch))

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        est_vram_gb = (base_allocated_bytes + optimal_batch * bytes_per_sample) / (1024**3)
        total_gb = total_vram_bytes / (1024**3)
        est_pct = (est_vram_gb / total_gb) * 100.0
        print("=" * 80)
        print("Dynamic VRAM Batch Size Auto-Tuner (Python Runtime Profiler)")
        print(f"  Model Parameters:       {param_count:.2f}M")
        print(f"  GPU Hardware:           {torch.cuda.get_device_name(device)} ({total_gb:.1f} GB total)")
        print(f"  Base Model+Optimizer:   {base_allocated_bytes / (1024**2):.1f} MiB")
        print(f"  Peak Activation/sample: {bytes_per_sample / (1024**2):.1f} MiB (calibrated on DAP=60, 4,096 slots)")
        print(f"  Target Utilization:     {target_vram_ratio * 100:.0f}% ({target_vram_bytes / (1024**3):.1f} GB with safety margin)")
        print(f"  Tuned Batch Size:       {optimal_batch} per GPU")
        print(f"  Estimated Peak VRAM:    {est_vram_gb:.1f} GB ({est_pct:.1f}% VRAM)")
        print("=" * 80, flush=True)

    return optimal_batch


def train_one_epoch(
    model: HierarchicalPartFlowMatchingModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: FlowMatchingScheduler,
    matcher: HierarchicalBotanicalMatcher,
    device: torch.device,
    epoch: int,
    rank: int = 0,
    renderer: Optional[HeliosPyTorchRenderer] = None,
    vae: Optional[nn.Module] = None,
    depth_loss_weight: float = 0.5,
    color_loss_weight: float = 0.2,
    silhouette_loss_weight: float = 2.0,
    render_ratio: float = 1.0,
    render_sub_batch: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_anchor_pos_loss = 0.0
    total_anchor_exist_loss = 0.0
    total_phy_count_loss = 0.0
    total_pred_phy_mean = 0.0
    total_gt_phy_mean = 0.0
    total_fine_vel_loss = 0.0
    total_fine_exist_loss = 0.0
    total_dense_depth_loss = 0.0
    total_cos_color_loss = 0.0
    total_dice_loss = 0.0
    total_cls_acc = 0.0
    count = 0

    slots_per_anchor = model.slots_per_anchor if not hasattr(model, "module") else model.module.slots_per_anchor

    for batch_idx, batch in enumerate(dataloader):
        step_metrics = forward_backward_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scheduler=scheduler,
            matcher=matcher,
            device=device,
            slots_per_anchor=slots_per_anchor,
            renderer=renderer,
            vae=vae,
            depth_loss_weight=depth_loss_weight,
            color_loss_weight=color_loss_weight,
            silhouette_loss_weight=silhouette_loss_weight,
            render_ratio=render_ratio,
            render_sub_batch=render_sub_batch,
        )
        if step_metrics is None:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += step_metrics["loss"]
        total_anchor_pos_loss += step_metrics["anchor_pos_loss"]
        total_anchor_exist_loss += step_metrics["anchor_exist_loss"]
        total_phy_count_loss += step_metrics.get("phy_count_loss", 0.0)
        total_pred_phy_mean += step_metrics.get("pred_phy_mean", 0.0)
        total_gt_phy_mean += step_metrics.get("gt_phy_mean", 0.0)
        total_fine_vel_loss += step_metrics["fine_vel_loss"]
        total_fine_exist_loss += step_metrics["fine_exist_loss"]
        total_dense_depth_loss += step_metrics["dense_depth_loss"]
        total_cos_color_loss += step_metrics["cos_color_loss"]
        total_dice_loss += step_metrics["silhouette_dice_loss"]
        total_cls_acc += step_metrics["cls_acc"]
        count += 1

        print_freq = max(1, len(dataloader) // 5)
        if rank == 0 and ((batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == len(dataloader)):
            print(
                f"  [Epoch {epoch:02d}] Step {batch_idx+1:03d}/{len(dataloader):03d} | "
                f"Loss: {step_metrics['loss']:.4f} (Vel: {step_metrics['fine_vel_loss']:.4f}, "
                f"AncPos: {step_metrics['anchor_pos_loss']:.4f}, "
                f"PhyLoss: {step_metrics.get('phy_count_loss', 0.0):.4f} [Pred:{step_metrics.get('pred_phy_mean', 0.0):.1f}/GT:{step_metrics.get('gt_phy_mean', 0.0):.1f}], "
                f"Exist: {step_metrics['fine_exist_loss']:.4f}, Depth: {step_metrics['dense_depth_loss']:.4f}, "
                f"Dice: {step_metrics['silhouette_dice_loss']:.4f}, "
                f"Acc: {step_metrics['cls_acc']*100:.1f}%)",
                flush=True,
            )

    return {
        "loss": total_loss / max(count, 1),
        "anchor_pos_loss": total_anchor_pos_loss / max(count, 1),
        "anchor_exist_loss": total_anchor_exist_loss / max(count, 1),
        "phy_count_loss": total_phy_count_loss / max(count, 1),
        "pred_phy_mean": total_pred_phy_mean / max(count, 1),
        "gt_phy_mean": total_gt_phy_mean / max(count, 1),
        "fine_vel_loss": total_fine_vel_loss / max(count, 1),
        "fine_exist_loss": total_fine_exist_loss / max(count, 1),
        "dense_depth_loss": total_dense_depth_loss / max(count, 1),
        "cos_color_loss": total_cos_color_loss / max(count, 1),
        "silhouette_dice_loss": total_dice_loss / max(count, 1),
        "cls_acc": total_cls_acc / max(count, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical Matryoshka Botanical Flow Matching")
    parser.add_argument("--data_dir", type=str, default="dataset/helios_data/cowpea")
    parser.add_argument("--cache_dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--output_dir", type=str, default="diffusion_based/checkpoints/hierarchical_fm")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size per GPU ('auto' for dynamic probe or integer)")
    parser.add_argument("--target_vram_ratio", type=float, default=0.90, help="Target fraction of total GPU VRAM (default: 0.90)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_anchors", type=int, default=512)
    parser.add_argument("--slots_per_anchor", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--vit_layers", type=int, default=8)
    parser.add_argument("--vit_heads", type=int, default=8)
    parser.add_argument("--coarse_layers", type=int, default=4)
    parser.add_argument("--fine_layers", type=int, default=6)
    parser.add_argument("--node_dim", type=int, default=16, help="Latent dimensionality for fine organ flow matching")
    parser.add_argument("--organ_vae_checkpoint", type=str, default="diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt", help="Path to pretrained OrganLatentVAE checkpoint")
    parser.add_argument("--depth_weight", type=float, default=0.5, help="Weight for in-loop differentiable dense depth loss")
    parser.add_argument("--color_weight", type=float, default=0.2, help="Weight for in-loop differentiable cosine color loss")
    parser.add_argument("--silhouette_weight", type=float, default=2.0, help="Weight for in-loop differentiable silhouette Dice loss")
    parser.add_argument("--render_ratio", type=float, default=1.0, help="Ratio of batch to render differentiably (0.0 to 1.0, default: 1.0 = 100 percent full batch)")
    parser.add_argument("--render_sub_batch", type=int, default=None, help="Explicit sub-batch size for in-loop differentiable rendering per GPU (overrides render_ratio if set)")
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--eval_every", type=int, default=1, help="Validation image generation interval in epochs (default: 1)")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to checkpoint to initialize weights from (strict=False)")
    parser.add_argument("--wandb_project", type=str, default="part-flow-matching")
    parser.add_argument("--wandb_run_name", type=str, default="hierarchical-matryoshka-cowpea")
    args = parser.parse_args()

    # DDP Initialization
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Dataset
    dataset = PartArrayDataset(
        data_root=args.data_dir,
        max_nodes=args.max_anchors * args.slots_per_anchor,
        cache_dir=args.cache_dir,
        species="cowpea",
        image_size=128,
    )

    # Differentiable Renderer for in-loop optical grounding and evaluation
    renderer = HeliosPyTorchRenderer(image_size=128, device=device).to(device)

    # Pretrained OrganLatentVAE loading (Option B: Latent Flow Matching)
    vae = None
    if args.organ_vae_checkpoint and os.path.exists(args.organ_vae_checkpoint):
        if rank == 0:
            print(f"Loading Pretrained OrganLatentVAE from {args.organ_vae_checkpoint}...")
        vae = OrganLatentVAE(latent_dim=args.node_dim, hidden_dim=256)
        vae_sd = torch.load(args.organ_vae_checkpoint, map_location="cpu", weights_only=True)
        vae.load_state_dict(vae_sd)
        vae = vae.to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        if rank == 0:
            print(f"OrganLatentVAE successfully initialized, frozen (requires_grad=False), and ready on {device}.")

    # Model & Optimizers (Instantiated first for memory profiling)
    model = HierarchicalPartFlowMatchingModel(
        max_anchors=args.max_anchors,
        slots_per_anchor=args.slots_per_anchor,
        node_dim=args.node_dim,
        num_classes=NUM_ORGAN_TYPES,
        image_size=128,
        patch_size=8,
        embed_dim=args.embed_dim,
        vit_layers=args.vit_layers,
        vit_heads=args.vit_heads,
        coarse_layers=args.coarse_layers,
        fine_layers=args.fine_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        print(f"Hierarchical Matryoshka FM Model Parameter Count: {total_params:.2f}M")

    # Parameter groups: DINOv2 backbone uses 0.1x learning rate (e.g. 2e-5) to preserve foundation priors
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "image_encoder.backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    if rank == 0:
        print(f"Optimizer setup: {len(backbone_params)} DINOv2 backbone tensors (lr={args.lr * 0.3:.1e}), "
              f"{len(other_params)} 3D/decoder tensors (lr={args.lr:.1e})")

    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.3},
        {"params": other_params, "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Optional Warm-Start Initialization or Full State Resume
    start_epoch = 1
    if args.init_checkpoint and os.path.isfile(args.init_checkpoint):
        if rank == 0:
            print(f"Loading checkpoint from: {args.init_checkpoint}")
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f"Model weights loaded successfully! Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        if args.resume and isinstance(ckpt, dict) and "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if rank == 0:
                        print("Optimizer state successfully restored from checkpoint.")
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: could not restore optimizer state: {e}")
            if rank == 0:
                print(f"Resuming training loop from epoch {start_epoch} of {args.epochs}")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6, last_epoch=start_epoch - 2 if start_epoch > 1 else -1
    )
    flow_scheduler = FlowMatchingScheduler()
    matcher = HierarchicalBotanicalMatcher(slots_per_anchor=args.slots_per_anchor).to(device)

    # Dynamic Batch Sizing (Python-side Profiling)
    if str(args.batch_size).lower() == "auto":
        resolved_batch_size = probe_optimal_batch_size(
            model=model,
            optimizer=optimizer,
            scheduler=flow_scheduler,
            matcher=matcher,
            dataset=dataset,
            device=device,
            target_vram_ratio=args.target_vram_ratio,
            rank=rank,
            renderer=renderer,
            vae=vae,
            depth_loss_weight=args.depth_weight,
            color_loss_weight=args.color_weight,
            silhouette_loss_weight=args.silhouette_weight,
            render_ratio=args.render_ratio,
            render_sub_batch=args.render_sub_batch,
        )
    else:
        resolved_batch_size = int(args.batch_size)

    # Synchronize resolved batch size across DDP workers
    if is_ddp:
        batch_tensor = torch.tensor([resolved_batch_size], dtype=torch.long, device=device)
        dist.broadcast(batch_tensor, src=0)
        resolved_batch_size = int(batch_tensor.item())

    # Build DataLoader with resolved batch size
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        print(f"Final Configuration: Batch per GPU = {resolved_batch_size} | Global Batch = {resolved_batch_size * world_size}")

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        if is_ddp and sampler is not None:
            sampler.set_epoch(epoch)

        epoch_metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=flow_scheduler,
            matcher=matcher,
            device=device,
            epoch=epoch,
            rank=rank,
            renderer=renderer,
            vae=vae,
            depth_loss_weight=args.depth_weight,
            color_loss_weight=args.color_weight,
            silhouette_loss_weight=args.silhouette_weight,
            render_ratio=args.render_ratio,
            render_sub_batch=args.render_sub_batch,
        )
        lr_scheduler.step()

        if rank == 0:
            max_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            vram_pct = (max_vram_gb / total_vram_gb) * 100.0 if total_vram_gb > 0 else 0.0

            print(
                f"Epoch {epoch:03d} | Loss: {epoch_metrics['loss']:.4f} | "
                f"VelLoss: {epoch_metrics['fine_vel_loss']:.4f} | "
                f"AncPosLoss: {epoch_metrics['anchor_pos_loss']:.4f} | "
                f"PhyLoss: {epoch_metrics['phy_count_loss']:.4f} (Pred:{epoch_metrics['pred_phy_mean']:.1f}/GT:{epoch_metrics['gt_phy_mean']:.1f}) | "
                f"ExistLoss: {epoch_metrics['fine_exist_loss']:.4f} | "
                f"DepthLoss: {epoch_metrics['dense_depth_loss']:.4f} | "
                f"CosLoss: {epoch_metrics['cos_color_loss']:.4f} | "
                f"DiceLoss: {epoch_metrics['silhouette_dice_loss']:.4f} | "
                f"ClsAcc: {epoch_metrics['cls_acc']*100:.1f}% | "
                f"VRAM: {max_vram_gb:.1f}/{total_vram_gb:.1f} GB ({vram_pct:.1f}%)"
            )
            wandb.log({
                "epoch": epoch,
                "train/loss": epoch_metrics["loss"],
                "train/fine_vel_loss": epoch_metrics["fine_vel_loss"],
                "train/anchor_pos_loss": epoch_metrics["anchor_pos_loss"],
                "train/phy_count_loss": epoch_metrics["phy_count_loss"],
                "train/pred_phy_mean": epoch_metrics["pred_phy_mean"],
                "train/gt_phy_mean": epoch_metrics["gt_phy_mean"],
                "train/fine_exist_loss": epoch_metrics["fine_exist_loss"],
                "train/dense_depth_loss": epoch_metrics["dense_depth_loss"],
                "train/cos_color_loss": epoch_metrics["cos_color_loss"],
                "train/silhouette_dice_loss": epoch_metrics["silhouette_dice_loss"],
                "train/cls_acc": epoch_metrics["cls_acc"],
                "train/vram_allocated_gb": max_vram_gb,
                "train/vram_utilization_pct": vram_pct,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_path = os.path.join(args.output_dir, f"hierarchical_fm_epoch_{epoch:03d}.pt")
                raw_model = model.module if hasattr(model, "module") else model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                }, save_path)
                print(f"Saved checkpoint to {save_path}", flush=True)

            # Differentiable Renderer Self-Consistency Check (Silhouette IoU & CHM Depth MAE & 3D Nodes)
            # Evaluated every args.eval_every epochs (default: 1 epoch)
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                try:
                    raw_model = model.module if hasattr(model, "module") else model
                    val_batch = next(iter(dataloader))
                    importlib.reload(ehsc)
                    val_metrics = ehsc.evaluate_self_consistency_batch(
                        model=raw_model,
                        val_batch=val_batch,
                        renderer=renderer,
                        device=device,
                        epoch=epoch,
                        output_dir="docs/results/assets",
                        num_samples_to_plot=4,
                        vae=vae,
                    )
                    print(
                        f"  [Self-Consistency Epoch {epoch:03d}] Silhouette IoU: {val_metrics['silhouette_iou']*100:.1f}% | "
                        f"Depth MAE: {val_metrics['depth_mae']*100:.2f} cm | "
                        f"Node RMSE: {val_metrics.get('val/node_rmse_cm', val_metrics.get('node_rmse_cm', 0.0)):.1f} cm",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [Self-Consistency Warning] Evaluation skipped: {e}", flush=True)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
