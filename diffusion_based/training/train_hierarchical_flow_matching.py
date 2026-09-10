"""
Distributed Training Script for Hierarchical Matryoshka Botanical Flow Matching.

Jointly trains:
  Stage 1: Coarse Skeletal Transformer (3D anchors, existence logits, aux DAP)
  Stage 2: Fine Botanical Flow Matching Decoder (local organ vector field, discrete classes)
"""

import os
import re
import sys
import argparse
import math
import json
import random
import time
from typing import Any, Dict, List, Optional

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
    FM_SCALE_START,
    GEOM_NODE_DIM,
    NUM_ORGAN_TYPES,
    BASE_SCALE,
    SCALE_SCALE,
    CURV_SCALE,
    decode_fm,
)
from diffusion_based.dataset.dap_bucket_sampler import DAPBucketBatchSampler
from diffusion_based.models.plant_organ_array import (
    ORGAN_SHOOT_META,
)
from diffusion_based.training.flow_matching import FlowMatchingScheduler
from diffusion_based.models.hierarchical_part_flow_matching import (
    HierarchicalPartFlowMatchingModel,
    compute_matryoshka_slice,
    estimate_anchor_capacity,
    ANCHOR_MARGIN,
    ANCHOR_MARGIN_FLAT,
    build_phytomer_flow_target,
    split_phytomer_flow_target,
    apply_ref_for_flow,
    PHYTO_FLOW_BASE_START,
    PHYTO_FLOW_BASE_END,
    PHYTO_FLOW_ROT_START,
    PHYTO_FLOW_ROT_END,
    PHYTO_FLOW_SCALE_START,
    PHYTO_FLOW_SCALE_END,
    PHYTO_FLOW_LATENT_START,
)
from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.dataset.phytomer_packets import (
    build_phytomer_packets,
    decode_packets,
    rot6d_to_matrix,
    assemble_packets,
    anchor_scale,
    denormalize_packet_scales,
)
from diffusion_based.training.hierarchical_hungarian_matcher import (
    HierarchicalBotanicalMatcher,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.organ_latent_vae import OrganLatentVAE
import importlib
import diffusion_based.eval.eval_hierarchical_self_consistency as ehsc


def _sync_cuda():
    """GPU flush for truthful wall-time section timers (CUDA is async).
    ~10us per call — negligible against second-scale sections."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    render_fraction: float = 1.0 / 6.0,
    capacity_schedule: Optional[Dict[str, float]] = None,
    flow_granularity: str = "organ",
    phytomer_vae: Optional[nn.Module] = None,
    phy_count_weight: float = 2.0,
    scale_weight: float = 2.0,
    render_grad: bool = True,
) -> Optional[Dict[str, float]]:
    images = batch["image"].to(device)
    nodes = batch["nodes"].to(device)  # (B, N_max, 26)
    existence_mask = batch["existence_mask"].to(device)
    phytomer_ids = batch.get("phytomer_ids", None)
    if phytomer_ids is not None:
        phytomer_ids = phytomer_ids.to(device)
    daps = batch.get("dap", None)
    if daps is not None:
        daps = daps.to(device)

    B = images.shape[0]

    # Extract GT labels and sub-slice for Matryoshka capacity
    type_labels = nodes[:, :, :FM_OT_END].argmax(dim=-1)  # (B, N_max)
    raw_model = model.module if hasattr(model, "module") else model

    # Scheduled capacity teacher forcing:
    #   'gt'   - K from GT DAP (noise-free teacher forcing)
    #   'pred' - K from the model's own predicted phytomer count, with multiplicative
    #            log-normal noise (~+30% sigma) so the capacity is >= the clean GT
    #            slice with high probability (under-allocation guard while exposing
    #            the model to inference-time capacity distribution).
    # p_pred = fraction of batches using the predicted path, linearly ramped from 0
    # at --capacity_warmup_epochs to 1.0 at --capacity_full_epochs.
    use_pred_capacity = False
    capacity_pred_prob = 0.0
    if capacity_schedule is not None and capacity_schedule.get("p_pred", 0.0) > 0.0:
        capacity_pred_prob = float(capacity_schedule["p_pred"])
        use_pred_capacity = (torch.rand(1).item() < capacity_pred_prob)

    # Probe tokens WITH grad (reused by the model forward below — saves one full
    # DINOv2 forward per step). Only the count scalar is detached for slicing.
    _sync_cuda()
    t_probe = time.time()
    image_tokens = raw_model.image_encoder(images)
    active_k = compute_matryoshka_slice(
        dap=daps, max_anchors=raw_model.max_anchors, margin=ANCHOR_MARGIN
    )
    if use_pred_capacity:
        # Two-pass: macro head first (gradient-safe, differentiable count), then
        # slice by the noisy predicted count (max'ed with GT slice for safety).
        with torch.no_grad():
            macro_probe = raw_model.coarse_stage.macro_head(
                image_tokens.detach()[:, 0], max_k=raw_model.max_anchors
            )
            pred_count = macro_probe["pred_num_phytomers"].detach().float().view(-1)
            # Multiplicative noise: x1.0 median with +30% log-normal sigma, biased
            # slightly upward (x1.10) so noisy capacity usually exceeds the clean
            # prediction (under-allocation guard) while exposing the model to the
            # inference-time capacity distribution.
            noise = torch.exp(torch.randn_like(pred_count) * 0.30) * 1.10
            noisy_count = pred_count * noise
            k_pred = math.ceil(
                float(noisy_count.max().item()) * ANCHOR_MARGIN + ANCHOR_MARGIN_FLAT
            )
            k_pred = int(min(max(k_pred, 8), raw_model.max_anchors))
            k_gt = compute_matryoshka_slice(
                dap=daps, max_anchors=raw_model.max_anchors, margin=ANCHOR_MARGIN
            )
            # Under-allocation guard: never slice below the GT-DAP curve, but
            # cap the predicted path at 2x the GT slice so an early over-predicting
            # macro head cannot explode the anchor bank (and step time) to
            # max_anchors during the first epochs.
            active_k = max(k_pred, int(k_gt))
            active_k = min(active_k, max(int(k_gt) * 2, 8))
    prof["probe"] = time.time() - t_probe
    active_fine = min(active_k * slots_per_anchor, nodes.shape[1])

    # Per-sample anchor capacity from the calibrated DAP curve. Slots beyond a
    # sample's own capacity are excluded from GT existence targets so the
    # existence head is never penalized for not predicting organs that the
    # sliced anchor bank cannot reach (under-allocation safety valve).
    per_sample_cap = estimate_anchor_capacity(
        daps, margin=ANCHOR_MARGIN, flat=ANCHOR_MARGIN_FLAT, max_anchors=active_k
    )  # (B,) in [8, active_k]

    nodes_sub = nodes[:, :active_fine]
    type_labels_sub = type_labels[:, :active_fine]
    existence_mask_sub = existence_mask[:, :active_fine]
    phytomer_ids_sub = None
    if phytomer_ids is not None:
        phytomer_ids_sub = phytomer_ids[:, :active_fine]
    act_mask_sub = (type_labels_sub > ORGAN_SHOOT_META) & (existence_mask_sub > 0.5)

    # Existence-aware per-sample mask: cap each sample's actives to its capacity slice
    slot_anchor_idx = torch.arange(active_fine, device=device) // slots_per_anchor  # (active_fine,)
    cap_mask_sub = slot_anchor_idx.unsqueeze(0) < per_sample_cap.to(device).unsqueeze(1)  # (B, active_fine)
    act_mask_sub = act_mask_sub & cap_mask_sub

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
    # (phytomer mode builds its own 9+D bridge prior below)
    if flow_granularity != "phytomer":
        z_0 = torch.randn(B, active_fine, node_dim, device=device)
        t = scheduler.sample_time(B, device)
        z_t = scheduler.sample_xt(z_0, tgt_z1, t)

    # ------------------------------------------------------------------
    # PHYTOMER MODE: build the 76D bridge flow target per anchor
    #   z_1 = [ GT_anchor_pos(3) | GT_anchor_rot(6) | GT_anchor_scale(3) | VAE_latent(64) ]
    # and the bridge prior x_0 = [ scaffold_pos+eps | scaffold_rot+eps | N(0,I) ].
    # The GT anchor pose comes from the matcher's cluster centers (position)
    # and the packet reference rotation (Option 1: anchor frame).
    # ------------------------------------------------------------------
    prof = {"packet_build": 0.0, "fwd1": 0.0, "fwd2": 0.0, "matcher": 0.0,
            "target_build": 0.0, "render": 0.0, "loss": 0.0,
            "backward": 0.0, "probe": 0.0, "other": 0.0}

    _sync_cuda()
    _t_fbs = time.time()
    t0 = time.time()
    phyto_targets = None
    if flow_granularity == "phytomer":
        D = raw_model.phytomer_latent_dim
        # FAST PATH: precomputed pkt cache (packets/presence/centers/refs/latent
        # per sample, from generate_cache.py / train_phytomer_vae.py --pkt-cache-dir).
        # The batch carries a 'pkt' list of per-sample dicts (or None where the
        # pkt cache is missing — those samples fall back to the on-the-fly build).
        pkt_batch = batch.get("pkt", None)
        phyto_targets = []
        for b in range(B):
            pb = pkt_batch[b] if pkt_batch is not None else None
            if pb is not None:
                phyto_targets.append({
                    "packets": pb["packets"].to(device),
                    "presence": pb["presence"].to(device),
                    "centers": pb["centers"].to(device),
                    "refs": pb["refs"].to(device),
                    "latent": pb["latent"].to(device),
                })
                continue
            # Per-sample: build canonical packets from the GT nodes, then encode
            # each packet's latent with the frozen PhytomerVAE. The matcher's
            # anchor_tgt_pos gives the GT anchor base; the packet reference_rot
            # gives the GT anchor rotation (the internode frame).
            act_b = act_mask_sub[b]
            if not act_b.any():
                phyto_targets.append(None)
                continue
            nodes_b = nodes_sub[b, act_b]
            ids_b = None
            if phytomer_ids_sub is not None:
                ids_b = phytomer_ids_sub[b, act_b]
            packets, presence, centers, refs = build_phytomer_packets(
                nodes_b, existence_mask=existence_mask_sub[b, act_b],
                phytomer_ids=ids_b)
            if packets.shape[0] == 0:
                phyto_targets.append(None)
                continue
            with torch.no_grad():
                lat = phytomer_vae.encode(
                    phytomer_vae.pack_input(packets.to(device), presence.to(device)))[0]
            # GT anchor pos = cluster center (metres); GT anchor rot = reference.
            phyto_targets.append({
                "packets": packets.to(device),
                "presence": presence.to(device),
                "centers": centers.to(device),
                "refs": refs.to(device),
                "latent": lat,  # (P, D)
            })
        # The matcher will provide anchor_tgt_pos per matched anchor; we build
        # the full (B, K, 9+D) target after matching (below).
    prof["packet_build"] = time.time() - t0

    optimizer.zero_grad()

    if flow_granularity == "phytomer":
        # Bridge prior: x_0 = [scaffold_pos + eps | scaffold_rot + eps | N(0,I)(D)]
        # The scaffold pose comes from the model's Stage-2 prediction (computed
        # below in the forward pass); we build x_0 AFTER the forward so we can
        # use the predicted scaffold as the bridge init. For the first forward
        # we use a placeholder Gaussian (the velocity target is independent of
        # x_0's exact value; the bridge init only affects the ODE at inference).
        D = raw_model.phytomer_latent_dim
        z_0 = torch.randn(B, active_k, 12 + D, device=device)
        t = scheduler.sample_time(B, device)
        # placeholder x_t (will be re-interpolated after forward with scaffold)
        z_t = z_0.clone()
    else:
        _sync_cuda()
        t0 = time.time()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                noisy_fine_nodes=z_t,
                timesteps=t,
                images=images,
                daps=daps,
                image_tokens=image_tokens,
            )
        prof["fwd1"] = time.time() - t0

    if flow_granularity == "phytomer":
        t0 = time.time()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                noisy_fine_nodes=z_t,
                timesteps=t,
                images=images,
                daps=daps,
                image_tokens=image_tokens,
            )
        prof["fwd1"] = time.time() - t0
        # Rebuild the bridge prior with the predicted scaffold pose.
        pred_anchor_pos = outputs["pred_anchor_pos"].float()   # (B, K, 3)
        pred_anchor_rot = outputs["pred_anchor_rot"].float()   # (B, K, 6)
        K_eff = pred_anchor_pos.shape[1]
        z_0 = torch.randn(B, K_eff, 12 + D, device=device)
        z_0[:, :, PHYTO_FLOW_BASE_START:PHYTO_FLOW_BASE_END] = (
            pred_anchor_pos + 0.05 * z_0[:, :, PHYTO_FLOW_BASE_START:PHYTO_FLOW_BASE_END])
        z_0[:, :, PHYTO_FLOW_ROT_START:PHYTO_FLOW_ROT_END] = (
            pred_anchor_rot + 0.05 * z_0[:, :, PHYTO_FLOW_ROT_START:PHYTO_FLOW_ROT_END])
        if "pred_anchor_scale" in outputs:
            z_0[:, :, PHYTO_FLOW_SCALE_START:PHYTO_FLOW_SCALE_END] = (
                outputs["pred_anchor_scale"].float()
                + 0.05 * z_0[:, :, PHYTO_FLOW_SCALE_START:PHYTO_FLOW_SCALE_END])
        # t=0 bridge init; the true x_t is re-interpolated after matching
        # (z_1 = [GT_pos | GT_rot | GT_scale | latent] built below).
        z_t = z_0.clone()

    # Model may resolve a wider anchor slice than the padded GT tensor provides
    # (active_fine is clamped to nodes.shape[1] above); align the z tensors.
    K_out = int(outputs["active_k"]) if "active_k" in outputs else active_k
    if flow_granularity == "phytomer":
        active_fine = K_out * slots_per_anchor  # legacy flat surface (K*8)
        if z_t.shape[1] != K_out:
            if z_t.shape[1] < K_out:
                pad_n = K_out - z_t.shape[1]
                z_t = torch.nn.functional.pad(z_t, (0, 0, 0, pad_n))
                z_0 = torch.nn.functional.pad(z_0, (0, 0, 0, pad_n))
    else:
        active_fine = K_out * slots_per_anchor
        if active_fine != z_t.shape[1]:
            if z_t.shape[1] < active_fine:
                pad = active_fine - z_t.shape[1]
                z_t = torch.nn.functional.pad(z_t, (0, 0, 0, pad))
                z_0 = torch.nn.functional.pad(z_0, (0, 0, 0, pad))
                tgt_z1 = torch.nn.functional.pad(tgt_z1, (0, 0, 0, pad))
    # Pad GT nodes/masks to the model's resolved width so matching/losses stay aligned
    if active_fine > nodes.shape[1]:
        pad_n = active_fine - nodes.shape[1]
        original_width = nodes.shape[1]
        nodes = torch.nn.functional.pad(nodes, (0, 0, 0, pad_n))
        type_labels = torch.nn.functional.pad(type_labels, (0, pad_n), value=0)
        existence_mask = torch.nn.functional.pad(existence_mask, (0, pad_n), value=0.0)
        nodes_sub = nodes[:, :active_fine]
        type_labels_sub = type_labels[:, :active_fine]
        existence_mask_sub = existence_mask[:, :active_fine]
        # Keep only the originally real slots (padded region is empty)
        original_mask = torch.arange(active_fine, device=device) < original_width
        act_mask_sub = torch.nn.functional.pad(act_mask_sub, (0, pad_n), value=False)
        act_mask_sub = act_mask_sub & original_mask.unsqueeze(0)
        if active_fine == cap_mask_sub.shape[1]:
            act_mask_sub = act_mask_sub & cap_mask_sub

    pred_anchor_pos = outputs["pred_anchor_pos"].float()              # (B, K, 3)
    pred_anchor_logits = outputs["pred_anchor_logits"].float()          # (B, K, 1)
    pred_velocity = outputs["pred_velocity"].float()                  # (B, active_fine, node_dim)
    pred_fine_exist_logits = outputs["pred_fine_exist_logits"].float()  # (B, active_fine, 1)
    pred_anchor_scale = outputs.get("pred_anchor_scale")  # (B, K, 3) or None
    if pred_anchor_scale is not None:
        pred_anchor_scale = pred_anchor_scale.float()
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
    t0 = time.time()
    matches = matcher(
        pred_anchor_pos=pred_anchor_pos,
        pred_anchor_logits=pred_anchor_logits,
        pred_fine_geom=clean_z1,
        pred_fine_logits=pred_fine_exist_logits,
        tgt_geoms=tgt_geoms_list,
        tgt_labels=tgt_labels_list,
        tgt_positions=tgt_positions_list,
        soft_margin_weights=soft_margin_weights,
        per_sample_max_phytomers=per_sample_cap,
        skip_fine=(flow_granularity == "phytomer"),
    )
    prof["matcher"] = time.time() - t0

    # Extract GT phytomer counts from matcher clusters
    gt_phy_counts = torch.tensor(
        [m["num_gt_phytomers"] for m in matches],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)  # (B, 1)

    # ------------------------------------------------------------------
    # PHYTOMER MODE: build the 73D flow target z_1 per matched anchor and
    # re-interpolate x_t with the true bridge prior.
    #   z_1 = [ GT_anchor_pos(3) | GT_anchor_rot(6) | VAE_latent(64) ]
    # GT anchor pos = matcher cluster center; GT anchor rot = packet reference.
    # ------------------------------------------------------------------
    if flow_granularity == "phytomer":
        D = raw_model.phytomer_latent_dim
        K_eff = pred_anchor_pos.shape[1]
        M = raw_model.slots_per_anchor
        tgt_z1_phyto = torch.zeros(B, K_eff, 12 + D, device=device)
        slot_presence_target = torch.zeros(B, K_eff, M, device=device)
        for b in range(B):
            m_b = matches[b]
            anc_src = m_b["anchor_src_idx"]
            anc_tgt = m_b["anchor_tgt_idx"]
            if len(anc_src) == 0 or phyto_targets[b] is None:
                continue
            pt = phyto_targets[b]
            # GT anchor pos from matcher cluster centers (metres)
            gt_pos = m_b["anchor_tgt_pos"]  # (M_anc, 3)
            # The matcher's anc_tgt indexes its CAPACITY-CLAMPED cluster set, but
            # pt (packets) is the FULL cluster list. Match by position: each GT
            # anchor pos corresponds to the packet whose center is nearest.
            if pt["centers"].shape[0] == 0:
                continue
            dist = torch.cdist(gt_pos, pt["centers"])  # (M_anc, P)
            pkt_idx = dist.argmin(dim=1)               # (M_anc,)
            # GT anchor rot = packet reference rotation (Option 1 frame)
            gt_rot = pt["refs"][pkt_idx]    # (M_anc, 6)
            gt_lat = pt["latent"][pkt_idx]  # (M_anc, D)
            # GT anchor scale = petiole (slot 1) scale row of the matched packets
            # (absolute FM units; the latent carries only normalized scales).
            gt_scl = anchor_scale(pt["packets"].to(device))[pkt_idx]  # (M_anc, 3)
            tgt_z1_phyto[b, anc_src] = build_phytomer_flow_target(gt_pos, gt_rot, gt_scl, gt_lat)
            slot_presence_target[b, anc_src] = pt["presence"][pkt_idx].float()
        z_0 = torch.randn(B, K_eff, 12 + D, device=device)
        z_0[:, :, PHYTO_FLOW_BASE_START:PHYTO_FLOW_BASE_END] = (
            pred_anchor_pos + 0.05 * z_0[:, :, PHYTO_FLOW_BASE_START:PHYTO_FLOW_BASE_END])
        z_0[:, :, PHYTO_FLOW_ROT_START:PHYTO_FLOW_ROT_END] = (
            pred_anchor_rot + 0.05 * z_0[:, :, PHYTO_FLOW_ROT_START:PHYTO_FLOW_ROT_END])
        # Bridge init for the scale dims from the Stage-2 coarse scale head
        # (mirrors the pose init; falls back to N(0,I) if unavailable).
        if "pred_anchor_scale" in outputs:
            z_0[:, :, PHYTO_FLOW_SCALE_START:PHYTO_FLOW_SCALE_END] = (
                outputs["pred_anchor_scale"].float()
                + 0.05 * z_0[:, :, PHYTO_FLOW_SCALE_START:PHYTO_FLOW_SCALE_END])
        z_t = scheduler.sample_xt(z_0, tgt_z1_phyto, t)
        # Re-run the model forward with the true x_t (bridge init).
        t0 = time.time()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                noisy_fine_nodes=z_t,
                timesteps=t,
                images=images,
                daps=daps,
                image_tokens=image_tokens,
            )
        prof["fwd2"] = time.time() - t0
        pred_velocity = outputs["pred_velocity"].float()          # (B, K, 12+D)
        pred_fine_exist_logits = outputs["pred_fine_exist_logits"].float()  # (B, K, M)
        clean_z1 = z_t + (1.0 - t.view(B, 1, 1)) * pred_velocity
        # Velocity target: v = z_1 - z_0 (bridge flow).
        tgt_velocity = tgt_z1_phyto - z_0
        # Loss: velocity MSE on matched anchors + slot existence BCE.
        loss_fine_vel_acc = torch.tensor(0.0, device=device)
        loss_fine_exist_acc = torch.tensor(0.0, device=device)
        total_matched_fine = 0
        correct_cls = 0
        total_cls_slots = 0
        # Decode the clean latent once per batch for the class-accuracy metric
        # (frozen VAE, no grad): (B*K, D) -> (B*K, M, 13) cls logits.
        with torch.no_grad():
            clean_lat = clean_z1[..., PHYTO_FLOW_LATENT_START:]
            vae_out = phytomer_vae.decode(clean_lat.reshape(-1, D), use_rot_branch=True)
            pred_cls_all = vae_out["cls_logits"].argmax(-1).reshape(B, K_eff, M)  # (B, K, M)
        for b in range(B):
            m_b = matches[b]
            anc_src = m_b["anchor_src_idx"]
            if len(anc_src) == 0:
                continue
            p_v = pred_velocity[b, anc_src]
            t_v = tgt_velocity[b, anc_src]
            loss_fine_vel_acc += F.mse_loss(p_v, t_v, reduction="sum") / float(12 + D)
            total_matched_fine += len(anc_src)
            # Slot existence BCE (K, M)
            exist_t = slot_presence_target[b]
            pos_w = torch.tensor([12.0], device=device)
            loss_fine_exist_acc += F.binary_cross_entropy_with_logits(
                pred_fine_exist_logits[b], exist_t, pos_weight=pos_w)
            # Class accuracy on matched anchors' present slots (frozen VAE decode).
            pt = phyto_targets[b]
            if pt is not None and pt["packets"].shape[0] > 0:
                dist = torch.cdist(m_b["anchor_tgt_pos"], pt["centers"])
                pkt_idx = dist.argmin(dim=1)
                tgt_cls = pt["packets"][pkt_idx, :, :FM_OT_END].argmax(-1)  # (M_anc, M)
                pres = pt["presence"][pkt_idx]  # (M_anc, M)
                pred_cls_m = pred_cls_all[b, anc_src]
                correct_cls += ((pred_cls_m == tgt_cls).float() * pres.float()).sum().item()
                total_cls_slots += int(pres.sum().item())
        # Idle anchor damping: unmatched anchors' velocity -> 0
        all_anc = torch.arange(K_eff, device=device)
        matched_set = torch.zeros(K_eff, dtype=torch.bool, device=device)
        for b in range(B):
            m_b = matches[b]
            if len(m_b["anchor_src_idx"]) > 0:
                matched_set[m_b["anchor_src_idx"]] = True
        idle_anc = all_anc[~matched_set]
        if len(idle_anc) > 0:
            loss_fine_vel_acc += 0.02 * (pred_velocity[:, idle_anc] ** 2).sum() / float(12 + D)
        norm_m = max(total_matched_fine, 1)
        loss_fine_vel = loss_fine_vel_acc / norm_m
        loss_fine_exist = loss_fine_exist_acc / max(B, 1)
        # Anchor losses (same as organ mode)
        loss_anchor_pos_acc = torch.tensor(0.0, device=device)
        loss_anchor_exist_acc = torch.tensor(0.0, device=device)
        loss_anchor_scale_acc = torch.tensor(0.0, device=device)
        total_matched_anc = 0
        for b in range(B):
            m_b = matches[b]
            anc_src = m_b["anchor_src_idx"]
            if len(anc_src) == 0:
                continue
            total_matched_anc += len(anc_src)
            exist_targets = torch.zeros(K_eff, 1, device=device)
            exist_targets[anc_src] = 1.0
            pos_weight_anc = torch.tensor([8.0], device=device)
            loss_anchor_exist_acc += F.binary_cross_entropy_with_logits(
                pred_anchor_logits[b], exist_targets, pos_weight=pos_weight_anc)
            if "anchor_tgt_pos" in m_b and len(m_b["anchor_tgt_pos"]) > 0:
                p_anc = pred_anchor_pos[b, anc_src]
                t_anc = m_b["anchor_tgt_pos"]
                loss_anchor_pos_acc += F.smooth_l1_loss(p_anc, t_anc, reduction="sum")
            # Anchor scale loss (phytomer mode): Stage-2 coarse scale vs the GT
            # anchor scale (petiole scale row of the matched packets). Only when
            # the model predicts it and GT packets are available.
            if (pred_anchor_scale is not None and phyto_targets[b] is not None
                    and flow_granularity == "phytomer"):
                pt = phyto_targets[b]
                if pt["packets"].shape[0] > 0:
                    dist = torch.cdist(m_b["anchor_tgt_pos"], pt["centers"])
                    pkt_idx = dist.argmin(dim=1)
                    gt_scl = anchor_scale(pt["packets"].to(device))[pkt_idx]
                    p_scl = pred_anchor_scale[b, anc_src]
                    loss_anchor_scale_acc += F.smooth_l1_loss(p_scl, gt_scl, reduction="sum")
        norm_anc = max(total_matched_anc, 1)
        loss_anchor_pos = loss_anchor_pos_acc / norm_anc
        loss_anchor_exist = loss_anchor_exist_acc / max(B, 1)
        loss_anchor_scale = loss_anchor_scale_acc / norm_anc
        # Skip the organ-mode loss loop below.
        phyto_mode_done = True
    else:
        phyto_mode_done = False

    # Loss Computation
    if not phyto_mode_done:
        loss_anchor_pos_acc = torch.tensor(0.0, device=device)
        loss_anchor_exist_acc = torch.tensor(0.0, device=device)
        loss_fine_vel_acc = torch.tensor(0.0, device=device)
        loss_fine_exist_acc = torch.tensor(0.0, device=device)

        total_matched_anc = 0
        total_matched_fine = 0
        correct_cls = 0
        total_cls_slots = 0

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
                # 0.02 (was 0.05): with capacity clamp + top-k guard + pred-capacity path,
                # unmatched-slot drift is already triple-suppressed; avoid over-regularizing
                # genuinely dormant reproductive slots.
                loss_fine_vel_acc += 0.02 * (p_v_idle ** 2).sum() / float(node_dim)

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
        # Anchor scale loss is phytomer-mode only (Stage-2 scale head supervises
        # the 76D flow's s_a); organ mode has no scale target.
        loss_anchor_scale = torch.tensor(0.0, device=device)

    # Stage 1: Macro Losses (Phytomer Count & Plant Age DAP)
    loss_phy_count = torch.tensor(0.0, device=device)
    if pred_num_phytomers is not None:
        loss_phy_count = F.smooth_l1_loss(pred_num_phytomers.float(), gt_phy_counts)

    loss_dap = torch.tensor(0.0, device=device)
    if daps is not None and pred_dap is not None:
        loss_dap = F.smooth_l1_loss(pred_dap.squeeze(-1), daps.float()) * 0.05

    # In-Loop Differentiable Optical Grounding (1-Step Clean Prediction -> 4-Scale Multi-Scale Pyramid Depth + Silhouette Dice + Cosine Color)
    # render_fraction is batch-relative: n_render = round(B * fraction), clamped to
    # [1, B]. Fraction (not absolute count) keeps per-sample photometric visit rate
    # constant across batch sizes — the 2026-09-08 incident (batch 24->48 with a
    # frozen absolute sub-batch halved supervision) cannot recur.
    loss_depth = torch.tensor(0.0, device=device)
    loss_cos = torch.tensor(0.0, device=device)
    loss_dice = torch.tensor(0.0, device=device)

    if renderer is not None and (depth_loss_weight > 0.0 or color_loss_weight > 0.0 or silhouette_loss_weight > 0.0):
        f = max(0.0, min(1.0, float(render_fraction)))
        n_render = max(1, min(B, int(round(B * f)))) if f > 0.0 else 0

        # Epoch gate for render GRADIENTS (not metrics): before
        # --render_grad_start_epoch the render loop runs under torch.no_grad()
        # (panels/metrics still computed, but the python-loop-heavy mesh/raster
        # backward is skipped entirely). From the gate epoch the full graph is
        # built so depth/dice keep shaping pose/geometry (part_14d gradient
        # retained per user decision — no straight-through hand-off).
        render_grad_on = bool(render_grad)
        if n_render > 0:
            _sync_cuda()
            t0 = time.time()
            if n_render < B:
                render_indices = torch.randperm(B, device=device)[:n_render]
            else:
                render_indices = torch.arange(B, device=device)

            loss_depth_acc = torch.tensor(0.0, device=device)
            loss_cos_acc = torch.tensor(0.0, device=device)
            loss_dice_acc = torch.tensor(0.0, device=device)

            # Multi-scale pyramid: 1x/2x only (4x/8x zoom covers a single leaf —
            # small, noisy gradients; profiling 2026-09-09: render = 70% of step
            # time, 4-scale -> 2-scale halves it with negligible supervision loss).
            pyramid_scales = [1.0, 2.0]
            num_scales = len(pyramid_scales)

            # Epoch gate for render GRADIENTS (metrics/panels still computed).
            # When render_grad is off, the render branch reads detached copies so
            # no autograd graph is built through the python-loop-heavy mesh/raster
            # path — the backward then skips it entirely. From the gate epoch the
            # full graph is built and depth/dice keep shaping pose/geometry.
            _rz = clean_z1 if render_grad_on else clean_z1.detach()
            _re = pred_fine_exist_logits if render_grad_on else pred_fine_exist_logits.detach()
            for b_idx in range(n_render):
                real_b = render_indices[b_idx].item()
                if flow_granularity == "phytomer":
                    # Decode the 76D flow vector: split pose + scale + latent,
                    # decode the latent to a scale-NORMALIZED relative packet,
                    # restore absolute scale with the refined anchor scale BEFORE
                    # assembling bases (the petiole-curve math needs the absolute
                    # petiole length), then re-anchor with the refined pose.
                    pos, rot, scl, lat = split_phytomer_flow_target(_rz[real_b], D)
                    out_vae = phytomer_vae.decode(lat)  # (K, M, 26), zeroed base, norm scale
                    recon_abs = denormalize_packet_scales(out_vae["recon_packets"], scl)
                    packet_hat = assemble_packets(recon_abs, rot)
                    abs_packets = apply_ref_for_flow(
                        packet_hat.unsqueeze(0), pos.unsqueeze(0), rot.unsqueeze(0)
                    )[0]
                    flat_abs = abs_packets.reshape(-1, 26)
                    keep = out_vae["cls_logits"].argmax(-1).reshape(-1) > 0
                    part_14d = decode_fm(flat_abs[keep])
                    exist_b = torch.sigmoid(_re[real_b]).reshape(-1)[keep]
                    probs = F.softmax(out_vae["cls_logits"], dim=-1).reshape(-1, 13)[keep]
                elif vae is not None:
                    # Differentiably decode 16D latents into 14D part tensor via frozen VAE!
                    part_14d, probs = vae.decode_to_part_tensor(_rz[real_b])
                    exist_b = torch.sigmoid(_re[real_b]).squeeze(-1)
                else:
                    base_xyz = _rz[real_b, :, 0:3] / BASE_SCALE
                    rot6d = _rz[real_b, :, 3:9]
                    scale_xyz = _rz[real_b, :, 9:12].clamp(min=1e-4) / SCALE_SCALE
                    curv_val = _rz[real_b, :, 12:13] / CURV_SCALE
                    cls_val = torch.ones((active_fine, 1), device=device) * 5.0
                    exist_b = torch.sigmoid(_re[real_b]).squeeze(-1)
                    part_14d = torch.cat([cls_val, base_xyz, rot6d, scale_xyz, curv_val], dim=-1)
                    probs = None

                # Differentiable mesh assembly with organ_probs attached for autograd backprop.
                # Semantic color: the model's learnable (13,3) palette replaces the
                # hardcoded per-type constants so the cos-color loss carries gradient
                # into the classifier (previously color was constant -> dead channel).
                _palette = getattr(raw_model, "color_palette", None)
                mesh_dict = renderer.geo_builder.build_mesh_from_part_tensor(
                    part_14d, existence=exist_b, organ_probs=probs, device=device,
                    color_palette=_palette,
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
            prof["render"] = time.time() - t0

    # Composite Loss (Macro Prior + 3D Node Scaffold + Intra-Phytomer Flow Matching + Photometric)
    # anchor_pos weight: 2.0 (pre-0389ca5 value that produced the balanced-data panels);
    # the 4.0 from 0389ca5 was compensating for the young-skewed 4-DAP dataset.
    loss = (
        2.0 * loss_anchor_pos
        + 1.0 * loss_anchor_exist
        + 2.0 * loss_fine_vel
        + 1.0 * loss_fine_exist
        + phy_count_weight * loss_phy_count
        + scale_weight * loss_anchor_scale
        + loss_dap
        + depth_loss_weight * loss_depth
        + color_loss_weight * loss_cos
        + silhouette_loss_weight * loss_dice
    )

    if torch.isnan(loss) or torch.isinf(loss):
        return None

    _sync_cuda()
    t_bwd = time.time()
    loss.backward()
    _sync_cuda()
    prof["backward"] = time.time() - t_bwd
    # Residual (data staging, probe decode, python overhead): total minus parts.
    prof["other"] = max(0.0, (time.time() - _t_fbs) - sum(
        prof.get(k, 0.0) for k in
        ("packet_build", "fwd1", "fwd2", "matcher", "render", "backward", "probe")))

    # Class accuracy: organ mode counts matched fine slots; phytomer mode counts
    # present slots of matched anchors (frozen VAE decode of the clean latent).
    cls_denom = total_cls_slots if flow_granularity == "phytomer" else norm_m
    cls_accuracy = (correct_cls / cls_denom) if cls_denom > 0 else 0.0

    return {
        "loss": loss.item(),
        "anchor_pos_loss": loss_anchor_pos.item(),
        "anchor_exist_loss": loss_anchor_exist.item(),
        "anchor_scale_loss": loss_anchor_scale.item(),
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
        "prof": prof,
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
    max_batch: int = 256,
    rank: int = 0,
    renderer: Optional[HeliosPyTorchRenderer] = None,
    vae: Optional[nn.Module] = None,
    depth_loss_weight: float = 0.5,
    color_loss_weight: float = 0.2,
    silhouette_loss_weight: float = 2.0,
    render_fraction: float = 1.0 / 6.0,
    flow_granularity: str = "organ",
    phytomer_vae: Optional[nn.Module] = None,
    phy_count_weight: float = 2.0,
    scale_weight: float = 2.0,
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
        render_fraction=render_fraction,
        flow_granularity=flow_granularity,
        phytomer_vae=phytomer_vae,
        phy_count_weight=phy_count_weight,
        scale_weight=scale_weight,
        render_grad=True,
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


# =============================================================================
# FIXED STRATIFIED EVAL SET (self-consistency as a reproducible dataset metric)
# =============================================================================
# The previous eval drew `next(iter(dataloader))` — a random batch whose composition
# (and thus IoU/Cos/Dice) swung ±30pt between panels. The fixed set is stratified by
# DAP bucket, chosen deterministically at launch, and persisted to JSON so panels
# and metrics are comparable across epochs and runs.

def build_eval_indices(dataset: PartArrayDataset, samples_per_bucket: int = 2, seed: int = 1234) -> List[int]:
    """Deterministically selects a DAP-stratified eval subset from dataset.samples.

    Returns sorted dataset indices. The selection (index, prefix, dap) is saved by
    the caller to eval_set.json for cross-run reproducibility.
    """
    import json as _json
    from collections import defaultdict as _dd

    by_bucket: Dict[int, List[int]] = _dd(list)
    for idx, s in enumerate(dataset.samples):
        m = re.search(r"dap(\d+)", s["prefix"])
        dap = int(m.group(1)) if m else 30
        by_bucket[(dap - 1) // 10].append(idx)

    rng = random.Random(seed)
    chosen: List[int] = []
    for bucket in sorted(by_bucket.keys()):
        pool = sorted(by_bucket[bucket])
        rng.shuffle(pool)
        chosen.extend(pool[:samples_per_bucket])
    return sorted(chosen)


def collate_eval_set(dataset: PartArrayDataset, indices: List[int], device: torch.device) -> Dict[str, Any]:
    """Collates the fixed eval set directly from the dataset (bypasses the sampler)."""
    items = [dataset[i] for i in indices]
    batch = {}
    keys = [k for k in items[0].keys() if isinstance(items[0][k], torch.Tensor)]
    for k in keys:
        batch[k] = torch.stack([it[k] for it in items]).to(device)
    # Preserve non-tensor extras (jpeg/prefix) — the evaluator uses them for the
    # Helios Raytrace reference column (Col 0).
    for k in ("jpeg", "prefix"):
        if k in items[0]:
            batch[k] = [it[k] for it in items]
    return batch


def collate_with_pkt(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """default_collate + per-sample 'pkt' dict passthrough (list of dicts)."""
    pkt_list = [it.pop("pkt", None) for it in batch]
    out = default_collate(batch)
    out["pkt"] = pkt_list
    return out


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
    render_fraction: float = 1.0 / 6.0,
    capacity_warmup_epochs: int = 0,
    capacity_full_epochs: int = 0,
    flow_granularity: str = "organ",
    phytomer_vae: Optional[nn.Module] = None,
    phy_count_weight: float = 2.0,
    scale_weight: float = 2.0,
    render_grad_start_epoch: int = 5,
    **kwargs,
) -> Dict[str, float]:
    model.train()
    # Scheduled capacity teacher-forcing decay: linear ramp of the predicted-
    # phytomer capacity path probability p_pred from 0 at warmup to 1 at full.
    p_pred = 0.0
    if capacity_full_epochs > capacity_warmup_epochs:
        frac = (epoch - capacity_warmup_epochs) / float(capacity_full_epochs - capacity_warmup_epochs)
        p_pred = max(0.0, min(1.0, frac))
    capacity_schedule = {"p_pred": p_pred}
    # Render-gradient epoch gate: before render_grad_start_epoch the render loop
    # runs graph-free (metrics/panels still produced); from the gate epoch the
    # full differentiable graph is built so depth/dice shape pose/geometry.
    render_grad = bool(epoch >= render_grad_start_epoch)
    # Optional per-step LR warmup callback (set by main(); scales optimizer lrs linearly)
    lr_warmup_cb = kwargs.pop("lr_warmup_cb", None)
    total_loss = 0.0
    total_anchor_pos_loss = 0.0
    total_anchor_exist_loss = 0.0
    total_anchor_scale_loss = 0.0
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
        # Per-step LR warmup scaling (before optimizer.step inside fwd/bwd)
        if lr_warmup_cb is not None:
            lr_warmup_cb()
        t_step0 = time.time()
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
            render_fraction=render_fraction,
            capacity_schedule=capacity_schedule,
            flow_granularity=flow_granularity,
            phytomer_vae=phytomer_vae,
            phy_count_weight=phy_count_weight,
            scale_weight=scale_weight,
            render_grad=render_grad,
        )
        t_step1 = time.time()
        if step_metrics is None:
            if lr_warmup_cb is not None:
                warmup_n = getattr(lr_warmup_cb, "advance", None)
                if warmup_n is not None:
                    warmup_n()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        t_step2 = time.time()

        if lr_warmup_cb is not None:
            warmup_n = getattr(lr_warmup_cb, "advance", None)
            if warmup_n is not None:
                warmup_n()

        total_loss += step_metrics["loss"]
        total_anchor_pos_loss += step_metrics["anchor_pos_loss"]
        total_anchor_exist_loss += step_metrics["anchor_exist_loss"]
        total_anchor_scale_loss += step_metrics.get("anchor_scale_loss", 0.0)
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
            p = step_metrics.get("prof", {})
            print(
                f"  [Epoch {epoch:02d}] Step {batch_idx+1:03d}/{len(dataloader):03d} | "
                f"Loss: {step_metrics['loss']:.4f} (Vel: {step_metrics['fine_vel_loss']:.4f}, "
                f"AncPos: {step_metrics['anchor_pos_loss']:.4f}, "
                f"Scl: {step_metrics.get('anchor_scale_loss', 0.0):.4f}, "
                f"PhyLoss: {step_metrics.get('phy_count_loss', 0.0):.4f} [Pred:{step_metrics.get('pred_phy_mean', 0.0):.1f}/GT:{step_metrics.get('gt_phy_mean', 0.0):.1f}], "
                f"Exist: {step_metrics['fine_exist_loss']:.4f}, Depth: {step_metrics['dense_depth_loss']:.4f}, "
                f"Dice: {step_metrics['silhouette_dice_loss']:.4f}, "
                f"Acc: {step_metrics['cls_acc']*100:.1f}%) | "
                f"fwd/bwd {t_step1-t_step0:.2f}s opt {t_step2-t_step1:.2f}s | "
                f"pkt {p.get('packet_build',0):.2f} fwd1 {p.get('fwd1',0):.2f} fwd2 {p.get('fwd2',0):.2f} "
                f"match {p.get('matcher',0):.2f} render {p.get('render',0):.2f} "
                f"backward {p.get('backward',0):.2f} probe {p.get('probe',0):.2f} other {p.get('other',0):.2f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(count, 1),
        "anchor_pos_loss": total_anchor_pos_loss / max(count, 1),
        "anchor_exist_loss": total_anchor_exist_loss / max(count, 1),
        "anchor_scale_loss": total_anchor_scale_loss / max(count, 1),
        "phy_count_loss": total_phy_count_loss / max(count, 1),
        "pred_phy_mean": total_pred_phy_mean / max(count, 1),
        "gt_phy_mean": total_gt_phy_mean / max(count, 1),
        "fine_vel_loss": total_fine_vel_loss / max(count, 1),
        "fine_exist_loss": total_fine_exist_loss / max(count, 1),
        "dense_depth_loss": total_dense_depth_loss / max(count, 1),
        "cos_color_loss": total_cos_color_loss / max(count, 1),
        "silhouette_dice_loss": total_dice_loss / max(count, 1),
        "cls_acc": total_cls_acc / max(count, 1),
        "capacity_p_pred": p_pred,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical Matryoshka Botanical Flow Matching")
    parser.add_argument("--data_dir", type=str, default="dataset/helios_data/cowpea")
    parser.add_argument("--cache_dir", type=str, default="dataset/cache/cowpea_curv26")
    parser.add_argument("--pkt_cache_dir", type=str, default="",
                        help="Precomputed phytomer packet targets cache dir "
                             "(from train_phytomer_vae.py --pkt-cache-dir). "
                             "Replaces the per-step packet build + VAE encode.")
    parser.add_argument("--output_dir", type=str, default="diffusion_based/checkpoints/hierarchical_fm")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size per GPU ('auto' for dynamic probe or integer)")
    parser.add_argument("--target_vram_ratio", type=float, default=0.90, help="Target fraction of total GPU VRAM (default: 0.90)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_anchors", type=int, default=512)
    parser.add_argument("--slots_per_anchor", type=int, default=10, help="Slots per phytomer packet (v2: 10 = stem, petiole, 3 leaflets, peduncle, 4 repro)")
    parser.add_argument("--anchor_locality_radius", type=float, default=None,
                        help="Soft locality radius (meters) for anchor matching: quadratic "
                             "penalty beyond R on top of L1 position cost. Calibrated "
                             "2026-09-09: R=0.10-0.15 covers >99%% of plausible matches "
                             "(GT NN-dist median 2.5cm vs anchor RMSE ~2cm). None = disabled "
                             "(exact legacy parity). Recommended: 0.12 with weight 50.0.")
    parser.add_argument("--anchor_locality_weight", type=float, default=50.0,
                        help="Weight of the soft locality penalty (see --anchor_locality_radius).")
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--vit_layers", type=int, default=8)
    parser.add_argument("--vit_heads", type=int, default=8)
    parser.add_argument("--coarse_layers", type=int, default=4)
    parser.add_argument("--fine_layers", type=int, default=6)
    parser.add_argument("--node_dim", type=int, default=16, help="Latent dimensionality for fine organ flow matching")
    parser.add_argument("--organ_vae_checkpoint", type=str, default="diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt", help="Path to pretrained OrganLatentVAE checkpoint")
    parser.add_argument("--flow_granularity", type=str, default="organ", choices=["organ", "phytomer"],
                        help="Stage-3 flow granularity: 'organ' = per-organ slots (8K x 16D, pose as conditioning); "
                             "'phytomer' = per-anchor 9+D vector [base(3) | rot(6) | latent(D)] refining the scaffold pose (bridge flow).")
    parser.add_argument("--phytomer_latent_dim", type=int, default=64, help="PhytomerVAE latent dim (flow_granularity=phytomer)")
    parser.add_argument("--phytomer_vae_checkpoint", type=str, default="diffusion_based/checkpoints/phytomer_vae_v2/phytomer_vae_64d_best.pt",
                        help="Path to frozen PhytomerVAE checkpoint (flow_granularity=phytomer)")
    parser.add_argument("--backbone", type=str, default="dinov2_vits14",
                        help="Image backbone: dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | "
                             "dinov3_vits16 | dinov3_vitb16 | dinov3_vitl16 | dinov3_vitl16_sat")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the DINO backbone (train only the task heads)")
    parser.add_argument("--depth_weight", type=float, default=0.5, help="Weight for in-loop differentiable dense depth loss")
    parser.add_argument("--color_weight", type=float, default=0.2, help="Weight for in-loop differentiable cosine color loss")
    parser.add_argument("--silhouette_weight", type=float, default=2.0, help="Weight for in-loop differentiable silhouette Dice loss")
    parser.add_argument("--render_fraction", type=float, default=1.0 / 6.0, help="Fraction of the batch rendered differentiably per step (batch-relative: n_render = round(B * fraction), clamped [1, B]). 1/6 restores the per-sample photometric visit rate of the 2026-09-07 runs; 1.0 renders the full batch (requires small batch — check VRAM).")
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--eval_every", type=int, default=25, help="Validation image generation interval in epochs (default: 25)")
    parser.add_argument("--render_grad_start_epoch", type=int, default=5,
                        help="Render-loop gradients (depth/dice through the mesh/raster graph) "
                             "activate at this epoch; before it the render loop runs under "
                             "torch.no_grad() (metrics/panels still computed). Saves the "
                             "55-65s render backward during warmup.")
    parser.add_argument("--scale_weight", type=float, default=2.0,
                        help="Loss weight for the Stage-2 anchor-scale head (76D flow s_a).")
    parser.add_argument("--eval_min_interval_minutes", type=int, default=30, help="Time-based eval fallback: force an eval panel if at least this many minutes elapsed since the last one (0 disables). Keeps diagnostic cadence roughly constant as dataset size grows per-epoch time.")
    parser.add_argument("--eval_samples_per_bucket", type=int, default=2, help="Fixed stratified eval set: samples per 10-DAP bucket (2 => ~20 samples).")
    parser.add_argument("--eval_seed", type=int, default=1234, help="Deterministic seed for the fixed stratified eval set.")
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.3, help="Backbone lr = args.lr * ratio (0.3: restored 2026-09-08 value; 0.15 starved macro-head CLS features)")
    parser.add_argument("--phy_count_weight", type=float, default=2.0, help="Loss weight for phytomer-count (was 0.5 — macro head learned ~6x slower than the Sep-8 organ run)")
    parser.add_argument("--init_phytomer_count", type=float, default=50.0,
                        help="Bias-init phy_head so pred_num starts near the dataset mean; removes the dead-anchor existence gate at epoch 0")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Linear LR warmup epochs (0.1x -> 1.0x per step; 0 disables)")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to checkpoint to initialize weights from (strict=False)")
    parser.add_argument("--dap_buckets", type=int, default=8, help="Number of DAP buckets for capacity-homogeneous batching (0 = plain shuffle)")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="DAP-stratified subset of the dataset for fast convergence smoke tests (0 = full dataset)")
    parser.add_argument("--capacity_warmup_epochs", type=int, default=50, help="Epochs of pure GT-DAP capacity teacher forcing before predicted-phytomer capacity ramps in")
    parser.add_argument("--capacity_full_epochs", type=int, default=150, help="Epoch at which predicted-phytomer capacity path reaches p_pred=1.0 (0 = disable ramp entirely)")
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
        pkt_cache_dir=args.pkt_cache_dir or None,
        species="cowpea",
        image_size=128,
    )
    if args.max_train_samples > 0 and args.max_train_samples < len(dataset.samples):
        # DAP-stratified subset (same bucket logic as the eval set) so young and
        # mature plants stay represented in fast smoke tests.
        import random as _random
        by_bucket: Dict[int, List[int]] = {}
        for idx, sm in enumerate(dataset.samples):
            m = re.search(r"dap(\d+)", sm["prefix"])
            dap = int(m.group(1)) if m else 30
            by_bucket.setdefault((dap - 1) // 10, []).append(idx)
        rng = _random.Random(42)
        keep: List[int] = []
        buckets = sorted(by_bucket.keys())
        per_bucket = max(1, args.max_train_samples // max(1, len(buckets)))
        for bkt in buckets:
            pool = sorted(by_bucket[bkt])
            rng.shuffle(pool)
            keep.extend(pool[:per_bucket])
        keep = sorted(keep)[: args.max_train_samples]
        dataset.samples = [dataset.samples[i] for i in keep]
        if rank == 0:
            daps = sorted({int(re.search(r"dap(\d+)", sm["prefix"]).group(1)) for sm in dataset.samples if re.search(r"dap(\d+)", sm["prefix"])})
            print(f"Train subset: {len(dataset.samples)} samples (max_train_samples={args.max_train_samples}), DAP range {daps[0]}-{daps[-1]}")

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
        flow_granularity=args.flow_granularity,
        phytomer_latent_dim=args.phytomer_latent_dim,
        backbone=args.backbone,
        freeze_backbone=args.freeze_backbone,
        init_phytomer_count=args.init_phytomer_count,
    ).to(device)

    # Frozen PhytomerVAE (flow_granularity=phytomer): encodes/decodes the
    # anchor-relative M-slot packet latent. Loaded once, frozen.
    phytomer_vae = None
    if args.flow_granularity == "phytomer":
        if not os.path.exists(args.phytomer_vae_checkpoint):
            raise FileNotFoundError(
                f"PhytomerVAE checkpoint not found: {args.phytomer_vae_checkpoint} "
                f"(required for --flow-granularity phytomer)")
        if rank == 0:
            print(f"Loading frozen PhytomerVAE from {args.phytomer_vae_checkpoint}...")
        phytomer_vae = PhytomerVAE(
            latent_dim=args.phytomer_latent_dim, hidden_dim=256).to(device)
        phytomer_vae.load_state_dict(torch.load(
            args.phytomer_vae_checkpoint, map_location=device, weights_only=True))
        phytomer_vae.eval()
        for p in phytomer_vae.parameters():
            p.requires_grad = False
        if rank == 0:
            print(f"PhytomerVAE-{args.phytomer_latent_dim}D frozen and ready.")

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
        print(f"Optimizer setup: {len(backbone_params)} DINOv2 backbone tensors (lr={args.lr * args.backbone_lr_ratio:.1e}), "
              f"{len(other_params)} 3D/decoder tensors (lr={args.lr:.1e})")

    param_groups = [
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_ratio},
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

    # Cosine decay over the full run.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6, last_epoch=start_epoch - 2 if start_epoch > 1 else -1
    )
    # Linear LR warmup for the first --warmup_epochs (default 3): removes the ep-1 loss
    # spike (166 -> 13.6 over 5 epochs in job 38146809) and protects the freshly
    # loaded DINOv2 backbone. Implemented as a per-step multiplier on top of the
    # cosine schedule (no optimizer state changes, DDP-safe).
    # NOTE: warmup_total_steps is finalized after the DataLoader is built (below).
    warmup_epochs = max(0, args.warmup_epochs)
    warmup_state = {"total_steps": 0, "n": 0}

    def current_warmup_scale() -> float:
        if warmup_state["total_steps"] == 0:
            return 1.0
        if start_epoch > 1:
            return 1.0  # resumed runs already passed warmup
        n = min(warmup_state["n"], warmup_state["total_steps"])
        return 0.1 + 0.9 * (n / warmup_state["total_steps"])

    def apply_warmup_scaling() -> None:
        if warmup_state["total_steps"] == 0 or start_epoch > 1:
            return
        scale = current_warmup_scale()
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * scale

    def lr_warmup_cb() -> None:
        apply_warmup_scaling()

    def lr_warmup_advance() -> None:
        warmup_state["n"] += 1

    lr_warmup_cb.advance = lr_warmup_advance  # type: ignore[attr-defined]

    # Snapshot base lrs for warmup scaling
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    flow_scheduler = FlowMatchingScheduler()
    matcher = HierarchicalBotanicalMatcher(
        slots_per_anchor=args.slots_per_anchor,
        anchor_locality_radius=args.anchor_locality_radius,
        anchor_locality_weight=args.anchor_locality_weight,
    ).to(device)

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
            render_fraction=args.render_fraction,
            flow_granularity=args.flow_granularity,
            phytomer_vae=phytomer_vae,
            phy_count_weight=args.phy_count_weight,
            scale_weight=args.scale_weight,
        )
    else:
        resolved_batch_size = int(args.batch_size)

    # Synchronize resolved batch size across DDP workers
    if is_ddp:
        batch_tensor = torch.tensor([resolved_batch_size], dtype=torch.long, device=device)
        dist.broadcast(batch_tensor, src=0)
        resolved_batch_size = int(batch_tensor.item())

    # Build DataLoader with resolved batch size
    # DAP-bucketed batching keeps each mini-batch developmentally homogeneous so the
    # anchor-slice capacity matches the botanical content (large Stage 3 FLOPs saving).
    if args.dap_buckets > 0:
        sample_daps = []
        for s in dataset.samples:
            m = re.search(r"dap(\d+)", s["prefix"])
            sample_daps.append(float(m.group(1)) if m else 30.0)
        sampler = DAPBucketBatchSampler(
            daps=sample_daps,
            batch_size=resolved_batch_size,
            num_buckets=args.dap_buckets,
            shuffle=True,
            world_size=world_size,
            rank=rank,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_with_pkt,
        )
    else:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
        dataloader = DataLoader(
            dataset,
            batch_size=resolved_batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_with_pkt,
        )

    if rank == 0:
        print(f"Final Configuration: Batch per GPU = {resolved_batch_size} | Global Batch = {resolved_batch_size * world_size}")

    # Finalize warmup step budget now that the DataLoader length is known
    warmup_state["total_steps"] = warmup_epochs * max(1, len(dataloader))
    if rank == 0 and warmup_state["total_steps"] > 0:
        print(f"LR warmup: {warmup_epochs} epochs ({warmup_state['total_steps']} steps), 0.1x -> 1.0x linear")

    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Training Loop
    # Fixed stratified eval set: deterministic, persisted, comparable across epochs/runs.
    eval_indices = build_eval_indices(
        dataset, samples_per_bucket=args.eval_samples_per_bucket, seed=args.eval_seed
    )
    if rank == 0:
        eval_meta = [{"index": i, "prefix": dataset.samples[i]["prefix"]} for i in eval_indices]
        eval_set_path = os.path.join(args.output_dir, "eval_set.json")
        with open(eval_set_path, "w") as f:
            json.dump({"seed": args.eval_seed, "samples_per_bucket": args.eval_samples_per_bucket,
                       "indices": eval_indices, "samples": eval_meta}, f, indent=2)
        print(f"Fixed eval set: {len(eval_indices)} samples -> {eval_set_path}")

    last_eval_time = {"t": time.time()}
    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None and hasattr(sampler, "set_epoch"):
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
            render_fraction=args.render_fraction,
            capacity_warmup_epochs=args.capacity_warmup_epochs,
            capacity_full_epochs=args.capacity_full_epochs,
            flow_granularity=args.flow_granularity,
            phytomer_vae=phytomer_vae,
            phy_count_weight=args.phy_count_weight,
            scale_weight=args.scale_weight,
            render_grad_start_epoch=args.render_grad_start_epoch,
            lr_warmup_cb=lr_warmup_cb,
        )
        lr_scheduler.step()
        # Restore the cosine-schedule lr after warmup epochs finish (warmup only
        # scales within the warmup window; cosine takes over cleanly afterwards).
        apply_warmup_scaling()

        if rank == 0:
            max_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            vram_pct = (max_vram_gb / total_vram_gb) * 100.0 if total_vram_gb > 0 else 0.0

            print(
                f"Epoch {epoch:03d} | Loss: {epoch_metrics['loss']:.4f} | "
                f"VelLoss: {epoch_metrics['fine_vel_loss']:.4f} | "
                f"AncPosLoss: {epoch_metrics['anchor_pos_loss']:.4f} | "
                f"SclLoss: {epoch_metrics.get('anchor_scale_loss', 0.0):.4f} | "
                f"PhyLoss: {epoch_metrics['phy_count_loss']:.4f} (Pred:{epoch_metrics['pred_phy_mean']:.1f}/GT:{epoch_metrics['gt_phy_mean']:.1f}) | "
                f"ExistLoss: {epoch_metrics['fine_exist_loss']:.4f} | "
                f"DepthLoss: {epoch_metrics['dense_depth_loss']:.4f} | "
                f"CosLoss: {epoch_metrics['cos_color_loss']:.4f} | "
                f"DiceLoss: {epoch_metrics['silhouette_dice_loss']:.4f} | "
                f"ClsAcc: {epoch_metrics['cls_acc']*100:.1f}% | "
                f"CapPredP: {epoch_metrics['capacity_p_pred']:.2f} | "
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
                "train/capacity_p_pred": epoch_metrics["capacity_p_pred"],
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
            # Triggered by (a) the --eval_every epoch cadence, or (b) a time-based
            # fallback: as the dataset grows, per-epoch time grows, so a pure epoch
            # cadence would silently stretch diagnostic panels to hours. The time
            # fallback forces a panel whenever --eval_min_interval_minutes elapsed.
            now = time.time()
            minutes_since_eval = (now - last_eval_time["t"]) / 60.0
            time_due = (
                args.eval_min_interval_minutes > 0
                and minutes_since_eval >= args.eval_min_interval_minutes
                and epoch > start_epoch  # skip the resume-instant eval
            )
            epoch_due = (epoch % args.eval_every == 0) or (epoch == args.epochs)
            if epoch_due or time_due:
                if time_due and not epoch_due and rank == 0:
                    print(f"  [Eval trigger] {minutes_since_eval:.0f} min since last panel (> {args.eval_min_interval_minutes})", flush=True)
                last_eval_time["t"] = time.time()
                try:
                    raw_model = model.module if hasattr(model, "module") else model
                    val_batch = collate_eval_set(dataset, eval_indices, device)
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
                        phytomer_vae=phytomer_vae,
                    )
                    print(
                        f"  [Self-Consistency Epoch {epoch:03d}] Silhouette IoU: {val_metrics['silhouette_iou']*100:.1f}% | "
                        f"Depth MAE: {val_metrics['depth_mae']*100:.2f} cm | "
                        f"Cos: {val_metrics.get('cos_color_loss', 0.0):.3f} | "
                        f"Dice: {val_metrics.get('dice_loss', 0.0):.3f} | "
                        f"Node RMSE: {val_metrics.get('val/node_rmse_cm', val_metrics.get('node_rmse_cm', 0.0)):.1f} cm "
                        f"(n={len(eval_indices)} fixed-set)",
                        flush=True,
                    )
                    wandb.log({
                        "epoch": epoch,
                        "val/silhouette_iou": val_metrics["silhouette_iou"],
                        "val/depth_mae_m": val_metrics["depth_mae"],
                        "val/cos_color_loss": val_metrics.get("cos_color_loss", 0.0),
                        "val/dice_loss": val_metrics.get("dice_loss", 0.0),
                        "val/node_rmse_cm": val_metrics.get("val/node_rmse_cm", 0.0),
                    })
                except Exception as e:
                    print(f"  [Self-Consistency Warning] Evaluation skipped: {e}", flush=True)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
