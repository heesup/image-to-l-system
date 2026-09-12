"""
Distributed Training Script for Hierarchical Matryoshka Botanical Flow Matching.

Jointly trains:
  Stage 1: Coarse Skeletal Transformer (3D phytomers, existence logits, aux DAP)
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
    estimate_phytomer_capacity,
    PHYTOMER_MARGIN,
    PHYTOMER_MARGIN_FLAT,
    build_phytomer_flow_target,
    split_phytomer_flow_target,
    apply_ref_for_flow,
    reconstruct_phytomer_rot,
    PHYTO_FLOW_BASE_START,
    PHYTO_FLOW_BASE_END,
    PHYTO_FLOW_ROLL_START,
    PHYTO_FLOW_ROLL_END,
    PHYTO_FLOW_SCALE_START,
    PHYTO_FLOW_SCALE_END,
    PHYTO_FLOW_LATENT_START,
)
from diffusion_based.models.phytomer_vae import PhytomerVAE
from diffusion_based.dataset.phytomer_packets import (
    build_phytomer_packets,
    decode_packets,
    rot6d_to_matrix,
    matrix_to_rot6d,
    assemble_packets,
    phytomer_scale,
    denormalize_packet_scales,
)
from diffusion_based.dataset.phytomer_roll import encode_roll, derive_forward
from diffusion_based.dataset.phytomer_topology import chain_phytomers
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
    slots_per_phytomer: int,
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
    dap_weight: float = 0.05,
    scale_weight: float = 2.0,
    order_weight: float = 0.5,
    stem_dir_weight: float = 0.5,
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

    prof = {"packet_build": 0.0, "fwd1": 0.0, "fwd2": 0.0, "matcher": 0.0,
            "target_build": 0.0, "render": 0.0, "loss": 0.0,
            "backward": 0.0, "probe": 0.0, "other": 0.0}
    _sync_cuda()
    _t_fbs = time.time()

    # Pre-extract visual image tokens once (reused across forwards).
    # If image encoder has no trainable parameters (e.g. frozen backbone), run under no_grad and detach
    # to guarantee zero autograd graph overhead and zero DDP reducer interference.
    encoder_has_grad = any(p.requires_grad for p in raw_model.image_encoder.parameters())
    if not encoder_has_grad:
        with torch.no_grad():
            image_tokens = raw_model.image_encoder(images).detach()
    else:
        image_tokens = raw_model.image_encoder(images)
        if image_tokens.requires_grad:
            # Defense Line: sanitize gradients flowing into DINOv2 backbone from downstream heads/rasterizer
            image_tokens.register_hook(lambda g: torch.nan_to_num(g.clamp(-1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0))

    # Matryoshka phytomer slicing directly from GT DAP (clean, deterministic, 100% stable)
    active_k = compute_matryoshka_slice(
        dap=daps, max_phytomers=raw_model.max_phytomers, margin=PHYTOMER_MARGIN
    )
    active_fine = min(active_k * slots_per_phytomer, nodes.shape[1])

    # Per-sample phytomer capacity from the calibrated DAP curve. Slots beyond a
    # sample's own capacity are excluded from GT existence targets so the
    # existence head is never penalized for not predicting organs that the
    # sliced phytomer bank cannot reach (under-allocation safety valve).
    per_sample_cap = estimate_phytomer_capacity(
        daps, margin=PHYTOMER_MARGIN, flat=PHYTOMER_MARGIN_FLAT, max_phytomers=active_k
    )  # (B,) in [8, active_k]

    nodes_sub = nodes[:, :active_fine]
    type_labels_sub = type_labels[:, :active_fine]
    existence_mask_sub = existence_mask[:, :active_fine]
    phytomer_ids_sub = None
    if phytomer_ids is not None:
        phytomer_ids_sub = phytomer_ids[:, :active_fine]
    act_mask_sub = (type_labels_sub > ORGAN_SHOOT_META) & (existence_mask_sub > 0.5)

    # Existence-aware per-sample mask: cap each sample's actives to its capacity slice
    slot_phytomer_idx = torch.arange(active_fine, device=device) // slots_per_phytomer  # (active_fine,)
    cap_mask_sub = slot_phytomer_idx.unsqueeze(0) < per_sample_cap.to(device).unsqueeze(1)  # (B, active_fine)
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
    # PHYTOMER MODE: build the 76D bridge flow target per phytomer
    #   z_1 = [ GT_phytomer_pos(3) | GT_phytomer_rot(6) | GT_phytomer_scale(3) | VAE_latent(64) ]
    # and the bridge prior x_0 = [ scaffold_pos+eps | scaffold_rot+eps | N(0,I) ].
    # The GT phytomer pose comes from the matcher's cluster centers (position)
    # and the packet reference rotation (Option 1: phytomer frame).
    # ------------------------------------------------------------------
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
                entry = {
                    "packets": pb["packets"].to(device, dtype=torch.float32),
                    "presence": pb["presence"].to(device, dtype=torch.float32),
                    "centers": pb["centers"].to(device, dtype=torch.float32),
                    "refs": pb["refs"].to(device, dtype=torch.float32),
                    "latent": pb["latent"].to(device, dtype=torch.float32),
                }
                if "keys" in pb:
                    entry["keys"] = pb["keys"].to(device)
                phyto_targets.append(entry)
                continue
            # Per-sample: build canonical packets from the GT nodes, then encode
            # each packet's latent with the frozen PhytomerVAE. The matcher's
            # phytomer_tgt_pos gives the GT phytomer base; the packet reference_rot
            # gives the GT phytomer rotation (the internode frame).
            act_b = act_mask_sub[b]
            if not act_b.any():
                phyto_targets.append(None)
                continue
            nodes_b = nodes_sub[b, act_b]
            ids_b = None
            if phytomer_ids_sub is not None:
                ids_b = phytomer_ids_sub[b, act_b]
            packets, presence, centers, refs, pkt_keys = build_phytomer_packets(
                nodes_b, existence_mask=existence_mask_sub[b, act_b],
                phytomer_ids=ids_b, return_keys=True)
            if packets.shape[0] == 0:
                phyto_targets.append(None)
                continue
            with torch.no_grad():
                lat = phytomer_vae.encode(
                    phytomer_vae.pack_input(packets.to(device), presence.to(device)))[0]
            # GT phytomer pos = cluster center (metres); GT phytomer rot = reference.
            phyto_targets.append({
                "keys": pkt_keys.to(device),
                "packets": packets.to(device),
                "presence": presence.to(device),
                "centers": centers.to(device),
                "refs": refs.to(device),
                "latent": lat,  # (P, D)
            })
        # The matcher will provide phytomer_tgt_pos per matched phytomer; we build
        # the full (B, K, 9+D) target after matching (below).
    prof["packet_build"] = time.time() - t0

    optimizer.zero_grad()

    if flow_granularity == "phytomer":
        # Bridge prior: x_0 = [scaffold_pos + eps | scaffold_rot + eps | N(0,I)(D)]
        # The scaffold pose comes from the model's Stage-2 prediction (computed
        # below in the forward pass); we build x_0 AFTER the forward so we can
        # use the predicted scaffold as the bridge init. For the first forward
        # we use a placeholder Gaussian (the velocity target is independent of
        # Decoupled standard Gaussian prior for 64D VAE latent space (Hybrid Decoupled Architecture)
        D = raw_model.phytomer_latent_dim
        z_0 = torch.randn(B, active_k, D, device=device)
        t = scheduler.sample_time(B, device)
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
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    noisy_fine_nodes=z_t,
                    timesteps=t,
                    images=images,
                    daps=daps,
                    image_tokens=image_tokens,
                )
        prof["fwd1"] = time.time() - t0
        pred_phytomer_pos = outputs["pred_phytomer_pos"].detach().float()   # (B, K, 3)
        pred_phytomer_roll = outputs["pred_phytomer_roll"].detach().float()  # (B, K, 2)
        K_eff = pred_phytomer_pos.shape[1]
        # Standard Gaussian prior in 64D VAE latent space (No bridge coupling into z_0)
        z_0 = torch.randn(B, K_eff, D, device=device)
        z_t = z_0.clone()

    # Model may resolve a wider phytomer slice than the padded GT tensor provides
    # (active_fine is clamped to nodes.shape[1] above); align the z tensors.
    K_out = int(outputs["active_k"]) if "active_k" in outputs else active_k
    if flow_granularity == "phytomer":
        active_fine = K_out * slots_per_phytomer  # legacy flat surface (K*8)
        if z_t.shape[1] != K_out:
            if z_t.shape[1] < K_out:
                pad_n = K_out - z_t.shape[1]
                z_t = torch.nn.functional.pad(z_t, (0, 0, 0, pad_n))
                z_0 = torch.nn.functional.pad(z_0, (0, 0, 0, pad_n))
    else:
        active_fine = K_out * slots_per_phytomer
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

    pred_phytomer_pos = outputs["pred_phytomer_pos"].float()              # (B, K, 3)
    pred_phytomer_logits = outputs["pred_phytomer_logits"].float()          # (B, K, 1)
    pred_velocity = outputs["pred_velocity"].float()                  # (B, active_fine, node_dim)
    pred_fine_exist_logits = outputs["pred_fine_exist_logits"].float()  # (B, active_fine, 1)
    pred_phytomer_scale = outputs.get("pred_phytomer_scale")  # (B, K, 3) or None
    if pred_phytomer_scale is not None:
        pred_phytomer_scale = pred_phytomer_scale.float()
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
        pred_phytomer_pos=pred_phytomer_pos,
        pred_phytomer_logits=pred_phytomer_logits,
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
    # PHYTOMER MODE: build the 64D VAE latent target z_1 per matched phytomer
    # with pure standard Gaussian prior z_0 ~ N(0, I_D).
    #   z_1 = VAE_latent(64)
    # GT phytomer pos = matcher cluster center; GT phytomer rot = packet reference.
    # ------------------------------------------------------------------
    if flow_granularity == "phytomer":
        D = raw_model.phytomer_latent_dim
        K_eff = pred_phytomer_pos.shape[1]
        M = raw_model.slots_per_phytomer
        tgt_z1_phyto = torch.zeros(B, K_eff, D, device=device)
        slot_presence_target = torch.zeros(B, K_eff, M, device=device)
        matched_phytomer_mask = torch.zeros(B, K_eff, dtype=torch.bool, device=device)
        phytomer_exist_targets = torch.zeros(B, K_eff, 1, device=device)
        gt_phytomer_pos_target = torch.zeros(B, K_eff, 3, device=device)
        # Roll target (2D, not the old 6D rot): the forward axis is DERIVED
        # from gt_forward_target below, not predicted, so only the roll about
        # that axis needs a supervised target -- see
        # phytomer_roll.py. There is no separate "loss_stem_dir" consistency
        # loss anymore: with the forward axis no longer independently
        # predicted, nothing can disagree with it by construction.
        gt_phytomer_roll_target = torch.zeros(B, K_eff, 2, device=device)
        gt_phytomer_scl_target = torch.zeros(B, K_eff, 3, device=device)
        # Position along the shoot, and whether this phytomer starts one.
        gt_phytomer_ord_target = torch.zeros(B, K_eff, device=device)
        gt_phytomer_base_target = torch.zeros(B, K_eff, device=device)
        # The unit forward axis encode_roll() reads the roll target off below:
        # ground-truth parent node up to this one, or for a base (no parent)
        # this node up to its successor. Built to match derive_forward()'s own
        # fallback order exactly, so the roll target stays defined relative to
        # whatever axis reconstruction will actually pair it with.
        gt_forward_target = torch.zeros(B, K_eff, 3, device=device)
        forward_mask = torch.zeros(B, K_eff, dtype=torch.bool, device=device)
        WORLD_UP = torch.tensor([0.0, 0.0, 1.0], device=device)
        # Which OTHER predicted node (if any) is this one's parent, so the
        # render-loss internode can be assembled from two PREDICTED positions
        # instead of an independently-regressed length (see phytomer_topology
        # module docstring on the redundancy). Built from the same GT-key
        # parent lookup as gt_forward_target, but only kept where the parent's
        # OWN GT packet was also matched to some predicted node this step --
        # using the model's own (possibly-wrong) predicted topology here would
        # bootstrap off unreliable early predictions; reusing the existing,
        # already-computed GT match keeps this stable at every epoch.
        gt_render_parent_idx = torch.full((B, K_eff), -1, dtype=torch.long, device=device)
        has_render_parent = torch.zeros(B, K_eff, dtype=torch.bool, device=device)
        cls_acc_data = []
        for b in range(B):
            m_b = matches[b]
            node_src = m_b["phytomer_src_idx"]
            node_tgt = m_b["phytomer_tgt_idx"]
            if len(node_src) == 0 or phyto_targets[b] is None:
                continue
            pt = phyto_targets[b]
            # GT phytomer pos from matcher cluster centers (metres)
            gt_pos = m_b["phytomer_tgt_pos"]  # (M_node, 3)
            # The matcher's node_tgt indexes its CAPACITY-CLAMPED cluster set, but
            # pt (packets) is the FULL cluster list. Match by position: each GT
            # phytomer pos corresponds to the packet whose center is nearest.
            if pt["centers"].shape[0] == 0:
                continue
            dist = torch.cdist(gt_pos.float(), pt["centers"].float())  # (M_node, P)
            pkt_idx = dist.argmin(dim=1)                               # (M_node,)
            # GT phytomer rot = packet reference rotation (Option 1 frame)
            gt_rot = pt["refs"][pkt_idx].float()    # (M_node, 6)
            gt_lat = pt["latent"][pkt_idx].float()  # (M_node, D)
            # GT phytomer scale = petiole (slot 1) scale row of the matched packets
            # (absolute FM units; the latent carries only normalized scales).
            gt_scl = phytomer_scale(pt["packets"].to(device, dtype=torch.float32))[pkt_idx].float()  # (M_node, 3)
            tgt_z1_phyto[b, node_src] = gt_lat
            slot_presence_target[b, node_src] = pt["presence"][pkt_idx].float()

            matched_phytomer_mask[b, node_src] = True
            phytomer_exist_targets[b, node_src, 0] = 1.0
            gt_phytomer_pos_target[b, node_src] = gt_pos.float()
            gt_phytomer_scl_target[b, node_src] = gt_scl.float()

            # (shoot_id, phytomer_idx) of the matched packets. Cached by
            # generate_cache.build_pkt_targets; skipped for older caches.
            if "keys" in pt:
                gt_keys = pt["keys"][pkt_idx].to(device)          # (M_node, 2)
                gt_phytomer_ord_target[b, node_src] = gt_keys[:, 1].float()
                gt_phytomer_base_target[b, node_src] = (gt_keys[:, 1] == 0).float()

                # Look up each matched packet's parent: same shoot, one step down.
                all_keys = pt["keys"].to(device)                   # (P, 2)
                want = gt_keys.clone()
                want[:, 1] = want[:, 1] - 1
                hit = (all_keys.unsqueeze(0) == want.unsqueeze(1)).all(dim=-1)  # (M, P)
                has_par = hit.any(dim=-1)
                if bool(has_par.any()):
                    par_row = hit.float().argmax(dim=-1)
                    par_pos = pt["centers"][par_row].to(device, dtype=torch.float32)
                    d = gt_pos.float() - par_pos
                    gt_forward_target[b, node_src] = F.normalize(d, dim=-1)
                    forward_mask[b, node_src] = has_par

                    # Was the parent's own GT row ALSO matched to a predicted
                    # node this step? same[m, j] = True iff phytomer m's
                    # parent row equals the GT row matched at position j.
                    same = pkt_idx.unsqueeze(0) == par_row.unsqueeze(1)   # (M_node, M_node)
                    parent_matched = same.any(dim=1) & has_par
                    if bool(parent_matched.any()):
                        j_idx = same.float().argmax(dim=1)
                        parent_node = node_src[j_idx]
                        sel = node_src[parent_matched]
                        gt_render_parent_idx[b, sel] = parent_node[parent_matched]
                        has_render_parent[b, sel] = True

                # A base has no parent, so its forward axis comes from its
                # successor -- same shoot, one step UP. Mirrors derive_forward's
                # own fallback order so the target matches the axis the roll
                # will be paired with at reconstruction time.
                want_succ = gt_keys.clone()
                want_succ[:, 1] = want_succ[:, 1] + 1
                hit_succ = (all_keys.unsqueeze(0) == want_succ.unsqueeze(1)).all(dim=-1)
                has_succ = hit_succ.any(dim=-1) & ~has_par
                if bool(has_succ.any()):
                    succ_row = hit_succ.float().argmax(dim=-1)
                    succ_pos = pt["centers"][succ_row].to(device, dtype=torch.float32)
                    gt_forward_target[b, node_src] = torch.where(
                        has_succ.unsqueeze(-1),
                        F.normalize(succ_pos - gt_pos.float(), dim=-1),
                        gt_forward_target[b, node_src])
                    forward_mask[b, node_src] |= has_succ

            # Roll target: encode_roll reads GT's own rotation relative to
            # whichever forward axis reconstruction will actually use for this
            # phytomer -- the parent direction where one exists, else the
            # successor direction, else the same WORLD_UP fallback
            # derive_forward() ends on (a single-phytomer shoot, or an older
            # cache with no "keys"/topology at all).
            fwd_for_roll = torch.where(
                forward_mask[b, node_src].unsqueeze(-1),
                gt_forward_target[b, node_src],
                WORLD_UP.expand(len(node_src), 3),
            )
            gt_phytomer_roll_target[b, node_src] = encode_roll(
                rot6d_to_matrix(gt_rot), fwd_for_roll)

            tgt_cls = pt["packets"][pkt_idx, :, :FM_OT_END].argmax(-1)
            pres = pt["presence"][pkt_idx]
            cls_acc_data.append((b, node_src, tgt_cls, pres))

        # Decoupled Flow Matching: standard normal Gaussian prior z_0 ~ N(0, I_D)
        z_0 = torch.randn(B, K_eff, D, device=device)
        z_t = scheduler.sample_xt(z_0, tgt_z1_phyto, t)
        # Re-run the model forward with the true x_t.
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
        pred_velocity = outputs["pred_velocity"].float()          # (B, K, D)
        pred_fine_exist_logits = outputs["pred_fine_exist_logits"].float()  # (B, K, M)
        pred_phytomer_pos = outputs["pred_phytomer_pos"].float()      # (B, K, 3) - live gradient to pos_head
        pred_phytomer_roll = outputs["pred_phytomer_roll"].float()    # (B, K, 2) - live gradient to roll_head
        pred_phytomer_logits = outputs["pred_phytomer_logits"].float() # (B, K, 1) - live gradient to exist_head
        pred_phytomer_scale = outputs.get("pred_phytomer_scale")
        if pred_phytomer_scale is not None:
            pred_phytomer_scale = pred_phytomer_scale.float()

        # Option B: Detached-feature render branch (shields Transformer & Backbone, trains only 3D heads)
        pred_phytomer_pos_render = outputs.get("pred_phytomer_pos_render", pred_phytomer_pos).float()
        pred_phytomer_roll_render = outputs.get("pred_phytomer_roll_render", pred_phytomer_roll).float()
        pred_phytomer_scale_render = outputs.get("pred_phytomer_scale_render", pred_phytomer_scale)
        if pred_phytomer_scale_render is not None:
            pred_phytomer_scale_render = pred_phytomer_scale_render.float()
        pred_dap = outputs["pred_dap"].float()
        pred_num_phytomers = outputs.get("pred_num_phytomers")
        soft_margin_weights = outputs.get("soft_margin_weights")
        clean_z1 = z_t + (1.0 - t.view(B, 1, 1)) * pred_velocity  # (B, K, D) in VAE latent space
        # Structural OOD guard: the 16D/64D latent is a standard Gaussian by
        # contract (z ~ N(0, I)), and the frozen VAE decoder is only invertible
        # on that manifold. An OOD latent decodes to degenerate organ geometry
        # whose vertices cross the camera near-plane (w -> 0), where the
        # rasterizer's 1/w^2 backward factor explodes fp32 even though every
        # element stays finite (Job 38237555). Bounding z1 to the support of the
        # prior makes explosive renderer Jacobians structurally impossible
        # instead of relying on post-hoc grad clamping.
        clean_z1 = clean_z1.clamp(-6.0, 6.0)
        # Velocity target: v = z_1 - z_0 in standard Gaussian space.
        tgt_velocity = tgt_z1_phyto - z_0
        total_matched_nodes = matched_phytomer_mask.sum().item()
        norm_nodes = max(total_matched_nodes, 1)
        norm_m = norm_nodes

        # Class accuracy on matched phytomers' present slots (frozen VAE decode)
        correct_cls = 0
        total_cls_slots = 0
        with torch.no_grad():
            clean_lat = clean_z1
            vae_out = phytomer_vae.decode(clean_lat.reshape(-1, D))
            pred_cls_all = vae_out["cls_logits"].argmax(-1).reshape(B, K_eff, M)  # (B, K, M)
            for b, node_src, tgt_cls_b, pres_b in cls_acc_data:
                pred_cls_m = pred_cls_all[b, node_src]
                correct_cls += ((pred_cls_m == tgt_cls_b).float() * pres_b.float()).sum().item()
                total_cls_slots += int(pres_b.sum().item())

        # 1. Fine Velocity MSE on matched phytomers (Vectorized 1-shot in 64D VAE space)
        if total_matched_nodes > 0:
            loss_fine_vel = F.mse_loss(
                pred_velocity[matched_phytomer_mask],
                tgt_velocity[matched_phytomer_mask],
                reduction="sum",
            ) / (float(D) * float(norm_m))
        else:
            loss_fine_vel = torch.tensor(0.0, device=device)

        # 2. Idle Phytomer Damping (Vectorized 1-shot per sample, properly normalized by num_idle)
        idle_mask = ~matched_phytomer_mask
        num_idle = idle_mask.sum().item()
        if num_idle > 0:
            loss_fine_vel = loss_fine_vel + (
                0.02 * (pred_velocity[idle_mask] ** 2).sum() / (float(D) * float(num_idle))
            )

        # 3. Fine Slot Existence Loss (Vectorized 1-shot across B, K, M)
        pos_w = torch.tensor([12.0], device=device)
        loss_fine_exist = F.binary_cross_entropy_with_logits(
            pred_fine_exist_logits.clamp(min=-10.0, max=10.0), slot_presence_target, pos_weight=pos_w, reduction="mean"
        )

        # 4. Phytomer Existence Loss (Vectorized 1-shot across B, K, 1)
        pos_weight_node = torch.tensor([8.0], device=device)
        loss_phytomer_exist = F.binary_cross_entropy_with_logits(
            pred_phytomer_logits.clamp(min=-10.0, max=10.0), phytomer_exist_targets, pos_weight=pos_weight_node, reduction="mean"
        )

        # 5. Phytomer Position Loss (Vectorized 1-shot)
        if total_matched_nodes > 0:
            loss_phytomer_pos = F.smooth_l1_loss(
                pred_phytomer_pos[matched_phytomer_mask],
                gt_phytomer_pos_target[matched_phytomer_mask],
                reduction="sum",
            ) / float(norm_nodes)
        else:
            loss_phytomer_pos = torch.tensor(0.0, device=device)

        # 6. Phytomer Roll Loss (Vectorized 1-shot). Not "rotation" anymore --
        # the forward axis is derived from position post-hoc (phytomer_roll.py),
        # so only the roll about it needs supervision.
        if total_matched_nodes > 0:
            loss_phytomer_roll = F.smooth_l1_loss(
                pred_phytomer_roll[matched_phytomer_mask],
                gt_phytomer_roll_target[matched_phytomer_mask],
                reduction="sum",
            ) / float(norm_nodes)
        else:
            loss_phytomer_roll = torch.tensor(0.0, device=device)

        # 7. Phytomer Scale Loss (Vectorized 1-shot)
        if pred_phytomer_scale is not None and total_matched_nodes > 0:
            loss_phytomer_scale = F.smooth_l1_loss(
                pred_phytomer_scale[matched_phytomer_mask],
                gt_phytomer_scl_target[matched_phytomer_mask],
                reduction="sum",
            ) / float(norm_nodes)
        else:
            loss_phytomer_scale = torch.tensor(0.0, device=device)

        # 7b. Position along the shoot. Supervises what the geometric chain
        # cannot resolve on its own: with 200 of 201 phytomers correctly
        # chained, the single error relocates a whole branch and costs ~50
        # points of rendered IoU, so the export needs a cue that does not
        # depend on node spacing.
        pred_ord = outputs.get("pred_phytomer_ordinal")
        pred_base_logits = outputs.get("pred_phytomer_base_logits")
        if pred_ord is not None and total_matched_nodes > 0:
            loss_phytomer_order = F.smooth_l1_loss(
                pred_ord[matched_phytomer_mask],
                gt_phytomer_ord_target[matched_phytomer_mask],
                reduction="sum",
            ) / float(norm_nodes)
            loss_phytomer_order = loss_phytomer_order + F.binary_cross_entropy_with_logits(
                pred_base_logits[matched_phytomer_mask],
                gt_phytomer_base_target[matched_phytomer_mask],
                reduction="sum",
            ) / float(norm_nodes)
        else:
            loss_phytomer_order = torch.tensor(0.0, device=device)

        # (No separate stem-direction consistency loss anymore: the forward
        # axis is derived from position, not independently predicted, so
        # nothing can disagree with it by construction -- see the roll target
        # comment above.)
        loss_stem_dir = torch.tensor(0.0, device=device)

        # Skip the organ-mode loss loop below.
        phyto_mode_done = True
    else:
        phyto_mode_done = False

    # Loss Computation
    if not phyto_mode_done:
        loss_phytomer_pos_acc = torch.tensor(0.0, device=device)
        loss_phytomer_exist_acc = torch.tensor(0.0, device=device)
        loss_fine_vel_acc = torch.tensor(0.0, device=device)
        loss_fine_exist_acc = torch.tensor(0.0, device=device)

        total_matched_nodes = 0
        total_matched_fine = 0
        correct_cls = 0
        total_cls_slots = 0

        for b in range(B):
            m_b = matches[b]
            node_src = m_b["phytomer_src_idx"]
            node_tgt = m_b["phytomer_tgt_idx"]
            fine_src = m_b["fine_src_idx"]
            fine_tgt = m_b["fine_tgt_idx"]

            num_node_m = len(node_src)
            total_matched_nodes += num_node_m
            num_fine_m = len(fine_src)
            total_matched_fine += num_fine_m

            # Stage 1 Phytomer Losses
            if num_node_m > 0:
                exist_targets = torch.zeros(active_k, 1, device=device)
                exist_targets[node_src] = 1.0
                pos_weight_node = torch.tensor([8.0], device=device)
                loss_phytomer_exist_acc += F.binary_cross_entropy_with_logits(
                    pred_phytomer_logits[b], exist_targets, pos_weight=pos_weight_node
                )

                if "phytomer_tgt_pos" in m_b and len(m_b["phytomer_tgt_pos"]) > 0:
                    p_node = pred_phytomer_pos[b, node_src]
                    t_node = m_b["phytomer_tgt_pos"]
                    loss_phytomer_pos_acc += F.smooth_l1_loss(p_node, t_node, reduction="sum")

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
        norm_nodes = max(total_matched_nodes, 1)

        loss_phytomer_pos = loss_phytomer_pos_acc / norm_nodes
        loss_phytomer_exist = loss_phytomer_exist_acc / max(B, 1)
        loss_fine_vel = loss_fine_vel_acc / norm_m
        loss_fine_exist = loss_fine_exist_acc / max(B, 1)
        # Phytomer scale and shoot-position losses are phytomer-mode only (they
        # supervise the Stage-2 heads that feed packet assembly); organ mode has
        # no packet targets to take them from.
        loss_phytomer_roll = torch.tensor(0.0, device=device)
        loss_phytomer_scale = torch.tensor(0.0, device=device)
        loss_phytomer_order = torch.tensor(0.0, device=device)
        loss_stem_dir = torch.tensor(0.0, device=device)

    # Stage 1: Macro Losses (Phytomer Count & DAP)
    loss_phy_count = torch.tensor(0.0, device=device)
    if pred_num_phytomers is not None:
        loss_phy_count = F.smooth_l1_loss(pred_num_phytomers.float(), gt_phy_counts)

    # loss_dap restored per user request with dap_weight = 0.05
    loss_dap = torch.tensor(0.0, device=device)
    if pred_dap is not None and daps is not None:
        loss_dap = F.smooth_l1_loss(pred_dap.float(), daps.float().view(-1, 1))

    # In-Loop Differentiable Optical Grounding (1-Step Clean Prediction -> Multi-Scale Pyramid Depth + Silhouette Dice)
    loss_depth = torch.tensor(0.0, device=device)
    loss_dice = torch.tensor(0.0, device=device)

    if renderer is not None and (depth_loss_weight > 0.0 or silhouette_loss_weight > 0.0):
        f = max(0.0, min(1.0, float(render_fraction)))
        n_render = max(1, min(B, int(round(B * f)))) if f > 0.0 else 0

        # Epoch gate for render: skip entirely when render gradients are off (saves ~0.7s per step during warmup)
        render_grad_on = bool(render_grad)
        if n_render > 0 and render_grad_on:
            _sync_cuda()
            t0 = time.time()
            if n_render < B:
                render_indices = torch.randperm(B, device=device)[:n_render]
            else:
                render_indices = torch.arange(B, device=device)

            loss_depth_acc = torch.tensor(0.0, device=device)
            loss_dice_acc = torch.tensor(0.0, device=device)

            pyramid_scales = [1.0, 2.0]
            num_scales = len(pyramid_scales)

            _rz = clean_z1 if render_grad_on else clean_z1.detach()
            _re = pred_fine_exist_logits if render_grad_on else pred_fine_exist_logits.detach()
            if render_grad_on and _re.requires_grad:
                _re.register_hook(lambda g: torch.nan_to_num(g.clamp(-5.0, 5.0), nan=0.0))
            if flow_granularity == "phytomer" and n_render > 0:
                _rz_render = _rz[render_indices]  # (n_render, K, D)
                lat_all = _rz_render.detach()     # Render gradients should NOT backprop through VAE latent flow
                pos_all = pred_phytomer_pos_render[render_indices] if render_grad_on else pred_phytomer_pos_render[render_indices].detach()
                # Full rotation is DERIVED for this render batch (forward axis
                # from resolved topology -- position + ordinal alone, no
                # rotation cue needed, see phytomer_topology docstring -- plus
                # this batch's predicted roll), not read off a predicted 6D
                # value: Stage 2 no longer produces one. reconstruct_phytomer_rot
                # is @no_grad internally regardless (topology resolution has no
                # gradient to give), matching this render path's pre-existing
                # behavior of detaching rotation (roll/pos-head training here
                # was never rotation's job -- "Rot supervised purely by 3D GT
                # loss" in the prior version of this comment).
                roll_all = pred_phytomer_roll_render[render_indices].detach()
                rot_all = reconstruct_phytomer_rot(
                    pos_all.detach(), roll_all,
                    pred_ord[render_indices].detach(), pred_base_logits[render_indices].detach(),
                )
                scl_all = pred_phytomer_scale_render[render_indices] if pred_phytomer_scale_render is not None else torch.ones_like(pos_all)
                if not render_grad_on:
                    scl_all = scl_all.detach()
                else:
                    if pos_all.requires_grad:
                        pos_all.register_hook(lambda g: torch.nan_to_num(g.clamp(-2.0, 2.0), nan=0.0))
                    if scl_all.requires_grad:
                        scl_all.register_hook(lambda g: torch.nan_to_num(g.clamp(-2.0, 2.0), nan=0.0))

                out_vae_all = phytomer_vae.decode(lat_all.reshape(-1, D))
                recon_abs_all = denormalize_packet_scales(
                    out_vae_all["recon_packets"], scl_all.reshape(-1, 3)
                )
                # Internode base/length/direction from two PREDICTED node
                # positions (this phytomer's and its GT-matched parent's) where
                # available, instead of an independently-regressed length that
                # can disagree with the scaffold and draw a disconnected stem
                # (see phytomer_packets.assemble_packets docstring). Falls back
                # to the decoded length for nodes with no matched parent this
                # step (shoot bases, or the parent's row wasn't matched) via
                # assemble_packets' own NaN-gated fallback.
                par_idx_r = gt_render_parent_idx[render_indices]            # (n_render, K)
                has_par_r = has_render_parent[render_indices]               # (n_render, K)
                parent_pos_r = torch.gather(
                    pos_all, dim=1,
                    index=par_idx_r.clamp(min=0).unsqueeze(-1).expand(-1, -1, 3),
                )
                parent_pos_r = torch.where(
                    has_par_r.unsqueeze(-1), parent_pos_r,
                    torch.full_like(parent_pos_r, float("nan")),
                )
                packet_hat_all = assemble_packets(
                    recon_abs_all, rot_all.reshape(-1, 6),
                    parent_pos=parent_pos_r.reshape(-1, 3),
                    centers=pos_all.reshape(-1, 3),
                )
                abs_packets_all = apply_ref_for_flow(
                    packet_hat_all.reshape(n_render, -1, M, 26), pos_all, rot_all
                )
                cls_logits_all = out_vae_all["cls_logits"].reshape(n_render, -1, M, 13)

            for b_idx in range(n_render):
                real_b = render_indices[b_idx].item()
                if flow_granularity == "phytomer":
                    flat_abs = abs_packets_all[b_idx].reshape(-1, 26)
                    cls_logits_b = cls_logits_all[b_idx].reshape(-1, 13)
                    keep = cls_logits_b.argmax(-1) > 0
                    part_14d = decode_fm(flat_abs[keep])
                    exist_b = torch.sigmoid(_re[real_b]).reshape(-1)[keep].detach()
                    probs = F.softmax(cls_logits_b, dim=-1)[keep]
                elif vae is not None:
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

                _palette = getattr(raw_model, "color_palette", None)
                if _palette is not None:
                    _palette = _palette.detach()
                mesh_dict = renderer.geo_builder.build_mesh_from_part_tensor(
                    part_14d, existence=exist_b, organ_probs=probs, device=device,
                    color_palette=_palette,
                )

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
                sample_loss_dice = torch.tensor(0.0, device=device)

                for k, s in enumerate(pyramid_scales):
                    gt_depth = images[real_b, 4 * k + 3]

                    pred_rgbd_s = pred_pyramid[float(s)]
                    pred_depth_s = pred_rgbd_s[3]

                    # 1. Masked Canopy CHM Depth Loss
                    canopy_mask = (gt_depth > 0.005) | (pred_depth_s > 0.005)
                    if canopy_mask.sum() > 0:
                        loss_d_s = F.smooth_l1_loss(pred_depth_s[canopy_mask], gt_depth[canopy_mask], beta=0.02)
                    else:
                        loss_d_s = F.smooth_l1_loss(pred_depth_s, gt_depth, beta=0.02)
                    sample_loss_depth = sample_loss_depth + loss_d_s

                    # 2. Top-View Silhouette Soft Dice Loss
                    pred_mask_soft = torch.sigmoid((pred_depth_s - 0.005) * 100.0)
                    gt_mask = (gt_depth > 0.005).float()
                    intersection = (pred_mask_soft * gt_mask).sum()
                    denominator = pred_mask_soft.sum() + gt_mask.sum()
                    loss_dice_s = 1.0 - (2.0 * intersection + 1e-4) / (denominator + 1e-4)
                    sample_loss_dice = sample_loss_dice + loss_dice_s

                loss_depth_acc = loss_depth_acc + (sample_loss_depth / num_scales)
                loss_dice_acc = loss_dice_acc + (sample_loss_dice / num_scales)

            loss_depth = torch.clamp(loss_depth_acc / max(n_render, 1), max=10.0)
            loss_dice = torch.clamp(loss_dice_acc / max(n_render, 1), max=5.0)
            prof["render"] = time.time() - t0

    # Composite Loss (8-Loss Ratified System: Macro + Scaffold + Micro Flow + Photometric)
    rot_weight = 1.0
    loss = (
        2.0 * loss_phytomer_pos
        + rot_weight * loss_phytomer_roll
        + scale_weight * loss_phytomer_scale
        + order_weight * loss_phytomer_order
        + stem_dir_weight * loss_stem_dir
        + 1.0 * loss_phytomer_exist
        + 1.0 * loss_fine_vel
        + 1.0 * loss_fine_exist
        + phy_count_weight * loss_phy_count
        + dap_weight * loss_dap
        + depth_loss_weight * loss_depth
        + silhouette_loss_weight * loss_dice
    )

    if torch.isnan(loss) or torch.isinf(loss):
        return None

    _sync_cuda()
    t_bwd = time.time()
    loss.backward()
    _sync_cuda()
    prof["backward"] = time.time() - t_bwd
    prof["other"] = max(0.0, (time.time() - _t_fbs) - sum(
        prof.get(k, 0.0) for k in
        ("packet_build", "fwd1", "fwd2", "matcher", "render", "backward", "probe")))

    cls_denom = total_cls_slots if flow_granularity == "phytomer" else norm_m
    cls_accuracy = (correct_cls / cls_denom) if cls_denom > 0 else 0.0

    return {
        "loss": loss.item(),
        "phytomer_pos_loss": loss_phytomer_pos.item(),
        "phytomer_rot_loss": loss_phytomer_roll.item(),
        "phytomer_exist_loss": loss_phytomer_exist.item(),
        "phytomer_scale_loss": loss_phytomer_scale.item(),
        "phytomer_order_loss": loss_phytomer_order.item(),
        "stem_dir_loss": loss_stem_dir.item(),
        "phy_count_loss": loss_phy_count.item(),
        "dap_loss": loss_dap.item(),
        "pred_phy_mean": pred_num_phytomers.mean().item() if pred_num_phytomers is not None else 0.0,
        "gt_phy_mean": gt_phy_counts.mean().item(),
        "pred_dap_mean": pred_dap.mean().item() if pred_dap is not None else 0.0,
        "gt_dap_mean": daps.mean().item() if daps is not None else 0.0,
        "fine_vel_loss": loss_fine_vel.item(),
        "fine_exist_loss": loss_fine_exist.item(),
        "dense_depth_loss": loss_depth.item(),
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
    dap_weight: float = 0.05,
    scale_weight: float = 2.0,
    order_weight: float = 0.5,
    stem_dir_weight: float = 0.5,
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
    slots_per_phytomer = raw_model.slots_per_phytomer
    max_fine_slots = raw_model.max_phytomers * slots_per_phytomer

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
        slots_per_phytomer=slots_per_phytomer,
        renderer=renderer,
        vae=vae,
        depth_loss_weight=depth_loss_weight,
        color_loss_weight=color_loss_weight,
        silhouette_loss_weight=silhouette_loss_weight,
        render_fraction=render_fraction,
        flow_granularity=flow_granularity,
        phytomer_vae=phytomer_vae,
        phy_count_weight=phy_count_weight,
        dap_weight=dap_weight,
        scale_weight=scale_weight,
        order_weight=order_weight,
        stem_dir_weight=stem_dir_weight,
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
    dap_weight: float = 0.05,
    scale_weight: float = 2.0,
    order_weight: float = 0.5,
    stem_dir_weight: float = 0.5,
    render_grad_start_epoch: int = 4,
    **kwargs,
) -> Dict[str, float]:
    model.train()
    # Render-gradient epoch gate: skip render pass during initial warmup (epochs < render_grad_start_epoch);
    # from render_grad_start_epoch onwards, full differentiable optical grounding is engaged.
    render_grad = bool(epoch >= render_grad_start_epoch)
    # Optional per-step LR warmup callback (set by main(); scales optimizer lrs linearly)
    lr_warmup_cb = kwargs.pop("lr_warmup_cb", None)
    total_loss = 0.0
    total_phytomer_pos_loss = 0.0
    total_phytomer_rot_loss = 0.0
    total_phytomer_exist_loss = 0.0
    total_phytomer_scale_loss = 0.0
    total_phytomer_order_loss = 0.0
    total_stem_dir_loss = 0.0
    total_phy_count_loss = 0.0
    total_dap_loss = 0.0
    total_pred_phy_mean = 0.0
    total_gt_phy_mean = 0.0
    total_pred_dap_mean = 0.0
    total_gt_dap_mean = 0.0
    total_fine_vel_loss = 0.0
    total_fine_exist_loss = 0.0
    total_dense_depth_loss = 0.0
    total_dice_loss = 0.0
    total_cls_acc = 0.0
    count = 0
    recovery_skips = 0
    canary_elem_hits = 0

    slots_per_phytomer = model.slots_per_phytomer if not hasattr(model, "module") else model.module.slots_per_phytomer

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
            slots_per_phytomer=slots_per_phytomer,
            renderer=renderer,
            vae=vae,
            depth_loss_weight=depth_loss_weight,
            color_loss_weight=color_loss_weight,
            silhouette_loss_weight=silhouette_loss_weight,
            render_fraction=render_fraction,
            flow_granularity=flow_granularity,
            phytomer_vae=phytomer_vae,
            phy_count_weight=phy_count_weight,
            dap_weight=dap_weight,
            scale_weight=scale_weight,
            order_weight=order_weight,
            stem_dir_weight=stem_dir_weight,
            render_grad=render_grad,
        )
        t_step1 = time.time()
        if step_metrics is None:
            if lr_warmup_cb is not None:
                warmup_n = getattr(lr_warmup_cb, "advance", None)
                if warmup_n is not None:
                    warmup_n()
            continue

        # Canary, not a soft clipper: clip_grad_norm_'s total norm is a sum of
        # squares in Float32, so a single finite-but-huge grad element makes
        # sum(g^2) overflow to inf on its own (fp32 max ~3.4e38, so any element
        # past ~1.8e19 does this alone) even though nothing is actually NaN/Inf
        # yet — the exact signature of Job 38237555 (grad_norm=inf, Culprit 0).
        # The earlier version of this guard clamped every element to +-100 on
        # EVERY step regardless of need, which quietly reshapes any merely-large
        # (but harmless) gradient — exactly the kind of blanket mangling that
        # hides whether a real leak still exists. This threshold is instead set
        # far above anything a healthy step ever produces and only there to
        # keep the norm computation itself well-defined; if canary_elem_hits
        # is ever nonzero, that is the real signal something is still leaking
        # upstream, not a normal event to silently absorb.
        GRAD_ELEM_CANARY = 1e15
        step_canary_hits = 0
        for p in model.parameters():
            if p.grad is not None:
                over = p.grad.abs() > GRAD_ELEM_CANARY
                if bool(over.any()):
                    step_canary_hits += int(over.sum().item())
                    p.grad.clamp_(-GRAD_ELEM_CANARY, GRAD_ELEM_CANARY)
        if step_canary_hits > 0:
            canary_elem_hits += step_canary_hits
            if rank == 0:
                print(f"  [Canary] Step {batch_idx+1}: {step_canary_hits} grad elements exceeded "
                      f"{GRAD_ELEM_CANARY:.0e} before clip_grad_norm_ -- should not happen if the "
                      f"known Stage2/Stage3 leaks stay fixed; investigate rather than trust the clamp.")

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            recovery_skips += 1
            if rank == 0:
                bad_params = []
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    nan_cnt = int(torch.isnan(p.grad).sum().item())
                    inf_cnt = int(torch.isinf(p.grad).sum().item())
                    finite_abs = p.grad[torch.isfinite(p.grad)].abs()
                    huge_cnt = int((finite_abs > 1e6).sum().item()) if finite_abs.numel() > 0 else 0
                    max_g = float(finite_abs.max().item()) if finite_abs.numel() > 0 else 0.0
                    if nan_cnt > 0 or inf_cnt > 0 or huge_cnt > 0:
                        bad_params.append(
                            f"{name} (NaN:{nan_cnt}, Inf:{inf_cnt}, Huge(>1e6):{huge_cnt}, shape:{list(p.shape)}, max:{max_g:.1e})"
                        )
                print(f"  [Recovery] Step {batch_idx+1}: grad_norm is NaN/Inf ({grad_norm}) | Culprit params ({len(bad_params)}): -> SKIPPING STEP (weights untouched, no sanitize-and-apply)")
                for bp in bad_params[:10]:
                    print(f"    -> {bp}")
            # Skip only -- deliberately NOT sanitizing (nan_to_num) and applying
            # this step anyway. That would train on a corrupted gradient; simply
            # not stepping loses one batch's update, which is the smaller cost.
            optimizer.zero_grad(set_to_none=True)
            if lr_warmup_cb is not None:
                warmup_n = getattr(lr_warmup_cb, "advance", None)
                if warmup_n is not None:
                    warmup_n()
            continue
        optimizer.step()
        t_step2 = time.time()

        if lr_warmup_cb is not None:
            warmup_n = getattr(lr_warmup_cb, "advance", None)
            if warmup_n is not None:
                warmup_n()

        total_loss += step_metrics["loss"]
        total_phytomer_pos_loss += step_metrics["phytomer_pos_loss"]
        total_phytomer_rot_loss += step_metrics.get("phytomer_rot_loss", 0.0)
        total_phytomer_exist_loss += step_metrics["phytomer_exist_loss"]
        total_phytomer_scale_loss += step_metrics.get("phytomer_scale_loss", 0.0)
        total_phytomer_order_loss += step_metrics.get("phytomer_order_loss", 0.0)
        total_stem_dir_loss += step_metrics.get("stem_dir_loss", 0.0)
        total_phy_count_loss += step_metrics.get("phy_count_loss", 0.0)
        total_dap_loss += step_metrics.get("dap_loss", 0.0)
        total_pred_phy_mean += step_metrics.get("pred_phy_mean", 0.0)
        total_gt_phy_mean += step_metrics.get("gt_phy_mean", 0.0)
        total_pred_dap_mean += step_metrics.get("pred_dap_mean", 0.0)
        total_gt_dap_mean += step_metrics.get("gt_dap_mean", 0.0)
        total_fine_vel_loss += step_metrics["fine_vel_loss"]
        total_fine_exist_loss += step_metrics["fine_exist_loss"]
        total_dense_depth_loss += step_metrics["dense_depth_loss"]
        total_dice_loss += step_metrics["silhouette_dice_loss"]
        total_cls_acc += step_metrics["cls_acc"]
        count += 1

        print_freq = max(1, len(dataloader) // 5)
        if rank == 0 and ((batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == len(dataloader)):
            p = step_metrics.get("prof", {})
            render_str = (
                f"Depth: {step_metrics['dense_depth_loss']:.4f}, Dice: {step_metrics['silhouette_dice_loss']:.4f}"
                if render_grad
                else "Off (Warmup)"
            )
            print(
                f"  [Epoch {epoch:02d}] Step {batch_idx+1:03d}/{len(dataloader):03d} | Loss: {step_metrics['loss']:.4f} | "
                f"[S1 Macro] Phy: {step_metrics.get('phy_count_loss', 0.0):.4f} [P:{step_metrics.get('pred_phy_mean', 0.0):.1f}/G:{step_metrics.get('gt_phy_mean', 0.0):.1f}], DAP: {step_metrics.get('dap_loss', 0.0):.4f} [P:{step_metrics.get('pred_dap_mean', 0.0):.1f}/G:{step_metrics.get('gt_dap_mean', 0.0):.1f}] | "
                f"[S2 Scaffold] Pos: {step_metrics['phytomer_pos_loss']:.4f}, Roll: {step_metrics.get('phytomer_rot_loss', 0.0):.4f}, Scl: {step_metrics.get('phytomer_scale_loss', 0.0):.4f}, Ord: {step_metrics.get('phytomer_order_loss', 0.0):.4f}, Ext: {step_metrics.get('phytomer_exist_loss', 0.0):.4f} | "
                f"[S3 Micro] Vel: {step_metrics['fine_vel_loss']:.4f}, Ext: {step_metrics['fine_exist_loss']:.4f}, Acc: {step_metrics['cls_acc']*100:.1f}% | "
                f"[S4 Render] {render_str} | "
                f"fwd/bwd {t_step1-t_step0:.2f}s opt {t_step2-t_step1:.2f}s | "
                f"pkt {p.get('packet_build',0):.2f} fwd1 {p.get('fwd1',0):.2f} fwd2 {p.get('fwd2',0):.2f} "
                f"match {p.get('matcher',0):.2f} render {p.get('render',0):.2f} "
                f"backward {p.get('backward',0):.2f} probe {p.get('probe',0):.2f} other {p.get('other',0):.2f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(count, 1),
        "phytomer_pos_loss": total_phytomer_pos_loss / max(count, 1),
        "phytomer_rot_loss": total_phytomer_rot_loss / max(count, 1),
        "phytomer_exist_loss": total_phytomer_exist_loss / max(count, 1),
        "phytomer_scale_loss": total_phytomer_scale_loss / max(count, 1),
        "phytomer_order_loss": total_phytomer_order_loss / max(count, 1),
        "stem_dir_loss": total_stem_dir_loss / max(count, 1),
        "phy_count_loss": total_phy_count_loss / max(count, 1),
        "dap_loss": total_dap_loss / max(count, 1),
        "pred_phy_mean": total_pred_phy_mean / max(count, 1),
        "gt_phy_mean": total_gt_phy_mean / max(count, 1),
        "pred_dap_mean": total_pred_dap_mean / max(count, 1),
        "gt_dap_mean": total_gt_dap_mean / max(count, 1),
        "fine_vel_loss": total_fine_vel_loss / max(count, 1),
        "fine_exist_loss": total_fine_exist_loss / max(count, 1),
        "dense_depth_loss": total_dense_depth_loss / max(count, 1),
        "silhouette_dice_loss": total_dice_loss / max(count, 1),
        "cls_acc": total_cls_acc / max(count, 1),
        "recovery_skips": recovery_skips,
        "canary_elem_hits": canary_elem_hits,
        "steps_seen": count + recovery_skips,
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
    parser.add_argument("--max_phytomers", type=int, default=512)
    parser.add_argument("--slots_per_phytomer", type=int, default=10, help="Slots per phytomer packet (v2: 10 = stem, petiole, 3 leaflets, peduncle, 4 repro)")
    parser.add_argument("--phytomer_locality_radius", type=float, default=None,
                        help="Soft locality radius (meters) for phytomer matching: quadratic "
                             "penalty beyond R on top of L1 position cost. Calibrated "
                             "2026-09-09: R=0.10-0.15 covers >99%% of plausible matches "
                             "(GT NN-dist median 2.5cm vs phytomer RMSE ~2cm). None = disabled "
                             "(exact legacy parity). Recommended: 0.12 with weight 50.0.")
    parser.add_argument("--phytomer_locality_weight", type=float, default=50.0,
                        help="Weight of the soft locality penalty (see --phytomer_locality_radius).")
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--vit_layers", type=int, default=8)
    parser.add_argument("--vit_heads", type=int, default=8)
    parser.add_argument("--coarse_layers", type=int, default=4)
    parser.add_argument("--fine_layers", type=int, default=6)
    parser.add_argument("--node_dim", type=int, default=16, help="Latent dimensionality for fine organ flow matching")
    parser.add_argument("--organ_vae_checkpoint", type=str, default="diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt", help="Path to pretrained OrganLatentVAE checkpoint")
    parser.add_argument("--flow_granularity", type=str, default="organ", choices=["organ", "phytomer"],
                        help="Stage-3 flow granularity: 'organ' = per-organ slots (8K x 16D, pose as conditioning); "
                             "'phytomer' = per-phytomer 12+D vector [base(3) | rot(6) | scale(3) | latent(D)] refining the scaffold pose (bridge flow).")
    parser.add_argument("--phytomer_latent_dim", type=int, default=128,
                        help="PhytomerVAE total latent dim = coarse_dim + slots_per_phytomer*residual_dim "
                             "(flow_granularity=phytomer). Must match the checkpoint.")
    parser.add_argument("--phytomer_residual_dim", type=int, default=8,
                        help="PhytomerVAE per-slot rotation-residual width. Must match the checkpoint.")
    parser.add_argument("--phytomer_vae_checkpoint", type=str, default="diffusion_based/checkpoints/phytomer_vae_v8/phytomer_vae_128d_best.pt",
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
    parser.add_argument("--render_grad_start_epoch", type=int, default=4,
                        help="Render-loop gradients activate at this epoch; before it the render "
                             "loop is bypassed completely to speed up early LR warmup.")
    parser.add_argument("--stem_dir_weight", type=float, default=0.5,
                        help="Weight on aligning the internode forward axis with the "
                             "direction from the parent node.")
    parser.add_argument("--order_weight", type=float, default=0.5,
                        help="Weight on the position-along-shoot loss (ordinal + "
                             "shoot-base flag). Supervises the cue the geometric "
                             "chain recovery cannot get from node spacing alone.")
    parser.add_argument("--scale_weight", type=float, default=2.0,
                        help="Loss weight for the Stage-2 phytomer-scale head (76D flow s_a).")
    parser.add_argument("--eval_min_interval_minutes", type=int, default=30, help="Time-based eval fallback: force an eval panel if at least this many minutes elapsed since the last one (0 disables). Keeps diagnostic cadence roughly constant as dataset size grows per-epoch time.")
    parser.add_argument("--eval_samples_per_bucket", type=int, default=2, help="Fixed stratified eval set: samples per 10-DAP bucket (2 => ~20 samples).")
    parser.add_argument("--eval_seed", type=int, default=1234, help="Deterministic seed for the fixed stratified eval set.")
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.3, help="Backbone lr = args.lr * ratio (0.3: restored 2026-09-08 value; 0.15 starved macro-head CLS features)")
    parser.add_argument("--phy_count_weight", type=float, default=2.0, help="Loss weight for phytomer-count (was 0.5 — macro head learned ~6x slower than the Sep-8 organ run)")
    parser.add_argument("--dap_weight", type=float, default=0.05, help="Loss weight for DAP prediction (auxiliary regularizer)")
    parser.add_argument("--init_phytomer_count", type=float, default=50.0,
                        help="Bias-init phy_head so pred_num starts near the dataset mean; removes the dead-phytomer existence gate at epoch 0")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Linear LR warmup epochs (0.1x -> 1.0x per step; 0 disables)")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to checkpoint to initialize weights from (strict=False)")
    parser.add_argument("--dap_buckets", type=int, default=8, help="Number of DAP buckets for capacity-homogeneous batching (0 = plain shuffle)")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="DAP-stratified subset of the dataset for fast convergence smoke tests (0 = full dataset)")
    parser.add_argument("--capacity_warmup_epochs", type=int, default=50, help="Epochs of pure GT-DAP capacity teacher forcing before predicted-phytomer capacity ramps in")
    parser.add_argument("--capacity_full_epochs", type=int, default=150, help="Epoch at which predicted-phytomer capacity path reaches p_pred=1.0 (0 = disable ramp entirely)")
    parser.add_argument("--detect_anomaly", action="store_true",
                        help="torch.autograd.set_detect_anomaly(True) — pinpoints inplace/NaN backward errors")
    parser.add_argument("--matcher_type", type=str, default="greedy", choices=["greedy", "hungarian"],
                        help="Phytomer bipartite matching algorithm: 'greedy' (GPU batched, ~0.02s) or 'hungarian' (CPU Scipy, ~0.24s)")
    parser.add_argument("--wandb_project", type=str, default="part-flow-matching")
    parser.add_argument("--wandb_run_name", type=str, default="hierarchical-matryoshka-cowpea")
    args = parser.parse_args()

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

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
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("eval/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")

    # Dataset
    dataset = PartArrayDataset(
        data_root=args.data_dir,
        max_nodes=args.max_phytomers * args.slots_per_phytomer,
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
        max_phytomers=args.max_phytomers,
        slots_per_phytomer=args.slots_per_phytomer,
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
    # phytomer-relative M-slot packet latent. Loaded once, frozen.
    phytomer_vae = None
    if args.flow_granularity == "phytomer":
        if not os.path.exists(args.phytomer_vae_checkpoint):
            raise FileNotFoundError(
                f"PhytomerVAE checkpoint not found: {args.phytomer_vae_checkpoint} "
                f"(required for --flow-granularity phytomer)")
        if rank == 0:
            print(f"Loading frozen PhytomerVAE from {args.phytomer_vae_checkpoint}...")
        phytomer_vae = PhytomerVAE(
            latent_dim=args.phytomer_latent_dim, residual_dim=args.phytomer_residual_dim,
            hidden_dim=256).to(device)
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

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr * args.backbone_lr_ratio})
    param_groups.append({"params": other_params, "lr": args.lr})
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
        slots_per_phytomer=args.slots_per_phytomer,
        phytomer_locality_radius=args.phytomer_locality_radius,
        phytomer_locality_weight=args.phytomer_locality_weight,
        matcher_type=args.matcher_type,
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
            dap_weight=args.dap_weight,
            scale_weight=args.scale_weight,
            order_weight=args.order_weight,
            stem_dir_weight=args.stem_dir_weight,
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
    # phytomer-slice capacity matches the botanical content (large Stage 3 FLOPs saving).
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
            broadcast_buffers=False,
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

        total_warmup_steps = warmup_state.get("total_steps", 0)
        print("=" * 80)
        print("Hierarchical Botanical Flow Matching — Training Configuration")
        print("=" * 80)
        print(f"{'Component / Schedule':<26} | {'Active Epochs':<20} | {'Mechanism & Description':<28}")
        print("-" * 80)
        print(f"{'1. LR Scheduling':<26} | {f'Epoch 1 ~ {args.epochs}':<20} | {f'Linear Warmup (1~{warmup_epochs}) -> Cosine Decay':<28}")
        print(f"{'2. Differentiable Render':<26} | {f'Epoch {args.render_grad_start_epoch} ~ {args.epochs}':<20} | {f'Active from Ep {args.render_grad_start_epoch} (Ep 1~{args.render_grad_start_epoch-1} bypassed)':<28}")
        print(f"{'3. Self-Consistency Eval':<26} | {f'Milestones + {args.eval_every} ep':<20} | {f'Epochs {{1, 3, 4, 5}} + Every {args.eval_every} ep':<28}")
        print("=" * 80)

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
            dap_weight=args.dap_weight,
            scale_weight=args.scale_weight,
            order_weight=args.order_weight,
            stem_dir_weight=args.stem_dir_weight,
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

            render_epoch_str = (
                f"Depth: {epoch_metrics['dense_depth_loss']:.4f}, Dice: {epoch_metrics['silhouette_dice_loss']:.4f}"
                if epoch >= args.render_grad_start_epoch
                else "Off (Warmup)"
            )
            print(
                f"Epoch {epoch:03d} | Loss: {epoch_metrics['loss']:.4f} | "
                f"[S1 Macro] Phy: {epoch_metrics['phy_count_loss']:.4f} (P:{epoch_metrics['pred_phy_mean']:.1f}/G:{epoch_metrics['gt_phy_mean']:.1f}), DAP: {epoch_metrics.get('dap_loss', 0.0):.4f} (P:{epoch_metrics.get('pred_dap_mean', 0.0):.1f}/G:{epoch_metrics.get('gt_dap_mean', 0.0):.1f}) | "
                f"[S2 Scaffold] Pos: {epoch_metrics['phytomer_pos_loss']:.4f}, Roll: {epoch_metrics.get('phytomer_rot_loss', 0.0):.4f}, Scl: {epoch_metrics.get('phytomer_scale_loss', 0.0):.4f}, Ord: {epoch_metrics.get('phytomer_order_loss', 0.0):.4f}, Ext: {epoch_metrics['phytomer_exist_loss']:.4f} | "
                f"[S3 Micro] Vel: {epoch_metrics['fine_vel_loss']:.4f}, Ext: {epoch_metrics['fine_exist_loss']:.4f}, Acc: {epoch_metrics['cls_acc']*100:.1f}% | "
                f"[S4 Render] {render_epoch_str} | "
                f"VRAM: {max_vram_gb:.1f}/{total_vram_gb:.1f} GB ({vram_pct:.1f}%)"
                + (f" | Recovery: {int(epoch_metrics.get('recovery_skips', 0))}/{int(epoch_metrics.get('steps_seen', 0))}"
                   if epoch_metrics.get('recovery_skips', 0) > 0 else "")
                + (f" | Canary: {int(epoch_metrics.get('canary_elem_hits', 0))} elems"
                   if epoch_metrics.get('canary_elem_hits', 0) > 0 else "")
            )
            wandb.log({
                "epoch": epoch,
                "train/loss": epoch_metrics["loss"],
                # Stage 1: Macro Prior
                "train/s1_phy_count_loss": epoch_metrics["phy_count_loss"],
                "train/s1_dap_loss": epoch_metrics.get("dap_loss", 0.0),
                "train/s1_pred_phy_mean": epoch_metrics["pred_phy_mean"],
                "train/s1_gt_phy_mean": epoch_metrics["gt_phy_mean"],
                "train/s1_pred_dap_mean": epoch_metrics.get("pred_dap_mean", 0.0),
                "train/s1_gt_dap_mean": epoch_metrics.get("gt_dap_mean", 0.0),
                # Stage 2: Coarse 3D Scaffold
                "train/s2_phytomer_pos_loss": epoch_metrics["phytomer_pos_loss"],
                "train/s2_phytomer_rot_loss": epoch_metrics.get("phytomer_rot_loss", 0.0),
                "train/s2_phytomer_scale_loss": epoch_metrics.get("phytomer_scale_loss", 0.0),
                "train/s2_phytomer_exist_loss": epoch_metrics["phytomer_exist_loss"],
                # Stage 3: Fine Flow Matching
                "train/s3_fine_vel_loss": epoch_metrics["fine_vel_loss"],
                "train/s3_fine_exist_loss": epoch_metrics["fine_exist_loss"],
                "train/s3_cls_acc": epoch_metrics["cls_acc"],
                # Stage 4: Differentiable Photometric
                "train/s4_dense_depth_loss": epoch_metrics["dense_depth_loss"],
                "train/s4_silhouette_dice_loss": epoch_metrics["silhouette_dice_loss"],
                # System & Metrics
                "train/vram_allocated_gb": max_vram_gb,
                "train/vram_utilization_pct": vram_pct,
                "lr": optimizer.param_groups[0]["lr"],
                # Legacy keys kept for dashboard continuity
                "train/phy_count_loss": epoch_metrics["phy_count_loss"],
                "train/phytomer_pos_loss": epoch_metrics["phytomer_pos_loss"],
                "train/fine_vel_loss": epoch_metrics["fine_vel_loss"],
            }, step=epoch)

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
            # Early milestone evals to rigorously benchmark rendering ablation (pre vs post render loss):
            early_eval_epochs = {1, 3, 4, 5}
            epoch_due = (epoch in early_eval_epochs) or (epoch % args.eval_every == 0) or (epoch == args.epochs)
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
                        f"Dice: {val_metrics.get('dice_loss', 0.0):.3f} | "
                        f"Node RMSE: {val_metrics.get('val/node_rmse_cm', val_metrics.get('node_rmse_cm', 0.0)):.1f} cm "
                        f"(n={len(eval_indices)} fixed-set)",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [Self-Consistency Warning] Evaluation skipped: {e}", flush=True)
                finally:
                    # Hard guarantee: the in-loop eval must never leave the DDP-wrapped
                    # model in eval() state for subsequent training steps.
                    model.train()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
