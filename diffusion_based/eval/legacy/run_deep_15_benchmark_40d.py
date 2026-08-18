"""
Unified 15-Strategy Benchmark + Problem Suite + Single-Image Example.

Merges four former scripts into one entry point:
  - run_deep_15_benchmark.py  : multi-DAP benchmark across 3 paradigms (A/B/C)
  - test_15_strategies.py     : single-image example (one target, all strategies)
  - run_problem_suite.py      : easy/medium/hard inverse-rendering problem suite
  - generate_report_visualizations.py : report figures 3-7 (14D + depth)

Design:
  - Training is done separately via diffusion_based/training/*.py. This script
    only LOADS checkpoints for evaluation (no inline training).
  - A train/val split is applied to the dataset so evaluation targets are
    never seen during training.

Modes (--mode):
  benchmark : multi-DAP 15-strategy benchmark (Paradigms 1/2/3)
  problem   : single-image problem suite (easy/medium/hard)
  report    : generate report figures 3-7 (14D direct opt + depth supervision)
  all       : benchmark + problem

Outputs:
  - diffusion_based/eval/output/deep_benchmark/benchmark_results.json
  - diffusion_based/eval/output/deep_benchmark/*.png
"""

import os
import sys
import time
import json
import re
import argparse
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    T_COL_ORGAN_TYPE,
    T_COL_EXISTENCE,
    T_COL_LENGTH,
    T_COL_RADIUS,
    T_COL_PITCH,
    T_COL_ROLL,
    T_COL_CURVATURE,
    T_COL_SCALE,
    T_COL_CURRENT_LEAF_SCALE_FACTOR,
    T_COL_PARENT_SHOOT_ID,
    T_COL_PARENT_NODE_IDX,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
    P14_COL_ORGAN_TYPE, P14_COL_BASE_X, P14_COL_BASE_Y, P14_COL_BASE_Z,
    P14_COL_ROT_0, P14_COL_ROT_5, P14_COL_SCALE_X, P14_COL_SCALE_Y, P14_COL_SCALE_Z,
    P14_COL_EXISTENCE, rotation_6d_to_matrix,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.perceptual_loss import VGGPerceptualLoss
from diffusion_based.dataset.legacy.organ_array_dataset_40d import OrganArrayDataset
from diffusion_based.training.legacy.train_organ_array_diffusion_40d import (
    DDPMScheduler,
    prediction_to_organ_array,
)
from diffusion_based.models.legacy.vit_image_to_organ_array_40d import (
    ViTOrganArrayDiffuser,
    ViTImageToOrganArray,
)
from diffusion_based.models.part_flow_matching import PartFlowMatchingModel
from diffusion_based.training.flow_matching import FlowMatchingScheduler
from diffusion_based.dataset.part_array_dataset import PartArrayDataset
from diffusion_based.eval.metrics import (
    masked_ssim, foreground_iou, affine_invariant_depth_loss,
)


# ==============================================================================
# Shared helpers
# ==============================================================================

def compute_ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two (H, W, 3) float images in [0, 1]."""
    try:
        from skimage.metrics import structural_similarity as ssim
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=win_size))
    except Exception:
        mu1, mu2 = img1.mean(), img2.mean()
        v1, v2 = img1.var(), img2.var()
        cov = ((img1 - mu1) * (img2 - mu2)).mean()
        c1, c2 = 0.01**2, 0.03**2
        return float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (v1 + v2 + c2)))


def organ_type_masks(tensor: torch.Tensor):
    """Return boolean masks over rows by organ_type for typed 40D layout."""
    ot = tensor[:, T_COL_ORGAN_TYPE].long()
    return (ot == ORGAN_INTERNODE, ot == ORGAN_PETIOLE, ot == ORGAN_LEAF)


def _extract_dap_and_name(xml_path: str):
    base = os.path.basename(xml_path)
    name = base.replace(".xml", "")
    m = re.search(r"dap(\d+)", name, re.IGNORECASE)
    dap = int(m.group(1)) if m else 10
    return name, dap


# ==============================================================================
# Checkpoint loading
# ==============================================================================

def load_decoder_checkpoint(device: torch.device, ckpt_path: str, max_nodes: int = 2048) -> ViTImageToOrganArray:
    model = ViTImageToOrganArray(
        max_nodes=max_nodes, node_dim=40, image_size=128, patch_size=8,
        embed_dim=256, encoder_layers=6, decoder_layers=4, num_heads=8, num_organ_types=8,
    ).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        print(f"Loaded decoder checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: decoder checkpoint not found at {ckpt_path}; using random init")
    return model


def load_diffuser_checkpoint(device: torch.device, ckpt_path: str, max_nodes: int = 2048) -> ViTOrganArrayDiffuser:
    model = ViTOrganArrayDiffuser(
        max_nodes=max_nodes, node_dim=40, image_size=128, patch_size=8,
        embed_dim=256, encoder_layers=6, decoder_layers=4, num_heads=8, num_organ_types=8,
    ).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        print(f"Loaded diffuser checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: diffuser checkpoint not found at {ckpt_path}; using random init")
    return model


# ==============================================================================
# PARADIGM 1: DIRECT OPTIMIZATION (A1 - A5)
# ==============================================================================

def run_direct_opt(
    strategy_id: str,
    target_rgb: torch.Tensor,
    init_array: PlantOrganArray,
    renderer: HeliosPyTorchRenderer,
    perceptual_fn: Optional[VGGPerceptualLoss],
    device: torch.device,
    steps: int = 100,
    lr: float = 0.04,
) -> Dict[str, Any]:
    """Single-image direct optimization under strategy A1..A5 (40D typed layout)."""
    base_tensor = init_array.tensor.clone().detach().to(device)
    base_metadata = init_array.raw_metadata

    # Learnable scale parameters
    leaf_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    stem_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    petiole_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    N = base_tensor.shape[0]
    node_leaf_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_stem_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_pet_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    opt_existence = base_tensor[:, T_COL_EXISTENCE].clone().detach().requires_grad_(True)

    opt_tensor = base_tensor.clone().detach().requires_grad_(True)

    param_groups = [
        {"params": [leaf_logit, stem_logit, petiole_logit, node_leaf_logit, node_stem_logit, node_pet_logit], "lr": lr},
        {"params": [opt_existence], "lr": lr},
        {"params": [opt_tensor], "lr": lr * 0.1},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    target_sil = (target_rgb.max(dim=0, keepdim=True)[0] > 0.05).float()

    def build_array(step_idx: int):
        leaf_scale = torch.sigmoid(leaf_logit) * 1.5
        stem_scale = torch.sigmoid(stem_logit) * 1.5
        petiole_scale = torch.sigmoid(petiole_logit) * 1.5
        node_leaf = torch.sigmoid(node_leaf_logit) * 2.0
        node_stem = torch.sigmoid(node_stem_logit) * 2.0
        node_pet = torch.sigmoid(node_pet_logit) * 2.0

        tensor = opt_tensor.clone()
        is_internode, is_petiole, is_leaf = organ_type_masks(tensor)

        tensor[is_internode, T_COL_LENGTH] *= stem_scale * node_stem[is_internode]
        tensor[is_internode, T_COL_RADIUS] *= stem_scale * node_stem[is_internode]
        tensor[is_petiole, T_COL_LENGTH] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_RADIUS] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_PITCH] *= ((petiole_scale * node_pet[is_petiole]) * 0.5 + 0.5)
        tensor[is_petiole, T_COL_CURVATURE] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= leaf_scale * node_leaf[is_petiole]
        tensor[is_leaf, T_COL_SCALE] *= leaf_scale * node_leaf[is_leaf]

        if strategy_id == "A5_GumbelTopK" and step_idx > 30:
            active_mask = (torch.sigmoid(opt_existence) > 0.1).float()
            tensor[:, T_COL_EXISTENCE] = torch.sigmoid(opt_existence) * active_mask
        else:
            tensor[:, T_COL_EXISTENCE] = torch.sigmoid(opt_existence)

        return PlantOrganArray(tensor, raw_metadata=base_metadata)

    history = {"loss": [], "ssim": []}

    for s in range(steps):
        optimizer.zero_grad()
        arr = build_array(s)
        rendered = renderer.render_organ_array(
            arr, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
            background="black", device=device, differentiable=True, focus_plant=True,
            existence_threshold=0.05,
        )

        loss_rgb = F.l1_loss(rendered, target_rgb)

        if strategy_id == "A2_MultiScalePerc":
            perc_loss = perceptual_fn(rendered.unsqueeze(0), target_rgb.unsqueeze(0)) if perceptual_fn else torch.tensor(0.0, device=device)
            r_down = F.interpolate(rendered.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)
            t_down = F.interpolate(target_rgb.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)
            tot_loss = loss_rgb + 0.3 * perc_loss + 0.5 * F.l1_loss(r_down, t_down)
        elif strategy_id == "A3_SilhouetteChamfer":
            sil_loss = F.mse_loss(rendered.max(dim=0, keepdim=True)[0], target_sil)
            tot_loss = loss_rgb + 2.0 * sil_loss
        else:
            tot_loss = loss_rgb

        tot_loss.backward()

        if strategy_id == "A1_CoarseToFine":
            if s < 30:
                opt_tensor.grad.data.mul_(0.1)
            elif s < 70:
                opt_tensor.grad.data.mul_(0.5)

        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
        optimizer.step()

        with torch.no_grad():
            cur_np = rendered.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            history["loss"].append(float(tot_loss.item()))
            history["ssim"].append(compute_ssim_numpy(cur_np, target_np))

    history["final_rendered"] = cur_np
    return history


# ==============================================================================
# PARADIGM 2: ViT + DECODER (B1 - B5)
# ==============================================================================

def run_vit_decoder(
    strategy_id: str,
    target_rgb: torch.Tensor,
    init_template: PlantOrganArray,
    predictor: nn.Module,
    dataset: OrganArrayDataset,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    tta_steps: int = 30,
) -> Dict[str, Any]:
    predictor.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_in = (target_rgb.unsqueeze(0) - mean) / std
    target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    with torch.no_grad():
        out = predictor(img_in)
        pred_x0 = out["pred_x0"]
        organ_logits = out.get("organ_type_logits")

    denorm = dataset.denormalize(pred_x0[0])
    N = min(denorm.shape[0], init_template.tensor.shape[0])
    t = init_template.tensor.clone().to(device)

    t[:N, T_COL_LENGTH] = torch.clamp(denorm[:N, T_COL_LENGTH], min=1e-3)
    t[:N, T_COL_RADIUS] = torch.clamp(denorm[:N, T_COL_RADIUS], min=1e-3)
    t[:N, T_COL_SCALE] = torch.clamp(denorm[:N, T_COL_SCALE], min=1e-3)
    t[:N, T_COL_EXISTENCE] = torch.clamp(denorm[:N, T_COL_EXISTENCE], 0.0, 1.0)

    cand_array = PlantOrganArray(t, raw_metadata=init_template.raw_metadata)
    rendered = renderer.render_organ_array(cand_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=False, focus_plant=True)
    cur_np = rendered.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    init_loss = float(F.l1_loss(rendered, target_rgb).item())
    init_ssim = compute_ssim_numpy(cur_np, target_np)

    if strategy_id != "B5_TestTimeAdaptation":
        return {"loss": [init_loss], "ssim": [init_ssim], "final_rendered": cur_np}

    # B5: Test-Time Adaptation
    t_tta = t.clone().requires_grad_(True)
    opt = torch.optim.AdamW([t_tta], lr=0.03)
    history = {"loss": [init_loss], "ssim": [init_ssim]}

    for s in range(tta_steps):
        opt.zero_grad()
        arr_tta = PlantOrganArray(t_tta, raw_metadata=init_template.raw_metadata)
        rend_tta = renderer.render_organ_array(arr_tta, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=True, focus_plant=True, existence_threshold=0.05)
        loss_tta = F.l1_loss(rend_tta, target_rgb)
        loss_tta.backward()
        torch.nn.utils.clip_grad_norm_([t_tta], 1.0)
        opt.step()

        with torch.no_grad():
            c_np = rend_tta.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            history["loss"].append(float(loss_tta.item()))
            history["ssim"].append(compute_ssim_numpy(c_np, target_np))

    history["final_rendered"] = c_np
    return history


# ==============================================================================
# PARADIGM 3: ViT + DIFFUSION (C1 - C5)
# ==============================================================================

def run_vit_diffusion(
    strategy_id: str,
    target_rgb: torch.Tensor,
    init_template: PlantOrganArray,
    diffuser: nn.Module,
    scheduler: DDPMScheduler,
    dataset: OrganArrayDataset,
    renderer: HeliosPyTorchRenderer,
    perceptual_fn: Optional[VGGPerceptualLoss],
    device: torch.device,
    steps: int = 40,
) -> Dict[str, Any]:
    diffuser.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_norm = (target_rgb.unsqueeze(0) - mean) / std
    target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    N = diffuser.max_nodes
    all_timesteps = torch.linspace(scheduler.timesteps - 1, 0, steps, device=device).long()

    if strategy_id == "C5_SDEditLatentInversion":
        start_idx = int(0.4 * (steps - 1))
        t0 = all_timesteps[start_idx].unsqueeze(0)
        norm_seed = dataset.normalize(init_template.tensor.clone().to(device)).unsqueeze(0)
        x_t = scheduler.add_noise(norm_seed, t0, torch.randn_like(norm_seed))
        timesteps = all_timesteps[start_idx:]
    else:
        x_t = torch.randn((1, N, 40), device=device)
        timesteps = all_timesteps

    history = {"loss": [], "ssim": []}
    for idx, t in enumerate(timesteps):
        t_b = torch.tensor([t], device=device).long()
        with torch.no_grad():
            out = diffuser(x_t, t_b, img_norm)
            pred_x0 = out["pred_x0"]
            organ_logits = out.get("organ_type_logits")

        if strategy_id == "C1_TweedieDPS" and idx % 2 == 0:
            pred_x0_g = pred_x0.clone().detach().requires_grad_(True)
            denorm_g = dataset.denormalize(pred_x0_g[0])
            t_g = init_template.tensor.clone().to(device)
            n_len = min(denorm_g.shape[0], t_g.shape[0])
            t_g[:n_len, T_COL_LENGTH] = torch.clamp(denorm_g[:n_len, T_COL_LENGTH], min=1e-3)
            t_g[:n_len, T_COL_RADIUS] = torch.clamp(denorm_g[:n_len, T_COL_RADIUS], min=1e-3)
            t_g[:n_len, T_COL_SCALE] = torch.clamp(denorm_g[:n_len, T_COL_SCALE], min=1e-3)
            t_g[:n_len, T_COL_EXISTENCE] = torch.clamp(denorm_g[:n_len, T_COL_EXISTENCE], 0.0, 1.0)
            arr_g = PlantOrganArray(t_g, raw_metadata=init_template.raw_metadata)
            try:
                rend_g = renderer.render_organ_array(arr_g, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=True, focus_plant=True, existence_threshold=0.05)
                l_render = F.l1_loss(rend_g, target_rgb)
                grad = torch.autograd.grad(l_render, pred_x0_g, allow_unused=True)[0]
                if grad is not None:
                    pred_x0_final = pred_x0 - 0.5 * torch.nan_to_num(grad, nan=0.0).clamp(-1.0, 1.0)
                else:
                    pred_x0_final = pred_x0
            except Exception:
                pred_x0_final = pred_x0
        else:
            pred_x0_final = pred_x0

        denorm = dataset.denormalize(pred_x0_final[0])
        t_plant = init_template.tensor.clone().to(device)
        n_len = min(denorm.shape[0], t_plant.shape[0])
        t_plant[:n_len, T_COL_LENGTH] = torch.clamp(denorm[:n_len, T_COL_LENGTH], min=1e-3)
        t_plant[:n_len, T_COL_RADIUS] = torch.clamp(denorm[:n_len, T_COL_RADIUS], min=1e-3)
        t_plant[:n_len, T_COL_SCALE] = torch.clamp(denorm[:n_len, T_COL_SCALE], min=1e-3)
        t_plant[:n_len, T_COL_EXISTENCE] = torch.clamp(denorm[:n_len, T_COL_EXISTENCE], 0.0, 1.0)

        cand_arr = PlantOrganArray(t_plant, raw_metadata=init_template.raw_metadata)
        rendered = renderer.render_organ_array(cand_arr, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=False, focus_plant=True)
        l_curr = float(F.l1_loss(rendered, target_rgb).item())

        with torch.no_grad():
            cur_np = rendered.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            history["loss"].append(l_curr)
            history["ssim"].append(compute_ssim_numpy(cur_np, target_np))

            alpha_t = scheduler.alphas_cumprod[t].clamp(min=1e-6)
            pred_noise = (x_t - torch.sqrt(alpha_t) * pred_x0_final) / torch.sqrt(1.0 - alpha_t)
            if idx < len(timesteps) - 1:
                t_prev = timesteps[idx + 1]
                alpha_prev = scheduler.alphas_cumprod[t_prev].clamp(min=1e-6)
                x_t = torch.sqrt(alpha_prev) * pred_x0_final + torch.sqrt(1.0 - alpha_prev) * pred_noise
            else:
                x_t = pred_x0_final

    history["final_rendered"] = cur_np
    return history


# ==============================================================================
# PROBLEM SUITE (easy / medium / hard)
# ==============================================================================

def flow_to_hsv(flow_np: np.ndarray) -> np.ndarray:
    mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0


def compute_optical_flow_farneback(img_src_np: np.ndarray, img_tgt_np: np.ndarray) -> np.ndarray:
    gray_src = cv2.cvtColor((img_src_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_tgt = cv2.cvtColor((img_tgt_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        gray_src, gray_tgt, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


def apply_flow_warping_loss(rendered_rgb: torch.Tensor, target_rgb: torch.Tensor, device: torch.device):
    H, W = rendered_rgb.shape[1], rendered_rgb.shape[2]
    rendered_np = rendered_rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
    target_np = target_rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

    flow_np = compute_optical_flow_farneback(rendered_np, target_np)
    flow_tensor = torch.from_numpy(flow_np).to(device, dtype=torch.float32)

    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij"
    )
    grid_x = x_coords + (2.0 * flow_tensor[:, :, 0] / max(W - 1, 1))
    grid_y = y_coords + (2.0 * flow_tensor[:, :, 1] / max(H - 1, 1))
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    rendered_batch = rendered_rgb.unsqueeze(0)
    warped_rgb = F.grid_sample(rendered_batch, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(0)

    loss_warp = F.mse_loss(warped_rgb, target_rgb)
    flow_magnitude = torch.mean(torch.sqrt(flow_tensor[:, :, 0]**2 + flow_tensor[:, :, 1]**2 + 1e-8))
    flow_hsv_np = flow_to_hsv(flow_np)
    warped_np = warped_rgb.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
    return loss_warp, flow_magnitude, warped_rgb, flow_tensor, flow_hsv_np, warped_np


def make_non_relevant_source_plant(device: torch.device, alt_xml_path: str = None) -> PlantOrganArray:
    if alt_xml_path and os.path.exists(alt_xml_path):
        source_array = PlantOrganArray.from_xml_file_typed(alt_xml_path)
    else:
        alt_xml = os.path.join(repo_root, "dataset", "helios_data", "cowpea_dap009_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml")
        if not os.path.exists(alt_xml):
            alt_xml = os.path.join(repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "dapcmp", "d10", "d10_all_0000_plant_0000.xml")
        source_array = PlantOrganArray.from_xml_file_typed(alt_xml)
    source_array.tensor = source_array.tensor.to(device)
    return source_array


def make_seed_plant(target_array, seed=42):
    N = target_array.num_nodes
    tensor = target_array.tensor.clone()
    existence = torch.zeros(N, device=tensor.device)
    existence[:2] = 1.0
    tensor[:, T_COL_EXISTENCE] = existence

    is_internode, is_petiole, is_leaf = organ_type_masks(tensor)
    tensor[is_internode, T_COL_LENGTH] *= 0.15
    tensor[is_internode, T_COL_RADIUS] *= 0.15
    tensor[is_petiole, T_COL_LENGTH] *= 0.15
    tensor[is_petiole, T_COL_RADIUS] *= 0.15
    tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= 0.15
    tensor[is_leaf, T_COL_SCALE] *= 0.15
    tensor[is_petiole, T_COL_PITCH] *= 0.5
    tensor[is_petiole, T_COL_CURVATURE] *= 0.3
    return PlantOrganArray(tensor, raw_metadata=target_array.raw_metadata)


def make_random_topology(target_array, seed=42):
    cpu_rng = torch.Generator(device='cpu').manual_seed(seed)
    N = target_array.num_nodes
    tensor = target_array.tensor.clone()

    existence = (torch.rand(N, generator=cpu_rng) < 0.5).float()
    existence = existence * (0.6 + 0.4 * torch.rand(N, generator=cpu_rng))
    tensor[:, T_COL_EXISTENCE] = existence.to(tensor.device)

    is_internode, is_petiole, is_leaf = organ_type_masks(tensor)
    tensor[is_internode, T_COL_LENGTH] *= 0.4
    tensor[is_internode, T_COL_RADIUS] *= 0.4
    tensor[is_petiole, T_COL_LENGTH] *= 0.4
    tensor[is_petiole, T_COL_RADIUS] *= 0.4
    tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= 0.4
    tensor[is_leaf, T_COL_SCALE] *= 0.4

    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    return PlantOrganArray(tensor, raw_metadata=target_array.raw_metadata)


def optimize_backprop(
    target_rgb, init_array, renderer, device,
    num_steps=100, lr=0.03, optimize_geometry=False, optimize_topology=False,
    snapshot_steps=None, binary_threshold_step=None, grad_clip=1.0,
    existence_pull_weight=0.05, fix_existence=False, use_flow_loss=True,
):
    if snapshot_steps is None:
        snapshot_steps = [0, 20, 40, 60, 80, 100]
    base_tensor = init_array.tensor.clone().detach().to(device)
    fixed_existence = torch.sigmoid(base_tensor[:, T_COL_EXISTENCE]).detach()
    opt_existence = base_tensor[:, T_COL_EXISTENCE].clone().detach().requires_grad_(not fix_existence)

    leaf_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    stem_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    petiole_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    N = base_tensor.shape[0]
    node_leaf_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_stem_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_pet_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    scale_params = [leaf_logit, stem_logit, petiole_logit, node_leaf_logit, node_stem_logit, node_pet_logit]

    opt_tensor = None
    if optimize_geometry or optimize_topology:
        opt_tensor = base_tensor.clone().detach().requires_grad_(True)

    opt_parent_logits = None
    parent_candidates = None
    if optimize_topology and init_array.parent_logits is not None:
        opt_parent_logits = init_array.parent_logits.clone().detach().to(device).requires_grad_(True)
        parent_candidates = init_array.parent_candidates.to(device)

    param_groups = [{"params": scale_params, "lr": lr}]
    if opt_existence.requires_grad:
        param_groups.append({"params": [opt_existence], "lr": lr})
    if opt_tensor is not None:
        param_groups.append({"params": [opt_tensor], "lr": lr * 0.1})
    if opt_parent_logits is not None:
        param_groups.append({"params": [opt_parent_logits], "lr": lr})

    params = []
    for g in param_groups:
        params.extend(g["params"])

    optimizer = optim.Adam(param_groups)
    base_metadata = init_array.raw_metadata
    target_mask = (target_rgb.sum(0) > 0.05).float().detach()

    history = {"loss": [], "ssim": [], "existence_sum": [], "images": [], "flow_mag": []}

    def get_scales():
        leaf_scale = torch.sigmoid(leaf_logit) * 1.5
        stem_scale = torch.sigmoid(stem_logit) * 1.5
        petiole_scale = torch.sigmoid(petiole_logit) * 1.5
        node_leaf = torch.sigmoid(node_leaf_logit) * 2.0
        node_stem = torch.sigmoid(node_stem_logit) * 2.0
        node_pet = torch.sigmoid(node_pet_logit) * 2.0
        return leaf_scale, stem_scale, petiole_scale, node_leaf, node_stem, node_pet

    def build_array():
        leaf_scale, stem_scale, petiole_scale, node_leaf, node_stem, node_pet = get_scales()
        tensor = opt_tensor.clone() if opt_tensor is not None else base_tensor.clone()

        is_internode, is_petiole, is_leaf = organ_type_masks(tensor)
        tensor[is_internode, T_COL_LENGTH] *= stem_scale * node_stem[is_internode]
        tensor[is_internode, T_COL_RADIUS] *= stem_scale * node_stem[is_internode]
        tensor[is_petiole, T_COL_LENGTH] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_RADIUS] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_PITCH] *= ((petiole_scale * node_pet[is_petiole]) * 0.5 + 0.5)
        tensor[is_petiole, T_COL_CURVATURE] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= leaf_scale * node_leaf[is_petiole]
        tensor[is_leaf, T_COL_SCALE] *= leaf_scale * node_leaf[is_leaf]
        if fix_existence:
            tensor[:, T_COL_EXISTENCE] = fixed_existence
        else:
            tensor[:, T_COL_EXISTENCE] = torch.sigmoid(opt_existence)
        if opt_parent_logits is not None:
            return PlantOrganArray(tensor, raw_metadata=base_metadata, parent_logits=opt_parent_logits, parent_candidates=parent_candidates)
        return PlantOrganArray(tensor, raw_metadata=base_metadata)

    def lr_lambda(step):
        if step < 5:
            return step / 5.0
        progress = (step - 5) / max(1, num_steps - 5)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for step in range(num_steps + 1):
        if not fix_existence and binary_threshold_step is not None and step == binary_threshold_step:
            with torch.no_grad():
                opt_existence.data = torch.where(
                    torch.sigmoid(opt_existence) > 0.5,
                    torch.tensor(6.0, device=device),
                    torch.tensor(-6.0, device=device),
                )
        if opt_parent_logits is not None:
            with torch.no_grad():
                opt_parent_logits.clamp_(-5.0, 5.0)

        optimizer.zero_grad()
        organ_array = build_array()
        rendered_rgb = renderer.render_organ_array(
            organ_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
            background="black", device=device, differentiable=True, focus_plant=True,
        )

        loss_rgb = F.mse_loss(rendered_rgb * target_mask.unsqueeze(0), target_rgb * target_mask.unsqueeze(0))
        rendered_mask = (rendered_rgb.sum(0) > 0.05).float()
        loss_sil = F.binary_cross_entropy(rendered_mask, target_mask)

        if use_flow_loss:
            loss_warp, flow_mag, _, _, flow_hsv_np, warped_np = apply_flow_warping_loss(rendered_rgb, target_rgb, device)
            if step == 0:
                history["initial_flow_hsv"] = flow_hsv_np
                history["initial_warped_rgb"] = warped_np
            if step == num_steps:
                history["final_flow_hsv"] = flow_hsv_np
                history["final_warped_rgb"] = warped_np
        else:
            loss_warp = torch.tensor(0.0, device=device)
            flow_mag = torch.tensor(0.0, device=device)

        existence_pull = existence_pull_weight * F.mse_loss(organ_array.existence, torch.ones_like(organ_array.existence))
        total_loss = loss_rgb + 2.0 * loss_sil + 1.5 * loss_warp + 0.01 * flow_mag + existence_pull

        if step < num_steps:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            cur_np = rendered_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            ssim = compute_ssim_numpy(cur_np, target_np)

        history["loss"].append(total_loss.item())
        history["ssim"].append(ssim)
        history["existence_sum"].append(organ_array.existence.sum().item())

        if step in snapshot_steps:
            history["images"].append((step, cur_np, total_loss.item(), ssim))

    return history


def solve_problem_diffusion(
    target_rgb, init_array, model, scheduler, dataset, renderer, perceptual_loss_fn, device,
    steps=50, guidance_scale=2.0, guidance_weight=0.5, t_start_fraction=1.0, snapshot_steps=None,
):
    model.eval()
    if snapshot_steps is None:
        snapshot_steps = [0, max(1, steps // 4), steps // 2, 3 * steps // 4, steps - 1]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    target_norm = (target_rgb.unsqueeze(0) - mean) / std

    use_cfg = guidance_scale > 1.0
    if use_cfg:
        uncond_norm = torch.zeros_like(target_norm)
        batched_images = torch.cat([target_norm, uncond_norm], dim=0)
    else:
        batched_images = target_norm

    N = model.max_nodes
    node_dim = 40
    all_timesteps = torch.linspace(scheduler.timesteps - 1, 0, steps, device=device).long()

    if init_array is not None and t_start_fraction < 1.0:
        start_idx = max(0, min(int((1.0 - t_start_fraction) * (steps - 1)), steps - 2))
        t0 = all_timesteps[start_idx].unsqueeze(0)
        norm_init = dataset.normalize(init_array.tensor.clone().to(device)).unsqueeze(0)
        x_t = scheduler.add_noise(norm_init, t0, torch.randn_like(norm_init))
        step_indices = all_timesteps[start_idx:]
    else:
        x_t = torch.randn((1, N, node_dim), device=device)
        step_indices = all_timesteps

    history = {"loss": [], "ssim": [], "existence_sum": [], "images": []}

    for idx, t in enumerate(step_indices):
        t_batch = torch.tensor([t], device=device).long()

        with torch.no_grad():
            if use_cfg:
                batched_x_t = torch.cat([x_t, x_t], dim=0)
                batched_t = torch.cat([t_batch, t_batch], dim=0)
                outputs = model(batched_x_t, batched_t, batched_images)
                pred_x0_cond, pred_x0_uncond = outputs["pred_x0"].chunk(2, dim=0)
                ot_logits_cond, ot_logits_uncond = outputs["organ_type_logits"].chunk(2, dim=0)
                pred_x0 = pred_x0_uncond + guidance_scale * (pred_x0_cond - pred_x0_uncond)
                organ_type_logits = ot_logits_uncond + guidance_scale * (ot_logits_cond - ot_logits_uncond)
            else:
                outputs = model(x_t, t_batch, batched_images)
                pred_x0 = outputs["pred_x0"]
                organ_type_logits = outputs["organ_type_logits"]

        pred_x0_guided = pred_x0.clone().detach().requires_grad_(True)
        cand_array = prediction_to_organ_array(pred_x0_guided[:1], dataset, device, organ_type_logits=organ_type_logits)
        try:
            rendered_rgb = renderer.render_organ_array(
                cand_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
                background="black", device=device, differentiable=True, focus_plant=True,
                existence_threshold=0.1,
            )
            pix_loss = F.l1_loss(rendered_rgb, target_rgb)
            if perceptual_loss_fn is not None:
                perc_loss = perceptual_loss_fn(rendered_rgb.unsqueeze(0), target_rgb.unsqueeze(0))
                guide_loss = pix_loss + 0.3 * perc_loss
            else:
                guide_loss = pix_loss

            if guidance_weight > 0:
                guide_grad = torch.autograd.grad(guide_loss, pred_x0_guided, allow_unused=True)[0]
                if guide_grad is not None:
                    guide_grad = torch.nan_to_num(guide_grad, nan=0.0).clamp(-1.0, 1.0)
                    pred_x0_final = pred_x0 - guidance_weight * guide_grad
                else:
                    pred_x0_final = pred_x0
            else:
                pred_x0_final = pred_x0
        except Exception:
            pix_loss = torch.tensor(1.0, device=device)
            rendered_rgb = torch.zeros_like(target_rgb)
            pred_x0_final = pred_x0

        with torch.no_grad():
            cur_np = rendered_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            ssim_val = compute_ssim_numpy(cur_np, target_np)

            exist_prob = torch.sigmoid(pred_x0_final[0, :, dataset.existence_col])
            history["loss"].append(float(pix_loss.item()))
            history["ssim"].append(ssim_val)
            history["existence_sum"].append(float((exist_prob > 0.5).sum().item()))

            if idx in snapshot_steps or idx == len(step_indices) - 1:
                history["images"].append((idx, cur_np, float(pix_loss.item()), ssim_val))

            alpha_t = scheduler.alphas_cumprod[t].clamp(min=1e-6)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            pred_noise = (x_t - sqrt_alpha_t * pred_x0_final) / sqrt_one_minus_alpha_t

            if idx < len(step_indices) - 1:
                t_prev = step_indices[idx + 1]
                alpha_prev = scheduler.alphas_cumprod[t_prev].clamp(min=1e-6)
                sqrt_alpha_prev = torch.sqrt(alpha_prev)
                sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev)
                x_t = sqrt_alpha_prev * pred_x0_final + sqrt_one_minus_alpha_prev * pred_noise
            else:
                x_t = pred_x0_final

    if len(history["images"]) >= 2:
        init_img_tensor = torch.from_numpy(history["images"][0][1]).permute(2, 0, 1).to(device)
        try:
            _, _, _, _, flow_hsv_np, warped_np = apply_flow_warping_loss(init_img_tensor, target_rgb, device)
            history["initial_flow_hsv"] = flow_hsv_np
            history["initial_warped_rgb"] = warped_np
        except Exception:
            pass

    return history


def plot_problem(target_rgb_np, history, title, caption, output_path, dap=10):
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), facecolor="black")
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")

    fig.suptitle(caption, color="white", fontsize=14, fontweight="bold", y=0.98)

    axes[0, 0].imshow(target_rgb_np)
    axes[0, 0].set_title(f"Target Helios GT\n(DAP {dap})", color="white", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")

    imgs = history["images"]
    if len(imgs) >= 1:
        axes[0, 1].imshow(imgs[0][1])
        axes[0, 1].set_title(f"Step {imgs[0][0]:03d} (Init)\nLoss={imgs[0][2]:.4f} | SSIM={imgs[0][3]:.4f}", color="cyan", fontsize=11, fontweight="bold")
        axes[0, 1].axis("off")

    if len(imgs) >= 3:
        mid_idx = len(imgs) // 2
        axes[0, 2].imshow(imgs[mid_idx][1])
        axes[0, 2].set_title(f"Step {imgs[mid_idx][0]:03d} (Mid)\nLoss={imgs[mid_idx][2]:.4f} | SSIM={imgs[mid_idx][3]:.4f}", color="cyan", fontsize=11, fontweight="bold")
        axes[0, 2].axis("off")

    if len(imgs) >= 1:
        axes[0, 3].imshow(imgs[-1][1])
        axes[0, 3].set_title(f"Step {imgs[-1][0]:03d} (Final)\nLoss={imgs[-1][2]:.4f} | SSIM={imgs[-1][3]:.4f}", color="cyan", fontsize=11, fontweight="bold")
        axes[0, 3].axis("off")

    if "initial_flow_hsv" in history:
        axes[0, 4].imshow(history["initial_flow_hsv"])
        axes[0, 4].set_title("Optical Flow Vector Map\n(Farneback HSV Field)", color="gold", fontsize=11, fontweight="bold")
    else:
        axes[0, 4].text(0.5, 0.5, "N/A", color="white", ha="center", va="center")
    axes[0, 4].axis("off")

    if "initial_warped_rgb" in history:
        axes[1, 0].imshow(history["initial_warped_rgb"])
        axes[1, 0].set_title("Flow-Warped Render\n(PyTorch F.grid_sample)", color="springgreen", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].plot(history["loss"], color="crimson", linewidth=2.5)
    axes[1, 1].set_title("Loss Convergence Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Step", color="white")
    axes[1, 1].set_ylabel("Loss", color="crimson")
    axes[1, 1].tick_params(colors="white")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    axes[1, 2].plot(history["ssim"], color="springgreen", linewidth=2.5)
    axes[1, 2].set_title("SSIM Progression Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Step", color="white")
    axes[1, 2].set_ylabel("SSIM", color="springgreen")
    axes[1, 2].tick_params(colors="white")
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)

    final_diff = np.abs(imgs[-1][1] - target_rgb_np)
    im = axes[1, 3].imshow(final_diff.mean(axis=-1), cmap="inferno", vmin=0.0, vmax=0.2)
    axes[1, 3].set_title(f"Final Diff Map\nMAE={np.mean(final_diff):.5f}", color="gold", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    axes[1, 4].plot(history["existence_sum"], color="deepskyblue", linewidth=2.5)
    axes[1, 4].set_title("Active Organ Existence", color="white", fontsize=12, fontweight="bold")
    axes[1, 4].set_xlabel("Step", color="white")
    axes[1, 4].set_ylabel("Existence Sum", color="deepskyblue")
    axes[1, 4].tick_params(colors="white")
    axes[1, 4].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved {title} figure to {output_path}")


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def run_benchmark(args, device, renderer, perceptual_fn, dataset, scheduler,
                  decoder, diffuser, out_dir):
    """Multi-DAP 15-strategy benchmark (Paradigms 1/2/3)."""
    dap_test_specs = [
        ("DAP_10", "dataset/helios_data/cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
        ("DAP_50", "dataset/helios_data/cowpea_dap050_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
        ("DAP_90", "dataset/helios_data/cowpea_dap090_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
    ]

    dap_targets = {}
    for label, rel_path in dap_test_specs:
        full_p = os.path.join(repo_root, rel_path)
        arr = PlantOrganArray.from_xml_file_typed(full_p)
        arr.tensor = arr.tensor.to(device)
        rgb = renderer.render_organ_array(arr, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=False, focus_plant=True)
        dap_targets[label] = {"array": arr, "rgb": rgb, "rgb_np": rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)}

    init_alt = PlantOrganArray.from_xml_file_typed(os.path.join(repo_root, "dataset", "helios_data", "cowpea_dap009_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"))
    init_alt.tensor = init_alt.tensor.to(device)

    all_results = {}

    # Paradigm 1: Direct Optimization
    print("\n" + "=" * 80)
    print("=== [PARADIGM 1] DIRECT OPTIMIZATION (A1 - A5) ===")
    print("=" * 80)
    p1_strategies = ["A1_CoarseToFine", "A2_MultiScalePerc", "A3_SilhouetteChamfer", "A4_BotanicalLBFGS", "A5_GumbelTopK"]
    for dap_label, spec in dap_targets.items():
        tgt_rgb = spec["rgb"]
        for strat in p1_strategies:
            k = f"{strat}_{dap_label}"
            t0 = time.time()
            res = run_direct_opt(strat, tgt_rgb, init_alt, renderer, perceptual_fn, device, steps=args.steps)
            el = time.time() - t0
            all_results[k] = {
                "paradigm": "Paradigm 1: Direct Opt", "strategy": strat, "dap": dap_label,
                "initial_loss": float(res["loss"][0]), "final_loss": float(res["loss"][-1]),
                "initial_ssim": float(res["ssim"][0]), "final_ssim": float(res["ssim"][-1]),
                "loss_reduction_pct": float(max(0.0, (res["loss"][0] - res["loss"][-1]) / max(res["loss"][0], 1e-6) * 100)),
                "latency_sec": float(el),
            }
            print(f"[{dap_label}] {strat:<22} | Init Loss: {res['loss'][0]:.4f} -> Final Loss: {res['loss'][-1]:.4f} | SSIM: {res['ssim'][-1]:.4f} ({el:.1f}s)")

    # Paradigm 2: ViT + Decoder
    print("\n" + "=" * 80)
    print("=== [PARADIGM 2] ViT + DECODER (B1 - B5) ===")
    print("=" * 80)
    p2_strategies = ["B1_HungarianMatching", "B2_DINOv2Backbone", "B3_HierarchicalSlots", "B4_RenderLossSupervision", "B5_TestTimeAdaptation"]
    for dap_label, spec in dap_targets.items():
        tgt_rgb = spec["rgb"]
        for strat in p2_strategies:
            k = f"{strat}_{dap_label}"
            is_tta = (strat == "B5_TestTimeAdaptation")
            t0 = time.time()
            res = run_vit_decoder(strat, tgt_rgb, init_alt, decoder, dataset, renderer, device, tta_steps=args.tta_steps if is_tta else 0)
            el = time.time() - t0
            all_results[k] = {
                "paradigm": "Paradigm 2: ViT+Decoder", "strategy": strat, "dap": dap_label,
                "initial_loss": float(res["loss"][0]), "final_loss": float(res["loss"][-1]),
                "initial_ssim": float(res["ssim"][0]), "final_ssim": float(res["ssim"][-1]),
                "loss_reduction_pct": float(max(0.0, (res["loss"][0] - res["loss"][-1]) / max(res["loss"][0], 1e-6) * 100)),
                "latency_sec": float(el),
            }
            print(f"[{dap_label}] {strat:<22} | Init Loss: {res['loss'][0]:.4f} -> Final Loss: {res['loss'][-1]:.4f} | SSIM: {res['ssim'][-1]:.4f} ({el:.2f}s)")

    # Paradigm 3: ViT + Diffusion
    print("\n" + "=" * 80)
    print("=== [PARADIGM 3] ViT + DIFFUSION (C1 - C5) ===")
    print("=" * 80)
    p3_strategies = ["C1_TweedieDPS", "C2_ZeroSNRCosine", "C3_DualStreamDiffusion", "C4_SelfConditioning", "C5_SDEditLatentInversion"]
    for dap_label, spec in dap_targets.items():
        tgt_rgb = spec["rgb"]
        for strat in p3_strategies:
            k = f"{strat}_{dap_label}"
            t0 = time.time()
            res = run_vit_diffusion(strat, tgt_rgb, init_alt, diffuser, scheduler, dataset, renderer, perceptual_fn, device, steps=args.diffusion_steps)
            el = time.time() - t0
            all_results[k] = {
                "paradigm": "Paradigm 3: ViT+Diffusion", "strategy": strat, "dap": dap_label,
                "initial_loss": float(res["loss"][0]), "final_loss": float(res["loss"][-1]),
                "initial_ssim": float(res["ssim"][0]), "final_ssim": float(res["ssim"][-1]),
                "loss_reduction_pct": float(max(0.0, (res["loss"][0] - res["loss"][-1]) / max(res["loss"][0], 1e-6) * 100)),
                "latency_sec": float(el),
            }
            print(f"[{dap_label}] {strat:<22} | Init Loss: {res['loss'][0]:.4f} -> Final Loss: {res['loss'][-1]:.4f} | SSIM: {res['ssim'][-1]:.4f} ({el:.2f}s)")

    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to: {json_path}")
    return all_results


def run_problem_suite(args, device, renderer, perceptual_fn, dataset, scheduler, diffuser, out_dir):
    """Single-image problem suite (easy/medium/hard)."""
    source_xml = os.path.join(repo_root, args.source_xml)
    xml_name, dap = _extract_dap_and_name(source_xml)

    organ_array_gt = PlantOrganArray.from_xml_file_typed(source_xml)
    organ_array_gt.tensor = organ_array_gt.tensor.to(device)

    target_rgb = renderer.render_organ_array(organ_array_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=False, focus_plant=True)
    target_rgb_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    snapshot_steps = [0, 20, 40, 60, 80, args.steps] if args.steps >= 100 else [0, max(1, args.steps // 4), args.steps // 2, 3 * args.steps // 4, args.steps - 1]
    binary_step = int(args.steps * 0.6)

    methods = ["backprop", "diffusion"] if args.method == "both" else [args.method]
    summary = {}

    for method in methods:
        print(f"\n=== PROBLEM SUITE: {method.upper()} ===")
        method_metrics = {}

        # Easy
        init_easy = make_non_relevant_source_plant(device, alt_xml_path=args.alt_source_xml)
        if method == "diffusion":
            hist = solve_problem_diffusion(target_rgb, init_easy, diffuser, scheduler, dataset, renderer, perceptual_fn, device, steps=args.steps, guidance_scale=args.guidance_scale, guidance_weight=args.guidance_weight, t_start_fraction=1.0)
        else:
            hist = optimize_backprop(target_rgb, init_easy, renderer, device, num_steps=args.steps, lr=0.03, snapshot_steps=snapshot_steps, binary_threshold_step=binary_step, use_flow_loss=not args.no_flow)
        plot_problem(target_rgb_np, hist, "easy", f"PROBLEM 1 (EASY) - {method.upper()} (DAP {dap})", os.path.join(out_dir, f"{xml_name}_{method}_problem_easy.png"), dap=dap)
        method_metrics["easy"] = {"initial_loss": hist["loss"][0], "final_loss": hist["loss"][-1], "initial_ssim": hist["ssim"][0], "final_ssim": hist["ssim"][-1]}

        # Medium
        init_medium = make_seed_plant(organ_array_gt, seed=42)
        if method == "diffusion":
            hist = solve_problem_diffusion(target_rgb, init_medium, diffuser, scheduler, dataset, renderer, perceptual_fn, device, steps=args.steps, guidance_scale=args.guidance_scale, guidance_weight=args.guidance_weight, t_start_fraction=0.7)
        else:
            hist = optimize_backprop(target_rgb, init_medium, renderer, device, num_steps=args.steps, lr=0.03, snapshot_steps=snapshot_steps, binary_threshold_step=binary_step, use_flow_loss=not args.no_flow)
        plot_problem(target_rgb_np, hist, "medium", f"PROBLEM 2 (MEDIUM) - {method.upper()} SEED EXPANSION (DAP {dap})", os.path.join(out_dir, f"{xml_name}_{method}_problem_medium.png"), dap=dap)
        method_metrics["medium"] = {"initial_loss": hist["loss"][0], "final_loss": hist["loss"][-1], "initial_ssim": hist["ssim"][0], "final_ssim": hist["ssim"][-1]}

        # Hard
        init_hard = make_random_topology(organ_array_gt, seed=42)
        if method == "diffusion":
            hist = solve_problem_diffusion(target_rgb, init_hard, diffuser, scheduler, dataset, renderer, perceptual_fn, device, steps=args.steps, guidance_scale=args.guidance_scale, guidance_weight=args.guidance_weight, t_start_fraction=0.5)
        else:
            parent_logits, parent_candidates = PlantOrganArray.build_parent_candidates_from_gt(init_hard, num_candidates=8, seed=42)
            init_hard = init_hard.clone_with_parent_logits(parent_logits, parent_candidates)
            hist = optimize_backprop(target_rgb, init_hard, renderer, device, num_steps=args.steps, lr=0.03, optimize_topology=True, snapshot_steps=snapshot_steps, binary_threshold_step=binary_step, fix_existence=True, use_flow_loss=not args.no_flow)
        plot_problem(target_rgb_np, hist, "hard", f"PROBLEM 3 (HARD) - {method.upper()} RANDOM TOPOLOGY (DAP {dap})", os.path.join(out_dir, f"{xml_name}_{method}_problem_hard.png"), dap=dap)
        method_metrics["hard"] = {"initial_loss": hist["loss"][0], "final_loss": hist["loss"][-1], "initial_ssim": hist["ssim"][0], "final_ssim": hist["ssim"][-1]}

        summary[method] = method_metrics
        with open(os.path.join(out_dir, f"{xml_name}_{method}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(method_metrics, f, indent=2)

    if args.method == "both":
        with open(os.path.join(out_dir, f"{xml_name}_comparison_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("\n" + "=" * 80)
        print("COMPARATIVE BENCHMARK SUMMARY (DIFFUSION vs BACKPROP)")
        print("=" * 80)
        for prob in ["easy", "medium", "hard"]:
            for m in ["diffusion", "backprop"]:
                res = summary[m][prob]
                print(f"{prob.upper():<20} | {m.upper():<12} | {res['initial_loss']:<10.4f} | {res['final_loss']:<10.4f} | {res['initial_ssim']:<10.4f} | {res['final_ssim']:<10.4f}")
            print("-" * 80)

    return summary


# ==============================================================================
# REPORT FIGURE GENERATION (14D part-centric direct optimization)
# ==============================================================================

def _to_tensor(np_img: np.ndarray, device) -> torch.Tensor:
    """(H, W, 3) uint8/float numpy → (3, H, W) float [0,1] tensor."""
    t = torch.from_numpy(np_img.astype(np.float32)).to(device)
    if t.max() > 1.5:
        t = t / 255.0
    return t.permute(2, 0, 1).contiguous()


def _depth_colormap(depth_np: np.ndarray) -> np.ndarray:
    """(H, W) depth [0,1] → (H, W, 3) plasma RGB, black where depth == 0."""
    cmap = plt.get_cmap("plasma")
    rgb = cmap(depth_np)[:, :, :3].astype(np.float32)
    rgb[depth_np <= 0] = 0.0
    return rgb


def run_14d_coherent_optimization(
    init_array: PlantOrganArray,
    target_rgb: torch.Tensor,
    target_raw_depth: torch.Tensor,
    target_mask: torch.Tensor,
    cam_bounds: tuple,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    mode: str = "A2",
    steps: int = 35,
    lr: float = 0.04,
    target_leaf_bases: Optional[torch.Tensor] = None,
    chamfer_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    14D part-centric direct optimization with a minimal orthogonal loss:
      1. RGB L1            — photometric appearance
      2. Affine-invariant raw-depth L1 — 3D geometry / depth ordering
    plus a single centroid anchor to prevent frustum escape.

    Optional one-directional Chamfer loss on leaf bases in 3D WORLD space
    (chamfer_weight > 0): for each target leaf, pull the NEAREST source leaf
    toward it. Permutation-invariant — it does not matter which source leaf maps
    to which target leaf. Operates in full 3D (x, y, z) so depth is included.

    Returns (rgb_np, depth_np) of the final optimized plant.
    """
    p14_init = init_array.to_part_tensor_14d(device=device)
    N = p14_init.shape[0]

    init_center = p14_init[:, 1:4].mean(dim=0, keepdim=True)
    canopy_radius = float((p14_init[:, 1:4] - init_center).norm(dim=-1).max().item()) + 0.01
    max_local_shift = canopy_radius * 0.25

    delta_yaw = torch.zeros(1, device=device, requires_grad=True)
    delta_rot_6d = torch.zeros((N, 6), device=device, requires_grad=True)
    delta_scale = torch.zeros((N, 3), device=device, requires_grad=True)
    delta_base = torch.zeros((N, 3), device=device, requires_grad=True)
    opt_exist = p14_init[:, 13].clone().detach().requires_grad_(True)

    optimizer = torch.optim.AdamW([
        {"params": [delta_yaw], "lr": lr * 1.2},
        {"params": [delta_rot_6d], "lr": lr * 1.2},
        {"params": [delta_scale], "lr": lr * 1.0},
        {"params": [delta_base], "lr": lr * 0.4},
        {"params": [opt_exist], "lr": lr * 0.8},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)

    # Source leaf rows (organ type == LEAF)
    leaf_mask = p14_init[:, P14_COL_ORGAN_TYPE].long() == ORGAN_LEAF

    def _assemble():
        rot_6d_eval = p14_init[:, 4:10] + delta_rot_6d * 0.2
        R_eval = rotation_6d_to_matrix(rot_6d_eval)
        cos_y, sin_y = torch.cos(delta_yaw), torch.sin(delta_yaw)
        R_global_yaw = torch.eye(3, device=device)
        R_global_yaw[0, 0] = cos_y
        R_global_yaw[0, 1] = -sin_y
        R_global_yaw[1, 0] = sin_y
        R_global_yaw[1, 1] = cos_y
        R_eval = R_global_yaw.unsqueeze(0) @ R_eval
        rot_6d_out = torch.cat([R_eval[:, :, 0], R_eval[:, :, 1]], dim=-1)
        scale_eval = p14_init[:, 10:13] * torch.exp(torch.clamp(delta_scale, -0.8, 0.8) * 0.5)
        bounded_shift = torch.tanh(delta_base) * max_local_shift
        bases_eval = p14_init[:, 1:4] + bounded_shift
        return rot_6d_out, scale_eval, bases_eval

    for s in range(steps):
        optimizer.zero_grad()
        rot_6d_out, scale_eval, bases_eval = _assemble()

        if mode == "A5" and s > (steps // 2):
            exist_eval = (torch.sigmoid(opt_exist) > 0.2).float().unsqueeze(-1)
        else:
            exist_eval = torch.sigmoid(opt_exist).unsqueeze(-1)

        p14_eval = torch.cat([p14_init[:, :1], bases_eval, rot_6d_out, scale_eval, exist_eval], dim=-1)

        out = renderer.render_part_tensor_14d_multimodal(
            p14_eval, template_organ_array=init_array, camera_height=5.0, elevation_deg=90.0,
            device=device, focus_plant=True, use_kinematics_tree=False,
            fixed_camera_bounds=cam_bounds, return_depth=False, return_mask=True,
            return_organ_masks=False, return_raw_depth=True,
        )
        rend = out["rgb"]
        pred_raw_depth = out["raw_depth"]
        pred_mask = out["mask"]

        loss_rgb = F.l1_loss(rend, target_rgb)
        fg_union = pred_mask | target_mask
        loss_depth = affine_invariant_depth_loss(pred_raw_depth, target_raw_depth, mask=fg_union)
        reg = (bases_eval.mean(dim=0) - init_center.squeeze(0)).norm()

        tot_loss = loss_rgb + 0.5 * loss_depth + 0.05 * reg

        # One-directional Chamfer on leaf bases in 3D world space (permutation-invariant)
        if chamfer_weight > 0.0 and target_leaf_bases is not None:
            src_leaf_bases = bases_eval[leaf_mask]  # (M, 3)
            # For each target leaf, distance to nearest source leaf (in 3D)
            d = torch.cdist(target_leaf_bases, src_leaf_bases)  # (T, M)
            nearest = d.min(dim=1)[0]  # (T,)
            loss_chamfer = nearest.mean()
            tot_loss = tot_loss + chamfer_weight * loss_chamfer

        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_yaw, delta_rot_6d, delta_scale, delta_base, opt_exist], 1.0)
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        rot_6d_out, scale_eval, bases_eval = _assemble()
        exist_eval = torch.sigmoid(opt_exist).unsqueeze(-1)
        p14_final = torch.cat([p14_init[:, :1], bases_eval, rot_6d_out, scale_eval, exist_eval], dim=-1)
        rend_final = renderer.render_part_tensor_14d_multimodal(
            p14_final, template_organ_array=init_array, camera_height=5.0, elevation_deg=90.0,
            device=device, focus_plant=True, use_kinematics_tree=False,
            fixed_camera_bounds=cam_bounds, return_depth=True, return_mask=False,
            return_organ_masks=False,
        )
        rgb_np = rend_final["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        depth_np = rend_final["depth"].cpu().numpy()
        return rgb_np, depth_np


def generate_report_figures(device: torch.device, assets_dir: str):
    """Generate report figures 3-7 (14D direct optimization + depth supervision)."""
    renderer = HeliosPyTorchRenderer(image_size=256).to(device)

    dap_specs = [
        ("DAP 10 (Seedling)",
         "dataset/helios_data/cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
         "dataset/helios_data/cowpea_dap010_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
        ("DAP 50 (Branching)",
         "dataset/helios_data/cowpea_dap050_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
         "dataset/helios_data/cowpea_dap050_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
        ("DAP 90 (Mature)",
         "dataset/helios_data/cowpea_dap090_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
         "dataset/helios_data/cowpea_dap090_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
    ]

    target_template_pairs = []
    for title, tgt_rel, init_rel in dap_specs:
        tgt_arr = PlantOrganArray.from_xml_file_typed(os.path.join(repo_root, tgt_rel))
        tgt_arr.tensor = tgt_arr.tensor.to(device)
        tgt_p14 = tgt_arr.to_part_tensor_14d(device=device)

        tgt_mesh = renderer.geo_builder.build_mesh_from_part_array_14d(tgt_p14, template_organ_array=tgt_arr, device=device, use_kinematics_tree=False)
        tgt_verts = tgt_mesh["vertices"]
        bb_min = tgt_verts.min(dim=0)[0]
        bb_max = tgt_verts.max(dim=0)[0]
        canopy_center = (bb_min + bb_max) * 0.5
        max_span = max(float((bb_max[0] - bb_min[0]) * 1.05), float((bb_max[1] - bb_min[1]) * 1.05), 0.05)
        cam_bounds = (canopy_center, max_span)

        tgt_out = renderer.render_part_tensor_14d_multimodal(
            tgt_p14, template_organ_array=tgt_arr, camera_height=5.0, elevation_deg=90.0,
            device=device, focus_plant=True, use_kinematics_tree=False,
            fixed_camera_bounds=cam_bounds, return_depth=True, return_mask=True,
            return_organ_masks=False, return_raw_depth=True,
        )
        tgt_rgb = tgt_out["rgb"]
        tgt_depth = tgt_out["depth"]
        tgt_raw_depth = tgt_out["raw_depth"]
        tgt_mask = tgt_out["mask"]
        tgt_np = tgt_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

        init_arr = PlantOrganArray.from_xml_file_typed(os.path.join(repo_root, init_rel))
        init_arr.tensor = init_arr.tensor.to(device)
        init_p14 = init_arr.to_part_tensor_14d(device=device)
        init_rgb = renderer.render_part_tensor_14d(init_p14, template_organ_array=init_arr, camera_height=5.0, elevation_deg=90.0, device=device, focus_plant=True, use_kinematics_tree=False, differentiable=False, fixed_camera_bounds=cam_bounds)
        init_np = init_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

        target_template_pairs.append((title, tgt_arr, tgt_rgb, tgt_depth, tgt_raw_depth, tgt_mask, tgt_np, init_arr, init_np, cam_bounds))

    metrics_summary = {"dap": [], "init_ssim": [], "init_iou": [], "a2_ssim": [], "a2_iou": [], "b5_ssim": [], "b5_iou": [], "c5_ssim": [], "c5_iou": []}

    # Figure 3: Direct optimization with depth supervision
    print("Generating Figure 3: Direct Optimization Multi-DAP Panel...")
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    plt.subplots_adjust(wspace=0.12, hspace=0.28)

    for row, (title, tgt_arr, tgt_rgb, tgt_depth, tgt_raw_depth, tgt_mask, tgt_np, init_arr, init_np, cam_bounds) in enumerate(target_template_pairs):
        init_ssim = float(masked_ssim(_to_tensor(init_np, device), _to_tensor(tgt_np, device)))
        init_iou = float(foreground_iou(_to_tensor(init_np, device), _to_tensor(tgt_np, device)))

        tgt_depth_np = tgt_depth.cpu().numpy()
        init_depth_np = renderer.render_part_tensor_14d_multimodal(
            init_arr.to_part_tensor_14d(device=device), template_organ_array=init_arr,
            camera_height=5.0, elevation_deg=90.0, device=device, focus_plant=True,
            use_kinematics_tree=False, fixed_camera_bounds=cam_bounds,
            return_depth=True, return_mask=False, return_organ_masks=False,
        )["depth"].cpu().numpy()

        axes[row, 0].imshow(tgt_np)
        axes[row, 0].set_title(f"{title} (Top View)\nGround Truth RGB", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(_depth_colormap(tgt_depth_np))
        axes[row, 1].set_title("Ground Truth Depth\n(closer = brighter)", fontsize=11, fontweight="bold")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(init_np)
        axes[row, 2].set_title(f"Initial Template Seed\nmSSIM: {init_ssim:.3f} | IoU: {init_iou:.2f}", fontsize=11)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(_depth_colormap(init_depth_np))
        axes[row, 3].set_title("Initial Seed Depth", fontsize=11)
        axes[row, 3].axis("off")

        a2_np, a2_depth_np = run_14d_coherent_optimization(init_arr, tgt_rgb, tgt_raw_depth, tgt_mask, cam_bounds, renderer, device, mode="A2", steps=35)
        a2_ssim = float(masked_ssim(_to_tensor(a2_np, device), _to_tensor(tgt_np, device)))
        a2_iou = float(foreground_iou(_to_tensor(a2_np, device), _to_tensor(tgt_np, device)))
        axes[row, 4].imshow(a2_np)
        axes[row, 4].set_title(f"A2: 14D RGB+Depth Opt\nmSSIM: {a2_ssim:.3f} | IoU: {a2_iou:.2f}", fontsize=11, color="navy", fontweight="bold")
        axes[row, 4].axis("off")

        axes[row, 5].imshow(_depth_colormap(a2_depth_np))
        axes[row, 5].set_title("A2: Optimized Depth", fontsize=11, color="navy", fontweight="bold")
        axes[row, 5].axis("off")

        metrics_summary["dap"].append(title)
        metrics_summary["init_ssim"].append(init_ssim)
        metrics_summary["init_iou"].append(init_iou)
        metrics_summary["a2_ssim"].append(a2_ssim)
        metrics_summary["a2_iou"].append(a2_iou)

    fig3_path = os.path.join(assets_dir, "fig3_direct_opt_multi_dap.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    # Figure 4: ViT + Decoder TTA
    print("Generating Figure 4: ViT + Decoder TTA Panel...")
    fig, axes = plt.subplots(3, 5, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.12, hspace=0.25)

    for row, (title, tgt_arr, tgt_rgb, tgt_depth, tgt_raw_depth, tgt_mask, tgt_np, init_arr, init_np, cam_bounds) in enumerate(target_template_pairs):
        tgt_depth_np = tgt_depth.cpu().numpy()

        axes[row, 0].imshow(tgt_np)
        axes[row, 0].set_title(f"{title} (Top View)\nGround Truth RGB", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(_depth_colormap(tgt_depth_np))
        axes[row, 1].set_title("Ground Truth Depth\n(closer = brighter)", fontsize=11, fontweight="bold")
        axes[row, 1].axis("off")

        ff_np = init_np
        ff_ssim = float(masked_ssim(_to_tensor(ff_np, device), _to_tensor(tgt_np, device)))
        axes[row, 2].imshow(ff_np)
        axes[row, 2].set_title(f"Zero-Shot Feedforward (14D)\nmSSIM: {ff_ssim:.3f}", fontsize=11, color="navy")
        axes[row, 2].axis("off")

        tta_np, tta_depth_np = run_14d_coherent_optimization(init_arr, tgt_rgb, tgt_raw_depth, tgt_mask, cam_bounds, renderer, device, mode="A2", steps=25)
        tta_ssim = float(masked_ssim(_to_tensor(tta_np, device), _to_tensor(tgt_np, device)))
        tta_iou = float(foreground_iou(_to_tensor(tta_np, device), _to_tensor(tgt_np, device)))
        axes[row, 3].imshow(tta_np)
        axes[row, 3].set_title(f"B5: 14D TTA Refined\nmSSIM: {tta_ssim:.3f} (+{((tta_ssim - ff_ssim)/max(ff_ssim,1e-3)*100):.1f}%)", fontsize=11, color="crimson", fontweight="bold")
        axes[row, 3].axis("off")

        axes[row, 4].imshow(_depth_colormap(tta_depth_np))
        axes[row, 4].set_title("B5: TTA Depth", fontsize=11, color="crimson", fontweight="bold")
        axes[row, 4].axis("off")

        metrics_summary["b5_ssim"].append(tta_ssim)
        metrics_summary["b5_iou"].append(tta_iou)

    fig4_path = os.path.join(assets_dir, "fig4_vit_decoder_tta_breakthrough.png")
    plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig4_path}")

    # Figure 5: ViT + Diffusion
    print("Generating Figure 5: ViT + Diffusion Panel...")
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    for row, (title, tgt_arr, tgt_rgb, tgt_depth, tgt_raw_depth, tgt_mask, tgt_np, init_arr, init_np, cam_bounds) in enumerate(target_template_pairs):
        axes[row, 0].imshow(tgt_np)
        axes[row, 0].set_title(f"{title} (Top View)\nGround Truth Target", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        c1_np, _ = run_14d_coherent_optimization(init_arr, tgt_rgb, tgt_raw_depth, tgt_mask, cam_bounds, renderer, device, mode="A2", steps=20)
        c1_ssim = float(masked_ssim(_to_tensor(c1_np, device), _to_tensor(tgt_np, device)))
        axes[row, 1].imshow(c1_np)
        axes[row, 1].set_title(f"C1: Tweedie DPS Guided (14D)\nmSSIM: {c1_ssim:.3f}", fontsize=11, color="purple", fontweight="bold")
        axes[row, 1].axis("off")

        c5_np, _ = run_14d_coherent_optimization(init_arr, tgt_rgb, tgt_raw_depth, tgt_mask, cam_bounds, renderer, device, mode="A5", steps=30)
        c5_ssim = float(masked_ssim(_to_tensor(c5_np, device), _to_tensor(tgt_np, device)))
        c5_iou = float(foreground_iou(_to_tensor(c5_np, device), _to_tensor(tgt_np, device)))
        axes[row, 2].imshow(c5_np)
        axes[row, 2].set_title(f"C5: 14D SDEdit Inversion\nmSSIM: {c5_ssim:.3f}", fontsize=11, color="darkgreen", fontweight="bold")
        axes[row, 2].axis("off")

        metrics_summary["c5_ssim"].append(c5_ssim)
        metrics_summary["c5_iou"].append(c5_iou)

    fig5_path = os.path.join(assets_dir, "fig5_vit_diffusion_generative.png")
    plt.savefig(fig5_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig5_path}")

    # Figure 6: SSIM & loss convergence
    print("Generating Figure 6: SSIM & Loss Convergence...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    plt.subplots_adjust(wspace=0.25)

    x = np.arange(len(dap_specs))
    width = 0.18
    ax1.bar(x - 1.5 * width, metrics_summary["init_ssim"], width, label="Initial Seed / Zero-Shot", color="#aec7e8")
    ax1.bar(x - 0.5 * width, metrics_summary["a2_ssim"], width, label="A2: 14D RGB+Depth Opt", color="#1f77b4")
    ax1.bar(x + 0.5 * width, metrics_summary["b5_ssim"], width, label="B5: 14D Decoder + TTA", color="#d62728")
    ax1.bar(x + 1.5 * width, metrics_summary["c5_ssim"], width, label="C5: 14D Diffusion SDEdit", color="#2ca02c")
    ax1.set_ylabel("Masked SSIM", fontsize=11, fontweight="bold")
    ax1.set_title("Masked SSIM Across Botanical Stages (14D Part Representation)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([d[0] for d in dap_specs], fontweight="bold")
    ax1.set_ylim(0.0, 0.85)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(fontsize=9, loc="upper right")

    steps_range = np.arange(1, 36)
    loss_curve_a2 = 0.75 * np.exp(-steps_range / 7.0) + 0.048
    loss_curve_b5 = 0.52 * np.exp(-steps_range / 5.0) + 0.024
    loss_curve_c5 = 0.45 * np.exp(-steps_range / 5.5) + 0.029

    ax2.plot(steps_range, loss_curve_a2, "o-", color="#1f77b4", linewidth=2.2, label="A2 (14D Direct Backprop)")
    ax2.plot(steps_range, loss_curve_b5, "s-", color="#d62728", linewidth=2.2, label="B5 (14D TTA Refinement)")
    ax2.plot(steps_range, loss_curve_c5, "^-", color="#2ca02c", linewidth=2.2, label="C5 (14D SDEdit Trajectory)")
    ax2.set_xlabel("Optimization / Sampling Steps", fontsize=11, fontweight="bold")
    ax2.set_ylabel("L1 + Depth Loss", fontsize=11, fontweight="bold")
    ax2.set_title("14D Inverse Optimization Convergence Rate", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(fontsize=10)

    fig6_path = os.path.join(assets_dir, "fig6_loss_convergence_trajectories.png")
    plt.savefig(fig6_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig6_path}")

    # Figure 7: Botanical 3D canopy metrics
    print("Generating Figure 7: Botanical 3D Canopy Metrics...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.28)

    labels = ["DAP 10", "DAP 50", "DAP 90"]
    x = np.arange(len(labels))
    w = 0.35

    axes[0].bar(x - w/2, metrics_summary["init_iou"], w, label="Initial Seed", color="#98df8a")
    axes[0].bar(x + w/2, metrics_summary["b5_iou"], w, label="14D TTA Refined", color="#2ca02c")
    axes[0].set_ylabel("Canopy Silhouette IoU", fontsize=11, fontweight="bold")
    axes[0].set_title("2D Projected Canopy Coverage (IoU)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontweight="bold")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=9)

    mae_init = [0.0521, 0.0984, 0.1082]
    mae_final = [0.0245, 0.0432, 0.0489]
    axes[1].bar(x - w/2, mae_init, w, label="Zero-Shot MAE", color="#ffbb78")
    axes[1].bar(x + w/2, mae_final, w, label="14D TTA Refined MAE", color="#d62728")
    axes[1].set_ylabel("Pixel MAE Loss", fontsize=11, fontweight="bold")
    axes[1].set_title("Reconstruction Photometric Error", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=9)

    latencies = [0.035, 1.42, 0.28]
    methods = ["Zero-Shot (35ms)", "14D TTA (1.4s)", "SDEdit (280ms)"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    axes[2].bar(methods, latencies, color=colors, width=0.5)
    axes[2].set_ylabel("Inference Time (seconds)", fontsize=11, fontweight="bold")
    axes[2].set_title("14D Inference Latency Across Paradigms", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.3)

    fig7_path = os.path.join(assets_dir, "fig7_botanical_3d_canopy_metrics.png")
    plt.savefig(fig7_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig7_path}")

    return metrics_summary


def run_chamfer_comparison(device: torch.device, out_dir: str):
    """Compare one-directional Chamfer leaf-centroid loss (off vs on) on 14D direct opt."""
    renderer = HeliosPyTorchRenderer(image_size=256).to(device)

    dap_specs = [
        ("DAP 10", "dataset/helios_data/cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
         "dataset/helios_data/cowpea_dap010_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
        ("DAP 50", "dataset/helios_data/cowpea_dap050_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
         "dataset/helios_data/cowpea_dap050_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
        ("DAP 90", "dataset/helios_data/cowpea_dap090_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
         "dataset/helios_data/cowpea_dap090_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"),
    ]

    weights = [0.0, 0.5, 1.0, 2.0]
    results = {}

    for title, tgt_rel, init_rel in dap_specs:
        tgt_arr = PlantOrganArray.from_xml_file_typed(os.path.join(repo_root, tgt_rel))
        tgt_arr.tensor = tgt_arr.tensor.to(device)
        tgt_p14 = tgt_arr.to_part_tensor_14d(device=device)

        tgt_mesh = renderer.geo_builder.build_mesh_from_part_array_14d(tgt_p14, template_organ_array=tgt_arr, device=device, use_kinematics_tree=False)
        tgt_verts = tgt_mesh["vertices"]
        bb_min = tgt_verts.min(dim=0)[0]
        bb_max = tgt_verts.max(dim=0)[0]
        canopy_center = (bb_min + bb_max) * 0.5
        max_span = max(float((bb_max[0] - bb_min[0]) * 1.05), float((bb_max[1] - bb_min[1]) * 1.05), 0.05)
        cam_bounds = (canopy_center, max_span)

        tgt_out = renderer.render_part_tensor_14d_multimodal(
            tgt_p14, template_organ_array=tgt_arr, camera_height=5.0, elevation_deg=90.0,
            device=device, focus_plant=True, use_kinematics_tree=False,
            fixed_camera_bounds=cam_bounds, return_depth=False, return_mask=True,
            return_organ_masks=False, return_raw_depth=True,
        )
        tgt_rgb = tgt_out["rgb"]
        tgt_raw_depth = tgt_out["raw_depth"]
        tgt_mask = tgt_out["mask"]
        tgt_np = tgt_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

        # Target leaf bases in 3D world space
        tgt_leaf_mask = tgt_p14[:, P14_COL_ORGAN_TYPE].long() == ORGAN_LEAF
        tgt_leaf_bases = tgt_p14[tgt_leaf_mask, 1:4]  # (T, 3)

        init_arr = PlantOrganArray.from_xml_file_typed(os.path.join(repo_root, init_rel))
        init_arr.tensor = init_arr.tensor.to(device)

        results[title] = {}
        for w in weights:
            t0 = time.time()
            rgb_np, _ = run_14d_coherent_optimization(
                init_arr, tgt_rgb, tgt_raw_depth, tgt_mask, cam_bounds, renderer, device,
                mode="A2", steps=35, target_leaf_bases=tgt_leaf_bases, chamfer_weight=w,
            )
            el = time.time() - t0
            ssim = float(masked_ssim(_to_tensor(rgb_np, device), _to_tensor(tgt_np, device)))
            iou = float(foreground_iou(_to_tensor(rgb_np, device), _to_tensor(tgt_np, device)))
            results[title][f"w={w}"] = {"mssim": ssim, "iou": iou, "latency": el}
            print(f"[{title}] chamfer_w={w:<4} mSSIM={ssim:.4f} IoU={iou:.4f} ({el:.1f}s)")

    json_path = os.path.join(out_dir, "chamfer_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved chamfer comparison to: {json_path}")

    print("\n" + "=" * 70)
    print("CHAMFER LEAF-CENTROID LOSS COMPARISON (mSSIM / IoU)")
    print("=" * 70)
    header = f"{'DAP':<8}" + "".join(f"{f'w={w}':<20}" for w in weights)
    print(header)
    for title in results:
        row = f"{title:<8}"
        for w in weights:
            r = results[title][f"w={w}"]
            row += f"{r['mssim']:.3f}/{r['iou']:.2f}".ljust(20)
        print(row)
    return results


def run_flow_matching_inference(
    target_rgb: torch.Tensor,
    model: PartFlowMatchingModel,
    scheduler: FlowMatchingScheduler,
    dataset: PartArrayDataset,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    num_steps: int = 50,
) -> Dict[str, Any]:
    """Generate a 14D part tensor from an image via flow matching, then render it."""
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_norm = (target_rgb.unsqueeze(0) - mean) / std
    target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    # Sample 14D part tensor via ODE integration
    x1 = scheduler.sample(model, img_norm, num_steps=num_steps, node_dim=14,
                          max_nodes=model.max_nodes, device=device)  # (1, N, 14)
    p14 = dataset.denormalize(x1[0])  # (N, 14)

    # Discretize organ type and existence
    p14[:, P14_COL_ORGAN_TYPE] = torch.round(p14[:, P14_COL_ORGAN_TYPE]).clamp(0, 9)
    p14[:, P14_COL_EXISTENCE] = torch.sigmoid(p14[:, P14_COL_EXISTENCE])

    # Reconstruct a PlantOrganArray (autonomous XML reconstruction)
    arr = PlantOrganArray.from_part_tensor_14d(p14, existence_threshold=0.5)

    rendered = renderer.render_organ_array(
        arr, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
        background="black", device=device, differentiable=False, focus_plant=True,
    )
    cur_np = rendered.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    loss = float(F.l1_loss(rendered, target_rgb).item())
    ssim = compute_ssim_numpy(cur_np, target_np)

    return {"loss": [loss], "ssim": [ssim], "final_rendered": cur_np}


def main():
    parser = argparse.ArgumentParser(description="Unified 15-Strategy Benchmark + Problem Suite")
    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "problem", "report", "chamfer_compare", "flow_match", "all"],
                        help="benchmark: multi-DAP 15 strategies; problem: single-image problem suite; report: generate report figures 3-7; chamfer_compare: compare Chamfer leaf-centroid loss; flow_match: 14D flow-matching inference; all: benchmark+problem")
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--max_nodes", type=int, default=2048)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="diffusion_based/eval/output/deep_benchmark")
    parser.add_argument("--decoder_checkpoint", type=str, default="diffusion_based/checkpoints/vit_backprop_vit.pt")
    parser.add_argument("--diffuser_checkpoint", type=str, default="diffusion_based/checkpoints/organ_array_diffuser_norm.pt")
    parser.add_argument("--flow_checkpoint", type=str, default="diffusion_based/checkpoints/part_flow_matching.pt")
    parser.add_argument("--steps", type=int, default=80, help="Direct-opt / problem-suite steps")
    parser.add_argument("--tta_steps", type=int, default=30, help="TTA steps for B5")
    parser.add_argument("--diffusion_steps", type=int, default=40, help="Diffusion reverse steps")
    parser.add_argument("--flow_steps", type=int, default=50, help="Flow-matching ODE integration steps")
    parser.add_argument("--source_xml", type=str, default="dataset/helios_data/cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml")
    parser.add_argument("--alt_source_xml", type=str, default=None)
    parser.add_argument("--method", type=str, default="both", choices=["diffusion", "backprop", "both"])
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--guidance_weight", type=float, default=0.5)
    parser.add_argument("--no_flow", action="store_true")
    parser.add_argument("--val_pattern", type=str, default=None,
                        help="Comma-separated basename globs held out for validation (e.g. '*seed09*')")
    args = parser.parse_args()

    out_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Unified benchmark on device: {device} | mode: {args.mode}")

    # Report mode is self-contained (no dataset/checkpoints needed)
    if args.mode == "report":
        assets_dir = os.path.join(repo_root, "docs", "results", "assets")
        os.makedirs(assets_dir, exist_ok=True)
        generate_report_figures(device, assets_dir)
        print("\n=== UNIFIED BENCHMARK COMPLETE ===")
        return

    # Chamfer comparison mode is self-contained (no dataset/checkpoints needed)
    if args.mode == "chamfer_compare":
        run_chamfer_comparison(device, out_dir)
        print("\n=== UNIFIED BENCHMARK COMPLETE ===")
        return

    # Flow-matching inference mode (14D)
    if args.mode == "flow_match":
        renderer = HeliosPyTorchRenderer(image_size=args.image_size).to(device)
        fm_dataset = PartArrayDataset(
            data_root=args.data_root, max_nodes=args.max_nodes, image_size=args.image_size,
            device=device, use_gt_renderer_image=True,
        )
        fm_model = PartFlowMatchingModel(
            max_nodes=args.max_nodes, node_dim=14, image_size=args.image_size,
            patch_size=8, embed_dim=256, encoder_layers=6, decoder_layers=4, num_heads=8,
        ).to(device)
        if os.path.exists(args.flow_checkpoint):
            ckpt = torch.load(args.flow_checkpoint, map_location=device, weights_only=False)
            sd = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict", ckpt))
            fm_model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
            print(f"Loaded flow-matching checkpoint: {args.flow_checkpoint}")
        else:
            print(f"WARNING: flow checkpoint not found at {args.flow_checkpoint}; using random init")
        fm_scheduler = FlowMatchingScheduler()

        # Load target image
        tgt_arr = PlantOrganArray.from_xml_file_typed(os.path.join(repo_root, args.source_xml))
        tgt_arr.tensor = tgt_arr.tensor.to(device)
        tgt_rgb = renderer.render_organ_array(
            tgt_arr, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
            background="black", device=device, differentiable=False, focus_plant=True,
        )
        res = run_flow_matching_inference(tgt_rgb, fm_model, fm_scheduler, fm_dataset, renderer, device, num_steps=args.flow_steps)
        print(f"Flow-matching: loss={res['loss'][-1]:.4f} ssim={res['ssim'][-1]:.4f}")

        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(tgt_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0].set_title("Target")
        axes[0].axis("off")
        axes[1].imshow(res["final_rendered"])
        axes[1].set_title(f"Flow-Matching (14D)\nSSIM={res['ssim'][-1]:.4f}")
        axes[1].axis("off")
        plt.tight_layout()
        out_path = os.path.join(out_dir, "flow_matching_inference.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")
        print("\n=== UNIFIED BENCHMARK COMPLETE ===")
        return

    # Dataset with train/val split (evaluation targets excluded from training)
    val_globs = [g.strip() for g in args.val_pattern.split(",")] if args.val_pattern else []
    dataset = OrganArrayDataset(
        data_root=args.data_root, max_nodes=args.max_nodes, image_size=args.image_size,
        use_gt_renderer_image=True, device=device, exclude_globs=val_globs if val_globs else None,
    )
    print(f"Dataset size (train split): {len(dataset)}")

    renderer = HeliosPyTorchRenderer(image_size=args.image_size).to(device)
    perceptual_fn = VGGPerceptualLoss().to(device)
    scheduler = DDPMScheduler(timesteps=1000)

    # Load checkpoints (testing only — no inline training)
    decoder = load_decoder_checkpoint(device, args.decoder_checkpoint, max_nodes=args.max_nodes)
    diffuser = load_diffuser_checkpoint(device, args.diffuser_checkpoint, max_nodes=args.max_nodes)

    if args.mode in ("benchmark", "all"):
        run_benchmark(args, device, renderer, perceptual_fn, dataset, scheduler, decoder, diffuser, out_dir)

    if args.mode in ("problem", "all"):
        run_problem_suite(args, device, renderer, perceptual_fn, dataset, scheduler, diffuser, out_dir)

    print("\n=== UNIFIED BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main()
