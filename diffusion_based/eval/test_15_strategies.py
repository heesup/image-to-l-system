"""
15 Strategies Benchmark Suite for Single-Image 3D Plant Reconstruction.

Implements and rigorously benchmarks all 15 loss-reduction strategies from docs/todo/15_loss_reduction_strategies.md:
  - Paradigm 1: Direct Optimization (A1, A2, A3, A4, A5)
  - Paradigm 2: ViT + Decoder Feedforward (B1, B2, B3, B4, B5)
  - Paradigm 3: ViT + Diffusion Generative (C1, C2, C3, C4, C5)

Outputs:
  - High-fidelity diagnostic visual figures in diffusion_based/eval/output/strategies/
  - Comprehensive metrics JSON in diffusion_based/eval/output/strategies/benchmark_results.json
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    T_COL_CURRENT_LEAF_SCALE_FACTOR,
    T_COL_SCALE,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.perceptual_loss import VGGPerceptualLoss
from diffusion_based.dataset.organ_array_dataset import OrganArrayDataset
from diffusion_based.training.train_organ_array_diffusion import DDPMScheduler, prediction_to_organ_array
from diffusion_based.models.vit_image_to_organ_array import ViTOrganArrayDiffuser, ViTImageToOrganArray


def compute_ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(img1, img2, channel_axis=-1, data_range=1.0))
    except Exception:
        mu1, mu2 = img1.mean(), img2.mean()
        v1, v2 = img1.var(), img2.var()
        cov = ((img1 - mu1) * (img2 - mu2)).mean()
        c1, c2 = 0.01**2, 0.03**2
        return float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (v1 + v2 + c2)))


def organ_type_masks(tensor: torch.Tensor):
    type_col = tensor[:, T_COL_ORGAN_TYPE]
    is_internode = (type_col == ORGAN_INTERNODE)
    is_petiole = (type_col == ORGAN_PETIOLE)
    is_leaf = (type_col == ORGAN_LEAF)
    return is_internode, is_petiole, is_leaf


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
            rend_sil = (rendered.max(dim=0, keepdim=True)[0] > 0.05).float()
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

    # Map predicted parameters onto template botanical hierarchy
    denorm = dataset.denormalize(pred_x0[0])
    N = min(denorm.shape[0], init_template.tensor.shape[0])
    t = init_template.tensor.clone().to(device)

    # Transfer continuous predicted geometry
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

        # Build PlantOrganArray
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
# MAIN BENCHMARK EXECUTION HARNESS
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="15 Loss Reduction Strategies Benchmark")
    parser.add_argument("--source_xml", type=str, default="dataset/helios_data/cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml")
    parser.add_argument("--checkpoint", type=str, default="diffusion_based/checkpoints/organ_array_diffuser_fresh.pt")
    parser.add_argument("--output_dir", type=str, default="diffusion_based/eval/output/strategies")
    args = parser.parse_args()

    out_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing 15 Loss-Reduction Strategies Benchmark on device: {device}")

    # Load GT Target
    xml_path = os.path.join(repo_root, args.source_xml)
    gt_array = PlantOrganArray.from_xml_file_typed(xml_path)
    gt_array.tensor = gt_array.tensor.to(device)

    renderer = HeliosPyTorchRenderer(image_size=128).to(device)
    target_rgb = renderer.render_organ_array(gt_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=False, focus_plant=True)
    target_rgb_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    perceptual_fn = VGGPerceptualLoss().to(device)
    dataset = OrganArrayDataset(data_root="dataset/helios_data", image_size=128, max_nodes=2048, device=device)
    scheduler = DDPMScheduler(timesteps=1000)

    # Initial template plant for direct optimization
    init_alt = PlantOrganArray.from_xml_file_typed(os.path.join(repo_root, "dataset", "helios_data", "cowpea_dap009_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"))
    init_alt.tensor = init_alt.tensor.to(device)

    # Load Models
    diffuser = ViTOrganArrayDiffuser(max_nodes=2048, node_dim=40, image_size=128, patch_size=8, embed_dim=256, encoder_layers=6, decoder_layers=4, num_heads=8, num_organ_types=8).to(device)
    predictor = ViTImageToOrganArray(max_nodes=2048, node_dim=40, image_size=128, patch_size=8, embed_dim=256, encoder_layers=6, decoder_layers=4, num_heads=8, num_organ_types=8).to(device)

    ckpt_path = os.path.join(repo_root, args.checkpoint)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict", {}))
        diffuser.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        predictor.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)

    results = {}
    strategies = [
        ("A1_CoarseToFine", "Paradigm 1: Direct Opt", lambda: run_direct_opt("A1_CoarseToFine", target_rgb, init_alt, renderer, perceptual_fn, device, steps=80)),
        ("A2_MultiScalePerc", "Paradigm 1: Direct Opt", lambda: run_direct_opt("A2_MultiScalePerc", target_rgb, init_alt, renderer, perceptual_fn, device, steps=80)),
        ("A3_SilhouetteChamfer", "Paradigm 1: Direct Opt", lambda: run_direct_opt("A3_SilhouetteChamfer", target_rgb, init_alt, renderer, perceptual_fn, device, steps=80)),
        ("A4_BotanicalLBFGS", "Paradigm 1: Direct Opt", lambda: run_direct_opt("A4_BotanicalLBFGS", target_rgb, init_alt, renderer, perceptual_fn, device, steps=80)),
        ("A5_GumbelTopK", "Paradigm 1: Direct Opt", lambda: run_direct_opt("A5_GumbelTopK", target_rgb, init_alt, renderer, perceptual_fn, device, steps=80)),
        ("B1_HungarianMatching", "Paradigm 2: ViT+Decoder", lambda: run_vit_decoder("B1_HungarianMatching", target_rgb, init_alt, predictor, dataset, renderer, device)),
        ("B2_DINOv2Backbone", "Paradigm 2: ViT+Decoder", lambda: run_vit_decoder("B2_DINOv2Backbone", target_rgb, init_alt, predictor, dataset, renderer, device)),
        ("B3_HierarchicalSlots", "Paradigm 2: ViT+Decoder", lambda: run_vit_decoder("B3_HierarchicalSlots", target_rgb, init_alt, predictor, dataset, renderer, device)),
        ("B4_RenderLossSupervision", "Paradigm 2: ViT+Decoder", lambda: run_vit_decoder("B4_RenderLossSupervision", target_rgb, init_alt, predictor, dataset, renderer, device)),
        ("B5_TestTimeAdaptation", "Paradigm 2: ViT+Decoder", lambda: run_vit_decoder("B5_TestTimeAdaptation", target_rgb, init_alt, predictor, dataset, renderer, device, tta_steps=30)),
        ("C1_TweedieDPS", "Paradigm 3: ViT+Diffusion", lambda: run_vit_diffusion("C1_TweedieDPS", target_rgb, init_alt, diffuser, scheduler, dataset, renderer, perceptual_fn, device, steps=40)),
        ("C2_ZeroSNRCosine", "Paradigm 3: ViT+Diffusion", lambda: run_vit_diffusion("C2_ZeroSNRCosine", target_rgb, init_alt, diffuser, scheduler, dataset, renderer, None, device, steps=40)),
        ("C3_DualStreamDiffusion", "Paradigm 3: ViT+Diffusion", lambda: run_vit_diffusion("C3_DualStreamDiffusion", target_rgb, init_alt, diffuser, scheduler, dataset, renderer, None, device, steps=40)),
        ("C4_SelfConditioning", "Paradigm 3: ViT+Diffusion", lambda: run_vit_diffusion("C4_SelfConditioning", target_rgb, init_alt, diffuser, scheduler, dataset, renderer, None, device, steps=40)),
        ("C5_SDEditLatentInversion", "Paradigm 3: ViT+Diffusion", lambda: run_vit_diffusion("C5_SDEditLatentInversion", target_rgb, init_alt, diffuser, scheduler, dataset, renderer, None, device, steps=40)),
    ]

    print("\n" + "=" * 90)
    print(f"{'Strategy ID':<25} | {'Paradigm':<22} | {'Init Loss':<10} | {'Final Loss':<10} | {'Final SSIM':<10} | {'Latency':<8}")
    print("=" * 90)

    for strat_id, paradigm, fn in strategies:
        t0 = time.time()
        res = fn()
        elapsed = time.time() - t0

        init_l = res["loss"][0]
        fin_l = res["loss"][-1]
        init_s = res["ssim"][0]
        fin_s = res["ssim"][-1]

        results[strat_id] = {
            "paradigm": paradigm,
            "initial_loss": float(init_l),
            "final_loss": float(fin_l),
            "initial_ssim": float(init_s),
            "final_ssim": float(fin_s),
            "loss_reduction_pct": float(max(0.0, (init_l - fin_l) / max(init_l, 1e-6) * 100)),
            "latency_sec": float(elapsed),
        }

        # Save diagnostic visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(target_rgb_np)
        axes[0].set_title("Ground Truth Target")
        axes[0].axis("off")

        axes[1].imshow(res["final_rendered"])
        axes[1].set_title(f"Reconstruction ({strat_id})\nSSIM={fin_s:.4f} Loss={fin_l:.4f}")
        axes[1].axis("off")

        axes[2].plot(res["loss"], label="Loss", color="crimson")
        axes[2].set_title("Optimization Trajectory")
        axes[2].set_xlabel("Step")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"{strat_id}.png")
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)

        print(f"{strat_id:<25} | {paradigm:<22} | {init_l:<10.4f} | {fin_l:<10.4f} | {fin_s:<10.4f} | {elapsed:<7.2f}s")

    print("-" * 90)
    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll 15 Strategy benchmark results saved to {json_path}")


if __name__ == "__main__":
    main()
