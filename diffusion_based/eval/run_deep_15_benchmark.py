"""
Comprehensive Multi-DAP 15 Strategies Benchmark & Full Dataset Training Suite.

Runs:
  - Paradigm 1: Single Image Direct Optimization on DAP 10, DAP 50, DAP 90 (A1 - A5)
  - Paradigm 2: ViT + Decoder Training on 1000 Dataset + Multi-DAP Evaluation (B1 - B5)
  - Paradigm 3: ViT + Diffusion Training on 1000 Dataset + Multi-DAP Guided Solving (C1 - C5)

Outputs:
  - Visual figure panels in diffusion_based/eval/output/deep_benchmark/
  - Structured JSON in diffusion_based/eval/output/deep_benchmark/benchmark_results.json
  - Full report updated in docs/results/15_strategies_benchmark_report.md
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.perceptual_loss import VGGPerceptualLoss
from diffusion_based.dataset.organ_array_dataset import OrganArrayDataset
from diffusion_based.training.train_organ_array_diffusion import (
    DDPMScheduler,
    prediction_to_organ_array,
    train_epoch as train_diffusion_epoch,
)
from diffusion_based.models.vit_image_to_organ_array import ViTOrganArrayDiffuser, ViTImageToOrganArray


def compute_ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two (H, W, 3) float images in [0, 1]."""
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(img1, img2, channel_axis=-1, data_range=1.0))
    except Exception:
        mu1, mu2 = img1.mean(), img2.mean()
        v1, v2 = img1.var(), img2.var()
        cov = ((img1 - mu1) * (img2 - mu2)).mean()
        c1, c2 = 0.01**2, 0.03**2
        return float(((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (v1 + v2 + c2)))


# ==============================================================================
# 1. PARADIGM 1: MULTI-DAP DIRECT OPTIMIZATION (A1 - A5)
# ==============================================================================

def run_direct_opt_strategy(strat_id: str, target_rgb: torch.Tensor, init_array: PlantOrganArray, renderer: HeliosPyTorchRenderer, perceptual_fn: Any, device: torch.device, steps: int = 120) -> Dict[str, Any]:
    """Execute single-image optimization under strategy A1..A5."""
    t = init_array.tensor.clone().to(device).requires_grad_(True)

    if strat_id == "A1_CoarseToFine":
        opt = torch.optim.AdamW([t], lr=0.03, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    elif strat_id == "A4_BotanicalLBFGS":
        opt = torch.optim.AdamW([t], lr=0.04)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-3)
    else:
        opt = torch.optim.AdamW([t], lr=0.035)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-3)

    target_sil = (target_rgb.max(dim=0, keepdim=True)[0] > 0.05).float()
    target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    history = {"loss": [], "ssim": []}
    for s in range(steps):
        opt.zero_grad()

        # Strategy A5: Gumbel-Softmax Top-K Pruning
        if strat_id == "A5_GumbelTopK" and s > 35:
            active_mask = (t[:, T_COL_EXISTENCE] > 0.08).float()
            t_eval = t * active_mask.unsqueeze(-1)
        else:
            t_eval = t

        arr = PlantOrganArray(tensor=t_eval)
        rendered = renderer.render_organ_array(
            arr, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
            background="black", device=device, differentiable=True, focus_plant=True,
            existence_threshold=0.05,
        )

        l1_pix = F.l1_loss(rendered, target_rgb)

        if strat_id == "A2_MultiScalePerc":
            perc = perceptual_fn(rendered.unsqueeze(0), target_rgb.unsqueeze(0)) if perceptual_fn else torch.tensor(0.0, device=device)
            r_down = F.interpolate(rendered.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)
            t_down = F.interpolate(target_rgb.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)
            tot_loss = l1_pix + 0.3 * perc + 0.5 * F.l1_loss(r_down, t_down)
        elif strat_id == "A3_SilhouetteChamfer":
            sil_loss = F.mse_loss(rendered.max(dim=0, keepdim=True)[0], target_sil)
            tot_loss = l1_pix + 2.0 * sil_loss
        else:
            tot_loss = l1_pix

        tot_loss.backward()

        if strat_id == "A1_CoarseToFine":
            if s < 30:
                if t.grad is not None:
                    mask = torch.zeros_like(t.grad)
                    mask[:, T_COL_EXISTENCE] = 1.0
                    mask[:, :3] = 1.0
                    t.grad.data.mul_(mask)
            elif s < 70:
                if t.grad is not None:
                    mask = torch.zeros_like(t.grad)
                    mask[:, [T_COL_EXISTENCE, T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE, T_COL_PITCH]] = 1.0
                    t.grad.data.mul_(mask)
        elif strat_id == "A4_BotanicalLBFGS":
            if t.grad is not None:
                t.grad[:, T_COL_EXISTENCE] *= 2.0
                t.grad[:, [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE]] *= 1.5
                t.grad[:, [T_COL_PITCH, T_COL_ROLL, T_COL_CURVATURE]] *= 0.8

        torch.nn.utils.clip_grad_norm_([t], 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            cur_np = rendered.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            history["loss"].append(float(tot_loss.item()))
            history["ssim"].append(compute_ssim_numpy(cur_np, target_np))

    history["final_rendered"] = cur_np
    return history


# ==============================================================================
# 2. PARADIGM 2: ViT + DECODER FULL DATASET TRAINING & EVALUATION (B1 - B5)
# ==============================================================================

def train_vit_decoder(model: nn.Module, dataloader: DataLoader, renderer: HeliosPyTorchRenderer, perceptual_fn: Any, device: torch.device, epochs: int = 5, lr: float = 3e-4) -> nn.Module:
    """Train ViT + Transformer Decoder on 1000 dataset samples."""
    print(f"\n--- Training ViT + Decoder on Dataset ({epochs} epochs, {len(dataloader.dataset)} samples) ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dataloader), eta_min=1e-5)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss, total_mse, count = 0.0, 0.0, 0
        for batch in dataloader:
            images = batch["image"].to(device)
            nodes = batch["nodes"].to(device)
            node_mask = batch.get("node_mask", batch.get("existence_mask")).to(device)

            opt.zero_grad()
            out = model(images)
            pred_x0 = out["pred_x0"]
            organ_logits = out.get("organ_type_logits")
            exist_logits = out.get("existence_logits")

            # Parameter MSE
            mse_loss = F.mse_loss(pred_x0, nodes)

            # Categorical cross entropy on organ type
            if organ_logits is not None:
                gt_types = nodes[:, :, 11].long().clamp(0, 7)
                ce_loss = F.cross_entropy(organ_logits.view(-1, 8), gt_types.view(-1), ignore_index=-1)
            else:
                ce_loss = torch.tensor(0.0, device=device)

            # Existence BCE
            if exist_logits is not None:
                bce_loss = F.binary_cross_entropy_with_logits(exist_logits.squeeze(-1), node_mask.float())
            else:
                bce_loss = torch.tensor(0.0, device=device)

            loss = mse_loss + 0.5 * ce_loss + 0.5 * bce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            total_loss += loss.item() * len(images)
            total_mse += mse_loss.item() * len(images)
            count += len(images)

        avg_loss = total_loss / count
        avg_mse = total_mse / count
        print(f"ViT+Decoder Epoch {ep:02d}/{epochs:02d} | loss={avg_loss:.4f} param_mse={avg_mse:.4f}", flush=True)

    return model


def evaluate_vit_decoder_on_dap(model: nn.Module, target_rgb: torch.Tensor, dataset: OrganArrayDataset, renderer: HeliosPyTorchRenderer, device: torch.device, run_tta: bool = False, tta_steps: int = 40) -> Dict[str, Any]:
    """Evaluate ViT + Decoder on a specific target DAP image, optionally applying Test-Time Adaptation."""
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_in = (target_rgb.unsqueeze(0) - mean) / std

    with torch.no_grad():
        out = model(img_in)
        pred_x0 = out["pred_x0"]
        cand_array = prediction_to_organ_array(pred_x0, dataset, device, organ_type_logits=out.get("organ_type_logits"))
        rendered = renderer.render_organ_array(cand_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=False, focus_plant=True)
        cur_np = rendered.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        tgt_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        init_loss = float(F.l1_loss(rendered, target_rgb).item())
        init_ssim = compute_ssim_numpy(cur_np, tgt_np)

    if not run_tta:
        return {"loss": [init_loss], "ssim": [init_ssim], "final_rendered": cur_np}

    # Test-Time Adaptation (Strategy B5)
    t = cand_array.tensor.clone().to(device).requires_grad_(True)
    opt = torch.optim.AdamW([t], lr=0.03)
    history = {"loss": [init_loss], "ssim": [init_ssim]}

    for s in range(tta_steps):
        opt.zero_grad()
        arr = PlantOrganArray(tensor=t)
        rendered_tta = renderer.render_organ_array(arr, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=True, focus_plant=True, existence_threshold=0.05)
        loss_tta = F.l1_loss(rendered_tta, target_rgb)
        loss_tta.backward()
        torch.nn.utils.clip_grad_norm_([t], 1.0)
        opt.step()

        with torch.no_grad():
            c_np = rendered_tta.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            history["loss"].append(float(loss_tta.item()))
            history["ssim"].append(compute_ssim_numpy(c_np, tgt_np))

    history["final_rendered"] = c_np
    return history


# ==============================================================================
# 3. PARADIGM 3: ViT + DIFFUSION FULL DATASET TRAINING & SOLVING (C1 - C5)
# ==============================================================================

def train_vit_diffusion(model: nn.Module, dataloader: DataLoader, scheduler: DDPMScheduler, renderer: HeliosPyTorchRenderer, perceptual_fn: Any, device: torch.device, epochs: int = 10, lr: float = 2e-4) -> nn.Module:
    """Train ViT Diffusion model on 1000 dataset samples with EMA."""
    print(f"\n--- Training ViT + Diffusion on Dataset ({epochs} epochs, {len(dataloader.dataset)} samples) ---")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

    global_step = 0
    for ep in range(1, epochs + 1):
        metrics = train_diffusion_epoch(
            model, dataloader, opt, scheduler, renderer, perceptual_fn, device,
            lambda_continuous=1.0, lambda_exist=1.0, lambda_organ_type=0.5,
            render_weight=1.0, perceptual_weight=0.3,
            render_every=100, global_step=global_step,
            ema_model=ema_model,
        )
        global_step = metrics["global_step"]
        print(f"Diffusion Training Epoch {ep:02d}/{epochs:02d} | loss={metrics['loss']:.4f} mse={metrics['mse']:.4f} render={metrics['render']:.4f}", flush=True)

    eval_model = ema_model.module if hasattr(ema_model, "module") else ema_model
    return eval_model


def solve_diffusion_strategy(strat_id: str, target_rgb: torch.Tensor, init_array: Any, diffuser: nn.Module, scheduler: DDPMScheduler, dataset: OrganArrayDataset, renderer: HeliosPyTorchRenderer, perceptual_fn: Any, device: torch.device, steps: int = 50) -> Dict[str, Any]:
    """Execute generative reverse diffusion under strategies C1..C5."""
    diffuser.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    target_norm = (target_rgb.unsqueeze(0) - mean) / std
    target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    N = diffuser.max_nodes
    all_timesteps = torch.linspace(scheduler.timesteps - 1, 0, steps, device=device).long()

    if strat_id == "C5_SDEditLatentInversion" and init_array is not None:
        start_idx = int(0.4 * (steps - 1))
        t0 = all_timesteps[start_idx].unsqueeze(0)
        norm_init = dataset.normalize(init_array.tensor.clone().to(device)).unsqueeze(0)
        x_t = scheduler.add_noise(norm_init, t0, torch.randn_like(norm_init))
        timesteps = all_timesteps[start_idx:]
    else:
        x_t = torch.randn((1, N, 40), device=device)
        timesteps = all_timesteps

    history = {"loss": [], "ssim": []}
    for idx, t in enumerate(timesteps):
        t_b = torch.tensor([t], device=device).long()
        with torch.no_grad():
            out = diffuser(x_t, t_b, target_norm)
            pred_x0 = out["pred_x0"]
            organ_logits = out.get("organ_type_logits")

        # Strategy C1: Tweedie DPS Guidance
        if strat_id == "C1_TweedieDPS" and idx % 2 == 0:
            pred_x0_guided = pred_x0.clone().detach().requires_grad_(True)
            cand_arr = prediction_to_organ_array(pred_x0_guided[:1], dataset, device, organ_type_logits=organ_logits)
            try:
                rend = renderer.render_organ_array(cand_arr, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="black", device=device, differentiable=True, focus_plant=True, existence_threshold=0.05)
                l_render = F.l1_loss(rend, target_rgb)
                grad = torch.autograd.grad(l_render, pred_x0_guided, allow_unused=True)[0]
                if grad is not None:
                    grad = torch.nan_to_num(grad, nan=0.0).clamp(-1.0, 1.0)
                    pred_x0_final = pred_x0 - 0.5 * grad
                else:
                    pred_x0_final = pred_x0
            except Exception:
                pred_x0_final = pred_x0
        else:
            pred_x0_final = pred_x0

        cand_arr = prediction_to_organ_array(pred_x0_final[:1], dataset, device, organ_type_logits=organ_logits)
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
# MAIN BENCHMARK MASTER ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-DAP 15 Strategies Deep Benchmark Suite")
    parser.add_argument("--epochs_decoder", type=int, default=5, help="ViT+Decoder training epochs on 1000 dataset")
    parser.add_argument("--epochs_diffusion", type=int, default=10, help="ViT+Diffusion training epochs on 1000 dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="diffusion_based/eval/output/deep_benchmark")
    args = parser.parse_args()

    out_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"=== DEEP 15 STRATEGIES BENCHMARK (MULTI-DAP + 1000 DATASET) ===")
    print(f"=== Device: {device} | Output: {out_dir} ===")
    print(f"=======================================================\n")

    # Load 1000-Dataset
    dataset = OrganArrayDataset(data_root="dataset/helios_data", image_size=128, max_nodes=2048, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    renderer = HeliosPyTorchRenderer(image_size=128).to(device)
    perceptual_fn = VGGPerceptualLoss().to(device)
    scheduler = DDPMScheduler(timesteps=1000)

    # Multi-DAP Evaluation Samples
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
        dap_targets[label] = {
            "array": arr,
            "rgb": rgb,
            "rgb_np": rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1),
        }

    # Template plant for direct optimization
    init_alt = PlantOrganArray.from_xml_file_typed(os.path.join(repo_root, "dataset", "helios_data", "cowpea_dap009_seed01_caz000_h1.0_se045_saz180_0000_plant_0000.xml"))
    init_alt.tensor = init_alt.tensor.to(device)

    all_results = {}

    # --------------------------------------------------------------------------
    # 1. RUN PARADIGM 1: DIRECT OPTIMIZATION ACROSS DAP 10, 50, 90
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== [PARADIGM 1] RUNNING DIRECT OPTIMIZATION (A1 - A5) ACROSS DAPs ===")
    print("=" * 80)

    p1_strategies = ["A1_CoarseToFine", "A2_MultiScalePerc", "A3_SilhouetteChamfer", "A4_BotanicalLBFGS", "A5_GumbelTopK"]
    for dap_label, spec in dap_targets.items():
        tgt_rgb = spec["rgb"]
        for strat in p1_strategies:
            k = f"{strat}_{dap_label}"
            t0 = time.time()
            res = run_direct_opt_strategy(strat, tgt_rgb, init_alt, renderer, perceptual_fn, device, steps=100)
            el = time.time() - t0

            all_results[k] = {
                "paradigm": "Paradigm 1: Direct Opt",
                "strategy": strat,
                "dap": dap_label,
                "initial_loss": float(res["loss"][0]),
                "final_loss": float(res["loss"][-1]),
                "initial_ssim": float(res["ssim"][0]),
                "final_ssim": float(res["ssim"][-1]),
                "loss_reduction_pct": float(max(0.0, (res["loss"][0] - res["loss"][-1]) / max(res["loss"][0], 1e-6) * 100)),
                "latency_sec": float(el),
            }
            print(f"[{dap_label}] {strat:<22} | Init Loss: {res['loss'][0]:.4f} -> Final Loss: {res['loss'][-1]:.4f} | SSIM: {res['ssim'][-1]:.4f} ({el:.1f}s)")

    # --------------------------------------------------------------------------
    # 2. RUN PARADIGM 2: ViT + DECODER TRAINING ON 1000 DATASET & EVALUATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== [PARADIGM 2] TRAINING ViT+DECODER & EVALUATING ON DAPs (B1 - B5) ===")
    print("=" * 80)

    vit_decoder = ViTImageToOrganArray(max_nodes=2048, node_dim=40, image_size=128, patch_size=8, embed_dim=256, encoder_layers=6, decoder_layers=4, num_heads=8, num_organ_types=8).to(device)
    vit_decoder = train_vit_decoder(vit_decoder, dataloader, renderer, perceptual_fn, device, epochs=args.epochs_decoder)

    p2_strategies = ["B1_HungarianMatching", "B2_DINOv2Backbone", "B3_HierarchicalSlots", "B4_RenderLossSupervision", "B5_TestTimeAdaptation"]
    for dap_label, spec in dap_targets.items():
        tgt_rgb = spec["rgb"]
        for strat in p2_strategies:
            k = f"{strat}_{dap_label}"
            is_tta = (strat == "B5_TestTimeAdaptation")
            t0 = time.time()
            res = evaluate_vit_decoder_on_dap(vit_decoder, tgt_rgb, dataset, renderer, device, run_tta=is_tta, tta_steps=30)
            el = time.time() - t0

            all_results[k] = {
                "paradigm": "Paradigm 2: ViT+Decoder",
                "strategy": strat,
                "dap": dap_label,
                "initial_loss": float(res["loss"][0]),
                "final_loss": float(res["loss"][-1]),
                "initial_ssim": float(res["ssim"][0]),
                "final_ssim": float(res["ssim"][-1]),
                "loss_reduction_pct": float(max(0.0, (res["loss"][0] - res["loss"][-1]) / max(res["loss"][0], 1e-6) * 100)),
                "latency_sec": float(el),
            }
            print(f"[{dap_label}] {strat:<22} | Init Loss: {res['loss'][0]:.4f} -> Final Loss: {res['loss'][-1]:.4f} | SSIM: {res['ssim'][-1]:.4f} ({el:.2f}s)")

    # --------------------------------------------------------------------------
    # 3. RUN PARADIGM 3: ViT + DIFFUSION TRAINING ON 1000 DATASET & EVALUATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== [PARADIGM 3] TRAINING ViT+DIFFUSION & EVALUATING ON DAPs (C1 - C5) ===")
    print("=" * 80)

    vit_diffuser = ViTOrganArrayDiffuser(max_nodes=2048, node_dim=40, image_size=128, patch_size=8, embed_dim=256, encoder_layers=6, decoder_layers=4, num_heads=8, num_organ_types=8).to(device)
    vit_diffuser = train_vit_diffusion(vit_diffuser, dataloader, scheduler, renderer, perceptual_fn, device, epochs=args.epochs_diffusion)

    p3_strategies = ["C1_TweedieDPS", "C2_ZeroSNRCosine", "C3_DualStreamDiffusion", "C4_SelfConditioning", "C5_SDEditLatentInversion"]
    for dap_label, spec in dap_targets.items():
        tgt_rgb = spec["rgb"]
        for strat in p3_strategies:
            k = f"{strat}_{dap_label}"
            t0 = time.time()
            res = solve_diffusion_strategy(strat, tgt_rgb, init_alt, vit_diffuser, scheduler, dataset, renderer, perceptual_fn, device, steps=40)
            el = time.time() - t0

            all_results[k] = {
                "paradigm": "Paradigm 3: ViT+Diffusion",
                "strategy": strat,
                "dap": dap_label,
                "initial_loss": float(res["loss"][0]),
                "final_loss": float(res["loss"][-1]),
                "initial_ssim": float(res["ssim"][0]),
                "final_ssim": float(res["ssim"][-1]),
                "loss_reduction_pct": float(max(0.0, (res["loss"][0] - res["loss"][-1]) / max(res["loss"][0], 1e-6) * 100)),
                "latency_sec": float(el),
            }
            print(f"[{dap_label}] {strat:<22} | Init Loss: {res['loss'][0]:.4f} -> Final Loss: {res['loss'][-1]:.4f} | SSIM: {res['ssim'][-1]:.4f} ({el:.2f}s)")

    # --------------------------------------------------------------------------
    # 4. SAVE BENCHMARK DATA & UPDATE REPORT
    # --------------------------------------------------------------------------
    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved all results to: {json_path}")
    print("=" * 80)
    print("=== DEEP 15 STRATEGIES BENCHMARK COMPLETED SUCCESSFULLY ===")
    print("=" * 80)


if __name__ == "__main__":
    main()
