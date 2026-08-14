"""
Quick hard-only experiment for soft-parent topology optimization.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray, COL_EXISTENCE
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def compute_ssim_numpy(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=win_size))
    except Exception:
        mse = float(np.mean((img1 - img2) ** 2))
        return float(max(0.0, 1.0 - 5.0 * mse))


def make_random_topology(target_array, seed=42):
    cpu_rng = torch.Generator(device='cpu').manual_seed(seed)
    N = target_array.num_nodes
    perm = torch.randperm(N, generator=cpu_rng)
    tensor = target_array.tensor.clone()[perm]

    existence = (torch.rand(N, generator=cpu_rng) < 0.5).float()
    existence = existence * (0.6 + 0.4 * torch.rand(N, generator=cpu_rng))
    tensor[:, COL_EXISTENCE] = existence.to(tensor.device)

    scale_cols = [11, 12, 21, 22, 25, 32, 36, 40]
    for c in scale_cols:
        tensor[:, c] *= 0.4
    return PlantOrganArray(tensor, raw_metadata=target_array.raw_metadata)


def optimize_hard(target_rgb, init_array, renderer, device, num_steps=100, lr=0.03):
    base_tensor = init_array.tensor.clone().detach().to(device)
    opt_existence = base_tensor[:, COL_EXISTENCE].clone().detach().requires_grad_(True)

    leaf_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    stem_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    petiole_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    N = base_tensor.shape[0]
    node_leaf_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_stem_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_pet_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    opt_parent_logits = init_array.parent_logits.clone().detach().to(device).requires_grad_(True)
    parent_candidates = init_array.parent_candidates.to(device)

    params = [opt_existence, leaf_logit, stem_logit, petiole_logit,
              node_leaf_logit, node_stem_logit, node_pet_logit, opt_parent_logits]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.1)

    target_mask = (target_rgb.sum(0) > 0.05).float().detach()
    history = {"loss": [], "ssim": []}

    def build_array():
        leaf_scale = torch.sigmoid(leaf_logit) * 1.5
        stem_scale = torch.sigmoid(stem_logit) * 1.5
        petiole_scale = torch.sigmoid(petiole_logit) * 1.5
        node_leaf = torch.sigmoid(node_leaf_logit) * 2.0
        node_stem = torch.sigmoid(node_stem_logit) * 2.0
        node_pet = torch.sigmoid(node_pet_logit) * 2.0

        tensor = base_tensor.clone()
        tensor[:, 11] *= stem_scale * node_stem
        tensor[:, 12] *= stem_scale * node_stem
        tensor[:, 21] *= petiole_scale * node_pet
        tensor[:, 22] *= petiole_scale * node_pet
        tensor[:, 23] *= ((petiole_scale * node_pet) * 0.5 + 0.5)
        tensor[:, 24] *= petiole_scale * node_pet
        tensor[:, 25] *= leaf_scale * node_leaf
        tensor[:, 32] *= leaf_scale * node_leaf
        tensor[:, 36] *= leaf_scale * node_leaf
        tensor[:, 40] *= leaf_scale * node_leaf
        tensor[:, COL_EXISTENCE] = torch.sigmoid(opt_existence)

        return PlantOrganArray(
            tensor,
            raw_metadata=init_array.raw_metadata,
            parent_logits=opt_parent_logits,
            parent_candidates=parent_candidates,
        )

    # Diagnostic: train parent_logits + scales; freeze existence
    opt_existence.requires_grad_(False)
    optimizer = optim.Adam([leaf_logit, stem_logit, petiole_logit,
                            node_leaf_logit, node_stem_logit, node_pet_logit,
                            opt_parent_logits], lr=lr)

    for step in range(num_steps + 1):
        # Bound parent logits to keep softmax numerically stable
        with torch.no_grad():
            opt_parent_logits.clamp_(-5.0, 5.0)

        optimizer.zero_grad()
        organ_array = build_array()
        rendered_rgb = renderer.render_organ_array(
            organ_array,
            azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
            background="black", device=device, differentiable=True, focus_plant=True,
        )
        loss_rgb = F.mse_loss(rendered_rgb * target_mask.unsqueeze(0), target_rgb * target_mask.unsqueeze(0))
        rendered_mask = (rendered_rgb.sum(0) > 0.05).float()
        loss_sil = F.binary_cross_entropy(rendered_mask, target_mask)
        existence_pull = 0.05 * F.mse_loss(organ_array.existence, torch.ones_like(organ_array.existence))
        total_loss = loss_rgb + 2.0 * loss_sil + existence_pull

        # Debug forward NaN
        if step <= 2:
            with torch.no_grad():
                has_nan = bool(organ_array.tensor.isnan().any()) or bool(organ_array.parent_logits.isnan().any())
                print(f"  [forward check step {step}] organ_array tensor nan={has_nan}")
                if has_nan:
                    for name, p in [
                        ("opt_existence", opt_existence),
                        ("leaf_logit", leaf_logit),
                        ("stem_logit", stem_logit),
                        ("petiole_logit", petiole_logit),
                        ("node_leaf_logit", node_leaf_logit),
                        ("node_stem_logit", node_stem_logit),
                        ("node_pet_logit", node_pet_logit),
                        ("opt_parent_logits", opt_parent_logits),
                    ]:
                        print(f"    {name}: min={p.min().item():.4f} max={p.max().item():.4f} nan={bool(p.isnan().any())}")
                    break

        if step < num_steps:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            if step <= 1:
                with torch.no_grad():
                    for name, p in [
                        ("opt_parent_logits", opt_parent_logits),
                    ]:
                        if p.grad is None:
                            print(f"  [grad step {step}] {name}: grad=None")
                        else:
                            print(f"  [grad step {step}] {name}: min={p.grad.min().item():.4f} max={p.grad.max().item():.4f} nan={bool(p.grad.isnan().any())} inf={bool(p.grad.isinf().any())}")
            optimizer.step()
            scheduler.step()

        # Debug NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            with torch.no_grad():
                print(f"  [NaN debug] step {step}")
                for name, p in [
                    ("opt_existence", opt_existence),
                    ("leaf_logit", leaf_logit),
                    ("stem_logit", stem_logit),
                    ("petiole_logit", petiole_logit),
                    ("node_leaf_logit", node_leaf_logit),
                    ("node_stem_logit", node_stem_logit),
                    ("node_pet_logit", node_pet_logit),
                    ("opt_parent_logits", opt_parent_logits),
                ]:
                    print(f"    {name}: min={p.min().item():.4f} max={p.max().item():.4f} nan={bool(p.isnan().any())} inf={bool(p.isinf().any())}")
                    if p.grad is not None:
                        print(f"      grad: min={p.grad.min().item():.4f} max={p.grad.max().item():.4f} nan={bool(p.grad.isnan().any())} inf={bool(p.grad.isinf().any())}")
            break

        with torch.no_grad():
            ssim = compute_ssim_numpy(
                rendered_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1),
                target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1),
            )
        history["loss"].append(total_loss.item())
        history["ssim"].append(ssim)
        if step % 20 == 0 or step == num_steps:
            print(f"  step {step:03d} | loss={total_loss.item():.4f} | ssim={ssim:.4f} | exist={organ_array.existence.sum().item():.1f}")

    return history


def main():
    output_dir = os.path.join(repo_root, "diffusion_based", "eval", "output")
    source_xml = os.path.join(output_dir, "dap10_gt_0000_plant_0000.xml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    organ_array_gt = PlantOrganArray.from_xml_file(source_xml)
    organ_array_gt.tensor = organ_array_gt.tensor.to(device)
    renderer = HeliosPyTorchRenderer(image_size=64)

    with torch.no_grad():
        target_rgb = renderer.render_organ_array(
            organ_array_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
            background="black", device=device, differentiable=True, focus_plant=True,
        )

    init_hard = make_random_topology(organ_array_gt, seed=42)
    parent_logits, parent_candidates = PlantOrganArray.build_parent_candidates_from_gt(
        init_hard, num_candidates=8, seed=42
    )
    init_hard = init_hard.clone_with_parent_logits(parent_logits, parent_candidates)

    print("\nHARD quick test: 100 steps, 64x64")
    t0 = time.time()
    hist = optimize_hard(target_rgb, init_hard, renderer, device, num_steps=100, lr=0.03)
    print(f"Time: {time.time() - t0:.1f}s")
    print(f"Initial loss={hist['loss'][0]:.4f} final loss={hist['loss'][-1]:.4f}")
    print(f"Initial SSIM={hist['ssim'][0]:.4f} final SSIM={hist['ssim'][-1]:.4f}")

    out_path = os.path.join(output_dir, "hard_quick_test_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"loss": hist["loss"], "ssim": hist["ssim"]}, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
