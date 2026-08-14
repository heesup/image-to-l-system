"""
Hard problem standalone: random topology + soft parent + fixed existence + scale optimization.
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

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    T_COL_LENGTH,
    T_COL_RADIUS,
    T_COL_SCALE,
    T_COL_CURRENT_LEAF_SCALE_FACTOR,
    T_COL_ORGAN_TYPE,
    T_COL_EXISTENCE,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
)
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
    tensor[:, T_COL_EXISTENCE] = existence.to(tensor.device)

    # Shrink continuous geometry of all organ rows by 40% (typed layout).
    geom_cols = [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE, T_COL_CURRENT_LEAF_SCALE_FACTOR]
    tensor[:, geom_cols] *= 0.4
    return PlantOrganArray(tensor, raw_metadata=target_array.raw_metadata)


def optimize_hard(target_rgb, init_array, renderer, device, num_steps=500, lr=0.03,
                  phase1_steps=200):
    base_tensor = init_array.tensor.clone().detach().to(device)
    fixed_existence = torch.sigmoid(base_tensor[:, T_COL_EXISTENCE]).detach()

    organ_type = base_tensor[:, T_COL_ORGAN_TYPE].long()
    is_internode = (organ_type == ORGAN_INTERNODE)
    is_petiole = (organ_type == ORGAN_PETIOLE)
    is_leaf = (organ_type == ORGAN_LEAF)

    leaf_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    stem_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    petiole_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    N = base_tensor.shape[0]
    node_leaf_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_stem_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_pet_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    opt_parent_logits = init_array.parent_logits.clone().detach().to(device).requires_grad_(False)
    parent_candidates = init_array.parent_candidates.to(device)

    scale_params = [leaf_logit, stem_logit, petiole_logit,
                    node_leaf_logit, node_stem_logit, node_pet_logit]

    # Phase 1: optimize scales only, parent_logits frozen, existence fixed
    optimizer = optim.Adam([{"params": scale_params, "lr": lr}])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_steps, eta_min=lr * 0.1)

    target_mask = (target_rgb.sum(0) > 0.05).float().detach()
    history = {"loss": [], "ssim": [], "existence_sum": [], "images": []}

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
        tensor = base_tensor.clone()
        tensor[is_internode, T_COL_LENGTH] *= stem_scale * node_stem[is_internode]
        tensor[is_internode, T_COL_RADIUS] *= stem_scale * node_stem[is_internode]
        tensor[is_petiole, T_COL_LENGTH] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_RADIUS] *= petiole_scale * node_pet[is_petiole]
        tensor[is_petiole, T_COL_CURRENT_LEAF_SCALE_FACTOR] *= leaf_scale * node_leaf[is_petiole]
        tensor[is_leaf, T_COL_SCALE] *= leaf_scale * node_leaf[is_leaf]
        tensor[:, T_COL_EXISTENCE] = fixed_existence
        return PlantOrganArray(
            tensor,
            raw_metadata=init_array.raw_metadata,
            parent_logits=opt_parent_logits,
            parent_candidates=parent_candidates,
        )

    for step in range(num_steps + 1):
        # Phase transition: enable parent_logits, keep scales with lower lr
        if step == phase1_steps:
            opt_parent_logits.requires_grad_(True)
            optimizer = optim.Adam(
                [
                    {"params": scale_params, "lr": lr * 0.1},
                    {"params": [opt_parent_logits], "lr": lr},
                ]
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, num_steps - phase1_steps), eta_min=lr * 0.05
            )
            print(f"  [phase transition at step {step}] parent_logits unfrozen, scales lr={lr*0.1:.4f}")

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

        if step < num_steps:
            total_loss.backward()
            params_to_clip = scale_params[:]
            if opt_parent_logits.requires_grad:
                params_to_clip.append(opt_parent_logits)
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            cur_np = rendered_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            target_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            ssim = compute_ssim_numpy(cur_np, target_np)

        history["loss"].append(total_loss.item())
        history["ssim"].append(ssim)
        history["existence_sum"].append(organ_array.existence.sum().item())

        if step % 50 == 0 or step == num_steps:
            leaf, stem, pet, _, _, _ = get_scales()
            print(f"  step {step:03d} | loss={total_loss.item():.4f} | ssim={ssim:.4f} | "
                  f"exist={organ_array.existence.sum().item():.1f} | "
                  f"leaf={leaf:.3f} stem={stem:.3f} pet={pet:.3f}")
            if len(history["images"]) < 6:
                history["images"].append((step, cur_np, total_loss.item(), ssim))

    return history


def plot_problem(target_rgb_np, history, output_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="black")
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")

    fig.suptitle(
        "PROBLEM 3 - HARD (redefined): Random topology + soft parent + fixed existence + scale optimization",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    axes[0, 0].imshow(target_rgb_np)
    axes[0, 0].set_title("Target Helios GT\n(DAP 10)", color="white", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    for idx, (step_num, img, loss_v, ssim_v) in enumerate(history["images"]):
        if idx < 3:
            ax = axes[0, idx + 1]
        else:
            ax = axes[1, 0]
        ax.imshow(img)
        ax.set_title(f"Step {step_num:03d}\nLoss={loss_v:.4f} | SSIM={ssim_v:.4f}", color="cyan", fontsize=12, fontweight="bold")
        ax.axis("off")

    axes[1, 1].plot(history["loss"], color="crimson", linewidth=2.5)
    axes[1, 1].set_title("Loss Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Step", color="white")
    axes[1, 1].set_ylabel("Loss", color="crimson")
    axes[1, 1].tick_params(colors="white")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    axes[1, 2].plot(history["ssim"], color="springgreen", linewidth=2.5)
    axes[1, 2].set_title("SSIM Curve", color="white", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("Step", color="white")
    axes[1, 2].set_ylabel("SSIM", color="springgreen")
    axes[1, 2].tick_params(colors="white")
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)

    final_diff = np.abs(history["images"][-1][1] - target_rgb_np)
    im = axes[1, 3].imshow(final_diff.mean(axis=-1), cmap="inferno", vmin=0.0, vmax=0.2)
    axes[1, 3].set_title(f"Final Diff Map\nMAE={np.mean(final_diff):.5f}", color="gold", fontsize=12, fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved hard figure to {output_path}")


def main():
    output_dir = os.path.join(repo_root, "diffusion_based", "eval", "output")
    os.makedirs(output_dir, exist_ok=True)
    source_xml = os.path.join(output_dir, "dap10_gt_0000_plant_0000.xml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running hard standalone on device: {device}")

    organ_array_gt = PlantOrganArray.from_xml_file_typed(source_xml)
    organ_array_gt.tensor = organ_array_gt.tensor.to(device)

    renderer = HeliosPyTorchRenderer(image_size=128)
    with torch.no_grad():
        target_rgb = renderer.render_organ_array(
            organ_array_gt,
            azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0,
            background="black", device=device, differentiable=True, focus_plant=True,
        )
    target_rgb_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    init_hard = make_random_topology(organ_array_gt, seed=42)
    parent_logits, parent_candidates = PlantOrganArray.build_parent_candidates_from_gt(
        init_hard, num_candidates=8, seed=42
    )
    init_hard = init_hard.clone_with_parent_logits(parent_logits, parent_candidates)

    print("\n=== HARD (random topology + soft parent + fixed existence + scale opt) ===")
    t0 = time.time()
    hist_hard = optimize_hard(target_rgb, init_hard, renderer, device, num_steps=500, lr=0.03)
    print(f"Time: {time.time() - t0:.1f}s")

    plot_problem(
        target_rgb_np,
        hist_hard,
        os.path.join(output_dir, "backprop_problem_hard_redefined.png"),
    )

    metrics = {
        "hard_redefined": {
            "initial_loss": hist_hard["loss"][0],
            "final_loss": hist_hard["loss"][-1],
            "initial_ssim": hist_hard["ssim"][0],
            "final_ssim": hist_hard["ssim"][-1],
        }
    }
    out_path = os.path.join(output_dir, "hard_redefined_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
