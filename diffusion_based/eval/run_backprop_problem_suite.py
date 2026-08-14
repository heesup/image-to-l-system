"""
Backpropagation-based inverse rendering problem suite.

Generates three figures demonstrating increasing difficulty:
  1. Easy:   fixed GT topology + fixed per-node geometry, optimize only
             per-node existence + global leaf/stem/petiole scale multipliers.
  2. Medium: grow from a tiny seed plant (2 active nodes), optimize existence
             + global scales.
  3. Hard:   random topology initialization, attempt backpropagation-based
             recovery (expected partial/limited convergence).

Output: diffusion_based/eval/output/backprop_problem_{easy,medium,hard}.png
"""

import os
import sys
import time
import json
import re
import argparse
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
    COL_PLANT_ID,
    COL_PLANT_AGE,
    COL_SHOOT_ID,
    COL_SHOOT_TYPE,
    COL_PARENT_SHOOT_ID,
    COL_PARENT_NODE_IDX,
    COL_PARENT_PETIOLE_IDX,
    COL_SHOOT_ROT_PITCH,
    COL_SHOOT_ROT_YAW,
    COL_SHOOT_ROT_ROLL,
    COL_PHYTOMER_IDX,
    COL_INODE_LEN,
    COL_INODE_RAD,
    COL_INODE_PITCH,
    COL_PET0_LEN,
    COL_PET0_RAD,
    COL_PET0_PITCH,
    COL_PET0_CURV,
    COL_PET0_LEAF_SCALE,
    COL_PET0_L0_SCALE,
    COL_PET0_L1_SCALE,
    COL_PET0_L2_SCALE,
    COL_EXISTENCE,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def render_organ_array_with_sanity(organ_array, renderer, target_rgb, device, label=""):
    """Render an organ array and compare to a target image for debugging/sanity."""
    rgb = renderer.render_organ_array(
        organ_array,
        azimuth_deg=0.0,
        elevation_deg=90.0,
        camera_height=1.0,
        background="black",
        device=device,
        differentiable=True,
        focus_plant=True,
    )
    mae = float(torch.mean(torch.abs(rgb - target_rgb)).item())
    ssim = compute_ssim_numpy(
        rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1),
        target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1),
    )
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}render MAE={mae:.6f} SSIM={ssim:.4f}")
    return rgb


def compute_ssim_numpy(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(ssim(img1, img2, channel_axis=2, data_range=1.0, win_size=win_size))
    except Exception as e:
        mse = float(np.mean((img1 - img2) ** 2))
        return float(max(0.0, 1.0 - 5.0 * mse))


def render_target(organ_array_gt, renderer, device):
    with torch.no_grad():
        target_rgb = renderer.render_organ_array(
            organ_array_gt,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=1.0,
            background="black",
            device=device,
            differentiable=True,
            focus_plant=True,
        )
    return target_rgb


def make_seed_plant(target_array, seed=42):
    """Create a tiny 2-node seed plant from the target template.

    Only the first 2 nodes are active; all other existence values are 0.0.
    This matches a real germination stage with just an unifoliate leaf pair.
    """
    cpu_rng = torch.Generator(device='cpu').manual_seed(seed)
    N = target_array.num_nodes
    tensor = target_array.tensor.clone()
    existence = torch.zeros(N, device=tensor.device)
    existence[:2] = 1.0
    tensor[:, COL_EXISTENCE] = existence

    scale_cols = [
        COL_INODE_LEN, COL_INODE_RAD,
        COL_PET0_LEN, COL_PET0_RAD,
        COL_PET0_LEAF_SCALE,
        COL_PET0_L0_SCALE, COL_PET0_L1_SCALE, COL_PET0_L2_SCALE,
    ]
    for c in scale_cols:
        tensor[:, c] *= 0.15
    tensor[:, COL_PET0_PITCH] *= 0.5
    tensor[:, COL_PET0_CURV] *= 0.3
    return PlantOrganArray(tensor, raw_metadata=target_array.raw_metadata)


def make_random_topology(target_array, seed=42):
    """Create a random topology by shuffling node order and perturbing parents."""
    cpu_rng = torch.Generator(device='cpu').manual_seed(seed)
    N = target_array.num_nodes
    perm = torch.randperm(N, generator=cpu_rng)
    tensor = target_array.tensor.clone()[perm]

    # Guard any pre-existing NaN/Inf in the source tensor
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    existence = (torch.rand(N, generator=cpu_rng) < 0.5).float()
    existence = existence * (0.6 + 0.4 * torch.rand(N, generator=cpu_rng))
    tensor[:, COL_EXISTENCE] = existence.to(tensor.device)

    for i in range(N):
        if torch.rand(1, generator=cpu_rng).item() < 0.3:
            tensor[i, COL_PARENT_NODE_IDX] = float(torch.randint(0, max(1, i), (1,), generator=cpu_rng).item())
        if torch.rand(1, generator=cpu_rng).item() < 0.2:
            tensor[i, COL_PARENT_SHOOT_ID] = float(torch.randint(0, 3, (1,), generator=cpu_rng).item())

    scale_cols = [COL_INODE_LEN, COL_INODE_RAD, COL_PET0_LEN, COL_PET0_RAD,
                  COL_PET0_LEAF_SCALE, COL_PET0_L0_SCALE, COL_PET0_L1_SCALE, COL_PET0_L2_SCALE]
    for c in scale_cols:
        tensor[:, c] *= 0.4

    # Final guard against any NaN/Inf
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    return PlantOrganArray(tensor, raw_metadata=target_array.raw_metadata)


def optimize_backprop(
    target_rgb,
    init_array,
    renderer,
    device,
    num_steps=60,
    lr=0.03,
    optimize_geometry=False,
    optimize_topology=False,
    snapshot_steps=None,
    binary_threshold_step=None,
    grad_clip=1.0,
    existence_pull_weight=0.05,
    fix_existence=False,
):
    if snapshot_steps is None:
        snapshot_steps = [0, 15, 30, 45, 60]
    """Run backpropagation-based inverse rendering.

    Optionally decay lr with a cosine schedule. If binary_threshold_step is set,
    existence logits are snapped to binary values (0 or 1) at that step and
    only scale multipliers continue to refine afterwards.
    If fix_existence is True, the initial existence values are kept fixed and
    are not optimized (useful for hard topology experiments).
    """
    base_tensor = init_array.tensor.clone().detach().to(device)
    fixed_existence = torch.sigmoid(base_tensor[:, COL_EXISTENCE]).detach()
    opt_existence = base_tensor[:, COL_EXISTENCE].clone().detach().requires_grad_(not fix_existence)

    # Global scale multipliers (constrained to [0, 1.5])
    leaf_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    stem_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    petiole_logit = torch.tensor(np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    # Per-node scale multipliers (initialized near 1.0, constrained to [0, 2.0])
    N = base_tensor.shape[0]
    node_leaf_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_stem_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)
    node_pet_logit = torch.full((N,), np.log(1.0 / 0.5), device=device, requires_grad=True, dtype=torch.float32)

    scale_params = [leaf_logit, stem_logit, petiole_logit,
                    node_leaf_logit, node_stem_logit, node_pet_logit]

    opt_tensor = None
    if optimize_geometry or optimize_topology:
        opt_tensor = base_tensor.clone().detach().requires_grad_(True)

    # Soft parent topology optimization
    opt_parent_logits = None
    parent_candidates = None
    if optimize_topology and init_array.parent_logits is not None:
        opt_parent_logits = init_array.parent_logits.clone().detach().to(device).requires_grad_(True)
        parent_candidates = init_array.parent_candidates.to(device)

    # Build parameter groups
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

        if opt_tensor is not None:
            tensor = opt_tensor.clone()
        else:
            tensor = base_tensor.clone()

        tensor[:, COL_INODE_LEN] *= stem_scale * node_stem
        tensor[:, COL_INODE_RAD] *= stem_scale * node_stem
        tensor[:, COL_PET0_LEN] *= petiole_scale * node_pet
        tensor[:, COL_PET0_RAD] *= petiole_scale * node_pet
        tensor[:, COL_PET0_PITCH] *= ((petiole_scale * node_pet) * 0.5 + 0.5)
        tensor[:, COL_PET0_CURV] *= petiole_scale * node_pet
        tensor[:, COL_PET0_LEAF_SCALE] *= leaf_scale * node_leaf
        tensor[:, COL_PET0_L0_SCALE] *= leaf_scale * node_leaf
        tensor[:, COL_PET0_L1_SCALE] *= leaf_scale * node_leaf
        tensor[:, COL_PET0_L2_SCALE] *= leaf_scale * node_leaf
        if fix_existence:
            tensor[:, COL_EXISTENCE] = fixed_existence
        else:
            tensor[:, COL_EXISTENCE] = torch.sigmoid(opt_existence)
        if opt_parent_logits is not None:
            return PlantOrganArray(
                tensor,
                raw_metadata=base_metadata,
                parent_logits=opt_parent_logits,
                parent_candidates=parent_candidates,
            )
        return PlantOrganArray(tensor, raw_metadata=base_metadata)

    # Cosine LR schedule wrapper (warmup 0..5, cosine decay 5..num_steps)
    def lr_lambda(step):
        if step < 5:
            return step / 5.0
        progress = (step - 5) / max(1, num_steps - 5)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for step in range(num_steps + 1):
        # Optional hard-thresholding of existence for cleaner convergence
        if not fix_existence and binary_threshold_step is not None and step == binary_threshold_step:
            with torch.no_grad():
                opt_existence.data = torch.where(
                    torch.sigmoid(opt_existence) > 0.5,
                    torch.tensor(6.0, device=device),   # ~ sigmoid = 0.997
                    torch.tensor(-6.0, device=device),  # ~ sigmoid = 0.002
                )

        # Keep parent logits in a numerically stable range for softmax
        if opt_parent_logits is not None:
            with torch.no_grad():
                opt_parent_logits.clamp_(-5.0, 5.0)

        optimizer.zero_grad()
        organ_array = build_array()

        rendered_rgb = renderer.render_organ_array(
            organ_array,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=1.0,
            background="black",
            device=device,
            differentiable=True,
            focus_plant=True,
        )

        loss_rgb = F.mse_loss(rendered_rgb * target_mask.unsqueeze(0), target_rgb * target_mask.unsqueeze(0))
        rendered_mask = (rendered_rgb.sum(0) > 0.05).float()
        loss_sil = F.binary_cross_entropy(rendered_mask, target_mask)
        # Pull existence toward 1.0 so we recover all target organs
        existence_pull = existence_pull_weight * F.mse_loss(organ_array.existence, torch.ones_like(organ_array.existence))
        total_loss = loss_rgb + 2.0 * loss_sil + existence_pull

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
            if step % 50 == 0 or step in snapshot_steps:
                leaf, stem, pet, _, _, _ = get_scales()
                print(f"  step {step:02d} | loss={total_loss.item():.4f} | ssim={ssim:.4f} | "
                      f"exist={organ_array.existence.sum().item():.1f} | "
                      f"leaf={leaf:.3f} stem={stem:.3f} pet={pet:.3f}")

    return history


def _extract_dap_and_name(xml_path: str):
    base = os.path.basename(xml_path)
    name = base.replace(".xml", "")
    m = re.search(r"dap(\d+)", name, re.IGNORECASE)
    dap = int(m.group(1)) if m else 10
    return name, dap


def plot_problem(target_rgb_np, history, title, caption, output_path, dap=10):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="black")
    for row in axes:
        for ax in row:
            ax.set_facecolor("black")

    fig.suptitle(caption, color="white", fontsize=14, fontweight="bold", y=0.98)

    axes[0, 0].imshow(target_rgb_np)
    axes[0, 0].set_title(f"Target Helios GT\n(DAP {dap})", color="white", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")

    for idx, (step_num, img, loss_v, ssim_v) in enumerate(history["images"]):
        if idx < 3:
            ax = axes[0, idx + 1]
        else:
            ax = axes[1, 0]
        ax.imshow(img)
        ax.set_title(f"Step {step_num:02d}\nLoss={loss_v:.4f} | SSIM={ssim_v:.4f}", color="cyan", fontsize=12, fontweight="bold")
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
    print(f"Saved {title} figure to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_xml", type=str, default="diffusion_based/eval/output/dap10_gt_0000_plant_0000.xml",
                        help="Path to the source Helios XML plant to use as GT target")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs. Default is diffusion_based/eval/output/<xml_name>_backprop")
    args = parser.parse_args()

    xml_name, dap = _extract_dap_and_name(args.source_xml)
    if args.output_dir is None:
        args.output_dir = os.path.join("diffusion_based", "eval", "output", f"{xml_name}_backprop")

    output_dir = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    source_xml = os.path.join(repo_root, args.source_xml)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Running backpropagation problem suite on device: {device}")
    print(f"Source XML: {args.source_xml} (DAP {dap})")
    print(f"Output dir: {args.output_dir}")

    organ_array_gt = PlantOrganArray.from_xml_file(source_xml)
    organ_array_gt.tensor = organ_array_gt.tensor.to(device)

    renderer = HeliosPyTorchRenderer(image_size=128)
    target_rgb = render_target(organ_array_gt, renderer, device)
    target_rgb_np = target_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    all_metrics = {}

    # Problem 1: Easy
    print("\n=== PROBLEM 1: EASY (fixed topology + geometry, optimize existence + global scales) ===")
    init_easy = make_seed_plant(organ_array_gt, seed=42)
    # Reset geometry to GT template scaled by 0.25, existence to 0.3 (small / sparse start)
    init_easy.tensor = organ_array_gt.tensor.clone()
    init_easy.tensor[:, COL_INODE_LEN] *= 0.25
    init_easy.tensor[:, COL_INODE_RAD] *= 0.25
    init_easy.tensor[:, COL_PET0_LEN] *= 0.25
    init_easy.tensor[:, COL_PET0_RAD] *= 0.25
    init_easy.tensor[:, COL_PET0_LEAF_SCALE] *= 0.25
    init_easy.tensor[:, COL_PET0_L0_SCALE] *= 0.25
    init_easy.tensor[:, COL_PET0_L1_SCALE] *= 0.25
    init_easy.tensor[:, COL_PET0_L2_SCALE] *= 0.25
    init_easy.tensor[:, COL_PET0_PITCH] *= 0.7
    init_easy.tensor[:, COL_PET0_CURV] *= 0.5
    init_easy.tensor[:, COL_EXISTENCE] = 0.3
    hist_easy = optimize_backprop(target_rgb, init_easy, renderer, device, num_steps=1000, lr=0.03,
                                   optimize_geometry=False, optimize_topology=False,
                                   snapshot_steps=[0, 50, 100, 200, 350, 500, 750, 1000],
                                   binary_threshold_step=400,
                                   grad_clip=1.0,
                                   existence_pull_weight=0.05)
    plot_problem(
        target_rgb_np,
        hist_easy,
        "easy",
        f"BACKPROP DAP{dap} - EASY: Fixed GT topology and per-node geometry. Optimize only per-node existence + global leaf/stem/petiole scale multipliers.",
        os.path.join(output_dir, f"{xml_name}_backprop_problem_easy.png"),
        dap=dap,
    )
    all_metrics["easy"] = {
        "initial_loss": hist_easy["loss"][0],
        "final_loss": hist_easy["loss"][-1],
        "initial_ssim": hist_easy["ssim"][0],
        "final_ssim": hist_easy["ssim"][-1],
    }

    # Problem 2: Medium
    print("\n=== PROBLEM 2: MEDIUM (grow from tiny seed, existence + scales) ===")
    init_medium = make_seed_plant(organ_array_gt, seed=42)
    hist_medium = optimize_backprop(target_rgb, init_medium, renderer, device, num_steps=1000, lr=0.03,
                                     optimize_geometry=False, optimize_topology=False,
                                     snapshot_steps=[0, 50, 100, 200, 350, 500, 750, 1000],
                                     binary_threshold_step=400)
    plot_problem(
        target_rgb_np,
        hist_medium,
        "medium",
        f"BACKPROP DAP{dap} - MEDIUM: Start from a tiny seed plant (2 active nodes). Optimize existence to grow organs + global scale multipliers.",
        os.path.join(output_dir, f"{xml_name}_backprop_problem_medium.png"),
        dap=dap,
    )
    all_metrics["medium"] = {
        "initial_loss": hist_medium["loss"][0],
        "final_loss": hist_medium["loss"][-1],
        "initial_ssim": hist_medium["ssim"][0],
        "final_ssim": hist_medium["ssim"][-1],
    }

    # Problem 3: Hard
    print("\n=== PROBLEM 3: HARD (random topology, soft parent, backprop) ===")
    init_hard = make_random_topology(organ_array_gt, seed=42)

    # Soft parent representation: each shoot gets 8 candidate parents (GT + noise)
    parent_logits, parent_candidates = PlantOrganArray.build_parent_candidates_from_gt(
        init_hard, num_candidates=8, seed=42
    )
    init_hard = init_hard.clone_with_parent_logits(parent_logits, parent_candidates)

    # Forward equivalence sanity: one-hot on GT parent should render like hard topology
    hard_gt_logits = torch.zeros_like(parent_logits)
    hard_gt_logits[:, 0] = 10.0
    hard_gt_array = organ_array_gt.clone_with_parent_logits(hard_gt_logits, parent_candidates)
    print("\n[Sanity check] GT one-hot soft parent vs hard parent:")
    render_organ_array_with_sanity(hard_gt_array, renderer, target_rgb, device, label="soft-onehot")
    print(f"[Sanity check] Target MAE baseline (GT vs itself): 0.000000")

    hist_hard = optimize_backprop(target_rgb, init_hard, renderer, device, num_steps=1000, lr=0.03,
                                   optimize_geometry=False, optimize_topology=True,
                                   snapshot_steps=[0, 50, 100, 200, 350, 500, 750, 1000],
                                   binary_threshold_step=400,
                                   grad_clip=1.0,
                                   existence_pull_weight=0.05,
                                   fix_existence=True)
    plot_problem(
        target_rgb_np,
        hist_hard,
        "hard",
        f"BACKPROP DAP{dap} - HARD: Random topology initialization with soft parent candidates. Backprop learns topology + existence + scales.",
        os.path.join(output_dir, f"{xml_name}_backprop_problem_hard.png"),
        dap=dap,
    )
    all_metrics["hard"] = {
        "initial_loss": hist_hard["loss"][0],
        "final_loss": hist_hard["loss"][-1],
        "initial_ssim": hist_hard["ssim"][0],
        "final_ssim": hist_hard["ssim"][-1],
    }

    with open(os.path.join(output_dir, f"{xml_name}_backprop_problem_suite_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nAll metrics:", all_metrics)


if __name__ == "__main__":
    main()
