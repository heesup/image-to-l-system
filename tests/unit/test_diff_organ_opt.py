"""
Demonstration & Verification: Differentiable Renderer Optimization of Organ Type & Existence.

1. Ground Truth: A minimal 2-organ plant (1 Stem/Internode + 1 Leaf).
   Render GT RGB-D image (256x256).
2. Perturbed Initial 26D Tensor with 3 Slots:
   - Slot 0: At Stem position, but initialized with logits preferring LEAF (wrong organ type!).
   - Slot 1: At Leaf position, but initialized with logits preferring STEM (wrong organ type!).
   - Slot 2: At an empty position, initialized with existence=0.9 (spurious extra organ!).
3. Differentiable Optimization Loop:
   - Optimizer: Adam on 26D continuous parameters (logits + 3D geometry).
   - Renderer: HeliosPyTorchRenderer(differentiable=True) with dr.antialias.
   - Loss: L1(RGB) + L1(Depth) + Entropy regularizer.
4. Log and plot optimization trajectory:
   - Organ probabilities over iterations (Slot 0, Slot 1, Slot 2).
   - Initial render vs Intermediate vs Final render vs Ground Truth.
   - Save to docs/results/assets/test_diff_organ_opt.png.
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from diffusion_based.models.plant_organ_array import (
    ORGAN_NONE, ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF, NUM_ORGAN_TYPES,
)
from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def run_diff_organ_opt_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DiffOrganOpt] Running on device: {device}")

    geo_builder = HeliosPlantGeometryBuilder()
    renderer = HeliosPyTorchRenderer(image_size=256)

    # -------------------------------------------------------------
    # 1. BUILD GROUND TRUTH (14D Part Tensor: 1 Stem, 1 Leaf)
    # -------------------------------------------------------------
    # Canonical 14D: [type(1), base(3), rot6d(6), scale(3), curvature(1)]
    # Stem: vertical at origin, length 0.15m, radius 0.005m
    # Leaf: attached near stem top, horizontal blade
    gt_14d = torch.zeros((2, 14), dtype=torch.float32, device=device)
    # Organ 0: Internode/Stem
    gt_14d[0, 0] = float(ORGAN_INTERNODE)
    gt_14d[0, 1:4] = torch.tensor([0.0, 0.0, 0.0], device=device)  # base
    gt_14d[0, 4:10] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device)  # fwd = +Z
    gt_14d[0, 10:13] = torch.tensor([0.15, 0.006, 0.0], device=device)  # length, radius
    gt_14d[0, 13] = 0.0

    # Organ 1: Leaf
    gt_14d[1, 0] = float(ORGAN_LEAF)
    gt_14d[1, 1:4] = torch.tensor([0.0, 0.0, 0.12], device=device)  # near top
    gt_14d[1, 4:10] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], device=device)  # tilted
    gt_14d[1, 10:13] = torch.tensor([0.10, 0.10, 0.10], device=device)  # leaf scale
    gt_14d[1, 13] = 0.0

    # Render Ground Truth RGB-D
    with torch.no_grad():
        gt_mesh = geo_builder.build_mesh_from_part_tensor(gt_14d, device=device)
        gt_rgbd = renderer(
            gt_mesh,
            azimuth_deg=45.0,
            elevation_deg=60.0,
            camera_height=1.0,
            include_depth=True,
            differentiable=False,
        )  # (4, 256, 256)
    print("[DiffOrganOpt] Ground truth RGB-D rendered.")

    # -------------------------------------------------------------
    # 2. INITIALIZE PERTURBED 26D TENSOR (3 Slots)
    # -------------------------------------------------------------
    # 26D: [logits(13), base(3), rot6d(6), scale(3), curvature(1)]
    # We deliberately swap the organ classes and add a spurious 3rd organ!
    init_26d = torch.zeros((3, 26), dtype=torch.float32, device=device)

    # Slot 0 (Stem location): Initialized with wrong logits -> LEAF!
    logits_0 = torch.full((13,), -2.0, device=device)
    logits_0[ORGAN_LEAF] = 3.0       # Strongly leaf!
    logits_0[ORGAN_INTERNODE] = -1.0 # Suppressed stem
    logits_0[ORGAN_NONE] = -2.0      # Exists
    init_26d[0, :13] = logits_0
    init_26d[0, 13:16] = gt_14d[0, 1:4].clone()
    init_26d[0, 16:22] = gt_14d[0, 4:10].clone()
    init_26d[0, 22:25] = torch.tensor([0.12, 0.008, 0.0], device=device)

    # Slot 1 (Leaf location): Initialized with wrong logits -> STEM!
    logits_1 = torch.full((13,), -2.0, device=device)
    logits_1[ORGAN_INTERNODE] = 3.0  # Strongly stem!
    logits_1[ORGAN_LEAF] = -1.0      # Suppressed leaf
    logits_1[ORGAN_NONE] = -2.0      # Exists
    init_26d[1, :13] = logits_1
    init_26d[1, 13:16] = gt_14d[1, 1:4].clone()
    init_26d[1, 16:22] = gt_14d[1, 4:10].clone()
    init_26d[1, 22:25] = torch.tensor([0.08, 0.08, 0.08], device=device)

    # Slot 2 (Empty position): Initialized with spurious existence!
    logits_2 = torch.full((13,), -2.0, device=device)
    logits_2[ORGAN_LEAF] = 2.5       # Spurious leaf!
    logits_2[ORGAN_NONE] = -1.5      # Exists!
    init_26d[2, :13] = logits_2
    init_26d[2, 13:16] = torch.tensor([0.12, 0.10, 0.05], device=device)  # off to the side
    init_26d[2, 16:22] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device)
    init_26d[2, 22:25] = torch.tensor([0.08, 0.08, 0.08], device=device)

    # Mask unused organ classes so probability strictly flows between NONE, STEM, and LEAF
    active_classes = [ORGAN_NONE, ORGAN_INTERNODE, ORGAN_LEAF]
    unused_mask = torch.ones(13, dtype=torch.bool, device=device)
    unused_mask[active_classes] = False
    init_26d[:, :13][:, unused_mask] = -1e4

    # Parameters to optimize: separate logits (lr=0.25) from physical meters (lr=0.002)
    opt_logits = nn.Parameter(init_26d[:, :13].clone())
    opt_geom = nn.Parameter(init_26d[:, 13:].clone())

    optimizer = optim.Adam([
        {'params': [opt_logits], 'lr': 0.20},
        {'params': [opt_geom], 'lr': 0.001},
    ])

    print("\n--- INITIAL STATE ---")
    with torch.no_grad():
        p0 = torch.softmax(opt_logits, dim=-1)
        print(f"Slot 0 (Stem pos): P(Leaf)={p0[0, ORGAN_LEAF]:.3f}, P(Stem)={p0[0, ORGAN_INTERNODE]:.3f}, P(None)={p0[0, ORGAN_NONE]:.3f}")
        print(f"Slot 1 (Leaf pos): P(Leaf)={p0[1, ORGAN_LEAF]:.3f}, P(Stem)={p0[1, ORGAN_INTERNODE]:.3f}, P(None)={p0[1, ORGAN_NONE]:.3f}")
        print(f"Slot 2 (Spurious): P(Leaf)={p0[2, ORGAN_LEAF]:.3f}, P(Stem)={p0[2, ORGAN_INTERNODE]:.3f}, P(None)={p0[2, ORGAN_NONE]:.3f}")

    # -------------------------------------------------------------
    # 3. OPTIMIZATION LOOP
    # -------------------------------------------------------------
    from diffusion_based.models.helios_pytorch_geometry import diff_mixture_to_part_tensor_14d

    num_steps = 60
    history = {
        'loss': [],
        'slot0_stem_p': [], 'slot0_leaf_p': [],
        'slot1_stem_p': [], 'slot1_leaf_p': [],
        'slot2_none_p': [], 'slot2_exist_p': [],
    }

    saved_renders = {}

    for step in range(num_steps + 1):
        optimizer.zero_grad()

        # Enforce unused classes stay -1e4
        with torch.no_grad():
            opt_logits[:, unused_mask] = -1e4

        opt_26d = torch.cat([opt_logits, opt_geom], dim=-1)

        # Step A: External Differentiable 26D -> 14D Conversion with Soft Existence
        part_14d, organ_exist = diff_mixture_to_part_tensor_14d(
            opt_26d, existence_threshold=0.005, return_existence=True
        )

        # Step B: Pure 14D Mesh Generation with Soft Existence
        mesh_dict = geo_builder.build_mesh_from_part_tensor(part_14d, existence=organ_exist, device=device)

        pred_rgbd = renderer(
            mesh_dict,
            azimuth_deg=45.0,
            elevation_deg=60.0,
            camera_height=1.0,
            include_depth=True,
            differentiable=True,
        )

        # Loss: RGB L1 + Depth L1
        rgb_loss = torch.mean(torch.abs(pred_rgbd[:3] - gt_rgbd[:3]))
        depth_loss = torch.mean(torch.abs(pred_rgbd[3:] - gt_rgbd[3:])) * 5.0
        total_loss = rgb_loss + depth_loss

        # Track history
        with torch.no_grad():
            probs = torch.softmax(opt_logits, dim=-1)
            history['loss'].append(total_loss.item())
            history['slot0_stem_p'].append(probs[0, ORGAN_INTERNODE].item())
            history['slot0_leaf_p'].append(probs[0, ORGAN_LEAF].item())
            history['slot1_stem_p'].append(probs[1, ORGAN_INTERNODE].item())
            history['slot1_leaf_p'].append(probs[1, ORGAN_LEAF].item())
            history['slot2_none_p'].append(probs[2, ORGAN_NONE].item())
            history['slot2_exist_p'].append((1.0 - probs[2, ORGAN_NONE]).item())

            if step in (0, 15, 30, 60):
                saved_renders[step] = pred_rgbd[:3].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

        if step < num_steps:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([opt_logits, opt_geom], max_norm=1.0)
            optimizer.step()

            if step % 10 == 0:
                print(f"Step {step:02d} | Loss: {total_loss.item():.4f} | "
                      f"Slot0 P(Stem)={probs[0, ORGAN_INTERNODE]:.2f}, "
                      f"Slot1 P(Leaf)={probs[1, ORGAN_LEAF]:.2f}, "
                      f"Slot2 P(None)={probs[2, ORGAN_NONE]:.2f}")

    print("\n--- FINAL CONVERGED STATE ---")
    with torch.no_grad():
        p_fin = torch.softmax(opt_logits, dim=-1)
        print(f"Slot 0 (Stem pos): P(Stem)={p_fin[0, ORGAN_INTERNODE]:.3f}, P(Leaf)={p_fin[0, ORGAN_LEAF]:.3f}, P(None)={p_fin[0, ORGAN_NONE]:.3f}")
        print(f"Slot 1 (Leaf pos): P(Stem)={p_fin[1, ORGAN_INTERNODE]:.3f}, P(Leaf)={p_fin[1, ORGAN_LEAF]:.3f}, P(None)={p_fin[1, ORGAN_NONE]:.3f}")
        print(f"Slot 2 (Spurious): P(None)={p_fin[2, ORGAN_NONE]:.3f}, P(Exist)={1.0 - p_fin[2, ORGAN_NONE]:.3f}")

    # -------------------------------------------------------------
    # 4. PLOT & SAVE DIAGNOSTIC VISUALIZATION
    # -------------------------------------------------------------
    os.makedirs("docs/results/assets", exist_ok=True)
    out_png = "docs/results/assets/test_diff_organ_opt.png"

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Visual Progression
    axes[0, 0].imshow(saved_renders[0])
    axes[0, 0].set_title("Step 0 (Inverted Types + Ghost)", fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(saved_renders[15])
    axes[0, 1].set_title("Step 15 (Morphing)", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(saved_renders[60])
    axes[0, 2].set_title("Step 60 (Converged)", fontsize=10)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(gt_rgbd[:3].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0, 3].set_title("Ground Truth Target", fontsize=10, fontweight='bold')
    axes[0, 3].axis('off')

    # Row 2: Optimization Curves
    axes[1, 0].plot(history['loss'], color='black', lw=2)
    axes[1, 0].set_title("Total Photometric Loss")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['slot0_stem_p'], label='Slot 0: P(Stem) [Target: 1.0]', color='green', lw=2)
    axes[1, 1].plot(history['slot0_leaf_p'], label='Slot 0: P(Leaf) [Initial: 0.9]', color='lightgreen', linestyle='--')
    axes[1, 1].set_title("Slot 0 Disambiguation (Target: Stem)")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(history['slot1_leaf_p'], label='Slot 1: P(Leaf) [Target: 1.0]', color='blue', lw=2)
    axes[1, 2].plot(history['slot1_stem_p'], label='Slot 1: P(Stem) [Initial: 0.9]', color='lightblue', linestyle='--')
    axes[1, 2].set_title("Slot 1 Disambiguation (Target: Leaf)")
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("Probability")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    axes[1, 3].plot(history['slot2_none_p'], label='Slot 2: P(None) [Target: 1.0]', color='red', lw=2)
    axes[1, 3].plot(history['slot2_exist_p'], label='Slot 2: P(Exist) [Initial: 0.9]', color='orange', linestyle='--')
    axes[1, 3].set_title("Slot 2 Pruning (Spurious Organ)")
    axes[1, 3].set_xlabel("Iteration")
    axes[1, 3].set_ylabel("Probability")
    axes[1, 3].legend(fontsize=8)
    axes[1, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"\n[DiffOrganOpt] Visual diagnostics saved to {out_png}")


if __name__ == "__main__":
    run_diff_organ_opt_test()
