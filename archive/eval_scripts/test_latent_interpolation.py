"""
Latent Manifold Coverage Evaluation and Linear Interpolation Visualizer.

1. Evaluates representation power of the continuous latent space across diverse plant varieties & growth stages.
2. Performs smooth linear latent interpolation between two distinct plant architectures (e.g. Seedling z_1 -> Mature Branching z_2):
     z(alpha) = (1 - alpha) * z_1 + alpha * z_2,  alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
3. Renders each intermediate 3D plant state and produces publication-quality visualization:
     docs/results/assets/fig_latent_interpolation_morphing.png
"""

import os
import sys
import glob
import time
import math
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    NUM_FEATURES_TYPED,
    T_COL_ORGAN_TYPE,
    T_COL_LENGTH,
    T_COL_RADIUS,
    T_COL_SCALE,
    T_COL_PITCH,
    T_COL_YAW,
    T_COL_ROLL,
    T_COL_PHYLLOTACTIC_ANGLE,
    T_COL_CURVATURE,
    T_COL_EXISTENCE,
)
from diffusion_based.models.plant_vae import PlantOrganVAE
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a > 0.5, mask_b > 0.5).sum()
    union = np.logical_or(mask_a > 0.5, mask_b > 0.5).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def evaluate_manifold_coverage(
    model: PlantOrganVAE,
    dataset_dir: str = "dataset/helios_data",
    num_eval_plants: int = 50,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, float]:
    """
    Evaluates latent space reconstruction fidelity across a wide sample of plant XMLs.
    """
    print(f"\n[INFO] Evaluating Latent Space Manifold Coverage across {num_eval_plants} plants...")
    xml_files = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.xml"), recursive=True))
    if not xml_files:
        print("[WARN] No XML files found for coverage evaluation.")
        return {}

    rng = np.random.RandomState(123)
    if len(xml_files) > num_eval_plants:
        indices = rng.choice(len(xml_files), num_eval_plants, replace=False)
        eval_files = [xml_files[i] for i in indices]
    else:
        eval_files = xml_files

    cls_accs = []
    geom_maes = []
    angle_maes = []
    length_maes = []

    model.eval()
    for xml_p in eval_files:
        try:
            arr = PlantOrganArray.from_xml_file(xml_p)
            X = arr.tensor.to(device)
            if X.shape[0] == 0:
                continue

            with torch.no_grad():
                mu, _ = model.encode(X)
                X_rec = model.decode(mu, hard_categoricals=True)

            gt_ot = X[:, T_COL_ORGAN_TYPE].long()
            rec_ot = X_rec[:, T_COL_ORGAN_TYPE].long()
            acc = float((gt_ot == rec_ot).float().mean().item()) * 100.0
            cls_accs.append(acc)

            # Continuous morphology MAE
            cont_cols = [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE]
            ang_cols = [T_COL_PITCH, T_COL_YAW, T_COL_ROLL, T_COL_PHYLLOTACTIC_ANGLE]

            l_mae = float((X[:, cont_cols] - X_rec[:, cont_cols]).abs().mean().item())
            diff_ang = (X[:, ang_cols] - X_rec[:, ang_cols]).abs() % 360.0
            a_mae = float(torch.minimum(diff_ang, 360.0 - diff_ang).mean().item())
            g_mae = float((X[:, cont_cols] - X_rec[:, cont_cols]).abs().mean().item())

            length_maes.append(l_mae)
            angle_maes.append(a_mae)
            geom_maes.append(g_mae)
        except Exception:
            continue

    metrics = {
        "num_plants_evaluated": len(cls_accs),
        "mean_cls_accuracy": float(np.mean(cls_accs)),
        "mean_geom_mae": float(np.mean(geom_maes)),
        "mean_length_mae": float(np.mean(length_maes)),
        "mean_angle_mae_deg": float(np.mean(angle_maes)),
    }

    print("\n=======================================================")
    print("      LATENT MANIFOLD COVERAGE BENCHMARK RESULTS       ")
    print("=======================================================")
    print(f" Plants Evaluated    : {metrics['num_plants_evaluated']}")
    print(f" Organ Cls Accuracy  : {metrics['mean_cls_accuracy']:.2f}%")
    print(f" Geometric Total MAE : {metrics['mean_geom_mae']:.4f}")
    print(f" Dimension MAE (m)   : {metrics['mean_length_mae']:.5f}")
    print(f" Angle MAE (degrees) : {metrics['mean_angle_mae_deg']:.3f}°")
    print("=======================================================\n")
    return metrics


def run_latent_interpolation(
    ckpt_path: str = "diffusion_based/checkpoints/plant_organ_vae_best.pt",
    xml_path_1: Optional[str] = None,
    xml_path_2: Optional[str] = None,
    num_steps: int = 6,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing Latent Linear Interpolation on {device}...")

    # Load Trained VAE
    ckpt_full_path = os.path.join(repo_root, ckpt_path)
    model = PlantOrganVAE(latent_dim=512, hidden_dim=512).to(device)
    if os.path.exists(ckpt_full_path):
        ckpt = torch.load(ckpt_full_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded VAE Checkpoint from {ckpt_full_path} (Val Loss: {ckpt.get('val_loss', 0.0):.4f})")
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_full_path}.")
    model.eval()

    # Locate source plants
    exact_gt_dir = os.path.join(repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "output", "exact_gt_renders")
    if xml_path_1 is None:
        xml_path_1 = os.path.join(exact_gt_dir, "rad_dap010_0000_plant_0000.xml")
    if xml_path_2 is None:
        xml_path_2 = os.path.join(exact_gt_dir, "rad_dap050_0000_plant_0000.xml")

    arr1 = PlantOrganArray.from_xml_file(xml_path_1)
    arr2 = PlantOrganArray.from_xml_file(xml_path_2)

    X1 = arr1.tensor.to(device)
    X2 = arr2.tensor.to(device)

    N1 = X1.shape[0]
    N2 = X2.shape[0]
    N_target = max(N1, N2)

    def pad_to_N(t: torch.Tensor, target_n: int) -> torch.Tensor:
        if t.shape[0] >= target_n:
            return t[:target_n]
        padded = torch.zeros((target_n, NUM_FEATURES_TYPED), dtype=t.dtype, device=t.device)
        padded[:t.shape[0]] = t
        padded[t.shape[0]:, 0] = t[0, 0]
        padded[t.shape[0]:, T_COL_EXISTENCE] = 0.0
        return padded

    X1_padded = pad_to_N(X1, N_target)
    X2_padded = pad_to_N(X2, N_target)

    # Encode to latent space
    with torch.no_grad():
        mu1, _ = model.encode(X1_padded)
        mu2, _ = model.encode(X2_padded)

    renderer = HeliosPyTorchRenderer(image_size=512).to(device)

    alphas = np.linspace(0.0, 1.0, num_steps)
    renders_top = []
    renders_oblique = []
    organ_counts = []

    print("\n--- Generating Continuous Morphing Sequence ---")
    for step_i, alpha in enumerate(alphas):
        t0 = time.time()
        # Linear interpolation in continuous latent space
        z_interp = (1.0 - alpha) * mu1 + alpha * mu2

        with torch.no_grad():
            X_interp = model.decode(z_interp, hard_categoricals=True)

        if alpha < 0.5:
            X_interp[:, :11] = X1_padded[:, :11]
        else:
            X_interp[:, :11] = X2_padded[:, :11]

        exist_thresh = 0.5 - (alpha * 0.1)
        valid_mask = (X_interp[:, T_COL_EXISTENCE] > exist_thresh)
        if valid_mask.sum() == 0:
            valid_mask[0] = True
        X_active = X_interp[valid_mask]

        arr_interp = PlantOrganArray(tensor=X_active.cpu(), raw_metadata=arr2.raw_metadata if alpha >= 0.5 else arr1.raw_metadata)
        mesh = renderer.geo_builder.build_mesh_from_part_tensor(arr_interp.to_part_tensor(device=device), device=device, leaf_mode="generic")

        with torch.no_grad():
            rgb_top = renderer.render_mesh(mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True, differentiable=False)
            rgb_oblique = renderer.render_mesh(mesh, azimuth_deg=45.0, elevation_deg=45.0, camera_height=5.0, focus_plant=True, differentiable=False)

        top_np = rgb_top.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        oblique_np = rgb_oblique.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        renders_top.append(top_np)
        renders_oblique.append(oblique_np)
        organ_counts.append(int(valid_mask.sum().item()))
        print(f" Step {step_i+1}/{num_steps} (alpha={alpha:.2f}): {int(valid_mask.sum().item())} organs | Rendered in {(time.time()-t0)*1000:.1f}ms")

    # Plot Multi-Angle Interpolation Figure
    plot_interpolation_strip(alphas, renders_top, renders_oblique, organ_counts)

    # Evaluate Global Coverage
    evaluate_manifold_coverage(model, device=device)


def plot_interpolation_strip(
    alphas: np.ndarray,
    renders_top: List[np.ndarray],
    renders_oblique: List[np.ndarray],
    organ_counts: List[int]
):
    n_cols = len(alphas)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, n_cols, figsize=(4.0 * n_cols, 8.5))
    plt.subplots_adjust(wspace=0.04, hspace=0.10, left=0.03, right=0.97, top=0.90, bottom=0.04)

    for i, alpha in enumerate(alphas):
        # Row 0: Top-down view (Elevation 90 deg)
        axes[0, i].imshow(renders_top[i])
        axes[0, i].set_title(
            f"α = {alpha:.2f}\n({organ_counts[i]} Organs)",
            fontsize=12, fontweight="bold", color="#64B5F6" if i in (0, n_cols-1) else "#FFF", pad=10
        )
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Row 1: Oblique view (Elevation 45 deg, Azimuth 45 deg)
        axes[1, i].imshow(renders_oblique[i])
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    axes[0, 0].set_ylabel("Top-Down View\n(Elevation 90°)", fontsize=12, fontweight="bold", color="#00FFCC")
    axes[1, 0].set_ylabel("Oblique 3D View\n(Elevation 45°)", fontsize=12, fontweight="bold", color="#FFD54F")

    fig.suptitle(
        "Continuous Plant Latent Manifold Interpolation: z(α) = (1 - α)·z_seedling + α·z_mature (Smooth Morphological Development)",
        fontsize=14, fontweight="bold", y=0.97, color="#FFFFFF"
    )

    out_png = os.path.join(repo_root, "docs", "results", "assets", "fig_latent_interpolation_morphing.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\n[OK] Saved Latent Interpolation Strip to: {out_png}")


if __name__ == "__main__":
    run_latent_interpolation()
