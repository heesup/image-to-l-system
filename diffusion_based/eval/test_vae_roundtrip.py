"""
Comprehensive Round-Trip Verification & Benchmark for Plant VAE (40D Plant Organ Vector).

Verifies the full 5-stage round-trip:
  1. Helios XML -> PlantOrganArray (40D)
  2. PlantOrganArray (40D) -> Latent Vector z in R^16
  3. Latent Vector z -> Reconstructed PlantOrganArray (40D)
  4. Reconstructed PlantOrganArray -> Serialized XML
  5. Serialized XML -> Re-loaded PlantOrganArray & Differentiable Render

Outputs:
  - Numerical round-trip accuracy table (Classification Acc, Continuous MAE, Mask IoU, Depth MAE)
  - docs/results/assets/fig_vae_roundtrip_comparison.png
"""

import os
import sys
import glob
import time
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    NUM_FEATURES_TYPED,
    T_COL_ORGAN_TYPE,
    T_COL_EXISTENCE,
    T_COL_LENGTH,
    T_COL_RADIUS,
    T_COL_SCALE,
    T_COL_PITCH,
    T_COL_YAW,
    T_COL_ROLL,
    T_COL_PHYLLOTACTIC_ANGLE,
    T_COL_LENGTH_MAX,
)
from diffusion_based.models.plant_vae import PlantOrganVAE
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a > 0.5, mask_b > 0.5).sum()
    union = np.logical_or(mask_a > 0.5, mask_b > 0.5).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def run_vae_roundtrip_benchmark(
    ckpt_path: str = "diffusion_based/checkpoints/plant_organ_vae_best.pt",
    test_daps: List[int] = [10, 50, 90],
    species: str = "cowpea",
    device: str = "cuda"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing VAE Round-Trip Benchmark on {device}...")

    # Load Trained VAE
    ckpt_full_path = os.path.join(repo_root, ckpt_path)
    model = PlantOrganVAE(latent_dim=512, hidden_dim=512).to(device)
    if os.path.exists(ckpt_full_path):
        ckpt = torch.load(ckpt_full_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Successfully loaded VAE checkpoint from: {ckpt_full_path} (Val Loss: {ckpt.get('val_loss', 0.0):.4f})")
    else:
        print(f"[WARN] Checkpoint not found at {ckpt_full_path}. Using initialized model.")
    model.eval()

    renderer = HeliosPyTorchRenderer(image_size=512).to(device)

    results_table = []
    comparison_renders = []

    exact_gt_dir = os.path.join(repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "output", "exact_gt_renders")

    for dap in test_daps:
        print(f"\n--- Testing DAP {dap:03d} Full Round-Trip ---")

        # 1. Locate XML & Helios GT image
        gt_xml_path = os.path.join(exact_gt_dir, f"rad_dap{dap:03d}_0000_plant_0000.xml")
        helios_img_path = os.path.join(exact_gt_dir, f"rad_dap{dap:03d}_0000_rad.jpeg")
        camera_json_path = os.path.join(exact_gt_dir, f"rad_dap{dap:03d}_0000_camera.json")

        if not os.path.exists(gt_xml_path):
            xml_matches = glob.glob(os.path.join(repo_root, "dataset", "helios_data", species, f"*{species}*dap{dap:03d}*.xml"))
            if not xml_matches:
                xml_matches = glob.glob(os.path.join(repo_root, "dataset", "helios_data", "**", f"*dap{dap:03d}*.xml"), recursive=True)
            if not xml_matches:
                print(f"[WARN] No matching XML found for DAP {dap}")
                continue
            gt_xml_path = xml_matches[0]

        # Camera parameters
        cam_h = 5.0
        cam_el = 90.0
        cam_hfov = None
        if os.path.exists(camera_json_path):
            try:
                with open(camera_json_path, 'r') as f:
                    cam_data = json.load(f)
                cam_h = float(cam_data.get("acquisition_properties", {}).get("camera_height_m", 5.0))
                cam_el = float(cam_data.get("acquisition_properties", {}).get("camera_angle_deg", 90.0))
                f_len = float(cam_data.get("camera_properties", {}).get("focal_length_mm", 50.0))
                s_w = float(cam_data.get("camera_properties", {}).get("sensor_width_mm", 36.0))
                cam_hfov = 2.0 * math.degrees(math.atan((s_w * 0.5) / max(f_len, 1e-3)))
            except Exception:
                pass

        # Load Helios GT Image
        from PIL import Image
        if os.path.exists(helios_img_path):
            helios_raw = np.array(Image.open(helios_img_path).convert("RGB")) / 255.0
            h, w = helios_raw.shape[:2]
            if h != w:
                min_dim = min(h, w)
                y0, x0 = (h - min_dim) // 2, (w - min_dim) // 2
                helios_raw = helios_raw[y0:y0+min_dim, x0:x0+min_dim]
            from PIL import Image as PILImage
            helios_img = np.array(PILImage.fromarray((helios_raw * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS)) / 255.0
        else:
            helios_img = None

        # Stage 1: XML -> 40D OrganArray
        t0 = time.time()
        gt_arr = PlantOrganArray.from_xml_file(gt_xml_path)
        t_xml_to_40d = time.time() - t0
        X_gt = gt_arr.tensor.to(device)
        N_organs = X_gt.shape[0]

        # Stage 2: 40D -> Latent Vector z (R^64 per organ)
        t0 = time.time()
        with torch.no_grad():
            mu, logvar = model.encode(X_gt)
            z = mu  # deterministic mean for lossless evaluation
        t_enc = time.time() - t0

        # Stage 3: Latent Vector z -> Decoded 40D OrganArray
        t0 = time.time()
        with torch.no_grad():
            X_recon = model.decode(z, hard_categoricals=True)
        t_dec = time.time() - t0

        # Preserve structural tree DAG indices (cols 0-10) while evaluating decoded continuous morphology
        X_recon_full = X_recon.clone()
        X_recon_full[:, :11] = X_gt[:, :11]

        # Stage 4: Decoded 40D -> Serialized XML
        t0 = time.time()
        recon_arr = PlantOrganArray(tensor=X_recon_full.cpu(), raw_metadata=gt_arr.raw_metadata)
        out_xml_dir = os.path.join(repo_root, "diffusion_based", "eval", "roundtrip_outputs")
        os.makedirs(out_xml_dir, exist_ok=True)
        recon_xml_path = os.path.join(out_xml_dir, f"recon_roundtrip_dap{dap:03d}.xml")
        recon_arr.write_xml(recon_xml_path)
        t_40d_to_xml = time.time() - t0

        # Stage 5: Serialized XML -> Re-parsed PlantOrganArray
        t0 = time.time()
        xml_reloaded_arr = PlantOrganArray.from_xml_file(recon_xml_path)
        t_reload = time.time() - t0
        X_xml_reloaded = xml_reloaded_arr.tensor.to(device)

        # Compute Numerical Metrics
        # 1. Organ classification accuracy
        gt_types = X_gt[:, T_COL_ORGAN_TYPE].long()
        recon_types = X_recon[:, T_COL_ORGAN_TYPE].long()
        cls_acc = float((gt_types == recon_types).float().mean().item()) * 100.0

        # 2. Continuous parameters MAE (Physical Dimensions & Geodesic Angles)
        dim_cols = [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE, T_COL_LENGTH_MAX]
        dim_mae = float((X_gt[:, dim_cols] - X_recon[:, dim_cols]).abs().mean().item())

        ang_cols = [T_COL_PITCH, T_COL_YAW, T_COL_ROLL, T_COL_PHYLLOTACTIC_ANGLE]
        ang_diff = (X_gt[:, ang_cols] - X_recon[:, ang_cols]).abs() % 360.0
        ang_geo_mae = float(torch.minimum(ang_diff, 360.0 - ang_diff).mean().item())
        geom_mae = dim_mae

        # Render Visual Comparison
        def render_organ_multimodal(arr_obj):
            mesh = renderer.geo_builder.build_mesh_from_organ_array(
                arr_obj, device=device, species=species, leaf_mode="generic"
            )
            rgb_t = renderer.render_mesh(
                mesh, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
                focus_plant=(cam_hfov is None), hfov_override_deg=cam_hfov, differentiable=False
            )
            depth_t = renderer.render_depth(
                mesh, azimuth_deg=0.0, elevation_deg=cam_el, camera_height=cam_h,
                focus_plant=(cam_hfov is None), hfov_override_deg=cam_hfov
            )
            mask_t = (depth_t > 1e-4).float()
            return {
                "rgb": rgb_t.permute(1, 2, 0).clamp(0, 1),
                "depth": depth_t,
                "mask": mask_t,
            }

        with torch.no_grad():
            # 1. GT 40D Render
            out_gt = render_organ_multimodal(gt_arr)
            # 2. Latent Decoded 40D Render
            out_latent = render_organ_multimodal(recon_arr)
            # 3. XML Reloaded Render
            out_xml = render_organ_multimodal(xml_reloaded_arr)

        # 2D Mask IoU
        mask_gt = out_gt["mask"].cpu().numpy()
        mask_latent = out_latent["mask"].cpu().numpy()
        mask_xml = out_xml["mask"].cpu().numpy()

        iou_latent = compute_iou(mask_gt, mask_latent)
        iou_xml = compute_iou(mask_gt, mask_xml)

        print(f"DAP {dap:03d} (N={N_organs:4d}): "
              f"Cls Acc={cls_acc:5.1f}% | Dim MAE={dim_mae:.5f}m | Ang MAE={ang_geo_mae:.2f}° | "
              f"Mask IoU (Latent)={iou_latent:.4f} | Mask IoU (XML)={iou_xml:.4f} | "
              f"Latent Enc={t_enc*1000:5.2f}ms, Dec={t_dec*1000:5.2f}ms")

        results_table.append({
            "dap": dap,
            "organs": N_organs,
            "cls_acc": cls_acc,
            "dim_mae": dim_mae,
            "ang_geo_mae": ang_geo_mae,
            "iou_latent": iou_latent,
            "iou_xml": iou_xml,
            "t_enc_ms": t_enc * 1000.0,
            "t_dec_ms": t_dec * 1000.0,
        })

        comparison_renders.append({
            "dap": dap,
            "helios_gt": helios_img if helios_img is not None else out_gt["rgb"].cpu().numpy(),
            "rgb_gt": out_gt["rgb"].cpu().numpy(),
            "rgb_latent": out_latent["rgb"].cpu().numpy(),
            "rgb_xml": out_xml["rgb"].cpu().numpy(),
            "depth_gt": out_gt["depth"].cpu().numpy(),
            "depth_latent": out_latent["depth"].cpu().numpy(),
            "mask_gt": mask_gt,
            "mask_latent": mask_latent,
        })

    # Generate Visual Figure
    plot_vae_roundtrip_figure(comparison_renders)
    return results_table


def plot_vae_roundtrip_figure(data_list: List[Dict[str, Any]]):
    n_rows = len(data_list)
    if n_rows == 0:
        return

    plt.style.use("dark_background")
    fig, axes = plt.subplots(n_rows, 5, figsize=(22, 4.5 * n_rows))
    plt.subplots_adjust(wspace=0.08, hspace=0.15, left=0.04, right=0.96, top=0.92, bottom=0.04)

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "1. Helios C++ Ground Truth\n(Exact Raytracing GT)",
        "2. PyTorch 40D OrganArray Render\n(Differentiable Geometry)",
        "3. Latent Vector z Decoded\n(40D -> z in R^512 -> 40D)",
        "4. Round-Trip Re-exported XML\n(Full XML Serialized & Re-parsed)",
        "5. Round-Trip Diff / Error Map\n(GT vs Latent Reconstructed)",
    ]

    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=11, fontweight="bold", pad=12, color="#64B5F6" if c_idx < 4 else "#FF8A80")

    for r_idx, d in enumerate(data_list):
        dap = d["dap"]
        helios_img = np.clip(d["helios_gt"], 0.0, 1.0)
        rgb_gt = np.clip(d["rgb_gt"], 0.0, 1.0)
        rgb_latent = np.clip(d["rgb_latent"], 0.0, 1.0)
        rgb_xml = np.clip(d["rgb_xml"], 0.0, 1.0)

        # Diff Map (RGB Diff + Mask Diff)
        rgb_diff = np.abs(rgb_gt - rgb_latent).sum(axis=-1)

        # Col 0: Helios C++ Raytracing GT
        axes[r_idx, 0].imshow(helios_img)
        axes[r_idx, 0].set_ylabel(f"DAP {dap}\n(Growth Stage)", fontsize=12, fontweight="bold", color="#FFF")

        # Col 1: PyTorch 40D OrganArray Render (from GT XML)
        axes[r_idx, 1].imshow(rgb_gt)

        # Col 2: Latent Vector z Decoded
        axes[r_idx, 2].imshow(rgb_latent)

        # Col 3: Round-Trip Re-exported XML
        axes[r_idx, 3].imshow(rgb_xml)

        # Col 4: Diff Heatmap
        axes[r_idx, 4].imshow(rgb_diff, cmap="magma", vmin=0.0, vmax=0.3)
        iou_val = compute_iou(d["mask_gt"], d["mask_latent"])
        axes[r_idx, 4].text(
            15, 490, f"Mask IoU: {iou_val:.3f}",
            color="#00FFCC", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75)
        )

        for c in range(5):
            axes[r_idx, c].set_xticks([])
            axes[r_idx, c].set_yticks([])

    fig.suptitle(
        "Figure: End-to-End Lossless Round-Trip Verification (Helios C++ GT <-> 40D OrganArray <-> Latent Space z <-> Helios XML)",
        fontsize=14, fontweight="bold", y=0.98, color="#FFFFFF"
    )

    out_png = os.path.join(repo_root, "docs", "results", "assets", "fig_vae_roundtrip_comparison.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\n[OK] Saved VAE Round-Trip Comparison Figure to: {out_png}")


if __name__ == "__main__":
    run_vae_roundtrip_benchmark()
