"""
Compare Hybrid ResNet-Transformer Global VAE vs Pure Transformer Global VAE.

Generates visual and metric comparisons across:
- DAP 010 (Juvenile, N=94)
- DAP 050 (Canopy Development, N=1,158)
- DAP 090 (Mature Canopy, N=1,558)

Saves comparison plot to:
- docs/results/assets/fig_resnet_vs_pure_transformer_vae.png
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    T_COL_LENGTH,
    T_COL_RADIUS,
    T_COL_SCALE,
    T_COL_ORGAN_TYPE,
)
from diffusion_based.models.plant_global_vae import PlantGlobalVAE
from diffusion_based.models.plant_pure_transformer_vae import PlantPureTransformerVAE
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5).sum()
    union = np.logical_or(mask1 > 0.5, mask2 > 0.5).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing ResNet vs Pure Transformer VAE Comparison on {device}...")

    resnet_ckpt = "diffusion_based/checkpoints/plant_global_vae_best.pt"
    pure_ckpt = "diffusion_based/checkpoints/plant_pure_transformer_vae_best.pt"

    # 1. Model A: Hybrid ResNet + Transformer (4+4 layers)
    model_resnet = PlantGlobalVAE(latent_dim=512, hidden_dim=512, encoder_layers=4, decoder_layers=4).to(device)
    if os.path.exists(resnet_ckpt):
        ckpt = torch.load(resnet_ckpt, map_location=device)
        model_resnet.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded Hybrid ResNet Global VAE Checkpoint from {resnet_ckpt}")
    model_resnet.eval()

    # 2. Model B: Pure Transformer (6+6 layers, no separate ResNet)
    model_pure = PlantPureTransformerVAE(latent_dim=512, hidden_dim=512, ffn_dim=2048, encoder_layers=6, decoder_layers=6).to(device)
    if os.path.exists(pure_ckpt):
        ckpt = torch.load(pure_ckpt, map_location=device)
        model_pure.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded Pure Transformer Global VAE Checkpoint from {pure_ckpt}")
    model_pure.eval()

    renderer = HeliosPyTorchRenderer()

    test_stages = [
        {"dap": 10, "xml": "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_plant_0000.xml"},
        {"dap": 50, "xml": "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_plant_0000.xml"},
        {"dap": 90, "xml": "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_plant_0000.xml"},
    ]

    results = []

    for stage in test_stages:
        dap = stage["dap"]
        xml_path = os.path.join(repo_root, stage["xml"])
        if not os.path.exists(xml_path):
            continue

        gt_arr = PlantOrganArray.from_xml_file(xml_path)
        X_gt = gt_arr.tensor.to(device)
        N = X_gt.shape[0]

        def render_obj(arr_obj):
            mesh = renderer.geo_builder.build_mesh_from_organ_array(
                arr_obj, device=device, species="cowpea", leaf_mode="generic"
            )
            rgb_t = renderer.render_mesh(
                mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                focus_plant=True, differentiable=False
            )
            depth_t = renderer.render_depth(
                mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                focus_plant=True
            )
            return {
                "rgb": rgb_t.permute(1, 2, 0).clamp(0, 1).cpu().numpy(),
                "mask": (depth_t > 1e-4).float().cpu().numpy(),
            }

        # 0. GT Render
        out_gt = render_obj(gt_arr)

        with torch.no_grad():
            # 1. Hybrid ResNet-Transformer (512D)
            t0 = time.time()
            mu_res, _ = model_resnet.encode(X_gt.unsqueeze(0))
            recon_res = model_resnet.decode(mu_res, target_len=N, tree_x=X_gt.unsqueeze(0), hard_categoricals=True).squeeze(0)
            t_res = time.time() - t0
            recon_res[:, :11] = X_gt[:, :11]
            arr_res = PlantOrganArray(tensor=recon_res.cpu(), raw_metadata=gt_arr.raw_metadata)
            out_res = render_obj(arr_res)
            iou_res = compute_iou(out_gt["mask"], out_res["mask"])
            dim_mae_res = float((X_gt[:, [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE]] - recon_res[:, [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE]]).abs().mean().item())
            cls_acc_res = float((X_gt[:, T_COL_ORGAN_TYPE] == recon_res[:, T_COL_ORGAN_TYPE]).float().mean().item() * 100.0)

            # 2. Pure Transformer (512D)
            t0 = time.time()
            mu_pure, _ = model_pure.encode(X_gt.unsqueeze(0))
            recon_pure = model_pure.decode(mu_pure, target_len=N, tree_x=X_gt.unsqueeze(0), hard_categoricals=True).squeeze(0)
            t_pure = time.time() - t0
            recon_pure[:, :11] = X_gt[:, :11]
            arr_pure = PlantOrganArray(tensor=recon_pure.cpu(), raw_metadata=gt_arr.raw_metadata)
            out_pure = render_obj(arr_pure)
            iou_pure = compute_iou(out_gt["mask"], out_pure["mask"])
            dim_mae_pure = float((X_gt[:, [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE]] - recon_pure[:, [T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE]]).abs().mean().item())
            cls_acc_pure = float((X_gt[:, T_COL_ORGAN_TYPE] == recon_pure[:, T_COL_ORGAN_TYPE]).float().mean().item() * 100.0)

        results.append({
            "dap": dap,
            "N": N,
            "out_gt": out_gt,
            "out_res": out_res,
            "out_pure": out_pure,
            "iou_res": iou_res,
            "iou_pure": iou_pure,
            "dim_mae_res": dim_mae_res * 1000.0,
            "dim_mae_pure": dim_mae_pure * 1000.0,
            "cls_acc_res": cls_acc_res,
            "cls_acc_pure": cls_acc_pure,
            "t_res": t_res * 1000.0,
            "t_pure": t_pure * 1000.0,
        })

        print(f"\n=== DAP {dap:03d} (N={N} Organs) ===")
        print(f"  Hybrid ResNet+Transformer: Dim MAE={dim_mae_res*1000.0:5.2f}mm | Mask IoU={iou_res:.4f} | Cls Acc={cls_acc_res:.1f}% | Time={t_res*1000.0:5.1f}ms")
        print(f"  Pure Transformer         : Dim MAE={dim_mae_pure*1000.0:5.2f}mm | Mask IoU={iou_pure:.4f} | Cls Acc={cls_acc_pure:.1f}% | Time={t_pure*1000.0:5.1f}ms")

    # Plot 3-column figure
    num_rows = len(results)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows), facecolor='black')
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)

    titles = [
        "1. Ground Truth 3D Plant\n(Exact Helios XML 40D)",
        "2. Hybrid ResNet + Transformer (512D)\n(Pointwise ResNet Channel Mixers + 4L Transformer)",
        "3. Pure Transformer (512D)\n(6L Encoder + 6L Decoder, Unified 2048D FFN)",
    ]

    for col, title in enumerate(titles):
        axes[0, col].set_title(title, color='white' if col == 0 else ('#58a6ff' if col == 1 else '#7ee787'), fontsize=13, fontweight='bold', pad=12)

    for row, res in enumerate(results):
        dap = res["dap"]
        N = res["N"]

        # GT
        axes[row, 0].imshow(res["out_gt"]["rgb"])
        axes[row, 0].axis('off')
        axes[row, 0].set_ylabel(f"DAP {dap:03d}\n({N} Organs)", color='white', fontsize=12, fontweight='bold')

        # Hybrid ResNet
        axes[row, 1].imshow(res["out_res"]["rgb"])
        axes[row, 1].axis('off')
        axes[row, 1].text(
            0.03, 0.05,
            f"Dim: {res['dim_mae_res']:.1f}mm | IoU: {res['iou_res']:.3f}\nCls: {res['cls_acc_res']:.1f}% | {res['t_res']:.1f}ms",
            transform=axes[row, 1].transAxes,
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='black', edgecolor='#58a6ff', alpha=0.85)
        )

        # Pure Transformer
        axes[row, 2].imshow(res["out_pure"]["rgb"])
        axes[row, 2].axis('off')
        axes[row, 2].text(
            0.03, 0.05,
            f"Dim: {res['dim_mae_pure']:.1f}mm | IoU: {res['iou_pure']:.3f}\nCls: {res['cls_acc_pure']:.1f}% | {res['t_pure']:.1f}ms",
            transform=axes[row, 2].transAxes,
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='black', edgecolor='#7ee787', alpha=0.85)
        )

    plt.suptitle("Architecture Benchmark: Hybrid ResNet+Transformer vs Pure Transformer Global VAE (512D)", color='white', fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_img = "docs/results/assets/fig_resnet_vs_pure_transformer_vae.png"
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"\n[OK] Saved Comparison Figure to: {out_img}")


if __name__ == "__main__":
    main()
