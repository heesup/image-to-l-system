"""
Visual & Quantitative Comparison: Helios Ground Truth vs 40D Kinematics Renderer vs Pure 14D Direct Part Renderer.

Compares:
1. Helios C++ Ground Truth Image
2. 40D Phytomer Kinematics PyTorch Differentiable Renderer
3. Pure 14D Direct Part PyTorch Differentiable Renderer (Direct Spatial Assembly without Tree Traversal)
"""

import json
import math
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE, ORGAN_PETIOLE,
    ORGAN_LEAF, ORGAN_BUD, ORGAN_PEDUNCLE, ORGAN_FLOWER,
    P14_COL_ORGAN_TYPE, P14_COL_BASE_X, P14_COL_BASE_Y, P14_COL_BASE_Z,
    P14_COL_ROT_0, P14_COL_ROT_5, P14_COL_SCALE_X, P14_COL_SCALE_Y, P14_COL_SCALE_Z,
    P14_COL_EXISTENCE, rotation_6d_to_matrix,
    T_COL_CHILD_INDEX, T_COL_BUD_STATE,
)
from diffusion_based.models.helios_pytorch_geometry import (
    HeliosPlantGeometryBuilder, generate_cone_tube_mesh_torch, compute_face_normals_torch
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer, compute_focus_plant_camera




def compute_metrics(img1: np.ndarray, img2: np.ndarray):
    """Computes MAE, MSE, and SSIM between two (H, W, 3) float images in [0, 1]."""
    mae = np.mean(np.abs(img1 - img2))
    mse = np.mean((img1 - img2) ** 2)

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    s1 = np.var(img1)
    s2 = np.var(img2)
    s12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
    return float(mae), float(mse), float(ssim)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 14D vs 40D vs Helios GT comparison on {device}...")

    base_dir = "/home/lion397/codes/image-to-l-system/Digital-Crops/projects/syntheticdata_generation/build/output"
    samples = [
        {"name": "DAP 10 (Vegetative)", "xml": "dap10_gt_0000_plant_0000.xml", "gt_img": "dap10_gt_0000_vis.jpeg", "params": "dap10_gt_0000_params.json"},
        {"name": "DAP 30 (Branching)", "xml": "dap30_gt_0000_plant_0000.xml", "gt_img": "dap30_gt_0000_vis.jpeg", "params": "dap30_gt_0000_params.json"},
        {"name": "DAP 50 (Canopy)", "xml": "dap50_gt_0000_plant_0000.xml", "gt_img": "dap50_gt_0000_vis.jpeg", "params": "dap50_gt_0000_params.json"},
        {"name": "DAP 100 (Flowering & Pods)", "xml": "dap100_gt_0000_plant_0000.xml", "gt_img": "dap100_gt_0000_vis.jpeg", "params": "dap100_gt_0000_params.json"},
    ]

    renderer = HeliosPyTorchRenderer(image_size=512)

    fig, axes = plt.subplots(len(samples), 4, figsize=(18, 4.5 * len(samples)))
    fig.patch.set_facecolor("#121212")

    for row_i, sample in enumerate(samples):
        xml_path = os.path.join(base_dir, sample["xml"])
        gt_img_path = os.path.join(base_dir, sample["gt_img"])
        params_path = os.path.join(base_dir, sample["params"])

        with open(xml_path, "r") as f:
            xml_str = f.read()
        with open(params_path, "r") as f:
            params = json.load(f)

        cam_pos = params.get("camera", {}).get("positioning", {})
        azimuth_deg = float(cam_pos.get("azimuth_angle", 0.0))
        camera_height = float(cam_pos.get("camera_height", 5.0))
        focus_plant = True

        # 1. Load GT image
        gt_pil = Image.open(gt_img_path).convert("RGB").resize((512, 512))
        gt_np = np.array(gt_pil, dtype=np.float32) / 255.0

        # 2. 40D Hierarchical Kinematics Render
        array_40d = PlantOrganArray.from_xml_string_typed(xml_str)
        rendered_40d = renderer.render_organ_array(
            array_40d, device=device, azimuth_deg=azimuth_deg, elevation_deg=90.0,
            camera_height=camera_height, focus_plant=focus_plant
        )
        img_40d = rendered_40d.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)

        # 3. Pure 14D Direct Part Render (using official HeliosPyTorchRenderer.render_part_tensor_14d)
        part_tensor_14d = array_40d.to_part_tensor_14d(device=device)
        rendered_14d = renderer.render_part_tensor_14d(
            part_tensor_14d,
            template_organ_array=array_40d,
            device=device,
            azimuth_deg=azimuth_deg,
            elevation_deg=90.0,
            camera_height=camera_height,
            focus_plant=focus_plant,
            use_kinematics_tree=False,
        )
        img_14d = rendered_14d.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)

        # 4. Difference Map (Pure 14D vs 40D)
        diff_14d_40d = np.abs(img_14d - img_40d)
        diff_vis = diff_14d_40d * 5.0  # amplify 5x for visual inspection

        # Metrics
        mae_14d_40d, mse_14d_40d, ssim_14d_40d = compute_metrics(img_14d, img_40d)
        mae_14d_gt, mse_14d_gt, ssim_14d_gt = compute_metrics(img_14d, gt_np)

        print(f"[{sample['name']}] N={part_tensor_14d.shape[0]} organs:")
        print(f"   14D Direct vs 40D Tree:  MAE={mae_14d_40d:.6f}, MSE={mse_14d_40d:.6e}, SSIM={ssim_14d_40d:.4f}")
        print(f"   14D Direct vs Helios GT: MAE={mae_14d_gt:.6f}, MSE={mse_14d_gt:.6e}, SSIM={ssim_14d_gt:.4f}")

        # Plot Row
        ax0 = axes[row_i, 0]
        ax0.imshow(gt_np)
        ax0.set_title(f"{sample['name']}\nHelios C++ Ground Truth", color="white", fontsize=11, fontweight="bold")
        ax0.axis("off")

        ax1 = axes[row_i, 1]
        ax1.imshow(img_40d)
        ax1.set_title(f"40D Tree Kinematics Render\n(Hierarchical Phytomer)", color="cyan", fontsize=11, fontweight="bold")
        ax1.axis("off")

        ax2 = axes[row_i, 2]
        ax2.imshow(img_14d)
        ax2.set_title(f"Pure 14D Direct Part Render\n(Direct Part Assembly, No Tree)", color="#00ff88", fontsize=11, fontweight="bold")
        ax2.axis("off")

        ax3 = axes[row_i, 3]
        ax3.imshow(diff_vis.clip(0.0, 1.0))
        ax3.set_title(f"14D Direct vs 40D Diff (5x amp)\nMAE: {mae_14d_40d:.4f} | SSIM: {ssim_14d_40d:.4f}", color="#ffaa00", fontsize=11, fontweight="bold")
        ax3.axis("off")

    plt.tight_layout()
    out_dir = "/home/lion397/codes/image-to-l-system/docs/results/assets"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig2_14d_part_renderer_identity_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved comparison figure to: {out_path}")

    # Copy to artifact temp storage for embedding
    artifact_dir = "/home/lion397/.gemini/antigravity-ide/brain/c148742b-205e-4e0f-8722-f0c0dbedcc27"
    artifact_path = os.path.join(artifact_dir, "fig2_14d_part_renderer_identity_comparison.png")
    import shutil
    shutil.copy(out_path, artifact_path)
    print(f"Copied to artifact directory: {artifact_path}")


if __name__ == "__main__":
    main()
