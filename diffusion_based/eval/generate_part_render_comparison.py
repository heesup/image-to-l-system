"""
Visual & Quantitative Comparison: Helios Ground Truth vs Part-Centric Direct Renderer.

Outputs a 3-column figure:
  1. Helios C++ Ground Truth Image
  2. Part-Centric Direct Part PyTorch Differentiable Renderer
  3. Difference map (5x amplified)

The part tensor is produced directly from XML (no 40D typed-array intermediate).
"""

import json
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

from diffusion_based.models.plant_organ_array import PlantOrganArray


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
    print(f"Running part-centric vs Helios GT comparison on {device}...")

    base_dir = "/home/lion397/codes/image-to-l-system/Digital-Crops/projects/syntheticdata_generation/build/output"
    samples = [
        {"name": "DAP 10 (Vegetative)", "xml": "dap10_gt_0000_plant_0000.xml", "gt_img": "dap10_gt_0000_vis.jpeg", "params": "dap10_gt_0000_params.json"},
        {"name": "DAP 30 (Branching)", "xml": "dap30_gt_0000_plant_0000.xml", "gt_img": "dap30_gt_0000_vis.jpeg", "params": "dap30_gt_0000_params.json"},
        {"name": "DAP 50 (Canopy)", "xml": "dap50_gt_0000_plant_0000.xml", "gt_img": "dap50_gt_0000_vis.jpeg", "params": "dap50_gt_0000_params.json"},
        {"name": "DAP 100 (Flowering & Pods)", "xml": "dap100_gt_0000_plant_0000.xml", "gt_img": "dap100_gt_0000_vis.jpeg", "params": "dap100_gt_0000_params.json"},
    ]

    from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
    renderer = HeliosPyTorchRenderer(image_size=512)

    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 4.5 * len(samples)))
    fig.patch.set_facecolor("#121212")

    metrics_summary = []
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

        # 2. Part-centric direct render (XML -> part tensor -> image)
        array = PlantOrganArray.from_xml_string(xml_str)
        part_tensor = array.to_part_tensor(device=device)
        rendered = renderer.render_part_tensor_14d(
            part_tensor,
            device=device,
            azimuth_deg=azimuth_deg,
            elevation_deg=90.0,
            camera_height=camera_height,
            focus_plant=focus_plant,
            use_kinematics_tree=False,
        )
        img_part = rendered.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)

        # 3. Difference Map (part render vs Helios GT)
        diff_gt = np.abs(img_part - gt_np)
        diff_vis = diff_gt * 5.0

        mae_gt, mse_gt, ssim_gt = compute_metrics(img_part, gt_np)
        metrics_summary.append({
            "name": sample["name"],
            "N": part_tensor.shape[0],
            "mae_vs_gt": mae_gt,
            "mse_vs_gt": mse_gt,
            "ssim_vs_gt": ssim_gt,
        })
        print(f"[{sample['name']}] N={part_tensor.shape[0]} organs: part vs Helios GT  MAE={mae_gt:.6f}, MSE={mse_gt:.6e}, SSIM={ssim_gt:.4f}")

        ax0 = axes[row_i, 0]
        ax0.imshow(gt_np)
        ax0.set_title(f"{sample['name']}\nHelios C++ Ground Truth", color="white", fontsize=11, fontweight="bold")
        ax0.axis("off")

        ax1 = axes[row_i, 1]
        ax1.imshow(img_part)
        ax1.set_title(f"Part-Centric Direct Render\n(XML -> part tensor -> image)", color="#00ff88", fontsize=11, fontweight="bold")
        ax1.axis("off")

        ax2 = axes[row_i, 2]
        ax2.imshow(diff_vis.clip(0.0, 1.0))
        ax2.set_title(f"Part vs Helios GT Diff (5x amp)\nMAE: {mae_gt:.4f} | SSIM: {ssim_gt:.4f}", color="#ffaa00", fontsize=11, fontweight="bold")
        ax2.axis("off")

    plt.tight_layout()
    out_dir = os.path.join(repo_root, "docs", "results", "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig2_14d_part_renderer_identity_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nSaved comparison figure to: {out_path}")

    import json as _json
    metrics_path = os.path.join(out_dir, "fig2_metrics.json")
    with open(metrics_path, "w") as f:
        _json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
