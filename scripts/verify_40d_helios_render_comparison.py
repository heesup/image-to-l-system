"""
Benchmark & Rendering Equivalence Verification for Typed (N, 40) PlantOrganArray.

Compares:
  1. Helios C++ Ground Truth Image (Raytraced)
  2. PyTorch Differentiable Render from 40D PlantOrganArray
  3. PyTorch Differentiable Depth Map
  4. PyTorch Differentiable Leaf/Organ Masks
  5. Re-rendered XML through Helios C++ Raytracer
across Cowpea, Bean, and Sorghum growth stages.
"""

import os
import sys
import glob
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import PlantOrganArray, NUM_FEATURES_TYPED
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.eval.metrics import masked_ssim, foreground_iou


def _depth_colormap(depth_np: np.ndarray) -> np.ndarray:
    valid = depth_np[depth_np > 0]
    if len(valid) == 0:
        return np.zeros((*depth_np.shape, 3), dtype=np.float32)
    d_min, d_max = float(valid.min()), float(valid.max())
    norm = np.clip((depth_np - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
    norm[depth_np <= 0] = 0.0
    cmap = plt.get_cmap("plasma")
    colored = cmap(norm)[:, :, :3]
    colored[depth_np <= 0] = 0.0
    return colored.astype(np.float32)


def render_xml_with_helios_cpp(xml_string: str, species: str = "cowpea") -> np.ndarray:
    """Renders XML using Helios C++ standalone raytracer binary."""
    import tempfile
    img_out = np.zeros((256, 256, 3), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="helios_eval_40d_") as tmp_dir:
        xml_path = os.path.join(tmp_dir, "temp_plant.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_string)

        build_dir = os.path.join(REPO_ROOT, "Digital-Crops/projects/syntheticdata_generation/build")
        cfg_file = os.path.join(REPO_ROOT, f"Digital-Crops/projects/syntheticdata_generation/configs/params_{species}.json")
        cmd = [
            "./main",
            "--renderer", "radiation",
            "--input-xml", xml_path,
            "--output", tmp_dir,
            "-n", "flow_render",
            "--focus-plant",
            "-f", os.path.abspath(cfg_file),
        ]
        try:
            subprocess.run(cmd, cwd=build_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            cand = os.path.join(tmp_dir, species, "flow_render_0000_rad.jpeg")
            if not os.path.exists(cand):
                cand = os.path.join(tmp_dir, species, "flow_render_0000_vis.jpeg")
            if os.path.exists(cand):
                with Image.open(cand) as img:
                    img_out = np.array(img.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Helios C++ binary render fallback: {e}")

    return img_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running 40D PlantOrganArray vs Helios Rendering Benchmark on {device}...")

    renderer = HeliosPyTorchRenderer(image_size=256).to(device)

    # Benchmark samples across species and growth stages
    test_cases = [
        ("Cowpea DAP 10 (Seedling)", "cowpea", "dataset/helios_data/cowpea/cowpea_dap010_seed03_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "dataset/helios_data/cowpea/cowpea_dap010_seed03_caz000_h1.0_se045_saz180_0000_rad.jpeg"),
        ("Cowpea DAP 50 (Canopy)", "cowpea", "dataset/helios_data/cowpea/cowpea_dap050_seed12_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "dataset/helios_data/cowpea/cowpea_dap050_seed12_caz000_h1.0_se045_saz180_0000_rad.jpeg"),
        ("Bean DAP 30 (Vegetative)", "bean", "dataset/helios_data/bean/bean_dap030_seed02_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "dataset/helios_data/bean/bean_dap030_seed02_caz000_h1.0_se045_saz180_0000_rad.jpeg"),
        ("Sorghum DAP 20 (Early)", "sorghum", "dataset/helios_data/sorghum/sorghum_dap020_seed05_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "dataset/helios_data/sorghum/sorghum_dap020_seed05_caz000_h1.0_se045_saz180_0000_rad.jpeg"),
        ("Sorghum DAP 60 (Tillering)", "sorghum", "dataset/helios_data/sorghum/sorghum_dap060_seed08_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "dataset/helios_data/sorghum/sorghum_dap060_seed08_caz000_h1.0_se045_saz180_0000_rad.jpeg"),
    ]

    fig, axes = plt.subplots(len(test_cases), 6, figsize=(18, 3.2 * len(test_cases)))

    for row, (title, species, xml_rel, orig_rel) in enumerate(test_cases):
        xml_path = os.path.join(REPO_ROOT, xml_rel)
        orig_path = os.path.join(REPO_ROOT, orig_rel)
        print(f"\n[{row+1}/{len(test_cases)}] Evaluating {title}...")

        # 1. Parse into 40D PlantOrganArray
        arr = PlantOrganArray.from_xml_file(xml_path)
        print(f"  Parsed 40D Organ Array: shape={arr.tensor.shape} (is_typed={arr.is_typed})")

        # 2. Build 3D Mesh using Forward Kinematics Tree
        mesh = renderer.geo_builder.build_mesh_from_organ_array(arr, device=device)
        verts = mesh["vertices"]
        print(f"  Forward Kinematics Mesh: {verts.shape[0]} vertices, {mesh['faces'].shape[0]} faces")
        cam_bounds = {"min": verts.min(dim=0)[0].tolist(), "max": verts.max(dim=0)[0].tolist()}

        # 3. Multi-modal PyTorch Differentiable Render
        rendered = renderer.render_mesh(
            mesh,
            azimuth_deg=0.0,
            elevation_deg=45.0,
            camera_height=1.0,
            background="ground",
            differentiable=False,
            focus_plant=True,
            image_size=256,
        )
        rgb_pt_np = rendered.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)

        # Render Depth
        depth_t = renderer.render_depth(
            mesh,
            azimuth_deg=0.0,
            elevation_deg=45.0,
            camera_height=1.0,
            focus_plant=True,
            image_size=256,
        )
        depth_pt_np = depth_t.detach().cpu().numpy()

        # Render Organ Semantic Mask
        type_buf = renderer.render_organ_type_buffer(
            mesh,
            azimuth_deg=0.0,
            elevation_deg=45.0,
            camera_height=1.0,
            focus_plant=True,
            image_size=256,
        )
        type_np = type_buf.detach().cpu().numpy()

        # 4. Load Helios GT image
        if os.path.exists(orig_path):
            with Image.open(orig_path) as img:
                gt_rgb_np = np.array(img.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
        else:
            gt_rgb_np = np.zeros((256, 256, 3), dtype=np.float32)

        # 5. Round-trip: 40D Array -> XML String -> Helios C++ Raytrace
        xml_roundtrip = arr.to_xml_string()
        helios_rerender_np = render_xml_with_helios_cpp(xml_roundtrip, species=species)

        # Compute Metrics
        ssim_val = masked_ssim(torch.from_numpy(rgb_pt_np).permute(2,0,1).to(device), torch.from_numpy(gt_rgb_np).permute(2,0,1).to(device))
        iou_val = foreground_iou(torch.from_numpy(rgb_pt_np).permute(2,0,1).to(device), torch.from_numpy(gt_rgb_np).permute(2,0,1).to(device))
        print(f"  Rendering Fidelity: mSSIM = {ssim_val:.4f} | Mask IoU = {iou_val:.4f}")

        # Plot 6 Columns
        axes[row, 0].imshow(gt_rgb_np)
        axes[row, 0].set_title(f"{title}\nHelios C++ GT Raytrace", fontsize=9, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(rgb_pt_np)
        axes[row, 1].set_title(f"PyTorch 40D Differentiable Render\nmSSIM: {ssim_val:.3f} | IoU: {iou_val:.2f}", fontsize=9, color="darkgreen", fontweight="bold")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(_depth_colormap(depth_pt_np))
        axes[row, 2].set_title("PyTorch 40D Depth Map\n(Ray-aligned Kinematics)", fontsize=9, color="purple", fontweight="bold")
        axes[row, 2].axis("off")

        cmap_types = plt.get_cmap("tab10")
        axes[row, 3].imshow(type_np, cmap="tab10", vmin=0, vmax=10)
        axes[row, 3].set_title("Organ Segmentation Mask\n(Leaf/Petiole/Internode)", fontsize=9, color="navy", fontweight="bold")
        axes[row, 3].axis("off")

        axes[row, 4].imshow(helios_rerender_np)
        axes[row, 4].set_title("Re-rendered from 40D XML\n(Helios C++ Engine)", fontsize=9, color="darkred", fontweight="bold")
        axes[row, 4].axis("off")

        # Column 5: Difference Heatmap
        diff_map = np.abs(rgb_pt_np - gt_rgb_np).mean(axis=-1)
        axes[row, 5].imshow(diff_map, cmap="inferno", vmin=0.0, vmax=0.5)
        axes[row, 5].set_title(f"Absolute Difference\nMean L1: {diff_map.mean():.4f}", fontsize=9, color="brown", fontweight="bold")
        axes[row, 5].axis("off")

    plt.tight_layout()
    assets_dir = os.path.join(REPO_ROOT, "docs/results/assets")
    os.makedirs(assets_dir, exist_ok=True)
    out_path = os.path.join(assets_dir, "fig_40d_helios_render_comparison.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\nSuccessfully generated 40D Helios Rendering Comparison figure to:\n  {out_path}")


if __name__ == "__main__":
    main()
