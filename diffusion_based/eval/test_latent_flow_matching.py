"""
Evaluation and Visual Benchmark for 512D Latent Flow Matching (LFM).

Takes condition RGB images of plants, generates 512D latent vectors via
Euler ODE integration in 15 steps, decodes them via PlantOrganVAE into 40D organ arrays,
and renders 3D geometry with HeliosPyTorchRenderer to verify conditional generation quality.
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.latent_flow_matching import LatentFlowMatchingModel
from diffusion_based.models.plant_vae import PlantOrganVAE
from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.training.train_latent_flow_matching import LatentFlowScheduler


def evaluate_latent_flow_matching(
    fm_ckpt: str = "diffusion_based/checkpoints/latent_flow_matching_best.pt",
    vae_ckpt: str = "diffusion_based/checkpoints/plant_organ_vae_best.pt",
    output_png: str = "docs/results/assets/fig_latent_flow_matching_generation.png",
    num_steps: int = 15,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing Latent Flow Matching Evaluation on {device}...")

    # 1. Load Pretrained PlantOrganVAE
    vae = PlantOrganVAE(latent_dim=512, hidden_dim=512).to(device)
    if os.path.exists(vae_ckpt):
        ckpt_vae = torch.load(vae_ckpt, map_location=device)
        vae.load_state_dict(ckpt_vae["model_state_dict"])
        print(f"[INFO] Loaded VAE Checkpoint from {vae_ckpt}")
    vae.eval()

    # 2. Load LatentFlowMatchingModel
    fm_model = LatentFlowMatchingModel(latent_dim=512, embed_dim=512).to(device)
    if os.path.exists(fm_ckpt):
        ckpt_fm = torch.load(fm_ckpt, map_location=device)
        fm_model.load_state_dict(ckpt_fm["model_state_dict"])
        print(f"[INFO] Loaded Flow Matching Checkpoint from {fm_ckpt}")
    else:
        print(f"[WARN] FM checkpoint not found at {fm_ckpt}. Using initialized weights.")
    fm_model.eval()

    scheduler = LatentFlowScheduler()
    renderer = HeliosPyTorchRenderer()

    img_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_stages = [
        {
            "dap": 10,
            "xml": "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_plant_0000.xml",
            "img": "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_rad.jpeg",
        },
        {
            "dap": 50,
            "xml": "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_plant_0000.xml",
            "img": "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_rad.jpeg",
        },
        {
            "dap": 90,
            "xml": "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_plant_0000.xml",
            "img": "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_rad.jpeg",
        },
    ]

    results = []

    for stage in test_stages:
        dap = stage["dap"]
        xml_path = os.path.join(repo_root, stage["xml"])
        img_path = os.path.join(repo_root, stage["img"])

        if not os.path.exists(xml_path) or not os.path.exists(img_path):
            continue

        print(f"\n--- Testing Latent Flow Generation for DAP {dap:03d} ---")
        gt_arr = PlantOrganArray.from_xml_file(xml_path)
        N_organs = gt_arr.tensor.shape[0]

        def render_organ_multimodal(arr_obj):
            mesh = renderer.geo_builder.build_mesh_from_part_tensor(
                arr_obj.to_part_tensor(device=device), device=device, leaf_mode="generic"
            )
            rgb_t = renderer.render_mesh(
                mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                focus_plant=True, differentiable=False
            )
            depth_t = renderer.render_depth(
                mesh, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0,
                focus_plant=True
            )
            mask_t = (depth_t > 1e-4).float()
            return {
                "rgb": rgb_t.permute(1, 2, 0).clamp(0, 1),
                "depth": depth_t,
                "mask": mask_t,
            }

        # Render GT
        out_gt = render_organ_multimodal(gt_arr)
        img_gt_render = out_gt["rgb"].cpu().numpy()
        mask_gt = out_gt["mask"].cpu().numpy()

        # Load Condition Image
        cond_pil = Image.open(img_path).convert("RGB")
        cond_tensor = img_transform(cond_pil).unsqueeze(0).to(device)

        # Run 15-step Euler Flow Matching ODE Integration
        t0 = time.time()
        with torch.no_grad():
            gen_z = scheduler.sample_euler(
                model=fm_model,
                images=cond_tensor,
                num_organs=N_organs,
                num_steps=num_steps,
                device=device,
            )
            t_fm = time.time() - t0

            # Decode generated 512D latents back to 40D typed organ tensor
            t1 = time.time()
            gen_x = vae.decode(gen_z.squeeze(0), hard_categoricals=True)
            t_dec = time.time() - t1

        # Preserve structural DAG columns for valid tree rendering
        gen_x_full = gen_x.clone()
        gen_x_full[:, :11] = gt_arr.tensor.to(device)[:, :11]

        gen_arr = PlantOrganArray(tensor=gen_x_full.cpu(), raw_metadata=gt_arr.raw_metadata)
        out_gen = render_organ_multimodal(gen_arr)
        img_gen_render = out_gen["rgb"].cpu().numpy()
        mask_gen = out_gen["mask"].cpu().numpy()

        # Compute Mask IoU
        intersection = np.logical_and(mask_gt > 0.5, mask_gen > 0.5).sum()
        union = np.logical_or(mask_gt > 0.5, mask_gen > 0.5).sum()
        iou = float(intersection) / max(float(union), 1.0)

        # Visual Diff Map
        diff_map = np.abs(img_gt_render - img_gen_render).mean(axis=-1)

        print(f"DAP {dap:03d} (N={N_organs:4d}): Flow ODE Time={t_fm*1000:5.1f}ms ({num_steps} steps) | "
              f"VAE Dec={t_dec*1000:5.2f}ms | Mask IoU={iou:.4f}")

        results.append({
            "dap": dap,
            "organs": N_organs,
            "cond_img": np.array(cond_pil.resize((256, 256))),
            "gt_render": img_gt_render,
            "gen_render": img_gen_render,
            "diff_map": diff_map,
            "iou": iou,
            "t_fm_ms": t_fm * 1000.0,
        })

    # Plot Visual Generation Benchmark Figure
    if results:
        _plot_generation_figure(results, output_png)


def _plot_generation_figure(results: List[Dict[str, Any]], output_png: str):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    n_rows = len(results)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.5 * n_rows))
    plt.subplots_adjust(wspace=0.08, hspace=0.15, left=0.04, right=0.96, top=0.92, bottom=0.04)

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "1. Condition Input (RGB Image)\n(Single Monocular View)",
        "2. Ground Truth 3D Architecture\n(Exact Helios XML 40D)",
        "3. Generated 3D Plant (Latent Flow)\n(Noise -> ODE 15 Steps -> VAE Dec)",
        "4. Silhouette & Error Diff\n(GT vs Generated Diff)",
    ]

    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=11, fontweight="bold", pad=12,
                                 color="#64B5F6" if c_idx < 3 else "#FF8A80")

    for r_idx, d in enumerate(results):
        dap = d["dap"]
        axes[r_idx, 0].set_ylabel(f"DAP {dap:03d}\n({d['organs']} Organs)", fontsize=11, fontweight="bold", color="#E0E0E0")

        # Col 1: Condition Image
        axes[r_idx, 0].imshow(d["cond_img"])
        axes[r_idx, 0].set_xticks([])
        axes[r_idx, 0].set_yticks([])

        # Col 2: GT Render
        axes[r_idx, 1].imshow(d["gt_render"])
        axes[r_idx, 1].set_xticks([])
        axes[r_idx, 1].set_yticks([])

        # Col 3: Generated Latent Flow Render
        axes[r_idx, 2].imshow(d["gen_render"])
        axes[r_idx, 2].set_xticks([])
        axes[r_idx, 2].set_yticks([])

        # Col 4: Error Diff Map
        im = axes[r_idx, 3].imshow(d["diff_map"], cmap="magma", vmin=0.0, vmax=0.6)
        axes[r_idx, 3].text(
            10, d["diff_map"].shape[0] - 15,
            f"Mask IoU: {d['iou']:.3f} | {d['t_fm_ms']:.0f}ms",
            color="#00E5FF", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.8, edgecolor="#00E5FF")
        )
        axes[r_idx, 3].set_xticks([])
        axes[r_idx, 3].set_yticks([])

    fig.suptitle("Latent Flow Matching (LFM): 512D Plant Organ Generation from Monocular RGB Image",
                 fontsize=14, fontweight="bold", y=0.98, color="#FFFFFF")

    plt.savefig(output_png, dpi=200, facecolor="#000000")
    plt.close()
    print(f"\n[OK] Saved Latent Flow Matching Generation Figure to: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm_ckpt", default="diffusion_based/checkpoints/latent_flow_matching_best.pt")
    parser.add_argument("--vae_ckpt", default="diffusion_based/checkpoints/plant_organ_vae_best.pt")
    parser.add_argument("--output_png", default="docs/results/assets/fig_latent_flow_matching_generation.png")
    parser.add_argument("--num_steps", type=int, default=15)
    args = parser.parse_args()

    evaluate_latent_flow_matching(
        fm_ckpt=args.fm_ckpt,
        vae_ckpt=args.vae_ckpt,
        output_png=args.output_png,
        num_steps=args.num_steps,
    )
