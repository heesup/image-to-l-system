"""
Evaluation & High-Resolution Benchmark Suite for VLM-Scaffold-DiT on 100K Drone Orthophoto Dataset.
Renders and saves 6-column multi-modal visual reconstruction panels locally to docs/results/assets/.
"""

import os
import sys
import glob
import json
import math
import argparse
from typing import List, Dict, Any, Optional

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from diffusion_based.models.plant_organ_array import PlantOrganArray, NUM_FEATURES_PART
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.vlm_scaffold_dit import VLMScaffoldDiTModel
from diffusion_based.models.part_assembly_to_xml import PartAssemblyToXMLConverter
from diffusion_based.dataset.part_array_dataset import (
    ORGAN_CATEGORIES, EMPTY_IDX, FM_NODE_DIM, FM_OT_END,
    FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END,
    FM_SCALE_START, FM_SCALE_END, FM_CURV_IDX, FM_PHYLLO_IDX,
    BASE_SCALE, SCALE_SCALE, CURVATURE_SCALE, PHYLLOTACTIC_SCALE
)

ORGAN_LEGEND_MAP = [
    ("Stem", "#8B4513"),
    ("Petiole", "#E68026"),
    ("Leaf", "#228B22"),
    ("Peduncle", "#9ACD32"),
    ("Flower", "#FFD700"),
    ("Pod/Fruit", "#E63333"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM-Scaffold-DiT Model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="Classifier-Free Guidance scale (s >= 1.0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-fig", type=str, default=None, help="Path to save evaluation figure")
    return parser.parse_args()


def depth_to_chm_rgb(depth_tensor: torch.Tensor, far_plane: float = 20.0):
    """Convert depth buffer (distance to camera) to Canopy Height Model (CHM) colormap.
    Taller plant parts are closer to camera (smaller depth) -> mapped to brighter yellow/orange in plasma.
    Ground/empty space -> pure pitch black background (0, 0, 0).
    """
    d = depth_tensor.detach().cpu().numpy().squeeze()
    fg_mask = (d < (far_plane - 0.5)) & (d > 0.01)
    if not np.any(fg_mask):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.float32), 0.0

    d_min = d[fg_mask].min()
    d_max = d[fg_mask].max()
    canopy_h_cm = (d_max - d_min) * 100.0  # height in cm

    d_norm = np.zeros_like(d)
    if d_max > d_min:
        d_norm[fg_mask] = (d_max - d[fg_mask]) / (d_max - d_min)  # invert: closer (taller) = 1.0 (brighter)
    else:
        d_norm[fg_mask] = 1.0

    cmap = plt.get_cmap("plasma")
    rgb = cmap(d_norm)[:, :, :3].astype(np.float32)
    rgb[~fg_mask] = 0.0  # pure black background
    return rgb, canopy_h_cm


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Discover checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt_path = args.checkpoint
    else:
        candidates = [
            os.path.join(repo_root, "diffusion_based", "checkpoints", "fm", "cowpea_vlm_scaffold_dit_h100_ddp.pt"),
            os.path.join(repo_root, "diffusion_based", "checkpoints", "fm", "cowpea_dit_large_2xh100_ddp.pt"),
            os.path.join(repo_root, "diffusion_based", "checkpoints", "fm", "cowpea_dit_large_150m.pt"),
        ]
        ckpt_path = next((c for c in candidates if os.path.exists(c)), None)
        if ckpt_path is None:
            raise FileNotFoundError("No valid model checkpoint found in diffusion_based/checkpoints/fm/")

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Detect model architecture
    model = VLMScaffoldDiTModel(
        dinov3_model="dinov3_vitb14",
        max_slots=4096,
        node_dim=26,
        embed_dim=768,
        decoder_layers=12,
        num_heads=12,
        freeze_vision_backbone=False,
    ).to(device)

    # Clean potential DDP module prefix
    cleaned_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_sd[k[7:]] = v
        else:
            cleaned_sd[k] = v

    model.load_state_dict(cleaned_sd, strict=False)
    model.eval()

    renderer = HeliosPyTorchRenderer(image_size=512).to(device)
    assembler = PartAssemblyToXMLConverter()

    data_dir = os.path.join(repo_root, "dataset", "helios_data", "cowpea")
    test_daps = [10, 30, 50, 70, 90]
    eval_samples = []

    for d in test_daps:
        pattern = os.path.join(data_dir, f"*_dap{d:03d}_*_plant_0000.xml")
        matches = sorted(glob.glob(pattern))
        if not matches:
            pattern2 = os.path.join(data_dir, f"*dap{d}*_plant_0000.xml")
            matches = sorted(glob.glob(pattern2))
        if matches:
            x_file = matches[0]
            prefix = x_file.split("_plant_0000.xml")[0]
            eval_samples.append({"name": f"Cowpea DAP {d:03d}", "dap": float(d), "xml": x_file, "cam": f"{prefix}_cam.json"})

    if not eval_samples:
        all_xmls = sorted(glob.glob(os.path.join(data_dir, "*_plant_0000.xml")))[:5]
        for x in all_xmls:
            prefix = x.split("_plant_0000.xml")[0]
            eval_samples.append({"name": os.path.basename(prefix), "dap": 30.0, "xml": x, "cam": f"{prefix}_cam.json"})

    print(f"Evaluating {len(eval_samples)} benchmark stages...")

    fig, axes = plt.subplots(len(eval_samples), 6, figsize=(24, 4.2 * len(eval_samples)))
    if len(eval_samples) == 1:
        axes = np.expand_dims(axes, 0)
    fig.patch.set_facecolor("#080C14")

    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    patches = [mpatches.Patch(color=c, label=name) for name, c in ORGAN_LEGEND_MAP]

    for row_idx, sc in enumerate(eval_samples):
        # 1. Ground Truth Renderings
        arr_gt = PlantOrganArray.from_xml_file(sc["xml"])
        mesh_gt = renderer.geo_builder.build_mesh_from_organ_array(arr_gt, device=device)
        rgb_gt = renderer.render_mesh(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
        depth_gt = renderer.render_depth(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
        seg_gt = renderer.render_organ_segmentation(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)

        rgb_gt_np = rgb_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        depth_gt_rgb, h_gt_cm = depth_to_chm_rgb(depth_gt)
        seg_gt_np = seg_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        # Input image for model
        img_pil = transforms.ToPILImage()(rgb_gt.clamp(0, 1).cpu())
        img_t = img_transform(img_pil).unsqueeze(0).to(device)

        # 2. Model Inference via Bridge Flow Matching
        with torch.no_grad():
            sample_res = model.sample_plant(
                img=img_t,
                num_steps=20,
                guidance_scale=args.guidance_scale,
                device=device
            )
            x_gen = sample_res["x_gen"][0]
            pred_dap = sample_res["pred_dap"][0].item()
            pred_h = sample_res["pred_height"][0].item()

        # Decode 26D -> XML -> 3D Mesh
        ot_probs = torch.softmax(x_gen[:, :FM_OT_END], dim=-1)
        ot_idx = torch.argmax(ot_probs, dim=-1)
        raw_ot = torch.tensor([ORGAN_CATEGORIES[min(i.item(), len(ORGAN_CATEGORIES)-1)] for i in ot_idx], device=device).float()

        part_16d = torch.zeros((x_gen.shape[0], NUM_FEATURES_PART), device=device)
        part_16d[:, 0] = raw_ot
        part_16d[:, 1:4] = x_gen[:, FM_BASE_START:FM_BASE_END] / BASE_SCALE
        part_16d[:, 4:10] = x_gen[:, FM_ROT_START:FM_ROT_END]
        part_16d[:, 10:13] = x_gen[:, FM_SCALE_START:FM_SCALE_END] / SCALE_SCALE
        part_16d[:, 13] = 1.0 - ot_probs[:, EMPTY_IDX]
        part_16d[:, 14] = x_gen[:, FM_CURV_IDX] * CURVATURE_SCALE
        part_16d[:, 15] = x_gen[:, FM_PHYLLO_IDX] * PHYLLOTACTIC_SCALE

        xml_str = assembler.convert_to_xml_string(part_16d, plant_id=0, existence_threshold=0.30)
        arr_gen = PlantOrganArray.from_xml_string(xml_str)
        mesh_gen = renderer.geo_builder.build_mesh_from_organ_array(arr_gen, device=device)

        rgb_gen = renderer(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
        depth_gen = renderer.render_depth(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
        seg_gen = renderer.render_organ_segmentation(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)

        rgb_gen_np = rgb_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        depth_gen_rgb, h_gen_cm = depth_to_chm_rgb(depth_gen)
        seg_gen_np = seg_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        vert_count = mesh_gen['vertices'].shape[0]
        active_n = (part_16d[:, 13] > 0.30).sum().item()

        # Plot 6 Columns
        # Col 1: GT RGB
        n_gt_organs = arr_gt.to_part_tensor().shape[0]
        axes[row_idx, 0].imshow(rgb_gt_np)
        axes[row_idx, 0].set_title(f"{sc['name']}\nDiff RGB Input ({n_gt_organs} organs)", color="#34D399", fontsize=10, fontweight="bold")
        axes[row_idx, 0].text(0.03, 0.03, f"N={n_gt_organs}", transform=axes[row_idx, 0].transAxes, fontsize=8, color='white', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        axes[row_idx, 0].axis("off")

        # Col 2: GT Canopy Height (CHM)
        depth_gt_rgb, h_gt_cm = depth_to_chm_rgb(depth_gt)
        axes[row_idx, 1].imshow(depth_gt_rgb)
        axes[row_idx, 1].set_title("Canopy Height (CHM)\n(taller = brighter)", color="#38BDF8", fontsize=10, fontweight="bold")
        axes[row_idx, 1].text(0.03, 0.03, f"Height: 0–{h_gt_cm:.1f} cm", transform=axes[row_idx, 1].transAxes, fontsize=8, color='#38BDF8', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        axes[row_idx, 1].axis("off")

        # Col 3: GT Organ Segmentation
        axes[row_idx, 2].imshow(seg_gt_np)
        axes[row_idx, 2].set_title("Diff Organ Seg Input", color="#A78BFA", fontsize=10, fontweight="bold")
        axes[row_idx, 2].legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.85, facecolor='#0D1117', labelcolor='white', edgecolor='#334155', ncol=1)
        axes[row_idx, 2].axis("off")

        # Col 4: Gen RGB
        axes[row_idx, 3].imshow(rgb_gen_np)
        axes[row_idx, 3].set_title(f"Gen Diff RGB\n(DAP: {pred_dap:.1f} | H: {pred_h*100:.1f}cm)", color="#60A5FA", fontsize=10, fontweight="bold")
        axes[row_idx, 3].text(0.03, 0.03, f"N={active_n} ({vert_count}v)", transform=axes[row_idx, 3].transAxes, fontsize=8, color='white', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        axes[row_idx, 3].axis("off")

        # Col 5: Gen Canopy Height (CHM)
        axes[row_idx, 4].imshow(depth_gen_rgb)
        axes[row_idx, 4].set_title("Gen Canopy Height\n(taller = brighter)", color="#F472B6", fontsize=10, fontweight="bold")
        axes[row_idx, 4].text(0.03, 0.03, f"Height: 0–{h_gen_cm:.1f} cm", transform=axes[row_idx, 4].transAxes, fontsize=8, color='#F472B6', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        axes[row_idx, 4].axis("off")

        # Col 6: Gen Organ Segmentation
        axes[row_idx, 5].imshow(seg_gen_np)
        axes[row_idx, 5].set_title("Gen Diff Organ Seg", color="#FB923C", fontsize=10, fontweight="bold")
        axes[row_idx, 5].legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.85, facecolor='#0D1117', labelcolor='white', edgecolor='#334155', ncol=1)
        axes[row_idx, 5].axis("off")

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.2)

    plt.tight_layout()
    save_fig_path = args.output_fig if args.output_fig else os.path.join(repo_root, "docs", "results", "assets", "fig_cowpea_100k_lifespan_benchmark.png")
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"✓ Saved updated 6-column evaluation figure to: {save_fig_path}")


if __name__ == "__main__":
    main()
