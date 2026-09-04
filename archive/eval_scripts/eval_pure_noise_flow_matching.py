"""
Benchmark & Visual Evaluation: Pure Gaussian Noise Plant Flow Matching with Depth Shape Comparison.
Conditioned strictly on single RGB input images with zero ground truth structure leakage.
"""

import os
import sys
import json
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.part_flow_matching import PartFlowMatchingModel
from diffusion_based.training.flow_matching import FlowMatchingScheduler, FM_OT_END, FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END, FM_SCALE_START, FM_SCALE_END, FM_CURV_IDX, FM_PHYLLO_IDX
from diffusion_based.models.part_assembly_to_xml import PartAssemblyToXMLConverter
from diffusion_based.models.plant_organ_array import PlantOrganArray, P_COL_ORGAN_TYPE, P_COL_BASE_X, P_COL_BASE_Y, P_COL_BASE_Z, P_COL_ROT_0, P_COL_ROT_5, P_COL_SCALE_X, P_COL_SCALE_Y, P_COL_SCALE_Z, P_COL_EXISTENCE, P_COL_CURVATURE, P_COL_PHYLLOTACTIC_ANGLE, NUM_FEATURES_PART
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a > 0.5, mask_b > 0.5).sum()
    union = np.logical_or(mask_a > 0.5, mask_b > 0.5).sum()
    return float(intersection / union) if union > 0 else 1.0


def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor, max_pts: int = 2048) -> float:
    if p1.shape[0] == 0 or p2.shape[0] == 0:
        return 999.0
    if p1.shape[0] > max_pts:
        idx1 = torch.randperm(p1.shape[0], device=p1.device)[:max_pts]
        p1 = p1[idx1]
    if p2.shape[0] > max_pts:
        idx2 = torch.randperm(p2.shape[0], device=p2.device)[:max_pts]
        p2 = p2[idx2]
    dist_mat = torch.cdist(p1, p2)
    min_dist_1 = dist_mat.min(dim=1).values.mean()
    min_dist_2 = dist_mat.min(dim=0).values.mean()
    return float((min_dist_1 + min_dist_2).item())


def normalize_depth_for_vis(depth_t: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    d = depth_t.detach().cpu().numpy()
    fg = d > 1e-4
    d_norm = np.zeros_like(d)
    if fg.any():
        d_min = d[fg].min()
        d_max = d[fg].max()
        if d_max > d_min:
            # Invert depth so taller/closer canopy is brighter
            d_norm[fg] = (d_max - d[fg]) / (d_max - d_min)
        else:
            d_norm[fg] = 1.0
    return d_norm, fg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Pure Gaussian Noise Flow Matching + Depth Shape Evaluation on {device}...")

    # Load 73M Cowpea DiT Model Checkpoint
    ckpt_path = os.path.join(repo_root, "diffusion_based", "checkpoints", "fm", "part_flow_matching_epoch60.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = PartFlowMatchingModel(
        max_nodes=1024,
        node_dim=26,
        image_size=128,
        patch_size=8,
        embed_dim=512,
        encoder_layers=12,
        decoder_layers=8,
        num_heads=16,
    ).to(device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    renderer = HeliosPyTorchRenderer(image_size=256).to(device)
    assembler = PartAssemblyToXMLConverter()
    scheduler = FlowMatchingScheduler()

    # Benchmark Test Plants
    exact_dir = os.path.join(repo_root, "Digital-Crops", "projects", "syntheticdata_generation", "build", "output", "exact_gt_renders")
    test_cases = [
        {"name": "DAP 010 (Juvenile)", "xml": "rad_dap010_0000_plant_0000.xml", "img": "rad_dap010_0000_rad.jpeg"},
        {"name": "DAP 050 (Canopy)", "xml": "rad_dap050_0000_plant_0000.xml", "img": "rad_dap050_0000_rad.jpeg"},
        {"name": "DAP 090 (Mature)", "xml": "rad_dap090_0000_plant_0000.xml", "img": "rad_dap090_0000_rad.jpeg"},
    ]

    # 6 Columns: Input RGB | GT 3D Mesh | GT Depth | Generated 3D Mesh | Generated Depth | Depth Shape Error Map
    fig, axes = plt.subplots(len(test_cases), 6, figsize=(24, 4.2 * len(test_cases)))
    fig.patch.set_facecolor("#0B0F19")

    results_table = []

    for row_idx, tc in enumerate(test_cases):
        xml_path = os.path.join(exact_dir, tc["xml"])
        img_path = os.path.join(exact_dir, tc["img"])

        # 1. Load Ground Truth Plant & Render GT Depth/RGB
        arr_gt = PlantOrganArray.from_xml_file(xml_path)
        mesh_gt = renderer.geo_builder.build_mesh_from_part_tensor(arr_gt.to_part_tensor(device=device), device=device)
        depth_gt = renderer.render_depth(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
        rgb_gt = renderer.render_mesh(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
        
        mask_gt = (depth_gt > 1e-4).float().cpu().numpy()
        rgb_gt_np = rgb_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        d_gt_norm, fg_gt = normalize_depth_for_vis(depth_gt)

        # 2. Load Conditioning Image
        pil_img = Image.open(img_path).convert("RGB").resize((128, 128))
        img_np = np.array(pil_img) / 255.0
        img_t = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # 3. Flow Matching Sampling strictly from Pure Gaussian Noise x_0 ~ N(0, I)
        torch.manual_seed(100 + row_idx)
        with torch.no_grad():
            x_gen = scheduler.sample(
                model=model,
                images=img_t,
                num_steps=35,
                node_dim=26,
                max_nodes=1024,
                device=device,
                x0=None # Pure Gaussian Noise (ZERO LEAKAGE)
            ).squeeze(0)

        # 4. Decode 26D -> 16D Part Tensor -> XML -> PlantOrganArray
        ot_probs = torch.softmax(x_gen[:, :FM_OT_END], dim=-1)
        ot_idx = torch.argmax(ot_probs, dim=-1)
        empty_idx = FM_OT_END - 1
        existence = 1.0 - ot_probs[:, empty_idx]

        part_16d = torch.zeros((1024, NUM_FEATURES_PART), device=device)
        part_16d[:, P_COL_ORGAN_TYPE] = ot_idx.float()
        part_16d[:, P_COL_BASE_X:P_COL_BASE_Z+1] = x_gen[:, FM_BASE_START:FM_BASE_END] / 20.0
        part_16d[:, P_COL_ROT_0:P_COL_ROT_5+1] = x_gen[:, FM_ROT_START:FM_ROT_END]
        part_16d[:, P_COL_SCALE_X:P_COL_SCALE_Z+1] = x_gen[:, FM_SCALE_START:FM_SCALE_END] / 50.0
        part_16d[:, P_COL_EXISTENCE] = existence
        part_16d[:, P_COL_CURVATURE] = x_gen[:, FM_CURV_IDX] * 100.0
        part_16d[:, P_COL_PHYLLOTACTIC_ANGLE] = x_gen[:, FM_PHYLLO_IDX] * 180.0

        xml_str = assembler.convert_to_xml_string(part_16d, plant_id=0, existence_threshold=0.5)
        arr_gen = PlantOrganArray.from_xml_string(xml_str)
        mesh_gen = renderer.geo_builder.build_mesh_from_part_tensor(arr_gen.to_part_tensor(device=device), device=device)
        
        # 5. Render Generated 3D Mesh & Depth
        depth_gen = renderer.render_depth(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
        rgb_gen = renderer.render_mesh(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
        
        mask_gen = (depth_gen > 1e-4).float().cpu().numpy()
        rgb_gen_np = rgb_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        d_gen_norm, fg_gen = normalize_depth_for_vis(depth_gen)

        # 6. Compute Depth Shape Error Map & Metrics
        fg_union = fg_gt | fg_gen
        depth_err_map = np.zeros_like(d_gt_norm)
        depth_err_map[fg_union] = np.abs(d_gt_norm[fg_union] - d_gen_norm[fg_union])
        depth_mae = float(depth_err_map[fg_union].mean()) if fg_union.any() else 0.0

        iou = compute_iou(mask_gt, mask_gen)
        chamfer = chamfer_distance(mesh_gen["vertices"], mesh_gt["vertices"])
        vert_count = mesh_gen["vertices"].shape[0]

        results_table.append({
            "stage": tc["name"],
            "mask_iou": iou,
            "depth_mae": depth_mae,
            "chamfer_dist": chamfer,
            "vertices": vert_count
        })

        # 7. Render 6-column panel
        # Col 0: Input Image Condition
        ax0 = axes[row_idx, 0]
        ax0.imshow(img_np)
        ax0.set_title(f"{tc['name']}\nInput RGB Condition", color="white", fontsize=11, fontweight="bold")
        ax0.axis("off")

        # Col 1: Ground Truth 3D Mesh
        ax1 = axes[row_idx, 1]
        ax1.imshow(rgb_gt_np)
        ax1.set_title(f"Ground Truth 3D Mesh\n(Verts: {mesh_gt['vertices'].shape[0]})", color="#38BDF8", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # Col 2: Ground Truth Depth Field (Shape)
        ax2 = axes[row_idx, 2]
        ax2.imshow(d_gt_norm, cmap="plasma")
        ax2.set_title("GT 3D Depth Shape\n(Canopy Height Map)", color="#38BDF8", fontsize=11, fontweight="bold")
        ax2.axis("off")

        # Col 3: Pure Noise Flow Matching 3D Mesh
        ax3 = axes[row_idx, 3]
        ax3.imshow(rgb_gen_np)
        ax3.set_title(f"Generated 3D Mesh (x0 ~ N(0,I))\nMask IoU: {iou:.3f} | Chamfer: {chamfer:.4f}", color="#34D399", fontsize=11, fontweight="bold")
        ax3.axis("off")

        # Col 4: Generated Depth Field (Shape)
        ax4 = axes[row_idx, 4]
        ax4.imshow(d_gen_norm, cmap="plasma")
        ax4.set_title("Generated 3D Depth Shape\n(Predicted Height Map)", color="#34D399", fontsize=11, fontweight="bold")
        ax4.axis("off")

        # Col 5: Depth Shape Error Heatmap
        ax5 = axes[row_idx, 5]
        im5 = ax5.imshow(depth_err_map, cmap="inferno", vmin=0.0, vmax=1.0)
        ax5.set_title(f"Depth Shape Error (|GT-Gen|)\nDepth MAE: {depth_mae:.3f}", color="#F43F5E", fontsize=11, fontweight="bold")
        ax5.axis("off")

    plt.tight_layout()
    out_png = os.path.join(repo_root, "docs", "results", "assets", "fig_pure_noise_flow_matching.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print("\n==========================================================================")
    print("PURE GAUSSIAN NOISE FLOW MATCHING + DEPTH SHAPE BENCHMARK RESULTS")
    print("==========================================================================")
    for r in results_table:
        print(f"{r['stage']:<20} | Mask IoU: {r['mask_iou']:.4f} | Depth MAE: {r['depth_mae']:.4f} | Chamfer: {r['chamfer_dist']:.4f} | Vertices: {r['vertices']}")
    print(f"Saved benchmark figure to: {out_png}")


if __name__ == "__main__":
    main()
