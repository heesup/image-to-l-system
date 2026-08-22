"""
Evaluation & Benchmark: Canonical Botanical Slot-Ordered Cowpea DiT (DAP 010 - DAP 030).

Evaluates:
  1. Pure Gaussian Noise Flow Matching (Zero Ground Truth Leakage)
  2. Canonical Botanical Slot Ordering (Phytomer-level slot consistency)
  3. Dynamic variable-length sampling with predicted organ count
  4. High-resolution 6-column 3D Depth shape comparison
"""

import os
import sys
import time
from typing import Dict, List, Tuple
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
    P_COL_ORGAN_TYPE, P_COL_BASE_X, P_COL_BASE_Y, P_COL_BASE_Z,
    P_COL_ROT_0, P_COL_ROT_5, P_COL_SCALE_X, P_COL_SCALE_Y, P_COL_SCALE_Z,
    P_COL_EXISTENCE, P_COL_CURVATURE, P_COL_PHYLLOTACTIC_ANGLE,
    NUM_FEATURES_PART
)
from diffusion_based.models.canonical_cowpea_dit import CanonicalCowpeaDiTModel
from diffusion_based.models.part_assembly_to_xml import PartAssemblyToXMLConverter
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.dataset.part_array_dataset import (
    FM_OT_END, FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END,
    FM_SCALE_START, FM_SCALE_END, FM_CURV_IDX, FM_PHYLLO_IDX
)


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
            d_norm[fg] = (d_max - d[fg]) / (d_max - d_min)
        else:
            d_norm[fg] = 1.0
    return d_norm, fg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Canonical Cowpea DiT Model on {device}...")

    ckpt_path = os.path.join(repo_root, "diffusion_based", "checkpoints", "fm", "canonical_cowpea_dit_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = CanonicalCowpeaDiTModel(
        max_slots=512,
        node_dim=26,
        image_size=128,
        patch_size=8,
        embed_dim=384,
        encoder_layers=8,
        decoder_layers=6,
        num_heads=12,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    renderer = HeliosPyTorchRenderer(image_size=256).to(device)
    assembler = PartAssemblyToXMLConverter()

    # Benchmark Test Plants: DAP 010, DAP 020, DAP 030
    exact_dir = os.path.join(repo_root, "dataset", "helios_data", "cowpea")
    test_cases = [
        {"name": "Cowpea DAP 010 (Juvenile)", "xml": "cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "img": "cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000_rad.jpeg", "dap": 10.0, "slots": 50},
        {"name": "Cowpea DAP 020 (Vegetative)", "xml": "cowpea_dap020_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "img": "cowpea_dap020_seed00_caz000_h1.0_se045_saz180_0000_rad.jpeg", "dap": 20.0, "slots": 151},
        {"name": "Cowpea DAP 030 (Canopy Branching)", "xml": "cowpea_dap030_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "img": "cowpea_dap030_seed00_caz000_h1.0_se045_saz180_0000_rad.jpeg", "dap": 30.0, "slots": 411},
    ]

    fig, axes = plt.subplots(len(test_cases), 6, figsize=(24, 4.2 * len(test_cases)))
    fig.patch.set_facecolor("#080C14")

    results_table = []

    for row_idx, tc in enumerate(test_cases):
        xml_path = os.path.join(exact_dir, tc["xml"])
        img_path = os.path.join(exact_dir, tc["img"])

        # 1. Ground Truth 3D Render
        arr_gt = PlantOrganArray.from_xml_file(xml_path)
        mesh_gt = renderer.geo_builder.build_mesh_from_organ_array(arr_gt, device=device)
        rgb_gt = renderer.render_mesh(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
        depth_gt = renderer.render_depth(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)

        mask_gt = (depth_gt > 1e-4).float().cpu().numpy()
        rgb_gt_np = rgb_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        d_gt_norm, fg_gt = normalize_depth_for_vis(depth_gt)

        # 2. Input RGB Condition & DAP Token
        pil_img = Image.open(img_path).convert("RGB").resize((128, 128))
        img_np = np.array(pil_img) / 255.0
        img_t = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        dap_t = torch.tensor([tc["dap"]], dtype=torch.float32, device=device)

        # 3. Dynamic Variable-Length Sampling from Pure Gaussian Noise x_0 ~ N(0, I)
        torch.manual_seed(300 + row_idx)
        with torch.no_grad():
            n_slots = tc["slots"]
            x_t = torch.randn((1, n_slots, 26), device=device)
            num_steps = 35
            dt = 1.0 / num_steps

            for s in range(num_steps):
                t_val = torch.full((1,), s * dt, device=device)
                out = model(x_t, t_val, img_t, dap_t)
                v_pred = out["pred_velocity"]
                x_t = x_t + v_pred * dt

            x_gen = x_t.squeeze(0)

        # 4. Decode 26D -> 16D Part Tensor -> Helios XML
        ot_probs = torch.softmax(x_gen[:, :FM_OT_END], dim=-1)
        ot_idx = torch.argmax(ot_probs, dim=-1)
        empty_idx = FM_OT_END - 1
        existence = 1.0 - ot_probs[:, empty_idx]

        from diffusion_based.dataset.part_array_dataset import ORGAN_CATEGORIES
        raw_ot = torch.tensor([ORGAN_CATEGORIES[min(i.item(), len(ORGAN_CATEGORIES)-1)] for i in ot_idx], device=device).float()

        part_16d = torch.zeros((n_slots, NUM_FEATURES_PART), device=device)
        part_16d[:, P_COL_ORGAN_TYPE] = raw_ot
        part_16d[:, P_COL_BASE_X:P_COL_BASE_Z+1] = x_gen[:, FM_BASE_START:FM_BASE_END] / 20.0
        part_16d[:, P_COL_ROT_0:P_COL_ROT_5+1] = x_gen[:, FM_ROT_START:FM_ROT_END]
        part_16d[:, P_COL_SCALE_X:P_COL_SCALE_Z+1] = x_gen[:, FM_SCALE_START:FM_SCALE_END] / 50.0
        part_16d[:, P_COL_EXISTENCE] = existence
        part_16d[:, P_COL_CURVATURE] = x_gen[:, FM_CURV_IDX] * 100.0
        part_16d[:, P_COL_PHYLLOTACTIC_ANGLE] = x_gen[:, FM_PHYLLO_IDX] * 180.0

        xml_str = assembler.convert_to_xml_string(part_16d, plant_id=0, existence_threshold=0.35)
        arr_gen = PlantOrganArray.from_xml_string(xml_str)
        mesh_gen = renderer.geo_builder.build_mesh_from_organ_array(arr_gen, device=device)

        # 5. Render Generated 3D Mesh & Depth
        rgb_gen = renderer.render_mesh(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
        depth_gen = renderer.render_depth(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)

        mask_gen = (depth_gen > 1e-4).float().cpu().numpy()
        rgb_gen_np = rgb_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        d_gen_norm, fg_gen = normalize_depth_for_vis(depth_gen)

        # Metrics
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
            "vertices": vert_count,
            "slots": n_slots
        })

        # Render 6-column panel
        # Col 0: Input Image Condition
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title(f"{tc['name']}\nInput RGB Condition", color="white", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # Col 1: Ground Truth 3D Mesh
        axes[row_idx, 1].imshow(rgb_gt_np)
        axes[row_idx, 1].set_title(f"Ground Truth 3D Mesh\n(Verts: {mesh_gt['vertices'].shape[0]})", color="#38BDF8", fontsize=11, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # Col 2: Ground Truth Depth Shape
        axes[row_idx, 2].imshow(d_gt_norm, cmap="plasma")
        axes[row_idx, 2].set_title("GT 3D Depth Shape\n(Canopy Height Map)", color="#38BDF8", fontsize=11, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # Col 3: Canonical DiT Generated 3D Mesh
        axes[row_idx, 3].imshow(rgb_gen_np)
        axes[row_idx, 3].set_title(f"Canonical DiT (Slots: {n_slots})\nMask IoU: {iou:.3f} | Chamfer: {chamfer:.4f}", color="#34D399", fontsize=11, fontweight="bold")
        axes[row_idx, 3].axis("off")

        # Col 4: Generated Depth Shape
        axes[row_idx, 4].imshow(d_gen_norm, cmap="plasma")
        axes[row_idx, 4].set_title("Generated 3D Depth Shape\n(Predicted Height Map)", color="#34D399", fontsize=11, fontweight="bold")
        axes[row_idx, 4].axis("off")

        # Col 5: Depth Shape Error Map
        axes[row_idx, 5].imshow(depth_err_map, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[row_idx, 5].set_title(f"Depth Shape Error (|GT-Gen|)\nDepth MAE: {depth_mae:.3f}", color="#F43F5E", fontsize=11, fontweight="bold")
        axes[row_idx, 5].axis("off")

    plt.tight_layout()
    out_png = os.path.join(repo_root, "docs", "results", "assets", "fig_canonical_cowpea_dap10_30.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print("\n" + "=" * 85)
    print("CANONICAL BOTANICAL SLOT-ORDERED COWPEA DiT BENCHMARK RESULTS (DAP 010 - DAP 030)")
    print("=" * 85)
    for r in results_table:
        print(f"{r['stage']:<30} | Mask IoU: {r['mask_iou']:.4f} | Depth MAE: {r['depth_mae']:.4f} | Chamfer: {r['chamfer_dist']:.4f} | Vertices: {r['vertices']} (Slots: {r['slots']})")
    print(f"Saved benchmark figure to: {out_png}")


if __name__ == "__main__":
    main()
