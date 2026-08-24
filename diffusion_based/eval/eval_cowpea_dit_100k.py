"""
High-Resolution Lifespan Benchmark for 150M DiT-Large Cowpea Model (DAP 010 to DAP 090).
Visualizes:
  - Col 0: Input Observation Image (RGB Condition)
  - Col 1: Ground Truth 3D Mesh (Helios Geometry)
  - Col 2: Ground Truth 3D Depth Shape Map
  - Col 3: Generated 3D Mesh from Pure Gaussian Noise x0 ~ N(0, I)
  - Col 4: Generated 3D Depth Shape Map
  - Col 5: Depth Shape Error Heatmap (|GT - Gen|)
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.canonical_cowpea_dit_large import CanonicalCowpeaDiTLargeModel
from diffusion_based.models.plant_organ_array import PlantOrganArray, NUM_FEATURES_PART
from diffusion_based.models.part_assembly_to_xml import PartAssemblyToXMLConverter
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.dataset.part_array_dataset import (
    ORGAN_CATEGORIES, EMPTY_IDX, FM_NODE_DIM, FM_OT_END,
    FM_BASE_START, FM_BASE_END, FM_ROT_START, FM_ROT_END,
    FM_SCALE_START, FM_SCALE_END, FM_CURV_IDX, FM_PHYLLO_IDX
)


def normalize_depth_for_vis(depth_tensor: torch.Tensor):
    d_np = depth_tensor.detach().cpu().numpy()
    fg_mask = d_np > 1e-4
    if not fg_mask.any():
        return np.zeros_like(d_np), fg_mask
    d_min, d_max = d_np[fg_mask].min(), d_np[fg_mask].max()
    d_norm = np.zeros_like(d_np)
    if d_max > d_min:
        d_norm[fg_mask] = (d_np[fg_mask] - d_min) / (d_max - d_min)
    else:
        d_norm[fg_mask] = 1.0
    return d_norm, fg_mask


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Evaluating 232M DiT-Large Cowpea Model on {device}...")

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt_path = args.checkpoint
    else:
        candidates = [
            os.path.join(repo_root, "diffusion_based", "checkpoints", "fm", "cowpea_dit_large_2xh100_ddp.pt"),
            os.path.join(repo_root, "diffusion_based", "checkpoints", "fm", "cowpea_dit_large_150m.pt"),
            os.path.join(repo_root, "diffusion_based", "checkpoints", "fm", "test_large.pt"),
        ]
        ckpt_path = next((c for c in candidates if os.path.exists(c)), None)
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoint found.")

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Detect max_slots and embed_dim from state_dict
    pos_shape = state_dict.get("slot_pos_embed", torch.zeros(1, 4096, 768)).shape
    max_slots = pos_shape[1] if len(pos_shape) > 1 else 4096
    embed_dim = pos_shape[2] if len(pos_shape) > 2 else 768

    model = CanonicalCowpeaDiTLargeModel(
        max_slots=max_slots,
        node_dim=26,
        image_size=128,
        patch_size=8,
        embed_dim=embed_dim,
        encoder_layers=16,
        decoder_layers=12,
        num_heads=16,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    renderer = HeliosPyTorchRenderer(image_size=256).to(device)
    assembler = PartAssemblyToXMLConverter()

    data_dir = os.path.join(repo_root, "dataset", "helios_data", "cowpea")
    test_cases = [
        {"name": "Cowpea DAP 010 (Juvenile)", "dap": 10.0, "slots": 50, "prefix": "cowpea_dap010_seed00_caz000_h1.0_se045_saz180_0000"},
        {"name": "Cowpea DAP 020 (Vegetative)", "dap": 20.0, "slots": 151, "prefix": "cowpea_dap020_seed00_caz000_h1.0_se045_saz180_0000"},
        {"name": "Cowpea DAP 030 (Branching)", "dap": 30.0, "slots": 411, "prefix": "cowpea_dap030_seed00_caz000_h1.0_se045_saz180_0000"},
        {"name": "Cowpea DAP 050 (Canopy)", "dap": 50.0, "slots": 512, "prefix": "cowpea_dap050_seed00_caz000_h1.0_se045_saz180_0000"},
        {"name": "Cowpea DAP 090 (Mature)", "dap": 90.0, "slots": 512, "prefix": "cowpea_dap090_seed00_caz000_h1.0_se045_saz180_0000"},
    ]

    # Filter to existing samples
    valid_cases = []
    for tc in test_cases:
        x_path = os.path.join(data_dir, f"{tc['prefix']}_plant_0000.xml")
        i_path = os.path.join(data_dir, f"{tc['prefix']}_rad.jpeg")
        if os.path.exists(x_path) and os.path.exists(i_path):
            tc["xml"] = x_path
            tc["img"] = i_path
            valid_cases.append(tc)

    if not valid_cases:
        print("No exact test cases found, discovering available samples...")
        xmls = sorted(glob.glob(os.path.join(data_dir, "*_plant_0000.xml")))[:4]
        for x in xmls:
            prefix = os.path.basename(x).split("_plant_0000.xml")[0]
            img = os.path.join(data_dir, f"{prefix}_rad.jpeg")
            if os.path.exists(img):
                valid_cases.append({"name": prefix, "xml": x, "img": img, "dap": 30.0, "slots": 256})

    fig, axes = plt.subplots(len(valid_cases), 6, figsize=(24, 4.2 * len(valid_cases)))
    if len(valid_cases) == 1:
        axes = np.expand_dims(axes, 0)
    fig.patch.set_facecolor("#080C14")

    results_table = []

    for row_idx, tc in enumerate(valid_cases):
        # 1. Ground Truth 3D Render
        arr_gt = PlantOrganArray.from_xml_file(tc["xml"])
        mesh_gt = renderer.geo_builder.build_mesh_from_organ_array(arr_gt, device=device)
        rgb_gt = renderer(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
        depth_gt = renderer.render_depth(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)

        seg_gt = renderer.render_organ_segmentation(mesh_gt, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
        seg_gt_np = seg_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        mask_gt = (depth_gt > 1e-4).float().cpu().numpy()
        rgb_gt_np = rgb_gt.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        d_gt_norm, fg_gt = normalize_depth_for_vis(depth_gt)

        # 2. Input RGB Condition & DAP Token
        pil_img = Image.open(tc["img"]).convert("RGB").resize((128, 128))
        img_np = np.array(pil_img) / 255.0
        img_t = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        dap_t = torch.tensor([tc["dap"]], dtype=torch.float32, device=device)

        # 3. Dynamic Variable-Length Sampling from Pure Gaussian Noise x_0 ~ N(0, I)
        torch.manual_seed(500 + row_idx)
        with torch.no_grad():
            n_slots = tc["slots"]
            x_t = torch.randn((1, n_slots, 26), device=device)
            num_steps = 35
            dt = 1.0 / num_steps
            for s in range(num_steps):
                t_val = torch.full((1,), s * dt, device=device)
                out = model(x_t, t_val, img_t, dap_t)
                x_t = x_t + out["pred_velocity"] * dt
            x_gen = x_t.squeeze(0)

        # 4. Decode 26D -> 16D Part Tensor -> XML -> PlantOrganArray
        ot_probs = torch.softmax(x_gen[:, :FM_OT_END], dim=-1)
        ot_idx = torch.argmax(ot_probs, dim=-1)
        raw_ot = torch.tensor([ORGAN_CATEGORIES[min(i.item(), len(ORGAN_CATEGORIES)-1)] for i in ot_idx], device=device).float()

        part_16d = torch.zeros((n_slots, NUM_FEATURES_PART), device=device)
        part_16d[:, 0] = raw_ot
        part_16d[:, 1:4] = x_gen[:, FM_BASE_START:FM_BASE_END] / 20.0
        part_16d[:, 4:10] = x_gen[:, FM_ROT_START:FM_ROT_END]
        part_16d[:, 10:13] = x_gen[:, FM_SCALE_START:FM_SCALE_END] / 50.0
        part_16d[:, 13] = 1.0 - ot_probs[:, FM_OT_END - 1]
        part_16d[:, 14] = x_gen[:, FM_CURV_IDX] * 100.0
        part_16d[:, 15] = x_gen[:, FM_PHYLLO_IDX] * 180.0

        xml_str = assembler.convert_to_xml_string(part_16d, plant_id=0, existence_threshold=0.30)
        arr_gen = PlantOrganArray.from_xml_string(xml_str)
        mesh_gen = renderer.geo_builder.build_mesh_from_organ_array(arr_gen, device=device)

        # 5. Render Generated 3D Mesh, Depth & Organ Segmentation
        depth_gen = renderer.render_depth(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)
        rgb_gen = renderer(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, background="white", focus_plant=True)
        seg_gen = renderer.render_organ_segmentation(mesh_gen, azimuth_deg=0.0, elevation_deg=90.0, camera_height=5.0, focus_plant=True)

        mask_gen = (depth_gen > 1e-4).float().cpu().numpy()
        rgb_gen_np = rgb_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        seg_gen_np = seg_gen.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        d_gen_norm, fg_gen = normalize_depth_for_vis(depth_gen)

        # 6. Compute Depth Shape Error Map & Metrics
        # True physical botanical canopy height in Z-axis (cm)
        h_gt_cm = (mesh_gt['vertices'][:, 2].max() - mesh_gt['vertices'][:, 2].min()).item() * 100.0 if mesh_gt['vertices'].shape[0] > 0 else 0.0
        h_gen_cm = (mesh_gen['vertices'][:, 2].max() - mesh_gen['vertices'][:, 2].min()).item() * 100.0 if mesh_gen['vertices'].shape[0] > 0 else 0.0

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

        import matplotlib.patches as mpatches
        ORGAN_LEGEND_MAP = {
            0: ("#8B4513", "Stem/Internode"),
            1: ("#E68026", "Petiole"),
            2: ("#228B22", "Leaf"),
            3: ("#9ACD32", "Peduncle"),
            4: ("#FFD700", "Flower"),
            5: ("#E63333", "Pod/Fruit"),
        }

        # Plot 6 columns
        # Col 1: PyTorch Differentiable RGB Input (GT)
        axes[row_idx, 0].imshow(rgb_gt_np)
        axes[row_idx, 0].set_title(f"{tc['name']}\nDiff RGB Input ({arr_gt.num_nodes} organs)", color="#4ADE80", fontsize=10, fontweight="bold")
        axes[row_idx, 0].text(0.03, 0.03, f"N={arr_gt.num_nodes}", transform=axes[row_idx, 0].transAxes, fontsize=8, color='white', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # Col 2: PyTorch Differentiable Depth Input (GT)
        axes[row_idx, 1].imshow(d_gt_norm, cmap="plasma")
        axes[row_idx, 1].set_title("Diff Depth Input", color="#22D3EE", fontsize=10, fontweight="bold")
        axes[row_idx, 1].text(0.03, 0.03, f"Height: 0–{h_gt_cm:.1f} cm", transform=axes[row_idx, 1].transAxes, fontsize=8, color='#22D3EE', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # Col 3: PyTorch Differentiable Organ Segmentation Mask Input (GT)
        axes[row_idx, 2].imshow(seg_gt_np)
        axes[row_idx, 2].set_title("Diff Organ Seg Input", color="#A78BFA", fontsize=10, fontweight="bold")
        patches = [mpatches.Patch(color=c, label=l) for ot, (c, l) in ORGAN_LEGEND_MAP.items()]
        axes[row_idx, 2].legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.85, facecolor='#0D1117', labelcolor='white', edgecolor='#334155', ncol=1)

        # Col 4: Generated PyTorch Differentiable RGB
        axes[row_idx, 3].imshow(rgb_gen_np)
        axes[row_idx, 3].set_title(f"Gen Diff RGB\nIoU: {iou:.3f} | CD: {chamfer:.3f}", color="#60A5FA", fontsize=10, fontweight="bold")
        axes[row_idx, 3].text(0.03, 0.03, f"N={arr_gen.num_nodes} ({vert_count}v)", transform=axes[row_idx, 3].transAxes, fontsize=8, color='white', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # Col 5: Generated PyTorch Differentiable Depth
        axes[row_idx, 4].imshow(d_gen_norm, cmap="plasma")
        axes[row_idx, 4].set_title("Gen Diff Depth", color="#F472B6", fontsize=10, fontweight="bold")
        axes[row_idx, 4].text(0.03, 0.03, f"Height: 0–{h_gen_cm:.1f} cm", transform=axes[row_idx, 4].transAxes, fontsize=8, color='#F472B6', va='bottom', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # Col 6: Generated PyTorch Differentiable Organ Segmentation Mask
        axes[row_idx, 5].imshow(seg_gen_np)
        axes[row_idx, 5].set_title("Gen Diff Organ Seg", color="#FB923C", fontsize=10, fontweight="bold")
        axes[row_idx, 5].legend(handles=patches, loc='lower right', fontsize=6, framealpha=0.85, facecolor='#0D1117', labelcolor='white', edgecolor='#334155', ncol=1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)

    plt.tight_layout()
    save_fig_path = os.path.join(repo_root, "docs", "results", "assets", "fig_cowpea_100k_lifespan_benchmark.png")
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print("\n" + "="*85)
    print("150M DiT-LARGE COWPEA LIFESPAN BENCHMARK RESULTS (DAP 010 - DAP 090)")
    print("="*85)
    for r in results_table:
        print(f"{r['stage']:<30} | Mask IoU: {r['mask_iou']:.4f} | Depth MAE: {r['depth_mae']:.4f} | Chamfer: {r['chamfer_dist']:.4f} | Vertices: {r['vertices']} (Slots: {r['slots']})")
    print(f"Saved benchmark figure to: {save_fig_path}")


if __name__ == "__main__":
    main()
