"""
Generates Space Carving baseline figure comparing dense organ soup optimization
against ground truth across multiple plant growth stages (DAP 10, DAP 50, DAP 90).

Figure Columns:
  1. Ground Truth Target (RGB)
  2. Ground Truth Depth
  3. Initial Dense Organ Soup (Step 0: Dense Field covering canvas)
  4. Initial Soup Depth Map
  5. Space-Carved RGB (Step 50: Background pruned, foreground retained)
  6. Space-Carved Depth Map (Disjoint floating organ surface)
  7. 3D Oblique Perspective Render (Visualizing lack of stem connectivity)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    P_COL_EXISTENCE, P_COL_ORGAN_TYPE,
    P_COL_BASE_X, P_COL_BASE_Y, P_COL_BASE_Z,
    P_COL_ROT_0, P_COL_ROT_5,
    P_COL_SCALE_X, P_COL_SCALE_Y, P_COL_SCALE_Z,
    P_COL_CURVATURE, P_COL_PHYLLOTACTIC_ANGLE,
    ORGAN_LEAF,
    rotation_6d_to_matrix,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.eval.metrics import masked_ssim, foreground_iou, affine_invariant_depth_loss

ELEVATION_DEG = 89.88


def _to_tensor(np_img: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np_img.astype(np.float32)).to(device)
    if t.max() > 1.5:
        t = t / 255.0
    return t.permute(2, 0, 1).contiguous()


def _depth_colormap(depth_np: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("plasma")
    rgb = cmap(depth_np)[:, :, :3].astype(np.float32)
    rgb[depth_np <= 0] = 0.0
    return rgb


def _resolve_xml(rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs) and os.path.exists(rel_or_abs):
        return rel_or_abs
    full_path = os.path.join(REPO_ROOT, rel_or_abs)
    if os.path.exists(full_path):
        return full_path
    import re
    m = re.search(r'dap0*(\d+)', rel_or_abs)
    if m:
        dap_num = int(m.group(1))
        dap_cand = os.path.join(REPO_ROOT, "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders", f"rad_dap{dap_num:03d}_0000_plant_0000.xml")
        if os.path.exists(dap_cand):
            return dap_cand
    return full_path


def load_dap_target(renderer: HeliosPyTorchRenderer, device: torch.device, xml_rel: str):
    xml_path = _resolve_xml(xml_rel)
    tgt_arr = PlantOrganArray.from_xml_file(xml_path)
    tgt_part = tgt_arr.to_part_tensor(device=device)

    tgt_mesh = renderer.geo_builder.build_mesh_from_part_array(
        tgt_part, template_organ_array=tgt_arr, device=device, use_kinematics_tree=False
    )
    tgt_verts = tgt_mesh["vertices"]
    bb_min = tgt_verts.min(dim=0)[0].tolist()
    bb_max = tgt_verts.max(dim=0)[0].tolist()
    cam_bounds = {"min": bb_min, "max": bb_max}

    tgt_out = renderer.render_part_tensor_multimodal(
        tgt_part, template_organ_array=tgt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
        device=device, focus_plant=True, use_kinematics_tree=False,
        fixed_camera_bounds=cam_bounds, return_depth=True, return_mask=True,
        return_organ_masks=False, return_raw_depth=True,
    )
    return {
        "arr": tgt_arr, "part": tgt_part,
        "rgb": tgt_out["rgb"], "depth": tgt_out["depth"],
        "raw_depth": tgt_out["raw_depth"], "mask": tgt_out["mask"],
        "tgt_np": tgt_out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1),
        "cam_bounds": cam_bounds,
    }


def run_space_carving(
    target_spec: dict,
    renderer: HeliosPyTorchRenderer,
    device: torch.device,
    grid_res: int = 14,
    steps: int = 50,
):
    # 1. Create Dense Uniform Grid of Organs covering canopy [-span, span]
    cam_bounds = target_spec["cam_bounds"]
    min_x, max_x = cam_bounds["min"][0], cam_bounds["max"][0]
    min_y, max_y = cam_bounds["min"][1], cam_bounds["max"][1]
    min_z, max_z = max(0.01, cam_bounds["min"][2]), cam_bounds["max"][2]

    gx = torch.linspace(min_x, max_x, grid_res, device=device)
    gy = torch.linspace(min_y, max_y, grid_res, device=device)
    mesh_x, mesh_y = torch.meshgrid(gx, gy, indexing="ij")
    init_bases = torch.stack([mesh_x.flatten(), mesh_y.flatten(), torch.full_like(mesh_x.flatten(), (min_z + max_z) * 0.5)], dim=-1)
    N_dense = init_bases.shape[0]

    init_rot_6d = torch.zeros((N_dense, 6), device=device)
    init_rot_6d[:, 0] = 1.0; init_rot_6d[:, 4] = 1.0
    init_scale = torch.full((N_dense, 3), 0.030, device=device)

    # Initial state: 100% active dense soup (Step 0)
    dummy_tensor = torch.zeros((N_dense, 16), device=device)
    dummy_tensor[:, P_COL_EXISTENCE] = 1.0
    dummy_tensor[:, P_COL_ORGAN_TYPE] = ORGAN_LEAF
    dummy_tensor[:, P_COL_BASE_X:P_COL_BASE_Z + 1] = init_bases
    dummy_tensor[:, P_COL_ROT_0:P_COL_ROT_5 + 1] = init_rot_6d
    dummy_tensor[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] = init_scale
    dummy_arr = PlantOrganArray(dummy_tensor.cpu())

    with torch.no_grad():
        soup_init_out = renderer.render_part_tensor_multimodal(
            dummy_tensor, template_organ_array=dummy_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
            device=device, focus_plant=True, fixed_camera_bounds=cam_bounds,
            return_depth=True, return_mask=True, return_organ_masks=False, soft_existence=True,
        )
        soup_rgb_np = soup_init_out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        soup_depth_np = soup_init_out["depth"].cpu().numpy()

    # 2. Learnable Parameters for Space Carving
    opt_exist = torch.zeros(N_dense, device=device, requires_grad=True)
    opt_base = init_bases.clone().requires_grad_(True)
    opt_rot_6d = init_rot_6d.clone().requires_grad_(True)
    opt_scale_log = torch.zeros((N_dense, 3), device=device, requires_grad=True)

    optimizer = torch.optim.AdamW([
        {"params": [opt_base], "lr": 0.015},
        {"params": [opt_rot_6d], "lr": 0.05},
        {"params": [opt_scale_log], "lr": 0.04},
        {"params": [opt_exist], "lr": 0.40},
    ], weight_decay=1e-4)

    for s in range(steps):
        optimizer.zero_grad()
        R = rotation_6d_to_matrix(opt_rot_6d)
        rot_norm = torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)
        scale_eval = init_scale * torch.exp(torch.clamp(opt_scale_log, -0.8, 0.8))
        exist_eval = torch.sigmoid(opt_exist).unsqueeze(-1)

        part_eval = torch.cat([
            exist_eval,
            dummy_tensor[:, P_COL_ORGAN_TYPE:P_COL_ORGAN_TYPE + 1],
            opt_base,
            rot_norm,
            scale_eval,
            dummy_tensor[:, P_COL_CURVATURE:P_COL_CURVATURE + 1],
            dummy_tensor[:, P_COL_PHYLLOTACTIC_ANGLE:P_COL_PHYLLOTACTIC_ANGLE + 1],
        ], dim=-1)

        out = renderer.render_part_tensor_multimodal(
            part_eval, template_organ_array=dummy_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
            device=device, focus_plant=True, fixed_camera_bounds=cam_bounds,
            return_depth=False, return_mask=True, return_organ_masks=False,
            return_raw_depth=True, soft_existence=True,
        )
        loss = F.l1_loss(out["rgb"], target_spec["rgb"])
        if s > 10:
            fg_inter = out["mask"] & target_spec["mask"]
            loss_depth = affine_invariant_depth_loss(out["raw_depth"], target_spec["raw_depth"], mask=fg_inter)
            loss = loss + 0.15 * loss_depth
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        R = rotation_6d_to_matrix(opt_rot_6d)
        rot_norm = torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)
        scale_eval = init_scale * torch.exp(torch.clamp(opt_scale_log, -0.8, 0.8))
        exist_sharp = torch.sigmoid((opt_exist - 0.0) * 3.5).unsqueeze(-1)

        part_final = torch.cat([
            exist_sharp,
            dummy_tensor[:, P_COL_ORGAN_TYPE:P_COL_ORGAN_TYPE + 1],
            opt_base,
            rot_norm,
            scale_eval,
            dummy_tensor[:, P_COL_CURVATURE:P_COL_CURVATURE + 1],
            dummy_tensor[:, P_COL_PHYLLOTACTIC_ANGLE:P_COL_PHYLLOTACTIC_ANGLE + 1],
        ], dim=-1)

        # Top-down render
        out_final = renderer.render_part_tensor_multimodal(
            part_final, template_organ_array=dummy_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
            device=device, focus_plant=True, fixed_camera_bounds=cam_bounds,
            return_depth=True, return_mask=True, return_organ_masks=False, soft_existence=True,
        )
        carved_rgb_np = out_final["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        carved_depth_np = out_final["depth"].cpu().numpy()

        # Oblique perspective render (45 deg angle) to show disjoint floating organ soup
        out_oblique = renderer.render_part_tensor_multimodal(
            part_final, template_organ_array=dummy_arr, camera_height=5.0, elevation_deg=45.0,
            device=device, focus_plant=True, fixed_camera_bounds=cam_bounds,
            return_depth=False, return_mask=False, return_organ_masks=False, soft_existence=True,
        )
        oblique_rgb_np = out_oblique["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        ssim = float(masked_ssim(_to_tensor(carved_rgb_np, device), _to_tensor(target_spec["tgt_np"], device)).item())
        iou = float(foreground_iou(_to_tensor(carved_rgb_np, device), _to_tensor(target_spec["tgt_np"], device)).item())
        active_count = int((torch.sigmoid(opt_exist) > 0.4).sum().item())

    return {
        "soup_rgb": soup_rgb_np,
        "soup_depth": soup_depth_np,
        "carved_rgb": carved_rgb_np,
        "carved_depth": carved_depth_np,
        "oblique_rgb": oblique_rgb_np,
        "ssim": ssim,
        "iou": iou,
        "active_count": active_count,
        "total_organs": N_dense,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assets_dir = os.path.join(REPO_ROOT, "docs/results/assets")
    os.makedirs(assets_dir, exist_ok=True)

    dap_targets = [
        ("DAP 10 (Seedling)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_plant_0000.xml"),
        ("DAP 50 (Branching)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_plant_0000.xml"),
        ("DAP 90 (Mature)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_plant_0000.xml"),
    ]

    renderer = HeliosPyTorchRenderer(image_size=256).to(device)

    print("Generating Figure: Space Carving Baseline & Topological Ablation...")
    fig, axes = plt.subplots(3, 7, figsize=(24, 11))
    plt.subplots_adjust(wspace=0.08, hspace=0.25)

    t0 = time.time()
    for row, (title, rel_xml) in enumerate(dap_targets):
        t_row = time.time()
        print(f"  Processing {title}...")
        spec = load_dap_target(renderer, device, rel_xml)
        res = run_space_carving(spec, renderer, device, grid_res=14, steps=45)

        # Col 0: Ground Truth RGB
        axes[row, 0].imshow(spec["tgt_np"])
        axes[row, 0].set_title(f"{title}\nGround Truth RGB", fontsize=9.5, fontweight="bold")
        axes[row, 0].axis("off")

        # Col 1: Ground Truth Depth
        axes[row, 1].imshow(_depth_colormap(spec["depth"].cpu().numpy()))
        axes[row, 1].set_title("Ground Truth Depth\n(Botanical Canopy)", fontsize=9.5, fontweight="bold")
        axes[row, 1].axis("off")

        # Col 2: Dense Organ Soup (Step 0)
        axes[row, 2].imshow(res["soup_rgb"])
        axes[row, 2].set_title(f"Initial Dense Soup (Step 0)\n(N={res['total_organs']} Grid Field)", fontsize=9.5, color="darkred", fontweight="bold")
        axes[row, 2].axis("off")

        # Col 3: Initial Soup Depth
        axes[row, 3].imshow(_depth_colormap(res["soup_depth"]))
        axes[row, 3].set_title("Initial Soup Depth\n(Uniform Flat Field)", fontsize=9.5)
        axes[row, 3].axis("off")

        # Col 4: Space-Carved RGB
        axes[row, 4].imshow(res["carved_rgb"])
        axes[row, 4].set_title(f"Space-Carved RGB (Step 45)\nmSSIM: {res['ssim']:.3f} | IoU: {res['iou']:.2f}", fontsize=9.5, color="navy", fontweight="bold")
        axes[row, 4].axis("off")

        # Col 5: Space-Carved Depth
        axes[row, 5].imshow(_depth_colormap(res["carved_depth"]))
        axes[row, 5].set_title(f"Carved Surface Depth\n(Active: {res['active_count']}/{res['total_organs']})", fontsize=9.5, color="navy")
        axes[row, 5].axis("off")

        # Col 6: Oblique 3D View (Disjoint Organ Soup)
        axes[row, 6].imshow(res["oblique_rgb"])
        axes[row, 6].set_title("3D Oblique View (45°)\n(No Stem/Branch Connectivity)", fontsize=9.5, color="crimson", fontweight="bold")
        axes[row, 6].axis("off")

        print(f"    ✓ {title} done in {time.time() - t_row:.2f}s (SSIM: {res['ssim']:.3f}, IoU: {res['iou']:.2f}, Active: {res['active_count']}/{res['total_organs']})")

    out_path = os.path.join(assets_dir, "fig_space_carving_baseline.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSuccessfully generated Space Carving Figure in {time.time() - t0:.2f}s: {out_path}")


if __name__ == "__main__":
    main()
