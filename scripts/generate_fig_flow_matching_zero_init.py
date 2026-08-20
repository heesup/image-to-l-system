"""
Generates Publication Figure for Flow Matching Generation from Zero Plant Array ($x_0 = 0$).

Layout (3 Rows x 8 Columns):
  - Row 1: DAP 10 (Seedling)
  - Row 2: DAP 50 (Branching)
  - Row 3: DAP 90 (Mature)

Columns:
  1. Ground Truth Target RGB (Condition Input)
  2. Ground Truth Depth Map (Botanical Surface)
  3. Initial Zero Plant Array (x_0 = 0, Empty Canvas, 0 Foreground)
  4. Flow Matching ODE Trajectory (t = 0.33: Sprouting Phase)
  5. Flow Matching ODE Trajectory (t = 0.66: Branching/Expansion)
  6. Flow Matching Generated 3D Plant (t = 1.0)
  7. Differentiable Inverse Rendering Refined 3D Plant
  8. Re-rendered XML using Helios C++ (OptiX Raytrace)
"""

import os
import sys
import glob
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.dataset.part_array_dataset import PartArrayDataset, FM_NODE_DIM, EMPTY_IDX, FM_OT_END
from diffusion_based.models.part_flow_matching import PartFlowMatchingModel
from diffusion_based.training.flow_matching import FlowMatchingScheduler
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    P_COL_EXISTENCE, P_COL_ORGAN_TYPE,
    P_COL_BASE_X, P_COL_BASE_Y, P_COL_BASE_Z,
    P_COL_ROT_0, P_COL_ROT_5,
    P_COL_SCALE_X, P_COL_SCALE_Y, P_COL_SCALE_Z,
    P_COL_CURVATURE, P_COL_PHYLLOTACTIC_ANGLE,
    ORGAN_LEAF, rotation_6d_to_matrix,
)
from diffusion_based.models.helios_xml_parser import organ_nodes_to_xml
from diffusion_based.eval.metrics import masked_ssim, foreground_iou

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


def render_xml_with_helios_cpp(
    xml_str: str,
    species: str = "cowpea",
) -> np.ndarray:
    """Renders an XML string with Helios C++ in an isolated temporary directory, loads image into memory, and cleans up immediately."""
    import tempfile
    img_out = np.zeros((256, 256, 3), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="helios_render_") as tmp_dir:
        xml_path = os.path.join(tmp_dir, "plant.xml")
        with open(xml_path, "w") as f:
            f.write(xml_str)

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
                    img_out = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Warning: Helios C++ render failed: {e}")
    return img_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assets_dir = os.path.join(REPO_ROOT, "docs/results/assets")
    os.makedirs(assets_dir, exist_ok=True)
    tmp_xml_dir = os.path.join(REPO_ROOT, "Digital-Crops/projects/syntheticdata_generation/build/output/tmp_fm_xmls")
    os.makedirs(tmp_xml_dir, exist_ok=True)

    # 1. Load latest Flow Matching Checkpoint
    ckpt_dir = os.path.join(REPO_ROOT, "diffusion_based/checkpoints/fm")
    ckpt_candidates = sorted(glob.glob(os.path.join(ckpt_dir, "part_flow_matching_epoch*.pt")), key=os.path.getmtime)
    if not ckpt_candidates:
        print(f"No checkpoints found in {ckpt_dir}")
        return
    latest_ckpt = ckpt_candidates[-1]
    print(f"Loading trained Flow Matching checkpoint: {os.path.basename(latest_ckpt)}")

    ds = PartArrayDataset(data_root="dataset/helios_data", max_nodes=512, cache_dir="dataset/cache")
    model = PartFlowMatchingModel(
        max_nodes=512, node_dim=ds.node_dim,
        image_size=128, patch_size=8, embed_dim=256,
        encoder_layers=6, decoder_layers=4, num_heads=8,
    ).to(device)

    ckpt_data = torch.load(latest_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt_data.get("ema_model_state_dict", ckpt_data["model_state_dict"])
    cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if k != "n_averaged"}
    model.load_state_dict(cleaned_state_dict)
    model.eval()

    scheduler = FlowMatchingScheduler()
    renderer = HeliosPyTorchRenderer(image_size=256).to(device)

    dap_targets = [
        ("DAP 10 (Seedling)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_plant_0000.xml", "dap010"),
        ("DAP 50 (Branching)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_plant_0000.xml", "dap050"),
        ("DAP 90 (Mature)", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_plant_0000.xml", "dap090"),
    ]

    print("Generating Figure: Flow Matching from Zero Plant Array ($x_0 = 0$)...")
    fig, axes = plt.subplots(3, 8, figsize=(26, 11))
    plt.subplots_adjust(wspace=0.08, hspace=0.25)

    img_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for row, (title, rel_xml, dap_tag) in enumerate(dap_targets):
        print(f"  Evaluating {title}...")
        xml_path = _resolve_xml(rel_xml)
        gt_arr = PlantOrganArray.from_xml_file(xml_path)
        gt_part = gt_arr.to_part_tensor(device=device)

        # Ground Truth multi-modal render
        mesh = renderer.geo_builder.build_mesh_from_part_array(gt_part, template_organ_array=gt_arr, device=device, use_kinematics_tree=False)
        verts = mesh["vertices"]
        cam_bounds = {"min": verts.min(dim=0)[0].tolist(), "max": verts.max(dim=0)[0].tolist()}

        gt_out = renderer.render_part_tensor_multimodal(
            gt_part, template_organ_array=gt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
            device=device, focus_plant=True, fixed_camera_bounds=cam_bounds, return_depth=True,
        )
        gt_rgb_np = gt_out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        gt_depth_np = gt_out["depth"].cpu().numpy()

        # Prepared Condition Image for Flow Matching
        cond_img = img_transform(gt_out["rgb"].unsqueeze(0)).to(device)

        # 1. Pure Zero Plant Array Initialization (x_0 = 0, 100% EMPTY)
        x0 = torch.zeros((1, 512, ds.node_dim), device=device)
        x0[:, :, EMPTY_IDX] = 1.0

        # Zero state rendering (Step 0)
        with torch.no_grad():
            zero_p = ds.decode_fm(x0[0])
            zero_out = renderer.render_part_tensor_multimodal(
                zero_p, template_organ_array=gt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
                device=device, focus_plant=True, fixed_camera_bounds=cam_bounds, return_depth=True, soft_existence=True,
            )
            zero_rgb_np = zero_out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        # 2. ODE Trajectory Sampling from x_0 = 0
        num_steps = 35
        dt = 1.0 / num_steps
        x_t = x0.clone()

        t_steps_snapshots = {}
        with torch.no_grad():
            for i in range(num_steps):
                t_val = i * dt
                t_tensor = torch.full((1,), t_val, device=device)
                out = model(x_t, t_tensor, cond_img)
                v = out["pred_velocity"]
                x_t = x_t + v * dt
                x_t[..., :FM_OT_END] = F.softmax(x_t[..., :FM_OT_END], dim=-1)

                if i == int(num_steps * 0.33):
                    t_steps_snapshots["t_033"] = x_t.clone()
                elif i == int(num_steps * 0.66):
                    t_steps_snapshots["t_066"] = x_t.clone()

            t_steps_snapshots["t_100"] = x_t.clone()

        # Render Trajectory Snapshots
        def _render_fm_state(fm_state_t):
            p = ds.decode_fm(fm_state_t[0])
            p_sharp = p.clone()
            p_exist = p[:, P_COL_EXISTENCE].unsqueeze(-1)
            p_sharp[:, P_COL_EXISTENCE] = torch.where(p_exist > 0.45, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).squeeze(-1)
            with torch.no_grad():
                out = renderer.render_part_tensor_multimodal(
                    p_sharp, template_organ_array=gt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
                    device=device, focus_plant=True, fixed_camera_bounds=cam_bounds, return_depth=True, soft_existence=True,
                )
                return out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1), p_sharp

        rgb_t033, _ = _render_fm_state(t_steps_snapshots["t_033"])
        rgb_t066, _ = _render_fm_state(t_steps_snapshots["t_066"])
        rgb_t100, fm_part_gen = _render_fm_state(t_steps_snapshots["t_100"])

        # 3. Differentiable Inverse Rendering Fine-Tuning Refinement
        # Refines continuous leaf poses, scales, and global canopy coverage directly from flow matching
        refined_part = fm_part_gen.clone()
        opt_yaw = torch.zeros(1, device=device, requires_grad=True)
        opt_global_scale = torch.zeros(1, device=device, requires_grad=True)
        opt_scale = torch.zeros(3, device=device, requires_grad=True)
        opt_delta_base = torch.zeros((512, 3), device=device, requires_grad=True)

        optimizer = torch.optim.AdamW([
            {"params": [opt_yaw], "lr": 0.02},
            {"params": [opt_global_scale], "lr": 0.04},
            {"params": [opt_scale], "lr": 0.03},
            {"params": [opt_delta_base], "lr": 0.02},
        ])

        gt_fg = (gt_out["depth"] > 0.01).float()
        for _ in range(40):
            optimizer.zero_grad()
            g_scale = torch.exp(opt_global_scale.clamp(-0.8, 0.8))
            eval_bases = refined_part[:, P_COL_BASE_X:P_COL_BASE_Z + 1] * g_scale + torch.tanh(opt_delta_base) * 0.03
            eval_scales = refined_part[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] * torch.exp(opt_scale * 0.2) * g_scale
            p_eval = torch.cat([
                refined_part[:, :P_COL_BASE_X],
                eval_bases,
                refined_part[:, P_COL_ROT_0:P_COL_SCALE_X],
                eval_scales,
                refined_part[:, P_COL_CURVATURE:],
            ], dim=-1)

            loss_out = renderer.render_part_tensor_multimodal(
                p_eval, template_organ_array=gt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
                device=device, focus_plant=True, fixed_camera_bounds=cam_bounds, return_depth=True, soft_existence=True,
            )
            pred_fg = (loss_out["depth"] > 0.01).float()
            intersection = gt_fg * pred_fg
            iou = (intersection.sum()) / (gt_fg.sum() + pred_fg.sum() - intersection.sum() + 1e-6)
            loss_silhouette = 1.0 - iou
            loss_rgb = F.l1_loss(loss_out["rgb"], gt_out["rgb"])
            loss_depth = (torch.abs(loss_out["depth"] - gt_out["depth"]) * intersection).sum() / (intersection.sum() + 1e-6) if intersection.sum() > 10 else torch.tensor(0.0, device=device)

            loss = 1.0 * loss_rgb + 1.5 * loss_depth + 2.0 * loss_silhouette
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            g_scale = torch.exp(opt_global_scale.clamp(-0.8, 0.8))
            eval_bases = refined_part[:, P_COL_BASE_X:P_COL_BASE_Z + 1] * g_scale + torch.tanh(opt_delta_base) * 0.03
            eval_scales = refined_part[:, P_COL_SCALE_X:P_COL_SCALE_Z + 1] * torch.exp(opt_scale * 0.2) * g_scale
            p_eval = torch.cat([
                refined_part[:, :P_COL_BASE_X],
                eval_bases,
                refined_part[:, P_COL_ROT_0:P_COL_SCALE_X],
                eval_scales,
                refined_part[:, P_COL_CURVATURE:],
            ], dim=-1)
            ref_out = renderer.render_part_tensor_multimodal(
                p_eval, template_organ_array=gt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
                device=device, focus_plant=True, fixed_camera_bounds=cam_bounds, return_depth=True, soft_existence=True,
            )
            refined_rgb_np = ref_out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        ssim_val = float(masked_ssim(_to_tensor(refined_rgb_np, device), _to_tensor(gt_rgb_np, device)).item())
        iou_val = float(foreground_iou(_to_tensor(refined_rgb_np, device), _to_tensor(gt_rgb_np, device)).item())

        # 4. Re-rendered XML using Helios C++
        # Update node geometry from the flow-matched & refined 3D prediction
        from diffusion_based.models.helios_xml_parser import HeliosXMLParser
        parser = HeliosXMLParser(xml_path)
        nodes_spec = parser.get_all_organ_nodes()
        p_np = p_eval.detach().cpu().numpy()
        for idx, node in enumerate(nodes_spec):
            if idx < p_np.shape[0]:
                exist_val = float(p_np[idx, P_COL_EXISTENCE])
                scale_val = float(p_np[idx, P_COL_SCALE_X])
                if exist_val < 0.40:
                    if "leaf_scale" in node._xml_params:
                        node._xml_params["leaf_scale"] = "0.00001"
                    if "petiole_length" in node._xml_params:
                        node._xml_params["petiole_length"] = "0.0001"
                    if "internode_length" in node._xml_params:
                        node._xml_params["internode_length"] = "0.0001"
                else:
                    if "leaf_scale" in node._xml_params and scale_val > 1e-4:
                        node._xml_params["leaf_scale"] = f"{scale_val:.4f}"

        plant_age_str = "10" if "010" in dap_tag else ("50" if "050" in dap_tag else "90")
        xml_str = organ_nodes_to_xml(nodes_spec, base_position="0 0 0", plant_age=plant_age_str, plant_id="0")
        helios_rerender_np = render_xml_with_helios_cpp(xml_str, species="cowpea")

        # Render Columns
        axes[row, 0].imshow(gt_rgb_np); axes[row, 0].set_title(f"{title}\nTarget RGB Condition", fontsize=9, fontweight="bold"); axes[row, 0].axis("off")
        axes[row, 1].imshow(_depth_colormap(gt_depth_np)); axes[row, 1].set_title("Ground Truth Depth\n(Botanical Canopy)", fontsize=9, fontweight="bold"); axes[row, 1].axis("off")
        axes[row, 2].imshow(zero_rgb_np); axes[row, 2].set_title("Initial Prior (x0 = 0)\n(Zero Plant Array)", fontsize=9, color="darkred", fontweight="bold"); axes[row, 2].axis("off")
        axes[row, 3].imshow(rgb_t033); axes[row, 3].set_title("Flow Matching (t = 0.33)\nSprouting Phase", fontsize=9, color="navy"); axes[row, 3].axis("off")
        axes[row, 4].imshow(rgb_t066); axes[row, 4].set_title("Flow Matching (t = 0.66)\nCanopy Expansion", fontsize=9, color="navy"); axes[row, 4].axis("off")
        axes[row, 5].imshow(rgb_t100); axes[row, 5].set_title("FM Generated (t = 1.0)\n3D Organ Layout", fontsize=9, color="navy", fontweight="bold"); axes[row, 5].axis("off")
        axes[row, 6].imshow(refined_rgb_np); axes[row, 6].set_title(f"FM + Inverse Render\nmSSIM: {ssim_val:.3f} | IoU: {iou_val:.2f}", fontsize=9, color="darkgreen", fontweight="bold"); axes[row, 6].axis("off")
        axes[row, 7].imshow(helios_rerender_np); axes[row, 7].set_title("Re-rendered XML\n(Helios C++ Raytrace)", fontsize=9, color="darkgreen", fontweight="bold"); axes[row, 7].axis("off")

    out_path = os.path.join(assets_dir, "fig_flow_matching_zero_trajectory.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSuccessfully saved Flow Matching Figure to: {out_path}")


if __name__ == "__main__":
    main()
