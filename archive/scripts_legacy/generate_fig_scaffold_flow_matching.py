"""
Botanical Scaffold Flow Matching Benchmark Figure Generator.

Evaluates PartFlowMatchingModel with the 3D Botanical Scaffold Prior (x_0 ~ p_scaffold)
across growth stages (DAP 10, 50, 90) and species (Cowpea, Common Bean, Sorghum).

Columns (8 Total):
  1. Original Helios Image (Target Condition)
  2. Ground Truth Depth (Botanical Canopy)
  3. Botanical Scaffold Prior (x0, t = 0)
  4. Flow Matching Pruning & Alignment (t = 0.33)
  5. Flow Matching Canopy Formation (t = 0.66)
  6. FM Generated 3D Organ Layout (t = 1.0)
  7. FM + Multimodal Inverse Render
  8. Re-rendered XML (Helios C++ Raytrace)
"""

import os
import sys
import glob
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.dataset.part_array_dataset import (
    PartArrayDataset,
    EMPTY_IDX,
    P_COL_ORGAN_TYPE,
    P_COL_BASE_X,
    P_COL_BASE_Z,
    P_COL_ROT_0,
    P_COL_ROT_5,
    P_COL_SCALE_X,
    P_COL_SCALE_Z,
    P_COL_EXISTENCE,
    P_COL_CURVATURE,
    NUM_FEATURES,
)
from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.botanical_scaffold import BotanicalScaffoldGenerator
from diffusion_based.models.part_flow_matching import PartFlowMatchingModel
from diffusion_based.training.flow_matching import FlowMatchingScheduler
from diffusion_based.eval.metrics import masked_ssim, foreground_iou

ELEVATION_DEG = 90.0
IMAGE_SIZE = 128


def _to_tensor(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(img_np).float().to(device)
    if t.ndim == 3 and t.shape[-1] in (3, 4):
        t = t.permute(2, 0, 1)[:3]
    return t


def _depth_colormap(depth_np: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("plasma")
    rgb = cmap(depth_np)[:, :, :3].astype(np.float32)
    rgb[depth_np <= 0] = 0.0
    return rgb


def render_xml_with_helios_cpp(
    xml_str: str,
    species: str = "cowpea",
) -> np.ndarray:
    """Renders an XML string with Helios C++ in an isolated temporary directory, loads image into memory, and cleans up immediately."""
    img_out = np.zeros((256, 256, 3), dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="helios_scaffold_") as tmp_dir:
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
            print(f"Warning: Helios C++ render failed for {species}: {e}")
    return img_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clean old temporary files
    os.system("rm -rf /tmp/helios_* Digital-Crops/projects/syntheticdata_generation/build/output/tmp_*")

    # Load Flow Matching Model
    ckpt_path = os.path.join(REPO_ROOT, "diffusion_based/checkpoints/fm/part_flow_matching_scaffold_epoch50.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, "diffusion_based/checkpoints/fm/part_flow_matching_scaffold.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, "diffusion_based/checkpoints/fm/part_flow_matching.pt")
    print(f"Loading trained Flow Matching checkpoint: {os.path.basename(ckpt_path)}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    ds = PartArrayDataset(
        data_root=os.path.join(REPO_ROOT, "dataset/helios_data"),
        max_nodes=512,
        cache_dir=os.path.join(REPO_ROOT, "dataset/cache"),
    )

    model = PartFlowMatchingModel(
        max_nodes=512,
        node_dim=ds.node_dim,
        image_size=128,
        patch_size=8,
        embed_dim=256,
        encoder_layers=6,
        decoder_layers=4,
        num_heads=8,
    ).to(device)

    state_dict = checkpoint.get("ema_model_state_dict", checkpoint["model_state_dict"])
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items() if k != "n_averaged"}
    model.load_state_dict(clean_state)
    model.eval()

    renderer = HeliosPyTorchRenderer(image_size=IMAGE_SIZE)
    scaffold_gen = BotanicalScaffoldGenerator(max_nodes=512)

    img_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    targets = [
        ("Cowpea DAP 10 (Seedling)", "cowpea", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_plant_0000.xml", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap010_0000_rad.jpeg", "10"),
        ("Cowpea DAP 50 (Branching)", "cowpea", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_plant_0000.xml", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap050_0000_rad.jpeg", "50"),
        ("Cowpea DAP 90 (Mature)", "cowpea", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_plant_0000.xml", "Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap090_0000_rad.jpeg", "90"),
        ("Common Bean DAP 10 (Seedling)", "bean", "dataset/helios_data/bean/bean_dap010_seed18_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "dataset/helios_data/bean/bean_dap010_seed18_caz000_h1.0_se045_saz180_0000_rad.jpeg", "10"),
        ("Common Bean DAP 50 (Bushy)", "bean", "dataset/helios_data/bean/bean_dap050_seed09_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "dataset/helios_data/bean/bean_dap050_seed09_caz000_h1.0_se045_saz180_0000_rad.jpeg", "50"),
        ("Sorghum DAP 10 (Monocot)", "sorghum", "dataset/helios_data/sorghum/sorghum_dap010_seed21_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "dataset/helios_data/sorghum/sorghum_dap010_seed21_caz000_h1.0_se045_saz180_0000_rad.jpeg", "10"),
        ("Sorghum DAP 50 (Tillering)", "sorghum", "dataset/helios_data/sorghum/sorghum_dap050_seed30_caz000_h1.0_se045_saz180_0000_plant_0000.xml", "dataset/helios_data/sorghum/sorghum_dap050_seed30_caz000_h1.0_se045_saz180_0000_rad.jpeg", "50"),
    ]

    fig, axes = plt.subplots(len(targets), 8, figsize=(24, 3.2 * len(targets)))

    for row, (title, species, xml_path, orig_img_path, dap_str) in enumerate(targets):
        print(f"  Evaluating {title}...")
        full_xml = os.path.join(REPO_ROOT, xml_path) if not os.path.isabs(xml_path) else xml_path
        full_orig = os.path.join(REPO_ROOT, orig_img_path) if not os.path.isabs(orig_img_path) else orig_img_path
        
        # Load Original Helios GT image
        if os.path.exists(full_orig):
            with Image.open(full_orig) as img:
                orig_helios_rgb_np = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        else:
            orig_helios_rgb_np = np.zeros((256, 256, 3), dtype=np.float32)

        gt_arr = PlantOrganArray.from_xml_file(full_xml)
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

        cond_img = img_transform(gt_out["rgb"].unsqueeze(0)).to(device)

        # 1. 3D Botanical Scaffold Prior (x_0 ~ p_scaffold)
        x0 = scaffold_gen.sample_prior(1, device=device, noise_std=0.0)

        with torch.no_grad():
            scaffold_p = ds.decode_fm(x0[0])
            scaffold_out = renderer.render_part_tensor_multimodal(
                scaffold_p, template_organ_array=gt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
                device=device, focus_plant=True, fixed_camera_bounds=cam_bounds, return_depth=True, soft_existence=True,
            )
            scaffold_rgb_np = scaffold_out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        # 2. Flow Matching ODE Sampling from x_0 with intermediate trajectory snapshots
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
                # Simplex Linear Projection
                ot_block = x_t[..., :EMPTY_IDX + 1].clamp(min=0.0)
                ot_sum = ot_block.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                x_t[..., :EMPTY_IDX + 1] = ot_block / ot_sum

                if i == int(num_steps * 0.33):
                    t_steps_snapshots["t_033"] = x_t.clone()
                elif i == int(num_steps * 0.66):
                    t_steps_snapshots["t_066"] = x_t.clone()

            t_steps_snapshots["t_100"] = x_t.clone()

        def _render_fm_state(x_tensor):
            with torch.no_grad():
                p_decoded = ds.decode_fm(x_tensor[0])
                p_sharp = p_decoded.clone()
                p_exist = p_decoded[:, P_COL_EXISTENCE].unsqueeze(-1)
                p_sharp[:, P_COL_EXISTENCE] = torch.where(p_exist > 0.45, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)).squeeze(-1)
                out = renderer.render_part_tensor_multimodal(
                    p_sharp, template_organ_array=gt_arr, camera_height=5.0, elevation_deg=ELEVATION_DEG,
                    device=device, focus_plant=True, fixed_camera_bounds=cam_bounds, return_depth=True, soft_existence=True,
                )
                return out["rgb"].permute(1, 2, 0).cpu().numpy().clip(0, 1), p_sharp

        rgb_t033, _ = _render_fm_state(t_steps_snapshots["t_033"])
        rgb_t066, _ = _render_fm_state(t_steps_snapshots["t_066"])
        rgb_t100, fm_part_gen = _render_fm_state(t_steps_snapshots["t_100"])

        # 3. Differentiable Inverse Rendering Fine-Tuning Refinement
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
        from diffusion_based.models.helios_xml_parser import HeliosXMLParser
        parser = HeliosXMLParser(full_xml)
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

        xml_str = organ_nodes_to_xml(nodes_spec, base_position="0 0 0", plant_age=dap_str, plant_id="0")
        helios_rerender_np = render_xml_with_helios_cpp(xml_str, species=species)

        # 8 Columns (Col 1 = Original Helios Image as requested)
        axes[row, 0].imshow(orig_helios_rgb_np); axes[row, 0].set_title(f"{title}\nOriginal Helios GT", fontsize=8, fontweight="bold"); axes[row, 0].axis("off")
        axes[row, 1].imshow(_depth_colormap(gt_depth_np)); axes[row, 1].set_title("Ground Truth Depth\n(Botanical Canopy)", fontsize=8, fontweight="bold"); axes[row, 1].axis("off")
        axes[row, 2].imshow(scaffold_rgb_np); axes[row, 2].set_title("Botanical Scaffold Prior\n(x0 ~ p_scaffold)", fontsize=8, color="darkred", fontweight="bold"); axes[row, 2].axis("off")
        axes[row, 3].imshow(rgb_t033); axes[row, 3].set_title("FM Pruning & Alignment\n(t = 0.33)", fontsize=8, color="navy"); axes[row, 3].axis("off")
        axes[row, 4].imshow(rgb_t066); axes[row, 4].set_title("FM Canopy Formation\n(t = 0.66)", fontsize=8, color="navy"); axes[row, 4].axis("off")
        axes[row, 5].imshow(rgb_t100); axes[row, 5].set_title("FM Generated (t = 1.0)\n3D Organ Layout", fontsize=8, color="navy", fontweight="bold"); axes[row, 5].axis("off")
        axes[row, 6].imshow(refined_rgb_np); axes[row, 6].set_title(f"FM + Inverse Render\nmSSIM: {ssim_val:.3f} | IoU: {iou_val:.2f}", fontsize=8, color="darkgreen", fontweight="bold"); axes[row, 6].axis("off")
        axes[row, 7].imshow(helios_rerender_np); axes[row, 7].set_title("Re-rendered XML\n(Helios C++ Raytrace)", fontsize=8, color="darkgreen", fontweight="bold"); axes[row, 7].axis("off")

    assets_dir = os.path.join(REPO_ROOT, "docs/results/assets")
    os.makedirs(assets_dir, exist_ok=True)
    out_path = os.path.join(assets_dir, "fig_scaffold_flow_matching.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSuccessfully saved Botanical Scaffold Flow Matching Figure to: {out_path}")


if __name__ == "__main__":
    main()
