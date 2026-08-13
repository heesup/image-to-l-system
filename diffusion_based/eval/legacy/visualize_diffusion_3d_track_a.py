import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Dict, Any

from dataset.plant3d_dataset import Plant3DDataset
from dataset.helios_dataset import HeliosPlantDataset
from diffusion_based.models.legacy.graph_diffuser_3d_track_a import PlantGraphDiffuser3D
from diffusion_based.training.train_diffusion import DDPMScheduler, get_device


ORGAN_COLORS = {
    0: (0.55, 0.27, 0.07),   # internode: brown
    1: (0.68, 0.85, 0.38),   # petiole: light green
    2: (0.13, 0.55, 0.13),   # leaf: forest green
    3: (0.85, 0.85, 0.20),   # floral_bud: yellow
}

ORGAN_NAMES = {0: "internode", 1: "petiole", 2: "leaf", 3: "floral_bud"}


def denormalize_angle(norm_val: float) -> float:
    return (norm_val - 0.5) * 360.0


def direction_from_angles(pitch_norm: float, yaw_norm: float, roll_norm: float) -> np.ndarray:
    """Approximate 3D direction from normalized Euler angles (X-Y-Z convention)."""
    pitch = math.radians(denormalize_angle(pitch_norm))
    yaw = math.radians(denormalize_angle(yaw_norm))
    roll = math.radians(denormalize_angle(roll_norm))

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    dir_x = cy * cp
    dir_y = sy * cp
    dir_z = -sp

    dir_y_rot = cr * dir_y - sr * dir_z
    dir_z_rot = sr * dir_y + cr * dir_z

    vec = np.array([dir_x, dir_y_rot, dir_z_rot], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


@torch.no_grad()
def sample_reverse_diffusion_3d(model: torch.nn.Module, image: torch.Tensor,
                                camera_pose: torch.Tensor = None,
                                dap: torch.Tensor = None,
                                steps: int = 50,
                                existence_threshold: float = 0.5) -> Dict[str, Any]:
    """Run DDPM reverse diffusion using the trained x0-prediction model.

    Uses the standard DDPM posterior mean formula:
        x_{t-1} = (sqrt(alpha_{t-1}) * beta_t * x0
                   + sqrt(alpha_t) * (1 - alpha_{t-1}) * x_t) / (1 - alpha_t)
                  + sigma_t * z
    with sigma_t = sqrt(beta_t).
    """
    device = image.device
    max_nodes = model.max_nodes

    scheduler = DDPMScheduler(timesteps=1000)
    betas = scheduler.betas.to(device)
    alphas = scheduler.alphas.to(device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    step_indices = torch.linspace(999, 0, steps).long().to(device)
    step_first = int(step_indices[0].item())
    step_mid = int(step_indices[steps // 2].item())
    step_last = int(step_indices[-1].item())

    x_t = torch.randn(1, max_nodes, 15, device=device)
    e_t = torch.ones(1, max_nodes, 1, device=device)
    pose_batch = camera_pose.unsqueeze(0).to(device) if camera_pose is not None else torch.zeros(1, 2, device=device)
    dap_batch = dap.unsqueeze(0).to(device) if dap is not None else torch.zeros(1, 1, device=device)

    snapshots = {}

    for idx, t in enumerate(step_indices):
        t_batch = torch.tensor([t], device=device).long()
        outputs = model(x_t, e_t, t_batch, image.unsqueeze(0),
                        camera_poses=pose_batch, dap=dap_batch)

        pred_x0 = torch.clamp(outputs["pred_x0"], 0.0, 1.0)
        pred_exist = torch.sigmoid(outputs["pred_existence_logits"])[0].cpu().numpy()

        # Sparse parent prediction: map best candidate position to actual parent index
        parent_candidates = outputs["pred_parent_candidates"][0]
        parent_logits = outputs["pred_parent_logits"][0]
        best_k = torch.argmax(parent_logits, dim=-1)  # (N,)
        pred_parents = torch.gather(parent_candidates, 1, best_k.unsqueeze(-1)).squeeze(-1).cpu().numpy()

        if t.item() == step_last:
            x_t = pred_x0
        else:
            # DDPM posterior sampling step
            t_int = int(t.item())
            alpha_t = alphas_cumprod[t_int]
            alpha_prev = alphas_cumprod[t_int - 1] if t_int > 0 else torch.tensor(1.0, device=device)
            beta_t = betas[t_int]

            coef_x0 = torch.sqrt(alpha_prev) * beta_t / (1.0 - alpha_t)
            coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_prev) / (1.0 - alpha_t)
            sigma_t = torch.sqrt(beta_t)

            x_t = coef_x0 * pred_x0 + coef_xt * x_t + sigma_t * torch.randn_like(x_t)

        if t.item() in (step_first, step_mid, step_last):
            snap_nodes = pred_x0[0].cpu().numpy() if t.item() == step_last else x_t[0].cpu().numpy()
            snap_exist = pred_exist
            if t.item() == step_last:
                # Use DAP-based predicted budget to prune to the expected active node count.
                pred_budget_count = max(1, int(outputs["pred_node_budget"][0].item() * max_nodes))
                top_k_indices = np.argsort(-snap_exist)[:pred_budget_count]
                snap_exist = (snap_exist >= snap_exist[top_k_indices[-1]])
            snapshots[int(t.item())] = {
                "nodes": snap_nodes,
                "parent_indices": pred_parents,
                "existence_mask": snap_exist
            }

    return {
        "snapshots": snapshots,
        "step_first": step_first,
        "step_mid": step_mid,
        "step_last": step_last,
        "existence_threshold": existence_threshold,
    }


def _get_organ_type(node: np.ndarray) -> int:
    return int(np.argmax(node[8:12]))


def _leaf_polygon_3d(base: np.ndarray, direction: np.ndarray, scale: float) -> np.ndarray:
    """Generate a cordate (heart-shaped) leaf polygon in 3D."""
    leaf_len = scale * 0.40
    leaf_w = leaf_len * 0.55

    # Build local frame: direction is 'up', pick arbitrary perpendicular
    if abs(direction[2]) < 0.9:
        perp = np.cross(direction, np.array([0, 0, 1], dtype=np.float32))
    else:
        perp = np.cross(direction, np.array([0, 1, 0], dtype=np.float32))
    perp = perp / (np.linalg.norm(perp) + 1e-8)
    side = np.cross(direction, perp)
    side = side / (np.linalg.norm(side) + 1e-8)

    local_pts = [
        (0.0, 0.0),
        (-leaf_w * 0.45, leaf_len * 0.25),
        (-leaf_w * 0.50, leaf_len * 0.55),
        (0.0, leaf_len),
        (leaf_w * 0.50, leaf_len * 0.55),
        (leaf_w * 0.45, leaf_len * 0.25),
    ]

    world_pts = []
    for u_perp, v_along in local_pts:
        pt = base + v_along * direction + u_perp * perp
        world_pts.append(pt)
    return np.array(world_pts, dtype=np.float32)


def draw_3d_plant_graph(ax3d, nodes, parents, active_mask, is_gt=False):
    num_nodes = len(nodes)

    # Draw edges between parents and children for internode/petiole/floral_bud
    for v in range(num_nodes):
        if not active_mask[v]:
            continue
        organ = _get_organ_type(nodes[v])
        u = parents[v]
        if u == v or u >= num_nodes or not active_mask[u]:
            continue

        x1, y1, z1 = nodes[u, 0], nodes[u, 1], nodes[u, 2]
        x2, y2, z2 = nodes[v, 0], nodes[v, 1], nodes[v, 2]

        color = ORGAN_COLORS.get(organ, (0.5, 0.5, 0.5))
        linewidth = 3 if organ == 0 else 2
        ax3d.plot([x1, x2], [z1, z2], [1.0 - y1, 1.0 - y2],
                  color=color, linewidth=linewidth, alpha=0.9)

    # Draw leaf polygons
    for v in range(num_nodes):
        if not active_mask[v]:
            continue
        organ = _get_organ_type(nodes[v])
        if organ == 2:  # leaf
            u = parents[v]
            if u >= num_nodes or not active_mask[u]:
                continue
            base = nodes[u, :3]
            direction = direction_from_angles(nodes[v, 5], nodes[v, 6], nodes[v, 7])
            scale = float(nodes[v, 3])
            poly = _leaf_polygon_3d(base, direction, scale)

            v_x = poly[:, 0]
            v_y = 1.0 - poly[:, 1]
            v_z = poly[:, 2]
            leaf_poly = [list(zip(v_x, v_z, v_y))]
            color = ORGAN_COLORS[2]
            ax3d.add_collection3d(Poly3DCollection(leaf_poly, facecolors=color,
                                                  edgecolors='darkgreen', alpha=0.9, zorder=6))

    # Draw nodes
    organ_types = np.array([_get_organ_type(nodes[i]) for i in range(num_nodes)])
    active_types = organ_types[active_mask]
    active_coords = nodes[active_mask, :3]
    for organ in range(4):
        mask = active_types == organ
        if mask.any():
            coords = active_coords[mask]
            ax3d.scatter(coords[:, 0], coords[:, 2], 1.0 - coords[:, 1],
                         c=[ORGAN_COLORS[organ]], s=25, edgecolors='black', zorder=5)

    ax3d.set_xlim(0, 1)
    ax3d.set_ylim(0, 1)
    ax3d.set_zlim(0, 1)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Z (Depth)')
    ax3d.set_zlabel('Y (Height)')


def visualize_reconstruction_3d(image_tensor: torch.Tensor, results: Dict[str, Any],
                                gt_sample: Dict[str, Any],
                                save_path: str = "diffusion_based/plots/diffusion_sample_3d.png",
                                existence_threshold: float = 0.5):
    snapshots = results["snapshots"]
    step_first = results["step_first"]
    step_mid = results["step_mid"]
    step_last = results["step_last"]

    raw_img = gt_sample["raw_image"]
    gt_nodes = gt_sample["nodes"].cpu().numpy()
    gt_parents = gt_sample["parent_indices"].cpu().numpy()
    gt_exist = gt_sample["existence_mask"].cpu().numpy()
    gt_active = (gt_exist >= 0.5)

    img_w, img_h = raw_img.size

    fig = plt.figure(figsize=(22, 10))

    # ROW 1: 3D visualizations
    ax_gt3d = fig.add_subplot(2, 4, 1, projection='3d')
    draw_3d_plant_graph(ax_gt3d, gt_nodes, gt_parents, gt_active, is_gt=True)
    ax_gt3d.set_title("Col 1: Ground Truth 3D Target Plant", fontsize=11, fontweight='bold', color='darkgreen')

    titles_3d = {
        step_first: f"Col 2: Step {step_first} 3D Random Noise",
        step_mid: f"Col 3: Step {step_mid} 3D Denoising Assembly",
        step_last: f"Col 4: Step {step_last} 3D Reconstructed Plant"
    }

    for i, step_k in enumerate([step_first, step_mid, step_last]):
        ax3d = fig.add_subplot(2, 4, i + 2, projection='3d')
        ax3d.set_title(titles_3d[step_k], fontsize=11, fontweight='bold')

        data = snapshots[step_k]
        nodes = data["nodes"]
        parents = data.get("parent_indices", np.arange(len(nodes)))
        exist = data["existence_mask"]
        active_mask = (exist >= existence_threshold) if step_k == step_last else (exist >= 0.2)
        if not np.any(active_mask):
            active_mask[:30] = True

        draw_3d_plant_graph(ax3d, nodes, parents, active_mask, is_gt=False)

    # ROW 2: 2D projections
    ax_input2d = fig.add_subplot(2, 4, 5)
    ax_input2d.imshow(raw_img)
    ax_input2d.set_title("Col 1: Input 2D Projection Target Image", fontsize=11, fontweight='bold', color='darkblue')
    ax_input2d.axis('off')

    titles_2d = {
        step_first: f"Col 2: Step {step_first} 2D Projection Noise",
        step_mid: f"Col 3: Step {step_mid} 2D Projection Denoising",
        step_last: f"Col 4: Step {step_last} 2D Projection Reconstructed"
    }

    for i, step_k in enumerate([step_first, step_mid, step_last]):
        ax2d = fig.add_subplot(2, 4, i + 6)
        ax2d.set_xlim(0, img_w)
        ax2d.set_ylim(img_h, 0)
        ax2d.set_facecolor('#f4f6f9')
        ax2d.set_title(titles_2d[step_k], fontsize=11, fontweight='bold')
        ax2d.set_xticks([])
        ax2d.set_yticks([])

        data = snapshots[step_k]
        nodes = data["nodes"]
        parents = data.get("parent_indices", np.arange(len(nodes)))
        exist = data["existence_mask"]

        active_mask = (exist >= existence_threshold) if step_k == step_last else (exist >= 0.2)
        if not np.any(active_mask):
            active_mask[:30] = True
        num_nodes = len(nodes)

        # 2D stem/petiole edges
        for v in range(num_nodes):
            if not active_mask[v]:
                continue
            organ = _get_organ_type(nodes[v])
            u = parents[v]
            px2, py2 = nodes[v, 0] * img_w, nodes[v, 1] * img_h
            if u != v and u < num_nodes and active_mask[u] and organ != 2:
                px1, py1 = nodes[u, 0] * img_w, nodes[u, 1] * img_h
                color_2d = 'crimson' if step_k == step_first else ('orange' if step_k == step_mid else 'black')
                ax2d.plot([px1, px2], [py1, py2], color=color_2d, linewidth=3, alpha=0.9)

        # 2D leaves
        for v in range(num_nodes):
            if active_mask[v] and _get_organ_type(nodes[v]) == 2:
                u = parents[v]
                if u >= num_nodes or not active_mask[u]:
                    continue
                px_base = nodes[u, 0] * img_w
                py_base = nodes[u, 1] * img_h
                scale_area = max(0.05, float(nodes[v, 3]))

                yaw = denormalize_angle(nodes[v, 6])
                rad = math.radians(yaw)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)

                leaf_len = scale_area * 180
                leaf_w = leaf_len * 0.55

                local_pts = [
                    (0.0, 0.0),
                    (-leaf_w * 0.45, leaf_len * 0.25),
                    (-leaf_w * 0.50, leaf_len * 0.55),
                    (0.0, leaf_len),
                    (leaf_w * 0.50, leaf_len * 0.55),
                    (leaf_w * 0.45, leaf_len * 0.25),
                ]
                poly_2d = [(px_base + v_along * cos_a - u_perp * sin_a,
                            py_base + v_along * sin_a + u_perp * cos_a)
                           for u_perp, v_along in local_pts]
                poly_patch = plt.Polygon(poly_2d, facecolor='forestgreen',
                                         edgecolor='darkgreen', alpha=0.9, zorder=6)
                ax2d.add_patch(poly_patch)

        # Draw active nodes
        for organ in range(4):
            mask = active_mask & (np.array([_get_organ_type(nodes[i]) for i in range(num_nodes)]) == organ)
            if mask.any():
                ax2d.scatter(nodes[mask, 0] * img_w, nodes[mask, 1] * img_h,
                             c=[ORGAN_COLORS[organ]], s=20, edgecolors='black', zorder=5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved updated 2-Row 3D/2D plant reconstruction visualization to '{save_path}'")
    plt.close()


def run_inference_on_real_image(jpeg_path: str, xml_path: Optional[str] = None,
                                dap: int = 10,
                                checkpoint_path: str = "diffusion_based/checkpoints/diffusion_model_3d.pt",
                                save_path: str = "diffusion_based/plots/real_image_reconstruction_3d.png",
                                steps: int = 50,
                                existence_threshold: float = 0.5,
                                max_nodes: int = 2048) -> Dict[str, Any]:
    """Run diffusion inference on a real Helios JPEG + optional XML ground truth."""
    device = get_device()

    if xml_path is not None:
        gt_dataset = HeliosPlantDataset(
            data_root=os.path.dirname(xml_path),
            max_nodes=max_nodes,
        )
        # Find the sample matching the XML
        gt_sample = None
        for s in gt_dataset:
            if s["xml_path"] == xml_path:
                gt_sample = s
                break
        if gt_sample is None:
            raise ValueError(f"Could not find dataset sample for XML: {xml_path}")
    else:
        # Use a dummy synthetic sample just for raw_image / shape
        gt_dataset = Plant3DDataset(num_samples=1, max_nodes=max_nodes)
        gt_sample = gt_dataset[0]
        from PIL import Image
        gt_sample["raw_image"] = Image.open(jpeg_path).convert("RGB")

    image_tensor = gt_sample["image"].to(device)
    camera_pose = gt_sample["camera_pose"].to(device)
    dap_tensor = torch.tensor([dap / 90.0], dtype=torch.float32).to(device)

    model = PlantGraphDiffuser3D(max_nodes=max_nodes, node_dim=15).to(device)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        print(f"Loaded 3D model weights from '{checkpoint_path}'")
    else:
        print(f"Warning: no checkpoint found at '{checkpoint_path}'; using random weights")

    results = sample_reverse_diffusion_3d(
        model, image_tensor, camera_pose=camera_pose, dap=dap_tensor, steps=steps
    )
    visualize_reconstruction_3d(image_tensor, results, gt_sample=gt_sample,
                                save_path=save_path, existence_threshold=existence_threshold)
    return results


def main():
    device = get_device()
    dataset = Plant3DDataset(num_samples=10, max_nodes=2048)
    sample = dataset[0]
    image_tensor = sample["image"].to(device)

    model = PlantGraphDiffuser3D(max_nodes=2048, node_dim=15).to(device)
    checkpoint_path = "diffusion_based/checkpoints/diffusion_model_3d.pt"

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        print(f"Loaded 3D model weights from '{checkpoint_path}'")

    results = sample_reverse_diffusion_3d(
        model, image_tensor,
        camera_pose=sample["camera_pose"].to(device),
        dap=sample["dap"].to(device),
        steps=50
    )
    visualize_reconstruction_3d(image_tensor, results, gt_sample=sample,
                                  save_path="diffusion_based/plots/diffusion_sample_3d.png")


if __name__ == "__main__":
    main()
