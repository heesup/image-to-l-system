import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Dict, Any
from dataset.plant3d_dataset import Plant3DDataset
from diffusion_based.models.graph_diffuser_3d import PlantGraphDiffuser3D
from diffusion_based.training.train_diffusion import DDPMScheduler, get_device

@torch.no_grad()
def sample_reverse_diffusion_3d(model: PlantGraphDiffuser3D, image: torch.Tensor, steps: int = 50) -> Dict[str, Any]:
    device = image.device
    model.eval()
    scheduler = DDPMScheduler(timesteps=1000)

    B = 1
    N = model.max_nodes
    x_t = torch.randn((B, N, 7), device=device)
    e_t = torch.zeros((B, N, 1), device=device)

    step_indices = np.linspace(999, 0, steps, dtype=int)
    snapshots = {}

    step_first = int(step_indices[0])
    step_mid = int(step_indices[steps // 2])
    step_last = int(step_indices[-1])

    for idx, t in enumerate(step_indices):
        t_batch = torch.tensor([t], device=device).long()
        outputs = model(x_t, e_t, t_batch, image.unsqueeze(0))

        pred_x0 = outputs["pred_x0"]
        pred_parents = torch.argmax(outputs["pred_parent_logits"][0], dim=-1).cpu().numpy()
        pred_exist = torch.sigmoid(outputs["pred_existence_logits"])[0].cpu().numpy()

        alpha = (idx + 1) / float(steps)
        x_t = (1.0 - alpha) * x_t + alpha * pred_x0

        if t.item() in (step_first, step_mid, step_last):
            snapshots[int(t.item())] = {
                "nodes": pred_x0[0].cpu().numpy() if idx == steps - 1 else x_t[0].cpu().numpy(),
                "parent_indices": pred_parents,
                "existence_mask": pred_exist
            }

    return {
        "snapshots": snapshots,
        "step_first": step_first,
        "step_mid": step_mid,
        "step_last": step_last
    }

def draw_3d_plant_graph(ax3d, nodes, parents, active_mask, is_gt=False):
    num_nodes = len(nodes)

    # 1. Draw 3D Stem Edges connecting parent u to child v
    for v in range(num_nodes):
        if not active_mask[v]:
            continue
        u = parents[v]
        x2, y2, z2 = nodes[v, 0], nodes[v, 1], nodes[v, 2]
        is_leaf = (nodes[v, 6] > 0.5)

        if not is_leaf and u != v and u < num_nodes and active_mask[u]:
            x1, y1, z1 = nodes[u, 0], nodes[u, 1], nodes[u, 2]
            color_3d = 'darkgreen' if is_gt else 'black'
            ax3d.plot([x1, x2], [z1, z2], [1.0 - y1, 1.0 - y2], color=color_3d, linewidth=4, alpha=0.9)

    # 2. Draw Stem Junction Nodes
    stems = active_mask & (nodes[:, 6] <= 0.5)
    if np.any(stems):
        ax3d.scatter(nodes[stems, 0], nodes[stems, 2], 1.0 - nodes[stems, 1], c='yellow', s=35, edgecolors='black', zorder=5)

    # 3. Draw 3D Elongated Heart Leaf Polygons (Attached EXACTLY at Node Joint)
    for v in range(num_nodes):
        if active_mask[v] and nodes[v, 6] > 0.5: # 3D Leaf
            u = parents[v]
            # Attached EXACTLY at parent node joint coordinate
            x_b, y_b, z_b = nodes[u, 0], nodes[u, 1], nodes[u, 2]
            scale_area = nodes[v, 5]

            # Direction angle
            cos_val = (nodes[v, 3] - 0.5) * 2.0
            sin_val = (nodes[v, 4] - 0.5) * 2.0
            leaf_angle_deg = math.degrees(math.atan2(sin_val, cos_val))
            rad = math.radians(leaf_angle_deg)

            leaf_len = scale_area * 0.40
            leaf_w = leaf_len * 0.55

            cos_a, sin_a = math.cos(rad), math.sin(rad)
            local_pts = [
                (0, 0),
                (-leaf_w * 0.45, -leaf_len * 0.25),
                (-leaf_w * 0.50, -leaf_len * 0.55),
                (0, -leaf_len),
                (leaf_w * 0.50, -leaf_len * 0.55),
                (leaf_w * 0.45, -leaf_len * 0.25),
            ]

            v_x = [x_b + lx * cos_a for lx, ly in local_pts]
            v_y = [1.0 - (y_b + ly) for lx, ly in local_pts]
            v_z = [z_b + lx * sin_a for lx, ly in local_pts]

            leaf_poly = [list(zip(v_x, v_z, v_y))]
            color_leaf = 'seagreen' if is_gt else 'forestgreen'
            ax3d.add_collection3d(Poly3DCollection(leaf_poly, facecolors=color_leaf, edgecolors='darkgreen', alpha=0.95, zorder=6))

    ax3d.set_xlim(0, 1); ax3d.set_ylim(0, 1); ax3d.set_zlim(0, 1)
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Z (Depth)'); ax3d.set_zlabel('Y (Height)')

def visualize_reconstruction_3d(image_tensor: torch.Tensor, results: Dict[str, Any], gt_sample: Dict[str, Any], save_path: str = "diffusion_based/plots/diffusion_sample_3d.png"):
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

    # --- ROW 1: 3D Perspective Visualizations ---
    # Col 1, Row 1: Ground Truth 3D Target Plant
    ax_gt3d = fig.add_subplot(2, 4, 1, projection='3d')
    draw_3d_plant_graph(ax_gt3d, gt_nodes, gt_parents, gt_active, is_gt=True)
    ax_gt3d.set_title("Col 1: Ground Truth 3D Target Plant", fontsize=11, fontweight='bold', color='darkgreen')

    # Col 2..4, Row 1: 3D Steps (Noise, Denoising, Reconstructed)
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
        active_mask = (exist >= 0.5) if step_k == step_last else (exist >= 0.2)
        if not np.any(active_mask):
            active_mask[:13] = True

        draw_3d_plant_graph(ax3d, nodes, parents, active_mask, is_gt=False)

    # --- ROW 2: 2D Projection Visualizations ---
    # Col 1, Row 2: 2D Input Target Projection Image
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
        ax2d.set_xticks([]); ax2d.set_yticks([])

        data = snapshots[step_k]
        nodes = data["nodes"]
        parents = data.get("parent_indices", np.arange(len(nodes)))
        exist = data["existence_mask"]

        active_mask = (exist >= 0.5) if step_k == step_last else (exist >= 0.2)
        if not np.any(active_mask):
            active_mask[:13] = True
        num_nodes = len(nodes)

        # Draw 2D projected stem edges
        for v in range(num_nodes):
            if not active_mask[v]:
                continue
            u = parents[v]
            px2, py2 = nodes[v, 0] * img_w, nodes[v, 1] * img_h
            is_leaf = (nodes[v, 6] > 0.5)

            if not is_leaf and u != v and u < num_nodes and active_mask[u]:
                px1, py1 = nodes[u, 0] * img_w, nodes[u, 1] * img_h
                color_2d = 'crimson' if step_k == step_first else ('orange' if step_k == step_mid else 'black')
                ax2d.plot([px1, px2], [py1, py2], color=color_2d, linewidth=4, alpha=0.9)

        # Draw 2D projected Elongated Heart Leaf Polygons (Attached EXACTLY at Node Joint)
        for v in range(num_nodes):
            if active_mask[v] and nodes[v, 6] > 0.5: # 2D Leaf
                u = parents[v]
                px_base = nodes[u, 0] * img_w
                py_base = nodes[u, 1] * img_h
                scale_area = nodes[v, 5]

                cos_val = (nodes[v, 3] - 0.5) * 2.0
                sin_val = (nodes[v, 4] - 0.5) * 2.0
                leaf_angle_deg = math.degrees(math.atan2(sin_val, cos_val))

                leaf_len = scale_area * 180
                leaf_w = leaf_len * 0.55
                rad = math.radians(leaf_angle_deg)
                cos_a, sin_a = math.cos(rad), math.sin(rad)

                local_pts = [
                    (0, 0),
                    (-leaf_w * 0.45, -leaf_len * 0.25),
                    (-leaf_w * 0.50, -leaf_len * 0.55),
                    (0, -leaf_len),
                    (leaf_w * 0.50, -leaf_len * 0.55),
                    (leaf_w * 0.45, -leaf_len * 0.25)
                ]
                poly_2d = [(px_base + lx * cos_a - ly * sin_a, py_base + lx * sin_a + ly * cos_a) for lx, ly in local_pts]
                poly_patch = plt.Polygon(poly_2d, facecolor='forestgreen', edgecolor='darkgreen', alpha=0.9, zorder=6)
                ax2d.add_patch(poly_patch)

        stems = active_mask & (nodes[:, 6] <= 0.5)
        if np.any(stems):
            ax2d.scatter(nodes[stems, 0] * img_w, nodes[stems, 1] * img_h, c='yellow', s=35, edgecolors='black', zorder=5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved updated 2-Row 3D/2D plant reconstruction visualization to '{save_path}'")
    plt.close()

def main():
    device = get_device()
    dataset = Plant3DDataset(num_samples=10)
    sample = dataset[0]
    image_tensor = sample["image"].to(device)

    model = PlantGraphDiffuser3D(max_nodes=64).to(device)
    checkpoint_path = "diffusion_based/checkpoints/diffusion_model_3d.pt"

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded 3D model weights from '{checkpoint_path}'")

    results = sample_reverse_diffusion_3d(model, image_tensor, steps=50)
    visualize_reconstruction_3d(image_tensor, results, gt_sample=sample, save_path="diffusion_based/plots/diffusion_sample_3d.png")

if __name__ == "__main__":
    main()
