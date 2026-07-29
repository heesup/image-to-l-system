import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Dict, Any

from diffusion_based.dataset.graph_dataset import PlantGraphDataset
from diffusion_based.models.graph_diffuser import PlantGraphDiffuser
from diffusion_based.training.train_diffusion import DDPMScheduler, get_device

@torch.no_grad()
def sample_reverse_diffusion(model: PlantGraphDiffuser, image: torch.Tensor, steps: int = 50) -> Dict[str, Any]:
    """Perform reverse diffusion sampling to reconstruct plant graph (V, A, e) from noise."""
    device = image.device
    model.eval()
    scheduler = DDPMScheduler(timesteps=1000)

    B = 1
    N = model.max_nodes

    # 1. Start from scattered Uniform(0, 1) noise for 5D Organ Primitives
    x_t = torch.rand(B, N, 5, device=device)
    e_t = torch.ones(B, N, 1, device=device) * 0.5

    # Reverse sampling timesteps
    step_indices = torch.linspace(999, 0, steps, device=device).long()
    snapshots = {}

    for idx, t in enumerate(step_indices):
        t_batch = torch.tensor([t], device=device).long()
        outputs = model(x_t, e_t, t_batch, image.unsqueeze(0))

        pred_x0 = outputs["pred_x0"]
        pred_parents = torch.argmax(outputs["pred_parent_logits"][0], dim=-1).cpu().numpy()
        pred_adj = torch.sigmoid(outputs["pred_adj_logits"])[0].cpu().numpy()
        pred_exist = torch.sigmoid(outputs["pred_existence_logits"])[0].cpu().numpy()

        # Save snapshot BEFORE blending for the initial noise step
        if idx == 0:
            snapshots[int(t.item())] = {
                "nodes": x_t[0].cpu().numpy(),
                "parent_indices": pred_parents,
                "adj_matrix": pred_adj,
                "existence_mask": pred_exist
            }

        # Progressive blending step towards predicted x0
        alpha = (idx + 1) / float(steps)
        x_t = (1.0 - alpha) * x_t + alpha * pred_x0

        if idx == steps // 2 or idx == steps - 1:
            snapshots[int(t.item())] = {
                "nodes": x_t[0].cpu().numpy(),
                "parent_indices": pred_parents,
                "adj_matrix": pred_adj,
                "existence_mask": pred_exist
            }

    final_snapshot = snapshots[min(snapshots.keys())]
    return {
        "snapshots": snapshots,
        "final_nodes": final_snapshot["nodes"],
        "final_parents": final_snapshot["parent_indices"],
        "final_existence": final_snapshot["existence_mask"]
    }

def visualize_reconstruction(image_tensor: torch.Tensor, results: Dict[str, Any], gt_sample: Dict[str, Any], save_path: str = "diffusion_based/plots/diffusion_sample.png"):
    """Visualize 4-panel figure with Ground Truth Tree Edges & Reconstructed Overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # Normalize image tensor for display
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    img_h, img_w, _ = img.shape

    snapshots = results["snapshots"]
    sorted_steps = sorted(snapshots.keys(), reverse=True) # e.g. [999, 489, 0]

    step_first = sorted_steps[0]
    step_mid = sorted_steps[1] if len(sorted_steps) > 1 else sorted_steps[0]
    step_last = sorted_steps[-1]

    # Panel 1: Target Image with Ground Truth Tree Graph Edges Overlay
    axes[0].imshow(img)
    axes[0].set_title("Input Image + GT Tree Graph", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    gt_nodes = gt_sample["nodes"].cpu().numpy()
    gt_parents = gt_sample["parent_indices"].cpu().numpy()
    gt_exist = gt_sample["existence_mask"].cpu().numpy()
    num_gt = len(gt_nodes)

    # 1. Panel 1: Ground Truth Overlay
    axes[0].imshow(img)
    axes[0].set_title("Panel 1: Input Target Image + GT Graph", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    gt_num_active = int((gt_exist >= 0.5).sum())
    gt_active_mask = gt_exist >= 0.5
    gt_total_len = 0.0

    for v in range(len(gt_nodes)):
        if not gt_active_mask[v]:
            continue
        u = gt_parents[v]
        px2, py2 = gt_nodes[v, 0] * img_w, gt_nodes[v, 1] * img_h
        w_val = max(1.5, min(8.0, gt_nodes[v, 4] * 25.0))

        if u != v and u < len(gt_nodes) and gt_active_mask[u]:
            px1, py1 = gt_nodes[u, 0] * img_w, gt_nodes[u, 1] * img_h
            axes[0].plot([px1, px2], [py1, py2], color='dodgerblue', linewidth=w_val, alpha=0.95, zorder=3)
            gt_total_len += math.hypot(px2 - px1, py2 - py1)

    axes[0].scatter(gt_nodes[gt_active_mask, 0] * img_w, gt_nodes[gt_active_mask, 1] * img_h, c='orange', s=35, edgecolors='black', zorder=5)

    # Debug Info Box for Panel 1
    axes[0].text(0.03, 0.05, f"Nodes: {gt_num_active} | Skeleton Length: {gt_total_len:.1f} px",
                 transform=axes[0].transAxes, fontsize=9, fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75))

    panel_titles = {
        step_first: f"Panel 2: Step {step_first} Noise",
        step_mid: f"Panel 3: Step {step_mid} Denoising",
        step_last: f"Panel 4: Step {step_last} Reconstructed"
    }

    for i, step_k in enumerate([step_first, step_mid, step_last]):
        ax = axes[i + 1]
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0) # Inverted Y to match image coordinates
        ax.set_facecolor('#f4f6f9')
        ax.set_title(panel_titles[step_k], fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        data = snapshots[step_k]
        nodes = data["nodes"]
        parents = data.get("parent_indices", np.arange(len(nodes)))
        exist = data["existence_mask"]

        active_mask = (exist >= 0.5)
        num_nodes = len(nodes)

        # Track which active nodes are actually connected to the plant tree graph
        connected_mask = np.zeros(num_nodes, dtype=bool)
        pred_total_len = 0.0

        # Draw Reconstructed Plant Graph Edges connecting parent vertex u to child vertex v
        for v in range(num_nodes):
            if not active_mask[v]:
                continue

            u = parents[v]
            px2, py2 = nodes[v, 0] * img_w, nodes[v, 1] * img_h
            w_val = max(1.5, min(8.0, nodes[v, 4] * 25.0))

            if u != v and u < num_nodes and active_mask[u]:
                px1, py1 = nodes[u, 0] * img_w, nodes[u, 1] * img_h
                line_color = 'crimson' if step_k == step_first else ('orange' if step_k == step_mid else 'lime')
                ax.plot([px1, px2], [py1, py2], color=line_color, linewidth=w_val, alpha=0.9, zorder=3)
                pred_total_len += math.hypot(px2 - px1, py2 - py1)
                connected_mask[v] = True
                connected_mask[u] = True

        # Node 0 is the root of the tree
        if active_mask[0]:
            connected_mask[0] = True

        floating_mask = active_mask & (~connected_mask)
        num_connected = int(connected_mask.sum())

        # Render Connected Plant Junction Nodes in Bright Yellow, and Floating Unconnected Nodes in Faded Grey-Yellow
        if np.any(connected_mask):
            ax.scatter(nodes[connected_mask, 0] * img_w, nodes[connected_mask, 1] * img_h, c='yellow', s=35, edgecolors='black', zorder=5, label='Connected Nodes')
        if np.any(floating_mask):
            ax.scatter(nodes[floating_mask, 0] * img_w, nodes[floating_mask, 1] * img_h, c='lightgray', s=20, edgecolors='gray', alpha=0.6, zorder=4, label='Floating Nodes')

        # Debug Info Box for Panels 2, 3, 4
        ax.text(0.03, 0.05, f"Connected Nodes: {num_connected} | Skeleton Length: {pred_total_len:.1f} px",
                transform=ax.transAxes, fontsize=9, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.85))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved plant reconstruction overlay visualization to '{save_path}'")
    plt.close()

def main():
    device = get_device()
    dataset = PlantGraphDataset(num_synthetic_samples=10)
    
    model = PlantGraphDiffuser(max_nodes=64).to(device)
    checkpoint_path = "diffusion_based/checkpoints/diffusion_model.pt"

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model weights from '{checkpoint_path}'")
    else:
        print("Warning: No checkpoint found, using initial model weights.")

    for i in range(4):
        sample = dataset[i]
        image_tensor = sample["image"].to(device)
        results = sample_reverse_diffusion(model, image_tensor, steps=50)
        save_path = f"diffusion_based/plots/diffusion_sample_{i+1}.png"
        visualize_reconstruction(image_tensor, results, gt_sample=sample, save_path=save_path)
        if i == 0:
            visualize_reconstruction(image_tensor, results, gt_sample=sample, save_path="diffusion_based/plots/diffusion_sample.png")

if __name__ == "__main__":
    main()
