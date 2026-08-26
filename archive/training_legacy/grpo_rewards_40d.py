"""
Composite Reward Engine for GRPO (Group Relative Policy Optimization) on 3D Plant Organ Arrays.

Evaluates:
1. Image-Space Structural Similarity (SSIM) between PyTorch forward render and target image.
2. Pixel-level Color MAE penalty.
3. Canopy Silhouette IoU.
4. Node Count Matching.
5. Structural / Mesh Connectivity Validity.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from skimage.metrics import structural_similarity as ssim

from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.plant_organ_array import PlantOrganArray


def compute_ssim_torch(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> float:
    """Compute SSIM between two (3, H, W) float tensors in [0, 1]."""
    p_np = (pred_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    g_np = (gt_rgb.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    return float(ssim(p_np, g_np, channel_axis=-1, data_range=255))


def compute_silhouette_iou(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, threshold: float = 0.05) -> float:
    """Compute 2D canopy silhouette intersection over union."""
    p_mask = (pred_rgb > threshold).any(dim=0).cpu().numpy()
    g_mask = (gt_rgb > threshold).any(dim=0).cpu().numpy()
    inter = np.logical_and(p_mask, g_mask).sum()
    union = np.logical_or(p_mask, g_mask).sum()
    return float(inter / union) if union > 0 else 0.0


class PlantGRPORewardEngine:
    """
    Computes composite scalar rewards for groups of candidate 3D plant architectures.
    """

    def __init__(
        self,
        renderer: HeliosPyTorchRenderer,
        w_ssim: float = 2.0,
        w_mae: float = 1.0,
        w_iou: float = 1.5,
        w_node: float = 0.5,
        w_validity: float = 0.5,
    ):
        self.renderer = renderer
        self.w_ssim = w_ssim
        self.w_mae = w_mae
        self.w_iou = w_iou
        self.w_node = w_node
        self.w_validity = w_validity

    @torch.no_grad()
    def evaluate_group_rewards(
        self,
        candidates_group: List[List[PlantOrganArray]],  # (B, G)
        target_images: torch.Tensor,                   # (B, 3, H, W)
        target_node_counts: Optional[torch.Tensor] = None, # (B,)
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for each candidate in each group.

        Returns:
            rewards: (B, G) tensor of total scalar rewards.
            metrics: dict of average reward components for logging.
        """
        B = len(candidates_group)
        G = len(candidates_group[0])
        rewards = torch.zeros((B, G), device=device, dtype=torch.float32)

        tot_ssim, tot_mae, tot_iou, tot_node, tot_valid = 0.0, 0.0, 0.0, 0.0, 0.0
        n_evals = B * G

        for b in range(B):
            gt_img = target_images[b]  # (3, H, W)
            target_nodes = float(target_node_counts[b].item()) if target_node_counts is not None else 30.0

            for g in range(G):
                candidate = candidates_group[b][g]
                n_active = float((candidate.existence > 0.1).sum().item())

                # 1. Render candidate through PyTorch renderer
                try:
                    pred_rgb = self.renderer.render_organ_array(
                        candidate,
                        azimuth_deg=0.0,
                        elevation_deg=90.0,
                        camera_height=1.0,
                        background="black",
                        device=device,
                        differentiable=False,
                        focus_plant=True,
                        existence_threshold=0.1,
                    )
                    r_valid = 1.0
                except Exception:
                    pred_rgb = torch.zeros_like(gt_img)
                    r_valid = -1.0

                # 2. Compute individual reward terms
                if r_valid > 0:
                    r_ssim = compute_ssim_torch(pred_rgb, gt_img)
                    r_mae = -float(torch.abs(pred_rgb - gt_img).mean().item())
                    r_iou = compute_silhouette_iou(pred_rgb, gt_img)
                else:
                    r_ssim = 0.0
                    r_mae = -1.0
                    r_iou = 0.0

                # Node count match penalty
                r_node = -abs(n_active - target_nodes) / (target_nodes + 5.0)

                # Composite reward
                total_r = (
                    self.w_ssim * r_ssim
                    + self.w_mae * r_mae
                    + self.w_iou * r_iou
                    + self.w_node * r_node
                    + self.w_validity * r_valid
                )
                rewards[b, g] = total_r

                tot_ssim += r_ssim
                tot_mae += r_mae
                tot_iou += r_iou
                tot_node += r_node
                tot_valid += r_valid

        metrics = {
            "r_total": float(rewards.mean().item()),
            "r_ssim": tot_ssim / n_evals,
            "r_mae": tot_mae / n_evals,
            "r_iou": tot_iou / n_evals,
            "r_node": tot_node / n_evals,
            "r_valid": tot_valid / n_evals,
        }
        return rewards, metrics
