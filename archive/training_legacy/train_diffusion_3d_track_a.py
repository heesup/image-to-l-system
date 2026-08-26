"""Training script for 3D Plant Graph Diffusion with 25D organ nodes.

Trains PlantGraphDiffuser3D with:
- Helios dataset (image + XML pairs)
- 25D organ-typed node representation (position, length, radius, 3x3 R matrix,
  6-class organ one-hot, shoot_id, phytomer_idx, existence, head_radius, parent_idx)
- Multi-view camera conditioning
- Optional 2D differentiable renderer loss using DifferentiableHeliosRenderer
- Optional 3D point-cloud Chamfer loss against a target PLY

The 15D legacy mode is still available via ``--node-dim 15`` for backward
compatibility, but defaults to 25D.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from diffusion_based.models.legacy.graph_diffuser_3d_track_a import PlantGraphDiffuser3D
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer
from diffusion_based.models.legacy.pointcloud_loss_3d_track_a import PlantPointCloudChamferLoss, load_ply_to_tensor
from diffusion_based.models import helios_geometry
from dataset.helios_dataset import HeliosPlantDataset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class DDPMScheduler:
    """Linear DDPM Noise Scheduler."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x0.device
        noise = torch.randn_like(x0)

        alphas_cumprod = self.alphas_cumprod.to(device)
        sqrt_alpha = torch.sqrt(alphas_cumprod[timesteps])[:, None, None]
        sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod[timesteps])[:, None, None]

        xt = sqrt_alpha * x0 + sqrt_one_minus * noise
        return xt, noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device).long()


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    gt_nodes: torch.Tensor,
    gt_existence: torch.Tensor,
    gt_parents: torch.Tensor,
    gt_adj: torch.Tensor,
    noisy_nodes: torch.Tensor,
    node_dim: int = 25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute multi-objective training losses for 25D/15D organ graph diffusion."""

    pred_x0 = outputs["pred_x0"]
    pred_existence_logits = outputs["pred_existence_logits"]
    pred_parent_logits = outputs["pred_parent_logits"]
    pred_parent_candidates = outputs["pred_parent_candidates"]
    pred_organ_type_logits = outputs.get("pred_organ_type_logits", None)
    pred_node_noise = outputs["pred_node_noise"]

    B, N, D = pred_x0.shape
    device = pred_x0.device

    # Existence-aware mask: focus loss on real nodes
    existence_mask = (gt_existence > 0).float()  # (B, N)

    # 1. Node coordinate MSE (3D position accuracy) — masked
    loss_coord = _masked_mse(pred_x0[:, :, :3], gt_nodes[:, :, :3], existence_mask)

    # 2. Full node attribute MSE — masked
    loss_x0 = _masked_mse(pred_x0, gt_nodes, existence_mask)

    # 3. Existence confidence BCE
    existence_target = existence_mask
    pos_weight = torch.tensor([5.0], device=device)
    loss_existence = F.binary_cross_entropy_with_logits(
        pred_existence_logits, existence_target, pos_weight=pos_weight
    )

    # 4. Parent cross-entropy over sparse k-NN candidates
    k = pred_parent_candidates.shape[-1]
    parent_candidates = pred_parent_candidates  # (B, N, k)
    gt_parents_exp = gt_parents.unsqueeze(-1).expand(-1, -1, k)
    candidate_match = (parent_candidates == gt_parents_exp)  # (B, N, k)
    target_idx = torch.argmax(candidate_match.float(), dim=-1)  # (B, N)
    valid = candidate_match.any(dim=-1) & (existence_mask > 0.5)  # (B, N)

    if valid.sum() > 0:
        loss_parent_all = F.cross_entropy(
            pred_parent_logits.view(-1, k),
            target_idx.view(-1),
            reduction='none'
        )
        loss_parent = (loss_parent_all * valid.view(-1).float()).sum() / (valid.sum().float() + 1e-8)
    else:
        loss_parent = torch.tensor(0.0, device=device)

    # 5. Joint Snap Loss (parent-to-child position agreement) — masked
    # The diffusion model outputs base positions; the child base should be near
    # the parent base plus an organ direction scaled by length. For 25D we use
    # the predicted R matrix midrib axis (column 0) as the direction for all organ
    # types. For 15D we fall back to pitch/yaw reconstruction.
    base = pred_x0[:, :, :3]
    length = pred_x0[:, :, 3]
    if node_dim >= 25:
        R = pred_x0[:, :, 5:14].reshape(B, N, 3, 3)
        direction = R[:, :, :, 0]  # (B, N, 3)
    else:
        pitch_rad = pred_x0[:, :, 5] * math.pi / 180.0
        yaw_rad = pred_x0[:, :, 6] * math.pi / 180.0
        dir_x = torch.cos(pitch_rad) * torch.cos(yaw_rad)
        dir_y = torch.cos(pitch_rad) * torch.sin(yaw_rad)
        dir_z = torch.sin(pitch_rad)
        direction = torch.stack([dir_x, dir_y, dir_z], dim=-1)

    tip = base + length.unsqueeze(-1) * direction
    diff = tip.unsqueeze(2) - base.unsqueeze(1)  # (B, N, N, 3)
    dist_sq = (diff ** 2).sum(dim=-1)
    loss_snap = (dist_sq * gt_adj).sum() / (gt_adj.sum() + 1e-5)

    # 6. Organ type classification (optional)
    loss_organ_type = torch.tensor(0.0, device=device)
    if pred_organ_type_logits is not None:
        organ_onehot_start = 14 if node_dim >= 25 else 8
        organ_onehot_end = 20 if node_dim >= 25 else 12
        organ_target = torch.argmax(gt_nodes[:, :, organ_onehot_start:organ_onehot_end], dim=-1)
        loss_organ_type = F.cross_entropy(
            pred_organ_type_logits.view(-1, 6),
            organ_target.view(-1),
            reduction='mean'
        )

    # 7. Rotation-matrix loss for 25D nodes
    loss_rot = torch.tensor(0.0, device=device)
    if node_dim >= 25:
        R_pred = pred_x0[:, :, 5:14].reshape(B, N, 3, 3)
        R_gt = gt_nodes[:, :, 5:14].reshape(B, N, 3, 3)
        # Element-wise MSE on the flattened rotation matrix, masked by existence
        loss_rot = _masked_mse(R_pred.reshape(B, N, 9), R_gt.reshape(B, N, 9), existence_mask)

    # 8. Noise prediction loss (for DDPM training) — masked
    gt_noise = noisy_nodes - gt_nodes
    loss_noise = _masked_mse(pred_node_noise, gt_noise, existence_mask)

    # Combined weighted loss
    loss = (
        10.0 * loss_coord +
        loss_x0 +
        0.5 * loss_existence +
        0.5 * loss_parent +
        0.2 * loss_snap +
        0.2 * loss_organ_type +
        loss_noise
    )
    if node_dim >= 25:
        loss = loss + loss_rot

    metrics = {
        "coord": loss_coord.item(),
        "x0": loss_x0.item(),
        "existence": loss_existence.item(),
        "parent": loss_parent.item(),
        "snap": loss_snap.item(),
        "organ": loss_organ_type.item(),
        "noise": loss_noise.item(),
    }
    if node_dim >= 25:
        metrics["rot"] = loss_rot.item()

    return loss, metrics


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss over the last dim, averaged only over active nodes (mask > 0.5)."""
    sq = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
    active = (mask > 0.5).float()
    if active.sum() > 0:
        return (sq * active).sum() / active.sum()
    return sq.mean()


def train_3d_diffusion(
    data_dir: str = "Digital-Crops/projects/syntheticdata_generation/build/output",
    num_epochs: int = 500,
    batch_size: int = 1,
    lr: float = 3e-4,
    save_path: str = "diffusion_based/checkpoints/diffusion_3d_25d.pt",
    node_dim: int = 25,
    max_nodes: int = 2048,
    render_loss_weight: float = 0.0,  # Set > 0 to enable 2D render-in-the-loop
    render_fast_mode: bool = True,     # Lower leaf subdivisions for tractable render loss
    pc_loss_weight: float = 0.0,       # Set > 0 to enable 3D point-cloud loss
    target_ply: Optional[str] = None,  # Path to target PLY for 3D supervision
    pc_samples: int = 1024,            # Number of target points to use per batch
):
    device = get_device()
    print(f"Training {node_dim}D 3D Plant Diffusion on device: {device}")

    if node_dim not in (15, 25):
        raise ValueError(f"--node-dim must be 15 or 25, got {node_dim}")

    # Dataset
    dataset = HeliosPlantDataset(
        data_root=data_dir,
        image_size=256,
        max_nodes=max_nodes,
        node_dim=node_dim,
    )

    if len(dataset) == 0:
        print(f"ERROR: No samples found in {data_dir}")
        print("Please run dataset generation first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset size: {len(dataset)} samples")

    # Model
    model = PlantGraphDiffuser3D(
        max_nodes=max_nodes,
        node_dim=node_dim,
        embed_dim=256,
        num_layers=4,
        k_nearest=16,
    ).to(device)

    # Differentiable renderer (Track B pipeline)
    renderer = None
    if render_loss_weight > 0:
        rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
        renderer = DifferentiableHeliosRenderer(rasterizer).to(device)
        # Enable fast leaf rendering to keep full-plant render loss tractable
        if render_fast_mode:
            helios_geometry.nodes_to_geometry_torch._fast_render_mode = True
        print(f"2D render-in-the-loop enabled (weight={render_loss_weight}, fast_mode={render_fast_mode})")

    # 3D point-cloud loss
    pc_loss_fn = None
    if pc_loss_weight > 0:
        if target_ply is None or not os.path.exists(target_ply):
            print(f"ERROR: target_ply required for 3D loss, got: {target_ply}")
            return
        pc_loss_fn = PlantPointCloudChamferLoss(
            n_cylinder_circ=6,
            n_cylinder_axis=3,
            n_leaf_u=5,
            n_leaf_v=8,
            organ_weights=(1.0, 1.0, 1.5, 2.0),
        ).to(device)
        # Pre-load target point cloud once and expand to batch size
        target_pc_full = load_ply_to_tensor(target_ply, opacity_threshold=0.5, device=device)
        print(f"3D point-cloud loss enabled (weight={pc_loss_weight}, target={target_pc_full.shape[1]} pts)")

    scheduler = DDPMScheduler(timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Training loop
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in dataloader:
            B = batch["nodes"].shape[0]
            images = batch["image"].to(device)
            gt_nodes = batch["nodes"].to(device)
            gt_existence = batch["existence_mask"].to(device)
            gt_adj = batch["adj_matrix"].to(device)
            gt_parents = batch["parent_indices"].to(device)
            cam_az_norm = batch["camera_pose"][:, 0].to(device)
            dap = batch["dap"].to(device)

            xyz_min = batch.get("xyz_min")
            xyz_scale = batch.get("xyz_scale")
            if xyz_min is not None:
                xyz_min = xyz_min.to(device)
                xyz_scale = xyz_scale.to(device)

            # Sample timesteps and add noise
            timesteps = scheduler.sample_timesteps(B, device)
            noisy_nodes, noise = scheduler.add_noise(gt_nodes, timesteps)

            # Camera pose: [azimuth_norm, elevation_norm]
            camera_pose = torch.stack([
                cam_az_norm,
                torch.full((B,), 0.0, device=device),  # elevation = 0 after normalization
            ], dim=1)

            # Forward pass
            noisy_existence = gt_existence.unsqueeze(-1)
            outputs = model(
                noisy_nodes, noisy_existence, timesteps, images,
                camera_poses=camera_pose,
                dap=dap.unsqueeze(-1)
            )

            # Compute losses
            loss, metrics = compute_losses(
                outputs, gt_nodes, gt_existence, gt_parents, gt_adj,
                noisy_nodes, node_dim=node_dim,
            )

            # Optional 2D render loss (Track B differentiable renderer)
            if renderer is not None and render_loss_weight > 0:
                pred_nodes = outputs["pred_x0"]
                pred_existence = torch.sigmoid(outputs["pred_existence_logits"])

                # Denormalize positions/scale/radius for rendering if needed
                render_nodes = pred_nodes.clone()
                if node_dim >= 25 and xyz_min is not None and xyz_scale is not None:
                    render_nodes[:, :, :3] = render_nodes[:, :, :3] * xyz_scale.unsqueeze(1) + xyz_min.unsqueeze(1)
                    render_nodes[:, :, 3] = render_nodes[:, :, 3] * 1.0
                    render_nodes[:, :, 4] = render_nodes[:, :, 4] * 0.1

                cam_az_deg = (cam_az_norm + 1.0) * 180.0
                sun_dir = torch.tensor([[0.0, 0.0, 1.0]], device=device)

                rendered_list = []
                for b in range(B):
                    # Optionally mask inactive nodes by setting existence to 0
                    rimg_t = renderer(
                        render_nodes[b:b+1],
                        camera_height=1.0,
                        distance_from_center=0.0,
                        azimuth_deg=cam_az_deg[b].item(),
                        focus_plant=True,
                        sun_dir=sun_dir,
                    )
                    rendered_list.append(rimg_t[0])
                rendered = torch.stack(rendered_list, dim=0).to(device)
                rendered = rendered[:, :3, :, :]  # drop alpha channel

                # Denormalize target image back to [0, 1]
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                target_rgb = images * std + mean
                target_rgb = torch.clamp(target_rgb, 0.0, 1.0)

                loss_render = F.mse_loss(rendered, target_rgb)
                loss = loss + render_loss_weight * loss_render
                metrics["render"] = loss_render.item()

            # Optional 3D point-cloud loss
            if pc_loss_fn is not None and pc_loss_weight > 0:
                pred_nodes = outputs["pred_x0"]
                # Replicate target point cloud for the whole batch
                target_pc_batch = target_pc_full.expand(B, -1, -1)
                # Optionally subsample target points per batch for memory
                K = target_pc_batch.shape[1]
                if pc_samples < K:
                    rand_idx = torch.randperm(K, device=device)[:pc_samples]
                    target_pc_batch = target_pc_batch[:, rand_idx]

                loss_pc, pc_info = pc_loss_fn(pred_nodes, target_pc_batch, parent_indices=gt_parents)
                loss = loss + pc_loss_weight * loss_pc
                metrics["pc"] = loss_pc.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

        lr_scheduler.step()

        # Logging
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if epoch % 10 == 0 or epoch == 1:
            log_msg = (f"Epoch [{epoch:03d}/{num_epochs}] - Loss: {avg_loss:.4f} "
                       f"Coord={metrics['coord']:.4f} X0={metrics['x0']:.4f} "
                       f"Exist={metrics['existence']:.4f} Parent={metrics['parent']:.4f} "
                       f"Snap={metrics['snap']:.4f} Organ={metrics['organ']:.4f} "
                       f"Noise={metrics['noise']:.4f}")
            if "rot" in metrics:
                log_msg += f" Rot={metrics['rot']:.4f}"
            if "render" in metrics:
                log_msg += f" Render={metrics['render']:.4f}"
            if "pc" in metrics:
                log_msg += f" PC={metrics['pc']:.4f}"
            print(log_msg)

        # Save checkpoint
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path.replace('.pt', f'_epoch{epoch}.pt'))

    # Final save
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, save_path)
    print(f"Saved final model to '{save_path}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="Digital-Crops/projects/syntheticdata_generation/build/output")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", type=str,
                        default="diffusion_based/checkpoints/diffusion_3d_25d.pt")
    parser.add_argument("--node-dim", type=int, default=25,
                        choices=[15, 25],
                        help="Node feature dimension: 15 (legacy) or 25 (full R-matrix)")
    parser.add_argument("--max-nodes", type=int, default=2048,
                        help="Maximum number of nodes per plant")
    parser.add_argument("--render-loss", type=float, default=0.0,
                        help="Enable 2D render-in-the-loop loss weight")
    parser.add_argument("--render-fast-mode", action="store_true", default=True,
                        help="Use lower leaf subdivisions for tractable render loss")
    parser.add_argument("--no-render-fast-mode", dest="render_fast_mode", action="store_false",
                        help="Disable fast render mode (slower, higher quality)")
    parser.add_argument("--pc-loss", type=float, default=0.0,
                        help="Enable 3D point-cloud Chamfer loss weight")
    parser.add_argument("--target-ply", type=str, default=None,
                        help="Path to target PLY for 3D point-cloud supervision")
    parser.add_argument("--pc-samples", type=int, default=1024,
                        help="Number of target points to use per batch")
    args = parser.parse_args()

    train_3d_diffusion(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        node_dim=args.node_dim,
        max_nodes=args.max_nodes,
        render_loss_weight=args.render_loss,
        render_fast_mode=args.render_fast_mode,
        pc_loss_weight=args.pc_loss,
        target_ply=args.target_ply,
        pc_samples=args.pc_samples,
    )
