import os
import math
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, default_collate


# Keys required by the training loop. Other keys (e.g., raw_image, xml_path) are dropped.
_TRAIN_KEYS = {
    "image", "nodes", "adj_matrix", "parent_indices", "existence_mask",
    "organ_types", "num_nodes", "camera_pose", "dap",
}


def collate_training_batch(batch):
    """Collate only training-relevant keys and drop everything else."""
    filtered = []
    for item in batch:
        filtered.append({k: item[k] for k in _TRAIN_KEYS if k in item})
    return default_collate(filtered)

from dataset.plant3d_dataset import Plant3DDataset
from dataset.helios_dataset import HeliosPlantDataset
from diffusion_based.models.graph_diffuser_3d import PlantGraphDiffuser3D
from diffusion_based.training.train_diffusion import DDPMScheduler, get_device


def denormalize_angle(norm_val: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] back to degrees in [-180, 180]."""
    return (norm_val - 0.5) * 360.0


def compute_snap_loss_3d(pred_x0: torch.Tensor, parent_indices: torch.Tensor,
                          adj_matrix: torch.Tensor, existence: torch.Tensor) -> torch.Tensor:
    """3D joint snap loss: parent tip should be close to child base.

    Approximate tip = base + length * direction_vector(pitch, yaw, roll).
    Uses the X-Y-Z (roll-pitch-yaw) convention.
    """
    B, N, _ = pred_x0.shape
    device = pred_x0.device

    base = pred_x0[:, :, :3]
    length = pred_x0[:, :, 3:4]

    pitch = denormalize_angle(pred_x0[:, :, 5])
    yaw = denormalize_angle(pred_x0[:, :, 6])
    roll = denormalize_angle(pred_x0[:, :, 7])

    # Direction vector from Euler XYZ (degrees)
    cr, sr = torch.cos(torch.deg2rad(roll)), torch.sin(torch.deg2rad(roll))
    cp, sp = torch.cos(torch.deg2rad(pitch)), torch.sin(torch.deg2rad(pitch))
    cy, sy = torch.cos(torch.deg2rad(yaw)), torch.sin(torch.deg2rad(yaw))

    # Unit vector after roll(X), pitch(Y), yaw(Z)
    dir_x = cy * cp
    dir_y = sy * cp
    dir_z = -sp

    # Apply roll around the local x-axis: this rotates (y,z) plane
    dir_y_rot = cr * dir_y - sr * dir_z
    dir_z_rot = sr * dir_y + cr * dir_z

    direction = torch.stack([dir_x, dir_y_rot, dir_z_rot], dim=-1)  # (B, N, 3)
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

    tip = base + length * direction

    # parent tip -> child base distance for active edges
    parent_tip = torch.gather(tip, 1, parent_indices.unsqueeze(-1).expand(-1, -1, 3))
    diff = base - parent_tip
    dist_sq = (diff ** 2).sum(dim=-1)  # (B, N)

    # Only penalize active edges and existing nodes
    active_edges = adj_matrix.sum(dim=-1)  # (B, N)
    mask = (active_edges > 0.5).float() * (existence.squeeze(-1) > 0.5).float()
    loss = (dist_sq * mask).sum() / (mask.sum() + 1e-5)
    return loss


def build_curriculum_sampler(synthetic_dataset, helios_dataset, synthetic_ratio: float,
                              batch_size: int) -> Optional[DataLoader]:
    """Build a DataLoader with weighted sampling mixing two datasets."""
    if helios_dataset is None or len(helios_dataset) == 0:
        return DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, drop_last=True, collate_fn=collate_training_batch)

    combined = ConcatDataset([synthetic_dataset, helios_dataset])
    n_syn = len(synthetic_dataset)
    n_hel = len(helios_dataset)

    # sampling weight per sample
    weights = [synthetic_ratio / n_syn] * n_syn + [(1.0 - synthetic_ratio) / n_hel] * n_hel
    sampler = WeightedRandomSampler(weights, num_samples=max(n_syn, n_hel) * 2, replacement=True)
    return DataLoader(combined, batch_size=batch_size, sampler=sampler,
                      num_workers=0, drop_last=True, collate_fn=collate_training_batch)


def train_diffusion_3d(
    num_samples: int = 100,
    epochs: int = 500,
    lr: float = 3e-4,
    batch_size: int = 4,
    max_nodes: int = 2048,
    helios_data_root: Optional[str] = None,
    save_path: str = "diffusion_based/checkpoints/diffusion_model_3d.pt",
    best_save_path: str = "diffusion_based/checkpoints/best_3d_model.pt",
    freeze_pretrained: Optional[str] = None,
    pretrain_existence_epochs: int = 0,
):
    device = get_device()
    print(f"--- Training 3D Botanical Plant Diffusion Model (2D Image -> 3D Plant Graph) on device: {device} ---")

    # Datasets
    synthetic_dataset = Plant3DDataset(num_samples=num_samples, max_nodes=max_nodes, fixed_seed=True)
    helios_dataset = None
    if helios_data_root and os.path.exists(helios_data_root):
        helios_dataset = HeliosPlantDataset(data_root=helios_data_root, max_nodes=max_nodes)
        print(f"Loaded Helios dataset: {len(helios_dataset)} samples")
    else:
        print("No Helios dataset provided; training on synthetic data only")

    # Train / validation split: keep synthetic always in train; split Helios 80/20 if available
    val_dataset = None
    if helios_dataset is not None and len(helios_dataset) >= 5:
        n_val = max(1, int(0.2 * len(helios_dataset)))
        n_train = len(helios_dataset) - n_val
        helios_train, helios_val = torch.utils.data.random_split(
            helios_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        helios_dataset = helios_train
        val_dataset = helios_val
        print(f"Helios split: train={len(helios_dataset)}, val={len(val_dataset)}")

    scheduler = DDPMScheduler(timesteps=1000)
    model = PlantGraphDiffuser3D(max_nodes=max_nodes, node_dim=15).to(device)

    if freeze_pretrained is not None and os.path.exists(freeze_pretrained):
        state = torch.load(freeze_pretrained, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from '{freeze_pretrained}'")
        # Freeze everything except the existence and budget heads.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.existence_pred_head.parameters():
            p.requires_grad = True
        for p in model.node_budget_head.parameters():
            p.requires_grad = True
        print("Frozen backbone; training existence + budget heads only.")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float("inf")

    # Optional existence-only pre-training phase
    pretrain_end = pretrain_existence_epochs

    for epoch in range(1, epochs + 1):
        # Curriculum mixing ratio: Helios-heavy from the start, gradually phase out synthetic.
        if epoch <= 50:
            syn_ratio = 0.25
        elif epoch <= 150:
            syn_ratio = 0.10
        else:
            syn_ratio = 0.0

        loader = build_curriculum_sampler(synthetic_dataset, helios_dataset, syn_ratio, batch_size)

        model.train()
        epoch_losses = []
        for batch in loader:
            images = batch["image"].to(device)
            gt_nodes = batch["nodes"].to(device)
            gt_adj = batch["adj_matrix"].to(device)
            gt_parents = batch["parent_indices"].to(device)
            gt_existence = batch["existence_mask"].unsqueeze(-1).to(device)
            gt_poses = batch["camera_pose"].to(device)
            gt_dap = batch["dap"].to(device)
            gt_organ_types = batch.get("organ_types", gt_nodes[:, :, 8:12].argmax(dim=-1)).to(device)

            B, N, _ = gt_nodes.shape
            timesteps = torch.randint(0, 1000, (B,), device=device).long()
            noisy_nodes, noise = scheduler.add_noise(gt_nodes, timesteps)

            outputs = model(noisy_nodes, gt_existence, timesteps, images,
                            camera_poses=gt_poses, dap=gt_dap)

            pred_x0 = outputs["pred_x0"]
            pred_organ_logits = outputs["pred_organ_type_logits"]
            pred_exist_logits = outputs["pred_existence_logits"]
            pred_parent_logits = outputs["pred_parent_logits"]

            # 3D coordinate and geometry attribute losses
            loss_coord = F.mse_loss(pred_x0[:, :, 0:3], gt_nodes[:, :, 0:3])
            loss_length = F.mse_loss(pred_x0[:, :, 3], gt_nodes[:, :, 3])
            loss_radius = F.mse_loss(pred_x0[:, :, 4], gt_nodes[:, :, 4])
            loss_orient = F.mse_loss(pred_x0[:, :, 5:8], gt_nodes[:, :, 5:8])
            loss_x0 = F.mse_loss(pred_x0, gt_nodes)

            # Organ type loss (CE + one-hot MSE as auxiliary)
            loss_organ = F.cross_entropy(
                pred_organ_logits.view(-1, 4), gt_organ_types.view(-1)
            )
            loss_type_mse = F.mse_loss(pred_x0[:, :, 8:12], gt_nodes[:, :, 8:12])

            # Existence loss with dynamic positive weight and focal-style penalty
            # Active nodes are rare (29..1217 out of 2048), so heavily up-weight positives.
            exist_pos = (gt_existence.squeeze(-1) > 0.5).float().sum()
            exist_neg = (gt_existence.squeeze(-1) <= 0.5).float().sum() + 1e-6
            dynamic_pos_weight = (exist_neg / exist_pos).clamp(min=1.0, max=50.0)
            loss_existence = F.binary_cross_entropy_with_logits(
                pred_exist_logits, gt_existence.squeeze(-1), pos_weight=dynamic_pos_weight
            )

            # DAP-based adaptive node-budget loss
            true_budget = (gt_existence.squeeze(-1) > 0.5).float().sum(dim=1) / float(N)
            pred_budget = outputs["pred_node_budget"]
            loss_budget = F.mse_loss(pred_budget, true_budget)

            # Sparse parent loss: gt_parent must be in top-k candidates.
            parent_candidates = outputs["pred_parent_candidates"]  # (B, N, k)
            k_val = parent_candidates.shape[-1]
            # (B, N, 1) vs (B, N, k) -> boolean mask where candidate == gt_parent
            match_mask = (parent_candidates == gt_parents.unsqueeze(-1))  # (B, N, k)
            # For each node, if gt parent is in candidates, target is its position in k list;
            # otherwise, mark as ignore (-100).
            parent_target = torch.full((B, N), -100, dtype=torch.long, device=device)
            has_match = match_mask.any(dim=-1)
            if has_match.any():
                matched_positions = torch.argmax(match_mask.int(), dim=-1)
                parent_target[has_match] = matched_positions[has_match]
            loss_parent = F.cross_entropy(
                pred_parent_logits.view(-1, k_val), parent_target.view(-1),
                ignore_index=-100
            )

            # 3D snap loss
            loss_snap3d = compute_snap_loss_3d(pred_x0, gt_parents, gt_adj, gt_existence)

            if epoch <= pretrain_end:
                # Pre-train only existence and node-budget before full multi-task training.
                loss = 5.0 * loss_existence + 2.0 * loss_budget
            else:
                loss = (10.0 * loss_coord
                        + 2.0 * loss_length
                        + 2.0 * loss_radius
                        + 2.0 * loss_orient
                        + 1.0 * loss_x0
                        + 5.0 * loss_organ
                        + 1.0 * loss_type_mse
                        + 2.0 * loss_existence
                        + 1.0 * loss_budget
                        + 0.5 * loss_parent
                        + 0.5 * loss_snap3d)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        lr_scheduler.step()

        # Validation
        val_loss = None
        if val_dataset is not None:
            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=collate_training_batch):
                    images = batch["image"].to(device)
                    gt_nodes = batch["nodes"].to(device)
                    gt_adj = batch["adj_matrix"].to(device)
                    gt_parents = batch["parent_indices"].to(device)
                    gt_existence = batch["existence_mask"].unsqueeze(-1).to(device)
                    gt_poses = batch["camera_pose"].to(device)
                    gt_dap = batch["dap"].to(device)
                    gt_organ_types = batch.get("organ_types", gt_nodes[:, :, 8:12].argmax(dim=-1)).to(device)

                    B, N, _ = gt_nodes.shape
                    timesteps = torch.randint(0, 1000, (B,), device=device).long()
                    noisy_nodes, _ = scheduler.add_noise(gt_nodes, timesteps)
                    outputs = model(noisy_nodes, gt_existence, timesteps, images,
                                    camera_poses=gt_poses, dap=gt_dap)
                    pred_x0 = outputs["pred_x0"]

                    loss_coord = F.mse_loss(pred_x0[:, :, 0:3], gt_nodes[:, :, 0:3])
                    loss_organ = F.cross_entropy(
                        outputs["pred_organ_type_logits"].view(-1, 4), gt_organ_types.view(-1)
                    )
                    loss_x0 = F.mse_loss(pred_x0, gt_nodes)
                    loss_existence_val = F.binary_cross_entropy_with_logits(
                        outputs["pred_existence_logits"], gt_existence.squeeze(-1)
                    )
                    vloss = 10.0 * loss_coord + 5.0 * loss_organ + 1.0 * loss_x0 + 2.0 * loss_existence_val
                    val_total += vloss.item() * B
                    val_count += B
            val_loss = val_total / val_count
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, best_save_path)
                print(f"  Saved best model (val_loss={val_loss:.4f})")

        if epoch % 50 == 0 or epoch == 1:
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            msg = (f"Epoch [{epoch:03d}/{epochs}] - Train Loss: {avg_loss:.4f} "
                   f"(Coord MSE: {loss_coord.item():.5f}, Organ CE: {loss_organ.item():.4f})")
            if val_loss is not None:
                msg += f" | Val Loss: {val_loss:.4f}"
            print(msg)

    torch.save(model.state_dict(), save_path)
    print(f"Saved final 3D diffusion model weights to '{save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_nodes", type=int, default=2048)
    parser.add_argument("--helios_data_root", type=str,
                        default="Digital-Crops/projects/syntheticdata_generation/build/output")
    parser.add_argument("--save_path", type=str,
                        default="diffusion_based/checkpoints/diffusion_model_3d.pt")
    parser.add_argument("--best_save_path", type=str,
                        default="diffusion_based/checkpoints/best_3d_model.pt")
    parser.add_argument("--freeze_pretrained", type=str, default=None,
                        help="Load a pretrained checkpoint and freeze backbone, only train existence+budget heads")
    parser.add_argument("--pretrain_existence_epochs", type=int, default=0,
                        help="Number of initial epochs to train only existence+budget before full multi-task loss")
    args = parser.parse_args()

    train_diffusion_3d(
        num_samples=args.num_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        helios_data_root=args.helios_data_root,
        save_path=args.save_path,
        best_save_path=args.best_save_path,
        freeze_pretrained=args.freeze_pretrained,
        pretrain_existence_epochs=args.pretrain_existence_epochs,
    )
