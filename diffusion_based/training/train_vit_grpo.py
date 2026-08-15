"""
GRPO (Group Relative Policy Optimization) Training Script for 3D Plant Inverse Modeling.

Fine-tunes the pre-trained ViT + Transformer Decoder using reinforcement learning
with group-relative advantage normalization and multi-modal image & topology rewards.
"""

import os
import sys
import argparse
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.vit_image_to_organ_array import ViTImageToOrganArray
from diffusion_based.models.vit_grpo_policy import ViTGRPOPolicy
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.training.grpo_rewards import PlantGRPORewardEngine
from diffusion_based.dataset.organ_array_dataset import OrganArrayDataset


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_kl_divergence(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Approximate KL divergence: log \pi_\theta - log \pi_ref."""
    return (policy_log_probs - ref_log_probs).mean()


def main():
    parser = argparse.ArgumentParser(description="Train ViT Image-to-Organ-Array via GRPO RL")
    parser.add_argument("--base_checkpoint", type=str, default="diffusion_based/checkpoints/vit_backprop_vit.pt",
                        help="Pre-trained supervised ViT checkpoint")
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--val_pattern", type=str, default="*seed09*")
    parser.add_argument("--epochs", type=int, default=20, help="GRPO training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of prompt images per batch")
    parser.add_argument("--group_size", type=int, default=4, help="Number of sampled plant candidates G per prompt")
    parser.add_argument("--ppo_epochs", type=int, default=2, help="PPO update steps per rollout batch")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for policy optimizer")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--kl_weight", type=float, default=0.04, help="KL divergence penalty coefficient")
    parser.add_argument("--entropy_weight", type=float, default=0.001, help="Entropy bonus coefficient")
    parser.add_argument("--checkpoint_dir", type=str, default="diffusion_based/checkpoints")
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print(f"GRPO Training on device: {device}")

    # 1. Load Base Model to read max_nodes and configuration
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    max_nodes = ckpt_args.get("max_nodes", 2048)
    image_size = ckpt_args.get("image_size", 128)
    patch_size = ckpt_args.get("patch_size", 8)
    embed_dim = ckpt_args.get("embed_dim", 256)
    encoder_layers = ckpt_args.get("encoder_layers", 6)
    decoder_layers = ckpt_args.get("decoder_layers", 4)

    # 2. Load Dataset
    val_globs = [g.strip() for g in args.val_pattern.split(",")] if args.val_pattern else []
    dataset = OrganArrayDataset(
        data_root=args.data_root,
        max_nodes=max_nodes,
        image_size=image_size,
        use_gt_renderer_image=True,
        device=device,
        exclude_globs=val_globs,
    )
    print(f"GRPO Train dataset size: {len(dataset)} | max_nodes: {max_nodes}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    base_model = ViTImageToOrganArray(
        max_nodes=max_nodes, node_dim=40,
        image_size=image_size, patch_size=patch_size,
        embed_dim=embed_dim, encoder_layers=encoder_layers,
        decoder_layers=decoder_layers, num_heads=8, num_organ_types=8,
    ).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded base model weights from {args.base_checkpoint}")

    # 3. Create Policy and Frozen Reference Policy
    policy = ViTGRPOPolicy(base_model, init_log_std=-2.5).to(device)
    ref_policy = copy.deepcopy(policy).to(device)
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad = False

    # 4. Setup Reward Engine
    renderer = HeliosPyTorchRenderer(image_size=128).to(device)
    reward_engine = PlantGRPORewardEngine(renderer=renderer, w_ssim=2.5, w_mae=1.5, w_iou=1.5, w_node=0.5, w_validity=0.5)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("\nStarting GRPO Reinforcement Learning loop...\n")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_loss, total_reward = 0.0, 0.0
        total_ssim, total_mae, total_iou = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in dataloader:
            images = batch["image"].to(device)  # (B, 3, 128, 128)
            num_nodes_gt = batch["num_nodes"].to(device) # (B,)
            B = images.shape[0]

            # A. Rollout: Sample G candidates per image prompt
            with torch.no_grad():
                rollout = policy.sample_group(
                    images, group_size=args.group_size, dataset=dataset, temperature=1.0
                )
                candidates = rollout["candidates"]         # (B, G) list of PlantOrganArray
                sampled_tensors = rollout["sampled_tensors"] # (B, G, N, node_dim)
                old_log_probs = rollout["log_probs"]       # (B, G)

                # Reference policy log-probs for KL constraint
                ref_log_probs, _ = ref_policy.evaluate_log_probs(images, sampled_tensors)

                # B. Evaluate Rewards: Render each candidate and compute SSIM/MAE/IoU
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                target_rgb_stack = (images * std + mean).clamp(0.0, 1.0)

                rewards, r_metrics = reward_engine.evaluate_group_rewards(
                    candidates, target_images=target_rgb_stack, target_node_counts=num_nodes_gt, device=device
                )

                # C. Compute Group Relative Advantage: A_i = (r_i - mean) / (std + 1e-4)
                mean_r = rewards.mean(dim=-1, keepdim=True)
                std_r = rewards.std(dim=-1, keepdim=True).clamp(min=1e-4)
                advantages = (rewards - mean_r) / std_r  # (B, G)

            # D. PPO Policy Gradient Update
            for ppo_iter in range(args.ppo_epochs):
                new_log_probs, entropy = policy.evaluate_log_probs(images, sampled_tensors)

                # Importance ratio
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL penalty against reference policy
                kl_div = compute_kl_divergence(new_log_probs, ref_log_probs)
                kl_loss = args.kl_weight * kl_div

                # Entropy bonus
                ent_loss = -args.entropy_weight * entropy.mean()

                loss = policy_loss + kl_loss + ent_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * B
            total_reward += r_metrics["r_total"] * B
            total_ssim += r_metrics["r_ssim"] * B
            total_mae += r_metrics["r_mae"] * B
            total_iou += r_metrics["r_iou"] * B
            n_batches += B

        avg_loss = total_loss / max(n_batches, 1)
        avg_r = total_reward / max(n_batches, 1)
        avg_ssim = total_ssim / max(n_batches, 1)
        avg_mae = total_mae / max(n_batches, 1)
        avg_iou = total_iou / max(n_batches, 1)

        print(
            f"Epoch {epoch:03d} | Loss={avg_loss:.4f} | Reward={avg_r:.4f} "
            f"| SSIM={avg_ssim:.4f} | MAE={avg_mae:.4f} | IoU={avg_iou:.4f}",
            flush=True
        )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"vit_grpo_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": policy.model.state_dict(),
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved GRPO checkpoint to {ckpt_path}", flush=True)

    # Final GRPO policy save
    final_ckpt = os.path.join(args.checkpoint_dir, "vit_grpo_policy.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": policy.model.state_dict(),
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, final_ckpt)
    print(f"\nGRPO Training complete. Saved to: {final_ckpt}")


if __name__ == "__main__":
    main()
