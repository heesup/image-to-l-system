import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple

from diffusion_based.dataset.graph_dataset import PlantGraphDataset
from diffusion_based.models.graph_diffuser import PlantGraphDiffuser

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class DDPMScheduler:
    """Linear DDPM Noise Scheduler for Node Coordinates & Attributes."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_nodes: torch.Tensor, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = original_nodes.device
        noise = torch.randn_like(original_nodes)
        
        alphas_cumprod = self.alphas_cumprod.to(device)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps])[:, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[timesteps])[:, None, None]

        noisy_nodes = sqrt_alphas_cumprod * original_nodes + sqrt_one_minus_alphas_cumprod * noise
        return noisy_nodes, noise

from diffusion_based.models.differentiable_renderer import DifferentiableLineRenderer

def train_diffusion(num_samples: int = 100, num_samples_to_fit: int = 4, epochs: int = 500, lr: float = 3e-4, save_path: str = "diffusion_based/checkpoints/diffusion_model.pt"):
    device = get_device()
    print(f"--- Training Plant Graph Diffusion Model on {num_samples_to_fit} Distinct Plant Images on device: {device} ---")

    dataset = PlantGraphDataset(num_synthetic_samples=num_samples)
    
    # Extract 4 distinct plant image samples
    four_samples = [dataset[i] for i in range(num_samples_to_fit)]
    
    images = torch.stack([s["image"] for s in four_samples]).to(device)
    gt_nodes = torch.stack([s["nodes"] for s in four_samples]).to(device)
    gt_adj = torch.stack([s["adj_matrix"] for s in four_samples]).to(device)
    gt_parents = torch.stack([s["parent_indices"] for s in four_samples]).to(device)
    gt_existence = torch.stack([s["existence_mask"] for s in four_samples]).unsqueeze(-1).to(device)

    scheduler = DDPMScheduler(timesteps=1000)
    model = PlantGraphDiffuser(max_nodes=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        B, N, _ = gt_nodes.shape
        timesteps = torch.randint(0, 1000, (B,), device=device).long()
        noisy_nodes, noise = scheduler.add_noise(gt_nodes, timesteps)

        outputs = model(noisy_nodes, gt_existence, timesteps, images)

        pred_x0 = outputs["pred_x0"]
        loss_coord = F.mse_loss(pred_x0[:, :, :2], gt_nodes[:, :, :2])
        loss_x0 = F.mse_loss(pred_x0, gt_nodes)
        pos_w = torch.tensor([5.0], device=device)
        loss_existence = F.binary_cross_entropy_with_logits(outputs["pred_existence_logits"], gt_existence.squeeze(-1), pos_weight=pos_w)
        loss_parent = F.cross_entropy(outputs["pred_parent_logits"].view(-1, N), gt_parents.view(-1))

        # Joint Snap Loss
        base_x, base_y = pred_x0[:, :, 0], pred_x0[:, :, 1]
        theta = (pred_x0[:, :, 2] * 2.0 - 1.0) * math.pi
        length = pred_x0[:, :, 3]
        tip_x = base_x + length * torch.cos(theta)
        tip_y = base_y - length * torch.sin(theta)

        diff_x = tip_x.unsqueeze(2) - base_x.unsqueeze(1)
        diff_y = tip_y.unsqueeze(2) - base_y.unsqueeze(1)
        dist_sq = diff_x**2 + diff_y**2
        loss_snap = (dist_sq * gt_adj).sum() / (gt_adj.sum() + 1e-5)

        loss = 10.0 * loss_coord + loss_x0 + 0.5 * loss_existence + 0.5 * loss_parent + 0.5 * loss_snap

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs}] - Total Loss: {loss.item():.4f} (Coord MSE: {loss_coord.item():.5f}, Parent CE: {loss_parent.item():.4f})")

    torch.save(model.state_dict(), save_path)
    print(f"Saved trained diffusion model weights to '{save_path}'")

if __name__ == "__main__":
    train_diffusion(num_samples=100, num_samples_to_fit=4, epochs=500)
