import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.plant3d_dataset import Plant3DDataset
from diffusion_based.models.graph_diffuser_3d import PlantGraphDiffuser3D
from diffusion_based.training.train_diffusion import DDPMScheduler, get_device

def train_diffusion_3d(num_samples: int = 100, epochs: int = 500, lr: float = 3e-4, save_path: str = "diffusion_based/checkpoints/diffusion_model_3d.pt"):
    device = get_device()
    print(f"--- Training 3D Botanical Plant Diffusion Model (2D Image -> 3D Plant Graph) on device: {device} ---")

    dataset = Plant3DDataset(num_samples=num_samples)
    four_samples = [dataset[i] for i in range(min(4, len(dataset)))]

    images = torch.stack([s["image"] for s in four_samples]).to(device)
    gt_nodes = torch.stack([s["nodes"] for s in four_samples]).to(device)
    gt_adj = torch.stack([s["adj_matrix"] for s in four_samples]).to(device)
    gt_parents = torch.stack([s["parent_indices"] for s in four_samples]).to(device)
    gt_existence = torch.stack([s["existence_mask"] for s in four_samples]).unsqueeze(-1).to(device)

    scheduler = DDPMScheduler(timesteps=1000)
    model = PlantGraphDiffuser3D(max_nodes=64).to(device)
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
        
        # 3D Position Loss (x, y, z)
        loss_coord3d = F.mse_loss(pred_x0[:, :, :3], gt_nodes[:, :, :3])
        loss_x0 = F.mse_loss(pred_x0, gt_nodes)
        
        pos_w = torch.tensor([5.0], device=device)
        loss_existence = F.binary_cross_entropy_with_logits(outputs["pred_existence_logits"], gt_existence.squeeze(-1), pos_weight=pos_w)
        loss_parent = F.cross_entropy(outputs["pred_parent_logits"].view(-1, N), gt_parents.view(-1))

        # 3D Organ Joint Snap Loss
        base_x, base_y, base_z = pred_x0[:, :, 0], pred_x0[:, :, 1], pred_x0[:, :, 2]
        theta = (pred_x0[:, :, 3] * 2.0 - 1.0) * math.pi
        phi = pred_x0[:, :, 4] * math.pi
        length = pred_x0[:, :, 5]

        tip_x = base_x + length * torch.sin(phi) * torch.cos(theta)
        tip_y = base_y - length * torch.cos(phi)
        tip_z = base_z + length * torch.sin(phi) * torch.sin(theta)

        diff_x = tip_x.unsqueeze(2) - base_x.unsqueeze(1)
        diff_y = tip_y.unsqueeze(2) - base_y.unsqueeze(1)
        diff_z = tip_z.unsqueeze(2) - base_z.unsqueeze(1)
        dist_sq_3d = diff_x**2 + diff_y**2 + diff_z**2
        loss_snap3d = (dist_sq_3d * gt_adj).sum() / (gt_adj.sum() + 1e-5)

        loss = 10.0 * loss_coord3d + loss_x0 + 0.5 * loss_existence + 0.5 * loss_parent + 0.5 * loss_snap3d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs}] - Total Loss: {loss.item():.4f} (3D Coord MSE: {loss_coord3d.item():.5f}, Parent CE: {loss_parent.item():.4f})")

    torch.save(model.state_dict(), save_path)
    print(f"Saved trained 3D diffusion model weights to '{save_path}'")

if __name__ == "__main__":
    train_diffusion_3d(num_samples=100, epochs=500)
