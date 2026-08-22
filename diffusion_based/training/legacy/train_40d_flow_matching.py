"""
Training Script for Direct 40D Plant Organ Array Flow Matching.

Trains a ViT + Cross-Attention Transformer to predict the 40D velocity field
transporting standard Gaussian noise x_0 ~ N(0, I) directly to normalized
PlantOrganArray x_1, conditioned on single RGB images.
"""

import os
import sys
import glob
import re
import random
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.plant_global_vae import OrganFeatureNormalizer
from diffusion_based.models.plant_organ_40d_flow_matching import PlantOrgan40DFlowMatchingModel
from diffusion_based.training.flow_matching import FlowMatchingScheduler


class Direct40DPlantDataset(Dataset):
    """Loads paired (RGB Image, (N, 40) PlantOrganArray tensor)."""

    def __init__(
        self,
        data_root: str = "dataset/helios_data",
        image_size: int = 128,
        max_organs: int = 1200,
        max_samples: Optional[int] = None,
    ):
        self.data_root = os.path.abspath(data_root)
        self.image_size = image_size
        self.max_organs = max_organs
        self.normalizer = OrganFeatureNormalizer()

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.samples = self._discover_pairs()
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def _discover_pairs(self) -> List[Dict[str, Any]]:
        pairs = []
        xml_files = sorted(glob.glob(os.path.join(self.data_root, "**", "*_plant_0000.xml"), recursive=True))

        for xml_path in xml_files:
            img_path = xml_path.replace("_plant_0000.xml", "_rad.jpeg")
            if not os.path.exists(img_path):
                img_path = xml_path.replace("_plant_0000.xml", "_vis.jpeg")
            if not os.path.exists(img_path):
                img_path = xml_path.replace(".xml", ".jpeg")
            if not os.path.exists(img_path):
                img_path = xml_path.replace(".xml", ".png")

            if os.path.exists(img_path):
                pairs.append({"xml": xml_path, "img": img_path})

        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        pil_img = Image.open(sample["img"]).convert("RGB")
        img_t = self.img_transform(pil_img)

        arr = PlantOrganArray.from_xml_file(sample["xml"])
        raw_tensor = arr.tensor  # (N_real, 40)
        N_real = raw_tensor.shape[0]

        # Pad to max_organs
        if N_real > self.max_organs:
            raw_tensor = raw_tensor[:self.max_organs]
            N_real = self.max_organs

        padded_tensor = torch.zeros((self.max_organs, 40), dtype=torch.float32)
        padded_tensor[:N_real] = raw_tensor
        mask = torch.ones((self.max_organs,), dtype=torch.bool)
        mask[:N_real] = False  # False = active, True = padding mask

        # Normalize 40D continuous attributes
        norm_tensor = self.normalizer.normalize(padded_tensor.unsqueeze(0)).squeeze(0)

        return {
            "image": img_t,
            "organs": norm_tensor,
            "mask": mask,
            "num_organs": torch.tensor(N_real, dtype=torch.long),
        }


def train_40d_flow_matching(
    epochs: int = 35,
    batch_size: int = 16,
    lr: float = 5e-4,
    max_organs: int = 1200,
    data_root: str = "dataset/helios_data",
    ckpt_dir: str = "diffusion_based/checkpoints/fm",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Starting 40D Direct Flow Matching Training on {device} (Epochs: {epochs}, Batch Size: {batch_size})...")

    dataset = Direct40DPlantDataset(data_root=data_root, max_organs=max_organs, max_samples=1500)
    print(f"Loaded {len(dataset)} paired image-plant samples.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = PlantOrgan40DFlowMatchingModel(
        max_organs=max_organs,
        organ_dim=40,
        image_size=128,
        patch_size=8,
        embed_dim=256,
        encoder_layers=6,
        decoder_layers=6,
        num_heads=8,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    fm_scheduler = FlowMatchingScheduler()
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float("inf")
    best_ckpt = os.path.join(ckpt_dir, "plant_organ_40d_flow_matching_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for batch in dataloader:
            images = batch["image"].to(device)         # (B, 3, 128, 128)
            organs_gt = batch["organs"].to(device)      # (B, N, 40) normalized x_1
            pad_mask = batch["mask"].to(device)         # (B, N) bool

            B, N, D = organs_gt.shape

            # 1. PURE GAUSSIAN NOISE PRIOR x_0 ~ N(0, I) (ZERO LEAKAGE)
            x_0 = torch.randn((B, N, D), device=device)
            x_1 = organs_gt

            # 2. Continuous Timestep Sampling
            t = fm_scheduler.sample_time(B, device)

            # 3. Straight Path Interpolation & Target Velocity
            x_t = fm_scheduler.sample_xt(x_0, x_1, t)
            v_target = fm_scheduler.velocity_target(x_0, x_1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(x_t, t, images, key_padding_mask=pad_mask)
                v_pred = outputs["pred_velocity"]

                # Weighted MSE Loss across 40 dimensions
                loss_weights = torch.ones(40, device=device)
                loss_weights[11] = 5.0   # organ_type
                loss_weights[13:16] = 5.0 # dimensions (length, radius, scale)
                loss_weights[16:21] = 5.0 # angles (pitch, yaw, roll, curv, phyllo)
                loss_weights[39] = 10.0  # existence

                active_mask = (~pad_mask).unsqueeze(-1).float()  # (B, N, 1)
                diff_sq = (v_pred - v_target) ** 2 * loss_weights.unsqueeze(0).unsqueeze(0)
                loss = (diff_sq * active_mask).sum() / (active_mask.sum() * 40.0 + 1e-6)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * B
            count += B

        lr_scheduler.step()
        avg_loss = total_loss / count
        print(f"Epoch {epoch:02d}/{epochs:02d} | Velocity MSE Loss: {avg_loss:.6f} | LR: {lr_scheduler.get_last_lr()[0]:.6e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, best_ckpt)

    print(f"Training Complete! Best 40D Model Saved to {best_ckpt} with Loss: {best_loss:.6f}")
    return best_ckpt


if __name__ == "__main__":
    train_40d_flow_matching(epochs=35, batch_size=16, lr=5e-4)
