"""
Unified Training Script for Global Plant VAE and Hierarchical Shoot VAE.

Trains both architectures on the Helios dataset and saves checkpoints to:
- diffusion_based/checkpoints/plant_global_vae_best.pt
- diffusion_based/checkpoints/plant_shoot_vae_best.pt
"""

import os
import sys
import glob
import time
import argparse
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.plant_global_vae import PlantGlobalVAE, compute_global_vae_loss
from diffusion_based.models.plant_shoot_vae import PlantShootVAE, compute_shoot_vae_loss


class FastPlantArrayDataset(Dataset):
    def __init__(self, data_root: str = "dataset/helios_data", max_samples: int = 1000, max_organs: int = 1600):
        self.data_root = os.path.abspath(data_root)
        self.max_organs = max_organs
        xmls = sorted(glob.glob(os.path.join(self.data_root, "**", "*.xml"), recursive=True))
        if max_samples is not None and len(xmls) > max_samples:
            xmls = xmls[:max_samples]
        self.xml_paths = xmls

    def __len__(self) -> int:
        return len(self.xml_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        xml_path = self.xml_paths[idx]
        arr = PlantOrganArray.from_xml_file(xml_path)
        tensor = arr.tensor
        N = tensor.shape[0]

        actual_n = min(N, self.max_organs)
        padded = torch.zeros((self.max_organs, tensor.shape[1]), dtype=torch.float32)
        padded[:actual_n] = tensor[:actual_n]

        mask = torch.zeros(self.max_organs, dtype=torch.bool)
        mask[:actual_n] = True

        return {
            "tensor": padded,
            "mask": mask,
            "num_organs": torch.tensor(actual_n, dtype=torch.long),
        }


def collate_plant_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    tensors = torch.stack([b["tensor"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    num_organs = torch.stack([b["num_organs"] for b in batch], dim=0)

    max_n = max(int(num_organs.max().item()), 1)
    return {
        "tensor": tensors[:, :max_n],
        "mask": masks[:, :max_n],
        "num_organs": num_organs,
    }


def train_model(
    model_type: str = "global",
    epochs: int = 25,
    batch_size: int = 8,
    lr: float = 3e-4,
    max_samples: int = 1000,
    checkpoint_dir: str = "diffusion_based/checkpoints",
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f"  Training {model_type.upper()} Plant VAE on {device}")
    print(f"=======================================================")

    dataset = FastPlantArrayDataset(max_samples=max_samples)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_plant_batch, num_workers=0
    )
    print(f"[INFO] Loaded {len(dataset)} plant XML samples.")

    if model_type == "global":
        model = PlantGlobalVAE(latent_dim=512, hidden_dim=512, ffn_dim=2048, encoder_layers=6, decoder_layers=6).to(device)
        loss_fn = compute_global_vae_loss
        save_path = os.path.join(checkpoint_dir, "plant_global_vae_best.pt")
    elif model_type == "shoot":
        model = PlantShootVAE(max_shoots=32, shoot_latent_dim=256, hidden_dim=512, ffn_dim=2048, encoder_layers=6, decoder_layers=6).to(device)
        loss_fn = compute_shoot_vae_loss
        save_path = os.path.join(checkpoint_dir, "plant_shoot_vae_best.pt")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_geom = 0.0
        running_cls = 0.0
        running_ang = 0.0

        for batch in dataloader:
            x = batch["tensor"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            losses = loss_fn(model, x, mask=mask, beta=1e-4)
            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_geom += losses["loss_geom"].item()
            running_cls += losses["loss_cls"].item()
            running_ang += losses["loss_angle"].item()

        lr_sched.step()
        n_batches = len(dataloader)
        ep_time = time.time() - t0
        avg_loss = running_loss / n_batches
        avg_geom = running_geom / n_batches
        avg_cls = running_cls / n_batches
        avg_ang = running_ang / n_batches

        print(f"Epoch {epoch:02d}/{epochs:02d} ({ep_time:4.1f}s) | Loss: {avg_loss:6.2f} | "
              f"Geom: {avg_geom:.4f} | Cls: {avg_cls:.4f} | Ang: {avg_ang:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "model_type": model_type,
            }, save_path)
            print(f"  --> Saved Best {model_type.upper()} Checkpoint to: {save_path}")

    print(f"\n[OK] Completed training {model_type.upper()} VAE. Best Checkpoint: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["global", "shoot", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=800)
    args = parser.parse_args()

    if args.model in ["global", "both"]:
        train_model("global", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_samples=args.max_samples)

    if args.model in ["shoot", "both"]:
        train_model("shoot", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_samples=args.max_samples)
