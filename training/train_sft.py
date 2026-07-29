import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.generator import LSystemDatasetGenerator
from dataset.dataloader import LSystemDataset, custom_collate_fn
from models.vlm_wrapper import LSystemVLM, get_device


def train_sft(
    data_dir: str = "data/synthetic",
    num_samples: int = 200,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    device_name: str = "auto"
):
    # Determine device (Apple Silicon MPS / CUDA / CPU)
    if device_name == "auto":
        device = get_device()
    else:
        device = torch.device(device_name)

    print(f"=== Starting SFT Training on Device: {device} ===")

    # 1. Ensure synthetic dataset exists
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"Generating synthetic dataset with {num_samples} samples in '{data_dir}'...")
        gen = LSystemDatasetGenerator(seed=42)
        gen.generate_dataset(num_samples=num_samples, output_dir=data_dir)

    # 2. Setup DataLoader & Model
    dataset = LSystemDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)


    vlm_wrapper = LSystemVLM(model_name="standalone", device=device)
    model = vlm_wrapper.model

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_lm = nn.CrossEntropyLoss(ignore_index=0)
    criterion_mse = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            # Create dummy target sequence for testing baseline pipeline
            batch_sz = images.size(0)
            input_ids = torch.randint(1, 30, (batch_sz, 32), device=device)
            target_ids = torch.randint(1, 30, (batch_sz, 32), device=device)
            target_params = torch.rand((batch_sz, 4), device=device) * 20.0

            optimizer.zero_grad()
            out = model(images, input_ids)

            logits = out["logits"]
            pred_params = out["pred_params"]

            loss_lm = criterion_lm(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss_param = criterion_mse(pred_params, target_params)

            loss = loss_lm + 0.1 * loss_param
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    ckpt_path = "checkpoints/sft_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"=== SFT Training Complete! Model saved to '{ckpt_path}' ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT model for Image-to-L-System")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    train_sft(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device
    )
