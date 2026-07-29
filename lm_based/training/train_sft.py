import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.generator import LSystemDatasetGenerator
from dataset.dataloader import LSystemDataset, custom_collate_fn
from dataset.lsystem import LSystemTokenizer
from models.vlm_wrapper import LSystemVLM, get_device

def train_sft(
    data_dir: str = "data/synthetic",
    num_samples: int = 300,
    epochs: int = 25,
    batch_size: int = 8,
    lr: float = 1e-3,
    device_name: str = "auto"
):
    device = get_device() if device_name == "auto" else torch.device(device_name)

    print(f"=== Starting Pure VLM (Language Modeling) SFT Training on Device: {device} ===")

    # 1. Ensure synthetic dataset exists
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"Generating synthetic dataset with {num_samples} samples in '{data_dir}'...")
        gen = LSystemDatasetGenerator(seed=42)
        gen.generate_dataset(num_samples=num_samples, output_dir=data_dir)

    # 2. Setup DataLoader & Pure VLM Model
    dataset = LSystemDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    tokenizer = LSystemTokenizer()
    vlm_wrapper = LSystemVLM(model_name="standalone", device=device)
    model = vlm_wrapper.model

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_lm = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    os.makedirs("checkpoints", exist_ok=True)

    # 3. Training Loop (Pure Causal Language Modeling over JSON Text Tokens)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            target_texts = batch["target_text"]

            # Encode JSON targets into token IDs
            max_len = 160
            encoded_tokens = [tokenizer.encode(txt, max_length=max_len) for txt in target_texts]
            tokens_tensor = torch.tensor(encoded_tokens, dtype=torch.long, device=device)

            input_ids = tokens_tensor[:, :-1]
            target_ids = tokens_tensor[:, 1:]

            optimizer.zero_grad()
            logits = model(images, input_ids)

            # Causal Language Modeling Loss
            loss = criterion_lm(logits.view(-1, logits.size(-1)), target_ids.reshape(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            mask = (target_ids != tokenizer.pad_id)
            correct_tokens += (preds[mask] == target_ids[mask]).sum().item()
            total_tokens += mask.sum().item()

        avg_loss = total_loss / max(1, len(dataloader))
        token_acc = (correct_tokens / max(1, total_tokens)) * 100.0
        print(f"Epoch [{epoch+1}/{epochs}] - LM Loss: {avg_loss:.4f} | Token Accuracy: {token_acc:.1f}%")

    ckpt_path = "checkpoints/sft_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"=== Pure VLM SFT Training Complete! Model saved to '{ckpt_path}' ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pure VLM Model for Image-to-L-System")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
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
