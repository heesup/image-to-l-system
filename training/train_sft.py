import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.generator import LSystemDatasetGenerator, PRESET_GRAMMARS
from dataset.dataloader import LSystemDataset, custom_collate_fn
from models.vlm_wrapper import LSystemVLM, get_device

def extract_rule_features(rule_str: str) -> list[float]:
    num_branches = rule_str.count("[")
    num_turns = rule_str.count("+") + rule_str.count("-")
    num_forwards = rule_str.count("F")
    return [float(num_branches), float(num_turns), float(num_forwards)]

def match_closest_preset_index(axiom: str, rules: dict) -> int:
    """Find closest preset index by feature similarity (branch count, turns, forwards)."""
    for idx, preset in enumerate(PRESET_GRAMMARS):
        if preset["axiom"] == axiom and preset["rules"] == rules:
            return idx

    rule_str = str(rules.get("X", rules.get("F", "")))
    feats = extract_rule_features(rule_str)

    best_idx = 0
    best_dist = float('inf')
    for idx, preset in enumerate(PRESET_GRAMMARS):
        p_str = str(preset["rules"].get("X", preset["rules"].get("F", "")))
        p_feats = extract_rule_features(p_str)
        dist = sum((a - b)**2 for a, b in zip(feats, p_feats))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def train_sft(
    data_dir: str = "data/synthetic",
    num_samples: int = 200,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    device_name: str = "auto"
):
    device = get_device() if device_name == "auto" else torch.device(device_name)

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
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss()

    os.makedirs("checkpoints", exist_ok=True)

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct_cls = 0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            lsystem_dicts = batch["lsystem_dict"]

            # Map target axioms and rules to template indices
            target_cls_list = []
            target_params_list = []
            for d in lsystem_dicts:
                idx = match_closest_preset_index(d.get("axiom", "X"), d.get("rules", {}))
                target_cls_list.append(idx)

                target_params_list.append([
                    float(d.get("angle", 25.0)) / 95.0,
                    float(d.get("iterations", 3)) / 5.0,
                    float(d.get("step_size", 1.0)) / 2.0,
                    float(d.get("line_width", 2.0)) / 3.0
                ])


            target_cls = torch.tensor(target_cls_list, dtype=torch.long, device=device)
            target_params = torch.tensor(target_params_list, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            out = model(images)

            logits = out["grammar_logits"]
            pred_params = torch.sigmoid(out["pred_params"])

            loss_cls = criterion_cls(logits, target_cls)
            loss_param = criterion_reg(pred_params, target_params)

            loss = loss_cls + 0.5 * loss_param
            loss.backward()
            optimizer.step()


            total_loss += loss.item()
            correct_cls += (logits.argmax(dim=-1) == target_cls).sum().item()
            total_samples += images.size(0)

        avg_loss = total_loss / max(1, len(dataloader))
        acc = (correct_cls / max(1, total_samples)) * 100.0
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | Topology Accuracy: {acc:.1f}%")

    ckpt_path = "checkpoints/sft_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"=== SFT Training Complete! Model saved to '{ckpt_path}' ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT model for Image-to-L-System")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=15)
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
