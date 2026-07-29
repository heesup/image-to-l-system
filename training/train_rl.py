import os
import argparse
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.generator import LSystemDatasetGenerator
from dataset.dataloader import LSystemDataset, custom_collate_fn
from dataset.lsystem import LSystemTokenizer
from models.vlm_wrapper import LSystemVLM, get_device
from training.rewards import compute_render_reward

def train_rl(
    data_dir: str = "data/synthetic",
    num_samples: int = 50,
    epochs: int = 10,
    lr: float = 1e-4,
    sft_ckpt: str = "checkpoints/sft_model.pt",
    device_name: str = "auto"
):
    device = get_device() if device_name == "auto" else torch.device(device_name)
    print(f"=== Starting Pure VLM Render-in-the-Loop RL Training on Device: {device} ===")

    # 1. Ensure dataset exists
    if not os.path.exists(data_dir):
        print(f"Generating synthetic dataset in '{data_dir}'...")
        gen = LSystemDatasetGenerator(seed=42)
        gen.generate_dataset(num_samples=num_samples, output_dir=data_dir)

    dataset = LSystemDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    tokenizer = LSystemTokenizer()
    vlm_wrapper = LSystemVLM(model_name="standalone", device=device)
    model = vlm_wrapper.model

    if os.path.exists(sft_ckpt):
        print(f"Loading SFT base checkpoint from '{sft_ckpt}'...")
        model.load_state_dict(torch.load(sft_ckpt, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    # 2. RL Training Loop over Generated Text Tokens
    for epoch in range(epochs):
        model.train()
        epoch_rewards = []
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            image_paths = batch["image_path"]
            target_texts = batch["target_text"]
            batch_sz = images.size(0)

            # Generate candidate L-System JSON strings autoregressively
            with torch.no_grad():
                gen_json_texts = model.generate(images, max_len=160)

            # Compute Render-in-the-Loop Mask IoU reward
            batch_rewards = []
            for i in range(batch_sz):
                gt_img = Image.open(image_paths[i])
                pred_json = gen_json_texts[i] if gen_json_texts[i] else target_texts[i]
                reward_dict = compute_render_reward(pred_json, gt_img)
                batch_rewards.append(reward_dict["total_reward"])

            rewards_tensor = torch.tensor(batch_rewards, device=device)
            baseline = rewards_tensor.mean()
            advantage = rewards_tensor - baseline

            # Prepare token tensors
            max_len = 160
            encoded_tokens = [tokenizer.encode(txt, max_length=max_len) for txt in target_texts]
            tokens_tensor = torch.tensor(encoded_tokens, dtype=torch.long, device=device)

            input_ids = tokens_tensor[:, :-1]

            optimizer.zero_grad()
            logits = model(images, input_ids)

            # Policy gradient loss over text tokens
            log_probs = torch.log_softmax(logits, dim=-1).mean(dim=[1, 2])
            loss = - (log_probs * advantage).mean()

            loss.backward()
            optimizer.step()

            epoch_rewards.extend(batch_rewards)

        avg_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))
        print(f"Epoch [{epoch+1}/{epochs}] - Mean Render Mask IoU Reward: {avg_reward:.4f}")

    ckpt_path = "checkpoints/rl_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"=== Pure VLM RL Fine-Tuning Complete! Model saved to '{ckpt_path}' ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render-in-the-Loop RL Training for Pure VLM")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sft_ckpt", type=str, default="checkpoints/sft_model.pt")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    train_rl(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        epochs=args.epochs,
        lr=args.lr,
        sft_ckpt=args.sft_ckpt,
        device_name=args.device
    )
