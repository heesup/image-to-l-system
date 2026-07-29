import os
import argparse
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.generator import LSystemDatasetGenerator, PRESET_GRAMMARS
from dataset.dataloader import LSystemDataset, custom_collate_fn
from dataset.lsystem import LSystem
from dataset.renderer import TurtleRenderer
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
    print(f"=== Starting Render-in-the-Loop RL Training on Device: {device} ===")

    # 1. Ensure dataset exists
    if not os.path.exists(data_dir):
        print(f"Generating synthetic dataset in '{data_dir}'...")
        gen = LSystemDatasetGenerator(seed=42)
        gen.generate_dataset(num_samples=num_samples, output_dir=data_dir)

    dataset = LSystemDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    vlm_wrapper = LSystemVLM(model_name="standalone", device=device)
    model = vlm_wrapper.model

    if os.path.exists(sft_ckpt):
        print(f"Loading SFT base checkpoint from '{sft_ckpt}'...")
        model.load_state_dict(torch.load(sft_ckpt, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    renderer = TurtleRenderer(image_size=(256, 256))

    os.makedirs("checkpoints", exist_ok=True)

    # 2. RL Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_rewards = []
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            image_paths = batch["image_path"]
            batch_sz = images.size(0)

            optimizer.zero_grad()
            out = model(images)
            grammar_logits = out["grammar_logits"]
            pred_params = out["pred_params"]

            predicted_cls = torch.argmax(grammar_logits, dim=-1).cpu().numpy()
            raw_params = pred_params.detach().cpu().numpy()

            # Render each prediction and calculate visual Mask IoU reward
            batch_rewards = []
            for i in range(batch_sz):
                gt_img = Image.open(image_paths[i])
                preset = PRESET_GRAMMARS[predicted_cls[i] % len(PRESET_GRAMMARS)]
                angle = float(round(max(10.0, min(75.0, abs(raw_params[i][0]))), 1))
                iterations = int(max(2, min(5, round(abs(raw_params[i][1])))))
                step_size = float(round(max(0.5, min(3.0, abs(raw_params[i][2]))), 2))
                line_width = float(round(max(1.0, min(5.0, abs(raw_params[i][3]))), 1))

                cand_lsystem = LSystem(
                    axiom=preset["axiom"],
                    rules=preset["rules"],
                    angle=angle,
                    iterations=iterations,
                    step_size=step_size,
                    line_width=line_width
                )
                reward_dict = compute_render_reward(cand_lsystem.to_json(), gt_img, renderer=renderer)
                batch_rewards.append(reward_dict["iou_reward"])

            rewards_tensor = torch.tensor(batch_rewards, device=device)
            baseline = rewards_tensor.mean()
            advantage = rewards_tensor - baseline

            log_probs = torch.log_softmax(grammar_logits, dim=-1)
            selected_log_probs = log_probs.gather(1, torch.tensor(predicted_cls, device=device).unsqueeze(1)).squeeze(1)

            # Policy gradient loss + parameter refinement loss weighted by reward advantage
            loss = - (selected_log_probs * advantage).mean() + 0.01 * pred_params.pow(2).mean()

            loss.backward()
            optimizer.step()

            epoch_rewards.extend(batch_rewards)

        avg_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))
        print(f"Epoch [{epoch+1}/{epochs}] - Mean Render Mask IoU: {avg_reward:.4f}")

    ckpt_path = "checkpoints/rl_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"=== RL Fine-Tuning Complete! Model saved to '{ckpt_path}' ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render-in-the-Loop RL Training")
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
