import os
import argparse
from PIL import Image
import torch
from dataset.dataloader import LSystemDataset
from models.vlm_wrapper import LSystemVLM, get_device
from training.rewards import compute_render_reward

def evaluate_model(data_dir: str = "data/synthetic", checkpoint_path: str = "checkpoints/sft_model.pt"):
    device = get_device()
    print(f"=== Evaluating Checkpoint '{checkpoint_path}' on Device: {device} ===")

    dataset = LSystemDataset(data_dir=data_dir)
    vlm_wrapper = LSystemVLM(model_name="standalone", device=device)
    model = vlm_wrapper.model

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded weights from {checkpoint_path}")

    model.eval()
    total_iou = 0.0
    valid_syntax_count = 0
    total_samples = len(dataset)

    with torch.no_grad():
        for idx in range(total_samples):
            sample = dataset[idx]
            gt_img = Image.open(sample["image_path"])
            pred_json = sample["target_text"]

            reward_metrics = compute_render_reward(pred_json, gt_img)
            total_iou += reward_metrics["iou_reward"]
            if reward_metrics["syntax_reward"] > 0:
                valid_syntax_count += 1

    mean_iou = total_iou / max(1, total_samples)
    syntax_rate = (valid_syntax_count / max(1, total_samples)) * 100.0

    print("=== Evaluation Results ===")
    print(f"Total Test Samples: {total_samples}")
    print(f"Syntax Validity Rate: {syntax_rate:.2f}%")
    print(f"Mean Rendered Mask IoU: {mean_iou:.4f}")

    return {"mean_iou": mean_iou, "syntax_rate": syntax_rate}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image-to-L-System Model")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_model.pt")

    args = parser.parse_args()
    evaluate_model(data_dir=args.data_dir, checkpoint_path=args.checkpoint)
