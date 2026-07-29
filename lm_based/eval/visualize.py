import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from dataset.lsystem import LSystem
from dataset.renderer import TurtleRenderer
from dataset.dataloader import LSystemDataset

def visualize_predictions(data_dir: str = "data/synthetic", num_samples: int = 3, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    dataset = LSystemDataset(data_dir=data_dir)
    renderer = TurtleRenderer()

    num_samples = min(num_samples, len(dataset))
    print(f"Generating {num_samples} visual comparisons in '{output_dir}'...")

    for i in range(num_samples):
        sample = dataset[i]
        gt_img = Image.open(sample["image_path"])

        # Create LSystem object from ground truth metadata
        lsystem_dict = sample["lsystem_dict"]
        lsystem = LSystem.from_dict(lsystem_dict)
        pred_img = renderer.render(lsystem)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gt_img)
        axes[0].set_title("Ground Truth Plant")
        axes[0].axis("off")

        axes[1].imshow(pred_img)
        axes[1].set_title(f"Estimated Render\n(Angle: {lsystem.angle}°, Iter: {lsystem.iterations})")
        axes[1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"comparison_{i:02d}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved visualization: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Ground Truth vs Estimated Render")
    parser.add_argument("--data_dir", type=str, default="data/synthetic")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="plots")

    args = parser.parse_args()
    visualize_predictions(data_dir=args.data_dir, num_samples=args.num_samples, output_dir=args.output_dir)
