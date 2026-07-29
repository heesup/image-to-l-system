import os
import json
import argparse
from typing import Optional
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset.lsystem import LSystem
from dataset.renderer import TurtleRenderer
from models.vlm_wrapper import LSystemVLM, get_device
from training.rewards import compute_render_reward

def predict_image(
    image_path: str,
    checkpoint_path: str = "checkpoints/rl_model.pt",
    output_plot: Optional[str] = None,
    output_render: Optional[str] = None,
    device_name: str = "auto"
) -> LSystem:
    """Inference entrypoint: Predict L-System specification from a plant image."""
    device = get_device() if device_name == "auto" else torch.device(device_name)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: '{image_path}'")

    input_img = Image.open(image_path).convert("RGB")

    # Load Model
    vlm_wrapper = LSystemVLM(model_name="standalone", device=device)
    model = vlm_wrapper.model

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded trained model weights from '{checkpoint_path}'")
    else:
        print(f"[Warning] Checkpoint '{checkpoint_path}' not found. Using default weights.")

    model.eval()

    # Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(input_img).unsqueeze(0).to(device)

    # Predict L-System specification object
    estimated_lsystem = model.predict_lsystem(img_tensor)

    renderer = TurtleRenderer(image_size=input_img.size)
    rendered_img = renderer.render(estimated_lsystem)

    metrics = compute_render_reward(estimated_lsystem.to_json(), input_img, renderer=renderer)

    print("\n=== Inference Prediction Results ===")
    print(estimated_lsystem.to_json())
    print(f"Mask IoU vs Input Image: {metrics['iou_reward']:.4f}")

    if output_render:
        os.makedirs(os.path.dirname(output_render) or ".", exist_ok=True)
        rendered_img.save(output_render)
        print(f"Saved reconstructed render to: '{output_render}'")

    if output_plot:
        os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(input_img)
        axes[0].set_title("Input Plant Image")
        axes[0].axis("off")

        axes[1].imshow(rendered_img)
        axes[1].set_title(f"Estimated Reconstructed Render\n(Angle: {estimated_lsystem.angle}°, IoU: {metrics['iou_reward']:.2f})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150)
        plt.close()
        print(f"Saved comparison plot to: '{output_plot}'")

    return estimated_lsystem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image-to-L-System Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input plant image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rl_model.pt", help="Path to model checkpoint")
    parser.add_argument("--output_plot", type=str, default=None, help="Optional output path for side-by-side plot")
    parser.add_argument("--output_render", type=str, default=None, help="Optional output path for reconstructed render")
    parser.add_argument("--device", type=str, default="auto", help="Device (mps/cuda/cpu)")

    args = parser.parse_args()
    predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_plot=args.output_plot,
        output_render=args.output_render,
        device_name=args.device
    )
