"""
Visualize 94D PlantOrganArray generation from a trained image-to-graph diffusion model.
"""

import os
import sys
import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.organ_array_diffuser import PlantOrganArrayDiffuser
from diffusion_based.dataset.organ_array_dataset import OrganArrayDataset
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer
from diffusion_based.models.plant_organ_array import PlantOrganArray


class DDPMScheduler:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_organ_array(
    model: PlantOrganArrayDiffuser,
    image: torch.Tensor,
    scheduler: DDPMScheduler,
    dataset: OrganArrayDataset,
    steps: int = 50,
    top_k_active: int = 24,
) -> PlantOrganArray:
    """Reverse diffusion sampling to generate a PlantOrganArray from noise."""
    device = image.device
    model.eval()

    B = 1
    N = model.max_nodes

    x_t = torch.randn(B, N, 94, device=device)

    step_indices = torch.linspace(scheduler.timesteps - 1, 0, steps, device=device).long()

    with torch.no_grad():
        for t in step_indices:
            t_batch = torch.tensor([t], device=device).long()
            outputs = model(x_t, t_batch, image.unsqueeze(0))
            pred_x0 = outputs["pred_x0"]

            alpha_t = scheduler.alphas_cumprod[t].clamp(min=1e-6)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-6))

            x_t = (x_t - sqrt_one_minus_alpha_t * pred_x0) / sqrt_alpha_t
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)

    organ_array = prediction_to_organ_array(pred_x0, dataset)
    exist = organ_array.tensor[:, -1]
    active = exist > 0.5
    num_active = int(active.sum().item())

    if num_active == 0:
        topk_values, topk_indices = torch.topk(exist, k=min(top_k_active, N))
        organ_array.tensor[topk_indices, -1] = 1.0
        num_active = min(top_k_active, N)
        print(f"No active nodes by threshold; forced top-{num_active} nodes active")
    else:
        print(f"Predicted {num_active} active nodes out of {N}")

    return organ_array


def prediction_to_organ_array(pred_x0: torch.Tensor, dataset: OrganArrayDataset) -> PlantOrganArray:
    assert pred_x0.shape[0] == 1
    denorm = dataset.denormalize(pred_x0[0])
    denorm[:, -1] = torch.sigmoid(denorm[:, -1])
    denorm[:, :93] = torch.clamp(denorm[:, :93], min=0.0)
    return PlantOrganArray(tensor=denorm.cpu())


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std + mean).clamp(0.0, 1.0)


def visualize_generation(
    target_image: torch.Tensor,
    predicted_organ_array: PlantOrganArray,
    renderer: HeliosPyTorchRenderer,
    save_path: str,
    device: torch.device,
):
    """Render predicted organ array and compare with target image."""
    try:
        rendered = renderer.render_organ_array(
            predicted_organ_array,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=1.0,
            background="ground",
            device=device,
            differentiable=False,
            focus_plant=True,
            existence_threshold=0.5,
        )
    except Exception as e:
        print(f"Rendering failed: {e}")
        rendered = torch.zeros(3, renderer.image_size, renderer.image_size, device=device)

    target_vis = denormalize_image(target_image).cpu()
    rendered_vis = rendered.detach().cpu().clamp(0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(target_vis.permute(1, 2, 0).numpy())
    axes[0].set_title("Target Image")
    axes[0].axis("off")

    axes[1].imshow(rendered_vis.permute(1, 2, 0).numpy())
    axes[1].set_title("Predicted Organ Array Rendered")
    axes[1].axis("off")

    diff = (target_vis - rendered_vis).abs()
    axes[2].imshow(diff.permute(1, 2, 0).numpy())
    axes[2].set_title(f"Absolute Diff (mean={diff.mean():.3f})")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="diffusion_based/checkpoints/organ_array_diffuser_norm.pt")
    parser.add_argument("--data_root", type=str, default="dataset/helios_data")
    parser.add_argument("--single_xml", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="diffusion_based/eval/output/organ_array_diffusion")
    parser.add_argument("--max_nodes", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    dataset = OrganArrayDataset(
        data_root=args.data_root,
        max_nodes=args.max_nodes,
        image_size=args.image_size,
        single_xml_path=args.single_xml,
    )

    model = PlantOrganArrayDiffuser(
        max_nodes=args.max_nodes,
        node_dim=94,
        embed_dim=256,
        num_layers=4,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")

    scheduler = DDPMScheduler(timesteps=1000)
    renderer = HeliosPyTorchRenderer(image_size=args.image_size).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        image = sample["image"].to(device)
        prefix = sample["prefix"]

        print(f"\nGenerating sample {i+1}/{args.num_samples}: {prefix}")
        organ_array = sample_organ_array(model, image, scheduler, dataset, steps=args.steps)

        xml_path = os.path.join(args.output_dir, f"{prefix}_pred.xml")
        organ_array.write_xml(xml_path)
        print(f"Saved predicted XML to {xml_path}")

        save_path = os.path.join(args.output_dir, f"{prefix}_compare.png")
        visualize_generation(image, organ_array, renderer, save_path, device)


if __name__ == "__main__":
    main()
