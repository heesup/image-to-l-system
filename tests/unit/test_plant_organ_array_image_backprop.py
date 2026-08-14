"""
Unit test: Image-loss backpropagation through the PlantOrganArray renderer.

Renders a PlantOrganArray through the PyTorch differentiable renderer,
computes a simple image MSE loss against a target, and checks that gradients
flow back to the organ-array tensor. Runs for both the legacy (N, 94) and
typed (N, 40) layouts.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def run_image_backprop_test(xml_path: str, image_size: int = 128, max_steps: int = 20,
                            use_typed_layout: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout_name = "typed (N, 40)" if use_typed_layout else "legacy (N, 94)"
    print(f"Device: {device}")
    print(f"Layout: {layout_name}")

    print(f"Loading PlantOrganArray from {xml_path}...")
    if use_typed_layout:
        organ_array = PlantOrganArray.from_xml_file_typed(xml_path)
    else:
        organ_array = PlantOrganArray.from_xml_file(xml_path)
    organ_array.tensor = organ_array.tensor.to(device).clone().detach().requires_grad_(True)

    renderer = HeliosPyTorchRenderer(image_size=image_size)

    # Render a target image from the initial (unmodified) array so that the
    # optimization has a non-trivial but reachable goal.
    with torch.no_grad():
        target_img = renderer.render_organ_array(
            organ_array,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=1.0,
            background="ground",
            device=device,
            differentiable=True,
        )

    optimizer = torch.optim.Adam([organ_array.tensor], lr=1e-3)

    print(f"Running image-backprop optimization for up to {max_steps} steps...")
    for step in range(max_steps):
        optimizer.zero_grad()
        rendered = renderer.render_organ_array(
            organ_array,
            azimuth_deg=0.0,
            elevation_deg=90.0,
            camera_height=1.0,
            background="ground",
            device=device,
            differentiable=True,
        )
        loss = F.mse_loss(rendered, target_img)
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == max_steps - 1:
            grad = organ_array.tensor.grad
            grad_norm = grad.norm().item() if grad is not None else 0.0
            non_zero = (grad != 0.0).sum().item() if grad is not None else 0
            print(f"  step {step:03d}  loss={loss.item():.6f}  grad_norm={grad_norm:.6f}  non_zero={non_zero}/{grad.numel() if grad is not None else 0}")

    grad = organ_array.tensor.grad
    assert grad is not None, "Gradient was None — renderer is not differentiable w.r.t. PlantOrganArray.tensor"
    assert grad.abs().max() > 0, "All gradients are zero"

    print("\nSUCCESS: image-loss gradients flow back to PlantOrganArray.tensor")
    print(f"  layout            : {layout_name}")
    print(f"  final loss        : {loss.item():.6f}")
    print(f"  grad norm         : {grad.norm().item():.6f}")
    print(f"  non-zero gradients: {(grad != 0.0).sum().item()}/{grad.numel()}")


if __name__ == "__main__":
    default_xml = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Digital-Crops",
        "projects",
        "syntheticdata_generation",
        "build",
        "output",
        "cowpea_dap005_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml",
    )
    print("=== Legacy layout ===")
    run_image_backprop_test(default_xml)
    print("\n=== Typed layout ===")
    run_image_backprop_test(default_xml, use_typed_layout=True)
