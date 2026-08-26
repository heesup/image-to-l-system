"""
Test script verifying PyTorch Image Loss Backpropagation to PlantOrganArray Tensor (N, 93).
Demonstrates end-to-end differentiability from rendered RGB pixels back to organ architecture parameters.
"""

import torch
from diffusion_based.models.plant_organ_array import PlantOrganArray
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer


def run_differentiable_backprop_test():
    xml_path = "Digital-Crops/projects/syntheticdata_generation/build/output/cowpea_dap005_seed00_caz000_h1.0_se045_saz180_0000_plant_0000.xml"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading Organ Array Tensor from {xml_path}...")
    organ_array = PlantOrganArray.from_xml_file(xml_path)

    # Enable autograd gradients on Organ Array Tensor (N, 93)
    organ_array.tensor = organ_array.tensor.to(device).clone().detach().requires_grad_(True)

    renderer = HeliosPyTorchRenderer(image_size=128)

    # Render image with soft differentiable rasterizer
    print(f"Rendering plant with Soft Differentiable Rasterizer (Device: {device})...")
    rendered_img = renderer.render_organ_array(
        organ_array, azimuth_deg=0.0, elevation_deg=90.0, camera_height=1.0, background="ground", device=device, differentiable=True
    ) # (3, 128, 128)

    # Define dummy target image loss
    target_img = torch.zeros_like(rendered_img)
    loss = F.mse_loss(rendered_img, target_img)

    print(f"Computed Image MSE Loss: {loss.item():.6f}")

    # Backpropagation from Image Loss -> Organ Array Tensor
    loss.backward()

    grad = organ_array.tensor.grad
    if grad is not None:
        grad_norm = grad.norm().item()
        non_zero_cnt = (grad != 0.0).sum().item()
        print("\n" + "=" * 50)
        print("   DIFFERENTIABLE IMAGE BACKPROPAGATION SUCCESS   ")
        print("=" * 50)
        print(f"  Organ Array Tensor Grad Norm  : {grad_norm:.6f}")
        print(f"  Non-zero Gradient Channels     : {non_zero_cnt} / {grad.numel()}")
        print("=" * 50 + "\n")
    else:
        print("FAIL: Gradient was None!")


if __name__ == "__main__":
    import torch.nn.functional as F
    run_differentiable_backprop_test()
