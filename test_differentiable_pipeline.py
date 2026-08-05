"""Quick integration test for the differentiable Helios pipeline."""

import torch
import torch.nn as nn

from diffusion_based.models.helios_geometry import nodes_to_geometry_torch
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer
from diffusion_based.models.differentiable_pipeline import DifferentiableHeliosRenderer


def test_no_syntax_and_imports():
    print("[PASS] Imports succeeded – no syntax errors.")


def test_forward_pass():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 2, 5

    # Build random 15D nodes
    nodes = torch.randn(B, N, 15, device=device, requires_grad=True)
    # Set organ logits to something deterministic (soft one-hot-ish)
    nodes.data[:, :, 8:12] = 0.0
    nodes.data[:, 0, 8] = 5.0   # internode
    nodes.data[:, 1, 9] = 5.0   # petiole
    nodes.data[:, 2, 10] = 5.0  # leaf
    nodes.data[:, 3, 11] = 5.0  # floral_bud
    nodes.data[:, 4, 8] = 5.0   # internode
    # Positive existence so geometry is generated
    nodes.data[:, :, 14] = 1.0
    # Positive lengths / radii
    nodes.data[:, :, 3] = nodes.data[:, :, 3].abs().clamp(min=0.1)
    nodes.data[:, :, 4] = nodes.data[:, :, 4].abs().clamp(min=0.01)

    rasterizer = HeliosGeometryRasterizer(image_size=128).to(device)
    renderer = DifferentiableHeliosRenderer(rasterizer).to(device)

    img = renderer(nodes)
    assert img.shape == (B, 4, 128, 128), f"Expected (B,4,128,128), got {img.shape}"
    assert not torch.isnan(img).any(), "NaN in output"
    assert not torch.isinf(img).any(), "Inf in output"
    print(f"[PASS] Forward pass OK – shape {tuple(img.shape)}, min={img.min():.4f}, max={img.max():.4f}")
    return img, nodes


def test_gradient_flow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 1, 4

    nodes = torch.randn(B, N, 15, device=device, requires_grad=True)
    nodes.data[:, :, 8:12] = 0.0
    nodes.data[:, 0, 8] = 5.0
    nodes.data[:, 1, 9] = 5.0
    nodes.data[:, 2, 10] = 5.0
    nodes.data[:, 3, 11] = 5.0
    nodes.data[:, :, 14] = 1.0
    nodes.data[:, :, 3] = nodes.data[:, :, 3].abs().clamp(min=0.1)
    nodes.data[:, :, 4] = nodes.data[:, :, 4].abs().clamp(min=0.01)

    rasterizer = HeliosGeometryRasterizer(image_size=128).to(device)
    renderer = DifferentiableHeliosRenderer(rasterizer).to(device)

    img = renderer(nodes)
    loss = img.mean()
    loss.backward()

    assert nodes.grad is not None, "No gradient reached nodes"
    grad_norm = nodes.grad.norm().item()
    assert grad_norm > 0, f"Zero gradient on nodes (norm={grad_norm})"
    assert not torch.isnan(nodes.grad).any(), "NaN in node gradients"
    print(f"[PASS] Gradients flow end-to-end – node.grad.norm = {grad_norm:.6f}")


def test_nodes_to_geometry_torch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 1, 4
    nodes = torch.randn(B, N, 15, device=device, requires_grad=True)
    nodes.data[:, :, 8:12] = 0.0
    nodes.data[:, 0, 8] = 5.0
    nodes.data[:, 1, 9] = 5.0
    nodes.data[:, 2, 10] = 5.0
    nodes.data[:, 3, 11] = 5.0
    nodes.data[:, :, 14] = 1.0
    nodes.data[:, :, 3] = nodes.data[:, :, 3].abs().clamp(min=0.1)
    nodes.data[:, :, 4] = nodes.data[:, :, 4].abs().clamp(min=0.01)

    (
        tube_verts,
        tube_radii,
        tube_organs,
        leaf_verts,
        leaf_faces,
        leaf_organs,
        bud_centers,
        bud_radii,
        bud_lengths,
        bud_organs,
    ) = nodes_to_geometry_torch(nodes)

    assert tube_verts.shape == (B, N, 2, 3)
    assert tube_radii.shape == (B, N, 2)
    assert leaf_verts.shape[0] == B and leaf_verts.shape[1] == N
    assert leaf_faces.dim() == 2 and leaf_faces.shape[1] == 3
    assert bud_centers.shape == (B, N, 3)
    print("[PASS] nodes_to_geometry_torch shapes OK")


if __name__ == "__main__":
    test_no_syntax_and_imports()
    test_nodes_to_geometry_torch()
    test_forward_pass()
    test_gradient_flow()
    print("\n=== All tests passed! ===")
