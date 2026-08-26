"""Quick integration test: GraphDiffuser3D 22D output -> DifferentiableHeliosRenderer."""
import os
import sys
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.legacy.graph_diffuser_3d_track_a import PlantGraphDiffuser3D
from diffusion_based.models.legacy.differentiable_pipeline_track_a import DifferentiableHeliosRenderer
from diffusion_based.models.legacy.helios_rasterizer_3d_track_a import HeliosGeometryRasterizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PlantGraphDiffuser3D(max_nodes=256, node_dim=22, embed_dim=128, num_layers=2, k_nearest=8).to(device)
    rasterizer = HeliosGeometryRasterizer(image_size=256).to(device)
    renderer = DifferentiableHeliosRenderer(rasterizer).to(device)

    B, N = 1, 128
    noisy_nodes = torch.randn(B, N, 22, device=device)
    existence = torch.rand(B, N, 1, device=device)
    t = torch.randint(0, 100, (B,), device=device)
    img = torch.randn(B, 3, 256, 256, device=device)

    out = model(noisy_nodes, existence, t, img)
    print("pred_x0 shape:", out["pred_x0"].shape)
    print("pred_organ_type_logits shape:", out["pred_organ_type_logits"].shape)
    print("pred_parent_logits shape:", out["pred_parent_logits"].shape)

    x0 = out["pred_x0"].clamp(0, 1)
    x0[..., 5:8] = x0[..., 5:8] / (x0[..., 5:8].norm(dim=-1, keepdim=True) + 1e-8)

    with torch.no_grad():
        rgba = renderer(x0, focus_plant=True, background="black")
    print("rendered rgba shape:", rgba.shape)
    print("Integration test passed: 22D GraphDiffuser3D -> DifferentiableHeliosRenderer works.")


if __name__ == "__main__":
    main()
