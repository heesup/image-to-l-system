"""Compare _leaflet_local_mesh (numpy) vs _leaflet_local_mesh_torch (torch)."""
import os
import sys
import numpy as np
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.legacy.helios_geometry_track_a import _leaflet_local_mesh_torch
from diffusion_based.models.legacy.helios_geometry_legacy import _leaflet_local_mesh


def main():
    device = torch.device("cpu")
    np_verts, np_faces = _leaflet_local_mesh(1.0, aspect=0.7)
    torch_verts, torch_faces = _leaflet_local_mesh_torch(device=device, subdivisions=8, aspect=0.7)

    print(f"numpy verts: {np_verts.shape}, torch verts: {torch_verts.shape}")
    print(f"numpy faces: {np_faces.shape}, torch faces: {torch_faces.shape}")

    t_verts = torch_verts.cpu().numpy()
    diff = np.abs(np_verts - t_verts)
    print(f"max vertex diff: {diff.max():.8f}")
    print(f"mean vertex diff: {diff.mean():.8f}")

    f_diff = np.abs(np_faces - torch_faces.cpu().numpy())
    print(f"max face diff: {f_diff.max()}")

    # Print first 5 vertices
    print("\nFirst 5 numpy vertices:")
    print(np_verts[:5])
    print("\nFirst 5 torch vertices:")
    print(t_verts[:5])


if __name__ == "__main__":
    main()
