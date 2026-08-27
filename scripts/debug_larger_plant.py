import os, sys, math
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF,
    T_COL_ORGAN_TYPE, T_COL_LENGTH, T_COL_RADIUS,
    T_COL_SCALE, T_COL_PITCH, T_COL_YAW, T_COL_ROLL,
    T_COL_EXISTENCE,
)
from diffusion_based.models.helios_pytorch_renderer import HeliosPyTorchRenderer

device = torch.device("cuda")

def make_plant(inode_len, pet_pitch, device):
    t = torch.zeros((3, 40), dtype=torch.float32, device=device)
    t[0, T_COL_ORGAN_TYPE] = ORGAN_INTERNODE
    t[0, T_COL_LENGTH] = inode_len
    t[0, T_COL_RADIUS] = 0.02
    t[0, T_COL_SCALE] = 1.0
    t[0, T_COL_PITCH] = 1.0
    t[0, T_COL_EXISTENCE] = 1.0

    t[1, T_COL_ORGAN_TYPE] = ORGAN_PETIOLE
    t[1, T_COL_LENGTH] = 0.18
    t[1, T_COL_RADIUS] = 0.01
    t[1, T_COL_SCALE] = 1.0
    t[1, T_COL_PITCH] = pet_pitch
    t[1, T_COL_EXISTENCE] = 1.0

    t[2, T_COL_ORGAN_TYPE] = ORGAN_LEAF
    t[2, T_COL_LENGTH] = 0.22
    t[2, T_COL_RADIUS] = 0.11
    t[2, T_COL_SCALE] = 1.0
    t[2, T_COL_PITCH] = 15.0
    t[2, T_COL_YAW] = 10.0
    t[2, T_COL_EXISTENCE] = 1.0
    return PlantOrganArray(t)

for inode_len, pet_pitch in [(0.25, 60.0), (0.08, 60.0), (0.25, 10.0)]:
    arr = make_plant(inode_len, pet_pitch, device)
    renderer = HeliosPyTorchRenderer(image_size=256).to(device)
    mesh = renderer.geo_builder.build_mesh_from_part_tensor(arr.to_part_tensor(device=device), device=device)
    rgbd = renderer.forward(mesh, elevation_deg=89.88, camera_height=4.0, focus_plant=True, include_depth=True)
    rgb = rgbd[:3].permute(1,2,0).cpu().numpy()
    depth = rgbd[3].cpu().numpy()
    print(f"inode={inode_len} pitch={pet_pitch}: verts max z={mesh['vertices'][:,2].max():.3f} depth max={depth.max():.3f}")
