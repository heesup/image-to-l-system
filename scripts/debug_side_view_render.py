import os, sys
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

renderer = HeliosPyTorchRenderer(image_size=256).to(device)

for az, el in [(0.0, 20.0), (90.0, 20.0), (45.0, 20.0)]:
    arr = make_plant(0.25, 60.0, device)
    mesh = renderer.geo_builder.build_mesh_from_organ_array(arr, device=device, species="cowpea")
    rgbd = renderer.forward(mesh, elevation_deg=el, azimuth_deg=az, camera_height=4.0, focus_plant=True, include_depth=True)
    rgb = rgbd[:3].permute(1,2,0).cpu().numpy()
    depth = rgbd[3].cpu().numpy()
    print(f"az={az} el={el}: depth max={depth.max():.3f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,1)
    ax.imshow(rgb)
    ax.set_title(f"az={az} el={el}")
    ax.axis("off")
    plt.savefig(f"/tmp/opencode/side_view_az{az:.0f}_el{el:.0f}.png")
    plt.close()
    print(f"saved /tmp/opencode/side_view_az{az:.0f}_el{el:.0f}.png")
