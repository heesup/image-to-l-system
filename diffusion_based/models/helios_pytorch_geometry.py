"""
PyTorch Geometry Generator for Helios Plant Architecture.
Converts PlantOrganArray Tensor (N, 93) directly into 3D meshes (internode tubes, petiole tubes, compound leaf meshes, flowers).
Supports both quad-triangulated 3D OBJ leaf assets and Helios GenericLeafPrototype parametric 15cm base leaf meshes.
"""

import os
import sys
import math
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
    COL_PLANT_ID, COL_PLANT_AGE, COL_SHOOT_ID, COL_SHOOT_TYPE,
    COL_PARENT_SHOOT_ID, COL_PARENT_NODE_IDX, COL_PARENT_PETIOLE_IDX,
    COL_SHOOT_ROT_PITCH, COL_SHOOT_ROT_YAW, COL_SHOOT_ROT_ROLL, COL_PHYTOMER_IDX,
    COL_INODE_LEN, COL_INODE_RAD, COL_INODE_PITCH, COL_INODE_PHYLLO_ANG,
    COL_INODE_LEN_MAX, COL_INODE_LEN_SEGS, COL_CURV_PERT_0, COL_CURV_PERT_1,
    COL_YAW_PERT_0, COL_YAW_PERT_1,
    COL_PET0_LEN, COL_PET0_RAD, COL_PET0_PITCH, COL_PET0_CURV, COL_PET0_LEAF_SCALE,
    COL_PET0_TAPER, COL_PET0_LEN_SEGS, COL_PET0_RAD_SUBDIV, COL_PET0_LFLT_SCALE,
    COL_PET0_LFLT_OFFSET, COL_PET0_NUM_LEAVES,
    COL_PET0_L0_SCALE, COL_PET0_L0_PITCH, COL_PET0_L0_YAW, COL_PET0_L0_ROLL,
    COL_PET0_L1_SCALE, COL_PET0_L1_PITCH, COL_PET0_L1_YAW, COL_PET0_L1_ROLL,
    COL_PET0_L2_SCALE, COL_PET0_L2_PITCH, COL_PET0_L2_YAW, COL_PET0_L2_ROLL,
    COL_HAS_PET1, COL_PET1_LEN, COL_PET1_RAD, COL_PET1_PITCH, COL_PET1_CURV, COL_PET1_LEAF_SCALE,
    COL_PET1_TAPER, COL_PET1_LEN_SEGS, COL_PET1_RAD_SUBDIV, COL_PET1_LFLT_SCALE,
    COL_PET1_LFLT_OFFSET, COL_PET1_NUM_LEAVES,
    COL_PET1_L0_SCALE, COL_PET1_L0_PITCH, COL_PET1_L0_YAW, COL_PET1_L0_ROLL,
    COL_HAS_BUD, COL_BUD_STATE, COL_PED_LEN, COL_PED_RAD, COL_PED_PITCH, COL_PED_CURV, COL_PED_ROLL,
    COL_NUM_FLOWERS, COL_FLOWER_OFFSET, COL_FL0_PITCH, COL_FL0_YAW, COL_FL0_ROLL, COL_FL0_AZIMUTH, COL_FL0_BASE_SCALE,
    T_COL_PLANT_ID, T_COL_PLANT_AGE, T_COL_SHOOT_ID,
    T_COL_BASE_X, T_COL_BASE_Y, T_COL_BASE_Z,
    T_COL_PARENT_SHOOT_ID, T_COL_PARENT_NODE_IDX, T_COL_PARENT_PETIOLE_IDX,
    T_COL_PHYTOMER_IDX, T_COL_CHILD_INDEX, T_COL_ORGAN_TYPE,
    T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE, T_COL_PITCH, T_COL_YAW, T_COL_ROLL,
    T_COL_CURVATURE, T_COL_PHYLLOTACTIC_ANGLE, T_COL_LENGTH_MAX, T_COL_LENGTH_SEGMENTS,
    T_COL_CURV_PERT_0, T_COL_CURV_PERT_1, T_COL_YAW_PERT_0, T_COL_YAW_PERT_1,
    T_COL_CURRENT_LEAF_SCALE_FACTOR, T_COL_TAPER, T_COL_RADIAL_SUBDIVISIONS,
    T_COL_LEAFLET_SCALE, T_COL_LEAFLET_OFFSET,
    T_COL_BUD_STATE, T_COL_BUD_PARENT_INDEX, T_COL_BUD_IS_TERMINAL, T_COL_FRUIT_SCALE,
    T_COL_FLOWER_AZIMUTH, T_COL_FLOWER_OFFSET, T_COL_EXISTENCE,
    ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF,
    ORGAN_BUD, ORGAN_PEDUNCLE, ORGAN_FLOWER,
    P14_COL_ORGAN_TYPE, P14_COL_BASE_X, P14_COL_BASE_Y, P14_COL_BASE_Z,
    P14_COL_ROT_0, P14_COL_ROT_1, P14_COL_ROT_2, P14_COL_ROT_3, P14_COL_ROT_4, P14_COL_ROT_5,
    P14_COL_SCALE_X, P14_COL_SCALE_Y, P14_COL_SCALE_Z, P14_COL_EXISTENCE, NUM_FEATURES_14D,
    rotation_matrix_to_6d, rotation_6d_to_matrix,
)


ASSET_DIR = "/home/lion397/codes/image-to-l-system/Digital-Crops/libs/Helios/plugins/plantarchitecture/assets/obj"


def load_obj_file(filepath: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads an OBJ file into PyTorch vertex tensor (V, 3) and face index tensor (F, 3) with quad triangulation and ZUP conversion."""
    vertices = []
    faces = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()[1:4]
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx_list = [int(p.split("/")[0]) - 1 for p in parts]
                for i_t in range(1, len(idx_list) - 1):
                    faces.append([idx_list[0], idx_list[i_t], idx_list[i_t + 1]])

    verts_raw = torch.tensor(vertices, dtype=torch.float32)
    faces_t = torch.tensor(faces, dtype=torch.int64)

    # These CowpeaLeaf OBJs are already in Helios Z-up convention:
    #   +x = midrib length, +y = blade width, +z = curvature/normal.
    return verts_raw, faces_t


def generate_generic_leaf_mesh_torch(
    scale: torch.Tensor,
    aspect_ratio: float = 0.65,
    Nx: int = 8,
    Ny: int = 8,
    device=torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates Helios GenericLeafPrototype parametric curved leaf mesh."""
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=torch.float32, device=device)

    width = scale * aspect_ratio
    xs = torch.linspace(0, 1.0, Nx + 1, device=device)
    ys = torch.linspace(-0.5, 0.5, Ny + 1, device=device)

    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')

    curve_x = torch.sin(math.pi * grid_x)
    curve_y = torch.cos(math.pi * grid_y)
    grid_z = -0.15 * curve_x * curve_y

    pts_x = grid_x * scale
    pts_y = grid_y * width
    pts_z = grid_z * scale

    verts = torch.stack([pts_x, pts_y, pts_z], dim=-1).reshape(-1, 3)

    faces = []
    for i in range(Nx):
        for j in range(Ny):
            v0 = i * (Ny + 1) + j
            v1 = i * (Ny + 1) + j + 1
            v2 = (i + 1) * (Ny + 1) + j
            v3 = (i + 1) * (Ny + 1) + j + 1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    faces_t = torch.tensor(faces, dtype=torch.int64, device=device)
    return verts, faces_t


_TUBE_PROTO_CACHE: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

def _get_cached_tube_prototype(n_seg: int, n_rad: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (n_seg, n_rad, str(device))
    if key not in _TUBE_PROTO_CACHE:
        _TUBE_PROTO_CACHE[key] = _make_straight_tube_prototype(n_seg, n_rad, device)
    return _TUBE_PROTO_CACHE[key]


_GENERIC_LEAF_CACHE: Dict[Tuple[float, int, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

def get_generic_leaf_template(aspect_ratio: float = 0.65, Nx: int = 8, Ny: int = 8, device=torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (aspect_ratio, Nx, Ny, str(device))
    if key not in _GENERIC_LEAF_CACHE:
        v_temp, f_temp = generate_generic_leaf_mesh_torch(scale=torch.tensor(1.0, device=device), aspect_ratio=aspect_ratio, Nx=Nx, Ny=Ny, device=device)
        n_temp = compute_face_normals_torch(v_temp, f_temp)
        _GENERIC_LEAF_CACHE[key] = (v_temp, f_temp, n_temp)
    return _GENERIC_LEAF_CACHE[key]


class HeliosAssetManager:
    """Loads and caches Helios OBJ assets for PyTorch rendering."""
    def __init__(self, asset_dir: str = ASSET_DIR):
        self.asset_dir = asset_dir
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.mesh_normals_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get_mesh(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if name not in self.cache:
            path = os.path.join(self.asset_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Helios asset missing: {path}")
            self.cache[name] = load_obj_file(path)
        v, f = self.cache[name]
        return v.clone(), f.clone()

    def get_mesh_with_normals(self, name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (name, str(device))
        if key not in self.mesh_normals_cache:
            v, f = self.get_mesh(name)
            v = v.to(device)
            f = f.to(device)
            n = compute_face_normals_torch(v, f)
            self.mesh_normals_cache[key] = (v, f, n)
        v, f, n = self.mesh_normals_cache[key]
        return v, f, n

    def get_inflorescence_mesh(self, name: str, load_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load an inflorescence asset with Helios loadOBJ(scale) semantics.

        scale 0  -> all axes 1 (flowers)
        scale>0  -> uniform scale.z / (z-extent) baked into the mesh (pod,
                    matching Context::loadOBJ box scaling on the z-axis).
        """
        v, f = self.get_mesh(name)
        if load_scale > 0:
            zmin = v[:, 2].min()
            zmax = v[:, 2].max()
            extent = zmax - zmin
            if extent > 1e-6:
                v = v * (load_scale / extent)
            else:
                v = v * load_scale
        return v, f


def rotr_x(angle_rad: torch.Tensor, device=torch.device('cpu')) -> torch.Tensor:
    if not isinstance(angle_rad, torch.Tensor):
        angle_rad = torch.tensor(angle_rad, dtype=torch.float32, device=device)
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    R = torch.eye(3, dtype=torch.float32, device=angle_rad.device)
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 1] = s
    R[2, 2] = c
    return R

def rotr_y(angle_rad: torch.Tensor, device=torch.device('cpu')) -> torch.Tensor:
    if not isinstance(angle_rad, torch.Tensor):
        angle_rad = torch.tensor(angle_rad, dtype=torch.float32, device=device)
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    R = torch.eye(3, dtype=torch.float32, device=angle_rad.device)
    R[0, 0] = c
    R[0, 2] = s
    R[2, 0] = -s
    R[2, 2] = c
    return R

def rotr_z(angle_rad: torch.Tensor, device=torch.device('cpu')) -> torch.Tensor:
    if not isinstance(angle_rad, torch.Tensor):
        angle_rad = torch.tensor(angle_rad, dtype=torch.float32, device=device)
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    R = torch.eye(3, dtype=torch.float32, device=angle_rad.device)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def get_rotation_matrix_between_vectors(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Computes exact 3x3 rotation matrix transforming unit vector v0 to unit vector v1."""
    device = v0.device
    v0_norm = v0 / (torch.linalg.norm(v0) + 1e-8)
    v1_norm = v1 / (torch.linalg.norm(v1) + 1e-8)
    cos_theta = torch.dot(v0_norm, v1_norm).clamp(-1.0, 1.0)

    if cos_theta > 0.9999:
        return torch.eye(3, dtype=torch.float32, device=device)
    elif cos_theta < -0.9999:
        return -torch.eye(3, dtype=torch.float32, device=device)

    axis = torch.linalg.cross(v0_norm, v1_norm)
    axis = axis / (torch.linalg.norm(axis) + 1e-8)
    sin_theta = torch.sqrt((1.0 - cos_theta**2).clamp(min=1e-10))

    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    K = torch.stack([
        torch.stack([zero, -axis[2], axis[1]]),
        torch.stack([axis[2], zero, -axis[0]]),
        torch.stack([-axis[1], axis[0], zero])
    ])

    R = torch.eye(3, dtype=torch.float32, device=device) + sin_theta * K + (1.0 - cos_theta) * (K @ K)
    return R


def rotate_vector_about_axis(vec: torch.Tensor, axis: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation formula: rotates 3D vector 'vec' about unit vector 'axis' by 'angle_rad'."""
    if torch.abs(angle_rad) < 1e-6:
        return vec
    axis_sq = (axis * axis).sum()
    if axis_sq < 1e-8:
        return vec
    if abs(axis_sq.item() - 1.0) > 1e-4:
        axis = axis / torch.sqrt(axis_sq.clamp(min=1e-10))

    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    cross_av = torch.linalg.cross(axis, vec)
    dot_av = (axis * vec).sum()
    return vec * cos_a + cross_av * sin_a + axis * dot_av * (1.0 - cos_a)


def rotate_points_about_axis(points: torch.Tensor, axis: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation of a batch of points (V,3) about a single unit axis by angle_rad."""
    axis_norm = torch.linalg.norm(axis)
    if axis_norm < 1e-4:
        return points.clone()
    axis = axis / axis_norm
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    if cos_a.ndim > 0:
        cos_a = cos_a.unsqueeze(-1)
        sin_a = sin_a.unsqueeze(-1)
    axis_b = axis.unsqueeze(0).expand(points.shape[0], 3)
    return points * cos_a + torch.linalg.cross(axis_b, points) * sin_a + axis_b * (axis * points).sum(-1, keepdim=True) * (1.0 - cos_a)


def rodrigues_matrix_torch(axis: torch.Tensor, angle_rad: torch.Tensor, device=torch.device('cpu')) -> torch.Tensor:
    """Return 3x3 rotation matrix for Rodrigues rotation about unit vector 'axis' by 'angle_rad'."""
    if not isinstance(angle_rad, torch.Tensor):
        angle_rad = torch.tensor(angle_rad, dtype=torch.float32, device=device)
    axis = axis / (torch.linalg.norm(axis) + 1e-8)
    c = torch.cos(angle_rad)
    s = torch.sin(angle_rad)
    x, y, z = axis[0], axis[1], axis[2]
    return torch.stack([
        torch.stack([c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s]),
        torch.stack([y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s]),
        torch.stack([z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]),
    ])


def interpolate_tube_torch(vertices: torch.Tensor, frac: float) -> torch.Tensor:
    """Interpolate along a polyline by fraction from base (0) to tip (1)."""
    n = vertices.shape[0] - 1
    if n <= 0 or frac <= 0:
        return vertices[0].clone()
    if frac >= 1.0:
        return vertices[-1].clone()
    pos = frac * n
    idx = int(pos)
    t = pos - idx
    if idx >= n:
        return vertices[-1].clone()
    return (1.0 - t) * vertices[idx] + t * vertices[idx + 1]


def get_axis_vector_torch(vertices: torch.Tensor, stem_fraction: float) -> torch.Tensor:
    """Port of Phytomer::getAxisVector (PlantArchitecture.cpp:435).

    Computes a unit axis at `stem_fraction` along the tube. Uses df=0.1 for the
    finite-difference secant. The C++ interpolateTube() is arc-length based, but
    all tubes built here (internode, petiole, peduncle) have equal-length
    segments, so the index-based interpolate_tube_torch() matches exactly.
    """
    df = 0.1
    if stem_fraction + df <= 1.0:
        frac_minus = stem_fraction
        frac_plus = stem_fraction + df
    else:
        frac_minus = stem_fraction - df
        frac_plus = stem_fraction

    node_minus = interpolate_tube_torch(vertices, frac_minus)
    node_plus = interpolate_tube_torch(vertices, frac_plus)
    axis = node_plus - node_minus
    n = torch.linalg.norm(axis)
    if n < 1e-12:
        return axis.clone()
    return axis / n


def clamp_offset_torch(count_per_axis: int, offset: float) -> float:
    """Port of clampOffset (PlantArchitecture.cpp:57)."""
    if count_per_axis > 2:
        denom = 0.5 * float(count_per_axis) - 1.0
        if offset * denom > 1.0:
            offset = 1.0 / denom
    return offset


def compute_face_normals_torch(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute per-vertex normals by area-weighted accumulation of face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = torch.linalg.cross(v1 - v0, v2 - v0)  # (F, 3)
    flat_idx = faces.flatten()                  # (3F,)
    flat_normals = fn.repeat_interleave(3, dim=0)  # (3F, 3)
    normals = torch.zeros_like(vertices)
    normals.index_add_(0, flat_idx, flat_normals)
    normals = normals / (torch.linalg.norm(normals, dim=-1, keepdim=True) + 1e-8)
    return normals


def _get_rotation_matrix_between_vectors_batch(
    v0: torch.Tensor,  # (B, 3) unit
    v1: torch.Tensor,  # (B, 3) unit
) -> torch.Tensor:
    """Batched rotation matrix mapping each v0 to v1 (Rodrigues)."""
    B = v0.shape[0]
    device = v0.device
    cos = (v0 * v1).sum(dim=-1).clamp(-1.0, 1.0)  # (B,)
    sin = torch.sqrt((1.0 - cos**2).clamp(min=0.0))  # (B,)
    axis = torch.linalg.cross(v0, v1)
    axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + 1e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    K = torch.zeros(B, 3, 3, device=device)
    K[:, 0, 1] = -z
    K[:, 0, 2] = y
    K[:, 1, 0] = z
    K[:, 1, 2] = -x
    K[:, 2, 0] = -y
    K[:, 2, 1] = x

    I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
    R = I + sin.unsqueeze(-1).unsqueeze(-1) * K + (1.0 - cos).unsqueeze(-1).unsqueeze(-1) * (K @ K)
    return R


def _make_straight_tube_prototype(n_seg: int, n_rad: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a unit-length straight tube prototype aligned with +Z.

    Prototype vertices have x,y on a unit circle and z in [0,1]. Callers scale
    x,y by the desired radius and z by the desired length, then rotate +Z to
    the segment axis.
    """
    z = torch.linspace(0.0, 1.0, n_seg + 1, device=device)
    angles = torch.linspace(0.0, 2.0 * math.pi, n_rad + 1, device=device)[:-1]

    verts = []
    for zi in z:
        for a in angles:
            verts.append(torch.stack([torch.cos(a), torch.sin(a), zi]))
    verts_t = torch.stack(verts)  # ((n_seg+1)*n_rad, 3)

    faces_list = []
    for i in range(n_seg):
        for j in range(n_rad):
            j_next = (j + 1) % n_rad
            v0 = i * n_rad + j
            v1 = i * n_rad + j_next
            v2 = (i + 1) * n_rad + j
            v3 = (i + 1) * n_rad + j_next
            faces_list.append([v0, v2, v1])
            faces_list.append([v1, v2, v3])
    faces_t = torch.tensor(faces_list, dtype=torch.int64, device=device)
    return verts_t, faces_t


def generate_cone_tube_mesh_torch(
    centerline: torch.Tensor, # (N, 3)
    radii: torch.Tensor,      # (N,)
    color: torch.Tensor,      # (3,)
    radial_subdivisions: int = 6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prototype-based tube mesh generator.

    Internally builds one straight tube prototype and transforms it to match the
    requested centerline. This avoids per-vertex Python loops and is much faster
    than the previous per-ring construction.
    """
    N = centerline.shape[0]
    device = centerline.device
    if N < 2:
        empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
        empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
        return empty3, empty_f, empty3, empty3

    n_seg = N - 1
    n_rad = max(3, radial_subdivisions)
    proto_v, faces_t = _get_cached_tube_prototype(n_seg, n_rad, device)
    # proto_v: (V,3) with x,y on unit circle, z in [0,1]

    z = proto_v[:, 2]                       # (V,)
    xy = proto_v[:, :2]                       # (V, 2)
    V = proto_v.shape[0]

    zv = z
    ring_idx = torch.arange(n_seg + 1, device=device).repeat_interleave(n_rad)  # (V,)
    seg_idx = ring_idx.clamp(max=n_seg - 1)   # last ring belongs to last segment
    r0 = radii[seg_idx]                       # (V,)
    r1 = radii[(seg_idx + 1).clamp(max=N - 1)] # (V,)
    r_interp = r0 * (1.0 - zv) + r1 * zv      # (V,)

    # Build world positions by linearly interpolating centerline along zv
    seg_idx_v = (zv * n_seg).long().clamp(max=n_seg - 1)  # (V,)
    t = (zv * n_seg) - seg_idx_v.float()                 # (V,)
    p0 = centerline[seg_idx_v]                            # (V, 3)
    p1 = centerline[seg_idx_v + 1]                        # (V, 3)
    pos = p0 + t.unsqueeze(-1) * (p1 - p0)                # (V, 3)

    # Per-segment rotation from +Z to segment axis using direct orthonormal frame
    axis = centerline[1:] - centerline[:-1]               # (n_seg, 3)
    L = torch.linalg.norm(axis, dim=-1, keepdim=True)
    axis_norm = axis / (L + 1e-8)
    seg_axis = axis_norm[seg_idx]                         # (V, 3)

    ax, ay, az = seg_axis[:, 0], seg_axis[:, 1], seg_axis[:, 2]
    xy_norm = torch.sqrt((ax * ax + ay * ay).clamp(min=1e-10))
    ux = torch.where(xy_norm > 1e-3, -ay / xy_norm, torch.ones_like(ax))
    uy = torch.where(xy_norm > 1e-3, ax / xy_norm, torch.zeros_like(ay))
    uz = torch.zeros_like(az)
    u = torch.stack([ux, uy, uz], dim=-1)

    vx = ay * uz - az * uy
    vy = az * ux - ax * uz
    vz = ax * uy - ay * ux
    v = torch.stack([vx, vy, vz], dim=-1)

    x_s = xy[:, 0:1]
    y_s = xy[:, 1:2]
    normals = x_s * u + y_s * v
    offsets = (r_interp.unsqueeze(-1) * x_s) * u + (r_interp.unsqueeze(-1) * y_s) * v

    verts_t = pos + offsets
    colors_t = color.unsqueeze(0).expand(verts_t.shape[0], 3)
    return verts_t, faces_t, normals, colors_t


class HeliosPlantGeometryBuilder:
    """Builds complete PyTorch 3D plant meshes directly from PlantOrganArray Tensor (N, 93)."""

    def __init__(self, asset_manager: Optional[HeliosAssetManager] = None, use_generic_leaves: bool = False, leaf_scale_factor: float = 1.0, tube_radial_subdivisions: int = 4):
        if asset_manager is None:
            asset_manager = HeliosAssetManager()
        self.asset_mgr = asset_manager
        self.use_generic_leaves = use_generic_leaves
        self.leaf_scale_factor = leaf_scale_factor
        self.tube_radial_subdivisions = tube_radial_subdivisions

        self.COLOR_STEM = torch.tensor([0.22, 0.45, 0.15], dtype=torch.float32)
        self.COLOR_PETIOLE = torch.tensor([0.25, 0.50, 0.18], dtype=torch.float32)
        self.COLOR_LEAF = torch.tensor([0.25, 0.62, 0.18], dtype=torch.float32)
        self.COLOR_FLOWER = torch.tensor([0.95, 0.85, 0.20], dtype=torch.float32)
        self.COLOR_PEDUNCLE = torch.tensor([0.17, 0.213, 0.051], dtype=torch.float32)
        self.COLOR_FLOWER_OPEN = torch.tensor([0.921582, 0.916492, 0.344423], dtype=torch.float32)
        self.COLOR_FLOWER_CLOSED = torch.tensor([0.5, 0.4, 0.1], dtype=torch.float32)
        self.COLOR_FRUIT = torch.tensor([0.299629, 0.400454, 0.209546], dtype=torch.float32)

        # Extended organ type encoding (0/1/2 unchanged for eval compatibility)
        self.OT_STEM = 0
        self.OT_PETIOLE = 1
        self.OT_LEAF = 2
        self.OT_PEDUNCLE = 3
        self.OT_FLOWER = 4
        self.OT_FRUIT = 5

        self._infl_assets: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None

    def _get_inflorescence_assets(self):
        """Lazily load inflorescence OBJ assets (pod scale baked at load)."""
        if self._infl_assets is None:
            self._infl_assets = {
                'flower_open': self.asset_mgr.get_inflorescence_mesh('CowpeaFlower_open_yellow.obj', load_scale=0.0),
                'flower_closed': self.asset_mgr.get_inflorescence_mesh('CowpeaFlower_closed_yellow.obj', load_scale=0.0),
                'fruit': self.asset_mgr.get_inflorescence_mesh('CowpeaPod.obj', load_scale=0.75),
            }
        return self._infl_assets

    def build_mesh_from_organ_array(
        self,
        organ_array: PlantOrganArray,
        device: torch.device = torch.device('cpu'),
        max_leaves: Optional[int] = None,
        existence_threshold: float = 0.5,
        compute_mesh: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Build a mesh dict from a PlantOrganArray (typed (N,40) or legacy (M,94)).

        The typed layout is the native representation consumed by this builder.
        Legacy (N, 94) inputs are converted losslessly via
        ``PlantOrganArray.from_legacy_tensor`` (XML round-trip, verified max diff
        0.0) so that both representations render identically. This path is
        eval-only (non-differentiable); the differentiable backprop suites feed
        typed arrays with parent_logits directly, which flow through untouched.

        Args:
            compute_mesh: If False, only compute the (N, 14) part_transforms_14d
                tensor and return empty vertex buffers. Much faster for 14D extraction.
        """
        if not organ_array.is_typed:
            legacy_tensor = organ_array.tensor
            if legacy_tensor.device.type != 'cpu':
                legacy_tensor = legacy_tensor.cpu()
            organ_array = PlantOrganArray.from_legacy_tensor(legacy_tensor, organ_array.raw_metadata)
        return self._build_mesh_typed(
            organ_array, device=device, max_leaves=max_leaves,
            existence_threshold=existence_threshold, compute_mesh=compute_mesh,
        )

    def _build_mesh_typed(
        self,
        organ_array: PlantOrganArray,
        device: torch.device = torch.device('cpu'),
        max_leaves: Optional[int] = None,
        existence_threshold: float = 0.5,
        compute_mesh: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Typed-native (N, 40) plant mesh builder.

        Internode / petiole / leaf geometry is numerically identical to the
        legacy builder (verified pixel-identical on dap10/30/50), reading the
        typed per-organ rows instead of phytomer-slot columns. Floral bud
        peduncle / flower / fruit geometry is ported from the Helios C++
        reconstruction (InputOutput.cpp / PlantArchitecture.cpp).

        Args:
            compute_mesh: If False, skip all vertex/face generation and only
                compute the (N, 14) part_transforms_14d tensor. Used by
                to_part_tensor_14d for fast 14D extraction.
        """
        t = organ_array.tensor.to(device)
        existence = organ_array.existence.to(device).clamp(0.0, 1.0)
        N = t.shape[0]

        if N == 0:
            empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
            empty_o = torch.zeros((0,), dtype=torch.int64, device=device)
            empty_p14 = torch.zeros((0, NUM_FEATURES_14D), dtype=torch.float32, device=device)
            return {'vertices': empty3, 'faces': empty_f, 'normals': empty3, 'colors': empty3, 'organ_types': empty_o, 'part_transforms_14d': empty_p14}

        part_transforms_14d = torch.zeros((N, NUM_FEATURES_14D), dtype=torch.float32, device=device)
        part_transforms_14d[:, P14_COL_ORGAN_TYPE] = t[:, T_COL_ORGAN_TYPE]
        part_transforms_14d[:, P14_COL_EXISTENCE] = existence
        part_transforms_14d[:, P14_COL_SCALE_X] = 1.0
        part_transforms_14d[:, P14_COL_SCALE_Y] = 1.0
        part_transforms_14d[:, P14_COL_SCALE_Z] = 1.0
        eye_6d = rotation_matrix_to_6d(torch.eye(3, device=device))
        part_transforms_14d[:, P14_COL_ROT_0:P14_COL_ROT_5+1] = eye_6d

        # ------------------------------------------------------------------
        # Index maps over the typed per-organ rows
        # ------------------------------------------------------------------
        shoot_meta_row: Dict[int, int] = {}            # sid -> row idx (SHOOT_META)
        internode_rows: Dict[int, List[Tuple[int, int]]] = {}  # sid -> [(p_idx, n_idx)]
        petiole_row: Dict[Tuple[int, int, int], int] = {}      # (sid, p_idx, pet_i) -> row idx
        leaf_rows: Dict[Tuple[int, int, int], Dict[int, int]] = {}  # (sid,p_idx,pet_i) -> {lf_idx: row idx}
        bud_rows: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}  # (sid,p_idx) -> [(bud_idx, row idx)]
        peduncle_rows: Dict[Tuple[int, int], int] = {}   # (sid,p_idx) -> row idx
        flower_rows: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}  # (sid,p_idx) -> [(fl_idx, row idx)]

        t_cpu = t.detach().cpu().numpy()
        sids = t_cpu[:, T_COL_SHOOT_ID].astype(int)
        p_idxs = t_cpu[:, T_COL_PHYTOMER_IDX].astype(int)
        otypes = t_cpu[:, T_COL_ORGAN_TYPE].astype(int)
        parent_pets = t_cpu[:, T_COL_PARENT_PETIOLE_IDX].astype(int)
        child_idxs = t_cpu[:, T_COL_CHILD_INDEX].astype(int)

        for idx, ot in enumerate(otypes):
            sid = int(sids[idx])
            p_idx = int(p_idxs[idx])
            otype = ot
            if otype == ORGAN_ROOT_META:
                part_transforms_14d[idx, P14_COL_EXISTENCE] = 1.0
                part_transforms_14d[idx, P14_COL_BASE_X] = t[idx, T_COL_BASE_X]
                part_transforms_14d[idx, P14_COL_BASE_Y] = t[idx, T_COL_BASE_Y]
                part_transforms_14d[idx, P14_COL_BASE_Z] = t[idx, T_COL_BASE_Z]
            elif otype == ORGAN_SHOOT_META:
                shoot_meta_row[sid] = idx
            elif otype == ORGAN_INTERNODE:
                internode_rows.setdefault(sid, []).append((p_idx, idx))
                part_transforms_14d[idx, P14_COL_SCALE_X] = t[idx, T_COL_RADIUS]
                part_transforms_14d[idx, P14_COL_SCALE_Y] = t[idx, T_COL_RADIUS]
                part_transforms_14d[idx, P14_COL_SCALE_Z] = t[idx, T_COL_LENGTH]
            elif otype == ORGAN_PETIOLE:
                pet_i = int(parent_pets[idx])
                petiole_row[(sid, p_idx, pet_i)] = idx
                part_transforms_14d[idx, P14_COL_SCALE_X] = t[idx, T_COL_RADIUS]
                part_transforms_14d[idx, P14_COL_SCALE_Y] = t[idx, T_COL_RADIUS]
                part_transforms_14d[idx, P14_COL_SCALE_Z] = t[idx, T_COL_LENGTH]
            elif otype == ORGAN_LEAF:
                pet_i = int(parent_pets[idx])
                lf_idx = int(child_idxs[idx])
                leaf_rows.setdefault((sid, p_idx, pet_i), {})[lf_idx] = idx
                part_transforms_14d[idx, P14_COL_SCALE_X] = t[idx, T_COL_SCALE]
                part_transforms_14d[idx, P14_COL_SCALE_Y] = t[idx, T_COL_SCALE]
                part_transforms_14d[idx, P14_COL_SCALE_Z] = t[idx, T_COL_SCALE]
            elif otype == ORGAN_BUD:
                bud_idx = int(child_idxs[idx])
                bud_rows.setdefault((sid, p_idx), []).append((bud_idx, idx))
            elif otype == ORGAN_PEDUNCLE:
                peduncle_rows[(sid, p_idx)] = idx
                part_transforms_14d[idx, P14_COL_SCALE_X] = t[idx, T_COL_RADIUS]
                part_transforms_14d[idx, P14_COL_SCALE_Y] = t[idx, T_COL_RADIUS]
                part_transforms_14d[idx, P14_COL_SCALE_Z] = t[idx, T_COL_LENGTH]
            elif otype == ORGAN_FLOWER:
                fl_idx = int(child_idxs[idx])
                flower_rows.setdefault((sid, p_idx), []).append((fl_idx, idx))
                part_transforms_14d[idx, P14_COL_SCALE_X] = t[idx, T_COL_SCALE]
                part_transforms_14d[idx, P14_COL_SCALE_Y] = t[idx, T_COL_SCALE]
                part_transforms_14d[idx, P14_COL_SCALE_Z] = t[idx, T_COL_SCALE]

        for sid in internode_rows:
            internode_rows[sid].sort(key=lambda x: x[0])
        for key in bud_rows:
            bud_rows[key].sort(key=lambda x: x[0])
        for key in flower_rows:
            flower_rows[key].sort(key=lambda x: x[0])

        sorted_shoot_ids = sorted(internode_rows.keys())
        shoot_id_to_sorted_idx = {sid: i for i, sid in enumerate(sorted_shoot_ids)}
        node_output_info: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Tensor-based node outputs for soft parent aggregation (indexed by the
        # internode row linear index, matching soft parent candidate indexing).
        node_tip_positions = torch.zeros((N, 3), dtype=torch.float32, device=device)
        node_internode_axes = torch.zeros((N, 3), dtype=torch.float32, device=device)
        node_petiole_axes = torch.zeros((N, 2, 3), dtype=torch.float32, device=device)
        node_has_petiole = torch.zeros((N, 2), dtype=torch.float32, device=device)
        # getAxisVector(1.f, internode) tip axis used by peduncle orientation
        node_internode_tip_axes = torch.zeros((N, 3), dtype=torch.float32, device=device)

        # Soft parent representation
        use_soft_parent = (
            organ_array.parent_logits is not None
            and organ_array.parent_candidates is not None
        )
        if use_soft_parent:
            parent_logits = organ_array.parent_logits.to(device)
            parent_candidates = organ_array.parent_candidates.to(device)
            num_shoots_k = parent_logits.shape[0]
            if num_shoots_k != len(sorted_shoot_ids):
                raise ValueError(
                    f"parent_logits has {num_shoots_k} shoots but organ_array has {len(sorted_shoot_ids)} shoots"
                )

        def compute_shoot_base(sid: int, meta_row: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute shoot base position, internode axis, and petiole axis.

            Uses soft parent aggregation when parent_logits/parent_candidates are
            provided, otherwise falls back to the hard parent columns on the
            shoot's SHOOT_META row.
            """
            if use_soft_parent:
                s_idx = shoot_id_to_sorted_idx[sid]
                cand = parent_candidates[s_idx]  # (K, 3)
                cand_shoot = cand[:, 0]
                cand_node = cand[:, 1]
                cand_pet = cand[:, 2]

                valid_mask = (cand_node >= 0).float() * (cand_shoot >= 0).float()
                cand_exist = existence[cand_node.clamp(min=0)]
                active_mask = valid_mask * (cand_exist > 0.5).float()

                logits = parent_logits[s_idx]
                masked_logits = torch.where(active_mask > 0.5, logits, torch.full_like(logits, -1e9))
                weights = torch.softmax(masked_logits, dim=0)
                if weights.sum() < 1e-8 or weights.isnan().any():
                    weights = torch.zeros_like(weights)
                    weights[0] = 1.0

                cand_tip = node_tip_positions[cand_node].detach()          # (K, 3)
                cand_axis = node_internode_axes[cand_node].detach()      # (K, 3)

                cand_pet_clamped = cand_pet.clamp(0, 1)
                cand_pet_axis = node_petiole_axes[cand_node, cand_pet_clamped].detach()  # (K, 3)
                cand_has_pet = node_has_petiole[cand_node, cand_pet_clamped]    # (K,)
                has_pet0 = node_has_petiole[cand_node, 0]
                cand_pet_axis = torch.where(
                    (cand_has_pet > 0.5).unsqueeze(-1),
                    cand_pet_axis,
                    node_petiole_axes[cand_node, 0].detach(),
                )
                cand_has_pet = torch.where(cand_has_pet > 0.5, cand_has_pet, has_pet0)

                shoot_base_pos = (weights.unsqueeze(-1) * cand_tip).sum(dim=0)
                parent_internode_axis = (weights.unsqueeze(-1) * cand_axis).sum(dim=0)
                parent_petiole_axis = (weights.unsqueeze(-1) * cand_pet_axis).sum(dim=0)
                valid_pet = (weights * cand_has_pet).sum()
                if valid_pet < 1e-8:
                    parent_petiole_axis = torch.tensor([0.0, -1.0, 0.0], device=device)

                if cand_shoot[weights.argmax()].item() < 0:
                    shoot_base_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
                    parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
                    parent_petiole_axis = torch.tensor([0.0, -1.0, 0.0], device=device)
                return shoot_base_pos, parent_internode_axis, parent_petiole_axis

            meta_idx = shoot_meta_row.get(sid, node_indices[0][1] if sid in internode_rows and len(internode_rows[sid]) > 0 else 0)
            parent_sid = int(t_cpu[meta_idx, T_COL_PARENT_SHOOT_ID])
            parent_node_idx = int(t_cpu[meta_idx, T_COL_PARENT_NODE_IDX])
            parent_petiole_index = int(t_cpu[meta_idx, T_COL_PARENT_PETIOLE_IDX])

            if parent_sid < 0 or (parent_sid, parent_node_idx) not in node_output_info:
                shoot_base_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
                parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
                parent_petiole_axis = torch.tensor([0.0, -1.0, 0.0], device=device)
            else:
                p_info = node_output_info[(parent_sid, parent_node_idx)]
                parent_internode_axis = p_info['internode_axis']
                pet_axes = p_info.get('petiole_axes', {})
                if parent_petiole_index in pet_axes:
                    parent_petiole_axis = pet_axes[parent_petiole_index]
                else:
                    parent_petiole_axis = p_info.get('petiole_axis', torch.tensor([0.0, -1.0, 0.0], device=device))
                shoot_base_pos = p_info['tip']
            return shoot_base_pos, parent_internode_axis, parent_petiole_axis

        all_verts = []
        all_faces = []
        all_normals = []
        all_colors = []
        all_organs = []
        vert_offset = 0
        batched_leaf_params = []

        deg2rad = torch.tensor(math.pi / 180.0, dtype=torch.float32, device=device)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
        rendered_leaf_groups = 0
        gravitropic_curvature = 200.0

        # Per-phytomer context captured in Phase A and consumed by Phase B
        phytomer_context: Dict[Tuple[int, int], Dict[str, Any]] = {}
        shoot_last_internode_tips: Dict[int, torch.Tensor] = {}
        phytomer_petiole_count: Dict[Tuple[int, int], int] = {}

        # ==================================================================
        # Phase A: internodes, petioles, leaves (identical to legacy math)
        # ==================================================================
        for sid in sorted_shoot_ids:
            node_indices = internode_rows[sid]
            meta_idx = shoot_meta_row.get(sid, node_indices[0][1] if len(node_indices) > 0 else 0)
            meta_row = t[meta_idx]

            base_pitch_rad = meta_row[T_COL_PITCH] * deg2rad
            base_yaw_rad = meta_row[T_COL_YAW] * deg2rad
            base_roll_rad = meta_row[T_COL_ROLL] * deg2rad

            shoot_base_pos, parent_internode_axis, parent_petiole_axis = compute_shoot_base(sid, meta_row)
            part_transforms_14d[meta_idx, P14_COL_EXISTENCE] = 1.0
            part_transforms_14d[meta_idx, P14_COL_BASE_X:P14_COL_BASE_Z+1] = shoot_base_pos
            R_shoot = (
                rotr_z(base_yaw_rad, device) @
                rotr_y(-base_pitch_rad, device) @
                rotr_x(base_roll_rad, device)
            )
            part_transforms_14d[meta_idx, P14_COL_ROT_0:P14_COL_ROT_5+1] = rotation_matrix_to_6d(R_shoot)

            curr_pos = shoot_base_pos.clone()
            prev_internode_axis = parent_internode_axis
            prev_petiole_axis = parent_petiole_axis

            p_len_by_phytomer: Dict[int, List[torch.Tensor]] = {}

            for p_idx_in_shoot, (p_idx, n_idx) in enumerate(node_indices):
                row = t[n_idx]
                node_exist = existence[n_idx]

                # ---- Internode orientation vectors ----
                petiole_rot_axis = torch.linalg.cross(prev_internode_axis, prev_petiole_axis)
                if torch.linalg.norm(petiole_rot_axis) < 1e-6:
                    petiole_rot_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
                else:
                    petiole_rot_axis = petiole_rot_axis / torch.linalg.norm(petiole_rot_axis)

                inode_pitch_rad = row[T_COL_PITCH] * deg2rad
                inode_phyllo_rad = row[T_COL_PHYLLOTACTIC_ANGLE] * deg2rad

                i_axis = prev_internode_axis.clone()
                if p_idx_in_shoot == 0:
                    if inode_pitch_rad != 0.0:
                        i_axis = rotate_vector_about_axis(i_axis, petiole_rot_axis, 0.5 * inode_pitch_rad)
                    if base_roll_rad != 0.0:
                        petiole_rot_axis = rotate_vector_about_axis(petiole_rot_axis, prev_internode_axis, base_roll_rad)
                        i_axis = rotate_vector_about_axis(i_axis, prev_internode_axis, base_roll_rad)
                    if base_pitch_rad != 0.0:
                        base_pitch_axis = -1.0 * torch.linalg.cross(prev_internode_axis, prev_petiole_axis)
                        if torch.linalg.norm(base_pitch_axis) > 1e-6:
                            base_pitch_axis = base_pitch_axis / torch.linalg.norm(base_pitch_axis)
                            petiole_rot_axis = rotate_vector_about_axis(petiole_rot_axis, base_pitch_axis, -base_pitch_rad)
                            i_axis = rotate_vector_about_axis(i_axis, base_pitch_axis, -base_pitch_rad)
                    if base_yaw_rad != 0.0:
                        petiole_rot_axis = rotate_vector_about_axis(petiole_rot_axis, prev_internode_axis, base_yaw_rad)
                        i_axis = rotate_vector_about_axis(i_axis, prev_internode_axis, base_yaw_rad)
                else:
                    if inode_pitch_rad != 0.0:
                        i_axis = rotate_vector_about_axis(i_axis, petiole_rot_axis, -1.25 * inode_pitch_rad)

                i_axis = i_axis / (torch.linalg.norm(i_axis) + 1e-6)

                shoot_bending_axis = torch.linalg.cross(i_axis, z_axis)
                shoot_bending_norm = torch.linalg.norm(shoot_bending_axis)
                if shoot_bending_norm < 1e-6:
                    shoot_bending_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
                else:
                    shoot_bending_axis = shoot_bending_axis / shoot_bending_norm

                # ---- Internode tube ----
                inode_len = torch.clamp(row[T_COL_LENGTH], min=1e-4) * node_exist
                inode_rad = torch.clamp(row[T_COL_RADIUS], min=1e-4) * node_exist
                seg_cnt = max(1, int(t_cpu[n_idx, T_COL_LENGTH_SEGMENTS]))
                seg_len = inode_len / seg_cnt
                seg_len_max = torch.clamp(row[T_COL_LENGTH_MAX], min=1e-4) / seg_cnt

                curv_p0, curv_p1 = row[T_COL_CURV_PERT_0], row[T_COL_CURV_PERT_1]
                yaw_p0, yaw_p1 = row[T_COL_YAW_PERT_0], row[T_COL_YAW_PERT_1]

                inode_verts_list = [curr_pos.clone()]
                step_p = curr_pos.clone()
                step_dir = i_axis.clone()
                for s in range(seg_cnt):
                    if p_idx_in_shoot > 0:
                        curv_pert = curv_p0 if s == 0 else curv_p1
                        yaw_pert = yaw_p0 if s == 0 else yaw_p1
                        curv_fact = 0.5 - step_dir[2] / 2.0
                        if step_dir[2] < 0:
                            curv_fact = curv_fact * 2.0
                        curvature_angle = deg2rad * (gravitropic_curvature * curv_fact * seg_len_max + curv_pert)
                        if curvature_angle != 0.0:
                            step_dir = rotate_vector_about_axis(step_dir, shoot_bending_axis, curvature_angle)
                        if yaw_pert != 0.0:
                            step_dir = rotate_vector_about_axis(step_dir, z_axis, deg2rad * yaw_pert)
                    step_p = step_p + step_dir * seg_len
                    inode_verts_list.append(step_p)

                inode_line = torch.stack(inode_verts_list)
                inode_radii = inode_rad.expand(inode_line.shape[0])

                if compute_mesh:
                    v_tub, f_tub, n_tub, c_tub = generate_cone_tube_mesh_torch(
                        inode_line, inode_radii, self.COLOR_STEM.to(device), radial_subdivisions=self.tube_radial_subdivisions
                    )

                    if v_tub.shape[0] > 0:
                        all_verts.append(v_tub)
                        all_faces.append(f_tub + vert_offset)
                        all_normals.append(n_tub)
                        all_colors.append(c_tub)
                        all_organs.append(torch.zeros(v_tub.shape[0], dtype=torch.int64, device=device))  # Organ 0 = Stem
                        vert_offset += v_tub.shape[0]

                curr_pos = inode_line[-1]
                inode_tip_axis = step_dir / (torch.linalg.norm(step_dir) + 1e-6)
                node_internode_tip_axes[n_idx] = get_axis_vector_torch(inode_line, 1.0)

                part_transforms_14d[n_idx, P14_COL_EXISTENCE] = 1.0
                part_transforms_14d[n_idx, P14_COL_BASE_X:P14_COL_BASE_Z+1] = inode_line[0]
                part_transforms_14d[n_idx, P14_COL_SCALE_X] = inode_rad
                part_transforms_14d[n_idx, P14_COL_SCALE_Y] = inode_rad
                part_transforms_14d[n_idx, P14_COL_SCALE_Z] = inode_len
                R_inode = _get_rotation_matrix_between_vectors_batch(
                    torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0),
                    inode_tip_axis.unsqueeze(0)
                ).squeeze(0)
                part_transforms_14d[n_idx, P14_COL_ROT_0:P14_COL_ROT_5+1] = rotation_matrix_to_6d(R_inode)

                if os.environ.get("HELIOS_DUMP_GEOM"):
                    tp = curr_pos.detach().cpu().numpy()
                    print(f"PTDEBUG I {sid} {p_idx} 1 {tp[0]:.6f} {tp[1]:.6f} {tp[2]:.6f}", file=sys.stderr)

                # ---- Petiole & Leaf Geometry ----
                pet_axes_stored = {}
                pet_line_stored: Dict[int, torch.Tensor] = {}
                node_info = {
                    'tip': curr_pos,
                    'internode_axis': inode_tip_axis,
                    'radius': inode_rad,
                }
                node_tip_positions[n_idx] = curr_pos
                node_internode_axes[n_idx] = inode_tip_axis

                petioles_here = [k for k in petiole_row if k[0] == sid and k[1] == p_idx]
                phytomer_petiole_count[(sid, p_idx)] = len(petioles_here)

                def process_petiole(pet_i, petiole_index):
                    nonlocal rendered_leaf_groups, vert_offset
                    pet_row = petiole_row.get((sid, p_idx, pet_i))
                    if pet_row is None:
                        return
                    pet_row_t = t[pet_row]
                    p_len_raw = pet_row_t[T_COL_LENGTH]
                    p_rad_raw = pet_row_t[T_COL_RADIUS]
                    p_pitch_deg = pet_row_t[T_COL_PITCH]
                    p_curv_deg = pet_row_t[T_COL_CURVATURE]
                    p_cls = pet_row_t[T_COL_CURRENT_LEAF_SCALE_FACTOR]
                    p_taper = pet_row_t[T_COL_TAPER]
                    p_seg_cnt = max(1, int(t_cpu[pet_row, T_COL_LENGTH_SEGMENTS]))
                    lflt_scale = pet_row_t[T_COL_LEAFLET_SCALE]
                    lflt_offset = pet_row_t[T_COL_LEAFLET_OFFSET]

                    leaf_dict = leaf_rows.get((sid, p_idx, pet_i), {})
                    leaf_list = sorted(leaf_dict.items(), key=lambda kv: kv[0])  # [(lf_idx, row idx)]
                    num_leaves = len(leaf_list)

                    pet_pitch_rad = p_pitch_deg * deg2rad
                    pet_axis = rotate_vector_about_axis(i_axis, petiole_rot_axis, torch.abs(pet_pitch_rad))
                    pet_rot_ax = petiole_rot_axis.clone()
                    if p_idx_in_shoot != 0 and inode_phyllo_rad != 0.0:
                        pet_axis = rotate_vector_about_axis(pet_axis, i_axis, inode_phyllo_rad)
                        pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, inode_phyllo_rad)
                    if petiole_index > 0:
                        petioles_per_internode = 2.0 if len(petioles_here) > 1 else 1.0
                        budrot = torch.tensor(petiole_index * 2.0 * math.pi / petioles_per_internode, device=device)
                        pet_axis = rotate_vector_about_axis(pet_axis, i_axis, budrot)
                        pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, budrot)
                    pet_axis = pet_axis / (torch.linalg.norm(pet_axis) + 1e-12)
                    pet_axes_stored[petiole_index] = pet_axis.clone()

                    p_len = p_len_raw * node_exist
                    p_rad = p_rad_raw * node_exist
                    if p_len <= 0 or p_rad <= 0:
                        return

                    pet_rot_ax_norm = pet_rot_ax / (torch.linalg.norm(pet_rot_ax) + 1e-8)
                    pet_base = inode_line[-1]
                    seq_len = p_len / p_seg_cnt

                    curv_per_seg = p_curv_deg * seq_len * deg2rad
                    if torch.abs(curv_per_seg) > 1e-12:
                        s_indices = torch.arange(1, p_seg_cnt + 1, device=device, dtype=torch.float32)
                        angles = -s_indices * curv_per_seg
                        dirs = rotate_points_about_axis(pet_axis.unsqueeze(0).expand(p_seg_cnt, 3), pet_rot_ax_norm, angles)
                        offsets = torch.cumsum(dirs * seq_len, dim=0)
                        pet_line = torch.cat([pet_base.unsqueeze(0), pet_base.unsqueeze(0) + offsets], dim=0)
                    else:
                        s_indices = torch.arange(1, p_seg_cnt + 1, device=device, dtype=torch.float32).unsqueeze(-1)
                        offsets = s_indices * (pet_axis * seq_len)
                        pet_line = torch.cat([pet_base.unsqueeze(0), pet_base.unsqueeze(0) + offsets], dim=0)
                    jj = torch.linspace(0.0, p_seg_cnt, pet_line.shape[0], device=device)
                    pet_radii = p_cls * p_rad * (1.0 - p_taper / float(p_seg_cnt) * jj)
                    pet_radii = torch.clamp(pet_radii, min=1e-6)

                    if compute_mesh:
                        v_pet, f_pet, n_pet, c_pet = generate_cone_tube_mesh_torch(
                            pet_line, pet_radii, self.COLOR_PETIOLE.to(device), radial_subdivisions=self.tube_radial_subdivisions
                        )

                        if v_pet.shape[0] > 0:
                            all_verts.append(v_pet)
                            all_faces.append(f_pet + vert_offset)
                            all_normals.append(n_pet)
                            all_colors.append(c_pet)
                            all_organs.append(torch.ones(v_pet.shape[0], dtype=torch.int64, device=device))  # Organ 1 = Petiole
                            vert_offset += v_pet.shape[0]

                    pet_tip = pet_line[-1]
                    pet_tip_axis = pet_line[-1] - pet_line[-2]
                    pet_tip_axis = pet_tip_axis / (torch.linalg.norm(pet_tip_axis) + 1e-8)
                    pet_line_stored[petiole_index] = pet_line.clone()

                    part_transforms_14d[pet_row, P14_COL_EXISTENCE] = 1.0
                    part_transforms_14d[pet_row, P14_COL_BASE_X:P14_COL_BASE_Z+1] = pet_base
                    part_transforms_14d[pet_row, P14_COL_SCALE_X] = p_rad
                    part_transforms_14d[pet_row, P14_COL_SCALE_Y] = p_rad
                    part_transforms_14d[pet_row, P14_COL_SCALE_Z] = p_len
                    R_pet = _get_rotation_matrix_between_vectors_batch(
                        torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0),
                        pet_tip_axis.unsqueeze(0)
                    ).squeeze(0)
                    part_transforms_14d[pet_row, P14_COL_ROT_0:P14_COL_ROT_5+1] = rotation_matrix_to_6d(R_pet)

                    if os.environ.get("HELIOS_DUMP_GEOM"):
                        iax = inode_tip_axis.detach().cpu().numpy()
                        pax2 = pet_tip_axis.detach().cpu().numpy()
                        print(f"PTDEBUG AX {sid} {p_idx} {petiole_index} {iax[0]:.6f} {iax[1]:.6f} {iax[2]:.6f} {pax2[0]:.6f} {pax2[1]:.6f} {pax2[2]:.6f}", file=sys.stderr)

                    if os.environ.get("HELIOS_DUMP_GEOM"):
                        pt = pet_line[-1].detach().cpu().numpy()
                        print(f"PTDEBUG P {sid} {p_idx} {petiole_index} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}", file=sys.stderr)

                    if num_leaves > 0 and (max_leaves is None or rendered_leaf_groups < max_leaves):
                        rendered_leaf_groups += 1

                        for lf_i in range(min(num_leaves, 3)):
                            lf_idx, leaf_row_idx = leaf_list[lf_i]
                            lr = t[leaf_row_idx]
                            l_scale = lr[T_COL_SCALE]
                            l_pitch_raw = lr[T_COL_PITCH] * deg2rad
                            l_yaw = lr[T_COL_YAW] * deg2rad
                            l_roll_raw = lr[T_COL_ROLL] * deg2rad

                            ind_from_tip = float(lf_i) - float(num_leaves - 1) / 2.0
                            compound_rotation = 0.0
                            if num_leaves > 1:
                                if lf_i == (num_leaves - 1) / 2.0:
                                    compound_rotation = 0.0
                                elif lf_i < (num_leaves - 1) / 2.0:
                                    compound_rotation = -0.5 * math.pi
                                else:
                                    compound_rotation = 0.5 * math.pi

                            tot_scale = l_scale * self.leaf_scale_factor * node_exist

                            asin_pz = torch.asin(torch.clamp(pet_tip_axis[2], -1.0, 1.0))

                            if num_leaves == 1:
                                roll_rot = torch.acos(torch.clamp(inode_tip_axis[2], -1.0, 1.0)) - l_roll_raw
                            elif ind_from_tip != 0:
                                sign_roll = compound_rotation / abs(compound_rotation)
                                roll_rot = (asin_pz + l_roll_raw) * sign_roll
                            else:
                                roll_rot = 0.0

                            pitch_rot = l_pitch_raw
                            if ind_from_tip == 0:
                                pitch_rot = pitch_rot + asin_pz

                            yaw_rot = 0.0
                            if ind_from_tip != 0:
                                yaw_rot = l_yaw

                            azimuth_rot = -torch.atan2(pet_tip_axis[1], pet_tip_axis[0] + 1e-8) + compound_rotation

                            leaf_base = pet_tip
                            if num_leaves > 1 and lflt_offset > 0.0 and ind_from_tip != 0:
                                offset = (abs(ind_from_tip) - 0.5) * lflt_offset * p_len
                                frac = 1.0 - offset / torch.clamp(p_len, min=1e-6)
                                frac = torch.clamp(frac, 0.0, 1.0)
                                if not (torch.isnan(frac) or torch.isinf(frac)):
                                    leaf_base = interpolate_tube_torch(pet_line, float(frac))

                            R_leaf = (
                                rotr_z(azimuth_rot + yaw_rot, device) @
                                rotr_y(-pitch_rot, device) @
                                rotr_x(roll_rot, device)
                            )
                            part_transforms_14d[leaf_row_idx, P14_COL_EXISTENCE] = 1.0
                            part_transforms_14d[leaf_row_idx, P14_COL_BASE_X:P14_COL_BASE_Z+1] = leaf_base
                            part_transforms_14d[leaf_row_idx, P14_COL_SCALE_X] = l_scale
                            part_transforms_14d[leaf_row_idx, P14_COL_SCALE_Y] = l_scale
                            part_transforms_14d[leaf_row_idx, P14_COL_SCALE_Z] = l_scale
                            part_transforms_14d[leaf_row_idx, P14_COL_ROT_0:P14_COL_ROT_5+1] = rotation_matrix_to_6d(R_leaf)

                            if os.environ.get("HELIOS_DUMP_GEOM"):
                                lb = leaf_base.detach().cpu().numpy()
                                print(f"PTDEBUG L {sid} {p_idx} {petiole_index} {lf_i} {lb[0]:.6f} {lb[1]:.6f} {lb[2]:.6f}", file=sys.stderr)

                            if compute_mesh:
                                if self.use_generic_leaves:
                                    batched_leaf_params.append((
                                        float(tot_scale),
                                        leaf_base,
                                        float(azimuth_rot + yaw_rot),
                                        float(-pitch_rot),
                                        float(roll_rot)
                                    ))
                                else:
                                    if num_leaves == 1:
                                        obj_name = "CowpeaLeaf_unifoliate.obj"
                                    else:
                                        if lf_i == 0:
                                            obj_name = "CowpeaLeaf_left_highres.obj"
                                        elif lf_i == 1:
                                            obj_name = "CowpeaLeaf_tip_highres.obj"
                                        else:
                                            obj_name = "CowpeaLeaf_right_highres.obj"

                                    try:
                                        v_raw, f_lf_b, n_tmpl = self.asset_mgr.get_mesh_with_normals(obj_name, device)
                                    except FileNotFoundError:
                                        continue
                                    v_lf_b = v_raw * tot_scale
                                    n_lf_b = n_tmpl

                                    v_lf_rot = (R_leaf @ v_lf_b.T).T
                                    n_lf = (R_leaf @ n_lf_b.T).T
                                    v_lf = v_lf_rot + leaf_base

                                    c_lf = self.COLOR_LEAF.to(device).unsqueeze(0).expand(v_lf.shape[0], 3)
                                    all_verts.append(v_lf)
                                    all_faces.append(f_lf_b + vert_offset)
                                    all_normals.append(n_lf)
                                    all_colors.append(c_lf)
                                    all_organs.append(torch.full((v_lf.shape[0],), 2, dtype=torch.int64, device=device))  # Organ 2 = Leaf
                                    vert_offset += v_lf.shape[0]

                for pet_i in sorted(k[2] for k in petioles_here):
                    process_petiole(pet_i, pet_i)

                node_info['petiole_axes'] = pet_axes_stored
                if 0 in pet_axes_stored:
                    node_info['petiole_axis'] = pet_axes_stored[0].clone()
                if 0 in pet_axes_stored:
                    node_petiole_axes[n_idx, 0] = pet_axes_stored[0]
                    node_has_petiole[n_idx, 0] = 1.0
                if 1 in pet_axes_stored:
                    node_petiole_axes[n_idx, 1] = pet_axes_stored[1]
                    node_has_petiole[n_idx, 1] = 1.0

                node_output_info[(sid, p_idx)] = node_info

                phytomer_context[(sid, p_idx)] = {
                    'inode_line': inode_line,
                    'inode_tip_axis': inode_tip_axis,
                    'tip_getaxis': node_internode_tip_axes[n_idx].clone(),
                    'pet_lines': {k: v.clone() for k, v in pet_line_stored.items()},
                    'p_idx_in_shoot': p_idx_in_shoot,
                    'n_idx': n_idx,
                }

                # Update parent context for the next phytomer on this shoot
                prev_internode_axis = inode_tip_axis
                if 0 in pet_axes_stored:
                    prev_petiole_axis = pet_axes_stored[0]
                else:
                    ghost = torch.linalg.cross(inode_tip_axis, z_axis)
                    if torch.linalg.norm(ghost) < 0.01:
                        ghost = torch.tensor([0.0, 1.0, 0.0], device=device)
                    prev_petiole_axis = ghost / torch.linalg.norm(ghost)

            shoot_last_internode_tips[sid] = curr_pos.clone()

        # Execute batched GPU leaf transformation for generic leaves (only when mesh needed)
        if compute_mesh and len(batched_leaf_params) > 0 and self.use_generic_leaves:
            K = len(batched_leaf_params)
            scales_t = torch.tensor([item[0] for item in batched_leaf_params], dtype=torch.float32, device=device).view(K, 1, 1)
            bases_t = torch.stack([item[1] for item in batched_leaf_params]).view(K, 1, 3)
            az_t = torch.tensor([item[2] for item in batched_leaf_params], dtype=torch.float32, device=device)
            pitch_t = torch.tensor([item[3] for item in batched_leaf_params], dtype=torch.float32, device=device)
            roll_t = torch.tensor([item[4] for item in batched_leaf_params], dtype=torch.float32, device=device)

            c_az, s_az = torch.cos(az_t), torch.sin(az_t)
            c_p, s_p = torch.cos(pitch_t), torch.sin(pitch_t)
            c_r, s_r = torch.cos(roll_t), torch.sin(roll_t)

            R_z = torch.zeros(K, 3, 3, device=device)
            R_z[:, 0, 0] = c_az; R_z[:, 0, 1] = -s_az; R_z[:, 1, 0] = s_az; R_z[:, 1, 1] = c_az; R_z[:, 2, 2] = 1.0

            R_y = torch.zeros(K, 3, 3, device=device)
            R_y[:, 0, 0] = c_p; R_y[:, 0, 2] = s_p; R_y[:, 1, 1] = 1.0; R_y[:, 2, 0] = -s_p; R_y[:, 2, 2] = c_p

            R_x = torch.zeros(K, 3, 3, device=device)
            R_x[:, 0, 0] = 1.0; R_x[:, 1, 1] = c_r; R_x[:, 1, 2] = -s_r; R_x[:, 2, 1] = s_r; R_x[:, 2, 2] = c_r

            R_all = torch.bmm(torch.bmm(R_z, R_y), R_x)

            v_tmpl, f_tmpl, n_tmpl = get_generic_leaf_template(aspect_ratio=0.65, Nx=8, Ny=8, device=device)
            V_l = v_tmpl.shape[0]

            v_scaled = v_tmpl.unsqueeze(0) * scales_t
            v_batched = torch.bmm(v_scaled, R_all.transpose(1, 2)) + bases_t
            n_batched = torch.bmm(n_tmpl.unsqueeze(0).expand(K, V_l, 3), R_all.transpose(1, 2))

            offsets_k = torch.arange(K, device=device, dtype=torch.int64) * V_l + vert_offset
            f_batched = (f_tmpl.unsqueeze(0) + offsets_k.view(K, 1, 1)).reshape(-1, 3)

            v_leaves_all = v_batched.reshape(-1, 3)
            n_leaves_all = n_batched.reshape(-1, 3)
            c_leaves_all = self.COLOR_LEAF.to(device).unsqueeze(0).expand(v_leaves_all.shape[0], 3)
            o_leaves_all = torch.full((v_leaves_all.shape[0],), 2, dtype=torch.int64, device=device)

            all_verts.append(v_leaves_all)
            all_faces.append(f_batched)
            all_normals.append(n_leaves_all)
            all_colors.append(c_leaves_all)
            all_organs.append(o_leaves_all)
            vert_offset += v_leaves_all.shape[0]

        # ==================================================================
        # Phase B: floral bud peduncle / flower / fruit geometry
        # ==================================================================
        has_floral_geometry = False
        for _, bud_list in bud_rows.items():
            for _, bidx in bud_list:
                if int(t[bidx, T_COL_BUD_STATE].item()) in (2, 3, 4):
                    has_floral_geometry = True
                    break
            if has_floral_geometry:
                break

        if has_floral_geometry:
            infl_assets = self._get_inflorescence_assets()

            for (sid, p_idx), bud_list in sorted(bud_rows.items()):
                ctx = phytomer_context.get((sid, p_idx))
                if ctx is None:
                    continue
                Nbuds = len(bud_list)
                petiole_count = max(1, phytomer_petiole_count.get((sid, p_idx), 1))
                petioles_per_internode = float(petiole_count)

                for bud_index, bud_row_idx in bud_list:
                    bud_row = t[bud_row_idx]
                    state = int(bud_row[T_COL_BUD_STATE].item())
                    if state not in (2, 3, 4):
                        continue
                    pet_i = int(bud_row[T_COL_PARENT_PETIOLE_IDX].item())
                    is_terminal = int(bud_row[T_COL_BUD_IS_TERMINAL].item()) > 0
                    current_fruit_scale_factor = bud_row[T_COL_FRUIT_SCALE]
                    flower_offset = float(bud_row[T_COL_FLOWER_OFFSET].item())

                    # ---- Bud base position & rotation ----
                    if is_terminal:
                        bud_base = shoot_last_internode_tips.get(sid, ctx['inode_line'][-1])
                        base_pitch = (math.pi / 6.0) if Nbuds > 1 else 0.0  # deg2rad(30)
                        base_yaw = bud_index * 2.0 * math.pi / float(Nbuds)
                    else:
                        pet_line0 = ctx['pet_lines'].get(pet_i)
                        bud_base = pet_line0[0] if pet_line0 is not None else ctx['inode_line'][-1]
                        base_pitch = bud_index * 0.1 * math.pi / float(Nbuds)
                        base_yaw = -0.25 * math.pi + bud_index * 0.5 * math.pi / float(Nbuds)
                    base_roll = 0.0
                    part_transforms_14d[bud_row_idx, P14_COL_BASE_X:P14_COL_BASE_Z+1] = bud_base

                    # ---- Peduncle params ----
                    ped_row_idx = peduncle_rows.get((sid, p_idx))
                    if ped_row_idx is None:
                        continue
                    ped_row = t[ped_row_idx]
                    p_len = float(ped_row[T_COL_LENGTH].item())
                    p_rad = float(ped_row[T_COL_RADIUS].item())
                    p_pitch_rad = ped_row[T_COL_PITCH] * deg2rad
                    p_curv_deg = float(ped_row[T_COL_CURVATURE].item())
                    if p_len <= 0 or p_rad <= 0:
                        continue

                    # ---- recomputePeduncleOrientationVectors ----
                    inode_line = ctx['inode_line']
                    peduncle_axis = ctx['tip_getaxis'].clone()  # getAxisVector(1.f, internode)

                    if ctx['p_idx_in_shoot'] > 0:
                        prev_n_idx = None
                        for (pp_idx, nn_idx) in internode_rows[sid]:
                            if pp_idx == p_idx:
                                break
                            prev_n_idx = nn_idx
                        if prev_n_idx is not None:
                            parent_internode_axis = node_internode_tip_axes[prev_n_idx]
                        else:
                            parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
                    else:
                        meta_idx = shoot_meta_row.get(sid, 0)
                        pmeta = t[meta_idx]
                        parent_sid = int(pmeta[T_COL_PARENT_SHOOT_ID].item())
                        if parent_sid >= 0:
                            parent_node_xml = int(pmeta[T_COL_PARENT_NODE_IDX].item())
                            parent_lin = PlantOrganArray._xml_parent_node_to_linear_idx(t, parent_sid, parent_node_xml)
                            parent_internode_axis = node_internode_tip_axes[parent_lin]
                        else:
                            parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], device=device)

                    pet_line = ctx['pet_lines'].get(pet_i)
                    if pet_line is not None:
                        current_petiole_axis = get_axis_vector_torch(pet_line, 0.0)
                        parent_petiole_base_axis = get_axis_vector_torch(pet_line, 0.0)
                    else:
                        current_petiole_axis = parent_internode_axis
                        parent_petiole_base_axis = ctx['tip_getaxis']

                    infl_bending = torch.linalg.cross(parent_internode_axis, current_petiole_axis)
                    if torch.linalg.norm(infl_bending) < 0.001:
                        infl_bending = torch.tensor([1.0, 0.0, 0.0], device=device)
                    else:
                        infl_bending = infl_bending / torch.linalg.norm(infl_bending)

                    if p_pitch_rad != 0 or base_pitch != 0:
                        peduncle_axis = rotate_vector_about_axis(peduncle_axis, infl_bending, p_pitch_rad + base_pitch)

                    internode_axis = ctx['tip_getaxis']
                    parent_petiole_azimuth = -torch.atan2(parent_petiole_base_axis[1], parent_petiole_base_axis[0])
                    current_peduncle_azimuth = -torch.atan2(peduncle_axis[1], peduncle_axis[0])
                    azimuthal_rotation = current_peduncle_azimuth - parent_petiole_azimuth
                    peduncle_axis = rotate_vector_about_axis(peduncle_axis, internode_axis, azimuthal_rotation)
                    infl_bending = rotate_vector_about_axis(infl_bending, internode_axis, azimuthal_rotation)
                    peduncle_axis = peduncle_axis / (torch.linalg.norm(peduncle_axis) + 1e-6)

                    # ---- Peduncle tube ----
                    segs = int(ped_row[T_COL_LENGTH_SEGMENTS].item())
                    segs = max(1, segs) if segs > 0 else 6
                    n_rad = int(ped_row[T_COL_RADIAL_SUBDIVISIONS].item())
                    n_rad = max(3, n_rad) if n_rad > 0 else 6
                    dr = p_len / segs
                    axis = peduncle_axis
                    verts_list = [bud_base.clone()]
                    radii_list = [p_rad]
                    for i in range(segs):
                        if abs(p_curv_deg) > 0:
                            hba = torch.linalg.cross(axis, z_axis)
                            m = torch.linalg.norm(hba)
                            if m > 0.001:
                                hba = hba / m
                                theta_curv = deg2rad * (p_curv_deg * dr)
                                zc = torch.clamp(axis[2], -1.0, 1.0)
                                theta_from_target = torch.acos(zc) if p_curv_deg > 0 else torch.acos(-zc)
                                if abs(theta_curv) >= theta_from_target:
                                    axis = z_axis if p_curv_deg > 0 else -z_axis
                                else:
                                    axis = rotate_vector_about_axis(axis, hba, theta_curv)
                                    axis = axis / (torch.linalg.norm(axis) + 1e-6)
                            else:
                                axis = z_axis if p_curv_deg > 0 else -z_axis
                        verts_list.append(verts_list[-1] + dr * axis)
                        radii_list.append(p_rad)

                    ped_line = torch.stack(verts_list)
                    ped_radii = torch.tensor(radii_list, dtype=torch.float32, device=device)

                    if compute_mesh:
                        v_ped, f_ped, n_ped, c_ped = generate_cone_tube_mesh_torch(
                            ped_line, ped_radii, self.COLOR_PEDUNCLE.to(device), radial_subdivisions=n_rad
                        )
                        if v_ped.shape[0] > 0:
                            all_verts.append(v_ped)
                            all_faces.append(f_ped + vert_offset)
                            all_normals.append(n_ped)
                            all_colors.append(c_ped)
                            all_organs.append(torch.full((v_ped.shape[0],), self.OT_PEDUNCLE, dtype=torch.int64, device=device))
                            vert_offset += v_ped.shape[0]

                    part_transforms_14d[ped_row_idx, P14_COL_EXISTENCE] = 1.0
                    part_transforms_14d[ped_row_idx, P14_COL_BASE_X:P14_COL_BASE_Z+1] = ped_line[0]
                    part_transforms_14d[ped_row_idx, P14_COL_SCALE_X] = p_rad
                    part_transforms_14d[ped_row_idx, P14_COL_SCALE_Y] = p_rad
                    part_transforms_14d[ped_row_idx, P14_COL_SCALE_Z] = p_len
                    ped_axis_dir = (ped_line[-1] - ped_line[0])
                    ped_axis_dir = ped_axis_dir / (torch.linalg.norm(ped_axis_dir) + 1e-8)
                    R_ped = _get_rotation_matrix_between_vectors_batch(
                        torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0),
                        ped_axis_dir.unsqueeze(0)
                    ).squeeze(0)
                    part_transforms_14d[ped_row_idx, P14_COL_ROT_0:P14_COL_ROT_5+1] = rotation_matrix_to_6d(R_ped)

                    if os.environ.get("HELIOS_DUMP_GEOM"):
                        btip = ped_line[-1].detach().cpu().numpy()
                        bbase = bud_base.detach().cpu().numpy()
                        print(f"PTDEBUG PD {sid} {p_idx} {btip[0]:.6f} {btip[1]:.6f} {btip[2]:.6f} {bbase[0]:.6f} {bbase[1]:.6f} {bbase[2]:.6f} len={p_len:.4f}", file=sys.stderr)

                    # ---- Flower / fruit placement ----
                    fl_list = flower_rows.get((sid, p_idx), [])
                    n_flowers = len(fl_list)
                    if n_flowers == 0:
                        continue

                    for fl_idx, fl_row_idx in fl_list:
                        fl_row = t[fl_row_idx]
                        saved_pitch = fl_row[T_COL_PITCH] * deg2rad
                        saved_yaw = fl_row[T_COL_YAW] * deg2rad
                        saved_roll = fl_row[T_COL_ROLL] * deg2rad
                        saved_azimuth = fl_row[T_COL_FLOWER_AZIMUTH] * deg2rad
                        base_scale = float(fl_row[T_COL_SCALE].item())

                        flower_offset_clamped = clamp_offset_torch(n_flowers, flower_offset)
                        ind_from_tip_computed = abs(float(fl_idx) - float(n_flowers - 1) / float(petioles_per_internode))
                        flower_base = ped_line[-1]
                        if n_flowers > 1 and flower_offset_clamped > 0 and ind_from_tip_computed != 0:
                            offset_computed = (ind_from_tip_computed - 0.5) * flower_offset_clamped * p_len
                            frac_computed = 1.0
                            if p_len > 0:
                                frac_computed = 1.0 - offset_computed / p_len
                            flower_base = interpolate_tube_torch(ped_line, frac_computed)

                        flower_offset_val = flower_offset
                        if n_flowers > 2:
                            denom = 0.5 * float(n_flowers) - 1.0
                            if flower_offset_val * denom > 1.0:
                                flower_offset_val = 1.0 / denom
                        ind_from_tip = abs(float(fl_idx) - float(n_flowers - 1) / float(petioles_per_internode))
                        frac = 1.0
                        if n_flowers > 1 and flower_offset_val > 0 and ind_from_tip != 0:
                            offset = (ind_from_tip - 0.5) * flower_offset_val * p_len
                            if p_len > 0:
                                frac = 1.0 - offset / p_len
                        recalculated_peduncle_axis = get_axis_vector_torch(ped_line, frac)

                        if state == 4:
                            base_fruit_scale = base_scale if base_scale >= 0 else 0.09
                            scale_factor = base_fruit_scale * current_fruit_scale_factor
                            is_open = False
                            asset_name = 'fruit'
                            obj_color = self.COLOR_FRUIT
                            obj_ot = self.OT_FRUIT
                        else:
                            scale_factor = base_scale if base_scale >= 0 else 0.03
                            is_open = (state == 3)
                            asset_name = 'flower_open' if is_open else 'flower_closed'
                            obj_color = self.COLOR_FLOWER_OPEN if is_open else self.COLOR_FLOWER_CLOSED
                            obj_ot = self.OT_FLOWER

                        v_asset, f_asset = infl_assets[asset_name]
                        v_asset = v_asset.to(device)
                        f_asset = f_asset.to(device)

                        v_obj = v_asset * scale_factor
                        v_obj = (rotr_x(saved_roll, device) @ v_obj.T).T
                        v_obj = (rotr_y(saved_pitch, device) @ v_obj.T).T
                        v_obj = (rotr_z(saved_azimuth, device) @ v_obj.T).T
                        v_obj = v_obj + flower_base
                        v_rel = v_obj - flower_base
                        v_rel = rotate_points_about_axis(v_rel, recalculated_peduncle_axis, saved_yaw)
                        v_obj = v_rel + flower_base

                        part_transforms_14d[fl_row_idx, P14_COL_EXISTENCE] = 1.0
                        part_transforms_14d[fl_row_idx, P14_COL_BASE_X:P14_COL_BASE_Z+1] = flower_base
                        part_transforms_14d[fl_row_idx, P14_COL_SCALE_X] = fl_row[T_COL_SCALE]
                        part_transforms_14d[fl_row_idx, P14_COL_SCALE_Y] = fl_row[T_COL_SCALE]
                        part_transforms_14d[fl_row_idx, P14_COL_SCALE_Z] = fl_row[T_COL_SCALE]
                        if state == 4:
                            part_transforms_14d[fl_row_idx, P14_COL_ORGAN_TYPE] = 8
                        elif state == 2:
                            part_transforms_14d[fl_row_idx, P14_COL_ORGAN_TYPE] = 9
                        else:
                            part_transforms_14d[fl_row_idx, P14_COL_ORGAN_TYPE] = 7
                        R_yaw = rodrigues_matrix_torch(recalculated_peduncle_axis, saved_yaw, device=device)
                        R_obj_net = (
                            R_yaw @
                            rotr_z(saved_azimuth, device) @
                            rotr_y(saved_pitch, device) @
                            rotr_x(saved_roll, device)
                        )
                        part_transforms_14d[fl_row_idx, P14_COL_ROT_0:P14_COL_ROT_5+1] = rotation_matrix_to_6d(R_obj_net)

                        if os.environ.get("HELIOS_DUMP_GEOM"):
                            fbs = flower_base.detach().cpu().numpy()
                            print(f"PTDEBUG FL {sid} {p_idx} {fl_idx} {fbs[0]:.6f} {fbs[1]:.6f} {fbs[2]:.6f} scale={scale_factor:.5f} state={state}", file=sys.stderr)

                        if compute_mesh:
                            n_obj = compute_face_normals_torch(v_obj, f_asset)
                            c_obj = obj_color.to(device).unsqueeze(0).repeat(v_obj.shape[0], 1)

                            all_verts.append(v_obj)
                            all_faces.append(f_asset + vert_offset)
                            all_normals.append(n_obj)
                            all_colors.append(c_obj)
                            all_organs.append(torch.full((v_obj.shape[0],), obj_ot, dtype=torch.int64, device=device))
                            vert_offset += v_obj.shape[0]

        if not compute_mesh or not all_verts:
            empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
            empty_o = torch.zeros((0,), dtype=torch.int64, device=device)
            return {
                'vertices': empty3, 'faces': empty_f, 'normals': empty3, 'colors': empty3,
                'organ_types': empty_o, 'part_transforms_14d': part_transforms_14d
            }

        return {
            'vertices': torch.cat(all_verts, dim=0),
            'faces': torch.cat(all_faces, dim=0),
            'normals': torch.cat(all_normals, dim=0),
            'colors': torch.cat(all_colors, dim=0),
            'organ_types': torch.cat(all_organs, dim=0),
            'part_transforms_14d': part_transforms_14d
        }

    def build_mesh_from_part_array_14d(
        self,
        part_tensor_14d: torch.Tensor,
        device: torch.device = torch.device('cpu'),
        existence_threshold: float = 0.5,
        template_organ_array: Optional[PlantOrganArray] = None,
        use_kinematics_tree: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Builds a 3D mesh directly from the 14D part-centric tensor representation.
        Each organ is placed in 3D space directly according to its (Base, 6D Rotation, Scale).
        """
        if use_kinematics_tree and template_organ_array is not None:
            recon = PlantOrganArray.from_part_tensor_14d(part_tensor_14d, template_organ_array)
            return self._build_mesh_typed(recon, device=device, existence_threshold=existence_threshold)

        p = part_tensor_14d.to(device)
        N = p.shape[0]
        if N == 0:
            empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
            empty_o = torch.zeros((0,), dtype=torch.int64, device=device)
            return {'vertices': empty3, 'faces': empty_f, 'normals': empty3, 'colors': empty3, 'organ_types': empty_o, 'part_transforms_14d': p}

        all_verts = []
        all_faces = []
        all_normals = []
        all_colors = []
        all_organs = []
        vert_offset = 0

        infl_assets = self._get_inflorescence_assets()

        child_indices = None
        leaf_pet_map = {}
        petiole_leaf_count = {}
        phytomer_bud_state = {}
        fruit_scale_map = {}
        if template_organ_array is not None:
            t_temp = template_organ_array.tensor
            if t_temp.shape[0] == N:
                child_indices = t_temp[:, T_COL_CHILD_INDEX].long().cpu().numpy()
                for i in range(N):
                    ot_i = int(t_temp[i, T_COL_ORGAN_TYPE].item())
                    sid = int(t_temp[i, T_COL_SHOOT_ID].item())
                    p_idx = int(t_temp[i, T_COL_PHYTOMER_IDX].item())
                    if ot_i == ORGAN_LEAF:
                        pet_i = int(t_temp[i, T_COL_PARENT_PETIOLE_IDX].item())
                        leaf_pet_map[i] = (sid, p_idx, pet_i)
                        petiole_leaf_count[(sid, p_idx, pet_i)] = petiole_leaf_count.get((sid, p_idx, pet_i), 0) + 1
                    elif ot_i == ORGAN_BUD:
                        phytomer_bud_state[(sid, p_idx)] = int(t_temp[i, T_COL_BUD_STATE].item())
                        fruit_scale_map[(sid, p_idx)] = float(t_temp[i, T_COL_FRUIT_SCALE].item())

        for idx in range(N):
            exist = p[idx, P14_COL_EXISTENCE].item()
            if exist < existence_threshold:
                continue

            otype = int(p[idx, P14_COL_ORGAN_TYPE].item())
            base = p[idx, P14_COL_BASE_X:P14_COL_BASE_Z+1]
            r6 = p[idx, P14_COL_ROT_0:P14_COL_ROT_5+1]
            R = rotation_6d_to_matrix(r6)
            sx = p[idx, P14_COL_SCALE_X]
            sy = p[idx, P14_COL_SCALE_Y]
            sz = p[idx, P14_COL_SCALE_Z]

            if otype == ORGAN_LEAF:
                lf_scale = sx * self.leaf_scale_factor
                child_idx = child_indices[idx] if child_indices is not None else 0
                pet_key = leaf_pet_map.get(idx)
                num_lf = petiole_leaf_count.get(pet_key, 1) if pet_key is not None else (1 if child_idx == 0 and child_indices is None else 3)
                if num_lf == 1:
                    obj_name = "CowpeaLeaf_unifoliate.obj"
                else:
                    if child_idx == 0:
                        obj_name = "CowpeaLeaf_left_highres.obj"
                    elif child_idx == 1:
                        obj_name = "CowpeaLeaf_tip_highres.obj"
                    else:
                        obj_name = "CowpeaLeaf_right_highres.obj"

                try:
                    v_raw, f_lf_b, n_tmpl = self.asset_mgr.get_mesh_with_normals(obj_name, device)
                except FileNotFoundError:
                    try:
                        v_raw, f_lf_b, n_tmpl = self.asset_mgr.get_mesh_with_normals("CowpeaLeaf_unifoliate.obj", device)
                    except FileNotFoundError:
                        continue

                v_lf_b = v_raw * lf_scale
                v_lf = (R @ v_lf_b.T).T + base
                n_lf = (R @ n_tmpl.T).T
                c_lf = self.COLOR_LEAF.to(device).unsqueeze(0).expand(v_lf.shape[0], 3)

                all_verts.append(v_lf)
                all_faces.append(f_lf_b + vert_offset)
                all_normals.append(n_lf)
                all_colors.append(c_lf)
                all_organs.append(torch.full((v_lf.shape[0],), self.OT_LEAF, dtype=torch.int64, device=device))
                vert_offset += v_lf.shape[0]

            elif otype in (ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_PEDUNCLE):
                if otype == ORGAN_PEDUNCLE and template_organ_array is not None:
                    sid = int(t_temp[idx, T_COL_SHOOT_ID].item())
                    p_idx = int(t_temp[idx, T_COL_PHYTOMER_IDX].item())
                    if phytomer_bud_state.get((sid, p_idx), 0) not in (2, 3, 4):
                        continue

                rad = sx
                len_val = sz
                if len_val <= 1e-6 or rad <= 1e-6:
                    continue

                if otype == ORGAN_INTERNODE:
                    col = self.COLOR_STEM.to(device)
                    ot_id = self.OT_STEM
                elif otype == ORGAN_PETIOLE:
                    col = self.COLOR_PETIOLE.to(device)
                    ot_id = self.OT_PETIOLE
                else:
                    col = self.COLOR_PEDUNCLE.to(device)
                    ot_id = self.OT_PEDUNCLE

                p0 = base
                p1 = base + R @ torch.tensor([0.0, 0.0, len_val], device=device)
                line = torch.stack([p0, p1])
                radii = torch.tensor([rad, rad], dtype=torch.float32, device=device)

                v_tub, f_tub, n_tub, c_tub = generate_cone_tube_mesh_torch(
                    line, radii, col, radial_subdivisions=self.tube_radial_subdivisions
                )
                if v_tub.shape[0] > 0:
                    all_verts.append(v_tub)
                    all_faces.append(f_tub + vert_offset)
                    all_normals.append(n_tub)
                    all_colors.append(c_tub)
                    all_organs.append(torch.full((v_tub.shape[0],), ot_id, dtype=torch.int64, device=device))
                    vert_offset += v_tub.shape[0]

            elif otype in (ORGAN_FLOWER, 8, 9):
                scale_factor = sx
                if template_organ_array is not None:
                    sid = int(t_temp[idx, T_COL_SHOOT_ID].item())
                    p_idx = int(t_temp[idx, T_COL_PHYTOMER_IDX].item())
                    st = phytomer_bud_state.get((sid, p_idx), 0)
                    if st not in (2, 3, 4):
                        continue
                    if st == 4 or otype == 8:
                        is_fruit = True
                        is_open = False
                    elif st == 2 or otype == 9:
                        is_fruit = False
                        is_open = False
                    else:
                        is_fruit = False
                        is_open = True
                else:
                    if otype == 8:
                        is_fruit = True
                        is_open = False
                    elif otype == 9:
                        is_fruit = False
                        is_open = False
                    else:
                        is_fruit = False
                        is_open = True

                if is_fruit:
                    if template_organ_array is not None:
                        f_scale = fruit_scale_map.get((sid, p_idx), 0.9)
                        if f_scale > 0:
                            scale_factor = sx * f_scale
                    asset_name = 'fruit'
                    obj_color = self.COLOR_FRUIT
                    obj_ot = self.OT_FRUIT
                elif is_open:
                    asset_name = 'flower_open'
                    obj_color = self.COLOR_FLOWER_OPEN
                    obj_ot = self.OT_FLOWER
                else:
                    asset_name = 'flower_closed'
                    obj_color = self.COLOR_FLOWER_CLOSED
                    obj_ot = self.OT_FLOWER

                v_asset, f_asset = infl_assets[asset_name]
                v_asset = v_asset.to(device)
                f_asset = f_asset.to(device)

                v_obj = (R @ (v_asset * scale_factor).T).T + base
                n_obj = compute_face_normals_torch(v_obj, f_asset)
                c_obj = obj_color.to(device).unsqueeze(0).repeat(v_obj.shape[0], 1)

                all_verts.append(v_obj)
                all_faces.append(f_asset + vert_offset)
                all_normals.append(n_obj)
                all_colors.append(c_obj)
                all_organs.append(torch.full((v_obj.shape[0],), obj_ot, dtype=torch.int64, device=device))
                vert_offset += v_obj.shape[0]

        if not all_verts:
            empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
            empty_o = torch.zeros((0,), dtype=torch.int64, device=device)
            return {'vertices': empty3, 'faces': empty_f, 'normals': empty3, 'colors': empty3, 'organ_types': empty_o, 'part_transforms_14d': p}

        return {
            'vertices': torch.cat(all_verts, dim=0),
            'faces': torch.cat(all_faces, dim=0),
            'normals': torch.cat(all_normals, dim=0),
            'colors': torch.cat(all_colors, dim=0),
            'organ_types': torch.cat(all_organs, dim=0),
            'part_transforms_14d': p
        }
