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
    ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF,
    ORGAN_BUD, ORGAN_BUD_ABORTED, ORGAN_PEDUNCLE, ORGAN_FLOWER, ORGAN_FRUIT, ORGAN_FLOWER_CLOSED,
    P_COL_ORGAN_TYPE, P_COL_BASE_X, P_COL_BASE_Y, P_COL_BASE_Z,
    P_COL_ROT_0, P_COL_ROT_1, P_COL_ROT_2, P_COL_ROT_3, P_COL_ROT_4, P_COL_ROT_5,
    P_COL_SCALE_X, P_COL_SCALE_Y, P_COL_SCALE_Z, P_COL_EXISTENCE, NUM_FEATURES,
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


def generate_sorghum_blade_mesh_torch(
    scale: torch.Tensor,
    aspect_ratio: float = 0.18,
    Nx: int = 24,
    Ny: int = 6,
    device=torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates Helios Sorghum/Grass parametric curved, drooping ribbon blade mesh."""
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=torch.float32, device=device)

    # Normalized longitudinal coordinate s in [0, 1]
    s = torch.linspace(0.0, 1.0, Nx + 1, device=device)
    # Normalized transversal coordinate t in [-0.5, 0.5]
    t = torch.linspace(-0.5, 0.5, Ny + 1, device=device)

    grid_s, grid_t = torch.meshgrid(s, t, indexing='ij')

    # Width profile along the leaf: sheath at s=0, peak at s=0.25~0.35, taper to tip at s=1.0
    width_envelope = torch.sin(math.pi * (grid_s.clamp(0.0, 1.0) ** 0.55)).clamp(min=0.04)
    local_width = scale * aspect_ratio * width_envelope

    # Longitudinal catenary droop (curves outwards and arches downwards)
    droop_z = scale * (0.04 * grid_s - 0.32 * (grid_s ** 2) - 0.12 * (grid_s ** 3))

    # Transversal midrib V-fold channel (V-shape crease along center midrib)
    v_fold = 0.22 * torch.abs(grid_t) * local_width * (1.0 - 0.6 * grid_s)

    pts_x = grid_s * scale
    pts_y = grid_t * local_width
    pts_z = droop_z + v_fold

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


_SORGHUM_LEAF_CACHE: Dict[Tuple[float, int, int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

def get_sorghum_leaf_template(aspect_ratio: float = 0.18, Nx: int = 24, Ny: int = 6, device=torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (aspect_ratio, Nx, Ny, str(device))
    if key not in _SORGHUM_LEAF_CACHE:
        v_temp, f_temp = generate_sorghum_blade_mesh_torch(scale=torch.tensor(1.0, device=device), aspect_ratio=aspect_ratio, Nx=Nx, Ny=Ny, device=device)
        n_temp = compute_face_normals_torch(v_temp, f_temp)
        _SORGHUM_LEAF_CACHE[key] = (v_temp, f_temp, n_temp)
    return _SORGHUM_LEAF_CACHE[key]


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


SPECIES_CONFIG: Dict[str, Dict[str, Any]] = {
    "cowpea": {
        "leaf_obj": "CowpeaLeaf_tip_highres.obj",
        "leaf_tip_obj": "CowpeaLeaf_tip_highres.obj",
        "leaf_left_obj": "CowpeaLeaf_left_highres.obj",
        "leaf_right_obj": "CowpeaLeaf_right_highres.obj",
        "leaf_unifoliate_obj": "CowpeaLeaf_unifoliate.obj",
        "leaf_aspect_ratio": 0.65,
        "flower_open_obj": "CowpeaFlower_open_yellow.obj",
        "flower_closed_obj": "CowpeaFlower_closed_yellow.obj",
        "fruit_obj": "CowpeaPod.obj",
        "fruit_load_scale": 0.47,
        "color_stem": torch.tensor([0.22, 0.45, 0.15]),
        "color_petiole": torch.tensor([0.25, 0.50, 0.18]),
        "color_leaf": torch.tensor([0.25, 0.62, 0.18]),
        "color_peduncle": torch.tensor([0.17, 0.213, 0.051]),
        "color_flower_open": torch.tensor([0.921582, 0.916492, 0.344423]),
        "color_flower_closed": torch.tensor([0.5, 0.4, 0.1]),
        "color_fruit": torch.tensor([0.299629, 0.400454, 0.209546]),
    },
    "bean": {
        "leaf_obj": "BeanLeaf_tip.obj",
        "leaf_tip_obj": "BeanLeaf_tip.obj",
        "leaf_left_obj": "BeanLeaf_left.obj",
        "leaf_right_obj": "BeanLeaf_right.obj",
        "leaf_unifoliate_obj": "BeanLeaf_unifoliate.obj",
        "leaf_aspect_ratio": 0.65,
        "flower_open_obj": "BeanFlower_open_white.obj",
        "flower_closed_obj": "BeanFlower_closed_white.obj",
        "fruit_obj": "BeanPod.obj",
        "fruit_load_scale": 0.40,
        "color_stem": torch.tensor([0.20, 0.42, 0.14]),
        "color_petiole": torch.tensor([0.23, 0.48, 0.16]),
        "color_leaf": torch.tensor([0.22, 0.58, 0.16]),
        "color_peduncle": torch.tensor([0.18, 0.24, 0.08]),
        "color_flower_open": torch.tensor([0.95, 0.95, 0.92]),
        "color_flower_closed": torch.tensor([0.72, 0.76, 0.58]),
        "color_fruit": torch.tensor([0.26, 0.38, 0.18]),
    },
    "sorghum": {
        "leaf_obj": None,  # Parametric lanceolate curved ribbon blade
        "leaf_aspect_ratio": 0.20,
        "flower_open_obj": "RiceGrain.obj",
        "flower_closed_obj": "RiceGrain.obj",
        "fruit_obj": "WheatSpike.obj",
        "fruit_load_scale": 0.35,
        "color_stem": torch.tensor([0.24, 0.44, 0.16]),
        "color_petiole": torch.tensor([0.22, 0.40, 0.14]),
        "color_leaf": torch.tensor([0.19, 0.52, 0.13]),
        "color_peduncle": torch.tensor([0.22, 0.32, 0.15]),
        "color_flower_open": torch.tensor([0.82, 0.72, 0.32]),
        "color_flower_closed": torch.tensor([0.65, 0.55, 0.25]),
        "color_fruit": torch.tensor([0.58, 0.38, 0.22]),
    },
    "soybean": {
        "leaf_obj": "SoybeanLeaf.obj",
        "leaf_aspect_ratio": 0.60,
        "flower_open_obj": "SoybeanFlower_open_white.obj",
        "flower_closed_obj": "SoybeanFlower_open_white.obj",
        "fruit_obj": "SoybeanPod.obj",
        "fruit_load_scale": 0.35,
        "color_stem": torch.tensor([0.21, 0.42, 0.14]),
        "color_petiole": torch.tensor([0.24, 0.46, 0.16]),
        "color_leaf": torch.tensor([0.23, 0.59, 0.17]),
        "color_peduncle": torch.tensor([0.18, 0.22, 0.06]),
        "color_flower_open": torch.tensor([0.90, 0.90, 0.95]),
        "color_flower_closed": torch.tensor([0.65, 0.70, 0.55]),
        "color_fruit": torch.tensor([0.32, 0.42, 0.22]),
    },
    "tomato": {
        "leaf_obj": "TomatoLeaf.obj",
        "leaf_aspect_ratio": 0.70,
        "flower_open_obj": "TomatoFlower.obj",
        "flower_closed_obj": "TomatoFlower.obj",
        "fruit_obj": "TomatoFruit.obj",
        "fruit_load_scale": 0.40,
        "color_stem": torch.tensor([0.20, 0.40, 0.12]),
        "color_petiole": torch.tensor([0.22, 0.45, 0.15]),
        "color_leaf": torch.tensor([0.16, 0.50, 0.12]),
        "color_peduncle": torch.tensor([0.18, 0.25, 0.08]),
        "color_flower_open": torch.tensor([0.95, 0.90, 0.10]),
        "color_flower_closed": torch.tensor([0.60, 0.55, 0.20]),
        "color_fruit": torch.tensor([0.85, 0.15, 0.10]),
    },
}


class HeliosPlantGeometryBuilder:
    """Builds complete PyTorch 3D plant meshes directly from PlantOrganArray Tensor (N, 93) with multi-species support."""

    def __init__(
        self,
        asset_manager: Optional[HeliosAssetManager] = None,
        species: str = "cowpea",
        use_generic_leaves: bool = False,
        leaf_scale_factor: float = 1.0,
        tube_radial_subdivisions: int = 4
    ):
        if asset_manager is None:
            asset_manager = HeliosAssetManager()
        self.asset_mgr = asset_manager
        self.species = species.lower()
        self.use_generic_leaves = use_generic_leaves
        self.leaf_scale_factor = leaf_scale_factor
        self.tube_radial_subdivisions = tube_radial_subdivisions

        # Default fallback colors
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

        self._species_infl_cache: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def _get_species_cfg(self, species_name: str) -> Dict[str, Any]:
        sp = species_name.lower().strip()
        if sp in SPECIES_CONFIG:
            return SPECIES_CONFIG[sp]
        # Match common aliases
        for k in SPECIES_CONFIG:
            if k in sp or sp in k:
                return SPECIES_CONFIG[k]
        return SPECIES_CONFIG["cowpea"]

    def _get_inflorescence_assets_for_species(self, species_name: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        sp = species_name.lower().strip()
        if sp not in self._species_infl_cache:
            cfg = self._get_species_cfg(sp)
            self._species_infl_cache[sp] = {
                'flower_open': self.asset_mgr.get_inflorescence_mesh(cfg["flower_open_obj"], load_scale=0.0),
                'flower_closed': self.asset_mgr.get_inflorescence_mesh(cfg["flower_closed_obj"], load_scale=0.0),
                'fruit': self.asset_mgr.get_inflorescence_mesh(cfg["fruit_obj"], load_scale=cfg.get("fruit_load_scale", 0.47)),
            }
        return self._species_infl_cache[sp]

    def _build_peduncle_has_flower(self, p: torch.Tensor) -> torch.Tensor:
        """Use the part-tensor record order to decide which peduncles carry flowers/fruits."""
        N = p.shape[0]
        ped_has_flower = torch.zeros(N, dtype=torch.bool, device=p.device)
        last_ped_idx = -1
        for i in range(N):
            ot = int(p[i, P_COL_ORGAN_TYPE].item())
            if ot == ORGAN_PEDUNCLE:
                last_ped_idx = i
            elif ot in (ORGAN_FLOWER, ORGAN_FRUIT, ORGAN_FLOWER_CLOSED) and last_ped_idx >= 0:
                ped_has_flower[last_ped_idx] = True
            elif ot == ORGAN_INTERNODE:
                last_ped_idx = -1
        return ped_has_flower

    def _is_dormant_peduncle(
        self,
        ped_idx: int,
        p: torch.Tensor,
        ped_has_flower: torch.Tensor,
        device: torch.device,
    ) -> bool:
        """Return True if this peduncle serves a dormant/aborted bud with no flower/fruit."""
        if ped_has_flower[ped_idx].item():
            return False

        base = p[ped_idx, P_COL_BASE_X:P_COL_BASE_Z + 1]
        for j in range(p.shape[0]):
            if j == ped_idx:
                continue
            ot = int(p[j, P_COL_ORGAN_TYPE].item())
            if ot in (ORGAN_BUD, ORGAN_BUD_ABORTED):
                bud_base = p[j, P_COL_BASE_X:P_COL_BASE_Z + 1]
                if torch.norm(bud_base - base).item() < 1e-4:
                    return True
        return False

    def build_mesh_from_part_array(
        self,
        part_tensor: torch.Tensor,
        device: torch.device = torch.device('cpu'),
        species: Optional[str] = None,
        existence_threshold: float = 0.5,
        template_organ_array: Optional[PlantOrganArray] = None,
        use_kinematics_tree: bool = False,
        soft_existence: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Builds a 3D mesh directly from the part-centric tensor representation.
        Supports multi-species plant architectures (cowpea, bean, sorghum, soybean, tomato).
        """
        p = part_tensor.to(device)
        N = p.shape[0]
        if N == 0:
            empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
            empty_o = torch.zeros((0,), dtype=torch.int64, device=device)
            return {'vertices': empty3, 'faces': empty_f, 'normals': empty3, 'colors': empty3, 'organ_types': empty_o, 'part_transforms': p}

        # Resolve species configuration
        sp_name = (species or self.species).lower()
        cfg = self._get_species_cfg(sp_name)

        col_stem = cfg["color_stem"]
        col_petiole = cfg["color_petiole"]
        col_leaf = cfg["color_leaf"]
        col_peduncle = cfg["color_peduncle"]
        col_flower_open = cfg["color_flower_open"]
        col_flower_closed = cfg["color_flower_closed"]
        col_fruit = cfg["color_fruit"]

        all_verts = []
        all_faces = []
        all_normals = []
        all_colors = []
        all_organs = []
        all_exist = []
        vert_offset = 0

        infl_assets = self._get_inflorescence_assets_for_species(sp_name)
        ped_has_flower = self._build_peduncle_has_flower(p)

        # Preload leaf templates and specialized leaflet meshes
        sorghum_leaf_tmpl = get_sorghum_leaf_template(aspect_ratio=cfg.get("leaf_aspect_ratio", 0.20), device=device)
        generic_leaf_tmpl = get_generic_leaf_template(aspect_ratio=cfg.get("leaf_aspect_ratio", 0.65), device=device)

        leaflet_tmpls = {}
        for k in ["leaf_obj", "leaf_tip_obj", "leaf_left_obj", "leaf_right_obj", "leaf_unifoliate_obj"]:
            if k in cfg and cfg[k] is not None:
                try:
                    leaflet_tmpls[k] = self.asset_mgr.get_mesh_with_normals(cfg[k], device)
                except Exception:
                    pass

        curr_petiole_leaf_count = 0

        for idx in range(N):
            exist_tensor = p[idx, P_COL_EXISTENCE]
            exist_weight = torch.sigmoid((exist_tensor - existence_threshold) * 20.0).clamp(min=1e-3)
            if soft_existence:
                # Render at full scale; existence is carried as a vertex attribute.
                # Use a linear (non-saturated) mapping so the gradient is strong
                # even when existence=0, letting organs grow from an empty plant.
                exist_weight = 1.0
                exist_attr = exist_tensor.clamp(min=0.0, max=1.0)

            otype = int(p[idx, P_COL_ORGAN_TYPE].item())
            base = p[idx, P_COL_BASE_X:P_COL_BASE_Z+1]
            r6 = p[idx, P_COL_ROT_0:P_COL_ROT_5+1]
            R = rotation_6d_to_matrix(r6.unsqueeze(0)).squeeze(0)
            sx = p[idx, P_COL_SCALE_X] * exist_weight
            sy = p[idx, P_COL_SCALE_Y] * exist_weight
            sz = p[idx, P_COL_SCALE_Z] * exist_weight

            if otype != ORGAN_LEAF:
                curr_petiole_leaf_count = 0

            if otype == ORGAN_LEAF:
                lf_scale = sx * self.leaf_scale_factor
                if lf_scale <= 1e-6:
                    continue

                leaf_sub_idx = curr_petiole_leaf_count
                curr_petiole_leaf_count += 1

                if sp_name == "sorghum":
                    v_tmpl, f_tmpl, n_tmpl = sorghum_leaf_tmpl
                    v_lf_raw = v_tmpl
                    f_lf_b = f_tmpl
                elif self.use_generic_leaves or len(leaflet_tmpls) == 0:
                    v_tmpl, f_tmpl, n_tmpl = generic_leaf_tmpl
                    v_lf_raw = v_tmpl
                    f_lf_b = f_tmpl
                else:
                    # Select appropriate leaflet mesh
                    target_key = "leaf_obj"
                    if leaf_sub_idx == 0 and "leaf_left_obj" in leaflet_tmpls:
                        target_key = "leaf_left_obj"
                    elif leaf_sub_idx == 1 and "leaf_tip_obj" in leaflet_tmpls:
                        target_key = "leaf_tip_obj"
                    elif leaf_sub_idx == 2 and "leaf_right_obj" in leaflet_tmpls:
                        target_key = "leaf_right_obj"

                    if target_key in leaflet_tmpls:
                        v_lf_raw, f_lf_b, n_tmpl = leaflet_tmpls[target_key]
                    else:
                        v_lf_raw, f_lf_b, n_tmpl = leaflet_tmpls.get("leaf_obj", generic_leaf_tmpl)

                v_lf_b = v_lf_raw * lf_scale
                v_lf = (R @ v_lf_b.T).T + base
                n_lf = (R @ n_tmpl.T).T
                c_lf = col_leaf.to(device).unsqueeze(0).expand(v_lf.shape[0], 3)

                all_verts.append(v_lf)
                all_faces.append(f_lf_b + vert_offset)
                all_normals.append(n_lf)
                all_colors.append(c_lf)
                all_organs.append(torch.full((v_lf.shape[0],), self.OT_LEAF, dtype=torch.int64, device=device))
                if soft_existence:
                    all_exist.append(exist_attr.expand(v_lf.shape[0]))
                vert_offset += v_lf.shape[0]

            elif otype in (ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_PEDUNCLE):
                rad = sx
                len_val = sz
                if len_val <= 1e-6 or rad <= 1e-6:
                    continue

                if otype == ORGAN_PEDUNCLE:
                    if self._is_dormant_peduncle(idx, p, ped_has_flower, device):
                        continue

                if otype == ORGAN_INTERNODE:
                    col = col_stem.to(device)
                    ot_id = self.OT_STEM
                elif otype == ORGAN_PETIOLE:
                    col = col_petiole.to(device)
                    ot_id = self.OT_PETIOLE
                else:
                    col = col_peduncle.to(device)
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
                    if soft_existence:
                        all_exist.append(exist_attr.expand(v_tub.shape[0]))
                    vert_offset += v_tub.shape[0]

            elif otype in (ORGAN_FLOWER, ORGAN_FRUIT, ORGAN_FLOWER_CLOSED):
                scale_factor = sx
                if scale_factor <= 1e-6:
                    continue
                if otype == ORGAN_FRUIT:
                    asset_name = 'fruit'
                    obj_color = col_fruit
                    obj_ot = self.OT_FRUIT
                elif otype == ORGAN_FLOWER_CLOSED:
                    asset_name = 'flower_closed'
                    obj_color = col_flower_closed
                    obj_ot = self.OT_FLOWER
                else:
                    asset_name = 'flower_open'
                    obj_color = col_flower_open
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
                if soft_existence:
                    all_exist.append(exist_attr.expand(v_obj.shape[0]))
                vert_offset += v_obj.shape[0]

        if not all_verts:
            empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
            empty_o = torch.zeros((0,), dtype=torch.int64, device=device)
            return {'vertices': empty3, 'faces': empty_f, 'normals': empty3, 'colors': empty3, 'organ_types': empty_o, 'part_transforms': p}

        out = {
            'vertices': torch.cat(all_verts, dim=0),
            'faces': torch.cat(all_faces, dim=0),
            'normals': torch.cat(all_normals, dim=0),
            'colors': torch.cat(all_colors, dim=0),
            'organ_types': torch.cat(all_organs, dim=0),
            'part_transforms': p
        }
        if soft_existence:
            out['existence'] = torch.cat(all_exist, dim=0)
        return out

