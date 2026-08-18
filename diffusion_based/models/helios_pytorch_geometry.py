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
    ORGAN_BUD, ORGAN_PEDUNCLE, ORGAN_FLOWER, ORGAN_FRUIT, ORGAN_FLOWER_CLOSED,
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

    def build_mesh_from_part_array(
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

        for idx in range(N):
            exist = p[idx, P_COL_EXISTENCE].item()
            if exist < existence_threshold:
                continue

            otype = int(p[idx, P_COL_ORGAN_TYPE].item())
            base = p[idx, P_COL_BASE_X:P_COL_BASE_Z+1]
            r6 = p[idx, P_COL_ROT_0:P_COL_ROT_5+1]
            R = rotation_6d_to_matrix(r6)
            sx = p[idx, P_COL_SCALE_X]
            sy = p[idx, P_COL_SCALE_Y]
            sz = p[idx, P_COL_SCALE_Z]

            if otype == ORGAN_LEAF:
                lf_scale = sx * self.leaf_scale_factor
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

            elif otype in (ORGAN_FLOWER, ORGAN_FRUIT, ORGAN_FLOWER_CLOSED):
                scale_factor = sx
                if otype == ORGAN_FRUIT:
                    asset_name = 'fruit'
                    obj_color = self.COLOR_FRUIT
                    obj_ot = self.OT_FRUIT
                elif otype == ORGAN_FLOWER_CLOSED:
                    asset_name = 'flower_closed'
                    obj_color = self.COLOR_FLOWER_CLOSED
                    obj_ot = self.OT_FLOWER
                else:
                    asset_name = 'flower_open'
                    obj_color = self.COLOR_FLOWER_OPEN
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

