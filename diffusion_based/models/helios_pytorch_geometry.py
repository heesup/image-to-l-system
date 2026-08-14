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
    COL_NUM_FLOWERS, COL_FLOWER_OFFSET, COL_FL0_PITCH, COL_FL0_YAW, COL_FL0_ROLL, COL_FL0_AZIMUTH, COL_FL0_BASE_SCALE
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


class HeliosAssetManager:
    """Loads and caches Helios OBJ assets for PyTorch rendering."""
    def __init__(self, asset_dir: str = ASSET_DIR):
        self.asset_dir = asset_dir
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_mesh(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if name not in self.cache:
            path = os.path.join(self.asset_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Helios asset missing: {path}")
            self.cache[name] = load_obj_file(path)
        v, f = self.cache[name]
        return v.clone(), f.clone()


def rotr_x(angle_rad: torch.Tensor, device=torch.device('cpu')) -> torch.Tensor:
    if not isinstance(angle_rad, torch.Tensor):
        angle_rad = torch.tensor(angle_rad, dtype=torch.float32, device=device)
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, c, -s]),
        torch.stack([zero, s, c])
    ])

def rotr_y(angle_rad: torch.Tensor, device=torch.device('cpu')) -> torch.Tensor:
    if not isinstance(angle_rad, torch.Tensor):
        angle_rad = torch.tensor(angle_rad, dtype=torch.float32, device=device)
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.stack([
        torch.stack([c, zero, s]),
        torch.stack([zero, one, zero]),
        torch.stack([-s, zero, c])
    ])

def rotr_z(angle_rad: torch.Tensor, device=torch.device('cpu')) -> torch.Tensor:
    if not isinstance(angle_rad, torch.Tensor):
        angle_rad = torch.tensor(angle_rad, dtype=torch.float32, device=device)
    c, s = torch.cos(angle_rad), torch.sin(angle_rad)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    return torch.stack([
        torch.stack([c, -s, zero]),
        torch.stack([s, c, zero]),
        torch.stack([zero, zero, one])
    ])


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
    sin_theta = torch.sqrt((1.0 - cos_theta**2).clamp(min=0.0))

    zero = torch.tensor(0.0, dtype=torch.float32, device=device)
    K = torch.stack([
        torch.stack([zero, -axis[2], axis[1]]),
        torch.stack([axis[2], zero, -axis[0]]),
        torch.stack([-axis[1], axis[0], zero])
    ])

    R = torch.eye(3, dtype=torch.float32, device=device) + sin_theta * K + (1.0 - cos_theta) * (K @ K)
    return R


def rotate_vector_about_axis(vec: torch.Tensor, axis: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation formula: rotates 3D vector 'vec' about unit vector 'axis' by 'angle_rad'.

    Numerically safe: if the axis is near-zero or the angle is near-zero, avoid
    operations whose gradients explode (division by zero, 0*inf forms).
    """
    axis_norm = torch.linalg.norm(axis)
    if axis_norm < 1e-4:
        # Zero-length axis: no rotation, return vec unchanged with stable gradient
        return vec.clone()
    axis = axis / axis_norm

    # Clamp angle magnitude to avoid the unstable 1-cos(a) near 2*pi multiple,
    # and wrap via angle sign so sin/cos are always finite.
    angle_safe = angle_rad
    if torch.abs(angle_rad) < 1e-6:
        # Tiny angle: first-order expansion, no singular 1-cos term
        return vec + torch.linalg.cross(axis, vec) * angle_rad

    cos_a = torch.cos(angle_safe)
    sin_a = torch.sin(angle_safe)
    return vec * cos_a + torch.linalg.cross(axis, vec) * sin_a + axis * torch.dot(axis, vec) * (1.0 - cos_a)


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
    proto_v, faces_t = _make_straight_tube_prototype(n_seg, n_rad, device)
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

    # Per-segment rotation from +Z to segment axis
    axis = centerline[1:] - centerline[:-1]               # (n_seg, 3)
    L = torch.linalg.norm(axis, dim=-1, keepdim=True)
    axis_norm = axis / (L + 1e-8)
    # For each vertex, pick segment axis and compute rotation from +Z
    seg_axis = axis_norm[seg_idx]                         # (V, 3)
    plus_z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(V, 3)
    R = _get_rotation_matrix_between_vectors_batch(plus_z, seg_axis)  # (V, 3, 3)

    # Radial offset = scaled xy, then rotated
    offsets = torch.zeros(V, 3, device=device)
    offsets[:, 0] = xy[:, 0] * r_interp
    offsets[:, 1] = xy[:, 1] * r_interp
    offsets = torch.einsum("vij,vj->vi", R, offsets)

    verts_t = pos + offsets

    # Normals: rotate the prototype radial normals (same as offsets direction, z=0)
    normals = torch.zeros(V, 3, device=device)
    normals[:, 0] = xy[:, 0]
    normals[:, 1] = xy[:, 1]
    normals = torch.einsum("vij,vj->vi", R, normals)
    normals = normals / (torch.linalg.norm(normals, dim=-1, keepdim=True) + 1e-8)

    colors_t = color.unsqueeze(0).repeat(verts_t.shape[0], 1)
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

    def build_mesh_from_organ_array(
        self,
        organ_array: PlantOrganArray,
        device: torch.device = torch.device('cpu'),
        max_leaves: Optional[int] = None,
        existence_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Processes PlantOrganArray Tensor (N, 93) with sequential shoot forward kinematics.
        Renders each shoot individually and connects child shoots to parent petiole/node attachments.
        """
        # If the input uses the typed (N, 40) layout, convert it to the legacy
        # (N, 94) phytomer-slot layout so the existing geometry builder can
        # consume it unchanged. This keeps rendering pixel-identical while
        # downstream code migrates to the typed representation. Use the
        # differentiable conversion so image-loss gradients can flow all the
        # way back to the typed tensor.
        if organ_array.is_typed:
            legacy_tensor = organ_array.to_legacy_tensor_diff()
            organ_array = PlantOrganArray(legacy_tensor, raw_metadata=[])

        t = organ_array.tensor.to(device)
        existence = organ_array.existence.to(device).clamp(0.0, 1.0)
        N = t.shape[0]

        if N == 0:
            empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
            empty_o = torch.zeros((0,), dtype=torch.int64, device=device)
            return {'vertices': empty3, 'faces': empty_f, 'normals': empty3, 'colors': empty3, 'organ_types': empty_o}

        all_verts = []
        all_faces = []
        all_normals = []
        all_colors = []
        all_organs = []
        vert_offset = 0

        shoots_dict: Dict[int, List[int]] = {}
        for i in range(N):
            sid = int(t[i, COL_SHOOT_ID].item())
            if sid not in shoots_dict:
                shoots_dict[sid] = []
            shoots_dict[sid].append(i)

        sorted_shoot_ids = sorted(shoots_dict.keys())
        shoot_id_to_sorted_idx = {sid: i for i, sid in enumerate(sorted_shoot_ids)}
        node_output_info: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Tensor-based node outputs for soft parent aggregation.
        # We pad arrays to N, storing per-node outputs. Missing entries are zeros.
        node_tip_positions = torch.zeros((N, 3), dtype=torch.float32, device=device)
        node_internode_axes = torch.zeros((N, 3), dtype=torch.float32, device=device)
        node_petiole_axes = torch.zeros((N, 2, 3), dtype=torch.float32, device=device)
        node_has_petiole = torch.zeros((N, 2), dtype=torch.float32, device=device)

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

        def compute_shoot_base(sid: int, first_row: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute shoot base position, internode axis, and petiole axis.

            Uses soft parent aggregation when parent_logits/parent_candidates are provided,
            otherwise falls back to the hard parent columns in the tensor.
            """
            if use_soft_parent:
                s_idx = shoot_id_to_sorted_idx[sid]
                cand = parent_candidates[s_idx]  # (K, 3)
                cand_shoot = cand[:, 0]
                cand_node = cand[:, 1]
                cand_pet = cand[:, 2]

                # Validity mask: candidate node index must be non-negative and parent shoot must be valid
                valid_mask = (cand_node >= 0).float() * (cand_shoot >= 0).float()

                # Existence mask for candidate parent nodes
                cand_exist = existence[cand_node.clamp(min=0)]
                active_mask = valid_mask * (cand_exist > 0.5).float()

                logits = parent_logits[s_idx]
                # Numerically stable softmax with active mask. Candidates with zero active mask are given a very negative logit.
                masked_logits = torch.where(active_mask > 0.5, logits, torch.full_like(logits, -1e9))
                weights = torch.softmax(masked_logits, dim=0)
                # If all candidates are inactive (e.g. root shoot with negative shoot id), fall back to one-hot on the first candidate.
                if weights.sum() < 1e-8 or weights.isnan().any():
                    weights = torch.zeros_like(weights)
                    weights[0] = 1.0

                # Candidate shoot IDs may be arbitrary. Build a mapping from (shoot_id, node_idx) -> node tensor index.
                # We precomputed node_tip_positions etc. by loop order, keyed by linear node index n_idx.
                # Detach candidate geometric features so gradient only flows through the selection weights,
                # making soft-parent topology optimization stable. The candidate anchors are fixed per step.
                cand_tip = node_tip_positions[cand_node].detach()          # (K, 3)
                cand_axis = node_internode_axes[cand_node].detach()      # (K, 3)

                # Petiole axis: requested petiole index may not exist on the candidate node.
                # Fall back to petiole 0 axis if the requested petiole is unavailable, matching hard-parent behavior.
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
                # Use weighted axis directly; downstream rotate_vector_about_axis normalizes internally.
                parent_internode_axis = (weights.unsqueeze(-1) * cand_axis).sum(dim=0)
                parent_petiole_axis = (weights.unsqueeze(-1) * cand_pet_axis).sum(dim=0)
                # If no candidate has a valid petiole axis, fall back to default
                valid_pet = (weights * cand_has_pet).sum()
                if valid_pet < 1e-8:
                    parent_petiole_axis = torch.tensor([0.0, -1.0, 0.0], device=device)

                # Negative candidate shoot_id means root (same as hard logic)
                if cand_shoot[weights.argmax()].item() < 0:
                    shoot_base_pos = torch.tensor([0.0, 0.0, 0.0], device=device)
                    parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
                    parent_petiole_axis = torch.tensor([0.0, -1.0, 0.0], device=device)
                return shoot_base_pos, parent_internode_axis, parent_petiole_axis

            parent_sid = int(first_row[COL_PARENT_SHOOT_ID].item())
            parent_node_idx = int(first_row[COL_PARENT_NODE_IDX].item())
            parent_petiole_index = int(first_row[COL_PARENT_PETIOLE_IDX].item())

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
                # Reconstruction (InputOutput.cpp:1457): the first internode of a child shoot
                # starts at the parent internode tip (no 0.9*radius petiole offset).
                shoot_base_pos = p_info['tip']
            return shoot_base_pos, parent_internode_axis, parent_petiole_axis

        deg2rad = torch.tensor(math.pi / 180.0, dtype=torch.float32, device=device)
        init_leaf_dir = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
        rendered_leaf_groups = 0

        for sid in sorted_shoot_ids:
            node_indices = shoots_dict[sid]
            first_row = t[node_indices[0]]

            base_pitch_rad = first_row[COL_SHOOT_ROT_PITCH] * deg2rad
            base_yaw_rad = first_row[COL_SHOOT_ROT_YAW] * deg2rad
            base_roll_rad = first_row[COL_SHOOT_ROT_ROLL] * deg2rad

            shoot_base_pos, parent_internode_axis, parent_petiole_axis = compute_shoot_base(sid, first_row)

            curr_pos = shoot_base_pos.clone()
            prev_internode_axis = parent_internode_axis
            prev_petiole_axis = parent_petiole_axis

            z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
            # Reconstructed gravitropic curvature (PlantLibrary.cpp cowpea trifoliate/unifoliate = 200);
            # not stored per-shoot in the XML, so it is taken from the fixed cowpea shoot library.
            gravitropic_curvature = 200.0

            for p_idx_in_shoot, n_idx in enumerate(node_indices):
                row = t[n_idx]
                p_idx = int(row[COL_PHYTOMER_IDX].item())
                node_exist = existence[n_idx]

                # ---- Internode orientation vectors (matches InputOutput.cpp recomputeInternodeOrientationVectors_local) ----
                petiole_rot_axis = torch.linalg.cross(prev_internode_axis, prev_petiole_axis)
                if torch.linalg.norm(petiole_rot_axis) < 1e-6:
                    petiole_rot_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
                else:
                    petiole_rot_axis = petiole_rot_axis / torch.linalg.norm(petiole_rot_axis)

                inode_pitch_rad = row[COL_INODE_PITCH] * deg2rad
                inode_phyllo_rad = row[COL_INODE_PHYLLO_ANG] * deg2rad

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

                # shoot_bending_axis: cross(internode_axis, z), fallback to (0,1,0) if parallel to z
                shoot_bending_axis = torch.linalg.cross(i_axis, z_axis)
                shoot_bending_norm = torch.linalg.norm(shoot_bending_axis)
                if shoot_bending_norm < 1e-6:
                    shoot_bending_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
                else:
                    shoot_bending_axis = shoot_bending_axis / shoot_bending_norm

                # ---- Internode tube (matches InputOutput.cpp:1483-1510) ----
                inode_len = torch.clamp(row[COL_INODE_LEN], min=1e-4) * node_exist
                inode_rad = torch.clamp(row[COL_INODE_RAD], min=1e-4) * node_exist
                seg_cnt = max(1, int(row[COL_INODE_LEN_SEGS].item()))
                seg_len = inode_len / seg_cnt
                seg_len_max = torch.clamp(row[COL_INODE_LEN_MAX], min=1e-4) / seg_cnt

                curv_p0, curv_p1 = row[COL_CURV_PERT_0], row[COL_CURV_PERT_1]
                yaw_p0, yaw_p1 = row[COL_YAW_PERT_0], row[COL_YAW_PERT_1]

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

                v_tub, f_tub, n_tub, c_tub = generate_cone_tube_mesh_torch(
                    inode_line, inode_radii, self.COLOR_STEM.to(device), radial_subdivisions=self.tube_radial_subdivisions
                )

                if v_tub.shape[0] > 0:
                    all_verts.append(v_tub)
                    all_faces.append(f_tub + vert_offset)
                    all_normals.append(n_tub)
                    all_colors.append(c_tub)
                    all_organs.append(torch.zeros(v_tub.shape[0], dtype=torch.int64, device=device)) # Organ 0 = Stem
                    vert_offset += v_tub.shape[0]

                # Apply continuous existence scale to petiole length and radii
                # (node_exist is in [0,1] from organ_array.existence)

                curr_pos = inode_line[-1]
                inode_tip_axis = step_dir / (torch.linalg.norm(step_dir) + 1e-6)

                if os.environ.get("HELIOS_DUMP_GEOM"):
                    tp = curr_pos.detach().cpu().numpy()
                    print(f"PTDEBUG I {sid} {p_idx} 1 {tp[0]:.6f} {tp[1]:.6f} {tp[2]:.6f}", file=sys.stderr)

                # ---- Petiole & Leaf Geometry (matches InputOutput.cpp recomputePetioleOrientationVectors + 1660-2019) ----
                pet_axes_stored = {}
                node_info = {
                    'tip': curr_pos,
                    'internode_axis': inode_tip_axis,
                    'radius': inode_rad,
                }
                node_tip_positions[n_idx] = curr_pos
                node_internode_axes[n_idx] = inode_tip_axis
                if 0 in pet_axes_stored:
                    node_petiole_axes[n_idx, 0] = pet_axes_stored[0]
                    node_has_petiole[n_idx, 0] = 1.0
                if 1 in pet_axes_stored:
                    node_petiole_axes[n_idx, 1] = pet_axes_stored[1]
                    node_has_petiole[n_idx, 1] = 1.0

                # Save node_info into dict as well for hard-parent fallback equivalence
                node_output_info[(sid, p_idx)] = node_info

                def process_petiole(p_len_raw, p_rad_raw, p_pitch_deg, p_curv_deg, p_cls, p_taper, p_seg_cnt,
                                    num_leaves, leaf_cols, lflt_scale, lflt_offset, petiole_index):
                    nonlocal rendered_leaf_groups, vert_offset
                    pet_pitch_rad = p_pitch_deg * deg2rad
                    pet_axis = rotate_vector_about_axis(i_axis, petiole_rot_axis, torch.abs(pet_pitch_rad))
                    pet_rot_ax = petiole_rot_axis.clone()
                    if p_idx_in_shoot != 0 and inode_phyllo_rad != 0.0:
                        pet_axis = rotate_vector_about_axis(pet_axis, i_axis, inode_phyllo_rad)
                        pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, inode_phyllo_rad)
                    if petiole_index > 0:
                        petioles_per_internode = 2.0 if int(row[COL_HAS_PET1].item()) > 0 else 1.0
                        budrot = torch.tensor(petiole_index * 2.0 * math.pi / petioles_per_internode, device=device)
                        pet_axis = rotate_vector_about_axis(pet_axis, i_axis, budrot)
                        pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, budrot)
                    pet_axis = pet_axis / (torch.linalg.norm(pet_axis) + 1e-12)
                    pet_axes_stored[petiole_index] = pet_axis.clone()

                    # Scale petiole geometry by node existence (continuous 0..1)
                    p_len = p_len_raw * node_exist
                    p_rad = p_rad_raw * node_exist
                    if p_len <= 0 or p_rad <= 0:
                        return

                    pet_rot_ax_norm = pet_rot_ax / (torch.linalg.norm(pet_rot_ax) + 1e-8)
                    pet_base = inode_line[-1]
                    seq_len = p_len / p_seg_cnt

                    pet_line_list = [pet_base]
                    curr_pet_p = pet_base
                    curr_pet_dir = pet_axis.clone()
                    for ps in range(p_seg_cnt):
                        # petiole_curvature in XML is degrees per unit length; per-segment
                        # rotation = -deg2rad(curvature * seg_len) (InputOutput.cpp:1745)
                        curv_per_seg = p_curv_deg * seq_len * deg2rad
                        if torch.abs(curv_per_seg) > 1e-12:
                            curr_pet_dir = rotate_vector_about_axis(curr_pet_dir, pet_rot_ax_norm, -curv_per_seg)
                        curr_pet_p = curr_pet_p + curr_pet_dir * seq_len
                        pet_line_list.append(curr_pet_p)

                    pet_line = torch.stack(pet_line_list)
                    # Radial taper as in reconstruction (InputOutput.cpp:1751-1752):
                    #   rad(j) = current_leaf_scale_factor * petiole_radius * (1 - taper/Ndiv * j)
                    jj = torch.linspace(0.0, p_seg_cnt, pet_line.shape[0], device=device)
                    pet_radii = p_cls * p_rad * (1.0 - p_taper / float(p_seg_cnt) * jj)
                    pet_radii = torch.clamp(pet_radii, min=1e-6)

                    v_pet, f_pet, n_pet, c_pet = generate_cone_tube_mesh_torch(
                        pet_line, pet_radii, self.COLOR_PETIOLE.to(device), radial_subdivisions=self.tube_radial_subdivisions
                    )

                    if v_pet.shape[0] > 0:
                        all_verts.append(v_pet)
                        all_faces.append(f_pet + vert_offset)
                        all_normals.append(n_pet)
                        all_colors.append(c_pet)
                        all_organs.append(torch.ones(v_pet.shape[0], dtype=torch.int64, device=device)) # Organ 1 = Petiole
                        vert_offset += v_pet.shape[0]

                    # Transformed Compound Leaves attached to Petiole (InputOutput.cpp:1922-2019)
                    pet_tip = pet_line[-1]
                    # Final petiole tip axis after curvature (getPetioleAxisVector(1, petiole))
                    pet_tip_axis = pet_line[-1] - pet_line[-2]
                    pet_tip_axis = pet_tip_axis / (torch.linalg.norm(pet_tip_axis) + 1e-8)

                    if os.environ.get("HELIOS_DUMP_GEOM"):
                        iax = inode_tip_axis.detach().cpu().numpy()
                        pax2 = pet_tip_axis.detach().cpu().numpy()
                        print(f"PTDEBUG AX {sid} {p_idx} {petiole_index} {iax[0]:.6f} {iax[1]:.6f} {iax[2]:.6f} {pax2[0]:.6f} {pax2[1]:.6f} {pax2[2]:.6f}", file=sys.stderr)

                    if os.environ.get("HELIOS_DUMP_GEOM"):
                        pt = pet_line[-1].detach().cpu().numpy()
                        print(f"PTDEBUG P {sid} {p_idx} {petiole_index} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}", file=sys.stderr)

                    if num_leaves > 0 and (max_leaves is None or rendered_leaf_groups < max_leaves):
                        rendered_leaf_groups += 1

                        for lf_i in range(num_leaves):
                            if lf_i >= len(leaf_cols):
                                break
                            sc_col, p_col, y_col, r_col = leaf_cols[lf_i]
                            l_scale = row[sc_col]
                            l_pitch_raw = row[p_col] * deg2rad
                            l_yaw = row[y_col] * deg2rad  # XML stores signed yaw_rot for lateral leaflets
                            l_roll_raw = row[r_col] * deg2rad

                            ind_from_tip = float(lf_i) - float(num_leaves - 1) / 2.0
                            compound_rotation = 0.0
                            if num_leaves > 1:
                                if lf_i == (num_leaves - 1) / 2.0:
                                    compound_rotation = 0.0
                                elif lf_i < (num_leaves - 1) / 2.0:
                                    compound_rotation = -0.5 * math.pi
                                else:
                                    compound_rotation = 0.5 * math.pi

                            # XML <leaf_scale> stores the final leaf_scale (meters); the OBJ prototype
                            # spans ~1 unit, so a scale factor of 1.0 maps it directly.
                            # Scale by node existence so absent organs contribute no visible geometry.
                            tot_scale = l_scale * self.leaf_scale_factor * node_exist

                            if self.use_generic_leaves:
                                v_lf_b, f_lf_b = generate_generic_leaf_mesh_torch(scale=tot_scale, aspect_ratio=0.65, Nx=8, Ny=8, device=device)
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
                                    v_lf_b, f_lf_b = self.asset_mgr.get_mesh(obj_name)
                                except FileNotFoundError:
                                    continue

                                v_lf_b = v_lf_b.to(device)
                                f_lf_b = f_lf_b.to(device)
                                v_lf_b = v_lf_b * tot_scale

                            # Helios leaf rotations (InputOutput.cpp:1966-1999), roll -> pitch -> yaw -> azimuth.
                            asin_pz = torch.asin(torch.clamp(pet_tip_axis[2], -1.0, 1.0))

                            if num_leaves == 1:
                                # (acos(internode_tip_axis.z) - saved_roll) * sign, sign=1 in reconstruction
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

                            R_leaf = (
                                rotr_z(azimuth_rot, device) @
                                rotr_z(yaw_rot, device) @
                                rotr_y(-pitch_rot, device) @
                                rotr_x(roll_rot, device)
                            )

                            v_lf_rot = (R_leaf @ v_lf_b.T).T

                            # Leaf base position with leaflet offset (InputOutput.cpp:2001-2015)
                            leaf_base = pet_tip
                            if num_leaves > 1 and lflt_offset > 0.0 and ind_from_tip != 0:
                                offset = (abs(ind_from_tip) - 0.5) * lflt_offset * p_len
                                frac = 1.0 - offset / torch.clamp(p_len, min=1e-6)
                                frac = torch.clamp(frac, 0.0, 1.0)
                                if not (torch.isnan(frac) or torch.isinf(frac)):
                                    leaf_base = interpolate_tube_torch(pet_line, float(frac.clamp(0.0, 1.0).item()))

                            v_lf = v_lf_rot + leaf_base

                            if os.environ.get("HELIOS_DUMP_GEOM"):
                                lb = leaf_base.detach().cpu().numpy()
                                print(f"PTDEBUG L {sid} {p_idx} {petiole_index} {lf_i} {lb[0]:.6f} {lb[1]:.6f} {lb[2]:.6f}", file=sys.stderr)

                            n_lf = compute_face_normals_torch(v_lf, f_lf_b)

                            c_lf = self.COLOR_LEAF.to(device).unsqueeze(0).repeat(v_lf.shape[0], 1)

                            all_verts.append(v_lf)
                            all_faces.append(f_lf_b + vert_offset)
                            all_normals.append(n_lf)
                            all_colors.append(c_lf)
                            all_organs.append(torch.full((v_lf.shape[0],), 2, dtype=torch.int64, device=device)) # Organ 2 = Leaf
                            vert_offset += v_lf.shape[0]

                process_petiole(
                    row[COL_PET0_LEN], row[COL_PET0_RAD], row[COL_PET0_PITCH], row[COL_PET0_CURV],
                    row[COL_PET0_LEAF_SCALE], row[COL_PET0_TAPER],
                    max(1, int(row[COL_PET0_LEN_SEGS].item())),
                    int(row[COL_PET0_NUM_LEAVES].item()),
                    [(COL_PET0_L0_SCALE, COL_PET0_L0_PITCH, COL_PET0_L0_YAW, COL_PET0_L0_ROLL),
                     (COL_PET0_L1_SCALE, COL_PET0_L1_PITCH, COL_PET0_L1_YAW, COL_PET0_L1_ROLL),
                     (COL_PET0_L2_SCALE, COL_PET0_L2_PITCH, COL_PET0_L2_YAW, COL_PET0_L2_ROLL)],
                    row[COL_PET0_LFLT_SCALE], row[COL_PET0_LFLT_OFFSET], 0,
                )

                if int(row[COL_HAS_PET1].item()) > 0:
                    process_petiole(
                        row[COL_PET1_LEN], row[COL_PET1_RAD], row[COL_PET1_PITCH], row[COL_PET1_CURV],
                        row[COL_PET1_LEAF_SCALE], row[COL_PET1_TAPER],
                        max(1, int(row[COL_PET1_LEN_SEGS].item())),
                        int(row[COL_PET1_NUM_LEAVES].item()),
                        [(COL_PET1_L0_SCALE, COL_PET1_L0_PITCH, COL_PET1_L0_YAW, COL_PET1_L0_ROLL)],
                        row[COL_PET1_LFLT_SCALE], row[COL_PET1_LFLT_OFFSET], 1,
                    )

                node_info['petiole_axes'] = pet_axes_stored
                if 0 in pet_axes_stored:
                    node_info['petiole_axis'] = pet_axes_stored[0].clone()
                # Update parent context for the next phytomer on this shoot
                prev_internode_axis = inode_tip_axis
                if 0 in pet_axes_stored:
                    prev_petiole_axis = pet_axes_stored[0]
                else:
                    ghost = torch.linalg.cross(inode_tip_axis, z_axis)
                    if torch.linalg.norm(ghost) < 0.01:
                        ghost = torch.tensor([0.0, 1.0, 0.0], device=device)
                    prev_petiole_axis = ghost / torch.linalg.norm(ghost)

        if not all_verts:
            empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
            empty_o = torch.zeros((0,), dtype=torch.int64, device=device)
            return {'vertices': empty3, 'faces': empty_f, 'normals': empty3, 'colors': empty3, 'organ_types': empty_o}

        return {
            'vertices': torch.cat(all_verts, dim=0),
            'faces': torch.cat(all_faces, dim=0),
            'normals': torch.cat(all_normals, dim=0),
            'colors': torch.cat(all_colors, dim=0),
            'organ_types': torch.cat(all_organs, dim=0)
        }
