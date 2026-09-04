"""
PyTorch Geometry Generator for Helios Plant Architecture.
Converts PlantOrganArray Tensor (N, 93) directly into 3D meshes (internode tubes, petiole tubes, compound leaf meshes, flowers).
Supports both quad-triangulated 3D OBJ leaf assets and Helios GenericLeafPrototype parametric 15cm base leaf meshes.
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
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
    aspect_ratio: float = 0.7,
    midrib_fold_fraction: float = 0.2,
    longitudinal_curvature: float = -0.2,
    lateral_curvature: float = -0.4,
    petiole_roll: float = 0.0,
    wave_period: float = 0.0,
    wave_amplitude: float = 0.0,
    Nx: int = 6,
    Ny: Optional[int] = None,
    device=torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates exact Helios C++ GenericLeafPrototype parametric curved leaf mesh (Assets.cpp:45-160)."""
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=torch.float32, device=device)

    if Ny is None:
        Ny = int(math.ceil(aspect_ratio * float(Nx)))
        if Ny % 2 != 0:
            Ny += 1

    dx = 1.0 / float(Nx)
    dy = aspect_ratio / float(Ny)

    verts_grid = []
    for j in range(Ny + 1):
        row_verts = []
        dtheta = 0.0
        for i in range(Nx + 1):
            x = float(i) * dx
            y = float(j) * dy - 0.5 * aspect_ratio

            # midrib leaf folding (Assets.cpp:69-70)
            y_fold = math.cos(0.5 * midrib_fold_fraction * math.pi) * y
            z_fold = math.sin(0.5 * midrib_fold_fraction * math.pi) * abs(y)

            # x-curvature & y-curvature (Assets.cpp:72-76)
            z_xcurve = longitudinal_curvature * (x ** 4)
            z_ycurve = lateral_curvature * ((y / aspect_ratio) ** 4)

            # petiole roll (Assets.cpp:78-83)
            z_petiole = 0.0
            if petiole_roll != 0.0:
                sign_pr = petiole_roll / abs(petiole_roll)
                z_petiole = min(0.1, petiole_roll * ((7.0 * y / aspect_ratio) ** 4) * math.exp(-70.0 * x)) - 0.01 * sign_pr

            # wave displacement (Assets.cpp:89-93)
            z_wave = 0.0
            if wave_period > 0.0 and wave_amplitude > 0.0:
                wave_phase = (x + wave_period * float(j >= 0.5 * Ny)) * math.pi / wave_period
                z_wave = 2.0 * abs(y) * wave_amplitude * math.sin(wave_phase)

            pt = torch.tensor([x, y_fold, z_fold + z_ycurve + z_petiole], dtype=torch.float32, device=device)
            rot_angle = 0.0

            # longitudinal curvature rotation about (0, 1, 0) (Assets.cpp:99-103)
            if longitudinal_curvature != 0.0 and i > 0:
                dtheta -= math.atan(4.0 * longitudinal_curvature * (x ** 3) * dx)
                c, s = math.cos(dtheta), math.sin(dtheta)
                pt_x = pt[0] * c + pt[2] * s
                pt_z = -pt[0] * s + pt[2] * c
                pt = torch.tensor([pt_x, pt[1], pt_z], dtype=torch.float32, device=device)
                rot_angle += dtheta

            # apply wave along rotated leaf-surface normal (Assets.cpp:118-122)
            if z_wave != 0.0:
                pt_x = pt[0] + z_wave * math.sin(rot_angle)
                pt_z = pt[2] + z_wave * math.cos(rot_angle)
                pt = torch.tensor([pt_x, pt[1], pt_z], dtype=torch.float32, device=device)

            row_verts.append(pt * scale)
        verts_grid.append(row_verts)

    verts_list = []
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            verts_list.append(verts_grid[j][i])
    verts_t = torch.stack(verts_list, dim=0)

    faces_list = []
    for j in range(Ny):
        for i in range(Nx):
            idx0 = j * (Nx + 1) + i
            idx1 = j * (Nx + 1) + (i + 1)
            idx2 = (j + 1) * (Nx + 1) + (i + 1)
            idx3 = (j + 1) * (Nx + 1) + i
            # Match Helios triangle winding (Assets.cpp:142-150)
            faces_list.append([idx0, idx1, idx2])
            faces_list.append([idx0, idx2, idx3])

    faces_t = torch.tensor(faces_list, dtype=torch.int64, device=device)
    return verts_t, faces_t


def generate_sorghum_leaf_mesh_torch(
    scale: torch.Tensor,
    aspect_ratio: float = 0.2,
    midrib_fold_fraction: float = 0.3,
    longitudinal_curvature: float = -0.3,
    lateral_curvature: float = -0.3,
    leaf_buckle_length: float = 0.5,
    leaf_buckle_angle_deg: float = 50.0,
    Nx: int = 30,
    Ny: int = 10,
    device=torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates Helios Sorghum GenericLeafPrototype parametric monocot curved leaf mesh."""
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=torch.float32, device=device)

    dx = 1.0 / float(Nx)
    dy = aspect_ratio / float(Ny)

    verts_grid = []
    for j in range(Ny + 1):
        row_verts = []
        dtheta = 0.0
        for i in range(Nx + 1):
            x = float(i) * dx
            y = float(j) * dy - 0.5 * aspect_ratio

            y_fold = math.cos(0.5 * midrib_fold_fraction * math.pi) * y
            z_fold = math.sin(0.5 * midrib_fold_fraction * math.pi) * abs(y)
            z_ycurve = lateral_curvature * ((y / aspect_ratio) ** 4)

            pt = torch.tensor([x, y_fold, z_fold + z_ycurve], dtype=torch.float32, device=device)

            if longitudinal_curvature != 0 and i > 0:
                dtheta -= math.atan(4.0 * longitudinal_curvature * (x**3) * dx)
                c, s = math.cos(dtheta), math.sin(dtheta)
                pt_x = pt[0] * c + pt[2] * s
                pt_z = -pt[0] * s + pt[2] * c
                pt = torch.tensor([pt_x, pt[1], pt_z], device=device)

            if leaf_buckle_angle_deg > 0:
                xf = leaf_buckle_length
                ang = 0.0
                if x <= xf and x + dx > xf:
                    ang = 0.5 * math.radians(leaf_buckle_angle_deg)
                elif x + dx > xf:
                    ang = math.radians(leaf_buckle_angle_deg)
                if ang > 0:
                    c, s = math.cos(ang), math.sin(ang)
                    dx_b = pt[0] - xf
                    pt_x = xf + dx_b * c + pt[2] * s
                    pt_z = -dx_b * s + pt[2] * c
                    pt = torch.tensor([pt_x, pt[1], pt_z], device=device)

            row_verts.append(pt * scale)
        verts_grid.append(row_verts)

    verts = torch.stack([pt for row in verts_grid for pt in row])

    faces = []
    for j in range(Ny):
        for i in range(Nx):
            v00 = j * (Nx + 1) + i
            v01 = j * (Nx + 1) + (i + 1)
            v10 = (j + 1) * (Nx + 1) + i
            v11 = (j + 1) * (Nx + 1) + (i + 1)
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
            faces.append([v00, v01, v10])
            faces.append([v01, v11, v10])

    faces_t = torch.tensor(faces, dtype=torch.int64, device=device)
    return verts, faces_t


TEXTURE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Digital-Crops", "libs", "Helios", "plugins", "plantarchitecture", "assets", "textures")
)


class HeliosAssetManager:
    """Loads and caches Helios OBJ assets and alpha-masked GenericLeaf meshes for PyTorch rendering."""
    def __init__(self, asset_dir: str = ASSET_DIR, texture_dir: str = TEXTURE_DIR):
        self.asset_dir = asset_dir
        self.texture_dir = texture_dir
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.generic_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._device_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_mesh(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if name not in self.cache:
            path = os.path.join(self.asset_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Helios asset missing: {path}")
            self.cache[name] = load_obj_file(path)
        v, f = self.cache[name]
        return v.clone(), f.clone()

    def get_mesh_device(self, name: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the OBJ mesh already on ``device``, cached to avoid a per-leaf
        host->device copy of the same prototype."""
        key = (name, str(device))
        cached = self._device_cache.get(key)
        if cached is not None:
            return cached
        v, f = self.get_mesh(name)
        v = v.to(device)
        f = f.to(device)
        self._device_cache[key] = (v, f)
        return v, f

    def get_generic_leaf_mesh(
        self,
        texture_name: str,
        Nx: int = 16,
        Ny: int = 16,
        aspect_ratio: float = 0.7,
        midrib_fold_fraction: float = 0.2,
        longitudinal_curvature: float = -0.2,
        lateral_curvature: float = -0.4,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds and caches Helios GenericLeafPrototype mesh with texture alpha-mask cutout."""
        cache_key = f"{texture_name}_{Nx}_{Ny}_{aspect_ratio}_{midrib_fold_fraction}_{longitudinal_curvature}_{lateral_curvature}"
        if cache_key not in self.generic_cache:
            dx = 1.0 / float(Nx)
            dy = aspect_ratio / float(Ny)

            verts_grid = []
            for j in range(Ny + 1):
                row_verts = []
                dtheta = 0.0
                for i in range(Nx + 1):
                    x = float(i) * dx
                    y = float(j) * dy - 0.5 * aspect_ratio

                    y_fold = math.cos(0.5 * midrib_fold_fraction * math.pi) * y
                    z_fold = math.sin(0.5 * midrib_fold_fraction * math.pi) * abs(y)
                    z_xcurve = longitudinal_curvature * (x ** 4)
                    z_ycurve = lateral_curvature * ((y / aspect_ratio) ** 4)

                    pt = [x, y_fold, z_fold + z_ycurve]
                    if longitudinal_curvature != 0.0 and i > 0:
                        dtheta -= math.atan(4.0 * longitudinal_curvature * (x ** 3) * dx)
                        c, s = math.cos(dtheta), math.sin(dtheta)
                        pt_x = pt[0] * c + pt[2] * s
                        pt_z = -pt[0] * s + pt[2] * c
                        pt = [pt_x, pt[1], pt_z]
                    row_verts.append(pt)
                verts_grid.append(row_verts)

            verts_list = [pt for row in verts_grid for pt in row]
            v_tensor = torch.tensor(verts_list, dtype=torch.float32)

            tex_path = os.path.join(self.texture_dir, texture_name)
            faces_list = []
            if os.path.exists(tex_path):
                from PIL import Image as PILImage
                img = PILImage.open(tex_path)
                alpha = np.array(img)[:, :, 3] / 255.0
                H_tex, W_tex = alpha.shape

                for j in range(Ny):
                    for i in range(Nx):
                        u_c = (i + 0.5) * dx
                        v_c = (j + 0.5) * dy / aspect_ratio
                        tx = max(0, min(W_tex - 1, int(u_c * (W_tex - 1))))
                        ty = max(0, min(H_tex - 1, int((1.0 - v_c) * (H_tex - 1))))

                        if alpha[ty, tx] > 0.35:
                            idx0 = j * (Nx + 1) + i
                            idx1 = j * (Nx + 1) + (i + 1)
                            idx2 = (j + 1) * (Nx + 1) + (i + 1)
                            idx3 = (j + 1) * (Nx + 1) + i
                            faces_list.append([idx0, idx1, idx2])
                            faces_list.append([idx0, idx2, idx3])
            else:
                for j in range(Ny):
                    for i in range(Nx):
                        idx0 = j * (Nx + 1) + i
                        idx1 = j * (Nx + 1) + (i + 1)
                        idx2 = (j + 1) * (Nx + 1) + (i + 1)
                        idx3 = (j + 1) * (Nx + 1) + i
                        faces_list.append([idx0, idx1, idx2])
                        faces_list.append([idx0, idx2, idx3])

            f_tensor = torch.tensor(faces_list, dtype=torch.int64)
            self.generic_cache[cache_key] = (v_tensor, f_tensor)

        v_cached, f_cached = self.generic_cache[cache_key]
        dev_key = (cache_key, str(device))
        dev_cached = self._device_cache.get(dev_key)
        if dev_cached is not None:
            return dev_cached
        v_dev = v_cached.to(device)
        f_dev = f_cached.to(device)
        self._device_cache[dev_key] = (v_dev, f_dev)
        return v_dev, f_dev


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

    if not isinstance(angle_rad, torch.Tensor):
        angle_rad = torch.tensor(angle_rad, dtype=vec.dtype, device=vec.device)

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


_TUBE_PROTO_CACHE: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _make_straight_tube_prototype(n_seg: int, n_rad: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a unit-length straight tube prototype aligned with +Z.

    Prototype vertices have x,y on a unit circle and z in [0,1]. Callers scale
    x,y by the desired radius and z by the desired length, then rotate +Z to
    the segment axis.

    The prototype is a pure function of (n_seg, n_rad) and is cached so the
    many tube meshes built per plant reuse the same vertex/face layout instead
    of rebuilding it from scratch each time.
    """
    cache_key = (n_seg, n_rad, str(device))
    cached = _TUBE_PROTO_CACHE.get(cache_key)
    if cached is not None:
        return cached

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
    _TUBE_PROTO_CACHE[cache_key] = (verts_t, faces_t)
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

    def __init__(
        self,
        asset_manager: Optional[HeliosAssetManager] = None,
        use_generic_leaves: bool = True,
        leaf_mode: str = "generic",
        leaf_scale_factor: float = 1.0,
        tube_radial_subdivisions: int = 4
    ):
        if asset_manager is None:
            asset_manager = HeliosAssetManager()
        self.asset_mgr = asset_manager
        self.leaf_mode = leaf_mode.lower() if leaf_mode is not None else ("generic" if use_generic_leaves else "obj")
        self.use_generic_leaves = (self.leaf_mode == "generic")
        self.leaf_scale_factor = leaf_scale_factor
        self.tube_radial_subdivisions = tube_radial_subdivisions

        self.COLOR_STEM = torch.tensor([0.22, 0.45, 0.15], dtype=torch.float32)
        self.COLOR_PETIOLE = torch.tensor([0.25, 0.50, 0.18], dtype=torch.float32)
        self.COLOR_LEAF = torch.tensor([0.25, 0.62, 0.18], dtype=torch.float32)
        self.COLOR_PEDUNCLE = torch.tensor([0.55, 0.52, 0.25], dtype=torch.float32)
        self.COLOR_FLOWER = torch.tensor([0.98, 0.85, 0.15], dtype=torch.float32)
        self.COLOR_POD = torch.tensor([0.85, 0.65, 0.13], dtype=torch.float32)

    def build_mesh_from_organ_array(
        self,
        organ_array: PlantOrganArray,
        device: torch.device = torch.device('cpu'),
        max_leaves: Optional[int] = None,
        existence_threshold: float = 0.5,
        species: Optional[str] = None,
        leaf_mode: Optional[str] = None,
        gravitropic_curvature: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Deprecated: Use build_mesh_from_part_tensor(organ_array.to_part_tensor()) instead.

        This wrapper runs the canonical XML -> 40D -> 14D Part Tensor pipeline and delegates to
        build_mesh_from_part_tensor. The legacy 94D forward-kinematics path has been
        removed from the rendering pipeline.
        """
        import warnings
        warnings.warn(
            "build_mesh_from_organ_array is deprecated. Use build_mesh_from_part_tensor(organ_array.to_part_tensor()).",
            DeprecationWarning,
            stacklevel=2,
        )
        pt = self.extract_part_tensor(organ_array, device=device, existence_threshold=existence_threshold)
        return self.build_mesh_from_part_tensor(
            pt, device=device, existence_threshold=existence_threshold, leaf_mode=leaf_mode
        )


    def extract_part_tensor(
        self,
        organ_array: PlantOrganArray,
        device: torch.device = torch.device('cpu'),
        existence_threshold: float = 0.5,
        gravitropic_curvature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Runs forward kinematics directly on the typed (N, 40) per-organ layout.
        Collects the world-space pose of every organ:
            [organ_type(1), base_xyz(3), rot6d(6), scale_xyz(3), curvature(1)]
        = 14 columns (Canonical 14D Part Tensor).
        """
        from diffusion_based.models.plant_organ_array import (
            T_COL_PLANT_ID, T_COL_PLANT_AGE, T_COL_BASE_X, T_COL_BASE_Y, T_COL_BASE_Z,
            T_COL_SHOOT_ID, T_COL_PARENT_SHOOT_ID, T_COL_PARENT_NODE_IDX, T_COL_PARENT_PETIOLE_IDX,
            T_COL_PHYTOMER_IDX, T_COL_CHILD_INDEX, T_COL_ORGAN_TYPE, T_COL_SHOOT_TYPE,
            T_COL_LENGTH, T_COL_RADIUS, T_COL_SCALE, T_COL_PITCH, T_COL_YAW, T_COL_ROLL,
            T_COL_CURVATURE, T_COL_PHYLLOTACTIC_ANGLE, T_COL_LENGTH_MAX, T_COL_LENGTH_SEGMENTS,
            T_COL_CURV_PERT_0, T_COL_CURV_PERT_1, T_COL_YAW_PERT_0, T_COL_YAW_PERT_1,
            T_COL_TAPER, T_COL_LEAFLET_OFFSET, T_COL_BUD_STATE, T_COL_CURRENT_LEAF_SCALE_FACTOR,
            T_COL_BUD_IS_TERMINAL, T_COL_FRUIT_SCALE, T_COL_FLOWER_AZIMUTH, T_COL_FLOWER_OFFSET,
            T_COL_RESERVED, T_COL_EXISTENCE, NUM_FEATURES_TYPED,
            ORGAN_NONE, ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE, ORGAN_PETIOLE, ORGAN_LEAF,
            ORGAN_PEDUNCLE, ORGAN_BUD_DORMANT, ORGAN_BUD_ACTIVE, ORGAN_FLOWER_CLOSED,
            ORGAN_FLOWER_OPEN, ORGAN_FRUIT, ORGAN_BUD_ABORTED,
            P_COL_ORGAN_TYPE, P_COL_BASE_X, P_COL_BASE_Y, P_COL_BASE_Z,
            P_COL_ROT_0, P_COL_SCALE_X, P_COL_SCALE_Y, P_COL_SCALE_Z,
            P_COL_CURVATURE, NUM_FEATURES_PART,
        )

        t = organ_array.tensor.to(device)
        existence = organ_array.existence.to(device).clamp(0.0, 1.0)
        N = t.shape[0]

        t_cpu = t.detach().cpu().numpy()
        shoot_id_arr = t_cpu[:, T_COL_SHOOT_ID].astype(int)
        phytomer_arr = t_cpu[:, T_COL_PHYTOMER_IDX].astype(int)
        organ_type_arr = t_cpu[:, T_COL_ORGAN_TYPE].astype(int)
        child_index_arr = t_cpu[:, T_COL_CHILD_INDEX].astype(int)
        parent_pet_arr = t_cpu[:, T_COL_PARENT_PETIOLE_IDX].astype(int)

        # Group typed rows: shoot meta, root meta, and per-phytomer organ rows.
        shoot_meta: Dict[int, int] = {}
        root_meta: Dict[int, int] = {}
        phytomers: Dict[Tuple[int, int], List[int]] = {}
        for i in range(N):
            ot = organ_type_arr[i]
            if ot == ORGAN_ROOT_META:
                root_meta[int(t_cpu[i, T_COL_PLANT_ID])] = i
            elif ot == ORGAN_SHOOT_META:
                shoot_meta[shoot_id_arr[i]] = i
            else:
                phytomers.setdefault((shoot_id_arr[i], phytomer_arr[i]), []).append(i)

        # Per-phytomer accessor dicts.
        phytomer_data: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for (sid, pidx), idxs in phytomers.items():
            d: Dict[str, Any] = {
                'internode': None, 'petioles': {}, 'leaves': [], 'bud': None,
                'peduncle': None, 'flowers': [],
            }
            for i in idxs:
                ot = organ_type_arr[i]
                if ot == ORGAN_INTERNODE:
                    d['internode'] = i
                elif ot == ORGAN_PETIOLE:
                    d['petioles'][parent_pet_arr[i]] = i
                elif ot == ORGAN_LEAF:
                    d['leaves'].append((parent_pet_arr[i], child_index_arr[i], i))
                elif ot in (ORGAN_BUD_DORMANT, ORGAN_BUD_ACTIVE, ORGAN_BUD_ABORTED):
                    d['bud'] = i
                elif ot == ORGAN_PEDUNCLE:
                    d['peduncle'] = i
                elif ot in (ORGAN_FLOWER_OPEN, ORGAN_FLOWER_CLOSED, ORGAN_FRUIT):
                    d['flowers'].append((child_index_arr[i], i))
            phytomer_data[(sid, pidx)] = d

        # Helper: pack a world-space organ into a 14D row
        def _make_row(organ_type_int: int, pos: torch.Tensor, forward: torch.Tensor,
                      up_hint: torch.Tensor, scale: torch.Tensor,
                      clamp_scale: bool = True, curvature: float = 0.0) -> torch.Tensor:
            """Build one part-tensor row from world-space pose."""
            fwd = forward / (torch.linalg.norm(forward) + 1e-8)
            up = up_hint - (up_hint * fwd).sum() * fwd
            up_norm = torch.linalg.norm(up)
            if up_norm < 1e-6:
                perp = torch.tensor([1.0, 0.0, 0.0], device=device)
                if abs(float(fwd[0].item())) > 0.9:
                    perp = torch.tensor([0.0, 1.0, 0.0], device=device)
                up = perp - (perp * fwd).sum() * fwd
                up = up / (torch.linalg.norm(up) + 1e-8)
            else:
                up = up / up_norm
            rot6d = torch.cat([up, fwd], dim=0)
            row = torch.zeros(NUM_FEATURES_PART, device=device)
            row[P_COL_ORGAN_TYPE] = float(organ_type_int)
            row[P_COL_BASE_X:P_COL_BASE_X + 3] = pos
            row[P_COL_ROT_0:P_COL_ROT_0 + 6] = rot6d
            row[P_COL_SCALE_X:P_COL_SCALE_X + 3] = scale.clamp(min=1e-6) if clamp_scale else scale
            if NUM_FEATURES_PART > 13:
                row[P_COL_CURVATURE] = float(curvature)
            return row

        def _make_row_rot(organ_type_int: int, pos: torch.Tensor, R: torch.Tensor,
                          scale: torch.Tensor, clamp_scale: bool = True, curvature: float = 0.0) -> torch.Tensor:
            """Build one part-tensor row from a full 3x3 rotation matrix R."""
            rot6d = torch.cat([R[:, 0], R[:, 1]], dim=0)
            row = torch.zeros(NUM_FEATURES_PART, device=device)
            row[P_COL_ORGAN_TYPE] = float(organ_type_int)
            row[P_COL_BASE_X:P_COL_BASE_X + 3] = pos
            row[P_COL_ROT_0:P_COL_ROT_0 + 6] = rot6d
            row[P_COL_SCALE_X:P_COL_SCALE_X + 3] = scale.clamp(min=1e-6) if clamp_scale else scale
            if NUM_FEATURES_PART > 13:
                row[P_COL_CURVATURE] = float(curvature)
            return row

        rows: list = []

        shoots_dict: Dict[int, List[int]] = {}
        for (sid, pidx) in phytomer_data:
            shoots_dict.setdefault(sid, []).append(pidx)
        for sid in shoots_dict:
            shoots_dict[sid].sort()
        sorted_shoot_ids = sorted(shoots_dict.keys())

        node_output_info: Dict = {}
        node_tip_positions = torch.zeros((N, 3), dtype=torch.float32, device=device)
        node_internode_axes = torch.zeros((N, 3), dtype=torch.float32, device=device)
        node_petiole_axes = torch.zeros((N, 2, 3), dtype=torch.float32, device=device)
        node_has_petiole = torch.zeros((N, 2), dtype=torch.float32, device=device)

        def compute_shoot_base(sid, first_pidx):
            sm_i = shoot_meta.get(sid)
            if sm_i is None:
                return (torch.zeros(3, device=device),
                        torch.tensor([0.0, 0.0, 1.0], device=device),
                        torch.tensor([0.0, -1.0, 0.0], device=device))
            parent_sid = int(t_cpu[sm_i, T_COL_PARENT_SHOOT_ID])
            parent_node_idx = int(t_cpu[sm_i, T_COL_PARENT_NODE_IDX])
            parent_pet_idx = int(t_cpu[sm_i, T_COL_PARENT_PETIOLE_IDX])
            if parent_sid < 0 or (parent_sid, parent_node_idx) not in node_output_info:
                return (torch.zeros(3, device=device),
                        torch.tensor([0.0, 0.0, 1.0], device=device),
                        torch.tensor([0.0, -1.0, 0.0], device=device))
            p_info = node_output_info[(parent_sid, parent_node_idx)]
            parent_axis = p_info['internode_axis']
            pet_axes = p_info.get('petiole_axes', {})
            if parent_pet_idx in pet_axes:
                parent_pet_axis = pet_axes[parent_pet_idx]
                axis_vec = parent_pet_axis
            else:
                ghost = torch.linalg.cross(parent_axis, torch.tensor([0.0, 0.0, 1.0], device=device))
                if torch.linalg.norm(ghost) < 0.01:
                    ghost = torch.tensor([0.0, 1.0, 0.0], device=device)
                ghost = ghost / torch.linalg.norm(ghost)
                phyllo_ang = p_info.get('phyllo_angle', torch.tensor(0.0, device=device))
                cum_rot = float(parent_node_idx) * phyllo_ang
                parent_pet_axis = rotate_vector_about_axis(ghost, parent_axis, cum_rot)
                axis_vec = parent_axis
            p_radius = p_info.get('radius', torch.tensor(0.0, device=device))
            base_pos = p_info['tip'] + 0.9 * p_radius * axis_vec
            return base_pos, parent_axis, parent_pet_axis

        deg2rad = torch.tensor(math.pi / 180.0, dtype=torch.float32, device=device)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)

        # 1. Emit plant root metadata row
        plant_base_pos = torch.zeros(3, device=device)
        plant_age = 0.0
        if 0 in root_meta:
            rm_i = root_meta[0]
            plant_base_pos = t[rm_i, T_COL_BASE_X:T_COL_BASE_Z + 1]
            plant_age = float(t[rm_i, T_COL_PLANT_AGE].item())
        rows.append(_make_row(
            ORGAN_ROOT_META, plant_base_pos, z_axis,
            torch.tensor([0.0, 1.0, 0.0], device=device),
            scale=torch.tensor([plant_age, 0.0, 0.0], device=device),
        ))

        # 2. Forward kinematics walk over shoots
        for sid in sorted_shoot_ids:
            pidx_list = shoots_dict[sid]
            if not pidx_list:
                continue

            sm_i = shoot_meta.get(sid)
            if sm_i is None:
                base_pitch_rad = torch.tensor(0.0, device=device)
                base_yaw_rad = torch.tensor(0.0, device=device)
                base_roll_rad = torch.tensor(0.0, device=device)
            else:
                base_pitch_rad = t[sm_i, T_COL_PITCH] * deg2rad
                base_yaw_rad = t[sm_i, T_COL_YAW] * deg2rad
                base_roll_rad = t[sm_i, T_COL_ROLL] * deg2rad

            shoot_base_pos, parent_internode_axis, parent_petiole_axis = compute_shoot_base(sid, pidx_list[0])
            rows.append(_make_row(
                ORGAN_SHOOT_META, shoot_base_pos, parent_internode_axis,
                up_hint=parent_petiole_axis,
                scale=torch.tensor([0.0, 0.0, 0.0], device=device),
            ))
            curr_pos = shoot_base_pos.clone()
            prev_internode_axis = parent_internode_axis
            prev_petiole_axis = parent_petiole_axis

            if gravitropic_curvature is not None:
                eff_grav = gravitropic_curvature
            elif sm_i is not None and float(t_cpu[sm_i, T_COL_RESERVED]) != 0.0:
                eff_grav = float(t_cpu[sm_i, T_COL_RESERVED])
            elif sid == 0:
                eff_grav = 0.0
            else:
                eff_grav = 200.0

            for p_idx_in_shoot, p_idx in enumerate(pidx_list):
                pdata = phytomer_data[(sid, p_idx)]
                inode_i = pdata['internode']
                node_exist = existence[inode_i] if inode_i is not None else torch.tensor(0.0, device=device)

                petiole_rot_axis = torch.linalg.cross(prev_internode_axis, prev_petiole_axis)
                if torch.linalg.norm(petiole_rot_axis) < 1e-6:
                    petiole_rot_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
                else:
                    petiole_rot_axis = petiole_rot_axis / torch.linalg.norm(petiole_rot_axis)

                if inode_i is None:
                    inode_pitch_rad = torch.tensor(0.0, device=device)
                    inode_phyllo_rad = torch.tensor(0.0, device=device)
                else:
                    inode_pitch_rad = t[inode_i, T_COL_PITCH] * deg2rad
                    inode_phyllo_rad = t[inode_i, T_COL_PHYLLOTACTIC_ANGLE] * deg2rad

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

                # --- Internode FK ---
                if inode_i is None:
                    inode_len = torch.tensor(0.0, device=device)
                    inode_rad = torch.tensor(0.0, device=device)
                    seg_cnt = 1
                    seg_len = torch.tensor(0.0, device=device)
                    seg_len_max = torch.tensor(0.0, device=device)
                    curv_p0 = torch.tensor(0.0, device=device)
                    curv_p1 = torch.tensor(0.0, device=device)
                    yaw_p0 = torch.tensor(0.0, device=device)
                    yaw_p1 = torch.tensor(0.0, device=device)
                else:
                    inode_len = torch.clamp(t[inode_i, T_COL_LENGTH], min=1e-4) * node_exist
                    inode_rad = torch.clamp(t[inode_i, T_COL_RADIUS], min=1e-4) * node_exist
                    seg_cnt = max(1, int(t_cpu[inode_i, T_COL_LENGTH_SEGMENTS]))
                    seg_len = inode_len / seg_cnt
                    seg_len_max = torch.clamp(t[inode_i, T_COL_LENGTH_MAX], min=1e-4) / seg_cnt
                    curv_p0, curv_p1 = t[inode_i, T_COL_CURV_PERT_0], t[inode_i, T_COL_CURV_PERT_1]
                    yaw_p0, yaw_p1 = t[inode_i, T_COL_YAW_PERT_0], t[inode_i, T_COL_YAW_PERT_1]

                inode_base = curr_pos.clone()
                step_p = curr_pos.clone()
                step_dir = i_axis.clone()
                for s in range(seg_cnt):
                    if p_idx_in_shoot > 0:
                        curv_pert = curv_p0 if s == 0 else curv_p1
                        yaw_pert = yaw_p0 if s == 0 else yaw_p1
                        curv_fact = 0.5 - step_dir[2] / 2.0
                        if step_dir[2] < 0:
                            curv_fact = curv_fact * 2.0
                        curvature_angle = deg2rad * (eff_grav * curv_fact * seg_len_max + curv_pert)
                        if curvature_angle != 0.0:
                            step_dir = rotate_vector_about_axis(step_dir, shoot_bending_axis, curvature_angle)
                        if yaw_pert != 0.0:
                            step_dir = rotate_vector_about_axis(step_dir, z_axis, deg2rad * yaw_pert)
                    step_p = step_p + step_dir * seg_len

                curr_pos = step_p
                inode_tip_axis = step_dir / (torch.linalg.norm(step_dir) + 1e-6)

                # Emit internode row: scale = (length, radius, 0.0)
                if float(node_exist.item()) > existence_threshold and inode_i is not None:
                    rows.append(_make_row(
                        ORGAN_INTERNODE, inode_base, inode_tip_axis,
                        up_hint=petiole_rot_axis,
                        scale=torch.stack([inode_len, inode_rad, torch.tensor(0.0, device=device)]),
                        curvature=float(curv_p0.item()) if inode_i is not None else 0.0,
                    ))
                else:
                    rows.append(torch.zeros(NUM_FEATURES_PART, device=device))

                if inode_i is not None:
                    node_tip_positions[inode_i] = curr_pos
                    node_internode_axes[inode_i] = inode_tip_axis

                node_info = {
                    'tip': curr_pos,
                    'internode_axis': inode_tip_axis,
                    'radius': inode_rad,
                    'phyllo_angle': inode_phyllo_rad,
                }
                node_output_info[(sid, p_idx)] = node_info
                pet_axes_stored: Dict[int, torch.Tensor] = {}

                # --- Petiole + Leaf helper ---
                def process_petiole_extract(pet_i, pdata):
                    pet_row_i = pdata['petioles'].get(pet_i)
                    if pet_row_i is None:
                        return None, None, None
                    p_len_raw = t[pet_row_i, T_COL_LENGTH]
                    p_rad_raw = t[pet_row_i, T_COL_RADIUS]
                    p_pitch_deg = t[pet_row_i, T_COL_PITCH]
                    p_curv_deg = t[pet_row_i, T_COL_CURVATURE]
                    p_seg_cnt = max(1, int(t_cpu[pet_row_i, T_COL_LENGTH_SEGMENTS]))
                    lflt_offset = t[pet_row_i, T_COL_LEAFLET_OFFSET]
                    leaves = [lf for lf in pdata['leaves'] if lf[0] == pet_i]
                    leaves.sort(key=lambda lf: lf[1])
                    num_leaves = len(leaves)

                    pet_pitch_rad = p_pitch_deg * deg2rad
                    pet_axis = rotate_vector_about_axis(i_axis, petiole_rot_axis, torch.abs(pet_pitch_rad))
                    pet_rot_ax = petiole_rot_axis.clone()
                    if p_idx_in_shoot != 0 and inode_phyllo_rad != 0.0:
                        pet_axis = rotate_vector_about_axis(pet_axis, i_axis, inode_phyllo_rad)
                        pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, inode_phyllo_rad)
                    if pet_i > 0:
                        petioles_per_internode = 2.0 if 1 in pdata['petioles'] else 1.0
                        budrot = torch.tensor(pet_i * 2.0 * math.pi / petioles_per_internode, device=device)
                        pet_axis = rotate_vector_about_axis(pet_axis, i_axis, budrot)
                        pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, budrot)
                    pet_axis = pet_axis / (torch.linalg.norm(pet_axis) + 1e-12)
                    pet_axes_stored[pet_i] = pet_axis.clone()

                    p_len = p_len_raw * node_exist
                    p_rad = p_rad_raw * node_exist
                    if float(p_len.item()) <= 0 or float(p_rad.item()) <= 0:
                        return None, None, None

                    pet_rot_ax_norm = pet_rot_ax / (torch.linalg.norm(pet_rot_ax) + 1e-8)
                    pet_base = curr_pos.clone()
                    seq_len = p_len / p_seg_cnt

                    pet_line_list = [pet_base]
                    cur_pet_p = pet_base
                    cur_pet_dir = pet_axis.clone()
                    for _ps in range(p_seg_cnt):
                        curv_per_seg = p_curv_deg * seq_len * deg2rad
                        if torch.abs(curv_per_seg) > 1e-12:
                            cur_pet_dir = rotate_vector_about_axis(cur_pet_dir, pet_rot_ax_norm, -curv_per_seg)
                        cur_pet_p = cur_pet_p + cur_pet_dir * seq_len
                        pet_line_list.append(cur_pet_p)

                    pet_line = torch.stack(pet_line_list)
                    pet_tip = pet_line[-1]
                    pet_tip_axis = pet_line[-1] - pet_line[-2]
                    pet_tip_axis = pet_tip_axis / (torch.linalg.norm(pet_tip_axis) + 1e-8)

                    # Emit petiole row: scale = (length, radius, 0.0)
                    if float(node_exist.item()) > existence_threshold:
                        rows.append(_make_row(
                            ORGAN_PETIOLE, pet_base, pet_axis,
                            up_hint=pet_rot_ax_norm,
                            scale=torch.stack([p_len, p_rad, torch.tensor(0.0, device=device)]),
                            curvature=float(p_curv_deg.item()) if torch.is_tensor(p_curv_deg) else float(p_curv_deg),
                        ))
                    else:
                        rows.append(torch.zeros(NUM_FEATURES_PART, device=device))

                    # Emit leaf rows: scale = (l_scale, 0.0, 0.0)
                    for lf_i, (lf_pet, lf_idx, lf_row_i) in enumerate(leaves):
                        l_scale = t[lf_row_i, T_COL_SCALE] * node_exist
                        l_pitch_raw = t[lf_row_i, T_COL_PITCH] * deg2rad
                        l_yaw = t[lf_row_i, T_COL_YAW] * deg2rad
                        l_roll_raw = t[lf_row_i, T_COL_ROLL] * deg2rad

                        ind_from_tip = float(lf_i) - float(num_leaves - 1) / 2.0
                        if num_leaves > 1:
                            if lf_i == (num_leaves - 1) / 2.0:
                                compound_rotation = 0.0
                            elif lf_i < (num_leaves - 1) / 2.0:
                                compound_rotation = -0.5 * math.pi
                            else:
                                compound_rotation = 0.5 * math.pi
                        else:
                            compound_rotation = 0.0

                        asin_pz = torch.asin(torch.clamp(pet_tip_axis[2], -1.0, 1.0))
                        if num_leaves == 1:
                            roll_rot = torch.acos(torch.clamp(inode_tip_axis[2], -1.0, 1.0)) - l_roll_raw
                        elif ind_from_tip != 0:
                            sign_roll = compound_rotation / abs(compound_rotation)
                            roll_rot = (asin_pz + l_roll_raw) * sign_roll
                        else:
                            roll_rot = torch.tensor(0.0, device=device)
                        pitch_rot = l_pitch_raw
                        if ind_from_tip == 0:
                            pitch_rot = pitch_rot + asin_pz
                        yaw_rot = l_yaw if ind_from_tip != 0 else torch.tensor(0.0, device=device)
                        azimuth_rot = -torch.atan2(pet_tip_axis[1], pet_tip_axis[0] + 1e-8) + compound_rotation

                        R_leaf = (
                            rotr_z(azimuth_rot, device) @
                            rotr_z(yaw_rot, device) @
                            rotr_y(-pitch_rot, device) @
                            rotr_x(roll_rot, device)
                        )

                        leaf_base = pet_tip
                        if num_leaves > 1 and float(lflt_offset.item()) > 0.0 and ind_from_tip != 0:
                            offset = (abs(ind_from_tip) - 0.5) * float(lflt_offset.item()) * float(p_len.item())
                            frac = 1.0 - offset / max(float(p_len.item()), 1e-6)
                            frac = max(0.0, min(1.0, frac))
                            idx_f = frac * (len(pet_line_list) - 1)
                            idx_0 = int(math.floor(idx_f))
                            idx_1 = min(idx_0 + 1, len(pet_line_list) - 1)
                            t_interp = idx_f - idx_0
                            leaf_base = pet_line_list[idx_0] * (1.0 - t_interp) + pet_line_list[idx_1] * t_interp

                        if float(node_exist.item()) > existence_threshold:
                            rows.append(_make_row_rot(
                                ORGAN_LEAF, leaf_base, R_leaf,
                                scale=torch.stack([l_scale, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)]),
                            ))
                        else:
                            rows.append(torch.zeros(NUM_FEATURES_PART, device=device))

                    return pet_base, pet_tip, pet_tip_axis

                for pet_i in sorted(pdata['petioles'].keys()):
                    process_petiole_extract(pet_i, pdata)

                node_info['petiole_axes'] = pet_axes_stored
                if inode_i is not None:
                    if 0 in pet_axes_stored:
                        node_petiole_axes[inode_i, 0] = pet_axes_stored[0]
                        node_has_petiole[inode_i, 0] = 1.0
                    if 1 in pet_axes_stored:
                        node_petiole_axes[inode_i, 1] = pet_axes_stored[1]
                        node_has_petiole[inode_i, 1] = 1.0

                # --- Peduncle / Flowers / Pods ---
                bud_i = pdata['bud']
                if bud_i is not None:
                    bud_state = int(t_cpu[bud_i, T_COL_BUD_STATE])
                    fruit_scale = float(t_cpu[bud_i, T_COL_FRUIT_SCALE])
                    is_terminal_bud = bool(float(t[bud_i, T_COL_BUD_IS_TERMINAL].item()) > 0.5)
                    fl_offset_v = float(t_cpu[bud_i, T_COL_FLOWER_OFFSET])
                else:
                    bud_state = 5
                    fruit_scale = 1.0
                    is_terminal_bud = False
                    fl_offset_v = 0.0

                if bud_i is not None:
                    bud_organ = (ORGAN_BUD_DORMANT if bud_state == 0 else
                                 (ORGAN_BUD_ACTIVE if bud_state in (1, 2, 3, 4) else ORGAN_BUD_ABORTED))
                    if float(node_exist.item()) > existence_threshold:
                        rows.append(_make_row(
                            bud_organ, curr_pos, inode_tip_axis,
                            up_hint=petiole_rot_axis,
                            scale=torch.tensor([fruit_scale, 0.0, 0.0], device=device),
                        ))
                    else:
                        rows.append(torch.zeros(NUM_FEATURES_PART, device=device))

                flowers = sorted(pdata['flowers'], key=lambda f: f[0])
                num_flowers = len(flowers)
                is_active_flower = bud_state in [2, 3, 4]
                ped_i = pdata['peduncle']
                has_ped = ped_i is not None
                is_active = is_active_flower and float(node_exist.item()) > existence_threshold

                if has_ped:
                    ped_base = curr_pos.clone()
                    ped_tip = ped_base.clone()
                    ped_axis_final = inode_tip_axis.clone()
                    ped_verts_list = [ped_base.clone()]
                    ped_axis_initial = inode_tip_axis.clone()
                    infl_bend_axis = torch.tensor([1.0, 0.0, 0.0], device=device)

                    ped_len = t[ped_i, T_COL_LENGTH]
                    ped_rad = t[ped_i, T_COL_RADIUS]

                    if float(ped_len.item()) > 1e-4 and float(ped_rad.item()) > 1e-5:
                        curr_pet_axis = pet_axes_stored[0] if 0 in pet_axes_stored else prev_petiole_axis
                        infl_bend_axis = torch.linalg.cross(prev_internode_axis, curr_pet_axis)
                        if torch.linalg.norm(infl_bend_axis) < 1e-4:
                            infl_bend_axis = torch.linalg.cross(inode_tip_axis, z_axis)
                            if torch.linalg.norm(infl_bend_axis) < 1e-4:
                                infl_bend_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
                        infl_bend_axis = infl_bend_axis / torch.linalg.norm(infl_bend_axis)

                        import math as _math
                        ped_pitch_rad = _math.radians(float(t_cpu[ped_i, T_COL_PITCH]))
                        ped_axis = rotate_vector_about_axis(inode_tip_axis, infl_bend_axis, ped_pitch_rad)
                        parent_pet_az = -_math.atan2(float(curr_pet_axis[1].item()), float(curr_pet_axis[0].item()))
                        cur_ped_az = -_math.atan2(float(ped_axis[1].item()), float(ped_axis[0].item()))
                        az_rot = cur_ped_az - parent_pet_az
                        ped_axis = rotate_vector_about_axis(ped_axis, inode_tip_axis, az_rot)
                        ped_axis = ped_axis / (torch.linalg.norm(ped_axis) + 1e-6)

                        ped_axis_initial = ped_axis.clone()
                        Ndiv_ped = 6
                        dr_ped = (ped_len * node_exist) / float(Ndiv_ped)
                        curv_val = float(t_cpu[ped_i, T_COL_CURVATURE])
                        step_p_ped = ped_base.clone()
                        ped_verts_list = [step_p_ped.clone()]
                        for _seg in range(Ndiv_ped):
                            if abs(curv_val) > 0.0:
                                h_bend = torch.linalg.cross(ped_axis, z_axis)
                                h_norm = torch.linalg.norm(h_bend)
                                if h_norm > 1e-4:
                                    h_bend = h_bend / h_norm
                                    theta_curv = _math.radians(curv_val * float(dr_ped.item()))
                                    if curv_val > 0.0:
                                        theta_from_target = _math.acos(min(1.0, max(-1.0, float(ped_axis[2].item()))))
                                        target_axis = z_axis
                                    else:
                                        theta_from_target = _math.acos(min(1.0, max(-1.0, float(-ped_axis[2].item()))))
                                        target_axis = -z_axis
                                    if abs(theta_curv) >= theta_from_target:
                                        ped_axis = target_axis.clone()
                                    else:
                                        ped_axis = rotate_vector_about_axis(ped_axis, h_bend, theta_curv)
                                        ped_axis = ped_axis / (torch.linalg.norm(ped_axis) + 1e-6)
                                else:
                                    ped_axis = (z_axis if curv_val > 0.0 else -z_axis).clone()
                            step_p_ped = step_p_ped + dr_ped * ped_axis
                            ped_verts_list.append(step_p_ped)

                        ped_tip = ped_verts_list[-1].clone()
                        ped_axis_final = ped_axis.clone()

                    if is_active:
                        rows.append(_make_row(
                            ORGAN_PEDUNCLE, ped_base, ped_axis_initial,
                            up_hint=infl_bend_axis,
                            scale=torch.stack([ped_len * node_exist, ped_rad * node_exist, torch.tensor(0.0, device=device)]),
                            curvature=curv_val,
                        ))
                    else:
                        rows.append(torch.zeros(NUM_FEATURES_PART, device=device))

                if is_active:
                    if not has_ped:
                        ped_base = curr_pos.clone()
                        ped_tip = ped_base.clone()
                        ped_axis_final = inode_tip_axis.clone()
                        ped_verts_list = [ped_base.clone()]
                    is_pod = (bud_state == 4)
                    is_closed_flower = (bud_state == 2)
                    fl_count = min(max(num_flowers, 1 if is_active_flower else 0), 4)
                    fl_offset_clamped = min(fl_offset_v, 1.0 / (0.5 * fl_count - 1.0) if fl_count > 2 else fl_offset_v)

                    for fl_i in range(fl_count):
                        fl_pitch_deg = 0.0
                        fl_yaw_deg = 0.0
                        fl_roll_deg = 0.0
                        fl_az_deg = 0.0
                        if fl_i < len(flowers):
                            fl_row_i = flowers[fl_i][1]
                            fl_scale = float(t[fl_row_i, T_COL_SCALE].item())
                            fl_pitch_deg = float(t_cpu[fl_row_i, T_COL_PITCH])
                            fl_yaw_deg = float(t_cpu[fl_row_i, T_COL_YAW])
                            fl_roll_deg = float(t_cpu[fl_row_i, T_COL_ROLL])
                            fl_az_deg = float(t_cpu[fl_row_i, T_COL_FLOWER_AZIMUTH])
                        else:
                            fl_scale = 0.09 if is_pod else 0.03
                        if fl_scale <= 1e-4:
                            fl_scale = 0.09 if is_pod else 0.03

                        ind_from_tip = abs(fl_i - (fl_count - 1.0))
                        if fl_count > 1 and fl_offset_clamped > 0.0 and ind_from_tip > 1e-4 and len(ped_verts_list) > 1:
                            frac = 1.0 - (ind_from_tip - 0.5) * fl_offset_clamped
                            frac = max(0.0, min(1.0, frac))
                            idx_f = frac * (len(ped_verts_list) - 1)
                            idx_0 = int(math.floor(idx_f))
                            idx_1 = min(idx_0 + 1, len(ped_verts_list) - 1)
                            t_interp = idx_f - idx_0
                            cur_fl_base = ped_verts_list[idx_0] * (1.0 - t_interp) + ped_verts_list[idx_1] * t_interp
                        else:
                            cur_fl_base = ped_tip

                        organ_type_int = ORGAN_FRUIT if is_pod else (ORGAN_FLOWER_CLOSED if is_closed_flower else ORGAN_FLOWER_OPEN)

                        pitch_r = math.radians(fl_pitch_deg)
                        yaw_r = math.radians(fl_yaw_deg)
                        roll_r = math.radians(fl_roll_deg)
                        az_r = math.radians(fl_az_deg)

                        R_fl = rotr_z(az_r, device) @ rotr_y(pitch_r, device) @ rotr_x(roll_r, device)
                        if abs(yaw_r) > 1e-4:
                            R_yaw = rodrigues_matrix_torch(ped_axis_final, torch.tensor(yaw_r, device=device), device=device)
                            R_fl = R_yaw @ R_fl

                        if is_pod:
                            tot_fl_scale = fl_scale * (fruit_scale if fruit_scale > 0.0 else 1.0) * float(node_exist.item())
                        else:
                            tot_fl_scale = fl_scale * float(node_exist.item())

                        rows.append(_make_row_rot(
                            organ_type_int, cur_fl_base, R_fl,
                            scale=torch.tensor([tot_fl_scale, 0.0, 0.0], device=device),
                            clamp_scale=False,
                        ))

                # Update parent context
                prev_internode_axis = inode_tip_axis
                if 0 in pet_axes_stored:
                    prev_petiole_axis = pet_axes_stored[0]
                else:
                    ghost = torch.linalg.cross(inode_tip_axis, z_axis)
                    if torch.linalg.norm(ghost) < 0.01:
                        ghost = torch.tensor([0.0, 1.0, 0.0], device=device)
                    ghost = ghost / torch.linalg.norm(ghost)
                    cum_rot = float(p_idx_in_shoot) * float(inode_phyllo_rad.item())
                    prev_petiole_axis = rotate_vector_about_axis(ghost, inode_tip_axis, cum_rot)

        if not rows:
            return torch.zeros((0, NUM_FEATURES_PART), dtype=torch.float32, device=device)
        return torch.stack(rows, dim=0)

    def build_mesh_from_part_tensor(
        self,
        part_tensor: torch.Tensor,
        existence: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu'),
        leaf_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Directly builds 3D meshes from a Canonical 14D Part Tensor on GPU.
        Zero XML serialization overhead.

        Canonical 14D Part Tensor Layout:
          - col 0:     organ_type (ORGAN_NONE=0, ORGAN_ROOT_META=1, ORGAN_SHOOT_META=2,
                       ORGAN_INTERNODE=3, ORGAN_PETIOLE=4, ORGAN_LEAF=5,
                       ORGAN_PEDUNCLE=6, ORGAN_BUD_DORMANT=7, ORGAN_BUD_ACTIVE=8,
                       ORGAN_FLOWER_CLOSED=9, ORGAN_FLOWER_OPEN=10, ORGAN_FRUIT=11,
                       ORGAN_BUD_ABORTED=12)
          - cols 1-3:  base world position (metres)
          - cols 4-9:  rot6d = [R[:,0], R[:,1]] packed (Gram-Schmidt -> full R)
          - cols 10-12: scale (length, radius, 0) for tubes; (s, 0, 0) for meshes
          - col 13:    curvature (metres^-1 or deg)

        existence:
          - Optional continuous existence weights in [0, 1] per organ row (N,).
          - XML Ground Truth: defaults to 1.0 (fully opaque, crisp solid rendering).
          - Diffusion / Optimization: can be in (0, 1), rendering the organ softly/translucent
            and allowing dense photometric image loss gradients to directly flow back to existence!
        """
        from diffusion_based.models.plant_organ_array import (
            ORGAN_NONE, ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE,
            ORGAN_PETIOLE, ORGAN_LEAF, ORGAN_PEDUNCLE, ORGAN_BUD_DORMANT,
            ORGAN_BUD_ACTIVE, ORGAN_FLOWER_CLOSED, ORGAN_FLOWER_OPEN,
            ORGAN_FRUIT, ORGAN_BUD_ABORTED, NUM_FEATURES_PART,
        )

        eff_leaf_mode = leaf_mode.lower() if leaf_mode is not None else self.leaf_mode

        p = part_tensor.to(device)
        if p.ndim == 1:
            p = p.unsqueeze(0)
        N, D = p.shape

        if D != NUM_FEATURES_PART:
            raise ValueError(
                f"build_mesh_from_part_tensor strictly expects Canonical 14D Part Tensor (N, 14), got (N, {D}). "
                f"If using a 26D continuous/differentiable tensor, call diff_node_to_part_tensor_14d(diff_tensor) "
                f"before passing to build_mesh_from_part_tensor."
            )

        if existence is None:
            exist = torch.ones((N,), device=device, dtype=torch.float32)
        else:
            exist = existence.to(device).float().clamp(0.0, 1.0)
            if exist.ndim > 1:
                exist = exist.squeeze(-1)

        ot = p[:, 0].long()
        base_pos = p[:, 1:4]
        rot_6d = p[:, 4:10]
        scale = p[:, 10:13]
        curvature = p[:, 13]
        active_mask = (ot != ORGAN_NONE) & (exist > 1e-4) & (scale[:, 0] > 1e-5)

        z_axis_build = torch.tensor([0.0, 0.0, 1.0], device=device)

        # Gram-Schmidt 6D to 3x3 rotation matrices
        u = rot_6d[:, :3]
        v = rot_6d[:, 3:6]
        u_norm = torch.linalg.norm(u, dim=-1, keepdim=True).clamp(min=1e-6)
        r1 = u / u_norm
        dot = (r1 * v).sum(dim=-1, keepdim=True)
        v_ortho = v - dot * r1
        v_norm = torch.linalg.norm(v_ortho, dim=-1, keepdim=True).clamp(min=1e-6)
        r2 = v_ortho / v_norm
        r3 = torch.linalg.cross(r1, r2)
        R_mats = torch.stack([r1, r2, r3], dim=-1)

        all_verts = []
        all_faces = []
        all_normals = []
        all_colors = []
        all_opacities = []
        all_organs = []
        vert_offset = 0

        # Pre-cache leaf meshes indexed by variant (0=unifoliate, 1=left, 2=tip, 3=right)
        _leaf_obj_map = {
            0: ("CowpeaLeaf_unifoliate.obj",      "CowpeaLeaf_unifoliate_centered.png"),
            1: ("CowpeaLeaf_left_highres.obj",     "CowpeaLeaf_left_centered.png"),
            2: ("CowpeaLeaf_tip_highres.obj",      "CowpeaLeaf_tip_centered.png"),
            3: ("CowpeaLeaf_right_highres.obj",    "CowpeaLeaf_right_centered.png"),
        }
        _leaf_cache: Dict[int, tuple] = {}

        def _get_leaf_mesh(variant: int):
            if variant in _leaf_cache:
                return _leaf_cache[variant]
            obj_name, tex_name = _leaf_obj_map.get(variant, _leaf_obj_map[2])
            if eff_leaf_mode == "generic":
                if variant == 0:
                    tex_name_use = "CowpeaLeaf_unifoliate_centered.png"
                elif variant == 1:
                    tex_name_use = "CowpeaLeaf_left_centered.png"
                elif variant == 2:
                    tex_name_use = "CowpeaLeaf_tip_centered.png"
                else:
                    tex_name_use = "CowpeaLeaf_right_centered.png"
                v_lf, f_lf = self.asset_mgr.get_generic_leaf_mesh(
                    texture_name=tex_name_use,
                    Nx=16, Ny=16,
                    aspect_ratio=0.7,
                    midrib_fold_fraction=0.2,
                    longitudinal_curvature=-0.2,
                    lateral_curvature=-0.4,
                    device=device,
                )
            else:
                v_lf, f_lf = self.asset_mgr.get_mesh_device(obj_name, device)
            _leaf_cache[variant] = (v_lf, f_lf)
            return v_lf, f_lf

        n_rad = max(3, self.tube_radial_subdivisions)

        ot_a = ot[active_mask]
        exist_a = exist[active_mask]
        base_pos_a = base_pos[active_mask]
        R_a = R_mats[active_mask]
        scale_a = scale[active_mask].clamp(min=1e-6)

        # -------------------------------------------------------------
        # 1. BATCH LEAVES (ORGAN_LEAF=5)
        # -------------------------------------------------------------
        leaf_mask = (ot_a == ORGAN_LEAF)
        if leaf_mask.any():
            v_proto, f_proto = _get_leaf_mesh(2)
            n_proto = compute_face_normals_torch(v_proto, f_proto)

            M = int(leaf_mask.sum().item())
            R_l = R_a[leaf_mask]
            pos_l = base_pos_a[leaf_mask]
            s_l = scale_a[leaf_mask][:, 0:1]  # uniform leaf scale
            exist_l = exist_a[leaf_mask]

            v_scaled = v_proto.unsqueeze(0) * s_l.unsqueeze(1)
            v_rot = torch.bmm(v_scaled, R_l.transpose(1, 2)) + pos_l.unsqueeze(1)
            n_rot = torch.bmm(n_proto.unsqueeze(0).expand(M, -1, -1), R_l.transpose(1, 2))

            V = v_proto.shape[0]
            f_offsets = (torch.arange(M, device=device) * V).unsqueeze(1).unsqueeze(2)
            f_batch = (f_proto.unsqueeze(0) + f_offsets).reshape(-1, 3) + vert_offset

            v_flat = v_rot.reshape(-1, 3)
            n_flat = n_rot.reshape(-1, 3)
            c_flat = self.COLOR_LEAF.to(device).unsqueeze(0).expand(v_flat.shape[0], -1)
            op_flat = exist_l.unsqueeze(1).expand(M, V).reshape(-1, 1)
            o_flat = torch.full((v_flat.shape[0],), 2, dtype=torch.int64, device=device)

            all_verts.append(v_flat)
            all_faces.append(f_batch)
            all_normals.append(n_flat)
            all_colors.append(c_flat)
            all_opacities.append(op_flat)
            all_organs.append(o_flat)
            vert_offset += v_flat.shape[0]

        # -------------------------------------------------------------
        # 2. BATCH STRAIGHT TUBES (ORGAN_INTERNODE=3, ORGAN_PETIOLE=4)
        # -------------------------------------------------------------
        tube_mask = (ot_a == ORGAN_INTERNODE) | (ot_a == ORGAN_PETIOLE)
        is_inode = (ot_a[tube_mask] == ORGAN_INTERNODE)
        ped_mask = (ot_a == ORGAN_PEDUNCLE)

        if tube_mask.any():
            _, proto_f = _make_straight_tube_prototype(1, n_rad, device)

            M = int(tube_mask.sum().item())
            R_t = R_a[tube_mask]
            pos_t = base_pos_a[tube_mask]
            s_t = scale_a[tube_mask]
            exist_t = exist_a[tube_mask]

            lengths = s_t[:, 0].clamp(min=1e-3)
            radii_0 = s_t[:, 1].clamp(min=1e-4)
            radii_1 = torch.where(is_inode, radii_0 * 0.85, radii_0 * 0.8)

            fwd = R_t[:, :, 1]   # (M, 3) forward axis
            e0 = R_t[:, :, 0]    # (M, 3) radial basis 0
            e1 = R_t[:, :, 2]    # (M, 3) radial basis 1

            angles = torch.linspace(0.0, 2.0 * math.pi, n_rad + 1, device=device)[:-1]
            ca = torch.cos(angles)
            sa = torch.sin(angles)

            radial = ca.unsqueeze(0).unsqueeze(2) * e0.unsqueeze(1) + sa.unsqueeze(0).unsqueeze(2) * e1.unsqueeze(1)

            center0 = pos_t.unsqueeze(1)
            center1 = (pos_t + fwd * lengths.unsqueeze(1)).unsqueeze(1)

            ring0 = center0 + radial * radii_0.unsqueeze(1).unsqueeze(2)
            ring1 = center1 + radial * radii_1.unsqueeze(1).unsqueeze(2)

            v_scaled = torch.cat([ring0, ring1], dim=1)
            v_flat = v_scaled.reshape(-1, 3)

            V_t = 2 * n_rad
            f_offsets = (torch.arange(M, device=device) * V_t).unsqueeze(1).unsqueeze(2)
            f_batch = (proto_f.unsqueeze(0) + f_offsets).reshape(-1, 3) + vert_offset

            n_flat = torch.cat([radial, radial], dim=1).reshape(-1, 3)

            c_stem = self.COLOR_STEM.to(device)
            c_pet = self.COLOR_PETIOLE.to(device)
            c_flat = torch.where(
                is_inode.unsqueeze(1),
                c_stem.unsqueeze(0),
                c_pet.unsqueeze(0)
            ).unsqueeze(1).expand(M, V_t, 3).reshape(-1, 3)

            op_flat = exist_t.unsqueeze(1).expand(M, V_t).reshape(-1, 1)
            o_vals = torch.where(is_inode, torch.tensor(0, device=device), torch.tensor(1, device=device))
            o_flat = o_vals.unsqueeze(1).expand(M, V_t).reshape(-1)

            all_verts.append(v_flat)
            all_faces.append(f_batch)
            all_normals.append(n_flat)
            all_colors.append(c_flat)
            all_opacities.append(op_flat)
            all_organs.append(o_flat)
            vert_offset += v_flat.shape[0]

        # -------------------------------------------------------------
        # 3. BATCH CURVED TUBES (ORGAN_PEDUNCLE=6 with upward gravitropic curvature)
        # -------------------------------------------------------------
        if ped_mask.any():
            n_seg = 6
            _, proto_f = _make_straight_tube_prototype(n_seg, n_rad, device)

            M = int(ped_mask.sum().item())
            R_ped = R_a[ped_mask]
            pos_ped = base_pos_a[ped_mask]
            s_ped = scale_a[ped_mask]
            exist_ped = exist_a[ped_mask]

            lengths = s_ped[:, 0].clamp(min=1e-3)
            radii = s_ped[:, 1].clamp(min=1e-4)

            fwd = R_ped[:, :, 1]
            e0 = R_ped[:, :, 0]
            e1 = R_ped[:, :, 2]

            z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
            dr = lengths / float(n_seg)
            curv_val = 160.0

            ring_centers = [pos_ped]
            ring_e0 = [e0]
            ring_e1 = [e1]

            cur_axis = fwd.clone()
            cur_p = pos_ped.clone()
            cur_e0 = e0.clone()
            cur_e1 = e1.clone()

            for s in range(n_seg):
                h_bend = torch.linalg.cross(cur_axis, z_axis.expand_as(cur_axis))
                h_norm = torch.linalg.norm(h_bend, dim=-1, keepdim=True)
                valid = (h_norm > 1e-4).squeeze(-1)

                h_unit = torch.where(
                    h_norm > 1e-4,
                    h_bend / (h_norm + 1e-8),
                    torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(cur_axis)
                )
                theta_curv = math.radians(curv_val) * dr
                theta_from_target = torch.acos(cur_axis[:, 2].clamp(-1.0, 1.0))

                cos_t = torch.cos(theta_curv).unsqueeze(-1)
                sin_t = torch.sin(theta_curv).unsqueeze(-1)

                k_cross_v = torch.linalg.cross(h_unit, cur_axis)
                k_dot_v = (h_unit * cur_axis).sum(dim=-1, keepdim=True)
                rot_axis = cur_axis * cos_t + k_cross_v * sin_t + h_unit * k_dot_v * (1.0 - cos_t)
                rot_axis = rot_axis / (torch.linalg.norm(rot_axis, dim=-1, keepdim=True) + 1e-8)

                exceed = (theta_curv.abs() >= theta_from_target).unsqueeze(-1)
                target = z_axis.expand_as(cur_axis)
                next_axis = torch.where(exceed, target, rot_axis)
                cur_axis = torch.where(valid.unsqueeze(-1), next_axis, target)

                k_cross_e0 = torch.linalg.cross(h_unit, cur_e0)
                k_dot_e0 = (h_unit * cur_e0).sum(dim=-1, keepdim=True)
                cur_e0 = cur_e0 * cos_t + k_cross_e0 * sin_t + h_unit * k_dot_e0 * (1.0 - cos_t)
                cur_e0 = cur_e0 / (torch.linalg.norm(cur_e0, dim=-1, keepdim=True) + 1e-8)
                cur_e1 = torch.linalg.cross(cur_axis, cur_e0)
                cur_e1 = cur_e1 / (torch.linalg.norm(cur_e1, dim=-1, keepdim=True) + 1e-8)

                cur_p = cur_p + cur_axis * dr.unsqueeze(-1)
                ring_centers.append(cur_p.clone())
                ring_e0.append(cur_e0.clone())
                ring_e1.append(cur_e1.clone())

            angles = torch.linspace(0.0, 2.0 * math.pi, n_rad + 1, device=device)[:-1]
            ca = torch.cos(angles)
            sa = torch.sin(angles)

            rings = []
            normals_list = []
            for seg_i in range(n_seg + 1):
                c_i = ring_centers[seg_i]
                e0_i = ring_e0[seg_i]
                e1_i = ring_e1[seg_i]
                rad_vec = ca.unsqueeze(0).unsqueeze(2) * e0_i.unsqueeze(1) + sa.unsqueeze(0).unsqueeze(2) * e1_i.unsqueeze(1)
                ring_v = c_i.unsqueeze(1) + rad_vec * radii.unsqueeze(1).unsqueeze(2)
                rings.append(ring_v)
                normals_list.append(rad_vec)

            v_ped = torch.stack(rings, dim=1)
            v_flat = v_ped.reshape(-1, 3)
            V_ped = (n_seg + 1) * n_rad
            f_offsets = (torch.arange(M, device=device) * V_ped).unsqueeze(1).unsqueeze(2)
            f_batch = (proto_f.unsqueeze(0) + f_offsets).reshape(-1, 3) + vert_offset

            n_ped = torch.stack(normals_list, dim=1).reshape(-1, 3)
            c_ped = self.COLOR_PEDUNCLE.to(device)
            c_flat = c_ped.unsqueeze(0).expand(v_flat.shape[0], 3)
            op_flat = exist_ped.unsqueeze(1).expand(M, V_ped).reshape(-1, 1)
            o_flat = torch.full((v_flat.shape[0],), 3, dtype=torch.long, device=device)

            all_verts.append(v_flat)
            all_faces.append(f_batch)
            all_normals.append(n_ped)
            all_colors.append(c_flat)
            all_opacities.append(op_flat)
            all_organs.append(o_flat)
            vert_offset += v_flat.shape[0]

        # -------------------------------------------------------------
        # 4. BATCH PODS (ORGAN_FRUIT=11)
        # -------------------------------------------------------------
        pod_mask = (ot_a == ORGAN_FRUIT)
        if pod_mask.any():
            try:
                v_pod, f_pod = self.asset_mgr.get_mesh_device("CowpeaPod.obj", device)
                n_pod = compute_face_normals_torch(v_pod, f_pod)
                M = int(pod_mask.sum().item())
                R_p = R_a[pod_mask]
                pos_p = base_pos_a[pod_mask]
                s_p = scale_a[pod_mask][:, 0:1]
                exist_pod = exist_a[pod_mask]

                v_scaled = v_pod.unsqueeze(0) * s_p.unsqueeze(1)
                v_rot = torch.bmm(v_scaled, R_p.transpose(1, 2)) + pos_p.unsqueeze(1)
                n_rot = torch.bmm(n_pod.unsqueeze(0).expand(M, -1, -1), R_p.transpose(1, 2))

                V = v_pod.shape[0]
                f_offsets = (torch.arange(M, device=device) * V).unsqueeze(1).unsqueeze(2)
                f_batch = (f_pod.unsqueeze(0) + f_offsets).reshape(-1, 3) + vert_offset

                v_flat = v_rot.reshape(-1, 3)
                n_flat = n_rot.reshape(-1, 3)
                c_flat = self.COLOR_POD.to(device).unsqueeze(0).expand(v_flat.shape[0], -1)
                op_flat = exist_pod.unsqueeze(1).expand(M, V).reshape(-1, 1)
                o_flat = torch.full((v_flat.shape[0],), 5, dtype=torch.int64, device=device)

                all_verts.append(v_flat)
                all_faces.append(f_batch)
                all_normals.append(n_flat)
                all_colors.append(c_flat)
                all_opacities.append(op_flat)
                all_organs.append(o_flat)
                vert_offset += v_flat.shape[0]
            except Exception:
                pass

        # -------------------------------------------------------------
        # 5. BATCH FLOWERS (ORGAN_FLOWER_OPEN=10, ORGAN_FLOWER_CLOSED=9)
        # -------------------------------------------------------------
        fl_mask = (ot_a == ORGAN_FLOWER_OPEN) | (ot_a == ORGAN_FLOWER_CLOSED)
        if fl_mask.any():
            fl_specs = ((ORGAN_FLOWER_OPEN, "CowpeaFlower_open_yellow.obj"), (ORGAN_FLOWER_CLOSED, "CowpeaFlower_closed_yellow.obj"))
            for fl_ot, fl_obj in fl_specs:
                fm = fl_mask & (ot_a == fl_ot)
                if not fm.any():
                    continue
                try:
                    v_fl, f_fl = self.asset_mgr.get_mesh_device(fl_obj, device)
                    n_fl = compute_face_normals_torch(v_fl, f_fl)
                    M = int(fm.sum().item())
                    R_f = R_a[fm]
                    pos_f = base_pos_a[fm]
                    s_f = scale_a[fm][:, 0:1]
                    exist_fl = exist_a[fm]

                    v_scaled = v_fl.unsqueeze(0) * s_f.unsqueeze(1)
                    v_rot = torch.bmm(v_scaled, R_f.transpose(1, 2)) + pos_f.unsqueeze(1)
                    n_rot = torch.bmm(n_fl.unsqueeze(0).expand(M, -1, -1), R_f.transpose(1, 2))

                    V = v_fl.shape[0]
                    f_offsets = (torch.arange(M, device=device) * V).unsqueeze(1).unsqueeze(2)
                    f_batch = (f_fl.unsqueeze(0) + f_offsets).reshape(-1, 3) + vert_offset

                    v_flat = v_rot.reshape(-1, 3)
                    n_flat = n_rot.reshape(-1, 3)
                    c_flat = self.COLOR_FLOWER.to(device).unsqueeze(0).expand(v_flat.shape[0], -1)
                    op_flat = exist_fl.unsqueeze(1).expand(M, V).reshape(-1, 1)
                    o_flat = torch.full((v_flat.shape[0],), 4, dtype=torch.int64, device=device)

                    all_verts.append(v_flat)
                    all_faces.append(f_batch)
                    all_normals.append(n_flat)
                    all_colors.append(c_flat)
                    all_opacities.append(op_flat)
                    all_organs.append(o_flat)
                    vert_offset += v_flat.shape[0]
                except Exception:
                    pass

        if not all_verts:
            empty3 = torch.zeros((0, 3), dtype=torch.float32, device=device)
            empty_f = torch.zeros((0, 3), dtype=torch.int64, device=device)
            empty_op = torch.zeros((0, 1), dtype=torch.float32, device=device)
            empty_o = torch.zeros((0,), dtype=torch.int64, device=device)
            return {'vertices': empty3, 'faces': empty_f, 'normals': empty3, 'colors': empty3, 'opacities': empty_op, 'organ_types': empty_o}

        return {
            'vertices': torch.cat(all_verts, dim=0),
            'faces': torch.cat(all_faces, dim=0),
            'normals': torch.cat(all_normals, dim=0),
            'colors': torch.cat(all_colors, dim=0),
            'opacities': torch.cat(all_opacities, dim=0),
            'organ_types': torch.cat(all_organs, dim=0)
        }


# =====================================================================
# Differentiable Converters (External to build_mesh_from_part_tensor)
# =====================================================================

def diff_node_to_part_tensor_14d(
    diff_tensor: torch.Tensor,
    return_existence: bool = True,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Converts a continuous/differentiable Node Tensor (N, 26) into a Canonical 14D Part Tensor (N, 14).

    Physical Scale Purity:
      - scale (cols 10-12) is ALWAYS preserved as true physical dimensions in metres.
      - existence is NEVER multiplied into scale (legacy scale-shrinking hack is completely deprecated).
      - If return_existence=True (default): returns (part_14d, existence) where existence is
        prob_selected * (1 - p(NONE)) for soft alpha rendering.
      - If return_existence=False: returns part_14d alone with uncorrupted physical scale.
    """
    from diffusion_based.models.plant_organ_array import ORGAN_NONE

    if device is not None:
        diff_tensor = diff_tensor.to(device)

    ot_probs = torch.softmax(diff_tensor[:, :13], dim=-1)
    exist = 1.0 - ot_probs[:, ORGAN_NONE]
    ot = torch.argmax(ot_probs, dim=-1)

    base_pos = diff_tensor[:, 13:16]
    rot_6d = diff_tensor[:, 16:22]
    scale = diff_tensor[:, 22:25]
    curvature = diff_tensor[:, 25:26]

    prob_selected = torch.gather(ot_probs, 1, ot.unsqueeze(1)).squeeze(-1)
    soft_exist = prob_selected * exist

    part_14d = torch.cat([ot.unsqueeze(-1).float(), base_pos, rot_6d, scale, curvature], dim=-1)
    if return_existence:
        return part_14d, soft_exist
    return part_14d





