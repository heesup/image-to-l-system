"""Exact Helios-style 3D plant geometry reconstruction from XML/15D graph.

This module replicates the forward kinematics of Helios's PlantArchitecture C++
class so that the Python-derived 3D points/meshes structurally match the Helios
PLY output. It is kept texture-free and uses simple geometric primitives:
  - internode/petiole: tube centerline vertices + radius
  - leaf: a simple quad mesh transformed by the same roll/pitch/yaw/compound
          rotation chain used in PlantArchitecture.cpp
  - floral bud/fruit/flower: ellipsoid

The module provides:
  - numpy helpers for ground-truth geometry from a Helios XML file
  - a PyTorch nn.Module that consumes either explicit geometry or 15D organ
    nodes and produces a differentiable point cloud / triangle mesh bank
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.helios_xml_parser import (
    HeliosXMLParser,
    OrganNode3D,
    Phytomer3D,
    ShootData,
)

_normalize = lambda x: x / (np.linalg.norm(x) + 1e-8)
_np_normalize = _normalize
_np_rodrigues = lambda p, axis, angle: p * math.cos(angle) + np.cross(axis, p) * math.sin(angle) + axis * np.dot(axis, p) * (1 - math.cos(angle))


# ═══════════════════════════════════════════════════════════════════════════════
# Numpy exact geometry from XML
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _as_safe(v: float) -> float:
    return _clamp(v, -1.0, 1.0)


@dataclass
class HeliosTube:
    """Centerline tube geometry (internode or petiole)."""
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    radii: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    organ: int = OrganNode3D.INTERNODE


@dataclass
class HeliosLeaflet:
    """Single leaflet mesh transformed into world space."""
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    organ: int = OrganNode3D.LEAF


@dataclass
class HeliosEllipsoid:
    """Flower / fruit / floral bud ellipsoid."""
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    radius: float = 0.0
    length: float = 0.0
    organ: int = OrganNode3D.FLORAL_BUD


@dataclass
class HeliosPlantGeometry:
    tubes: List[HeliosTube] = field(default_factory=list)
    leaflets: List[HeliosLeaflet] = field(default_factory=list)
    ellipsoids: List[HeliosEllipsoid] = field(default_factory=list)

    def get_geometry_tensors(
        self,
        device: Optional[torch.device] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Convert this numpy geometry to batched torch tensors for rasterization.

        Matches the 10-tensor output expected by HeliosGeometryRasterizer.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tube_verts_list, tube_radii_list, tube_organ_list = [], [], []
        for tube in self.tubes:
            if tube.vertices.shape[0] < 2:
                continue
            v = torch.from_numpy(tube.vertices).float().to(device)
            r = torch.from_numpy(tube.radii).float().to(device)
            o = torch.tensor(tube.organ, dtype=torch.long, device=device)
            for seg in range(v.shape[0] - 1):
                tube_verts_list.append(torch.stack([v[seg], v[seg + 1]], dim=0))
                tube_radii_list.append(torch.stack([r[seg], r[seg + 1]], dim=0))
                tube_organ_list.append(o)

        leaf_verts_list, leaf_faces_list, leaf_organ_list = [], [], []
        for lf in self.leaflets:
            if lf.vertices.shape[0] >= 3:
                v = torch.from_numpy(lf.vertices).float().to(device)
                f = torch.from_numpy(lf.faces).long().to(device) if lf.faces.shape[0] > 0 else torch.zeros((0, 3), dtype=torch.long, device=device)
                o = torch.tensor(lf.organ, dtype=torch.long, device=device)
                leaf_verts_list.append(v)
                leaf_faces_list.append(f)
                leaf_organ_list.append(o)

        ell_center_list, ell_radius_list, ell_length_list, ell_organ_list = [], [], [], []
        for ell in self.ellipsoids:
            ell_center_list.append(torch.from_numpy(ell.center).float().to(device))
            ell_radius_list.append(torch.tensor(ell.radius, dtype=torch.float32, device=device))
            ell_length_list.append(torch.tensor(ell.length, dtype=torch.float32, device=device))
            ell_organ_list.append(torch.tensor(ell.organ, dtype=torch.long, device=device))

        if tube_verts_list:
            tube_verts = torch.stack(tube_verts_list, dim=0).unsqueeze(0)
            tube_radii = torch.stack(tube_radii_list, dim=0).unsqueeze(0)
            tube_organs = torch.stack(tube_organ_list, dim=0).unsqueeze(0)
        else:
            tube_verts = torch.zeros((1, 0, 2, 3), device=device)
            tube_radii = torch.zeros((1, 0, 2), device=device)
            tube_organs = torch.zeros((1, 0), dtype=torch.long, device=device)

        if leaf_verts_list:
            max_v = max(v.shape[0] for v in leaf_verts_list)
            padded = [torch.cat([v, torch.zeros((max_v - v.shape[0], 3), device=device)], dim=0) if v.shape[0] < max_v else v for v in leaf_verts_list]
            leaf_verts = torch.stack(padded, dim=0).unsqueeze(0)
            leaf_organs = torch.stack(leaf_organ_list, dim=0).unsqueeze(0)
            leaf_faces = leaf_faces_list[0]
        else:
            leaf_verts = torch.zeros((1, 0, 4, 3), device=device)
            leaf_organs = torch.zeros((1, 0), dtype=torch.long, device=device)
            leaf_faces = torch.zeros((0, 3), dtype=torch.long, device=device)

        if ell_center_list:
            ell_centers = torch.stack(ell_center_list, dim=0).unsqueeze(0)
            ell_radii = torch.stack(ell_radius_list, dim=0).unsqueeze(0)
            ell_lengths = torch.stack(ell_length_list, dim=0).unsqueeze(0)
            ell_organs = torch.stack(ell_organ_list, dim=0).unsqueeze(0)
        else:
            ell_centers = torch.zeros((1, 0, 3), device=device)
            ell_radii = torch.zeros((1, 0), device=device)
            ell_lengths = torch.zeros((1, 0), device=device)
            ell_organs = torch.zeros((1, 0), dtype=torch.long, device=device)

        return (
            tube_verts, tube_radii, tube_organs,
            leaf_verts, leaf_faces, leaf_organs,
            ell_centers, ell_radii, ell_lengths, ell_organs,
        )

    def to_point_cloud(
        self,
        n_circ: int = 8,
        n_axis_per_seg: int = 2,
        leaf_subdiv_u: int = 6,
        leaf_subdiv_v: int = 8,
        ellipsoid_theta: int = 12,
        ellipsoid_phi: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample points on all geometry and return (xyz, colors, organ)."""
        pts_list, col_list, org_list = [], [], []

        for tube in self.tubes:
            if tube.vertices.shape[0] < 2:
                continue
            v = tube.vertices
            r = tube.radii
            # sample tube surface
            axis_dirs = v[1:] - v[:-1]
            seg_lengths = np.linalg.norm(axis_dirs, axis=-1)
            seg_dirs = np.where(
                seg_lengths[:, None] > 1e-8,
                axis_dirs / seg_lengths[:, None],
                np.array([0.0, 0.0, 1.0]),
            )
            for seg in range(len(seg_dirs)):
                z = seg_dirs[seg]
                # pick perpendicular
                if abs(z[2]) < 0.9:
                    tmp = np.array([0.0, 0.0, 1.0])
                else:
                    tmp = np.array([0.0, 1.0, 0.0])
                x = _np_normalize(np.cross(tmp, z))
                y = np.cross(z, x)
                for t in np.linspace(0.0, 1.0, n_axis_per_seg):
                    c = v[seg] + t * (v[seg + 1] - v[seg])
                    radius = r[seg] * (1.0 - t) + r[seg + 1] * t
                    for th in np.linspace(0.0, 2.0 * math.pi, n_circ, endpoint=False):
                        pts_list.append(
                            c + radius * (math.cos(th) * x + math.sin(th) * y)
                        )
                        org_list.append(tube.organ)

        for lf in self.leaflets:
            if lf.vertices.shape[0] < 3:
                continue
            # sample the actual triangle mesh (or a grid over a 4-vertex quad)
            v = lf.vertices
            if v.shape[0] == 4:
                for u in np.linspace(0.0, 1.0, leaf_subdiv_u):
                    for w in np.linspace(0.0, 1.0, leaf_subdiv_v):
                        p = (
                            (1 - u) * (1 - w) * v[0]
                            + u * (1 - w) * v[1]
                            + u * w * v[2]
                            + (1 - u) * w * v[3]
                        )
                        pts_list.append(p)
                        org_list.append(lf.organ)
            elif lf.faces.shape[0] > 0:
                for f in lf.faces:
                    a, b, c = v[f[0]], v[f[1]], v[f[2]]
                    tri_center = (a + b + c) / 3.0
                    # sample center + midpoints to cover the triangle
                    for frac in [(a + b) / 2.0, (b + c) / 2.0, (a + c) / 2.0, tri_center]:
                        pts_list.append(frac)
                        org_list.append(lf.organ)

        for ell in self.ellipsoids:
            a = ell.radius
            b = ell.radius
            c = ell.length * 0.5 if ell.length > 0 else ell.radius
            for th in np.linspace(0.0, 2.0 * math.pi, ellipsoid_theta, endpoint=False):
                for ph in np.linspace(0.0, math.pi, ellipsoid_phi):
                    pts_list.append(
                        ell.center
                        + np.array([
                            a * math.sin(ph) * math.cos(th),
                            b * math.sin(ph) * math.sin(th),
                            c * math.cos(ph),
                        ])
                    )
                    org_list.append(ell.organ)

        if not pts_list:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros((0,), dtype=np.uint8),
            )

        xyz = np.array(pts_list, dtype=np.float32)
        organs = np.array(org_list, dtype=np.uint8)
        colors = np.array([_ORGAN_COLORS[o] for o in organs], dtype=np.uint8)
        return xyz, colors, organs


_ORGAN_COLORS = {
    OrganNode3D.INTERNODE: np.array([139, 69, 19], dtype=np.uint8),
    OrganNode3D.PETIOLE: np.array([173, 255, 47], dtype=np.uint8),
    OrganNode3D.LEAF: np.array([34, 139, 34], dtype=np.uint8),
    OrganNode3D.FLORAL_BUD: np.array([255, 215, 0], dtype=np.uint8),
}


def _leaflet_local_mesh(
    leaf_scale: float,
    aspect: float = 0.7,
    subdivisions: int = 8,
    longitudinal_curvature: float = 0.0,
    lateral_curvature: float = 0.0,
    midrib_fold_fraction: float = 0.0,
    petiole_roll: float = 0.0,
    leaf_buckle_angle: float = 0.0,
    leaf_buckle_length: float = 1.0,
    wave_period: float = 0.0,
    wave_amplitude: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a Helios GenericLeafPrototype-style mesh in its local frame.

    Local frame:
        +x = along midrib (length), 0 at petiole base, 1 at tip
        +y = across width, centered on midrib
        +z = normal / curvature direction (positive = upward)

    Matches the procedural builder in
    Digital-Crops/libs/Helios/plugins/plantarchitecture/src/Assets.cpp:21-178.
    """
    L = max(leaf_scale, 1e-6)
    W = L * aspect

    # Grid dimensions: match C++ (Nx longitudinal, Ny lateral, forced even)
    Nx = max(1, subdivisions)
    Ny = max(1, int(math.ceil(aspect * Nx)))
    if Ny % 2 != 0:
        Ny += 1

    dx = 1.0 / float(Nx)
    dy = aspect / float(Ny)

    verts = np.zeros((Ny + 1, Nx + 1, 3), dtype=np.float64)

    for j in range(Ny + 1):
        dtheta = 0.0
        for i in range(Nx + 1):
            x = float(i) * dx
            y = float(j) * dy - 0.5 * aspect

            # Elliptical taper: leaf is narrow at base and tip, widest in the middle
            taper = math.sin(math.pi * x)
            y_tapered = y * taper

            # Midrib fold
            y_fold = math.cos(0.5 * midrib_fold_fraction * math.pi) * y_tapered
            z_fold = math.sin(0.5 * midrib_fold_fraction * math.pi) * abs(y_tapered)

            # Curvatures (positive = upward)
            z_xcurve = longitudinal_curvature * (x ** 4)
            z_ycurve = lateral_curvature * ((y_tapered / aspect) ** 4) if abs(aspect) > 1e-6 else 0.0

            # Petiole roll
            z_petiole = 0.0
            if abs(petiole_roll) > 1e-10:
                z_petiole = min(0.1, petiole_roll * ((7.0 * y_tapered / aspect) ** 4) * math.exp(-70.0 * x)) - 0.01 * (petiole_roll / abs(petiole_roll))

            # Start with folded/cambered position
            verts[j, i] = np.array([x, y_fold, z_fold + z_ycurve + z_petiole])

            # Longitudinal incremental rotation about local y
            rot_angle = 0.0
            if abs(longitudinal_curvature) > 1e-10 and i > 0:
                dtheta -= math.atan(4.0 * longitudinal_curvature * (x ** 3) * dx)
                verts[j, i] = _np_rodrigues(verts[j, i], np.array([0.0, 1.0, 0.0]), dtheta)
                rot_angle += dtheta

            # Leaf buckle (distal portion rotated downward about line parallel to y)
            if leaf_buckle_angle > 0.0:
                xf = leaf_buckle_length
                x_next = x + dx
                if x <= xf < x_next:
                    ang = 0.5 * math.radians(leaf_buckle_angle)
                    verts[j, i] = _np_rodrigues(verts[j, i] - np.array([xf, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), ang) + np.array([xf, 0.0, 0.0])
                    rot_angle += ang
                elif x > xf:
                    ang = math.radians(leaf_buckle_angle)
                    verts[j, i] = _np_rodrigues(verts[j, i] - np.array([xf, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), ang) + np.array([xf, 0.0, 0.0])
                    rot_angle += ang

            # Wave displacement along rotated local normal
            if wave_period > 0.0 and wave_amplitude > 0.0:
                wave_phase = (x + wave_period * float(j >= 0.5 * Ny)) * math.pi / wave_period
                z_wave = 2.0 * abs(y_tapered) * wave_amplitude * math.sin(wave_phase)
                verts[j, i, 0] += z_wave * math.sin(rot_angle)
                verts[j, i, 2] += z_wave * math.cos(rot_angle)

    # Scale to leaf length and flatten to (V, 3)
    verts_flat = verts.reshape(-1, 3).copy()
    verts_flat[:, 0] *= L
    verts_flat[:, 1] *= L
    verts_flat[:, 2] *= L

    faces = []
    for j in range(Ny):
        for i in range(Nx):
            v0 = j * (Nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (Nx + 1) + 1
            v3 = v0 + (Nx + 1)
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    faces = np.array(faces, dtype=np.int32)
    return verts_flat, faces


def _leaflet_from_node(node: OrganNode3D) -> HeliosLeaflet:
    """Build a HeliosLeaflet mesh from a 15D leaf node.

    The node stores position, direction, pitch/yaw/roll, and length/width.
    We orient the local prototype so the midrib aligns with the node's direction
    and apply yaw/roll similarly to PlantArchitecture.cpp.
    """
    verts, faces = _leaflet_local_mesh(node.length, aspect=0.7)
    midrib_dir = _np_normalize(node.direction)
    if np.linalg.norm(midrib_dir) < 1e-12:
        midrib_dir = np.array([0.0, 0.0, 1.0])

    # Choose width axis: perpendicular to midrib, mostly horizontal.
    if abs(midrib_dir[2]) < 0.9:
        world_up = np.array([0.0, 0.0, 1.0])
    else:
        world_up = np.array([0.0, 1.0, 0.0])
    width_axis = _np_normalize(np.cross(world_up, midrib_dir))
    if np.linalg.norm(width_axis) < 1e-12:
        width_axis = np.array([1.0, 0.0, 0.0])
    normal_axis = _np_normalize(np.cross(midrib_dir, width_axis))

    R = np.stack([midrib_dir, width_axis, normal_axis], axis=1)

    # The midrib direction already encodes pitch/yaw (direction = f(pitch, yaw)).
    # Only apply the leaf roll (twist about the midrib) to avoid double rotation.
    roll = math.radians(node.roll)

    cr, sr = math.cos(roll), math.sin(roll)
    Rr = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    R_total = R @ Rr
    verts = (R_total @ verts.T).T + node.position
    return HeliosLeaflet(vertices=verts.astype(np.float32), faces=faces, organ=OrganNode3D.LEAF)


def build_helios_geometry_from_nodes(nodes: List[OrganNode3D]) -> HeliosPlantGeometry:
    geom = HeliosPlantGeometry()

    for node in nodes:
        if node.existence <= 0.0:
            continue

        if node.organ_type == OrganNode3D.INTERNODE:
            if np.linalg.norm(node.direction) < 1e-12:
                continue
            axis = _np_normalize(node.direction)
            tip = node.tip_position if np.linalg.norm(node.tip_position) > 1e-12 else node.position + axis * node.length
            geom.tubes.append(HeliosTube(
                vertices=np.array([node.position, tip], dtype=np.float32),
                radii=np.array([node.radius, node.radius], dtype=np.float32),
                organ=OrganNode3D.INTERNODE,
            ))

        elif node.organ_type == OrganNode3D.PETIOLE:
            if np.linalg.norm(node.direction) < 1e-12:
                continue
            axis = _np_normalize(node.direction)
            tip = node.tip_position if np.linalg.norm(node.tip_position) > 1e-12 else node.position + axis * node.length
            geom.tubes.append(HeliosTube(
                vertices=np.array([node.position, tip], dtype=np.float32),
                radii=np.array([node.radius, node.radius * 0.5], dtype=np.float32),
                organ=OrganNode3D.PETIOLE,
            ))

        elif node.organ_type == OrganNode3D.LEAF:
            geom.leaflets.append(_leaflet_from_node(node))

        elif node.organ_type in (OrganNode3D.FLORAL_BUD, OrganNode3D.FLOWER, OrganNode3D.POD):
            if node.length > 1e-6:
                axis = _np_normalize(node.direction)
                tip = node.tip_position if np.linalg.norm(node.tip_position) > 1e-12 else node.position + axis * node.length
                geom.tubes.append(HeliosTube(
                    vertices=np.array([node.position, tip], dtype=np.float32),
                    radii=np.array([node.radius, node.radius * 0.5], dtype=np.float32),
                    organ=node.organ_type,
                ))
            geom.ellipsoids.append(HeliosEllipsoid(
                center=node.position.astype(np.float32),
                radius=float(node.radius),
                length=float(node.length),
                organ=node.organ_type,
            ))

    return geom


def _ghost_petiole_axis(parent_internode_axis: np.ndarray, cumulative_rotation: float) -> np.ndarray:
    """Create a ghost petiole reference vector for phytomers without explicit petioles.

    Matches PlantArchitecture.cpp (L1078-1088): ghost = cross(internode, z), with
    cumulative rotation about the internode axis by parent_node_index * phyllotactic_angle.
    """
    ghost = np.cross(parent_internode_axis, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(ghost) < 0.01:
        ghost = np.array([0.0, 1.0, 0.0])
    ghost = _np_normalize(ghost)
    if abs(cumulative_rotation) > 1e-10:
        ghost = _np_rodrigues(ghost, parent_internode_axis, cumulative_rotation)
    return ghost


def _get_perp(v: np.ndarray) -> np.ndarray:
    if abs(v[0]) < 0.9:
        perp = np.cross(v, np.array([1.0, 0.0, 0.0]))
    else:
        perp = np.cross(v, np.array([0.0, 1.0, 0.0]))
    return _np_normalize(perp)


def _interpolate_tube(vertices: List[np.ndarray], frac: float) -> np.ndarray:
    if not vertices:
        return np.zeros(3)
    if frac <= 0:
        return vertices[0].copy()
    if frac >= 1.0:
        return vertices[-1].copy()
    n = len(vertices) - 1
    pos = frac * n
    idx = int(pos)
    t = pos - idx
    if idx >= n:
        return vertices[-1].copy()
    return (1.0 - t) * vertices[idx] + t * vertices[idx + 1]


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch differentiable geometry sampler
# ═══════════════════════════════════════════════════════════════════════════════

class DifferentiableHeliosGeometry(nn.Module):
    """Differentiable point-cloud generator from explicit 3D geometry buffers.

    Input: a batch of explicit geometry descriptors (to be produced from 15D
    nodes by a helper). For now this module exposes sampling functions that
    operate on raw vertex/radius tensors so the forward pass stays differentiable.
    """

    INTERNODE = 0
    PETIOLE = 1
    LEAF = 2
    FLORAL_BUD = 3

    def __init__(
        self,
        n_cylinder_circ: int = 8,
        n_cylinder_axis_per_seg: int = 2,
        n_leaf_u: int = 6,
        n_leaf_v: int = 8,
        n_ellipsoid_theta: int = 12,
        n_ellipsoid_phi: int = 8,
    ):
        super().__init__()
        self.n_cylinder_circ = n_cylinder_circ
        self.n_cylinder_axis_per_seg = n_cylinder_axis_per_seg
        self.n_leaf_u = n_leaf_u
        self.n_leaf_v = n_leaf_v
        self.n_ellipsoid_theta = n_ellipsoid_theta
        self.n_ellipsoid_phi = n_ellipsoid_phi

    def sample_tube(
        self,
        vertices: torch.Tensor,  # (B, T, 3)
        radii: torch.Tensor,       # (B, T)
    ) -> torch.Tensor:
        """Sample points on a piecewise-linear tube surface.

        Returns (B, (T-1)*n_axis*n_circ, 3).
        """
        B, T, _ = vertices.shape
        if T < 2:
            return vertices.new_zeros((B, 0, 3))
        device = vertices.device
        seg_dir = vertices[:, 1:] - vertices[:, :-1]  # (B, T-1, 3)
        seg_len = torch.norm(seg_dir, dim=-1, keepdim=True).clamp(min=1e-8)
        z = seg_dir / seg_len

        not_z = torch.where(z[..., 2:3].abs() < 0.9,
                            torch.tensor([0.0, 0.0, 1.0], device=device),
                            torch.tensor([0.0, 1.0, 0.0], device=device))
        x = torch.cross(not_z.expand_as(z), z, dim=-1)
        x = F.normalize(x, dim=-1)
        y = torch.cross(z, x, dim=-1)

        t = torch.linspace(0.0, 1.0, self.n_cylinder_axis_per_seg, device=device)
        theta = torch.linspace(0.0, 2.0 * math.pi, self.n_cylinder_circ, device=device)

        # centers along segments: (B, T-1, n_axis, 3)
        v0 = vertices[:, :-1].unsqueeze(-2)
        v1 = vertices[:, 1:].unsqueeze(-2)
        centers = v0 + t.view(1, 1, -1, 1) * (v1 - v0)

        # radii interpolated along segment
        r0 = radii[:, :-1].unsqueeze(-1)  # (B, T-1, 1)
        r1 = radii[:, 1:].unsqueeze(-1)
        r_interp = r0 + t.view(1, 1, -1) * (r1 - r0)  # (B, T-1, n_axis)

        # circle offsets: (B, T-1, n_circ, 1, 3)
        cos_t = torch.cos(theta).view(1, 1, -1, 1)
        sin_t = torch.sin(theta).view(1, 1, -1, 1)
        x = x.unsqueeze(-2)  # (B, T-1, 1, 3)
        y = y.unsqueeze(-2)
        offsets = r_interp.unsqueeze(-1).unsqueeze(-1) * (
            cos_t * x + sin_t * y
        )  # (B, T-1, n_axis, n_circ, 3)

        centers = centers.unsqueeze(-2)  # (B, T-1, n_axis, 1, 3)
        pts = centers + offsets  # broadcast to (B, T-1, n_axis, n_circ, 3)
        pts = pts.permute(0, 1, 2, 3, 4).reshape(B, -1, 3)
        return pts

    def sample_leaf_quad(
        self,
        leaf_verts: torch.Tensor,  # (B, L, 4, 3)
    ) -> torch.Tensor:
        """Sample points on a bilinear leaf quad with slight arch."""
        B, L, _, _ = leaf_verts.shape
        if L == 0:
            return leaf_verts.new_zeros((B, 0, 3))
        device = leaf_verts.device
        u = torch.linspace(0.0, 1.0, self.n_leaf_u, device=device)
        v = torch.linspace(0.0, 1.0, self.n_leaf_v, device=device)
        uu = u.view(1, 1, 1, -1, 1, 1)
        vv = v.view(1, 1, 1, 1, -1, 1)

        v0 = leaf_verts[:, :, 0:1, :].unsqueeze(-2).unsqueeze(-2)
        v1 = leaf_verts[:, :, 1:2, :].unsqueeze(-2).unsqueeze(-2)
        v2 = leaf_verts[:, :, 2:3, :].unsqueeze(-2).unsqueeze(-2)
        v3 = leaf_verts[:, :, 3:4, :].unsqueeze(-2).unsqueeze(-2)

        pts = (
            (1 - uu) * (1 - vv) * v0
            + uu * (1 - vv) * v1
            + uu * vv * v2
            + (1 - uu) * vv * v3
        )
        pts = pts.reshape(B, L, -1, 3)
        return pts

    def sample_ellipsoid(
        self,
        center: torch.Tensor,  # (B, E, 3)
        radius: torch.Tensor,    # (B, E)
        length: torch.Tensor,    # (B, E)
    ) -> torch.Tensor:
        """Sample ellipsoid surface points."""
        B, E, _ = center.shape
        if E == 0:
            return center.new_zeros((B, 0, 3))
        device = center.device
        theta = torch.linspace(0.0, 2.0 * math.pi, self.n_ellipsoid_theta, device=device)
        phi = torch.linspace(0.0, math.pi, self.n_ellipsoid_phi, device=device)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="xy")
        theta_grid = theta_grid.unsqueeze(0).unsqueeze(0)  # (1,1,theta,phi)
        phi_grid = phi_grid.unsqueeze(0).unsqueeze(0)

        a = radius
        b = radius
        c = length * 0.5
        c = torch.where(c > 0, c, radius)

        dx = a.view(B, E, 1, 1) * torch.sin(phi_grid) * torch.cos(theta_grid)
        dy = b.view(B, E, 1, 1) * torch.sin(phi_grid) * torch.sin(theta_grid)
        dz = c.view(B, E, 1, 1) * torch.cos(phi_grid)

        pts = center.view(B, E, 1, 1, 3) + torch.stack([dx, dy, dz], dim=-1)
        pts = pts.reshape(B, E, -1, 3)
        return pts


def _direction_from_angles(pitches: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    """Convert pitch/yaw in degrees to (B, N, 3) direction vectors."""
    p = pitches * math.pi / 180.0
    y = yaws * math.pi / 180.0
    dx = torch.cos(p) * torch.cos(y)
    dy = torch.cos(p) * torch.sin(y)
    dz = torch.sin(p)
    return torch.stack([dx, dy, dz], dim=-1)


def _leaflet_local_mesh_torch(
    device: Optional[torch.device] = None,
    subdivisions: int = 8,
    aspect: float = 0.7,
    longitudinal_curvature: float = 0.0,
    lateral_curvature: float = 0.0,
    midrib_fold_fraction: float = 0.0,
    petiole_roll: float = 0.0,
    leaf_buckle_angle: float = 0.0,
    leaf_buckle_length: float = 1.0,
    wave_period: float = 0.0,
    wave_amplitude: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a Helios GenericLeafPrototype-style mesh in its local frame (torch).

    This is the differentiable torch counterpart of ``_leaflet_local_mesh``.
    Local frame convention:
      +x = along midrib (length), from base to tip
      +y = across width, centered on midrib
      +z = normal / upward curvature

    Scale is fixed to 1.0; callers should multiply by the desired leaf length.
    """
    L = 1.0
    W = L * aspect
    Nx = max(1, subdivisions)
    Ny = max(1, int(math.ceil(aspect * Nx)))
    if Ny % 2 != 0:
        Ny += 1

    dx = 1.0 / float(Nx)
    dy = aspect / float(Ny)

    j = torch.arange(Ny + 1, dtype=torch.float32, device=device)
    i = torch.arange(Nx + 1, dtype=torch.float32, device=device)
    y_grid = j * dy - 0.5 * aspect
    x_grid = i * dx
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing="ij")  # (Ny+1, Nx+1)

    # Elliptical taper: leaf is narrow at base and tip, widest in the middle
    taper = torch.sin(math.pi * x_grid)
    y_tapered = y_grid * taper

    # Midrib fold
    fold_scalar = torch.tensor(0.5 * midrib_fold_fraction * math.pi, device=device)
    y_fold = torch.cos(fold_scalar) * y_tapered
    z_fold = torch.sin(fold_scalar) * torch.abs(y_tapered)

    # Curvatures (positive = upward)
    z_xcurve = longitudinal_curvature * (x_grid ** 4)
    z_ycurve = lateral_curvature * ((y_tapered / max(aspect, 1e-6)) ** 4)

    # Petiole roll
    z_petiole = torch.zeros_like(x_grid)
    if abs(petiole_roll) > 1e-10:
        z_petiole = (
            torch.clamp(petiole_roll * ((7.0 * y_tapered / max(aspect, 1e-6)) ** 4) * torch.exp(-70.0 * x_grid), max=0.1)
            - 0.01 * torch.sign(torch.tensor(petiole_roll, device=device))
        )

    verts = torch.stack([x_grid, y_fold, z_fold + z_ycurve + z_petiole], dim=-1).clone()

    # Longitudinal incremental rotation about local y
    dtheta = torch.zeros_like(x_grid)
    rot_angle = torch.zeros_like(x_grid)
    if abs(longitudinal_curvature) > 1e-10:
        for col in range(1, Nx + 1):
            x_val = x_grid[:, col]
            dtheta[:, col] = dtheta[:, col - 1] - math.atan(4.0 * longitudinal_curvature * (x_val ** 3) * dx)
            rot_angle[:, col] = rot_angle[:, col - 1] + dtheta[:, col]

    # Apply longitudinal rotation around local y axis
    if abs(longitudinal_curvature) > 1e-10:
        cos_dt = torch.cos(dtheta)
        sin_dt = torch.sin(dtheta)
        # Rotate each column around y-axis by cumulative dtheta
        x_rot = verts[:, :, 0] * cos_dt - verts[:, :, 2] * sin_dt
        z_rot = verts[:, :, 0] * sin_dt + verts[:, :, 2] * cos_dt
        verts = torch.stack([x_rot, verts[:, :, 1], z_rot], dim=-1)

    # Leaf buckle (distal portion rotated downward about line parallel to y)
    if leaf_buckle_angle > 0.0:
        xf = leaf_buckle_length
        x_next = x_grid + dx
        mask_before = (x_grid <= xf) & (xf < x_next)
        mask_after = x_grid > xf
        ang_half = 0.5 * math.radians(leaf_buckle_angle)
        ang_full = math.radians(leaf_buckle_angle)

        cos_half = math.cos(ang_half)
        sin_half = math.sin(ang_half)
        cos_full = math.cos(ang_full)
        sin_full = math.sin(ang_full)

        x_rel = verts[:, :, 0] - xf
        # before buckle: half angle
        x_rot_b = x_rel * cos_half - verts[:, :, 2] * sin_half + xf
        z_rot_b = x_rel * sin_half + verts[:, :, 2] * cos_half
        # after buckle: full angle
        x_rot_a = x_rel * cos_full - verts[:, :, 2] * sin_full + xf
        z_rot_a = x_rel * sin_full + verts[:, :, 2] * cos_full

        x_new = torch.where(mask_before, x_rot_b, torch.where(mask_after, x_rot_a, verts[:, :, 0]))
        z_new = torch.where(mask_before, z_rot_b, torch.where(mask_after, z_rot_a, verts[:, :, 2]))
        verts = torch.stack([x_new, verts[:, :, 1], z_new], dim=-1)

    # Wave displacement along rotated local normal
    if wave_period > 0.0 and wave_amplitude > 0.0:
        half_ny = 0.5 * Ny
        wave_phase = (x_grid + wave_period * (j >= half_ny).float().unsqueeze(1)) * math.pi / wave_period
        z_wave = 2.0 * torch.abs(y_tapered) * wave_amplitude * torch.sin(wave_phase)
        verts[:, :, 0] += z_wave * torch.sin(rot_angle)
        verts[:, :, 2] += z_wave * torch.cos(rot_angle)

    # Scale to leaf length and flatten to (V, 3)
    verts = verts.reshape(-1, 3) * L

    faces = []
    for jj in range(Ny):
        for ii in range(Nx):
            v0 = jj * (Nx + 1) + ii
            v1 = v0 + 1
            v2 = v0 + (Nx + 1) + 1
            v3 = v0 + (Nx + 1)
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    faces_t = torch.tensor(faces, dtype=torch.int64, device=device)
    return verts, faces_t


def _build_leaflet_rotation_matrix(
    directions: torch.Tensor,
    rolls: torch.Tensor,
) -> torch.Tensor:
    """Build a (B, N, 3, 3) rotation matrix that maps local leaflet frame to world.

    Local frame convention (matches Helios GenericLeafPrototype):
      +x = along midrib (length), from base to tip
      +y = across width
      +z = normal / upward curvature

    The world-space midrib is given by ``directions`` (already unit vectors) and
    ``rolls`` is a rotation about the midrib in degrees.
    """
    B, N, _ = directions.shape
    device = directions.device

    x_axis = F.normalize(directions, dim=-1)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(x_axis)
    alt_up = torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(x_axis)
    up = torch.where(x_axis[..., 2:3].abs() < 0.9, world_up, alt_up)

    y_axis = F.normalize(torch.cross(up, x_axis, dim=-1), dim=-1)
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1)

    # Columns of R are the local axes expressed in world coordinates.
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (B, N, 3, 3)

    # Apply roll about the midrib (local x-axis).
    cr = torch.cos(rolls * math.pi / 180.0)
    sr = torch.sin(rolls * math.pi / 180.0)
    Rr = torch.zeros(B, N, 3, 3, device=device)
    Rr[..., 0, 0] = 1.0
    Rr[..., 1, 1] = cr
    Rr[..., 1, 2] = -sr
    Rr[..., 2, 1] = sr
    Rr[..., 2, 2] = cr

    return R @ Rr


def _expand_trifoliate_leaflets(
    local_verts: torch.Tensor,
    leaf_lengths: torch.Tensor,
    R_total: torch.Tensor,
    positions: torch.Tensor,
    is_leaf: torch.Tensor,
    exist_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand each leaf node into 3 trifoliate leaflets (Center, Left, Right).

    Returns leaf_verts (B, 3*N, V, 3) and a leaf existence/organ mask (B, 3*N).
    This is intended for legacy 15D/18D node arrays where one node represents
    one compound leaf rather than an individual leaflet.
    """
    B, N, _ = positions.shape
    device = positions.device
    V = local_verts.shape[0]

    leaflet_angles = [0.0, -math.pi / 2.0, math.pi / 2.0]
    leaflet_scale_factors = [1.0, 0.85, 0.85]
    all_leaflet_verts = []

    for rot_angle, scale_fac in zip(leaflet_angles, leaflet_scale_factors):
        local_scaled = local_verts.view(1, 1, V, 3) * (leaf_lengths * scale_fac).view(B, N, 1, 1)

        if abs(rot_angle) > 1e-5:
            ca = math.cos(rot_angle)
            sa = math.sin(rot_angle)
            R_rot = torch.zeros(B, N, 3, 3, device=device)
            R_rot[..., 0, 0] = ca
            R_rot[..., 0, 1] = -sa
            R_rot[..., 1, 0] = sa
            R_rot[..., 1, 1] = ca
            R_rot[..., 2, 2] = 1.0
            R_compound = R_total @ R_rot
        else:
            R_compound = R_total

        world_v = torch.einsum("bnij,bnvj->bniv", R_compound, local_scaled)
        world_v = torch.movedim(world_v, -2, -1) + positions.unsqueeze(2)
        all_leaflet_verts.append(world_v)

    world_verts = torch.cat(all_leaflet_verts, dim=1)  # (B, 3N, V, 3)

    is_leaf_expanded = is_leaf.repeat(1, 3)
    exist_expanded = exist_mask.repeat(1, 3)
    pos_expanded = positions.repeat(1, 3, 1)

    leaf_verts = torch.where(
        (is_leaf_expanded * exist_expanded).view(B, 3 * N, 1, 1) > 0,
        world_verts,
        pos_expanded.unsqueeze(2),
    )
    return leaf_verts, is_leaf_expanded


def _render_single_leaflets(
    local_verts: torch.Tensor,
    leaf_lengths: torch.Tensor,
    R_total: torch.Tensor,
    positions: torch.Tensor,
    is_leaf: torch.Tensor,
    exist_mask: torch.Tensor,
) -> torch.Tensor:
    """Render one leaflet per leaf node (for 22D arrays already expanded by the parser)."""
    B, N, _ = positions.shape
    V = local_verts.shape[0]

    local_scaled = local_verts.view(1, 1, V, 3) * leaf_lengths.view(B, N, 1, 1)
    world_v = torch.einsum("bnij,bnvj->bniv", R_total, local_scaled)
    world_v = torch.movedim(world_v, -2, -1) + positions.unsqueeze(2)  # (B, N, V, 3)

    leaf_verts = torch.where(
        (is_leaf * exist_mask).view(B, N, 1, 1) > 0,
        world_v,
        positions.unsqueeze(2),
    )
    return leaf_verts


def nodes_to_geometry_torch(
    nodes: torch.Tensor,
    parent_indices: Optional[torch.Tensor] = None,
    use_absolute_positions: bool = True,
    expand_trifoliate: Optional[bool] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Convert a batch of organ-graph nodes to explicit Helios geometry (torch).

    Supports 25D (current), 22D, 19D, 18D, and legacy 15D layouts. For 25D node
    arrays produced by ``HeliosXMLParser.get_all_organ_nodes()``, each LEAF node
    stores a full 3x3 local-to-world orientation matrix, which is used directly
    for leaflet rendering. For older 15D/18D/22D node arrays, the function falls
    back to direction+roll reconstruction.

    25D Layout:
      [0:3]   xyz          - base position (m)
      [3]     length       - organ length (m)
      [4]     radius       - organ radius (m)
      [5:14]  R_flat       - 3x3 orientation matrix (row-major), local frame
                             to world. For non-leaf organs the first column is
                             the direction vector and the rest are zero-padded.
      [14:20] organ_onehot - 6-channel one-hot
      [20]    shoot_id
      [21]    phytomer_idx
      [22]    existence    - confidence [0, 1]
      [23]    head_radius  - flower/pod head radius (m); 0 for non-floral
      [24]    parent_idx   - global parent node index (-1 = root)

    22D Legacy Layout (kept for backward compatibility):
      [5:8]   dir_xyz, [8] pitch, [9] yaw, [10] roll, [11:17] organ_onehot, etc.

    Returns:
        tube_verts:   (B, N, 2, 3)
        tube_radii:   (B, N, 2)
        tube_organs:  (B, N)
        leaf_verts:   (B, N_leaflets, V, 3)   world-space
        leaf_faces:   (F, 3)
        leaf_organs:  (B, N_leaflets)
        bud_centers:  (B, N, 3)
        bud_radii:    (B, N)
        bud_lengths:  (B, N)
        bud_organs:   (B, N)
    """
    B, N, D = nodes.shape
    device = nodes.device

    positions = nodes[..., :3]
    lengths = nodes[..., 3].clamp(min=1e-6)
    radii = nodes[..., 4].clamp(min=1e-6)

    has_R_matrix = D >= 25
    if has_R_matrix:
        R_flat = nodes[..., 5:14].reshape(B, N, 3, 3)
        organ_logits = nodes[..., 14:20]
        existence = nodes[..., 22]
        flower_head_radius = nodes[..., 23].clamp(min=0.0)
        if parent_indices is None:
            parent_indices = nodes[..., 24].long()
        # direction for tubes / buds is the first column of R
        dir_raw = R_flat[..., :, 0]
        pitches = torch.zeros_like(dir_raw[..., 0])
        yaws = torch.zeros_like(pitches)
        rolls = torch.zeros_like(pitches)
        if expand_trifoliate is None:
            expand_trifoliate = False
    elif D >= 22:
        dir_raw = nodes[..., 5:8]
        pitches = nodes[..., 8]
        yaws = nodes[..., 9]
        rolls = nodes[..., 10]
        organ_logits = nodes[..., 11:17]
        existence = nodes[..., 19]
        flower_head_radius = nodes[..., 20].clamp(min=0.0)
        if parent_indices is None:
            parent_indices = nodes[..., 21].long()
        if expand_trifoliate is None:
            expand_trifoliate = False
    else:
        dir_raw = nodes[..., 5:8]
        organ_logits = nodes[..., 8:14]   # 6-channel
        existence = nodes[..., 16] if D >= 17 else torch.ones_like(lengths)
        flower_head_radius = nodes[..., 17].clamp(min=0.0) if D >= 18 else torch.zeros_like(existence)
        if parent_indices is None and D >= 19:
            parent_indices = nodes[..., 18].long()
        pitches = dir_raw[..., 0]
        yaws = dir_raw[..., 1]
        rolls = torch.zeros_like(pitches)
        if expand_trifoliate is None:
            expand_trifoliate = True

    norm_567 = torch.linalg.norm(dir_raw, dim=-1, keepdim=True)
    is_dir_vec = (torch.abs(norm_567 - 1.0) < 0.2).float()
    unit_dirs = dir_raw / (norm_567 + 1e-8)
    angle_dirs = _direction_from_angles(pitches, yaws)
    directions = is_dir_vec * unit_dirs + (1.0 - is_dir_vec) * angle_dirs  # (B, N, 3)
    organ = organ_logits.argmax(dim=-1)                 # (B, N)

    # Existence mask: nodes with existence<=0.5 are not rendered.
    exist_mask = (existence > 0.5).float()

    # ------------------------------------------------------------------
    # Differentiable Position Forward Kinematics (parent_indices support)
    # ------------------------------------------------------------------
    organ_type = organ_logits.argmax(dim=-1)  # (B, N)
    is_internode = (organ_type == OrganNode3D.INTERNODE).float()
    is_petiole = (organ_type == OrganNode3D.PETIOLE).float()
    is_leaf = (organ_type == OrganNode3D.LEAF).float()

    scaled_lengths = torch.clamp(lengths, min=1e-4)
    tube_lengths = scaled_lengths * (is_internode + is_petiole) * exist_mask

    if parent_indices is not None and not use_absolute_positions:
        tips = positions.clone()
        updated_positions = positions.clone()
        for i in range(N):
            p_idx = parent_indices[:, i]  # (B,)
            mask = (p_idx >= 0) & (p_idx != i)  # (B,)
            if mask.any():
                b_idx = torch.arange(B, device=device)
                p_tips = tips[b_idx, p_idx]  # (B, 3)
                updated_positions[:, i] = torch.where(mask.unsqueeze(-1), p_tips, updated_positions[:, i])
            tips[:, i] = updated_positions[:, i] + tube_lengths[:, i].unsqueeze(-1) * directions[:, i]
        positions = updated_positions
    else:
        tips = positions + tube_lengths.unsqueeze(-1) * directions

    # ------------------------------------------------------------------
    # Tubes (internodes + petioles)
    # ------------------------------------------------------------------
    tube_verts = torch.stack([positions, tips], dim=2)          # (B, N, 2, 3)

    is_petiole = (organ == OrganNode3D.PETIOLE).float()
    r_tip = radii * (1.0 - 0.5 * is_petiole) * exist_mask
    tube_radii = torch.stack([radii * exist_mask, r_tip], dim=2)   # (B, N, 2)
    tube_organs = organ                                         # (B, N)

    # ------------------------------------------------------------------
    # Leaflets
    # ------------------------------------------------------------------
    leaf_subdivisions = 4 if getattr(nodes_to_geometry_torch, "_fast_render_mode", False) else 8
    local_verts, leaf_faces = _leaflet_local_mesh_torch(device=device, subdivisions=leaf_subdivisions, aspect=0.7)

    leaf_lengths = scaled_lengths * is_leaf * exist_mask                  # (B, N)
    if has_R_matrix:
        R_total = R_flat
        # Orthonormalize predicted rotation matrices via Gram-Schmidt to prevent
        # drift from noisy diffusion predictions.
        r1 = F.normalize(R_total[..., 0, :], dim=-1)
        r2 = R_total[..., 1, :] - (R_total[..., 1, :] * r1).sum(dim=-1, keepdim=True) * r1
        r2 = F.normalize(r2, dim=-1)
        r3 = torch.cross(r1, r2, dim=-1)
        R_total = torch.stack([r1, r2, r3], dim=-2)
    else:
        R_total = _build_leaflet_rotation_matrix(directions, rolls)

    if expand_trifoliate:
        leaf_verts, leaf_organs = _expand_trifoliate_leaflets(
            local_verts, leaf_lengths, R_total, positions, is_leaf, exist_mask
        )
    else:
        leaf_verts = _render_single_leaflets(
            local_verts, leaf_lengths, R_total, positions, is_leaf, exist_mask
        )
        leaf_organs = organ  # (B, N)

    # ------------------------------------------------------------------
    # Buds / Flowers / Pods
    # ------------------------------------------------------------------
    is_bud = (
        (organ == OrganNode3D.FLORAL_BUD) |
        (organ == OrganNode3D.FLOWER) |
        (organ == OrganNode3D.POD)
    ).float()

    flower_head_radius_clamped = flower_head_radius.clamp(min=0.0, max=0.015)
    has_head = (flower_head_radius_clamped > 1e-4).float()
    head_radius = flower_head_radius_clamped * has_head * exist_mask * is_bud     # (B, N)
    bud_radii = head_radius
    bud_centers = positions + directions * lengths.unsqueeze(-1)  # peduncle tip (B, N, 3)
    bud_lengths = bud_radii
    bud_organs = organ

    return (
        tube_verts, tube_radii, tube_organs,
        leaf_verts, leaf_faces, leaf_organs,
        bud_centers, bud_radii, bud_lengths, bud_organs,
    )


def nodes_to_geometry(
    nodes: torch.Tensor,
    parent_indices: Optional[torch.Tensor] = None,
) -> Tuple[List[List[HeliosTube]], List[List[HeliosLeaflet]], List[List[HeliosEllipsoid]]]:
    """Convert a batch of 19D organ-graph nodes to explicit Helios geometry.

    Returns per-batch lists of tubes, leaflets, and ellipsoids. The conversion is
    not fully differentiable (numpy output), so use this only for rendering.
    """
    nodes_np = nodes.detach().cpu().numpy()
    B, N, _ = nodes_np.shape

    organ_labels = nodes_np[..., 8:14].argmax(axis=-1)
    if parent_indices is not None:
        parents = parent_indices.detach().cpu().numpy()
    elif nodes_np.shape[-1] >= 19:
        parents = nodes_np[..., 18].astype(np.int64)
    else:
        parents = np.full((B, N), -1, dtype=np.int64)

    all_tubes: List[List[HeliosTube]] = [[] for _ in range(B)]
    all_leaflets: List[List[HeliosLeaflet]] = [[] for _ in range(B)]
    all_ellipsoids: List[List[HeliosEllipsoid]] = [[] for _ in range(B)]

    for b in range(B):
        positions = nodes_np[b, :, :3]
        directions = _direction_from_angles(
            torch.from_numpy(nodes_np[b, :, 5]),
            torch.from_numpy(nodes_np[b, :, 6]),
        ).numpy()
        lengths = nodes_np[b, :, 3]
        radii = nodes_np[b, :, 4]
        existence = nodes_np[b, :, 16] if nodes_np.shape[-1] >= 17 else nodes_np[b, :, 14]
        organ = organ_labels[b]

        for i in range(N):
            if existence[i] <= 0.5:
                continue
            base = positions[i]
            tip = base + lengths[i] * directions[i]
            r = max(radii[i], 1e-6)
            verts = np.stack([base, tip], axis=0).astype(np.float32)
            rad = np.array([r, r], dtype=np.float32)

            if organ[i] in (OrganNode3D.INTERNODE, OrganNode3D.PETIOLE):
                all_tubes[b].append(HeliosTube(vertices=verts, radii=rad, organ=int(organ[i])))
            elif organ[i] == OrganNode3D.LEAF:
                # Build a single leaflet mesh oriented by the 15D node direction.
                # The 15D graph already contains one node per Helios leaflet, so we
                # do not expand a single node into multiple leaflets here.
                scale = max(lengths[i], 1e-6)
                x = directions[i] / (np.linalg.norm(directions[i]) + 1e-8)

                # Perpendicular width axis in the horizontal-ish plane.
                # Use cross(tmp, x) (not cross(x, tmp)) so that the leaf width axis
                # matches the Helios XML parser convention, where local +y ends up
                # perpendicular to world z and the leaf midrib.
                tmp = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                y = np.cross(tmp, x)
                y = y / (np.linalg.norm(y) + 1e-8)
                z = np.cross(x, y)

                local_verts, faces = _leaflet_local_mesh(scale, aspect=0.7)
                world_verts = (local_verts[:, 0:1] * x
                               + local_verts[:, 1:2] * y
                               + local_verts[:, 2:3] * z)
                world_verts = (world_verts + base).astype(np.float32)
                all_leaflets[b].append(HeliosLeaflet(vertices=world_verts, faces=faces, organ=OrganNode3D.LEAF))
            elif organ[i] in (OrganNode3D.FLORAL_BUD, OrganNode3D.FLOWER, OrganNode3D.POD):
                all_ellipsoids[b].append(HeliosEllipsoid(
                    center=base.astype(np.float32),
                    radius=float(r),
                    length=float(lengths[i]),
                    organ=int(organ[i]),
                ))
    return all_tubes, all_leaflets, all_ellipsoids


def nodes_to_point_cloud(
    nodes: torch.Tensor,
    n_cylinder_circ: int = 8,
    n_cylinder_axis: int = 4,
    n_leaf_u: int = 6,
    n_leaf_v: int = 10,
    n_ellipsoid_theta: int = 8,
    n_ellipsoid_phi: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable point-cloud sampler from 15D organ-graph nodes.

    Input shape: (B, N, 15) where channels are
      [x, y, z, length, radius, pitch, yaw, parent_index, one_hot_organ (4D),
       existence].
    Returns:
      pts: (B, M, 3)
      organ_weights: (B, M, 4) soft one-hot weights
      existence_weights: (B, M, 1)
    """
    B, N, _ = nodes.shape
    device = nodes.device

    positions = nodes[..., :3]
    lengths = nodes[..., 3].clamp(min=1e-6)
    radii = nodes[..., 4].clamp(min=1e-6)
    pitches = nodes[..., 5]
    yaws = nodes[..., 6]
    organ_types = nodes[..., 8:12]
    existence = nodes[..., 14].unsqueeze(-1)  # (B, N, 1)

    p_rad = pitches * math.pi / 180.0
    y_rad = yaws * math.pi / 180.0
    dx = torch.cos(p_rad) * torch.cos(y_rad)
    dy = torch.cos(p_rad) * torch.sin(y_rad)
    dz = torch.sin(p_rad)
    directions = torch.stack([dx, dy, dz], dim=-1)
    tips = positions + lengths.unsqueeze(-1) * directions

    sampler = DifferentiableHeliosGeometry(
        n_cylinder_circ=n_cylinder_circ,
        n_cylinder_axis_per_seg=n_cylinder_axis,
        n_leaf_u=n_leaf_u,
        n_leaf_v=n_leaf_v,
        n_ellipsoid_theta=n_ellipsoid_theta,
        n_ellipsoid_phi=n_ellipsoid_phi,
    ).to(device)

    # Cylinder samples for internode (organ 0) and petiole (organ 1)
    # Treat each node as a 2-vertex tube. Flatten batch+nodes into one tube batch,
    # sample, then reshape back.
    node_tube_verts = torch.stack([positions, tips], dim=2)  # (B, N, 2, 3)
    node_tube_radii = torch.stack([radii, radii], dim=2)    # (B, N, 2)
    flat_verts = node_tube_verts.reshape(B * N, 2, 3)
    flat_radii = node_tube_radii.reshape(B * N, 2)
    flat_cyl = sampler.sample_tube(flat_verts, flat_radii)  # (B*N, K_cyl, 3)
    K_cyl = flat_cyl.shape[1]
    cylinder_pts = flat_cyl.reshape(B, N, K_cyl, 3)

    # Leaf quad samples (organ 2): build a simple quad oriented by direction
    # Pick perpendicular vector for width axis
    not_z = torch.where(
        directions[..., 2:3].abs() < 0.9,
        torch.tensor([0.0, 0.0, 1.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )
    width_axis = F.normalize(torch.cross(not_z.expand_as(directions), directions, dim=-1), dim=-1)
    up_axis = torch.cross(directions, width_axis, dim=-1)

    u = torch.linspace(-0.5, 0.5, n_leaf_u, device=device)
    v = torch.linspace(0.0, 1.0, n_leaf_v, device=device)
    vv = v.view(1, 1, 1, -1, 1)  # length direction
    uu = u.view(1, 1, -1, 1, 1)  # width direction
    # taper to a point at the tip
    taper = (1.0 - v ** 0.8).view(1, 1, 1, -1, 1)

    length_grid = lengths.view(B, N, 1, 1, 1) * vv
    width_grid = (lengths * 0.7).view(B, N, 1, 1, 1) * uu * taper
    base_grid = positions.view(B, N, 1, 1, 3)
    dir_grid = directions.view(B, N, 1, 1, 3)
    w_axis_grid = width_axis.view(B, N, 1, 1, 3)
    u_axis_grid = up_axis.view(B, N, 1, 1, 3)
    leaf_pts = (
        base_grid
        + dir_grid * length_grid
        + w_axis_grid * width_grid
        + u_axis_grid * (length_grid * 0.02 * torch.sin(math.pi * vv))
    )
    leaf_pts = leaf_pts.reshape(B, N, -1, 3)

    # Ellipsoid samples for floral bud/flower/fruit (organ 3)
    ellipsoid_pts = sampler.sample_ellipsoid(positions, radii, lengths)

    pts_per_organ = [cylinder_pts, cylinder_pts, leaf_pts, ellipsoid_pts]
    all_pts, all_weights, all_organ_idx = [], [], []
    for organ_idx, organ_pts in enumerate(pts_per_organ):
        K = organ_pts.shape[2]
        organ_weight = organ_types[..., organ_idx].unsqueeze(-1)  # (B, N, 1)
        organ_weight = organ_weight.expand(B, N, K).reshape(B, -1)
        pts_flat = organ_pts.reshape(B, -1, 3)
        exist_flat = existence.expand(B, N, K).reshape(B, -1)
        combined_weight = organ_weight * exist_flat
        all_pts.append(pts_flat)
        all_weights.append(combined_weight)
        all_organ_idx.append(torch.full((B, pts_flat.shape[1]), organ_idx, dtype=torch.long, device=device))

    pts = torch.cat(all_pts, dim=1)
    existence_weights = torch.cat(all_weights, dim=1).unsqueeze(-1)
    organ_idx_cat = torch.cat(all_organ_idx, dim=1)
    organ_weights = F.one_hot(organ_idx_cat, num_classes=4).float() * existence_weights
    return pts, organ_weights, existence_weights


class DifferentiablePlantPointCloud(nn.Module):
    """Backward-compatible wrapper: 15D organ nodes -> point cloud."""

    def __init__(
        self,
        n_cylinder_circ: int = 8,
        n_cylinder_axis: int = 4,
        n_leaf_u: int = 6,
        n_leaf_v: int = 10,
        n_ellipsoid_theta: int = 8,
        n_ellipsoid_phi: int = 6,
    ):
        super().__init__()
        self.n_cylinder_circ = n_cylinder_circ
        self.n_cylinder_axis = n_cylinder_axis
        self.n_leaf_u = n_leaf_u
        self.n_leaf_v = n_leaf_v
        self.n_ellipsoid_theta = n_ellipsoid_theta
        self.n_ellipsoid_phi = n_ellipsoid_phi

    def forward(
        self,
        nodes: torch.Tensor,
        parent_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return nodes_to_point_cloud(
            nodes,
            n_cylinder_circ=self.n_cylinder_circ,
            n_cylinder_axis=self.n_cylinder_axis,
            n_leaf_u=self.n_leaf_u,
            n_leaf_v=self.n_leaf_v,
            n_ellipsoid_theta=self.n_ellipsoid_theta,
            n_ellipsoid_phi=self.n_ellipsoid_phi,
        )


# Keep backward-compatible alias
DifferentiablePlantPointCloud = DifferentiablePlantPointCloud


class HeliosPlantGeometryTorch(nn.Module):
    """PyTorch 3D Plant Geometry Model with Bi-Directional XML Sync.

    Stores explicit 3D plant geometry (tubes, leaflets, ellipsoids) directly as
    PyTorch Parameter / Tensor objects on GPU or CPU.

    Supported Operations:
      - geom_torch = HeliosPlantGeometryTorch.from_xml(xml_path, device=device)
      - renderer(geom_torch, focus_plant=True, background="black")
      - optimizer = torch.optim.Adam(geom_torch.parameters(), lr=0.01)
    """

    def __init__(
        self,
        tube_verts: torch.Tensor,       # (N_tubes, 2, 3)
        tube_radii: torch.Tensor,       # (N_tubes, 2)
        tube_organs: torch.Tensor,      # (N_tubes,)
        leaf_verts: torch.Tensor,       # (N_leaflets, V, 3)
        leaf_faces: torch.Tensor,       # (F, 3)
        leaf_organs: torch.Tensor,      # (N_leaflets,)
        ell_centers: torch.Tensor,      # (N_ellipsoids, 3)
        ell_radii: torch.Tensor,        # (N_ellipsoids,)
        ell_lengths: torch.Tensor,      # (N_ellipsoids,)
        ell_organs: torch.Tensor,       # (N_ellipsoids,)
        leaf_scales: Optional[torch.Tensor] = None, # (N_leaflets,) trainable scale factors
        tube_scales: Optional[torch.Tensor] = None, # (N_tubes,) trainable scale factors
        leaf_existence: Optional[torch.Tensor] = None, # (N_leaflets,) continuous existence [0,1]
        tube_existence: Optional[torch.Tensor] = None, # (N_tubes,) continuous existence [0,1]
        bud_existence: Optional[torch.Tensor] = None, # (N_ellipsoids,) continuous existence [0,1]
    ):
        super().__init__()
        self.register_buffer("tube_verts_base", tube_verts)
        self.register_buffer("tube_radii_base", tube_radii)
        self.register_buffer("tube_organs", tube_organs)

        self.register_buffer("leaf_verts_base", leaf_verts)
        self.register_buffer("leaf_faces", leaf_faces)
        self.register_buffer("leaf_organs", leaf_organs)

        self.register_buffer("ell_centers", ell_centers)
        self.register_buffer("ell_radii", ell_radii)
        self.register_buffer("ell_lengths", ell_lengths)
        self.register_buffer("ell_organs", ell_organs)

        N_leaves = leaf_verts.shape[0]
        N_tubes = tube_verts.shape[0]
        N_buds = ell_centers.shape[0]

        if leaf_scales is None:
            leaf_scales = torch.ones(N_leaves, device=leaf_verts.device)
        if tube_scales is None:
            tube_scales = torch.ones(N_tubes, device=tube_verts.device)
        if leaf_existence is None:
            leaf_existence = torch.ones(N_leaves, device=leaf_verts.device)
        if tube_existence is None:
            tube_existence = torch.ones(N_tubes, device=tube_verts.device)
        if bud_existence is None:
            bud_existence = torch.ones(N_buds, device=ell_centers.device)

        self.leaf_scales = nn.Parameter(leaf_scales)
        self.tube_scales = nn.Parameter(tube_scales)
        self.leaf_existence = nn.Parameter(leaf_existence)
        self.tube_existence = nn.Parameter(tube_existence)
        self.bud_existence = nn.Parameter(bud_existence)


    def get_geometry_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute current 3D geometry tensors with autograd scaling and existence applied."""
        if self.leaf_verts_base.shape[0] > 0:
            center = self.leaf_verts_base.mean(dim=1, keepdim=True)
            scales = self.leaf_scales.clamp(0.1, 3.0).view(-1, 1, 1)
            leaf_verts = center + (self.leaf_verts_base - center) * scales
            # Apply continuous existence: existence -> scale multiplier on each leaf
            leaf_exist = self.leaf_existence.clamp(0.0, 1.0).view(-1, 1, 1)
            leaf_verts = leaf_verts * leaf_exist
        else:
            leaf_verts = self.leaf_verts_base

        if self.tube_radii_base.shape[0] > 0:
            scales = self.tube_scales.clamp(0.1, 3.0).view(-1, 1)
            tube_radii = self.tube_radii_base * scales
            tube_exist = self.tube_existence.clamp(0.0, 1.0).view(-1, 1)
            tube_radii = tube_radii * tube_exist
        else:
            tube_radii = self.tube_radii_base

        ell_radii = self.ell_radii
        if self.ell_radii.shape[0] > 0:
            bud_exist = self.bud_existence.clamp(0.0, 1.0).view(-1)
            ell_radii = self.ell_radii * bud_exist

        return (
            self.tube_verts_base.unsqueeze(0),
            tube_radii.unsqueeze(0),
            self.tube_organs.unsqueeze(0),
            leaf_verts.unsqueeze(0),
            self.leaf_faces,
            self.leaf_organs.unsqueeze(0),
            self.ell_centers.unsqueeze(0),
            ell_radii.unsqueeze(0),
            self.ell_lengths.unsqueeze(0),
            self.ell_organs.unsqueeze(0),
            torch.cat([self.leaf_existence.clamp(0.0, 1.0),
                       self.tube_existence.clamp(0.0, 1.0),
                       self.bud_existence.clamp(0.0, 1.0)], dim=0),
            torch.cat([self.leaf_scales.clamp(0.1, 3.0),
                       self.tube_scales.clamp(0.1, 3.0)], dim=0),
        )



