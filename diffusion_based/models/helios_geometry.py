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
    _normalize as _np_normalize,
    _rotate_point_about_line as _np_rodrigues,
)


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


def _leaflet_local_mesh(leaf_scale: float, aspect: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """Return a simple leaf prototype in its local frame.

    Local frame matches Helios GenericLeafPrototype:
        +x = along midrib (length), from base to tip
        +y = across width
        +z = normal / curvature direction
    Returns a dense grid of vertices and triangle faces.
    """
    L = max(leaf_scale, 1e-6)
    W = L * aspect
    Nx = 8
    Ny = 6
    dx = 1.0 / Nx
    dy = W / Ny

    verts = []
    for j in range(Ny + 1):
        y = j * dy - 0.5 * W
        for i in range(Nx + 1):
            x = i * dx
            # elliptical taper in y: widest near middle of length
            taper = math.sin(math.pi * x)
            y_eff = y * taper
            # slight longitudinal arch (concave/upward)
            z_arch = 0.08 * L * math.sin(math.pi * x) * (1.0 - 4.0 * (y / max(W, 1e-6)) ** 2)
            verts.append([x * L, y_eff, z_arch])

    verts = np.array(verts, dtype=np.float64)
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
    return verts, faces


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

        elif node.organ_type == OrganNode3D.FLORAL_BUD:
            if node.length > 1e-6:
                axis = _np_normalize(node.direction)
                tip = node.tip_position if np.linalg.norm(node.tip_position) > 1e-12 else node.position + axis * node.length
                geom.tubes.append(HeliosTube(
                    vertices=np.array([node.position, tip], dtype=np.float32),
                    radii=np.array([node.radius, node.radius * 0.5], dtype=np.float32),
                    organ=OrganNode3D.FLORAL_BUD,
                ))

    return geom


def build_helios_geometry_from_xml(xml_path: str) -> HeliosPlantGeometry:
    """Reconstruct explicit 3D geometry from a Helios XML file.

    This follows PlantArchitecture.cpp as closely as possible using the raw
    parameters stored in the XML.
    """
    parser = HeliosXMLParser(xml_path)
    parser.parse()

    geom = HeliosPlantGeometry()

    # We will re-run the forward kinematics from the raw phytomer parameters.
    # The parser already has helper data structures in self.shoots.
    # For exactness we recompute here from the raw XML elements directly.
    root = parser.root
    plant_elem = root.find(".//plant_instance")
    if plant_elem is None:
        return geom

    base_position = parser.base_position.copy()
    shoots_data: Dict[int, ShootData] = {}

    # First pass: parse all shoots
    for shoot_elem in plant_elem.findall("shoot"):
        sd = parser._parse_shoot_element(shoot_elem)
        shoots_data[sd.shoot_id] = sd

    # Compute parent axis info from parsed shoots
    for sd in sorted(shoots_data.values(), key=lambda s: s.shoot_id):
        _reconstruct_shoot_geometry_exact(sd, shoots_data, base_position)

    # Build geometry objects
    for sd in sorted(shoots_data.values(), key=lambda s: s.shoot_id):
        for phyt in sd.phytomers:
            # Internode tube
            if phyt.internode_vertices:
                geom.tubes.append(HeliosTube(
                    vertices=np.array(phyt.internode_vertices, dtype=np.float32),
                    radii=np.array(phyt.internode_radii, dtype=np.float32),
                    organ=OrganNode3D.INTERNODE,
                ))

            for pet in phyt.petioles:
                # Petiole tube
                if pet.get("vertices"):
                    geom.tubes.append(HeliosTube(
                        vertices=np.array(pet["vertices"], dtype=np.float32),
                        radii=_petiole_radii(pet, phyt, sd),
                        organ=OrganNode3D.PETIOLE,
                    ))

                # Leaflets
                for leaf in pet.get("leaves", []):
                    if "mesh_verts" in leaf:
                        geom.leaflets.append(HeliosLeaflet(
                            vertices=leaf["mesh_verts"].astype(np.float32),
                            faces=leaf.get("mesh_faces", _leaflet_local_mesh(1.0)[1]),
                            organ=OrganNode3D.LEAF,
                        ))

                # Floral buds
                for fbud in pet.get("floral_buds", []):
                    state = fbud.get("bud_state", 0)
                    # only render expanded flowers/fruits
                    if state in [3, 4] or (state >= 5 and fbud.get("peduncle_length", 0.0) < 0.05):
                        center = fbud.get("base_pos", pet.get("tip_pos", phyt.internode_tip))
                        geom.ellipsoids.append(HeliosEllipsoid(
                            center=center.copy(),
                            radius=fbud.get("peduncle_radius", 0.0),
                            length=fbud.get("peduncle_length", 0.0),
                            organ=OrganNode3D.FLORAL_BUD,
                        ))

    return geom


def _petiole_radii(pet: Dict, phyt: Phytomer3D, sd: ShootData) -> np.ndarray:
    """Build per-vertex petiole radii with taper, matching Helios."""
    verts = pet.get("vertices", [])
    n = len(verts)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    base_r = pet.get("radius", 0.0)
    taper = pet.get("taper", 0.25)
    radii = np.zeros(n, dtype=np.float64)
    for j in range(n):
        radii[j] = base_r * (1.0 - taper / max(n - 1, 1) * j)
    return radii.astype(np.float32)


def _reconstruct_shoot_geometry_exact(
    sd: ShootData,
    all_shoots: Dict[int, ShootData],
    base_position: np.ndarray,
):
    """Recompute exact 3D geometry for one shoot from raw phytomer parameters.

    Mirrors PlantArchitecture.cpp Phytomer constructor.
    """
    sd.internode_vertices = []
    sd.internode_radii = []

    for phyt_idx, phyt in enumerate(sd.phytomers):
        # --- determine parent axes ---
        parent_internode_axis = np.array([0.0, 0.0, 1.0])
        parent_petiole_axis = np.array([0.0, -1.0, 0.0])

        if phyt_idx > 0:
            prev = sd.phytomers[phyt_idx - 1]
            parent_internode_axis = _np_normalize(prev.internode_dir)
            if prev.petioles and "axis" in prev.petioles[0]:
                parent_petiole_axis = _np_normalize(prev.petioles[0]["axis"])
            else:
                parent_petiole_axis = _get_perp(parent_internode_axis)
        elif sd.parent_shoot_id >= 0:
            parent_sd = all_shoots.get(sd.parent_shoot_id)
            if parent_sd and sd.parent_node_index < len(parent_sd.phytomers):
                parent_phyt = parent_sd.phytomers[sd.parent_node_index]
                parent_internode_axis = _np_normalize(parent_phyt.internode_dir)
                pet_idx = min(sd.parent_petiole_index, max(0, len(parent_phyt.petioles) - 1))
                if parent_phyt.petioles and "axis" in parent_phyt.petioles[pet_idx]:
                    parent_petiole_axis = _np_normalize(parent_phyt.petioles[pet_idx]["axis"])
                else:
                    parent_petiole_axis = _get_perp(parent_internode_axis)

        petiole_rotation_axis = np.cross(parent_internode_axis, parent_petiole_axis)
        if np.linalg.norm(petiole_rotation_axis) < 1e-6:
            petiole_rotation_axis = np.array([1.0, 0.0, 0.0])
        else:
            petiole_rotation_axis = _np_normalize(petiole_rotation_axis)

        internode_axis = parent_internode_axis.copy()

        # --- base rotation for first phytomer ---
        if phyt_idx == 0:
            pitch_rad = math.radians(phyt.internode_pitch)
            if abs(pitch_rad) > 1e-10:
                internode_axis = _np_rodrigues(internode_axis, petiole_rotation_axis, 0.5 * pitch_rad)

            base_rot = np.radians(sd.base_rotation)
            if abs(base_rot[2]) > 1e-10:
                petiole_rotation_axis = _np_rodrigues(petiole_rotation_axis, parent_internode_axis, base_rot[2])
                internode_axis = _np_rodrigues(internode_axis, parent_internode_axis, base_rot[2])

            if abs(base_rot[0]) > 1e-10:
                base_pitch_axis = -np.cross(parent_internode_axis, parent_petiole_axis)
                if np.linalg.norm(base_pitch_axis) > 1e-10:
                    base_pitch_axis = _np_normalize(base_pitch_axis)
                    petiole_rotation_axis = _np_rodrigues(petiole_rotation_axis, base_pitch_axis, -base_rot[0])
                    internode_axis = _np_rodrigues(internode_axis, base_pitch_axis, -base_rot[0])

            if abs(base_rot[1]) > 1e-10:
                petiole_rotation_axis = _np_rodrigues(petiole_rotation_axis, parent_internode_axis, base_rot[1])
                internode_axis = _np_rodrigues(internode_axis, parent_internode_axis, base_rot[1])
        else:
            pitch_rad = math.radians(phyt.internode_pitch)
            if abs(pitch_rad) > 1e-10:
                internode_axis = _np_rodrigues(internode_axis, petiole_rotation_axis, -1.25 * pitch_rad)

        shoot_bending_axis = np.cross(internode_axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(shoot_bending_axis) < 1e-6:
            shoot_bending_axis = np.array([0.0, 1.0, 0.0])
        else:
            shoot_bending_axis = _np_normalize(shoot_bending_axis)

        internode_axis = _np_normalize(internode_axis)

        # --- internode base ---
        if phyt_idx == 0:
            if sd.parent_shoot_id < 0:
                internode_base = base_position.copy()
            else:
                parent_sd = all_shoots.get(sd.parent_shoot_id)
                if parent_sd and sd.parent_node_index < len(parent_sd.internode_vertices):
                    internode_base = parent_sd.internode_vertices[sd.parent_node_index][-1].copy()
                else:
                    internode_base = base_position.copy()
        else:
            internode_base = sd.internode_vertices[phyt_idx - 1][-1].copy()

        # --- internode segments with curvature/yaw perturbations ---
        n_seg = max(1, getattr(phyt, "_internode_length_segments", 2))
        dr = phyt.internode_length / float(n_seg)
        curv_pert = getattr(phyt, "_curvature_perturbations", [])
        yaw_pert = getattr(phyt, "_yaw_perturbations", [])

        vertices = [internode_base.copy()]
        radii = [phyt.internode_radius]
        axis = internode_axis.copy()

        for seg in range(1, n_seg + 1):
            if curv_pert:
                idx = seg - 1
                if idx < len(curv_pert):
                    ang = math.radians(curv_pert[idx])
                    if abs(ang) > 1e-10:
                        axis = _np_rodrigues(axis, shoot_bending_axis, ang)
            if yaw_pert:
                idx = seg - 1
                if idx < len(yaw_pert):
                    ang = math.radians(yaw_pert[idx])
                    if abs(ang) > 1e-10:
                        axis = _np_rodrigues(axis, np.array([0.0, 0.0, 1.0]), ang)
            vertices.append(vertices[-1] + dr * _np_normalize(axis))
            radii.append(phyt.internode_radius)

        phyt.internode_vertices = [v.copy() for v in vertices]
        phyt.internode_radii = [r for r in radii]

        sd.internode_vertices.append(np.array(vertices, dtype=np.float64))
        sd.internode_radii.append(np.array(radii, dtype=np.float64))

        phyt.internode_pos = vertices[0].copy()
        phyt.internode_tip = vertices[-1].copy()
        phyt.internode_dir = _np_normalize(axis)

        # --- petioles ---
        for pet_idx, pet in enumerate(phyt.petioles):
            petiole_axis = internode_axis.copy()
            pet_pitch = math.radians(pet.get("pitch", 5.0))
            if abs(pet_pitch) > 1e-10:
                petiole_axis = _np_rodrigues(petiole_axis, petiole_rotation_axis, abs(pet_pitch))

            phyllo_rad = math.radians(phyt.internode_phyllotactic_angle)
            if phyt_idx != 0 and abs(phyllo_rad) > 1e-10:
                petiole_axis = _np_rodrigues(petiole_axis, internode_axis, phyllo_rad)
                petiole_rotation_axis = _np_rodrigues(petiole_rotation_axis, internode_axis, phyllo_rad)

            n_petioles = len(phyt.petioles)
            pet_rot_axis = petiole_rotation_axis.copy()
            if pet_idx > 0 and n_petioles > 1:
                budrot = float(pet_idx) * 2.0 * math.pi / float(n_petioles)
                petiole_axis = _np_rodrigues(petiole_axis, internode_axis, budrot)
                pet_rot_axis = _np_rodrigues(pet_rot_axis, internode_axis, budrot)

            n_pet_seg = max(1, pet.get("length_segments", 5))
            pet_len = pet.get("length", 0.0)
            dr_pet = pet_len / float(n_pet_seg)
            curvature = pet.get("curvature", 0.0)

            pet_vertices = [phyt.internode_tip.copy()]
            pet_axis = _np_normalize(petiole_axis)
            for j in range(1, n_pet_seg + 1):
                if abs(curvature) > 1e-10:
                    ang = math.radians(curvature * dr_pet)
                    pet_axis = _np_rodrigues(pet_axis, pet_rot_axis, -ang)
                pet_vertices.append(pet_vertices[-1] + dr_pet * _np_normalize(pet_axis))

            pet["base_pos"] = pet_vertices[0].copy()
            pet["tip_pos"] = pet_vertices[-1].copy()
            pet["axis"] = _np_normalize(pet_axis)
            pet["vertices"] = pet_vertices

            # --- leaflets ---
            leaves = pet.get("leaves", [])
            leaves_per_petiole = len(leaves)
            leaflet_offset = pet.get("leaflet_offset", 0.0)
            petiole_tip_axis = _np_normalize(pet_axis)

            for leaf_idx, leaf in enumerate(leaves):
                ind_from_tip = float(leaf_idx) - float(leaves_per_petiole - 1) / 2.0

                # leaflet base along petiole
                if leaves_per_petiole > 1 and leaflet_offset > 0 and ind_from_tip != 0:
                    offset = (abs(ind_from_tip) - 0.5) * leaflet_offset * pet_len
                    frac = max(0.0, 1.0 - offset / max(pet_len, 1e-6))
                    leaf_base = _interpolate_tube(pet_vertices, frac)
                else:
                    leaf_base = pet_vertices[-1].copy()

                # compound rotation
                compound_rotation = 0.0
                if leaves_per_petiole > 1:
                    if leaflet_offset == 0:
                        dphi = math.pi / (math.floor(0.5 * (leaves_per_petiole - 1)) + 1)
                        compound_rotation = -math.pi + dphi * (leaf_idx + 0.5)
                    else:
                        if leaf_idx == (leaves_per_petiole - 1) / 2.0:
                            compound_rotation = 0.0
                        elif leaf_idx < (leaves_per_petiole - 1) / 2.0:
                            compound_rotation = -0.5 * math.pi
                        else:
                            compound_rotation = 0.5 * math.pi

                # Build local rotation matrix matching PlantArchitecture.cpp
                leaf_scale = leaf.get("scale", 0.0)
                leaf_pitch = math.radians(leaf.get("pitch", 0.0))
                leaf_yaw = math.radians(leaf.get("yaw", 0.0))
                leaf_roll = math.radians(leaf.get("roll", 0.0))

                # 1. roll about local x (matches C++ rotateObject "x")
                roll_rot = 0.0
                if leaves_per_petiole == 1:
                    sign = 1 if (phyt_idx % 2 == 0) else -1
                    roll_rot = -leaf_roll * sign
                elif ind_from_tip != 0:
                    sign = 1.0 if compound_rotation > 0 else -1.0
                    roll_rot = (math.asin(_as_safe(petiole_tip_axis[2])) + leaf_roll) * sign

                # 2. pitch about local y (matches C++ rotateObject "y")
                pitch_rot = leaf_pitch
                if ind_from_tip == 0:
                    pitch_rot += math.asin(_as_safe(petiole_tip_axis[2]))

                # 3. yaw about local z for lateral leaflets
                yaw_rot = 0.0
                if ind_from_tip != 0:
                    sign = -compound_rotation / abs(compound_rotation)
                    yaw_rot = sign * leaf_yaw

                # 4. rotate to petiole azimuth + compound rotation about world z
                azimuth_rot = -math.atan2(petiole_tip_axis[1], petiole_tip_axis[0]) + compound_rotation

                # Local prototype mesh
                proto_verts, proto_faces = _leaflet_local_mesh(leaf_scale, aspect=0.7)
                verts = proto_verts.copy()

                # Apply roll about local x
                if abs(roll_rot) > 1e-10:
                    verts = np.array([_np_rodrigues(v, np.array([1.0, 0.0, 0.0]), roll_rot) for v in verts])

                # Apply pitch about local y (negative to match C++ convention)
                if abs(pitch_rot) > 1e-10:
                    verts = np.array([_np_rodrigues(v, np.array([0.0, 1.0, 0.0]), -pitch_rot) for v in verts])

                # Apply yaw about local z
                if abs(yaw_rot) > 1e-10:
                    verts = np.array([_np_rodrigues(v, np.array([0.0, 0.0, 1.0]), yaw_rot) for v in verts])

                # Apply azimuth + compound rotation about world z
                if abs(azimuth_rot) > 1e-10:
                    verts = np.array([_np_rodrigues(v, np.array([0.0, 0.0, 1.0]), azimuth_rot) for v in verts])

                # Blade-up correction for single leaves (simplified)
                if leaves_per_petiole == 1:
                    r_h = math.sqrt(petiole_tip_axis[0] ** 2 + petiole_tip_axis[1] ** 2)
                    if r_h > 1e-4:
                        blade_correction = math.atan2(petiole_tip_axis[2] * r_h, r_h * r_h)
                        length_ratio = min(pet_len / max(leaf_scale, 1e-6), 1.0)
                        blade_correction *= length_ratio
                        blade_correction = _clamp(blade_correction, -0.5 * math.pi + math.radians(1.0), 0.5 * math.pi - math.radians(1.0))
                        sign = 1 if (phyt_idx % 2 == 0) else -1
                        if abs(blade_correction) > 1e-10:
                            verts = np.array([_np_rodrigues(v, petiole_tip_axis, blade_correction * sign) for v in verts])

                # Translate to leaf base
                verts = verts + leaf_base

                leaf["base_pos"] = leaf_base.copy()
                leaf["mesh_verts"] = verts
                leaf["mesh_faces"] = proto_faces

        # --- floral buds ---
        for fbud in pet.get("floral_buds", []):
            fbud["base_pos"] = pet_vertices[-1].copy()


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


def _leaflet_local_mesh_torch(device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a simple leaf prototype in its local frame as torch Tensors.

    Uses pure torch ops for vertex grid creation.  Scale is fixed to 1.0;
    callers should multiply by the desired leaf length.
    """
    L = 1.0
    aspect = 0.7
    W = L * aspect
    Nx = 8
    Ny = 6

    j = torch.arange(Ny + 1, dtype=torch.float32, device=device)
    i = torch.arange(Nx + 1, dtype=torch.float32, device=device)
    yy = j * (W / Ny) - 0.5 * W          # (Ny+1,)
    xx = i * (1.0 / Nx)                   # (Nx+1,)

    y_grid, x_grid = torch.meshgrid(yy, xx, indexing="ij")  # (Ny+1, Nx+1)

    taper = torch.sin(math.pi * x_grid)
    y_eff = y_grid * taper
    z_arch = (
        0.08 * L
        * torch.sin(math.pi * x_grid)
        * (1.0 - 4.0 * (y_grid / max(W, 1e-6)) ** 2)
    )

    verts = torch.stack([x_grid * L, y_eff, z_arch], dim=-1).reshape(-1, 3)

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


def nodes_to_geometry_torch(
    nodes: torch.Tensor,
    parent_indices: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Convert a batch of 15D organ-graph nodes to explicit Helios geometry (torch).

    All returned tensors preserve the node dimension ``N`` so the forward pass
    stays fully differentiable.  Non-matching organ types produce degenerate
    geometry that is masked out in the rasterizer.

    Returns:
        tube_verts:   (B, N, 2, 3)
        tube_radii:   (B, N, 2)
        tube_organs:  (B, N)
        leaf_verts:   (B, N, V, 3)   world-space
        leaf_faces:   (F, 3)
        leaf_organs:  (B, N)
        bud_centers:  (B, N, 3)
        bud_radii:    (B, N)
        bud_lengths:  (B, N)
        bud_organs:   (B, N)
    """
    B, N, _ = nodes.shape
    device = nodes.device

    positions = nodes[..., :3]
    lengths = nodes[..., 3].clamp(min=1e-6)
    radii = nodes[..., 4].clamp(min=1e-6)
    pitches = nodes[..., 5]
    yaws = nodes[..., 6]
    rolls = nodes[..., 7]
    organ_logits = nodes[..., 8:12]
    existence = nodes[..., 14]

    directions = _direction_from_angles(pitches, yaws)  # (B, N, 3)
    organ = organ_logits.argmax(dim=-1)                 # (B, N)

    # Existence mask: nodes with existence<=0.5 (e.g. dormant floral buds) are
    # not rendered. Applied consistently to tubes, leaves and buds below.
    exist_mask = (existence > 0.5).float()

    # ------------------------------------------------------------------
    # Tubes (internodes + petioles)
    # ------------------------------------------------------------------
    is_tube = (
        (organ == OrganNode3D.INTERNODE) | (organ == OrganNode3D.PETIOLE)
    ).float()
    tube_lengths = lengths * is_tube * exist_mask
    tips = positions + tube_lengths.unsqueeze(-1) * directions
    tube_verts = torch.stack([positions, tips], dim=2)          # (B, N, 2, 3)

    is_petiole = (organ == OrganNode3D.PETIOLE).float()
    r_tip = radii * (1.0 - 0.5 * is_petiole) * exist_mask
    tube_radii = torch.stack([radii * exist_mask, r_tip], dim=2)   # (B, N, 2)
    tube_organs = organ                                         # (B, N)

    # ------------------------------------------------------------------
    # Leaves
    # ------------------------------------------------------------------
    local_verts, leaf_faces = _leaflet_local_mesh_torch(device=device)
    V = local_verts.shape[0]

    is_leaf = (organ == OrganNode3D.LEAF).float()
    leaf_lengths = lengths * is_leaf * exist_mask                  # (B, N)

    local_scaled = local_verts.view(1, 1, V, 3) * leaf_lengths.view(B, N, 1, 1)

    x_axis = directions
    tmp = torch.where(
        x_axis[..., 2:3].abs() < 0.9,
        torch.tensor([0.0, 0.0, 1.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )
    y_axis = torch.cross(tmp.expand_as(x_axis), x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)

    # Basis change from the leaf-local frame (x=midrib, y=width, z=normal) to world.
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)             # (B, N, 3, 3) columns

    # The midrib direction already encodes pitch/yaw (direction = f(pitch, yaw)).
    # Only apply the leaf roll (twist about the midrib) to avoid double rotation.
    cr = torch.cos(rolls * math.pi / 180.0)
    sr = torch.sin(rolls * math.pi / 180.0)

    Rr = torch.zeros(B, N, 3, 3, device=device)
    Rr[..., 0, 0] = 1.0
    Rr[..., 1, 1] = cr
    Rr[..., 1, 2] = -sr
    Rr[..., 2, 1] = sr
    Rr[..., 2, 2] = cr

    R_total = R @ Rr                                               # (B, N, 3, 3)

    # world = R_total (B,N,3,3) @ local (B,N,V,3); einsum over last two axes.
    world_verts = torch.einsum("bnij,bnvj->bniv", R_total, local_scaled)  # (B,N,3,V)
    world_verts = torch.movedim(world_verts, -2, -1)               # (B, N, V, 3)
    world_verts = world_verts + positions.unsqueeze(2)

    # Collapse non-leaves and non-existent leaves to a single point so
    # triangles have ~zero area.
    leaf_verts = torch.where(
        (is_leaf * exist_mask).view(B, N, 1, 1) > 0,
        world_verts,
        positions.unsqueeze(2),
    )
    leaf_organs = organ  # Actual organ types (not all LEAF)

    # ------------------------------------------------------------------
    # Buds
    # ------------------------------------------------------------------
    is_bud = (organ == OrganNode3D.FLORAL_BUD).float()
    bud_centers = positions                                     # (B, N, 3)
    bud_radii = radii * is_bud * exist_mask                     # (B, N)
    bud_lengths = lengths * is_bud * exist_mask                 # (B, N)
    bud_organs = organ                                          # (B, N) actual organ types

    return (
        tube_verts, tube_radii, tube_organs,
        leaf_verts, leaf_faces, leaf_organs,
        bud_centers, bud_radii, bud_lengths, bud_organs,
    )


def nodes_to_geometry(
    nodes: torch.Tensor,
    parent_indices: Optional[torch.Tensor] = None,
) -> Tuple[List[List[HeliosTube]], List[List[HeliosLeaflet]], List[List[HeliosEllipsoid]]]:
    """Convert a batch of 15D organ-graph nodes to explicit Helios geometry.

    Returns per-batch lists of tubes, leaflets, and ellipsoids. The conversion is
    not fully differentiable (numpy output), so use this only for rendering.
    """
    nodes_np = nodes.detach().cpu().numpy()
    B, N, _ = nodes_np.shape

    organ_labels = nodes_np[..., 8:12].argmax(axis=-1)
    if parent_indices is not None:
        parents = parent_indices.detach().cpu().numpy()
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
        existence = nodes_np[b, :, 14]
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
            elif organ[i] == OrganNode3D.FLORAL_BUD:
                all_ellipsoids[b].append(HeliosEllipsoid(
                    center=base.astype(np.float32),
                    radius=float(r),
                    length=float(lengths[i]),
                    organ=OrganNode3D.FLORAL_BUD,
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
