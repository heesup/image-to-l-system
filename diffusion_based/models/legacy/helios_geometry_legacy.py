"""Exact Helios-style 3D plant geometry reconstruction from XML.

This module is the **Track A** (XML-native) implementation of the Helios
forward kinematics. It reconstructs explicit 3D geometry objects that match the
C++ PlantArchitecture output as closely as possible. The resulting geometry is
stored in the shared dataclasses defined in
``diffusion_based.models.helios_geometry`` so that both Track A and Track B
converge on the same rasterization path.

Responsibilities:
  - Parse Helios XML and run exact C++-style forward kinematics.
  - Build ``HeliosPlantGeometry`` (tubes, leaflets, ellipsoids).
  - Provide ``build_helios_geometry_from_xml`` for GT benchmarks and exact
    pixel-matching evaluation.
  - Provide ``DifferentiableHeliosXMLRenderer`` which converts the numpy
    geometry into batched PyTorch tensors and renders through
    ``HeliosGeometryRasterizer``.

All torch-aware node-array conversion (15D/19D/22D) lives in
``diffusion_based.models.helios_geometry`` and is re-exported here for backward
compatibility only.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_based.models.helios_geometry import (
    HeliosEllipsoid,
    HeliosLeaflet,
    HeliosPlantGeometry,
    HeliosTube,
    nodes_to_geometry as _nodes_to_geometry,
    nodes_to_geometry_torch as _nodes_to_geometry_torch,
    nodes_to_point_cloud as _nodes_to_point_cloud,
)
from diffusion_based.models.helios_xml_parser import (
    HeliosXMLParser,
    OrganNode3D,
    Phytomer3D,
    ShootData,
    _normalize as _np_normalize,
    _rotate_point_about_line as _np_rodrigues,
)
from diffusion_based.models.helios_rasterizer_3d import HeliosGeometryRasterizer


# Backward-compatible aliases for the shared geometry dataclasses.
HeliosTube = HeliosTube
HeliosLeaflet = HeliosLeaflet
HeliosEllipsoid = HeliosEllipsoid
HeliosPlantGeometry = HeliosPlantGeometry


# ═══════════════════════════════════════════════════════════════════════════════
# Local leaf prototype mesh (numpy, used by XML-native exact reconstruction)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Numpy exact geometry from XML / 15D nodes
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _as_safe(v: float) -> float:
    return _clamp(v, -1.0, 1.0)


def _leaflet_from_node(node: OrganNode3D) -> HeliosLeaflet:
    """Build a HeliosLeaflet mesh from a 15D/22D leaf node.

    The node stores position, direction, pitch/yaw/roll, and length/width. We
    orient the local prototype so the midrib aligns with the node's direction
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
    """Build explicit Helios geometry from a list of OrganNode3D objects."""
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
# Backward-compatible re-exports of the unified torch node-array pipeline
# ═══════════════════════════════════════════════════════════════════════════════

# For 15D/19D/22D node arrays, the canonical implementation now lives in
# diffusion_based.models.helios_geometry. Re-export here so legacy callers
# keep working without modification.

nodes_to_geometry = _nodes_to_geometry
nodes_to_geometry_torch = _nodes_to_geometry_torch
nodes_to_point_cloud = _nodes_to_point_cloud


# ═══════════════════════════════════════════════════════════════════════════════
# XML-native differentiable renderer
# ═══════════════════════════════════════════════════════════════════════════════

class DifferentiableHeliosXMLRenderer(nn.Module):
    """XML-native PyTorch Differentiable Renderer.

    Converts a ``HeliosPlantGeometry`` (built directly from XML) into batched
    PyTorch tensors and renders through ``HeliosGeometryRasterizer``. This is the
    Track A renderer: it uses the exact mesh already computed by
    ``build_helios_geometry_from_xml`` rather than re-deriving it from a node
    vector, guaranteeing pixel-level consistency with the C++ ground truth.
    """

    def __init__(self, rasterizer: HeliosGeometryRasterizer):
        super().__init__()
        self.rasterizer = rasterizer

    def forward(
        self,
        geom_xml: HeliosPlantGeometry,
        camera_height: float = 1.0,
        distance_from_center: float = 0.0,
        azimuth_deg: float = 270.0,
        hfov_deg: Optional[float] = None,
        target_center: Optional[torch.Tensor] = None,
        sun_dir: Optional[torch.Tensor] = None,
        focus_plant: bool = True,
        background: Union[str, torch.Tensor] = "black",
        leaf_sigma: Optional[float] = None,
    ) -> torch.Tensor:
        """Render a HeliosPlantGeometry object and return RGBA (B, 4, H, W)."""
        tensors = geom_xml.get_geometry_tensors(device=next(self.rasterizer.buffers()).device)
        return self.rasterizer.render_torch_geometry(
            *tensors[:10],
            camera_height=camera_height,
            distance_from_center=distance_from_center,
            azimuth_deg=azimuth_deg,
            hfov_deg=hfov_deg,
            target_center=target_center,
            sun_dir=sun_dir,
            focus_plant=focus_plant,
            background=background,
            leaf_sigma=leaf_sigma,
        )
