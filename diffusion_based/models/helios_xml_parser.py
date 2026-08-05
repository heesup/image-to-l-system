"""Helios XML Parser: Extract hierarchical 3D plant structure with accurate forward kinematics.

Parses Helios plant XML into differentiable 3D geometry, faithfully mirroring
the C++ PlantArchitecture reconstruction logic from InputOutput.cpp.

Hierarchy: plant → shoot → phytomer → internode → petiole → leaf / floral_bud

Key fixes:
- Multi-shoot hierarchy support (parent_shoot_ID, parent_node_index)
- Accurate internode forward kinematics (pitch, phyllotactic_angle, base_rotation, curvature/yaw perturbations)
- Accurate petiole geometry (pitch, curvature, taper, multi-petiole per internode)
- Trifoliate compound leaf accurate rotation (side leaflets rotated ±90° via compound_rotation)
- Floral bud filtering: dormant/unexpanded buds marked existence = 0.0 to prevent unrendered peduncle artifacts
"""

import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
import copy


def _rotate_point_about_line(point: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of *point* about *axis* through the origin by *angle_rad*."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return point.copy()
    axis = axis / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return point * c + np.cross(axis, point) * s + axis * np.dot(axis, point) * (1 - c)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


class OrganNode3D:
    """A single organ node in 3D space (15D representation compatible)."""

    INTERNODE = 0
    PETIOLE = 1
    LEAF = 2
    FLORAL_BUD = 3

    def __init__(self, organ_type: int):
        self.organ_type = organ_type
        self.position = np.zeros(3)       # 3D base/center position
        self.length = 0.0                 # length or scale
        self.radius = 0.0                 # radius or width
        self.pitch = 0.0                  # degrees
        self.yaw = 0.0                    # degrees
        self.roll = 0.0                   # degrees
        self.shoot_id = 0                 # shoot hierarchy ID
        self.phytomer_idx = 0             # phytomer index along shoot
        self.existence = 1.0              # confidence [0, 1]
        self.flower_head_radius = 0.0     # visible flower/fruit head radius (m)

        self.tip_position = np.zeros(3)   # tip position
        self.direction = np.array([0.0, 0.0, 1.0])
        self.parent_idx = -1              # global parent index in node list

    def to_15d(self) -> np.ndarray:
        """Convert to 15D feature vector (no flower-head radius)."""
        v = self.to_16d()
        return v[:15]

    def to_16d(self) -> np.ndarray:
        """Convert to 16D feature vector.

        Layout: [xyz(3), length, radius, pitch, yaw, roll, organ_onehot(4),
                 shoot_id, phytomer_idx, existence, flower_head_radius].
        Channel 15 (flower_head_radius) is the radius of the visible flower /
        fruit head at the tip of a floral bud. It is 0 for non-floral organs.
        """
        one_hot = np.zeros(4)
        one_hot[self.organ_type] = 1.0
        head_r = self.flower_head_radius if self.organ_type == OrganNode3D.FLORAL_BUD else 0.0
        return np.array([
            self.position[0], self.position[1], self.position[2],
            self.length, self.radius,
            self.pitch, self.yaw, self.roll,
            one_hot[0], one_hot[1], one_hot[2], one_hot[3],
            float(self.shoot_id), float(self.phytomer_idx),
            self.existence, float(head_r),
        ])


class Phytomer3D:
    """Single phytomer (node) in 3D space with full geometry."""
    def __init__(self):
        self.internode_pos = np.array([0.0, 0.0, 0.0])
        self.internode_tip = np.array([0.0, 0.0, 0.0])
        self.internode_dir = np.array([0.0, 0.0, 1.0])
        self.internode_length = 0.0
        self.internode_radius = 0.0
        self.internode_pitch = 0.0
        self.internode_phyllotactic_angle = 0.0

        self.internode_vertices: List[np.ndarray] = []
        self.internode_radii: List[float] = []

        self.petioles: List[Dict] = []
        self.leaves: List[Dict] = []
        self.floral_buds: List[Dict] = []

        self.shoot_id = 0
        self.phytomer_index = 0

    def get_organ_nodes(self) -> List[OrganNode3D]:
        """Convert this phytomer into a list of OrganNode3D for 15D representation."""
        nodes = []

        # 1. Internode node
        inode = OrganNode3D(OrganNode3D.INTERNODE)
        inode.position = self.internode_pos.copy()
        inode.tip_position = self.internode_tip.copy()
        inode.direction = self.internode_dir.copy()
        inode.length = self.internode_length
        inode.radius = self.internode_radius
        inode.pitch = self.internode_pitch
        inode.yaw = self.internode_phyllotactic_angle
        inode.shoot_id = self.shoot_id
        inode.phytomer_idx = self.phytomer_index
        nodes.append(inode)

        # 2. Petiole nodes
        for j, pet in enumerate(self.petioles):
            pnode = OrganNode3D(OrganNode3D.PETIOLE)
            pnode.position = pet.get('base_pos', self.internode_tip).copy()
            pnode.tip_position = pet.get('tip_pos', self.internode_tip).copy()
            pnode.direction = pet.get('axis', np.array([0, 0, 1])).copy()
            pnode.length = pet.get('length', 0.0)
            pnode.radius = pet.get('radius', 0.0)
            pnode.pitch = pet.get('pitch', 0.0)
            pnode.shoot_id = self.shoot_id
            pnode.phytomer_idx = self.phytomer_index
            nodes.append(pnode)

            # 3. Leaf nodes attached to this petiole
            for leaf in pet.get('leaves', []):
                lnode = OrganNode3D(OrganNode3D.LEAF)
                lnode.position = leaf.get('base_pos', pnode.tip_position).copy()
                lnode.tip_position = leaf.get('tip_pos', lnode.position).copy()
                lnode.direction = leaf.get('direction', pnode.direction).copy()
                lnode.length = leaf.get('scale', 0.0)
                lnode.radius = leaf.get('scale', 0.0) * 0.70  # width aspect ratio ~0.70
                lnode.pitch = leaf.get('pitch', 0.0)
                lnode.yaw = leaf.get('yaw', 0.0)
                lnode.roll = leaf.get('roll', 0.0)
                lnode.shoot_id = self.shoot_id
                lnode.phytomer_idx = self.phytomer_index
                nodes.append(lnode)

            # 4. Floral bud nodes (Filter out dormant / unexpanded peduncles)
            for fbud in pet.get('floral_buds', []):
                fnode = OrganNode3D(OrganNode3D.FLORAL_BUD)
                fnode.position = fbud.get('base_pos', pnode.tip_position).copy()
                fnode.length = fbud.get('peduncle_length', 0.0)
                fnode.radius = fbud.get('peduncle_radius', 0.0)
                fnode.pitch = fbud.get('peduncle_pitch', 0.0)
                fnode.shoot_id = self.shoot_id
                fnode.phytomer_idx = self.phytomer_index

                # In Helios C++, buds only render geometry if bud_state is FLOWER_OPEN or FLOWER_CLOSED
                # bud_state: 0=dormant, 1=dead, 2=active, 3=flower_closed, 4=flower_open, 5=fruit_developing
                state = fbud.get('bud_state', 0)
                # In early DAP (short internodes), floral buds with unexpanded peduncle (length > 0.1m) are not rendered
                if state in [3, 4] or (state >= 5 and fnode.length < 0.05):
                    fnode.existence = 1.0
                    # Visible flower/fruit head sits at the peduncle tip. C++ renders
                    # an inflorescence prototype scaled by flower_prototype_scale
                    # (e.g. 0.03 m for cowpea / bean). We surface that radius as a
                    # 16D channel so the differentiable renderer can draw it.
                    fnode.flower_head_radius = 0.03
                else:
                    fnode.existence = 0.0  # Hide unrendered dormant floral bud

                nodes.append(fnode)

        return nodes


class ShootData:
    """Parsed data for a single shoot."""
    def __init__(self):
        self.shoot_id = 0
        self.shoot_type_label = ""
        self.parent_shoot_id = -1
        self.parent_node_index = 0
        self.parent_petiole_index = 0
        self.base_rotation = np.zeros(3)
        self.phytomers: List[Phytomer3D] = []

        self.internode_vertices: List[List[np.ndarray]] = []
        self.internode_radii: List[List[float]] = []


class HeliosXMLParser:
    """Parse Helios plant XML into 3D hierarchical structure."""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.shoots: Dict[int, ShootData] = {}
        self.phytomers: List[Phytomer3D] = []
        self.base_position = np.zeros(3)
        self.plant_age = 0.0
        self.is_dap1 = self._extract_dap(xml_path) == 1

    @staticmethod
    def _extract_dap(xml_path: str) -> int:
        """Extract DAP number from a filename like ..._dap010_... or ..._dap10_...."""
        lower = xml_path.lower()
        # Find 'dap' followed by digits
        for i in range(len(lower)):
            if lower[i:i+3] == "dap" and i + 3 < len(lower) and lower[i+3].isdigit():
                j = i + 3
                while j < len(lower) and lower[j].isdigit():
                    j += 1
                try:
                    return int(lower[i+3:j])
                except ValueError:
                    pass
        return 0

    def parse(self) -> List[Phytomer3D]:
        """Parse XML and return list of all phytomers in 3D with shoot hierarchy."""
        self.shoots = {}
        self.phytomers = []

        plant_elem = self.root.find(".//plant_instance")
        if plant_elem is None:
            return []

        bp_text = plant_elem.findtext("base_position", "0 0 0")
        self.base_position = self._parse_vec3(bp_text)

        age_text = plant_elem.findtext("plant_age", "0")
        self.plant_age = float(age_text.strip())

        shoot_elems = plant_elem.findall("shoot")
        shoot_data_list = []

        for shoot_elem in shoot_elems:
            sd = self._parse_shoot_element(shoot_elem)
            shoot_data_list.append(sd)

        shoot_data_list.sort(key=lambda s: s.shoot_id)

        for sd in shoot_data_list:
            self._reconstruct_shoot_geometry(sd)
            self.shoots[sd.shoot_id] = sd
            self.phytomers.extend(sd.phytomers)

        return self.phytomers

    def get_all_organ_nodes(self) -> List[OrganNode3D]:
        """Get all organ nodes (15D compatible) from parsed phytomers."""
        if not self.phytomers:
            self.parse()

        all_nodes = []
        internode_global_idx = {}

        for phyt in self.phytomers:
            organ_nodes = phyt.get_organ_nodes()

            if organ_nodes:
                internode_global_idx[(phyt.shoot_id, phyt.phytomer_index)] = len(all_nodes)

            for i, node in enumerate(organ_nodes):
                if node.organ_type == OrganNode3D.INTERNODE:
                    if phyt.phytomer_index > 0:
                        prev_key = (phyt.shoot_id, phyt.phytomer_index - 1)
                        node.parent_idx = internode_global_idx.get(prev_key, -1)
                    else:
                        shoot_data = self.shoots.get(phyt.shoot_id)
                        if shoot_data and shoot_data.parent_shoot_id >= 0:
                            parent_key = (shoot_data.parent_shoot_id, shoot_data.parent_node_index)
                            node.parent_idx = internode_global_idx.get(parent_key, -1)
                        else:
                            node.parent_idx = len(all_nodes)
                elif node.organ_type == OrganNode3D.PETIOLE:
                    node.parent_idx = internode_global_idx.get((phyt.shoot_id, phyt.phytomer_index), -1)
                elif node.organ_type in [OrganNode3D.LEAF, OrganNode3D.FLORAL_BUD]:
                    for k in range(len(all_nodes) - 1, -1, -1):
                        if all_nodes[k].organ_type == OrganNode3D.PETIOLE:
                            node.parent_idx = k
                            break

                # DAP 1 Override: Hide un-emerged secondary shoots (Shoot 1+) on Day 1 (DAP 1)
                if self.is_dap1 and node.shoot_id > 0:
                    node.existence = 0.0

                all_nodes.append(node)

        if all_nodes and all_nodes[0].parent_idx < 0:
            all_nodes[0].parent_idx = 0

        return all_nodes

    def _parse_shoot_element(self, shoot_elem: ET.Element) -> ShootData:
        sd = ShootData()
        sd.shoot_id = int(shoot_elem.get("ID", 0))
        sd.shoot_type_label = shoot_elem.findtext("shoot_type_label", "").strip()
        sd.parent_shoot_id = int(shoot_elem.findtext("parent_shoot_ID", "-1").strip())
        sd.parent_node_index = int(shoot_elem.findtext("parent_node_index", "0").strip())
        sd.parent_petiole_index = int(shoot_elem.findtext("parent_petiole_index", "0").strip())

        br_text = shoot_elem.findtext("base_rotation", "0 0 0")
        sd.base_rotation = self._parse_vec3(br_text)

        for phyt_idx, phyt_elem in enumerate(shoot_elem.findall("phytomer")):
            p3d = self._parse_phytomer_element(phyt_elem, sd.shoot_id, phyt_idx)
            sd.phytomers.append(p3d)

        return sd

    def _parse_geometry(self, elem: ET.Element, default_pos: np.ndarray) -> dict:
        """Parse explicit <geometry> child if present."""
        geom = {}
        geom_elem = elem.find("geometry")
        if geom_elem is not None:
            pos = geom_elem.findtext("position")
            if pos is not None:
                geom['position'] = self._parse_vec3(pos)
            tip = geom_elem.findtext("tip_position")
            if tip is not None:
                geom['tip_position'] = self._parse_vec3(tip)
            direc = geom_elem.findtext("direction")
            if direc is not None:
                geom['direction'] = self._parse_vec3(direc)
        if 'position' not in geom:
            geom['position'] = default_pos.copy()
        if 'tip_position' not in geom:
            geom['tip_position'] = default_pos.copy()
        if 'direction' not in geom:
            geom['direction'] = np.array([0.0, 0.0, 1.0])
        return geom

    def _parse_phytomer_element(self, phyt_elem: ET.Element, shoot_id: int, phyt_idx: int) -> Phytomer3D:
        p3d = Phytomer3D()
        p3d.shoot_id = shoot_id
        p3d.phytomer_index = phyt_idx

        internode = phyt_elem.find("internode")
        if internode is None:
            return p3d

        int_geom = self._parse_geometry(internode, self.base_position)
        p3d.internode_pos = int_geom['position']
        p3d.internode_tip = int_geom['tip_position']
        p3d.internode_dir = int_geom['direction']
        # Only store explicit vertices when the XML actually provided a <geometry> block.
        # Otherwise leave them empty so parameter-based forward kinematics runs for
        # legacy Helios XML files that don't contain explicit positions.
        has_explicit_int_geom = internode.find("geometry") is not None
        if has_explicit_int_geom:
            p3d.internode_vertices = [p3d.internode_pos.copy(), p3d.internode_tip.copy()]
            p3d.internode_radii = [p3d.internode_radius, p3d.internode_radius]

        p3d.internode_length = float(internode.findtext("internode_length", "0"))
        p3d.internode_radius = float(internode.findtext("internode_radius", "0"))
        p3d.internode_pitch = float(internode.findtext("internode_pitch", "0"))
        p3d.internode_phyllotactic_angle = float(internode.findtext("internode_phyllotactic_angle", "0"))

        p3d._internode_length_max = float(internode.findtext("internode_length_max", str(p3d.internode_length)))
        p3d._internode_length_segments = int(internode.findtext("internode_length_segments", "1"))

        p3d._curvature_perturbations = self._parse_semicolon_floats(
            internode.findtext("curvature_perturbations", ""))
        p3d._yaw_perturbations = self._parse_semicolon_floats(
            internode.findtext("yaw_perturbations", ""))

        for petiole_elem in internode.findall("petiole"):
            pet_geom = self._parse_geometry(petiole_elem, p3d.internode_tip)
            petiole_data = {
                'base_pos': pet_geom['position'],
                'tip_pos': pet_geom['tip_position'],
                'axis': pet_geom['direction'],
                'length': float(petiole_elem.findtext("petiole_length", "0")),
                'radius': float(petiole_elem.findtext("petiole_radius", "0")),
                'pitch': float(petiole_elem.findtext("petiole_pitch", "0")),
                'curvature': float(petiole_elem.findtext("petiole_curvature", "0")),
                'taper': float(petiole_elem.findtext("petiole_taper", "0.25")),
                'length_segments': int(petiole_elem.findtext("petiole_length_segments", "5")),
                'scale_factor': float(petiole_elem.findtext("current_leaf_scale_factor", "1")),
                'leaflet_scale': float(petiole_elem.findtext("leaflet_scale", "1")),
                'leaflet_offset': float(petiole_elem.findtext("leaflet_offset", "0")),
                'leaves': [],
                'floral_buds': [],
            }

            for leaf_elem in petiole_elem.findall("leaf"):
                leaf_geom = self._parse_geometry(leaf_elem, pet_geom['tip_position'])
                leaf_data = {
                    'base_pos': leaf_geom['position'],
                    'tip_pos': leaf_geom['tip_position'],
                    'direction': leaf_geom['direction'],
                    'scale': float(leaf_elem.findtext("leaf_scale", "0")),
                    'pitch': float(leaf_elem.findtext("leaf_pitch", "0")),
                    'yaw': float(leaf_elem.findtext("leaf_yaw", "0")),
                    'roll': float(leaf_elem.findtext("leaf_roll", "0")),
                    'scale_factor': petiole_data['scale_factor'],
                }
                petiole_data['leaves'].append(leaf_data)

            for fbud_elem in petiole_elem.findall("floral_bud"):
                bud_geom = self._parse_geometry(fbud_elem, pet_geom['tip_position'])
                fbud_data = {
                    'base_pos': bud_geom['position'],
                    'tip_pos': bud_geom['tip_position'],
                    'direction': bud_geom['direction'],
                    'bud_state': int(fbud_elem.findtext("bud_state", "0")),
                    'parent_index': int(fbud_elem.findtext("parent_index", "0")),
                    'bud_index': int(fbud_elem.findtext("bud_index", "0")),
                    'is_terminal': int(fbud_elem.findtext("is_terminal", "0")),
                    'fruit_scale_factor': float(fbud_elem.findtext("current_fruit_scale_factor", "1")),
                }

                peduncle_elem = fbud_elem.find("peduncle")
                if peduncle_elem is not None:
                    fbud_data['peduncle_length'] = float(peduncle_elem.findtext("length", "0"))
                    fbud_data['peduncle_radius'] = float(peduncle_elem.findtext("radius", "0"))
                    fbud_data['peduncle_pitch'] = float(peduncle_elem.findtext("pitch", "0"))
                    fbud_data['peduncle_curvature'] = float(peduncle_elem.findtext("curvature", "0"))
                    fbud_data['peduncle_roll'] = float(peduncle_elem.findtext("roll", "0"))

                petiole_data['floral_buds'].append(fbud_data)

            p3d.petioles.append(petiole_data)

        return p3d

    def _reconstruct_shoot_geometry(self, sd: ShootData):
        sd.internode_vertices = []
        sd.internode_radii = []

        for phyt_idx, phyt in enumerate(sd.phytomers):
            # If the XML provided explicit <geometry> blocks, trust them and skip
            # parameter-based forward kinematics so that round-trips stay exact.
            has_explicit_geometry = phyt.internode_vertices is not None and len(phyt.internode_vertices) > 0
            if has_explicit_geometry:
                vertices = [v.copy() for v in phyt.internode_vertices]
                radii = list(phyt.internode_radii) if phyt.internode_radii else [phyt.internode_radius, phyt.internode_radius]
                sd.internode_vertices.append(vertices)
                sd.internode_radii.append(radii)
                self._reconstruct_petiole_geometry(sd, phyt_idx, phyt, use_explicit=True)
                continue

            internode_base = self._get_internode_base(sd, phyt_idx)
            internode_axis, petiole_rotation_axis, shoot_bending_axis = \
                self._compute_internode_orientation(sd, phyt_idx)

            n_segments = max(1, phyt._internode_length_segments)
            dr = phyt.internode_length / float(n_segments)
            dr_max = phyt._internode_length_max / float(n_segments) if phyt._internode_length_max > 0 else dr

            vertices = [internode_base.copy()]
            radii = [phyt.internode_radius]

            axis = internode_axis.copy()
            for seg in range(1, n_segments + 1):
                if phyt_idx > 0 and phyt._curvature_perturbations:
                    pert_idx = seg - 1
                    if pert_idx < len(phyt._curvature_perturbations):
                        curvature_angle = math.radians(phyt._curvature_perturbations[pert_idx])
                        if abs(curvature_angle) > 1e-10:
                            axis = _rotate_point_about_line(axis, shoot_bending_axis, curvature_angle)

                if phyt_idx > 0 and phyt._yaw_perturbations:
                    pert_idx = seg - 1
                    if pert_idx < len(phyt._yaw_perturbations):
                        yaw_angle = math.radians(phyt._yaw_perturbations[pert_idx])
                        if abs(yaw_angle) > 1e-10:
                            axis = _rotate_point_about_line(axis, np.array([0.0, 0.0, 1.0]), yaw_angle)

                next_vertex = vertices[-1] + dr * _normalize(axis)
                vertices.append(next_vertex)
                radii.append(phyt.internode_radius)

            sd.internode_vertices.append(vertices)
            sd.internode_radii.append(radii)

            phyt.internode_pos = vertices[0].copy()
            phyt.internode_tip = vertices[-1].copy()
            phyt.internode_dir = _normalize(axis)
            phyt.internode_vertices = vertices
            phyt.internode_radii = radii

            self._reconstruct_petiole_geometry(sd, phyt_idx, phyt)

    def _get_internode_base(self, sd: ShootData, phyt_idx: int) -> np.ndarray:
        if phyt_idx == 0:
            if sd.parent_shoot_id < 0:
                return self.base_position.copy()
            else:
                parent_sd = self.shoots.get(sd.parent_shoot_id)
                if parent_sd and sd.parent_node_index < len(parent_sd.internode_vertices):
                    return parent_sd.internode_vertices[sd.parent_node_index][-1].copy()
                return self.base_position.copy()
        else:
            return sd.internode_vertices[phyt_idx - 1][-1].copy()

    def _compute_internode_orientation(self, sd: ShootData, phyt_idx: int
                                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        phyt = sd.phytomers[phyt_idx]
        pitch_rad = math.radians(phyt.internode_pitch)
        phyllo_rad = math.radians(phyt.internode_phyllotactic_angle)

        parent_internode_axis = np.array([0.0, 0.0, 1.0])
        parent_petiole_axis = np.array([0.0, -1.0, 0.0])

        if phyt_idx > 0:
            prev = sd.phytomers[phyt_idx - 1]
            parent_internode_axis = _normalize(prev.internode_dir)
            if prev.petioles and 'axis' in prev.petioles[0]:
                parent_petiole_axis = _normalize(prev.petioles[0]['axis'])
            else:
                parent_petiole_axis = self._get_perpendicular(parent_internode_axis)
        elif sd.parent_shoot_id >= 0:
            parent_sd = self.shoots.get(sd.parent_shoot_id)
            if parent_sd and sd.parent_node_index < len(parent_sd.phytomers):
                parent_phyt = parent_sd.phytomers[sd.parent_node_index]
                parent_internode_axis = _normalize(parent_phyt.internode_dir)
                pet_idx = min(sd.parent_petiole_index, len(parent_phyt.petioles) - 1) \
                    if parent_phyt.petioles else 0
                if parent_phyt.petioles and 'axis' in parent_phyt.petioles[pet_idx]:
                    parent_petiole_axis = _normalize(parent_phyt.petioles[pet_idx]['axis'])
                else:
                    parent_petiole_axis = self._get_perpendicular(parent_internode_axis)

        petiole_rotation_axis = np.cross(parent_internode_axis, parent_petiole_axis)
        if np.linalg.norm(petiole_rotation_axis) < 1e-6:
            petiole_rotation_axis = np.array([1.0, 0.0, 0.0])
        else:
            petiole_rotation_axis = _normalize(petiole_rotation_axis)

        internode_axis = parent_internode_axis.copy()

        if phyt_idx == 0:
            if abs(pitch_rad) > 1e-10:
                internode_axis = _rotate_point_about_line(
                    internode_axis, petiole_rotation_axis, 0.5 * pitch_rad)

            base_rot_pitch = math.radians(sd.base_rotation[0])
            base_rot_yaw = math.radians(sd.base_rotation[1])
            base_rot_roll = math.radians(sd.base_rotation[2])

            if abs(base_rot_roll) > 1e-10:
                petiole_rotation_axis = _rotate_point_about_line(
                    petiole_rotation_axis, parent_internode_axis, base_rot_roll)
                internode_axis = _rotate_point_about_line(
                    internode_axis, parent_internode_axis, base_rot_roll)

            if abs(base_rot_pitch) > 1e-10:
                base_pitch_axis = -np.cross(parent_internode_axis, parent_petiole_axis)
                if np.linalg.norm(base_pitch_axis) > 1e-10:
                    base_pitch_axis = _normalize(base_pitch_axis)
                    petiole_rotation_axis = _rotate_point_about_line(
                        petiole_rotation_axis, base_pitch_axis, -base_rot_pitch)
                    internode_axis = _rotate_point_about_line(
                        internode_axis, base_pitch_axis, -base_rot_pitch)

            if abs(base_rot_yaw) > 1e-10:
                petiole_rotation_axis = _rotate_point_about_line(
                    petiole_rotation_axis, parent_internode_axis, base_rot_yaw)
                internode_axis = _rotate_point_about_line(
                    internode_axis, parent_internode_axis, base_rot_yaw)
        else:
            if abs(pitch_rad) > 1e-10:
                internode_axis = _rotate_point_about_line(
                    internode_axis, petiole_rotation_axis, -1.25 * pitch_rad)

        shoot_bending_axis = np.cross(internode_axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(shoot_bending_axis) < 1e-6:
            shoot_bending_axis = np.array([0.0, 1.0, 0.0])
        else:
            shoot_bending_axis = _normalize(shoot_bending_axis)

        internode_axis = _normalize(internode_axis)

        return internode_axis, petiole_rotation_axis, shoot_bending_axis

    def _reconstruct_petiole_geometry(self, sd: ShootData, phyt_idx: int, phyt: Phytomer3D, use_explicit: bool = False):
        internode_axis, petiole_rotation_axis, _ = \
            self._compute_internode_orientation(sd, phyt_idx)

        internode_tip = phyt.internode_tip.copy()

        for pet_idx, petiole in enumerate(phyt.petioles):
            # If explicit geometry was written into the XML, trust the stored
            # positions/directions and only rebuild the vertices buffer for rendering.
            if use_explicit and 'base_pos' in petiole:
                petiole['vertices'] = [petiole['base_pos'].copy(), petiole['tip_pos'].copy()]
                for leaf in petiole.get('leaves', []):
                    if 'base_pos' in leaf:
                        leaf['direction'] = _normalize(leaf['tip_pos'] - leaf['base_pos'])
                for fbud in petiole.get('floral_buds', []):
                    if 'base_pos' not in fbud:
                        fbud['base_pos'] = petiole['tip_pos'].copy()
                continue

            petiole_base = internode_tip.copy()
            petiole_pitch_rad = math.radians(petiole['pitch'])
            petiole_curvature = petiole['curvature']
            petiole_length = petiole['length']
            n_segments = max(1, petiole.get('length_segments', 5))
            phyllo_rad = math.radians(phyt.internode_phyllotactic_angle)

            petiole_axis = internode_axis.copy()

            if abs(petiole_pitch_rad) > 1e-10:
                petiole_axis = _rotate_point_about_line(
                    petiole_axis, petiole_rotation_axis, abs(petiole_pitch_rad))

            pet_rot_axis = petiole_rotation_axis.copy()
            if phyt_idx != 0 and abs(phyllo_rad) > 1e-10:
                petiole_axis = _rotate_point_about_line(
                    petiole_axis, internode_axis, phyllo_rad)
                pet_rot_axis = _rotate_point_about_line(
                    pet_rot_axis, internode_axis, phyllo_rad)

            n_petioles = len(phyt.petioles)
            if pet_idx > 0 and n_petioles > 1:
                budrot = float(pet_idx) * 2.0 * math.pi / float(n_petioles)
                petiole_axis = _rotate_point_about_line(
                    petiole_axis, internode_axis, budrot)
                pet_rot_axis = _rotate_point_about_line(
                    pet_rot_axis, internode_axis, budrot)

            dr_petiole = petiole_length / float(n_segments)
            dr_petiole_max = petiole_length / float(n_segments)
            pet_vertices = [petiole_base.copy()]
            pet_axis = _normalize(petiole_axis)

            for j in range(1, n_segments + 1):
                if abs(petiole_curvature) > 1e-10:
                    curvature_rad = math.radians(petiole_curvature * dr_petiole_max)
                    pet_axis = _rotate_point_about_line(
                        pet_axis, pet_rot_axis, -curvature_rad)

                next_pos = pet_vertices[-1] + dr_petiole * _normalize(pet_axis)
                pet_vertices.append(next_pos)

            petiole['base_pos'] = petiole_base.copy()
            petiole['tip_pos'] = pet_vertices[-1].copy()
            petiole['axis'] = _normalize(pet_axis)
            petiole['vertices'] = pet_vertices

            # ── Trifoliate Compound Leaf Rotation (PlantArchitecture.cpp lines 2087-2101) ──
            leaves_per_petiole = len(petiole['leaves'])
            petiole_tip_axis = _normalize(pet_axis)
            leaflet_offset = petiole.get('leaflet_offset', 0.0)

            for leaf_idx, leaf in enumerate(petiole['leaves']):
                ind_from_tip = float(leaf_idx) - float(leaves_per_petiole - 1) / 2.0

                if leaves_per_petiole > 1 and leaflet_offset > 0 and ind_from_tip != 0:
                    offset = (abs(ind_from_tip) - 0.5) * leaflet_offset * petiole_length
                    frac = max(0.0, 1.0 - offset / max(petiole_length, 1e-6))
                    leaf_base = self._interpolate_tube(pet_vertices, frac)
                else:
                    leaf_base = pet_vertices[-1].copy()

                # Compound rotation matching C++ line 2087-2101
                compound_rotation = 0.0
                if leaves_per_petiole > 1:
                    if leaflet_offset == 0:
                        dphi = math.pi / (math.floor(0.5 * (leaves_per_petiole - 1)) + 1)
                        compound_rotation = -math.pi + dphi * (leaf_idx + 0.5)
                    else:
                        if leaf_idx == (leaves_per_petiole - 1) / 2.0:
                            compound_rotation = 0.0  # Tip leaflet: 0°
                        elif leaf_idx < (leaves_per_petiole - 1) / 2.0:
                            compound_rotation = -0.5 * math.pi  # Left leaflet: -90°
                        else:
                            compound_rotation = 0.5 * math.pi   # Right leaflet: +90°

                # Leaf direction vector
                leaf_direction = _normalize(petiole_tip_axis)
                if abs(compound_rotation) > 1e-10:
                    # Rotate leaflet by ±90° about internode_axis (C++ line 2143)
                    leaf_direction = _rotate_point_about_line(
                        leaf_direction, internode_axis, compound_rotation)

                # Combine with leaf yaw angle from XML
                leaf_yaw_rad = math.radians(leaf.get('yaw', 0.0))
                if abs(leaf_yaw_rad) > 1e-10:
                    leaf_direction = _rotate_point_about_line(
                        leaf_direction, petiole_tip_axis, leaf_yaw_rad)

                leaf_scale = leaf['scale']
                leaf_tip = leaf_base + leaf_scale * leaf_direction

                leaf['base_pos'] = leaf_base.copy()
                leaf['tip_pos'] = leaf_tip.copy()
                leaf['direction'] = leaf_direction.copy()
                # Store total yaw in degrees for renderer
                leaf['yaw'] = math.degrees(math.atan2(leaf_direction[1], leaf_direction[0]))
                leaf['pitch'] = math.degrees(math.asin(clamp_val(leaf_direction[2], -1.0, 1.0)))

            for fbud in petiole.get('floral_buds', []):
                fbud['base_pos'] = pet_vertices[-1].copy()

    def _parse_vec3(self, text: str) -> np.ndarray:
        parts = text.strip().split()
        if len(parts) >= 3:
            return np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.zeros(3)

    def _parse_semicolon_floats(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []
        return [float(x) for x in text.strip().split(";") if x.strip()]

    def _get_perpendicular(self, v: np.ndarray) -> np.ndarray:
        if abs(v[0]) < 0.9:
            perp = np.cross(v, np.array([1, 0, 0]))
        else:
            perp = np.cross(v, np.array([0, 1, 0]))
        return _normalize(perp)

    @staticmethod
    def _interpolate_tube(vertices: List[np.ndarray], frac: float) -> np.ndarray:
        if not vertices or frac <= 0:
            return vertices[0].copy() if vertices else np.zeros(3)
        if frac >= 1.0:
            return vertices[-1].copy()
        n = len(vertices) - 1
        pos = frac * n
        idx = int(pos)
        t = pos - idx
        if idx >= n:
            return vertices[-1].copy()
        return (1.0 - t) * vertices[idx] + t * vertices[idx + 1]


def clamp_val(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))
