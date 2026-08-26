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
from typing import List, Dict, Tuple, Optional, Any
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


def _rodrigues_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for Rodrigues rotation about *axis*."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)],
    ], dtype=np.float64)


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
    FLORAL_BUD = 3   # dormant / active bud (not yet open)
    FLOWER = 4       # open or closed flower (bud_state 3 or 4)
    POD = 5          # developing fruit / pod (bud_state >= 5)
    NUM_ORGAN_TYPES = 6

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
        self.petiole_idx = 0              # petiole index within phytomer (for leaves/buds)
        self._xml_params: Dict[str, str] = {}  # raw XML text values for lossless round-trip

    def to_vec(self) -> np.ndarray:
        """Serialize this node to a fixed-length feature vector (25D).

        Layout:
          [0:3]   xyz          - base position (m)
          [3]     length       - organ length (m)
          [4]     radius       - organ radius (m)
          [5:14]  R_flat       - 3x3 orientation matrix (row-major), local frame
                                 to world. For non-leaf organs the first column
                                 is the direction vector and the remaining columns
                                 are zero-padded.
          [14:20] organ_onehot - 6-channel one-hot (INTERNODE, PETIOLE, LEAF,
                                  FLORAL_BUD, FLOWER, POD)
          [20]    shoot_id
          [21]    phytomer_idx
          [22]    existence    - confidence [0, 1]
          [23]    head_radius  - flower/pod head radius (m); 0 for non-floral
          [24]    parent_idx   - global parent node index (-1 = root)
        """
        one_hot = np.zeros(OrganNode3D.NUM_ORGAN_TYPES)  # 6D
        ot = min(self.organ_type, OrganNode3D.NUM_ORGAN_TYPES - 1)
        one_hot[ot] = 1.0
        head_r = self.flower_head_radius if self.organ_type in (
            OrganNode3D.FLORAL_BUD, OrganNode3D.FLOWER, OrganNode3D.POD) else 0.0

        # Build orientation matrix.  Leaf nodes use the exact local-to-world R matrix
        # when available; otherwise fall back to direction-based frame.
        if self.organ_type == OrganNode3D.LEAF and hasattr(self, "R_matrix") and self.R_matrix is not None:
            R = np.asarray(self.R_matrix, dtype=np.float64)
        else:
            d = self.direction if np.linalg.norm(self.direction) > 1e-6 else np.array([0.0, 0.0, 1.0])
            d = d / np.linalg.norm(d)
            R = np.zeros((3, 3), dtype=np.float64)
            R[:, 0] = d
        R_flat = R.flatten(order="C")  # row-major: 9 elements

        return np.array([
            self.position[0], self.position[1], self.position[2],  # 0:3
            self.length, self.radius,                              # 3, 4
            *R_flat,                                               # 5:14
            one_hot[0], one_hot[1], one_hot[2], one_hot[3], one_hot[4], one_hot[5],  # 14:20
            float(self.shoot_id), float(self.phytomer_idx),        # 20, 21
            self.existence, float(head_r),                         # 22, 23
            float(self.parent_idx),                                # 24 (parent_idx)
        ])

    @classmethod
    def from_vec(cls, vec: np.ndarray) -> "OrganNode3D":
        """Construct an OrganNode3D from a serialized feature vector.

        Supports 25D (current with 3x3 R matrix), 22D, 19D, 18D, and legacy 16D layouts.
        """
        if len(vec) >= 25:
            organ_type = int(np.argmax(vec[14:20]))
        elif len(vec) >= 22:
            organ_type = int(np.argmax(vec[11:17]))
        elif len(vec) >= 18:
            organ_type = int(np.argmax(vec[8:14]))
        else:
            organ_type = int(np.argmax(vec[8:12]))

        node = cls(organ_type)
        node.position = np.array(vec[0:3], dtype=np.float64)
        node.length = float(vec[3])
        node.radius = float(vec[4])

        if len(vec) >= 25:
            R_flat = np.array(vec[5:14], dtype=np.float64)
            R = R_flat.reshape(3, 3, order="C")
            node.R_matrix = R.copy()
            # direction is the first column of R if it is non-zero
            if np.linalg.norm(R[:, 0]) > 1e-6:
                node.direction = R[:, 0] / np.linalg.norm(R[:, 0])
            else:
                node.direction = np.array([0.0, 0.0, 1.0])
            node.shoot_id = int(round(float(vec[20])))
            node.phytomer_idx = int(round(float(vec[21])))
            node.existence = float(vec[22])
            node.flower_head_radius = float(vec[23])
            node.parent_idx = int(round(float(vec[24])))
        else:
            dir_vec = np.array(vec[5:8], dtype=np.float64)
            if np.linalg.norm(dir_vec) > 1e-6:
                node.direction = dir_vec / np.linalg.norm(dir_vec)

            if len(vec) >= 22:
                node.pitch = float(vec[8])
                node.yaw = float(vec[9])
                node.roll = float(vec[10])
                node.shoot_id = int(round(float(vec[17])))
                node.phytomer_idx = int(round(float(vec[18])))
                node.existence = float(vec[19])
                node.flower_head_radius = float(vec[20])
                node.parent_idx = int(round(float(vec[21])))
            else:
                node.shoot_id = int(round(float(vec[14 if len(vec) >= 18 else 12])))
                node.phytomer_idx = int(round(float(vec[15 if len(vec) >= 18 else 13])))
                node.existence = float(vec[16 if len(vec) >= 18 else 14])
                if len(vec) >= 18:
                    node.flower_head_radius = float(vec[17])
                if len(vec) >= 19:
                    node.parent_idx = int(round(float(vec[18])))
        return node







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

    def _segment_nodes_from_vertices(
        self,
        organ_type: int,
        vertices: List[np.ndarray],
        radii: List[float],
        base_attrs: Dict,
    ) -> List[OrganNode3D]:
        """Create one OrganNode3D per linear tube segment from a vertex chain.

        Each node uses its segment start as ``position`` and the segment end as
        ``tip_position``. The node's ``direction`` points from start to end and
        ``length`` is the segment length. This lets the 22D node-array renderer
        reconstruct curved tubes without requiring explicit curvature parameters.
        """
        nodes = []
        if len(vertices) < 2:
            return nodes
        for i in range(len(vertices) - 1):
            node = OrganNode3D(organ_type)
            p0 = vertices[i]
            p1 = vertices[i + 1]
            node.position = p0.copy()
            node.tip_position = p1.copy()
            seg_dir = p1 - p0
            seg_len = np.linalg.norm(seg_dir)
            node.direction = seg_dir / (seg_len + 1e-12)
            node.length = float(seg_len)
            node.radius = float(radii[i]) if i < len(radii) else float(radii[-1])
            node.shoot_id = base_attrs.get('shoot_id', 0)
            node.phytomer_idx = base_attrs.get('phytomer_idx', 0)
            node.pitch = base_attrs.get('pitch', 0.0)
            node.yaw = base_attrs.get('yaw', 0.0)
            node.roll = base_attrs.get('roll', 0.0)
            node.petiole_idx = base_attrs.get('petiole_idx', 0)
            nodes.append(node)
        return nodes

    def get_organ_nodes(self) -> List[OrganNode3D]:
        """Convert this phytomer into a list of OrganNode3D for 22D representation.

        Internodes and petioles are expanded into one node per linear segment so
        that the node-array renderer can reproduce curved stems without storing
        curvature parameters in the 22D feature vector. Leaflets remain one node
        per leaflet (trifoliate leaves already produce three leaf nodes).
        """
        nodes: List[OrganNode3D] = []

        # 1. Internode segments
        internode_xml_params = getattr(self, '_xml_params', {})
        shoot_xml_params = getattr(self, '_shoot_params', {})
        if self.internode_vertices and len(self.internode_vertices) >= 2:
            internode_segments = self._segment_nodes_from_vertices(
                OrganNode3D.INTERNODE,
                self.internode_vertices,
                self.internode_radii,
                {
                    'shoot_id': self.shoot_id,
                    'phytomer_idx': self.phytomer_index,
                    'pitch': self.internode_pitch,
                    'yaw': self.internode_phyllotactic_angle,
                },
            )
        else:
            # Fallback: single internode node from phytomer summary fields
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
            internode_segments = [inode]
        for inode in internode_segments:
            inode._xml_params = {**internode_xml_params, **shoot_xml_params}
        nodes.extend(internode_segments)

        # 2. Petiole nodes and attached organs
        for j, pet in enumerate(self.petioles):
            pet_vertices = pet.get('vertices')
            if pet_vertices and len(pet_vertices) >= 2:
                # Expand petiole into linear segments matching the reconstructed curve
                radii = []
                base_r = pet.get('radius', 0.0)
                taper = pet.get('taper', 0.25)
                n = len(pet_vertices)
                for k in range(n):
                    radii.append(base_r * (1.0 - taper / max(n - 1, 1) * k))
                petiole_segments = self._segment_nodes_from_vertices(
                    OrganNode3D.PETIOLE,
                    pet_vertices,
                    radii,
                    {
                        'shoot_id': self.shoot_id,
                        'phytomer_idx': self.phytomer_index,
                        'pitch': pet.get('pitch', 0.0),
                        'petiole_idx': j,
                    },
                )
            else:
                pnode = OrganNode3D(OrganNode3D.PETIOLE)
                pnode.position = pet.get('base_pos', self.internode_tip).copy()
                pnode.tip_position = pet.get('tip_pos', self.internode_tip).copy()
                pnode.direction = pet.get('axis', np.array([0, 0, 1])).copy()
                scale_fac = pet.get('scale_factor', 1.0)
                pnode.length = pet.get('length', 0.0) * scale_fac
                pnode.radius = max(pet.get('radius', 0.0) * scale_fac, 0.0008)
                pnode.pitch = pet.get('pitch', 0.0)
                pnode.shoot_id = self.shoot_id
                pnode.phytomer_idx = self.phytomer_index
                pnode.petiole_idx = j
                petiole_segments = [pnode]
            pet_xml_params = pet.get('_xml_params', {})
            first_seg = True
            for pseg in petiole_segments:
                pseg.petiole_idx = j
                if first_seg:
                    pseg._xml_params = pet_xml_params
                    first_seg = False
                else:
                    pseg._xml_params = {}
            nodes.extend(petiole_segments)

            # 3. Leaf nodes attached to this petiole
            for leaf in pet.get('leaves', []):
                lnode = OrganNode3D(OrganNode3D.LEAF)
                lnode.position = leaf.get('base_pos', petiole_segments[-1].tip_position).copy()
                lnode.tip_position = leaf.get('tip_pos', lnode.position).copy()
                scale_fac = leaf.get('scale_factor', 1.0)
                lnode.length = leaf.get('scale', 0.0) * scale_fac
                lnode.radius = leaf.get('scale', 0.0) * scale_fac * 0.70  # width radius
                lnode.pitch = leaf.get('pitch', 0.0)
                lnode.yaw = leaf.get('yaw', 0.0)
                lnode.shoot_id = self.shoot_id
                lnode.phytomer_idx = self.phytomer_index

                # Use the reconstructed world-space mesh to set the final midrib
                # direction. Roll is preserved separately because the node-array
                # renderer applies it as a twist about the midrib (matching
                # _leaflet_from_node).
                mesh_verts = leaf.get('mesh_verts')
                if mesh_verts is not None and mesh_verts.shape[0] >= 3:
                    n_cols = int(np.sqrt(mesh_verts.shape[0]))  # approximate grid width
                    # Midrib runs along the central row (Ny/2) from base (x=0) to tip (x=Nx)
                    ny = int(np.round((mesh_verts.shape[0] / (n_cols + 1)) - 1))
                    if ny <= 0:
                        ny = 1
                    mid_row = ny // 2
                    v0 = mesh_verts[mid_row * (n_cols + 1)]
                    v1 = mesh_verts[mid_row * (n_cols + 1) + n_cols]
                    midrib_dir = v1 - v0
                    midrib_norm = np.linalg.norm(midrib_dir)
                    if midrib_norm > 1e-12:
                        lnode.direction = midrib_dir / midrib_norm
                    else:
                        lnode.direction = leaf.get('direction', petiole_segments[-1].direction).copy()
                else:
                    lnode.direction = leaf.get('direction', petiole_segments[-1].direction).copy()
                lnode.roll = leaf.get('roll', 0.0)
                lnode.petiole_idx = j
                lnode._xml_params = leaf.get('_xml_params', {})
                # Preserve exact local-to-world orientation matrix for 25D node layout.
                if 'R_matrix' in leaf:
                    lnode.R_matrix = leaf['R_matrix'].copy()
                nodes.append(lnode)

            # 4. Floral bud nodes (Filter out dormant / unexpanded peduncles)
            for fbud in pet.get('floral_buds', []):
                fnode = OrganNode3D(OrganNode3D.FLORAL_BUD)
                fnode.position = fbud.get('base_pos', petiole_segments[-1].tip_position).copy()
                fnode.length = fbud.get('peduncle_length', 0.0)
                fnode.radius = fbud.get('peduncle_radius', 0.0)
                fnode.pitch = fbud.get('peduncle_pitch', 0.0)
                fnode.shoot_id = self.shoot_id
                fnode.phytomer_idx = self.phytomer_index

                # Use the reconstructed peduncle direction/head when available
                # (set by _reconstruct_petiole_geometry); otherwise fall back to
                # the geometry stored in the XML (or a vertical default).
                if 'head_pos' in fbud:
                    fnode.tip_position = fbud['head_pos'].copy()
                    fnode.direction = fbud['peduncle_dir'].copy()
                elif np.linalg.norm(fbud.get('tip_pos', np.zeros(3))) > 1e-9:
                    fnode.tip_position = fbud['tip_pos'].copy()
                    d = fnode.tip_position - fnode.position
                    fnode.direction = d / (np.linalg.norm(d) + 1e-12)
                else:
                    fnode.tip_position = fbud.get('base_pos', fnode.position).copy()
                    fnode.direction = np.array([0.0, 0.0, 1.0])

                # Helios C++ BudState (PlantArchitecture.h:272):
                # 0=BUD_DORMANT, 1=BUD_ACTIVE, 2=BUD_FLOWER_CLOSED, 3=BUD_FLOWER_OPEN, 4=BUD_FRUITING, 5=BUD_DEAD
                state = fbud.get('bud_state', 0)
                if state in [2, 3]:
                    # Open/closed flower → FLOWER organ type (yellow blossom ~8mm radius)
                    fnode.organ_type = OrganNode3D.FLOWER
                    fnode.existence = 1.0
                    fnode.flower_head_radius = min(float(fbud.get('flower_prototype_scale', 0.008)), 0.012)
                elif state == 4:
                    # Developing fruit / pod → POD organ type (cyan pod capsule ~5mm radius)
                    fnode.organ_type = OrganNode3D.POD
                    fnode.existence = 1.0
                    fnode.flower_head_radius = min(float(fbud.get('flower_prototype_scale', 0.005)), 0.008)
                elif state == 1:
                    # Active (unexpanded) bud → FLORAL_BUD, hide if peduncle not extended
                    fnode.organ_type = OrganNode3D.FLORAL_BUD
                    fnode.existence = 1.0 if fnode.length >= 0.01 else 0.0
                    fnode.flower_head_radius = 0.0
                else:
                    # 0=BUD_DORMANT, 5=BUD_DEAD → Do not render
                    fnode.existence = 0.0
                    fnode.flower_head_radius = 0.0

                fnode.petiole_idx = j
                fnode._xml_params = fbud.get('_xml_params', {})
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

        # Sort shoots by topological order: main stem (parent_shoot_id < 0) first,
        # then secondary shoots sorted by parent_shoot_id and shoot_id.
        shoot_data_list.sort(key=lambda s: (s.parent_shoot_id >= 0, s.parent_shoot_id, s.shoot_id))

        for sd in shoot_data_list:
            self.shoots[sd.shoot_id] = sd

        for sd in shoot_data_list:
            self._reconstruct_shoot_geometry(sd)
            self.phytomers.extend(sd.phytomers)

        return self.phytomers

    def get_all_organ_nodes(self) -> List[OrganNode3D]:
        """Get all organ nodes (22D compatible) from parsed phytomers.

        Internodes and petioles are expanded into one node per linear segment so
        that the 22D node-array renderer can reproduce exact C++ Helios curved
        geometry. The parent graph topology is stored in ``parent_idx`` so the
        array is fully self-contained.
        """
        if not self.phytomers:
            self.parse()

        all_nodes: List[OrganNode3D] = []
        # Map (shoot_id, phytomer_index) -> list of global indices for internode segments
        internode_segment_indices: Dict[Tuple[int, int], List[int]] = {}
        # Map (shoot_id, phytomer_index, petiole_index) -> list of global indices for petiole segments
        petiole_segment_indices: Dict[Tuple[int, int, int], List[int]] = {}

        for phyt in self.phytomers:
            organ_nodes = phyt.get_organ_nodes()
            start_idx = len(all_nodes)  # global index of organ_nodes[0]

            # Collect segment indices for parent wiring
            for idx, node in enumerate(organ_nodes):
                global_idx = start_idx + idx
                if node.organ_type == OrganNode3D.INTERNODE:
                    internode_segment_indices.setdefault((phyt.shoot_id, phyt.phytomer_index), []).append(global_idx)
                elif node.organ_type == OrganNode3D.PETIOLE:
                    # Use the petiole index stored during node construction (0-based petiole
                    # within this phytomer), not the segment index.
                    petiole_segment_indices.setdefault(
                        (phyt.shoot_id, phyt.phytomer_index, node.petiole_idx), []
                    ).append(global_idx)

            # Assign parent indices using stable global offsets
            internode_seg_idx = 0
            petiole_cnt = 0
            for i, node in enumerate(organ_nodes):
                if node.organ_type == OrganNode3D.INTERNODE:
                    # First segment of a phytomer attaches to the last segment of the previous phytomer
                    if phyt.phytomer_index > 0:
                        prev_segs = internode_segment_indices.get((phyt.shoot_id, phyt.phytomer_index - 1), [])
                        if internode_seg_idx == 0 and prev_segs:
                            node.parent_idx = prev_segs[-1]
                        elif internode_seg_idx > 0:
                            node.parent_idx = start_idx + i - 1
                    else:
                        shoot_data = self.shoots.get(phyt.shoot_id)
                        if shoot_data and shoot_data.parent_shoot_id >= 0:
                            # parent_node_index = k maps to (k-1)-th phytomer (0-based)
                            p_phyt_idx = shoot_data.parent_node_index - 1 if shoot_data.parent_node_index > 0 else 0
                            p_pet_idx = shoot_data.parent_petiole_index
                            parent_pet_segs = petiole_segment_indices.get((shoot_data.parent_shoot_id, p_phyt_idx, p_pet_idx), [])
                            parent_int_segs = internode_segment_indices.get((shoot_data.parent_shoot_id, p_phyt_idx), [])
                            if internode_seg_idx == 0 and parent_pet_segs:
                                node.parent_idx = parent_pet_segs[-1]
                            elif internode_seg_idx == 0 and parent_int_segs:
                                node.parent_idx = parent_int_segs[-1]
                            elif internode_seg_idx > 0:
                                node.parent_idx = start_idx + i - 1
                        else:
                            if internode_seg_idx > 0:
                                node.parent_idx = start_idx + i - 1
                            else:
                                node.parent_idx = start_idx + i
                    internode_seg_idx += 1

                elif node.organ_type == OrganNode3D.PETIOLE:
                    # First petiole segment attaches to the internode tip (last internode segment)
                    int_segs = internode_segment_indices.get((phyt.shoot_id, phyt.phytomer_index), [])
                    if petiole_cnt == 0:
                        node.parent_idx = int_segs[-1] if int_segs else start_idx + i - 1
                    else:
                        node.parent_idx = start_idx + i - 1
                    petiole_cnt += 1

                elif node.organ_type in [OrganNode3D.LEAF, OrganNode3D.FLORAL_BUD, OrganNode3D.FLOWER, OrganNode3D.POD]:
                    # Attach to the last petiole segment before this node in the local list
                    local_petiole_idx = -1
                    for k in range(i - 1, -1, -1):
                        if organ_nodes[k].organ_type == OrganNode3D.PETIOLE:
                            local_petiole_idx = k
                            break
                    if local_petiole_idx >= 0:
                        node.parent_idx = start_idx + local_petiole_idx
                    else:
                        # Fallback to internode if no petiole exists
                        int_segs = internode_segment_indices.get((phyt.shoot_id, phyt.phytomer_index), [])
                        node.parent_idx = int_segs[-1] if int_segs else -1

                # DAP 1 Override: Hide un-emerged secondary shoots (Shoot 1+) on Day 1 (DAP 1)
                if self.is_dap1 and node.shoot_id > 0:
                    node.existence = 0.0

            all_nodes.extend(organ_nodes)

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
            # Preserve shoot-level XML parameters so the organ-node serializer can
            # reconstruct the original <shoot> block exactly.
            p3d._shoot_params = {
                'ID': str(sd.shoot_id),
                'shoot_type_label': sd.shoot_type_label,
                'parent_shoot_ID': str(sd.parent_shoot_id),
                'parent_node_index': str(sd.parent_node_index),
                'parent_petiole_index': str(sd.parent_petiole_index),
                'base_rotation': ' '.join(str(int(x)) if float(x) == int(float(x)) else str(x) for x in sd.base_rotation),
            }
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

        def _raw_text(parent: ET.Element, tag: str, default: Optional[str] = None) -> Optional[str]:
            child = parent.find(tag)
            return child.text if child is not None else default

        p3d._xml_params = {
            'internode_length': _raw_text(internode, "internode_length", "0"),
            'internode_radius': _raw_text(internode, "internode_radius", "0"),
            'internode_pitch': _raw_text(internode, "internode_pitch", "0"),
            'internode_phyllotactic_angle': _raw_text(internode, "internode_phyllotactic_angle", "0"),
            'internode_length_max': _raw_text(internode, "internode_length_max", str(p3d.internode_length) if p3d.internode_length else "0"),
            'internode_length_segments': _raw_text(internode, "internode_length_segments", "1"),
            'curvature_perturbations': _raw_text(internode, "curvature_perturbations", ""),
            'yaw_perturbations': _raw_text(internode, "yaw_perturbations", ""),
        }

        p3d.internode_length = float(p3d._xml_params['internode_length'])
        p3d.internode_radius = float(p3d._xml_params['internode_radius'])
        p3d.internode_pitch = float(p3d._xml_params['internode_pitch'])
        p3d.internode_phyllotactic_angle = float(p3d._xml_params['internode_phyllotactic_angle'])

        p3d._internode_length_max = float(p3d._xml_params['internode_length_max'])
        p3d._internode_length_segments = int(p3d._xml_params['internode_length_segments'])

        p3d._curvature_perturbations = self._parse_semicolon_floats(
            p3d._xml_params['curvature_perturbations'])
        p3d._yaw_perturbations = self._parse_semicolon_floats(
            p3d._xml_params['yaw_perturbations'])

        for petiole_elem in internode.findall("petiole"):
            pet_geom = self._parse_geometry(petiole_elem, p3d.internode_tip)
            petiole_data = {
                'base_pos': pet_geom['position'],
                'tip_pos': pet_geom['tip_position'],
                'axis': pet_geom['direction'],
                '_xml_params': {
                    'petiole_length': _raw_text(petiole_elem, "petiole_length", "0"),
                    'petiole_radius': _raw_text(petiole_elem, "petiole_radius", "0"),
                    'petiole_pitch': _raw_text(petiole_elem, "petiole_pitch", "0"),
                    'petiole_curvature': _raw_text(petiole_elem, "petiole_curvature", "0"),
                    'current_leaf_scale_factor': _raw_text(petiole_elem, "current_leaf_scale_factor", "1"),
                    'petiole_taper': _raw_text(petiole_elem, "petiole_taper", "0.25"),
                    'petiole_length_segments': _raw_text(petiole_elem, "petiole_length_segments", "5"),
                    'petiole_radial_subdivisions': _raw_text(petiole_elem, "petiole_radial_subdivisions", "6"),
                    'leaflet_scale': _raw_text(petiole_elem, "leaflet_scale", "1"),
                    'leaflet_offset': _raw_text(petiole_elem, "leaflet_offset", "0"),
                },
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
                    '_xml_params': {
                        'leaf_scale': _raw_text(leaf_elem, "leaf_scale", "0"),
                        'leaf_pitch': _raw_text(leaf_elem, "leaf_pitch", "0"),
                        'leaf_yaw': _raw_text(leaf_elem, "leaf_yaw", "0"),
                        'leaf_roll': _raw_text(leaf_elem, "leaf_roll", "0"),
                    },
                    'scale': float(leaf_elem.findtext("leaf_scale", "0")),
                    'pitch': float(leaf_elem.findtext("leaf_pitch", "0")),
                    'yaw': float(leaf_elem.findtext("leaf_yaw", "0")),
                    'roll': float(leaf_elem.findtext("leaf_roll", "0")),
                    'scale_factor': petiole_data['scale_factor'],
                }
                petiole_data['leaves'].append(leaf_data)

            for fbud_elem in petiole_elem.findall("floral_bud"):
                bud_geom = self._parse_geometry(fbud_elem, pet_geom['tip_position'])
                fbud_xml_params = {
                    'bud_state': _raw_text(fbud_elem, "bud_state", "0"),
                    'parent_index': _raw_text(fbud_elem, "parent_index", "0"),
                    'bud_index': _raw_text(fbud_elem, "bud_index", "0"),
                    'is_terminal': _raw_text(fbud_elem, "is_terminal", "0"),
                    'current_fruit_scale_factor': _raw_text(fbud_elem, "current_fruit_scale_factor", "1"),
                }

                peduncle_elem = fbud_elem.find("peduncle")
                if peduncle_elem is not None:
                    fbud_xml_params['peduncle_length'] = _raw_text(peduncle_elem, "length", "0")
                    fbud_xml_params['peduncle_radius'] = _raw_text(peduncle_elem, "radius", "0")
                    fbud_xml_params['peduncle_pitch'] = _raw_text(peduncle_elem, "pitch", "0")
                    fbud_xml_params['peduncle_curvature'] = _raw_text(peduncle_elem, "curvature", "0")
                    fbud_xml_params['peduncle_roll'] = _raw_text(peduncle_elem, "roll", "0")

                # flower_offset / flower_prototype_scale = sphere radius for the flower/pod head
                inflorescence_elem = fbud_elem.find("inflorescence")
                if inflorescence_elem is not None:
                    flower_offset_raw = _raw_text(inflorescence_elem, "flower_offset", None)
                    if flower_offset_raw is not None:
                        fbud_xml_params['flower_offset'] = flower_offset_raw
                    flowers = []
                    for flower_elem in inflorescence_elem.findall("flower"):
                        flowers.append({
                            'flower_pitch': _raw_text(flower_elem, "flower_pitch", "0"),
                            'flower_yaw': _raw_text(flower_elem, "flower_yaw", "0"),
                            'flower_roll': _raw_text(flower_elem, "flower_roll", "0"),
                            'flower_azimuth': _raw_text(flower_elem, "flower_azimuth", "0"),
                            'flower_base_scale': _raw_text(flower_elem, "flower_base_scale", "0"),
                        })
                    if flowers:
                        fbud_xml_params['flowers'] = flowers
                else:
                    flower_offset_raw = _raw_text(fbud_elem, "flower_offset", None)
                    if flower_offset_raw is not None:
                        fbud_xml_params['flower_offset'] = flower_offset_raw

                fbud_data = {
                    'base_pos': bud_geom['position'],
                    'tip_pos': bud_geom['tip_position'],
                    'direction': bud_geom['direction'],
                    '_xml_params': fbud_xml_params,
                    'bud_state': int(fbud_xml_params['bud_state']),
                    'parent_index': int(fbud_xml_params['parent_index']),
                    'bud_index': int(fbud_xml_params['bud_index']),
                    'is_terminal': int(fbud_xml_params['is_terminal']),
                    'fruit_scale_factor': float(fbud_xml_params['current_fruit_scale_factor']),
                }

                if peduncle_elem is not None:
                    fbud_data['peduncle_length'] = float(fbud_xml_params['peduncle_length'])
                    fbud_data['peduncle_radius'] = float(fbud_xml_params['peduncle_radius'])
                    fbud_data['peduncle_pitch'] = float(fbud_xml_params['peduncle_pitch'])
                    fbud_data['peduncle_curvature'] = float(fbud_xml_params['peduncle_curvature'])
                    fbud_data['peduncle_roll'] = float(fbud_xml_params['peduncle_roll'])

                # flower_offset / flower_prototype_scale = sphere radius for the flower/pod head
                if 'flower_offset' in fbud_xml_params:
                    fbud_data['flower_prototype_scale'] = float(fbud_xml_params['flower_offset'])

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
                phyt.internode_pos = vertices[0].copy()
                phyt.internode_tip = vertices[-1].copy()
                phyt.internode_dir = _normalize(vertices[-1] - vertices[0])
                phyt.internode_vertices = vertices
                phyt.internode_radii = radii
                self._reconstruct_petiole_geometry(sd, phyt_idx, phyt, use_explicit=True)
                continue

            internode_base = self._get_internode_base(sd, phyt_idx)
            internode_axis, petiole_rotation_axis, shoot_bending_axis = \
                self._compute_internode_orientation(sd, phyt_idx)

            n_segments = max(1, phyt._internode_length_segments)
            dr = phyt.internode_length / float(n_segments)

            vertices = [internode_base.copy()]
            radii = [phyt.internode_radius]

            axis = internode_axis.copy()
            for seg in range(1, n_segments + 1):
                if phyt._curvature_perturbations:
                    pert_idx = seg - 1
                    if pert_idx < len(phyt._curvature_perturbations):
                        curvature_angle = math.radians(phyt._curvature_perturbations[pert_idx])
                        if abs(curvature_angle) > 1e-10:
                            axis = _rotate_point_about_line(axis, shoot_bending_axis, curvature_angle)

                if phyt._yaw_perturbations:
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
                if parent_sd:
                    # parent_node_index = k maps to (k-1)-th phytomer (0-based)
                    target_phyt_idx = sd.parent_node_index - 1 if sd.parent_node_index > 0 else 0
                    if 0 <= target_phyt_idx < len(parent_sd.phytomers):
                        parent_phyt = parent_sd.phytomers[target_phyt_idx]
                        pet_idx = sd.parent_petiole_index
                        if parent_phyt.petioles and 0 <= pet_idx < len(parent_phyt.petioles):
                            pet = parent_phyt.petioles[pet_idx]
                            if 'tip_pos' in pet:
                                return pet['tip_pos'].copy()
                            elif 'base_pos' in pet:
                                return pet['base_pos'].copy()
                        return parent_phyt.internode_tip.copy()
                    elif 0 <= target_phyt_idx < len(parent_sd.internode_vertices):
                        return parent_sd.internode_vertices[target_phyt_idx][-1].copy()
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
                parent_petiole_axis = self._ghost_petiole_axis(
                    parent_internode_axis,
                    float(phyt_idx - 1) * math.radians(prev.internode_phyllotactic_angle),
                )
        elif sd.parent_shoot_id >= 0:
            parent_sd = self.shoots.get(sd.parent_shoot_id)
            if parent_sd:
                target_phyt_idx = sd.parent_node_index - 1 if sd.parent_node_index > 0 else 0
                if 0 <= target_phyt_idx < len(parent_sd.phytomers):
                    parent_phyt = parent_sd.phytomers[target_phyt_idx]
                    parent_internode_axis = _normalize(parent_phyt.internode_dir)
                    pet_idx = min(sd.parent_petiole_index, len(parent_phyt.petioles) - 1) \
                        if parent_phyt.petioles else 0
                    if parent_phyt.petioles and 0 <= pet_idx < len(parent_phyt.petioles) and 'axis' in parent_phyt.petioles[pet_idx]:
                        parent_petiole_axis = _normalize(parent_phyt.petioles[pet_idx]['axis'])
                    else:
                        parent_petiole_axis = self._ghost_petiole_axis(
                            parent_internode_axis,
                            float(sd.parent_node_index) * math.radians(parent_phyt.internode_phyllotactic_angle),
                        )

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

                # Build exact local-to-world rotation matrix for 25D node layout.
                # This matches the C++ rotation chain in helios_geometry.py (which mirrors
                # PlantArchitecture.cpp) so that Track B renders leaflets at the same
                # orientation as Track A.
                leaf_pitch = math.radians(leaf.get('pitch', 0.0))
                leaf_yaw = math.radians(leaf.get('yaw', 0.0))
                leaf_roll = math.radians(leaf.get('roll', 0.0))

                # 1. roll about local x
                roll_rot = 0.0
                if leaves_per_petiole == 1:
                    sign = 1 if (phyt_idx % 2 == 0) else -1
                    roll_rot = -leaf_roll * sign
                elif ind_from_tip != 0:
                    sign = 1.0 if compound_rotation > 0 else -1.0
                    roll_rot = (math.asin(clamp_val(petiole_tip_axis[2], -1.0, 1.0)) + leaf_roll) * sign

                # 2. pitch about local y
                pitch_rot = leaf_pitch
                if ind_from_tip == 0:
                    pitch_rot += math.asin(clamp_val(petiole_tip_axis[2], -1.0, 1.0))

                # 3. yaw about local z for lateral leaflets
                yaw_rot = 0.0
                if ind_from_tip != 0:
                    sign = -compound_rotation / abs(compound_rotation)
                    yaw_rot = sign * leaf_yaw

                # 4. azimuth + compound rotation about world z
                azimuth_rot = -math.atan2(petiole_tip_axis[1], petiole_tip_axis[0]) + compound_rotation

                # Apply in the same order as helios_geometry.py: roll -> pitch -> yaw -> azimuth
                R = np.eye(3, dtype=np.float64)
                if abs(roll_rot) > 1e-10:
                    R = _rodrigues_matrix(np.array([1.0, 0.0, 0.0]), roll_rot) @ R
                if abs(pitch_rot) > 1e-10:
                    R = _rodrigues_matrix(np.array([0.0, 1.0, 0.0]), -pitch_rot) @ R
                if abs(yaw_rot) > 1e-10:
                    R = _rodrigues_matrix(np.array([0.0, 0.0, 1.0]), yaw_rot) @ R
                if abs(azimuth_rot) > 1e-10:
                    R = _rodrigues_matrix(np.array([0.0, 0.0, 1.0]), azimuth_rot) @ R

                # Blade-up correction for single leaves (simplified, matches helios_geometry.py)
                if leaves_per_petiole == 1:
                    r_h = math.sqrt(petiole_tip_axis[0] ** 2 + petiole_tip_axis[1] ** 2)
                    if r_h > 1e-4:
                        blade_correction = math.atan2(petiole_tip_axis[2] * r_h, r_h * r_h)
                        length_ratio = min(petiole_length / max(leaf_scale, 1e-6), 1.0)
                        blade_correction *= length_ratio
                        blade_correction = max(
                            -0.5 * math.pi + math.radians(1.0),
                            min(0.5 * math.pi - math.radians(1.0), blade_correction),
                        )
                        sign = 1 if (phyt_idx % 2 == 0) else -1
                        if abs(blade_correction) > 1e-10:
                            R = _rodrigues_matrix(petiole_tip_axis, blade_correction * sign) @ R
                leaf['R_matrix'] = R

            for fbud in petiole.get('floral_buds', []):
                # Reconstruct the peduncle (floral-bud stem) direction and head
                # position following Helios PlantArchitecture.cpp
                # `Phytomer::updateInflorescence`:
                #   1. start with the internode axis,
                #   2. apply peduncle pitch about the inflorescence bending axis
                #      (= cross(parent_internode_axis, petiole_axis)),
                #   3. accumulate curvature toward vertical along segments.
                fbud['base_pos'] = pet_vertices[-1].copy()
                bud_base = pet_vertices[-1].copy()
                peduncle_pitch = fbud.get('peduncle_pitch', 0.0)
                peduncle_curvature = fbud.get('peduncle_curvature', 0.0)
                peduncle_length = fbud.get('peduncle_length', 0.0)

                # Helios: peduncle_axis = getInternodeAxisVector(1) (internode axis).
                bud_axis = _normalize(internode_axis)
                # inflorescence_bending_axis = cross(parent_internode_axis, petiole_axis_actual)
                # where petiole_axis_actual is the petiole's final (curved) axis = pet_axis.
                bend_axis = _normalize(np.cross(internode_axis, pet_axis))
                if not np.isfinite(bend_axis).all() or np.linalg.norm(bend_axis) < 1e-6:
                    bend_axis = np.array([1.0, 0.0, 0.0])
                if abs(peduncle_pitch) > 1e-10:
                    bud_axis = _rotate_point_about_line(
                        bud_axis, bend_axis, math.radians(peduncle_pitch))

                n_seg = max(1, int(petiole.get('length_segments', 3)))
                dr = peduncle_length / float(n_seg)
                bud_pos = bud_base.copy()
                if abs(peduncle_curvature) > 1e-10:
                    # Horizontal bending axis perpendicular to current direction.
                    # Match C++: theta_curvature = deg2rad(curvature * dr)  (degrees per segment).
                    h_bend = np.cross(bud_axis, np.array([0.0, 0.0, 1.0]))
                    h_norm = np.linalg.norm(h_bend)
                    sign = 1.0 if peduncle_curvature > 0 else -1.0
                    target = np.array([0.0, 0.0, sign])
                    theta_curv = math.radians(peduncle_curvature * dr)
                    for _ in range(n_seg):
                        if h_norm > 1e-3:
                            h_bend_u = h_bend / h_norm
                            cos_ang = max(-1.0, min(1.0, np.dot(bud_axis, target)))
                            theta_target = math.acos(cos_ang)
                            if abs(theta_curv) >= theta_target:
                                bud_axis = target.copy()
                            else:
                                bud_axis = _rotate_point_about_line(
                                    bud_axis, h_bend_u, theta_curv)
                            h_bend = np.cross(bud_axis, np.array([0.0, 0.0, 1.0]))
                            h_norm = np.linalg.norm(h_bend)
                        else:
                            bud_axis = target.copy()
                        bud_pos = bud_pos + dr * _normalize(bud_axis)
                else:
                    bud_pos = bud_base + peduncle_length * _normalize(bud_axis)

                fbud['head_pos'] = bud_pos.copy()
                fbud['peduncle_dir'] = _normalize(bud_axis)

    def _parse_vec3(self, text: str) -> np.ndarray:
        parts = text.strip().split()
        if len(parts) >= 3:
            return np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.zeros(3)

    def _parse_semicolon_floats(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []
        return [float(x) for x in text.strip().split(";") if x.strip()]

    def _ghost_petiole_axis(self, parent_internode_axis: np.ndarray, cumulative_rotation: float) -> np.ndarray:
        """Create a ghost petiole reference vector for phytomers without explicit petioles.

        Matches PlantArchitecture.cpp (L1078-1088): ghost = cross(internode, z), with
        cumulative rotation about the internode axis by parent_node_index * phyllotactic_angle.
        """
        ghost = np.cross(parent_internode_axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(ghost) < 0.01:
            ghost = np.array([0.0, 1.0, 0.0])
        ghost = _normalize(ghost)
        if abs(cumulative_rotation) > 1e-10:
            ghost = _rotate_point_about_line(ghost, parent_internode_axis, cumulative_rotation)
        return ghost

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


# ═══════════════════════════════════════════════════════════════════════════════
# XML round-trip serializer: OrganNode3D list -> Helios plant XML text
# ═══════════════════════════════════════════════════════════════════════════════

def _group_nodes_for_xml(nodes: List[OrganNode3D]) -> Dict:
    """Group a flat organ-node list back into the original XML hierarchy.

    Returns a nested dict:
        {shoot_id: {'phytomers': {phytomer_idx: {'internode': {...},
                                                'petioles': {petiole_idx: {...}}}}}}
    """
    shoots: Dict[int, Dict] = {}
    for node in nodes:
        if node.shoot_id not in shoots:
            shoots[node.shoot_id] = {'phytomers': {}}
        phytomers = shoots[node.shoot_id]['phytomers']
        if node.phytomer_idx not in phytomers:
            phytomers[node.phytomer_idx] = {'internode': {}, 'petioles': {}}
        phyt = phytomers[node.phytomer_idx]

        if node.organ_type == OrganNode3D.INTERNODE:
            # First internode segment carries the original XML parameters.
            if not phyt['internode'] and node._xml_params:
                phyt['internode'] = node._xml_params
        elif node.organ_type == OrganNode3D.PETIOLE:
            pidx = node.petiole_idx
            if pidx not in phyt['petioles']:
                phyt['petioles'][pidx] = {'params': {}, 'leaves': [], 'floral_buds': []}
            if node._xml_params and not phyt['petioles'][pidx]['params']:
                phyt['petioles'][pidx]['params'] = node._xml_params
        elif node.organ_type == OrganNode3D.LEAF:
            pidx = node.petiole_idx
            if pidx not in phyt['petioles']:
                phyt['petioles'][pidx] = {'params': {}, 'leaves': [], 'floral_buds': []}
            if node._xml_params:
                phyt['petioles'][pidx]['leaves'].append(node._xml_params)
        elif node.organ_type in (OrganNode3D.FLORAL_BUD, OrganNode3D.FLOWER, OrganNode3D.POD):
            # Original XML always stores flowers/pods as <floral_bud> elements;
            # organ_type may be remapped for rendering, but _xml_params holds the
            # original floral-bud parameters.
            pidx = node.petiole_idx
            if pidx not in phyt['petioles']:
                phyt['petioles'][pidx] = {'params': {}, 'leaves': [], 'floral_buds': []}
            if node._xml_params:
                phyt['petioles'][pidx]['floral_buds'].append(node._xml_params)
    return shoots


def _fmt_xml_text(value: str, compact: bool = False) -> str:
    """Return a string ready to be written inside an XML element.

    The Helios C++ writer adds spaces around shoot-level labels and vectors but
    writes numeric leaf elements without extra spaces. We mirror that convention.
    """
    stripped = value.strip()
    if compact:
        return stripped
    return f" {stripped} "


def _write_xml_element(lines: List[str], indent_level: int, tag: str, text: str, compact: bool = False) -> None:
    """Append a single XML element line with tab indentation."""
    indent = "\t" * indent_level
    lines.append(f"{indent}<{tag}>{_fmt_xml_text(text, compact=compact)}</{tag}>\n")


def _write_xml_open(lines: List[str], indent_level: int, tag: str, attrs: str = "") -> None:
    indent = "\t" * indent_level
    if attrs:
        lines.append(f"{indent}<{tag} {attrs}>\n")
    else:
        lines.append(f"{indent}<{tag}>\n")


def _write_xml_close(lines: List[str], indent_level: int, tag: str) -> None:
    indent = "\t" * indent_level
    lines.append(f"{indent}</{tag}>\n")


def organ_nodes_to_xml(
    nodes: List[OrganNode3D],
    base_position: str = "0 0 0",
    plant_age: str = "0",
    plant_id: str = "0",
) -> str:
    """Serialize a list of OrganNode3D objects back to a Helios plant XML string.

    The output reproduces the original XML text hierarchy and parameter values
    exactly when the nodes were produced by ``HeliosXMLParser.get_all_organ_nodes``.
    """
    grouped = _group_nodes_for_xml(nodes)

    lines: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>\n', '<helios>\n']
    _write_xml_open(lines, 1, "plant_instance", f'ID="{plant_id}"')
    _write_xml_element(lines, 2, "base_position", base_position)
    _write_xml_element(lines, 2, "plant_age", plant_age)

    for shoot_id in sorted(grouped.keys()):
        shoot = grouped[shoot_id]
        phytomers = shoot['phytomers']
        # Use any internode node's _xml_params for shoot-level attributes.
        shoot_params = {}
        for phyt in phytomers.values():
            if phyt['internode']:
                shoot_params = phyt['internode']
                break
        shoot_id_str = shoot_params.get('ID', str(shoot_id))
        _write_xml_open(lines, 2, "shoot", f'ID="{shoot_id_str}"')
        _write_xml_element(lines, 3, "shoot_type_label", shoot_params.get('shoot_type_label', 'main'))
        _write_xml_element(lines, 3, "parent_shoot_ID", shoot_params.get('parent_shoot_ID', '-1'))
        _write_xml_element(lines, 3, "parent_node_index", shoot_params.get('parent_node_index', '0'))
        _write_xml_element(lines, 3, "parent_petiole_index", shoot_params.get('parent_petiole_index', '0'))
        _write_xml_element(lines, 3, "base_rotation", shoot_params.get('base_rotation', '0 0 0'))

        for phyt_idx in sorted(phytomers.keys()):
            phyt = phytomers[phyt_idx]
            int_params = phyt['internode']
            _write_xml_open(lines, 3, "phytomer")
            _write_xml_open(lines, 4, "internode")
            for key in [
                "internode_length", "internode_radius", "internode_pitch",
                "internode_phyllotactic_angle", "internode_length_max",
                "internode_length_segments", "curvature_perturbations",
                "yaw_perturbations",
            ]:
                if key in int_params:
                    _write_xml_element(lines, 5, key, int_params[key], compact=True)

            for pet_idx in sorted(phyt['petioles'].keys()):
                pet = phyt['petioles'][pet_idx]
                pet_params = pet['params']
                _write_xml_open(lines, 5, "petiole")
                for key in [
                    "petiole_length", "petiole_radius", "petiole_pitch",
                    "petiole_curvature", "current_leaf_scale_factor", "petiole_taper",
                    "petiole_length_segments", "petiole_radial_subdivisions",
                    "leaflet_scale", "leaflet_offset",
                ]:
                    if key in pet_params:
                        _write_xml_element(lines, 6, key, pet_params[key], compact=True)

                for leaf_params in pet['leaves']:
                    _write_xml_open(lines, 6, "leaf")
                    for key in ["leaf_scale", "leaf_pitch", "leaf_yaw", "leaf_roll"]:
                        if key in leaf_params:
                            _write_xml_element(lines, 7, key, leaf_params[key], compact=True)
                    _write_xml_close(lines, 6, "leaf")

                for fbud_params in pet['floral_buds']:
                    _write_xml_open(lines, 6, "floral_bud")
                    for key in ["bud_state", "parent_index", "bud_index", "is_terminal", "current_fruit_scale_factor"]:
                        if key in fbud_params:
                            _write_xml_element(lines, 7, key, fbud_params[key], compact=True)
                    if any(k in fbud_params for k in ["peduncle_length", "peduncle_radius", "peduncle_pitch", "peduncle_curvature", "peduncle_roll"]):
                        _write_xml_open(lines, 7, "peduncle")
                        for key in ["peduncle_length", "peduncle_radius", "peduncle_pitch", "peduncle_curvature", "peduncle_roll"]:
                            if key in fbud_params:
                                # strip the "peduncle_" prefix when writing the <peduncle> child tags
                                tag = key.replace("peduncle_", "")
                                _write_xml_element(lines, 8, tag, fbud_params[key], compact=True)
                        _write_xml_close(lines, 7, "peduncle")
                    if "flower_offset" in fbud_params or "flowers" in fbud_params:
                        _write_xml_open(lines, 7, "inflorescence")
                        if "flower_offset" in fbud_params:
                            _write_xml_element(lines, 8, "flower_offset", fbud_params["flower_offset"], compact=True)
                        for flower_params in fbud_params.get("flowers", []):
                            _write_xml_open(lines, 8, "flower")
                            for key in ["flower_pitch", "flower_yaw", "flower_roll", "flower_azimuth", "flower_base_scale"]:
                                if key in flower_params:
                                    _write_xml_element(lines, 9, key, flower_params[key], compact=True)
                            _write_xml_close(lines, 8, "flower")
                        _write_xml_close(lines, 7, "inflorescence")
                    _write_xml_close(lines, 6, "floral_bud")

                _write_xml_close(lines, 5, "petiole")

            _write_xml_close(lines, 4, "internode")
            _write_xml_close(lines, 3, "phytomer")

        _write_xml_close(lines, 2, "shoot")

    _write_xml_close(lines, 1, "plant_instance")
    lines.append("</helios>\n")
    return "".join(lines)


def extract_xml_tag_coverage(xml_path: str) -> Dict[str, List[str]]:
    """Audit which XML element tags are consumed vs ignored by the parser.

    Returns a dict:
      - consumed: list of raw tag names that are read and stored in _xml_params
      - ignored:  list of tag names that appear in the file but are not preserved
    """
    consumed_tags = set()
    ignored_tags = set()

    parser = HeliosXMLParser(xml_path)
    parser.parse()

    # Tags we explicitly read into _xml_params (raw tag names as they appear in XML)
    for phyt in parser.phytomers:
        consumed_tags.update(phyt._xml_params.keys())
        for pet in phyt.petioles:
            consumed_tags.update(pet.get('_xml_params', {}).keys())
            for leaf in pet.get('leaves', []):
                consumed_tags.update(leaf.get('_xml_params', {}).keys())
            for fbud in pet.get('floral_buds', []):
                fbud_params = fbud.get('_xml_params', {})
                for key in fbud_params:
                    if key == 'flowers':
                        for flower in fbud_params['flowers']:
                            consumed_tags.update(flower.keys())
                    elif key.startswith('peduncle_'):
                        consumed_tags.add(key.replace('peduncle_', ''))
                    else:
                        consumed_tags.add(key)
        consumed_tags.update(getattr(phyt, '_shoot_params', {}).keys())

    # Tags actually present in the XML file (leaf elements only)
    for elem in parser.root.iter():
        if len(elem) == 0:
            ignored_tags.add(elem.tag)

    # Plant-level tags are consumed by the serializer
    consumed_tags.add('base_position')
    consumed_tags.add('plant_age')

    ignored_tags -= consumed_tags
    return {
        'consumed': sorted(consumed_tags),
        'ignored': sorted(ignored_tags),
    }


def verify_xml_round_trip(xml_path: str) -> Dict[str, Any]:
    """Parse a Helios XML file, serialize it from organ nodes, and compare.

    Returns a report dict with:
      - semantic_equal: bool (all tag values identical after re-parsing)
      - text_equal: bool (byte-identical to original, ignoring trailing newline)
      - original_text: str
      - roundtrip_text: str
      - first_diff_line: Optional[int]
      - mismatches: list of (path, original_value, roundtrip_value)
    """
    import xml.etree.ElementTree as ET

    parser = HeliosXMLParser(xml_path)
    parser.parse()
    nodes = parser.get_all_organ_nodes()

    def _fmt_vec(v) -> str:
        # Use the shortest round-tripping representation for vector components.
        def _fmt_comp(x):
            s = f"{float(x):.6g}"
            # Ensure integer-like values stay as "0" instead of "0.0" etc.
            try:
                if float(s) == int(float(s)):
                    return str(int(float(s)))
            except ValueError:
                pass
            return s
        return ' '.join(_fmt_comp(x) for x in v)

    base_pos = _fmt_vec(parser.base_position)
    # If the original text used integer-like ages (e.g. "11"), preserve that.
    age_text = parser.root.find(".//plant_age")
    plant_age = (age_text.text or str(parser.plant_age)).strip() if age_text is not None else str(parser.plant_age)
    plant_id = parser.root.find(".//plant_instance").get("ID", "0")
    roundtrip_text = organ_nodes_to_xml(nodes, base_position=base_pos, plant_age=plant_age, plant_id=plant_id)

    with open(xml_path, 'r', encoding='utf-8') as f:
        original_text = f.read()

    # Byte-identical check (allow trailing newline difference)
    text_equal = original_text.rstrip('\n') == roundtrip_text.rstrip('\n')

    # Semantic comparison: re-parse and compare all leaf tag texts.
    orig_tree = ET.parse(xml_path)
    round_tree = ET.fromstring(roundtrip_text)
    mismatches = []

    def elem_path(elem: ET.Element) -> str:
        parent = elem.find('..')
        tag = elem.tag
        if parent is None:
            return tag
        siblings = [c for c in parent if c.tag == tag]
        pos = siblings.index(elem) + 1
        return f"{elem_path(parent)}/{tag}[{pos}]"

    def collect_elems(root: ET.Element):
        out = []
        for elem in root.iter():
            if len(elem) == 0:  # leaf
                out.append((elem_path(elem), elem.tag, elem.text))
        return out

    orig_leaves = collect_elems(orig_tree.getroot())
    round_leaves = collect_elems(round_tree)

    semantic_equal = len(orig_leaves) == len(round_leaves)
    if semantic_equal:
        for (o_path, o_tag, o_text), (r_path, r_tag, r_text) in zip(orig_leaves, round_leaves):
            o_val = (o_text or '').strip()
            r_val = (r_text or '').strip()
            if o_val != r_val:
                semantic_equal = False
                mismatches.append((o_path, o_val, r_val))
    else:
        semantic_equal = False
        mismatches.append(('leaf-count', len(orig_leaves), len(round_leaves)))

    # Line-by-line first diff
    first_diff_line = None
    orig_lines = original_text.splitlines()
    round_lines = roundtrip_text.splitlines()
    for i, (o, r) in enumerate(zip(orig_lines, round_lines)):
        if o != r:
            first_diff_line = i
            break
    else:
        if len(orig_lines) != len(round_lines):
            first_diff_line = min(len(orig_lines), len(round_lines))

    return {
        'semantic_equal': semantic_equal,
        'text_equal': text_equal,
        'first_diff_line': first_diff_line,
        'mismatches': mismatches,
        'original_text': original_text,
        'roundtrip_text': roundtrip_text,
    }
