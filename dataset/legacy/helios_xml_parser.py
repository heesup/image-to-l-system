"""Helios XML parser for 3D plant organ graph extraction.

Converts Helios L-system XML output into a 15D node feature representation
suitable for the PlantGraphDiffuser3D model.
"""

import math
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, Any, Tuple, List


# Normalization constants (metric -> [0, 1])
LENGTH_MAX = 1.0          # meters
RADIUS_MAX = 0.01         # meters
ANGLE_MIN = -180.0        # degrees
ANGLE_MAX = 180.0         # degrees

ORGAN_TYPE_MAP = {
    "internode": 0,
    "petiole": 1,
    "leaf": 2,
    "floral_bud": 3,
}


def _parse_float(text: str) -> float:
    """Parse a space-separated triple or single float."""
    parts = text.strip().split()
    if len(parts) == 1:
        return float(parts[0])
    return [float(p) for p in parts]


def _parse_vec3(text: str) -> np.ndarray:
    """Parse '<x> <y> <z>' into a numpy array."""
    parts = text.strip().split()
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _normalize_angle(deg: float) -> float:
    """Map degree angle to [0, 1] given ANGLE_MIN/MAX."""
    return (deg - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)


def _normalize_length(m: float) -> float:
    return np.clip(m / LENGTH_MAX, 0.0, 1.0)


def _normalize_radius(m: float) -> float:
    return np.clip(m / RADIUS_MAX, 0.0, 1.0)


def _rotation_matrix_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float32)


def _rotation_matrix_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


def _rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float32)


def _apply_euler_xyz(pitch_deg: float, yaw_deg: float, roll_deg: float,
                     vec: np.ndarray) -> np.ndarray:
    """Apply intrinsic X-Y-Z (roll-pitch-yaw) rotation.

    Helios leaf geometry uses roll(X), pitch(Y), yaw(Z) order for the first
    three angles. For other organs we use the same convention as a practical
    approximation.
    """
    R = _rotation_matrix_z(_deg2rad(yaw_deg)) @ \
        _rotation_matrix_y(_deg2rad(pitch_deg)) @ \
        _rotation_matrix_x(_deg2rad(roll_deg))
    return R @ vec


def _normalize_xyz(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalize 3D coordinates to [0, 1]^3.

    Returns normalized coords, min vector, and scale vector (max - min).
    For inverse: coords_orig = coords_norm * scale + min.
    """
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    scale = max_vals - min_vals
    scale = np.where(scale < 1e-6, 1.0, scale)
    norm_coords = (coords - min_vals) / scale
    return np.clip(norm_coords, 0.0, 1.0), min_vals, scale


def parse_helios_xml(xml_path: str, max_nodes: int = 2048,
                     normalize: bool = True) -> Dict[str, Any]:
    """Parse a Helios plant XML into a 15D organ graph.

    Node feature layout (15D):
        0-2: x, y, z (base position)
        3:   length / scale
        4:   radius / thickness
        5-7: pitch, yaw, roll (degrees normalized to [0,1])
        8-11: organ_type one-hot [internode, petiole, leaf, floral_bud]
        12:  shoot_id
        13:  phytomer_idx
        14:  existence

    Attachment hierarchy encoded in parent_indices:
        - shoot[>0] base  -> parent shoot's last internode tip node
        - internode      -> previous internode on same shoot, or shoot base
        - petiole        -> parent internode
        - leaf           -> parent petiole
        - floral_bud     -> parent petiole
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    plant_instance = root.find("plant_instance")
    if plant_instance is None:
        raise ValueError(f"No <plant_instance> found in {xml_path}")

    base_position = _parse_vec3(plant_instance.find("base_position").text)
    plant_age_text = plant_instance.find("plant_age")
    dap = int(plant_age_text.text.strip()) if plant_age_text is not None else 10

    # Raw node list building: each entry = (node_index, attrs)
    raw_nodes: List[Dict[str, Any]] = []

    # Track last internode node per shoot for sequential internode parenting.
    last_internode_idx: Dict[int, int] = {}

    def add_node(organ_type: str, shoot_id: int, phytomer_idx: int,
                 base_xyz: np.ndarray, length: float, radius: float,
                 pitch: float, yaw: float, roll: float) -> int:
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[ORGAN_TYPE_MAP[organ_type]] = 1.0
        node = {
            "organ_type": ORGAN_TYPE_MAP[organ_type],
            "shoot_id": shoot_id,
            "phytomer_idx": phytomer_idx,
            "base_xyz": base_xyz.astype(np.float32),
            "length": length,
            "radius": radius,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "one_hot": one_hot,
        }
        raw_nodes.append(node)
        return len(raw_nodes) - 1

    # Iterate shoots
    for shoot in plant_instance.findall("shoot"):
        shoot_id = int(shoot.get("ID"))
        base_rotation = _parse_vec3(shoot.find("base_rotation").text)
        parent_shoot_id = int(shoot.find("parent_shoot_ID").text)
        parent_node_index = int(shoot.find("parent_node_index").text)
        # parent_petiole_index = int(shoot.find("parent_petiole_index").text)

        # Shoot base node
        if parent_shoot_id == -1:
            shoot_base_xyz = base_position.copy()
            shoot_base_parent = 0  # self-loop for root
        else:
            # Attach to the last internode of the parent shoot.
            parent_idx = last_internode_idx.get(parent_shoot_id, 0)
            shoot_base_xyz = raw_nodes[parent_idx]["base_xyz"].copy() if parent_idx < len(raw_nodes) else base_position.copy()
            shoot_base_parent = parent_idx

        # Add a virtual shoot base node (type internode-ish, but encoded as internode)
        # We keep existence=1 but mark it as a shoot base with phytomer_idx=0.
        shoot_base_idx = add_node(
            organ_type="internode",
            shoot_id=shoot_id,
            phytomer_idx=0,
            base_xyz=shoot_base_xyz,
            length=0.0,
            radius=0.0,
            pitch=base_rotation[0],
            yaw=base_rotation[1],
            roll=base_rotation[2],
        )

        # Initial shoot direction: start pointing up (Y-up in Helios-ish convention)
        # We'll use base_rotation as Euler XYZ.
        shoot_dir = _apply_euler_xyz(base_rotation[0], base_rotation[1], base_rotation[2],
                                     np.array([0.0, 1.0, 0.0], dtype=np.float32))
        shoot_dir = shoot_dir / (np.linalg.norm(shoot_dir) + 1e-8)

        prev_internode_idx: int = shoot_base_idx
        last_internode_idx[shoot_id] = shoot_base_idx

        phytomer_idx = 0
        for phytomer in shoot.findall("phytomer"):
            phytomer_idx += 1

            internode = phytomer.find("internode")
            if internode is None:
                continue

            internode_length = float(internode.find("internode_length").text)
            internode_radius = float(internode.find("internode_radius").text)
            internode_pitch = float(internode.find("internode_pitch").text)
            internode_yaw = float(internode.find("internode_phyllotactic_angle").text)
            internode_roll = 0.0  # XML doesn't expose roll for internodes

            # Compute internode base: previous internode tip
            internode_base = raw_nodes[prev_internode_idx]["base_xyz"].copy()
            # Direction: apply pitch/yaw to shoot direction
            internode_dir = _apply_euler_xyz(internode_pitch, internode_yaw, 0.0, shoot_dir)
            internode_dir = internode_dir / (np.linalg.norm(internode_dir) + 1e-8)
            internode_tip = internode_base + internode_length * internode_dir

            internode_idx = add_node(
                organ_type="internode",
                shoot_id=shoot_id,
                phytomer_idx=phytomer_idx,
                base_xyz=internode_base,
                length=internode_length,
                radius=internode_radius,
                pitch=internode_pitch,
                yaw=internode_yaw,
                roll=internode_roll,
            )
            prev_internode_idx = internode_idx
            last_internode_idx[shoot_id] = internode_idx

            # Petioles attach at internode tip
            for petiole in internode.findall("petiole"):
                petiole_length = float(petiole.find("petiole_length").text)
                petiole_radius = float(petiole.find("petiole_radius").text)
                petiole_pitch = float(petiole.find("petiole_pitch").text)
                petiole_curvature = float(petiole.find("petiole_curvature").text)
                petiole_yaw = 0.0  # lateral offset not explicitly given; inferred from leaf yaws
                petiole_roll = 0.0

                petiole_base = internode_tip.copy()
                # Petiole direction: tilt away from internode direction by petiole_pitch,
                # then apply curvature as a simple secondary pitch.
                petiole_dir = _apply_euler_xyz(petiole_pitch + 0.2 * petiole_curvature,
                                               petiole_yaw, 0.0, internode_dir)
                petiole_dir = petiole_dir / (np.linalg.norm(petiole_dir) + 1e-8)
                petiole_tip = petiole_base + petiole_length * petiole_dir

                petiole_idx = add_node(
                    organ_type="petiole",
                    shoot_id=shoot_id,
                    phytomer_idx=phytomer_idx,
                    base_xyz=petiole_base,
                    length=petiole_length,
                    radius=petiole_radius,
                    pitch=petiole_pitch,
                    yaw=petiole_yaw,
                    roll=petiole_roll,
                )

                # Leaves attach to petiole (parent = petiole)
                for leaf in petiole.findall("leaf"):
                    leaf_scale = float(leaf.find("leaf_scale").text)
                    leaf_pitch = float(leaf.find("leaf_pitch").text)
                    leaf_yaw = float(leaf.find("leaf_yaw").text)
                    leaf_roll = float(leaf.find("leaf_roll").text)

                    # Leaf base = petiole tip
                    leaf_base = petiole_tip.copy()
                    leaf_dir = _apply_euler_xyz(leaf_pitch, leaf_yaw, leaf_roll, petiole_dir)
                    leaf_dir = leaf_dir / (np.linalg.norm(leaf_dir) + 1e-8)

                    add_node(
                        organ_type="leaf",
                        shoot_id=shoot_id,
                        phytomer_idx=phytomer_idx,
                        base_xyz=leaf_base,
                        length=leaf_scale,
                        radius=leaf_scale * 0.5,  # approximate width ratio
                        pitch=leaf_pitch,
                        yaw=leaf_yaw,
                        roll=leaf_roll,
                    )

                # Floral buds attach to petiole (parent = petiole)
                for floral_bud in petiole.findall("floral_bud"):
                    peduncle = floral_bud.find("peduncle")
                    if peduncle is None:
                        continue
                    peduncle_length = float(peduncle.find("length").text)
                    peduncle_radius = float(peduncle.find("radius").text)
                    peduncle_pitch = float(peduncle.find("pitch").text)
                    peduncle_curvature = float(peduncle.find("curvature").text)
                    peduncle_roll = float(peduncle.find("roll").text)
                    peduncle_yaw = 0.0

                    # Peduncle base = petiole tip (same as leaf base)
                    peduncle_base = petiole_tip.copy()
                    peduncle_dir = _apply_euler_xyz(peduncle_pitch + 0.2 * peduncle_curvature,
                                                    peduncle_yaw, peduncle_roll, petiole_dir)
                    peduncle_dir = peduncle_dir / (np.linalg.norm(peduncle_dir) + 1e-8)

                    add_node(
                        organ_type="floral_bud",
                        shoot_id=shoot_id,
                        phytomer_idx=phytomer_idx,
                        base_xyz=peduncle_base,
                        length=peduncle_length,
                        radius=peduncle_radius,
                        pitch=peduncle_pitch,
                        yaw=peduncle_yaw,
                        roll=peduncle_roll,
                    )

    num_nodes = len(raw_nodes)
    if num_nodes > max_nodes:
        # Prune the oldest/highest-index phytomers to fit within max_nodes.
        # This preserves the plant topology up to the capacity limit.
        print(f"Warning: Plant has {num_nodes} nodes, pruning to max_nodes={max_nodes}.")
        raw_nodes = raw_nodes[:max_nodes]
        num_nodes = len(raw_nodes)

    # Build 15D node matrix
    nodes_15d = np.zeros((max_nodes, 15), dtype=np.float32)
    organ_types = np.zeros(max_nodes, dtype=np.int64)
    shoot_ids = np.zeros(max_nodes, dtype=np.int64)
    phytomer_indices = np.zeros(max_nodes, dtype=np.int64)
    existence_mask = np.zeros(max_nodes, dtype=np.float32)
    parent_indices = np.zeros(max_nodes, dtype=np.int64)
    adj_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)

    # Collect base coordinates for normalization
    base_xyzs = np.stack([n["base_xyz"] for n in raw_nodes], axis=0)
    if normalize:
        norm_xyzs, xyz_min, xyz_scale = _normalize_xyz(base_xyzs)
    else:
        norm_xyzs = base_xyzs
        xyz_min = np.zeros(3, dtype=np.float32)
        xyz_scale = np.ones(3, dtype=np.float32)

    max_shoot_id = max((n["shoot_id"] for n in raw_nodes), default=0)
    max_phytomer_idx = max((n["phytomer_idx"] for n in raw_nodes), default=1)

    for i, n in enumerate(raw_nodes):
        nodes_15d[i, 0:3] = norm_xyzs[i]
        nodes_15d[i, 3] = _normalize_length(n["length"])
        nodes_15d[i, 4] = _normalize_radius(n["radius"])
        nodes_15d[i, 5] = _normalize_angle(n["pitch"])
        nodes_15d[i, 6] = _normalize_angle(n["yaw"])
        nodes_15d[i, 7] = _normalize_angle(n["roll"])
        nodes_15d[i, 8:12] = n["one_hot"]
        nodes_15d[i, 12] = n["shoot_id"] / max(1.0, float(max_shoot_id))
        nodes_15d[i, 13] = n["phytomer_idx"] / max(1.0, float(max_phytomer_idx))
        nodes_15d[i, 14] = 1.0

        organ_types[i] = n["organ_type"]
        shoot_ids[i] = n["shoot_id"]
        phytomer_indices[i] = n["phytomer_idx"]
        existence_mask[i] = 1.0

    # Build parent indices using the hierarchy
    # We need to know which raw node is parent for each child.
    # We do a second pass now that indices are fixed.
    parent_indices[0] = 0  # root self-loop

    # Re-derive parent mapping by matching node metadata with the order we added.
    # Because nodes were added in DFS order, the parent for each organ is known
    # from the construction state. We can recompute by scanning raw_nodes.
    # Simpler: re-run the logic using a small state machine.
    # For correctness, we will build parent map during the parse using the index
    # returned by add_node. We already lost it; reconstruct by tracking.
    _build_parent_indices(raw_nodes, parent_indices, adj_matrix, existence_mask)

    return {
        "nodes": nodes_15d,
        "adj_matrix": adj_matrix,
        "parent_indices": parent_indices,
        "existence_mask": existence_mask,
        "organ_types": organ_types,
        "shoot_ids": shoot_ids,
        "phytomer_indices": phytomer_indices,
        "dap": dap,
        "num_nodes": num_nodes,
        "xyz_min": xyz_min,
        "xyz_scale": xyz_scale,
    }


def _build_parent_indices(raw_nodes: List[Dict[str, Any]],
                          parent_indices: np.ndarray,
                          adj_matrix: np.ndarray,
                          existence_mask: np.ndarray) -> None:
    """Reconstruct parent indices from raw node list.

    We reconstruct by matching the hierarchical construction order:
      - The first node of each shoot is the shoot base.
      - Internodes are sequential on a shoot.
      - Petioles belong to the current internode.
      - Leaves / floral_buds belong to the current petiole.
    """
    # Track current parent for each organ type per shoot/phytomer.
    current_shoot_base: Dict[int, int] = {}
    current_internode: Dict[Tuple[int, int], int] = {}
    current_petiole: int = -1
    prev_shoot_id: int = -1
    prev_phytomer_idx: int = -1

    for idx, n in enumerate(raw_nodes):
        if idx == 0:
            parent_indices[idx] = 0
            current_shoot_base[n["shoot_id"]] = idx
            continue

        organ = n["organ_type"]
        shoot_id = n["shoot_id"]
        phytomer_idx = n["phytomer_idx"]

        if organ == ORGAN_TYPE_MAP["internode"]:
            # Shoot base has length==0 and is the first internode-like node of a shoot.
            if n["length"] == 0.0:
                # This is a shoot base node.
                if shoot_id == 0:
                    parent_indices[idx] = 0
                else:
                    # Attach to last internode of parent shoot.
                    # Find any internode from the parent shoot with highest phytomer index.
                    parent_idx = 0
                    parent_shoot_id = -1
                    for candidate in reversed(raw_nodes[:idx]):
                        if candidate["shoot_id"] != shoot_id and candidate["organ_type"] == ORGAN_TYPE_MAP["internode"]:
                            if candidate["shoot_id"] != parent_shoot_id:
                                parent_shoot_id = candidate["shoot_id"]
                                parent_idx = raw_nodes.index(candidate)
                                break
                    parent_indices[idx] = parent_idx
                current_shoot_base[shoot_id] = idx
                current_internode[(shoot_id, phytomer_idx)] = idx
            else:
                # True internode: attach to previous internode or shoot base.
                key = (shoot_id, phytomer_idx)
                # If phytomer_idx == 0 or 1, previous is shoot base.
                if phytomer_idx <= 1:
                    parent = current_shoot_base.get(shoot_id, 0)
                else:
                    parent = current_internode.get((shoot_id, phytomer_idx - 1),
                                                    current_shoot_base.get(shoot_id, 0))
                parent_indices[idx] = parent
                current_internode[key] = idx

        elif organ == ORGAN_TYPE_MAP["petiole"]:
            # Attach to current internode.
            parent = current_internode.get((shoot_id, phytomer_idx), 0)
            parent_indices[idx] = parent
            current_petiole = idx
            prev_shoot_id = shoot_id
            prev_phytomer_idx = phytomer_idx

        elif organ in (ORGAN_TYPE_MAP["leaf"], ORGAN_TYPE_MAP["floral_bud"]):
            # Attach to most recent petiole with matching shoot/phytomer.
            # Fallback: current internode.
            if shoot_id == prev_shoot_id and phytomer_idx == prev_phytomer_idx and current_petiole != -1:
                parent = current_petiole
            else:
                parent = current_internode.get((shoot_id, phytomer_idx), 0)
            parent_indices[idx] = parent

        if parent_indices[idx] != idx and existence_mask[parent_indices[idx]] > 0.5:
            adj_matrix[parent_indices[idx], idx] = 1.0
            adj_matrix[idx, parent_indices[idx]] = 1.0


def denormalize_xyz(norm_xyz: np.ndarray, xyz_min: np.ndarray, xyz_scale: np.ndarray) -> np.ndarray:
    """Convert normalized [0,1] coordinates back to metric coordinates."""
    return norm_xyz * xyz_scale + xyz_min
