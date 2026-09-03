"""
Autonomous Part Assembly to Helios XML Converter (Disentangled 13D Analytical IK Edition).

Reconstructs a fully valid, standalone Helios XML plant architecture document
from an unorganized (N, 13) spatial part tensor without requiring any original
XML template.

Every column of the 13D part tensor has strictly one immutable physical meaning:
    [organ_type(1), base_xyz(3), rot6d(6), scale_xyz(3)] = 13D
Relative Euler angles (pitch, yaw, roll) are computed analytically via inverse
kinematics (R_rel = R_parent^T * R_child).
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from scipy.spatial import cKDTree

from diffusion_based.models.plant_organ_array import (
    ORGAN_NONE,
    ORGAN_ROOT_META,
    ORGAN_SHOOT_META,
    ORGAN_INTERNODE,
    ORGAN_PETIOLE,
    ORGAN_LEAF,
    ORGAN_PEDUNCLE,
    ORGAN_BUD_DORMANT,
    ORGAN_BUD_ACTIVE,
    ORGAN_FLOWER_CLOSED,
    ORGAN_FLOWER_OPEN,
    ORGAN_FRUIT,
    ORGAN_BUD_ABORTED,
    P_COL_ORGAN_TYPE,
    P_COL_BASE_X,
    P_COL_BASE_Y,
    P_COL_BASE_Z,
    P_COL_ROT_0,
    P_COL_ROT_5,
    P_COL_SCALE_X,
    P_COL_SCALE_Y,
    P_COL_SCALE_Z,
    NUM_FEATURES_PART,
    rotation_6d_to_matrix,
)


def _fmt(val: float, precision: int = 7) -> str:
    """Format floating point values cleanly for XML."""
    if abs(val) < 1e-12:
        return "0"
    s = f"{val:.{precision}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _matrix_to_euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """Decompose 3x3 rotation matrix R = Rz(az) * Ry(pitch) * Rx(roll) into radians."""
    pitch = -math.asin(np.clip(R[2, 0], -1.0, 1.0))
    cp = math.cos(pitch)
    if abs(cp) > 1e-5:
        roll = math.atan2(R[2, 1], R[2, 2])
        az = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        az = 0.0
    return roll, pitch, az


class PartAssemblyToXMLConverter:
    """Autonomous converter from (N, 13) part tensor to Helios XML."""

    def __init__(self, connectivity_tolerance: float = 0.08):
        self.tol = connectivity_tolerance

    def convert_to_xml_string(
        self,
        part_tensor: torch.Tensor,
        plant_id: int = 0,
        plant_type: str = "cowpea",
        existence_threshold: float = 0.5,
    ) -> str:
        """Converts (N, 13) part tensor to a valid Helios XML string."""
        p_np = part_tensor.detach().cpu().numpy()
        N = p_np.shape[0]

        # 1. Separate active parts by organ type
        # In 13D: 0 is ORGAN_NONE, active is ot > 0
        ot_all = np.round(p_np[:, P_COL_ORGAN_TYPE]).astype(int)
        active_mask = ot_all > ORGAN_NONE
        
        root_meta_idx = None
        shoot_metas = []
        internodes = []
        petioles = []
        leaves = []
        peduncles = []
        flowers = []
        fruits = []
        buds = []

        for idx in range(N):
            if not active_mask[idx]:
                continue
            ot = ot_all[idx]
            if ot == ORGAN_ROOT_META:
                root_meta_idx = idx
            elif ot == ORGAN_SHOOT_META:
                shoot_metas.append(idx)
            elif ot in (ORGAN_INTERNODE, 2):
                internodes.append(idx)
            elif ot in (ORGAN_PETIOLE, 3):
                petioles.append(idx)
            elif ot in (ORGAN_LEAF, 4, 5):
                leaves.append(idx)
            elif ot in (ORGAN_PEDUNCLE, 6):
                peduncles.append(idx)
            elif ot in (ORGAN_FLOWER_OPEN, ORGAN_FLOWER_CLOSED, 7, 9, 10):
                flowers.append(idx)
            elif ot in (ORGAN_FRUIT, 8, 11):
                fruits.append(idx)
            elif ot in (ORGAN_BUD_DORMANT, ORGAN_BUD_ACTIVE, ORGAN_BUD_ABORTED, 12):
                buds.append(idx)

        # Batch 6D rotation conversion for speed
        r6_all = torch.from_numpy(p_np[:, P_COL_ROT_0:P_COL_ROT_5+1]).float()
        R_all = rotation_6d_to_matrix(r6_all).numpy()

        part_info = {}
        for idx in range(N):
            base = p_np[idx, P_COL_BASE_X:P_COL_BASE_Z+1]
            R = R_all[idx]
            sx = float(p_np[idx, P_COL_SCALE_X])
            sy = float(p_np[idx, P_COL_SCALE_Y])
            sz = float(p_np[idx, P_COL_SCALE_Z])
            dir_fwd = R @ np.array([0.0, 1.0, 0.0])
            tip = base + dir_fwd * sx
            part_info[idx] = {
                "ot": ot_all[idx],
                "base": base,
                "tip": tip,
                "R": R,
                "dir": dir_fwd,
                "sx": sx,
                "sy": sy,
                "sz": sz,
                "orig_idx": idx,
            }

        # Reconstruct the record-order grouping
        phytomer_groups: List[Dict[str, Any]] = []
        current_group: Optional[Dict[str, Any]] = None
        current_petiole: Optional[int] = None
        current_shoot_id = -1
        for idx in range(N):
            if not active_mask[idx]:
                continue
            ot = ot_all[idx]
            if ot == ORGAN_ROOT_META:
                continue
            if ot == ORGAN_SHOOT_META:
                current_shoot_id += 1
                current_group = None
                current_petiole = None
                continue
            if ot in (ORGAN_INTERNODE, 2):
                current_group = {
                    "shoot_id": max(0, current_shoot_id),
                    "internode": idx,
                    "petioles": [],
                    "peduncle": None,
                    "flowers": [],
                    "fruits": [],
                    "bud": None,
                }
                phytomer_groups.append(current_group)
                current_petiole = None
                continue
            if current_group is None:
                continue
            if ot in (ORGAN_PETIOLE, 3):
                pet_entry = {"idx": idx, "leaves": []}
                current_group["petioles"].append(pet_entry)
                current_petiole = len(current_group["petioles"]) - 1
                continue
            if ot in (ORGAN_LEAF, 4, 5):
                if current_petiole is not None:
                    current_group["petioles"][current_petiole]["leaves"].append(idx)
                continue
            if ot in (ORGAN_BUD_DORMANT, ORGAN_BUD_ACTIVE, ORGAN_BUD_ABORTED, 12):
                current_group["bud"] = idx
                continue
            if ot in (ORGAN_PEDUNCLE, 6):
                current_group["peduncle"] = idx
                continue
            if ot in (ORGAN_FLOWER_OPEN, ORGAN_FLOWER_CLOSED, 7, 9, 10):
                current_group["flowers"].append(idx)
                continue
            if ot in (ORGAN_FRUIT, 8, 11):
                current_group["fruits"].append(idx)
                continue

        if not internodes:
            return '<?xml version="1.0" encoding="UTF-8"?>\\n<helios>\\n\\t<plant_instance id="0">\\n\\t\\t<plant_base_position>0;0;0</plant_base_position>\\n\\t\\t<plant_type>cowpea</plant_type>\\n\\t</plant_instance>\\n</helios>\\n'

        # 2. Reconstruct Stem / Shoot Graph
        if shoot_metas:
            shoot_groups: Dict[int, List[int]] = {}
            for grp in phytomer_groups:
                sid = grp["shoot_id"]
                shoot_groups.setdefault(sid, []).append(grp["internode"])
            shoots = [shoot_groups[sid] for sid in sorted(shoot_groups.keys())]
            inode_parent = {i: None for i in internodes}
            inode_children: Dict[int, List[int]] = {i: [] for i in internodes}
            for s_idx, sh in enumerate(shoots):
                if s_idx == 0:
                    continue
                first_base = part_info[sh[0]]["base"]
                best_parent = None
                best_dist = float("inf")
                for cand_sid, cand_sh in enumerate(shoots[:s_idx]):
                    for cand_inode in cand_sh:
                        d = float(np.linalg.norm(part_info[cand_inode]["tip"] - first_base))
                        if d < best_dist and d < self.tol:
                            best_dist = d
                            best_parent = cand_inode
                if best_parent is not None:
                    inode_parent[sh[0]] = best_parent
                    inode_children[best_parent].append(sh[0])
        else:
            internodes.sort(key=lambda i: part_info[i]["base"][2])
            inode_children = {i: [] for i in internodes}
            inode_parent = {i: None for i in internodes}

            inode_tips = np.array([part_info[i]["tip"] for i in internodes])
            inode_tree = cKDTree(inode_tips)

            for i in internodes:
                i_base = part_info[i]["base"]
                dists, idxs = inode_tree.query(i_base, k=min(5, len(internodes)))
                if not isinstance(dists, np.ndarray):
                    dists, idxs = [dists], [idxs]
                for d, idx_cand in zip(dists, idxs):
                    p_cand = internodes[idx_cand]
                    if p_cand != i and d < self.tol:
                        inode_parent[i] = p_cand
                        inode_children[p_cand].append(i)
                        break

            root_inodes = [i for i in internodes if inode_parent[i] is None]
            if not root_inodes:
                root_inodes = [internodes[0]]

            shoots: List[List[int]] = []
            visited = set()

            def trace_shoot(start_inode: int) -> List[int]:
                path = []
                curr = start_inode
                while curr is not None and curr not in visited:
                    visited.add(curr)
                    path.append(curr)
                    children = inode_children.get(curr, [])
                    if not children:
                        break
                    curr_dir = part_info[curr]["dir"]
                    child_scores = [np.dot(curr_dir, part_info[c]["dir"]) for c in children]
                    best_c_idx = int(np.argmax(child_scores))
                    for ci, c in enumerate(children):
                        if ci != best_c_idx and c not in visited:
                            branch_path = trace_shoot(c)
                            if branch_path:
                                shoots.append(branch_path)
                    curr = children[best_c_idx]
                return path

            for r in root_inodes:
                main_path = trace_shoot(r)
                if main_path:
                    shoots.insert(0, main_path)

            for i in internodes:
                if i not in visited:
                    p = trace_shoot(i)
                    if p:
                        shoots.append(p)

        # 3. Associate Petioles, Leaves, Peduncles, Flowers
        phytomer_parts = {grp["internode"]: {"petioles": [p["idx"] for p in grp["petioles"]],
                                              "peduncles": ([grp["peduncle"]] if grp["peduncle"] is not None else [])}
                          for grp in phytomer_groups}
        petiole_leaves = {p["idx"]: p["leaves"] for grp in phytomer_groups for p in grp["petioles"]}
        bud_state_by_inode = {}
        for grp in phytomer_groups:
            if grp["bud"] is not None:
                bidx = grp["bud"]
                bot = part_info[bidx]["ot"]
                if bot == ORGAN_BUD_DORMANT:
                    bs = 0
                elif bot == ORGAN_BUD_ACTIVE:
                    bs = 1
                elif bot == ORGAN_FLOWER_CLOSED:
                    bs = 2
                elif bot == ORGAN_FLOWER_OPEN:
                    bs = 3
                elif bot == ORGAN_FRUIT:
                    bs = 4
                elif bot == ORGAN_BUD_ABORTED:
                    bs = 5
                else:
                    bs = 1
                bud_state_by_inode[grp["internode"]] = bs

        peduncle_infls = {grp["peduncle"]: grp["flowers"] + grp["fruits"]
                          for grp in phytomer_groups if grp["peduncle"] is not None}
        peduncle_bud = {grp["peduncle"]: grp["bud"]
                        for grp in phytomer_groups if grp["peduncle"] is not None and grp["bud"] is not None}

        # 4. Serialize to Helios XML
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<helios>']
        lines.append(f'	<plant_instance ID="{plant_id}">')

        root_pos = part_info[root_meta_idx]["base"] if root_meta_idx is not None else (part_info[internodes[0]]["base"] if internodes else np.zeros(3))
        plant_age = part_info[root_meta_idx]["sx"] if root_meta_idx is not None else 20.0
        lines.append(f'		<base_position> {_fmt(root_pos[0])} {_fmt(root_pos[1])} {_fmt(root_pos[2])} </base_position>')
        lines.append(f'		<plant_age> {_fmt(plant_age if plant_age > 0 else 20.0)} </plant_age>')

        inode_to_location: Dict[int, Tuple[int, int]] = {}
        for _si, _sh in enumerate(shoots):
            for _li, _inode in enumerate(_sh):
                inode_to_location[_inode] = (_si, _li)

        for s_idx, shoot_inodes in enumerate(shoots):
            p_shoot_id = -1 if s_idx == 0 else 0
            p_node_idx = 0
            p_pet_idx = 0
            if s_idx > 0:
                first_i = shoot_inodes[0]
                parent_i = inode_parent.get(first_i)
                if parent_i is not None and parent_i in inode_to_location:
                    p_shoot_id, p_node_idx = inode_to_location[parent_i]

            shoot_label = "unifoliate" if s_idx == 0 else "trifoliate"
            lines.append(f'		<shoot ID="{s_idx}">')
            lines.append(f'			<shoot_type_label> {shoot_label} </shoot_type_label>')
            lines.append(f'			<parent_shoot_ID> {p_shoot_id} </parent_shoot_ID>')
            lines.append(f'			<parent_node_index> {p_node_idx} </parent_node_index>')
            lines.append(f'			<parent_petiole_index> {p_pet_idx} </parent_petiole_index>')

            # Compute shoot base rotation analytically
            if s_idx == 0:
                R_shoot = part_info[shoot_inodes[0]]["R"]
            else:
                parent_i = inode_parent.get(shoot_inodes[0])
                if parent_i is not None:
                    R_parent = part_info[parent_i]["R"]
                    R_child = part_info[shoot_inodes[0]]["R"]
                    R_shoot = R_parent.T @ R_child
                else:
                    R_shoot = part_info[shoot_inodes[0]]["R"]

            roll_rad, pitch_rad, yaw_rad = _matrix_to_euler_xyz(R_shoot)
            base_pitch = math.degrees(pitch_rad)
            base_yaw = math.degrees(yaw_rad)
            base_roll = math.degrees(roll_rad)
            lines.append(f'			<base_rotation> {_fmt(base_pitch)} {_fmt(base_yaw)} {_fmt(base_roll)} </base_rotation>')

            for node_i, inode_idx in enumerate(shoot_inodes):
                info = part_info[inode_idx]
                inode_len = max(info["sx"], 0.001)
                inode_rad = max(info["sy"], 0.0005)

                lines.append('			<phytomer>')
                lines.append('				<internode>')
                lines.append(f'					<internode_length>{_fmt(inode_len)}</internode_length>')
                lines.append(f'					<internode_radius>{_fmt(inode_rad)}</internode_radius>')
                lines.append(f'					<internode_pitch>0</internode_pitch>')
                lines.append(f'					<internode_phyllotactic_angle>137.5</internode_phyllotactic_angle>')
                lines.append(f'					<internode_length_max>{_fmt(inode_len)}</internode_length_max>')
                lines.append(f'					<internode_length_segments>1</internode_length_segments>')
                lines.append('					<curvature_perturbations>0;0</curvature_perturbations>')
                lines.append('					<yaw_perturbations>0;0</yaw_perturbations>')

                # Petioles & Leaves
                pets = phytomer_parts.get(inode_idx, {}).get("petioles", [])
                if not pets:
                    lines.append('					<petiole>')
                    lines.append('						<petiole_length>0.05</petiole_length>')
                    lines.append('						<petiole_radius>0.001</petiole_radius>')
                    lines.append('						<petiole_pitch>45</petiole_pitch>')
                    lines.append('						<petiole_curvature>0</petiole_curvature>')
                    lines.append('						<current_leaf_scale_factor>1</current_leaf_scale_factor>')
                    lines.append('						<petiole_taper>0</petiole_taper>')
                    lines.append('						<petiole_length_segments>3</petiole_length_segments>')
                    lines.append('						<petiole_radial_subdivisions>6</petiole_radial_subdivisions>')
                    lines.append('						<leaflet_scale>1</leaflet_scale>')
                    lines.append('						<leaflet_offset>0</leaflet_offset>')
                    lines.append('					</petiole>')
                else:
                    for pet_idx in pets:
                        p_info = part_info[pet_idx]
                        pet_leaves = petiole_leaves.get(pet_idx, [])
                        
                        pet_len = max(p_info["sx"], 0.001)
                        pet_rad = max(p_info["sy"], 0.0005)

                        # Compute petiole relative pitch from internode frame
                        R_rel_pet = info["R"].T @ p_info["R"]
                        _, pet_pitch_r, _ = _matrix_to_euler_xyz(R_rel_pet)
                        pet_pitch_deg = math.degrees(pet_pitch_r)

                        lines.append('					<petiole>')
                        lines.append(f'						<petiole_length>{_fmt(pet_len)}</petiole_length>')
                        lines.append(f'						<petiole_radius>{_fmt(pet_rad)}</petiole_radius>')
                        lines.append(f'						<petiole_pitch>{_fmt(pet_pitch_deg if abs(pet_pitch_deg) > 1e-3 else 45.0)}</petiole_pitch>')
                        lines.append(f'						<petiole_curvature>0</petiole_curvature>')
                        lines.append(f'						<current_leaf_scale_factor>1</current_leaf_scale_factor>')
                        lines.append('						<petiole_taper>0.25</petiole_taper>')
                        lines.append('						<petiole_length_segments>3</petiole_length_segments>')
                        lines.append('						<petiole_radial_subdivisions>6</petiole_radial_subdivisions>')
                        leaflet_scale = 1.0 if len(pet_leaves) == 1 else 0.9
                        lines.append(f'						<leaflet_scale>{_fmt(leaflet_scale)}</leaflet_scale>')
                        lines.append('						<leaflet_offset>0.4</leaflet_offset>')

                        for lf_idx in pet_leaves:
                            lf_info = part_info[lf_idx]
                            leaf_scale = max(lf_info["sx"], 0.001)

                            # Analytical inverse Euler angles from Petiole frame
                            R_rel_lf = p_info["R"].T @ lf_info["R"]
                            roll_r, pitch_r, yaw_r = _matrix_to_euler_xyz(R_rel_lf)
                            leaf_pitch = math.degrees(pitch_r)
                            leaf_yaw = math.degrees(yaw_r)
                            leaf_roll = math.degrees(roll_r)

                            lines.append('						<leaf>')
                            lines.append(f'							<leaf_scale>{_fmt(leaf_scale)}</leaf_scale>')
                            lines.append(f'							<leaf_pitch>{_fmt(leaf_pitch)}</leaf_pitch>')
                            lines.append(f'							<leaf_yaw>{_fmt(leaf_yaw)}</leaf_yaw>')
                            lines.append(f'							<leaf_roll>{_fmt(leaf_roll)}</leaf_roll>')
                            lines.append('						</leaf>')

                        # Buds, Peduncles, and Inflorescences
                        peds = phytomer_parts.get(inode_idx, {}).get("peduncles", [])
                        has_bud = inode_idx in bud_state_by_inode
                        if peds:
                            for pd_idx in peds:
                                pd_info = part_info[pd_idx]
                                infls = peduncle_infls.get(pd_idx, [])
                                infl_ots = [part_info[fl]["ot"] for fl in infls]
                                if any(ot in (ORGAN_FRUIT, 8, 11) for ot in infl_ots):
                                    bud_state = 4
                                elif any(ot in (ORGAN_FLOWER_CLOSED, 9) for ot in infl_ots):
                                    bud_state = 2
                                elif any(ot in (ORGAN_FLOWER_OPEN, 7, 10) for ot in infl_ots):
                                    bud_state = 3
                                else:
                                    bud_state = bud_state_by_inode.get(inode_idx, 1)

                                bud_idx = peduncle_bud.get(pd_idx)
                                fruit_scale = float(part_info[bud_idx]["sx"]) if bud_idx is not None else 1.0

                                pd_len = max(pd_info["sx"], 0.001)
                                pd_rad = max(pd_info["sy"], 0.0005)

                                R_rel_pd = info["R"].T @ pd_info["R"]
                                roll_r, pitch_r, _ = _matrix_to_euler_xyz(R_rel_pd)
                                pd_pitch = math.degrees(pitch_r)
                                pd_roll = math.degrees(roll_r)

                                lines.append('						<floral_bud>')
                                lines.append(f'							<bud_state>{bud_state}</bud_state>')
                                lines.append('							<parent_index>0</parent_index>')
                                lines.append('							<bud_index>0</bud_index>')
                                lines.append('							<is_terminal>0</is_terminal>')
                                lines.append(f'							<current_fruit_scale_factor>{_fmt(fruit_scale if fruit_scale > 0 else 1.0)}</current_fruit_scale_factor>')
                                lines.append('							<peduncle>')
                                lines.append(f'								<length>{_fmt(pd_len)}</length>')
                                lines.append(f'								<radius>{_fmt(pd_rad)}</radius>')
                                lines.append(f'								<pitch>{_fmt(pd_pitch)}</pitch>')
                                lines.append(f'								<curvature>0</curvature>')
                                lines.append(f'								<roll>{_fmt(pd_roll)}</roll>')
                                lines.append('							</peduncle>')

                                if infls:
                                    lines.append('							<inflorescence>')
                                    lines.append('								<flower_offset>0.05</flower_offset>')
                                    for fl_idx in infls:
                                        fl_info = part_info[fl_idx]
                                        fl_scale = max(fl_info["sx"], 0.001)

                                        R_rel_fl = pd_info["R"].T @ fl_info["R"]
                                        roll_r, pitch_r, yaw_r = _matrix_to_euler_xyz(R_rel_fl)
                                        fl_pitch = math.degrees(pitch_r)
                                        fl_yaw = math.degrees(yaw_r)
                                        fl_roll = math.degrees(roll_r)

                                        lines.append('								<flower>')
                                        lines.append(f'									<flower_pitch>{_fmt(fl_pitch)}</flower_pitch>')
                                        lines.append(f'									<flower_yaw>{_fmt(fl_yaw)}</flower_yaw>')
                                        lines.append(f'									<flower_roll>{_fmt(fl_roll)}</flower_roll>')
                                        lines.append(f'									<flower_azimuth>0</flower_azimuth>')
                                        lines.append(f'									<flower_base_scale>{_fmt(fl_scale)}</flower_base_scale>')
                                        lines.append('								</flower>')
                                    lines.append('							</inflorescence>')
                                lines.append('						</floral_bud>')
                        elif has_bud:
                            bud_state = bud_state_by_inode.get(inode_idx, 1)
                            lines.append('						<floral_bud>')
                            lines.append(f'							<bud_state>{bud_state}</bud_state>')
                            lines.append('							<parent_index>0</parent_index>')
                            lines.append('							<bud_index>0</bud_index>')
                            lines.append('							<is_terminal>0</is_terminal>')
                            lines.append('							<current_fruit_scale_factor>1</current_fruit_scale_factor>')
                            lines.append('						</floral_bud>')

                        lines.append('					</petiole>')

                lines.append('				</internode>')
                lines.append('			</phytomer>')

            lines.append('		</shoot>')

        lines.append('	</plant_instance>')
        lines.append('</helios>\n')
        return "\n".join(lines) + "\n"


def assemble_part_tensor_to_xml(
    part_tensor: torch.Tensor,
    plant_id: int = 0,
    plant_type: str = "cowpea",
    existence_threshold: float = 0.5,
) -> str:
    """Convenience helper to convert a 13D part tensor into a Helios XML string."""
    converter = PartAssemblyToXMLConverter()
    return converter.convert_to_xml_string(
        part_tensor,
        plant_id=plant_id,
        plant_type=plant_type,
        existence_threshold=existence_threshold,
    )
