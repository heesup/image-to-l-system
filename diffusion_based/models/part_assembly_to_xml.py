"""
Autonomous 14D Part Assembly to Helios XML Converter (Ultra-Fast Vectorized).

Reconstructs a fully valid, standalone Helios XML plant architecture document
from an unorganized (N, 14) spatial part tensor without requiring any original
XML template.

Pipeline:
  1. Spatial Graph & Topological Connectivity Inference (using cKDTree)
  2. Inverse Kinematics (Euler Angles, Pitch, Yaw, Roll, Phyllotaxis)
  3. Strict Helios XML Schema Serialization
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from scipy.spatial import cKDTree

from diffusion_based.models.plant_organ_array import (
    P_COL_ORGAN_TYPE,
    P_COL_BASE_X,
    P_COL_BASE_Y,
    P_COL_BASE_Z,
    P_COL_ROT_0,
    P_COL_ROT_5,
    P_COL_SCALE_X,
    P_COL_SCALE_Y,
    P_COL_SCALE_Z,
    P_COL_EXISTENCE,
    P_COL_BUD_STATE,
    P_COL_CURVATURE,
    P_COL_PHYLLOTACTIC_ANGLE,
    ORGAN_ROOT_META,
    ORGAN_SHOOT_META,
    ORGAN_LEAF,
    ORGAN_PETIOLE,
    ORGAN_BUD,
    ORGAN_INTERNODE,
    ORGAN_PEDUNCLE,
    ORGAN_FLOWER,
    ORGAN_FRUIT,
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
    """Autonomous converter from (N, 14) part tensor to Helios XML."""

    def __init__(self, connectivity_tolerance: float = 0.08):
        self.tol = connectivity_tolerance

    def convert_to_xml_string(
        self,
        part_tensor_14d: torch.Tensor,
        plant_id: int = 0,
        plant_type: str = "cowpea",
        existence_threshold: float = 0.5,
    ) -> str:
        """Converts (N, 14) part tensor to a valid Helios XML string."""
        p_np = part_tensor_14d.detach().cpu().numpy()
        N = p_np.shape[0]

        # 1. Separate active parts by organ type
        active_mask = p_np[:, P_COL_EXISTENCE] >= existence_threshold
        
        root_meta_idx = None
        shoot_metas = []
        internodes = []
        petioles = []
        leaves = []
        peduncles = []
        flowers = []
        fruits = []

        for idx in range(N):
            if not active_mask[idx]:
                continue
            ot = int(round(p_np[idx, P_COL_ORGAN_TYPE]))
            if ot == ORGAN_ROOT_META:
                root_meta_idx = idx
            elif ot == ORGAN_SHOOT_META:
                shoot_metas.append(idx)
            elif ot == ORGAN_INTERNODE:
                internodes.append(idx)
            elif ot == ORGAN_PETIOLE:
                petioles.append(idx)
            elif ot == ORGAN_LEAF:
                leaves.append(idx)
            elif ot == ORGAN_PEDUNCLE:
                peduncles.append(idx)
            elif ot == ORGAN_FLOWER:
                flowers.append(idx)
            elif ot in (ORGAN_FRUIT, 8):
                fruits.append(idx)

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
            dir_z = R @ np.array([0.0, 0.0, 1.0])
            tip = base + dir_z * sz
            part_info[idx] = {
                "ot": int(round(p_np[idx, P_COL_ORGAN_TYPE])),
                "base": base,
                "tip": tip,
                "R": R,
                "dir": dir_z,
                "sx": sx,
                "sy": sy,
                "sz": sz,
                "bud_state": int(round(p_np[idx, P_COL_BUD_STATE])),
                "curvature": float(p_np[idx, P_COL_CURVATURE]),
                "phyllotactic_angle": float(p_np[idx, P_COL_PHYLLOTACTIC_ANGLE]),
            }

        if not internodes:
            # Fallback for empty plant
            return '<?xml version="1.0" encoding="UTF-8"?>\n<helios>\n\t<plant_instance id="0">\n\t\t<plant_base_position>0;0;0</plant_base_position>\n\t\t<plant_type>cowpea</plant_type>\n\t</plant_instance>\n</helios>\n'

        # 2. Reconstruct Stem / Shoot Graph from Internodes using cKDTree
        # Sort internodes by base Z height
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

        # 3. Associate Petioles, Leaves, Peduncles, Flowers using cKDTree
        phytomer_parts = {i: {"petioles": [], "peduncles": []} for i in internodes}

        # Associate Petioles to closest internode base
        inode_bases = np.array([part_info[i]["base"] for i in internodes])
        inode_base_tree = cKDTree(inode_bases)

        if petioles:
            pet_bases = np.array([part_info[p]["base"] for p in petioles])
            _, nearest_inodes = inode_base_tree.query(pet_bases)
            if not isinstance(nearest_inodes, np.ndarray):
                nearest_inodes = [nearest_inodes]
            for p_idx, i_local_idx in zip(petioles, nearest_inodes):
                best_inode = internodes[i_local_idx]
                phytomer_parts[best_inode]["petioles"].append(p_idx)

        # Associate Leaves to closest petiole tip
        petiole_leaves = {p: [] for p in petioles}
        if leaves and petioles:
            pet_tips = np.array([part_info[p]["tip"] for p in petioles])
            pet_tree = cKDTree(pet_tips)
            leaf_bases = np.array([part_info[l]["base"] for l in leaves])
            _, nearest_pets = pet_tree.query(leaf_bases)
            if not isinstance(nearest_pets, np.ndarray):
                nearest_pets = [nearest_pets]
            for l_idx, p_local_idx in zip(leaves, nearest_pets):
                best_pet = petioles[p_local_idx]
                petiole_leaves[best_pet].append(l_idx)

        # Associate Peduncles to closest internode tip (bud base == internode tip)
        if peduncles:
            inode_tips = np.array([part_info[i]["tip"] for i in internodes])
            inode_tip_tree = cKDTree(inode_tips)
            pd_bases = np.array([part_info[pd]["base"] for pd in peduncles])
            _, nearest_pd_inodes = inode_tip_tree.query(pd_bases)
            if not isinstance(nearest_pd_inodes, np.ndarray):
                nearest_pd_inodes = [nearest_pd_inodes]
            for pd_idx, i_local_idx in zip(peduncles, nearest_pd_inodes):
                best_inode = internodes[i_local_idx]
                phytomer_parts[best_inode]["peduncles"].append(pd_idx)

        # Associate Flowers & Fruits to closest peduncle tip
        peduncle_infls = {pd: [] for pd in peduncles}
        all_infls = flowers + fruits
        if all_infls and peduncles:
            pd_tips = np.array([part_info[pd]["tip"] for pd in peduncles])
            pd_tree = cKDTree(pd_tips)
            fl_bases = np.array([part_info[fl]["base"] for fl in all_infls])
            _, nearest_pds = pd_tree.query(fl_bases)
            if not isinstance(nearest_pds, np.ndarray):
                nearest_pds = [nearest_pds]
            for fl_idx, pd_local_idx in zip(all_infls, nearest_pds):
                best_pd = peduncles[pd_local_idx]
                peduncle_infls[best_pd].append(fl_idx)

        # 4. Serialize to Helios XML
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<helios>']
        lines.append(f'\t<plant_instance id="{plant_id}">')

        root_pos = part_info[root_meta_idx]["base"] if root_meta_idx is not None else (part_info[internodes[0]]["base"] if internodes else np.zeros(3))
        lines.append(f'\t\t<plant_base_position>{_fmt(root_pos[0])};{_fmt(root_pos[1])};{_fmt(root_pos[2])}</plant_base_position>')
        lines.append(f'\t\t<plant_type>{plant_type}</plant_type>')

        for s_idx, shoot_inodes in enumerate(shoots):
            p_shoot_id = -1 if s_idx == 0 else 0
            p_node_idx = 0
            if s_idx > 0:
                first_i = shoot_inodes[0]
                parent_i = inode_parent.get(first_i)
                if parent_i is not None and parent_i in internodes:
                    p_node_idx = internodes.index(parent_i)

            lines.append(f'\t\t<shoot shoot_id="{s_idx}">')
            lines.append(f'\t\t\t<parent_shoot_id>{p_shoot_id}</parent_shoot_id>')
            lines.append(f'\t\t\t<parent_node_index>{p_node_idx}</parent_node_index>')
            lines.append('\t\t\t<shoot_type_label>vegetative</shoot_type_label>')
            lines.append('\t\t\t<shoot_base_pitch>0</shoot_base_pitch>')
            lines.append('\t\t\t<shoot_base_roll>0</shoot_base_roll>')
            lines.append('\t\t\t<shoot_base_yaw>0</shoot_base_yaw>')

            prev_dir = np.array([0.0, 0.0, 1.0])
            for node_i, inode_idx in enumerate(shoot_inodes):
                info = part_info[inode_idx]
                curr_dir = info["dir"]
                
                # Inverse pitch & phyllotactic angle
                cos_pitch = np.clip(np.dot(prev_dir, curr_dir), -1.0, 1.0)
                inode_pitch_deg = math.degrees(math.acos(cos_pitch))
                prev_dir = curr_dir

                lines.append('\t\t\t<phytomer>')
                lines.append('\t\t\t\t<internode>')
                lines.append(f'\t\t\t\t\t<internode_length>{_fmt(info["sz"])}</internode_length>')
                lines.append(f'\t\t\t\t\t<internode_radius>{_fmt(info["sx"])}</internode_radius>')
                lines.append(f'\t\t\t\t\t<internode_pitch>{_fmt(inode_pitch_deg)}</internode_pitch>')
                phyllo = info["phyllotactic_angle"] if info["phyllotactic_angle"] > 0 else 137.5
                lines.append(f'\t\t\t\t\t<internode_phyllotactic_angle>{_fmt(phyllo)}</internode_phyllotactic_angle>')
                lines.append(f'\t\t\t\t\t<internode_length_max>{_fmt(info["sz"])}</internode_length_max>')
                lines.append('\t\t\t\t\t<internode_length_segments>3</internode_length_segments>')
                lines.append('\t\t\t\t\t<curvature_perturbations>0;0</curvature_perturbations>')
                lines.append('\t\t\t\t\t<yaw_perturbations>0;0</yaw_perturbations>')

                # Petioles & Leaves
                pets = phytomer_parts[inode_idx]["petioles"]
                if not pets:
                    # Default dummy petiole for empty phytomer
                    lines.append('\t\t\t\t\t<petiole>')
                    lines.append('\t\t\t\t\t\t<petiole_length>0.05</petiole_length>')
                    lines.append('\t\t\t\t\t\t<petiole_radius>0.001</petiole_radius>')
                    lines.append('\t\t\t\t\t\t<petiole_pitch>45</petiole_pitch>')
                    lines.append('\t\t\t\t\t\t<petiole_curvature>0</petiole_curvature>')
                    lines.append('\t\t\t\t\t\t<current_leaf_scale_factor>1</current_leaf_scale_factor>')
                    lines.append('\t\t\t\t\t\t<petiole_taper>0</petiole_taper>')
                    lines.append('\t\t\t\t\t\t<petiole_length_segments>3</petiole_length_segments>')
                    lines.append('\t\t\t\t\t\t<petiole_radial_subdivisions>6</petiole_radial_subdivisions>')
                    lines.append('\t\t\t\t\t\t<leaflet_scale>1</leaflet_scale>')
                    lines.append('\t\t\t\t\t\t<leaflet_offset>0</leaflet_offset>')
                    lines.append('\t\t\t\t\t</petiole>')
                else:
                    for pet_idx in pets:
                        p_info = part_info[pet_idx]
                        pet_leaves = petiole_leaves.get(pet_idx, [])
                        
                        # Relative petiole pitch
                        cos_p = np.clip(np.dot(info["dir"], p_info["dir"]), -1.0, 1.0)
                        pet_pitch_deg = math.degrees(math.acos(cos_p))

                        lines.append('\t\t\t\t\t<petiole>')
                        lines.append(f'\t\t\t\t\t\t<petiole_length>{_fmt(p_info["sz"])}</petiole_length>')
                        lines.append(f'\t\t\t\t\t\t<petiole_radius>{_fmt(p_info["sx"])}</petiole_radius>')
                        lines.append(f'\t\t\t\t\t\t<petiole_pitch>{_fmt(pet_pitch_deg)}</petiole_pitch>')
                        lines.append(f'\t\t\t\t\t\t<petiole_curvature>{_fmt(p_info["curvature"])}</petiole_curvature>')
                        lines.append('\t\t\t\t\t\t<current_leaf_scale_factor>1</current_leaf_scale_factor>')
                        lines.append('\t\t\t\t\t\t<petiole_taper>0</petiole_taper>')
                        lines.append('\t\t\t\t\t\t<petiole_length_segments>3</petiole_length_segments>')
                        lines.append('\t\t\t\t\t\t<petiole_radial_subdivisions>6</petiole_radial_subdivisions>')
                        lines.append('\t\t\t\t\t\t<leaflet_scale>1</leaflet_scale>')
                        lines.append('\t\t\t\t\t\t<leaflet_offset>0.08</leaflet_offset>')

                        for lf_idx in pet_leaves:
                            lf_info = part_info[lf_idx]
                            # Decompose relative rotation R_rel = R_pet.T @ R_leaf
                            R_rel = p_info["R"].T @ lf_info["R"]
                            r_rad, p_rad, az_rad = _matrix_to_euler_xyz(R_rel)

                            lines.append('\t\t\t\t\t\t<leaf>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_scale>{_fmt(lf_info["sx"])}</leaf_scale>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_pitch>{_fmt(math.degrees(p_rad))}</leaf_pitch>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_yaw>{_fmt(math.degrees(az_rad))}</leaf_yaw>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_roll>{_fmt(math.degrees(r_rad))}</leaf_roll>')
                            lines.append('\t\t\t\t\t\t</leaf>')

                        # Buds, Peduncles, and Inflorescences
                        peds = phytomer_parts[inode_idx]["peduncles"]
                        if peds:
                            for pd_idx in peds:
                                pd_info = part_info[pd_idx]
                                infls = peduncle_infls.get(pd_idx, [])
                                is_fruit = any(part_info[fl]["ot"] in (ORGAN_FRUIT, 8) for fl in infls)
                                stored_state = pd_info["bud_state"]
                                if stored_state in (2, 3, 4):
                                    bud_state = stored_state
                                else:
                                    bud_state = 4 if is_fruit else (3 if infls else 1)

                                lines.append('\t\t\t\t\t\t<floral_bud>')
                                lines.append(f'\t\t\t\t\t\t\t<bud_state>{bud_state}</bud_state>')
                                lines.append('\t\t\t\t\t\t\t<parent_index>0</parent_index>')
                                lines.append('\t\t\t\t\t\t\t<bud_index>0</bud_index>')
                                lines.append('\t\t\t\t\t\t\t<is_terminal>0</is_terminal>')
                                lines.append('\t\t\t\t\t\t\t<current_fruit_scale_factor>1</current_fruit_scale_factor>')
                                lines.append('\t\t\t\t\t\t\t<peduncle>')
                                lines.append(f'\t\t\t\t\t\t\t\t<length>{_fmt(pd_info["sz"])}</length>')
                                lines.append(f'\t\t\t\t\t\t\t\t<radius>{_fmt(pd_info["sx"])}</radius>')
                                lines.append('\t\t\t\t\t\t\t\t<pitch>15</pitch>')
                                lines.append(f'\t\t\t\t\t\t\t\t<curvature>{_fmt(pd_info["curvature"])}</curvature>')
                                lines.append('\t\t\t\t\t\t\t\t<roll>0</roll>')
                                lines.append('\t\t\t\t\t\t\t</peduncle>')

                                if infls:
                                    lines.append('\t\t\t\t\t\t\t<inflorescence>')
                                    lines.append('\t\t\t\t\t\t\t\t<flower_offset>0.05</flower_offset>')
                                    for fl_idx in infls:
                                        fl_info = part_info[fl_idx]
                                        R_rel = pd_info["R"].T @ fl_info["R"]
                                        r_rad, p_rad, az_rad = _matrix_to_euler_xyz(R_rel)

                                        lines.append('\t\t\t\t\t\t\t\t<flower>')
                                        lines.append(f'\t\t\t\t\t\t\t\t\t<flower_pitch>{_fmt(math.degrees(p_rad))}</flower_pitch>')
                                        lines.append(f'\t\t\t\t\t\t\t\t\t<flower_yaw>{_fmt(math.degrees(az_rad))}</flower_yaw>')
                                        lines.append(f'\t\t\t\t\t\t\t\t\t<flower_roll>{_fmt(math.degrees(r_rad))}</flower_roll>')
                                        lines.append('\t\t\t\t\t\t\t\t\t<flower_azimuth>0</flower_azimuth>')
                                        lines.append(f'\t\t\t\t\t\t\t\t\t<flower_base_scale>{_fmt(fl_info["sx"])}</flower_base_scale>')
                                        lines.append('\t\t\t\t\t\t\t\t</flower>')
                                    lines.append('\t\t\t\t\t\t\t</inflorescence>')
                                lines.append('\t\t\t\t\t\t</floral_bud>')
                        else:
                            # Dormant floral bud
                            lines.append('\t\t\t\t\t\t<floral_bud>')
                            lines.append('\t\t\t\t\t\t\t<bud_state>0</bud_state>')
                            lines.append('\t\t\t\t\t\t\t<parent_index>0</parent_index>')
                            lines.append('\t\t\t\t\t\t\t<bud_index>0</bud_index>')
                            lines.append('\t\t\t\t\t\t\t<is_terminal>0</is_terminal>')
                            lines.append('\t\t\t\t\t\t\t<current_fruit_scale_factor>1</current_fruit_scale_factor>')
                            lines.append('\t\t\t\t\t\t</floral_bud>')

                        lines.append('\t\t\t\t\t</petiole>')

                lines.append('\t\t\t\t</internode>')
                lines.append('\t\t\t</phytomer>')

            lines.append('\t\t</shoot>')

        lines.append('\t</plant_instance>')
        lines.append('</helios>\n')
        return "\n".join(lines)
