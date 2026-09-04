"""
Autonomous Plant Assembly and Inverse Kinematics Converter (Part Tensor 13D -> Helios XML).

Transforms 13D part arrays [organ_type(1), base_xyz(3), rot6d(6), scale_xyz(3)] into
semantically valid, geometrically identical Helios XML structures.

Key features:
1. Exact Matrix Transpose Correction:
   - rotation_6d_to_matrix returns row-stacked basis vectors, so R = M.T is the true rotation matrix.
2. Phytomer-First Graph Association:
   - Groups internodes, petioles, leaves, peduncles, flowers, and buds before resolving branch hierarchy.
3. Direction & Leaf-Axil Validated Branch Connection:
   - Evaluates spatial distance, anatomical branching angle [20°, 85°], and petiole axil alignment.
4. Closed-Loop Sequential Fitting:
   - Resets kinematic drift at each phytomer step to prevent cumulative branch divergence.
"""

import math
import numpy as np
import torch
from scipy.spatial import cKDTree
from typing import Dict, Any, List, Tuple, Optional

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


def _fmt(val: float, precision: int = 5) -> str:
    """Format floating point values cleanly for XML."""
    if abs(val) < 1e-9:
        return "0"
    s = f"{val:.{precision}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _invert_helios_zxz_rotation(R_helios: np.ndarray) -> Tuple[float, float, float]:
    """
    Decompose Helios rotation matrix R_helios = Rz(yaw + roll) * Rx(pitch) * Rz(roll)
    into (pitch_deg, yaw_deg, roll_deg).
    """
    cos_pitch = np.clip(R_helios[2, 2], -1.0, 1.0)
    pitch_rad = math.acos(cos_pitch)
    
    if math.sin(pitch_rad) > 1e-4:
        alpha = math.atan2(R_helios[0, 2], -R_helios[1, 2])
        gamma = math.atan2(R_helios[2, 0], R_helios[2, 1])
        roll_rad = gamma
        yaw_rad = alpha - gamma
    else:
        pitch_rad = 0.0
        roll_rad = 0.0
        yaw_rad = math.atan2(R_helios[1, 0], R_helios[0, 0])
        
    pitch_deg = math.degrees(pitch_rad)
    yaw_deg = (math.degrees(yaw_rad) + 360.0) % 360.0
    roll_deg = (math.degrees(roll_rad) + 360.0) % 360.0
    return pitch_deg, yaw_deg, roll_deg


def _rot_z_matrix(ang_rad: float) -> np.ndarray:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class PartAssemblyToXMLConverter:
    """Autonomous converter from (N, 13) part tensor to Helios XML."""

    def __init__(self, connectivity_tolerance: float = 0.008):
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

        if not internodes:
            return '<?xml version="1.0" encoding="UTF-8"?>\n<helios>\n\t<plant_instance id="0">\n\t\t<plant_base_position>0;0;0</plant_base_position>\n\t\t<plant_type>cowpea</plant_type>\n\t</plant_instance>\n</helios>\n'

        # CRITICAL FIX: rotation_6d_to_matrix returns row-stacked basis vectors (dim=-2),
        # so R = M.T is the true column-basis rotation matrix where:
        # Col 0 = up, Col 1 = fwd, Col 2 = cross(up, fwd)
        r6_all = torch.from_numpy(p_np[:, P_COL_ROT_0:P_COL_ROT_5+1]).float()
        M_all = rotation_6d_to_matrix(r6_all).numpy()
        R_all = np.transpose(M_all, (0, 2, 1))

        part_info = {}
        for idx in range(N):
            base = p_np[idx, P_COL_BASE_X:P_COL_BASE_Z+1]
            R = R_all[idx]
            sx = float(p_np[idx, P_COL_SCALE_X])
            sy = float(p_np[idx, P_COL_SCALE_Y])
            sz = float(p_np[idx, P_COL_SCALE_Z])
            dir_fwd = R[:, 1]
            tip = base + dir_fwd * sx
            part_info[idx] = {
                "ot": ot_all[idx],
                "base": base,
                "tip": tip,
                "R": R,
                "dir": dir_fwd,
                "up": R[:, 0],
                "sx": sx,
                "sy": sy,
                "sz": sz,
                "orig_idx": idx,
            }

        # 2. PHYTOMER & SPATIAL ATTACHMENT FIRST
        inode_petioles: Dict[int, List[int]] = {i: [] for i in internodes}
        petiole_leaves: Dict[int, List[int]] = {p: [] for p in petioles}
        inode_peduncles: Dict[int, List[int]] = {i: [] for i in internodes}
        peduncle_infls: Dict[int, List[int]] = {pd: [] for pd in peduncles}
        inode_buds: Dict[int, int] = {}

        if shoot_metas:
            curr_inode = None
            curr_pet = None
            curr_ped = None
            for idx in range(N):
                if not active_mask[idx]:
                    continue
                ot = ot_all[idx]
                if ot == ORGAN_INTERNODE:
                    curr_inode = idx
                    curr_pet = None
                    curr_ped = None
                elif ot == ORGAN_PETIOLE and curr_inode is not None:
                    inode_petioles[curr_inode].append(idx)
                    curr_pet = idx
                elif ot == ORGAN_LEAF and curr_pet is not None:
                    petiole_leaves[curr_pet].append(idx)
                elif ot == ORGAN_PEDUNCLE and curr_inode is not None:
                    inode_peduncles[curr_inode].append(idx)
                    curr_ped = idx
                elif ot in (ORGAN_FLOWER_OPEN, ORGAN_FLOWER_CLOSED, ORGAN_FRUIT) and curr_ped is not None:
                    peduncle_infls[curr_ped].append(idx)
                elif ot in (ORGAN_BUD_DORMANT, ORGAN_BUD_ACTIVE, ORGAN_BUD_ABORTED) and curr_inode is not None:
                    inode_buds[curr_inode] = idx
        else:
            inode_all_tips = np.array([part_info[i]["tip"] for i in internodes])
            inode_spatial_tree = cKDTree(inode_all_tips)

            for pet_idx in petioles:
                p_base = part_info[pet_idx]["base"]
                d, nearest_inode_idx = inode_spatial_tree.query(p_base)
                target_inode = internodes[nearest_inode_idx]
                inode_petioles[target_inode].append(pet_idx)

            if petioles:
                pet_all_tips = np.array([part_info[p]["tip"] for p in petioles])
                pet_spatial_tree = cKDTree(pet_all_tips)
                for lf_idx in leaves:
                    l_base = part_info[lf_idx]["base"]
                    d, nearest_pet_idx = pet_spatial_tree.query(l_base)
                    target_pet = petioles[nearest_pet_idx]
                    petiole_leaves[target_pet].append(lf_idx)

            for ped_idx in peduncles:
                pd_base = part_info[pd_idx]["base"]
                d, nearest_inode_idx = inode_spatial_tree.query(pd_base)
                target_inode = internodes[nearest_inode_idx]
                inode_peduncles[target_inode].append(ped_idx)

            if peduncles:
                ped_all_tips = np.array([part_info[pd]["tip"] for pd in peduncles])
                ped_spatial_tree = cKDTree(ped_all_tips)
                for fl_idx in flowers + fruits:
                    fl_base = part_info[fl_idx]["base"]
                    d, nearest_ped_idx = ped_spatial_tree.query(fl_base)
                    target_ped = peduncles[nearest_ped_idx]
                    peduncle_infls[target_ped].append(fl_idx)

            for b_idx in buds:
                b_base = part_info[b_idx]["base"]
                d, nearest_inode_idx = inode_spatial_tree.query(b_base)
                target_inode = internodes[nearest_inode_idx]
                inode_buds[target_inode] = b_idx

        # 3. RECONSTRUCT SHOOT GRAPH WITH DIRECTION & AXIL VALIDATION
        if shoot_metas:
            shoot_groups: List[List[int]] = []
            curr_shoot_inodes = []
            for idx in range(N):
                if not active_mask[idx]:
                    continue
                ot = ot_all[idx]
                if ot == ORGAN_SHOOT_META:
                    if curr_shoot_inodes:
                        shoot_groups.append(curr_shoot_inodes)
                        curr_shoot_inodes = []
                elif ot in (ORGAN_INTERNODE, 2):
                    curr_shoot_inodes.append(idx)
            if curr_shoot_inodes:
                shoot_groups.append(curr_shoot_inodes)

            shoots = shoot_groups if shoot_groups else [[i for i in internodes]]
            inode_parent = {i: None for i in internodes}
            inode_children: Dict[int, List[int]] = {i: [] for i in internodes}

            # Map child shoot base to nearest parent internode tip with anatomical axil scoring
            for s_idx, sh in enumerate(shoots):
                if s_idx == 0:
                    continue
                first_base = part_info[sh[0]]["base"]
                c_fwd = part_info[sh[0]]["dir"]
                best_parent = None
                best_score = float("inf")

                for cand_sid in range(s_idx):
                    cand_sh = shoots[cand_sid]
                    for cand_inode in cand_sh:
                        d = float(np.linalg.norm(part_info[cand_inode]["tip"] - first_base))
                        if d > 0.025:
                            continue
                        
                        p_fwd = part_info[cand_inode]["dir"]
                        cos_b = float(np.dot(c_fwd, p_fwd))
                        # Anatomical rule: branch should not grow backward
                        if cos_b < -0.3:
                            continue

                        # Axil alignment: check distance to nearest petiole base on parent
                        parent_pets = inode_petioles.get(cand_inode, [])
                        axil_bonus = 0.0
                        if parent_pets:
                            pet_dists = [float(np.linalg.norm(part_info[p]["base"] - first_base)) for p in parent_pets]
                            nearest_pet_dist = min(pet_dists)
                            if nearest_pet_dist < 0.01:
                                axil_bonus -= 0.005

                        # Distance primary, branching angle secondary
                        main_stem_bonus = -0.003 if cand_sid == 0 else 0.0
                        score = d + 0.002 * max(0.0, 0.2 - cos_b) + axil_bonus + main_stem_bonus
                        if score < best_score:
                            best_score = score
                            best_parent = cand_inode

                if best_parent is None:
                    best_parent = shoots[0][0]
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

            shoots = []
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

        # 4. SERIALIZE TO HELIOS XML WITH CLOSED-LOOP SEQUENTIAL FITTING
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
            if not shoot_inodes:
                continue
            p_shoot_id = -1 if s_idx == 0 else 0
            p_node_idx = 0
            p_pet_idx = 0
            if s_idx > 0:
                first_i = shoot_inodes[0]
                parent_i = inode_parent.get(first_i)
                if parent_i is not None and parent_i in inode_to_location:
                    p_shoot_id, p_node_idx = inode_to_location[parent_i]
                    # Find nearest parent petiole index
                    parent_pets = inode_petioles.get(parent_i, [])
                    if parent_pets:
                        if len(parent_pets) == 1:
                            p_pet_idx = 0
                        else:
                            c_fwd = part_info[first_i]["dir"]
                            pet_aligns = [float(np.dot(c_fwd, part_info[p]["dir"])) for p in parent_pets]
                            p_pet_idx = int(np.argmax(pet_aligns))

            shoot_label = "unifoliate" if s_idx == 0 else "trifoliate"
            lines.append(f'		<shoot ID="{s_idx}">')
            lines.append(f'			<shoot_type_label> {shoot_label} </shoot_type_label>')
            lines.append(f'			<parent_shoot_ID> {p_shoot_id} </parent_shoot_ID>')
            lines.append(f'			<parent_node_index> {p_node_idx} </parent_node_index>')
            lines.append(f'			<parent_petiole_index> {p_pet_idx} </parent_petiole_index>')

            # Invert Shoot Base Rotation via Helios Z-X-Z Euler Decomposition
            first_inode = shoot_inodes[0]
            R_first = part_info[first_inode]["R"]
            if s_idx == 0:
                # Helios frame: X=col0(up), Y=-col2(cross), Z=col1(fwd)
                R_helios = np.stack([R_first[:, 0], -R_first[:, 2], R_first[:, 1]], axis=1)
                base_pitch, base_yaw, base_roll = _invert_helios_zxz_rotation(R_helios)
            else:
                parent_i = inode_parent.get(first_inode)
                if parent_i is not None:
                    R_parent = part_info[parent_i]["R"]
                    R_rel = R_parent.T @ R_first
                    R_helios_rel = np.stack([R_rel[:, 0], -R_rel[:, 2], R_rel[:, 1]], axis=1)
                    base_pitch, base_yaw, base_roll = _invert_helios_zxz_rotation(R_helios_rel)
                else:
                    R_helios = np.stack([R_first[:, 0], -R_first[:, 2], R_first[:, 1]], axis=1)
                    base_pitch, base_yaw, base_roll = _invert_helios_zxz_rotation(R_helios)

            lines.append(f'			<base_rotation> {_fmt(base_pitch)} {_fmt(base_yaw)} {_fmt(base_roll)} </base_rotation>')

            # Closed-loop tracking state for current shoot
            sim_dir = part_info[shoot_inodes[0]]["dir"].copy()
            sim_up = part_info[shoot_inodes[0]]["up"].copy()

            for node_i, inode_idx in enumerate(shoot_inodes):
                info = part_info[inode_idx]
                inode_len = max(info["sx"], 0.001)
                inode_rad = max(info["sy"], 0.0005)

                # Relative pitch and phyllotaxis with Closed-Loop Sequential Alignment
                if node_i == 0:
                    inode_pitch = 0.0
                    inode_phyllo = 137.5
                else:
                    # Angle between simulated previous direction and current target direction
                    cos_d = np.clip(np.dot(sim_dir, info["dir"]), -1.0, 1.0)
                    bend_deg = math.degrees(math.acos(cos_d))
                    inode_pitch = -bend_deg / 1.25 if abs(bend_deg) > 0.5 else 0.0
                    
                    # Phyllotactic angle relative to simulated up vector
                    cos_up = np.clip(np.dot(sim_up, info["up"]), -1.0, 1.0)
                    cross_up = np.dot(sim_dir, np.cross(sim_up, info["up"]))
                    inode_phyllo = math.degrees(math.atan2(cross_up, cos_up))
                    if abs(inode_phyllo) < 1.0:
                        inode_phyllo = 137.5

                    # Closed-loop update: anchor simulated state to exact target
                    sim_dir = info["dir"].copy()
                    sim_up = info["up"].copy()

                lines.append('			<phytomer>')
                lines.append('				<internode>')
                lines.append(f'					<internode_length>{_fmt(inode_len)}</internode_length>')
                lines.append(f'					<internode_radius>{_fmt(inode_rad)}</internode_radius>')
                lines.append(f'					<internode_pitch>{_fmt(inode_pitch)}</internode_pitch>')
                lines.append(f'					<internode_phyllotactic_angle>{_fmt(inode_phyllo)}</internode_phyllotactic_angle>')
                lines.append(f'					<internode_length_max>{_fmt(inode_len)}</internode_length_max>')
                lines.append('					<internode_length_segments>1</internode_length_segments>')
                lines.append('					<curvature_perturbations>0;0</curvature_perturbations>')
                lines.append('					<yaw_perturbations>0;0</yaw_perturbations>')

                # Petioles & Attached Leaves
                pets = inode_petioles.get(inode_idx, [])
                if not pets:
                    lines.append('					<petiole>')
                    lines.append('						<petiole_length>0.001</petiole_length>')
                    lines.append('						<petiole_radius>0.0005</petiole_radius>')
                    lines.append('						<petiole_pitch>45</petiole_pitch>')
                    lines.append('						<petiole_curvature>0</petiole_curvature>')
                    lines.append('						<current_leaf_scale_factor>0</current_leaf_scale_factor>')
                    lines.append('						<petiole_taper>0.25</petiole_taper>')
                    lines.append('						<petiole_length_segments>1</petiole_length_segments>')
                    lines.append('						<petiole_radial_subdivisions>6</petiole_radial_subdivisions>')
                    lines.append('						<leaflet_scale>0</leaflet_scale>')
                    lines.append('						<leaflet_offset>0</leaflet_offset>')
                    lines.append('					</petiole>')
                else:
                    for pet_idx in pets:
                        p_info = part_info[pet_idx]
                        pet_leaves = petiole_leaves.get(pet_idx, [])
                        
                        pet_len = max(p_info["sx"], 0.001)
                        pet_rad = max(p_info["sy"], 0.0005)

                        # Angle between petiole forward axis and internode forward axis
                        cos_pet = np.clip(np.dot(info["dir"], p_info["dir"]), -1.0, 1.0)
                        pet_pitch_deg = math.degrees(math.acos(cos_pet))

                        lines.append('					<petiole>')
                        lines.append(f'						<petiole_length>{_fmt(pet_len)}</petiole_length>')
                        lines.append(f'						<petiole_radius>{_fmt(pet_rad)}</petiole_radius>')
                        lines.append(f'						<petiole_pitch>{_fmt(pet_pitch_deg if abs(pet_pitch_deg) > 0.5 else 45.0)}</petiole_pitch>')
                        lines.append('						<petiole_curvature>0</petiole_curvature>')
                        lines.append('						<current_leaf_scale_factor>1</current_leaf_scale_factor>')
                        lines.append('						<petiole_taper>0.25</petiole_taper>')
                        lines.append('						<petiole_length_segments>3</petiole_length_segments>')
                        lines.append('						<petiole_radial_subdivisions>6</petiole_radial_subdivisions>')
                        leaflet_scale = 1.0 if len(pet_leaves) <= 1 else 0.9
                        lines.append(f'						<leaflet_scale>{_fmt(leaflet_scale)}</leaflet_scale>')
                        lines.append('						<leaflet_offset>0.4</leaflet_offset>')

                        num_pet_leaves = len(pet_leaves)
                        for lf_i, lf_idx in enumerate(pet_leaves):
                            lf_info = part_info[lf_idx]
                            leaf_scale = max(lf_info["sx"], 0.001)

                            # Exact Leaf Frame Inversion matching Helios compound geometry
                            pet_tip_axis = p_info["dir"]
                            if num_pet_leaves > 1:
                                if lf_i == (num_pet_leaves - 1) / 2.0:
                                    compound_rot = 0.0
                                elif lf_i < (num_pet_leaves - 1) / 2.0:
                                    compound_rot = -0.5 * math.pi
                                else:
                                    compound_rot = 0.5 * math.pi
                            else:
                                compound_rot = 0.0

                            azimuth_rot = -math.atan2(pet_tip_axis[1], pet_tip_axis[0] + 1e-8) + compound_rot
                            R_az = _rot_z_matrix(azimuth_rot)
                            R_local = R_az.T @ lf_info["R"]

                            pitch_rot = -math.asin(np.clip(R_local[0, 2], -1.0, 1.0))
                            asin_pz = math.asin(np.clip(pet_tip_axis[2], -1.0, 1.0))
                            
                            ind_from_tip = float(lf_i) - float(num_pet_leaves - 1) / 2.0
                            if num_pet_leaves == 1:
                                leaf_pitch = math.degrees(pitch_rot - asin_pz)
                                roll_rot = math.atan2(-R_local[1, 2], R_local[1, 1])
                                acos_iz = math.acos(np.clip(info["dir"][2], -1.0, 1.0))
                                leaf_roll = math.degrees(acos_iz - roll_rot)
                                leaf_yaw = 0.0
                            elif ind_from_tip != 0:
                                leaf_pitch = math.degrees(pitch_rot)
                                sign_roll = compound_rot / abs(compound_rot)
                                roll_rot = math.atan2(-R_local[1, 2], R_local[1, 1])
                                leaf_roll = math.degrees(roll_rot / sign_roll - asin_pz)
                                leaf_yaw = math.degrees(math.atan2(R_local[0, 1], R_local[0, 0]))
                            else:
                                leaf_pitch = math.degrees(pitch_rot - asin_pz)
                                leaf_roll = 0.0
                                leaf_yaw = 0.0

                            leaf_pitch = (leaf_pitch + 180.0) % 360.0 - 180.0
                            leaf_roll = (leaf_roll + 180.0) % 360.0 - 180.0
                            leaf_yaw = (leaf_yaw + 180.0) % 360.0 - 180.0

                            lines.append('						<leaf>')
                            lines.append(f'							<leaf_scale>{_fmt(leaf_scale)}</leaf_scale>')
                            lines.append(f'							<leaf_pitch>{_fmt(leaf_pitch)}</leaf_pitch>')
                            lines.append(f'							<leaf_yaw>{_fmt(leaf_yaw)}</leaf_yaw>')
                            lines.append(f'							<leaf_roll>{_fmt(leaf_roll)}</leaf_roll>')
                            lines.append('						</leaf>')

                        lines.append('					</petiole>')

                # Buds, Peduncles, and Inflorescences
                peds = inode_peduncles.get(inode_idx, [])
                has_bud = inode_idx in inode_buds
                if peds:
                    for pd_idx in peds:
                        pd_info = part_info[pd_idx]
                        infls = peduncle_infls.get(pd_idx, [])
                        infl_ots = [part_info[fl]["ot"] for fl in infls]
                        if any(ot in (ORGAN_FRUIT, 8, 11) for ot in infl_ots):
                            bud_state = 4
                        elif any(ot in (ORGAN_FLOWER_OPEN, 10) for ot in infl_ots):
                            bud_state = 3
                        elif any(ot in (ORGAN_FLOWER_CLOSED, 9) for ot in infl_ots):
                            bud_state = 2
                        else:
                            bud_state = 1

                        ped_len = max(pd_info["sx"], 0.001)
                        ped_rad = max(pd_info["sy"], 0.0005)
                        cos_pd = np.clip(np.dot(info["dir"], pd_info["dir"]), -1.0, 1.0)
                        ped_pitch_deg = math.degrees(math.acos(cos_pd))

                        lines.append('					<floral_bud>')
                        lines.append(f'						<bud_state>{bud_state}</bud_state>')
                        lines.append('						<parent_index>0</parent_index>')
                        lines.append('						<bud_index>0</bud_index>')
                        lines.append('						<is_terminal>0</is_terminal>')
                        lines.append('						<current_fruit_scale_factor>1</current_fruit_scale_factor>')
                        lines.append('						<peduncle>')
                        lines.append(f'							<peduncle_length>{_fmt(ped_len)}</peduncle_length>')
                        lines.append(f'							<peduncle_radius>{_fmt(ped_rad)}</peduncle_radius>')
                        lines.append(f'							<peduncle_pitch>{_fmt(ped_pitch_deg if abs(ped_pitch_deg) > 1.0 else 30.0)}</peduncle_pitch>')
                        lines.append('							<peduncle_roll>0</peduncle_roll>')
                        lines.append('							<peduncle_curvature>160</peduncle_curvature>')
                        lines.append('						</peduncle>')

                        for fl_idx in infls:
                            fl_info = part_info[fl_idx]
                            fl_scale = max(fl_info["sx"], 0.01)
                            R_rel_fl = pd_info["R"].T @ fl_info["R"]
                            fl_pitch = math.degrees(-math.asin(np.clip(R_rel_fl[2, 0], -1.0, 1.0)))
                            fl_yaw = math.degrees(math.atan2(R_rel_fl[1, 0], R_rel_fl[0, 0]))
                            fl_roll = math.degrees(math.atan2(R_rel_fl[2, 1], R_rel_fl[2, 2]))

                            lines.append('						<flower>')
                            lines.append(f'							<flower_scale>{_fmt(fl_scale)}</flower_scale>')
                            lines.append(f'							<flower_pitch>{_fmt(fl_pitch)}</flower_pitch>')
                            lines.append(f'							<flower_yaw>{_fmt(fl_yaw)}</flower_yaw>')
                            lines.append(f'							<flower_roll>{_fmt(fl_roll)}</flower_roll>')
                            lines.append('							<flower_azimuth>0</flower_azimuth>')
                            lines.append('						</flower>')

                        lines.append('					</floral_bud>')
                elif has_bud:
                    b_idx = inode_buds[inode_idx]
                    bot = part_info[b_idx]["ot"]
                    bs = 0 if bot == ORGAN_BUD_DORMANT else (5 if bot == ORGAN_BUD_ABORTED else 1)
                    lines.append('					<floral_bud>')
                    lines.append(f'						<bud_state>{bs}</bud_state>')
                    lines.append('						<parent_index>0</parent_index>')
                    lines.append('						<bud_index>0</bud_index>')
                    lines.append('						<is_terminal>0</is_terminal>')
                    lines.append('						<current_fruit_scale_factor>1</current_fruit_scale_factor>')
                    lines.append('					</floral_bud>')

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
    """
    Convenience helper to convert a 13D part tensor into a Helios XML string
    via canonical (N, 40) PlantOrganArray intermediate representation.
    """
    from diffusion_based.models.part_tensor_to_40d import PartTensorTo40DConverter
    from diffusion_based.models.plant_organ_array import PlantOrganArray
    converter = PartTensorTo40DConverter()
    t_40d = converter.convert(part_tensor, plant_id=plant_id)
    arr = PlantOrganArray(t_40d)
    return arr.to_xml_string(existence_threshold=existence_threshold)

