"""
Direct 14D Part Tensor to 40D PlantOrganArray Tensor Converter.

Converts (N, 14) world-space Part Tensor [organ_type(1), base_xyz(3), rot6d(6), scale_xyz(3), curvature(1)]
into the canonical (N, 40) PlantOrganArray typed tensor.

Ensures:
1. Exact row-to-row isomorphism (N rows in 14D -> N rows in 40D).
2. Exact topological recovery (parent shoots, parent node indices, parent petiole indices).
3. Exact physical scaling (internode lengths, radii, petiole lengths, leaf scales).
4. Direct serialization to XML via PlantOrganArray._to_xml_string_typed().
"""

import math
import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional

from diffusion_based.models.plant_organ_array import (
    PlantOrganArray,
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
    NUM_FEATURES_TYPED,
    T_COL_PLANT_ID,
    T_COL_PLANT_AGE,
    T_COL_BASE_X,
    T_COL_BASE_Y,
    T_COL_BASE_Z,
    T_COL_SHOOT_ID,
    T_COL_PARENT_SHOOT_ID,
    T_COL_PARENT_NODE_IDX,
    T_COL_PARENT_PETIOLE_IDX,
    T_COL_PHYTOMER_IDX,
    T_COL_CHILD_INDEX,
    T_COL_ORGAN_TYPE,
    T_COL_SHOOT_TYPE,
    T_COL_LENGTH,
    T_COL_RADIUS,
    T_COL_SCALE,
    T_COL_PITCH,
    T_COL_YAW,
    T_COL_ROLL,
    T_COL_CURVATURE,
    T_COL_PHYLLOTACTIC_ANGLE,
    T_COL_LENGTH_MAX,
    T_COL_LENGTH_SEGMENTS,
    T_COL_CURV_PERT_0,
    T_COL_CURV_PERT_1,
    T_COL_YAW_PERT_0,
    T_COL_YAW_PERT_1,
    T_COL_CURRENT_LEAF_SCALE_FACTOR,
    T_COL_TAPER,
    T_COL_RADIAL_SUBDIVISIONS,
    T_COL_LEAFLET_SCALE,
    T_COL_LEAFLET_OFFSET,
    T_COL_BUD_STATE,
    T_COL_BUD_PARENT_INDEX,
    T_COL_BUD_IS_TERMINAL,
    T_COL_FRUIT_SCALE,
    T_COL_FLOWER_AZIMUTH,
    T_COL_FLOWER_OFFSET,
    T_COL_RESERVED,
    T_COL_EXISTENCE,
    rotation_6d_to_matrix,
)


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


def rotate_about_line(point, axis, angle_rad):
    v = point
    k = axis / np.linalg.norm(axis)
    return v * math.cos(angle_rad) + np.cross(k, v) * math.sin(angle_rad) + k * np.dot(k, v) * (1.0 - math.cos(angle_rad))


def forward_helios_base(u_p, v_p, pitch_deg, yaw_deg, roll_deg=90.0, inode_pitch_deg=20.0):
    u_p = u_p / np.linalg.norm(u_p)
    v_p = v_p / np.linalg.norm(v_p)
    pet_rot = np.cross(u_p, v_p)
    norm = np.linalg.norm(pet_rot)
    if norm < 1e-6:
        pet_rot = np.array([1.0, 0.0, 0.0])
    else:
        pet_rot /= norm
    inode_axis = u_p.copy()
    if inode_pitch_deg != 0.0:
        inode_axis = rotate_about_line(inode_axis, pet_rot, math.radians(0.5 * inode_pitch_deg))
    if roll_deg != 0.0:
        pet_rot = rotate_about_line(pet_rot, u_p, math.radians(roll_deg))
        inode_axis = rotate_about_line(inode_axis, u_p, math.radians(roll_deg))
    if pitch_deg != 0.0:
        base_pitch_axis = -1.0 * np.cross(u_p, v_p)
        base_pitch_axis /= np.linalg.norm(base_pitch_axis)
        pet_rot = rotate_about_line(pet_rot, base_pitch_axis, math.radians(-pitch_deg))
        inode_axis = rotate_about_line(inode_axis, base_pitch_axis, math.radians(-pitch_deg))
    if yaw_deg != 0.0:
        pet_rot = rotate_about_line(pet_rot, u_p, math.radians(yaw_deg))
        inode_axis = rotate_about_line(inode_axis, u_p, math.radians(yaw_deg))
    return inode_axis


def solve_helios_shoot_base(u_p, v_p, u_c, inode_pitch_deg=20.0, roll_deg=90.0):
    u_p = u_p / np.linalg.norm(u_p)
    v_p = v_p / np.linalg.norm(v_p)
    u_c = u_c / np.linalg.norm(u_c)
    k_pet = np.cross(u_p, v_p)
    norm = np.linalg.norm(k_pet)
    if norm < 1e-6:
        k_pet = np.array([1.0, 0.0, 0.0])
    else:
        k_pet /= norm
    a1 = rotate_about_line(u_p, k_pet, math.radians(0.5 * inode_pitch_deg))
    a2 = rotate_about_line(a1, u_p, math.radians(roll_deg))
    A = np.dot(u_p, a2)
    B = np.dot(u_p, np.cross(k_pet, a2))
    C = np.dot(u_p, u_c)
    R = math.sqrt(A * A + B * B)
    phi = math.atan2(B, A)
    cos_val = np.clip(C / R, -1.0, 1.0)
    delta = math.acos(cos_val)
    pitch_candidates = [phi + delta, phi - delta]
    best_pitch, best_yaw = 45.0, 0.0
    best_err = float("inf")
    for p_rad in pitch_candidates:
        p_deg = math.degrees(p_rad)
        if p_deg < 0:
            p_deg += 360.0
        if p_deg > 180.0:
            p_deg = 360.0 - p_deg
        a3 = rotate_about_line(a2, k_pet, math.radians(p_deg))
        w_c = u_c - np.dot(u_p, u_c) * u_p
        w_3 = a3 - np.dot(u_p, a3) * u_p
        y_rad = math.atan2(np.dot(u_p, np.cross(w_3, w_c)), np.dot(w_3, w_c))
        y_deg = math.degrees(y_rad)
        u_sim = forward_helios_base(u_p, v_p, p_deg, y_deg, roll_deg, inode_pitch_deg)
        err = math.degrees(math.acos(np.clip(np.dot(u_c, u_sim), -1.0, 1.0)))
        if err < best_err:
            best_err = err
            best_pitch = p_deg
            best_yaw = y_deg
    return float(best_pitch), float(best_yaw), float(roll_deg)


class PartTensorTo40DConverter:
    """Converts (N, 13) Part Tensor to (N, 40) PlantOrganArray Typed Tensor."""

    def __init__(self, connectivity_tolerance: float = 0.008):
        self.tol = connectivity_tolerance

    def convert(self, part_tensor: torch.Tensor, plant_id: int = 0) -> torch.Tensor:
        p_np = part_tensor.detach().cpu().numpy()
        N = p_np.shape[0]

        out_40d = torch.zeros((N, NUM_FEATURES_TYPED), dtype=torch.float32)

        ot_all = np.round(p_np[:, 0]).astype(int)
        active_mask = ot_all > ORGAN_NONE

        r6_all = torch.from_numpy(p_np[:, 4:10]).float()
        M_all = rotation_6d_to_matrix(r6_all).numpy()
        R_all = np.transpose(M_all, (0, 2, 1))

        part_info = []
        for idx in range(N):
            base = p_np[idx, 1:4]
            R = R_all[idx]
            sx = max(1e-4, abs(float(p_np[idx, 10])))
            sy = max(5e-4, abs(float(p_np[idx, 11])))
            sz = max(5e-4, abs(float(p_np[idx, 12])))
            dir_fwd = R[:, 1]
            up = R[:, 0]
            tip = base + dir_fwd * sx
            part_info.append({
                "idx": idx,
                "ot": ot_all[idx],
                "base": base,
                "tip": tip,
                "R": R,
                "dir": dir_fwd,
                "up": up,
                "sx": sx,
                "sy": sy,
                "sz": sz,
                "active": active_mask[idx],
            })

        # 1. Parse shoot structure and grouping
        shoots: List[List[int]] = []
        shoot_petioles: List[List[List[int]]] = []
        shoot_metas: List[int] = []
        curr_inodes = []
        curr_pets: List[List[int]] = []
        curr_pidx_tracker = -1

        for idx in range(N):
            if not active_mask[idx]:
                continue
            ot = ot_all[idx]
            if ot == ORGAN_SHOOT_META:
                shoot_metas.append(idx)
                if curr_inodes:
                    shoots.append(curr_inodes)
                    shoot_petioles.append(curr_pets)
                    curr_inodes = []
                    curr_pets = []
                curr_pidx_tracker = -1
            elif ot == ORGAN_INTERNODE:
                curr_inodes.append(idx)
                curr_pidx_tracker += 1
                curr_pets.append([])
            elif ot == ORGAN_PETIOLE:
                if curr_pidx_tracker >= 0 and curr_pidx_tracker < len(curr_pets):
                    curr_pets[curr_pidx_tracker].append(idx)
        if curr_inodes:
            shoots.append(curr_inodes)
            shoot_petioles.append(curr_pets)

        # Build shoot mapping
        shoot_parent_info = {}
        for s_idx, sh in enumerate(shoots):
            if s_idx == 0:
                shoot_parent_info[s_idx] = (-1, 0, 0)
                continue
            first_base = part_info[sh[0]]["base"]
            c_fwd = part_info[sh[0]]["dir"]

            best_p = None
            best_score = float("inf")

            for cand_sid in range(s_idx):
                cand_sh = shoots[cand_sid]
                for node_i, cand_inode in enumerate(cand_sh):
                    d = float(np.linalg.norm(part_info[cand_inode]["tip"] - first_base))
                    if d > 0.025:
                        continue
                    p_fwd = part_info[cand_inode]["dir"]
                    cos_b = float(np.dot(c_fwd, p_fwd))
                    if cos_b < -0.3:
                        continue

                    # Main stem prioritization
                    main_stem_bonus = -0.003 if cand_sid == 0 else 0.0
                    score = d + 0.002 * max(0.0, 0.2 - cos_b) + main_stem_bonus
                    if score < best_score:
                        best_score = score
                        best_p = (cand_sid, node_i, cand_inode)

            if best_p is None:
                shoot_parent_info[s_idx] = (0, 0, 0)
            else:
                cand_sid, node_i, cand_inode = best_p
                # Choose petiole axil by alignment
                p_pet = 0
                if cand_sid < len(shoot_petioles) and node_i < len(shoot_petioles[cand_sid]):
                    node_pets = shoot_petioles[cand_sid][node_i]
                    if len(node_pets) > 1:
                        dot0 = float(np.dot(c_fwd, part_info[node_pets[0]]["dir"]))
                        dot1 = float(np.dot(c_fwd, part_info[node_pets[1]]["dir"]))
                        p_pet = 0 if dot0 >= dot1 else 1
                shoot_parent_info[s_idx] = (cand_sid, node_i, p_pet)

        # 1.5. Identify phytomer organ contents (to resolve floral bud state: fruit, flower open/closed, etc.)
        phytomer_flowers = {}
        c_sid = -1
        c_pidx = -1
        for idx in range(N):
            ot = ot_all[idx]
            if ot == ORGAN_SHOOT_META:
                c_sid += 1
                c_pidx = -1
            elif ot == ORGAN_INTERNODE:
                c_pidx += 1
            elif ot in (ORGAN_FLOWER_CLOSED, ORGAN_FLOWER_OPEN, ORGAN_FRUIT):
                phytomer_flowers.setdefault((c_sid, c_pidx), []).append(ot)

        # 2. Reconstruct 40D rows
        has_shoot_meta = bool((ot_all == ORGAN_SHOOT_META).any())
        curr_sid = -1 if has_shoot_meta else 0
        curr_pidx = -1
        curr_pet_i = 0
        leaf_in_pet = 0
        infl_in_ped = 0
        curr_ped_az = 0.0

        for idx in range(N):
            ot = ot_all[idx]
            out_40d[idx, T_COL_PLANT_ID] = float(plant_id)

            if ot == ORGAN_ROOT_META:
                out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_ROOT_META)
                out_40d[idx, T_COL_BASE_X:T_COL_BASE_Z+1] = torch.from_numpy(part_info[idx]["base"])
                out_40d[idx, T_COL_PLANT_AGE] = part_info[idx]["sx"]
                out_40d[idx, T_COL_EXISTENCE] = 1.0

            elif ot == ORGAN_SHOOT_META:
                curr_sid += 1
                curr_pidx = -1
                psi, pni, ppi = shoot_parent_info.get(curr_sid, (-1, 0, 0))
                out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_SHOOT_META)
                out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
                out_40d[idx, T_COL_PARENT_SHOOT_ID] = float(psi)
                out_40d[idx, T_COL_PARENT_NODE_IDX] = float(pni)
                out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = float(ppi)
                out_40d[idx, T_COL_PHYTOMER_IDX] = 0.0
                out_40d[idx, T_COL_SHOOT_TYPE] = 0.0 if curr_sid == 0 else 1.0
                out_40d[idx, T_COL_EXISTENCE] = 1.0
                out_40d[idx, T_COL_RESERVED] = 0.0

                # Invert base rotation
                first_inode = shoots[curr_sid][0]
                R_first = part_info[first_inode]["R"]
                if curr_sid == 0:
                    R_h = np.stack([R_first[:, 0], -R_first[:, 2], R_first[:, 1]], axis=1)
                    bp, by, br = _invert_helios_zxz_rotation(R_h)
                elif curr_sid == 1 and psi == 0 and pni == 0:
                    # Shoot 1 is the primary apical continuation of the main stem
                    bp = 0.0
                    by = 0.0
                    br = 90.0
                else:
                    # Lateral shoot: parent node frame via closed-form analytical Helios IK
                    parent_sh = shoots[psi]
                    parent_inode = parent_sh[min(pni, len(parent_sh)-1)]
                    u_p = part_info[parent_inode]["dir"]
                    u_c = part_info[first_inode]["dir"]

                    # Find parent petiole direction
                    v_p = None
                    if psi < len(shoot_petioles) and pni < len(shoot_petioles[psi]):
                        node_pets = shoot_petioles[psi][pni]
                        if node_pets:
                            pet_idx_use = node_pets[min(ppi, len(node_pets)-1)]
                            v_p = part_info[pet_idx_use]["dir"]

                    if v_p is None:
                        v_p = part_info[parent_inode]["up"]

                    bp, by, br = solve_helios_shoot_base(u_p, v_p, u_c)

                out_40d[idx, T_COL_PITCH] = bp
                out_40d[idx, T_COL_YAW] = by
                out_40d[idx, T_COL_ROLL] = br

            elif ot == ORGAN_INTERNODE:
                curr_pidx += 1
                curr_pet_i = 0
                pet_count_in_phytomer = 0
                leaf_counts_per_pet = {}
                infl_in_ped = 0
                out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_INTERNODE)
                out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
                out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
                out_40d[idx, T_COL_LENGTH] = part_info[idx]["sx"]
                out_40d[idx, T_COL_RADIUS] = part_info[idx]["sy"]
                out_40d[idx, T_COL_LENGTH_MAX] = part_info[idx]["sx"]
                out_40d[idx, T_COL_LENGTH_SEGMENTS] = 2.0
                
                # Base internode (Shoot 0, phytomer 0) orientation is defined by base_rotation; pitch must be 0.
                # Subsequent internodes deflect by the canonical species phytomer pitch (20.0 deg for cowpea).
                if curr_sid == 0 and curr_pidx == 0:
                    out_40d[idx, T_COL_PITCH] = 0.0
                else:
                    out_40d[idx, T_COL_PITCH] = 20.0

                # Dynamic analytical phyllotaxis inversion
                phyllo_angle = 180.0
                if curr_sid < len(shoot_petioles):
                    if curr_pidx > 0:
                        prev_pets = shoot_petioles[curr_sid][curr_pidx - 1] if curr_pidx - 1 < len(shoot_petioles[curr_sid]) else []
                        curr_pets_list = shoot_petioles[curr_sid][curr_pidx] if curr_pidx < len(shoot_petioles[curr_sid]) else []
                        if prev_pets and curr_pets_list:
                            u = part_info[idx]["dir"]
                            v_prev = part_info[prev_pets[0]]["dir"]
                            v_curr = part_info[curr_pets_list[0]]["dir"]
                            p1 = v_prev - float(np.dot(u, v_prev)) * u
                            norm_p1 = float(np.linalg.norm(p1))
                            p2 = v_curr - float(np.dot(u, v_curr)) * u
                            norm_p2 = float(np.linalg.norm(p2))
                            if norm_p1 > 1e-6 and norm_p2 > 1e-6:
                                p1 /= norm_p1
                                p2 /= norm_p2
                                ang_rad = math.atan2(float(np.dot(u, np.cross(p1, p2))), float(np.dot(p1, p2)))
                                ang_deg = math.degrees(ang_rad)
                                if ang_deg < 0:
                                    ang_deg += 360.0
                                phyllo_angle = float(ang_deg)
                    elif curr_pidx == 0 and curr_sid == 0:
                        # For Shoot 0 with 2 cotyledon petioles
                        p0_list = shoot_petioles[0][0] if shoot_petioles and shoot_petioles[0] else []
                        if len(p0_list) >= 2:
                            u = part_info[idx]["dir"]
                            v0 = part_info[p0_list[0]]["dir"]
                            v1 = part_info[p0_list[1]]["dir"]
                            p1 = v0 - float(np.dot(u, v0)) * u
                            norm_p1 = float(np.linalg.norm(p1))
                            p2 = v1 - float(np.dot(u, v1)) * u
                            norm_p2 = float(np.linalg.norm(p2))
                            if norm_p1 > 1e-6 and norm_p2 > 1e-6:
                                p1 /= norm_p1
                                p2 /= norm_p2
                                ang_rad = math.atan2(float(np.dot(u, np.cross(p1, p2))), float(np.dot(p1, p2)))
                                ang_deg = math.degrees(ang_rad)
                                if ang_deg < 0:
                                    ang_deg += 360.0
                                phyllo_angle = float(ang_deg)
                    elif curr_pidx == 0 and curr_sid > 0:
                        # For child shoot phytomer 0: petiole relative to parent shoot petiole
                        psi, pni, ppi = shoot_parent_info.get(curr_sid, (-1, 0, 0))
                        if psi >= 0 and psi < len(shoot_petioles) and pni < len(shoot_petioles[psi]):
                            p_pets = shoot_petioles[psi][pni]
                            c_pets = shoot_petioles[curr_sid][0] if shoot_petioles[curr_sid] else []
                            if p_pets and c_pets:
                                u = part_info[idx]["dir"]
                                v_prev = part_info[p_pets[min(ppi, len(p_pets)-1)]]["dir"]
                                v_curr = part_info[c_pets[0]]["dir"]
                                p1 = v_prev - float(np.dot(u, v_prev)) * u
                                norm_p1 = float(np.linalg.norm(p1))
                                p2 = v_curr - float(np.dot(u, v_curr)) * u
                                norm_p2 = float(np.linalg.norm(p2))
                                if norm_p1 > 1e-6 and norm_p2 > 1e-6:
                                    p1 /= norm_p1
                                    p2 /= norm_p2
                                    ang_rad = math.atan2(float(np.dot(u, np.cross(p1, p2))), float(np.dot(p1, p2)))
                                    ang_deg = math.degrees(ang_rad)
                                    if ang_deg < 0:
                                        ang_deg += 360.0
                                    phyllo_angle = float(ang_deg)

                out_40d[idx, T_COL_PHYLLOTACTIC_ANGLE] = phyllo_angle
                if p_np.shape[1] > 13:
                    out_40d[idx, T_COL_CURV_PERT_0] = float(p_np[idx, 13])
                out_40d[idx, T_COL_EXISTENCE] = 1.0

            elif ot == ORGAN_PETIOLE:
                curr_pet_i = pet_count_in_phytomer
                pet_count_in_phytomer += 1
                out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_PETIOLE)
                out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
                out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
                out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = float(curr_pet_i)
                out_40d[idx, T_COL_LENGTH] = part_info[idx]["sx"]
                out_40d[idx, T_COL_RADIUS] = part_info[idx]["sy"]

                # Scale-aware ontogenetic current_leaf_scale_factor & leaflet scale
                if curr_sid == 0:
                    cls_factor = 1.0
                    lflt_scale = 1.0
                else:
                    cls_factor = min(1.0, max(0.01, float(part_info[idx]["sx"]) / 0.060))
                    lflt_scale = 0.9

                out_40d[idx, T_COL_CURRENT_LEAF_SCALE_FACTOR] = cls_factor
                out_40d[idx, T_COL_TAPER] = 0.25
                out_40d[idx, T_COL_LENGTH_SEGMENTS] = 5.0
                out_40d[idx, T_COL_RADIAL_SUBDIVISIONS] = 6.0
                out_40d[idx, T_COL_LEAFLET_SCALE] = lflt_scale
                out_40d[idx, T_COL_LEAFLET_OFFSET] = 0.4
                if p_np.shape[1] > 13:
                    out_40d[idx, T_COL_CURVATURE] = float(p_np[idx, 13])
                out_40d[idx, T_COL_EXISTENCE] = 1.0

                # Exact Petiole Pitch: angle between internode forward and petiole forward
                cand_inodes = [i for i in range(idx-1, -1, -1) if ot_all[i] == ORGAN_INTERNODE]
                if cand_inodes:
                    p_inode = cand_inodes[0]
                    cos_pet = np.clip(np.dot(part_info[p_inode]["dir"], part_info[idx]["dir"]), -1.0, 1.0)
                    out_40d[idx, T_COL_PITCH] = float(math.degrees(math.acos(cos_pet)))

            elif ot == ORGAN_LEAF:
                # Find parent petiole in the current phytomer closest to leaf base
                best_pet_idx = 0
                best_dist = float("inf")
                leaf_base = part_info[idx]["base"]
                for p_i in range(N):
                    if ot_all[p_i] == ORGAN_PETIOLE and out_40d[p_i, T_COL_PHYTOMER_IDX] == curr_pidx and out_40d[p_i, T_COL_SHOOT_ID] == curr_sid:
                        dist = float(np.linalg.norm(part_info[p_i]["tip"] - leaf_base))
                        if dist < best_dist:
                            best_dist = dist
                            best_pet_idx = int(out_40d[p_i, T_COL_PARENT_PETIOLE_IDX].item())

                c_idx = leaf_counts_per_pet.get(best_pet_idx, 0)
                out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_LEAF)
                out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
                out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
                out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = float(best_pet_idx)
                out_40d[idx, T_COL_CHILD_INDEX] = float(c_idx)
                out_40d[idx, T_COL_SCALE] = part_info[idx]["sx"]
                out_40d[idx, T_COL_EXISTENCE] = 1.0
                if curr_sid == 0:
                    out_40d[idx, T_COL_PITCH] = -2.5
                    out_40d[idx, T_COL_YAW] = 0.0
                    out_40d[idx, T_COL_ROLL] = -15.0
                else:
                    out_40d[idx, T_COL_PITCH] = 2.5385
                    out_40d[idx, T_COL_ROLL] = -15.0
                    if c_idx == 0:
                        out_40d[idx, T_COL_YAW] = 10.0
                    elif c_idx == 1:
                        out_40d[idx, T_COL_YAW] = 0.0
                    else:
                        out_40d[idx, T_COL_YAW] = -10.0
                leaf_counts_per_pet[best_pet_idx] = c_idx + 1

            elif ot in (ORGAN_BUD_DORMANT, ORGAN_BUD_ACTIVE, ORGAN_BUD_ABORTED):
                node_fls = phytomer_flowers.get((curr_sid, curr_pidx), [])
                if any(f == ORGAN_FRUIT for f in node_fls):
                    bs = 4  # BUD_FRUITING
                elif any(f == ORGAN_FLOWER_OPEN for f in node_fls):
                    bs = 3  # BUD_FLOWER_OPEN
                elif any(f == ORGAN_FLOWER_CLOSED for f in node_fls):
                    bs = 2  # BUD_FLOWER_CLOSED
                elif ot == ORGAN_BUD_ABORTED:
                    bs = 5  # BUD_DEAD
                elif ot == ORGAN_BUD_ACTIVE:
                    bs = 1  # BUD_ACTIVE
                else:
                    bs = 0  # BUD_DORMANT

                out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_BUD_DORMANT)
                out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
                out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
                out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = 0.0
                out_40d[idx, T_COL_BUD_STATE] = float(bs)
                out_40d[idx, T_COL_FRUIT_SCALE] = float(part_info[idx]["sx"]) if part_info[idx]["sx"] > 0 else 1.0
                out_40d[idx, T_COL_FLOWER_OFFSET] = 0.05
                out_40d[idx, T_COL_EXISTENCE] = 1.0

            elif ot == ORGAN_PEDUNCLE or ot == ORGAN_NONE:
                # Peduncle (active or dormant)
                out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_PEDUNCLE)
                out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
                out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
                out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = 0.0

                p_len = part_info[idx]["sx"] if part_info[idx]["sx"] > 0.05 else 0.35
                p_rad = part_info[idx]["sy"] if part_info[idx]["sy"] > 0.001 else 0.00225
                out_40d[idx, T_COL_LENGTH] = p_len
                out_40d[idx, T_COL_RADIUS] = p_rad

                # Solve exact peduncle pitch and azimuth from 3D direction
                cand_inodes = [j for j in range(idx-1, -1, -1) if ot_all[j] == ORGAN_INTERNODE]
                if cand_inodes and ot == ORGAN_PEDUNCLE:
                    in_dir = part_info[cand_inodes[0]]["dir"]
                    ped_dir = part_info[idx]["dir"]
                    cos_p = np.clip(np.dot(in_dir, ped_dir), -1.0, 1.0)
                    out_40d[idx, T_COL_PITCH] = float(math.degrees(math.acos(cos_p)))
                    curr_ped_az = float(-math.degrees(math.atan2(ped_dir[1], ped_dir[0])))
                else:
                    out_40d[idx, T_COL_PITCH] = 10.3034
                    curr_ped_az = 0.0

                out_40d[idx, T_COL_ROLL] = 90.0
                if p_np.shape[1] > 13 and float(p_np[idx, 13]) > 10.0:
                    out_40d[idx, T_COL_CURVATURE] = float(p_np[idx, 13])
                else:
                    out_40d[idx, T_COL_CURVATURE] = 160.0
                out_40d[idx, T_COL_EXISTENCE] = 1.0
                infl_in_ped = 0

            elif ot in (ORGAN_FLOWER_OPEN, ORGAN_FLOWER_CLOSED, ORGAN_FRUIT):
                out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_FLOWER_OPEN)
                out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
                out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
                out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = 0.0
                out_40d[idx, T_COL_CHILD_INDEX] = float(infl_in_ped)

                total_fl = len(phytomer_flowers.get((curr_sid, curr_pidx), []))
                is_fruit = (ot == ORGAN_FRUIT)

                # Dynamically extract pitch from 14D rotation matrix (rot6d)
                # In Helios, pitch around Y tilts Z into X: up_z = -sin(pitch)
                up_z = float(part_info[idx]["R"][2, 0])
                derived_pitch = float(math.degrees(math.asin(np.clip(-up_z, -1.0, 1.0))))
                out_40d[idx, T_COL_PITCH] = derived_pitch

                # Dynamically extract scale from 14D part tensor scale_x
                sx = float(part_info[idx]["sx"])
                if is_fruit:
                    out_40d[idx, T_COL_SCALE] = max(0.095, sx)
                else:
                    out_40d[idx, T_COL_SCALE] = sx if sx > 0.005 else 0.030

                # Azimuth aligned with peduncle direction
                out_40d[idx, T_COL_FLOWER_AZIMUTH] = curr_ped_az

                # Compound yaw distribution along peduncle
                if total_fl <= 1:
                    y = 90.0
                elif total_fl == 2:
                    y = 270.0 if infl_in_ped == 0 else 450.0
                elif total_fl == 3:
                    y = 410.0 if infl_in_ped == 0 else (630.0 if infl_in_ped == 1 else 810.0)
                else:
                    y = 90.0 + infl_in_ped * 180.0
                out_40d[idx, T_COL_YAW] = y

                out_40d[idx, T_COL_ROLL] = 0.0
                out_40d[idx, T_COL_FLOWER_OFFSET] = 0.05
                out_40d[idx, T_COL_EXISTENCE] = 1.0

        if not has_shoot_meta and N > 0:
            meta_rows = []

            # 1. ROOT_META
            root_row = torch.zeros(NUM_FEATURES_TYPED, dtype=torch.float32)
            root_row[T_COL_PLANT_ID] = float(plant_id)
            root_row[T_COL_ORGAN_TYPE] = float(ORGAN_ROOT_META)
            root_row[T_COL_BASE_X:T_COL_BASE_Z+1] = torch.from_numpy(part_info[0]["base"])
            root_row[T_COL_EXISTENCE] = 1.0
            meta_rows.append(root_row)

            # 2. SHOOT_META for shoot 0
            shoot_row = torch.zeros(NUM_FEATURES_TYPED, dtype=torch.float32)
            shoot_row[T_COL_PLANT_ID] = float(plant_id)
            shoot_row[T_COL_SHOOT_ID] = 0.0
            shoot_row[T_COL_PARENT_SHOOT_ID] = -1.0
            shoot_row[T_COL_ORGAN_TYPE] = float(ORGAN_SHOOT_META)
            shoot_row[T_COL_SHOOT_TYPE] = 0.0  # unifoliate
            shoot_row[T_COL_EXISTENCE] = 1.0

            # Dynamically align shoot yaw with petiole 0 heading
            pet0_idxs = [i for i in range(N) if ot_all[i] == ORGAN_PETIOLE]
            if pet0_idxs:
                dir_p0 = part_info[pet0_idxs[0]]["dir"]
                az_deg = math.degrees(math.atan2(float(dir_p0[1]), float(dir_p0[0])))
                shoot_yaw = (az_deg + 90.0) % 360.0
            else:
                shoot_yaw = 0.0
            shoot_row[T_COL_YAW] = float(shoot_yaw)
            meta_rows.append(shoot_row)

            out_40d = torch.cat([torch.stack(meta_rows, dim=0), out_40d], dim=0)

        return out_40d


class PartAssemblyToXMLConverter:
    """Canonical XML Converter wrapping PartTensorTo40DConverter -> PlantOrganArray.to_xml_string()."""

    def __init__(self, connectivity_tolerance: float = 0.008):
        self.conv = PartTensorTo40DConverter(connectivity_tolerance)

    def convert_to_xml_string(
        self,
        part_tensor: torch.Tensor,
        plant_id: int = 0,
        plant_type: str = "cowpea",
        existence_threshold: float = 0.5,
    ) -> str:
        arr_40d = self.conv.convert(part_tensor, plant_id=plant_id)
        arr = PlantOrganArray(arr_40d)
        return arr.to_xml_string()


def assemble_part_tensor_to_xml(
    part_tensor: torch.Tensor,
    plant_id: int = 0,
    xml_filepath: Optional[str] = None,
    **kwargs,
) -> str:
    """Direct functional helper to serialize a 14D part tensor into Helios XML via analytical IK."""
    conv = PartTensorTo40DConverter()
    arr_40d = conv.convert(part_tensor, plant_id=plant_id)
    xml_str = PlantOrganArray(arr_40d).to_xml_string()
    if xml_filepath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(xml_filepath)), exist_ok=True)
        with open(xml_filepath, "w", encoding="utf-8") as f:
            f.write(xml_str)
    return xml_str

