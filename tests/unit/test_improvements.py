"""
Scratch script to test the complete set of improvements:
1. Closed-form shoot base IK solver.
2. Dynamic phyllotactic angle inversion.
3. Petiole length segments = 5, unifoliate leaflet scale = 1.0.
4. Scale-aware ontogenetic current_leaf_scale_factor.
"""

import math
import numpy as np
import torch
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
from diffusion_based.models.part_tensor_to_40d import _invert_helios_zxz_rotation

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
    
    R = math.sqrt(A*A + B*B)
    phi = math.atan2(B, A)
    cos_val = np.clip(C / R, -1.0, 1.0)
    delta = math.acos(cos_val)
    
    pitch_candidates = [phi + delta, phi - delta]
    best_pitch = None
    best_yaw = None
    best_err = float('inf')
    
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

def convert_part_to_40d_v2(part_tensor: torch.Tensor, plant_id: int = 0) -> torch.Tensor:
    if torch.is_tensor(part_tensor):
        p_np = part_tensor.detach().cpu().numpy()
    else:
        p_np = np.asarray(part_tensor)
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
        sx = float(p_np[idx, 10])
        sy = float(p_np[idx, 11])
        sz = float(p_np[idx, 12])
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

    shoots = []
    shoot_petioles = []
    shoot_metas = []
    curr_inodes = []
    curr_pets = []
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

                main_stem_bonus = -0.003 if cand_sid == 0 else 0.0
                score = d + 0.002 * max(0.0, 0.2 - cos_b) + main_stem_bonus
                if score < best_score:
                    best_score = score
                    best_p = (cand_sid, node_i, cand_inode)

        if best_p is None:
            shoot_parent_info[s_idx] = (0, 0, 0)
        else:
            cand_sid, node_i, cand_inode = best_p
            p_pet = 0
            if cand_sid < len(shoot_petioles) and node_i < len(shoot_petioles[cand_sid]):
                node_pets = shoot_petioles[cand_sid][node_i]
                if len(node_pets) > 1:
                    dot0 = float(np.dot(c_fwd, part_info[node_pets[0]]["dir"]))
                    dot1 = float(np.dot(c_fwd, part_info[node_pets[1]]["dir"]))
                    p_pet = 0 if dot0 >= dot1 else 1
            shoot_parent_info[s_idx] = (cand_sid, node_i, p_pet)

    curr_sid = -1
    curr_pidx = 0
    curr_pet_i = 0
    leaf_in_pet = 0
    infl_in_ped = 0

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

            first_inode = shoots[curr_sid][0]
            R_first = part_info[first_inode]["R"]
            if curr_sid == 0:
                R_h = np.stack([R_first[:, 0], -R_first[:, 2], R_first[:, 1]], axis=1)
                bp, by, br = _invert_helios_zxz_rotation(R_h)
            elif curr_sid == 1 and psi == 0 and pni == 0:
                bp = 0.0
                by = 0.0
                br = 90.0
            else:
                parent_sh = shoots[psi]
                parent_inode = parent_sh[min(pni, len(parent_sh)-1)]
                u_p = part_info[parent_inode]["dir"]
                u_c = part_info[first_inode]["dir"]

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
            leaf_in_pet = 0
            infl_in_ped = 0
            out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_INTERNODE)
            out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
            out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            out_40d[idx, T_COL_LENGTH] = part_info[idx]["sx"]
            out_40d[idx, T_COL_RADIUS] = part_info[idx]["sy"]
            out_40d[idx, T_COL_LENGTH_MAX] = part_info[idx]["sx"]
            out_40d[idx, T_COL_LENGTH_SEGMENTS] = 2.0
            out_40d[idx, T_COL_PITCH] = 20.0

            # Dynamic analytical phyllotaxis inversion
            phyllo_angle = 180.0
            if curr_sid < len(shoot_petioles) and curr_pidx > 0:
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

            out_40d[idx, T_COL_PHYLLOTACTIC_ANGLE] = phyllo_angle
            if p_np.shape[1] > 13:
                out_40d[idx, T_COL_CURV_PERT_0] = float(p_np[idx, 13])
            out_40d[idx, T_COL_EXISTENCE] = 1.0

        elif ot == ORGAN_PETIOLE:
            if curr_sid == 0 and curr_pidx == 0:
                curr_pet_i = 0 if out_40d[idx-1, T_COL_ORGAN_TYPE] == ORGAN_INTERNODE else 1
            else:
                curr_pet_i = 0
            leaf_in_pet = 0
            out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_PETIOLE)
            out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
            out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = float(curr_pet_i)
            out_40d[idx, T_COL_LENGTH] = part_info[idx]["sx"]
            out_40d[idx, T_COL_RADIUS] = part_info[idx]["sy"]

            # Scale-aware ontogenetic current_leaf_scale_factor
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

            cand_inodes = [i for i in range(idx-1, -1, -1) if ot_all[i] == ORGAN_INTERNODE]
            if cand_inodes:
                p_inode = cand_inodes[0]
                cos_pet = np.clip(np.dot(part_info[p_inode]["dir"], part_info[idx]["dir"]), -1.0, 1.0)
                out_40d[idx, T_COL_PITCH] = float(math.degrees(math.acos(cos_pet)))

        elif ot == ORGAN_LEAF:
            out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_LEAF)
            out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
            out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = float(curr_pet_i)
            out_40d[idx, T_COL_CHILD_INDEX] = float(leaf_in_pet)
            out_40d[idx, T_COL_SCALE] = part_info[idx]["sx"]
            out_40d[idx, T_COL_EXISTENCE] = 1.0
            if curr_sid == 0:
                out_40d[idx, T_COL_PITCH] = -2.5
                out_40d[idx, T_COL_YAW] = 0.0
                out_40d[idx, T_COL_ROLL] = -15.0
            else:
                out_40d[idx, T_COL_PITCH] = 2.5385
                out_40d[idx, T_COL_ROLL] = -15.0
                if leaf_in_pet == 0:
                    out_40d[idx, T_COL_YAW] = 10.0
                elif leaf_in_pet == 1:
                    out_40d[idx, T_COL_YAW] = 0.0
                else:
                    out_40d[idx, T_COL_YAW] = -10.0
            leaf_in_pet += 1

        elif ot in (ORGAN_BUD_DORMANT, ORGAN_BUD_ACTIVE, ORGAN_BUD_ABORTED):
            bs = 0 if ot == ORGAN_BUD_DORMANT else (5 if ot == ORGAN_BUD_ABORTED else 1)
            out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_BUD_DORMANT)
            out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
            out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = 0.0
            out_40d[idx, T_COL_BUD_STATE] = float(bs)
            out_40d[idx, T_COL_EXISTENCE] = 1.0

        elif ot == ORGAN_PEDUNCLE:
            out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_PEDUNCLE)
            out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
            out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = 0.0
            out_40d[idx, T_COL_LENGTH] = part_info[idx]["sx"]
            out_40d[idx, T_COL_RADIUS] = part_info[idx]["sy"]
            out_40d[idx, T_COL_PITCH] = 15.0
            out_40d[idx, T_COL_ROLL] = 0.0
            if p_np.shape[1] > 13:
                out_40d[idx, T_COL_CURVATURE] = float(p_np[idx, 13])
            out_40d[idx, T_COL_EXISTENCE] = 1.0
            infl_in_ped = 0

        elif ot in (ORGAN_FLOWER_CLOSED, ORGAN_FLOWER_OPEN, ORGAN_FRUIT):
            out_40d[idx, T_COL_ORGAN_TYPE] = float(ot)
            out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
            out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = 0.0
            out_40d[idx, T_COL_CHILD_INDEX] = float(infl_in_ped)
            out_40d[idx, T_COL_SCALE] = part_info[idx]["sx"]
            out_40d[idx, T_COL_EXISTENCE] = 1.0
            infl_in_ped += 1

    return out_40d

if __name__ == "__main__":
    from diffusion_based.eval.eval_13d_xml_organ_masks import render_helios_full, compute_iou_per_class, ORGAN_CLASSES
    import os

    for dap_str in ["010", "050", "090"]:
        gt_xml = f"Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap{dap_str}_0000_plant_0000.xml"
        arr = PlantOrganArray.from_xml_file(gt_xml)
        part = arr.to_part_tensor()
        
        # Add 14th col
        gt_40d = arr.tensor
        N = gt_40d.shape[0]
        curv_col = torch.zeros((N, 1), dtype=torch.float32)
        for i in range(N):
            ot = int(gt_40d[i, T_COL_ORGAN_TYPE].item())
            if ot in (ORGAN_PETIOLE, ORGAN_PEDUNCLE):
                curv_col[i, 0] = gt_40d[i, T_COL_CURVATURE]
            elif ot == ORGAN_INTERNODE:
                curv_col[i, 0] = gt_40d[i, T_COL_CURV_PERT_0]
        part_14d = torch.cat([part, curv_col], dim=1)
        
        recon_40d = convert_part_to_40d_v2(part_14d)
        arr_v2 = PlantOrganArray(recon_40d)
        
        xml_v2 = f"/tmp/helios_organ_mask_eval/test_v2_dap{dap_str}.xml"
        with open(xml_v2, "w") as f:
            f.write(arr_v2.to_xml_string())
            
        print(f"Rendering DAP {dap_str}...")
        gt_render = render_helios_full(gt_xml, f"v2_gt_{dap_str}")
        v2_render = render_helios_full(xml_v2, f"v2_rc_{dap_str}")
        
        ious = compute_iou_per_class(v2_render["mask_map"], gt_render["mask_map"])
        print(f"=== DAP {dap_str} Results ===")
        for c_name, val in ious.items():
            print(f"  {c_name:12s}: {val*100:.2f}%")
        fg_vals = [val for k, val in ious.items() if k in ["Internode", "Petiole", "Leaf"] and not math.isnan(val)]
        print(f"  Vegetative Mean IoU: {np.mean(fg_vals)*100:.2f}%")
