"""
Test improved reproductive organ conversion for DAP 90:
1. Two-pass or phytomer-lookahead bud_state determination:
   - If phytomer has ORGAN_FRUIT (11): bud_state = 4 (BUD_FRUITING)
   - If phytomer has ORGAN_FLOWER_OPEN (10): bud_state = 3 (BUD_FLOWER_OPEN)
   - If phytomer has ORGAN_FLOWER_CLOSED (9): bud_state = 2 (BUD_FLOWER_CLOSED)
   - If phytomer has ORGAN_BUD_ABORTED (12): bud_state = 5 (BUD_DEAD)
   - If phytomer has ORGAN_BUD_ACTIVE (8): bud_state = 1 (BUD_ACTIVE)
   - If phytomer has ORGAN_BUD_DORMANT (7): bud_state = 0 (BUD_DORMANT)
2. current_fruit_scale_factor = part_info[bud]["sx"] if sx > 0 else 1.0
3. flower_offset = 0.05
4. peduncle: length = sx if sx > 0.05 else 0.35, radius = sy if sy > 0.001 else 0.00225, pitch = 15.0, roll = 90.0, existence = 1.0
5. flower rows: organ_type = 10, scale = sx, pitch = -7.0, yaw = 90.0 + infl_in_ped * 180.0, existence = 1.0
"""

import math
import numpy as np
import torch
import xml.etree.ElementTree as ET

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
from diffusion_based.models.part_tensor_to_40d import solve_helios_shoot_base
from diffusion_based.models.part_tensor_to_40d import _invert_helios_zxz_rotation

def convert_with_reproductive_fix(part_tensor: torch.Tensor, plant_id: int = 0) -> torch.Tensor:
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

    # Grouping passes
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

    # Shoot parent tree
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

    # Pass 1.5: Identify phytomer organ contents (to know bud_state beforehand)
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

    # Pass 2: Reconstruct 40D rows
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
            # Check what flowers or fruits are on this phytomer
            node_fls = phytomer_flowers.get((curr_sid, curr_pidx), [])
            if any(f == ORGAN_FRUIT for f in node_fls):
                bs = 4 # BUD_FRUITING
            elif any(f == ORGAN_FLOWER_OPEN for f in node_fls):
                bs = 3 # BUD_FLOWER_OPEN
            elif any(f == ORGAN_FLOWER_CLOSED for f in node_fls):
                bs = 2 # BUD_FLOWER_CLOSED
            elif ot == ORGAN_BUD_ABORTED:
                bs = 5 # BUD_DEAD
            elif ot == ORGAN_BUD_ACTIVE:
                bs = 1 # BUD_ACTIVE
            else:
                bs = 0 # BUD_DORMANT

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
            out_40d[idx, T_COL_PITCH] = 15.0
            out_40d[idx, T_COL_ROLL] = 90.0
            if p_np.shape[1] > 13 and float(p_np[idx, 13]) > 10.0:
                out_40d[idx, T_COL_CURVATURE] = float(p_np[idx, 13])
            else:
                out_40d[idx, T_COL_CURVATURE] = 160.0
            out_40d[idx, T_COL_EXISTENCE] = 1.0
            infl_in_ped = 0

        elif ot in (ORGAN_FLOWER_CLOSED, ORGAN_FLOWER_OPEN, ORGAN_FRUIT):
            out_40d[idx, T_COL_ORGAN_TYPE] = float(ORGAN_FLOWER_OPEN)
            out_40d[idx, T_COL_SHOOT_ID] = float(curr_sid)
            out_40d[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            out_40d[idx, T_COL_PARENT_PETIOLE_IDX] = 0.0
            out_40d[idx, T_COL_CHILD_INDEX] = float(infl_in_ped)
            out_40d[idx, T_COL_SCALE] = part_info[idx]["sx"]
            out_40d[idx, T_COL_PITCH] = -7.0
            out_40d[idx, T_COL_YAW] = 90.0 + infl_in_ped * 180.0
            out_40d[idx, T_COL_ROLL] = 0.0
            out_40d[idx, T_COL_FLOWER_AZIMUTH] = 0.0
            out_40d[idx, T_COL_FLOWER_OFFSET] = 0.05
            out_40d[idx, T_COL_EXISTENCE] = 1.0
            infl_in_ped += 1

    return out_40d

if __name__ == "__main__":
    from diffusion_based.eval.eval_13d_xml_organ_masks import render_helios_full, compute_iou_per_class, ORGAN_CLASSES
    dap_str = "090"
    gt_xml = f"Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap{dap_str}_0000_plant_0000.xml"
    arr = PlantOrganArray.from_xml_file(gt_xml)
    part = arr.to_part_tensor()

    gt_40d = arr.tensor
    N = gt_40d.shape[0]
    curv_col = torch.zeros((N, 1), dtype=torch.float32)
    for i in range(N):
        ot = int(gt_40d[i, 11].item())
        if ot in (ORGAN_PETIOLE, ORGAN_PEDUNCLE):
            curv_col[i, 0] = gt_40d[i, 19]
        elif ot == ORGAN_INTERNODE:
            curv_col[i, 0] = gt_40d[i, 23]
    part_14d = torch.cat([part, curv_col], dim=1)

    recon_40d = convert_with_reproductive_fix(part_14d)
    arr_rep = PlantOrganArray(recon_40d)

    xml_rep = f"/tmp/helios_organ_mask_eval/test_rep_dap{dap_str}.xml"
    with open(xml_rep, "w") as f:
        f.write(arr_rep.to_xml_string())

    tree_rep = ET.fromstring(arr_rep.to_xml_string())
    print(f"Reproductive fix XML tags in DAP {dap_str}:")
    print("  floral_buds:", len(tree_rep.findall(".//floral_bud")))
    print("  peduncles:", len(tree_rep.findall(".//peduncle")))
    print("  inflorescences:", len(tree_rep.findall(".//inflorescence")))
    print("  flowers:", len(tree_rep.findall(".//flower")))

    from collections import Counter
    bss = [int(b.find("bud_state").text) for b in tree_rep.findall(".//floral_bud")]
    print("  bud_states:", Counter(bss))

    print(f"Rendering DAP {dap_str} with Helios C++...")
    gt_render = render_helios_full(gt_xml, f"rep_gt_{dap_str}")
    rep_render = render_helios_full(xml_rep, f"rep_rc_{dap_str}")

    ious = compute_iou_per_class(rep_render["mask_map"], gt_render["mask_map"])
    print(f"=== DAP {dap_str} Results with Reproductive Fix ===")
    for c_name, val in ious.items():
        print(f"  {c_name:12s}: {val*100:.2f}%")

    fg_gt = (gt_render["mask_map"] >= 0)
    fg_rc = (rep_render["mask_map"] >= 0)
    fg_iou = np.logical_and(fg_gt, fg_rc).sum() / max(1, np.logical_or(fg_gt, fg_rc).sum())
    print(f"  Foreground IoU : {fg_iou*100:.2f}%")
