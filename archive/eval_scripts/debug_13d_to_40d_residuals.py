"""
13D Part Tensor -> 40D Organ Array Column-by-Column Residual Diagnostics.

Compares:
  - Ground Truth 40D Tensor (PlantOrganArray.from_xml_file)
  - Reconstructed 40D Tensor (13D Part Tensor -> 40D Inverter)

Measures:
  - Mean Absolute Error (MAE) and Max Absolute Error for every continuous column
  - Categorical Accuracy for discrete columns (organ_type, shoot_id, parent links, bud_state)
  - Pinpoints exact organ types and columns with numerical discrepancies
"""

import os
import math
import numpy as np
import torch
from scipy.spatial import cKDTree

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
from diffusion_based.models.part_assembly_to_xml import PartAssemblyToXMLConverter, _invert_helios_zxz_rotation, _rot_z_matrix


def reconstruct_40d_from_13d(part_tensor: torch.Tensor) -> torch.Tensor:
    """
    Inverse mapper: (N, 13) World-Space Part Tensor -> (N, 40) PlantOrganArray Typed Tensor.
    """
    p_np = part_tensor.detach().cpu().numpy()
    N = p_np.shape[0]

    # Initialize 40D output tensor
    recon = torch.zeros((N, NUM_FEATURES_TYPED), dtype=torch.float32)

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

    # Group shoots and phytomers in sequential order
    curr_sid = -1
    curr_pidx = 0
    curr_inode_idx = None
    curr_pet_idx = 0
    leaf_in_pet = 0

    shoot_meta_rows = []
    for idx in range(N):
        if not active_mask[idx]:
            continue
        ot = ot_all[idx]
        if ot == ORGAN_ROOT_META:
            recon[idx, T_COL_ORGAN_TYPE] = float(ORGAN_ROOT_META)
            recon[idx, T_COL_BASE_X:T_COL_BASE_Z+1] = torch.from_numpy(part_info[idx]["base"])
            recon[idx, T_COL_PLANT_AGE] = part_info[idx]["sx"]
            recon[idx, T_COL_EXISTENCE] = 1.0

        elif ot == ORGAN_SHOOT_META:
            curr_sid += 1
            curr_pidx = -1
            shoot_meta_rows.append(idx)
            recon[idx, T_COL_ORGAN_TYPE] = float(ORGAN_SHOOT_META)
            recon[idx, T_COL_SHOOT_ID] = float(curr_sid)
            recon[idx, T_COL_SHOOT_TYPE] = 0.0 if curr_sid == 0 else 1.0
            recon[idx, T_COL_EXISTENCE] = 1.0

        elif ot == ORGAN_INTERNODE:
            curr_pidx += 1
            curr_inode_idx = idx
            curr_pet_idx = 0
            leaf_in_pet = 0
            recon[idx, T_COL_ORGAN_TYPE] = float(ORGAN_INTERNODE)
            recon[idx, T_COL_SHOOT_ID] = float(curr_sid)
            recon[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            recon[idx, T_COL_LENGTH] = part_info[idx]["sx"]
            recon[idx, T_COL_RADIUS] = part_info[idx]["sy"]
            recon[idx, T_COL_LENGTH_MAX] = part_info[idx]["sx"]
            recon[idx, T_COL_LENGTH_SEGMENTS] = 2.0
            recon[idx, T_COL_EXISTENCE] = 1.0

        elif ot == ORGAN_PETIOLE:
            curr_pet_idx = 0 if curr_sid > 0 else (0 if recon[idx-1, T_COL_ORGAN_TYPE] == ORGAN_INTERNODE else 1)
            leaf_in_pet = 0
            recon[idx, T_COL_ORGAN_TYPE] = float(ORGAN_PETIOLE)
            recon[idx, T_COL_SHOOT_ID] = float(curr_sid)
            recon[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            recon[idx, T_COL_PARENT_PETIOLE_IDX] = float(curr_pet_idx)
            recon[idx, T_COL_LENGTH] = part_info[idx]["sx"]
            recon[idx, T_COL_RADIUS] = part_info[idx]["sy"]
            recon[idx, T_COL_TAPER] = 0.25
            recon[idx, T_COL_RADIAL_SUBDIVISIONS] = 6.0
            recon[idx, T_COL_LEAFLET_SCALE] = 0.9
            recon[idx, T_COL_LEAFLET_OFFSET] = 0.4
            recon[idx, T_COL_EXISTENCE] = 1.0

            # Pitch between internode forward and petiole forward
            if curr_inode_idx is not None:
                cos_pet = np.clip(np.dot(part_info[curr_inode_idx]["dir"], part_info[idx]["dir"]), -1.0, 1.0)
                recon[idx, T_COL_PITCH] = float(math.degrees(math.acos(cos_pet)))

        elif ot == ORGAN_LEAF:
            recon[idx, T_COL_ORGAN_TYPE] = float(ORGAN_LEAF)
            recon[idx, T_COL_SHOOT_ID] = float(curr_sid)
            recon[idx, T_COL_PHYTOMER_IDX] = float(curr_pidx)
            recon[idx, T_COL_PARENT_PETIOLE_IDX] = float(curr_pet_idx)
            recon[idx, T_COL_CHILD_INDEX] = float(leaf_in_pet)
            recon[idx, T_COL_SCALE] = part_info[idx]["sx"]
            recon[idx, T_COL_EXISTENCE] = 1.0
            leaf_in_pet += 1

    return recon


def run_residual_diagnostics(dap_str: str = "010"):
    xml_path = f"Digital-Crops/projects/syntheticdata_generation/build/output/exact_gt_renders/rad_dap{dap_str}_0000_plant_0000.xml"
    if not os.path.exists(xml_path):
        print(f"File not found: {xml_path}")
        return

    print(f"\n================================================================================")
    print(f"RUNNING 13D -> 40D RESIDUAL DIAGNOSTICS (DAP {dap_str})")
    print(f"================================================================================")

    arr = PlantOrganArray.from_xml_file(xml_path)
    gt_40d = arr.tensor.cpu()
    part_13d = arr.to_part_tensor().cpu()

    from diffusion_based.models.part_tensor_to_40d import PartTensorTo40DConverter
    converter = PartTensorTo40DConverter()
    recon_40d = converter.convert(part_13d)

    N = gt_40d.shape[0]
    print(f"Total Rows: {N}")

    col_names = {
        T_COL_PLANT_ID: "PLANT_ID",
        T_COL_PLANT_AGE: "PLANT_AGE",
        T_COL_BASE_X: "BASE_X",
        T_COL_BASE_Y: "BASE_Y",
        T_COL_BASE_Z: "BASE_Z",
        T_COL_SHOOT_ID: "SHOOT_ID",
        T_COL_PARENT_SHOOT_ID: "PARENT_SHOOT_ID",
        T_COL_PARENT_NODE_IDX: "PARENT_NODE_IDX",
        T_COL_PARENT_PETIOLE_IDX: "PARENT_PET_IDX",
        T_COL_PHYTOMER_IDX: "PHYTOMER_IDX",
        T_COL_CHILD_INDEX: "CHILD_INDEX",
        T_COL_ORGAN_TYPE: "ORGAN_TYPE",
        T_COL_SHOOT_TYPE: "SHOOT_TYPE",
        T_COL_LENGTH: "LENGTH",
        T_COL_RADIUS: "RADIUS",
        T_COL_SCALE: "SCALE",
        T_COL_PITCH: "PITCH",
        T_COL_YAW: "YAW",
        T_COL_ROLL: "ROLL",
        T_COL_CURVATURE: "CURVATURE",
        T_COL_PHYLLOTACTIC_ANGLE: "PHYLLO_ANG",
        T_COL_LENGTH_MAX: "LENGTH_MAX",
        T_COL_LENGTH_SEGMENTS: "LEN_SEGS",
        T_COL_BUD_STATE: "BUD_STATE",
        T_COL_EXISTENCE: "EXISTENCE",
    }

    print(f"\n--- Column-by-Column Residual Summary (GT vs Recon) ---")
    print(f"{'Col Name':<20} | {'MAE':<12} | {'Max Err':<12} | {'Exact Match %':<15}")
    print("-" * 65)

    for col, name in sorted(col_names.items()):
        gt_col = gt_40d[:, col].numpy()
        rc_col = recon_40d[:, col].numpy()

        diff = np.abs(gt_col - rc_col)
        mae = float(np.mean(diff))
        max_err = float(np.max(diff))
        exact = float(np.mean(diff < 1e-4) * 100.0)

        print(f"{name:<20} | {mae:<12.5f} | {max_err:<12.5f} | {exact:<14.1f}%")

    # Organ-by-organ breakdown
    print(f"\n--- Organ-by-Organ Breakdown of Angle & Scale Residuals ---")
    organ_names = {
        ORGAN_ROOT_META: "ROOT_META",
        ORGAN_SHOOT_META: "SHOOT_META",
        ORGAN_INTERNODE: "INTERNODE",
        ORGAN_PETIOLE: "PETIOLE",
        ORGAN_LEAF: "LEAF",
        ORGAN_BUD_DORMANT: "BUD",
        ORGAN_PEDUNCLE: "PEDUNCLE",
        ORGAN_FLOWER_OPEN: "FLOWER",
    }
    gt_ot = gt_40d[:, T_COL_ORGAN_TYPE].numpy().astype(int)
    for ot_val, ot_name in sorted(organ_names.items()):
        mask = (gt_ot == ot_val)
        if not np.any(mask):
            continue
        p_mae = float(np.mean(np.abs(gt_40d[mask, T_COL_PITCH].numpy() - recon_40d[mask, T_COL_PITCH].numpy())))
        y_mae = float(np.mean(np.abs(gt_40d[mask, T_COL_YAW].numpy() - recon_40d[mask, T_COL_YAW].numpy())))
        r_mae = float(np.mean(np.abs(gt_40d[mask, T_COL_ROLL].numpy() - recon_40d[mask, T_COL_ROLL].numpy())))
        l_mae = float(np.mean(np.abs(gt_40d[mask, T_COL_LENGTH].numpy() - recon_40d[mask, T_COL_LENGTH].numpy())))
        s_mae = float(np.mean(np.abs(gt_40d[mask, T_COL_SCALE].numpy() - recon_40d[mask, T_COL_SCALE].numpy())))
        print(f"{ot_name:<12} (N={np.sum(mask):2d}) | Pitch MAE: {p_mae:6.2f}° | Yaw MAE: {y_mae:6.2f}° | Roll MAE: {r_mae:6.2f}° | Len MAE: {l_mae:.4f}m | Scale MAE: {s_mae:.4f}")


if __name__ == "__main__":
    import sys
    dap = sys.argv[1] if len(sys.argv) > 1 else "010"
    run_residual_diagnostics(dap)
