"""
Plant Organ Array: part-centric (N, 14) representation and XML utilities.

This module stores plant architecture as a dimension-agnostic part-centric
tensor. The current layout uses 14 columns (organ type, base XYZ, 6D rotation,
scale XYZ, existence), but the class is dimension-agnostic: any tensor with
NUM_FEATURES_14D columns is accepted.
"""

import math
from typing import Optional
import torch
import numpy as np
import xml.etree.ElementTree as ET


# =============================================================================
# ORGAN TYPE IDs
# =============================================================================

ORGAN_ROOT_META = 0
ORGAN_SHOOT_META = 1
ORGAN_INTERNODE = 2
ORGAN_PETIOLE = 3
ORGAN_LEAF = 4
ORGAN_BUD = 5
ORGAN_PEDUNCLE = 6
ORGAN_FLOWER = 7
ORGAN_FRUIT = 8
ORGAN_FLOWER_CLOSED = 9


# =============================================================================
# PART-CENTRIC COLUMN IDs
# =============================================================================

P14_COL_ORGAN_TYPE = 0
P14_COL_BASE_X = 1
P14_COL_BASE_Y = 2
P14_COL_BASE_Z = 3
P14_COL_ROT_0 = 4
P14_COL_ROT_1 = 5
P14_COL_ROT_2 = 6
P14_COL_ROT_3 = 7
P14_COL_ROT_4 = 8
P14_COL_ROT_5 = 9
P14_COL_SCALE_X = 10
P14_COL_SCALE_Y = 11
P14_COL_SCALE_Z = 12
P14_COL_EXISTENCE = 13
NUM_FEATURES_14D = 14


# =============================================================================
# ROTATION HELPERS
# =============================================================================

def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Converts 3x3 rotation matrix (or (..., 3, 3)) to continuous 6D representation
    by taking the first two column vectors.
    """
    if R.dim() == 2:
        return torch.cat([R[:, 0], R[:, 1]], dim=0)
    col1 = R[..., :, 0]
    col2 = R[..., :, 1]
    return torch.cat([col1, col2], dim=-1)


def rotation_6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D continuous rotation representation (..., 6) to 3x3 rotation
    matrix (..., 3, 3) using Gram-Schmidt orthogonalization (Zhou et al.,
    CVPR 2019).
    """
    if r6.dim() == 1:
        r6_b = r6.unsqueeze(0)
    else:
        r6_b = r6

    a1 = r6_b[..., 0:3]
    a2 = r6_b[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-1)
    if r6.dim() == 1:
        return R.squeeze(0)
    return R


def _rotation_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Tait-Bryan XYZ rotation matrix (roll, pitch, yaw in radians)."""
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )
    return R


def _matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert a numpy 3x3 rotation matrix to a 6D continuous representation."""
    r6 = rotation_matrix_to_6d(torch.from_numpy(R).float())
    return r6.numpy()


def _make_row(
    organ_type: int,
    base: np.ndarray,
    R: np.ndarray,
    scale: np.ndarray,
    existence: float = 1.0,
) -> np.ndarray:
    """Build a single (NUM_FEATURES_14D,) row from spatial part data."""
    row = np.zeros(NUM_FEATURES_14D, dtype=np.float32)
    row[P14_COL_ORGAN_TYPE] = float(organ_type)
    row[P14_COL_BASE_X] = float(base[0])
    row[P14_COL_BASE_Y] = float(base[1])
    row[P14_COL_BASE_Z] = float(base[2])
    r6 = _matrix_to_6d(R)
    row[P14_COL_ROT_0:P14_COL_ROT_5 + 1] = r6
    row[P14_COL_SCALE_X] = float(scale[0])
    row[P14_COL_SCALE_Y] = float(scale[1])
    row[P14_COL_SCALE_Z] = float(scale[2])
    row[P14_COL_EXISTENCE] = float(existence)
    return row


def _parse_xml_to_part_tensor(xml_content: str) -> torch.Tensor:
    """
    Direct XML -> part tensor parser.

    Parses a Helios XML document into a flat list of per-organ records, then runs
    the same forward-kinematics chain as the original typed mesh builder
    (internode/petiole/leaf curvature, phyllotaxis, leaf roll/yaw, peduncle and
    flower/fruit orientation) to produce the part-centric tensor directly, with
    no intermediate 40D typed array.
    """
    from diffusion_based.models.helios_pytorch_geometry import (
        rotr_x, rotr_y, rotr_z,
        rotate_vector_about_axis,
        rotate_points_about_axis,
        rodrigues_matrix_torch,
        interpolate_tube_torch,
        get_axis_vector_torch,
        clamp_offset_torch,
        _get_rotation_matrix_between_vectors_batch,
    )

    root = ET.fromstring(xml_content)
    if root.tag != "helios":
        raise ValueError("Root tag must be <helios>")

    device = torch.device("cpu")
    deg2rad = torch.tensor(math.pi / 180.0, dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
    gravitropic_curvature = 200.0

    # ------------------------------------------------------------------
    # Phase 0: parse XML into a flat list of organ records (dicts).
    # Each record carries the same fields the old 40D typed rows carried.
    # ------------------------------------------------------------------
    records: list = []  # list of dicts

    for plant_elem in root.findall("plant_instance"):
        plant_id = int(plant_elem.attrib.get("ID", plant_elem.attrib.get("id", 0)))
        bp_text = _get_text_default(plant_elem, "base_position", None)
        if bp_text is None:
            bp_text = _get_text_default(plant_elem, "plant_base_position", "0 0 0")
        bp_vals = [float(x) for x in bp_text.replace(";", " ").split() if x.strip()]
        if len(bp_vals) < 3:
            bp_vals = [0.0, 0.0, 0.0]
        plant_base = torch.tensor(bp_vals[:3], dtype=torch.float32, device=device)
        plant_age = _get_float_text(plant_elem, "plant_age", 0.0)

        records.append({
            "organ_type": ORGAN_ROOT_META,
            "plant_id": plant_id,
            "plant_age": plant_age,
            "base": plant_base,
        })

        for shoot_elem in plant_elem.findall("shoot"):
            sid = int(shoot_elem.attrib.get("ID", shoot_elem.attrib.get("shoot_id", shoot_elem.attrib.get("id", 0))))
            stl = _get_text_default(shoot_elem, "shoot_type_label", "unifoliate")
            shoot_type = 0 if "unifoliate" in (stl or "") else 1
            psi = _get_int_text(shoot_elem, "parent_shoot_ID", -1)
            pni = _get_int_text(shoot_elem, "parent_node_index", 0)
            ppi = _get_int_text(shoot_elem, "parent_petiole_index", 0)
            br_text = _get_text_default(shoot_elem, "base_rotation", None)
            if br_text is not None:
                br_vals = [float(x) for x in br_text.split() if x.strip()]
                br_pitch, br_yaw, br_roll = (br_vals + [0.0, 0.0, 0.0])[:3]
            else:
                br_pitch = _get_float_text(shoot_elem, "shoot_base_pitch", 0.0)
                br_yaw = _get_float_text(shoot_elem, "shoot_base_yaw", 0.0)
                br_roll = _get_float_text(shoot_elem, "shoot_base_roll", 0.0)

            records.append({
                "organ_type": ORGAN_SHOOT_META,
                "shoot_id": sid,
                "shoot_type": shoot_type,
                "parent_shoot_id": psi,
                "parent_node_idx": pni,
                "parent_petiole_idx": ppi,
                "pitch": br_pitch,
                "yaw": br_yaw,
                "roll": br_roll,
            })

            for phyto_idx, phyto_elem in enumerate(shoot_elem.findall("phytomer")):
                internode_elem = phyto_elem.find("internode")
                if internode_elem is None:
                    continue

                il = _get_float_text(internode_elem, "internode_length", 0.0)
                ir = _get_float_text(internode_elem, "internode_radius", 0.0)
                ip = _get_float_text(internode_elem, "internode_pitch", 0.0)
                ipa = _get_float_text(internode_elem, "internode_phyllotactic_angle", 0.0)
                ilm = _get_float_text(internode_elem, "internode_length_max", 0.0)
                ils = _get_int_text(internode_elem, "internode_length_segments", 2)
                cp_text = _get_text_default(internode_elem, "curvature_perturbations", "0;0")
                cp_list = [float(x) for x in (cp_text or "").split(";") if x.strip()]
                cp0 = cp_list[0] if len(cp_list) > 0 else 0.0
                cp1 = cp_list[1] if len(cp_list) > 1 else 0.0
                yp_text = _get_text_default(internode_elem, "yaw_perturbations", "0;0")
                yp_list = [float(x) for x in (yp_text or "").split(";") if x.strip()]
                yp0 = yp_list[0] if len(yp_list) > 0 else 0.0
                yp1 = yp_list[1] if len(yp_list) > 1 else 0.0

                records.append({
                    "organ_type": ORGAN_INTERNODE,
                    "shoot_id": sid,
                    "phytomer_idx": phyto_idx,
                    "length": il,
                    "radius": ir,
                    "pitch": ip,
                    "phyllotactic_angle": ipa,
                    "length_max": ilm,
                    "length_segments": ils,
                    "curv_pert_0": cp0,
                    "curv_pert_1": cp1,
                    "yaw_pert_0": yp0,
                    "yaw_pert_1": yp1,
                })

                for pet_i, pet_elem in enumerate(internode_elem.findall("petiole")):
                    pl = _get_float_text(pet_elem, "petiole_length", 0.0)
                    pr = _get_float_text(pet_elem, "petiole_radius", 0.0)
                    pp = _get_float_text(pet_elem, "petiole_pitch", 0.0)
                    pc = _get_float_text(pet_elem, "petiole_curvature", 0.0)
                    cls_val = _get_float_text(pet_elem, "current_leaf_scale_factor", 1.0)
                    pt = _get_float_text(pet_elem, "petiole_taper", 0.25)
                    pls = _get_int_text(pet_elem, "petiole_length_segments", 5)
                    lflt_scale = _get_float_text(pet_elem, "leaflet_scale", 1.0)
                    lflt_offset = _get_float_text(pet_elem, "leaflet_offset", 0.4)

                    records.append({
                        "organ_type": ORGAN_PETIOLE,
                        "shoot_id": sid,
                        "phytomer_idx": phyto_idx,
                        "parent_petiole_idx": pet_i,
                        "length": pl,
                        "radius": pr,
                        "pitch": pp,
                        "curvature": pc,
                        "current_leaf_scale_factor": cls_val,
                        "taper": pt,
                        "length_segments": pls,
                        "leaflet_scale": lflt_scale,
                        "leaflet_offset": lflt_offset,
                    })

                    for lf_idx, leaf_elem in enumerate(pet_elem.findall("leaf")):
                        lfs = _get_float_text(leaf_elem, "leaf_scale", 1.0)
                        lfp = _get_float_text(leaf_elem, "leaf_pitch", 0.0)
                        lfy = _get_float_text(leaf_elem, "leaf_yaw", 0.0)
                        lfr = _get_float_text(leaf_elem, "leaf_roll", 0.0)
                        records.append({
                            "organ_type": ORGAN_LEAF,
                            "shoot_id": sid,
                            "phytomer_idx": phyto_idx,
                            "parent_petiole_idx": pet_i,
                            "child_index": lf_idx,
                            "scale": lfs,
                            "pitch": lfp,
                            "yaw": lfy,
                            "roll": lfr,
                        })

                    fb_elem = pet_elem.find("floral_bud")
                    if fb_elem is not None:
                        bs = _get_int_text(fb_elem, "bud_state", 5)
                        bpi = _get_int_text(fb_elem, "parent_index", 0)
                        bidx = _get_int_text(fb_elem, "bud_index", 0)
                        biterm = _get_int_text(fb_elem, "is_terminal", 0)
                        bcfs = _get_float_text(fb_elem, "current_fruit_scale_factor", 1.0)
                        records.append({
                            "organ_type": ORGAN_BUD,
                            "shoot_id": sid,
                            "phytomer_idx": phyto_idx,
                            "parent_petiole_idx": pet_i,
                            "child_index": bidx,
                            "bud_state": bs,
                            "bud_parent_index": bpi,
                            "bud_is_terminal": biterm,
                            "fruit_scale": bcfs,
                        })

                        ped_elem = fb_elem.find("peduncle")
                        if ped_elem is not None:
                            pdl = _get_float_text(ped_elem, "length", 0.0)
                            pdr = _get_float_text(ped_elem, "radius", 0.0)
                            pdp = _get_float_text(ped_elem, "pitch", 0.0)
                            pdc = _get_float_text(ped_elem, "curvature", 0.0)
                            pdrl = _get_float_text(ped_elem, "roll", 0.0)
                            records.append({
                                "organ_type": ORGAN_PEDUNCLE,
                                "shoot_id": sid,
                                "phytomer_idx": phyto_idx,
                                "parent_petiole_idx": pet_i,
                                "length": pdl,
                                "radius": pdr,
                                "pitch": pdp,
                                "curvature": pdc,
                                "roll": pdrl,
                            })

                        infl_elem = fb_elem.find("inflorescence")
                        if infl_elem is not None:
                            foff = _get_float_text(infl_elem, "flower_offset", 0.05)
                            for fl_idx, fl_elem in enumerate(infl_elem.findall("flower")):
                                fp = _get_float_text(fl_elem, "flower_pitch", 0.0)
                                fy = _get_float_text(fl_elem, "flower_yaw", 0.0)
                                fr = _get_float_text(fl_elem, "flower_roll", 0.0)
                                fa = _get_float_text(fl_elem, "flower_azimuth", 0.0)
                                fbs = _get_float_text(fl_elem, "flower_base_scale", 1.0)
                                records.append({
                                    "organ_type": ORGAN_FLOWER,
                                    "shoot_id": sid,
                                    "phytomer_idx": phyto_idx,
                                    "parent_petiole_idx": pet_i,
                                    "child_index": fl_idx,
                                    "pitch": fp,
                                    "yaw": fy,
                                    "roll": fr,
                                    "flower_azimuth": fa,
                                    "scale": fbs,
                                    "flower_offset": foff,
                                })

    if not records:
        return torch.zeros((0, NUM_FEATURES_14D), dtype=torch.float32)

    N = len(records)
    part = torch.zeros((N, NUM_FEATURES_14D), dtype=torch.float32, device=device)
    eye_6d = rotation_matrix_to_6d(torch.eye(3, device=device))

    # ------------------------------------------------------------------
    # Build index maps over records (mirrors the old typed index maps).
    # ------------------------------------------------------------------
    shoot_meta_row: dict = {}
    internode_rows: dict = {}   # sid -> [(p_idx, rec_idx)]
    petiole_row: dict = {}      # (sid, p_idx, pet_i) -> rec_idx
    leaf_rows: dict = {}        # (sid, p_idx, pet_i) -> {lf_idx: rec_idx}
    bud_rows: dict = {}         # (sid, p_idx) -> [(bud_idx, rec_idx)]
    peduncle_rows: dict = {}    # (sid, p_idx) -> rec_idx
    flower_rows: dict = {}      # (sid, p_idx) -> [(fl_idx, rec_idx)]

    for idx, rec in enumerate(records):
        ot = rec["organ_type"]
        sid = rec.get("shoot_id", 0)
        p_idx = rec.get("phytomer_idx", 0)
        if ot == ORGAN_ROOT_META:
            part[idx, P14_COL_ORGAN_TYPE] = ORGAN_ROOT_META
            part[idx, P14_COL_EXISTENCE] = 1.0
            part[idx, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = rec["base"]
        elif ot == ORGAN_SHOOT_META:
            shoot_meta_row[sid] = idx
            part[idx, P14_COL_ORGAN_TYPE] = ORGAN_SHOOT_META
        elif ot == ORGAN_INTERNODE:
            internode_rows.setdefault(sid, []).append((p_idx, idx))
            part[idx, P14_COL_ORGAN_TYPE] = ORGAN_INTERNODE
            part[idx, P14_COL_SCALE_X] = rec["radius"]
            part[idx, P14_COL_SCALE_Y] = rec["radius"]
            part[idx, P14_COL_SCALE_Z] = rec["length"]
        elif ot == ORGAN_PETIOLE:
            pet_i = rec["parent_petiole_idx"]
            petiole_row[(sid, p_idx, pet_i)] = idx
            part[idx, P14_COL_ORGAN_TYPE] = ORGAN_PETIOLE
            part[idx, P14_COL_SCALE_X] = rec["radius"]
            part[idx, P14_COL_SCALE_Y] = rec["radius"]
            part[idx, P14_COL_SCALE_Z] = rec["length"]
        elif ot == ORGAN_LEAF:
            pet_i = rec["parent_petiole_idx"]
            lf_idx = rec["child_index"]
            leaf_rows.setdefault((sid, p_idx, pet_i), {})[lf_idx] = idx
            part[idx, P14_COL_ORGAN_TYPE] = ORGAN_LEAF
            part[idx, P14_COL_SCALE_X] = rec["scale"]
            part[idx, P14_COL_SCALE_Y] = rec["scale"]
            part[idx, P14_COL_SCALE_Z] = rec["scale"]
        elif ot == ORGAN_BUD:
            bud_idx = rec["child_index"]
            bud_rows.setdefault((sid, p_idx), []).append((bud_idx, idx))
            part[idx, P14_COL_ORGAN_TYPE] = ORGAN_BUD
        elif ot == ORGAN_PEDUNCLE:
            peduncle_rows[(sid, p_idx)] = idx
            part[idx, P14_COL_ORGAN_TYPE] = ORGAN_PEDUNCLE
            part[idx, P14_COL_SCALE_X] = rec["radius"]
            part[idx, P14_COL_SCALE_Y] = rec["radius"]
            part[idx, P14_COL_SCALE_Z] = rec["length"]
        elif ot == ORGAN_FLOWER:
            fl_idx = rec["child_index"]
            flower_rows.setdefault((sid, p_idx), []).append((fl_idx, idx))
            part[idx, P14_COL_ORGAN_TYPE] = ORGAN_FLOWER
            part[idx, P14_COL_SCALE_X] = rec["scale"]
            part[idx, P14_COL_SCALE_Y] = rec["scale"]
            part[idx, P14_COL_SCALE_Z] = rec["scale"]

    for sid in internode_rows:
        internode_rows[sid].sort(key=lambda x: x[0])
    for key in bud_rows:
        bud_rows[key].sort(key=lambda x: x[0])
    for key in flower_rows:
        flower_rows[key].sort(key=lambda x: x[0])

    sorted_shoot_ids = sorted(internode_rows.keys())
    node_output_info: dict = {}
    node_internode_tip_axes = torch.zeros((N, 3), dtype=torch.float32, device=device)

    def compute_shoot_base(sid: int, meta_rec: dict):
        parent_sid = meta_rec.get("parent_shoot_id", -1)
        parent_node_idx = meta_rec.get("parent_node_idx", 0)
        parent_petiole_index = meta_rec.get("parent_petiole_idx", 0)
        if parent_sid < 0 or (parent_sid, parent_node_idx) not in node_output_info:
            shoot_base_pos = torch.zeros(3, dtype=torch.float32, device=device)
            parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
            parent_petiole_axis = torch.tensor([0.0, -1.0, 0.0], device=device)
        else:
            p_info = node_output_info[(parent_sid, parent_node_idx)]
            parent_internode_axis = p_info["internode_axis"]
            pet_axes = p_info.get("petiole_axes", {})
            if parent_petiole_index in pet_axes:
                parent_petiole_axis = pet_axes[parent_petiole_index]
            else:
                parent_petiole_axis = p_info.get("petiole_axis", torch.tensor([0.0, -1.0, 0.0], device=device))
            shoot_base_pos = p_info["tip"]
        return shoot_base_pos, parent_internode_axis, parent_petiole_axis

    phytomer_context: dict = {}
    shoot_last_internode_tips: dict = {}
    phytomer_petiole_count: dict = {}

    # ==================================================================
    # Phase A: internodes, petioles, leaves
    # ==================================================================
    for sid in sorted_shoot_ids:
        node_indices = internode_rows[sid]
        meta_idx = shoot_meta_row.get(sid, node_indices[0][1] if len(node_indices) > 0 else 0)
        meta_rec = records[meta_idx]

        base_pitch_rad = meta_rec["pitch"] * deg2rad
        base_yaw_rad = meta_rec["yaw"] * deg2rad
        base_roll_rad = meta_rec["roll"] * deg2rad

        shoot_base_pos, parent_internode_axis, parent_petiole_axis = compute_shoot_base(sid, meta_rec)
        part[meta_idx, P14_COL_EXISTENCE] = 1.0
        part[meta_idx, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = shoot_base_pos
        R_shoot = (
            rotr_z(base_yaw_rad, device) @
            rotr_y(-base_pitch_rad, device) @
            rotr_x(base_roll_rad, device)
        )
        part[meta_idx, P14_COL_ROT_0:P14_COL_ROT_5 + 1] = rotation_matrix_to_6d(R_shoot)

        curr_pos = shoot_base_pos.clone()
        prev_internode_axis = parent_internode_axis
        prev_petiole_axis = parent_petiole_axis

        for p_idx_in_shoot, (p_idx, n_idx) in enumerate(node_indices):
            rec = records[n_idx]

            petiole_rot_axis = torch.linalg.cross(prev_internode_axis, prev_petiole_axis)
            if torch.linalg.norm(petiole_rot_axis) < 1e-6:
                petiole_rot_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
            else:
                petiole_rot_axis = petiole_rot_axis / torch.linalg.norm(petiole_rot_axis)

            inode_pitch_rad = rec["pitch"] * deg2rad
            inode_phyllo_rad = rec["phyllotactic_angle"] * deg2rad

            i_axis = prev_internode_axis.clone()
            if p_idx_in_shoot == 0:
                if inode_pitch_rad != 0.0:
                    i_axis = rotate_vector_about_axis(i_axis, petiole_rot_axis, 0.5 * inode_pitch_rad)
                if base_roll_rad != 0.0:
                    petiole_rot_axis = rotate_vector_about_axis(petiole_rot_axis, prev_internode_axis, base_roll_rad)
                    i_axis = rotate_vector_about_axis(i_axis, prev_internode_axis, base_roll_rad)
                if base_pitch_rad != 0.0:
                    base_pitch_axis = -1.0 * torch.linalg.cross(prev_internode_axis, prev_petiole_axis)
                    if torch.linalg.norm(base_pitch_axis) > 1e-6:
                        base_pitch_axis = base_pitch_axis / torch.linalg.norm(base_pitch_axis)
                        petiole_rot_axis = rotate_vector_about_axis(petiole_rot_axis, base_pitch_axis, -base_pitch_rad)
                        i_axis = rotate_vector_about_axis(i_axis, base_pitch_axis, -base_pitch_rad)
                if base_yaw_rad != 0.0:
                    petiole_rot_axis = rotate_vector_about_axis(petiole_rot_axis, prev_internode_axis, base_yaw_rad)
                    i_axis = rotate_vector_about_axis(i_axis, prev_internode_axis, base_yaw_rad)
            else:
                if inode_pitch_rad != 0.0:
                    i_axis = rotate_vector_about_axis(i_axis, petiole_rot_axis, -1.25 * inode_pitch_rad)

            i_axis = i_axis / (torch.linalg.norm(i_axis) + 1e-6)

            shoot_bending_axis = torch.linalg.cross(i_axis, z_axis)
            shoot_bending_norm = torch.linalg.norm(shoot_bending_axis)
            if shoot_bending_norm < 1e-6:
                shoot_bending_axis = torch.tensor([0.0, 1.0, 0.0], device=device)
            else:
                shoot_bending_axis = shoot_bending_axis / shoot_bending_norm

            inode_len = max(rec["length"], 1e-4)
            inode_rad = max(rec["radius"], 1e-4)
            seg_cnt = max(1, rec["length_segments"])
            seg_len = inode_len / seg_cnt
            seg_len_max = max(rec["length_max"], 1e-4) / seg_cnt

            curv_p0, curv_p1 = rec["curv_pert_0"], rec["curv_pert_1"]
            yaw_p0, yaw_p1 = rec["yaw_pert_0"], rec["yaw_pert_1"]

            inode_verts_list = [curr_pos.clone()]
            step_p = curr_pos.clone()
            step_dir = i_axis.clone()
            for s in range(seg_cnt):
                if p_idx_in_shoot > 0:
                    curv_pert = curv_p0 if s == 0 else curv_p1
                    yaw_pert = yaw_p0 if s == 0 else yaw_p1
                    curv_fact = 0.5 - step_dir[2] / 2.0
                    if step_dir[2] < 0:
                        curv_fact = curv_fact * 2.0
                    curvature_angle = deg2rad * (gravitropic_curvature * curv_fact * seg_len_max + curv_pert)
                    if curvature_angle != 0.0:
                        step_dir = rotate_vector_about_axis(step_dir, shoot_bending_axis, curvature_angle)
                    if yaw_pert != 0.0:
                        step_dir = rotate_vector_about_axis(step_dir, z_axis, deg2rad * yaw_pert)
                step_p = step_p + step_dir * seg_len
                inode_verts_list.append(step_p)

            inode_line = torch.stack(inode_verts_list)
            curr_pos = inode_line[-1]
            inode_tip_axis = step_dir / (torch.linalg.norm(step_dir) + 1e-6)
            node_internode_tip_axes[n_idx] = get_axis_vector_torch(inode_line, 1.0)

            part[n_idx, P14_COL_EXISTENCE] = 1.0
            part[n_idx, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = inode_line[0]
            part[n_idx, P14_COL_SCALE_X] = inode_rad
            part[n_idx, P14_COL_SCALE_Y] = inode_rad
            part[n_idx, P14_COL_SCALE_Z] = inode_len
            R_inode = _get_rotation_matrix_between_vectors_batch(
                torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0),
                inode_tip_axis.unsqueeze(0),
            ).squeeze(0)
            part[n_idx, P14_COL_ROT_0:P14_COL_ROT_5 + 1] = rotation_matrix_to_6d(R_inode)

            pet_axes_stored = {}
            pet_line_stored = {}
            node_info = {
                "tip": curr_pos,
                "internode_axis": inode_tip_axis,
                "radius": inode_rad,
            }

            petioles_here = [k for k in petiole_row if k[0] == sid and k[1] == p_idx]
            phytomer_petiole_count[(sid, p_idx)] = len(petioles_here)

            def process_petiole(pet_i, petiole_index):
                pet_row = petiole_row.get((sid, p_idx, pet_i))
                if pet_row is None:
                    return
                pet_rec = records[pet_row]
                p_len_raw = pet_rec["length"]
                p_rad_raw = pet_rec["radius"]
                p_pitch_deg = pet_rec["pitch"]
                p_curv_deg = pet_rec["curvature"]
                p_cls = pet_rec["current_leaf_scale_factor"]
                p_taper = pet_rec["taper"]
                p_seg_cnt = max(1, pet_rec["length_segments"])
                lflt_scale = pet_rec["leaflet_scale"]
                lflt_offset = pet_rec["leaflet_offset"]

                leaf_dict = leaf_rows.get((sid, p_idx, pet_i), {})
                leaf_list = sorted(leaf_dict.items(), key=lambda kv: kv[0])
                num_leaves = len(leaf_list)

                pet_pitch_rad = p_pitch_deg * deg2rad
                pet_axis = rotate_vector_about_axis(i_axis, petiole_rot_axis, abs(pet_pitch_rad))
                pet_rot_ax = petiole_rot_axis.clone()
                if p_idx_in_shoot != 0 and inode_phyllo_rad != 0.0:
                    pet_axis = rotate_vector_about_axis(pet_axis, i_axis, inode_phyllo_rad)
                    pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, inode_phyllo_rad)
                if petiole_index > 0:
                    petioles_per_internode = 2.0 if len(petioles_here) > 1 else 1.0
                    budrot = torch.tensor(petiole_index * 2.0 * math.pi / petioles_per_internode, device=device)
                    pet_axis = rotate_vector_about_axis(pet_axis, i_axis, budrot)
                    pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, budrot)
                pet_axis = pet_axis / (torch.linalg.norm(pet_axis) + 1e-12)
                pet_axes_stored[petiole_index] = pet_axis.clone()

                p_len = p_len_raw
                p_rad = p_rad_raw
                if p_len <= 0 or p_rad <= 0:
                    return

                pet_rot_ax_norm = pet_rot_ax / (torch.linalg.norm(pet_rot_ax) + 1e-8)
                pet_base = inode_line[-1]
                seq_len = p_len / p_seg_cnt

                curv_per_seg = p_curv_deg * seq_len * deg2rad
                if abs(curv_per_seg) > 1e-12:
                    s_indices = torch.arange(1, p_seg_cnt + 1, device=device, dtype=torch.float32)
                    angles = -s_indices * curv_per_seg
                    dirs = rotate_points_about_axis(pet_axis.unsqueeze(0).expand(p_seg_cnt, 3), pet_rot_ax_norm, angles)
                    offsets = torch.cumsum(dirs * seq_len, dim=0)
                    pet_line = torch.cat([pet_base.unsqueeze(0), pet_base.unsqueeze(0) + offsets], dim=0)
                else:
                    s_indices = torch.arange(1, p_seg_cnt + 1, device=device, dtype=torch.float32).unsqueeze(-1)
                    offsets = s_indices * (pet_axis * seq_len)
                    pet_line = torch.cat([pet_base.unsqueeze(0), pet_base.unsqueeze(0) + offsets], dim=0)

                pet_tip = pet_line[-1]
                pet_tip_axis = pet_line[-1] - pet_line[-2]
                pet_tip_axis = pet_tip_axis / (torch.linalg.norm(pet_tip_axis) + 1e-8)
                pet_line_stored[petiole_index] = pet_line.clone()

                part[pet_row, P14_COL_EXISTENCE] = 1.0
                part[pet_row, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = pet_base
                part[pet_row, P14_COL_SCALE_X] = p_rad
                part[pet_row, P14_COL_SCALE_Y] = p_rad
                part[pet_row, P14_COL_SCALE_Z] = p_len
                R_pet = _get_rotation_matrix_between_vectors_batch(
                    torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0),
                    pet_tip_axis.unsqueeze(0),
                ).squeeze(0)
                part[pet_row, P14_COL_ROT_0:P14_COL_ROT_5 + 1] = rotation_matrix_to_6d(R_pet)

                if num_leaves > 0:
                    for lf_i in range(min(num_leaves, 3)):
                        lf_idx, leaf_row_idx = leaf_list[lf_i]
                        lr = records[leaf_row_idx]
                        l_scale = lr["scale"]
                        l_pitch_raw = lr["pitch"] * deg2rad
                        l_yaw = lr["yaw"] * deg2rad
                        l_roll_raw = lr["roll"] * deg2rad

                        ind_from_tip = float(lf_i) - float(num_leaves - 1) / 2.0
                        compound_rotation = 0.0
                        if num_leaves > 1:
                            if lf_i == (num_leaves - 1) / 2.0:
                                compound_rotation = 0.0
                            elif lf_i < (num_leaves - 1) / 2.0:
                                compound_rotation = -0.5 * math.pi
                            else:
                                compound_rotation = 0.5 * math.pi

                        asin_pz = torch.asin(torch.clamp(pet_tip_axis[2], -1.0, 1.0))

                        if num_leaves == 1:
                            roll_rot = torch.acos(torch.clamp(inode_tip_axis[2], -1.0, 1.0)) - l_roll_raw
                        elif ind_from_tip != 0:
                            sign_roll = compound_rotation / abs(compound_rotation)
                            roll_rot = (asin_pz + l_roll_raw) * sign_roll
                        else:
                            roll_rot = 0.0

                        pitch_rot = l_pitch_raw
                        if ind_from_tip == 0:
                            pitch_rot = pitch_rot + asin_pz

                        yaw_rot = 0.0
                        if ind_from_tip != 0:
                            yaw_rot = l_yaw

                        azimuth_rot = -torch.atan2(pet_tip_axis[1], pet_tip_axis[0] + 1e-8) + compound_rotation

                        leaf_base = pet_tip
                        if num_leaves > 1 and lflt_offset > 0.0 and ind_from_tip != 0:
                            offset = (abs(ind_from_tip) - 0.5) * lflt_offset * p_len
                            frac = 1.0 - offset / max(p_len, 1e-6)
                            frac = max(0.0, min(1.0, frac))
                            if not (math.isnan(frac) or math.isinf(frac)):
                                leaf_base = interpolate_tube_torch(pet_line, frac)

                        R_leaf = (
                            rotr_z(azimuth_rot + yaw_rot, device) @
                            rotr_y(-pitch_rot, device) @
                            rotr_x(roll_rot, device)
                        )
                        part[leaf_row_idx, P14_COL_EXISTENCE] = 1.0
                        part[leaf_row_idx, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = leaf_base
                        part[leaf_row_idx, P14_COL_SCALE_X] = l_scale
                        part[leaf_row_idx, P14_COL_SCALE_Y] = l_scale
                        part[leaf_row_idx, P14_COL_SCALE_Z] = l_scale
                        part[leaf_row_idx, P14_COL_ROT_0:P14_COL_ROT_5 + 1] = rotation_matrix_to_6d(R_leaf)

            for pet_i in sorted(k[2] for k in petioles_here):
                process_petiole(pet_i, pet_i)

            node_info["petiole_axes"] = pet_axes_stored
            if 0 in pet_axes_stored:
                node_info["petiole_axis"] = pet_axes_stored[0].clone()
            node_output_info[(sid, p_idx)] = node_info

            phytomer_context[(sid, p_idx)] = {
                "inode_line": inode_line,
                "inode_tip_axis": inode_tip_axis,
                "tip_getaxis": node_internode_tip_axes[n_idx].clone(),
                "pet_lines": {k: v.clone() for k, v in pet_line_stored.items()},
                "p_idx_in_shoot": p_idx_in_shoot,
                "n_idx": n_idx,
            }

            prev_internode_axis = inode_tip_axis
            if 0 in pet_axes_stored:
                prev_petiole_axis = pet_axes_stored[0]
            else:
                ghost = torch.linalg.cross(inode_tip_axis, z_axis)
                if torch.linalg.norm(ghost) < 0.01:
                    ghost = torch.tensor([0.0, 1.0, 0.0], device=device)
                prev_petiole_axis = ghost / torch.linalg.norm(ghost)

        shoot_last_internode_tips[sid] = curr_pos.clone()

    # ==================================================================
    # Phase B: floral bud peduncle / flower / fruit
    # ==================================================================
    has_floral_geometry = False
    for _, bud_list in bud_rows.items():
        for _, bidx in bud_list:
            if records[bidx]["bud_state"] in (2, 3, 4):
                has_floral_geometry = True
                break
        if has_floral_geometry:
            break

    if has_floral_geometry:
        for (sid, p_idx), bud_list in sorted(bud_rows.items()):
            ctx = phytomer_context.get((sid, p_idx))
            if ctx is None:
                continue
            Nbuds = len(bud_list)
            petiole_count = max(1, phytomer_petiole_count.get((sid, p_idx), 1))
            petioles_per_internode = float(petiole_count)

            for bud_index, bud_row_idx in bud_list:
                bud_rec = records[bud_row_idx]
                state = bud_rec["bud_state"]
                if state not in (2, 3, 4):
                    continue
                pet_i = bud_rec["parent_petiole_idx"]
                is_terminal = bud_rec["bud_is_terminal"] > 0
                current_fruit_scale_factor = bud_rec["fruit_scale"]
                flower_offset = bud_rec.get("flower_offset", 0.05)

                if is_terminal:
                    bud_base = shoot_last_internode_tips.get(sid, ctx["inode_line"][-1])
                    base_pitch = (math.pi / 6.0) if Nbuds > 1 else 0.0
                    base_yaw = bud_index * 2.0 * math.pi / float(Nbuds)
                else:
                    pet_line0 = ctx["pet_lines"].get(pet_i)
                    bud_base = pet_line0[0] if pet_line0 is not None else ctx["inode_line"][-1]
                    base_pitch = bud_index * 0.1 * math.pi / float(Nbuds)
                    base_yaw = -0.25 * math.pi + bud_index * 0.5 * math.pi / float(Nbuds)
                part[bud_row_idx, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = bud_base

                ped_row_idx = peduncle_rows.get((sid, p_idx))
                if ped_row_idx is None:
                    continue
                ped_rec = records[ped_row_idx]
                p_len = ped_rec["length"]
                p_rad = ped_rec["radius"]
                p_pitch_rad = ped_rec["pitch"] * deg2rad
                p_curv_deg = ped_rec["curvature"]
                if p_len <= 0 or p_rad <= 0:
                    continue

                inode_line = ctx["inode_line"]
                peduncle_axis = ctx["tip_getaxis"].clone()

                if ctx["p_idx_in_shoot"] > 0:
                    prev_n_idx = None
                    for (pp_idx, nn_idx) in internode_rows[sid]:
                        if pp_idx == p_idx:
                            break
                        prev_n_idx = nn_idx
                    if prev_n_idx is not None:
                        parent_internode_axis = node_internode_tip_axes[prev_n_idx]
                    else:
                        parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
                else:
                    meta_idx = shoot_meta_row.get(sid, 0)
                    pmeta = records[meta_idx]
                    parent_sid = pmeta.get("parent_shoot_id", -1)
                    if parent_sid >= 0:
                        parent_node_xml = pmeta.get("parent_node_idx", 0)
                        parent_lin = _xml_parent_node_to_linear_idx(records, parent_sid, parent_node_xml)
                        parent_internode_axis = node_internode_tip_axes[parent_lin]
                    else:
                        parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], device=device)

                pet_line = ctx["pet_lines"].get(pet_i)
                if pet_line is not None:
                    current_petiole_axis = get_axis_vector_torch(pet_line, 0.0)
                    parent_petiole_base_axis = get_axis_vector_torch(pet_line, 0.0)
                else:
                    current_petiole_axis = parent_internode_axis
                    parent_petiole_base_axis = ctx["tip_getaxis"]

                infl_bending = torch.linalg.cross(parent_internode_axis, current_petiole_axis)
                if torch.linalg.norm(infl_bending) < 0.001:
                    infl_bending = torch.tensor([1.0, 0.0, 0.0], device=device)
                else:
                    infl_bending = infl_bending / torch.linalg.norm(infl_bending)

                if p_pitch_rad != 0 or base_pitch != 0:
                    peduncle_axis = rotate_vector_about_axis(peduncle_axis, infl_bending, p_pitch_rad + base_pitch)

                internode_axis = ctx["tip_getaxis"]
                parent_petiole_azimuth = -torch.atan2(parent_petiole_base_axis[1], parent_petiole_base_axis[0])
                current_peduncle_azimuth = -torch.atan2(peduncle_axis[1], peduncle_axis[0])
                azimuthal_rotation = current_peduncle_azimuth - parent_petiole_azimuth
                peduncle_axis = rotate_vector_about_axis(peduncle_axis, internode_axis, azimuthal_rotation)
                infl_bending = rotate_vector_about_axis(infl_bending, internode_axis, azimuthal_rotation)
                peduncle_axis = peduncle_axis / (torch.linalg.norm(peduncle_axis) + 1e-6)

                segs = max(1, ped_rec.get("length_segments", 6))
                dr = p_len / segs
                axis = peduncle_axis
                verts_list = [bud_base.clone()]
                for i in range(segs):
                    if abs(p_curv_deg) > 0:
                        hba = torch.linalg.cross(axis, z_axis)
                        m = torch.linalg.norm(hba)
                        if m > 0.001:
                            hba = hba / m
                            theta_curv = deg2rad * (p_curv_deg * dr)
                            zc = torch.clamp(axis[2], -1.0, 1.0)
                            theta_from_target = torch.acos(zc) if p_curv_deg > 0 else torch.acos(-zc)
                            if abs(theta_curv) >= theta_from_target:
                                axis = z_axis if p_curv_deg > 0 else -z_axis
                            else:
                                axis = rotate_vector_about_axis(axis, hba, theta_curv)
                                axis = axis / (torch.linalg.norm(axis) + 1e-6)
                        else:
                            axis = z_axis if p_curv_deg > 0 else -z_axis
                    verts_list.append(verts_list[-1] + dr * axis)

                ped_line = torch.stack(verts_list)

                part[ped_row_idx, P14_COL_EXISTENCE] = 1.0
                part[ped_row_idx, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = ped_line[0]
                part[ped_row_idx, P14_COL_SCALE_X] = p_rad
                part[ped_row_idx, P14_COL_SCALE_Y] = p_rad
                part[ped_row_idx, P14_COL_SCALE_Z] = p_len
                ped_axis_dir = (ped_line[-1] - ped_line[0])
                ped_axis_dir = ped_axis_dir / (torch.linalg.norm(ped_axis_dir) + 1e-8)
                R_ped = _get_rotation_matrix_between_vectors_batch(
                    torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0),
                    ped_axis_dir.unsqueeze(0),
                ).squeeze(0)
                part[ped_row_idx, P14_COL_ROT_0:P14_COL_ROT_5 + 1] = rotation_matrix_to_6d(R_ped)

                fl_list = flower_rows.get((sid, p_idx), [])
                n_flowers = len(fl_list)
                if n_flowers == 0:
                    continue

                for fl_idx, fl_row_idx in fl_list:
                    fl_rec = records[fl_row_idx]
                    saved_pitch = fl_rec["pitch"] * deg2rad
                    saved_yaw = fl_rec["yaw"] * deg2rad
                    saved_roll = fl_rec["roll"] * deg2rad
                    saved_azimuth = fl_rec["flower_azimuth"] * deg2rad
                    base_scale = fl_rec["scale"]

                    flower_offset_clamped = clamp_offset_torch(n_flowers, flower_offset)
                    ind_from_tip_computed = abs(float(fl_idx) - float(n_flowers - 1) / float(petioles_per_internode))
                    flower_base = ped_line[-1]
                    if n_flowers > 1 and flower_offset_clamped > 0 and ind_from_tip_computed != 0:
                        offset_computed = (ind_from_tip_computed - 0.5) * flower_offset_clamped * p_len
                        frac_computed = 1.0
                        if p_len > 0:
                            frac_computed = 1.0 - offset_computed / p_len
                        flower_base = interpolate_tube_torch(ped_line, frac_computed)

                    flower_offset_val = flower_offset
                    if n_flowers > 2:
                        denom = 0.5 * float(n_flowers) - 1.0
                        if flower_offset_val * denom > 1.0:
                            flower_offset_val = 1.0 / denom
                    ind_from_tip = abs(float(fl_idx) - float(n_flowers - 1) / float(petioles_per_internode))
                    frac = 1.0
                    if n_flowers > 1 and flower_offset_val > 0 and ind_from_tip != 0:
                        offset = (ind_from_tip - 0.5) * flower_offset_val * p_len
                        if p_len > 0:
                            frac = 1.0 - offset / p_len
                    recalculated_peduncle_axis = get_axis_vector_torch(ped_line, frac)

                    part[fl_row_idx, P14_COL_EXISTENCE] = 1.0
                    part[fl_row_idx, P14_COL_BASE_X:P14_COL_BASE_Z + 1] = flower_base
                    part[fl_row_idx, P14_COL_SCALE_X] = fl_rec["scale"]
                    part[fl_row_idx, P14_COL_SCALE_Y] = fl_rec["scale"]
                    part[fl_row_idx, P14_COL_SCALE_Z] = fl_rec["scale"]
                    if state == 4:
                        part[fl_row_idx, P14_COL_ORGAN_TYPE] = ORGAN_FRUIT
                    elif state == 2:
                        part[fl_row_idx, P14_COL_ORGAN_TYPE] = ORGAN_FLOWER_CLOSED
                    else:
                        part[fl_row_idx, P14_COL_ORGAN_TYPE] = ORGAN_FLOWER
                    R_yaw = rodrigues_matrix_torch(recalculated_peduncle_axis, saved_yaw, device=device)
                    R_obj_net = (
                        R_yaw @
                        rotr_z(saved_azimuth, device) @
                        rotr_y(saved_pitch, device) @
                        rotr_x(saved_roll, device)
                    )
                    part[fl_row_idx, P14_COL_ROT_0:P14_COL_ROT_5 + 1] = rotation_matrix_to_6d(R_obj_net)

    return part


def _xml_parent_node_to_linear_idx(records: list, parent_sid: int, parent_node_xml: int) -> int:
    """Map a (shoot_id, xml node index) to the linear record index of that internode."""
    for idx, rec in enumerate(records):
        if rec.get("organ_type") == ORGAN_INTERNODE and rec.get("shoot_id") == parent_sid and rec.get("phytomer_idx") == parent_node_xml:
            return idx
    return 0


# =============================================================================
# PLANT ORGAN ARRAY CLASS
# =============================================================================

class PlantOrganArray:
    """
    Stores plant architecture as a part-centric tensor.

    The tensor must have shape (N, NUM_FEATURES_14D). No legacy or typed layout
    variants are supported.
    """

    def __init__(self, tensor: torch.Tensor):
        if tensor.ndim != 2:
            raise ValueError(
                f"PlantOrganArray tensor must be 2D, got shape {tensor.shape}"
            )
        if tensor.shape[1] != NUM_FEATURES_14D:
            raise ValueError(
                f"PlantOrganArray tensor must have {NUM_FEATURES_14D} columns, "
                f"got {tensor.shape[1]}"
            )
        self.tensor = tensor

    @property
    def num_nodes(self) -> int:
        return self.tensor.shape[0]

    @property
    def existence(self) -> torch.Tensor:
        return self.tensor[:, P14_COL_EXISTENCE]

    @existence.setter
    def existence(self, value: torch.Tensor) -> None:
        if value.shape[0] != self.tensor.shape[0]:
            raise ValueError("existence length must match number of nodes")
        self.tensor[:, P14_COL_EXISTENCE] = value

    def to_part_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns the stored part tensor, optionally moved to ``device``."""
        return self.tensor.to(device)

    @classmethod
    def from_part_tensor(cls, part_tensor: torch.Tensor) -> "PlantOrganArray":
        """Wraps a raw part tensor in a PlantOrganArray."""
        return cls(part_tensor)

    def to_xml_string(self, existence_threshold: float = 0.5) -> str:
        """Serializes the part tensor to a Helios XML string."""
        from diffusion_based.models.part_assembly_to_xml import PartAssemblyToXMLConverter

        converter = PartAssemblyToXMLConverter()
        return converter.convert_to_xml_string(
            self.tensor,
            plant_id=0,
            plant_type="cowpea",
            existence_threshold=existence_threshold,
        )

    def write_xml(self, filepath: str) -> None:
        """Writes ``to_xml_string()`` output to ``filepath``."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_xml_string())

    @classmethod
    def from_xml_string(cls, xml_content: str) -> "PlantOrganArray":
        """Parses a Helios XML string directly into a part-centric PlantOrganArray."""
        return cls(_parse_xml_to_part_tensor(xml_content))

    @classmethod
    def from_xml_string_typed(cls, xml_content: str) -> "PlantOrganArray":
        """Alias for :meth:`from_xml_string` for backward-compatible naming."""
        return cls.from_xml_string(xml_content)

    @classmethod
    def from_xml_file(cls, filepath: str) -> "PlantOrganArray":
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_xml_string(content)

    @classmethod
    def from_xml_file_typed(cls, filepath: str) -> "PlantOrganArray":
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_xml_string(content)


# =============================================================================
# SMALL XML PARSING HELPERS
# =============================================================================

def _fmt(val: float) -> str:
    """Formats float for exact XML strings."""
    if isinstance(val, str):
        return val
    return f"{val:g}"


def _to_int(x) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def _get_text_default(elem: Optional[ET.Element], tag: str, default: Optional[str]) -> Optional[str]:
    child = elem.find(tag) if elem is not None else None
    if child is not None and child.text:
        return child.text
    return default


def _get_float_text(elem: Optional[ET.Element], tag: str, default: float) -> float:
    text = _get_text_default(elem, tag, "")
    if text:
        try:
            return float(text.strip())
        except ValueError:
            return default
    return default


def _get_int_text(elem: Optional[ET.Element], tag: str, default: int) -> int:
    text = _get_text_default(elem, tag, "")
    if text:
        try:
            return int(text.strip())
        except ValueError:
            return default
    return default
