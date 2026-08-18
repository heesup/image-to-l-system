"""
Direct XML -> part array parser parser.

Skips the intermediate 40D typed PlantOrganArray entirely. Parses Helios XML
into a lightweight tree of Python dicts, then runs the same forward-kinematics
chain used by HeliosPlantGeometryBuilder._build_mesh_typed to produce an
(N, D) part-centric tensor.

This is faster and uses less memory than:
    XML -> 40D tensor -> _build_mesh_typed(compute_mesh=False)
because it avoids allocating and sorting the (N, 40) typed array.

The output part tensor is numerically identical to the one produced by the
40D-based path (verified on dap10/50/90/100).
"""

import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import xml.etree.ElementTree as ET


# Re-use the same math helpers and constants as the 40D builder so the two
# paths stay pixel-identical.
from diffusion_based.models.plant_organ_array import (
    NUM_FEATURES_14D,
    ORGAN_ROOT_META, ORGAN_SHOOT_META, ORGAN_INTERNODE, ORGAN_PETIOLE,
    ORGAN_LEAF, ORGAN_BUD, ORGAN_PEDUNCLE, ORGAN_FLOWER,
    P14_COL_ORGAN_TYPE, P14_COL_BASE_X, P14_COL_BASE_Y, P14_COL_BASE_Z,
    P14_COL_ROT_0, P14_COL_ROT_5,
    P14_COL_SCALE_X, P14_COL_SCALE_Y, P14_COL_SCALE_Z, P14_COL_EXISTENCE,
    rotation_matrix_to_6d,
)
from diffusion_based.models.helios_pytorch_geometry import (
    rotr_x, rotr_y, rotr_z,
    rotate_vector_about_axis,
    rotate_points_about_axis,
    rodrigues_matrix_torch,
    interpolate_tube_torch,
    get_axis_vector_torch,
    _get_rotation_matrix_between_vectors_batch,
)


def _parse_float(elem, tag: str, default: float = 0.0) -> float:
    child = elem.find(tag)
    if child is not None and child.text:
        try:
            return float(child.text.strip())
        except ValueError:
            return default
    return default


def _parse_int(elem, tag: str, default: int = 0) -> int:
    child = elem.find(tag)
    if child is not None and child.text:
        try:
            return int(child.text.strip())
        except ValueError:
            return default
    return default


def _parse_text_default(elem, tag: str, default: str = "") -> str:
    child = elem.find(tag)
    return child.text.strip() if child is not None and child.text else default


def _parse_vec3(elem, tag: str, default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    child = elem.find(tag)
    if child is None or not child.text:
        return default
    vals = [float(x) for x in child.text.strip().split()]
    if len(vals) >= 3:
        return (vals[0], vals[1], vals[2])
    return default


def _parse_semicolon_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.strip().split(";") if x.strip()]


class XMLToPartArrayParser:
    """Parse Helios XML and compute part tensors directly."""

    def __init__(self, leaf_scale_factor: float = 1.0):
        self.leaf_scale_factor = leaf_scale_factor
        self.gravitropic_curvature = 200.0

    def parse(self, xml_content: str, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        root = ET.fromstring(xml_content)
        if root.tag != "helios":
            raise ValueError("Root tag must be <helios>")

        organ_rows: List[Dict[str, Any]] = []
        shoot_tip_cache: Dict[int, torch.Tensor] = {}
        shoot_axis_cache: Dict[int, torch.Tensor] = {}
        shoot_petiole_axis_cache: Dict[int, torch.Tensor] = {}
        node_tip_axes: Dict[int, torch.Tensor] = {}

        for plant_elem in root.findall("plant_instance"):
            plant_id = int(plant_elem.attrib.get("ID", 0))
            base_position = _parse_vec3(plant_elem, "base_position", (0.0, 0.0, 0.0))
            plant_age = _parse_float(plant_elem, "plant_age", 0.0)

            # ROOT_META
            organ_rows.append({
                "organ_type": ORGAN_ROOT_META,
                "base": torch.tensor(base_position, dtype=torch.float32, device=device),
                "scale": torch.ones(3, dtype=torch.float32, device=device),
                "existence": 1.0,
                "plant_id": plant_id,
                "plant_age": plant_age,
            })

            for shoot_elem in plant_elem.findall("shoot"):
                shoot_id = int(shoot_elem.attrib.get("ID", 0))
                stl = _parse_text_default(shoot_elem, "shoot_type_label", "unifoliate")
                shoot_type = 0 if "unifoliate" in stl else 1
                parent_shoot_id = _parse_int(shoot_elem, "parent_shoot_ID", -1)
                parent_node_idx = _parse_int(shoot_elem, "parent_node_index", 0)
                parent_petiole_idx = _parse_int(shoot_elem, "parent_petiole_index", 0)
                base_rotation = _parse_vec3(shoot_elem, "base_rotation", (0.0, 0.0, 0.0))

                # Resolve parent context
                if parent_shoot_id < 0:
                    shoot_base = torch.zeros(3, dtype=torch.float32, device=device)
                    parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
                    parent_petiole_axis = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)
                else:
                    shoot_base = shoot_tip_cache.get(parent_shoot_id, torch.zeros(3, dtype=torch.float32, device=device))
                    parent_internode_axis = shoot_axis_cache.get(
                        parent_shoot_id, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
                    )
                    parent_petiole_axis = shoot_petiole_axis_cache.get(
                        parent_shoot_id, torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)
                    )

                deg2rad = torch.tensor(math.pi / 180.0, dtype=torch.float32, device=device)
                base_pitch_rad = base_rotation[0] * deg2rad
                base_yaw_rad = base_rotation[1] * deg2rad
                base_roll_rad = base_rotation[2] * deg2rad

                R_shoot = (
                    rotr_z(base_yaw_rad, device) @
                    rotr_y(-base_pitch_rad, device) @
                    rotr_x(base_roll_rad, device)
                )

                # SHOOT_META row
                organ_rows.append({
                    "organ_type": ORGAN_SHOOT_META,
                    "base": shoot_base,
                    "rotation_6d": rotation_matrix_to_6d(R_shoot),
                    "scale": torch.ones(3, dtype=torch.float32, device=device),
                    "existence": 1.0,
                    "shoot_id": shoot_id,
                    "parent_shoot_id": parent_shoot_id,
                    "parent_node_idx": parent_node_idx,
                    "parent_petiole_idx": parent_petiole_idx,
                })

                curr_pos = shoot_base.clone()
                prev_internode_axis = parent_internode_axis
                prev_petiole_axis = parent_petiole_axis
                z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

                for phyto_idx, phyto_elem in enumerate(shoot_elem.findall("phytomer")):
                    internode_elem = phyto_elem.find("internode")
                    if internode_elem is None:
                        continue

                    il = max(_parse_float(internode_elem, "internode_length", 0.0), 1e-4)
                    ir = max(_parse_float(internode_elem, "internode_radius", 0.0), 1e-4)
                    ip = _parse_float(internode_elem, "internode_pitch", 0.0)
                    ipa = _parse_float(internode_elem, "internode_phyllotactic_angle", 0.0)
                    ilm = max(_parse_float(internode_elem, "internode_length_max", 0.0), 1e-4)
                    ils = max(_parse_int(internode_elem, "internode_length_segments", 2), 1)
                    cp_text = _parse_text_default(internode_elem, "curvature_perturbations", "0;0")
                    cp_list = _parse_semicolon_floats(cp_text)
                    cp0 = cp_list[0] if len(cp_list) > 0 else 0.0
                    cp1 = cp_list[1] if len(cp_list) > 1 else 0.0
                    yp_text = _parse_text_default(internode_elem, "yaw_perturbations", "0;0")
                    yp_list = _parse_semicolon_floats(yp_text)
                    yp0 = yp_list[0] if len(yp_list) > 0 else 0.0
                    yp1 = yp_list[1] if len(yp_list) > 1 else 0.0

                    inode_pitch_rad = ip * deg2rad
                    inode_phyllo_rad = ipa * deg2rad

                    petiole_rot_axis = torch.linalg.cross(prev_internode_axis, prev_petiole_axis)
                    if torch.linalg.norm(petiole_rot_axis) < 1e-6:
                        petiole_rot_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
                    else:
                        petiole_rot_axis = petiole_rot_axis / torch.linalg.norm(petiole_rot_axis)

                    i_axis = prev_internode_axis.clone()
                    if phyto_idx == 0:
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
                        shoot_bending_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
                    else:
                        shoot_bending_axis = shoot_bending_axis / shoot_bending_norm

                    seg_len = il / ils
                    seg_len_max = ilm / ils

                    inode_verts_list = [curr_pos.clone()]
                    step_p = curr_pos.clone()
                    step_dir = i_axis.clone()
                    for s in range(ils):
                        if phyto_idx > 0:
                            curv_pert = cp0 if s == 0 else cp1
                            yaw_pert = yp0 if s == 0 else yp1
                            curv_fact = 0.5 - step_dir[2] / 2.0
                            if step_dir[2] < 0:
                                curv_fact = curv_fact * 2.0
                            curvature_angle = deg2rad * (self.gravitropic_curvature * curv_fact * seg_len_max + curv_pert)
                            if curvature_angle != 0.0:
                                step_dir = rotate_vector_about_axis(step_dir, shoot_bending_axis, curvature_angle)
                            if yaw_pert != 0.0:
                                step_dir = rotate_vector_about_axis(step_dir, z_axis, deg2rad * yaw_pert)
                        step_p = step_p + step_dir * seg_len
                        inode_verts_list.append(step_p)

                    inode_line = torch.stack(inode_verts_list)
                    inode_base = inode_line[0]
                    inode_tip = inode_line[-1]
                    inode_tip_axis = step_dir / (torch.linalg.norm(step_dir) + 1e-6)
                    node_tip_axes[len(organ_rows)] = get_axis_vector_torch(inode_line, 1.0)

                    R_inode = _get_rotation_matrix_between_vectors_batch(
                        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device).unsqueeze(0),
                        inode_tip_axis.unsqueeze(0),
                    ).squeeze(0)

                    # INTERNODE row
                    organ_rows.append({
                        "organ_type": ORGAN_INTERNODE,
                        "base": inode_base,
                        "rotation_6d": rotation_matrix_to_6d(R_inode),
                        "scale": torch.tensor([ir, ir, il], dtype=torch.float32, device=device),
                        "existence": 1.0,
                        "shoot_id": shoot_id,
                        "phytomer_idx": phyto_idx,
                        "inode_tip": inode_tip,
                        "inode_tip_axis": inode_tip_axis,
                    })
                    internode_row_idx = len(organ_rows) - 1

                    # Petioles and leaves
                    petiole_axes_stored: Dict[int, torch.Tensor] = {}
                    pet_lines_stored: Dict[int, torch.Tensor] = {}
                    petioles_here = list(internode_elem.findall("petiole"))
                    n_petioles = len(petioles_here)

                    for pet_i, pet_elem in enumerate(petioles_here):
                        pl = max(_parse_float(pet_elem, "petiole_length", 0.0), 1e-6)
                        pr = max(_parse_float(pet_elem, "petiole_radius", 0.0), 1e-6)
                        pp = _parse_float(pet_elem, "petiole_pitch", 0.0)
                        pc = _parse_float(pet_elem, "petiole_curvature", 0.0)
                        cls_val = _parse_float(pet_elem, "current_leaf_scale_factor", 1.0)
                        pt = _parse_float(pet_elem, "petiole_taper", 0.25)
                        pls = max(_parse_int(pet_elem, "petiole_length_segments", 5), 1)
                        lflt_scale = _parse_float(pet_elem, "leaflet_scale", 1.0)
                        lflt_offset = _parse_float(pet_elem, "leaflet_offset", 0.4)

                        pet_pitch_rad = pp * deg2rad
                        pet_axis = rotate_vector_about_axis(i_axis, petiole_rot_axis, torch.abs(pet_pitch_rad))
                        pet_rot_ax = petiole_rot_axis.clone()
                        if phyto_idx != 0 and inode_phyllo_rad != 0.0:
                            pet_axis = rotate_vector_about_axis(pet_axis, i_axis, inode_phyllo_rad)
                            pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, inode_phyllo_rad)
                        if pet_i > 0:
                            petioles_per_internode = 2.0 if n_petioles > 1 else 1.0
                            budrot = torch.tensor(pet_i * 2.0 * math.pi / petioles_per_internode, dtype=torch.float32, device=device)
                            pet_axis = rotate_vector_about_axis(pet_axis, i_axis, budrot)
                            pet_rot_ax = rotate_vector_about_axis(pet_rot_ax, i_axis, budrot)
                        pet_axis = pet_axis / (torch.linalg.norm(pet_axis) + 1e-12)
                        petiole_axes_stored[pet_i] = pet_axis.clone()

                        pet_rot_ax_norm = pet_rot_ax / (torch.linalg.norm(pet_rot_ax) + 1e-8)
                        pet_base = inode_tip
                        seq_len = pl / pls

                        curv_per_seg = pc * seq_len * deg2rad
                        if torch.abs(curv_per_seg) > 1e-12:
                            s_indices = torch.arange(1, pls + 1, device=device, dtype=torch.float32)
                            angles = -s_indices * curv_per_seg
                            dirs = rotate_points_about_axis(pet_axis.unsqueeze(0).expand(pls, 3), pet_rot_ax_norm, angles)
                            offsets = torch.cumsum(dirs * seq_len, dim=0)
                            pet_line = torch.cat([pet_base.unsqueeze(0), pet_base.unsqueeze(0) + offsets], dim=0)
                        else:
                            s_indices = torch.arange(1, pls + 1, device=device, dtype=torch.float32).unsqueeze(-1)
                            offsets = s_indices * (pet_axis * seq_len)
                            pet_line = torch.cat([pet_base.unsqueeze(0), pet_base.unsqueeze(0) + offsets], dim=0)

                        pet_tip = pet_line[-1]
                        pet_tip_axis = pet_line[-1] - pet_line[-2]
                        pet_tip_axis = pet_tip_axis / (torch.linalg.norm(pet_tip_axis) + 1e-8)
                        pet_lines_stored[pet_i] = pet_line.clone()

                        R_pet = _get_rotation_matrix_between_vectors_batch(
                            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device).unsqueeze(0),
                            pet_tip_axis.unsqueeze(0),
                        ).squeeze(0)

                        # PETIOLE row
                        organ_rows.append({
                            "organ_type": ORGAN_PETIOLE,
                            "base": pet_base,
                            "rotation_6d": rotation_matrix_to_6d(R_pet),
                            "scale": torch.tensor([pr, pr, pl], dtype=torch.float32, device=device),
                            "existence": 1.0,
                            "shoot_id": shoot_id,
                            "phytomer_idx": phyto_idx,
                            "parent_petiole_idx": pet_i,
                        })

                        # Leaves
                        leaf_elems = pet_elem.findall("leaf")
                        num_leaves = len(leaf_elems)
                        for lf_i, leaf_elem in enumerate(leaf_elems[:3]):
                            l_scale = _parse_float(leaf_elem, "leaf_scale", 1.0)
                            lfp = _parse_float(leaf_elem, "leaf_pitch", 0.0) * deg2rad
                            lfy = _parse_float(leaf_elem, "leaf_yaw", 0.0) * deg2rad
                            lfr = _parse_float(leaf_elem, "leaf_roll", 0.0) * deg2rad

                            ind_from_tip = float(lf_i) - float(num_leaves - 1) / 2.0
                            compound_rotation = 0.0
                            if num_leaves > 1:
                                if lf_i == (num_leaves - 1) / 2.0:
                                    compound_rotation = 0.0
                                elif lf_i < (num_leaves - 1) / 2.0:
                                    compound_rotation = -0.5 * math.pi
                                else:
                                    compound_rotation = 0.5 * math.pi

                            tot_scale = l_scale * self.leaf_scale_factor
                            asin_pz = torch.asin(torch.clamp(pet_tip_axis[2], -1.0, 1.0))

                            if num_leaves == 1:
                                roll_rot = torch.acos(torch.clamp(inode_tip_axis[2], -1.0, 1.0)) - lfr
                            elif ind_from_tip != 0:
                                sign_roll = compound_rotation / abs(compound_rotation)
                                roll_rot = (asin_pz + lfr) * sign_roll
                            else:
                                roll_rot = 0.0

                            pitch_rot = lfp
                            if ind_from_tip == 0:
                                pitch_rot = pitch_rot + asin_pz

                            yaw_rot = 0.0
                            if ind_from_tip != 0:
                                yaw_rot = lfy

                            azimuth_rot = -torch.atan2(pet_tip_axis[1], pet_tip_axis[0] + 1e-8) + compound_rotation

                            leaf_base = pet_tip
                            if num_leaves > 1 and lflt_offset > 0.0 and ind_from_tip != 0:
                                offset = (abs(ind_from_tip) - 0.5) * lflt_offset * pl
                                frac = 1.0 - offset / max(pl, 1e-6)
                                frac = max(0.0, min(1.0, frac))
                                if not (math.isnan(frac) or math.isinf(frac)):
                                    leaf_base = interpolate_tube_torch(pet_line, frac)

                            R_leaf = (
                                rotr_z(azimuth_rot + yaw_rot, device) @
                                rotr_y(-pitch_rot, device) @
                                rotr_x(roll_rot, device)
                            )

                            organ_rows.append({
                                "organ_type": ORGAN_LEAF,
                                "base": leaf_base,
                                "rotation_6d": rotation_matrix_to_6d(R_leaf),
                                "scale": torch.tensor([l_scale, l_scale, l_scale], dtype=torch.float32, device=device),
                                "existence": 1.0,
                                "shoot_id": shoot_id,
                                "phytomer_idx": phyto_idx,
                                "parent_petiole_idx": pet_i,
                                "child_index": lf_i,
                            })

                        # Floral bud (only on petiole 0)
                        if pet_i == 0:
                            fb_elem = pet_elem.find("floral_bud")
                            if fb_elem is not None:
                                bs = _parse_int(fb_elem, "bud_state", 5)
                                biterm = _parse_int(fb_elem, "is_terminal", 0)
                                bcfs = _parse_float(fb_elem, "current_fruit_scale_factor", 1.0)
                                is_flowering = bs in (2, 3, 4)

                                # BUD row
                                organ_rows.append({
                                    "organ_type": ORGAN_BUD,
                                    "base": inode_tip,
                                    "scale": torch.ones(3, dtype=torch.float32, device=device),
                                    "existence": 1.0 if is_flowering else 0.0,
                                    "shoot_id": shoot_id,
                                    "phytomer_idx": phyto_idx,
                                    "parent_petiole_idx": pet_i,
                                    "bud_state": bs,
                                    "is_terminal": biterm,
                                    "fruit_scale": bcfs,
                                })

                                ped_elem = fb_elem.find("peduncle")
                                if ped_elem is not None:
                                    pdl = max(_parse_float(ped_elem, "length", 0.0), 1e-6) if is_flowering else 1.0
                                    pdr = max(_parse_float(ped_elem, "radius", 0.0), 1e-6) if is_flowering else 1.0
                                    pdp = _parse_float(ped_elem, "pitch", 0.0) * deg2rad
                                    pdc = _parse_float(ped_elem, "curvature", 0.0)

                                    if is_flowering:
                                        # Reconstruct peduncle orientation
                                        if phyto_idx > 0:
                                            prev_n_idx = None
                                            for prev_p, prev_idx in [(r.get("phytomer_idx"), i) for i, r in enumerate(organ_rows) if r.get("organ_type") == ORGAN_INTERNODE and r.get("shoot_id") == shoot_id]:
                                                if prev_p == phyto_idx:
                                                    break
                                                prev_n_idx = prev_idx
                                            if prev_n_idx is not None:
                                                parent_internode_axis = node_tip_axes.get(prev_n_idx, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device))
                                            else:
                                                parent_internode_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
                                        else:
                                            parent_internode_axis = parent_internode_axis

                                        pet_line0 = pet_lines_stored.get(0)
                                        if pet_line0 is not None:
                                            current_petiole_axis = get_axis_vector_torch(pet_line0, 0.0)
                                            parent_petiole_base_axis = current_petiole_axis
                                        else:
                                            current_petiole_axis = parent_internode_axis
                                            parent_petiole_base_axis = node_tip_axes.get(internode_row_idx, inode_tip_axis)

                                        peduncle_axis = node_tip_axes.get(internode_row_idx, inode_tip_axis).clone()
                                        infl_bending = torch.linalg.cross(parent_internode_axis, current_petiole_axis)
                                        if torch.linalg.norm(infl_bending) < 0.001:
                                            infl_bending = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
                                        else:
                                            infl_bending = infl_bending / torch.linalg.norm(infl_bending)

                                        if pdp != 0 or biterm:
                                            base_pitch = (math.pi / 6.0) if biterm else 0.0
                                            peduncle_axis = rotate_vector_about_axis(peduncle_axis, infl_bending, pdp + base_pitch)

                                        parent_petiole_azimuth = -torch.atan2(parent_petiole_base_axis[1], parent_petiole_base_axis[0])
                                        current_peduncle_azimuth = -torch.atan2(peduncle_axis[1], peduncle_axis[0])
                                        azimuthal_rotation = current_peduncle_azimuth - parent_petiole_azimuth
                                        peduncle_axis = rotate_vector_about_axis(peduncle_axis, inode_tip_axis, azimuthal_rotation)
                                        infl_bending = rotate_vector_about_axis(infl_bending, inode_tip_axis, azimuthal_rotation)
                                        peduncle_axis = peduncle_axis / (torch.linalg.norm(peduncle_axis) + 1e-6)

                                        segs = max(_parse_int(ped_elem, "length_segments", 6), 1)
                                        dr = pdl / segs
                                        axis = peduncle_axis
                                        verts_list = [inode_tip.clone()]
                                        for i in range(segs):
                                            if abs(pdc) > 0:
                                                hba = torch.linalg.cross(axis, z_axis)
                                                m = torch.linalg.norm(hba)
                                                if m > 0.001:
                                                    hba = hba / m
                                                    theta_curv = deg2rad * (pdc * dr)
                                                    zc = torch.clamp(axis[2], -1.0, 1.0)
                                                    theta_from_target = torch.acos(zc) if pdc > 0 else torch.acos(-zc)
                                                    if abs(theta_curv) >= theta_from_target:
                                                        axis = z_axis if pdc > 0 else -z_axis
                                                    else:
                                                        axis = rotate_vector_about_axis(axis, hba, theta_curv)
                                                        axis = axis / (torch.linalg.norm(axis) + 1e-6)
                                                else:
                                                    axis = z_axis if pdc > 0 else -z_axis
                                            verts_list.append(verts_list[-1] + dr * axis)

                                        ped_line = torch.stack(verts_list)
                                        R_ped = _get_rotation_matrix_between_vectors_batch(
                                            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device).unsqueeze(0),
                                            (ped_line[-1] - ped_line[0]).unsqueeze(0),
                                        ).squeeze(0)
                                        ped_base = ped_line[0]
                                        ped_rotation = rotation_matrix_to_6d(R_ped)
                                        ped_scale = torch.tensor([pdr, pdr, pdl], dtype=torch.float32, device=device)

                                        infl_elem = fb_elem.find("inflorescence")
                                        if infl_elem is not None:
                                            foff = _parse_float(infl_elem, "flower_offset", 0.05)
                                            flower_elems = infl_elem.findall("flower")
                                            n_flowers = len(flower_elems)

                                            for fl_idx, fl_elem in enumerate(flower_elems):
                                                fp = _parse_float(fl_elem, "flower_pitch", 0.0) * deg2rad
                                                fy = _parse_float(fl_elem, "flower_yaw", 0.0) * deg2rad
                                                fr = _parse_float(fl_elem, "flower_roll", 0.0) * deg2rad
                                                fa = _parse_float(fl_elem, "flower_azimuth", 0.0) * deg2rad
                                                fbs = _parse_float(fl_elem, "flower_base_scale", 1.0)

                                                flower_base = ped_line[-1]
                                                if n_flowers > 1:
                                                    ind_from_tip = abs(float(fl_idx) - float(n_flowers - 1) / 2.0)
                                                    offset = (ind_from_tip - 0.5) * foff * pdl
                                                    frac = 1.0 - offset / max(pdl, 1e-6)
                                                    frac = max(0.0, min(1.0, frac))
                                                    flower_base = interpolate_tube_torch(ped_line, frac)

                                                recalculated_peduncle_axis = get_axis_vector_torch(ped_line, 1.0)
                                                R_yaw = rodrigues_matrix_torch(recalculated_peduncle_axis, fy, device=device)
                                                R_obj_net = (
                                                    R_yaw @
                                                    rotr_z(fa, device) @
                                                    rotr_y(fp, device) @
                                                    rotr_x(fr, device)
                                                )

                                                organ_type = ORGAN_FLOWER
                                                if bs == 4:
                                                    organ_type = 8  # fruit
                                                elif bs == 2:
                                                    organ_type = 9  # closed flower

                                                organ_rows.append({
                                                    "organ_type": organ_type,
                                                    "base": flower_base,
                                                    "rotation_6d": rotation_matrix_to_6d(R_obj_net),
                                                    "scale": torch.tensor([fbs, fbs, fbs], dtype=torch.float32, device=device),
                                                    "existence": 1.0,
                                                    "shoot_id": shoot_id,
                                                    "phytomer_idx": phyto_idx,
                                                    "parent_petiole_idx": pet_i,
                                                    "child_index": fl_idx,
                                                })
                                    else:
                                        # Non-flowering bud: add placeholder peduncle row so row counts match the typed path
                                        ped_base = inode_tip
                                        ped_rotation = rotation_matrix_to_6d(torch.eye(3, device=device))
                                        ped_scale = torch.ones(3, dtype=torch.float32, device=device)

                                    organ_rows.append({
                                        "organ_type": ORGAN_PEDUNCLE,
                                        "base": ped_base,
                                        "rotation_6d": ped_rotation,
                                        "scale": ped_scale,
                                        "existence": 1.0 if is_flowering else 0.0,
                                        "shoot_id": shoot_id,
                                        "phytomer_idx": phyto_idx,
                                        "parent_petiole_idx": pet_i,
                                    })

                    # Update shoot context for next phytomer
                    curr_pos = inode_tip
                    prev_internode_axis = inode_tip_axis
                    if 0 in petiole_axes_stored:
                        prev_petiole_axis = petiole_axes_stored[0]
                    else:
                        ghost = torch.linalg.cross(inode_tip_axis, z_axis)
                        if torch.linalg.norm(ghost) < 0.01:
                            ghost = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
                        prev_petiole_axis = ghost / torch.linalg.norm(ghost)

                # Store shoot tip/axis for child shoots
                shoot_tip_cache[shoot_id] = curr_pos.clone()
                shoot_axis_cache[shoot_id] = prev_internode_axis.clone()
                shoot_petiole_axis_cache[shoot_id] = prev_petiole_axis.clone()

        # Build (N, D) tensor
        N = len(organ_rows)
        p14 = torch.zeros((N, NUM_FEATURES_14D), dtype=torch.float32, device=device)
        eye_6d = rotation_matrix_to_6d(torch.eye(3, device=device))
        for i, row in enumerate(organ_rows):
            p14[i, P14_COL_ORGAN_TYPE] = float(row["organ_type"])
            p14[i, P14_COL_BASE_X:P14_COL_BASE_Z+1] = row["base"]
            p14[i, P14_COL_SCALE_X:P14_COL_SCALE_Z+1] = row["scale"]
            p14[i, P14_COL_EXISTENCE] = row["existence"]
            rot = row.get("rotation_6d", eye_6d)
            p14[i, P14_COL_ROT_0:P14_COL_ROT_5+1] = rot

        return p14


def parse_xml_to_part_array(xml_content: str, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Standalone convenience function: XML string -> (N, D) tensor."""
    parser = XMLToPartArrayParser()
    return parser.parse(xml_content, device=device)
