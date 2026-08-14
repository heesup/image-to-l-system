"""
Plant Organ Array representation and XML Round-trip Parser/Writer.
Encodes plant hierarchy, nested shoot structure, phytomers, petioles, leaves, and buds into a 2D PyTorch Tensor (N, 93).
"""

import os
import math
import glob
import torch
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any, Optional


# Channel indices for Plant Organ Array Tensor (N, 93)
COL_PLANT_ID = 0
COL_PLANT_AGE = 1
COL_SHOOT_ID = 2
COL_SHOOT_TYPE = 3           # 0=unifoliate, 1=trifoliate
COL_PARENT_SHOOT_ID = 4
COL_PARENT_NODE_IDX = 5
COL_PARENT_PETIOLE_IDX = 6
COL_SHOOT_ROT_PITCH = 7
COL_SHOOT_ROT_YAW = 8
COL_SHOOT_ROT_ROLL = 9
COL_PHYTOMER_IDX = 10

# Internode
COL_INODE_LEN = 11
COL_INODE_RAD = 12
COL_INODE_PITCH = 13
COL_INODE_PHYLLO_ANG = 14
COL_INODE_LEN_MAX = 15
COL_INODE_LEN_SEGS = 16
COL_CURV_PERT_0 = 17
COL_CURV_PERT_1 = 18
COL_YAW_PERT_0 = 19
COL_YAW_PERT_1 = 20

# Petiole 0
COL_PET0_LEN = 21
COL_PET0_RAD = 22
COL_PET0_PITCH = 23
COL_PET0_CURV = 24
COL_PET0_LEAF_SCALE = 25
COL_PET0_TAPER = 26
COL_PET0_LEN_SEGS = 27
COL_PET0_RAD_SUBDIV = 28
COL_PET0_LFLT_SCALE = 29
COL_PET0_LFLT_OFFSET = 30
COL_PET0_NUM_LEAVES = 31

# Leaves of Petiole 0 (up to 3 leaves)
COL_PET0_L0_SCALE = 32
COL_PET0_L0_PITCH = 33
COL_PET0_L0_YAW = 34
COL_PET0_L0_ROLL = 35

COL_PET0_L1_SCALE = 36
COL_PET0_L1_PITCH = 37
COL_PET0_L1_YAW = 38
COL_PET0_L1_ROLL = 39

COL_PET0_L2_SCALE = 40
COL_PET0_L2_PITCH = 41
COL_PET0_L2_YAW = 42
COL_PET0_L2_ROLL = 43

# Petiole 1 (if present in unifoliate shoots)
COL_HAS_PET1 = 44
COL_PET1_LEN = 45
COL_PET1_RAD = 46
COL_PET1_PITCH = 47
COL_PET1_CURV = 48
COL_PET1_LEAF_SCALE = 49
COL_PET1_TAPER = 50
COL_PET1_LEN_SEGS = 51
COL_PET1_RAD_SUBDIV = 52
COL_PET1_LFLT_SCALE = 53
COL_PET1_LFLT_OFFSET = 54
COL_PET1_NUM_LEAVES = 55

COL_PET1_L0_SCALE = 56
COL_PET1_L0_PITCH = 57
COL_PET1_L0_YAW = 58
COL_PET1_L0_ROLL = 59

# Floral Bud
COL_HAS_BUD = 60
COL_BUD_STATE = 61
COL_BUD_PARENT_IDX = 62
COL_BUD_IDX = 63
COL_BUD_IS_TERMINAL = 64
COL_BUD_FRUIT_SCALE = 65

# Peduncle
COL_PED_LEN = 66
COL_PED_RAD = 67
COL_PED_PITCH = 68
COL_PED_CURV = 69
COL_PED_ROLL = 70

# Inflorescence & Flowers (up to 4 flowers)
COL_NUM_FLOWERS = 71
COL_FLOWER_OFFSET = 72

# Flower 0
COL_FL0_PITCH = 73
COL_FL0_YAW = 74
COL_FL0_ROLL = 75
COL_FL0_AZIMUTH = 76
COL_FL0_BASE_SCALE = 77

# Flower 1
COL_FL1_PITCH = 78
COL_FL1_YAW = 79
COL_FL1_ROLL = 80
COL_FL1_AZIMUTH = 81
COL_FL1_BASE_SCALE = 82

# Flower 2
COL_FL2_PITCH = 83
COL_FL2_YAW = 84
COL_FL2_ROLL = 85
COL_FL2_AZIMUTH = 86
COL_FL2_BASE_SCALE = 87

# Flower 3
COL_FL3_PITCH = 88
COL_FL3_YAW = 89
COL_FL3_ROLL = 90
COL_FL3_AZIMUTH = 91
COL_FL3_BASE_SCALE = 92

COL_EXISTENCE = 93
NUM_FEATURES = 94


def _fmt(val: float) -> str:
    """Formats float for exact XML strings."""
    if isinstance(val, str):
        return val
    return f"{val:g}"


class PlantOrganArray:
    """
    Stores plant architecture as a 2D Organ Array Tensor (N, 93) plus raw string metadata tables.
    Supports lossless XML parsing, exact XML generation, and PyTorch geometry building.
    """
    def __init__(
        self,
        tensor: torch.Tensor,
        raw_metadata: Optional[List[Dict[str, Any]]] = None,
        parent_logits: Optional[torch.Tensor] = None,
        parent_candidates: Optional[torch.Tensor] = None,
    ):
        """Tensor shape (N, 94); last column is existence.

        Optional soft parent representation for topology optimization:
          parent_logits: (num_shoots, K) soft weights over K candidate parents.
          parent_candidates: (num_shoots, K, 3) int tensor with candidate
                             (parent_shoot_idx, parent_node_idx, parent_petiole_idx).
        """
        self.tensor = tensor  # (N, 94)
        self.raw_metadata = raw_metadata if raw_metadata is not None else []
        if self.tensor.shape[1] != NUM_FEATURES:
            raise ValueError(f"PlantOrganArray tensor must have {NUM_FEATURES} columns, got {self.tensor.shape[1]}")

        self.parent_logits = parent_logits
        self.parent_candidates = parent_candidates
        if parent_logits is not None and parent_candidates is not None:
            if parent_logits.shape[:1] != parent_candidates.shape[:1]:
                raise ValueError("parent_logits and parent_candidates must have the same number of shoots")
            if parent_logits.shape[1] != parent_candidates.shape[1]:
                raise ValueError("parent_logits and parent_candidates must have the same K")

    def to_xml_string(self, existence_threshold: float = 0.5) -> str:
        """Serializes Organ Array Tensor and raw metadata back to exact Helios XML string.
        Nodes with existence < existence_threshold are skipped."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<helios>'
        ]

        if self.num_nodes == 0:
            lines.append('</helios>')
            return "\n".join(lines) + "\n"

        plant_groups: Dict[int, Dict[int, List[int]]] = {}
        for idx in range(self.num_nodes):
            pid = int(self.tensor[idx, COL_PLANT_ID].item())
            sid = int(self.tensor[idx, COL_SHOOT_ID].item())

            if pid not in plant_groups:
                plant_groups[pid] = {}
            if sid not in plant_groups[pid]:
                plant_groups[pid][sid] = []
            plant_groups[pid][sid].append(idx)

        # Build shoot-level ordering for soft parent hardening
        shoots_dict_for_parent: Dict[int, int] = {}
        for idx in range(self.num_nodes):
            sid = int(self.tensor[idx, COL_SHOOT_ID].item())
            if sid not in shoots_dict_for_parent:
                shoots_dict_for_parent[sid] = idx

        for pid, shoots in plant_groups.items():
            first_idx = list(shoots.values())[0][0]
            first_meta = self.raw_metadata[first_idx] if first_idx < len(self.raw_metadata) else {}

            bp_str = first_meta.get("raw_bp", " 0 0 0 ")
            pa_str = first_meta.get("raw_pa", _fmt(self.tensor[first_idx, COL_PLANT_AGE].item()))

            lines.append(f'\t<plant_instance ID="{pid}">')
            lines.append(f'\t\t<base_position>{bp_str}</base_position>')
            lines.append(f'\t\t<plant_age> {pa_str.strip()} </plant_age>')

            for sid, node_indices in shoots.items():
                s_first_idx = node_indices[0]
                s_meta = self.raw_metadata[s_first_idx] if s_first_idx < len(self.raw_metadata) else {}

                stl_str = s_meta.get("raw_stl", "unifoliate" if self.tensor[s_first_idx, COL_SHOOT_TYPE] == 0 else "trifoliate")

                # If soft parent representation exists, harden to argmax candidate for XML export.
                if self.parent_logits is not None and self.parent_candidates is not None:
                    # Need sorted shoot order index; construct mapping here safely
                    sorted_sids = sorted(shoots_dict_for_parent.keys())
                    sid_to_sorted_idx = {s: i for i, s in enumerate(sorted_sids)}
                    s_idx = sid_to_sorted_idx.get(sid, 0)
                    if s_idx < self.parent_logits.shape[0]:
                        best_k = int(torch.argmax(self.parent_logits[s_idx]).item())
                        best_parent = self.parent_candidates[s_idx, best_k]
                        psi_str = str(int(best_parent[0].item()))
                        pni_str = str(int(best_parent[1].item()))
                        ppi_str = str(int(best_parent[2].item()))
                    else:
                        psi_str = s_meta.get("raw_psi", str(int(self.tensor[s_first_idx, COL_PARENT_SHOOT_ID].item())))
                        pni_str = s_meta.get("raw_pni", str(int(self.tensor[s_first_idx, COL_PARENT_NODE_IDX].item())))
                        ppi_str = s_meta.get("raw_ppi", str(int(self.tensor[s_first_idx, COL_PARENT_PETIOLE_IDX].item())))
                else:
                    psi_str = s_meta.get("raw_psi", str(int(self.tensor[s_first_idx, COL_PARENT_SHOOT_ID].item())))
                    pni_str = s_meta.get("raw_pni", str(int(self.tensor[s_first_idx, COL_PARENT_NODE_IDX].item())))
                    ppi_str = s_meta.get("raw_ppi", str(int(self.tensor[s_first_idx, COL_PARENT_PETIOLE_IDX].item())))

                br_str = s_meta.get("raw_br", f" {_fmt(self.tensor[s_first_idx, COL_SHOOT_ROT_PITCH].item())} {_fmt(self.tensor[s_first_idx, COL_SHOOT_ROT_YAW].item())} {_fmt(self.tensor[s_first_idx, COL_SHOOT_ROT_ROLL].item())} ")

                lines.append(f'\t\t<shoot ID="{sid}">')
                lines.append(f'\t\t\t<shoot_type_label> {stl_str.strip()} </shoot_type_label>')
                lines.append(f'\t\t\t<parent_shoot_ID> {psi_str.strip()} </parent_shoot_ID>')
                lines.append(f'\t\t\t<parent_node_index> {pni_str.strip()} </parent_node_index>')
                lines.append(f'\t\t\t<parent_petiole_index> {ppi_str.strip()} </parent_petiole_index>')
                lines.append(f'\t\t\t<base_rotation>{br_str}</base_rotation>')

                for n_idx in node_indices:
                    if self.tensor[n_idx, COL_EXISTENCE].item() < existence_threshold:
                        continue
                    node_vec = self.tensor[n_idx]
                    meta = self.raw_metadata[n_idx] if n_idx < len(self.raw_metadata) else {}

                    lines.append('\t\t\t<phytomer>')
                    lines.append('\t\t\t\t<internode>')
                    lines.append(f'\t\t\t\t\t<internode_length>{meta.get("raw_il", _fmt(node_vec[COL_INODE_LEN].item()))}</internode_length>')
                    lines.append(f'\t\t\t\t\t<internode_radius>{meta.get("raw_ir", _fmt(node_vec[COL_INODE_RAD].item()))}</internode_radius>')
                    lines.append(f'\t\t\t\t\t<internode_pitch>{meta.get("raw_ip", _fmt(node_vec[COL_INODE_PITCH].item()))}</internode_pitch>')
                    lines.append(f'\t\t\t\t\t<internode_phyllotactic_angle>{meta.get("raw_ipa", _fmt(node_vec[COL_INODE_PHYLLO_ANG].item()))}</internode_phyllotactic_angle>')
                    lines.append(f'\t\t\t\t\t<internode_length_max>{meta.get("raw_ilm", _fmt(node_vec[COL_INODE_LEN_MAX].item()))}</internode_length_max>')
                    lines.append(f'\t\t\t\t\t<internode_length_segments>{meta.get("raw_ils", str(int(node_vec[COL_INODE_LEN_SEGS].item())))}</internode_length_segments>')
                    lines.append(f'\t\t\t\t\t<curvature_perturbations>{meta.get("raw_cp", f"{_fmt(node_vec[COL_CURV_PERT_0].item())};{_fmt(node_vec[COL_CURV_PERT_1].item())}")}</curvature_perturbations>')
                    lines.append(f'\t\t\t\t\t<yaw_perturbations>{meta.get("raw_yp", f"{_fmt(node_vec[COL_YAW_PERT_0].item())};{_fmt(node_vec[COL_YAW_PERT_1].item())}")}</yaw_perturbations>')

                    # Petiole 0
                    lines.append('\t\t\t\t\t<petiole>')
                    lines.append(f'\t\t\t\t\t\t<petiole_length>{meta.get("raw_pet0_l", _fmt(node_vec[COL_PET0_LEN].item()))}</petiole_length>')
                    lines.append(f'\t\t\t\t\t\t<petiole_radius>{meta.get("raw_pet0_r", _fmt(node_vec[COL_PET0_RAD].item()))}</petiole_radius>')
                    lines.append(f'\t\t\t\t\t\t<petiole_pitch>{meta.get("raw_pet0_p", _fmt(node_vec[COL_PET0_PITCH].item()))}</petiole_pitch>')
                    lines.append(f'\t\t\t\t\t\t<petiole_curvature>{meta.get("raw_pet0_c", _fmt(node_vec[COL_PET0_CURV].item()))}</petiole_curvature>')
                    lines.append(f'\t\t\t\t\t\t<current_leaf_scale_factor>{meta.get("raw_pet0_cls", _fmt(node_vec[COL_PET0_LEAF_SCALE].item()))}</current_leaf_scale_factor>')
                    lines.append(f'\t\t\t\t\t\t<petiole_taper>{meta.get("raw_pet0_t", _fmt(node_vec[COL_PET0_TAPER].item()))}</petiole_taper>')
                    lines.append(f'\t\t\t\t\t\t<petiole_length_segments>{meta.get("raw_pet0_ls", str(int(node_vec[COL_PET0_LEN_SEGS].item())))}</petiole_length_segments>')
                    lines.append(f'\t\t\t\t\t\t<petiole_radial_subdivisions>{meta.get("raw_pet0_rs", str(int(node_vec[COL_PET0_RAD_SUBDIV].item())))}</petiole_radial_subdivisions>')
                    lines.append(f'\t\t\t\t\t\t<leaflet_scale>{meta.get("raw_pet0_lfls", _fmt(node_vec[COL_PET0_LFLT_SCALE].item()))}</leaflet_scale>')
                    lines.append(f'\t\t\t\t\t\t<leaflet_offset>{meta.get("raw_pet0_lflo", _fmt(node_vec[COL_PET0_LFLT_OFFSET].item()))}</leaflet_offset>')

                    num_leaves0 = int(node_vec[COL_PET0_NUM_LEAVES].item())
                    leaf_metas0 = meta.get("pet0_leaves", [])
                    for lf_idx in range(num_leaves0):
                        lf_m = leaf_metas0[lf_idx] if lf_idx < len(leaf_metas0) else {}
                        base_col = COL_PET0_L0_SCALE + lf_idx * 4
                        lines.append('\t\t\t\t\t\t<leaf>')
                        lines.append(f'\t\t\t\t\t\t\t<leaf_scale>{lf_m.get("raw_scale", _fmt(node_vec[base_col].item()))}</leaf_scale>')
                        lines.append(f'\t\t\t\t\t\t\t<leaf_pitch>{lf_m.get("raw_pitch", _fmt(node_vec[base_col+1].item()))}</leaf_pitch>')
                        lines.append(f'\t\t\t\t\t\t\t<leaf_yaw>{lf_m.get("raw_yaw", _fmt(node_vec[base_col+2].item()))}</leaf_yaw>')
                        lines.append(f'\t\t\t\t\t\t\t<leaf_roll>{lf_m.get("raw_roll", _fmt(node_vec[base_col+3].item()))}</leaf_roll>')
                        lines.append('\t\t\t\t\t\t</leaf>')

                    if node_vec[COL_HAS_BUD] > 0:
                        lines.append('\t\t\t\t\t\t<floral_bud>')
                        lines.append(f'\t\t\t\t\t\t\t<bud_state>{meta.get("raw_bs", str(int(node_vec[COL_BUD_STATE].item())))}</bud_state>')
                        lines.append(f'\t\t\t\t\t\t\t<parent_index>{meta.get("raw_bpidx", str(int(node_vec[COL_BUD_PARENT_IDX].item())))}</parent_index>')
                        lines.append(f'\t\t\t\t\t\t\t<bud_index>{meta.get("raw_bidx", str(int(node_vec[COL_BUD_IDX].item())))}</bud_index>')
                        lines.append(f'\t\t\t\t\t\t\t<is_terminal>{meta.get("raw_biterm", str(int(node_vec[COL_BUD_IS_TERMINAL].item())))}</is_terminal>')
                        lines.append(f'\t\t\t\t\t\t\t<current_fruit_scale_factor>{meta.get("raw_bcfs", _fmt(node_vec[COL_BUD_FRUIT_SCALE].item()))}</current_fruit_scale_factor>')

                        lines.append('\t\t\t\t\t\t\t<peduncle>')
                        lines.append(f'\t\t\t\t\t\t\t\t<length>{meta.get("raw_ped_l", _fmt(node_vec[COL_PED_LEN].item()))}</length>')
                        lines.append(f'\t\t\t\t\t\t\t\t<radius>{meta.get("raw_ped_r", _fmt(node_vec[COL_PED_RAD].item()))}</radius>')
                        lines.append(f'\t\t\t\t\t\t\t\t<pitch>{meta.get("raw_ped_p", _fmt(node_vec[COL_PED_PITCH].item()))}</pitch>')
                        lines.append(f'\t\t\t\t\t\t\t\t<curvature>{meta.get("raw_ped_c", _fmt(node_vec[COL_PED_CURV].item()))}</curvature>')
                        lines.append(f'\t\t\t\t\t\t\t\t<roll>{meta.get("raw_ped_rl", _fmt(node_vec[COL_PED_ROLL].item()))}</roll>')
                        lines.append('\t\t\t\t\t\t\t</peduncle>')

                        lines.append('\t\t\t\t\t\t\t<inflorescence>')
                        lines.append(f'\t\t\t\t\t\t\t\t<flower_offset>{meta.get("raw_foff", _fmt(node_vec[COL_FLOWER_OFFSET].item()))}</flower_offset>')
                        num_fl = int(node_vec[COL_NUM_FLOWERS].item())
                        fl_metas = meta.get("flowers", [])
                        for fl_idx in range(num_fl):
                            fl_m = fl_metas[fl_idx] if fl_idx < len(fl_metas) else {}
                            fl_base_col = COL_FL0_PITCH + fl_idx * 5
                            lines.append('\t\t\t\t\t\t\t\t<flower>')
                            lines.append(f'\t\t\t\t\t\t\t\t\t<flower_pitch>{fl_m.get("raw_pitch", _fmt(node_vec[fl_base_col].item()))}</flower_pitch>')
                            lines.append(f'\t\t\t\t\t\t\t\t\t<flower_yaw>{fl_m.get("raw_yaw", _fmt(node_vec[fl_base_col+1].item()))}</flower_yaw>')
                            lines.append(f'\t\t\t\t\t\t\t\t\t<flower_roll>{fl_m.get("raw_roll", _fmt(node_vec[fl_base_col+2].item()))}</flower_roll>')
                            lines.append(f'\t\t\t\t\t\t\t\t\t<flower_azimuth>{fl_m.get("raw_azimuth", _fmt(node_vec[fl_base_col+3].item()))}</flower_azimuth>')
                            lines.append(f'\t\t\t\t\t\t\t\t\t<flower_base_scale>{fl_m.get("raw_base_scale", _fmt(node_vec[fl_base_col+4].item()))}</flower_base_scale>')
                            lines.append('\t\t\t\t\t\t\t\t</flower>')
                        lines.append('\t\t\t\t\t\t\t</inflorescence>')
                        lines.append('\t\t\t\t\t\t</floral_bud>')

                    lines.append('\t\t\t\t\t</petiole>')

                    # Petiole 1 (if present)
                    if node_vec[COL_HAS_PET1] > 0:
                        lines.append('\t\t\t\t\t<petiole>')
                        lines.append(f'\t\t\t\t\t\t<petiole_length>{meta.get("raw_pet1_l", _fmt(node_vec[COL_PET1_LEN].item()))}</petiole_length>')
                        lines.append(f'\t\t\t\t\t\t<petiole_radius>{meta.get("raw_pet1_r", _fmt(node_vec[COL_PET1_RAD].item()))}</petiole_radius>')
                        lines.append(f'\t\t\t\t\t\t<petiole_pitch>{meta.get("raw_pet1_p", _fmt(node_vec[COL_PET1_PITCH].item()))}</petiole_pitch>')
                        lines.append(f'\t\t\t\t\t\t<petiole_curvature>{meta.get("raw_pet1_c", _fmt(node_vec[COL_PET1_CURV].item()))}</petiole_curvature>')
                        lines.append(f'\t\t\t\t\t\t<current_leaf_scale_factor>{meta.get("raw_pet1_cls", _fmt(node_vec[COL_PET1_LEAF_SCALE].item()))}</current_leaf_scale_factor>')
                        lines.append(f'\t\t\t\t\t\t<petiole_taper>{meta.get("raw_pet1_t", _fmt(node_vec[COL_PET1_TAPER].item()))}</petiole_taper>')
                        lines.append(f'\t\t\t\t\t\t<petiole_length_segments>{meta.get("raw_pet1_ls", str(int(node_vec[COL_PET1_LEN_SEGS].item())))}</petiole_length_segments>')
                        lines.append(f'\t\t\t\t\t\t<petiole_radial_subdivisions>{meta.get("raw_pet1_rs", str(int(node_vec[COL_PET1_RAD_SUBDIV].item())))}</petiole_radial_subdivisions>')
                        lines.append(f'\t\t\t\t\t\t<leaflet_scale>{meta.get("raw_pet1_lfls", _fmt(node_vec[COL_PET1_LFLT_SCALE].item()))}</leaflet_scale>')
                        lines.append(f'\t\t\t\t\t\t<leaflet_offset>{meta.get("raw_pet1_lflo", _fmt(node_vec[COL_PET1_LFLT_OFFSET].item()))}</leaflet_offset>')

                        num_leaves1 = int(node_vec[COL_PET1_NUM_LEAVES].item())
                        leaf_metas1 = meta.get("pet1_leaves", [])
                        for lf_idx in range(num_leaves1):
                            lf_m = leaf_metas1[lf_idx] if lf_idx < len(leaf_metas1) else {}
                            base_col = COL_PET1_L0_SCALE + lf_idx * 4
                            lines.append('\t\t\t\t\t\t<leaf>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_scale>{lf_m.get("raw_scale", _fmt(node_vec[base_col].item()))}</leaf_scale>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_pitch>{lf_m.get("raw_pitch", _fmt(node_vec[base_col+1].item()))}</leaf_pitch>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_yaw>{lf_m.get("raw_yaw", _fmt(node_vec[base_col+2].item()))}</leaf_yaw>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_roll>{lf_m.get("raw_roll", _fmt(node_vec[base_col+3].item()))}</leaf_roll>')
                            lines.append('\t\t\t\t\t\t</leaf>')

                        lines.append('\t\t\t\t\t</petiole>')

                    lines.append('\t\t\t\t</internode>')
                    lines.append('\t\t\t</phytomer>')

                lines.append('\t\t</shoot>')

            lines.append('\t</plant_instance>')

        lines.append('</helios>')
        return "\n".join(lines) + "\n"

    def write_xml(self, filepath: str) -> None:
        content = self.to_xml_string()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    @property
    def num_nodes(self) -> int:
        return self.tensor.shape[0]

    @property
    def existence(self) -> torch.Tensor:
        """Continuous per-node existence probability, last column of tensor."""
        return self.tensor[:, COL_EXISTENCE]

    @existence.setter
    def existence(self, value: torch.Tensor) -> None:
        if value.shape[0] != self.tensor.shape[0]:
            raise ValueError("existence length must match number of nodes")
        self.tensor[:, COL_EXISTENCE] = value

    def clone_with_parent_logits(
        self,
        parent_logits: torch.Tensor,
        parent_candidates: torch.Tensor,
    ) -> "PlantOrganArray":
        """Return a new PlantOrganArray with the same tensor/metadata but new soft parent representation."""
        return PlantOrganArray(
            tensor=self.tensor.clone(),
            raw_metadata=self.raw_metadata,
            parent_logits=parent_logits,
            parent_candidates=parent_candidates,
        )

    @staticmethod
    def _xml_parent_node_to_linear_idx(
        tensor: torch.Tensor,
        parent_shoot_id: int,
        parent_node_xml: int,
    ) -> int:
        """Map XML parent_node_index (1-based phytomer index) to a linear node index.

        The XML stores parent_node_index as a 1-based phytomer index along the parent shoot.
        Our geometry builder indexes nodes by their linear row index. This helper finds
        the node in the parent shoot whose COL_PHYTOMER_IDX equals (parent_node_xml - 1).
        """
        if parent_shoot_id < 0:
            return -1
        target_phyt_idx = parent_node_xml - 1 if parent_node_xml > 0 else 0
        N = tensor.shape[0]
        for idx in range(N):
            sid = int(tensor[idx, COL_SHOOT_ID].item())
            phyt_idx = int(tensor[idx, COL_PHYTOMER_IDX].item())
            if sid == parent_shoot_id and phyt_idx == target_phyt_idx:
                return idx
        # Fallback to the first node of the parent shoot if exact phytomer not found
        for idx in range(N):
            if int(tensor[idx, COL_SHOOT_ID].item()) == parent_shoot_id:
                return idx
        return 0

    @staticmethod
    def build_parent_candidates_from_gt(
        organ_array: "PlantOrganArray",
        num_candidates: int = 8,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create soft parent candidates: 1 GT parent + (K-1) random noise candidates per shoot.

        Returns:
          parent_logits: (num_shoots, K), initialized with large logit on GT candidate.
          parent_candidates: (num_shoots, K, 3) int tensor of
                             (parent_shoot_idx, parent_linear_node_idx, parent_petiole_idx).
        """
        cpu_rng = torch.Generator(device="cpu").manual_seed(seed)
        t = organ_array.tensor
        N = organ_array.num_nodes

        # Map shoot_id -> list of node indices and first node per shoot
        shoots_dict: Dict[int, List[int]] = {}
        for idx in range(N):
            sid = int(t[idx, COL_SHOOT_ID].item())
            shoots_dict.setdefault(sid, []).append(idx)

        sorted_shoot_ids = sorted(shoots_dict.keys())
        num_shoots = len(sorted_shoot_ids)
        node_index_to_shoot_first = {}
        for sid in sorted_shoot_ids:
            node_index_to_shoot_first[sid] = shoots_dict[sid][0]

        parent_candidates = torch.zeros((num_shoots, num_candidates, 3), dtype=torch.int64)
        for s_idx, sid in enumerate(sorted_shoot_ids):
            first_node = node_index_to_shoot_first[sid]
            gt_shoot = int(t[first_node, COL_PARENT_SHOOT_ID].item())
            gt_node_xml = int(t[first_node, COL_PARENT_NODE_IDX].item())
            gt_pet = int(t[first_node, COL_PARENT_PETIOLE_IDX].item())
            if gt_shoot < 0:
                # Root shoot: use sentinel node index 0 (will be ignored by negative shoot check in soft path)
                gt_node_linear = 0
            else:
                gt_node_linear = PlantOrganArray._xml_parent_node_to_linear_idx(t, gt_shoot, gt_node_xml)
            parent_candidates[s_idx, 0] = torch.tensor([gt_shoot, gt_node_linear, gt_pet])

            for k in range(1, num_candidates):
                if torch.rand(1, generator=cpu_rng).item() < 0.5:
                    rand_shoot = int(torch.randint(min(sorted_shoot_ids), max(sorted_shoot_ids) + 1, (1,), generator=cpu_rng).item())
                else:
                    rand_shoot = gt_shoot
                rand_node = int(torch.randint(0, N, (1,), generator=cpu_rng).item())
                rand_pet = int(torch.randint(0, 2, (1,), generator=cpu_rng).item())
                parent_candidates[s_idx, k] = torch.tensor([rand_shoot, rand_node, rand_pet])

        # Initialize logits: strong prior on GT candidate (index 0)
        logits = torch.full((num_shoots, num_candidates), -2.0, dtype=torch.float32)
        logits[:, 0] = 2.0
        return logits, parent_candidates

    @classmethod
    def from_xml_string(cls, xml_content: str) -> "PlantOrganArray":
        root = ET.fromstring(xml_content)
        if root.tag != "helios":
            raise ValueError("Root tag must be <helios>")

        rows = []
        raw_metadata = []

        for plant_elem in root.findall("plant_instance"):
            plant_id = int(plant_elem.attrib.get("ID", 0))

            bp_elem = plant_elem.find("base_position")
            raw_bp = bp_elem.text if bp_elem is not None else " 0 0 0 "
            bp_vals = [float(x) for x in raw_bp.strip().split()]
            bp = (bp_vals[0], bp_vals[1], bp_vals[2]) if len(bp_vals) >= 3 else (0.0, 0.0, 0.0)

            age_elem = plant_elem.find("plant_age")
            raw_pa = age_elem.text if age_elem is not None else "0"
            plant_age = float(raw_pa.strip())

            for shoot_elem in plant_elem.findall("shoot"):
                shoot_id = int(shoot_elem.attrib.get("ID", 0))

                stl_elem = shoot_elem.find("shoot_type_label")
                raw_stl = stl_elem.text if stl_elem is not None else "unifoliate"
                shoot_type = 0 if "unifoliate" in raw_stl else 1

                psi_elem = shoot_elem.find("parent_shoot_ID")
                raw_psi = psi_elem.text if psi_elem is not None else "-1"
                psi = int(raw_psi.strip())

                pni_elem = shoot_elem.find("parent_node_index")
                raw_pni = pni_elem.text if pni_elem is not None else "0"
                pni = int(raw_pni.strip())

                ppi_elem = shoot_elem.find("parent_petiole_index")
                raw_ppi = ppi_elem.text if ppi_elem is not None else "0"
                ppi = int(raw_ppi.strip())

                br_elem = shoot_elem.find("base_rotation")
                raw_br = br_elem.text if br_elem is not None else " 0 0 0 "
                br_vals = [float(x) for x in raw_br.strip().split()]
                br = (br_vals[0], br_vals[1], br_vals[2]) if len(br_vals) >= 3 else (0.0, 0.0, 0.0)

                for phyto_idx, phyto_elem in enumerate(shoot_elem.findall("phytomer")):
                    row = [0.0] * NUM_FEATURES
                    meta: Dict[str, Any] = {
                        "raw_bp": raw_bp,
                        "raw_pa": raw_pa,
                        "raw_stl": raw_stl,
                        "raw_psi": raw_psi,
                        "raw_pni": raw_pni,
                        "raw_ppi": raw_ppi,
                        "raw_br": raw_br,
                    }

                    row[COL_PLANT_ID] = float(plant_id)
                    row[COL_PLANT_AGE] = plant_age
                    row[COL_SHOOT_ID] = float(shoot_id)
                    row[COL_SHOOT_TYPE] = float(shoot_type)
                    row[COL_PARENT_SHOOT_ID] = float(psi)
                    row[COL_PARENT_NODE_IDX] = float(pni)
                    row[COL_PARENT_PETIOLE_IDX] = float(ppi)
                    row[COL_SHOOT_ROT_PITCH] = br[0]
                    row[COL_SHOOT_ROT_YAW] = br[1]
                    row[COL_SHOOT_ROT_ROLL] = br[2]
                    row[COL_PHYTOMER_IDX] = float(phyto_idx)

                    internode_elem = phyto_elem.find("internode")
                    if internode_elem is not None:
                        il_elem = internode_elem.find("internode_length")
                        meta["raw_il"] = il_elem.text if il_elem is not None else "0"
                        row[COL_INODE_LEN] = float(meta["raw_il"].strip())

                        ir_elem = internode_elem.find("internode_radius")
                        meta["raw_ir"] = ir_elem.text if ir_elem is not None else "0"
                        row[COL_INODE_RAD] = float(meta["raw_ir"].strip())

                        ip_elem = internode_elem.find("internode_pitch")
                        meta["raw_ip"] = ip_elem.text if ip_elem is not None else "0"
                        row[COL_INODE_PITCH] = float(meta["raw_ip"].strip())

                        ipa_elem = internode_elem.find("internode_phyllotactic_angle")
                        meta["raw_ipa"] = ipa_elem.text if ipa_elem is not None else "0"
                        row[COL_INODE_PHYLLO_ANG] = float(meta["raw_ipa"].strip())

                        ilm_elem = internode_elem.find("internode_length_max")
                        meta["raw_ilm"] = ilm_elem.text if ilm_elem is not None else "0"
                        row[COL_INODE_LEN_MAX] = float(meta["raw_ilm"].strip())

                        ils_elem = internode_elem.find("internode_length_segments")
                        meta["raw_ils"] = ils_elem.text if ils_elem is not None else "2"
                        row[COL_INODE_LEN_SEGS] = float(meta["raw_ils"].strip())

                        cp_elem = internode_elem.find("curvature_perturbations")
                        meta["raw_cp"] = cp_elem.text if cp_elem is not None else "0;0"
                        cp_list = [float(x) for x in meta["raw_cp"].strip().split(";") if x.strip()]
                        if len(cp_list) > 0: row[COL_CURV_PERT_0] = cp_list[0]
                        if len(cp_list) > 1: row[COL_CURV_PERT_1] = cp_list[1]

                        yp_elem = internode_elem.find("yaw_perturbations")
                        meta["raw_yp"] = yp_elem.text if yp_elem is not None else "0;0"
                        yp_list = [float(x) for x in meta["raw_yp"].strip().split(";") if x.strip()]
                        if len(yp_list) > 0: row[COL_YAW_PERT_0] = yp_list[0]
                        if len(yp_list) > 1: row[COL_YAW_PERT_1] = yp_list[1]

                        petiole_elems = internode_elem.findall("petiole")
                        for pet_i, pet_elem in enumerate(petiole_elems):
                            if pet_i > 1: break # Support up to 2 petioles per phytomer
                            prefix = "pet0_" if pet_i == 0 else "pet1_"

                            pl_elem = pet_elem.find("petiole_length")
                            meta[prefix + "l"] = pl_elem.text if pl_elem is not None else "0"
                            pr_elem = pet_elem.find("petiole_radius")
                            meta[prefix + "r"] = pr_elem.text if pr_elem is not None else "0"
                            pp_elem = pet_elem.find("petiole_pitch")
                            meta[prefix + "p"] = pp_elem.text if pp_elem is not None else "0"
                            pc_elem = pet_elem.find("petiole_curvature")
                            meta[prefix + "c"] = pc_elem.text if pc_elem is not None else "0"
                            cls_elem = pet_elem.find("current_leaf_scale_factor")
                            meta[prefix + "cls"] = cls_elem.text if cls_elem is not None else "1"
                            pt_elem = pet_elem.find("petiole_taper")
                            meta[prefix + "t"] = pt_elem.text if pt_elem is not None else "0.25"
                            pls_elem = pet_elem.find("petiole_length_segments")
                            meta[prefix + "ls"] = pls_elem.text if pls_elem is not None else "5"
                            prs_elem = pet_elem.find("petiole_radial_subdivisions")
                            meta[prefix + "rs"] = prs_elem.text if prs_elem is not None else "6"
                            ls_elem = pet_elem.find("leaflet_scale")
                            meta[prefix + "lfls"] = ls_elem.text if ls_elem is not None else "1"
                            lo_elem = pet_elem.find("leaflet_offset")
                            meta[prefix + "lflo"] = lo_elem.text if lo_elem is not None else "0.4"

                            base_col = COL_PET0_LEN if pet_i == 0 else COL_PET1_LEN
                            if pet_i == 1:
                                row[COL_HAS_PET1] = 1.0

                            row[base_col]     = float(meta[prefix + "l"].strip())
                            row[base_col + 1] = float(meta[prefix + "r"].strip())
                            row[base_col + 2] = float(meta[prefix + "p"].strip())
                            row[base_col + 3] = float(meta[prefix + "c"].strip())
                            row[base_col + 4] = float(meta[prefix + "cls"].strip())
                            row[base_col + 5] = float(meta[prefix + "t"].strip())
                            row[base_col + 6] = float(meta[prefix + "ls"].strip())
                            row[base_col + 7] = float(meta[prefix + "rs"].strip())
                            row[base_col + 8] = float(meta[prefix + "lfls"].strip())
                            row[base_col + 9] = float(meta[prefix + "lflo"].strip())

                            leaf_elems = pet_elem.findall("leaf")
                            row[base_col + 10] = float(len(leaf_elems))
                            leaf_metas = []

                            leaf_base_col = COL_PET0_L0_SCALE if pet_i == 0 else COL_PET1_L0_SCALE
                            for lf_idx, leaf_elem in enumerate(leaf_elems):
                                if lf_idx >= 3: break
                                lfs_elem = leaf_elem.find("leaf_scale")
                                raw_lfs = lfs_elem.text if lfs_elem is not None else "1"
                                lfp_elem = leaf_elem.find("leaf_pitch")
                                raw_lfp = lfp_elem.text if lfp_elem is not None else "0"
                                lfy_elem = leaf_elem.find("leaf_yaw")
                                raw_lfy = lfy_elem.text if lfy_elem is not None else "0"
                                lfr_elem = leaf_elem.find("leaf_roll")
                                raw_lfr = lfr_elem.text if lfr_elem is not None else "0"

                                leaf_metas.append({
                                    "raw_scale": raw_lfs,
                                    "raw_pitch": raw_lfp,
                                    "raw_yaw": raw_lfy,
                                    "raw_roll": raw_lfr
                                })

                                cur_col = leaf_base_col + lf_idx * 4
                                row[cur_col]     = float(raw_lfs.strip())
                                row[cur_col + 1] = float(raw_lfp.strip())
                                row[cur_col + 2] = float(raw_lfy.strip())
                                row[cur_col + 3] = float(raw_lfr.strip())

                            meta[prefix + "leaves"] = leaf_metas

                            # Floral bud (on petiole 0)
                            if pet_i == 0:
                                fb_elem = pet_elem.find("floral_bud")
                                if fb_elem is not None:
                                    row[COL_HAS_BUD] = 1.0
                                    bs_elem = fb_elem.find("bud_state")
                                    meta["raw_bs"] = bs_elem.text if bs_elem is not None else "5"
                                    row[COL_BUD_STATE] = float(meta["raw_bs"].strip())

                                    pidx_elem = fb_elem.find("parent_index")
                                    meta["raw_bpidx"] = pidx_elem.text if pidx_elem is not None else "0"
                                    row[COL_BUD_PARENT_IDX] = float(meta["raw_bpidx"].strip())

                                    bidx_elem = fb_elem.find("bud_index")
                                    meta["raw_bidx"] = bidx_elem.text if bidx_elem is not None else "0"
                                    row[COL_BUD_IDX] = float(meta["raw_bidx"].strip())

                                    iterm_elem = fb_elem.find("is_terminal")
                                    meta["raw_biterm"] = iterm_elem.text if iterm_elem is not None else "0"
                                    row[COL_BUD_IS_TERMINAL] = float(meta["raw_biterm"].strip())

                                    cfs_elem = fb_elem.find("current_fruit_scale_factor")
                                    meta["raw_bcfs"] = cfs_elem.text if cfs_elem is not None else "1"
                                    row[COL_BUD_FRUIT_SCALE] = float(meta["raw_bcfs"].strip())

                                    ped_elem = fb_elem.find("peduncle")
                                    if ped_elem is not None:
                                        plen_elem = ped_elem.find("length")
                                        meta["raw_ped_l"] = plen_elem.text if plen_elem is not None else "0"
                                        row[COL_PED_LEN] = float(meta["raw_ped_l"].strip())

                                        prad_elem = ped_elem.find("radius")
                                        meta["raw_ped_r"] = prad_elem.text if prad_elem is not None else "0"
                                        row[COL_PED_RAD] = float(meta["raw_ped_r"].strip())

                                        ppitch_elem = ped_elem.find("pitch")
                                        meta["raw_ped_p"] = ppitch_elem.text if ppitch_elem is not None else "0"
                                        row[COL_PED_PITCH] = float(meta["raw_ped_p"].strip())

                                        pcurv_elem = ped_elem.find("curvature")
                                        meta["raw_ped_c"] = pcurv_elem.text if pcurv_elem is not None else "0"
                                        row[COL_PED_CURV] = float(meta["raw_ped_c"].strip())

                                        proll_elem = ped_elem.find("roll")
                                        meta["raw_ped_rl"] = proll_elem.text if proll_elem is not None else "0"
                                        row[COL_PED_ROLL] = float(meta["raw_ped_rl"].strip())

                                    infl_elem = fb_elem.find("inflorescence")
                                    if infl_elem is not None:
                                        foff_elem = infl_elem.find("flower_offset")
                                        meta["raw_foff"] = foff_elem.text if foff_elem is not None else "0.05"
                                        row[COL_FLOWER_OFFSET] = float(meta["raw_foff"].strip())

                                        flower_elems = infl_elem.findall("flower")
                                        row[COL_NUM_FLOWERS] = float(len(flower_elems))
                                        fl_metas = []

                                        for fl_idx, fl_elem in enumerate(flower_elems):
                                            if fl_idx >= 4: break # Support up to 4 flowers
                                            fp_elem = fl_elem.find("flower_pitch")
                                            raw_fp = fp_elem.text if fp_elem is not None else "0"
                                            fy_elem = fl_elem.find("flower_yaw")
                                            raw_fy = fy_elem.text if fy_elem is not None else "0"
                                            fr_elem = fl_elem.find("flower_roll")
                                            raw_fr = fr_elem.text if fr_elem is not None else "0"
                                            fa_elem = fl_elem.find("flower_azimuth")
                                            raw_fa = fa_elem.text if fa_elem is not None else "0"
                                            fbs_elem = fl_elem.find("flower_base_scale")
                                            raw_fbs = fbs_elem.text if fbs_elem is not None else "1"

                                            fl_metas.append({
                                                "raw_pitch": raw_fp,
                                                "raw_yaw": raw_fy,
                                                "raw_roll": raw_fr,
                                                "raw_azimuth": raw_fa,
                                                "raw_base_scale": raw_fbs
                                            })

                                            fl_base_col = COL_FL0_PITCH + fl_idx * 5
                                            row[fl_base_col]     = float(raw_fp.strip())
                                            row[fl_base_col + 1] = float(raw_fy.strip())
                                            row[fl_base_col + 2] = float(raw_fr.strip())
                                            row[fl_base_col + 3] = float(raw_fa.strip())
                                            row[fl_base_col + 4] = float(raw_fbs.strip())

                                        meta["flowers"] = fl_metas

                    # existence appended below (integrated into tensor)
                    rows.append(row)
                    raw_metadata.append(meta)

        tensor = torch.tensor(rows, dtype=torch.float32)
        # Ensure last column is existence=1.0
        tensor[:, COL_EXISTENCE] = 1.0
        return cls(tensor=tensor, raw_metadata=raw_metadata)

    @classmethod
    def from_xml_file(cls, filepath: str) -> "PlantOrganArray":
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_xml_string(content)
