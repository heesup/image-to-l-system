"""
Plant Organ Array representation and XML Round-trip Parser/Writer.

This module provides two tensor layouts:

  1. LEGACY (N, 94): fixed phytomer-slot layout used by the initial diffusion
     pipeline. Kept for backward compatibility and marked for deletion.

  2. TYPED   (N, 40): per-organ-row layout with a categorical organ_type
     column. Fields are shared only when they have the same physical meaning.
     XML round-trip is honest: every tag is reconstructed from the tensor,
     no raw XML text is cached.

Organ types (categorical integer, col ORGAN_TYPE):
  0 = ROOT_META   (plant base + age)
  1 = SHOOT_META  (shoot metadata + base rotation)
  2 = INTERNODE
  3 = PETIOLE
  4 = LEAF
  5 = BUD
  6 = PEDUNCLE
  7 = FLOWER
"""

import os
import math
import glob
import torch
import torch.nn.functional as F
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any, Optional


# =============================================================================
# LEGACY 94D COLUMN CONSTANTS
# =============================================================================
# DEPRECATED: scheduled for deletion after the typed (N, 40) migration.

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
NUM_FEATURES_LEGACY = 94


# =============================================================================
# TYPED 40D COLUMN CONSTANTS
# =============================================================================

T_COL_PLANT_ID = 0
T_COL_PLANT_AGE = 1
T_COL_BASE_X = 2
T_COL_BASE_Y = 3
T_COL_BASE_Z = 4
T_COL_SHOOT_ID = 5
T_COL_PARENT_SHOOT_ID = 6
T_COL_PARENT_NODE_IDX = 7
T_COL_PARENT_PETIOLE_IDX = 8
T_COL_PHYTOMER_IDX = 9
T_COL_CHILD_INDEX = 10
T_COL_ORGAN_TYPE = 11
T_COL_SHOOT_TYPE = 12
T_COL_LENGTH = 13
T_COL_RADIUS = 14
T_COL_SCALE = 15
T_COL_PITCH = 16
T_COL_YAW = 17
T_COL_ROLL = 18
T_COL_CURVATURE = 19
T_COL_PHYLLOTACTIC_ANGLE = 20
T_COL_LENGTH_MAX = 21
T_COL_LENGTH_SEGMENTS = 22
T_COL_CURV_PERT_0 = 23
T_COL_CURV_PERT_1 = 24
T_COL_YAW_PERT_0 = 25
T_COL_YAW_PERT_1 = 26
T_COL_CURRENT_LEAF_SCALE_FACTOR = 27
T_COL_TAPER = 28
T_COL_RADIAL_SUBDIVISIONS = 29
T_COL_LEAFLET_SCALE = 30
T_COL_LEAFLET_OFFSET = 31
T_COL_BUD_STATE = 32
T_COL_BUD_PARENT_INDEX = 33
T_COL_BUD_IS_TERMINAL = 34
T_COL_FRUIT_SCALE = 35
T_COL_FLOWER_AZIMUTH = 36
T_COL_FLOWER_OFFSET = 37
T_COL_RESERVED = 38
T_COL_EXISTENCE = 39
NUM_FEATURES_TYPED = 40

# Categorical organ types
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
ORGAN_BUD_ABORTED = 10
NUM_ORGAN_TYPES = 11

# =============================================================================
# PART TENSOR 16D COLUMN CONSTANTS
# =============================================================================
P_COL_ORGAN_TYPE = 0
P_COL_BASE_X = 1
P_COL_BASE_Y = 2
P_COL_BASE_Z = 3
P_COL_ROT_0 = 4
P_COL_ROT_1 = 5
P_COL_ROT_2 = 6
P_COL_ROT_3 = 7
P_COL_ROT_4 = 8
P_COL_ROT_5 = 9
P_COL_SCALE_X = 10
P_COL_SCALE_Y = 11
P_COL_SCALE_Z = 12
P_COL_EXISTENCE = 13
P_COL_CURVATURE = 14
P_COL_PHYLLOTACTIC_ANGLE = 15
NUM_FEATURES = 16
NUM_FEATURES_PART = 16


# =============================================================================
# SMALL HELPERS
# =============================================================================

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Converts 6D rotation representation (Zhou et al. CVPR 2019) to (..., 3, 3) rotation matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


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


# =============================================================================
# PLANT ORGAN ARRAY CLASS
# =============================================================================

class PlantOrganArray:
    """
    Stores plant architecture as a 2D Organ Array Tensor.

    Supports both the legacy (N, 94) phytomer-slot layout and the new typed
    (N, 40) per-organ-row layout. The legacy path is kept for backward
    compatibility and is marked for deletion.

    Optional soft parent representation for topology optimization:
      parent_logits: (num_shoots, K) soft weights over K candidate parents.
      parent_candidates: (num_shoots, K, 3) int tensor with candidate
                           (parent_shoot_idx, parent_node_idx, parent_petiole_idx).
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        raw_metadata: Optional[List[Dict[str, Any]]] = None,
        parent_logits: Optional[torch.Tensor] = None,
        parent_candidates: Optional[torch.Tensor] = None,
    ):
        self.tensor = tensor
        self.raw_metadata = raw_metadata if raw_metadata is not None else []

        if self.tensor.ndim != 2:
            raise ValueError(f"PlantOrganArray tensor must be 2D, got shape {self.tensor.shape}")

        self.num_features = self.tensor.shape[1]
        if self.num_features not in (NUM_FEATURES_LEGACY, NUM_FEATURES_TYPED):
            raise ValueError(
                f"PlantOrganArray tensor must have {NUM_FEATURES_LEGACY} (legacy) or "
                f"{NUM_FEATURES_TYPED} (typed) columns, got {self.num_features}"
            )

        self.parent_logits = parent_logits
        self.parent_candidates = parent_candidates
        if parent_logits is not None and parent_candidates is not None:
            if parent_logits.shape[:1] != parent_candidates.shape[:1]:
                raise ValueError("parent_logits and parent_candidates must have the same number of shoots")
            if parent_logits.shape[1] != parent_candidates.shape[1]:
                raise ValueError("parent_logits and parent_candidates must have the same K")

    @property
    def is_typed(self) -> bool:
        return self.tensor.shape[1] == NUM_FEATURES_TYPED

    @property
    def is_legacy(self) -> bool:
        return self.tensor.shape[1] == NUM_FEATURES_LEGACY

    @property
    def num_nodes(self) -> int:
        return self.tensor.shape[0]

    @property
    def existence(self) -> torch.Tensor:
        if self.is_typed:
            return self.tensor[:, T_COL_EXISTENCE]
        return self.tensor[:, COL_EXISTENCE]

    @existence.setter
    def existence(self, value: torch.Tensor) -> None:
        if value.shape[0] != self.tensor.shape[0]:
            raise ValueError("existence length must match number of nodes")
        if self.is_typed:
            self.tensor[:, T_COL_EXISTENCE] = value
        else:
            self.tensor[:, COL_EXISTENCE] = value

    # -------------------------------------------------------------------------
    # XML OUTPUT DISPATCH
    # -------------------------------------------------------------------------
    def to_xml_string(self, existence_threshold: float = 0.5) -> str:
        """Serializes the tensor back to exact Helios XML string."""
        if self.is_typed:
            return self._to_xml_string_typed(existence_threshold=existence_threshold)
        return self._to_xml_string_legacy(existence_threshold=existence_threshold)

    def write_xml(self, filepath: str) -> None:
        content = self.to_xml_string()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    # -------------------------------------------------------------------------
    # XML INPUT DISPATCH
    # -------------------------------------------------------------------------
    @classmethod
    def from_xml_string(cls, xml_content: str) -> "PlantOrganArray":
        """Returns a PlantOrganArray with the typed (N, 40) per-organ layout."""
        return cls._from_xml_string_typed(xml_content)

    @classmethod
    def from_xml_string_typed(cls, xml_content: str) -> "PlantOrganArray":
        """TYPED: returns a PlantOrganArray with the (N, 40) per-organ layout."""
        return cls._from_xml_string_typed(xml_content)

    @classmethod
    def from_xml_file(cls, filepath: str) -> "PlantOrganArray":
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls._from_xml_string_typed(content)

    @classmethod
    def from_xml_file_typed(cls, filepath: str) -> "PlantOrganArray":
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_xml_string_typed(content)

    # -------------------------------------------------------------------------
    # LEGACY 94D XML WRITER
    # -------------------------------------------------------------------------
    def _to_xml_string_legacy(self, existence_threshold: float = 0.5) -> str:
        """DEPRECATED: legacy (N, 94) XML writer."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<helios>'
        ]
        t = self.tensor

        if self.num_nodes == 0:
            lines.append('</helios>')
            return "\n".join(lines) + "\n"

        plant_groups: Dict[int, Dict[int, List[int]]] = {}
        for idx in range(self.num_nodes):
            pid = int(t[idx, COL_PLANT_ID].item())
            sid = int(t[idx, COL_SHOOT_ID].item())
            plant_groups.setdefault(pid, {}).setdefault(sid, []).append(idx)

        shoots_dict_for_parent: Dict[int, int] = {}
        for idx in range(self.num_nodes):
            sid = int(t[idx, COL_SHOOT_ID].item())
            if sid not in shoots_dict_for_parent:
                shoots_dict_for_parent[sid] = idx

        for pid, shoots in plant_groups.items():
            first_idx = list(shoots.values())[0][0]
            first_meta = self.raw_metadata[first_idx] if first_idx < len(self.raw_metadata) else {}

            bp_str = first_meta.get("raw_bp", " 0 0 0 ")
            pa_str = first_meta.get("raw_pa", _fmt(t[first_idx, COL_PLANT_AGE].item()))

            lines.append(f'\t<plant_instance ID="{pid}">')
            lines.append(f'\t\t<base_position>{bp_str}</base_position>')
            lines.append(f'\t\t<plant_age> {pa_str.strip()} </plant_age>')

            for sid, node_indices in shoots.items():
                s_first_idx = node_indices[0]
                s_meta = self.raw_metadata[s_first_idx] if s_first_idx < len(self.raw_metadata) else {}
                stl_str = s_meta.get("raw_stl", "unifoliate" if t[s_first_idx, COL_SHOOT_TYPE] == 0 else "trifoliate")

                if self.parent_logits is not None and self.parent_candidates is not None:
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
                        psi_str = s_meta.get("raw_psi", str(int(t[s_first_idx, COL_PARENT_SHOOT_ID].item())))
                        pni_str = s_meta.get("raw_pni", str(int(t[s_first_idx, COL_PARENT_NODE_IDX].item())))
                        ppi_str = s_meta.get("raw_ppi", str(int(t[s_first_idx, COL_PARENT_PETIOLE_IDX].item())))
                else:
                    psi_str = s_meta.get("raw_psi", str(int(t[s_first_idx, COL_PARENT_SHOOT_ID].item())))
                    pni_str = s_meta.get("raw_pni", str(int(t[s_first_idx, COL_PARENT_NODE_IDX].item())))
                    ppi_str = s_meta.get("raw_ppi", str(int(t[s_first_idx, COL_PARENT_PETIOLE_IDX].item())))

                br_str = s_meta.get("raw_br", f" {_fmt(t[s_first_idx, COL_SHOOT_ROT_PITCH].item())} {_fmt(t[s_first_idx, COL_SHOOT_ROT_YAW].item())} {_fmt(t[s_first_idx, COL_SHOOT_ROT_ROLL].item())} ")

                lines.append(f'\t\t<shoot ID="{sid}">')
                lines.append(f'\t\t\t<shoot_type_label> {stl_str.strip()} </shoot_type_label>')
                lines.append(f'\t\t\t<parent_shoot_ID> {psi_str.strip()} </parent_shoot_ID>')
                lines.append(f'\t\t\t<parent_node_index> {pni_str.strip()} </parent_node_index>')
                lines.append(f'\t\t\t<parent_petiole_index> {ppi_str.strip()} </parent_petiole_index>')
                lines.append(f'\t\t\t<base_rotation>{br_str}</base_rotation>')

                for n_idx in node_indices:
                    if t[n_idx, COL_EXISTENCE].item() < existence_threshold:
                        continue
                    node_vec = t[n_idx]
                    meta = self.raw_metadata[n_idx] if n_idx < len(self.raw_metadata) else {}
                    p_idx = int(node_vec[COL_PHYTOMER_IDX].item())

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

                    # Petiole 1
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

    # -------------------------------------------------------------------------
    # TYPED 40D XML WRITER (honest round-trip)
    # -------------------------------------------------------------------------
    def _to_xml_string_typed(self, existence_threshold: float = 0.5) -> str:
        """TYPED (N, 40) XML writer. Reconstructs XML directly from tensor."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<helios>'
        ]
        t = self.tensor
        N = self.num_nodes
        if N == 0:
            lines.append('</helios>')
            return "\n".join(lines) + "\n"

        # Build dictionaries: plant_id -> shoot_id -> phytomer_idx -> organ rows
        plants: Dict[int, Dict[int, Dict[int, List[torch.Tensor]]]] = {}
        root_meta: Dict[int, torch.Tensor] = {}
        shoot_meta: Dict[Tuple[int, int], torch.Tensor] = {}

        for idx in range(N):
            if t[idx, T_COL_EXISTENCE].item() < existence_threshold:
                continue
            pid = _to_int(t[idx, T_COL_PLANT_ID])
            sid = _to_int(t[idx, T_COL_SHOOT_ID])
            ot = _to_int(t[idx, T_COL_ORGAN_TYPE])
            if ot == ORGAN_ROOT_META:
                root_meta[pid] = t[idx]
                continue
            if ot == ORGAN_SHOOT_META:
                shoot_meta[(pid, sid)] = t[idx]
                continue
            pidx = _to_int(t[idx, T_COL_PHYTOMER_IDX])
            plants.setdefault(pid, {}).setdefault(sid, {}).setdefault(pidx, []).append(t[idx])

        for pid in sorted(plants.keys()):
            rm = root_meta.get(pid, torch.zeros(NUM_FEATURES_TYPED))
            bp_x = _fmt(_to_float(rm[T_COL_BASE_X]))
            bp_y = _fmt(_to_float(rm[T_COL_BASE_Y]))
            bp_z = _fmt(_to_float(rm[T_COL_BASE_Z]))
            pa = _fmt(_to_float(rm[T_COL_PLANT_AGE]))

            lines.append(f'\t<plant_instance ID="{pid}">')
            lines.append(f'\t\t<base_position> {bp_x} {bp_y} {bp_z} </base_position>')
            lines.append(f'\t\t<plant_age> {pa} </plant_age>')

            shoots = plants[pid]
            for sid in sorted(shoots.keys()):
                sm = shoot_meta.get((pid, sid), torch.zeros(NUM_FEATURES_TYPED))
                st = _to_int(sm[T_COL_SHOOT_TYPE])
                stl_str = "unifoliate" if st == 0 else "trifoliate"
                psi = _to_int(sm[T_COL_PARENT_SHOOT_ID])
                pni = _to_int(sm[T_COL_PARENT_NODE_IDX])
                ppi = _to_int(sm[T_COL_PARENT_PETIOLE_IDX])
                br_p = _fmt(_to_float(sm[T_COL_PITCH]))
                br_y = _fmt(_to_float(sm[T_COL_YAW]))
                br_r = _fmt(_to_float(sm[T_COL_ROLL]))

                lines.append(f'\t\t<shoot ID="{sid}">')
                lines.append(f'\t\t\t<shoot_type_label> {stl_str} </shoot_type_label>')
                lines.append(f'\t\t\t<parent_shoot_ID> {psi} </parent_shoot_ID>')
                lines.append(f'\t\t\t<parent_node_index> {pni} </parent_node_index>')
                lines.append(f'\t\t\t<parent_petiole_index> {ppi} </parent_petiole_index>')
                lines.append(f'\t\t\t<base_rotation> {br_p} {br_y} {br_r} </base_rotation>')

                phytomers = shoots[sid]
                for pidx in sorted(phytomers.keys()):
                    rows = phytomers[pidx]
                    lines.append('\t\t\t<phytomer>')
                    lines.append('\t\t\t\t<internode>')

                    internode = None
                    petioles: Dict[int, List[torch.Tensor]] = {}
                    buds: List[torch.Tensor] = []
                    peduncles: List[torch.Tensor] = []
                    flowers: List[torch.Tensor] = []

                    for row in rows:
                        ot = _to_int(row[T_COL_ORGAN_TYPE])
                        if ot == ORGAN_INTERNODE:
                            internode = row
                        elif ot == ORGAN_PETIOLE:
                            pet_i = _to_int(row[T_COL_PARENT_PETIOLE_IDX])
                            petioles.setdefault(pet_i, []).append(row)
                        elif ot == ORGAN_LEAF:
                            pet_i = _to_int(row[T_COL_PARENT_PETIOLE_IDX])
                            petioles.setdefault(pet_i, []).append(row)
                        elif ot == ORGAN_BUD:
                            pet_i = _to_int(row[T_COL_PARENT_PETIOLE_IDX])
                            petioles.setdefault(pet_i, []).append(row)
                        elif ot == ORGAN_PEDUNCLE:
                            pet_i = _to_int(row[T_COL_PARENT_PETIOLE_IDX])
                            petioles.setdefault(pet_i, []).append(row)
                        elif ot == ORGAN_FLOWER:
                            pet_i = _to_int(row[T_COL_PARENT_PETIOLE_IDX])
                            petioles.setdefault(pet_i, []).append(row)

                    # Internode fields
                    if internode is not None:
                        r = internode
                        lines.append(f'\t\t\t\t\t<internode_length>{_fmt(_to_float(r[T_COL_LENGTH]))}</internode_length>')
                        lines.append(f'\t\t\t\t\t<internode_radius>{_fmt(_to_float(r[T_COL_RADIUS]))}</internode_radius>')
                        lines.append(f'\t\t\t\t\t<internode_pitch>{_fmt(_to_float(r[T_COL_PITCH]))}</internode_pitch>')
                        lines.append(f'\t\t\t\t\t<internode_phyllotactic_angle>{_fmt(_to_float(r[T_COL_PHYLLOTACTIC_ANGLE]))}</internode_phyllotactic_angle>')
                        lines.append(f'\t\t\t\t\t<internode_length_max>{_fmt(_to_float(r[T_COL_LENGTH_MAX]))}</internode_length_max>')
                        lines.append(f'\t\t\t\t\t<internode_length_segments>{_to_int(r[T_COL_LENGTH_SEGMENTS])}</internode_length_segments>')
                        lines.append(f'\t\t\t\t\t<curvature_perturbations>{_fmt(_to_float(r[T_COL_CURV_PERT_0]))};{_fmt(_to_float(r[T_COL_CURV_PERT_1]))}</curvature_perturbations>')
                        lines.append(f'\t\t\t\t\t<yaw_perturbations>{_fmt(_to_float(r[T_COL_YAW_PERT_0]))};{_fmt(_to_float(r[T_COL_YAW_PERT_1]))}</yaw_perturbations>')

                    # Petioles in order 0, 1, ...
                    for pet_i in sorted(petioles.keys()):
                        pet_rows = petioles[pet_i]
                        pet = None
                        leaves = []
                        bud = None
                        peduncle = None
                        flowers = []
                        for pr in pet_rows:
                                ot = _to_int(pr[T_COL_ORGAN_TYPE])
                                if ot == ORGAN_PETIOLE:
                                    pet = pr
                                elif ot == ORGAN_LEAF:
                                    leaves.append(pr)
                                elif ot == ORGAN_BUD:
                                    bud = pr
                                elif ot == ORGAN_PEDUNCLE:
                                    peduncle = pr
                                elif ot == ORGAN_FLOWER:
                                    flowers.append(pr)

                        if pet is None:
                            continue
                        lines.append('\t\t\t\t\t<petiole>')
                        lines.append(f'\t\t\t\t\t\t<petiole_length>{_fmt(_to_float(pet[T_COL_LENGTH]))}</petiole_length>')
                        lines.append(f'\t\t\t\t\t\t<petiole_radius>{_fmt(_to_float(pet[T_COL_RADIUS]))}</petiole_radius>')
                        lines.append(f'\t\t\t\t\t\t<petiole_pitch>{_fmt(_to_float(pet[T_COL_PITCH]))}</petiole_pitch>')
                        lines.append(f'\t\t\t\t\t\t<petiole_curvature>{_fmt(_to_float(pet[T_COL_CURVATURE]))}</petiole_curvature>')
                        lines.append(f'\t\t\t\t\t\t<current_leaf_scale_factor>{_fmt(_to_float(pet[T_COL_CURRENT_LEAF_SCALE_FACTOR]))}</current_leaf_scale_factor>')
                        lines.append(f'\t\t\t\t\t\t<petiole_taper>{_fmt(_to_float(pet[T_COL_TAPER]))}</petiole_taper>')
                        lines.append(f'\t\t\t\t\t\t<petiole_length_segments>{_to_int(pet[T_COL_LENGTH_SEGMENTS])}</petiole_length_segments>')
                        lines.append(f'\t\t\t\t\t\t<petiole_radial_subdivisions>{_to_int(pet[T_COL_RADIAL_SUBDIVISIONS])}</petiole_radial_subdivisions>')
                        lines.append(f'\t\t\t\t\t\t<leaflet_scale>{_fmt(_to_float(pet[T_COL_LEAFLET_SCALE]))}</leaflet_scale>')
                        lines.append(f'\t\t\t\t\t\t<leaflet_offset>{_fmt(_to_float(pet[T_COL_LEAFLET_OFFSET]))}</leaflet_offset>')

                        leaves = sorted(leaves, key=lambda r: _to_int(r[T_COL_CHILD_INDEX]))
                        for lf in leaves:
                            lines.append('\t\t\t\t\t\t<leaf>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_scale>{_fmt(_to_float(lf[T_COL_SCALE]))}</leaf_scale>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_pitch>{_fmt(_to_float(lf[T_COL_PITCH]))}</leaf_pitch>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_yaw>{_fmt(_to_float(lf[T_COL_YAW]))}</leaf_yaw>')
                            lines.append(f'\t\t\t\t\t\t\t<leaf_roll>{_fmt(_to_float(lf[T_COL_ROLL]))}</leaf_roll>')
                            lines.append('\t\t\t\t\t\t</leaf>')

                        if bud is not None:
                            lines.append('\t\t\t\t\t\t<floral_bud>')
                            lines.append(f'\t\t\t\t\t\t\t<bud_state>{_to_int(bud[T_COL_BUD_STATE])}</bud_state>')
                            lines.append(f'\t\t\t\t\t\t\t<parent_index>{_to_int(bud[T_COL_BUD_PARENT_INDEX])}</parent_index>')
                            lines.append(f'\t\t\t\t\t\t\t<bud_index>{_to_int(bud[T_COL_CHILD_INDEX])}</bud_index>')
                            lines.append(f'\t\t\t\t\t\t\t<is_terminal>{_to_int(bud[T_COL_BUD_IS_TERMINAL])}</is_terminal>')
                            lines.append(f'\t\t\t\t\t\t\t<current_fruit_scale_factor>{_fmt(_to_float(bud[T_COL_FRUIT_SCALE]))}</current_fruit_scale_factor>')

                            if peduncle is not None:
                                lines.append('\t\t\t\t\t\t\t<peduncle>')
                                lines.append(f'\t\t\t\t\t\t\t\t<length>{_fmt(_to_float(peduncle[T_COL_LENGTH]))}</length>')
                                lines.append(f'\t\t\t\t\t\t\t\t<radius>{_fmt(_to_float(peduncle[T_COL_RADIUS]))}</radius>')
                                lines.append(f'\t\t\t\t\t\t\t\t<pitch>{_fmt(_to_float(peduncle[T_COL_PITCH]))}</pitch>')
                                lines.append(f'\t\t\t\t\t\t\t\t<curvature>{_fmt(_to_float(peduncle[T_COL_CURVATURE]))}</curvature>')
                                lines.append(f'\t\t\t\t\t\t\t\t<roll>{_fmt(_to_float(peduncle[T_COL_ROLL]))}</roll>')
                                lines.append('\t\t\t\t\t\t\t</peduncle>')

                            if flowers or (bud is not None and _to_float(bud[T_COL_FLOWER_OFFSET]) != 0.0):
                                lines.append('\t\t\t\t\t\t\t<inflorescence>')
                                if flowers:
                                    foff = _fmt(_to_float(flowers[0][T_COL_FLOWER_OFFSET]))
                                else:
                                    foff = _fmt(_to_float(bud[T_COL_FLOWER_OFFSET]))
                                lines.append(f'\t\t\t\t\t\t\t\t<flower_offset>{foff}</flower_offset>')
                                flowers = sorted(flowers, key=lambda r: _to_int(r[T_COL_CHILD_INDEX]))
                                for fl in flowers:
                                    lines.append('\t\t\t\t\t\t\t\t<flower>')
                                    lines.append(f'\t\t\t\t\t\t\t\t\t<flower_pitch>{_fmt(_to_float(fl[T_COL_PITCH]))}</flower_pitch>')
                                    lines.append(f'\t\t\t\t\t\t\t\t\t<flower_yaw>{_fmt(_to_float(fl[T_COL_YAW]))}</flower_yaw>')
                                    lines.append(f'\t\t\t\t\t\t\t\t\t<flower_roll>{_fmt(_to_float(fl[T_COL_ROLL]))}</flower_roll>')
                                    lines.append(f'\t\t\t\t\t\t\t\t\t<flower_azimuth>{_fmt(_to_float(fl[T_COL_FLOWER_AZIMUTH]))}</flower_azimuth>')
                                    lines.append(f'\t\t\t\t\t\t\t\t\t<flower_base_scale>{_fmt(_to_float(fl[T_COL_SCALE]))}</flower_base_scale>')
                                    lines.append('\t\t\t\t\t\t\t\t</flower>')
                                lines.append('\t\t\t\t\t\t\t</inflorescence>')

                            lines.append('\t\t\t\t\t\t</floral_bud>')

                        lines.append('\t\t\t\t\t</petiole>')

                    lines.append('\t\t\t\t</internode>')
                    lines.append('\t\t\t</phytomer>')

                lines.append('\t\t</shoot>')

            lines.append('\t</plant_instance>')

        lines.append('</helios>')
        return "\n".join(lines) + "\n"

    # -------------------------------------------------------------------------
    # LEGACY 94D XML PARSER
    # -------------------------------------------------------------------------
    @classmethod
    def _from_xml_string_legacy(cls, xml_content: str) -> "PlantOrganArray":
        """DEPRECATED: legacy (N, 94) parser."""
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
                    row = [0.0] * NUM_FEATURES_LEGACY
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
                            if pet_i > 1:
                                break
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

                            row[base_col] = float(meta[prefix + "l"].strip())
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
                                if pet_i == 0 and lf_idx >= 3:
                                    break
                                if pet_i == 1 and lf_idx >= 1:
                                    break
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
                                row[cur_col] = float(raw_lfs.strip())
                                row[cur_col + 1] = float(raw_lfp.strip())
                                row[cur_col + 2] = float(raw_lfy.strip())
                                row[cur_col + 3] = float(raw_lfr.strip())

                            meta[prefix + "leaves"] = leaf_metas

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
                                            if fl_idx >= 4:
                                                break
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
                                            row[fl_base_col] = float(raw_fp.strip())
                                            row[fl_base_col + 1] = float(raw_fy.strip())
                                            row[fl_base_col + 2] = float(raw_fr.strip())
                                            row[fl_base_col + 3] = float(raw_fa.strip())
                                            row[fl_base_col + 4] = float(raw_fbs.strip())

                                        meta["flowers"] = fl_metas

                    rows.append(row)
                    raw_metadata.append(meta)

        tensor = torch.tensor(rows, dtype=torch.float32)
        tensor[:, COL_EXISTENCE] = 1.0
        return cls(tensor=tensor, raw_metadata=raw_metadata)

    # -------------------------------------------------------------------------
    # TYPED 40D XML PARSER (honest round-trip)
    # -------------------------------------------------------------------------
    @classmethod
    def _from_xml_string_typed(cls, xml_content: str) -> "PlantOrganArray":
        """TYPED (N, 40) parser. Produces per-organ rows."""
        root = ET.fromstring(xml_content)
        if root.tag != "helios":
            raise ValueError("Root tag must be <helios>")

        rows: List[List[float]] = []

        for plant_elem in root.findall("plant_instance"):
            plant_id = int(plant_elem.attrib.get("ID", 0))

            bp_elem = plant_elem.find("base_position")
            raw_bp = bp_elem.text if bp_elem is not None else " 0 0 0 "
            bp_vals = [float(x) for x in raw_bp.strip().split()]
            bp = (bp_vals[0], bp_vals[1], bp_vals[2]) if len(bp_vals) >= 3 else (0.0, 0.0, 0.0)

            age_elem = plant_elem.find("plant_age")
            plant_age = float(age_elem.text.strip()) if age_elem is not None and age_elem.text else 0.0

            # ROOT_META row
            root_row = [0.0] * NUM_FEATURES_TYPED
            root_row[T_COL_PLANT_ID] = float(plant_id)
            root_row[T_COL_PLANT_AGE] = plant_age
            root_row[T_COL_BASE_X] = bp[0]
            root_row[T_COL_BASE_Y] = bp[1]
            root_row[T_COL_BASE_Z] = bp[2]
            root_row[T_COL_ORGAN_TYPE] = float(ORGAN_ROOT_META)
            root_row[T_COL_EXISTENCE] = 1.0
            rows.append(root_row)

            for shoot_elem in plant_elem.findall("shoot"):
                shoot_id = int(shoot_elem.attrib.get("ID", 0))

                stl_elem = shoot_elem.find("shoot_type_label")
                raw_stl = stl_elem.text if stl_elem is not None else "unifoliate"
                shoot_type = 0 if "unifoliate" in raw_stl else 1

                psi_elem = shoot_elem.find("parent_shoot_ID")
                psi = int(psi_elem.text.strip()) if psi_elem is not None and psi_elem.text else -1

                pni_elem = shoot_elem.find("parent_node_index")
                pni = int(pni_elem.text.strip()) if pni_elem is not None and pni_elem.text else 0

                ppi_elem = shoot_elem.find("parent_petiole_index")
                ppi = int(ppi_elem.text.strip()) if ppi_elem is not None and ppi_elem.text else 0

                br_elem = shoot_elem.find("base_rotation")
                raw_br = br_elem.text if br_elem is not None else " 0 0 0 "
                br_vals = [float(x) for x in raw_br.strip().split()]
                br = (br_vals[0], br_vals[1], br_vals[2]) if len(br_vals) >= 3 else (0.0, 0.0, 0.0)

                # SHOOT_META row
                shoot_row = [0.0] * NUM_FEATURES_TYPED
                shoot_row[T_COL_PLANT_ID] = float(plant_id)
                shoot_row[T_COL_SHOOT_ID] = float(shoot_id)
                shoot_row[T_COL_PARENT_SHOOT_ID] = float(psi)
                shoot_row[T_COL_PARENT_NODE_IDX] = float(pni)
                shoot_row[T_COL_PARENT_PETIOLE_IDX] = float(ppi)
                shoot_row[T_COL_ORGAN_TYPE] = float(ORGAN_SHOOT_META)
                shoot_row[T_COL_SHOOT_TYPE] = float(shoot_type)
                shoot_row[T_COL_PITCH] = br[0]
                shoot_row[T_COL_YAW] = br[1]
                shoot_row[T_COL_ROLL] = br[2]
                shoot_row[T_COL_EXISTENCE] = 1.0
                rows.append(shoot_row)

                for phyto_idx, phyto_elem in enumerate(shoot_elem.findall("phytomer")):
                    internode_elem = phyto_elem.find("internode")
                    if internode_elem is None:
                        continue

                    # Internode row
                    il = _get_float_text(internode_elem, "internode_length", 0.0)
                    ir = _get_float_text(internode_elem, "internode_radius", 0.0)
                    ip = _get_float_text(internode_elem, "internode_pitch", 0.0)
                    ipa = _get_float_text(internode_elem, "internode_phyllotactic_angle", 0.0)
                    ilm = _get_float_text(internode_elem, "internode_length_max", 0.0)
                    ils = _get_int_text(internode_elem, "internode_length_segments", 2)
                    cp_text = _get_text_default(internode_elem, "curvature_perturbations", "0;0")
                    cp_list = [float(x) for x in cp_text.strip().split(";") if x.strip()]
                    cp0 = cp_list[0] if len(cp_list) > 0 else 0.0
                    cp1 = cp_list[1] if len(cp_list) > 1 else 0.0
                    yp_text = _get_text_default(internode_elem, "yaw_perturbations", "0;0")
                    yp_list = [float(x) for x in yp_text.strip().split(";") if x.strip()]
                    yp0 = yp_list[0] if len(yp_list) > 0 else 0.0
                    yp1 = yp_list[1] if len(yp_list) > 1 else 0.0

                    inode_row = [0.0] * NUM_FEATURES_TYPED
                    inode_row[T_COL_PLANT_ID] = float(plant_id)
                    inode_row[T_COL_PLANT_AGE] = plant_age
                    inode_row[T_COL_SHOOT_ID] = float(shoot_id)
                    inode_row[T_COL_PHYTOMER_IDX] = float(phyto_idx)
                    inode_row[T_COL_ORGAN_TYPE] = float(ORGAN_INTERNODE)
                    inode_row[T_COL_LENGTH] = il
                    inode_row[T_COL_RADIUS] = ir
                    inode_row[T_COL_PITCH] = ip
                    inode_row[T_COL_PHYLLOTACTIC_ANGLE] = ipa
                    inode_row[T_COL_LENGTH_MAX] = ilm
                    inode_row[T_COL_LENGTH_SEGMENTS] = float(ils)
                    inode_row[T_COL_CURV_PERT_0] = cp0
                    inode_row[T_COL_CURV_PERT_1] = cp1
                    inode_row[T_COL_YAW_PERT_0] = yp0
                    inode_row[T_COL_YAW_PERT_1] = yp1
                    inode_row[T_COL_EXISTENCE] = 1.0
                    rows.append(inode_row)

                    # Petioles, leaves, buds
                    for pet_i, pet_elem in enumerate(internode_elem.findall("petiole")):
                        pl = _get_float_text(pet_elem, "petiole_length", 0.0)
                        pr = _get_float_text(pet_elem, "petiole_radius", 0.0)
                        pp = _get_float_text(pet_elem, "petiole_pitch", 0.0)
                        pc = _get_float_text(pet_elem, "petiole_curvature", 0.0)
                        cls_val = _get_float_text(pet_elem, "current_leaf_scale_factor", 1.0)
                        pt = _get_float_text(pet_elem, "petiole_taper", 0.25)
                        pls = _get_int_text(pet_elem, "petiole_length_segments", 5)
                        prs = _get_int_text(pet_elem, "petiole_radial_subdivisions", 6)
                        lfls = _get_float_text(pet_elem, "leaflet_scale", 1.0)
                        lflo = _get_float_text(pet_elem, "leaflet_offset", 0.4)

                        pet_row = [0.0] * NUM_FEATURES_TYPED
                        pet_row[T_COL_PLANT_ID] = float(plant_id)
                        pet_row[T_COL_SHOOT_ID] = float(shoot_id)
                        pet_row[T_COL_PHYTOMER_IDX] = float(phyto_idx)
                        pet_row[T_COL_PARENT_PETIOLE_IDX] = float(pet_i)
                        pet_row[T_COL_ORGAN_TYPE] = float(ORGAN_PETIOLE)
                        pet_row[T_COL_LENGTH] = pl
                        pet_row[T_COL_RADIUS] = pr
                        pet_row[T_COL_PITCH] = pp
                        pet_row[T_COL_CURVATURE] = pc
                        pet_row[T_COL_CURRENT_LEAF_SCALE_FACTOR] = cls_val
                        pet_row[T_COL_TAPER] = pt
                        pet_row[T_COL_LENGTH_SEGMENTS] = float(pls)
                        pet_row[T_COL_RADIAL_SUBDIVISIONS] = float(prs)
                        pet_row[T_COL_LEAFLET_SCALE] = lfls
                        pet_row[T_COL_LEAFLET_OFFSET] = lflo
                        pet_row[T_COL_EXISTENCE] = 1.0
                        rows.append(pet_row)

                        leaf_elems = pet_elem.findall("leaf")
                        for lf_idx, leaf_elem in enumerate(leaf_elems):
                            lfs = _get_float_text(leaf_elem, "leaf_scale", 1.0)
                            lfp = _get_float_text(leaf_elem, "leaf_pitch", 0.0)
                            lfy = _get_float_text(leaf_elem, "leaf_yaw", 0.0)
                            lfr = _get_float_text(leaf_elem, "leaf_roll", 0.0)

                            leaf_row = [0.0] * NUM_FEATURES_TYPED
                            leaf_row[T_COL_PLANT_ID] = float(plant_id)
                            leaf_row[T_COL_SHOOT_ID] = float(shoot_id)
                            leaf_row[T_COL_PHYTOMER_IDX] = float(phyto_idx)
                            leaf_row[T_COL_PARENT_PETIOLE_IDX] = float(pet_i)
                            leaf_row[T_COL_CHILD_INDEX] = float(lf_idx)
                            leaf_row[T_COL_ORGAN_TYPE] = float(ORGAN_LEAF)
                            leaf_row[T_COL_SCALE] = lfs
                            leaf_row[T_COL_PITCH] = lfp
                            leaf_row[T_COL_YAW] = lfy
                            leaf_row[T_COL_ROLL] = lfr
                            leaf_row[T_COL_EXISTENCE] = 1.0
                            rows.append(leaf_row)

                        # Floral bud is only on petiole 0
                        if pet_i == 0:
                            fb_elem = pet_elem.find("floral_bud")
                            if fb_elem is not None:
                                bs = _get_int_text(fb_elem, "bud_state", 5)
                                bpi = _get_int_text(fb_elem, "parent_index", 0)
                                bidx = _get_int_text(fb_elem, "bud_index", 0)
                                biterm = _get_int_text(fb_elem, "is_terminal", 0)
                                bcfs = _get_float_text(fb_elem, "current_fruit_scale_factor", 1.0)

                                bud_row = [0.0] * NUM_FEATURES_TYPED
                                bud_row[T_COL_PLANT_ID] = float(plant_id)
                                bud_row[T_COL_SHOOT_ID] = float(shoot_id)
                                bud_row[T_COL_PHYTOMER_IDX] = float(phyto_idx)
                                bud_row[T_COL_PARENT_PETIOLE_IDX] = float(pet_i)
                                bud_row[T_COL_CHILD_INDEX] = float(bidx)
                                bud_row[T_COL_ORGAN_TYPE] = float(ORGAN_BUD)
                                bud_row[T_COL_BUD_STATE] = float(bs)
                                bud_row[T_COL_BUD_PARENT_INDEX] = float(bpi)
                                bud_row[T_COL_BUD_IS_TERMINAL] = float(biterm)
                                bud_row[T_COL_FRUIT_SCALE] = bcfs
                                bud_row[T_COL_EXISTENCE] = 1.0
                                rows.append(bud_row)

                                ped_elem = fb_elem.find("peduncle")
                                if ped_elem is not None:
                                    pdl = _get_float_text(ped_elem, "length", 0.0)
                                    pdr = _get_float_text(ped_elem, "radius", 0.0)
                                    pdp = _get_float_text(ped_elem, "pitch", 0.0)
                                    pdc = _get_float_text(ped_elem, "curvature", 0.0)
                                    pdrl = _get_float_text(ped_elem, "roll", 0.0)

                                    ped_row = [0.0] * NUM_FEATURES_TYPED
                                    ped_row[T_COL_PLANT_ID] = float(plant_id)
                                    ped_row[T_COL_SHOOT_ID] = float(shoot_id)
                                    ped_row[T_COL_PHYTOMER_IDX] = float(phyto_idx)
                                    ped_row[T_COL_PARENT_PETIOLE_IDX] = float(pet_i)
                                    ped_row[T_COL_ORGAN_TYPE] = float(ORGAN_PEDUNCLE)
                                    ped_row[T_COL_LENGTH] = pdl
                                    ped_row[T_COL_RADIUS] = pdr
                                    ped_row[T_COL_PITCH] = pdp
                                    ped_row[T_COL_CURVATURE] = pdc
                                    ped_row[T_COL_ROLL] = pdrl
                                    ped_row[T_COL_EXISTENCE] = 1.0
                                    rows.append(ped_row)

                                infl_elem = fb_elem.find("inflorescence")
                                if infl_elem is not None:
                                    foff = _get_float_text(infl_elem, "flower_offset", 0.05)
                                    # Store flower_offset on the bud row so it survives even when
                                    # the inflorescence contains no <flower> tags.
                                    if bud_row is not None:
                                        bud_row[T_COL_FLOWER_OFFSET] = foff
                                    flower_elems = infl_elem.findall("flower")
                                    for fl_idx, fl_elem in enumerate(flower_elems):
                                        fp = _get_float_text(fl_elem, "flower_pitch", 0.0)
                                        fy = _get_float_text(fl_elem, "flower_yaw", 0.0)
                                        fr = _get_float_text(fl_elem, "flower_roll", 0.0)
                                        fa = _get_float_text(fl_elem, "flower_azimuth", 0.0)
                                        fbs = _get_float_text(fl_elem, "flower_base_scale", 1.0)

                                        fl_row = [0.0] * NUM_FEATURES_TYPED
                                        fl_row[T_COL_PLANT_ID] = float(plant_id)
                                        fl_row[T_COL_SHOOT_ID] = float(shoot_id)
                                        fl_row[T_COL_PHYTOMER_IDX] = float(phyto_idx)
                                        fl_row[T_COL_PARENT_PETIOLE_IDX] = float(pet_i)
                                        fl_row[T_COL_CHILD_INDEX] = float(fl_idx)
                                        fl_row[T_COL_ORGAN_TYPE] = float(ORGAN_FLOWER)
                                        fl_row[T_COL_PITCH] = fp
                                        fl_row[T_COL_YAW] = fy
                                        fl_row[T_COL_ROLL] = fr
                                        fl_row[T_COL_FLOWER_AZIMUTH] = fa
                                        fl_row[T_COL_SCALE] = fbs
                                        fl_row[T_COL_FLOWER_OFFSET] = foff
                                        fl_row[T_COL_EXISTENCE] = 1.0
                                        rows.append(fl_row)

        tensor = torch.tensor(rows, dtype=torch.float32)
        return cls(tensor=tensor, raw_metadata=[])

    # -------------------------------------------------------------------------
    # CONVERSIONS BETWEEN LEGACY AND TYPED
    # -------------------------------------------------------------------------
    def to_legacy_tensor(self) -> torch.Tensor:
        """Convert a typed (N, 40) tensor to legacy (M, 94) phytomer-slot tensor.

        This is a lossy grouping operation: per-organ rows are grouped back into
        phytomer slots. It is provided only for compatibility with code that has
        not yet migrated to the typed layout.
        """
        if self.is_legacy:
            return self.tensor.clone()

        t = self.tensor
        N = self.num_nodes

        # Group by (plant_id, shoot_id, phytomer_idx)
        phytomers: Dict[Tuple[int, int, int], List[torch.Tensor]] = {}
        shoot_meta_rows: Dict[Tuple[int, int], torch.Tensor] = {}
        root_meta_rows: Dict[int, torch.Tensor] = {}

        for idx in range(N):
            pid = _to_int(t[idx, T_COL_PLANT_ID])
            sid = _to_int(t[idx, T_COL_SHOOT_ID])
            pidx = _to_int(t[idx, T_COL_PHYTOMER_IDX])
            ot = _to_int(t[idx, T_COL_ORGAN_TYPE])
            if ot == ORGAN_ROOT_META:
                root_meta_rows[pid] = t[idx]
            elif ot == ORGAN_SHOOT_META:
                shoot_meta_rows[(pid, sid)] = t[idx]
            else:
                phytomers.setdefault((pid, sid, pidx), []).append(t[idx])

        rows = []
        raw_metadata = []

        for (pid, sid, pidx), organ_rows in sorted(phytomers.items()):
            row = [0.0] * NUM_FEATURES_LEGACY
            meta: Dict[str, Any] = {}

            root = root_meta_rows.get(pid, torch.zeros(NUM_FEATURES_TYPED))
            shoot = shoot_meta_rows.get((pid, sid), torch.zeros(NUM_FEATURES_TYPED))

            row[COL_PLANT_ID] = float(pid)
            row[COL_PLANT_AGE] = _to_float(root[T_COL_PLANT_AGE])
            row[COL_SHOOT_ID] = float(sid)
            row[COL_SHOOT_TYPE] = _to_float(shoot[T_COL_SHOOT_TYPE])
            row[COL_PARENT_SHOOT_ID] = _to_float(shoot[T_COL_PARENT_SHOOT_ID])
            row[COL_PARENT_NODE_IDX] = _to_float(shoot[T_COL_PARENT_NODE_IDX])
            row[COL_PARENT_PETIOLE_IDX] = _to_float(shoot[T_COL_PARENT_PETIOLE_IDX])
            row[COL_SHOOT_ROT_PITCH] = _to_float(shoot[T_COL_PITCH])
            row[COL_SHOOT_ROT_YAW] = _to_float(shoot[T_COL_YAW])
            row[COL_SHOOT_ROT_ROLL] = _to_float(shoot[T_COL_ROLL])
            row[COL_PHYTOMER_IDX] = float(pidx)

            meta["raw_bp"] = f" {_fmt(_to_float(root[T_COL_BASE_X]))} {_fmt(_to_float(root[T_COL_BASE_Y]))} {_fmt(_to_float(root[T_COL_BASE_Z]))} "
            meta["raw_pa"] = _fmt(_to_float(root[T_COL_PLANT_AGE]))
            meta["raw_stl"] = "unifoliate" if _to_int(shoot[T_COL_SHOOT_TYPE]) == 0 else "trifoliate"
            meta["raw_psi"] = str(_to_int(shoot[T_COL_PARENT_SHOOT_ID]))
            meta["raw_pni"] = str(_to_int(shoot[T_COL_PARENT_NODE_IDX]))
            meta["raw_ppi"] = str(_to_int(shoot[T_COL_PARENT_PETIOLE_IDX]))
            meta["raw_br"] = f" {_fmt(_to_float(shoot[T_COL_PITCH]))} {_fmt(_to_float(shoot[T_COL_YAW]))} {_fmt(_to_float(shoot[T_COL_ROLL]))} "

            for r in organ_rows:
                ot = _to_int(r[T_COL_ORGAN_TYPE])
                if ot == ORGAN_INTERNODE:
                    row[COL_INODE_LEN] = _to_float(r[T_COL_LENGTH])
                    row[COL_INODE_RAD] = _to_float(r[T_COL_RADIUS])
                    row[COL_INODE_PITCH] = _to_float(r[T_COL_PITCH])
                    row[COL_INODE_PHYLLO_ANG] = _to_float(r[T_COL_PHYLLOTACTIC_ANGLE])
                    row[COL_INODE_LEN_MAX] = _to_float(r[T_COL_LENGTH_MAX])
                    row[COL_INODE_LEN_SEGS] = _to_float(r[T_COL_LENGTH_SEGMENTS])
                    row[COL_CURV_PERT_0] = _to_float(r[T_COL_CURV_PERT_0])
                    row[COL_CURV_PERT_1] = _to_float(r[T_COL_CURV_PERT_1])
                    row[COL_YAW_PERT_0] = _to_float(r[T_COL_YAW_PERT_0])
                    row[COL_YAW_PERT_1] = _to_float(r[T_COL_YAW_PERT_1])
                elif ot == ORGAN_PETIOLE:
                    pet_i = _to_int(r[T_COL_PARENT_PETIOLE_IDX])
                    base_col = COL_PET0_LEN if pet_i == 0 else COL_PET1_LEN
                    row[base_col] = _to_float(r[T_COL_LENGTH])
                    row[base_col + 1] = _to_float(r[T_COL_RADIUS])
                    row[base_col + 2] = _to_float(r[T_COL_PITCH])
                    row[base_col + 3] = _to_float(r[T_COL_CURVATURE])
                    row[base_col + 4] = _to_float(r[T_COL_CURRENT_LEAF_SCALE_FACTOR])
                    row[base_col + 5] = _to_float(r[T_COL_TAPER])
                    row[base_col + 6] = _to_float(r[T_COL_LENGTH_SEGMENTS])
                    row[base_col + 7] = _to_float(r[T_COL_RADIAL_SUBDIVISIONS])
                    row[base_col + 8] = _to_float(r[T_COL_LEAFLET_SCALE])
                    row[base_col + 9] = _to_float(r[T_COL_LEAFLET_OFFSET])
                    if pet_i == 1:
                        row[COL_HAS_PET1] = 1.0
                elif ot == ORGAN_LEAF:
                    pet_i = _to_int(r[T_COL_PARENT_PETIOLE_IDX])
                    lf_idx = _to_int(r[T_COL_CHILD_INDEX])
                    leaf_base_col = COL_PET0_L0_SCALE if pet_i == 0 else COL_PET1_L0_SCALE
                    cur_col = leaf_base_col + lf_idx * 4
                    row[cur_col] = _to_float(r[T_COL_SCALE])
                    row[cur_col + 1] = _to_float(r[T_COL_PITCH])
                    row[cur_col + 2] = _to_float(r[T_COL_YAW])
                    row[cur_col + 3] = _to_float(r[T_COL_ROLL])
                    # Update num_leaves
                    base_col = COL_PET0_LEN if pet_i == 0 else COL_PET1_LEN
                    row[base_col + 10] = max(row[base_col + 10], float(lf_idx + 1))
                elif ot == ORGAN_BUD:
                    row[COL_HAS_BUD] = 1.0
                    row[COL_BUD_STATE] = _to_float(r[T_COL_BUD_STATE])
                    row[COL_BUD_PARENT_IDX] = _to_float(r[T_COL_BUD_PARENT_INDEX])
                    row[COL_BUD_IDX] = _to_float(r[T_COL_CHILD_INDEX])
                    row[COL_BUD_IS_TERMINAL] = _to_float(r[T_COL_BUD_IS_TERMINAL])
                    row[COL_BUD_FRUIT_SCALE] = _to_float(r[T_COL_FRUIT_SCALE])
                elif ot == ORGAN_PEDUNCLE:
                    row[COL_PED_LEN] = _to_float(r[T_COL_LENGTH])
                    row[COL_PED_RAD] = _to_float(r[T_COL_RADIUS])
                    row[COL_PED_PITCH] = _to_float(r[T_COL_PITCH])
                    row[COL_PED_CURV] = _to_float(r[T_COL_CURVATURE])
                    row[COL_PED_ROLL] = _to_float(r[T_COL_ROLL])
                elif ot == ORGAN_FLOWER:
                    fl_idx = _to_int(r[T_COL_CHILD_INDEX])
                    fl_base_col = COL_FL0_PITCH + fl_idx * 5
                    row[fl_base_col] = _to_float(r[T_COL_PITCH])
                    row[fl_base_col + 1] = _to_float(r[T_COL_YAW])
                    row[fl_base_col + 2] = _to_float(r[T_COL_ROLL])
                    row[fl_base_col + 3] = _to_float(r[T_COL_FLOWER_AZIMUTH])
                    row[fl_base_col + 4] = _to_float(r[T_COL_SCALE])
                    row[COL_NUM_FLOWERS] = max(row[COL_NUM_FLOWERS], float(fl_idx + 1))
                    row[COL_FLOWER_OFFSET] = _to_float(r[T_COL_FLOWER_OFFSET])

            rows.append(row)
            raw_metadata.append(meta)

        legacy_tensor = torch.tensor(rows, dtype=torch.float32)
        legacy_tensor[:, COL_EXISTENCE] = 1.0
        return legacy_tensor

    def to_legacy_tensor_diff(self) -> torch.Tensor:
        """Differentiable typed->legacy conversion.

        Equivalent values to :meth:`to_legacy_tensor` but preserves the
        autograd graph: every legacy cell that comes from a typed parameter is
        produced with advanced-indexing gather from the typed tensor, so
        gradients flow back to the continuous columns. Cells that are pure
        constants (existence flags, petiole/leaf/flower counts, ...) are baked
        as non-differentiable constants. Returns a legacy (M, 94) tensor.
        """
        t = self.tensor
        N = self.num_nodes

        # Group by (plant_id, shoot_id, phytomer_idx) -- same logic as
        # to_legacy_tensor, but record source (row, col) references instead of
        # materializing float values.
        phytomers: Dict[Tuple[int, int, int], List[int]] = {}
        shoot_meta_rows: Dict[Tuple[int, int], int] = {}
        root_meta_rows: Dict[int, int] = {}

        for idx in range(N):
            pid = _to_int(t[idx, T_COL_PLANT_ID])
            sid = _to_int(t[idx, T_COL_SHOOT_ID])
            pidx = _to_int(t[idx, T_COL_PHYTOMER_IDX])
            ot = _to_int(t[idx, T_COL_ORGAN_TYPE])
            if ot == ORGAN_ROOT_META:
                root_meta_rows[pid] = idx
            elif ot == ORGAN_SHOOT_META:
                shoot_meta_rows[(pid, sid)] = idx
            else:
                phytomers.setdefault((pid, sid, pidx), []).append(idx)

        M = len(phytomers)
        pad = N  # index of the zero-padding row appended to t

        src_row = np.full((M, NUM_FEATURES_LEGACY), pad, dtype=np.int64)
        src_col = np.zeros((M, NUM_FEATURES_LEGACY), dtype=np.int64)
        use_gather = np.zeros((M, NUM_FEATURES_LEGACY), dtype=bool)
        const_val = np.zeros((M, NUM_FEATURES_LEGACY), dtype=np.float32)

        def set_cell(L: int, col: int, row_idx: int, t_col: int) -> None:
            src_row[L, col] = row_idx
            src_col[L, col] = t_col
            use_gather[L, col] = True

        for row_i, (key, organ_indices) in enumerate(sorted(phytomers.items())):
            (pid, sid, pidx) = key
            root_i = root_meta_rows.get(pid, pad)
            shoot_i = shoot_meta_rows.get((pid, sid), pad)
            first = organ_indices[0]

            set_cell(row_i, COL_PLANT_ID, first, T_COL_PLANT_ID)
            set_cell(row_i, COL_PLANT_AGE, root_i, T_COL_PLANT_AGE)
            set_cell(row_i, COL_SHOOT_ID, first, T_COL_SHOOT_ID)
            set_cell(row_i, COL_SHOOT_TYPE, shoot_i, T_COL_SHOOT_TYPE)
            set_cell(row_i, COL_PARENT_SHOOT_ID, shoot_i, T_COL_PARENT_SHOOT_ID)
            set_cell(row_i, COL_PARENT_NODE_IDX, shoot_i, T_COL_PARENT_NODE_IDX)
            set_cell(row_i, COL_PARENT_PETIOLE_IDX, shoot_i, T_COL_PARENT_PETIOLE_IDX)
            set_cell(row_i, COL_SHOOT_ROT_PITCH, shoot_i, T_COL_PITCH)
            set_cell(row_i, COL_SHOOT_ROT_YAW, shoot_i, T_COL_YAW)
            set_cell(row_i, COL_SHOOT_ROT_ROLL, shoot_i, T_COL_ROLL)
            set_cell(row_i, COL_PHYTOMER_IDX, first, T_COL_PHYTOMER_IDX)

            for idx in organ_indices:
                ot = _to_int(t[idx, T_COL_ORGAN_TYPE])
                if ot == ORGAN_INTERNODE:
                    set_cell(row_i, COL_INODE_LEN, idx, T_COL_LENGTH)
                    set_cell(row_i, COL_INODE_RAD, idx, T_COL_RADIUS)
                    set_cell(row_i, COL_INODE_PITCH, idx, T_COL_PITCH)
                    set_cell(row_i, COL_INODE_PHYLLO_ANG, idx, T_COL_PHYLLOTACTIC_ANGLE)
                    set_cell(row_i, COL_INODE_LEN_MAX, idx, T_COL_LENGTH_MAX)
                    set_cell(row_i, COL_INODE_LEN_SEGS, idx, T_COL_LENGTH_SEGMENTS)
                    set_cell(row_i, COL_CURV_PERT_0, idx, T_COL_CURV_PERT_0)
                    set_cell(row_i, COL_CURV_PERT_1, idx, T_COL_CURV_PERT_1)
                    set_cell(row_i, COL_YAW_PERT_0, idx, T_COL_YAW_PERT_0)
                    set_cell(row_i, COL_YAW_PERT_1, idx, T_COL_YAW_PERT_1)
                elif ot == ORGAN_PETIOLE:
                    pet_i = _to_int(t[idx, T_COL_PARENT_PETIOLE_IDX])
                    base_col = COL_PET0_LEN if pet_i == 0 else COL_PET1_LEN
                    set_cell(row_i, base_col, idx, T_COL_LENGTH)
                    set_cell(row_i, base_col + 1, idx, T_COL_RADIUS)
                    set_cell(row_i, base_col + 2, idx, T_COL_PITCH)
                    set_cell(row_i, base_col + 3, idx, T_COL_CURVATURE)
                    set_cell(row_i, base_col + 4, idx, T_COL_CURRENT_LEAF_SCALE_FACTOR)
                    set_cell(row_i, base_col + 5, idx, T_COL_TAPER)
                    set_cell(row_i, base_col + 6, idx, T_COL_LENGTH_SEGMENTS)
                    set_cell(row_i, base_col + 7, idx, T_COL_RADIAL_SUBDIVISIONS)
                    set_cell(row_i, base_col + 8, idx, T_COL_LEAFLET_SCALE)
                    set_cell(row_i, base_col + 9, idx, T_COL_LEAFLET_OFFSET)
                    if pet_i == 1:
                        const_val[row_i, COL_HAS_PET1] = 1.0
                elif ot == ORGAN_LEAF:
                    pet_i = _to_int(t[idx, T_COL_PARENT_PETIOLE_IDX])
                    lf_idx = _to_int(t[idx, T_COL_CHILD_INDEX])
                    leaf_base_col = COL_PET0_L0_SCALE if pet_i == 0 else COL_PET1_L0_SCALE
                    cur_col = leaf_base_col + lf_idx * 4
                    if cur_col + 3 < NUM_FEATURES_LEGACY:
                        set_cell(row_i, cur_col, idx, T_COL_SCALE)
                        set_cell(row_i, cur_col + 1, idx, T_COL_PITCH)
                        set_cell(row_i, cur_col + 2, idx, T_COL_YAW)
                        set_cell(row_i, cur_col + 3, idx, T_COL_ROLL)
                    base_col = COL_PET0_LEN if pet_i == 0 else COL_PET1_LEN
                    if base_col + 10 < NUM_FEATURES_LEGACY:
                        const_val[row_i, base_col + 10] = max(
                            const_val[row_i, base_col + 10], float(lf_idx + 1)
                        )
                elif ot == ORGAN_BUD:
                    const_val[row_i, COL_HAS_BUD] = 1.0
                    set_cell(row_i, COL_BUD_STATE, idx, T_COL_BUD_STATE)
                    set_cell(row_i, COL_BUD_PARENT_IDX, idx, T_COL_BUD_PARENT_INDEX)
                    set_cell(row_i, COL_BUD_IDX, idx, T_COL_CHILD_INDEX)
                    set_cell(row_i, COL_BUD_IS_TERMINAL, idx, T_COL_BUD_IS_TERMINAL)
                    set_cell(row_i, COL_BUD_FRUIT_SCALE, idx, T_COL_FRUIT_SCALE)
                elif ot == ORGAN_PEDUNCLE:
                    set_cell(row_i, COL_PED_LEN, idx, T_COL_LENGTH)
                    set_cell(row_i, COL_PED_RAD, idx, T_COL_RADIUS)
                    set_cell(row_i, COL_PED_PITCH, idx, T_COL_PITCH)
                    set_cell(row_i, COL_PED_CURV, idx, T_COL_CURVATURE)
                    set_cell(row_i, COL_PED_ROLL, idx, T_COL_ROLL)
                elif ot == ORGAN_FLOWER:
                    fl_idx = _to_int(t[idx, T_COL_CHILD_INDEX])
                    fl_base_col = COL_FL0_PITCH + fl_idx * 5
                    if fl_base_col + 4 < NUM_FEATURES_LEGACY:
                        set_cell(row_i, fl_base_col, idx, T_COL_PITCH)
                        set_cell(row_i, fl_base_col + 1, idx, T_COL_YAW)
                        set_cell(row_i, fl_base_col + 2, idx, T_COL_ROLL)
                        set_cell(row_i, fl_base_col + 3, idx, T_COL_FLOWER_AZIMUTH)
                        set_cell(row_i, fl_base_col + 4, idx, T_COL_SCALE)
                    const_val[row_i, COL_NUM_FLOWERS] = max(
                        const_val[row_i, COL_NUM_FLOWERS], float(fl_idx + 1)
                    )
                    set_cell(row_i, COL_FLOWER_OFFSET, idx, T_COL_FLOWER_OFFSET)

            const_val[row_i, COL_EXISTENCE] = 1.0

        if M == 0:
            return torch.zeros((0, NUM_FEATURES_LEGACY), dtype=torch.float32, device=t.device)

        src_row_t = torch.from_numpy(src_row).to(device=t.device)
        src_col_t = torch.from_numpy(src_col).to(device=t.device)
        use_gather_t = torch.from_numpy(use_gather).to(device=t.device)
        const_val_t = torch.from_numpy(const_val).to(device=t.device)

        t_pad = torch.cat(
            [t, torch.zeros((1, NUM_FEATURES_TYPED), dtype=t.dtype, device=t.device)],
            dim=0,
        )
        gathered = t_pad[src_row_t, src_col_t]
        return torch.where(use_gather_t, gathered, const_val_t)

    @classmethod
    def from_legacy_tensor(cls, legacy_tensor: torch.Tensor, raw_metadata: Optional[List[Dict[str, Any]]] = None) -> "PlantOrganArray":
        """Build a typed (N, 40) PlantOrganArray from a legacy (M, 94) tensor.

        This is a convenience wrapper: it writes the legacy tensor to XML and
        re-parses it with the typed parser, ensuring semantic consistency.
        """
        tmp = cls(tensor=legacy_tensor, raw_metadata=raw_metadata or [])
        xml = tmp._to_xml_string_legacy()
        return cls._from_xml_string_typed(xml)

    def to_part_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Converts this PlantOrganArray into canonical 16D Part Tensor on the given device.
        (organ_type, base_x, base_y, base_z, rot6d_0..5, scale_x, scale_y, scale_z, existence, curvature, phyllotaxis).
        Evaluates forward kinematics tree to ensure correct 3D world positions and orientations.
        """
        t = self.tensor.to(device=device) if device is not None else self.tensor
        if t.shape[1] == NUM_FEATURES_PART:
            return t

        from diffusion_based.models.helios_pytorch_geometry import HeliosPlantGeometryBuilder
        builder = HeliosPlantGeometryBuilder()
        return builder.extract_part_tensor(self, device=device or t.device)

    # -------------------------------------------------------------------------
    # SOFT PARENT HELPERS (work for both layouts)
    # -------------------------------------------------------------------------
    @staticmethod
    def _xml_parent_node_to_linear_idx(
        tensor: torch.Tensor,
        parent_shoot_id: int,
        parent_node_xml: int,
    ) -> int:
        """Map XML parent_node_index (1-based phytomer index) to a linear node index."""
        if parent_shoot_id < 0:
            return -1
        target_phyt_idx = parent_node_xml - 1 if parent_node_xml > 0 else 0
        N = tensor.shape[0]
        for idx in range(N):
            if tensor.shape[1] == NUM_FEATURES_LEGACY:
                sid = int(tensor[idx, COL_SHOOT_ID].item())
                phyt_idx = int(tensor[idx, COL_PHYTOMER_IDX].item())
            else:
                sid = int(tensor[idx, T_COL_SHOOT_ID].item())
                phyt_idx = int(tensor[idx, T_COL_PHYTOMER_IDX].item())
                if int(tensor[idx, T_COL_ORGAN_TYPE].item()) != ORGAN_INTERNODE:
                    continue
            if sid == parent_shoot_id and phyt_idx == target_phyt_idx:
                return idx
        for idx in range(N):
            if tensor.shape[1] == NUM_FEATURES_LEGACY:
                if int(tensor[idx, COL_SHOOT_ID].item()) == parent_shoot_id:
                    return idx
            else:
                if (int(tensor[idx, T_COL_SHOOT_ID].item()) == parent_shoot_id
                        and int(tensor[idx, T_COL_ORGAN_TYPE].item()) == ORGAN_INTERNODE):
                    return idx
        return 0

    @staticmethod
    def build_parent_candidates_from_gt(
        organ_array: "PlantOrganArray",
        num_candidates: int = 8,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create soft parent candidates from a ground-truth PlantOrganArray.

        Works for both legacy and typed tensors.
        """
        cpu_rng = torch.Generator(device="cpu").manual_seed(seed)
        t = organ_array.tensor
        N = organ_array.num_nodes

        # Map shoot_id -> list of internode (or phytomer) node indices and first node per shoot
        shoots_dict: Dict[int, List[int]] = {}
        for idx in range(N):
            if t.shape[1] == NUM_FEATURES_TYPED:
                if int(t[idx, T_COL_ORGAN_TYPE].item()) not in (ORGAN_INTERNODE, ORGAN_SHOOT_META):
                    continue
                sid = int(t[idx, T_COL_SHOOT_ID].item())
            else:
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
            if t.shape[1] == NUM_FEATURES_LEGACY:
                gt_shoot = int(t[first_node, COL_PARENT_SHOOT_ID].item())
                gt_node_xml = int(t[first_node, COL_PARENT_NODE_IDX].item())
                gt_pet = int(t[first_node, COL_PARENT_PETIOLE_IDX].item())
            else:
                # For typed layout, read parent refs from the SHOOT_META row
                gt_shoot = int(t[first_node, T_COL_PARENT_SHOOT_ID].item())
                gt_node_xml = int(t[first_node, T_COL_PARENT_NODE_IDX].item())
                gt_pet = int(t[first_node, T_COL_PARENT_PETIOLE_IDX].item())

            if gt_shoot < 0:
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

        logits = torch.full((num_shoots, num_candidates), -2.0, dtype=torch.float32)
        logits[:, 0] = 2.0
        return logits, parent_candidates

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


# =============================================================================
# SMALL XML PARSING HELPERS FOR TYPED PARSER
# =============================================================================

def _get_text_default(elem: Optional[ET.Element], tag: str, default: str) -> str:
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
