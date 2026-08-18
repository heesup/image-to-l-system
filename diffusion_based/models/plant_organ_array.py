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
    Minimal direct XML -> part tensor parser.

    Computes base positions, 6D rotations, and scales for each organ from a
    Helios XML document. Flower/fruit support is best-effort.
    """
    root = ET.fromstring(xml_content)
    if root.tag != "helios":
        raise ValueError("Root tag must be <helios>")

    rows = []
    # Cache internode tips for child-shoot base lookup: (shoot_id, phytomer_idx) -> tip
    tip_cache: dict = {}

    for plant_elem in root.findall("plant_instance"):
        pid = int(plant_elem.attrib.get("ID", plant_elem.attrib.get("id", 0)))

        bp_text = _get_text_default(plant_elem, "base_position", None)
        if bp_text is None:
            bp_text = _get_text_default(plant_elem, "plant_base_position", "0 0 0")
        bp_vals = [float(x) for x in bp_text.replace(";", " ").split() if x.strip()]
        if len(bp_vals) < 3:
            bp_vals = [0.0, 0.0, 0.0]
        plant_base = np.array(bp_vals[:3], dtype=np.float32)

        # Root meta row
        rows.append(_make_row(ORGAN_ROOT_META, plant_base, np.eye(3), np.zeros(3), 1.0))

        for shoot_elem in plant_elem.findall("shoot"):
            sid = int(
                shoot_elem.attrib.get(
                    "ID",
                    shoot_elem.attrib.get("shoot_id", shoot_elem.attrib.get("id", 0)),
                )
            )
            psi_text = _get_text_default(shoot_elem, "parent_shoot_ID", None)
            if psi_text is None:
                psi_text = _get_text_default(shoot_elem, "parent_shoot_id", "-1")
            psi = int(psi_text.strip()) if psi_text.strip() else -1

            pni_text = _get_text_default(shoot_elem, "parent_node_index", "0")
            pni = int(pni_text.strip()) if pni_text.strip() else 0
            ppi_text = _get_text_default(shoot_elem, "parent_petiole_index", "0")
            ppi = int(ppi_text.strip()) if ppi_text.strip() else 0

            br_text = _get_text_default(shoot_elem, "base_rotation", None)
            if br_text is not None:
                br_vals = [float(x) for x in br_text.split() if x.strip()]
                if len(br_vals) >= 3:
                    br_pitch, br_yaw, br_roll = br_vals[0], br_vals[1], br_vals[2]
                else:
                    br_pitch = br_yaw = br_roll = 0.0
            else:
                br_pitch = _get_float_text(shoot_elem, "shoot_base_pitch", 0.0)
                br_yaw = _get_float_text(shoot_elem, "shoot_base_yaw", 0.0)
                br_roll = _get_float_text(shoot_elem, "shoot_base_roll", 0.0)

            R_shoot = _rotation_matrix(
                math.radians(br_roll), math.radians(br_pitch), math.radians(br_yaw)
            )

            # Determine shoot base from parent references if possible
            if psi < 0:
                shoot_base = plant_base.copy()
            else:
                key = (psi, max(0, pni - 1))
                shoot_base = tip_cache.get(key, plant_base).copy()

            rows.append(_make_row(ORGAN_SHOOT_META, shoot_base, R_shoot, np.zeros(3), 1.0))

            cum_phyllo = 0.0
            prev_tip = shoot_base.copy()
            prev_R = R_shoot.copy()

            for phyto_idx, phyto_elem in enumerate(shoot_elem.findall("phytomer")):
                internode_elem = phyto_elem.find("internode")
                if internode_elem is None:
                    continue

                il = _get_float_text(internode_elem, "internode_length", 0.0)
                ir = _get_float_text(internode_elem, "internode_radius", 0.0)
                ip = math.radians(_get_float_text(internode_elem, "internode_pitch", 0.0))
                ipa = math.radians(
                    _get_float_text(internode_elem, "internode_phyllotactic_angle", 0.0)
                )

                R_inode = prev_R @ _rotation_matrix(0.0, ip, cum_phyllo)
                inode_base = prev_tip.copy()
                inode_tip = inode_base + R_inode[:, 2] * il
                rows.append(
                    _make_row(ORGAN_INTERNODE, inode_base, R_inode, np.array([ir, ir, il]), 1.0)
                )
                tip_cache[(sid, phyto_idx)] = inode_tip
                cum_phyllo += ipa
                prev_tip = inode_tip
                prev_R = R_inode

                # Petioles (including leaves, buds, peduncles, flowers)
                for pet_i, pet_elem in enumerate(internode_elem.findall("petiole")):
                    pl = _get_float_text(pet_elem, "petiole_length", 0.0)
                    pr = _get_float_text(pet_elem, "petiole_radius", 0.0)
                    pp = math.radians(_get_float_text(pet_elem, "petiole_pitch", 0.0))

                    R_pet = R_inode @ _rotation_matrix(0.0, pp, 0.0)
                    pet_base = inode_tip.copy()
                    pet_tip = pet_base + R_pet[:, 2] * pl
                    rows.append(
                        _make_row(ORGAN_PETIOLE, pet_base, R_pet, np.array([pr, pr, pl]), 1.0)
                    )

                    for lf_idx, leaf_elem in enumerate(pet_elem.findall("leaf")):
                        lfs = _get_float_text(leaf_elem, "leaf_scale", 1.0)
                        lfp = math.radians(_get_float_text(leaf_elem, "leaf_pitch", 0.0))
                        lfy = math.radians(_get_float_text(leaf_elem, "leaf_yaw", 0.0))
                        lfr = math.radians(_get_float_text(leaf_elem, "leaf_roll", 0.0))
                        R_leaf = R_pet @ _rotation_matrix(lfr, lfp, lfy)
                        rows.append(
                            _make_row(ORGAN_LEAF, pet_tip.copy(), R_leaf, np.full(3, lfs), 1.0)
                        )

                    fb_elem = pet_elem.find("floral_bud")
                    if fb_elem is not None:
                        bud_state = _get_int_text(fb_elem, "bud_state", 0)
                        ot_bud = ORGAN_FLOWER_CLOSED if bud_state in (5, 6, 7) else ORGAN_BUD
                        rows.append(
                            _make_row(ot_bud, pet_base.copy(), R_pet, np.full(3, 0.01), 1.0)
                        )

                        ped_elem = fb_elem.find("peduncle")
                        if ped_elem is not None:
                            pdl = _get_float_text(ped_elem, "length", 0.0)
                            pdr = _get_float_text(ped_elem, "radius", 0.0)
                            pdp = math.radians(_get_float_text(ped_elem, "pitch", 0.0))
                            pdrl = math.radians(_get_float_text(ped_elem, "roll", 0.0))
                            R_ped = R_inode @ _rotation_matrix(pdrl, pdp, 0.0)
                            ped_base = inode_tip.copy()
                            ped_tip = ped_base + R_ped[:, 2] * pdl
                            rows.append(
                                _make_row(
                                    ORGAN_PEDUNCLE, ped_base, R_ped, np.array([pdr, pdr, pdl]), 1.0
                                )
                            )

                            infl_elem = fb_elem.find("inflorescence")
                            if infl_elem is not None:
                                for fl_idx, fl_elem in enumerate(infl_elem.findall("flower")):
                                    fp = math.radians(
                                        _get_float_text(fl_elem, "flower_pitch", 0.0)
                                    )
                                    fy = math.radians(
                                        _get_float_text(fl_elem, "flower_yaw", 0.0)
                                    )
                                    fr = math.radians(
                                        _get_float_text(fl_elem, "flower_roll", 0.0)
                                    )
                                    fbs = _get_float_text(fl_elem, "flower_base_scale", 1.0)
                                    R_fl = R_ped @ _rotation_matrix(fr, fp, fy)
                                    rows.append(
                                        _make_row(
                                            ORGAN_FLOWER,
                                            ped_tip.copy(),
                                            R_fl,
                                            np.full(3, fbs),
                                            1.0,
                                        )
                                    )

    if not rows:
        return torch.zeros((0, NUM_FEATURES_14D), dtype=torch.float32)
    return torch.from_numpy(np.stack(rows, axis=0)).float()


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
