"""
[LEGACY REDIRECT] Part Assembly to XML Converter.

This module has been replaced by the canonical 14D dynamic inverse kinematics converter:
    `diffusion_based.models.part_tensor_to_40d`

The legacy 13D heuristic implementation is archived at:
    `archive/models_legacy/part_assembly_to_xml_13d_legacy.py`
"""

from diffusion_based.models.part_tensor_to_40d import (
    PartTensorTo40DConverter,
    PartAssemblyToXMLConverter,
    assemble_part_tensor_to_xml,
    _invert_helios_zxz_rotation,
    _rot_z_matrix,
)

__all__ = [
    "PartTensorTo40DConverter",
    "PartAssemblyToXMLConverter",
    "assemble_part_tensor_to_xml",
    "_invert_helios_zxz_rotation",
    "_rot_z_matrix",
]
