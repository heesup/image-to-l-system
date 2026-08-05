"""Write Helios plant XML from a 15D organ-node graph.

This is the inverse of HeliosXMLParser.get_all_organ_nodes(). It reconstructs a
valid Helios plant_instance XML from a flat list of OrganNode3D objects so that
XML -> 15D -> XML round-trips preserve geometry.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Optional
import numpy as np
import math

from diffusion_based.models.helios_xml_parser import OrganNode3D


def _fmt(v) -> str:
    """Format a number for XML text."""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _fmt_vec(v) -> str:
    return f" {float(v[0])} {float(v[1])} {float(v[2])} "


def _organ_nodes_to_xml(
    nodes: List[OrganNode3D],
    base_position: Optional[np.ndarray] = None,
    plant_age: int = 0,
) -> ET.Element:
    """Convert flat organ nodes to a Helios XML ElementTree root.

    Args:
        nodes: List of OrganNode3D objects (must include parent_idx, shoot_id,
            phytomer_idx, and organ_type one-hot semantics).
        base_position: Optional plant base position; defaults to nodes[0].position.
        plant_age: Plant age in days for the XML header.

    Returns:
        Root <helios> ElementTree element.
    """
    if base_position is None:
        base_position = nodes[0].position if nodes else np.zeros(3)

    root = ET.Element("helios")
    plant = ET.SubElement(root, "plant_instance")
    plant.set("ID", "0")

    bp = ET.SubElement(plant, "base_position")
    bp.text = _fmt_vec(base_position)

    age = ET.SubElement(plant, "plant_age")
    age.text = str(plant_age)

    # Group nodes by shoot_id, then phytomer_idx, then by parent for petiole/leaf/bud
    shoot_data: dict = {}
    for node in nodes:
        if node.organ_type == OrganNode3D.INTERNODE:
            shoot_data.setdefault(node.shoot_id, {}).setdefault(
                node.phytomer_idx, {"internode": node, "petioles": {}}
            )

    # Assign petioles to their parent internode
    for node in nodes:
        if node.organ_type == OrganNode3D.PETIOLE:
            parent_internode = node.parent_idx
            # Find the internode node this petiole belongs to
            # Parent_idx points to the internode global index
            internode_node = None
            if 0 <= parent_internode < len(nodes):
                pnode = nodes[parent_internode]
                if pnode.organ_type == OrganNode3D.INTERNODE:
                    internode_node = pnode
            if internode_node is None:
                # Fallback: match by shoot/phytomer
                for n in nodes:
                    if (n.organ_type == OrganNode3D.INTERNODE and
                            n.shoot_id == node.shoot_id and
                            n.phytomer_idx == node.phytomer_idx):
                        internode_node = n
                        break
            if internode_node is None:
                continue

            shoot_id = internode_node.shoot_id
            phyt_idx = internode_node.phytomer_idx
            phyt = shoot_data.setdefault(shoot_id, {}).setdefault(
                phyt_idx, {"internode": internode_node, "petioles": {}}
            )
            # Use node index as petiole key
            pet_key = id(node)
            phyt["petioles"][pet_key] = {"petiole": node, "leaves": [], "buds": []}

    # Assign leaves and buds to their parent petiole
    for node in nodes:
        if node.organ_type not in (OrganNode3D.LEAF, OrganNode3D.FLORAL_BUD):
            continue
        parent_petiole_idx = node.parent_idx
        parent_petiole_node = None
        if 0 <= parent_petiole_idx < len(nodes):
            pnode = nodes[parent_petiole_idx]
            if pnode.organ_type == OrganNode3D.PETIOLE:
                parent_petiole_node = pnode
        if parent_petiole_node is None:
            continue

        # Find the petiole slot by object identity
        pet_key = id(parent_petiole_node)
        for phyt in shoot_data.get(parent_petiole_node.shoot_id, {}).values():
            if pet_key in phyt["petioles"]:
                if node.organ_type == OrganNode3D.LEAF:
                    phyt["petioles"][pet_key]["leaves"].append(node)
                else:
                    phyt["petioles"][pet_key]["buds"].append(node)
                break

    # Emit shoots in order
    for shoot_id in sorted(shoot_data.keys()):
        shoot_elem = ET.SubElement(plant, "shoot")
        shoot_elem.set("ID", str(shoot_id))

        # Determine parent shoot from internode parent relationships
        first_internode = None
        for phyt in shoot_data[shoot_id].values():
            if phyt["internode"] is not None:
                first_internode = phyt["internode"]
                break

        parent_shoot_id = -1
        parent_node_index = 0
        parent_petiole_index = 0
        if first_internode is not None and first_internode.parent_idx >= 0:
            pnode = nodes[first_internode.parent_idx]
            if pnode.organ_type == OrganNode3D.INTERNODE:
                # The parent XML shoot is the one that contains the parent internode.
                parent_shoot_id = pnode.shoot_id
                parent_node_index = pnode.phytomer_idx

        st = ET.SubElement(shoot_elem, "shoot_type_label")
        st.text = "trifoliate" if shoot_id > 0 else "unifoliate"

        psid = ET.SubElement(shoot_elem, "parent_shoot_ID")
        psid.text = str(parent_shoot_id)
        pni = ET.SubElement(shoot_elem, "parent_node_index")
        pni.text = str(parent_node_index)
        ppi = ET.SubElement(shoot_elem, "parent_petiole_index")
        ppi.text = str(parent_petiole_index)

        br = ET.SubElement(shoot_elem, "base_rotation")
        br.text = " 0 0 0 "

        for phyt_idx in sorted(shoot_data[shoot_id].keys()):
            phyt = shoot_data[shoot_id][phyt_idx]
            internode_node = phyt["internode"]
            if internode_node is None:
                continue

            phyt_elem = ET.SubElement(shoot_elem, "phytomer")
            int_elem = ET.SubElement(phyt_elem, "internode")

            int_geom = ET.SubElement(int_elem, "geometry")
            ET.SubElement(int_geom, "position").text = _fmt_vec(internode_node.position)
            ET.SubElement(int_geom, "tip_position").text = _fmt_vec(internode_node.tip_position)
            ET.SubElement(int_geom, "direction").text = _fmt_vec(internode_node.direction)

            ET.SubElement(int_elem, "internode_length").text = _fmt(internode_node.length)
            ET.SubElement(int_elem, "internode_radius").text = _fmt(internode_node.radius)
            ET.SubElement(int_elem, "internode_pitch").text = _fmt(internode_node.pitch)
            ET.SubElement(int_elem, "internode_phyllotactic_angle").text = _fmt(internode_node.yaw)
            ET.SubElement(int_elem, "internode_length_max").text = _fmt(internode_node.length)
            ET.SubElement(int_elem, "internode_length_segments").text = "2"
            ET.SubElement(int_elem, "curvature_perturbations").text = "0;0"
            ET.SubElement(int_elem, "yaw_perturbations").text = "0;0"

            for pet_data in phyt["petioles"].values():
                pet_node = pet_data["petiole"]
                pet_elem = ET.SubElement(int_elem, "petiole")

                pet_geom = ET.SubElement(pet_elem, "geometry")
                ET.SubElement(pet_geom, "position").text = _fmt_vec(pet_node.position)
                ET.SubElement(pet_geom, "tip_position").text = _fmt_vec(pet_node.tip_position)
                ET.SubElement(pet_geom, "direction").text = _fmt_vec(pet_node.direction)

                ET.SubElement(pet_elem, "petiole_length").text = _fmt(pet_node.length)
                ET.SubElement(pet_elem, "petiole_radius").text = _fmt(pet_node.radius)
                ET.SubElement(pet_elem, "petiole_pitch").text = _fmt(pet_node.pitch)
                ET.SubElement(pet_elem, "petiole_curvature").text = "0"
                ET.SubElement(pet_elem, "current_leaf_scale_factor").text = "1"
                ET.SubElement(pet_elem, "petiole_taper").text = "0.25"
                ET.SubElement(pet_elem, "petiole_length_segments").text = "5"
                ET.SubElement(pet_elem, "petiole_radial_subdivisions").text = "6"
                ET.SubElement(pet_elem, "leaflet_scale").text = "1"
                ET.SubElement(pet_elem, "leaflet_offset").text = "0.4"

                # Sort leaves deterministically (preserve original order by phytomer/leaf index)
                for leaf_node in pet_data["leaves"]:
                    leaf_elem = ET.SubElement(pet_elem, "leaf")
                    leaf_geom = ET.SubElement(leaf_elem, "geometry")
                    ET.SubElement(leaf_geom, "position").text = _fmt_vec(leaf_node.position)
                    ET.SubElement(leaf_geom, "tip_position").text = _fmt_vec(leaf_node.tip_position)
                    ET.SubElement(leaf_geom, "direction").text = _fmt_vec(leaf_node.direction)
                    ET.SubElement(leaf_elem, "leaf_scale").text = _fmt(leaf_node.length)
                    ET.SubElement(leaf_elem, "leaf_pitch").text = _fmt(leaf_node.pitch)
                    ET.SubElement(leaf_elem, "leaf_yaw").text = _fmt(leaf_node.yaw)
                    ET.SubElement(leaf_elem, "leaf_roll").text = _fmt(leaf_node.roll)

                for bud_node in pet_data["buds"]:
                    bud_elem = ET.SubElement(pet_elem, "floral_bud")
                    ET.SubElement(bud_elem, "bud_state").text = "5"
                    ET.SubElement(bud_elem, "parent_index").text = "0"
                    ET.SubElement(bud_elem, "bud_index").text = "0"
                    ET.SubElement(bud_elem, "is_terminal").text = "0"
                    ET.SubElement(bud_elem, "current_fruit_scale_factor").text = "1"
                    bud_geom = ET.SubElement(bud_elem, "geometry")
                    ET.SubElement(bud_geom, "position").text = _fmt_vec(bud_node.position)
                    ET.SubElement(bud_geom, "tip_position").text = _fmt_vec(bud_node.tip_position)
                    ET.SubElement(bud_geom, "direction").text = _fmt_vec(bud_node.direction)

                    ped_elem = ET.SubElement(bud_elem, "peduncle")
                    ET.SubElement(ped_elem, "length").text = _fmt(bud_node.length)
                    ET.SubElement(ped_elem, "radius").text = _fmt(bud_node.radius)
                    ET.SubElement(ped_elem, "pitch").text = _fmt(bud_node.pitch)
                    ET.SubElement(ped_elem, "curvature").text = "0"
                    ET.SubElement(ped_elem, "roll").text = _fmt(bud_node.roll)
                    inf_elem = ET.SubElement(bud_elem, "inflorescence")
                    ET.SubElement(inf_elem, "flower_offset").text = "0.05"

    return root


def write_organ_nodes_to_xml(
    nodes: List[OrganNode3D],
    xml_path: str,
    base_position: Optional[np.ndarray] = None,
    plant_age: int = 0,
) -> str:
    """Write a Helios XML file from 15D organ nodes and return the path."""
    root = _organ_nodes_to_xml(nodes, base_position, plant_age)
    rough = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent="\t")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(pretty)
    return xml_path
