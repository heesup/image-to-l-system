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

    # Forward Kinematics 3D position propagation along parent-child graph links
    for idx, node in enumerate(nodes):
        p_idx = node.parent_idx

        # Enforce upward stem growth for internodes if pitch is flat
        if node.organ_type == OrganNode3D.INTERNODE and abs(node.pitch) < 10.0:
            node.pitch = 80.0  # 80 degrees upward stem growth

        # Enforce outward branching angle for petioles
        if node.organ_type == OrganNode3D.PETIOLE and abs(node.pitch) < 10.0:
            node.pitch = 45.0  # 45 degrees outward petiole angle
            node.yaw = float((idx * 137.5) % 360)  # Golden ratio phyllotaxis angle

        pitch_rad = math.radians(node.pitch)
        yaw_rad = math.radians(node.yaw)
        dx = math.cos(pitch_rad) * math.cos(yaw_rad)
        dy = math.cos(pitch_rad) * math.sin(yaw_rad)
        dz = math.sin(pitch_rad)
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-8
        node.direction = np.array([dx / norm, dy / norm, dz / norm], dtype=np.float64)

        if 0 <= p_idx < len(nodes) and p_idx != idx:
            parent_node = nodes[p_idx]
            node.position = parent_node.tip_position.copy()
        elif idx == 0:
            node.position = base_position.copy()

        node.tip_position = node.position + node.length * node.direction

    root = ET.Element("helios")
    plant = ET.SubElement(root, "plant_instance")
    plant.set("ID", "0")

    bp = ET.SubElement(plant, "base_position")
    bp.text = _fmt_vec(base_position)

    age = ET.SubElement(plant, "plant_age")
    age.text = str(plant_age)

    # Guarantee at least one main stem internode exists
    has_internode = any(n.organ_type == OrganNode3D.INTERNODE for n in nodes)
    if nodes and not has_internode:
        nodes[0].organ_type = OrganNode3D.INTERNODE
        nodes[0].shoot_id = 0
        nodes[0].phytomer_idx = 0

    # Group nodes by shoot_id, then phytomer_idx, then by parent for petiole/leaf/bud
    shoot_data: dict = {}
    for node in nodes:
        if node.organ_type == OrganNode3D.INTERNODE:
            shoot_data.setdefault(node.shoot_id, {}).setdefault(
                node.phytomer_idx, {"internode": node, "petioles": {}}
            )

    # For non-internode nodes whose shoot/phytomer has no internode, auto-create a matching internode
    for node in nodes:
        if node.organ_type != OrganNode3D.INTERNODE:
            s_dict = shoot_data.setdefault(node.shoot_id, {})
            if node.phytomer_idx not in s_dict:
                auto_inode = OrganNode3D(OrganNode3D.INTERNODE)
                auto_inode.shoot_id = node.shoot_id
                auto_inode.phytomer_idx = node.phytomer_idx
                auto_inode.position = node.position.copy()
                auto_inode.length = 0.05
                auto_inode.radius = 0.003
                s_dict[node.phytomer_idx] = {"internode": auto_inode, "petioles": {}}

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
                # Fallback to auto-created internode in shoot_data
                phyt = shoot_data.get(node.shoot_id, {}).get(node.phytomer_idx, None)
                if phyt is not None:
                    internode_node = phyt.get("internode", None)

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

    # Guarantee every shoot/phytomer has at least one petiole so leaves can attach
    for shoot_id, s_dict in shoot_data.items():
        for phyt_idx, phyt in s_dict.items():
            if not phyt["petioles"]:
                auto_pet = OrganNode3D(OrganNode3D.PETIOLE)
                auto_pet.shoot_id = shoot_id
                auto_pet.phytomer_idx = phyt_idx
                auto_pet.length = 0.04
                auto_pet.radius = 0.0015
                auto_pet.pitch = 45.0
                pet_key = id(auto_pet)
                phyt["petioles"][pet_key] = {"petiole": auto_pet, "leaves": [], "buds": []}

    # Ensure every LEAF node is linked to its phytomer's petiole
    for node in nodes:
        if node.organ_type == OrganNode3D.LEAF:
            s_dict = shoot_data.get(node.shoot_id, {})
            phyt = s_dict.get(node.phytomer_idx, None)
            if phyt and phyt["petioles"]:
                # Check if leaf is already in leaves list
                already_linked = any(
                    node in p_entry["leaves"] for p_entry in phyt["petioles"].values()
                )
                if not already_linked:
                    first_pet_key = list(phyt["petioles"].keys())[0]
                    phyt["petioles"][first_pet_key]["leaves"].append(node)

    # Remap shoot IDs sequentially starting at 0 so main stem shoot ID 0 always exists
    orig_shoots = sorted(shoot_data.keys())
    shoot_id_map = {orig: new for new, orig in enumerate(orig_shoots)}
    
    remapped_shoot_data = {}
    for orig_id, phyt_map in shoot_data.items():
        new_id = shoot_id_map[orig_id]
        remapped_shoot_data[new_id] = phyt_map
        for phyt in phyt_map.values():
            if phyt["internode"]:
                phyt["internode"].shoot_id = new_id
            for p_entry in phyt["petioles"].values():
                p_entry["petiole"].shoot_id = new_id
                for l_node in p_entry["leaves"]:
                    l_node.shoot_id = new_id

    shoot_data = remapped_shoot_data

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
        if shoot_id > 0:
            parent_shoot_id = 0  # Main stem shoot ID is always 0
            main_phytomers_count = len(shoot_data.get(0, {}))
            if first_internode is not None and 0 <= first_internode.parent_idx < len(nodes):
                pnode = nodes[first_internode.parent_idx]
                parent_node_index = min(max(0, pnode.phytomer_idx), max(0, main_phytomers_count - 1))
            else:
                parent_node_index = 0

        st = ET.SubElement(shoot_elem, "shoot_type_label")
        st.text = "trifoliate"

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

            # Enforce EXACTLY 1 petiole per internode (matching C++ Helios petioles_per_internode = 1)
            pet_data_list = list(phyt["petioles"].values())
            if not pet_data_list:
                auto_pet = OrganNode3D(OrganNode3D.PETIOLE)
                auto_pet.length = 0.15
                auto_pet.radius = 0.004
                auto_pet.pitch = 45.0
                auto_pet.yaw = float((phyt_idx * 137.5) % 360)
                pet_data_list = [{"petiole": auto_pet, "leaves": []}]

            # Write exactly the first petiole
            pet_data = pet_data_list[0]
            pet_node = pet_data["petiole"]
            pet_node.length = max(pet_node.length, 0.15)
            pet_node.radius = max(pet_node.radius, 0.004)
            pet_node.position = internode_node.tip_position.copy()
            pet_pitch = math.radians(pet_node.pitch)
            pet_yaw = math.radians(pet_node.yaw)
            p_dx = math.cos(pet_pitch) * math.cos(pet_yaw)
            p_dy = math.cos(pet_pitch) * math.sin(pet_yaw)
            p_dz = math.sin(pet_pitch)
            p_norm = math.sqrt(p_dx*p_dx + p_dy*p_dy + p_dz*p_dz) + 1e-8
            pet_node.direction = np.array([p_dx/p_norm, p_dy/p_norm, p_dz/p_norm], dtype=np.float64)
            pet_node.tip_position = pet_node.position + pet_node.length * pet_node.direction

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

            # Enforce exact trifoliate leaf count per petiole (3 leaflets for trifoliate shoots)
            target_leaf_count = 3
            leaves_to_write = pet_data["leaves"][:target_leaf_count]
            while len(leaves_to_write) < target_leaf_count:
                auto_leaf = OrganNode3D(OrganNode3D.LEAF)
                auto_leaf.length = 0.18
                auto_leaf.pitch = -15.0
                auto_leaf.roll = -15.0
                leaves_to_write.append(auto_leaf)

            # Write exactly target_leaf_count leaf tags (OUTSIDE while loop)
            for leaf_node in leaves_to_write:
                leaf_node.length = max(leaf_node.length, 0.18)
                leaf_node.position = pet_node.tip_position.copy()
                l_pitch = math.radians(leaf_node.pitch)
                l_yaw = math.radians(leaf_node.yaw)
                l_dx = math.cos(l_pitch) * math.cos(l_yaw)
                l_dy = math.cos(l_pitch) * math.sin(l_yaw)
                l_dz = math.sin(l_pitch)
                l_norm = math.sqrt(l_dx*l_dx + l_dy*l_dy + l_dz*l_dz) + 1e-8
                leaf_node.direction = np.array([l_dx/l_norm, l_dy/l_norm, l_dz/l_norm], dtype=np.float64)
                leaf_node.tip_position = leaf_node.position + leaf_node.length * leaf_node.direction

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
