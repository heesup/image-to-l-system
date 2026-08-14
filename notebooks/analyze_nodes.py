"""Analyze 22D node arrays from XML to understand leaf expansion."""
import os
import sys
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from diffusion_based.models.helios_xml_parser import HeliosXMLParser, OrganNode3D


def main():
    output_dir = os.path.join(repo_root, "notebooks", "output_dap_benchmark")
    for dap in [10, 50, 90]:
        xml_path = os.path.join(output_dir, f"dap{dap}_gt_0000_plant_0000.xml")
        parser = HeliosXMLParser(xml_path)
        parser.parse()
        nodes = parser.get_all_organ_nodes()
        nodes_np = np.stack([n.to_vec() for n in nodes], axis=0)
        organs = nodes_np[:, 11:17].argmax(axis=1)
        print(f"\nDAP {dap}: total nodes = {len(nodes)}, shape = {nodes_np.shape}")
        print(f"  Internodes: {(organs == 0).sum()}")
        print(f"  Petioles:   {(organs == 1).sum()}")
        print(f"  Leaves:     {(organs == 2).sum()}")
        print(f"  Buds:       {(organs == 3).sum()}")
        print(f"  Flowers:    {(organs == 4).sum()}")
        print(f"  Pods:       {(organs == 5).sum()}")
        # Check if leaves are already in groups of 3 (trifoliate)
        leaf_indices = np.where(organs == 2)[0]
        if len(leaf_indices) > 0:
            print(f"  Leaf node count divisible by 3? {len(leaf_indices) % 3 == 0}")
            # Check parent petiole indices for leaves
            parent_petioles = [nodes[i].parent_idx for i in leaf_indices]
            from collections import Counter
            pet_counts = Counter(parent_petioles)
            print(f"  Petiole->leaf counts: min={min(pet_counts.values())}, max={max(pet_counts.values())}, common={pet_counts.most_common(5)}")
            # Sample first few leaf directions
            for i in leaf_indices[:6]:
                n = nodes[i]
                print(f"    Leaf {i}: len={n.length:.4f}, dir={n.direction}, pitch={n.pitch:.2f}, yaw={n.yaw:.2f}, roll={n.roll:.2f}")


if __name__ == "__main__":
    main()
