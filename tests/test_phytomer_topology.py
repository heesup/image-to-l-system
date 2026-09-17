"""Tests for gt_parent_links: one parent rule, no exceptions.

A node is the TOP of its internode, so its parent is one internode below it --
the previous node in the same shoot, the origin for the main stem's first node,
or the branch-point node on the parent shoot for a lateral's first node.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.dataset.phytomer_topology import gt_parent_links


class TestGtParentLinks(unittest.TestCase):
    def _straight_stem(self):
        """Main stem of 3 nodes, one internode (0.03 m) apart, rising from the
        origin -- so the first node sits at z=0.03, not at z=0."""
        centers = torch.tensor([[0.0, 0.0, 0.03],
                                [0.0, 0.0, 0.06],
                                [0.0, 0.0, 0.09]])
        keys = torch.tensor([[0, 0], [0, 1], [0, 2]])
        return centers, keys

    def test_same_shoot_parent_is_the_previous_node(self):
        centers, keys = self._straight_stem()
        pos, idx, _ = gt_parent_links(centers, keys)
        self.assertEqual(idx[1].item(), 0)
        self.assertEqual(idx[2].item(), 1)
        self.assertTrue(torch.allclose(pos[1], centers[0]))
        self.assertTrue(torch.allclose(pos[2], centers[1]))

    def test_main_stem_first_node_parents_to_the_origin(self):
        """Not a fallback guess: the first node really is one internode above
        the origin, so the origin is its true geometric parent."""
        centers, keys = self._straight_stem()
        pos, idx, _ = gt_parent_links(centers, keys)
        self.assertTrue(torch.allclose(pos[0], torch.zeros(3), atol=1e-9))
        # -1 because the origin is not one of the rows.
        self.assertEqual(idx[0].item(), -1)
        # The derived axis is then real, and vertical here.
        d = centers[0] - pos[0]
        self.assertTrue(torch.allclose(d / d.norm(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-6))

    def test_lateral_first_node_parents_to_its_branch_point(self):
        """A lateral's first node is one internode from the node it branches
        off, which lies below it on a different shoot."""
        centers = torch.tensor([[0.0, 0.0, 0.03],
                                [0.0, 0.0, 0.06],
                                [0.0, 0.0, 0.09],
                                [0.03, 0.0, 0.07]])   # lateral off node 1
        keys = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0]])
        pos, idx, _ = gt_parent_links(centers, keys)
        self.assertEqual(idx[3].item(), 1)
        self.assertTrue(torch.allclose(pos[3], centers[1]))

    def test_every_node_resolves_so_no_fallback_is_needed(self):
        centers = torch.tensor([[0.0, 0.0, 0.03],
                                [0.0, 0.0, 0.06],
                                [0.03, 0.0, 0.07],
                                [0.05, 0.0, 0.10]])
        keys = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        pos, idx, _ = gt_parent_links(centers, keys)
        # Row 0 parents to the origin (idx -1 but a real position); the rest to rows.
        self.assertTrue(torch.allclose(pos[0], torch.zeros(3), atol=1e-9))
        self.assertTrue((idx[1:] >= 0).all())
        # No row was left pointing at itself, which is how "unresolved" shows up.
        moved = (centers - pos).norm(dim=-1)
        self.assertTrue((moved > 1e-6).all())

    def test_distant_lateral_is_left_unresolved_rather_than_linked_absurdly(self):
        """The branch lookup is gated by max_edge_factor x median spacing, so a
        lateral far from everything is not attached to a random node."""
        centers = torch.tensor([[0.0, 0.0, 0.03],
                                [0.0, 0.0, 0.06],
                                [0.0, 0.0, 0.09],
                                [9.0, 0.0, 0.07]])    # absurdly far lateral
        keys = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0]])
        pos, idx, _ = gt_parent_links(centers, keys)
        self.assertEqual(idx[3].item(), -1)
        # Unresolved rows return their own position, so child - parent is zero
        # rather than a garbage direction.
        self.assertTrue(torch.allclose(pos[3], centers[3]))

    def test_main_stem_is_chosen_by_lowest_first_node_not_by_shoot_id(self):
        """shoot_id 0 happens to be the main stem in this dataset, but the rule
        derives it from geometry so a relabelled cache still works."""
        centers = torch.tensor([[0.20, 0.0, 0.40],    # shoot 0, high up: a lateral
                                [0.0, 0.0, 0.03],     # shoot 1, lowest: the main stem
                                [0.0, 0.0, 0.06]])
        keys = torch.tensor([[0, 0], [1, 0], [1, 1]])
        pos, idx, _ = gt_parent_links(centers, keys)
        self.assertTrue(torch.allclose(pos[1], torch.zeros(3), atol=1e-9))
        self.assertFalse(torch.allclose(pos[0], torch.zeros(3), atol=1e-9))

    def test_depth_from_root_makes_every_parent_exactly_one_step_below(self):
        """Depth is the ordinal the head learns: with it a lateral's first node
        is depth(branch node) + 1, so chain_phytomers' |(o_child - o_parent) - 1|
        cost is zero for the TRUE branch parent. With per-shoot ordinals that
        cost was 6 x ORD_WEIGHT = 0.12 m and rejected it."""
        centers = torch.tensor([[0.0, 0.0, 0.03],     # main 0  depth 0 (root)
                                [0.0, 0.0, 0.06],     # main 1  depth 1
                                [0.0, 0.0, 0.09],     # main 2  depth 2
                                [0.03, 0.0, 0.07],    # lateral off main 1 -> depth 2
                                [0.06, 0.0, 0.08]])   # lateral's second node -> depth 3
        keys = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]])
        pos, idx, depth = gt_parent_links(centers, keys)
        self.assertEqual(depth.tolist(), [0, 1, 2, 2, 3])
        # Every node with a parent row sits exactly one step above it.
        has = idx >= 0
        self.assertTrue((depth[has] - depth[idx[has]] == 1).all())
        # The root is the only depth-0 node, and its parent is the origin.
        self.assertEqual(int((depth == 0).sum()), 1)
        self.assertTrue(torch.allclose(pos[0], torch.zeros(3), atol=1e-9))

    def test_depth_handles_a_long_chain_without_a_per_level_loop(self):
        n = 200
        centers = torch.stack([torch.zeros(n), torch.zeros(n), 0.03 * torch.arange(1, n + 1)], -1)
        keys = torch.stack([torch.zeros(n, dtype=torch.long), torch.arange(n)], -1)
        _, _, depth = gt_parent_links(centers, keys)
        self.assertEqual(depth.tolist(), list(range(n)))

    def test_empty_input(self):
        pos, idx, _ = gt_parent_links(torch.zeros(0, 3), torch.zeros(0, 2, dtype=torch.long))
        self.assertEqual(pos.shape, (0, 3))
        self.assertEqual(idx.shape, (0,))


if __name__ == "__main__":
    unittest.main()


class TestBranchPointFromInternodeBase(unittest.TestCase):
    """A lateral's first internode starts at its branch point, so the decoded
    internode base names the parent node directly, wherever it sits."""

    def _drooping_lateral(self):
        # Main stem of 4 nodes 3 cm apart; a lateral branching from node 2
        # whose first node sits BELOW the branch point (drooping shoot).
        centers = torch.tensor([[0.0, 0.0, 0.03], [0.0, 0.0, 0.06], [0.0, 0.0, 0.09], [0.0, 0.0, 0.12],
                                [0.025, 0.0, 0.085], [0.05, 0.0, 0.08]])
        keys = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1]])
        base = centers.clone()
        base[1:4] = centers[0:3]
        base[4] = centers[2] + torch.tensor([0.003, 0.0, 0.0])   # starts at node 2, a few mm out
        base[5] = centers[4]
        return centers, keys, base

    def test_nearest_below_picks_the_wrong_node_for_a_drooping_lateral(self):
        centers, keys, _ = self._drooping_lateral()
        _, idx, _ = gt_parent_links(centers, keys)
        self.assertNotEqual(int(idx[4]), 2, "the fallback cannot see a branch point above the first node")

    def test_internode_base_names_the_true_branch_point(self):
        centers, keys, base = self._drooping_lateral()
        pos, idx, depth = gt_parent_links(centers, keys, internode_base=base)
        self.assertEqual(int(idx[4]), 2)
        self.assertTrue(torch.allclose(pos[4], centers[2]))
        self.assertEqual(depth.tolist(), [0, 1, 2, 3, 3, 4])


class TestChainPhytomersDroopingShoot(unittest.TestCase):
    """chain_phytomers must follow a shoot that descends: until 2026-09-14 a
    parent had to sit below its child, which cut every drooping lateral."""

    def test_descending_shoot_is_one_chain(self):
        from plant_recon.dataset.phytomer_topology import chain_phytomers
        pos = torch.tensor([[0.0, 0.0, 0.10], [0.03, 0.0, 0.098], [0.06, 0.0, 0.095], [0.09, 0.0, 0.091]])
        ordinal = torch.tensor([3.0, 4.0, 5.0, 6.0])
        parent, shoot, phy = chain_phytomers(pos, ordinal=ordinal)
        self.assertEqual(parent.tolist(), [-1, 0, 1, 2])
        self.assertEqual(shoot.tolist(), [0, 0, 0, 0])
        self.assertEqual(phy.tolist(), [0, 1, 2, 3])

    def test_mutual_nearest_pair_is_cut_at_the_upper_node(self):
        """Two nodes that are each other's nearest neighbour form a 2-cycle;
        the edge whose parent is HIGHER is the one to cut."""
        from plant_recon.dataset.phytomer_topology import chain_phytomers
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])
        parent, shoot, phy = chain_phytomers(pos)
        self.assertEqual(parent.tolist(), [-1, 0])
        self.assertEqual(phy.tolist(), [0, 1])
