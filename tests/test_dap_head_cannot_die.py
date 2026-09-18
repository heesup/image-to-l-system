"""The DAP head must never reach a state with zero gradient.

On 2026-09-18 both A/B training runs lost their DAP head the moment the LR warmup peaked at
epoch 3: `pred_dap = F.relu(self.dap_head(h)) * 100.0` went to a hard 0.0 and stayed there for
every batch. A ReLU whose input is negative for all inputs outputs exactly 0 AND has exactly 0
gradient, so the head is dead permanently -- no later batch and no LR decay can revive it.

This is not merely a broken auxiliary metric. `pred_dap` is fed back as the `dap_clue`
conditioning added to every phytomer query, so a dead head conditions every plant as DAP 0
whatever its real maturity. Holdout silhouette IoU fell to 1.7-4.6% and depth MAE climbed
monotonically while node RMSE stayed clean, because the node positions are supervised directly
and it is the maturity conditioning that was gone.

The fix is the idiom the neighbouring count head already uses, ELU+1, which is strictly positive
and has a strictly positive gradient everywhere.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from plant_recon.models.hierarchical_part_flow_matching import MacroBiologicalHead

EMBED = 384


def _drive_head_negative(head):
    """Force both heads' pre-activations strongly negative, the state the runs fell into."""
    with torch.no_grad():
        for lin in (head.dap_head, head.phy_head):
            lin.weight.fill_(0.0)
            lin.bias.fill_(-50.0)


class TestDapHeadCannotDie(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260918)
        self.head = MacroBiologicalHead(embed_dim=EMBED)
        self.cls = torch.randn(4, EMBED)

    def test_dap_is_positive_and_has_gradient_when_preactivation_is_negative(self):
        _drive_head_negative(self.head)
        out = self.head(self.cls, max_k=64)
        dap = out["pred_dap"]

        self.assertTrue(torch.isfinite(dap).all(), f"pred_dap is not finite: {dap.flatten().tolist()}")

        self.head.zero_grad(set_to_none=True)
        dap.sum().backward()
        g = self.head.dap_head.weight.grad
        self.assertIsNotNone(g, "no gradient reached dap_head")
        # A meaningful floor, not merely "> 0". The original ReLU gives exactly 0 here, and the
        # soft-positive alternatives (elu+1, softplus) give ~1e-22, which is numerically dead even
        # though it passes a bare > 0 check. Only an activation-free head keeps a usable gradient.
        self.assertGreater(
            float(g.abs().sum()), 1e-4,
            "dap_head gradient has vanished -- the head cannot recover from a negative pre-activation",
        )

    def test_the_count_head_carries_the_same_latent_risk(self):
        """`pred_num_phytomers` still uses ELU+1, which has the same failure mode further out.

        It survived 2026-09-18 -- its predictions tracked ground truth right through the DAP
        collapse -- so it is deliberately left alone rather than changed alongside a head that
        actually broke. This test pins the risk rather than hiding it: driven to the same negative
        pre-activation, the count head's gradient IS numerically dead. If it ever pins at a constant
        in training the way DAP did, make it linear too.
        """
        _drive_head_negative(self.head)
        out = self.head(self.cls, max_k=64)
        self.head.zero_grad(set_to_none=True)
        out["pred_num_phytomers"].sum().backward()
        g = self.head.phy_head.weight.grad
        self.assertLess(
            float(g.abs().sum()), 1e-12,
            "phy_head now survives a negative pre-activation -- if it was made linear, update this "
            "test to assert a real gradient floor instead",
        )

    def test_dap_still_spans_the_useful_range(self):
        """The head must still span the DAP values in the data (1-100 for this dataset).

        A linear head can momentarily predict a negative DAP; that is acceptable and is what buys
        the constant gradient. The regression loss pulls it back, and the dap_clue embedding is a
        learned Linear that takes a negative input without trouble.
        """
        with torch.no_grad():
            self.head.dap_head.weight.fill_(0.0)
            for bias, lo, hi in ((-0.1, -20.0, 0.0), (0.0, -1.0, 1.0), (0.5, 40.0, 60.0)):
                self.head.dap_head.bias.fill_(bias)
                dap = float(self.head(self.cls, max_k=64)["pred_dap"].mean())
                self.assertGreaterEqual(dap, lo, f"bias {bias} -> DAP {dap:.1f}, below {lo}")
                self.assertLessEqual(dap, hi, f"bias {bias} -> DAP {dap:.1f}, above {hi}")


if __name__ == "__main__":
    unittest.main()
