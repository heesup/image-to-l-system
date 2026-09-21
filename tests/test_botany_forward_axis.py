"""The botany forward-axis loss must use rot6d's FORWARD column, not its x column.

The rot6d convention here is COLUMNS: roll_to_matrix returns stack([x, forward, z]) and
matrix_to_rot6d keeps the first two columns, so a rot6d vector is [x | forward]. Component 0:3 is
therefore perpendicular to the branch, and 3:6 is along it.

Until 2026-09-20 `loss_axis` in the training loop aligned rot6d[0:3] with the parent->node
direction, driving the absolute layout's rotations 90 degrees away from correct. Every merged_*
run trained with stage3_botany_weight=1.0 under that term.
"""
import os

import torch
import torch.nn.functional as F

from plant_recon.dataset.phytomer_roll import roll_to_matrix
from plant_recon.dataset.phytomer_packets import matrix_to_rot6d


def test_rot6d_forward_axis_is_components_3_to_6():
    fwd = F.normalize(torch.tensor([[0.3, 0.9, -0.2]]), dim=-1)
    roll = torch.tensor([[0.6, 0.8]])
    d6 = matrix_to_rot6d(roll_to_matrix(fwd, roll))
    a_x = F.normalize(d6[..., 0:3], dim=-1)
    a_f = F.normalize(d6[..., 3:6], dim=-1)
    cos_x = (a_x * fwd).sum(-1).abs().item()
    cos_f = (a_f * fwd).sum(-1).item()
    assert cos_f > 0.999, f"rot6d[3:6] must BE the forward axis, got cos={cos_f:.4f}"
    assert cos_x < 0.01, f"rot6d[0:3] is the x column and must be perpendicular, got |cos|={cos_x:.4f}"


def test_training_loss_uses_the_forward_column():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "plant_recon", "training", "train_hierarchical_flow_matching.py")).read()
    i = src.index("loss_axis =")
    window = src[max(0, i - 900):i]
    assert "rot_hat[..., 3:6]" in window, (
        "the botany forward-axis term must align rot6d[3:6]; [0:3] is perpendicular to the branch "
        "and aligning it rotates the node 90 degrees away from correct")
    assert "rot_hat[..., 0:3]" not in window
