"""An absolute checkpoint must not be evaluated through the relative rotation path.

sample_ode returns the node's own rot6d in `phytomer_roll` when --stage3_absolute is set: the field
keeps its name but carries 6 components rather than the relative layout's 2. Feeding that to
reconstruct_phytomer_rot silently evaluates the absolute model AS IF it were relative --
roll_to_matrix normalises all six components together, reads only [0:1] and [1:2] as (cos, sin),
discards the rest, and rebuilds the forward axis from the chain, which is exactly what the absolute
layout exists to avoid. Every absolute number measured before 2026-09-20 came through that path.

These tests pin the two halves of the fix: the eval script must branch on the layout, and
roll_to_matrix must be understood to consume only two components (so a 6-wide input is never
silently acceptable).
"""
import os
import re

import torch

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "plant_recon", "eval", "eval_test_time_refinement.py")


def test_eval_branches_on_the_layout():
    s = open(SRC).read()
    assert "if s3abs:" in s, "the eval must branch on stage3_absolute when building the rotation"
    # the relative call must no longer be the only one reached
    i = s.index("if s3abs:")
    window = s[i:i + 1400]
    assert "rot, par = roll.float()" in window, (
        "under the absolute layout the rotation must be the model's own rot6d, not the chain's")
    assert "_roll_dummy" in window, (
        "the chain is still needed for parent positions; its rotation output must be discarded")


def test_roll_to_matrix_reads_only_two_components():
    """Documents WHY a 6-wide roll is destructive rather than merely unused."""
    from plant_recon.dataset.phytomer_roll import roll_to_matrix
    fwd = torch.tensor([[0.0, 1.0, 0.0]])
    roll2 = torch.tensor([[1.0, 0.0]])
    # the same leading pair, padded to rot6d width, must NOT give the same rotation --
    # normalisation runs over every component, so extra components change the result
    roll6 = torch.tensor([[1.0, 0.0, 0.3, 0.4, 0.5, 0.6]])
    R2 = roll_to_matrix(fwd, roll2)
    R6 = roll_to_matrix(fwd, roll6)
    assert not torch.allclose(R2, R6, atol=1e-4), (
        "a 6-wide roll silently produces a DIFFERENT rotation from its leading pair, which is why "
        "passing rot6d here corrupted the absolute evaluation instead of merely ignoring it")
