"""An empty plant must score +inf, not 0, and must not crash the eval.

Two separate failures live here. Before 2026-09-19 a plant that materialised zero organs broke out
of the refinement loop before `loss` was ever assigned, so `float(loss)` raised UnboundLocalError
and killed the whole run -- all four Helios-override jobs died this way after two plants.

The obvious fix (initialise `loss = 0`) is worse than the bug: zero is the BEST achievable score, so
a collapsed plant would outrank every real hypothesis and `--n_starts` would select the empty one on
purpose. This pins the sign.
"""
import ast
import os

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "plant_recon", "eval", "eval_test_time_refinement.py")


def _refine_fn():
    tree = ast.parse(open(SRC).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "refine_plant":
            return n
    raise AssertionError("refine_plant not found")


def test_loss_is_initialised_to_inf_not_zero():
    """The pre-loop initialiser of `loss` must be inf; zero would make an empty plant win."""
    fn = _refine_fn()
    src = ast.get_source_segment(open(SRC).read(), fn)
    i = src.index("for step in range(")
    pre = src[:i]
    assert "loss = torch.tensor(float(\"inf\")" in pre, (
        "`loss` must be initialised to +inf before the refinement loop, so that a plant which "
        "materialises zero organs on step 0 neither crashes nor scores as the best hypothesis")
    assert "loss = torch.zeros((), device=dev)\n    degenerate" not in pre


def test_degenerate_flag_forces_inf():
    src = open(SRC).read()
    assert "degenerate = True" in src, "the zero-organ break must record that it happened"
    assert 'float("inf") if degenerate else float(loss)' in src, (
        "a degenerate plant must report +inf, not the stale loss of the previous (non-empty) step")
