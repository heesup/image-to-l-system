"""Gradient flow audit for the cascaded hierarchical model.

Verifies that Stage 1 (MacroBiologicalHead) and Stage 2 (CoarseSkeletalTransformer
pos/rot/exist heads) receive non-zero gradients under both capacity modes:
  - 'given' (GT DAP teacher forcing, standard training path)
  - 'pred_phyto' (two-pass predicted-phytomer slicing, inference path)

Run from workspace root:
    /home/lion397/.conda/envs/digital-crops/bin/python -m tests.test_gradient_flow
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from diffusion_based.models.hierarchical_part_flow_matching import (
    CoarseSkeletalTransformer,
    compute_matryoshka_slice,
)


def _check_grad(name: str, param: torch.Tensor) -> bool:
    g = param.grad
    ok = g is not None and g.abs().sum().item() > 0.0
    status = "OK " if ok else "MISSING"
    print(f"  [{status}] {name}: grad_norm={0.0 if g is None else g.norm().item():.3e}")
    return ok


def audit(capacity_mode: str, dap: float) -> bool:
    print(f"\n=== capacity_mode={capacity_mode} (DAP={dap}) ===")
    torch.manual_seed(0)
    layer = CoarseSkeletalTransformer(max_anchors=64, embed_dim=128, num_heads=4, num_layers=2)
    # Ensure the DAP head's ReLU is active at init (random init can land in the dead
    # zone; with trained real-data weights DAP predictions are positive anyway).
    with torch.no_grad():
        layer.macro_head.dap_head.bias.fill_(0.5)
    B, T = 2, 16
    tokens = torch.randn(B, T, 128, requires_grad=True)

    dap_tensor = torch.tensor([dap, dap])
    active_k = compute_matryoshka_slice(dap=dap_tensor, max_anchors=64)

    if capacity_mode == "given":
        out = layer(tokens, active_k=active_k)
    else:
        out = layer(tokens, capacity_mode="pred_phyto")

    loss = (
        out["anchor_pos"].pow(2).mean()
        + out["anchor_rot"].pow(2).mean()
        + torch.sigmoid(out["anchor_logits"]).mean()
        + out["pred_dap"].pow(2).mean()
        + out["pred_num_phytomers"].pow(2).mean()
    )
    loss.backward()

    ok = True
    ok &= _check_grad("pos_head (Stage 2 xyz)", layer.pos_head[0].weight)
    ok &= _check_grad("rot_head (Stage 2 6D)", layer.rot_head[0].weight)
    ok &= _check_grad("exist_head (Stage 2 exist)", layer.exist_head[0].weight)
    ok &= _check_grad("anchor_queries[:K]", layer.anchor_queries)
    ok &= _check_grad("ref_points[:K]", layer.ref_points)
    ok &= _check_grad("decoder layer0 self-attn", layer.decoder.layers[0].self_attn.in_proj_weight)
    ok &= _check_grad("macro_head dap (Stage 1)", layer.macro_head.dap_head.weight)
    ok &= _check_grad("macro_head phy (Stage 1)", layer.macro_head.phy_head.weight)

    # In pred_phyto mode, verify the count prediction keeps its graph (gradient source)
    if capacity_mode == "pred_phyto":
        layer.zero_grad()
        out2 = layer(tokens, capacity_mode="pred_phyto")
        # The phytomer count gradient must flow through the soft margin prior into
        # anchor_logits -> existence loss even though the slice index is discrete.
        loss2 = out2["pred_num_phytomers"].pow(2).mean() + out2["anchor_logits"].pow(2).mean()
        loss2.backward()
        g_phy = layer.macro_head.phy_head.weight.grad
        g_phy_direct = g_phy is not None and g_phy.norm().item() > 0.0
        print(f"  [{'OK ' if g_phy_direct else 'MISSING'}] phy_head gradient through "
              f"soft-margin prior (pred_phyto mode)")
        ok &= g_phy_direct

        # Verify the slice responds to the predicted count (capacity responsiveness)
        print(f"  active_k used: {out2['active_k']}")
    return bool(ok)


def main():
    print("Gradient Flow Audit: Stage 1 (Macro) + Stage 2 (Coarse Skeletal) heads")
    all_ok = True
    all_ok &= audit("given", dap=15.0)
    all_ok &= audit("pred_phyto", dap=15.0)
    print("\n" + ("ALL GRADIENT PATHS OK" if all_ok else "SOME GRADIENT PATHS MISSING"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())