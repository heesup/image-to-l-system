# The absolute layout's botany loss aligned the wrong rot6d column

2026-09-20. Found while building the generative-structure probe: ground-truth plants showed a
forward-axis deviation of 90 degrees, which cannot be right, and the reason was the probe and the
training loss making the same wrong assumption about the rot6d convention.

## The convention

`roll_to_matrix` returns `stack([x, forward, z], dim=-1)` — **columns** `[x, forward, z]` — and
`matrix_to_rot6d` keeps the first two **columns**. So a rot6d vector is `[x | forward]`:

    rot6d[..., 0:3]   the x column, PERPENDICULAR to the branch
    rot6d[..., 3:6]   the forward column, ALONG the branch

Measured over 167 ground-truth internodes, as the angle to the parent->node direction:

    rot6d[0:3]   mean 90.2 deg   median 90.0
    rot6d[3:6]   mean  1.1 deg   median  0.3

## The bug

The hybrid's botany term — the loss that exists to put back the structure the relative layout
carries in its coordinate system — used the x column:

```python
# forward axis: rot6d's x column (rot6d_to_matrix normalises d6[..., 0:3]) should
# point along the branch the node actually sits on
fwd_hat = F.normalize(rot_hat[..., 0:3], dim=-1, eps=1e-6)
loss_axis = (1.0 - (fwd_hat * fwd_tgt).sum(-1)).mean()
```

The comment's parenthetical is true and misleading: `rot6d_to_matrix` really does normalise
`d6[..., 0:3]` into the x column. That does not make the x column the forward axis.

So the term drove each node's perpendicular axis toward the branch direction — rotating the node a
full **90 degrees away from correct**, with weight 1.0, for every run that set
`--stage3_botany_weight`. The loss meant to restore the relative layout's botany was fighting it.

## Scope

Affects **absolute runs only** — the block is guarded by `if s3abs and total_matched_nodes > 0`, so
the relative lineage never evaluated it. Every `merged_*` run trained under it:
`merged_abs`, `merged_aug`, `merged_full`, `merged_fullaug`.

This is a plausible contributor to the absolute layout's measured deficit. It does not explain the
deficit away — the raw appearance gap and the in-domain/out-of-domain result stand on their own —
but **the absolute layout has never been trained with a correct botany constraint**, so its numbers
are a floor, not a verdict.

## Fixed

`rot_hat[..., 3:6]`, with `tests/test_botany_forward_axis.py` pinning both that rot6d[3:6] IS the
forward axis (constructed from `roll_to_matrix` / `matrix_to_rot6d`, not asserted) and that the
training loss uses it.

## The runs, restarted (2026-09-20)

`merged_full` and `merged_fullaug` were cancelled and relaunched as **`absolute_full`** and
**`absolute_fullaug`**, from `v10_cam` ep160 — **not** resumed from their own ep170. Those ep170
checkpoints carry ten epochs trained under the 90-degree-wrong term; resuming them would have
carried that forward into the very comparison the runs exist to settle. Ten epochs (~10 h each) is
the price of a clean start, and the runs were only 1.5 h past that point anyway.

The new names also retire the `merged_*` misnomer: every checkpoint in this lineage is merged
Stage 2+3, and what these runs vary is the **absolute** layout.

`outputs/checkpoints/merged_full/` and `merged_fullaug/` are left on disk for reference. **Do not
resume or evaluate from them** — everything past ep160 in those directories was trained with the
broken botany term.

`relative_fullaug` was left running: the botany block is guarded by `if s3abs`, so the relative
lineage never evaluated it.
