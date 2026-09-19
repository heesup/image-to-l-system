---
title: "Merging Stage 2 and 3: the hybrid absolute flow, and why it costs ~5 points (2026-09-19)"
date: 2026-09-19
tags: [experiment, architecture, stage2, stage3, flow-matching]
status: active
---

# Merging Stage 2 and 3: the hybrid absolute flow

**Status**: implemented, trained in two independent runs, evaluated under Gate G. **Both acceptance
criteria fail.** One positive result survives and is the thread worth pulling.

**Question asked**: Stage 3's refinement of Stage 2 does not work well and sits behind a
`phytomer_pos.detach()` barrier. Should the two stages be merged into a single generative model?

---

## 1. What was built

Stage 1 (DAP, node count K) is unchanged. Stages 2 and 3 become **one rectified flow** over the node
set, and Stage 3 in the new numbering is decode + render + self-consistency.

    x1 = [ pos_abs(3) | rot6d(6) | scale(3) | latent(D) ]   per node, 12 + D
    x_t = (1-t) * x0 + t * x1 ,  the model predicts  v = x1 - x0

The state is **absolute**, not parent-relative. That is the design decision the whole experiment
turns on:

- **No chain is resolved while sampling.** A node's position depends on no other node, which removes
  the error compounding down the chain at inference AND the gradient accumulation back up it during
  the render pass -- the likeliest source of the 1e13-vs-1e9 asymmetry that motivated the Stage 2/3
  barrier on 2026-09-12.
- **Rotation widens from roll(2) to rot6d(6).** Parent-relative could carry only the twist because
  the forward axis came from the resolved chain; with no chain there is nothing to express it
  relative to. rot6d is also the part tensor's own format (`FM_ROT` 16..22), so the state maps onto
  it without conversion.
- **The botany moves from the state into the loss**: internode length (distance to the GT parent) and
  forward-axis agreement (rot6d's x column along the branch). The axis target is **detached** --
  undetached it would couple a child's rotation error into its parent's position and reintroduce
  exactly the up-the-chain accumulation the absolute layout exists to remove.

`phytomer_pos.detach()` disappears for a principled reason rather than by fiat: the conditioning a
node's shape channels see is now an interpolant of noise and ground truth -- data, with no upstream
module's parameters behind it -- rather than another stage's live prediction.

Flags: `--stage3_absolute`, `--stage3_botany_weight` (launcher: `STAGE3_ABSOLUTE=1`,
`STAGE3_BOTANY_WEIGHT`), nested under `--stage3_geometry`.

## 2. The 2x2 that isolates the merge

The obvious comparison -- merged versus the appearance-augmented continuation -- is confounded,
because that control has `--appearance_augment` and Gate C showed augmentation is worth several
points on its own. So a second merged run was launched **with** augmentation, giving a clean cell:

| | plain | + augmentation |
| :--- | ---: | ---: |
| **unmerged** | (not run) | **control** `sub10_v10_cam_aug` |
| **merged** | `merged_abs` | `merged_aug` |

All three continue from the same `hierarchical_fm_v10_cam/hierarchical_fm_epoch_160.pt`, same 10k
subset, same eval and holdout sets, same LR.

**Result, holdout silhouette IoU:**

| run | epochs of continuation | last-10 mean | best |
| :--- | ---: | ---: | ---: |
| `merged_abs` (merged, plain) | 48 | 37.5% | 40.4% |
| `merged_aug` (merged, augmented) | 32 | 36.0% | 38.9% |
| `control_aug` (unmerged, augmented) | 73 | **41.2%** | **44.8%** |

**Merge effect with augmentation held constant: −4.8 points** over 32 matched epochs
(ep161–192), and stable across readings at −2.6, −5.7, −5.1, −4.8 as epochs accumulated.

Two things worth noting beyond the headline. Augmentation buys the merged model **nothing**
(36.0 vs 37.5, i.e. slightly worse), where it is worth several points to the unmerged one --
consistent with the merged model being limited by something appearance robustness cannot fix. And
both merged runs are **flat**, not climbing: 70 combined epochs with no upward slope.

## 3. Gate G: the one positive result

Multi-hypothesis render-and-select (`--n_starts 8 --steps 200`) on the same 20-plant eval set:

| | raw | refined | **lift** |
| :--- | ---: | ---: | ---: |
| unmerged, `n_starts 8` (the 74.9 record) | ~38.7 | **74.9** | +36.2 |
| merged `ep190_ema`, `n_starts 8` | 35.3 | **73.0** | **+37.7** |

The merged model starts 3.4 points lower and ends only 1.9 lower, so **it recovers more of its
deficit under refinement than the unmerged model does**. That is the single piece of evidence
supporting the merge's rationale, and it is the predicted direction: if the merged flow's *proposal
distribution* is better shaped even though its *point estimate* is worse, refinement should exploit
it. It is also what the 2026-09-18 conditioning probes implied would matter, having found per-organ
shape unrecoverable from a nadir view at any resolution.

## 4. Verdict

- **Criterion 1 (trains clean, holdout >= baseline): FAILED.** −4.8 points, two independent runs
  agreeing, flat rather than converging.
- **Criterion 2 (Gate G > 74.9): FAILED narrowly**, 73.0. It depended on raw quality recovering,
  which across 70 combined epochs it did not.

The "fresh rot6d dims are still converging" explanation was offered three times during the run and
should be retired: 48 epochs in one run and 32 in the other is enough opportunity.

**What this does not establish.** It does not show the *idea* is wrong, only this instantiation. The
merged model pays for making position generative when the 2026-09-18 probes showed position is the
one thing the conditioning **does** carry (R² 0.27, and 0.34 for occluded nodes) while shape is not
(~0.06). Imposing one stochastic treatment on two quantities of different epistemic status is a
plausible reading of the −4.8, and it argues for the split the architecture already had:
deterministic where the observation determines the answer, generative where it does not.

**What to pull on instead.** The refinement-lift result says the proposal distribution is the useful
axis, not the point estimate. That points at Gate G's mechanism (sample N, refine, select), at
`--spread_weight` (implemented, never launched), and at shape statistics -- hull ratio, boundary F --
rather than at further architectural merging.

## 5. Reproducing

```bash
# merged, plain
STAGE3_GEOMETRY=1 STAGE3_ABSOLUTE=1 STAGE3_BOTANY_WEIGHT=1.0 \
INIT_CHECKPOINT=outputs/checkpoints/hierarchical_fm_v10_cam/hierarchical_fm_epoch_160.pt \
EPOCHS=280 OUTPUT_DIR=outputs/checkpoints/merged_abs bash scripts/train_hierarchical_flow_matching.sh

# merged, augmented (the clean cell)
... APPEARANCE_AUGMENT=1 AUGMENT_P=0.8 SOIL_BANK=dataset/soil_bank_agml \
    OUTPUT_DIR=outputs/checkpoints/merged_aug ...

# Gate G on a merged checkpoint
python plant_recon/eval/eval_test_time_refinement.py \
    --checkpoint outputs/checkpoints/merged_abs/hierarchical_fm_epoch_190_ema.pt \
    --eval_set outputs/checkpoints/hierarchical_fm_v9/eval_set.json --steps 200 --n_starts 8
```

**Traps met while building this**, each silent in the training log and each caught only by checking
saved args rather than output:

- `--stage3_absolute` passed via `EXTRA_ARGS` was **discarded** -- the launcher overwrites that
  variable rather than appending to it -- and the run trained the parent-relative path while
  reporting success.
- The geometry velocity loss sliced a hardcoded `STAGE3_GEOM_DIM = 8`, dropping 4 of the absolute
  layout's 12 dims.
- Warm start: `v10_cam ep160` is itself 136-wide, so the generic "copy into the LAST slice" rule
  shifted everything by 4 and landed old `dpos`/`roll` in the new rot6d slots. Scale and latent still
  lined up by coincidence, which is what made it hard to see. Blocks are now mapped explicitly.
- The **EMA** shadow is loaded separately and strictly, so the width change aborted the whole job.
- **Five eval scripts** rebuild the model from stored args and none threaded `stage3_absolute`, so
  every one of them failed to load a merged checkpoint until patched.
- `EPOCHS=120` against a checkpoint restoring to epoch 161 produced an empty training loop that
  exited **COMPLETED** in 34 seconds.
