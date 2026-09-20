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

---

## Final verdict at epoch 280 (2026-09-19, replicated)

`merged_abs` finished its 280 epochs. Evaluated on the shared 20-plant set (its own `eval_set.json`
is exactly `hierarchical_fm_v9/eval_set.json` — all 20 prefixes overlap, so this is a head-to-head
on identical plants), `--steps 200 --sample_seed 0`, with replicates throughout because the
refinement is nondeterministic (see
[the refinement report](../20260919-refinement-levers-are-seedling-levers/20260919-refinement-levers-are-seedling-levers.md)).

| | n | raw | refined | SD | lift |
| :--- | ---: | ---: | ---: | ---: | ---: |
| baseline (v10_cam ep160) default | 4 | 37.9 | 71.23 | 0.43 | +33.3 |
| baseline **+ Gate G** | 5 | 38.0 | **74.65** | 1.02 | +36.7 |
| merged default | 5 | 33.2 | 73.34 | 1.12 | +40.2 |
| merged + Gate G | 5 | 34.7 | 73.56 | 0.61 | +38.8 |
| merged, held-out plants | 2 | 27.9 | 71.93 | 1.62 | +44.0 |

| comparison | delta | SE | verdict |
| :--- | ---: | ---: | :--- |
| merged vs baseline, default | +2.11 | 0.55 | **clear** (3.9 SE) |
| merged vs baseline, with Gate G | −1.09 | 0.53 | suggestive (2.1 SE) |
| Gate G on baseline | +3.42 | 0.50 | **clear** (6.8 SE) |
| Gate G on merged | +0.22 | 0.57 | **nothing** (0.4 SE) |

**Gate G succeeds; the merged architecture does not.** The goal was for both to succeed and only one
did.

Three findings, in order of how well they are established.

**1. Gate G helps the baseline and does nothing for the merged model.** +3.42 at 6.8 SE against
+0.22 at 0.4 SE — the interaction is unambiguous even though the head-to-head under Gate G (−1.09,
2.1 SE) is not. This is the cleanest result of the comparison.

The mechanism is *not* collapsed sample diversity: both models spread their selections over all 8
starts (baseline histogram 18/11/11/10/13/16/8/13, merged 23/7/6/21/9/20/6/8). It is that the merged
model's refinement is **start-insensitive**. Selection improves its raw score (33.2 -> 34.7) but buys
almost nothing refined (+0.2), whereas on the baseline selection barely moves raw (37.9 -> 38.0) and
gains +3.4 refined. The baseline's outcome depends on where it starts, so choosing well pays; the
merged model converges to ~73.5 from anywhere. Its larger lift (+40.2 vs +33.3) is the same fact
seen from the other side — it refines hard enough to wash out its starting point, and that caps it.

**2. The merged model still starts clearly worse.** Raw 33.2 vs 37.9, −4.7 points, reproducing the
criterion-1 failure (−4.8 to −5.7) at full training length. Absolute geometry in the flow state with
parent-relative supervision as a loss term did not fix the scaffold. Note the budget asymmetry:
`merged_abs` trained on `max_train_samples = 10000`, the baseline on the full dataset, so this
compares architectures at unequal budget, not outright.

**3. At default settings the merged model is genuinely ahead** (+2.11, 3.9 SE) — it converts a worse
start into a slightly better finish. That is a real result and it is the one that looked promising
before Gate G was applied to both. It does not survive giving the baseline the better inference
procedure.

**Best configuration on the table: baseline + Gate G, 74.65.**

Held-out: the merged model scores 71.93 refined against 73.34 in-sample, a gap not resolved at n=2,
but its **raw** held-out score is 27.9 against 33.2 in-sample — a clearer 5.3-point generalisation
gap in the feedforward prediction itself.


## Appearance augmentation: the clearest win of the day, but it does not close the gap

`merged_aug` is `merged_abs` plus `APPEARANCE_AUGMENT=1, AUGMENT_P=0.8`, everything else identical,
so the pair isolates augmentation. Evaluated on the flat cache render (the usual protocol) and on
Helios raytraced re-renders at the cache's framing (`--rgb_override_dir`, the appearance-gap
protocol), replicated.

| | n | raw | refined | SD |
| :--- | ---: | ---: | ---: | ---: |
| merged_abs, flat | 5 | 33.2 | 73.34 | 1.12 |
| merged_abs, **Helios** | 2 | 18.7 | **48.36** | 0.67 |
| merged_aug, flat | 3 | 34.9 | 73.37 | 0.38 |
| merged_aug, **Helios** | 2 | 22.0 | **59.11** | 0.42 |

**Augmentation is worth +10.75 on Helios at 19.2 SE** — the largest and least ambiguous effect
measured today. It costs nothing on the flat render (73.34 vs 73.37, identical).

The appearance gap, flat minus Helios on the refined score:

  merged_abs   -24.98
  merged_aug   -14.26

So augmentation removes about **43%** of the gap and leaves 14.3 points standing. Note the
asymmetry: the RAW gap barely moves (-14.4 -> -13.0) while the REFINED gap almost halves. The
network does not read shifted pixels much better; it produces starts that survive refinement under
a shifted appearance. Refinement lift on Helios goes +29.7 -> +37.1.

### The discrepancy, resolved: the merged architecture is what is brittle

The 2026-09-18 verification-gates report records Gate C as **PASS** with the appearance gap "CLOSED
on the refined path" for the non-merged appearance fine-tune: `sub10_v10_cam_aug` ep190 refined flat
71.0 / Helios 71.6. Nothing here reproduces that. `merged_aug` sits at 73.4 / 59.1.

`sub10_v10_cam_aug` ep230 was evaluated under exactly this protocol. **Gate C reproduces**, and the
comparison isolates the cause: the two checkpoints share `max_train_samples=10000`,
`appearance_augment=True, augment_p=0.8` and `stage3_geometry=True`, and differ **only** in
`stage3_absolute`.

| stage3_absolute | | n | raw | refined |
| :--- | :--- | ---: | ---: | ---: |
| None (plain) | flat | 2 | 38.5 | 72.65 |
| None (plain) | **Helios** | 2 | 30.4 | **70.16** |
| True (merged) | flat | 3 | 34.9 | 73.37 |
| True (merged) | **Helios** | 2 | 22.0 | **59.11** |

    appearance gap    plain  -2.49       merged -14.26
    Helios, plain minus merged   +11.05   SE 0.32   (+34.4 SE)
    flat,   plain minus merged    -0.72   SE 0.22   ( -3.3 SE)

**The absolute flow state buys +0.7 on flat synthetic renders and costs 11.1 points on realistic
appearance.** Both are statistically clear; they are not the same size. Since the project's actual
target is real imagery, the Helios column is the one that matters, and on it the plain lineage wins
by a margin no other lever measured today comes close to.

This also explains the earlier `merged_abs` results without needing the start-insensitivity story to
carry all the weight: an absolute-state model whose raw prediction degrades from 34.9 to 22.0 under
an appearance shift is reading pixels much less robustly, and refinement was compensating on flat
renders where the degradation does not appear.

**Consequence.** `merged_full` and `merged_fullaug` (full data, ~63 h each) are training the
architecture that loses on the metric that matters. They still answer the open data-budget question
— whether 10x the data changes any of this — but the case for promoting the merged architecture is
now much weaker than the flat-render numbers suggested.
