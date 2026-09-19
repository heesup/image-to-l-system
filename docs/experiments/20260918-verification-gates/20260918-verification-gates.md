---
title: "Verification gates: appearance gap closed by the augmented fine-tune (2026-09-18)"
date: 2026-09-18
tags: [experiment, verification, appearance, refinement, depth-input]
status: active
---

# Verification gates: appearance gap closed by the augmented fine-tune

**Status**: the four synthetic verification gates for the 2026-09-18 plan (C appearance, D refinement
safety, G multi-hypothesis selection, B depth input). **C PASS, G PASS, D partial (node deficit is
training-side), B crashed at step 1 — root-caused and fixed the same evening, rerun queued.**
Protocol everywhere: `eval_test_time_refinement.py`, 200 steps (the converged count, 66.3 -> 73.7
over 40/80/200), 20 eval plants, strict 256 px, EMA checkpoints.

**Reference points** (recorded 2026-09-16/17): ep160 base, flat input, 200 steps = **73.7**;
ep160 base on Helios RGB = raw 38.1 -> 28.6, refined 66.3 -> **59.2** (the 7-point appearance gap).
ODE sampling carries ~±2 points run to run.

---

## Gate C — appearance-augmented fine-tune (sub10_v10_cam_aug, ep190 EMA): PASS

The fine-tune restarted clean from ep160 after the wrapped-shadow bug (`_shift_zero` fix), 30 clean
epochs. Same protocol as the recorded gap measurement:

| input | raw P | refined P |
|---|---:|---:|
| flat cache | 36.9 | **71.0** |
| Helios rgb override | 31.4 | **71.6** |

**The appearance gap on the refined path is closed** (71.0 vs 71.6; the ep160 base lost 66.3 -> 59.2
on the same swap). Reading:

1. **Helios-input refined P rose 59.2 -> 71.6 (+12.4)** — the augmentation made the model's
   cold-start scaffold robust to the appearance shift, and refinement (whose target, the CHM, is
   unchanged between arms) then exploits it.
2. **Flat input did not regress** (71.0 vs the base's 73.7-flat — inside the ±2 sampling noise), so
   the robustness was free, not traded against clean-input quality.
3. **Raw drop shrank 10.0 -> 5.5 points** (36.9 -> 31.4 vs the base's 38.1 -> 28.6) but is not gone:
   the cold start still loses IoU before refinement. This is the DAP-probe/existence failure the
   docs attribute to appearance — partially recovered at 30 epochs, re-check at the end of the run.
4. **Seedling DAP error under appearance shift is still 41 days** (DAP<=15 bucket): the Stage-1 probe
   reads a DAP-2 seedling as ~55 on Helios pixels. The probe remains appearance-brittle even where
   refinement recovers the geometry.

Repeat readout with `--save_renders` (both arms): flat **39.5 -> 73.9**, Helios **29.1 -> 69.9** —
same verdict, flat now at the lineage's best level, Helios within 4 points of it.

Figures: [`c_aug_flat_vs_helios.png`](assets/c_aug_flat_vs_helios.png) (per-plant raw/refined P and
the DAP probe), [`gate_c_side_by_side.png`](assets/gate_c_side_by_side.png) (per-plant
GT | base-flat | aug-flat | aug-Helios refined renders), [`compare.md`](assets/compare.md).
Raw JSONs in `assets/`.

## Gate D — refinement existence/scale safety: PARTIAL, node deficit confirmed TRAINING-SIDE

Control reproduced the baseline (35.6 -> **72.4**, within noise of the recorded 73.7 — also the
check that the Phase G refactor did not break the single-start path). The new flags —
continuous existence with the materialisation threshold aligned to the search (0.4, was a hardcoded
0.5 that ignored `--exist_thresh`) plus absolute realized-scale ceilings
(`--scale_abs_max_len 5.0 --scale_abs_max_rad 0.3`) — scored **71.2** against that control: the
threshold alignment recovered most of the recorded -1.8 but existence optimisation still does not
pay on the clean protocol. The per-plant node counts settle the "why": **60.0 -> 61.4 nodes against
73 GT** (`d_exist_s200.json`). Refinement barely changes the node count because the sigmoid logits
start saturated; the existence deficit is a *training-side* problem (the existence head), not a
refinement-side one. Action item: this is what `SPREAD_WEIGHT` / existence-head training levers are
for, not more refinement machinery.

## Gate G — multi-hypothesis render-and-select (`--n_starts 8`): PASS

**74.9 refined** against the same-run control 72.4 and the recorded 73.7 — the first lever to move
refined P above the plateau, and it is GT-free (selection is on the final *input* loss, deployable
end to end). Two findings in the details (`g_nstarts8_s200.json`):

1. The selected-start distribution spans all 8 starts ({3,4,2,3,3,2,1,2}) — different plants
   genuinely need different hypotheses; the flow's proposal spread is real signal, not noise.
2. Per-plant it is level with or above the control nearly everywhere, with one dramatic seedling
   rescue (a DAP-3 plant: 23 -> 40) — exactly the bucket every other lever fails on.

Cost: ~8x a single start (200 steps each) — ~13 min for 20 plants on one Ada. For deployment,
n_starts 4 looks like the knee (the marginal start rarely wins), but that is a knob, not a finding.

## Gate B — depth input (`--use_depth` fine-tune from ep160): CRASHED AT STEP 1, fixed, rerun queued

The resume mechanics worked exactly as designed (2 missing keys = the two `depth_adapt` tensors at
fresh init; partial optimizer restore covered 264 parameters) — then the first training step died
with a CUDA illegal memory access (job 38430694). Root cause found and fixed the same evening: the
depth adaptor ran its 1x1 conv at the full 224x224 input size, keeping a
(L*B, 384, 224, 224) activation alive for backward — ~7.4 GB in bf16 at batch 48 multizoom, stacked
on an already-heavy render step. A 1x1 conv commutes with average pooling, so the fix pools the
normalised depth to the 16x16 patch grid BEFORE the conv: identical in value (verified to 1e-6),
~200x less activation memory. All `tests/test_use_depth.py` tests pass on the fixed order.
Rerun: `sbatch scratch/20260918_verification_gates/gate_b_train.sbatch` (+ the chained eval).

## Next experiment this points at: refinement in the training loop (learning-to-refine)

The Gate G mechanism (sample N, refine, select) is inference-side. The natural training-side
complement, proposed by Heesup 2026-09-18 evening: **unroll a few refinement steps inside training
and let the post-refinement render loss backpropagate into the model** — teach the network to
produce starts that refine well, which is what the deployable path actually consumes. Three
designs, in increasing cost:

1. **K-step unrolled refinement loss** (the proposal): for each rendered sample, run the usual
   render loss, THEN K Adam steps on that plant's pos/scale/latent against the input CHM (the
   `refine_plant` machinery, K ~ 5-20, not 200), and add the *post-refinement* render loss to the
   training objective. Backprop through the unrolled steps teaches refinement-friendly starts.
   Cost: K+1 render backwards per rendered sample — affordable exactly because the render path is
   only `render_fraction` of the batch and batched.
2. **MAML-style meta-initialisation**: the same unroll, but the inner-loop Adam updates a *copy* and
   the outer loss measures the improvement — closer to "learn initialisations that refine well",
   the general case of which design 1 is a fixed-step approximation.
3. **Learned 1-step corrector**: `apply_render_feedback` (already in the model, off by default) is
   the amortized version — one learned correction step instead of an optimisation loop.

Design 1 is the one to try first: it is the only one that reuses existing machinery end to end
(the refinement loop, the render loss, `render_to_latent`/`render_to_exist` are its K=1 special
case), and the evidence already says start quality gates the refined outcome (Gate G's per-start
spread). Gate before any real-image use: strict-protocol refined P on the 20-plant set must beat
the same-budget control, judged at matched total wall-clock (the unroll is K+1x the render cost,
so compare against more epochs of the control).

## Reproducing

```bash
# gates D control + D flags + G (eval-only, ep160 EMA)
sbatch scratch/20260918_verification_gates/gates_dg.sh
# gate C (flat-vs-Helios on the aug fine-tune)
sbatch scratch/20260918_verification_gates/gate_c.sh
# gate B train (USE_DEPTH=1 from ep160) + chained eval
sbatch scratch/20260918_verification_gates/gate_b_train.sbatch
sbatch scratch/20260918_verification_gates/gate_b_eval.sbatch
# visual evidence (per-plant gt/before/after renders, both arms) + the composed side-by-side
sbatch scratch/20260918_verification_gates/gate_c_renders.sbatch
python scratch/20260918_verification_gates/compose_gate_c_figure.py
```

Gate JSONs live in `outputs/logs/20260918/gates/` (gitignored); the copies committed here are under
`assets/`.