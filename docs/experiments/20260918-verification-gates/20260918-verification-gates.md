---
title: "Verification gates: appearance gap closed by the augmented fine-tune (2026-09-18)"
date: 2026-09-18
tags: [experiment, verification, appearance, refinement, depth-input]
status: active
---

# Verification gates: appearance gap closed by the augmented fine-tune

**Status**: the four synthetic verification gates for the 2026-09-18 plan (C appearance, D refinement
safety, G multi-hypothesis selection, B depth input). Gates C and D are read; G and B were still
running at writing. Protocol everywhere: `eval_test_time_refinement.py`, 200 steps (the converged
count, 66.3 -> 73.7 over 40/80/200), 20 eval plants, strict 256 px, EMA checkpoints.

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

## Gate D — refinement existence/scale safety: PARTIAL (verdict pending per-plant reads)

Control reproduced the baseline (35.6 -> **72.4**, within noise of the recorded 73.7 — also the
check that the Phase G refactor did not break the single-start path). The new flags —
continuous existence with the materialisation threshold aligned to the search (0.4, was a hardcoded
0.5 that ignored `--exist_thresh`) plus absolute realized-scale ceilings
(`--scale_abs_max_len 5.0 --scale_abs_max_rad 0.3`) — scored **71.2** against that control: the
threshold alignment recovered most of the recorded -1.8 but existence optimisation still does not
pay on the clean protocol. The per-plant node counts (in the JSON) say whether the active-node deficit
(60/73) moved at all; the verdict needs that read, not just the mean IoU.

## Gate G — multi-hypothesis render-and-select (`--n_starts 8`): RUNNING

8 flow samples per plant, each refined 200 steps against the input, best-by-input-loss kept. ~8x
the control's cost; verdict when job 38430189 finishes.

## Gate B — depth input (`--use_depth` fine-tune from ep160): RUNNING

Single-variable complement of Gate C (depth on, augment off; the aug run is augment on, depth off).
Job 38430694 on gpu-10-50; the chained eval polls for an ep185+ EMA and then runs the same
flat-vs-Helios protocol. Gate: the Helios drop must shrink toward <5 points raw — depth is
appearance-independent, so it should carry the height signal where RGB degrades. The 2026-09-18
conditioning report found depth redundant against *clean* synthetic RGB (+0.055 vs +0.057 R^2); this
gate tests the *degraded-RGB* case, which that report explicitly left open (§5).

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