---
title: "Developer Onboarding — Where to Start (Plant Architecture Reconstruction)"
date: 2026-09-16
tags: [handover, onboarding, getting-started]
status: active
---

# Developer Onboarding — Where to Start

**One page for the next agent/developer picking up this repo.** Project: single RGB-D drone image
→ 3D plant architecture (phytomer-level Flow Matching + differentiable rendering + Helios C++ XML).
All paths use the post-restructure layout (see [`docs/architecture/code-structure/code-structure.md`](../architecture/code-structure/code-structure.md)).

---

## 1. The repo in 60 seconds

| Where | What |
| :--- | :--- |
| `plant_recon/` | The library — `models/` (Stages 0–4, renderer, XML export), `training/`, `dataset/`, `eval/` |
| `use_cases/real_world/` | Sim-to-real application: detectors, real-field datasets, multi-plant pipeline |
| `submodules/Digital-Crops/` | Helios C++ OptiX engine (git submodule, pinned) |
| `dataset/` | Data only: 100k Helios XMLs + `.pt` caches (gitignored) |
| `outputs/` | Everything generated: `checkpoints/`, `logs/`, `wandb/`, `weights/`, `eval/` |
| `scripts/` | The 3 cluster launchers (`.sh` only) |
| `docs/handovers/` | `current-status.md` (live dashboard) + `agent-takeover-guide/` (master manual) |

**Environment**: `mamba activate digital-crops`, `export PYTHONPATH=.` from the repo root.
**Sanity check**: `pytest tests/` → 97 passed (2 pre-existing stale failures: `test_dinov2_3d_spatial.py`,
`tests/unit/test_ik_fix.py` — old model API, not your problem unless you touch them).

## 2. Current state (2026-09-16) — read this before writing code

- **Synthetic track has plateaued.** Strict-protocol P sits at 33–39 no matter which training lever
  was pulled. This is the *finding*: it's an architecture/information limit, not a tuning problem.
- **Test-time refinement is the biggest lever found**: 40 Adam steps on node positions/scales/latents
  against the input CHM lift the same checkpoints to **P 67–68** — no retraining
  (`plant_recon/eval/eval_test_time_refinement.py`; defaults are already the winning recipe:
  `--input_camera --reg_scale 5 --reg_latent 0.5 --target_zooms 1,2,4,8`).
- **Best model**: `outputs/checkpoints/hierarchical_fm_v10_cam/hierarchical_fm_epoch_160_ema.pt`
  (raw 38.7 / refined 68.3). VAEs: `phytomer_vae_v9_tl_rw4_20k` (default), `organ_vae/`.
- **Real-image track runs end-to-end but is not geometrically usable yet.** Detection is solved
  (AgML box detector mAP50 0.975). The failure is the network reading RGB appearance: on Helios
  raytraced pixels (identical geometry!) raw P drops 38.1 → 28.6 and refined 66.3 → 59.2, with the
  DAP probe and the existence gate collapsing on seedlings/mature plants. That single measurement
  (`docs/handovers/20260916-sim-to-real-assessment/20260916-sim-to-real-assessment.md` §5) says the
  gap is *pixels*, not structure.

## 3. Where to start developing (ranked)

### A. Close the appearance gap in training — the top item

The network reads **RGB only** (`DINORayEncoder.forward` takes channels 0:3); training RGB is
flat-shaded green on uniform tan with **zero augmentation**, while 100k Helios raytraced renders
(`dataset/helios_data/cowpea/*_rad.jpeg`, textured soil, shadows, specular) sit unused.

1. Fine-tune from `hierarchical_fm_v10_cam` ep160 on Helios raytraced crops at the cache's fixed
   1.2 m window framing (`plant_recon/eval/render_helios_eval_crops.py` shows the exact recipe).
2. Add ordinary augmentation: colour jitter, blur, sensor noise, random shadows, real soil
   composites under the flat render mask.
3. Re-measure with the same flat-vs-Helios protocol used in the assessment (§4 there) — that is the
   number that must move. Watch the **DAP probe error** and **active-node count**, not just P.

### B. Fix the real-image crop framing (one line, then a measurement)

`use_cases/real_world/dataset/real_plant_crop_utils.py::build_pyramid_16ch` uses 1.2× the *detector
bbox* as the zoom-1× window; the training cache uses a **fixed 1.2 m ground window**. Every plant on
a real crop therefore looks ~4× too big, and Stage 1's DAP head reads size in a fixed window — the
reported "DAP uncorrelated with truth" follows. Fix: `window_px = 1.2 m × (frame_px / plot_width_m)`
(the multi-plant script already assumes the 1.3 m plot width).

### C. Refinement safety on real crops

- **Bound realized organ size in metres**, not the phytomer scale `s_a` — refinement currently
  inflates leaf scale 2.4–2.6× against a loose blob-mask target.
- **Make existence continuous** in refinement (soft compositing weight on Stage 2's logits) so the
  optimiser can switch nodes on/off; lowering the threshold alone adds false positives it can't remove.

### D. Lower-priority / don't bother

- **Stop investing in the Stage 3 latent.** Refinement from a mean latent reaches the same 66.5 as
  the sampled latent; the network's real contribution is node positions/existence/topology. If
  Stage 3 stays, a deterministic regression head is enough.
- **Guided sampling is not worth it** (neutral alone, *hurts* when chained with refinement).
- Larger backbone with multizoom (`dinov2_vitb14`) is one flag away if node error (4–6 cm at 7.5 cm
  per token) ever becomes the binding constraint.
- Use 50–100 eval plants for decisions — epoch spread is ±3 points on the current 20-plant set.

## 4. Everyday workflows

```bash
# Evaluate a checkpoint under the strict protocol + refinement
python plant_recon/eval/eval_test_time_refinement.py --checkpoint outputs/checkpoints/hierarchical_fm_v10_cam/hierarchical_fm_epoch_160_ema.pt

# Self-consistency panels
python plant_recon/eval/eval_hierarchical_self_consistency.py --help

# Train (cluster): one launcher, env overrides for every knob
sbatch scripts/train_hierarchical_flow_matching.sh            # current v9/v10 recipe
TRAIN_VAE=1 sbatch scripts/train_hierarchical_flow_matching.sh # fresh PhytomerVAE first

# Real image: one frame -> detections -> per-plant XML -> refinement -> rendered plot
python use_cases/real_world/eval/run_multiplant_scene.py --image <frame.jpg>
```

## 5. Gotchas worth memorising

- `PYTHONPATH=.` from the repo root is mandatory; the package imports as `plant_recon.*`.
- CHM depth = height **above ground** (ground 0, apex max) — inverted vs camera distance.
- Helios XML with any scale/radius ≤ 0 segfaults the C++ binary; clamp at export.
- Packet cache order must match the VAE (`PHYTOMER_TERMINAL_LAST=1` ⇔ `_pkt_v9` cache). Never edit
  cache files in place — bump `PKT_VERSION` and regenerate.
- Don't stack a foreground eval on the same GPU as a live local training run (two stalls this week).
- Never cancel Heesup's `regen_*` jobs or the OnDemand desktop jobs; training goes on
  `gpu-6000_ada-h`/`low`.

## 6. Further reading (in order)

1. [`docs/handovers/current-status.md`](current-status.md) — live dashboard
2. [`docs/handovers/agent-takeover-guide/agent-takeover-guide.md`](agent-takeover-guide/agent-takeover-guide.md) — master manual
3. [`docs/handovers/20260916-sim-to-real-assessment/20260916-sim-to-real-assessment.md`](20260916-sim-to-real-assessment/20260916-sim-to-real-assessment.md) — why the next effort goes where it does
4. [`docs/handovers/20260912-stage2-stage3-boundary/20260912-stage2-stage3-boundary.md`](20260912-stage2-stage3-boundary/20260912-stage2-stage3-boundary.md) — architecture record
5. [`docs/architecture/current-architecture/current-architecture.md`](../architecture/current-architecture/current-architecture.md) — the math
