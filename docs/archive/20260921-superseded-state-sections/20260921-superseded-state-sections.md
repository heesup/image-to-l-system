---
title: "Superseded state sections (handover guide + current-state)"
date: 2026-09-21
tags: [archive]
status: archived
---

# Superseded state sections

Moved out of `agent-handover-guide.md` and `current-state.md` on 2026-09-21, when the docs were
given one job each: `_index.md` is the entry point, `current-state.md` holds the evidence, and the
handover guide is the durable manual.

These are dated snapshots — several of them said "read this first" while contradicting one another.
They are kept because they carry reasoning and job numbers that the live documents no longer need,
not because anything here should be trusted as current.

## 0-C. State as of 2026-09-16 (newest; read this first)

The live dashboard is [`current-status.md`](../current-state/current-state.md); the plan and the measurement of the day are in
[`20260916-sim-to-real-assessment.md`](../experiments/20260916-sim-to-real-assessment/20260916-sim-to-real-assessment.md).
What §0-A and §0-B do not know:

- **Synthetic track (2026-09-15/16).** The `hierarchical_fm_v10_cam` lineage (v10 flags + `RENDER_INPUT_CAMERA=1` + EMA,
  from the s3geom ep78 checkpoint) ran to ep160 on the cluster (`38275054` → `38279147` → `38332343`, checkpoints
  `outputs/checkpoints/hierarchical_fm_v10_cam/`). Strict-protocol P plateaus at 33–39 for every training lever tried
  (Stage 3 geometry, teacher forcing, latent normalisation, multizoom, node-token window, existence-count weight,
  render-to-latent / -exist, a render-fraction curriculum, 10% vs 100% of the data). **Test-time refinement**
  (`plant_recon/eval/eval_test_time_refinement.py`: 40 Adam steps on nodes / scales / latents against the input CHM;
  defaults `--input_camera`, four zoom targets, `reg_scale 5`, `reg_latent 0.5`) lifts ep160 EMA from 38.7 to **68.3**
  and is the deployable path; the flow-sampled latent adds nothing over the mean latent once refinement runs; guided
  sampling inside the ODE does not compose with it. Chronology: §0-B.9 below and the 2026-09-14 report §11.
- **Real-image track (2026-09-15/16).** `use_cases/real_world/`: Roboflow and AgML GEMINI cowpea sources, YOLO
  detectors (mAP50 0.84 / 0.975), Depth Anything pseudo-CHM, per-crop cold start and refinement, and the whole-frame
  multi-plant pipeline (`eval/run_multiplant_scene.py`). Nothing is geometrically usable yet: the network's cold start
  collapses to a single phytomer on most real crops and refinement inflates leaves against loose targets. Reports:
  `docs/experiments/20260915-real-image-first-test/`, `20260916-agml-dataset-swap/`, `20260916-multiplant-scene/`.
- **Appearance gap measured (2026-09-16 evening).** The network reads only the RGB planes (`DINORayEncoder.forward`),
  training RGB is a flat render with no augmentation, and swapping it for a Helios raytraced re-render of the same 20
  eval plants at the same framing (`plant_recon/eval/render_helios_eval_crops.py`) costs strict P 38.1 → 28.6 raw and
  66.3 → 59.2 refined, with the DAP probe and the existence gate failing on seedlings and mature plants exactly as on
  real crops. Next bet: fine-tune on Helios RGB with augmentation; fix the real crop framing (a fixed 1.2 m window, not
  1.2× the detector box) before the next real-image test.
- **Repository restructure (2026-09-16).** `plant_recon/` package, `use_cases/real_world/`,
  `outputs/{checkpoints,logs,wandb,weights,eval}`, `scripts/` launchers only, `submodules/Digital-Crops/`, docs as a
  topic tree with co-located assets (`docs/_index.md`). Map: `docs/architecture/code-structure/code-structure.md`.
  Paths in §0-A / §0-B were bulk-rewritten; the §6 source map is the 2026-09-11 tree with the moved locations annotated.


## 0-A. State as of 2026-09-13 (read this first; supersedes the older sections where they disagree)

The full record is `docs/../engineering/20260912-stage2-stage3-boundary/20260912-stage2-stage3-boundary.md`
(§0 status, §1.9.2 the training blocker, §2.4-2.5 the round-trip, §2.6 the dataset-plant round-trip, §5 reading order, §6 commit log); the 2026-09-14 report is `docs/experiments/20260914-stage2-burst-fix-roundtrip/20260914-stage2-burst-fix-roundtrip.md`.

**Round-trip (the "very important" requirement): solved.** `docs/experiments/20260914-stage2-burst-fix-roundtrip/assets/fig14_phytomer_vae_helios_roundtrip.png`
reads IK-only 99.9 / 99.6 / 97.7% and VAE round-trip **93.7 / 98.3 / 95.8%** FG IoU (DAP 10 / 50 / 90; regenerated
2026-09-14 with the leaf inverse, previously 95.7 / 99.5 / 96.7 and 92.2 / 98.4 / 95.7); on 2026-09-12 morning it was 81.4 / 90.2 / 79.3. Three fixes, all landed and on by default:
- leaflet emit order (the XML converter assigns leaf yaw by encounter order and reads only the scale; the terminal
  leaflet is identified per node, `phytomer_packets.terminal_leaflet_is_slot2`), and leaf size is one scalar per node
  (1 : 1 : 10/9) re-imposed in `assemble_packets`;
- **stem inverse kinematics** in the export (`plant_recon/models/part_tensor_stem_ik.py`, run by
  `assemble_part_tensor_to_xml`; `PART_TENSOR_STEM_IK=0` gives the old analytical export). Helios rebuilds a shoot by FK
  from per-node angles; the solver makes that FK land on the predicted nodes. This, not VAE fidelity, was the ceiling
  (the identity export went 94.2 -> 99.6% at DAP 50);
- a better VAE, `outputs/checkpoints/phytomer_vae_v9_tl_rw4_20k/` (128D = 48 + 10x8, `--rot-weight 4`, 20,000
  files, terminal-last packets). The eval scripts default to it and set `PHYTOMER_TERMINAL_LAST=1` for any `_tl`
  checkpoint. The *legacy* `_unreferenced/phytomer_9slot_roundtrip_comparison.png` is the old soft-rasterizer pipeline;
  ignore it.

![Figure 14 phytomer vae helios roundtrip](../experiments/20260914-stage2-burst-fix-roundtrip/assets/fig14_phytomer_vae_helios_roundtrip.png)

**Dataset plants (2026-09-14, design doc §2.6).** fig14 above is measured on the exact_gt trio, which is generated with
the converter's own default angles; on dataset plants (DAP 15/40/75 in the regenerated
`docs/experiments/20260914-stage2-burst-fix-roundtrip/assets/fig12_phytomer_10slot_helios_roundtrip.png`, script `plant_recon/eval/eval_phytomer_10slot_assembly_views.py`)
the export alone read 81.8 / 92.2 / 54.6% and the VAE added nothing. Fixed, all on by default: `chain_phytomers` no longer
requires a parent below its child (drooping laterals were cut; cycles are now broken at their most expensive edge, and
`root_own_shoot=True` keeps the cotyledon node as shoot 0); `gt_parent_links(..., internode_base=)` resolves a lateral's
branch point from the decoded slot-0 base (66 -> 98% of laterals right -- the training call site passes it, so every
earlier run trained on wrong parent/depth targets for a third of the laterals); and `part_tensor_leaf_ik.py` inverts the
FK's leaf rotation per leaf (`PART_TENSOR_LEAF_IK=0` disables). Packet path on those plants: **98.3 / 98.2 / 95.6%**, VAE round-trip **95.1 / 96.8 / 95.6%**.

![Figure 12 phytomer 10slot helios roundtrip](../experiments/20260914-stage2-burst-fix-roundtrip/assets/fig12_phytomer_10slot_helios_roundtrip.png)

**VAE / cache lineage.** Every FM checkpoint so far read its Stage-3 target latents from a packet cache stamped with a VAE's latents. Since 2026-09-14 that coupling is gone: `train_hierarchical_flow_matching.py` encodes the target latent on the fly from the cached packets with the VAE the run loads, so `dataset/cache/cowpea_curv26_pkt_v9/` (pkt_version 7, terminal-last) and `cowpea_curv26_pkt/` (pkt_version 6, bottom-to-top) hold only packets/presence/centers/refs/keys and are VAE-independent. What still binds VAE ↔ cache is the packet ORDER (terminal-last vs bottom-to-top), via `PHYTOMER_TERMINAL_LAST`. So: bump `PKT_VERSION` and regenerate ONLY when the packet format changes; to swap the VAE, point `PHYTOMER_VAE_CHECKPOINT` at it (or `TRAIN_VAE=1` in the launcher) with `PKT_VERSION` matching its packets. The old standalone launchers `archive/slurm_scripts/train_phytomer_vae.sh` and `generate_phytomer_packets_jobs.sh` are folded into `train_hierarchical_flow_matching.sh` (`TRAIN_VAE=1`) and `generate_helios_dataset_jobs.sh` (`--packets-only`), and now live under `archive/slurm_scripts/`.

**Training: the Stage 2 gradient burst is FIXED (2026-09-14).** Root cause: the coarse `nn.TransformerDecoder` (`norm_first=True`) had no final norm, so its raw residual stream (magnitude ~1e3 late in training) went into the bf16 phytomer self-attention as query/key/value; on a frozen burst state the backward amplified the gradient ~6000x on every batch. A final LayerNorm (`FM_DECODER_FINAL_NORM`, default on) gives 0/8 burst steps vs 8/8; fp32 self-attention (`FM_SELFATTN_FP32`, default on) is a partial mitigation kept as well. The v9 run restarted from scratch with both (`outputs/logs/local_v9_run2.log`, epochs 1-15, 0 canary hits; continued on 2026-09-14 10:30 from epoch 15 as cluster job `38252603` (geminigrp, 2 GPUs, log `hierarchical_fm_38252603.log`, checkpoints `hierarchical_fm_v9/`) with the §2.6 topology targets; the launcher's defaults are now the v9 recipe, and the queued `low`-partition jobs 38249632 / 38250275 pick up the current code when they start). The history below is kept for the record. It recurred at epoch 27-28 in two runs that
differ in seed and learning rate (`38240479` at 1e-4, `38242849` at 5e-5, both resumed from the same lineage) at the
same steps, and `DistributedSampler` is seeded by epoch only, so those runs saw the same batches. Refuted: weight decay,
decoder token collapse (`FM_ACT_PROBE=1`), a 512-sample subset replay. Live: a full-dataset replay of epochs 26-28 from
`hierarchical_fm_depth_ord/hierarchical_fm_epoch_025.pt` with `FM_SPIKE_DUMP=1` (names the samples on Stage 2 loss
spikes and on the first canary) -- running locally (`outputs/logs/local_replay_full.log`, slow) and queued on the
cluster as `38243735` behind the group's GPU quota. The lineage's checkpoints: epoch 15 in `hierarchical_fm_render_on/`,
epochs 20 and 25 in `hierarchical_fm_depth_ord/`. Use `LR=1e-4`, `FORCE_BATCH_SIZE=48`, `RENDER_GRAD_START_EPOCH=11`,
`RENDER_FRACTION=0.03` (and `SEED`).

**Model changes since 2026-09-11 that the next run carries**: ordinal = depth from the root with a parent-step loss;
`gt_parent_links` one-parent rule (origin for the root, branch point for laterals); inference draws internodes from the
chain over live slots and forces slot-0 rotation to identity; Stage 3 now takes its **(parent, self) pair** with the
parent held fixed (GT parent + noise in training: `--parent_jitter_cm 1.5`, `--parent_substitution 0.05`; chain parent
at inference; module `node_parent_mlp`, zero-initialised). `FM_PARENT_COND=0` builds the model without that module for an
exact resume of an older checkpoint; otherwise older checkpoints resume with a name-aligned partial optimizer restore.

**Debug switches**: `FM_GRAD_PROBE=1` (per-layer gradient amplification), `FM_ACT_PROBE=1` (decoder LayerNorm input
std), `FM_SPIKE_DUMP=1` (sample names on Stage 2 spikes), `FM_ASSEMBLY_PROBE=1` (assembly dump).


## 0-B. Handover state as of 2026-09-14 ~11:30 PDT (Claude Code; read this first)

### 0-B.1 Pipeline decoupling & launcher consolidation (committed + pushed today)

Two commits, `main` == `origin/main`:

- **`369c3b4` — pipeline: decouple VAE from packet cache; fold VAE/pkt launchers into the two current launchers.**
  - `train_hierarchical_flow_matching.py` now encodes the Stage-3 target latent **on the fly** from the cached
    packets with the VAE the run loaded (verified against the old cached v9 latents: max |diff| ≈ 2e-3, the fp16
    floor). A `latent` field in older caches is ignored. The packet cache is therefore **VAE-independent**: a new VAE
    needs no 40-job cache rebuild; only the packet order (`pkt_version`, `PHYTOMER_TERMINAL_LAST`) must still match.
  - `generate_cache.py`: `--vae-checkpoint` defaults to `""` (latent is an opt-in extra); dead
    `DEFAULT_VAE_CHECKPOINT` constant removed; docstrings updated (`pkt` mode = packets/presence/centers/refs/keys).
  - `train_hierarchical_flow_matching.sh`: opt-in **`TRAIN_VAE=1`** stage trains the PhytomerVAE first in the same
    allocation with the v9 recipe (`--rot-weight 4`, 20,000 files, 120 epochs), then runs FM with it.
    Standalone `archive/slurm_scripts/train_phytomer_vae.sh` moved to `archive/slurm_scripts/` (indexed in `archive/README.md`).
  - `generate_helios_dataset_jobs.sh`: the XML-direct packet backfill is now the **`--packets-only`** phase
    (`--pkt-version`, `--terminal-last`, `--pkt-out-dir`, `--num-jobs`), no VAE. Standalone
    `archive/slurm_scripts/generate_phytomer_packets_jobs.sh` moved to `archive/slurm_scripts/`.
  - `tests/test_hierarchical_3stage_cascaded.py` fixed to the current model contract (pure-latent flow width,
    10 slots, `phytomer_parent_rel` input, valid 1+16×16 patch grid) — 2 stale failures closed. Full relevant suite:
    **57 passed**.
- **`f735ce9` — docs: repoint stale scratch/ paths to archive/scratch/ and add 2026-09-14 result figures** (fig13,
  fig14, `hierarchical_self_consistency_epoch_005.png`, metrics JSONs; `tests/unit/` path fixes).

### 0-B.2 Training state — GPU-efficiency experiment in flight, needs a decision

- Job `38252603` (batch 48/GPU, the validated v9 recipe; ~6.7 GB/49 GB VRAM, ~20% GPU util, **32 min/epoch**) was
  cancelled at ~11:15 and replaced by **`38252937`** (`gpu-6000_ada-h`, gpu-10-54, 24 h, log
  `outputs/logs/hierarchical_fm_38252937.log`): `FORCE_BATCH_SIZE=auto NUM_WORKERS=8`, resume
  (`INIT_CHECKPOINT=.../hierarchical_fm_v9_local2/hierarchical_fm_epoch_015.pt RESUME=1`) → resumed at **epoch 16**,
  output `outputs/checkpoints/hierarchical_fm_v9/`, same LR 1e-4 / seed 1234 / render-from-11 / save-every-5.
- The VRAM probe (`probe_optimal_batch_size`) tuned **256 per GPU** (global 512; est peak 29.4 GB). Its `max_batch`
  is hardcoded to 256 in `train_hierarchical_flow_matching.py`; without the cap it would have picked ~392.
- **Finding (measured, not guessed):** step time went **2.2 s → ~18 s** (prof: backward 1.3→11.2 s, render
  0.65→5.27 s — roughly linear in batch), while steps/epoch fell only 1044→200. Net: **~55–60 min/epoch, WORSE than
  the 32 min/epoch at batch 48**. Backward scaling is slightly superlinear (8.6× for 5.3× batch).
- **Uncommitted changes backing the experiment** (commit or revert them):
  - `plant_recon/training/train_hierarchical_flow_matching.py`: new `--num_workers` arg (default 4);
    both DataLoaders now use `num_workers=args.num_workers, prefetch_factor=4, persistent_workers=True`.
  - `scripts/train_hierarchical_flow_matching.sh`: passes `--num_workers "${NUM_WORKERS:-8}"`.
- **Recommendation for the next agent:** wall-clock favours the small batch. Either (a) cancel 38252937 and
  resubmit with `FORCE_BATCH_SIZE=48` (the validated recipe; keeps LR/optimizer-momentum consistency of the lineage)
  while keeping the loader improvements (they are harmless and shave the small `other` wait), or (b) sweep one
  intermediate batch (96 or 128) once, measuring step profs from the log, before deciding. Batch 256 also changes the
  optimisation landscape mid-resume (Adam moments were calibrated at batch 48) — another reason to prefer (a).

- **Decision taken 2026-09-14 11:40 (Claude Code):** option (a). `38252937` cancelled; **`38252981`** resubmitted with
  `FORCE_BATCH_SIZE=48` (launcher default), `NUM_WORKERS=8`, resume from `hierarchical_fm_v9_local2/hierarchical_fm_epoch_015.pt`
  (no newer checkpoint existed: `hierarchical_fm_v9/` held only `eval_set.json`), output `hierarchical_fm_v9/`. The loader
  changes are committed. Batch 48 stays the recipe until an intermediate batch is measured deliberately.

### 0-B.4 Step time root-caused: the packet assembly loop, not the renderer (2026-09-14 ~12:20, Claude Code, `f3c5366`)

Synchronized per-section timers (`_sync_cuda` is unconditional) on a 1-GPU smoke at batch 48, resumed from epoch 15:

| | render ON, old code | render OFF | render ON, vectorized assembly |
|---|---|---|---|
| step (fwd/bwd) | 1.8-2.1 s | 0.11-0.31 s | **0.19-0.39 s** |
| render block | 0.6 s (of which `assemble_packets` region 0.55 s; mesh 0.01, raster 0.00) | - | 0.05 s |
| backward | 1.1 s | 0.02-0.04 s | 0.05 s |

Standalone: mesh build 5-11 ms, rasterize 2 ms, render backward 10-21 ms per sample (DAP 15/40/75). So the render
path was never the cost; `assemble_packets` ran a Python loop per phytomer (~1,000 per render batch) building the
curved-petiole leaflet and peduncle attachment points, and its autograd graph of tiny ops made the backward worse.
Now `_curve_points_batched` / `_point_on_curve` do it for all phytomers at once (`tests/test_assemble_packets_batched.py`
pins values and gradients to the loop). This also answers the batch-size question: batch 256 was slow because the loop
scaled with the number of rendered phytomers; with the loop gone a larger batch may pay — re-measure before deciding.
nvdiffrast 0.4 supports range-mode batching (one `rasterize` over concatenated meshes) if rendering ever becomes the
cost; it is not today. Job **`38253029`** runs the vectorized code from the epoch-15 checkpoint (batch 48, `NUM_WORKERS=8`).

### 0-B.5 The remaining per-sample loops (2026-09-14 ~13:40, `a1a82bc`)

With the assembly vectorized, the same timers at batch 48 / 256 (1 GPU) put the rest of the step in three per-sample
loops: the GT target loop (`gt_parent_links` + `decode_packets` per sample on the GPU: 0.10 / 0.53 s), the per-sample
VAE encode of the target latent (0.04 / 0.19 s), and the render block's per-plant render loop (0.05 / 0.30 s).
Landed: `part_array_dataset.attach_parent_links` computes `parent_pos/parent_idx/depth` in the DataLoader worker
(the step reads them; target loop 0.02-0.05 / 0.17-0.28 s), one `phytomer_vae.encode` over the whole batch (0.01 / 0.02 s),
and `HeliosPyTorchRenderer.render_batched` (one nvdiffrast range-mode pass per pyramid level for all rendered plants,
own camera each; `tests/test_render_batched.py`, `tests/test_render_loss_vectorized.py`). Step: 0.17-0.31 s at 48,
1.1-1.8 s at 256. Still per sample and still linear in batch: `chain_phytomers` for the rendered plants (`topo`
0.14-0.21 s for 8 plants -- its walk is a Python loop over nodes), the greedy matcher (0.09-0.24 s), the rest of the
target loop, and the model backward. **Per-sample cost is flat in batch, so batch 48 remains the recipe**; the render
fraction (3% of the batch) is unchanged throughout. Job **`38253401`** runs this code, `AUTO_RESUME` from
`hierarchical_fm_v9/hierarchical_fm_epoch_020.pt` (its predecessor `38253029` did epochs 16-21 at ~9 min/epoch).

### 0-B.6 chain_phytomers and the matcher vectorized (2026-09-14 ~14:40, `25c2251`)

`chain_phytomers` resolves the forest with pointer jumping (`_resolve_forest`: cycles by parent doubling, cut at the
most expensive edge with a float64 key -- the 1e-6 height tie-break was below float32 resolution and mutual-nearest
pairs tied exactly; continuation child by ordinal gap then direction/distance via `scatter_reduce`; starts, positions
and shoot ids by doubling). The loops stay behind `vectorized=False` for `tests/test_chain_vectorized.py`. 34 ms → 2 ms
per plant at N=512. The matcher's phytomer-level pass (`_forward_phytomer_batched`) works on the padded batch --
clusters compacted before any K-sized tensor, one greedy loop, one host sync -- and takes the padded GT directly
(`tgt_labels_padded` / `tgt_positions_padded` / `tgt_valid`), so the step no longer builds per-sample target lists
(three boolean-index syncs each). `tests/test_matcher_batched.py`. 42 → 22 ms at B=48, 166 → 56 ms at B=256.

Step (1 GPU): **batch 48: 0.15-0.24 s; batch 256: 0.9-1.5 s.** What is left per sample: the GT target loop
(`tgt_loop`, 0.02-0.05 / 0.17-0.28 s) and the batch-to-device head (0.01-0.03 / 0.10-0.17 s); the residual next to
`backward` is the forward passes' own GPU time (the fwd1/fwd2 timers are not synchronized). Per-sample cost is still
roughly flat in batch (3-5 ms at 48, 3.5-6 ms at 256), so batch 48 remains the recipe. Job **`38253656`** runs this
code, `AUTO_RESUME` from `hierarchical_fm_v9/hierarchical_fm_epoch_025.pt`.

### 0-B.7 GT-substitution ablation: node position is the first-order error (2026-09-14 ~15:50, epoch 40)

`plant_recon/eval/eval_gt_substitution_ablation.py` (run dir: `gt_substitution_epochNNN.json`): every predicted node
matched to a GT phytomer, one quantity of the matched nodes replaced by ground truth, re-rendered, silhouette IoU against
the GT render, 20 plants of the fixed eval set. P 24.8% → pos←GT 42.3 → ALL (pos+topo+rot+scale+latent) 82.1;
leave-one-out from ALL: −pos 30.6, −rot 49.1, −latent 47.5, −scale 61.6, −topo 82.1. Position first, then rotation
and latent (−33 each when the rest is right), then scale; the ordinal head costs nothing given the rest; the predicted
node SET (missing / spurious, 0.5 gate) is the last 18 points; seedlings (DAP ≤ 15) are at 2.5% as predicted. This is
the evidence for §2.1's remaining piece -- child position/roll/scale generated in Stage 3 relative to the fixed parent
-- with position first. Details: `docs/experiments/20260914-stage2-burst-fix-roundtrip/20260914-stage2-burst-fix-roundtrip.md` §7.

### 0-B.8 Stage 3 geometry implemented, A/B in flight (2026-09-14 ~16:10, `abdbaf1`)

`--stage3_geometry` (launcher `STAGE3_GEOMETRY=1`): Stage 3's flow state becomes `[(pos − parent_pos)·BASE_SCALE | roll | scale | latent]`
(`split_flow_state` / `geometry_from_flow` in `hierarchical_part_flow_matching.py`; decoder built with base/rot/scale dims 3/2/3).
Targets are relative to the NOISED parent the model is conditioned on (jitter + substitution unchanged), velocity loss =
latent MSE + `--stage3_geom_weight` (4.0) × geometry MSE, `sample_ode` returns the refined pos/roll/scale under the usual
keys (Stage 2's under `phytomer_pos_stage2`), the render block renders the refined plant and its loss reaches Stage 3's
geometry block through pos and scale. A latent-only checkpoint widens on load (latent block kept in `geom_proj` / the
velocity head; Adam moments widened by `_widen_optimizer_state`). Stage 2 keeps its heads (matching, parents, ordinal,
existence). Off by default. **A/B** from `hierarchical_fm_v9/hierarchical_fm_epoch_045.pt`: baseline `38257989`
(cluster, 2 GPU, render 1/6) vs the geometry run running locally on 1 GPU (`outputs/logs/local_s3geom_ab.log`,
`hierarchical_fm_v9_s3geom/`, eval every epoch). Read the per-epoch `[Self-Consistency]` IoU of both; the run's velocity
loss starts high (fresh geometry dims, ~12-20) and should fall within the first epochs.

### 0-B.9 Three runs, and what the latent path actually knows (2026-09-14 ~17:30, `b348fd7`)

**Runs** (all from `hierarchical_fm_v9/hierarchical_fm_epoch_045.pt`, same recipe, self-consistency IoU on the fixed
20-plant set every epoch):

| run | where | log / checkpoints | flag |
| :--- | :--- | :--- | :--- |
| baseline (latent-only Stage 3) | cluster `38257989`, 2 GPU | `outputs/logs/hierarchical_fm_38257989.log`, `hierarchical_fm_v9/` | — |
| geometry (stopped 10:55 at ep80) | local 1 GPU | `outputs/logs/local_s3geom_ab2.log`, `hierarchical_fm_v9_s3geom/`, panels `run_local_20260914_164059/` | `STAGE3_GEOMETRY=1` |
| gt_nodes (teacher forcing, stopped 10:55 at ep78) | local 1 GPU, started 17:07 | `outputs/logs/local_gtnodes_ab.log`, `hierarchical_fm_v9_gtnodes/`, panels `run_local_20260914_170731/` | `STAGE3_GT_NODES=1` |
| **10% chain** (all: `MAX_TRAIN_SAMPLES=10000 EVAL_SET_FILE=hierarchical_fm_v9/eval_set.json HOLDOUT_PER_BUCKET=2`, six independent 1-GPU jobs on geminigrp `gpu-6000_ada-h` — Heesup: up to 8 high-priority GPUs there — first started 10:32 on gpu-10-54, the rest as GPUs free up (baseline ends 15:53; gpu-10-50 draining with 3 idle GPUs), ~2 min/epoch, 50 epochs each) | `38274493` baseline-on-10% → `38274494` `LATENT_NORM=1` → `38274495` scheduled TF (p 0.5, jitter 1 cm) → `38274496` `RENDER_TO_LATENT=1` → `38274497` combination (geometry ep78 + latent_norm + render→latent) → `38274498` v10 full (+ `MULTIZOOM=1 COVERAGE_WEIGHT=1.0 EXIST_COUNT_WEIGHT=0.5`) | `outputs/logs/hierarchical_fm_<job>.log`, `outputs/checkpoints/sub10_{base,lnorm,stf,r2l,combo,v10}/` | see §2.7 of the design doc |

| epoch | baseline IoU % | geometry IoU % | gt_nodes IoU % |
| :---: | :---: | :---: | :---: |
| 46 | 31.5 | (crashed at eval, fixed `985ba26`) | |
| 47 | 31.0 | 30.6 | |
| 48 | 32.7 | **35.0** (Vel 2.08, still falling) | |
| 49 | 26.9 | 28.6 (Vel 1.95) | |
| 50 | 29.9 | 30.9 (Vel 1.87) | |
| 47 (gt_nodes) | 31.0 | | 31.2 (Vel 0.156) |
| 48 (gt_nodes) | 32.7 | | 35.8 (Vel 0.154) |
| 49 (gt_nodes) | 26.9 | | **39.4** (Vel 0.150) |
| 50 (gt_nodes) | 29.9 | | 32.6 (Vel 0.147) |
| 51 (gt_nodes) | 31.4 | | 32.2 (Vel 0.148) |
| 52 (gt_nodes) | 30.7 | | 33.9 (Vel 0.145) |
| 53 | 33.5 | 30.9 (Vel 1.73) | |
| 54–55 | 33.9 / 29.1 | 31.5 / 31.2 (Vel 1.68) | |
| 66–75 (baseline only) | 35.9 / 28.8 / 33.1 / 37.6 / 31.3 / 37.4 / 35.7 / 30.7 / 34.1 / **36.9** (mean **34.2**) | | |
| 52 | 30.7 | 29.6 (Vel 1.81) | |
| 51 | 31.4 | 31.6 (Vel 1.83) | |
| 51–59 | 31.4 / 30.7 / 33.5 / 33.9 / 29.1 / 32.4 / 32.0 / 31.6 / 32.7 | | |

Epoch-to-epoch spread is ±3 points on the 20-plant set (baseline 26.9 → 33.9 within eight epochs), so read runs by
their mean over several epochs, not by one epoch: baseline 46–59 mean 31.4; geometry 47–50 mean 31.3 (4 epochs);
gt_nodes 46–52 mean 34.0 (7 epochs; baseline 46–52 mean 30.6). Note the baseline itself climbs later: 46–65 mean 31.3, 66–75 mean 34.2 — so the gt_nodes run reaches at epochs 46–52 what the baseline reaches ~20 epochs later; compare runs at equal epoch index, and expect the plateau to move. Geometry 47–55 mean 31.1 = baseline over the same span. The gt_nodes run is separating: 39.4 at epoch 49 is the highest any run has reached (baseline max 33.9 over 20 epochs, geometry max 35.0), its Vel falls steadily (0.169 → 0.150), and this in-training eval samples with Stage 2's own nodes, so it is the deployable output, not the teacher-forced one. Reading: training Stage 3 against clean node conditioning gives the latent path a consistent geometry → shape mapping, and Stage 2's nodes at inference (RMSE 1.7 cm) are close enough to use it. Confirm with the epoch 50 checkpoint (ablation `_latent` R², teacher-forced vs not) before promoting it. Under the ablation protocol
the geometry run's P went 27.5 → 26.3 → 29.6 (ep47/48/50; baseline ep50 27.5) and pos←GT 38.3 → 39.4 → 41.7, i.e.
directionally up but inside the noise; its latent spread stayed high at ep50 (0.46 vs GT 0.18, R² −1.27), so the
noisier latent has not corrected itself as the geometry rows settled.
| 46 (gt_nodes) | 31.5 | | 32.8 (Vel 0.159 vs baseline 0.169; in-training eval uses Stage 2 nodes) |

The gt_nodes run was launched with the launcher's default `SAVE_EVERY=5`, so its first checkpoint (for the teacher-forced
ablation and the latent probe) is epoch 50; the geometry run saves every epoch. Geometry ep48 under the ablation protocol
(`run_local_20260914_164059/gt_substitution_epoch048.json`): P 26.3, pos←GT 39.4, ALL−pos 33.6, ALL−latent 42.1,
ALL 77.7 — and a warning: the sampled latent's per-dim spread is **0.42 against the GT's 0.18** (baseline ep50: 0.17),
RMSE vs GT 7.1 (R² −1.3). The widened flow state is making the latent block noisier while the geometry rows are still
settling; watch whether it comes back down by epoch 50 before moving the run to the cluster. The two protocols are not
comparable with each other (in-training eval: chain topology + training renderer; ablation: greedy match + 256 px
PyTorch renderer, young plants zoomed 8×) — compare runs within one protocol only.

Read the table with the training-run caveat: the gt_nodes run is *trained* with GT nodes but the in-training eval
samples with Stage 2's nodes (train/test mismatch by design); its meaningful readout is the substitution ablation run
on its checkpoints, which teacher-forces the sampler automatically (`--teacher_force` is implied by the checkpoint's
`stage3_gt_nodes`), and the latent probe below. The geometry run's Vel loss (3.1 → 2.4 at epochs 46–47) is still
falling; the run is not readable before ~epoch 50.

**`--stage3_gt_nodes`** (`STAGE3_GT_NODES=1`): `model.forward(stage3_cond_override=)` / `sample_ode(cond_override=)`
replace Stage 2's pos/roll/scale by the given tensors for Stage 3's conditioning only (Stage 2's outputs and losses
untouched); training passes the GT target of every matched node (Stage 2's prediction for unmatched ones). Refused
together with `--stage3_geometry`. Test `tests/test_stage3_teacher_forcing.py`.

**Substitution ablation with the mean latent** (`eval_gt_substitution_ablation.py`, variants `meanlat` = predicted
geometry + dataset-mean latent, `ALL-latent+meanlat` = GT geometry + mean latent; JSON next to the run logs:
`run_38257989/gt_substitution_epoch050.json`, `run_local_20260914_164059/gt_substitution_epoch047.json`):

| variant | baseline ep50 | geometry ep47 |
| :--- | :---: | :---: |
| P (all predicted) | 27.5 | 27.5 |
| pos ← GT | 38.5 | 38.3 |
| ALL − pos | 33.1 | 36.4 |
| ALL − latent (GT geometry, predicted latent) | 43.3–44.1 | 42.3 |
| ALL − latent + meanlat (GT geometry, mean latent) | 40.8 | 40.9 |
| meanlat (predicted geometry, mean latent) | 28.5 | 28.5 |
| ALL | 76.1 | 80.9 |

So with perfect geometry the *predicted* latent is worth 2–3 IoU points over a constant mean latent, while the GT latent
is worth 33–38. The per-node readout confirms it — the ablation now prints the RMSE of the sampled latent against the
GT latent of the matched phytomer next to the mean latent's: **5.39 vs 4.66 (R² −0.34)** on baseline ep50, with the
right per-dim spread (0.174 vs GT 0.177). The sampled latents are draws from the marginal, not the plant's phytomers.

**Latent probe** (`scratchpad/latent_probe.py`, not in the repo — copies the ablation's setup): x1_hat = x_t + (1−t)·v at
t = 0 is what the network reads off the conditioning alone (x_0 is pure noise), averaged over 8 draws:

| conditioning | t | R² vs dataset mean | R² vs per-plant mean |
| :--- | :---: | :---: | :---: |
| Stage 2 nodes (baseline ep50) | 0 | 0.12 | −0.06 |
| GT nodes, same checkpoint | 0 | 0.15 | −0.03 |
| Stage 2 nodes | 0.5 | 0.85 | 0.82 |
| geometry ep47, Stage 2 nodes | 0 | 0.14 | −0.03 |

The conditioning carries a plant-level shift (DAP / size) and nothing per node; the R² at t = 0.5 is x_t leaking z1.
Stage 3 is a denoiser of the marginal, which is exactly what the flow loss rewards here: the latent's per-dim std is
≈0.41 (4.66/√128) against unit noise, so the velocity target is ~85% noise, the unconditional floor at small t is
Var(z1) ≈ 0.17 per dim, and the run's Vel loss (0.16) sits on that floor. Giving Stage 3 the GT nodes at eval on the
untrained-for-it checkpoint does not change this (row 2), so node error is not what hides the per-node shape.

**gt_nodes epoch 50 checkpoint, read three ways** (`run_local_20260914_170731/gt_substitution_epoch050_{tf,notf}.json`;
`--no_teacher_force` / `--tag` added to the ablation script):

| protocol | P | pos←GT | ALL−latent (GT geometry + predicted latent) | GT geometry + mean latent | ALL | latent R² vs mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| baseline ep50, Stage 2 nodes | 27.5 | 38.5 | 43.3 | 40.8 | 76.1 | −0.34 |
| gt_nodes ep50, **teacher-forced** (GT nodes for matched) | 45.0 | = P | **46.8** | 39.0 | 71.7 | −0.19 |
| gt_nodes ep50, Stage 2 nodes (deployable) | **20.2** | 38.0 | 43.3 | 39.0 | 72.1 | −0.45 |

And the latent probe (t = 0, conditioning only): R² over the per-plant mean **0.115 with GT nodes** (baseline ≈ 0), −0.02
with Stage 2's nodes. So (a) node error does starve the latent path: trained on clean nodes, Stage 3's latent is worth
+7.8 IoU over the mean latent given the right geometry (baseline +2.5) and 45.0 vs 38.5 when the nodes are right;
(b) it pays with exposure bias: with Stage 2's own nodes the strict protocol falls to 20.2 (mid-DAP plants 15.7 vs
baseline 24.8), while the in-training eval (128 px, no zoom) still scores it above the baseline (32.6 vs 29.9 at ep50;
46–50 mean 34.4 vs 30.4). The two protocols differ in strictness, not in what they measure: 256 px + 8× zoom punishes
organ misplacement that 128 px hides. **Next run: scheduled teacher forcing** — GT nodes with probability
`--stage3_gt_nodes_p` per matched node and Gaussian jitter `--stage3_gt_nodes_jitter_cm` on the GT position — to keep
(a) without (b). Implemented `117f86f` (launcher `STAGE3_GT_NODES_P` / `STAGE3_GT_NODES_JITTER_CM`, smoke-tested) and
submitted as `low` job **`38273174`** (p 0.5, jitter 1 cm, from epoch 45 into `hierarchical_fm_v9_stf/`). Both `low`
jobs (`38260124` render→latent, `38273174`) were still queued on priority at 20:10; the low partition's GPU nodes were
all in use since 17:45.

**gt_nodes epoch 55 and baseline epoch 80 under the strict protocol (23:05;
`run_local_20260914_170731/gt_substitution_epoch055_{tf,notf}.json`, `run_38257989/gt_substitution_epoch080.json`):**

| protocol | P | pos←GT | ALL−latent | GT geom + mean latent | ALL | active/GT nodes | latent R² (probe, GT nodes, vs plant mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| baseline ep50 | 27.5 | 38.5 | 43.3 | 40.8 | 76.1 | 69.5/73 | ≈0 |
| **baseline ep80** | **27.3** | 37.6 | 39.6 | 36.7 | **67.1** | **57.4/73** | – |
| gt_nodes ep50 teacher-forced / deployable | 45.0 / 20.2 | – / 38.0 | 46.8 / 43.3 | 39.0 | 71.7 / 72.1 | 63.2/73 | 0.115 |
| gt_nodes ep55 teacher-forced / deployable | **46.4** / 23.9 | – / 41.2 | **48.7** / 46.0 | 39.1 | 77.0 / 77.9 | 69.6/73 | **0.187** |

Two things this settles. (1) **The baseline's in-training climb (66–82 mean ~35, best 38.1) is not visible under the
strict protocol**: P 27.5 → 27.3 between epochs 50 and 80, young plants better (2.1 → 13.7) but mid/old worse (24.8 →
21.9, 43.0 → 39.4), and its existence head now under-predicts (57 active of 73 GT, was 69.5), which drops the ALL
ceiling from 76 to 67. The 128 px in-training eval rewards something the 256 px protocol does not; the three runs do
evaluate the same 20 plants (`eval_set.json` identical), so the disagreement is the metric, not the sample. Decisions
should rest on the strict protocol plus the latent readouts, with the in-training IoU as a trend indicator only.
(2) **The gt_nodes run keeps improving on every latent readout**: with GT nodes the latent is worth +9.6 IoU over the
mean latent (ep50 +7.8, baseline +2.5), per-node R² 0.187 (ep50 0.115), and its deployable P recovers 20.2 → 23.9 with
pos←GT already above the baseline (41.2 vs 37.6) — the remaining gap to the baseline's P is exposure bias to its own
node error, which the scheduled-teacher-forcing job `38273174` targets.

**Overnight (2026-09-15 08:40) — latest checkpoints under the strict protocol** (`gt_substitution_epoch070_{tf,notf}.json`
in `run_local_20260914_170731/`, `gt_substitution_epoch076.json` in `run_local_20260914_164059/`, `gt_substitution_epoch135.json`
in `run_38257989/`):

| run / checkpoint | P (deployable) | pos←GT | ALL−latent | ALL | active/GT | latent R² (GT nodes, vs plant mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| baseline ep135 | 28.9 | 35.2 | 37.9 | 67.0 | 55/73 | – |
| geometry ep76 | **33.9** | 42.9 | 45.5 | 79.2 | 63/73 | – (latent std still 0.47) |
| gt_nodes ep70, deployable / teacher-forced | 29.4 / 45.3 | 40.6 / – | 41.0 / 48.2 | 82.5 | 64/73 | **0.409** (ep55 0.187) |

Strict-protocol P over the checkpoints: baseline 27.5 → 27.3 → 28.9 (ep50/80/135, flat, ceiling falling to 67 with the
existence head at 55/73); geometry 27.5 → 26.3 → 29.6 → 33.9 (ep47/48/50/76, climbing); gt_nodes deployable 20.2 →
23.9 → 29.4 (ep50/55/70, recovering from the exposure bias, now above the baseline) with the teacher-forced ceiling
steady at 45–46 and the per-node latent R² doubling again to 0.41. In-training IoU over the same period: baseline peaked
at 36.2 (76–85) and fell to 29.9 (121–137, several epochs below 25); geometry 32–33; gt_nodes 34. Both runs beat the
baseline on both protocols now; the geometry run leads the deployable strict P, the gt_nodes run leads everything that
depends on the latent. The baseline job ends at its 24 h limit at 15:53; a dependent continuation (`38274192`, same
partition, `AUTO_RESUME=1`) was queued behind it, then **replaced (09:55) by the standardized-latent run `38274201`**
(`LATENT_NORM=1`, from epoch 45 into `hierarchical_fm_v9_lnorm/`, same 2-GPU slot, starts when the baseline ends):
the baseline's strict P has been flat at 27–29 for 90 epochs and its checkpoints every 5 epochs already serve as the
reference, while `--latent_norm` (`d46d8fe`) is the lever the latent diagnostics point at — the flow now matches the
per-dim standardized latent (σ mean 0.18, min 0.001 floored to 0.05, max 1.44), so the latent block is unit variance
against the noise, the one property of the 09-07 Option B model worth keeping. Its Vel restarts at ~8 and must fall
first; read its latent probe (`scratchpad/latent_probe.py`, R² at t = 0) before its IoU. Low jobs `38260124` /
`38273174` still queued at 09:55 (16 h; low's GPU nodes are held by other users' multi-day jobs).

**Why the 2026-09-07/08 "45.4%" panel is not a bar to beat (2026-09-15 09:20).** Heesup asked whether
`docs/experiments/20260907-latent-fm-500epoch/assets/hierarchical_self_consistency_epoch_125.png` (Option B, organ-level 16D latent, job
`38145444`, epoch 125) was a coincidence and whether to go back to it. Measured, not argued:

![Hierarchical self consistency epoch 125](../experiments/20260907-latent-fm-500epoch/assets/hierarchical_self_consistency_epoch_125.png)

- That number was the mean over the **first 4 plants of one random batch** (DAP 17/42/72/49, no seedlings; the eval of
  that era used `num_samples_to_plot=4` on `next(iter(dataloader))`, different plants every time). The job evaluated
  six times in all — epochs 25/50/75/100/125/150 → 39.1 / 26.4 / 33.8 / 26.9 / 45.4 / 49.2 — a series dominated by
  draw variance. The 55.1% / 67.9% in the 09-07 report are the same kind of number.
- The same checkpoint (`hierarchical_latent_fm/hierarchical_fm_epoch_125.pt`), run with its exact code (worktree at
  `870074f`, state dict loads with 0 missing / 0 unexpected; `73edc46` gives 27.5 with 4 random head weights) on
  today's fixed 20-plant set with the same 128 px eval lineage: **24.8%** (DAP ≥ 17: 30.8, 40–75: 34.5, > 60: 37.0,
  ≤ 15: 0.9; per plant 0–60%). Today's runs on the same 20 plants: baseline ~30, geometry ~32–33, gt_nodes ~34.
  Script + per-plant JSON: `outputs/logs/archive_20260914/optionb_ep125_reeval/`.
- What Option B did have that today's model lacks: its per-organ latent was a **unit-variance N(0, I)** space, so the
  flow loss was not dominated by noise prediction — the exact property whose absence (128D latent, per-dim std ≈ 0.41)
  explains why today's Stage 3 latent carries no per-node image information. That is the piece to port (the pending
  unit-variance latent scale for the flow), not the architecture.
- **Same-plant visual comparison** (`docs/archive/unreferenced-assets/20260915_optionb_vs_today_same_plants.png`, results report §10):
  on 7 mature plants Option B 38.3 vs baseline 42.0 / geometry 43.1 / gt_nodes 42.9 mean IoU. Heesup's impression that
  Option B *looks* closer to the input is real and explained: it generated organs one by one (spread canopy, visible
  stems and flowers, but floating organs and over-long stems), while today's phytomer packets with a mean-like latent
  give compact uniform blobs whose IoU is higher because they cover the plant's centre. Both miss the canopy spread of
  star-shaped plants (DAP 72 / 88). The architecture stays; the latent's variance and per-node content are the levers.
- **Not better nodes either** (`nodes_cmp.json` there): on the same 7 plants Option B's predicted nodes sit 18.3 cm (nearest
  GT phytomer RMSE) with a convex hull 4.35× the GT's — scatter, not spread; today's runs 4.5–5.9 cm but hull 0.5–0.7×
  and only ~25% of GT phytomers have a predicted node within 3 cm (the geometry run is best: 4.5 cm / 29%). That
  under-spread is the "pos←GT +10–14 IoU" of the substitution ablation, and it is the next lever after the latent.
- **10% protocol (10:15, Heesup's proposal)**: every chain run now trains on a DAP-stratified 10k-plant subset with the
same 20 eval plants force-included (`--eval_set_file`, matched by prefix) and 20 held-out plants removed from training and
reported as `[Holdout]` each eval (`--holdout_samples_per_bucket 2`; caveat: runs resumed from epoch 45 saw those plants
during epochs 1–45, so the held-out is "unseen since 45" — a clean held-out needs training from scratch with the
exclusion). The point: the baseline saw each of its 100k plants 140 times and still plateaued, so data volume is not the
limit; 10k plants answer "can this structure memorize the data at all?" ten times faster, and the held-out line tells
whether what it learns transfers. If the 10% runs reach 70–80% on the training plants the bottleneck is
optimization/scale; if not, it is the structure/loss (the current diagnosis). The winner gets the full data.
**10:55 — the two local full-data runs were stopped on Heesup's request** (geometry at epoch 80, last checkpoint
`hierarchical_fm_v9_s3geom/hierarchical_fm_epoch_080.pt`; gt_nodes at epoch 78, last checkpoint
`hierarchical_fm_v9_gtnodes/hierarchical_fm_epoch_075.pt`; both had answered their questions) and this node's GPU now runs
the two 10% runs that could not get a cluster GPU: **combination** (`outputs/logs/local_sub10_combo.log`,
`sub10_combo/`, panels `run_local_20260915_105522/`) and **v10 full** (`local_sub10_v10.log`, `sub10_v10/`,
`run_local_20260915_105524/`), both from geometry ep78 with `LATENT_NORM=1 RENDER_TO_LATENT=1` (+ `MULTIZOOM=1
COVERAGE_WEIGHT=1.0 EXIST_COUNT_WEIGHT=0.5` for v10), sharing one Ada GPU. Their cluster copies were cancelled. Heesup
also opened `jmearlesgrp`'s `gpum` partition (and the association lists `gpuh`, `gpu-a100-h`), but at 10:50 every one of
those nodes was RAM-blocked by other users' jobs, and a job cannot list several partitions under this association.
**12:10 — second 10% reading (results report §11.2)**: geometry-based runs lead the deployable strict P (v10 full ep85
**36.0**, combination ep85 33.6) against 25–29 for the latent-only single changes (baseline-on-10% 26.5, standardized
latent 25.4, scheduled TF 27.3 deployable / 43.7 teacher-forced, render→latent 27.5); no run memorizes the 10k plants
(train-plant IoU flat at 33–36 for 25–30 epochs, held-out within 5 points), and no run's latent carries per-node image
information yet. The standardized-latent and render→latent runs were cancelled as flat; two v10 variants were submitted
in their place (`38274747` render fraction 1.0, `38274748` lr 2e-4, both from geometry ep78) — **but the freed GPUs were
taken at once by other groups' queued jobs on the shared Ada node** (js2552 ×2 running, 4 more pending), so they wait.
Lesson: on `gpu-6000_ada-h` a cancelled job's GPU does not come back to us; keep runs going until their budget ends.
**2026-09-16 02:40 — full-data lineage reached ep128 (job `38279147` COMPLETED, 9 h 25 min); continued to ep160 as `38332310`.**
Readings over ep95–125 (strict P raw / EMA → refined): 35.9/35.1 → 66.3/64.7, 34.5/35.1 → 64.8/66.2, 37.4/36.8 → 66.2/66.8,
38.3/39.0 → 64.6/65.9, 35.4/35.9 → 65.4/66.7, 36.3/37.3 → 67.9/67.1, 34.3/38.3 → 66.8/67.6. Raw P drifts up slowly and
noisily (EMA 35 → 38), the refined level is flat at 65–68. ep128: raw 38.1 / EMA 37.7, refined 67.0 / 67.0. Since the
slot on gpu-10-54 would otherwise go to other groups, the lineage continues (`AUTO_RESUME` from ep128, `EPOCHS=160`,
same flags, EMA restored from the checkpoint) under job `38332343`, log `outputs/logs/20260916/hierarchical_fm_38332343.log` (the first attempt, `38332310`, resumed from `hierarchical_fm_epoch_128_ema.pt` because the launcher's AUTO_RESUME took the newest `hierarchical_fm_epoch_*.pt`, which the EMA files now match — no optimizer state, fresh warm-up; cancelled after 3 min, launcher fixed with `grep -v '_ema\.pt$'`, commit `ce8f150`);
a detached loop scores its EMA files ep130–160 (`full_ema_readings2.log` in the session scratchpad). ep130 EMA: 37.5, refined 66.0; ep135 EMA: 35.9, refined 64.9; ep140 EMA: 37.3, refined 65.5; ep145 EMA: 38.3, refined 65.7; ep150 EMA: 36.5, refined 63.8 — flat at 35-38 / 64-66 since ep135. Job `38332343` is still running (ep154 at 09:11 on 2026-09-16); ep140/145/150 EMA readings recovered from JSON on disk (37.3/65.5, 38.3/65.7, 36.5/63.8 -- flat). The detached scorer loops and the local `cnt2` run died when the session restarted around 09:00.

![Optionb vs today same plants](../archive/unreferenced-assets/20260915_optionb_vs_today_same_plants.png)

**09:15 relaunch note:** the local machine turned out to be a different node than before -- a single TITAN RTX with 24 GB VRAM, not the ~49 GB Ada card the earlier local runs profiled against (`Per-GPU VRAM: 49140 MiB` in the old logs). Resuming `cnt2` with the old fixed `--batch_size 48` OOM'd immediately; relaunched with `FORCE_BATCH_SIZE=auto` (runtime VRAM probe) instead, which is now the way to resume any local run whose original node is unknown. Both scorer loops were restarted from the first unscored epoch: `full_ema_readings3.sh` (full lineage, ep155/160) and `cnt2_readings2.sh` (cnt2, ep110 onward).

**20:30 — refinement on GT-substituted variants: the gap inside refinement is the latent, not the nodes (results report §11.10).**
`eval_gt_substitution_ablation.py --refine P,pos,rot,pos+rot,scale,ALL-latent,ALL` (calls
`eval_test_time_refinement.refine_plant`, the loop factored out of the script; default run re-verified 36.1 → 65.7).
Full-data ep95 raw: P 37.8 → 65.1 refined; GT pos 67.1, GT rot 65.6, GT pos+rot 67.5, GT scale 67.1, ALL-latent 68.6,
ALL 77.0 (78.6 unrefined). GT node geometry on the matched nodes adds only 3.5 points after refinement; the GT latent
adds 10 — the top-view CHM does not constrain leaf shape/orientation enough for the optimiser to find it. Young plants
are the exception (P+refine 26.9 vs ALL+refine 59.0): there the nodes are the bottleneck.

**19:20 — second full-data run, local: existence-count weight 2.0 (`hierarchical_fm_v10_cam_cnt2`).**
Every inference-side lever is now exhausted (steps, learning rates, prior strength, rotation/roll, zoom targets,
mean latent, existence threshold) and the remaining error is the network's existence deficit (~60 of 73 GT nodes
active) plus node error; lowering the threshold only adds false positives. The Ada and A100 nodes are full (CPUs and
RAM), so the free local GPU runs a full-data v10 + input camera + EMA lineage with `EXIST_COUNT_WEIGHT=2.0` (v10 used
0.5), from the s3geom ep78 checkpoint, 1 GPU (~30 min/epoch), log
`outputs/logs/20260915/local_v10_cam_cnt2_full.log`, checkpoints `outputs/checkpoints/hierarchical_fm_v10_cam_cnt2/`.


**10:15 — `sub10_v10_rampR` (the render-fraction curriculum) launched via `srun --jobid=38340946 --overlap` into
Heesup's own OnDemand desktop job's GPU allocation on `gpu-10-54`, at Heesup's instruction** (the desktop session
holds 1 GPU / 32 CPU / 64 GB that a normal `sbatch` can't see as free since the node's GRES accounting already
shows all 4 GPUs allocated node-wide; `srun --jobid=<job>` attaches a new step to an ALREADY-GRANTED allocation,
which only the job's own owner can do -- this only works because 38340946 belongs to this account). Readings: ep80 31.2, ep85 32.9, ep90 38.0, ep95 **35.4** (the ep90 spike reverted, as flagged -- back in the usual 31-38 noise band, no clear lift over the flat 0.167 baseline yet). Redirected
from the pending `sbatch` job `38341387` (cancelled to avoid a duplicate run once a normal slot freed up).
Log `outputs/logs/20260916/sub10_v10_rampR_srun.log`, checkpoints `outputs/checkpoints/sub10_v10_rampR/`,
`RENDER_GRAD_START_EPOCH=79` so the 0.167->0.5 ramp begins at this run's very first render-active epoch. The other
queued job (`hfm_s10_v10rexist`, render_to_exist test) stays in the normal sbatch queue -- the desktop GPU is now
busy with this run, so stacking a second training step on the same single GPU would risk the same contention stall
`cnt2` hit earlier today.

**10:20 — second one stacked on anyway, on Heesup's instruction after checking nvidia-smi directly (`srun_bash 38340946` + `nvidia-smi`): the RTX 6000 Ada card has 49 GB, `rampR` alone was only using 9 GB.** `sub10_v10_rexist`
(`--render_to_exist`, the other queued job, `38341780` cancelled) launched the same way, `srun --jobid=38340946 --overlap --gres=gpu:1`,
log `outputs/logs/20260916/sub10_v10_rexist_srun.log`, checkpoints `outputs/checkpoints/sub10_v10_rexist/`.
Confirmed safe via `srun --jobid=38340946 --overlap nvidia-smi`: 17.9 GB / 49.1 GB used with both running, 98% compute
utilization (the two processes time-slice the SM, so each trains somewhat slower than alone, but this is a throughput
cost, not a stability risk the way the earlier 24 GB-card memory pressure was). Three processes now share
`gpu-10-54`'s one physical GPU inside job `38340946`'s allocation: the desktop session itself (near-idle), `rampR`,
and `rexist`.Read it against the cluster lineage (`38279147`) at matching epochs with the strict protocol and the refinement. First reading ep80 (2 epochs in): raw strict P 32.8 (meanlat 33.8, ALL 68.6), refined 35.0 → 64.9 — level with the cluster lineage at the same stage (cluster ep100 raw refined 34.9 → 64.8). ep85: raw refined 35.8 → 64.3, EMA P 33.0 / refined 63.5 — ep90: raw 34.9 / EMA 34.5, refined 63.9 / 65.8 — three checkpoints in, the doubled count weight has not changed the level (main lineage at the same stage: 35–39 / 65–68). ep95: EMA 36.6, refined 64.0 (raw in `cnt2_readings.log`) — still within the main lineage's band; stop it after ep100 if unchanged.

**17:20 — refinement from the mean latent reaches the same 66.5 for both models (results report §11.10).**
`eval_test_time_refinement.py --init_mean_latent` starts from the mean GT phytomer latent of 300 random training
plants across all growth stages (not the model's `latent_mu` buffer, which is gathered from the first plants in
dataset order and starts at 9–12%): 10% v10 ep95 36.1 → 66.5 (sampled-latent start 67.6), full-data v10_cam ep90
raw 35.5 → 66.5 (sampled start 57.3). So the flow-sampled latent adds nothing over the mean once refinement runs,
and a checkpoint whose sampled latent is poor loses nothing. Deployable pipeline = Stage 2 nodes + mean latent +
render refinement; the network's real contribution is node positions, existence and topology, which is where the
remaining error (existence 60/73, node error 4–6 cm) lives. Lowering the existence threshold at inference does not help (`--exist_thresh` 0.3: 65.3, 0.4: 64.2 vs 67.6 at 0.5): the extra nodes are false positives the refinement cannot switch off, so the existence deficit is a training-side problem.

**16:30 — EMA weights for evaluation; full run continues as `38279147` with `EMA_DECAY=0.999`.**
The full-data run's ep85 checkpoint scored strict P 14.0 while ep80 scored 27.5 and the in-training holdout went
32.5 → 15.0 → 37.6 over epochs 84–86: checkpoints land on whatever state the last epoch left, and on full data the
swing is large. `--ema_decay` (launcher `EMA_DECAY=`, commit `064f1d3`) keeps an EMA of the weights through an
optimizer post-step hook (`WeightEMA` in the trainer, `tests/test_weight_ema.py`) and saves
`hierarchical_fm_epoch_NNN_ema.pt` next to every checkpoint with the same layout, so every eval script loads it
unchanged; the EMA state also rides inside the raw checkpoint (`ema_state_dict`) for resume. Job `38279147`
(`--dependency=afterany:38275054`, same OUTPUT_DIR, AUTO_RESUME) takes over the lineage the moment `38275054` ends:
`38275054` was cancelled at 17:14 right after its ep90 checkpoint was saved and `38279147` resumed from that checkpoint at 17:15 (`EMA: decay 0.999 per step`), so the first EMA files are `hierarchical_fm_epoch_095_ema.pt` onward. A detached loop scores each
`_ema.pt` (strict reading with `--tag ema` + default refinement; log `full_ema_readings.log` in the session scratchpad,
JSONs in `outputs/logs/20260915/run_38275054/`). Raw-checkpoint readings so far: ep80 27.5 (refined 64.6),
ep85 14.0 (refined 53.3), ep90 28.9 (meanlat 35.9), **ep95 35.9** (meanlat 35.2; refined 36.7 → 66.3) — the full-data model has caught up with the 10% v10 (35.8 → 67.6) and its sampled latent with the mean latent. EMA ep95: 35.1, refined 36.0 → 64.7 (only ~4 epochs of averaging so far). ep100: raw 34.5, EMA 35.1 (refined 66.2). The full-data lineage sits at strict P 34.5–35.9 / refined 64.7–66.3 over ep95–100 — the same plateau as the 10% runs, so 10× data does not move it. ep105: raw 37.4 / EMA 36.8, refined 66.2 / 66.8. ep110: raw 38.3 / **EMA 39.0** (best strict P of any run so far, still climbing slowly on full data), refined 64.6 / 65.9 (refined level unchanged). ep115: raw 35.4 / EMA 35.9, refined 65.4 / 66.7 — the ep110 high was partly noise; the lineage sits at 35–39 before and 65–67 after refinement. ep120: raw 36.3 / EMA 37.3, refined **67.9** / 67.1 (best refined reading of the lineage). ep125: raw 34.3 / EMA 38.3, refined 66.8 / 67.6. The run ends at ep128 (EPOCHS=128); its last EMA file is scored by the same loop.

**14:15 — scale-up phase: full-data v10 + input-camera run `38275054` on gpu-6000_ada-h (2 GPUs, 96 GB, 24 h).**
The 10% protocol has done its job: every 10% variant converges within ~10 epochs and plateaus at strict P 33–36
(v10 35.0 mean; cam ep80/85 35.8/34.0; w3t0 34.4; lr 2e-4 33.0; render-all 34.6; unfrozen 34.3), and the deployable
number is now set by test-time refinement (67.6–69.1). So the two plateaued 10% cluster jobs (38274747 render-all,
38274748 lr 2e-4) were cancelled and their slot given to the full-data run: v10 flags + `RENDER_INPUT_CAMERA=1`, from
the s3geom ep78 checkpoint, `EVAL_SET_FILE` + `HOLDOUT_PER_BUCKET=2`, `SAVE_EVERY=5`, 128 epochs, output
`outputs/checkpoints/hierarchical_fm_v10_cam/`, log `outputs/logs/20260915/hierarchical_fm_38275054.log`.
It is the generalisation check of the 10% findings and the candidate deployable model for refinement. Full-run readings
(detached scorer, JSONs in `outputs/logs/20260915/run_38275054/`): ep80 strict P 27.5, meanlat 34.4, ALL 79.6, refined 31.3 → 64.6 (~15–18 min/epoch). The two local 10% runs were stopped at 15:20 (`sub10_v10_cam` ep100, `sub10_v10_w3t0` ep113; both plateaued, checkpoints kept) so the local GPU serves evaluations; a second detached loop runs the default test-time refinement on every full-run checkpoint after its strict reading (log `full_refine.log` in the session scratchpad).

**09:58 (2026-09-16) — the local `cnt2` run stalled for ~20+ min (all worker processes near-0%% CPU, GPU util 39%%
while still holding ~20 GB) after several `eval_test_time_refinement.py` test invocations ran back-to-back on the
same GPU alongside it.** Likely GPU-context/DDP-heartbeat contention from stacking a live 1-GPU torchrun training
job with concurrent foreground eval processes on the same card, not a code bug -- the training log simply stopped
advancing mid-epoch-106 with no error. Killed (SIGTERM then SIGKILL on the surviving worker) and restarted with
`AUTO_RESUME=1 FORCE_BATCH_SIZE=auto` from the ep105 checkpoint (log
`outputs/logs/20260916/local_v10_cam_cnt2_full_r2.log`). Going forward: avoid running more than one
foreground eval/refinement script on this local GPU while a local training run is live; the detached scorer loop
(`cnt2_readings2.sh`, mostly idle/sleeping between checkpoints) is fine to keep running alongside it.

**10:35 — stalled a SECOND time, at the identical step (Epoch 106 Step 625/3128), this time with no competing
process on the GPU** (verified: only the training process held GPU compute, 104%% CPU on the main thread in `R`
state -- genuinely spinning, not blocked in a driver/IO wait -- so this was not the same GPU-contention cause as
the first stall). `py-spy` is not installed, so the exact hang site is unknown; killed and restarted again
(`AUTO_RESUME` from ep105, log `outputs/logs/20260916/local_v10_cam_cnt2_full_r3.log`). Because the resume
always restarts partway through the SAME epoch on the SAME (likely seeded) data order, a third stall at the same
step would point to one specific batch/sample under `EXIST_COUNT_WEIGHT=2.0`'s render loss (e.g. a batch where a
plant's existence collapses toward all-off, producing a degenerate empty mesh) rather than environment noise --
worth installing `py-spy` and dumping the stack before restarting again if it recurs.

**11:19 -- stalled a THIRD time, at the identical step (Epoch 106 Step 625/3128), after ~19 min with no log
advance while the main thread stayed at ~104%% CPU.** `py-spy` (installed after the second stall) needs ptrace
permission this environment does not grant (`Permission Denied` even with `sudo -n`), so the exact hang site is
still unknown. Three stalls at the EXACT same step across three independent restarts, always resuming from the
same ep105 checkpoint into epoch 106, is strong evidence it is a specific batch (deterministic epoch-seeded
dataloader order) that hangs under `EXIST_COUNT_WEIGHT=2.0`'s render loss -- plausibly a plant whose existence
collapses toward all-off under the doubled count penalty, producing a degenerate near-empty mesh that some part
of the differentiable-render path (nvdiffrast context, or a divide-by-near-zero-area triangle) spins on, rather
than environment noise. **Retired the run rather than restart a fourth time**: this experiment's actual question
(does doubling the existence-count weight move the plateau) was already answered by its five completed readings
(ep80-105, strict P 32.6-36.6, refined 63.4-66.6 raw and EMA -- indistinguishable from the main v10/v10_cam
lineage), so a fourth restart into a known reproducible hang was not worth the wall-clock cost. If this
existence-count-weight direction is revisited, worth checking the render block for a guard against a batch where
every phytomer's existence probability collapses near zero before touching this again.

**13:30 — training render loss now has the same camera option; run `sub10_v10_cam` launched (results report §11.10).**
`--render_input_camera 1` (launcher `RENDER_INPUT_CAMERA=1`, commit `d492e01`): for every rendered plant the training
loss now frames the prediction on the bbox centre of that plant's GT mesh (decoded from the batch `nodes` with
`ehsc.decode_predictions_to_part_tensor`, mesh built under no_grad, passed as `render_batched(centers=)`), i.e. the
camera generate_cache used for the input CHM. `tests/test_render_input_camera.py` checks the centred batched render
equals `forward(focus_plant=True)` for an off-centre plant. Run: v10 flags + RENDER_INPUT_CAMERA=1 from the s3geom ep78
checkpoint, 10% data, local GPU 0, log `outputs/logs/20260915/local_sub10_v10_cam.log`, checkpoints
`outputs/checkpoints/sub10_v10_cam/`. To make room the plateaued unfrozen-backbone run was stopped at ep88
(strict P 34.3 at ep80; resumable from its ep85 checkpoint with AUTO_RESUME=1). Second strict reading of the other
variants: w3t0 ep85 34.4, lr 2e-4 ep100 33.0 — every training-side variant sits at 33–35. First reading of `sub10_v10_cam` at ep80: strict P **35.8** (v10 itself was 35.4 at ep80) — ep85 34.0, ep90 38.5, ep95 34.6, ep100 34.8 (mean 35.5 = v10's 35.0; the corrected training camera does not move the 10% plateau) — ep90/95 readings are queued (detached scorer, log in the session scratchpad `cam_readings.log`, JSONs in `outputs/logs/20260915/run_sub10_v10_cam/`).

**13:20 — refinement in the input's camera frame: 33.5 → 62.9 strict P, but watch the geometry (results report §11.10).**
`eval_test_time_refinement.py --input_camera` renders the prediction with the camera that produced the cached input CHM
(GT plant bbox centre, via `render_batched(centers=)`) so the input loss is no longer shifted. v10 ep95, 20 plants, 40
steps, keep_best: **33.5 → 62.9** (DAP > 15: 38.4 → 74.5; every DAP > 15 plant rises, the two that collapsed before now
reach 80.8 / 81.3; DAP ≤ 15 unchanged). IoU is still the strict origin-frame 256 px protocol. BUT the saved renders
(`assets/20260915_test_time_refinement_before_after_input_camera_noprior.png`) show a few leaves inflated into
large flat polygons on 3 of 6 plants: the silhouette/depth loss has no prior on scale or latent, so the optimiser fills
the silhouette with implausible geometry. The origin-frame version (`assets/20260915_test_time_refinement_before_after.png`, 39.8 → 64.3 on the same six)
keeps leaf shapes. **With a shape prior** (`--reg_scale 5 --reg_latent 0.5`, squared deviation from the sampled values)
the same six plants go 39.6 → 70.4 and the inflated polygons are gone (now the main figure,
`assets/20260915_test_time_refinement_before_after.png`; the two earlier variants are `assets/20260915_test_time_refinement_before_after_origin_frame.png` and `assets/20260915_test_time_refinement_before_after_input_camera_noprior.png`).
On all 20 plants with the prior: **35.8 → 63.9** strict P (DAP > 15: 39.3 → 72.2) — the best deployable number so far, with plausible geometry. With all four cache zoom levels as targets (`--target_zooms 1,2,4,8`, now the default together with `--input_camera` and the prior; `--origin_camera` restores the old frame): **35.8 → 67.6** (DAP > 15: 41.1 → 74.0), the young plants finally gaining (DAP 2/3/11/12: 0/11/21/27 → 35/22/53/58). 80 steps with the defaults: 36.0 → 69.1 (DAP > 15 41.3 → 78.6). `--recompute_rot` (rotation and parent position re-derived from the moving nodes) 33.5 → 60.0 and roll as a fourth variable 35.1 → 61.4: both worse than the frozen-rotation default, so rotation stays fixed at the sampled node positions. Baseline ep135 (full data) with the defaults: 27.0 → 63.1 (DAP > 15 31.7 → 72.4) — the training-side gap to v10 shrinks from 8.8 to 4.5 after refinement. Lesson: silhouette IoU alone is not sufficient, the
refinement needs a prior on scale/latent — and the same GT-bbox camera must go into the training render block.

![Test time refinement before after input camera noprior](assets/20260915_test_time_refinement_before_after_input_camera_noprior.png)

![Test time refinement before after](assets/20260915_test_time_refinement_before_after.png)

![Test time refinement before after origin frame](assets/20260915_test_time_refinement_before_after_origin_frame.png)

**13:00 — the render loss has been comparing against a shifted target.** The cached input CHM (`generate_cache.py`,
`focus_plant=True`) is framed on the **GT plant's bounding-box centre**, while the training render block
(`render_batched`, `focus_plant=False` + `reference_window_size`) and every evaluation render frame the plant at the
**origin**. Measured on eval plants: the cached CHM overlaps a bbox-centred GT render at 73–89% IoU but an origin-framed
GT render at only 41–76% (DAP 94: 88.6 vs 40.7). So the depth/dice loss in training has never been able to reach 100%
even for a perfect prediction, its gradient partly pushes the whole plant toward a wrong offset, and the two mature
plants that regressed under test-time refinement (DAP 89/94) are exactly the large-offset cases. Fix in progress:
render the prediction in the input's frame (plant-bbox-centred; `--plant_centered` in
`eval_test_time_refinement.py`, re-running; the training render block needs the GT bbox centre per sample next).
**12:55 — test-time refinement is the biggest lever found today (results report §11.7).**
`plant_recon/eval/eval_test_time_refinement.py`: after sampling, optimise each plant's node positions, scales and
phytomer latents for 40 Adam steps against the INPUT canopy height map with the training render loss (no GT), keeping
the step with the lowest input loss (`--keep_best`). v10 ep95 on the 20 eval plants, strict 256 px protocol: **34.9 →
46.9** (DAP > 15: 38.2 → 51.2); pos+scale alone +9.8, latent alone +7.8. Every training-side lever plateaued at 35, so
this — analysis-by-synthesis at inference — is the path to "rendering close to the original": the renderer recovers
per-node information the network does not read from the image. Follow-ups running: 80 steps, and the same on the
full-data baseline ep135 (model-agnostic check). Self-conditioning re-sampling (`--self_cond_passes`) gave nothing.
**12:45 — `outputs/logs/` is now organized by start date** (Heesup: "Can't tell which one is the latest"): one folder
per day, `outputs/logs/YYYYMMDD/`, holding that day's job logs, local-run logs, `run_*` panel folders (with a
relative `run.log` link) and evaluation folders. Paths quoted earlier in this guide as `outputs/logs/<x>` now live
under the date folder of their run (9/14 items under `20260914/`, 9/15 under `20260915/`). The launcher writes new run
folders into today's folder; `tools/organize_logs.py --apply` files anything left at the top level once it has been
quiet for 20 minutes (live runs are never moved). `outputs/logs/README.md` describes the layout.
**12:25 — fourth reading and the current set.** Four readings put v10 full at a strict P mean of 35.0, the combination
at 33.5, scheduled TF at 28.0 and baseline-on-10% at 27.0 (results report §11.4); nothing rises further on 10k plants.
The combination run was stopped on this node at epoch 95 (v10 is its superset) and the **v10 + unfrozen backbone** variant
started in its place (`outputs/logs/local_sub10_v10_unfreeze.log`, `sub10_v10_unfreeze/`, `FREEZE_BACKBONE=0`,
backbone lr ratio 0.3), testing whether node precision is capped by the frozen DINOv2 features. On the cluster:
scheduled TF (`38274495`, to ep95), v10 render-every-sample (`38274747`, ep80: 34.4 after 2 epochs), v10 lr 2e-4
(`38274748`, started 12:05). The 10% baseline finished its budget at epoch 95.
**`low` cannot run our runs now (10:00)**: its GPU nodes have idle A100/H100s but their RAM is fully allocated by other
  users' jobs (5–13 GB free per node) and one training needs ~15 GB (6 GB main + 8 workers × 1.1 GB). The two low jobs
  were cancelled and the three runs chained on the baseline's slot: `38274220` latent_norm (46–80) → `38274221`
  scheduled TF (46–75) → `38274222` render→latent (46–70), ~2 h each at 2 GPUs.

**Node caveat (21:05):** `gpu-10-50`, where both local runs run inside the OnDemand desktop job `38252204`, is
`MIXED+DRAIN` since 17:20 (`Reason=Kill task failed (JobId=38249157)`, an automatic SLURM drain). Running jobs are not
affected and the desktop job has ~36 h left, but an admin reboot to clear the drain would kill both local runs. Both
resume from their checkpoints: geometry saves every epoch (`hierarchical_fm_v9_s3geom/`), gt_nodes every 5
(`hierarchical_fm_v9_gtnodes/`, next at 55). To move either to the cluster:
`sbatch --partition=low --account=publicgrp --gres=gpu:a100:2 --time=7-00:00:00 --requeue --export=ALL,AUTO_RESUME=1,STAGE3_GEOMETRY=1,OUTPUT_DIR=outputs/checkpoints/hierarchical_fm_v9_s3geom scripts/train_hierarchical_flow_matching.sh`
(gt_nodes: `STAGE3_GT_NODES=1,OUTPUT_DIR=…_gtnodes` instead). The baseline `38257989` ends at its 24 h limit 2026-09-15 ~15:53
— resubmit with `AUTO_RESUME=1` into `hierarchical_fm_v9/` before then.

**What follows.** The render loss is the only per-node image signal that does not pass through the flow loss, and the
latent rows were detached from it. `--render_to_latent` (`RENDER_TO_LATENT=1`, same commit series) keeps the latent
block attached in the render block; the smoke run's `FM_RENDER_GRAD_PROBE` shows the render loss on the velocity head's
latent rows at |g| 0.02–2.7 per step. **Run 4 submitted 17:45: job `38260124`** (`low`/publicgrp, 2×A100, `--requeue`,
`AUTO_RESUME=1 RENDER_TO_LATENT=1`, from epoch 45 into `hierarchical_fm_v9_r2l/`, log
`outputs/logs/hierarchical_fm_38260124.log`). Read its per-node latent R² (ablation summary `_latent`) before its IoU. Two
further levers are cheap and principled if that is not enough: scale the latent to unit variance for the flow (an
SD-style scale factor; changes the velocity head, so fine-tune from epoch 45) and sample t toward 0 where the
conditioning matters. The gt_nodes run answers the other half: if per-node R² rises when the nodes are right, the
latent path is starved by node error after all.

### 0-B.3 Working-tree hygiene

- Anything not in `git status` clean + the two files named in 0-B.2 is either untracked run artifacts
  (`outputs/eval/`, `outputs/checkpoints/`) or belongs to the archived-launcher index (`archive/README.md`). Keep `scripts/`
  to the current launchers (`train_hierarchical_flow_matching.sh`, `generate_helios_dataset_jobs.sh`).
- Cluster etiquette: Heesup's `regen_*` jobs (geminigrp) hold the group GPU quota — do not cancel; training goes on
  `gpu-6000_ada-h` / `low`. The OnDemand desktop (38252204) must not be killed.


## 2. Active SLURM Jobs (as of 2026-09-11 ~13:15 PDT)

| Job ID | Name | Status | Node | Notes |
| :--- | :---: | :---: | :--- | :--- |
| **38230613** | `ondemand/sys/dashboa` | RUNNING | `gpu-5-58` | User's interactive OnDemand desktop — **DO NOT CANCEL** |
| **38237555** | `hierarchical_fm` | **RUNNING** | `gpu-10-50` | 4x RTX 6000 Ada, **Current all-step grad_norm=inf deadlock** — needs restart |
| 38236803 | `hierarchical_fm` | CANCELLED | — | Previous attempt |
| 38236800 | `hierarchical_fm` | CANCELLED | — | Previous attempt |

### ⚠️ Current Training Status (Job 38237555 — CRITICAL)

**Job 38237555** entered an **all-step grad_norm=inf deadlock** after Epoch 40:
```
[Recovery] Step 1~31+: grad_norm is NaN/Inf (inf) | Culprit params (0): -> SKIPPING STEP
Epoch 040 | Loss: 0.0000 | ... | Vel: 0.0000 ... | VRAM: 26.9/47.4 GB
```
- `Culprit params (0)` — parameter name tracking failed (grad is None or tensor itself is already inf)
- Loss frozen at 0.0000 → every step skipped, causing a **completely frozen** state
- **Must cancel immediately, fix root cause, and restart**

### Candidate Root Causes (Unresolved as of 2026-09-11)
| Candidate | Symptom | Investigation Method |
| :--- | :--- | :--- |
| `init_logits` Float32 overflow | $\ell_k = (N_{phy}+m-k)/\tau \to 96.25 \to e^{96.25} = \infty$ | clamp to [-15, +15] |
| Stage 2→3 gradient leak (missing `.detach()`) | `tgt_velocity = z1 - z0` where `z0` contains `pred_phytomer_pos` without detach | verify all `z_0` construction |
| NaN propagation from Depth/Dice render loss | nvdiffrast nan on degenerate mesh at early epochs | `torch.nan_to_num` on render output |

### Recommended Restart Procedure:
```bash
# 1. Cancel current job
scancel 38237555

# 2. Check init_logits clamp patch
grep -n "init_logits\|soft_margin\|clamp" plant_recon/models/hierarchical_part_flow_matching.py | head -20

# 3. Check z_0 detach
grep -n "z_0\|detach" plant_recon/training/train_hierarchical_flow_matching.py | head -30

# 4. Restart
sbatch scripts/train_hierarchical_flow_matching.sh
```

### Failed-launch forensics (earlier)
| Job | Failure | Root cause | Fix commit |
| :--- | :--- | :--- | :--- |
| 38224489 | `UnboundLocalError: _sync` | Probe path referenced helper before def | `db4e493` |
| 38225532 | `UnboundLocalError: prof` | Same class, `prof`/`_t_fbs` before def | `3bdae4a` |
| 38226665 | `AsStridedBackward0` version conflict `[351,10,3]` | Packet-scale view modified inplace across fwd/bwd | `69ce959` |
| 38233491 | DDP `.color_palette` marked ready twice | `color_palette` was `nn.Parameter` outside `forward()` | `9dd45be` (`register_buffer` + `.detach()`) |
| 38233914 | `[256, 3]` version conflict in `canonical_rays` | DDP `broadcast_buffers=True` modified ray buffer inplace | `5255efa` (`clone()` + `broadcast_buffers=False`) |

---


## 3. Session Changes & Technical Breakthroughs (2026-09-08)

### 3.1 Algorithm Improvements (Commits `0389ca5`, `5b9fbf0`, `54579bc`)
- **Idle Slot Damping**: Added $L_2$ velocity regularizer $\mathcal{L}_{\text{idle}} = 0.05 \sum_{i \notin \mathcal{M}} \|\mathbf{v}_i\|^2$ for unmatched slots to prevent ghost organs during vegetative stages (DAP < 40).
- **Top-k Threshold Guard**: `valid_topk = top_k_indices[combined_prob[b, top_k_indices] > 0.15]` prevents dormant ghost slot emergence.
- **Backbone & Node Acceleration**: `loss_phytomer_pos` weight increased to **4.0**, DINOv2 backbone LR increased to **6e-5** (`args.lr * 0.3`).

### 3.2 Batch Size Scaling (Commit `0389ca5`)
- Batch size scaled to 48 per GPU (global batch 192).
- VRAM stable at **37.2 GB (78.6%)** across all 4 GPUs on `gpu-10-54`.
- Throughput: ~50 seconds per epoch (70 steps/epoch).

### 3.3 Dataset Cache Filtering (Commit `027118d`)
- `PartArrayDataset` filters XML paths by `cache_dir` `.pt` files upfront. Prevents dimension mismatch during live Helios data generation.

### 3.4 Helios 24-Hour Job Extension (Commit `97420ee`)
- Set default `TIME_LIMIT="24:00:00"` for all 40 shards to accommodate mature crop simulation (DAP 73–100 takes 7–8 hours).

### 3.5 Validation Decoupled to Every Epoch (Commit `c587767`)
- Added `--eval_every 1` to `train_hierarchical_flow_matching.py` for per-epoch self-consistency checks without bloating checkpoint storage.

### 3.6 Checkpoint Resume Support (Commit `0435325`)
- Added `--resume` flag with optimizer state and `CosineAnnealingLR` alignment.

### 3.7 10-Slot Phytomer Ordered Assembly (2026-09-11)

**Problem**: The previous flat concatenation approach scrambled the Internode-Petiole-Leaflets order per phytomer, collapsing plant geometric topology.

**Solution**: Implemented `assemble_phytomer_ordered_14d_tensor` in `phytomer_packets.py`:
- **Slot Order**: `[0]=Internode | [1]=Petiole | [2,3,4]=Leaflets | [5]=Peduncle | [6,7,8,9]=Repro`
- Restored Internode to slot 0, recovering 1:1 binding between internode (stem) - petiole - leaflets
- Fixed bug discarding nodes $\le$ 5mm (threshold lowered to 0.1mm)
- Restored cotyledon opposite-leaf structure → achieved exact part count agreement ($\Delta=0$) across all DAPs

### 3.8 Per-Organ Mask IoU Roundtrip Diagnosis (2026-09-11)

Execution results for `plant_recon/eval/eval_13d_xml_organ_masks.py` (`fig10_helios_per_organ_mask_comparison.png`):

| DAP | Foreground IoU | Mean Organ IoU | Internode | Petiole | Leaf | Peduncle | Flower | Fruit | Depth PSNR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 (Seedling)** | **95.7%** | **48.3%** | — | 1.0% | 95.6% | — | — | — | **38.90 dB** |
| **50 (Branching)** | **94.1%** | **36.0%** | 0.0% | 14.6% | 93.3% | — | — | — | **26.12 dB** |
| **90 (Fruiting)** | **87.4%** | **21.7%** | 12.9% | 4.9% | 80.6% | 7.0% | 19.2% | 5.7% | **21.43 dB** |

**Key Findings**:
- **Leaf** class achieved 80~96% IoU across all stages — leaf geometry accurately recovered
- **Internode/Petiole** IoU is very low (0~15%) — likely caused by stem/petiole **coordinate offsets** or **radius/length scale** mismatches in 14D XML
- At DAP 50, Internode IoU = 0.0% → stems rendered at entirely different positions
- **Next Step**: Numerically compare world pose in `extract_part_tensor` ↔ Helios XML inverse kinematics (IK) transform agreement

*(Result figure: [`docs/assetsfig10_helios_per_organ_mask_comparison.png`](assets/fig10_helios_per_organ_mask_comparison.png))*

![Figure 10 helios per organ mask comparison](assets/fig10_helios_per_organ_mask_comparison.png)

---


## 4. Current Checkpoints & Job Status

Checkpoints saved on disk:
```bash
# Hybrid decoupled 3-stage run (Sep 11) — 390 MB each, new architecture
outputs/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt  (390 MB, Sep 11 11:57)
outputs/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_050.pt  (390 MB, Sep 11 05:13)
outputs/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_075.pt  (390 MB, Sep 11 07:19)
outputs/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_100.pt  (390 MB, Sep 11 09:25)

# Old organ-mode 500-epoch run (Sep 7–9) — 552 MB, OLD architecture (incompatible)
outputs/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_125.pt ~ epoch_500.pt  (552 MB, Sep 7–9)

# PhytomerVAE v3 (ACCEPTED DEFAULT)
outputs/checkpoints/phytomer_vae_v3/phytomer_vae_64d_best.pt  (1.9 MB, Sep 10 14:54)
outputs/checkpoints/phytomer_vae_v3/phytomer_vae_64d_last.pt  (1.9 MB, Sep 10 14:54)
```

> **WARNING**: epoch_025~100.pt (390 MB) = new 3-stage hybrid arch. epoch_125~500.pt (552 MB) = old organ-mode arch. These are **NOT interchangeable** — model structure differs.

### Current state (2026-09-11) — v3 stack is the default:
- **VAE v3** `phytomer_vae_v3/` is the launcher default
  (`PHYTOMER_VAE_CHECKPOINT` in `train_hierarchical_flow_matching.sh`). Higher
  recon (0.070 vs v2's 0.0263) is EXPECTED — normalized-scale targets have a
  wider distribution; cls 100% retained. v2/v1 checkpoints kept for lineage only.
- **pkt cache v3**: `dataset/cache/cowpea_curv26_pkt/` holds **100,000/100,000**
  files stamped `pkt_version: 3` (10-slot, ABSOLUTE packets + normalized-space
  latent). Spot-checked. v1/v2 files were auto-invalidated by the version gate
  (stale files are skipped → on-the-fly fallback, never silently loaded).
- **76D flow state**: `[pos(3) | rot(6) | s_a(3) | latent(64)]`.
  `s_a` = petiole (slot-1) scale row `[len, radius, unused]`, normalized-space
  packets, Stage-2 `scale_head` bridge (softplus, smooth_l1 weight 2.0).
  Physical floors: petiole len ≥ 0.25 (5mm), radius ≥ 0.025 (0.5mm).
- **Macro-head fixes** (commit `b3c9e30`): `phy_head` bias-init at dataset mean
  (`log(50)`), `phy_count_weight` 2.0, `backbone_lr_ratio` 0.3, DAP clue
  `dap_embed` (zero-init) into Stage-2/3 queries. Local test: pred count
  1.6 → 13.5 in 5 epochs (was stuck at 1.7).
- **Render pipeline**: epoch-gated render grads (`--render_grad_start_epoch 5`),
  probe token reuse (`image_tokens` kwarg), semantic color palette `(13,3)`
  nn.Parameter (cos-color loss alive), CUDA-synced timers (`backward/probe/other`).
- **Assembly rules verified**: stem base T-joint (roundtrip median 0.02cm /
  p99 0.14cm, was 2.7cm), curved-peduncle-tip fruit/flower assembly (100%),
  leaflets 0.8/0.8/1.0 petiole curve, `ORGAN_LEAF=5` in `DETERMINISTIC_ORGAN_TYPES`.
- **Pipeline**: `generate_cache.py` (`cache`/`pkt` modes, `PKT_VERSION=3` gate in
  worker skip + save stamp); standalone pkt backfill tools deleted.
- **Uncommitted**: `--detect_anomaly` flag in `train_hierarchical_flow_matching.py`
  (parser arg + `set_detect_anomaly`). Commit together with the next fix.

### Resume (recommended order):
```bash
# 1. OPTIONAL: quick local smoke first (subset4k, catches inplace/DDP errors in ~min)
#    env FLOW_GRANULARITY=phytomer CAPACITY_WARMUP=0 CAPACITY_FULL=1 \
#      CACHE_DIR=dataset/cache/cowpea_curv26_subset4k \
#      WANDB_RUN_NAME=smoke-4k-v3-76d bash scripts/train_hierarchical_flow_matching.sh
#    (with --max_train_samples small + --detect_anomaly if debugging)

# 2. Cluster submission (defaults now carry v3 VAE + pkt v3 cache)
sbatch scripts/train_hierarchical_flow_matching.sh
# env overrides if needed: FLOW_GRANULARITY=phytomer CAPACITY_WARMUP=0 CAPACITY_FULL=1

# 3. WATCH THE FIRST 3 MINUTES: if it survives probe_optimal_batch_size +
#    first train steps, the AsStrided fix holds. Epoch-1 loss should decrease.
```

---


## 5. Next Steps (Priority Order)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P0** | **Fix grad_norm=inf deadlock & resubmit** | Cancel 38237555. Patch `init_logits` clamp [-15,+15] in `hierarchical_part_flow_matching.py`, verify `z_0.detach()` in `train_hierarchical_flow_matching.py`, add `torch.nan_to_num` on render outputs. Run local smoke test then `sbatch` |
| **P1** | **Diagnose Internode/Petiole IoU=0~15%** | Compare `extract_part_tensor` world pose → Helios IK → XML → re-render numerically. Check coordinate convention (Z-up vs Y-up) between PyTorch mesh builder and Helios XML parser in `part_tensor_to_40d.py` |
| **P2** | **Verify 10-slot ordered assembly roundtrip** | Run `test_phytomer_ordered_assembly.py` with DAP 15/40/75; confirm Δ-parts=0 across all growth stages |
| **P3** | Epoch-1 sanity after resubmit | Check `outputs/logs/<YYYYMMDD>/run_<jobid>/hierarchical_self_consistency_epoch_001.png` (each run's panels sit beside its own `run.log` symlink; they used to overwrite each other under `docs/results/assets`): loss ↓, pred count ~50, ClsAcc rising, no Recovery-skip lines |
| **P4** | Monitor 6D rotation convergence | Panels epoch 25/50; s_a (petiole len) should track DAP growth |
| **P5** | Evaluate Bidirectional Chamfer Distance | Add max/mean distance GT→Pred to avoid one-way clustering metric bias |
| **P6** | Backbone A/B (DINOv2-scale vs frozen runs) | `archive/slurm_scripts/submit_backbone_ablation.sh` — only after single-run training is stable |

---


## 7. Training Loss Progression

### Old organ-mode run (Job 38146809 — for reference baseline)
> Superseded by 3-stage phytomer-mode. Use for metric baseline only.

| Epoch | Total Loss | VelLoss | AncPosLoss | PhyLoss | ExistLoss | DepthLoss | DiceLoss | ClsAcc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 166.07 | 3.07 | 0.300 | 33.20 | 1.20 | 0.15 | 0.88 | 42.0% |
| **10** | 8.52 | 1.10 | 0.035 | 3.80 | 0.35 | 0.09 | 0.74 | 58.5% |
| **25** | 5.52 | 0.85 | 0.016 | 2.40 | 0.24 | 0.07 | 0.70 | 63.2% |
| **40** | 4.12 | 0.70 | 0.011 | 1.30 | 0.18 | 0.06 | 0.67 | 64.8% |
| **50** | **3.58** | **0.57** | **0.0089** | **0.85** | **0.15** | **0.065** | **0.65** | **66.3%** |

### Current 3-stage phytomer run (Job 38237555 — DEADLOCKED at Epoch 40+)
| Epoch | Status | Notes |
| :---: | :---: | :--- |
| 1~39 | Training | grad_norm occasionally inf, recovery skips |
| 40+ | **DEADLOCKED** | All steps Recovery-skip, Loss=0.0000, needs restart |

---



---

# From `current-state.md`

## 🚨 CURRENT SYSTEM STATE (2026-09-16, Claude Code handover)

Read this section first; the 2026-09-14 table below it is the previous state, still accurate for
anything this section does not contradict.

### Synthetic track — plateaued, and the plateau is the finding

| Component | Status | Details |
| :--- | :---: | :--- |
| **Raw model quality** | 🟡 PLATEAU | Strict-protocol P sits at **33–39** no matter what. Every training lever tried over 2026-09-15/16 — Stage 3 geometry, teacher forcing, latent normalisation, multizoom, node-token window, existence-count weight, render-to-latent, render-to-exist, a render-fraction curriculum, 10% vs 100% of the data — lands in that band. Treat this as an architecture/information limit, not a tuning problem, until something structural changes. |
| **Test-time refinement** | 🟢 THE LEVER | 40 AdamW steps on node positions / scales / latents against the input CHM lifts the same checkpoints to **P 67–68** (`eval_test_time_refinement.py`, defaults now `--input_camera`, `reg_scale 5`, `reg_latent 0.5`, four zoom targets). This is the largest single gain found and it needs no retraining. |
| **Guided sampling** | 🔴 NOT WORTH IT | DPS-style render guidance inside the ODE (`eval_guided_sampling.py`, `guidance_fn` hook in `sample_ode`) is roughly neutral alone (35.2 → 37.3 at `--guide_start_frac 0.5`) and actively **hurts** when chained into refinement (63.2 vs 67.6). The two mechanisms interfere; do not combine them without a new idea. |
| **Best lineage** | 🟢 | `hierarchical_fm_v10_cam/` — full data + input camera + EMA. ep160 raw 38.7 / refined **68.3**, the best numbers to date. |

### Real-image track — the pipeline runs end to end; the reconstructions are not yet usable

**Be blunt with yourself here: no real-image reconstruction has been shown to be geometrically
correct.** What exists is plumbing that works and a set of well-localised reasons it does not.

| Component | Status | Details |
| :--- | :---: | :--- |
| **Plant detection** | 🟢 WORKS | AgML `gemini_plant_detection_2022` box detector mAP50 **0.975**; Roboflow `t4_plant_weed_seg` instance-segmentation detector mAP50 0.844 plant / 0.612 weed. Detection is not the bottleneck. |
| **Whole-frame multi-plant pipeline** | 🟢 RUNS | `use_cases/real_world/eval/run_multiplant_scene.py`: frame → detections → plot metres → Helios `params.json` → per-plant XML → model state → differentiable-renderer refinement → XML → one rendered plot. Follows the `Image2PlantArchitecture_v2` params.json convention. |
| **Helios XML export fidelity** | 🟢 FIXED | Exports now match the source structurally (leaves 76/77, 100/101, 160/161, 130/131; shoots 5/5, 8/8, 12/12, 10/10). Previously a packet-decoded plant exported as ONE unifoliate shoot, which cost two of every three leaflets — see [20260916-multiplant-scene.md §7](../experiments/20260916-multiplant-scene/20260916-multiplant-scene.md). |
| **Network cold start on real crops** | 🔴 BROKEN | Collapses: 14 of 20 AgML crops sampled a **single live phytomer**; on the Roboflow frame two plants came back with 8 and 38 organs. Refinement cannot fix what is not there (nodes move 0.0–1.0 cm). |
| **Refinement on real crops** | 🔴 INFLATES | Against a loose target it grows leaves rather than fitting them: leaf scale max 0.0865 → **0.208–0.221** (2.4–2.6x) on the AgML frame. The per-component absolute cap (`--scale_abs_max_len/_rad`, calibrated 2026-09-16) stops the flat-polygon failure but bounds the phytomer scale `s_a`, not realized organ size. |
| **Does a real mask fix it?** | 🔴 NO | On the Roboflow frame (real instance masks, a much tighter Dice target) the Helios-init data losses are **0.76–3.03**, i.e. WORSE than the AgML bbox-target numbers (0.58–0.67). That confirms the AgML numbers were inflation filling a loose target, and that fit quality on real images is genuinely poor. |
| **Pixel → metre scale** | 🟡 ASSUMED | The rover camera has no intrinsics here; only its 1.5 m height is known. `--plot_width_m` (default 1.3, the Davis plot width) sets the scene's absolute scale by assumption. |
| **Stage 3 conditioning, settled (2026-09-17)** | 🔴 THE LIMIT | Follow-up run `sub10_v10_cam_s3reg_tf` (regression head conditioned on GT nodes, p 0.5, 1 cm jitter): with correct node positions the head still outputs the dataset mean exactly (latent RMSE 4.656 vs the mean latent's 4.658, R² +0.001; spread 0.07 vs GT 0.177). Node error is not what hides the per-node shape and neither is the flow objective — what Stage 3 is *given*, one bilinear token at 7.5 cm pitch from a frozen backbone, does not contain the phytomer's shape. Per-node high-resolution ROI patches are now the necessary next change for Stage 3. The flow stays as the readout: its latents have the only realistic spread (0.183 vs GT 0.177) and are the best refinement start (flat 66.3 vs 61-65 regressed/mean). Plan §6.2. |
| **Appearance: the target is real photos, not Helios (2026-09-17)** | 🟢 BUILT | Measured with DINOv2 MMD against 220 real plant crops cut at the cache's own framing: the new augmentation (real soil from the rover frames + shading and cast shadow computed from the CHM gradient + photometric jitter, `plant_recon/dataset/appearance_augment.py`, `--appearance_augment`) moves the training distribution toward real photos at every zoom (0.61→0.42, 0.54→0.34, 0.55→0.43, 0.54→0.47), while the **Helios raytraced render is further from real than the flat render** at zooms 2x-8x. So §5's 10-point drop shows brittleness to any appearance shift, not that Helios is the target; training toward Helios (plan §2.1 as first written) would have been a mistake. Helios keeps its value as a held-out robustness probe. Plan §7. |
| **Real crop framing, fixed (2026-09-17)** | 🟢 FIXED | The real-image crops were cut at 1.2× the detector box while training renders a fixed 1.2 m ground window: **3.7× too large** for a typical detection (480 px vs 1795 px), which is the likely mechanism behind the over-predicted DAP. `base_window_px()` now cuts the fixed metre window (`--plot_width_m`, legacy behaviour at 0). Plan §7.3. |
| **Stage 3 regression readout (2026-09-17 01:15)** | 🔴 NEGATIVE | `--stage3_regression` (same decoder and conditioning, z_0 = 0 / t = 0 so the state is regressed directly) fine-tuned 10 epochs from v10_cam ep160 on the 10% protocol (`sub10_v10_cam_s3reg`, paused at ep170): the head converged to the conditional mean in two epochs (latent R² −0.59 → −0.06, per-dim spread 0.18 → 0.07, GT geometry + predicted latent 47.0 → 41.3, P 38.7 → 37.2) and is a worse refinement init than the mean latent (flat 61.3 vs 66.6, Helios RGB 55.3 vs 58.3). Conclusion: the conditioning at 7.5 cm token pitch gathered at Stage 2's nodes carries no per-node shape a direct readout can extract; the objective was not the limit. IoU rewards realistic spread (flow's plausible-but-wrong latents +8 over the mean under GT geometry), so the flow stays as a proposal distribution. Next: GT-node-conditioned regression as the node-error vs resolution diagnostic, then the per-node high-res ROI patch — plan §6.1. |
| **Appearance gap, measured (16:30–18:30)** | 🔴 LARGE | Same 20 eval plants, same ep160 EMA checkpoint, same CHM planes and refinement targets, only the 12 RGB planes swapped from the flat cache render to a Helios raytraced re-render at the cache's exact framing (`render_helios_eval_crops.py`, framing verified per plant): strict P **38.1 → 28.6** raw, **66.3 → 59.2** refined. Seedlings (DAP ≤ 15) go 31.4 → 12.0 refined with the DAP probe off by 51 days (DAP 2 read as 67); mature plants (> 75) 48.8 → 32.4 raw with DAP under-read by 25 days and 40% fewer active nodes; mid-season plants unchanged. This reproduces the real-image failure modes (wrong DAP, existence collapse) on synthetic geometry, so pixels alone account for them — see [20260916-sim-to-real-assessment.md §5](../experiments/20260916-sim-to-real-assessment/20260916-sim-to-real-assessment.md). The network reads only RGB (`DINORayEncoder.forward`), and training RGB is flat with no augmentation, which is the lever. |

**Where to start next on this track** (in the order most likely to matter; updated 18:30 after the
appearance-gap measurement, [20260916-sim-to-real-assessment.md §5](../experiments/20260916-sim-to-real-assessment/20260916-sim-to-real-assessment.md)):
1. Close the appearance gap in training: fine-tune the ep160 lineage on Helios raytraced RGB crops
   at the cache framing (`render_helios_eval_crops.py` shows the recipe; the training set needs the
   same re-render or a crop of the auto-framed dataset JPEGs) plus ordinary augmentation, and
   re-measure with the same flat-vs-Helios protocol. The single-phytomer collapse and the wrong DAP
   on real crops are what this gap does to the DAP probe and the existence gate.
2. Fix the real crop framing: a fixed 1.2 m ground window from the pixel-to-metre scale, not 1.2×
   the detector box (`real_plant_crop_utils.py`, plan §1.3).
3. Bound realized organ size, not `s_a`, so refinement cannot inflate leaves regardless of target;
   make existence a continuous variable in refinement.
4. Only then revisit the cold-start comparison, on the segmentation-mask source.

---

## Previous system state (2026-09-14 ~11:30 PDT, Claude Code handover)

| Component | Status | Details |
| :--- | :---: | :--- |
| **Main training** | 🟢 CLUSTER, 2 GPU | Job **`38253656`** (gpu-6000_ada-h, gpu-10-54, 24 h): v9 lineage, `AUTO_RESUME` from `hierarchical_fm_v9/hierarchical_fm_epoch_025.pt` at batch 48/GPU, `NUM_WORKERS=8`, on `25c2251` — the training step without per-sample Python: vectorized packet assembly (`f3c5366`), parent links in the loader workers + one VAE encode per batch + one nvdiffrast pass for all rendered plants (`a1a82bc`), pointer-jumping `chain_phytomers` + batched matcher over the padded batch (`25c2251`). Step at batch 48: 1.9 s this morning → 0.15-0.24 s (1 GPU). Output `hierarchical_fm_v9/`, log `outputs/logs/20260914/hierarchical_fm_38253656.log`, panels `outputs/logs/20260914/run_38253656/`. Predecessors today: `38253029` (epochs 16-21, ~9 min/epoch), `38253401` (epochs 22-25, ~7 min/epoch; IoU 32.6-35.5%, node RMSE 1.8-1.9 cm). |
| **Cluster jobs (queued)** | ⚪ NONE | The `low`-partition resubmissions (`38249632`, `38250275`) were cancelled 9/14 10:13. The lineage lives only in `38253656`; it hits the 24 h limit ~9/15 14:45 — resubmit then with `AUTO_RESUME=1` (same OUTPUT_DIR) or queue a `low`/`publicgrp` continuation with `--requeue`. |
| **Stage 2 gradient burst** | 🟢 FIXED | The coarse `nn.TransformerDecoder` had no final LayerNorm, so its raw residual stream fed the bf16 phytomer self-attention (design doc §1.9.2). `FM_DECODER_FINAL_NORM=1` + `FM_SELFATTN_FP32=1` (defaults): 0/8 burst steps on the frozen burst state vs 8/8. Consider it closed once the run passes epoch ~30 (the v8 lineage burst at 27-28). |
| **PhytomerVAE** | 🟢 DEFAULT | `phytomer_vae_v9_tl_rw4_20k` — 128D hybrid (48 coarse + 10×8 per-slot residual), terminal-last packets, 20,000 files. Export VAE (eval scripts default to it, `PHYTOMER_TERMINAL_LAST=1`) and the FM training VAE of the v9 run. |
| **Packet cache** | 🟢 VAE-INDEPENDENT | `dataset/cache/cowpea_curv26_pkt_v9/` — 100,000 files, `pkt_version` 7, terminal-last packets. Since 2026-09-14 the cache stores only packets/presence/centers/refs/keys and FM encodes the Stage-3 latents on the fly from them, so the cache is VAE-independent (a new VAE needs no rebuild). The packet ORDER still must match the VAE: v9 runs pair this cache with `PHYTOMER_TERMINAL_LAST=1`; v8 runs keep `cowpea_curv26_pkt/` (pkt 6, `PHYTOMER_TERMINAL_LAST=0`). |
| **Helios round-trip, exact_gt DAP 10/50/90** | 🟢 SOLVED | fig14: IK-only **99.9 / 99.6 / 97.7%**, VAE round-trip **93.7 / 98.3 / 95.8%** FG IoU (`docs/experiments/20260914-stage2-burst-fix-roundtrip/assets/fig14_phytomer_vae_helios_roundtrip.png`). |
| **Helios round-trip, dataset DAP 15/40/75** | 🟢 SOLVED | fig12: packet path **98.3 / 98.2 / 95.6%**, VAE **95.1 / 96.8 / 95.6%** (was 81.8 / 92.2 / 54.6 on the morning of 2026-09-14; `docs/experiments/20260914-stage2-burst-fix-roundtrip/assets/fig12_phytomer_10slot_helios_roundtrip.png`). |
| **Topology targets** | 🟢 FIXED | `gt_parent_links(..., internode_base=)` resolves 97.8% of lateral branch points (was 66.2%; the training call site passes the decoded base). `chain_phytomers` follows drooping shoots (same-shoot links 100% on 30 plants) and keeps the cotyledon node as shoot 0 (`root_own_shoot`). |
| **Export (14D → Helios XML)** | 🟢 | Analytical converter + **stem IK** (`part_tensor_stem_ik.py`) + **leaf IK** (`part_tensor_leaf_ik.py`), both on by default (`PART_TENSOR_STEM_IK`, `PART_TENSOR_LEAF_IK`). |
| **The 09-07/08 "45.4%" Option B panel** | 🟢 EXPLAINED | A 4-plant random-batch mean at one of six evaluations (series 39/26/34/27/45/49). The same checkpoint with its exact code scores **24.8%** on today's 20-plant set (today's runs: 30–34). Not a bar; its one transferable property is the unit-variance latent (takeover guide §0-B.9). |
| **Heesup's regeneration jobs** | ⛔ DO NOT CANCEL | `regen_shard` / `regen_synth` / `regen_mopup` (geminigrp) — they hold the group's GPU quota; training goes to `low`/`publicgrp`. |

![Figure 14 phytomer vae helios roundtrip](../experiments/20260914-stage2-burst-fix-roundtrip/assets/fig14_phytomer_vae_helios_roundtrip.png)

![Figure 12 phytomer 10slot helios roundtrip](../experiments/20260914-stage2-burst-fix-roundtrip/assets/fig12_phytomer_10slot_helios_roundtrip.png)

---

## Next steps as of 2026-09-14 ~11:30 (superseded, kept for the record)

Every job named below has ended; the v10_cam lineage finished at ep160 on 2026-09-16 (raw 38.7 / refined 68.3). The P2 cell is the compressed 2026-09-14/15 chronology; the full version is takeover guide §0-B.9.

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P0** | **Watch the v9 run past epoch 30** | canary hits, Stage 2 Scl/Ord/Ext losses, and the per-epoch self-consistency panels; the goal is rendering that matches the original, not the loss alone. With the assembly vectorized, epochs are ~6x faster; re-measure whether a larger batch now pays (the 256 measurement was made with the old assembly loop dominating). |
| **P1** | **Keep the lineage alive past the 24 h limit** | `38253656` ends ~9/15 14:45; resubmit with `AUTO_RESUME=1` into `hierarchical_fm_v9/` (or a `low` job with `--requeue`). |
| **P2** | **Stage 3 A/B, three runs (takeover guide §0-B.9)** | From epoch 45: baseline `38257989` (cluster), geometry run `STAGE3_GEOMETRY=1` (local, `local_s3geom_ab2.log`), teacher-forcing run `STAGE3_GT_NODES=1` (local, `local_gtnodes_ab.log`). Baseline 46-65 mean 31.3, 66-82 mean ~35 (best 38.1) — but under the strict 256 px protocol its P is flat (27.5 at ep50 → 27.3 at ep80) and its existence head under-predicts (57 of 73 GT nodes active), so read the in-training IoU as a trend only. Geometry 47-55 mean 31.1 = baseline over the same span (Vel still falling; ablation P 27.5→29.6 but its sampled latent stays noisy, std 0.46 vs GT 0.18). gt_nodes 46-52: 32.8/31.2/35.8/**39.4**/32.6/32.2/33.9% (mean 34.0 vs baseline 30.6 at the same epochs, deployable output). Overnight (9/15 08:40, strict protocol at the latest checkpoints): baseline ep135 P 28.9 (flat, existence 55/73), geometry ep76 P **33.9** (climbing), gt_nodes ep70 deployable 29.4 / teacher-forced 45.3, latent R² 0.41; in-training the baseline fell back to ~30 after peaking at 36. **10% protocol readings (9/15 12:20, results report §11.1-11.3):** every run plateaus within 10 epochs on 10k plants — geometry family (combination 34.2, **v10 full 35.1-36.0** strict P, young plants 18) vs 26-29 for baseline-on-10%, standardized latent, scheduled TF (deployable; teacher-forced 44) and render→latent; no run memorizes the training plants, no run's latent carries per-node information, and the coverage loss does not spread the nodes (the matcher already pairs every GT phytomer; the deficit is existence gating and 4-6 cm node error at 7.5 cm per DINOv2 token). Running v10 variants: render every sample (`38274747`), lr 2e-4 (`38274748`), unfrozen backbone and 3×3 token window + t0 (local). **Test-time refinement** (`eval_test_time_refinement.py`, 40 Adam steps of nodes/scales/latents against the input CHM, no GT) lifts v10 ep95 from 37.0 to **44.3** strict P (DAP > 15: 41.7 → 52.0) — the largest gain of the day; variants with input-loss model selection running. Its ep50/ep55 checkpoints: with GT nodes the latent is worth +7.8/+9.6 IoU over the mean latent (baseline +2.5) and per-node R² 0.115/0.187 (baseline ≈ 0) -> node error does starve the latent; with Stage 2's nodes the strict protocol gives P 20.2/23.9 (baseline 27.3) = exposure bias, recovering. `low` is blocked by other users' RAM, not GPUs, and Heesup says geminigrp's `gpu-6000_ada-h` allows up to 8 high-priority GPUs, so the runs run as six independent 1-GPU jobs there (first started 10:32 9/15, rest as GPUs free) — on **10% of the data** (Heesup's proposal: check convergence on 10k plants first, ~1 min/epoch, same 20 eval plants force-included via `EVAL_SET_FILE`, 20 held-out plants reported as `[Holdout]`): `38274493` baseline-on-10% → `38274494` `LATENT_NORM=1` → `38274495` scheduled TF → `38274496` `RENDER_TO_LATENT=1` → `38274497` combination → `38274498` v10 full (all six v10 pieces, design doc §2.7). Same-plant check vs the 9/8 Option B: its nodes are scatter (RMSE 18 cm, hull 4.4× GT), today's runs under-spread (hull 0.5–0.7×, 25% of GT phytomers within 3 cm) — node spread is the lever after the latent. 10:55 9/15: the local full-data geometry (ep80) and gt_nodes (ep78) runs were stopped on Heesup's request; this node's GPU now runs the 10% combination and v10-full runs (`local_sub10_combo.log`, `local_sub10_v10.log`). Caveat: `gpu-10-50` (inside desktop job `38252204`, ~24 h left) has been draining since 17:20 — resubmit from the checkpoints if it reboots (commands in takeover guide §0-B.9). **Finding:** Stage 3's latent carries no per-node image information (sampled latent worse than the dataset-mean latent, R² −0.34; t=0 estimate R² ≈ 0.1) while the GT latent is worth +33-38 IoU with GT geometry. Run 4 `RENDER_TO_LATENT=1` = `low` job `38260124` (2×A100, from epoch 45, `hierarchical_fm_v9_r2l/`); then a unit-variance latent scale for the flow.  **9/15 13:20, best deployable so far: test-time refinement** (`eval_test_time_refinement.py`, 40 Adam steps on node positions/scales/latents against the input CHM, no GT): v10 ep95 strict P 34.9 → 46.9 origin-frame; 33.5 → **62.9** in the input's camera frame (`--input_camera`, DAP > 15 38.4 → 74.5) but with inflated flat leaves on some plants; a scale/latent prior (`--reg_scale 5 --reg_latent 0.5`) removes the inflation and keeps the gain (six figure plants 39.6 → 70.4; all 20 plants strict P 35.8 → 63.9 with 1×/2× targets, **35.8 → 67.6** with all four zoom levels as targets, DAP > 15 41.1 → 74.0; these are now the script defaults); figures `docs/../agent-handover-guide/assets20260915_test_time_refinement_before_after*.png` (results report §11.7–11.10). The same camera frame now exists for the training loss (`RENDER_INPUT_CAMERA=1`; 10% run `sub10_v10_cam` local, ep80/85 strict P 35.8/34.0). **Scale-up (14:15): full-data v10 + input camera, job `38275054` on ada-h (2 GPUs), checkpoints `hierarchical_fm_v10_cam/`; the plateaued 10% cluster jobs were cancelled for it. Raw checkpoints swing (ep80 27.5, ep85 14.0 strict P; refined 64.6 / 53.3), so EMA weights were added (`EMA_DECAY=0.999`, `<ckpt>_ema.pt`) and job `38279147` continues the lineage with EMA from ep90. Refinement from a DAP-spanning mean latent (`--init_mean_latent`) reaches 66.5 for both the 10% v10 and the full-data ep90 checkpoint: the sampled latent is dispensable, the network's contribution is the nodes. Full-data ep95–110: strict P climbs slowly (raw 35.9 → 38.3, EMA 35.1 → 39.0 at ep110, best of any run) while the refined level stays 65–68; the lineage completed ep128 and continues to ep160 as job `38332343` (2026-09-16; ep128 raw 38.1 / EMA 37.7, refined 67.0).** |
| **P3** | **Export residuals** | terminal leaflets (one free angle, 1-1.6° mean), first-node petiole azimuth (only the shoot base roll can set it; a naive roll step was reverted), 2.2% of laterals still resolve to the wrong branch point |
| **P4** | Housekeeping | delete diagnostic checkpoint dirs (`local_probe_burst`, `local_act_probe`, `local_replay_*`, `local_smoke_parent`) if space matters |
