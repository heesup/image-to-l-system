# Agent Takeover & Engineering Handover Guide
**Project: Image-to-L-System / 3D Inverse Procedural Plant Reconstruction**  
**Last Updated:** 2026-09-14 ~11:30 PDT (see §0-A below — it is the live handover state for the **Claude Code** session that takes over next; the sections after it are the 2026-09-11 state and are superseded where §0 says so)  
**Primary Author/Agent:** Antigravity Autonomous Agent → **Claude Code** (since 2026-09-14; pair programming with Heesup Yun)  
**Environment:** Linux, Python 3.10+, Mamba (`mamba activate digital-crops`), CUDA, PyTorch, `nvdiffrast`, Helios C++ OptiX Raytracer.

> **Takeover for Claude Code (2026-09-14):** this guide is the handover doc. Claude Code session
> memory additionally lives outside the repo at `~/.claude/projects/-home-lion397-codes-image-to-l-system/memory/*.md`
> (launcher layout, terminology, run-log conventions, HPC partition rules, VAE round-trip rules).
> Read §0-B first (live handover state), then `docs/ongoing/README.md` (status dashboard) and
> `docs/ongoing/20260912_stage2_stage3_boundary_and_remaining_redundancy.md` (§5 reading order).  

---


## 0-A. State as of 2026-09-13 (read this first; supersedes the older sections where they disagree)

The full record is `docs/ongoing/20260912_stage2_stage3_boundary_and_remaining_redundancy.md`
(§0 status, §1.9.2 the training blocker, §2.4-2.5 the round-trip, §2.6 the dataset-plant round-trip, §5 reading order, §6 commit log); the 2026-09-14 report is `docs/results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md`.

**Round-trip (the "very important" requirement): solved.** `docs/results/assets/fig14_phytomer_vae_helios_roundtrip.png`
reads IK-only 99.9 / 99.6 / 97.7% and VAE round-trip **93.7 / 98.3 / 95.8%** FG IoU (DAP 10 / 50 / 90; regenerated
2026-09-14 with the leaf inverse, previously 95.7 / 99.5 / 96.7 and 92.2 / 98.4 / 95.7); on 2026-09-12 morning it was 81.4 / 90.2 / 79.3. Three fixes, all landed and on by default:
- leaflet emit order (the XML converter assigns leaf yaw by encounter order and reads only the scale; the terminal
  leaflet is identified per node, `phytomer_packets.terminal_leaflet_is_slot2`), and leaf size is one scalar per node
  (1 : 1 : 10/9) re-imposed in `assemble_packets`;
- **stem inverse kinematics** in the export (`diffusion_based/models/part_tensor_stem_ik.py`, run by
  `assemble_part_tensor_to_xml`; `PART_TENSOR_STEM_IK=0` gives the old analytical export). Helios rebuilds a shoot by FK
  from per-node angles; the solver makes that FK land on the predicted nodes. This, not VAE fidelity, was the ceiling
  (the identity export went 94.2 -> 99.6% at DAP 50);
- a better VAE, `diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/` (128D = 48 + 10x8, `--rot-weight 4`, 20,000
  files, terminal-last packets). The eval scripts default to it and set `PHYTOMER_TERMINAL_LAST=1` for any `_tl`
  checkpoint. The *legacy* `_unreferenced/phytomer_9slot_roundtrip_comparison.png` is the old soft-rasterizer pipeline;
  ignore it.

**Dataset plants (2026-09-14, design doc §2.6).** fig14 above is measured on the exact_gt trio, which is generated with
the converter's own default angles; on dataset plants (DAP 15/40/75 in the regenerated
`docs/results/assets/fig12_phytomer_10slot_helios_roundtrip.png`, script `diffusion_based/eval/eval_phytomer_10slot_assembly_views.py`)
the export alone read 81.8 / 92.2 / 54.6% and the VAE added nothing. Fixed, all on by default: `chain_phytomers` no longer
requires a parent below its child (drooping laterals were cut; cycles are now broken at their most expensive edge, and
`root_own_shoot=True` keeps the cotyledon node as shoot 0); `gt_parent_links(..., internode_base=)` resolves a lateral's
branch point from the decoded slot-0 base (66 -> 98% of laterals right -- the training call site passes it, so every
earlier run trained on wrong parent/depth targets for a third of the laterals); and `part_tensor_leaf_ik.py` inverts the
FK's leaf rotation per leaf (`PART_TENSOR_LEAF_IK=0` disables). Packet path on those plants: **98.3 / 98.2 / 95.6%**, VAE round-trip **95.1 / 96.8 / 95.6%**.

**VAE / cache lineage.** Every FM checkpoint so far read its Stage-3 target latents from a packet cache stamped with a VAE's latents. Since 2026-09-14 that coupling is gone: `train_hierarchical_flow_matching.py` encodes the target latent on the fly from the cached packets with the VAE the run loads, so `dataset/cache/cowpea_curv26_pkt_v9/` (pkt_version 7, terminal-last) and `cowpea_curv26_pkt/` (pkt_version 6, bottom-to-top) hold only packets/presence/centers/refs/keys and are VAE-independent. What still binds VAE ↔ cache is the packet ORDER (terminal-last vs bottom-to-top), via `PHYTOMER_TERMINAL_LAST`. So: bump `PKT_VERSION` and regenerate ONLY when the packet format changes; to swap the VAE, point `PHYTOMER_VAE_CHECKPOINT` at it (or `TRAIN_VAE=1` in the launcher) with `PKT_VERSION` matching its packets. The old standalone launchers `slurm_scripts/train_phytomer_vae.sh` and `generate_phytomer_packets_jobs.sh` are folded into `train_hierarchical_flow_matching.sh` (`TRAIN_VAE=1`) and `generate_helios_dataset_jobs.sh` (`--packets-only`), and now live under `archive/slurm_scripts/`.

**Training: the Stage 2 gradient burst is FIXED (2026-09-14).** Root cause: the coarse `nn.TransformerDecoder` (`norm_first=True`) had no final norm, so its raw residual stream (magnitude ~1e3 late in training) went into the bf16 phytomer self-attention as query/key/value; on a frozen burst state the backward amplified the gradient ~6000x on every batch. A final LayerNorm (`FM_DECODER_FINAL_NORM`, default on) gives 0/8 burst steps vs 8/8; fp32 self-attention (`FM_SELFATTN_FP32`, default on) is a partial mitigation kept as well. The v9 run restarted from scratch with both (`slurm_scripts/logs/local_v9_run2.log`, epochs 1-15, 0 canary hits; continued on 2026-09-14 10:30 from epoch 15 as cluster job `38252603` (geminigrp, 2 GPUs, log `hierarchical_fm_38252603.log`, checkpoints `hierarchical_fm_v9/`) with the §2.6 topology targets; the launcher's defaults are now the v9 recipe, and the queued `low`-partition jobs 38249632 / 38250275 pick up the current code when they start). The history below is kept for the record. It recurred at epoch 27-28 in two runs that
differ in seed and learning rate (`38240479` at 1e-4, `38242849` at 5e-5, both resumed from the same lineage) at the
same steps, and `DistributedSampler` is seeded by epoch only, so those runs saw the same batches. Refuted: weight decay,
decoder token collapse (`FM_ACT_PROBE=1`), a 512-sample subset replay. Live: a full-dataset replay of epochs 26-28 from
`hierarchical_fm_depth_ord/hierarchical_fm_epoch_025.pt` with `FM_SPIKE_DUMP=1` (names the samples on Stage 2 loss
spikes and on the first canary) -- running locally (`slurm_scripts/logs/local_replay_full.log`, slow) and queued on the
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
    Standalone `slurm_scripts/train_phytomer_vae.sh` moved to `archive/slurm_scripts/` (indexed in `archive/README.md`).
  - `generate_helios_dataset_jobs.sh`: the XML-direct packet backfill is now the **`--packets-only`** phase
    (`--pkt-version`, `--terminal-last`, `--pkt-out-dir`, `--num-jobs`), no VAE. Standalone
    `slurm_scripts/generate_phytomer_packets_jobs.sh` moved to `archive/slurm_scripts/`.
  - `tests/test_hierarchical_3stage_cascaded.py` fixed to the current model contract (pure-latent flow width,
    10 slots, `phytomer_parent_rel` input, valid 1+16×16 patch grid) — 2 stale failures closed. Full relevant suite:
    **57 passed**.
- **`f735ce9` — docs: repoint stale scratch/ paths to archive/scratch/ and add 2026-09-14 result figures** (fig13,
  fig14, `hierarchical_self_consistency_epoch_005.png`, metrics JSONs; `tests/unit/` path fixes).

### 0-B.2 Training state — GPU-efficiency experiment in flight, needs a decision

- Job `38252603` (batch 48/GPU, the validated v9 recipe; ~6.7 GB/49 GB VRAM, ~20% GPU util, **32 min/epoch**) was
  cancelled at ~11:15 and replaced by **`38252937`** (`gpu-6000_ada-h`, gpu-10-54, 24 h, log
  `slurm_scripts/logs/hierarchical_fm_38252937.log`): `FORCE_BATCH_SIZE=auto NUM_WORKERS=8`, resume
  (`INIT_CHECKPOINT=.../hierarchical_fm_v9_local2/hierarchical_fm_epoch_015.pt RESUME=1`) → resumed at **epoch 16**,
  output `diffusion_based/checkpoints/hierarchical_fm_v9/`, same LR 1e-4 / seed 1234 / render-from-11 / save-every-5.
- The VRAM probe (`probe_optimal_batch_size`) tuned **256 per GPU** (global 512; est peak 29.4 GB). Its `max_batch`
  is hardcoded to 256 in `train_hierarchical_flow_matching.py`; without the cap it would have picked ~392.
- **Finding (measured, not guessed):** step time went **2.2 s → ~18 s** (prof: backward 1.3→11.2 s, render
  0.65→5.27 s — roughly linear in batch), while steps/epoch fell only 1044→200. Net: **~55–60 min/epoch, WORSE than
  the 32 min/epoch at batch 48**. Backward scaling is slightly superlinear (8.6× for 5.3× batch).
- **Uncommitted changes backing the experiment** (commit or revert them):
  - `diffusion_based/training/train_hierarchical_flow_matching.py`: new `--num_workers` arg (default 4);
    both DataLoaders now use `num_workers=args.num_workers, prefetch_factor=4, persistent_workers=True`.
  - `slurm_scripts/train_hierarchical_flow_matching.sh`: passes `--num_workers "${NUM_WORKERS:-8}"`.
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

`diffusion_based/eval/eval_gt_substitution_ablation.py` (run dir: `gt_substitution_epochNNN.json`): every predicted node
matched to a GT phytomer, one quantity of the matched nodes replaced by ground truth, re-rendered, silhouette IoU against
the GT render, 20 plants of the fixed eval set. P 24.8% → pos←GT 42.3 → ALL (pos+topo+rot+scale+latent) 82.1;
leave-one-out from ALL: −pos 30.6, −rot 49.1, −latent 47.5, −scale 61.6, −topo 82.1. Position first, then rotation
and latent (−33 each when the rest is right), then scale; the ordinal head costs nothing given the rest; the predicted
node SET (missing / spurious, 0.5 gate) is the last 18 points; seedlings (DAP ≤ 15) are at 2.5% as predicted. This is
the evidence for §2.1's remaining piece -- child position/roll/scale generated in Stage 3 relative to the fixed parent
-- with position first. Details: `docs/results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md` §7.

### 0-B.8 Stage 3 geometry implemented, A/B in flight (2026-09-14 ~16:10, `abdbaf1`)

`--stage3_geometry` (launcher `STAGE3_GEOMETRY=1`): Stage 3's flow state becomes `[(pos − parent_pos)·BASE_SCALE | roll | scale | latent]`
(`split_flow_state` / `geometry_from_flow` in `hierarchical_part_flow_matching.py`; decoder built with base/rot/scale dims 3/2/3).
Targets are relative to the NOISED parent the model is conditioned on (jitter + substitution unchanged), velocity loss =
latent MSE + `--stage3_geom_weight` (4.0) × geometry MSE, `sample_ode` returns the refined pos/roll/scale under the usual
keys (Stage 2's under `phytomer_pos_stage2`), the render block renders the refined plant and its loss reaches Stage 3's
geometry block through pos and scale. A latent-only checkpoint widens on load (latent block kept in `geom_proj` / the
velocity head; Adam moments widened by `_widen_optimizer_state`). Stage 2 keeps its heads (matching, parents, ordinal,
existence). Off by default. **A/B** from `hierarchical_fm_v9/hierarchical_fm_epoch_045.pt`: baseline `38257989`
(cluster, 2 GPU, render 1/6) vs the geometry arm running locally on 1 GPU (`slurm_scripts/logs/local_s3geom_ab.log`,
`hierarchical_fm_v9_s3geom/`, eval every epoch). Read the per-epoch `[Self-Consistency]` IoU of both; the arm's velocity
loss starts high (fresh geometry dims, ~12-20) and should fall within the first epochs.

### 0-B.9 Three arms, and what the latent path actually knows (2026-09-14 ~17:30, `b348fd7`)

**Arms** (all from `hierarchical_fm_v9/hierarchical_fm_epoch_045.pt`, same recipe, self-consistency IoU on the fixed
20-plant set every epoch):

| arm | where | log / checkpoints | flag |
| :--- | :--- | :--- | :--- |
| baseline (latent-only Stage 3) | cluster `38257989`, 2 GPU | `slurm_scripts/logs/hierarchical_fm_38257989.log`, `hierarchical_fm_v9/` | — |
| geometry | local 1 GPU | `slurm_scripts/logs/local_s3geom_ab2.log`, `hierarchical_fm_v9_s3geom/`, panels `run_local_20260914_164059/` | `STAGE3_GEOMETRY=1` |
| gt_nodes (teacher forcing) | local 1 GPU, started 17:07 | `slurm_scripts/logs/local_gtnodes_ab.log`, `hierarchical_fm_v9_gtnodes/`, panels `run_local_20260914_170731/` | `STAGE3_GT_NODES=1` |
| render→latent | cluster `low` job `38260124`, 2×A100, queued 17:45 | `slurm_scripts/logs/hierarchical_fm_38260124.log`, `hierarchical_fm_v9_r2l/` | `RENDER_TO_LATENT=1` |

| epoch | baseline IoU % | geometry IoU % | gt_nodes IoU % |
| :---: | :---: | :---: | :---: |
| 46 | 31.5 | (crashed at eval, fixed `985ba26`) | |
| 47 | 31.0 | 30.6 | |
| 48–52 | 32.7 / 26.9 / 29.9 / 31.4 / 30.7 | | |

Read the table with the training-arm caveat: the gt_nodes arm is *trained* with GT nodes but the in-training eval
samples with Stage 2's nodes (train/test mismatch by design); its meaningful readout is the substitution ablation run
on its checkpoints, which teacher-forces the sampler automatically (`--teacher_force` is implied by the checkpoint's
`stage3_gt_nodes`), and the latent probe below. The geometry arm's Vel loss (3.1 → 2.4 at epochs 46–47) is still
falling; the arm is not readable before ~epoch 50.

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

**What follows.** The render loss is the only per-node image signal that does not pass through the flow loss, and the
latent rows were detached from it. `--render_to_latent` (`RENDER_TO_LATENT=1`, same commit series) keeps the latent
block attached in the render block; the smoke run's `FM_RENDER_GRAD_PROBE` shows the render loss on the velocity head's
latent rows at |g| 0.02–2.7 per step. **Arm 4 submitted 17:45: job `38260124`** (`low`/publicgrp, 2×A100, `--requeue`,
`AUTO_RESUME=1 RENDER_TO_LATENT=1`, from epoch 45 into `hierarchical_fm_v9_r2l/`, log
`slurm_scripts/logs/hierarchical_fm_38260124.log`). Read its per-node latent R² (ablation summary `_latent`) before its IoU. Two
further levers are cheap and principled if that is not enough: scale the latent to unit variance for the flow (an
SD-style scale factor; changes the velocity head, so fine-tune from epoch 45) and sample t toward 0 where the
conditioning matters. The gt_nodes arm answers the other half: if per-node R² rises when the nodes are right, the
latent path is starved by node error after all.

### 0-B.3 Working-tree hygiene

- Anything not in `git status` clean + the two files named in 0-B.2 is either untracked run artifacts
  (`docs/results/assets/`) or belongs to the archived-launcher index (`archive/README.md`). Keep `slurm_scripts/`
  to the two current launchers (`train_hierarchical_flow_matching.sh`, `generate_helios_dataset_jobs.sh`).
- Cluster etiquette: Heesup's `regen_*` jobs (geminigrp) hold the group GPU quota — do not cancel; training goes on
  `gpu-6000_ada-h` / `low`. The OnDemand desktop (38252204) must not be killed.

## 0. Quick State Check (Run This First)

```bash
# Where is training right now? (latest job log)
ls -t slurm_scripts/logs/hierarchical_fm_*.log | head -3
tail -n 30 slurm_scripts/logs/hierarchical_fm_$(ls -t slurm_scripts/logs/ | grep ^hierarchical_fm | head -1 | sed 's/hierarchical_fm_//;s/.log//').log

# All running/pending jobs
squeue -u lion397

# Dataset size
find dataset/helios_data/cowpea -name "*.xml" | wc -l   # 100,000
find dataset/cache/cowpea_curv26 -name "*.pt" | wc -l   # 100,000 (all have phytomer_ids)

# Available checkpoints
ls -lh diffusion_based/checkpoints/hierarchical_latent_fm/*.pt
ls -lh diffusion_based/checkpoints/phytomer_vae_v3/*.pt   # v3 is the DEFAULT now
```

---

## 1. Executive Summary & Mission

The core mission is **inverse 3D botanical reconstruction**:
Given a single monocular top-view RGB-D ($256 \times 256 \times 4$) image containing Canopy Height Model (CHM) depth, reconstruct the exact 3D plant architecture into:
1. **Canonical 14D Part Tensor Representation**: Disentangled per-organ metric state.
2. **Native Helios C++ XML Tree**: Full procedural L-system specification for physical raytracing.

### Architecture (Current): Hybrid Decoupled 3-Stage Cascaded Botanical Flow Matching

```
Input: RGB-D 4ch image (256×256)
       ↓
[Stage 0] DINOv2 ViT-S/14 → PETR 3D Ray PE → 3D-aware image tokens
       ↓
[Stage 1] MacroBiologicalHead (CLS token) → Phytomer count (Soft margin capacity prior)
       ↓
[Stage 2] CoarseSkeletalTransformer — DETERMINISTIC Set Transformer (Scaffold)
          → node pos (3D), rot (6D), scale (3D), existence logits (1D)
          → supervised by loss_phytomer_pos, loss_phytomer_rot, loss_phytomer_scale, loss_phytomer_exist
       ↓
[Stage 3] PhytomerFlowMatchingDecoder (Conditioned on Stage 2 Scaffold & Image)
          → Flow Vector: 64D Phytomer VAE Latent ONLY
          → Prior: Standard Gaussian z_0 ~ N(0, I_64) (No bridge coupling → VelLoss stable ~1.0)
          → Conditioning: phytomer_pos.detach(), phytomer_rot.detach(), phytomer_scale.detach(), phytomer_features
          → 10 organ slots existence logits (P(organ) = P(node) * P(slot|node))
       ↓
[Stage 4] PhytomerVAE v3 (frozen) → 64D Latent decoded to 10-organ canonical packet
          → Assembled via denormalize_packet_scales(scale) & apply_ref_for_flow(pos, rot)
          → HeliosPyTorchRenderer (Multi-scale pyramid CHM depth + Soft Dice silhouette)
```

> **Design Decision (2026-09-10)**:
> - 76D Bridge Flow ($z_0 = \text{node\_pos} + \epsilon$)의 오차 증폭(`VelLoss: 1050~2000`)을 해소하기 위해 **하이브리드 디커플링**으로 완전 전환.
> - Stage 2는 전신 3D 골격(`pos`, `rot`, `scale`, `exist`)을 전담하고, Stage 3은 표준 정규분포에서 파이토머 내부 64D 잠재공간 Flow Matching을 전담.
> - 손실 함수 10개 $\to$ 8개 정예화 (`loss_cos`, `loss_dap` 제거, `loss_phytomer_rot` 정규 지도 추가).

### Milestone History:
- ✅ **Phase 1** (ICP / Diff Render / Flow Matching benchmark): Complete.
- ✅ **Phase 2** (Over-allocation topology, variable organ counts): Complete.
- ✅ **Option B: 500-Epoch 16D Latent Hierarchical FM** (Job `38143585`): 55.1% mean IoU, peak 67.9%, 2.49cm height error, 0 ghost organs.
- ✅ **3-Stage Cascaded Architecture** (Current, Job `38146809`): Running past Epoch 54 with idle slot damping, scaled global batch size (192), and accelerated node convergence.

---

## 2. Active SLURM Jobs (as of 2026-09-11 ~13:15 PDT)

| Job ID | Name | Status | Node | Notes |
| :--- | :---: | :---: | :--- | :--- |
| **38230613** | `ondemand/sys/dashboa` | RUNNING | `gpu-5-58` | User's interactive OnDemand desktop — **DO NOT CANCEL** |
| **38237555** | `hierarchical_fm` | **RUNNING** | `gpu-10-50` | 4x RTX 6000 Ada, **현재 전체 스텝 grad_norm=inf 데드락** — 재시작 필요 |
| 38236803 | `hierarchical_fm` | CANCELLED | — | 이전 시도 |
| 38236800 | `hierarchical_fm` | CANCELLED | — | 이전 시도 |

### ⚠️ 현재 학습 상태 (Job 38237555 — CRITICAL)

**Job 38237555**은 Epoch 40 이후 **전 스텝 grad_norm=inf 데드락**에 진입:
```
[Recovery] Step 1~31+: grad_norm is NaN/Inf (inf) | Culprit params (0): -> SKIPPING STEP
Epoch 040 | Loss: 0.0000 | ... | Vel: 0.0000 ... | VRAM: 26.9/47.4 GB
```
- `Culprit params (0)` — 파라미터 이름 추적 실패 (grad가 None 또는 텐서 자체가 이미 inf)
- Loss가 0.0000으로 고정 → 모든 스텝이 skip되어 **완전 동결(frozen)** 상태
- **즉시 취소 후 root cause 수정하여 재시작 필요**

### 근본 원인 후보 (2026-09-11 기준 미해결)
| 후보 | 증상 | 조사 방법 |
| :--- | :--- | :--- |
| `init_logits` Float32 overflow | $\ell_k = (N_{phy}+m-k)/\tau \to 96.25 \to e^{96.25} = \infty$ | clamp to [-15, +15] |
| Stage 2→3 gradient leak (missing `.detach()`) | `tgt_velocity = z1 - z0` where `z0` contains `pred_phytomer_pos` without detach | verify all `z_0` construction |
| NaN propagation from Depth/Dice render loss | nvdiffrast nan on degenerate mesh at early epochs | `torch.nan_to_num` on render output |

### 권장 재시작 절차:
```bash
# 1. 현재 잡 취소
scancel 38237555

# 2. init_logits clamp 패치 확인
grep -n "init_logits\|soft_margin\|clamp" diffusion_based/models/hierarchical_part_flow_matching.py | head -20

# 3. z_0 detach 확인
grep -n "z_0\|detach" diffusion_based/training/train_hierarchical_flow_matching.py | head -30

# 4. 재시작
sbatch slurm_scripts/train_hierarchical_flow_matching.sh
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

**문제**: 기존 flat concatenation 방식이 Phytomer별 Internode-Petiole-Leaflets 순서를 뒤섞어 식물 기하학적 형상을 붕괴시킴.

**해결**: `phytomer_packets.py`에 `assemble_phytomer_ordered_14d_tensor` 함수 신규 구현:
- **슬롯 순서**: `[0]=Internode | [1]=Petiole | [2,3,4]=Leaflets | [5]=Peduncle | [6,7,8,9]=Repro`
- 줄기(Internode)를 0번 슬롯으로 복귀시켜 마디간(줄기)-엽자루-소엽의 1:1 결속 복원
- 5mm 이하 마디 버림 버그 수정 (임계값 0.1mm로 하향)
- 떡잎(Cotyledon) 대생 구조 복원 → 모든 DAP에서 파트 수 일치(Δ=0) 달성

### 3.8 Per-Organ Mask IoU Roundtrip Diagnosis (2026-09-11)

`diffusion_based/eval/eval_13d_xml_organ_masks.py` 실행 결과 (`fig10_helios_per_organ_mask_comparison.png`):

| DAP | Foreground IoU | Mean Organ IoU | Internode | Petiole | Leaf | Peduncle | Flower | Fruit | Depth PSNR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10 (Seedling)** | **95.7%** | **48.3%** | — | 1.0% | 95.6% | — | — | — | **38.90 dB** |
| **50 (Branching)** | **94.1%** | **36.0%** | 0.0% | 14.6% | 93.3% | — | — | — | **26.12 dB** |
| **90 (Fruiting)** | **87.4%** | **21.7%** | 12.9% | 4.9% | 80.6% | 7.0% | 19.2% | 5.7% | **21.43 dB** |

**Key Findings**:
- **Leaf** 클래스는 모든 단계에서 80~96% IoU — 잎 형상은 정확히 복원됨
- **Internode/Petiole** IoU가 0~15%로 매우 낮음 — 14D XML에서 줄기/엽병의 **좌표 오프셋** 또는 **반경/길이 스케일** 미스매치가 원인으로 추정
- DAP 50에서 Internode IoU = 0.0% → 줄기가 아예 다른 위치에 렌더링됨
- **다음 단계**: `extract_part_tensor`의 월드 포즈 ↔ Helios XML 역기구학(IK) 변환 일치 여부를 수치 비교

*(Result figure: [`docs/results/assets/fig10_helios_per_organ_mask_comparison.png`](../results/assets/fig10_helios_per_organ_mask_comparison.png))*

---

## 4. Current Checkpoints & Job Status

Checkpoints saved on disk:
```bash
# Hybrid decoupled 3-stage run (Sep 11) — 390 MB each, new architecture
diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt  (390 MB, Sep 11 11:57)
diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_050.pt  (390 MB, Sep 11 05:13)
diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_075.pt  (390 MB, Sep 11 07:19)
diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_100.pt  (390 MB, Sep 11 09:25)

# Old organ-mode 500-epoch run (Sep 7–9) — 552 MB, OLD architecture (incompatible)
diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_125.pt ~ epoch_500.pt  (552 MB, Sep 7–9)

# PhytomerVAE v3 (ACCEPTED DEFAULT)
diffusion_based/checkpoints/phytomer_vae_v3/phytomer_vae_64d_best.pt  (1.9 MB, Sep 10 14:54)
diffusion_based/checkpoints/phytomer_vae_v3/phytomer_vae_64d_last.pt  (1.9 MB, Sep 10 14:54)
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
#      WANDB_RUN_NAME=smoke-4k-v3-76d bash slurm_scripts/train_hierarchical_flow_matching.sh
#    (with --max_train_samples small + --detect_anomaly if debugging)

# 2. Cluster submission (defaults now carry v3 VAE + pkt v3 cache)
sbatch slurm_scripts/train_hierarchical_flow_matching.sh
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
| **P3** | Epoch-1 sanity after resubmit | Check `slurm_scripts/logs/run_<jobid>/hierarchical_self_consistency_epoch_001.png` (each run's panels sit beside its own `run.log` symlink; they used to overwrite each other under `docs/results/assets`): loss ↓, pred count ~50, ClsAcc rising, no Recovery-skip lines |
| **P4** | Monitor 6D rotation convergence | Panels epoch 25/50; s_a (petiole len) should track DAP growth |
| **P5** | Evaluate Bidirectional Chamfer Distance | Add max/mean distance GT→Pred to avoid one-way clustering metric bias |
| **P6** | Backbone A/B (DINOv2-scale vs frozen arms) | `slurm_scripts/submit_backbone_ablation.sh` — only after single-arm training is stable |

---

## 6. Key Source Code Map

```
/home/lion397/codes/image-to-l-system/
├── diffusion_based/
│   ├── models/
│   │   ├── hierarchical_part_flow_matching.py    ← [CRITICAL] 3-stage model, PhytomerFlowMatchingDecoder (64D latent flow, NOT bridge), dap_embed, color_palette
│   │   ├── dinov2_ray_encoder.py                 ← DINOv2 + PETR 3D ray PE; canonical_rays buffer (inplace gotcha §8.7)
│   │   ├── plant_organ_array.py                  ← 14D constants, organ types, XML roundtrip
│   │   ├── helios_pytorch_geometry.py            ← mesh builder + differentiable mapping
│   │   ├── helios_pytorch_renderer.py            ← nvdiffrast renderer + multi-scale pyramid
│   │   ├── part_tensor_to_40d.py                 ← [OPEN BUG] closed-form IK + XML assembler; Internode/Petiole IK convention mismatch
│   │   ├── organ_latent_vae.py                   ← frozen OrganLatentVAE bridge (16D/organ)
│   │   └── phytomer_vae.py                       ← PhytomerVAE (240D in / 64D latent; normalizes scales in pack_input+compute_loss)
│   ├── training/
│   │   ├── train_hierarchical_flow_matching.py   ← [CRITICAL] main loop: --flow-granularity phytomer, probe reuse, --render_grad_start_epoch, --detect_anomaly
│   │   ├── train_phytomer_vae.py                 ← PhytomerVAE trainer (canonical packets)
│   │   ├── hierarchical_hungarian_matcher.py     ← [CRITICAL] coarse node + local fine bipartite matching
│   │   └── flow_matching.py                      ← Rectified Flow scheduler
│   ├── dataset/
│   │   ├── part_array_dataset.py                 ← [CRITICAL] cache-first filtering, 16-ch pyramid, pkt v3 gate
│   │   ├── generate_cache.py                     ← unified cache/pkt generator (PKT_VERSION=3 stamp + skip gate)
│   │   └── phytomer_packets.py                   ← [UPDATED 2026-09-11] assemble_phytomer_ordered_14d_tensor; strict [Internode|Petiole|Leaflets|Repro] order
│   ├── eval/
│   │   ├── eval_hierarchical_self_consistency.py ← 7-column diagnostic panel generator (DAP-spread samples)
│   │   └── eval_13d_xml_organ_masks.py           ← [NEW 2026-09-11] Per-organ COCO mask IoU + Depth PSNR roundtrip eval
│   └── checkpoints/
│       ├── hierarchical_latent_fm/               ← epoch_025~100 (390MB, new 3-stage arch); epoch_125~500 (552MB, OLD arch — incompatible)
│       ├── organ_vae/organ_latent_vae_best.pt    ← frozen OrganLatentVAE bridge
│       └── phytomer_vae_v3/                      ← [ACCEPTED DEFAULT] PhytomerVAE-64 v3, 10-slot normalized (val recon 0.070, cls 100%)
├── slurm_scripts/
│   ├── train_hierarchical_flow_matching.sh       ← launcher; defaults: VAE v3, SLOTS_PER_PHYTOMER=10, BACKBONE_LR_RATIO=0.3
│   ├── generate_helios_dataset_jobs.sh           ← full pipeline: XML synth + cache (+ pkt/latent) in one pass
│   └── submit_backbone_ablation.sh               ← A/B dispatcher (DAP-spread arms, sequential chain)
├── dataset/
│   ├── helios_data/cowpea/                       ← 100,000 XML files (complete)
│   ├── cache/cowpea_curv26/                      ← 100,000 cached .pt (image+nodes+phytomer_ids, complete)
│   ├── cache/cowpea_curv26_pkt/                  ← 100,000 pkt v3 (10-slot, absolute packets + normalized latent)
│   └── cache/cowpea_curv26_subset4k/             ← 4,000 symlinks (40/DAP × 100 DAP) for fast smoke tests
└── docs/
    ├── ongoing/
    │   ├── README.md                             ← ongoing status dashboard
    │   └── AGENT_TAKEOVER_GUIDE.md               ← [THIS FILE] master handoff
    └── results/
        ├── 20260910_gradient_explosion_debug_and_architecture_comparison.md ← grad explosion root cause
        ├── 20260910_current_architecture.md      ← full math derivation of 4-stage pipeline
        └── assets/
            ├── fig10_helios_per_organ_mask_comparison.png  ← [NEW 2026-09-11] Per-organ IoU diagnosis (DAP 10/50/90)
            └── fig12_phytomer_10slot_helios_roundtrip.png  ← [2026-09-14] dataset plants DAP 15/40/75: GT + 10-slot assembly (nadir, 45°), Helios round-trips
```

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

## 8. Common Gotchas (Critical for Successors)

1. **`PYTHONPATH=.` is Mandatory**: Always run commands from workspace root with `PYTHONPATH=.`.
2. **Helios XML Non-Negative**: Any `<internode_radius>` or `<leaf_scale>` $\le 0$ causes a C++ core dump (`SIGABRT`). Always clamp to $\ge 1\text{e-}4$.
3. **CHM Depth = Height Above Ground**: Inverted relative to standard LiDAR/camera distance. Ground is 0, plant canopy apex is maximum height.
4. **Cache-First Filtering**: Only XMLs that have a corresponding `.pt` tensor in `dataset/cache/cowpea_curv26/` are loaded by the dataloader.
5. **Node RMSE vs. Tree Topology**: Do not rely solely on `Node RMSE`. Check Column 5 (Oblique 3D Mesh Composite) and Column 6 (Botanical Skeleton) to verify 3D rotation and branch divergence.
6. **OnDemand Desktop (Job 38230613)**: Running on `gpu-5-58` — do not touch or kill.
7. **Inplace-autograd traps**: Three distinct failures came from autograd graph issues, not model logic:
   - Packet-scale `as_strided` views (`[P,10,3]`) must not be modified inplace across fwd→bwd — fixed via no-inplace `normalize/denormalize_packet_scales` (`69ce959`).
   - Probe-token reuse (`image_tokens` kwarg) can surface DDP "mark a variable ready only once" if any parameter is used outside a single forward.
   - Encoder-alone repro (DINOv2RayEncoder fwd/bwd in isolation) is CLEAN — `canonical_rays` buffer version stays 0; do not chase that buffer without reproducing in the training loop first.
   - Debug with `--detect_anomaly` on 1 GPU, small `--max_train_samples`.
8. **pkt cache version gate**: `PartArrayDataset` and `generate_cache.py` skip pkt files whose `pkt_version != 3`. If you change packet format, bump `PKT_VERSION` and regenerate; never edit files in place.
9. **VAE v3 recon 0.070 vs v2 0.0263 is NOT a regression**: v3 targets are scale-normalized (wider distribution). Do not "fix" by switching back to v2.
10. **10-slot ordered assembly MUST match VAE training order**: `[0]=Internode | [1]=Petiole | [2,3,4]=Leaflets | [5]=Peduncle | [6,7,8,9]=Repro`. Breaking this order corrupts all VAE decoding.
11. **Internode/Petiole IoU ≈ 0% is a known open bug**: As of 2026-09-11, the 14D XML roundtrip correctly reproduces leaf geometry (>80% IoU) but stem/petiole position is wrong. Root cause: IK convention mismatch in `part_tensor_to_40d.py` vs Helios XML parser. See P1 in §5.

---

## 9. Milestone Documents (Chronological)

| File | Date | Summary |
| :--- | :--- | :--- |
| [`docs/ongoing/20260909_phytomer_latent_and_local_matching.md`](../ongoing/20260909_phytomer_latent_and_local_matching.md) | 2026-09-09/10 | **[MASTER ENGINEERING LOG]** §4.6–4.9: pipeline refactor, backbone A/B, fruit fix, v3 scale-normalized packets + 76D flow, render-pipeline timing |
| [`docs/results/20260907_latent_hierarchical_flow_matching_500epoch_report.md`](../results/20260907_latent_hierarchical_flow_matching_500epoch_report.md) | 2026-09-07 | Option B 500-epoch report: 55.1% IoU, 2.49cm height error |
| [`docs/results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md`](../results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md) | 2026-09-08 | Epoch 150 spatial vision breakthrough: 49.2% mean IoU, 2.6cm RMSE |
| [`docs/results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md`](../results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md) | 2026-09-08 | 3-stage cascaded architecture scaling: batch 192, dormant slot damping |
| [`docs/results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md`](../results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md) | 2026-09-08 | Root cause analysis of Epoch 50 skeleton geometry vs AncPos loss |
| [`docs/results/20260910_gradient_explosion_debug_and_architecture_comparison.md`](../results/20260910_gradient_explosion_debug_and_architecture_comparison.md) | 2026-09-10 | **[KEY]** Gradient explosion diagnosis: Float32 overflow + z0 detach bug + deadlock mechanism |
| `docs/results/assets/fig10_helios_per_organ_mask_comparison.png` | **2026-09-11** | **[NEW]** Per-organ COCO mask IoU + Depth PSNR roundtrip: Leaf ✅, Internode/Petiole ❌ |
| `docs/results/assets/fig12_phytomer_10slot_helios_roundtrip.png` | **2026-09-14** | Dataset plants DAP 15/40/75: GT mesh and 10-slot assembly under the same nadir and 45° cameras, Helios round-trip via packets and via the VAE (`eval_phytomer_10slot_assembly_views.py`) |
