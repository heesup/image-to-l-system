# Agent Handoff Notes — FM Curvature-26D & Pyramid Conditioning (2026-09-05)

**Purpose**: Session log + next-step roadmap for the agent continuing this work.
**Read first**: [`docs/ongoing/AGENT_TAKEOVER_GUIDE.md`](AGENT_TAKEOVER_GUIDE.md) (single source of truth for
data contracts, 5 golden rules, gotchas §7.1–7.17, benchmark tables). This document adds the FM-curvature
session details on top of it.

**Last commit**: `bb6c5c8` (pushed to origin/main). All work below is committed except where noted.

---

## 1. What was accomplished this session

### 1.1 Anti-erasure rewrite (Phase 1 Method 2 fix)
The original diff-render optimizer silently erased thin tubes (internode + both petioles): 2/5 organ
retention, IoU 81.6%. Root causes found by debugging:
1. **Existence was only one erasure channel** — the optimizer also erased via `scale_y` (radius →
   sub-pixel = invisible = zero gradient = unrecoverable) and position drift (zero-grad + Adam
   normalized steps = random walk).
2. **Nadir-view CHM blindness**: a vertical tube projects a ~12px sliver whose visible pixels all
   interpolate CHM=0 → `clamp(depth*100) = 0` → the stem contributes ZERO image signal in top-view.
   (Renderer semantic, not a bug — compensated by a 3D tip anchor.)
3. **Zero-overlap tubes need long-range attraction**: misaligned tubes have no pixel overlap with
   their target, so IoU/depth/RGB only reward shrinking.

Fixes live in `scratch/phase2_core.py` (shared by exp2/4/5/7):
- `ExistenceWarden` — warmup freeze (existence + scale frozen for first 15–25 steps), existence floor
  hinge for seeded slots.
- `apply_scale_floor` — HARD radius floor (straight-through clamp, 0.7×init): radius collapse is a
  one-way trap. Length uses the SOFT `scale_floor_loss` hinge instead (hard-clamping length zeroes the
  gradient at the boundary and pins organs at 70% length — observed).
- `tube_pull_loss` — dilated-corridor coverage deficit around target tubes (annealed 40px→8px).
- `organ_tip_positions` + quadratic tip/base anchor (per-organ SUM, not mean — mean dilutes
  single-organ errors into stalling). Auto-detects 14D vs 26D layout.

**Result: exp2 IoU 81.6% → 92%, Chamfer 2.29 → 1.04 mm, 5/5 organ retention.**

### 1.2 Phase 2 — Variable organ topology
- **Strategy B (exp4, PASS)**: 10 over-allocated slots (5 real + 5 coincident ghosts) → exactly 5.
  Key mechanisms: ghost `ORGAN_NONE` logit init +2.0 ("dead until proven useful"), soft-NMS mutual
  exclusion with seeded-slot gradient protection (detach the real slot's factor in ghost–real pairs,
  else exclusion erases real organs), late ghost NONE-logit annealing (+0.05/step after step 60 —
  softmax saturation floors `p(1-p)` at ~1e-3 under Adam). IoU 94.29%, Chamfer 1.43 mm.
- **Strategy A (exp5, FEASIBLE)**: 3 slots (no leaves) + residual-driven leaf spawning → 82.7% IoU
  (+48 pp over the 3-slot cap). 1 bad spawn self-pruned. Not SOTA — probe only.

### 1.3 Representation coverage audit (exp6 + exp7)
- **exp6**: per-dimension analytic gradient + finite-difference image sensitivity. Found **2 DEAD
  dimensions**: `curvature` (14D col 13 — read but never used by the mesh builder) and `scale_z`
  (col 12 — ignored for tubes AND leaves). Both fixed in `helios_pytorch_geometry.py`:
  tubes bend (7-ring sagittal, Helios gravitropic convention, bitwise-identical at curvature=0,
  ±720°/m clamp); leaves use `scale_z` as blade-width aspect factor (`s_z/s_x`, clamped [0.2, 5]).
  Post-fix: 26/26 OPTIMIZABLE. **Lesson: re-run exp6 after any geometry edit.**
- **exp7**: group-wise perturb→recover. base_xyz RECOVERED (4.8→~1.1 mm); curvature PARTIAL (weak
  gradient ~3e-4, needs 200+ steps + per-group lr 2.0); rot6d LIMITED (gauge freedoms); scale LIMITED
  (sub-pixel radius observability); organ-type argmax mobility FAILED (Phase 3 scope).
- **Curvature dead-dim handling per pipeline**: direct optimization = fully fixed (geometry wired);
  ICP = doesn't estimate curvature by design (hardcoded 0, fine for canonical seedlings); FM = layout
  didn't even contain curvature → fixed this session (see §2).

---

## 2. FM Curvature-26D training (COMPLETED this session)

### 2.1 Data/representation changes
- `FM_NODE_DIM` 25 → **26** (`part_array_dataset.py`): added `FM_CURV = 25`,
  `CURV_SCALE = 1.0/60.0`. `encode_fm` writes `part[:, 13] * CURV_SCALE` for active organs;
  `decode_fm` now returns **14D** (was 13D): `[type, base(3), rot6d(6), scale(3), curvature]`.
  Roundtrip verified < 2e-6.
- Shards/cache regenerated. IMPORTANT layout note: old 27D shards exist under
  `dataset/helios_data/cowpea_shard_25d_legacy/` (actually 27D from a historical layout) —
  `cowpea_shard_dataset.py` auto-pads 25D→26D; treat 27D legacy as stale.
- `scripts/cache_dataset_tensors.py` **merged into `generate_tensor_shards.py`** and deleted.
  New `--mode cache|shard` and `--pyramid none|concat` flags.

### 2.2 Trainer bug fixes (the training was broken before)
- **loss_cat NaN since the beginning**: `loss_cat_active` sliced `[:, :, :EMPTY_IDX]` where
  `EMPTY_IDX = ORGAN_NONE = 0` → EMPTY slice → NaN. Correct: `[:, :, :FM_OT_END]` (13 cols).
- **bf16 overflow**: scale-block targets reach 100.0 (2 m × SCALE_SCALE=50) → square overflow in
  bf16 → inf/NaN. Loss is now computed in **fp32 outside the autocast block**.
- NaN-batch skip guard added (`nan_skips` counter printed per epoch).
- **DDP added** to `train_part_flow_matching.py` (torchrun, `setup_ddp()` no-ops without
  LOCAL_RANK, `DistributedSampler` + `set_epoch`, DDP wrap with `find_unused_parameters=True`,
  rank-0-only logging/checkpointing/vis, `raw_model` unwrapped for saves). Custom `fm_collate`
  crops per-sample nodes/existence to `max_nodes` (cache stores 4096-wide) and returns the
  key names train_epoch expects.
- W&B init restricted to rank 0 (previously each rank opened its own run → 2 duplicate runs
  on the dashboard with 2 GPUs).

### 2.3 Dedicated `loss_curv` — REQUIRED, not redundant
Variance-share analysis on fresh 26D shards (13,874 active organs): a unified 26-col MSE gives
curvature **0.00%** of the gradient signal (shard-normalized scale std ≈ 70.9 vs curvature ≈ 0.61;
scale alone = 97.28% of variance). Each block term is a per-slot-per-col mean, and curvature has no
coverage elsewhere (`loss_inactive_geom` only regularizes INACTIVE slots). Weight 1.0 → curvature
gets 1/7 of the loss budget. Observed curvature velocity loss 5.4 → 0.42 in smoke tests.

### 2.4 How curvature is predicted
The model predicts per-slot velocity `v(x_t, t, image)`. The curvature CHANNEL of that velocity
transports the normalized curvature component from the scaffold prior (≈0) to the data value along
the ODE. At t=1: `curvature_deg_m = fm[:, FM_CURV] / CURV_SCALE`. Caveats:
- Degenerate bend axis at curvature=0 for vertical tubes (`cross(fwd, z) = 0`) — probes require
  |curv| > 0.
- Curvature velocity loss is 2–5 orders weaker than position; needs 150–300 steps to recover, and
  a HIGH curv_lr destabilizes other groups (Adam per-parameter normalization). Use per-group LRs
  if doing group recovery experiments.

### 2.5 Training results
- **50 epochs COMPLETED** on 2×RTX 6000 Ada (DDP, global batch 256): loss **20,606 → 68**,
  zero NaN skips. Checkpoint: `diffusion_based/checkpoints/fm_curv/part_flow_matching.pt`
  (max_nodes=512 — load with `PartFlowMatchingModel(max_nodes=512, node_dim=26, image_size=128)`).
- **Slot-aligned curvature eval**: MAE 25.5°/m, median 8.8°/m (n=2,326 tube organs, 40 DAP-diverse
  samples, 15-step Euler). Caveats: (a) FM slot order ≠ canonical GT order — apply
  `canonical_sort_nodes` before comparing (unsorted comparison inflates MAE to ~47); (b) MAE is
  inflated by DAP-100 plants (curvature up to ±140°/m) — report median too.
- W&B run: project `part-flow-matching`, group `fm-curv26`.

### 2.6 Pyramid-concat conditioning (16-ch image)
- `generate_tensor_shards.py --pyramid concat` stores **16 channels**:
  `[zoom1(4ch) | zoom2 | zoom4 | zoom8]` where each zoom = RGB(3, [-1,1]) + CHM depth(1, meters).
  All zooms share the same canvas (central 1/k window crop upsampled back) → pixel-aligned, lossless.
- **Motivation**: DAP-1 seedlings fill <2% of the fixed 5m-drone frame (looks like bare ground at 1x).
  Zoom 8 gives the encoder stem/leaf detail.
- `ViTImageEncoder.forward`: for 16-ch input, computes per-zoom patch embeddings with the same
  4-ch `patch_embed` conv and **averages** them → output token shape unchanged (B, Np+1, D).
  `PartFlowMatchingModel` now constructs the encoder with `in_channels=4` (RGB+depth per zoom).
- `render_mesh` alias in `helios_pytorch_renderer.py` now forwards `zoom_factor` /
  `reference_window_size` (was dropping them → the SLURM cache run produced 0 files).

### 2.7 Visualization (`fm_visualization.py`)
Per-epoch panel saved to `docs/results/assets/fm_curv_epoch_NNN.png` +
`fig_fm_curv_latest_eval.png`:
- Row 1: target image — now shows **zoom-8 channels (12:15)** because DAP-1 at 1x looks like bare ground.
- Row 2: GT render | FM-generated composite (focus_plant=True @ 0.4 m) + per-tube curvature
  GT-vs-prediction bar chart.
- W&B: scalars (`train/loss`), panel image, curvature GT/pred histograms. Rank 0 only.

### 2.8 Repo/config changes
- `botanical_scaffold.py` restored to `diffusion_based/models/` (active dependency of the trainer —
  was archived by mistake during cleanup).
- Species filter in `PartArrayDataset`: when `cache_dir` is crop-named (`.../<crop>_curv26`),
  only that crop's XMLs are globbed (bean/cowpea never mix).
- `slurm_scripts/train_part_fm_curv.sh`: torchrun DDP launcher, cpus=8, mem=64G,
  global batch 256 (per-rank = 256/NPROC), `--cache_dir dataset/cache/cowpea_curv26`,
  vis + wandb flags on.
- `slurm_scripts/generate_helios_dataset_jobs.sh`: `--mode` and `--pyramid` passthrough; crop-named
  output dirs (`dataset/cache/<crop>_curv26` for cache mode).
- Old mixed cache deleted. Current cache: `dataset/cache/cowpea_curv26/` (10,000 cowpea samples,
  16-ch pyramid, verified all-cowpea).

---

## 3. Verification commands (all confirmed working)

```bash
mamba activate digital-crops && export PYTHONPATH=.

# Roundtrip 26D encode/decode with curvature
python - <<'EOF'
import torch
from diffusion_based.dataset.part_array_dataset import encode_fm, decode_fm
p = torch.zeros((2, 14)); p[:, 0] = torch.tensor([3.0, 4.0]); p[0, 13] = 45.0; p[1, 13] = -30.0
fm = encode_fm(p); rec = decode_fm(fm)
assert (rec - p).abs().max() < 1e-5 and fm.shape[-1] == 26
print('roundtrip OK')
EOF

# Dimension coverage audit (26/26 must be OPTIMIZABLE)
python scratch/exp6_dimension_coverage.py

# Full anti-erasure benchmark
python scratch/exp2_diff_render_opt.py        # expect ~90% IoU, 5/5 retention

# Train (multi-GPU DDP)
sbatch --export=FM_EPOCHS=50 slurm_scripts/train_part_fm_curv.sh

# Curvature quality eval (slot-aligned)
# see §2.5; the eval snippet is in this doc's git history and W&B panels
```

---

## 4. Known issues & gotchas discovered this session

1. **FM slot order ≠ GT order**: FM generates slots in arbitrary order; GT is canonically sorted.
   Always apply `canonical_sort_nodes` to both before any slot-wise comparison. Unsorted curvature
   MAE reads ~47°/m vs the true 25.5°/m.
2. **DAP-1 fixed-camera invisibility**: at camera_height=5.0 (drone standard), a 2.3 cm seedling
   fills <2% of the frame. Keep the data as-is (matches real drone orthophotos) but use zoom
   channels for visualization/conditioning.
3. **In-place masking cuts the autograd graph**: `p14[:, :13][:, mask] = -100` on the CONVERTED
   14D tensor kills gradient flow. Role-lock logits on the 26D node BEFORE
   `diff_node_to_part_tensor_14d` when guidance gradients are needed.
4. **organ_tip_positions layout**: auto-detects 14D (base 1:4, rot 4:10, len 10) vs 26D
   (base 13:16, rot 16:22, len 22). Never index a 26D tensor with 14D constants (silently wrong).
5. **compute node /tmp is node-local**: scripts written to /tmp/opencode on the login node are
   invisible to SLURM jobs. Put job scripts inside the repo.
6. **`HeliosPyTorchRenderer(image_size=..., device=...)`**: device kwarg accepted but rendering
   follows the mesh's device; `.to(device)` semantics are partial — meshes must already be on CUDA.
7. **SLURM queue**: `low` partition was 300+ jobs deep this session; `gpu-6000_ada-h` partition
   scheduled immediately. Prefer it (user's `gpu-free` + `squeue_me` aliases exist).
8. **gt_yaw30 raytrace caveat**: Helios C++ leaf roll is applied about WORLD axes
   (PlantArchitecture.cpp:2136) → XML pipeline is not yaw-rotation-equivariant. PyTorch-render IoU
   is the trustworthy metric for yawed reconstructions.

---

## 5. Recommended next steps (priority order)

1. **Slot-order canonicalization at sampling**: sort FM output with `canonical_sort_nodes` inside
   the sampling loop's final step (or right after), so downstream XML assembly and eval always see
   canonical order. Trivial change, removes the 13%-type-agreement artifact.
2. **Type mobility** (the real variable-topology frontier): FM argmax crossing was proven FAILED
   (exp7). Candidate approaches: (a) Gumbel-softmax straight-through on the one-hot block during
   sampling; (b) role-prior + residual type correction head; (c) two-stage: FM proposes geometry,
   a small classifier assigns types per slot conditioned on geometry.
3. **Scale-block normalization**: shard normalization (scale×50) makes scale std ≈ 70.9 —
   rebalancing to unit variance per block would let a single unified MSE work and remove the need
   for per-block loss weights. Requires shard regeneration.
4. **Latent-space curvature eval**: current MAE 25.5°/m is on the raw channel. Compare against
   exp2's direct-optimization curvature fits as an upper bound.
5. **Bean/sorghum caches**: the pipeline is crop-parameterized; generate `bean_curv26`,
   `sorghum_curv26` when multi-species training resumes.
6. **Update `eval_phase1_comparison.py`**: Method 3 row still reads the old exp3 output; point it at
   exp3b and add the trained-FM checkpoint row.

---

## 6. Key files touched this session (all committed as bb6c5c8 + follow-ups)

| File | Change |
|---|---|
| `diffusion_based/dataset/part_array_dataset.py` | FM 26D layout, species filter, cache fast-path (pyramid resize + node crop) |
| `diffusion_based/dataset/generate_tensor_shards.py` | merged cache/shard modes + pyramid concat + crop-named dirs |
| `diffusion_based/dataset/cowpea_shard_dataset.py` | 25D→26D auto-pad for legacy shards |
| `diffusion_based/models/helios_pytorch_geometry.py` | curvature wired into tubes, scale_z into leaf width, ±720°/m clamp, physicality clamp in diff_node_to_part_tensor_14d |
| `diffusion_based/models/helios_pytorch_renderer.py` | render_mesh alias forwards zoom kwargs |
| `diffusion_based/models/vit_image_encoder.py` | 16-ch pyramid per-zoom embedding averaging |
| `diffusion_based/models/part_flow_matching.py` | in_channels=4 encoder |
| `diffusion_based/models/botanical_scaffold.py` | restored from archive (active dependency) |
| `diffusion_based/training/train_part_flow_matching.py` | NaN fixes, fp32 loss, loss_curv, DDP, fm_collate, vis/wandb wiring |
| `diffusion_based/training/fm_visualization.py` | NEW: per-epoch panel + W&B |
| `slurm_scripts/train_part_fm_curv.sh` | torchrun DDP launcher (2×6000_ada, batch 256, cache_dir) |
| `slurm_scripts/generate_helios_dataset_jobs.sh` | --mode/--pyramid passthrough, crop-named dirs |
| `scratch/phase2_core.py` | anti-erasure core (warden, pull loss, scale floors, tip anchor) |
| `scratch/exp2/4/5/6/7*.py` | benchmarks (see §1) |
| `docs/results/assets/` | exp4/5/6/7 + fm_curv panels |
| `scripts/cache_dataset_tensors.py` | DELETED (merged) |

---

*End of handoff notes. The guide (§7 gotchas + §8 roadmap) plus this document contain everything
needed to continue.*