# Agent Takeover & Engineering Handover Guide
**Project: Image-to-L-System / 3D Inverse Procedural Plant Reconstruction**
**Last Updated:** 2026-09-08 (3-Stage Cascaded Flow Matching + SLURM Scaling session)
**Primary Author/Agent:** Antigravity Autonomous Agent (Pair programming with Heesup Yun)
**Environment:** Linux, Python 3.10+, Mamba (`mamba activate digital-crops`), CUDA, PyTorch, `nvdiffrast`, Helios C++ OptiX Raytracer.

---

## 0. Quick State Check (Run This First)

```bash
# Where is training right now?
tail -n 5 slurm_scripts/logs/hierarchical_fm_38146809.log

# All running/pending jobs
squeue -u lion397

# Dataset size
find dataset/helios_data/cowpea -name "*.xml" | wc -l   # target ~120,000
find dataset/cache/cowpea_curv26 -name "*.pt" | wc -l   # target ~120,000

# Available checkpoints
ls -lh diffusion_based/checkpoints/hierarchical_latent_fm/*.pt
```

---

## 1. Executive Summary & Mission

The core mission is **inverse 3D botanical reconstruction**:
Given a single monocular top-view RGB-D ($256 \times 256 \times 4$) image containing Canopy Height Model (CHM) depth, reconstruct the exact 3D plant architecture into:
1. **Canonical 14D Part Tensor Representation**: Disentangled per-organ metric state.
2. **Native Helios C++ XML Tree**: Full procedural L-system specification for physical raytracing.

### Architecture (Current): 3-Stage Cascaded Hierarchical Matryoshka Flow Matching

```
Input: RGB-D 4ch image (256×256)
       ↓
[Stage 1] DINOv2 ViT-S/14 → PETR-DETR3D → 512 Anchor points (3D positions)
       ↓
[Stage 2] Coarse Flow Matching (4 transformer layers) → DAP-conditioned anchor slot assignment
       ↓
[Stage 3] Fine Flow Matching (6 transformer layers) → 8 organ slots per anchor (4,096 max slots)
       ↓
OrganLatentVAE (frozen) → 16D latent → 14D Part Tensor → Helios XML
```

### Milestone History:
- ✅ **Phase 1** (ICP / Diff Render / Flow Matching benchmark): Complete.
- ✅ **Phase 2** (Over-allocation topology, variable organ counts): Complete.
- ✅ **Option B: 500-Epoch 16D Latent Hierarchical FM** (Job `38143585`): 55.1% mean IoU, peak 67.9%, 2.49cm height error, 0 ghost organs.
- ✅ **3-Stage Cascaded Architecture** (Current, Job `38146809`): Fresh training from Epoch 1 with all improvements from 2026-09-08 session (see §9).

---

## 2. Active SLURM Jobs (as of 2026-09-08 ~16:56 PDT)

| Job ID | Name | Status | Node | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **38146809** | `hierarchical_fm` | **RUNNING** | `gpu-10-54` | Main training job, **Epoch 26/500**, 4× RTX 6000 Ada, 40.5 GB VRAM (82.5%), TIME_LEFT ~22h |
| **38146114** | `ondemand/desktop` | RUNNING | `gpu-5-58` | User's interactive OnDemand desktop — **DO NOT CANCEL** |
| **38147133–38147171** | `helios_pipe_*` | R/PD | multiple | 40-shard Helios dataset synthesis, 24-hour time limit, ~34 still running/pending |

### Helios Dataset Generation Progress
- **51,880 XMLs** generated so far out of target ~120,000
- **22,600 cache `.pt` tensors** in `dataset/cache/cowpea_curv26/`
- All jobs have `TIME_LIMIT=1-00:00:00` (24 hours) — should complete without timeout
- Jobs skip already-generated files automatically

---

## 3. Session Changes (2026-09-08) — What Was Done Today

### 3.1 Algorithm Improvements (Commits `0389ca5`, `5b9fbf0`, `54579bc`)

**Root Cause Analysis** — Why today's run initially underperformed yesterday's Epoch 75:
- **Root Cause 1** (Unmatched Slot Wandering): Reproductive slots (6, 7) during vegetative stages (DAP < 40) received zero velocity gradients and wandered into empty air (+20–30 cm). When sampled on mature plants (DAP 88), these slots produced ghost organs.
- **Root Cause 2** (Slow Anchor Convergence): Visual backbone LR was 10× smaller (`2e-5` vs 6e-5), causing slower 3D anchor convergence (`AncPosLoss`: 0.0068 vs yesterday's 0.0035).

**Fixes Applied:**
1. **Idle Slot Damping** (`train_hierarchical_flow_matching.py`): Added $L_2$ velocity regularizer $\mathcal{L}_{\text{idle}} = 0.05 \sum_{i \notin \mathcal{M}} \|\mathbf{v}_i\|^2$ for unmatched slots.
2. **Top-k Threshold Guard** (`hierarchical_part_flow_matching.py:sample()`): `valid_topk = top_k_indices[combined_prob[b, top_k_indices] > 0.15]` prevents ghost activation in unmatched slots.
3. **Backbone & Anchor Acceleration**: `loss_anchor_pos` weight 2.0→**4.0**, DINOv2 backbone LR `2e-5`→**6e-5** (`args.lr * 0.3`).

### 3.2 Batch Size Scaling (Commit `0389ca5`)
- Upgraded `train_hierarchical_flow_matching.sh` from `BATCH_ARG=24` to `BATCH_ARG=48` (global batch 192)
- VRAM scaled from 24.5 GB (51.7%) → **40.5 GB (82.5%)** on all 4 RTX 6000 Ada GPUs
- Training speed doubled: ~50 seconds/epoch (70 steps/epoch)

### 3.3 Dataset Cache Filtering (Commit `027118d`)
- Fixed `part_array_dataset.py` to filter XML paths by `cache_dir` `.pt` files upfront
- Prevents 3-channel vs 16-channel `RuntimeError` during concurrent Helios data generation
- Only samples with pre-computed 16-channel pyramid cache tensors are loaded

### 3.4 Helios 24-Hour Time Reservation (Commit `97420ee`)
- Changed default `TIME_LIMIT="24:00:00"` in `slurm_scripts/generate_helios_dataset_jobs.sh`
- Canceled 26 old 4-hour jobs, submitted all 40 shards (38147132–38147171) with 24h limits
- Mature crop DAP 73–100 stages require 7–8h of simulation — 4h was insufficient

### 3.5 Validation Decoupled to Every Epoch (Commit `c587767`)
- Added `--eval_every` argument (default: 1) to `train_hierarchical_flow_matching.py`
- Self-consistency evaluation (`ehsc.evaluate_self_consistency_batch`) now runs every epoch
- Checkpoint saving (`--save_every 25`) remains independent
- Diagnostic panels saved to `docs/results/assets/hierarchical_self_consistency_epoch_{epoch:03d}.png`

### 3.6 Checkpoint Resume Support (Commit `0435325`)
- Added `--resume` flag: restores epoch counter + optimizer state from checkpoint
- `CosineAnnealingLR.last_epoch` aligned to `start_epoch - 1` for correct LR schedule
- Shell script now reads `INIT_CHECKPOINT` env var with `RESUME=1` default

---

## 4. Immediately Actionable: When Epoch 25 Checkpoint Exists

> [!IMPORTANT]
> Job 38146809 runs `--eval_every 1` but **without `--resume`** (started from epoch 1).
> It does NOT have the `--eval_every` update (started before commit `c587767`).
> This means **no per-epoch panels** are being saved right now.

**Check if Epoch 25 checkpoint exists:**
```bash
ls -lh diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt
```

**When it exists, cancel old job and restart with all updates:**
```bash
# 1. Check training log to confirm epoch 25 saved
grep "Saved checkpoint" slurm_scripts/logs/hierarchical_fm_38146809.log

# 2. Cancel old job (it's running on the OLD code without --eval_every)
scancel 38146809

# 3. Submit new job with checkpoint resume
INIT_CHECKPOINT=diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt \
  sbatch slurm_scripts/train_hierarchical_flow_matching.sh

# 4. Watch new log for per-epoch self-consistency panels
tail -f slurm_scripts/logs/hierarchical_fm_<NEW_JOB_ID>.log
```

---

## 5. Next Steps (Priority Order)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P1** | Wait for Epoch 25 checkpoint → cancel job 38146809 → re-submit with `INIT_CHECKPOINT` | Gets `--eval_every 1` + `--resume` active |
| **P2** | Monitor Helios shard completion | `squeue -u lion397 \| grep helios`; target 120k XMLs + cache |
| **P3** | Once all Helios shards complete, expand dataset to ~120k | Restart training or run a new job with fuller dataset |
| **P4** | Monitor IoU improvement past Epoch 50 (when dormant slot damping starts showing effect) | Compare `hierarchical_self_consistency_epoch_050.png` vs `epoch_100.png` |
| **P5** | After Epoch 100: Consider adding stage-specific DAP masking for reproductive slots | Only matters if ghost organs appear in Epoch 50–100 panels |

---

## 6. Key Source Code Map (Current Active Files)

```
/home/lion397/codes/image-to-l-system/
├── diffusion_based/
│   ├── models/
│   │   ├── hierarchical_part_flow_matching.py    ← [CRITICAL] 3-stage model + sample() with threshold guard
│   │   ├── plant_organ_array.py                  ← 14D constants, organ types, XML roundtrip
│   │   ├── helios_pytorch_geometry.py            ← mesh builder + differentiable mapping
│   │   ├── helios_pytorch_renderer.py            ← nvdiffrast renderer + multi-scale pyramid
│   │   └── part_tensor_to_40d.py                 ← closed-form IK + XML assembler
│   ├── training/
│   │   ├── train_hierarchical_flow_matching.py   ← [CRITICAL] main training loop + --eval_every + --resume
│   │   └── flow_matching.py                      ← Rectified Flow scheduler
│   ├── dataset/
│   │   └── part_array_dataset.py                 ← [CRITICAL] cache-first filtering, 16-ch pyramid
│   ├── eval/
│   │   └── eval_hierarchical_self_consistency.py ← 7-column diagnostic panel generator
│   └── checkpoints/
│       ├── hierarchical_latent_fm/               ← training checkpoints (epoch_025.pt, epoch_050.pt, ...)
│       └── organ_vae/organ_latent_vae_best.pt    ← frozen OrganLatentVAE bridge
├── slurm_scripts/
│   ├── train_hierarchical_flow_matching.sh       ← [CRITICAL] launcher w/ INIT_CHECKPOINT support
│   └── generate_helios_dataset_jobs.sh           ← TIME_LIMIT="24:00:00" (updated)
├── dataset/
│   ├── helios_data/cowpea/                       ← 51,880 XML files (growing)
│   └── cache/cowpea_curv26/                      ← 22,600 cached .pt tensors (growing)
└── docs/
    ├── ongoing/AGENT_TAKEOVER_GUIDE.md           ← [THIS FILE]
    ├── results/assets/                           ← diagnostic panels + milestone figures
    └── results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md
```

---

## 7. Training Configuration (Current Job 38146809)

```bash
# Key hyperparameters (from log line 4 + 77)
Batch per GPU = 48 | Global Batch = 192
DINOv2 backbone: 175 tensors, lr=6.0e-05
3D/decoder:      240 tensors, lr=2.0e-04
VRAM: 37.2/47.4 GB (78.6%)
Epochs: 500
save_every: 25
eval_every: 1  ← NOTE: this was added in c587767 but 38146809 started on 0389ca5 (no eval_every)
```

**Loss components being optimized:**
- `VelLoss`: Flow matching velocity MSE (~0.80 at Epoch 26)
- `AncPosLoss`: 3D anchor position regression (~0.015)
- `PhyLoss`: Physical plant height error (Pred vs GT DAP height ~43cm)
- `ExistLoss`: Slot existence classification BCE (~0.22)
- `DepthLoss`: CHM depth MAE (~0.06)
- `CosLoss`: Cosine color similarity (~0.64)
- `DiceLoss`: Silhouette Dice (~0.68)
- `ClsAcc`: Classification accuracy (~66%)

**Loss trend (Epoch 1→26):**
```
Epoch 001 Loss: 166.07 → Epoch 010 ~8.5 → Epoch 020 ~6.0 → Epoch 026 ~5.36
```

---

## 8. Data Contracts & Tensor Specifications

### A. Canonical 14D Part Tensor
Every organ: $\mathbf{p} \in \mathbb{R}^{14} = [\text{organ\_type}(1), \mathbf{x}_{\text{base}}(3), \mathbf{r}_{6\text{D}}(6), \mathbf{s}_{\text{scale}}(3), \text{curvature}(1)]$

| Index | Field | Description |
| :---: | :--- | :--- |
| `0` | `organ_type` | Categorical integer organ identifier |
| `1:4` | `base_xyz` | 3D base coordinate in meters |
| `4:10` | `rot6d` | Continuous 6D rotation (Gram-Schmidt to SO(3)) |
| `10:13` | `scale_xyz` | Physical dimensions in meters |
| `13` | `curvature` | 3D bending curvature (°/m) |

### B. Organ Types
```python
ORGAN_NONE=0, ROOT_META=1, SHOOT_META=2, INTERNODE=3, PETIOLE=4, LEAF=5,
PEDUNCLE=6, BUD_DORMANT=7, BUD_ACTIVE=8, FLOWER_CLOSED=9, FLOWER_OPEN=10,
FRUIT=11, BUD_ABORTED=12
```

### C. FM Encoding (16-channel pyramid input)
- Channels: `[z1(4ch) | z2(4ch) | z4(4ch) | z8(4ch)]` = 16ch total
- Per zoom: `[RGB(-1,1)×3 | CHM_depth×1]`
- Cache key: `{xml_basename}.pt` in `dataset/cache/cowpea_curv26/`

### D. Physical Scale Purity Contract
- `scale_xyz` **MUST ALWAYS** represent true physical dimensions in meters.
- **NEVER** multiply existence into `scale_xyz`.
- Soft existence passed independently via vertex opacities.

---

## 9. Common Gotchas (Critical)

1. **`PYTHONPATH=.` is Mandatory**: Always set before running any script.
2. **`--eval_every` requires NEW job**: Job 38146809 started on old code. Panels only appear after re-submission with `INIT_CHECKPOINT`.
3. **Helios XML Non-Negative**: Any `<internode_radius>` or `<leaf_scale>` ≤ 0 causes `SIGABRT`. Always clamp to ≥ 1e-4 m.
4. **CHM Depth = Height from Ground** (inverse of camera-distance depth).
5. **Nadir-View CHM Blindness**: Vertical stems project ~12px sliver with CHM=0. Use 3D tip anchor loss as compensation.
6. **Cache-First Filtering**: `PartArrayDataset` now filters XML paths by cache `.pt` existence. Adding new XMLs without caching them first has no effect on training.
7. **Dormant Slot Damping**: New in 0389ca5. Weight = 0.05. Reduces unmatched slot velocity norms to prevent reproductive ghost artifacts.
8. **Helios Leaf Roll Non-Equivariance**: `PlantArchitecture.cpp` applies leaf roll about world axes → XML pipeline is not yaw-equivariant. Use PyTorch renderer for optimization-time verification.

---

## 10. Verification Commands

```bash
# Check training epoch + key losses
tail -n 5 slurm_scripts/logs/hierarchical_fm_38146809.log

# Check if Epoch 25 checkpoint exists (trigger for job restart)
ls -lh diffusion_based/checkpoints/hierarchical_latent_fm/

# View latest evaluation panel (once per-epoch eval starts)
ls -lt docs/results/assets/hierarchical_self_consistency_epoch_*.png | head -5

# Monitor Helios dataset growth
watch -n 60 "find dataset/helios_data/cowpea -name '*.xml' | wc -l && find dataset/cache/cowpea_curv26 -name '*.pt' | wc -l"

# Helios job status
squeue -u lion397 | grep helios

# W&B dashboard
# Project: part-flow-matching
# Run: hierarchical-3stage-cascaded-cowpea (run bz7assfz)
```

---

## 11. Commit History (2026-09-08 Session)

| Hash | Commit Message |
| :--- | :--- |
| `0435325` | feat(training): support --resume to restore epoch counter and optimizer state seamlessly |
| `c587767` | feat(eval): decouple validation image generation to run every 1 epoch (--eval_every 1) |
| `97420ee` | fix(slurm): increase helios dataset pipeline time reservation to 24 hours |
| `99c2b5a` | docs(milestone): track 3-stage cascaded flow matching milestone |
| `027118d` | fix(dataset): early filter xml_paths by cache_dir for consistent 16-channel tensors |
| `0389ca5` | feat(training): increase batch size to 48 (global 192), add dormant slot damping, accelerate anchor convergence |

---

## 12. Milestone Documents (Chronological)

| File | Date | Summary |
| :--- | :--- | :--- |
| [`docs/results/20260907_latent_hierarchical_flow_matching_500epoch_report.md`](../results/20260907_latent_hierarchical_flow_matching_500epoch_report.md) | 2026-09-07 | Option B 500-epoch report: 55.1% IoU, 2.49cm height error |
| [`docs/results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md`](../results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md) | 2026-09-08 | Epoch 150 spatial vision breakthrough: 49.2% mean IoU, 2.6cm RMSE |
| [`docs/results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md`](../results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md) | 2026-09-08 | 3-stage cascaded architecture scaling: batch 192, dormant slot damping |
| [`docs/ongoing/20260908_cascaded_architecture_design_space_analysis.md`](20260908_cascaded_architecture_design_space_analysis.md) | 2026-09-08 | ADR: Det+Diff cascade analysis |

---

*You are now fully equipped to take over. The most critical immediate action is waiting for Epoch 25 checkpoint (`hierarchical_fm_epoch_025.pt`) and restarting the job with `INIT_CHECKPOINT` to activate per-epoch validation panels.*
