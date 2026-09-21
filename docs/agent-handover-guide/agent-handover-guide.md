---
title: "Agent Takeover & Engineering Handover Guide"
date: 2026-09-16
tags: [handover, session]
status: active
---

# Agent Takeover & Engineering Handover Guide
**Project: Image-to-L-System / 3D Inverse Procedural Plant Reconstruction**  
**Last Updated:** 2026-09-21
**Environment:** Linux, Python 3.10+, Mamba (`mamba activate digital-crops`), CUDA, PyTorch, `nvdiffrast`, Helios C++ OptiX Raytracer.

> **This document is the durable manual: how the repo is laid out, how to run things, and what to
> watch out for. It carries no project state.** For where the project stands, start at
> [`docs/_index.md`](../_index.md); for the evidence behind it, read
> [`current-state.md`](../current-state/current-state.md).
>
> The dated state sections this guide used to carry (§0-A, 0-B, 0-C and the 2026-09-11 numbered
> status sections) were moved on 2026-09-21 to
> [`archive/20260921-superseded-state-sections`](../archive/20260921-superseded-state-sections/20260921-superseded-state-sections.md).
> Three of them each said "read this first" while contradicting one another.

> **2026-09-16 repository restructure:** the package formerly at `diffusion_based/` is now
> `plant_recon/`; checkpoints moved to `outputs/checkpoints/`, job logs to `outputs/logs/`,
> wandb to `outputs/wandb/`, the real-image track to `use_cases/real_world/`, Digital-Crops to
> `submodules/Digital-Crops/`, and the VLM track to `archive/lm_based/`. See
> [`code-structure.md`](../architecture/code-structure/code-structure.md).

---


## 0. Orientation — start here

Absorbed from the former `developer-onboarding.md` (2026-09-17) so there is one entry point rather than two overlapping ones. The live experiment and engineering state is NOT repeated here — start at [`_index.md`](../_index.md), which routes to it.

### 0.1 The repo in 60 seconds

| Where | What |
| :--- | :--- |
| `plant_recon/` | The library — `models/` (Stages 0–4, renderer, XML export), `training/`, `dataset/`, `eval/` |
| `use_cases/real_world/` | Sim-to-real application: detectors, real-field datasets, multi-plant pipeline |
| `submodules/Digital-Crops/` | Helios C++ OptiX engine (git submodule, pinned) |
| `dataset/` | Data only: 100k Helios XMLs + `.pt` caches (gitignored) |
| `outputs/` | Everything generated: `checkpoints/`, `logs/`, `wandb/`, `weights/`, `eval/` |
| `scripts/` | The 3 cluster launchers (`.sh` only) |
| `docs/` | `current-state/` (live dashboard) + `agent-handover-guide/` (this manual) |

**Environment**: `mamba activate digital-crops`, `export PYTHONPATH=.` from the repo root.
**Sanity check**: `pytest tests/` → 99 collected (2 pre-existing stale failures: `test_dinov2_3d_spatial.py`,
`tests/unit/test_ik_fix.py` — old model API, not your problem unless you touch them).

### 0.2 Where to start developing (ranked)

#### A. Close the appearance gap in training — the top item

The network reads **RGB only** (`DINORayEncoder.forward` takes channels 0:3); training RGB is
flat-shaded green on uniform tan with **zero augmentation**, while 100k Helios raytraced renders
(`dataset/helios_data/cowpea/*_rad.jpeg`, textured soil, shadows, specular) sit unused.

1. Fine-tune from `hierarchical_fm_v10_cam` ep160 on Helios raytraced crops at the cache's fixed
   1.2 m window framing (`plant_recon/eval/render_helios_eval_crops.py` shows the exact recipe).
2. Add ordinary augmentation: colour jitter, blur, sensor noise, random shadows, real soil
   composites under the flat render mask.
3. Re-measure with the same flat-vs-Helios protocol used in the assessment (§4 there) — that is the
   number that must move. Watch the **DAP probe error** and **active-node count**, not just P.

#### B. Fix the real-image crop framing (one line, then a measurement)

`use_cases/real_world/dataset/real_plant_crop_utils.py::build_pyramid_16ch` uses 1.2× the *detector
bbox* as the zoom-1× window; the training cache uses a **fixed 1.2 m ground window**. Every plant on
a real crop therefore looks ~4× too big, and Stage 1's DAP head reads size in a fixed window — the
reported "DAP uncorrelated with truth" follows. Fix: `window_px = 1.2 m × (frame_px / plot_width_m)`
(the multi-plant script already assumes the 1.3 m plot width).

#### C. Refinement safety on real crops

- **Bound realized organ size in metres**, not the phytomer scale `s_a` — refinement currently
  inflates leaf scale 2.4–2.6× against a loose blob-mask target.
- **Make existence continuous** in refinement (soft compositing weight on Stage 2's logits) so the
  optimiser can switch nodes on/off; lowering the threshold alone adds false positives it can't remove.

#### D. Lower-priority / don't bother

- **Stop investing in the Stage 3 latent.** Refinement from a mean latent reaches the same 66.5 as
  the sampled latent; the network's real contribution is node positions/existence/topology. If
  Stage 3 stays, a deterministic regression head is enough.
- **Guided sampling is not worth it** (neutral alone, *hurts* when chained with refinement).
- Larger backbone with multizoom (`dinov2_vitb14`) is one flag away if node error (4–6 cm at 7.5 cm
  per token) ever becomes the binding constraint.
- Use 50–100 eval plants for decisions — epoch spread is ±3 points on the current 20-plant set.

### 0.3 Everyday workflows

```bash
# Evaluate a checkpoint under the strict protocol + refinement (200 steps, not the 40-step default:
# 2026-09-17 measured 66.3 -> 73.7 by step count alone)
python plant_recon/eval/eval_test_time_refinement.py --checkpoint outputs/checkpoints/hierarchical_fm_v10_cam/hierarchical_fm_epoch_160_ema.pt --steps 200

# Self-consistency panels
python plant_recon/eval/eval_hierarchical_self_consistency.py --help

# Train (cluster): one launcher, env overrides for every knob
sbatch scripts/train_hierarchical_flow_matching.sh            # current v9/v10 recipe
TRAIN_VAE=1 sbatch scripts/train_hierarchical_flow_matching.sh # fresh PhytomerVAE first

# Real image: one frame -> detections -> per-plant XML -> refinement -> rendered plot
python use_cases/real_world/eval/run_multiplant_scene.py --image <frame.jpg>
```

### 0.4 Gotchas worth memorising

- `PYTHONPATH=.` from the repo root is mandatory; the package imports as `plant_recon.*`.
- CHM depth = height **above ground** (ground 0, apex max) — inverted vs camera distance.
- Helios XML with any scale/radius ≤ 0 segfaults the C++ binary; clamp at export.
- Packet cache order must match the VAE (`PHYTOMER_TERMINAL_LAST=1` ⇔ `_pkt_v9` cache). Never edit
  cache files in place — bump `PKT_VERSION` and regenerate.
- Don't stack a foreground eval on the same GPU as a live local training run (two stalls this week).
- A checkpoint written before the 2026-09-16 restructure stores `diffusion_based/...` paths inside its
  own args; `plant_recon/eval/ckpt_compat.py` remaps them, and every eval script calls it.
- Never cancel Heesup's `regen_*` jobs or the OnDemand desktop jobs; training goes on
  `gpu-6000_ada-h`/`low`.

---

## 0-D. Quick state check (commands)

```bash
# Where is training right now? (latest job log; logs sit in one folder per start date)
ls -t outputs/logs/*/hierarchical_fm_*.log | head -3
tail -n 30 "$(ls -t outputs/logs/*/hierarchical_fm_*.log | head -1)"

# All running/pending jobs
squeue -u lion397

# Dataset size
find dataset/helios_data/cowpea -name "*.xml" | wc -l   # 100,000
find dataset/cache/cowpea_curv26 -name "*.pt" | wc -l   # 100,000 (all have phytomer_ids)

# Available checkpoints
ls -lh outputs/checkpoints/hierarchical_fm_v10_cam/*.pt        # current lineage (ep160 raw + _ema files)
ls -lh outputs/checkpoints/phytomer_vae_v9_tl_rw4_20k/*.pt   # current VAE (v9, terminal-last packets)
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
> - Fully transitioned to **hybrid decoupling** to resolve error amplification (`VelLoss: 1050~2000`) of the 76D Bridge Flow ($z_0 = \text{node\_pos} + \epsilon$).
> - Stage 2 is dedicated to the full-body 3D skeleton (`pos`, `rot`, `scale`, `exist`), while Stage 3 is dedicated to 64D intra-phytomer latent space Flow Matching from a standard normal distribution.
> - Streamlined loss functions from 10 $\to$ 8 (removed `loss_cos`, `loss_dap`, added `loss_phytomer_rot` canonical supervision).

### Milestone History:
- ✅ **Phase 1** (ICP / Diff Render / Flow Matching benchmark): Complete.
- ✅ **Phase 2** (Over-allocation topology, variable organ counts): Complete.
- ✅ **Option B: 500-Epoch 16D Latent Hierarchical FM** (Job `38143585`): 55.1% mean IoU, peak 67.9%, 2.49cm height error, 0 ghost organs.
- ✅ **3-Stage Cascaded Architecture** (Current, Job `38146809`): Running past Epoch 54 with idle slot damping, scaled global batch size (192), and accelerated node convergence.

---

## 6. Key Source Code Map

```
/home/lion397/codes/image-to-l-system/
├── plant_recon/
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
│   └── checkpoints/                              ← MOVED 2026-09-16 to outputs/checkpoints/ (2026-09-11 entries below; current: hierarchical_fm_v10_cam/, phytomer_vae_v9_tl_rw4_20k/)
│       ├── hierarchical_latent_fm/               ← epoch_025~100 (390MB, new 3-stage arch); epoch_125~500 (552MB, OLD arch — incompatible)
│       ├── organ_vae/organ_latent_vae_best.pt    ← frozen OrganLatentVAE bridge
│       └── phytomer_vae_v3/                      ← [ACCEPTED DEFAULT] PhytomerVAE-64 v3, 10-slot normalized (val recon 0.070, cls 100%)
├── scripts/
│   ├── train_hierarchical_flow_matching.sh       ← launcher; defaults: VAE v3, SLOTS_PER_PHYTOMER=10, BACKBONE_LR_RATIO=0.3
│   ├── generate_helios_dataset_jobs.sh           ← full pipeline: XML synth + cache (+ pkt/latent) in one pass
│   └── archive/slurm_scripts/submit_backbone_ablation.sh  ← A/B dispatcher (archived; DAP-spread runs, sequential chain)
├── dataset/
│   ├── helios_data/cowpea/                       ← 100,000 XML files (complete)
│   ├── cache/cowpea_curv26/                      ← 100,000 cached .pt (image+nodes+phytomer_ids, complete)
│   ├── cache/cowpea_curv26_pkt/                  ← 100,000 pkt v3 (10-slot, absolute packets + normalized latent)
│   └── cache/cowpea_curv26_subset4k/             ← 4,000 symlinks (40/DAP × 100 DAP) for fast smoke tests
└── docs/                                         ← restructured 2026-09-16; docs/_index.md is the map
    ├── handovers/
    │   ├── current-status.md                     ← live status dashboard (was docs/ongoing/README.md)
    │   └── agent-takeover-guide/agent-takeover-guide.md  ← [THIS FILE] master handoff
    ├── experiments/                              ← results reports (was docs/results/), figures in each report's assets/
    │   ├── 20260910-gradient-explosion-debug/    ← grad explosion root cause
    │   └── 20260914-stage2-burst-fix-roundtrip/assets/fig12_phytomer_10slot_helios_roundtrip.png  ← [2026-09-14] dataset plants DAP 15/40/75, Helios round-trips
    ├── architecture/current-architecture/        ← full math derivation of the 4-stage pipeline
    └── assetsfig10_helios_per_organ_mask_comparison.png  ← [2026-09-11] per-organ IoU diagnosis (DAP 10/50/90)
```

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
| [`docs/engineering/20260909-phytomer-latent-matching/20260909-phytomer-latent-matching.md`](../engineering/20260909-phytomer-latent-matching/20260909-phytomer-latent-matching.md) | 2026-09-09/10 | **[MASTER ENGINEERING LOG]** §4.6–4.9: pipeline refactor, backbone A/B, fruit fix, v3 scale-normalized packets + 76D flow, render-pipeline timing |
| [`docs/experiments/20260907-latent-fm-500epoch/20260907-latent-fm-500epoch.md`](../experiments/20260907-latent-fm-500epoch/20260907-latent-fm-500epoch.md) | 2026-09-07 | Option B 500-epoch report: 55.1% IoU, 2.49cm height error |
| [`docs/experiments/20260908-3d-spatial-vision-milestone/20260908-3d-spatial-vision-milestone.md`](../experiments/20260908-3d-spatial-vision-milestone/20260908-3d-spatial-vision-milestone.md) | 2026-09-08 | Epoch 150 spatial vision breakthrough: 49.2% mean IoU, 2.6cm RMSE |
| [`docs/experiments/20260908-3stage-cascaded-milestone/20260908-3stage-cascaded-milestone.md`](../experiments/20260908-3stage-cascaded-milestone/20260908-3stage-cascaded-milestone.md) | 2026-09-08 | 3-stage cascaded architecture scaling: batch 192, dormant slot damping |
| [`docs/experiments/20260908-skeleton-geometry-chamfer/20260908-skeleton-geometry-chamfer.md`](../experiments/20260908-skeleton-geometry-chamfer/20260908-skeleton-geometry-chamfer.md) | 2026-09-08 | Root cause analysis of Epoch 50 skeleton geometry vs AncPos loss |
| [`docs/experiments/20260910-gradient-explosion-debug/20260910-gradient-explosion-debug.md`](../experiments/20260910-gradient-explosion-debug/20260910-gradient-explosion-debug.md) | 2026-09-10 | **[KEY]** Gradient explosion diagnosis: Float32 overflow + z0 detach bug + deadlock mechanism |
| `docs/assetsfig10_helios_per_organ_mask_comparison.png` | **2026-09-11** | **[NEW]** Per-organ COCO mask IoU + Depth PSNR roundtrip: Leaf ✅, Internode/Petiole ❌ |
| `docs/experiments/20260914-stage2-burst-fix-roundtrip/assets/fig12_phytomer_10slot_helios_roundtrip.png` | **2026-09-14** | Dataset plants DAP 15/40/75: GT mesh and 10-slot assembly under the same nadir and 45° cameras, Helios round-trip via packets and via the VAE (`eval_phytomer_10slot_assembly_views.py`) |
