# Agent Takeover & Engineering Handover Guide
**Project: Image-to-L-System / 3D Inverse Procedural Plant Reconstruction**  
**Last Updated:** 2026-09-09 ~23:30 PDT (XML-Phytomer Clustering + Structural Assembly + Profiling)  
**Primary Author/Agent:** Antigravity Autonomous Agent (Pair programming with Heesup Yun)  
**Environment:** Linux, Python 3.10+, Mamba (`mamba activate digital-crops`), CUDA, PyTorch, `nvdiffrast`, Helios C++ OptiX Raytracer.  

---

## 0. Quick State Check (Run This First)

```bash
# Where is training right now?
tail -n 10 slurm_scripts/logs/hierarchical_fm_38183271.log

# All running/pending jobs
squeue -u lion397

# Dataset size
find dataset/helios_data/cowpea -name "*.xml" | wc -l   # 100,000
find dataset/cache/cowpea_curv26 -name "*.pt" | wc -l   # 100,000 (all have phytomer_ids)

# Available checkpoints
ls -lh diffusion_based/checkpoints/hierarchical_latent_fm/*.pt
ls -lh diffusion_based/checkpoints/phytomer_vae_xml/*.pt
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
[Stage 0] DINOv2 ViT-S/14 → PETR 3D Ray PE → 3D-aware image tokens
       ↓
[Stage 1] MacroBiologicalHead (CLS token) → DAP + Phytomer count + soft margin existence prior
       ↓
[Stage 2] CoarseSkeletalTransformer — DETERMINISTIC set transformer (4 TransformerDecoder
          layers, NO flow matching) → 3D node scaffold: anchor xyz (ref_points + delta),
          6D rotation, existence logits (GT-DAP Matryoshka slicing, power-of-2 tiers)
       ↓
[Stage 3] FineBotanicalFlowMatchingDecoder (6 transformer layers) → 8 organ slots per anchor
          (4,096 max slots): Rectified Flow Matching velocity over 16D latents,
          20-step Heun ODE at inference
       ↓
OrganLatentVAE (frozen) → 16D latent → 14D Part Tensor → Helios XML
```

> **Naming note**: "Stage 2" was historically labeled "Coarse Flow Matching" in older docs.
> Per the ratified design-space decision (Combination 2: Deterministic Scaffold + 16D Latent FM,
> `docs/archived/design/20260908_cascaded_architecture_design_space_analysis.md`), Stage 2 is a
> **deterministic regression head** (xyz + 6D rot + existence); flow matching occurs only in Stage 3.

### Milestone History:
- ✅ **Phase 1** (ICP / Diff Render / Flow Matching benchmark): Complete.
- ✅ **Phase 2** (Over-allocation topology, variable organ counts): Complete.
- ✅ **Option B: 500-Epoch 16D Latent Hierarchical FM** (Job `38143585`): 55.1% mean IoU, peak 67.9%, 2.49cm height error, 0 ghost organs.
- ✅ **3-Stage Cascaded Architecture** (Current, Job `38146809`): Running past Epoch 54 with idle slot damping, scaled global batch size (192), and accelerated anchor convergence.

---

## 2. Active SLURM Jobs (as of 2026-09-08 ~18:45 PDT)

| Job ID | Name | Status | Node | Notes |
| :--- | :--- | :---: | :--- | :--- |
| **38146809** | `hierarchical_fm` | **RUNNING** | `gpu-10-54` | Main training job, **Epoch 54/500**, 4× RTX 6000 Ada, 37.2 GB VRAM (78.6%), TIME_LEFT ~20h |
| **38147219** | `ondemand/sys/dashboa` | RUNNING | `gpu-5-58` | User's interactive OnDemand desktop — **DO NOT CANCEL** |
| **38147132–38147171** | `helios_pipe_*` | R/PD | multiple | 40-shard Helios dataset synthesis, 24-hour time limit, ~25 running, ~5 pending |

### Helios Dataset Generation Progress
- **65,488 XMLs** generated so far out of target ~120,000 (54.6% complete)
- **35,266 cache `.pt` tensors** in `dataset/cache/cowpea_curv26/`
- All jobs running with `TIME_LIMIT=1-00:00:00` (24 hours); automatic resumption/skip on existing files.

---

## 3. Session Changes & Technical Breakthroughs (2026-09-08)

### 3.1 Algorithm Improvements (Commits `0389ca5`, `5b9fbf0`, `54579bc`)
- **Idle Slot Damping**: Added $L_2$ velocity regularizer $\mathcal{L}_{\text{idle}} = 0.05 \sum_{i \notin \mathcal{M}} \|\mathbf{v}_i\|^2$ for unmatched slots to prevent ghost organs during vegetative stages (DAP < 40).
- **Top-k Threshold Guard**: `valid_topk = top_k_indices[combined_prob[b, top_k_indices] > 0.15]` prevents dormant ghost slot emergence.
- **Backbone & Anchor Acceleration**: `loss_anchor_pos` weight increased to **4.0**, DINOv2 backbone LR increased to **6e-5** (`args.lr * 0.3`).

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

### 3.7 Epoch 50 Milestone & 3D Botanical Skeleton Analysis
At Epoch 50, self-consistency evaluation yielded:
- **`AncPosLoss`**: `0.0089` (normalized smooth L1)
- **`Node RMSE`**: `1.7 cm`
- **`Depth MAE`**: `8.25 cm` (improved from 15.73 cm at Epoch 25)
- **`Silhouette IoU`**: `9.7%`

#### Why Visual Skeletons in Column 6 Look Inaccurate Despite `AncPos: 0.009`
Visual inspection of `docs/results/assets/hierarchical_self_consistency_epoch_050.png` showed apparent distortions in Column 6 ("3D Botanical Skeleton"). Investigation revealed three primary factors:
1. **Position vs. Visual Segment Geometry:**  
   `AncPosLoss` only checks anchor $(x, y, z)$ coordinates. In contrast, Column 6 plots internode segments:
   $$\mathbf{x}_{\text{tip}} = \mathbf{x}_{\text{base}} + \mathbf{R}_{6\text{D}}[:, 1] \times L_{\text{internode}}$$
   At Epoch 50, positional bases are within ~1.7 cm, but 6D continuous rotation matrices ($\mathbf{R}_{6\text{D}}$) and length scales are still converging. A 20° rotation error deflects stem tubes into the ground or empty space.
2. **One-Way Chamfer Distance Metric Bias (`Node RMSE`):**  
   The evaluation code computes $\min_{j} \|\mathbf{p}_i - \mathbf{g}_j\|$. At Epoch 50, predicted nodes cluster near the plant stem/center. Because every central predicted node is close to at least one GT node, the RMSE is artificially small (1.7 cm), even though outer radiating canopy branches (DAP 55) are not yet populated.
3. **Classification Accuracy (`ClsAcc: 66%`):**  
   Approximately one-third of active slots have imperfect class assignment, leading to occasional internodes where petioles/leaves belong.
4. **Natural Training Timeline:**  
   In hierarchical botanical flow matching, coarse anchor coordinates settle early (Epochs 1–40), while fine rotations, branch outward spread, and topological continuity align during Epochs 50–200.

*(Full technical report: [`docs/results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md`](../results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md))*

---

## 4. Current Checkpoints & Job Status

Checkpoints saved on disk:
```bash
diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt  (1.4 GB)
diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_050.pt  (1.4 GB)
diffusion_based/checkpoints/phytomer_vae_xml/phytomer_vae_64d_best.pt  (accepted VAE, XML-phytomer clustering)
```

### Active Job (2026-09-09 ~23:30 PDT):
- **Job `38183271`** (gpu-10-54): phytomer-mode training on the FULL 100k dataset,
  `FLOW_GRANULARITY=phytomer`, frozen `phytomer_vae_xml` checkpoint, render
  fraction 0.167 + 2-scale pyramid (1x/2x). Epoch 1: ClsAcc 92.6% (was 0.0%
  before the XML-phytomer fix), step ~8s (render 2.6s, packet build 4.2s).

### Actionable Choice for Incoming Agent:
- **Option A (Recommended - Let it run):**  
  Job `38183271` is healthy and past Epoch 1 with ClsAcc 92.6%. It will save
  `epoch_025.pt` around Epoch 25 and evaluate the next panel automatically.
- **Option B (Immediate Per-Epoch Eval Panels):**  
  If you specifically require per-epoch evaluation images (`--eval_every 1`),
  cancel `38183271` and launch with:
  ```bash
  scancel 38183271
  INIT_CHECKPOINT=diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt \
    sbatch slurm_scripts/train_hierarchical_flow_matching.sh
  ```

---

## 5. Next Steps (Priority Order)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P1** | Monitor Job `38183271` to Epoch 25 / Epoch 50 | Check `hierarchical_self_consistency_epoch_025.png` |
| **P2** | Track Helios dataset generation | 100k XMLs / 100k cache tensors (all with `phytomer_ids`) |
| **P3** | Monitor 6D rotation convergence in Stage 3 | Check whether stem segments align upward/outward in Epoch 25/50 panels |
| **P4** | Evaluate Bidirectional Chamfer Distance | Add max/mean distance from GT $\to$ Pred to avoid one-way clustering metric bias |
| **P5** | Expand training to full 120k dataset | Once all Helios shards complete, launch next 500-epoch scaling run |
| **P6 (NEW 2026-09-09)** | PhytomerVAE-64 validation gate | Training on local TITAN RTX (`diffusion_based/checkpoints/phytomer_vae/`); GO/NO-GO = roundtrip parity vs OrganLatentVAE. See `docs/ongoing/20260909_phytomer_latent_and_local_matching.md` §2 |
| **P7 (NEW 2026-09-09)** | Stage-3 phytomer integration (ONLY after P6 passes) | Behind `flow_granularity` flag; plan in design doc §2 integration plan |
| **P8 (NEW 2026-09-09)** | PhytomerVAE GUI app | **DELEGATED to another agent** — PCA latent cloud + 64D sliders → 3D render. Handoff spec in design doc §4.2 |
| **P9 (NEW 2026-09-09)** | Launch phytomer-mode training | `--flow-granularity phytomer` wired + smoke-tested; launch on the 100k dataset once shards complete |

---

## 6. Key Source Code Map

```
/home/lion397/codes/image-to-l-system/
├── diffusion_based/
│   ├── models/
│   │   ├── hierarchical_part_flow_matching.py    ← [CRITICAL] 3-stage model + sample() with threshold guard
│   │   ├── plant_organ_array.py                  ← 14D constants, organ types, XML roundtrip
│   │   ├── helios_pytorch_geometry.py            ← mesh builder + differentiable mapping
│   │   ├── helios_pytorch_renderer.py            ← nvdiffrast renderer + multi-scale pyramid
│   │   └── part_tensor_to_40d.py                 ← closed-form IK + XML assembler
│   ├── models/
│   │   ├── organ_latent_vae.py                   ← frozen OrganLatentVAE bridge (16D/organ)
│   │   ├── phytomer_vae.py                       ← [NEW 2026-09-09] PhytomerVAE (whole-phytomer 64D latent)
│   │   └── hierarchical_part_flow_matching.py    ← [CRITICAL] 3-stage model + PhytomerFlowMatchingDecoder (flow_granularity)
│   ├── training/
│   │   ├── train_hierarchical_flow_matching.py   ← [CRITICAL] main training loop + --flow-granularity phytomer (73D bridge targets)
│   │   ├── train_phytomer_vae.py                 ← [NEW 2026-09-09] PhytomerVAE trainer (canonical packets)
│   │   ├── hierarchical_hungarian_matcher.py     ← [CRITICAL] coarse anchor + local fine bipartite matching (+soft locality radius opt-in, skip_fine)
│   │   └── flow_matching.py                      ← Rectified Flow scheduler
│   ├── dataset/
│   │   ├── part_array_dataset.py                 ← [CRITICAL] cache-first filtering, 16-ch pyramid
│   │   └── phytomer_packets.py                   ← [NEW 2026-09-09] canonical 8-slot packet builder (relative rotations, Option-1 reference)
│   ├── eval/
│   │   └── eval_hierarchical_self_consistency.py ← 7-column diagnostic panel generator
│   └── checkpoints/
│       ├── hierarchical_latent_fm/               ← training checkpoints (epoch_025.pt, epoch_050.pt, ...)
│       ├── organ_vae/organ_latent_vae_best.pt    ← frozen OrganLatentVAE bridge
│       └── phytomer_vae_relative_d/              ← [NEW] accepted PhytomerVAE-64 (rot 3.16°, relative-rotation)
├── slurm_scripts/
│   ├── train_hierarchical_flow_matching.sh       ← launcher w/ INIT_CHECKPOINT support
│   └── generate_helios_dataset_jobs.sh           ← TIME_LIMIT="24:00:00"
├── dataset/
│   ├── helios_data/cowpea/                       ← 65,488 XML files (growing)
│   └── cache/cowpea_curv26/                      ← 35,266 cached .pt tensors (growing)
└── docs/
    ├── ongoing/
    │   ├── README.md                             ← ongoing status dashboard
    │   └── AGENT_TAKEOVER_GUIDE.md               ← [THIS FILE] master handoff
    ├── results/
    │   ├── 20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md ← Epoch 50 geometry report
    │   └── assets/                               ← diagnostic panels
```

---

## 7. Training Loss Progression (Job 38146809)

| Epoch | Total Loss | VelLoss | AncPosLoss | PhyLoss | ExistLoss | DepthLoss | DiceLoss | ClsAcc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 166.07 | 3.07 | 0.300 | 33.20 | 1.20 | 0.15 | 0.88 | 42.0% |
| **10** | 8.52 | 1.10 | 0.035 | 3.80 | 0.35 | 0.09 | 0.74 | 58.5% |
| **25** | 5.52 | 0.85 | 0.016 | 2.40 | 0.24 | 0.07 | 0.70 | 63.2% |
| **40** | 4.12 | 0.70 | 0.011 | 1.30 | 0.18 | 0.06 | 0.67 | 64.8% |
| **50** | **3.58** | **0.57** | **0.0089** | **0.85** | **0.15** | **0.065** | **0.65** | **66.3%** |
| **54** | 3.75 | 0.58 | 0.0087 | 0.92 | 0.16 | 0.061 | 0.67 | 65.7% |

---

## 8. Common Gotchas (Critical for Successors)

1. **`PYTHONPATH=.` is Mandatory**: Always run commands from workspace root with `PYTHONPATH=.`.
2. **Helios XML Non-Negative**: Any `<internode_radius>` or `<leaf_scale>` $\le 0$ causes a C++ core dump (`SIGABRT`). Always clamp to $\ge 1\text{e-}4$.
3. **CHM Depth = Height Above Ground**: Inverted relative to standard LiDAR/camera distance. Ground is 0, plant canopy apex is maximum height.
4. **Cache-First Filtering**: Only XMLs that have a corresponding `.pt` tensor in `dataset/cache/cowpea_curv26/` are loaded by the dataloader.
5. **Node RMSE vs. Tree Topology**: Do not rely solely on `Node RMSE`. Check Column 5 (Oblique 3D Mesh Composite) and Column 6 (Botanical Skeleton) to verify 3D rotation and branch divergence.
6. **OnDemand Desktop (Job 38147219)**: Running on `gpu-5-58` — do not touch or kill.

---

## 9. Milestone Documents (Chronological)

| File | Date | Summary |
| :--- | :--- | :--- |
| [`docs/results/20260907_latent_hierarchical_flow_matching_500epoch_report.md`](../results/20260907_latent_hierarchical_flow_matching_500epoch_report.md) | 2026-09-07 | Option B 500-epoch report: 55.1% IoU, 2.49cm height error |
| [`docs/results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md`](../results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md) | 2026-09-08 | Epoch 150 spatial vision breakthrough: 49.2% mean IoU, 2.6cm RMSE |
| [`docs/results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md`](../results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md) | 2026-09-08 | 3-stage cascaded architecture scaling: batch 192, dormant slot damping |
| [`docs/results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md`](../results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md) | 2026-09-08 | **[NEW]** Root cause analysis of Epoch 50 skeleton geometry vs AncPos loss |
