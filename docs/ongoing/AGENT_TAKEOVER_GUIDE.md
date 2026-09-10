# Agent Takeover & Engineering Handover Guide
**Project: Image-to-L-System / 3D Inverse Procedural Plant Reconstruction**  
**Last Updated:** 2026-09-10 PDT (v2 10-slot migration + packet cache complete + GUI hardening + stem/leaflet base-rule fixes)  
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
ls -lh diffusion_based/checkpoints/phytomer_vae_v2/*.pt
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
diffusion_based/checkpoints/phytomer_vae_v2/phytomer_vae_64d_best.pt  (ACCEPTED v2: 10-slot, VAE 240D in, val recon 0.0263, cls 100%)
```

### Current state (2026-09-10) — v2 10-slot migration complete:
- **Job `38183271`** (phytomer training) remains cancelled; resume per P2.
- **Phytomer packet cache**: `dataset/cache/cowpea_curv26_pkt/` holds
  **100,000 / 100,000** v2 10-slot samples (frozen-VAE latent included;
  regenerated 2026-09-10, 0 errors). v1 8-slot backed up as
  `cowpea_curv26_pkt_v1_8slot/`.
- **VAE v2**: `phytomer_vae_v2/` (NUM_SLOTS 10 — repro x4 to stop the 3.29%
  3-repro overflow loss, VAE input 240D, latent still 64D). val recon 0.0263,
  cls 100%. Prior accepted `phytomer_vae_xml/` (8-slot) kept for lineage.
- **Assembly rules verified** (2026-09-10): stem base = −fwd·L (internode tip
  = node), leaflets 0.8/0.8/1.0 petiole curve, flowers/fruit @ peduncle tip.
  `ORGAN_LEAF=5` restored to `DETERMINISTIC_ORGAN_TYPES` (leaflet-center bug).
- **Pipeline refactored** (see §4.6 of the 2026-09-09 doc):
  `generate_tensor_shards.py` → `generate_cache.py` with `cache`/`pkt` modes;
  `phytomer_ids` + packet targets (+ VAE latent) are now produced **inside**
  dataset generation. `add_phytomer_ids_to_cache.py` and the standalone
  `precompute_phytomer_packets*.py` tools were deleted.
- **tools/ cleaned** (2026-09-10): only active scripts remain —
  `phytomer_vae_visualizer.py`, `precompute_phytomer_latent_pca.py`,
  `calibrate_anchor_capacity.py`. One-off eval/diag/relabel tools deleted.

### Resume (recommended order):
```bash
# 1. (done) packet cache complete — skip backfill

# 2. Relaunch phytomer training with the pkt fast path
PHYTOMER_VAE_CHECKPOINT=diffusion_based/checkpoints/phytomer_vae_v2/phytomer_vae_64d_best.pt \
  PKT_CACHE_DIR=dataset/cache/cowpea_curv26_pkt \
  sbatch slurm_scripts/train_hierarchical_flow_matching.sh
```

---

## 5. Next Steps (Priority Order)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P1** | ~~Finish pkt backfill~~ | **DONE** — 100k/100k v2 10-slot (2026-09-10) |
| **P2** | Relaunch phytomer training | `PHYTOMER_VAE_CHECKPOINT=.../phytomer_vae_v2/phytomer_vae_64d_best.pt PKT_CACHE_DIR=dataset/cache/cowpea_curv26_pkt sbatch slurm_scripts/train_hierarchical_flow_matching.sh` |
| **P3** | Monitor 6D rotation convergence in Stage 3 | Check whether stem segments align upward/outward in Epoch 25/50 panels |
| **P4** | Evaluate Bidirectional Chamfer Distance | Add max/mean distance from GT $\to$ Pred to avoid one-way clustering metric bias |
| **P5** | Dataset scaling beyond 100k | 100k XML+cache complete; future crops/DAPs use the unified `generate_helios_dataset_jobs.sh` pipeline |
| **P6** | PhytomerVAE-64 validation gate | **PASSED** — v2 10-slot val recon 0.0263, cls 100.0% |
| **P7** | Stage-3 phytomer integration | **WIRED** — `flow_granularity=phytomer` (73D bridge targets); slots_per_anchor=10 |
| **P8** | PhytomerVAE GUI app | **DONE + hardened** — `tools/phytomer_vae_visualizer.py` (see GUI doc notes 8–16) |
| **P9** | Launch phytomer-mode training | **PENDING** — VAE retrain NOT needed (base rules are decode-side; see GUI doc note 15) |

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
│   │   ├── part_array_dataset.py                 ← [CRITICAL] cache-first filtering, 16-ch pyramid, pkt passthrough
│   │   ├── generate_cache.py                     ← [NEW 2026-09-09] unified cache/pkt generator (phytomer_ids + packets + latent)
│   │   └── phytomer_packets.py                   ← [NEW 2026-09-09] canonical 8-slot packet builder (relative rotations, Option-1 reference)
│   ├── eval/
│   │   └── eval_hierarchical_self_consistency.py ← 7-column diagnostic panel generator
│   └── checkpoints/
│       ├── hierarchical_latent_fm/               ← training checkpoints (epoch_025.pt, epoch_050.pt, ...)
│       ├── organ_vae/organ_latent_vae_best.pt    ← frozen OrganLatentVAE bridge
│       └── phytomer_vae_v2/                      ← [ACCEPTED] PhytomerVAE-64 v2, 10-slot (val recon 0.0263, cls 100%); lineage: phytomer_vae_xml/
├── slurm_scripts/
│   ├── train_hierarchical_flow_matching.sh       ← launcher w/ INIT_CHECKPOINT / PKT_CACHE_DIR support
│   ├── generate_helios_dataset_jobs.sh           ← full pipeline: XML synth + cache (+ pkt/latent) in one pass
│   └── generate_phytomer_packets_jobs.sh         ← XML-direct pkt backfill for existing image caches
├── dataset/
│   ├── helios_data/cowpea/                       ← 100,000 XML files (complete)
│   ├── cache/cowpea_curv26/                      ← 100,000 cached .pt (image+nodes+phytomer_ids, complete)
│   └── cache/cowpea_curv26_pkt/                  ← 100,000 v2 10-slot pkt targets (latent; v1 8-slot in cowpea_curv26_pkt_v1_8slot/)
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
