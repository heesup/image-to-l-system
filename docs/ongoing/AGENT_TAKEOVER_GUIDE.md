# Agent Takeover & Engineering Handover Guide
**Project: Image-to-L-System / 3D Inverse Procedural Plant Reconstruction**  
**Last Updated:** 2026-09-10 PDT night (gradient explosion .detach() bug fixed, architecture comparison documented, Job 38235969 running with recovery mode — epoch 3 re-explosion under investigation)  
**Primary Author/Agent:** Antigravity Autonomous Agent (Pair programming with Heesup Yun)  
**Environment:** Linux, Python 3.10+, Mamba (`mamba activate digital-crops`), CUDA, PyTorch, `nvdiffrast`, Helios C++ OptiX Raytracer.  

---

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
          → anchor pos (3D), rot (6D), scale (3D), existence logits (1D)
          → supervised by loss_anchor_pos, loss_anchor_rot, loss_anchor_scale, loss_anchor_exist
       ↓
[Stage 3] PhytomerFlowMatchingDecoder (Conditioned on Stage 2 Scaffold & Image)
          → Flow Vector: 64D Phytomer VAE Latent ONLY
          → Prior: Standard Gaussian z_0 ~ N(0, I_64) (No bridge coupling → VelLoss stable ~1.0)
          → Conditioning: anchor_pos.detach(), anchor_rot.detach(), anchor_scale.detach(), anchor_features
          → 10 organ slots existence logits (P(organ) = P(anchor) * P(slot|anchor))
       ↓
[Stage 4] PhytomerVAE v3 (frozen) → 64D Latent decoded to 10-organ canonical packet
          → Assembled via denormalize_packet_scales(scale) & apply_ref_for_flow(pos, rot)
          → HeliosPyTorchRenderer (Multi-scale pyramid CHM depth + Soft Dice silhouette)
```

> **Design Decision (2026-09-10)**:
> - 76D Bridge Flow ($z_0 = \text{anchor\_pos} + \epsilon$)의 오차 증폭(`VelLoss: 1050~2000`)을 해소하기 위해 **하이브리드 디커플링**으로 완전 전환.
> - Stage 2는 전신 3D 골격(`pos`, `rot`, `scale`, `exist`)을 전담하고, Stage 3은 표준 정규분포에서 파이토머 내부 64D 잠재공간 Flow Matching을 전담.
> - 손실 함수 10개 $\to$ 8개 정예화 (`loss_cos`, `loss_dap` 제거, `loss_anchor_rot` 정규 지도 추가).

### Milestone History:
- ✅ **Phase 1** (ICP / Diff Render / Flow Matching benchmark): Complete.
- ✅ **Phase 2** (Over-allocation topology, variable organ counts): Complete.
- ✅ **Option B: 500-Epoch 16D Latent Hierarchical FM** (Job `38143585`): 55.1% mean IoU, peak 67.9%, 2.49cm height error, 0 ghost organs.
- ✅ **3-Stage Cascaded Architecture** (Current, Job `38146809`): Running past Epoch 54 with idle slot damping, scaled global batch size (192), and accelerated anchor convergence.

---

## 2. Active SLURM Jobs (as of 2026-09-10 ~21:50 PDT)

| Job ID | Name | Status | Node | Notes |
| :--- | :---: | :---: | :--- | :--- |
| **38230613** | `ondemand/sys/dashboa` | RUNNING | `gpu-5-58` | User's interactive OnDemand desktop — **DO NOT CANCEL** |
| **38236720** | `hierarchical_fm` | **RUNNING** | `gpu-10-50` | 4x RTX 6000 Ada, **하이브리드 디커플링 + GPU Batch Greedy Matcher 적용** |
| 38236714 | `hierarchical_fm` | CANCELLED | `gpu-10-54` | Stage별 로그 포맷 확인 후 GPU Greedy Matcher 적용을 위해 재시작 |
| 38236699 | `hierarchical_fm` | CANCELLED | `gpu-10-50` | 하이브리드 디커플링 정상 가동 검증 후 재시작 |
| 38234682 | `hierarchical_fm` | CANCELLED | `gpu-10-50` | 이전 시도; 38235936에서 그라디언트 폭발 확인 후 취소 |
| 38224489..38233914 | `hierarchical_fm` | FAILED | — | 이전 launch 실패들 (아래 표 참조) |

### Failed-launch forensics (2026-09-10 오전)
| Job | Failure | Root cause | Fix commit |
| :--- | :--- | :--- | :--- |
| 38224489 | `UnboundLocalError: _sync` | Probe path referenced helper before def | `db4e493` |
| 38225532 | `UnboundLocalError: prof` | Same class, `prof`/`_t_fbs` before def | `3bdae4a` |
| 38226665 | `AsStridedBackward0` version conflict `[351,10,3]` | Packet-scale view modified inplace across fwd/bwd | `69ce959` |
| 38233491 | DDP `.color_palette` marked ready twice | `color_palette` was `nn.Parameter` outside `forward()` | `9dd45be` (`register_buffer` + `.detach()`) |
| 38233914 | `[256, 3]` version conflict in `canonical_rays` | DDP `broadcast_buffers=True` modified ray buffer inplace | `5255efa` (`clone()` + `broadcast_buffers=False`) |

### 그라디언트 폭발 근본 해결 (2026-09-10 야간)
| 원인 | 증상 | 해결책 (하이브리드 디커플링) | 상태 |
| :--- | :--- | :--- | :---: |
| **76D Bridge Flow Coupling** ($z_0 = \text{anchor\_pos} + \epsilon$) | Stage 2 초기 오차가 $v^* = z_1 - z_0$에 직접 주입되어 VelLoss가 1000~2000대로 폭발, grad_norm inf 유발 | Stage 3을 **순수 64D VAE Latent Flow Matching**으로 분리하고 Prior를 표준 정규분포 $z_0 \sim \mathcal{N}(0, I_{64})$로 복원. Stage 2의 3D 뼈대(pos, rot, scale)는 conditioning으로만 주입 | ✅ **해결됨** |
| **rot_head 무지도** | Stage 2의 6D 회전 예측값에 직접적인 지도 손실이 없어 회전 발산 가능성 | `loss_anchor_rot` (Smooth L1, weight 1.0) 신규 추가 | ✅ **반영됨** |
| **중복/노이즈 손실** | CAD-실사 색상 불일치(`loss_cos`), DAP 중복(`loss_dap`) | `loss_cos`, `loss_dap` 제거하여 8대 정예 손실 체계 확립 | ✅ **반영됨** |

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
diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt  (1.4 GB, old organ-mode run)
diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_050.pt  (1.4 GB, old organ-mode run)
diffusion_based/checkpoints/phytomer_vae_v3/phytomer_vae_64d_best.pt  (ACCEPTED v3: 10-slot, 240D in, NORMALIZED scales, val recon 0.070, cls 100%)
```

### Current state (2026-09-10 evening) — v3 stack is the default:
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
| **P1** | Verify AsStrided fix (`69ce959`) with a surviving training job | Submit per §4; watch first 3 min past `probe_optimal_batch_size`. If it re-fails, use `--detect_anomaly` (staged, uncommitted) to get the exact op |
| **P2** | Commit `--detect_anomaly` flag + doc updates | `train_hierarchical_flow_matching.py` has 5 uncommitted lines |
| **P3** | Epoch-1 sanity: loss ↓, Pred count ~50 (bias-init), ClsAcc rising | Check `hierarchical_self_consistency_epoch_001/002.png` panels; Reference column must no longer be N/A |
| **P4** | Monitor 6D rotation + s_a convergence in Stage 3 | Panels epoch 25/50; s_a (petiole len) should track DAP growth |
| **P5** | Evaluate Bidirectional Chamfer Distance | Add max/mean distance from GT→Pred to avoid one-way clustering metric bias |
| **P6** | Backbone A/B (DINOv2-scale vs frozen arms) | `slurm_scripts/submit_backbone_ablation.sh` — only after single-arm training is stable |
| **P7** | Dataset scaling beyond 100k | 100k XML+cache+pkt complete; future crops/DAPs use `generate_helios_dataset_jobs.sh` |

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
│   │   ├── hierarchical_part_flow_matching.py    ← [CRITICAL] 3-stage model, PhytomerFlowMatchingDecoder (76D flow: pos|rot|s_a|latent), scale_head, dap_embed, color_palette, image_tokens kwarg
│   │   ├── dinov2_ray_encoder.py                 ← DINOv2 + PETR 3D ray PE; canonical_rays buffer @ line ~130/161 (see inplace gotcha §8.7)
│   │   ├── plant_organ_array.py                  ← 14D constants, organ types, XML roundtrip
│   │   ├── helios_pytorch_geometry.py            ← mesh builder + differentiable mapping (color_palette kwarg)
│   │   ├── helios_pytorch_renderer.py            ← nvdiffrast renderer + multi-scale pyramid
│   │   ├── part_tensor_to_40d.py                 ← closed-form IK + XML assembler
│   │   ├── organ_latent_vae.py                   ← frozen OrganLatentVAE bridge (16D/organ)
│   │   ├── phytomer_vae.py                       ← PhytomerVAE (240D in / 64D latent; normalizes scales in pack_input+compute_loss)
│   │   └── hierarchical_part_flow_matching.py    ← [CRITICAL] 3-stage model + PhytomerFlowMatchingDecoder (flow_granularity)
│   ├── training/
│   │   ├── train_hierarchical_flow_matching.py   ← [CRITICAL] main loop: --flow-granularity phytomer, 76D bridge, probe reuse, --render_grad_start_epoch, --scale_weight, --detect_anomaly (uncommitted), prof timers
│   │   ├── train_phytomer_vae.py                 ← PhytomerVAE trainer (canonical packets)
│   │   ├── hierarchical_hungarian_matcher.py     ← [CRITICAL] coarse anchor + local fine bipartite matching
│   │   └── flow_matching.py                      ← Rectified Flow scheduler
│   ├── dataset/
│   │   ├── part_array_dataset.py                 ← [CRITICAL] cache-first filtering, 16-ch pyramid, pkt v3 gate
│   │   ├── generate_cache.py                     ← unified cache/pkt generator (PKT_VERSION=3 stamp + skip gate)
│   │   └── phytomer_packets.py                   ← 10-slot packet builder; anchor_scale, no-inplace normalize/denormalize (commit 69ce959)
│   ├── eval/
│   │   └── eval_hierarchical_self_consistency.py ← 7-column diagnostic panel generator (DAP-spread samples, jpeg/prefix preserved)
│   └── checkpoints/
│       ├── hierarchical_latent_fm/               ← training checkpoints (old organ-mode epoch_025/050.pt)
│       ├── organ_vae/organ_latent_vae_best.pt    ← frozen OrganLatentVAE bridge
│       └── phytomer_vae_v3/                      ← [ACCEPTED DEFAULT] PhytomerVAE-64 v3, 10-slot normalized (val recon 0.070, cls 100%); lineage: phytomer_vae_v2/, phytomer_vae_xml/
├── slurm_scripts/
│   ├── train_hierarchical_flow_matching.sh       ← launcher; defaults: VAE v3, SLOTS_PER_ANCHOR=10, BACKBONE_LR_RATIO=0.3, PHY_COUNT_WEIGHT=2.0, RENDER_GRAD_START_EPOCH=5, SCALE_WEIGHT=2.0
│   ├── generate_helios_dataset_jobs.sh           ← full pipeline: XML synth + cache (+ pkt/latent) in one pass
│   ├── generate_phytomer_packets_jobs.sh         ← XML-direct pkt backfill (GPU default, --vae-checkpoint)
│   └── submit_backbone_ablation.sh               ← A/B dispatcher (DAP-spread arms, sequential chain)
├── dataset/
│   ├── helios_data/cowpea/                       ← 100,000 XML files (complete)
│   ├── cache/cowpea_curv26/                      ← 100,000 cached .pt (image+nodes+phytomer_ids, complete)
│   ├── cache/cowpea_curv26_pkt/                  ← 100,000 pkt v3 (10-slot, absolute packets + normalized latent, pkt_version: 3)
│   └── cache/cowpea_curv26_subset4k/             ← 4,000 symlinks (40/DAP × 100 DAP) for fast smoke tests
└── docs/
    ├── ongoing/
    │   ├── README.md                             ← ongoing status dashboard
    │   └── AGENT_TAKEOVER_GUIDE.md               ← [THIS FILE] master handoff
    ├── results/
    │   ├── 20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md ← Epoch 50 geometry report
    │   └── assets/                               ← diagnostic panels
```

---

## 7. Training Loss Progression (Job 38146809 — OLD organ-mode run, for reference)

> **Note**: the table below is from the superseded organ-slot run (2026-09-08).
> No phytomer-mode training run has survived past startup yet (see §2 forensics).
> Populate a new table from the first successful phytomer run.

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
6. **OnDemand Desktop (Job 38230613)**: Running on `gpu-5-58` — do not touch or kill. (Old ID 38147219 may appear in older logs.)
7. **Inplace-autograd traps (TODAY'S PAIN)**: Three distinct failures came from autograd graph issues, not model logic:
   - Packet-scale `as_strided` views (`[P,10,3]`) must not be modified inplace across fwd→bwd — fixed via no-inplace `normalize/denormalize_packet_scales` (`69ce959`).
   - Probe-token reuse (`image_tokens` kwarg) can surface DDP "mark a variable ready only once" if any parameter is used outside a single forward — if it re-appears, check that the probe forward and the model forward don't both touch the same params in separate graphs on the same step.
   - Encoder-alone repro (DINOv2RayEncoder fwd/bwd in isolation) is CLEAN — `canonical_rays` buffer version stays 0; do not chase that buffer without reproducing in the training loop first.
   - Debug with `--detect_anomaly` (staged flag) on 1 GPU, small `--max_train_samples`.
8. **pkt cache version gate**: `PartArrayDataset` and `generate_cache.py` skip pkt files whose `pkt_version != 3` (fall back to on-the-fly packet building — slow but correct). If you change packet format, bump `PKT_VERSION` and regenerate; never edit files in place.
9. **VAE v3 recon 0.070 vs v2 0.0263 is NOT a regression**: v3 targets are scale-normalized (wider distribution). Do not "fix" by switching back to v2 — it is 8-slot-incompatible and unnormalized.

---

## 9. Milestone Documents (Chronological)

| File | Date | Summary |
| :--- | :--- | :--- |
| [`docs/ongoing/20260909_phytomer_latent_and_local_matching.md`](../ongoing/20260909_phytomer_latent_and_local_matching.md) | 2026-09-09/10 | **[MASTER ENGINEERING LOG]** §4.6–4.9: pipeline refactor, backbone A/B, fruit fix, v3 scale-normalized packets + 76D flow, render-pipeline timing |
| [`docs/results/20260907_latent_hierarchical_flow_matching_500epoch_report.md`](../results/20260907_latent_hierarchical_flow_matching_500epoch_report.md) | 2026-09-07 | Option B 500-epoch report: 55.1% IoU, 2.49cm height error |
| [`docs/results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md`](../results/20260908_3d_spatial_vision_and_hierarchical_reconstruction_milestone.md) | 2026-09-08 | Epoch 150 spatial vision breakthrough: 49.2% mean IoU, 2.6cm RMSE |
| [`docs/results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md`](../results/20260908_3stage_cascaded_flow_matching_scaling_milestone.md) | 2026-09-08 | 3-stage cascaded architecture scaling: batch 192, dormant slot damping |
| [`docs/results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md`](../results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md) | 2026-09-08 | **[NEW]** Root cause analysis of Epoch 50 skeleton geometry vs AncPos loss |
