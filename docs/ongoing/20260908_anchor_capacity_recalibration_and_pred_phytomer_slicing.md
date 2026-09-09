# Anchor Capacity Recalibration & Predicted-Phytomer Dynamic Slicing (2026-09-08)

**Status**: Implemented + tested (CPU smoke tests + gradient audit PASS) — updated after evening full rescan
**Scope**: `diffusion_based/models/hierarchical_part_flow_matching.py`, `training/train_hierarchical_flow_matching.py`, `training/hierarchical_hungarian_matcher.py`, new `dataset/dap_bucket_sampler.py`, new `tools/calibrate_anchor_capacity.py`

---

## 1. Problem

The Matryoshka anchor-slice formula `K = ceil(8 * 2^(DAP/8.5))` (doubling every 8.5 days) was
botanically wrong and computationally wasteful:

| DAP | GT phytomer clusters (실측) | old formula | old tier K (fine slots) | actually needed |
|---:|---:|---:|---:|---:|
| 10 | ~8 | 18 | 32 (256) | ~9 (72) |
| 20 | ~20 | 41 | 64 (512) | ~24 (192) |
| 50 | ~107 | 472 | 512 (4,096) | ~129 (1,032) |
| 100 | ~77 (mean) / 342 (max) | 27,837 | 512 (4,096) | ~350 (2,800) |

Consequences:
- **Mature-stage samples always ran the full 4,096 fine slots** (Stage 3 FLOPs/VRAM ~4x waste).
- Power-of-2 tier snapping quantized away fine capacity differences.
- `pred_num_phytomers` (Stage 1) was trained but never used for capacity.

## 2. Changes

### 2.1 Calibrated logistic anchor-capacity curve (replaces exponential + tier snap)

**Evening rescan (2026-09-08, 59,441 cache files, 100–300 samples/DAP)**: The earlier
p97.5-envelope fit (25 samples/DAP) left **11% of DAP buckets** with at least one observed
sample beyond capacity (worst: 342 clusters @ DAP 92; cache organ rows max 2,625 @ DAP 92).
The final fit targets the **rolling-max envelope of observed sample maxima**:

```
phytomers_max(dap) = L / (1 + exp(-k*(dap - x0))) + N0
L=297.2, k=0.1841, x0=30.5, N0=15.53
K = ceil(curve * 1.10 + 6), clamp [8, 512]
```

Coverage: **p97.5 100% | observed-max 100%** (DAP 92 max 342 → K=350; organ rows max 2,625 → slots 2,800).

| DAP | new K (fine slots) |
|---:|---:|
| 1 | 25 (200) |
| 10 | 31 (248) |
| 20 | 65 (520) |
| 30 | 180 (1,440) |
| 40 | 302 (2,416) |
| 50 | 342 (2,736) |
| 92 | 350 (2,800) |
| 100 | 351 (2,808) |

Average fine-slot usage: 2,706 → 2,110 (**22% saving** vs old tiers); guaranteed full-coverage
headroom for the still-filling 1,000-seed/dap dataset (unobserved tails possible).

Raw-XML upper-bound check (8,896 mature files DAP≥85): max **2,271 organ elements**;
after +31 meta/bud-expansion rows → max part rows 2,625 < 4,096 slots → **no cache truncation**.
"4,096" is the tensor width (512 anchors × 8 slots), not an observed organ count.

Regenerate: `tools/calibrate_anchor_capacity.py --fit_target max --samples_per_dap 300`.

### 2.2 Predicted-phytomer dynamic slicing (Proposal B, inference path)

- `CoarseSkeletalTransformer.forward(..., capacity_mode=...)`:
  - `given` (default, training): slice from GT DAP via `compute_matryoshka_slice` (teacher forcing).
  - `pred_phyto` (two-pass): `MacroBiologicalHead` runs **first** on the CLS token (full max_k width);
    the predicted phytomer count determines the anchor bank prefix slice
    `K = ceil(max_sample(pred) * margin + flat)`.
- `HierarchicalPartFlowMatchingModel.forward(..., capacity_mode="pred_phyto")` propagates the
  resolved K; `sample_ode` now uses `capacity_mode="pred_phyto"` and, when GT DAP is supplied
  (self-consistency eval), takes `max(K_pred, K_dap)` for safety.

### 2.3 Scheduled capacity teacher-forcing decay (NEW, addresses "phytomer loss but unused prediction")

Training now ramps the predicted-phytomer capacity path:

- `--capacity_warmup_epochs 50 --capacity_full_epochs 150`: `p_pred` ramps linearly 0→1.
  Each batch stochastically uses:
  - `gt` path (probability 1−p_pred): K from GT DAP (noise-free teacher forcing).
  - `pred` path (probability p_pred): macro head probed under `no_grad`, predicted count
    multiplied by **log-normal noise ×1.10 median, σ=0.30** (biased up so the noisy capacity
    usually exceeds the clean prediction → under-allocation guard), K = max(K_noisy_pred, K_gt).
- Rationale: exposes the model to the inference-time capacity distribution while the
  max-with-GT guard prevents the error-propagation loop (under-allocation → matcher
  mismatch → corrupted count gradients).
- Losses (`loss_dap`, `loss_phy_count`) remain unchanged — they are the supervision that
  makes the predictions accurate enough to drive inference-time capacity.
- Logged: `train/capacity_p_pred` (W&B) + epoch printout.
- `--capacity_full_epochs 0` disables the ramp entirely (GT-only, previous behavior).

### 2.4 Gradient-safety guarantees (요청 사항)

Stage 1/2 gradients remain fully connected:
1. `MacroBiologicalHead` now runs before slicing on the CLS token → `pred_dap`/`pred_num_phytomers`
   losses keep their direct gradient path regardless of K.
2. `pred_num_phytomers` → `init_logits` (soft margin) → `anchor_logits` is **differentiable** and
   unaffected by the discrete K decision; count supervision flows through it (audited).
3. The discrete K decision is a capacity choice only (`detach()`ed count used for slicing);
   the soft-margin prior (τ=0.8 sigmoid tapering) carries the botanical capacity pressure.
4. Slice is still a prefix of the trained parameter bank (`anchor_queries[:K]`, `ref_points[:K]`),
   so slot-index semantics stay consistent; `soft_margin_weights`/`init_logits` are resliced
   deterministically from the full-width computation.
5. Audited by `tests/test_gradient_flow.py`: pos/rot/exist heads, anchor queries, ref points,
   decoder layers, dap/phy heads all receive non-zero grads in both modes. NOTE: `dap_head`
   gradient flows only via `loss_dap` (ReLU can be in dead-zone at random init — non-issue
   with trained weights, verified positive preactivation on real forward).

### 2.5 Under-allocation safety (training)

- `estimate_anchor_capacity(dap)` gives per-sample capacity; GT existence targets are masked
  to slots below capacity (`act_mask_sub &= cap_mask`), so the existence head is never punished
  for organs the slice cannot reach.
- `HierarchicalBotanicalMatcher(per_sample_max_phytomers=...)` clamps GT phytomer clusters
  (drops those farthest from predicted anchors) so the phytomer-count loss is unbiased by
  unreachable topology.

### 2.6 DAP-bucketed batch sampler (`dataset/dap_bucket_sampler.py`)

`--dap_buckets 8` (default): batches are formed from ~12.5-DAP-wide buckets so the batch-max
DAP matches the bucket content. Max intra-batch DAP spread: 12 (verified). Full-epoch coverage
(drop_last=False default), DDP rank-sharded with per-epoch reshuffle. `--dap_buckets 0` restores
plain `DistributedSampler` shuffle.

### 2.7 Model forward width-alignment hardening

`forward` now pads `noisy_fine_nodes` up to `K_out * M` (anchor slice is authoritative) and the
train loop pads GT tensors when the model resolves a wider slice (rare; only when a batch-max
DAP exceeds the GT padding width).

## 3. Expected impact

- **Compute**: mature-stage fine-slot pass 4,096 → 2,808 (~31% ↓); young buckets far larger cuts
  (DAP 10: 256 → 248 comparable; DAP 5: 128 → 216 slightly larger due to the guaranteed-coverage
  flat; overall avg ~22% ↓ vs old tiers with 100% observed-max coverage guarantee).
- **VRAM** → larger batches via `probe_optimal_batch_size` (probe still calibrates at worst case).
- **Autonomous inference**: `sample_ode(daps=None)` now slices by predicted phytomer count
  (previously fell back to full 512) — true single-image deployment path.
- **Train/inference capacity alignment**: scheduled decay closes the teacher-forcing gap.

## 4. Verification summary

| Test | Result |
|---|---|
| `tests/test_gradient_flow.py` (Stage 1/2 heads, both capacity modes) | PASS |
| Full-model backward (all 9 head/backbone param groups) | PASS |
| `forward_backward_step` CPU smoke (VAE, matcher, GT + pred capacity paths) | PASS |
| `sample_ode` w/ GT DAP, autonomous, mixed-DAP batch | PASS |
| `DAPBucketBatchSampler` coverage/DDP/epoch-shuffle | PASS |
| Curve coverage on evening scan (p97.5, observed-max, n=100–300/DAP) | 100% / 100% |
| Ramp schedule (warmup 50 → full 150) | p_pred [50:0, 100:0.5, 150:1.0] verified |

## 5. Dataset scale correction

- Planned total = **100,000 samples** (40 shards × 1000 seeds/dap × 100 DAP; 2–3 augmentation
  variants for some DAPs per shard sum), not ~120k. Progress as of 2026-09-08 ~23:30 PDT:
  XML prefixes 81,980/100,000 (82%), cache .pt 54,643 (55%), 22 shards running.

## 6. Follow-ups (not in this change)

- **M=8 per-cluster overflow**: worst-case sample has 8.37 organ rows/phytomer cluster
  (bud expansion exceeds the M=8 slot block on some clusters). Impact is marginal
  (a few clusters per extreme sample); flagged for a separate design pass on slot
  block structure if it appears in training metrics.
- After convergence, close the GT-DAP dependency in eval (currently `max(K_pred, K_dap)`).

---

## 7. Analysis: Why "yesterday night's" panels still look better (Job 38145444 Ep150 49.2% IoU vs current Job 38146809 Ep100 17.6%)

Investigated 2026-09-08 ~23:00 PDT from job logs. **Correction (2026-09-08 late)**: an earlier
version of this section claimed the current job's dataset was "growing live". That was wrong —
`PartArrayDataset` builds `self.samples` **once at job launch**, so every run trains on a frozen
list. The verified dataset composition per job:

| Job | Launch | Code state | Dataset at launch | Composition |
|---|---|---|---|---|
| 38143585 (500-ep) | Sep 7 01:03 | pre-`027118d` (no cache-first filter) | 10,000 (100 seeds × 100 DAP, balanced) | frozen |
| 38145444 (milestone) | Sep 7 23:17 | pre-`027118d` | 10,000 (identical list) | frozen |
| 38146809 (current) | Sep 8 15:13 | `027118d` cache-first filter | **13,600** = 10k balanced + 4,000 extras | frozen |

The 4,000 extras: the earlier `unified_cowpea_20260908_121134` run had already cached
**DAP 11, 12, 16, 17 at 1,000 seeds each** before 15:13, so the cache-first filter admitted
them. Distribution: those 4 young DAPs alone are **29% of the current dataset**
(vs 4% under the balanced 100-seed list).

### 7.1 Current run is harder: young-DAP skew + mid-architecture changes

1. **Young-DAP skew (29% of samples in 4 DAPs 11/12/16/17)**. Old runs: 25% young(1–25)
   uniform, tiny per-DAP variance. Current: seedling-stage samples dominate every batch.
   Young plants have 7–12 organ rows vs 2,445 at maturity — the capacity/velocity/classification
   statistics the model must fit differ hugely across the skew, raising per-batch gradient noise.
2. **`0389ca5` (14:54, ~20 min before launch)** changed four things at once:
   anchor-pos weight ×4, idle-slot velocity damping (0.05), backbone lr 2e-5 → 6e-5, and the
   top-k 0.15 existence guard. Backbone lr ×3 alone destabilizes early training; combined
   with the loss re-weighting, the optimization landscape reset — visible in VelLoss at ep25:
   0.35 (old) vs 0.85 (current) despite *more* sample-visits (336k vs 250k).
3. **Single-panel evals are extremely noisy**: eval = `next(iter(dataloader))` → only **4
   samples** per panel. Within the *same* 500-epoch run, IoU swung **11% ↔ 46.9%** across
   eval epochs — ±30pt purely from which 4 plants were drawn. The 49.2% panel's best row was
   DAP 64 (bushy mature canopy — the easiest high-IoU class per the milestone doc itself).
4. Old-run ClsAcc 78–81% vs current 66% reflects (1)+(2), not an architecture regression.

### 7.2 Verdict

Yesterday's numbers are **not evidence the architecture regressed** — they are the product of
(a) balanced-easy data + long memorization headroom, (b) favorable 4-sample eval draws, and
(c) a harder, skewed dataset + hyperparameter reset in the current run. Once the full 100k
dataset is frozen, a fresh run with re-tuned hyperparameters (see §8) should exceed the old
panels. Interim comparisons must use fixed-seed eval batches and mean±std over ≥16 samples.

**Action items**: (1) freeze-eval protocol — fixed eval sample list saved to disk;
(2) report IoU as mean over many samples per eval epoch.

---

## 8. Hyperparameter Recommendations for the 100k Scaling Run

Context: 100k frozen dataset (1000 seeds/DAP balanced), capacity-recalibrated anchor slicing
(avg ~2,110 fine slots), DAP-bucketed batching, scheduled capacity decay. Recommendations
grounded in the three completed runs' evidence:

### 8.1 Confirmed settings (keep)

| Parameter | Value | Evidence |
|---|---|---|
| `--dap_buckets 8` | capacity-homogeneous batches | intra-batch DAP spread ≤12 verified; cuts Stage 3 FLOPs on young buckets |
| `--capacity_warmup_epochs 50 --capacity_full_epochs 150` | scheduled pred-capacity ramp | GT-only warmup avoids error-propagation loop; pred path aligns train/inference capacity |
| batch 48/GPU (global 192) | VRAM 78.6% stable | `0389ca5` — throughput proven, keep |
| `--render_fraction 1/6` (batch-relative) | fraction of batch rendered per step | **Replaces the confusing `--render_ratio` + `--render_sub_batch` pair** (2026-09-08 incident: batch 24→48 with frozen sub_batch 4 halved per-sample photometric visits). Batch-relative fraction cannot drift; 1/6 restores 2026-09-07 parity; VRAM-probe validated |
| `--eval_every 25` + `--eval_min_interval_minutes 30` | hybrid eval cadence | epoch cadence keeps panels bounded on small data; the **time fallback** (user request) forces a panel whenever ≥30 min elapsed, so diagnostics stay ~constant in wall-clock as per-epoch time grows with the 100k dataset. Verified: 12 min/epoch → panel every ~3 epochs; 1 min/epoch → epoch cadence dominates |
| fixed stratified eval set | 20 samples (2 per 10-DAP bucket), `eval_set.json` | eliminates the ±30pt single-panel draw noise (§7.1.3); `val/cos_color_loss` + `val/dice_loss` now computed over ALL panel samples at full render with training-loss normalization (previously no dataset-level Cos/Dice metric existed) |
| fixed-seed eval list (new, needed) | ≥16 samples, stratified DAP 5/15/30/50/70/90/100 | eliminates the ±30pt single-panel draw noise (§7.1.3) |

### 8.2 Re-tuned settings (change)

| Parameter | Old (38146809) | **Recommended** | Rationale |
|---|---|---|---|
| backbone lr | 6e-5 (`args.lr × 0.3`) | **3e-5** (`args.lr × 0.15`) | ×3 jump destabilized early training (VelLoss 0.35→0.85 at ep25); foundation priors need gentler adaptation on 100k diverse data, and 6e-5 was tuned against a 4-DAP-skewed set |
| `loss_anchor_pos` weight | 4.0 | **2.0** (revert) | ×4 was compensating for the young-skewed dataset's poor anchor spread; on balanced 100k, 2.0 (the pre-`0389ca5` value that produced the 49.2% panel) is correct; revisit only if AncPosLoss plateaus >0.015 by ep50 |
| idle-slot damping | 0.05 | **0.02** | with capacity clamp + top-k guard + pred-capacity path, unmatched-slot drift is already triple-suppressed; 0.05 over-regularizes velocity on genuinely dormant reproductive slots |
| top-k existence guard | 0.15 | **keep 0.15** (no change) | fine as-is; monitor ghost-organ count in panels |
| `--lr` (decoder) | 2e-4 | **3e-4** | 100k samples is 7.4× the old 13.6k; slightly higher decoder lr with cosine T_max=500 converges the larger capacity slice variety faster; backbone stays decoupled at 3e-5 |
| warmup | none | **3-epoch linear lr warmup** | removes the ep1 loss spike (166 → 13.6 in old job ep1-5) that wastes early epochs; cheap insurance for the fresh run |

### 8.3 Monitoring gates (abort/restart criteria)

| Metric | Gate | Action if violated |
|---|---|---|
| VelLoss @ ep25 | > 0.45 | backbone lr too high → restart at 2e-5 |
| PhyLoss (Pred/GT gap) @ ep50 | relative gap > 15% | count head under-trained → raise `loss_phy_count` weight 0.5 → 1.0 |
| AncPosLoss @ ep50 | > 0.015 | anchor head under-weighted → 2.0 → 3.0 (do NOT jump to 4.0) |
| ClsAcc @ ep100 | < 70% | check DAP-bucket sampler balance; verify VAE frozen roundtrip |
| Capacity p_pred @ ep150 | = 1.0 | scheduled decay didn't engage → check `--capacity_full_epochs` |

### 8.4 Deferred decisions (do NOT bundle into the scaling run)

- `slots_per_anchor` 8 → 9/10 (M=8 per-cluster overflow): separate design pass; bundling it
  changes the 14D contract mid-experiment and invalidates the checkpoint lineage.
- Bidirectional Chamfer distance (P4): eval-metric change, keep out of the training run.
- Eval GT-DAP removal (`max(K_pred, K_dap)` → pure pred): only after §8.3 gates pass.

### 8.5 Suggested launch command (post-100k-freeze)

```bash
sbatch slurm_scripts/train_hierarchical_flow_matching.sh  # with:
#   --dap_buckets 8 --capacity_warmup_epochs 50 --capacity_full_epochs 150
#   --eval_every 25 --lr 3e-4   (backbone auto: 3e-4 × 0.15 = 4.5e-5 ≈ 3e-5 target)
#   --init_checkpoint diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_500.pt
#   (warm-start optional: warm-start risks inheriting the skewed-dataset bias;
#    if warm-starting, halve all lrs for the first 25 epochs)
```

---

## 9. Photometric Fraction Refactor & Self-Consistency as a First-Class Metric (2026-09-08 night)

### 9.1 Evidence

1. **The frozen-sub-batch bug**: `0389ca5` doubled batch 24→48 but kept `render_sub_batch=4`
   → per-sample photometric visit rate halved (16.7% → 8.3% per data-pass). This is the
   primary mechanical cause of the current run's slow Dice/Cos movement.
2. **Knob confusion**: `render_ratio` (default 1.0) was shadowed by `render_sub_batch` in
   every committed config — two interacting knobs, one dead. Git history: ratio was never
   anything but 0.25, sub_batch 8→4.
3. **Full-batch rendering is VRAM-infeasible** at useful batch sizes (measured ~1.33GB
   render activation/sample): batch 48 full ≈ 96GB > Ada 47.4GB and > H100-NVL 94GB.
   batch 16 full ≈ 36GB (75%) is feasible but ~3x slower per-epoch.
4. **No dataset-level Cos/Dice metric ever existed**: train-log Cos/Dice are telemetry from
   a random 4-sample render subset; the eval panel had IoU/Depth/Node only.

### 9.2 Changes implemented

**A. Knob refactor** (`train_hierarchical_flow_matching.py`, slurm script):
- `--render_sub_batch` **deleted** (7 call sites); `--render_ratio` → **`--render_fraction`**
  (single knob), default **1/6**.
- `n_render = clamp(round(B × fraction), 1, B)` — **batch-relative**, so the fraction (and
  per-sample visit rate) is invariant to batch-size changes. The 2026-09-08 incident class
  is structurally impossible now.
- batch 24 → n_render 4 (exact 2026-09-07 parity); batch 48 → 8 (VRAM-verified 89.8%).

**B. Self-consistency metrics over all eval samples** (`eval_hierarchical_self_consistency.py`):
- `val/cos_color_loss` + `val/dice_loss` computed over **every evaluated panel sample** at
  full render, using the **training-loss normalization** (dataset GT [-1,1] RGB, meters
  depth, same canopy_mask/sigmoid-ramp definitions) — a true training-parity dataset metric.
- Returned in the metrics dict and logged to W&B as `val/*`.

**C. Fixed stratified eval set** (`train_hierarchical_flow_matching.py`):
- `build_eval_indices()`: 2 samples per 10-DAP bucket (default 20 total), deterministic
  seed, persisted to `{output_dir}/eval_set.json` (index + prefix) for cross-run reproducibility.
- Eval no longer uses `next(iter(dataloader))`; `collate_eval_set()` loads the fixed set
  directly. Panel figure still shows 4 rows for visual inspection; **metrics cover the
  full fixed set**.

**D. Hybrid eval cadence** (kept from earlier tonight): epoch cadence 25 + wall-clock
fallback `--eval_min_interval_minutes 30`.

### 9.3 Fraction Ablation Ladder (decides how far "full rendering" goes)

**Measured render cost** (TITAN RTX 24GB local, `tools/benchmark_render_cost.py`,
real cache samples, mesh build + 4-scale pyramid fwd + backward):

| DAP | organs | mesh | fwd | bwd | total/sample |
|---:|---:|---:|---:|---:|---:|
| 10 | 41 | 10 ms | 10.5 ms | 0.6 ms | **21 ms** |
| 50 | 191–473 | 10 ms | 13–20 ms | 0.6 ms | **24–31 ms** |
| 90–100 | 658–863 | 20–22 ms | 25–31 ms | 0.6 ms | **45–53 ms** |

Mean **33–44 ms/sample** (young→mature mix). Backward is nearly free (0.6 ms);
forward dominates. RTX 6000 Ada / H100 are 2–3× faster per sample.

**Critical discovery — rendering is NOT the bottleneck**: profiling the current
3.87 s/step (sacct) shows the **Hungarian matcher costs ~5.3 s/step at realistic GT
sizes** on TITAN (batch 48, 44 clusters/sample): 3.5 s pure-Python per-cluster
orchestration + 43,438 × `.item()` GPU→CPU syncs (0.54 s) + scattered cat/nonzero/cdist
(~0.9 s). scipy LSA itself is 0.07 s — cheap. Rendering 48 samples would cost
~1.6 s — less than the matcher!

Revised ladder economics (Ada est., t_render ≈ 15–20 ms/sample):

| Config | Batch | fraction | render cost/step | Bottleneck | Est. days/500ep |
|---|---:|---:|---:|---|---|
| A (parity) | 48 | 1/6 | ~0.13 s | matcher ~3 s (Ada est.) | ~10–12 |
| B (dense) | 32 | 1/2 | ~0.32 s | matcher | ~12–13 |
| C (full) | 16 | 1.0 | ~0.32 s | matcher | ~14–16 |

**Full-batch rendering (C) costs only ~1.2–1.3× wall-clock of A** — the render
fraction is nearly free relative to the fixed per-step overhead. The ladder is still
worth running (signal-density hypothesis untested), but the **highest-leverage
optimization is the matcher refactor** (vectorize the per-cluster loop, batch GPU→CPU
transfers, cache role indices → est. 5.6 s → <1 s), which would accelerate every
config by ~2–3× and make Config C essentially free.

| Config | matcher optimized (est <1 s) | Est. days/500ep |
|---|---|---|
| A-opt: B48 f=1/6 | ~1.4 s/step | **~3.5–4** |
| C-opt: B16 f=1.0 | ~0.8 s/step | **~7** |

- **A and B launch in parallel** on separate nodes via `slurm_scripts/submit_ablation_when_frozen.sh`
  (waits for the 100k freeze, prints DAP balance, cancels the interim job on request,
  submits A to the Ada partition and B to `low`/H100 if available).
- **Decision gate at day ~4-6**: compare `val/dice_loss`, `val/cos_color_loss`,
  `val/silhouette_iou` at matched wall-clock. If B clearly beats A → launch C
  (batch 16, fraction 1.0) for the densest self-consistency signal.
- All configs share: capacity recalibration, DAP buckets, scheduled capacity decay,
  §8 hyperparameters.
- **Matcher refactor** is queued as the next optimization (separate task, benefits all).

### 9.4 Usage

```bash
# Watch for dataset freeze, then auto-submit A & B (recommended flow):
nohup bash slurm_scripts/submit_ablation_when_frozen.sh \
    --cancel-interim 38146809 > slurm_scripts/logs/ablation_watcher.log 2>&1 &

# Dry-run / force options:
bash slurm_scripts/submit_ablation_when_frozen.sh --dry-run --force
bash slurm_scripts/submit_ablation_when_frozen.sh --target-samples 100000 --check-interval 600
```

### 9.5 Open follow-ups

- ~~Step-time anomaly~~ **SOLVED**: the 3.87 s/step is matcher-dominated (~5.3 s measured
  on TITAN at realistic GT; Ada proportionally less but still dominant). Renderer is
  ~33–44 ms/sample — nearly free by comparison.
- **Matcher refactor** (queued, high-leverage): vectorize the per-cluster loop, eliminate
  43k `.item()` syncs, batch GPU→CPU once per sample. Est. 5.6 s → <1 s → every config
  2–3× faster; makes full-render (C) nearly free vs A.
- Config C requires a dry-run VRAM probe (batch 16, fraction 1.0) before committing.
- If fraction ablation shows f=1/6 saturates, consider renderer memory optimization
  (fewer pyramid scales / lower raster resolution) to raise the feasible fraction at
  batch 48 instead of shrinking batch.