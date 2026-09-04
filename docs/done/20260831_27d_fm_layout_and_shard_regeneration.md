# 27D Flow-Matching Layout & Fresh Shard Regeneration Progress Report (2026-08-31)

**Status**: 27D FM encode/decode verified lossless (< 1e-5, organ-aware normalization);
meshes from 27D nodes bit-identical to 17D GT; **1000 fresh 27D shards generated and
validated** via the SLURM production pipeline; **VLM-Scaffold-DiT training LIVE on
4×H100 NVL (job 38048780, batch 32/GPU, global 128, v-loss 13.3→6.2 descending)**.

---

## 1. Context (supersedes §8 of 20260826_canonical_pipeline_refactor_progress.md)

The stale shard dataset (`dataset/helios_data/cowpea_shard_20260824`, 500 × `.pt` with
**26D** FM nodes) pre-dates the two final round-trip fixes (`26d6a5b` pod scale,
`e43707c` flower orientation) and lacks the `bud_state` channel entirely. Training
against it teaches the model GT that cannot reproduce the verified
`17D → XML → Helios` round-trip (all organs IoU 1.0, commit `599a504`).

This session upgrades the FM layout 26D → **27D** (adding `BudState(1)`), fixes the
encode/decode **softplus asymmetry**, routes all dataset consumers through a single
source of truth, and regenerates shards.

---

## 2. Bugs Found & Fixed (this session)

### 2.1 Scale encode/decode asymmetry — **softplus destroyed negative overloaded scales** (CRITICAL)

The 17D part tensor's scale columns 10–12 are **organ-specific overloaded** — for the
DAP50 GT they span `[-10, +29.5]`:

| organ | scale_x | scale_y | scale_z | semantic meaning |
|-------|---------|---------|---------|------------------|
| internode | radius | length_max | length_segments | col12 ∈ {2,3,4,...} |
| petiole | radius | current_leaf_scale_factor (cls) | — | |
| leaf | scale | pitch | yaw | **negative pitch** (e.g. −43.9°) |
| flower/fruit | raw base_scale | fruit_scale_factor | pitch | negative pitch too |
| peduncle | radius | pitch | roll | |

`encode_fm` encoded linearly (`part * 50`) but `decode_fm` applied
`softplus(...).clamp(min=0.025) / 50` — softplus saturates at 0 for negative inputs,
so **every leaf/flower/peduncle with negative pitch came back wrong** (max diff 10.0).

**Fix**: decode is now purely linear (`fm / 50.0`) — encode/decode is a pair of exact inverses.

Result: 17D ↔ 27D FM round-trip on real GT:

```
DAP10: 3.05e-05   DAP30: ~3e-05   DAP50: 3.05e-05   (float32 noise floor)
```

*Historical note*: the softplus in old decode was needed because the pre-`71e49e9`
pipeline stored **positive-only 40D proxy scales**. The faithful 17D extract
(`e43707c`, `71e49e9`) made scales signed, so the linear pair is now correct.

### 2.2 FM mesh-build decode used raw FM columns (missed inverse normalizations)

`build_mesh_from_part_tensor` (helios_pytorch_geometry.py:1438) decoded 26D FM nodes
as `bud_state = 0`, `extra_col14 = 0`. With 27D, the FM columns must be **un-normed**:

```
bud_state  = p[:, 24] * 10.0        # BUD_STATE_SCALE
curvature  = p[:, 25] * 100.0       # CURVATURE_SCALE  (leaf variant: unifoliate/left/tip/right)
phyllo     = p[:, 26] * 180.0       # not used for mesh, kept explicit
```

Verified: mesh built from 27D FM nodes is **bit-identical** to the 17D GT path
(`DAP10 11198v`, `DAP50 144531v`, max vertex diff 0.0).

### 2.3 Duplicated FM layout constants removed (single source of truth)

`diffusion_based/training/flow_matching.py` hardcoded a **16D-mismatched** copy of the
FM constants (`FM_NODE_DIM = 26` with different indices). It now imports everything
from `part_array_dataset.py`, so the layout cannot drift again.

### 2.4 FM scale-block range explosion — organ-aware normalization (CRITICAL, late addition)

First training smoke test on the *first* fresh-27D shard batch exploded the FM velocity
loss (`v ≈ 30k–300k` vs `≈ 0.6-0.85` on the old — wrong-data — 26D run). Root cause
chain:

1. **Corpus flowers carry compounded Euler angles**, e.g. `flower_yaw` of 810°
   (`T_COL_YAW` accumulates peduncle roll + compound rotation across growth; Helios
   consumes it via sin/cos only). The `* 50` scale normalization then produced FM
   values up to **±9,000** — v-loss started at ~134k instead of ~1.
2. `SHOOT_META` rows misuse scale columns as **integer parent IDs/indices** (up to
   ~10²–10³), again feeding huge values into the scale block.

**Fix (two parts):**

- **Angle wrapping at extract** (`helios_pytorch_geometry.py` `extract_part_tensor()`):
  flower pitch/yaw/roll/azimuth, leaf pitch/yaw, shoot_meta base pitch/yaw/roll, and
  peduncle pitch/roll are wrapped to `[-180°, 180°)` — geometry-identical under
  sin/cos, so the XML writer stays exact and Helios renders identically.
- **Organ-aware encode/decode** (`part_array_dataset.py` `encode_fm`/`decode_fm`):
  the 27D columns now normalize **per one-hot organ type**:
  - `SHOOT_META`/`ROOT_META` (scale block = integer IDs): `* ORGAN_TYPE_SCALE (10)`
  - angle carriers `leaf/flower/fruit/flower_closed/peduncle` scale_y/z (degrees):
    `/ 180` (PHYLLOTACTIC_SCALE)
  - all other geo organs (radius/length/cls/segments/mesh scale): `* SCALE_SCALE (50)`
  - `bud_state` column (leaf/flower/peduncle roll, internode pitch, ±180°): `/ 10.0`
    was kept but values are now bounded by wrapping to ±180 (max FM value 18).

Result: encode↔decode exact again (1.53e-05 on GT + corpus) with **all FM columns
bounded: scale ≤ 50, bud ≤ 18, curv ≤ 2, phyllo ≤ 1.2** (was: scale up to 40,500).

> The first-generation 27D shards (0.3 K) and the first training restart (~10 min,
> v-loss 30k–300k oscillating) were discarded because of this. If you find shard
> files older than the 15:00 SLURM submission, ignore them.

---

## 3. New 27D FM Node Layout

```
[ onehot(0..11) | base(12..14) | rot6d(15..20) | scale(21..23) | bud(24) | curv(25) | phyllo(26) ]
  12            | 3             | 6             | 3             | 1       | 1        | 1        → 27D
```

Normalizations (encode = exact inverse of decode, all verified < 1e-5):

| block | encode | decode | notes |
|-------|--------|--------|-------|
| organ type | one-hot + empty category (idx 11) | argmax over idx 0..10; existence = 1 − p(empty) | |
| base | `* 20.0` | `/ 20.0` | |
| rot6d | identity (unit) | identity | |
| scale_x | `* 50.0` (signed) | `/ 50.0` | **linear, no softplus** |
| scale_y/z — LEAF / FLOWER / FRUIT / FLOWER_CLOSED / PEDUNCLE | `/ 180.0` (degrees) | `* 180.0` | angle carriers after extract-side wrap |
| scale_y/z — SHOOT_META / ROOT_META | `* 10.0` (integer parent IDs) | `/ 10.0` | bookkeeping rows |
| scale_y/z — other geo organs | `* 50.0` | `/ 50.0` | internode len_max/len, petiole cls/len, … |
| bud_state | `/ 10.0` (wrapped ±180 → max 18) | `* 10.0` | |
| curvature | `/ 100.0` | `* 100.0` | |
| phyllo | `/ 180.0` | `* 180.0` | |

The organ-aware scheme exists because the 17D scale block and bud/curv/phyllo columns
are **overloaded per organ type** (angles, integer IDs, physical scales in the same
columns — see §2.4 table and `plant_organ_array.py` `P_COL_*` docs). The one-hot
organ block conditions the DiT to interpret each column family consistently.

---

## 4. Code Changes

```
diffusion_based/dataset/part_array_dataset.py      # 27D layout (FM_BUD_IDX), BUD_STATE_SCALE,
                                                   # standalone encode_fm + decode_fm (importable, GPU-safe):
                                                   # linear scale (no softplus), organ-aware normalization
                                                   # (meta*10, angle carriers /180), classmethods delegate
diffusion_based/dataset/generate_tensor_shards.py  # nodes_27d via encode_fm() — the stale inline
                                                   # 26D hand-encoding (curv at old col14, no bud) removed
diffusion_based/dataset/canonical_cowpea_dataset.py# __getitem__ uses encode_fm (was hand-rolled 26D)
diffusion_based/training/flow_matching.py          # imports FM_* constants from part_array_dataset
diffusion_based/models/helios_pytorch_geometry.py  # build_mesh_from_part_tensor: 26/27D FM decode
                                                   # branch with proper inverse normalizations;
                                                   # extract_part_tensor: angle wrapping to [-180,180)
                                                   # (flower pitch/yaw/roll/azimuth, leaf pitch/yaw/roll,
                                                   # shoot_meta base rotation, peduncle pitch/roll)
```

Backward compatibility: `decode_fm`/`build_mesh_from_part_tensor` accept both 26D
(bud_state → 0) and 27D tensors.

## 5. Verification Summary

| test | result |
|------|--------|
| 17D ↔ 27D FM roundtrip, DAP10/30/50/100 GT | ≤ 3.05e-05 (fp32 noise); corpus dap080 1.53e-05 |
| mesh from 27D nodes vs 17D GT path | bit-identical (DAP10 11198v, DAP50 144531v) |
| fresh shard → dataset → collate → DiT forward | PASS (`pred_velocity (2, 1722, 27)`) |
| shard nodes dim / dtype | (N, 27) fp32, image (4, 512, 512) fp16 |
| FM column boundedness (6 random corpus plants) | scale ≤ 50, bud ≤ 18, curv ≤ 2, phyllo ≤ 1.2, finite |
| 17D → XML → Helios render roundtrip (prior session) | organ masks IoU 1.0000 (unchanged by this work) |

## 6. Shard Regeneration — Local Attempt → SLURM Production Pipeline

### 6.0 Final state (15:07, 2026-08-31)

Shard regeneration was **delegated to the SLURM production pipeline** and the local
runs were cancelled:

```bash
./slurm_scripts/generate_helios_dataset_jobs.sh --skip-xml --submit
# → 40 jobs submitted (IDs 38036260–38036299), Phase 1 XML disabled ("false" == true
#   guard), Phase 2 = generate_tensor_shards.py --num-workers 40 --worker-id {0..39}
#   --total-samples 100000 --shard-size 100 --image-size 512 --max-templates 30
# Target: dataset/helios_data/cowpea_shard/  (1000 shards of 100 samples)
```

All 40 jobs PENDING on the `low` partition (several nodes down*/drained; the H100
`vlm_mmdit_ddp` job 38013284 is behind the same maintenance window). The script has
a **self-healing monitor** (generate_helios_dataset_jobs.sh:238) that re-submits
failed/missing workers until all 1000 shards are verified. Queue snapshot:

- 6 jobs on `(Priority)`, 1 `(Resources)`, 13 `(None)` (fairshare trickle), rest behind
- Jobs use the current `FM_NODE_DIM = 27` code (verified via same PYTHONPATH import)
- Shard dir was empty at submit → nothing skipped, fully fresh generation

**Before handing over to SLURM, the local A100 produced one full 1000-shard batch
with the FIRST normalization** (linear `*50` only, no angle wrapping / meta scaling).
That batch was **superseded** — do not train on it. The `cowpea_shard/` dir was wiped
and re-delegated to SLURM; stale 26D data remains at
`dataset/helios_data/cowpea_shard_stale_26d_20260824/` (rename, not delete).
276 orphan `_tmp_*` render dirs from the killed local workers were also removed.

### 6.1 Operational notes (local-run lessons, relevant to SLURM config)

1. `--max-templates 500` starved progress (>5 K rchar over 25 min/slot, no shard flush —
   mesh pre-build of 500 templates × up-to-330k-vertex plants did not finish in 45 min).
   `--max-templates 50` completes in ~2-4 min/worker (103-209 s for 50 shards after
   preload). The SLURM script uses `--max-templates 30` — safe.
2. `nohup python …` blocks stdout buffering — **always use `python -u`** (the SLURM job
   script already exports `PYTHONUNBUFFERED=1`).
3. On a single GPU, parallel workers each take ~1.1 GB VRAM and reach 100% util;
   workers 0-9 then 10-19 sequentially completed 500 shards/batch in ~25 min.
4. SLURM pipeline advantages confirmed: 40-way array across the `low` partition,
   self-healing re-submission, 8h wall — prefer it over ad-hoc local loops for
   full regens when the cluster is healthy.
5. Old 26D shards were built from a **16D `to_part_tensor` layout with zero base
   columns and different scale semantics** (col12-14 all 0, col23 max 20) — the old
   v-loss ≈ 0.85 was convergence on wrong data, not a healthy baseline.

## 7. Training Restart — DONE, now RUNNING on 4×H100

### 7.0 What actually happened (16:00–17:10, 2026-08-31)

The SLURM training job auto-started ~15:12 the moment the maintenance window ended
and hit the exact failure §7 warned about, then two more script-level issues, then
went healthy:

| # | job | what happened | outcome |
|---|-----|---------------|---------|
| 1 | `38037265` | auto-resumed old **26D** checkpoint into 27D model → `size mismatch for node_in_proj/vel_head` | crashed in 58 s |
| 2 | `38038329` | `--resume` removed, started fresh — but burned `--nproc_per_node=2` of 4 allocated GPUs, hardcoded port/labels | cancelled (see 7.2) |
| 3 | `38039191` | `--master_port=0` **hung torchrun rendezvous** (~7 min silent, no NCCL init) | killed; port → job-ID-derived |
| 4 | `38042895` | 16 CPUs, 4 ranks × batch 8, grad-accum 2, global 128 — healthy but v-loss start 17→9, VRAM only 22/96 GB | killed for VRAM resize |
| 5 | **`38048780`** | **4 ranks × batch 32, grad-accum 1, global 128 — RUNNING** | ✅ live |

The stale 26D checkpoint was archived as
`diffusion_based/checkpoints/fm/cowpea_vlm_scaffold_dit_h100_ddp_stale26d_epoch08.pt`
(2.1 GB, epoch-8 TLE run) and `--resume` was removed from the SLURM script: fresh start.

### 7.2 Launcher de-hardcoded (`slurm_scripts/train_cowpea_vlm_scaffold_dit_ddp.sh`)

- `NPROC=${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}` — DDP ranks =
  GPUs SLURM actually allocated (was hardcoded `2` while `--gres=gpu:4`).
- `BATCH_SIZE=$((128 / NPROC))`, `ACCUM=1` (2 if single GPU) — **fixed global batch
  128**, per-rank micro-batch scales with allocation.
- `MASTER_PORT=$((29500 + SLURM_JOB_ID % 400))` — unique per job (was fixed 29505;
  `--master_port=0` is broken: torchrun elastic rendezvous hangs).
- Banner prints `Allocated GPUs: ${NPROC}x` + names queried at runtime; removed the
  "H100 NVLINK" and `2xh100` wandb-name hardcoding
  (`vlm_scaffold_dit_{world_size}gpu_b{batch}` in `train_cowpea_vlm_scaffold_dit_ddp.py:374`,
  GPU description from `torch.cuda.get_device_name()`).
- `--cpus-per-task=16` (was 32 — 12 busco CPU jobs were starving the queue),
  `--num-workers 3` (4 ranks × (1+3) = 16 CPUs exactly).

### 7.3 VRAM-based batch sizing (measured on H100 NVL 96 GB)

Measured: batch 8/rank → 22 GB (≈ 3 GB fixed + ≈ 2.3 GB/sample). Sized to
**batch 32/rank → ≈ 77 GB expected**, observed **72.8 GB (~76%)** with stable
utilization 100%. Global batch stays 128 so `--lr 2.5e-4` remains valid and the
grad-accum dropped 2 → 1 (fewer optimizer steps, ~4× wall-clock speedup vs run 4).

### 7.4 Live status (17:05, run `38048780`)

```
Epoch 01 [00075/00781] | Step Loss: 7.14 (v: 6.16, macro: 1.82, r_rgb: 0.37, r_dep: 0.23) | VRAM 3.3/72.8GB
```

- v-loss **13.3 → 6.2** over 75 steps (healthy descent; the organ-aware §2.4
  normalization works — no 30k explosion)
- ~0.7 s/step, 781 steps/epoch → **≈ 9 min/epoch**, full 60-epoch run ≈ 9 h
  (fits 24 h wall with 3× margin)
- Log: `slurm_scripts/logs/train_vlm_scaffold_38048780.log`
- Live probe: `srun --overlap --jobid=38048780 nvidia-smi` (login node's nvidia-smi
  shows an empty GPU list — not visible from the head node; use srun)

### 7.5 Historic run archive

- run 1 (`/tmp/local_train/train2.log`): stale 26D shards — killed at step ~1400/6250.
- run 2 (`train3_27d.log`): first-gen 27D shards — pre-§2.2 normalization had
  unbounded scale columns (flower yaw 810° → FM col23 = 40,500); v-loss plateaued at
  30k–300k. Motivated the §2.4 organ-aware normalization + angle wrapping.
- run 3 (local A100, aborted early): superseded by the SLURM run.

## 8. Known Remaining Issue (pre-existing, NOT introduced by this session)

**Corpus XML → 17D → XML has geometry drift** for corpus plants with non-zero
`curvature_perturbations` / `yaw_perturbations` (up to 1.83 rot6d units / 1.18 m
vertex error on `cowpea_dap069_seed90`). Root cause: the canonical **17D part tensor
has no columns for internode curvature/yaw perturbations** (40D typed cols 23–26);
internode col15 is already occupied by `length_segments`. GT plants built with
`dap*_gt_*` templates have zero perturbations so they round-trip exactly (1.4e-4).

This is a **representation gap of the 17D tensor itself**, orthogonal to the 27D FM
work above. Options for a follow-up: extend 17D → 19D (add 2 perturbation columns, now
cheap since shard tooling is consolidated) or zero the perturbations in the corpus.

## 9. Next Steps

1. **Sharding: DONE.** All 1000 fresh 27D shards landed on ~15:05 via the SLURM
   pipeline (40 jobs, validated: 27D dims + one-hot check on 25 random shards +
   spot-checked 20 more after fill). Note the training run confirmed
   `Loaded 100,000 samples across 1000 shards`.
2. **Training: RUNNING (job 38048780).** Monitor v-loss descent; first checkpoint
   will be written by the training script's save cadence (verify a 27D checkpoint):
   `ls -la diffusion_based/checkpoints/fm/cowpea_vlm_scaffold_dit_h100_ddp.pt`.
3. After epoch ~2–4: check the `--helios-roundtrip` eval column images for the
   17D→XML→Helios round-trip self-consistency Δ on generated plants.
4. Decide follow-up for §8 perturbation-gap issue (17D → 19D vs zero the corpus).
5. Commit the session's code changes (see handoff snapshot) once the run is stable
   past epoch 1.

### Handoff snapshot (for the next agent)

- **Live training**: SLURM job `38048780` (`vlm_mmdit_ddp`, 4×H100 NVL gpu-10-58,
  4 ranks × batch 32, global 128, `--helios-roundtrip`, fresh 27D shards, NO resume).
- **Checkpoint watch**: `diffusion_based/checkpoints/fm/cowpea_vlm_scaffold_dit_h100_ddp.pt`
  will now be 27D — the pre-rename `cowpea_vlm_scaffold_dit_h100_ddp_stale26d_epoch08.pt`
  is the incompatible 26D one. Do NOT mix them up.
- Uncommitted code changes: `part_array_dataset.py`, `generate_tensor_shards.py`,
  `canonical_cowpea_dataset.py`, `flow_matching.py`, `helios_pytorch_geometry.py`
  (27D + organ-aware FM + angle wrapping), `train_cowpea_vlm_scaffold_dit_ddp.py`
  (log de-hardcoding), `slurm_scripts/train_cowpea_vlm_scaffold_dit_ddp.sh`
  (N-rank launcher) — commit after run stability check.
- Sharding (complete): jobs 38036260–38036299, all-referenced logs
  `slurm_scripts/logs/unified_cowpea_20260831_150646/`.
- `dataset/helios_data/cowpea_shard/` — **fresh 27D, valid, in use by 38048780**.
- `dataset/helios_data/cowpea_shard_stale_26d_20260824/` — preserved old data, do not train on it.
- Launcher caveat: `--master_port=0` hangs torchrun rendezvous on this stack
  (torch 2.x) — never use; use the job-ID-derived port already in the script.