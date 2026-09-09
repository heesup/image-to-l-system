# Ongoing Documentation Index

This directory tracks **only actively running work** for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Folder conventions:**
- `docs/ongoing/` ← 지금 진행 중인 것만 (this folder)
- `docs/done/` ← 완료된 구현 세션 핸드오프
- `docs/archived/design/` ← 확정됐거나 채택되지 않은 설계 문서 (ADR, spec)
- `docs/results/` ← 실험 결과 및 마일스톤 리포트

---

## 🚨 CURRENT SYSTEM STATE (2026-09-08 ~17:30 PDT)

| Component | Status | Details |
| :--- | :---: | :--- |
| **Main Training Job** | 🟢 RUNNING | Job `38146809`, Epoch 26/500, gpu-10-54, 4× RTX 6000 Ada, 40.5 GB VRAM |
| **Helios Dataset Synthesis** | 🟡 IN PROGRESS | 40 shards (38147132–38147171), ~51,880/~120,000 XMLs done |
| **Cache Tensors Available** | ~22,600 `.pt` | In `dataset/cache/cowpea_curv26/` |
| **Per-Epoch Eval Panels** | ⚠️ PENDING | Job 38146809 started before `--eval_every`. Restart with INIT_CHECKPOINT at Epoch 25 |
| **OnDemand Desktop** | 🟢 RUNNING | Job `38147219`, gpu-5-58 — **DO NOT CANCEL** |

### ⚡ Most Urgent Action
```bash
# 1. Check if Epoch 25 checkpoint is ready
ls -lh diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt

# 2. If it exists → cancel old job → re-submit with resume (gets --eval_every 1)
scancel 38146809
INIT_CHECKPOINT=diffusion_based/checkpoints/hierarchical_latent_fm/hierarchical_fm_epoch_025.pt \
  sbatch slurm_scripts/train_hierarchical_flow_matching.sh
```

---

## Active Documents

| Document | Purpose |
| :--- | :--- |
| **[AGENT_TAKEOVER_GUIDE.md](AGENT_TAKEOVER_GUIDE.md)** | Master handover: full system state, all 2026-09-08 changes, next steps, gotchas, commit history |

---

## Next Steps (from AGENT_TAKEOVER_GUIDE)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P1** | Epoch 25 checkpoint → cancel 38146809 → re-submit with `INIT_CHECKPOINT` | Activates `--eval_every 1` + `--resume` |
| **P2** | Monitor Helios shard completion | Target 120k XMLs + cache |
| **P3** | Once all shards done → expand training with fuller dataset | ~120k samples available |
| **P4** | Inspect `hierarchical_self_consistency_epoch_050.png` vs `epoch_100.png` | Check dormant slot damping effectiveness |
