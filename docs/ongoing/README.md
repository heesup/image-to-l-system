# Ongoing Documentation Index

This directory tracks **only actively running work** for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Folder conventions:**
- `docs/ongoing/` ← 지금 진행 중인 것만 (this folder)
- `docs/done/` ← 완료된 구현 세션 핸드오프
- `docs/archived/design/` ← 확정됐거나 채택되지 않은 설계 문서 (ADR, spec)
- `docs/results/` ← 실험 결과 및 마일스톤 리포트

---

## 🚨 CURRENT SYSTEM STATE (2026-09-08 ~18:45 PDT)

| Component | Status | Details |
| :--- | :---: | :--- |
| **Main Training Job** | 🟢 RUNNING | Job `38146809`, **Epoch 54/500**, gpu-10-54, 4× RTX 6000 Ada, 37.2 GB VRAM (78.6%), TIME_LEFT ~20h |
| **Saved Checkpoints** | 🟢 READY | `hierarchical_fm_epoch_025.pt`, `hierarchical_fm_epoch_050.pt` (both 1.4 GB) |
| **Helios Dataset Synthesis** | 🟢 IN PROGRESS | 40 shards, **65,488 / ~120,000 XMLs (54.6%)**, **35,266 cache `.pt` tensors** |
| **Epoch 50 Self-Consistency** | 🟢 EVALUATED | IoU: 9.7%, Depth MAE: 8.25 cm, Node RMSE: 1.7 cm, Loss: 3.58 |
| **OnDemand Desktop** | 🟢 RUNNING | Job `38147219`, gpu-5-58 — **DO NOT CANCEL** |

---

## Active Documents

| Document | Purpose |
| :--- | :--- |
| **[AGENT_TAKEOVER_GUIDE.md](AGENT_TAKEOVER_GUIDE.md)** | Master handover: full system state, all 2026-09-08 changes, next steps, gotchas, commit history |
| **[`docs/results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md`](../results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md)** | Technical analysis of Epoch 50 results (`AncPos: 0.009` vs Column 6 skeleton distortion) |

---

## Next Steps (from AGENT_TAKEOVER_GUIDE)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P1** | Let Job `38146809` continue past Epoch 75 / 100 | Checkpoint `epoch_075.pt` and next visual panel expected in ~20–30 min |
| **P2** | Track Helios shard completion | Currently 65.5k XMLs / 35.3k cache tensors; target 120k |
| **P3** | Monitor 6D rotation convergence in Stage 3 | Verify outward radial branch divergence in Epoch 75/100 panels |
| **P4** | Evaluate Bidirectional Chamfer Distance | Add max/mean distance from GT $\to$ Pred to avoid one-way clustering metric bias |
| **P5** | Expand training to full 120k dataset | Once all Helios shards complete, launch next 500-epoch scaling run |
