# Ongoing Documentation Index

This directory tracks **only actively running work** for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Folder conventions:**
- `docs/ongoing/` ← 지금 진행 중인 것만 (this folder)
- `docs/done/` ← 완료된 구현 세션 핸드오프
- `docs/archived/design/` ← 확정됐거나 채택되지 않은 설계 문서 (ADR, spec)
- `docs/results/` ← 실험 결과 및 마일스톤 리포트

---

## 🚨 CURRENT SYSTEM STATE (2026-09-09 ~02:00 PDT)

| Component | Status | Details |
| :--- | :---: | :--- |
| **Main Training Job** | 🟢 RUNNING | Job `38146809`, **Epoch ~304/500**, gpu-10-54, 4× RTX 6000 Ada (organ path, untouched) |
| **PhytomerVAE-64 (relative)** | 🟢 DONE | `phytomer_vae_relative_d` — **rot 3.16°** (near 2.10° baseline), class 99.99%, base 0.16cm |
| **Stage-3 phytomer flow** | 🟢 WIRED | `--flow-granularity phytomer` (73D bridge targets) implemented + smoke-tested |
| **PhytomerVAE GUI** | 🟢 DONE | `tools/phytomer_vae_visualizer.py` — PCA cloud + 64D sliders + Web 3D (see GUI doc) |
| **Helios Dataset Synthesis** | 🟢 IN PROGRESS | 40 shards, ~82k XMLs / ~78k cache `.pt` tensors (target 100k) |
| **OnDemand Desktop** | 🟢 RUNNING | Job `38147219`, gpu-5-58 — **DO NOT CANCEL** |

---

## Active Documents

| Document | Purpose |
| :--- | :--- |
| **[AGENT_TAKEOVER_GUIDE.md](AGENT_TAKEOVER_GUIDE.md)** | Master handover: full system state, all 2026-09-08/09 changes, next steps, gotchas, commit history |
| **[20260909_phytomer_latent_visualizer_gui.md](20260909_phytomer_latent_visualizer_gui.md)** | **[NEW]** Completed PhytomerVAE latent visualizer GUI (features, run steps, gotchas) |
| **[20260909_phytomer_latent_and_local_matching.md](20260909_phytomer_latent_and_local_matching.md)** | Phytomer-level latent + relative rotations (3.16°), Stage-3 flow decoder, training wiring, GUI handoff spec |
| **[20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md](20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md)** | Anchor capacity logistic recalibration (p97.5 curve), predicted-phytomer dynamic slicing, DAP-bucketed batching, gradient-safety audit |
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
| **P6** | PhytomerVAE-64 validation gate | **PASSED** — relative-D rot 3.16° (see design doc §2) |
| **P7** | Stage-3 phytomer integration | **WIRED** — `--flow-granularity phytomer` (73D bridge targets), smoke-tested |
| **P8** | PhytomerVAE GUI app | **DELEGATED** to another agent (handoff spec in design doc §4.2) |
| **P9** | Launch phytomer-mode training | On the 100k dataset once shards complete |
