# Ongoing Documentation Index

This directory tracks **only actively running work** for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Folder conventions:**
- `docs/ongoing/` ← 지금 진행 중인 것만 (this folder)
- `docs/done/` ← 완료된 구현 세션 핸드오프
- `docs/archived/design/` ← 확정됐거나 채택되지 않은 설계 문서 (ADR, spec)
- `docs/results/` ← 실험 결과 및 마일스톤 리포트

---

## 🚨 CURRENT SYSTEM STATE (2026-09-10 evening PDT)

| Component | Status | Details |
| :--- | :---: | :--- |
| **Main Training Job** | 🟢 RUNNING | Job `38234682` on `gpu-10-50` (4x RTX 6000 Ada, batch 152/GPU, global 608, 85.3% VRAM, 76D flow v3 stack) |
| **PhytomerVAE v3 (normalized)** | 🟢 DEFAULT | `phytomer_vae_v3` — val recon **0.070**, cls **100%** (10-slot, 240D in, scale-normalized targets; higher recon is expected, NOT a regression) |
| **Dataset (images+nodes)** | 🟢 COMPLETE | **100,000 / 100,000** XMLs + cache `.pt` (all with `phytomer_ids`) |
| **Phytomer packet cache** | 🟢 COMPLETE | **100,000 / 100,000** v3 (10-slot, absolute packets + normalized latent, `pkt_version: 3`; version gate auto-skips stale files) |
| **76D flow + macro heads** | 🟢 WIRED | `[pos\|rot\|s_a\|latent]`, Stage-2 `scale_head`, `phy_head` bias-init @ log(50), DAP clue `dap_embed` — pred count 1.6→13.5 in local 5-epoch test |
| **Render pipeline** | 🟢 UPGRADED | epoch-gated render grads (epoch 5), probe token reuse, semantic color palette Parameter, CUDA-synced `backward/probe/other` timers |
| **Dataset pipeline** | 🟢 REFACTORED | `generate_cache.py` (`cache`/`pkt` modes, `PKT_VERSION=3` stamp+gate) |
| **GUI visualizer** | 🟢 LIVE | `tools/phytomer_vae_visualizer.py` — v3 denormalize-on-decode (s_a per packet) |
| **Assembly rules** | 🟢 VERIFIED | stem base T-joint (roundtrip p99 0.14cm), leaflets 0.8/0.8/1.0 petiole curve, repro @ peduncle tip |
| **OnDemand Desktop** | 🟢 RUNNING | Job `38230613`, gpu-5-58 — **DO NOT CANCEL** |

---

## Active Documents

| Document | Purpose |
| :--- | :--- |
| **[AGENT_TAKEOVER_GUIDE.md](AGENT_TAKEOVER_GUIDE.md)** | Master handover: full system state, 2026-09-08→10 changes, failed-launch forensics, next steps, gotchas |
| **[20260909_phytomer_latent_and_local_matching.md](20260909_phytomer_latent_and_local_matching.md)** | Master engineering log: phytomer latent, Stage-3 decoder, pipeline refactor (§4.6), v3+76D (§4.9), launch-debugging marathon (§4.10) |
| **[20260909_phytomer_latent_visualizer_gui.md](20260909_phytomer_latent_visualizer_gui.md)** | Completed PhytomerVAE latent visualizer GUI (features, run steps, gotchas) |
| **[20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md](20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md)** | Anchor capacity logistic recalibration (p97.5 curve), predicted-phytomer dynamic slicing, DAP-bucketed batching, gradient-safety audit |
| **[`docs/results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md`](../results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md)** | Technical analysis of Epoch 50 results (`AncPos: 0.009` vs Column 6 skeleton distortion) |

---

## Next Steps (from AGENT_TAKEOVER_GUIDE)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P1** | **Monitor Job 38234682** | Running on `gpu-10-50` (batch 152/GPU, global 608); check first epoch metrics & WandB |
| **P2** | Epoch-1 sanity on cluster | Loss ↓, Pred count ~50 (bias-init), Reference column not N/A, DAP-spread panels |
| **P3** | Monitor s_a / 6D rotation convergence to epoch 25/50 | s_a (petiole length) should track DAP growth |
| **P4** | Evaluate Bidirectional Chamfer Distance | Add max/mean distance from GT→Pred to avoid one-way clustering metric bias |
| **P5** | Backbone A/B arms | `slurm_scripts/submit_backbone_ablation.sh` — only after single-arm training is stable |
