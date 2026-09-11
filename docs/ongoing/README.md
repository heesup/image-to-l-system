# Ongoing Documentation Index

This directory tracks **only actively running work** for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Folder conventions:**
- `docs/ongoing/` ← 지금 진행 중인 것만 (this folder)
- `docs/done/` ← 완료된 구현 세션 핸드오프
- `docs/archived/design/` ← 확정됐거나 채택되지 않은 설계 문서 (ADR, spec)
- `docs/results/` ← 실험 결과 및 마일스톤 리포트

---

## 🚨 CURRENT SYSTEM STATE (2026-09-10 night PDT)

| Component | Status | Details |
| :--- | :---: | :--- |
| **Main Training Job** | 🟢 RUNNING | Job `38236699` (4x RTX 6000 Ada) — **하이브리드 디커플링 정식 가동 중** |
| **Architecture Decision**| 🟢 RATIFIED | **하이브리드 디커플링 (Hybrid Decoupled Architecture)** 확정: Stage 2 3D 뼈대 전담 + Stage 3 64D VAE Latent Flow Matching ($z_0 \sim \mathcal{N}(0, I_{64})$) |
| **Loss Function Diet** | 🟢 RATIFIED | 10개 $\to$ 8개 정예 손실 체계 (`loss_cos`, `loss_dap` 제거, `loss_anchor_rot` 정규 지도 추가) |
| **PhytomerVAE v3 (normalized)** | 🟢 DEFAULT | `phytomer_vae_v3` — val recon **0.070**, cls **100%** (10-slot, 240D in, scale-normalized targets) |
| **Dataset (images+nodes)** | 🟢 COMPLETE | **100,000 / 100,000** XMLs + cache `.pt` (all with `phytomer_ids`) |
| **Phytomer packet cache** | 🟢 COMPLETE | **100,000 / 100,000** v3 (10-slot, absolute packets + normalized latent, `pkt_version: 3`) |
| **OnDemand Desktop** | 🟢 RUNNING | Job `38230613`, gpu-5-58 — **DO NOT CANCEL** |

---

## Active Documents

| Document | Purpose |
| :--- | :--- |
| **[AGENT_TAKEOVER_GUIDE.md](AGENT_TAKEOVER_GUIDE.md)** | Master handover: full system state, 2026-09-08→10 changes, failed-launch forensics, next steps, gotchas |
| **[`docs/results/20260910_gradient_explosion_debug_and_architecture_comparison.md`](../results/20260910_gradient_explosion_debug_and_architecture_comparison.md)** | 오늘 세션: .detach() 버그 해부, 하이브리드 디커플링 아키텍처 확정 및 Loss 정예화 분석 |
| **[20260909_phytomer_latent_and_local_matching.md](20260909_phytomer_latent_and_local_matching.md)** | Master engineering log: phytomer latent, Stage-3 decoder, pipeline refactor |
| **[20260909_phytomer_latent_visualizer_gui.md](20260909_phytomer_latent_visualizer_gui.md)** | Completed PhytomerVAE latent visualizer GUI |
| **[20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md](20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md)** | Anchor capacity logistic recalibration, gradient-safety audit |

---

## Next Steps (updated 2026-09-10 night)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P0** | **하이브리드 디커플링 코드 구현** | ✅ Stage 3 64D VAE Flow Matching, Stage 2 rot_head 지도, 8대 손실 정리 완료 |
| **P1** | **단일 배치/유닛 테스트 검증** | `loss_fine_vel` $\sim 1.0$ 및 그래디언트 안전성 확인 |
| **P2** | **Slurm 신규 잡 제출** | Job `38235969` 취소 후 신규 학습 잡 제출 및 초기 에폭 안정성 모니터 |
| **P3** | **AncPos / PhyLoss 모니터** | Epoch 30까지 0.01~0.03m 안착 모니터 |
