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
| **Main Training Job** | 🟡 UNSTABLE | Job `38235969` on `gpu-10-50` (4x RTX 6000 Ada, epoch 3에서 VelLoss 1050으로 재폭발 중, Recovery 모드 동작 중) |
| **Gradient Bug** | 🟢 FIXED | `.detach()` 누락 → fwd1 `no_grad()`, z_0 detach, fine_stage conditioning detach 모두 적용 |
| **Residual Issue** | 🔴 INVESTIGATING | `loss_fine_vel` 스케일(1050)이 `loss_anchor_pos`(292)를 3.6배 압도 → 에폭 3 양성 피드백 |
| **PhytomerVAE v3 (normalized)** | 🟢 DEFAULT | `phytomer_vae_v3` — val recon **0.070**, cls **100%** (10-slot, 240D in, scale-normalized targets) |
| **Dataset (images+nodes)** | 🟢 COMPLETE | **100,000 / 100,000** XMLs + cache `.pt` (all with `phytomer_ids`) |
| **Phytomer packet cache** | 🟢 COMPLETE | **100,000 / 100,000** v3 (10-slot, absolute packets + normalized latent, `pkt_version: 3`) |
| **76D flow + macro heads** | 🟢 WIRED | `[pos\|rot\|s_a\|latent]`, Stage-2 `scale_head`, `phy_head` bias-init @ log(50) |
| **OnDemand Desktop** | 🟢 RUNNING | Job `38230613`, gpu-5-58 — **DO NOT CANCEL** |

---

## Active Documents

| Document | Purpose |
| :--- | :--- |
| **[AGENT_TAKEOVER_GUIDE.md](AGENT_TAKEOVER_GUIDE.md)** | Master handover: full system state, 2026-09-08→10 changes, failed-launch forensics, next steps, gotchas |
| **[20260909_phytomer_latent_and_local_matching.md](20260909_phytomer_latent_and_local_matching.md)** | Master engineering log: phytomer latent, Stage-3 decoder, pipeline refactor |
| **[20260909_phytomer_latent_visualizer_gui.md](20260909_phytomer_latent_visualizer_gui.md)** | Completed PhytomerVAE latent visualizer GUI |
| **[20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md](20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md)** | Anchor capacity logistic recalibration, gradient-safety audit |
| **[`docs/results/20260910_gradient_explosion_debug_and_architecture_comparison.md`](../results/20260910_gradient_explosion_debug_and_architecture_comparison.md)** | 오늘 세션: .detach() 버그 해부, 9/8 vs. 현재 아키텍처 비교, 잔류 VelLoss 폭발 문제 |

---

## Next Steps (updated 2026-09-10 night)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P0** | **VelLoss 폭발 원인 조사** | `tgt_z1_phyto` 스케일 및 VAE 잠재벡터 정규화 확인; `loss_fine_vel` 가중치 2.0 → 0.1~0.5 감소 검토 |
| **P1** | **에폭 3번 이후 안정화 확인** | Job `38235969` Recovery가 끝나고 VelLoss가 낮아지는지 모니터 |
| **P2** | PhyLoss 수렴 모니터 | Epoch 50까지 Pred/GT 간격이 10% 이내로 좁혀지는지 확인 |
| **P3** | AncPos 모니터 | Epoch 30까지 0.01~0.03m 안착 모니터 |
| **P4** | Backbone A/B ablation | 단일-팜 안정 후 실행 |
