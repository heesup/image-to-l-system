# Ongoing Documentation Index

This directory tracks **only actively running work** for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Folder conventions:**
- `docs/ongoing/` ← 지금 진행 중인 것만 (this folder)
- `docs/done/` ← 완료된 구현 세션 핸드오프
- `docs/archived/design/` ← 확정됐거나 채택되지 않은 설계 문서 (ADR, spec)
- `docs/results/` ← 실험 결과 및 마일스톤 리포트

---

## 🚨 CURRENT SYSTEM STATE (2026-09-10 PDT)

| Component | Status | Details |
| :--- | :---: | :--- |
| **Main Training Job** | 🔴 STOPPED | `38183271` cancelled for pipeline refactor; resume per P2 below |
| **PhytomerVAE v2 (10-slot)** | 🟢 ACCEPTED | `phytomer_vae_v2` — val recon **0.0263**, cls **100%** (NUM_SLOTS 10, VAE 240D in, latent 64D; pkt cache 100k regenerated) |
| **Dataset (images+nodes)** | 🟢 COMPLETE | **100,000 / 100,000** XMLs + cache `.pt` (all with `phytomer_ids`) |
| **Phytomer packet cache** | 🟢 COMPLETE | **100,000 / 100,000** v2 10-slot (`cowpea_curv26_pkt`; v1 8-slot backed up as `cowpea_curv26_pkt_v1_8slot`) |
| **Dataset pipeline** | 🟢 REFACTORED | `generate_cache.py` (`cache`/`pkt` modes) emits `phytomer_ids` + pkt + latent inline |
| **GUI visualizer** | 🟢 LIVE | `tools/phytomer_vae_visualizer.py` on :7860 — v2 10-slot, PC sliders, stem/leaflet base rules fixed (see GUI doc notes 12–15) |
| **Assembly rules** | 🟢 VERIFIED | stem base = −fwd·L (tip=node), leaflets 0.8/0.8/1.0 petiole curve, repro @ peduncle tip (residual ≤0.24cm mean) |
| **OnDemand Desktop** | 🟢 RUNNING | Job `38147219`, gpu-5-58 — **DO NOT CANCEL** |

---

## Active Documents

| Document | Purpose |
| :--- | :--- |
| **[AGENT_TAKEOVER_GUIDE.md](AGENT_TAKEOVER_GUIDE.md)** | Master handover: full system state, all 2026-09-08/09 changes, next steps, gotchas, commit history |
| **[20260909_phytomer_latent_and_local_matching.md](20260909_phytomer_latent_and_local_matching.md)** | Phytomer-level latent + XML-phytomer clustering, Stage-3 flow decoder, pipeline refactor (§4.6) |
| **[20260909_phytomer_latent_visualizer_gui.md](20260909_phytomer_latent_visualizer_gui.md)** | Completed PhytomerVAE latent visualizer GUI (features, run steps, gotchas) |
| **[20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md](20260908_anchor_capacity_recalibration_and_pred_phytomer_slicing.md)** | Anchor capacity logistic recalibration (p97.5 curve), predicted-phytomer dynamic slicing, DAP-bucketed batching, gradient-safety audit |
| **[`docs/results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md`](../results/20260908_epoch050_skeleton_geometry_and_chamfer_bias_analysis.md)** | Technical analysis of Epoch 50 results (`AncPos: 0.009` vs Column 6 skeleton distortion) |

---

## Next Steps (from AGENT_TAKEOVER_GUIDE)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P1** | ~~Finish pkt backfill~~ | **DONE** — 100k/100k v2 10-slot (2026-09-10) |
| **P2** | Relaunch phytomer training | `FLOW_GRANULARITY=phytomer sbatch slurm_scripts/train_hierarchical_flow_matching.sh` (VAE + pkt paths are now defaults; slots_per_anchor=10) |
| **P3** | Monitor ClsAcc / render loss to Epoch 25/50 | Check `hierarchical_self_consistency_epoch_025.png` |
| **P4** | Evaluate Bidirectional Chamfer Distance | Add max/mean distance from GT $\to$ Pred to avoid one-way clustering metric bias |
| **P5** | ~~PhytomerVAE GUI app~~ | **DONE + hardened** — see GUI doc notes 8–16 (PC sliders, GLB race, stem/leaflet base fixes) |
