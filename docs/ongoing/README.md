# Ongoing Documentation Index

This directory tracks **only actively running work** for the **Image-to-L-System / 3D Inverse Plant Reconstruction** project.

**Folder conventions:**
- `docs/ongoing/` ← 지금 진행 중인 것만 (this folder)
- `docs/done/` ← 완료된 구현 세션 핸드오프
- `docs/archived/design/` ← 확정됐거나 채택되지 않은 설계 문서 (ADR, spec)
- `docs/results/` ← 실험 결과 및 마일스톤 리포트

---

## 🚨 CURRENT SYSTEM STATE (2026-09-14 ~11:30 PDT, Claude Code handover)

| Component | Status | Details |
| :--- | :---: | :--- |
| **Main training** | 🟢 CLUSTER, 2 GPU | Job **`38253656`** (gpu-6000_ada-h, gpu-10-54, 24 h): v9 lineage, `AUTO_RESUME` from `hierarchical_fm_v9/hierarchical_fm_epoch_025.pt` at batch 48/GPU, `NUM_WORKERS=8`, on `25c2251` — the training step without per-sample Python: vectorized packet assembly (`f3c5366`), parent links in the loader workers + one VAE encode per batch + one nvdiffrast pass for all rendered plants (`a1a82bc`), pointer-jumping `chain_phytomers` + batched matcher over the padded batch (`25c2251`). Step at batch 48: 1.9 s this morning → 0.15-0.24 s (1 GPU). Output `hierarchical_fm_v9/`, log `slurm_scripts/logs/hierarchical_fm_38253656.log`, panels `run_38253656/`. Predecessors today: `38253029` (epochs 16-21, ~9 min/epoch), `38253401` (epochs 22-25, ~7 min/epoch; IoU 32.6-35.5%, node RMSE 1.8-1.9 cm). |
| **Cluster jobs (queued)** | ⚪ NONE | The `low`-partition resubmissions (`38249632`, `38250275`) were cancelled 9/14 10:13. The lineage lives only in `38253656`; it hits the 24 h limit ~9/15 14:45 — resubmit then with `AUTO_RESUME=1` (same OUTPUT_DIR) or queue a `low`/`publicgrp` continuation with `--requeue`. |
| **Stage 2 gradient burst** | 🟢 FIXED | The coarse `nn.TransformerDecoder` had no final LayerNorm, so its raw residual stream fed the bf16 phytomer self-attention (design doc §1.9.2). `FM_DECODER_FINAL_NORM=1` + `FM_SELFATTN_FP32=1` (defaults): 0/8 burst steps on the frozen burst state vs 8/8. Consider it closed once the run passes epoch ~30 (the v8 lineage burst at 27-28). |
| **PhytomerVAE** | 🟢 DEFAULT | `phytomer_vae_v9_tl_rw4_20k` — 128D hybrid (48 coarse + 10×8 per-slot residual), terminal-last packets, 20,000 files. Export VAE (eval scripts default to it, `PHYTOMER_TERMINAL_LAST=1`) and the FM training VAE of the v9 run. |
| **Packet cache** | 🟢 VAE-INDEPENDENT | `dataset/cache/cowpea_curv26_pkt_v9/` — 100,000 files, `pkt_version` 7, terminal-last packets. Since 2026-09-14 the cache stores only packets/presence/centers/refs/keys and FM encodes the Stage-3 latents on the fly from them, so the cache is VAE-independent (a new VAE needs no rebuild). The packet ORDER still must match the VAE: v9 runs pair this cache with `PHYTOMER_TERMINAL_LAST=1`; v8 runs keep `cowpea_curv26_pkt/` (pkt 6, `PHYTOMER_TERMINAL_LAST=0`). |
| **Helios round-trip, exact_gt DAP 10/50/90** | 🟢 SOLVED | fig14: IK-only **99.9 / 99.6 / 97.7%**, VAE round-trip **93.7 / 98.3 / 95.8%** FG IoU (`docs/results/assets/fig14_phytomer_vae_helios_roundtrip.png`). |
| **Helios round-trip, dataset DAP 15/40/75** | 🟢 SOLVED | fig12: packet path **98.3 / 98.2 / 95.6%**, VAE **95.1 / 96.8 / 95.6%** (was 81.8 / 92.2 / 54.6 on the morning of 2026-09-14; `docs/results/assets/fig12_phytomer_10slot_helios_roundtrip.png`). |
| **Topology targets** | 🟢 FIXED | `gt_parent_links(..., internode_base=)` resolves 97.8% of lateral branch points (was 66.2%; the training call site passes the decoded base). `chain_phytomers` follows drooping shoots (same-shoot links 100% on 30 plants) and keeps the cotyledon node as shoot 0 (`root_own_shoot`). |
| **Export (14D → Helios XML)** | 🟢 | Analytical converter + **stem IK** (`part_tensor_stem_ik.py`) + **leaf IK** (`part_tensor_leaf_ik.py`), both on by default (`PART_TENSOR_STEM_IK`, `PART_TENSOR_LEAF_IK`). |
| **Heesup's regeneration jobs** | ⛔ DO NOT CANCEL | `regen_shard` / `regen_synth` / `regen_mopup` (geminigrp) — they hold the group's GPU quota; training goes to `low`/`publicgrp`. |

---

## Active Documents

| Document | Purpose |
| :--- | :--- |
| **[20260912_stage2_stage3_boundary_and_remaining_redundancy.md](20260912_stage2_stage3_boundary_and_remaining_redundancy.md)** | **Primary record (read §5 first):** §0 status, §1.9.2 gradient-burst root cause + ablation, §2.1-2.3 Stage 2/3 boundary redesign, §2.4-2.5 round-trip + stem IK, **§2.6 dataset-plant round-trip (chaining, branch points, leaf IK)**, §6 commit log |
| **[AGENT_TAKEOVER_GUIDE.md](AGENT_TAKEOVER_GUIDE.md)** | Master handover; §0-A is the 2026-09-14 state, later sections are the 2026-09-11 state where not contradicted |
| **[`docs/results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md`](../results/20260914_stage2_burst_fix_and_dataset_plant_roundtrip.md)** | 2026-09-14 report: burst fix, dataset-plant round-trip, figures, training state (Korean) |
| [20260911_takeover_grad_norm_fix_roundtrip_diagnosis_10slot_restore.md](20260911_takeover_grad_norm_fix_roundtrip_diagnosis_10slot_restore.md) | 2026-09-11: grad-norm deadlock fix, 10-slot contract restore, first round-trip collapse diagnosis |
| [20260911_hybrid_vae_rotation_capacity_and_topology_experiments.md](20260911_hybrid_vae_rotation_capacity_and_topology_experiments.md) | 2026-09-11: hybrid 128D VAE, rotation capacity, topology-recovery ablation |
| [20260909_phytomer_latent_and_local_matching.md](20260909_phytomer_latent_and_local_matching.md) | Engineering log: phytomer latent, Stage-3 decoder, pipeline refactor |
| [20260909_phytomer_latent_visualizer_gui.md](20260909_phytomer_latent_visualizer_gui.md) | PhytomerVAE latent visualizer GUI |
| [20260908_phytomer_capacity_recalibration_and_pred_phytomer_slicing.md](20260908_phytomer_capacity_recalibration_and_pred_phytomer_slicing.md) | Node capacity logistic recalibration, gradient-safety audit |

---

## Next Steps (updated 2026-09-14 ~11:30, Claude Code handover)

| Priority | Task | Notes |
| :--- | :--- | :--- |
| **P0** | **Watch the v9 run past epoch 30** | canary hits, Stage 2 Scl/Ord/Ext losses, and the per-epoch self-consistency panels; the goal is rendering that matches the original, not the loss alone. With the assembly vectorized, epochs are ~6x faster; re-measure whether a larger batch now pays (the 256 measurement was made with the old assembly loop dominating). |
| **P1** | **Keep the lineage alive past the 24 h limit** | `38253656` ends ~9/15 14:45; resubmit with `AUTO_RESUME=1` into `hierarchical_fm_v9/` (or a `low` job with `--requeue`). |
| **P2** | **Stage 3 A/B, three arms (takeover guide §0-B.9)** | From epoch 45: baseline `38257989` (cluster), geometry arm `STAGE3_GEOMETRY=1` (local, `local_s3geom_ab2.log`), teacher-forcing arm `STAGE3_GT_NODES=1` (local, `local_gtnodes_ab.log`). Epochs 46-59: baseline mean 31.4 (spread ±3), geometry 47-50: 30.6/35.0/28.6/30.9% (mean 31.3; Vel still falling; ablation P 27.5→29.6 but its sampled latent stays noisy, std 0.46 vs GT 0.18), gt_nodes 46-49: 32.8/31.2/35.8/**39.4**% (mean 34.8 vs baseline 30.5 over the same epochs; deployable output, not teacher-forced; first checkpoint at epoch 50). The gt_nodes arm is separating upward. **Finding:** Stage 3's latent carries no per-node image information (sampled latent worse than the dataset-mean latent, R² −0.34; t=0 estimate R² ≈ 0.1) while the GT latent is worth +33-38 IoU with GT geometry. Arm 4 `RENDER_TO_LATENT=1` = `low` job `38260124` (2×A100, from epoch 45, `hierarchical_fm_v9_r2l/`); then a unit-variance latent scale for the flow. |
| **P3** | **Export residuals** | terminal leaflets (one free angle, 1-1.6° mean), first-node petiole azimuth (only the shoot base roll can set it; a naive roll step was reverted), 2.2% of laterals still resolve to the wrong branch point |
| **P4** | Housekeeping | delete diagnostic checkpoint dirs (`local_probe_burst`, `local_act_probe`, `local_replay_*`, `local_smoke_parent`) if space matters |
