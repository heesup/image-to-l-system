# Empirical Findings: Plant Organ Vector Sparsity & Gradient Unbalance — Measured via Frozen OrganLatentVAE

**Date**: September 8, 2026
**Status**: Verified (all numbers measured from live data, zero synthetic/mock values)
**Figure**: [`docs/results/assets/fig_organ_vae_sparsity_gradient_balance.png`](assets/fig_organ_vae_sparsity_gradient_balance.png)
**Measured Summary**: [`docs/results/assets/fig_organ_vae_sparsity_gradient_balance.json`](assets/fig_organ_vae_sparsity_gradient_balance.json)
**Generator Script**: `scripts/generate_fig_vae_sparsity_gradient_balance.py`

---

## 1. Purpose

The Option B specification (`ongoing/20260907_latent_hierarchical_flow_matching_specification.md`) motivates the 16D latent transition with a theoretical "150× gradient scale disparity" claim. This document **empirically measures** the actual sparsity and gradient-unbalance statistics of the plant organ vector representation directly from the shard corpus and the frozen `OrganLatentVAE`, replacing theoretical claims with verified numbers.

---

## 2. Measurement Methodology

| Item | Value |
| :--- | :--- |
| Corpus | `dataset/cache/cowpea_curv26/` — 10,000 Helios Cowpea 26D shard samples |
| Stratified scan | 600 randomly-sampled shards (seed 0) → **184,859 physical organs** (classes ≥ 3, i.e. `INTERNODE` … `BUD_ABORTED`) |
| Gradient probe | $dL/dz$ evaluated at the Flow Matching prior $z_0 \sim \mathcal{N}(0, I)$ through the **frozen** decoder of `diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt`, using the exact composite decode loss (cls CE + base + rot + log-space scale + curv, `organ_latent_vae.py::compute_loss` weights) |
| Raw-space gradient baseline | Per-channel gradient RMS at the mean predictor = $2\sigma$ per channel ( SmoothL1/MSE regression target variance) |
| Latent manifold | PCA (lowrank) of encoded $\mu$ vectors + silhouette scoring on 8k subsample |

Regenerate everything with:
```bash
PYTHONPATH=. python scripts/generate_fig_vae_sparsity_gradient_balance.py
```

---

## 3. Measured Findings

### 3.1 Botanical Class Imbalance (184,859 organs)

| Class | Share | Class | Share |
| :--- | :---: | :--- | :---: |
| LEAF | **47.40%** | FRUIT | 1.79% |
| PETIOLE | 16.23% | PEDUNCLE | 1.61% |
| INTERNODE | 15.91% | FLOWER_OPEN | 0.82% |
| BUD_ACTIVE | 8.05% | FLOWER_CLOSED | **0.67%** |
| BUD_ABORTED | 7.54% | | |

- **LEAF : FLOWER_CLOSED = 71×** frequency imbalance; all reproductive organs < 1.8% each.
- Consequence: an unweighted regression/classification objective is dominated by foliage geometry; peduncle/flower gradient signal is drowned ~70:1.

### 3.2 Raw 13D Gradient Unbalance (tube organs: INTERNODE + PETIOLE + PEDUNCLE, n = 47,125)

Per-channel gradient RMS ($2\sigma$, FM-encoded units):

| Channel | $2\sigma$ | Channel | $2\sigma$ |
| :--- | :---: | :--- | :---: |
| $s_{len}$ | **7.15** | rot channels | 0.76 – 1.22 |
| bx, by, bz | 3.44 – 4.23 | $s_{rad}$ | **0.12** |
| curv | 2.29 | $s_z$ | **0.00 (DEAD)** |

- **Spread $s_{len}/s_{rad}$ = 61×** (CV = 0.93) — matches the specification's qualitative claim (the "150×" figure corresponds to physical peduncle units, measured below at 156×).
- **$s_z$ is structurally dead** for tube organs ($\sigma = 0$: stems/petioles carry `[L, r, 0]`), so a unified MSE wastes a full output channel.
- **Peduncle physical $L/r$ = 156×** (mean length 35.1 cm vs radius 2.25 mm) — this is the biological root cause: a raw-space SmoothL1 on scales assigns ~156× more loss magnitude to length than radius, collapsing thin stalks to sub-pixel radii ("vanished peduncle", "porcupine" leaflet artifacts).

### 3.3 Latent Gradient Balance (frozen VAE, dL/dz at FM prior)

- Per-dim mean $|dL/dz|$ spread: **3.3×** (max dim $z_{11}$ = 1.65e-4, min $z_0$ = 5.1e-5), **CV = 0.38**.
- **16/16 latent dimensions alive** (all carry gradient > 1e-6) — no dead channels.
- Gradient spread tightens **19× vs raw 13D** (61× → 3.3×).
- Interpretation: at the FM prior — exactly where Stage-2 ODE integration starts — every latent coordinate receives a comparable-magnitude, non-zero learning signal through the frozen decoder, regardless of whether the organ is a 35 cm internode or a 2 mm peduncle stalk.

### 3.4 16D Latent Manifold Structure (PCA-2D)

- **74% of variance** captured in 2 PCs; **PCA-2D silhouette = 0.50** on 9 organ classes.
- Organ classes form separable semantic clusters (LEAF occupies a distinct PC region; reproductive classes cluster together) while the space simultaneously supports isotropic $\mathcal{N}(0, I)$ sampling — the property that makes latent Flow Matching well-conditioned.

---

## 4. Implications for the Architecture (validated)

1. **Class imbalance (71×)** + **scale unbalance (61× encoded / 156× physical)** jointly explain the historical raw-regression failure modes; the 16D latent + per-head decoder losses neutralize both by construction.
2. The dead $s_z$ tube channel disappears in latent space (VAE decoder emits softplus-positive scales for all organ classes).
3. The gradient probe confirms the Stage-2 training assumption: rendering/decode gradients injected at $z_0 \sim \mathcal{N}(0, I)$ are isotropic and dense, so velocity-field learning sees a stationary, balanced target (cf. Combination 2 ADR, `ongoing/20260908_cascaded_architecture_design_space_analysis.md`).
4. Roundtrip benchmark context (from `benchmark_organ_vae_roundtrip.py`, cited in the spec §3): IoU 74.2% → 91.9%, depth MAE 20.4 → 1.39 cm, scale MAE 1.82 → 0.24 cm (radius 0.04 cm), cls acc 89.1% → 100%, peduncle radius restored 2 μm → 2.25 mm.

---

## 5. Artifacts & Reproduction

| Artifact | Path |
| :--- | :--- |
| Figure (A–D panels) | `docs/results/assets/fig_organ_vae_sparsity_gradient_balance.png` |
| Measured summary JSON | `docs/results/assets/fig_organ_vae_sparsity_gradient_balance.json` |
| Generator | `scripts/generate_fig_vae_sparsity_gradient_balance.py` |
| VAE checkpoint | `diffusion_based/checkpoints/organ_vae/organ_latent_vae_best.pt` |
| Corpus | `dataset/cache/cowpea_curv26/` (10,000 shards) |

```bash
# Environment: mamba activate digital-crops; PYTHONPATH=. mandatory
PYTHONPATH=. python scripts/generate_fig_vae_sparsity_gradient_balance.py
```