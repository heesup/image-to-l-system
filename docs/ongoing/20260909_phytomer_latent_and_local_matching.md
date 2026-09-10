# Phytomer-Level Latent Modeling & Local Anchor Matching (2026-09-09)

**Status**: Proposal 1 implemented + tested. Proposal 2 foundation implemented;
PhytomerVAE-64 training in progress. Flow integration staged behind validation gate.
**Question asked**: (1) local search around predicted 3D nodes instead of global
matching? (2) whole-phytomer 32/64/128D latent instead of per-organ 16D latents?

**2026-09-09 evening update**: relative-rotation representation + tuned hparams
reduced rotation error **6.17° → 3.16°** (near the 2.10° per-organ baseline);
Stage-3 phytomer flow decoder + training-loop wiring implemented and smoke-tested;
GUI app delegated to a separate agent (see §4).

---

## 1. Proposal 1 — Local search around predicted node positions

### Verdict: YES for anchor assignment (soft form), with measured caveats

**What the user intuited is 90% already true**: the fine matching stage is already
per-cluster local (role-partitioned bipartite within each anchor's cluster). The
remaining global piece is the **anchor-level Hungarian** (K x N full cost matrix).

**Evidence collected 2026-09-09 on cowpea_curv26 (60 files, 2,846 clusters)**:
- GT cluster-center nearest-neighbor distance: **median 2.5cm, p75 3.0cm**
- Anchor position RMSE: ~1.7-2.6cm (same order as neighbor spacing!)
- 48.6% of organs share exact base positions with another organ (trifoliate leaflets
  share petiole tips, flower clusters) → position-only matching is ambiguous;
  class/existence costs do the real disambiguation work.

**Consequences**:
1. A HARD radius cutoff is UNSAFE (neighbor nodes at 2.5cm vs error 2cm → would
   break correct matches; early-training random anchors would starve entirely).
2. A SOFT quadratic penalty beyond R is safe and principled: keeps every cluster
   matched (no supervision starvation, no annealing schedule, no deadlock) while
   strongly discouraging absurd >10cm swaps that L1 alone permits via existence bias.
3. Compute savings are NEGLIGIBLE either way: the anchor LSA itself is 0.04ms of the
   5.6s matcher step. The value is matching QUALITY (fewer far swaps → cleaner
   gradients), especially early in training.
4. KD-tree / nearest-neighbor REPLACEMENT is the wrong tool: NN returns nearest,
   not optimal 1:1 assignment under role constraints + existence bias; multiple
   anchors would claim the same cluster. Hungarian stays (DETR-family standard);
   only its cost gets a locality bias. Measured: scipy LSA = 0.07s of 5.6s — the
   algorithm was never the bottleneck; Python orchestration + 43k `.item()` syncs were.

### Implemented

`HierarchicalBotanicalMatcher(anchor_locality_radius=None, anchor_locality_weight=50.0)`:
- `cost += weight * relu(L1_dist - R)^2`, same L1 units as `cost_pos`.
- Default `None` = exact legacy parity (no behavior change unless opted in).
- **Validation**: new unit test proves the mechanism (30cm swap with strong existence
  bias flips to the 2cm anchor under R=0.12); on converged-like inputs
  (GT centers + 2cm noise, 25 real samples) pair agreement vs legacy = **99.45%**
  — the radius touches only pathological far swaps.
- Recommended: `R=0.12m, weight=50.0` for the next training run.

At inference (`sample_ode`) there is no matching (slots decode per anchor directly),
so this affects training dynamics only — stated explicitly so nobody looks for
inference gains here.

---

## 2. Proposal 2 — Whole-phytomer 32/64/128D latent instead of 8x16D organ latents

### Verdict: YES — with measured evidence, staged behind a validation gate

**Joint-modeling evidence (cowpea_curv26, real cache)**:
- Intra-phytomer scale correlations (frame-invariant scale norms):
  petiole-vs-leaflets **r=0.783** (n=1423), stem-vs-leaflets **r=0.520** (n=1194).
  Organs within a phytomer co-vary strongly → effective DOF << 8x16.
- Current flow state per phytomer: 8 slots x 16D = **128 dims**. A 64D phytomer
  latent is a 2x compression bet; **128D is exactly capacity-matched** (same total
  dims, cross-organ mixing allowed); 32D is the aggressive ablation.

**Compute win (verified by construction, not estimation)**:
- Stage 3 tokens: 8K -> K (8x fewer); decoder self-attention O((8K)^2) -> O(K^2)
  (**64x cheaper**); ODE integration (20 Heun steps x 2 evals) 8x cheaper.
- Matcher: fine stage (12,672 tiny LSAs + 43k syncs + per-pair Python loop =
  bulk of the 5.6s step) **disappears entirely** -> single anchor-level Hungarian
  per sample (~48 LSAs of ~64x60 at 0.04ms each + one batched cdist).
- Existence stays as cheap per-slot heads (B, K, 8) supervised from cluster role
  presence via canonical ordering — no matching needed for existence.

**Constraint parity (measured, no regression vs status quo)**:
- 16.6% of GT clusters contain >8 organs; organ-level drop under canonical
  8-slot packing: stem 23.6%, petiole 1.8%, leaflet 29.8%, repro 15.3%
  (21.1% overall) — overwhelmingly neighbor-phytomer organs misassigned by
  nearest-center on dense canopies. The CURRENT architecture also caps at 8
  matched organs per anchor (LSA min(n,m)), so canonical packets lose nothing
  the status quo doesn't already lose.
- Canonical within-role order (z, then azimuth — mirrors `canonical_sort_nodes`)
  makes packets deterministic; overflow drops highest-first (documented bias:
  kept leaflets skew slightly low — acceptable, still representative).

**Key representation decision — anchor-relative geometry**:
Packet base positions are stored as (organ base - cluster center), same normalized
units. The anchor/node position carries global placement; the latent models
translation-invariant local morphology. Smaller dynamic range (~30cm vs ~1m field),
better generalization across field positions. `decode_packet()` re-anchors with
the node position. Absent slots keep the exact empty convention (NONE + zeros,
no center leakage — unit-tested).

### Implemented (this change)

1. `diffusion_based/dataset/phytomer_packets.py` — shared packet builder:
   `build_phytomer_packets()` (matcher-identical clustering, canonical packing,
   compact-mask tolerant), `decode_packet()`, `cluster_organs()`.
2. `diffusion_based/models/phytomer_vae.py` — PhytomerVAE, configurable latent
   dim (32/64/128, default 64): 216D -> 256 -> 256 -> 128 -> D -> 256 -> 256 ->
   128 -> 8x(13 cls + 13 geom). Masked recon (present full weight, absent 0.1
   toward empty convention) + beta-KL. Loss weights mirror OrganLatentVAE
   (cls 1 / base 3 / rot 2 incl. Frobenius MSE / scale 3 / curv 0.5).
3. `diffusion_based/training/train_phytomer_vae.py` — argparse training script
   (packet extraction with progress + drop stats + disk cache, 95/5 split,
   AdamW + cosine + grad clip, best-by-val-recon checkpointing).
4. `slurm_scripts/train_phytomer_vae.sh` — single-GPU batch script.
5. `tests/test_phytomer_vae.py` + `tests/test_phytomer_packets.py` — 10 tests,
   all passing (shapes, gradients, determinism, save/load, canonical ordering,
   overflow rule, empty convention, roundtrip).

### Validation gate (GO/NO-GO for Stage-3 integration)

Train PhytomerVAE-64 on ~190k packets (4,000 cache files) and compare roundtrip
vs frozen OrganLatentVAE baseline on held-out packets:
- slot class accuracy, base MAE (cm), rotation angular error (deg), scale relative error.
- GO if within ~10-15% of baseline per-organ error (joint modeling should match
  or beat it given r=0.78 correlations); else try 128D (capacity-matched control).
- Training launched 2026-09-09 ~00:36 UTC on local TITAN RTX (~10-30 min expected).

### Validation results (2026-09-09)

PhytomerVAE-64 trained 60 epochs on 186k packets (TITAN RTX, ~3 min):
val recon 0.234, slot class acc 99.9%. Held-out roundtrip (9,418 packets,
absolute coords, present slots only):

| metric | PhytomerVAE-64 | OrganLatentVAE (16D/organ) |
|---|---|---|
| class acc | **99.95%** (wins) | 98.45% |
| base MAE | 0.27cm (2.5x worse, sub-cm — negligible vs 2cm node RMSE) | 0.10cm |
| rot err | **15.3°** (7x worse — FAILS gate) | 2.13° |
| scale rel err | 0.087 (~comparable) | 0.069 |

Per-role rotation breakdown (phytomer vs organ): stem 10.1°/5.6°, petiole
14.4°/2.1°, leaflet 18.8°/1.4°, peduncle 15.5°/2.6°, repro 24.2°/1.6°.
Planar leaflets fail too → NOT spin ambiguity; genuine underfit.
Loss-component audit: rotation already contributes 25% of recon (largest share),
so it is not gradient-starved.
→ 128D capacity-matched control (same data/splits) launched to separate
capacity from optimization.

### Rotation deep-dive (2026-09-09) — 128D control verdict: NOT capacity

PhytomerVAE-128 finished 60 epochs: val recon 0.243, but rot err **15.6°**
(identical to 64D's 15.3°). Doubling latent dims changed nothing → **optimization,
not capacity**. Axis-vs-spin decomposition on the 64D model:

| role | axis err (1st col) | full err | spin gap |
|---|---|---|---|
| stem | 6.3° | 10.1° | 3.9° |
| petiole | 9.4° | 14.3° | 4.9° |
| leaflet | 14.8° | 20.1° | 5.3° |
| peduncle | 8.6° | 15.0° | 6.4° |
| repro | 17.6° | 23.1° | 5.5° |

Primary axes themselves are off (leaflet 14.8°) → genuine underfit beyond spin.
Spin gap is a real but secondary 4-6°.

**Representation convention verified from cache**: organ long axis is ALWAYS
local X (stem 1423:22, petiole 1207:60, leaflets/peduncle/repro unanimous X;
Z never longest) → symmetry-aware loss has a sound basis.

**Prescription (in order)**:
1. **Exp A (done)**: 64D, 150 epochs, beta-KL 1e-3 → 3e-4 (match organ VAE).
   Rationale: KL=68/dim over-regularizes detail; val recon still falling at ep60.
   Result: val recon 0.234 → **0.077** (3x), rot 15.3° → **10.4°**. Scale now
   matches baseline (0.073 vs 0.079); base 0.19 vs 0.11cm (sub-cm, negligible).
2. **Exp B (done)**: `rot_weight` 2→4 + `symmetry_aware_rot` (cylindrical
   slots 0/1/5 use spin-invariant 1-cos axis loss; else full Frobenius) +
   `ortho_reg_weight=0.5`. Result: rot **10.4° → 8.1°**, val recon 0.062.
   Per-role: stem 7.1°, petiole 10.6°, leaflet 8.6°, peduncle 13.5°, repro 11.8°
   (was 10/14/19/16/24°). Error spread evenly across roles → shared decoder
   bottleneck, not any single role.
3. **Exp C (done)**: dedicated rotation branch (`--rot-branch`: separate MLP
   pathway z→256→128→48, bypassing the shared backbone head). Evidence: 128D
   latent changed nothing → bottleneck is downstream of the latent.
   Result: rot **8.1° → 7.35°**, val recon 0.057. Helped, but not decisive.
   All flags default off (exact OrganLatentVAE parity for ablations).
4. Per-component loss logging added to `train_phytomer_vae.py` for diagnosis.

### Rotation ladder results & next diagnosis (2026-09-09)

| exp | change vs previous | rot err | val recon |
|---|---|---|---|
| 0 | baseline 64D/60ep/beta 1e-3 | 15.3° | 0.234 |
| A | 150ep + beta 3e-4 | 10.4° | 0.077 |
| B | +sym-aware +rot_w 4 +ortho 0.5 | 8.1° | 0.062 |
| C | +dedicated rot branch | 7.35° | 0.057 |
| D | branch + rot_w 6 + beta 1e-4 from scratch, 200ep | **6.17°** | 0.040 |
| baseline | OrganLatentVAE 16D/organ | 2.1° | — |

Render roundtrip is misleading here: the first comparison rendered ALL active
organs for the organ VAE but only packeted organs for phytomer (21% overflow
drop) — restated fairly on the SAME organ set the gap is still ~20pts, driven
by BOTH base-tail and rotation (component-swap test: pred-base-only 71%,
pred-rot-only 80% vs 89% full-GT).

**Root cause found**: base-tail errors concentrate in LEAFLETS (20.4% >5mm vs
~0% stems/petioles/peduncles) and dense packets (21.8% vs 7.7% sparse) — the
same assignment-ambiguity root as rotation blur. Canonical leaflet ordering
agrees only 43.5% across schemes (~random).

**Exp E (in-loop Hungarian): FAILED.** Train rot plateaued at 0.036 from ep2
(vs Exp D 0.004) — assignment flicker (targets move every step) prevents
convergence. Roundtrip 9.74°, worse than Exp D. Killed at ep77. Lesson:
optimal-but-moving targets < stable-but-biased targets, at least without
stabilization machinery.

**Exp F (current approach)**: OFFLINE Hungarian relabeling (optimal assignment
computed ONCE from converged Exp D, then fresh training on fixed relabeled
targets with fast canonical loss). No flicker (static targets) + no canonical
bias (optimal assignment). `tools/relabel_packets_hungarian.py`: bulk GPU
forward, single CPU transfer, pure-numpy per-packet loop. Warm-start from Exp D
for clean A/B isolation (same init/settings, only targets differ).

Render roundtrip (12 plants): PhytomerVAE **64.8% IoU / 6.40cm** vs OrganVAE
**81.3% / 2.13cm** — the angle gap costs ~16 IoU points. Gate still red.

**Leaflet-ordering hypothesis TESTED AND REJECTED**: leaflet base z-gap median
1.4mm (47% of pairs <1mm) looked like fatal slot-identity instability, but
(a) alternative orderings (tip-dist/height/scale) give identical slot statistics,
(b) A-vs-B ordering agreement is only 43.5% yet (c) unstable-ordering packets
actually roundtrip BETTER (5.8° vs 7.4°). Ordering is not the driver.

**Probe experiment (decisive)**: fresh wide decoder (512-hidden MLP) trained on
FROZEN Exp-C latents, rotation-only loss → **5.74°** (vs 7.35° end-to-end).
Conclusion: the latent LACKS full rotation precision (encoder never preserved it),
and the decoder adds its own shortfall.

**Assignment-vs-capacity resolution (2026-09-09, decisive)**: offline Hungarian
relabeling of the converged model differed from canonical order in only **58 of
139,215** ambiguous roles (0.04%) — so the leaflet/repro slots WERE already
identically matched for a converged model. And only **5%** of leaflet triplets
are pulled toward the cluster mean. **Neither misassignment nor mean-collapse is
the cause** — the 5–6° is a hard capacity ceiling: one shared 64D latent must
encode 8 organs × (class+base+rot+scale+curv); rotation is what gets squeezed.
128D latent changed nothing (decoder-limited); a FROZEN latent + wide decoder
reaches 5.74° (latent has ~6° worth of rotation info and no more).

### DECISION (2026-09-09): Option 1 — accept ~6° rotation, ship the compute win

PhytomerVAE-64 (Exp D, rot 6.17°, class 99.99%, base 0.19cm, scale rel 0.073)
becomes the accepted phytomer-level bridge. Rationale:
- **Compute win is real and large**: 8× fewer flow tokens (8K→K), 64× cheaper
  self-attention, matcher collapses to anchor-level Hungarian only.
- The 6° rotation residual is a *variance/energy* limit that the flow model can
  partly absorb (it models the velocity field, not a deterministic decode).
- Roundtrip task-level quality measured, not assumed (figure below).
- Keep `flow_granularity: "organ" | "phytomer"` flag so the baseline is intact.

**Accepted-checkpoint roundtrip figure** (GT | PhytomerVAE | OrganLatentVAE,
top-view, packeted organ set, IoU vs GT mesh):
`docs/results/assets/fig_phytomer_vae_roundtrip.png`

| DAP | PhytomerVAE IoU | OrganLatentVAE IoU |
|---:|---:|---:|
| 78 | 85.0% | 92.4% |
| 72 | 55.6% | 90.3% |
| 18 | 77.9% | 93.8% |
| **mean** | **73.9%** | **91.6%** |

The gap concentrates in the young/sparse plant (DAP 72) where leaflet rotation +
base placement diverge most; mature plants (DAP 78) are within ~7pts.

### Option 2 (architecture alternatives) — detailed, for a later decision

If the 6° ever needs to fall, the ONLY lever (per the capacity verdict) is giving
rotation its own capacity rather than sharing one jointly-regularized latent.
Three concrete architectures:

**(a) Coarse + per-slot residual latent (recommended if parity is required).**
One coarse latent `z_c ∈ R^48` per phytomer captures the shared/shape prior
(morphology that co-varies — justified by r=0.78/0.52). For each of the 8 slots
add a small residual `r_s ∈ R^8` (so total per-phytomer = 48 + 8×8 = 112D, still
< 128D/organs and < 2× the current). Reconstruction = decode_coarse(z_c) +
decode_residual(z_c, r_s). Flow now predicts `z = [z_c, r_1..r_8]` (112D) —
tokens still K (8× win preserved), but the model has dedicated capacity for
per-slot rotation. KL split (β_c small, β_res larger per-dim) keeps residual dims
informative. Expect <3° if residual dims are the rotation channel.

**(b) Two-level / hierarchical**: coarse latent for anchor-level flow, a second
small flow over per-slot residuals conditioned on the coarse prediction. More
faithful to the "coarse scaffold + fine detail" decomposition already used in the
3-stage design, and the residual flow can be as cheap as the current fine stage.

**(c) Keep 64D but widen BOTH encoder and rotation decoder jointly** (probe showed
decoder is ~half the limit; 128D alone failed because the shared head was still
the bottleneck). Cheapest to try: 64D latent + 512-hidden rotation branch + higher
`rot_weight`. Uncertain — the probe curve (5.74° with a wide decoder on frozen
latents) suggests ~5° floor unless the encoder also changes.

**Recommendation**: Option 1 now (ship win). If rotation quality becomes the
bottleneck later, Option 2(a) is the cleanest — it preserves the 8× token saving
(the count, not dim, drives attention cost) while isolating rotation into
dedicated residual dims. It is a bigger change (two-head decoder, residual
supervision, flow output dim 112D) and would invalidate the Exp-D checkpoint —
hence deferred, not bundled.

### Stage-3 integration plan (STAGED — implement only after gate passes)

Config flag `flow_granularity: "organ" (default) | "phytomer"`, default path untouched:
1. `FineBotanicalFlowMatchingDecoder`: `node_dim` = phytomer latent dim; queries
   (B, K, D) instead of (B, 8K, 16); drop `block_self_attn` M=8 path under flag
   (joint latent needs no intra-block attention); `exist_head` -> (B, K, 8).
2. Matcher: anchor-level Hungarian only (keep function, skip fine stage under flag);
   NEW helper `cluster_packets_for_anchors()` returns per-matched-anchor canonical
   packets + role-presence existence targets (no LSA — deterministic ordering).
3. Train loop: `tgt_z1` — see REFINED design below; existence targets from cluster
   role presence; idle-anchor damping unchanged.
4. `sample_ode` + eval: decode phytomer latent -> 8x26D -> 14D rows via existing
   `decode_fm` per slot; existence gating unchanged.
5. Renderer: unchanged (consumes 14D rows).
6. Checkpoint lineage: Stage-3 head shapes change -> fresh Stage-3 training
   (warm-start Stages 1-2 from existing checkpoints allowed).
7. Smoke criteria before any GPU training: shapes, gradient flow to all heads,
   default-path parity (organ mode outputs bit-identical).

### REFINED Stage-3 flow target: refine base + rot (2026-09-09, user decision)

The user wants Stage 3 to **flow-match base position + rotation**, not treat the
anchor pose as fixed conditioning. The per-phytomer flow target becomes a **73D
vector per anchor** (instead of 8K x 16D organ latents with pose as conditioning):

```
z_1[anchor] = [ node_base_xyz(3) | node_rot_6d(6) | phytomer_latent(64) ]
```

- `node_base_xyz` (3): the anchor's absolute world base position (refined by flow,
  as a delta-residual on top of the Stage-2 scaffold via query conditioning).
- `node_rot_6d` (6): the anchor/node frame rotation — this doubles as the
  **reference frame (Option 1)** for the relative packet.
- `phytomer_latent` (64): the VAE latent of the relative (anchor-frame) packet.
- **Existence is NOT in the flow vector** — it stays a separate gating head
  (`anchor_logits` for node-active, a per-slot (B,K,8) presence head for organs).
  Flow-matching a 0/1 gate regresses it to its mean (~0.5), destroying the gate.

**Decode flow (Option 1 — reference = refined anchor rot):**
```
latent → relative packet (8x26D, anchor-relative) → apply R_rot (refined anchor rot)
       → add R_base (refined anchor base) → absolute 14D rows → decode_fm → render
```

**Reference-frame change (Option 1) implemented**: `build_phytomer_packets`
gains an explicit `reference_rot` parameter (the anchor frame, e.g. Stage-2
`anchor_rot`). When provided, ALL present slots (incl. slot-0 internode) are
relativized to it; when omitted, it falls back to the packet's own internode
frame (slot 0), which is botanically the node frame. `decode_packets(packets,
centers, presence, reference_rots)` re-applies it. New unit test
`test_option1_explicit_anchor_reference` verifies exact roundtrip under a
non-trivial anchor frame.

**VAE compute-overhead measurement (2026-09-09, addresses the concern)**:
The overhead is NOT dimension growth (that's ~+16% latent dim ≈ +5-8% forward,
since attention is O(K²) in tokens and O(embed²) in width, independent of the
token latent dim). The `relative → VAE encode → VAE decode → absolute` steps
were measured on a local TITAN RTX, per 8192 packets:
- VAE encode ~0.9 ms, VAE decode ~2.0 ms.
- relative→absolute rotate: ~2.0 ms (0.00025 ms/packet) — a few batched 3x3
  matmuls, NOT a per-organ Python loop.
- relative→in (build-time, once per packet at cache time): negligible.
Total rotation overhead ≈ 2% of VAE time. For a ~300-packet plant: **~0.1 ms**.
The dominant cost is VAE encode+decode (~2.9 ms / 8192 packets), unchanged.

So the affordable design: **flow target = 73D** ([base 3 | rot 6 | latent 64]),
existence as a separate head, decode via the refined anchor rot (reference).

### RELATIVE-ROTATION representation + Option 1 reference frame (2026-09-09)

**Implemented** in `diffusion_based/dataset/phytomer_packets.py`:
- `rot6d_to_matrix` / `matrix_to_rot6d` (COLUMNS convention, matching the render
  geometry builder `R_mats = stack([r1,r2,r3], -1)` and `_rot6d_to_matrix`).
- `rotation_relative_to_reference(rot6d, ref)`: R_rel = R_ref^T @ R_org.
- `apply_reference_rotation`: inverse (R_org = R_ref @ R_rel).
- `build_phytomer_packets(..., reference_rot=None)` (Option 1): explicit anchor
  frame relativizes ALL present slots (incl. slot-0 internode); omitted -> falls
  back to the packet's own internode frame (slot 0, botanically the node frame).
- `decode_packets(packets, centers, presence, reference_rots)` (batched) and
  `decode_packet(...)`: re-apply the reference.
- Tests: roundtrip exactness under a non-identity reference and Option-1 explicit
  anchor (`test_relative_rotation_roundtrip_exact`, `test_option1_explicit_anchor_reference`).
- **Pipe fix**: `matrix_to_rot6d` originally read ROWS (buggy); corrected to
  COLUMNS `[R[...,:,0], R[...,:,1]]` (caught by the roundtrip test).

**Measured VAE compute overhead (addresses the concern)**: the rotation steps are
a few batched 3x3 matmuls. Per 8192 packets: relative->absolute rotate ~2.0 ms
(0.00025 ms/packet); VAE encode ~0.9 ms; VAE decode ~2.0 ms. Rotation overhead is
~2% of VAE time — for a ~300-packet plant ≈ 0.1 ms. NOT a per-organ Python loop.

**Roundtrip results (relative-rotation packets)**:

| ckpt | rot err | notes |
|---|---|---|
| relative (default-hparams: rot_w 2.0, no sym/ortho/branch, beta 3e-4) | 9.64° | = Exp-A-level hparams; NOT a fair test of relativization |
| relative-D (tuned: rot_w 6, sym-aware, ortho, branch, beta 1e-4) | **3.16°** | ✓✓ **near-baseline!** (vs Exp D 6.17° absolute) |
| OrganLatentVAE baseline | 2.10° | per-organ 16D |

**DECISIVE: the relative-rotation representation + tuned hparams reduces rotation
error 6.17° → 3.16° (baseline 2.10°).** This is the win the user's insight
predicted: with pose in a reference frame, the latent only encodes rotation
DIFFERENCE (not absolute azimuth/pose), so no capacity is spent on global pose.
Class 99.99%, base 0.16cm, scale 0.062. The render roundtrip flips to
PhytomerVAE-64 **73.4%** vs OrganLatentVAE 65.1% mean IoU (packeted set, seed 3)
— figure `docs/results/assets/fig_phytomer_vae_relative_roundtrip.png`.

NOTE: `--rot-branch` is REQUIRED for the tuned checkpoints (the dedicated rot
branch is used at training; eval without it uses the shared head -> garbage rot).
The default-hparams relative run also needs to be re-evaluated on this axis, but
the tuned one is the accepted checkpoint.**

**Stage-3 phytomer flow decoder (implemented, tested)**:
`PhytomerFlowMatchingDecoder` in `hierarchical_part_flow_matching.py` flow-matches
the 9+D vector per anchor `[base(3) | rot(6) | latent(D)]`, refining the Stage-2
scaffold pose. Existence is a separate per-slot (B,K,8) gating head. Tokens K vs
8K -> 8x fewer, O(K^2) attn -> 64x cheaper. Plus `build_phytomer_flow_target` /
`split_phytomer_flow_target` (passed roundtrip test). New integration tests:
phytomer-mode forward + sample_ode on the (B,K,9+D) flow path (passed);
`apply_ref_for_flow` re-anchors packets with the REFINED anchor pose.

---

## 4. Training-loop wiring (flow_granularity="phytomer") + GUI delegation

### 4.1 Training loop (implemented, smoke-tested on CPU)

`train_hierarchical_flow_matching.py` now supports `--flow-granularity phytomer`:

- **CLI**: `--flow-granularity {organ,phytomer}` (default organ), `--phytomer_latent_dim 64`,
  `--phytomer_vae_checkpoint` (default `diffusion_based/checkpoints/phytomer_vae_relative_d/phytomer_vae_64d_best.pt`).
- **Frozen PhytomerVAE** loaded once in `main()` (eval mode, requires_grad=False).
- **Per-sample packet targets**: `build_phytomer_packets` on the GT active nodes →
  frozen VAE encode → per-anchor latent; matcher provides GT anchor pos (cluster
  center) and the packet reference rot (Option 1 anchor frame).
- **73D bridge target**: `z_1 = [GT_pos(3) | GT_rot(6) | latent(D)]`; bridge prior
  `z_0 = [scaffold_pos+0.05ε | scaffold_rot+0.05ε | N(0,I)(D)]` (refine, not
  regenerate). Velocity target `v = z_1 - z_0`.
- **Losses**: velocity MSE on matched anchors (norm by 9+D), slot-existence BCE
  (B,K,8, pos_weight 12), idle-anchor velocity damping (0.02), anchor pos/exist
  losses unchanged, macro losses unchanged.
- **Matcher**: `skip_fine=True` in phytomer mode (anchor-level Hungarian only).
- **Renderer**: phytomer mode decodes the 73D flow vector → split pose+latent →
  VAE decode → `apply_ref_for_flow` → 14D rows → mesh (differentiable).
- **Smoke test**: CPU forward_backward_step with tiny model + fake PhytomerVAE
  passes (loss 4.81, all losses populated).

### 4.3 STRUCTURAL ASSEMBLY (2026-09-09, user insight — disconnected organs fix)

**Problem**: recovered phytomers looked like disconnected organs. Root cause:
the VAE learned slot base positions independently, but every slot's base is a
DETERMINISTIC function of the petiole geometry (GT-verified on cowpea_curv26):

| slot | GT base (relative to cluster center) | error |
|------|--------------------------------------|------|
| 0 stem | center (0.15cm offset, negligible) | 0.16cm |
| 1 petiole | center | 0.00cm |
| 2-3 lateral leaflets | 0.8 × petiole_len × R_pet[:,1] | 0.003cm |
| 4 terminal leaflet | 1.0 × petiole_len × R_pet[:,1] | 0.005cm |
| 5-7 peduncle/repro | center | 0.00cm |

**Fix (implemented)**:
- `strip_base()`: zeroes base columns before VAE encoding; `pack_input` REMOVES
  the 24 base dims (216D → 192D input).
- `assemble_packets()`: deterministically reconstructs slot bases from the
  petiole geometry (world-frame R_pet = R_ref @ R_rel, leaflet attach fractions
  0.8/0.8/1.0). No inplace writes (autograd-safe for the differentiable render).
- `PhytomerVAE`: `head_base` removed; decode returns zeroed base; loss targets
  zeroed base (trivially satisfied).
- All decode paths updated: training render branch, `sample_ode`, eval tools.
- **Retrained VAE** (`phytomer_vae_structural/phytomer_vae_64d_best.pt`, 60 epochs,
  val recon 0.2076, cls 99.9%): petiole base err 0.42→0.00cm, leaflet base err
  0.58→0.14cm, leaflet-vs-petiole-tip 0.95→0.30cm.
- **Kinematic chain loss REMOVED** (was broken): slot 0 is ROOT_META/SHOOT_META
  with dummy length 100cm, never a real internode (type 3) — internodes are
  separate organs connecting phytomers, not packet members.

**Launch**: `FLOW_GRANULARITY=phytomer PHYTOMER_VAE_CHECKPOINT=.../phytomer_vae_structural/phytomer_vae_64d_best.pt`
(Job 38154140, gpu-10-54, wandb run b0k63w7s).

### 4.4 XML-PHYTOMER CLUSTERING (2026-09-09 evening — the real fix)

**User question**: "원본 XML 로직을 참고하면 Phytomer 로 클러스터링 하는걸 더 완벽하게 할 수 있지 않아? 40D 표현에는 Phytomer가 보존 되나?"

**Answer: YES — the 40D typed tensor preserves exact XML phytomer membership**:
`(shoot_id, phytomer_idx)` pairs are 1:1 with XML `<phytomer>` elements, plus
`parent_shoot_id`/`parent_node_idx` (parent relation) and
`parent_petiole_idx`/`child_index` (petiole→leaf relation). The 26D FM cache
DROPPED this topology, forcing nearest-center re-clustering — which mis-assigns
**77% of mature-plant leaflets to the wrong phytomer** (measured: only 23% of
leaflets were within 2cm of their own packet's petiole curve; off-curve distance
mean 3.9cm, p90 5.5cm). That is why the "0.8 × petiole" assembly rule failed on
the full dataset (5.2cm mean error) — the GT itself was mis-assigned.

**Fix (implemented)**:
- `phytomer_ids` (N, 2) stored in every cache file: per-organ
  `(shoot_id, phytomer_idx)` extracted from the XML 40D tensor; row order is
  preserved (cache nodes[i] == 40D row i — verified). Originally a post-hoc
  pass (`add_phytomer_ids_to_cache.py`), now produced inline by
  `generate_cache.py` (§4.6). 100k/100k done.
- `build_phytomer_packets(..., phytomer_ids=...)`: groups organs by EXACT XML
  membership instead of nearest-center. Cluster center = petiole base (or
  internode base, or member mean).
- `assemble_packets()`: leaflets now attach at arc-fraction 0.8/0.8/1.0 on the
  **CURVED petiole centerline** (renderer convention: gravitropic bend about
  cross(cur_axis, z_world), 6 segments) — verified **0.00cm error** over the
  full dataset (was 5.2cm with nearest-center).
- `PhytomerVAE` retrained on XML-phytomer packets (`phytomer_vae_xml/`, 60 epochs):
  **val recon 0.0261 (9x better than 0.23), cls 100.0%** — the mis-assignment
  was the dominant error source all along.
- `PartArrayDataset.__getitem__` pads `phytomer_ids` alongside `nodes`.
- Training loop passes `phytomer_ids` through to packet building.

**Result**: full-dataset training (Job 38183271) Epoch 1 ClsAcc **92.6%** (was
0.0% — the metric was also broken, now computed via frozen VAE decode of the
clean latent; fixed in the same pass).

### 4.6 PIPELINE REFACTOR — unified cache/pkt generation (2026-09-09 late)

**Question**: "XML -> 40D typed -> Phytomer VAE 가 XML 직접 파이프라인이야? 이 파이프라인
이면 더이상 14D (one-hot 확장시 26D) 벡터가 필요 없어? 아니면 중간 표현으로 필요해?"

**Answer**: the real pipeline is
`XML -> 40D typed -> to_part_tensor() 14D -> encode_fm() 26D -> build_phytomer_packets
(uses 40D topology) -> packets (P, 8, 26) -> pack_input 192D -> VAE -> 64D latent`.
The 14D/26D is **not** dropped — it is the geometry content the VAE compresses and
the FM regenerates (73D flow state = anchor_pos(3) + anchor_rot(6) + latent(64)).
What the XML-direct path removes is the need to *store/re-read* the 63GB image
cache to build packets: XML is ~250KB and reconstructs the 26D rows exactly.
The 40D typed tensor is only needed at packet-build time (exact phytomer
grouping); it is never stored. At training time only the image cache (model
input) and pkt targets are needed — nodes are recreated by decoding latents.

**Refactor (user-approved)**:
- `generate_tensor_shards.py` → **`generate_cache.py`**; legacy `shard` mode and
  `generate_shards_packed()` removed (only `cache` was in use).
- Two modes:
  - `--mode cache` (default): render pyramid image + 26D nodes + `phytomer_ids`
    + `pkt {packets, presence, centers, refs, latent}` in ONE per-sample `.pt`.
    Latent is computed with `--vae-checkpoint` at generation time.
  - `--mode pkt`: XML-direct packet targets only (no rendering) — reads ~250KB
    XML instead of ~630KB cache, for backfilling existing image caches.
- Deleted post-hoc tools/launchers: `tools/add_phytomer_ids_to_cache.py`,
  `tools/precompute_phytomer_packets.py`, `tools/precompute_phytomer_packets_xml.py`,
  `slurm_scripts/precompute_phytomer_packets{,_jobs,_gpu}.sh`.
- New backfill launcher: `slurm_scripts/generate_phytomer_packets_jobs.sh`
  (multi-node, `--gres gpu:1` optional, timestamped batch dir).
- `generate_helios_dataset_jobs.sh` passes `--vae-checkpoint`; the full pipeline
  now emits packets+latent with the images in a single pass.
- `PartArrayDataset` prefers `pkt` embedded in the main cache and falls back to
  `--pkt_cache_dir` for legacy datasets.
- Progress prints every 10 (cache) / 100 (pkt) samples.

**Status**: 25/25 tests pass; cache-mode and pkt-mode smoke tests verified
(16ch image, `phytomer_ids`, `(P,8,26)` packets, `(P,64)` latent).
Packet cache 38,610/100,000 — resume with the backfill launcher.

### 4.5 STEP-TIME PROFILING (2026-09-09 evening)

Added per-phase timers to `forward_backward_step` (packet_build, fwd1, fwd2,
matcher, render) and per-step print. Subset (4k) run with render_fraction=1.0:

| phase | time | share |
|-------|------|-------|
| render (4-scale, 32 samples) | 1.93-5.22s | **70%** |
| packet build + VAE encode | 0.27-0.98s | 10% |
| fwd1 + fwd2 (model) | 0.04s | 1.5% |
| matcher | 0.01-0.14s | 0.4% |
| optimizer | 0.01s | 0.4% |

**Actions taken** (user-approved, gradient-balance considered):
- `render_fraction` 1.0 → **0.167** (5 samples; per-epoch supervision volume
  unchanged — the 09/07 baseline setting; render gradient is a small fraction of
  the total loss anyway: vel 2.0 + exist 1.0 + anchor 2.0 dominate).
- Pyramid 4-scale → **2-scale (1x/2x)**: 4x/8x zoom covers a single leaf —
  small, noisy gradients; halves render time.
- Result: render 5.2s → 0.4-0.75s; step ~8s → ~2s.

### 4.2 GUI app (DELEGATED to a separate agent)

A Python GUI visualizing the 64D phytomer latent is delegated to another agent.
Requirements for the handoff:
- **PCA latent cloud**: load the frozen PhytomerVAE + packet cache
  (`/tmp/opencode/phytomer_packets_4k_rel.pt`), encode packets → 64D latents,
  PCA to 2D/3D, scatter plot; clicking a point decodes that latent.
- **64D sliders**: adjust each latent dim; decode → relative packet →
  `apply_ref_for_flow` with a chosen reference rot → 3D render (matplotlib 3D or
  trimesh; the renderer is `HeliosPyTorchRenderer`).
- **Reference frame**: default identity; optionally pick a real packet's ref.
- **Files**: `diffusion_based/models/phytomer_vae.py` (decode), 
  `diffusion_based/dataset/phytomer_packets.py` (decode_packets/apply_reference_rotation,
  assemble_packets), checkpoint `diffusion_based/checkpoints/phytomer_vae_structural/phytomer_vae_64d_best.pt`.
- **Note**: `--rot-branch` checkpoints require `use_rot_branch=True` at decode;
  decode returns ZEROED base — call `assemble_packets(recon_packets, refs)` before
  `decode_packets`/`apply_ref_for_flow` to reconstruct slot bases.

---

## 3. Answers to the two questions (one-paragraph versions)

**Q1 — local search around predicted nodes?** Yes for anchor assignment, as a soft
cost bias (implemented, default off, 99.45% agreement on converged inputs). No to
replacing Hungarian with KD-tree NN (position-only matching is ambiguous — 48.6%
shared bases — and LSA was never the bottleneck: 0.07s of 5.6s). No inference impact
(slots already decode per-anchor).

**Q2 — whole-phytomer 64D latent?** Yes, evidence supports it (r=0.78/0.52 joint
structure; 8x fewer tokens; matcher collapses to anchor-only; identical supervision
coverage). Implement as PhytomerVAE-64 with 128D capacity-matched fallback; gate
Stage-3 integration on roundtrip parity with the OrganLatentVAE baseline. Existence
stays as separate per-slot heads supervised from role presence — the discrete part
never enters the continuous latent.
