---
title: "Session Continuation: Hybrid PhytomerVAE Rotation Capacity, Render-Loss Internode Fix, Canary-Only Gradient Guard, and the Next Topology Experiment"
date: 2026-09-11
tags: [engineering, implementation]
status: active
---

# Session Continuation: Hybrid PhytomerVAE Rotation Capacity, Render-Loss Internode Fix, Canary-Only Gradient Guard, and the Next Topology Experiment

- **Author**: Claude (pair programming with Heesup Yun)
- **Date**: 2026-09-11 (continuation, later same day)
- **Status**: PhytomerVAE v8 trained and validated | pkt cache regenerated (v6, with `keys`) | render-loss internode fix committed and smoke-tested clean | gradient guard made canary-only | **NEXT**: cheap topology-recovery measurement (no training) to decide between (b) shrinking Stage 2's rotation head to a 1D roll, (c) a learned graph/PAF connectivity mechanism, or the refined (b)/(c) hybrid in §6.1 -- Stage 2 rotation+scale as an explicit parent-relative PDF instead of a point estimate, giving Stage 3's flow an adaptive per-node bridge-noise scale and `chain_phytomers` a likelihood-weighted cost instead of one fixed hyperparameter
- **Supersedes**: [`20260911_takeover_grad_norm_fix_roundtrip_diagnosis_10slot_restore.md`](../20260911-grad-norm-fix-roundtrip/20260911-grad-norm-fix-roundtrip.md) for the PhytomerVAE checkpoint lineage (that doc's "MUST be retrained from scratch" instruction is now satisfied by v8 below)

---

## 1. Executive Summary

Picking up from the previous takeover doc's instruction ("PhytomerVAE MUST be retrained from scratch"), this continuation:

1. **Redesigned PhytomerVAE's latent as hybrid coarse+residual** (commit `4512926`) instead of a single shared 64D/128D bottleneck, specifically to fix the ~12-13° rotation error the single-head simplification (previous doc §7.9) produced. Root cause (established across many prior ablations, not re-litigated here): a shared latent's KL budget gets spent on strongly-correlated affine shape (petiole-vs-leaflet scale r=0.78) at rotation's expense, because rotation has no comparable cross-organ correlation to share capacity with.
2. **Trained the result as `phytomer_vae_v8`** (128D = 48 coarse + 10×8 residual): rotation error **0.4-3.3°** across DAP 1-100 (packet-level, no Helios) — matching or beating the best prior tuned result (3.16°) and close to the per-organ baseline (2.10°).
3. **Found and fixed a real bug**: `generate_cache.py --mode pkt` (the fast multi-node backfill path) never saved the `(shoot_id, phytomer_idx)` `keys` field, so `loss_phytomer_order` and `loss_stem_dir` silently stayed at exactly 0 for any run reading from that cache — a fourth instance of the "loss silently reports 0.0000" failure class this project has now hit multiple times, caught only because a smoke test's `Dir:` column never moved. Fixed (commit `9cf7041`), `PKT_VERSION` bumped to 6, full 100k-file cache regenerated via `archive/slurm_scripts/generate_phytomer_packets_jobs.sh` (~3 min wall time).
4. **Closed the train/eval internode-assembly gap**: the differentiable render loss was still assembling each phytomer's internode from an independently-regressed length, even though eval/inference reconstruction has derived it from two node positions since earlier this session. Fixed (commit `0da4dc0`) by reusing the existing GT-key parent lookup (already built for `loss_stem_dir`) to find, for each matched node whose parent was *also* matched this step, that parent's **predicted** (not GT) position — giving the render loss a real cross-phytomer position-consistency gradient that did not exist before.
5. **Made the gradient safety net canary-only** (commit `f554d2a`): the per-element pre-clamp before `clip_grad_norm_` ran at ±100 on every step unconditionally, silently reshaping any merely-large (not actually broken) gradient. Raised to 1e15 (a threshold no healthy step should ever reach; it exists only to keep the sum-of-squares norm computation itself from a spurious fp32 overflow) and instrumented both this and the existing NaN/Inf step-skip with counters that surface loudly in the epoch summary line whenever nonzero, instead of requiring someone to grep logs.
6. **Validated all of the above with a real training smoke test**: 6 epochs, render gate active for epochs 4-6 (the historically dangerous zone), **0 recovery skips, 0 canary hits**.
7. **Open architectural question, staged for the next session**: does Stage 2's node rotation need its own full 6D output, or is most of it redundant with node position once topology is known? §6 below lays out a cheap, training-free measurement to answer this before committing to any architecture change.

---

## 2. PhytomerVAE v8: hybrid coarse (48D) + per-slot residual (10×8D) latent

Full design rationale and code are in `plant_recon/models/phytomer_vae.py`'s module docstring (edit this doc, not that one, if the rationale needs updating — the code docstring is the source of truth). Summary:

- `z[:48]` — shared "coarse" channel: class, base (always zero, deterministic), scale, curvature. Encoded/decoded exactly as the old single-latent design was.
- `z[48:128]` — 10 independent 8D "residual" channels, one per organ slot, dedicated to that slot's rotation only. Each is encoded directly from that slot's own raw rot6d (not pooled away by the shared trunk) and decoded by a shared-weight per-slot head conditioned on `[coarse | that slot's residual]`.
- Token-count win preserved: residual channels widen the *same* per-phytomer token; Stage 3 still sees K tokens, not 8K.

**Measured** (`eval_phytomer_vae_packet_fidelity.py`, no Helios, checkpoint `outputs/checkpoints/phytomer_vae_v8/phytomer_vae_128d_best.pt`, 60 epochs / 208s on a single TITAN RTX):

| Role | DAP10 | DAP50 | DAP90 |
|---|---:|---:|---:|
| Stem | 0.53° | 0.38° | 0.56° |
| Petiole | 2.94° | 3.02° | 2.93° |
| Leaflets | 2.33° | 2.46° | 2.39° |
| Peduncle | — | — | 5.05° |
| Repro | 0.48° | 0.38° | 1.32° |

Class accuracy 100%, scale relative error 2-15% (unchanged from the pre-hybrid baseline — the coarse channel wasn't touched).

**Caveat, not yet resolved**: raytraced Helios full-cycle IoU on a random 2-plants-per-DAP sweep (`eval_packet_roundtrip_sweep.py --helios`) did *not* show a correspondingly clean win over the old architecture (32-75% range, some DAPs worse with the VAE than without). Decomposition pointed at `chain_phytomers` topology mis-recovery on some of those random samples (DAP30: 14 predicted shoots vs 8 actual; DAP90: 24 vs 14) as a likely confound distinct from VAE quality, but this was not conclusively isolated. Revisit once the pkt-level win is confirmed to survive into the trained FM model's own predictions (this doc's §6 experiment is a step toward that, by removing the rotation-head cross-node consistency question from the picture first).

---

## 3. pkt-cache `keys` bug (PKT_VERSION 4→6)

`_XmlPktDataset.__getitem__` (the `--mode pkt` fast path, used by `generate_phytomer_packets_jobs.sh`) called `build_phytomer_packets(...)` **without** `return_keys=True`, and `generate_pkt()`'s two `torch.save()` blocks never wrote a `"keys"` field. `--mode cache`'s `build_pkt_targets()` (a different function) had always included it correctly — this was specifically a `--mode pkt` gap.

Consequence: `train_hierarchical_flow_matching.py` gates both `loss_phytomer_order` and `loss_stem_dir` on `"keys" in pt`. Every sample served from a `--mode pkt`-built cache trained with **both losses silently disabled** — exactly the failure class already documented once this session (three unrelated causes then; this is a fourth, in a different file). Caught by a smoke test whose `Dir:` column read `0.0000` on every single step.

Fixed in commit `9cf7041`: `_XmlPktDataset` now calls `return_keys=True` and threads `keys` through both save paths. `PKT_VERSION` bumped 5→6 to force a full rebuild of the just-regenerated (buggy) cache. Full 100k-file regeneration via the SLURM parallel launcher took about 3 minutes; verified post-hoc that a sampled cache file has `"keys"` present and that a training run's `Dir:`/`Ord:` loss columns move (they do — see §5).

**Side note, corrected from an earlier statement in this conversation**: a transient "99,949/100,000" file count reported mid-session was a timing artifact (checked before the last SLURM job's writes landed), not genuine empty/failed plants. Cross-checking all 40 job logs afterward showed `empty=0 err=0` in every single one, `sum(ok) = 41,760` newly processed + `58,240` already-done (from an earlier local partial run) = exactly 100,000. There are no known-empty XML files in this dataset.

---

## 4. Render-loss internode now assembled from two predicted positions

**The gap**: `assemble_packets()` gained a `parent_pos`/`centers` override earlier this session (derives a chained phytomer's internode base/length/direction from `‖this_node − parent_node‖` instead of an independently-regressed length — see `phytomer_packets.py` docstring for the full rationale and the 0.000cm-agreement measurement behind it). Eval/inference reconstruction uses it. The **training-time differentiable render loss did not** — `train_hierarchical_flow_matching.py`'s render block called `assemble_packets(recon_abs_all, rot_all.reshape(-1, 6))` with no `parent_pos`, meaning the actual training signal never saw the geometrically consistent version of what it optimizes, and neighboring phytomers' positions had no shared gradient path through the internode at all.

**Why not just run `chain_phytomers` on the model's own predictions inside the training loop** (the most literal way to close the gap): rejected. That would bootstrap the render loss off the model's own early, unreliable rotation/ordinal predictions — the same "targets move every step" instability already documented for the Exp E in-loop-Hungarian failure (a different problem, same shape of risk).

**What was done instead** (commit `0da4dc0`): reuse the GT-key parent lookup already built for `loss_stem_dir`. For each matched node whose parent's GT packet was *also* matched to some predicted node this step, record that parent node's index (`gt_render_parent_idx`, `has_render_parent`, both `(B, K_eff)`). In the render block, gather the parent's **predicted** (live-gradient, not GT) position through this map and pass it to `assemble_packets(parent_pos=..., centers=pos_all)`. Nodes with no matched parent this step (shoot bases, or an unmatched parent) fall through to the existing independent-length path via `assemble_packets`'s own NaN gating — no behavior change for those.

This is tied to the *already-computed, stable* GT match rather than to self-predicted topology, so it carries none of the Exp E risk, and it adds a **new** gradient path: the render loss can now push two different phytomers' positions to be mutually consistent through the internode's visible length/placement, which did not exist before (each phytomer's position previously only received gradient from the "where do I place this decoded organ cluster" step, independent of every other phytomer).

Validated by the smoke test in §5.

---

## 5. Gradient guard made canary-only (commit `f554d2a`)

Raised during this session's design discussion: is the three-layer defense (detach the Stage2→Stage3 feedback path / bound each head's output range / sanitize-and-clip gradients) over-engineered? Conclusion, on reflection:

- **Detaching the feedback path** is the one genuine root-cause fix (a missing `.detach()` was a real bug). Kept as-is.
- **Bounding each head's output** (Gram-Schmidt-orthonormalized rotation, `tanh`-bounded position/scale/existence) isn't really "defense" — it's just parameterizing a physically-bounded quantity correctly, the same category of choice as using `sigmoid` for a probability. Kept as-is, but described more accurately in conversation with the user going forward: not a safety layer, a modeling choice.
- **The per-element gradient pre-clamp before `clip_grad_norm_`** *was* the one piece that ran unconditionally on every step (±100, regardless of whether anything was actually wrong) — a real instance of the "silently mangle to hide whether a leak still exists" pattern the user flagged. Fixed: raised to 1e15 (a threshold that exists solely to keep the sum-of-squares norm computation well-defined against a specific fp32 overflow mode, not to soft-clip ordinary large gradients) and instrumented with a hit counter (`canary_elem_hits`) that prints loudly and surfaces in the epoch summary whenever nonzero.
- **The top-level NaN/Inf `grad_norm` handler** turned out, on inspection of the current code (not the stale proposal in an earlier doc), to already just skip the step — it does not sanitize-and-apply. No behavior change there; only added a per-epoch `recovery_skips` counter so a return of the "stuck skipping every step" deadlock pattern would show up in the normal epoch log line instead of requiring a log grep.

**Validation**: reran `archive/slurm_scripts/smoke_test_fix.sh` locally (6 epochs, batch 8, TITAN RTX, render gate active epochs 4-6) against the fixed pkt cache and the new render-loss code. Result: **0 recovery_skips, 0 canary_elem_hits**, checkpoint saved, self-consistency eval ran every epoch without incident. This also confirms retroactively that the old ±100 threshold was never actually needed during this run (nothing exceeded even that lower bar), so removing its unconditional application does not mask anything this specific test would have caught.

---

## 6. Next: is Stage 2's 6D rotation earning its keep, or is 2 of its 3 DOF redundant with position?

**The question, precisely.** A node's 6D rotation carries 3 rotational DOF: 2 for "which direction is forward" (the column-1 axis `chain_phytomers` already uses as a directional cue, and that `loss_stem_dir` already ties to the parent→node direction) and 1 for "roll about that axis" (which determines, e.g., which way a petiole/leaflet points sideways — genuinely not recoverable from positions alone). Once topology is known, the 2-DOF "which direction" part is **fully and exactly determined** by two node positions (`normalize(child_pos − parent_pos)`) — this is the identical derivation `assemble_packets(parent_pos=...)` already uses for the internode itself (§4), just one level up, applied to Stage 2's own node-rotation output instead of the packet's internal slot-0 organ.

**Why this isn't the same idea as the rejected chain-rule position proposal (option (a) from the design discussion)**: predicting position by *integrating* direction+length along a chain requires strictly sequential per-node computation and accumulates error down the chain — rejected for good, specific reasons (breaks the parallel 512-query set-prediction architecture; matching becomes unstable). This is the reverse direction: positions are still predicted exactly as today (in parallel, one Stage 2/3 forward pass, unchanged), and only *afterward* — once topology is resolved from those positions — is a *derived* direction computed for each node in one batched, vectorized step. No sequential dependency, no accumulation, no architecture change to Stage 2/3's core prediction mechanism.

**Why it isn't free**: `chain_phytomers`'s directional cost term needs a *predicted* direction to disambiguate close candidates (measured: 100% parent-recovery with the directional term vs 40-52% on distance alone, pre-ordinal). If direction is no longer independently predicted, that specific disambiguation signal disappears from the chaining cost — *unless* the already-added `ordinal` signal (this session's earlier work, not yet stress-tested on this specific question) compensates well enough on its own. This has not been measured.

### The cheap experiment (do this first, no training, no architecture change)

Measure `chain_phytomers`'s parent-recovery accuracy on ground truth with the directional cost term turned **off** (`dir_weight=0`) but the ordinal cost term still **on**, across the same DAP 10/50/90 + noise-sweep methodology already established in `phytomer_topology.py`'s own docstring (which currently only reports the *combined* with-direction number as 100% and *nothing-at-all* as 40-52%; the missing data point is "ordinal alone, no direction").

**Decision rule**:
- If ordinal+distance-only recovery is close to the 100% combined number (say, ≥95% across DAP 10/50/90 and reasonably robust under the existing position-noise sweep), proceed to actually shrink Stage 2's rotation head to a 1-DOF roll output (derive the 2-DOF direction post-hoc from resolved topology + positions) — this removes a real, now-confirmed-redundant piece of the architecture at the source, the cleanest of the three options discussed.
- If it degrades substantially without the directional cue, do **not** shrink the rotation head — instead treat the remaining topology-recovery gap (already visible in the Helios sweep confound noted in §2) as a case for a *learned* connectivity mechanism (graph edge-scoring / Part-Affinity-Field-style dense direction field, replacing the hand-tuned 3-term greedy cost in `chain_phytomers`) rather than a case for removing the rotation head. That is a substantially bigger investment (a new trainable component, careful staged training to avoid the Exp E class of instability) and should only be started once this measurement shows the current heuristic — not the input noise — is actually the bottleneck.

Recommended explicit order, per the user's direction: (b) cheap measurement first → (c) learned connectivity only if (b) shows direction is genuinely load-bearing for chaining. Chain-rule position integration (option (a)) is excluded from consideration; it was evaluated earlier this session and rejected for the reasons above.

### 6.1 Refinement (user proposal, 2026-09-11 evening): Stage 2 rotation+scale as a parent-relative PDF, not a point estimate

A stronger reframing of (b)/(c), proposed mid-session: instead of Stage 2 predicting a point rotation+scale per node and separately checking it against the parent direction via `loss_stem_dir` (a post-hoc consistency patch), predict a **probability distribution over the parent-relative displacement** (direction + distance) directly. The true parent need not sit exactly at the distribution's mode -- under good training it usually will, but the distribution's spread is itself useful signal, not noise to eliminate.

**Two concrete uses this buys, both real (not just "more flexible" in the abstract):**

1. **Adaptive per-node bridge noise for Stage 3.** Today `x_0 = Stage2_estimate + 0.05*eps` uses a *fixed* noise scale for every node regardless of how confident Stage 2 actually is. If Stage 2 instead outputs a concentration/variance parameter, that same value sets the flow's starting noise scale per node -- confident nodes (a clear straight internode) start tight and need little correction; ambiguous nodes (near a branch fork) start wide and get real work from Stage 3's refinement. This is the point where "refinement" stops being a fixed-radius search and starts meaning what the word implies.
2. **Likelihood-weighted `chain_phytomers` cost.** Replace the fixed `dir_weight=0.05 * (1-cos)` term with `-log p_i(direction_to_j | dist_i)` -- a node's own predicted confidence now directly controls how much its directional cue is trusted during topology recovery, instead of one global hyperparameter treating every node as equally reliable.

**What this does not remove:** the roll DOF (1 of rotation's 3 DOF; nothing about a parent-relative displacement says which way a leaf points sideways) still needs its own small head, and the shoot-base ("no parent") case still needs a discrete gate -- naturally a mixture-model component (`P(has parent) x direction-distribution | has parent`), which is conceptually the same job `is_shoot_base` already does, just formalized as a mixture weight. Implementation needs the same variance-floor/ceiling discipline already used elsewhere in this codebase (`SCALE_CEIL`'s soft tanh bound, orthonormalized rotation) -- an unconstrained learned variance is a well-known way to make a negative-log-likelihood loss diverge.

**Why this is a genuinely joint-inference architecture, not just an analogy.** `PhytomerFlowMatchingDecoder` already self-attends across all K phytomer query tokens at every ODE step before cross-attending to the image (`nn.TransformerDecoder`'s standard self-attn -> cross-attn per layer), and `CoarseSkeletalTransformer` already biases phytomer self-attention by 3D proximity (GraphFormer-style). So the system already resolves per-node ambiguity using every other node's current state and the image evidence jointly, at every refinement step -- it just does so today starting from point priors. Making Stage 2's output an explicit per-node PDF turns "Stage 3 finds the joint configuration consistent with the image and every node's prior belief about its parent" from an implicit behavior into the literal, intended objective.

**Revised experiment scope for §6's cheap measurement**: in addition to the plain ordinal+distance-only recovery number, also test whether an uncertainty-*proxy*-weighted directional cost beats the fixed-weight one on GT data -- e.g., weight the direction term by local node density (nearest-neighbor spacing) as a stand-in for "how ambiguous is this node's assignment," since a real learned concentration doesn't exist without training. A clear win from adaptive weighting over fixed weighting would be a strong signal to invest in the full distributional-head design rather than either simpler alternative.

### 6.2 Measurement result (2026-09-11, `/tmp/topology_measure.py`, GT positions/rotations, 8 plants/DAP)

`chain_phytomers` parent-recovery accuracy, all four configurations sharing the same GT `is_base` gate (so these numbers are internally comparable to each other but **not** directly comparable to the "40-52% distance alone" figure in this module's own docstring above -- that number did not have GT `is_base` fed in, this measurement always does):

| Config | DAP10 | DAP50 | DAP90 |
|---|---:|---:|---:|
| A: dist + direction + ordinal (current default) | 100.0% | 95.4% | 98.9% |
| **B: dist + ordinal, NO direction** | **97.5%** | **94.5%** | **97.7%** |
| C: dist + direction, NO ordinal | 100.0% | 94.7% | 98.1% |
| D: dist only, neither cue | 97.5% | 89.7% | 93.6% |

**Reading**: dropping the rotation-based directional cue entirely (B vs A) costs at most 2.5 points, and less than 1 point at DAP50 -- the ordinal signal (this session's earlier addition) already carries almost all of what direction was contributing to topology recovery. Direction and ordinal are also largely redundant with *each other* (C is nearly as good as A): either one, combined with distance, gets most of the way there.

The density-weighted variant (approximating adaptive/uncertainty weighting by rescaling the global `dir_weight` from local nearest-neighbor spacing) reproduced config A's numbers almost exactly -- this is expected and **not informative**: it is still one global scalar per plant, not a real per-node learned confidence, so it cannot be read as evidence for or against §6.1's distributional-head proposal. A real test of that idea needs an actual trained concentration/variance output, not a GT-derived proxy.

**Reading against the decision rule in §6**: config B clears the "≥95% and close to the 100% combined number" bar at DAP10 and DAP90 (97.5%/97.7%) and is within 1 point of the combined number at DAP50 (94.5% vs 95.4%) -- **this supports proceeding with (b): shrinking Stage 2's rotation head to a 1-DOF roll**, deriving the 2-DOF direction from resolved topology + positions instead. §6.1's PDF-based reframing remains the more principled longer-term direction but is not needed to justify this specific, smaller step, and its actual benefit is still unmeasured pending a real trained uncertainty output.

**Caveats**: n=8 plants/DAP (small); GT `is_base` was fed to every leg (isolates the direction/ordinal question but does not reproduce the original zero-side-information baseline); this measures topology recovery on **ground-truth** positions/rotations only, not on a trained model's actual (noisier) predictions, so it establishes an upper bound / feasibility signal, not a guarantee that shrinking the rotation head will not cost accuracy once real prediction noise is in the loop.

---

## 7. Commit log (this continuation)

```
4512926 feat(phytomer-vae): hybrid coarse+residual latent, give rotation its own capacity
9cf7041 fix(pkt-cache): --mode pkt was never saving the (shoot_id, phytomer_idx) keys
0da4dc0 feat(train): assemble the render-loss internode from two predicted nodes
f554d2a refactor(train): make the gradient safety net canary-only, not a soft clipper
```

Checkpoint lineage: `outputs/checkpoints/phytomer_vae_v8/phytomer_vae_128d_best.pt` is the current accepted PhytomerVAE. `dataset/cache/cowpea_curv26_pkt/` is fully regenerated at `pkt_version=6` (100,000/100,000 files, `keys` present, verified). No hierarchical FM training has been launched beyond the 6-epoch local smoke test in `outputs/checkpoints/fm_smoke_test/` — a real training run is still pending, and per this doc's §6, should wait for the rotation-head decision so it isn't immediately obsoleted by an architecture change.
