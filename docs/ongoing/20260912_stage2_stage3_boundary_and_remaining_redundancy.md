# Session Handoff: Gradient Explosion Root-Caused to the Learning Rate, Stage 2/3 Boundary Redesign Underway

- **Author**: Claude (pair programming with Heesup Yun)
- **Date**: 2026-09-12 (continuation of 2026-09-11's work)
- **Read this doc's own §5 first if you are starting a new session** — it lists exactly which prior docs to read and in what order.

## 0. Where things stand

**Landed and tested this session** (suite stays at its 3 known failures + 1 known error throughout):

| | commit |
|---|---|
| "anchor" retired from the vocabulary; say node / node position | `558d3ee`, `d4a745c` |
| Shoot-base forward axis from the successor (later superseded, see below) | `87ff1fb` |
| `ord_mae` and `ord_step_mae` diagnostics; topology gates on the STEP error | `02a7e29`, `e6e1f32` |
| Gradient canary names the parameter it fired on | `5e34a3d` |
| `FM_GRAD_PROBE=1` intermediate + per-layer gradient probes | `ab0910f`, `582d03a` |
| `--seed`, and an `UnboundLocalError` it exposed | `2fac3dc`, `d766a12` |
| Query-boundary gradient barrier (a barrier, not a fix — see §1.9) | `684a9f1` |
| Per-run figure folders under `slurm_scripts/logs/run_<jobid>/` | `12c0457` |
| **§2.1 redesign: `gt_parent_links`, one parent rule, no exceptions** | `8331abf`, `5e10c15` |
| **§2.1 redesign: every node gets a real parent; +Z fallback deleted** | `e902487` |
| `--resume` flag restored (its parser line had been lost; launcher passed it by default) | `c192f5f` |
| **Inference drew internodes from the decoded length, over ALL slots incl. non-existent ones** — fixed | `abce662` |
| **Ordinal = depth from the root; parent-step loss** | `ae26ccb` |
| **Packet slot-0 rotation forced to identity (it is the reference frame)** | `ef5802d` |
| **Leaflets emitted as lateral, terminal, lateral — the packet path is now lossless (§2.4)** | `ff5beb4` |
| **Leaf size is one scalar per node (1 : 1 : 10/9); terminal leaflet identified per node; opt-in terminal-last packet order for v9** | `2f1691b` |
| **Stem inverse kinematics in the export: Helios's FK now follows the predicted nodes (§2.5)** | `d3731d3`, `7d92840`, `5827327` |

**The blocker is NOT resolved -- see §1.9.2 (2026-09-12 afternoon): it recurred at epoch 28 with lr below 1e-4.** The earlier reading, kept for the record: the intermittent Stage 2 gradient explosion tracks the **learning rate**, not the architecture: every observed onset sat above ~1.6e-4 effective lr, and job `38240281` at `LR=1e-4` has now run **10 epochs clean** with 0 canary hits, through and well past the point where three of four runs at 3e-4 died. Eight architecture-level hypotheses were tested against minimal reproductions and refuted first; §1.9 records them so nobody pays for them twice.

**Quality on that run** (render loss still off, it enables at epoch 40): IoU 14.0% -> 25.9%, depth MAE 17.67 -> 13.74 cm, node RMSE 2.4 -> 2.0 cm, predicted phytomer count 52.2 against 54.4. Node RMSE is already inside the 2.5 cm target; IoU is far from the 50%+ target but the photometric signal has not been switched on yet. **Roll has started to learn** — 0.885 (the no-information value, §1.6) down to 0.8250 — which is why it was right not to restructure the roll head on the strength of a few flat epochs.

**Careful about attribution**: `38240281` carries *both* the low lr and the query barrier from `684a9f1`, and "1e-4 without the barrier" has not been run. The barrier alone is known insufficient (`38240158` exploded with it at 3e-4). Test removal when the redesign next changes this code, rather than assuming.

**Also note**: `38240281` was launched *before* the `8331abf`/`5e10c15`/`e902487` redesign commits, so it is training the old target convention. It is a stability and quality baseline, not a measurement of the redesign.

**What the render loss did once switched on** (job `38240323`, resumed from the epoch-10 checkpoint with `RENDER_GRAD_START_EPOCH=11`, `RENDER_FRACTION=0.03`): IoU 24.2% -> 27.5% -> 27.1% -> **32.2%** over epochs 11-14, depth MAE 14.52 -> **13.29 cm**, node RMSE 1.9-2.2 cm, roll 0.8244 -> **0.7979**. A clear upward trend, and these numbers are still measured through the *old* inference path below, so the true silhouette is better than they say.

**Two inference bugs found by looking at the epoch-11 panel rather than the numbers** (§2.3). A 2 cm seedling with 0.6 cm node RMSE rendered at IoU 0.0% because metre-long stems shot across the frame. Fixed in `abce662`; the ordinal follow-up is `ae26ccb`.

**Helios round-trip (§2.4, §2.5): solved.** `docs/results/assets/fig14_phytomer_vae_helios_roundtrip.png` (v9_tl_rw4_20k VAE, stem IK on both export columns) reads IK-only 95.7 / 99.5 / 96.7% and VAE round-trip **92.2 / 98.4 / 95.7%** FG IoU, against 81.4 / 90.2 / 79.3 this morning (v8, analytical export). Three things were wrong and all are fixed: the packet path lost the leaflet order (`ff5beb4`, `2f1691b`); the analytical 14D -> XML export was the real ceiling for every VAE, because Helios rebuilds a shoot by forward kinematics from per-node angles the converter derived from petiole azimuths and a decoded curvature, so a 1-3 degree error at one node moved every node above it -- `part_tensor_stem_ik` (§2.5) solves those parameters so the same FK lands on the predicted nodes and lifts even the identity export above the old IK-only number; and the seedling's young nodes were a small minority of the VAE's training packets, so their leaflet scale was 4.4% off -- training the same recipe on 20,000 files instead of 8,000 brings that to 1.0% and DAP 10 from 83.2 to 92.2. `phytomer_vae_v9_tl_rw4_20k` is the accepted VAE for the export path; the eval scripts default to it. It is NOT yet the FM training VAE -- that needs the packet cache regenerated with terminal-last packets and its latents (PKT_VERSION 7) and a Stage 3 retrain, see §5.

**Next**: `38242849` (resumed from epoch 25 at `LR=5e-5`, §1.9.2) is the live run; watch the Stage 2 Scl/Ord/Ext losses for the epoch-27-style precursor, not just the canary. In parallel: pick a v9 VAE (§2.4, end), then §2.1's remaining pieces -- Stage 3 consuming `(parent, self)` pairs with the parent held fixed, and noise injection on the fixed parent -- and the gradient bisection in §1.9.2.

---

## 1. What changed and was validated since the previous doc

> **Chronological record, partly superseded.** §1 describes the state as of the roll-head reduction. Its world-+Z fallback for shoot bases was replaced by a successor rule in §1.5 and then deleted entirely in §2.2, where every node gets a real parent instead. Read §0 for the current state.

Previous doc: [`20260911_hybrid_vae_rotation_capacity_and_topology_experiments.md`](20260911_hybrid_vae_rotation_capacity_and_topology_experiments.md) ended with a cheap, training-free measurement showing `chain_phytomers` loses at most 2.5 points of parent-recovery accuracy without a rotation-based directional cue, once the ordinal cue is present. That measurement justified shrinking Stage 2's node rotation head.

**Implemented** (commits `0341be2`, `8a98cfd`):
- New `diffusion_based/dataset/phytomer_roll.py`: `derive_forward(pos, parent_idx)` (forward axis from resolved topology, world-+Z fallback for shoot bases), `encode_roll(R, forward)` / `roll_to_matrix(forward, roll)` (exact round-trip, tests in `tests/test_phytomer_roll.py`).
- `phytomer_topology.chain_phytomers`'s `rot6d` parameter is now `Optional` (`None` drops the directional cost term) -- this is what makes "resolve topology from position+ordinal, THEN derive rotation" possible instead of circular.
- `CoarseSkeletalTransformer`: `rot_head` (6D) -> `roll_head` (2D). Output dict key `phytomer_rot` -> `phytomer_roll`.
- New `hierarchical_part_flow_matching.reconstruct_phytomer_rot(pos, roll, ordinal, is_base_logits)` (`@no_grad` -- topology resolution has no gradient regardless of caller): chains on position+ordinal alone, derives forward, combines with predicted roll -> full 6D rotation. Called in `sample_ode` (inference) and per-render-sample in the training loop's render block.
- `PhytomerFlowMatchingDecoder` / `FineBotanicalFlowMatchingDecoder`: `node_rot_mlp` (Linear(6,·)) -> `node_roll_mlp` (Linear(2,·)); `phytomer_rot` parameter -> `phytomer_roll` throughout.
- Training loop: `gt_phytomer_roll_target` (2D) replaces `gt_phytomer_rot_target` (6D), computed via `encode_roll(GT_rotation, forward)` where `forward` is the same GT-parent-direction already computed for the now-removed `loss_stem_dir` (with a world-+Z fallback for shoot bases, matching `derive_forward`'s inference-time convention exactly, so the target stays valid against whatever axis reconstruction will actually pair it with). **`loss_stem_dir` is gone** -- with the forward axis no longer independently predicted, nothing can disagree with it by construction.
- Made the gradient safety net canary-only (commit `f554d2a`, same day): the unconditional +-100 per-element pre-clamp before `clip_grad_norm_` (which silently reshaped any merely-large gradient every step, not just broken ones) is now a +-1e15 threshold that only exists to keep the norm computation itself from a specific fp32-overflow mode, with hit-counting (`canary_elem_hits`) that prints loudly if it's ever nonzero. The top-level NaN/Inf handler already only skipped the step (no sanitize-and-apply); added a `recovery_skips` counter surfaced in the epoch summary line so a return of the old "stuck skipping every step" deadlock would be immediately visible.

**Correction made while implementing this** (worth remembering): the model does **not** actually run the "Stage 3 refines base+rot+scale+latent via a combined flow vector" design its own class docstrings describe. `HierarchicalPartFlowMatchingModel.__init__` constructs `PhytomerFlowMatchingDecoder` with `base_dim=rot_dim=scale_dim=0`, so the real, active flow target is the 128D VAE latent alone (pure `N(0,I)` prior, no bridge coupling); pos/roll/scale are Stage-2-only outputs with their own direct losses, used only as conditioning ("Hybrid Decoupled" in the code's own comments). This is presumably a deliberate simplification after an earlier bridge-flow attempt hit a real gradient explosion from a missing `.detach()` at the exact point where position entered the flow state. Keep this in mind before assuming the docstrings describe the live behavior -- trace `forward_backward_step`'s actual `z_0`/`z_1` construction directly when in doubt.

**Validated**: 6-epoch local smoke test (`slurm_scripts/smoke_test_fix.sh`, TITAN RTX, gpu-5-58), render gate active epochs 4-6 (the historically dangerous zone) -- **0 `[Recovery]` lines, 0 `[Canary]` lines**, checkpoint saved (`diffusion_based/checkpoints/fm_smoke_test/hierarchical_fm_epoch_006.pt`, disposable, smoke-test only). Full `pytest tests/` after all of the above: 38 passed, exactly the same 3 failures + 1 error confirmed pre-existing via `git stash` before this session touched anything (`test_end_to_end_model_forward_backward`, `test_end_to_end_phytomer_mode_forward_and_ode`, `test_phytomer_flow_decoder_shapes_and_gradients`, `test_ik_fix::test_eval`) -- no regressions.

---

## 1.5 Later the same day: terminology, and the shoot-base forward axis

**Terminology (commit `558d3ee`)**: the word **"anchor" is retired from this project**. Say **node** or **node position**. It named a live predicted 3D position that bipartite matching pairs with a ground-truth phytomer, but imported connotations from detection anchor boxes -- a fixed, pre-defined prior grid -- that describe nothing this model does. Local identifiers were renamed (`anc_src`/`anc_tgt` -> `node_src`/`node_tgt`), "re-anchor" was reworded to say what it does (place a packet back into the world frame), and active docs were updated; doc mentions of long-renamed symbols (`loss_anchor_pos`, `SLOTS_PER_ANCHOR`, ...) were corrected to their current names in the same pass. Historical records under `docs/done`, `docs/results` and `docs/archived` were deliberately left untouched.

**Shoot-base forward axis (commit `87ff1fb`)**, prompted by the user asking whether Stage 3 should be predicting which direction a phytomer faces. The answer is that it already does -- "facing" is 3 DOF, of which 2 (the forward axis) are determined by position and 1 (roll) is not, and roll is exactly what §2 proposes moving to Stage 3. But the question exposed a real gap underneath it:

`derive_forward` fell back to world +Z for any node with no parent. "Shoot base" sounds like it means the handful of stems rising from the plant's own base, but **a lateral branch's first phytomer is a base too** -- Stage 2's is-base head is trained to flag it (its GT key ordinal is 0), and `chain_phytomers` honours that flag by refusing it a parent. Lateral branches are rarely vertical. Measured against ground-truth reference frames over 392 base phytomers:

| forward axis for a base | mean | median | p90 | max |
|---|---|---|---|---|
| world +Z (old) | **50.8°** | 54.2° | 87.7° | 129.4° |
| successor direction (new) | **14.5°** | 12.5° | 23.1° | 24.5° |
| *(chained nodes, parent direction, for scale)* | *0.6°* | *0.5°* | *1.3°* | *2.8°* |

Bases are ~13% of all phytomers at DAP 50 (96 of 757). End to end, reconstructing the full node rotation from GT positions with a perfect roll head, over ALL phytomers: mean **10.40° -> 7.40°**, p90 **46.13° -> 12.96°**.

The base tail does get worse (p90 98.9° -> 123.0°), and it is worth knowing why before anyone "fixes" it: 6% of bases pick the wrong successor, and that group is **entirely `chain_phytomers` mislabelling the shoot** (chain-vs-GT successor agreement 0% in that group, 96.8% in the other 94%), with a mean axis error of 120°. It is the same topology-resolution exposure chained nodes already carry, not a flaw in the successor rule, and a distance gate cannot separate the two groups (successor distance / median spacing is 0.26 for the bad group vs 0.14 for the good one -- both well inside any plausible gate). Improving it means improving chaining, which has its own measurement harness in `diffusion_based/eval/measure_topology_recovery_ablation.py`.

The training target follows the same fallback order (parent direction, else successor, else +Z), read from GT `keys` rather than from `chain_phytomers`, so the target teaches the correct relationship and only inference carries the chaining error. `gt_stem_dir_target` was renamed `gt_forward_target` to match what it now holds.

### 1.6 What the roll target actually looks like, and what to do if roll stalls

Worth knowing before anyone reads the roll loss. The scaffold losses reduce as `sum / norm_nodes`, so they are **per-node sums of per-component errors**, not per-element means. For roll, a 2-vector, that doubles the familiar number: a prediction carrying *no information* about the target sits at **0.88**, not 0.44. (Reference values, smooth_l1 on unit 2-vectors, doubled: random or collapsed 0.88; exactly 45 deg off 0.29; 15 deg off 0.034; the theoretical maximum, predicting the exact opposite, 1.65.) Training runs so far sit at 0.87-0.89 through the first several epochs, which is exactly the no-information value -- not a bug, but the number to watch.

Measured on ground truth (32 plants, DAP 30-90, 2274 consecutive pairs):

- **Absolute roll is uniform.** All twelve 30-degree bins are populated within a factor of two. There is no prior to exploit and no constant a model can collapse onto profitably.
- **The increment between consecutive phytomers in a shoot is alternate (distichous) phyllotaxy.** Circular mean **-179.9 deg** -- dead on 180 -- with resultant length R = 0.441. 51% of increments fall within +-30 deg of 180, 62% within +-45; only 12% sit near 0.

So the *hard* part of roll is one phase per shoot, which only the image can supply; everything after it is "add 180 and repeat", recoverable from the ordinal the model already predicts. The current `roll_head` predicts each node's roll independently from its own features, which is a strictly harder problem than the data requires, and it carries a specific failure mode: where the image does not pin the phase down, the posterior over roll is near-uniform, and the smooth_l1-optimal point estimate of a near-uniform circular target is the **zero vector** -- which `safe_normalize` then turns into an arbitrary unit direction. A model in that state is indistinguishable from an untrained one by the loss alone.

**If roll is still at ~0.88 after the scaffold losses have otherwise converged**, do not conclude roll is unlearnable. Check the *raw* `roll_head` output norm before `safe_normalize` first: a collapse to near-zero norm confirms the averaging failure above. The fix that follows from the measurement is to stop predicting K independent rolls -- predict one phase per shoot plus a parity term driven by the predicted ordinal -- rather than to reweight the loss.

### 1.7 The ordinal head gates everything downstream, and how accurate it has to be

Since the roll-head reduction, the ordinal is no longer just a tidying signal: `chain_phytomers` resolves topology from **position + ordinal alone**, and that chain is what produces the internode geometry, the derived forward axis, and (as of §1.5) a shoot base's successor. If the ordinal is wrong, all three are wrong together.

Measured (20 plants, DAP 10/30/50/90, GT positions throughout so the ordinal cue is the only variable; gaussian noise in ordinal steps, parent recovery vs the GT parent):

| noise sigma | 0 | 0.25 | 0.5 | 1.0 | 1.5 | 2.0 | 2.6 | 4.0 |
|---|---|---|---|---|---|---|---|---|
| DAP 10 | 100.0 | 97.6 | 78.9 | 54.2 | 55.3 | 49.2 | 46.6 | 39.7 |
| DAP 30 | 96.8 | 96.8 | 89.0 | 73.3 | 64.5 | 57.4 | 53.2 | 45.0 |
| DAP 50 | 96.3 | 95.9 | 91.2 | 76.8 | 68.5 | 60.8 | 55.3 | 45.4 |
| DAP 90 | 96.1 | 95.5 | 92.9 | 84.6 | 78.6 | 70.2 | 59.8 | 49.3 |

That table looks alarming, but **it is the worst case, and the absolute MAE is the wrong thing to measure.** `chain_phytomers` only ever reads `(o_child - o_parent)` and compares it against 1, so any error *shared* along a shoot cancels there. Repeating the sweep with three error structures scaled to the same absolute MAE (15 plants, DAP 10/50/90):

| structure | MAE 0.5 | 1.0 | 2.0 | 3.0 |
|---|---|---|---|---|
| per-shoot offset (one constant per shoot) | 96.7-100.0 | 97.0-100.0 | 97.0-100.0 | **97.4-100.0** |
| smooth drift along the shoot, DAP 50/90 | 96.3 | 94.6 | 92.4 | 90.3 |
| smooth drift along the shoot, DAP 10 | 82.5 | 69.9 | 37.7 | 29.5 |
| independent per node (the first table) | 72.2-92.4 | 54.2-80.5 | 43.7-63.6 | **43.7-52.1** |

A pure per-shoot offset leaves topology **completely untouched even at 3.0 steps**; independent per-node error of the same size roughly halves parent recovery. DAP 10 remains the fragile case throughout (its nodes sit 0.47 cm apart), and it is the one stage where even smooth drift hurts.

**So the requirement is local consistency, not small absolute error**, and the metric to watch is the error on the *difference* to the parent, not the ordinal itself.

**Observability fix made while measuring this**: `loss_phytomer_order` sums two unrelated terms -- the smooth_l1 on the ordinal and the BCE on the is-base logits -- so the single logged "Ord:" number cannot say which is failing, nor what the error is in steps. Two diagnostics now sit beside it in the step line, the epoch line and wandb:

- `phytomer_ord_mae` -- absolute error in steps. Useful context, but do not gate on it.
- `phytomer_ord_step_mae` -- `|(pred_ord[node] - pred_ord[parent]) - 1|` over nodes whose parent was also matched this step, reusing the `gt_render_parent_idx` map the render path already builds. **This is the one topology depends on.** Its no-information value is exactly **1.0** (a constant ordinal makes every difference 0), so anything below 1.0 means the head has started to carry ordering, and it should be driven toward 0.

### 1.8 Throughput: budget the render loss before launching, and do not trust the batch auto-tuner

The first real (non-smoke) hierarchical FM run, job `38239988`, had to be cancelled after four epochs. It was not broken -- 0 recovery skips, 0 canary hits through the render gate, loss 102.9 -> 48.8, self-consistency IoU 6.5% -> 20.0% -- it was simply **unable to finish anything**. Two causes, both worth checking before any future launch.

**1. The render cost per epoch is fixed by `render_fraction`, not by batch size.** Total rendered samples per epoch is `dataset_size * render_fraction`, independent of how the batch is shaped. At the default `render_fraction = 1/6` over 100k samples that is ~16,700 renders per epoch; measured on 4x RTX 6000 Ada at ~1.9 s per rendered sample (forward + backward), that is **~2.4 hours per epoch**. With `--save_every 25` the job would not have written its first checkpoint for 60 hours, inside a 24-hour limit. Do the arithmetic first: `dataset_size * render_fraction * 1.9s / n_gpus` is the per-epoch floor.

The step timing makes the split obvious -- warmup epochs ran at `fwd/bwd 1.33s`, and the first render epoch jumped to `fwd/bwd 84.31s` with `render 25.54` and `backward 56.50`. Note also that the two pyramid levels are **zoom factors, not resolutions** (a 1.2 m window and a 0.6 m window at the same pixel count, see `render_multiscale_pyramid`), so they cost about the same and dropping one is a real loss of signal -- absolute metric scale from the wide level, organ detail from the zoomed one -- rather than free savings. `render_fraction` and `render_grad_start_epoch` are the honest levers.

**2. `--batch_size auto` picked 256 per GPU (global 1024).** It probes model and activation memory, which the render block barely touches, so it settled at 256 while actual VRAM sat at 57.6% of 47.4 GB -- and a global batch of 1024 leaves only 100 optimizer steps per epoch on 100k samples, at an `lr` of 3e-4 that was tuned for the documented 48-per-GPU (global 192) configuration. Pass `FORCE_BATCH_SIZE=48` unless there is a reason not to.

**What replaced it**: job `38240070`, `FORCE_BATCH_SIZE=48` (520 steps/epoch), `RENDER_FRACTION=0.05`, `RENDER_GRAD_START_EPOCH=40`, `SAVE_EVERY=10`, output `checkpoints/hierarchical_fm_derived_rot_v2/`. The reasoning behind deferring the gate to epoch 40: in the cancelled run, self-consistency IoU had already reached 18.3% at epoch 3 with the render loss still off, and only moved to 20.0% once it came on -- so the early gains come from the scaffold and latent losses, and the expensive photometric signal is better spent refining a model that is already roughly right.

### 1.9 RESOLVED: the intermittent Stage 2 gradient explosion, and the eight hypotheses it was not

**Resolution is in §1.9.1: the learning rate.** This section is kept for the eight architecture-level hypotheses that were tested and refuted on the way there -- read it before proposing a ninth.

**Symptom.** With the render loss OFF, Stage 2 diverges somewhere in epoch 2-3. Onset is a handful of gradient elements past 1e15 with **no NaN**; within ~60 steps it floods, and `clip_grad_norm_` then rescales the whole gradient by ~1e-19, so real signal is crushed and later steps are skipped outright. Job `38240070` ended at `Recovery: 118/524` in epoch 3 and every step of epoch 4 skipped; self-consistency collapsed to IoU 0.0% / Dice 1.000 (an empty silhouette).

**It is not reproducible run-to-run**: epoch 2 in one probe, epoch 3 in another, and clean through epoch 2 in a third, all byte-identical. Nothing seeded weight init or shuffling, which is why `--seed` now exists (commit `2fac3dc`). **Use it from now on when chasing this.**

**What the instrumentation says.** Two tools were added, both of which the previous canary work implied but did not provide:

- The canary now names the parameter it fires on (commit `5e34a3d`). The `[Recovery]` dump cannot: by the time `grad_norm` is NaN, every `coarse_stage` parameter is NaN with `max_finite 0.0`. Only the FIRST canary burst is trustworthy, and it named `coarse_stage.ref_points` -- **one element of 1536, 1.325e15, no NaN**.
- `FM_GRAD_PROBE=1` hooks the Stage 2 intermediates (commit `ab0910f`). It showed **only `q_pos` (the `ref_pos_mlp` output) carrying the explosion, at 6.3e10 to 1.0e13**, while `edge_bias`, `delta_pos`, `phytomer_pos` and the final `phytomer_features` all stayed under 1e9.

So the gradient **arriving at Stage 2's transformer output is normal (<1e9) while the gradient leaving its input is ~1e13** -- the decoder amplifies internally by about four orders of magnitude, and `ref_points` is simply where that lands, via `q_pos`. Later bursts name `decoder.layers.{1,2,3}` LayerNorm weights and biases and the attention `in_proj` weights, spread widely rather than concentrated.

**Five mechanisms were hypothesised and all five refuted by minimal reproduction** -- recorded so nobody pays for them twice:

1. *The `anc` -> `node` identifier rename aliased two variables.* No: none of the new names existed in either file beforehand.
2. *`cdist` self-distance backward at coincident points.* No: finite even at exactly zero separation (max grad ~21 with a uniform upstream gradient, ~2.8e-3 through the real attention module).
3. *Perspective division in `PhytomerVisualProjector`.* No: it is `tanh(pos_to_uv(...))` into a border-padded `grid_sample`, with no division at all.
4. *An unguarded loss denominator.* No: idle damping is behind `num_idle > 0` and the scaffold losses divide by `max(matched, 1)`, which scales with the numerator.
5. *The edge-bias magnitude growing as `ref_points` spread.* No: driving the bias from -7 to -988 leaves the gradient into the query flat at ~2e-8.

**The most promising untested lead is loss imbalance, not a numerical singularity.** The evidence for divergence rather than a singularity: in the epoch where the canary first fires, several losses are simultaneously getting *worse* (`Scl` 0.40 -> 0.88, `DAP` 18.4 -> 28.0, ordinal MAE 1.99 -> 2.31, `Pos` 0.0074 -> 0.0126). And the objective is badly unbalanced -- the logged macro losses are printed **pre-weight**, so `Phy: 12.89` at `phy_count_weight 2.0` contributes ~25.8 of a ~33 total, about **78% of the objective**, from a single count regression that feeds the shared trunk. `DAP: 27.99` at weight 0.05 contributes only 1.4 by comparison.

**Caveat on that hypothesis, added after looking at the gradient rather than the value.** `loss_phy_count` is `smooth_l1_loss` on raw counts, and smooth_l1's gradient **saturates at +-1** once the error exceeds 1. So its 78% share of the loss *value* does not translate into a 78% share of the gradient *magnitude* -- it contributes a bounded +-2.0 per sample at weight 2.0. Divergence is driven by gradients, not by loss values, so this is a weaker lead than the value breakdown first suggested. What survives of it: that bounded gradient is nonetheless the largest single push on the macro head, it persists because the count error stays large, and that head both feeds the shared trunk and sets the active capacity K.

Also note that *down-weighting* is not the fix even if the hypothesis holds, because it cuts the head's effective learning rate along with the scale: at `PHY_COUNT_WEIGHT=0.25` the epoch-1 count prediction fell to 11.1 against a ground truth of 54.9 (baseline at the same point: 22.5), and self-consistency IoU dropped from 22.0% to 13.0% because too few slots go active. If scale is the problem, normalise the target (count/100, or log-count) so conditioning is preserved.

**That test ran and refuted the hypothesis.** Job `38240147` (`PHY_COUNT_WEIGHT=0.25`, `SEED=1234`) exploded anyway, ending at `Recovery: 524/524` with loss 0.0000 -- every step of the epoch skipped. So the imbalance is not the cause, and down-weighting cost real accuracy on the way to finding that out (epoch-1 count 11.1 vs 22.5 at weight 2.0, IoU 13.0% vs 22.0%). `PHY_COUNT_WEIGHT` is back at its 2.0 default.

**Two further hypotheses tested and refuted after that**, bringing the total to seven:

6. *The residual stream growing without a final LayerNorm amplifies the mask gradient.* `nn.TransformerDecoder` here genuinely has no `norm=` argument while its layers are `norm_first=True`, so the omission is real -- but it is not this: larger activations make the mask gradient *smaller*, because the downstream LayerNorm normalises the scale away (9.8e-11 at unit activations, 4.9e-13 at 1e2, 7.9e-33 at 1e3).
7. *The edge-bias magnitude.* Driving the bias from -7 to -988 leaves the gradient into the query flat at ~2e-8.

**Why bisecting stalled.** Even with `--seed` the onset moves: one run exploded in epoch 3 and its same-seed sibling was still clean at epoch 4, because cuDNN, the DDP reduction order and the dataloader workers are not deterministic (full determinism would need `torch.use_deterministic_algorithms(True)`, which not every op here supports). There is no fixed reproduction point to bisect against.

**What was applied instead: a gradient barrier at the query boundary** (commit `684a9f1`). `q.register_hook(clamp(-5, 5) + nan_to_num)` in `CoarseSkeletalTransformer.forward`, the same pattern already used on `pos_all` and the fine-exist logits in the render path. The justification is the measurement, not a mechanism: the gradient reaching the decoder's output is under 1e9 while the gradient leaving its input is ~1e13, decoder layer 0 alone amplifies ~107x at healthy magnitudes (measured by `_make_layer_probe`), and all three leaves feeding the query -- `phytomer_queries`, `ref_points`, `ref_pos_mlp` -- hang off the end of that amplifier. Healthy gradients at that boundary are ~1e-2, so the clamp does not bind in normal training, and epoch-1 metrics with it are within run-to-run variance of without (IoU 19.6% vs 22.0%, depth MAE 14.80 vs 15.16 cm, node RMSE 2.1 vs 2.2 cm).

**Be clear about what this is.** It stops a pathological step from destroying the reference-point prior, which is what actually ends a run. It does not explain the amplifier. It is deliberately narrower than the blanket +-100 per-element clamp the previous session removed for masking leaks -- one boundary identified by measurement, with the canary left in place to report if it fires anyway. **If a future session finds the real mechanism, remove the barrier rather than leaving both.**

**The barrier was then tested and is NOT sufficient** (job `38240158`). It did do its stated job -- when the canary fired, `ref_points` and `ref_pos_mlp` were **absent** from the affected list for the first time, and no step was skipped at all (`Canary: 63008 elems` with no `Recovery` field), so the run survived where every predecessor deadlocked. But the model was destroyed anyway: the DAP head collapsed to predicting 0.0 against a ground truth of 50.5, and epoch-3 self-consistency fell to **IoU 0.0% with node RMSE 4.1 cm** (from 2.1). Surviving the step is not the same as surviving the event, because `clip_grad_norm_` still rescales the whole gradient by ~1/1e15 whenever one element is that large, so every head gets an effectively zero update for the duration and some of them die.

**The eighth hypothesis, and the pattern that motivated it.** Every explosion hits **all four decoder layers' `multihead_attn.in_proj_weight` simultaneously**, usually alongside their `norm3`. Those four layers all consume the *same* memory (the image tokens), and `d(in_proj_weight) = grad_out x input^T` picks up the memory magnitude on the K and V rows -- so a single pathological sample would produce exactly that all-layers-at-once signature, and DINOv2 is known to emit high-norm artifact tokens. **Refuted**: scanning 720 cached samples across DAP 1-90 through the frozen encoder, the maximum token element is 33.10 against a median of 28.82 (ratio 1.1x) and the maximum token L2 norm 63.15 against a median of 59.41. There are no outliers.

**State when this section was written: open, eight hypotheses refuted, no root cause.** §1.9.1 then found it. The tooling below is still what to reach for if anything like this recurs:

- `FM_GRAD_PROBE=1` gives the intermediate hooks (`q_pos`, `edge_bias`, `phytomer_features`, `delta_pos`, `phytomer_pos`) and the per-layer amplification hooks.
- The canary names the parameters of its first burst per epoch.
- `--seed` exists, though it does not make onset reproducible on its own.
- The two unexplained numbers, both worth attacking directly: the gradient *arriving at* `edge_bias` reached **2.1e20**, an order of magnitude above anything else in the system (hypothesis 7 refuted only that the bias's *value* drives amplification, not the size of the gradient flowing into it), and decoder layer 0 amplifies **~107x at healthy magnitudes**, which is large enough to be worth understanding on its own terms.
- One structural oddity found but not implicated: `nn.TransformerDecoder` is built with `norm_first=True` layers and **no final `norm=`**, so the residual stream is never normalised at the output. Hypothesis 6 showed this does not amplify the mask gradient, but a pre-norm stack without a final norm is still non-standard and cheap to fix.

**A recommendation on framing.** Eight architecture-level guesses failed, which is itself evidence: this is more likely an optimisation-dynamics problem than a broken op. Do not spend another session enumerating ops.

### 1.9.1 The learning rate tracks the failure, and it explains the control run too

Acting on that framing immediately produced the strongest correlation in the whole investigation. Warmup is linear from **0.1x to 1.0x over `--warmup_epochs 3`** (default), so the peak `lr` of 3e-4 is only reached at the end of epoch 3. Computing the multiplier at each observed onset:

| run | onset | lr multiplier | effective lr |
|---|---|---|---|
| `38240070` | epoch 3, step 358 | 0.90x | 2.71e-4 |
| `38240120` | epoch 2, step ~262 | 0.55x | 1.65e-4 |
| `38240147` | epoch 3, step ~262 | 0.85x | 2.55e-4 |
| `38240158` | epoch 3, step ~262 | 0.85x | 2.55e-4 |

**Every explosion happened above ~1.6e-4**, clustered in the last third of the warmup ramp.

**And this retracts an earlier conclusion in this doc.** The single-process control run was clean for 2,083 steps and that was read as implicating 4-rank DDP or per-rank data sharding. It does not: with global batch 48 instead of 192 it runs 2,083 steps per epoch rather than 524, so its 3-epoch warmup spans 6,250 steps, and at the end of epoch 1 -- where it was stopped -- it had only reached **0.40x of peak, 1.20e-4**. It never entered the regime where anything ever failed. There is no DDP-specific effect in evidence.

This also fits the one run that survived longest at batch 256 (global 1024, job `38239988`, four epochs clean): the same `lr` over a 5.3x larger batch is a much smaller effective step.

**CONFIRMED.** Job `38240281` (`LR=1e-4`, everything else unchanged from `38240158`: `SEED=1234`, batch 48, render deferred to epoch 40) ran **10 epochs with 0 canary hits**, through the end of warmup at epoch 3 where the lr reaches its peak and well past it. For comparison, at 3e-4 three of four runs were already dead by epoch 3. Quality improved monotonically over those epochs rather than collapsing: IoU 14.0% -> 25.9%, depth MAE 17.67 -> 13.74 cm, node RMSE 2.4 -> 2.0 cm.

So the mechanism is **optimisation dynamics, and the fix is a schedule** -- not an architecture change, and not the per-element clamp the previous session removed. Use `LR=1e-4` for now; raising it later is a tuning question to revisit only from a known-stable baseline, and the onset threshold (~1.6e-4) is the number to stay under.

**Two caveats on this result, both worth respecting.** `38240281` carries the query barrier from `684a9f1` as well as the low lr, and "1e-4 without the barrier" was never run -- so the barrier's necessity is unproven either way (it is known *insufficient* on its own: `38240158` exploded with it at 3e-4). Re-test removal when the redesign next touches that code. And the run predates the §2.1 redesign commits, so it measures the old target convention.
2. If it survives, reproduce deterministically with `--seed`, then bisect inside the decoder by extending `_probe_grad` to each layer's input and output -- the amplification is somewhere in those four layers.
3. Only then consider re-introducing bounded per-element clipping. Note that the previous session *removed* a blanket +-100 per-element pre-clamp specifically to stop it masking leaks, and this is plausibly the leak it was masking -- so re-adding it would hide a real defect and should be a deliberate, documented choice, not a reflex.

### 1.9.2 It came back at 1e-4: the learning rate delays the explosion, it does not prevent it

Job `38240479` (`LR=1e-4`, `SEED=1234`, resumed from epoch 15 with the depth
ordinal and step loss, render on) ran clean for 12 more epochs and then went the
same way as the 3e-4 runs, at an effective lr *below* 1e-4 (cosine decay, epoch
28 of 500). The precursor is visible a full epoch earlier and is the thing to
watch for: in epoch 27 the Stage 2 scale, ordinal and existence losses jumped
mid-epoch (Scl 0.30 -> 2.94, Ord MAE 2.4 -> 11.0 at step 416, Ext 0.5 -> 1.6)
and did not come back (epoch-27 means Scl 0.75, Ord MAE 4.6, Ext 1.0; eval IoU
33.7 -> 27.7%). At epoch 28 step 210 the canary fired for the first time with
NaN gradients on **every** `coarse_stage` parameter (queries, `ref_points`,
`ref_pos_mlp`, all decoder layers), and every step after it was NaN too -- the
recovery path skips those steps, but a run whose forward pass produces NaN on
every batch is not going to recover, and its Stage 2 heads had already
regressed. Cancelled at epoch 28 and resumed from the clean epoch-25 checkpoint
as `38242849` with `LR=5e-5`, `SEED=1235` (a new seed so the same batch order
does not replay), output `hierarchical_fm_depth_ord_lr5e-5`.

So §1.9.1's conclusion needs narrowing: lowering the lr moved the onset from
epoch 3-10 to epoch 27, which is consistent with "optimisation dynamics" but
not with "the fix is a schedule". Something in Stage 2 drifts toward a
singular configuration over tens of epochs, and a lower lr only slows the drift.
The two leads that survive: the epoch-27 precursor (whatever grows there is the
cause, and it is measurable before the NaN), and the untested query barrier
(`684a9f1`). Bisecting with `FM_GRAD_PROBE=1` from the epoch-25 checkpoint at
1e-4 with seed 1234 should reproduce it in about two epochs.

**What has been tried since (same afternoon).** `38242849` (resumed from epoch 25,
`LR=5e-5`, `SEED=1235`) reproduced the precursor at the *same steps* of epoch 27
(step 208: Ord MAE 3.84 vs 3.63; step 416: Scl 1.20 vs 2.94) and burst at step
445 with huge-but-finite gradients on every `coarse_stage` parameter, first in
the decoder's `norm3` weights/biases and cross-attention projections; cancelled.
Two hypotheses were then tested and refuted: (a) weight decay driving LayerNorm
weights or query embeddings to zero -- the checkpoints show LN weights at 1.00
and query norms growing (0.44 -> 0.52 median) from epoch 15 to 25; (b) a
collapsed decoder token (a pre-LN block amplifies gradients by 1/std of the
token) -- `FM_ACT_PROBE=1` (new, forward probe on every decoder LayerNorm input)
shows the smallest per-token std at 0.11 and growing through the layers. A
15-epoch replay from the epoch-25 checkpoint on a 512-sample subset (`FM_ACT_PROBE`
+ `FM_GRAD_PROBE`, batch 8, 1 GPU) never fired the canary; the layer probe
reported at most 108x amplification in `decoder.layer0`.

**The lead that remains is the data.** `DistributedSampler` is constructed with
its default seed, so the shuffle is keyed by the epoch only: two runs with
different `--seed` see the *same batch at the same step*, which is exactly what
the two runs did at epoch 27. The trainer now names the batch's samples on the
first canary and, with `FM_SPIKE_DUMP=1`, on any step whose Stage 2 scale /
ordinal / existence loss spikes (`dc3846e`). A full-dataset replay of epochs
26-28 from the epoch-25 checkpoint is running locally under torchrun (1 GPU,
batch 48; the sampler reproduces the cluster's permutation, so the cluster's
global step k is local steps 4k..4k+3). If the spikes land on the same samples,
the fix is a per-sample guard (or excluding the samples); if not, the
state-drift reading stands and the query barrier is next.

*2026-09-13 morning*: the cluster replay (`38243735`, 4 GPUs, exact optimizer
restore via `FM_PARENT_COND=0`) has not started -- the group's GPU quota
(`QOSGrpGRES`, 8 GPUs) is held by Heesup's own `regen_shard` array from the
Image2PlantArchitecture_v2 project (submitted 2026-09-12 12:46 and 16:22, still
running), and the A100 partition is refused at submit time under the same
group limit. The replay is therefore running locally on the single RTX 6000
Ada (`slurm_scripts/logs/local_replay_full.log`): same epoch permutation
(`DistributedSampler` seed 0 + epoch), batches of 48 that are quarter-slices
of the cluster's 192, so cluster step k of epoch 27 is local steps 4k..4k+3;
about 45 minutes per epoch with the GPU otherwise idle.

*2026-09-13 12:00, result*: the local full-dataset replay went through epochs 26
and 27 with **no canary and no large Stage 2 spike** (epoch 27: Scl 0.28, Ord MAE
2.17, Ext 0.46 -- healthier than the cluster's epoch 27), on the same epoch
permutation with exact optimizer state. So the burst is not a property of the
samples alone. What the replay did *not* do, and both cluster runs did, is the
per-epoch self-consistency evaluation (`sample_ode` + the fixed-set render
between epochs; the replay ran with `EVAL_EVERY=1000`). The 512-sample subset
replay also skipped it. The replay is therefore running again with
`EVAL_EVERY=1 EVAL_MIN_INTERVAL_MINUTES=30 EVAL_SAMPLES_PER_BUCKET=2`
(`slurm_scripts/logs/local_replay_eval.log`). The other difference left is
4-rank DDP itself (192-sample global batches, gradient all-reduce), which only
the queued cluster replay can test. If the eval reproduces it, look at what
`evaluate_self_consistency` / `sample_ode` leave behind: train/eval mode,
in-place writes to buffers or parameters, autocast state, RNG.

---

## 2. Proposed next step A: move roll + scale prediction from Stage 2 to Stage 3

**The question that motivated this** (user, 2026-09-12): does Stage 3 actually need Stage 2's roll and scale as conditioning at all?

**Answer: very likely no, for roll; probably not required for scale either, but weaker case.** The reasoning ties directly to how packets are built: base is cluster-center-relative, rotation is phytomer-reference-frame-relative (`R_ref^T @ R_org`), and scale is normalized by `s_a`. The 128D shape latent is therefore designed to be **invariant** to the phytomer's absolute world-frame roll and absolute size by construction -- conditioning Stage 3 on Stage 2's *absolute* roll to help predict a *roll-invariant* target is very likely giving it information with no relationship to the thing it's predicting. Scale has a weaker justification (absolute size correlates with developmental stage, which could carry soft information about class distribution/curvature even in a size-normalized target), but is not a hard requirement either.

**Position is the one exception, for a different reason**: `PhytomerVisualProjector` uses `phytomer_pos` to sample a *local* image-token feature at that specific 3D location -- this is about visual grounding (what does the image actually show here), unrelated to whether roll/scale help predict shape. Position conditioning into Stage 3 should stay.

**Proposed restructuring**:
- **Stage 2 (skeleton only)**: position, existence, ordinal, is-shoot-base. Drop `roll_head` and `scale_head` from `CoarseSkeletalTransformer` entirely.
- **Stage 3 (everything about this phytomer's own appearance)**: given position + image conditioning, produce roll, scale, AND the 128D shape latent. Roll/scale should NOT be flow-matched (that would reopen exactly the "two moving targets" risk in §4) -- add them as small auxiliary feedforward heads on `PhytomerFlowMatchingDecoder`, read at t=1 from the converged query features, using **exactly the existing pattern `pred_slot_exist_logits` already uses** (`exist_head` reads the decoder's internal features `x` directly, evaluated once via `final_out` at `timesteps=1.0` in `sample_ode`, never part of the flow state itself). Add `roll_head`/`scale_head` the same way, alongside `exist_head`.

**Why this is lower-risk than it might look**: it doesn't touch the ODE state at all (still pure 128D latent, `N(0,I)` prior, unchanged) -- it only moves WHERE two small feedforward predictions are computed from (Stage 2's earlier, less-contextualized features -> Stage 3's later features, which have been through more self-attention across phytomers and more cross-attention to the image). The existence-head precedent proves this pattern is already safe in this codebase.

**Scope estimate**: comparable to the roll-head reduction just completed -- touches `CoarseSkeletalTransformer.__init__`/`forward` (remove 2 heads), `PhytomerFlowMatchingDecoder.__init__`/`forward` (add 2 heads + a `final_out`-style t=1 readout path, which doesn't cleanly exist for phytomer mode inside `forward()` itself the way `sample_ode` currently does it only at the end -- needs a decision on whether the *training* loop also needs a final/t=1 evaluation of roll/scale, or whether it's acceptable to read them from an arbitrary-t query as an approximation during training and only trust the t=1 read at inference; look at how `pred_slot_exist_logits` is actually supervised during training before assuming the precedent transfers exactly), training loop (move `loss_phytomer_roll`/`loss_phytomer_scale` targets and their matched-node lookup to read from the Stage-3 output instead of Stage-2's, remove the two conditioning MLPs `node_roll_mlp`/`node_scale_mlp` from `PhytomerFlowMatchingDecoder` since Stage 3 no longer receives them as input).

**Not yet resolved before starting**: check precisely how `pred_slot_exist_logits` is supervised at TRAINING time (not just inference) -- does `forward_backward_step` also do a final/t=1-style evaluation for it, or does it supervise the exist logits from whatever `x_t` the main training forward pass used (a specific, possibly non-1.0 `t` per sample)? Whichever answer applies determines whether roll/scale (if moved to Stage 3) can follow the exact same training-time supervision path or need a different one. Read this from `forward_backward_step` directly before implementing -- do not assume from the inference-time (`sample_ode`) pattern alone.

### 2.1 ADOPTED: Stage 3 refines a child against a fixed parent

Agreed with Heesup, 2026-09-12, after a round of proposal and measurement. This supersedes §2's plumbing (though §2's reasoning about why roll and scale belong in Stage 3 still holds and is subsumed here).

**The shape of it.** Stage 2 keeps predicting each node **once**, as now. Stage 3 consumes **(parent, self) pairs**, holds the parent **fixed**, and predicts the child's position, roll, scale and 128D shape against that fixed backdrop. An earlier variant had Stage 2 predict a parent *and* a child per node; that was dropped because each physical node would then be predicted twice -- once as somebody's child and once as somebody's parent -- which reintroduces exactly the redundancy the roll-head reduction and the internode work have been removing, and lets the two copies disagree.

**"Direction" must mean roll only.** If Stage 3 predicted the child's position *and* its direction, those two are redundant: direction is `normalize(child - parent)`. The free degree of freedom is the 1-DOF roll and nothing else, so Stage 3 predicts **position + roll + scale + shape**. (§1.6's measurement makes this a good fit: the GT roll increment between consecutive phytomers has a circular mean of -179.9 deg, so relative-to-parent roll is the easy quantity to learn.)

**How (parent, self) is paired -- no new matching is needed.** The training loop already computes it, at `train_hierarchical_flow_matching.py` around the `gt_render_parent_idx` / `has_render_parent` construction. The link is **derived transitively**, not solved: the existing bipartite matcher pairs predicted node `i` with GT row `r(i)`; GT topology gives `r(i)`'s parent row `p`; if `p` was also matched, to predicted node `j`, then `parent(i) = j`. Because the parent link is GT topology composed with the matching that already exists, this adds **no second assignment problem** and so none of the Exp E assignment-flicker risk that §4.1 warns about. At inference there is no GT, so the parent comes from `chain_phytomers(pos, ordinal, is_base)` -- already running inside `reconstruct_phytomer_rot`.

**Every node has a parent one internode below it.** Heesup's correction, and the measurement that settles it. A node is the TOP of its internode, so the main stem's first node sits one internode *above* the origin rather than at it. Measured (24 plants):

| DAP | main-stem 1st node to origin | lateral 1st node to origin | median internode | lateral 1st to nearest node below |
|---|---|---|---|---|
| 10 | 1.5 cm | 1.9 cm | 0.3 cm | 0.6 cm |
| 30 | 3.0 cm | 10.0 cm | 1.0 cm | 1.2 cm |
| 50 | 3.0 cm | 16.4 cm | 1.8 cm | 2.9 cm |
| 90 | **3.0 cm** | 16.5 cm | **2.9 cm** | **3.1 cm** |

At DAP 90 the main stem's first node sits 3.0 cm from the origin against a 2.9 cm median internode -- one internode, as predicted. A lateral's first node is far from the origin (16.5 cm) but 3.1 cm from the nearest node below it on another shoot -- also one internode, from its branch point. So the rule is uniform:

> **A node's parent is one internode below it**: the **origin** (a virtual node carrying no phytomer) for the main stem's first node, the **branch-point node on the parent shoot** for a lateral's first node, and the previous node in the same shoot for everything else.

This is worth adopting for three separate reasons. It takes parent coverage from **86.2% to 100%** (measured: 13.8% of GT phytomers are first-of-shoot and find no parent under the current same-shoot, ordinal-1 lookup). It **deletes `derive_forward`'s world-+Z fallback structurally**, including the successor-direction patch added in §1.5 -- the main stem's first node derives its axis from `self - origin` and a lateral's from `self - branch point`, both real geometric parents rather than a guess. And it closes the gap on exactly the 13.8% that §1.5 measured as 50.8 deg mis-oriented under the +Z assumption.

**Implementation note**: the `keys` field is `(shoot_id, phytomer_idx)` only and does **not** record which node a lateral branches from, so the branch link has to come from somewhere. Prefer deriving it geometrically -- nearest GT node below, on another shoot, gated by the `MAX_EDGE_FACTOR = 3.0` x median-spacing rule `chain_phytomers` already uses -- over regenerating the 100k-file packet cache for a new field. The measurement above validates that lookup: the correct parent really is at one-internode distance. Deriving it also has the side benefit that training and inference then resolve the branch link by the same rule.

**Noise injection on the fixed parent (adopted).** Holding the parent at GT during training is the asymmetric design §4.1 sanctions, but it creates the train/inference gap §4.1 warns about, over chains of tens to hundreds of phytomers. Mitigate with noise -- and note that the dominant inference error is **which node was picked**, not where it is, so the noise needs two components:

- **position jitter**, at Stage 2's node RMSE, ~2 cm;
- **parent substitution**, at `chain_phytomers`' parent-recovery failure rate: 0-4% when the ordinal is good (96-100% recovery), but 48-60% when the ordinal carries no information.

**Calibrate to the achievable rate, not the current one.** The ordinal head currently sits at the no-information value (`ord_step_mae` ~1.0, §1.7), so using today's measured failure rate would train against a wrong parent roughly half the time and the supervision would be close to worthless. Start at **5% substitution plus 1-2 cm jitter** -- the operating point chaining reaches when the ordinal works -- and re-calibrate from `ord_step_mae` once it actually falls below 1.0.

### 2.2 §2.1 implementation status

**Landed.**

- `phytomer_topology.gt_parent_links(centers, keys)` (`8331abf`, `5e10c15`) implements the one-parent rule and returns the parent **position** plus its row index, so the root's origin parent needs no special case at any call site. Validated on 24 plants over DAP 10-90: coverage **86.2% -> 100.0%** (85.6% same-shoot, 1.7% origin, 12.7% branch-point, 0 unresolved), and the child-to-parent distance is **1.00x the median internode at the median** -- direct confirmation of the rule. Vectorised to 1.28 ms per call from 3.99 ms (the per-shoot Python loop's `.item()` syncs would have cost +64% of step time at 48 samples per step); it is still +20.5%, and the way to remove that is to cache per plant in the dataloader workers, since GT topology is fixed across epochs. Gated on the plant's own median internode via `MAX_INTERNODE_FACTOR = 6.0` rather than `chain_phytomers`' median-of-all-pairwise (which is 10.7 cm at DAP 90, so a 32 cm gate that rejects almost nothing).
- The training loop resolves parents through it (`e902487`), so the forward-axis target is defined for 100% of matched nodes, and `is_base` now means **the single plant root** -- one positive per plant instead of 4-11.
- `derive_forward`'s fallback chain is **deleted**. This also removes the successor patch from §1.5 made earlier the same day: that took the 13.8% of nodes it covered from 50.8 deg of error to 14.5, but the residual was real curvature, and a true branch-point parent removes the error rather than shrinking it. A genuinely degenerate row now returns zeros instead of claiming vertical.
- Checked end to end with the render loss on from epoch 1: no recovery skips, no canary hits.

- **Stage 3 now sees its (parent, self) pair, with the parent held fixed** (late
  2026-09-12). `PhytomerFlowMatchingDecoder` gains `node_parent_mlp`, fed
  `[parent - self (metres), has_parent]` (`parent_relative()` in the model
  module); its output layer starts at zero, so every existing checkpoint loads
  and behaves exactly as before until trained. During training the parent is
  the **GT parent position** of the node's matched GT phytomer (the origin for
  the plant root), taken from the same `gt_parent_links` pass that builds the
  forward-axis target -- so it does not move with the prediction -- and it is
  attached to the second (grad) forward, after matching. At inference
  `sample_ode` runs `reconstruct_phytomer_rot` once before the ODE loop and
  feeds the chain's parent. **Noise on the fixed parent** is in: Gaussian jitter
  (`--parent_jitter_cm`, default 1.5) and substitution by another matched
  node's GT position (`--parent_substitution`, default 0.05), the §2.1 starting
  point. Smoke-tested end to end on a 24-sample local run.

**Not yet done.**

1. **Stage 3 predicting the child's position, roll and scale against the fixed
   parent** (today it still predicts only the 128D shape latent; the parent is
   conditioning). Then drop `roll_head`/`scale_head` from Stage 2 (§2's
   original plumbing).
2. Validating the conditioning on a real run -- blocked on §1.9.2 (the Stage 2
   burst at epoch 27; the instrumented cluster replay `38243730` is queued).
3. Re-calibrating the noise from `ord_step_mae` once it falls below 1.0.
4. Worth considering while doing (1): with `is_base` reduced to the plant root, `chain_phytomers` may not need the `is_base` gate at all -- the lowest node has no candidate below it and so becomes parentless on its own.

### 2.3 What the epoch-11 panel showed, and the two inference bugs behind it

The self-consistency numbers are one mean over four DAP buckets and hid two separate failure modes that the panel (`slurm_scripts/logs/run_38240323/hierarchical_self_consistency_epoch_011.png`) made obvious:

- **DAP 2 seedling**: node RMSE **0.6 cm**, count 4.1 of 4 -- the skeleton was essentially perfect -- yet IoU **0.0%**, because two internodes a metre long crossed the whole 1.2 m window on a 2 cm plant.
- **DAP 68**: phytomer count **117 of 56**, a 2x overcount, so far too many organs. IoU 12.7%.
- **DAP 97**: IoU 53.5%, already at target. Mature plants work.

The seedling case was two bugs in `sample_ode`, neither present in the training render path:

1. `assemble_packets` was called **without `parent_pos`/`centers`**, so every internode used the VAE-decoded slot-0 length rather than the parent->node segment the training block draws. (§3 of this doc had flagged that decoded length as "thrown away downstream" -- true in training, false at inference.)
2. Fixing (1) alone made it *worse*, and an `FM_ASSEMBLY_PROBE=1` dump (added in `abce662`, off by default) showed why: `chain_phytomers` ran over **all `active_k` slots, existence ignored**. Non-existent slots sat 20 cm *below the soil* and 30 cm to the side; being lowest, the height rule left them parentless, they became "roots" whose parent is the origin, and 36 cm stems were drawn to them -- while real nodes chained to them as parents. Node RMSE never saw any of this because it is one-way (GT -> nearest prediction) and stray predictions are invisible to it. Chaining only slots with existence > 0.5 removes both; a non-live row's parent is now its own position (zero length, no NaN). Same fixed 10-sample set: **11.7% -> 14.4%**.

A gap gate was then tried -- distrust a chained stem longer than ~6x the decoded length -- and **rejected by measurement**: 14.4% -> 12.1% on the same set, because the fallback stem still points at the wrong parent, so the stem gets shorter but no more correct. What remains after the fixes is live nodes chained **6-14 cm apart on 1-3 cm internodes**, which is the ordinal head carrying no ordering (`ord_step_mae` ~1.1 for 13 epochs). That is what `ae26ccb` addresses.

**`ae26ccb` also corrects a flaw in the ordinal itself, not just its training.** Under the new parent rule a lateral's first node parents to a branch-point node on *another* shoot. `chain_phytomers` scores a candidate by `|(o_child - o_parent) - 1|`, so with per-shoot ordinals that true parent (ordinal 0 vs, say, 5) is charged 6 x `ORD_WEIGHT` = 0.12 m -- more than any internode -- and actively rejected. §1.7's 96-100% recovery figures were measured under the old convention where laterals were forced parentless and never had to be linked. The ordinal the head learns is now **depth from the root** (`gt_parent_links` returns it), under which every parent is exactly depth - 1, laterals included; the chain cost, the step loss and `is_base` (depth == 0) all follow from that one rule. Expect `ord_mae` to jump on resume (the target's range grew from a per-shoot index to ~0-50) and watch `ord_step_mae` instead -- below 1.0 is the first sign of ordering.

### 2.4 The Helios round-trip: where the VAE path actually loses the plant

Heesup's requirement (2026-09-12, "very important"): the VAE round-trip
(14D -> packet -> VAE -> 14D -> XML -> Helios) must reproduce the IK-only
reconstruction (14D -> XML -> Helios, no VAE). The figure is
`docs/results/assets/fig14_phytomer_vae_helios_roundtrip.png`; the old copy in
`docs/results/assets/_unreferenced/` (52.8 / 20.4 / 24.7%) predates the 2026-09-11
emit-order and shoot-partition fixes. FG IoU against the Helios GT render, same
three plants throughout:

| DAP | IK-only | packet path only (VAE = identity) | VAE-v8 round-trip |
|---|---|---|---|
| 10 | 95.7% | 78.9 -> 95.1 -> **95.8%** | 70.3 -> 70.5 -> 83.0 -> **81.4%** |
| 50 | 94.1% | 91.6 -> 93.8 -> **94.2%** | 87.8 -> 90.0 -> 89.9 -> **90.2%** |
| 90 | 87.4% | 87.4 -> 88.0 -> **87.9%** | 76.2 -> 79.9 -> 79.2 -> **79.3%** |

The middle column runs the same post-decode path with the VAE replaced by identity
(`strip_base(normalize(packets))`), so it isolates the deterministic packet
representation from the VAE. The arrows are the three fixes of the afternoon, in
order (`ef5802d`, `ff5beb4`, `2f1691b`):

**`ef5802d` -- slot 0's rotation is the reference frame.** A packet's slot-0
(internode) rotation is stored relative to `refs`, which *is* that internode's
rotation, so its target is exactly the identity; the VAE was decoding a small
non-identity there and Helios's shoot-base IK rotated whole shoots by it. Forcing
identity in `assemble_packets` gave +2.2 / +2.2 / +3.7 on the VAE path and nothing
on the identity path (which was already exact there).

**`ff5beb4` -- leaflets go out as (lateral, terminal, lateral).** Found by refusing
the "small rotation errors integrate through FK" story for the identity column,
where the 14D rows are exact to 1e-4 in every column. Rendering the plain
encode->decode tensor (no `assemble_packets`, no chain override) gave the same
92.0% at DAP 50 as the full identity path, which cleared assembly and the parent
override; a per-phytomer numeric diff of the IK XML against the emitted XML then
showed exactly one systematic difference: `leaf_scale`, on 326 of 491 leaves, by
+/-11%. `PartTensorTo40DConverter` gives a trifoliate leaf its yaw from the order
in which its leaflet rows are encountered (child index 0 -> +10 deg, 1 -> 0 deg
terminal, 2 -> -10 deg) and reads **only the scale** from the row -- the leaflet's
own rotation is ignored -- while the packet keeps the terminal leaflet in slot 4
(`LEAFLET_ATTACH_FRAC`, tip). Emitting slots in numeric order handed the terminal
leaflet's scale to a lateral and the lateral's to the terminal. Patching only
`leaf_scale` in the emitted XML back to the IK values gave 94.3% (IK-only 94.1);
the code fix gives 95.1 / 93.8 / 88.0. The seedling's 17-point loss was this too:
the cotyledon second-petiole story below was real but worth well under a point.

**`2f1691b` -- the terminal leaflet is not always slot 4, and leaf size is one
scalar.** `_canonical_order_key` fills slots 2-4 bottom-to-top; the two laterals
share a height (0.8 of the petiole) and the terminal sits at the tip, which is
*below* them when the petiole droops. Terminal in slot 4: 98 of 142 trifoliate
nodes at DAP 50, 116 of 162 at DAP 90; **in slot 2: the other 44 / 46**; never
slot 3. `ff5beb4`'s fixed (2, 4, 3) order therefore still swapped 31% of nodes.
The terminal is now identified per node as the larger of slots 2 and 4 -- exact
on GT because, on 4,214 trifoliate nodes from 80 plants, the laterals are
*exactly* equal and the terminal is *exactly* 10/9 of a lateral (std 0.0 on
both). That same fact makes leaf size one scalar per node, so `assemble_packets`
re-imposes the 1 : 1 : 10/9 ratio from the mean of the three decoded values and
puts the identified terminal at the petiole tip. Identity: 95.8 / 94.2 / 87.9 --
the packet path is exact. On the VAE path the ratio rule is a wash (+/- 1.6
points; the argmax misfires when the decoded scales are within their ~8% noise
of each other), which is expected: it cannot create information the VAE lost.
`build_phytomer_packets(terminal_leaflet_last=True)` (or
`PHYTOMER_TERMINAL_LAST=1`) orders leaflets by size so the terminal is always
slot 4; it is opt-in until the v9 cache and VAE adopt it.

**What Helios actually reads from a packet**, established along the way, and the
reason rotation-error tables for leaflets are irrelevant to this figure: internode
rotation only for a shoot's first node (chained nodes take the parent->node
segment); petiole rotation, length and curvature; leaflet **scale** only; peduncle
rotation and curvature; flower/fruit scale. A VAE that spends capacity on leaflet
rotations is spending it on something the export discards.

**Refuted this round** (recorded so nobody pays for them twice):
- "Slot-0 lengths come back at 0.2x" -- a nearest-base matching artifact between
  packets and GT internodes; matched by tip, 0 of 164 differ.
- The chain override (`parent_pos`) as the DAP 50 cause: identity without it
  renders 92.0 vs 91.6 with it (pre-fix numbers), i.e. 0.4 points, and the
  override is required at inference anyway.
- `assemble_packets`: the encode->decode-only tensor renders identically.
- `plant_age` (0.0001 vs 51) and the 143 dormant `<peduncle>` blocks at DAP 50:
  0.0 change, as at DAP 10.

**A negative result that corrects §2.2 and the `e902487` commit message.** Routing the
round-trip through `gt_parent_links` (laterals get their branch-point parent, root
gets the origin) instead of the old chain (every shoot's first node parentless)
drops DAP 50 from 91.6% to **75.8%** (identity) and 87.8% to 75.9% (VAE). Attributed
cleanly: root -> origin costs nothing (91.6% unchanged); **the lateral link costs all
16 points** -- even though the geometric pick matches the XML's true attachment in 8
of 10 laterals. So it is not *which* node: `assemble_packets` overwrites a lateral's
first internode with the straight branch-node -> node segment, and **a lateral's
first internode axis is not that segment** (the same ~10-15 deg residual the
successor rule showed in §1.5). Helios's shoot-base IK turns that into a rotation of
the entire branch. Consequences:
1. The XML/Helios export path must keep the **decoded** rotation and length for a
   shoot's first internode (the round-trip eval already does; do not "upgrade" it).
2. `e902487`'s claim that the branch-point parent "removes the error rather than
   shrinking it" is **wrong for laterals' first nodes**: no position-derived rule
   recovers that axis, so the roll target there is ~10-15 deg off under either rule.
   Not a regression versus this morning, but not the fix the message claimed. The
   only exact source is the GT internode rotation, which at inference would have to
   be predicted -- Heesup's original question about Stage 3 predicting direction is
   back on the table for exactly these ~13% of nodes.
3. For the PyTorch render used in training and self-consistency, the straight
   segment is a 3 cm tube a few degrees off -- minor. It is Helios's FK that
   amplifies it.

**The seedling's residual (95.1 vs 95.7)**, chased before the leaflet fix when it
looked like 17 points, and still true at its real size:
- `--focus-plant` **framing**: the eval zooms each render to its own bounding box,
  so any growth difference re-frames the plant; with a fixed camera the pre-fix
  XMLs gave IK 95.8% / identity 83.4% against 95.7 / 78.9. The eval is not a pure
  geometry measure. (At a fixed 5 m camera a seedling is ~820 pixels, so that
  variant is too coarse to adopt as-is.)
- The **cotyledon node's second petiole**: Helios instantiates 5 fewer petiole
  primitives and 1 fewer leaf from the emitted XML at every DAP. The packet has one
  petiole slot, so `emit_part_tensor_with_shoot_meta` mirrors petiole 1 by 180
  degrees about Z; both cotyledon leaves then land on petiole 1's tip (the petioles
  are 0.0001 m long), the converter assigns both to petiole 1, and the XML writer
  drops the second. Emitting the mirror before the leaves and re-basing the
  opposing leaf gets both `<leaf>` blocks into the XML, but Helios still renders
  37 leaves, not 38, and patching the mirror's `petiole_pitch`/`petiole_curvature`
  to IK's values changes 0 pixels. Lost between the 40D tensor and Helios's
  unifoliate phytomer; seedling-only in practice (1 leaf of ~600 at DAP 90).

**What is left is the VAE**: 14.3 / 3.9 / 8.1 points. Re-measured with the fixed
emit order (before `2f1691b`; FG IoU, GT presence throughout, each row replaces
the named decoded columns with the exact ones):

| variant | DAP 10 | DAP 50 | DAP 90 |
|---|---|---|---|
| VAE-v8 as-is | 83.0 | 89.9 | 79.2 |
| + exact leaflet scales (slots 2-4) | 88.8 | 89.4 | 80.1 |
| + exact petiole scale+curvature | 83.2 | 89.8 | 79.1 |
| + exact petiole rotation | 84.8 | 90.5 | 83.0 |
| + exact internode rotation | 83.0 | 89.9 | 79.2 |
| + exact rotations, all slots | 84.8 | 90.5 | 82.6 |
| + exact scales, all slots | 90.1 | 89.9 | 79.7 |
| + exact scales+curvature, all slots | 90.1 | 89.5 | 83.9 |
| + exact rot+scale+curv, all slots (= identity) | 95.1 | 93.8 | 88.0 |

Two things to read off. (1) **The loss is not additive**: at DAP 50 no single
family recovers anything (rotations 90.5, scales 89.9) yet both together give
93.8 -- a leaf that is both displaced and mis-sized fails to overlap until *both*
are right, so a v9 has to improve rotation and scale fidelity together, not one
of them. (2) The internode row is flat because `ef5802d` already forces slot 0 to
identity. The VAE-v8 errors that matter, per slot (present slots, mean): petiole
rotation 2.9 deg (p90 4.1), peduncle 5.0 deg; relative scale error leaflets 7-8%,
petiole 1.5-5.6%, peduncle 11%, flowers/fruit 12-22%, internode 17-39% (unused
for chained nodes). Leaflet rotation error is irrelevant to Helios (see "What
Helios actually reads"), though the PyTorch training render does draw it.

**v9 candidates, measured** (local GPU, logs in `slurm_scripts/logs/local_vae_v9/`; all
128D = 48 + 10x8, 8,000 files, 120 epochs, ~5 min each; `_ctrl` keeps the v6/v8
bottom-to-top leaflet order, `_tl` uses `PHYTOMER_TERMINAL_LAST=1`, `_rw4` adds
`--rot-weight 4`). Packet-level fidelity at DAP 50 (mean, p90 in parens) and the
Helios round-trip:

| VAE | petiole rot (deg) | leaflet rot | leaflet scale | stem scale | Helios FG IoU 10 / 50 / 90 |
|---|---|---|---|---|---|
| v8 (4,000 files, 60 ep) | 3.02 (4.30) | 2.46 (3.70) | 5.1% (11.5) | 13.5% | 81.4 / **90.2** / 79.3 |
| v9_ctrl | 2.65 (4.88) | 1.55 (2.42) | 0.8% (1.4) | 9.9% | **83.5** / 88.0 / **85.0** |
| v9_tl | 3.02 (5.12) | 1.52 (2.46) | 0.9% (1.6) | 8.7% | 83.2 / 84.7 / 83.1 |
| v9_tl_rw4 | **1.58 (2.48)** | **1.07 (1.76)** | 0.8% (1.6) | 8.9% | 81.7 / 82.9 / 75.8 |
| v9_tl_rw4_20k (20,000 files) | 1.56 (3.18) | **0.86 (1.48)** | **0.6% (1.1)**; DAP 10 3.4% -> 1.0% | -- | with stem IK: **92.2 / 98.4 / 95.7** |

Read together with the attribution rows, this says the Helios number does **not**
track packet fidelity. `_rw4` has the best rotations by a wide margin (petiole
direction error 1.4 deg mean, azimuth p99 4.4 deg vs v8's 20.9, and *no* node over
10 deg) and the best scales, yet renders worst; v8 misidentifies the terminal
leaflet on a third of its nodes (55 / 163 at DAP 50) and renders best at DAP 50.
Its attribution: `_rw4` + exact petiole rotations = 92.7 at DAP 50 and 85.1 at
DAP 90 -- ten points from a 1.4-degree error. So the loss is a few nodes where
Helios's forward kinematics is ill-conditioned, not average fidelity, and which
nodes those are is luck per model.

**Where the DAP 50 loss actually sits (v9_tl, bisection by shoot and by column).**
Replacing the decoded petiole rotations with exact ones one shoot at a time:
shoot 3 (20 nodes) alone gives 84.7 -> **89.7**; every other shoot 0.0-0.8; the
five lateral base nodes together +1.3. Zeroing the decoded *internode* curvature:
84.7 -> **87.9** (exact curvature on every slot: 88.0). Both together: **93.7**,
i.e. the identity number (94.2). The mechanism for curvature is now understood
from `PlantArchitecture.cpp`: the converter writes the 14D internode curvature
into Helios's `curvature_perturbations`, which Helios applies as a per-segment
bend angle in degrees while building the shoot by FK, so a decoded 0.4-2 deg
per node accumulates over a 20-node shoot into a tip displaced by centimetres.
The three round-trip plants (`exact_gt_renders/`) have that value exactly 0 on
every internode (and constant 180 deg phyllotaxy, zero yaw perturbations); the
training distribution does **not** (internode curvature -6..+7, phyllotaxy
195-213 per node, yaw perturbations up to 9.5 on a 60-plant sample), so zeroing
it in export is exact for the figure and wrong for the dataset, and was not
done. The petiole mechanism is the same in kind: `PartTensorTo40DConverter`
ignores our internode directions (pitch is the species constant 20 deg) and
derives the stem's shape from petiole azimuth differences, so the whole shoot is
placed by FK from decoded petiole rotations.

**What this means for the requirement.** The export path integrates a 20-node
shoot from per-node angles that the VAE reproduces to 1-3 degrees; Helios then
amplifies whichever node happens to be worst. Two ways forward, in order of
leverage: (1) **stem IK in the converter** -- solve Helios's per-node
(phyllotactic angle, curvature perturbation, yaw perturbation) from the chain's
node positions, which Stage 2 predicts directly and the identity path has
exactly, instead of from petiole rotations and a decoded curvature; this removes
the FK amplification for every model at once and is the change that makes the
round-trip equal the IK-only column by construction. (2) A VAE that is better on
the ill-conditioned nodes, which is what `_rw4` already is on average and still
lost -- so (2) alone does not get there. Until (1) lands, judge VAE candidates
on packet fidelity (`_rw4` is the best) and treat the Helios figure as a
property of the export, not of the VAE.

**About `_unreferenced/phytomer_9slot_roundtrip_comparison.png`** (Heesup asked why
its columns 2-3 look transparent and why column 3 differs from column 2): that
figure is from 2026-09-11, produced by the 9-slot / 64D-VAE pipeline that no
longer exists (packets have 10 slots, the VAE is 128D, and no script in the
tree references "9slot" any more). Columns 2-3 are the PyTorch soft rasterizer
used by the training render loss -- alpha-blended soft silhouettes on a flat
ground, hence the washed-out look -- not Helios; and column 3 differs from
column 2 because it went through every defect fixed since (leaflet order,
slot-0 rotation, the analytical export's FK drift). `fig14` is the figure that
answers the question now: opaque Helios raytraces, GT / IK-only / VAE
round-trip side by side.

### 2.5 Stem inverse kinematics: the export now follows the nodes (`d3731d3`, `7d92840`, `5827327`)

`diffusion_based/models/part_tensor_stem_ik.py`, run by `assemble_part_tensor_to_xml`
after `PartTensorTo40DConverter` (default on; `PART_TENSOR_STEM_IK=0` for the
plain analytical export). It treats the 14D internode rows -- the parent-node ->
node chords Stage 2 predicts and the identity path has exactly -- as the target
and solves the parameters Helios integrates the stem from: each internode's
curvature and yaw perturbation (elevation and azimuth of its chord), each
petiole's pitch and its node's phyllotactic angle (the petiole's direction), and
each shoot's base pitch, yaw and first-internode length (the first internode,
which Helios never perturbs). It is a fixed-point iteration on the Python mirror
of Helios's phytomer builder (`HeliosPlantGeometryBuilder.extract_part_tensor`,
which now returns per-row poses), so it cannot drift from the FK.

Three versions were needed to make it converge, each recorded in the module
docstring because the failure modes are not obvious: (1) aiming each internode
at its absolute target tip from the current FK base diverges, because a
millimetre of accumulated base error at a 5 mm shoot-tip internode is a
100-degree "correction"; (2) fitting each internode's own direction but
updating all nodes at once diverges too, because Helios builds each internode
relative to the previous one's *final* axis but not to that axis's effect on
the petiole frame, so an upstream correction propagates downstream with
alternating sign; (3) a sequential sweep -- one node, re-run the FK, next node
-- converges in one pass, since every step is then exactly the unit-gain
response measured by finite differences (+1 deg of curvature = +1 deg of
elevation, +1 deg of yaw = +1 deg of azimuth). Cost: one FK per node per sweep,
about 2 minutes for a 164-node plant on CPU; a second sweep changes nothing and
is skipped.

| DAP 50, Helios FG IoU vs GT | analytical export | with stem IK | FK stem residual (mean / max) |
|---|---|---|---|
| identity (VAE bypassed) | 94.2 | **99.4** | 0.47 / 1.20 -> 0.04 / 0.11 cm |
| v9_tl_rw4 | 82.9 | **98.3** | 1.71 / 3.84 -> 0.04 cm |
| v9_tl | 84.7 | 96.8 | 1.35 / 4.41 -> 0.05 / 0.11 cm |
| v8 | 90.2 | 94.2 | 1.51 / 4.31 -> 0.61 cm (before the base step) |
| IK-only (GT 14D, old ceiling) | 94.1 | -- | -- |

The full three-plant figure with `v9_tl_rw4` and the completed solver (base
pitch/yaw/length included, both export columns solved):

| DAP | IK-only, analytical -> solved | VAE round-trip: v8 analytical -> v9_tl_rw4 solved -> v9_tl_rw4_20k solved |
|---|---|---|
| 10 | 95.7 -> 95.7 | 81.4 -> 83.2 -> **92.2** |
| 50 | 94.1 -> **99.5** | 90.2 -> 98.3 -> **98.4** |
| 90 | 87.4 -> **96.7** | 79.3 -> 95.8 -> **95.7** |

DAP 10 is the one place the solver does not help, because a seedling's stem is
13 short internodes and the FK drift there was already small. Its attribution
with the solver on: VAE as-is 83.2; exact petiole rotations 86.1; exact leaflet
scales **95.6** (exact scales on the cotyledon node alone 83.2, on the 12
trifoliate nodes alone 95.6); exact everything 95.9. So the seedling's whole
remaining gap is the trifoliate leaflet scale, 4.4% relative error on those
young nodes against 0.8% at DAP 50 -- the leaves *are* the seedling's
silhouette and `--focus-plant` re-frames on them, so 4% of size is 12 points of
IoU. Young nodes are a small minority of the training packets (a DAP 10 plant
has 13 nodes, a DAP 90 plant 200); `phytomer_vae_v9_tl_rw4_20k` (same recipe,
20,000 files, best val recon 0.0037 vs 0.0065) brings the DAP 10 leaflet scale
error to 1.0% (p90 2%) and the seedling to 92.2 -- the committed figure.

Two readings. First, the identity export now exceeds the old IK-only ceiling,
because IK-only went through the same analytical converter and carried the
same FK drift (0.47 cm mean at DAP 50 on exact input): the ceiling was the
export, not the representation. Second, the VAE ranking is now the packet
fidelity ranking (`_rw4` best), as it should be -- the round-trip finally
measures the VAE. What the solver does not touch: leaflet scales (already one
scalar per node, §2.4), peduncle/flower placement, and the ~1 pixel-level
framing effects of `--focus-plant`.

The unit tests (`tests/test_part_tensor_stem_ik.py`) run the solver on a
dataset seedling with real perturbations and check that it reduces the FK tip
error on exact input and recovers from 3-5 degree noise on curvature, yaw and
phyllotaxy. An unguarded version of the shoot-base step made the exact-input
case *worse* on that seedling (0.13 -> 0.24 cm) -- the finite-difference
Jacobian is degenerate when the child axis is parallel to the parent's --
which is why the base step skips small singular values and caps its step at
1.5x the angular error.

---

## 3. Proposed next step B: the internode redundancy that's STILL open at the packet level

**Two separate redundancies existed, only one is fixed.** ① Stage 2's own node rotation vs the node-to-node position relationship -- fixed this session (§1, the roll-head reduction). ② The phytomer *packet's* slot-0 (internode organ) rotation+length, encoded inside the 128D VAE latent, vs the SAME node-to-node position relationship -- **not yet fixed**.

**Why ② still exists**: `assemble_packets(parent_pos=..., centers=...)` (implemented 2026-09-11, earlier in this session) already *overrides* the decoded slot-0 rotation+length at assembly time for any chained (non-base) phytomer -- so the final geometry is correct. But the VAE itself is still trained to *predict* a meaningful slot-0 rotation+length for every packet, chained or not, because packet construction (`build_phytomer_packets`) and the VAE's `compute_loss` don't know or care whether a given packet is chained. The shape latent therefore still spends some of its capacity (specifically, slot 0's 8D residual channel in the hybrid architecture) encoding a quantity that gets thrown away downstream whenever a parent exists -- which is most phytomers (the earlier-session measurement: only 4-11 shoot bases per plant vs 50-500+ total phytomers).

**Concrete fix** (not yet implemented): extend the existing `strip_base()` pattern (which already zeroes deterministic slots' base columns, both in the encoder INPUT and the reconstruction TARGET) to ALSO zero slot 0's rotation (6D) and length (`scale` index 0, not the radius at index 1) for any packet where `phytomer_idx > 0` (i.e., has a parent within its own shoot -- this is already available directly from the `keys` field, no new computation needed). Radius (`scale[1]`) is not derivable from positions and must stay a real, learned quantity regardless of chaining. This requires:
- A new masking step (call it `strip_chained_internode(packets, keys)`) applied wherever `strip_base` currently is (packet-cache generation, `train_phytomer_vae.py`'s data pipeline).
- `compute_loss`'s existing masking pattern (present-slot weight 1.0, absent-slot weight `absent_weight=0.1`) extended with a third case, or folded into the existing zero-target convention if the loss is already computed against a target that's now correctly zero for these rows -- check whether `compute_loss` needs to know about this distinction explicitly or whether zeroing both input and target is sufficient on its own (likely sufficient, mirroring how deterministic-slot bases work today with no special-casing in `compute_loss` beyond the target being zero).
- A fresh VAE retrain (this would be `phytomer_vae_v9`) and, if this changes measured rotation error for the stem role specifically (currently already the best of all roles at 0.4-0.6 deg, unlikely to improve much further), a re-measurement of the Helios full-cycle roundtrip to see if freed capacity helps other roles (petiole/leaflet, currently 2.3-3.0 deg) -- **not guaranteed to help**, since the residual channel is per-slot and doesn't automatically reallocate freed capacity to a different slot's channel (see the discussion in the 2026-09-11 doc's own hybrid-VAE section about why capacity isn't fungible across per-slot residual channels the way it might be in a single shared bottleneck). Frame this explicitly as "removes a known-wasteful allocation, cleanliness/correctness improvement" rather than "will measurably improve accuracy" when deciding whether it's worth a retrain cycle.

---

## 4. Durable design lesson: do not let two co-evolving quantities chase each other

Raised by the user in the context of a rejected idea (predicting BOTH a phytomer's own position AND an estimate of its parent's position, both refined together, where "parent" means another member of the same evolving set): **if quantity A is being updated to match quantity B, and B is a DIFFERENT part of the SAME thing being updated in the SAME process, A's target moves every step and the whole thing is prone to not converging.**

This project has hit this exact failure mode once already and documented it: "Exp E (in-loop Hungarian): FAILED. Train rot plateaued... assignment flicker (targets move every step) prevents convergence... Lesson: optimal-but-moving targets < stable-but-biased targets, at least without stabilization machinery" (2026-09-09, referenced in the prior doc). Treat any future proposal that has one predicted quantity's target depend on another simultaneously-refined quantity from the same batch/set as suspect by default, and check specifically whether one side of the pair can be made to reference something already-fixed (e.g., Stage 2's position, treated as settled before any parent-relative reasoning happens) rather than something still in flux in the same forward pass.

This does not forbid *asymmetric* designs (e.g., predicting a parent-relative displacement as a function of ALREADY-DECIDED positions, refined by its own small process against that fixed backdrop) -- those are fine. It specifically forbids two things converging toward each other in the same loop with nothing fixed to reference.

### 4.1 The stabiliser, and the two things it does not fix

**The proposal** (user, 2026-09-12): if two quantities chasing each other is the problem, pin one of them -- the parent, say, or the child -- to ground truth during training. Then only one side moves.

**This is sound, and it is exactly the asymmetric shape the paragraph above allows.** It is worth recording as the tool to reach for *if* a future design genuinely needs parent-relative prediction. Three caveats decide whether it is the right tool at the time:

1. **It buys stability with train/inference mismatch.** At training the parent is perfect; at inference it is the model's own, imperfect prediction. A cowpea shoot is tens to hundreds of phytomers long, so error compounding along the chain is precisely the failure mode teacher forcing conceals until the first real rollout. The cheap mitigation is **not** scheduled sampling (which adds a schedule to tune) but **noise injection**: perturb the ground-truth parent by roughly the position error the model actually makes, so the downstream quantity learns to be robust against a wrong parent rather than assuming a right one. Measure the model's own position RMSE first and use that as the noise scale.

2. **It fixes co-evolving VALUES, not a co-evolving ASSIGNMENT.** Exp E did not fail because two values chased each other; it failed because the bipartite *matching* changed every step. Pinning a parent to ground truth cannot help there, because knowing which ground-truth row is "this node's parent" already presupposes a resolved matching. The matching in `hierarchical_hungarian_matcher.py` is still solved fresh every step from predicted node positions against GT cluster centres, so it sits *below* this stabiliser, not above it. The codebase's own answer to assignment flicker was different and is worth copying instead: **warm-starting** from an already-sane checkpoint (`train_phytomer_vae.py`'s `--hungarian-roles` refuses to run without `--init-checkpoint`, for exactly this reason).

3. **There is currently nothing to apply it to.** Once Stage 2's forward axis became a quantity *derived* from already-decided positions (2026-09-11/12, §1) rather than one predicted alongside them, the "A chases B" pair inside Stage 2 disappeared -- topology resolution is `@no_grad`, so nothing flows back into a target. Reach for this only when reintroducing a parent-relative prediction, e.g. if §2's Stage 3 restructuring ever grows one.

---

## 5. Reading order for the next session

1. **`docs/ongoing/AGENT_TAKEOVER_GUIDE.md`** -- general orientation (environment, how to launch jobs locally vs SLURM, established conventions).
2. **`docs/ongoing/20260911_hybrid_vae_rotation_capacity_and_topology_experiments.md`** -- hybrid PhytomerVAE architecture (coarse+residual latent), the pkt-cache `keys` bug, the render-loss internode `parent_pos` fix, why the gradient guard is canary-only, and the topology-recovery measurement that justified the roll-head reduction.
3. **This doc** -- what was actually implemented from that plan (roll-head reduction, validated), and the two next proposed changes (§2 Stage 2/3 boundary, §3 packet-level internode redundancy) plus the moving-target design lesson (§4).
4. **§2.1 and §2.2** -- the adopted Stage 2/3 redesign, what has landed of it, and what has not. This is where to start implementing.
5. Before the next piece of §2.1 (Stage 3 taking `(parent, self)` pairs): read `forward_backward_step` in `train_hierarchical_flow_matching.py` directly to see exactly how `pred_slot_exist_logits` is supervised during TRAINING (not just `sample_ode`'s inference-time t=1 read) -- this determines whether roll/scale can reuse that exact mechanism when they move to Stage 3.
6. Before starting §3's implementation specifically: re-read `phytomer_packets.strip_base()` and `PhytomerVAE.compute_loss()` in full -- the fix is meant to be a small, mechanical extension of the pattern already there, not a new mechanism.
7. **Before launching any training run**: §1.8 (budget the render loss with arithmetic; do not trust `--batch_size auto`) and §1.9.1 (**use `LR=1e-4`**; 3e-4 reliably destroys the run).

**Checkpoint lineage**: `diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/phytomer_vae_128d_best.pt` is the accepted PhytomerVAE for the export/round-trip path (§2.4: 128D = 48 + 10x8, `--rot-weight 4`, 20,000 files, 120 epochs, trained on terminal-last packets -- `PHYTOMER_TERMINAL_LAST=1`, which the eval scripts set for any `_tl` checkpoint). `phytomer_vae_v8` remains what the FM checkpoints and the v6 packet cache were built on; switching the FM to v9 means `PKT_VERSION` 7, a cache regeneration with terminal-last packets, and a Stage 3 retrain. `dataset/cache/cowpea_curv26_pkt/` is at `pkt_version=6` (100,000/100,000, `keys` present). No real (non-smoke) hierarchical FM training has been launched yet. `diffusion_based/checkpoints/fm_smoke_test/hierarchical_fm_epoch_006.pt` is a disposable smoke-test artifact, not a real checkpoint to build on -- delete it before a real run occupies that directory, or point `--output_dir` elsewhere.

**How to launch the v9 run when the GPU quota frees** (2026-09-13): the dataset's cache
gate accepts `pkt_version >= 3`, so no code change is needed -- only the environment:

```bash
sbatch --export=ALL,SEED=1234,LR=1e-4,FORCE_BATCH_SIZE=48,RENDER_GRAD_START_EPOCH=11,RENDER_FRACTION=0.03,EPOCHS=500,\
PKT_CACHE_DIR=dataset/cache/cowpea_curv26_pkt_v9,\
PHYTOMER_VAE_CHECKPOINT=diffusion_based/checkpoints/phytomer_vae_v9_tl_rw4_20k/phytomer_vae_128d_best.pt,\
PHYTOMER_TERMINAL_LAST=1,OUTPUT_DIR=diffusion_based/checkpoints/hierarchical_fm_v9,\
INIT_CHECKPOINT=diffusion_based/checkpoints/hierarchical_fm_depth_ord/hierarchical_fm_epoch_025.pt,RESUME=1 \
  slurm_scripts/train_hierarchical_flow_matching.sh
```

`INIT_CHECKPOINT`/`RESUME` warm-start Stage 1-2 from the v8 lineage's epoch 25 (the
partial optimizer restore covers the new `node_parent_mlp`); Stage 3's latent space
changes with the VAE, so its loss restarts high. Drop those two variables for a
from-scratch run. Do this only once §1.9.2's burst has an answer, or accept that the
run may need the same treatment at epoch ~27.

*2026-09-13 10:00*: `dataset/cache/cowpea_curv26_pkt_v9` is complete and verified
(100,000 files, 0 errors; sampled files all pkt_version 7, terminal-last, 128D
latents). The from-scratch v9 run is queued as the job after the replay (see the
commit log); both wait on the group's GPU quota. If the replay names a cause
before the v9 run reaches epoch ~25, patch and resume it from its latest
checkpoint rather than restarting.

**Recommendation on sequencing §2 vs §3 vs a real training run**: §2 and §3 both change what the FM/VAE checkpoints look like, so either should happen *before* committing to a long real training run, not after (avoid training for hours against an architecture you're about to change again). §3 is the smaller, more mechanical change (a VAE-only retrain, ~3.5 minutes on a TITAN RTX per the v8 precedent) and has no open design questions -- do it first. §2 has one open question (§2's "not yet resolved" paragraph) to settle before writing code. Only after both land does a real, non-smoke hierarchical FM training run make sense.

---

## 6. Commit log (2026-09-11 evening through 2026-09-12)

```
0341be2 feat(topology): derive node rotation's forward axis instead of predicting it
8a98cfd feat(model): shrink Stage 2's node rotation head from 6D to a 1-DOF roll
558d3ee refactor: drop "anchor" vocabulary in favour of node / node position
87ff1fb fix(topology): derive a shoot base's forward axis from its successor
d4a745c refactor: rename the anchor identifiers missed by 558d3ee
02a7e29 feat(train): report ordinal MAE, the metric topology actually depends on
e6e1f32 feat(train): gate topology on the ordinal STEP error, not absolute error
5e34a3d feat(train): make the gradient canary name the parameter it fired on
ab0910f feat(debug): opt-in backward probe on the Stage 2 intermediates
2fac3dc feat(train): add --seed, and fix an UnboundLocalError it exposed
d766a12 feat(launcher): pass SEED through to --seed
582d03a feat(debug): per-layer gradient amplification probe for Stage 2's decoder
684a9f1 fix(model): gradient barrier at Stage 2's query boundary
12c0457 feat(train): write each run's figures into its own log folder
8331abf feat(topology): gt_parent_links — one parent rule, no exceptions
5e10c15 perf(topology): vectorise gt_parent_links and gate it on the internode scale
e902487 feat(topology): every node gets a real parent; delete the +Z fallback
c192f5f fix(train): restore the --resume flag whose parser line had been lost
abce662 fix(inference): draw internodes from the chain, over live nodes only
ae26ccb feat(topology): ordinal is depth from the root; supervise the parent step
c62e995 docs: regenerate fig14 and decompose the Helios round-trip gap
c83d8c1 docs: decompose the seedling round-trip loss; two more refuted causes
ef5802d fix(packets): slot-0 rotation is the reference frame; force it to identity
ff5beb4 fix(packets): emit leaflets as lateral, terminal, lateral
c658416 docs: the packet path is lossless; regenerate fig14 after the leaflet-order fix
2f1691b feat(packets): leaf size is one scalar per node; identify the terminal leaflet per node
b7fce9a docs: packet path exact at all three stages; VAE attribution table; v9 plan
daffc46 docs: the Stage 2 explosion recurred at lr below 1e-4; run resumed from epoch 25 at 5e-5
bd04b9c docs: v9 VAE candidates; the Helios round-trip is bounded by the export's FK, not VAE fidelity
d3731d3 feat(export): stem inverse kinematics so Helios's FK follows the predicted nodes
7d92840 feat(stem-ik): solve each shoot's base pitch/yaw and base-internode length
5827327 fix(stem-ik): guard the shoot-base solve against degenerate Jacobians and oversized steps
```
(Earlier the same day, see the previous doc's own commit log for the hybrid VAE / render-loss / canary-guard commits.)
