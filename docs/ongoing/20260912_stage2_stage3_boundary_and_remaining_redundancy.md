# Session Handoff: Roll-Head Reduction Validated, Stage 2/3 Boundary Redesign Proposed, One Redundancy Still Open

- **Author**: Claude (pair programming with Heesup Yun)
- **Date**: 2026-09-12 (continuation of 2026-09-11's work)
- **Status**: Stage 2 rotation head (6D -> 1-DOF roll) implemented, tested, and validated by a real 6-epoch training smoke test through the epoch-4 render-gate danger zone (0 recovery skips, 0 canary hits). **NEXT (not yet implemented)**: two concrete design changes below, both discussed and agreed in principle but not yet coded.
- **Read this doc's own §5 first if you are starting a new session** — it lists exactly which prior docs to read and in what order.

---

## 1. What changed and was validated since the previous doc

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

### 1.9 OPEN: an intermittent Stage 2 gradient explosion, localised but not root-caused

**Status: blocking a real training run.** Read this before launching one.

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

**State at the end of 2026-09-12: open, well-instrumented, eight hypotheses refuted, no root cause.** No job is left running. What a next session has to work with:

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

**Prediction, and the test now running**: capping `lr` below ~1.5e-4 should remove the explosion. Job `38240281` is `LR=1e-4` with everything else unchanged from `38240158` (`SEED=1234`, batch 48, render deferred to epoch 40, `SAVE_EVERY=10`). If it clears epochs 3-6, the mechanism is optimisation dynamics and the fix is a schedule, not an architecture change -- and the query barrier from `684a9f1` should then be re-examined and probably removed, since it would no longer be load-bearing.
2. If it survives, reproduce deterministically with `--seed`, then bisect inside the decoder by extending `_probe_grad` to each layer's input and output -- the amplification is somewhere in those four layers.
3. Only then consider re-introducing bounded per-element clipping. Note that the previous session *removed* a blanket +-100 per-element pre-clamp specifically to stop it masking leaks, and this is plausibly the leak it was masking -- so re-adding it would hide a real defect and should be a deliberate, documented choice, not a reflex.

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
4. Before starting §2's implementation specifically: read `forward_backward_step` in `train_hierarchical_flow_matching.py` directly to see exactly how `pred_slot_exist_logits` is supervised during TRAINING (not just `sample_ode`'s inference-time t=1 read) -- this determines whether moving roll/scale to Stage 3 can reuse that exact mechanism.
5. Before starting §3's implementation specifically: re-read `phytomer_packets.strip_base()` and `PhytomerVAE.compute_loss()` in full -- the fix is meant to be a small, mechanical extension of the pattern already there, not a new mechanism.

**Checkpoint lineage**: `diffusion_based/checkpoints/phytomer_vae_v8/phytomer_vae_128d_best.pt` is the current accepted PhytomerVAE (unchanged by this doc's work -- §3's fix would produce v9). `dataset/cache/cowpea_curv26_pkt/` is at `pkt_version=6` (100,000/100,000, `keys` present). No real (non-smoke) hierarchical FM training has been launched yet. `diffusion_based/checkpoints/fm_smoke_test/hierarchical_fm_epoch_006.pt` is a disposable smoke-test artifact, not a real checkpoint to build on -- delete it before a real run occupies that directory, or point `--output_dir` elsewhere.

**Recommendation on sequencing §2 vs §3 vs a real training run**: §2 and §3 both change what the FM/VAE checkpoints look like, so either should happen *before* committing to a long real training run, not after (avoid training for hours against an architecture you're about to change again). §3 is the smaller, more mechanical change (a VAE-only retrain, ~3.5 minutes on a TITAN RTX per the v8 precedent) and has no open design questions -- do it first. §2 has one open question (§2's "not yet resolved" paragraph) to settle before writing code. Only after both land does a real, non-smoke hierarchical FM training run make sense.

---

## 6. Commit log (2026-09-11 evening through 2026-09-12, roll-head portion)

```
0341be2 feat(topology): derive node rotation's forward axis instead of predicting it
8a98cfd feat(model): shrink Stage 2's node rotation head from 6D to a 1-DOF roll
558d3ee refactor: drop "anchor" vocabulary in favour of node / node position
87ff1fb fix(topology): derive a shoot base's forward axis from its successor
```
(Earlier the same day, see the previous doc's own commit log for the hybrid VAE / render-loss / canary-guard commits.)
