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

## 2. Proposed next step A: move roll + scale prediction from Stage 2 to Stage 3

**The question that motivated this** (user, 2026-09-12): does Stage 3 actually need Stage 2's roll and scale as conditioning at all?

**Answer: very likely no, for roll; probably not required for scale either, but weaker case.** The reasoning ties directly to how packets are built: base is cluster-center-relative, rotation is phytomer-reference-frame-relative (`R_ref^T @ R_org`), and scale is normalized by `s_a`. The 128D shape latent is therefore designed to be **invariant** to the phytomer's absolute world-frame roll and absolute size by construction -- conditioning Stage 3 on Stage 2's *absolute* roll to help predict a *roll-invariant* target is very likely giving it information with no relationship to the thing it's predicting. Scale has a weaker justification (absolute size correlates with developmental stage, which could carry soft information about class distribution/curvature even in a size-normalized target), but is not a hard requirement either.

**Position is the one exception, for a different reason**: `PhytomerVisualProjector` uses `phytomer_pos` to sample a *local* image-token feature at that specific 3D location -- this is about visual grounding (what does the image actually show here), unrelated to whether roll/scale help predict shape. Position conditioning into Stage 3 should stay.

**Proposed restructuring**:
- **Stage 2 (skeleton only)**: position, existence, ordinal, is-shoot-base. Drop `roll_head` and `scale_head` from `CoarseSkeletalTransformer` entirely.
- **Stage 3 (everything about this phytomer's own appearance)**: given position + image conditioning, produce roll, scale, AND the 128D shape latent. Roll/scale should NOT be flow-matched (that would reopen exactly the "two moving targets" risk in §4) -- add them as small auxiliary feedforward heads on `PhytomerFlowMatchingDecoder`, read at t=1 from the converged query features, using **exactly the existing pattern `pred_slot_exist_logits` already uses** (`exist_head` reads the decoder's internal features `x` directly, evaluated once via `final_out` at `timesteps=1.0` in `sample_ode`, never part of the flow state itself). Add `roll_head`/`scale_head` the same way, alongside `exist_head`.

**Why this is lower-risk than it might look**: it doesn't touch the ODE state at all (still pure 128D latent, `N(0,I)` prior, unchanged) -- it only moves WHERE two small feedforward predictions are computed from (Stage 2's earlier, less-contextualized features -> Stage 3's later features, which have been through more self-attention across phytomers and more cross-attention to the image). The existence-head precedent proves this pattern is already safe in this codebase.

**Scope estimate**: comparable to the roll-head reduction just completed -- touches `CoarseSkeletalTransformer.__init__`/`forward` (remove 2 heads), `PhytomerFlowMatchingDecoder.__init__`/`forward` (add 2 heads + a `final_out`-style t=1 readout path, which doesn't cleanly exist for phytomer mode inside `forward()` itself the way `sample_ode` currently does it only at the end -- needs a decision on whether the *training* loop also needs a final/t=1 evaluation of roll/scale, or whether it's acceptable to read them from an arbitrary-t query as an approximation during training and only trust the t=1 read at inference; look at how `pred_slot_exist_logits` is actually supervised during training before assuming the precedent transfers exactly), training loop (move `loss_phytomer_roll`/`loss_phytomer_scale` targets and their matched-anchor lookup to read from the Stage-3 output instead of Stage-2's, remove the two conditioning MLPs `node_roll_mlp`/`node_scale_mlp` from `PhytomerFlowMatchingDecoder` since Stage 3 no longer receives them as input).

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

This does not forbid *asymmetric* designs (e.g., predicting a parent-relative displacement as a function of ALREADY-DECIDED positions, refined by its own small process against that fixed backdrop) -- those are fine. It specifically forbids two things converging toward each other in the same loop with no anchor.

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
```
(Earlier the same day, see the previous doc's own commit log for the hybrid VAE / render-loss / canary-guard commits.)
