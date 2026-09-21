# The 2026-09-19 → 21 measurement campaign

A three-day run of measurement rather than training. It produced a handful of durable results, three
bug fixes, and an unusual number of **corrections to its own earlier conclusions** — which is the
part most worth reading, because several superseded claims are still quoted in older documents.

Thirty-five commits. Everything below is replicated (2–5 runs) on the fixed 20-plant set at
`--steps 200 --sample_seed 0` unless stated.

---

## 1. The measurement protocol was the problem

**The single most consequential finding.** Four checkpoints forming a 2×2 over `stage3_absolute` ×
appearance augmentation, all on 10 000 samples, scored two ways:

| | rasterised | raytraced | gap |
| :--- | ---: | ---: | ---: |
| relative, no aug | 71.23 | 66.53 | −4.70 |
| relative, **aug** | 72.65 | **70.16** | −2.49 |
| absolute, no aug | 72.88 | 49.52 | −23.35 |
| absolute, aug | 73.82 | 58.31 | −15.50 |

**Rasterised spreads these four over 2.1 points; raytraced spreads them over 21.8, and nearly
reverses the order.** The rasterised cache render is produced by the same renderer that computes the
training render loss, so input, loss and evaluation form one closed loop — the model is graded on
its own training domain.

A day later the full-data checkpoints sharpened this: they score **lower** rasterised and
substantially **higher** raytraced, the signature of reduced memorisation. So the rasterised
protocol does not merely fail to discriminate — **it rewards memorising the eval set.**

**Protocol from here:** report raytraced alongside rasterised for anything comparing architectures or
training recipes. Rasterised stays valid for things acting on geometry after the image is consumed —
refinement step count, optimiser, regularisation weights.

## 2. Refinement is nondeterministic, and it cannot be switched off

An identical command reproduces the flow's raw sample bit-for-bit but not the refined result — one
seedling scored 41.0 and 9.7 on two runs. PyTorch's determinism switches do not fix it and raise no
warning, which places the cause in nvdiffrast's custom CUDA kernels.

It is **sensitive dependence, not a large noise source**: plants sitting at a bifurcation of the
loss surface swing tens of points, while the 20-plant mean is stable to **SD 0.58**. A DAP 3 seedling
was bit-identical across three runs; a DAP 2 one was not. So "seedlings are noisy" is wrong — plants
near a bifurcation are, and seedlings are likelier to be among them.

**Rule: a single-run difference on 20 plants must exceed ~1.6 points to mean anything, and no
per-plant claim is safe without replicates.**

## 3. Refinement levers are seedling levers

Across nine configurations, mature plants (DAP > 15) sit flat at 79.8–82.2 while seedlings span
26.2–48.0. `--reg_latent 0` is worth **+2.04 overall** [95% CI +1.2, +2.9] and **+5.8** on a
purpose-built 24-seedling set. Raising render resolution **hurts** (−2.0 at 256 px): a sharper
silhouette narrows the rasteriser's gradient support band, so an organ must already overlap to feel
any pull.

## 4. The latent discretises per coordinate, but not as a whole

Replacing every phytomer latent with one of K prototypes saturates near 57 % and is still 35 points
down at K = 256 — **there is no small set of phytomer types.** But quantising each of the 128
coordinates independently costs only **2.4 points at 64 levels**. That would turn the shape readout
from a 128-D regression into 128 softmaxes, making the measured mean-collapse (R² +0.001)
structurally impossible.

## 5. The render residual carries no recoverable correction signal

Displacing visible organs by a *known* delta and ridge-regressing that delta from a local patch of
the render residual gives held-out R² of −0.004 to +0.021, even at 10 cm. Feeding a raw residual
patch back as per-node conditioning is not the design to build. Caveat: a linear probe on raw depth
patches bounds the easily accessible signal, not what a CNN could extract.

---

## Three bugs, in ascending order of how much they cost

**An empty plant crashed the evaluation.** A plant materialising zero organs broke the refinement
loop before `loss` was assigned, killing the run — and that is exactly the appearance-collapse
failure mode, so it targeted the one comparison it would invalidate. The naive fix is worse than the
bug: `loss = 0` is the *best* possible score, so a collapsed plant would outrank every real
hypothesis and `--n_starts` would deliberately select the empty one. It now starts at `+inf`.

**Absolute checkpoints were evaluated through the relative rotation path.** `sample_ode` returns the
node's own **rot6d (6)** in `phytomer_roll` under `--stage3_absolute`; relative carries roll (2). The
eval never branched on the layout, so `roll_to_matrix` normalised all six together, read only the
first two as (cos, sin), discarded the rest, and rebuilt the forward axis **from the chain** — what
the absolute layout exists to avoid. Re-running all 18 evaluations moved rasterised −0.46 to +2.35
and raytraced −0.80 to +1.20, mostly inside the noise floor, so **every conclusion survived**. Raw
moved more (+1.2 to +2.6), the expected shape.

**The botany loss aligned the wrong rotation column.** The convention is COLUMNS — `roll_to_matrix`
returns `stack([x, forward, z])`, `matrix_to_rot6d` keeps the first two — so a rot6d is
`[x | forward]`. Measured over 167 GT internodes against the parent→node direction: `rot6d[0:3]`
gives **90.2°**, `rot6d[3:6]` gives **1.1°**. The hybrid's botany term used `[0:3]`, driving each
node's *perpendicular* axis toward the branch direction at weight 1.0. The loss whose whole job was
to restore the structure the relative layout carries for free was fighting it. **Absolute runs only**
(guarded by `if s3abs`), so the relative lineage was never affected — but **the absolute layout has
never been trained with a correct botany constraint**, which makes every absolute number to date a
floor rather than a verdict.

---

## Corrections to this campaign's own conclusions

| Claimed | Corrected to | Why |
| :--- | :--- | :--- |
| `--reg_latent 0` is worth +18 seedling points | **+5.8** [+1.6, +10.1] | n=4 did not replicate; measured on a purpose-built 24-seedling set |
| The prior ladder is cleanly monotone | only zero separates | 0.25 and 2.0 sit inside the noise floor |
| The scale prior is worth 1.7 points | not shown | the gap is 0.8, inside noise |
| `--reg_latent 0` matches Gate G at ⅛ the compute | Gate G is ~1 point better | the single runs that tied them differed by less than the noise floor |
| **Gate G is the strongest lever** | **rasterised-specific** | +3.42 at 6.8 SE rasterised, +1.08 at 1.3 SE raytraced — not resolved |
| **The merged architecture ties the baseline** | **a rasterised artifact** | 73.37 vs 71.23 rasterised; 59.11 vs 66.53 raytraced |
| `stage3_absolute` costs 20.3 points of gap | overstated ~3× | measured entirely at 10k; at full data the gap is −5.93 |
| Merging was worth ~10 points | **+4.26 raytraced refined** | the 10-point figure was the *raw* prediction; refinement always ships |
| Every checkpoint is "merged" / a merge was never run | **both, under two meanings of one word** | see below |

The recurring pattern is worth naming: **refinement compensates for most of an architectural
difference**, so any architecture claim measured before refinement overstates what the change is
worth deployed. And **the rasterised protocol compresses real differences toward zero**, so anything
measured only there understates them.

## Vocabulary settled

Four terms were fixed during the campaign, each because the old one named the wrong thing:

- **relative / absolute**, not "plain / absolute" — "plain" names nothing; it implied the relative
  layout was a default rather than a parameterisation carrying the chain structure as an inductive
  bias, which is most of why it generalises better from a small subset.
- **rasterised / raytraced**, not "flat / Helios" — "flat" named an incidental appearance when the
  load-bearing fact is provenance, and "Helios" was ambiguous because the rasteriser's own class is
  `HeliosPyTorchRenderer`.
- **bulk parameters → phytomer parameters → render**, not stage numbers — the paper's own vocabulary;
  the numbering was never stable and hid that only three parts are learned and that refinement is
  not a stage at all.
- **"merged" retired entirely.** It meant `--stage3_geometry` (the flow predicts geometry — every
  checkpoint has this) and it was read as "the two geometry predictions became one" (**never built**).
  The coarse stage still predicts position, roll and scale in every run and the flow overwrites it at
  sampling: a cascade, not a merge. The 2026-09-12 design proposed dropping `roll_head`/`scale_head`
  once the flow carried geometry; only the first half shipped, and no flag exists for the second.

---

## Best configuration measured

**relative + appearance augmentation + Gate G = 71.24** on the raytraced protocol
(`sub10_v10_cam_aug` ep230, `--n_starts 8`, 200 refinement steps). Quote this from the raytraced
column; the rasterised figure flatters the wrong designs.

## Running

`relative_fullaug`, `absolute_full` and `absolute_fullaug` — all three on the full 100 000 plants
from `v10_cam` ep160, 60 epochs each. The two absolute runs were restarted from the clean warm start
after the botany fix rather than resumed from checkpoints trained under the broken term. They will
give the **first fair relative-vs-absolute comparison at full data**, with a correct botany
constraint and the fixed evaluation path.

An operational note that cost 42 GPU-hours: long runs die at ~21 h to an **NCCL watchdog hang**, and
neither `--requeue` (which covers preemption, not a non-zero exit) nor `AUTO_RESUME=1` (which only
helps on a restart someone performs) catches it. Mitigated with
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`; the real fix — a one-GPU job should build no process group
at all — is deferred until the queue is clear.

## Open, in priority order

1. **Replace gradient refinement with a closed-form solve.** Zhang et al. (IJCAI 2025) show the SDS
   gradient reduces to `(P_t − P*)` for rigid transforms, so ICP solves it directly. That removes the
   nondeterminism, the 200 backward passes, and the narrow-gradient-band failure at once, and the
   correspondences already exist (`collect_part_visibility` + the depth channel).
2. **Ablate the coarse geometry heads** — the duplication the 2026-09-12 design planned to remove and
   nobody has measured. No flag exists yet.
3. **Discrete per-coordinate shape readout**, at the measured cost of 2.4 points.
4. **Seedlings**, which every refinement lever turned out to be about, still at ~19 % raw.
5. **Generative structure**, never measured: every comparison to date scores reconstruction, not
   whether sampled plants are structurally plausible. The probe exists
   (`eval_generative_structure.py`) and found the botany bug before producing a single result.
