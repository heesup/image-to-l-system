# Two probes: can the phytomer latent be discretised, and is the render residual informative?

2026-09-19. Both run without training, on `hierarchical_fm_v10_cam/hierarchical_fm_epoch_160.pt`.
Scripts: `plant_recon/eval/eval_latent_quantization_ladder.py`,
`plant_recon/eval/eval_render_residual_probe.py`. Data: `outputs/logs/20260919/{quant_ladder,rv_*}.json`.

## 1. The latent discretises cleanly per coordinate, and not at all as a whole

Every Stage 3 failure this project has measured is a failure of *continuous regression*: shape is
not recoverable from a nadir view (R^2 ~ 0), a regression head collapses to the dataset mean exactly
(R^2 +0.001), and refinement scores higher with the latent prior switched off entirely (+2.0). A
discrete readout cannot collapse to a mean — there is no average of two classes to fall into — so
the question is whether discretising costs anything.

120 plants scored, codebook fit on 400 *different* plants. Reference is the IK-only reconstruction,
because the VAE round-trip is required to reproduce it (a hard requirement of this project); at
92.20 the round-trip is healthy, so the numbers below are readable.

| k | whole-latent VQ | delta | per-dimension | delta |
| ---: | ---: | ---: | ---: | ---: |
| exact | 92.20 | +0.00 | 92.20 | +0.00 |
| 2 | 37.56 | −54.64 | 52.95 | −39.25 |
| 4 | 40.48 | −51.72 | 62.42 | −29.78 |
| 8 | 49.50 | −42.70 | 72.69 | −19.51 |
| 16 | 50.86 | −41.34 | 80.90 | −11.30 |
| 32 | 53.09 | −39.11 | 86.66 | −5.54 |
| 64 | 54.23 | −37.97 | 89.80 | **−2.40** |
| 128 | 56.34 | −35.86 | 91.28 | **−0.92** |
| 256 | 57.00 | −35.20 | 91.95 | −0.25 |

**There is no small set of phytomer types.** Whole-latent vector quantisation — one symbol per
phytomer, k shapes for the entire species — saturates near 57 and is still 35 points down at k=256.
Any design that classifies a phytomer into one of K prototypes is refuted by this column.

**Per-coordinate discretisation is nearly free.** k=64 costs 2.4 points and k=128 costs 0.9. So the
128-D continuous latent can be replaced by 128 independent 64-way choices with almost no geometric
loss, which converts Stage 3's readout from a 128-D regression into 128 softmaxes and makes the
measured mean-collapse structurally impossible. The cost is 128 x 64 = 8192 logits per phytomer.

The two columns answer different questions and only the second is the analogue of quantising
individual scalar parameters: k levels per coordinate leaves k^128 configurations reachable, where
one symbol per phytomer leaves k.

## 2. The local render residual carries no recoverable displacement signal

The other idea on the table is closing the loop — feeding `render(hypothesis) - input` back as
*conditioning* rather than only as a loss. Training already carries a render loss (Stage 4,
`--render_fraction`), but at inference the model never sees its own render, which is why 200 steps
of refinement still add ~33 points on top of it.

The probe displaces every organ by a KNOWN delta, renders, normalises both depths per plant,
differences them, and ridge-regresses each organ's delta from a local patch of that residual. If a
linear probe cannot recover a delta it was told about, a learned loop will not recover one it has to
discover.

A first round was run and discarded: it included organs that paint no pixels at all (53.4% of them,
per the 2026-09-18 visibility probe), whose displacement cannot be in the residual by construction,
and it displaced every organ at once so each patch was a superposition. Both were fixed
(`--visible_only`, `--perturb_frac`). Held-out R^2 for delta x,z from the residual:

| run | displacement | R^2 |
| :--- | :--- | ---: |
| `rv_vis_s2` | 2 cm, all organs, visible only | −0.0040 |
| `rv_iso_s2` | 2 cm, 20% of organs, visible only | −0.0272 |
| `rv_iso_s5` | 5 cm, 20%, visible only | −0.0078 |
| `rv_iso_s10` | 10 cm, 20%, visible only | **+0.0211** |

Null throughout. Even a 10 cm displacement of a visible organ, with few other organs moving to
interfere, yields 2% of variance. The control (input render alone) also sits at ~0, as it must since
the delta is drawn independently of the plant, which confirms nothing is leaking.

**Limit of this result.** This is a *linear* probe on *raw depth patches*. The Stage 3 conditioning
probes that settled their question ran on DINOv2 features — already a deep representation — so a
null there was strong. A convolutional encoder could extract more than ridge on raw pixels. What
this bounds is the easily accessible signal, and it is enough to say that feeding a raw residual
patch in as local per-node conditioning is not the design to build.

**What it does not argue against.** It says nothing about the ICP route (Zhang et al., IJCAI 2025),
which does not learn a map from residual to correction at all — it solves geometric correspondence
in closed form. The two mechanisms are unrelated, and this null makes the closed-form one relatively
more attractive, not less.
