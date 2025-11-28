# auon-optimizer-mlx

A small MLX implementation of AuON, the method in
*AuON: A Linear-time Alternative to Orthogonal Momentum Updates*
(arXiv:2509.24320, submitted 29 September 2025). The paper asks what actually makes orthogonal momentum updates
such as Muon work: applying momentum first and then flattening the singular
values of the update equalises the step in every direction, but the orthogonal
projection costs a polynomial number of matrix products and pushes a layer with
a spiky gradient just as hard as a well-behaved one. AuON replaces the
projection with a linear-time rescaling of the *normalised* momentum by a
hyperbolic-cosine statistic of its own entries, which the paper calls an
emergency brake on heavy-tailed (spiky) updates, and keeps a spectral
trust-region guarantee. Hybrid-AuON puts the Newton-Schulz iterations back for
the matrix parameters only.

This repo is a small independent implementation written from the paper, and it
is **not** a reproduction of the paper's results. The paper trains a
40M-parameter nanoGPT on SmolLM-Corpus on 4xL4 GPUs, a 168M-parameter decoder on
FineWebEDU-10B on an H100 and a 19.9M CNN on CIFAR-10. Nothing here is
comparable: the model is a 182,112-parameter 2-layer transformer trained on a
synthetic induction/copy task for 400 steps on an Apple-silicon **CPU**, and at
this scale every arm that gets a tuned learning rate solves the task, so the
interesting measurements are how *fast* they get there, what the attention
logits do on the way, and how the per-step cost scales with matrix size.

## What is in here

### The update and its two scalings — `auon/normalizers.py`

For a momentum buffer `m` of a 2-D parameter with `N = numel(m)`, AuON computes

```
x = m / (||m||_F + 1e-7)                              # unit-Frobenius direction
r = (mean(cosh(gamma * x) ** 2)) ** brake_exponent     # the cosh "brake" statistic
U = rho * x / (r + 1e-8)                               # spectral trust-region step
```

`gamma = N ** temperature_alpha` is the paper's temperature, `brake_exponent` is
how hard the brake bites, and `rho` is the trust-region radius. Nothing in this
pipeline iterates and there is no inner-loop hyperparameter: five elementwise
passes plus one reduction, i.e. `O(N)`.

The paper writes the same object in two normalisations. Vanilla AuON (its
section 3.1, and the "Vanilla AuON" line of section 4.2) uses `gamma = 1` and
`r = sqrt(mean(cosh(x) ** 2))`, i.e. `temperature_alpha = 0, brake_exponent =
0.5`; the temperature-scaled variant used for its language-model runs (section
3.3 eq. 5, section 4.2) uses `gamma = N ** 0.48` and raises the statistic to a
power. Both are exported as `auon.optim.VANILLA` and
`auon.optim.TEMPERATURE_SCALED`, and both are implemented by the same three
lines.

**Trust region.** Since `cosh >= 1`, `r >= 1` for any non-negative exponent, and
since `||x||_2 <= ||x||_F <= 1`, every update satisfies `||U||_2 < rho` — for
vanilla AuON, for the temperature-scaled variant and for Hybrid-AuON (which
re-normalises by the Frobenius norm after Newton-Schulz). The paper proves a
weaker version of this in its section 3.4; the version above needs no
assumption about the distribution of the entries. The tests check it by power
iteration over many random matrices, and check that it is *tight*: for a 100x100
rank-one gradient with `gamma = 1`, `||U||_2` reaches `rho` to within 0.1%.

**The inference this repo makes about the brake.** With `gamma = 1` the argument
of `cosh` is a Frobenius-normalised entry, so `|arg| <= 1` always: the brake
statistic is bounded by `cosh(1) = 1.543`, and vanilla AuON can never shrink the
step by more than that (by `cosh(1) ** (2 * brake_exponent) = 4.58` at the
temperature-scaled exponent). The exponential suppression of the paper's
sections 3.5.1-3.5.3 (`r ~ exp(|xi|)`, `||U||_2 -> 0`) therefore cannot come
from `gamma = 1`. It needs the temperature scaling: section 3.5.1 writes
`gamma = sqrt(N)` and sections 3.3/4.2 write `gamma = N ** 0.48`, which is the
same thing to within the exponent. We measure exactly this in
`results/experiment.json` (`spike`): with `gamma = N ** 0.48` a single spike
takes the update norm from 4.592e-2 to 1.473e-7 and then to 0, while with `gamma = 1`
the same spike takes it from 0.247 to 0.9998 — i.e. vanilla AuON drifts *towards*
the trust-region boundary instead of braking.

### The optimizers — `auon/optim.py`

`AuON` is an `mlx.optimizers.Optimizer` whose state is one momentum buffer per
parameter and whose `apply_single` is the seven-line pipeline

```
m <- momentum * m + (1 - momentum) * g      # EMA momentum
W <- W * (1 - lr * weight_decay)            # decoupled weight decay
x <- m / (||m||_F + 1e-7)
gamma <- N ** temperature_alpha
r <- (mean(cosh(gamma * x) ** 2)) ** brake_exponent
U <- rho * x / (r + 1e-8)
W <- W - lr * U
```

`HybridAuON` is the same class with `ns_steps` Newton-Schulz iterations
(`auon.normalizers.newton_schulz`, coefficients `(3.4445, -4.7750, 2.0315)`,
cycling through a coefficient list) inserted before the normalisation, and the
paper's `AuON+`/`Hybrid-AuON(NS=3)`/`(NS=5)` are `ns_steps = 1/3/5`. It is also
the only place in the package with an iteration count. `Muon` is the baseline:
momentum, then Newton-Schulz, then the shape factor `sqrt(max(1, m/n))`, and no
cosh brake. `build_optimizer(name, lr)` returns the arm, wrapped in MLX's
`MultiOptimizer` so that embeddings, the LM head and 1-D parameters go to Adam
and only the hidden matrices get the rule under test — the split of the paper's
section 4.2.

### The model and the telemetry — `auon/model.py`

`TinyLM` is a pre-norm decoder-only transformer: learned token and positional
embeddings, RMSNorm, 4 heads, a GELU MLP, 2 blocks, `d_model = 96`, 182,112
parameters. Attention is written out by hand so that the **largest absolute
pre-softmax attention logit** (over heads, positions and layers, restricted to
the causal region) can be reported per step; that is the quantity the paper
tracks when it talks about exploding attention logits. The telemetry forward is
a second forward pass with the same weights, so it is deterministic but does add
one forward to every training step (for every arm equally).

### The task — `auon/data.py`

The induction/copy task: `[BOS] a [SEP] a`, where `a` is 12 random tokens from a
32-token vocabulary, and the loss is computed only on the 12 predictions of the
second copy. Solving it needs an induction head, so the attention pattern is
load-bearing rather than decorative; chance is 1/32. The eval set is 256 fixed
sequences (3,072 scored predictions). A small built-in character corpus
(`CORPUS`, 1,516 characters written for this repo) is included with the same batch
interface but is not used by the experiment.

### The training loop and cost model — `auon/train.py`, `auon/scaling.py`

`train_arm` runs one optimizer on one model and logs per-step loss, gradient
norm, maximum attention logit, the RMS of the actual parameter change, wall time
and the optimizer's own share of it (the forward and backward pass are evaluated
outside the optimizer timer). `auon/scaling.py` times a single
`apply_gradients` call on one square matrix and fits the exponent of
`time ~ side ** slope`, with an analytic multiply-accumulate proxy alongside it
(5 passes over the entries for AuON, `3 * steps` matrix products for
Newton-Schulz).

## Checks

`pytest` runs 93 tests. The ones that carry the weight:

* **Trust region.** `test_spectral_norm_of_the_update_never_exceeds_rho` runs 24
  random matrices (8 shapes x 3 magnitudes) for each of 2 radii and 3
  scalings and verifies `||U||_2 <= rho` with
  power iteration (not with the algebra it is supposed to test);
  `test_spectral_bound_is_tight_for_a_large_dense_matrix` and
  `test_trust_region_bound_is_tight_for_a_large_dense_update` show the bound is
  reached to within 0.1%, so it is not vacuous;
  `test_applied_update_stays_inside_the_spectral_trust_region` does the same on
  the update the optimizer actually applies, over several steps of momentum
  accumulation.
* **The neutral element.** `test_neutralised_scaling_reduces_to_plain_normalised_momentum`
  (three parameterisations of the identity setting) and
  `test_null_temperature_is_an_exact_identity` /
  `test_zero_brake_exponent_is_an_exact_identity` assert that with `gamma = 0` or
  `brake_exponent = 0` AuON is exactly `rho * m / (||m||_F + 1e-7) / (1 + 1e-8)`,
  to 1e-6, over six steps of momentum accumulation.
* **The pipeline written twice.**
  `test_single_step_matches_a_numpy_reimplementation_of_the_paper_equations`
  recomputes the whole seven-step pipeline in NumPy, including the decoupled
  weight decay and the `N ** 0.48` temperature, for two consecutive steps.
* **No iteration count.** `test_auon_has_no_iteration_count_and_only_a_momentum_buffer`
  asserts the optimizer has no `ns_steps` attribute and that its per-parameter
  state is exactly `{"m"}`.
* **The baseline is really Muon.**
  `test_muon_matches_the_mlx_builtin_muon_step_for_step` reproduces
  `mlx.optimizers.Muon` (independent code, same published operator) to 1e-5 over
  three steps and three shapes, and
  `test_newton_schulz_matches_the_mlx_builtin_muon_operator` does the same for the
  Newton-Schulz operator itself.
* **Is the baseline semi-orthogonal?** Approximately, and the tests say how
  approximately. With Muon's standard single coefficient trio, five iterations
  leave the singular values spread around 1 rather than at 1 — the same operator
  MLX's Muon applies, and the one NVIDIA's emerging-optimizers documentation
  describes as "noisy values around 1". The tests therefore assert the spectral
  norm lies in 0.9-1.3 and is invariant to the gradient scale, that the singular
  values stay in 0.5-1.3 with the spectrum compressed, and that the residual
  `||Q^T Q - I||_max` is *not* small (> 0.05). Separately they assert that the
  per-iteration quintic coefficients reach a residual below 0.15 in five steps
  and that the textbook quintic converges to the polar factor to 1e-3.
* **Emergency brake.** `test_temperature_scaled_brake_suppresses_a_spike_exponentially`
  asserts monotone damping and a 1e4-fold collapse of the update norm;
  `test_vanilla_brake_cannot_shrink_the_step_by_more_than_cosh_one` pins the
  limitation above; `test_huge_gradient_is_damped_and_never_produces_nan` checks
  the same through the optimizer and compares with Muon on the same gradient
  (AuON 1e-3, Muon > 0.5).
* **Everything stays finite.** `test_every_arm_stays_finite_and_bounded_under_extreme_gradients`
  pushes gradients of magnitude 1e9 through all six arms: every parameter stays
  finite and inside the trust region (`||dW||_2 <= lr * rho`) for the AuON arms.
* **The instrument itself.** `test_max_attention_logit_matches_a_manual_recomputation`
  recomputes the attention logits in NumPy from the model's own weights and
  compares, `test_attention_logits_are_causal` checks that changing the last
  token cannot move earlier logits, and `test_cross_entropy_only_counts_masked_positions`
  checks the masked loss against hand-computed values.
* **The README itself.** `tests/test_results.py` checks that every numeric cell
  of the four tables below is a rounding of a number in
  `results/experiment.json`, that every headline metric in the JSON is quoted in
  the tables, and that the paper's own benchmark numbers appear only inside the
  "nothing here reproduces" bullet. It is the honesty checklist of the repo's
  house style, mechanised.
* **Task and loop contracts.** The induction layout, the mask covering exactly
  the second copy, seeded determinism, `lr_at`'s warm-up/decay values, the
  history contract (lengths, `peak == max`), a real end-to-end loss decrease on
  the copy task, and bit-identical losses for two runs on the CPU.

## Experiment

`scripts/experiment.py`, one run, 452.6 s total on the CPU device. Identical
model initialisation and identical batch order per seed; only the rule applied
to the hidden matrices changes. Six arms, 3 seeds, 400 steps, batch 32, and the
learning rate tuned per arm on seed 0 *on the same 400-step budget* (the grid is
in the log). All numbers below are copied from `results/experiment_log.txt` and
`results/experiment.json`.

### Table 1 — the toy induction task (mean +- std over seeds 0, 1, 2)

| arm | lr | final loss | final accuracy | accuracy @100 steps | peak max attn logit | ms/step | optimizer ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| `adam` | 0.003 | 0.0002 +- 0.0000 | 1.0000 | 0.9978 | 248.5 +- 37.0 | 17.88 | 1.83 |
| `sgd_momentum` | 0.01 | 0.0000 +- 0.0000 | 1.0000 | 0.9998 | 59.4 +- 26.2 | 16.90 | 1.34 |
| `muon` | 0.005 | 0.0000 +- 0.0000 | 1.0000 | 1.0000 | 147.0 +- 23.0 | 21.05 | 5.16 |
| `auon` (vanilla) | 0.005 | 0.0000 +- 0.0000 | 1.0000 | 1.0000 | 24.2 +- 2.7 | 23.27 | 3.66 |
| `auon_temp` | 0.02 | 0.0015 +- 0.0008 | 0.9998 +- 0.0002 | 0.0335 | 40.0 +- 5.3 | 21.70 | 3.67 |
| `hybrid_auon` (NS=5) | 0.005 | 0.0000 +- 0.0000 | 1.0000 | 1.0000 | 14.0 +- 0.4 | 23.92 | 6.98 |

"The peak max attn logit" is the largest causal pre-softmax attention logit seen
in any training batch during the 400 steps; chance accuracy is 0.03125.

### Table 2 — cost of one optimizer step on a single `n x n` matrix (ms, CPU)

| arm | 128 | 256 | 512 | 1024 | 2048 | 4096 | slope (128-4096) | slope (1024-4096) | MAC proxy @1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `adam` | 0.161 | 0.219 | 0.597 | 2.131 | 8.635 | 34.534 | 1.61 | 2.01 | 4.2e6 |
| `sgd_momentum` | 0.113 | 0.118 | 0.319 | 0.927 | 3.781 | 13.875 | 1.46 | 1.95 | 4.2e6 |
| `auon` | 0.224 | 0.598 | 2.086 | 7.381 | 28.813 | 99.887 | 1.79 | 1.88 | 5.2e6 |
| `auon_temp` | 0.231 | 0.474 | 1.562 | 5.941 | 24.210 | 110.249 | 1.81 | 2.11 | 5.2e6 |
| `hybrid_auon_ns1` | 0.453 | 1.049 | 3.860 | 16.993 | 85.932 | 496.609 | 2.05 | 2.43 | 3.2e9 |
| `muon_ns1` | 0.283 | 0.690 | 3.152 | 13.003 | 63.040 | 468.893 | 2.14 | 2.59 | 3.2e9 |
| `muon` | 1.084 | 2.511 | 11.802 | 64.514 | 345.755 | 2053.453 | 2.23 | 2.50 | 1.6e10 |
| `hybrid_auon` (NS=5) | 1.054 | 2.548 | 12.069 | 68.234 | 322.384 | 1875.289 | 2.21 | 2.39 | 1.6e10 |

### Table 3 — the controlled brake sweep (64x64 dense gradient, one entry spiked)

`||U||_2` of the update each rule produces, and AuON's brake statistic `r`
(`inf` means `cosh` overflowed, which sends the update to exactly 0).

| spike | `auon_temp` `r` | `auon_temp` `\|\|U\|\|_2` | `auon` vanilla `\|\|U\|\|_2` | `hybrid_auon` `\|\|U\|\|_2` | `muon` `\|\|U\|\|_2` |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.369 | 0.04592 | 0.2465 | 0.032 | 1.187 |
| 1 | 5.365 | 0.04596 | 0.2466 | 0.03202 | 1.188 |
| 10 | 1.767e+06 | 1.473e-07 | 0.2602 | 0.02309 | 1.182 |
| 100 | inf | 0 | 0.8497 | 7.872e-06 | 1.202 |
| 1e6 | inf | 0 | 0.9998 | 0 | 0.6964 |

### Table 4 — deliberately large learning rate (10x the tuned value, 200 steps, seed 0)

| arm | lr | final loss | final accuracy | peak max attn logit | mean update RMS |
|---|---:|---:|---:|---:|---:|
| `adam` | 0.03 | 3.4791 | 0.0299 | 99421.70 | 6.84e-3 |
| `sgd_momentum` | 0.1 | 3.5305 | 0.0254 | 1591.07 | 8.44e-3 |
| `muon` | 0.05 | 0.0955 | 0.9743 | 5019.52 | 8.11e-3 |
| `auon` | 0.05 | 3.4775 | 0.0358 | 59.83 | 2.81e-3 |
| `auon_temp` | 0.2 | 3.6033 | 0.0312 | 111.20 | 1.72e-2 |
| `hybrid_auon` | 0.05 | 3.4466 | 0.0498 | 15.03 | 3.16e-3 |

### What these numbers mean, and what they do not

* **The task is solved by every arm at this scale, so accuracy is not the
  measurement.** Final accuracy is 1.0000 for five arms and 0.9998 for
  temperature-scaled AuON. The one real difference in Table 1 is *escape speed*:
  `muon`, `auon` and `hybrid_auon` are at exactly 1.0 at the first eval (step
  50) on all three seeds, `adam` is at 0.9954-0.9984 and `sgd_momentum` at
  0.0752-0.8473 there but above 0.9997 by step 100, while `auon_temp` sits at
  chance (0.0335, chance = 0.03125) until step 250 and only reaches
  0.82/0.07/0.20 at step 300, then 1.0/0.9997/0.9997 at step 400. Every seed
  shows the same delayed jump. That is a statement about one toy task and one
  learning-rate grid, not evidence that AuON is worse than Muon.
* **The attention logits are the paper's claim, and the ordering echoes it.**
  At the same final accuracy, `hybrid_auon` peaks at 14.0, vanilla `auon` at
  24.2 and `muon` at 147.0 (Adam, which the paper does not compare against
  here, peaks at 248.5 and SGD-momentum at 59.4). So the two AuON arms keep the
  largest logit 6x to 10x below Muon's on this task. It is a 2-layer model on a
  synthetic task with a single architecture and one seed set: this is consistent
  with the paper's stability claim, it is not a test of it. The logits also grow
  because all of these models end up *confident* on a deterministic task, so a
  high logit is not by itself instability.
* **The cost claim is about the exponent, and the exponent is ~2 versus ~2.5.**
  From `n = 1024` to `n = 4096` (16x the entries, 64x if the cost were cubic)
  AuON's step time grows by 13.5x and Muon's by 31.8x; at `n = 4096` one Muon
  step costs 20.6x one AuON step (2053 ms vs 99.9 ms). The fitted slopes over the
  largest three sizes are 1.88 and 2.11 for the two AuON variants (a cost linear
  in the number of entries predicts 2) against 2.39 and 2.50 for the two 5-step
  Newton-Schulz arms (a pure cubic cost predicts 3). Adam, whose update is pure
  elementwise work, gives 2.01 in the same fit, so 2 is the floor imposed by this
  machine; Muon lands between 2 and 3 because the larger matrix products run
  relatively closer to the CPU's peak rate than the smaller ones. That relative
  efficiency gain belongs to the hardware, not to the algorithms: the analytic
  MAC column, which is exactly cubic for Newton-Schulz, is there so the two
  effects can be told apart. The small-size slopes are lower for everything
  because per-step dispatch overhead dominates there.
* **The emergency brake is real, but only with the temperature scaling.** In
  Table 3, a single spike large enough to dominate the Frobenius norm takes
  temperature-scaled AuON's update from 0.046 to 1.5e-7 and then to 0, while
  Muon's stays at 1.18-1.20 for every spike size (it renormalises the gradient
  away), and vanilla `auon` moves the *other* way: 0.247 -> 0.9998, towards the
  trust-region boundary. The reason is the `gamma` inference in "What is in
  here": at `gamma = 1` the cosh argument cannot exceed 1. This is the clearest
  thing this repo measures, and it is a property of the temperature-scaled
  variant, which is the one the paper uses for its language-model runs.
* **The 10x learning-rate stress run does not settle the stability question.**
  Every arm stays finite, and the AuON arms do have by far the smallest peak
  logits (15 to 111 against 1591 to 99422), but the comparison is not
  like-for-like: Muon still solves the task at 10x its tuned rate (0.974
  accuracy) while the AuON arms do not, and because the tuned rates differ by up
  to 6.7x (0.003 for Adam, 0.02 for `auon_temp`) "10x" is a different absolute
  step size for each arm. The controlled evidence for the brake is Table 3, not this
  table.
* **Nothing here reproduces the paper's benchmarks.** The paper's headline
  numbers — 4.2882 versus 4.4349 final validation loss for Hybrid-AuON(NS=5)
  against Muon on FineWebEDU-10B, 74.42% against 65.99% CIFAR-10 test accuracy,
  the 13% runtime overhead on MNIST — come from models three orders of magnitude
  larger, real corpora and GPUs. This repo makes no claim about any of them.

## Where this may differ from the paper

* **The brake statistic and its exponent.** Section 3.3 eq. 5 defines
  `s = E[cosh^2(gamma * g_hat)]`, `r = s ** (3.5 / 2)` with `gamma = N ** 0.48`,
  while section 4.2 says "temperature-scaled variants use
  `r = (E[cosh^2(g_hat)]) ** beta, beta = 3.5`". Those differ by a factor of two
  in the exponent. `TEMPERATURE_SCALED` here uses eq. 5 (`brake_exponent =
  1.75`), and `brake_exponent` is a constructor argument, so `3.5` can be
  selected directly. The brake statistic is also monotone in the exponent, so
  this changes only how hard the brake bites.
* **`gamma` and the identity of the "emergency brake".** Sections 3.1 and 4.2
  give vanilla AuON `r = sqrt(mean(cosh^2(x)))` with no temperature, while
  section 3.5.1 uses `gamma = sqrt(N)` and sections 3.3/4.2 use `gamma =
  N ** 0.48`. I read these as the same family with the temperature as the knob,
  and the experiments above show that the exponential-suppression analysis needs
  the temperature. The paper does not state that vanilla AuON's brake is
  bounded; this repo does, and measures the bound.
* **The official implementation's numerator.** The reference implementation
  linked from the paper (`github.com/ryyzn9/AuON`, read for comparison, not
  copied) computes `update = G / (r_scaled + 1e-8)` where `G` is the
  *un-normalised* momentum and `r_scaled` is the cosh statistic of
  `gamma * G / ||G||`. That is not the paper's `U = X / (r + eps)` with
  `X = G / (||G||_F + 1e-7)`: it keeps the raw momentum magnitude, so its step
  is not bounded by a spectral trust region, and with the default `gamma = 1.5`
  its brake is a mild rescaling rather than the paper's exponential one. This
  repo follows the paper's equations. The reference also `continue`s on an
  exactly zero gradient; here a zero momentum simply produces a zero update,
  which keeps the arithmetic well defined.
* **Momentum convention.** The paper writes `m <- (1 - alpha) m + alpha g`
  (section 2.3) and Muon's usual decay coefficient. `AuON` here uses the decay
  form `m <- momentum * m + (1 - momentum) * g` with `momentum = 0.95`, i.e.
  `alpha = 0.05` in the paper's notation. The reference implementation blends the
  raw gradient with the buffer a second time (`grad.lerp_(m, momentum)`) before
  the rescaling; this repo uses the buffer itself, which is what "momentum before
  orthogonalisation" means in the paper's text.
* **Newton-Schulz coefficients.** `Muon` here uses the single standard trio
  `(3.4445, -4.7750, 2.0315)` for all five iterations, which is what MLX's
  built-in Muon and the paper's reference implementation use, and what the tests
  compare against. The per-iteration "quintic" and textbook coefficient sets are
  available (`NS_QUINTIC`, `NS_CLASSIC`) and are used in tests, but not in the
  experiment; with the standard trio, five iterations leave a noticeably
  non-unit singular spectrum (measured above), so "approximately semi-orthogonal"
  is doing real work in that sentence.
* **Weight decay and the parameter split.** Decoupled decay
  (`W <- W * (1 - lr * wd)`) is implemented for all arms; MLX's `Muon` couples it
  into the gradient instead, so the equivalence test runs with `wd = 0`. The
  experiment uses `weight_decay = 0` throughout for that reason. Embeddings, the
  LM head and 1-D parameters always go to Adam, as in the paper's section 4.2
  setup.
* **What is missing.** No FP8 kernels, no distributed or tensor-parallel path, no
  fused implementation, no learning-rate schedule (constant rate; the paper
  decays to 10% over the last 45% of its 5,000 iterations), no gradient clipping,
  no QK-clipping, no Muon(NS=3) arm, one architecture, one synthetic task, one
  learning-rate grid per arm, three seeds, CPU only. The paper's own stability
  experiment — a `d = 4096` transformer with all normalisation layers removed,
  run for 1,000 to 20,000 steps so that the logits either saturate or decay — has
  no counterpart here; the closest analogue is the synthetic spike sweep, which
  isolates the mechanism but says nothing about training at scale.
* **Timing caveat.** The wall-clock columns come from one CPU under varying load:
  within the same arm, per-seed `ms/step` ranges from 12.3 to 22.5 for `adam` and
  from 17.6 to 27.7 for `auon`, which is why the optimizer-only column and the
  fitted slopes are the numbers to read. The slopes and the per-step ordering of
  the arms are stable across the three seeds; a single `ms/step` number is not.

## Running it

From the repository root:

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/experiment.py
```

`python -m pytest` rather than the bare `pytest` script is what puts the
repository root on `sys.path`, so `auon` imports without an editable install;
`pip install -e ".[dev]"` works too. The tests set MLX's default device to the
CPU (`tests/conftest.py`) because the run has to be bit-reproducible and
`mx.linalg.svd` is CPU-only.

The committed `results/` were produced by exactly

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/experiment.py
```

which took 452.6 s and wrote `results/experiment_log.txt` (everything the script
printed) and `results/experiment.json` (the same numbers keyed for reuse:
`lr_sweep`, `tuned_learning_rates`, `runs[seed][arm]` with the curves, `table`
with the aggregates, and the `scaling`, `spike` and `brake` sections).
`--quick` shrinks the budget for a smoke run, `--parts` selects individual
sections, `--device gpu` switches MLX's default device (the scaling numbers and
the reproduced runs would change), and `--seeds`, `--steps`, `--repeat-len`,
`--vocab-size`, `--lr` grids and the geometric parameters are all flags;
`--help` lists them.
