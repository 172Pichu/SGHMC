# Sampler design notes

This note is intended to accompany `adaptive_sghmc/samplers.py` and make the
commentary in the code easier to skim on GitHub.

## 1. Which paper and which formula?

The custom sampler in this repository is based on the scale-adapted SGHMC update
from Springenberg et al. (NIPS 2016), Eq. (10). The core design choice is the
`epsilon`-parameterized update with a diagonal preconditioner built from a
second-moment estimate `V_hat`.

This repository does **not** claim to implement every auxiliary variable from
that paper verbatim. Instead, it implements a practical diagonal version whose
main update is:

```text
x <- x + v
v <- v - epsilon^2 * prec * g - mdecay * v + momentum_noise
prec = 1 / sqrt(V_hat)
```

with adaptation of `V_hat` during burn-in and fixed preconditioning afterwards.

## 2. Why the Chen et al. (2014) paper also matters

The 2014 SGHMC paper is the source of the underlying SGHMC idea and of the toy
experiment intuition used throughout this repository.

In particular:

- Fig. 1 motivates the 1D double-well toy setup
- Fig. 1 uses artificial gradient noise with variance `4.0`
- Fig. 1 resamples momentum every `50` steps
- Fig. 3 motivates the correlated 2D Gaussian toy problem
- Eq. (15) gives the eta-parameterized interpretation connected to SGD with
  momentum

## 3. Why this repository introduces atomic / inner / outer steps

The SGHMC literature and modern software stacks often use overlapping but not
identical terms, for example:

- step
- integration step
- trajectory length
- orbit length
- momentum refresh interval

That makes cross-framework comparisons unnecessarily confusing, especially for
first-time researchers comparing custom code, Posteriors, BlackJAX, and
optimizer-style baselines.

This repository uses the following vocabulary:

- **atomic step**: one indivisible SGHMC update
- **inner step**: one atomic step inside an outer iteration
- **outer step**: one bookkeeping unit for momentum refresh, thinning, burn-in,
  logging, and sample collection

The goal is not to redefine SGHMC theory, but to make experiment configurations
more comparable and more transparent.

## 4. Parameter provenance summary

### `grad_estimator`
A user-supplied gradient callback for real data or custom targets. If it is not
provided, the sampler falls back to the historical toy targets:

- 1D -> Fig. 1 double-well gradient
- 2D -> Fig. 3 correlated Gaussian gradient

### `evaluater`
Online evaluation hook used only in benchmark experiments. It is not part of the
transition kernel.

### `burn_in`
Inspired by the 2016 paper's adapt-then-freeze workflow.

### `inner_steps`
Repository-level abstraction for reconciling different software notions of
trajectory length or integration steps.

### `reset_prob`
A repository-level engineering choice. Using a float instead of a boolean lets us
represent both deterministic refresh (`1.0`) and gentle stochastic refresh
(e.g. `0.05`).

### `grad_noise_var`
Mostly motivated by the 2014 toy experiments. Setting it to `4.0` reproduces the
canonical noisy-gradient setting from Fig. 1.

### `epsilon`
The step size in the epsilon-form update from the 2016 paper. It should not be
mixed up with the eta-form step size often used in later engineering code.

### `mdecay`
Practical replacement for hand-tuning the friction term `C` directly. In the
2016 paper, one practical recommendation is to fix `epsilon = 1e-2` and choose
`C` so that `epsilon * V_hat^{-1/2} * C = 0.05 I`.

### `warm_up_frac`
Repository-level warm-up gate that delays when the simplified `V_hat`
adaptation starts inside burn-in.

### `ema_beta`
Simplified EMA coefficient for second-moment tracking. Conceptually similar to
second-moment smoothing in adaptive optimization, but used here to support a
practical diagonal SGHMC preconditioner.
