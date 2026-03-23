from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor

from .distributions import dU_dx_fig1, grad_U_fig3_xy

GradEstimator = Callable[[Tensor], Tensor]
Evaluator = Optional[Callable[[list[list[list[float]]], int], None]]

def _default_grad_estimator(num_dims: int) -> GradEstimator:
    """Return the default toy gradient used when no gradient callback is given.

    Historical note
    ---------------
    The sampler was originally written for toy SGHMC experiments inspired by
    Fig. 1 and Fig. 3 of Chen et al. (2014). To preserve that workflow, the
    sampler falls back to:

    - ``dU_dx_fig1`` when ``num_dims == 1``
    - ``grad_U_fig3_xy`` when ``num_dims == 2``

    For real datasets such as Yacht, callers should provide an explicit
    gradient estimator.
    """
    if num_dims == 1:
        return dU_dx_fig1
    if num_dims == 2:
        return grad_U_fig3_xy
    raise ValueError("grad_estimator must be provided when num_dims is not 1 (Fig. 1) or 2 (Fig. 3).")

def samples_eps(grad_estimator: Optional[GradEstimator], evaluater: Evaluator,
                seed: int, num_dims: int, num_chains: int, num_samples: int, num_discarded: int,
                burn_in: int, keep_every: int, print_every: int, inner_steps: int, reset_prob: float,
                grad_noise_var: float, epsilon: float, mdecay: float, *,
                warm_up_frac: float = 0.5, ema_beta: float = 0.999) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Sample from a target distribution using an epsilon-parameterized,
    scale-adapted SGHMC-style sampler.

    Paper provenance
    ----------------
    This implementation is based on the scale-adapted SGHMC update proposed in

        Springenberg et al. (NIPS 2016),
        "Bayesian Optimization with Robust Bayesian Neural Networks",
        Eq. (10)

    The paper introduces the variable substitution

        v = epsilon * M^{-1} r = epsilon * V_hat^{-1/2} r

    and obtains the dynamics

        Delta theta = v
        Delta v = - epsilon^2 V_hat^{-1/2} grad U_tilde(theta)
                  - epsilon V_hat^{-1/2} C v
                  + N(0, 2 epsilon^3 V_hat^{-1/2} C V_hat^{-1/2} - epsilon^4 I)

    after estimating the diagonal preconditioner during burn-in.

    This code implements a practical diagonal realization of that update using

        prec = 1 / sqrt(V_hat)
        x <- x + v
        v <- v - epsilon^2 * prec * g - mdecay * v + xi
        xi ~ N(0, 2 * epsilon^2 * mdecay * prec - epsilon^4)

    Therefore, the sampler should be described as a *paper-inspired practical
    implementation of Eq. (10)*, not as a literal transcription of every
    auxiliary state variable in the original derivation.

    Relation to Chen et al. (2014)
    ------------------------------
    The broader SGHMC idea comes from Chen, Fox, and Guestrin (ICML 2014):

    - stochastic gradients require friction to preserve a well-behaved sampler,
    - artificial gradient noise is useful in toy experiments,
    - the eta-parameterized interpretation in their Eq. (15) helps explain how
      SGHMC relates to SGD with momentum.

    In particular, their Fig. 1 uses noisy gradients of the form

        grad U_tilde(theta) = grad U(theta) + N(0, 4)

    and resamples momentum every 50 steps.

    Directly paper-inspired vs repository-specific design
    -----------------------------------------------------
    Directly paper-inspired:
    - diagonal scale adaptation through ``V_hat``
    - burn-in adaptation followed by fixed-scale sampling
    - friction plus momentum noise in the velocity update
    - optional artificial gradient noise for toy experiments

    Repository-specific engineering choices:
    - the explicit terminology "atomic step", "inner step", and "outer step"
    - ``inner_steps`` and probabilistic ``reset_prob`` to standardize
      comparisons across different SGHMC implementations
    - ``warm_up_frac`` as a delayed adaptation gate inside burn-in
    - ``ema_beta`` as a simplified EMA control instead of the full
      tau-and-g dynamics from Eq. (9) of the 2016 paper

    Atomic / inner / outer step terminology
    ---------------------------------------
    One goal of this repository is to make SGHMC configurations comparable
    across implementations that expose different concepts such as integration
    steps, trajectory length, or mandatory momentum refresh.

    - atomic step:
        one indivisible SGHMC state update
    - inner step:
        one atomic update inside the current outer iteration
    - outer step:
        the wrapper iteration used for momentum refresh, burn-in accounting,
        thinning, logging, and sample collection

    This terminology is especially useful because some implementations always
    refresh momentum every trajectory while others do not expose refresh as a
    user-level option at all.

    Pseudocode
    ----------
    Burn-in phase:

        initialize x, v, V_hat
        for outer_step in 1 .. burn_in:
            optionally refresh momentum with probability reset_prob
            repeat inner_steps times:
                g_det <- grad_fn(x)
                g <- g_det + optional gradient noise
                if adaptation has started:
                    V_hat <- ema_beta * V_hat + (1 - ema_beta) * g_det^2
                prec <- 1 / sqrt(V_hat)
                xi <- Normal(0, 2 * epsilon^2 * mdecay * prec - epsilon^4)
                x <- x + v
                v <- v - epsilon^2 * prec * g - mdecay * v + xi

    Sampling phase:

        freeze V_hat and all derived preconditioners
        while not enough samples have been collected:
            optionally refresh momentum with probability reset_prob
            repeat inner_steps times:
                g_det <- grad_fn(x)
                g <- g_det + optional gradient noise
                x <- x + v
                v <- v - epsilon^2 * prec * g - mdecay * v + xi
            every keep_every outer steps:
                save one sample per chain
                optionally call evaluater(all_chains, sample_idx)

    Parameter notes
    ---------------
    grad_estimator:
        Callable returning the gradient of the target potential. Historically
        this sampler defaulted to toy targets; in real-data experiments the
        caller should pass a dataset- and model-specific gradient estimator.

    evaluater:
        Optional callback for online evaluation or diagnostics. It is not part
        of the SGHMC transition kernel itself.

    seed:
        Base random seed. Chain k uses ``seed + k``.

    num_dims:
        Number of flattened parameters / coordinates in one chain state.

    num_chains:
        Number of parallel chains.

    num_samples:
        Number of kept samples requested per chain before the
        post-collection discard filter is applied.

    num_discarded:
        Number of early kept samples discarded from each chain after burn-in.
        This is separate from burn-in itself.

    burn_in:
        Number of outer steps used for adaptation and early mixing. Inspired by
        the 2016 paper's strategy of adapting scale during burn-in and keeping
        the parameters fixed afterwards.

    keep_every:
        Thinning interval measured in outer steps after burn-in.

    print_every:
        Evaluation interval measured in saved samples per chain.

    inner_steps:
        Number of atomic SGHMC updates per outer step.

        Practical guidance:
        - use ``inner_steps=50`` together with deterministic refresh for toy
          settings meant to mimic the 2014 Fig. 1 momentum-resampling schedule
        - values around 10 are common in modern SGHMC/NUTS-style examples and
          make outer-step accounting shorter and easier to inspect

    reset_prob:
        Probability of refreshing momentum at the beginning of each outer step.

        Why this is a float instead of a bool:
        - ``reset_prob = 1.0`` reproduces deterministic refresh every outer step
        - small values such as ``0.05`` can act as a gentle partial refresh and
          often improve mixing in neural-network experiments

    grad_noise_var:
        Variance of artificial Gaussian noise added directly to the gradient:

            g = g_det + Normal(0, grad_noise_var I)

        This is separate from the momentum noise term required by SGHMC.
        For toy experiments, ``grad_noise_var = 4.0`` matches the canonical
        noisy-gradient setup from Chen et al. (2014) Fig. 1.

    epsilon:
        Step size in the epsilon-parameterized update used here. This repository
        intentionally distinguishes it from the eta-parameterized form often
        used in later engineering implementations and in Chen et al. (2014)
        Eq. (15).

    mdecay:
        Effective momentum-decay coefficient in this implementation.

        Design origin:
        the 2016 paper recommends fixing ``epsilon = 1e-2`` and choosing ``C``
        so that ``epsilon * V_hat^{-1/2} * C = 0.05 I``. This repository folds
        that practical decay target into the scalar hyperparameter ``mdecay``.

    warm_up_frac:
        Fraction of burn-in after which second-moment adaptation begins. This is
        a repository-level simplification that delays adaptation until the very
        earliest transient behavior has passed.

    ema_beta:
        Exponential moving-average coefficient used in the simplified V_hat
        update. Conceptually, it is the smoothing coefficient of the diagonal
        second-moment estimate.

    Returns
    -------
    all_chains:
        Array with shape ``(num_chains, kept_samples_after_discard, num_dims)``.

    all_samples:
        Flattened view of all retained samples with shape
        ``(num_chains * kept_samples_after_discard, num_dims)``.
    """

    if num_chains <= 0:
        raise ValueError(f"num_chains must be positive, got {num_chains}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if inner_steps <= 0:
        raise ValueError(f"inner_steps must be positive, got {inner_steps}")
    if keep_every <= 0:
        raise ValueError(f"keep_every must be positive, got {keep_every}")
    if not (0.0 <= reset_prob):
        raise ValueError(f"reset_prob must be non-negative, got {reset_prob}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if mdecay < 0.0:
        raise ValueError(f"mdecay must be non-negative, got {mdecay}")

    grad_fn = grad_estimator if grad_estimator is not None else _default_grad_estimator(num_dims)

    grad_noise_std = math.sqrt(max(0.0, float(grad_noise_var)))
    eps = float(epsilon)
    eps_2 = eps * eps
    eps_4 = eps_2 * eps_2
    adapt_start = int(burn_in * warm_up_frac)
    ema = float(ema_beta)
    one_minus_ema = 1.0 - ema

    gens = [torch.Generator().manual_seed(int(seed + chain_idx)) for chain_idx in range(num_chains)]
    x = [torch.randn(num_dims, generator=gens[chain_idx], dtype=torch.float32) for chain_idx in range(num_chains)]
    v = [torch.zeros_like(x[chain_idx]) for chain_idx in range(num_chains)]
    v_hats = [torch.ones_like(x[chain_idx]) for chain_idx in range(num_chains)]

    atomic_steps = 0
    outer_steps = 0

    # Burn-in phase: adapt the diagonal second-moment estimate V_hat.
    for _ in range(burn_in):

        outer_steps += 1

        if reset_prob > 0.0:
            for chain_idx in range(num_chains):
                # Momentum refresh is modeled as a Bernoulli trial at the outer-step level.
                if reset_prob >= 1.0 or torch.rand((), generator=gens[chain_idx]).item() < reset_prob:
                    prec = 1.0 / (torch.sqrt(v_hats[chain_idx]) + 1e-12)
                    v[chain_idx] = eps * torch.sqrt(prec) * torch.randn(
                        num_dims, generator=gens[chain_idx], dtype=x[chain_idx].dtype
                    )

        for _ in range(inner_steps):

            atomic_steps += 1

            for chain_idx in range(num_chains):
                g_det = grad_fn(x[chain_idx])
                g = g_det if grad_noise_std == 0.0 else g_det + grad_noise_std * torch.randn(
                    num_dims, generator=gens[chain_idx], dtype=x[chain_idx].dtype
                )

                if outer_steps >= adapt_start:
                    # Simplified EMA-based adaptation of V_hat. The original 2016 paper
                    # also adapts the averaging window tau and a smoothed gradient g_theta.
                    v_hats[chain_idx] = ema * v_hats[chain_idx] + one_minus_ema * (g_det * g_det)

                # Diagonal preconditioner corresponding to V_hat^{-1/2}.
                prec = 1.0 / (torch.sqrt(v_hats[chain_idx]) + 1e-12)

                # Momentum-noise variance in the practical diagonal Eq. (10) realization.
                mom_noise_var = torch.clamp(2.0 * eps_2 * mdecay * prec - eps_4, min=0.0)
                mom_noise = torch.sqrt(mom_noise_var) * torch.randn(
                    num_dims, generator=gens[chain_idx], dtype=x[chain_idx].dtype
                )

                # One atomic SGHMC update.
                x[chain_idx] = x[chain_idx] + v[chain_idx]
                v[chain_idx] = v[chain_idx] - eps_2 * prec * g - mdecay * v[chain_idx] + mom_noise

    # Freeze adaptation after burn-in, exactly as intended in the 2016 scale-adapted workflow.
    precs = []
    mom_noise_stds = []
    for chain_idx in range(num_chains):
        prec = 1.0 / (torch.sqrt(v_hats[chain_idx]) + 1e-12)
        precs.append(prec)
        mom_noise_var = torch.clamp(2.0 * eps_2 * mdecay * prec - eps_4, min=0.0)
        mom_noise_stds.append(torch.sqrt(mom_noise_var))

    num_collected = [0] * num_chains
    all_chains: list[list[list[float]]] = [[] for _ in range(num_chains)]
    all_samples: list[list[float]] = []

    while min(num_collected) < num_samples:

        outer_steps += 1

        if reset_prob > 0.0:
            for chain_idx in range(num_chains):
                if reset_prob >= 1.0 or torch.rand((), generator=gens[chain_idx]).item() < reset_prob:
                    v[chain_idx] = eps * torch.sqrt(precs[chain_idx]) * torch.randn(
                        num_dims, generator=gens[chain_idx], dtype=x[chain_idx].dtype
                    )

        for _ in range(inner_steps):

            atomic_steps += 1

            for chain_idx in range(num_chains):
                g_det = grad_fn(x[chain_idx])
                g = g_det if grad_noise_std == 0.0 else g_det + grad_noise_std * torch.randn(
                    num_dims, generator=gens[chain_idx], dtype=x[chain_idx].dtype
                )

                mom_noise = mom_noise_stds[chain_idx] * torch.randn(
                    num_dims, generator=gens[chain_idx], dtype=x[chain_idx].dtype
                )

                x[chain_idx] = x[chain_idx] + v[chain_idx]
                v[chain_idx] = v[chain_idx] - eps_2 * precs[chain_idx] * g - mdecay * v[chain_idx] + mom_noise

        if (outer_steps - burn_in) % keep_every == 0:
            sample_idx = None
            for chain_idx in range(num_chains):
                num_collected[chain_idx] += 1
                if num_collected[chain_idx] > num_discarded:
                    sample = x[chain_idx].detach().cpu().tolist()
                    all_chains[chain_idx].append(sample)
                    all_samples.append(sample)
                    sample_idx = num_collected[chain_idx] - num_discarded

            if evaluater is not None and print_every > 0 and sample_idx is not None and sample_idx % print_every == 0:
                evaluater(all_chains, sample_idx)

    print(f"atomic_steps = {atomic_steps}, outer_steps = {outer_steps}")
    return np.asarray(all_chains, dtype=np.float64), np.asarray(all_samples, dtype=np.float64)