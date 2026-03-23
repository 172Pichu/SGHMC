from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

from ..utils.params import ParamSpec
from ..mcmc import evaluate_with_samples, make_online_evaluator, diagnose_with_samples
from ..objective import make_fullbatch_grad_estimator, make_minibatch_grad_estimator
from ..samplers import samples_eps

def run_eps(X_train: Tensor, y_train: Tensor, X_val: Tensor, y_val: Tensor, X_test: Tensor, y_test: Tensor,
            spec: ParamSpec, act_fn: str, prior_var: float, likelihood_var: float,
            seed: int, num_chains: int, num_samples: int, num_discarded: int,
            burn_in: int, keep_every: int, print_every: int, inner_steps: int, reset_prob: float,
            grad_noise_var: float, epsilon: float, mdecay: float, y_scaler, *,
            warm_up_frac: float = 0.5, ema_beta: float = 0.999, batch_size: Optional[int] = None) -> Dict[str, object]:
    """
    Run one eps-SGHMC experiment and evaluate on validation/test data.
    """
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(-1)
    if y_val.ndim == 1:
        y_val = y_val.unsqueeze(-1)
    if y_test.ndim == 1:
        y_test = y_test.unsqueeze(-1)

    if batch_size is None:
        grad_mode = "fullbatch"
        grad_est = make_fullbatch_grad_estimator(X_train, y_train, spec, act_fn, prior_var, likelihood_var)
    else:
        grad_mode = f"minibatch[{batch_size}]"
        grad_est = make_minibatch_grad_estimator(X_train, y_train, spec, act_fn, prior_var, likelihood_var, seed, batch_size) 

    evaluator = make_online_evaluator(X_val, y_val, spec, act_fn, likelihood_var, y_scaler)

    chains, samples = samples_eps(grad_est, evaluator, seed, spec.total_numel, num_chains, num_samples, num_discarded,
                                  burn_in, keep_every, print_every, inner_steps, reset_prob, grad_noise_var, epsilon, mdecay,
                                  warm_up_frac=warm_up_frac, ema_beta=ema_beta)

    val_metrics = evaluate_with_samples(samples, X_val, y_val, spec, act_fn, likelihood_var, y_scaler=y_scaler)
    test_metrics = evaluate_with_samples(samples, X_test, y_test, spec, act_fn, likelihood_var, y_scaler=y_scaler)
    diagnostics = diagnose_with_samples(chains)

    result = {
        "method": "eps",
        "grad_mode": grad_mode,
        "sampler_hyper_params": {
            "seed": seed,
            "num_dims": spec.total_numel,
            "num_chains": num_chains,
            "num_samples": num_samples,
            "num_discarded": num_discarded,
            "burn_in": burn_in,
            "keep_every": keep_every,
            "print_every": print_every,
            "inner_steps": inner_steps,
            "reset_prob": reset_prob,
            "grad_noise_var": grad_noise_var,
            "epsilon": epsilon,
            "mdecay": mdecay,
            "warm_up_frac": warm_up_frac,
            "ema_beta": ema_beta,
            "batch_size": batch_size,
        },
        "raw_chains_shape": tuple(chains.shape),
        "raw_samples_shape": tuple(samples.shape),
        "num_saved_total": int(samples.shape[0]),
        "chains": chains,
        "samples": samples,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "diagnostics": diagnostics,
    }
    
    return result