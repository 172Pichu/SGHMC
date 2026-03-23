from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ..utils.params import ParamSpec
from ..third_party.adaptive_sghmc import AdaptiveSGHMC
from ..mcmc import evaluate_with_samples, diagnose_with_samples
from ..objective import mlp_forward_from_theta

def _make_minibatch_indices(n_train: int, batch_size: int, gen: torch.Generator) -> Tensor:
    if batch_size >= n_train:
        return torch.arange(n_train)
    return torch.randint(0, n_train, (batch_size,), generator=gen)

def _negative_log_posterior(theta: Tensor, X: Tensor, y: Tensor, spec: ParamSpec, act_fn: str,
                            prior_var: float, likelihood_var: float, num_train: int) -> Tensor:
    pred = mlp_forward_from_theta(theta=theta, spec=spec, X=X, act_fn=act_fn)
    diff = y - pred
    # data term (mini-batch scaled)
    batch_nll = 0.5 * (torch.log(torch.tensor(2.0 * torch.pi * likelihood_var, device=pred.device, dtype=pred.dtype))
                       + diff.pow(2) / likelihood_var).sum()
    scaled_nll = batch_nll * (num_train / X.shape[0])
    prior = 0.5 * (torch.log(torch.tensor(2.0 * torch.pi * prior_var, device=theta.device, dtype=theta.dtype))
                   + theta.pow(2) / prior_var).sum()
    return scaled_nll + prior

def run_optbnn(X_train: Tensor, y_train: Tensor, X_val: Tensor, y_val: Tensor, X_test: Tensor, y_test: Tensor,
               spec: ParamSpec, act_fn: str, prior_var: float, likelihood_var: float,
               seed: int, num_chains: int, num_samples: int, burn_in: int, keep_every: int,
               batch_size: int, lr: float, mdecay: float, num_burn_in_steps: Optional[int] = None, *,
               y_scaler=None) -> Dict[str, object]:
    """
    Run the GitHub-style AdaptiveSGHMC implementation in optimizer form.

    == Notes ==
    - num_samples means saved draws per chain
    - burn_in here means number of optimizer steps before saving starts
    - keep_every means thinning interval in optimizer steps
    """
    if num_burn_in_steps is None:
        num_burn_in_steps = burn_in

    device = X_train.device
    dtype = torch.float32
    num_train = X_train.shape[0]

    total_steps = burn_in + keep_every * num_samples

    gens = [torch.Generator(device=device).manual_seed(seed + c) for c in range(num_chains)]
    theta_list: List[Tensor] = []
    opt_list: List[AdaptiveSGHMC] = []

    for chain_idx in range(num_chains):
        theta = torch.randn(spec.total_numel, generator=gens[chain_idx], device=device, dtype=dtype, requires_grad=True)
        opt = AdaptiveSGHMC([theta], lr=lr, mdecay=mdecay, num_burn_in_steps=num_burn_in_steps, scale_grad=num_train)
        theta_list.append(theta)
        opt_list.append(opt)

    all_chains = [[] for _ in range(num_chains)]
    all_samples = []

    for step_idx in range(total_steps):

        for chain_idx in range(num_chains):

            idx = _make_minibatch_indices(num_train, batch_size, gens[chain_idx]).to(device)
            X_epoch = X_train[idx]
            y_epoch = y_train[idx]

            opt_list[chain_idx].zero_grad()

            loss = _negative_log_posterior(theta=theta_list[chain_idx], X=X_epoch, y=y_epoch, spec=spec, act_fn=act_fn,
                                           prior_var=prior_var, likelihood_var=likelihood_var, num_train=num_train)
            
            loss.backward()
            opt_list[chain_idx].step()

        # save after burn-in
        if step_idx >= burn_in and ((step_idx - burn_in) % keep_every == 0):
            for chain_idx in range(num_chains):
                sample = theta_list[chain_idx].detach().cpu().numpy().astype(np.float64).copy()
                all_chains[chain_idx].append(sample)
                all_samples.append(sample)

    chains = np.asarray(all_chains, dtype=np.float64)   # (C, S, D)
    samples = np.asarray(all_samples, dtype=np.float64) # (C * S, D)

    val_metrics = evaluate_with_samples(samples=samples, X=X_val, y=y_val, spec=spec, act_fn=act_fn,
                                        likelihood_var=likelihood_var, y_scaler=y_scaler)

    test_metrics = evaluate_with_samples(samples=samples, X=X_test, y=y_test, spec=spec, act_fn=act_fn,
                                         likelihood_var=likelihood_var, y_scaler=y_scaler)
    
    diagnostics = diagnose_with_samples(chains)

    result = {
        "method": "optbnn_sghmc",
        "grad_mode": f"minibatch[{batch_size}]",
        "model_hyper_params": {
            "num_hidden_units": None,
            "num_hidden_layers": None,
            "act_fn": act_fn,
        },
        "sampler_hyper_params": {
            "seed": seed,
            "num_dims": spec.total_numel,
            "num_chains": num_chains,
            "num_samples": num_samples,
            "burn_in": burn_in,
            "keep_every": keep_every,
            "batch_size": batch_size,
            "lr": lr,
            "mdecay": mdecay,
            "prior_var": prior_var,
            "likelihood_var": likelihood_var,
            "num_burn_in_steps": num_burn_in_steps,
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