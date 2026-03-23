from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import jax
from jax import numpy as jnp, random

from ..utils.params import ParamSpec
from ..mcmc import evaluate_with_samples, diagnose_with_samples
from ..nuts import sample_nuts

def _activation_jax(name: str):
    name = name.lower()
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    if name == "leaky_relu":
        return lambda x: jax.nn.leaky_relu(x, negative_slope=0.01)
    if name == "sigmoid":
        return jax.nn.sigmoid
    raise ValueError(f"Unsupported activation: {name}")

def _unflatten_theta_jax(theta: jnp.ndarray, spec: ParamSpec) -> Dict[str, jnp.ndarray]:
    params = {}
    start = 0
    for name in spec.names:
        shape = spec.shapes[name]
        size = int(np.prod(shape))
        params[name] = theta[start:start + size].reshape(shape)
        start += size
    return params

def _mlp_forward_jax(theta: jnp.ndarray, X: jnp.ndarray, spec: ParamSpec, act_fn: str) -> jnp.ndarray:
    params = _unflatten_theta_jax(theta, spec)
    act = _activation_jax(act_fn)
    num_layers = len(spec.names) // 2
    h = X
    for i in range(1, num_layers + 1):
        W = params[f"W{i}"]
        b = params[f"b{i}"]
        h = h @ W + b
        if i < num_layers:
            h = act(h)
    return h

def _make_logprob_fn(X_train: torch.Tensor, y_train: torch.Tensor,
                     spec: ParamSpec, act_fn: str, prior_var: float, likelihood_var: float):
    
    X_np = X_train.detach().cpu().numpy().astype(np.float32)
    y_np = y_train.detach().cpu().numpy().astype(np.float32)

    X_j = jnp.asarray(X_np)
    y_j = jnp.asarray(y_np)

    def logprob_fn(theta: jnp.ndarray) -> jnp.ndarray:

        pred = _mlp_forward_jax(theta, X_j, spec, act_fn)
        diff = y_j - pred

        # log likelihood up to additive constants is also fine for NUTS, but we keep full Gaussian form for clarity.
        llike = -0.5 * (jnp.log(2.0 * jnp.pi * likelihood_var) + (diff ** 2) / likelihood_var).sum()
        lpri = -0.5 * (jnp.log(2.0 * jnp.pi * prior_var) + (theta ** 2) / prior_var).sum()

        return llike + lpri

    return logprob_fn

def _make_initial_positions(*, num_chains: int, num_dims: int, seed: int, init_scale: float = 1.0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    init = rng.normal(loc=0.0, scale=init_scale, size=(num_chains, num_dims)).astype(np.float32)
    return jnp.asarray(init)

def run_nuts(X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor,
             y_val: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor,
             spec: ParamSpec, act_fn: str, prior_var: float, likelihood_var: float,
             seed: int, num_chains: int, num_samples: int, burn_in: int, *, y_scaler=None) -> Dict[str, object]:
    """
    Run BlackJAX NUTS and return the same result structure as your eps runner.

    == Notes ==
    - num_samples here means saved samples per chain.
    - burn_in is mapped to NUTS warmup.
    """
    logprob_fn = _make_logprob_fn(X_train=X_train, y_train=y_train,
                                  spec=spec, act_fn=act_fn, prior_var=prior_var, likelihood_var=likelihood_var)

    initial_positions = _make_initial_positions(num_chains=num_chains, num_dims=spec.total_numel, seed=seed, init_scale=1.0)
    rng_key = random.PRNGKey(seed)

    nuts_samples, nuts_params = sample_nuts(rng_key=rng_key, logprob_fn=logprob_fn, initial_positions=initial_positions,
                                            num_warmup=burn_in, num_samples=num_samples, num_chains=num_chains)

    # shape -> (C, S, D)
    chains = np.asarray(jax.device_get(nuts_samples), dtype=np.float64)
    if chains.ndim == 2:
        chains = chains[None, :, :]

    samples = chains.reshape(-1, chains.shape[-1])

    diagnostics = diagnose_with_samples(chains)

    val_metrics = evaluate_with_samples(samples=samples, X=X_val, y=y_val, spec=spec, act_fn=act_fn,
                                        likelihood_var=likelihood_var, y_scaler=y_scaler)

    test_metrics = evaluate_with_samples(samples=samples, X=X_test, y=y_test, spec=spec, act_fn=act_fn,
                                         likelihood_var=likelihood_var, y_scaler=y_scaler)

    result = {
        "method": "nuts",
        "grad_mode": "fullbatch",
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
            "num_warmup": burn_in,
            "prior_var": prior_var,
            "likelihood_var": likelihood_var,
            "nuts_params": nuts_params,
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