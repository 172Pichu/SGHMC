import numpy as np
import jax, blackjax
from jax import numpy as jnp, random

def _to_py(x):
    x = jax.device_get(x)
    if hasattr(x, "shape") and x.shape == ():
        return float(x)
    arr = np.asarray(x)
    return arr.tolist()

def _inv_mass_to_diag(inv_mass):
    """Return a 1D diag representation (works for scalar / vector / matrix)."""
    inv_mass = jax.device_get(inv_mass)
    arr = np.asarray(inv_mass, dtype=np.float64)
    # scalar -> (1,)
    if arr.ndim == 0:
        return arr.reshape(1)
    # already diag
    if arr.ndim == 1:
        return arr
    # full matrix -> diag
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return np.diag(arr)
    raise ValueError(f"Unsupported inverse_mass_matrix shape: {arr.shape}")

def _aggregate_params(params_per_chain):
    """
    Build a stable 'aggregate' params summary across chains.
    - step_size: median across chains
    - inverse_mass_matrix_diag: mean across chains (diag form)
    """
    step_sizes = []
    inv_mass_diags = []
    for p in params_per_chain:
        if "step_size" in p:
            step_sizes.append(float(jax.device_get(p["step_size"])))
        if "inverse_mass_matrix" in p:
            inv_mass_diags.append(_inv_mass_to_diag(p["inverse_mass_matrix"]))
        elif "inverse_mass_matrix_diag" in p:
            inv_mass_diags.append(np.asarray(p["inverse_mass_matrix_diag"], dtype=np.float64))
    out = {}
    if step_sizes:
        out["step_size"] = float(np.median(np.asarray(step_sizes, dtype=np.float64)))
    if inv_mass_diags:
        diags = np.stack(inv_mass_diags, axis=0)  # (C, D)
        out["inverse_mass_matrix_diag"] = np.mean(diags, axis=0).tolist()
    return out

def _sample_nuts_single(rng_key, logprob_fn, initial_position, num_warmup: int, num_samples: int):
    
    window_adaptation = blackjax.window_adaptation
    rng_key, warmup_key, sample_key = random.split(rng_key, 3)

    warmup = window_adaptation(blackjax.nuts, logprob_fn)
    (warmup_state, parameters), _ = warmup.run(warmup_key, initial_position, num_warmup)

    kernel = blackjax.nuts(logprob_fn, **parameters)
    state = kernel.init(warmup_state.position)

    def step(state, key):
        state, _ = kernel.step(key, state)
        return state, state.position

    keys = random.split(sample_key, num_samples)
    _, samples = jax.lax.scan(step, state, keys)

    params_kept = {}
    if "step_size" in parameters:
        params_kept["step_size"] = parameters["step_size"]
    if "inverse_mass_matrix" in parameters:
        params_kept["inverse_mass_matrix"] = parameters["inverse_mass_matrix"]
    if "inverse_mass_matrix" in params_kept:
        params_kept["inverse_mass_matrix_diag"] = _inv_mass_to_diag(params_kept["inverse_mass_matrix"])

    params_serializable = {k: _to_py(v) for k, v in params_kept.items()}
    return samples, params_serializable

def sample_nuts(rng_key, logprob_fn, initial_positions, num_warmup: int, num_samples: int, num_chains: int = 1):
    
    assert num_chains >= 1
    assert initial_positions.shape[0] == num_chains

    if num_chains == 1:
        return _sample_nuts_single(rng_key, logprob_fn, initial_positions[0], num_warmup, num_samples)

    chain_keys = random.split(rng_key, num_chains)
    chain_samples = []
    params_per_chain = []

    for chain_idx in range(num_chains):
        print(f"nuts chain {chain_idx}")
        s, p = _sample_nuts_single(chain_keys[chain_idx], logprob_fn, initial_positions[chain_idx], num_warmup, num_samples)
        chain_samples.append(s)
        params_per_chain.append(p)

    params_pack = {"per_chain": params_per_chain, "aggregate": _aggregate_params(params_per_chain)}
    return jnp.stack(chain_samples), params_pack