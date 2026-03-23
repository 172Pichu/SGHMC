from __future__ import annotations

from typing import Dict

from .utils.data import load_predefined_yacht_fold, normalize_split
from .utils.params import build_mlp_param_spec

def run_yacht_fold(method: str, root: str, fold_id: int, act_fn: str, num_hidden_layers: int, num_hidden_units: int,
                   prior_var: float, likelihood_var: float, seed: int, num_chains: int, num_samples: int, num_discarded: int,
                   burn_in: int, keep_every: int, print_every: int, inner_steps: int, reset_prob: float,
                   grad_noise_var: float, epsilon: float, mdecay: float, *,
                   warm_up_frac: float = 0.5, ema_beta: float = 0.999, batch_size: int | None = None) -> Dict[str, object]:
    """Run one predefined Yacht fold with the requested sampling method."""
    split = load_predefined_yacht_fold(root=root, fold_id=fold_id)
    split_std, X_scaler, y_scaler = normalize_split(split)

    spec = build_mlp_param_spec(in_dim=split_std.X_train.shape[1], out_dim=1,
                                num_hidden_layers=num_hidden_layers, num_hidden_units=num_hidden_units)

    if method == "eps":
        from .runners.run_eps import run_eps

        result = run_eps(split_std.X_train, split_std.y_train, split_std.X_val,
                         split_std.y_val, split_std.X_test, split_std.y_test,
                         spec, act_fn, prior_var, likelihood_var, seed, num_chains, num_samples, num_discarded,
                         burn_in, keep_every, print_every, inner_steps, reset_prob, grad_noise_var, epsilon, mdecay, y_scaler,
                         warm_up_frac=warm_up_frac, ema_beta=ema_beta, batch_size=batch_size)

    elif method == "nuts":
        from .runners.run_nuts import run_nuts

        result = run_nuts(X_train=split_std.X_train, y_train=split_std.y_train, X_val=split_std.X_val,
                          y_val=split_std.y_val, X_test=split_std.X_test, y_test=split_std.y_test,
                          spec=spec, act_fn=act_fn, prior_var=prior_var, likelihood_var=likelihood_var,
                          seed=seed, num_chains=num_chains, num_samples=num_samples, burn_in=burn_in, y_scaler=y_scaler)

    elif method == "optbnn":
        from .runners.run_optbnn import run_optbnn

        result = run_optbnn(X_train=split_std.X_train, y_train=split_std.y_train, X_val=split_std.X_val,
                            y_val=split_std.y_val, X_test=split_std.X_test, y_test=split_std.y_test,
                            spec=spec, act_fn=act_fn, prior_var=prior_var, likelihood_var=likelihood_var,
                            seed=seed, num_chains=num_chains, num_samples=num_samples,
                            burn_in=burn_in, keep_every=keep_every, batch_size=batch_size,
                            lr=epsilon, mdecay=mdecay, y_scaler=y_scaler)

    else:
        raise ValueError(f"Unknown method: {method}")

    result["dataset"] = "yacht"
    result["fold_id"] = fold_id
    result["model_hyper_params"] = {
        "num_hidden_units": num_hidden_units,
        "num_hidden_layers": num_hidden_layers,
        "act_fn": act_fn
    }
    result["split_sizes"] = {
        "train": int(split_std.X_train.shape[0]),
        "val": int(split_std.X_val.shape[0]),
        "test": int(split_std.X_test.shape[0]),
    }
    result["X_scaler"] = X_scaler
    result["y_scaler"] = y_scaler
    return result