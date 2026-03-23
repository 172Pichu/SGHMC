from __future__ import annotations

from typing import Dict

import arviz as az
import numpy as np

import torch
from torch import Tensor
from .utils.params import ParamSpec
from .objective import mlp_forward_from_theta

def _to_torch(x, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype)
    raise TypeError(f"Unsupported type: {type(x)}")

def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}")

def predict_from_theta_samples(theta_samples, X: Tensor, spec: ParamSpec, act_fn: str) -> Tensor:
    theta_samples = _to_torch(theta_samples)
    if theta_samples.ndim != 2:
        raise ValueError(f"theta_samples must be 2D, got shape {tuple(theta_samples.shape)}")
    preds = []
    for theta in theta_samples:
        pred = mlp_forward_from_theta(theta=theta, spec=spec, X=X, act_fn=act_fn)
        preds.append(pred)
    pred_stack = torch.stack(preds, dim=0)
    return pred_stack.mean(dim=0)

def rmse_from_predictions(pred: Tensor, target: Tensor) -> Tensor:
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    return torch.sqrt(torch.mean((pred - target) ** 2))

def gaussian_nll_mean(pred: Tensor, target: Tensor, noise_var: float) -> Tensor:
    if noise_var <= 0.0:
        raise ValueError(f"noise_var must be positive, got {noise_var}")
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    diff = target - pred
    c = torch.tensor(2.0 * torch.pi * noise_var, device=pred.device, dtype=pred.dtype)
    per_entry = 0.5 * torch.log(c) + 0.5 * diff.pow(2) / noise_var
    per_example = per_entry.sum(dim=-1)
    return per_example.mean()

def compute_param_norm_summary(theta_samples) -> Dict[str, float]:
    theta_samples = _to_torch(theta_samples)
    if theta_samples.ndim != 2:
        raise ValueError(f"theta_samples must be 2D, got shape {tuple(theta_samples.shape)}")
    norms = torch.norm(theta_samples, dim=1)
    return {"num_samples": float(theta_samples.shape[0]),
            "param_norm_min": norms.min().item(), "param_norm_max": norms.max().item(),
            "param_norm_mean": norms.mean().item(), "param_norm_std": norms.std(unbiased=False).item()}

def compute_L2D_summary(theta_samples) -> Dict[str, float]:
    theta_samples = _to_torch(theta_samples)
    if theta_samples.ndim != 2:
        raise ValueError(f"theta_samples must be 2D, got shape {tuple(theta_samples.shape)}")
    if theta_samples.shape[0] < 2:
        return {"L2D_mean": float("nan"), "L2D_std": float("nan"), "L2D_min": float("nan"), "L2D_max": float("nan")}
    diffs = theta_samples[1:] - theta_samples[:-1]
    L2Ds = torch.norm(diffs, dim=1)
    return {"L2D_min": L2Ds.min().item(), "L2D_max": L2Ds.max().item(),
            "L2D_mean": L2Ds.mean().item(), "L2D_std": L2Ds.std(unbiased=False).item()}

def evaluate_with_samples(samples, X: Tensor, y: Tensor,
                          spec: ParamSpec, act_fn: str, likelihood_var: float, y_scaler=None) -> Dict[str, float]:

    samples = _to_torch(samples)
    if samples.ndim != 2:
        raise ValueError(f"theta_samples must be 2D, got shape {tuple(samples.shape)}")
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    pred_mean = predict_from_theta_samples(samples, X, spec, act_fn)

    # Default: evaluate in the current scale
    y_eval = y
    pred_eval = pred_mean
    likelihood_var_eval = likelihood_var

    # If a target scaler is provided, evaluate in the original target scale
    if y_scaler is not None:

        pred_eval = y_scaler.unnormalize(pred_mean)
        y_eval = y_scaler.unnormalize(y)

        # Assuming standard z-score normalization:
        # y_norm = (y_raw - mean) / std
        # then var_raw = var_norm * std^2
        y_std = y_scaler.std
        if isinstance(y_std, torch.Tensor):
            y_std_scalar = float(y_std.reshape(-1)[0].item())
        else:
            y_std_scalar = float(y_std)

        likelihood_var_eval = likelihood_var * (y_std_scalar ** 2)

    rmse = rmse_from_predictions(pred_eval, y_eval).item()
    nll = gaussian_nll_mean(pred_eval, y_eval, likelihood_var_eval).item()

    out = {"rmse": rmse, "nll": nll}
    out.update(compute_param_norm_summary(samples))
    out.update(compute_L2D_summary(samples))
    return out

def make_online_evaluator(X, y, spec, act_fn: str, likelihood_var: float, y_scaler = None):
    def evaluator(all_chains, sample_idx):
        flat_samples = []
        for chain in all_chains:
            for sample in chain:
                flat_samples.append(sample)
        if len(flat_samples) == 0:
            return
        samples = torch.tensor(flat_samples, dtype=torch.float32)
        metrics = evaluate_with_samples(samples, X, y, spec, act_fn, likelihood_var, y_scaler)
        print(
            f"[online] Samples # {sample_idx:>3d}: "
            f"RMSE = {metrics['rmse']:>8.4f}  |  "
            f"NLL = {metrics['nll']:>8.4f}  |  "
            f"Param_Norm = {metrics['param_norm_mean']:8.4e}  |  "
            f"L2D = {metrics['L2D_mean']:8.4e}"
        )
    return evaluator

def diagnose_with_samples(chains) -> Dict[str, float]:

    chains_np = _to_numpy(chains).astype(np.float64, copy=False)
    if chains_np.ndim != 3:
        raise ValueError(f"chains must be 3D with shape (num_chains, num_draws, num_params), got {chains_np.shape}")
    
    num_chains, num_draws, num_params = chains_np.shape
    if num_chains < 2 or num_draws < 2:
        return {"num_chains": float(num_chains), "num_draws": float(num_draws), "num_params": float(num_params),
                "ess_min": float("nan"), "ess_max": float("nan"), "ess_mean": float("nan"), "ess_median": float("nan"),
                "r_hat_min": float("nan"), "r_hat_max": float("nan"), "r_hat_mean": float("nan"), "r_hat_median": float("nan")}

    idata = az.from_dict(posterior={"theta": chains_np})
    ess_ds = az.ess(idata, var_names=["theta"])
    rhat_ds = az.rhat(idata, var_names=["theta"])

    ess = np.asarray(ess_ds["theta"].values).reshape(-1)
    rhat = np.asarray(rhat_ds["theta"].values).reshape(-1)

    ess = ess[np.isfinite(ess)]
    rhat = rhat[np.isfinite(rhat)]

    if ess.size == 0:
        ess_min = ess_max = ess_mean = ess_median = float("nan")
    else:
        ess_min = float(np.min(ess))
        ess_max = float(np.max(ess))
        ess_mean = float(np.mean(ess))
        ess_median = float(np.median(ess))

    if rhat.size == 0:
        rhat_min = rhat_max = rhat_mean = rhat_median = float("nan")
    else:
        rhat_min = float(np.min(rhat))
        rhat_max = float(np.max(rhat))
        rhat_mean = float(np.mean(rhat))
        rhat_median = float(np.median(rhat))

    return {"num_chains": float(num_chains), "num_draws": float(num_draws), "num_params": float(num_params),
            "ess_min": ess_min, "ess_max": ess_max, "ess_mean": ess_mean, "ess_median": ess_median,
            "r_hat_min": rhat_min, "r_hat_max": rhat_max, "r_hat_mean": rhat_mean, "r_hat_median": rhat_median}