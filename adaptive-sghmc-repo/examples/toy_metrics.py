from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math
from dataclasses import dataclass
from typing import Any, List, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import arviz as az

try:
    import torch
    from torch import Tensor
except Exception:
    torch = None
    Tensor = None

try:
    import jax
    from jax import numpy as jnp
    JaxArray = jnp.ndarray
except Exception:
    jax = None
    jnp = None
    JaxArray = None

ArrayLike = Any
ParamDict = Dict[str, Any]
ChainsParamDict = List[List[ParamDict]]
ChainsScalars = Union[np.ndarray, Sequence[Sequence[float]], Sequence[np.ndarray]]

def _is_torch(x: Any) -> bool:
    return Tensor is not None and isinstance(x, Tensor)

def _is_jax(x: Any) -> bool:
    if jax is None:
        return False
    try:
        return isinstance(x, (jnp.ndarray, getattr(jax, "Array", ())))
    except Exception:
        return False

def to_numpy(x: Any) -> np.ndarray:
    """
    Convert torch/jax/numpy/scalar into numpy array (no copy if possible).
    """
    if _is_torch(x):
        return x.detach().cpu().numpy()
    if _is_jax(x):
        return np.asarray(jax.device_get(x))
    return np.asarray(x)

def to_numpy_1d(x: Any) -> np.ndarray:
    a = to_numpy(x)
    return a.reshape(-1)

def flatten_param_sample(sample: Mapping[str, Any]) -> np.ndarray:
    """
    Deterministic flatten by sorted keys.
    Returns (D,) float64.
    """
    parts: List[np.ndarray] = []
    for k in sorted(sample.keys()):
        parts.append(to_numpy_1d(sample[k]))
    if not parts:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(parts, axis=0).astype(np.float64, copy=False)

def chains_paramdict_to_theta(chains_samples: ChainsParamDict) -> np.ndarray:
    """
    Convert chains->draws->param_dict into theta array shape (C, T, D).
    """
    if len(chains_samples) == 0:
        raise ValueError("chains_samples is empty")
    if len(chains_samples[0]) == 0:
        raise ValueError("chain 0 has 0 draws")

    draws = [len(ch) for ch in chains_samples]
    if len(set(draws)) != 1:
        raise ValueError(f"All chains must have same number of draws. got {draws}")

    C = len(chains_samples)
    T = draws[0]
    D = flatten_param_sample(chains_samples[0][0]).shape[0]

    theta = np.empty((C, T, D), dtype=np.float64)
    for c, chain in enumerate(chains_samples):
        for t, s in enumerate(chain):
            v = flatten_param_sample(s)
            if v.shape[0] != D:
                raise ValueError(f"Param dim mismatch at chain {c} draw {t}: {v.shape[0]} vs {D}")
            theta[c, t, :] = v
    return theta

def chains_scalar_to_theta(chains: ChainsScalars) -> np.ndarray:
    """
    Convert scalar toy samples into theta array shape (C, T, 1).
    Accepts:
      - np.ndarray of shape (T,), (C,T), or (C,T,1)
      - list[list[float]] (C,T)
      - list[np.ndarray] each (T,)
    """
    arr = np.asarray(chains, dtype=np.float64)

    if arr.ndim == 1:
        # (T,) -> (1,T,1)
        return arr[None, :, None]
    if arr.ndim == 2:
        # (C,T) -> (C,T,1)
        return arr[:, :, None]
    if arr.ndim == 3:
        # assume already (C,T,D)
        if arr.shape[-1] == 0:
            raise ValueError("theta has D=0?")
        return arr
    raise ValueError(f"Unsupported scalar chains shape: {arr.shape} (ndim={arr.ndim})")

def summarize_vector(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "median": float("nan")}
    return {"min": float(np.min(x)), "max": float(np.max(x)), "mean": float(np.mean(x)), "median": float(np.median(x))}

def rmse(pred: Any, y: Any) -> float:
    if _is_torch(pred) or _is_torch(y):
        if torch is None:
            raise RuntimeError("torch not available")
        pred_t = pred if _is_torch(pred) else torch.as_tensor(pred)
        y_t = y if _is_torch(y) else torch.as_tensor(y)
        return float(torch.sqrt(torch.mean((pred_t - y_t) ** 2)).item())
    a = to_numpy(pred)
    b = to_numpy(y)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def nll(y: Any, mean: Any, var: Any, eps: float = 1e-8) -> float:
    if _is_torch(y) or _is_torch(mean) or _is_torch(var):
        if torch is None:
            raise RuntimeError("torch not available")
        y_t = y if _is_torch(y) else torch.as_tensor(y)
        m_t = mean if _is_torch(mean) else torch.as_tensor(mean)
        v_t = var if _is_torch(var) else torch.as_tensor(var)
        v_t = torch.clamp(v_t, min=eps)
        return float(torch.mean(0.5 * torch.log(2 * torch.pi * v_t) + 0.5 * ((y_t - m_t) ** 2) / v_t).item())
    y_np = to_numpy(y)
    m_np = to_numpy(mean)
    v_np = np.clip(to_numpy(var), a_min=eps, a_max=None)
    return float(np.mean(0.5 * np.log(2 * np.pi * v_np) + 0.5 * ((y_np - m_np) ** 2) / v_np))

def param_norm(sample: Mapping[str, Any]) -> float:
    """
    L2 norm of concatenated parameters in a dict.
    """
    s = 0.0
    for v in sample.values():
        a = to_numpy(v).astype(np.float64, copy=False)
        s += float(np.sum(a ** 2))
    return float(math.sqrt(s))

def L2_distance(p: Mapping[str, Any], q: Mapping[str, Any]) -> float:
    """
    L2 distance between two parameter dictionaries (same keys/shapes).
    Supports torch or jax/numpy arrays (mixed ok, will route through numpy).
    """
    first = next(iter(p.values()))
    if _is_torch(first) and all(_is_torch(v) for v in p.values()) and all(_is_torch(v) for v in q.values()):
        s = 0.0
        for k in p.keys():
            a = p[k]
            b = q[k]
            s += float(((a - b) ** 2).sum().item())
        return float(math.sqrt(s))
    s = 0.0
    for k in p.keys():
        a = to_numpy(p[k]).astype(np.float64, copy=False)
        b = to_numpy(q[k]).astype(np.float64, copy=False)
        s += float(np.sum((a - b) ** 2))
    return float(math.sqrt(s))

def _idata_from_theta(theta: np.ndarray):
    """
    theta: (C, T, D) float64 recommended.
    Returns arviz InferenceData with posterior['theta'].
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 3:
        raise ValueError(f"theta must be (chains, draws, dims). got {theta.shape}")
    return az.from_dict(posterior={"theta": theta})

def n_eff_from_theta(theta: np.ndarray, *, method: str = "bulk", relative: bool = False) -> np.ndarray:
    """
    Returns ESS per-dimension as a 1D numpy array (D,).
    """
    idata = _idata_from_theta(theta)
    ess = az.ess(idata, var_names=["theta"], method=method, relative=relative)
    vals = np.asarray(ess["theta"].values, dtype=np.float64).reshape(-1)
    return vals

def r_hat_from_theta(theta: np.ndarray, *, method: str = "rank") -> np.ndarray:
    """
    Returns R-hat per-dimension as a 1D numpy array (D,).
    Requires theta.shape[0] >= 2.
    """
    theta = np.asarray(theta)
    if theta.shape[0] < 2:
        raise ValueError("R-hat requires at least 2 chains")
    idata = _idata_from_theta(theta)
    rhat = az.rhat(idata, var_names=["theta"], method=method)
    vals = np.asarray(rhat["theta"].values, dtype=np.float64).reshape(-1)
    return vals

def n_eff(chains_samples: Union[ChainsParamDict, ChainsScalars, np.ndarray], *, method: str = "bulk", relative: bool = False) -> Dict[str, float]:
    """
    Unified API:
      - If input is chains->draws->dict, computes theta via flattening.
      - If input is scalar samples, computes theta as (C,T,1).
      - If input is already theta (C,T,D), uses directly.
    Returns summary dict (min/max/mean/median) across parameters.
    """
    theta = _to_theta_any(chains_samples)
    ess_vals = n_eff_from_theta(theta, method=method, relative=relative)
    return summarize_vector(ess_vals)

def r_hat(chains_samples: Union[ChainsParamDict, ChainsScalars, np.ndarray], *, method: str = "rank") -> Dict[str, float]:
    """
    Unified API for R-hat. Requires >=2 chains.
    Returns summary dict (min/max/mean/median) across parameters.
    """
    theta = _to_theta_any(chains_samples)
    rhat_vals = r_hat_from_theta(theta, method=method)
    return summarize_vector(rhat_vals)

def _to_theta_any(x: Union[ChainsParamDict, ChainsScalars, np.ndarray]) -> np.ndarray:
    """
    Convert various chain formats into theta (C,T,D).
    """
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list) and len(x[0]) > 0 and isinstance(x[0][0], dict):
        return chains_paramdict_to_theta(x)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 3:
        return arr
    return chains_scalar_to_theta(arr)

@dataclass
class MetricsReport:
    n_eff: Optional[Dict[str, float]] = None
    r_hat: Optional[Dict[str, float]] = None
    rmse: Optional[float] = None
    nll: Optional[float] = None
    param_norm: Optional[float] = None
    l2_distance: Optional[float] = None

def compute_diagnostics(chains_samples: Union[ChainsParamDict, ChainsScalars, np.ndarray], *,
                        ess_method: str = "bulk", ess_relative: bool = False,
                        rhat_method: str = "rank", compute_rhat: bool = True) -> MetricsReport:
    rep = MetricsReport()
    rep.n_eff = n_eff(chains_samples, method=ess_method, relative=ess_relative)
    if compute_rhat:
        rep.r_hat = r_hat(chains_samples, method=rhat_method)
    return rep