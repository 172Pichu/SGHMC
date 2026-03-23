from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor

A = 1.0
B = 0.2
PHI = np.pi / 6.0

ArrayLike = np.ndarray | Tensor

def _is_torch(x: object) -> bool:
    return isinstance(x, torch.Tensor)

def U_fig1(x: ArrayLike) -> ArrayLike:
    return (x ** 2) * (x ** 2 - 2.0)

def unnormed_density_fig1(plot_domain: np.ndarray) -> np.ndarray:
    return np.exp(-U_fig1(plot_domain))

def integral_even(f: Callable[[np.ndarray], np.ndarray], num_integral_grids_per_unit: int, integral_max: float) -> float:
    integral_domain = np.linspace(0.0, integral_max, int(num_integral_grids_per_unit * integral_max) + 1)
    y = f(integral_domain)
    return np.trapz(y, integral_domain) * 2.0

def true_density_fig1(*, num_integral_grids_per_unit: int = 2000, integral_max: float = 4.0,
                      num_plot_grids_per_unit: int = 500, plot_max: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    plot_domain = np.linspace(-plot_max, plot_max, int(num_plot_grids_per_unit * plot_max * 2.0) + 1)
    Z = integral_even(unnormed_density_fig1, num_integral_grids_per_unit, integral_max)
    return plot_domain, unnormed_density_fig1(plot_domain) / Z

def dU_dx_fig1(x: ArrayLike) -> ArrayLike:
    return 4.0 * x * (x + 1.0) * (x - 1.0)

def rotation_matrix(phi: float) -> np.ndarray:
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    return np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]], dtype=np.float64)

def covariance_matrix(a: float = A, b: float = B, phi: float = PHI) -> Tuple[np.ndarray, np.ndarray]:
    R = rotation_matrix(phi)
    sigma = R @ np.diag([a * a, b * b]) @ R.T
    inv_sigma = np.linalg.inv(sigma)
    return sigma, inv_sigma

def U_fig3_xy(xy: ArrayLike, *, a: float = A, b: float = B, phi: float = PHI) -> ArrayLike:
    _, inv_sigma = covariance_matrix(a, b, phi)
    if _is_torch(xy):
        inv_sigma_t = torch.as_tensor(inv_sigma, dtype=xy.dtype, device=xy.device)
        return 0.5 * torch.einsum('...i,ij,...j->...', xy, inv_sigma_t, xy)
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[-1] != 2:
        raise ValueError(f"xy must have last dim 2, got shape {xy.shape}")
    return 0.5 * np.einsum('...i,ij,...j->...', xy, inv_sigma, xy)

def grad_U_fig3_xy(xy: ArrayLike, *, a: float = A, b: float = B, phi: float = PHI) -> ArrayLike:
    _, inv_sigma = covariance_matrix(a, b, phi)
    if _is_torch(xy):
        inv_sigma_t = torch.as_tensor(inv_sigma, dtype=xy.dtype, device=xy.device)
        return torch.einsum('ij,...j->...i', inv_sigma_t, xy)
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[-1] != 2:
        raise ValueError(f"xy must have last dim 2, got shape {xy.shape}")
    return np.einsum('ij,...j->...i', inv_sigma, xy)

def density_fig3_xy(xy: np.ndarray, *, a: float = A, b: float = B, phi: float = PHI) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    return np.exp(-U_fig3_xy(xy, a=a, b=b, phi=phi)) / (2.0 * np.pi * a * b)