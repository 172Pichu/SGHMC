from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from .utils.params import ParamSpec, unflatten_theta

@dataclass(frozen=True)
class RegressionBatch:
    """
    A simple regression batch container.
    X: Input features of shape (batch_size, input_dim)
    y: Regression targets of shape (batch_size, 1)
    """
    X: Tensor
    y: Tensor

def _activation_fn(name: str) -> Callable[[Tensor], Tensor]:
    """
    Return a Torch activation function by name.
    """
    name = name.lower()
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu
    if name == "relu":
        return torch.relu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")

def mlp_forward_from_theta(theta: Tensor, spec: ParamSpec, X: Tensor, act_fn: str = "leaky_relu") -> Tensor:
    """
    Forward pass of an MLP whose parameters are packed into a flat vector theta.
    """
    params = unflatten_theta(theta, spec)
    act = _activation_fn(act_fn)
    num_linear_layers = len(spec.names) // 2
    h = X
    for layer_idx in range(1, num_linear_layers + 1):
        w = params[f"W{layer_idx}"]
        b = params[f"b{layer_idx}"]
        h = h @ w + b
        if layer_idx < num_linear_layers:
            h = act(h)
    return h

def gaussian_nll_sum(pred: Tensor, target: Tensor, likelihood_var: float) -> Tensor:
    """
    Sum of Gaussian negative log-likelihood terms.
    """
    if likelihood_var <= 0.0:
        raise ValueError(f"noise_var must be positive, got {likelihood_var}")
    diff = target - pred
    c = torch.tensor(2.0 * torch.pi * likelihood_var, device=pred.device, dtype=pred.dtype)
    return 0.5 * (torch.log(c) + diff.pow(2) / likelihood_var).sum()

def gaussian_prior_nll_sum(theta: Tensor, prior_var: float,) -> Tensor:
    """
    Sum of Gaussian prior negative log-likelihood terms.
    """
    if prior_var <= 0.0:
        raise ValueError(f"prior_var must be positive, got {prior_var}")
    c = torch.tensor(2.0 * torch.pi * prior_var, device=theta.device, dtype=theta.dtype)
    return 0.5 * (torch.log(c) + theta.pow(2) / prior_var).sum()

def make_fullbatch_grad_estimator(X_train: Tensor, y_train: Tensor, spec: ParamSpec, act_fn: str,
                                  prior_var: float, likelihood_var: float) -> Callable[[Tensor], Tensor]:
    """
    Build a full-batch gradient estimator for posterior sampling.
    """
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(-1)

    def grad_estimator(theta: Tensor) -> Tensor:

        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D, got shape {tuple(theta.shape)}")
        
        theta_local = theta.detach().clone().requires_grad_(True)
        pred = mlp_forward_from_theta(theta=theta_local, spec=spec, X=X_train, act_fn=act_fn)

        loss = gaussian_nll_sum(pred, y_train, likelihood_var) + gaussian_prior_nll_sum(theta_local, prior_var)
        (grad,) = torch.autograd.grad(loss, theta_local)

        return grad.detach()

    return grad_estimator

def make_minibatch_grad_estimator(X_train: Tensor, y_train: Tensor, spec: ParamSpec, act_fn: str,
                                  prior_var: float, likelihood_var: float, seed: int, batch_size: int) -> Callable[[Tensor], Tensor]:
    """
    Build a mini-batch stochastic gradient estimator for posterior sampling.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(-1)

    num_train = X_train.shape[0]
    if batch_size > num_train:
        raise ValueError(f"batch_size={batch_size} cannot exceed num_train={num_train} in this implementation")
    
    gen = torch.Generator(device=X_train.device)
    gen.manual_seed(int(seed))

    def grad_estimator(theta: Tensor) -> Tensor:

        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D, got shape {tuple(theta.shape)}")
        
        idx = torch.randperm(num_train, generator=gen, device=X_train.device)[:batch_size]
        X_epoch = X_train[idx]
        y_epoch = y_train[idx]

        theta_local = theta.detach().clone().requires_grad_(True)
        pred = mlp_forward_from_theta(theta=theta_local, spec=spec, X=X_epoch, act_fn=act_fn)

        data_nll = gaussian_nll_sum(pred, y_epoch, likelihood_var)
        scaled_data_nll = data_nll * (num_train / batch_size)

        loss = scaled_data_nll + gaussian_prior_nll_sum(theta_local, prior_var)
        (grad,) = torch.autograd.grad(loss, theta_local)

        return grad.detach()

    return grad_estimator