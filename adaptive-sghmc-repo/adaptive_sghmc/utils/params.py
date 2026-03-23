from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch import Tensor

@dataclass(frozen=True)
class ParamSpec:
    """
    Specification for flattening / unflattening an MLP parameter dictionary.
    names: Deterministic parameter ordering, e.g. ["W1", "b1", "W2", "b2", ...].
    shapes: Tensor shape of each parameter in "names".
    sizes: Flattened number of scalars for each parameter.
    total_numel: Total flattened dimension of the whole parameter vector theta.
    """
    names: List[str]
    shapes: Dict[str, Tuple[int, ...]]
    sizes: Dict[str, int]
    total_numel: int

def build_mlp_param_spec(in_dim: int, out_dim: int, num_hidden_layers: int, num_hidden_units: int) -> ParamSpec:
    """
    Build a deterministic parameter specification for a fully-connected MLP.
    Naming convention: W1, b1, W2, b2, ..., WL, bL
    Example: hidden_layers = 2 means: input -> hidden1 -> hidden2 -> output
    """
    if in_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {in_dim}")
    if out_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {out_dim}")
    if num_hidden_layers < 0:
        raise ValueError(f"the number of hidden_layers must be >= 0, got {num_hidden_layers}")
    if num_hidden_units <= 0:
        raise ValueError(f"the number of hidden_units must be positive, got {num_hidden_units}")

    layer_dims = [in_dim]
    for _ in range(num_hidden_layers):
        layer_dims.append(num_hidden_units)
    layer_dims.append(out_dim)

    names: List[str] = []
    shapes: Dict[str, Tuple[int, ...]] = {}
    sizes: Dict[str, int] = {}

    total = 0
    num_linear_layers = len(layer_dims) - 1

    for layer_idx in range(num_linear_layers):

        in_dim = layer_dims[layer_idx]
        out_dim = layer_dims[layer_idx + 1]

        w_name = f"W{layer_idx + 1}"
        b_name = f"b{layer_idx + 1}"

        w_shape = (in_dim, out_dim)
        b_shape = (out_dim,)

        w_size = in_dim * out_dim
        b_size = out_dim

        names.extend([w_name, b_name])

        shapes[w_name] = w_shape
        shapes[b_name] = b_shape

        sizes[w_name] = w_size
        sizes[b_name] = b_size

        total += w_size + b_size

    return ParamSpec(names=names, shapes=shapes, sizes=sizes, total_numel=total)

def flatten_params(params: Dict[str, Tensor], spec: ParamSpec) -> Tensor:
    """
    Flatten a parameter dictionary into a 1D tensor theta with deterministic order.

    The output keeps the dtype / device of the input tensors.
    """
    missing = [name for name in spec.names if name not in params]
    if missing:
        raise KeyError(f"Missing parameters: {missing}")
    parts = []
    first = params[spec.names[0]]
    device = first.device
    dtype = first.dtype
    for name in spec.names:
        value = params[name]
        if tuple(value.shape) != spec.shapes[name]:
            raise ValueError(f"Shape mismatch for {name}: got {tuple(value.shape)}, expected {spec.shapes[name]}")
        parts.append(value.reshape(-1))
    if not parts:
        return torch.empty(0, device=device, dtype=dtype)
    return torch.cat(parts, dim=0)

def unflatten_theta(theta: Tensor, spec: ParamSpec) -> Dict[str, Tensor]:
    """
    Convert a flat 1D parameter vector theta into a parameter dictionary.
    """
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1D, got shape {tuple(theta.shape)}")
    if theta.numel() != spec.total_numel:
        raise ValueError(f"theta.numel()={theta.numel()} does not match spec.total_numel={spec.total_numel}")
    params: Dict[str, Tensor] = {}
    offset = 0
    for name in spec.names:
        size = spec.sizes[name]
        shape = spec.shapes[name]
        chunk = theta[offset: offset + size]
        params[name] = chunk.reshape(shape)
        offset += size
    return params

def glorot_std(fan_in: int, fan_out: int) -> float:
    """
    Xavier/Glorot normal initializer standard deviation.
    """
    return (2.0 / float(fan_in + fan_out)) ** 0.5

def init_theta_glorot(spec: ParamSpec, *, seed: int, device: str = "cpu", dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Initialize theta by first sampling each weight/bias tensor, then flattening.
    Weights: Normal(0, glorot_std^2)
    Biases: Zero
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    params: Dict[str, Tensor] = {}
    for name in spec.names:
        shape = spec.shapes[name]
        if name.startswith("W"):
            fan_in, fan_out = shape
            std = glorot_std(fan_in, fan_out)
            value = torch.randn(shape, generator=gen, device=device, dtype=dtype) * std
        elif name.startswith("b"):
            value = torch.zeros(shape, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown parameter name pattern: {name}")
        params[name] = value
    return flatten_params(params, spec)