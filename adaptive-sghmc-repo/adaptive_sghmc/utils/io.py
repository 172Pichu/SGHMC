from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported array type: {type(x)}")

def _json_ready(obj: Any) -> Any:
    """
    Convert nested experiment results into JSON-serializable objects.
    """
    if is_dataclass(obj):
        return _json_ready(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return {"__tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def make_run_name(dataset: str, method: str, fold_id: int, seed: int, layers: int, units: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{dataset}_{method}_fold{fold_id}_seed{seed}_{layers}h{units}d"

def save_experiment_result(result: dict, *, save_root: str | Path, run_name: str) -> Path:
    """
    Save experiment summary and raw arrays separately.

    Output layout:
        save_root/
          run_name/
            summary.json
            chains.npy          (optional)
            samples.npy         (optional)
            scalers.pt          (optional)
    """
    save_dir = Path(save_root) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(result)

    # Pop large / non-JSON objects and save them separately.
    chains = summary.pop("chains", None)
    samples = summary.pop("samples", None)
    X_scaler = summary.pop("X_scaler", None)
    y_scaler = summary.pop("y_scaler", None)

    if chains is not None:
        np.save(save_dir / "chains.npy", _to_numpy(chains))

    if samples is not None:
        np.save(save_dir / "samples.npy", _to_numpy(samples))

    if X_scaler is not None or y_scaler is not None:
        torch.save({"X_scaler": X_scaler, "y_scaler": y_scaler}, save_dir / "scalers.pt")

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready(summary), f, indent=2, ensure_ascii=False)

    return save_dir