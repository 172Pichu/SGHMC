from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

@dataclass(frozen=True)
class RegressionSplit:
    """
    One pre-defined train / validation / test split for regression.
    """
    X_train: Tensor
    y_train: Tensor
    X_val: Tensor
    y_val: Tensor
    X_test: Tensor
    y_test: Tensor

@dataclass(frozen=True)
class TensorNormalizer:
    """
    Feature/target normalizer using train-set statistics only.
    """
    mean: Tensor
    std: Tensor

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

def _safe_std(x: Tensor, dim: int = 0, eps: float = 1e-8) -> Tensor:
    std = x.std(dim=dim, unbiased=False)
    return torch.clamp(std, min=eps)

def fit_normalizer(x: Tensor) -> TensorNormalizer:
    mean = x.mean(dim=0)
    std = _safe_std(x, dim=0)
    return TensorNormalizer(mean=mean, std=std)

def _load_csv_2d(path: str | Path) -> Tensor:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return torch.tensor(arr, dtype=torch.float32)

def load_predefined_yacht_fold(root: str | Path, fold_id: int) -> RegressionSplit:

    if not (1 <= fold_id <= 10):
        raise ValueError(f"fold_id must be in [1, 10], got {fold_id}")
    fold_dir = Path(root) / "yacht" / str(fold_id)
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory does not exist: {fold_dir}")

    X_train = _load_csv_2d(fold_dir / "train_X.csv")
    y_train = _load_csv_2d(fold_dir / "train_y.csv")
    X_val = _load_csv_2d(fold_dir / "val_X.csv")
    y_val = _load_csv_2d(fold_dir / "val_y.csv")
    X_test = _load_csv_2d(fold_dir / "test_X.csv")
    y_test = _load_csv_2d(fold_dir / "test_y.csv")

    if X_train.shape[1] != 6:
        raise ValueError(f"Expected 6 input features, got x_train.shape={tuple(X_train.shape)}")
    if y_train.shape[1] != 1:
        raise ValueError(f"Expected scalar target, got y_train.shape={tuple(y_train.shape)}")

    return RegressionSplit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

def normalize_split(split: RegressionSplit) -> tuple[RegressionSplit, TensorNormalizer | None, TensorNormalizer | None]:

    X_scaler = fit_normalizer(split.X_train)
    y_scaler = fit_normalizer(split.y_train)

    X_train = X_scaler.normalize(split.X_train) if X_scaler is not None else split.X_train
    X_val = X_scaler.normalize(split.X_val) if X_scaler is not None else split.X_val
    X_test = X_scaler.normalize(split.X_test) if X_scaler is not None else split.X_test

    y_train = y_scaler.normalize(split.y_train) if y_scaler is not None else split.y_train
    y_val = y_scaler.normalize(split.y_val) if y_scaler is not None else split.y_val
    y_test = y_scaler.normalize(split.y_test) if y_scaler is not None else split.y_test

    out = RegressionSplit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    return out, X_scaler, y_scaler