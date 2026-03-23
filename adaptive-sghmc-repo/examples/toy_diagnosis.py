from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, csv, os
import numpy as np
from typing import Dict, Optional
from toy_metrics import n_eff, r_hat

def finite_chains(chains: np.ndarray, name: str) -> np.ndarray:
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim == 3:
        pass
    elif chains.ndim == 2:
        pass
    else:
        raise ValueError(f"{name}: chains must be (C,T) or (C,T,D), got {chains.shape}")
    if np.isfinite(chains).all():
        return chains
    bad = int((~np.isfinite(chains)).sum())
    print(f"[WARN] {name}: found {bad} non-finite entries in chains; impute with per-chain median.")
    out = chains.copy()
    if out.ndim == 2:
        for c in range(out.shape[0]):
            m = np.isfinite(out[c])
            fill = float(np.median(out[c, m])) if np.any(m) else 0.0
            out[c, ~m] = fill
        return out
    C, T, D = out.shape
    for c in range(C):
        for d in range(D):
            m = np.isfinite(out[c, :, d])
            fill = float(np.median(out[c, m, d])) if np.any(m) else 0.0
            out[c, ~m, d] = fill
    return out

def print_diag(name: str, ess: Dict[str, float], rh: Optional[Dict[str, float]]) -> None:
    print(f"==== {name} diagnostics ====")
    print(f"n_eff    min/max/mean/median = {ess['min']:.3f} / {ess['max']:.3f} / {ess['mean']:.3f} / {ess['median']:.3f}")
    if rh is None:
        print("r_hat    (need >=2 chains)  = skipped")
    else:
        print(f"r_hat    min/max/mean/median = {rh['min']:.6f} / {rh['max']:.6f} / {rh['mean']:.6f} / {rh['median']:.6f}")

def append_csv(csv_path: str, sampler: str, args: argparse.Namespace,
               ess: Dict[str, float], rh: Optional[Dict[str, float]]) -> None:
    
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    header = ["sampler", "seed", "chains", "samples", "burn_in", "keep_every",
              "inner_steps", "reset_prob", "eps", "mdecay",
              "ess_min", "ess_max", "ess_mean", "ess_median", "rhat_min", "rhat_max", "rhat_mean", "rhat_median"]

    row = [sampler, int(args.seed), int(args.chains), int(args.samples), int(args.burn_in), int(args.keep_every),
           int(args.inner_steps), float(args.reset_prob), float(args.eps), float(args.mdecay),
           float(ess["min"]), float(ess["max"]), float(ess["mean"]), float(ess["median"])]

    if rh is None:
        row += [np.nan, np.nan, np.nan, np.nan]
    else:
        row += [float(rh["min"]), float(rh["max"]), float(rh["mean"]), float(rh["median"])]

    need_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)

def diagnose_with_samples(name: str, chains: np.ndarray, args):
    ess = n_eff(chains)
    rh = None
    if chains.shape[0] >= 2:
        rh = r_hat(chains)
    print_diag(name, ess, rh)
    append_csv("result/toy_diag.csv", name, args, ess, rh)