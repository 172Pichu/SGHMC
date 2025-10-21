import argparse
import csv
import os
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Tuple

import numpy as np
from jax import jit, numpy as jnp, random, tree_util
import blackjax

@dataclass
class DataNormalizer:

    X_mean: jnp.ndarray
    X_std: jnp.ndarray
    y_mean: jnp.ndarray
    y_std: jnp.ndarray

    @classmethod
    def fit(cls, X: jnp.ndarray, y: jnp.ndarray):
        X_mean = jnp.mean(X, axis=0)
        X_std = jnp.std(X, axis=0) + 1e-8
        y_mean = jnp.mean(y, axis=0)
        y_std = jnp.std(y, axis=0) + 1e-8
        return cls(X_mean, X_std, y_mean, y_std)

    def zscore_normalize_x(self, X: jnp.ndarray) -> jnp.ndarray:
        return (X - self.X_mean) / self.X_std

    def zscore_normalize_y(self, y: jnp.ndarray) -> jnp.ndarray:
        return (y - self.y_mean) / self.y_std

    def zscore_unnormalize_y(self, y_norm: jnp.ndarray) -> jnp.ndarray:
        return y_norm * self.y_std + self.y_mean


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def round_list(xs: Iterable[float]) -> List[float]:
    return [float(np.round(x, 5)) for x in xs]


def load_csv(path: str, has_header: bool = False) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0)
    if data.ndim == 1:
        data = data[None, :]
    return data


def load_split(dataset: str, fold_id: int, usage: str, data_root: str, has_header: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    base_path = os.path.join(data_root, dataset, str(fold_id))
    x_path = os.path.join(base_path, f"{usage}_X.csv")
    y_path = os.path.join(base_path, f"{usage}_y.csv")

    X_np = load_csv(x_path, has_header=has_header).astype(np.float32)
    y_np = load_csv(y_path, has_header=has_header).astype(np.float32)

    if y_np.ndim == 1:
        y_np = y_np[:, None]
    elif y_np.shape[0] == 1 and y_np.shape[1] > 1:
        y_np = y_np.T
    else:
        raise ValueError(f"Unexpected y shape {y_np.shape}. Please check if the CSV file has correct orientation.")

    if X_np.shape[0] != y_np.shape[0]:
        if X_np.T.shape[0] == y_np.shape[0]:
            X_np = X_np.T
        else:
            raise ValueError(f"[{usage}] X/Y size mismatch after normalization: X = {X_np.shape}, y = {y_np.shape}.")
        
    return jnp.array(X_np), jnp.array(y_np)


def glorot_init(rng, in_dim, out_dim):
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    rng, wk, bk = random.split(rng, 3)
    W = random.uniform(wk, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return rng, W, b


def init_mlp_params(rng, input_dim: int, output_dim: int, hidden_units: int, n_hidden: int):
    params = {}
    dims = [input_dim] + [hidden_units] * n_hidden + [output_dim]
    for i in range(len(dims) - 1):
        rng, W, b = glorot_init(rng, dims[i], dims[i + 1])
        params[f"W{i}"] = W
        params[f"b{i}"] = b
    return rng, params


def mlp_forward(params: Dict[str, jnp.ndarray], x: jnp.ndarray, act_fn: str) -> jnp.ndarray:
    n_layers = len(params) // 2
    h = x
    for i in range(n_layers - 1):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        if act_fn == "relu":
            h = jnp.maximum(h, 0)
        elif act_fn == "tanh":
            h = jnp.tanh(h)
        else:
            raise ValueError(f"Unknown act_fn: {act_fn}")
    y = h @ params[f"W{n_layers-1}"] + params[f"b{n_layers-1}"]
    return y


def rmse(pred: jnp.ndarray, y: jnp.ndarray) -> float:
    return float(jnp.sqrt(jnp.mean((pred - y) ** 2)))


def nll(y: jnp.ndarray, mean: jnp.ndarray, var: jnp.ndarray) -> float:
    eps = 1e-8
    var = jnp.clip(var, a_min=eps)
    return float(jnp.mean(0.5 * jnp.log(2 * jnp.pi * var) + 0.5 * ((y - mean) ** 2) / var))


def build_grad_estimator(data_size: int, act_fn: str, prior_var: float, noise_var: float):

    inv_prior_var = 0.0 if (prior_var is None) else (1.0 / float(prior_var))

    def logprior_fn(params):
        if inv_prior_var == 0.0:
            return 0.0
        sqsum = sum(jnp.sum(w**2) for w in tree_util.tree_leaves(params))
        return -0.5 * sqsum * inv_prior_var

    def loglikelihood_fn(params, batch):
        x, y = batch
        pred_norm = mlp_forward(params, x, act_fn)
        resid = y - pred_norm
        return -0.5 * jnp.sum(resid**2) / noise_var

    return blackjax.sgmcmc.gradients.grad_estimator(logprior_fn, loglikelihood_fn, data_size)


def build_sghmc_kernel(data_size: int, num_integration_steps: int, alpha: float, beta: float,
                       act_fn: str, prior_var: float, noise_var: float):
    grad_est = build_grad_estimator(data_size, act_fn, prior_var, noise_var)
    return blackjax.sghmc(grad_est, num_integration_steps=num_integration_steps, alpha=alpha, beta=beta)


def evaluate_with_samples(samples: List[dict], normalizer: DataNormalizer,
                          x_cpu: jnp.ndarray, y_cpu: jnp.ndarray, act_fn: str,
                          noise_var: float) -> Tuple[float, float]:

    Xn = normalizer.zscore_normalize_x(x_cpu)
    preds = []

    for params in samples:
        pred_norm = mlp_forward(params, Xn, act_fn)
        pred = normalizer.zscore_unnormalize_y(pred_norm)
        preds.append(pred)

    S = len(preds)
    preds_stacked = jnp.stack(preds, axis=0)        # (S, N, 1)
    mean_pred = jnp.mean(preds_stacked, axis=0)     # (N, 1)
    var_model = jnp.var(preds_stacked, axis=0)      # (N, 1)

    noise_var_adj = noise_var * (normalizer.y_std ** 2)
    total_var = var_model + noise_var_adj

    r = rmse(mean_pred, y_cpu)
    n = nll(y_cpu, mean_pred, total_var)

    return r, n


def batchify(X: jnp.ndarray, Y: jnp.ndarray, batch_size: int) -> Iterable[Tuple[jnp.ndarray, jnp.ndarray]]:
    n = X.shape[0]
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        yield X[sl], Y[sl]


def _l2dist_params(p: Dict[str, jnp.ndarray], q: Dict[str, jnp.ndarray]) -> float:
    keys = p.keys()
    s = 0.0
    for k in keys:
        a, b = p[k], q[k]
        s += float(jnp.sum((a - b) ** 2))
    return float(jnp.sqrt(s))


def run_fold(args, fold_id: int):

    # Load data
    Xtr, ytr = load_split(args.dataset, fold_id, 'train', args.data_root, args.has_header)
    Xva, yva = load_split(args.dataset, fold_id, 'val', args.data_root, args.has_header)
    Xte, yte = load_split(args.dataset, fold_id, 'test', args.data_root, args.has_header)

    # Show ratios
    n_tr, n_va, n_te = Xtr.shape[0], Xva.shape[0], Xte.shape[0]
    print(f"train/val/test sizes = {n_tr}/{n_va}/{n_te}")

    # Fit normalizer on TRAIN only
    normalizer = DataNormalizer.fit(Xtr, ytr)
    Xtr_n = normalizer.zscore_normalize_x(Xtr)
    ytr_n = normalizer.zscore_normalize_y(ytr)

    input_dim = Xtr_n.shape[1]
    output_dim = 1

    # Build model params
    master = random.PRNGKey(args.seed)
    master, k_params = random.split(master)

    # SGHMC kernel
    data_size = Xtr_n.shape[0]
    beta_to_use = args.beta if args.beta is not None else (2.0 * args.alpha)
    sghmc = build_sghmc_kernel(
        data_size=data_size,
        num_integration_steps=args.num_integration_steps,
        alpha=args.alpha,
        beta=beta_to_use,
        act_fn=args.act_fn,
        prior_var=args.prior_var,
        noise_var=args.noise_var,
    )
    step = jit(sghmc.step)

    batch_size = args.batch
    burn_in = args.num_burn_in_steps
    keep_every = args.keep_every
    n_keep = args.num_samples
    n_discarded = args.n_discarded
    print_every = args.print_every_n_samples

    all_samples: List[dict] = []
    start_t = time.perf_counter()

    # Multiple chains (sequential)
    for chain_idx in range(args.num_chains):

        chain_seed = args.seed + chain_idx * args.chain_seed_stride
        rng = random.PRNGKey(chain_seed)

        # Fresh params per chain
        rng, k_params = random.split(rng)
        _, init_params = init_mlp_params(k_params, input_dim, output_dim, args.n_units, args.n_hidden)

        state = sghmc.init(init_params)
        params = init_params

        total_steps = 0
        collected = 0
        chain_samples: List[dict] = []

        print(f"Chain: {chain_idx}")

        # Epoch-like loops for shuffling
        while collected < n_keep:

            # Shuffle each epoch
            rng, k_perm = random.split(rng)
            perm = random.permutation(k_perm, data_size)
            Xep, Yep = Xtr_n[perm], ytr_n[perm]

            for batch in batchify(Xep, Yep, batch_size):

                rng, k_step = random.split(rng)
                out = step(k_step, state, batch, args.lr)
                params = None

                if isinstance(out, (tuple, list)):
                    state = out[0]
                elif isinstance(out, dict):
                    cand_state = out.get("state") or out.get("new_state")
                    if cand_state is not None:
                        state = cand_state
                    else:
                        if all(isinstance(k, str) for k in out.keys()) and any(k.startswith(("W", "b")) for k in out.keys()):
                            params = out
                            state = out
                        else:
                            state = out
                else:
                    if hasattr(out, "state"):
                        state = out.state
                    else:
                        state = out
                if params is None:
                    params = state.position if hasattr(state, "position") else state

                total_steps += 1

                if total_steps > burn_in and (total_steps - burn_in) % keep_every == 0:

                    collected += 1

                    if collected > n_discarded:

                        sample = {k: v.copy() for k, v in params.items()}
                        chain_samples.append(sample)
                        all_samples.append(sample)

                        if (collected - n_discarded) % print_every == 0:

                            r_te, n_te = evaluate_with_samples(chain_samples, normalizer, Xte, yte, args.act_fn, args.noise_var)

                            if len(chain_samples) >= 2:
                                d_last = _l2dist_params(chain_samples[-1], chain_samples[-2])
                                print(
                                    f"Samples # {collected - n_discarded:>3d}:  "
                                    f"RMSE = {r_te:>8.4f}  |  "
                                    f"NLL = {n_te:>8.4f}  |  "
                                    f"Δparams = {d_last:>10.3e}"
                                )
                            else:
                                print(
                                    f"Samples # {collected - n_discarded:>3d}:  "
                                    f"RMSE = {r_te:>8.4f}  |  "
                                    f"NLL = {n_te:>8.4f}"
                                )

                if collected >= n_keep:
                    break

    elapsed = int(time.perf_counter() - start_t)

    if not all_samples:
        raise RuntimeError("No SGHMC samples collected; consider lowering burn-in or keep_every.")

    # Final eval
    r_tr, n_tr = evaluate_with_samples(all_samples, normalizer, Xtr, ytr, args.act_fn, args.noise_var)
    r_va, n_va = evaluate_with_samples(all_samples, normalizer, Xva, yva, args.act_fn, args.noise_var)
    r_te, n_te = evaluate_with_samples(all_samples, normalizer, Xte, yte, args.act_fn, args.noise_var)

    row = [args.lr, args.prior_var, args.noise_var,
           *round_list([r_tr, r_va, r_te, n_tr, n_va, n_te]), elapsed]
    
    return row


def main():

    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")

    parser = argparse.ArgumentParser()

    # Dataset Configuration
    parser.add_argument('--dataset', type=str, default='yacht', help="Dataset name (default: 'yacht')")
    parser.add_argument('--id', type=int, default=1, help="Fold ID specifier (default: 1)")
    parser.add_argument('--data_root', type=str, default='./data', help="Root directory containing CSV splits")
    parser.add_argument('--has_header', action='store_true', help="Set if CSV files contain a header row")

    # Model Configuration
    parser.add_argument('--act_fn', type=str, choices=['relu', 'tanh'], default='relu')
    parser.add_argument('--n_units', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=2)

    # Execution Setup
    parser.add_argument('--train_eval', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--pre_learn', type=str, choices=['opt', 'cv'], default='cv')

    # Device (fixed to CPU here for simplicity)
    parser.add_argument('--device', type=str, choices=['cpu'], default='cpu')

    # Seeds & batching
    parser.add_argument('--seed', type=int, default=77210)
    parser.add_argument('--chain_seed_stride', type=int, default=1)
    parser.add_argument('--batch', type=int, default=32)

    # SGHMC kernel & schedule
    parser.add_argument('--alpha', type=float, default=1e-2, help="momentum decay (friction)")
    parser.add_argument('--beta', type=float, default=None, help="SGHMC diffusion term, default = 2 * mdecay")
    parser.add_argument('--num_chains', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=40)
    parser.add_argument('--n_discarded', type=int, default=10)
    parser.add_argument('--num_burn_in_steps', type=int, default=500)
    parser.add_argument('--keep_every', type=int, default=100)
    parser.add_argument('--num_integration_steps', type=int, default=10)
    parser.add_argument('--print_every_n_samples', type=int, default=5)

    # Posterior / Likelihood
    parser.add_argument('--lr', type=float, default=1e-3, help='SGHMC step size (η)')
    parser.add_argument('--prior_var', type=float, default=None)
    parser.add_argument('--noise_var', type=float, default=None)

    # Grid for CV mode
    parser.add_argument('--grid_lrs', type=str, default='', help="Comma-separated list; overrides --lr in cv mode")
    parser.add_argument('--grid_prior_vars', type=str, default='', help="Comma-separated list; overrides --prior_var in cv mode")
    parser.add_argument('--grid_noise_vars', type=str, default='', help="Comma-separated list; overrides --noise_var in cv mode")

    args = parser.parse_args()

    # Resolve fold ids (single or list). For parity with torch, keep a single fold from --id.
    fold_ids = [args.id]

    # Prepare output directory and filename
    save_path = os.path.join('./results/sghmc_bjx', args.dataset, args.pre_learn)
    if args.train_eval == 'eval':
        save_path = os.path.join(save_path, 'eval')
    ensure_dir(save_path)

    filename = f"sghmc_jax_{args.n_hidden}h{args.n_units}_id{args.id}.csv"
    out_csv = os.path.join(save_path, filename)

    header = ['lr', 'prior_var', 'noise_var',
              'train_rmse', 'val_rmse', 'test_rmse', 'train_nll', 'val_nll', 'test_nll', 'elapsed_t']

    total_rows = []

    if args.pre_learn == 'cv':

        # Parse grids or fall back to singletons
        def parse_grid(s, fallback):
            if s.strip() == '':
                return [fallback]
            return [float(x) for x in s.split(',') if x]

        lrs = parse_grid(args.grid_lrs, args.lr)
        if args.prior_var is None:
            pvars = [1000.0, 50.0, 1.0]
        else:
            pvars = parse_grid(args.grid_prior_vars, args.prior_var)
        if args.noise_var is None:
            nvars = [1.0, 0.5, 0.1, 0.01]
        else:
            nvars = parse_grid(args.grid_noise_vars, args.noise_var)

        for lr, prior_var, noise_var in product(lrs, pvars, nvars):

            print(f"lr = {lr}, prior_var = {prior_var}, noise_var = {noise_var}")

            # Shadow args with current hyper-params
            run_args = argparse.Namespace(**vars(args))
            run_args.lr = float(lr)
            run_args.prior_var = float(prior_var)
            run_args.noise_var = float(noise_var)
            row = run_fold(run_args, fold_ids[0])
            total_rows.append(row)

    else:
        # opt-mode: single run
        row = run_fold(args, fold_ids[0])
        total_rows.append(row)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(total_rows)

    print("実験終了！")


if __name__ == '__main__':
    main()