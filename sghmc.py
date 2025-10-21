import argparse
import csv
import math
import multiprocessing
import os
import shutil
import time
import random
from functools import partial
from itertools import product
from types import SimpleNamespace

import numpy as np
import torch

import commons
from commons import LR, NOISE_VAR, PRIOR_VAR
from extended_models import getModel
from optbnn.bnn.likelihoods import LikGaussian
from optbnn.bnn.priors import FixedGaussianPrior, OptimGaussianPrior
from optbnn.metrics import uncertainty as uncertainty_metrics
from optbnn.sgmcmc_bayes_net.regression_net import RegressionNet
from optbnn.utils import util
from utilities.util import (
    ensure_dir,
    ess_rhat_func,
    from_torch_to_numpy,
    load_uci_data,
    round_func,
    show_data_ratios,
)


def sghmc(id, save_dir_anchor, tmp_id, lr, prior_var, noise_var, **specs):
    
    s = SimpleNamespace(**specs)
    assert s.train_eval == 'train'

    # Device Setup
    if s.device == "cpu":
        device = torch.device("cpu")
        n_gpu_use = None
    else:
        device = torch.device(f"cuda:{s.n_gpu_use}")
        torch.cuda.set_device(device.index)
        n_gpu_use = s.n_gpu_use
        print("USE GPU", torch.cuda.current_device())

    hidden_dims = [s.n_units] * s.n_hidden
    print(f"Loading split {id} of {s.dataset}")

    # Load data on CPU
    X_train_t, y_train_t = load_uci_data('train', id=id, name=s.dataset, device=torch.device('cpu'))
    X_val_t, y_val_t = load_uci_data('val', id=id, name=s.dataset, device=torch.device('cpu'))
    X_test_t, y_test_t = load_uci_data('test', id=id, name=s.dataset, device=torch.device('cpu'))

    show_data_ratios(y_train_t, y_val_t, y_test_t)

    X_train, y_train = from_torch_to_numpy(X_train_t, y_train_t)
    X_val,   y_val   = from_torch_to_numpy(X_val_t,   y_val_t)
    X_test,  y_test  = from_torch_to_numpy(X_test_t,  y_test_t)

    input_dim, output_dim = int(X_train.shape[-1]), 1

    # Sampler configs
    sampling_configs = {
        "batch_size": int(getattr(s, "batch", 32)),
        "num_samples": int(getattr(s, "num_samples", 40)),
        "n_discarded": int(getattr(s, "n_discarded", 10)),
        "num_burn_in_steps": int(getattr(s, "num_burn_in_steps", 2000)),
        "keep_every": int(getattr(s, "keep_every", 2000)),
        "lr": float(lr),
        "num_chains": int(getattr(s, "num_chains", 4)),
        "mdecay": float(getattr(s, "mdecay", 1e-2)),
        "print_every_n_samples": int(getattr(s, "print_every_n_samples", 5)),
    }

    # Build network
    net = getModel(
        save_dir_anchor,
        id,
        task="regression",
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        s=s
    )
    likelihood = LikGaussian(noise_var)

    if s.pre_learn == 'opt':
        # If you have an OptBNN pre-learned ckpt for the prior, specify here.
        ckpt_path = f'./results/prelearn_pvar/{s.dataset}/ckpts/it-200_nvar{noise_var}_{s.n_hidden}h{s.n_units}_id{id}.ckpt'
        prior = OptimGaussianPrior(ckpt_path, device)
    elif s.pre_learn == 'cv':
        prior_std = math.sqrt(prior_var)
        prior = FixedGaussianPrior(std=prior_std, device=device)
    else:
        raise ValueError(f"Unknown pre_learn mode: {s.pre_learn}")

    # RegressionNet will handle normalization internally
    bayes_net = RegressionNet(
        net,
        likelihood,
        prior,
        tmp_id,
        n_gpu=n_gpu_use,
        normalize_input=True,
        normalize_output=True
    )

    # Sampling
    s_time = time.perf_counter()
    bayes_net.sample_multi_chains(X_train, y_train, **sampling_configs)
    elapsed_t = int(time.perf_counter() - s_time)

    # Collect samples for ESS/R-hat (last-layer weights)
    sampled_models = bayes_net.sampled_weights
    print("len(sampled_models) = ", len(sampled_models))
    expected_used = sampling_configs['num_chains'] * (sampling_configs['num_samples'] - sampling_configs['n_discarded'])
    assert len(sampled_models) == expected_used, f"Unexpected samples: got {len(sampled_models)}, expected {expected_used}"

    sampled_lastW = [t[-2] for t in sampled_models]
    sampled_last_W = np.hstack(sampled_lastW).T

    # Evaluation (train/val/test)
    pred_mean, pred_var = bayes_net.predict(X_train)
    train_rmse = uncertainty_metrics.rmse(pred_mean, y_train)
    train_nll  = uncertainty_metrics.gaussian_nll(y_train, pred_mean, pred_var)

    pred_mean, pred_var = bayes_net.predict(X_val)
    val_rmse = uncertainty_metrics.rmse(pred_mean, y_val)
    val_nll  = uncertainty_metrics.gaussian_nll(y_val, pred_mean, pred_var)

    pred_mean, pred_var, _, raw_preds = bayes_net.predict(X_test, return_raw=True, return_all=True)
    test_rmse = uncertainty_metrics.rmse(pred_mean, y_test)
    test_nll  = uncertainty_metrics.gaussian_nll(y_test, pred_mean, pred_var)

    print("run evaluation of ESS and R_hat")

    # ESS/R-hat with last-layer weights
    mean_ess, min_ess, mean_rhat, max_rhat = ess_rhat_func(sampled_last_W, sampling_configs['num_chains'])

    # ESS/R-hat with predictive distribution (raw predictions)
    mean_ess_raw, min_ess_raw, mean_rhat_raw, max_rhat_raw = ess_rhat_func(raw_preds, sampling_configs['num_chains'])

    print(f"> RMSE = {test_rmse:.4f} | NLL = {test_nll:.4f}")

    row = [lr, prior_var, noise_var,
           train_rmse, val_rmse, test_rmse, train_nll, val_nll, test_nll,
           mean_ess, min_ess, mean_rhat, max_rhat,
           mean_ess_raw, min_ess_raw, mean_rhat_raw, max_rhat_raw, elapsed_t]

    return round_func(row)


def main(id, **specs):

    s = SimpleNamespace(**specs)

    ############################################################################################################

    # manage result path
    SAVE_DIR = f'./results/{s.method}/{s.dataset}/{s.pre_learn}'

    if s.method == 'sghmc':
        filename = f'sghmc_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'sghmc_ext':
        filename = f'sghmc_ext_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'abs_fix':
        filename = f'{s.max_min}_abs_fix_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'layer_fix':
        filename = f'layer_fix{s.num_fix_layer}_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'row_fix':
        filename = f'{s.max_min}_row_fix_{s.n_hidden}h{s.n_units}_id{id}.csv'
    elif s.method == 'sharma_fix':
        filename = f'sharma_fix_{s.n_hidden}h{s.n_units}_id{id}.csv'
    else:
        assert(False)

    # ensure if save path exists, else create
    ensure_dir(SAVE_DIR)
    out_csv = os.path.join(SAVE_DIR, filename)

    ############################################################################################################

    # Anchor path (used by getModel internally if needed, keep consistent with your repo)
    map_csv_anchor = f'./results/map_new/{s.dataset}/{s.pre_learn}/map_new_{s.n_hidden}h{s.n_units}_id{id}.csv'

    # Temporary directory identifier (cleaned up at the end)
    tmp_id = f"tmp_{s.dataset}_{id}_{s.min_max}_{s.method}_{s.n_hidden}_{s.n_units}_{s.num_fix_layer}_{s.pre_learn}_{s.train_eval}"
    print("tmp_file_identifier = ", tmp_id)

    # Sweep (cv only here)
    if s.pre_learn != 'cv':
        raise NotImplementedError("pre_learn='opt' path is disabled here (to keep dependencies intact).")

    header = ['lr', 'prior_var', 'noise_var',
              'train_rmse', 'val_rmse', 'test_rmse', 'train_nll', 'val_nll', 'test_nll',
              'mean_ess', 'min_ess', 'mean_rhat', 'max_rhat',
              'mean_ess_raw', 'min_ess_raw', 'mean_rhat_raw', 'max_rhat_raw', 'elapsed_t']

    total_rows = []

    print("ALL_LR = ", LR)
    print("ALL_PRIOR_VAR = ", PRIOR_VAR)
    print("ALL_NOISE_VAR = ", NOISE_VAR)

    for lr, prior_var, noise_var in product(LR, PRIOR_VAR, NOISE_VAR):
        print(f'lr = {lr}, prior_var = {prior_var}, noise_var = {noise_var}')
        row = sghmc(id, map_csv_anchor, tmp_id, lr=lr, prior_var=prior_var, noise_var=noise_var, **specs)
        total_rows.append(row)

    # Write CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(total_rows)

    # Cleanup temporary directory
    tmp_dir = f'./{tmp_id}'
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    return


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    parser = argparse.ArgumentParser()

    # Dataset Configuration
    parser.add_argument('--dataset', type=str, default='yacht', help="Dataset name (default: 'yacht')")
    parser.add_argument('--id', type=int, default=1, help="Fold ID specifier (default: 1)")

    # Model Configuration
    parser.add_argument('--act_fn', type=str, choices=['relu', 'tanh'], default='relu', help="Activation function for the MLP (default: 'relu')")
    parser.add_argument('--n_units', type=int, default=100, help="Number of hidden units per layer (default: 100)")
    parser.add_argument('--n_hidden', type=int, default=2, help="Number of hidden layers (default: 2)")
    parser.add_argument('--fix_bias', type=str, choices=['t', 'f'], default='f', help="Whether to fix bias terms: 't' (true) or 'f' (false, default)")
    parser.add_argument('--num_fix_layer', type=int, default=1, help="Number of layers to fix when using fixed-bias methods (default: 1)")

    # Training / Evaluation Setup
    parser.add_argument('--train_eval', type=str, choices=['train', 'eval'], default='train', help="Execution mode: 'train' (default) or 'eval'")
    parser.add_argument('--pre_learn', type=str, choices=['opt', 'cv'], default='cv', help="Pre-learning method: 'cv' (cross-validation) or 'opt' (OptBNN)")

    # Optimization Method Selection
    parser.add_argument('--method', type=str, choices=['sghmc', 'sghmc_ext', 'fix', 'abs_fix', 'layer_fix', 'row_fix', 'sharma_fix'], default='sghmc', help="SGHMC variant or bias-fix method (default: 'sghmc')")
    parser.add_argument('--min_max', type=str, choices=['min', 'max'], default='max', help="Optimization criterion: 'min' or 'max' (default: 'max')")

    # Device Configuration
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help="Device to run the experiment (default: 'cpu')")

    # Seeds & Execution Control
    parser.add_argument('--seed', type=int, default=77210, help="Random seed (torch + numpy + python)")
    parser.add_argument('--mp', action='store_true', help="Run folds in parallel (default: sequential)")

    # Performance Tuning
    parser.add_argument('--batch', type=int, default=32, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers (0 = no multiprocessing)")
    parser.add_argument('--pin_memory', action='store_true', help="Enable pin_memory for DataLoader")

    # ---- SGHMC Hyperparams ----
    # step_size          = float(lr)            # η: step size
    # friction           = 1e-2                 # α: friction
    # burn_in_steps      = 500                  # burn-in steps (not saved)
    # sample_interval    = 100                  # thinning interval
    # max_samples        = 40                   # number of posterior samples to save
    # scale_likelihood   = 1.0 / (2.0 * 1.0)    # training likelihood scale

    # Sampler Knobs
    parser.add_argument('--num_chains', type=int, default=4, help="number of SGHMC chains")
    parser.add_argument('--num_samples', type=int, default=40, help="kept samples per chain")
    parser.add_argument('--n_discarded', type=int, default=10, help="discarded samples per chain")
    parser.add_argument('--num_burn_in_steps', type=int, default=500, help="burn-in steps per chain")
    parser.add_argument('--keep_every', type=int, default=100, help="thinning interval")
    parser.add_argument('--mdecay', type=float, default=1e-2, help="momentum decay (friction)")
    parser.add_argument('--print_every_n_samples', type=int, default=5, help="progress print interval")

    args = parser.parse_args()

    # Dataset-specific defaults for n_units
    if args.n_units is None:
        n_units = 200 if args.dataset.startswith("protein") else 100
    else:
        n_units = args.n_units

    print("foldId_specifier = ", args.id)
    all_fold_ids = commons.get_all_fold_ids(args.dataset, args.id)
    print("all_fold_ids = ", all_fold_ids)

    # Pick GPU index if available
    if args.device == "cuda":
        n_gpu_use = commons.get_most_freemem_gpu()
    else:
        n_gpu_use = None

    # Fixed specs passed to main/sghmc
    fixed_specs = dict(

        dataset=args.dataset, train_eval=args.train_eval,
        n_hidden=args.n_hidden, n_units=n_units,
        act_fn=args.act_fn, device=args.device, pre_learn=args.pre_learn,
        fix_bias=(args.fix_bias == 't'),
        min_max=args.min_max, method=args.method,
        num_fix_layer=args.num_fix_layer,
        n_gpu_use=n_gpu_use,
        seed=args.seed,

        # sampler configs
        batch=args.batch,
        num_chains=args.num_chains,
        num_samples=args.num_samples,
        n_discarded=args.n_discarded,
        num_burn_in_steps=args.num_burn_in_steps,
        keep_every=args.keep_every,
        mdecay=args.mdecay,
        print_every_n_samples=args.print_every_n_samples,
    )

    # Default: run folds sequentially; reseed at the beginning of each fold
    if not args.mp:

        for _id in all_fold_ids:

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            if args.device == "cuda":
                torch.cuda.manual_seed_all(args.seed)

            print(f"\n==== START FOLD {_id} (seed={args.seed}) ====")
            main(_id, **fixed_specs)
            print(f"==== END   FOLD {_id} ====\n")

    else:
        with multiprocessing.Pool(processes=all_fold_ids.shape[0]) as pool:
            pool.map(partial(main, **fixed_specs), all_fold_ids)

    print("実験終了！")