from __future__ import annotations

import argparse
from pprint import pprint

from .run_fold import run_yacht_fold
from .utils.io import make_run_name, save_experiment_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Yacht experiments for the custom epsilon-form adaptive SGHMC sampler and baselines."
    )

    # Method selection
    parser.add_argument("--method", type=str, default="eps", choices=["eps", "nuts", "optbnn"],
                        help="Sampling backend to run.")

    # Data configuration
    parser.add_argument("--dataset", type=str, default="yacht", help="Currently only 'yacht' is supported.")
    parser.add_argument("--root", type=str, default="data", help="Dataset root directory.")
    parser.add_argument("--id", type=int, default=1, help="Fold id in [1, 10].")

    # Model configuration
    parser.add_argument("--act_fn", type=str, default="leaky_relu", choices=["relu", "leaky_relu", "sigmoid", "tanh"])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--units", type=int, default=20)

    # Posterior and likelihood
    parser.add_argument("--pvar", type=float, default=50.0, help="Gaussian prior variance for theta.")
    parser.add_argument("--lvar", type=float, default=0.2, help="Observation noise variance.")

    # Stochasticity
    parser.add_argument("--seed", type=int, default=77210)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Sampler settings
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--discarded", type=int, default=10)
    parser.add_argument("--burn_in", type=int, default=500)
    parser.add_argument("--keep_every", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--reset_prob", type=float, default=0.05)
    parser.add_argument("--gnvar", type=float, default=0.0, help="Artificial gradient-noise variance.")
    parser.add_argument("--eps", type=float, default=2e-3, help="Epsilon-form SGHMC step size.")
    parser.add_argument("--mdecay", type=float, default=0.08, help="Effective momentum decay.")
    parser.add_argument("--warm_up_frac", type=float, default=0.5)
    parser.add_argument("--ema_beta", type=float, default=0.999)

    # IO settings
    parser.add_argument("--save_dir", type=str, default="result")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset.lower() != "yacht":
        raise ValueError("This CLI currently supports only the predefined Yacht benchmark.")

    result = run_yacht_fold(
        args.method,
        args.root,
        args.id,
        args.act_fn,
        args.layers,
        args.units,
        args.pvar,
        args.lvar,
        args.seed,
        args.chains,
        args.samples,
        args.discarded,
        args.burn_in,
        args.keep_every,
        args.print_every,
        args.inner_steps,
        args.reset_prob,
        args.gnvar,
        args.eps,
        args.mdecay,
        warm_up_frac=args.warm_up_frac,
        ema_beta=args.ema_beta,
        batch_size=args.batch_size,
    )

    print("\n==== Experiment Summary ====")
    print(f"dataset = {result['dataset']}")
    print(f"fold_id = {result['fold_id']}")
    print(f"method = {result['method']}")
    print(f"grad_mode = {result['grad_mode']}")
    print(f"split_sizes = {result['split_sizes']}")
    print(f"model_hyper_params = {result['model_hyper_params']}")

    print("\nval_metrics:")
    pprint(result["val_metrics"])

    print("\ntest_metrics:")
    pprint(result["test_metrics"])

    print("\ndiagnostics:")
    pprint(result["diagnostics"])

    run_name = make_run_name(result["dataset"], result["method"], result["fold_id"], args.seed, args.layers, args.units)
    save_path = save_experiment_result(result, save_root=args.save_dir, run_name=run_name)
    print(f"\nsaved_to = {save_path}")


if __name__ == "__main__":
    main()
