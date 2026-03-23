from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import numpy as np

from adaptive_sghmc.distributions import true_density_fig1
from adaptive_sghmc.samplers import samples_eps
from toy_diagnosis import finite_chains, diagnose_with_samples
from toy_plot_utils import finite_only, kde_1d_gaussian, fig1_curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recreate the 1D double-well toy experiment.")
    parser.add_argument("--seed", type=int, default=77210)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--discarded", type=int, default=0)
    parser.add_argument("--burn_in", type=int, default=500)
    parser.add_argument("--keep_every", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument("--inner_steps", type=int, default=50)
    parser.add_argument("--reset_prob", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--mdecay", type=float, default=0.05)
    parser.add_argument("--plot_max", type=float, default=2.0)
    parser.add_argument("--plot_grids_per_unit", type=int, default=500)
    parser.add_argument("--integral_grids_per_unit", type=int, default=2000)
    parser.add_argument("--integral_max", type=float, default=4.0)
    parser.add_argument("--output", type=str, default="result/fig1_like_toy_double_well_curves.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_domain, dens = true_density_fig1(
        num_integral_grids_per_unit=args.integral_grids_per_unit,
        integral_max=args.integral_max,
        num_plot_grids_per_unit=args.plot_grids_per_unit,
        plot_max=args.plot_max,
    )

    c_eps, s_eps = samples_eps(None, None, args.seed, 1, args.chains, args.samples, args.discarded,
                               args.burn_in, args.keep_every, args.print_every, args.inner_steps,
                               0.0, 0.0, args.eps, args.mdecay)

    c_eps_mr, s_eps_mr = samples_eps(None, None, args.seed, 1, args.chains, args.samples, args.discarded,
                                     args.burn_in, args.keep_every, args.print_every, args.inner_steps,
                                     1.0, 0.0, 0.03, args.mdecay)

    c_eps_gn, s_eps_gn = samples_eps(None, None, args.seed, 1, args.chains, args.samples, args.discarded,
                                     args.burn_in, args.keep_every, args.print_every, args.inner_steps,
                                     0.0, 4.0, args.eps, args.mdecay)

    c_eps_mr_gn, s_eps_mr_gn = samples_eps(None, None, args.seed, 1, args.chains, args.samples, args.discarded,
                                           args.burn_in, args.keep_every, args.print_every, args.inner_steps,
                                           1.0, 4.0, 0.03, args.mdecay)

    diagnose_with_samples("eps", finite_chains(c_eps, "eps"), args)
    diagnose_with_samples("eps with momentum_resetting", finite_chains(c_eps_mr, "eps"), args)
    diagnose_with_samples("eps with grad_noise", finite_chains(c_eps_gn, "eps with grad_noise"), args)
    diagnose_with_samples("eps with momentum_resetting & grad_noise", finite_chains(c_eps_mr_gn, "eps with grad_noise"), args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig1_curves(
        dens,
        args,
        eps=kde_1d_gaussian(args, finite_only(s_eps, "eps"), bw=0.2),
        eps_mr=kde_1d_gaussian(args, finite_only(s_eps_mr, "eps with momentum_resetting"), bw=0.2),
        eps_gn=kde_1d_gaussian(args, finite_only(s_eps_gn, "eps with grad_noise"), bw=0.2),
        eps_mr_gn=kde_1d_gaussian(args, finite_only(s_eps_mr_gn, "eps with momentum_resetting & grad_noise"), bw=0.2),
    )
    print(f"saved figure to {args.output}")


if __name__ == "__main__":
    main()
