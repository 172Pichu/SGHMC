from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import numpy as np

from adaptive_sghmc.samplers import samples_eps
from toy_diagnosis import finite_chains, diagnose_with_samples
from toy_plot_utils import fig3_like_compare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recreate the 2D correlated-Gaussian toy experiment.")
    parser.add_argument("--seed", type=int, default=77210)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--discarded", type=int, default=0)
    parser.add_argument("--burn_in", type=int, default=500)
    parser.add_argument("--keep_every", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=0)
    parser.add_argument("--inner_steps", type=int, default=10)
    parser.add_argument("--reset_prob", type=float, default=0.05)
    parser.add_argument("--grad_nvar", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--mdecay", type=float, default=0.05)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=0.2)
    parser.add_argument("--phi", type=float, default=np.pi / 6.0)
    parser.add_argument("--scatters", type=int, default=50)
    parser.add_argument("--output", type=str, default="result/fig3_like_compare.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    eps_list = [args.eps * 0.5, args.eps, args.eps * 2.0]
    runs_eps = []
    scatter_eps_chains = None
    scatter_eps_samples = None

    for eps in eps_list:
        c_eps, s_eps = samples_eps(None, None, args.seed, 2, args.chains, args.samples, args.discarded,
                                   args.burn_in, args.keep_every, args.print_every, args.inner_steps,
                                   args.reset_prob, args.grad_nvar, eps, args.mdecay)
        label = f"eps={eps:g}"
        runs_eps.append((label, np.asarray(s_eps, dtype=np.float64)))
        if abs(eps - args.eps) < 1e-12:
            scatter_eps_chains = np.asarray(c_eps, dtype=np.float64)
            scatter_eps_samples = np.asarray(s_eps, dtype=np.float64)

    if scatter_eps_chains is not None:
        diagnose_with_samples("eps", finite_chains(scatter_eps_chains, "eps"), args)

    metrics_inputs = {"eps_variant": runs_eps}
    scatter_inputs = {"eps_variant": scatter_eps_samples}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig3_like_compare(args, metrics_inputs=metrics_inputs, scatter_inputs=scatter_inputs,
                      num_scatter_first=int(args.scatters), ellipse_levels=(0.1, 0.3, 0.6))
    print(f"saved figure to {args.output}")


if __name__ == "__main__":
    main()
