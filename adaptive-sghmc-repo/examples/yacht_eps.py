from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive_sghmc.run_fold import run_yacht_fold


def main() -> None:
    result = run_yacht_fold(
        method="eps",
        root="data",
        fold_id=1,
        act_fn="leaky_relu",
        num_hidden_layers=1,
        num_hidden_units=20,
        prior_var=50.0,
        likelihood_var=0.2,
        seed=77210,
        num_chains=4,
        num_samples=20,
        num_discarded=5,
        burn_in=50,
        keep_every=10,
        print_every=0,
        inner_steps=10,
        reset_prob=0.05,
        grad_noise_var=0.0,
        epsilon=2e-3,
        mdecay=0.08,
        warm_up_frac=0.5,
        ema_beta=0.999,
        batch_size=32,
    )
    print(result["val_metrics"])
    print(result["test_metrics"])
    print(result["diagnostics"])


if __name__ == "__main__":
    main()
