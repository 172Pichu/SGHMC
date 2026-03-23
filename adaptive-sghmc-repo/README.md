# Adaptive SGHMC repository

This repository packages a practical **epsilon-form adaptive SGHMC** sampler,
small **toy experiments**, and a **Yacht regression benchmark** pipeline.

The main sampler is documented in detail because the intended GitHub use case is
not only to run experiments, but also to make the provenance of every design
choice explicit.

## What is implemented here?

### Core sampler
The custom sampler in `adaptive_sghmc/samplers.py` is a practical diagonal
implementation inspired by **Eq. (10)** of:

- Springenberg et al., *Bayesian Optimization with Robust Bayesian Neural Networks* (NIPS 2016)

It combines:

- epsilon-form SGHMC updates,
- diagonal scale adaptation via `V_hat`,
- burn-in-only adaptation followed by fixed preconditioning,
- optional probabilistic momentum refresh,
- optional artificial gradient noise for toy experiments.

### Why the 2014 SGHMC paper is still central
The repository also follows the experimental intuition of:

- Chen, Fox, and Guestrin, *Stochastic Gradient Hamiltonian Monte Carlo* (ICML 2014)

especially for:

- the 1D double-well toy setup from Fig. 1,
- the correlated 2D Gaussian toy setup from Fig. 3,
- the `grad_noise_var = 4.0` noisy-gradient setting,
- the "refresh momentum every 50 steps" toy configuration,
- the eta-form interpretation connected to SGD with momentum.

## Repository layout

```text
adaptive-sghmc-repo/
├─ adaptive_sghmc/
│  ├─ cli.py
│  ├─ distributions.py
│  ├─ mcmc.py
│  ├─ objective.py
│  ├─ nuts.py
│  ├─ run_fold.py
│  ├─ samplers.py
│  ├─ runners/
│  ├─ third_party/
│  └─ utils/
├─ data/
│  └─ yacht/
├─ docs/
│  └─ sampler_design.md
├─ examples/
│  ├─ toy_fig1.py
│  ├─ toy_fig3.py
│  ├─ toy_diagnosis.py
│  ├─ toy_metrics.py
│  ├─ toy_plot_utils.py
│  └─ yacht_eps.py
├─ pyproject.toml
├─ requirements.txt
└─ requirements-optional.txt
```

## Installation

Base dependencies:

```bash
pip install -r requirements.txt
```

Optional baseline dependencies for NUTS / BlackJAX:

```bash
pip install -r requirements-optional.txt
```

Editable install from the repository root:

```bash
pip install -e .
```

## Quick start

Run the Yacht benchmark with the custom epsilon-form sampler:

```bash
python -m adaptive_sghmc.cli --method eps --root data --id 1
```

Run the 1D toy experiment:

```bash
python examples/toy_fig1.py
```

Run the 2D toy experiment:

```bash
python examples/toy_fig3.py
```

## Parameter vocabulary used in this repository

To reduce ambiguity across implementations, the repository uses three explicit
notions of "step":

- **atomic step**: one indivisible SGHMC update
- **inner step**: one atomic update inside an outer iteration
- **outer step**: one bookkeeping step used for refresh, thinning, logging,
  burn-in accounting, and sample collection

This vocabulary was introduced here as a reproducibility aid when comparing
custom SGHMC code against frameworks that either force a refresh schedule or do
not expose one directly.

## What is worth putting on GitHub?

For a public research-oriented repository, the following pieces are worth
keeping together:

- the core sampler,
- target-distribution code for toy experiments,
- minimal plotting utilities for the toy figures,
- one real-data benchmark pipeline,
- one or two runnable example commands,
- a short design note explaining which parts are paper-derived and which parts
  are engineering abstractions.

That is the layout used here. Result folders, temporary checkpoints, and old
experiment dumps are intentionally left out.

## Notes on baselines

- `adaptive_sghmc/runners/run_optbnn.py` wraps the optimizer-style adaptive
  SGHMC baseline used for comparison.
- `adaptive_sghmc/runners/run_nuts.py` wraps a BlackJAX NUTS baseline.
- `adaptive_sghmc/third_party/adaptive_sghmc.py` is included as the user-provided
  third-party optimizer snapshot. Verify upstream licensing details before
  publishing the repository broadly.

## Recommended reading order

1. `adaptive_sghmc/samplers.py`
2. `docs/sampler_design.md`
3. `examples/toy_fig1.py`
4. `examples/yacht_eps.py`
