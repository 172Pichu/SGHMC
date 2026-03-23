from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import numpy as np
from matplotlib import pyplot as plt

from adaptive_sghmc.distributions import density_fig3_xy, covariance_matrix

def finite_only(samples: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(samples, dtype=np.float64).reshape(-1)
    m = np.isfinite(a)
    bad = int((~m).sum())
    if bad > 0:
        print(f"[WARN] {name}: filtered {bad}/{a.size} non-finite samples (NaN/Inf).")
    return a[m]

def kde_1d_gaussian(args, samples: np.ndarray, bw: float = None) -> np.ndarray:
    plot_domain = np.linspace(-args.plot_max, args.plot_max, int(args.plot_grids_per_unit * args.plot_max * 2.0) + 1)
    samples = np.asarray(samples, dtype=np.float64).ravel()[np.isfinite(samples)]
    n = samples.size
    if n == 0:
        return np.zeros_like(plot_domain, dtype=np.float64)
    std = np.std(samples, ddof=1) if n > 1 else 1.0
    if bw is None:
        bw = 1.06 * std * (n ** (-1.0 / 5.0))
        bw = max(bw, 1e-3)
    else:
        bw = float(bw)
        bw = max(bw, 1e-6)
    z = (plot_domain[:, None] - samples[None, :]) / bw
    dens = np.exp(-0.5 * z * z).mean(axis=1) / (bw * np.sqrt(2.0 * np.pi))
    return dens

def fig1_bins(true_density, args, **all_samples):

    plot_domain = np.linspace(-args.plot_max, args.plot_max, int(args.plot_grids_per_unit * args.plot_max * 2.0) + 1)
    
    plt.figure(figsize=(8.0, 4.5), dpi=100)
    plt.plot(plot_domain, true_density, linewidth=2.0, label="true_density")

    for label, samples in all_samples.items():
        if samples is not None:
            plt.hist(samples, bins=args.bins, density=True, alpha=0.30, label=label)

    plt.title(f"Fig1-like (toy double-well):")
    plt.xlabel(r"$\theta$")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    os.makedirs("./result", exist_ok=True)
    plt.savefig("./result/fig1_like_toy_double_well_bins.png", dpi=200)
    plt.close()

def fig1_curves(true_density, args, **curves):

    plot_domain = np.linspace(-args.plot_max, args.plot_max, int(args.plot_grids_per_unit * args.plot_max * 2.0) + 1)

    plt.figure(figsize=(8.0, 4.5), dpi=100)
    plt.plot(plot_domain, true_density, linewidth=2.0, label="true_density")

    for label, density in curves.items():
        if density is not None:
            plt.plot(plot_domain, density, linewidth=1.2, label=label)

    plt.title(f"Fig1-like (toy double-well):")
    plt.xlabel(r"$\theta$")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    save_path = getattr(args, "output", "result/fig1_like_toy_double_well_curves.png")
    os.makedirs(Path(save_path).parent or Path("result"), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def as_xy(samples: np.ndarray, name: str) -> np.ndarray:
    """
    Normalize samples to shape (N, 2) and filter non-finite rows.
    Accepts (N, 2) or (2, N).
    """
    if samples is None:
        return None
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D array, got shape={a.shape}")
    if a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    if a.shape[1] != 2:
        raise ValueError(f"{name} must have 2 columns (x, y), got shape={a.shape}")
    m = np.isfinite(a).all(axis=1)
    bad = int((~m).sum())
    if bad > 0:
        print(f"[WARN] {name}: filtered {bad}/{a.shape[0]} non-finite sample rows (NaN/Inf).")
    return a[m]

def fig3_surface_density(args, *, surface_density: np.ndarray | None = None, density_fn=None,
                         samples_torch: np.ndarray | None = None, samples_blackjax: np.ndarray | None = None,
                         samples_other: dict[str, np.ndarray] | None = None,
                         scatter_max_points: int = 5000, surface_alpha: float = 0.85, scatter_alpha: float = 0.35):
    
    """
    Plot Fig3-like 2D density surface and optionally overlay sampler samples as scatter points.

    You can provide either:
      - surface_density: precomputed Z grid (shape [G, G])
      - density_fn: callable taking (N, 2) array and returning density (N, ) or logdensity (N, ) depending on your fn.
        (Assumed density. If your fn returns log density, exp it outside before passing.)

    Sample inputs:
      - samples_torch / samples_blackjax: arrays shaped (N, 2) or (2, N)
      - samples_other: dict of {label: samples_array}

    Notes:
      - Uses args.plot_max and args.plot_grids_per_unit for grid.
    """

    plot_domain = np.linspace(-args.plot_max, args.plot_max, int(args.plot_grids_per_unit * args.plot_max * 2.0) + 1)
    X, Y = np.meshgrid(plot_domain, plot_domain, indexing="xy")
    
    if surface_density is not None:
        Z = np.asarray(surface_density, dtype=np.float64)
        if Z.shape != X.shape:
            raise ValueError(f"surface_density shape {Z.shape} != grid shape {X.shape}")
    elif density_fn is not None:
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        z = density_fn(xy)
        z = np.asarray(z, dtype=np.float64).reshape(X.shape)
        Z = z
    else:
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        z = density_fig3_xy(xy, a=args.a, b=args.b, phi=args.phi)
        Z = np.asarray(z, dtype=np.float64).reshape(X.shape)

    groups: list[tuple[str, np.ndarray]] = []
    st = as_xy(samples_torch, "samples_torch")
    sb = as_xy(samples_blackjax, "samples_blackjax")
    if st is not None:
        groups.append(("torch_variant", st))
    if sb is not None:
        groups.append(("BlackJax", sb))
    if samples_other:
        for k, v in samples_other.items():
            vv = as_xy(v, f"samples_other[{k}]")
            if vv is not None:
                groups.append((k, vv))

    fig = plt.figure(figsize=(8.0, 6.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=surface_alpha)

    for label, pts in groups:
        if pts.shape[0] > scatter_max_points:
            idx = np.random.choice(pts.shape[0], size=scatter_max_points, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], np.zeros((pts.shape[0],), dtype=np.float64),
                   s=6, alpha=scatter_alpha, label=label)

    title = f"Fig3 density surface (a={args.a}, b={args.b}, phi={args.phi:.3f})"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("density")

    if groups:
        ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs("./result", exist_ok=True)
    plt.savefig("./result/fig3_like_toy_bivariate_elliptical_Gaussian_surface.png", dpi=200)
    plt.close()

def finite_rows_xy(samples: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {a.shape}")
    if a.shape[0] == 2 and a.shape[1] != 2:
        a = a.T
    if a.shape[1] != 2:
        raise ValueError(f"{name} must be (N,2) or (2,N), got {a.shape}")
    m = np.isfinite(a).all(axis=1)
    bad = int((~m).sum())
    if bad:
        print(f"[WARN] {name}: filtered {bad}/{a.shape[0]} non-finite rows.")
    return a[m]

def cov_mae(S_hat: np.ndarray, S_true: np.ndarray) -> float:
    return float(np.mean(np.abs(S_hat - S_true)))

def autocorr_1d_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.array([1.0], dtype=np.float64)
    x = x - x.mean()
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=m)
    acf = np.fft.irfft(f * np.conjugate(f), n=m)[:n]
    acf = acf / acf[0]
    return acf

def iat_initial_positive_sequence(x: np.ndarray, max_lag: int | None = None) -> float:
    """
    IAT estimator using Geyer's initial positive sequence on ACF:
    tau = 1 + 2 * sum_{k>=1} rho_k, but stop when rho_{2k-1}+rho_{2k} <= 0
    """
    acf = autocorr_1d_fft(x)
    n = acf.size
    if max_lag is None:
        max_lag = min(n - 1, 2000)
    max_lag = int(max_lag)
    s = 0.0
    k = 1
    while (2*k) <= max_lag:
        pair = float(acf[2*k - 1] + acf[2*k])
        if pair <= 0.0:
            break
        s += pair
        k += 1
    tau = 1.0 + 2.0 * s
    return float(max(tau, 1.0))

def iat_multidim_mean(samples_xy: np.ndarray) -> float:
    x = samples_xy[:, 0]
    y = samples_xy[:, 1]
    return 0.5 * (iat_initial_positive_sequence(x) + iat_initial_positive_sequence(y))

def evaluate_sampler_xy(samples_xy: np.ndarray, *, a: float, b: float, phi: float) -> tuple[float, float]:
    S_true, _ = covariance_matrix(a=a, b=b, phi=phi)
    S_hat = np.cov(samples_xy.T, ddof=1)
    err = cov_mae(S_hat, S_true)
    tau = iat_multidim_mean(samples_xy)
    return err, tau

def plot_gaussian_ellipses(ax, *, a: float, b: float, phi: float, levels=(0.1, 0.3, 0.6)):
    L = 3.5 * max(a, b)
    grid = np.linspace(-L, L, 200)
    X, Y = np.meshgrid(grid, grid, indexing="xy")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = density_fig3_xy(xy, a=a, b=b, phi=phi).reshape(X.shape)
    zmax = float(Z.max())
    lv = [zmax * float(t) for t in levels]
    ax.contour(X, Y, Z, levels=lv, linewidths=1.0)

def draw_fig3_left(ax, *, left_series: dict[str, list[tuple[float, float, str]]]):
    """
    Left panel: covariance error vs autocorrelation time.

    left_series: label -> list of (tau, err, step_label)
    """
    for label, pts in left_series.items():
        taus = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        ax.plot(taus, errs, marker="o", linewidth=1.2, label=label)
        for (tau, err, step_label) in pts:
            ax.annotate(step_label, (tau, err), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Autocorrelation time (IAT)")
    ax.set_ylabel("Average Absolute Error of Sample Covariance")
    ax.set_title("Covariance error vs autocorrelation")
    ax.legend()

def draw_fig3_right(ax, *, a: float, b: float, phi: float, 
                    scatter_inputs: dict[str, np.ndarray], scatter_first_n: int = 50, ellipse_levels=(0.1, 0.3, 0.6)):
    """
    Right panel: density ellipses + first N samples for each sampler.
    """
    plot_gaussian_ellipses(ax, a=a, b=b, phi=phi, levels=ellipse_levels)

    for label, samp in scatter_inputs.items():
        sxy = finite_rows_xy(samp, f"scatter:{label}")
        sxy = sxy[: int(scatter_first_n)]
        ax.scatter(sxy[:, 0], sxy[:, 1], s=18, alpha=0.8, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"First {scatter_first_n} samples")
    ax.legend()

def fig3_like_compare(args, *, metrics_inputs: dict[str, list[tuple[str, np.ndarray]]], scatter_inputs: dict[str, np.ndarray],
                      num_scatter_first: int = 50, ellipse_levels=(0.1, 0.3, 0.6)):
    
    """
    Assemble Fig3-like 1x2 plot:
      - Left: covariance estimation error vs IAT across step settings
      - Right: first N samples over density ellipses
    """

    a = args.a
    b = args.b
    phi = args.phi
    
    left_series: dict[str, list[tuple[float, float, str]]] = {}
    for label, runs in metrics_inputs.items():
        pts: list[tuple[float, float, str]] = []
        for step_label, samp in runs:
            sxy = finite_rows_xy(samp, f"{label}:{step_label}")
            err, tau = evaluate_sampler_xy(sxy, a=a, b=b, phi=phi)
            pts.append((tau, err, step_label))
        pts.sort(key=lambda t: t[0])
        left_series[label] = pts

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=120)

    draw_fig3_left(axL, left_series=left_series)
    draw_fig3_right(axR, a=a, b=b, phi=phi, scatter_inputs=scatter_inputs, scatter_first_n=num_scatter_first, ellipse_levels=ellipse_levels)

    fig.suptitle(f"Fig3-like: bivariate Gaussian (a={a}, b={b}, phi={phi:.3f})", y=1.02)
    fig.tight_layout()

    save_path = getattr(args, "output", "result/fig3_like_compare.png")
    os.makedirs(Path(save_path).parent or Path("result"), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return left_series