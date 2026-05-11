"""Plotting functions for DGP concept and analysis tutorials.

Covers: dgp_concept.qmd (and later: dgp_dag_concept.qmd,
        dgp_quadrature_analysis.qmd, dgp_dsvi_analysis.qmd, ...).

The figures in this module fit a real two-layer DeepGaussianProcess on
a 1D step-function regression problem and reuse the same fitted model
across three of the four figures. The remaining figure (the composition
cartoon, plot_composition_cartoon) is a hand-constructed illustration
of stationary-function composition and does not use a fit.

A module-level cache keeps the fit cost to one Adam optimisation per
Quarto render. Combined with Quarto's `freeze: auto`, this means a
fresh fit only happens when the .qmd is edited.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Backend / dependency check
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment issue
    raise ImportError(
        "_dgp.py requires PyTorch (the DGP code path uses TorchBkd). "
        "Install torch to render the deep-GP tutorials."
    ) from exc


# ---------------------------------------------------------------------------
# Synthetic step-function dataset (shared across figures 1, 3, 4)
# ---------------------------------------------------------------------------

_STEP_DATA_SEED = 11
_STEP_NTRAIN = 30
_STEP_NOISE_STD = 0.05


def _step_function_dataset():
    """Return (X_train, y_train, x_grid, y_grid_truth) for the step demo.

    X_train: (1, N), y_train: (1, N), x_grid: (1, n_grid),
    y_grid_truth: (n_grid,) — the noise-free underlying step.
    """
    rng = np.random.default_rng(_STEP_DATA_SEED)
    X_train = np.sort(rng.uniform(-2.0, 2.0, _STEP_NTRAIN)).reshape(1, -1)
    y_true = np.where(X_train[0] > 0, 1.0, -1.0)
    y_train = (
        y_true + _STEP_NOISE_STD * rng.standard_normal(_STEP_NTRAIN)
    ).reshape(1, -1)
    x_grid = np.linspace(-2.5, 2.5, 250).reshape(1, -1)
    y_grid_truth = np.where(x_grid[0] > 0, 1.0, -1.0)
    return X_train, y_train, x_grid, y_grid_truth


# ---------------------------------------------------------------------------
# Real DGP fits — cached across figures
# ---------------------------------------------------------------------------

_FIT_CACHE = {}


def _matern_factory(nvars, bkd):
    from pyapprox.surrogates.kernels.matern import Matern52Kernel
    return Matern52Kernel(
        lenscale=[1.0] * nvars,
        lenscale_bounds=(0.05, 10.0),
        nvars=nvars,
        bkd=bkd,
    )


def _fit_single_layer_gp(X_train_np, y_train_np):
    """Fit a 1-layer DGP (== SVGP) on the step data. Returns the
    fitted DGP for prediction."""
    if "single_layer" in _FIT_CACHE:
        return _FIT_CACHE["single_layer"]

    from pyapprox.optimization.minimize.adam.adam_optimizer import (
        AdamOptimizer,
    )
    from pyapprox.surrogates.gaussianprocess.deep.builders import (
        build_single_fidelity_dgp,
    )
    from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
        TensorProductGHRule,
    )
    from pyapprox.surrogates.gaussianprocess.deep.propagator import (
        LayerPropagator,
    )
    from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
        DGPMaximumLikelihoodFitter,
    )
    from pyapprox.util.backends.torch import TorchBkd

    bkd = TorchBkd()
    X_train = bkd.array(X_train_np)
    y_train = bkd.array(y_train_np)

    # Single-layer DGP with GH-1 (degenerate) propagation.
    dgp = build_single_fidelity_dgp(
        n_layers=1,
        nvars=1,
        num_inducing=8,
        kernel_factory=_matern_factory,
        bkd=bkd,
        noise_std=_STEP_NOISE_STD,
        n_propagation=1,
        seed=42,
    )

    optimizer = AdamOptimizer(lr=1e-2, maxiter=600, verbosity=0)
    fitter = DGPMaximumLikelihoodFitter(
        bkd, optimizer=optimizer, n_propagation=1,
    )
    result = fitter.fit(dgp, {0: (X_train, y_train)})
    fitted = result.surrogate()
    _FIT_CACHE["single_layer"] = (fitted, bkd)
    return _FIT_CACHE["single_layer"]


def _fit_two_layer_dgp(X_train_np, y_train_np):
    """Fit a 2-layer DGP on the step data with GH propagation and an
    Adam → L-BFGS-B chained optimiser.

    The chain is the standard "warm up with stochastic-friendly Adam,
    refine with deterministic-friendly L-BFGS-B" pattern from GPflow
    and similar libraries. Adam handles the noisy early phase where
    the loss landscape is far from quadratic; L-BFGS-B uses curvature
    information to refine to a tightly converged optimum once Adam
    has reached the right basin. Deterministic GH propagation is
    required for L-BFGS-B's line search to behave correctly.

    Returns the fitted DGP and the backend.
    """
    if "two_layer" in _FIT_CACHE:
        return _FIT_CACHE["two_layer"]

    from pyapprox.optimization.minimize.adam.adam_optimizer import (
        AdamOptimizer,
    )
    from pyapprox.optimization.minimize.chained.chained_optimizer import (
        ChainedOptimizer,
    )
    from pyapprox.optimization.minimize.scipy.lbfgsb import (
        LBFGSBOptimizer,
    )
    from pyapprox.surrogates.gaussianprocess.deep.builders import (
        build_single_fidelity_dgp,
    )
    from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
        TensorProductGHRule,
    )
    from pyapprox.surrogates.gaussianprocess.deep.propagator import (
        LayerPropagator,
    )
    from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
        DGPMaximumLikelihoodFitter,
    )
    from pyapprox.util.backends.torch import TorchBkd

    bkd = TorchBkd()
    X_train = bkd.array(X_train_np)
    y_train = bkd.array(y_train_np)

    # Build the DGP with deterministic GH propagation. For a 2-layer
    # chain both nodes get a noise dimension, so TensorProductGHRule(order=5)
    # produces 5^2 = 25 chain paths.
    dgp = build_single_fidelity_dgp(
        n_layers=2,
        nvars=1,
        num_inducing=8,
        kernel_factory=_matern_factory,
        bkd=bkd,
        noise_std=_STEP_NOISE_STD,
        n_propagation=25,
        seed=42,
    )
    gh_rule = TensorProductGHRule(order=5)
    dgp.set_propagator(LayerPropagator(bkd, rule=gh_rule))

    # Adam (warm-up) → L-BFGS-B (refinement). Adam reaches the right
    # basin in ~500 iterations on this 30-point problem; L-BFGS-B
    # then converges in well under its 150-iteration cap once it has
    # a good starting point.
    chained = ChainedOptimizer(
        AdamOptimizer(lr=1e-2, maxiter=500, verbosity=0),
        LBFGSBOptimizer(maxiter=150, verbosity=0),
    )
    fitter = DGPMaximumLikelihoodFitter(
        bkd, optimizer=chained, n_propagation=25,
    )
    result = fitter.fit(dgp, {1: (X_train, y_train)})
    fitted = result.surrogate()
    _FIT_CACHE["two_layer"] = (fitted, bkd)
    return _FIT_CACHE["two_layer"]


def _predict_with_std(fitted_dgp, bkd, x_grid_np, n_propagation):
    """Predict mean and std at x_grid_np from a fitted DGP. Returns
    numpy arrays of shape (n_grid,)."""
    x_grid_t = bkd.array(x_grid_np)
    mean = fitted_dgp.predict(x_grid_t, n_propagation=n_propagation)
    std = fitted_dgp.predict_std(x_grid_t, n_propagation=n_propagation)
    return (
        bkd.to_numpy(mean).ravel(),
        bkd.to_numpy(std).ravel(),
    )


# ---------------------------------------------------------------------------
# Held-out test set + quality metrics (shared across figures 1 and the
# new calibration figure)
# ---------------------------------------------------------------------------

_TEST_DATA_SEED = 23
_NTEST = 200


def _step_function_testset():
    """Held-out test set for quality metrics.

    Uses a different seed than the training set so points don't
    coincide, but the same underlying step function. Returns
    X_test (1, N_test) and y_test_truth (N_test,) noise-free.
    """
    rng = np.random.default_rng(_TEST_DATA_SEED)
    X_test = np.sort(rng.uniform(-2.0, 2.0, _NTEST)).reshape(1, -1)
    y_test_truth = np.where(X_test[0] > 0, 1.0, -1.0)
    return X_test, y_test_truth


def _compute_test_metrics(
    fitted_dgp, bkd, X_test_np, y_test_truth, n_propagation,
):
    """Test-set RMSE and mean log-likelihood under a Gaussian
    predictive with the model's predicted mean and std.

    The mean log-likelihood treats each test point as a Gaussian
    with mean = predicted mean, variance = predicted variance.
    Higher = better calibration of both mean and uncertainty.
    """
    mean, std = _predict_with_std(
        fitted_dgp, bkd, X_test_np, n_propagation,
    )
    residual = y_test_truth - mean
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    var = std ** 2 + 1e-12
    mean_loglik = float(np.mean(
        -0.5 * np.log(2 * np.pi * var) - 0.5 * residual ** 2 / var
    ))
    return {"rmse": rmse, "mean_loglik": mean_loglik}


def _compute_coverage_curve(
    fitted_dgp, bkd, X_test_np, y_test_truth, n_propagation,
    nominal_levels,
):
    """Empirical coverage at a sequence of nominal credible-interval
    levels.

    For each nominal level alpha (e.g. 0.90), compute z = z-score
    such that a Gaussian has alpha probability mass within +/- z * std,
    then count test points where |truth - mean| < z * std. Returns
    an array of empirical coverages, same length as nominal_levels.
    """
    from scipy.stats import norm as _norm
    mean, std = _predict_with_std(
        fitted_dgp, bkd, X_test_np, n_propagation,
    )
    abs_resid = np.abs(y_test_truth - mean)
    coverages = []
    for alpha in nominal_levels:
        # Two-sided CI: z = Phi^{-1}((1+alpha)/2)
        z = float(_norm.ppf(0.5 * (1.0 + alpha)))
        within = abs_resid < z * std
        coverages.append(float(np.mean(within)))
    return np.asarray(coverages)


# ---------------------------------------------------------------------------
# Figure 1: step function comparison (single-layer SVGP vs 2-layer DGP)
# ---------------------------------------------------------------------------


def plot_step_function_comparison(axes):
    """dgp_concept.qmd -> fig-step-function-comparison

    Single-layer SVGP (left) vs two-layer DGP (right) on a 1D step
    function. Shows characteristic ringing of the stationary GP and
    how a 2-layer DGP avoids it.

    Both fits use the same data, same kernel family, same fitter.
    Only the depth differs.
    """
    from ._style import apply_style

    X_train, y_train, x_grid, y_grid_truth = _step_function_dataset()
    X_test, y_test_truth = _step_function_testset()

    fitted_1, bkd_1 = _fit_single_layer_gp(X_train, y_train)
    fitted_2, bkd_2 = _fit_two_layer_dgp(X_train, y_train)

    mean_1, std_1 = _predict_with_std(fitted_1, bkd_1, x_grid, n_propagation=1)
    mean_2, std_2 = _predict_with_std(fitted_2, bkd_2, x_grid, n_propagation=25)

    metrics_1 = _compute_test_metrics(
        fitted_1, bkd_1, X_test, y_test_truth, n_propagation=1,
    )
    metrics_2 = _compute_test_metrics(
        fitted_2, bkd_2, X_test, y_test_truth, n_propagation=25,
    )

    # ---- Single-layer panel ----
    ax = axes[0]
    ax.plot(
        x_grid[0], y_grid_truth, color="k", lw=1.0, ls="--", alpha=0.6,
        label="True step",
    )
    ax.fill_between(
        x_grid[0], mean_1 - 2 * std_1, mean_1 + 2 * std_1,
        color="#C0392B", alpha=0.18, label=r"Single GP $\pm 2\sigma$",
    )
    ax.plot(
        x_grid[0], mean_1, color="#C0392B", lw=1.8,
        label="Single GP mean",
    )
    ax.scatter(
        X_train[0], y_train[0], s=20, color="k", zorder=5,
        label="Data",
    )
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$f(x)$", fontsize=11)
    ax.set_title("Single-layer GP: ringing at the discontinuity",
                 fontsize=10)
    ax.text(
        0.04, 0.96,
        f"test RMSE = {metrics_1['rmse']:.3f}\n"
        f"mean log-lik = {metrics_1['mean_loglik']:.2f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.85,
                  boxstyle="round,pad=0.3"),
    )
    ax.legend(fontsize=8, loc="lower right")
    apply_style(ax)

    # ---- Two-layer DGP panel ----
    ax = axes[1]
    ax.plot(
        x_grid[0], y_grid_truth, color="k", lw=1.0, ls="--", alpha=0.6,
        label="True step",
    )
    ax.fill_between(
        x_grid[0], mean_2 - 2 * std_2, mean_2 + 2 * std_2,
        color="#117A65", alpha=0.18, label=r"DGP $\pm 2\sigma$",
    )
    ax.plot(
        x_grid[0], mean_2, color="#117A65", lw=1.8, label="DGP mean",
    )
    ax.scatter(
        X_train[0], y_train[0], s=20, color="k", zorder=5,
        label="Data",
    )
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_title("Two-layer DGP: sharper transition, honest extrapolation",
                 fontsize=10)
    ax.text(
        0.04, 0.96,
        f"test RMSE = {metrics_2['rmse']:.3f}\n"
        f"mean log-lik = {metrics_2['mean_loglik']:.2f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.85,
                  boxstyle="round,pad=0.3"),
    )
    ax.legend(fontsize=8, loc="lower right")
    apply_style(ax)


# ---------------------------------------------------------------------------
# Figure 1b: calibration of predictive intervals (single-layer vs DGP)
# ---------------------------------------------------------------------------


def plot_calibration_curves(ax):
    """dgp_concept.qmd -> fig-calibration

    Reliability diagram for the single-layer SVGP and the two-layer
    DGP on the step-function held-out test set.

    For each nominal credible-interval level alpha (e.g. 0.90), we
    compute the empirical fraction of test points where the truth
    falls inside mean +/- z_alpha * std. A perfectly calibrated
    model lies on the diagonal: nominal coverage matches empirical
    coverage. Above the diagonal = underconfident (intervals too
    wide). Below the diagonal = overconfident (intervals too narrow).

    For a step function, the single GP is expected to be visibly
    overconfident: its tight uncertainty bands around the ringing
    artifact don't actually cover the true value at the discontinuity.
    The DGP, with its honest step-aware uncertainty, should track
    the diagonal much more closely.
    """
    from ._style import apply_style

    X_train, y_train, _, _ = _step_function_dataset()
    X_test, y_test_truth = _step_function_testset()

    fitted_1, bkd_1 = _fit_single_layer_gp(X_train, y_train)
    fitted_2, bkd_2 = _fit_two_layer_dgp(X_train, y_train)

    nominal = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    cov_1 = _compute_coverage_curve(
        fitted_1, bkd_1, X_test, y_test_truth,
        n_propagation=1, nominal_levels=nominal,
    )
    cov_2 = _compute_coverage_curve(
        fitted_2, bkd_2, X_test, y_test_truth,
        n_propagation=25, nominal_levels=nominal,
    )

    # Reference diagonal
    ax.plot(
        [0, 1], [0, 1], color="k", lw=1.0, ls="--", alpha=0.6,
        label="Perfect calibration",
    )
    ax.plot(
        nominal, cov_1, marker="o", color="#C0392B", lw=1.8,
        markersize=6, label="Single GP",
    )
    ax.plot(
        nominal, cov_2, marker="s", color="#117A65", lw=1.8,
        markersize=6, label="Two-layer DGP",
    )

    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Nominal credible-interval coverage", fontsize=11)
    ax.set_ylabel("Empirical coverage on test set", fontsize=11)
    ax.set_title(
        "Reliability diagram: are the predictive bands honest?",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    apply_style(ax)


# ---------------------------------------------------------------------------
# Figure 2: composition cartoon (synthetic illustration only)
# ---------------------------------------------------------------------------


def plot_composition_cartoon(axes):
    """dgp_concept.qmd -> fig-composition-cartoon

    Three panels showing how f2 composed with f1 produces a
    non-stationary effect from two stationary ingredients.

    This is an illustration of the composition idea. It does not
    use any fitted model — both f1 and f2 are hand-chosen smooth
    functions whose composition is sharp.
    """
    from ._style import apply_style

    x = np.linspace(-2.5, 2.5, 300)

    # Layer 1: smooth nonlinear warp (sigmoid-like)
    f1 = 1.6 * np.tanh(2.5 * x)

    # Layer 2: a smooth function defined on the f1-output space
    h_grid = np.linspace(-2.5, 2.5, 300)
    f2_h = 0.8 * h_grid + 0.4 * np.sin(2.0 * h_grid)

    # Composition
    f12 = 0.8 * f1 + 0.4 * np.sin(2.0 * f1)

    # Panel 1: f1 as function of x
    ax = axes[0]
    ax.plot(x, f1, color="#2C7FB8", lw=2.2)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel(r"$f_1(x)$", fontsize=11)
    ax.set_title(r"Layer 1: $f_1: \mathbb{R} \to \mathbb{R}$",
                 fontsize=10)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.0, 2.0)
    ax.axhline(0, color="k", lw=0.6, alpha=0.3)
    ax.axvline(0, color="k", lw=0.6, alpha=0.3)
    apply_style(ax)

    # Panel 2: f2 as function of h (= f1 output)
    ax = axes[1]
    ax.plot(h_grid, f2_h, color="#E67E22", lw=2.2)
    ax.set_xlabel(r"$h = f_1(x)$", fontsize=11)
    ax.set_ylabel(r"$f_2(h)$", fontsize=11)
    ax.set_title(r"Layer 2: $f_2: \mathbb{R} \to \mathbb{R}$",
                 fontsize=10)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.0, 2.0)
    ax.axhline(0, color="k", lw=0.6, alpha=0.3)
    ax.axvline(0, color="k", lw=0.6, alpha=0.3)
    apply_style(ax)

    # Panel 3: composition f2 ∘ f1 as function of x
    ax = axes[2]
    ax.plot(x, f12, color="#117A65", lw=2.2)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel(r"$f_2(f_1(x))$", fontsize=11)
    ax.set_title(r"Composition: non-stationary effect", fontsize=10)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.0, 2.0)
    ax.axhline(0, color="k", lw=0.6, alpha=0.3)
    ax.axvline(0, color="k", lw=0.6, alpha=0.3)
    apply_style(ax)


# ---------------------------------------------------------------------------
# Figure 3: predictive density at three test points
# ---------------------------------------------------------------------------


def plot_predictive_density(axes):
    """dgp_concept.qmd -> fig-predictive-density

    Histograms of DGP predictive samples at three test points, showing
    that the predictive can be Gaussian (smooth region), skewed (steep
    warp region), or wide and asymmetric (extrapolation).

    Uses the same fitted 2-layer DGP as figures 1 and 4. The DGP was
    fitted with deterministic Gauss–Hermite propagation, but for
    drawing many predictive samples we temporarily swap in a Monte
    Carlo propagator. Predictive sampling at MC budgets of thousands
    is what histograms need; the GH rule fixes a small number of
    weighted nodes which is the wrong tool for an empirical density
    plot.
    """
    from ._style import apply_style
    from scipy.stats import norm as _norm

    from pyapprox.surrogates.gaussianprocess.deep.propagator import (
        LayerPropagator,
    )
    from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
        MonteCarloRule,
    )

    X_train, y_train, _, _ = _step_function_dataset()
    fitted, bkd = _fit_two_layer_dgp(X_train, y_train)

    # Three test locations
    x_test_locs = np.array([-1.5, 0.0, 2.5])
    test_labels = [
        "Smooth region\n($x = -1.5$)",
        "Near hidden-layer warp\n($x = 0.0$)",
        "Extrapolation region\n($x = 2.5$)",
    ]

    x_test = bkd.array(x_test_locs.reshape(1, -1))
    n_samples = 4000

    # Swap to MC propagator for sampling, restore GH afterwards.
    original_propagator = fitted.propagator()
    mc_propagator = LayerPropagator(
        bkd, rule=MonteCarloRule(rng=np.random.default_rng(17)),
    )
    try:
        fitted.set_propagator(mc_propagator)
        samples = fitted.predictive_samples(x_test, n_samples=n_samples)
    finally:
        fitted.set_propagator(original_propagator)

    samples_np = bkd.to_numpy(samples)[:, 0, :]  # (n_samples, 3)

    colors = ["#2C7FB8", "#E67E22", "#7D3C98"]
    for i, ax in enumerate(axes):
        s = samples_np[:, i]
        ax.hist(
            s, bins=50, density=True, color=colors[i],
            alpha=0.55, edgecolor="k", lw=0.3,
            label="DGP samples",
        )

        mu = float(s.mean())
        sd = float(s.std()) + 1e-9
        xx = np.linspace(s.min() - 0.2, s.max() + 0.2, 200)
        ax.plot(
            xx, _norm.pdf(xx, mu, sd), color="k", lw=1.5, ls="--",
            label="Gaussian (mean, std)",
        )

        ax.set_title(test_labels[i], fontsize=10)
        ax.set_xlabel(r"$f(x)$", fontsize=10)
        if i == 0:
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
        apply_style(ax)


# ---------------------------------------------------------------------------
# Figure 4: latent representation with inducing points
# ---------------------------------------------------------------------------


def _layer1_input_skip(layer1, x_grid_np, hidden_value, bkd):
    """Build the leaf-layer input [x, h] for a chain DGP with skip
    connections. x_grid_np: (1, n), hidden_value: (n,) or scalar.
    Returns a (2, n) bkd-array."""
    if np.ndim(hidden_value) == 0:
        hidden_value = np.full(x_grid_np.shape[1], float(hidden_value))
    aug = np.vstack([x_grid_np, hidden_value.reshape(1, -1)])
    return bkd.array(aug)


def plot_latent_representation(axes):
    """dgp_concept.qmd -> fig-latent-representation

    Two-panel decomposition of what the fitted 2-layer DGP learned.

    Left:  hidden-layer posterior mean f_1(x) as a function of x,
           with layer-0 inducing points marked on the x-axis.
    Right: leaf-layer posterior mean as a function of the hidden
           value h, holding the skip-connected x-component fixed at
           the median training input. Layer-1 inducing points
           (projected onto the h-axis) marked on the h-axis.
    """
    from ._style import apply_style

    X_train, y_train, _, _ = _step_function_dataset()
    fitted, bkd = _fit_two_layer_dgp(X_train, y_train)

    layers = fitted.layers()
    layer_0 = layers[0]
    layer_1 = layers[1]

    # ---- Left panel: hidden mean f_1(x) ----
    x_grid = np.linspace(-2.5, 2.5, 200).reshape(1, -1)
    x_grid_t = bkd.array(x_grid)
    hidden_mean_t, hidden_var_t = layer_0.predict_marginal(x_grid_t)
    hidden_mean = bkd.to_numpy(hidden_mean_t).ravel()
    hidden_std = np.sqrt(np.clip(bkd.to_numpy(hidden_var_t).ravel(), 0, None))

    Z0 = bkd.to_numpy(
        layer_0.inducing_points().get_samples()
    ).ravel()  # (M0,) — locations in x-space

    ax = axes[0]
    ax.fill_between(
        x_grid[0], hidden_mean - 2 * hidden_std,
        hidden_mean + 2 * hidden_std,
        color="#2C7FB8", alpha=0.18,
        label=r"Hidden $\pm 2\sigma$",
    )
    ax.plot(
        x_grid[0], hidden_mean, color="#2C7FB8", lw=2,
        label=r"$\mu_{f_1}(x)$",
    )
    # Training-point hidden values: each training input x_i, plotted at
    # its fitted hidden value h_i = mu_{f_1}(x_i). These lie on the
    # warping curve by construction; they show where in x-space the
    # warping is most strongly constrained by data.
    X_train_t = bkd.array(X_train)
    h_train_t, _ = layer_0.predict_marginal(X_train_t)
    h_train = bkd.to_numpy(h_train_t).ravel()
    ax.scatter(
        X_train[0], h_train, s=20, color="k", zorder=5,
        label=r"Data $x_i \to h_i$ on warping",
    )
    # Inducing-point markers on the bottom edge
    y_lo, y_hi = ax.get_ylim()
    marker_y = y_lo + 0.04 * (y_hi - y_lo)
    ax.scatter(
        Z0, np.full_like(Z0, marker_y),
        marker="^", s=55, facecolor="white",
        edgecolor="#1B4F72", linewidth=1.4,
        label=r"Layer-0 inducing $z_0$",
        zorder=6, clip_on=False,
    )
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel(r"Hidden value $f_1(x)$", fontsize=11)
    ax.set_title("Layer 0: learned input warping", fontsize=10)
    ax.set_xlim(-2.5, 2.5)
    ax.legend(fontsize=8, loc="upper left")
    apply_style(ax)

    # ---- Right panel: leaf mean as function of h ----
    # Sweep h across the range of fitted hidden means; hold the
    # skip-connected x-component at the median training input.
    h_min, h_max = hidden_mean.min(), hidden_mean.max()
    h_pad = 0.15 * (h_max - h_min + 1e-6)
    h_grid = np.linspace(h_min - h_pad, h_max + h_pad, 200)
    x_anchor = float(np.median(X_train[0]))
    x_anchor_grid = np.full((1, h_grid.shape[0]), x_anchor)
    leaf_input = _layer1_input_skip(
        layer_1, x_anchor_grid, h_grid, bkd,
    )  # (2, 200)

    leaf_mean_t, leaf_var_t = layer_1.predict_marginal(leaf_input)
    leaf_mean = bkd.to_numpy(leaf_mean_t).ravel()
    leaf_std = np.sqrt(np.clip(bkd.to_numpy(leaf_var_t).ravel(), 0, None))

    # Project layer-1 inducing locations onto the h-axis (component 1
    # of the (x, h) augmented input).
    Z1 = bkd.to_numpy(layer_1.inducing_points().get_samples())  # (2, M1)
    Z1_h = Z1[1, :]

    ax = axes[1]
    ax.fill_between(
        h_grid, leaf_mean - 2 * leaf_std, leaf_mean + 2 * leaf_std,
        color="#117A65", alpha=0.10,
        label=r"Slice $\pm 2\sigma$",
    )
    ax.plot(
        h_grid, leaf_mean, color="#7F8C8D", lw=1.5, ls="--",
        label=r"Slice $\mu_{f_2}(x_{\mathrm{med}}, h)$",
    )
    # Effective leaf-layer data: each training observation y_i plotted
    # at its propagated hidden value h_i = mu_{f_1}(x_i). These are the
    # points the leaf layer is effectively performing GP regression on,
    # after layer 0 has rewritten x_i into h_i.
    ax.scatter(
        h_train, y_train[0], s=22, color="k", zorder=5,
        label=r"Data $(h_i, y_i)$",
    )

    # Per-point leaf predictions at the actual training inputs, NOT
    # at x_med. These show the leaf's true fit quality. Each is
    # mu_{f_2}(x_i, h_i), with x_i and h_i their training values.
    leaf_train_input = bkd.array(np.vstack([X_train, h_train.reshape(1, -1)]))
    leaf_yhat_t, _ = layer_1.predict_marginal(leaf_train_input)
    leaf_yhat = bkd.to_numpy(leaf_yhat_t).ravel()

    # Residual lines connecting (h_i, y_i) to (h_i, yhat_i). Short
    # = good fit, long = poor fit.
    for hi, yi, yhi in zip(h_train, y_train[0], leaf_yhat):
        ax.plot(
            [hi, hi], [yi, yhi], color="#888888", lw=0.7,
            alpha=0.7, zorder=4,
        )
    ax.scatter(
        h_train, leaf_yhat, s=22, marker="o",
        facecolor="#117A65", edgecolor="#0E6655", linewidth=0.6,
        zorder=6,
        label=r"Leaf prediction $\hat y_i = \mu_{f_2}(x_i, h_i)$",
    )

    # Residual RMS as a quantitative quality indicator
    residual_rms = float(np.sqrt(np.mean(
        (y_train[0] - leaf_yhat) ** 2
    )))
    ax.text(
        0.04, 0.96,
        f"residual RMS = {residual_rms:.3f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.85,
                  boxstyle="round,pad=0.3"),
    )

    y_lo, y_hi = ax.get_ylim()
    marker_y = y_lo + 0.04 * (y_hi - y_lo)
    ax.scatter(
        Z1_h, np.full_like(Z1_h, marker_y),
        marker="^", s=55, facecolor="white",
        edgecolor="#0E6655", linewidth=1.4,
        label=r"Layer-1 inducing $z_1$ (h-component)",
        zorder=6, clip_on=False,
    )
    ax.set_xlabel(r"Hidden value $h$", fontsize=11)
    ax.set_ylabel(r"Leaf output $f_2(x_{\mathrm{med}}, h)$", fontsize=11)
    ax.set_title("Layer 1: smooth function of hidden value", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    apply_style(ax)
