"""Function recovery tests for Deep GP end-to-end validation (Phase 12).

These tests validate that the full pipeline (build -> fit -> predict)
produces accurate predictions on known test functions.
"""

import numpy as np
import pytest

from tests._helpers.markers import slow_test

from pyapprox.optimization.minimize.adam.adam_optimizer import AdamOptimizer
from pyapprox.optimization.minimize.chained.chained_optimizer import (
    ChainedOptimizer,
)
from pyapprox.optimization.minimize.scipy.lbfgsb import LBFGSBOptimizer
from pyapprox.surrogates.gaussianprocess.deep.builders import (
    build_multilevel_dgp,
    build_single_fidelity_dgp,
)
from pyapprox.surrogates.gaussianprocess.deep.propagator import (
    LayerPropagator,
)
from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
    TensorProductGHRule,
)
from pyapprox.surrogates.gaussianprocess.deep_gp_loss import (
    TorchDGPELBOLoss,
)
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.fitters.deep_gp_fitter import (
    DGPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.kernels.iid_gaussian_noise import IIDGaussianNoise
from pyapprox.surrogates.kernels.matern import Matern52Kernel

_TRAIN_NPROP = 1
_PRED_NPROP = 10
_MAXITER = 500
_LR = 1e-2


def _matern_factory(nvars, bkd):
    return Matern52Kernel(
        lenscale=[1.0] * nvars,
        lenscale_bounds=(0.1, 10.0),
        nvars=nvars,
        bkd=bkd,
    )


def _make_fitter(bkd, maxiter=_MAXITER):
    optimizer = AdamOptimizer(lr=_LR, maxiter=maxiter, verbosity=0)
    return DGPMaximumLikelihoodFitter(
        bkd, optimizer=optimizer, n_propagation=_TRAIN_NPROP,
    )


class TestSingleFidelityRecovery:
    """Single-fidelity DGP should recover smooth and non-smooth functions."""

    @slow_test
    def test_sinusoidal_recovery(self, torch_bkd):
        """2-layer DGP recovers sin(x) with low RMSE."""
        bkd = torch_bkd
        rng = np.random.RandomState(42)
        X_train = bkd.array(rng.uniform(-3, 3, (1, 20)))
        y_train = bkd.array(np.sin(bkd.to_numpy(X_train)))

        dgp = build_single_fidelity_dgp(
            2, nvars=1, num_inducing=10,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.01, n_propagation=_TRAIN_NPROP, seed=0,
        )

        result = _make_fitter(bkd).fit(dgp, {1: (X_train, y_train)})
        fitted = result.surrogate()
        assert fitted.is_fitted()

        X_test = bkd.array(np.linspace(-3, 3, 30).reshape(1, -1))
        y_true = np.sin(bkd.to_numpy(X_test))
        y_pred = bkd.to_numpy(fitted.predict(X_test, n_propagation=_PRED_NPROP))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        assert rmse < 0.5, f"sin(x) RMSE {rmse:.4f} too high"

    @slow_test
    def test_step_function_dgp_beats_single_layer(self, torch_bkd):
        """2-layer DGP should fit a step function better than 1-layer."""
        bkd = torch_bkd
        rng = np.random.RandomState(42)
        X_train_np = rng.uniform(-2, 2, (1, 30))
        y_train_np = np.where(X_train_np > 0, 1.0, -1.0)
        y_train_np += 0.05 * rng.randn(1, 30)
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        X_test_np = np.linspace(-2, 2, 30).reshape(1, -1)
        y_test_np = np.where(X_test_np > 0, 1.0, -1.0)
        X_test = bkd.array(X_test_np)

        optimizer = ChainedOptimizer(
            AdamOptimizer(lr=_LR, maxiter=300, verbosity=0),
            LBFGSBOptimizer(maxiter=500),
        )

        dgp_1 = build_single_fidelity_dgp(
            1, nvars=1, num_inducing=10,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.05, n_propagation=_TRAIN_NPROP, seed=0,
        )
        fitter_1 = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=_TRAIN_NPROP,
        )
        result_1 = fitter_1.fit(dgp_1, {0: (X_train, y_train)})
        pred_1 = bkd.to_numpy(
            result_1.surrogate().predict(X_test, n_propagation=_PRED_NPROP)
        )
        rmse_1 = np.sqrt(np.mean((pred_1 - y_test_np) ** 2))

        gh_order = 5
        gh_nprop = gh_order ** 2
        dgp_2 = build_single_fidelity_dgp(
            2, nvars=1, num_inducing=10,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.05, n_propagation=gh_nprop, seed=0,
        )
        dgp_2.set_propagator(
            LayerPropagator(bkd, rule=TensorProductGHRule(gh_order))
        )
        fitter_2 = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=gh_nprop,
        )
        result_2 = fitter_2.fit(dgp_2, {1: (X_train, y_train)})
        pred_2 = bkd.to_numpy(
            result_2.surrogate().predict(X_test, n_propagation=gh_nprop)
        )
        rmse_2 = np.sqrt(np.mean((pred_2 - y_test_np) ** 2))

        assert rmse_2 < rmse_1, (
            f"2-layer RMSE ({rmse_2:.4f}) should be lower than "
            f"1-layer RMSE ({rmse_1:.4f}) on step function"
        )

    @slow_test
    def test_uncertainty_covers_truth(self, torch_bkd):
        """Predictive intervals should cover the true function."""
        bkd = torch_bkd
        rng = np.random.RandomState(42)
        X_train = bkd.array(rng.uniform(-2, 2, (1, 15)))
        y_train = bkd.array(np.sin(bkd.to_numpy(X_train)))

        dgp = build_single_fidelity_dgp(
            2, nvars=1, num_inducing=8,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.01, n_propagation=_TRAIN_NPROP, seed=0,
        )

        result = _make_fitter(bkd).fit(dgp, {1: (X_train, y_train)})
        fitted = result.surrogate()

        X_test = bkd.array(np.linspace(-2, 2, 20).reshape(1, -1))
        y_true = np.sin(bkd.to_numpy(X_test))
        y_pred = bkd.to_numpy(fitted.predict(X_test, n_propagation=_PRED_NPROP))
        y_std = bkd.to_numpy(fitted.predict_std(X_test, n_propagation=_PRED_NPROP))

        residuals = np.abs(y_pred - y_true)
        coverage = np.mean(residuals < 3.0 * y_std)
        assert coverage > 0.7, (
            f"3-sigma coverage {coverage:.2f} too low (expect > 0.7)"
        )

    @slow_test
    def test_nonzero_predictive_skewness(self, torch_bkd):
        """2-layer DGP predictive should have nonzero skewness
        at points where warping is steep, confirming non-Gaussian
        uncertainty propagation."""
        bkd = torch_bkd
        rng = np.random.RandomState(42)
        X_train_np = rng.uniform(-2, 2, (1, 20))
        y_train_np = np.tanh(3.0 * X_train_np)
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        dgp = build_single_fidelity_dgp(
            2, nvars=1, num_inducing=8,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.01, n_propagation=_TRAIN_NPROP, seed=0,
        )

        result = _make_fitter(bkd).fit(dgp, {1: (X_train, y_train)})
        fitted = result.surrogate()

        X_test = bkd.array(np.array([[-0.1, 0.0, 0.1]]))
        samples = bkd.to_numpy(
            fitted.predictive_samples(X_test, n_samples=100)
        )
        skewness_per_point = np.zeros(X_test.shape[1])
        for j in range(X_test.shape[1]):
            s = samples[:, 0, j]
            mu = np.mean(s)
            sigma = np.std(s)
            if sigma > 1e-10:
                skewness_per_point[j] = np.mean(((s - mu) / sigma) ** 3)

        max_skew = np.max(np.abs(skewness_per_point))
        assert max_skew > 0.01, (
            f"Max |skewness| {max_skew:.4f} is too small; "
            "expected non-Gaussian predictive from warping"
        )


class TestMultilevelRecovery:
    """Multilevel DGP tests with correlated fidelity levels."""

    @slow_test
    def test_two_level_linear_correlation(self, torch_bkd):
        """Multilevel DGP with linearly correlated fidelities.

        f_lo(x) = sin(x), f_hi(x) = 2*sin(x) + 0.5
        Dense lo-fi + sparse hi-fi should beat hi-fi only.
        """
        bkd = torch_bkd
        rng = np.random.RandomState(42)

        X_lo_np = rng.uniform(-3, 3, (1, 20))
        y_lo_np = np.sin(X_lo_np) + 0.05 * rng.randn(1, 20)
        X_hi_np = rng.uniform(-3, 3, (1, 6))
        y_hi_np = 2.0 * np.sin(X_hi_np) + 0.5 + 0.05 * rng.randn(1, 6)

        X_lo = bkd.array(X_lo_np)
        y_lo = bkd.array(y_lo_np)
        X_hi = bkd.array(X_hi_np)
        y_hi = bkd.array(y_hi_np)

        optimizer = ChainedOptimizer(
            AdamOptimizer(lr=_LR, maxiter=300, verbosity=0),
            LBFGSBOptimizer(maxiter=500),
        )
        gh_order = 5
        gh_nprop = gh_order ** 2

        dgp_mf = build_multilevel_dgp(
            level_nvars=[1, 1], num_inducing=8,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.05, n_propagation=gh_nprop, seed=0,
        )
        dgp_mf.set_propagator(
            LayerPropagator(bkd, rule=TensorProductGHRule(gh_order))
        )
        fitter_mf = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=gh_nprop,
        )
        result_mf = fitter_mf.fit(
            dgp_mf, {0: (X_lo, y_lo), 1: (X_hi, y_hi)},
        )
        fitted_mf = result_mf.surrogate()
        assert fitted_mf.is_fitted()

        X_test_np = np.linspace(-3, 3, 20).reshape(1, -1)
        y_true = 2.0 * np.sin(X_test_np) + 0.5
        X_test = bkd.array(X_test_np)

        pred_mf = bkd.to_numpy(
            fitted_mf.predict(X_test, n_propagation=gh_nprop)
        )
        rmse_mf = np.sqrt(np.mean((pred_mf - y_true) ** 2))

        dgp_sf = build_single_fidelity_dgp(
            1, nvars=1, num_inducing=8,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.05, n_propagation=_TRAIN_NPROP, seed=0,
        )
        fitter_sf = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=_TRAIN_NPROP,
        )
        result_sf = fitter_sf.fit(dgp_sf, {0: (X_hi, y_hi)})
        pred_sf = bkd.to_numpy(
            result_sf.surrogate().predict(X_test, n_propagation=_PRED_NPROP)
        )
        rmse_sf = np.sqrt(np.mean((pred_sf - y_true) ** 2))

        assert rmse_mf < rmse_sf, (
            f"Multilevel RMSE ({rmse_mf:.4f}) should beat single-fidelity "
            f"RMSE ({rmse_sf:.4f}) with correlated lo/hi data"
        )

    @slow_test
    def test_high_fidelity_dominance(self, torch_bkd):
        """Dense hi-fi + sparse irrelevant lo-fi: predictions should
        still be reasonable, not corrupted by noise lo-fi."""
        bkd = torch_bkd
        rng = np.random.RandomState(42)

        X_lo_np = rng.uniform(-2, 2, (1, 15))
        y_lo_np = rng.randn(1, 15)
        X_hi_np = rng.uniform(-2, 2, (1, 20))
        y_hi_np = np.sin(X_hi_np) + 0.05 * rng.randn(1, 20)

        dgp_mf = build_multilevel_dgp(
            level_nvars=[1, 1], num_inducing=8,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.1, n_propagation=_TRAIN_NPROP, seed=0,
        )

        result = _make_fitter(bkd).fit(
            dgp_mf,
            {0: (bkd.array(X_lo_np), bkd.array(y_lo_np)),
             1: (bkd.array(X_hi_np), bkd.array(y_hi_np))},
        )
        fitted = result.surrogate()

        X_test_np = np.linspace(-2, 2, 20).reshape(1, -1)
        y_true = np.sin(X_test_np)
        X_test = bkd.array(X_test_np)

        pred = bkd.to_numpy(fitted.predict(X_test, n_propagation=_PRED_NPROP))
        rmse = np.sqrt(np.mean((pred - y_true) ** 2))

        assert rmse < 0.5, (
            f"With irrelevant lo-fi, hi-fi RMSE {rmse:.4f} should still "
            "be reasonable (< 0.5)"
        )

    @slow_test
    def test_three_level_convergence(self, torch_bkd):
        """Three correlated fidelity levels should produce reasonable
        predictions at the highest fidelity."""
        bkd = torch_bkd
        rng = np.random.RandomState(42)

        X_0_np = rng.uniform(-2, 2, (1, 20))
        y_0_np = 0.5 * np.sin(X_0_np) + 0.1 * rng.randn(1, 20)
        X_1_np = rng.uniform(-2, 2, (1, 10))
        y_1_np = np.sin(X_1_np) + 0.1 * rng.randn(1, 10)
        X_2_np = rng.uniform(-2, 2, (1, 6))
        y_2_np = np.sin(X_2_np) + 0.1 * X_2_np + 0.05 * rng.randn(1, 6)

        dgp = build_multilevel_dgp(
            level_nvars=[1, 1, 1], num_inducing=8,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.1, n_propagation=_TRAIN_NPROP, seed=0,
        )

        data = {
            0: (bkd.array(X_0_np), bkd.array(y_0_np)),
            1: (bkd.array(X_1_np), bkd.array(y_1_np)),
            2: (bkd.array(X_2_np), bkd.array(y_2_np)),
        }
        result = _make_fitter(bkd).fit(dgp, data)
        fitted = result.surrogate()
        assert fitted.is_fitted()

        X_test_np = np.linspace(-2, 2, 15).reshape(1, -1)
        y_true = np.sin(X_test_np) + 0.1 * X_test_np
        X_test = bkd.array(X_test_np)

        pred = bkd.to_numpy(fitted.predict(X_test, n_propagation=_PRED_NPROP))
        rmse = np.sqrt(np.mean((pred - y_true) ** 2))

        assert rmse < 0.5, (
            f"3-level DGP RMSE {rmse:.4f} should be < 0.5"
        )


def _build_generative_two_layer_dgp(
    bkd,
    noise_std=0.05,
    seed=42,
    num_inducing=10,
    nvars=1,
    m_tilde_scale=1.0,
    l_tilde_diag_scale=0.3,
    l_tilde_offdiag_scale=0.0,
):
    """Build a 2-layer DGP at hand-chosen parameters for synthetic data.

    Kernel hyperparameters are fixed. The variational q(u) at each layer
    is set with controlled scale so the generative function has high SNR
    and is recoverable from a moderate-sized training set.

    Parameters
    ----------
    m_tilde_scale : float
        Scale of the variational mean.
    l_tilde_diag_scale : float
        Diagonal scale of the variational Cholesky factor.
    l_tilde_offdiag_scale : float
        Off-diagonal scale. 0 gives a diagonal covariance.
    """
    rng = np.random.RandomState(seed)

    def kernel_factory(nv, b):
        return Matern52Kernel(
            lenscale=[2.0] * nv, lenscale_bounds=(0.1, 10.0),
            nvars=nv, bkd=b, fixed=True,
        )

    dgp = build_single_fidelity_dgp(
        n_layers=2, nvars=nvars, num_inducing=num_inducing,
        kernel_factory=kernel_factory, bkd=bkd,
        noise_std=noise_std, n_propagation=50, seed=seed,
    )

    for node_id, layer in dgp.layers().items():
        M = layer.variational_dist().num_inducing()
        vd = layer.variational_dist()
        m_np = m_tilde_scale * rng.randn(M)
        L_np = l_tilde_diag_scale * np.eye(M)
        if l_tilde_offdiag_scale > 0:
            L_np += l_tilde_offdiag_scale * np.tril(rng.randn(M, M), k=-1)
        vd._mean_param.set_values(bkd.array(m_np))
        mask_np = np.tril(np.ones((M, M), dtype=bool))
        vd._chol_param.set_values(bkd.array(L_np[mask_np]))

    return dgp


def _check_generative_snr(gen_dgp, X_test, bkd, min_snr=2.0):
    """Verify the generative DGP has high enough SNR to be recoverable."""
    mean = bkd.to_numpy(gen_dgp.predict(X_test))
    std = bkd.to_numpy(gen_dgp.predict_std(X_test))
    mean_range = mean.max() - mean.min()
    avg_std = std.mean()
    snr = mean_range / avg_std if avg_std > 0 else float("inf")
    assert snr > min_snr, (
        f"Generative DGP SNR = {snr:.2f} < {min_snr}. "
        f"Mean range = {mean_range:.3f}, avg std = {avg_std:.3f}. "
        f"Recovery test would be testing noise-fitting, not function recovery."
    )


def _sample_generative_dgp_at(dgp, X, bkd, n_samples=200):
    """Draw predictive samples and compute mean/std from a generative DGP.

    Returns
    -------
    mean : ndarray, shape (1, n_test)
    std : ndarray, shape (1, n_test)
    samples : ndarray, shape (n_samples, 1, n_test)
    """
    mean = bkd.to_numpy(dgp.predict(X))
    std = bkd.to_numpy(dgp.predict_std(X))
    samples = bkd.to_numpy(dgp.predictive_samples(X, n_samples=n_samples))
    return mean, std, samples


class TestDGPClosedFormRecovery:
    """Compare fitted DGP predictions against closed-form references."""

    @slow_test
    def test_one_layer_dgp_recovers_exact_gp(self, torch_bkd):
        """A 1-layer DGP (single SVGP) should approximate an exact GP.

        Trains both on the same data, then compares predictive mean and
        std on a test grid.
        """
        bkd = torch_bkd
        rng = np.random.RandomState(42)

        noise_std = 0.1
        nvars = 1
        n_train = 20
        M = 15

        X_train_np = rng.uniform(-3, 3, (nvars, n_train))
        y_train_np = np.sin(X_train_np) + noise_std * rng.randn(1, n_train)
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        kernel_exact = Matern52Kernel(
            lenscale=[1.0] * nvars, lenscale_bounds=(0.1, 10.0),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        noise_kernel = IIDGaussianNoise(
            noise_std**2, (1e-6, 1.0), bkd, fixed=True,
        )
        gp_kernel = kernel_exact + noise_kernel
        exact_gp = ExactGaussianProcess(gp_kernel, nvars, bkd, nugget=1e-10)
        exact_gp._fit_internal(X_train, y_train)

        def dgp_kernel_factory(nv, b):
            return Matern52Kernel(
                lenscale=[1.0] * nv, lenscale_bounds=(0.1, 10.0),
                nvars=nv, bkd=b, fixed=True,
            )

        dgp = build_single_fidelity_dgp(
            n_layers=1, nvars=nvars, num_inducing=M,
            kernel_factory=dgp_kernel_factory, bkd=bkd,
            noise_std=noise_std, n_propagation=1, seed=0,
            inducing_bounds=(-3.5, 3.5),
        )
        dgp.layers()[0].likelihood().hyp_list().set_all_inactive()

        optimizer = ChainedOptimizer(
            AdamOptimizer(lr=5e-3, maxiter=300, verbosity=0),
            LBFGSBOptimizer(maxiter=500),
        )
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=1,
        )
        result = fitter.fit(dgp, {0: (X_train, y_train)})
        fitted = result.surrogate()

        X_test = bkd.array(np.linspace(-3, 3, 30).reshape(1, -1))
        gp_mean = bkd.to_numpy(exact_gp.predict(X_test))
        gp_std = bkd.to_numpy(exact_gp.predict_std(X_test))

        dgp_mean = bkd.to_numpy(fitted.predict(X_test, n_propagation=1))
        dgp_std = bkd.to_numpy(fitted.predict_std(X_test, n_propagation=1))

        mean_rmse = np.sqrt(np.mean((dgp_mean - gp_mean) ** 2))
        assert mean_rmse < 0.015, (
            f"DGP mean RMSE vs exact GP = {mean_rmse:.4f}, expect < 0.015"
        )

        std_mask = gp_std > 1e-3
        if np.any(std_mask):
            rel_err = np.mean(
                np.abs(dgp_std[std_mask] - gp_std[std_mask])
                / gp_std[std_mask]
            )
            assert rel_err < 0.30, (
                f"DGP std relative error = {rel_err:.4f}, expect < 0.30"
            )

    @slow_test
    def test_recovery_of_generative_two_layer_dgp(self, torch_bkd):
        """Fit a 2-layer DGP on data sampled from a known 2-layer DGP.

        Draws ONE function realization from the generative DGP at both
        train and test points. The fitted model should:
        1. Recover the realization's mean (RMSE < 0.15)
        2. Learn the noise level (std at training ~ noise_std)
        3. Show growing uncertainty at extrapolation
        4. Cover the truth realization with 95% CIs (coverage > 0.85)
        """
        bkd = torch_bkd
        rng = np.random.RandomState(7)

        noise_std = 0.05
        gen_dgp = _build_generative_two_layer_dgp(bkd, noise_std=noise_std)

        n_train = 40
        n_test = 25
        X_train_np = rng.uniform(-2, 2, (1, n_train))
        X_test_np = np.linspace(-2, 2, n_test).reshape(1, -1)
        X_extrap_np = np.array([[-3.5, -3.0, 3.0, 3.5]])

        X_all_np = np.concatenate(
            [X_train_np, X_test_np, X_extrap_np], axis=1,
        )
        X_all = bkd.array(X_all_np)
        _check_generative_snr(gen_dgp, bkd.array(X_test_np), bkd)

        all_samples = bkd.to_numpy(
            gen_dgp.predictive_samples(X_all, n_samples=1)
        )
        f_truth_all = all_samples[0, 0, :]
        f_truth_train = f_truth_all[:n_train]
        f_truth_test = f_truth_all[n_train:n_train + n_test]

        y_train_np = f_truth_train.reshape(1, -1) + (
            noise_std * rng.randn(1, n_train)
        )
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        def kernel_factory(nv, b):
            return Matern52Kernel(
                lenscale=[2.0] * nv, lenscale_bounds=(0.1, 10.0),
                nvars=nv, bkd=b,
            )

        gh_order = 5
        gh_nprop = gh_order ** 2
        fit_dgp = build_single_fidelity_dgp(
            n_layers=2, nvars=1, num_inducing=10,
            kernel_factory=kernel_factory, bkd=bkd,
            noise_std=noise_std, n_propagation=gh_nprop, seed=7,
        )
        fit_dgp.set_propagator(
            LayerPropagator(bkd, rule=TensorProductGHRule(gh_order))
        )

        optimizer = ChainedOptimizer(
            AdamOptimizer(lr=1e-2, maxiter=300, verbosity=0),
            LBFGSBOptimizer(maxiter=500),
        )
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=gh_nprop,
        )
        result = fitter.fit(fit_dgp, {1: (X_train, y_train)})
        fitted = result.surrogate()

        X_test = bkd.array(X_test_np)
        fit_mean = bkd.to_numpy(
            fitted.predict(X_test, n_propagation=gh_nprop)
        )
        fit_std = bkd.to_numpy(
            fitted.predict_std(X_test, n_propagation=gh_nprop)
        )

        mean_rmse = np.sqrt(
            np.mean((fit_mean[0, :] - f_truth_test) ** 2)
        )
        assert mean_rmse < 0.05, (
            f"Fitted DGP mean RMSE vs truth realization = {mean_rmse:.4f}, "
            "expect < 0.05"
        )

        std_at_train = bkd.to_numpy(
            fitted.predict_std(X_train, n_propagation=gh_nprop)
        )
        avg_std_train = np.mean(std_at_train)
        assert abs(avg_std_train - noise_std) / noise_std < 0.5, (
            f"avg std at training = {avg_std_train:.4f}, "
            f"expected ~{noise_std} (within 50%)"
        )

        X_extrap = bkd.array(X_extrap_np)
        std_at_extrap = bkd.to_numpy(
            fitted.predict_std(X_extrap, n_propagation=gh_nprop)
        )
        avg_std_extrap = np.mean(std_at_extrap)
        assert avg_std_extrap > 2 * avg_std_train, (
            f"Extrapolation std ({avg_std_extrap:.4f}) should be > "
            f"2x training std ({avg_std_train:.4f})"
        )

        within_ci = (
            np.abs(fit_mean[0, :] - f_truth_test) < 1.96 * fit_std[0, :]
        )
        coverage = np.mean(within_ci)
        assert coverage > 0.95, (
            f"95% CI coverage of truth realization = {coverage:.2f}, "
            "expect > 0.95"
        )


class TestDGPCalibration:
    """Predictive calibration on DGP-generated data."""

    @slow_test
    def test_predictive_calibration_on_dgp_generated_data(self, torch_bkd):
        """95% credible intervals from a fitted DGP should cover the
        truth function realization at most test points.

        Draws one realization from a generative DGP at concatenated
        train+test points, trains on the train portion, and checks
        coverage at the test portion.
        """
        bkd = torch_bkd
        rng = np.random.RandomState(11)

        noise_std = 0.05
        gen_dgp = _build_generative_two_layer_dgp(
            bkd, noise_std=noise_std, seed=11,
        )

        n_train = 40
        n_test = 20
        X_train_np = rng.uniform(-2, 2, (1, n_train))
        X_test_np = np.linspace(-2, 2, n_test).reshape(1, -1)

        X_all_np = np.concatenate([X_train_np, X_test_np], axis=1)
        X_all = bkd.array(X_all_np)
        _check_generative_snr(gen_dgp, bkd.array(X_test_np), bkd)

        all_samples = bkd.to_numpy(
            gen_dgp.predictive_samples(X_all, n_samples=1)
        )
        f_truth_all = all_samples[0, 0, :]
        f_truth_train = f_truth_all[:n_train]
        f_truth_test = f_truth_all[n_train:]

        y_train_np = f_truth_train.reshape(1, -1) + (
            noise_std * rng.randn(1, n_train)
        )
        X_train = bkd.array(X_train_np)
        y_train = bkd.array(y_train_np)

        def kernel_factory(nv, b):
            return Matern52Kernel(
                lenscale=[2.0] * nv, lenscale_bounds=(0.1, 10.0),
                nvars=nv, bkd=b,
            )

        gh_order = 5
        gh_nprop = gh_order ** 2
        fit_dgp = build_single_fidelity_dgp(
            n_layers=2, nvars=1, num_inducing=10,
            kernel_factory=kernel_factory, bkd=bkd,
            noise_std=noise_std, n_propagation=gh_nprop, seed=11,
        )
        fit_dgp.set_propagator(
            LayerPropagator(bkd, rule=TensorProductGHRule(gh_order))
        )

        optimizer = ChainedOptimizer(
            AdamOptimizer(lr=1e-2, maxiter=300, verbosity=0),
            LBFGSBOptimizer(maxiter=500),
        )
        fitter = DGPMaximumLikelihoodFitter(
            bkd, optimizer=optimizer, n_propagation=gh_nprop,
        )
        result = fitter.fit(fit_dgp, {1: (X_train, y_train)})
        fitted = result.surrogate()

        X_test = bkd.array(X_test_np)
        fit_mean = bkd.to_numpy(
            fitted.predict(X_test, n_propagation=gh_nprop)
        )
        fit_std = bkd.to_numpy(
            fitted.predict_std(X_test, n_propagation=gh_nprop)
        )

        within_ci = (
            np.abs(fit_mean[0, :] - f_truth_test) < 1.96 * fit_std[0, :]
        )
        coverage = np.mean(within_ci)

        assert coverage > 0.95, (
            f"95% CI coverage = {coverage:.2f}, expect > 0.90"
        )


class TestELBOConvergence:
    """ELBO should decrease during optimization."""

    @slow_test
    def test_elbo_decreases_during_fit(self, torch_bkd):
        """Negative ELBO should be lower after fitting than before."""
        bkd = torch_bkd
        rng = np.random.RandomState(42)
        X_train = bkd.array(rng.uniform(-2, 2, (1, 15)))
        y_train = bkd.array(np.sin(bkd.to_numpy(X_train)))

        dgp = build_single_fidelity_dgp(
            2, nvars=1, num_inducing=8,
            kernel_factory=_matern_factory, bkd=bkd,
            noise_std=0.1, n_propagation=_TRAIN_NPROP, seed=0,
        )

        data = {1: (X_train, y_train)}
        loss_before = TorchDGPELBOLoss(dgp, data, n_propagation=_TRAIN_NPROP)
        params_before = dgp.hyp_list().get_active_values()
        neg_elbo_before = float(
            bkd.to_numpy(loss_before(params_before))[0, 0]
        )

        result = _make_fitter(bkd).fit(dgp, data)
        neg_elbo_after = float(
            bkd.to_numpy(result.neg_log_marginal_likelihood())[0, 0]
        )

        assert neg_elbo_after < neg_elbo_before, (
            f"Neg ELBO after ({neg_elbo_after:.2f}) should be less than "
            f"before ({neg_elbo_before:.2f})"
        )
