"""
Tests for GP ensemble uncertainty quantification.

Tests the GaussianProcessEnsemble class for sampling GP realizations and
computing the distribution of Sobol sensitivity indices via the refit
algorithm.
"""

import math

import numpy as np

from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.statistics import (
    GaussianProcessStatistics,
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.gaussianprocess.statistics.ensemble import (
    GaussianProcessEnsemble,
    SobolThresholdSelector,
)
from pyapprox.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)
from pyapprox.surrogates.kernels.composition import (
    SeparableProductKernel,
)
from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from tests._helpers.markers import slow_test


def _create_quadrature_bases(marginals, nquad_points, bkd):
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


def _make_ensemble(bkd, nugget=1e-6, n_train=15):
    """Create a standard test ensemble with 2D GP."""
    np.random.seed(42)

    k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
    kernel = SeparableProductKernel([k1, k2], bkd)

    gp = ExactGaussianProcess(
        kernel, nvars=2, bkd=bkd, nugget=nugget
    )
    gp.hyp_list().set_all_inactive()

    X_train_np = np.random.rand(2, n_train) * 2 - 1
    X_train = bkd.array(X_train_np)
    y_train = bkd.reshape(
        bkd.sin(math.pi * X_train[0, :])
        * bkd.cos(math.pi * X_train[1, :]),
        (1, -1),
    )
    gp.fit(X_train, y_train)

    marginals = [
        UniformMarginal(-1.0, 1.0, bkd),
        UniformMarginal(-1.0, 1.0, bkd),
    ]

    nquad_points = 30
    bases = _create_quadrature_bases(marginals, nquad_points, bkd)
    calc = SeparableKernelIntegralCalculator(
        gp, bases, marginals, bkd=bkd
    )
    stats = GaussianProcessStatistics(gp, calc)
    sens = GaussianProcessSensitivity(stats)
    ensemble = GaussianProcessEnsemble(gp, sens)

    return ensemble, sens, gp, marginals


class TestGaussianProcessEnsemble:
    """Base tests for GaussianProcessEnsemble."""

    def test_nvars(self, bkd) -> None:
        ensemble, _, _, _ = _make_ensemble(bkd)
        assert ensemble.nvars() == 2

    def test_sample_realizations_shape(self, bkd) -> None:
        ensemble, _, _, _ = _make_ensemble(bkd)
        Z = bkd.array(np.random.rand(2, 50) * 2 - 1)
        realizations = ensemble.sample_realizations(Z, 10, seed=42)
        assert realizations.shape == (10, 50)

    def test_realizations_have_variance(self, bkd) -> None:
        """Realizations at non-training points have nonzero variance."""
        ensemble, _, _, _ = _make_ensemble(bkd)
        Z = bkd.array(np.random.rand(2, 100) * 2 - 1)
        realizations = ensemble.sample_realizations(Z, 10, seed=42)

        mean_r = bkd.mean(realizations, axis=0)
        var_r = bkd.mean(
            (realizations - mean_r) ** 2, axis=0
        )
        var_np = bkd.to_numpy(var_r)
        n_nonzero = np.sum(var_np > 1e-12)
        assert n_nonzero > len(var_np) * 0.5


class TestSobolThresholdSelector:
    """Tests for the sample point selector."""

    def test_select_shape(self, bkd) -> None:
        _, _, gp, marginals = _make_ensemble(bkd)
        selector = SobolThresholdSelector(gp, marginals, bkd)
        Z = selector.select(50, seed=42)
        assert Z.shape == (2, 50)

    def test_selected_points_in_domain(self, bkd) -> None:
        _, _, gp, marginals = _make_ensemble(bkd)
        selector = SobolThresholdSelector(gp, marginals, bkd)
        Z = selector.select(50, seed=42)
        min_val = float(bkd.to_numpy(bkd.min(Z)))
        max_val = float(bkd.to_numpy(bkd.max(Z)))
        assert min_val >= -1.0 - 1e-10
        assert max_val <= 1.0 + 1e-10


class TestZeroVarianceAtTraining:
    """Demonstrate why training points fail for MC integration."""

    def test_zero_variance_at_training_points(self, bkd) -> None:
        np.random.seed(42)
        k = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        gp = ExactGaussianProcess(k, nvars=1, bkd=bkd, nugget=1e-10)

        X_train = bkd.array([[-0.5, 0.0, 0.5]])
        y_train = bkd.array([[1.0, 0.0, -1.0]])
        gp.hyp_list().set_all_inactive()
        gp.fit(X_train, y_train)

        std_at_train = gp.predict_std(X_train)
        std_np = bkd.to_numpy(std_at_train)
        assert np.all(std_np < 1e-4)

        X_test = bkd.array([[0.25]])
        std_at_test = gp.predict_std(X_test)
        std_test = float(bkd.to_numpy(std_at_test[0, 0]))
        assert std_test > 1e-3


class TestSobolDistributionRefit:
    """Tests for refit-based Sobol distribution algorithm."""

    def test_shape(self, bkd) -> None:
        ensemble, _, _, _ = _make_ensemble(bkd)
        S_dist = ensemble.sobol_distribution_refit(
            n_realizations=5, n_sample_points=50, seed=42
        )
        assert set(S_dist.keys()) == {0, 1}
        for i in range(2):
            assert S_dist[i].shape == (5,)

    def test_bounded(self, bkd) -> None:
        ensemble, _, _, _ = _make_ensemble(bkd)
        S_dist = ensemble.sobol_distribution_refit(
            n_realizations=10, n_sample_points=50, seed=42
        )
        for i, S_i in S_dist.items():
            min_val = float(bkd.to_numpy(bkd.min(S_i)))
            max_val = float(bkd.to_numpy(bkd.max(S_i)))
            assert min_val >= -1e-10, f"S_{i} min={min_val}"
            assert max_val <= 1.0 + 1e-10, f"S_{i} max={max_val}"

    def test_reproducibility(self, bkd) -> None:
        ensemble, _, _, _ = _make_ensemble(bkd)
        S1 = ensemble.sobol_distribution_refit(
            n_realizations=5, n_sample_points=50, seed=42
        )
        S2 = ensemble.sobol_distribution_refit(
            n_realizations=5, n_sample_points=50, seed=42
        )
        for i in range(2):
            bkd.assert_allclose(S1[i], S2[i], rtol=1e-10)

    @slow_test
    def test_mean_near_analytical(self, bkd) -> None:
        """Mean of sampled S_i close to analytical."""
        ensemble, sens, _, _ = _make_ensemble(bkd)

        main_pm = sens.main_effect_indices_of_posterior_mean()

        S_dist = ensemble.sobol_distribution_refit(
            n_realizations=100, n_sample_points=200, seed=42
        )

        for i in range(2):
            S_pm = float(bkd.to_numpy(main_pm[i]))
            S_mean = float(bkd.to_numpy(bkd.mean(S_dist[i])))
            assert abs(S_mean - S_pm) < 0.2, (
                f"S_{i} mean={S_mean:.3f}, analytical={S_pm:.3f}"
            )

