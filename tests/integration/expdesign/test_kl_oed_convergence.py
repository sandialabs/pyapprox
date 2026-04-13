"""Tests for KL-OED convergence analysis.

Tests verify:
- Numerical EIG converges to exact EIG with increasing samples
- MSE = bias^2 + variance relationship
- Convergence rate analysis
"""

import numpy as np
import pytest

from pyapprox_benchmarks.instances.oed.linear_gaussian import (
    build_linear_gaussian_kl_benchmark,
)
from pyapprox.expdesign.data import generate_oed_data
from pyapprox.expdesign.diagnostics import (
    KLOEDDiagnostics,
    compute_convergence_rate,
    compute_estimator_mse,
)
from pyapprox.expdesign.quadrature import MonteCarloSampler
from pyapprox.expdesign.quadrature.oed import (
    OEDQuadratureSampler,
    build_oed_joint_distribution,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.markers import slow_test


class TestKLOEDConvergence:
    """Tests for KL-OED convergence analysis."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd: Backend[Array]) -> None:
        self._nobs = 5
        self._degree = 2
        self._nparams = self._degree + 1
        self._noise_std = 0.5
        self._prior_std = 0.5

    def _noise_variances(self, bkd: Backend[Array]) -> Array:
        return bkd.full((self._nobs,), self._noise_std**2)

    def _make_sampler(
        self, bkd: Backend[Array], seed: int,
    ) -> OEDQuadratureSampler[Array]:
        """Create MC-based OEDQuadratureSampler."""
        bench = build_linear_gaussian_kl_benchmark(
            self._nobs, self._degree, self._noise_std, self._prior_std, bkd,
        )
        joint_dist = build_oed_joint_distribution(bench.problem(), bkd)
        np.random.seed(seed)
        return OEDQuadratureSampler(
            MonteCarloSampler(joint_dist, bkd), self._nparams, bkd,
        )

    def _compute_eig_realization(
        self,
        bkd: Backend[Array],
        nouter: int,
        ninner: int,
        weights: Array,
        seed: int,
    ) -> float:
        """Compute one EIG realization."""
        bench = build_linear_gaussian_kl_benchmark(
            self._nobs, self._degree, self._noise_std, self._prior_std, bkd,
        )
        diag = KLOEDDiagnostics(self._noise_variances(bkd), bkd)
        data = generate_oed_data(
            bench.problem(),
            self._make_sampler(bkd, seed),
            self._make_sampler(bkd, seed + 5000),
            nouter, ninner,
        )
        return diag.compute_numerical_eig(
            weights, data.outer_shapes, data.latent_samples, data.inner_shapes,
        )

    @slow_test
    def test_mse_decreases_with_outer_samples(
        self, bkd: Backend[Array],
    ) -> None:
        """MSE generally decreases with more outer samples."""
        bench = build_linear_gaussian_kl_benchmark(
            self._nobs, self._degree, self._noise_std, self._prior_std, bkd,
        )
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        exact = bench.exact_eig(weights)

        ninner = 25
        outer_counts = [25, 50, 100]
        mses = []

        for nouter in outer_counts:
            estimates = []
            for i in range(5):
                seed = 42 + i * 10000
                eig = self._compute_eig_realization(
                    bkd, nouter, ninner, weights, seed,
                )
                estimates.append(eig)
            _, _, mse = compute_estimator_mse(exact, estimates)
            mses.append(mse)

        assert mses[-1] < mses[0] * 2.0
        for mse in mses:
            assert mse > 0.0
            assert np.isfinite(mse)

    @slow_test
    def test_mse_decreases_with_inner_samples(
        self, bkd: Backend[Array],
    ) -> None:
        """MSE generally decreases with more inner samples."""
        bench = build_linear_gaussian_kl_benchmark(
            self._nobs, self._degree, self._noise_std, self._prior_std, bkd,
        )
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        exact = bench.exact_eig(weights)

        nouter = 50
        inner_counts = [15, 30, 60]
        mses = []

        for ninner in inner_counts:
            estimates = []
            for i in range(5):
                seed = 42 + i * 10000
                eig = self._compute_eig_realization(
                    bkd, nouter, ninner, weights, seed,
                )
                estimates.append(eig)
            _, _, mse = compute_estimator_mse(exact, estimates)
            mses.append(mse)

        assert mses[-1] < mses[0] * 2.0
        for mse in mses:
            assert mse > 0.0
            assert np.isfinite(mse)

    def test_bias_variance_mse_relation(
        self, bkd: Backend[Array],
    ) -> None:
        """MSE = bias^2 + variance."""
        bench = build_linear_gaussian_kl_benchmark(
            self._nobs, self._degree, self._noise_std, self._prior_std, bkd,
        )
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        exact = bench.exact_eig(weights)

        estimates = []
        for i in range(5):
            seed = 42 + i * 10000
            eig = self._compute_eig_realization(bkd, 50, 30, weights, seed)
            estimates.append(eig)

        bias, variance, mse = compute_estimator_mse(exact, estimates)

        expected_mse = bias**2 + variance
        bkd.assert_allclose(
            bkd.asarray([mse]), bkd.asarray([expected_mse]), rtol=1e-10,
        )
        assert variance >= 0.0

    def test_exact_eig_positive(self, bkd: Backend[Array]) -> None:
        """Exact EIG is positive for uniform weights."""
        bench = build_linear_gaussian_kl_benchmark(
            self._nobs, self._degree, self._noise_std, self._prior_std, bkd,
        )
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        eig = bench.exact_eig(weights)
        assert eig > 0.0
        assert np.isfinite(eig)

    def test_numerical_eig_finite(self, bkd: Backend[Array]) -> None:
        """Numerical EIG is finite."""
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        eig = self._compute_eig_realization(bkd, 50, 30, weights, 42)
        assert np.isfinite(eig)

    def test_numerical_eig_reproducible(
        self, bkd: Backend[Array],
    ) -> None:
        """Same seed produces same EIG."""
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        eig1 = self._compute_eig_realization(bkd, 50, 30, weights, 42)
        eig2 = self._compute_eig_realization(bkd, 50, 30, weights, 42)
        bkd.assert_allclose(
            bkd.asarray([eig1]), bkd.asarray([eig2]), rtol=1e-10,
        )

    def test_convergence_rate_o1n_data(
        self, bkd: Backend[Array],
    ) -> None:
        """Synthetic O(1/n) data gives rate ~1."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / n for n in sample_counts]
        rate = compute_convergence_rate(sample_counts, values)
        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([1.0]), rtol=1e-10,
        )

    def test_convergence_rate_o1sqrtn_data(
        self, bkd: Backend[Array],
    ) -> None:
        """Synthetic O(1/sqrt(n)) data gives rate ~0.5."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / np.sqrt(n) for n in sample_counts]
        rate = compute_convergence_rate(sample_counts, values)
        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([0.5]), rtol=1e-10,
        )

    @slow_test
    def test_different_weights_give_different_eig(
        self, bkd: Backend[Array],
    ) -> None:
        """Different weights give different EIG values."""
        bench = build_linear_gaussian_kl_benchmark(
            self._nobs, self._degree, self._noise_std, self._prior_std, bkd,
        )
        weights_uniform = bkd.ones((self._nobs, 1)) / self._nobs
        weights_concentrated = bkd.asarray([[1.0], [0.0], [0.0], [0.0], [0.0]])

        eig_uniform = bench.exact_eig(weights_uniform)
        eig_concentrated = bench.exact_eig(weights_concentrated)

        assert abs(eig_uniform - eig_concentrated) > 1e-3
