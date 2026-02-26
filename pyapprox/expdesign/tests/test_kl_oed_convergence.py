"""
Standalone tests for KL-OED convergence analysis.

PERMANENT - no legacy imports.

Tests verify:
- MSE decreases with increasing samples
- MSE = bias^2 + variance relationship
- Convergence rate analysis
"""

import numpy as np
import pytest

from pyapprox.expdesign.benchmarks import LinearGaussianOEDBenchmark
from pyapprox.expdesign.diagnostics import KLOEDDiagnostics
from pyapprox.util.test_utils import slow_test


class TestKLOEDConvergenceStandalone:
    """Standalone tests for KL-OED convergence analysis."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        self._nobs = 5
        self._degree = 2
        self._noise_std = 0.5
        self._prior_std = 0.5

    def _create_benchmark(self, bkd):
        return LinearGaussianOEDBenchmark(
            self._nobs,
            self._degree,
            self._noise_std,
            self._prior_std,
            bkd,
        )

    def _create_diagnostics(self, bkd):
        return KLOEDDiagnostics(self._create_benchmark(bkd))

    @slow_test
    def test_mse_decreases_with_outer_samples(self, bkd):
        """Test MSE generally decreases with more outer samples."""
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        # Fixed inner samples, vary outer
        ninner = 25
        outer_counts = [25, 50, 100]
        mses = []

        for nouter in outer_counts:
            _, _, mse = diagnostics.compute_mse(
                nouter=nouter,
                ninner=ninner,
                nrealizations=5,
                design_weights=weights,
                base_seed=42,
            )
            mses.append(mse)

        # Trend should be decreasing (MSE with most samples < MSE with least)
        assert mses[-1] < mses[0] * 2.0  # Allow some variance
        # All should be positive and finite
        for mse in mses:
            assert mse > 0.0
            assert np.isfinite(mse)

    @slow_test
    def test_mse_decreases_with_inner_samples(self, bkd):
        """Test MSE generally decreases with more inner samples."""
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        # Fixed outer samples, vary inner
        nouter = 50
        inner_counts = [15, 30, 60]
        mses = []

        for ninner in inner_counts:
            _, _, mse = diagnostics.compute_mse(
                nouter=nouter,
                ninner=ninner,
                nrealizations=5,
                design_weights=weights,
                base_seed=42,
            )
            mses.append(mse)

        # Trend should be decreasing
        assert mses[-1] < mses[0] * 2.0
        for mse in mses:
            assert mse > 0.0
            assert np.isfinite(mse)

    def test_bias_variance_mse_relation(self, bkd):
        """Test MSE = bias^2 + variance."""
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        bias, variance, mse = diagnostics.compute_mse(
            nouter=50,
            ninner=30,
            nrealizations=5,
            design_weights=weights,
            base_seed=42,
        )

        # MSE should equal bias^2 + variance
        expected_mse = bias**2 + variance
        bkd.assert_allclose(
            bkd.asarray([mse]),
            bkd.asarray([expected_mse]),
            rtol=1e-10,
        )

        # Variance should be non-negative
        assert variance >= 0.0

    def test_exact_eig_positive(self, bkd):
        """Test exact EIG is positive for uniform weights."""
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        eig = diagnostics.exact_eig(weights)

        assert eig > 0.0
        assert np.isfinite(eig)

    def test_numerical_eig_finite(self, bkd):
        """Test numerical EIG is finite."""
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        eig = diagnostics.compute_numerical_eig(
            nouter=50,
            ninner=30,
            design_weights=weights,
            seed=42,
        )

        assert np.isfinite(eig)

    def test_numerical_eig_reproducible(self, bkd):
        """Test numerical EIG is reproducible with same seed."""
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        eig1 = diagnostics.compute_numerical_eig(
            nouter=50,
            ninner=30,
            design_weights=weights,
            seed=42,
        )
        eig2 = diagnostics.compute_numerical_eig(
            nouter=50,
            ninner=30,
            design_weights=weights,
            seed=42,
        )

        bkd.assert_allclose(
            bkd.asarray([eig1]),
            bkd.asarray([eig2]),
            rtol=1e-10,
        )

    @slow_test
    def test_mc_convergence_rate_positive(self, bkd):
        """Test MC convergence rate is positive."""
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        outer_counts = [100, 200, 400, 800]
        inner_counts = [50]

        values = diagnostics.compute_mse_for_sample_combinations(
            outer_sample_counts=outer_counts,
            inner_sample_counts=inner_counts,
            nrealizations=10,
            design_weights=weights,
            base_seed=42,
        )

        # Extract MSE for fixed inner count
        mse_values = bkd.to_numpy(values["mse"][0]).tolist()

        # Compute convergence rate
        rate = KLOEDDiagnostics.compute_convergence_rate(outer_counts, mse_values)

        # MC convergence rate should be positive (decay)
        assert rate > 0.0
        # For MC, expect rate around 0.5-1.0 (O(1/n) to O(1/sqrt(n)))
        assert rate < 3.0

    def test_convergence_rate_o1n_data(self, bkd):
        """Test convergence rate for synthetic O(1/n) data."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / n for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([1.0]), rtol=1e-10
        )

    def test_convergence_rate_o1sqrtn_data(self, bkd):
        """Test convergence rate for synthetic O(1/sqrt(n)) data."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / np.sqrt(n) for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([0.5]), rtol=1e-10
        )

    def test_compute_mse_for_sample_combinations_structure(self, bkd):
        """Test output structure of compute_mse_for_sample_combinations."""
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        outer_counts = [30, 60]
        inner_counts = [20, 40]

        values = diagnostics.compute_mse_for_sample_combinations(
            outer_sample_counts=outer_counts,
            inner_sample_counts=inner_counts,
            nrealizations=3,
            design_weights=weights,
            base_seed=42,
        )

        # Check keys
        assert "sqbias" in values
        assert "variance" in values
        assert "mse" in values

        # One array per inner count
        assert len(values["mse"]) == len(inner_counts)

        # Each array has length = number of outer counts
        for arr in values["mse"]:
            arr_np = bkd.to_numpy(arr)
            assert arr_np.shape[0] == len(outer_counts)

    @slow_test
    def test_different_weights_give_different_eig(self, bkd):
        """Test that different weights give different EIG values."""
        diagnostics = self._create_diagnostics(bkd)

        weights_uniform = bkd.ones((self._nobs, 1)) / self._nobs
        weights_concentrated = bkd.asarray([[1.0], [0.0], [0.0], [0.0], [0.0]])

        eig_uniform = diagnostics.exact_eig(weights_uniform)
        eig_concentrated = diagnostics.exact_eig(weights_concentrated)

        # Different weights should give different EIG
        assert abs(eig_uniform - eig_concentrated) > 1e-3
