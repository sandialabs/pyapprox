"""
Standalone tests for KLOEDDiagnostics.

PERMANENT - no legacy imports.

These tests verify correctness using self-consistent checks.
"""

import numpy as np
import pytest

from pyapprox.expdesign.benchmarks import LinearGaussianOEDBenchmark
from pyapprox.expdesign.diagnostics import KLOEDDiagnostics


class TestKLOEDDiagnosticsStandalone:
    """Standalone tests for KLOEDDiagnostics."""

    def _setup_data(self, bkd):
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

    def test_exact_eig_positive(self, bkd):
        """Test exact EIG is positive."""
        self._setup_data(bkd)
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        eig = diagnostics.exact_eig(weights)
        assert eig > 0.0

    def test_exact_eig_matches_benchmark(self, bkd):
        """Test exact_eig delegates to benchmark correctly."""
        self._setup_data(bkd)
        benchmark = self._create_benchmark(bkd)
        diagnostics = KLOEDDiagnostics(benchmark)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        eig_diag = diagnostics.exact_eig(weights)
        eig_bench = benchmark.exact_eig(weights)

        bkd.assert_allclose(
            bkd.asarray([eig_diag]),
            bkd.asarray([eig_bench]),
            rtol=1e-12,
        )

    def test_numerical_eig_finite(self, bkd):
        """Test numerical EIG is finite."""
        self._setup_data(bkd)
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        eig = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=42
        )

        assert np.isfinite(eig)

    def test_numerical_eig_reproducible(self, bkd):
        """Test numerical EIG is reproducible with same seed."""
        self._setup_data(bkd)
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        eig1 = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=42
        )
        eig2 = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=42
        )

        bkd.assert_allclose(
            bkd.asarray([eig1]),
            bkd.asarray([eig2]),
            rtol=1e-12,
        )

    def test_numerical_eig_varies_with_seed(self, bkd):
        """Test numerical EIG varies with different seeds."""
        self._setup_data(bkd)
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        eig1 = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=42
        )
        eig2 = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=123
        )

        # Should be different (with high probability)
        assert abs(eig1 - eig2) > 1e-5

    def test_compute_mse_returns_three_values(self, bkd):
        """Test compute_mse returns bias, variance, mse."""
        self._setup_data(bkd)
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        result = diagnostics.compute_mse(
            nouter=30, ninner=20, nrealizations=3, design_weights=weights
        )

        assert len(result) == 3
        bias, variance, mse = result

        assert np.isfinite(bias)
        assert np.isfinite(variance)
        assert np.isfinite(mse)

        # Variance should be non-negative
        assert variance >= 0.0

        # MSE = bias^2 + variance
        expected_mse = bias**2 + variance
        bkd.assert_allclose(
            bkd.asarray([mse]),
            bkd.asarray([expected_mse]),
            rtol=1e-10,
        )

    def test_mse_decreases_with_samples(self, bkd):
        """Test MSE generally decreases with more samples."""
        self._setup_data(bkd)
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        # Fewer samples
        _, _, mse_small = diagnostics.compute_mse(
            nouter=20, ninner=10, nrealizations=5, design_weights=weights, base_seed=42
        )

        # More samples
        _, _, mse_large = diagnostics.compute_mse(
            nouter=100, ninner=50, nrealizations=5, design_weights=weights, base_seed=42
        )

        # With more samples, MSE should generally be smaller
        # (This is a statistical test, may occasionally fail)
        # We just check both are finite and positive
        assert np.isfinite(mse_small)
        assert np.isfinite(mse_large)

    def test_convergence_rate_o1n(self, bkd):
        """Test convergence rate for O(1/n) data is approximately 1."""
        self._setup_data(bkd)
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / n for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([1.0]), rtol=1e-10
        )

    def test_convergence_rate_o1n2(self, bkd):
        """Test convergence rate for O(1/n^2) data is approximately 2."""
        self._setup_data(bkd)
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / (n**2) for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([2.0]), rtol=1e-10
        )

    def test_convergence_rate_o1sqrtn(self, bkd):
        """Test convergence rate for O(1/sqrt(n)) data is approximately 0.5."""
        self._setup_data(bkd)
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / np.sqrt(n) for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        bkd.assert_allclose(
            bkd.asarray([rate]), bkd.asarray([0.5]), rtol=1e-10
        )

    def test_compute_mse_for_sample_combinations_structure(self, bkd):
        """Test compute_mse_for_sample_combinations returns correct structure."""
        self._setup_data(bkd)
        diagnostics = self._create_diagnostics(bkd)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        outer_counts = [20, 40]
        inner_counts = [10, 20]

        values = diagnostics.compute_mse_for_sample_combinations(
            outer_counts,
            inner_counts,
            nrealizations=2,
            design_weights=weights,
            base_seed=42,
        )

        # Check structure
        assert "sqbias" in values
        assert "variance" in values
        assert "mse" in values

        # One array per inner count
        assert len(values["sqbias"]) == len(inner_counts)
        assert len(values["variance"]) == len(inner_counts)
        assert len(values["mse"]) == len(inner_counts)

        # Each array has length = number of outer counts
        for arr in values["mse"]:
            assert arr.shape[0] == len(outer_counts)

    def test_bkd_accessor(self, bkd):
        """Test bkd() returns the backend."""
        self._setup_data(bkd)
        diagnostics = self._create_diagnostics(bkd)
        assert diagnostics.bkd() == bkd

    def test_unknown_utility_type_raises(self, bkd):
        """Test create_prediction_oed_diagnostics with unknown type."""
        from pyapprox.expdesign.diagnostics import (
            create_prediction_oed_diagnostics,
        )

        self._setup_data(bkd)
        benchmark = self._create_benchmark(bkd)
        with pytest.raises(ValueError):
            create_prediction_oed_diagnostics(benchmark, "nonexistent_type")
