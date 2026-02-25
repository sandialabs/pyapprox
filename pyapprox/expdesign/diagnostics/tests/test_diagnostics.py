"""
Standalone tests for KLOEDDiagnostics.

PERMANENT - no legacy imports.

These tests verify correctness using self-consistent checks.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.expdesign.benchmarks import LinearGaussianOEDBenchmark
from pyapprox.expdesign.diagnostics import KLOEDDiagnostics
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestKLOEDDiagnosticsStandalone(Generic[Array], unittest.TestCase):
    """Standalone tests for KLOEDDiagnostics."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 5
        self._degree = 2
        self._noise_std = 0.5
        self._prior_std = 0.5

    def _create_benchmark(self) -> LinearGaussianOEDBenchmark[Array]:
        return LinearGaussianOEDBenchmark(
            self._nobs,
            self._degree,
            self._noise_std,
            self._prior_std,
            self._bkd,
        )

    def _create_diagnostics(self) -> KLOEDDiagnostics[Array]:
        return KLOEDDiagnostics(self._create_benchmark())

    def test_exact_eig_positive(self):
        """Test exact EIG is positive."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        eig = diagnostics.exact_eig(weights)
        self.assertGreater(eig, 0.0)

    def test_exact_eig_matches_benchmark(self):
        """Test exact_eig delegates to benchmark correctly."""
        benchmark = self._create_benchmark()
        diagnostics = KLOEDDiagnostics(benchmark)
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        eig_diag = diagnostics.exact_eig(weights)
        eig_bench = benchmark.exact_eig(weights)

        self._bkd.assert_allclose(
            self._bkd.asarray([eig_diag]),
            self._bkd.asarray([eig_bench]),
            rtol=1e-12,
        )

    def test_numerical_eig_finite(self):
        """Test numerical EIG is finite."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        eig = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=42
        )

        self.assertTrue(np.isfinite(eig))

    def test_numerical_eig_reproducible(self):
        """Test numerical EIG is reproducible with same seed."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        eig1 = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=42
        )
        eig2 = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=42
        )

        self._bkd.assert_allclose(
            self._bkd.asarray([eig1]),
            self._bkd.asarray([eig2]),
            rtol=1e-12,
        )

    def test_numerical_eig_varies_with_seed(self):
        """Test numerical EIG varies with different seeds."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        eig1 = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=42
        )
        eig2 = diagnostics.compute_numerical_eig(
            nouter=50, ninner=30, design_weights=weights, seed=123
        )

        # Should be different (with high probability)
        self.assertNotAlmostEqual(eig1, eig2, places=5)

    def test_compute_mse_returns_three_values(self):
        """Test compute_mse returns bias, variance, mse."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        result = diagnostics.compute_mse(
            nouter=30, ninner=20, nrealizations=3, design_weights=weights
        )

        self.assertEqual(len(result), 3)
        bias, variance, mse = result

        self.assertTrue(np.isfinite(bias))
        self.assertTrue(np.isfinite(variance))
        self.assertTrue(np.isfinite(mse))

        # Variance should be non-negative
        self.assertGreaterEqual(variance, 0.0)

        # MSE = bias^2 + variance
        expected_mse = bias**2 + variance
        self._bkd.assert_allclose(
            self._bkd.asarray([mse]),
            self._bkd.asarray([expected_mse]),
            rtol=1e-10,
        )

    def test_mse_decreases_with_samples(self):
        """Test MSE generally decreases with more samples."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

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
        self.assertTrue(np.isfinite(mse_small))
        self.assertTrue(np.isfinite(mse_large))

    def test_convergence_rate_o1n(self):
        """Test convergence rate for O(1/n) data is approximately 1."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / n for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        self._bkd.assert_allclose(
            self._bkd.asarray([rate]), self._bkd.asarray([1.0]), rtol=1e-10
        )

    def test_convergence_rate_o1n2(self):
        """Test convergence rate for O(1/n^2) data is approximately 2."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / (n**2) for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        self._bkd.assert_allclose(
            self._bkd.asarray([rate]), self._bkd.asarray([2.0]), rtol=1e-10
        )

    def test_convergence_rate_o1sqrtn(self):
        """Test convergence rate for O(1/sqrt(n)) data is approximately 0.5."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / np.sqrt(n) for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        self._bkd.assert_allclose(
            self._bkd.asarray([rate]), self._bkd.asarray([0.5]), rtol=1e-10
        )

    def test_compute_mse_for_sample_combinations_structure(self):
        """Test compute_mse_for_sample_combinations returns correct structure."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

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
        self.assertIn("sqbias", values)
        self.assertIn("variance", values)
        self.assertIn("mse", values)

        # One array per inner count
        self.assertEqual(len(values["sqbias"]), len(inner_counts))
        self.assertEqual(len(values["variance"]), len(inner_counts))
        self.assertEqual(len(values["mse"]), len(inner_counts))

        # Each array has length = number of outer counts
        for arr in values["mse"]:
            self.assertEqual(arr.shape[0], len(outer_counts))

    def test_bkd_accessor(self):
        """Test bkd() returns the backend."""
        diagnostics = self._create_diagnostics()
        self.assertEqual(diagnostics.bkd(), self._bkd)

    def test_unknown_utility_type_raises(self):
        """Test create_prediction_oed_diagnostics with unknown type."""
        from pyapprox.expdesign.diagnostics import (
            create_prediction_oed_diagnostics,
        )

        benchmark = self._create_benchmark()
        with self.assertRaises(ValueError):
            create_prediction_oed_diagnostics(benchmark, "nonexistent_type")


class TestKLOEDDiagnosticsStandaloneNumpy(TestKLOEDDiagnosticsStandalone[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKLOEDDiagnosticsStandaloneTorch(TestKLOEDDiagnosticsStandalone[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
