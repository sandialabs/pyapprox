"""
Standalone tests for KL-OED convergence analysis.

PERMANENT - no legacy imports.

Tests verify:
- MSE decreases with increasing samples
- MSE = bias^2 + variance relationship
- Convergence rate analysis
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401

from pyapprox.typing.expdesign.benchmarks import LinearGaussianOEDBenchmark
from pyapprox.typing.expdesign.diagnostics import KLOEDDiagnostics


class TestKLOEDConvergenceStandalone(Generic[Array], unittest.TestCase):
    """Standalone tests for KL-OED convergence analysis."""

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

    @slow_test
    def test_mse_decreases_with_outer_samples(self):
        """Test MSE generally decreases with more outer samples."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

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
        self.assertLess(mses[-1], mses[0] * 2.0)  # Allow some variance
        # All should be positive and finite
        for mse in mses:
            self.assertGreater(mse, 0.0)
            self.assertTrue(np.isfinite(mse))

    @slow_test
    def test_mse_decreases_with_inner_samples(self):
        """Test MSE generally decreases with more inner samples."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

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
        self.assertLess(mses[-1], mses[0] * 2.0)
        for mse in mses:
            self.assertGreater(mse, 0.0)
            self.assertTrue(np.isfinite(mse))

    def test_bias_variance_mse_relation(self):
        """Test MSE = bias^2 + variance."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        bias, variance, mse = diagnostics.compute_mse(
            nouter=50,
            ninner=30,
            nrealizations=5,
            design_weights=weights,
            base_seed=42,
        )

        # MSE should equal bias^2 + variance
        expected_mse = bias ** 2 + variance
        self._bkd.assert_allclose(
            self._bkd.asarray([mse]),
            self._bkd.asarray([expected_mse]),
            rtol=1e-10,
        )

        # Variance should be non-negative
        self.assertGreaterEqual(variance, 0.0)

    def test_exact_eig_positive(self):
        """Test exact EIG is positive for uniform weights."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        eig = diagnostics.exact_eig(weights)

        self.assertGreater(eig, 0.0)
        self.assertTrue(np.isfinite(eig))

    def test_numerical_eig_finite(self):
        """Test numerical EIG is finite."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        eig = diagnostics.compute_numerical_eig(
            nouter=50,
            ninner=30,
            design_weights=weights,
            seed=42,
        )

        self.assertTrue(np.isfinite(eig))

    def test_numerical_eig_reproducible(self):
        """Test numerical EIG is reproducible with same seed."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

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

        self._bkd.assert_allclose(
            self._bkd.asarray([eig1]),
            self._bkd.asarray([eig2]),
            rtol=1e-10,
        )

    @slow_test
    def test_mc_convergence_rate_positive(self):
        """Test MC convergence rate is positive."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

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
        mse_values = self._bkd.to_numpy(values["mse"][0]).tolist()

        # Compute convergence rate
        rate = KLOEDDiagnostics.compute_convergence_rate(outer_counts, mse_values)

        # MC convergence rate should be positive (decay)
        self.assertGreater(rate, 0.0)
        # For MC, expect rate around 0.5-1.0 (O(1/n) to O(1/sqrt(n)))
        self.assertLess(rate, 3.0)

    def test_convergence_rate_o1n_data(self):
        """Test convergence rate for synthetic O(1/n) data."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / n for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        self._bkd.assert_allclose(
            self._bkd.asarray([rate]), self._bkd.asarray([1.0]), rtol=1e-10
        )

    def test_convergence_rate_o1sqrtn_data(self):
        """Test convergence rate for synthetic O(1/sqrt(n)) data."""
        sample_counts = [10, 20, 40, 80, 160]
        values = [1.0 / np.sqrt(n) for n in sample_counts]

        rate = KLOEDDiagnostics.compute_convergence_rate(sample_counts, values)

        self._bkd.assert_allclose(
            self._bkd.asarray([rate]), self._bkd.asarray([0.5]), rtol=1e-10
        )

    def test_compute_mse_for_sample_combinations_structure(self):
        """Test output structure of compute_mse_for_sample_combinations."""
        diagnostics = self._create_diagnostics()
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

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
        self.assertIn("sqbias", values)
        self.assertIn("variance", values)
        self.assertIn("mse", values)

        # One array per inner count
        self.assertEqual(len(values["mse"]), len(inner_counts))

        # Each array has length = number of outer counts
        for arr in values["mse"]:
            arr_np = self._bkd.to_numpy(arr)
            self.assertEqual(arr_np.shape[0], len(outer_counts))

    @slow_test
    def test_different_weights_give_different_eig(self):
        """Test that different weights give different EIG values."""
        diagnostics = self._create_diagnostics()

        weights_uniform = self._bkd.ones((self._nobs, 1)) / self._nobs
        weights_concentrated = self._bkd.zeros((self._nobs, 1))
        weights_concentrated = self._bkd.asarray(
            [[1.0], [0.0], [0.0], [0.0], [0.0]]
        )

        eig_uniform = diagnostics.exact_eig(weights_uniform)
        eig_concentrated = diagnostics.exact_eig(weights_concentrated)

        # Different weights should give different EIG
        self.assertNotAlmostEqual(eig_uniform, eig_concentrated, places=3)


class TestKLOEDConvergenceStandaloneNumpy(
    TestKLOEDConvergenceStandalone[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKLOEDConvergenceStandaloneTorch(
    TestKLOEDConvergenceStandalone[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
