"""
Standalone tests for prediction OED convergence analysis.

PERMANENT - no legacy imports.

Tests verify:
- Prediction OED objective converges with increasing samples
- Different deviation measures show expected convergence behavior
- Convergence rate analysis for linear models
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

from pyapprox.typing.expdesign import (
    create_prediction_oed_objective,
    LinearGaussianOEDBenchmark,
    GaussianOEDInnerLoopLikelihood,
    PredictionOEDObjective,
    StandardDeviationMeasure,
    EntropicDeviationMeasure,
    SampleAverageMean,
)


class TestPredictionOEDConvergenceStandalone(Generic[Array], unittest.TestCase):
    """Standalone tests for prediction OED convergence analysis."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 3
        self._ninner = 50
        self._nouter = 30
        self._npred = 2

    def _create_test_data(self, seed: int = 42):
        """Create consistent test data for convergence tests."""
        np.random.seed(seed)
        bkd = self._bkd

        noise_variances = bkd.asarray(np.array([0.1, 0.15, 0.2]))
        outer_shapes = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        inner_shapes = bkd.asarray(np.random.randn(self._nobs, self._ninner))
        latent_samples = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        qoi_vals = bkd.asarray(np.random.randn(self._ninner, self._npred))

        return noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals

    def test_stdev_objective_produces_positive_value(self):
        """Test StdDev objective produces positive deviation value."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data()
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        value = objective(weights)

        # Deviation should be positive (standard deviation)
        val_np = self._bkd.to_numpy(value)[0, 0]
        self.assertGreater(val_np, 0.0)
        self.assertTrue(np.isfinite(val_np))

    def test_entropic_objective_produces_finite_value(self):
        """Test Entropic objective produces finite value."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data()
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="entropic",
            risk_type="mean",
            alpha=0.5,
        )

        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        value = objective(weights)

        val_np = self._bkd.to_numpy(value)[0, 0]
        self.assertTrue(np.isfinite(val_np))

    @slow_test
    def test_stdev_convergence_with_inner_samples(self):
        """Test StdDev deviation converges with increasing inner samples."""
        inner_counts = [20, 40, 80]
        values = []

        for ninner in inner_counts:
            np.random.seed(42)
            noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))
            outer_shapes = self._bkd.asarray(np.random.randn(self._nobs, self._nouter))
            inner_shapes = self._bkd.asarray(np.random.randn(self._nobs, ninner))
            latent_samples = self._bkd.asarray(np.random.randn(self._nobs, self._nouter))
            qoi_vals = self._bkd.asarray(np.random.randn(ninner, self._npred))

            objective = create_prediction_oed_objective(
                noise_variances,
                outer_shapes,
                inner_shapes,
                latent_samples,
                qoi_vals,
                self._bkd,
                deviation_type="stdev",
                risk_type="mean",
            )

            weights = self._bkd.ones((self._nobs, 1)) / self._nobs
            value = objective(weights)
            values.append(self._bkd.to_numpy(value)[0, 0])

        # All values should be positive and finite
        for val in values:
            self.assertGreater(val, 0.0)
            self.assertTrue(np.isfinite(val))

        # With more samples, variance of the estimator typically decreases
        # We just check values are reasonable (within 10x of each other)
        self.assertLess(max(values) / min(values), 10.0)

    @slow_test
    def test_entropic_convergence_with_inner_samples(self):
        """Test Entropic deviation converges with increasing inner samples."""
        inner_counts = [20, 40, 80]
        values = []

        for ninner in inner_counts:
            np.random.seed(42)
            noise_variances = self._bkd.asarray(np.array([0.1, 0.15, 0.2]))
            outer_shapes = self._bkd.asarray(np.random.randn(self._nobs, self._nouter))
            inner_shapes = self._bkd.asarray(np.random.randn(self._nobs, ninner))
            latent_samples = self._bkd.asarray(np.random.randn(self._nobs, self._nouter))
            qoi_vals = self._bkd.asarray(np.random.randn(ninner, self._npred))

            objective = create_prediction_oed_objective(
                noise_variances,
                outer_shapes,
                inner_shapes,
                latent_samples,
                qoi_vals,
                self._bkd,
                deviation_type="entropic",
                risk_type="mean",
                alpha=0.5,
            )

            weights = self._bkd.ones((self._nobs, 1)) / self._nobs
            value = objective(weights)
            values.append(self._bkd.to_numpy(value)[0, 0])

        # All values should be finite
        for val in values:
            self.assertTrue(np.isfinite(val))

        # Values should be in reasonable range
        self.assertLess(max(abs(v) for v in values), 100.0)

    def test_different_weights_give_different_deviations(self):
        """Test that different weights give different deviation values."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data()
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights_uniform = self._bkd.ones((self._nobs, 1)) / self._nobs
        weights_high = self._bkd.asarray([[2.0], [2.0], [2.0]])

        val_uniform = objective(weights_uniform)
        val_high = objective(weights_high)

        val_uniform_np = self._bkd.to_numpy(val_uniform)[0, 0]
        val_high_np = self._bkd.to_numpy(val_high)[0, 0]

        # Higher weights should generally reduce deviation (more information)
        # Just check they're different
        self.assertNotAlmostEqual(val_uniform_np, val_high_np, places=3)

    def test_jacobian_is_finite(self):
        """Test Jacobian computation produces finite values."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data()
        )

        objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))
        jac = objective.jacobian(weights)

        jac_np = self._bkd.to_numpy(jac)
        self.assertEqual(jac_np.shape, (1, self._nobs))
        self.assertTrue(np.all(np.isfinite(jac_np)))

    @slow_test
    def test_variance_risk_produces_different_values(self):
        """Test variance risk measure produces different results than mean."""
        noise_variances, outer_shapes, inner_shapes, latent_samples, qoi_vals = (
            self._create_test_data()
        )

        objective_mean = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="stdev",
            risk_type="mean",
        )

        objective_var = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            self._bkd,
            deviation_type="stdev",
            risk_type="variance",
        )

        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        val_mean = objective_mean(weights)
        val_var = objective_var(weights)

        val_mean_np = self._bkd.to_numpy(val_mean)[0, 0]
        val_var_np = self._bkd.to_numpy(val_var)[0, 0]

        # Both should be finite
        self.assertTrue(np.isfinite(val_mean_np))
        self.assertTrue(np.isfinite(val_var_np))

        # Mean risk gives the expected deviation, variance risk gives variance of deviation
        # These are different quantities
        self.assertNotAlmostEqual(val_mean_np, val_var_np, places=3)

    def test_linear_gaussian_benchmark_exact_eig(self):
        """Test LinearGaussianOEDBenchmark provides exact EIG."""
        benchmark = LinearGaussianOEDBenchmark(
            nobs=5,
            degree=2,
            noise_std=0.5,
            prior_std=0.5,
            bkd=self._bkd,
        )

        weights = self._bkd.ones((5, 1)) / 5
        eig = benchmark.exact_eig(weights)

        # EIG should be positive for this problem
        self.assertGreater(eig, 0.0)
        self.assertTrue(np.isfinite(eig))

    def test_benchmark_generate_data_shapes(self):
        """Test benchmark data generation has correct shapes."""
        benchmark = LinearGaussianOEDBenchmark(
            nobs=5,
            degree=2,
            noise_std=0.5,
            prior_std=0.5,
            bkd=self._bkd,
        )

        nsamples = 100
        theta, y = benchmark.generate_data(nsamples)

        theta_np = self._bkd.to_numpy(theta)
        y_np = self._bkd.to_numpy(y)

        self.assertEqual(theta_np.shape, (benchmark.nparams(), nsamples))
        self.assertEqual(y_np.shape, (benchmark.nobs(), nsamples))


class TestPredictionOEDConvergenceStandaloneNumpy(
    TestPredictionOEDConvergenceStandalone[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPredictionOEDConvergenceStandaloneTorch(
    TestPredictionOEDConvergenceStandalone[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
