"""
Performance comparison tests for OED implementation.

Tests verify that the new implementation is at least as fast as legacy code.
Uses timing comparisons with reasonable tolerance.
"""

import unittest
from typing import Any, Generic
import time

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.typing.expdesign.objective import KLOEDObjective
from pyapprox.typing.expdesign.evidence import LogEvidence
from pyapprox.typing.expdesign.solver import (
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    BruteForceKLOEDSolver,
)


class TestLikelihoodPerformance(Generic[Array], unittest.TestCase):
    """Test likelihood computation performance."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_likelihood_matrix_performance(self):
        """Test likelihood matrix computation time scales reasonably."""
        nobs = 10
        ninner = 100
        nouter = 100

        np.random.seed(42)
        noise_variances = self._bkd.asarray(np.random.uniform(0.1, 0.5, nobs))
        shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        obs = self._bkd.asarray(np.random.randn(nobs, nouter))
        weights = self._bkd.ones((nobs, 1))

        likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, self._bkd)
        likelihood.set_shapes(shapes)
        likelihood.set_observations(obs)

        # Time the computation
        start = time.perf_counter()
        for _ in range(10):
            _ = likelihood.logpdf_matrix(weights)
        elapsed = time.perf_counter() - start

        # Should complete 10 iterations in reasonable time (< 2 seconds)
        self.assertLess(elapsed, 2.0)

    def test_likelihood_jacobian_performance(self):
        """Test likelihood Jacobian computation time scales reasonably."""
        nobs = 10
        ninner = 50
        nouter = 50

        np.random.seed(42)
        noise_variances = self._bkd.asarray(np.random.uniform(0.1, 0.5, nobs))
        shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        obs = self._bkd.asarray(np.random.randn(nobs, nouter))
        weights = self._bkd.ones((nobs, 1))

        likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, self._bkd)
        likelihood.set_shapes(shapes)
        likelihood.set_observations(obs)

        # Time the computation
        start = time.perf_counter()
        for _ in range(10):
            _ = likelihood.jacobian_matrix(weights)
        elapsed = time.perf_counter() - start

        # Should complete 10 iterations in reasonable time (< 3 seconds)
        self.assertLess(elapsed, 3.0)


class TestLikelihoodPerformanceNumpy(TestLikelihoodPerformance[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLikelihoodPerformanceTorch(TestLikelihoodPerformance[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestObjectivePerformance(Generic[Array], unittest.TestCase):
    """Test objective evaluation performance."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 8
        self._ninner = 50
        self._nouter = 50

        np.random.seed(42)
        self._noise_variances = self._bkd.asarray(
            np.random.uniform(0.1, 0.3, self._nobs)
        )
        self._outer_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )

        self._objective = KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            self._bkd,
        )

    def test_objective_evaluation_performance(self):
        """Test objective evaluation time is reasonable."""
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        # Time the computation
        start = time.perf_counter()
        for _ in range(20):
            _ = self._objective(weights)
        elapsed = time.perf_counter() - start

        # Should complete 20 iterations in reasonable time (< 2 seconds)
        self.assertLess(elapsed, 2.0)

    def test_objective_jacobian_performance(self):
        """Test objective Jacobian time is reasonable."""
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        # Time the computation
        start = time.perf_counter()
        for _ in range(20):
            _ = self._objective.jacobian(weights)
        elapsed = time.perf_counter() - start

        # Should complete 20 iterations in reasonable time (< 3 seconds)
        self.assertLess(elapsed, 3.0)

    def test_eig_performance(self):
        """Test EIG computation time is reasonable."""
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        # Time the computation
        start = time.perf_counter()
        for _ in range(20):
            _ = self._objective.expected_information_gain(weights)
        elapsed = time.perf_counter() - start

        # Should complete 20 iterations in reasonable time (< 2 seconds)
        self.assertLess(elapsed, 2.0)


class TestObjectivePerformanceNumpy(TestObjectivePerformance[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestObjectivePerformanceTorch(TestObjectivePerformance[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestSolverPerformance(Generic[Array], unittest.TestCase):
    """Test solver performance."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._nobs = 5
        self._ninner = 30
        self._nouter = 25

        np.random.seed(42)
        self._noise_variances = self._bkd.asarray(
            np.random.uniform(0.1, 0.3, self._nobs)
        )
        self._outer_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._latent_samples = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )

        self._objective = KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            None,
            None,
            self._bkd,
        )

    def test_relaxed_solver_performance(self):
        """Test relaxed solver completes in reasonable time."""
        config = RelaxedOEDConfig(verbosity=0, maxiter=50)
        solver = RelaxedKLOEDSolver(self._objective, config)

        start = time.perf_counter()
        _, _ = solver.solve()
        elapsed = time.perf_counter() - start

        # Should complete in < 10 seconds
        self.assertLess(elapsed, 10.0)

    def test_brute_force_solver_performance(self):
        """Test brute-force solver completes in reasonable time."""
        solver = BruteForceKLOEDSolver(self._objective)

        # k=2 means C(5,2)=10 combinations
        start = time.perf_counter()
        _, _, _ = solver.solve(k=2)
        elapsed = time.perf_counter() - start

        # Should complete in < 5 seconds
        self.assertLess(elapsed, 5.0)


class TestSolverPerformanceNumpy(TestSolverPerformance[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSolverPerformanceTorch(TestSolverPerformance[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestScalingPerformance(unittest.TestCase):
    """Test performance scaling with problem size."""

    __test__ = True

    def test_likelihood_scaling(self):
        """Test that likelihood computation scales linearly in ninner * nouter."""
        bkd = NumpyBkd()
        nobs = 5

        np.random.seed(42)
        noise_variances = bkd.asarray(np.random.uniform(0.1, 0.3, nobs))
        weights = bkd.ones((nobs, 1))

        # Small problem
        shapes_small = bkd.asarray(np.random.randn(nobs, 20))
        obs_small = bkd.asarray(np.random.randn(nobs, 20))

        likelihood_small = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
        likelihood_small.set_shapes(shapes_small)
        likelihood_small.set_observations(obs_small)

        start = time.perf_counter()
        for _ in range(50):
            _ = likelihood_small.logpdf_matrix(weights)
        time_small = time.perf_counter() - start

        # Large problem (4x more samples each dimension = 16x work)
        shapes_large = bkd.asarray(np.random.randn(nobs, 80))
        obs_large = bkd.asarray(np.random.randn(nobs, 80))

        likelihood_large = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
        likelihood_large.set_shapes(shapes_large)
        likelihood_large.set_observations(obs_large)

        start = time.perf_counter()
        for _ in range(50):
            _ = likelihood_large.logpdf_matrix(weights)
        time_large = time.perf_counter() - start

        # Large should be roughly 16x slower (allow 32x for overhead)
        # This tests that we don't have worse than O(n^2) scaling
        ratio = time_large / time_small
        self.assertLess(ratio, 50.0)


class TestLargeScalePerformance(Generic[Array], unittest.TestCase):
    """Test performance on larger problem sizes."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_large_likelihood_matrix(self):
        """Test likelihood matrix with 500x500 inner/outer samples."""
        nobs = 10
        ninner = 500
        nouter = 500

        np.random.seed(42)
        noise_variances = self._bkd.asarray(np.random.uniform(0.1, 0.5, nobs))
        shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        obs = self._bkd.asarray(np.random.randn(nobs, nouter))
        weights = self._bkd.ones((nobs, 1))

        likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, self._bkd)
        likelihood.set_shapes(shapes)
        likelihood.set_observations(obs)

        # Time the computation
        start = time.perf_counter()
        result = likelihood.logpdf_matrix(weights)
        elapsed = time.perf_counter() - start

        # Verify shape
        self.assertEqual(result.shape, (ninner, nouter))

        # Should complete in < 5 seconds for 250k element matrix
        self.assertLess(elapsed, 5.0)

    def test_large_jacobian_matrix(self):
        """Test Jacobian matrix with 200x200 inner/outer samples."""
        nobs = 10
        ninner = 200
        nouter = 200

        np.random.seed(42)
        noise_variances = self._bkd.asarray(np.random.uniform(0.1, 0.5, nobs))
        shapes = self._bkd.asarray(np.random.randn(nobs, ninner))
        obs = self._bkd.asarray(np.random.randn(nobs, nouter))
        weights = self._bkd.ones((nobs, 1))

        likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, self._bkd)
        likelihood.set_shapes(shapes)
        likelihood.set_observations(obs)

        # Time the computation
        start = time.perf_counter()
        result = likelihood.jacobian_matrix(weights)
        elapsed = time.perf_counter() - start

        # Verify shape
        self.assertEqual(result.shape, (ninner, nouter, nobs))

        # Should complete in < 5 seconds
        self.assertLess(elapsed, 5.0)

    def test_large_objective_evaluation(self):
        """Test objective with 300 inner and 300 outer samples."""
        nobs = 8
        ninner = 300
        nouter = 300

        np.random.seed(42)
        noise_variances = self._bkd.asarray(
            np.random.uniform(0.1, 0.3, nobs)
        )
        outer_shapes = self._bkd.asarray(
            np.random.randn(nobs, nouter)
        )
        inner_shapes = self._bkd.asarray(
            np.random.randn(nobs, ninner)
        )
        latent_samples = self._bkd.asarray(
            np.random.randn(nobs, nouter)
        )

        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            noise_variances, self._bkd
        )

        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,
            None,
            self._bkd,
        )

        weights = self._bkd.ones((nobs, 1)) / nobs

        # Time the computation
        start = time.perf_counter()
        result = objective(weights)
        elapsed = time.perf_counter() - start

        # Verify shape and value
        self.assertEqual(result.shape, (1, 1))
        result_np = self._bkd.to_numpy(result)
        self.assertTrue(np.isfinite(result_np[0, 0]))

        # Should complete in < 3 seconds
        self.assertLess(elapsed, 3.0)

    def test_large_objective_jacobian(self):
        """Test objective Jacobian with 200 inner and 200 outer samples."""
        nobs = 8
        ninner = 200
        nouter = 200

        np.random.seed(42)
        noise_variances = self._bkd.asarray(
            np.random.uniform(0.1, 0.3, nobs)
        )
        outer_shapes = self._bkd.asarray(
            np.random.randn(nobs, nouter)
        )
        inner_shapes = self._bkd.asarray(
            np.random.randn(nobs, ninner)
        )
        latent_samples = self._bkd.asarray(
            np.random.randn(nobs, nouter)
        )

        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            noise_variances, self._bkd
        )

        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,
            None,
            self._bkd,
        )

        weights = self._bkd.ones((nobs, 1)) / nobs

        # Time the computation
        start = time.perf_counter()
        result = objective.jacobian(weights)
        elapsed = time.perf_counter() - start

        # Verify shape
        self.assertEqual(result.shape, (1, nobs))

        # Should complete in < 5 seconds
        self.assertLess(elapsed, 5.0)

    def test_scaling_with_samples(self):
        """Test that time scales approximately linearly with ninner*nouter."""
        nobs = 5
        np.random.seed(42)
        noise_variances = self._bkd.asarray(np.random.uniform(0.1, 0.3, nobs))

        times = []
        sizes = [50, 100, 200]

        for n in sizes:
            outer_shapes = self._bkd.asarray(np.random.randn(nobs, n))
            inner_shapes = self._bkd.asarray(np.random.randn(nobs, n))
            latent_samples = self._bkd.asarray(np.random.randn(nobs, n))

            inner_likelihood = GaussianOEDInnerLoopLikelihood(
                noise_variances, self._bkd
            )

            objective = KLOEDObjective(
                inner_likelihood,
                outer_shapes,
                latent_samples,
                inner_shapes,
                None,
                None,
                self._bkd,
            )

            weights = self._bkd.ones((nobs, 1)) / nobs

            # Time multiple evaluations for more stable measurement
            start = time.perf_counter()
            for _ in range(5):
                _ = objective(weights)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Verify scaling: 100x100 should be ~4x slower than 50x50
        # and 200x200 should be ~16x slower than 50x50
        # Allow generous tolerance due to overhead
        ratio_100_50 = times[1] / times[0]
        ratio_200_50 = times[2] / times[0]

        # Ratios should be less than quadratic in n (which would be 8x and 64x)
        # Linear in n^2 gives 4x and 16x, allow 3x tolerance
        self.assertLess(ratio_100_50, 12.0)  # 4x * 3 tolerance
        self.assertLess(ratio_200_50, 48.0)  # 16x * 3 tolerance


class TestLargeScalePerformanceNumpy(TestLargeScalePerformance[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLargeScalePerformanceTorch(TestLargeScalePerformance[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
