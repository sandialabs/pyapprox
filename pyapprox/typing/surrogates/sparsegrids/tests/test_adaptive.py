"""Dual-backend tests for AdaptiveCombinationSparseGrid.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.surrogates.sparsegrids import (
    AdaptiveCombinationSparseGrid,
)
from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.typing.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)


class TestAdaptiveSparseGrid(Generic[Array], unittest.TestCase):
    """Tests for AdaptiveCombinationSparseGrid - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_step_samples_values_pattern(self) -> None:
        """Test the step_samples/step_values pattern."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [basis, basis], growth, admis
        )

        # First step should return samples
        samples = grid.step_samples()
        self.assertIsNotNone(samples)
        self.assertGreater(samples.shape[1], 0)

        # Values should be accepted
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + y ** 2, (-1, 1))
        grid.step_values(values)

        # Second step should also return samples (candidates exist)
        samples2 = grid.step_samples()
        self.assertIsNotNone(samples2)
        self.assertGreater(samples2.shape[1], 0)

    def test_evaluation_after_refinement(self) -> None:
        """Test that evaluation works after refinement."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [basis, basis], growth, admis
        )

        # Test function: f(x, y) = x^2 + y^2
        def test_func(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            return self._bkd.reshape(x ** 2 + y ** 2, (-1, 1))

        # Perform several refinement steps
        for _ in range(3):
            samples = grid.step_samples()
            if samples is None:
                break
            values = test_func(samples)
            grid.step_values(values)

        # Evaluation should work
        test_pts = self._bkd.asarray([[0.3, -0.5], [0.2, 0.4]])
        result = grid(test_pts)
        expected = test_func(test_pts)

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-8))

    def test_convergence_on_polynomial(self) -> None:
        """Test that adaptive grid converges for polynomial target."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [basis, basis], growth, admis
        )

        # Polynomial that should be exactly represented
        def poly_func(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            return self._bkd.reshape(x ** 2 + y ** 2, (-1, 1))

        # Refine until convergence
        max_steps = 10
        for _ in range(max_steps):
            samples = grid.step_samples()
            if samples is None:
                break
            values = poly_func(samples)
            grid.step_values(values)

        # Should have refined at least once
        self.assertGreater(grid.nsubspaces(), 1)

        # Evaluation should be exact
        test_pts = self._bkd.asarray([[0.3, -0.5, 0.8], [0.2, 0.4, -0.7]])
        result = grid(test_pts)
        expected = poly_func(test_pts)

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-10))

    def test_multi_qoi_adaptive(self) -> None:
        """Test adaptive grid with multiple quantities of interest."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=2, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [basis, basis], growth, admis
        )

        # Two QoIs
        def multi_func(samples: Array) -> Array:
            x, y = samples[0, :], samples[1, :]
            return self._bkd.stack([x + y, x * y], axis=1)

        # Refine
        for _ in range(3):
            samples = grid.step_samples()
            if samples is None:
                break
            values = multi_func(samples)
            grid.step_values(values)

        # Test evaluation
        test_pts = self._bkd.asarray([[0.3, -0.2], [0.4, 0.5]])
        result = grid(test_pts)

        self.assertEqual(result.shape[1], 2)

    def test_3d_adaptive_grid(self) -> None:
        """Test 3D adaptive sparse grid."""
        bases = [LegendrePolynomial1D(self._bkd) for _ in range(3)]
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=2, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, bases, growth, admis
        )

        # Linear function
        def linear_func(samples: Array) -> Array:
            return self._bkd.reshape(
                samples[0, :] + samples[1, :] + samples[2, :], (-1, 1)
            )

        # Refine until convergence (or max steps)
        for _ in range(10):
            samples = grid.step_samples()
            if samples is None:
                break
            values = linear_func(samples)
            grid.step_values(values)

        # Test evaluation - should be exact for linear
        test_pts = self._bkd.asarray([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])
        result = grid(test_pts)
        expected = linear_func(test_pts)

        self.assertTrue(self._bkd.allclose(result, expected, rtol=1e-10))


# NumPy backend tests
class TestAdaptiveSparseGridNumpy(TestAdaptiveSparseGrid[NDArray[Any]]):
    """NumPy backend tests for AdaptiveCombinationSparseGrid."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestAdaptiveSparseGridTorch(TestAdaptiveSparseGrid[torch.Tensor]):
    """PyTorch backend tests for AdaptiveCombinationSparseGrid."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
