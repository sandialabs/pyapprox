"""Tests for AdaptiveCombinationSparseGrid.

Note: This file will be renamed to test_adaptive.py in Phase 7 of the
sparse grid refactoring plan. Tests for other modules have been moved to:
- test_smolyak.py: Smolyak coefficients, downward closure, admissibility
- test_subspace.py: TensorProductSubspace
- test_isotropic.py: IsotropicCombinationSparseGrid
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
from pyapprox.typing.surrogates.sparsegrids import (
    AdaptiveCombinationSparseGrid,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests


class TestAdaptiveSparseGrid(Generic[Array], unittest.TestCase):
    """Tests for AdaptiveCombinationSparseGrid."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_step_samples_values_pattern(self):
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
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + y ** 2, (1, -1))
        grid.step_values(values)

        # Second step should also return samples (candidates exist)
        samples2 = grid.step_samples()
        self.assertIsNotNone(samples2)
        self.assertGreater(samples2.shape[1], 0)

    def test_evaluation_after_refinement(self):
        """Test that evaluation works after refinement."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [basis, basis], growth, admis
        )

        # Test function: f(x, y) = x^2 + y^2
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        def test_func(samples):
            x, y = samples[0, :], samples[1, :]
            return self._bkd.reshape(x ** 2 + y ** 2, (1, -1))

        # Perform several refinement steps
        for _ in range(3):
            samples = grid.step_samples()
            if samples is None:
                break
            values = test_func(samples)
            grid.step_values(values)

        # Evaluation should work
        test_pts = self._bkd.asarray([[0.3, -0.5],
                                      [0.2, 0.4]])
        result = grid(test_pts)
        expected = test_func(test_pts)

        self._bkd.assert_allclose(result, expected, rtol=1e-8)

    def test_convergence_on_polynomial(self):
        """Test that adaptive grid converges for polynomial target."""
        basis = LegendrePolynomial1D(self._bkd)
        growth = LinearGrowthRule(scale=2, shift=1)
        admis = MaxLevelCriteria(max_level=3, pnorm=1.0, bkd=self._bkd)

        grid = AdaptiveCombinationSparseGrid(
            self._bkd, [basis, basis], growth, admis
        )

        # Polynomial that should be exactly represented
        # Values shape: (nqoi, nsamples) = (1, nsamples)
        def poly_func(samples):
            x, y = samples[0, :], samples[1, :]
            return self._bkd.reshape(x ** 2 + y ** 2, (1, -1))

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
        test_pts = self._bkd.asarray([[0.3, -0.5, 0.8],
                                      [0.2, 0.4, -0.7]])
        result = grid(test_pts)
        expected = poly_func(test_pts)

        self._bkd.assert_allclose(result, expected, rtol=1e-10)


class TestAdaptiveSparseGridNumpy(TestAdaptiveSparseGrid[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveSparseGridTorch(TestAdaptiveSparseGrid[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
