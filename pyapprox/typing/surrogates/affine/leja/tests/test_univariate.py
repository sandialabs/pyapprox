"""Tests for univariate Leja sequence generation."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests


class TestLejaSequence1D(Generic[Array], unittest.TestCase):
    """Tests for univariate Leja sequence generation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_quadrature_rule_shape(self) -> None:
        """Test that quadrature_rule returns correct shapes."""
        from pyapprox.typing.surrogates.affine.leja import (
            LejaSequence1D,
            ChristoffelWeighting,
        )
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        leja = LejaSequence1D(self._bkd, poly, weighting, bounds=(-1.0, 1.0))

        samples, weights = leja.quadrature_rule(5)

        self.assertEqual(samples.shape, (1, 5))
        self.assertEqual(weights.shape, (5, 1))

    def test_extend_sequence(self) -> None:
        """Test extending the Leja sequence."""
        from pyapprox.typing.surrogates.affine.leja import (
            LejaSequence1D,
            ChristoffelWeighting,
        )
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        leja = LejaSequence1D(self._bkd, poly, weighting, bounds=(-1.0, 1.0))

        # Initial point
        self.assertEqual(leja.npoints(), 1)

        # Extend by 4 points
        leja.extend(4)
        self.assertEqual(leja.npoints(), 5)

    def test_nested_property(self) -> None:
        """Test that Leja sequences are nested."""
        from pyapprox.typing.surrogates.affine.leja import (
            LejaSequence1D,
            ChristoffelWeighting,
        )
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        leja = LejaSequence1D(self._bkd, poly, weighting, bounds=(-1.0, 1.0))

        # Get 3 points
        samples_3, _ = leja.quadrature_rule(3)

        # Get 5 points
        samples_5, _ = leja.quadrature_rule(5)

        # First 3 points should match
        self._bkd.assert_allclose(samples_3, samples_5[:, :3], rtol=1e-12)

    def test_points_within_bounds(self) -> None:
        """Test that all points are within bounds."""
        from pyapprox.typing.surrogates.affine.leja import (
            LejaSequence1D,
            ChristoffelWeighting,
        )
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        lb, ub = -1.0, 1.0
        leja = LejaSequence1D(self._bkd, poly, weighting, bounds=(lb, ub))

        samples, _ = leja.quadrature_rule(10)

        # All samples should be within bounds (with small tolerance)
        self.assertTrue(self._bkd.all_bool(samples >= lb - 1e-10))
        self.assertTrue(self._bkd.all_bool(samples <= ub + 1e-10))

    def test_clear_cache(self) -> None:
        """Test clearing the cache resets to initial point."""
        from pyapprox.typing.surrogates.affine.leja import (
            LejaSequence1D,
            ChristoffelWeighting,
        )
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        leja = LejaSequence1D(self._bkd, poly, weighting, bounds=(-1.0, 1.0))

        # Extend sequence
        leja.quadrature_rule(5)
        self.assertEqual(leja.npoints(), 5)

        # Clear cache
        leja.clear_cache()
        self.assertEqual(leja.npoints(), 1)

    def test_with_lagrange_basis(self) -> None:
        """Test using LejaSequence1D with LagrangeBasis1D."""
        from pyapprox.typing.surrogates.affine.leja import (
            LejaSequence1D,
            ChristoffelWeighting,
        )
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )
        from pyapprox.typing.surrogates.affine.univariate.lagrange import (
            LagrangeBasis1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        leja = LejaSequence1D(self._bkd, poly, weighting, bounds=(-1.0, 1.0))

        # Create Lagrange basis using Leja sequence
        lagrange = LagrangeBasis1D(self._bkd, leja.quadrature_rule)
        lagrange.set_nterms(5)

        # Evaluate at sample points
        samples = self._bkd.asarray([[0.0, 0.5, -0.5]])
        values = lagrange(samples)

        self.assertEqual(values.shape, (3, 5))


class TestLejaObjective(Generic[Array], unittest.TestCase):
    """Tests for LejaObjective."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_objective_shape(self) -> None:
        """Test that objective returns correct shape."""
        from pyapprox.typing.surrogates.affine.leja.univariate import (
            LejaObjective,
        )
        from pyapprox.typing.surrogates.affine.leja import ChristoffelWeighting
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        objective = LejaObjective(self._bkd, poly, weighting, bounds=(-1.0, 1.0))

        # Set initial sequence
        initial = self._bkd.asarray([[0.0]])
        objective.set_sequence(initial)

        # Evaluate objective
        test_points = self._bkd.asarray([[0.5, -0.5, 0.8]])
        values = objective(test_points)

        self.assertEqual(values.shape, (3, 1))

    def test_objective_negative(self) -> None:
        """Test that objective values are non-positive (minimization)."""
        from pyapprox.typing.surrogates.affine.leja.univariate import (
            LejaObjective,
        )
        from pyapprox.typing.surrogates.affine.leja import ChristoffelWeighting
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        objective = LejaObjective(self._bkd, poly, weighting, bounds=(-1.0, 1.0))

        initial = self._bkd.asarray([[0.0]])
        objective.set_sequence(initial)

        test_points = self._bkd.asarray([[0.5, -0.5, 0.8]])
        values = objective(test_points)

        # Objective should be non-positive (we minimize negative weighted residual)
        self.assertTrue(self._bkd.all_bool(values <= 0))

    def test_jacobian_shape(self) -> None:
        """Test that Jacobian has correct shape."""
        from pyapprox.typing.surrogates.affine.leja.univariate import (
            LejaObjective,
        )
        from pyapprox.typing.surrogates.affine.leja import ChristoffelWeighting
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        objective = LejaObjective(self._bkd, poly, weighting, bounds=(-1.0, 1.0))

        initial = self._bkd.asarray([[0.0]])
        objective.set_sequence(initial)

        sample = self._bkd.asarray([[0.5]])
        jac = objective.jacobian(sample)

        self.assertEqual(jac.shape, (1, 1))


# NumPy backend tests
class TestLejaSequence1DNumpy(TestLejaSequence1D[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLejaObjectiveNumpy(TestLejaObjective[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestLejaSequence1DTorch(TestLejaSequence1D[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestLejaObjectiveTorch(TestLejaObjective[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
