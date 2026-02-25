"""Tests for univariate Leja sequence generation.

Covers both one-point and two-point Leja objectives and sequences,
including derivative checks via DerivativeChecker and integration
accuracy tests matching legacy test_leja.py test cases.
"""

import unittest
from typing import Any, Callable, Generic, Tuple

import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.probability import (
    BetaMarginal,
    GaussianMarginal,
    UniformMarginal,
)
from pyapprox.surrogates.affine.leja import (
    ChristoffelWeighting,
    PDFWeighting,
)
from pyapprox.surrogates.affine.leja.univariate import (
    LejaObjective,
    LejaSequence1D,
    TwoPointLejaObjective,
)
from pyapprox.surrogates.affine.univariate import (
    HermitePolynomial1D,
    JacobiPolynomial1D,
    LegendrePolynomial1D,
)
from pyapprox.surrogates.affine.univariate.lagrange import (
    LagrangeBasis1D,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


# =============================================================================
# Helper functions for PDFWeighting wrappers
# =============================================================================
def _make_pdf_wrappers(
    marginal: Any, bkd: Backend[Array]
) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Create (nsamples,)->(nsamples,) wrappers for marginal pdf/jacobian.

    PDFWeighting expects 1D callables but marginals use (1, nsamples) shapes.
    """

    def pdf_1d(samples: Array) -> Array:
        return marginal.pdf(bkd.reshape(samples, (1, -1)))[0, :]

    def pdf_jac_1d(samples: Array) -> Array:
        return marginal.pdf_jacobian(bkd.reshape(samples, (1, -1)))[0, :]

    return pdf_1d, pdf_jac_1d


# =============================================================================
# Integration test combo configurations
# =============================================================================
# Each combo: (name, poly_type, weighting_type, bounds, exact_integral_x4)
# Poly types: "legendre", "hermite", "jacobi11"
# Weighting types: "christoffel", "pdf_uniform", "pdf_beta22", "pdf_gaussian"
ONE_POINT_COMBOS = [
    ("hermite_christoffel", "hermite", "christoffel", (-10.0, 10.0), 3.0),
    ("legendre_christoffel", "legendre", "christoffel", (-1.0, 1.0), 0.2),
    ("legendre_pdf_uniform", "legendre", "pdf_uniform", (-1.0, 1.0), 0.2),
    (
        "jacobi11_pdf_beta22",
        "jacobi11",
        "pdf_beta22",
        (-1.0, 1.0),
        3.0 / 35.0,
    ),
]

TWO_POINT_COMBOS = [
    ("legendre_christoffel", "legendre", "christoffel", (-1.0, 1.0), 0.2),
    ("legendre_pdf_uniform", "legendre", "pdf_uniform", (-1.0, 1.0), 0.2),
    (
        "jacobi11_pdf_beta22",
        "jacobi11",
        "pdf_beta22",
        (-1.0, 1.0),
        3.0 / 35.0,
    ),
    ("hermite_christoffel", "hermite", "christoffel", (-10.0, 10.0), 3.0),
]


# =============================================================================
# LejaSequence1D tests
# =============================================================================
class TestLejaSequence1D(Generic[Array], unittest.TestCase):
    """Tests for univariate Leja sequence generation."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_leja(self) -> LejaSequence1D[Array]:
        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        return LejaSequence1D(self._bkd, poly, weighting, bounds=(-1.0, 1.0))

    def test_quadrature_rule_shape(self) -> None:
        """Test that quadrature_rule returns correct shapes."""
        leja = self._create_leja()
        samples, weights = leja.quadrature_rule(5)
        self.assertEqual(samples.shape, (1, 5))
        self.assertEqual(weights.shape, (5, 1))

    def test_extend_sequence(self) -> None:
        """Test extending the Leja sequence."""
        leja = self._create_leja()
        self.assertEqual(leja.npoints(), 1)
        leja.extend(4)
        self.assertEqual(leja.npoints(), 5)

    def test_nested_property(self) -> None:
        """Test that Leja sequences are nested."""
        leja = self._create_leja()
        samples_3, _ = leja.quadrature_rule(3)
        samples_5, _ = leja.quadrature_rule(5)
        self._bkd.assert_allclose(samples_3, samples_5[:, :3], rtol=1e-12)

    def test_points_within_bounds(self) -> None:
        """Test that all points are within bounds."""
        leja = self._create_leja()
        samples, _ = leja.quadrature_rule(10)
        self.assertTrue(self._bkd.all_bool(samples >= -1.0 - 1e-10))
        self.assertTrue(self._bkd.all_bool(samples <= 1.0 + 1e-10))

    def test_clear_cache(self) -> None:
        """Test clearing the cache resets to initial point."""
        leja = self._create_leja()
        leja.quadrature_rule(5)
        self.assertEqual(leja.npoints(), 5)
        leja.clear_cache()
        self.assertEqual(leja.npoints(), 1)

    def test_with_lagrange_basis(self) -> None:
        """Test using LejaSequence1D with LagrangeBasis1D."""
        leja = self._create_leja()
        lagrange = LagrangeBasis1D(self._bkd, leja.quadrature_rule)
        lagrange.set_nterms(5)
        samples = self._bkd.asarray([[0.0, 0.5, -0.5]])
        values = lagrange(samples)
        self.assertEqual(values.shape, (3, 5))


# =============================================================================
# LejaObjective tests
# =============================================================================
class TestLejaObjective(Generic[Array], unittest.TestCase):
    """Tests for LejaObjective.

    Matches legacy test_one_point_leja_objective:
    - Shape checks (nqoi, nsamples) convention
    - Non-positivity
    - Jacobian shape (nqoi, nvars)
    - Jacobian via DerivativeChecker (2 combos: Christoffel, PDF)
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_objective(
        self, weighting_cls: str = "christoffel"
    ) -> LejaObjective[Array]:
        poly = LegendrePolynomial1D(self._bkd)
        if weighting_cls == "christoffel":
            weighting = ChristoffelWeighting(self._bkd)
        else:
            marginal = UniformMarginal(-1.0, 1.0, self._bkd)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, self._bkd)
            weighting = PDFWeighting(self._bkd, pdf_1d, pdf_jac_1d)
        objective = LejaObjective(self._bkd, poly, weighting, bounds=(-1.0, 1.0))
        initial = self._bkd.asarray([[0.1, 0.9]])
        objective.set_sequence(initial)
        return objective

    def test_objective_shape(self) -> None:
        """Test that objective returns (nqoi, nsamples) shape."""
        objective = self._create_objective()
        test_points = self._bkd.asarray([[0.5, -0.5, 0.8]])
        values = objective(test_points)
        self.assertEqual(values.shape, (1, 3))

    def test_objective_negative(self) -> None:
        """Test that objective values are non-positive (minimization)."""
        objective = self._create_objective()
        test_points = self._bkd.asarray([[0.5, -0.5, 0.8]])
        values = objective(test_points)
        self.assertTrue(self._bkd.all_bool(values <= 0))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian shape is (nqoi, nvars) = (1, 1)."""
        objective = self._create_objective()
        sample = self._bkd.asarray([[0.5]])
        jac = objective.jacobian(sample)
        self.assertEqual(jac.shape, (1, 1))

    def test_jacobian_christoffel_derivative_checker(self) -> None:
        """Test Jacobian via DerivativeChecker with Christoffel weighting."""
        objective = self._create_objective("christoffel")
        sample = self._bkd.asarray([[0.5]])
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = float(checker.error_ratio(errors[0]).item())
        self.assertLess(error_ratio, 1e-6)

    def test_jacobian_pdf_derivative_checker(self) -> None:
        """Test Jacobian via DerivativeChecker with PDF weighting."""
        objective = self._create_objective("pdf")
        sample = self._bkd.asarray([[0.5]])
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = float(checker.error_ratio(errors[0]).item())
        self.assertLess(error_ratio, 1e-6)


# =============================================================================
# TwoPointLejaObjective tests
# =============================================================================
class TestTwoPointLejaObjective(Generic[Array], unittest.TestCase):
    """Tests for TwoPointLejaObjective.

    Matches legacy test_two_point_leja_objective:
    - Shape checks following (nqoi, nsamples) convention
    - Non-positivity
    - Jacobian shape (nqoi, nvars) = (1, 2)
    - Jacobian via DerivativeChecker (Christoffel and PDF weightings)
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_objective(
        self, weighting_cls: str = "christoffel"
    ) -> TwoPointLejaObjective[Array]:
        poly = LegendrePolynomial1D(self._bkd)
        if weighting_cls == "christoffel":
            weighting = ChristoffelWeighting(self._bkd)
        else:
            marginal = UniformMarginal(-1.0, 1.0, self._bkd)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, self._bkd)
            weighting = PDFWeighting(self._bkd, pdf_1d, pdf_jac_1d)
        objective = TwoPointLejaObjective(
            self._bkd, poly, weighting, bounds=(-1.0, 1.0)
        )
        initial = self._bkd.asarray([[0.0, -0.8, 0.8]])
        objective.set_sequence(initial)
        return objective

    def test_objective_shape(self) -> None:
        """Test that two-point objective returns (nqoi, nsamples) shape."""
        objective = self._create_objective()
        test_points = self._bkd.asarray([[0.5, -0.5], [0.3, 0.7]])
        values = objective(test_points)
        self.assertEqual(values.shape, (1, 2))

    def test_objective_negative(self) -> None:
        """Test that objective values are non-positive."""
        objective = self._create_objective()
        test_points = self._bkd.asarray([[0.5, -0.5, 0.1], [0.3, 0.7, -0.2]])
        values = objective(test_points)
        self.assertTrue(self._bkd.all_bool(values <= 0))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian shape is (nqoi, nvars) = (1, 2)."""
        objective = self._create_objective()
        sample = self._bkd.asarray([[0.3], [0.7]])
        jac = objective.jacobian(sample)
        self.assertEqual(jac.shape, (1, 2))

    def test_nvars(self) -> None:
        """Test nvars returns 2."""
        objective = self._create_objective()
        self.assertEqual(objective.nvars(), 2)

    def test_initial_iterates_shape(self) -> None:
        """Test initial iterates are pairs."""
        objective = self._create_objective()
        iterates, bounds_list = objective.initial_iterates_and_bounds()
        self.assertEqual(iterates.shape[0], 2)
        for b in bounds_list:
            self.assertEqual(b.shape, (2, 2))

    def test_jacobian_christoffel_derivative_checker(self) -> None:
        """Test Jacobian via DerivativeChecker with Christoffel weighting."""
        objective = self._create_objective("christoffel")
        sample = self._bkd.asarray([[0.3], [0.7]])
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = float(checker.error_ratio(errors[0]).item())
        self.assertLess(error_ratio, 1e-6)

    def test_jacobian_pdf_derivative_checker(self) -> None:
        """Test Jacobian via DerivativeChecker with PDF weighting."""
        objective = self._create_objective("pdf")
        sample = self._bkd.asarray([[0.3], [0.7]])
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = float(checker.error_ratio(errors[0]).item())
        self.assertLess(error_ratio, 1e-6)


# =============================================================================
# TwoPointLejaSequence1D tests
# =============================================================================
class TestTwoPointLejaSequence1D(Generic[Array], unittest.TestCase):
    """Tests for LejaSequence1D with TwoPointLejaObjective."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_leja(self) -> LejaSequence1D[Array]:
        poly = LegendrePolynomial1D(self._bkd)
        weighting = ChristoffelWeighting(self._bkd)
        return LejaSequence1D(
            self._bkd,
            poly,
            weighting,
            bounds=(-1.0, 1.0),
            objective_class=TwoPointLejaObjective,
        )

    def test_extend_adds_two_points(self) -> None:
        """Test that each step adds 2 points."""
        leja = self._create_leja()
        self.assertEqual(leja.npoints(), 1)
        leja.extend(2)
        self.assertEqual(leja.npoints(), 3)
        leja.extend(2)
        self.assertEqual(leja.npoints(), 5)

    def test_extend_validation(self) -> None:
        """Test ValueError for non-even n_new_points."""
        leja = self._create_leja()
        with self.assertRaises(ValueError):
            leja.extend(3)

    def test_nested_property(self) -> None:
        """Test that two-point sequence is nested."""
        leja = self._create_leja()
        samples_3, _ = leja.quadrature_rule(3)
        samples_5, _ = leja.quadrature_rule(5)
        self._bkd.assert_allclose(samples_3, samples_5[:, :3], rtol=1e-12)

    def test_points_within_bounds(self) -> None:
        """Test that all points are within bounds."""
        leja = self._create_leja()
        samples, _ = leja.quadrature_rule(7)
        self.assertTrue(self._bkd.all_bool(samples >= -1.0 - 1e-10))
        self.assertTrue(self._bkd.all_bool(samples <= 1.0 + 1e-10))

    def test_quadrature_rule_shape(self) -> None:
        """Test quadrature_rule returns correct shapes."""
        leja = self._create_leja()
        samples, weights = leja.quadrature_rule(5)
        self.assertEqual(samples.shape, (1, 5))
        self.assertEqual(weights.shape, (5, 1))


# =============================================================================
# Parametrized integration accuracy tests (one-point)
# =============================================================================
class TestOnePointLejaIntegration(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Integration accuracy tests for one-point Leja sequences.

    Matches legacy test_leja_sequence: 5-point Leja sequence should
    exactly integrate x^4 (degree 4 polynomial with 5 points).

    Tests 4 combos:
    - Hermite + Christoffel
    - Legendre + Christoffel
    - Legendre + PDF(uniform)
    - Jacobi(1,1) + PDF(beta(2,2))
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_poly_and_weighting(
        self, poly_type: str, weighting_type: str
    ) -> Tuple[Any, Any]:
        """Create polynomial and weighting from type strings."""
        if poly_type == "hermite":
            poly = HermitePolynomial1D(self._bkd)
        elif poly_type == "legendre":
            poly = LegendrePolynomial1D(self._bkd)
        elif poly_type == "jacobi11":
            # Jacobi(alpha=1, beta=1) for Beta(2,2)
            poly = JacobiPolynomial1D(1.0, 1.0, self._bkd)
        else:
            raise ValueError(f"Unknown poly_type: {poly_type}")

        if weighting_type == "christoffel":
            weighting = ChristoffelWeighting(self._bkd)
        elif weighting_type == "pdf_uniform":
            marginal = UniformMarginal(-1.0, 1.0, self._bkd)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, self._bkd)
            weighting = PDFWeighting(self._bkd, pdf_1d, pdf_jac_1d)
        elif weighting_type == "pdf_beta22":
            marginal = BetaMarginal(2.0, 2.0, self._bkd, lb=-1.0, ub=1.0)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, self._bkd)
            weighting = PDFWeighting(self._bkd, pdf_1d, pdf_jac_1d)
        elif weighting_type == "pdf_gaussian":
            marginal = GaussianMarginal(0.0, 1.0, self._bkd)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, self._bkd)
            weighting = PDFWeighting(self._bkd, pdf_1d, pdf_jac_1d)
        else:
            raise ValueError(f"Unknown weighting_type: {weighting_type}")

        return poly, weighting

    @parametrize(
        "name,poly_type,weighting_type,bounds,exact",
        ONE_POINT_COMBOS,
    )
    def test_integration_x4(
        self,
        name: str,
        poly_type: str,
        weighting_type: str,
        bounds: Tuple[float, float],
        exact: float,
    ) -> None:
        """Test that 5-point one-point Leja sequence exactly integrates x^4."""
        poly, weighting = self._create_poly_and_weighting(poly_type, weighting_type)
        leja = LejaSequence1D(
            self._bkd,
            poly,
            weighting,
            bounds=bounds,
            objective_class=LejaObjective,
        )
        samples, weights = leja.quadrature_rule(5)

        # Integrate x^4 using quadrature: sum(w_i * x_i^4)
        integral = self._bkd.sum(weights[:, 0] * samples[0, :] ** 4)
        self._bkd.assert_allclose(
            self._bkd.reshape(integral, (1,)),
            self._bkd.asarray([exact]),
            rtol=1e-6,
        )


# =============================================================================
# Parametrized integration accuracy tests (two-point)
# =============================================================================
class TestTwoPointLejaIntegration(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Integration accuracy tests for two-point Leja sequences.

    Same as one-point but using TwoPointLejaObjective. 5-point sequence
    (1 initial + 2 two-point steps) should exactly integrate x^4.

    Tests 4 combos:
    - Legendre + Christoffel
    - Legendre + PDF(uniform)
    - Jacobi(1,1) + PDF(beta(2,2))
    - Hermite + Christoffel
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_poly_and_weighting(
        self, poly_type: str, weighting_type: str
    ) -> Tuple[Any, Any]:
        """Create polynomial and weighting from type strings."""
        if poly_type == "hermite":
            poly = HermitePolynomial1D(self._bkd)
        elif poly_type == "legendre":
            poly = LegendrePolynomial1D(self._bkd)
        elif poly_type == "jacobi11":
            poly = JacobiPolynomial1D(1.0, 1.0, self._bkd)
        else:
            raise ValueError(f"Unknown poly_type: {poly_type}")

        if weighting_type == "christoffel":
            weighting = ChristoffelWeighting(self._bkd)
        elif weighting_type == "pdf_uniform":
            marginal = UniformMarginal(-1.0, 1.0, self._bkd)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, self._bkd)
            weighting = PDFWeighting(self._bkd, pdf_1d, pdf_jac_1d)
        elif weighting_type == "pdf_beta22":
            marginal = BetaMarginal(2.0, 2.0, self._bkd, lb=-1.0, ub=1.0)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, self._bkd)
            weighting = PDFWeighting(self._bkd, pdf_1d, pdf_jac_1d)
        elif weighting_type == "pdf_gaussian":
            marginal = GaussianMarginal(0.0, 1.0, self._bkd)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, self._bkd)
            weighting = PDFWeighting(self._bkd, pdf_1d, pdf_jac_1d)
        else:
            raise ValueError(f"Unknown weighting_type: {weighting_type}")

        return poly, weighting

    @parametrize(
        "name,poly_type,weighting_type,bounds,exact",
        TWO_POINT_COMBOS,
    )
    def test_integration_x4(
        self,
        name: str,
        poly_type: str,
        weighting_type: str,
        bounds: Tuple[float, float],
        exact: float,
    ) -> None:
        """Test that 5-point two-point Leja sequence exactly integrates x^4."""
        poly, weighting = self._create_poly_and_weighting(poly_type, weighting_type)
        leja = LejaSequence1D(
            self._bkd,
            poly,
            weighting,
            bounds=bounds,
            objective_class=TwoPointLejaObjective,
        )
        samples, weights = leja.quadrature_rule(5)

        # Integrate x^4 using quadrature: sum(w_i * x_i^4)
        integral = self._bkd.sum(weights[:, 0] * samples[0, :] ** 4)
        self._bkd.assert_allclose(
            self._bkd.reshape(integral, (1,)),
            self._bkd.asarray([exact]),
            rtol=1e-6,
        )


# =============================================================================
# NumPy backend concrete classes
# =============================================================================
class TestLejaSequence1DNumpy(TestLejaSequence1D[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLejaObjectiveNumpy(TestLejaObjective[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTwoPointLejaObjectiveNumpy(TestTwoPointLejaObjective[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTwoPointLejaSequence1DNumpy(TestTwoPointLejaSequence1D[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOnePointLejaIntegrationNumpy(TestOnePointLejaIntegration[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTwoPointLejaIntegrationNumpy(TestTwoPointLejaIntegration[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# =============================================================================
# PyTorch backend concrete classes
# =============================================================================
class TestLejaSequence1DTorch(TestLejaSequence1D[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestLejaObjectiveTorch(TestLejaObjective[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestTwoPointLejaObjectiveTorch(TestTwoPointLejaObjective[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestTwoPointLejaSequence1DTorch(TestTwoPointLejaSequence1D[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestOnePointLejaIntegrationTorch(TestOnePointLejaIntegration[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestTwoPointLejaIntegrationTorch(TestTwoPointLejaIntegration[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
