"""Tests for univariate Leja sequence generation.

Covers both one-point and two-point Leja objectives and sequences,
including derivative checks via DerivativeChecker and integration
accuracy tests matching legacy test_leja.py test cases.
"""

from typing import Any, Callable, Tuple

import pytest

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
from tests._helpers.markers import slow_test


# =============================================================================
# Helper functions for PDFWeighting wrappers
# =============================================================================
def _make_pdf_wrappers(
    marginal: Any, bkd
) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Create (nsamples,)->(nsamples,) wrappers for marginal pdf/jacobian.

    PDFWeighting expects 1D callables but marginals use (1, nsamples) shapes.
    """

    def pdf_1d(samples):
        return marginal.pdf(bkd.reshape(samples, (1, -1)))[0, :]

    def pdf_jac_1d(samples):
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


def _create_poly_and_weighting(bkd, poly_type, weighting_type):
    """Create polynomial and weighting from type strings."""
    if poly_type == "hermite":
        poly = HermitePolynomial1D(bkd)
    elif poly_type == "legendre":
        poly = LegendrePolynomial1D(bkd)
    elif poly_type == "jacobi11":
        # Jacobi(alpha=1, beta=1) for Beta(2,2)
        poly = JacobiPolynomial1D(1.0, 1.0, bkd)
    else:
        raise ValueError(f"Unknown poly_type: {poly_type}")

    if weighting_type == "christoffel":
        weighting = ChristoffelWeighting(bkd)
    elif weighting_type == "pdf_uniform":
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, bkd)
        weighting = PDFWeighting(bkd, pdf_1d, pdf_jac_1d)
    elif weighting_type == "pdf_beta22":
        marginal = BetaMarginal(2.0, 2.0, bkd, lb=-1.0, ub=1.0)
        pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, bkd)
        weighting = PDFWeighting(bkd, pdf_1d, pdf_jac_1d)
    elif weighting_type == "pdf_gaussian":
        marginal = GaussianMarginal(0.0, 1.0, bkd)
        pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, bkd)
        weighting = PDFWeighting(bkd, pdf_1d, pdf_jac_1d)
    else:
        raise ValueError(f"Unknown weighting_type: {weighting_type}")

    return poly, weighting


# =============================================================================
# LejaSequence1D tests
# =============================================================================
class TestLejaSequence1D:
    """Tests for univariate Leja sequence generation."""

    def _create_leja(self, bkd):
        poly = LegendrePolynomial1D(bkd)
        weighting = ChristoffelWeighting(bkd)
        return LejaSequence1D(bkd, poly, weighting, bounds=(-1.0, 1.0))

    def test_quadrature_rule_shape(self, bkd) -> None:
        """Test that quadrature_rule returns correct shapes."""
        leja = self._create_leja(bkd)
        samples, weights = leja.quadrature_rule(5)
        assert samples.shape == (1, 5)
        assert weights.shape == (5, 1)

    def test_extend_sequence(self, bkd) -> None:
        """Test extending the Leja sequence."""
        leja = self._create_leja(bkd)
        assert leja.npoints() == 1
        leja.extend(4)
        assert leja.npoints() == 5

    def test_nested_property(self, bkd) -> None:
        """Test that Leja sequences are nested."""
        leja = self._create_leja(bkd)
        samples_3, _ = leja.quadrature_rule(3)
        samples_5, _ = leja.quadrature_rule(5)
        bkd.assert_allclose(samples_3, samples_5[:, :3], rtol=1e-12)

    def test_points_within_bounds(self, bkd) -> None:
        """Test that all points are within bounds."""
        leja = self._create_leja(bkd)
        samples, _ = leja.quadrature_rule(10)
        assert bkd.all_bool(samples >= -1.0 - 1e-10)
        assert bkd.all_bool(samples <= 1.0 + 1e-10)

    def test_clear_cache(self, bkd) -> None:
        """Test clearing the cache resets to initial point."""
        leja = self._create_leja(bkd)
        leja.quadrature_rule(5)
        assert leja.npoints() == 5
        leja.clear_cache()
        assert leja.npoints() == 1

    def test_with_lagrange_basis(self, bkd) -> None:
        """Test using LejaSequence1D with LagrangeBasis1D."""
        leja = self._create_leja(bkd)
        lagrange = LagrangeBasis1D(bkd, leja.quadrature_rule)
        lagrange.set_nterms(5)
        samples = bkd.asarray([[0.0, 0.5, -0.5]])
        values = lagrange(samples)
        assert values.shape == (3, 5)


# =============================================================================
# LejaObjective tests
# =============================================================================
class TestLejaObjective:
    """Tests for LejaObjective.

    Matches legacy test_one_point_leja_objective:
    - Shape checks (nqoi, nsamples) convention
    - Non-positivity
    - Jacobian shape (nqoi, nvars)
    - Jacobian via DerivativeChecker (2 combos: Christoffel, PDF)
    """

    def _create_objective(self, bkd, weighting_cls="christoffel"):
        poly = LegendrePolynomial1D(bkd)
        if weighting_cls == "christoffel":
            weighting = ChristoffelWeighting(bkd)
        else:
            marginal = UniformMarginal(-1.0, 1.0, bkd)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, bkd)
            weighting = PDFWeighting(bkd, pdf_1d, pdf_jac_1d)
        objective = LejaObjective(bkd, poly, weighting, bounds=(-1.0, 1.0))
        initial = bkd.asarray([[0.1, 0.9]])
        objective.set_sequence(initial)
        return objective

    def test_objective_shape(self, bkd) -> None:
        """Test that objective returns (nqoi, nsamples) shape."""
        objective = self._create_objective(bkd)
        test_points = bkd.asarray([[0.5, -0.5, 0.8]])
        values = objective(test_points)
        assert values.shape == (1, 3)

    def test_objective_negative(self, bkd) -> None:
        """Test that objective values are non-positive (minimization)."""
        objective = self._create_objective(bkd)
        test_points = bkd.asarray([[0.5, -0.5, 0.8]])
        values = objective(test_points)
        assert bkd.all_bool(values <= 0)

    def test_jacobian_shape(self, bkd) -> None:
        """Test Jacobian shape is (nqoi, nvars) = (1, 1)."""
        objective = self._create_objective(bkd)
        sample = bkd.asarray([[0.5]])
        jac = objective.jacobian(sample)
        assert jac.shape == (1, 1)

    def test_jacobian_christoffel_derivative_checker(self, bkd) -> None:
        """Test Jacobian via DerivativeChecker with Christoffel weighting."""
        objective = self._create_objective(bkd, "christoffel")
        sample = bkd.asarray([[0.5]])
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = float(checker.error_ratio(errors[0]).item())
        assert error_ratio < 1e-6

    def test_jacobian_pdf_derivative_checker(self, bkd) -> None:
        """Test Jacobian via DerivativeChecker with PDF weighting."""
        objective = self._create_objective(bkd, "pdf")
        sample = bkd.asarray([[0.5]])
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = float(checker.error_ratio(errors[0]).item())
        assert error_ratio < 1e-6


# =============================================================================
# TwoPointLejaObjective tests
# =============================================================================
class TestTwoPointLejaObjective:
    """Tests for TwoPointLejaObjective.

    Matches legacy test_two_point_leja_objective:
    - Shape checks following (nqoi, nsamples) convention
    - Non-positivity
    - Jacobian shape (nqoi, nvars) = (1, 2)
    - Jacobian via DerivativeChecker (Christoffel and PDF weightings)
    """

    def _create_objective(self, bkd, weighting_cls="christoffel"):
        poly = LegendrePolynomial1D(bkd)
        if weighting_cls == "christoffel":
            weighting = ChristoffelWeighting(bkd)
        else:
            marginal = UniformMarginal(-1.0, 1.0, bkd)
            pdf_1d, pdf_jac_1d = _make_pdf_wrappers(marginal, bkd)
            weighting = PDFWeighting(bkd, pdf_1d, pdf_jac_1d)
        objective = TwoPointLejaObjective(
            bkd, poly, weighting, bounds=(-1.0, 1.0)
        )
        initial = bkd.asarray([[0.0, -0.8, 0.8]])
        objective.set_sequence(initial)
        return objective

    def test_objective_shape(self, bkd) -> None:
        """Test that two-point objective returns (nqoi, nsamples) shape."""
        objective = self._create_objective(bkd)
        test_points = bkd.asarray([[0.5, -0.5], [0.3, 0.7]])
        values = objective(test_points)
        assert values.shape == (1, 2)

    def test_objective_negative(self, bkd) -> None:
        """Test that objective values are non-positive."""
        objective = self._create_objective(bkd)
        test_points = bkd.asarray([[0.5, -0.5, 0.1], [0.3, 0.7, -0.2]])
        values = objective(test_points)
        assert bkd.all_bool(values <= 0)

    def test_jacobian_shape(self, bkd) -> None:
        """Test Jacobian shape is (nqoi, nvars) = (1, 2)."""
        objective = self._create_objective(bkd)
        sample = bkd.asarray([[0.3], [0.7]])
        jac = objective.jacobian(sample)
        assert jac.shape == (1, 2)

    def test_nvars(self, bkd) -> None:
        """Test nvars returns 2."""
        objective = self._create_objective(bkd)
        assert objective.nvars() == 2

    def test_initial_iterates_shape(self, bkd) -> None:
        """Test initial iterates are pairs."""
        objective = self._create_objective(bkd)
        iterates, bounds_list = objective.initial_iterates_and_bounds()
        assert iterates.shape[0] == 2
        for b in bounds_list:
            assert b.shape == (2, 2)

    def test_jacobian_christoffel_derivative_checker(self, bkd) -> None:
        """Test Jacobian via DerivativeChecker with Christoffel weighting."""
        objective = self._create_objective(bkd, "christoffel")
        sample = bkd.asarray([[0.3], [0.7]])
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = float(checker.error_ratio(errors[0]).item())
        assert error_ratio < 1e-6

    def test_jacobian_pdf_derivative_checker(self, bkd) -> None:
        """Test Jacobian via DerivativeChecker with PDF weighting."""
        objective = self._create_objective(bkd, "pdf")
        sample = bkd.asarray([[0.3], [0.7]])
        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = float(checker.error_ratio(errors[0]).item())
        assert error_ratio < 1e-6


# =============================================================================
# TwoPointLejaSequence1D tests
# =============================================================================
class TestTwoPointLejaSequence1D:
    """Tests for LejaSequence1D with TwoPointLejaObjective."""

    def _create_leja(self, bkd):
        poly = LegendrePolynomial1D(bkd)
        weighting = ChristoffelWeighting(bkd)
        return LejaSequence1D(
            bkd,
            poly,
            weighting,
            bounds=(-1.0, 1.0),
            objective_class=TwoPointLejaObjective,
        )

    def test_extend_adds_two_points(self, bkd) -> None:
        """Test that each step adds 2 points."""
        leja = self._create_leja(bkd)
        assert leja.npoints() == 1
        leja.extend(2)
        assert leja.npoints() == 3
        leja.extend(2)
        assert leja.npoints() == 5

    def test_extend_validation(self, bkd) -> None:
        """Test ValueError for non-even n_new_points."""
        leja = self._create_leja(bkd)
        with pytest.raises(ValueError):
            leja.extend(3)

    def test_nested_property(self, bkd) -> None:
        """Test that two-point sequence is nested."""
        leja = self._create_leja(bkd)
        samples_3, _ = leja.quadrature_rule(3)
        samples_5, _ = leja.quadrature_rule(5)
        bkd.assert_allclose(samples_3, samples_5[:, :3], rtol=1e-12)

    def test_points_within_bounds(self, bkd) -> None:
        """Test that all points are within bounds."""
        leja = self._create_leja(bkd)
        samples, _ = leja.quadrature_rule(7)
        assert bkd.all_bool(samples >= -1.0 - 1e-10)
        assert bkd.all_bool(samples <= 1.0 + 1e-10)

    def test_quadrature_rule_shape(self, bkd) -> None:
        """Test quadrature_rule returns correct shapes."""
        leja = self._create_leja(bkd)
        samples, weights = leja.quadrature_rule(5)
        assert samples.shape == (1, 5)
        assert weights.shape == (5, 1)


# =============================================================================
# Parametrized integration accuracy tests (one-point)
# =============================================================================
class TestOnePointLejaIntegration:
    """Integration accuracy tests for one-point Leja sequences.

    Matches legacy test_leja_sequence: 5-point Leja sequence should
    exactly integrate x^4 (degree 4 polynomial with 5 points).

    Tests 4 combos:
    - Hermite + Christoffel
    - Legendre + Christoffel
    - Legendre + PDF(uniform)
    - Jacobi(1,1) + PDF(beta(2,2))
    """

    @pytest.mark.parametrize(
        "name,poly_type,weighting_type,bounds,exact",
        ONE_POINT_COMBOS,
    )
    def test_integration_x4(
        self,
        bkd,
        name: str,
        poly_type: str,
        weighting_type: str,
        bounds: Tuple[float, float],
        exact: float,
    ) -> None:
        """Test that 5-point one-point Leja sequence exactly integrates x^4."""
        poly, weighting = _create_poly_and_weighting(bkd, poly_type, weighting_type)
        leja = LejaSequence1D(
            bkd,
            poly,
            weighting,
            bounds=bounds,
            objective_class=LejaObjective,
        )
        samples, weights = leja.quadrature_rule(5)

        # Integrate x^4 using quadrature: sum(w_i * x_i^4)
        integral = bkd.sum(weights[:, 0] * samples[0, :] ** 4)
        bkd.assert_allclose(
            bkd.reshape(integral, (1,)),
            bkd.asarray([exact]),
            rtol=1e-6,
        )


# =============================================================================
# Parametrized integration accuracy tests (two-point)
# =============================================================================
class TestTwoPointLejaIntegration:
    """Integration accuracy tests for two-point Leja sequences.

    Same as one-point but using TwoPointLejaObjective. 5-point sequence
    (1 initial + 2 two-point steps) should exactly integrate x^4.

    Tests 4 combos:
    - Legendre + Christoffel
    - Legendre + PDF(uniform)
    - Jacobi(1,1) + PDF(beta(2,2))
    - Hermite + Christoffel
    """

    @pytest.mark.parametrize(
        "name,poly_type,weighting_type,bounds,exact",
        TWO_POINT_COMBOS,
    )
    @slow_test
    def test_integration_x4(
        self,
        bkd,
        name: str,
        poly_type: str,
        weighting_type: str,
        bounds: Tuple[float, float],
        exact: float,
    ) -> None:
        """Test that 5-point two-point Leja sequence exactly integrates x^4."""
        poly, weighting = _create_poly_and_weighting(bkd, poly_type, weighting_type)
        leja = LejaSequence1D(
            bkd,
            poly,
            weighting,
            bounds=bounds,
            objective_class=TwoPointLejaObjective,
        )
        samples, weights = leja.quadrature_rule(5)

        # Integrate x^4 using quadrature: sum(w_i * x_i^4)
        integral = bkd.sum(weights[:, 0] * samples[0, :] ** 4)
        bkd.assert_allclose(
            bkd.reshape(integral, (1,)),
            bkd.asarray([exact]),
            rtol=1e-6,
        )
