"""Dual-backend tests for LagrangeBasis1D.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import unittest
from typing import Any, Generic, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.surrogates.affine.univariate.lagrange import (
    LagrangeBasis1D,
    univariate_lagrange_polynomial,
    univariate_lagrange_first_derivative,
    univariate_lagrange_second_derivative,
)


class TestLagrangeBasis1D(Generic[Array], unittest.TestCase):
    """Tests for LagrangeBasis1D - dual backend."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _get_quadrature_rule(self) -> Any:
        """Return a simple Legendre Gauss quadrature rule."""
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(10)  # Ensure enough terms
        return poly.gauss_quadrature_rule

    def test_init(self) -> None:
        """Test initialization."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        self.assertEqual(basis.nterms(), 0)

    def test_set_nterms(self) -> None:
        """Test set_nterms method."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)

        basis.set_nterms(5)
        self.assertEqual(basis.nterms(), 5)

        basis.set_nterms(3)
        self.assertEqual(basis.nterms(), 3)

    def test_set_nterms_invalid(self) -> None:
        """Test set_nterms with invalid values."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)

        with self.assertRaises(ValueError):
            basis.set_nterms(0)

        with self.assertRaises(ValueError):
            basis.set_nterms(-1)

    def test_call_shape(self) -> None:
        """Test output shape of __call__."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        samples = self._bkd.asarray([[0.0, 0.5, -0.5, 0.25]])
        values = basis(samples)

        self.assertEqual(values.shape[0], 4)  # nsamples
        self.assertEqual(values.shape[1], 5)  # nterms

    def test_call_without_set_nterms(self) -> None:
        """Test that call raises error without set_nterms."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)

        samples = self._bkd.asarray([[0.0, 0.5]])
        with self.assertRaises(ValueError):
            basis(samples)

    def test_partition_of_unity(self) -> None:
        """Test that Lagrange basis sums to 1 everywhere."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        samples = self._bkd.linspace(-1.0, 1.0, 21)[None, :]
        values = basis(samples)
        row_sums = self._bkd.sum(values, axis=1)

        expected = self._bkd.ones((21,))
        self._bkd.assert_allclose(row_sums, expected, rtol=1e-10)

    def test_nodal_property(self) -> None:
        """Test that L_i(x_j) = delta_ij at the nodes."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        nterms = 5
        basis.set_nterms(nterms)

        # Get the quadrature points (the nodes)
        nodes, _ = quad_rule(nterms)  # shape: (1, nterms)

        # Evaluate at nodes
        values = basis(nodes)

        # Should be identity matrix
        expected = self._bkd.eye(nterms)
        self._bkd.assert_allclose(values, expected, rtol=1e-10)

    def test_jacobian_batch_shape(self) -> None:
        """Test jacobian_batch output shape."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        samples = self._bkd.asarray([[0.0, 0.5, -0.5]])
        jac = basis.jacobian_batch(samples)

        self.assertEqual(jac.shape[0], 3)  # nsamples
        self.assertEqual(jac.shape[1], 5)  # nterms

    def test_hessian_batch_shape(self) -> None:
        """Test hessian_batch output shape."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        samples = self._bkd.asarray([[0.0, 0.5, -0.5]])
        hess = basis.hessian_batch(samples)

        self.assertEqual(hess.shape[0], 3)  # nsamples
        self.assertEqual(hess.shape[1], 5)  # nterms

    def test_derivative_sum_zero(self) -> None:
        """Test that sum of derivatives is zero (constant preserving)."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        samples = self._bkd.linspace(-0.9, 0.9, 15)[None, :]
        jac = basis.jacobian_batch(samples)
        row_sums = self._bkd.sum(jac, axis=1)

        expected = self._bkd.zeros((15,))
        self._bkd.assert_allclose(row_sums, expected, atol=1e-10)

    def test_second_derivative_sum_zero(self) -> None:
        """Test that sum of second derivatives is zero."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        samples = self._bkd.linspace(-0.9, 0.9, 15)[None, :]
        hess = basis.hessian_batch(samples)
        row_sums = self._bkd.sum(hess, axis=1)

        expected = self._bkd.zeros((15,))
        self._bkd.assert_allclose(row_sums, expected, atol=1e-10)

    def test_derivatives_method(self) -> None:
        """Test derivatives method with different orders."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        samples = self._bkd.asarray([[0.0, 0.5]])

        # Order 0 = values
        vals = basis.derivatives(samples, 0)
        expected = basis(samples)
        self._bkd.assert_allclose(vals, expected, rtol=1e-12)

        # Order 1 = first derivative
        d1 = basis.derivatives(samples, 1)
        expected = basis.jacobian_batch(samples)
        self._bkd.assert_allclose(d1, expected, rtol=1e-12)

        # Order 2 = second derivative
        d2 = basis.derivatives(samples, 2)
        expected = basis.hessian_batch(samples)
        self._bkd.assert_allclose(d2, expected, rtol=1e-12)

    def test_derivatives_invalid_order(self) -> None:
        """Test derivatives method with invalid order."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        samples = self._bkd.asarray([[0.0, 0.5]])
        with self.assertRaises(ValueError):
            basis.derivatives(samples, 3)

    def test_gauss_quadrature_rule(self) -> None:
        """Test gauss_quadrature_rule method."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)

        points, weights = basis.gauss_quadrature_rule(5)
        self.assertEqual(points.shape, (1, 5))
        self.assertEqual(weights.shape, (5, 1))

    def test_repr(self) -> None:
        """Test string representation."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        repr_str = repr(basis)
        self.assertIn("LagrangeBasis1D", repr_str)
        self.assertIn("5", repr_str)

    def test_get_samples_shape(self) -> None:
        """Test get_samples returns correct shape."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)

        samples = basis.get_samples(5)
        self.assertEqual(samples.shape, (1, 5))

        samples = basis.get_samples(3)
        self.assertEqual(samples.shape, (1, 3))

    def test_get_samples_matches_quadrature(self) -> None:
        """Test get_samples returns same points as quadrature rule."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)

        samples_from_get = basis.get_samples(5)
        samples_from_quad, _ = basis.gauss_quadrature_rule(5)

        self._bkd.assert_allclose(samples_from_get, samples_from_quad, rtol=1e-12)

    def test_interpolation_protocol_compliance(self) -> None:
        """Test that LagrangeBasis1D satisfies InterpolationBasis1DProtocol."""
        from pyapprox.typing.surrogates.affine.protocols import (
            InterpolationBasis1DProtocol,
        )

        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)

        self.assertTrue(isinstance(basis, InterpolationBasis1DProtocol))

    def test_orthogonal_poly_not_interpolation_protocol(self) -> None:
        """Test that orthogonal polynomials do NOT satisfy InterpolationBasis1DProtocol."""
        from pyapprox.typing.surrogates.affine.protocols import (
            InterpolationBasis1DProtocol,
        )
        from pyapprox.typing.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(self._bkd)
        poly.set_nterms(5)

        # Should NOT satisfy the protocol (no get_samples method)
        self.assertFalse(isinstance(poly, InterpolationBasis1DProtocol))

    def test_numerical_first_derivative(self) -> None:
        """Test first derivative against finite differences."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        x0 = 0.3
        h = 1e-7

        samples_center = self._bkd.asarray([[x0]])
        samples_plus = self._bkd.asarray([[x0 + h]])
        samples_minus = self._bkd.asarray([[x0 - h]])

        vals_plus = basis(samples_plus)
        vals_minus = basis(samples_minus)
        fd_deriv = (vals_plus - vals_minus) / (2 * h)

        analytic_deriv = basis.jacobian_batch(samples_center)

        self._bkd.assert_allclose(analytic_deriv, fd_deriv, rtol=1e-5)

    def test_numerical_second_derivative(self) -> None:
        """Test second derivative against finite differences."""
        quad_rule = self._get_quadrature_rule()
        basis = LagrangeBasis1D(self._bkd, quad_rule)
        basis.set_nterms(5)

        x0 = 0.3
        h = 1e-5

        samples_center = self._bkd.asarray([[x0]])
        samples_plus = self._bkd.asarray([[x0 + h]])
        samples_minus = self._bkd.asarray([[x0 - h]])

        # Use first derivative for second derivative check
        d1_plus = basis.jacobian_batch(samples_plus)
        d1_minus = basis.jacobian_batch(samples_minus)
        fd_d2 = (d1_plus - d1_minus) / (2 * h)

        analytic_d2 = basis.hessian_batch(samples_center)

        self._bkd.assert_allclose(analytic_d2, fd_d2, rtol=1e-4)


class TestLagrangeFunctions(Generic[Array], unittest.TestCase):
    """Tests for standalone Lagrange functions."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_univariate_lagrange_polynomial_shape(self) -> None:
        """Test shape of univariate_lagrange_polynomial."""
        abscissa = self._bkd.asarray([-1.0, 0.0, 1.0])
        samples = self._bkd.asarray([0.5, -0.5, 0.0, 0.25])

        values = univariate_lagrange_polynomial(abscissa, samples, self._bkd)

        self.assertEqual(values.shape[0], 4)  # nsamples
        self.assertEqual(values.shape[1], 3)  # nabscissa

    def test_univariate_lagrange_polynomial_nodal(self) -> None:
        """Test nodal property: L_i(x_j) = delta_ij."""
        abscissa = self._bkd.asarray([-1.0, 0.0, 1.0])

        values = univariate_lagrange_polynomial(abscissa, abscissa, self._bkd)

        expected = self._bkd.eye(3)
        self._bkd.assert_allclose(values, expected, rtol=1e-10)

    def test_univariate_lagrange_first_derivative_shape(self) -> None:
        """Test shape of first derivative function."""
        abscissa = self._bkd.asarray([-1.0, 0.0, 1.0])
        samples = self._bkd.asarray([0.5, -0.5])

        derivs = univariate_lagrange_first_derivative(
            abscissa, samples, self._bkd
        )

        self.assertEqual(derivs.shape[0], 2)  # nsamples
        self.assertEqual(derivs.shape[1], 3)  # nabscissa

    def test_univariate_lagrange_second_derivative_shape(self) -> None:
        """Test shape of second derivative function."""
        abscissa = self._bkd.asarray([-1.0, 0.0, 1.0])
        samples = self._bkd.asarray([0.5, -0.5])

        derivs = univariate_lagrange_second_derivative(
            abscissa, samples, self._bkd
        )

        self.assertEqual(derivs.shape[0], 2)  # nsamples
        self.assertEqual(derivs.shape[1], 3)  # nabscissa

    def test_linear_interpolation_exact(self) -> None:
        """Test that linear functions are interpolated exactly."""
        # Two nodes = linear interpolation
        abscissa = self._bkd.asarray([0.0, 1.0])
        samples = self._bkd.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

        values = univariate_lagrange_polynomial(abscissa, samples, self._bkd)

        # L_0(x) = 1 - x, L_1(x) = x
        expected_L0 = self._bkd.asarray([1.0, 0.75, 0.5, 0.25, 0.0])
        expected_L1 = self._bkd.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

        self._bkd.assert_allclose(values[:, 0], expected_L0, rtol=1e-10)
        self._bkd.assert_allclose(values[:, 1], expected_L1, rtol=1e-10)


# NumPy backend tests
class TestLagrangeBasis1DNumpy(TestLagrangeBasis1D[NDArray[Any]]):
    """NumPy backend tests for LagrangeBasis1D."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLagrangeFunctionsNumpy(TestLagrangeFunctions[NDArray[Any]]):
    """NumPy backend tests for Lagrange functions."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestLagrangeBasis1DTorch(TestLagrangeBasis1D[torch.Tensor]):
    """PyTorch backend tests for LagrangeBasis1D."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestLagrangeFunctionsTorch(TestLagrangeFunctions[torch.Tensor]):
    """PyTorch backend tests for Lagrange functions."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
