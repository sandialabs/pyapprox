"""Dual-backend tests for LagrangeBasis1D.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import pytest

from pyapprox.surrogates.affine.univariate.lagrange import (
    LagrangeBasis1D,
    univariate_lagrange_first_derivative,
    univariate_lagrange_polynomial,
    univariate_lagrange_second_derivative,
)


class TestLagrangeBasis1D:
    """Tests for LagrangeBasis1D - dual backend."""

    def _get_quadrature_rule(self, bkd):
        """Return a simple Legendre Gauss quadrature rule."""
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(10)  # Ensure enough terms
        return poly.gauss_quadrature_rule

    def test_init(self, bkd) -> None:
        """Test initialization."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        assert basis.nterms() == 0

    def test_set_nterms(self, bkd) -> None:
        """Test set_nterms method."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)

        basis.set_nterms(5)
        assert basis.nterms() == 5

        basis.set_nterms(3)
        assert basis.nterms() == 3

    def test_set_nterms_invalid(self, bkd) -> None:
        """Test set_nterms with invalid values."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)

        with pytest.raises(ValueError):
            basis.set_nterms(0)

        with pytest.raises(ValueError):
            basis.set_nterms(-1)

    @pytest.mark.slow_on("TorchBkd")
    def test_call_shape(self, bkd) -> None:
        """Test output shape of __call__."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        samples = bkd.asarray([[0.0, 0.5, -0.5, 0.25]])
        values = basis(samples)

        assert values.shape[0] == 4  # nsamples
        assert values.shape[1] == 5  # nterms

    def test_call_without_set_nterms(self, bkd) -> None:
        """Test that call raises error without set_nterms."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)

        samples = bkd.asarray([[0.0, 0.5]])
        with pytest.raises(ValueError):
            basis(samples)

    def test_partition_of_unity(self, bkd) -> None:
        """Test that Lagrange basis sums to 1 everywhere."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        samples = bkd.linspace(-1.0, 1.0, 21)[None, :]
        values = basis(samples)
        row_sums = bkd.sum(values, axis=1)

        expected = bkd.ones((21,))
        bkd.assert_allclose(row_sums, expected, rtol=1e-10)

    def test_nodal_property(self, bkd) -> None:
        """Test that L_i(x_j) = delta_ij at the nodes."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        nterms = 5
        basis.set_nterms(nterms)

        # Get the quadrature points (the nodes)
        nodes, _ = quad_rule(nterms)  # shape: (1, nterms)

        # Evaluate at nodes
        values = basis(nodes)

        # Should be identity matrix
        expected = bkd.eye(nterms)
        bkd.assert_allclose(values, expected, rtol=1e-10)

    def test_jacobian_batch_shape(self, bkd) -> None:
        """Test jacobian_batch output shape."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        samples = bkd.asarray([[0.0, 0.5, -0.5]])
        jac = basis.jacobian_batch(samples)

        assert jac.shape[0] == 3  # nsamples
        assert jac.shape[1] == 5  # nterms

    def test_hessian_batch_shape(self, bkd) -> None:
        """Test hessian_batch output shape."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        samples = bkd.asarray([[0.0, 0.5, -0.5]])
        hess = basis.hessian_batch(samples)

        assert hess.shape[0] == 3  # nsamples
        assert hess.shape[1] == 5  # nterms

    def test_derivative_sum_zero(self, bkd) -> None:
        """Test that sum of derivatives is zero (constant preserving)."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        samples = bkd.linspace(-0.9, 0.9, 15)[None, :]
        jac = basis.jacobian_batch(samples)
        row_sums = bkd.sum(jac, axis=1)

        expected = bkd.zeros((15,))
        bkd.assert_allclose(row_sums, expected, atol=1e-10)

    def test_second_derivative_sum_zero(self, bkd) -> None:
        """Test that sum of second derivatives is zero."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        samples = bkd.linspace(-0.9, 0.9, 15)[None, :]
        hess = basis.hessian_batch(samples)
        row_sums = bkd.sum(hess, axis=1)

        expected = bkd.zeros((15,))
        bkd.assert_allclose(row_sums, expected, atol=1e-10)

    def test_derivatives_method(self, bkd) -> None:
        """Test derivatives method with different orders."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        samples = bkd.asarray([[0.0, 0.5]])

        # Order 0 = values
        vals = basis.derivatives(samples, 0)
        expected = basis(samples)
        bkd.assert_allclose(vals, expected, rtol=1e-12)

        # Order 1 = first derivative
        d1 = basis.derivatives(samples, 1)
        expected = basis.jacobian_batch(samples)
        bkd.assert_allclose(d1, expected, rtol=1e-12)

        # Order 2 = second derivative
        d2 = basis.derivatives(samples, 2)
        expected = basis.hessian_batch(samples)
        bkd.assert_allclose(d2, expected, rtol=1e-12)

    def test_derivatives_invalid_order(self, bkd) -> None:
        """Test derivatives method with invalid order."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        samples = bkd.asarray([[0.0, 0.5]])
        with pytest.raises(ValueError):
            basis.derivatives(samples, 3)

    def test_quadrature_rule(self, bkd) -> None:
        """Test quadrature_rule method."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        points, weights = basis.quadrature_rule()
        assert points.shape == (1, 5)
        assert weights.shape == (5, 1)

    def test_quadrature_rule_before_set_nterms(self, bkd) -> None:
        """Test quadrature_rule raises error if set_nterms not called."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)

        with pytest.raises(ValueError):
            basis.quadrature_rule()

    def test_repr(self, bkd) -> None:
        """Test string representation."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        repr_str = repr(basis)
        assert "LagrangeBasis1D" in repr_str
        assert "5" in repr_str

    def test_get_samples_shape(self, bkd) -> None:
        """Test get_samples returns correct shape."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)

        samples = basis.get_samples(5)
        assert samples.shape == (1, 5)

        samples = basis.get_samples(3)
        assert samples.shape == (1, 3)

    def test_get_samples_matches_quadrature(self, bkd) -> None:
        """Test get_samples returns same points as quadrature rule."""
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        samples_from_get = basis.get_samples(5)
        samples_from_quad, _ = basis.quadrature_rule()

        bkd.assert_allclose(samples_from_get, samples_from_quad, rtol=1e-12)

    def test_interpolation_protocol_compliance(self, bkd) -> None:
        """Test that LagrangeBasis1D satisfies InterpolationBasis1DProtocol."""
        from pyapprox.surrogates.affine.protocols import (
            InterpolationBasis1DProtocol,
        )

        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)

        assert isinstance(basis, InterpolationBasis1DProtocol)

    def test_orthogonal_poly_not_interpolation_protocol(self, bkd) -> None:
        """
        Test that orthogonal polynomials do NOT satisfy
        InterpolationBasis1DProtocol.
        """
        from pyapprox.surrogates.affine.protocols import (
            InterpolationBasis1DProtocol,
        )
        from pyapprox.surrogates.affine.univariate import (
            LegendrePolynomial1D,
        )

        poly = LegendrePolynomial1D(bkd)
        poly.set_nterms(5)

        # Should NOT satisfy the protocol (no get_samples method)
        assert not isinstance(poly, InterpolationBasis1DProtocol)

    def test_numerical_first_derivative(self, bkd) -> None:
        """Test first derivative against finite differences."""
        # TODO use derivative checker for testing first derivative
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        x0 = 0.3
        h = 1e-7

        samples_center = bkd.asarray([[x0]])
        samples_plus = bkd.asarray([[x0 + h]])
        samples_minus = bkd.asarray([[x0 - h]])

        vals_plus = basis(samples_plus)
        vals_minus = basis(samples_minus)
        fd_deriv = (vals_plus - vals_minus) / (2 * h)

        analytic_deriv = basis.jacobian_batch(samples_center)

        bkd.assert_allclose(analytic_deriv, fd_deriv, rtol=1e-5)

    def test_numerical_second_derivative(self, bkd) -> None:
        """Test second derivative against finite differences."""
        # TODO use derivative checker for testing second derivative
        quad_rule = self._get_quadrature_rule(bkd)
        basis = LagrangeBasis1D(bkd, quad_rule)
        basis.set_nterms(5)

        x0 = 0.3
        h = 1e-5

        samples_center = bkd.asarray([[x0]])
        samples_plus = bkd.asarray([[x0 + h]])
        samples_minus = bkd.asarray([[x0 - h]])

        # Use first derivative for second derivative check
        d1_plus = basis.jacobian_batch(samples_plus)
        d1_minus = basis.jacobian_batch(samples_minus)
        fd_d2 = (d1_plus - d1_minus) / (2 * h)

        analytic_d2 = basis.hessian_batch(samples_center)

        bkd.assert_allclose(analytic_d2, fd_d2, rtol=1e-4)


class TestLagrangeFunctions:
    """Tests for standalone Lagrange functions."""

    def test_univariate_lagrange_polynomial_shape(self, bkd) -> None:
        """Test shape of univariate_lagrange_polynomial."""
        abscissa = bkd.asarray([-1.0, 0.0, 1.0])
        samples = bkd.asarray([0.5, -0.5, 0.0, 0.25])

        values = univariate_lagrange_polynomial(abscissa, samples, bkd)

        assert values.shape[0] == 4  # nsamples
        assert values.shape[1] == 3  # nabscissa

    def test_univariate_lagrange_polynomial_nodal(self, bkd) -> None:
        """Test nodal property: L_i(x_j) = delta_ij."""
        abscissa = bkd.asarray([-1.0, 0.0, 1.0])

        values = univariate_lagrange_polynomial(abscissa, abscissa, bkd)

        expected = bkd.eye(3)
        bkd.assert_allclose(values, expected, rtol=1e-10)

    def test_univariate_lagrange_first_derivative_shape(self, bkd) -> None:
        """Test shape of first derivative function."""
        abscissa = bkd.asarray([-1.0, 0.0, 1.0])
        samples = bkd.asarray([0.5, -0.5])

        derivs = univariate_lagrange_first_derivative(abscissa, samples, bkd)

        assert derivs.shape[0] == 2  # nsamples
        assert derivs.shape[1] == 3  # nabscissa

    def test_univariate_lagrange_second_derivative_shape(self, bkd) -> None:
        """Test shape of second derivative function."""
        abscissa = bkd.asarray([-1.0, 0.0, 1.0])
        samples = bkd.asarray([0.5, -0.5])

        derivs = univariate_lagrange_second_derivative(abscissa, samples, bkd)

        assert derivs.shape[0] == 2  # nsamples
        assert derivs.shape[1] == 3  # nabscissa

    def test_linear_interpolation_exact(self, bkd) -> None:
        """Test that linear functions are interpolated exactly."""
        # Two nodes = linear interpolation
        abscissa = bkd.asarray([0.0, 1.0])
        samples = bkd.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

        values = univariate_lagrange_polynomial(abscissa, samples, bkd)

        # L_0(x) = 1 - x, L_1(x) = x
        expected_L0 = bkd.asarray([1.0, 0.75, 0.5, 0.25, 0.0])
        expected_L1 = bkd.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

        bkd.assert_allclose(values[:, 0], expected_L0, rtol=1e-10)
        bkd.assert_allclose(values[:, 1], expected_L1, rtol=1e-10)
