"""Tests for interface basis implementations."""


import numpy as np
import pytest

from pyapprox.pde.decomposition.interface.basis import (
    LegendreInterfaceBasis1D,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestLegendreInterfaceBasis1D:
    """Tests for LegendreInterfaceBasis1D."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bkd = NumpyBkd()

    def test_initialization(self, bkd):
        """Test basic initialization."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=5, physical_bounds=(0.0, 1.0))

        # degree 5 -> 6 Gauss-Legendre points (degree + 1)
        assert basis.ndofs() == 6
        assert basis.npts() == 6
        assert basis.degree() == 5

    def test_minimum_degree(self, bkd):
        """Test that degree must be at least 1."""
        # degree=1 should work (gives 2 Gauss points)
        basis = LegendreInterfaceBasis1D(self.bkd, degree=1, physical_bounds=(0.0, 1.0))
        assert basis.ndofs() == 2

        # degree=0 should fail
        with pytest.raises(ValueError):
            LegendreInterfaceBasis1D(self.bkd, degree=0, physical_bounds=(0.0, 1.0))

    def test_reference_points_exclude_endpoints(self, bkd):
        """Test that Gauss-Legendre points are strictly interior to [-1, 1]."""
        basis = LegendreInterfaceBasis1D(
            self.bkd, degree=4, physical_bounds=(-1.0, 1.0)
        )
        ref_pts = basis.reference_points()

        # Shape should be (1, npts) where npts = degree + 1 = 5
        assert ref_pts.shape == (1, 5)

        # Gauss-Legendre points are strictly interior to [-1, 1]
        assert np.all(ref_pts > -1.0)
        assert np.all(ref_pts < 1.0)

    def test_physical_points_mapping(self, bkd):
        """Test mapping from reference to physical coordinates."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=4, physical_bounds=(2.0, 5.0))
        phys_pts = basis.physical_points()

        # Points should be strictly interior to [2, 5]
        assert np.all(phys_pts > 2.0)
        assert np.all(phys_pts < 5.0)

    def test_nodal_evaluation(self, bkd):
        """Test that nodal evaluation returns coefficients unchanged."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=5, physical_bounds=(0.0, 1.0))
        # degree=5 -> 6 DOFs
        coeffs = self.bkd.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        values = basis.evaluate(coeffs)
        np.testing.assert_allclose(values, coeffs)

    def test_interpolation_at_nodes(self, bkd):
        """Test interpolation recovers nodal values at nodes."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=5, physical_bounds=(0.0, 1.0))
        # degree=5 -> 6 DOFs
        coeffs = self.bkd.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        # Evaluate at the nodal points
        phys_pts = basis.physical_points()[0, :]  # Shape: (npts,)
        values = basis.evaluate_at_points(coeffs, phys_pts)

        np.testing.assert_allclose(values, coeffs, rtol=1e-12)

    def test_interpolation_polynomial_exactness(self, bkd):
        """Test that interpolation is exact for polynomials up to degree."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=4, physical_bounds=(0.0, 2.0))

        # Test polynomial: p(x) = x^2 + 2x + 1
        def poly(x):
            return x**2 + 2 * x + 1

        # Get nodal values
        phys_pts = basis.physical_points()[0, :]
        coeffs = self.bkd.asarray(poly(phys_pts))

        # Evaluate at some other points
        eval_pts = self.bkd.asarray([0.3, 0.7, 1.2, 1.8])
        interp_values = basis.evaluate_at_points(coeffs, eval_pts)
        exact_values = poly(eval_pts)

        np.testing.assert_allclose(interp_values, exact_values, rtol=1e-10)

    def test_interpolation_matrix(self, bkd):
        """Test interpolation matrix gives same result as evaluate_at_points."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=5, physical_bounds=(0.0, 1.0))
        # degree=5 -> 6 DOFs
        coeffs = self.bkd.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        eval_pts = self.bkd.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
        interp_matrix = basis.interpolation_matrix_to_points(eval_pts)
        values_matrix = interp_matrix @ coeffs
        values_direct = basis.evaluate_at_points(coeffs, eval_pts)

        np.testing.assert_allclose(values_matrix, values_direct, rtol=1e-12)

    def test_quadrature_integration(self, bkd):
        """Test quadrature integration accuracy.

        Gauss-Legendre with n points integrates polynomials of degree 2n-1 exactly.
        For degree d, we have d+1 Gauss points, which integrates degree 2d+1 exactly.
        """
        # Use degree=3, giving 4 Gauss points, which integrates degree 7 exactly
        basis = LegendreInterfaceBasis1D(self.bkd, degree=3, physical_bounds=(0.0, 1.0))

        phys_pts = basis.physical_points()[0, :]

        # Test constant = 1: integral from 0 to 1 = 1
        values_const = self.bkd.ones(phys_pts.shape)
        integral_const = basis.integrate(values_const)
        np.testing.assert_allclose(integral_const, 1.0, rtol=1e-12)

        # Test linear function: f(x) = 2x, integral from 0 to 1 = 1
        values_linear = 2 * phys_pts
        integral_linear = basis.integrate(values_linear)
        np.testing.assert_allclose(integral_linear, 1.0, rtol=1e-12)

        # Test quadratic: f(x) = 3x^2, integral from 0 to 1 = 1
        values_quad = 3 * phys_pts**2
        integral_quad = basis.integrate(values_quad)
        np.testing.assert_allclose(integral_quad, 1.0, rtol=1e-12)

        # Test degree 7 polynomial (max for 4 Gauss points):
        # f(x) = 8x^7, integral from 0 to 1 = 1
        values_deg7 = 8 * phys_pts**7
        integral_deg7 = basis.integrate(values_deg7)
        np.testing.assert_allclose(integral_deg7, 1.0, rtol=1e-12)

    def test_quadrature_weights_sum(self, bkd):
        """Test that quadrature weights sum to domain length."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=5, physical_bounds=(2.0, 5.0))
        weights = basis.quadrature_weights()

        # Gauss-Legendre weights sum to domain length (exactly)
        assert np.all(weights > 0)
        np.testing.assert_allclose(np.sum(weights), 3.0, rtol=1e-12)  # 5.0 - 2.0 = 3.0

    @pytest.mark.parametrize(
        "start,end",
        [(0.0, 1.0), (-1.0, 1.0), (10.0, 20.0), (-5.0, 5.0)],
    )
    def test_different_physical_bounds(self, bkd, start, end):
        """Test with various physical bounds."""
        basis = LegendreInterfaceBasis1D(
            self.bkd, degree=4, physical_bounds=(start, end)
        )
        phys_pts = basis.physical_points()[0, :]

        # Points should be strictly interior
        assert np.all(phys_pts > start)
        assert np.all(phys_pts < end)

    def test_repr(self, bkd):
        """Test string representation."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=5, physical_bounds=(0.0, 1.0))
        repr_str = repr(basis)

        assert "LegendreInterfaceBasis1D" in repr_str
        assert "degree=5" in repr_str
        assert "ndofs=6" in repr_str  # degree + 1 = 6


class TestLegendreInterfaceBasis1DEdgeCases:
    """Edge case tests for LegendreInterfaceBasis1D."""

    def setup_method(self):
        self.bkd = NumpyBkd()

    def test_minimum_working_degree(self, bkd):
        """Test degree=1 gives exactly 2 DOFs (minimum valid degree)."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=1, physical_bounds=(0.0, 1.0))
        assert basis.ndofs() == 2  # degree + 1 = 2
        assert basis.npts() == 2

    def test_coefficient_shape_validation(self, bkd):
        """Test that wrong coefficient shape raises error."""
        basis = LegendreInterfaceBasis1D(self.bkd, degree=5, physical_bounds=(0.0, 1.0))
        # degree=5 -> 6 DOFs

        with pytest.raises(ValueError):
            basis.evaluate(self.bkd.asarray([1.0, 2.0]))  # Wrong size (need 6)

        with pytest.raises(ValueError):
            basis.evaluate_at_points(
                self.bkd.asarray([1.0, 2.0]),  # Wrong size (need 6)
                self.bkd.asarray([0.5]),
            )
