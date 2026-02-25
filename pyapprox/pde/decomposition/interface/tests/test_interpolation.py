"""Tests for interpolation operators."""

import unittest

import numpy as np

from pyapprox.pde.decomposition.interface.interpolation import (
    InterpolationOperator,
    RestrictionOperator,
    lagrange_interpolation_matrix,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestLagrangeInterpolationMatrix(unittest.TestCase):
    """Tests for lagrange_interpolation_matrix function."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_identity_at_source_points(self):
        """Test that interpolation is exact at source points."""
        source_pts = self.bkd.asarray([0.0, 0.5, 1.0])
        target_pts = source_pts  # Same points

        matrix = lagrange_interpolation_matrix(source_pts, target_pts, self.bkd)

        # Should be identity matrix
        np.testing.assert_allclose(matrix, np.eye(3), rtol=1e-12)

    def test_polynomial_exactness(self):
        """Test that interpolation is exact for polynomials."""
        source_pts = self.bkd.asarray([0.0, 0.5, 1.0])  # 3 points -> degree 2
        target_pts = self.bkd.asarray([0.25, 0.75])

        matrix = lagrange_interpolation_matrix(source_pts, target_pts, self.bkd)

        # Test polynomial: p(x) = x^2 + x + 1
        def poly(x):
            return x**2 + x + 1

        source_vals = poly(source_pts)
        target_vals = matrix @ source_vals
        expected_vals = poly(target_pts)

        np.testing.assert_allclose(target_vals, expected_vals, rtol=1e-12)

    def test_linear_interpolation(self):
        """Test linear interpolation with 2 points."""
        source_pts = self.bkd.asarray([0.0, 1.0])
        target_pts = self.bkd.asarray([0.25, 0.5, 0.75])

        matrix = lagrange_interpolation_matrix(source_pts, target_pts, self.bkd)

        # Linear function: f(x) = 2x + 1
        source_vals = self.bkd.asarray([1.0, 3.0])  # f(0)=1, f(1)=3
        target_vals = matrix @ source_vals
        expected_vals = self.bkd.asarray([1.5, 2.0, 2.5])

        np.testing.assert_allclose(target_vals, expected_vals, rtol=1e-12)


class TestInterpolationOperator(unittest.TestCase):
    """Tests for InterpolationOperator class."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_apply(self):
        """Test apply method."""
        source_pts = self.bkd.asarray([0.0, 0.5, 1.0])
        target_pts = self.bkd.asarray([0.25, 0.75])

        op = InterpolationOperator(source_pts, target_pts, self.bkd)

        self.assertEqual(op.n_source(), 3)
        self.assertEqual(op.n_target(), 2)

        # Test with polynomial
        source_vals = self.bkd.asarray([1.0, 1.25, 2.0])  # x^2 + 1 at 0, 0.5, 1
        target_vals = op.apply(source_vals)

        expected = self.bkd.asarray([1.0625, 1.5625])  # x^2 + 1 at 0.25, 0.75
        np.testing.assert_allclose(target_vals, expected, rtol=1e-10)

    def test_matrix_property(self):
        """Test that matrix property returns correct matrix."""
        source_pts = self.bkd.asarray([0.0, 1.0])
        target_pts = self.bkd.asarray([0.5])

        op = InterpolationOperator(source_pts, target_pts, self.bkd)
        matrix = op.matrix()

        self.assertEqual(matrix.shape, (1, 2))

        # For midpoint with 2 source points, weights should be [0.5, 0.5]
        np.testing.assert_allclose(matrix[0, :], [0.5, 0.5], rtol=1e-12)


class TestRestrictionOperator(unittest.TestCase):
    """Tests for RestrictionOperator class."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_restriction_square(self):
        """Test restriction when dimensions match."""
        source_pts = self.bkd.asarray([0.0, 0.5, 1.0])
        target_pts = self.bkd.asarray([0.0, 0.5, 1.0])

        restrict_op = RestrictionOperator(source_pts, target_pts, self.bkd)

        # Should be identity
        target_vals = self.bkd.asarray([1.0, 2.0, 3.0])
        source_vals = restrict_op.apply(target_vals)

        np.testing.assert_allclose(source_vals, target_vals, rtol=1e-10)

    def test_restriction_overdetermined(self):
        """Test restriction with more target points than source."""
        source_pts = self.bkd.asarray([0.0, 1.0])  # 2 DOFs
        target_pts = self.bkd.asarray([0.0, 0.5, 1.0])  # 3 points

        restrict_op = RestrictionOperator(source_pts, target_pts, self.bkd)

        # Linear function: f(x) = 2x + 1
        target_vals = self.bkd.asarray([1.0, 2.0, 3.0])

        # Restriction should find best-fit linear coefficients
        source_vals = restrict_op.apply(target_vals)

        # Should recover f(0) = 1, f(1) = 3
        np.testing.assert_allclose(source_vals, [1.0, 3.0], rtol=1e-10)

    def test_interp_restrict_roundtrip(self):
        """Test that interpolation followed by restriction recovers coefficients."""
        source_pts = self.bkd.asarray([0.0, 0.5, 1.0])
        target_pts = self.bkd.asarray([0.25, 0.5, 0.75])

        interp_op = InterpolationOperator(source_pts, target_pts, self.bkd)
        restrict_op = RestrictionOperator(source_pts, target_pts, self.bkd)

        source_vals = self.bkd.asarray([1.0, 2.0, 3.0])

        # Interpolate then restrict
        target_vals = interp_op.apply(source_vals)
        recovered_vals = restrict_op.apply(target_vals)

        np.testing.assert_allclose(recovered_vals, source_vals, rtol=1e-10)


class TestInterface1D(unittest.TestCase):
    """Tests for Interface1D class."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_basic_properties(self):
        """Test basic interface properties."""
        from pyapprox.pde.decomposition.interface import Interface1D

        interface = Interface1D(
            self.bkd,
            interface_id=0,
            subdomain_ids=(0, 1),
            interface_point=0.5,
        )

        self.assertEqual(interface.interface_id(), 0)
        self.assertEqual(interface.ndofs(), 1)
        self.assertEqual(interface.npts(), 1)
        self.assertEqual(interface.subdomain_ids(), (0, 1))
        self.assertEqual(interface.physical_point(), 0.5)

    def test_normals(self):
        """Test normal vectors for 1D interface."""
        from pyapprox.pde.decomposition.interface import Interface1D

        interface = Interface1D(
            self.bkd,
            interface_id=0,
            subdomain_ids=(0, 1),
            interface_point=0.5,
        )

        # Left domain (0) normal points right (+1)
        normal_0 = interface.normal(0)
        np.testing.assert_allclose(normal_0, [1.0])

        # Right domain (1) normal points left (-1)
        normal_1 = interface.normal(1)
        np.testing.assert_allclose(normal_1, [-1.0])

    def test_invalid_subdomain(self):
        """Test error for invalid subdomain ID."""
        from pyapprox.pde.decomposition.interface import Interface1D

        interface = Interface1D(
            self.bkd,
            interface_id=0,
            subdomain_ids=(0, 1),
            interface_point=0.5,
        )

        with self.assertRaises(ValueError):
            interface.normal(2)  # Invalid subdomain


if __name__ == "__main__":
    unittest.main()
