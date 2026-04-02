"""Tests for elliptical coordinate transform."""

import math

import numpy as np
import pytest

from pyapprox.pde.collocation.mesh.transforms.elliptical import (
    EllipticalTransform,
)


class TestEllipticalTransform:
    """Tests for EllipticalTransform."""

    def test_elliptical_basic_mapping(self, bkd):
        """Test basic elliptical to Cartesian mapping."""
        a = 2.0
        transform = EllipticalTransform(
            (0.5, 3.0), (0.0, math.pi), a, bkd, from_reference=False
        )

        # Test specific points
        # u=1, v=0 -> x = a*cosh(1)*cos(0) = a*cosh(1), y = 0
        # u=1, v=pi/2 -> x = 0, y = a*sinh(1)
        u_val = 1.0
        ref_pts = bkd.asarray([[u_val, u_val], [0.0, math.pi / 2]])
        phys_pts = transform.map_to_physical(ref_pts)

        cosh_u = np.cosh(u_val)
        sinh_u = np.sinh(u_val)
        expected_x = bkd.asarray([a * cosh_u, 0.0])
        expected_y = bkd.asarray([0.0, a * sinh_u])

        bkd.assert_allclose(phys_pts[0, :], expected_x, atol=1e-14)
        bkd.assert_allclose(phys_pts[1, :], expected_y, atol=1e-14)

    def test_elliptical_inverse_mapping(self, bkd):
        """Test inverse mapping recovers original points."""
        a = 1.5
        transform = EllipticalTransform(
            (0.3, 2.0), (0.1, math.pi - 0.1), a, bkd, from_reference=False
        )

        # Test points avoiding singularities
        ref_pts = bkd.asarray([[0.5, 1.0, 1.5], [0.5, 1.0, 2.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        ref_pts_back = transform.map_to_reference(phys_pts)

        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-12)

    def test_elliptical_from_reference_mapping(self, bkd):
        """Test from_reference=True mode with affine mapping."""
        a = 2.0
        u_bounds = (0.5, 2.5)
        v_bounds = (0.1, math.pi - 0.1)

        # Create transform with from_reference=True (default)
        transform = EllipticalTransform(u_bounds, v_bounds, a, bkd)

        # Reference points in [-1, 1]^2
        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])

        # Expected elliptical coordinates after affine mapping
        # u = u_scale * xi + u_shift = (u_max - u_min)/2 * xi + (u_max + u_min)/2
        # At xi=-1: u = u_min = 0.5
        # At xi=0: u = (u_max + u_min)/2 = 1.5
        # At xi=1: u = u_max = 2.5
        u_expected = np.array([0.5, 1.5, 2.5])
        v_expected = np.array([0.1, (math.pi - 0.1 + 0.1) / 2, math.pi - 0.1])

        # Expected physical points
        x_expected = a * np.cosh(u_expected) * np.cos(v_expected)
        y_expected = a * np.sinh(u_expected) * np.sin(v_expected)

        phys_pts = transform.map_to_physical(ref_pts)

        bkd.assert_allclose(phys_pts[0, :], bkd.asarray(x_expected), atol=1e-12)
        bkd.assert_allclose(phys_pts[1, :], bkd.asarray(y_expected), atol=1e-12)

        # Test inverse: physical -> reference should give back original ref_pts
        ref_pts_back = transform.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-12)

    def test_elliptical_jacobian_determinant(self, bkd):
        """Test Jacobian determinant formula: a^2 * (sinh^2(u) + sin^2(v))."""
        a = 2.0
        transform = EllipticalTransform(
            (0.5, 3.0), (0.0, math.pi), a, bkd, from_reference=False
        )

        ref_pts = bkd.asarray([[0.5, 1.0, 2.0], [0.3, 1.0, 2.5]])
        jac_det = transform.jacobian_determinant(ref_pts)

        # Expected: a^2 * (sinh^2(u) + sin^2(v))
        u = bkd.to_numpy(ref_pts[0, :])
        v = bkd.to_numpy(ref_pts[1, :])
        expected = a**2 * (np.sinh(u) ** 2 + np.sin(v) ** 2)
        bkd.assert_allclose(jac_det, bkd.asarray(expected), atol=1e-14)

    def test_elliptical_jacobian_matrix_consistency(self, bkd):
        """Test Jacobian matrix determinant matches jacobian_determinant."""
        a = 1.0
        transform = EllipticalTransform(
            (0.3, 2.0), (0.1, math.pi - 0.1), a, bkd, from_reference=False
        )

        ref_pts = bkd.asarray([[0.5, 1.0, 1.5], [0.5, 1.5, 2.0]])
        jac_mat = transform.jacobian_matrix(ref_pts)
        jac_det = transform.jacobian_determinant(ref_pts)

        # Compute determinant from matrix: ad - bc
        det_from_mat = (
            jac_mat[:, 0, 0] * jac_mat[:, 1, 1] - jac_mat[:, 0, 1] * jac_mat[:, 1, 0]
        )
        bkd.assert_allclose(det_from_mat, jac_det, atol=1e-14)

    def test_elliptical_scale_factors(self, bkd):
        """Test scale factors h_u = h_v = a * sqrt(sinh^2(u) + sin^2(v))."""
        a = 2.0
        transform = EllipticalTransform(
            (0.5, 3.0), (0.0, math.pi), a, bkd, from_reference=False
        )

        ref_pts = bkd.asarray([[0.5, 1.0, 2.0], [0.3, 1.0, 2.5]])
        scale = transform.scale_factors(ref_pts)

        # Expected: h_u = h_v = a * sqrt(sinh^2(u) + sin^2(v))
        u = bkd.to_numpy(ref_pts[0, :])
        v = bkd.to_numpy(ref_pts[1, :])
        h_expected = a * np.sqrt(np.sinh(u) ** 2 + np.sin(v) ** 2)

        bkd.assert_allclose(scale[:, 0], bkd.asarray(h_expected), atol=1e-14)
        bkd.assert_allclose(scale[:, 1], bkd.asarray(h_expected), atol=1e-14)
        # Verify h_u = h_v
        bkd.assert_allclose(scale[:, 0], scale[:, 1], atol=1e-14)

    def test_elliptical_unit_basis_orthonormality(self, bkd):
        """Test unit curvilinear basis vectors are orthonormal."""
        a = 1.5
        transform = EllipticalTransform(
            (0.3, 2.0), (0.1, math.pi - 0.1), a, bkd, from_reference=False
        )

        ref_pts = bkd.asarray([[0.5, 1.0, 1.5], [0.5, 1.0, 2.0]])
        basis = transform.unit_curvilinear_basis(ref_pts)

        npts = ref_pts.shape[1]

        # Check each basis vector has unit length
        for ii in range(2):
            e_i = basis[:, :, ii]  # (npts, 2)
            norm_sq = bkd.sum(e_i**2, axis=1)
            bkd.assert_allclose(norm_sq, bkd.ones((npts,)), atol=1e-14)

        # Check orthogonality: e_u . e_v = 0
        e_u = basis[:, :, 0]
        e_v = basis[:, :, 1]
        dot = bkd.sum(e_u * e_v, axis=1)
        bkd.assert_allclose(dot, bkd.zeros((npts,)), atol=1e-14)

    def test_elliptical_gradient_factors_consistency(self, bkd):
        """Test gradient factors equal unit_basis / scale_factors."""
        a = 2.0
        transform = EllipticalTransform(
            (0.5, 3.0), (0.0, math.pi), a, bkd, from_reference=False
        )

        ref_pts = bkd.asarray([[0.5, 1.0], [0.5, 1.5]])
        grad_factors = transform.gradient_factors(ref_pts)
        unit_basis = transform.unit_curvilinear_basis(ref_pts)
        scale = transform.scale_factors(ref_pts)

        # gradient_factors should equal unit_basis / scale[:, None, :]
        expected = unit_basis / scale[:, None, :]
        bkd.assert_allclose(grad_factors, expected, atol=1e-14)

    def test_elliptical_bounds_validation(self, bkd):
        """Test bounds validation."""
        # u_min <= 0 (singularity at focal points)
        with pytest.raises(ValueError):
            EllipticalTransform((0.0, 2.0), (0.0, math.pi), 1.0, bkd)

        with pytest.raises(ValueError):
            EllipticalTransform((-0.5, 2.0), (0.0, math.pi), 1.0, bkd)

        # u_max <= u_min
        with pytest.raises(ValueError):
            EllipticalTransform((2.0, 1.0), (0.0, math.pi), 1.0, bkd)

        # v out of [0, 2*pi]
        with pytest.raises(ValueError):
            EllipticalTransform((0.5, 2.0), (-0.1, math.pi), 1.0, bkd)

        with pytest.raises(ValueError):
            EllipticalTransform((0.5, 2.0), (0.0, 2 * math.pi + 0.1), 1.0, bkd)

        # v_max <= v_min
        with pytest.raises(ValueError):
            EllipticalTransform((0.5, 2.0), (2.0, 1.0), 1.0, bkd)

        # a <= 0
        with pytest.raises(ValueError):
            EllipticalTransform((0.5, 2.0), (0.0, math.pi), 0.0, bkd)

        with pytest.raises(ValueError):
            EllipticalTransform((0.5, 2.0), (0.0, math.pi), -1.0, bkd)

    def test_elliptical_constant_u_is_ellipse(self, bkd):
        """Verify that constant u curves are ellipses."""
        a = 1.0
        u_val = 1.0
        transform = EllipticalTransform(
            (0.5, 2.0), (0.0, 2 * math.pi - 0.01), a, bkd, from_reference=False
        )

        # Sample many v values at constant u
        v_vals = np.linspace(0.01, 2 * np.pi - 0.01, 50)
        u_vals = np.full_like(v_vals, u_val)
        ref_pts = bkd.asarray(np.array([u_vals, v_vals]))
        phys_pts = transform.map_to_physical(ref_pts)

        x = bkd.to_numpy(phys_pts[0, :])
        y = bkd.to_numpy(phys_pts[1, :])

        # For elliptical coords, constant u gives ellipse with:
        # semi-major axis = a * cosh(u), semi-minor axis = a * sinh(u)
        semi_major = a * np.cosh(u_val)
        semi_minor = a * np.sinh(u_val)

        # Check ellipse equation: (x/a_major)^2 + (y/a_minor)^2 = 1
        ellipse_eq = (x / semi_major) ** 2 + (y / semi_minor) ** 2
        bkd.assert_allclose(
            bkd.asarray(ellipse_eq), bkd.ones((len(v_vals),)), atol=1e-12
        )

    def test_elliptical_jacobian_via_finite_differences(self, bkd):
        """Verify Jacobian matrix via finite differences."""
        a = 2.0
        transform = EllipticalTransform(
            (0.5, 3.0), (0.1, math.pi - 0.1), a, bkd, from_reference=False
        )

        # Test at a single point
        u0, v0 = 1.0, 0.8
        ref_pt = bkd.asarray([[u0], [v0]])
        jac = transform.jacobian_matrix(ref_pt)

        # Finite difference approximation
        eps = 1e-7
        ref_u_plus = bkd.asarray([[u0 + eps], [v0]])
        ref_u_minus = bkd.asarray([[u0 - eps], [v0]])
        ref_v_plus = bkd.asarray([[u0], [v0 + eps]])
        ref_v_minus = bkd.asarray([[u0], [v0 - eps]])

        phys_u_plus = transform.map_to_physical(ref_u_plus)
        phys_u_minus = transform.map_to_physical(ref_u_minus)
        phys_v_plus = transform.map_to_physical(ref_v_plus)
        phys_v_minus = transform.map_to_physical(ref_v_minus)

        dx_du_fd = (phys_u_plus[0, 0] - phys_u_minus[0, 0]) / (2 * eps)
        dy_du_fd = (phys_u_plus[1, 0] - phys_u_minus[1, 0]) / (2 * eps)
        dx_dv_fd = (phys_v_plus[0, 0] - phys_v_minus[0, 0]) / (2 * eps)
        dy_dv_fd = (phys_v_plus[1, 0] - phys_v_minus[1, 0]) / (2 * eps)

        bkd.assert_allclose(jac[0, 0, 0], dx_du_fd, rtol=1e-6)
        bkd.assert_allclose(jac[0, 0, 1], dx_dv_fd, rtol=1e-6)
        bkd.assert_allclose(jac[0, 1, 0], dy_du_fd, rtol=1e-6)
        bkd.assert_allclose(jac[0, 1, 1], dy_dv_fd, rtol=1e-6)
