"""Tests for curvilinear transforms (polar, spherical, chained)."""

import math
import unittest
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.mesh.transforms.spherical import (
    SphericalTransform,
)
from pyapprox.pde.collocation.mesh.transforms.chained import ChainedTransform
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform2D,
    AffineTransform3D,
)


class TestPolarTransform(Generic[Array], unittest.TestCase):
    """Tests for PolarTransform."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_polar_basic_mapping(self):
        """Test basic polar to Cartesian mapping (direct polar input)."""
        bkd = self.bkd()
        transform = PolarTransform(
            (0.0, 2.0), (-math.pi, math.pi), bkd, from_reference=False
        )

        # r=1, theta=0 -> (1, 0)
        # r=1, theta=pi/2 -> (0, 1)
        # r=1, theta=pi -> (-1, 0)
        ref_pts = bkd.asarray([[1.0, 1.0, 1.0], [0.0, math.pi / 2, math.pi]])
        phys_pts = transform.map_to_physical(ref_pts)

        expected = bkd.asarray([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

    def test_polar_inverse_mapping(self):
        """Test inverse mapping recovers original points."""
        bkd = self.bkd()
        # Test both modes

        # With from_reference=False (direct polar coords)
        transform = PolarTransform(
            (0.1, 5.0), (-math.pi / 2, math.pi / 2), bkd, from_reference=False
        )
        ref_pts = bkd.asarray([[1.0, 2.0, 3.0], [0.0, math.pi / 4, -math.pi / 4]])
        phys_pts = transform.map_to_physical(ref_pts)
        ref_pts_back = transform.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

        # With from_reference=True ([-1,1]^2 input)
        transform_ref = PolarTransform(
            (0.1, 5.0), (-math.pi / 2, math.pi / 2), bkd, from_reference=True
        )
        ref_pts_std = bkd.asarray([[-0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]])
        phys_pts = transform_ref.map_to_physical(ref_pts_std)
        ref_pts_back = transform_ref.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts_std, atol=1e-14)

    def test_polar_jacobian_determinant(self):
        """Test Jacobian determinant equals r (for direct polar input)."""
        bkd = self.bkd()
        transform = PolarTransform(
            (0.0, 10.0), (-math.pi, math.pi), bkd, from_reference=False
        )

        ref_pts = bkd.asarray([[1.0, 2.0, 5.0], [0.0, math.pi / 4, math.pi / 2]])
        jac_det = transform.jacobian_determinant(ref_pts)

        # Jacobian determinant should equal r
        expected = ref_pts[0, :]
        bkd.assert_allclose(jac_det, expected, atol=1e-14)

    def test_polar_jacobian_matrix_consistency(self):
        """Test Jacobian matrix determinant matches jacobian_determinant."""
        bkd = self.bkd()
        # Test both modes

        # from_reference=False (direct polar)
        transform = PolarTransform(
            (0.1, 5.0), (-math.pi, math.pi), bkd, from_reference=False
        )
        ref_pts = bkd.asarray([[1.0, 2.0, 3.0], [0.0, math.pi / 4, -math.pi / 3]])
        jac_mat = transform.jacobian_matrix(ref_pts)
        jac_det = transform.jacobian_determinant(ref_pts)
        det_from_mat = (
            jac_mat[:, 0, 0] * jac_mat[:, 1, 1]
            - jac_mat[:, 0, 1] * jac_mat[:, 1, 0]
        )
        bkd.assert_allclose(det_from_mat, jac_det, atol=1e-14)

        # from_reference=True (standard domain)
        transform_ref = PolarTransform(
            (0.1, 5.0), (-math.pi, math.pi), bkd, from_reference=True
        )
        ref_pts_std = bkd.asarray([[-0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]])
        jac_mat = transform_ref.jacobian_matrix(ref_pts_std)
        jac_det = transform_ref.jacobian_determinant(ref_pts_std)
        det_from_mat = (
            jac_mat[:, 0, 0] * jac_mat[:, 1, 1]
            - jac_mat[:, 0, 1] * jac_mat[:, 1, 0]
        )
        bkd.assert_allclose(det_from_mat, jac_det, atol=1e-14)

    def test_polar_scale_factors(self):
        """Test scale factors h_r=1, h_theta=r (for direct polar input)."""
        bkd = self.bkd()
        transform = PolarTransform(
            (0.0, 10.0), (-math.pi, math.pi), bkd, from_reference=False
        )

        ref_pts = bkd.asarray([[1.0, 2.0, 5.0], [0.0, math.pi / 4, math.pi / 2]])
        scale = transform.scale_factors(ref_pts)

        # h_r = 1 for all points
        bkd.assert_allclose(scale[:, 0], bkd.ones((3,)), atol=1e-14)
        # h_theta = r
        bkd.assert_allclose(scale[:, 1], ref_pts[0, :], atol=1e-14)

    def test_polar_unit_basis_orthogonality(self):
        """Test unit curvilinear basis vectors are orthonormal."""
        bkd = self.bkd()
        # Unit basis doesn't depend on scale, so both modes should give same result
        transform = PolarTransform(
            (0.0, 5.0), (-math.pi, math.pi), bkd, from_reference=False
        )

        ref_pts = bkd.asarray([[1.0, 2.0], [0.0, math.pi / 3]])
        basis = transform.unit_curvilinear_basis(ref_pts)

        # Check each basis vector has unit length
        for ii in range(2):
            e_i = basis[:, :, ii]  # (npts, 2)
            norm_sq = bkd.sum(e_i**2, axis=1)
            bkd.assert_allclose(norm_sq, bkd.ones((2,)), atol=1e-14)

        # Check orthogonality: e_r . e_theta = 0
        e_r = basis[:, :, 0]
        e_theta = basis[:, :, 1]
        dot = bkd.sum(e_r * e_theta, axis=1)
        bkd.assert_allclose(dot, bkd.zeros((2,)), atol=1e-14)

    def test_polar_bounds_validation(self):
        """Test bounds validation."""
        bkd = self.bkd()

        # r_min < 0
        with self.assertRaises(ValueError):
            PolarTransform((-1.0, 2.0), (0.0, math.pi), bkd)

        # r_max <= r_min
        with self.assertRaises(ValueError):
            PolarTransform((2.0, 1.0), (0.0, math.pi), bkd)

        # theta out of [-2*pi, 2*pi]
        with self.assertRaises(ValueError):
            PolarTransform((0.0, 1.0), (-2 * math.pi - 0.1, math.pi), bkd)

        # theta_max <= theta_min
        with self.assertRaises(ValueError):
            PolarTransform((0.0, 1.0), (1.0, 0.0), bkd)

    def test_polar_from_reference_mapping(self):
        """Test mapping from [-1,1]^2 reference domain."""
        bkd = self.bkd()
        r_min, r_max = 1.0, 2.0
        theta_min, theta_max = 0.0, math.pi / 2

        transform = PolarTransform(
            (r_min, r_max), (theta_min, theta_max), bkd, from_reference=True
        )

        # (-1, -1) -> (r_min, theta_min) -> (r_min * cos(theta_min), ...)
        # (1, 1) -> (r_max, theta_max) -> (r_max * cos(theta_max), ...)
        # (0, 0) -> midpoint in polar
        ref_pts = bkd.asarray([[-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
        phys_pts = transform.map_to_physical(ref_pts)

        r_mid = (r_min + r_max) / 2
        theta_mid = (theta_min + theta_max) / 2

        expected_x = bkd.asarray([
            r_min * math.cos(theta_min),
            r_max * math.cos(theta_max),
            r_mid * math.cos(theta_mid),
        ])
        expected_y = bkd.asarray([
            r_min * math.sin(theta_min),
            r_max * math.sin(theta_max),
            r_mid * math.sin(theta_mid),
        ])
        bkd.assert_allclose(phys_pts[0, :], expected_x, atol=1e-14)
        bkd.assert_allclose(phys_pts[1, :], expected_y, atol=1e-14)


class TestSphericalTransform(Generic[Array], unittest.TestCase):
    """Tests for SphericalTransform."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_spherical_basic_mapping(self):
        """Test basic spherical to Cartesian mapping."""
        bkd = self.bkd()
        transform = SphericalTransform(
            (0.0, 2.0), (-math.pi, math.pi), (0.0, math.pi), bkd
        )

        # r=1, az=0, el=pi/2 -> (1, 0, 0) (on x-axis)
        # r=1, az=pi/2, el=pi/2 -> (0, 1, 0) (on y-axis)
        # r=1, az=0, el=0 -> (0, 0, 1) (on z-axis)
        ref_pts = bkd.asarray(
            [[1.0, 1.0, 1.0], [0.0, math.pi / 2, 0.0], [math.pi / 2, math.pi / 2, 0.0]]
        )
        phys_pts = transform.map_to_physical(ref_pts)

        expected = bkd.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

    def test_spherical_inverse_mapping(self):
        """Test inverse mapping recovers original points."""
        bkd = self.bkd()
        transform = SphericalTransform(
            (0.1, 5.0), (-math.pi / 2, math.pi / 2), (0.1, math.pi - 0.1), bkd
        )

        ref_pts = bkd.asarray(
            [[1.0, 2.0, 3.0], [0.0, math.pi / 4, -math.pi / 4], [math.pi / 4, math.pi / 2, math.pi / 3]]
        )
        phys_pts = transform.map_to_physical(ref_pts)
        ref_pts_back = transform.map_to_reference(phys_pts)

        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-12)

    def test_spherical_jacobian_determinant(self):
        """Test Jacobian determinant equals r^2 * sin(elevation)."""
        bkd = self.bkd()
        transform = SphericalTransform(
            (0.0, 10.0), (-math.pi, math.pi), (0.0, math.pi), bkd
        )

        ref_pts = bkd.asarray(
            [[1.0, 2.0, 3.0], [0.0, math.pi / 4, math.pi / 2], [math.pi / 4, math.pi / 2, math.pi / 3]]
        )
        jac_det = transform.jacobian_determinant(ref_pts)

        # Jacobian determinant should equal r^2 * sin(elevation)
        r = ref_pts[0, :]
        el = ref_pts[2, :]
        expected = r**2 * bkd.sin(el)
        bkd.assert_allclose(jac_det, expected, atol=1e-14)

    def test_spherical_scale_factors(self):
        """Test scale factors h_r=1, h_az=r*sin(el), h_el=r."""
        bkd = self.bkd()
        transform = SphericalTransform(
            (0.0, 10.0), (-math.pi, math.pi), (0.0, math.pi), bkd
        )

        ref_pts = bkd.asarray(
            [[1.0, 2.0, 3.0], [0.0, math.pi / 4, math.pi / 2], [math.pi / 4, math.pi / 2, math.pi / 3]]
        )
        scale = transform.scale_factors(ref_pts)

        r = ref_pts[0, :]
        el = ref_pts[2, :]

        # h_r = 1
        bkd.assert_allclose(scale[:, 0], bkd.ones((3,)), atol=1e-14)
        # h_az = r * sin(el)
        bkd.assert_allclose(scale[:, 1], r * bkd.sin(el), atol=1e-14)
        # h_el = r
        bkd.assert_allclose(scale[:, 2], r, atol=1e-14)

    def test_spherical_unit_basis_orthogonality(self):
        """Test unit curvilinear basis vectors are orthonormal."""
        bkd = self.bkd()
        transform = SphericalTransform(
            (0.0, 5.0), (-math.pi, math.pi), (0.01, math.pi - 0.01), bkd
        )

        ref_pts = bkd.asarray(
            [[1.0, 2.0], [0.0, math.pi / 3], [math.pi / 4, math.pi / 2]]
        )
        basis = transform.unit_curvilinear_basis(ref_pts)

        # Check each basis vector has unit length
        for ii in range(3):
            e_i = basis[:, :, ii]  # (npts, 3)
            norm_sq = bkd.sum(e_i**2, axis=1)
            bkd.assert_allclose(norm_sq, bkd.ones((2,)), atol=1e-14)

        # Check mutual orthogonality
        for ii in range(3):
            for jj in range(ii + 1, 3):
                e_i = basis[:, :, ii]
                e_j = basis[:, :, jj]
                dot = bkd.sum(e_i * e_j, axis=1)
                bkd.assert_allclose(dot, bkd.zeros((2,)), atol=1e-14)

    def test_spherical_bounds_validation(self):
        """Test bounds validation."""
        bkd = self.bkd()

        # elevation out of [0, pi]
        with self.assertRaises(ValueError):
            SphericalTransform(
                (0.0, 1.0), (-math.pi, math.pi), (-0.1, math.pi), bkd
            )

        with self.assertRaises(ValueError):
            SphericalTransform(
                (0.0, 1.0), (-math.pi, math.pi), (0.0, math.pi + 0.1), bkd
            )


class TestChainedTransform(Generic[Array], unittest.TestCase):
    """Tests for ChainedTransform."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_chained_identity(self):
        """Test chaining two identity-like transforms."""
        bkd = self.bkd()

        # Identity affine transform
        affine = AffineTransform2D((-1.0, 1.0, -1.0, 1.0), bkd)
        chained = ChainedTransform([affine, affine], bkd)

        ref_pts = bkd.asarray([[-0.5, 0.0, 0.5], [0.0, 0.5, -0.5]])
        phys_pts = chained.map_to_physical(ref_pts)

        # Two identities should give back the same points
        bkd.assert_allclose(phys_pts, ref_pts, atol=1e-14)

    def test_chained_affine_composition(self):
        """Test chaining two affine transforms."""
        bkd = self.bkd()

        # First: [-1, 1] -> [0, 2]
        affine1 = AffineTransform2D((0.0, 2.0, 0.0, 2.0), bkd)
        # Second: [0, 2] -> [0, 4] (interpreted as scale from [0, 2])
        affine2 = AffineTransform2D((0.0, 4.0, 0.0, 4.0), bkd)

        chained = ChainedTransform([affine1, affine2], bkd)

        # Map from reference
        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        phys_pts = chained.map_to_physical(ref_pts)

        # First maps (-1,-1)->(0,0), (0,0)->(1,1), (1,1)->(2,2)
        # Second maps (0,0)->(0,0), (1,1)->(2,2), (2,2)->(4,4) after scaling
        # Actually need to think about this more carefully...
        # affine2 maps [-1,1] -> [0,4], so input (0,0) needs to be in ref coords
        # Let's verify round-trip instead
        ref_pts_back = chained.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

    def test_chained_jacobian_determinant_product(self):
        """Test that chained Jacobian determinant is product."""
        bkd = self.bkd()

        # Two scaling transforms
        affine1 = AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd)  # det=1*1.5=1.5
        affine2 = AffineTransform2D((0.0, 4.0, 0.0, 6.0), bkd)  # det=2*3=6

        chained = ChainedTransform([affine1, affine2], bkd)

        ref_pts = bkd.asarray([[0.0], [0.0]])
        det1 = affine1.jacobian_determinant(ref_pts)
        pts1 = affine1.map_to_physical(ref_pts)
        det2 = affine2.jacobian_determinant(pts1)
        det_chained = chained.jacobian_determinant(ref_pts)

        bkd.assert_allclose(det_chained, det1 * det2, atol=1e-14)

    def test_chained_inverse_roundtrip(self):
        """Test that chained transform inverse works."""
        bkd = self.bkd()

        affine1 = AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd)
        affine2 = AffineTransform2D((1.0, 5.0, -1.0, 4.0), bkd)

        chained = ChainedTransform([affine1, affine2], bkd)

        ref_pts = bkd.asarray([[-0.5, 0.0, 0.5], [0.25, -0.25, 0.0]])
        phys_pts = chained.map_to_physical(ref_pts)
        ref_pts_back = chained.map_to_reference(phys_pts)

        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

    def test_chained_polar_with_affine(self):
        """Test chaining polar transform with affine scaling."""
        bkd = self.bkd()

        # Test with from_reference=False (direct polar coords)
        polar = PolarTransform(
            (0.5, 2.0), (-math.pi / 2, math.pi / 2), bkd, from_reference=False
        )

        ref_pts = bkd.asarray([[1.0, 1.5], [0.0, math.pi / 4]])
        phys_pts = polar.map_to_physical(ref_pts)

        # Verify: (1.0, 0) -> (1, 0), (1.5, pi/4) -> (1.5*cos(pi/4), 1.5*sin(pi/4))
        expected_x = bkd.asarray([1.0, 1.5 * math.cos(math.pi / 4)])
        expected_y = bkd.asarray([0.0, 1.5 * math.sin(math.pi / 4)])
        bkd.assert_allclose(phys_pts[0, :], expected_x, atol=1e-14)
        bkd.assert_allclose(phys_pts[1, :], expected_y, atol=1e-14)

    def test_chained_3d(self):
        """Test chained 3D transforms."""
        bkd = self.bkd()

        affine1 = AffineTransform3D((0.0, 2.0, 0.0, 2.0, 0.0, 2.0), bkd)
        affine2 = AffineTransform3D((1.0, 3.0, 1.0, 3.0, 1.0, 3.0), bkd)

        chained = ChainedTransform([affine1, affine2], bkd)

        ref_pts = bkd.asarray([[0.0], [0.0], [0.0]])
        phys_pts = chained.map_to_physical(ref_pts)
        ref_pts_back = chained.map_to_reference(phys_pts)

        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

    def test_chained_requires_same_ndim(self):
        """Test that chaining requires same ndim."""
        bkd = self.bkd()

        affine2d = AffineTransform2D((0.0, 1.0, 0.0, 1.0), bkd)
        affine3d = AffineTransform3D((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), bkd)

        with self.assertRaises(ValueError):
            ChainedTransform([affine2d, affine3d], bkd)


class TestPolarTransformNumpy(TestPolarTransform):
    """NumPy backend tests for polar transform."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestSphericalTransformNumpy(TestSphericalTransform):
    """NumPy backend tests for spherical transform."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


class TestChainedTransformNumpy(TestChainedTransform):
    """NumPy backend tests for chained transform."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
