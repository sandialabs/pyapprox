"""Tests for affine transforms."""

import unittest
from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.collocation.mesh.transforms.affine import (
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
)


class TestAffineTransforms(Generic[Array], unittest.TestCase):
    """Base test class for affine transforms."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_transform_1d_identity(self):
        """Test 1D transform maps [-1, 1] to [-1, 1]."""
        bkd = self.bkd()
        transform = AffineTransform1D((-1.0, 1.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        bkd.assert_allclose(phys_pts, ref_pts, atol=1e-14)

    def test_transform_1d_scaling(self):
        """Test 1D transform maps [-1, 1] to [0, 2]."""
        bkd = self.bkd()
        transform = AffineTransform1D((0.0, 2.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        expected = bkd.asarray([[0.0, 1.0, 2.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

    def test_transform_1d_inverse(self):
        """Test 1D transform inverse."""
        bkd = self.bkd()
        transform = AffineTransform1D((0.0, 10.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 0.5, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        ref_pts_back = transform.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

    def test_transform_1d_jacobian(self):
        """Test 1D Jacobian is constant scale factor."""
        bkd = self.bkd()
        transform = AffineTransform1D((0.0, 4.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0]])
        jac_mat = transform.jacobian_matrix(ref_pts)
        jac_det = transform.jacobian_determinant(ref_pts)

        # Scale factor is (4-0)/2 = 2
        self.assertEqual(jac_mat.shape, (3, 1, 1))
        bkd.assert_allclose(jac_mat[:, 0, 0], bkd.full((3,), 2.0), atol=1e-14)
        bkd.assert_allclose(jac_det, bkd.full((3,), 2.0), atol=1e-14)

    def test_transform_2d_identity(self):
        """Test 2D transform maps [-1, 1]^2 to itself."""
        bkd = self.bkd()
        transform = AffineTransform2D((-1.0, 1.0, -1.0, 1.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        bkd.assert_allclose(phys_pts, ref_pts, atol=1e-14)

    def test_transform_2d_scaling(self):
        """Test 2D transform maps [-1, 1]^2 to [0, 2] x [0, 3]."""
        bkd = self.bkd()
        transform = AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd)

        # Test corners
        ref_pts = bkd.asarray([[-1.0, 1.0, -1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        expected = bkd.asarray([[0.0, 2.0, 0.0, 2.0], [0.0, 0.0, 3.0, 3.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

    def test_transform_2d_inverse(self):
        """Test 2D transform inverse."""
        bkd = self.bkd()
        transform = AffineTransform2D((1.0, 5.0, -2.0, 2.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.5, 1.0], [0.0, -0.5, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        ref_pts_back = transform.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

    def test_transform_2d_jacobian(self):
        """Test 2D Jacobian is diagonal with scale factors."""
        bkd = self.bkd()
        transform = AffineTransform2D((0.0, 4.0, 0.0, 6.0), bkd)

        ref_pts = bkd.asarray([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        jac_mat = transform.jacobian_matrix(ref_pts)
        jac_det = transform.jacobian_determinant(ref_pts)

        # Scale factors: (4-0)/2=2, (6-0)/2=3
        self.assertEqual(jac_mat.shape, (3, 2, 2))
        bkd.assert_allclose(jac_mat[:, 0, 0], bkd.full((3,), 2.0), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 1, 1], bkd.full((3,), 3.0), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 0, 1], bkd.zeros((3,)), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 1, 0], bkd.zeros((3,)), atol=1e-14)
        bkd.assert_allclose(jac_det, bkd.full((3,), 6.0), atol=1e-14)

    def test_transform_3d_identity(self):
        """Test 3D transform maps [-1, 1]^3 to itself."""
        bkd = self.bkd()
        transform = AffineTransform3D(
            (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), bkd
        )

        ref_pts = bkd.asarray([[0.0, 0.5], [0.0, -0.5], [0.0, 0.25]])
        phys_pts = transform.map_to_physical(ref_pts)
        bkd.assert_allclose(phys_pts, ref_pts, atol=1e-14)

    def test_transform_3d_scaling(self):
        """Test 3D transform scaling."""
        bkd = self.bkd()
        transform = AffineTransform3D(
            (0.0, 2.0, 0.0, 4.0, 0.0, 6.0), bkd
        )

        # Test origin
        ref_pts = bkd.asarray([[-1.0], [-1.0], [-1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        expected = bkd.asarray([[0.0], [0.0], [0.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

        # Test opposite corner
        ref_pts = bkd.asarray([[1.0], [1.0], [1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        expected = bkd.asarray([[2.0], [4.0], [6.0]])
        bkd.assert_allclose(phys_pts, expected, atol=1e-14)

    def test_transform_3d_inverse(self):
        """Test 3D transform inverse."""
        bkd = self.bkd()
        transform = AffineTransform3D(
            (1.0, 3.0, -1.0, 5.0, 0.0, 10.0), bkd
        )

        ref_pts = bkd.asarray([[-0.5, 0.5], [0.25, -0.25], [0.0, 1.0]])
        phys_pts = transform.map_to_physical(ref_pts)
        ref_pts_back = transform.map_to_reference(phys_pts)
        bkd.assert_allclose(ref_pts_back, ref_pts, atol=1e-14)

    def test_transform_3d_jacobian(self):
        """Test 3D Jacobian is diagonal with scale factors."""
        bkd = self.bkd()
        transform = AffineTransform3D(
            (0.0, 2.0, 0.0, 4.0, 0.0, 6.0), bkd
        )

        ref_pts = bkd.asarray([[0.0, 0.5], [0.0, 0.0], [0.0, -0.5]])
        jac_mat = transform.jacobian_matrix(ref_pts)
        jac_det = transform.jacobian_determinant(ref_pts)

        # Scale factors: 1, 2, 3
        self.assertEqual(jac_mat.shape, (2, 3, 3))
        bkd.assert_allclose(jac_mat[:, 0, 0], bkd.full((2,), 1.0), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 1, 1], bkd.full((2,), 2.0), atol=1e-14)
        bkd.assert_allclose(jac_mat[:, 2, 2], bkd.full((2,), 3.0), atol=1e-14)
        # Determinant = 1*2*3 = 6
        bkd.assert_allclose(jac_det, bkd.full((2,), 6.0), atol=1e-14)


class TestAffineTransformsNumpy(TestAffineTransforms):
    """NumPy backend tests for affine transforms."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
