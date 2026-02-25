"""Tests for Cartesian mesh classes."""

import unittest
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.pde.collocation.mesh import (
    CartesianMesh1D,
    CartesianMesh2D,
    CartesianMesh3D,
    create_uniform_mesh_1d,
    create_uniform_mesh_2d,
    create_uniform_mesh_3d,
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
)


class TestCartesianMesh(Generic[Array], unittest.TestCase):
    """Base test class for Cartesian mesh functionality."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_mesh_1d_basic(self):
        """Test basic 1D mesh properties."""
        bkd = self.bkd()
        npts = 5
        ref_pts = bkd.linspace(-1.0, 1.0, npts)
        mesh = CartesianMesh1D(ref_pts, bkd)

        self.assertEqual(mesh.ndim(), 1)
        self.assertEqual(mesh.npts(), npts)
        self.assertEqual(mesh.npts_per_dim(), (npts,))
        self.assertEqual(mesh.nboundaries(), 2)
        self.assertEqual(mesh.points().shape, (1, npts))

    def test_mesh_1d_with_transform(self):
        """Test 1D mesh with affine transform."""
        bkd = self.bkd()
        npts = 5
        ref_pts = bkd.linspace(-1.0, 1.0, npts)
        transform = AffineTransform1D((0.0, 2.0), bkd)
        mesh = CartesianMesh1D(ref_pts, bkd, transform)

        # Physical points should be in [0, 2]
        pts = mesh.points()
        bkd.assert_allclose(pts[0, 0], 0.0, atol=1e-14)
        bkd.assert_allclose(pts[0, -1], 2.0, atol=1e-14)

        # Jacobian determinant should be 1.0 (scale factor)
        jac_det = mesh.jacobian_determinant()
        bkd.assert_allclose(jac_det, bkd.full((npts,), 1.0), atol=1e-14)

    def test_mesh_1d_boundary_indices(self):
        """Test 1D mesh boundary indices."""
        bkd = self.bkd()
        npts = 5
        ref_pts = bkd.linspace(-1.0, 1.0, npts)
        mesh = CartesianMesh1D(ref_pts, bkd)

        left = mesh.boundary_indices(0)
        right = mesh.boundary_indices(1)

        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertEqual(int(left[0]), 0)
        self.assertEqual(int(right[0]), npts - 1)

    def test_create_uniform_mesh_1d(self):
        """Test uniform 1D mesh factory function."""
        bkd = self.bkd()
        mesh = create_uniform_mesh_1d(5, (0.0, 10.0), bkd)

        pts = mesh.points()
        bkd.assert_allclose(pts[0, 0], 0.0, atol=1e-14)
        bkd.assert_allclose(pts[0, -1], 10.0, atol=1e-14)

    def test_mesh_2d_basic(self):
        """Test basic 2D mesh properties."""
        bkd = self.bkd()
        npts_x, npts_y = 3, 4
        ref_pts = [
            bkd.linspace(-1.0, 1.0, npts_x),
            bkd.linspace(-1.0, 1.0, npts_y),
        ]
        mesh = CartesianMesh2D(ref_pts, bkd)

        self.assertEqual(mesh.ndim(), 2)
        self.assertEqual(mesh.npts(), npts_x * npts_y)
        self.assertEqual(mesh.npts_per_dim(), (npts_x, npts_y))
        self.assertEqual(mesh.nboundaries(), 4)
        self.assertEqual(mesh.points().shape, (2, npts_x * npts_y))

    def test_mesh_2d_with_transform(self):
        """Test 2D mesh with affine transform."""
        bkd = self.bkd()
        npts_x, npts_y = 3, 4
        ref_pts = [
            bkd.linspace(-1.0, 1.0, npts_x),
            bkd.linspace(-1.0, 1.0, npts_y),
        ]
        transform = AffineTransform2D((0.0, 2.0, 0.0, 3.0), bkd)
        mesh = CartesianMesh2D(ref_pts, bkd, transform)

        pts = mesh.points()
        # Check corners
        # First point should be (0, 0)
        bkd.assert_allclose(pts[0, 0], 0.0, atol=1e-14)
        bkd.assert_allclose(pts[1, 0], 0.0, atol=1e-14)

        # Jacobian determinant should be 1.0 * 1.5 = 1.5
        jac_det = mesh.jacobian_determinant()
        expected = 1.0 * 1.5  # (2-0)/2 * (3-0)/2
        bkd.assert_allclose(jac_det[0], expected, atol=1e-14)

    def test_mesh_2d_boundary_indices(self):
        """Test 2D mesh boundary indices."""
        bkd = self.bkd()
        npts_x, npts_y = 3, 4
        ref_pts = [
            bkd.linspace(-1.0, 1.0, npts_x),
            bkd.linspace(-1.0, 1.0, npts_y),
        ]
        mesh = CartesianMesh2D(ref_pts, bkd)

        left = mesh.boundary_indices(0)
        right = mesh.boundary_indices(1)
        bottom = mesh.boundary_indices(2)
        top = mesh.boundary_indices(3)

        self.assertEqual(len(left), npts_y)
        self.assertEqual(len(right), npts_y)
        self.assertEqual(len(bottom), npts_x)
        self.assertEqual(len(top), npts_x)

    def test_create_uniform_mesh_2d(self):
        """Test uniform 2D mesh factory function."""
        bkd = self.bkd()
        mesh = create_uniform_mesh_2d((3, 4), (0.0, 2.0, 0.0, 3.0), bkd)

        self.assertEqual(mesh.npts(), 12)
        pts = mesh.points()
        # First point should be (0, 0)
        bkd.assert_allclose(pts[0, 0], 0.0, atol=1e-14)
        bkd.assert_allclose(pts[1, 0], 0.0, atol=1e-14)

    def test_mesh_3d_basic(self):
        """Test basic 3D mesh properties."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 2, 3, 2
        ref_pts = [
            bkd.linspace(-1.0, 1.0, npts_x),
            bkd.linspace(-1.0, 1.0, npts_y),
            bkd.linspace(-1.0, 1.0, npts_z),
        ]
        mesh = CartesianMesh3D(ref_pts, bkd)

        self.assertEqual(mesh.ndim(), 3)
        self.assertEqual(mesh.npts(), npts_x * npts_y * npts_z)
        self.assertEqual(mesh.npts_per_dim(), (npts_x, npts_y, npts_z))
        self.assertEqual(mesh.nboundaries(), 6)
        self.assertEqual(
            mesh.points().shape, (3, npts_x * npts_y * npts_z)
        )

    def test_mesh_3d_with_transform(self):
        """Test 3D mesh with affine transform."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 2, 3, 2
        ref_pts = [
            bkd.linspace(-1.0, 1.0, npts_x),
            bkd.linspace(-1.0, 1.0, npts_y),
            bkd.linspace(-1.0, 1.0, npts_z),
        ]
        transform = AffineTransform3D(
            (0.0, 2.0, 0.0, 3.0, 0.0, 4.0), bkd
        )
        mesh = CartesianMesh3D(ref_pts, bkd, transform)

        pts = mesh.points()
        # First point should be (0, 0, 0)
        bkd.assert_allclose(pts[0, 0], 0.0, atol=1e-14)
        bkd.assert_allclose(pts[1, 0], 0.0, atol=1e-14)
        bkd.assert_allclose(pts[2, 0], 0.0, atol=1e-14)

        # Jacobian determinant should be 1.0 * 1.5 * 2.0 = 3.0
        jac_det = mesh.jacobian_determinant()
        expected = 1.0 * 1.5 * 2.0
        bkd.assert_allclose(jac_det[0], expected, atol=1e-14)

    def test_mesh_3d_boundary_indices(self):
        """Test 3D mesh boundary indices."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 2, 3, 2
        ref_pts = [
            bkd.linspace(-1.0, 1.0, npts_x),
            bkd.linspace(-1.0, 1.0, npts_y),
            bkd.linspace(-1.0, 1.0, npts_z),
        ]
        mesh = CartesianMesh3D(ref_pts, bkd)

        left = mesh.boundary_indices(0)
        right = mesh.boundary_indices(1)
        bottom = mesh.boundary_indices(2)
        top = mesh.boundary_indices(3)
        front = mesh.boundary_indices(4)
        back = mesh.boundary_indices(5)

        self.assertEqual(len(left), npts_y * npts_z)
        self.assertEqual(len(right), npts_y * npts_z)
        self.assertEqual(len(bottom), npts_x * npts_z)
        self.assertEqual(len(top), npts_x * npts_z)
        self.assertEqual(len(front), npts_x * npts_y)
        self.assertEqual(len(back), npts_x * npts_y)

    def test_create_uniform_mesh_3d(self):
        """Test uniform 3D mesh factory function."""
        bkd = self.bkd()
        mesh = create_uniform_mesh_3d(
            (2, 3, 2), (0.0, 2.0, 0.0, 3.0, 0.0, 4.0), bkd
        )

        self.assertEqual(mesh.npts(), 12)
        pts = mesh.points()
        # First point should be (0, 0, 0)
        bkd.assert_allclose(pts[0, 0], 0.0, atol=1e-14)
        bkd.assert_allclose(pts[1, 0], 0.0, atol=1e-14)
        bkd.assert_allclose(pts[2, 0], 0.0, atol=1e-14)

    def test_invalid_boundary_id(self):
        """Test that invalid boundary IDs raise errors."""
        bkd = self.bkd()

        mesh1d = CartesianMesh1D(bkd.linspace(-1.0, 1.0, 3), bkd)
        with self.assertRaises(ValueError):
            mesh1d.boundary_indices(2)

        mesh2d = CartesianMesh2D(
            [bkd.linspace(-1.0, 1.0, 3), bkd.linspace(-1.0, 1.0, 3)], bkd
        )
        with self.assertRaises(ValueError):
            mesh2d.boundary_indices(4)

        mesh3d = CartesianMesh3D(
            [
                bkd.linspace(-1.0, 1.0, 2),
                bkd.linspace(-1.0, 1.0, 2),
                bkd.linspace(-1.0, 1.0, 2),
            ],
            bkd,
        )
        with self.assertRaises(ValueError):
            mesh3d.boundary_indices(6)


class TestCartesianMeshNumpy(TestCartesianMesh):
    """NumPy backend tests for Cartesian mesh."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
