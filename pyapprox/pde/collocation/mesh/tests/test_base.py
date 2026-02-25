"""Tests for mesh base module."""

import unittest
from typing import Generic

from pyapprox.pde.collocation.mesh.base import (
    MeshData,
    compute_boundary_indices_1d,
    compute_boundary_indices_2d,
    compute_boundary_indices_3d,
    compute_cartesian_product,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


class TestMeshBase(Generic[Array], unittest.TestCase):
    """Base test class for mesh functionality."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def test_mesh_data_properties(self):
        """Test MeshData properties."""
        bkd = self.bkd()
        points = bkd.asarray([[0.0, 0.5, 1.0]])  # 1D, 3 points
        boundary_indices = compute_boundary_indices_1d(3, bkd)
        mesh = MeshData(
            points=points,
            npts_per_dim=(3,),
            boundary_indices=boundary_indices,
        )

        self.assertEqual(mesh.ndim, 1)
        self.assertEqual(mesh.npts, 3)
        self.assertEqual(mesh.nboundaries, 2)

    def test_cartesian_product_1d(self):
        """Test Cartesian product for 1D."""
        bkd = self.bkd()
        x = bkd.asarray([0.0, 1.0, 2.0])
        result = compute_cartesian_product([x], bkd)

        self.assertEqual(result.shape, (1, 3))
        bkd.assert_allclose(result[0, :], x)

    def test_cartesian_product_2d(self):
        """Test Cartesian product for 2D."""
        bkd = self.bkd()
        x = bkd.asarray([0.0, 1.0])  # 2 points
        y = bkd.asarray([0.0, 1.0, 2.0])  # 3 points

        result = compute_cartesian_product([x, y], bkd)

        # Should have 2*3 = 6 points
        self.assertEqual(result.shape, (2, 6))

        # x varies fastest: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
        expected_x = bkd.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        expected_y = bkd.asarray([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])

        bkd.assert_allclose(result[0, :], expected_x)
        bkd.assert_allclose(result[1, :], expected_y)

    def test_cartesian_product_3d(self):
        """Test Cartesian product for 3D."""
        bkd = self.bkd()
        x = bkd.asarray([0.0, 1.0])  # 2 points
        y = bkd.asarray([0.0, 1.0])  # 2 points
        z = bkd.asarray([0.0, 1.0])  # 2 points

        result = compute_cartesian_product([x, y, z], bkd)

        # Should have 2*2*2 = 8 points
        self.assertEqual(result.shape, (3, 8))

    def test_boundary_indices_1d(self):
        """Test 1D boundary indices."""
        bkd = self.bkd()
        npts = 5
        boundaries = compute_boundary_indices_1d(npts, bkd)

        self.assertEqual(len(boundaries), 2)

        # Left boundary: index 0
        self.assertEqual(len(boundaries[0]), 1)
        self.assertEqual(int(boundaries[0][0]), 0)

        # Right boundary: index 4
        self.assertEqual(len(boundaries[1]), 1)
        self.assertEqual(int(boundaries[1][0]), npts - 1)

    def test_boundary_indices_2d(self):
        """Test 2D boundary indices."""
        bkd = self.bkd()
        npts_x, npts_y = 3, 4
        boundaries = compute_boundary_indices_2d(npts_x, npts_y, bkd)

        self.assertEqual(len(boundaries), 4)

        # Left boundary: x=0, all y -> indices 0, 3, 6, 9
        left = boundaries[0]
        self.assertEqual(len(left), npts_y)
        for j, idx in enumerate(left):
            self.assertEqual(int(idx), j * npts_x)

        # Right boundary: x=2, all y -> indices 2, 5, 8, 11
        right = boundaries[1]
        self.assertEqual(len(right), npts_y)
        for j, idx in enumerate(right):
            self.assertEqual(int(idx), j * npts_x + (npts_x - 1))

        # Bottom boundary: y=0, all x -> indices 0, 1, 2
        bottom = boundaries[2]
        self.assertEqual(len(bottom), npts_x)
        for i, idx in enumerate(bottom):
            self.assertEqual(int(idx), i)

        # Top boundary: y=3, all x -> indices 9, 10, 11
        top = boundaries[3]
        self.assertEqual(len(top), npts_x)
        for i, idx in enumerate(top):
            self.assertEqual(int(idx), (npts_y - 1) * npts_x + i)

    def test_boundary_indices_3d(self):
        """Test 3D boundary indices."""
        bkd = self.bkd()
        npts_x, npts_y, npts_z = 2, 3, 2
        boundaries = compute_boundary_indices_3d(npts_x, npts_y, npts_z, bkd)

        self.assertEqual(len(boundaries), 6)

        # Left: x=0 -> npts_y * npts_z points
        self.assertEqual(len(boundaries[0]), npts_y * npts_z)

        # Right: x=npts_x-1 -> npts_y * npts_z points
        self.assertEqual(len(boundaries[1]), npts_y * npts_z)

        # Bottom: y=0 -> npts_x * npts_z points
        self.assertEqual(len(boundaries[2]), npts_x * npts_z)

        # Top: y=npts_y-1 -> npts_x * npts_z points
        self.assertEqual(len(boundaries[3]), npts_x * npts_z)

        # Front: z=0 -> npts_x * npts_y points
        self.assertEqual(len(boundaries[4]), npts_x * npts_y)

        # Back: z=npts_z-1 -> npts_x * npts_y points
        self.assertEqual(len(boundaries[5]), npts_x * npts_y)


class TestMeshBaseNumpy(TestMeshBase):
    """NumPy backend tests for mesh base."""

    __test__ = True

    def bkd(self) -> Backend[Array]:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
