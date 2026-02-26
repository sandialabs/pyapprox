"""Tests for mesh base module."""


from pyapprox.pde.collocation.mesh.base import (
    MeshData,
    compute_boundary_indices_1d,
    compute_boundary_indices_2d,
    compute_boundary_indices_3d,
    compute_cartesian_product,
)


class TestMeshBase:
    """Base test class for mesh functionality."""

    def test_mesh_data_properties(self, bkd):
        """Test MeshData properties."""
        points = bkd.asarray([[0.0, 0.5, 1.0]])  # 1D, 3 points
        boundary_indices = compute_boundary_indices_1d(3, bkd)
        mesh = MeshData(
            points=points,
            npts_per_dim=(3,),
            boundary_indices=boundary_indices,
        )

        assert mesh.ndim == 1
        assert mesh.npts == 3
        assert mesh.nboundaries == 2

    def test_cartesian_product_1d(self, bkd):
        """Test Cartesian product for 1D."""
        x = bkd.asarray([0.0, 1.0, 2.0])
        result = compute_cartesian_product([x], bkd)

        assert result.shape == (1, 3)
        bkd.assert_allclose(result[0, :], x)

    def test_cartesian_product_2d(self, bkd):
        """Test Cartesian product for 2D."""
        x = bkd.asarray([0.0, 1.0])  # 2 points
        y = bkd.asarray([0.0, 1.0, 2.0])  # 3 points

        result = compute_cartesian_product([x, y], bkd)

        # Should have 2*3 = 6 points
        assert result.shape == (2, 6)

        # x varies fastest: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
        expected_x = bkd.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        expected_y = bkd.asarray([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])

        bkd.assert_allclose(result[0, :], expected_x)
        bkd.assert_allclose(result[1, :], expected_y)

    def test_cartesian_product_3d(self, bkd):
        """Test Cartesian product for 3D."""
        x = bkd.asarray([0.0, 1.0])  # 2 points
        y = bkd.asarray([0.0, 1.0])  # 2 points
        z = bkd.asarray([0.0, 1.0])  # 2 points

        result = compute_cartesian_product([x, y, z], bkd)

        # Should have 2*2*2 = 8 points
        assert result.shape == (3, 8)

    def test_boundary_indices_1d(self, bkd):
        """Test 1D boundary indices."""
        npts = 5
        boundaries = compute_boundary_indices_1d(npts, bkd)

        assert len(boundaries) == 2

        # Left boundary: index 0
        assert len(boundaries[0]) == 1
        assert int(boundaries[0][0]) == 0

        # Right boundary: index 4
        assert len(boundaries[1]) == 1
        assert int(boundaries[1][0]) == npts - 1

    def test_boundary_indices_2d(self, bkd):
        """Test 2D boundary indices."""
        npts_x, npts_y = 3, 4
        boundaries = compute_boundary_indices_2d(npts_x, npts_y, bkd)

        assert len(boundaries) == 4

        # Left boundary: x=0, all y -> indices 0, 3, 6, 9
        left = boundaries[0]
        assert len(left) == npts_y
        for j, idx in enumerate(left):
            assert int(idx) == j * npts_x

        # Right boundary: x=2, all y -> indices 2, 5, 8, 11
        right = boundaries[1]
        assert len(right) == npts_y
        for j, idx in enumerate(right):
            assert int(idx) == j * npts_x + (npts_x - 1)

        # Bottom boundary: y=0, all x -> indices 0, 1, 2
        bottom = boundaries[2]
        assert len(bottom) == npts_x
        for i, idx in enumerate(bottom):
            assert int(idx) == i

        # Top boundary: y=3, all x -> indices 9, 10, 11
        top = boundaries[3]
        assert len(top) == npts_x
        for i, idx in enumerate(top):
            assert int(idx) == (npts_y - 1) * npts_x + i

    def test_boundary_indices_3d(self, bkd):
        """Test 3D boundary indices."""
        npts_x, npts_y, npts_z = 2, 3, 2
        boundaries = compute_boundary_indices_3d(npts_x, npts_y, npts_z, bkd)

        assert len(boundaries) == 6

        # Left: x=0 -> npts_y * npts_z points
        assert len(boundaries[0]) == npts_y * npts_z

        # Right: x=npts_x-1 -> npts_y * npts_z points
        assert len(boundaries[1]) == npts_y * npts_z

        # Bottom: y=0 -> npts_x * npts_z points
        assert len(boundaries[2]) == npts_x * npts_z

        # Top: y=npts_y-1 -> npts_x * npts_z points
        assert len(boundaries[3]) == npts_x * npts_z

        # Front: z=0 -> npts_x * npts_y points
        assert len(boundaries[4]) == npts_x * npts_y

        # Back: z=npts_z-1 -> npts_x * npts_y points
        assert len(boundaries[5]) == npts_x * npts_y


class TestMeshBaseNumpy(TestMeshBase):
    """NumPy backend tests for mesh base."""
