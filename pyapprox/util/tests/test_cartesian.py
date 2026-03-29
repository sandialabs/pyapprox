"""Tests for cartesian product utilities.

Dual-backend tests for NumPy and PyTorch.
"""

from pyapprox.util.cartesian import (
    cartesian_product,
    cartesian_product_indices,
    cartesian_product_samples,
    outer_product_weights,
)


class TestCartesianProductIndices:
    """Base tests for cartesian_product_indices."""

    def test_1d_grid(self, bkd):
        """Test 1D grid (trivial case)."""
        indices = cartesian_product_indices([3], bkd)
        assert indices.shape == (1, 3)
        expected = bkd.asarray([[0, 1, 2]], dtype=bkd.int64_dtype())
        bkd.assert_allclose(indices, expected)

    def test_2d_grid(self, bkd):
        """Test 2D grid with different sizes."""
        indices = cartesian_product_indices([2, 3], bkd)
        assert indices.shape == (2, 6)
        # Last dimension varies fastest
        expected = bkd.asarray(
            [
                [0, 0, 0, 1, 1, 1],
                [0, 1, 2, 0, 1, 2],
            ],
            dtype=bkd.int64_dtype(),
        )
        bkd.assert_allclose(indices, expected)

    def test_3d_grid(self, bkd):
        """Test 3D grid."""
        indices = cartesian_product_indices([2, 2, 2], bkd)
        assert indices.shape == (3, 8)
        # Verify total product
        assert indices.shape[1] == 2 * 2 * 2

    def test_asymmetric_grid(self, bkd):
        """Test grid with different sizes in each dimension."""
        indices = cartesian_product_indices([3, 2, 4], bkd)
        assert indices.shape == (3, 24)

    def test_indices_cover_all_points(self, bkd):
        """Verify indices cover all possible combinations."""
        dims = [2, 3]
        indices = cartesian_product_indices(dims, bkd)
        # Convert to tuples and check uniqueness
        indices_np = bkd.to_numpy(indices)
        tuples = set(tuple(indices_np[:, i]) for i in range(indices_np.shape[1]))
        assert len(tuples) == 6  # 2 * 3 = 6 unique points


class TestCartesianProductSamples:
    """Base tests for cartesian_product_samples."""

    def test_1d_samples(self, bkd):
        """Test 1D samples (trivial case)."""
        x = bkd.asarray([0.0, 0.5, 1.0])
        samples = cartesian_product_samples([x], bkd)
        assert samples.shape == (1, 3)
        bkd.assert_allclose(samples[0, :], x)

    def test_2d_samples(self, bkd):
        """Test 2D tensor product samples."""
        x = bkd.asarray([0.0, 1.0])
        y = bkd.asarray([0.0, 0.5, 1.0])
        samples = cartesian_product_samples([x, y], bkd)
        assert samples.shape == (2, 6)
        # Last dimension varies fastest: (0,0), (0,0.5), (0,1), (1,0), (1,0.5), (1,1)
        expected = bkd.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            ]
        )
        bkd.assert_allclose(samples, expected)

    def test_2d_input_format(self, bkd):
        """Test that 2D input arrays (1, npts) work correctly."""
        x = bkd.asarray([[0.0, 1.0]])  # Shape (1, 2)
        y = bkd.asarray([[0.0, 0.5, 1.0]])  # Shape (1, 3)
        samples = cartesian_product_samples([x, y], bkd)
        assert samples.shape == (2, 6)

    def test_3d_samples(self, bkd):
        """Test 3D tensor product samples."""
        x = bkd.asarray([0.0, 1.0])
        y = bkd.asarray([-1.0, 1.0])
        z = bkd.asarray([0.0, 0.5])
        samples = cartesian_product_samples([x, y, z], bkd)
        assert samples.shape == (3, 8)


class TestOuterProductWeights:
    """Base tests for outer_product_weights."""

    def test_1d_weights(self, bkd):
        """Test 1D weights (trivial case)."""
        wx = bkd.asarray([0.5, 0.5])
        weights = outer_product_weights([wx], bkd)
        assert weights.shape == (2,)
        bkd.assert_allclose(weights, wx)

    def test_2d_weights(self, bkd):
        """Test 2D tensor product weights."""
        wx = bkd.asarray([0.5, 0.5])
        wy = bkd.asarray([1.0 / 3, 1.0 / 3, 1.0 / 3])
        weights = outer_product_weights([wx, wy], bkd)
        assert weights.shape == (6,)
        # Sum should equal sum(wx) * sum(wy)
        expected_sum = bkd.sum(wx) * bkd.sum(wy)
        actual_sum = bkd.sum(weights)
        bkd.assert_allclose(
            bkd.asarray([actual_sum]),
            bkd.asarray([expected_sum]),
            rtol=1e-10,
        )

    def test_weights_ordering(self, bkd):
        """Test that weights match samples ordering."""
        wx = bkd.asarray([1.0, 2.0])
        wy = bkd.asarray([3.0, 4.0, 5.0])
        weights = outer_product_weights([wx, wy], bkd)
        # Last dimension varies fastest, so weights should be:
        # (1*3, 1*4, 1*5, 2*3, 2*4, 2*5)
        expected = bkd.asarray([3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
        bkd.assert_allclose(weights, expected)

    def test_3d_weights(self, bkd):
        """Test 3D tensor product weights."""
        wx = bkd.asarray([1.0, 1.0])
        wy = bkd.asarray([1.0, 1.0])
        wz = bkd.asarray([0.5, 0.5])
        weights = outer_product_weights([wx, wy, wz], bkd)
        assert weights.shape == (8,)
        # All weights should be 0.5 (1*1*0.5)
        expected = bkd.asarray([0.5] * 8)
        bkd.assert_allclose(weights, expected)

    def test_gauss_legendre_integration(self, bkd):
        """Test that weights integrate correctly for polynomial."""
        # 2-point Gauss-Legendre on [-1, 1]
        # Points: +/- 1/sqrt(3), weights: [1, 1]
        import math

        1.0 / math.sqrt(3.0)
        w = bkd.asarray([1.0, 1.0])

        # 2D integral of f(x,y) = 1 over [-1,1]^2 = 4
        weights = outer_product_weights([w, w], bkd)
        integral = bkd.sum(weights)  # Integral of f=1
        bkd.assert_allclose(
            bkd.asarray([integral]),
            bkd.asarray([4.0]),
            rtol=1e-10,
        )


class TestFirstDimFastest:
    """Base tests for first_dim_fastest ordering option.

    These tests verify that the first_dim_fastest option produces Fortran-order
    (column-major) iteration where the first dimension varies fastest. This
    matches Kronecker product conventions used in spectral methods.
    """

    def test_cartesian_product_default_ordering(self, bkd):
        """Test default ordering: last dimension varies fastest (C-order)."""
        x = bkd.asarray([0, 1])
        y = bkd.asarray([0, 1, 2])
        result = cartesian_product(bkd, [x, y], first_dim_fastest=False)
        # Last dim (y) varies fastest: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
        expected = bkd.asarray(
            [
                [0, 0, 0, 1, 1, 1],
                [0, 1, 2, 0, 1, 2],
            ]
        )
        bkd.assert_allclose(result, expected)

    def test_cartesian_product_first_dim_fastest(self, bkd):
        """Test first_dim_fastest: first dimension varies fastest (Fortran-order)."""
        x = bkd.asarray([0, 1])
        y = bkd.asarray([0, 1, 2])
        result = cartesian_product(bkd, [x, y], first_dim_fastest=True)
        # First dim (x) varies fastest: (0,0), (1,0), (0,1), (1,1), (0,2), (1,2)
        expected = bkd.asarray(
            [
                [0, 1, 0, 1, 0, 1],
                [0, 0, 1, 1, 2, 2],
            ]
        )
        bkd.assert_allclose(result, expected)

    def test_cartesian_product_indices_first_dim_fastest(self, bkd):
        """Test cartesian_product_indices with first_dim_fastest."""
        indices = cartesian_product_indices([2, 3], bkd, first_dim_fastest=True)
        # First dim varies fastest
        expected = bkd.asarray(
            [
                [0, 1, 0, 1, 0, 1],
                [0, 0, 1, 1, 2, 2],
            ],
            dtype=bkd.int64_dtype(),
        )
        bkd.assert_allclose(indices, expected)

    def test_cartesian_product_samples_first_dim_fastest(self, bkd):
        """Test cartesian_product_samples with first_dim_fastest."""
        x = bkd.asarray([0.0, 1.0])
        y = bkd.asarray([0.0, 0.5, 1.0])
        samples = cartesian_product_samples([x, y], bkd, first_dim_fastest=True)
        # First dim (x) varies fastest
        expected = bkd.asarray(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            ]
        )
        bkd.assert_allclose(samples, expected)

    def test_3d_first_dim_fastest(self, bkd):
        """Test 3D cartesian product with first_dim_fastest.

        For 3 dimensions [x, y, z] with first_dim_fastest=True:
        - x (dim 0) varies fastest
        - y (dim 1) varies next
        - z (dim 2) varies slowest
        """
        x = bkd.asarray([0, 1])
        y = bkd.asarray([0, 1])
        z = bkd.asarray([0, 1])
        result = cartesian_product(bkd, [x, y, z], first_dim_fastest=True)

        # Expected ordering with first_dim_fastest:
        # (x,y,z): (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1),
        # (1,1,1)
        expected = bkd.asarray(
            [
                [0, 1, 0, 1, 0, 1, 0, 1],  # x varies fastest
                [0, 0, 1, 1, 0, 0, 1, 1],  # y varies next
                [0, 0, 0, 0, 1, 1, 1, 1],  # z varies slowest
            ]
        )
        bkd.assert_allclose(result, expected)

    def test_kronecker_product_consistency(self, bkd):
        """Test that first_dim_fastest matches Kronecker product ordering.

        In Kronecker products: A x B has A varying slowest and B varying fastest.
        So for tensor product of 1D bases, the first dimension should vary fastest
        to match the standard Kronecker ordering used in spectral methods.
        """
        # Simulate 1D basis nodes
        nodes_x = bkd.asarray([-1.0, 0.0, 1.0])  # 3 points
        nodes_y = bkd.asarray([-1.0, 1.0])  # 2 points

        # With first_dim_fastest, should match kron(I_y, I_x) ordering
        samples = cartesian_product_samples(
            [nodes_x, nodes_y], bkd, first_dim_fastest=True
        )

        # Expected: x varies fastest (cycles through -1, 0, 1 twice)
        # For each y value, we get all x values
        expected_x = bkd.asarray([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        expected_y = bkd.asarray([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

        bkd.assert_allclose(samples[0, :], expected_x)
        bkd.assert_allclose(samples[1, :], expected_y)

    def test_1d_case_unaffected(self, bkd):
        """Test that 1D case is unaffected by first_dim_fastest."""
        x = bkd.asarray([0.0, 0.5, 1.0])
        samples_default = cartesian_product_samples([x], bkd)
        samples_first = cartesian_product_samples(
            [x], bkd, first_dim_fastest=True
        )
        bkd.assert_allclose(samples_default, samples_first)

    def test_outer_product_first_dim_fastest(self, bkd):
        """Test outer_product with first_dim_fastest."""
        from pyapprox.util.cartesian import outer_product

        x = bkd.asarray([1.0, 2.0])
        y = bkd.asarray([3.0, 4.0, 5.0])

        # Default: shape (2, 3), last dim varies fastest when flattened
        result_default = outer_product(bkd, [x, y])
        assert result_default.shape == (2, 3)
        # Flattened: (1*3, 1*4, 1*5, 2*3, 2*4, 2*5) = (3, 4, 5, 6, 8, 10)
        expected_flat_default = bkd.asarray([3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
        bkd.assert_allclose(
            bkd.ravel(result_default), expected_flat_default
        )

        # first_dim_fastest: shape (3, 2), first dim varies fastest when flattened
        result_first = outer_product(bkd, [x, y], first_dim_fastest=True)
        assert result_first.shape == (3, 2)
        # Flattened: (1*3, 2*3, 1*4, 2*4, 1*5, 2*5) = (3, 6, 4, 8, 5, 10)
        expected_flat_first = bkd.asarray([3.0, 6.0, 4.0, 8.0, 5.0, 10.0])
        bkd.assert_allclose(bkd.ravel(result_first), expected_flat_first)

    def test_outer_product_weights_first_dim_fastest(self, bkd):
        """Test outer_product_weights with first_dim_fastest."""
        wx = bkd.asarray([1.0, 2.0])
        wy = bkd.asarray([3.0, 4.0, 5.0])

        # Default: last dim varies fastest
        weights_default = outer_product_weights([wx, wy], bkd)
        expected_default = bkd.asarray([3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
        bkd.assert_allclose(weights_default, expected_default)

        # first_dim_fastest: first dim varies fastest
        weights_first = outer_product_weights(
            [wx, wy], bkd, first_dim_fastest=True
        )
        expected_first = bkd.asarray([3.0, 6.0, 4.0, 8.0, 5.0, 10.0])
        bkd.assert_allclose(weights_first, expected_first)

    def test_samples_weights_consistency(self, bkd):
        """Test that samples and weights use consistent ordering.

        This is the key property: weight[i] should be the weight for sample[:, i].
        """
        x_pts = bkd.asarray([0.0, 1.0])
        y_pts = bkd.asarray([0.0, 0.5, 1.0])
        x_wts = bkd.asarray([0.5, 0.5])
        y_wts = bkd.asarray([1.0, 2.0, 3.0])

        # With first_dim_fastest=True, both should have x varying fastest
        samples = cartesian_product_samples(
            [x_pts, y_pts], bkd, first_dim_fastest=True
        )
        weights = outer_product_weights(
            [x_wts, y_wts], bkd, first_dim_fastest=True
        )

        # Check that the ordering is consistent
        # At point (x=0, y=0), weight should be 0.5 * 1.0 = 0.5
        # At point (x=1, y=0), weight should be 0.5 * 1.0 = 0.5
        # At point (x=0, y=0.5), weight should be 0.5 * 2.0 = 1.0
        # etc.

        # Find which column has x=0, y=0 (should be column 0 with first_dim_fastest)
        bkd.assert_allclose(samples[0, 0:1], bkd.asarray([0.0]))
        bkd.assert_allclose(samples[1, 0:1], bkd.asarray([0.0]))
        bkd.assert_allclose(weights[0:1], bkd.asarray([0.5]))

        # Find which column has x=1, y=0 (should be column 1 with first_dim_fastest)
        bkd.assert_allclose(samples[0, 1:2], bkd.asarray([1.0]))
        bkd.assert_allclose(samples[1, 1:2], bkd.asarray([0.0]))
        bkd.assert_allclose(weights[1:2], bkd.asarray([0.5]))

        # Find which column has x=0, y=0.5 (should be column 2 with first_dim_fastest)
        bkd.assert_allclose(samples[0, 2:3], bkd.asarray([0.0]))
        bkd.assert_allclose(samples[1, 2:3], bkd.asarray([0.5]))
        bkd.assert_allclose(weights[2:3], bkd.asarray([1.0]))
