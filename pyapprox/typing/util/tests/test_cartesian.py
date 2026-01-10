"""Tests for cartesian product utilities.

Dual-backend tests for NumPy and PyTorch.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.cartesian import (
    cartesian_product_indices,
    cartesian_product_samples,
    outer_product_weights,
)
from pyapprox.typing.util.test_utils import load_tests


class TestCartesianProductIndices(Generic[Array], unittest.TestCase):
    """Base tests for cartesian_product_indices."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_1d_grid(self):
        """Test 1D grid (trivial case)."""
        indices = cartesian_product_indices([3], self._bkd)
        self.assertEqual(indices.shape, (1, 3))
        expected = self._bkd.asarray([[0, 1, 2]])
        self._bkd.assert_allclose(indices, expected)

    def test_2d_grid(self):
        """Test 2D grid with different sizes."""
        indices = cartesian_product_indices([2, 3], self._bkd)
        self.assertEqual(indices.shape, (2, 6))
        # Last dimension varies fastest
        expected = self._bkd.asarray([
            [0, 0, 0, 1, 1, 1],
            [0, 1, 2, 0, 1, 2],
        ])
        self._bkd.assert_allclose(indices, expected)

    def test_3d_grid(self):
        """Test 3D grid."""
        indices = cartesian_product_indices([2, 2, 2], self._bkd)
        self.assertEqual(indices.shape, (3, 8))
        # Verify total product
        self.assertEqual(indices.shape[1], 2 * 2 * 2)

    def test_asymmetric_grid(self):
        """Test grid with different sizes in each dimension."""
        indices = cartesian_product_indices([3, 2, 4], self._bkd)
        self.assertEqual(indices.shape, (3, 24))

    def test_indices_cover_all_points(self):
        """Verify indices cover all possible combinations."""
        dims = [2, 3]
        indices = cartesian_product_indices(dims, self._bkd)
        # Convert to tuples and check uniqueness
        indices_np = self._bkd.to_numpy(indices)
        tuples = set(tuple(indices_np[:, i]) for i in range(indices_np.shape[1]))
        self.assertEqual(len(tuples), 6)  # 2 * 3 = 6 unique points


class TestCartesianProductSamples(Generic[Array], unittest.TestCase):
    """Base tests for cartesian_product_samples."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_1d_samples(self):
        """Test 1D samples (trivial case)."""
        x = self._bkd.asarray([0.0, 0.5, 1.0])
        samples = cartesian_product_samples([x], self._bkd)
        self.assertEqual(samples.shape, (1, 3))
        self._bkd.assert_allclose(samples[0, :], x)

    def test_2d_samples(self):
        """Test 2D tensor product samples."""
        x = self._bkd.asarray([0.0, 1.0])
        y = self._bkd.asarray([0.0, 0.5, 1.0])
        samples = cartesian_product_samples([x, y], self._bkd)
        self.assertEqual(samples.shape, (2, 6))
        # Last dimension varies fastest: (0,0), (0,0.5), (0,1), (1,0), (1,0.5), (1,1)
        expected = self._bkd.asarray([
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        ])
        self._bkd.assert_allclose(samples, expected)

    def test_2d_input_format(self):
        """Test that 2D input arrays (1, npts) work correctly."""
        x = self._bkd.asarray([[0.0, 1.0]])  # Shape (1, 2)
        y = self._bkd.asarray([[0.0, 0.5, 1.0]])  # Shape (1, 3)
        samples = cartesian_product_samples([x, y], self._bkd)
        self.assertEqual(samples.shape, (2, 6))

    def test_3d_samples(self):
        """Test 3D tensor product samples."""
        x = self._bkd.asarray([0.0, 1.0])
        y = self._bkd.asarray([-1.0, 1.0])
        z = self._bkd.asarray([0.0, 0.5])
        samples = cartesian_product_samples([x, y, z], self._bkd)
        self.assertEqual(samples.shape, (3, 8))


class TestOuterProductWeights(Generic[Array], unittest.TestCase):
    """Base tests for outer_product_weights."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_1d_weights(self):
        """Test 1D weights (trivial case)."""
        wx = self._bkd.asarray([0.5, 0.5])
        weights = outer_product_weights([wx], self._bkd)
        self.assertEqual(weights.shape, (2,))
        self._bkd.assert_allclose(weights, wx)

    def test_2d_weights(self):
        """Test 2D tensor product weights."""
        wx = self._bkd.asarray([0.5, 0.5])
        wy = self._bkd.asarray([1.0 / 3, 1.0 / 3, 1.0 / 3])
        weights = outer_product_weights([wx, wy], self._bkd)
        self.assertEqual(weights.shape, (6,))
        # Sum should equal sum(wx) * sum(wy)
        expected_sum = self._bkd.sum(wx) * self._bkd.sum(wy)
        actual_sum = self._bkd.sum(weights)
        self._bkd.assert_allclose(
            self._bkd.asarray([actual_sum]),
            self._bkd.asarray([expected_sum]),
            rtol=1e-10,
        )

    def test_weights_ordering(self):
        """Test that weights match samples ordering."""
        wx = self._bkd.asarray([1.0, 2.0])
        wy = self._bkd.asarray([3.0, 4.0, 5.0])
        weights = outer_product_weights([wx, wy], self._bkd)
        # Last dimension varies fastest, so weights should be:
        # (1*3, 1*4, 1*5, 2*3, 2*4, 2*5)
        expected = self._bkd.asarray([3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
        self._bkd.assert_allclose(weights, expected)

    def test_3d_weights(self):
        """Test 3D tensor product weights."""
        wx = self._bkd.asarray([1.0, 1.0])
        wy = self._bkd.asarray([1.0, 1.0])
        wz = self._bkd.asarray([0.5, 0.5])
        weights = outer_product_weights([wx, wy, wz], self._bkd)
        self.assertEqual(weights.shape, (8,))
        # All weights should be 0.5 (1*1*0.5)
        expected = self._bkd.asarray([0.5] * 8)
        self._bkd.assert_allclose(weights, expected)

    def test_gauss_legendre_integration(self):
        """Test that weights integrate correctly for polynomial."""
        # 2-point Gauss-Legendre on [-1, 1]
        # Points: +/- 1/sqrt(3), weights: [1, 1]
        import math
        pts = 1.0 / math.sqrt(3.0)
        w = self._bkd.asarray([1.0, 1.0])

        # 2D integral of f(x,y) = 1 over [-1,1]^2 = 4
        weights = outer_product_weights([w, w], self._bkd)
        integral = self._bkd.sum(weights)  # Integral of f=1
        self._bkd.assert_allclose(
            self._bkd.asarray([integral]),
            self._bkd.asarray([4.0]),
            rtol=1e-10,
        )


# Concrete test classes for each backend


class TestCartesianProductIndicesNumpy(TestCartesianProductIndices[NDArray[Any]]):
    """NumPy backend tests for cartesian_product_indices."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCartesianProductIndicesTorch(TestCartesianProductIndices[torch.Tensor]):
    """PyTorch backend tests for cartesian_product_indices."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestCartesianProductSamplesNumpy(TestCartesianProductSamples[NDArray[Any]]):
    """NumPy backend tests for cartesian_product_samples."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCartesianProductSamplesTorch(TestCartesianProductSamples[torch.Tensor]):
    """PyTorch backend tests for cartesian_product_samples."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


class TestOuterProductWeightsNumpy(TestOuterProductWeights[NDArray[Any]]):
    """NumPy backend tests for outer_product_weights."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOuterProductWeightsTorch(TestOuterProductWeights[torch.Tensor]):
    """PyTorch backend tests for outer_product_weights."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = self.bkd()


if __name__ == "__main__":
    unittest.main()
