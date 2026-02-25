"""Tests for indexing utilities."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.util.linalg.indexing import extract_submatrix


class TestExtractSubmatrix(Generic[Array], unittest.TestCase):
    """Test extract_submatrix function."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_basic_extraction(self) -> None:
        """Test basic submatrix extraction."""
        mat = self._bkd.asarray([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ])
        row_indices = self._bkd.asarray([0, 2], dtype=int)
        col_indices = self._bkd.asarray([1, 3], dtype=int)

        result = extract_submatrix(mat, row_indices, col_indices)

        expected = self._bkd.asarray([
            [2.0, 4.0],
            [10.0, 12.0],
        ])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_single_row_single_col(self) -> None:
        """Test extraction of single element as 2D array."""
        mat = self._bkd.asarray([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        row_indices = self._bkd.asarray([1], dtype=int)
        col_indices = self._bkd.asarray([0], dtype=int)

        result = extract_submatrix(mat, row_indices, col_indices)

        expected = self._bkd.asarray([[3.0]])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_full_matrix(self) -> None:
        """Test extraction of full matrix."""
        mat = self._bkd.asarray([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        row_indices = self._bkd.asarray([0, 1], dtype=int)
        col_indices = self._bkd.asarray([0, 1, 2], dtype=int)

        result = extract_submatrix(mat, row_indices, col_indices)

        self._bkd.assert_allclose(result, mat, rtol=1e-12)

    def test_reordered_indices(self) -> None:
        """Test that indices can reorder the result."""
        mat = self._bkd.asarray([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        # Reverse order of rows and columns
        row_indices = self._bkd.asarray([2, 1, 0], dtype=int)
        col_indices = self._bkd.asarray([2, 0], dtype=int)

        result = extract_submatrix(mat, row_indices, col_indices)

        expected = self._bkd.asarray([
            [9.0, 7.0],
            [6.0, 4.0],
            [3.0, 1.0],
        ])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_matches_numpy_ix(self) -> None:
        """Test that result matches np.ix_ behavior."""
        np.random.seed(42)
        mat_np = np.random.randn(5, 6)
        row_idx_np = np.array([1, 3, 4])
        col_idx_np = np.array([0, 2, 5])

        # NumPy ix_ result
        expected_np = mat_np[np.ix_(row_idx_np, col_idx_np)]

        # Our function
        mat = self._bkd.asarray(mat_np)
        row_indices = self._bkd.asarray(row_idx_np, dtype=int)
        col_indices = self._bkd.asarray(col_idx_np, dtype=int)
        result = extract_submatrix(mat, row_indices, col_indices)

        self._bkd.assert_allclose(result, self._bkd.asarray(expected_np), rtol=1e-12)


class TestExtractSubmatrixNumpy(TestExtractSubmatrix[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestExtractSubmatrixTorch(TestExtractSubmatrix[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main(verbosity=2)
