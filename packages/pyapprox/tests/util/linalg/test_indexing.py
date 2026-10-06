"""Tests for indexing utilities."""

import numpy as np

from pyapprox.util.linalg.indexing import extract_submatrix


class TestExtractSubmatrix:
    """Test extract_submatrix function."""

    def test_basic_extraction(self, bkd) -> None:
        """Test basic submatrix extraction."""
        mat = bkd.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )
        row_indices = bkd.asarray([0, 2], dtype=int)
        col_indices = bkd.asarray([1, 3], dtype=int)

        result = extract_submatrix(mat, row_indices, col_indices)

        expected = bkd.asarray(
            [
                [2.0, 4.0],
                [10.0, 12.0],
            ]
        )
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_single_row_single_col(self, bkd) -> None:
        """Test extraction of single element as 2D array."""
        mat = bkd.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        row_indices = bkd.asarray([1], dtype=int)
        col_indices = bkd.asarray([0], dtype=int)

        result = extract_submatrix(mat, row_indices, col_indices)

        expected = bkd.asarray([[3.0]])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_full_matrix(self, bkd) -> None:
        """Test extraction of full matrix."""
        mat = bkd.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        row_indices = bkd.asarray([0, 1], dtype=int)
        col_indices = bkd.asarray([0, 1, 2], dtype=int)

        result = extract_submatrix(mat, row_indices, col_indices)

        bkd.assert_allclose(result, mat, rtol=1e-12)

    def test_reordered_indices(self, bkd) -> None:
        """Test that indices can reorder the result."""
        mat = bkd.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        # Reverse order of rows and columns
        row_indices = bkd.asarray([2, 1, 0], dtype=int)
        col_indices = bkd.asarray([2, 0], dtype=int)

        result = extract_submatrix(mat, row_indices, col_indices)

        expected = bkd.asarray(
            [
                [9.0, 7.0],
                [6.0, 4.0],
                [3.0, 1.0],
            ]
        )
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_matches_numpy_ix(self, bkd) -> None:
        """Test that result matches np.ix_ behavior."""
        np.random.seed(42)
        mat_np = np.random.randn(5, 6)
        row_idx_np = np.array([1, 3, 4])
        col_idx_np = np.array([0, 2, 5])

        # NumPy ix_ result
        expected_np = mat_np[np.ix_(row_idx_np, col_idx_np)]

        # Our function
        mat = bkd.asarray(mat_np)
        row_indices = bkd.asarray(row_idx_np, dtype=int)
        col_indices = bkd.asarray(col_idx_np, dtype=int)
        result = extract_submatrix(mat, row_indices, col_indices)

        bkd.assert_allclose(result, bkd.asarray(expected_np), rtol=1e-12)
