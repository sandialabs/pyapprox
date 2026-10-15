"""Tests for apply_dirichlet_rows.

The oracle is the documented semantics computed densely: every
Dirichlet row is zeroed and its diagonal set to 1.0, every other row is
unchanged, and the sparse result is CSR with no explicit zeros.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, isspmatrix_csr

from pyapprox.pde.sparse_utils import apply_dirichlet_rows
from pyapprox.util.backends.numpy import NumpyBkd


def _dense_oracle(
    dense: NDArray[np.floating[Any]], dofs: NDArray[np.integer[Any]]
) -> NDArray[np.floating[Any]]:
    expected = dense.copy()
    expected[dofs, :] = 0.0
    expected[dofs, dofs] = 1.0
    return expected


def _random_sparse_dense(
    nrows: int, density: float, seed: int
) -> NDArray[np.floating[Any]]:
    rng = np.random.default_rng(seed)
    dense = rng.normal(0.0, 1.0, (nrows, nrows))
    dense[rng.random((nrows, nrows)) > density] = 0.0
    np.fill_diagonal(dense, rng.normal(0.0, 1.0, nrows) + 5.0)
    return dense


def _check(
    numpy_bkd: NumpyBkd,
    result: Any,
    expected: NDArray[np.floating[Any]],
) -> None:
    assert isspmatrix_csr(result)
    assert result.shape == expected.shape
    assert not np.any(result.data == 0.0)
    numpy_bkd.assert_allclose(result.toarray(), expected, rtol=0, atol=0)


class TestApplyDirichletRows:
    @pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
    def test_matches_dense_oracle(self, numpy_bkd: NumpyBkd, fmt: str) -> None:
        dense = _random_sparse_dense(12, 0.3, seed=0)
        dofs = np.array([0, 5, 11])
        matrix = {"csr": csr_matrix, "csc": csc_matrix, "coo": coo_matrix}[fmt](dense)
        result = apply_dirichlet_rows(matrix, dofs)
        _check(numpy_bkd, result, _dense_oracle(dense, dofs))

    def test_missing_diagonal_is_inserted(self, numpy_bkd: NumpyBkd) -> None:
        """Dirichlet DOFs whose diagonal is absent from the sparsity
        pattern (e.g. some pressure DOFs) still get a unit diagonal."""
        dense = _random_sparse_dense(8, 0.4, seed=1)
        dense[2, 2] = 0.0
        dense[6, 6] = 0.0
        dense[2, 3] = 1.5
        matrix = csr_matrix(dense)
        assert matrix[2, 2] == 0.0
        dofs = np.array([2, 4, 6])
        result = apply_dirichlet_rows(matrix, dofs)
        _check(numpy_bkd, result, _dense_oracle(dense, dofs))

    def test_unsorted_column_indices(self, numpy_bkd: NumpyBkd) -> None:
        """Column indices need not be sorted within a row."""
        # Row 0: cols (2, 0); row 1: cols (1, 2, 0); row 2: cols (2, 1).
        indptr = np.array([0, 2, 5, 7])
        indices = np.array([2, 0, 1, 2, 0, 2, 1])
        data = np.array([3.0, 1.0, 4.0, 5.0, 2.0, 7.0, 6.0])
        matrix = csr_matrix((data, indices, indptr), shape=(3, 3))
        assert not matrix.has_sorted_indices
        dense = matrix.toarray()
        dofs = np.array([1, 2])
        result = apply_dirichlet_rows(matrix, dofs)
        _check(numpy_bkd, result, _dense_oracle(dense, dofs))

    def test_duplicate_dofs(self, numpy_bkd: NumpyBkd) -> None:
        """Repeated DOFs set the diagonal to 1.0, not a multiple."""
        dense = _random_sparse_dense(6, 0.5, seed=2)
        dense[3, 3] = 0.0
        dofs = np.array([1, 3, 1, 3, 3])
        result = apply_dirichlet_rows(csr_matrix(dense), dofs)
        _check(numpy_bkd, result, _dense_oracle(dense, np.unique(dofs)))

    def test_empty_dofs_only_drops_explicit_zeros(self, numpy_bkd: NumpyBkd) -> None:
        dense = _random_sparse_dense(5, 0.5, seed=3)
        matrix = csr_matrix(dense)
        matrix.data[0] = 0.0  # an explicit stored zero
        result = apply_dirichlet_rows(matrix, np.array([], dtype=np.int64))
        _check(numpy_bkd, result, matrix.toarray())

    def test_input_not_modified(self, numpy_bkd: NumpyBkd) -> None:
        dense = _random_sparse_dense(7, 0.4, seed=4)
        matrix = csr_matrix(dense)
        apply_dirichlet_rows(matrix, np.array([0, 3]))
        numpy_bkd.assert_allclose(matrix.toarray(), dense, rtol=0, atol=0)

    def test_dense_branch(self, numpy_bkd: NumpyBkd) -> None:
        dense = _random_sparse_dense(7, 0.6, seed=5)
        dofs = np.array([1, 6, 1])
        result = apply_dirichlet_rows(dense, dofs)
        assert isinstance(result, np.ndarray)
        numpy_bkd.assert_allclose(result, _dense_oracle(dense, dofs), rtol=0, atol=0)
        # the input array is left untouched
        numpy_bkd.assert_allclose(
            dense, _random_sparse_dense(7, 0.6, seed=5), rtol=0, atol=0
        )
