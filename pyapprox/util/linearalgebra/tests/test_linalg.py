import unittest

import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.linearalgebra.linalg import (
    PivotedCholeskyFactorizer,
    get_pivot_matrix_from_vector,
)


class TestLinalg:
    def test_update_cholesky_decomposition(self):
        bkd = self.get_backend()
        nvars = 5
        B = bkd.atleast2d(np.random.normal(0, 1, (nvars, nvars)))
        A = B.T @ B

        L = bkd.cholesky(A)
        A_11 = A[: nvars - 2, : nvars - 2]
        A_12 = A[: nvars - 2, nvars - 2 :]
        A_22 = A[nvars - 2 :, nvars - 2 :]
        assert bkd.allclose(bkd.block([[A_11, A_12], [A_12.T, A_22]]), A)
        L_11 = bkd.cholesky(A_11)
        L_up = bkd.update_cholesky_factorization(L_11, A_12, A_22)[0]
        assert bkd.allclose(L, L_up)

    def test_pivoted_cholesky_decomposition(self):
        bkd = self.get_backend()
        nrows, npivots = 4, 4
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, nrows)))
        A = A.T @ A
        factorizer = PivotedCholeskyFactorizer(A, bkd=bkd)
        L = factorizer.factorize(npivots)
        assert np.allclose(L @ L.T, A)

        nrows, npivots = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (npivots, nrows)))
        A = A.T @ A
        factorizer = PivotedCholeskyFactorizer(A, bkd=bkd)
        L = factorizer.factorize(npivots)
        assert L.shape == (nrows, npivots)
        assert factorizer.pivots().shape[0] == npivots
        assert factorizer.npivots() == npivots
        assert np.allclose(L @ L.T, A)

        # check init_pivots are enforced
        nrows, npivots = 4, 3
        A = bkd.asarray(np.random.normal(0.0, 1.0, (npivots, nrows)))
        A = A.T @ A
        factorizer1 = PivotedCholeskyFactorizer(A, bkd=bkd)
        factorizer1.factorize(npivots)
        pivots1 = factorizer1.pivots()
        factorizer2 = PivotedCholeskyFactorizer(A, bkd=bkd)
        L2 = factorizer2.factorize(npivots, init_pivots=pivots1[1:2])
        pivots2 = factorizer2.pivots()
        assert np.allclose(pivots2, pivots1[[1, 0, 2]])
        L = L2[pivots1, :]
        assert np.allclose(A[pivots1, :][:, pivots1], L @ L.T)
        assert np.allclose(A[np.ix_(pivots1, pivots1)], L @ L.T)
        P = get_pivot_matrix_from_vector(pivots1, nrows, bkd)
        assert np.allclose(P @ A @ P.T, L @ L.T)

        A = bkd.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98.0]])
        factorizer = PivotedCholeskyFactorizer(A, bkd=bkd)
        L = factorizer.factorize(A.shape[0])
        # reorder entries of A so that cholesky requires pivoting
        true_pivots = np.array([2, 1, 0])
        A_no_pivots = A[true_pivots, :][:, true_pivots]
        L_np = np.linalg.cholesky(A_no_pivots)
        assert np.allclose(L[factorizer.pivots(), :], L_np)

        # Create A with which needs cholesky with certain pivots
        A = bkd.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98.0]])
        true_pivots = np.array([1, 0, 2])
        A = A[true_pivots, :][:, true_pivots]
        factorizer = PivotedCholeskyFactorizer(A, bkd=bkd)
        L = factorizer.factorize(A.shape[0])
        assert np.allclose(L[factorizer.pivots(), :], L_np)

    def test_update_pivoted_cholesky(self):
        bkd = self.get_backend()
        nrows = 10
        A = bkd.asarray(np.random.normal(0, 1, (nrows, nrows)))
        A = A.T @ A

        # perform full factorization
        pivot_weights = bkd.asarray(np.random.uniform(1, 2, A.shape[0]))
        factorizer1 = PivotedCholeskyFactorizer(A, bkd=bkd)
        L1 = factorizer1.factorize(A.shape[0], pivot_weights=pivot_weights)

        # perform partial facorization
        npivots = A.shape[0]-2
        factorizer2 = PivotedCholeskyFactorizer(A, bkd=bkd)
        factorizer2.factorize(npivots, pivot_weights=pivot_weights)
        assert factorizer2.npivots() == npivots

        # update to complete the facorization
        npivots = A.shape[0]
        L2 = factorizer2.update(npivots)
        assert np.allclose(L2, L1)
        assert np.allclose(factorizer2.pivots(), factorizer1.pivots())


class TestNumpyLinalg(unittest.TestCase, TestLinalg):
    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchLinalg(unittest.TestCase, TestLinalg):
    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
