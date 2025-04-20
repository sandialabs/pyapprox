import unittest

import numpy as np
import scipy

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.util.backends.linalg import (
    PivotedCholeskyFactorizer,
    get_pivot_matrix_from_vector,
    PivotedLUFactorizer,
    pivot_rows,
)


class TestLinalg:
    def setUp(self):
        np.random.seed(1)

    def test_update_cholesky_decomposition(self):
        bkd = self.get_backend()
        nvars = 5
        B = bkd.asarray(np.random.normal(0, 1, (nvars, nvars)))
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
        npivots = A.shape[0] - 2
        factorizer2 = PivotedCholeskyFactorizer(A, bkd=bkd)
        factorizer2.factorize(npivots, pivot_weights=pivot_weights)
        assert factorizer2.npivots() == npivots

        # update to complete the facorization
        npivots = A.shape[0]
        L2 = factorizer2.update(npivots)
        assert np.allclose(L2, L1)
        assert np.allclose(factorizer2.pivots(), factorizer1.pivots())

    def test_pivoted_lu_decomposition(self):
        bkd = self.get_backend()
        nrows, npivots = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, nrows)))
        factorizer = PivotedLUFactorizer(A, bkd=bkd)
        L, U = factorizer.factorize(npivots)
        assert bkd.allclose(L @ U, A[factorizer.pivots(), :npivots])
        scipy_L, scipy_U = scipy.linalg.lu(bkd.to_numpy(A), permute_l=True)
        assert bkd.allclose(L, bkd.asarray(scipy_L)[:npivots, :npivots])
        assert bkd.allclose(U, bkd.asarray(scipy_U)[:npivots, :npivots])

    def test_add_columns_to_pivoted_lu_factorization(self):
        """
        Let
        A  = [1 2 4]
             [2 1 3]
             [3 2 4]

        Recursive Algorithm
        -------------------
        The following Permutation swaps the thrid and first rows
        P1 = [0 0 1]
             [0 1 0]
             [1 0 0]

        Gives
        P1*A  = [3 2 4]
                [2 1 3]
                [1 2 4]

        Conceptually partition matrix into block matrix
        P1*A = [A11 A12]
               [A21 A22]

             = [1    0 ][u11 U12]
               [L21 L22][ 0  U22]
             = [u11           U12      ]
               [u11*L21 L21*U12+L22*U22]

        Then
        u11 = a11
        L21 = 1/a11 A21
        U12 = A12

        e.g.
        a11 = 3  L21 = [2/3]  U12 = [2 4]  u11 = 3
                       [1/3]

        Because A22 = L21*U12+L22*U22
        L22*U22 = A22-L21*U12
        We also know L22=I

        LU sublock after 1 step is
        S1 = L22*U22 = A22-L21*U12

           = [1 3]-[4/3 8/3] = [-1/3 1/3]
             [2 4] [2/3 4/3]   [ 4/3 8/3]

        LU after 1 step is
        LU1 = [u11 U12]
              [L21 S1 ]

              [3     2   4  ]
            = [1/3 -1/3 1/3 ]
              [2/3  4/3 8/32]

        The following Permutation swaps the first and second rows of S1
        P2 = [0 1]
             [1 0]

        Conceptually partition matrix into block matrix
        P2*S1 = [ 4/3 8/3] = [A11 A12]
                [-1/3 1/3] = [A21 A22]

        L21 = 1/a11 A21
        U12 = A12

        e.g.
        a11 = 4/3   L21 = [-1/4]  U12 = [8/3] u11 = 4/3

        LU sublock after 1 step is
        S2 = A22-L21*U12
           = 1/3 + 1/4*8/3 = 1

        LU after 2 step is
        LU2 = [ 3    2   4 ]
              [1/3  u11 U12]
              [2/3  L21 S2 ]

            = [ 3    2   4 ]
              [1/3  4/3 8/3]
              [2/3 -1/4 S2 ]


        Matrix multiplication algorithm
        -------------------------------
        The following Permutation swaps the thrid and first rows
        P1 = [0 0 1]
             [0 1 0]
             [1 0 0]

        Gives
        P1*A  = [3 2 4]
                [2 1 3]
                [1 2 4]

        Use Matrix M1 to eliminate entries in second and third row of column 1
             [  1  0 1]
        M1 = [-2/3 1 0]
             [-1/3 0 1]

        So U factor after step 1 is
        U1  = M1*P1*A

              [3   2   4  ]
            = [0 -1/3 1/3 ]
              [0  4/3 8/32]

        The following Permutation swaps the third and second rows
        P2 = [1 0 0]
             [0 0 1]
             [0 1 0]

        M2 = [1  0  0]
             [0  1  0]
             [0 1/4 1]

        U factor after step 2 is
        U2  = M2*P2*M1*P1*A

              [3  2   4  ]
            = [0 4/3 8/3 ]
              [0  0   1  ]

        L2 = (M2P2M1P1)^{-1}
           = [ 1    0  0]
             [1/3   1  0]
             [2/3 -1/4 1]

        P*A = P2*P1*A = L2U2
        """
        bkd = self.get_backend()
        A = bkd.asarray(np.random.normal(0, 1, (6, 6)))

        npivots = 2
        factorizer = PivotedLUFactorizer(A[:, :npivots], bkd=bkd)
        L_init, U_init = factorizer.factorize(npivots)
        new_cols = bkd.copy(A[:, npivots:])
        factorizer.add_columns(new_cols)

        full_factorizer = PivotedLUFactorizer(A, bkd=bkd)
        full_factorizer.factorize(npivots)
        assert np.allclose(factorizer._LU_factor, full_factorizer._LU_factor)

    def test_add_rows_to_pivoted_lu_factorization(self):
        bkd = self.get_backend()
        np.random.seed(3)
        A = np.random.normal(0, 1, (10, 3))

        npivots = A.shape[1]
        factorizer1 = PivotedLUFactorizer(A, bkd=bkd)
        factorizer1.factorize(npivots)

        # create matrix for which pivots do not matter
        A = pivot_rows(factorizer1.pivots(), A, False)
        # check no pivoting is necessary
        factorizer2 = PivotedLUFactorizer(A, bkd=bkd)
        factorizer2.factorize(npivots)
        assert bkd.allclose(factorizer2.pivots(), np.arange(npivots))

        factorizer3 = PivotedLUFactorizer(A[:npivots], bkd=bkd)
        factorizer3.factorize(npivots)
        new_rows = bkd.copy(A[npivots:, :])
        factorizer3.add_rows(new_rows)
        print(factorizer3._LU_factor)
        print(factorizer1._LU_factor)
        assert bkd.allclose(factorizer3._LU_factor, factorizer1._LU_factor)

        #######
        # only pivot some of the rows

        A = np.random.normal(0, 1, (10, 5))

        npivots = 3
        LU_factor, pivots = truncated_pivoted_lu_factorization(
            A, npivots, truncate_L_factor=False
        )

        # create matrix for which pivots do not matter
        A = pivot_rows(pivots, A, False)
        # print(A.shape)
        # check no pivoting is necessary
        L, U, pivots = truncated_pivoted_lu_factorization(
            A, npivots, truncate_L_factor=True
        )
        assert np.allclose(pivots, np.arange(npivots))

        LU_factor_init, pivots_init = truncated_pivoted_lu_factorization(
            A[:npivots, :], npivots, truncate_L_factor=False
        )

        new_rows = A[npivots:, :].copy()

        LU_factor_final = add_rows_to_pivoted_lu_factorization(
            LU_factor_init, new_rows, npivots
        )
        assert np.allclose(LU_factor_final, LU_factor)


class TestNumpyLinalg(TestLinalg, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin

    def test_repeat(self):
        # I had to implement torch repeat to be consistent with numpy
        # so test here
        np_bkd = self.get_backend()
        torch_bkd = TorchMixin
        nreps = 5

        # 1D array
        np_Amat = np_bkd.arange(12)
        torch_Amat = torch_bkd.asarray(np_Amat, dtype=int)
        assert np.allclose(
            np_bkd.repeat(np_Amat, nreps),
            torch_bkd.repeat(torch_Amat, nreps).numpy(),
        )

        # 2D array
        np_Amat = np_bkd.arange(12).reshape(4, 3)
        torch_Amat = torch_bkd.asarray(np_Amat, dtype=int)
        assert np.allclose(
            np_bkd.repeat(np_Amat, nreps),
            torch_bkd.repeat(torch_Amat, nreps).numpy(),
        )
        assert np.allclose(
            np_bkd.repeat(np_Amat, nreps, axis=0),
            torch_bkd.repeat(torch_Amat, nreps, axis=0).numpy(),
        )
        assert np.allclose(
            np_bkd.repeat(np_Amat, nreps, axis=1),
            torch_bkd.repeat(torch_Amat, nreps, axis=1).numpy(),
        )


class TestTorchLinalg(TestLinalg, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
