import unittest
import warnings


import numpy as np
import scipy

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.util.linalg import (
    PivotedCholeskyFactorizer,
    get_pivot_matrix_from_vector,
    PivotedLUFactorizer,
    pivot_rows,
    nentries_square_triangular_matrix,
    nentries_rectangular_triangular_matrix,
    flattened_rectangular_lower_triangular_matrix_index,
    inverse_from_cholesky_factor,
    inverse_of_cholesky_factor,
    cholesky_solve_linear_system,
    log_determinant_from_cholesky_factor,
    qr_solve,
    diag_of_mat_mat_product,
    trace_of_mat_mat_product,
    get_low_rank_matrix,
    DenseMatVecOperator,
    DenseSymmetricMatVecOperator,
    SinglePassRandomizedSVD,
    SymmetricMatrixDoublePassRandomizedSVD,
    adjust_sign_svd,
    adjust_sign_eig,
    invert_permutation_vector,
    update_cholesky_factorization_inverse,
    update_trace_involving_cholesky_inverse,
)


class TestLinalg:
    def setUp(self):
        warnings.filterwarnings("error")
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
        assert factorizer.success()
        assert bkd.allclose(L @ L.T, A)

        nrows, npivots = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (npivots, nrows)))
        A = A.T @ A
        factorizer = PivotedCholeskyFactorizer(A, bkd=bkd)
        L = factorizer.factorize(npivots)
        assert L.shape == (nrows, npivots)
        assert factorizer.pivots().shape[0] == npivots
        assert factorizer.npivots() == npivots
        assert bkd.allclose(L @ L.T, A)

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
        assert bkd.allclose(pivots2, pivots1[[1, 0, 2]])
        L = L2[pivots1, :]
        assert bkd.allclose(A[pivots1, :][:, pivots1], L @ L.T)
        assert bkd.allclose(A[np.ix_(pivots1, pivots1)], L @ L.T)
        P = get_pivot_matrix_from_vector(pivots1, nrows, bkd)
        assert bkd.allclose(P @ A @ P.T, L @ L.T)

        A = bkd.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98.0]])
        factorizer = PivotedCholeskyFactorizer(A, bkd=bkd)
        L = factorizer.factorize(A.shape[0])
        # reorder entries of A so that cholesky requires pivoting
        true_pivots = np.array([2, 1, 0])
        A_no_pivots = A[true_pivots, :][:, true_pivots]
        L_np = bkd.cholesky(A_no_pivots)
        assert bkd.allclose(L[factorizer.pivots(), :], L_np)

        # Create A with which needs cholesky with certain pivots
        A = bkd.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98.0]])
        true_pivots = np.array([1, 0, 2])
        A = A[true_pivots, :][:, true_pivots]
        factorizer = PivotedCholeskyFactorizer(A, bkd=bkd)
        L = factorizer.factorize(A.shape[0])
        assert bkd.allclose(L[factorizer.pivots(), :], L_np)

        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, nrows)))
        A = A.T @ A
        factorizer = PivotedCholeskyFactorizer(A, econ=False, bkd=bkd)
        L = factorizer.factorize(nrows)
        assert factorizer.success()
        assert bkd.allclose(L @ L.T, A)

        rank = 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, rank)))
        A = A.T @ A
        factorizer = PivotedCholeskyFactorizer(A, econ=False, bkd=bkd)
        self.assertRaises(ValueError, factorizer.factorize, nrows)

        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, rank)))
        A = A @ A.T
        factorizer = PivotedCholeskyFactorizer(A, econ=True, bkd=bkd)
        L = factorizer.factorize(nrows)
        assert bkd.allclose(L @ L.T, A)

        factorizer = PivotedCholeskyFactorizer(A, econ=False, bkd=bkd)
        L = factorizer.factorize(nrows)
        assert bkd.allclose(L @ L.T, A)

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
        assert bkd.allclose(L2, L1)
        assert bkd.allclose(factorizer2.pivots(), factorizer1.pivots())

    def test_pivoted_lu_decomposition(self):
        bkd = self.get_backend()
        nrows, npivots = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, nrows)))
        factorizer = PivotedLUFactorizer(A, bkd=bkd)
        L, U = factorizer.factorize(npivots)
        assert factorizer.success()
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
        assert bkd.allclose(factorizer._LU_factor, full_factorizer._LU_factor)

        factorizer.update(3)
        full_factorizer = PivotedLUFactorizer(A, bkd=bkd)
        full_factorizer.factorize(3, init_pivots=factorizer.pivots())
        assert bkd.allclose(factorizer._LU_factor, full_factorizer._LU_factor)

    def test_add_rows_to_pivoted_lu_factorization(self):
        bkd = self.get_backend()
        np.random.seed(3)
        A = bkd.asarray(np.random.normal(0, 1, (10, 3)))

        npivots = A.shape[1]
        factorizer1 = PivotedLUFactorizer(A, bkd=bkd)
        factorizer1.factorize(npivots)

        # create matrix for which pivots do not matter
        A_reordered = pivot_rows(factorizer1._seq_pivots, A, False, bkd=bkd)
        assert bkd.allclose(A[factorizer1.pivots()], A_reordered[:npivots])
        # check no pivoting is necessary
        factorizer2 = PivotedLUFactorizer(A_reordered, bkd=bkd)
        factorizer2.factorize(npivots)
        assert bkd.allclose(factorizer2.pivots(), bkd.arange(npivots))

        factorizer3 = PivotedLUFactorizer(A_reordered[:npivots], bkd=bkd)
        factorizer3.factorize(npivots)
        new_rows = bkd.copy(A_reordered[npivots:, :])
        factorizer3.add_rows(new_rows)
        assert bkd.allclose(factorizer3._LU_factor, factorizer1._LU_factor)

        #######
        # only pivot some of the rows

        A = bkd.asarray(np.random.normal(0, 1, (10, 5)))

        npivots = 3
        factorizer1 = PivotedLUFactorizer(A, bkd=bkd)
        factorizer1.factorize(npivots)
        assert (
            factorizer1.termination_message()
            == "Factorization completed successfully"
        )

        # create matrix for which pivots do not matter
        A_reordered = pivot_rows(factorizer1._seq_pivots, A, False, bkd=bkd)
        # check no pivoting is necessary
        factorizer2 = PivotedLUFactorizer(A_reordered, bkd=bkd)
        factorizer2.factorize(npivots)
        assert bkd.allclose(factorizer2.pivots(), bkd.arange(npivots))

        factorizer3 = PivotedLUFactorizer(A_reordered[:npivots], bkd=bkd)
        factorizer3.factorize(npivots)
        new_rows = bkd.copy(A_reordered[npivots:, :])
        new_rows = bkd.copy(A_reordered[npivots:, :])
        factorizer3.add_rows(new_rows)
        assert bkd.allclose(factorizer3._LU_factor, factorizer1._LU_factor)

        # after rows are added make sure factorization can be updated
        factorizer3.update(npivots + 1)
        full_factorizer = PivotedLUFactorizer(A_reordered, bkd=bkd)
        full_factorizer.factorize(
            npivots + 1, init_pivots=factorizer3.pivots()
        )
        assert bkd.allclose(factorizer3._LU_factor, full_factorizer._LU_factor)

    def test_undo_lu_preconditioning(self):
        # set seed so that pivots are not taken in order by chance
        # this picked up a previous bug did not stop test passing
        # when pivots were in sequential order
        np.random.seed(2)
        bkd = self.get_backend()
        A = bkd.asarray(np.random.normal(0, 1, (4, 4)))
        precond_weights = 1 / bkd.norm(A, axis=1)[:, None]
        npivots = A.shape[1]
        factorizer1 = PivotedLUFactorizer(precond_weights * A, bkd=bkd)
        L_precond, U_precond = factorizer1.factorize(npivots)
        assert bkd.allclose(
            L_precond @ U_precond, (precond_weights * A)[factorizer1.pivots()]
        )

        factorizer2 = PivotedLUFactorizer(A, bkd=bkd)
        L, U = factorizer2.factorize(npivots, init_pivots=factorizer1.pivots())
        assert bkd.allclose(factorizer2.pivots(), factorizer1.pivots())
        assert bkd.allclose(L @ U, A[factorizer2.pivots()])

        pivoted_precond_weights = precond_weights[factorizer1.pivots()]
        W = bkd.diag(pivoted_precond_weights[:, 0])
        Winv = bkd.inv(W)
        assert bkd.allclose(
            Winv @ (L_precond @ U_precond), A[factorizer1.pivots()]
        )
        assert bkd.allclose(
            (L_precond / pivoted_precond_weights) @ U_precond,
            A[factorizer1.pivots()],
        )
        # inv(W)*L*W*inv(W)*U
        L_adjusted = (
            1 / pivoted_precond_weights * L_precond * pivoted_precond_weights.T
        )
        U_adjusted = U_precond / pivoted_precond_weights
        assert bkd.allclose(L_adjusted @ U_adjusted, A[factorizer1.pivots()])
        assert bkd.allclose(L_adjusted, L)
        assert bkd.allclose(U_adjusted, U)

        L_adjusted, U_adjusted = factorizer1.undo_preconditioning(
            precond_weights, npivots
        )
        assert bkd.allclose(L_adjusted, L)
        assert bkd.allclose(U_adjusted, U)

        # test undo preconditioning when not all pivots are taken
        npivots = A.shape[1] - 1
        factorizer1 = PivotedLUFactorizer(precond_weights * A, bkd=bkd)
        L_precond, U_precond = factorizer1.factorize(npivots)

        factorizer2 = PivotedLUFactorizer(A, bkd=bkd)
        L, U = factorizer2.factorize(npivots, init_pivots=factorizer1.pivots())

        L_adjusted, U_adjusted = factorizer1.undo_preconditioning(
            precond_weights, npivots, update_internal_state=True
        )
        assert bkd.allclose(factorizer1._LU_factor, factorizer2._LU_factor)

    def test_update_lu_preconditioning(self):
        # set seed so that pivots are not taken in order by chance
        # this picked up a previous bug did not stop test passing
        # when pivots were in sequential order
        np.random.seed(2)
        bkd = self.get_backend()
        A = bkd.asarray(np.random.normal(0, 1, (4, 4)))
        precond_weights = 1 / bkd.norm(A, axis=1)[:, None]
        npivots = A.shape[1]
        factorizer1 = PivotedLUFactorizer(precond_weights * A, bkd=bkd)
        L_precond, U_precond = factorizer1.factorize(npivots)
        assert bkd.allclose(
            L_precond @ U_precond, (precond_weights * A)[factorizer1.pivots()]
        )
        new_precond_weights = precond_weights**2
        factorizer2 = PivotedLUFactorizer(new_precond_weights * A, bkd=bkd)
        L, U = factorizer2.factorize(npivots, init_pivots=factorizer1.pivots())
        assert bkd.allclose(factorizer2.pivots(), factorizer1.pivots())
        assert bkd.allclose(
            L @ U, (precond_weights**2 * A)[factorizer2.pivots()]
        )

        L_adjusted, U_adjusted = factorizer1.update_preconditioning(
            precond_weights,
            new_precond_weights,
            npivots,
            update_internal_state=True,
        )
        assert bkd.allclose(L_adjusted, L)
        assert bkd.allclose(U_adjusted, U)
        assert bkd.allclose(factorizer1._LU_factor, factorizer2._LU_factor)

        # test undo preconditioning when not all pivots are taken
        npivots = A.shape[1] - 1
        factorizer1 = PivotedLUFactorizer(precond_weights * A, bkd=bkd)
        L_precond, U_precond = factorizer1.factorize(npivots)

        factorizer2 = PivotedLUFactorizer(new_precond_weights * A, bkd=bkd)
        L, U = factorizer2.factorize(npivots, init_pivots=factorizer1.pivots())

        L_adjusted, U_adjusted = factorizer1.update_preconditioning(
            precond_weights,
            new_precond_weights,
            npivots,
            update_internal_state=True,
        )
        assert bkd.allclose(factorizer1._LU_factor, factorizer2._LU_factor)

    def test_nentries_triangular_matrix(self):
        M = 4
        A = np.ones([M, M])
        L = np.tril(A)
        nentries = nentries_square_triangular_matrix(M, include_diagonal=True)
        assert nentries == np.count_nonzero(L)

        nentries = nentries_square_triangular_matrix(M, include_diagonal=False)
        assert nentries == np.count_nonzero(L) - M

        M, N = 4, 3
        A = np.ones([M, N])
        L = np.tril(A)
        nentries = nentries_rectangular_triangular_matrix(M, N, upper=False)
        assert nentries == np.count_nonzero(L)

        A = np.ones([M, N])
        U = np.triu(A)
        nentries = nentries_rectangular_triangular_matrix(M, N, upper=True)
        assert nentries == np.count_nonzero(U)

    def test_flattened_rectangular_lower_triangular_matrix_index(self):

        M, N = 4, 3
        tril_indices = np.tril_indices(M, m=N)
        for nn in range(tril_indices[0].shape[0]):
            ii, jj = tril_indices[0][nn], tril_indices[1][nn]
            kk = flattened_rectangular_lower_triangular_matrix_index(
                ii, jj, M, N
            )
            assert kk == nn

    def test_cholesky_functions(self):
        bkd = self.get_backend()
        nvars = 5
        B = bkd.asarray(np.random.normal(0, 1, (nvars, nvars)))
        A = B.T @ B
        chol = bkd.cholesky(A)
        assert bkd.allclose(
            inverse_from_cholesky_factor(chol, bkd=bkd), bkd.inv(A)
        )
        assert bkd.allclose(
            inverse_of_cholesky_factor(chol, bkd=bkd), bkd.inv(chol)
        )
        rhs = bkd.asarray(np.random.normal(0, 1, (nvars, 1)))
        assert bkd.allclose(
            cholesky_solve_linear_system(chol, rhs, bkd=bkd),
            bkd.inv(A) @ rhs,
        )
        assert bkd.allclose(
            log_determinant_from_cholesky_factor(chol, bkd=bkd),
            bkd.slogdet(A)[1],
        )

    def test_matrix_functions(self):
        bkd = self.get_backend()
        nvars = 5
        B = bkd.asarray(np.random.normal(0, 1, (nvars, nvars)))
        A = B.T @ B
        rhs = bkd.asarray(np.random.normal(0, 1, (nvars, 1)))
        Q, R = bkd.qr(A)
        assert bkd.allclose(
            qr_solve(Q, R, rhs, bkd=bkd),
            bkd.inv(A) @ rhs,
        )

        assert bkd.allclose(diag_of_mat_mat_product(Q, R, bkd), bkd.diag(A))
        assert bkd.allclose(trace_of_mat_mat_product(Q, R, bkd), bkd.trace(A))

    def test_get_low_rank_matrix(self):
        bkd = self.get_backend()
        A = get_low_rank_matrix(4, 5, 2, bkd)
        assert bkd.rank(A) == 2

    def test_randomized_svd(self):
        bkd = self.get_backend()
        rank = 2

        A = get_low_rank_matrix(4, 5, rank, bkd)
        matvec = DenseMatVecOperator(A, backend=bkd)
        svd = SinglePassRandomizedSVD(matvec)
        U, S, Vh = svd.compute(rank)
        U_true, S_true, Vh_true = bkd.svd(A)
        U_true, Vh_true = adjust_sign_svd(U, Vh, bkd=bkd)
        assert bkd.allclose(S, S_true[:rank])
        assert bkd.allclose(U, U_true[:, :rank])
        assert bkd.allclose(Vh, Vh_true[:rank])

        B = A.T @ A
        matvec = DenseSymmetricMatVecOperator(B, backend=bkd)
        svd = SinglePassRandomizedSVD(matvec)
        U, S, Vh = svd.compute(rank)
        U_true, S_true, Vh_true = bkd.svd(B)
        U_true, Vh_true = adjust_sign_svd(U, Vh, bkd=bkd)
        assert bkd.allclose(S, S_true[:rank])
        assert bkd.allclose(U, U_true[:, :rank])
        assert bkd.allclose(Vh, Vh_true[:rank])

        svd = SymmetricMatrixDoublePassRandomizedSVD(matvec)
        U, S, Vh = svd.compute(rank)
        U_true, S_true, Vh_true = bkd.svd(B)
        U_true, Vh_true = adjust_sign_svd(U, Vh, bkd=bkd)
        assert bkd.allclose(S, S_true[:rank])
        assert bkd.allclose(U, U_true[:, :rank])
        assert bkd.allclose(Vh, Vh_true[:rank])

    def test_invert_permutation_vector(self):
        nvars = 5
        bkd = self.get_backend()
        B = bkd.asarray(np.random.normal(0, 1, (nvars, nvars)))
        pivots = bkd.array([0, 2, 1, 4, 3], dtype=int)
        C = B[pivots]
        assert bkd.allclose(C[invert_permutation_vector(pivots, bkd=bkd)], B)

    def test_adjust_sign_eig(self):
        bkd = self.get_backend()
        nvars = 5
        B = bkd.asarray(np.random.normal(0, 1, (nvars, nvars)))
        A = B.T @ B
        S, U = bkd.eigh(A)
        U_adjusted = adjust_sign_eig(U, bkd=bkd)
        # regression test for np.random.seed(1)
        assert bkd.allclose(U_adjusted, U)

    def test_update_cholesky_factorization_inverse(self):
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
        L_inv = bkd.inv(L)
        L_11_inv = bkd.inv(L_11)
        L_12_T = L[L_11.shape[0] :, : L_11.shape[1]]
        L_12 = L_12_T.T
        L_22 = L[L_11.shape[0] :, L_11.shape[0] :]
        assert bkd.allclose(
            L_inv,
            update_cholesky_factorization_inverse(
                L_11_inv, L_12, L_22, bkd=bkd
            ),
        )

        L_22_inv = bkd.inv(L_22)
        C = -(L_22_inv @ L_12.T) @ L_11_inv
        A_inv = bkd.block(
            [
                [L_11_inv.T @ (L_11_inv) + C.T @ C, C.T @ (L_22_inv)],
                [L_22_inv.T @ (C), L_22_inv.T @ (L_22_inv)],
            ]
        )
        assert bkd.allclose(A_inv, bkd.inv(A))

        B_11 = B[: A_11.shape[0], : A_11.shape[1]]
        prev_trace = np.trace(bkd.inv(A_11) @ B_11)
        trace = update_trace_involving_cholesky_inverse(
            L_11_inv, L_12, L_22_inv, B, prev_trace, bkd=bkd
        )
        assert bkd.allclose(trace, bkd.trace(bkd.inv(A) @ B))


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
    unittest.main(verbosity=2)
