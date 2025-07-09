from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import scipy

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin


def swap_rows(
    matrix: Array, ii: int, jj: int, bkd: BackendMixin = NumpyMixin
) -> Array:
    matrix[bkd.hstack([ii, jj])] = matrix[bkd.hstack([jj, ii])]


def pivot_rows(
    pivots: Array,
    matrix: Array,
    in_place: bool = True,
    bkd: BackendMixin = NumpyMixin,
) -> Array:
    if not in_place:
        matrix = bkd.copy(matrix)
    npivots = pivots.shape[0]
    assert npivots <= matrix.shape[0]
    for ii in range(npivots):
        swap_rows(matrix, ii, pivots[ii])
    return matrix


def get_pivot_matrix_from_vector(
    pivots: Array, nrows: int, bkd: BackendMixin = NumpyMixin
) -> Array:
    P = bkd.eye(nrows)
    P = P[pivots, :]
    return P


def get_final_pivots_from_sequential_pivots(
    sequential_pivots: Array,
    npivots: int = None,
    bkd: BackendMixin = NumpyMixin,
) -> Array:
    """
    Parameters
    ----------
    sequential_pivots: Array
        Pivot vector obtained by inserting pivot at each iteratation

    final_pivots: Array
        The vector that changes the original array to the final permuated
        array in 1 shot
    """
    if npivots is None:
        npivots = sequential_pivots.shape[0]
    assert npivots >= sequential_pivots.shape[0]
    pivots = bkd.arange(npivots)
    return pivot_rows(sequential_pivots, pivots, False, bkd=bkd)


class PivotedFactorizer(ABC):
    def __init__(
        self,
        Amat: Array,
        tol: float = 1e-14,
        econ: bool = True,
        bkd: BackendMixin = NumpyMixin,
    ):
        self._Amat = bkd.copy(Amat)
        self._tol = tol
        self._econ = econ
        self._bkd = bkd

        self._nrows = self._Amat.shape[0]
        self._pivots = self._bkd.arange(self._nrows)

    def pivots(self) -> Array:
        return self._pivots[: self._ncompleted_pivots]

    def npivots(self) -> int:
        return self._ncompleted_pivots

    def termination_message(self) -> str:
        return self._termination_msg

    @abstractmethod
    def success(self) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        if self.npivots() == 0:
            return "{0}(nrows={1}, ncols={2})".format(
                self.__class__.__name__, *self._Amat.shape
            )
        return "{0}(nrows={1}, ncols={2}, npivots={3}, msg={4})".format(
            self.__class__.__name__,
            *self._Amat.shape,
            self.npivots(),
            self.termination_message(),
        )

    @abstractmethod
    def update(self, npivots: int) -> Array:
        raise NotImplementedError

    @abstractmethod
    def factorize(
        self,
        npivots: int,
        init_pivots: Array = None,
        pivot_weights: Array = None,
    ) -> Array:
        raise NotImplementedError


class PivotedCholeskyFactorizer(PivotedFactorizer):

    def factorize(
        self,
        npivots: int,
        init_pivots: Array = None,
        pivot_weights: Array = None,
    ) -> Array:
        r"""
        Return a low-rank pivoted Cholesky decomposition of matrix A.

        If A is positive definite and npivots is equal to the number of
        rows of A then L @ L.T == A

        To obtain the pivoted form of L set
        L = L[pivots,:]

        Then P.T @ A @ P == L @ L.T

        where P is the standard pivot matrix which can be obtained from the
        pivot vector using the function
        """
        assert self._Amat.shape[1] == self._nrows
        if npivots > self._nrows:
            raise ValueError(
                "Number of pivots requested exceeds number of matrix rows"
            )
        self._init_pivots = init_pivots
        self._pivot_weights = pivot_weights
        self._ncompleted_pivots = 0

        self._L = self._bkd.zeros(((self._nrows, self._nrows)))
        # return a view of diag
        self._diag = self._Amat.ravel()[:: self._Amat.shape[0] + 1]
        # return a copy of diag
        # diag1 = bkd.diag(Amat).copy()
        # assert bkd.allclose(diag,diag1)
        self._init_error = self._bkd.sum(self._bkd.abs(self._diag))
        return self.update(npivots)[:, : self._ncompleted_pivots]

    def success(self) -> bool:
        return self._termination_flag < 2

    def _compute_pivot(self, Amat: Array, ii: int) -> int:
        if self._init_pivots is not None and ii < len(self._init_pivots):
            return self._bkd.where(self._pivots == self._init_pivots[ii])[0][0]

        if self._econ:
            if self._pivot_weights is None:
                return self._bkd.argmax(self._diag[self._pivots[ii:]]) + ii
            return (
                self._bkd.argmax(
                    self._pivot_weights[self._pivots[ii:]]
                    * self._diag[self._pivots[ii:]]
                )
                + ii
            )

        schur_complement = Amat[
            np.ix_(self._pivots[ii:], self._pivots[ii:])
        ] - self._L[self._pivots[ii:], :ii] @ (
            self._L[self._pivots[ii:], :ii].T
        )
        schur_diag = self._bkd.diag(schur_complement)
        pivot = self._bkd.argmax(
            self._bkd.norm(schur_complement, axis=0) ** 2 / schur_diag
        )
        pivot += ii
        return pivot

    def _is_positive_semi_definite(self, ii: int) -> bool:
        if self._diag[self._pivots[ii]] <= 1e-14:
            self._termination_flag = 3
            raise RuntimeError("matrix is not positive definite")
        return True

    def update(self, npivots: int) -> Array:
        Amat = self._bkd.copy(self._Amat)  # Do not overwrite incoming Amat
        if self._econ is False and self._pivot_weights is not None:
            msg = "pivot weights not used when econ is False"
            raise ValueError(msg)
        if self._ncompleted_pivots >= npivots:
            raise ValueError("Too many pivots requested")
        for ii in self._bkd.arange(self._ncompleted_pivots, npivots):
            pivot = self._compute_pivot(Amat, ii)
            swap_rows(self._pivots, ii, pivot, self._bkd)
            self._is_positive_semi_definite(ii)

            self._L[self._pivots[ii], ii] = self._bkd.sqrt(
                self._diag[self._pivots[ii]]
            )

            self._L[self._pivots[ii + 1 :], ii] = (
                Amat[self._pivots[ii + 1 :], self._pivots[ii]]
                - self._L[self._pivots[ii + 1 :], :ii]
                @ (self._L[self._pivots[ii], :ii])
            ) / self._L[self._pivots[ii], ii]
            self._diag[self._pivots[ii + 1 :]] -= (
                self._L[self._pivots[ii + 1 :], ii] ** 2
            )

            self._ncompleted_pivots += 1
            if self._terminate(ii):
                return self._L
        self._termination_flag = 0
        self._termination_msg = "Factorization completed successfully"
        return self._L

    def _terminate(self, ii: int) -> bool:
        rel_error = self._diag[self._pivots[ii + 1 :]].sum() / self._init_error
        if rel_error >= self._tol:
            return False

        # If matrix is rank r then then error will be machine precision
        # In such a case exiting without an error is the right thing to
        # do
        self._termination_flag = 1
        self._termination_msg = "Tolerance reached. "
        f"Iteration:{ii}. Tol={self._tol}. Rel. Error={rel_error}"
        return True

    def solve_linear_system(self, rhs: Array) -> Array:
        # rhs must use pivoted ordering
        return cholesky_solve_linear_system(
            self._L[self.pivots(), : self._ncompleted_pivots],
            rhs,
            bkd=self._bkd,
        )


class PivotedLUFactorizer(PivotedFactorizer):
    def _best_pivot(self, it: int) -> int:
        if self._init_pivots is not None and it < self._init_pivots.shape[0]:
            return self._init_pivots[it]
        else:
            return np.argmax(np.absolute(self._LU_factor[it:, it])) + it

    def _terminate(self, it: int) -> bool:
        # check for singularity
        if abs(self._LU_factor[it, it]) < np.finfo(float).eps:
            self._termination_msg = (
                "pivot %1.2e" % abs(self._LU_factor[it, it])
                + " is to small. Stopping factorization."
            )
            return True

    def _split_lu(self, LU_factor: Array, npivots: int) -> Tuple[Array, Array]:
        r"""
        Return the L and U factors of an inplace LU factorization

        Parameters
        ----------
        npivots : integer
            The number of pivots performed. This allows LU in place matrix
            to be split during evolution of LU algorithm
        """
        if npivots is None:
            npivots = min(*LU_factor.shape)
        L_factor = self._bkd.tril(LU_factor)
        if L_factor.shape[1] < L_factor.shape[0]:
            # if matrix over-determined ensure L is a square matrix
            n0 = L_factor.shape[0] - L_factor.shape[1]
            L_factor = self._bkd.hstack(
                [L_factor, self._bkd.zeros((L_factor.shape[0], n0))]
            )
        # np.fill_diagonal(L_factor, 1.0)
        for ii in range(L_factor.shape[0]):
            L_factor[ii, ii] = 1.0
        U_factor = self._bkd.triu(LU_factor)
        return L_factor[:npivots, :npivots], U_factor[:npivots, :npivots]

    def factorize(
        self,
        npivots: int,
        init_pivots: Array = None,
        pivot_weights: Array = None,
    ) -> Array:
        self._init_pivots = init_pivots
        self._ncompleted_pivots = 0
        self._LU_factor = self._bkd.copy(self._Amat)
        self._seq_pivots = self._bkd.arange(self._Amat.shape[0])
        npivots = min(npivots, min(*self._Amat.shape))
        for it in range(npivots):
            pivot = self._best_pivot(it)
            # update pivots vector
            self._seq_pivots[it] = pivot
            # apply pivots(swap rows) in L factorization
            swap_rows(self._LU_factor, it, pivot)
            if self._terminate(it):
                self._termination_flag = 1
                return self._split_lu(self._LU_factor, it)
            # update L_factor
            self._LU_factor[it + 1 :, it] /= self._LU_factor[it, it]
            # udpate U_factor
            col_vector = self._LU_factor[it + 1 :, it]
            row_vector = self._LU_factor[it, it + 1 :]
            update = col_vector[:, None] @ row_vector[None, :]
            self._LU_factor[it + 1 :, it + 1 :] -= update
            self._ncompleted_pivots += 1
        self._termination_msg = "Factorization completed successfully"
        self._termination_flag = 0
        return self._split_lu(self._LU_factor, self._ncompleted_pivots)

    def update(self, npivots: int) -> Array:
        raise NotImplementedError

    def success(self) -> bool:
        return self._termination_flag == 0

    def add_rows(self, new_rows: Array):
        if self._LU_factor.shape[1] != new_rows.shape[1]:
            raise ValueError("new_rows has the wrong number of columns")
        LU_factor_extra = self._bkd.copy(new_rows)
        for it in range(self._ncompleted_pivots):
            LU_factor_extra[:, it] /= self._LU_factor[it, it]
            col_vector = LU_factor_extra[:, it]
            row_vector = self._LU_factor[it, it + 1 :]
            update = col_vector[:, None] @ row_vector[None, :]
            LU_factor_extra[:, it + 1 :] -= update
        self._LU_factor = self._bkd.vstack([self._LU_factor, LU_factor_extra])

    def add_columns(self, new_cols: Array):
        if self._LU_factor.shape[0] != new_cols.shape[0]:
            raise ValueError("new_cols has the wrong number of rows")
        for it, pivot in enumerate(
            self._seq_pivots[: self._ncompleted_pivots]
        ):
            # inlined swap_rows() for performance
            # copy required by torch
            new_cols[[it, pivot], :] = self._bkd.copy(new_cols[[pivot, it], :])

            # update LU_factor
            # recover state of col vector from permuted LU factor
            # Let  (jj,kk) represent iteration and pivot pairs
            # then if lu factorization produced sequence of pairs
            # (0,4),(1,2),(2,4) then LU_factor[:,0] here will be col_vector
            # in LU algorithm with the second and third permutations
            # so undo these permutations in reverse order
            next_idx = it + 1

            # `col_vector` is a copy of the LU_factor subset
            col_vector = self._bkd.copy(self._LU_factor[next_idx:, it])
            for ii in range(self._ncompleted_pivots - it - 1):
                # (it+1) necessary in two lines below because only dealing
                # with compressed col vector which starts at row it in
                # LU_factor
                jj = (
                    self._seq_pivots[self._ncompleted_pivots - 1 - ii]
                    - next_idx
                )
                kk = self._ncompleted_pivots - ii - 1 - next_idx

                # inlined swap_rows()
                # copy required by torch
                col_vector[jj], col_vector[kk] = (
                    self._bkd.copy(col_vector[kk]),
                    self._bkd.copy(col_vector[jj]),
                )

            new_cols[next_idx:, :] -= np.outer(col_vector, new_cols[it, :])
        self._LU_factor = self._bkd.hstack((self._LU_factor, new_cols))

    def pivots(self) -> Array:
        return get_final_pivots_from_sequential_pivots(
            self._seq_pivots, bkd=self._bkd
        )[: self._ncompleted_pivots]

    def undo_preconditionining(
        self,
        precond_weights: Array,
        npivots: int = None,
    ) -> Array:
        r"""
        A=LU and WA=XY
        Then WLU=XY
        We also know Y=WU
        So WLU=XWU => WL=XW so L=inv(W)*X*W
        and U = inv(W)Y
        """
        if npivots is None:
            npivots = min(*self._LU_factor.shape)
        if precond_weights.shape != (self._LU_factor.shape[0], 1):
            raise ValueError("precond_weights must be a 2d column vector")
        # left multiply L an U by inv(W), i.e. compute inv(W).dot(L)
        # and inv(W).dot(U)
        LU_factor = self._bkd.copy(self._LU_factor) / precond_weights
        # right multiply L by W, i.e. compute L.dot(W)
        # Do not overwrite columns past npivots. If not all pivots have been
        # performed the columns to the right of this point contain U factor
        for ii in range(npivots):
            LU_factor[ii + 1 :, ii] *= precond_weights[ii, 0]
        return self._split_lu(LU_factor, npivots)


def adjust_sign_eig(U: Array, bkd: BackendMixin = NumpyMixin) -> Array:
    r"""
    Ensure uniquness of eigenvalue decompotision by ensuring the largest
    (magnitude) entry of the first singular vector of U is positive.

    Parameters
    ----------
    U : (M x M) matrix
        left singular vectors of a singular value decomposition of a (M x M)
        matrix A.

    Returns
    -------
    U : (M x M) matrix
       left singular vectors with first entry of the first
       singular vector always being positive.
    """
    idx = bkd.argmax(bkd.abs(U[0, :]))
    s = bkd.sign(U[idx, :])
    II = bkd.where(s == 0)[0]
    s[II] = 1.0
    U *= s
    return U


class MatVecOperator(ABC):
    @abstractmethod
    def apply(self, vecs: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def apply_transpose(self, vecs: Array) -> Array:
        raise NotImplementedError

    def right_apply(self, vecs: Array) -> Array:
        raise NotImplementedError("right_apply is not implemented")

    def right_apply_implemented(self) -> bool:
        return False

    @abstractmethod
    def nrows(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def ncols(self) -> int:
        raise NotImplementedError


class SymmetricMatVecOperator(MatVecOperator):
    def apply_transpose(self, vecs: Array) -> Array:
        return self.apply(vecs)

    def ncols(self) -> int:
        return self.nrows()


class DenseMatVecOperator(MatVecOperator):
    def __init__(self, mat: Array, backend: BackendMixin = NumpyMixin):
        self._mat = mat
        self._bkd = backend

    def apply(self, vecs: Array) -> Array:
        return self._mat @ vecs

    def apply_transpose(self, vecs: Array) -> Array:
        return self._mat.T @ vecs

    def right_apply(self, vecs: Array) -> Array:
        return vecs @ self._mat

    def right_apply_implemented(self) -> bool:
        return True

    def nrows(self) -> int:
        return self._mat.shape[0]

    def ncols(self) -> int:
        return self._mat.shape[1]


class DenseSymmetricMatVecOperator(SymmetricMatVecOperator):
    def __init__(self, mat: Array, backend: BackendMixin = NumpyMixin):
        if mat.shape[0] != mat.shape[1] or not backend.allclose(
            mat, mat.T, atol=1e-16
        ):
            raise ValueError("matrix must be symmetric")
        self._mat = mat
        self._bkd = backend

    def apply(self, vecs: Array) -> Array:
        return self._mat @ vecs

    def nrows(self) -> int:
        return self._mat.shape[0]

    def right_apply_implemented(self) -> bool:
        return True

    def right_apply(self, vecs: Array) -> Array:
        return self.apply(vecs)


def adjust_sign_svd(
    U: Array, V: Array, adjust_based_upon_U: bool = True
) -> Tuple[Array, Array]:
    r"""
    Ensure uniquness of svd by ensuring the first entry of each left singular
    singular vector be positive. Only works for np.linalg.svd
    if full_matrices=False

    Parameters
    ----------
    U : (M x M) matrix
        left singular vectors of a singular value decomposition of a (M x N)
        matrix A.

    V : (N x N) matrix
        right singular vectors of a singular value decomposition of a (M x N)
        matrix A.

    adjust_based_upon_U : boolean (default=True)
        True - make the first entry of each column of U positive
        False - make the first entry of each row of V positive

    Returns
    -------
    U : (M x M) matrix
       left singular vectors with first entry of the first
       singular vector always being positive.

    V : (M x M) matrix
        right singular vectors consistent with sign adjustment applied to U.
    """
    if U.shape[1] != V.shape[0]:
        msg = "U.shape[1] must equal V.shape[0]. If using np.linalg.svd set "
        msg += "full_matrices=False"
        raise ValueError(msg)

    if adjust_based_upon_U:
        s = np.sign(U[0, :])
    else:
        s = np.sign(V[:, 0])
    U *= s
    V *= s[:, np.newaxis]
    return U, V


class RandomizedSVD(ABC):
    def __init__(
        self,
        matvec: MatVecOperator,
        noversampling: int = 10,
        npower_iters: int = 1,
    ):
        self._check_matvec(matvec)
        self._bkd = matvec._bkd
        self._matvec = matvec
        self._noversampling = noversampling
        self._npower_iters = npower_iters

    def _check_matvec(self, matvec: MatVecOperator):
        if not isinstance(matvec, MatVecOperator):
            raise ValueError("matvec must be an instance of MatVecOperator")

    @abstractmethod
    def compute(self, rank: int) -> Tuple[Array, Array, Array]:
        raise NotImplementedError

    def adjust_sign(self, U: Array, Vh: Array) -> Tuple[Array, Array]:
        return adjust_sign_svd(U, Vh)

    def _sample_column_space(self, rank):
        nsamples = rank + self._noversampling
        # use transpose so omega samples are nested if nsamples are increased
        omega = self._bkd.asarray(
            np.random.normal(0, 1, (nsamples, self._matvec.ncols()))
        ).T
        # sample column space
        Y = self._matvec.apply(omega)
        for ii in range(self._npower_iters):
            G = self._matvec.apply_transpose(Y)
            Y = self._matvec.apply(G)
        return Y


class SinglePassRandomizedSVD(RandomizedSVD):
    def compute(self, rank: int) -> Tuple[Array, Array, Array]:
        if not self._matvec.right_apply_implemented():
            raise ValueError("matvec must implement right_apply")
        cspace_samples = self._sample_column_space(rank)
        # ortogonalize column space samples
        Q = self._bkd.qr(cspace_samples, mode="reduced")[0]
        B = self._matvec.right_apply(Q.T)
        U, S, Vh = self._bkd.svd(B)
        U = Q @ U
        U, Vh = self.adjust_sign(U[:, :rank], Vh[:rank])
        return U, S[:rank], Vh


class SymmetricMatrixDoublePassRandomizedSVD(RandomizedSVD):
    def _check_matvec(self, matvec: SymmetricMatVecOperator):
        if not isinstance(matvec, SymmetricMatVecOperator):
            raise ValueError(
                "matvec must be an instance of SymmetricMatVecOperator"
            )

    def compute(self, rank: int) -> Tuple[Array, Array, Array]:
        # first pass
        cspace_samples = self._sample_column_space(rank)
        Q1 = self._bkd.qr(cspace_samples, mode="reduced")[0]
        # second pass
        rspace_samples = self._matvec.apply_transpose(Q1)
        Q2 = self._bkd.qr(rspace_samples, mode="reduced")[0]
        # svd of compressed row space samples
        U, S, Vh = self._bkd.svd(Q2.T @ rspace_samples)
        # Project row space
        Vh = ((Q1 @ Vh.T).T)[:rank]
        # Project column space
        U = (Q2 @ U)[:, :rank]
        U, Vh = self.adjust_sign(U, Vh)
        return U, S[:rank], Vh


def get_low_rank_matrix(
    nrows: int,
    ncols: int,
    rank: int,
    bkd: BackendMixin = NumpyMixin,
) -> Array:
    r"""
    Construct a matrix of size nrows x ncols with a given rank.

    Parameters
    ----------
    nrows : integer
        The number rows in the matrix

    ncols : integer
        The number columns in the matrix

    rank : integer
        The rank of the matrix

    Returns
    -------
    Amatrix : np.ndarray (nrows,ncols)
        The low-rank matrix generated
    """
    assert rank <= min(nrows, ncols)
    # Generate a matrix with normally distributed entries
    N = max(nrows, ncols)
    Amatrix = bkd.asarray(np.random.normal(0, 1, (N, N)))
    # Make A symmetric positive definite
    Amatrix = Amatrix.T @ Amatrix
    # Construct low rank approximation of A
    eigvals, eigvecs = bkd.eigh(bkd.copy(Amatrix))
    # Set smallest eigenvalues to zero. Note eigenvals are in
    # ascending order
    eigvals[: (eigvals.shape[0] - rank)] = 0.0
    # Construct rank r A matrix
    Amatrix = bkd.multidot((eigvecs, bkd.diag(eigvals), eigvecs.T))
    # Resize matrix to have requested size
    Amatrix = Amatrix[:nrows, :ncols]
    return Amatrix


def invert_permutation_vector(
    p: Array, dtype=int, bkd: BackendMixin = NumpyMixin
) -> Array:
    r"""
    Returns the "inverse" of a permutation vector. I.e., returns the
    permutation vector that performs the inverse of the original
    permutation operation.

    Parameters
    ----------
    p: np.ndarray
        Permutation vector
    dtype: type
        Data type passed to np.ndarray constructor

    Returns
    -------
    pt: np.ndarray
        Permutation vector that accomplishes the inverse of the
        permutation p.
    """

    N = bkd.max(p) + 1
    pt = bkd.zeros(p.shape[0], dtype=dtype)
    pt[p] = bkd.arange(N, dtype=dtype)
    return pt


def trace_of_mat_mat_product(
    Amat: Array, Bmat: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
    """
    Compute Trace(A @ B)
    """
    assert Amat.shape == Bmat.T.shape
    # use einsum because unlike other approaches, e.g. np.sum(A*B.T)
    # it does not use any explicit intermediate storage
    return bkd.einsum("ij,ji->", Amat, Bmat)


def diag_of_mat_mat_product(
    Amat: Array, Bmat: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
    """
    Compute Diag(A @ B)
    """
    assert Amat.shape == Bmat.T.shape
    # use einsum because unlike other approaches, e.g. np.diag(A@B)
    # it does not use any explicit intermediate storage
    return bkd.einsum("ij,ji->i", Amat, Bmat)


def equality_constrained_linear_least_squares(
    A: Array, B: Array, y: Array, z: Array
):
    """
    Solve equality constrained least squares regression

    minimize || y - A*x ||_2   subject to   B*x = z

    It is assumed that

    Parameters
    ----------
    A : np.ndarray (M, N)
        P <= N <= M+P, and

    B : np.ndarray (N, P)
        P <= N <= M+P, and

    y : np.ndarray (M, 1)
        P <= N <= M+P, and

    z : np.ndarray (P, 1)
        P <= N <= M+P, and

    Returns
    -------
    x : np.ndarray (N, 1)
        The solution
    """
    return scipy.linalg.lapack.dgglse(A, B, y, z)[3]


def log_determinant_triangular_matrix(
    matrix: Array, bkd: BackendMixin = NumpyMixin
):
    return bkd.sum(bkd.log(bkd.diag(matrix)))


def log_determinant_from_cholesky_factor(
    L: Array, bkd: BackendMixin = NumpyMixin
):
    """Get determinant of LL@.T"""
    return 2.0 * log_determinant_triangular_matrix(L, bkd)


def inverse_from_cholesky_factor(L: Array, bkd: BackendMixin = NumpyMixin):
    """
    Inverse of a matrix using its cholesky factor
    I.e. compute inv(A) A =L@L.T
    """
    rhs = bkd.eye(L.shape[0])
    A_inv = bkd.cholesky_solve(L, rhs, lower=True)
    return A_inv


def inverse_of_cholesky_factor(L: Array, bkd: BackendMixin = NumpyMixin):
    """Compute inveres of cholesy factor"""
    rhs = bkd.eye(L.shape[0])
    L_inv = bkd.solve_triangular(L, rhs, lower=True)
    return L_inv


def cholesky_solve_linear_system(
    L: Array, rhs: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
    r"""
    Solve LL'x = b using forwards and backwards substitution
    """
    # Use forward subsitution to solve Ly = b
    y = bkd.solve_triangular(L, rhs, lower=True)
    # Use backwards subsitution to solve L'x = y
    x = bkd.solve_triangular(L.T, y, lower=False)
    return x


def update_cholesky_factorization_inverse(
    L_11_inv, L_12, L_22, bkd: BackendMixin = NumpyMixin
) -> Array:
    nrows, ncols = L_12.shape
    L_22_inv = bkd.inv(L_22)
    L_inv = bkd.block(
        [
            [L_11_inv, bkd.zeros((nrows, ncols))],
            [-L_22_inv @ (L_12.T @ L_11_inv), L_22_inv],
        ]
    )
    return L_inv


def update_trace_involving_cholesky_inverse(
    L_11_inv, L_12, L_22_inv, B, prev_trace, bkd: BackendMixin = NumpyMixin
) -> Array:
    r"""
    Update the trace of matrix matrix product involving the inverse of a
    matrix with a cholesky factorization.

    That is compute

    .. math:: \mathrm{Trace}\leftA^{inv}B\right}

    where :math:`A=LL^T`
    """
    nrows, ncols = L_12.shape
    assert B.shape == (nrows + ncols, nrows + ncols)
    B_11 = B[:nrows, :nrows]
    B_12 = B[:nrows, nrows:]
    B_21 = B[nrows:, :nrows]
    B_22 = B[nrows:, nrows:]
    # assert bkd.allclose(B, bkd.block([[B_11, B_12],[B_21, B_22]]))

    C = -(L_22_inv @ L_12.T) @ L_11_inv
    C_T_L_22_inv = C.T @ L_22_inv
    trace = (
        prev_trace
        + bkd.sum(C.T @ C * B_11)
        + bkd.sum(C_T_L_22_inv * B_12)
        + bkd.sum(C_T_L_22_inv.T * B_21)
        + bkd.sum((L_22_inv.T @ L_22_inv) * B_22)
    )
    return trace


def qr_solve(
    Q: Array, R: Array, rhs: Array, bkd: BackendMixin = NumpyMixin
) -> Array:
    """
    Find the least squares solution Ax = rhs given a QR factorization of the
    matrix A

    Parameters
    ----------
    Q : Array (nrows, nrows)
        The unitary/upper triangular Q factor

    R : Array (nrows, ncols)
        The upper triangular R matrix

    rhs : Array (nrows, nqoi)
        The right hand side vectors

    Returns
    -------
    x : np.ndarray (nrows, nqoi)
        The solution
    """
    return bkd.solve_triangular(R, Q.T @ rhs, lower=False)


def nentries_square_triangular_matrix(
    N: int, include_diagonal: bool = True
) -> int:
    r"""Num entries in upper (or lower) NxN traingular matrix"""
    if include_diagonal:
        return int(N * (N + 1) / 2)
    else:
        return int(N * (N - 1) / 2)


def nentries_rectangular_triangular_matrix(
    M: int, N: int, upper: bool = True
) -> int:
    r"""Num entries in upper (or lower) MxN traingular matrix.
    This is useful for nested for loops like

    (upper=True)

    for ii in range(M):
        for jj in range(ii+1):

    (upper=False)

    for jj in range(N):
        for ii in range(jj+1):

    """
    assert M >= N
    if upper:
        return nentries_square_triangular_matrix(N)
    else:
        return nentries_square_triangular_matrix(
            M
        ) - nentries_square_triangular_matrix(M - N)


def flattened_rectangular_lower_triangular_matrix_index(
    ii: int, jj: int, M: int, N: int
) -> int:
    r"""
    Get flattened index kk from row and column indices (ii,jj) of a
    lower triangular part of MxN matrix
    """
    assert M >= N
    assert ii >= jj
    if ii == 0:
        return 0
    T = nentries_rectangular_triangular_matrix(ii, min(ii, N), upper=False)
    kk = T + jj
    return kk
