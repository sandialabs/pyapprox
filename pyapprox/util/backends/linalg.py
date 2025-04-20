from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin


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


def swap_rows(
    matrix: Array, ii: int, jj: int, bkd: BackendMixin = NumpyMixin
) -> Array:
    matrix[bkd.hstack([ii, jj])] = matrix[bkd.hstack([jj, ii])]


def pivot_rows(pivots, matrix, in_place=True):
    if not in_place:
        matrix = matrix.copy()
    num_pivots = pivots.shape[0]
    assert num_pivots <= matrix.shape[0]
    for ii in range(num_pivots):
        swap_rows(matrix, ii, pivots[ii])
    return matrix


def get_pivot_matrix_from_vector(
    pivots: Array, nrows: int, bkd: BackendMixin = NumpyMixin
) -> Array:
    P = bkd.eye(nrows)
    P = P[pivots, :]
    return P


class PivotedFactorizer(ABC):
    def __init__(
        self,
        Amat: Array,
        tol: float = 0.0,
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
            self._bkd.ix_(self._pivots[ii:], self._pivots[ii:])
        ] - self._L[self._pivots[ii:], :ii].dot(
            self._L[self._pivots[ii:], :ii].T
        )
        schur_diag = self._bkd.diagonal(schur_complement)
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
        if self._bkd.abs(rel_error) < 2 * np.finfo(float).eps:
            # If matrix is rank r then then error will be machine precision
            # In such a case exiting without an error is the right thing to
            # do
            self._termination_flag = 1
            self._termination_msg = "Tolerance reached. "
            f"Iteration:{ii}. Tol={self._tol}. Rel. Error={rel_error}"
            return True
        self._termination_msg = "Negative error. Should not happen because of"
        " checks is_positive_definite"
        raise RuntimeError(self._termination_msg)


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

    def _split_lu(self, npivots: int) -> Tuple[Array, Array]:
        r"""
        Return the L and U factors of an inplace LU factorization

        Parameters
        ----------
        npivots : integer
            The number of pivots performed. This allows LU in place matrix
            to be split during evolution of LU algorithm
        """
        if npivots is None:
            npivots = min(*self._LU_factor.shape)
        L_factor = self._bkd.tril(self._LU_factor)
        if L_factor.shape[1] < L_factor.shape[0]:
            # if matrix over-determined ensure L is a square matrix
            n0 = L_factor.shape[0] - L_factor.shape[1]
            L_factor = self._bkd.hstack(
                [L_factor, np.zeros((L_factor.shape[0], n0))]
            )
        # np.fill_diagonal(L_factor, 1.0)
        for ii in range(L_factor.shape[0]):
            L_factor[ii, ii] = 1.0
        U_factor = self._bkd.triu(self._LU_factor)
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
        self._raw_pivots = self._bkd.arange(self._Amat.shape[0])
        for it in range(npivots):
            pivot = self._best_pivot(it)
            # update pivots vector
            self._raw_pivots[it] = pivot
            # apply pivots(swap rows) in L factorization
            swap_rows(self._LU_factor, it, pivot)
            if self._terminate(it):
                self._termination_flag = 1
                return self._split_lu(it)
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
        print(self._ncompleted_pivots)
        return self._split_lu(self._ncompleted_pivots)

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
            self._raw_pivots[: self._ncompleted_pivots]
        ):
            # inlined swap_rows() for performance
            new_cols[[it, pivot]] = new_cols[[pivot, it]]

            # update LU_factor
            # recover state of col vector from permuted LU factor
            # Let  (jj,kk) represent iteration and pivot pairs
            # then if lu factorization produced sequence of pairs
            # (0,4),(1,2),(2,4) then LU_factor[:,0] here will be col_vector
            # in LU algorithm with the second and third permutations
            # so undo these permutations in reverse order
            next_idx = it + 1

            # `col_vector` is a copy of the LU_factor subset
            print(next_idx, it, self._LU_factor.shape)
            col_vector = self._bkd.copy(self._LU_factor[next_idx:, it])
            for ii in range(self._ncompleted_pivots - it - 1):
                # (it+1) necessary in two lines below because only dealing
                # with compressed col vector which starts at row it in
                # LU_factor
                jj = (
                    self._raw_pivots[self._ncompleted_pivots - 1 - ii]
                    - next_idx
                )
                kk = self._ncompleted_pivots - ii - 1 - next_idx

                # inlined swap_rows()
                col_vector[jj], col_vector[kk] = col_vector[kk], col_vector[jj]

            new_cols[next_idx:, :] -= np.outer(col_vector, new_cols[it, :])
        self._LU_factor = self._bkd.hstack((self._LU_factor, new_cols))
