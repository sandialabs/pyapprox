import numpy as np

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


def qr_solve(
    Q: Array, R: Array, rhs: Array, bkd: LinAlgMixin = NumpyLinAlgMixin
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
    matrix: Array, ii: int, jj: int, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Array:
    matrix[bkd.hstack([ii, jj])] = matrix[bkd.hstack([jj, ii])]


def get_pivot_matrix_from_vector(
    pivots: Array, nrows: int, bkd: LinAlgMixin = NumpyLinAlgMixin
) -> Array:
    P = bkd.eye(nrows)
    P = P[pivots, :]
    return P


class PivotedCholeskyFactorizer:
    def __init__(
        self,
        Amat: Array,
        tol: float = 0.0,
        econ: bool = True,
        bkd: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._Amat = bkd.copy(Amat)
        self._tol = tol
        self._econ = econ
        self._bkd = bkd

        self._nrows = self._Amat.shape[0]
        assert self._Amat.shape[1] == self._nrows
        self._pivots = self._bkd.arange(self._nrows)

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
            self._chol_flag = 3
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
        self._chol_flag = 0
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
            self._chol_flag = 1
            self._termination_msg = "Tolerance reached. "
            f"Iteration:{ii}. Tol={self._tol}. Rel. Error={rel_error}"
            return True
        self._termination_msg = "Negative error. Should not happen because of"
        " checks is_positive_definite"
        raise RuntimeError(self._termination_msg)

    def pivots(self) -> Array:
        return self._pivots[: self._ncompleted_pivots]

    def npivots(self) -> int:
        return self._ncompleted_pivots

    def termination_message(self) -> str:
        return self._termination_msg

    def success(self) -> bool:
        return self._chol_flag < 2

    def __repr__(self) -> str:
        if self.npivots() == 0:
            return "{0}(nrows={1})".format(
                self.__class__.__name__, self._nrows
            )
        return "{0}(nrows={1}, npivots={2}, msg={3})".format(
            self.__class__.__name__,
            self._nrows,
            self.npivots(),
            self.termination_message(),
        )
