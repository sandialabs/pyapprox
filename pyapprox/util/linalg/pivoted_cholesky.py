"""Backend-agnostic pivoted Cholesky factorization."""

from typing import Generic

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def _swap_rows(matrix: Array, ii: int, jj: int, bkd: Backend[Array]) -> None:
    """Swap rows ii and jj of a 1D array in place."""
    indices = bkd.asarray([ii, jj], dtype=bkd.int64_dtype())
    swap_indices = bkd.asarray([jj, ii], dtype=bkd.int64_dtype())
    matrix[indices] = bkd.copy(matrix[swap_indices])


class PivotedCholeskyFactorizer(Generic[Array]):
    """Backend-agnostic pivoted Cholesky factorization.

    Computes a low-rank pivoted Cholesky decomposition of a symmetric
    positive semi-definite matrix A. If A is positive definite and
    npivots equals the number of rows, then L @ L.T == A (up to
    permutation).

    Parameters
    ----------
    K : Array
        Symmetric positive semi-definite matrix to factorize.
    bkd : Backend[Array]
        Backend for numerical computations.
    econ : bool
        If True (default), use diagonal for pivot selection.
        If False, use Schur complement norm.
    tol : float
        Tolerance for early termination based on relative residual.
    """

    def __init__(
        self,
        K: Array,
        bkd: Backend[Array],
        econ: bool = True,
        tol: float = 1e-14,
    ) -> None:
        self._Amat = bkd.copy(K)
        self._bkd = bkd
        self._econ = econ
        self._tol = tol
        self._nrows = K.shape[0]
        self._ncompleted_pivots = 0
        self._termination_flag = -1
        self._termination_msg = ""
        self._L: Array | None = None
        self._pivots_arr: Array | None = None
        self._diag: Array | None = None
        self._init_pivots: Array | None = None
        self._pivot_weights: Array | None = None
        self._init_error: Array | None = None

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def factorize(
        self,
        npivots: int,
        init_pivots: Array | None = None,
        pivot_weights: Array | None = None,
    ) -> None:
        """Perform pivoted Cholesky factorization.

        Parameters
        ----------
        npivots : int
            Number of pivots to compute.
        init_pivots : Array | None
            Initial pivots to enforce at the beginning of factorization.
        pivot_weights : Array | None
            Weights for pivot selection (econ mode only).
        """
        bkd = self._bkd
        if npivots > self._nrows:
            raise ValueError("Number of pivots requested exceeds number of matrix rows")
        self._init_pivots = init_pivots
        self._pivot_weights = pivot_weights
        self._ncompleted_pivots = 0
        self._L = bkd.zeros((self._nrows, self._nrows))
        self._pivots_arr = bkd.arange(self._nrows)
        self._diag = bkd.ravel(self._Amat)[:: self._Amat.shape[0] + 1]
        self._init_error = bkd.sum(bkd.abs(self._diag))
        self._update_loop(npivots)

    def update(self, npivots: int) -> None:
        """Continue factorization from current state.

        Parameters
        ----------
        npivots : int
            Total number of pivots to reach.
        """
        if self._L is None:
            raise RuntimeError("Must call factorize() before update()")
        if self._ncompleted_pivots >= npivots:
            raise ValueError("Already have enough pivots")
        self._update_loop(npivots)

    def _update_loop(self, npivots: int) -> None:
        bkd = self._bkd
        Amat = bkd.copy(self._Amat)
        assert self._L is not None
        assert self._pivots_arr is not None
        assert self._diag is not None
        assert self._init_error is not None

        if not self._econ and self._pivot_weights is not None:
            raise ValueError("pivot weights not used when econ is False")

        for ii in range(self._ncompleted_pivots, npivots):
            pivot = self._compute_pivot(Amat, ii)
            _swap_rows(self._pivots_arr, ii, pivot, bkd)

            if self._diag[self._pivots_arr[ii]] <= 1e-14:
                self._termination_flag = 3
                raise ValueError("Matrix is not positive semi-definite")

            self._L[self._pivots_arr[ii], ii] = bkd.sqrt(
                self._diag[self._pivots_arr[ii]]
            )

            self._L[self._pivots_arr[ii + 1 :], ii] = (
                Amat[self._pivots_arr[ii + 1 :], self._pivots_arr[ii]]
                - self._L[self._pivots_arr[ii + 1 :], :ii]
                @ self._L[self._pivots_arr[ii], :ii]
            ) / self._L[self._pivots_arr[ii], ii]

            self._diag[self._pivots_arr[ii + 1 :]] -= (
                self._L[self._pivots_arr[ii + 1 :], ii] ** 2
            )

            self._ncompleted_pivots += 1
            rel_error = (
                bkd.sum(self._diag[self._pivots_arr[ii + 1 :]]) / self._init_error
            )
            if rel_error < self._tol:
                self._termination_flag = 1
                self._termination_msg = "Tolerance reached"
                return

        self._termination_flag = 0
        self._termination_msg = "Factorization completed successfully"

    def _compute_pivot(self, Amat: Array, ii: int) -> int:
        bkd = self._bkd
        assert self._pivots_arr is not None
        assert self._diag is not None

        if self._init_pivots is not None and ii < len(self._init_pivots):
            return int(
                bkd.to_numpy(
                    bkd.where(self._pivots_arr == self._init_pivots[ii])[0][0:1]
                )[0]
            )

        if self._econ:
            if self._pivot_weights is None:
                return int(
                    bkd.to_numpy(
                        bkd.reshape(
                            bkd.argmax(self._diag[self._pivots_arr[ii:]]) + ii, (1,)
                        )
                    )[0]
                )
            return int(
                bkd.to_numpy(
                    bkd.reshape(
                        bkd.argmax(
                            self._pivot_weights[self._pivots_arr[ii:]]
                            * self._diag[self._pivots_arr[ii:]]
                        )
                        + ii,
                        (1,),
                    )
                )[0]
            )

        assert self._L is not None
        schur_complement = (
            Amat[
                np.ix_(
                    bkd.to_numpy(self._pivots_arr[ii:]),
                    bkd.to_numpy(self._pivots_arr[ii:]),
                )
            ]
            - self._L[self._pivots_arr[ii:], :ii]
            @ self._L[self._pivots_arr[ii:], :ii].T
        )
        schur_diag = bkd.diag(schur_complement)
        pivot = int(
            bkd.to_numpy(
                bkd.reshape(
                    bkd.argmax(bkd.norm(schur_complement, axis=0) ** 2 / schur_diag)
                    + ii,
                    (1,),
                )
            )[0]
        )
        return pivot

    def pivots(self) -> Array:
        """Return pivot indices."""
        if self._pivots_arr is None:
            raise RuntimeError("Must call factorize() first")
        return self._pivots_arr[: self._ncompleted_pivots]

    def npivots(self) -> int:
        """Return number of completed pivots."""
        return self._ncompleted_pivots

    def factor(self) -> Array:
        """Return the L factor of shape (nrows, npivots)."""
        if self._L is None:
            raise RuntimeError("Must call factorize() first")
        return self._L[:, : self._ncompleted_pivots]

    def success(self) -> bool:
        """Return True if factorization completed without errors."""
        return self._termination_flag < 2
