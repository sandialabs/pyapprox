"""Pivoted LU factorization for Leja point selection.

This module provides incremental LU factorization with partial pivoting,
enabling efficient sequential Leja point selection by maintaining
factorization state across column/row additions.
"""

from typing import Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend


def swap_rows(matrix: Array, ii: int, jj: int, bkd: Backend[Array]) -> None:
    """Swap rows ii and jj in matrix (in-place)."""
    indices = bkd.asarray([ii, jj], dtype=bkd.int64_dtype())
    swap_indices = bkd.asarray([jj, ii], dtype=bkd.int64_dtype())
    matrix[indices] = bkd.copy(matrix[swap_indices])


def get_final_pivots_from_sequential_pivots(
    sequential_pivots: Array,
    npivots: int,
    bkd: Backend[Array],
) -> Array:
    """Convert sequential pivots to final permutation.

    Parameters
    ----------
    sequential_pivots : Array
        Pivot vector obtained by inserting pivot at each iteration.
    npivots : int
        Total number of elements in permutation.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        The vector that changes the original array to the final permuted
        array in one shot.
    """
    pivots = bkd.arange(npivots, dtype=bkd.int64_dtype())
    for ii in range(sequential_pivots.shape[0]):
        pivot = int(sequential_pivots[ii])
        # swap
        tmp = int(pivots[ii])
        pivots[ii] = pivots[pivot]
        pivots[pivot] = tmp
    return pivots


class PivotedLUFactorizer(Generic[Array]):
    """Incremental LU factorization with partial pivoting.

    Enables efficient sequential Leja point selection by maintaining
    factorization state across column additions.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    matrix : Array
        Initial matrix to factorize. Shape: (nrows, ncols)
    tol : float
        Tolerance for detecting singular pivots.
    init_pivots : Array, optional
        Initial pivot indices to use before choosing based on magnitude.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> A = bkd.asarray([[4.0, 3.0], [6.0, 3.0]])
    >>> factorizer = PivotedLUFactorizer(bkd, A)
    >>> L, U = factorizer.factorize(2)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        matrix: Array,
        tol: float = 1e-14,
        init_pivots: Optional[Array] = None,
    ):
        self._bkd = bkd
        self._Amat = bkd.copy(matrix)
        self._tol = tol
        self._init_pivots = init_pivots

        self._nrows = self._Amat.shape[0]
        self._ncols = self._Amat.shape[1] if self._Amat.ndim > 1 else 1
        self._pivots = self._bkd.arange(self._nrows, dtype=bkd.int64_dtype())
        self._seq_pivots = self._bkd.arange(self._nrows, dtype=bkd.int64_dtype())
        self._ncompleted_pivots = 0
        self._LU_factor: Optional[Array] = None
        self._termination_flag = -1
        self._termination_msg = ""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def npivots(self) -> int:
        """Return number of completed pivots."""
        return self._ncompleted_pivots

    def pivots(self) -> Array:
        """Return pivot indices (up to completed pivots)."""
        return self._pivots[: self._ncompleted_pivots]

    def success(self) -> bool:
        """Return True if factorization completed successfully."""
        return self._termination_flag == 0

    def termination_message(self) -> str:
        """Return termination status message."""
        return self._termination_msg

    def _best_pivot(self, it: int) -> int:
        """Select best pivot for iteration it."""
        if self._init_pivots is not None and it < len(self._init_pivots):
            # Find where the requested pivot is in current pivot order
            target = int(self._init_pivots[it])
            for idx in range(it, self._nrows):
                if int(self._pivots[idx]) == target:
                    return idx
            return it  # fallback
        else:
            # Choose pivot with maximum magnitude
            return int(self._bkd.argmax(self._bkd.abs(self._LU_factor[it:, it]))) + it

    def _terminate(self, it: int) -> bool:
        """Check if factorization should terminate at iteration it."""
        pivot_val = float(self._bkd.abs(self._LU_factor[it, it]))
        if pivot_val < self._tol:
            self._termination_msg = (
                f"pivot {pivot_val:.2e} is too small. Stopping factorization."
            )
            return True
        return False

    def _split_lu(self, npivots: int) -> Tuple[Array, Array]:
        """Split LU factor into L and U matrices.

        Parameters
        ----------
        npivots : int
            Number of pivots performed.

        Returns
        -------
        Tuple[Array, Array]
            L factor (npivots x npivots), U factor (npivots x npivots)
        """
        if npivots is None:
            npivots = min(self._nrows, self._ncols)

        L_factor = self._bkd.tril(self._LU_factor[:npivots, :npivots])
        # Set diagonal to 1
        for ii in range(npivots):
            L_factor[ii, ii] = 1.0

        U_factor = self._bkd.triu(self._LU_factor[:npivots, :npivots])
        return L_factor, U_factor

    def factorize(
        self,
        npivots: int,
        init_pivots: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Perform LU factorization with given number of pivots.

        Parameters
        ----------
        npivots : int
            Number of pivots to perform.
        init_pivots : Array, optional
            Initial pivot indices to use.

        Returns
        -------
        Tuple[Array, Array]
            L and U factors.
        """
        if init_pivots is not None:
            self._init_pivots = init_pivots
        self._ncompleted_pivots = 0
        self._LU_factor = self._bkd.copy(self._Amat)
        self._seq_pivots = self._bkd.arange(self._nrows, dtype=self._bkd.int64_dtype())
        self._pivots = self._bkd.arange(self._nrows, dtype=self._bkd.int64_dtype())
        return self.update(npivots)

    def update(self, npivots: int) -> Tuple[Array, Array]:
        """Continue factorization to given number of pivots.

        Parameters
        ----------
        npivots : int
            Target number of pivots.

        Returns
        -------
        Tuple[Array, Array]
            L and U factors.
        """
        # Initialize LU factor if not yet set
        if self._LU_factor is None:
            self._LU_factor = self._bkd.copy(self._Amat)

        npivots = min(npivots, min(self._nrows, self._ncols))

        for it in range(self._ncompleted_pivots, npivots):
            pivot = self._best_pivot(it)

            # Record sequential pivot
            self._seq_pivots[it] = pivot

            # Swap rows in LU factor
            swap_rows(self._LU_factor, it, pivot, self._bkd)
            swap_rows(self._pivots, it, pivot, self._bkd)

            if self._terminate(it):
                self._termination_flag = 1
                return self._split_lu(it)

            # Update L factor (column below diagonal)
            self._LU_factor[it + 1 :, it] /= self._LU_factor[it, it]

            # Update U factor (Schur complement)
            col_vector = self._LU_factor[it + 1 :, it : it + 1]
            row_vector = self._LU_factor[it : it + 1, it + 1 :]
            self._LU_factor[it + 1 :, it + 1 :] -= col_vector @ row_vector

            self._ncompleted_pivots += 1

        self._termination_msg = "Factorization completed successfully"
        self._termination_flag = 0
        return self._split_lu(self._ncompleted_pivots)

    def add_rows(self, new_rows: Array) -> None:
        """Add rows to the matrix and update factorization.

        Parameters
        ----------
        new_rows : Array
            New rows to add. Shape: (nnew, ncols)
        """
        if self._LU_factor.shape[1] != new_rows.shape[1]:
            raise ValueError("new_rows has the wrong number of columns")

        nnew = new_rows.shape[0]
        self._seq_pivots = self._bkd.concatenate(
            [
                self._seq_pivots,
                self._bkd.arange(
                    self._nrows, self._nrows + nnew, dtype=self._bkd.int64_dtype()
                ),
            ]
        )
        self._pivots = self._bkd.concatenate(
            [
                self._pivots,
                self._bkd.arange(
                    self._nrows, self._nrows + nnew, dtype=self._bkd.int64_dtype()
                ),
            ]
        )
        self._Amat = self._bkd.concatenate([self._Amat, new_rows], axis=0)

        # Update LU factor for new rows
        LU_extra = self._bkd.copy(new_rows)
        for it in range(self._ncompleted_pivots):
            LU_extra[:, it] /= self._LU_factor[it, it]
            col_vector = LU_extra[:, it : it + 1]
            row_vector = self._LU_factor[it : it + 1, it + 1 :]
            LU_extra[:, it + 1 :] -= col_vector @ row_vector

        self._LU_factor = self._bkd.concatenate([self._LU_factor, LU_extra], axis=0)
        self._nrows += nnew

    def add_columns(self, new_cols: Array) -> None:
        """Add columns to the matrix and update factorization.

        Parameters
        ----------
        new_cols : Array
            New columns to add. Shape: (nrows, nnew)
        """
        if self._LU_factor.shape[0] != new_cols.shape[0]:
            raise ValueError("new_cols has the wrong number of rows")

        self._Amat = self._bkd.concatenate([self._Amat, new_cols], axis=1)

        # Apply existing row permutations to new columns
        new_cols = self._bkd.copy(new_cols)
        for it in range(self._ncompleted_pivots):
            pivot = int(self._seq_pivots[it])
            swap_rows(new_cols, it, pivot, self._bkd)

            # Update LU factor for new columns
            next_idx = it + 1
            col_vector = self._bkd.copy(self._LU_factor[next_idx:, it : it + 1])

            # Undo permutations in reverse order
            for ii in range(self._ncompleted_pivots - it - 1):
                jj = int(self._seq_pivots[self._ncompleted_pivots - 1 - ii]) - next_idx
                kk = self._ncompleted_pivots - ii - 1 - next_idx
                if 0 <= jj < col_vector.shape[0] and 0 <= kk < col_vector.shape[0]:
                    swap_rows(col_vector, jj, kk, self._bkd)

            new_cols[next_idx:, :] -= col_vector @ new_cols[it : it + 1, :]

        self._LU_factor = self._bkd.concatenate([self._LU_factor, new_cols], axis=1)
        self._ncols += new_cols.shape[1]

    def undo_preconditioning(
        self,
        precond_weights: Array,
        npivots: Optional[int] = None,
        update_internal_state: bool = False,
    ) -> Tuple[Array, Array]:
        """Remove preconditioning from factorization.

        If the matrix was preconditioned as W*A, this recovers A's factors.

        Parameters
        ----------
        precond_weights : Array
            The preconditioning weights (column vector).
        npivots : int, optional
            Number of pivots to use.
        update_internal_state : bool
            If True, update internal LU factor.

        Returns
        -------
        Tuple[Array, Array]
            Unpreconditioned L and U factors.
        """
        if npivots is None:
            npivots = min(self._nrows, self._ncols)

        # Get pivoted weights
        final_pivots = get_final_pivots_from_sequential_pivots(
            self._seq_pivots[:npivots], self._nrows, self._bkd
        )
        pivoted_weights = precond_weights[final_pivots]

        # Undo preconditioning: inv(W) @ L and inv(W) @ U
        LU_factor = self._bkd.copy(self._LU_factor) / pivoted_weights

        # Right multiply L by W
        for ii in range(npivots):
            LU_factor[ii + 1 :, ii] *= float(pivoted_weights[ii, 0])

        if update_internal_state:
            self._LU_factor = LU_factor

        # Split and return
        L_factor = self._bkd.tril(LU_factor[:npivots, :npivots])
        for ii in range(npivots):
            L_factor[ii, ii] = 1.0
        U_factor = self._bkd.triu(LU_factor[:npivots, :npivots])
        return L_factor, U_factor

    def update_preconditioning(
        self,
        prev_weights: Array,
        new_weights: Array,
        npivots: Optional[int] = None,
        update_internal_state: bool = False,
    ) -> Tuple[Array, Array]:
        """Update preconditioning from prev_weights to new_weights.

        Parameters
        ----------
        prev_weights : Array
            Previous preconditioning weights.
        new_weights : Array
            New preconditioning weights.
        npivots : int, optional
            Number of pivots.
        update_internal_state : bool
            If True, update internal state.

        Returns
        -------
        Tuple[Array, Array]
            Updated L and U factors.
        """
        return self.undo_preconditioning(
            prev_weights / new_weights, npivots, update_internal_state
        )

    def __repr__(self) -> str:
        if self.npivots() == 0:
            return f"PivotedLUFactorizer(nrows={self._nrows}, ncols={self._ncols})"
        return (
            f"PivotedLUFactorizer(nrows={self._nrows}, ncols={self._ncols}, "
            f"npivots={self.npivots()}, msg={self.termination_message()})"
        )
