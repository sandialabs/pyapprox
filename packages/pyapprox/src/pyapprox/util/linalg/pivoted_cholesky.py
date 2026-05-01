"""Backend-agnostic pivoted Cholesky factorization.

Accepts either a dense matrix or a ``ColumnOperatorProtocol``.  The core
algorithm fetches columns via ``operator.column(j)`` and never requires the
full matrix.  When the backend is NumpyBkd and numba is available, the inner
update dispatches to a JIT-compiled kernel.  A ``KernelColumnOperator``
with a recognized kernel type triggers a fully fused numba path that
avoids materializing the dense kernel matrix entirely.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol, runtime_checkable

import numpy as np

NumbaScalarKernelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Array_co, Backend
from pyapprox.util.optional_deps import package_available

_HAS_NUMBA = package_available("numba")


# -------------------------------------------------------------------
# Column operator protocol and implementations
# -------------------------------------------------------------------


@runtime_checkable
class KernelLike(Protocol[Array]):
    """Minimal kernel interface for KernelColumnOperator.

    Any object with ``__call__(X1, X2) -> Array`` and ``diag(X) -> Array``
    satisfies this. The surrogates KernelProtocol is a structural subtype.
    """

    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
        ...

    def diag(self, X1: Array) -> Array:
        ...


@runtime_checkable
class NumbaKernelLike(Protocol):
    """Kernel that provides a numba-compiled scalar evaluator."""

    def numba_eval(self) -> NumbaScalarKernelFn:
        ...

    def numba_kernel_params(self) -> np.ndarray:
        ...


@runtime_checkable
class ColumnOperatorProtocol(Protocol[Array_co]):
    """Symmetric matrix represented by column access and diagonal."""

    def column(self, j: int) -> Array_co:
        ...

    def diagonal(self) -> Array_co:
        ...

    def nvars(self) -> int:
        ...


class DenseColumnOperator(Generic[Array]):
    """Column operator backed by a dense matrix."""

    def __init__(self, K: Array, bkd: Backend[Array]) -> None:
        self._K = K
        self._bkd = bkd

    def column(self, j: int) -> Array:
        return self._K[:, j]

    def diagonal(self) -> Array:
        bkd = self._bkd
        return bkd.ravel(self._K)[:: self._K.shape[0] + 1]

    def nvars(self) -> int:
        return int(self._K.shape[0])

    def matrix(self) -> Array:
        return self._K


class KernelColumnOperator(Generic[Array]):
    """Column operator that evaluates kernel(X, x_j) on demand.

    Parameters
    ----------
    kernel : KernelLike[Array]
        Kernel with ``__call__(X1, X2) -> Array`` and ``diag(X) -> Array``.
    X : Array
        Data points, shape ``(nvars_in, n)``.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        kernel: KernelLike[Array],
        X: Array,
        bkd: Backend[Array],
    ) -> None:
        self._kernel = kernel
        self._X = X
        self._bkd = bkd
        self._n = int(X.shape[1])

    def column(self, j: int) -> Array:
        col = self._kernel(self._X, self._X[:, j : j + 1])
        return self._bkd.reshape(col, (-1,))

    def diagonal(self) -> Array:
        return self._kernel.diag(self._X)

    def nvars(self) -> int:
        return self._n

    def kernel(self) -> KernelLike[Array]:
        return self._kernel

    def points(self) -> Array:
        return self._X


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _swap_rows(matrix: Array, ii: int, jj: int, bkd: Backend[Array]) -> None:
    """Swap rows ii and jj of a 1D array in place."""
    indices = bkd.asarray([ii, jj], dtype=bkd.int64_dtype())
    swap_indices = bkd.asarray([jj, ii], dtype=bkd.int64_dtype())
    matrix[indices] = bkd.copy(matrix[swap_indices])




# -------------------------------------------------------------------
# Factorizer
# -------------------------------------------------------------------


class PivotedCholeskyFactorizer(Generic[Array]):
    """Backend-agnostic pivoted Cholesky factorization.

    Computes a low-rank pivoted Cholesky decomposition of a symmetric
    positive semi-definite matrix.

    Parameters
    ----------
    K_or_op : Array or ColumnOperatorProtocol
        Dense matrix or column operator.
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
        K_or_op: Array | ColumnOperatorProtocol[Array],
        bkd: Backend[Array],
        econ: bool = True,
        tol: float = 1e-14,
    ) -> None:
        self._bkd = bkd
        self._econ = econ
        self._tol = tol
        self._ncompleted_pivots = 0
        self._termination_flag = -1
        self._termination_msg = ""
        self._L: Array | None = None
        self._pivots_arr: Array | None = None
        self._diag: Array | None = None
        self._init_pivots: Array | None = None
        self._pivot_weights: Array | None = None
        self._init_error: Array | None = None

        if isinstance(K_or_op, ColumnOperatorProtocol):
            self._op = K_or_op
        else:
            self._op = DenseColumnOperator(bkd.copy(K_or_op), bkd)

        self._nrows = self._op.nvars()
        self._use_numba = (
            isinstance(bkd, NumpyBkd) and _HAS_NUMBA and econ
        )

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
        if npivots > self._nrows:
            raise ValueError(
                "Number of pivots requested exceeds number of matrix rows"
            )
        self._init_pivots = init_pivots
        self._pivot_weights = pivot_weights
        self._ncompleted_pivots = 0

        if self._use_numba and npivots > 0:
            if self._try_fused_numba(npivots, init_pivots, pivot_weights):
                return
            if self._try_dense_numba(npivots, init_pivots, pivot_weights):
                return

        self._factorize_generic(npivots)

    # -- fused numba: kernel eval baked into pchol loop ----------------

    def _try_fused_numba(
        self,
        npivots: int,
        init_pivots: Array | None,
        pivot_weights: Array | None,
    ) -> bool:
        if not isinstance(self._op, KernelColumnOperator):
            return False

        kernel = self._op.kernel()
        if not isinstance(kernel, NumbaKernelLike):
            return False

        from pyapprox.util.linalg.pivoted_cholesky_numba import (
            pivoted_cholesky_fused_numba,
        )

        X_np = np.ascontiguousarray(
            self._bkd.to_numpy(self._op.points()), dtype=np.float64,
        )
        kernel_func = kernel.numba_eval()
        kernel_params = np.ascontiguousarray(
            kernel.numba_kernel_params(), dtype=np.float64,
        )
        w_np, ip_np, n_ip, use_weights = _pack_numba_args(
            init_pivots, pivot_weights,
        )

        L, perm, diag, nc, ie = pivoted_cholesky_fused_numba(
            X_np, kernel_func, kernel_params,
            npivots, self._tol, w_np, ip_np, n_ip, use_weights,
        )
        self._store_numba_result(L, perm, diag, nc, npivots, init_error=ie)
        return True

    # -- dense numba: precomputed K passed to numba loop ---------------

    def _try_dense_numba(
        self,
        npivots: int,
        init_pivots: Array | None,
        pivot_weights: Array | None,
    ) -> bool:
        if not isinstance(self._op, DenseColumnOperator):
            return False

        from pyapprox.util.linalg.pivoted_cholesky_numba import (
            pivoted_cholesky_econ_numba,
        )

        K_np = np.ascontiguousarray(
            self._op.matrix(), dtype=np.float64,
        )
        w_np, ip_np, n_ip, use_weights = _pack_numba_args(
            init_pivots, pivot_weights,
        )

        L, perm, diag, nc = pivoted_cholesky_econ_numba(
            K_np, npivots, self._tol, w_np, ip_np, n_ip, use_weights,
        )
        self._store_numba_result(L, perm, diag, nc, npivots, K_np)
        return True

    # -- generic bkd path: column operator interface -------------------

    def _factorize_generic(self, npivots: int) -> None:
        bkd = self._bkd
        self._L = bkd.zeros((self._nrows, self._nrows))
        self._pivots_arr = bkd.arange(self._nrows)
        self._diag = self._op.diagonal()
        self._init_error = bkd.sum(bkd.abs(self._diag))
        self._update_loop(npivots)

    # -- result storage ------------------------------------------------

    def _store_numba_result(
        self,
        L: np.ndarray,
        perm: np.ndarray,
        diag: np.ndarray,
        nc: int,
        npivots: int,
        K_np: np.ndarray | None = None,
        init_error: float | None = None,
    ) -> None:
        bkd = self._bkd
        L_full = np.zeros((self._nrows, self._nrows), dtype=np.float64)
        L_full[:, :npivots] = L
        self._L = bkd.asarray(L_full)
        self._pivots_arr = bkd.asarray(perm)
        self._diag = bkd.asarray(diag)
        if init_error is not None:
            self._init_error = bkd.asarray(init_error)
        elif K_np is not None:
            self._init_error = bkd.asarray(
                np.sum(np.abs(np.diag(K_np)))
            )
        else:
            self._init_error = bkd.asarray(float(self._nrows))
        self._ncompleted_pivots = nc
        if nc < npivots:
            self._termination_flag = 1
            self._termination_msg = "Tolerance reached"
        else:
            self._termination_flag = 0
            self._termination_msg = "Factorization completed successfully"

    # -- update / continuation -----------------------------------------

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
        assert self._L is not None
        assert self._pivots_arr is not None
        assert self._diag is not None
        assert self._init_error is not None

        if not self._econ and self._pivot_weights is not None:
            raise ValueError("pivot weights not used when econ is False")

        for ii in range(self._ncompleted_pivots, npivots):
            pivot = self._compute_pivot(ii)
            _swap_rows(self._pivots_arr, ii, pivot, bkd)

            if self._diag[self._pivots_arr[ii]] <= 1e-14:
                self._termination_flag = 3
                raise ValueError("Matrix is not positive semi-definite")

            self._L[self._pivots_arr[ii], ii] = bkd.sqrt(
                self._diag[self._pivots_arr[ii]]
            )

            pivot_idx = int(bkd.to_numpy(
                self._pivots_arr[ii : ii + 1]
            )[0])
            full_col = self._op.column(pivot_idx)
            col_pivot = full_col[self._pivots_arr[ii + 1 :]]

            self._L[self._pivots_arr[ii + 1 :], ii] = (
                col_pivot
                - self._L[self._pivots_arr[ii + 1 :], :ii]
                @ self._L[self._pivots_arr[ii], :ii]
            ) / self._L[self._pivots_arr[ii], ii]

            self._diag[self._pivots_arr[ii + 1 :]] -= (
                self._L[self._pivots_arr[ii + 1 :], ii] ** 2
            )

            self._ncompleted_pivots += 1
            rel_error = (
                bkd.sum(self._diag[self._pivots_arr[ii + 1 :]])
                / self._init_error
            )
            if rel_error < self._tol:
                self._termination_flag = 1
                self._termination_msg = "Tolerance reached"
                return

        self._termination_flag = 0
        self._termination_msg = "Factorization completed successfully"

    def _compute_pivot(self, ii: int) -> int:
        bkd = self._bkd
        assert self._pivots_arr is not None
        assert self._diag is not None

        if self._init_pivots is not None and ii < len(self._init_pivots):
            mask = bkd.asarray(
                self._pivots_arr == self._init_pivots[ii]
            )
            return int(bkd.to_numpy(bkd.where(mask)[0][0:1])[0])

        if self._econ:
            if self._pivot_weights is None:
                return int(
                    bkd.to_numpy(
                        bkd.reshape(
                            bkd.argmax(
                                self._diag[self._pivots_arr[ii:]]
                            ) + ii,
                            (1,),
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

        # Schur complement mode — requires dense matrix
        if not isinstance(self._op, DenseColumnOperator):
            raise ValueError(
                "econ=False requires a dense matrix, not a column operator"
            )
        Amat = self._op.matrix()
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
                    bkd.argmax(
                        bkd.norm(schur_complement, axis=0) ** 2
                        / schur_diag
                    )
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


def _pack_numba_args(
    init_pivots: Array | None,
    pivot_weights: Array | None,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    use_weights = pivot_weights is not None
    w_np = (
        np.ascontiguousarray(pivot_weights, dtype=np.float64)
        if use_weights
        else np.empty(0, dtype=np.float64)
    )
    if init_pivots is not None:
        ip_np = np.ascontiguousarray(init_pivots, dtype=np.int64)
        n_ip = len(ip_np)
    else:
        ip_np = np.empty(0, dtype=np.int64)
        n_ip = 0
    return w_np, ip_np, n_ip, use_weights
