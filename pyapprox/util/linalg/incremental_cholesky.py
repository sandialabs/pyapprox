"""Incremental Cholesky factorization with rank-1 updates."""

import warnings
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class IncrementalCholeskyFactorization(Generic[Array]):
    """Maintains L and L_inv with incremental rank-1 updates.

    Given a kernel matrix K and a sequence of pivot indices, incrementally
    builds the Cholesky factor L and its inverse L_inv such that
    K[pivots, pivots] = L @ L.T.

    Parameters
    ----------
    K : Array
        Symmetric positive semi-definite matrix.
    bkd : Backend[Array]
        Backend for numerical computations.
    """

    def __init__(self, K: Array, bkd: Backend[Array]) -> None:
        self._K = K
        self._bkd = bkd
        self._pivots: list[int] = []
        self._L: Array | None = None
        self._L_inv: Array | None = None

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def L(self) -> Array:
        """Return the current Cholesky factor."""
        if self._L is None:
            raise RuntimeError("No pivots added yet")
        return self._L

    def L_inv(self) -> Array:
        """Return the inverse of the current Cholesky factor."""
        if self._L_inv is None:
            raise RuntimeError("No pivots added yet")
        return self._L_inv

    def npivots(self) -> int:
        """Return number of pivots added so far."""
        return len(self._pivots)

    def pivots(self) -> Array:
        """Return pivot indices as an array."""
        return self._bkd.asarray(self._pivots)

    def add_pivot(self, pivot: int) -> None:
        """Add one pivot, updating L and L_inv incrementally.

        Uses the block Cholesky update formula. Falls back to full
        recomputation if L_22^2 <= 0, with warning.

        Parameters
        ----------
        pivot : int
            Index into the K matrix to add as the next pivot.
        """
        bkd = self._bkd
        n = len(self._pivots)

        if n == 0:
            # First pivot: L = [[sqrt(K[p,p])]], L_inv = [[1/sqrt(K[p,p])]]
            k_pp = self._K[pivot, pivot]
            if k_pp <= 0:
                warnings.warn(
                    "Non-positive diagonal encountered; "
                    "falling back to full recomputation."
                )
                self._pivots.append(pivot)
                self._recompute_full()
                return
            sqrt_kpp = bkd.sqrt(k_pp)
            self._L = bkd.reshape(sqrt_kpp, (1, 1))
            self._L_inv = bkd.reshape(1.0 / sqrt_kpp, (1, 1))
            self._pivots.append(pivot)
            return

        # Block update: K_sub = K[pivots+[pivot], pivots+[pivot]]
        # L_new = [[L,    0   ],
        #          [l_21, l_22]]
        # where l_21 = L_inv @ K[pivots, pivot] and l_22 = sqrt(k_pp - l_21.T @ l_21)
        pivot_indices = self._pivots
        k_col = self._K[pivot_indices, pivot]  # (n,)
        k_pp = self._K[pivot, pivot]

        # l_21 = L_inv @ k_col  =>  L @ l_21 = k_col  =>  l_21 = L^{-1} k_col
        l_21 = self._L_inv @ bkd.reshape(k_col, (-1, 1))  # (n, 1)
        l_21_flat = bkd.flatten(l_21)  # (n,)

        l_22_sq = k_pp - l_21_flat @ l_21_flat
        if bkd.to_float(l_22_sq) <= 0:
            warnings.warn(
                "Non-positive diagonal encountered during incremental "
                "update; pivot is degenerate and will be skipped."
            )
            return

        l_22 = bkd.sqrt(l_22_sq)

        # Build new L
        zeros_col = bkd.zeros((n, 1))
        top_row = bkd.hstack([self._L, zeros_col])
        bot_row = bkd.hstack(
            [bkd.reshape(l_21_flat, (1, -1)), bkd.reshape(l_22, (1, 1))]
        )
        self._L = bkd.vstack([top_row, bot_row])

        # Build new L_inv using block inverse formula:
        # L_inv_new = [[L_inv,                  0       ],
        #              [-1/l_22 * l_21^T L_inv,  1/l_22  ]]
        l_22_inv = 1.0 / l_22
        bkd.zeros((1, n))
        # -l_22_inv * l_21^T @ L_inv
        bot_left = -l_22_inv * (l_21_flat @ self._L_inv)
        bot_left = bkd.reshape(bot_left, (1, n))
        top = bkd.hstack([self._L_inv, bkd.zeros((n, 1))])
        bot = bkd.hstack([bot_left, bkd.reshape(l_22_inv, (1, 1))])
        self._L_inv = bkd.vstack([top, bot])

        self._pivots.append(pivot)

    def _recompute_full(self) -> None:
        """Recompute L and L_inv from scratch using stored K matrix."""
        bkd = self._bkd
        n = len(self._pivots)
        if n == 0:
            self._L = None
            self._L_inv = None
            return
        idx = self._pivots
        K_sub = self._K[idx, :][:, idx]
        self._L = bkd.cholesky(K_sub)
        self._L_inv = bkd.inv(self._L)

    def update_K(self, K: Array) -> None:
        """Update the K matrix reference (after kernel change).

        Parameters
        ----------
        K : Array
            New kernel matrix.
        """
        self._K = K

    def reset(self) -> None:
        """Clear factorization to empty state."""
        self._pivots = []
        self._L = None
        self._L_inv = None
