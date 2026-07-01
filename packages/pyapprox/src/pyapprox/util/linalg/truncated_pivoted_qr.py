"""Matrix-free, truncated column-pivoted QR via Householder reflectors.

``scipy.linalg.qr(pivoting=True)`` (wrapped by :class:`PivotedQRFactorizer`)
always computes the *full* ``geqp3`` factorization and materializes the entire
``Q``/``R`` -- the ``k`` argument only slices the result afterwards, so it costs
``O(nrows * ncols * min(nrows, ncols))`` regardless of how few pivots are wanted.

:class:`TruncatedPivotedQRFactorizer` is the column-pivoted QR analogue of
:class:`pyapprox.util.linalg.pivoted_cholesky.PivotedCholeskyFactorizer`: it runs
only ``npivots`` reflector steps (or stops at the numerical-rank cliff), and the
trailing-submatrix updates shrink each step, so the cost is
``O(nrows * ncols * npivots)`` -- linear in the budget.

It uses Householder reflectors and Businger-Golub column pivoting with the
Drmac-Bujanovic norm safeguard -- numerically the same algorithm as LAPACK
``geqp3`` -- so it is *unconditionally stable* and matches scipy's pivot order
and ``R``-diagonal.  (A Gram-Schmidt variant is simpler but loses orthogonality
on matrices with a wide dynamic range and then under-resolves the rank, so it is
not used.)

Pivoting on ``A`` directly (not the Gram ``A^T A``) keeps the spectrum
unsquared, so it resolves the true numerical rank rather than the
``sqrt(eps) * sigma_max`` floor that pivoted Cholesky on ``A^T A`` hits.

When the backend is NumpyBkd and numba is available, the inner loop dispatches
to a JIT-compiled kernel; otherwise a vectorized backend-generic numpy path runs.
"""

from __future__ import annotations

from typing import Generic, Optional, Tuple

import numpy as np

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available

_HAS_NUMBA = package_available("numba")


class TruncatedPivotedQRFactorizer(Generic[Array]):
    """Matrix-free, truncated column-pivoted QR via Householder reflectors.

    Computes the leading ``npivots`` pivots of the column-pivoted QR of ``A``
    without computing the full factorization, stopping early at the
    numerical-rank cliff.  Uses Householder reflectors with Businger-Golub
    pivoting and the geqp3 norm safeguard, so it matches LAPACK / scipy to
    machine precision and stays stable on ill-conditioned data.  Mirrors the
    interface of
    :class:`pyapprox.util.linalg.pivoted_cholesky.PivotedCholeskyFactorizer`.

    Parameters
    ----------
    A : Array
        Matrix whose columns are selected, shape ``(nrows, ncols)``.
    bkd : Backend[Array]
        Computational backend.
    tol : float
        Relative tolerance on the diagonal of ``R``: stop once
        ``|R[ii, ii]| < tol * |R[0, 0]|``.  ``0`` runs to the hard rank.
    """

    def __init__(
        self,
        A: Array,
        bkd: Backend[Array],
        tol: float = 1e-12,
    ) -> None:
        self._bkd = bkd
        self._A = A
        self._tol = tol
        self._nrows = int(A.shape[0])
        self._ncols = int(A.shape[1])
        self._pivots_arr: Optional[Array] = None
        self._R: Optional[Array] = None
        self._betas: Optional[Array] = None
        self._V: Optional[Array] = None
        self._ncompleted_pivots = 0
        self._termination_flag = -1
        self._termination_msg = ""
        self._use_numba = isinstance(bkd, NumpyBkd) and _HAS_NUMBA

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def factorize(self, npivots: Optional[int] = None) -> None:
        """Run truncated Householder pivoted QR for up to ``npivots`` pivots."""
        max_piv = min(self._nrows, self._ncols)
        npivots = max_piv if npivots is None else min(int(npivots), max_piv)
        if npivots > self._ncols:
            raise ValueError(
                "Number of pivots requested exceeds number of columns"
            )

        if self._use_numba and npivots > 0:
            R_out, V, betas, perm, nc, flag = self._factorize_numba(npivots)
        else:
            R_out, V, betas, perm, nc, flag = self._factorize_generic(npivots)

        self._R = R_out
        self._V = V[:, :nc]
        self._betas = betas[:nc]
        self._pivots_arr = perm[:nc]
        self._ncompleted_pivots = nc
        self._termination_flag = 0 if flag == 0 else 1
        self._termination_msg = (
            "Tolerance reached" if flag == 1
            else "Factorization completed successfully"
        )

    # -- numba fast path (numpy backend only) --------------------------

    def _factorize_numba(
        self, npivots: int,
    ) -> Tuple[Array, Array, Array, Array, int, int]:
        from pyapprox.util.linalg.truncated_pivoted_qr_numba import (
            householder_pivoted_qr_numba,
        )

        bkd = self._bkd
        R = np.ascontiguousarray(
            bkd.to_numpy(self._A).astype(np.float64, copy=True)
        )
        R_out, V, betas, perm, nc, flag = householder_pivoted_qr_numba(
            R, npivots, self._tol,
        )
        return (
            bkd.asarray(R_out),
            bkd.asarray(V),
            bkd.asarray(betas),
            bkd.asarray(perm, dtype=bkd.int64_dtype()),
            int(nc),
            int(flag),
        )

    # -- backend-generic path ------------------------------------------

    def _factorize_generic(
        self, npivots: int,
    ) -> Tuple[Array, Array, Array, Array, int, int]:
        bkd = self._bkd
        nrows, ncols = self._nrows, self._ncols
        R = bkd.copy(self._A)
        V = bkd.zeros((nrows, npivots))
        betas = bkd.zeros((npivots,))
        perm = bkd.arange(ncols, dtype=bkd.int64_dtype())

        # Businger-Golub column norms, with the geqp3 norm safeguard.
        col_norms = bkd.sqrt(bkd.einsum("ij,ij->j", R, R))
        exact_norms = bkd.copy(col_norms)
        r00 = float(bkd.to_numpy(bkd.max(col_norms))) if ncols else 0.0
        cliff = self._tol * r00
        recompute_ratio = float(np.sqrt(np.finfo(np.float64).eps))

        flag = 0
        nc = 0
        for ii in range(npivots):
            # Pivot: largest downdated residual column norm in [ii:].  Among
            # equal/near-equal norms (e.g. duplicate columns) the choice is
            # tie-arbitrary and may differ across backends, but every such choice
            # spans the same subspace -- the R-diagonal and reconstruction are
            # invariant.  No problem-dependent tie tolerance is imposed.
            p = ii + int(bkd.to_numpy(bkd.argmax(col_norms[ii:])))
            if p != ii:
                R[:, [ii, p]] = R[:, [p, ii]]
                col_norms[[ii, p]] = col_norms[[p, ii]]
                exact_norms[[ii, p]] = exact_norms[[p, ii]]
                perm[[ii, p]] = perm[[p, ii]]

            # xnorm is the EXACT residual norm of the pivot column; the cliff is
            # tested on it -- the downdated estimate stalls near
            # sqrt(eps)*||col|| and would not detect the cliff.
            x = R[ii:, ii]
            xnorm = float(bkd.to_numpy(bkd.norm(x)))
            if xnorm <= cliff:
                flag = 1
                break
            x0 = float(bkd.to_numpy(x[:1])[0])
            alpha = -(1.0 if x0 >= 0.0 else -1.0) * xnorm

            v = bkd.copy(x)
            v[0] = v[0] - alpha
            vnorm = float(bkd.to_numpy(bkd.norm(v)))
            if vnorm == 0.0:
                R[ii, ii] = alpha
                R[ii + 1:, ii] = bkd.zeros((nrows - ii - 1,))
                nc = ii + 1
            else:
                v = v / vnorm
                beta = 2.0
                # Apply H = I - beta v v^T to the trailing submatrix R[ii:, ii:].
                w = beta * (v @ R[ii:, ii:])
                R[ii:, ii:] = R[ii:, ii:] - bkd.outer(v, w)
                R[ii, ii] = alpha
                R[ii + 1:, ii] = bkd.zeros((nrows - ii - 1,))
                V[ii:, ii] = v
                betas[ii] = beta
                nc = ii + 1

            self._downdate_norms_generic(
                R, col_norms, exact_norms, ii, ncols, recompute_ratio, bkd,
            )

        return R, V, betas, perm, nc, flag

    @staticmethod
    def _downdate_norms_generic(
        R: Array,
        col_norms: Array,
        exact_norms: Array,
        ii: int,
        ncols: int,
        recompute_ratio: float,
        bkd: Backend[Array],
    ) -> None:
        """Businger-Golub norm downdate of columns ``ii+1:`` after eliminating
        column ``ii``, with the LAPACK ``dlaqps`` (Drmac-Bujanovic) recompute
        safeguard.  Mutates ``col_norms`` (the running downdated norm ``VN1``)
        and ``exact_norms`` (the norm ``VN2`` at the last exact recompute) in
        place.

        The recompute trigger follows LAPACK exactly: with
        ``temp = 1 - (|R[ii,j]| / VN1)^2`` the fractional norm kept this step and
        ``VN1/VN2`` the shrinkage since the last recompute,
        recompute the true residual norm when ``temp * (VN1/VN2)^2 <= tol3z``
        (``tol3z = sqrt(eps)``).  Comparing the *accumulated relative* loss --
        not just the absolute norm -- is what keeps the pivot order faithful to
        geqp3 deep into the spectrum; an absolute ``VN1 <= sqrt(eps)*orig`` test
        triggers far too late and lets pivoting drift."""
        j0 = ii + 1
        if j0 >= ncols:
            return
        # Copy the read slices so the final in-place writes to col_norms /
        # exact_norms cannot alias the operands mid-expression (torch slices are
        # views; numpy arithmetic copies -- copying makes both backends agree).
        vn1 = bkd.copy(col_norms[j0:])
        vn2 = bkd.copy(exact_norms[j0:])
        safe_vn1 = bkd.where(vn1 > 0.0, vn1, bkd.ones(vn1.shape))
        safe_vn2 = bkd.where(vn2 > 0.0, vn2, bkd.ones(vn2.shape))

        t = bkd.abs(R[ii, j0:]) / safe_vn1
        temp = bkd.maximum(bkd.zeros(t.shape), (1.0 - t) * (1.0 + t))
        ratio = vn1 / safe_vn2
        temp2 = temp * ratio * ratio

        trailing = R[ii + 1:, j0:]
        true_nrm = bkd.sqrt(bkd.einsum("ij,ij->j", trailing, trailing))
        downdated = vn1 * bkd.sqrt(temp)

        need = (vn1 > 0.0) & (temp2 <= recompute_ratio)
        col_norms[j0:] = bkd.where(need, true_nrm, downdated)
        # VN2 is refreshed to the exact norm only where we recomputed.
        exact_norms[j0:] = bkd.where(need, true_nrm, vn2)

    # -- results --------------------------------------------------------

    def pivots(self) -> Array:
        """Return the selected pivot column indices (selection order)."""
        if self._pivots_arr is None:
            raise RuntimeError("Must call factorize() first")
        return self._pivots_arr

    def npivots(self) -> int:
        """Return the number of completed pivots."""
        return self._ncompleted_pivots

    def rdiag(self) -> Array:
        """Return ``|diag(R)|`` for the completed pivots (the residual norms)."""
        if self._R is None:
            raise RuntimeError("Must call factorize() first")
        R = self._bkd.to_numpy(self._R)
        nc = self._ncompleted_pivots
        return self._bkd.asarray(np.abs(np.diag(R)[:nc]))

    def factor(self) -> Array:
        """Return the explicit orthonormal factor ``Q`` of shape
        ``(nrows, npivots)``.

        Formed on request by applying the stored reflectors to the identity --
        the factorization itself never materializes it.
        """
        if self._V is None or self._betas is None:
            raise RuntimeError("Must call factorize() first")
        V = self._bkd.to_numpy(self._V)
        betas = self._bkd.to_numpy(self._betas)
        nc = self._ncompleted_pivots
        Q = np.eye(self._nrows, nc, dtype=np.float64)
        for ii in range(nc - 1, -1, -1):
            v = V[:, ii]
            if betas[ii] != 0.0:
                Q -= betas[ii] * np.outer(v, v @ Q)
        return self._bkd.asarray(Q)

    def success(self) -> bool:
        """Return True if the factorization completed without errors."""
        return self._termination_flag >= 0
