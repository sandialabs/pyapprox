"""Numba JIT-compiled truncated Householder pivoted-QR kernel.

Matrix-free, truncated column-pivoted QR via Householder reflectors with
Businger-Golub pivoting and the Drmac-Bujanovic norm safeguard -- the same
algorithm as LAPACK ``geqp3``, run for only ``npivots`` reflector steps.

Note: this module requires numba.  If numba is not available, importing it
raises ImportError, which the dispatch in truncated_pivoted_qr.py handles.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def householder_pivoted_qr_numba(
    R: np.ndarray,
    npivots: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Truncated Householder pivoted QR (column pivoting).

    Parameters
    ----------
    R : (nrows, ncols) float64 array
        Working copy of the matrix (overwritten with the R factor in its upper
        triangle over the pivoted columns).
    npivots : int
        Maximum number of reflector steps.
    tol : float
        Relative tolerance on |R[ii,ii]|: stop once it is < tol * |R[0,0]|.

    Returns
    -------
    R : (nrows, ncols) float64 array
        The overwritten matrix (R factor in the leading columns).
    V : (nrows, npivots) float64 array
        Householder vectors (column ii valid for rows ii:).
    betas : (npivots,) float64 array
        Reflector scalars.
    perm : (ncols,) int64 array
        Column permutation; the first ``ncompleted`` entries are the pivots.
    ncompleted : int
        Number of reflector steps actually performed.
    termination_flag : int
        0 if run to npivots, 1 if stopped at the rank cliff.
    """
    nrows = R.shape[0]
    ncols = R.shape[1]

    V = np.zeros((nrows, npivots), dtype=np.float64)
    betas = np.zeros(npivots, dtype=np.float64)
    perm = np.arange(ncols, dtype=np.int64)

    col_norms = np.empty(ncols, dtype=np.float64)
    exact_norms = np.empty(ncols, dtype=np.float64)
    r00 = 0.0
    for j in range(ncols):
        s = 0.0
        for i in range(nrows):
            s += R[i, j] * R[i, j]
        s = np.sqrt(s)
        col_norms[j] = s
        exact_norms[j] = s
        if s > r00:
            r00 = s
    cliff = tol * r00
    recompute_ratio = np.sqrt(np.finfo(np.float64).eps)

    flag = 0
    nc = 0
    for ii in range(npivots):
        # Pivot: largest residual column norm in [ii:] (first max on ties, to
        # match numpy argmax; ties span the same subspace regardless).
        p = ii
        best = col_norms[ii]
        for j in range(ii + 1, ncols):
            if col_norms[j] > best:
                best = col_norms[j]
                p = j
        if p != ii:
            for i in range(nrows):
                tmp = R[i, ii]
                R[i, ii] = R[i, p]
                R[i, p] = tmp
            tc = col_norms[ii]
            col_norms[ii] = col_norms[p]
            col_norms[p] = tc
            te = exact_norms[ii]
            exact_norms[ii] = exact_norms[p]
            exact_norms[p] = te
            tp = perm[ii]
            perm[ii] = perm[p]
            perm[p] = tp

        # Householder reflector zeroing R[ii+1:, ii].  xnorm is the EXACT
        # residual norm of the pivot column, so the cliff is tested on it -- the
        # downdated col_norms estimate stalls near sqrt(eps)*||col|| and would
        # not detect the cliff.
        xnorm = 0.0
        for i in range(ii, nrows):
            xnorm += R[i, ii] * R[i, ii]
        xnorm = np.sqrt(xnorm)
        if xnorm <= cliff:
            flag = 1
            break
        x0 = R[ii, ii]
        sign = 1.0 if x0 >= 0.0 else -1.0
        alpha = -sign * xnorm
        if alpha == 0.0:
            flag = 1
            break

        # v = x; v[0] -= alpha; normalize.
        v0 = x0 - alpha
        vnorm_sq = v0 * v0
        for i in range(ii + 1, nrows):
            vnorm_sq += R[i, ii] * R[i, ii]
        vnorm = np.sqrt(vnorm_sq)

        if vnorm == 0.0:
            R[ii, ii] = alpha
            for i in range(ii + 1, nrows):
                R[i, ii] = 0.0
            nc = ii + 1
        else:
            inv = 1.0 / vnorm
            V[ii, ii] = v0 * inv
            for i in range(ii + 1, nrows):
                V[i, ii] = R[i, ii] * inv
            beta = 2.0
            betas[ii] = beta
            # Apply H = I - beta v v^T to trailing submatrix R[ii:, ii:].
            for j in range(ii, ncols):
                dot = 0.0
                for i in range(ii, nrows):
                    dot += V[i, ii] * R[i, j]
                dot *= beta
                for i in range(ii, nrows):
                    R[i, j] -= V[i, ii] * dot
            R[ii, ii] = alpha
            for i in range(ii + 1, nrows):
                R[i, ii] = 0.0
            nc = ii + 1

        # Downdate trailing column norms with the LAPACK dlaqps safeguard:
        # recompute the exact norm when temp * (VN1/VN2)^2 <= tol3z (the
        # accumulated relative precision loss since the last recompute crosses
        # sqrt(eps)), which keeps the pivot order faithful to geqp3 deep into
        # the spectrum -- an absolute VN1 <= sqrt(eps)*orig test fires too late.
        for j in range(ii + 1, ncols):
            vn1 = col_norms[j]
            if vn1 <= 0.0:
                continue
            vn2 = exact_norms[j]
            if vn2 < 1e-300:
                vn2 = 1e-300
            t = abs(R[ii, j]) / vn1
            temp = (1.0 - t) * (1.0 + t)
            if temp < 0.0:
                temp = 0.0
            ratio = vn1 / vn2
            temp2 = temp * ratio * ratio
            if temp2 <= recompute_ratio:
                s = 0.0
                for i in range(ii + 1, nrows):
                    s += R[i, j] * R[i, j]
                tn = np.sqrt(s)
                col_norms[j] = tn
                exact_norms[j] = tn
            else:
                col_norms[j] = vn1 * np.sqrt(temp)

    return R, V, betas, perm, nc, flag
