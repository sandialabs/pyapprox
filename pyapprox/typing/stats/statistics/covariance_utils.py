"""Covariance structure utilities for multifidelity statistics.

This module provides functions to compute the W, B, and V matrices needed
for variance estimation in multifidelity Monte Carlo methods.

W matrix: Cov[(f-E[f])^{⊗2}, (g-E[g])^{⊗2}] - covariance of Kronecker products
B matrix: Cov[f, (g-E[g])^{⊗2}] - cross-covariance of means and variance estimators
V matrix: Deterministic term from covariance (no pilot samples needed)
"""

from typing import List

from pyapprox.typing.util.backends.protocols import Array, Backend


def compute_W_entry(
    pilot_values_i: Array,
    pilot_values_j: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute W matrix entry for a pair of models.

    Computes Cov[(f_i - E[f_i])^{⊗2}, (f_j - E[f_j])^{⊗2}].

    Parameters
    ----------
    pilot_values_i : Array
        Pilot sample values for model i. Shape: (npilot, nqoi)
    pilot_values_j : Array
        Pilot sample values for model j. Shape: (npilot, nqoi)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        W entry block. Shape: (nqoi^2, nqoi^2)
    """
    nqoi = pilot_values_i.shape[1]
    npilot = pilot_values_i.shape[0]

    # Center the values
    mean_i = bkd.sum(pilot_values_i, axis=0) / npilot
    mean_j = bkd.sum(pilot_values_j, axis=0) / npilot
    centered_i = pilot_values_i - mean_i
    centered_j = pilot_values_j - mean_j

    # Compute outer products (Kronecker product approximation)
    # einsum("nk,nl->nkl", ...) computes outer product for each sample
    sq_i = bkd.reshape(
        bkd.einsum("nk,nl->nkl", centered_i, centered_i), (npilot, nqoi**2)
    )
    sq_j = bkd.reshape(
        bkd.einsum("nk,nl->nkl", centered_j, centered_j), (npilot, nqoi**2)
    )

    # Center the squared terms
    sq_i_mean = bkd.sum(sq_i, axis=0) / npilot
    sq_j_mean = bkd.sum(sq_j, axis=0) / npilot
    sq_i_centered = sq_i - sq_i_mean
    sq_j_centered = sq_j - sq_j_mean

    # Compute covariance via einsum
    W_ij = bkd.reshape(
        bkd.einsum("nk,nl->nkl", sq_i_centered, sq_j_centered), (npilot, -1)
    )
    W_ij = bkd.reshape(bkd.sum(W_ij, axis=0), (nqoi**2, nqoi**2)) / npilot

    return W_ij


def compute_W_from_pilot(
    pilot_values: List[Array],
    bkd: Backend[Array],
) -> Array:
    """Build full W matrix from pilot values.

    Parameters
    ----------
    pilot_values : List[Array]
        List of pilot sample values per model. Each has shape (npilot, nqoi).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Full W matrix. Shape: (nmodels * nqoi^2, nmodels * nqoi^2)
    """
    nmodels = len(pilot_values)
    nqoi = pilot_values[0].shape[1]

    # Build block matrix using nested lists and concatenation
    # W[i][j] is shape (nqoi^2, nqoi^2)
    rows = []
    for ii in range(nmodels):
        row_blocks = []
        for jj in range(nmodels):
            if jj < ii:
                # Use symmetry: W[i][j] = W[j][i].T
                W_ij = compute_W_entry(
                    pilot_values[jj], pilot_values[ii], bkd
                ).T
            else:
                W_ij = compute_W_entry(
                    pilot_values[ii], pilot_values[jj], bkd
                )
            row_blocks.append(W_ij)
        rows.append(bkd.hstack(row_blocks))

    return bkd.vstack(rows)


def compute_B_entry(
    pilot_values_i: Array,
    pilot_values_j: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute B matrix entry for a pair of models.

    Computes Cov[f_i, (f_j - E[f_j])^{⊗2}].

    Note: B is NOT symmetric. B[i,j] != B[j,i]^T in general.

    Parameters
    ----------
    pilot_values_i : Array
        Pilot sample values for model i (raw values). Shape: (npilot, nqoi)
    pilot_values_j : Array
        Pilot sample values for model j (for centering). Shape: (npilot, nqoi)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        B entry block. Shape: (nqoi, nqoi^2)
    """
    nqoi = pilot_values_i.shape[1]
    npilot = pilot_values_i.shape[0]

    # Raw values for model i (NOT centered)
    raw_i = pilot_values_i

    # Centered squared terms for model j
    mean_j = bkd.sum(pilot_values_j, axis=0) / npilot
    centered_j = pilot_values_j - mean_j
    sq_j = bkd.reshape(
        bkd.einsum("nk,nl->nkl", centered_j, centered_j), (npilot, nqoi**2)
    )
    sq_j_mean = bkd.sum(sq_j, axis=0) / npilot
    sq_j_centered = sq_j - sq_j_mean

    # Cross-covariance
    B_ij = bkd.reshape(
        bkd.einsum("nk,nl->nkl", raw_i, sq_j_centered), (npilot, -1)
    )
    B_ij = bkd.reshape(bkd.sum(B_ij, axis=0), (nqoi, nqoi**2)) / npilot

    return B_ij


def compute_B_from_pilot(
    pilot_values: List[Array],
    bkd: Backend[Array],
) -> Array:
    """Build full B matrix from pilot values.

    Note: B is NOT symmetric. B[i][j] != B[j][i]^T in general.

    Parameters
    ----------
    pilot_values : List[Array]
        List of pilot sample values per model. Each has shape (npilot, nqoi).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Full B matrix. Shape: (nmodels * nqoi, nmodels * nqoi^2)
    """
    nmodels = len(pilot_values)

    # Build block matrix - B is NOT symmetric
    rows = []
    for ii in range(nmodels):
        row_blocks = []
        for jj in range(nmodels):
            B_ij = compute_B_entry(pilot_values[ii], pilot_values[jj], bkd)
            row_blocks.append(B_ij)
        rows.append(bkd.hstack(row_blocks))

    return bkd.vstack(rows)


def compute_V_entry(
    cov_block: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute V matrix entry from a covariance block.

    V_entry = kron(C, C) + (1^T ⊗ C ⊗ 1) * (1 ⊗ C ⊗ 1^T)

    This is a deterministic computation from the covariance only,
    no pilot samples needed.

    Parameters
    ----------
    cov_block : Array
        Covariance matrix for a single model pair. Shape: (nqoi, nqoi)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        V entry block. Shape: (nqoi^2, nqoi^2)
    """
    nqoi = cov_block.shape[0]
    ones = bkd.ones((nqoi, 1))

    V = bkd.kron(cov_block, cov_block)
    V = V + bkd.kron(bkd.kron(ones.T, cov_block), ones) * bkd.kron(
        bkd.kron(ones, cov_block), ones.T
    )
    return V


def compute_V_from_covariance(
    cov: Array,
    nmodels: int,
    bkd: Backend[Array],
) -> Array:
    """Build full V matrix from covariance.

    Parameters
    ----------
    cov : Array
        Full cross-model covariance. Shape: (nmodels * nqoi, nmodels * nqoi)
    nmodels : int
        Number of models.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Full V matrix. Shape: (nmodels * nqoi^2, nmodels * nqoi^2)
    """
    nqoi = cov.shape[0] // nmodels

    # Build block matrix using symmetry
    rows = []
    for ii in range(nmodels):
        row_blocks = []
        for jj in range(nmodels):
            cov_block = cov[ii * nqoi : (ii + 1) * nqoi, jj * nqoi : (jj + 1) * nqoi]
            if jj < ii:
                # Use symmetry
                V_ij = compute_V_entry(
                    cov[jj * nqoi : (jj + 1) * nqoi, ii * nqoi : (ii + 1) * nqoi], bkd
                ).T
            else:
                V_ij = compute_V_entry(cov_block, bkd)
            row_blocks.append(V_ij)
        rows.append(bkd.hstack(row_blocks))

    return bkd.vstack(rows)


def covariance_of_variance_estimator(
    W: Array,
    V: Array,
    nsamples: int,
) -> Array:
    """Compute covariance of unbiased sample variance estimators.

    For the unbiased variance estimator s^2 = sum((x-mean)^2)/(n-1),
    the covariance is:

        Cov[s^2_i, s^2_j] = W[i,j] / n + V[i,j] / (n * (n-1))

    Parameters
    ----------
    W : Array
        W matrix (covariance of Kronecker products).
        Shape: (nmodels * nqoi^2, nmodels * nqoi^2)
    V : Array
        V matrix (deterministic from covariance).
        Shape: (nmodels * nqoi^2, nmodels * nqoi^2)
    nsamples : int
        Number of samples used in variance estimation.

    Returns
    -------
    Array
        Covariance of variance estimators.
        Shape: (nmodels * nqoi^2, nmodels * nqoi^2)
    """
    return W / nsamples + V / (nsamples * (nsamples - 1))


def extract_nqoi_nqoi_subproblem(
    C: Array,
    nmodels: int,
    nqoi: int,
    model_idx: List[int],
    qoi_idx: List[int],
    bkd: Backend[Array],
) -> Array:
    """Extract subproblem from covariance matrix for model/QoI subsets.

    Parameters
    ----------
    C : Array
        Full covariance matrix. Shape: (nmodels * nqoi, nmodels * nqoi)
    nmodels : int
        Total number of models.
    nqoi : int
        Number of QoIs per model.
    model_idx : List[int]
        Which models to keep.
    qoi_idx : List[int]
        Which QoIs to keep.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Extracted subproblem.
        Shape: (len(model_idx) * len(qoi_idx), len(model_idx) * len(qoi_idx))
    """
    nsub_models = len(model_idx)
    nsub_qoi = len(qoi_idx)
    C_new = bkd.zeros((nsub_models * nsub_qoi, nsub_models * nsub_qoi))

    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            cnt2 = 0
            idx1 = jj1 * nqoi + kk1
            for jj2 in model_idx:
                for kk2 in qoi_idx:
                    idx2 = jj2 * nqoi + kk2
                    C_new[cnt1, cnt2] = C[idx1, idx2]
                    cnt2 += 1
            cnt1 += 1
    return C_new


def extract_nqoisq_nqoisq_subproblem(
    V: Array,
    nmodels: int,
    nqoi: int,
    model_idx: List[int],
    qoi_idx: List[int],
    bkd: Backend[Array],
) -> Array:
    """Extract subproblem from W or V matrix for model/QoI subsets.

    Parameters
    ----------
    V : Array
        Full W or V matrix. Shape: (nmodels * nqoi^2, nmodels * nqoi^2)
    nmodels : int
        Total number of models.
    nqoi : int
        Number of QoIs per model.
    model_idx : List[int]
        Which models to keep.
    qoi_idx : List[int]
        Which QoIs to keep.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Extracted subproblem.
        Shape: (len(model_idx) * len(qoi_idx)^2, len(model_idx) * len(qoi_idx)^2)
    """
    nsub_models = len(model_idx)
    nsub_qoi = len(qoi_idx)
    V_new = bkd.zeros(
        (nsub_models * nsub_qoi**2, nsub_models * nsub_qoi**2)
    )

    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            for ll1 in qoi_idx:
                cnt2 = 0
                idx1 = jj1 * nqoi**2 + kk1 * nqoi + ll1
                for jj2 in model_idx:
                    for kk2 in qoi_idx:
                        for ll2 in qoi_idx:
                            idx2 = jj2 * nqoi**2 + kk2 * nqoi + ll2
                            V_new[cnt1, cnt2] = V[idx1, idx2]
                            cnt2 += 1
                cnt1 += 1
    return V_new


def extract_nqoi_nqoisq_subproblem(
    B: Array,
    nmodels: int,
    nqoi: int,
    model_idx: List[int],
    qoi_idx: List[int],
    bkd: Backend[Array],
) -> Array:
    """Extract subproblem from B matrix for model/QoI subsets.

    Parameters
    ----------
    B : Array
        Full B matrix. Shape: (nmodels * nqoi, nmodels * nqoi^2)
    nmodels : int
        Total number of models.
    nqoi : int
        Number of QoIs per model.
    model_idx : List[int]
        Which models to keep.
    qoi_idx : List[int]
        Which QoIs to keep.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Extracted subproblem.
        Shape: (len(model_idx) * len(qoi_idx), len(model_idx) * len(qoi_idx)^2)
    """
    nsub_models = len(model_idx)
    nsub_qoi = len(qoi_idx)
    B_new = bkd.zeros((nsub_models * nsub_qoi, nsub_models * nsub_qoi**2))

    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            cnt2 = 0
            idx1 = jj1 * nqoi + kk1
            for jj2 in model_idx:
                for kk2 in qoi_idx:
                    for ll2 in qoi_idx:
                        idx2 = jj2 * nqoi**2 + kk2 * nqoi + ll2
                        B_new[cnt1, cnt2] = B[idx1, idx2]
                        cnt2 += 1
            cnt1 += 1
    return B_new


def compute_covariance_from_pilot(
    pilot_values: List[Array],
    bkd: Backend[Array],
) -> Array:
    """Compute cross-model covariance from pilot values.

    Parameters
    ----------
    pilot_values : List[Array]
        List of pilot sample values per model. Each has shape (npilot, nqoi).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Full covariance matrix. Shape: (nmodels * nqoi, nmodels * nqoi)
    """
    # Stack all model outputs horizontally
    stacked = bkd.hstack(pilot_values)  # (npilot, nmodels * nqoi)
    npilot = stacked.shape[0]

    # Compute covariance with Bessel correction
    mean = bkd.sum(stacked, axis=0) / npilot
    centered = stacked - mean
    cov = centered.T @ centered / (npilot - 1)

    return cov
