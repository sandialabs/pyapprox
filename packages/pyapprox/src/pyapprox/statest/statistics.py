"""Multi-output statistics for Monte Carlo estimators.

This module provides statistics classes for computing mean, variance, and
combined mean+variance from model evaluations.
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from pyapprox.statest.known import KnownMean, KnownStatistic, KnownVariance
from pyapprox.statest.protocols import ACVDiscrepancyStatistic, TargetArray
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend

# Helper functions


def _fancy_index_2d(arr: Array, idx0: Array, idx1: Array) -> Array:
    """Index a 2D array with two 1D index arrays (like np.ix_).

    Returns arr[idx0, :][:, idx1] which selects rows by idx0 and cols by idx1.
    """
    return arr[idx0, :][:, idx1]


def block_2x2(blocks: List[List[Array]], bkd: Backend[Array]) -> Array:
    """Create a 2x2 block matrix from nested lists of blocks."""
    return bkd.vstack([bkd.hstack(blocks[0]), bkd.hstack(blocks[1])])


def _get_nsamples_intersect(
    allocation_mat: Array, npartition_samples: Array, bkd: Backend[Array]
) -> Array:
    r"""
    Returns
    -------
    nsamples_intersect : Array (2*nmodels, 2*nmodels)
        The i,j entry contains contains
        :math:`|z^\star_i\cap\z^\star_j|` when i%2==0 and j%2==0
        :math:`|z_i\cap\z^\star_j|` when i%2==1 and j%2==0
        :math:`|z_i^\star\cap\z_j|` when i%2==0 and j%2==1
        :math:`|z_i\cap\z_j|` when i%2==1 and j%2==1
    """
    nmodels = allocation_mat.shape[0]
    nsubset_samples = npartition_samples[:, None] * allocation_mat
    rows = [
        bkd.sum(nsubset_samples[bkd.equal(allocation_mat[:, ii], 1)], axis=0)
        for ii in range(2 * nmodels)
    ]
    return bkd.asarray(bkd.stack(rows, axis=0), dtype=bkd.real_dtype())


def _get_nsamples_subset(
    allocation_mat: Array, npartition_samples: Array, bkd: Backend[Array]
) -> Array:
    r"""
    Get the number of samples allocated to the sample subsets
    :math:`|z^\star_i` and :math:`|z_i|`

    npartition_samples : Array (nmodels)
        The size of the partitions that make up the subsets
        :math:`z_i, i=0\ldots, M-1`. These are represented by different
        color blocks in the ACV papers figures of sample allocation
    """
    nmodels = allocation_mat.shape[0]
    entries = [
        bkd.sum(npartition_samples[bkd.equal(allocation_mat[:, ii], 1)])
        for ii in range(2 * nmodels)
    ]
    return bkd.asarray(bkd.stack(entries), dtype=bkd.real_dtype())


def _matrix_from_entries(
    entries: List[List[Array]], ncols: int, bkd: Backend[Array]
) -> Array:
    """Stack a nested list of 0-d arrays into a (len(entries), ncols)
    matrix of the real dtype; an empty list gives an empty matrix."""
    if not entries:
        return bkd.zeros((0, ncols), dtype=bkd.real_dtype())
    return bkd.stack([bkd.stack(row) for row in entries], axis=0)


def _vector_from_entries(entries: List[Array], bkd: Backend[Array]) -> Array:
    """Stack 0-d arrays into a vector; an empty list gives an empty one."""
    if not entries:
        return bkd.zeros((0,), dtype=bkd.real_dtype())
    return bkd.stack(entries)


def _get_acv_mean_discrepancy_covariances_multipliers(
    allocation_mat: Array, npartition_samples: Array, bkd: Backend[Array]
) -> Tuple[Array, Array]:
    nmodels = allocation_mat.shape[0]
    if bkd.to_bool(bkd.any(bkd.asarray(npartition_samples < 0))):
        raise RuntimeError(
            "An entry in npartition samples {0} was negative".format(npartition_samples)
        )
    nsamples_intersect = _get_nsamples_intersect(
        allocation_mat, npartition_samples, bkd
    )
    nsamples_subset = _get_nsamples_subset(allocation_mat, npartition_samples, bkd)
    g_entries: List[Array] = []
    G_entries: List[List[Array]] = []
    for ii in range(1, nmodels):
        g_entries.append(
            nsamples_intersect[2 * ii, 0 + 1]
            / (nsamples_subset[2 * ii] * nsamples_subset[0 + 1])
            - nsamples_intersect[2 * ii + 1, 0 + 1]
            / (nsamples_subset[2 * ii + 1] * nsamples_subset[0 + 1])
        )
        G_entries.append(
            [
                nsamples_intersect[2 * ii, 2 * jj]
                / (nsamples_subset[2 * ii] * nsamples_subset[2 * jj])
                - nsamples_intersect[2 * ii, 2 * jj + 1]
                / (nsamples_subset[2 * ii] * nsamples_subset[2 * jj + 1])
                - nsamples_intersect[2 * ii + 1, 2 * jj]
                / (nsamples_subset[2 * ii + 1] * nsamples_subset[2 * jj])
                + nsamples_intersect[2 * ii + 1, 2 * jj + 1]
                / (nsamples_subset[2 * ii + 1] * nsamples_subset[2 * jj + 1])
                for jj in range(1, nmodels)
            ]
        )
    Gmat = _matrix_from_entries(G_entries, nmodels - 1, bkd)
    return Gmat, _vector_from_entries(g_entries, bkd)


def _get_acv_variance_discrepancy_covariances_multipliers(
    allocation_mat: Array, npartition_samples: Array, bkd: Backend[Array]
) -> Tuple[Array, Array]:
    """
    Compute H from Equation 3.14 of Dixon et al.
    """
    nmodels = allocation_mat.shape[0]
    if bkd.to_bool(bkd.any(bkd.asarray(npartition_samples < 0))):
        raise RuntimeError("An entry in npartition samples was negative")
    nsamples_intersect = _get_nsamples_intersect(
        allocation_mat, npartition_samples, bkd
    )
    nsamples_subset = _get_nsamples_subset(allocation_mat, npartition_samples, bkd)
    h_entries: List[Array] = []
    H_entries: List[List[Array]] = []

    N0 = nsamples_subset[0 + 1]
    for ii in range(1, nmodels):
        Nis_0 = nsamples_intersect[2 * ii, 0 + 1]  # N_{0\cap i\star}
        Ni_0 = nsamples_intersect[2 * ii + 1, 0 + 1]  # N_{0\cap i}$
        Nis = nsamples_subset[2 * ii]  # N_{i\star}
        Ni = nsamples_subset[2 * ii + 1]  # N_{i}
        h_entries.append(
            Nis_0 * (Nis_0 - 1) / (N0 * (N0 - 1) * Nis * (Nis - 1))
            - Ni_0 * (Ni_0 - 1) / (N0 * (N0 - 1) * Ni * (Ni - 1))
        )
        H_row: List[Array] = []
        for jj in range(1, nmodels):
            Nis_js = nsamples_intersect[2 * ii, 2 * jj]  # N_{i\cap j\star}
            Ni_j = nsamples_intersect[2 * ii + 1, 2 * jj + 1]  # N_{i\cap j}$
            Ni_js = nsamples_intersect[2 * ii + 1, 2 * jj]  # N_{i\cap j\star}
            Nis_j = nsamples_intersect[2 * ii, 2 * jj + 1]  # N_{i\star\cap j}$
            Njs = nsamples_subset[2 * jj]  # N_{j\star}
            Nj = nsamples_subset[2 * jj + 1]  # N_{j}
            H_row.append(
                Nis_js * (Nis_js - 1) / (Nis * (Nis - 1) * Njs * (Njs - 1))
                - Nis_j * (Nis_j - 1) / (Nis * (Nis - 1) * Nj * (Nj - 1))
                - Ni_js * (Ni_js - 1) / (Ni * (Ni - 1) * Njs * (Njs - 1))
                + Ni_j * (Ni_j - 1) / (Ni * (Ni - 1) * Nj * (Nj - 1))
            )
        H_entries.append(H_row)
    Hmat = _matrix_from_entries(H_entries, nmodels - 1, bkd)
    return Hmat, _vector_from_entries(h_entries, bkd)


def _get_multioutput_acv_mean_discrepancy_covariances(
    cov: Array, Gmat: Array, gvec: Array, bkd: Backend[Array]
) -> Tuple[Array, Array]:
    r"""
    Compute the ACV discrepancies for estimating means

    Parameters
    ----------
    cov : Array (nmodels*nqoi, nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e. covariance between its QoI
        is cov[:nqoi, :nqoi]

    Gmat : Array (nmodels, nmodels)
        Encodes sample partition into mean-based delta covariances

    gvec : Array (nmodels, nmodels)
        Encodes sample partition into covariances between high-fidelity mean
        and deltas

    Returns
    -------
    discp_cov : array (nqoi*(nmodels-1), nqoi*(nmodels-1))
        The covariance between the delta
        :math:`\mathrm{Cov}[\Delta, \Delta]`

    discp_vec : array (nqoi, nqoi*(nmodels-1))
        The covariance between the highest fidelity estimators
        and the deltas :math:`\mathrm{Cov}[Q_0, \Delta]`
    """
    nmodels = len(gvec) + 1
    nqoi = cov.shape[0] // nmodels

    def cov_block(ii: int, jj: int) -> Array:
        return cov[(ii + 1) * nqoi : (ii + 2) * nqoi, (jj + 1) * nqoi : (jj + 2) * nqoi]

    discp_cov = _symmetric_block_matrix(
        lambda ii, jj: Gmat[ii, jj] * cov_block(ii, jj), nmodels - 1, bkd
    )
    discp_vec = _block_row(
        [
            gvec[ii] * cov[:nqoi, (ii + 1) * nqoi : (ii + 2) * nqoi]
            for ii in range(nmodels - 1)
        ],
        nqoi,
        bkd,
    )
    return discp_cov, discp_vec


def _symmetric_block_matrix(
    upper_block: Callable[[int, int], Array], nblocks: int, bkd: Backend[Array]
) -> Array:
    """The symmetric block matrix whose (ii, jj) block, jj >= ii, is
    ``upper_block(ii, jj)``; each lower block is the transpose of its
    mirror. No blocks give an empty matrix."""
    if nblocks == 0:
        return bkd.zeros((0, 0), dtype=bkd.real_dtype())
    upper = {
        (ii, jj): upper_block(ii, jj)
        for ii in range(nblocks)
        for jj in range(ii, nblocks)
    }
    return bkd.block(
        [
            [
                upper[(ii, jj)] if jj >= ii else upper[(jj, ii)].T
                for jj in range(nblocks)
            ]
            for ii in range(nblocks)
        ]
    )


def _block_row(blocks: List[Array], nrows: int, bkd: Backend[Array]) -> Array:
    """Blocks side by side; no blocks give an (nrows, 0) matrix."""
    if not blocks:
        return bkd.zeros((nrows, 0), dtype=bkd.real_dtype())
    return bkd.hstack(blocks)


def _get_multioutput_acv_variance_discrepancy_covariances(
    V: Array,
    W: Array,
    Gmat: Array,
    gvec: Array,
    Hmat: Array,
    hvec: Array,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    r"""
    Compute the ACV discrepancies for estimating variance

    Parameters
    ----------
    V : Array (nmodels*nqoi**2, nmodels**nqoi**2)
        Kroneker product of flattened covariance with itself

    W : Array (nmodels*nqoi**2, nmodels**nqoi**2)
        Covariance of Kroneker product of mean-centered values

    Gmat : Array (nmodels, nmodels)
        Encodes sample partition into mean-based delta covariances

    gvec : Array (nmodels, nmodels)
        Encodes sample partition into covariances between high-fidelity mean
        and deltas

    Hmat : Array (nmodels, nmodels)
        Encodes sample partition into variance-based delta covariances

    hvec : Array (nmodels, nmodels)
        Encodes sample partition into covariances between
        high-fidelity variance and deltas

    Returns
    -------
    discp_cov : array (nqoi*(nmodels-1), nqoi*(nmodels-1))
        The covariance of the estimator covariances
        :math:`\mathrm{Cov}[\Delta, \Delta]`

    discp_vec : array (nqoi, nqoi*(nmodels-1))
        The covariance between the highest fidelity estimators
        and the discrepancies :math:`\mathrm{Cov}[Q_0, \Delta]`
    """
    nmodels = len(gvec) + 1
    nqsq = V.shape[0] // nmodels

    def model_block(A: Array, ii: int, jj: int) -> Array:
        return A[(ii + 1) * nqsq : (ii + 2) * nqsq, (jj + 1) * nqsq : (jj + 2) * nqsq]

    discp_cov = _symmetric_block_matrix(
        lambda ii, jj: (
            Gmat[ii, jj] * model_block(W, ii, jj)
            + Hmat[ii, jj] * model_block(V, ii, jj)
        ),
        nmodels - 1,
        bkd,
    )
    discp_vec = _block_row(
        [
            gvec[ii] * W[0:nqsq, (ii + 1) * nqsq : (ii + 2) * nqsq]
            + hvec[ii] * V[0:nqsq, (ii + 1) * nqsq : (ii + 2) * nqsq]
            for ii in range(nmodels - 1)
        ],
        nqsq,
        bkd,
    )
    return discp_cov, discp_vec


def _get_multioutput_acv_mean_and_variance_discrepancy_covariances(
    cov: Array,
    V: Array,
    W: Array,
    B: Array,
    Gmat: Array,
    gvec: Array,
    Hmat: Array,
    hvec: Array,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    CF_mean, cf_mean = _get_multioutput_acv_mean_discrepancy_covariances(
        cov, Gmat, gvec, bkd
    )
    CF_var, cf_var = _get_multioutput_acv_variance_discrepancy_covariances(
        V, W, Gmat, gvec, Hmat, hvec, bkd
    )
    nmodels = len(gvec) + 1
    nqoi = cov.shape[0] // nmodels
    nqsq = V.shape[0] // nmodels
    stride = nqoi + nqsq
    nblocks = nmodels - 1
    if nblocks == 0:
        return (
            bkd.zeros((0, 0), dtype=bkd.real_dtype()),
            bkd.zeros((stride, 0), dtype=bkd.real_dtype()),
        )

    def mean_var_block(ii: int, jj: int) -> Array:
        # Gmat[ii, jj] times B between the means of model ii + 1 and the
        # variances of model jj + 1.
        B_ij = B[(ii + 1) * nqoi : (ii + 2) * nqoi, (jj + 1) * nqsq : (jj + 2) * nqsq]
        return Gmat[ii, jj] * B_ij

    # Each model contributes a mean row and a variance row of blocks.
    CF_rows: List[List[Array]] = []
    for ii in range(nblocks):
        mean_row: List[Array] = []
        var_row: List[Array] = []
        for jj in range(nblocks):
            mean_row += [
                CF_mean[ii * nqoi : (ii + 1) * nqoi, jj * nqoi : (jj + 1) * nqoi],
                mean_var_block(ii, jj),
            ]
            var_row += [
                mean_var_block(jj, ii).T,
                CF_var[ii * nqsq : (ii + 1) * nqsq, jj * nqsq : (jj + 1) * nqsq],
            ]
        CF_rows += [mean_row, var_row]

    cf_top: List[Array] = []
    cf_bottom: List[Array] = []
    for ii in range(nblocks):
        B_0i = B[0:nqoi, (ii + 1) * nqsq : (ii + 2) * nqsq]
        B_0i_T = B.T[0:nqsq, (ii + 1) * nqoi : (ii + 2) * nqoi]
        cf_top += [cf_mean[:, ii * nqoi : (ii + 1) * nqoi], gvec[ii] * B_0i]
        cf_bottom += [gvec[ii] * B_0i_T, cf_var[:, ii * nqsq : (ii + 1) * nqsq]]
    return bkd.block(CF_rows), bkd.block([cf_top, cf_bottom])


def _V_entry(cov: Array, bkd: Backend[Array]) -> Array:
    V = bkd.kron(cov, cov)
    ones = bkd.ones((cov.shape[0], 1), dtype=bkd.real_dtype())
    V += bkd.kron(bkd.kron(ones.T, cov), ones) * bkd.kron(bkd.kron(ones, cov), ones.T)
    return V


def _get_V_from_covariance(cov: Array, nmodels: int, bkd: Backend[Array]) -> Array:
    nqoi = cov.shape[0] // nmodels
    V: List[List[Optional[Array]]] = [
        [None for jj in range(nmodels)] for ii in range(nmodels)
    ]
    for ii in range(nmodels):
        V[ii][ii] = _V_entry(
            cov[ii * nqoi : (ii + 1) * nqoi, ii * nqoi : (ii + 1) * nqoi],
            bkd,
        )
        for jj in range(ii + 1, nmodels):
            V[ii][jj] = _V_entry(
                cov[ii * nqoi : (ii + 1) * nqoi, jj * nqoi : (jj + 1) * nqoi],
                bkd,
            )
            # We know V[ii][jj] is not None here
            V[jj][ii] = V[ii][jj].T  # type: ignore
    return bkd.block(V)  # type: ignore


def _covariance_of_variance_estimator(W: Array, V: Array, nsamples: Array) -> Array:
    return W / nsamples + V / (nsamples * (nsamples - 1))


def _W_entry(
    pilot_values_ii: Array, pilot_values_jj: Array, bkd: Backend[Array]
) -> Array:
    """Compute W matrix entry for variance estimator covariance.

    Parameters
    ----------
    pilot_values_ii : Array
        Pilot values for model i. Shape: (nqoi, nsamples)
    pilot_values_jj : Array
        Pilot values for model j. Shape: (nqoi, nsamples)
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        W matrix block. Shape: (nqoi^2, nqoi^2)
    """
    nqoi = pilot_values_ii.shape[0]
    npilot_samples = pilot_values_ii.shape[1]
    assert pilot_values_jj.shape[1] == npilot_samples
    means_ii = bkd.mean(pilot_values_ii, axis=1, keepdims=True)
    means_jj = bkd.mean(pilot_values_jj, axis=1, keepdims=True)
    centered_values_ii = pilot_values_ii - means_ii
    centered_values_jj = pilot_values_jj - means_jj
    # einsum "kn,ln->nkl" for (nqoi, nsamples) -> (nsamples, nqoi, nqoi)
    centered_values_sq_ii = bkd.reshape(
        bkd.einsum("kn,ln->nkl", centered_values_ii, centered_values_ii),
        (npilot_samples, -1),
    )
    centered_values_sq_jj = bkd.reshape(
        bkd.einsum("kn,ln->nkl", centered_values_jj, centered_values_jj),
        (npilot_samples, -1),
    )
    centered_values_sq_ii_mean = bkd.mean(centered_values_sq_ii, axis=0)
    centered_values_sq_jj_mean = bkd.mean(centered_values_sq_jj, axis=0)
    centered_values_sq = bkd.reshape(
        bkd.einsum(
            "nk,nl->nkl",
            centered_values_sq_ii - centered_values_sq_ii_mean,
            centered_values_sq_jj - centered_values_sq_jj_mean,
        ),
        (npilot_samples, -1),
    )
    mc_cov = bkd.reshape(bkd.sum(centered_values_sq, axis=0), (nqoi**2, nqoi**2)) / (
        npilot_samples
    )
    return mc_cov


def _get_W_from_pilot(pilot_values: Array, nmodels: int, bkd: Backend[Array]) -> Array:
    """Compute W matrix from pilot samples.

    Parameters
    ----------
    pilot_values : Array
        Stacked pilot values. Shape: (nqoi*nmodels, nsamples)
    nmodels : int
        Number of models.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        W matrix. Shape: (nqoi^2*nmodels, nqoi^2*nmodels)
    """
    # for one model 1 qoi this is the kurtosis
    nqoi = pilot_values.shape[0] // nmodels
    W: List[List[Optional[Array]]] = [
        [None for jj in range(nmodels)] for ii in range(nmodels)
    ]
    for ii in range(nmodels):
        pilot_values_ii = pilot_values[ii * nqoi : (ii + 1) * nqoi, :]
        W[ii][ii] = _W_entry(pilot_values_ii, pilot_values_ii, bkd)
        for jj in range(ii + 1, nmodels):
            pilot_values_jj = pilot_values[jj * nqoi : (jj + 1) * nqoi, :]
            W[ii][jj] = _W_entry(pilot_values_ii, pilot_values_jj, bkd)
            # We know W[ii][jj] is not None here
            W[jj][ii] = W[ii][jj].T  # type: ignore
    return bkd.block(W)  # type: ignore


def _B_entry(
    pilot_values_ii: Array, pilot_values_jj: Array, bkd: Backend[Array]
) -> Array:
    """Compute B matrix entry for mean-variance estimator covariance.

    Parameters
    ----------
    pilot_values_ii : Array
        Pilot values for model i. Shape: (nqoi, nsamples)
    pilot_values_jj : Array
        Pilot values for model j. Shape: (nqoi, nsamples)
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        B matrix block. Shape: (nqoi, nqoi^2)
    """
    nqoi = pilot_values_ii.shape[0]
    npilot_samples = pilot_values_ii.shape[1]
    assert pilot_values_jj.shape[1] == npilot_samples
    means_jj = bkd.mean(pilot_values_jj, axis=1, keepdims=True)
    centered_values_jj = pilot_values_jj - means_jj
    # einsum "kn,ln->nkl" for (nqoi, nsamples) -> (nsamples, nqoi, nqoi)
    centered_values_sq_jj = bkd.reshape(
        bkd.einsum("kn,ln->nkl", centered_values_jj, centered_values_jj),
        (npilot_samples, -1),
    )
    centered_values_sq_jj_mean = bkd.mean(centered_values_sq_jj, axis=0)
    # pilot_values_ii is (nqoi, nsamples), need (nsamples, nqoi) for einsum
    pilot_values_ii_T = bkd.transpose(pilot_values_ii)
    centered_values_sq = bkd.reshape(
        bkd.einsum(
            "nk,nl->nkl",
            pilot_values_ii_T,
            centered_values_sq_jj - centered_values_sq_jj_mean,
        ),
        (npilot_samples, -1),
    )
    mc_cov = bkd.reshape(bkd.sum(centered_values_sq, axis=0), (nqoi, nqoi**2)) / (
        npilot_samples
    )
    return mc_cov


def _get_B_from_pilot(pilot_values: Array, nmodels: int, bkd: Backend[Array]) -> Array:
    """Compute B matrix from pilot samples.

    Parameters
    ----------
    pilot_values : Array
        Stacked pilot values. Shape: (nqoi*nmodels, nsamples)
    nmodels : int
        Number of models.
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        B matrix. Shape: (nqoi*nmodels, nqoi^2*nmodels)
    """
    nqoi = pilot_values.shape[0] // nmodels
    model_values = [
        pilot_values[ii * nqoi : (ii + 1) * nqoi, :] for ii in range(nmodels)
    ]
    return bkd.block(
        [
            [_B_entry(model_values[ii], model_values[jj], bkd) for jj in range(nmodels)]
            for ii in range(nmodels)
        ]
    )


def _nqoi_nqoi_subproblem(
    C: Array,
    nmodels: int,
    nqoi: int,
    model_idx: Array,
    qoi_idx: Array,
    bkd: Backend[Array],
) -> Array:
    rows = _qoi_positions(model_idx, qoi_idx, nqoi, bkd)
    return _gather(C, rows, rows, bkd)


def _qoi_positions(
    model_idx: Array, qoi_idx: Array, nqoi: int, bkd: Backend[Array]
) -> List[int]:
    """Rows of the selected (model, QoI) pairs, model-major."""
    return [bkd.to_int(jj * nqoi + kk) for jj in model_idx for kk in qoi_idx]


def _qoisq_positions(
    model_idx: Array, qoi_idx: Array, nqoi: int, bkd: Backend[Array]
) -> List[int]:
    """Rows of the selected (model, QoI, QoI) triples, model-major."""
    return [
        bkd.to_int(jj * nqoi**2 + kk * nqoi + ll)
        for jj in model_idx
        for kk in qoi_idx
        for ll in qoi_idx
    ]


def _gather(A: Array, rows: List[int], cols: List[int], bkd: Backend[Array]) -> Array:
    """The submatrix of ``A`` at ``rows`` x ``cols``."""
    index = bkd.index_dtype()
    return _fancy_index_2d(
        A, bkd.asarray(rows, dtype=index), bkd.asarray(cols, dtype=index)
    )


def _nqoisq_nqoisq_subproblem(
    V: Array,
    nmodels: int,
    nqoi: int,
    model_idx: Array,
    qoi_idx: Array,
    bkd: Backend[Array],
) -> Array:
    rows = _qoisq_positions(model_idx, qoi_idx, nqoi, bkd)
    return _gather(V, rows, rows, bkd)


def _nqoi_nqoisq_subproblem(
    B: Array,
    nmodels: int,
    nqoi: int,
    model_idx: Array,
    qoi_idx: Array,
    bkd: Backend[Array],
) -> Array:
    return _gather(
        B,
        _qoi_positions(model_idx, qoi_idx, nqoi, bkd),
        _qoisq_positions(model_idx, qoi_idx, nqoi, bkd),
        bkd,
    )


def _ncovariance_stats(nqoi: int) -> int:
    """Number of distinct covariance entries estimated for ``nqoi`` QoIs.

    The lower triangle of the covariance, determined entirely by the
    number of quantities of interest, fixed at construction. No pilot
    data enters, so this is answerable before any is supplied -- which is
    why the statistics that report it need no pilot quantities to do so.
    """
    return nqoi * (nqoi + 1) // 2


def log_determinant_variance(bkd: Backend[Array], variance: Array) -> Array:
    eigvals = bkd.eigh(variance)[0]
    return bkd.sum(bkd.log(eigvals[eigvals > 1e-14]))


# Base class


class MultiOutputStatistic(ABC, Generic[Array]):
    """Abstract base class for multi-output statistics."""

    def __init__(self, nqoi: int, bkd: Backend[Array]):
        """
        Parameters
        ----------
        nqoi : integer
            The number of quantities of interest (QoI) that each model returns
        """
        self._nqoi = nqoi
        self._bkd = bkd
        self._nmodels: int = 0
        self._cov: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nqoi(self) -> int:
        """
        The number of quantities of interest (QoI) that each model returns
        """
        return self._nqoi

    def nmodels(self) -> int:
        """The number of models the pilot quantities describe.

        Zero until :meth:`set_pilot_quantities` is called, which is where
        the count is learned -- it follows from the shape of the pilot
        covariance rather than being declared at construction.
        """
        return self._nmodels

    def pilot_covariance(self) -> Array:
        """The covariance estimated from the pilot sample.

        Shape ``(nmodels * nqoi, nmodels * nqoi)``. Raises if pilot
        quantities have not been supplied; use
        :meth:`has_pilot_covariance` to ask without provoking that.
        """
        if self._cov is None:
            raise ValueError("must call set_pilot_quantities")
        return self._cov

    def has_pilot_covariance(self) -> bool:
        """Whether a pilot covariance has been supplied yet.

        For callers that branch on it rather than requiring it -- a
        question the accessor above cannot answer, since it raises.
        """
        return self._cov is not None

    @abstractmethod
    def with_backend(
        self, bkd: Backend[TargetArray]
    ) -> "ACVDiscrepancyStatistic[TargetArray]":
        """Return the same statistic, pilot quantities included, on ``bkd``."""
        raise NotImplementedError

    @abstractmethod
    def nstats(self) -> int:
        """The number of statistics computed"""
        raise NotImplementedError

    def stat_slot_indices(self, stat_name: str) -> List[int]:
        """Per-model stat-slot indices for a given statistic type.

        Parameters
        ----------
        stat_name : str
            Name of the statistic: ``"mean"`` or ``"variance"``.

        Returns
        -------
        List[int]
            Indices into the per-model stat vector for that statistic.

        Raises
        ------
        ValueError
            If ``stat_name`` is not available on this stat class.
        """
        raise ValueError(f"'{stat_name}' not available on {type(self).__name__}")

    def known_slots(self, known: KnownStatistic[Array]) -> List[int]:
        """Per-model stat slots a known statistic fills.

        Each statistic accepts the known kinds it can use and refuses
        the rest; none by default.

        Raises
        ------
        ValueError
            If this statistic does not accept that kind.
        """
        raise ValueError(
            f"{type(self).__name__} accepts no known {type(known).__name__}"
        )

    def check_known(self, known: Sequence[KnownStatistic[Array]]) -> None:
        """Raise unless the known statistics are consistent with this one.

        Every kind must be accepted, each (model, kind) given at most
        once, and each value shaped to its slots. A statistic needing
        more -- known values that must come together -- extends this.

        Raises
        ------
        ValueError
            If they are not.
        """
        seen: Set[Tuple[int, Type[KnownStatistic[Array]]]] = set()
        for item in known:
            slots = self.known_slots(item)
            key = (item.model, type(item))
            if key in seen:
                raise ValueError(
                    f"model {item.model} has more than one known {type(item).__name__}"
                )
            seen.add(key)
            if tuple(item.values.shape) != (len(slots),):
                raise ValueError(
                    f"known {type(item).__name__} for model {item.model} "
                    f"has shape {tuple(item.values.shape)}; "
                    f"{type(self).__name__} expects ({len(slots)},)"
                )

    @abstractmethod
    def sample_estimate(self, values: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def high_fidelity_estimator_covariance(self, nhf_samples: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def compute_pilot_quantities(self, pilot_values: List[Array]) -> Tuple[Any, ...]:
        raise NotImplementedError

    @abstractmethod
    def set_pilot_quantities(self, *args: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def _get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def _get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def get_pilot_quantities_subset(
        self, nmodels: int, nqoi: int, model_idx: Array, qoi_idx: Optional[Array] = None
    ) -> Tuple[Any, ...]:
        raise NotImplementedError

    @abstractmethod
    def subset(
        self,
        model_indices: List[int],
        qoi_indices: Optional[List[int]] = None,
    ) -> "MultiOutputStatistic[Array]":
        """Create statistic for a subset of models and/or QoI.

        Parameters
        ----------
        model_indices : List[int]
            Indices of models to include. Must include 0 (high-fidelity model).
        qoi_indices : List[int], optional
            Indices of QoI to include. If None, uses all QoI.

        Returns
        -------
        MultiOutputStatistic
            New statistic instance for the subset.

        Raises
        ------
        ValueError
            If model_indices does not include 0.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}(nmodels={1}, nqoi={2}, nstats={3})".format(
            self.__class__.__name__, self._nmodels, self._nqoi, self.nstats()
        )

    @abstractmethod
    def min_nsamples(self) -> int:
        """Min number of samples to compute the statistic"""
        raise NotImplementedError

    def continuous_dead_threshold(self) -> float:
        """Threshold below which the sigma-block formula is invalid.

        For continuous (non-integer) sample counts used during optimization.
        Returns 0.0 if the formula is valid for all n > 0.
        """
        return float(self.min_nsamples())

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: Union[int, Array],
        nsamples_subset0: Union[int, Array],
        nsamples_subset1: Union[int, Array],
    ) -> Array:
        # should resemble high_fidelity_estimator_covariance()
        raise NotImplementedError

    def _group_acv_sigma_block_derivs(
        self,
        subset: Array,
        nsamples: Union[int, Array],
    ) -> Tuple[Array, Array]:
        """Derivatives of the diagonal sigma block for IS estimators.

        Parameters
        ----------
        subset : Array
            Subset indices (same for row and column since IS diagonal).
        nsamples : int or Array
            Number of samples in this partition.

        Returns
        -------
        d1 : Array
            First derivative of sigma block w.r.t. nsamples.
        d2 : Array
            Second derivative of sigma block w.r.t. nsamples.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _group_acv_sigma_block_derivs"
        )

    def _check_pilot_values(self, pilot_values: List[Array]) -> None:
        if not isinstance(pilot_values, list):
            raise ValueError("pilot_values must be a list")
        for vals in pilot_values:
            if not isinstance(vals, ArrayProtocol):
                raise ValueError("pilot_values entry must be an ArrayProtocol")
            if vals.ndim != 2:
                raise ValueError("pilot_values entry must be 2D array")


class MultiOutputMean(MultiOutputStatistic[Array]):
    """Statistics for computing means across multiple models and QoIs."""

    def __init__(self, nqoi: int, bkd: Backend[Array]):
        super().__init__(nqoi, bkd)
        self._cov: Optional[Array] = None

    def with_backend(
        self, bkd: Backend[TargetArray]
    ) -> "ACVDiscrepancyStatistic[TargetArray]":
        stat = MultiOutputMean(self._nqoi, bkd)
        if self._cov is not None:
            stat.set_pilot_quantities(bkd.asarray(self._cov, dtype=bkd.real_dtype()))
        return stat

    def nstats(self) -> int:
        return self.nqoi()

    def stat_slot_indices(self, stat_name: str) -> List[int]:
        if stat_name == "mean":
            return list(range(self.nqoi()))
        raise ValueError(f"'{stat_name}' not available on {type(self).__name__}")

    def known_slots(self, known: KnownStatistic[Array]) -> List[int]:
        if isinstance(known, KnownMean):
            return list(range(self.nqoi()))
        return super().known_slots(known)

    def sample_estimate(self, values: Array) -> Array:
        """Compute sample mean estimate.

        Parameters
        ----------
        values : Array
            Model outputs. Shape: (nqoi, nsamples)

        Returns
        -------
        Array
            Mean estimate. Shape: (nqoi,)
        """
        return self._bkd.mean(values, axis=1)

    def high_fidelity_estimator_covariance(self, nhf_samples: Array) -> Array:
        if self._cov is None:
            raise ValueError("set_pilot_quantities must be called first")
        return self._cov[: self._nqoi, : self._nqoi] / nhf_samples

    def compute_pilot_quantities(self, pilot_values: List[Array]) -> Tuple[Array]:
        """Compute covariance from pilot samples.

        Parameters
        ----------
        pilot_values : List[Array]
            Pilot samples for each model. Each array has shape (nqoi, nsamples).

        Returns
        -------
        Tuple[Array]
            Covariance matrix of shape (nqoi*nmodels, nqoi*nmodels).
        """
        self._check_pilot_values(pilot_values)
        # Stack to (nqoi*nmodels, nsamples), then compute cov with rowvar=True
        pilot_values_stacked = self._bkd.vstack(pilot_values)
        return (self._bkd.cov(pilot_values_stacked, rowvar=True, ddof=1),)

    def set_pilot_quantities(self, *args: Any) -> None:
        cov = args[0]
        self._cov = self._bkd.asarray(cov, dtype=self._bkd.real_dtype())
        self._nmodels = self._cov.shape[0] // self._nqoi

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array
    ) -> Tuple[Array, Array]:
        if self._cov is None:
            raise ValueError("set_pilot_quantities must be called first")
        return _get_multioutput_acv_mean_discrepancy_covariances(
            self._cov, Gmat, gvec, self._bkd
        )

    def _get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        Gmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0 / self._bkd.to_float(npartition_samples[0]),
            dtype=self._bkd.real_dtype(),
        )
        gvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0 / self._bkd.to_float(npartition_samples[0]),
            dtype=self._bkd.real_dtype(),
        )
        return self._get_discrepancy_covariances(Gmat, gvec)

    def _get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        Gmat, gvec = _get_acv_mean_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples, self._bkd
        )
        return self._get_discrepancy_covariances(Gmat, gvec)

    def get_pilot_quantities_subset(
        self, nmodels: int, nqoi: int, model_idx: Array, qoi_idx: Optional[Array] = None
    ) -> Tuple[Array, ...]:
        if self._cov is None:
            raise ValueError("set_pilot_quantities must be called first")
        if qoi_idx is None:
            qoi_idx = self._bkd.arange(nqoi)
        cov_sub = _nqoi_nqoi_subproblem(
            self._cov, nmodels, nqoi, model_idx, qoi_idx, self._bkd
        )
        return (cov_sub,)

    def subset(
        self,
        model_indices: List[int],
        qoi_indices: Optional[List[int]] = None,
    ) -> "MultiOutputMean[Array]":
        """Create statistic for a subset of models and/or QoI.

        Parameters
        ----------
        model_indices : List[int]
            Indices of models to include. Must include 0 (high-fidelity model).
        qoi_indices : List[int], optional
            Indices of QoI to include. If None, uses all QoI.

        Returns
        -------
        MultiOutputMean
            New statistic instance for the subset.

        Raises
        ------
        ValueError
            If model_indices does not include 0.
        """
        if 0 not in model_indices:
            raise ValueError("model_indices must include 0 (high-fidelity)")

        model_idx = self._bkd.array(model_indices, dtype=self._bkd.index_dtype())

        if qoi_indices is None:
            qoi_idx = None
            new_nqoi = self._nqoi
        else:
            qoi_idx = self._bkd.array(qoi_indices, dtype=self._bkd.index_dtype())
            new_nqoi = len(qoi_indices)

        (cov_sub,) = self.get_pilot_quantities_subset(
            self._nmodels, self._nqoi, model_idx, qoi_idx
        )

        new_stat: MultiOutputMean[Array] = MultiOutputMean(new_nqoi, self._bkd)
        new_stat.set_pilot_quantities(cov_sub)
        return new_stat

    def min_nsamples(self) -> int:
        return 1

    def continuous_dead_threshold(self) -> float:
        return 0.0

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: Union[int, Array],
        nsamples_subset0: Union[int, Array],
        nsamples_subset1: Union[int, Array],
    ) -> Array:
        # should resemble high_fidelity_estimator_covariance()
        if self._cov is None:
            raise ValueError("must call set_pilot_quantities")
        cov = _fancy_index_2d(self._cov, subset0, subset1)
        return cov * nsamples_intersect / (nsamples_subset0 * nsamples_subset1)

    def _group_acv_sigma_block_derivs(
        self,
        subset: Array,
        nsamples: Union[int, Array],
    ) -> Tuple[Array, Array]:
        r"""Sigma = C/n  =>  dΣ/dn = -C/n²,  d²Σ/dn² = 2C/n³"""
        if self._cov is None:
            raise ValueError("must call set_pilot_quantities")
        C = _fancy_index_2d(self._cov, subset, subset)
        n2 = nsamples * nsamples
        n3 = n2 * nsamples
        return -C / n2, 2 * C / n3


class MultiOutputVariance(MultiOutputStatistic[Array]):
    """Statistics for computing variances across multiple models and QoIs."""

    def __init__(self, nqoi: int, bkd: Backend[Array]):
        super().__init__(nqoi, bkd)
        self._cov: Optional[Array] = None
        self._W: Optional[Array] = None
        self._V: Optional[Array] = None
        self._tril_idx: Optional[Tuple[Array, Array]] = None
        self._tril_idx_flat: Optional[Array] = None
        self._comp_idx: Optional[Array] = None
        self._Vcomp: Optional[Array] = None
        self._Wcomp: Optional[Array] = None
        self._lf_delta_idx: Optional[Array] = None
        self._hf_delta_idx: Optional[Array] = None

    def with_backend(
        self, bkd: Backend[TargetArray]
    ) -> "ACVDiscrepancyStatistic[TargetArray]":
        stat = MultiOutputVariance(self._nqoi, bkd)
        if self._cov is not None and self._W is not None:
            stat.set_pilot_quantities(
                bkd.asarray(self._cov, dtype=bkd.real_dtype()),
                bkd.asarray(self._W, dtype=bkd.real_dtype()),
            )
        return stat

    def _set_compressed_data(self) -> None:
        # subset0 wil contain indices into lower diagonal of covariance
        # e.g. for model subset [0, 1] with 2 qoi
        # subset0 = ([0, 1, 2], [3, 4, 5])
        # ([brackets denote qoi for each model]
        # but V and W are stored in terms on indices into entire covariance
        # e.g. ([0, 1, 2, 3], [4, 5, 6, 7])
        # so when subset  ([0, 1, 2], [3, 4, 5]) comes in we want to extract
        # entries of V corresponding to rows (and columns)
        # ([0, 2, 3], [4, 6, 7])

        if self._V is None or self._W is None:
            raise ValueError("V and W must be set before _set_compressed_data")

        # get compressed V
        self._tril_idx = self._bkd.tril_indices(self._nqoi)

        self._tril_idx_flat = self._bkd.reshape(
            self._bkd.arange(self._nqoi**2, dtype=self._bkd.index_dtype()),
            (self._nqoi, self._nqoi),
        )[self._tril_idx[0], self._tril_idx[1]]

        # for group acv
        self._comp_idx = self._bkd.hstack(
            [self._tril_idx_flat + ii * self._nqoi**2 for ii in range(self._nmodels)]
        )
        self._Vcomp = _fancy_index_2d(self._V, self._comp_idx, self._comp_idx)
        self._Wcomp = _fancy_index_2d(self._W, self._comp_idx, self._comp_idx)

        # for acv discrepancies (must exclude first model)
        if self._nmodels == 1:
            return

        self._lf_delta_idx = self._bkd.hstack(
            [
                self._tril_idx_flat + ii * self._nqoi**2
                for ii in range(self._nmodels - 1)
            ]
        )
        self._hf_delta_idx = self._lf_delta_idx[: self.nstats()]

    def nstats(self) -> int:
        ncov = _ncovariance_stats(self.nqoi())
        if self._tril_idx_flat is not None and (self._tril_idx_flat.shape[0] != ncov):
            raise RuntimeError(
                f"pilot quantities imply {self._tril_idx_flat.shape[0]} "
                f"covariance statistics but nqoi implies {ncov}"
            )
        return ncov

    def stat_slot_indices(self, stat_name: str) -> List[int]:
        if stat_name == "variance":
            return list(range(self.nstats()))
        raise ValueError(f"'{stat_name}' not available on {type(self).__name__}")

    def known_slots(self, known: KnownStatistic[Array]) -> List[int]:
        if isinstance(known, KnownVariance):
            return list(range(self.nstats()))
        return super().known_slots(known)

    def sample_estimate(self, values: Array) -> Array:
        """Compute sample variance estimate.

        Centers on the sample mean (not a known population mean) so the
        unbiased estimator covariance depends only on V and W matrices.

        Parameters
        ----------
        values : Array
            Model outputs. Shape: (nqoi*nmodels_in_subset, nsamples)

        Returns
        -------
        Array
            Variance estimate (lower triangular entries). Shape:
            (nstats*nmodels_in_subset,)
        """
        if self._tril_idx_flat is None:
            raise ValueError("set_pilot_quantities must be called first")
        nmodels_in_subset = values.shape[0] // self._nqoi
        flat_covs = [
            self._bkd.flatten(
                self._bkd.cov(
                    values[ii * self._nqoi : (ii + 1) * self._nqoi, :],
                    ddof=1,
                    rowvar=True,
                )
            )[self._tril_idx_flat]
            for ii in range(nmodels_in_subset)
        ]
        return self._bkd.hstack(flat_covs)

    def high_fidelity_estimator_covariance(self, nhf_samples: Array) -> Array:
        if self._W is None or self._V is None or self._tril_idx_flat is None:
            raise ValueError("set_pilot_quantities must be called first")
        cov_est = _covariance_of_variance_estimator(
            self._W[: self._nqoi**2, : self._nqoi**2],
            self._V[: self._nqoi**2, : self._nqoi**2],
            nhf_samples,
        )
        return _fancy_index_2d(cov_est, self._tril_idx_flat, self._tril_idx_flat)

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, Array]:
        """Compute covariance and W matrix from pilot samples.

        Parameters
        ----------
        pilot_values : List[Array]
            Pilot samples for each model. Each array has shape (nqoi, nsamples).

        Returns
        -------
        Tuple[Array, Array]
            Covariance matrix and W matrix.
        """
        self._check_pilot_values(pilot_values)
        nmodels = len(pilot_values)
        # Stack to (nqoi*nmodels, nsamples)
        pilot_values_stacked = self._bkd.vstack(pilot_values)
        cov = self._bkd.cov(pilot_values_stacked, rowvar=True, ddof=1)
        return cov, _get_W_from_pilot(pilot_values_stacked, nmodels, self._bkd)

    def set_pilot_quantities(self, *args: Any) -> None:
        cov, W = args[0], args[1]
        self._cov = self._bkd.asarray(cov, dtype=self._bkd.real_dtype())
        self._nmodels = self._cov.shape[0] // self._nqoi
        self._V = self._bkd.asarray(
            _get_V_from_covariance(self._cov, self._nmodels, self._bkd),
            dtype=self._bkd.real_dtype(),
        )
        if W.shape != self._V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self._V.shape
            )
            raise ValueError(msg)
        self._W = self._bkd.asarray(W, dtype=self._bkd.real_dtype())
        self._set_compressed_data()

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array, Hmat: Array, hvec: Array
    ) -> Tuple[Array, Array]:
        if (
            self._V is None
            or self._W is None
            or self._lf_delta_idx is None
            or self._hf_delta_idx is None
        ):
            raise ValueError("set_pilot_quantities must be called first")
        CF, cf = _get_multioutput_acv_variance_discrepancy_covariances(
            self._V, self._W, Gmat, gvec, Hmat, hvec, self._bkd
        )
        return (
            _fancy_index_2d(CF, self._lf_delta_idx, self._lf_delta_idx),
            _fancy_index_2d(cf, self._hf_delta_idx, self._lf_delta_idx),
        )

    def _get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        Gmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0 / self._bkd.to_float(npartition_samples[0]),
            dtype=self._bkd.real_dtype(),
        )
        gvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0 / self._bkd.to_float(npartition_samples[0]),
            dtype=self._bkd.real_dtype(),
        )
        Hmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0
            / (
                self._bkd.to_float(npartition_samples[0])
                * (self._bkd.to_float(npartition_samples[0]) - 1.0)
            ),
            dtype=self._bkd.real_dtype(),
        )
        hvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0
            / (
                self._bkd.to_float(npartition_samples[0])
                * (self._bkd.to_float(npartition_samples[0]) - 1.0)
            ),
            dtype=self._bkd.real_dtype(),
        )
        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def _get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        Gmat, gvec = _get_acv_mean_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples, self._bkd
        )
        Hmat, hvec = _get_acv_variance_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples, self._bkd
        )
        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def get_pilot_quantities_subset(
        self, nmodels: int, nqoi: int, model_idx: Array, qoi_idx: Optional[Array] = None
    ) -> Tuple[Array, ...]:
        if self._cov is None or self._W is None:
            raise ValueError("set_pilot_quantities must be called first")
        if qoi_idx is None:
            qoi_idx = self._bkd.arange(nqoi)
        cov_sub = _nqoi_nqoi_subproblem(
            self._cov, nmodels, nqoi, model_idx, qoi_idx, self._bkd
        )
        W_sub = _nqoisq_nqoisq_subproblem(
            self._W, nmodels, nqoi, model_idx, qoi_idx, self._bkd
        )
        return cov_sub, W_sub

    def subset(
        self,
        model_indices: List[int],
        qoi_indices: Optional[List[int]] = None,
    ) -> "MultiOutputVariance[Array]":
        """Create statistic for a subset of models and/or QoI.

        Parameters
        ----------
        model_indices : List[int]
            Indices of models to include. Must include 0 (high-fidelity model).
        qoi_indices : List[int], optional
            Indices of QoI to include. If None, uses all QoI.

        Returns
        -------
        MultiOutputVariance
            New statistic instance for the subset.

        Raises
        ------
        ValueError
            If model_indices does not include 0.
        """
        if 0 not in model_indices:
            raise ValueError("model_indices must include 0 (high-fidelity)")

        model_idx = self._bkd.array(model_indices, dtype=self._bkd.index_dtype())

        if qoi_indices is None:
            qoi_idx = None
            new_nqoi = self._nqoi
        else:
            qoi_idx = self._bkd.array(qoi_indices, dtype=self._bkd.index_dtype())
            new_nqoi = len(qoi_indices)

        cov_sub, W_sub = self.get_pilot_quantities_subset(
            self._nmodels, self._nqoi, model_idx, qoi_idx
        )

        new_stat: MultiOutputVariance[Array] = MultiOutputVariance(new_nqoi, self._bkd)
        new_stat.set_pilot_quantities(cov_sub, W_sub)
        return new_stat

    def min_nsamples(self) -> int:
        return 2

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: Union[int, Array],
        nsamples_subset0: Union[int, Array],
        nsamples_subset1: Union[int, Array],
    ) -> Array:
        # should resemble high_fidelity_estimator_covariance()
        if self._cov is None or self._Vcomp is None or self._Wcomp is None:
            raise ValueError("must call set_pilot_quantities")
        W_ratio = nsamples_intersect / (nsamples_subset0 * nsamples_subset1)
        V_ratio = (nsamples_intersect * (nsamples_intersect - 1)) / (
            (nsamples_subset0 * (nsamples_subset0 - 1))
            * (nsamples_subset1 * (nsamples_subset1 - 1))
        )
        block = (
            _fancy_index_2d(self._Vcomp, subset0, subset1) * V_ratio
            + _fancy_index_2d(self._Wcomp, subset0, subset1) * W_ratio
        )
        return block

    def _group_acv_sigma_block_derivs(
        self,
        subset: Array,
        nsamples: Union[int, Array],
    ) -> Tuple[Array, Array]:
        r"""Sigma = V/(n(n-1)) + W/n  .

        dΣ/dn = -V(2n-1)/(n(n-1))² - W/n²
        d²Σ/dn² = 2V(3n²-3n+1)/(n(n-1))³ + 2W/n³
        """
        if self._Vcomp is None or self._Wcomp is None:
            raise ValueError("must call set_pilot_quantities")
        V = _fancy_index_2d(self._Vcomp, subset, subset)
        W = _fancy_index_2d(self._Wcomp, subset, subset)
        n = nsamples
        nm1 = n - 1
        nnm1 = n * nm1
        nnm1_sq = nnm1 * nnm1
        nnm1_cu = nnm1_sq * nnm1
        n2 = n * n
        n3 = n2 * n
        d1 = -V * (2 * n - 1) / nnm1_sq - W / n2
        d2 = 2 * V * (3 * n2 - 3 * n + 1) / nnm1_cu + 2 * W / n3
        return d1, d2


class MultiOutputMeanAndVariance(MultiOutputStatistic[Array]):
    """Statistics for computing both mean and variance."""

    def __init__(self, nqoi: int, bkd: Backend[Array]):
        super().__init__(nqoi, bkd)
        self._cov: Optional[Array] = None
        self._W: Optional[Array] = None
        self._V: Optional[Array] = None
        self._B: Optional[Array] = None
        self._tril_idx: Optional[Tuple[Array, Array]] = None
        self._tril_idx_flat: Optional[Array] = None
        self._comp_idx: Optional[Array] = None
        self._Vcomp: Optional[Array] = None
        self._Wcomp: Optional[Array] = None
        self._Bcomp: Optional[Array] = None
        self._lf_delta_idx: Optional[Array] = None
        self._hf_delta_idx: Optional[Array] = None

    def with_backend(
        self, bkd: Backend[TargetArray]
    ) -> "ACVDiscrepancyStatistic[TargetArray]":
        stat = MultiOutputMeanAndVariance(self._nqoi, bkd)
        if self._cov is not None and self._W is not None and self._B is not None:
            stat.set_pilot_quantities(
                bkd.asarray(self._cov, dtype=bkd.real_dtype()),
                bkd.asarray(self._W, dtype=bkd.real_dtype()),
                bkd.asarray(self._B, dtype=bkd.real_dtype()),
            )
        return stat

    def _set_compressed_data(self) -> None:
        # subset0 wil contain indices into lower diagonal of covariance
        # e.g. for model subset [0, 1] with 2 qoi
        # subset0 = ([0, 1, 2], [3, 4, 5])
        # ([brackets denote qoi for each model]
        # but V and W are stored in terms on indices into entire covariance
        # e.g. ([0, 1, 2, 3], [4, 5, 6, 7])
        # so when subset  ([0, 1, 2], [3, 4, 5]) comes in we want to extract
        # entries of V corresponding to rows (and columns)
        # ([0, 2, 3], [4, 6, 7])

        # get compressed V
        self._tril_idx = self._bkd.tril_indices(self._nqoi)
        assert self._tril_idx is not None
        self._tril_idx_flat = self._bkd.reshape(
            self._bkd.arange(self._nqoi**2, dtype=self._bkd.index_dtype()),
            (self._nqoi, self._nqoi),
        )[self._tril_idx[0], self._tril_idx[1]]
        self._comp_idx = self._bkd.hstack(
            [self._tril_idx_flat + ii * self._nqoi**2 for ii in range(self._nmodels)]
        )
        if self._V is None or self._W is None or self._B is None:
            raise ValueError("V, W, and B must be set before compression")
        self._Vcomp = _fancy_index_2d(self._V, self._comp_idx, self._comp_idx)
        self._Wcomp = _fancy_index_2d(self._W, self._comp_idx, self._comp_idx)
        self._Bcomp = self._B[:, self._comp_idx]

        # for acv discrepancies (must exclude first model)
        if self._nmodels == 1:
            return

        self._lf_delta_idx = self._bkd.hstack(
            [
                self._bkd.hstack(
                    (
                        self._bkd.arange(self.nqoi()),
                        self._tril_idx_flat + self.nqoi(),
                    )
                )
                + ii * (self.nqoi() ** 2 + self.nqoi())
                for ii in range(self._nmodels - 1)
            ]
        )
        self._hf_delta_idx = self._lf_delta_idx[: self.nstats()]

    def nstats(self) -> int:
        ncov = _ncovariance_stats(self.nqoi())
        if self._tril_idx_flat is not None and (self._tril_idx_flat.shape[0] != ncov):
            raise RuntimeError(
                f"pilot quantities imply {self._tril_idx_flat.shape[0]} "
                f"covariance statistics but nqoi implies {ncov}"
            )
        return self.nqoi() + ncov

    def stat_slot_indices(self, stat_name: str) -> List[int]:
        if stat_name == "mean":
            return list(range(self.nqoi()))
        if stat_name == "variance":
            return list(range(self.nqoi(), self.nstats()))
        raise ValueError(f"'{stat_name}' not available on {type(self).__name__}")

    def known_slots(self, known: KnownStatistic[Array]) -> List[int]:
        if isinstance(known, KnownMean):
            return list(range(self.nqoi()))
        if isinstance(known, KnownVariance):
            return list(range(self.nqoi(), self.nstats()))
        return super().known_slots(known)

    def check_known(self, known: Sequence[KnownStatistic[Array]]) -> None:
        """Also: a model's known mean and variance come together or not at all."""
        super().check_known(known)
        kinds: Dict[int, Set[Type[KnownStatistic[Array]]]] = {}
        for item in known:
            kinds.setdefault(item.model, set()).add(type(item))
        both: Set[Type[KnownStatistic[Array]]] = {KnownMean, KnownVariance}
        for model, present in kinds.items():
            if present != both:
                (have,) = present
                missing = (both - present).pop()
                raise ValueError(
                    f"{type(self).__name__} needs a model's known mean and "
                    f"variance together: model {model} has a known "
                    f"{have.__name__} but no {missing.__name__}"
                )

    def sample_estimate(self, values: Array) -> Array:
        """Compute sample mean and variance estimate.

        Variance component centers on the sample mean (not a known
        population mean) so the estimator covariance depends on V, W,
        and B matrices.

        Parameters
        ----------
        values : Array
            Model outputs. Shape: (nqoi*nmodels_in_subset, nsamples)

        Returns
        -------
        Array
            Mean and variance estimate. Shape: (nstats*nmodels_in_subset,)
        """
        # Note ordering of statistics must be consistent
        # with  _group_acv_sigma_block which uses all means in a group
        # and then all covariance entries
        nmodels_in_subset = values.shape[0] // self._nqoi
        # Need to compute covariances of each model in the subset
        # separately because we do not want to compute covariance of
        # entire values vector, i.e. off diagonal blocks containing covariance
        # of outputs of different models, but we only want diagonal
        # blocks
        flat_covs = [
            self._bkd.flatten(
                self._bkd.cov(
                    values[ii * self._nqoi : (ii + 1) * self._nqoi, :],
                    ddof=1,
                    rowvar=True,
                )
            )[self._tril_idx_flat]
            for ii in range(nmodels_in_subset)
        ]
        means = self._bkd.mean(values, axis=1)

        return self._bkd.hstack(
            [
                self._bkd.hstack(
                    (
                        means[ii * self._nqoi : (ii + 1) * self._nqoi],
                        flat_covs[ii],
                    )
                )
                for ii in range(nmodels_in_subset)
            ]
        )

    def high_fidelity_estimator_covariance(self, nhf_samples: Array) -> Array:
        if (
            self._cov is None
            or self._W is None
            or self._V is None
            or self._B is None
            or self._tril_idx_flat is None
        ):
            raise RuntimeError("must call set_pilot_quantities first")
        block_11 = self._cov[: self._nqoi, : self._nqoi] / nhf_samples
        cov_est = _covariance_of_variance_estimator(
            self._W[: self._nqoi**2, : self._nqoi**2],
            self._V[: self._nqoi**2, : self._nqoi**2],
            nhf_samples,
        )
        block_22 = _fancy_index_2d(cov_est, self._tril_idx_flat, self._tril_idx_flat)
        block_12 = (
            self._B[: self._nqoi, : self._nqoi**2][:, self._tril_idx_flat] / nhf_samples
        )
        return block_2x2([[block_11, block_12], [block_12.T, block_22]], self._bkd)

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, Array, Array]:
        """Compute covariance, W, and B matrices from pilot samples.

        Parameters
        ----------
        pilot_values : List[Array]
            Pilot samples for each model. Each array has shape (nqoi, nsamples).

        Returns
        -------
        Tuple[Array, Array, Array]
            Covariance, W, and B matrices.
        """
        self._check_pilot_values(pilot_values)
        nmodels = len(pilot_values)
        # Stack to (nqoi*nmodels, nsamples)
        pilot_values_stacked = self._bkd.vstack(pilot_values)
        cov = self._bkd.cov(pilot_values_stacked, rowvar=True, ddof=1)
        W = _get_W_from_pilot(pilot_values_stacked, nmodels, self._bkd)
        B = _get_B_from_pilot(pilot_values_stacked, nmodels, self._bkd)
        return cov, W, B

    def set_pilot_quantities(self, *args: Any) -> None:
        cov, W, B = args[0], args[1], args[2]
        self._cov = self._bkd.asarray(cov, dtype=self._bkd.real_dtype())
        self._nmodels = self._cov.shape[0] // self._nqoi
        self._V = self._bkd.asarray(
            _get_V_from_covariance(self._cov, self._nmodels, self._bkd),
            dtype=self._bkd.real_dtype(),
        )
        if W.shape != self._V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self._V.shape
            )
            raise ValueError(msg)
        self._W = self._bkd.asarray(W, dtype=self._bkd.real_dtype())
        B_shape = cov.shape[0], self._V.shape[1]
        if B.shape != B_shape:
            msg = "B has the wrong shape {0}. Should be {1}".format(B.shape, B_shape)
            raise ValueError(msg)
        self._B = self._bkd.asarray(B, dtype=self._bkd.real_dtype())
        self._set_compressed_data()

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array, Hmat: Array, hvec: Array
    ) -> Tuple[Array, Array]:
        if (
            self._cov is None
            or self._V is None
            or self._W is None
            or self._B is None
            or self._lf_delta_idx is None
            or self._hf_delta_idx is None
        ):
            raise RuntimeError("must call set_pilot_quantities first")
        CF, cf = _get_multioutput_acv_mean_and_variance_discrepancy_covariances(
            self._cov,
            self._V,
            self._W,
            self._B,
            Gmat,
            gvec,
            Hmat,
            hvec,
            self._bkd,
        )
        return (
            _fancy_index_2d(CF, self._lf_delta_idx, self._lf_delta_idx),
            _fancy_index_2d(cf, self._hf_delta_idx, self._lf_delta_idx),
        )

    def _get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        Gmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0 / self._bkd.to_float(npartition_samples[0]),
            dtype=self._bkd.real_dtype(),
        )
        gvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0 / self._bkd.to_float(npartition_samples[0]),
            dtype=self._bkd.real_dtype(),
        )
        Hmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0
            / (
                self._bkd.to_float(npartition_samples[0])
                * (self._bkd.to_float(npartition_samples[0]) - 1.0)
            ),
            dtype=self._bkd.real_dtype(),
        )
        hvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0
            / (
                self._bkd.to_float(npartition_samples[0])
                * (self._bkd.to_float(npartition_samples[0]) - 1.0)
            ),
            dtype=self._bkd.real_dtype(),
        )
        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def _get_acv_discrepancy_covariances(
        self, allocation_mat: Array, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        Gmat, gvec = _get_acv_mean_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples, self._bkd
        )
        Hmat, hvec = _get_acv_variance_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples, self._bkd
        )
        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def get_pilot_quantities_subset(
        self, nmodels: int, nqoi: int, model_idx: Array, qoi_idx: Optional[Array] = None
    ) -> Tuple[Array, ...]:
        if self._cov is None or self._W is None or self._B is None:
            raise RuntimeError("must call set_pilot_quantities first")
        if qoi_idx is None:
            qoi_idx = self._bkd.arange(nqoi)
        cov_sub = _nqoi_nqoi_subproblem(
            self._cov, nmodels, nqoi, model_idx, qoi_idx, self._bkd
        )
        W_sub = _nqoisq_nqoisq_subproblem(
            self._W, nmodels, nqoi, model_idx, qoi_idx, self._bkd
        )
        B_sub = _nqoi_nqoisq_subproblem(
            self._B, nmodels, nqoi, model_idx, qoi_idx, self._bkd
        )
        return cov_sub, W_sub, B_sub

    def subset(
        self,
        model_indices: List[int],
        qoi_indices: Optional[List[int]] = None,
    ) -> "MultiOutputMeanAndVariance[Array]":
        """Create statistic for a subset of models and/or QoI.

        Parameters
        ----------
        model_indices : List[int]
            Indices of models to include. Must include 0 (high-fidelity model).
        qoi_indices : List[int], optional
            Indices of QoI to include. If None, uses all QoI.

        Returns
        -------
        MultiOutputMeanAndVariance
            New statistic instance for the subset.

        Raises
        ------
        ValueError
            If model_indices does not include 0.
        """
        if 0 not in model_indices:
            raise ValueError("model_indices must include 0 (high-fidelity)")

        model_idx = self._bkd.array(model_indices, dtype=int)

        if qoi_indices is None:
            qoi_idx = None
            new_nqoi = self._nqoi
        else:
            qoi_idx = self._bkd.array(qoi_indices, dtype=int)
            new_nqoi = len(qoi_indices)

        cov_sub, W_sub, B_sub = self.get_pilot_quantities_subset(
            self._nmodels, self._nqoi, model_idx, qoi_idx
        )

        new_stat: MultiOutputMeanAndVariance[Array] = MultiOutputMeanAndVariance(
            new_nqoi, self._bkd
        )
        new_stat.set_pilot_quantities(cov_sub, W_sub, B_sub)
        return new_stat

    def min_nsamples(self) -> int:
        return 2

    def _mean_idx(self, subset: Array) -> Array:
        if self._tril_idx_flat is None:
            raise RuntimeError("Must call set_pilot_quantities() first")
        mean_idx = []
        cnt = 0
        # TODO this can be done once at initilization by storing
        # mean_idx_per_model as List[Array] then just indexing into list
        nstats = self.nstats()
        nmodels_in_subset = subset.shape[0] // nstats
        for ii in range(nmodels_in_subset):
            model_id = self._bkd.to_int(self._bkd.floor_divide(subset[cnt], nstats))
            mean_idx.append(
                subset[cnt : cnt + self._nqoi] - self._tril_idx_flat.shape[0] * model_id
            )
            cnt += nstats
        return self._bkd.hstack(mean_idx)

    def _var_idx(self, subset: Array) -> Array:
        if self._tril_idx_flat is None:
            raise RuntimeError("Must call set_pilot_quantities() first")
        var_idx = []
        nstats = self.nstats()
        cnt = self._nqoi  # skip means of first model
        nmodels_in_subset = subset.shape[0] // nstats
        for ii in range(nmodels_in_subset):
            model_id = self._bkd.to_int(
                self._bkd.floor_divide(subset[cnt - self._nqoi], nstats)
            )
            var_idx.append(
                subset[cnt : cnt + self._tril_idx_flat.shape[0]]
                - self._nqoi * (model_id + 1)
            )
            cnt += nstats
        return self._bkd.hstack(var_idx)

    def _assemble_joint_block(
        self,
        block_11: Array,
        block_22: Array,
        block_12: Array,
        block_21: Array,
        nmodels_row: int,
        nmodels_col: int,
    ) -> Array:
        """Reorder 2x2 block structure into interleaved [μ₀, σ²₀, μ₁, σ²₁, ...].

        The blocks are indexed by model: block_11[i*nqoi:(i+1)*nqoi, j*nqoi:(j+1)*nqoi]
        is the mean-mean covariance between models i and j.
        """
        if self._tril_idx_flat is None:
            raise RuntimeError("must call set_pilot_quantities")
        nqoi = self.nqoi()
        ncov_stats = self._tril_idx_flat.shape[0]
        rows = []
        for ii in range(nmodels_row):
            # mean row
            row = []
            for jj in range(nmodels_col):
                row.append(
                    block_11[
                        ii * nqoi : (ii + 1) * nqoi,
                        jj * nqoi : (jj + 1) * nqoi,
                    ]
                )
                row.append(
                    block_12[
                        ii * nqoi : (ii + 1) * nqoi,
                        jj * ncov_stats : (jj + 1) * ncov_stats,
                    ]
                )
            rows.append(self._bkd.hstack(row))
            # covariance row
            row = []
            for jj in range(nmodels_col):
                row.append(
                    block_21[
                        ii * ncov_stats : (ii + 1) * ncov_stats,
                        jj * nqoi : (jj + 1) * nqoi,
                    ]
                )
                row.append(
                    block_22[
                        ii * ncov_stats : (ii + 1) * ncov_stats,
                        jj * ncov_stats : (jj + 1) * ncov_stats,
                    ]
                )
            rows.append(self._bkd.hstack(row))
        return self._bkd.vstack(rows)

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: Union[int, Array],
        nsamples_subset0: Union[int, Array],
        nsamples_subset1: Union[int, Array],
    ) -> Array:
        # should resemble high_fidelity_estimator_covariance()
        if (
            self._cov is None
            or self._Vcomp is None
            or self._Wcomp is None
            or self._Bcomp is None
            or self._tril_idx_flat is None
        ):
            raise RuntimeError("must call set_pilot_quantities")

        mean_idx0 = self._mean_idx(subset0)
        mean_idx1 = self._mean_idx(subset1)
        var_idx0 = self._var_idx(subset0)
        var_idx1 = self._var_idx(subset1)
        P_MN = nsamples_intersect / (nsamples_subset0 * nsamples_subset1)
        V_ratio = (nsamples_intersect * (nsamples_intersect - 1)) / (
            (nsamples_subset0 * (nsamples_subset0 - 1))
            * (nsamples_subset1 * (nsamples_subset1 - 1))
        )

        block_11 = _fancy_index_2d(self._cov, mean_idx0, mean_idx1) * P_MN
        block_22 = (
            _fancy_index_2d(self._Vcomp, var_idx0, var_idx1) * V_ratio
            + _fancy_index_2d(self._Wcomp, var_idx0, var_idx1) * P_MN
        )
        block_12 = _fancy_index_2d(self._Bcomp, mean_idx0, var_idx1) * P_MN
        block_21 = _fancy_index_2d(self._Bcomp, mean_idx1, var_idx0).T * P_MN
        # Note ordering of statistics must be consistent with sample_estimate
        # and est._subsets which is stats model 0, stats_model 1 and so on
        # blocks 11, 12, 21, and 22 do not follow this ordering so
        # reorder
        nstats = self.nstats()
        model_ids0: Array = self._bkd.asarray(
            self._bkd.floor_divide(subset0[::nstats], nstats),
            dtype=self._bkd.index_dtype(),
        )
        model_ids1: Array = self._bkd.asarray(
            self._bkd.floor_divide(subset1[::nstats], nstats),
            dtype=self._bkd.index_dtype(),
        )
        return self._assemble_joint_block(
            block_11,
            block_22,
            block_12,
            block_21,
            model_ids0.shape[0],
            model_ids1.shape[0],
        )

    def _group_acv_sigma_block_derivs(
        self,
        subset: Array,
        nsamples: Union[int, Array],
    ) -> Tuple[Array, Array]:
        if (
            self._cov is None
            or self._Vcomp is None
            or self._Wcomp is None
            or self._Bcomp is None
            or self._tril_idx_flat is None
        ):
            raise RuntimeError("must call set_pilot_quantities")

        mean_idx = self._mean_idx(subset)
        var_idx = self._var_idx(subset)
        nstats = self.nstats()
        nmodels_in_subset = subset.shape[0] // nstats
        n = nsamples
        nm1 = n - 1
        nnm1 = n * nm1
        nnm1_sq = nnm1 * nnm1
        nnm1_cu = nnm1_sq * nnm1
        n2 = n * n
        n3 = n2 * n

        C = _fancy_index_2d(self._cov, mean_idx, mean_idx)
        V = _fancy_index_2d(self._Vcomp, var_idx, var_idx)
        W = _fancy_index_2d(self._Wcomp, var_idx, var_idx)
        B = _fancy_index_2d(self._Bcomp, mean_idx, var_idx)

        # mean-mean: C/n  =>  d1 = -C/n^2,  d2 = 2C/n^3
        d1_11 = -C / n2
        d2_11 = 2 * C / n3
        # var-var: V/(n(n-1)) + W/n
        d1_22 = -V * (2 * n - 1) / nnm1_sq - W / n2
        d2_22 = 2 * V * (3 * n2 - 3 * n + 1) / nnm1_cu + 2 * W / n3
        # mean-var: B/n  =>  d1 = -B/n^2,  d2 = 2B/n^3
        d1_12 = -B / n2
        d2_12 = 2 * B / n3
        d1_21 = d1_12.T
        d2_21 = d2_12.T

        d1 = self._assemble_joint_block(
            d1_11,
            d1_22,
            d1_12,
            d1_21,
            nmodels_in_subset,
            nmodels_in_subset,
        )
        d2 = self._assemble_joint_block(
            d2_11,
            d2_22,
            d2_12,
            d2_21,
            nmodels_in_subset,
            nmodels_in_subset,
        )
        return d1, d2
