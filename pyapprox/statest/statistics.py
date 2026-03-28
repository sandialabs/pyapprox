"""Multi-output statistics for Monte Carlo estimators.

This module provides statistics classes for computing mean, variance, and
combined mean+variance from model evaluations.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, List, Optional, Tuple

from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.cartesian import cartesian_product

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
    nsamples_intersect = bkd.zeros(
        (2 * nmodels, 2 * nmodels), dtype=bkd.double_dtype())
    for ii in range(2 * nmodels):
        mask = (bkd.asarray(allocation_mat[:, ii] == 1))
        nsamples_intersect[ii] = bkd.sum(nsubset_samples[mask], axis=0)
    return nsamples_intersect


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
    nsamples_subset = bkd.zeros((2 * nmodels, ), dtype=bkd.double_dtype())
    for ii in range(2 * nmodels):
        mask = (bkd.asarray(allocation_mat[:, ii] == 1))
        nsamples_subset[ii] = bkd.sum(npartition_samples[mask])
    return nsamples_subset


def _get_acv_mean_discrepancy_covariances_multipliers(
    allocation_mat: Array, npartition_samples: Array, bkd: Backend[Array]
) -> Tuple[Array, Array]:
    nmodels = allocation_mat.shape[0]
    if bkd.any_bool(bkd.asarray(npartition_samples < 0)):
        raise RuntimeError(
            "An entry in npartition samples {0} was negative".format(
                npartition_samples)
        )
    nsamples_intersect = _get_nsamples_intersect(
        allocation_mat, npartition_samples, bkd
    )
    nsamples_subset = _get_nsamples_subset(
        allocation_mat, npartition_samples, bkd)
    Gmat = bkd.zeros((nmodels - 1, nmodels - 1), dtype=bkd.double_dtype())
    gvec = bkd.zeros((nmodels - 1,), dtype=bkd.double_dtype())
    for ii in range(1, nmodels):
        gvec[ii - 1] = nsamples_intersect[2 * ii, 0 + 1] / (
            nsamples_subset[2 * ii] * nsamples_subset[0 + 1]
        ) - nsamples_intersect[2 * ii + 1, 0 + 1] / (
            nsamples_subset[2 * ii + 1] * nsamples_subset[0 + 1]
        )
        for jj in range(1, nmodels):
            Gmat[ii - 1, jj - 1] = (
                nsamples_intersect[2 * ii, 2 * jj]
                / (nsamples_subset[2 * ii] * nsamples_subset[2 * jj])
                - nsamples_intersect[2 * ii, 2 * jj + 1]
                / (nsamples_subset[2 * ii] * nsamples_subset[2 * jj + 1])
                - nsamples_intersect[2 * ii + 1, 2 * jj]
                / (nsamples_subset[2 * ii + 1] * nsamples_subset[2 * jj])
                + nsamples_intersect[2 * ii + 1, 2 * jj + 1]
                / (nsamples_subset[2 * ii + 1] * nsamples_subset[2 * jj + 1])
            )
    return Gmat, gvec


def _get_acv_variance_discrepancy_covariances_multipliers(
    allocation_mat: Array, npartition_samples: Array, bkd: Backend[Array]
) -> Tuple[Array, Array]:
    """
    Compute H from Equation 3.14 of Dixon et al.
    """
    nmodels = allocation_mat.shape[0]
    if bkd.any_bool(bkd.asarray(npartition_samples < 0)):
        raise RuntimeError("An entry in npartition samples was negative")
    nsamples_intersect = _get_nsamples_intersect(
        allocation_mat, npartition_samples, bkd
    )
    nsamples_subset = _get_nsamples_subset(
        allocation_mat, npartition_samples, bkd)
    Hmat = bkd.zeros((nmodels - 1, nmodels - 1), dtype=bkd.double_dtype())
    hvec = bkd.zeros((nmodels - 1,), dtype=bkd.double_dtype())

    N0 = nsamples_subset[0 + 1]
    for ii in range(1, nmodels):
        Nis_0 = nsamples_intersect[2 * ii, 0 + 1]  # N_{0\cap i\star}
        Ni_0 = nsamples_intersect[2 * ii + 1, 0 + 1]  # N_{0\cap i}$
        Nis = nsamples_subset[2 * ii]  # N_{i\star}
        Ni = nsamples_subset[2 * ii + 1]  # N_{i}
        hvec[ii - 1] = Nis_0 * (Nis_0 - 1) / (
            N0 * (N0 - 1) * Nis * (Nis - 1)
        ) - Ni_0 * (Ni_0 - 1) / (N0 * (N0 - 1) * Ni * (Ni - 1))
        for jj in range(1, nmodels):
            Nis_js = nsamples_intersect[2 * ii, 2 * jj]  # N_{i\cap j\star}
            Ni_j = nsamples_intersect[2 * ii + 1, 2 * jj + 1]  # N_{i\cap j}$
            Ni_js = nsamples_intersect[2 * ii + 1, 2 * jj]  # N_{i\cap j\star}
            Nis_j = nsamples_intersect[2 * ii, 2 * jj + 1]  # N_{i\star\cap j}$
            Njs = nsamples_subset[2 * jj]  # N_{j\star}
            Nj = nsamples_subset[2 * jj + 1]  # N_{j}
            Hmat[ii - 1, jj - 1] = (
                Nis_js * (Nis_js - 1) / (Nis * (Nis - 1) * Njs * (Njs - 1))
                - Nis_j * (Nis_j - 1) / (Nis * (Nis - 1) * Nj * (Nj - 1))
                - Ni_js * (Ni_js - 1) / (Ni * (Ni - 1) * Njs * (Njs - 1))
                + Ni_j * (Ni_j - 1) / (Ni * (Ni - 1) * Nj * (Nj - 1))
            )
    return Hmat, hvec


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
    discp_cov = bkd.empty(
        (nqoi * (nmodels - 1), nqoi * (nmodels - 1)), dtype=bkd.double_dtype()
    )
    discp_vec = bkd.empty((nqoi, nqoi * (nmodels - 1)),
                          dtype=bkd.double_dtype())
    for ii in range(nmodels - 1):
        discp_cov[ii * nqoi: (ii + 1) * nqoi, ii * nqoi: (ii + 1) * nqoi] = (
            Gmat[ii, ii]
            * (
                cov[
                    (ii + 1) * nqoi: (ii + 2) * nqoi,
                    (ii + 1) * nqoi: (ii + 2) * nqoi,
                ]
            )
        )
        discp_vec[:, ii * nqoi: (ii + 1) * nqoi] = (
            gvec[ii] * cov[:nqoi, (ii + 1) * nqoi: (ii + 2) * nqoi]
        )
        for jj in range(ii + 1, nmodels - 1):
            discp_cov[ii * nqoi: (ii + 1) * nqoi, jj * nqoi: (jj + 1) * nqoi] = (
                Gmat[ii, jj]
                * (
                    cov[
                        (ii + 1) * nqoi: (ii + 2) * nqoi,
                        (jj + 1) * nqoi: (jj + 2) * nqoi,
                    ]
                )
            )
            discp_cov[jj * nqoi: (jj + 1) * nqoi, ii * nqoi: (ii + 1) * nqoi] = (
                discp_cov[ii * nqoi: (ii + 1) * nqoi,
                          jj * nqoi: (jj + 1) * nqoi].T
            )
    return discp_cov, discp_vec


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
    discp_cov = bkd.empty(
        (nqsq * (nmodels - 1), nqsq * (nmodels - 1)), dtype=bkd.double_dtype()
    )
    discp_vec = bkd.empty((nqsq, nqsq * (nmodels - 1)),
                          dtype=bkd.double_dtype())
    for ii in range(nmodels - 1):
        V_ii = V[
            (ii + 1) * nqsq: (ii + 2) * nqsq,
            (ii + 1) * nqsq: (ii + 2) * nqsq,
        ]
        W_ii = W[
            (ii + 1) * nqsq: (ii + 2) * nqsq,
            (ii + 1) * nqsq: (ii + 2) * nqsq,
        ]
        V_0i = V[0:nqsq, (ii + 1) * nqsq: (ii + 2) * nqsq]
        W_0i = W[0:nqsq, (ii + 1) * nqsq: (ii + 2) * nqsq]
        discp_cov[ii * nqsq: (ii + 1) * nqsq, ii * nqsq: (ii + 1) * nqsq] = (
            Gmat[ii, ii] * W_ii + Hmat[ii, ii] * V_ii
        )
        discp_vec[:, ii * nqsq: (ii + 1) * nqsq] = gvec[ii] * \
            W_0i + hvec[ii] * V_0i
        for jj in range(ii + 1, nmodels - 1):
            V_ij = V[
                (ii + 1) * nqsq: (ii + 2) * nqsq,
                (jj + 1) * nqsq: (jj + 2) * nqsq,
            ]
            W_ij = W[
                (ii + 1) * nqsq: (ii + 2) * nqsq,
                (jj + 1) * nqsq: (jj + 2) * nqsq,
            ]
            discp_cov[ii * nqsq: (ii + 1) * nqsq, jj * nqsq: (jj + 1) * nqsq] = (
                Gmat[ii, jj] * W_ij + Hmat[ii, jj] * V_ij
            )
            discp_cov[jj * nqsq: (jj + 1) * nqsq, ii * nqsq: (ii + 1) * nqsq] = (
                discp_cov[ii * nqsq: (ii + 1) * nqsq,
                          jj * nqsq: (jj + 1) * nqsq].T
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
    CF = bkd.empty(
        (
            nqoi * (nmodels - 1) + nqsq * (nmodels - 1),
            nqoi * (nmodels - 1) + nqsq * (nmodels - 1),
        ),
        dtype=bkd.double_dtype(),
    )
    cf = bkd.empty((stride, stride * (nmodels - 1)), dtype=bkd.double_dtype())
    for ii in range(nmodels - 1):
        B_0i = B[0:nqoi, (ii + 1) * nqsq: (ii + 2) * nqsq]
        B_0i_T = B.T[0:nqsq, (ii + 1) * nqoi: (ii + 2) * nqoi]
        cf[0:nqoi, ii * stride: ii * stride + nqoi] = cf_mean[
            :, ii * nqoi: (ii + 1) * nqoi
        ]
        cf[0:nqoi, ii * stride + nqoi: (ii + 1) * stride] = gvec[ii] * B_0i
        cf[nqoi:stride, ii * stride: ii * stride + nqoi] = gvec[ii] * B_0i_T
        cf[nqoi:stride, ii * stride + nqoi: (ii + 1) * stride] = cf_var[
            :, ii * nqsq: (ii + 1) * nqsq
        ]
        for jj in range(nmodels - 1):
            B_ij = B[
                (ii + 1) * nqoi: (ii + 2) * nqoi,
                (jj + 1) * nqsq: (jj + 2) * nqsq,
            ]
            CF[
                ii * stride: ii * stride + nqoi,
                jj * stride: jj * stride + nqoi,
            ] = CF_mean[ii * nqoi: (ii + 1) * nqoi, jj * nqoi: (jj + 1) * nqoi]
            CF[
                ii * stride: ii * stride + nqoi,
                jj * stride + nqoi: (jj + 1) * stride,
            ] = Gmat[ii, jj] * B_ij
            CF[
                jj * stride + nqoi: (jj + 1) * stride,
                ii * stride: ii * stride + nqoi,
            ] = bkd.copy(
                CF[
                    ii * stride: ii * stride + nqoi,
                    jj * stride + nqoi: (jj + 1) * stride,
                ]
            ).T
            CF[
                ii * stride + nqoi: (ii + 1) * stride,
                jj * stride + nqoi: (jj + 1) * stride,
            ] = CF_var[ii * nqsq: (ii + 1) * nqsq, jj * nqsq: (jj + 1) * nqsq]
    return CF, cf


def _V_entry(cov: Array, bkd: Backend[Array]) -> Array:
    V = bkd.kron(cov, cov)
    ones = bkd.ones((cov.shape[0], 1), dtype=bkd.double_dtype())
    V += bkd.kron(bkd.kron(ones.T, cov), ones) * \
        bkd.kron(bkd.kron(ones, cov), ones.T)
    return V


def _get_V_from_covariance(cov: Array, nmodels: int, bkd: Backend[Array]) -> Array:
    nqoi = cov.shape[0] // nmodels
    V: List[List[Optional[Array]]] = [
        [None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        V[ii][ii] = _V_entry(
            cov[ii * nqoi: (ii + 1) * nqoi, ii * nqoi: (ii + 1) * nqoi],
            bkd,
        )
        for jj in range(ii + 1, nmodels):
            V[ii][jj] = _V_entry(
                cov[ii * nqoi: (ii + 1) * nqoi, jj * nqoi: (jj + 1) * nqoi],
                bkd,
            )
            # We know V[ii][jj] is not None here
            V[jj][ii] = V[ii][jj].T  # type: ignore
    return bkd.block(V)  # type: ignore


def _covariance_of_variance_estimator(W: Array, V: Array, nsamples: int) -> Array:
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
    centered_values_sq_ii = bkd.reshape(bkd.einsum(
        "kn,ln->nkl", centered_values_ii, centered_values_ii
    ), (npilot_samples, -1))
    centered_values_sq_jj = bkd.reshape(bkd.einsum(
        "kn,ln->nkl", centered_values_jj, centered_values_jj
    ), (npilot_samples, -1))
    centered_values_sq_ii_mean = bkd.mean(centered_values_sq_ii, axis=0)
    centered_values_sq_jj_mean = bkd.mean(centered_values_sq_jj, axis=0)
    centered_values_sq = bkd.reshape(bkd.einsum(
        "nk,nl->nkl",
        centered_values_sq_ii - centered_values_sq_ii_mean,
        centered_values_sq_jj - centered_values_sq_jj_mean,
    ), (npilot_samples, -1))
    mc_cov = bkd.reshape(bkd.sum(centered_values_sq, axis=0),
                         (nqoi**2, nqoi**2)) / (npilot_samples)
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
        [None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        pilot_values_ii = pilot_values[ii * nqoi: (ii + 1) * nqoi, :]
        W[ii][ii] = _W_entry(pilot_values_ii, pilot_values_ii, bkd)
        for jj in range(ii + 1, nmodels):
            pilot_values_jj = pilot_values[jj * nqoi: (jj + 1) * nqoi, :]
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
    centered_values_sq_jj = bkd.reshape(bkd.einsum(
        "kn,ln->nkl", centered_values_jj, centered_values_jj
    ), (npilot_samples, -1))
    centered_values_sq_jj_mean = bkd.mean(centered_values_sq_jj, axis=0)
    # pilot_values_ii is (nqoi, nsamples), need (nsamples, nqoi) for einsum
    pilot_values_ii_T = bkd.transpose(pilot_values_ii)
    centered_values_sq = bkd.reshape(bkd.einsum(
        "nk,nl->nkl",
        pilot_values_ii_T,
        centered_values_sq_jj - centered_values_sq_jj_mean,
    ), (npilot_samples, -1))
    mc_cov = bkd.reshape(bkd.sum(centered_values_sq, axis=0),
                         (nqoi, nqoi**2)) / (npilot_samples)
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
    B: List[List[Optional[Array]]] = [
        [None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        pilot_values_ii = pilot_values[ii * nqoi: (ii + 1) * nqoi, :]
        B[ii][ii] = _B_entry(pilot_values_ii, pilot_values_ii, bkd)
        for jj in range(ii + 1, nmodels):
            pilot_values_jj = pilot_values[jj * nqoi: (jj + 1) * nqoi, :]
            B[ii][jj] = _B_entry(pilot_values_ii, pilot_values_jj, bkd)
            B[jj][ii] = _B_entry(pilot_values_jj, pilot_values_ii, bkd)
    return bkd.block(B)  # type: ignore


def _nqoi_nqoi_subproblem(
    C: Array,
    nmodels: int,
    nqoi: int,
    model_idx: Array,
    qoi_idx: Array,
    bkd: Backend[Array],
) -> Array:
    nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
    C_new = bkd.empty((nsub_models * nsub_qoi, nsub_models * nsub_qoi))
    # We must iterate and assign because index_update only does one at a time
    # and we want to preserve autograd if possible, although here we are
    # building a new matrix.
    # Actually, fancy indexing might be better if supported.
    # But C_new[cnt1, cnt2] = C[idx1, idx2] is what was there.
    # To keep it backend-agnostic and potentially autograd-friendly,
    # we should use bkd.zeros and bkd.index_update or similar if we can,
    # but the current way is what was there.
    # Given the constraint "avoiding to_numpy() for computation" and
    # "do not use float(), int(), or .item()", let's use what's safest.
    # But Array doesn't necessarily support __setitem__ with autograd.
    # Wait, the prompt says "Ensuring all class instance variables are private
    # and have proper type hints."

    # Let's try to use fancy indexing if possible, but the original loop
    # might be more compatible with some backends.
    # However, bkd.empty and then assignment is usually not autograd friendly.
    # But pilot quantities are often not part of the autograd graph.

    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            cnt2 = 0
            idx1 = bkd.to_int(jj1 * nqoi + kk1)
            for jj2 in model_idx:
                for kk2 in qoi_idx:
                    idx2 = bkd.to_int(jj2 * nqoi + kk2)
                    C_new[cnt1, cnt2] = C[idx1, idx2]
                    cnt2 += 1
            cnt1 += 1
    return C_new


def _nqoisq_nqoisq_subproblem(
    V: Array,
    nmodels: int,
    nqoi: int,
    model_idx: Array,
    qoi_idx: Array,
    bkd: Backend[Array],
) -> Array:
    nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
    V_new = bkd.empty((nsub_models * nsub_qoi**2, nsub_models * nsub_qoi**2))
    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            for ll1 in qoi_idx:
                cnt2 = 0
                idx1 = bkd.to_int(jj1 * nqoi**2 + kk1 * nqoi + ll1)
                for jj2 in model_idx:
                    for kk2 in qoi_idx:
                        for ll2 in qoi_idx:
                            idx2 = bkd.to_int(jj2 * nqoi**2 + kk2 * nqoi + ll2)
                            V_new[cnt1, cnt2] = V[idx1, idx2]
                            cnt2 += 1
                cnt1 += 1
    return V_new


def _nqoi_nqoisq_subproblem(
    B: Array,
    nmodels: int,
    nqoi: int,
    model_idx: Array,
    qoi_idx: Array,
    bkd: Backend[Array],
) -> Array:
    nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
    B_new = bkd.empty((nsub_models * nsub_qoi, nsub_models * nsub_qoi**2))
    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            cnt2 = 0
            idx1 = bkd.to_int(jj1 * nqoi + kk1)
            for jj2 in model_idx:
                for kk2 in qoi_idx:
                    for ll2 in qoi_idx:
                        idx2 = bkd.to_int(jj2 * nqoi**2 + kk2 * nqoi + ll2)
                        B_new[cnt1, cnt2] = B[idx1, idx2]
                        cnt2 += 1
            cnt1 += 1
    return B_new


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

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nqoi(self) -> int:
        """
        The number of quantities of interest (QoI) that each model returns
        """
        return self._nqoi

    @abstractmethod
    def nstats(self) -> int:
        """The number of statistics computed"""
        raise NotImplementedError

    @abstractmethod
    def sample_estimate(self, values: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
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

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: int,
        nsamples_subset0: int,
        nsamples_subset1: int,
    ) -> Array:
        # should resemble high_fidelity_estimator_covariance()
        raise NotImplementedError

    def _check_pilot_values(self, pilot_values: List[Array]) -> None:
        if not isinstance(pilot_values, list):
            raise ValueError("pilot_values must be a list")
        for vals in pilot_values:
            if not isinstance(vals, ArrayProtocol):
                raise ValueError(
                    "pilot_values entry must be an ArrayProtocol"
                )
            if vals.ndim != 2:
                raise ValueError("pilot_values entry must be 2D array")


class MultiOutputMean(MultiOutputStatistic[Array]):
    """Statistics for computing means across multiple models and QoIs."""

    def __init__(self, nqoi: int, bkd: Backend[Array]):
        super().__init__(nqoi, bkd)
        self._cov: Optional[Array] = None

    def nstats(self) -> int:
        return self.nqoi()

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

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
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
        self._cov = self._bkd.asarray(cov, dtype=self._bkd.double_dtype())
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
            self._bkd.to_float(1.0 / npartition_samples[0]),
            dtype=self._bkd.double_dtype(),
        )
        gvec = self._bkd.full(
            (self._nmodels - 1,),
            self._bkd.to_float(1.0 / npartition_samples[0]),
            dtype=self._bkd.double_dtype(),
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
        self, nmodels: int, nqoi: int, model_idx: Array,
        qoi_idx: Optional[Array] = None
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

        model_idx = self._bkd.array(model_indices, dtype=self._bkd.int64_dtype())

        if qoi_indices is None:
            qoi_idx = None
            new_nqoi = self._nqoi
        else:
            qoi_idx = self._bkd.array(qoi_indices, dtype=self._bkd.int64_dtype())
            new_nqoi = len(qoi_indices)

        (cov_sub,) = self.get_pilot_quantities_subset(
            self._nmodels, self._nqoi, model_idx, qoi_idx
        )

        new_stat: MultiOutputMean[Array] = MultiOutputMean(new_nqoi, self._bkd)
        new_stat.set_pilot_quantities(cov_sub)
        return new_stat

    def min_nsamples(self) -> int:
        return 1

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: int,
        nsamples_subset0: int,
        nsamples_subset1: int,
    ) -> Array:
        # should resemble high_fidelity_estimator_covariance()
        if self._cov is None:
            raise ValueError("must call set_pilot_quantities")
        cov = _fancy_index_2d(self._cov, subset0, subset1)
        return cov * nsamples_intersect / (nsamples_subset0 * nsamples_subset1)

    def pilot_covariance(self) -> Array:
        if self._cov is None:
            raise ValueError("must call set_pilot_quantities")
        return self._cov


class MultiOutputVariance(MultiOutputStatistic[Array]):
    """Statistics for computing variances across multiple models and QoIs."""

    def __init__(self, nqoi: int, bkd: Backend[Array], tril: bool = True):
        super().__init__(nqoi, bkd)
        self._cov: Optional[Array] = None
        self._W: Optional[Array] = None
        self._V: Optional[Array] = None
        self._tril: bool = tril  # todo deprecated remove once testing complete
        self._tril_idx: Optional[Tuple[Array, Array]] = None
        self._tril_idx_flat: Optional[Array] = None
        self._comp_idx: Optional[Array] = None
        self._Vcomp: Optional[Array] = None
        self._Wcomp: Optional[Array] = None
        self._lf_delta_idx: Optional[Array] = None
        self._hf_delta_idx: Optional[Array] = None

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
        if self._tril:
            self._tril_idx = self._bkd.tril_indices(self._nqoi)
        else:
            self._tril_idx = (
                cartesian_product(
                    self._bkd, [self._bkd.arange(self._nqoi)] * 2
                )[[1, 0], :][0],
                cartesian_product(
                    self._bkd, [self._bkd.arange(self._nqoi)] * 2
                )[[1, 0], :][1]
            )

        self._tril_idx_flat = self._bkd.reshape(
            self._bkd.arange(self._nqoi**2, dtype=self._bkd.int64_dtype()),
            (self._nqoi, self._nqoi)
        )[self._tril_idx[0], self._tril_idx[1]]

        # for group acv
        self._comp_idx = self._bkd.hstack(
            [self._tril_idx_flat + ii * self._nqoi**2
             for ii in range(self._nmodels)]
        )
        self._Vcomp = _fancy_index_2d(self._V, self._comp_idx, self._comp_idx)
        self._Wcomp = _fancy_index_2d(self._W, self._comp_idx, self._comp_idx)

        # for acv discrepancies (must exclude first model)
        if self._nmodels == 1:
            return

        if self._tril:
            self._lf_delta_idx = self._bkd.hstack(
                [
                    self._tril_idx_flat + ii * self._nqoi**2
                    for ii in range(self._nmodels - 1)
                ]
            )
        else:
            self._lf_delta_idx = self._bkd.arange(
                self.nstats() * (self._nmodels - 1),
                dtype=self._bkd.int64_dtype())
        self._hf_delta_idx = self._lf_delta_idx[: self.nstats()]

    def nstats(self) -> int:
        if self._tril_idx_flat is None:
            if self._tril:
                return self._nqoi * (self._nqoi + 1) // 2
            return self._nqoi ** 2
        return self._tril_idx_flat.shape[0]  # self.nqoi() ** 2

    def sample_estimate(self, values: Array) -> Array:
        """Compute sample variance estimate.

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
            self._bkd.flatten(self._bkd.cov(
                values[ii * self._nqoi: (ii + 1) * self._nqoi, :],
                ddof=1,
                rowvar=True,
            ))[self._tril_idx_flat]
            for ii in range(nmodels_in_subset)
        ]
        return self._bkd.hstack(flat_covs)

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
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
        self._cov = self._bkd.asarray(cov, dtype=self._bkd.double_dtype())
        self._nmodels = self._cov.shape[0] // self._nqoi
        self._V = self._bkd.asarray(
            _get_V_from_covariance(self._cov, self._nmodels, self._bkd),
            dtype=self._bkd.double_dtype(),
        )
        if W.shape != self._V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self._V.shape
            )
            raise ValueError(msg)
        self._W = self._bkd.asarray(W, dtype=self._bkd.double_dtype())
        self._set_compressed_data()

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array, Hmat: Array, hvec: Array
    ) -> Tuple[Array, Array]:
        if (self._V is None or self._W is None or self._lf_delta_idx is None or
                self._hf_delta_idx is None):
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
            self._bkd.to_float(1.0 / npartition_samples[0]),
            dtype=self._bkd.double_dtype(),
        )
        gvec = self._bkd.full(
            (self._nmodels - 1,),
            self._bkd.to_float(1.0 / npartition_samples[0]),
            dtype=self._bkd.double_dtype(),
        )
        Hmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            self._bkd.to_float(
                1.0 / (npartition_samples[0] * (npartition_samples[0] - 1))),
            dtype=self._bkd.double_dtype(),
        )
        hvec = self._bkd.full(
            (self._nmodels - 1,),
            self._bkd.to_float(
                1.0 / (npartition_samples[0] * (npartition_samples[0] - 1))),
            dtype=self._bkd.double_dtype(),
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
        self, nmodels: int, nqoi: int, model_idx: Array,
        qoi_idx: Optional[Array] = None
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

        model_idx = self._bkd.array(model_indices, dtype=self._bkd.int64_dtype())

        if qoi_indices is None:
            qoi_idx = None
            new_nqoi = self._nqoi
        else:
            qoi_idx = self._bkd.array(qoi_indices, dtype=self._bkd.int64_dtype())
            new_nqoi = len(qoi_indices)

        cov_sub, W_sub = self.get_pilot_quantities_subset(
            self._nmodels, self._nqoi, model_idx, qoi_idx
        )

        new_stat: MultiOutputVariance[Array] = MultiOutputVariance(
            new_nqoi, self._bkd, tril=self._tril
        )
        new_stat.set_pilot_quantities(cov_sub, W_sub)
        return new_stat

    def min_nsamples(self) -> int:
        return 1

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: int,
        nsamples_subset0: int,
        nsamples_subset1: int,
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


class MultiOutputMeanAndVariance(MultiOutputStatistic[Array]):
    """Statistics for computing both mean and variance."""

    def __init__(self, nqoi: int, bkd: Backend[Array], tril: bool = True):
        super().__init__(nqoi, bkd)
        self._cov: Optional[Array] = None
        self._W: Optional[Array] = None
        self._V: Optional[Array] = None
        self._B: Optional[Array] = None
        self._tril = tril  # todo deprecated remove once testing complete
        self._tril_idx: Optional[Tuple[Array, Array]] = None
        self._tril_idx_flat: Optional[Array] = None
        self._comp_idx: Optional[Array] = None
        self._Vcomp: Optional[Array] = None
        self._Wcomp: Optional[Array] = None
        self._Bcomp: Optional[Array] = None
        self._lf_delta_idx: Optional[Array] = None
        self._hf_delta_idx: Optional[Array] = None

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
        if self._tril:
            self._tril_idx = self._bkd.tril_indices(self._nqoi)
        else:
            self._tril_idx = (
                cartesian_product(
                    self._bkd, [self._bkd.arange(self._nqoi)] * 2
                )[[1, 0], :][0],
                cartesian_product(
                    self._bkd, [self._bkd.arange(self._nqoi)] * 2
                )[[1, 0], :][1]
            )
        assert self._tril_idx is not None
        self._tril_idx_flat = self._bkd.reshape(
            self._bkd.arange(self._nqoi**2, dtype=self._bkd.int64_dtype()),
            (self._nqoi, self._nqoi)
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

        if self._tril:
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
        else:
            self._lf_delta_idx = self._bkd.arange(self.nstats() * (self._nmodels - 1))
        self._hf_delta_idx = self._lf_delta_idx[: self.nstats()]

    def nstats(self) -> int:
        return self.nqoi() + self._tril_idx_flat.shape[0]

    def sample_estimate(self, values: Array) -> Array:
        """Compute sample mean and variance estimate.

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
            self._bkd.cov(
                values[ii * self._nqoi : (ii + 1) * self._nqoi, :],
                ddof=1,
                rowvar=True,
            ).flatten()[self._tril_idx_flat]
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

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
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
        self._cov = self._bkd.asarray(cov, dtype=self._bkd.double_dtype())
        self._nmodels = self._cov.shape[0] // self._nqoi
        self._V = self._bkd.asarray(
            _get_V_from_covariance(self._cov, self._nmodels, self._bkd),
            dtype=self._bkd.double_dtype(),
        )
        if W.shape != self._V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self._V.shape
            )
            raise ValueError(msg)
        self._W = self._bkd.asarray(W, dtype=self._bkd.double_dtype())
        B_shape = cov.shape[0], self._V.shape[1]
        if B.shape != B_shape:
            msg = "B has the wrong shape {0}. Should be {1}".format(B.shape, B_shape)
            raise ValueError(msg)
        self._B = self._bkd.asarray(B, dtype=self._bkd.double_dtype())
        self._set_compressed_data()

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array, Hmat: Array, hvec: Array
    ) -> Tuple[Array, Array]:
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
            self._bkd.to_float(1.0 / npartition_samples[0]),
            dtype=self._bkd.double_dtype(),
        )
        gvec = self._bkd.full(
            (self._nmodels - 1,),
            self._bkd.to_float(1.0 / npartition_samples[0]),
            dtype=self._bkd.double_dtype(),
        )
        Hmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            self._bkd.to_float(
                1.0 / (npartition_samples[0] * (npartition_samples[0] - 1))),
            dtype=self._bkd.double_dtype(),
        )
        hvec = self._bkd.full(
            (self._nmodels - 1,),
            self._bkd.to_float(
                1.0 / (npartition_samples[0] * (npartition_samples[0] - 1))),
            dtype=self._bkd.double_dtype(),
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
        self, nmodels: int, nqoi: int, model_idx: Array,
        qoi_idx: Optional[Array] = None
    ) -> Tuple[Array, ...]:
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
            new_nqoi, self._bkd, tril=self._tril
        )
        new_stat.set_pilot_quantities(cov_sub, W_sub, B_sub)
        return new_stat

    def min_nsamples(self) -> int:
        return 1

    def _mean_idx(self, subset: Array) -> Array:
        mean_idx = []
        cnt = 0
        # TODO this can be done once at initilization by storing
        # mean_idx_per_model as List[Array] then just indexing into list
        nstats = self.nstats()
        nmodels_in_subset = subset.shape[0] // nstats
        for ii in range(nmodels_in_subset):
            model_id = self._bkd.to_int(subset[cnt] // nstats)
            mean_idx.append(
                subset[cnt : cnt + self._nqoi] -
                self._tril_idx_flat.shape[0] * model_id
            )
            cnt += nstats
        return self._bkd.hstack(mean_idx)

    def _var_idx(self, subset: Array) -> Array:
        var_idx = []
        nstats = self.nstats()
        cnt = self._nqoi  # skip means of first model
        nmodels_in_subset = subset.shape[0] // nstats
        for ii in range(nmodels_in_subset):
            model_id = self._bkd.to_int(subset[cnt - self._nqoi] // nstats)
            var_idx.append(
                subset[cnt : cnt + self._tril_idx_flat.shape[0]]
                - self._nqoi * (model_id + 1)
            )
            cnt += nstats
        return self._bkd.hstack(var_idx)

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect: int,
        nsamples_subset0: int,
        nsamples_subset1: int,
    ) -> Array:
        # should resemble high_fidelity_estimator_covariance()
        if self._cov is None:
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
        nqoi = self.nqoi()
        ncov_stats = self._tril_idx_flat.shape[0]
        model_ids0 = subset0[::nstats] // nstats
        model_ids1 = subset1[::nstats] // nstats
        rows = []
        for ii in range(model_ids0.shape[0]):
            # mean row
            row = []
            for jj in range(model_ids1.shape[0]):
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
            for jj in range(len(model_ids1)):
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
