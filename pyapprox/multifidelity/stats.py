import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

from pyapprox.util.backends.template import Array, BackendMixin


def block_2x2(blocks: List[Array], bkd: BackendMixin) -> Array:
    return bkd.vstack([bkd.hstack(blocks[0]), bkd.hstack(blocks[1])])


def _get_nsamples_intersect(
    allocation_mat: Array, npartition_samples: Array, bkd
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
        (2 * nmodels, 2 * nmodels), dtype=bkd.double_type()
    )
    for ii in range(2 * nmodels):
        nsamples_intersect[ii] = (
            nsubset_samples[allocation_mat[:, ii] == 1]
        ).sum(axis=0)
    return nsamples_intersect


def _get_nsamples_subset(
    allocation_mat: Array, npartition_samples: Array, bkd
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
    nsamples_subset = bkd.zeros((2 * nmodels), dtype=bkd.double_type())
    for ii in range(2 * nmodels):
        nsamples_subset[ii] = npartition_samples[
            allocation_mat[:, ii] == 1
        ].sum()
    return nsamples_subset


def _get_acv_mean_discrepancy_covariances_multipliers(
    allocation_mat: Array, npartition_samples: Array, bkd
) -> Tuple[Array, Array]:
    nmodels = allocation_mat.shape[0]
    if bkd.any(npartition_samples < 0):
        raise RuntimeError(
            "An entry in npartition samples {0} was negative".format(
                npartition_samples
            )
        )
    nsamples_intersect = _get_nsamples_intersect(
        allocation_mat, npartition_samples, bkd
    )
    nsamples_subset = _get_nsamples_subset(
        allocation_mat, npartition_samples, bkd
    )
    Gmat = bkd.zeros((nmodels - 1, nmodels - 1), dtype=bkd.double_type())
    gvec = bkd.zeros((nmodels - 1), dtype=bkd.double_type())
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
    allocation_mat: Array, npartition_samples: Array, bkd
) -> Tuple[Array, Array]:
    """
    Compute H from Equation 3.14 of Dixon et al.
    """
    nmodels = allocation_mat.shape[0]
    if bkd.any(npartition_samples < 0):
        raise RuntimeError("An entry in npartition samples was negative")
    nsamples_intersect = _get_nsamples_intersect(
        allocation_mat, npartition_samples, bkd
    )
    nsamples_subset = _get_nsamples_subset(
        allocation_mat, npartition_samples, bkd
    )
    Hmat = bkd.zeros((nmodels - 1, nmodels - 1), dtype=bkd.double_type())
    hvec = bkd.zeros((nmodels - 1), dtype=bkd.double_type())

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
    cov: Array, Gmat: Array, gvec: Array, bkd
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
        (nqoi * (nmodels - 1), nqoi * (nmodels - 1)), dtype=bkd.double_type()
    )
    discp_vec = bkd.empty(
        (nqoi, nqoi * (nmodels - 1)), dtype=bkd.double_type()
    )
    for ii in range(nmodels - 1):
        discp_cov[
            ii * nqoi : (ii + 1) * nqoi, ii * nqoi : (ii + 1) * nqoi
        ] = Gmat[ii, ii] * (
            cov[
                (ii + 1) * nqoi : (ii + 2) * nqoi,
                (ii + 1) * nqoi : (ii + 2) * nqoi,
            ]
        )
        discp_vec[:, ii * nqoi : (ii + 1) * nqoi] = (
            gvec[ii] * cov[:nqoi, (ii + 1) * nqoi : (ii + 2) * nqoi]
        )
        for jj in range(ii + 1, nmodels - 1):
            discp_cov[
                ii * nqoi : (ii + 1) * nqoi, jj * nqoi : (jj + 1) * nqoi
            ] = Gmat[ii, jj] * (
                cov[
                    (ii + 1) * nqoi : (ii + 2) * nqoi,
                    (jj + 1) * nqoi : (jj + 2) * nqoi,
                ]
            )
            discp_cov[
                jj * nqoi : (jj + 1) * nqoi, ii * nqoi : (ii + 1) * nqoi
            ] = discp_cov[
                ii * nqoi : (ii + 1) * nqoi, jj * nqoi : (jj + 1) * nqoi
            ].T
    return discp_cov, discp_vec


def _get_multioutput_acv_variance_discrepancy_covariances(
    V: Array, W: Array, Gmat: Array, gvec: Array, Hmat: Array, hvec: Array, bkd
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
        (nqsq * (nmodels - 1), nqsq * (nmodels - 1)), dtype=bkd.double_type()
    )
    discp_vec = bkd.empty(
        (nqsq, nqsq * (nmodels - 1)), dtype=bkd.double_type()
    )
    for ii in range(nmodels - 1):
        V_ii = V[
            (ii + 1) * nqsq : (ii + 2) * nqsq,
            (ii + 1) * nqsq : (ii + 2) * nqsq,
        ]
        W_ii = W[
            (ii + 1) * nqsq : (ii + 2) * nqsq,
            (ii + 1) * nqsq : (ii + 2) * nqsq,
        ]
        V_0i = V[0:nqsq, (ii + 1) * nqsq : (ii + 2) * nqsq]
        W_0i = W[0:nqsq, (ii + 1) * nqsq : (ii + 2) * nqsq]
        discp_cov[ii * nqsq : (ii + 1) * nqsq, ii * nqsq : (ii + 1) * nqsq] = (
            Gmat[ii, ii] * W_ii + Hmat[ii, ii] * V_ii
        )
        discp_vec[:, ii * nqsq : (ii + 1) * nqsq] = (
            gvec[ii] * W_0i + hvec[ii] * V_0i
        )
        for jj in range(ii + 1, nmodels - 1):
            V_ij = V[
                (ii + 1) * nqsq : (ii + 2) * nqsq,
                (jj + 1) * nqsq : (jj + 2) * nqsq,
            ]
            W_ij = W[
                (ii + 1) * nqsq : (ii + 2) * nqsq,
                (jj + 1) * nqsq : (jj + 2) * nqsq,
            ]
            discp_cov[
                ii * nqsq : (ii + 1) * nqsq, jj * nqsq : (jj + 1) * nqsq
            ] = (Gmat[ii, jj] * W_ij + Hmat[ii, jj] * V_ij)
            discp_cov[
                jj * nqsq : (jj + 1) * nqsq, ii * nqsq : (ii + 1) * nqsq
            ] = discp_cov[
                ii * nqsq : (ii + 1) * nqsq, jj * nqsq : (jj + 1) * nqsq
            ].T
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
    bkd: BackendMixin,
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
        dtype=bkd.double_type(),
    )
    cf = bkd.empty((stride, stride * (nmodels - 1)), dtype=bkd.double_type())
    for ii in range(nmodels - 1):
        B_0i = B[0:nqoi, (ii + 1) * nqsq : (ii + 2) * nqsq]
        B_0i_T = B.T[0:nqsq, (ii + 1) * nqoi : (ii + 2) * nqoi]
        cf[0:nqoi, ii * stride : ii * stride + nqoi] = cf_mean[
            :, ii * nqoi : (ii + 1) * nqoi
        ]
        cf[0:nqoi, ii * stride + nqoi : (ii + 1) * stride] = gvec[ii] * B_0i
        cf[nqoi:stride, ii * stride : ii * stride + nqoi] = gvec[ii] * B_0i_T
        cf[nqoi:stride, ii * stride + nqoi : (ii + 1) * stride] = cf_var[
            :, ii * nqsq : (ii + 1) * nqsq
        ]
        for jj in range(nmodels - 1):
            B_ij = B[
                (ii + 1) * nqoi : (ii + 2) * nqoi,
                (jj + 1) * nqsq : (jj + 2) * nqsq,
            ]
            CF[
                ii * stride : ii * stride + nqoi,
                jj * stride : jj * stride + nqoi,
            ] = CF_mean[
                ii * nqoi : (ii + 1) * nqoi, jj * nqoi : (jj + 1) * nqoi
            ]
            CF[
                ii * stride : ii * stride + nqoi,
                jj * stride + nqoi : (jj + 1) * stride,
            ] = (
                Gmat[ii, jj] * B_ij
            )
            CF[
                jj * stride + nqoi : (jj + 1) * stride,
                ii * stride : ii * stride + nqoi,
            ] = bkd.copy(
                CF[
                    ii * stride : ii * stride + nqoi,
                    jj * stride + nqoi : (jj + 1) * stride,
                ]
            ).T
            CF[
                ii * stride + nqoi : (ii + 1) * stride,
                jj * stride + nqoi : (jj + 1) * stride,
            ] = CF_var[
                ii * nqsq : (ii + 1) * nqsq, jj * nqsq : (jj + 1) * nqsq
            ]
    return CF, cf


def _V_entry(cov: Array, bkd: BackendMixin) -> Array:
    V = bkd.kron(cov, cov)
    ones = bkd.ones((cov.shape[0], 1))
    V += bkd.kron(bkd.kron(ones.T, cov), ones) * bkd.kron(
        bkd.kron(ones, cov), ones.T
    )
    return V


def _get_V_from_covariance(cov: Array, nmodels, bkd: BackendMixin) -> Array:
    nqoi = cov.shape[0] // nmodels
    V = [[None for jj in range(nmodels)] for ii in range(nmodels)]
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
            V[jj][ii] = V[ii][jj].T
    return bkd.block(V)


def _covariance_of_variance_estimator(W: Array, V: Array, nsamples) -> Array:
    return W / nsamples + V / (nsamples * (nsamples - 1))


def _W_entry(
    pilot_values_ii: Array, pilot_values_jj: Array, bkd: BackendMixin
) -> Array:
    nqoi = pilot_values_ii.shape[1]
    npilot_samples = pilot_values_ii.shape[0]
    assert pilot_values_jj.shape[0] == npilot_samples
    means_ii = pilot_values_ii.mean(axis=0)
    means_jj = pilot_values_jj.mean(axis=0)
    centered_values_ii = pilot_values_ii - means_ii
    centered_values_jj = pilot_values_jj - means_jj
    centered_values_sq_ii = bkd.einsum(
        "nk,nl->nkl", centered_values_ii, centered_values_ii
    ).reshape(npilot_samples, -1)
    centered_values_sq_jj = bkd.einsum(
        "nk,nl->nkl", centered_values_jj, centered_values_jj
    ).reshape(npilot_samples, -1)
    centered_values_sq_ii_mean = centered_values_sq_ii.mean(axis=0)
    centered_values_sq_jj_mean = centered_values_sq_jj.mean(axis=0)
    centered_values_sq = bkd.einsum(
        "nk,nl->nkl",
        centered_values_sq_ii - centered_values_sq_ii_mean,
        centered_values_sq_jj - centered_values_sq_jj_mean,
    ).reshape(npilot_samples, -1)
    mc_cov = centered_values_sq.sum(axis=0).reshape(nqoi**2, nqoi**2) / (
        npilot_samples
    )
    return mc_cov


def _get_W_from_pilot(
    pilot_values: Array, nmodels, bkd: BackendMixin
) -> Array:
    # for one model 1 qoi this is the kurtosis
    nqoi = pilot_values.shape[1] // nmodels
    W = [[None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        pilot_values_ii = pilot_values[:, ii * nqoi : (ii + 1) * nqoi]
        W[ii][ii] = _W_entry(pilot_values_ii, pilot_values_ii, bkd)
        for jj in range(ii + 1, nmodels):
            pilot_values_jj = pilot_values[:, jj * nqoi : (jj + 1) * nqoi]
            W[ii][jj] = _W_entry(pilot_values_ii, pilot_values_jj, bkd)
            W[jj][ii] = W[ii][jj].T
    return bkd.block(W)


def _B_entry(
    pilot_values_ii: Array, pilot_values_jj: Array, bkd: BackendMixin
) -> Array:
    nqoi = pilot_values_ii.shape[1]
    npilot_samples = pilot_values_ii.shape[0]
    assert pilot_values_jj.shape[0] == npilot_samples
    means_jj = pilot_values_jj.mean(axis=0)
    centered_values_jj = pilot_values_jj - means_jj
    centered_values_sq_jj = bkd.einsum(
        "nk,nl->nkl", centered_values_jj, centered_values_jj
    ).reshape(npilot_samples, -1)
    centered_values_sq_jj_mean = centered_values_sq_jj.mean(axis=0)
    centered_values_sq = bkd.einsum(
        "nk,nl->nkl",
        pilot_values_ii,
        centered_values_sq_jj - centered_values_sq_jj_mean,
    ).reshape(npilot_samples, -1)
    mc_cov = centered_values_sq.sum(axis=0).reshape(nqoi, nqoi**2) / (
        npilot_samples
    )
    return mc_cov


def _get_B_from_pilot(
    pilot_values: Array, nmodels, bkd: BackendMixin
) -> Array:
    nqoi = pilot_values.shape[1] // nmodels
    B = [[None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        pilot_values_ii = pilot_values[:, ii * nqoi : (ii + 1) * nqoi]
        B[ii][ii] = _B_entry(pilot_values_ii, pilot_values_ii, bkd)
        for jj in range(ii + 1, nmodels):
            pilot_values_jj = pilot_values[:, jj * nqoi : (jj + 1) * nqoi]
            B[ii][jj] = _B_entry(pilot_values_ii, pilot_values_jj, bkd)
            B[jj][ii] = _B_entry(pilot_values_jj, pilot_values_ii, bkd)
    return bkd.block(B)


def _nqoi_nqoi_subproblem(
    C: Array, nmodels, nqoi, model_idx, qoi_idx, bkd: BackendMixin
) -> Array:
    nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
    C_new = bkd.empty((nsub_models * nsub_qoi, nsub_models * nsub_qoi))
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


def _nqoisq_nqoisq_subproblem(
    V: Array, nmodels, nqoi, model_idx, qoi_idx, bkd
) -> Array:
    nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
    V_new = bkd.empty((nsub_models * nsub_qoi**2, nsub_models * nsub_qoi**2))
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


def _nqoi_nqoisq_subproblem(
    B: Array, nmodels, nqoi, model_idx, qoi_idx, bkd: BackendMixin
) -> Array:
    nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
    B_new = bkd.empty((nsub_models * nsub_qoi, nsub_models * nsub_qoi**2))
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


class MultiOutputStatistic(ABC):
    def __init__(self, nqoi: int, backend: BackendMixin):
        """
        Parameters
        ----------
        nqoi : integer
            The number of quantities of interest (QoI) that each model returns
        """
        self._nqoi = nqoi
        self._bkd = backend

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
    def compute_pilot_quantities(self, pilot_values: Array) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def set_pilot_quantities(self):
        raise NotImplementedError

    @abstractmethod
    def _get_cv_discrepancy_covariances(
        self, estimator, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def _get_acv_discrepancy_covariances(
        self, estimator, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        raise NotImplementedError

    @abstractmethod
    def get_pilot_quantities_subset(
        self, nmodels: int, nqoi: int, model_idx: int, qoi_idx: int = None
    ):
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
        nsamples_intersect,
        nsamples_subset0,
        nsamples_subset1,
    ) -> Array:
        # should resemble high_fidelity_estimator_covariance()
        raise NotImplementedError

    def _check_pilot_values(self, pilot_values: List[Array]):
        if not isinstance(pilot_values, list):
            raise ValueError("pilot_values must be a list")
        for vals in pilot_values:
            if not isinstance(vals, self._bkd.array_type()):
                raise ValueError(
                    "pilot_values entry must be {0}".format(
                        self._bkd.array_type()
                    )
                )
            if vals.ndim != 2:
                raise ValueError("pilot_values entry must be 2D array")


class MultiOutputMean(MultiOutputStatistic):
    def __init__(self, nqoi: int, backend: BackendMixin):
        super().__init__(nqoi, backend)
        self._nmodels = None
        self._cov = None

    def nstats(self) -> int:
        return self.nqoi()

    def sample_estimate(self, values: Array) -> Array:
        return self._bkd.mean(values, axis=0)

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        return self._cov[: self._nqoi, : self._nqoi] / nhf_samples

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array]:
        self._check_pilot_values(pilot_values)
        pilot_values = self._bkd.hstack(pilot_values)
        return (self._bkd.cov(pilot_values, rowvar=False, ddof=1),)

    def set_pilot_quantities(self, cov: Array):
        self._cov = self._bkd.asarray(cov, dtype=self._bkd.double_type())
        self._nmodels = self._cov.shape[0] // self._nqoi

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array
    ) -> Tuple[Array, Array]:
        return _get_multioutput_acv_mean_discrepancy_covariances(
            self._cov, Gmat, gvec, self._bkd
        )

    def _get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        Gmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0 / npartition_samples[0],
            dtype=self._bkd.double_type(),
        )
        gvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0 / npartition_samples[0],
            dtype=self._bkd.double_type(),
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
        self, nmodels: int, nqoi: int, model_idx: int, qoi_idx: int = None
    ) -> Tuple[Array]:
        if qoi_idx is None:
            qoi_idx = self._bkd.arange(nqoi)
        cov_sub = _nqoi_nqoi_subproblem(
            self._cov, nmodels, nqoi, model_idx, qoi_idx, self._bkd
        )
        return (cov_sub,)

    def min_nsamples(self) -> int:
        return 1

    def _group_acv_sigma_block(
        self,
        subset0: Array,
        subset1: Array,
        nsamples_intersect,
        nsamples_subset0,
        nsamples_subset1,
    ) -> Array:
        # should resemble high_fidelity_estimator_covariance()
        if self._cov is None:
            raise RuntimeError("must call set_pilot_quantities")
        cov = self._cov[np.ix_(subset0, subset1)]
        return cov * nsamples_intersect / (nsamples_subset0 * nsamples_subset1)


class MultiOutputVariance(MultiOutputStatistic):
    def __init__(self, nqoi: int, backend: BackendMixin, tril: bool = True):
        super().__init__(nqoi, backend)

        self._nmodels = None
        self._cov = None
        self._W = None
        self._V = None
        self._tril = tril  # todo deprecated remove once testing complete

    def _set_compressed_data(self):
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
            self._tril_idx = self._bkd.cartesian_product(
                [self._bkd.arange(self._nqoi)] * 2
            )[[1, 0], :]

        self._tril_idx_flat = self._bkd.arange(
            self._nqoi**2, dtype=int
        ).reshape((self._nqoi, self._nqoi))[*self._tril_idx]

        # for group acv
        self._comp_idx = self._bkd.hstack(
            [
                self._tril_idx_flat + ii * self._nqoi**2
                for ii in range(self._nmodels)
            ]
        )
        self._Vcomp = self._V[np.ix_(self._comp_idx, self._comp_idx)]
        self._Wcomp = self._W[np.ix_(self._comp_idx, self._comp_idx)]

        # for acv discrepancies (must exclude first model)
        if self._nmodels > 1:
            self._delta_idx = self._bkd.hstack(
                [
                    self._tril_idx_flat + ii * self._nqoi**2
                    for ii in range(self._nmodels - 1)
                ]
            )

    def nstats(self) -> int:
        return self._tril_idx_flat.shape[0]  # self.nqoi() ** 2

    def sample_estimate(self, values: Array) -> Array:
        nmodels_in_subset = values.shape[1] // self._nqoi
        flat_covs = [
            self._bkd.cov(
                values[:, ii * self._nqoi : (ii + 1) * self._nqoi],
                ddof=1,
                rowvar=False,
            ).flatten()[self._tril_idx_flat]
            for ii in range(nmodels_in_subset)
        ]
        return self._bkd.hstack(flat_covs)

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        return _covariance_of_variance_estimator(
            self._W[: self._nqoi**2, : self._nqoi**2],
            self._V[: self._nqoi**2, : self._nqoi**2],
            nhf_samples,
        )[np.ix_(self._tril_idx_flat, self._tril_idx_flat)]

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, Array]:
        self._check_pilot_values(pilot_values)
        nmodels = len(pilot_values)
        pilot_values = self._bkd.hstack(pilot_values)
        cov = self._bkd.cov(pilot_values, rowvar=False, ddof=1)
        return cov, _get_W_from_pilot(pilot_values, nmodels, self._bkd)

    def set_pilot_quantities(self, cov: Array, W: Array):
        self._cov = self._bkd.asarray(cov, dtype=self._bkd.double_type())
        self._nmodels = self._cov.shape[0] // self._nqoi
        self._V = self._bkd.asarray(
            _get_V_from_covariance(self._cov, self._nmodels, self._bkd),
            dtype=self._bkd.double_type(),
        )
        if W.shape != self._V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self._V.shape
            )
            raise ValueError(msg)
        self._W = self._bkd.asarray(W, dtype=self._bkd.double_type())
        self._set_compressed_data()

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array, Hmat: Array, hvec: Array
    ) -> Tuple[Array, Array]:
        CF, cf = _get_multioutput_acv_variance_discrepancy_covariances(
            self._V, self._W, Gmat, gvec, Hmat, hvec, self._bkd
        )
        return (
            CF[np.ix_(self._delta_idx, self._delta_idx)],
            cf[np.ix_(self._tril_idx_flat, self._delta_idx)],
        )

    def _get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        Gmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0 / npartition_samples[0],
            dtype=self._bkd.double_type(),
        )
        gvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0 / npartition_samples[0],
            dtype=self._bkd.double_type(),
        )
        Hmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0 / (npartition_samples[0] * (npartition_samples[0] - 1)),
            dtype=self._bkd.double_type(),
        )
        hvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0 / (npartition_samples[0] * (npartition_samples[0] - 1)),
            dtype=self._bkd.double_type(),
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
        self, nmodels: int, nqoi: int, model_idx, qoi_idx=None
    ) -> Tuple[Array, Array]:
        if qoi_idx is None:
            qoi_idx = self._bkd.arange(nqoi)
        cov_sub = _nqoi_nqoi_subproblem(
            self._cov, nmodels, nqoi, model_idx, qoi_idx, self._bkd
        )
        W_sub = _nqoisq_nqoisq_subproblem(
            self._W, nmodels, nqoi, model_idx, qoi_idx, self._bkd
        )
        return cov_sub, W_sub

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
            raise RuntimeError("must call set_pilot_quantities")
        W_ratio = nsamples_intersect / (nsamples_subset0 * nsamples_subset1)
        V_ratio = (nsamples_intersect * (nsamples_intersect - 1)) / (
            (nsamples_subset0 * (nsamples_subset0 - 1))
            * (nsamples_subset1 * (nsamples_subset1 - 1))
        )
        block = (
            self._Vcomp[np.ix_(subset0, subset1)] * V_ratio
            + self._Wcomp[np.ix_(subset0, subset1)] * W_ratio
        )
        return block


class MultiOutputMeanAndVariance(MultiOutputStatistic):
    def __init__(self, nqoi: int, backend: BackendMixin, tril: bool = True):
        super().__init__(nqoi, backend)

        self._nmodels = None
        self._cov = None
        self._W = None
        self._V = None
        self._B = None
        self._tril = tril  # todo deprecated remove once testing complete

    def _set_compressed_data(self):
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
            self._tril_idx = self._bkd.cartesian_product(
                [self._bkd.arange(self._nqoi)] * 2
            )[[1, 0], :]
        self._tril_idx_flat = self._bkd.arange(
            self._nqoi**2, dtype=int
        ).reshape((self._nqoi, self._nqoi))[*self._tril_idx]
        self._comp_idx = self._bkd.hstack(
            [
                self._tril_idx_flat + ii * self._nqoi**2
                for ii in range(self._nmodels)
            ]
        )
        self._Vcomp = self._V[np.ix_(self._comp_idx, self._comp_idx)]
        self._Wcomp = self._W[np.ix_(self._comp_idx, self._comp_idx)]
        self._Bcomp = self._B[:, self._comp_idx]

    def nstats(self) -> int:
        return self.nqoi() + self._tril_idx_flat.shape[0]

    def sample_estimate(self, values: Array) -> Array:
        # Note ordering of statistics must be consistent
        # with  _group_acv_sigma_block which uses all means in a group
        # and then all covariance entries
        nmodels_in_subset = values.shape[1] // self._nqoi
        # Need to compute covariances of each model in the subset
        # separately because we do not want to compute covariance of
        # entire values vector, i.e. off diagonal blocks containing covariance
        # of outputs of different models, but we only want diagonal
        # blocks
        flat_covs = [
            self._bkd.cov(
                values[:, ii * self._nqoi : (ii + 1) * self._nqoi],
                ddof=1,
                rowvar=False,
            ).flatten()[self._tril_idx_flat]
            for ii in range(nmodels_in_subset)
        ]
        return self._bkd.hstack(
            [self._bkd.mean(values, axis=0), self._bkd.hstack(flat_covs)]
        )

    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        block_11 = self._cov[: self._nqoi, : self._nqoi] / nhf_samples
        block_22 = _covariance_of_variance_estimator(
            self._W[: self._nqoi**2, : self._nqoi**2],
            self._V[: self._nqoi**2, : self._nqoi**2],
            nhf_samples,
        )[np.ix_(self._tril_idx_flat, self._tril_idx_flat)]
        block_12 = (
            self._B[: self._nqoi, : self._nqoi**2][:, self._tril_idx_flat]
            / nhf_samples
        )
        return block_2x2(
            [[block_11, block_12], [block_12.T, block_22]], self._bkd
        )

    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, Array, Array]:
        self._check_pilot_values(pilot_values)
        nmodels = len(pilot_values)
        pilot_values = self._bkd.hstack(pilot_values)
        cov = self._bkd.cov(pilot_values, rowvar=False, ddof=1)
        W = _get_W_from_pilot(pilot_values, nmodels, self._bkd)
        B = _get_B_from_pilot(pilot_values, nmodels, self._bkd)
        return cov, W, B

    def set_pilot_quantities(self, cov: Array, W: Array, B: Array):
        self._cov = self._bkd.asarray(cov, dtype=self._bkd.double_type())
        self._nmodels = self._cov.shape[0] // self._nqoi
        self._V = self._bkd.asarray(
            _get_V_from_covariance(self._cov, self._nmodels, self._bkd),
            dtype=self._bkd.double_type(),
        )
        if W.shape != self._V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self._V.shape
            )
            raise ValueError(msg)
        self._W = self._bkd.asarray(W, dtype=self._bkd.double_type())
        B_shape = cov.shape[0], self._V.shape[1]
        if B.shape != B_shape:
            msg = "B has the wrong shape {0}. Should be {1}".format(
                B.shape, B_shape
            )
            raise ValueError(msg)
        self._B = self._bkd.asarray(B, dtype=self._bkd.double_type())
        self._set_compressed_data()

    def _get_discrepancy_covariances(
        self, Gmat: Array, gvec: Array, Hmat: Array, hvec: Array
    ) -> Tuple[Array, Array]:
        return _get_multioutput_acv_mean_and_variance_discrepancy_covariances(
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

    def _get_cv_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        Gmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0 / npartition_samples[0],
            dtype=self._bkd.double_type(),
        )
        gvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0 / npartition_samples[0],
            dtype=self._bkd.double_type(),
        )
        Hmat = self._bkd.full(
            (self._nmodels - 1, self._nmodels - 1),
            1.0 / (npartition_samples[0] * (npartition_samples[0] - 1)),
            dtype=self._bkd.double_type(),
        )
        hvec = self._bkd.full(
            (self._nmodels - 1,),
            1.0 / (npartition_samples[0] * (npartition_samples[0] - 1)),
            dtype=self._bkd.double_type(),
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
        self, nmodels: int, nqoi: int, model_idx: int, qoi_idx: int = None
    ) -> Tuple[Array, Array, Array]:
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

    def min_nsamples(self) -> int:
        return 1

    # LEAVE mean_idx_full and var_idx full here for now so I can tranlate
    # example to new indexing scheme that only computes estimator covariance
    # of lowerdiagonal terms of qoi covariance + mean of qoi
    # these commented functions assume computing entire covariance qoi and mean

    # def _mean_idx_full(self, subset: Array) -> Array:
    #     # comptue indices assuming subset indexes into full covariance
    #     # matrix of qoi
    #     # get index into mean covariance array of subset defined over
    #     # both model means and variances for each qoi
    #     # subsets include indices into each statistic
    #     # For three models with two stats are
    #     # mean1,mean2,cov11,cov12,cov21,cov22
    #     # possible entries in subsets are
    #     # (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
    #     # split by models each bracket () corresponds to a model
    #     # (0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11), (12, 13, 14, 15, 16, 17)
    #     # split by models and mean and covr for each qoi
    #     # split by models each bracket [] corresponds to mean and {} to var
    #     # ([0, 1], {2, 3, 4, 5}), ([6, 7], {8, 9, 10, 11}),
    #     # ([12, 13],{14, 15, 16, 17})
    #     # so for model subset (1, 2) will be
    #     # ([6, 7], {8, 9, 10, 11}), ([12, 13], {14, 15, 16, 17})
    #     # index associated with mean covariances will be
    #     # i = [6, 7], [12, 13]
    #     # These must be adjusted to index into self._cov
    #     # j = [2, 3], [4, 5]
    #     # I.e. j = i - model_id * nqoi ** 2
    #     mean_idx = []
    #     cnt = 0
    #     # TODO this can be done once at initilization by storing
    #     # mean_idx_per_model as List[Array] then just indexing into list
    #     nmodels_in_subset = len(subset) // self.nstats()
    #     for ii in range(nmodels_in_subset):
    #         model_id = subset[cnt] // self.nstats()
    #         mean_idx.append(
    #             subset[cnt : cnt + self._nqoi] - self._nqoi**2 * model_id
    #         )
    #         # first term in sum moves to end of mean stats of all qoi
    #         # second term in sum moves to end of var stats
    #         cnt += self._nqoi + self._nqoi**2
    #     return self._bkd.hstack(mean_idx)

    # def _var_idx_full(self, subset: Array) -> Array:
    #     # comptue indices assuming subset indexes into full covariance
    #     # matrix of qoi
    #     # for model subset (1, 2) will be stats are [mean], {var}
    #     # ([6, 7], {8, 9, 10, 11}), ([12, 13], {14, 15, 16, 17})
    #     # index associated with var covariances will be
    #     # i = {8, 9, 10, 11}, {14, 15, 16, 17}
    #     # These must be adjusted to index into self._V
    #     # j = {4, 5, 6, 7}, {8, 9, 10, 11}
    #     # I.e. j = i - model_id * nqoi
    #     var_idx = []
    #     cnt = self._nqoi  # skip means of first model
    #     nmodels_in_subset = len(subset) // self.nstats()
    #     for ii in range(nmodels_in_subset):
    #         model_id = subset[cnt - self._nqoi] // self.nstats()
    #         var_idx.append(
    #             subset[cnt : cnt + self._nqoi**2] - self._nqoi * (model_id + 1)
    #         )
    #         # first term in sum moves to end of var stats for current model
    #         # second term in sum moves to end of mean stats of next model
    #         cnt += self._nqoi + +self._tril_idx_flat.shape[0]
    #     return self._bkd.hstack(var_idx)

    def _mean_idx(self, subset: Array) -> Array:
        mean_idx = []
        cnt = 0
        # TODO this can be done once at initilization by storing
        # mean_idx_per_model as List[Array] then just indexing into list
        nmodels_in_subset = len(subset) // self.nstats()
        for ii in range(nmodels_in_subset):
            model_id = subset[cnt] // self.nstats()
            mean_idx.append(
                subset[cnt : cnt + self._nqoi]
                - self._tril_idx_flat.shape[0] * model_id
            )
            cnt += self.nstats()
        return self._bkd.hstack(mean_idx)

    def _var_idx(self, subset: Array) -> Array:
        var_idx = []
        cnt = self._nqoi  # skip means of first model
        nmodels_in_subset = len(subset) // self.nstats()
        for ii in range(nmodels_in_subset):
            model_id = subset[cnt - self._nqoi] // self.nstats()
            var_idx.append(
                subset[cnt : cnt + self._tril_idx_flat.shape[0]]
                - self._nqoi * (model_id + 1)
            )
            cnt += self.nstats()
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

        block_11 = self._cov[np.ix_(mean_idx0, mean_idx1)] * P_MN
        block_22 = (
            self._Vcomp[np.ix_(var_idx0, var_idx1)] * V_ratio
            + self._Wcomp[np.ix_(var_idx0, var_idx1)] * P_MN
        )
        block_12 = self._Bcomp[np.ix_(mean_idx0, var_idx1)] * P_MN
        block_21 = self._Bcomp[np.ix_(mean_idx1, var_idx0)].T * P_MN
        # Note ordering of statistics must be consistent with sample_estimate
        return block_2x2(
            [[block_11, block_12], [block_21, block_22]], self._bkd
        )
