import torch
import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from scipy.optimize import minimize

from pyapprox.interface.wrappers import ModelEnsemble
from pyapprox.util.utilities import get_correlation_from_covariance
from pyapprox.multifidelity.control_variate_monte_carlo import (
    get_nsamples_intersect,
    get_nsamples_subset, get_nsamples_per_model,
    separate_model_values_acv, separate_samples_per_model_acv,
    generate_samples_acv, round_nsample_ratios, bootstrap_acv_estimator,
    get_sample_allocation_matrix_acvmf, get_sample_allocation_matrix_acvis,
    get_nhf_samples, allocate_samples_mfmc,
    check_mfmc_model_costs_and_correlations,
    acv_sample_allocation_nhf_samples_constraint,
    acv_sample_allocation_nhf_samples_constraint_jac,
    acv_sample_allocation_gmf_ratio_constraint,
    acv_sample_allocation_gmf_ratio_constraint_jac,
    acv_sample_allocation_nlf_gt_nhf_ratio_constraint,
    acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac,
    get_acv_initial_guess, get_npartition_samples_mlmc,
    get_sample_allocation_matrix_mlmc, allocate_samples_mlmc,
    get_sample_allocation_matrix_mfmc, get_npartition_samples_mfmc,
    get_acv_recursion_indices, generate_samples_and_values_mfmc)


def _V_entry(cov):
    V = np.kron(cov, cov)
    ones = np.ones((cov.shape[0], 1))
    V += (np.kron(np.kron(ones.T, cov), ones) *
          np.kron(np.kron(ones, cov), ones.T))
    return V


def get_V_from_covariance(cov, nmodels):
    nqoi = cov.shape[0] // nmodels
    V = [[None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        V[ii][ii] = _V_entry(cov[ii*nqoi:(ii+1)*nqoi, ii*nqoi:(ii+1)*nqoi])
        for jj in range(ii+1, nmodels):
            V[ii][jj] = _V_entry(cov[ii*nqoi:(ii+1)*nqoi, jj*nqoi:(jj+1)*nqoi])
            V[jj][ii] = V[ii][jj].T
    return np.block(V)


def covariance_of_variance_estimator(W, V, nsamples):
    return W/nsamples+V/(nsamples*(nsamples-1))


def _W_entry(pilot_values_ii, pilot_values_jj):
    nqoi = pilot_values_ii.shape[1]
    npilot_samples = pilot_values_ii.shape[0]
    assert pilot_values_jj.shape[0] == npilot_samples
    means_ii = pilot_values_ii.mean(axis=0)
    means_jj = pilot_values_jj.mean(axis=0)
    centered_values_ii = pilot_values_ii - means_ii
    centered_values_jj = pilot_values_jj - means_jj
    centered_values_sq_ii = np.einsum(
        'nk,nl->nkl', centered_values_ii, centered_values_ii).reshape(
            npilot_samples, -1)
    centered_values_sq_jj = np.einsum(
        'nk,nl->nkl', centered_values_jj, centered_values_jj).reshape(
            npilot_samples, -1)
    centered_values_sq_ii_mean = centered_values_sq_ii.mean(axis=0)
    centered_values_sq_jj_mean = centered_values_sq_jj.mean(axis=0)
    centered_values_sq = np.einsum(
        'nk,nl->nkl',
        centered_values_sq_ii-centered_values_sq_ii_mean,
        centered_values_sq_jj-centered_values_sq_jj_mean).reshape(
        npilot_samples, -1)
    mc_cov = centered_values_sq.sum(axis=0).reshape(nqoi**2, nqoi**2)/(
        npilot_samples)
    return mc_cov


def get_W_from_pilot(pilot_values, nmodels):
    # for one model 1 qoi this is the kurtosis
    nqoi = pilot_values.shape[1] // nmodels
    W = [[None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        pilot_values_ii = pilot_values[:, ii*nqoi:(ii+1)*nqoi]
        W[ii][ii] = _W_entry(pilot_values_ii, pilot_values_ii)
        for jj in range(ii+1, nmodels):
            pilot_values_jj = pilot_values[:, jj*nqoi:(jj+1)*nqoi]
            W[ii][jj] = _W_entry(pilot_values_ii, pilot_values_jj)
            W[jj][ii] = W[ii][jj].T
    return np.block(W)


def _B_entry(pilot_values_ii, pilot_values_jj):
    nqoi = pilot_values_ii.shape[1]
    npilot_samples = pilot_values_ii.shape[0]
    assert pilot_values_jj.shape[0] == npilot_samples
    means_jj = pilot_values_jj.mean(axis=0)
    centered_values_jj = pilot_values_jj - means_jj
    centered_values_sq_jj = np.einsum(
        'nk,nl->nkl', centered_values_jj, centered_values_jj).reshape(
            npilot_samples, -1)
    centered_values_sq_jj_mean = centered_values_sq_jj.mean(axis=0)
    centered_values_sq = np.einsum(
        'nk,nl->nkl',
        pilot_values_ii,
        centered_values_sq_jj-centered_values_sq_jj_mean).reshape(
        npilot_samples, -1)
    mc_cov = centered_values_sq.sum(axis=0).reshape(nqoi, nqoi**2)/(
        npilot_samples)
    return mc_cov


def get_B_from_pilot(pilot_values, nmodels):
    nqoi = pilot_values.shape[1] // nmodels
    B = [[None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        pilot_values_ii = pilot_values[:, ii*nqoi:(ii+1)*nqoi]
        B[ii][ii] = _B_entry(pilot_values_ii, pilot_values_ii)
        for jj in range(ii+1, nmodels):
            pilot_values_jj = pilot_values[:, jj*nqoi:(jj+1)*nqoi]
            B[ii][jj] = _B_entry(pilot_values_ii, pilot_values_jj)
            B[jj][ii] = _B_entry(pilot_values_jj, pilot_values_ii)
    return np.block(B)


def reorder_allocation_matrix_acvgmf(allocation_mat, nsamples_per_model,
                                     recursion_index):
    """
    Allocation matrix is the reference sample allocation

    Must make sure that allocation matrix used for sample allocation and
    computing estimated variances has the largest sample sizes containing
    the largest subset

    """
    # WARNING Will only work for acvmf and not acvgis
    II = np.unique(nsamples_per_model[1:].detach().numpy(),
                   return_inverse=True)[1]+1
    tmp = allocation_mat.copy()
    tmp[:, 3::2] = allocation_mat[:, 2*II+1]
    tmp[:, 2::2] = tmp[:, 2*recursion_index+1]
    return tmp


def get_acv_mean_discrepancy_covariances_multipliers(
        reorder_allocation_mat, nsamples_per_model, get_npartition_samples):
    nmodels = reorder_allocation_mat.shape[0]
    npartition_samples = get_npartition_samples(nsamples_per_model)
    assert all(npartition_samples >= 0), (npartition_samples)
    nsamples_intersect = get_nsamples_intersect(
        reorder_allocation_mat, npartition_samples)
    nsamples_subset = get_nsamples_subset(
        reorder_allocation_mat, npartition_samples)
    Gmat = torch.zeros(
        (nmodels-1, nmodels-1), dtype=torch.double)
    gvec = torch.zeros((nmodels-1), dtype=torch.double)
    for ii in range(1, nmodels):
        gvec[ii-1] = (
            nsamples_intersect[2*ii, 0+1]/(
                nsamples_subset[2*ii]*nsamples_subset[0+1]) -
            nsamples_intersect[2*ii+1, 0+1]/(
                nsamples_subset[2*ii+1]*nsamples_subset[0+1]))
        for jj in range(1, nmodels):
            Gmat[ii-1, jj-1] = (
                nsamples_intersect[2*ii, 2*jj]/(
                    nsamples_subset[2*ii]*nsamples_subset[2*jj]) -
                nsamples_intersect[2*ii, 2*jj+1]/(
                    nsamples_subset[2*ii]*nsamples_subset[2*jj+1]) -
                nsamples_intersect[2*ii+1, 2*jj]/(
                    nsamples_subset[2*ii+1]*nsamples_subset[2*jj]) +
                nsamples_intersect[2*ii+1, 2*jj+1]/(
                    nsamples_subset[2*ii+1]*nsamples_subset[2*jj+1]))
    return Gmat, gvec


def get_acv_variance_discrepancy_covariances_multipliers(
        reorder_allocation_mat, nsamples_per_model, get_npartition_samples):
    """
    Compute H from Equation 3.14 of Dixon et al.
    """
    nmodels = reorder_allocation_mat.shape[0]
    npartition_samples = get_npartition_samples(nsamples_per_model)
    assert all(npartition_samples >= 0), (npartition_samples)
    nsamples_intersect = get_nsamples_intersect(
        reorder_allocation_mat, npartition_samples)
    nsamples_subset = get_nsamples_subset(
        reorder_allocation_mat, npartition_samples)
    Hmat = torch.zeros(
        (nmodels-1, nmodels-1), dtype=torch.double)
    hvec = torch.zeros((nmodels-1), dtype=torch.double)

    N0 = nsamples_subset[0+1]
    for ii in range(1, nmodels):
        Nis_0 = nsamples_intersect[2*ii, 0+1]  # N_{0\cap i\star}
        Ni_0 = nsamples_intersect[2*ii+1, 0+1]  # N_{0\cap i}$
        Nis = nsamples_subset[2*ii]  # N_{i\star}
        Ni = nsamples_subset[2*ii+1]  # N_{i}
        hvec[ii-1] = (
            Nis_0*(Nis_0-1)/(N0*(N0-1)*Nis*(Nis-1)) -
            Ni_0*(Ni_0-1)/(N0*(N0-1)*Ni*(Ni-1)))
        for jj in range(1, nmodels):
            Nis_js = nsamples_intersect[2*ii, 2*jj]  # N_{i\cap j\star}
            Ni_j = nsamples_intersect[2*ii+1, 2*jj+1]  # N_{i\cap j}$
            Ni_js = nsamples_intersect[2*ii+1, 2*jj]  # N_{i\cap j\star}
            Nis_j = nsamples_intersect[2*ii, 2*jj+1]  # N_{i\star\cap j}$
            Njs = nsamples_subset[2*jj]  # N_{j\star}
            Nj = nsamples_subset[2*jj+1]  # N_{j}
            Hmat[ii-1, jj-1] = (
                Nis_js*(Nis_js-1)/(Nis*(Nis-1)*Njs*(Njs-1)) -
                Nis_j*(Nis_j-1)/(Nis*(Nis-1)*Nj*(Nj-1)) -
                Ni_js*(Ni_js-1)/(Ni*(Ni-1)*Njs*(Njs-1)) +
                Ni_j*(Ni_j-1)/(Ni*(Ni-1)*Nj*(Nj-1)))
    return Hmat, hvec


def get_multioutput_acv_mean_discrepancy_covariances(
        cov, Gmat, gvec):
    r"""
    Compute the ACV discrepancies for estimating means

    Parameters
    ----------
    cov : np.ndarray (nmodels*nqoi, nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e. covariance between its QoI
        is cov[:nqoi, :nqoi]

    Gmat : np.ndarray (nmodels, nmodels)
        Encodes sample partition into mean-based delta covariances

    gvec : np.ndarray (nmodels, nmodels)
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
    nmodels = len(gvec)+1
    nqoi = cov.shape[0]//nmodels
    discp_cov = torch.empty((nqoi*(nmodels-1), nqoi*(nmodels-1)),
                            dtype=torch.double)
    discp_vec = torch.empty((nqoi, nqoi*(nmodels-1)),
                            dtype=torch.double)
    for ii in range(nmodels-1):
        discp_cov[ii*nqoi:(ii+1)*nqoi, ii*nqoi:(ii+1)*nqoi] = Gmat[ii, ii]*(
            cov[(ii+1)*nqoi:(ii+2)*nqoi, (ii+1)*nqoi:(ii+2)*nqoi])
        discp_vec[:, ii*nqoi:(ii+1)*nqoi] = (
            gvec[ii]*cov[:nqoi, (ii+1)*nqoi:(ii+2)*nqoi])
        for jj in range(ii+1, nmodels-1):
            discp_cov[ii*nqoi:(ii+1)*nqoi, jj*nqoi:(jj+1)*nqoi] = (
                Gmat[ii, jj]*(
                    cov[(ii+1)*nqoi:(ii+2)*nqoi, (jj+1)*nqoi:(jj+2)*nqoi]))
            discp_cov[jj*nqoi:(jj+1)*nqoi, ii*nqoi:(ii+1)*nqoi] = (
                discp_cov[ii*nqoi:(ii+1)*nqoi, jj*nqoi:(jj+1)*nqoi].T)
    return discp_cov, discp_vec


def get_multioutput_acv_variance_discrepancy_covariances(
        V, W, Gmat, gvec, Hmat, hvec):
    r"""
    Compute the ACV discrepancies for estimating variance

    Parameters
    ----------
    V : np.ndarray (nmodels*nqoi**2, nmodels**nqoi**2)
        Kroneker product of flattened covariance with itself

    W : np.ndarray (nmodels*nqoi**2, nmodels**nqoi**2)
        Covariance of Kroneker product of mean-centered values

    Gmat : np.ndarray (nmodels, nmodels)
        Encodes sample partition into mean-based delta covariances

    gvec : np.ndarray (nmodels, nmodels)
        Encodes sample partition into covariances between high-fidelity mean
        and deltas

    Hmat : np.ndarray (nmodels, nmodels)
        Encodes sample partition into variance-based delta covariances

    hvec : np.ndarray (nmodels, nmodels)
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
    nmodels = len(gvec)+1
    nqsq = V.shape[0]//nmodels
    discp_cov = torch.empty(
        (nqsq*(nmodels-1), nqsq*(nmodels-1)), dtype=torch.double)
    discp_vec = torch.empty((nqsq, nqsq*(nmodels-1)), dtype=torch.double)
    for ii in range(nmodels-1):
        V_ii = V[(ii+1)*nqsq:(ii+2)*nqsq, (ii+1)*nqsq:(ii+2)*nqsq]
        W_ii = W[(ii+1)*nqsq:(ii+2)*nqsq, (ii+1)*nqsq:(ii+2)*nqsq]
        V_0i = V[0:nqsq, (ii+1)*nqsq:(ii+2)*nqsq]
        W_0i = W[0:nqsq, (ii+1)*nqsq:(ii+2)*nqsq]
        discp_cov[ii*nqsq:(ii+1)*nqsq, ii*nqsq:(ii+1)*nqsq] = (
            Gmat[ii, ii]*W_ii + Hmat[ii, ii]*V_ii)
        discp_vec[:, ii*nqsq:(ii+1)*nqsq] = gvec[ii]*W_0i+hvec[ii]*V_0i
        for jj in range(ii+1, nmodels-1):
            V_ij = V[(ii+1)*nqsq:(ii+2)*nqsq, (jj+1)*nqsq:(jj+2)*nqsq]
            W_ij = W[(ii+1)*nqsq:(ii+2)*nqsq, (jj+1)*nqsq:(jj+2)*nqsq]
            discp_cov[ii*nqsq:(ii+1)*nqsq, jj*nqsq:(jj+1)*nqsq] = (
                Gmat[ii, jj]*W_ij+Hmat[ii, jj]*V_ij)
            discp_cov[jj*nqsq:(jj+1)*nqsq, ii*nqsq:(ii+1)*nqsq] = (
                discp_cov[ii*nqsq:(ii+1)*nqsq, jj*nqsq:(jj+1)*nqsq].T)
    return discp_cov, discp_vec


def torch_2x2_block(blocks):
    return torch.vstack(
        [torch.hstack(blocks[0]),
         torch.hstack(blocks[1])])


# def get_multioutput_acv_mean_and_variance_discrepancy_covariances(
#         cov, V, W, B, Gmat, gvec, Hmat, hvec):
#     CF_mean, cf_mean = get_multioutput_acv_mean_discrepancy_covariances(
#         cov, Gmat, gvec)
#     CF_var, cf_var = get_multioutput_acv_variance_discrepancy_covariances(
#         V, W, Gmat, gvec, Hmat, hvec)
#     nmodels = len(gvec)+1
#     nqoi = cov.shape[0]//nmodels
#     nqsq = V.shape[0]//nmodels
#     CF_mean_var = torch.empty(
#         (nqoi*(nmodels-1), nqsq*(nmodels-1)), dtype=torch.double)
#     cf_mean_var = torch.empty((nqoi, nqsq*(nmodels-1)), dtype=torch.double)
#     cf_mean_var_T = torch.empty((nqsq, nqoi*(nmodels-1)), dtype=torch.double)
#     for ii in range(nmodels-1):
#         B_0i = B[0:nqoi, (ii+1)*nqsq:(ii+2)*nqsq]
#         B_0i_T = B.T[0:nqsq, (ii+1)*nqoi:(ii+2)*nqoi]
#         cf_mean_var[ii*nqoi:(ii+1)*nqoi, ii*nqsq:(ii+1)*nqsq] = gvec[ii]*B_0i
#         cf_mean_var_T[ii*nqsq:(ii+1)*nqsq, ii*nqoi:(ii+1)*nqoi] = (
#             gvec[ii]*B_0i_T)
#         for jj in range(nmodels-1):
#             B_ij = B[(ii+1)*nqoi:(ii+2)*nqoi, (jj+1)*nqsq:(jj+2)*nqsq]
#             CF_mean_var[ii*nqoi:(ii+1)*nqsq, jj*nqsq:(jj+1)*nqsq] = (
#                 Gmat[ii, jj]*B_ij)
#     CF = torch_2x2_block(
#         [[CF_mean, CF_mean_var],
#          [CF_mean_var.T, CF_var]])
#     cf = torch_2x2_block([[cf_mean, cf_mean_var],
#                           [cf_mean_var_T, cf_var]])
#     return CF, cf


def get_multioutput_acv_mean_and_variance_discrepancy_covariances(
        cov, V, W, B, Gmat, gvec, Hmat, hvec):
    CF_mean, cf_mean = get_multioutput_acv_mean_discrepancy_covariances(
        cov, Gmat, gvec)
    CF_var, cf_var = get_multioutput_acv_variance_discrepancy_covariances(
        V, W, Gmat, gvec, Hmat, hvec)
    nmodels = len(gvec)+1
    nqoi = cov.shape[0]//nmodels
    nqsq = V.shape[0]//nmodels
    stride = nqoi+nqsq
    CF = torch.empty(
        (nqoi*(nmodels-1)+nqsq*(nmodels-1), nqoi*(nmodels-1)+nqsq*(nmodels-1)),
        dtype=torch.double)
    cf = torch.empty((stride, stride*(nmodels-1)), dtype=torch.double)
    for ii in range(nmodels-1):
        B_0i = B[0:nqoi, (ii+1)*nqsq:(ii+2)*nqsq]
        B_0i_T = B.T[0:nqsq, (ii+1)*nqoi:(ii+2)*nqoi]
        cf[0:nqoi, ii*stride:ii*stride+nqoi] = (
            cf_mean[:, ii*nqoi:(ii+1)*nqoi])
        cf[0:nqoi, ii*stride+nqoi:(ii+1)*stride] = (
            gvec[ii]*B_0i)
        cf[nqoi:stride, ii*stride:ii*stride+nqoi] = (
            gvec[ii]*B_0i_T)
        cf[nqoi:stride, ii*stride+nqoi:(ii+1)*stride] = (
            cf_var[:, ii*nqsq:(ii+1)*nqsq])
        for jj in range(nmodels-1):
            B_ij = B[(ii+1)*nqoi:(ii+2)*nqoi, (jj+1)*nqsq:(jj+2)*nqsq]
            CF[ii*stride:ii*stride+nqoi, jj*stride:jj*stride+nqoi] = (
                CF_mean[ii*nqoi:(ii+1)*nqoi, jj*nqoi:(jj+1)*nqoi])
            CF[ii*stride:ii*stride+nqoi, jj*stride+nqoi:(jj+1)*stride] = (
                Gmat[ii, jj]*B_ij)
            CF[jj*stride+nqoi:(jj+1)*stride, ii*stride:ii*stride+nqoi] = (
                CF[ii*stride:ii*stride+nqoi,
                   jj*stride+nqoi:(jj+1)*stride].clone().T)
            CF[ii*stride+nqoi:(ii+1)*stride, jj*stride+nqoi:(jj+1)*stride] = (
                CF_var[ii*nqsq:(ii+1)*nqsq, jj*nqsq:(jj+1)*nqsq])
    return CF, cf


def get_npartition_samples_acvmf(nsamples_per_model):
    nmodels = len(nsamples_per_model)
    II = np.unique(nsamples_per_model.detach().numpy(), return_index=True)[1]
    sort_array = nsamples_per_model[II]
    if sort_array.shape[0] < nmodels:
        pad = sort_array[-1]*torch.ones(
            (nmodels-sort_array.shape[0]), dtype=torch.double)
        sort_array = torch.hstack((sort_array, pad))
    npartition_samples = torch.hstack(
        (nsamples_per_model[0], torch.diff(sort_array)))
    return npartition_samples


def get_npartition_samples_acvis(nsamples_per_model):
    r"""
    Get the size of the subsets :math:`z_i\setminus z_i^\star, i=0\ldots, M-1`.

    # Warning this will likely not work when recursion index is not [0, 0]
    """
    npartition_samples = torch.hstack(
        (nsamples_per_model[0], nsamples_per_model[1:]-nsamples_per_model[0]))
    return npartition_samples


def log_determinant_variance(variance):
    return torch.logdet(variance)


def log_trace_variance(variance):
    return torch.log(torch.trace(variance))


def log_linear_combination_diag_variance(weights, variance):
    # must be used with partial, e.g.
    # opt_criteria = partial(log_linear_combination_diag_variance, weights)
    return torch.log(torch.trace(variance))


class ACVEstimator():
    def __init__(self, stat, costs, variable, cov, partition="mf",
                 recursion_index=None, opt_criteria=None):
        """
        Constructor.

        Parameters
        ----------
        stat : :class:`~pyapprox.multifidelity.multioutput_monte_carlo.MultiOutputStatistic`
            Object defining what statistic will be calculated

        costs : np.ndarray (nmodels)
            The relative costs of evaluating each model

        variable : :class:`~pyapprox.variables.IndependentMarginalsVariable`
            The uncertain model parameters

        cov : np.ndarray (nmodels*nqoi, nmodels)
            The covariance C between each of the models. The highest fidelity
            model is the first model, i.e. covariance between its QoI
            is cov[:nqoi, :nqoi]

        partition : string
            What sample partition scheme to use. Must be 'mf'

        recursion_index : np.ndarray (nmodels-1)
            The recusion index that specifies which ACV estimator is used

        opt_criteria : callable
            Function of the the covariance between the high-fidelity
            QoI estimators with signature

            ``opt_criteria(variance) -> float

            where variance is np.ndarray with size that depends on
            what statistics are being estimated. E.g. when estimating means
            then variance shape is (nqoi, nqoi), when estimating variances
            then variance shape is (nqoi**2, nqoi**2), when estimating mean
            and variance then shape (nqoi+nqoi**2, nqoi+nqoi**2)
        """
        self.stat = stat
        self.cov, self.costs, self.nmodels, self.nqoi = self._check_cov(
            cov, costs)
        self.variable = variable
        self.partition = partition
        self.set_recursion_index(recursion_index)
        self.optimization_criteria = self._set_optimization_criteria(
            opt_criteria)

        self.nsamples_per_model, self.optimized_variance = None, None
        self.rounded_target_cost = None
        self.model_labels = None
        self.set_initial_guess(None)
        self.set_random_state(None)

    def _check_cov(self, cov, costs):
        nmodels = len(costs)
        if cov.shape[0] % nmodels:
            print(cov.shape, costs.shape)
            raise ValueError("cov and costs are inconsistent")
        return (torch.as_tensor(cov, dtype=torch.double).clone(),
                torch.as_tensor(costs, dtype=torch.double),
                nmodels, cov.shape[0]//nmodels)

    def _set_optimization_criteria(self, opt_criteria):
        if opt_criteria is None:
            opt_criteria = log_determinant_variance
        return opt_criteria

    def set_random_state(self, random_state):
        """
        Set the state of the numpy random generator. This effects
        self.generate_samples

        Parameters
        ----------
        random_state : :class:`numpy.random.RandmState`
            Set the random state of the numpy random generator

        Notes
        -----
        To create reproducible results when running numpy.random in parallel
        must use RandomState. If not the results will be non-deterministic.
        This is happens because of a race condition. numpy.random.* uses only
        one global PRNG that is shared across all the threads without
        synchronization. Since the threads are running in parallel, at the same
        time, and their access to this global PRNG is not synchronized between
        them, they are all racing to access the PRNG state (so that the PRNG's
        state might change behind other threads' backs). Giving each thread its
        own PRNG (RandomState) solves this problem because there is no longer
        any state that's shared by multiple threads without synchronization.
        Also see new features
        https://docs.scipy.org/doc/numpy/reference/random/parallel.html
        https://docs.scipy.org/doc/numpy/reference/random/multithreading.html
        """
        self.generate_samples = partial(
                self.variable.rvs, random_state=random_state)

    def _weights(self, CF, cf):
        #  weights = -torch.linalg.solve(CF, cf.T)
        weights = -torch.linalg.multi_dot(
            (torch.linalg.pinv(CF), cf.T))
        # try:
        #     if CF.shape == (1, 1):
        #         weights = -cf.T/CF[0, 0]
        #     else:
        #         weights = -torch.linalg.solve(CF, cf.T)
        # except (np.linalg.LinAlgError, RuntimeError):
        #     weights = torch.ones(cf.shape, dtype=torch.double)*1e16
        return weights.T

    def _estimate(self, values, weights):
        nmodels = len(values)
        assert len(values) == nmodels
        # high fidelity monte carlo estimate of mean
        deltas = np.hstack(
            [self.stat.sample_estimate(values[ii][0]) -
             self.stat.sample_estimate(values[ii][1])
             for ii in range(1, nmodels)])
        est = (self.stat.sample_estimate(values[0][1]) +
               weights.numpy().dot(deltas))
        return est

    def __call__(self, values):
        r"""
        Return the value of the Monte Carlo like estimator

        Parameters
        ----------
        values : list (nmodels)
        Each entry of the list contains

        values0 : np.ndarray (num_samples_i0,num_qoi)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

        values1: np.ndarray (num_samples_i1,num_qoi)
            Evaluations used compute the approximate
            mean :math:`\mu_{i,r_iN}` of the low fidelity models.

        Returns
        -------
        est : np.ndarray (nqoi, nqoi)
            The covariance of the estimator values for
            each high-fidelity model QoI
        """
        CF, cf = self.stat.get_discrepancy_covariances(
            self, self.nsamples_per_model)
        weights = self._weights(CF, cf)
        return self._estimate(values, weights)

    def __repr__(self):
        if self.optimized_variance is None:
            return "{0}(stat={1})".format(self.__class__.__name__, self.stat)
        return "{0}(stat={1}, variance={2:.3g}, target_cost={3:.3g})".format(
            self.__class__.__name__, self.stat, self.optimized_variance,
            self.rounded_target_cost)

    def set_optimized_params(self, nsample_ratios, rounded_target_cost,
                             optimized_variance):
        """
        Set the parameters needed to generate samples for evaluating the
        estimator

        nsample_ratios : np.ndarray (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
            the number of samples in the two different discrepancies involving
            the ith model.

        rounded_target_cost : float
            The cost of the new sample allocation

        optimized_variance : float
            The variance of the estimator using the integer sample allocations
        """
        self.nsample_ratios = nsample_ratios
        self.rounded_target_cost = rounded_target_cost
        self.nsamples_per_model = get_nsamples_per_model(
            self.rounded_target_cost, self.costs, self.nsample_ratios,
            True).astype(int)
        self.optimized_variance = optimized_variance

    def _objective(self, target_cost, x, return_grad=True):
        # return_grad argument used for testing with finte difference
        ratios = torch.tensor(x, dtype=torch.double)
        if return_grad:
            ratios.requires_grad = True
        variance = self.get_variance(target_cost, ratios)
        val = self.optimization_criteria(variance)
        if not return_grad:
            return val.item()
        val.backward()
        grad = ratios.grad.detach().numpy().copy()
        ratios.grad.zero_()
        return val.item(), grad

    def set_initial_guess(self, initial_guess):
        if initial_guess is not None:
            self.initial_guess = torch.as_tensor(
                initial_guess, dtype=torch.double)
        else:
            self.initial_guess = None

    def _allocate_samples_opt_slsqp(
            self, costs, target_cost, initial_guess, optim_options, cons):
        if optim_options is None:
            optim_options = {'disp': True, 'ftol': 1e-10,
                             'maxiter': 10000, 'iprint': 0}

        if target_cost < costs.sum():
            msg = "Target cost does not allow at least one sample from "
            msg += "each model"
            raise ValueError(msg)

        nmodels = len(costs)
        nunknowns = len(initial_guess)
        # bounds = [(1.0, 1e10)]*nunknowns
        # bounds = [(1.0, np.ceil(target_cost/cost)) for cost in costs[1:]]
        max_nhf = target_cost/costs[0]
        bounds = [(1+1/(max_nhf),
                   np.ceil(target_cost/cost)) for cost in costs[1:]]
        assert nunknowns == nmodels-1

        # constraint
        # nhf*r-nhf >= 1
        # nhf*(r-1) >= 1
        # r-1 >= 1/nhf
        # r >= 1+1/nhf
        # smallest lower bound whenn nhf = max_nhf

        return_grad = True
        method = "SLSQP"
        # method = "trust-constr"
        # print(optim_options)
        # del optim_options['ftol']
        # del optim_options['iprint']
        # optim_options["maxiter"] = 10000
        # optim_options["gtol"] = 1e-6
        opt = minimize(
            partial(self._objective, target_cost, return_grad=return_grad),
            initial_guess, method=method, jac=return_grad,
            bounds=bounds, constraints=cons, options=optim_options)
        return opt

    def _allocate_samples_opt(self, cov, costs, target_cost,
                              cons=[],
                              initial_guess=None,
                              optim_options=None, optim_method='SLSQP'):
        initial_guess = get_acv_initial_guess(
            initial_guess, cov, costs, target_cost)
        assert optim_method == "SLSQP"
        opt = self._allocate_samples_opt_slsqp(
            costs, target_cost, initial_guess, optim_options, cons)
        nsample_ratios = torch.as_tensor(opt.x)
        if not opt.success:
            raise RuntimeError('SLSQP optimizer failed'+f'{opt}')
        else:
            val = self.get_variance(target_cost, nsample_ratios)
        return nsample_ratios, val

    def _allocate_samples(self, target_cost, **kwargs):
        cons = self.get_constraints(target_cost)
        opt = self._allocate_samples_opt(
            self.cov, self.costs, target_cost, cons,
            initial_guess=self.initial_guess)

        if (check_mfmc_model_costs_and_correlations(
                self.costs,
                get_correlation_from_covariance(self.cov.numpy())) and
                len(self.cov) == len(self.costs)):
            # second condition above will not be true for multiple qoi
            mfmc_initial_guess = torch.as_tensor(allocate_samples_mfmc(
                self.cov, self.costs, target_cost)[0], dtype=torch.double)
            opt_mfmc = self._allocate_samples_opt(
                self.cov, self.costs, target_cost, cons,
                initial_guess=mfmc_initial_guess)
            if opt_mfmc[1] < opt[1]:
                opt = opt_mfmc
        return opt

    @staticmethod
    def _scipy_wrapper(fun, xx, *args):
        # convert argument to fun to tensor before passing to fun
        return fun(torch.as_tensor(xx, dtype=torch.double), *args)

    def _get_constraints(self, target_cost):
        # Must ensure that the samples of any model acting as a recursive
        # control variate has at least one more sample than its parent.
        # Otherwise Fmat will not be invertable sample ratios are rounded to
        # integers. Fmat is not invertable when two or more sample ratios
        # are equal
        cons = [
            {'type': 'ineq',
             'fun': partial(self._scipy_wrapper,
                            acv_sample_allocation_gmf_ratio_constraint),
             'jac': partial(self._scipy_wrapper,
                            acv_sample_allocation_gmf_ratio_constraint_jac),
             'args': (ii, jj, target_cost, self.costs)}
            for ii, jj in zip(range(1, self.nmodels), self.recursion_index)
            if jj > 0]
        # Ensure that all low-fidelity models have at least one more sample
        # than high-fidelity model. Otherwise Fmat will not be invertable after
        # rounding to integers
        cons += [
            {'type': 'ineq',
             'fun': partial(
                 self._scipy_wrapper,
                 acv_sample_allocation_nlf_gt_nhf_ratio_constraint),
             'jac': partial(
                 self._scipy_wrapper,
                 acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac),
             'args': (ii, target_cost, self.costs)}
            for ii in range(1, self.nmodels)]
        return cons

    def get_constraints(self, target_cost):
        cons = [{'type': 'ineq',
                 'fun': partial(
                     self._scipy_wrapper,
                     acv_sample_allocation_nhf_samples_constraint),
                 'jac': partial(
                     self._scipy_wrapper,
                     acv_sample_allocation_nhf_samples_constraint_jac),
                 'args': (target_cost, self.costs)}]
        cons += self._get_constraints(target_cost)
        return cons

    def allocate_samples(self, target_cost):
        """
        Determine the samples (integers) that must be allocated to
        each model to compute the Monte Carlo like estimator
    
        Parameters
        ----------
        target_cost : float
            The total cost budget

        Returns
        -------
        nsample_ratios : np.ndarray (nmodels-1, dtype=int)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
            the number of samples in the two different discrepancies involving
            the ith model.

        variance : float
            The variance of the estimator using the integer sample allocations

        rounded_target_cost : float
            The cost of the new sample allocation
        """
        nsample_ratios, obj_val = self._allocate_samples(
            target_cost)
        nsample_ratios = nsample_ratios.detach().numpy()
        nsample_ratios, rounded_target_cost = round_nsample_ratios(
            target_cost, self.costs.numpy(), nsample_ratios)
        nsample_ratios = torch.as_tensor(nsample_ratios, dtype=torch.double)
        variance = self.get_variance(rounded_target_cost, nsample_ratios)
        self.set_optimized_params(
            nsample_ratios, rounded_target_cost, variance)
        return nsample_ratios, variance, rounded_target_cost

    def _get_npartition_samples(self, nsamples_per_model):
        if self.partition == "mf":
            return get_npartition_samples_acvmf(nsamples_per_model)
        # if self.partition == "is":
        #     return get_npartition_samples_acvis(nsamples_per_model)
        raise ValueError("partition must one of ['mf']")

    def _create_allocation_matrix(self):
        if self.partition == "mf":
            self.allocation_mat = get_sample_allocation_matrix_acvmf(
                np.zeros(self.nmodels-1, dtype=int))
        # elif self.partition == "is":
        #     self.allocation_mat = get_sample_allocation_matrix_acvis(
        #         np.zeros(self.nmodels-1, dtype=int))
        else:
            raise ValueError("partition must one of ['mf', 'is']")

    def _get_reordered_sample_allocation_matrix(self, nsamples_per_model):
        r"""
        Compute the reordered allocation matrix corresponding to
        self.nsamples_per_model set by set_optimized_params

        Returns
        -------
        mat : np.ndarray (nmodels, 2*nmodels)
            For columns :math:`2j, j=0,\ldots,M-1` the ith row contains a
            flag specifiying if :math:`z_i^\star\subseteq z_j^\star`
            For columns :math:`2j+1, j=0,\ldots,M-1` the ith row contains a
            flag specifiying if :math:`z_i\subseteq z_j`
        """
        if self.partition == "mf":
            return torch.as_tensor(reorder_allocation_matrix_acvgmf(
                self.allocation_mat, nsamples_per_model,
                self.recursion_index), dtype=torch.double)
        # TODO create reordering for acvis
        raise ValueError("partition must one of ['mf']")

    def set_recursion_index(self, index):
        if index is None:
            index = np.zeros(self.nmodels-1, dtype=int)
        if index.shape[0] != self.nmodels-1:
            raise ValueError("index is the wrong shape")
        self._create_allocation_matrix()
        self.recursion_index = index

    def _get_variance(self, nsamples_per_model):
        CF, cf = self.stat.get_discrepancy_covariances(
            self, nsamples_per_model)
        weights = self._weights(CF, cf)
        return (self.stat.high_fidelity_estimator_covariance(
            nsamples_per_model) + torch.linalg.multi_dot((weights, cf.T)))

    def get_variance(self, target_cost, nsample_ratios):
        """
        Get the variance of the Monte Carlo estimator from costs and cov.

        Parameters
        ----------
        target_cost : float
            The total cost budget

        nsample_ratios : np.ndarray or torch.tensor (nmodels-1)
            The sample ratios r used to specify the number of samples of the
            lower fidelity models, e.g. N_i = r_i*nhf_samples,
            i=1,...,nmodels-1

        Returns
        -------
        variance : float
            The variance of the estimator
            """
        nsamples_per_model = get_nsamples_per_model(
            target_cost, self.costs, nsample_ratios, False)
        return self._get_variance(nsamples_per_model)

    def generate_sample_allocations(self):
        """
        Returns
        -------
        samples_per_model : list (nmodels)
                The ith entry contains the set of samples
                np.narray(nvars, nsamples_ii) used to evaluate the ith model.

        partition_indices_per_model : list (nmodels)
                The ith entry contains the indices np.narray(nsamples_ii)
                mapping each sample to a sample allocation partition
        """
        npartition_samples = self._get_npartition_samples(
            self.nsamples_per_model)
        reorder_allocation_mat = self._get_reordered_sample_allocation_matrix(
            self.nsamples_per_model)
        samples_per_model, partition_indices_per_model = generate_samples_acv(
            reorder_allocation_mat, self.nsamples_per_model,
            npartition_samples, self.generate_samples)
        return samples_per_model, partition_indices_per_model

    def generate_data(self, functions):
        r"""
        Generate the samples and values needed to compute the Monte Carlo like
        estimator.

        Parameters
        ----------
        functions : list of callables
            The functions used to evaluate each model with signature

            `function(samples)->np.ndarray (nsamples, 1)`

            whre samples : np.ndarray (nvars, nsamples)

        generate_samples : callable
            Function used to generate realizations of the random variables

        Returns
        -------
        acv_samples : list (nmodels)
            List containing the samples :math:`\mathcal{Z}_{i,1}` and
            :math:`\mathcal{Z}_{i,2}` for each model :math:`i=0,\ldots,M-1`.
            The list is [[:math:`\mathcal{Z}_{0,1}`,:math:`\mathcal{Z}_{0,2}`],
            ...,[:math:`\mathcal{Z}_{M-1,1}`,:math:`\mathcal{Z}_{M-1,2}`]],
            where :math:`M` is the number of models

        acv_values : list (nmodels)
            Each entry of the list contains

        values0 : np.ndarray (num_samples_i0,num_qoi)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

        values1: np.ndarray (num_samples_i1,num_qoi)
            Evaluations used compute the approximate
            mean :math:`\mu_{i,r_iN}` of the low fidelity models.
        """
        samples_per_model, partition_indices_per_model = \
            self.generate_sample_allocations()
        if type(functions) == list:
            functions = ModelEnsemble(functions)
        values_per_model = functions.evaluate_models(samples_per_model)
        reorder_allocation_mat = self._get_reordered_sample_allocation_matrix(
            self.nsamples_per_model)
        acv_values = separate_model_values_acv(
            reorder_allocation_mat, values_per_model,
            partition_indices_per_model)
        acv_samples = separate_samples_per_model_acv(
            reorder_allocation_mat, samples_per_model,
            partition_indices_per_model)
        return acv_samples, acv_values

    def estimate_from_values_per_model(self, values_per_model,
                                       partition_indices_per_model):
        reorder_allocation_mat = self._get_reordered_sample_allocation_matrix(
            self.nsamples_per_model)
        acv_values = separate_model_values_acv(
            reorder_allocation_mat, values_per_model,
            partition_indices_per_model)
        return self(acv_values)

    def bootstrap(self, values_per_model, partition_indices_per_model,
                  nbootstraps=1000):
        return bootstrap_acv_estimator(
            values_per_model, partition_indices_per_model,
            self._get_npartition_samples(self.nsamples_per_model),
            self._get_reordered_sample_allocation_matrix(
                self.nsamples_per_model),
            self._get_approximate_control_variate_weights(), nbootstraps)


class MultiOutputStatistic(ABC):
    @abstractmethod
    def get_discrepancy_covariances(self, estimator, nsamples_per_model):
        raise NotImplementedError

    def sample_estimate(self, values):
        raise NotImplementedError

    def high_fidelity_estimator_covariance(self, nsamples_per_model):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class MultiOutputMean(MultiOutputStatistic):
    def __init__(self, nmodels, cov):
        self.cov = torch.as_tensor(cov)
        self.nqoi = self.cov.shape[0]//nmodels

    def get_discrepancy_covariances(self, estimator, nsamples_per_model):
        reorder_allocation_mat = (
            estimator._get_reordered_sample_allocation_matrix(
                nsamples_per_model))
        Gmat, gvec = get_acv_mean_discrepancy_covariances_multipliers(
            reorder_allocation_mat, nsamples_per_model,
            estimator._get_npartition_samples)
        return get_multioutput_acv_mean_discrepancy_covariances(
            self.cov, Gmat, gvec)

    def sample_estimate(self, values):
        return np.mean(values, axis=0)

    def high_fidelity_estimator_covariance(self, nsamples_per_model):
        return self.cov[:self.nqoi, :self.nqoi]/nsamples_per_model[0]


class MultiOutputVariance(MultiOutputStatistic):
    def __init__(self, nmodels, cov, W):
        self.cov = torch.as_tensor(cov)
        self.nqoi = self.cov.shape[0]//nmodels
        self.V = get_V_from_covariance(self.cov, nmodels)
        if W.shape != self.V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self.V.shape)
            raise ValueError(msg)
        self.W = W

    def get_discrepancy_covariances(self, estimator, nsamples_per_model):
        reorder_allocation_mat = (
            estimator._get_reordered_sample_allocation_matrix(
                nsamples_per_model))
        Gmat, gvec = get_acv_mean_discrepancy_covariances_multipliers(
            reorder_allocation_mat, nsamples_per_model,
            estimator._get_npartition_samples)
        Hmat, hvec = (
            get_acv_variance_discrepancy_covariances_multipliers(
                reorder_allocation_mat, nsamples_per_model,
                estimator._get_npartition_samples))
        return get_multioutput_acv_variance_discrepancy_covariances(
            self.V, self.W, Gmat, gvec, Hmat, hvec)

    def sample_estimate(self, values):
        return np.cov(values.T, ddof=1).flatten()
        # return np.cov(values.T, ddof=1)[
        #    torch.triu_indices(values.shape[1], values.shape[1]).unbind()]

    def high_fidelity_estimator_covariance(self, nsamples_per_model):
        return covariance_of_variance_estimator(
            self.W[:self.nqoi**2, :self.nqoi**2],
            self.V[:self.nqoi**2, :self.nqoi**2], nsamples_per_model[0])


class MultiOutputMeanAndVariance(MultiOutputStatistic):
    def __init__(self, nmodels, cov, W, B):
        self.cov = torch.as_tensor(cov)
        self.nqoi = self.cov.shape[0]//nmodels
        self.V = get_V_from_covariance(self.cov, nmodels)
        if W.shape != self.V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self.V.shape)
            raise ValueError(msg)
        self.W = W
        B_shape = cov.shape[0], self.V.shape[1]
        if B.shape != B_shape:
            msg = "B has the wrong shape {0}. Should be {1}".format(
                B.shape, B_shape)
            raise ValueError(msg)
        self.B = B

    def get_discrepancy_covariances(self, estimator, nsamples_per_model):
        reorder_allocation_mat = (
            estimator._get_reordered_sample_allocation_matrix(
                nsamples_per_model))
        Gmat, gvec = get_acv_mean_discrepancy_covariances_multipliers(
            reorder_allocation_mat, nsamples_per_model,
            estimator._get_npartition_samples)
        Hmat, hvec = (
            get_acv_variance_discrepancy_covariances_multipliers(
                reorder_allocation_mat, nsamples_per_model,
                estimator._get_npartition_samples))
        return get_multioutput_acv_mean_and_variance_discrepancy_covariances(
            self.cov, self.V, self.W, self.B, Gmat, gvec, Hmat, hvec)

    def sample_estimate(self, values):
        return np.hstack([np.mean(values, axis=0),
                          np.cov(values.T, ddof=1).flatten()])

    def high_fidelity_estimator_covariance(self, nsamples_per_model):
        block_11 = self.cov[:self.nqoi, :self.nqoi]/nsamples_per_model[0]
        block_22 = covariance_of_variance_estimator(
            self.W[:self.nqoi**2, :self.nqoi**2],
            self.V[:self.nqoi**2, :self.nqoi**2], nsamples_per_model[0])
        block_12 = self.B[:self.nqoi, :self.nqoi**2]/nsamples_per_model[0]
        return torch_2x2_block(
            [[block_11, block_12],
             [block_12.T, block_22]])


class MFMCEstimator(ACVEstimator):
    def __init__(self, stat, costs, variable, cov, opt_criteria=None):
        # Use the sample analytical sample allocation for estimating a scalar
        # mean when estimating any statistic
        super().__init__(stat, costs, variable, cov, partition='mf',
                         recursion_index=None, opt_criteria=None)

    def _allocate_samples(self, target_cost):
        # nsample_ratios returned will be listed in according to
        # self.model_order which is what self.get_rsquared requires
        return allocate_samples_mfmc(
            self.cov.numpy(), self.costs.numpy(), target_cost)

    def _get_reordered_sample_allocation_matrix(self, nsamples_per_model):
        return get_sample_allocation_matrix_mfmc(self.nmodels)

    # No need for specialization because recusion index sets data correctly
    # def generate_data(self, functions):
    #     return generate_samples_and_values_mfmc(
    #         self.nsamples_per_model, functions, self.generate_samples,
    #         acv_modification=False)

    def _get_npartition_samples(self, nsamples_per_model):
        return get_npartition_samples_mfmc(nsamples_per_model)


class MLMCEstimator(ACVEstimator):
    def __init__(self, stat, costs, variable, cov, opt_criteria=None):
        """
        Use the sample analytical sample allocation for estimating a scalar
        mean when estimating any statistic
        """
        super().__init__(stat, costs, variable, cov, partition='mf',
                         recursion_index=None, opt_criteria=None)

    def _allocate_samples(self, target_cost):
        return allocate_samples_mlmc(
            self.cov.numpy(), self.costs.numpy(), target_cost)

    def _get_reordered_sample_allocation_matrix(self, nsamples_per_model):
        return get_sample_allocation_matrix_mlmc(self.nmodels)

    def _get_npartition_samples(self, nsamples_per_model):
        return get_npartition_samples_mlmc(nsamples_per_model)

    def _weights(self, CF, cf):
        return -torch.ones(cf.shape, dtype=torch.double)

    def _get_variance(self, nsamples_per_model):
        CF, cf = self.stat.get_discrepancy_covariances(
            self, nsamples_per_model)
        weights = self._weights(CF, cf)
        return (
            self.stat.high_fidelity_estimator_covariance(nsamples_per_model)
            + torch.linalg.multi_dot((weights, CF, weights.T))
            + torch.linalg.multi_dot((cf, weights.T))
            + torch.linalg.multi_dot((weights, cf.T))
        )


class BestACVEstimator(ACVEstimator):
    def __init__(self, stat, costs, variable,  cov, partition="mf",
                 opt_criteria=None, tree_depth=None):
        super().__init__(stat, costs, variable, cov, partition, None,
                         opt_criteria)
        self._depth = tree_depth

    def allocate_samples(self, target_cost, verbosity=0):
        best_variance = torch.as_tensor(np.inf, dtype=torch.double)
        best_result = None
        for index in get_acv_recursion_indices(self.nmodels, self._depth):
            self.set_recursion_index(index)
            try:
                super().allocate_samples(target_cost)
            except RuntimeError:
                # typically solver fails because trying to use
                # uniformative model as a recursive control variate
                print("Optimizer failed")
                self.optimized_variance = np.inf
            if verbosity > 0:
                msg = "{0} Objective: best {1}, current {2}".format(
                    index, best_variance.item(),
                    self.optimized_variance.item())
                print(msg)
            if self.optimized_variance < best_variance:
                best_result = [self.nsample_ratios, self.rounded_target_cost,
                               self.optimized_variance, index]
                best_variance = self.optimized_variance
        if best_result is None:
            raise RuntimeError("No solutions were found")
        self.set_recursion_index(best_result[3])
        self.set_optimized_params(*best_result[:3])
        return best_result[:3]


multioutput_estimators = {
    "acvmf": ACVEstimator,
    "mfmc": MFMCEstimator,
    "mlmc": MLMCEstimator,
    "acvmfb": BestACVEstimator}


multioutput_stats = {
    "mean": MultiOutputMean,
    "variance": MultiOutputVariance,
    "mean_variance": MultiOutputMeanAndVariance,
}


def get_estimator(estimator_type, stat_type, variable, costs, cov, *args,
                  **kwargs):
    if estimator_type not in multioutput_estimators:
        msg = f"Estimator {estimator_type} not supported"
        msg += f"Must be one of {multioutput_estimators.keys()}"
        raise ValueError(msg)

    if stat_type not in multioutput_stats:
        msg = f"Statistic {stat_type} not supported"
        msg += f"Must be one of {multioutput_stats.keys()}"
        raise ValueError(msg)

    nmodels = len(costs)
    stat = multioutput_stats[stat_type](nmodels, cov, *args)

    return multioutput_estimators[estimator_type](
        stat, costs, variable, cov, **kwargs)
