import torch
import numpy as np
from abc import ABC, abstractmethod

from pyapprox.interface.wrappers import ModelEnsemble
from pyapprox.multifidelity.control_variate_monte_carlo import (
    reorder_allocation_matrix_acvgmf, get_nsamples_intersect,
    get_nsamples_subset, get_nsamples_per_model,
    separate_model_values_acv, separate_samples_per_model_acv,
    generate_samples_acv, round_nsample_ratios, bootstrap_acv_estimator,
    get_sample_allocation_matrix_acvmf, get_sample_allocation_matrix_acvis,
    get_nhf_samples)


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


def get_acv_mean_discrepancy_covariances_multipliers(
        allocation_mat, nsamples_per_model, get_npartition_samples,
        recursion_index):
    nmodels = allocation_mat.shape[0]
    reorder_allocation_mat = reorder_allocation_matrix_acvgmf(
        allocation_mat, nsamples_per_model, recursion_index)
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
        allocation_mat, nsamples_per_model, get_npartition_samples,
        recursion_index):
    """
    Compute H from Equation 3.14 of Dixon et al.
    """
    nmodels = allocation_mat.shape[0]
    reorder_allocation_mat = reorder_allocation_matrix_acvgmf(
        allocation_mat, nsamples_per_model, recursion_index)
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
            V_ij = V[(ii+1)*nqsq:(ii+2)*nqsq, (jj+1)*nqsq:(ii+2)*nqsq]
            W_ij = W[(ii+1)*nqsq:(ii+2)*nqsq, (jj+1)*nqsq:(ii+2)*nqsq]
            discp_cov[ii*nqsq:(ii+1)*nqsq, jj*nqsq:(jj+1)*nqsq] = (
                Gmat[ii, jj]*W_ij+Hmat[ii, jj]*V_ij)
            discp_cov[jj*nqsq:(jj+1)*nqsq, ii*nqsq:(ii+1)*nqsq] = (
                discp_cov[ii*nqsq:(ii+1)*nqsq, jj*nqsq:(jj+1)*nqsq].T)
    return discp_cov, discp_vec


def torch_2x2_block(blocks):
    return torch.vstack(
        [torch.hstack(blocks[0]),
         torch.hstack(blocks[1])])


def get_multioutput_acv_mean_and_variance_discrepancy_covariances(
        cov, V, W, B, Gmat, gvec, Hmat, hvec):
    CF_mean, cf_mean = get_multioutput_acv_mean_discrepancy_covariances(
        cov, Gmat, gvec)
    CF_var, cf_var = get_multioutput_acv_variance_discrepancy_covariances(
        V, W, Gmat, gvec, Hmat, hvec)
    nmodels = len(gvec)+1
    nqoi = cov.shape[0]//nmodels
    nqsq = V.shape[0]//nmodels
    CF_mean_var = torch.empty(
        (nqoi*(nmodels-1), nqsq*(nmodels-1)), dtype=torch.double)
    cf_mean_var = torch.empty((nqoi, nqsq*(nmodels-1)), dtype=torch.double)
    cf_mean_var_T = torch.empty((nqsq*(nmodels-1), nqoi), dtype=torch.double)
    for ii in range(nmodels-1):
        B_0i = B[0:nqoi, (ii+1)*nqsq:(ii+2)*nqsq]
        B_0i_T = B.T[0:nqsq, (ii+1)*nqoi:(ii+2)*nqoi]
        cf_mean_var[ii*nqoi:(ii+1)*nqoi, ii*nqsq:(ii+1)*nqsq] = gvec[ii]*B_0i
        cf_mean_var_T[ii*nqsq:(ii+1)*nqsq, ii*nqoi:(ii+1)*nqoi] = (
            gvec[ii]*B_0i_T)
        for jj in range(nmodels-1):
            B_ij = B[(ii+1)*nqoi:(ii+2)*nqoi, (jj+1)*nqsq:(ii+2)*nqsq]
            CF_mean_var[ii*nqoi:(ii+1)*nqsq, jj*nqsq:(jj+1)*nqsq] = (
                Gmat[ii, jj]*B_ij)
    CF = torch_2x2_block(
        [[CF_mean, CF_mean_var],
         [CF_mean_var.T, CF_var]])
    cf = torch_2x2_block([[cf_mean, cf_mean_var],
                          [cf_mean_var_T, cf_var]])
    return CF, cf


def get_npartition_samples_acvmf(nsamples_per_model):
    nmodels = len(nsamples_per_model)
    II = np.unique(nsamples_per_model.numpy(), return_index=True)[1]
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


class MultiOutputACVEstimator(ABC):
    def __init__(self, cov, costs, variable, partition="mf",
                 recursion_index=None):
        """
        Constructor.

        Parameters
        ----------
        cov : np.ndarray (nmodels*nqoi, nmodels)
            The covariance C between each of the models. The highest fidelity
            model is the first model, i.e. covariance between its QoI
            is cov[:nqoi, :nqoi]

        costs : np.ndarray (nmodels)
            The relative costs of evaluating each model

        variable : :class:`~pyapprox.variables.IndependentMarginalsVariable`
            The uncertain model parameters

        partition : string
            What sample partition scheme to use. Must be 'mf'

        recursion_index : np.ndarray (nmodels-1)
            The recusion index that specifies which ACV estimator is used
        """
        self.cov, self.costs, self.nmodels, self.nqoi = self._check_cov(
            cov, costs)
        self.variable = variable
        self.partition = partition
        self.set_recursion_index(recursion_index)

        self.cov_opt = torch.tensor(self.cov, dtype=torch.double)
        self.costs_opt = torch.tensor(self.costs, dtype=torch.double)

        self.nsamples_per_model, self.optimized_variance = None, None
        self.rounded_target_cost = None
        self.model_labels = None

    def _check_cov(self, cov, costs):
        nmodels = len(costs)
        if cov.shape[0] % nmodels:
            print(cov.shape, costs.shape)
            raise ValueError("cov and costs are inconsistent")
        return cov.copy(), np.array(costs), nmodels, cov.shape[0]//nmodels

    def _weights(self, CF, cf):
        try:
            if CF.shape == (1, 1):
                weights = -cf.T/CF[0, 0]
            else:
                weights = -torch.linalg.solve(CF, cf.T)
        except (np.linalg.LinAlgError, RuntimeError):
            weights = torch.ones(cf.shape, dtype=torch.double)*1e16
        return weights.T

    @abstractmethod
    def _get_discpreancy_covariances(self):
        raise NotImplementedError

    @abstractmethod
    def _sample_estimate(self, values):
        raise NotImplementedError

    def _estimate(self, values, weights):
        nmodels = len(values)
        assert len(values) == nmodels
        # high fidelity monte carlo estimate of mean
        deltas = np.hstack(
            [self._sample_estimate(values[ii][0]) -
             self._sample_estimate(values[ii][1])
             for ii in range(1, nmodels)])
        est = self._sample_estimate(values[0][1]) + weights.numpy().dot(deltas)
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
        CF, cf = self._get_discpreancy_covariances()
        weights = self._weights(CF, cf)
        # item returns value as scalar
        return self._estimate(values, weights)

    def __repr__(self):
        if self.optimized_variance is None:
            return "{0}".format(self.__class__.__name__)
        return "{0}(variance={1:.3g}, target_cost={2:.3g})".format(
            self.__class__.__name__, self.optimized_variance,
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
        nsample_ratios, log10_var = self._allocate_samples(target_cost)
        nsample_ratios = nsample_ratios.detach().numpy()
        nsample_ratios, rounded_target_cost = round_nsample_ratios(
            target_cost, self.costs, nsample_ratios)
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

    def _get_reordered_sample_allocation_matrix(self):
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
            return reorder_allocation_matrix_acvgmf(
                self.allocation_mat, self.nsamples_per_model,
                self.recursion_index)
        # TODO create reordering for acvis
        raise ValueError("partition must one of ['mf']")

    def set_recursion_index(self, index):
        if index is None:
            index = np.zeros(self.nmodels-1, dtype=int)
        if index.shape[0] != self.nmodels-1:
            raise ValueError("index is the wrong shape")
        self.recursion_index = index
        self._create_allocation_matrix()

    @abstractmethod
    def high_fidelity_estimator_covariance(self):
        raise NotImplementedError

    def _get_variance(self, nsamples_per_model):
        CF, cf = self._get_discpreancy_covariances()
        weights = self._weights(CF, cf)
        return (self.high_fidelity_estimator_covariance(
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
        reorder_allocation_mat = self._get_reordered_sample_allocation_matrix()
        samples_per_model, partition_indices_per_model = generate_samples_acv(
            reorder_allocation_mat, self.nsamples_per_model,
            npartition_samples, self.variable.rvs)
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
        reorder_allocation_mat = self._get_reordered_sample_allocation_matrix()
        acv_values = separate_model_values_acv(
            reorder_allocation_mat, values_per_model,
            partition_indices_per_model)
        acv_samples = separate_samples_per_model_acv(
            reorder_allocation_mat, samples_per_model,
            partition_indices_per_model)
        return acv_samples, acv_values

    def estimate_from_values_per_model(self, values_per_model,
                                       partition_indices_per_model):
        reorder_allocation_mat = self._get_reordered_sample_allocation_matrix()
        acv_values = separate_model_values_acv(
            reorder_allocation_mat, values_per_model,
            partition_indices_per_model)
        return self(acv_values)

    def bootstrap(self, values_per_model, partition_indices_per_model,
                  nbootstraps=1000):
        return bootstrap_acv_estimator(
            values_per_model, partition_indices_per_model,
            self._get_npartition_samples(self.nsamples_per_model),
            self._get_reordered_sample_allocation_matrix(),
            self._get_approximate_control_variate_weights(), nbootstraps)


class MultiOutputACVMeanEstimator(MultiOutputACVEstimator):
    def _get_discpreancy_covariances(self):
        Gmat, gvec = get_acv_mean_discrepancy_covariances_multipliers(
            self.allocation_mat, self.nsamples_per_model,
            self._get_npartition_samples, self.recursion_index)
        return get_multioutput_acv_mean_discrepancy_covariances(
            self.cov, Gmat, gvec)

    def _sample_estimate(self, values):
        return np.mean(values, axis=0)

    def high_fidelity_estimator_covariance(self, nsamples_per_model):
        return self.cov[:self.nqoi, :self.nqoi]/nsamples_per_model[0]


class MultiOutputACVVarianceEstimator(MultiOutputACVEstimator):
    def __init__(self, cov, costs, variable, W, partition="mf",
                 recursion_index=None):
        super().__init__(cov, costs, variable, partition="mf",
                         recursion_index=None)
        self.V = get_V_from_covariance(self.cov, self.nmodels)
        if W.shape != self.V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self.V.shape)
            raise ValueError(msg)
        self.W = W

    def _get_discpreancy_covariances(self):
        Gmat, gvec = get_acv_mean_discrepancy_covariances_multipliers(
            self.allocation_mat, self.nsamples_per_model,
            self._get_npartition_samples, self.recursion_index)
        Hmat, hvec = (
            get_acv_variance_discrepancy_covariances_multipliers(
                self.allocation_mat, self.nsamples_per_model,
                self._get_npartition_samples, self.recursion_index))
        return get_multioutput_acv_variance_discrepancy_covariances(
            self.V, self.W, Gmat, gvec, Hmat, hvec)

    def _sample_estimate(self, values):
        return np.cov(values.T, ddof=1).flatten()

    def high_fidelity_estimator_covariance(self, nsamples_per_model):
        return covariance_of_variance_estimator(
            self.W[:self.nqoi**2, :self.nqoi**2],
            self.V[:self.nqoi**2, :self.nqoi**2], nsamples_per_model[0])


class MultiOutputACVMeanAndVarianceEstimator(MultiOutputACVVarianceEstimator):
    def __init__(self, cov, costs, variable, W, B, partition="mf",
                 recursion_index=None):
        super().__init__(cov, costs, variable, W, partition="mf",
                         recursion_index=None)
        B_shape = cov.shape[0], self.V.shape[1]
        if B.shape != B_shape:
            msg = "B has the wrong shape {0}. Should be {1}".format(
                B.shape, B_shape)
            raise ValueError(msg)
        self.B = B

    def _get_discpreancy_covariances(self):
        Gmat, gvec = get_acv_mean_discrepancy_covariances_multipliers(
            self.allocation_mat, self.nsamples_per_model,
            self._get_npartition_samples, self.recursion_index)
        Hmat, hvec = (
            get_acv_variance_discrepancy_covariances_multipliers(
                self.allocation_mat, self.nsamples_per_model,
                self._get_npartition_samples, self.recursion_index))
        return get_multioutput_acv_mean_and_variance_discrepancy_covariances(
            self.cov, self.V, self.W, self.B, Gmat, gvec, Hmat, hvec)

    def _sample_estimate(self, values):
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
