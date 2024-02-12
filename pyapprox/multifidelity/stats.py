import numpy as np
import torch
from abc import ABC, abstractmethod


def torch_2x2_block(blocks):
    return torch.vstack(
        [torch.hstack(blocks[0]),
         torch.hstack(blocks[1])])


def _get_nsamples_intersect(allocation_mat, npartition_samples):
    r"""
    Returns
    -------
    nsamples_intersect : np.ndarray (2*nmodels, 2*nmodels)
        The i,j entry contains contains
        :math:`|z^\star_i\cap\z^\star_j|` when i%2==0 and j%2==0
        :math:`|z_i\cap\z^\star_j|` when i%2==1 and j%2==0
        :math:`|z_i^\star\cap\z_j|` when i%2==0 and j%2==1
        :math:`|z_i\cap\z_j|` when i%2==1 and j%2==1
    """
    nmodels = allocation_mat.shape[0]
    nsubset_samples = npartition_samples[:, None] * allocation_mat
    nsamples_intersect = torch.zeros(
        (2*nmodels, 2*nmodels), dtype=torch.double)
    for ii in range(2*nmodels):
        nsamples_intersect[ii] = (
            nsubset_samples[allocation_mat[:, ii] == 1]).sum(axis=0)
    return nsamples_intersect


def _get_nsamples_subset(allocation_mat, npartition_samples):
    r"""
    Get the number of samples allocated to the sample subsets
    :math:`|z^\star_i` and :math:`|z_i|`

    npartition_samples : np.ndarray (nmodels)
        The size of the partitions that make up the subsets
        :math:`z_i, i=0\ldots, M-1`. These are represented by different
        color blocks in the ACV papers figures of sample allocation
    """
    nmodels = allocation_mat.shape[0]
    nsamples_subset = torch.zeros((2*nmodels), dtype=torch.double)
    for ii in range(2*nmodels):
        nsamples_subset[ii] = \
            npartition_samples[allocation_mat[:, ii] == 1].sum()
    return nsamples_subset


def _get_acv_mean_discrepancy_covariances_multipliers(
        allocation_mat, npartition_samples):
    nmodels = allocation_mat.shape[0]
    if np.any(npartition_samples.detach().numpy() < 0):
        raise RuntimeError("An entry in npartition samples was negative")
    nsamples_intersect = _get_nsamples_intersect(
        allocation_mat, npartition_samples)
    nsamples_subset = _get_nsamples_subset(
        allocation_mat, npartition_samples)
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


def _get_acv_variance_discrepancy_covariances_multipliers(
        allocation_mat, npartition_samples):
    """
    Compute H from Equation 3.14 of Dixon et al.
    """
    nmodels = allocation_mat.shape[0]
    if np.any(npartition_samples.detach().numpy() < 0):
        raise RuntimeError("An entry in npartition samples was negative")
    nsamples_intersect = _get_nsamples_intersect(
        allocation_mat, npartition_samples)
    nsamples_subset = _get_nsamples_subset(
        allocation_mat, npartition_samples)
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


def _get_multioutput_acv_mean_discrepancy_covariances(
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


def _get_multioutput_acv_variance_discrepancy_covariances(
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


def _get_multioutput_acv_mean_and_variance_discrepancy_covariances(
        cov, V, W, B, Gmat, gvec, Hmat, hvec):
    CF_mean, cf_mean = _get_multioutput_acv_mean_discrepancy_covariances(
        cov, Gmat, gvec)
    CF_var, cf_var = _get_multioutput_acv_variance_discrepancy_covariances(
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


def _V_entry(cov):
    V = np.kron(cov, cov)
    ones = np.ones((cov.shape[0], 1))
    V += (np.kron(np.kron(ones.T, cov), ones) *
          np.kron(np.kron(ones, cov), ones.T))
    return V


def _get_V_from_covariance(cov, nmodels):
    nqoi = cov.shape[0] // nmodels
    V = [[None for jj in range(nmodels)] for ii in range(nmodels)]
    for ii in range(nmodels):
        V[ii][ii] = _V_entry(cov[ii*nqoi:(ii+1)*nqoi, ii*nqoi:(ii+1)*nqoi])
        for jj in range(ii+1, nmodels):
            V[ii][jj] = _V_entry(cov[ii*nqoi:(ii+1)*nqoi, jj*nqoi:(jj+1)*nqoi])
            V[jj][ii] = V[ii][jj].T
    return np.block(V)


def _covariance_of_variance_estimator(W, V, nsamples):
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


def _get_W_from_pilot(pilot_values, nmodels):
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


def _get_B_from_pilot(pilot_values, nmodels):
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


def _nqoi_nqoi_subproblem(C, nmodels, nqoi, model_idx, qoi_idx):
    nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
    C_new = np.empty(
        (nsub_models*nsub_qoi, nsub_models*nsub_qoi))
    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            cnt2 = 0
            idx1 = jj1*nqoi + kk1
            for jj2 in model_idx:
                for kk2 in qoi_idx:
                    idx2 = jj2*nqoi + kk2
                    C_new[cnt1, cnt2] = C[idx1, idx2]
                    cnt2 += 1
            cnt1 += 1
    return C_new


def _nqoisq_nqoisq_subproblem(V, nmodels, nqoi, model_idx, qoi_idx):
    nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
    V_new = np.empty(
        (nsub_models*nsub_qoi**2, nsub_models*nsub_qoi**2))
    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            for ll1 in qoi_idx:
                cnt2 = 0
                idx1 = jj1*nqoi**2 + kk1*nqoi + ll1
                for jj2 in model_idx:
                    for kk2 in qoi_idx:
                        for ll2 in qoi_idx:
                            idx2 = jj2*nqoi**2 + kk2*nqoi + ll2
                            V_new[cnt1, cnt2] = V[idx1, idx2]
                            cnt2 += 1
                cnt1 += 1
    return V_new


def _nqoi_nqoisq_subproblem(B, nmodels, nqoi, model_idx, qoi_idx):
    nsub_models, nsub_qoi = len(model_idx), len(qoi_idx)
    B_new = np.empty(
        (nsub_models*nsub_qoi, nsub_models*nsub_qoi**2))
    cnt1 = 0
    for jj1 in model_idx:
        for kk1 in qoi_idx:
            cnt2 = 0
            idx1 = jj1*nqoi + kk1
            for jj2 in model_idx:
                for kk2 in qoi_idx:
                    for ll2 in qoi_idx:
                        idx2 = jj2*nqoi**2 + kk2*nqoi + ll2
                        B_new[cnt1, cnt2] = B[idx1, idx2]
                        cnt2 += 1
            cnt1 += 1
    return B_new


class MultiOutputStatistic(ABC):
    @abstractmethod
    def sample_estimate(self, values):
        raise NotImplementedError

    @abstractmethod
    def high_fidelity_estimator_covariance(self, nhf_samples):
        raise NotImplementedError

    @abstractmethod
    def compute_pilot_quantities(pilot_values):
        raise NotImplementedError

    @abstractmethod
    def set_pilot_quantities(self):
        raise NotImplementedError

    @abstractmethod
    def _get_cv_discrepancy_covariances(self, estimator, npartition_samples):
        raise NotImplementedError

    @abstractmethod
    def _get_acv_discrepancy_covariances(self, estimator, npartition_samples):
        raise NotImplementedError

    @abstractmethod
    def get_pilot_quantities_subset(
            self, nmodels, nqoi, model_idx, qoi_idx=None):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class MultiOutputMean(MultiOutputStatistic):
    def __init__(self, nqoi):
        """
        Parameters
        ----------
        nqoi : integer
            The number of quantities of interest (QoI) that each model returns
        """
        self._nqoi = nqoi

        self._nmodels = None
        self._cov = None

    def sample_estimate(self, values):
        return np.mean(values, axis=0)

    def high_fidelity_estimator_covariance(self, nhf_samples):
        return self._cov[:self._nqoi, :self._nqoi]/nhf_samples

    @staticmethod
    def compute_pilot_quantities(pilot_values):
        pilot_values = np.hstack(pilot_values)
        return (np.cov(pilot_values, rowvar=False, ddof=1), )

    def set_pilot_quantities(self, cov):
        self._cov = torch.as_tensor(cov, dtype=torch.double)
        self._nmodels = self._cov.shape[0] // self._nqoi

    def _get_discrepancy_covariances(self, Gmat, gvec):
        return _get_multioutput_acv_mean_discrepancy_covariances(
            self._cov, Gmat, gvec)

    def _get_cv_discrepancy_covariances(self, npartition_samples):
        Gmat = torch.full(
            (self._nmodels-1, self._nmodels-1), 1./npartition_samples[0],
            dtype=torch.double)
        gvec = torch.full(
            (self._nmodels-1,), 1./npartition_samples[0], dtype=torch.double)
        return self._get_discrepancy_covariances(Gmat, gvec)

    def _get_acv_discrepancy_covariances(
            self, allocation_mat, npartition_samples):
        Gmat, gvec = _get_acv_mean_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples)
        return self._get_discrepancy_covariances(Gmat, gvec)

    def get_pilot_quantities_subset(
            self, nmodels, nqoi, model_idx, qoi_idx=None):
        if qoi_idx is None:
            qoi_idx = np.arange(nqoi)
        cov_sub = _nqoi_nqoi_subproblem(
            self._cov, nmodels, nqoi, model_idx, qoi_idx)
        return (cov_sub, )


class MultiOutputVariance(MultiOutputStatistic):
    def __init__(self, nqoi):
        self._nqoi = nqoi

        self._nmodels = None
        self._cov = None
        self._W = None
        self._V = None

    def sample_estimate(self, values):
        return np.cov(values.T, ddof=1).flatten()

    def high_fidelity_estimator_covariance(self, nhf_samples):
        return _covariance_of_variance_estimator(
            self._W[:self._nqoi**2, :self._nqoi**2],
            self._V[:self._nqoi**2, :self._nqoi**2], nhf_samples)

    @staticmethod
    def compute_pilot_quantities(pilot_values):
        nmodels = len(pilot_values)
        pilot_values = np.hstack(pilot_values)
        cov = np.cov(pilot_values, rowvar=False, ddof=1)
        return cov, _get_W_from_pilot(pilot_values, nmodels)

    def set_pilot_quantities(self, cov, W):
        self._cov = torch.as_tensor(cov, dtype=torch.double)
        self._nmodels = self._cov.shape[0] // self._nqoi
        self._V = torch.as_tensor(
            _get_V_from_covariance(self._cov, self._nmodels),
            dtype=torch.double)
        if W.shape != self._V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self._V.shape)
            raise ValueError(msg)
        self._W = torch.as_tensor(W, dtype=torch.double)

    def _get_discrepancy_covariances(self, Gmat, gvec, Hmat, hvec):
        return _get_multioutput_acv_variance_discrepancy_covariances(
            self._V, self._W, Gmat, gvec, Hmat, hvec)

    def _get_cv_discrepancy_covariances(self, npartition_samples):
        Gmat = torch.full(
            (self._nmodels-1, self._nmodels-1), 1./npartition_samples[0],
            dtype=torch.double)
        gvec = torch.full(
            (self._nmodels-1,), 1./npartition_samples[0], dtype=torch.double)
        Hmat = torch.full(
            (self._nmodels-1, self._nmodels-1),
            1./(npartition_samples[0]*(npartition_samples[0]-1)),
            dtype=torch.double)
        hvec = torch.full(
            (self._nmodels-1,),
            1./(npartition_samples[0]*(npartition_samples[0]-1)),
            dtype=torch.double)
        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def _get_acv_discrepancy_covariances(
            self, allocation_mat, npartition_samples):
        Gmat, gvec = _get_acv_mean_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples)
        Hmat, hvec = (
            _get_acv_variance_discrepancy_covariances_multipliers(
                allocation_mat, npartition_samples))
        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def get_pilot_quantities_subset(
            self, nmodels, nqoi, model_idx, qoi_idx=None):
        if qoi_idx is None:
            qoi_idx = np.arange(nqoi)
        cov_sub = _nqoi_nqoi_subproblem(
            self._cov, nmodels, nqoi, model_idx, qoi_idx)
        W_sub = _nqoisq_nqoisq_subproblem(
            self._W, nmodels, nqoi, model_idx, qoi_idx)
        return cov_sub, W_sub


class MultiOutputMeanAndVariance(MultiOutputStatistic):
    def __init__(self, nqoi):
        self._nqoi = nqoi

        self._nmodels = None
        self._cov = None
        self._W = None
        self._V = None
        self._B = None

    def sample_estimate(self, values):
        return np.hstack([np.mean(values, axis=0),
                          np.cov(values.T, ddof=1).flatten()])

    def high_fidelity_estimator_covariance(self, nhf_samples):
        block_11 = self._cov[:self._nqoi, :self._nqoi]/nhf_samples
        block_22 = _covariance_of_variance_estimator(
            self._W[:self._nqoi**2, :self._nqoi**2],
            self._V[:self._nqoi**2, :self._nqoi**2], nhf_samples)
        block_12 = self._B[:self._nqoi, :self._nqoi**2]/nhf_samples
        return torch_2x2_block(
            [[block_11, block_12],
             [block_12.T, block_22]])

    @staticmethod
    def compute_pilot_quantities(pilot_values):
        nmodels = len(pilot_values)
        pilot_values = np.hstack(pilot_values)
        cov = np.cov(pilot_values, rowvar=False, ddof=1)
        W = _get_W_from_pilot(pilot_values, nmodels)
        B = _get_B_from_pilot(pilot_values, nmodels)
        return cov, W, B

    def set_pilot_quantities(self, cov, W, B):
        self._cov = torch.as_tensor(cov, dtype=torch.double)
        self._nmodels = self._cov.shape[0] // self._nqoi
        self._V = torch.as_tensor(
            _get_V_from_covariance(self._cov, self._nmodels),
            dtype=torch.double)
        if W.shape != self._V.shape:
            msg = "W has the wrong shape {0}. Should be {1}".format(
                W.shape, self._V.shape)
            raise ValueError(msg)
        self._W = torch.as_tensor(W, dtype=torch.double)
        B_shape = cov.shape[0], self._V.shape[1]
        if B.shape != B_shape:
            msg = "B has the wrong shape {0}. Should be {1}".format(
                B.shape, B_shape)
            raise ValueError(msg)
        self._B = torch.as_tensor(B, dtype=torch.double)

    def _get_discrepancy_covariances(self, Gmat, gvec, Hmat, hvec):
        return _get_multioutput_acv_mean_and_variance_discrepancy_covariances(
            self._cov, self._V, self._W, self._B, Gmat, gvec, Hmat, hvec)

    def _get_cv_discrepancy_covariances(self, npartition_samples):
        Gmat = torch.full(
            (self._nmodels-1, self._nmodels-1), 1./npartition_samples[0],
            dtype=torch.double)
        gvec = torch.full(
            (self._nmodels-1,), 1./npartition_samples[0], dtype=torch.double)
        Hmat = torch.full(
            (self._nmodels-1, self._nmodels-1),
            1./(npartition_samples[0]*(npartition_samples[0]-1)),
            dtype=torch.double)
        hvec = torch.full(
            (self._nmodels-1,),
            1./(npartition_samples[0]*(npartition_samples[0]-1)),
            dtype=torch.double)
        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def _get_acv_discrepancy_covariances(
            self, allocation_mat, npartition_samples):
        Gmat, gvec = _get_acv_mean_discrepancy_covariances_multipliers(
            allocation_mat, npartition_samples)
        Hmat, hvec = (
            _get_acv_variance_discrepancy_covariances_multipliers(
                allocation_mat, npartition_samples))
        return self._get_discrepancy_covariances(Gmat, gvec, Hmat, hvec)

    def get_pilot_quantities_subset(
            self, nmodels, nqoi, model_idx, qoi_idx=None):
        if qoi_idx is None:
            qoi_idx = np.arange(nqoi)
        cov_sub = _nqoi_nqoi_subproblem(
            self._cov, nmodels, nqoi, model_idx, qoi_idx)
        W_sub = _nqoisq_nqoisq_subproblem(
            self._W, nmodels, nqoi, model_idx, qoi_idx)
        B_sub = _nqoi_nqoisq_subproblem(
            self._B, nmodels, nqoi, model_idx, qoi_idx)
        return cov_sub, W_sub, B_sub
