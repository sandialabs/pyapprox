from functools import reduce
from itertools import product

import numpy as np

from pyapprox.util.utilities import get_correlation_from_covariance


def _check_safe_cast_to_integers(array):
    array_int = np.array(np.round(array), dtype=int)
    if not np.allclose(array, array_int, 1e-15):
        raise ValueError("Arrays entries are not integers")
    return array_int


def _cast_to_integers(array):
    return _check_safe_cast_to_integers(array)


def _variance_reduction(get_rsquared, cov, nsample_ratios):
    r"""
    Compute the variance reduction:

    .. math:: \gamma = 1-r^2

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Returns
    -------
    gamma : float
        The variance reduction
    """
    return 1-get_rsquared(cov, nsample_ratios)


def _check_mfmc_model_costs_and_correlations(costs, corr):
    """
    Check that the model costs and correlations satisfy equation 3.12
    in MFMC paper.
    """
    nmodels = len(costs)
    for ii in range(1, nmodels):
        if ii < nmodels-1:
            denom = corr[0, ii]**2 - corr[0, ii+1]**2
        else:
            denom = corr[0, ii]**2
        if denom <= np.finfo(float).eps:
            return False
        corr_ratio = (corr[0, ii-1]**2 - corr[0, ii]**2)/denom
        cost_ratio = costs[ii-1] / costs[ii]
        if corr_ratio >= cost_ratio:
            return False
    return True


def _get_rsquared_mfmc(cov, nsample_ratios):
    r"""
    Compute r^2 used to compute the variance reduction  of
    Multifidelity Monte Carlo (MFMC)

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Returns
    -------
    rsquared : float
        The value r^2
    """
    nmodels = cov.shape[0]
    assert len(nsample_ratios) == nmodels-1
    rsquared = (nsample_ratios[0]-1)/(nsample_ratios[0])*cov[0, 1]/(
        cov[0, 0]*cov[1, 1])*cov[0, 1]
    for ii in range(1, nmodels-1):
        p1 = (nsample_ratios[ii]-nsample_ratios[ii-1])/(
            nsample_ratios[ii]*nsample_ratios[ii-1])
        p1 *= cov[0, ii+1]/(cov[0, 0]*cov[ii+1, ii+1])*cov[0, ii+1]
        rsquared += p1
    return rsquared


def _allocate_samples_mfmc(cov, costs, target_cost):
    r"""
    Determine the samples to be allocated to each model when using MFMC

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    target_cost : float
        The total cost budget

    Returns
    -------
    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i=r_i*nhf_samples, i=1,...,nmodels-1

    log_variance : float
        The logarithm of the variance of the estimator
    """

    nmodels = len(costs)
    corr = get_correlation_from_covariance(cov)
    II = np.argsort(np.absolute(corr[0, 1:]))[::-1]
    if (II.shape[0] != nmodels-1):
        msg = "Correlation shape {0} inconsistent with len(costs) {1}.".format(
            corr.shape, len(costs))
        raise RuntimeError(msg)
    if not np.allclose(II, np.arange(nmodels-1)):
        msg = 'Models must be ordered with decreasing correlation with '
        msg += 'high-fidelity model'
        raise RuntimeError(msg)

    r = []
    for ii in range(nmodels-1):
        # Step 3 in Algorithm 2 in Peherstorfer et al 2016
        num = costs[0] * (corr[0, ii]**2 - corr[0, ii+1]**2)
        den = costs[ii] * (1 - corr[0, 1]**2)
        r.append(np.sqrt(num/den))

    num = costs[0]*corr[0, -1]**2
    den = costs[-1] * (1 - corr[0, 1]**2)
    r.append(np.sqrt(num/den))

    # Step 4 in Algorithm 2 in Peherstorfer et al 2016
    nhf_samples = target_cost / np.dot(costs, r)
    nsample_ratios = r[1:]

    gamma = _variance_reduction(_get_rsquared_mfmc, cov, nsample_ratios)
    log_variance = np.log(gamma)+np.log(cov[0, 0])-np.log(
        nhf_samples)
    return np.atleast_1d(nsample_ratios), log_variance


def _get_sample_allocation_matrix_mfmc(nmodels):
    mat = np.zeros((nmodels, 2*nmodels))
    mat[0, 1:] = 1
    for ii in range(1, nmodels):
        mat[ii, 2*ii+1:] = 1
    return mat


def _get_npartition_samples_mfmc(nsamples_per_model):
    npartition_samples = np.hstack(
        (nsamples_per_model[0], np.diff(nsamples_per_model)))
    return npartition_samples


def _get_rsquared_mlmc(cov, nsample_ratios):
    r"""
    Compute r^2 used to compute the variance reduction of
    Multilevel Monte Carlo (MLMC)

    See Equation 2.24 in ARXIV paper where alpha_i=-1 for all i

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples,
        i=1,...,nmodels-1.
        The values r_i correspond to eta_i in Equation 2.24

    Returns
    -------
    gamma : float
        The variance reduction
    """
    nmodels = cov.shape[0]
    assert len(nsample_ratios) == nmodels-1
    gamma = 0.0
    rhat = np.ones((nmodels), dtype=float)
    for ii in range(1, nmodels):
        rhat[ii] = nsample_ratios[ii-1] - rhat[ii-1]

    for ii in range(nmodels-1):
        vardelta = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
        gamma += vardelta / (rhat[ii])

    v = cov[nmodels-1, nmodels-1]
    gamma += v / (rhat[-1])

    gamma /= cov[0, 0]
    return 1-gamma


def _allocate_samples_mlmc(cov, costs, target_cost):
    r"""
    Determine the samples to be allocated to each model when using MLMC

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    target_cost : float
        The total cost budget

    Returns
    -------
    nhf_samples : integer
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples,
        i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
        the number of samples in the two different discrepancies involving
        the ith model.

    log_variance : float
        The logarithm of the variance of the estimator
    """
    nmodels = cov.shape[0]
    costs = np.asarray(costs)

    II = np.argsort(costs)[::-1]
    if not np.allclose(II, np.arange(nmodels)):
        # print(costs)
        raise ValueError("Models cost do not decrease monotonically")

    # compute the variance of the discrepancy
    var_deltas = np.empty(nmodels)
    for ii in range(nmodels-1):
        var_deltas[ii] = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
    var_deltas[nmodels-1] = cov[nmodels-1, nmodels-1]

    # compute the cost of one sample of the discrepancy
    cost_deltas = np.empty(nmodels)
    cost_deltas[:nmodels-1] = (costs[:nmodels-1] + costs[1:nmodels])
    cost_deltas[nmodels-1] = costs[nmodels-1]

    # compute variance * cost
    var_cost_prods = var_deltas * cost_deltas

    # compute variance / cost
    var_cost_ratios = var_deltas / cost_deltas

    # compute the lagrange multiplier
    lagrange_multiplier = target_cost / np.sqrt(var_cost_prods).sum()

    # compute the number of samples needed for each discrepancy
    nsamples_per_delta = lagrange_multiplier*np.sqrt(var_cost_ratios)

    # compute the ML estimator variance from the target cost
    variance = np.sum(var_deltas/nsamples_per_delta)

    # compute the number of samples allocated to each model. For
    # all but the highest fidelity model we need to collect samples
    # from two discrepancies.
    nhf_samples = nsamples_per_delta[0]
    nsample_ratios = np.empty(nmodels-1)
    for ii in range(nmodels-1):
        nsample_ratios[ii] = (
            nsamples_per_delta[ii]+nsamples_per_delta[ii+1])/nhf_samples

    assert np.allclose(
        nhf_samples*costs[0] + (nsample_ratios*nhf_samples).dot(costs[1:]),
        cost_deltas.dot(nsamples_per_delta))

    gamma = _variance_reduction(_get_rsquared_mlmc, cov, nsample_ratios)
    log_variance = np.log(gamma)+np.log(cov[0, 0])-np.log(
        nhf_samples)
    # print(log_variance)
    if np.isnan(log_variance):
        raise RuntimeError('MLMC variance is NAN')
    return np.atleast_1d(nsample_ratios), log_variance


def _get_sample_allocation_matrix_mlmc(nmodels):
    r"""
    Get the sample allocation matrix

    Parameters
    ----------
    nmodel : integer
        The number of models :math:`M`

    Returns
    -------
    mat : np.ndarray (nmodels, 2*nmodels)
        For columns :math:`2j, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i^\star\subseteq z_j^\star`
        For columns :math:`2j+1, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i\subseteq z_j`
    """
    mat = np.zeros((nmodels, 2*nmodels))
    for ii in range(nmodels-1):
        mat[ii, 2*ii+1:2*ii+3] = 1
    mat[-1, -1] = 1
    return mat


def _get_npartition_samples_mlmc(nsamples_per_model):
    r"""
    Get the size of the partitions combined to form
        :math:`z_i, i=0\ldots, M-1`.


    Parameters
    ----------
    nsamples_per_model : np.ndarray (nmodels)
         The number of total samples allocated to each model. I.e.
         :math:`|z_i\cup\z^\star_i|, i=0,\ldots,M-1`

    Returns
    -------
    npartition_samples : np.ndarray (nmodels)
        The size of the partitions that make up the subsets
        :math:`z_i, i=0\ldots, M-1`. These are represented by different
        color blocks in the ACV papers figures of sample allocation
    """
    nmodels = nsamples_per_model.shape[0]
    npartition_samples = np.empty((nmodels), dtype=float)
    npartition_samples[0] = nsamples_per_model[0]
    for ii in range(1, nmodels):
        npartition_samples[ii] = (
            nsamples_per_model[ii]-npartition_samples[ii-1])
    return npartition_samples


class ModelTree():
    def __init__(self, root, children=[]):
        if type(children) == np.ndarray:
            children = list(children)
        self.children = children
        for ii in range(len(self.children)):
            if type(self.children[ii]) != ModelTree:
                self.children[ii] = ModelTree(self.children[ii])
        self.root = root

    def num_nodes(self):
        nnodes = 1
        for child in self.children:
            if type(child) == ModelTree:
                nnodes += child.num_nodes()
            else:
                nnodes += 1
        return nnodes

    def to_index(self):
        index = [None for ii in range(self.num_nodes())]
        index[0] = self.root
        self._to_index_recusive(index, self)
        return np.array(index)

    def _to_index_recusive(self, index, root):
        for child in root.children:
            index[child.root] = root.root
            self._to_index_recusive(index, child)

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, self.to_index())


def _update_list_for_reduce(mylist, indices):
    mylist[indices[0]].append(indices[1])
    return mylist


def _generate_all_trees(children, root, tree_depth):
    if tree_depth < 2 or len(children) == 0:
        yield ModelTree(root, children)
    else:
        for prod in product((0, 1), repeat=len(children)):
            if not any(prod):
                continue
            nexts, sub_roots = reduce(
                _update_list_for_reduce, zip(prod, children), ([], []))
            for q in product(range(len(sub_roots)), repeat=len(nexts)):
                sub_children = reduce(
                    _update_list_for_reduce, zip(q, nexts),
                    [[] for ii in sub_roots])
                yield from [
                    ModelTree(root, list(children))
                    for children in product(
                            *(_generate_all_trees(sc, sr, tree_depth-1)
                              for sr, sc in zip(sub_roots, sub_children)))]


def _get_acv_recursion_indices(nmodels, depth=None):
    if depth is None:
        depth = nmodels-1
    if depth > nmodels-1:
        msg = f"Depth {depth} exceeds number of lower-fidelity models"
        raise ValueError(msg)
    for index in _generate_all_trees(np.arange(1, nmodels), 0, depth):
        yield index.to_index()[1:]
