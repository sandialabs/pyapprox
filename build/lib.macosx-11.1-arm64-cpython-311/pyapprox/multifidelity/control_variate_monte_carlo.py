"""
Functions for estimating expectations using frequentist control-variate
Monte-Carlo based methods such as multi-level Monte-Carlo,
control-variate Monte-Carlo, and approximate control-variate Monte-Carlo.
"""
import numpy as np
from scipy.optimize import minimize
from functools import partial, reduce
from itertools import product


from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.util.utilities import get_all_sample_combinations

try:
    # use torch to compute gradients for sample allocation optimization
    import torch
    use_torch = True
    pkg = torch
except ImportError:
    # msg = 'Could not import Torch'
    # print(msg)
    use_torch = False
    pkg = np
# use_torch = False
# pkg = np


def _ndarray_as_pkg_format(array):
    """
    Shallow copy an np.ndarray to the format used by pkg, e.g.
    either np.ndarray or torch.tensor. If already in the correct format
    a reference to the original object will be returned
    """
    if use_torch and not pkg.is_tensor(array):
        return pkg.tensor(array, dtype=pkg.double)
    elif use_torch and pkg.is_tensor(array):
        return array
    return array


def to_numpy(array):
    if use_torch and pkg.is_tensor(array):
        return array.detach().numpy()
    return array


def pkg_copy(array):
    if use_torch and pkg.is_tensor(array):
        return array.clone()
    return array.copy()


def pkg_hstack(objs):
    if use_torch and any([pkg.is_tensor(o) for o in objs]):
        return pkg.hstack(objs)
    return np.hstack(objs)


def pkg_zeros(shape, array_type, dtype):
    if array_type != np.ndarray:
        return pkg.zeros(shape, dtype=pkg.double)
    return np.zeros(shape, dtype=np.double)


def pkg_ones(shape, array_type, dtype):
    if array_type != np.ndarray:
        return pkg.ones(shape, dtype=pkg.double)
    return np.ones(shape, dtype=np.double)


def pkg_empty(shape, array_type, dtype):
    if array_type != np.ndarray:
        return pkg.empty(shape, dtype=pkg.double)
    return np.empty(shape, dtype=np.double)


def pkg_diff(array):
    if use_torch and pkg.is_tensor(array):
        return pkg.diff(array)
    return np.diff(array)


def pkg_solve(A, b):
    if use_torch and pkg.is_tensor(A):
        return pkg.linalg.solve(A, b)
    return np.linalg.solve(A, b)


def check_safe_cast_to_integers(array):
    array_int = np.array(np.round(array), dtype=int)
    if not np.allclose(array, array_int, 1e-15):
        raise ValueError("Arrays entries are not integers")
    return array_int


def cast_to_integers(array):
    return check_safe_cast_to_integers(array)


def compute_correlations_from_covariance(cov):
    """
    Compute the correlation matrix of a covariance matrix.

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    Returns
    -------
    corr : np.ndarray (nmodels,nmodels)
        The correlation matrix
    """
    corr_sqrt = np.diag(1/np.sqrt((np.diag(cov))))
    corr = np.dot(corr_sqrt, np.dot(cov, corr_sqrt))
    return corr


def get_variance_reduction(get_rsquared, cov, nsample_ratios):
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


def get_control_variate_rsquared(cov):
    r"""
    Compute :math:`r^2` used to compute the variance reduction of
    control variate Monte Carlo

    .. math:: \gamma = 1-r^2, \qquad     r^2 = c^TC^{-1}c

    where c is the first column of C

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    Returns
    -------
    rsquared : float
        The value  :math:`r^2`
    """
    # nmodels = cov.shape[0]
    rsquared = cov[0, 1:].dot(pkg_solve(cov[1:, 1:], cov[1:, 0]))
    rsquared /= cov[0, 0]
    return rsquared


def get_rsquared_mfmc(cov, nsample_ratios):
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


def get_rsquared_mlmc(cov, nsample_ratios):
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
    rhat = pkg_ones((nmodels), type(cov), dtype=pkg.double)
    for ii in range(1, nmodels):
        rhat[ii] = nsample_ratios[ii-1] - rhat[ii-1]

    for ii in range(nmodels-1):
        vardelta = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
        gamma += vardelta / (rhat[ii])

    v = cov[nmodels-1, nmodels-1]
    gamma += v / (rhat[-1])

    gamma /= cov[0, 0]
    return 1-gamma


def get_mlmc_control_variate_weights(nmodels):
    r"""
    Get the weights used by the MLMC control variate estimator

    Returns
    -------
    weights : np.ndarray (nmodels-1)
        The control variate weights
    """
    return -np.ones(nmodels-1, dtype=np.double)


def compute_approximate_control_variate_mean_estimate(weights, values):
    r"""
    Use approximate control variate Monte Carlo to estimate the mean of
    high-fidelity data with low-fidelity models with unknown means

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

    weights : np.ndarray (nmodels-1)
        the control variate weights

    Returns
    -------
    est : float
        The control variate estimate of the mean
    """
    nmodels = len(values)
    assert len(values) == nmodels
    # high fidelity monte carlo estimate of mean
    est = values[0][1].mean()
    for ii in range(nmodels-1):
        est += weights[ii]*(values[ii+1][0].mean()-values[ii+1][1].mean())
    return est


def compute_control_variate_mean_estimate(weights, values, lf_means):
    r"""
    Use control variate Monte Carlo to estimate the mean of
    high-fidelity data with low-fidelity models with known means

    Parameters
    ----------
    values : list (nmodels)
        Each entry of the list contains

        values0 : np.ndarray (num_samples_i0,num_qoi)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

    weights : np.ndarray (nmodels-1)
        the control variate weights


    lf_means : np.ndarray (nmodels-1):
        The known means of the low fidelity models

    Returns
    -------
    est : float
        The control variate estimate of the mean
    """
    nmodels = len(values)
    assert len(values) == nmodels
    # high fidelity monte carlo estimate of mean
    est = values[0].mean()
    for ii in range(nmodels-1):
        est += weights[ii]*(values[ii+1].mean()-lf_means[ii])
    return est


def check_mfmc_model_costs_and_correlations(costs, corr):
    """
    Check that the model costs and correlations satisfy equation 3.12
    in MFMC paper.
    """
    nmodels = corr.shape[0]
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


def allocate_samples_mfmc(cov, costs, target_cost):
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

    log10_variance : float
        The base 10 logarithm of the variance of the estimator
    """

    nmodels = cov.shape[0]
    corr = compute_correlations_from_covariance(cov)
    II = np.argsort(np.absolute(corr[0, 1:]))[::-1]
    if not np.allclose(II, np.arange(nmodels-1)):
        msg = 'Models must be ordered with decreasing correlation with '
        msg += 'high-fidelity model'
        raise Exception(msg)

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
    # nhf_samples = max(nhf_samples, 1)
    nsample_ratios = r[1:]

    gamma = get_variance_reduction(get_rsquared_mfmc, cov, nsample_ratios)
    log10_variance = np.log10(gamma)+np.log10(cov[0, 0])-np.log10(
        nhf_samples)
    return np.atleast_1d(nsample_ratios), log10_variance


def allocate_samples_mlmc(cov, costs, target_cost):
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

    log10_variance : float
        The base 10 logarithm of the variance of the estimator
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

    gamma = get_variance_reduction(get_rsquared_mlmc, cov, nsample_ratios)
    log10_variance = np.log10(gamma)+np.log10(cov[0, 0])-np.log10(
        nhf_samples)
    assert np.allclose(variance, 10**(log10_variance))
    # print(log10_variance)
    if np.isnan(log10_variance):
        raise Exception('MLMC variance is NAN')
    return np.atleast_1d(nsample_ratios), log10_variance


def get_lagrange_multiplier_mlmc(cov, costs, target_cost, eps):
    r"""
    Given an optimal sample allocation recover the optimal value of the
    Lagrange multiplier. This is only used for testing
    """
    nmodels = cov.shape[0]
    var_deltas = np.empty(nmodels)
    for ii in range(nmodels-1):
        var_deltas[ii] = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
    var_deltas[nmodels-1] = cov[nmodels-1, nmodels-1]
    cost_deltas = np.empty(nmodels)
    cost_deltas[:nmodels-1] = (costs[:nmodels-1] + costs[1:nmodels])
    cost_deltas[nmodels-1] = costs[nmodels-1]
    var_cost_prods = var_deltas * cost_deltas
    lagrange_mult = target_cost / np.sqrt(var_cost_prods).sum()
    return lagrange_mult


def get_discrepancy_covariances_IS(cov, nsample_ratios):
    r"""
    Get the covariances of the discrepancies :math:`\delta`
    between each low-fidelity model and its estimated mean when the same
    :math:`N` samples are used to compute the covariance between each models
    and :math:`N-r_\alpha` samples are allocated to
    estimate the low-fidelity means, and each of these sets are drawn
    independently from one another.

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The estimated covariance between each model.

    nsample_ratios : iterable (nmodels-1)
        The sample ratioss :math:`r_\alpha>1` for each low-fidelity model

    Results
    -------
    CF : np.ndarray (nmodels-1,nmodels-1)
        The matrix of covariances between the discrepancies :math:`\delta`

    cf : np.ndarray (nmodels-1)
        The vector of covariances between the discrepancies and the
        high-fidelity model.
    """
    nmodels = cov.shape[0]
    F = pkg_zeros(
        (nmodels-1, nmodels-1), type(nsample_ratios), dtype=pkg.double)
    f = pkg_zeros((nmodels-1), type(nsample_ratios), dtype=pkg.double)
    for ii in range(nmodels-1):
        F[ii, ii] = (nsample_ratios[ii]-1)/nsample_ratios[ii]
        for jj in range(ii+1, nmodels-1):
            F[ii, jj] = (nsample_ratios[ii]-1)/nsample_ratios[ii] * (
                nsample_ratios[jj]-1)/nsample_ratios[jj]
            F[jj, ii] = F[ii, jj]
        f[ii] = F[ii, ii]

    CF = cov[1:, 1:] * F
    cf = f * cov[1:, 0]
    return CF, cf


def get_discrepancy_covariances_MF(cov, nsample_ratios):
    r"""
    Get the covariances of the discrepancies :math:`\delta`
    between each low-fidelity model and its estimated mean using the MFMC
    sampling strategy.

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The estimated covariance between each model.

    nsample_ratios : iterable (nmodels-1)
        The sample ratioss :math:`r_\alpha>1` for each low-fidelity model

    Results
    -------
    CF : np.ndarray (nmodels-1,nmodels-1)
        The matrix of covariances between the discrepancies :math:`\delta`

    cf : np.ndarray (nmodels-1)
        The vector of covariances between the discrepancies and the
        high-fidelity model.
    """
    nmodels = cov.shape[0]
    F = pkg_zeros(
        (nmodels-1, nmodels-1), type(nsample_ratios), dtype=pkg.double)
    f = pkg_zeros(nmodels-1, type(nsample_ratios), dtype=pkg.double)
    for ii in range(nmodels-1):
        for jj in range(nmodels-1):
            rr = min(nsample_ratios[ii], nsample_ratios[jj])
            F[ii, jj] = (rr - 1) / rr
        f[ii] = F[ii, ii]
    CF = cov[1:, 1:] * F
    cf = f * cov[1:, 0]
    return CF, cf


def get_discrepancy_covariances_KL(cov, nsample_ratios, K, L):
    r"""
    Get the covariances of the discrepancies :math:`\delta`
    between each low-fidelity model and its estimated mean using the MFMC
    sampling strategy and the ACV KL estimator.

    The ACV-KL estimator partitions all of the control variates into two
    groups; the first K variables form a K -level approximate control
    variate, and the last :math:`M-K` variables are used to reduce the variance
    of estimating :math:`\mu_L` some :math:`L \le K` . The resulting estimator
    accelerates convergence to OCV-K , and L provides a degree of freedom
    for targeting a control variate level that contributes the greatest to
    the estimator variance.

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The estimated covariance between each model.

    nsample_ratios : iterable (nmodels-1)
        The sample ratioss :math:`r_\alpha>1` for each low-fidelity model

    K : integer (K<=nmodels-1)
        The number of effective control variates.

    L : integer (1<=L<=K+1)
        The id of the models whose mean is being targeted by the
        remaining nmodels-K low fidelity models.

    Results
    -------
    CF : np.ndarray (nmodels-1,nmodels-1)
        The matrix of covariances between the discrepancies :math:`\delta`

    cf : np.ndarray (nmodels-1)
        The vector of covariances between the discrepancies and the
        high-fidelity model.
    """
    nmodels = cov.shape[0]
    assert L <= K+1 and L >= 1 and K < nmodels
    K, L = K-1, L-1
    F = pkg_zeros(
        (nmodels-1, nmodels-1), type(nsample_ratios), dtype=pkg.double)
    f = pkg_zeros(nmodels-1, type(nsample_ratios), dtype=pkg.double)
    rs = nsample_ratios
    for ii in range(nmodels-1):
        # Diagonal terms
        if ii <= K:
            F[ii, ii] = (rs[ii]-1)/(rs[ii]+1e-20)
        else:
            F[ii, ii] = (rs[ii]-rs[L])/(rs[ii]*rs[L])
        # Off-diagonal terms
        for jj in range(ii+1, nmodels-1):
            if (ii <= K) and (jj <= K):
                ri = min(rs[ii], rs[jj])
                F[ii, jj] = (ri - 1) / (ri + 1e-20)
            elif (jj > K) and (ii > K):
                ri = min(rs[ii], rs[jj])
                t1 = (rs[ii]-rs[L])*(rs[jj]-rs[L])/(rs[ii]*rs[jj]*rs[L]
                                                    + 1e-20)
                t2 = (ri - rs[L]) / (rs[ii] * rs[jj] + 1e-20)
                F[ii, jj] = t1 + t2
            elif (ii > L) and (ii <= K) and (jj > K):
                F[ii, jj] = (rs[ii] - rs[L]) / (rs[ii] * rs[L] + 1e-20)
            elif (jj > L) and (jj <= K) and (ii > K):
                F[ii, jj] = (rs[jj] - rs[L]) / (rs[jj] * rs[L] + 1e-20)
            else:
                F[ii, jj] = 0.0
            F[jj, ii] = F[ii, jj]
        f[ii] = F[ii, ii]

    CF = cov[1:, 1:] * F
    cf = f * cov[1:, 0]
    return CF, cf


def get_control_variate_weights(cov):
    r"""
    Get the weights used by the control variate estimator with known low
    fidelity means.

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The estimated covariance between each model.

    Returns
    -------
    weights : np.ndarray (nmodels-1)
        The control variate weights
    """
    weights = -np.linalg.solve(cov[1:, 1:], cov[0, 1:])
    return weights


def get_approximate_control_variate_weights(CF, cf):
    r"""
    Get the weights used by the approximate control variate estimator.

    Parameters
    ----------
    CF : np.ndarray (nmodels-1,nmodels-1)
        The matrix of covariances between the discrepancies :math:`\delta`

    cf : np.ndarray (nmodels-1)
        The vector of covariances between the discrepancies and the
        high-fidelity model.

    Returns
    -------
    weights : np.ndarray (nmodels-1)
        The control variate weights
    """
    if type(CF) == np.ndarray:
        weights = -np.linalg.solve(CF, cf)
    else:
        if CF.shape == (1, 1):
            weights = -cf/CF[0, 0]
        else:
            weights = -pkg.linalg.solve(CF, cf)
    return weights


def get_rsquared_acv(cov, nsample_ratios, get_discrepancy_covariances):
    r"""
    Compute r^2 used to compute the variance reduction of
    Approximate Control Variate Algorithms

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    get_discrepancy_covariances : callable
        Function that returns the covariances of the control variate
        discrepancies. Functions must have the signature
        CF,cf = get_discrepancy_covariances(cov,nsample_ratios)

    Returns
    -------
    rsquared : float
        The value r^2
    """
    CF, cf = get_discrepancy_covariances(cov, nsample_ratios)
    try:
        weights = get_approximate_control_variate_weights(CF, cf)
        rsquared = -cf.dot(weights)/cov[0, 0]
    except RuntimeError:
        rsquared = -1e-16
    return rsquared


def acv_sample_allocation_gmf_ratio_constraint(ratios, *args):
    model_idx, parent_idx, target_cost, costs = args
    assert parent_idx > 0 and model_idx > 0
    eps = 0
    nhf_samples = get_nhf_samples(target_cost, costs, ratios)
    # ratios are only for low-fidelity models so use index-1
    # return nhf_samples*(ratios[model_idx-1]-ratios[parent_idx-1])-(1+eps)
    return (nhf_samples*ratios[model_idx-1]-nhf_samples*ratios[parent_idx-1] -
            (1+eps))


def acv_sample_allocation_gmf_ratio_constraint_jac(ratios, *args):
    model_idx, parent_idx, target_cost, costs = args
    assert parent_idx > 0 and model_idx > 0
    nhf_samples = get_nhf_samples(
        target_cost, costs, ratios)
    # ratios are only for low-fidelity models so use index-1
    grad = -costs[1:]*nhf_samples**2/target_cost*(
        ratios[model_idx-1]-ratios[parent_idx-1])
    grad[model_idx-1] += nhf_samples
    grad[parent_idx-1] -= nhf_samples
    return grad


def acv_sample_allocation_nlf_gt_nhf_ratio_constraint(ratios, *args):
    model_idx, target_cost, costs = args
    assert model_idx > 0
    eps = 0
    nhf_samples = get_nhf_samples(target_cost, costs, ratios)
    # print(model_idx, nhf_samples)
    # ratios are only for low-fidelity models so use index-1
    val = nhf_samples*ratios[model_idx-1]-nhf_samples-(1+eps)
    # print(val)
    return val


def acv_sample_allocation_nlf_gt_nhf_ratio_constraint_jac(ratios, *args):
    model_idx, target_cost, costs = args
    assert model_idx > 0
    nhf_samples = get_nhf_samples(
        target_cost, costs, ratios)
    # ratios are only for low-fidelity models so use index-1
    grad = -costs[1:]*nhf_samples**2/target_cost*(ratios[model_idx-1]-1)
    grad[model_idx-1] += nhf_samples
    return grad


def acv_sample_allocation_nhf_samples_constraint(ratios, *args):
    target_cost, costs = args
    # add to ensure that when constraint is violated by small numerical value
    # nhf samples generated from ratios will be greater than 1
    nhf_samples = get_nhf_samples(target_cost, costs, ratios)
    eps = 0
    return nhf_samples-(1+eps)


def acv_sample_allocation_nhf_samples_constraint_jac(ratios, *args):
    target_cost, costs = args
    nhf_samples = get_nhf_samples(
        target_cost, costs, ratios)
    grad = -costs[1:]*nhf_samples**2/target_cost
    return grad


def generate_samples_and_values_acv_IS(nsamples_per_model,
                                       functions, generate_samples):
    # nmodels = len(nsample_ratios)+1
    nmodels = nsamples_per_model.shape[0]
    nhf_samples = nsamples_per_model[0]
    if not callable(functions):
        assert len(functions) == nmodels
    samples1 = [None]+[generate_samples(nhf_samples)]*nmodels
    samples2 = [samples1[1]]+[np.hstack(
         [samples1[ii], generate_samples(
             int(nsamples_per_model[ii]-nhf_samples))])
         for ii in range(1, nmodels)]

    values2 = [f(s) for f, s in zip(functions, samples2)]
    values1 = [functions[0](samples2[0])]
    values1 = [None]+[values2[ii][:nhf_samples]
                      for ii in range(1, nmodels)]
    samples = [[s1, s2] for s1, s2 in zip(samples1, samples2)]
    values = [[v1, v2] for v1, v2 in zip(values1, values2)]
    return samples, values


def generate_samples_and_values_mlmc(nsamples_per_model, functions,
                                     generate_samples):
    r"""
    Parameters
    ----------
    nsamples_per_model : np.ndarray (nsamples)
            The number of samples allocated to each model

    functions : list of callables
        The functions used to evaluate each model

    generate_samples : callable
        Function used to generate realizations of the random variables

    Returns
    -------
    samples : list
        List containing the samples :math:`\mathcal{Z}_{i,1}` and
        :math:`\mathcal{Z}_{i,2}` for each model :math:`i=0,\ldots,M-1`.
        The list is [[:math:`\mathcal{Z}_{0,1}`,:math:`\mathcal{Z}_{0,2}`],...,[:math:`\mathcal{Z}_{M-1,1}`,:math:`\mathcal{Z}_{M-1,2}`]],
        where :math:`M` is the number of models

    values : list
        Model values at the points in samples
    """
    nmodels = len(nsamples_per_model)
    if not callable:
        assert nmodels == len(functions)
    nhf_samples = nsamples_per_model[0]
    samples1 = [None]
    samples2 = [generate_samples(nhf_samples)]
    prev_samples = samples2[0]
    for ii in range(nmodels-1):
        # total_samples = nsample_ratios[ii] * nhf_samples
        total_samples = nsamples_per_model[ii+1]
        assert total_samples/int(total_samples) == 1.0
        total_samples = int(total_samples)
        samples1.append(prev_samples)
        nnew_samples = total_samples - prev_samples.shape[1]
        samples2.append(generate_samples(nnew_samples))
        prev_samples = samples2[-1]

    values2 = [functions[0](samples2[0])]
    values1 = [None]
    for ii in range(1, nmodels):
        values1.append(functions[ii](samples1[ii]))
        values2.append(functions[ii](samples2[ii]))
    samples = [[s1, s2] for s1, s2 in zip(samples1, samples2)]
    values = [[v1, v2] for v1, v2 in zip(values1, values2)]

    return samples, values


def get_mfmc_control_variate_weights(cov):
    weights = -cov[0, 1:]/np.diag(cov[1:, 1:])
    return weights


def generate_samples_and_values_acv_KL(nsamples_per_model, functions,
                                       generate_samples, K, L):
    r"""
    Parameters
    ----------
    K : integer (K<=nmodels-1)
        The number of effective control variates.

    L : integer (1<=L<=K+1)
        The id of the models whose mean is being targeted by the
        remaining nmodels-K low fidelity models.
    """
    # nsample_ratios = np.asarray(nsample_ratios)
    # nlf_samples = validate_nsample_ratios(nhf_samples, nsample_ratios)
    # nmodels = nsample_ratios.shape[0]+1
    nmodels = nsamples_per_model.shape[0]
    nhf_samples, nlf_samples = nsamples_per_model[0], nsamples_per_model[1:]
    assert L <= K+1 and L >= 1 and K < nmodels
    K, L = K-1, L-1

    max_nsamples = nlf_samples.max()
    samples = generate_samples(max_nsamples)
    samples1 = [None]
    samples2 = [samples[:, :nhf_samples]]
    nprev_samples1 = nhf_samples
    # nprev_samples_total = nhf_samples
    for ii in range(1, nmodels):
        samples1.append(samples[:, :nprev_samples1])
        samples2.append(samples[:, :nlf_samples[ii-1]])
        if (ii <= K):
            nprev_samples1 = nhf_samples
        else:
            nprev_samples1 = nlf_samples[L]
        # nprev_samples_total = nlf_samples[ii-1]

    values2 = [functions[0](samples1[0])]
    values1 = [None]
    for ii in range(1, nmodels):
        values_ii = functions[ii](samples2[ii])
        values1.append(values_ii[:samples1[ii].shape[1]])
        values2.append(values_ii)

    samples = [[s1, s2] for s1, s2 in zip(samples1, samples2)]
    values = [[v1, v2] for v1, v2 in zip(values1, values2)]

    return samples, values


def get_sample_allocation_matrix_mlmc(nmodels):
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


def get_npartition_samples_mlmc(nsamples_per_model):
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
    npartition_samples = pkg_empty(
        (nmodels), type(nsamples_per_model), dtype=pkg.double)
    npartition_samples[0] = nsamples_per_model[0]
    for ii in range(1, nmodels):
        npartition_samples[ii] = (
            nsamples_per_model[ii]-npartition_samples[ii-1])
    return npartition_samples


def get_sample_allocation_matrix_mfmc(nmodels):
    mat = np.zeros((nmodels, 2*nmodels))
    mat[0, 1:] = 1
    for ii in range(1, nmodels):
        mat[ii, 2*ii+1:] = 1
    return mat


def get_npartition_samples_mfmc(nsamples_per_model):
    npartition_samples = pkg_hstack(
        (nsamples_per_model[0], pkg_diff(nsamples_per_model)))
    return npartition_samples


def get_sample_allocation_matrix_acvmf(recursion_index):
    nmodels = len(recursion_index)+1
    mat = np.zeros((nmodels, 2*nmodels))
    for ii in range(nmodels):
        mat[ii, 2*ii+1] = 1.0
    for ii in range(1, nmodels):
        mat[:, 2*ii] = mat[:, recursion_index[ii-1]*2+1]
    for ii in range(2, 2*nmodels):
        II = np.where(mat[:, ii] == 1)[0][-1]
        mat[:II, ii] = 1.0
    return mat


def get_npartition_samples_acvmf(nsamples_per_model):
    nmodels = len(nsamples_per_model)
    II = np.unique(to_numpy(nsamples_per_model), return_index=True)[1]
    sort_array = nsamples_per_model[II]
    if sort_array.shape[0] < nmodels:
        pad = sort_array[-1]*pkg_ones(
            (nmodels-sort_array.shape[0]), type(sort_array),
            dtype=pkg.double)
        sort_array = pkg_hstack((sort_array, pad))
    npartition_samples = pkg_hstack(
        (nsamples_per_model[0], pkg_diff(sort_array)))
    return npartition_samples


def get_sample_allocation_matrix_acvis(recursion_index):
    nmodels = len(recursion_index)+1
    mat = np.zeros((nmodels, 2*nmodels))
    for ii in range(nmodels):
        mat[ii, 2*ii+1] = 1
    for ii in range(1, nmodels):
        mat[:, 2*ii] = mat[:, recursion_index[ii-1]*2+1]
    for ii in range(1, nmodels):
        mat[:, 2*ii+1] = np.maximum(mat[:, 2*ii], mat[:, 2*ii+1])
    return mat


def get_npartition_samples_acvis(nsamples_per_model):
    r"""
    Get the size of the subsets :math:`z_i\setminus z_i^\star, i=0\ldots, M-1`.

    # Warning this will likely not work when recursion index is not [0, 0]
    """
    npartition_samples = pkg_hstack(
        (nsamples_per_model[0], nsamples_per_model[1:]-nsamples_per_model[0]))
    return npartition_samples


def get_nsamples_intersect(reorder_allocation_mat, npartition_samples):
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
    nmodels = reorder_allocation_mat.shape[0]
    nsubset_samples = npartition_samples[:, None] * reorder_allocation_mat
    nsamples_intersect = pkg_zeros(
        (2*nmodels, 2*nmodels), type(npartition_samples), dtype=pkg.double)
    for ii in range(2*nmodels):
        nsamples_intersect[ii] = (
            nsubset_samples[reorder_allocation_mat[:, ii] == 1]).sum(axis=0)
    return nsamples_intersect


def get_nsamples_interesect_from_z_subsets_acvgmf(
        nsamples_z_subsets, recursion_index):
    r"""
    Parameters
    ----------
    nsamples_z_subsets : np.ndarray (nmodels)
        The number of samples in the subset :math:`z_i`
       (not :math::`z_i^\star`) for each model
    """
    nmodels = len(recursion_index)+1
    nsamples_intersect = np.zeros((2*nmodels, 2*nmodels))
    nsamples_intersect[0, :] = 0
    nsamples_intersect[:, 0] = 0
    nsamples_intersect[1, 1:] = nsamples_z_subsets[0]
    nsamples_intersect[1:, 1] = nsamples_z_subsets[0]
    for ii in range(1, nmodels):
        for jj in range(1, nmodels):
            nsamples_intersect[2*ii, 2*jj] = min(
                nsamples_z_subsets[recursion_index[ii-1]],
                nsamples_z_subsets[recursion_index[jj-1]])
            nsamples_intersect[2*ii+1, 2*jj] = min(
                nsamples_z_subsets[ii],
                nsamples_z_subsets[recursion_index[jj-1]])
            nsamples_intersect[2*ii, 2*jj+1] = min(
                nsamples_z_subsets[recursion_index[ii-1]],
                nsamples_z_subsets[jj])
            nsamples_intersect[2*ii+1, 2*jj+1] = min(
                nsamples_z_subsets[ii], nsamples_z_subsets[jj])
    return nsamples_intersect


def get_nsamples_subset(reorder_allocation_mat, npartition_samples):
    r"""
    Get the number of samples allocated to the sample subsets
    :math:`|z^\star_i` and :math:`|z_i|`

    npartition_samples : np.ndarray (nmodels)
        The size of the partitions that make up the subsets
        :math:`z_i, i=0\ldots, M-1`. These are represented by different
        color blocks in the ACV papers figures of sample allocation
    """
    nmodels = reorder_allocation_mat.shape[0]
    nsamples_subset = pkg_zeros((
        2*nmodels), type(npartition_samples), dtype=pkg.double)
    for ii in range(2*nmodels):
        nsamples_subset[ii] = \
            npartition_samples[reorder_allocation_mat[:, ii] == 1].sum()
    return nsamples_subset


def reorder_allocation_matrix_acvgmf(allocation_mat, nsamples_per_model,
                                     recursion_index):
    """
    Allocation matrix is the reference sample allocation

    Must make sure that allocation matrix used for sample allocation and
    computing estimated variances has the largest sample sizes containing
    the largest subset

    """
    # WARNING Will only work for acvmf and not acvgis
    II = np.unique(to_numpy(nsamples_per_model[1:]), return_inverse=True)[1]+1
    tmp = pkg_copy(allocation_mat)
    tmp[:, 3::2] = allocation_mat[:, 2*II+1]
    tmp[:, 2::2] = tmp[:, 2*recursion_index+1]
    return tmp


def get_acv_discrepancy_covariances_multipliers(
        allocation_mat, nsamples_per_model, get_npartition_samples,
        recursion_index):
    nmodels = allocation_mat.shape[0]
    reorder_allocation_mat = reorder_allocation_matrix_acvgmf(
        allocation_mat, nsamples_per_model, recursion_index)
    npartition_samples = get_npartition_samples(nsamples_per_model)
    assert np.all(to_numpy(npartition_samples) >= 0), (
        npartition_samples)
    nsamples_intersect = get_nsamples_intersect(
        reorder_allocation_mat, npartition_samples)
    nsamples_subset = get_nsamples_subset(
        reorder_allocation_mat, npartition_samples)
    Gmat = pkg_zeros(
        (nmodels-1, nmodels-1), type(npartition_samples), dtype=pkg.double)
    gvec = pkg_zeros((nmodels-1), type(npartition_samples), dtype=pkg.double)
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


def get_acv_discrepancy_covariances(cov, Gmat, gvec):
    return Gmat*cov[1:, 1:], gvec*cov[0, 1:]


def get_generalized_approximate_control_variate_weights(
        allocation_mat, nsamples_per_model, get_npartition_samples,
        cov, recursion_index):
    Gmat, gvec = get_acv_discrepancy_covariances_multipliers(
        allocation_mat, nsamples_per_model, get_npartition_samples,
        recursion_index)
    nhf_samples = nsamples_per_model[0]
    Fmat, fvec = Gmat*nhf_samples, gvec*nhf_samples
    CF, cf = get_acv_discrepancy_covariances(cov, Fmat, fvec)
    try:
        weights = get_approximate_control_variate_weights(CF, cf)
    except np.linalg.LinAlgError as err:
        # print("linalgerror: acv weights failed")
        weights = pkg_ones(cf.shape, type(cf), pkg.double)*1e16
    except RuntimeError as err:
        # print("runtime: acv weights failed")
        weights = pkg_ones(cf.shape, type(cf), pkg.double)*1e16
    return weights, cf


def acv_estimator_variance(allocation_mat, target_cost, costs,
                           get_npartition_samples, cov, recursion_index,
                           nsample_ratios):
    nsamples_per_model = get_nsamples_per_model(
        target_cost, costs, nsample_ratios, False)
    weights, cf = get_generalized_approximate_control_variate_weights(
            allocation_mat, nsamples_per_model,
            get_npartition_samples, cov, recursion_index)
    rsquared = -cf.dot(weights)/cov[0, 0]
    # assert rsquared <= 1 or np.all(weights == 1e16), (rsquared, weights)
    variance_reduction = (1-rsquared)
    variance = variance_reduction*cov[0, 0]/nsamples_per_model[0]
    return variance


def generate_samples_and_values_mfmc(nsamples_per_model, functions,
                                     generate_samples, acv_modification=False):
    r"""
    Parameters
    ----------
    nsamples_per_model : np.ndarray (nsamples)
            The number of samples allocated to each model

    functions : list of callables
        The functions used to evaluate each model

    generate_samples : callable
        Function used to generate realizations of the random variables

    Returns
    -------
    samples : list
        List containing the samples :math:`\mathcal{Z}_{i,1}` and
        :math:`\mathcal{Z}_{i,2}` for each model :math:`i=0,\ldots,M-1`.
        The list is [[:math:`\mathcal{Z}_{0,1}`,:math:`\mathcal{Z}_{0,2}`],...,[:math:`\mathcal{Z}_{M-1,1}`,:math:`\mathcal{Z}_{M-1,2}`]],
        where :math:`M` is the number of models

    values : list
        Model values at the points in samples

    """
    # nsample_ratios = np.asarray(nsample_ratios)
    # nlf_samples = validate_nsample_ratios(nhf_samples, nsample_ratios)
    # nmodels = nsample_ratios.shape[0]+1
    nmodels = nsamples_per_model.shape[0]

    # max_nsamples = nlf_samples.max()
    nhf_samples = nsamples_per_model[0]
    max_nsamples = nsamples_per_model[1:].max()
    samples = generate_samples(max_nsamples)
    samples2 = [samples[:, :nhf_samples]]
    samples1 = [None]
    nprev_samples = nhf_samples
    for ii in range(1, nmodels):
        samples1.append(samples[:, :nprev_samples])
        # samples2.append(samples[:, :nlf_samples[ii-1]])
        samples2.append(samples[:, :nsamples_per_model[ii]])
        if acv_modification:
            nprev_samples = nhf_samples
        else:
            nprev_samples = samples2[ii].shape[1]

    values2 = [functions[0](samples2[0])]
    values1 = [None]
    for ii in range(1, nmodels):
        values_ii = functions[ii](samples2[ii])
        values1.append(values_ii[:samples1[ii].shape[1]])
        values2.append(values_ii)

    samples = [[s1, s2] for s1, s2 in zip(samples1, samples2)]
    values = [[v1, v2] for v1, v2 in zip(values1, values2)]

    return samples, values


def acv_sample_allocation_objective_all(
        estimator, target_cost, x, return_grad=False):
    if use_torch:
        ratios = torch.tensor(x, dtype=pkg.double)
        if return_grad:
            ratios.requires_grad = True
    else:
        ratios = x
    variance = estimator._get_variance_for_optimizer(target_cost, ratios)
    log_10_var = pkg.log10(variance)
    if not return_grad:
        if use_torch:
            return log_10_var.item()
        return log_10_var
    log_10_var.backward()
    grad = ratios.grad.detach().numpy().copy()
    ratios.grad.zero_()
    # print(x, log_10_var.item(), grad)
    return log_10_var.item(), grad


def mlmc_sample_allocation_objective_all_lagrange(
        estimator, target_variance, costs, xrats):
    nhf_samples, ratios, lagrange_mult = xrats[0], xrats[1:-1], xrats[-1]
    total_cost = nhf_samples*costs[0] + ((ratios*nhf_samples).dot(costs[1:]))
    var_red = estimator._variance_reduction(estimator.cov, ratios)
    variance = var_red*estimator.cov[0, 0]/nhf_samples
    obj = total_cost+lagrange_mult**2*(variance-target_variance)
    if use_torch:
        obj = torch.log10(obj)
        return obj.item()
    else:
        return np.log10(obj)


def mlmc_sample_allocation_jacobian_all_lagrange_torch(
        estimator, target_variance, costs, x):
    xrats = torch.tensor(x, dtype=pkg.double, requires_grad=True)
    # xrats.requires_grad = True
    nhf_samples, ratios, lagrange_mult = xrats[0], xrats[1:-1], xrats[-1]
    total_cost = costs[0]*nhf_samples+costs[1:].dot(ratios*nhf_samples)
    var_red = estimator._variance_reduction(
        _ndarray_as_pkg_format(estimator.cov), ratios)
    variance = var_red*estimator.cov[0, 0]/nhf_samples
    obj = total_cost+lagrange_mult**2*(variance-target_variance)
    obj = torch.log10(obj)
    obj.backward()
    grad = xrats.grad.detach().numpy().copy()
    xrats.grad.zero_()
    return grad


def get_acv_initial_guess(initial_guess, cov, costs, target_cost):
    if initial_guess is not None:
        constraint_val = acv_sample_allocation_nhf_samples_constraint(
            initial_guess, target_cost, costs)
        if constraint_val < 0:
            raise ValueError("Not a feasiable initial guess")
        return initial_guess

    nmodels = len(costs)
    return np.arange(2, nmodels+1)

    # # nmodels = cov.shape[0]
    # # nratios = np.empty(nmodels-1)
    # # for ii in range(1, nmodels):
    # #     idx = np.array([0, ii])
    # #     nratio = allocate_samples_mfmc(
    # #         cov[np.ix_(idx, idx)], costs[idx], target_cost)[0]
    # #     nratios[ii-1] = nratio

    # # scale ratios so that nhf_samples is one
    # nhf_samples = 1
    # # use (1-1e-8) to avoid numerical precision problems so that
    # # acv_sample_allocation_nhf_samples_constraint is always positive
    # delta = (target_cost*(1-1e-8) - costs[0]*nhf_samples)/nratios.dot(
    #     costs[1:])
    # initial_guess = np.array(nratios)*delta

    # constraint_val = acv_sample_allocation_nhf_samples_constraint(
    #     initial_guess, target_cost, costs)
    # if constraint_val < 0:
    #     raise ValueError("Not a feasiable initial guess")

    return initial_guess


def solve_allocate_samples_acv_slsqp_optimization(
        estimator, costs, target_cost, initial_guess, optim_options, cons):
    if optim_options is None:
        optim_options = {'disp': True, 'ftol': 1e-10,
                         'maxiter': 10000, 'iprint': 0}
        # set iprint=2 to printing iteration info

    if target_cost < costs.sum():
        msg = "Target cost does not allow at least one sample from each model"
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

    return_grad = False
    if use_torch:
        return_grad = True
    method = "SLSQP"
    # method = "trust-constr"
    # print(optim_options)
    # del optim_options['ftol']
    # del optim_options['iprint']
    # optim_options["maxiter"] = 10000
    # optim_options["gtol"] = 1e-6
    opt = minimize(
        partial(estimator.objective, target_cost, return_grad=return_grad),
        initial_guess, method=method, jac=return_grad,
        bounds=bounds, constraints=cons, options=optim_options)
    return opt


def allocate_samples_acv(cov, costs, target_cost, estimator,
                         cons=[],
                         initial_guess=None,
                         optim_options=None, optim_method='SLSQP'):
    r"""
    Determine the samples to be allocated to each model

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest
        fidelity model is the first model, i.e its variance is cov[0,0]

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    target_cost : float
        The total cost budget

    Returns
    -------
    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i=r_i*nhf_samples,i=1,...,nmodels-1

    log10_variance : float
        The base 10 logarithm of the variance of the estimator
    """
    initial_guess = get_acv_initial_guess(
        initial_guess, cov, costs, target_cost)
    assert optim_method == "SLSQP"
    opt = solve_allocate_samples_acv_slsqp_optimization(
        estimator, costs, target_cost, initial_guess, optim_options, cons)
    nsample_ratios = opt.x
    if not opt.success:# or opt.nit == 1: nit will be 1 if using only two models
        # for con in cons:
        #     print(con['fun'], con['fun'](opt.x, *con['args']))
        raise RuntimeError('SLSQP optimizer failed'+f'{opt}')
        # var = 1e99
        # log10_var = 99
    else:
        var = estimator.get_variance(target_cost, nsample_ratios)
        log10_var = np.log10(var.item())
    # print('Optimized Variance', 10**opt.fun, 10**log10_var)
    # assert np.allclose(log10_var, opt.fun)
    return nsample_ratios, log10_var


def get_rsquared_acv_KL_best(cov, nsample_ratios):
    nmodels = cov.shape[1]
    opt_rsquared = -1
    # KL = None
    for K in range(1, nmodels):
        for L in range(1, K+1):
            get_discrepancy_covariances = partial(
                get_discrepancy_covariances_KL, K=K, L=L)
            get_rsquared = partial(
                get_rsquared_acv,
                get_discrepancy_covariances=get_discrepancy_covariances)
            rsquared = get_rsquared(cov, nsample_ratios)
            # print(K,L,rsquared)
            if rsquared > opt_rsquared:
                opt_rsquared = rsquared
                # KL = (K, L)
    return opt_rsquared


def estimate_model_ensemble_covariance(npilot_samples, generate_samples,
                                       model_ensemble, nmodels):
    r"""
    Estimate the covariance of a model ensemble from a set of pilot samples

    Parameters
    ----------
    npilot_samples : integer
        The number of samples used to estimate the covariance

    generate_samples : callable
        Function used to generate realizations of the random variable with
        call signature samples = generate_samples(npilot_samples)

    model_ensemble : callable or np.ndarray  (nvars, nsamples)
        Function that takes a set of samples and models ids and evaluates
        a set of models. See ModelEnsemble.
        call signature values = model_emsemble(samples)

        relizations of the random variable

    nmodels : integer
        The number of models in the ensemble

    Returns
    -------
    cov : np.ndarray (nqoi,nqoi)
        The covariance between the model qoi

    pilot_random_samples : np.ndarray (nvars,npilot_samples)
        The random samples used to compute the covariance. These samples
        DO NOT have a model id

    pilot_values : np.ndaray (npilot_samples,nmodels)
        The values of each model at the pilot samples
    """
    # generate pilot samples
    if callable(generate_samples):
        pilot_random_samples = generate_samples(npilot_samples)
    else:
        pilot_random_samples = generate_samples[:, :npilot_samples]
    config_vars = np.arange(nmodels)[np.newaxis, :]
    # append model ids to pilot smaples
    pilot_samples = get_all_sample_combinations(
        pilot_random_samples, config_vars)
    # evaluate models at pilot samples
    pilot_values = model_ensemble(pilot_samples)
    pilot_values = np.reshape(
        pilot_values, (npilot_samples, nmodels))
    # compute covariance
    cov = np.cov(pilot_values, rowvar=False)
    return cov, pilot_random_samples, pilot_values


def compute_covariance_from_control_variate_samples(values):
    r"""
    Compute the covariance between information sources from a set
    of evaluations of each information source.

    Parameters
    ----------
    values : list (nmodels)
        The evaluations of each information source seperated in form
        necessary for control variate estimators.
        Each entry of the list contains

        values0 : np.ndarray (num_samples_i0,num_qoi)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

        values1: np.ndarray (num_samples_i1,num_qoi)
            Evaluations used compute the approximate
             mean :math:`\mu_{i,r_iN}` of the low fidelity models.

    Returns
    -------
    cov : np.ndarray (nmodels)
        The covariance between the information sources
    """
    shared_samples_values = np.hstack(
        [v[0].squeeze()[:, np.newaxis] for v in values])
    cov = np.cov(shared_samples_values, rowvar=False)
    # print(cov,'\n',cov_matrix)
    return cov


def plot_correlation_matrix(corr_matrix, ax=None, model_names=None,
                            format_string='{:1.3f}', cmap="jet"):
    """
    Plot a correlation matrix

    Parameters
    ----------
    corr_matrix : np.ndarray (nvars, nvars)
         The correlation between a set of random variabels
    """
    from pyapprox.util.configure_plots import plt
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.matshow(corr_matrix, cmap=cmap, aspect="auto")
    for (i, j), z in np.ndenumerate(corr_matrix):
        ax.text(j, i, format_string.format(z), ha='center', va='center',
                fontsize=12, color='w')
    plt.colorbar(im, ax=ax)
    if model_names is None:
        nmodels = corr_matrix.shape[0]
        model_names = [r"$f_{%d}$" % ii for ii in range(nmodels)]
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xticklabels(model_names, rotation=60)
    return ax


def get_nhf_samples(target_cost, costs, nsample_ratios):
    """
    Get the number of high-fidelity samples from a target cost and
    set of lower-fidelity sample ratios.

    Parameters
    ----------
    target_cost : float
        The total cost budget

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Returns
    -------
    nhf_samples : float
        The number of high-fidelity samples. This may not be an integer
        rouning is taken care of elsewhere
    """
    nhf_samples = target_cost/(
        costs[0]+(nsample_ratios*costs[1:]).sum())
    return nhf_samples


def get_nsamples_per_model(target_cost, costs, nsample_ratios, isinteger=True):
    """
    Compute the number of samples allocated to each model. The samples may not
    be integers.

    Parameters
    ----------
    target_cost : float
        The total cost budget

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of

    isinteger : boolean
        Ensure that nsamples are integers.

    Returns
    -------
    nsamples_per_model : np.ndarray (nmodels)
        The number of high-fidelity samples. This may not be an integer
        rouning is taken care of elsewhere
    """
    nhf_samples = get_nhf_samples(target_cost, costs, nsample_ratios)
    nsamples_per_model = pkg_hstack(
        [nhf_samples, nsample_ratios*nhf_samples])
    if isinteger:
        return cast_to_integers(nsamples_per_model)
    return nsamples_per_model


def round_nsample_ratios(target_cost, costs, nsample_ratios):
    """
    Return sample ratios that produce integer sample allocations.
    The cost of the returned allocation will not usually equal target cost

    Parameters
    ----------
    target_cost : float
        The total cost budget

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Returns
    -------
    nsample_ratios_floor : float
         ratios r used to specify INTEGER number of samples of the lower
         fidelity models. These ratios will also force nhf_samples to
         be an integer

    rounded_target_cost : float
         The cost of the new sample allocation
    """
    nsamples_float = get_nsamples_per_model(
        target_cost, costs, nsample_ratios, False)
    nsamples_floor = nsamples_float.astype(int)
    # ensure all low-fidelity samples > nhf_samples after rounding
    if nsamples_floor[0] < 1 and nsamples_float[0] < 1-1e-8:
        # print(nsample_ratios)
        # print(nsamples_floor[0], nsamples_float[0], nsamples_float[0]-1, costs)
        raise RuntimeError("Rounding likely caused nhf samples to be zero")
    elif nsamples_floor[0] < 1:
        nsamples_floor[0] = 1
    II = np.where(nsamples_floor[1:] == nsamples_floor[0])[0]+1
    nsamples_floor[II] = 2
    nsample_ratios_floor = nsamples_floor[1:]/nsamples_floor[0]
    rounded_target_cost = nsamples_floor[0]*(costs[0]+np.dot(
        nsample_ratios_floor, costs[1:]))
    # nhf_samples_floor = int(nsamples_float[0])
    # nlf_samples_floor = (
    #     nhf_samples_floor*np.asarray(nsample_ratios)).astype(int)
    # nsample_ratios_floor = nlf_samples_floor/nhf_samples_floor
    # rounded_target_cost = nhf_samples_floor*(costs[0]+np.dot(
    #     nsample_ratios_floor, costs[1:]))
    return nsample_ratios_floor, rounded_target_cost


def generate_samples_acv(reorder_allocation_mat, nsamples_per_model,
                         npartition_samples, generate_samples):
    r"""
    Generate the samples needded to evaluate the approximate the control
    variate estimator

    Parameters
    ----------
    allocation_mat : np.ndarray (nmodels, 2*nmodels)
        For columns :math:`2j, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i^\star\subseteq z_j^\star`
        For columns :math:`2j+1, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i\subseteq z_j`

    nsamples_per_model : np.ndarray (nsamples)
            The number of samples allocated to each model

    npartition_samples : np.ndarray (nmodels)
        The size of the partitions that make up the subsets
        :math:`z_i, i=0\ldots, M-1`. These are represented by different
        color blocks in the ACV papers figures of sample allocation

    generate_samples : callable
        Function used to generate realizations of the random variables with
        signature

        `generate_samples(nsamples) -> np.ndarray(nvars, nsamples)`

    Returns
    -------
    samples_per_model : list (nmodels)
            The ith entry contains the set of samples
            np.narray(nvars, nsamples_ii) used to evaluate the ith model.

    partition_indices_per_model : list (nmodels)
            The ith entry contains the indices np.narray(nsamples_ii)
            mapping each sample to a sample allocation partition
    """
    npartition_samples = cast_to_integers(npartition_samples)
    assert np.all(npartition_samples >= 0)
    nmodels = reorder_allocation_mat.shape[0]
    nsamples = npartition_samples.sum()
    samples = generate_samples(nsamples)
    nvars = samples.shape[0]
    ubs = np.cumsum([ss for ss in npartition_samples]).astype(int)
    lbs = np.hstack((0, ubs[:-1]))
    samples_per_model, partition_indices_per_model = [], []
    for ii in range(nmodels):
        active_partitions = np.where(
            (reorder_allocation_mat[:, 2*ii] == 1) |
            (reorder_allocation_mat[:, 2*ii+1] == 1))[0]
        nsamples_ii = npartition_samples[active_partitions].sum()
        samples_per_model_ii = np.empty((nvars, nsamples_ii))
        partition_indices_per_model_ii = np.empty((nsamples_ii))
        cnt = 0
        for idx in active_partitions:
            samples_per_model_ii[:, cnt:cnt+npartition_samples[idx]] = \
                samples[:, lbs[idx]:ubs[idx]]
            partition_indices_per_model_ii[
                cnt:cnt+npartition_samples[idx]] = idx
            cnt += npartition_samples[idx]
        samples_per_model.append(samples_per_model_ii)
        partition_indices_per_model.append(partition_indices_per_model_ii)
    return samples_per_model, partition_indices_per_model


def separate_model_values_acv(reorder_allocation_mat,
                              values_per_model, partition_indices_per_model):
    r"""
    Separate a list of model evaluations for each model into the separated
    form necessary to evaluate the approximate the control variate estimator

    Parameters
    ----------
    allocation_mat : np.ndarray (nmodels, 2*nmodels)
        For columns :math:`2j, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i^\star\subseteq z_j^\star`
        For columns :math:`2j+1, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i\subseteq z_j`

    values_per_model : list (nmodels)
            The ith entry contains the set of evaluations
            np.narray(nsamples_ii, nqoi) of the ith model.

    partition_indices_per_model : list (nmodels)
            The ith entry contains the indices np.narray(nsamples_ii)
            mapping each sample to a sample allocation partition

    Returns
    -------
    acv_values : list (nmodels)
        The evaluations of each information source seperated in form
        necessary for control variate estimators.
        Each entry of the list contains

        values0 : np.ndarray (num_samples_i0,num_qoi)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

        values1: np.ndarray (num_samples_i1,num_qoi)
            Evaluations used compute the approximate
            mean :math:`\mu_{i,r_iN}` of the low fidelity models.
    """
    nmodels = len(values_per_model)
    acv_values = []
    for ii in range(nmodels):
        active_partitions_ii_1 = np.where(
            reorder_allocation_mat[:, 2*ii] == 1)[0]
        values_ii_1_list = []
        for idx in active_partitions_ii_1:
            values_ii_1_list.append(
                values_per_model[ii][partition_indices_per_model[ii] == idx])
        if ii > 0:
            values_ii_1 = np.vstack(values_ii_1_list)
        else:
            values_ii_1 = None
        active_partitions_ii_2 = np.where(
            reorder_allocation_mat[:, 2*ii+1] == 1)[0]
        values_ii_2_list = []
        for idx in active_partitions_ii_2:
            values_ii_2_list.append(
                values_per_model[ii][partition_indices_per_model[ii] == idx])
        values_ii_2 = np.vstack(values_ii_2_list)
        acv_values.append([values_ii_1, values_ii_2])
    return acv_values


def separate_samples_per_model_acv(reorder_allocation_mat, samples_per_model,
                                   subset_indices_per_model):
    r"""
    Separate a list of samples for each model into the separated
    form necessary to evaluate the approximate the control variate estimator

    Parameters
    ----------
    allocation_mat : np.ndarray (nmodels, 2*nmodels)
        For columns :math:`2j, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i^\star\subseteq z_j^\star`
        For columns :math:`2j+1, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i\subseteq z_j`

    samples_per_model : list (nmodels)
            The ith entry contains the set of evaluations
            np.narray(nvars, nsamples_ii) of the ith model.

    partition_indices_per_model : list (nmodels)
            The ith entry contains the indices np.narray(nsamples_ii)
            mapping each sample to a sample allocation partition

    Returns
    -------
    acv_samples : list (nmodels)
        The samples for each information source seperated in form
        necessary for control variate estimators.
        Each entry of the list contains

        samples0 : np.ndarray (nvars, num_samples_i0)
           Evaluations  of each model
           used to compute the estimator :math:`Q_{i,N}` of

        samples1: np.ndarray (nvars, num_samples_i1)
            Evaluations used compute the approximate
            mean :math:`\mu_{i,r_iN}` of the low fidelity models.
    """
    nmodels = len(samples_per_model)
    tmp = []
    for ii in range(nmodels):
        tmp.append(samples_per_model[ii].T)
    acv_samples = separate_model_values_acv(
        reorder_allocation_mat, tmp, subset_indices_per_model)
    for ii in range(nmodels):
        if ii > 0:
            acv_samples[ii][0] = acv_samples[ii][0].T
        acv_samples[ii][1] = acv_samples[ii][1].T
    return acv_samples


def bootstrap_acv_estimator(values_per_model, partition_indices_per_model,
                            npartition_samples, reorder_allocation_mat,
                            acv_weights, nbootstraps=10):
    r"""
    Approximate the variance of the Monte Carlo estimate of the mean using
    bootstraping

    Parameters
    ----------
    acv_weights : np.ndarray (nmodels-1)
        The control variate weights

    nbootstraps : integer
        The number of boostraps used to compute estimator variance

    Returns
    -------
    bootstrap_mean : float
        The bootstrap estimate of the estimator mean

    bootstrap_var : float
        The bootstrap estimate of the estimator variance
    """
    nmodels = len(values_per_model)
    npartitions = len(npartition_samples)
    npartition_samples = cast_to_integers(npartition_samples)
    # preallocate memory so do not have to do it repeatedly
    permuted_partition_indices = [
        np.empty(npartition_samples[jj], dtype=int)
        for jj in range(npartitions)]
    permuted_values_per_model = [v.copy() for v in values_per_model]
    active_partitions = []
    for ii in range(nmodels):
        active_partitions.append(np.where(
            (reorder_allocation_mat[:, 2*ii] == 1) |
            (reorder_allocation_mat[:, 2*ii+1] == 1))[0])

    estimator_vals = np.empty((nbootstraps, 1))
    for kk in range(nbootstraps):
        for jj in range(npartitions):
            n_jj = npartition_samples[jj]
            permuted_partition_indices[jj][:] = (
                np.random.choice(np.arange(n_jj, dtype=int), size=(n_jj),
                                 replace=True))
        for ii in range(nmodels):
            for idx in active_partitions[ii]:
                II = np.where(partition_indices_per_model[ii] == idx)[0]
                permuted_values_per_model[ii][II] = values_per_model[ii][
                    II[permuted_partition_indices[idx]]]
        permuted_acv_values = separate_model_values_acv(
            reorder_allocation_mat, permuted_values_per_model,
            partition_indices_per_model)
        estimator_vals[kk] = \
            compute_approximate_control_variate_mean_estimate(
                acv_weights, permuted_acv_values)
    bootstrap_mean = estimator_vals.mean()
    bootstrap_var = estimator_vals.var()
    return bootstrap_mean, bootstrap_var


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


def update_list_for_reduce(mylist, indices):
    mylist[indices[0]].append(indices[1])
    return mylist


def generate_all_trees(children, root, tree_depth):
    if tree_depth < 2 or len(children) == 0:
        yield ModelTree(root, children)
    else:
        for prod in product((0, 1), repeat=len(children)):
            if not any(prod):
                continue
            nexts, sub_roots = reduce(
                update_list_for_reduce, zip(prod, children), ([], []))
            for q in product(range(len(sub_roots)), repeat=len(nexts)):
                sub_children = reduce(
                    update_list_for_reduce, zip(q, nexts),
                    [[] for ii in sub_roots])
                yield from [
                    ModelTree(root, list(children))
                    for children in product(
                            *(generate_all_trees(sc, sr, tree_depth-1)
                              for sr, sc in zip(sub_roots, sub_children)))]


def get_acv_recursion_indices(nmodels, depth=None):
    if depth is None:
        depth = nmodels-1
    if depth > nmodels-1:
        msg = f"Depth {depth} exceeds number of lower-fidelity models"
        raise ValueError(msg)
    for index in generate_all_trees(np.arange(1, nmodels), 0, depth):
        yield index.to_index()[1:]


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with
           other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    import networkx as nx
    if not nx.is_tree(G):
        msg = 'cannot use hierarchy_pos on a graph that is not a tree'
        raise TypeError(msg)

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0,
                       xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G, child, width=dx, vert_gap=vert_gap,
                    vert_loc=vert_loc-vert_gap, xcenter=nextx,
                    pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def plot_model_recursion(recursion_index, ax):
    nmodels = len(recursion_index)+1
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(nmodels))
    for ii, jj in enumerate(recursion_index):
        graph.add_edge(ii+1, jj)
    pos = hierarchy_pos(graph, 0, vert_gap=0.1, width=0.1)
    nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=[2000],
            font_size=24)


def plot_sample_allocation(reorder_allocation_mat, npartition_samples, ax,
                           colors=None):
    from pyapprox.util.configure_plots import plt
    nmodels = reorder_allocation_mat.shape[0]
    active_partitions = []
    for ii in range(nmodels):
        active_partitions.append(np.where(
            (reorder_allocation_mat[:, 2*ii] == 1) |
            (reorder_allocation_mat[:, 2*ii+1] == 1))[0])
    # remove z_0^* which is never nonzero
    reorder_allocation_mat = reorder_allocation_mat[:, 1:]
    index = np.arange(reorder_allocation_mat.shape[1])+0.3
    y_offset = 0
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 0.5, 2*nmodels))
    assert len(colors) == 2*nmodels
    for jj in range(len(npartition_samples)):
        used_by = np.ones(reorder_allocation_mat.shape[1], dtype=bool)
        used_by[reorder_allocation_mat[jj, :] != 1] = False
        selected_colors = [
            colors[jj] if uu else 'white' for uu in used_by]
        ax.bar(index, npartition_samples[jj], 0.8, bottom=y_offset,
               color=selected_colors)
        y_offset += npartition_samples[jj]

    # xticklabels = []
    xticklabels = [r"$\mathcal{Z}_%d$" % 0]
    for ii in range(1, nmodels):
        xticklabels += [
            r"$\mathcal{Z}_%d^\star$" % ii, r"$\mathcal{Z}_%d$" % ii]
    ax.set_xticks(index)
    ax.set_xticklabels(xticklabels)


def plot_model_costs(costs, model_names=None, ax=None):
    from pyapprox.util.configure_plots import plt
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nmodels = len(costs)
    if model_names is None:
        model_names = [r"$f_{%d}$" % ii for ii in range(nmodels)]
    ax.bar(np.arange(nmodels), costs)
    ax.set_xticks(np.arange(nmodels))
    ax.set_xticklabels(model_names)



# Notes
# using pkg.double when ever creating a torch tensor is esssential.
# Otherwise autograd will not work.
