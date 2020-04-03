"""
Functions for estimating expectations using frequentist control-variate Monte-Carlo based methods such as multi-level Monte-Carlo, control-variate Monte-Carlo, and approximate control-variate Monte-Carlo.
"""
import numpy as np, os
from scipy.optimize import minimize
try:
    #use torch to compute gradients for sample allocation optimization
    import torch
except:
    msg = 'Could not import Torch'
    print(msg)
import copy
from pyapprox.utilities import get_all_sample_combinations
from functools import partial

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

def standardize_sample_ratios(nhf_samples,nsample_ratios):
    """
    Ensure num high fidelity samples is positive (>0) and then recompute 
    sample ratios. This is useful when num high fidelity samples and 
    sample ratios are computed by an optimization process. This function 
    is useful for optimization problems with a numerical or analytical 
    solution.

    Parameters
    ----------
    nhf_samples : integer
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, 
        i=1,...,nmodels-1

    Returns
    -------
    nhf_samples : integer
        The corrected number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The corrected sample ratios
    """
    nsamples = np.array([r*nhf_samples for r in nsample_ratios])
    nhf_samples = int(max(1,np.floor(nhf_samples)))
    nsample_ratios = np.floor(nsamples)/nhf_samples
    #nhf_samples = int(max(1,np.round(nhf_samples)))
    #nsample_ratios = [max(np.round(nn/nhf_samples),0) for nn in nsamples]
    return nhf_samples, np.asarray(nsample_ratios)

def get_variance_reduction(get_rsquared,cov,nsample_ratios):
    """
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
    return 1-get_rsquared(cov,nsample_ratios)

def get_control_variate_rsquared(cov):
    """
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
    nmodels = cov.shape[0]
    rsquared = cov[0,1:].dot(np.linalg.solve(cov[1:, 1:], cov[1:, 0]))
    rsquared /= cov[0,0]
    return rsquared

def get_rsquared_mfmc(cov,nsample_ratios):
    """
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
    assert len(nsample_ratios)==nmodels-1
    rsquared=(nsample_ratios[0]-1)/(nsample_ratios[0])*cov[0,1]/(
        cov[0,0]*cov[1,1])*cov[0,1]
    for ii in range(1,nmodels-1):
        p1 = (nsample_ratios[ii]-nsample_ratios[ii-1])/(
            nsample_ratios[ii]*nsample_ratios[ii-1])
        p1 *= cov[0,ii+1]/(cov[0,0]*cov[ii+1,ii+1])*cov[0,ii+1]
        rsquared += p1
    return rsquared

def get_rsquared_mlmc(cov,nsample_ratios,pkg=np):
    """
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
    assert len(nsample_ratios)==nmodels-1
    gamma = 0.0
    rhat = pkg.ones(nmodels)
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
    """
    Get the weights used by the MLMC control variate estimator

    Returns
    -------
    weights : np.ndarray (nmodels-1)
        The control variate weights
    """
    return -np.ones(nmodels-1)

def compute_approximate_control_variate_mean_estimate(weights,values):
    """
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
    assert len(values)==nmodels
    # high fidelity monte carlo estimate of mean
    est = values[0][0].mean()
    for ii in range(nmodels-1):
        est += weights[ii]*(values[ii+1][0].mean()-values[ii+1][1].mean())
    return est


def compute_control_variate_mean_estimate(weights,values,lf_means):
    """
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
    assert len(values)==nmodels
    # high fidelity monte carlo estimate of mean
    est = values[0].mean()
    for ii in range(nmodels-1):
        est += weights[ii]*(values[ii+1].mean()-lf_means[ii])
    return est

def allocate_samples_mfmc(cov, costs, target_cost, standardize=True):
    """
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

    standardize : boolean
        If true make sure that nhf_samples is an integer and that 
        nhf_samples*nsamples_ratios are integers. False is only ever used 
        for testing.

    Returns
    -------
    nhf_samples : integer 
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i=r_i*nhf_samples, i=1,...,nmodels-1

    log10_variance : float
        The base 10 logarithm of the variance of the estimator
    """

    nmodels = cov.shape[0]
    corr = compute_correlations_from_covariance(cov)
    I = np.argsort(np.absolute(corr[0,1:]))[::-1]
    if not np.allclose(I, np.arange(nmodels-1)):
        msg = 'Models must be ordered with decreasing correlation with '
        msg += 'high-fidelity model'
        raise Exception(msg)

    r = []
    for ii in range(nmodels-1):
        # Step 3 in Algorithm 2 in Peherstorfer et al 2016
        num = costs[0] * (corr[0, ii]**2 - corr[0, ii+1]**2)
        den = costs[ii] * ( 1 - corr[0, 1]**2)
        r.append(np.sqrt(num/den))


    num = costs[0]*corr[0,-1]**2
    den = costs[-1] * (1 - corr[0, 1]**2)
    r.append(np.sqrt(num/den))

    # Step 4 in Algorithm 2 in Peherstorfer et al 2016
    nhf_samples = target_cost / np.dot(costs, r)
    nhf_samples = max(nhf_samples, 1)
    nsample_ratios = r[1:]

    if standardize:
        nhf_samples, nsample_ratios = standardize_sample_ratios(
            nhf_samples, nsample_ratios)

    gamma = get_variance_reduction(get_rsquared_mfmc,cov,nsample_ratios)
    log10_variance = np.log10(gamma)+np.log10(cov[0, 0])-np.log10(
        nhf_samples)

    return nhf_samples, np.atleast_1d(nsample_ratios), log10_variance

def allocate_samples_mlmc(cov, costs, target_cost, standardize=True):
    """
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

    standardize : boolean
        If true make sure that nhf_samples is an integer and that 
        nhf_samples*nsamples_ratios are integers. False is only ever used 
        for testing.


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
    sum1 = 0.0
    nsamples = []
    vardeltas=[]
    for ii in range(nmodels-1):
        # compute the variance of the discrepancy 
        vardelta = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
        vardeltas.append(vardelta)
        # compute the variance * cost
        vc = vardelta * (costs[ii] + costs[ii+1])
        # compute the unnormalized number of samples\
        # these values will be normalized by lamda later
        nsamp = np.sqrt(vardelta / (costs[ii] + costs[ii+1]))
        nsamples.append(nsamp)
        sum1 += np.sqrt(vc)
    I = np.argsort(vardeltas)
    #assert np.allclose(I,np.arange(nmodels-1))

    # compute information for lowest fidelity model
    v = cov[nmodels-1, nmodels-1]
    c = costs[nmodels-1]
    nsamples.append(np.sqrt(v/c))
    sum1 += np.sqrt(v*c)

    # compute the ML estimator variance from the target cost
    variance = sum1**2 / target_cost
    # compute the lagrangian parameter
    sqrt_lamda = sum1/variance
    # compute the number of samples allocated to resolving each
    # discrepancy.
    nl = [sqrt_lamda * n for n in nsamples]

    # compute the number of samples allocated to each model. For
    # all but the highest fidelity model we need to collect samples
    # from two discrepancies.
    nhf_samples = nl[0]
    nsample_ratios = []
    for ii in range(1, nmodels-1):
        nsample_ratios.append((nl[ii-1] + nl[ii])/nl[0])
    if nmodels>1:
        nsample_ratios.append((nl[-2]+nl[-1])/nl[0])

    nsample_ratios = np.asarray(nsample_ratios)

    if standardize:
        nhf_samples = max(nhf_samples, 1)
        nhf_samples, nsample_ratios = standardize_sample_ratios(
            nhf_samples, nsample_ratios)
    gamma = get_variance_reduction(get_rsquared_mlmc,cov,nsample_ratios)
    log10_variance=np.log10(gamma)+np.log10(cov[0, 0])-np.log10(
        nhf_samples)
    #print(log10_variance)
    if np.isnan(log10_variance):
        raise Exception('MLMC variance is NAN')
    return nhf_samples, np.atleast_1d(nsample_ratios), log10_variance

def get_lagrange_multiplier_mlmc(cov,costs,nhf_samples):
    """
    Given an optimal sample allocation recover the optimal value of the 
    Lagrange multiplier. This is only used for testing
    """
    ii=0 # 0th discrepancy
    var_delta = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
    cost_delta = (costs[ii] + costs[ii+1])
    lagrange_mult=nhf_samples**2/(var_delta/cost_delta)
    return lagrange_mult

def get_discrepancy_covariances_IS(cov,nsample_ratios,pkg=np):
    """
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

    pkg : package (optional)
        A python package (numpy or torch) used to store the covariances.

    Results
    -------
    CF : np.ndarray (nmodels-1,nmodels-1)
        The matrix of covariances between the discrepancies :math:`\delta`

    cf : np.ndarray (nmodels-1)
        The vector of covariances between the discrepancies and the 
        high-fidelity model.
    """
    nmodels = cov.shape[0]
    F = pkg.zeros((nmodels-1, nmodels-1), dtype=pkg.double)
    for ii in range(nmodels-1):
        F[ii, ii]=(nsample_ratios[ii]-1)/nsample_ratios[ii]
        for jj in range(ii+1,nmodels-1):
            F[ii, jj] = (nsample_ratios[ii]-1)/nsample_ratios[ii] * (
                nsample_ratios[jj]-1)/nsample_ratios[jj]
            F[jj, ii] = F[ii, jj]

    CF = cov[1:,1:] * F
    cf = pkg.diag(F) * cov[1:, 0]
    return CF,cf


def get_discrepancy_covariances_MF(cov,nsample_ratios,pkg=np):
    """
    Get the covariances of the discrepancies :math:`\delta` 
    between each low-fidelity model and its estimated mean using the MFMC
    sampling strategy.

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The estimated covariance between each model.

    nsample_ratios : iterable (nmodels-1)
        The sample ratioss :math:`r_\alpha>1` for each low-fidelity model

    pkg : package (optional)
        A python package (numpy or torch) used to store the covariances.

    Results
    -------
    CF : np.ndarray (nmodels-1,nmodels-1)
        The matrix of covariances between the discrepancies :math:`\delta`

    cf : np.ndarray (nmodels-1)
        The vector of covariances between the discrepancies and the 
        high-fidelity model.
    """
    nmodels = cov.shape[0]
    F = pkg.zeros((nmodels-1, nmodels-1), dtype=pkg.double)
    for ii in range(nmodels-1):
        for jj in range(nmodels-1):
            rr = min(nsample_ratios[ii],nsample_ratios[jj])
            F[ii, jj] = (rr - 1) / rr

    CF = cov[1:,1:] * F
    cf = pkg.diag(F) * cov[1:, 0]
    return CF,cf

def get_discrepancy_covariances_KL(cov,nsample_ratios,K,L,pkg=np):
    """
    Get the covariances of the discrepancies :math:`\delta` 
    between each low-fidelity model and its estimated mean using the MFMC
    sampling strategy and the ACV KL estimator.

    The ACV-KL estimator partitions all of the control variates into two 
    groups; the first K variables form a K -level approximate control 
    variate, and the last M − K variables are used to reduce the variance of 
    estimating :math:`\mu_L` some :math:`L \le K` . The resulting estimator 
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

    pkg : package (optional)
        A python package (numpy or torch) used to store the covariances.

    Results
    -------
    CF : np.ndarray (nmodels-1,nmodels-1)
        The matrix of covariances between the discrepancies :math:`\delta`

    cf : np.ndarray (nmodels-1)
        The vector of covariances between the discrepancies and the 
        high-fidelity model.
    """
    nmodels = cov.shape[0]
    assert L<=K+1 and L>=1 and K<nmodels
    K,L=K-1,L-1
    F = pkg.zeros((nmodels-1, nmodels-1), dtype=pkg.double)
    rs = nsample_ratios
    for ii in range(nmodels-1):
        if ii <= K:
            F[ii, ii] = (rs[ii]-1)/(rs[ii]+1e-20)
        else:
            F[ii, ii] = (rs[ii]-rs[L])/(rs[ii]*rs[L])
        for jj in range(ii+1,nmodels-1):
            if (ii <= K) and (jj <= K):
                ri = min(rs[ii], rs[jj])
                F[ii, jj] = (ri - 1) / (ri + 1e-20)
            elif (jj > K) and (ii > K):
                ri = min(rs[ii], rs[jj])
                t1 = (rs[ii]-rs[L])*(rs[jj]-rs[L])/(rs[ii]*rs[jj]*rs[L]
                                                        +1e-20)
                t2 = (ri - rs[L]) / (rs[ii] * rs[jj] + 1e-20)
                F[ii, jj] = t1 + t2
            elif (ii > L) and (ii <= K) and (jj > K):
                F[ii, jj] = (rs[ii] - rs[L]) / (rs[ii] * rs[L] + 1e-20)
            elif (jj > L) and (jj <= K) and (ii > K):
                F[ii, jj] = (rs[jj] - rs[L]) / (rs[jj] * rs[L] + 1e-20)
            else:
                F[ii, jj] = 0.0
            F[jj, ii] = F[ii, jj]

    CF = cov[1:,1:] * F
    cf = pkg.diag(F) * cov[1:, 0]
    return CF,cf

def get_control_variate_weights(cov):
    """
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
    weights = -np.linalg.solve(cov[1:,1:], cov[0,1:])
    return weights


def get_approximate_control_variate_weights(cov,nsample_ratios,
                                            get_discrepancy_covariances):
    """
    Get the weights used by the approximate control variate estimator.

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The estimated covariance between each model.

    nsample_ratios : iterable (nmodels-1)
        The sample ratioss :math:`r_\alpha>1` for each low-fidelity model

    get_discrepancy_covariances : callable
        Function with signature get_discrepancy_covariances(cov,nsample_ratios)
        which returns the covariances between the discrepancies betweem the 
        low-fidelity models and their approximated mean.

    Returns
    -------
    weights : np.ndarray (nmodels-1)
        The control variate weights
    """
    CF,cf = get_discrepancy_covariances(cov,nsample_ratios)
    weights = -np.linalg.solve(CF, cf)
    return weights

def get_rsquared_acv(cov,nsample_ratios,get_discrepancy_covariances):
    """
    Compute r^2 used to compute the variance reduction  of 
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
    CF,cf = get_discrepancy_covariances(cov,nsample_ratios)
    if type(cov)==np.ndarray:
        rsquared = np.dot(cf,np.linalg.solve(CF,cf))/cov[0, 0]
    else:
        try:
            rsquared = torch.dot(cf, torch.mv(torch.inverse(CF),cf))/cov[0, 0]
        except:
            #print("Error computing inverse of CF")
            return torch.tensor([0.0], dtype=torch.double)*nsample_ratios[0]
    return rsquared


def acv_sample_allocation_sample_ratio_constraint(ratios, *args):
    ind = args[0]
    return ratios[ind] - ratios[ind-1]

def generate_samples_and_values_acv_IS(nhf_samples,nsample_ratios,
                                       functions,generate_samples):
    nmodels = len(nsample_ratios)+1
    if not callable(functions):
        assert len(functions)==nmodels
    samples1 = [generate_samples(nhf_samples)]*nmodels
    samples2 = [None]+[np.hstack(
        [samples1[ii+1],generate_samples(int(nhf_samples*r-nhf_samples))])
                       for ii,r in enumerate(nsample_ratios)]
    if not callable(functions):
        values2  = [None]+[f(s) for f,s in zip(functions[1:],samples2[1:])]
        values1  = [functions[0](samples1[0])]
        values1 += [values2[ii][:nhf_samples] for ii in range(1,nmodels)]
    else:
        nsamples2 = [0]
        samples_with_id = np.vstack([samples1[0],np.zeros((1,nhf_samples))])
        for ii in range(1,nmodels):
            samples2_ii = np.vstack(
                [samples2[ii],ii*np.ones((1,samples2[ii].shape[1]))])
            nsamples2.append(samples2[ii].shape[1])
            samples_with_id = np.hstack([
                samples_with_id,samples2_ii])
        values_flattened = functions(samples_with_id)
        values1 = [values_flattened[:nhf_samples]]
        values2 = [None]
        cnt = nhf_samples
        for ii in range(1,nmodels):
            values1.append(values_flattened[cnt:cnt+nhf_samples])
            values2.append(values_flattened[cnt:cnt+nsamples2[ii]])
            cnt += nsamples2[ii]
        
    samples = [[s1,s2] for s1,s2 in zip(samples1,samples2)]
    values  = [[v1,v2] for v1,v2 in zip(values1,values2)]
    return samples,values

def generate_samples_and_values_mlmc(nhf_samples,nsample_ratios,functions,
                                     generate_samples):
    """
    Parameters
    ==========
    nhf_samples : integer
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    functions : list of callables
        The functions used to evaluate each model

    generate_samples : callable
        Function used to generate realizations of the random variables

    Returns
    =======
    
    """
    nmodels = len(nsample_ratios)+1
    if not callable:
        assert nmodels==len(functions)
    assert np.all(nsample_ratios>=1)
    samples1 = [generate_samples(nhf_samples)]
    samples2 = [None]
    prev_samples = samples1[0]
    for ii in range(nmodels-1):
        total_samples = nsample_ratios[ii] * nhf_samples
        assert total_samples/int(total_samples)==1.0
        total_samples = int(total_samples)
        samples1.append(prev_samples)
        nnew_samples = total_samples - prev_samples.shape[1]
        samples2.append(generate_samples(nnew_samples))
        prev_samples = samples2[-1]

    if not callable(functions):
        values1 = [functions[0](samples1[0])]
        values2 = [None]
        for ii in range(1,nmodels):
            values1.append(functions[ii](samples1[ii]))
            values2.append(functions[ii](samples2[ii]))
    else:
        samples_with_id = np.vstack([samples1[0],np.zeros((1,nhf_samples))])
        nsamples1 = [nhf_samples]
        nsamples2 = [0]
        for ii in range(1,nmodels):
            samples1_ii = np.vstack(
                [samples1[ii],ii*np.ones((1,samples1[ii].shape[1]))])
            samples2_ii = np.vstack(
                [samples2[ii],ii*np.ones((1,samples2[ii].shape[1]))])
            nsamples1.append(samples1[ii].shape[1])
            nsamples2.append(samples2[ii].shape[1])
            samples_with_id = np.hstack([
                samples_with_id,samples1_ii,samples2_ii])
        values_flattened = functions(samples_with_id)
        values1 = [values_flattened[:nsamples1[0]]]
        values2 = [None]
        cnt = nsamples1[0]
        for ii in range(1,nmodels):
            values1.append(values_flattened[cnt:cnt+nsamples1[ii]])
            cnt += nsamples1[ii]
            values2.append(values_flattened[cnt:cnt+nsamples2[ii]])
            cnt += nsamples2[ii]

    samples = [[s1,s2] for s1,s2 in zip(samples1,samples2)]
    values  = [[v1,v2] for v1,v2 in zip(values1,values2)]
    
    return samples, values

def get_mfmc_control_variate_weights(cov):
    weights = -cov[0,1:]/np.diag(cov[1:,1:])
    return weights

def validate_nsample_ratios(nhf_samples,nsample_ratios):
    """
    Check that nsample_ratios* nhf_samples are all integers
    and that nsample_ratios are all larger than 1
    """
    nmodels = len(nsample_ratios)+1
    assert np.all(nsample_ratios>=1)
    # check nhf_samples is an integer
    assert nhf_samples/int(nhf_samples)==1.0
    # convert to int if a float because numpy random assumes nsamples
    # is an int
    nhf_samples = int(nhf_samples)
    nlf_samples = nhf_samples*nsample_ratios
    for ii in range(nmodels-1):
        assert nlf_samples[ii]/int(nlf_samples[ii])==1.0
    nlf_samples = np.asarray(nlf_samples,dtype=int)
    return nlf_samples

def generate_samples_and_values_acv_KL(nhf_samples,nsample_ratios,functions,
                                       generate_samples,K,L):
    """

    K : integer (K<=nmodels-1)
        The number of effective control variates.

    L : integer (1<=L<=K+1)
        The id of the models whose mean is being targeted by the 
        remaining nmodels-K low fidelity models. 
    """
    nsample_ratios = np.asarray(nsample_ratios)
    nlf_samples = validate_nsample_ratios(nhf_samples,nsample_ratios)
    nmodels = nsample_ratios.shape[0]+1
    assert L<=K+1 and L>=1 and K<nmodels
    K,L=K-1,L-1
    
    max_nsamples = nlf_samples.max()
    samples = generate_samples(max_nsamples)
    samples1 = [samples[:,:nhf_samples]]
    samples2 = [None]
    nprev_samples1 = nhf_samples
    nprev_samples_total = nhf_samples
    for ii in range(1,nmodels):
        samples1.append(samples[:,:nprev_samples1])
        samples2.append(samples[:,:nlf_samples[ii-1]])
        if (ii<=K):
            nprev_samples1 = nhf_samples
        else:
            nprev_samples1 = nlf_samples[L]
        nprev_samples_total= nlf_samples[ii-1]
        
    if not callable(functions):
        values1 = [functions[0](samples1[0])]
        values2 = [None]
        for ii in range(1,nmodels):
            values_ii = functions[ii](samples2[ii])
            values1.append(values_ii[:samples1[ii].shape[1]])
            values2.append(values_ii)
    else:
        # collect all samples assign an id and then evaluate in one batch
        # this can be faster if functions is something like a pool model
        samples_with_id = np.vstack([samples1[0],np.zeros((1,nhf_samples))])
        for ii in range(1,nmodels):
            samples_with_id = np.hstack([
                samples_with_id,
                np.vstack(
                    [samples2[ii],ii*np.ones((1,samples2[ii].shape[1]))])])
        assert samples_with_id.shape[1]==nhf_samples+np.sum(nlf_samples)
            
        values_flattened = functions(samples_with_id)
        values1 = [values_flattened[:nhf_samples]]
        values2 = [None]
        nprev_samples1 = nhf_samples
        nprev_samples_total = nhf_samples
        cnt = nhf_samples
        for ii in range(1,nmodels):
            values1.append(values_flattened[cnt:cnt+nprev_samples1])
            values2.append(values_flattened[cnt:cnt+nlf_samples[ii-1]])
            cnt += nlf_samples[ii-1]
            if (ii <= K):
                nprev_samples1 = nhf_samples
            else:
                nprev_samples1 = nlf_samples[L]
            nprev_samples_total = nlf_samples[ii-1]
        assert cnt==values_flattened.shape[0]
            
    samples = [[s1,s2] for s1,s2 in zip(samples1,samples2)]
    values  = [[v1,v2] for v1,v2 in zip(values1,values2)]

    return samples,values

def generate_samples_and_values_mfmc(nhf_samples,nsample_ratios,functions,
                                     generate_samples,acv_modification=False):
    """
    Parameters
    ==========
    nhf_samples : integer
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    functions : list of callables
        The functions used to evaluate each model

    generate_samples : callable
        Function used to generate realizations of the random variables

    Returns
    =======
    samples : list 
        List containing the samples :math:`\matchcal{Z}_{i,1}` and 
        :math:`\matchcal{Z}_{i,2}` for each model :math:`i=0,\ldots,M-1`.
        The list is [[:math:`\matchcal{Z}_{0,1}`,:math:`\matchcal{Z}_{0,2}`],...,[:math:`\matchcal{Z}_{M-1,1}`,:math:`\matchcal{Z}_{M-1,2}`]], 
        where :math:`M` is the number of models

    values : list 
        Model values at the points in samples
    
    """
    nsample_ratios = np.asarray(nsample_ratios)
    nlf_samples = validate_nsample_ratios(nhf_samples,nsample_ratios)
    nmodels = nsample_ratios.shape[0]+1

    max_nsamples = nlf_samples.max()
    samples = generate_samples(max_nsamples)
    samples1 = [samples[:,:nhf_samples]]
    samples2 = [None]
    nprev_samples = nhf_samples
    for ii in range(1,nmodels):
        samples1.append(samples[:,:nprev_samples])
        samples2.append(samples[:,:nlf_samples[ii-1]])
        if acv_modification:
            nprev_samples = nhf_samples
        else:
            nprev_samples = samples2[ii].shape[1]

    if not callable(functions):
        values1 = [functions[0](samples1[0])]
        values2 = [None]
        for ii in range(1,nmodels):
            values_ii = functions[ii](samples2[ii])
            values1.append(values_ii[:samples1[ii].shape[1]])
            values2.append(values_ii)
    else:
        # collect all samples assign an id and then evaluate in one batch
        # this can be faster if functions is something like a pool model
        samples_with_id = np.vstack([samples1[0],np.zeros((1,nhf_samples))])
        for ii in range(1,nmodels):
            samples_with_id = np.hstack([
                samples_with_id,
                np.vstack(
                    [samples2[ii],ii*np.ones((1,samples2[ii].shape[1]))])])
        values_flattened = functions(samples_with_id)
        values1 = [values_flattened[:nhf_samples]]
        values2 = [None]
        nprev_samples = nhf_samples
        cnt = nhf_samples
        for ii in range(1,nmodels):
            values1.append(values_flattened[cnt:cnt+nprev_samples])
            values2.append(values_flattened[cnt:cnt+nlf_samples[ii-1]])
            cnt += nlf_samples[ii-1]
            if acv_modification:
                nprev_samples = nhf_samples
            else:
                nprev_samples = samples2[ii].shape[1]
    assert cnt==values_flattened.shape[0]
    assert cnt==nhf_samples + np.sum(nlf_samples)
            
    samples = [[s1,s2] for s1,s2 in zip(samples1,samples2)]
    values  = [[v1,v2] for v1,v2 in zip(values1,values2)]

    return samples,values

def acv_sample_allocation_cost_constraint(ratios, nhf, costs, target_cost):
    cost = nhf*(costs[0] + np.dot(ratios, costs[1:]))
    return target_cost - cost

def acv_sample_allocation_cost_constraint_all(ratios, costs, target_cost):
    nhf, rats = ratios[0], ratios[1:]
    cost = nhf*(costs[0] + np.dot(rats, costs[1:]))
    return target_cost - cost

def acv_sample_allocation_cost_constraint_jacobian_all(ratios, costs,
                                                       target_cost):
    nhf, rats = ratios[0], ratios[1:]
    jac = costs.copy().astype(float)
    jac[0] += np.dot(rats, costs[1:])
    jac[1:] *= nhf
    return -jac

def acv_sample_allocation_objective(estimator, nsample_ratios):
    ratios = torch.tensor(nsample_ratios)
    gamma = estimator.variance_reduction(ratios)
    gamma = torch.log10(gamma)
    return gamma.item()

def acv_sample_allocation_jacobian(estimator, nsample_ratios):
    ratios = torch.tensor(nsample_ratios, dtype=torch.double)
    ratios.requires_grad=True
    gamma = estimator.variance_reduction(ratios)
    gamma = torch.log10(gamma)
    gamma.backward()
    grad = ratios.grad.numpy().copy()
    ratios.grad.zero_()
    return grad

def acv_sample_allocation_objective_all(estimator, x):
    xrats = torch.tensor(x, dtype=torch.double)
    xrats.requires_grad=True
    nhf, ratios = xrats[0], xrats[1:]
    #TODO make this consistent with other objective which does not use
    #variance as is used below. It is necessary here because need to include
    #the impact of nhf on objective
    gamma = estimator.variance_reduction(ratios) * estimator.cov[0, 0] / nhf
    gamma = torch.log10(gamma)
    return gamma.item()

def acv_sample_allocation_jacobian_all(estimator,x):
    xrats = torch.tensor(x, dtype=torch.double)
    xrats.requires_grad=True
    nhf, ratios = xrats[0], xrats[1:]
    gamma = estimator.variance_reduction(ratios)*estimator.cov[0,0]/nhf
    gamma = torch.log10(gamma)
    gamma.backward()
    grad = xrats.grad.numpy().copy()
    xrats.grad.zero_()
    return grad

def acv_sample_allocation_objective_all_lagrange(estimator, x):
    xrats = torch.tensor(x, dtype=torch.double)
    xrats.requires_grad=True
    nhf, ratios, lagrange_mult = xrats[0], xrats[1:-1], xrats[-1]
    gamma = estimator.variance_reduction(ratios)*estimator.cov[0, 0]/nhf
    total_cost = estimator.costs[0]*nhf + estimator.costs[1:].dot(
        ratios*nhf)
    obj = lagrange_mult*gamma+total_cost
    obj = torch.log10(obj)
    return obj.item()

def acv_sample_allocation_jacobian_all_lagrange(estimator,x):
    xrats = torch.tensor(x, dtype=torch.double)
    xrats.requires_grad=True
    nhf, ratios, lagrange_mult = xrats[0], xrats[1:-1], xrats[-1]
    gamma = estimator.variance_reduction(ratios)*estimator.cov[0,0]/nhf
    total_cost = estimator.costs[0]*nhf+estimator.costs[1:].dot(
        ratios*nhf)
    obj = lagrange_mult*gamma+total_cost
    obj = torch.log10(obj)
    obj.backward()
    grad = xrats.grad.numpy().copy()
    xrats.grad.zero_()
    return grad

def get_allocate_samples_acv_trust_region_constraints(costs,target_cost):
    from scipy.optimize import NonlinearConstraint
    nonlinear_constraint = NonlinearConstraint(
        partial(acv_sample_allocation_cost_constraint_all,
                costs=costs, target_cost=target_cost),0,0)
    return [nonlinear_constraint]

def solve_allocate_samples_acv_trust_region_optimization(
        estimator,costs,target_cost,initial_guess,optim_options):
    nmodels = costs.shape[0]
    constraints = get_allocate_samples_acv_trust_region_constraints(
        costs,target_cost)
    if optim_options is None:
        tol=1e-10
        optim_options={'verbose': 1, 'maxiter':1000,
                       'gtol':tol, 'xtol':1e-4*tol, 'barrier_tol':tol}
    from scipy.optimize import Bounds
    lbs,ubs = [1]*nmodels,[np.inf]*nmodels
    bounds = Bounds(lbs,ubs)
    opt = minimize(estimator.objective,initial_guess,method='trust-constr',
        jac=estimator.jacobian,#hess=self.objective_hessian,
        constraints=constraints,options=optim_options,
        bounds=bounds)
    if opt.success == False:
        raise Exception('Trust-constr optimizer failed')
    return opt

def get_initial_guess(initial_guess, cov, costs, target_cost):
    if initial_guess is None:
        nhf_samples_init, nsample_ratios_init =  allocate_samples_mfmc(
            cov, costs, target_cost, standardize=True)[:2]
        initial_guess = np.concatenate(
            [[nhf_samples_init],nsample_ratios_init])
    return initial_guess

def solve_allocate_samples_acv_slsqp_optimization(
         estimator,costs,target_cost,initial_guess,optim_options):
    nmodels = len(costs)
    #alex had these bounds and constraints
    # bounds = [(1,np.inf)] + [(2, np.inf)]*(nmodels-1)
    # cons = [dict({'type':'ineq',
    #             'fun':acv_sample_allocation_cost_constraint_all,
    #             'args':(costs, target_cost)})]
    # for jj in range(2,nmodels-1):
    #     cons.append( dict({'type':'ineq',
    #                        'fun':acv_sample_allocation_ratio_constraint_all,
    #                        'args':[jj]}))
    if optim_options is None:
        optim_options = {'disp':True,'ftol':1e-12,
                         'maxiter':1000,'iprint':0}
        #set iprint=2 to printing iteration info
    
    bounds = [(1,np.inf)] + [(1.1, np.inf)]*(nmodels-1)
    cons = [{'type':'eq',
               'fun':acv_sample_allocation_cost_constraint_all,
               'jac':acv_sample_allocation_cost_constraint_jacobian_all,
               'args':(np.asarray(costs), target_cost)}]

    opt = minimize(
        estimator.objective, initial_guess,
        method='SLSQP',jac=estimator.jacobian, bounds=bounds,
        constraints=cons,
        options = optim_options)
    if opt.success == False:
        raise Exception('SLSQP optimizer failed')
    return opt
        

def allocate_samples_acv(cov, costs, target_cost, estimator,
                         standardize=True, initial_guess=None,
                         optim_options=None, optim_method='SLSQP'):
    """
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
    nhf_samples : integer 
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i=r_i*nhf_samples,i=1,...,nmodels-1

    log10_variance : float
        The base 10 logarithm of the variance of the estimator
    """
    initial_guess=get_initial_guess(
        initial_guess, cov, costs, target_cost)
    if optim_method=='trust-constr':
        opt = solve_allocate_samples_acv_trust_region_optimization(
            estimator,costs,target_cost,initial_guess,optim_options)
    else:
        opt = solve_allocate_samples_acv_slsqp_optimization(
            estimator,costs,target_cost,initial_guess,optim_options)
    nhf_samples, nsample_ratios = opt.x[0], opt.x[1:]

    if standardize:
        nhf_samples, nsample_ratios = standardize_sample_ratios(
            nhf_samples, nsample_ratios)
    var = estimator.get_variance(nhf_samples,nsample_ratios)
    log10_var = np.log10(var.item())
    return nhf_samples, nsample_ratios, log10_var

def get_rsquared_acv_KL_best(cov, nsample_ratios):
    """ 
    """
    nmodels = cov.shape[1]
    opt_rsquared = -1
    KL = None
    for K in range(1, nmodels):
        for L in range(1, K+1):
            get_discrepancy_covariances =  partial(
                get_discrepancy_covariances_KL,K=K,L=L)
            get_rsquared = partial(
                get_rsquared_acv,
                get_discrepancy_covariances=get_discrepancy_covariances)
            rsquared = get_rsquared(cov, nsample_ratios)
            #print(K,L,rsquared)
            if rsquared > opt_rsquared:
                opt_rsquared = rsquared
                KL = (K, L)
    return opt_rsquared

def allocate_samples_acv_best_kl(cov,costs,target_cost,standardize=True,
                                 initial_guess=None,optim_options=None,
                                 optim_method='SLSQP'):
    nmodels = len(costs)
    sol, KL, opt_log10_var = None, None, np.inf

    for K in range(1,nmodels):
        for L in range(1, K+1):
            estimator = ACVMFKL(cov, costs,target_cost, K, L)
            nhf_samples, nsample_ratios, log10_var = allocate_samples_acv(
                cov, costs, target_cost, estimator,standardize,
                initial_guess,optim_options,optim_method)
            #print("K, L = ", K, L)
            #print("\t ", log10_var)
            if log10_var < opt_log10_var:
                opt_log10_var = log10_var
                sol = (nhf_samples, nsample_ratios)
                KL  = (K, L)

    return sol[0], sol[1], opt_log10_var

class ModelEnsemble(object):
    """
    Wrapper class to allow easy one-dimensional 
    indexing of models in an ensemble.
    """
    def __init__(self,functions):
        """
        Parameters
        ----------
        functions : list of callable
            A list of functions defining the model ensemble. The functions must
            have the call signature values=function(samples)
        """
        self.functions=functions
        self.nmodels = len(self.functions)
    
    def __call__(self,samples):
        """
        Evaluate a set of models at a set of samples

        Parameters
        ----------
        samples : np.ndarray (nvars+1,nsamples)
            Realizations of a multivariate random variable each with an 
            additional scalar model id indicating which model to evaluate.

        Returns
        -------
        values : np.ndarray (nsamples,nqoi)
            The values of the models at samples
        """
        model_ids = samples[-1,:]
        assert model_ids.max()<self.nmodels
        I = np.where(model_ids==0)[0]
        values_0 = self.functions[0](samples[:-1,I])
        assert values_0.ndim==2
        nqoi = values_0.shape[1]
        values = np.empty((samples.shape[1],nqoi))
        values[I,:]=values_0
        for ii,f in enumerate(self.functions[1:]):
            I = np.where(model_ids==ii+1)[0]
            values[I] = f(samples[:-1,I])
        return values

def estimate_model_ensemble_covariance(npilot_samples,generate_samples,
                                       model_ensemble):
    """
    Estimate the covariance of a model ensemble from a set of pilot samples

    Parameters
    ----------
    npilot_samples : integer
        The number of samples used to estimate the covariance

    generate_samples : callable
        Function used to generate realizations of the random variables with 
        call signature samples = generate_samples(npilot_samples)

    model_emsemble : callable
        Function that takes a set of samples and models ids and evaluates
        a set of models. See ModelEnsemble.
        call signature values = model_emsemble(samples)

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
    pilot_random_samples = generate_samples(npilot_samples)
    config_vars = np.arange(model_ensemble.nmodels)[np.newaxis,:]
    # append model ids to pilot smaples
    pilot_samples = get_all_sample_combinations(
        pilot_random_samples,config_vars)
    # evaluate models at pilot samples
    pilot_values = model_ensemble(pilot_samples)
    pilot_values = np.reshape(
        pilot_values,(npilot_samples,model_ensemble.nmodels))
    # compute covariance
    cov = np.cov(pilot_values,rowvar=False)
    return cov, pilot_random_samples, pilot_values

class ACVMF(object):
    def __init__(self,cov,costs):
        self.cov=torch.tensor(np.copy(cov), dtype=torch.double)
        self.costs=torch.tensor(np.copy(costs), dtype=torch.double)
        
        #self.objective_fun = partial(
        #    acv_sample_allocation_objective,self)
        #self.jacobian_fun = partial(
        #    acv_sample_allocation_jacobian,self)
        self.objective_fun_all = partial(
            acv_sample_allocation_objective_all,self)
        self.jacobian_fun_all = partial(
            acv_sample_allocation_jacobian_all,self)
    
    def get_rsquared(self,nsample_ratios):
        return get_rsquared_acv(
            self.cov,nsample_ratios,
            partial(get_discrepancy_covariances_MF,pkg=torch))

    def variance_reduction(self,nsample_ratios):
        return 1-self.get_rsquared(nsample_ratios)

    def objective(self,x):
        return self.objective_fun_all(x)

    def jacobian(self,x):
        return self.jacobian_fun_all(x)

    def allocate_samples(self,target_cost,**kwargs):
        return allocate_samples_acv(self.cov, self.costs, target_cost, self,
                                    **kwargs)

    def get_nsamples(self,nhf_samples,nsample_ratios):
        return np.concatenate([[nhf_samples],nsample_ratios*nhf_samples])

    def get_variance(self,nhf_samples,nsample_ratios):
        gamma = (1-self.get_rsquared(nsample_ratios))
        return gamma*self.cov[0,0]/nhf_samples

class MC(object):
    def __init__(self,cov,costs):
        self.costs=costs
        self.cov=cov
    
    def get_variance(self,nhf_samples,nsample_ratios):
        return self.cov[0,0]/nhf_samples

    def allocate_samples(self,target_cost):
        return np.floor(target_cost/self.costs[0]),None,None

    def get_nsamples(self,nhf_samples,nsample_ratios):
        return np.concatenate([[nhf_samples],np.zeros(self.cov.shape[0]-1)])
    
        

class ACVMFKL(ACVMF):
    def __init__(self,cov,costs,target_cost,K,L):
        self.K, self.L = K, L
        super().__init__(cov, costs)
    
    def get_rsquared(self,nsample_ratios):
        return get_rsquared_acv(
            self.cov,nsample_ratios,
            partial(get_discrepancy_covariances_KL,K=self.K,L=self.L,
                    pkg=torch))

class MFMC(ACVMF):
    def get_rsquared(self,nsample_ratios):
        return get_rsquared_mfmc(self.cov,nsample_ratios)

    def allocate_samples(self,target_cost):
        return allocate_samples_mfmc(self.cov, self.costs, target_cost)


class MLMC(ACVMF):
    def use_lagrange_formulation(self,flag):
        """For testing purposes only"""
        if flag:
            self.objective_fun_all = partial(
                acv_sample_allocation_objective_all_lagrange,self)
            self.jacobian_fun_all = partial(
                acv_sample_allocation_jacobian_all_lagrange,self)
        else:
            self.objective_fun_all = partial(
                acv_sample_allocation_objective_all,self)
            self.jacobian_fun_all = partial(
                acv_sample_allocation_jacobian_all,self)
            
    def get_rsquared(self,nsample_ratios):
        return get_rsquared_mlmc(self.cov,nsample_ratios,torch)

    def allocate_samples(self,target_cost):
        return allocate_samples_mlmc(self.cov, self.costs, target_cost)

def compute_single_fidelity_and_approximate_control_variate_mean_estimates(
        nhf_samples,nsample_ratios,
        model_ensemble,generate_samples,
        generate_samples_and_values,cov,
        get_cv_weights,seed):
    """
    Compute the approximate control variate estimate of a high-fidelity
    model from using it and a set of lower fidelity models. 
    Also compute the single fidelity Monte Carlo estimate of the mean from
    only the high-fidelity data.

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
    random_state = np.random.RandomState(seed)
    local_generate_samples = partial(
        generate_samples, random_state=random_state)
    samples,values =generate_samples_and_values(
        nhf_samples,nsample_ratios,model_ensemble,local_generate_samples)
    # compute mean using only hf data
    hf_mean = values[0][0].mean()
    # compute ACV mean
    eta = get_cv_weights(cov,nsample_ratios)
    acv_mean = compute_approximate_control_variate_mean_estimate(eta,values)
    return hf_mean, acv_mean

def estimate_variance_reduction(model_ensemble, cov, generate_samples,
                                allocate_samples,generate_samples_and_values,
                                get_cv_weights,get_rsquared=None,
                                ntrials=1e3,max_eval_concurrency=1,
                                target_cost=None, costs=None):
    """
    Numerically estimate the variance of an approximate control variate estimator
    and compare its value to the estimator using only the high-fidelity data.

    Parameters
    ----------
    ntrials : integer
        The number of times to compute estimator using different randomly 
        generated set of samples

    max_eval_concurrency : integer
        The number of processors used to compute realizations of the estimators,
        which can be run independently and in parallel.
    """
    
    M = cov.shape[0]-1 # number of lower fidelity models
    if costs is None:
        costs = np.asarray([100//2**ii for ii in range(M+1)])
    if target_cost is None:
        target_cost = int(1e4)

    nhf_samples,nsample_ratios = allocate_samples(
        cov, costs, target_cost)[:2]

    ntrials = int(ntrials)
    from multiprocessing import Pool
    pool = Pool(max_eval_concurrency)
    func = partial(
        compute_single_fidelity_and_approximate_control_variate_mean_estimates,
        nhf_samples,nsample_ratios,model_ensemble,generate_samples,
        generate_samples_and_values,cov,get_cv_weights)
    if max_eval_concurrency>1:
        assert int(os.environ['OMP_NUM_THREADS'])==1
        means = np.asarray(pool.map(func,[ii for ii in range(ntrials)]))
    else:
        means = np.empty((ntrials,2))
        for ii in range(ntrials):
            means[ii,:] = func(ii)

    numerical_var_reduction=means[:,1].var(axis=0)/means[:,0].var(axis=0)
    if get_rsquared is not None:
        true_var_reduction = 1-get_rsquared(cov[:M+1,:M+1],nsample_ratios)
        return means, numerical_var_reduction, true_var_reduction

    return means, numerical_var_reduction

def get_mfmc_control_variate_weights_pool_wrapper(cov,nsamples):
    """
    Create interface that adhears to assumed api for variance reduction check
    cannot be defined as a lambda locally in a test when using with 
    multiprocessing pool because python cannot pickle such lambda functions
    """
    return get_mfmc_control_variate_weights(cov)

def get_mlmc_control_variate_weights_pool_wrapper(cov,nsamples):
    """
    Create interface that adhears to assumed api for variance reduction check
    cannot be defined as a lambda locally in a test when using with 
    multiprocessing pool because python cannot pickle such lambda functions
    """
    return get_mlmc_control_variate_weights(cov.shape[0])

def plot_acv_sample_allocation(nsamples_history,costs,labels,ax):
    def autolabel(ax,rects,labels):
        #Attach a text label in each bar in *rects*
        for rect,label in zip(rects,labels):
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width()/2,
                            rect.get_y() + rect.get_height()/2),
                        xytext=(0, -10),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    nsamples_history = np.asarray(nsamples_history)
    xlocs = np.arange(nsamples_history.shape[0])
    nmodels = nsamples_history.shape[1]
    
    cnt = 0
    total_costs = nsamples_history.dot(costs)
    for ii in range(nmodels):
        rel_cost = nsamples_history[:,ii]*costs[ii]
        rel_cost /= total_costs
        rects = ax.bar(xlocs,rel_cost,bottom=cnt,edgecolor='white',
                       label=labels[ii])
        autolabel(ax,rects,['$%d$'%int(n) for n in nsamples_history[:,ii]])
        cnt+=rel_cost
    ax.set_xticks(xlocs)
    ax.set_xticklabels(['$%d$'%t for t in total_costs])
    ax.set_xlabel(r'$\mathrm{Total}\;\mathrm{Cost}$')
    ax.set_ylabel(r'$\mathrm{Percentage}\;\mathrm{of}\;\mathrm{Total}\;\mathrm{Cost}$ / $N_\alpha$')
    ax.legend(loc=[0.925,0.25])
