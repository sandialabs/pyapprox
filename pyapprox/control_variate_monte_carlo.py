"""
Functions for estimating expectations using frequentist control-variate Monte-Carlo based methods such as multi-level Monte-Carlo, control-variate Monte-Carlo, and approximate control-variate Monte-Carlo.
"""
import numpy as np
from scipy.optimize import minimize
#use torch to compute gradients for sample allocation optimization
import torch
import copy
from pyapprox.utilities import get_all_sample_combinations

def compute_correlations_from_covariance(cov):
    """
    Compute the correlation matrix of a covariance matrix.

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]
    
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
    #nhf_samples = int(max(1,np.round(nhf_samples)))
    nhf_samples = int(max(1,np.floor(nhf_samples)))
    nsample_ratios = np.floor(nsamples)/nhf_samples
    #nsample_ratios = [max(np.round(nn/nhf_samples),0) for nn in nsamples]
    return nhf_samples, nsample_ratios

def get_variance_reduction(get_rsquared,cov,nsample_ratios):
    """
    Compute the variance reduction:
    
    .. math:: \gamma = 1-r^2
    
    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Returns
    -------
    gamma : float
        The variance reduction
    """
    return 1-get_rsquared(cov,nsample_ratios)

def get_control_variate_rsquared(cov,nsample_ratios):
    """
    Compute :math:`r^2` used to compute the variance reduction of 
    control variate Monte Carlo

    .. math:: \gamma = 1-r^2, \qquad     r^2 = c^TC^{-1}c
    
    where c is the first column of C

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1
        For control variate with known means these values are meaningless
        The parameter is here to maintain a consistent API

    Returns
    -------
    rsquared : float
        The value  :math:`r^2`
    """
    assert nsample_ratios is None
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
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

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

def get_rsquared_mlmc(cov,nsample_ratios):
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
    rhat = np.ones(nmodels)
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

def compute_control_variate_mean_estimate(weights,values):
    """
    Use control variate Monte Carlo to estimate the mean of high-fidelity data.

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


def allocate_samples_mfmc(cov, costs, target_cost, nhf_samples_fixed=None):
    """
    Determine the samples to be allocated to each model when using MFMC

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    target_cost : float
        The total cost budget

    nhf_samples_fixed : integer default=None
        If not None fix the number of high-fidelity samples and compute
        the samples assigned to the remaining models to respect this

    Returns
    -------
    nhf_samples : integer 
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    log10_variance : float
        The base 10 logarithm of the variance of the estimator
    """
    nmodels = cov.shape[0]
    if nhf_samples_fixed is not None:
        assert nmodels>2 # remove this assumption
        cost_left=target_cost-nhf_samples_fixed*costs[0]-\
            nhf_samples_fixed*costs[1]
        n2, nsample_ratios_left, var = allocate_samples_mfmc(
            cov[1:, 1:], costs[1:], cost_left)

        nhf_samples = nhf_samples_fixed
        nsample_ratios = [None] * (nmodels-1)
        nsample_ratios[0] = (nhf_samples_fixed + n2) / nhf_samples_fixed
        nsample_ratios[1:]=[r*n2/nhf_samples_fixed for r in nsample_ratios_left]
        nhf_samples = max(nhf_samples, 1)
    else:
        # this can be used for two models
        corr = compute_correlations_from_covariance(cov)
        r = []
        for ii in range(nmodels-1):
            # Step 3 in Algorithm 2 in Peherstorfer et al 2016
            num = costs[0] * (corr[0, ii]**2 - corr[0, ii+1]**2)
            den = costs[ii] * ( 1 - corr[0, 1]**2)
            r.append(np.sqrt(num / den))


        num = costs[0]*corr[0,-1]**2
        den = costs[-1] * (1 - corr[0, 1]**2)
        r.append(np.sqrt(num / den))

        # Step 4 in Algorithm 2 in Peherstorfer et al 2016
        nhf_samples = target_cost / np.dot(costs, r)
        nhf_samples = max(nhf_samples, 1)
        nsample_ratios = r[1:]

    nhf_samples, nsample_ratios = standardize_sample_ratios(
        nhf_samples, nsample_ratios)
    gamma = get_variance_reduction(get_rsquared_mfmc,cov,nsample_ratios)
    log10_variance = np.log10(gamma)+np.log10(cov[0, 0])-np.log10(
        nhf_samples)

    return nhf_samples, nsample_ratios, log10_variance

def allocate_samples_mlmc(cov, costs, target_cost, nhf_samples_fixed=None):
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

    nhf_samples_fixed : integer default=None
        If not None fix the number of high-fidelity samples and compute
        the samples assigned to the remaining models to respect this

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
    if nhf_samples_fixed is not None:
        cost_left=target_cost-nhf_samples_fixed*costs[0]-\
            nhf_samples_fixed*costs[1]
        n2, nsample_ratios_left, var = allocate_samples_mlmc(
            cov[1:, 1:], costs[1:], cost_left)

        nhf_samples = nhf_samples_fixed
        nsample_ratios = [None] * (nmodels-1)
        nsample_ratios[0] = (nhf_samples_fixed + n2) / nhf_samples_fixed
        nsample_ratios[1:]=[
            r*n2/nhf_samples_fixed for r in nsample_ratios_left]
        nhf_samples, nsample_ratios = standardize_sample_ratios(
            nhf_samples, nsample_ratios)
        for ii in range(len(nsample_ratios)-1):
            if nsample_ratios[ii] == nsample_ratios[ii+1]:
                nsample_ratios[ii+1] += 0.5
    else:
        sum1 = 0.0
        nsamples = []
        for ii in range(nmodels-1):
            # compute the variance of the discrepancy 
            vardelta = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
            # compute the variance * cost
            vc = vardelta * (costs[ii] + costs[ii+1])
            # compute the unnormalized number of samples\
            # these values will be normalized by lamda later
            nsamp = np.sqrt(vardelta / (costs[ii] + costs[ii+1]))
            nsamples.append(nsamp)
            sum1 += np.sqrt(vc)

        # compute information for lowest fidelity model
        v = cov[nmodels-1, nmodels-1]
        c = costs[nmodels-1]
        nsamples.append(np.sqrt(v/c))
        sum1 += np.sqrt(v*c)

        # compute the ML estimator variance from the target cost
        variance = sum1**2 / target_cost
        # compute the lagrangian parameter
        lamda = sum1/variance
        # compute the number of samples allocated to resolving each
        # discrepancy.
        nl = [lamda * n for n in nsamples]

        # compute the number of samples allocated to each model. For
        # all but the highest fidelity model we need to collect samples
        # from two discrepancies.
        nhf_samples = nl[0]
        nsample_ratios = []
        for ii in range(1, nmodels-1):
            nsample_ratios.append((nl[ii-1] + nl[ii])/nl[0])
        if nmodels>1:
            nsample_ratios.append((nl[-2]+nl[-1])/nl[0])

        nhf_samples = max(nhf_samples, 1)
    nsample_ratios = np.asarray(nsample_ratios)

    nhf_samples, nsample_ratios = standardize_sample_ratios(
        nhf_samples, nsample_ratios)
    gamma = get_variance_reduction(get_rsquared_mlmc,cov,nsample_ratios)
    log10_variance=np.log10(gamma)+np.log10(cov[0, 0])-np.log10(
        nhf_samples)
    #print(log10_variance)
    if np.isnan(log10_variance):
        raise Exception('MLMC variance is NAN')
    return nhf_samples, np.asarray(nsample_ratios), log10_variance

def get_discrepancy_covariances_IS(cov,nsample_ratios,pkg=np):
    """
    Get the covariances of the discrepancies :math:`\delta` 
    between each low-fidelity model and its estimated mean when the same 
    :math:`N` samples are used to compute the covariance between each models and
    :math:`N-r_\alpha` samples are allocated to 
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
        which returns the covariances between the discrepnacies betweem the 
        low-fidelity models and their approximated mean.

    Returns
    -------
    weights : np.ndarray (nmodels-1)
        The control variate weights
    """
    CF,cf = get_discrepancy_covariances(cov,nsample_ratios)
    weights = -np.linalg.solve(CF, cf)
    return weights
    
def get_rsquared_acv1(cov,nsample_ratios,get_discrepancy_covariances):
    """
    Compute r^2 used to compute the variance reduction  of 
    Approximate Control Variate Algorithm 1 (ACV1)
    
    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Returns
    -------
    rsquared : float
        The value r^2
    """
    CF,cf = get_discrepancy_covariances(cov,nsample_ratios)
    if type(cov)==np.ndarray:
        rs = np.dot(cf,np.linalg.solve(CF,cf))/cov[0, 0]
    else:
        rs = torch.dot(cf, torch.mv(torch.inverse(CF),cf))/cov[0, 0]
    return rs

def acv_sample_allocation_sample_ratio_constraint(ratios, *args):
    ind = args[0]
    return ratios[ind] - ratios[ind-1]

def generate_samples_and_values_acv_IS(nhf_samples,nsample_ratios,functions,
                                       generate_samples):
    """
    WARNING: A function may be evaluated at the same sample twice. To avoid
    this pass in a function that does a look up before evaluating a sample
    and returns the pecomputed value if it is found.
    TODO: Create a version of this function that avoids redundant computations
    and evaluates all samples at once so something pool wrapper can be used.
    """
    nmodels = len(functions)
    samples1 = [generate_samples(nhf_samples)]*nmodels
    samples2 = [None]+[np.hstack(
        [generate_samples(nhf_samples*r-nhf_samples),samples1[ii+1]])
                       for ii,r in enumerate(nsample_ratios)]
    values1  = [f(s) for f,s in zip(functions,samples1)]
    values2  = [None]+[f(s) for f,s in zip(functions[1:],samples2[1:])]
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

def generate_samples_and_values_mfmc(nhf_samples,nsample_ratios,functions,
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
    nsample_ratios = np.asarray(nsample_ratios)
    nmodels = len(nsample_ratios)+1
    if not callable(functions):
        assert len(functions)==nmodels
    assert np.all(nsample_ratios>=1)
    # check nhf_samples is an integer
    assert nhf_samples/int(nhf_samples)==1.0
    # convert to int if a float because numpy random assumes nsamples is an int
    nhf_samples = int(nhf_samples)
    nlf_samples = nhf_samples*nsample_ratios
    for ii in range(nmodels-1):
        assert nlf_samples[ii]/int(nlf_samples[ii])==1.0
    nlf_samples = np.asarray(nlf_samples,dtype=int)

    max_nsamples = nlf_samples.max()
    samples = generate_samples(max_nsamples)
    samples1 = [samples[:,:nhf_samples]]
    samples2 = [None]
    nprev_samples = nhf_samples
    for ii in range(1,nmodels):
        samples1.append(samples[:,:nprev_samples])
        samples2.append(samples[:,:nlf_samples[ii-1]])
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
            nprev_samples = samples2[ii].shape[1]
            
    samples = [[s1,s2] for s1,s2 in zip(samples1,samples2)]
    values  = [[v1,v2] for v1,v2 in zip(values1,values2)]
        
                        
    return samples,values

def acv_sample_allocation_cost_constraint(ratios, *args):
    nhf, costs, target_cost = args
    cost = nhf*(costs[0] + np.dot(ratios, costs[1:]))
    return target_cost - cost

def acv_sample_allocation_cost_constraint_all(ratios, *args):
    costs, target_cost = args
    # nhf, rats = standardize(ratios[0], ratios[1:])
    nhf, rats = ratios[0], ratios[1:]
    cost = nhf*(costs[0] + np.dot(rats, costs[1:]))
    return target_cost - cost 

def acv_sample_allocation_objective(ratios, acv_in):
    rats = torch.tensor(ratios)
    gamma = acv_in.gamma(rats)
    gamma = torch.log10(gamma)
    return gamma.item()

def acv_sample_allocation_jacobian(ratios, acv_in):
    rats = torch.tensor(ratios, dtype=torch.double)
    rats.requires_grad=True
    gamma = acv_in.gamma(rats)
    gamma = torch.log10(gamma)
    gamma.backward()
    grad = rats.grad.numpy().copy()
    rats.grad.zero_()
    return grad

def acv_sample_allocation_objective_all(x, acv_in):
    xrats = torch.tensor(x, dtype=torch.double)
    xrats.requires_grad=True
    nhf, ratios = xrats[0], xrats[1:]
    gamma = acv_in.gamma(ratios) * acv_in.cov[0, 0] / nhf
    gamma = torch.log10(gamma)
    return gamma.item()

def acv_sample_allocation_jacobian_all(x, acv_in):
    xrats = torch.tensor(x, dtype=torch.double)
    xrats.requires_grad=True
    nhf, ratios = xrats[0], xrats[1:]
    gamma = acv_in.gamma(ratios) * acv_in.cov[0, 0] / nhf
    gamma = torch.log10(gamma)
    gamma.backward()
    grad = xrats.grad.numpy().copy()
    xrats.grad.zero_()
    return grad

def allocate_samples_acv(cov, costs, target_cost, nhf_samples_fixed=None):
    """
    Determine the samples to be allocated to each model when using ACV1 or ACV2

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    target_cost : float
        The total cost budget

    nhf_samples_fixed : integer default=None
        If not None fix the number of high-fidelity samples and compute
        the samples assigned to the remaining models to respect this

    Returns
    -------
    nhf_samples : integer 
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    log10_variance : float
        The base 10 logarithm of the variance of the estimator
    """
    cov = torch.tensor(np.copy(cov), dtype=torch.double)
    nmodels = cov.shape[0]
    nhf_samples, nsample_ratios =  allocate_samples_mlmc(
        cov, costs, target_cost, nhf_samples_fixed)[1:]
    nhf_samples_start = copy.deepcopy(nhf_samples)
    nsample_ratios_start = copy.deepcopy(nsample_ratios)
    if nhf_samples_fixed is not None:
        bounds = [(2, 10**20)]*(nmodels-1)
        cons = dict({'type':'ineq', 'fun':acv_sample_allocation_cost_constraint,
                     'args':(nhf_samples_fixed, costs, target_cost)})
        cons = [cons]
        for jj in range(1,nmodels-1):
            cons.append( dict({'type':'ineq',
                               'fun':acv_sample_allocation_ratio_constraint,
                               'args':[jj]}))
        r_start = nsample_ratios            
        opt = minimize(objective_function, r_start, acv,
                       method='SLSQP',
                       jac=jacobian, bounds=bounds,
                       constraints=cons,
                       options = {'disp':False, 'ftol':1e-8,
                                  'maxiter':600})

        ratio = opt.x
    else:

        bounds = [(1,10**20)] + [(2, 10**20)]*(nmodels-1)
        cons = dict({'type':'ineq',
                     'fun':acv_sample_allocation_cost_constraint_all,
                     'args':(costs, target_cost)})
        cons = [cons]

        for jj in range(2,nmodels-1):
            cons.append( dict({'type':'ineq',
                               'fun':acv_sample_allocation_ratio_constraint_all,
                               'args':[jj]}))

        print (type([nhf_samples]),type(nsample_ratios))
        r_start = [nhf_samples] + nsample_ratios
        opt = minimize(objective_function_all, r_start, acv,
                       method='SLSQP',
                       jac=jacobian_all, bounds=bounds,
                       constraints=cons,
                       options = {'disp':False,
                                  'ftol':1e-8,
                                  'maxiter':400})

        max_eval_total = 100000000
        if opt.success == False or np.max(opt.x[0] * opt.x[1:])>max_eval_total:
            raise Exception('Optimizer failed')
        nhf_samples, ratio = opt.x[0], opt.x[1:]

    nhf_samples, nsample_ratios = standardize_sample_ratios(
        nhf_samples_fixed, ratio)
    nsample_ratios = np.array(nsample_ratios)
    #var = acv.gamma(nsample_ratios) * acv.cov[0, 0] / float(nhf_samples)
    var = get_variance_reduction(get_rsquared_acv,cov,nsample_ratios)
    var *= cov[0, 0]/float(nhf_samples)
    return nhf_samples, nsample_ratios, np.log10(var.item())

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
    cov : np.ndarray(nqoi,nqoi)
        The covariance between the model qoi
    """
    # generate pilot samples
    pilot_samples = generate_samples(npilot_samples)
    config_vars = np.arange(model_ensemble.nmodels)[np.newaxis,:]
    # append model ids to pilot smaples
    pilot_samples = get_all_sample_combinations(pilot_samples,config_vars)
    # evaluate models at pilot samples
    pilot_values = model_ensemble(pilot_samples)
    pilot_values = np.reshape(
        pilot_values,(npilot_samples,model_ensemble.nmodels))
    # compute covariance
    cov = np.cov(pilot_values,rowvar=False)
    return cov, pilot_samples, pilot_values
