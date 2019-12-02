#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def compute_correlations_from_covariance(cov):
    """
    Compute the correlation matrix of a covariance matrix.

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    
    Return
    ------
    corr : np.ndarray (nmodels,nmodels)
        The correlation matrix
    """
    corr_sqrt = np.diag(np.sqrt((np.diag(cov)))**-1)        
    corr = np.dot(corr_sqrt, np.dot(cov, corr_sqrt))
    return corr

def standardize_sample_ratios(nhf_samples,nsample_ratios):
    """
    Ensure num high fidelity samples is positive (>0) and then recompute 
    sample ratios. This is useful when num high fidelity samples and sample 
    ratios are computed by an optimization process. This function is useful for
    optimization problems with a numerical or analytical solution.

    Parameters
    ----------
    nhf_samples : integer
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Return
    ------
    nhf_samples : integer
        The corrected number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The corrected sample ratios
    """
    nsamples = [r*nhf_samples for r in nsample_ratios]
    num_hf_samples = max(1,np.round(nhf_samples))
    sample_ratios = [max(np.round(nn/nhf_samples),0) for nn in nsamples]
    return nhf_samples, nsample_ratios

def get_variance_reduction(get_rsquared,cov,nsample_ratios):
    """
    Compute the variance reduction:
    \gamma = 1-r^2
    
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
    Compute r^2 used to compute the variance reduction of 
    control variate Monte Carlo:
    \gamma = 1-r^2
    
    r^2 = c^TC^{-1}c
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
        The value r^2
    """
    nmodels = cov.shape[0]
    assert len(costs)==cov
    rsquared = self.cov[0,:1].dot(np.linalg.solve(cov[1:, 1:], cov[:1, 0]))
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
    corr = compute_correlations_from_covariance(cov)
    
    nmodels = cov.shape[0]
    assert len(nsample_ratios)==nmodels-1
    rsquared=(nsample_ratios[0]-1)/(nsample_ratios[0]+1e-20)*corr[0, 1]**2
    for ii in range(1, nmodels-1):
        p1 = (nsample_ratios[ii]-nsample_ratios[ii-1])/(
            nsample_ratios[ii]*nsample_ratios[ii-1] + 1e-20)
        p1 *= corr[0, ii]**2
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
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1.
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
    return -gamma #TODO check if need negative sign

def get_mlmc_control_variate_weights(nmodels):
    """
    Get the weights used by the MLMC control variate estimator

    Return
    ------
    weights : np.ndarray (nmodels-1)
        The control variate weights
    """
    return -np.ones(nmodels-1)

def get_approximate_control_variate_weights(nsample_ratios):
    """
    Get the weights used by the approximate control variate estimator

    Return
    ------
    weights : np.ndarray (nmodels-1)
        The control variate weights
    """
    
    
    weights = - np.linalg.solve(covh, covhh)
    return weights

def compute_control_variate_mean_estimate(weights,values):
    """
    Use control variate Monte Carlo to estimate the mean of high-fidelity data

    Parameters
    ----------
    values : list (nmodels)
        Evaluations of each model. Each model has two sets of data. 
        [np.ndarray (num_samples_i0,num_qoi),
         np.ndarray (num_samples_i0,num_qoi)]

        The first data set is used to compute the estimator \hat{Q}_i of 
        the model and the second one is used to compute the approximate 
        mean \hat{\mu}_i of the model.

    weights : np.ndarray (nmodels-1)
        the control variate weights

    Return
    ------
    est : float
        The control variate estimate of the mean
    """
    nmodels = len(values)
    # high fidelity monte carlo estimate of mean
    est = values[0].mean()
    for ii in range(nmodels-1):
        assert len(values[ii+1])==2
        est += weights[ii]*(values[ii+1][0].mean()-values[ii+1][1].mean())
        return est


def allocate_samples_mfmc(self, cov, costs, cost_goal, nhf_samples_fixed=None):
    """
    Determine the samples to be allocated to each model when using MFMC

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    cost_goal : float
        The total cost budget

    nhf_samples_fixed : integer default=None
        If not None fix the number of high-fidelity samples and compute
        the samples assigned to the remaining models to respect this

    Return
    ------
    The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    var : float
        The ...
    """
    nmodels = cov.shape[0]
    if nhf_samples_fixed is not None:
        cost_left=cost_goal-nhf_samples_fixed*costs[0]-\
            nhf_samples_fixed*costs[1]
        n2, nsample_ratios_left, var = allocate_samples_mfmc(
            cov[1:, 1:], costs[1:], cost_left)

        nhf_samples = nhf_samples_fixed
        nsample_ratios = [None] * (nmodels-1)
        nsample_ratios[0] = (nhf_samples_fixed + n2) / nhf_samples_fixed
        nsample_ratios[1:]=[r*n2/nhf_samples_fixed for r in nsample_ratios_left]
        nhf_samples = max(nhf_samples, 1)
    else:
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
        nhf_samples = cost_goal / np.dot(costs, r)
        nhf_samples = max(nhf_samples, 1)
        nsample_ratios = r[1:]

    nhf_samples, nsample_ratios = standardize(nhf_samples, nsample_ratios)
    gamma = get_variance_reduction(get_rsquared_mfmc,cov,nsample_ratios)
    var = np.log10(gamma) + np.log10(cov[0, 0]) - np.log10(nhf_samples)

    return nhf_samples, nsample_ratios, var

def allocate_samples_mlmc(self, cov, costs, cost_goal, nhf_samples_fixed=None):
    """
    Determine the samples to be allocated to each model when using MLMC

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity model
        is the first model, i.e its variance is cov[0,0]

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    cost_goal : float
        The total cost budget

    nhf_samples_fixed : integer default=None
        If not None fix the number of high-fidelity samples and compute
        the samples assigned to the remaining models to respect this

    Return
    ------
    The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the 
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    var : float
        The ...
    """
    nmodels = cov.shape[0]
    if nhf_samples_fixed is not None:
        cost_left=cost_goal-nhf_samples_fixed*costs[0]-\
            nhf_samples_fixed*costs[1]
        n2, nsample_ratios_left, var = allocate_samples_mlmc(
            cov[1:, 1:], costs[1:], cost_left)

        nhf_samples = nhf_samples_fixed
        nsample_ratios = [None] * (nmodels-1)
        nsample_ratios[0] = (nhf_samples_fixed + n2) / nhf_samples_fixed
        nsample_ratios[1:]=[r*n2/nhf_samples_fixed for r in nsample_ratios_left]
        nhf_samples, nsample_ratios = standardize(nhf_samples, nsample_ratios)
        for ii in range(len(nsample_ratios)-1):
            if nsample_ratios[ii] == nsample_ratios[ii+1]:
                nsample_ratios[ii+1] += 0.5
    else:
        mu = 0.0
        nsamples = []
        for ii in range(nmodels-1):
            vardelta = cov[ii, ii] + cov[ii+1, ii+1] - 2*cov[ii, ii+1]
            vc = vardelta * (costs[ii] + costs[ii+1])
            nsamp = np.sqrt(vardelta / (costs[ii] + costs[ii+1]))
            nsamples.append(nsamp)
            mu += np.sqrt(vc)

        v = cov[nmodels-1, nmodels-1]
        c = costs[-1]
        nsamples.append(np.sqrt(v/c))
        mu += np.sqrt(v*c)

        variance = mu**2 / cost_goal
        mu /= variance
        nl = [mu * n for n in nsamples]

        nhf_samples = nl[0]
        nsample_ratios = []
        for ii in range(1, nmodels-1):
            nsample_ratios.append((nl[ii-1] + nl[ii])/nl[0])
        nsample_ratios.append((nl[-2]+nl[-1])/nl[0])

        nhf_samples = max(nhf_samples, 1)
        
    nhf_samples, nsample_ratios = standardize(nhf_samples, nsample_ratios)
    gamma = get_variance_reduction(get_rsquared_mlmc,cov,nsample_ratios)
    var = np.log10(gamma) + np.log10(cov[0, 0]) - np.log10(nhf_samples)
    if np.isnan(var):
        raise Exception('MLMC variance is NAN')
    return nhf_samples, nsample_ratios, var    
