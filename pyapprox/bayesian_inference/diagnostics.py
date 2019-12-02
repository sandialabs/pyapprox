from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
def auto_correlation(x):
    assert x.ndim==1
    auto_correlation = np.correlate(x,x,mode='full')[x.shape[0]-1:]
    auto_correlation /= auto_correlation[0]
    return auto_correlation
    
def effective_sample_size(samples):
    num_vars, num_samples = samples.shape
    effective_sample_sizes = np.empty((num_vars),dtype=int)
    for dd in range(num_vars):
        auto_corr = auto_correlation(samples[dd,:])
        # find first negative instance of correlation
        idx = np.where(auto_corr<0)[0]
        if len(idx)>0:
            auto_corr = auto_corr[:idx[0]]
        #else:
        #    auto_corr = auto_corr

        effective_sample_sizes[dd] = num_samples/(1.+2.*auto_corr[1:].sum())
    return effective_sample_sizes

def get_random_traces(mcmc_chain,num_traces,num_samples_per_trace):
    """
    Get a set of contiguous subsets of an mcmc_chain that start at
    random times in the mcmc chain.

    Parameters
    ----------
    mcmc_chain : (num_vars x num_mcmc_samples)
        The full mcmc_chain

    num_traces : integer
        The number of subsets to extract

    num_samples_per_trace : integer
        The number of samples in each subset (trace)

    Returns
    -------
    traces : (num_traces,num_vars,num_samples_per_trace) 3D array
        The set of trace subsets
    """
    num_vars, num_mcmc_samples = mcmc_chain.shape
    traces_starting_index = np.random.permutation(
        num_mcmc_samples-num_samples_per_trace)[:num_traces]
    traces = np.empty((num_traces,num_vars,num_samples_per_trace),float)
    for i in range(num_traces):
        traces[i,:,:] = mcmc_chain[:,traces_starting_index[i]:traces_starting_index[i]+num_samples_per_trace]
    return traces

def plot_auto_correlation(samples,maxlags=100,show=False):
    num_vars = samples.shape[0]
    import matplotlib.pyplot as plt
    maxlags = min(maxlags,samples.shape[1]-1)
    nrows = num_vars//3+1
    ncols = min(num_vars,3)
    f,axs=plt.subplots(nrows,ncols,sharey=True,figsize=(16, 6))
    for dd in range(num_vars):
        axs[dd].acorr(
            samples[dd,:],detrend=plt.mlab.detrend_mean,maxlags=maxlags)
        axs[dd].set_xlim(-.1, 1.1)
        axs[dd].set_xlim(0, maxlags)
    if show:
        plt.show()

def plot_posterior_predictive_distribution(posterior_samples):
    pass
    

def gelman_rubin(traces):
    """Compute the Gelman Rubin diagnostic statistic R for a set of traces.

    This function tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If the chain has convergenced, the within-chain variances and and
    between-chain variances should be the same.

    Comment from https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introbayes_sect008.htm

    If all chains have reached the target distribution, this posterior variance estimate should be very close to the within-chain variance . Therefore, you would expect to see the ratio be close to 1. The square root of this ratio is referred to as the potential scale reduction factor (PSRF). A large PSRF indicates that the between-chain variance is substantially greater than the within-chain variance, so that longer simulation is needed. If the PSRF is close to 1, you can conclude that each of the chains has stabilized, and they are likely to have reached the target distribution.

    Parameters
    ----------
    traces : list of num_trances (num_vars x num_samples) matrices
      A set of traces (sub chains of full MCMC chain)

    Returns
    -------
    potential_scale_reduction_factors : float
      Return the Gelman Rubin potential scale reduction factors for each
      dimension
    """
    num_traces, num_vars, num_samples_per_trace = traces.shape
    potential_scale_reduction_factors = np.empty((num_vars),float)
    over_estimate_of_variances = np.empty((num_vars),float)
    for d in range(num_vars):
        traces_d = traces[:,d,:]
        between_chain_var = num_samples_per_trace*np.sum(
            (np.mean(traces_d,1)-np.mean(traces_d))**2)/float(num_traces-1.)
        within_chain_var = np.sum(
            (traces_d-np.mean(traces_d,1)[:,np.newaxis])**2)
        within_chain_var /= float(num_traces*(num_samples_per_trace-1.))
        a = between_chain_var/float(num_samples_per_trace)/+\
          within_chain_var*float((num_samples_per_trace-1.)/(num_samples_per_trace))
        posterior_marginal_variance = \
          float(num_traces+1.)/float(num_traces*num_samples_per_trace)*between_chain_var+\
          within_chain_var*(num_samples_per_trace-1.)/float(num_samples_per_trace)

        potential_scale_reduction_factors[d] = \
          posterior_marginal_variance/within_chain_var

        over_estimate_of_variances[d] = posterior_marginal_variance-between_chain_var/float(num_traces*num_samples_per_trace)

    return potential_scale_reduction_factors, over_estimate_of_variances

