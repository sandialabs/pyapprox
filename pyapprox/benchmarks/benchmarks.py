#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from functools import partial

import pyapprox as pya

def ishigami_function(samples,a=7,b=0.1):
    if samples.ndim==1:
        samples = samples[:,np.newaxis]
    vals = np.sin(samples[0,:])+a*np.sin(samples[1,:])**2+\
        b*samples[2,:]**4*np.sin(samples[0,:])
    return vals[:,np.newaxis]

def ishigami_function_jacobian(samples,a=7,b=0.1):
    if samples.ndim==1:
        samples = samples[:,np.newaxis]
    assert samples.shape[1]==1
    nvars=3
    assert samples.shape[0]==nvars
    jac = np.empty((nvars,1))
    jac[0] = np.cos(samples[0,:]) + b*samples[2,:]**4*np.cos(samples[0,:])
    jac[1] = 2*a*np.sin(samples[1,:])*np.cos(samples[1,:])
    jac[2] = 4*b*samples[2,:]**3*np.sin(samples[0,:])
    return jac

def ishigami_function_hessian(samples,a=7,b=0.1):
    if samples.ndim==1:
        samples = samples[:,np.newaxis]
    assert samples.shape[1]==1
    nvars=3
    assert samples.shape[0]==nvars
    hess = np.empty((nvars,nvars))
    hess[0,0] = -np.sin(samples[0,:]) - b*samples[2,:]**4*np.sin(samples[0,:])
    hess[1,1] = 2*a*(np.cos(samples[1,:])**2-np.sin(samples[1,:])**2)
    hess[2,2] = 12*b*samples[2,:]**2*np.sin(samples[0,:])
    hess[0,1],hess[1,0]=0,0
    hess[0,2]=4*b*samples[2,:]**3*np.cos(samples[0,:]); hess[2,0]=hess[0,2]
    hess[1,2],hess[2,1]=0,0
    return hess

def get_ishigami_funciton_statistics(a=7,b=0.1):
    """
    p_i(X_i) ~ U[-pi,pi]
    """
    mean = a/2
    variance = a**2/8+b*np.pi**4/5+b**2*np.pi**8/18+0.5
    D_1 = b*np.pi**4/5+b**2*np.pi**8/50+0.5
    D_2,D_3,D_12,D_13 = a**2/8,0,0,b**2*np.pi**8/18-b**2*np.pi**8/50
    D_23,D_123=0,0
    main_effects =  np.array([D_1,D_2,D_3])/variance
    # the following two ways of calulating the total effects are equivalent
    total_effects1 = np.array(
        [D_1+D_12+D_13+D_123,D_2+D_12+D_23+D_123,D_3+D_13+D_23+D_123])/variance
    total_effects=1-np.array([D_2+D_3+D_23,D_1+D_3+D_13,D_1+D_2+D_12])/variance
    assert np.allclose(total_effects1,total_effects)
    sobol_indices = np.array([D_1,D_2,D_3,D_12,D_13,D_23,D_123])/variance
    return mean, variance, main_effects, total_effects, sobol_indices

def sobol_g_function(coefficients,samples):
    """
    The coefficients control the sensitivity of each variable. Specifically
    they limit the range of the outputs, i.e.
    1-1/(1+a_i) <= (abs(4*x-2)+a_i)/(a_i+1) <= 1-1/(1+a_i)
    """
    nvars,nsamples = samples.shape
    assert coefficients.shape[0]==nvars
    vals = np.prod((np.absolute(4*samples-2)+coefficients[:,np.newaxis])/
                   (1+coefficients[:,np.newaxis]),axis=0)[:,np.newaxis]
    assert vals.shape[0]==nsamples
    return vals

def get_sobol_g_function_statistics(a,interaction_terms=None):
    """
    See article: Variance based sensitivity analysis of model output. 
    Design and estimator for the total sensitivity index
    """
    nvars = a.shape[0]
    mean = 1
    unnormalized_main_effects = 1/(3*(1+a)**2)
    variance = np.prod(unnormalized_main_effects+1)-1
    main_effects = unnormalized_main_effects/variance
    total_effects = np.tile(np.prod(unnormalized_main_effects+1),(nvars))
    total_effects *= unnormalized_main_effects/(unnormalized_main_effects+1)
    total_effects /= variance
    if interaction_terms is None:
        return mean,variance,main_effects,total_effects
    
    sobol_indices = np.array([
        unnormalized_main_effects[index].prod()/variance
        for index in interaction_terms])
    return mean,variance,main_effects,total_effects,sobol_indices

def setup_sobol_g_function(nvars):
    univariate_variables = [stats.uniform(0,1)]*nvars
    variable=pya.IndependentMultivariateRandomVariable(univariate_variables)
    a = (np.arange(1,nvars+1)-2)/2
    mean, variance, main_effects, total_effects = \
        get_sobol_g_function_statistics(a)
    return {'fun':partial(sobol_g_function,a),
            'mean':mean,'variance':variance,'main_effects':main_effects,
            'total_effects':total_effects,'var':variable}

def setup_ishigami_function(a,b):
    univariate_variables = [stats.uniform(-np.pi,2*np.pi)]*3
    variable=pya.IndependentMultivariateRandomVariable(univariate_variables)
    mean, variance, main_effects, total_effects, sobol_indices = \
        get_ishigami_funciton_statistics()
    return {'fun':partial(ishigami_function,a=a,b=b),
            'jac':partial(ishigami_function_jacobian,a=a,b=b),
            'hess':partial(ishigami_function_hessian,a=a,b=b),'var':variable,
            'mean':mean,'variance':variance,'main_effects':main_effects,
            'total_effects':total_effects,'sobol_indices':sobol_indices}

from scipy.optimize import rosen, rosen_der, rosen_hess_prod
def rosenbrock_function(samples):
    return rosen(samples)[:,np.newaxis]

def rosenbrock_function_jacobian(samples):
    assert samples.shape[1]==1
    return rosen_der(samples)

def rosenbrock_function_hessian_prod(samples,vec):
    assert samples.shape[1]==1
    return rosen_hess_prod(samples[:,0],vec[:,0])[:,np.newaxis]

def setup_rosenbrock_function(nvars):
    univariate_variables = [stats.uniform(-2,4)]*nvars
    variable=pya.IndependentMultivariateRandomVariable(univariate_variables)
    
    return {'fun':rosenbrock_function,'jac':rosenbrock_function_jacobian,
            'hessp':rosenbrock_function_hessian_prod,'var':variable}

def setup_benchmark(name,**kwargs):
    benchmarks = {'sobol_g':setup_sobol_g_function,
                  'ishigami':setup_ishigami_function,
                  'rosenbrock':setup_rosenbrock_function}

    if name not in benchmarks:
        msg = f'Benchmark "{name}" not found.\n Avaiable benchmarks are:\n'
        for key in benchmarks.keys():
            msg += f"\t{key}\n"
        raise Exception(msg)

    return benchmarks[name](**kwargs)
