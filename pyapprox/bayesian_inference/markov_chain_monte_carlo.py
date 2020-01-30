import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
from scipy.optimize import approx_fprime


class GaussianLogLike(object):
    """
    A Gaussian log-likelihood function for a model with parameters given in 
    sample
    """
    def __init__(self,model,data,noise_stdev):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        model : callable
            The model relating the data and noise 

        data : np.ndarray (nobs)
            The "observed" data that our log-likelihood function takes in

        noise_stdev : float
            The noise standard deviation that our function requires.
        """
        self.model=model
        self.data=data
        self.noise_stdev=noise_stdev

    def __call__(self,samples):
        model_vals = self.model(samples)
        return -(0.5/self.noise_stdev**2)*np.sum((self.data - model_vals)**2)

class LogLike(tt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        samples, = inputs # important
        # call the log-likelihood function
        logl = self.likelihood(samples)
        outputs[0][0] = np.array(logl) # output the log-likelihood

# define a theano Op for our likelihood function
class LogLikeWithGrad(LogLike):

    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        super().__init__(loglike)
        
        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood)

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        samples, = inputs # important
        return [g[0]*self.logpgrad(samples)]

class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        samples,=inputs
        
        # define version of likelihood function to pass to derivative function
        def lnlike(values):
            return self.likelihood(values)

        # calculate gradients
        grads = approx_fprime(samples,lnlike,2*np.sqrt(np.finfo(float).eps))
        outputs[0][0] = grads

def extract_mcmc_chain_from_pymc3_trace(trace,var_names,ndraws,nburn,njobs):
    nvars = len(var_names)
    samples = np.empty((nvars,(ndraws-nburn)*njobs))
    effective_sample_size = np.empty(nvars)
    for ii in range(nvars):
        samples[ii,:]=trace.get_values(
            var_names[ii],burn=nburn,chains=np.arange(njobs))
        effective_sample_size[ii]=pm.ess(trace)[var_names[ii]].values

    return samples,effective_sample_size

def extract_map_sample_from_pymc3_dict(map_sample_dict,var_names):
    nvars = len(var_names)
    map_sample = np.empty((nvars,1))
    for ii in range(nvars):
        map_sample[ii]=map_sample_dict[var_names[ii]]
    return map_sample

from pyapprox.variables import get_distribution_info
def get_pymc_variables(variables,pymc_var_names=None):
    nvars = len(variables)
    if pymc_var_names is None:
        pymc_var_names = ['z_%d'%ii for ii in range(nvars)]
    assert len(pymc_var_names)==nvars
    pymc_vars = []
    for ii in range(nvars):
        pymc_vars.append(get_pymc_variable(variables[ii],pymc_var_names[ii]))
    return pymc_vars, pymc_var_names

def get_pymc_variable(rv,pymc_var_name):
    name, scales, shapes = get_distribution_info(rv)
    if rv.dist.name=='norm':
        return pm.Normal(pymc_var_name,mu=scales['loc'],sigma=scales['scale'])
    if rv.dist.name=='uniform':        
        return pm.Uniform(pymc_var_name,lower=scales['loc'],
                          upper=scales['loc']+scales['scale'])
    msg = f'Variable type: {name} not supported'
    raise Exception(msg)

def run_bayesian_inference_gaussian_error_model(
        loglike,variables,ndraws,nburn,njobs,
        algorithm='nuts',get_map=False,print_summary=False):
    # create our Op
    if algorithm!='nuts':
        logl = LogLike(loglike)
    else:
        logl = LogLikeWithGrad(loglike)

    # use PyMC3 to sampler from log-likelihood
    with pm.Model():
        # must be defined inside with pm.Model() block
        pymc_variables, pymc_var_names = get_pymc_variables(
            variables.all_variables())

        # convert m and c to a tensor vector
        theta = tt.as_tensor_variable(pymc_variables)

        # use a DensityDist (use a lamdba function to "call" the Op)
        pm.DensityDist(
            'likelihood', lambda v: logl(v), observed={'v': theta})

        if get_map:
            map_sample_dict = pm.find_MAP()

        if algorithm=='smc':
            assert njobs==1 # njobs is always 1 when using smc
            trace = pm.sample_smc(ndraws);
        else:
            if algorithm=='metropolis':
                step=pm.Metropolis(pymc_variables)
            elif algorithm=='nuts':
                step=pm.NUTS(pymc_variables)
            
            trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True,
                              start=None,cores=njobs,step=step)

    if print_summary:
        print(pm.summary(trace))
        
    samples, effective_sample_size = extract_mcmc_chain_from_pymc3_trace(
        trace,pymc_var_names,ndraws,nburn,njobs)
    
    if get_map:
        map_sample = extract_map_sample_from_pymc3_dict(
            map_sample_dict,pymc_var_names)
    else:
        map_samples = None
        
    return samples, effective_sample_size, map_sample
