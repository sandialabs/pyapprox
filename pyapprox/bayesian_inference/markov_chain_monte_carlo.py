import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
from scipy.optimize import approx_fprime

class GaussianLogLike(object):
    r"""
    A Gaussian log-likelihood function for a model with parameters given in 
    sample
    """
    def __init__(self,model,data,noise_covar):
        r"""
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        model : callable
            The model relating the data and noise 

        data : np.ndarray (nobs)
            The "observed" data

        noise_covar : float
            The noise covariance
        """
        self.model=model
        self.data=data
        assert self.data.ndim==1
        self.ndata = data.shape[0]
        self.noise_covar_inv = self.noise_covariance_inverse(noise_covar)

    def noise_covariance_inverse(self,noise_covar):
        if np.isscalar(noise_covar):
            inv_covar = 1/noise_covar
        elif noise_covar.ndim==1:
            assert noise_covar.shape[0]==self.data.shape[0]
            inv_covar = 1/noise_covar
        elif noise_covar.ndim==2:
            assert noise_covar.shape==[self.ndata,self.ndata]
            inv_covar = np.linalg.inv(noise_covar)
        return inv_covar

    # def noise_covariance_determinant(self, noise_covar):
    #     r"""The determinant is only necessary in log likelihood if the noise 
    #     covariance has a hyper-parameter which is being inferred which is
    #     not currently supported"""
    #     if np.isscalar(noise_covar):
    #         determinant = noise_covar**self.ndata
    #     elif noise_covar.ndim==1:
    #         determinant = np.prod(noise_covar)
    #     else:
    #         determinant = np.linalg.det(noise_covar)
    #     return determinant

    def __call__(self,samples):
        model_vals = self.model(samples)
        assert model_vals.ndim==2
        assert model_vals.shape[1]==self.ndata
        vals = np.empty((model_vals.shape[0],1))
        for ii in range(model_vals.shape[0]):
            residual = self.data - model_vals[ii,:]
            if np.isscalar(self.noise_covar_inv) or self.noise_covar_inv.ndim==1:
                vals[ii] = (residual.T*self.noise_covar_inv).dot(residual)
            else:
                vals[ii] = residual.T.dot(self.noise_covar_inv).dot(residual)
        vals *= -0.5
        return vals

class LogLike(tt.Op):
    r"""
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

    def __init__(self, loglike, loglike_grad=None):
        r"""
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined

        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        super().__init__(loglike)
        
        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood,loglike_grad)

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        samples, = inputs # important
        return [g[0]*self.logpgrad(samples)]

class LogLikeGrad(tt.Op):

    r"""
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, loglike_grad=None):
        r"""
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """
        self.likelihood = loglike
        self.likelihood_grad = loglike_grad

    def perform(self, node, inputs, outputs):
        samples,=inputs
        
        # calculate gradients
        if self.likelihood_grad is None:
            # define version of likelihood function to pass to
            # derivative function
            def lnlike(values):
                return self.likelihood(values)
            grads = approx_fprime(samples,lnlike,2*np.sqrt(np.finfo(float).eps))
        else:
            grads = self.likelihood_grad(samples)
        outputs[0][0] = grads

def extract_mcmc_chain_from_pymc3_trace(trace,var_names,ndraws,nburn,njobs):
    nvars = len(var_names)
    samples = np.empty((nvars,(ndraws-nburn)*njobs))
    effective_sample_size = -np.ones(nvars)
    for ii in range(nvars):
        samples[ii,:]=trace.get_values(
            var_names[ii],burn=nburn,chains=np.arange(njobs))
        try:
            effective_sample_size[ii]=pm.ess(trace)[var_names[ii]].values
        except:
            print('could not compute ess. likely issue with theano')


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
        algorithm='nuts',get_map=False,print_summary=False,loglike_grad=None,
        seed=None):
    r"""
    Draw samples from the posterior distribution using Markov Chain Monte Carlo
    for data that satisfies

    .. math:: y=f(z)+\epsilon

    where :math:`y` is a vector of observations, :math:`z` are the parameters of a function which are to be inferred, and :math:`\epsilon` is Gaussian noise.

    Parameters
    ----------
    loglike : pyapprox.bayesian_inference.markov_chain_monte_carlo.GaussianLogLike
        A log-likelihood function associated with a Gaussian error model

    variables : pya.IndependentMultivariateRandomVariable
        Object containing information of the joint density of the inputs z.
        This is used to generate random samples from this join density

    ndraws : integer
        The number of posterior samples

    nburn : integer
        The number of samples to discard during initialization

    njobs : integer
        The number of prallel chains

    algorithm : string
        The MCMC algorithm should be one of

        - 'nuts'
        - 'metropolis'
        - 'smc'

    get_map : boolean
        If true return the MAP
    
    print_summary : boolean
        If true print summary statistics about the posterior samples

    loglike_grad : callable
        Function with signature
      
       ``loglikegrad(z) -> np.ndarray (nvars)``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples

    random_seed : int or list of ints
        A list is accepted if ``cores`` is greater than one. PyMC3 does not 
        produce consistent results by setting numpy.random.seed instead
        seed must be passed in
    """
    
    # create our Op
    if algorithm!='nuts':
        logl = LogLike(loglike)
    else:
        logl = LogLikeWithGrad(loglike,loglike_grad)

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
                              start=None,cores=njobs,step=step,
                              compute_convergence_checks=False,random_seed=seed)
            # compute_convergence_checks=False avoids bugs in theano
            
        if print_summary:
            try:
                print(pm.summary(trace))
            except:
                print('could not print summary. likely issue with theano')
        
        samples, effective_sample_size = extract_mcmc_chain_from_pymc3_trace(
            trace,pymc_var_names,ndraws,nburn,njobs)
    
        if get_map:
            map_sample = extract_map_sample_from_pymc3_dict(
                map_sample_dict,pymc_var_names)
        else:
            map_samples = None
        
    return samples, effective_sample_size, map_sample

class PYMC3LogLikeWrapper():
    r"""
    Turn pyapprox model in to one which can be used by PYMC3.
    Main difference is that PYMC3 often passes 1d arrays where as
    Pyapprox assumes 2d arrays.
    """
    def __init__(self,loglike,loglike_grad=None):
        self.loglike=loglike
        self.loglike_grad=loglike_grad

    def __call__(self,x):
        if x.ndim==1:
            xr = x[:,np.newaxis]
        else:
            xr=x
        vals = self.loglike(xr)
        return vals.squeeze()

    def gradient(self,x):
        if x.ndim==1:
            xr = x[:,np.newaxis]
        else:
            xr=x
        return self.loglike_grad(xr).squeeze()
