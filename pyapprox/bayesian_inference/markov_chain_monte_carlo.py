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
    def __init__(self,model,data,sigma):
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

        sigma : float
            The noise standard deviation that our function requires.
        """
        self.model=model
        self.data=data
        self.sigma=sigma

    def __call__(self,samples):
        model_vals = self.model(samples)
        return -(0.5/self.sigma**2)*np.sum((self.data - model_vals)**2)

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
