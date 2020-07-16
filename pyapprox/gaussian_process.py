#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
class GaussianProcess(GaussianProcessRegressor):       
    def fit(self,train_samples,train_values):
        """
        A light weight wrapper of sklearn GaussianProcessRegressor.fit
        function. See sklearn documentation for more info. This wrapper
        is needed because sklearn stores a unique sample in each row
        of a samples matrix whereas pyapprox uses the transpose.

        Parameters
        ----------
        samples : np.ndarray (nvars,nsamples)
            Samples at which to evaluate the GP. Sklearn requires the
            transpose of this matrix, i.e a matrix with size (nsamples,nvars)
        """

        return super().fit(train_samples.T,train_values)

    def __call__(self, samples, return_std=False, return_cov=False):
        """
        A light weight wrapper of sklearn GaussianProcessRegressor.predict
        function. See sklearn documentation for more info. This wrapper
        is needed because sklearn stores a unique sample in each row
        of a samples matrix whereas pyapprox uses the transpose.

        Parameters
        ----------
        samples : np.ndarray (nvars,nsamples)
            Samples at which to evaluate the GP. Sklearn requires the
            transpose of this matrix, i.e a matrix with size (nsamples,nvars)
        """
        return self.predict(samples.T,return_std,return_cov)

def approximate_gaussian_process(train_samples,train_vals,nu=np.inf,n_restarts_optimizer=5,verbosity=0):
    r"""
    Compute a Gaussian process approximation of a function from a fixed data 
    set.

    Parameters
    ----------
    train_samples : np.ndarray (nvars,nsamples)
        The inputs of the function used to train the approximation

    train_vals : np.ndarray (nvars,nsamples)
        The values of the function at ``train_samples``

    kernel_nu : string
        The parameter :math:`\nu` of the Matern kernel

    n_restarts_optimizer : int
        The number of local optimizeation problems solved to find the 
        GP hyper-parameters

    verbosity : integer
        Controls the amount of information printed to screen

    Returns
    -------
    gaussian_process : :class:`pyapprox.gaussian_process.GaussianProcess`
        The PCE approximation
    """
    kernel = 1*Matern(length_scale_bounds=(1e-2, 10), nu=nu)
    # optimize variance
    kernel = 1*kernel
    # optimize gp noise
    kernel += WhiteKernel(noise_level_bounds=(1e-8, 1))
    gp = GaussianProcess(kernel,n_restarts_optimizer=n_restarts_optimizer)
    gp.fit(train_samples,train_vals)
    return gp
    


