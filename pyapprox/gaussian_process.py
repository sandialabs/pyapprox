#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
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


from pyapprox import get_polynomial_from_variable, \
    get_univariate_quadrature_rules_from_pce
from pyapprox.utilities import cartesian_product, outer_product
def integrate_gaussian_process(gp,variable):
    from sklearn.gaussian_process.kernels import Matern, RBF
    kernel = gp.kernel_
    if (not type(kernel)==RBF and not 
        (type(kernel)==Matern and not np.isfinite(kernel.nu))):
        #squared exponential kernel
        raise Exception('Only squared exponential kernel supported')
    
    nvars = variable.num_vars()
    degrees = [100]*nvars
    pce = get_polynomial_from_variable(variable)
    indices = []
    for ii in range(pce.num_vars()):
        indices_ii = np.zeros((pce.num_vars(),degrees[ii]+1),dtype=int)
        indices_ii[ii,:]=np.arange(degrees[ii]+1,dtype=int)
        indices.append(indices_ii)
    pce.set_indices(np.hstack(indices))
    univariate_quad_rules=get_univariate_quadrature_rules_from_pce(
        pce,degrees)
    
    
    length_scale = np.atleast_1d(kernel.length_scale)
    T,U=1,1
    from scipy.spatial.distance import cdist
    quad_points = []
    for ii in range(nvars):
        xx,ww=univariate_quad_rules[ii](degrees[ii]+1)
        jj = pce.basis_type_index_map[ii]
        loc,scale = pce.var_trans.scale_parameters[ii,:]
        xx = xx*scale+loc
        quad_points.append(xx)
        dists = cdist(
            xx[:,np.newaxis]/length_scale[ii],
            gp.X_train_[:,ii:ii+1]/length_scale[ii],
            metric='sqeuclidean')
        K = np.exp(-.5*dists)
        T*=ww.dot(K)
        
        XX = cartesian_product([xx]*2)
        WW = outer_product([ww]*2)
        dists = (XX[0,:].T/length_scale[ii]-XX[1,:].T/length_scale[ii])**2
        K = np.exp(-.5*dists)
        U*=WW.dot(K)

    expected_random_mean = T.dot(gp.alpha_)
    from scipy.linalg import solve_triangular
    if gp._K_inv is None:
        L_inv = solve_triangular(gp.L_.T,np.eye(gp.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
    else:
        K_inv = gp._K_inv
    variance_random_mean = U-T.dot(K_inv).dot(T.T)
    assert np.all(gp._y_train_mean==0)
    return expected_random_mean, variance_random_mean
