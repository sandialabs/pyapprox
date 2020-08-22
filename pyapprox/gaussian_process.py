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

def is_covariance_kernel(kernel,kernel_types):
    return (type(kernel) in kernel_types)

#(type(kernel)==Matern and not np.isfinite(kernel.nu)))

def extract_covariance_kernel(kernel,kernel_types):
    cov_kernel = None
    if is_covariance_kernel(kernel,kernel_types):
        return kernel
    if type(kernel) == Product or type(kernel) == Sum:
        cov_kernel = extract_covariance_kernel(kernel.k1,kernel_types)
        if cov_kernel is None:
            cov_kernel = extract_covariance_kernel(kernel.k2,kernel_types)
    return cov_kernel

from sklearn.gaussian_process.kernels import Matern, RBF, Product, Sum, \
    ConstantKernel
from pyapprox import get_polynomial_from_variable, \
    get_univariate_quadrature_rules_from_pce
from pyapprox.utilities import cartesian_product, outer_product
def integrate_gaussian_process(gp,variable):
    kernel_types = [RBF,Matern]
    kernel = extract_covariance_kernel(gp.kernel_,kernel_types)

    constant_kernel = extract_covariance_kernel(gp.kernel_,[ConstantKernel])
    if constant_kernel is not None:
        kernel_var = constant_kernel.constant_value
    else:
        kernel_var = 1
        
    if (not type(kernel)==RBF and not 
        (type(kernel)==Matern and not np.isfinite(kernel.nu))):
        #squared exponential kernel
        msg = f'GP Kernel type: {type(kernel)} '
        msg += 'Only squared exponential kernel supported'
        raise Exception(msg)

    if np.any(gp._y_train_mean!=0):
        msg = 'Mean of training data was not zero. This is not supported'
        raise Exception(msg)

    from scipy.linalg import solve_triangular
    if gp._K_inv is None:
        L_inv = solve_triangular(gp.L_.T,np.eye(gp.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
    else:
        K_inv = gp._K_inv
    
    return integrate_gaussian_process_squared_exponential_kernel(
        gp.X_train_.T,gp.y_train_,K_inv,kernel.length_scale,kernel_var,
        variable)

def integrate_gaussian_process_squared_exponential_kernel(X_train,Y_train,K_inv,
                                                          length_scale,
                                                          kernel_var,
                                                          variable):
    """
    Compute

    .. math:: I = \int \eta(\rv) \rho(\rv) ;d\rv

    where :math:`\rho(\rv)` is the joint density of independent random 
    variables and :math:`\eta(\rv)` is a Gaussian process (GP) 
    constructed with the squared exponential kernel
    
    .. math: K(x,y;L)=\sigma_K^2 \exp(-\frac{\lVert x-y\rVert_2^2}{2*L^2})

    with :math:`L` being a np.ndarray of shape (nvars) containing the 
    length scales of the covariance kernel.

    Because the GP is a random process, the expectation :math:`I` of the GP 
    with respect to :math:`\rv` is itself a random variable. Specifically the
    expectation is a Gaussian random variable with mean :math:`\mu` and variance
    :math:`v^2`.

    Parameters
    ----------
    X_train : np.ndarray (nvars,nsamples)
        The locations of the training data used to train the GP

    Y_train : np.ndarray (nvars,nsamples)
        The data values at ``X_train`` used to train the GP

    K_inv : np.ndarray (nsamples,nsamples)
        The inverse of the covariance matrix 
        :math:`K(X_train,X_train;length_scale)`

    length_scale : np.ndarray (nvars)
        The length scales :math:`L`

    kernel_var : float
        The variance :math:`\sigma_K^2` of the kernel :math:`K`

    variable : :class:`pyapprox.variable.IndependentMultivariateRandomVariable`
        A set of independent univariate random variables. The tensor-product 
        of the 1D PDFs yields the joint density :math:`\rho`

    Returns
    -------
    expected_random_mean : float
        The mean :math:`\mu` of the Gaussian random variable representing the 
        expectation :math:`I`

    variance_random_mean : float
        The variance :math:`v^2` of the Gaussian random variable representing 
        the expectation :math:`I`
    """
    
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
    
    length_scale = np.atleast_1d(length_scale)
    T,U=1,1
    P=np.ones((X_train.shape[1],X_train.shape[1]))
    from scipy.spatial.distance import cdist
    quad_points = []
    for ii in range(nvars):
        xx,ww=univariate_quad_rules[ii](degrees[ii]+1)
        jj = pce.basis_type_index_map[ii]
        loc,scale = pce.var_trans.scale_parameters[jj,:]
        xx = xx*scale+loc
        quad_points.append(xx)
        dists = cdist(
            xx[:,np.newaxis]/length_scale[ii],
            X_train[ii:ii+1,:].T/length_scale[ii],
            metric='sqeuclidean')
        K = np.exp(-.5*dists)
        # T in Haylock is defined without kernel_var
        T*=ww.dot(K)

        for mm in range(X_train.shape[1]):
            for nn in range(mm,X_train.shape[1]):
                P[mm,nn]*=ww.dot(K[:,mm]*K[:,nn])
                P[nn,mm]=P[mm,nn]
        
        XX = cartesian_product([xx]*2)
        WW = outer_product([ww]*2)
        dists = (XX[0,:].T/length_scale[ii]-XX[1,:].T/length_scale[ii])**2
        K = np.exp(-.5*dists)
        U*=WW.dot(K)

    #K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get
    #Haylock formula
    A_inv = K_inv*kernel_var
    expected_random_mean = T.dot(A_inv.dot(Y_train))

    variance_random_mean = kernel_var*(U-T.dot(A_inv).dot(T.T))

    expected_random_var = Y_train.T.dot(A_inv.dot(P).dot(A_inv)).dot(Y_train)+kernel_var*(1-np.trace(A_inv.dot(P)))-expected_random_mean**2-variance_random_mean

    return expected_random_mean, variance_random_mean, expected_random_var
