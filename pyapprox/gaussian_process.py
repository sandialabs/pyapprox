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

    def predict_random_realization(self,samples):
        mean,cov = self(samples,return_cov=True)
        #add small number to diagonal to ensure covariance matrix is positive definite
        cov[np.arange(cov.shape[0]),np.arange(cov.shape[0])]+=1e-14
        L = np.linalg.cholesky(cov)
        return mean + L.dot(np.random.normal(0,1,mean.shape))
        
        

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
def integrate_gaussian_process(gp,variable,return_full=False):
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
        variable,return_full)

def integrate_gaussian_process_squared_exponential_kernel(X_train,Y_train,K_inv,
                                                          length_scale,
                                                          kernel_var,
                                                          variable,
                                                          return_full=False):
    """
    Compute

    .. math:: I = \int \eta(\rv) \rho(\rv) ;d\rv

    and 

    .. math:: \Sigma = I_2 - I^2, \qquad I_2 = \int \eta^2(\rv) \rho(\rv) ;d\rv

    where :math:`\rho(\rv)` is the joint density of independent random 
    variables and :math:`\eta(\rv)` is a Gaussian process (GP) 
    constructed with the squared exponential kernel
    
    .. math: K(x,y;L)=\sigma_K^2 \exp(-\frac{\lVert x-y\rVert_2^2}{2*L^2})

    with :math:`L` being a np.ndarray of shape (nvars) containing the 
    length scales of the covariance kernel.

    Because the GP is a random process, the expectation :math:`I` and the 
    variance :math:`\Sigma` of the GP with respect to :math:`\rv` are 
    themselves  random variables. Specifically the expectation is a Gaussian 
    random  variable with mean :math:`\mu` and variance :math:`v^2`. The 
    distribution of :math:`\Sigma` is harder to compute, but we can compute 
    its mean and variance

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

    return_full : boolean
       If true return intermediate quantities used to compute statistics. 
       This is only necessary for testing

    Returns
    -------
    expected_random_mean : float
        The mean :math:`\mu_I` of the Gaussian random variable representing the 
        expectation :math:`I`

    variance_random_mean : float
        The variance :math:`v_I^2` of the Gaussian random variable representing 
        the expectation :math:`I`

    expected_random_var : float
        The mean :math:`\mu_\Sigma` of the Gaussian random variable 
        representing the variance :math:`\Sigma`

    variance_random_var : float
        The variance :math:`v_\Sigma^2` of the Gaussian random variable 
        representing the variance :math:`\Sigma`
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
    
    lscale = np.atleast_1d(length_scale)
    T,U=1,1
    P=np.ones((X_train.shape[1],X_train.shape[1]))
    MC2=np.ones(X_train.shape[1])
    MMC3=np.ones((X_train.shape[1],X_train.shape[1]))
    CC1,C1_sq=1,1
    
    from scipy.spatial.distance import cdist
    quad_points = []
    for ii in range(nvars):
        xx2,ww2=univariate_quad_rules[ii](degrees[ii]+1)
        jj = pce.basis_type_index_map[ii]
        loc,scale = pce.var_trans.scale_parameters[jj,:]
        xx2 = xx2*scale+loc
        quad_points.append(xx2)
        dists = cdist(
            xx2[:,np.newaxis]/lscale[ii],
            X_train[ii:ii+1,:].T/lscale[ii],
            metric='sqeuclidean')
        K = np.exp(-.5*dists)
        # T in Haylock is defined without kernel_var
        T*=ww2.dot(K)

        for mm in range(X_train.shape[1]):
            for nn in range(mm,X_train.shape[1]):
                P[mm,nn]*=ww2.dot(K[:,mm]*K[:,nn])
                P[nn,mm]=P[mm,nn]

        XX2 = cartesian_product([xx2]*2)
        WW2 = outer_product([ww2]*2)
        dists = (XX2[0,:].T/lscale[ii]-XX2[1,:].T/lscale[ii])**2
        K = np.exp(-.5*dists)
        U*=WW2.dot(K)

        for mm in range(X_train.shape[1]):
            print(X_train[ii,mm])
            dists1 = (XX2[0,:]/lscale[ii]-XX2[1,:]/lscale[ii])**2
            dists2 = (XX2[1,:]/lscale[ii]-X_train[ii,mm]/lscale[ii])**2
            MC2[mm] *= np.exp(-.5*dists1-.5*dists2).dot(WW2)

        for mm in range(X_train.shape[1]):
            for nn in range(X_train.shape[1]):
                dists1 = (X_train[ii,mm]/lscale[ii]-XX2[0,:]/lscale[ii])**2
                dists2 = (X_train[ii,nn]/lscale[ii]-XX2[1,:]/lscale[ii])**2
                dists3 = (XX2[0,:]/lscale[ii]-XX2[1,:]/lscale[ii])**2
                MMC3[mm,nn]*=np.exp(-.5*dists1-.5*dists3-.5*dists2).dot(WW2)

        dists1 = (XX2[0,:]/lscale[ii]-XX2[1,:]/lscale[ii])**2
        #C1_sq *= np.exp(-.5*dists1)*np.exp(-.5*dists1).dot(WW2)
        C1_sq *= np.exp(-dists1).dot(WW2)

        xx3,ww3=univariate_quad_rules[ii](30)
        jj = pce.basis_type_index_map[ii]
        loc,scale = pce.var_trans.scale_parameters[jj,:]
        xx3 = xx3*scale+loc
        XX3 = cartesian_product([xx3]*3)
        WW3 =  outer_product([ww3]*3)
        dists1 = (XX3[0,:]-XX3[1,:])**2
        dists2 = (XX3[1,:]-XX3[2,:])**2
        CC1 *= np.exp(-.5*dists1-.5*dists2).dot(WW3)


    #K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get
    #Haylock formula
    A_inv = K_inv*kernel_var
    expected_random_mean = T.dot(A_inv.dot(Y_train))

    variance_random_mean = kernel_var*(U-T.dot(A_inv).dot(T.T))

    expected_random_var = Y_train.T.dot(A_inv.dot(P).dot(A_inv)).dot(Y_train)+kernel_var*(1-np.trace(A_inv.dot(P)))-expected_random_mean**2-variance_random_mean

    #[C]
    C=U
    #[M]
    M=T.dot(A_inv.dot(Y_train))
    #[M^2]
    M_sq = Y_train.T.dot(A_inv.dot(P).dot(A_inv)).dot(Y_train)
    #[V]
    V = (1-np.trace(A_inv.dot(P)))
    #[MC]
    A = np.linalg.inv(A_inv)
    MC = MC2.T.dot(A.dot(Y_train))
    #[MMC]
    MMC = Y_train.T.dot(A_inv).dot(MMC3).dot(A_inv).dot(Y_train)
    #[CC]
    CC = CC1
    #[C^2]
    C_sq = C1_sq

    #E[I_2^2]
    variance_random_var = 4*MMC*kernel_var + 2*C_sq*kernel_var**2+(
        M_sq+V*kernel_var)**2
    #-2E[I_2I^2]
    variance_random_var += -2*(4*M*MC*kernel_var+2*CC*kernel_var**2+M_sq*M**2+
        V*C*kernel_var**2 + M**2*V*kernel_var+M_sq*C*kernel_var)
    #E[I^4]
    variance_random_var += 3*C**2*kernel_var**2+6*M**2*C*kernel_var+M**4

    variance_random_var -= expected_random_var**2

    if not return_full:
        return expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var

    intermeadiate_quantities={'C':C,'M':M,'M_sq':M_sq,'V':V,'MC':MC,'MMC':MMC,
                              'CC':CC,'C_sq':C_sq,'MC2':MC2,'MMC3':MMC3}
    return expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var, intermeadiate_quantities
