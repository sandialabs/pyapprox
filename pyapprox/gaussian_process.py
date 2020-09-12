#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, Product, Sum, \
    ConstantKernel
from pyapprox import get_polynomial_from_variable, \
    get_univariate_quadrature_rules_from_pce
from pyapprox.utilities import cartesian_product, outer_product
from scipy.spatial.distance import cdist
from functools import partial
from scipy.linalg import solve_triangular

class GaussianProcess(GaussianProcessRegressor):       
    def fit(self,train_samples,train_values):
        r"""
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
        r"""
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

def gaussian_tau(train_samples,delta,mu,sigma):
    dists = (train_samples-mu)**2
    return np.prod(np.sqrt(delta/(delta+2*sigma**2))*np.exp(
        -(dists)/(delta+2*sigma**2)),axis=0)

def gaussian_u(delta,sigma):
    return np.sqrt(delta/(delta+4*sigma**2)).prod()

def gaussian_sigma_sq(delta,sigma,tau,A_inv):
    return np.sqrt(delta/(delta+4*sigma**2)).prod()-tau.dot(A_inv).dot(tau)

def gaussian_P(train_samples,delta,mu,sigma):
    nvars,ntrain_samples = train_samples.shape
    P=np.ones((ntrain_samples,ntrain_samples))
    for ii in range(nvars):
        si,mi,di = sigma[ii,0],mu[ii,0],delta[ii,0]
        denom1 = 4*(di+4*si**2)
        term2 = np.sqrt(di/(di+4*si**2))
        for mm in range(ntrain_samples):
            xm = train_samples[ii,mm]
            for nn in range(mm,ntrain_samples):
                xn = train_samples[ii,nn]
                P[mm,nn]*=np.exp(-1/(2*si**2*di)*(2*si**2*(xm**2+xn**2)+di*mi**2-(4*si**2*(xm+xn)+2*di*mi)**2/denom1))*term2
                P[nn,mm]=P[mm,nn]
    return P

def gaussian_nu(delta,sigma):
    return np.sqrt(delta/(delta+8.*sigma**2)).prod()

def gaussian_Pi(train_samples,delta,mu,sigma):
    nvars,ntrain_samples = train_samples.shape
    Pi=np.ones((ntrain_samples,ntrain_samples))
    for ii in range(nvars):
        si,mi,di = sigma[ii,0],mu[ii,0],delta[ii,0]
        denom1 = (12*si**4+8*di*si**2+di**2)
        denom2 = (di+4*si**2)*di
        for mm in range(ntrain_samples):
            xm = train_samples[ii,mm]
            for nn in range(mm,ntrain_samples):
                xn = train_samples[ii,nn]
                t1 = (-32*xm*si**6*xn+20*mi*di**2*si**2+2*mi*di**3)/denom1/denom2
                t2 = (-8*xm*si**4*xn*di+48*mi**2*di*si**4)/denom1/denom2
                t3 = (xm**2+xn**2)*(28*si**4*di+10*si**2*di**2+16*si**6+di**3)/denom1/denom2
                t4 = -(xm+xn)*(48*si**4*mi*si+20*si**2*mi*di**2+2*di**3*mi)/denom1/denom2
                Pi[mm,nn] = np.exp(-t1-t2-t3-t4)*np.sqrt(di/(denom1))
                Pi[nn,mm]=Pi[mm,nn]
    return Pi

def compute_v_sq(A_inv,P):
    return (1-np.trace(A_inv.dot(P)))

def compute_zeta(y,A_inv,P):
    return y.T.dot(A_inv.dot(P).dot(A_inv)).dot(y)

def compute_rho(A_inv,P):
    tmp = A_inv.dot(P)
    rho = np.sum(tmp.T*tmp)
    return rho

def compute_psi(A_inv,Pi):
    return np.sum(A_inv.T*Pi)

def compute_chi(nu,rho,psi):
    return nu+rho-2*psi

def compute_phi(train_vals,A_inv,Pi,P):
        return train_vals.T.dot(A_inv).dot(Pi).dot(A_inv).dot(train_vals)-train_vals.T.dot(A_inv).dot(P).dot(A_inv).dot(P).dot(A_inv).dot(train_vals)

def compute_varrho(lamda,A_inv,train_vals,P,tau):
    #TODO reduce redundant computations by computing once and then storing
    return lamda.T.dot(A_inv.dot(train_vals)) - tau.T.dot(A_inv.dot(P).dot(A_inv.dot(train_vals)))

def variance_of_mean(kernel_var,sigma_sq):
    return kernel_var*sigma_sq

def mean_of_variance(zeta,v_sq,expected_random_mean,variance_random_mean):
    return zeta+v_sq-expected_random_mean**2-variance_random_mean

def variance_of_variance_gaussian_CC1(delta,sigma):
    return (delta/np.sqrt((delta+2*sigma**2)*(delta+6*sigma**2))).prod()

def variance_of_variance_gaussian_CC(delta,sigma,T,P,CC1,lamda,A_inv):
    return CC1+T.dot(A_inv).dot(P).dot(A_inv).dot(T)-2*lamda.dot(A_inv).dot(T)

def variance_of_variance_gaussian_lamda(train_samples,delta,mu,sigma):
    nvars = train_samples.shape[0]
    lamda = 1
    for ii in range(nvars):
        xxi,si,mi,di = train_samples[ii,:],sigma[ii,0],mu[ii,0],delta[ii,0]
        numer1 = (xxi**2+mi**2)
        denom1,denom2 = 4*si**2+6*di*si**2+di**2,di+2*si**2
        lamda *= np.exp(-((8*si**4+6*si**2*di)*numer1)/(denom1*denom2))
        lamda *= np.exp(-(numer1*di**2-16*xxi*si**4*mi-12*xxi*si**2*mi*di-2*xxi*si**2*mi)/(denom1*denom2))
        lamda *= np.sqrt(di/denom2)*np.sqrt(di*denom2/denom1)
    return lamda

def variance_of_variance_gaussian_Pi(train_samples,delta,mu,sigma):
    nvars,ntrain_samples = train_samples.shape
    Pi=np.ones((ntrain_samples,ntrain_samples))
    for ii in range(nvars):
        si,mi,di = sigma[ii,0],mu[ii,0],delta[ii,0]
        denom1,denom2 = (12*si**4+8*di*si**2+di**2),di*(di+4*si)
        for mm in range(ntrain_samples):
            xm = train_samples[ii,mm]
            for nn in range(ntrain_samples):
                zn = train_samples[ii,nn]
                Pi[mm,nn]*=np.exp(-(32*xm*si**6*zn+20*mi*di**2*si**2+2*mi*di**3)/(denom1*denom2))
                Pi[mm,nn]*=np.exp(-(8*xm*si**4*zn*di+48*mi**2*di*si**4)/(denom1*denom2))
                Pi[mm,nn]*=np.exp(-((xm**2+zn**2)*(28*si**4*di+10*si**2*di**2+16*si**6+di**3))/(denom1*denom2))
                Pi[mm,nn]*=np.exp(-(-(xm+zn)*(48*si**4*mi*si+20*si**2*di**2*mi+2*di**3*mi))/(denom1*denom2))*np.sqrt(di**2/denom1**2)
    return Pi
        

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
    r"""
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
    ntrain_samples = X_train.shape[1]
    nvars = variable.num_vars()
    degrees = [50]*nvars
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
    tau,u=1,1
    P=np.ones((ntrain_samples,ntrain_samples))
    lamda=np.ones(ntrain_samples)
    Pi=np.ones((ntrain_samples,ntrain_samples))
    CC1,nu=1,1
    
    for ii in range(nvars):
        #TODO only compute quadrature once for each unique quadrature rules
        #But all quantities must be computed for all dimensions because
        #distances depend on either of both dimension dependent length scale
        #and training sample values
        #But others like u only needed to be computed for each unique
        #quadrature rule and raised to the power equal to the number of
        #instances of a unique rule

        #define distance function
        dist_func = partial(cdist,metric='sqeuclidean')
        
        # Training samples of ith variable
        xtr = X_train[ii:ii+1,:]
        
        # Get 1D quadrature rule
        xx_1d,ww_1d=univariate_quad_rules[ii](degrees[ii]+1)
        jj = pce.basis_type_index_map[ii]
        loc,scale = pce.var_trans.scale_parameters[jj,:]
        xx_1d = xx_1d*scale+loc

        # Evaluate 1D integrals
        dists_1d_x1_xtr=dist_func(
            xx_1d[:,np.newaxis]/lscale[ii],xtr.T/lscale[ii])
        K = np.exp(-.5*dists_1d_x1_xtr)
        tau*=ww_1d.dot(K)
        P*=K.T.dot(ww_1d[:,np.newaxis]*K)

        # Get 2D tensor product quadrature rule
        xx_2d = cartesian_product([xx_1d]*2)
        ww_2d = outer_product([ww_1d]*2)

        # Evaluate 2D integrals
        dists_2d_x1_x2 = (xx_2d[0,:].T/lscale[ii]-xx_2d[1,:].T/lscale[ii])**2
        K = np.exp(-.5*dists_2d_x1_x2)
        u*=ww_2d.dot(K)

        dists_2d_x1_x2=(xx_2d[0:1,:].T/lscale[ii]-xx_2d[1:2,:].T/lscale[ii])**2
        dists_2d_x2_xtr=dist_func(xx_2d[1:2,:].T/lscale[ii],xtr.T/lscale[ii])
        lamda*=np.exp(-.5*dists_2d_x1_x2.T-.5*dists_2d_x2_xtr.T).dot(ww_2d)

        dists_2d_x1_xtr=dist_func(xx_2d[0:1,:].T/lscale[ii],xtr.T/lscale[ii])
        for mm in range(ntrain_samples):
            dists1=dists_2d_x1_xtr[:,mm:mm+1]
            Pi[mm,:]*=np.exp(
                -.5*dists1-.5*dists_2d_x1_x2-.5*dists_2d_x2_xtr).T.dot(ww_2d)

        nu *= np.exp(-dists_2d_x1_x2)[:,0].dot(ww_2d)

        # Create 3D tensor product quadrature rule
        xx_3d = cartesian_product([xx_1d]*3)
        ww_3d =  outer_product([ww_1d]*3)
        dists_3d_x1_x2 = (xx_3d[0,:]/lscale[ii]-xx_3d[1,:]/lscale[ii])**2
        dists_3d_x2_x3 = (xx_3d[1,:]/lscale[ii]-xx_3d[2,:]/lscale[ii])**2
        CC1 *= np.exp(-.5*dists_3d_x1_x2-.5*dists_3d_x2_x3).dot(ww_3d)
    
    #K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get
    #Haylock formula
    A_inv = K_inv*kernel_var
    expected_random_mean = tau.dot(A_inv.dot(Y_train))

    sigma_sq = u-tau.dot(A_inv).dot(tau.T)
    variance_random_mean = variance_of_mean(kernel_var,sigma_sq)

    v_sq = compute_v_sq(A_inv,P)
    zeta = compute_zeta(Y_train,A_inv,P)
    
    expected_random_var = mean_of_variance(
        zeta,v_sq,expected_random_mean,variance_random_mean)

    rho = compute_rho(A_inv,P)
    psi = compute_psi(A_inv,Pi)
    chi = compute_chi(nu,rho,psi)
    
    mu = expected_random_mean
    varrho = compute_varrho(lamda,A_inv,Y_train,P,tau)
    phi = compute_phi(Y_train,A_inv,Pi,P)
    #[CC]
    CC = CC1+tau.dot(A_inv).dot(P).dot(A_inv).dot(tau)-2*lamda.dot(A_inv).dot(tau)

    #E[I_2^2] (term1)
    variance_random_var = 4*phi*kernel_var + 2*chi*kernel_var**2+(
        zeta+v_sq*kernel_var)**2
    #-2E[I_2I^2] (term2)
    variance_random_var += -2*(4*mu*varrho*kernel_var+2*CC*kernel_var**2+zeta*mu**2+
        v_sq*sigma_sq*kernel_var**2 + mu**2*v_sq*kernel_var+zeta*sigma_sq*kernel_var)
    #E[I^4]
    variance_random_var += 3*sigma_sq**2*kernel_var**2+6*mu**2*sigma_sq*kernel_var+mu**4
    #E[I_2-I^2]^2
    variance_random_var -= expected_random_var**2

    if not return_full:
        return expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var

    intermeadiate_quantities=tau,u,sigma_sq,P,v_sq,zeta,nu,rho,Pi,psi,phi,varrho,CC,chi,lamda,Pi,CC1
    return expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var, intermeadiate_quantities
