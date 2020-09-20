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
from pyapprox.utilities import transformed_halton_sequence, \
    pivoted_cholesky_decomposition, continue_pivoted_cholesky_decomposition

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

    def num_training_samples(self):
        return self.X_train_.shape[0]


class AdaptiveGaussianProcess(GaussianProcess):
    def setup(self,func,sampler):
        self.func=func
        self.sampler=sampler

    def refine(self,num_samples):
        new_samples = self.sampler(num_samples)[0]
        new_values = self.func(new_samples)
        if hasattr(self, 'X_train_'):
            train_samples = np.hstack([self.X_train_.T, new_samples])
            train_values = np.vstack([self.y_train_, new_values])
            #print(self.kernel_.length_scale)
        else:
            train_samples, train_values = new_samples, new_values
        self.fit(train_samples, train_values)


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
        denom2,denom3 = (di+2*si**2),(di+6*si**2)
        for mm in range(ntrain_samples):
            xm = train_samples[ii,mm]
            for nn in range(mm,ntrain_samples):
                xn = train_samples[ii,nn]
                t1=2*(xm-xn)**2/di+3*(-2*mi+xm+xn)**2/denom2+(xm-xn)**2/denom3
                Pi[mm,nn] *= np.exp(-t1/6)*np.sqrt(di**2/(denom1))
                Pi[nn,mm]=Pi[mm,nn]
    return Pi

def compute_v_sq(A_inv,P):
    v_sq = (1-np.sum(A_inv*P))
    return v_sq

def compute_zeta(y,A_inv,P):
    return y.T.dot(A_inv.dot(P).dot(A_inv)).dot(y)

def compute_zeta_econ(y,A_inv_y,A_inv_P):
    return y.T.dot(A_inv_P.dot(A_inv_y))

def compute_varpi(tau,A_inv):
    return tau.dot(A_inv).dot(tau.T)

def compute_varsigma_sq(u,varpi):
    return u-varpi

def compute_varphi(A_inv,P):
    tmp = A_inv.dot(P)
    varphi = np.sum(tmp.T*tmp)
    return varphi

def compute_varphi_econ(A_inv_P):
    varphi = np.sum(A_inv_P.T*A_inv_P)
    return varphi

def compute_psi(A_inv,Pi):
    return np.sum(A_inv.T*Pi)

def compute_chi(nu,varphi,psi):
    return nu+varphi-2*psi

def compute_phi(train_vals,A_inv,Pi,P):
    return train_vals.T.dot(A_inv).dot(Pi).dot(A_inv).dot(train_vals)-train_vals.T.dot(A_inv).dot(P).dot(A_inv).dot(P).dot(A_inv).dot(train_vals)

def compute_phi_econ(A_inv_y,A_inv_P,Pi,P):
    return A_inv_y.T.dot(Pi.dot(A_inv_y))-A_inv_y.T.dot(P.dot(A_inv_P.dot(A_inv_y)))

def compute_varrho(lamda,A_inv,train_vals,P,tau):
    return lamda.T.dot(A_inv.dot(train_vals)) - tau.T.dot(A_inv.dot(P).dot(A_inv.dot(train_vals)))

def compute_varrho_econ(lamda,A_inv_y,A_inv_P,tau):
    return lamda.T.dot(A_inv_y) - tau.T.dot(A_inv_P.dot(A_inv_y))

def compute_xi(xi_1,lamda,tau,P,A_inv):
    return xi_1+tau.dot(A_inv).dot(P).dot(A_inv).dot(tau)-\
        2*lamda.dot(A_inv).dot(tau)

def compute_xi_econ(xi_1,lamda,tau,A_inv_P,A_inv_tau):
    return xi_1+tau.dot(A_inv_P.dot(A_inv_tau))-\
        2*lamda.dot(A_inv_tau)

def compute_var_of_var_term1(phi,kernel_var,chi,zeta,v_sq):
    #E[I_2^2] (term1)
    return 4*phi*kernel_var + 2*chi*kernel_var**2+(
        zeta+v_sq*kernel_var)**2

def compute_var_of_var_term2(eta,varrho,kernel_var,xi,zeta,v_sq,varsigma_sq):
    #-2E[I_2I^2] (term2)
    return 4*eta*varrho*kernel_var+2*xi*kernel_var**2+\
        zeta*varsigma_sq*kernel_var+v_sq*varsigma_sq*kernel_var**2+\
        zeta*eta**2+eta**2*v_sq*kernel_var

def compute_var_of_var_term3(varsigma_sq,kernel_var,eta,v_sq):
    #E[I^4]
    return 3*varsigma_sq**2*kernel_var**2+6*eta**2*varsigma_sq*kernel_var+eta**4

def gaussian_lamda(train_samples,delta,mu,sigma):
    nvars = train_samples.shape[0]
    lamda = 1
    for ii in range(nvars):
        xxi,si,mi,di = train_samples[ii,:],sigma[ii,0],mu[ii,0],delta[ii,0]
        denom1 = 4*si**4+6*di*si**2+di**2
        t1 = (di+4*si**2)/denom1*(mi-xxi)**2
        lamda *= di/np.sqrt(denom1)*np.exp(-t1)
    return lamda

def gaussian_xi_1(delta,sigma):
    return (delta/np.sqrt((delta+2*sigma**2)*(delta+6*sigma**2))).prod()

def variance_of_mean(kernel_var,varsigma_sq):
    return kernel_var*varsigma_sq

def mean_of_variance(zeta,v_sq,kernel_var,expected_random_mean,
                     variance_random_mean):
    return zeta+v_sq*kernel_var-expected_random_mean**2-variance_random_mean

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

def integrate_tau_P(xx_1d,ww_1d,xtr,lscale_ii):
    dist_func = partial(cdist,metric='sqeuclidean')
    dists_1d_x1_xtr=dist_func(
        xx_1d[:,np.newaxis]/lscale_ii,xtr.T/lscale_ii)
    K = np.exp(-.5*dists_1d_x1_xtr)
    tau=ww_1d.dot(K)
    P=K.T.dot(ww_1d[:,np.newaxis]*K)
    return tau, P

def integrate_u_lamda_Pi_nu(xx_1d,ww_1d,xtr,lscale_ii):
    # Get 2D tensor product quadrature rule
    xx_2d = cartesian_product([xx_1d]*2)
    ww_2d = outer_product([ww_1d]*2)
    dists_2d_x1_x2 = (xx_2d[0,:].T/lscale_ii-xx_2d[1,:].T/lscale_ii)**2
    K = np.exp(-.5*dists_2d_x1_x2)
    u = ww_2d.dot(K)

    ntrain_samples = xtr.shape[1]
    dist_func = partial(cdist,metric='sqeuclidean')
    dists_2d_x1_x2=(xx_2d[0:1,:].T/lscale_ii-xx_2d[1:2,:].T/lscale_ii)**2
    dists_2d_x2_xtr=dist_func(xx_2d[1:2,:].T/lscale_ii,xtr.T/lscale_ii)
    lamda = np.exp(-.5*dists_2d_x1_x2.T-.5*dists_2d_x2_xtr.T).dot(ww_2d)

    dists_2d_x1_xtr=dist_func(xx_2d[0:1,:].T/lscale_ii,xtr.T/lscale_ii)
    Pi = np.empty((ntrain_samples,ntrain_samples))
    for mm in range(ntrain_samples):
        dists1=dists_2d_x1_xtr[:,mm:mm+1]
        Pi[mm,:]= np.exp(
           -.5*dists1-.5*dists_2d_x1_x2-.5*dists_2d_x2_xtr).T.dot(ww_2d)
    assert np.allclose(Pi,Pi.T)

    nu = np.exp(-dists_2d_x1_x2)[:,0].dot(ww_2d)
    return u, lamda, Pi, nu

def integrate_xi_1(xx_1d,ww_1d,lscale_ii):
    xx_3d = cartesian_product([xx_1d]*3)
    ww_3d =  outer_product([ww_1d]*3)
    dists_3d_x1_x2 = (xx_3d[0,:]/lscale_ii-xx_3d[1,:]/lscale_ii)**2
    dists_3d_x2_x3 = (xx_3d[1,:]/lscale_ii-xx_3d[2,:]/lscale_ii)**2
    xi_1 = np.exp(-.5*dists_3d_x1_x2-.5*dists_3d_x2_x3).dot(ww_3d)
    return xi_1

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
    xi_1,nu=1,1
    
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
        tau_ii, P_ii=integrate_tau_P(xx_1d,ww_1d,xtr,lscale[ii])
        tau*=tau_ii; P *= P_ii
        
        u_ii,lamda_ii,Pi_ii,nu_ii = integrate_u_lamda_Pi_nu(xx_1d,ww_1d,xtr,lscale[ii])
        u *= u_ii; lamda *= lamda_ii; Pi *= Pi_ii; nu *= nu_ii
        
        xi_1 *= integrate_xi_1(xx_1d,ww_1d,lscale[ii])
    
    #K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get
    #Haylock formula
    A_inv = K_inv*kernel_var
    expected_random_mean = tau.dot(A_inv.dot(Y_train))

    varpi = compute_varpi(tau,A_inv)
    varsigma_sq = compute_varsigma_sq(u,varpi)
    variance_random_mean = variance_of_mean(kernel_var,varsigma_sq)

    A_inv_P = A_inv.dot(P)
    A_inv_y = A_inv.dot(Y_train)
    A_inv_tau = A_inv.dot(tau)
    v_sq = compute_v_sq(A_inv,P)
    #zeta = compute_zeta(Y_train,A_inv,P)
    zeta = compute_zeta_econ(Y_train,A_inv_y,A_inv_P)
    
    expected_random_var = mean_of_variance(
        zeta,v_sq,kernel_var,expected_random_mean,variance_random_mean)

    #varphi = compute_varphi(A_inv,P)
    varphi = compute_varphi_econ(A_inv_P)
    psi = compute_psi(A_inv,Pi)
    chi = compute_chi(nu,varphi,psi)
    
    eta = expected_random_mean
    #varrho = compute_varrho(lamda,A_inv,Y_train,P,tau)
    varrho = compute_varrho_econ(lamda,A_inv_y,A_inv_P,tau)
    #phi = compute_phi(Y_train,A_inv,Pi,P)
    phi = compute_phi_econ(A_inv_y,A_inv_P,Pi,P)
    #xi = compute_xi(xi_1,lamda,tau,P,A_inv)
    xi = compute_xi_econ(xi_1,lamda,tau,A_inv_P,A_inv_tau)

    term1 = compute_var_of_var_term1(phi,kernel_var,chi,zeta,v_sq)
    term2 = compute_var_of_var_term2(
        eta,varrho,kernel_var,xi,zeta,v_sq,varsigma_sq)
    term3 = compute_var_of_var_term3(varsigma_sq,kernel_var,eta,v_sq)
    variance_random_var = term1-2*term2+term3
    variance_random_var -= expected_random_var**2

    if not return_full:
        return expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var

    intermeadiate_quantities=tau,u,varpi,varsigma_sq,P,v_sq,zeta,nu,varphi,Pi,psi,chi,phi,lamda,varrho,xi_1,xi
    return expected_random_mean, variance_random_mean, expected_random_var,\
            variance_random_var, intermeadiate_quantities

class CholeskySampler(object):
    """
    Compute samples for kernel based approximation using the power-function
    method.

    Parameters
    ----------
    num_vars : integer
        The number of variables

    num_candidate_samples : integer
        The number of candidate samples from which final samples are chosen

    variables : list
        list of scipy.stats random variables used to transform candidate
        samples. If variables == None candidates will be generated on [0,1]

    max_num_samples : integer
        The maximum number of samples to be generated

    weight_function : callable
        Function used to precondition kernel with the signature

        ``weight_function(samples) -> np.ndarray (num_samples)``

        where samples is a np.ndarray (num_vars,num_samples)

    generate_random_samples : callable
        Function used to generate samples to enrich default candidate set.
        If this is not None then num_candidate_samples//2 will be created
        by this function and the other half of samples will be from a Halton
        sequence.
    """
    def __init__(self, num_vars, num_candidate_samples, variables=None,
                 generate_random_samples=None):
        self.num_vars = num_vars
        self.kernel_theta = None
        self.init_pivots = None
        self.chol_flag = None
        self.variables = variables
        self.generate_random_samples = generate_random_samples
        self.generate_candidate_samples(num_candidate_samples)
        self.set_weight_function(None)

    def generate_candidate_samples(self, num_candidate_samples):
        self.num_candidate_samples = num_candidate_samples
        if self.generate_random_samples is not None:
            num_halton_candidates = num_candidate_samples//2
            num_random_candidates = num_candidate_samples//2
        else:
            num_halton_candidates = num_candidate_samples
            num_random_candidates = 0

        if self.variables is None:
            marginal_icdfs = None
        else:
            marginal_icdfs = [v.ppf for v in self.variables]
        self.candidate_samples = transformed_halton_sequence(
            marginal_icdfs, self.num_vars, num_halton_candidates)

        if num_random_candidates > 0:
            self.candidate_samples = np.hstack((
                self.candidate_samples, generate_random_samples(
                    num_random_candidates)))

    def set_weight_function(self, weight_function):
        self.pivot_weights = None
        self.weight_function = weight_function
        if self.weight_function is not None:
            self.pivot_weights = weight_function(self.candidate_samples)
        self.weight_function_changed = True

    def set_kernel(self, kernel):
        if not hasattr(self, 'kernel') or self.kernel != kernel:
            self.kernel_changed = True
        self.kernel = kernel
        self.kernel_theta = self.kernel.theta

    def __call__(self, num_samples):
        if not hasattr(self, 'kernel'):
            raise Exception('Must call set_kernel')
        if not hasattr(self, 'weight_function'):
            raise Exception('Must call set_weight_function')

        if self.kernel_theta is None:
            assert self.kernel_changed

        if self.weight_function_changed or self.kernel_changed:
            self.Kmatrix = self.kernel(self.candidate_samples.T)
            self.L, self.pivots, error, self.chol_flag, self.diag, \
                self.init_error, self.num_completed_pivots = \
                pivoted_cholesky_decomposition(
                    self.Kmatrix, num_samples, init_pivots=self.init_pivots,
                    pivot_weights=self.pivot_weights,
                    error_on_small_tol=False, return_full=True)

            self.weight_function_changed = False
            self.kernel_changed = False
        else:
            self.L, self.pivots, self.diag, self.chol_flag, \
                self.num_completed_pivots, error = \
                continue_pivoted_cholesky_decomposition(
                    self.Kmatrix, self.L, num_samples, self.init_pivots,
                    0., False, self.pivot_weights, self.pivots, self.diag,
                    self.num_completed_pivots, self.init_error)

        if self.chol_flag == 0:
            assert self.num_completed_pivots == num_samples
        if self.init_pivots is None:
            num_prev_pivots = 0
        else:
            num_prev_pivots = self.init_pivots.shape[0]
        self.init_pivots = self.pivots[:self.num_completed_pivots].copy()

        # extract samples that were not already in sample set
        # pivots has alreay been reduced to have the size of the number of
        # samples requested
        new_samples = \
            self.candidate_samples[:, self.pivots[
                num_prev_pivots:self.num_completed_pivots]]
        return new_samples, self.chol_flag
