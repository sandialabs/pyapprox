#!/usr/bin/env python
import numpy as np
import copy
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, Product, Sum, \
    ConstantKernel
from pyapprox import get_polynomial_from_variable, \
    get_univariate_quadrature_rules_from_variable
from pyapprox.utilities import cartesian_product, outer_product, \
    cholesky_solve_linear_system, update_cholesky_factorization
from scipy.spatial.distance import cdist
from functools import partial
from scipy.linalg import solve_triangular
from pyapprox.utilities import transformed_halton_sequence, \
    pivoted_cholesky_decomposition, continue_pivoted_cholesky_decomposition


class GaussianProcess(GaussianProcessRegressor):
    def fit(self, train_samples, train_values):
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
        return super().fit(train_samples.T, train_values)

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
        return self.predict(samples.T, return_std, return_cov)

    def predict_random_realization(self, samples):
        mean, cov = self(samples, return_cov=True)
        # add small number to diagonal to ensure covariance matrix is
        # positive definite
        cov[np.arange(cov.shape[0]), np.arange(cov.shape[0])] += 1e-14
        L = np.linalg.cholesky(cov)
        return mean + L.dot(np.random.normal(0, 1, mean.shape))

    def num_training_samples(self):
        return self.X_train_.shape[0]

    def condition_number(self):
        return np.linalg.cond(self.L_.dot(self.L_.T))


class AdaptiveGaussianProcess(GaussianProcess):
    def setup(self, func, sampler):
        self.func = func
        self.sampler = sampler

    def refine(self, num_samples):
        new_samples = self.sampler(num_samples)[0]
        new_values = self.func(new_samples)
        assert new_values.shape[1] == 1  # must be scalar values QoI
        if hasattr(self, 'X_train_'):
            train_samples = np.hstack([self.X_train_.T, new_samples])
            train_values = np.vstack([self.y_train_, new_values])
        else:
            train_samples, train_values = new_samples, new_values
        self.fit(train_samples, train_values)


def is_covariance_kernel(kernel, kernel_types):
    return (type(kernel) in kernel_types)


def extract_covariance_kernel(kernel, kernel_types):
    cov_kernel = None
    if is_covariance_kernel(kernel, kernel_types):
        return kernel
    if type(kernel) == Product or type(kernel) == Sum:
        cov_kernel = extract_covariance_kernel(kernel.k1, kernel_types)
        if cov_kernel is None:
            cov_kernel = extract_covariance_kernel(kernel.k2, kernel_types)
    return cov_kernel


def gaussian_tau(train_samples, delta, mu, sigma):
    dists = (train_samples-mu)**2
    return np.prod(np.sqrt(delta/(delta+2*sigma**2))*np.exp(
        -(dists)/(delta+2*sigma**2)), axis=0)


def gaussian_u(delta, sigma):
    return np.sqrt(delta/(delta+4*sigma**2)).prod()


def gaussian_P(train_samples, delta, mu, sigma):
    nvars, ntrain_samples = train_samples.shape
    P = np.ones((ntrain_samples, ntrain_samples))
    for ii in range(nvars):
        si, mi, di = sigma[ii, 0], mu[ii, 0], delta[ii, 0]
        denom1 = 4*(di+4*si**2)
        term2 = np.sqrt(di/(di+4*si**2))
        for mm in range(ntrain_samples):
            xm = train_samples[ii, mm]
            xn = train_samples[ii, mm:]
            P[mm, mm:] *= np.exp(-1/(2*si**2*di)*(
                2*si**2*(xm**2+xn**2)+di*mi**2 -
                (4*si**2*(xm+xn)+2*di*mi)**2/denom1))*term2
            P[mm:, mm] = P[mm, mm:]
    return P


def gaussian_nu(delta, sigma):
    return np.sqrt(delta/(delta+8.*sigma**2)).prod()


def gaussian_Pi(train_samples, delta, mu, sigma):
    nvars, ntrain_samples = train_samples.shape
    Pi = np.ones((ntrain_samples, ntrain_samples))
    for ii in range(nvars):
        si, mi, di = sigma[ii, 0], mu[ii, 0], delta[ii, 0]
        denom1 = (12*si**4+8*di*si**2+di**2)
        denom2, denom3 = (di+2*si**2), (di+6*si**2)
        for mm in range(ntrain_samples):
            xm = train_samples[ii, mm]
            xn = train_samples[ii, mm:]
            t1 = 2*(xm-xn)**2/di+3*(-2*mi+xm+xn)**2/denom2+(xm-xn)**2/denom3
            Pi[mm, mm:] *= np.exp(-t1/6)*np.sqrt(di**2/(denom1))
            Pi[mm:, mm] = Pi[mm, mm:]
    return Pi


def compute_v_sq(A_inv, P):
    v_sq = (1-np.sum(A_inv*P))
    return v_sq


def compute_zeta(y, A_inv, P):
    return y.T.dot(A_inv.dot(P).dot(A_inv)).dot(y)


def compute_zeta_econ(y, A_inv_y, A_inv_P):
    return y.T.dot(A_inv_P.dot(A_inv_y))


def compute_varpi(tau, A_inv):
    return tau.T.dot(A_inv).dot(tau)


def compute_varsigma_sq(u, varpi):
    return u-varpi


def compute_varphi(A_inv, P):
    tmp = A_inv.dot(P)
    varphi = np.sum(tmp.T*tmp)
    return varphi


def compute_varphi_econ(A_inv_P):
    varphi = np.sum(A_inv_P.T*A_inv_P)
    return varphi


def compute_psi(A_inv, Pi):
    return np.sum(A_inv.T*Pi)


def compute_chi(nu, varphi, psi):
    return nu+varphi-2*psi


def compute_phi(train_vals, A_inv, Pi, P):
    return train_vals.T.dot(A_inv).dot(Pi).dot(A_inv).dot(train_vals) -\
        train_vals.T.dot(A_inv).dot(P).dot(A_inv).dot(P).dot(A_inv).dot(
            train_vals)


def compute_phi_econ(A_inv_y, A_inv_P, Pi, P):
    return A_inv_y.T.dot(Pi.dot(A_inv_y))-A_inv_y.T.dot(
        P.dot(A_inv_P.dot(A_inv_y)))


def compute_varrho(lamda, A_inv, train_vals, P, tau):
    return lamda.T.dot(A_inv.dot(train_vals)) - tau.T.dot(
        A_inv.dot(P).dot(A_inv.dot(train_vals)))


def compute_varrho_econ(lamda, A_inv_y, A_inv_P, tau):
    return lamda.T.dot(A_inv_y) - tau.T.dot(A_inv_P.dot(A_inv_y))


def compute_xi(xi_1, lamda, tau, P, A_inv):
    return xi_1+tau.dot(A_inv).dot(P).dot(A_inv).dot(tau) -\
        2*lamda.dot(A_inv).dot(tau)


def compute_xi_econ(xi_1, lamda, tau, A_inv_P, A_inv_tau):
    return xi_1+tau.dot(A_inv_P.dot(A_inv_tau)) -\
        2*lamda.dot(A_inv_tau)


def compute_var_of_var_term1(phi, kernel_var, chi, zeta, v_sq):
    # E[ I_2^2] (term1)
    return 4*phi*kernel_var + 2*chi*kernel_var**2+(
        zeta+v_sq*kernel_var)**2


def compute_var_of_var_term2(eta, varrho, kernel_var, xi, zeta, v_sq,
                             varsigma_sq):
    # -2E[I_2I^2] (term2)
    return 4*eta*varrho*kernel_var+2*xi*kernel_var**2 +\
        zeta*varsigma_sq*kernel_var+v_sq*varsigma_sq*kernel_var**2 +\
        zeta*eta**2+eta**2*v_sq*kernel_var


def compute_var_of_var_term3(varsigma_sq, kernel_var, eta, v_sq):
    # E[I^4]
    return 3*varsigma_sq**2*kernel_var**2+6*eta**2*varsigma_sq*kernel_var +\
        eta**4


def gaussian_lamda(train_samples, delta, mu, sigma):
    nvars = train_samples.shape[0]
    lamda = 1
    for ii in range(nvars):
        xxi, si = train_samples[ii, :], sigma[ii, 0]
        mi, di = mu[ii, 0], delta[ii, 0]
        denom1 = 4*si**4+6*di*si**2+di**2
        t1 = (di+4*si**2)/denom1*(mi-xxi)**2
        lamda *= di/np.sqrt(denom1)*np.exp(-t1)
    return lamda


def gaussian_xi_1(delta, sigma):
    return (delta/np.sqrt((delta+2*sigma**2)*(delta+6*sigma**2))).prod()


def variance_of_mean(kernel_var, varsigma_sq):
    return kernel_var*varsigma_sq


def mean_of_variance(zeta, v_sq, kernel_var, expected_random_mean,
                     variance_random_mean):
    return zeta+v_sq*kernel_var-expected_random_mean**2-variance_random_mean


def integrate_gaussian_process(gp, variable, return_full=False):
    kernel_types = [RBF, Matern]
    kernel = extract_covariance_kernel(gp.kernel_, kernel_types)

    constant_kernel = extract_covariance_kernel(gp.kernel_, [ConstantKernel])
    if constant_kernel is not None:
        kernel_var = constant_kernel.constant_value
    else:
        kernel_var = 1

    if (not type(kernel) == RBF and not
            (type(kernel) == Matern and not np.isfinite(kernel.nu))):
        # Squared exponential kernel
        msg = f'GP Kernel type: {type(kernel)} '
        msg += 'Only squared exponential kernel supported'
        raise Exception(msg)

    if np.any(gp._y_train_mean != 0):
        msg = 'Mean of training data was not zero. This is not supported'
        raise Exception(msg)

    if gp._K_inv is None:
        L_inv = solve_triangular(gp.L_.T, np.eye(gp.L_.shape[0]), lower=False)
        K_inv = L_inv.dot(L_inv.T)
    else:
        K_inv = gp._K_inv

    return integrate_gaussian_process_squared_exponential_kernel(
        gp.X_train_.T, gp.y_train_, K_inv, kernel.length_scale, kernel_var,
        variable, return_full)


def integrate_tau_P(xx_1d, ww_1d, xtr, lscale_ii):
    dist_func = partial(cdist, metric='sqeuclidean')
    dists_1d_x1_xtr = dist_func(
        xx_1d[:, np.newaxis]/lscale_ii, xtr.T/lscale_ii)
    K = np.exp(-.5*dists_1d_x1_xtr)
    tau = ww_1d.dot(K)
    P = K.T.dot(ww_1d[:, np.newaxis]*K)
    return tau, P


def integrate_u_lamda_Pi_nu(xx_1d, ww_1d, xtr, lscale_ii):
    # Get 2D tensor product quadrature rule
    xx_2d = cartesian_product([xx_1d]*2)
    ww_2d = outer_product([ww_1d]*2)
    dists_2d_x1_x2 = (xx_2d[0, :].T/lscale_ii-xx_2d[1, :].T/lscale_ii)**2
    K = np.exp(-.5*dists_2d_x1_x2)
    u = ww_2d.dot(K)

    ntrain_samples = xtr.shape[1]
    dist_func = partial(cdist, metric='sqeuclidean')
    dists_2d_x1_x2 = (xx_2d[0:1, :].T/lscale_ii-xx_2d[1:2, :].T/lscale_ii)**2
    dists_2d_x2_xtr = dist_func(xx_2d[1:2, :].T/lscale_ii, xtr.T/lscale_ii)
    lamda = np.exp(-.5*dists_2d_x1_x2.T-.5*dists_2d_x2_xtr.T).dot(ww_2d)

    dists_2d_x1_xtr = dist_func(xx_2d[0:1, :].T/lscale_ii, xtr.T/lscale_ii)
    # Pi = np.empty((ntrain_samples, ntrain_samples))
    # for mm in range(ntrain_samples):
    #     dists1=dists_2d_x1_xtr[:, mm:mm+1]
    #     Pi[mm, mm:]= np.exp(
    #         -.5*dists1-.5*dists_2d_x1_x2-.5*dists_2d_x2_xtr[:, mm:]).T.dot(
    #             ww_2d)
    #     Pi[mm:, mm] = Pi[mm, mm:]
    w = np.exp(-.5*dists_2d_x1_x2[:, 0])*ww_2d
    Pi = np.exp(-.5*dists_2d_x1_xtr).T.dot(w[:, np.newaxis]*np.exp(
        -.5*dists_2d_x2_xtr))

    nu = np.exp(-dists_2d_x1_x2)[:, 0].dot(ww_2d)
    return u, lamda, Pi, nu


def integrate_xi_1(xx_1d, ww_1d, lscale_ii):
    xx_3d = cartesian_product([xx_1d]*3)
    ww_3d = outer_product([ww_1d]*3)
    dists_3d_x1_x2 = (xx_3d[0, :]/lscale_ii-xx_3d[1, :]/lscale_ii)**2
    dists_3d_x2_x3 = (xx_3d[1, :]/lscale_ii-xx_3d[2, :]/lscale_ii)**2
    xi_1 = np.exp(-.5*dists_3d_x1_x2-.5*dists_3d_x2_x3).dot(ww_3d)
    return xi_1

def integrate_gaussian_process_squared_exponential_kernel(X_train, Y_train,
                                                          K_inv,
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
    univariate_quad_rules, pce = get_univariate_quadrature_rules_from_variable(
        variable, degrees)

    lscale = np.atleast_1d(length_scale)
    tau, u = 1, 1
    P = np.ones((ntrain_samples, ntrain_samples))
    lamda = np.ones(ntrain_samples)
    Pi = np.ones((ntrain_samples, ntrain_samples))
    xi_1, nu = 1, 1

    for ii in range(nvars):
        # TODO only compute quadrature once for each unique quadrature rules
        # But all quantities must be computed for all dimensions because
        # distances depend on either of both dimension dependent length scale
        # and training sample values
        # But others like u only needed to be computed for each unique
        # Quadrature rule and raised to the power equal to the number of
        # instances of a unique rule

        # Define distance function
        dist_func = partial(cdist, metric='sqeuclidean')

        # Training samples of ith variable
        xtr = X_train[ii:ii+1, :]

        # Get 1D quadrature rule
        xx_1d, ww_1d = univariate_quad_rules[ii](degrees[ii]+1)
        jj = pce.basis_type_index_map[ii]
        loc, scale = pce.var_trans.scale_parameters[jj, :]
        xx_1d = xx_1d*scale+loc

        # Evaluate 1D integrals
        tau_ii, P_ii = integrate_tau_P(xx_1d, ww_1d, xtr, lscale[ii])
        tau *= tau_ii
        P *= P_ii

        u_ii, lamda_ii, Pi_ii, nu_ii = integrate_u_lamda_Pi_nu(
            xx_1d, ww_1d, xtr, lscale[ii])
        u *= u_ii
        lamda *= lamda_ii
        Pi *= Pi_ii
        nu *= nu_ii
        xi_1 *= integrate_xi_1(xx_1d, ww_1d, lscale[ii])

    # K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get
    # Haylock formula
    A_inv = K_inv*kernel_var
    # No kernel_var because it cancels out because it appears in K (1/s^2)
    # and t (s^2)
    expected_random_mean = tau.dot(A_inv.dot(Y_train))

    varpi = compute_varpi(tau, A_inv)
    varsigma_sq = compute_varsigma_sq(u, varpi)
    variance_random_mean = variance_of_mean(kernel_var, varsigma_sq)

    A_inv_P = A_inv.dot(P)
    A_inv_y = A_inv.dot(Y_train)
    A_inv_tau = A_inv.dot(tau)
    v_sq = compute_v_sq(A_inv, P)
    # zeta = compute_zeta(Y_train, A_inv, P)
    zeta = compute_zeta_econ(Y_train, A_inv_y, A_inv_P)

    expected_random_var = mean_of_variance(
        zeta, v_sq, kernel_var, expected_random_mean, variance_random_mean)

    # varphi = compute_varphi(A_inv, P)
    varphi = compute_varphi_econ(A_inv_P)
    psi = compute_psi(A_inv, Pi)
    chi = compute_chi(nu, varphi, psi)

    eta = expected_random_mean
    # varrho = compute_varrho(lamda, A_inv, Y_train, P, tau)
    varrho = compute_varrho_econ(lamda, A_inv_y, A_inv_P, tau)
    # phi = compute_phi(Y_train, A_inv, Pi, P)
    phi = compute_phi_econ(A_inv_y, A_inv_P, Pi, P)
    # xi = compute_xi(xi_1, lamda, tau, P, A_inv)
    xi = compute_xi_econ(xi_1, lamda, tau, A_inv_P, A_inv_tau)

    term1 = compute_var_of_var_term1(phi, kernel_var, chi, zeta, v_sq)
    term2 = compute_var_of_var_term2(
        eta, varrho, kernel_var, xi, zeta, v_sq, varsigma_sq)
    term3 = compute_var_of_var_term3(varsigma_sq, kernel_var, eta, v_sq)
    variance_random_var = term1-2*term2+term3
    variance_random_var -= expected_random_var**2

    if not return_full:
        return expected_random_mean, variance_random_mean, \
            expected_random_var, variance_random_var

    intermeadiate_quantities = tau, u, varpi, varsigma_sq, P, v_sq, zeta, nu, \
        varphi, Pi, psi, chi, phi, lamda, varrho, xi_1, xi
    return expected_random_mean, variance_random_mean, expected_random_var,\
        variance_random_var, intermeadiate_quantities


def generate_candidate_samples(nvars, num_candidate_samples,
                               generate_random_samples, variables):
    if generate_random_samples is not None:
        num_halton_candidates = num_candidate_samples//2
        num_random_candidates = num_candidate_samples//2
    else:
        num_halton_candidates = num_candidate_samples
        num_random_candidates = 0

    if variables is None:
        marginal_icdfs = None
    else:
        # marginal_icdfs = [v.ppf for v in self.variables]
        from scipy import stats
        marginal_icdfs = []
        for v in variables.all_variables():
            lb, ub = v.interval(1)
            if not np.isfinite(lb) or not np.isfinite(ub):
                lb, ub = variable.interval(1-1e-6)
            marginal_icdfs.append(stats.uniform(lb, ub-lb).ppf)

    candidate_samples = transformed_halton_sequence(
        marginal_icdfs, nvars, num_halton_candidates)

    if num_random_candidates > 0:
        candidate_samples = np.hstack((
            candidate_samples, generate_random_samples(num_random_candidates)))

    return candidate_samples


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

    variable : :class:`pyapprox.variable.IndependentMultivariateRandomVariable`
        A set of independent univariate random variables. The tensor-product
        of the 1D PDFs yields the joint density :math:`\rho`

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

    init_pivots : np.ndarray (ninit_pivots)
        The array indices of the candidate_samples to keep
    """
    def __init__(self, num_vars, num_candidate_samples, variables=None,
                 generate_random_samples=None, init_pivots=None):
        self.nvars = num_vars
        self.kernel_theta = None
        self.chol_flag = None
        self.variables = variables
        self.generate_random_samples = generate_random_samples
        self.candidate_samples = generate_candidate_samples(
            self.nvars, num_candidate_samples, self.generate_random_samples,
            self.variables)
        self.set_weight_function(None)
        self.ntraining_samples = 0
        self.set_init_pivots(init_pivots)

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

    def set_init_pivots(self, init_pivots):
        self.init_pivots = init_pivots
        self.training_samples = \
            self.candidate_samples[:, :self.ntraining_samples]
        self.init_pivots_changed = True

    def __call__(self, num_samples):
        if not hasattr(self, 'kernel'):
            raise Exception('Must call set_kernel')
        if not hasattr(self, 'weight_function'):
            raise Exception('Must call set_weight_function')

        if self.kernel_theta is None:
            assert self.kernel_changed

        if (self.weight_function_changed or self.kernel_changed or
            self.init_pivots_changed):
            self.Kmatrix = self.kernel(self.candidate_samples.T)
            self.L, self.pivots, error, self.chol_flag, self.diag, \
                self.init_error, self.ntraining_samples = \
                pivoted_cholesky_decomposition(
                    self.Kmatrix, num_samples, init_pivots=self.init_pivots,
                    pivot_weights=self.pivot_weights,
                    error_on_small_tol=False, return_full=True)
            
            self.weight_function_changed = False
            self.kernel_changed = False
        else:
            self.L, self.pivots, self.diag, self.chol_flag, \
                self.ntraining_samples, error = \
                continue_pivoted_cholesky_decomposition(
                    self.Kmatrix, self.L, num_samples, self.init_pivots,
                    0., False, self.pivot_weights, self.pivots, self.diag,
                    self.ntraining_samples, self.init_error)

        if self.chol_flag == 0:
            assert self.ntraining_samples == num_samples
        if self.init_pivots is None:
            nprev_train_samples = 0
        else:
            nprev_train_samples = self.init_pivots.shape[0]
        self.init_pivots = self.pivots[:self.ntraining_samples].copy()

        # extract samples that were not already in sample set
        # pivots has alreay been reduced to have the size of the number of
        # samples requested
        new_samples = \
            self.candidate_samples[:, self.pivots[
                nprev_train_samples:self.ntraining_samples]]
        self.training_samples = np.hstack(
                [self.training_samples, new_samples])
        
        return new_samples, self.chol_flag


class AdaptiveCholeskyGaussianProcessFixedKernel(object):
    """
    Efficient implementation when Gaussian process kernel has no tunable
    hyper-parameters. Cholesky factor computed to generate training samples
    is reused for fiting
    """
    def __init__(self, sampler, func):
        self.sampler = sampler
        self.func = func
        self.chol_flag = 0

    def refine(self, num_samples):
        if self.chol_flag > 0:
            msg = 'Cannot refine. No well conditioned candidate samples '
            msg += 'remaining'
            print(msg)
            return
        new_samples, self.chol_flag = self.sampler(num_samples)
        new_values = self.func(new_samples)
        assert new_values.shape[0] == new_samples.shape[1]
        if hasattr(self, 'train_samples'):
            self.train_samples = np.hstack([self.train_samples, new_samples])
            self.train_values = np.vstack([self.train_values, new_values])
        else:
            self.train_samples, self.train_values = new_samples, new_values
        self.fit()

    def fit(self):
        nn = self.sampler.ntraining_samples
        chol_factor = self.sampler.L[self.sampler.pivots[:nn], :nn]
        self.coef = cholesky_solve_linear_system(chol_factor, self.train_values)

    def __call__(self, samples):
        return self.sampler.kernel(samples.T, self.train_samples.T).dot(
            self.coef)

    def num_training_samples(self):
        return self.train_samples.shape[1]

    def condition_number(self):
        nn = self.sampler.num_completed_pivots
        chol_factor = self.sampler.L[self.sampler.pivots[:nn], :nn]
        return np.linalg.cond(chol_factor.dot(chol_factor.T))


def gaussian_process_pointwise_variance(kernel, pred_samples, train_samples):
    r"""
    Compute the pointwise variance of a Gaussian process, that is

    .. math::

       K(\hat{x}, \hat{x}) - K(\hat{X}, y)^T  K(\hat{X}, \hat{X}) K(\hat{X}, y)

    for each sample :math:`\hat{x}=[\hat{x}_1,\ldots,\hat{x}_d]` and a set of
    training samples :math:`X=[x^{(1)},\ldots,x^{(N)}]`

    Parameters
    ----------
    kernel : callable
        Function with signature

        ``K(X, Y) -> np.ndarray(X.shape[0], Y.shape[0])``

        where X and Y are samples with shape (nsamples_X, nvars) and
        (nsamples_Y, nvars). Note this function accepts sample sets stored in
        the transpose of the typical pyapprox format

    train_samples : np.ndarray (nvars, ntrain_samples)
        The locations of the training data used to train the GP

    pred_samples : np.ndarray (nvars, npred_samples)
        The data values at ``X_train`` used to train the GP

    Returns
    -------
    variance : np.ndarray (npred_samples)
       The pointwise variance at each prediction sample
    """
    K_train = kernel(train_samples.T)
    # add small number to diagonal to ensure covariance matrix is
    # positive definite
    ntrain_samples = train_samples.shape[1]
    K_train[np.arange(ntrain_samples), np.arange(ntrain_samples)] += 1e-14
    k_pred = kernel(train_samples.T, pred_samples.T)
    L = np.linalg.cholesky(K_train)
    tmp = solve_triangular(L, k_pred, lower=True)
    variance = kernel.diag(pred_samples.T) - np.sum(tmp*tmp, axis=0)
    return variance


def RBF_gradient_wrt_sample_coordinates(query_sample, other_samples,
                                        length_scale):
    r"""
    Gradient of the squared exponential kernel

    .. math::

       \frac{\partial}{\partial x}K(x, Y) = -K(x, Y).T \circ D\lambda^{-1}

    for a sample :math:`x=[x_1,\ldots,x_d]^T` and
    :math:`y=[y^{(1)},\ldots,y^{(N)}]` and
    :math:`\lambda^{-1}=\mathrm{diag}([l_1^2,\ldots,l_d^2])`

    where
    :math:`D=[\hat{x}-\hat{y}^{(1)},\ldots,\hat{x}-\hat{y}^{(N)})` with
    :math:`\hat{x} = x\circ l^{-1}\circ l^{-1}`,
    :math:`\hat{y} = y\circ l^{-1}\circ l^{-1}` and

    .. math::

       K(x, y^{(i)}) =
       \exp\left(-\frac{1}{2}(x-y^{(i)})^T\Lambda^{-1}(x-y^{(i)})\right)

    Parameters
    ----------
    query_sample : np.ndarray (nvars, 1)
        The sample :math:`x`

    other_samples : np.ndarray (nvars, nother_samples)
        The samples :math:`y`

    length_scale : np.ndarray (nvars)
        The length scales `l` in each dimension

    Returns
    -------
    grad : np.ndarray (nother_samples, nvars)
        The gradient of the kernel
    """
    dists = cdist(query_sample.T/length_scale, other_samples.T/length_scale,
                  metric='sqeuclidean')
    K = np.exp(-.5 * dists)
    grad = -K.T*(
        np.tile(query_sample.T, (other_samples.shape[1], 1))-other_samples.T)/(
            np.asarray(length_scale)**2)
    return grad


def RBF_jacobian_wrt_sample_coordinates(train_samples, pred_samples,
                                        kernel, new_samples_index=0):
    r"""
    Gradient of the posterior covariance of a Gaussian process built
    using the squared exponential kernel. Let :math:`\hat{x}^{(i)}` be a
    prediction sample and :math:`x=[x^{(1)}, \ldots, x^{(N)}]` be the
    training samples then the posterior covariance is

    .. math::

       c(\hat{x}^{(i)}, x)=c(\hat{x}^{(i)}, \hat{x}^{(i)}) -
       K(\hat{x}^{(i)}, x)R K(\hat{x}^{(i)}, x)^T

    and

    .. math::

       \frac{\partial c(\hat{x}^{(i)}, x)}{\partial x_l}=
       2\left(\frac{\partial}{\partial x_l}K(\hat{x}^{(i)}, x_l)\right)
       \sum_{k=1}^N
       R[l,k]K(\hat{x}^{(i)}, x_k) - \sum_{j=1}^N\sum_{k=1}^N K(\hat{x}^{(i)},
       x_j)\frac{\partial}{\partial x_l}\left(R[j,k]\right)(\hat{x}^{(i)}, x_k)

    where :math: R = K(x, x)^{-1} and

    .. math::

       \frac{\partial R^{-1}}{\partial x_l} = R^{-1}
       \frac{\partial R}{\partial x_l} R^{-1}


    Parameters
    ----------
    train_samples : np.ndarray (nvars, ntrain_samples)
        The locations of the training data used to train the GP

    pred_samples : np.ndarray (nvars, npred_samples)
        The data values at ``X_train`` used to train the GP

    kernel : callable
        Function with signature

        ``K(X, Y) -> np.ndarray(X.shape[0], Y.shape[0])``

        where X and Y are samples with shape (nsamples_X, nvars) and
        (nsamples_Y, nvars). Note this function accepts sample sets stored in
        the transpose of the typical pyapprox format

    new_samples_index : integer
        Index in train samples that indicates the train samples for which
        derivatives will be computed. That is compute the derivatives of the
        coordinates of train_samples[:,new_sample_index:]

    Returns
    -------
    jac : np.ndarray (npred_samples, (ntrain_samples-new_sample_index)*nvars)
    """
    length_scale = kernel.length_scale
    nvars, npred_samples = pred_samples.shape
    ntrain_samples = train_samples.shape[1]
    noptimized_train_samples = ntrain_samples-new_samples_index
    k_pred_grad_all_train_points = np.zeros(
        (noptimized_train_samples, npred_samples, nvars))
    ii = 0
    for jj in range(new_samples_index, ntrain_samples):
        k_pred_grad_all_train_points[ii, :, :] = \
            RBF_gradient_wrt_sample_coordinates(
            train_samples[:, jj:jj+1], pred_samples, length_scale)
        ii += 1

    K_train = kernel(train_samples.T)
    # add small number to diagonal to ensure covariance matrix is
    # positive definite
    ntrain_samples = train_samples.shape[1]
    K_train[np.arange(ntrain_samples), np.arange(ntrain_samples)] += 1e-14
    
    K_inv = np.linalg.inv(K_train)
    k_pred = kernel(train_samples.T, pred_samples.T)
    jac = np.zeros((npred_samples, nvars*noptimized_train_samples))
    tau = k_pred.T.dot(K_inv)
    K_train_grad = np.zeros((ntrain_samples, ntrain_samples))
    ii = 0
    for jj in range(new_samples_index, ntrain_samples):
        K_train_grad_all_train_points_jj = \
            RBF_gradient_wrt_sample_coordinates(
                train_samples[:, jj:jj+1], train_samples, length_scale)
        for kk in range(nvars):
            jac[:, ii*nvars+kk] += \
                2*k_pred_grad_all_train_points[ii, :, kk]*tau[:, jj]
            K_train_grad[jj, :] = K_train_grad_all_train_points_jj[:, kk]
            K_train_grad[:, jj] = K_train_grad[jj, :]
            # The following takes advantage of sparsity of
            # tmp = tau.dot(K_train_grad)
            tmp = K_train_grad_all_train_points_jj[:, kk:kk+1].T *\
                np.tile(tau[:, jj:jj+1], (1, ntrain_samples))
            tmp[:, jj] = tau.dot(K_train_grad_all_train_points_jj[:, kk])
            jac[:, ii*nvars+kk] -= np.sum(tmp*tau, axis=1)
            # Reset to zero
            K_train_grad[jj, :] = 0
            K_train_grad[:, jj] = 0
        ii += 1
    jac *= -1
    return jac


class IVARSampler(object):
    """
    Parameters
    ----------
    num_vars : integer
        The number of dimensions

    nmonte_carlo_samples : integer
        The number of samples used to compute the sample based estimate
        of the integrated variance (IVAR)

    ncandidate_samples : integer
        The number of samples used by the greedy downselection procedure
        used to determine the initial guess (set of points) for the gradient 
        based optimization
    """
    def __init__(self, num_vars, nmonte_carlo_samples,
                 ncandidate_samples, generate_random_samples, variables=None,
                 greedy_method='givar'):
        self.nvars = num_vars
        self.nmonte_carlo_samples = nmonte_carlo_samples

        if greedy_method == 'chol':
            self.pred_samples = generate_random_samples(
                self.nmonte_carlo_samples)
            self.greedy_sampler = CholeskySampler(
                self.nvars, ncandidate_samples, variables,
                generate_random_samples=generate_random_samples)
        elif greedy_method == 'givar':
            self.greedy_sampler = GreedyIVARSampler(
                    self.nvars, nmonte_carlo_samples, ncandidate_samples,
                generate_random_samples, variables)
            self.pred_samples = self.greedy_sampler.pred_samples
        else:
            msg = f'Incorrect greedy_method {greedy_method}'
            raise Exception(msg)

        self.ntraining_samples = 0
        self.training_samples = np.empty((num_vars, self.ntraining_samples))

        self.nsamples_requested = []
        self.set_optimization_options(
            {'gtol':1e-3, 'ftol':0, 'disp':False, 'iprint':0})

    def objective(self, new_train_samples_flat):
        train_samples = np.hstack(
            [self.training_samples,
             new_train_samples_flat.reshape(
                 (self.nvars, new_train_samples_flat.shape[0]//self.nvars),
                  order='F')])
        val = gaussian_process_pointwise_variance(
            self.greedy_sampler.kernel, self.pred_samples,
            train_samples).mean()
        # print('f',val)
        return val

    def objective_gradient(self, new_train_samples_flat):
        train_samples = np.hstack(
            [self.training_samples,
             new_train_samples_flat.reshape(
                 (self.nvars, new_train_samples_flat.shape[0]//self.nvars),
                  order='F')])
        new_samples_index = self.training_samples.shape[1]
        return RBF_jacobian_wrt_sample_coordinates(
            train_samples, self.pred_samples, self.greedy_sampler.kernel,
            new_samples_index).mean(axis=0)

    def set_weight_function(self, weight_function):
        self.greedy_sampler.set_weight_function(weight_function)
        
    def set_kernel(self, kernel):
        if (not type(kernel) == RBF and not
            (type(kernel) == Matern and not np.isfinite(kernel.nu))):
            # TODO: To deal with sum kernel with noise, need to ammend gradient
            # computation which currently assumes no noise
            msg = f'GP Kernel type: {type(kernel)} '
            msg += 'Only squared exponential kernel supported'
            raise Exception(msg)

        self.greedy_sampler.set_kernel(copy.deepcopy(kernel))

    def set_optimization_options(self, opts):
        self.optim_opts = opts.copy()

    def set_bounds(self, nsamples):
        if self.greedy_sampler.variables is None:
            lbs, ubs = np.zeros(self.nvars), np.ones(self.nvars)
        else:
            variables = self.greedy_sampler.variables.all_variables()
            lbs = [v.interval(1)[0] for v in variables]
            ubs = [v.interval(1)[1] for v in variables]
        lbs = np.repeat(lbs, nsamples)
        ubs = np.repeat(ubs, nsamples)
        self.bounds = Bounds(lbs,ubs)

    def __call__(self, nsamples):
        self.nsamples_requested.append(nsamples)

        # Remove previous training samples from candidate set to prevent
        # adding them twice
        candidate_samples = self.greedy_sampler.candidate_samples
        if len(self.nsamples_requested) > 1:
            candidate_samples = candidate_samples[
                :, self.nsamples_requested[-2]:]

        # Add previous optimized sample set to candidate samples. This could
        # potentially add a candidate twice if the optimization picks some
        # of the original candidate samples chosen by
        # greedy_sampler.generate_samples, but this is unlikely. If it does
        # happen these points will never be chosen by the cholesky algorithm
        self.greedy_sampler.candidate_samples = np.hstack([
            self.training_samples.copy(), candidate_samples])

        # Make sure greedy_sampler chooses self.training_samples
        # only used if greedy_sampler is a Choleskysampler.
        # self.greedy_sampler.candidate_samples must be called before this
        # function call
        self.greedy_sampler.set_init_pivots(np.arange(self.ntraining_samples))

        # Get the initial guess for new samples to add.
        # Note the Greedy sampler will return only new samples not in
        # self.training_samples
        self.init_guess, chol_flag = self.greedy_sampler(nsamples)
        assert chol_flag == 0

        self.set_bounds(nsamples-self.ntraining_samples)

        init_guess = self.init_guess.flatten(order='F')
        # Optimize the locations of only the new training samples
        res = minimize(self.objective, init_guess, jac=self.objective_gradient,
                       method='L-BFGS-B', options=self.optim_opts,
                       bounds=self.bounds)

        new_samples = res.x.reshape(
            (self.nvars,res.x.shape[0]//self.nvars), order='F')
        self.training_samples = np.hstack([self.training_samples, new_samples])
        self.ntraining_samples = self.training_samples.shape[1]

        return new_samples, 0


class GreedyIVARSampler(object):
    """
    Parameters
    ----------
    num_vars : integer
        The number of dimensions

    nmonte_carlo_samples : integer
        The number of samples used to compute the sample based estimate
        of the integrated variance (IVAR)

    ncandidate_samples : integer
        The number of samples used by the greedy downselection procedure
        used to determine the initial guess (set of points) for the gradient 
        based optimization
    """
    def __init__(self, num_vars, nmonte_carlo_samples,
                 ncandidate_samples, generate_random_samples, variables=None,
                 use_gauss_quadrature=False):
        self.nvars = num_vars
        self.nmonte_carlo_samples = nmonte_carlo_samples
        self.variables = variables
        self.ntraining_samples = 0
        self.training_samples = np.empty((num_vars, self.ntraining_samples))
        
        self.use_gauss_quadrature = use_gauss_quadrature
        if self.use_gauss_quadrature is False:
            self.pred_samples = generate_random_samples(
                self.nmonte_carlo_samples)

        self.candidate_samples = generate_candidate_samples(
            self.nvars, ncandidate_samples, generate_random_samples,
            self.variables)
        self.nsamples_requested = []
        self.pivots = []

        self.use_gauss_quadrature = use_gauss_quadrature
        self.L = np.zeros((0, 0))

        self.econ = True
        if self.econ:
            self.y_1 = np.zeros((0))
            self.candidate_y_2 = np.empty(ncandidate_samples)

    def monte_carlo_objective(self, new_sample_index):
        train_samples = np.hstack(
            [self.training_samples,
             self.candidate_samples[:, new_sample_index:new_sample_index+1]])
        return gaussian_process_pointwise_variance(
            self.kernel, self.pred_samples,
            train_samples).mean()

    def precompute(self):
        nvars = self.variables.num_vars()
        length_scale = self.kernel.length_scale
        if np.isscalar(length_scale):
            length_scale = [length_scale]*nvars
            degrees = [self.nmonte_carlo_samples]*nvars
            univariate_quad_rules, pce = \
                get_univariate_quadrature_rules_from_variable(
                    self.variables, degrees)
            dist_func = partial(cdist, metric='sqeuclidean')
            self.tau = 1
        for ii in range(self.nvars):
            # Get 1D quadrature rule
            xx_1d, ww_1d = univariate_quad_rules[ii](degrees[ii]+1)
            jj = pce.basis_type_index_map[ii]
            loc, scale = pce.var_trans.scale_parameters[jj, :]
            xx_1d = xx_1d*scale+loc
            
            # Training samples of ith variable
            xtr = self.candidate_samples[ii:ii+1, :]
            lscale_ii = length_scale[ii]
            dists_1d_x1_xtr = dist_func(
                xx_1d[:, np.newaxis]/lscale_ii, xtr.T/lscale_ii)
            K = np.exp(-.5*dists_1d_x1_xtr)
            self.tau *= ww_1d.dot(K)

        self.A = self.kernel(self.candidate_samples.T,self.candidate_samples.T)

    def quadrature_objective(self, new_sample_index):
        # A can be updated rather than recomputed from scratch as below
        # train_samples = np.hstack(
        #    [self.training_samples,
        #     self.candidate_samples[:, new_sample_index:new_sample_index+1]])
        # A = self.kernel(train_samples.T, train_samples.T)
        if not self.econ:
            indices = np.concatenate(
                [self.pivots, [new_sample_index]]).astype(int)
            A = self.A[np.ix_(indices, indices)]
            L = np.linalg.cholesky(A)
            tau = self.tau[indices]
            return -tau.dot(cholesky_solve_linear_system(L, tau))
        
        if self.L.shape[0] == 0:
            indices = np.concatenate(
                [self.pivots, [new_sample_index]]).astype(int)
            A = self.A[np.ix_(indices, indices)]
            L1 = np.linalg.cholesky(A)
            L = np.sqrt(self.A[new_sample_index, new_sample_index])
            tau = self.tau[indices]
            assert np.allclose(-self.tau[new_sample_index]**2/self.A[
                new_sample_index, new_sample_index],-tau.dot(cholesky_solve_linear_system(L1, tau)))
            assert np.allclose(solve_triangular(L1,tau),self.tau[new_sample_index]/L)
            self.candidate_y_2[new_sample_index] = self.tau[new_sample_index]/L
            return -self.tau[new_sample_index]**2/self.A[
                new_sample_index, new_sample_index]

        A_12 = self.A[self.pivots, new_sample_index:new_sample_index+1]
        L_12 = solve_triangular(self.L, A_12, lower=True)
        L_22 = np.sqrt(
            self.A[new_sample_index, new_sample_index] - L_12.T.dot(L_12))
        #--------------------
        self.L_up[-1, :len(self.pivots)] = L_12.T
        self.L_up[-1,-1] = L_22
        #assert np.allclose(self.L_up, L), (self.L_up -L)
        L = self.L_up
        indices = np.concatenate(
            [self.pivots, [new_sample_index]]).astype(int)
        tau = self.tau[indices]
        # no need to compute L_up once below is working
        #--------------------
        
        y_2 = (self.tau[new_sample_index]-L_12.T.dot(self.y_1))/L_22[0,0]
        self.candidate_y_2[new_sample_index] = y_2
        assert np.allclose(solve_triangular(L,tau,lower=True)[-1],y_2)
        
        z_2 = y_2/L_22[0,0]
        assert np.allclose(cholesky_solve_linear_system(L, tau)[-1],z_2)
        val = -(self.prev_best_obj + self.tau[new_sample_index]*z_2)
        print(val, -tau.dot(cholesky_solve_linear_system(L, tau)))
        assert np.allclose(val, -tau.dot(cholesky_solve_linear_system(L, tau)))
        return val
        
    def set_kernel(self, kernel):
        if (not type(kernel) == RBF and not
            (type(kernel) == Matern and not np.isfinite(kernel.nu))):
            # TODO: To deal with sum kernel with noise, need to ammend gradient
            # computation which currently assumes no noise
            msg = f'GP Kernel type: {type(kernel)} '
            msg += 'Only squared exponential kernel supported'
            raise Exception(msg)
        self.kernel = kernel

        if self.use_gauss_quadrature:
            self.precompute()
            self.objective = self.quadrature_objective
        else:
            self.objective = self.monte_carlo_objective

    def set_init_pivots(self, init_pivots):
        self.pivots = list(init_pivots)
        self.training_samples = self.candidate_samples[:,init_pivots]

    def __call__(self, nsamples):
        if not hasattr(self, 'kernel'):
            raise Exception('Must call set_kernel')
        
        self.nsamples_requested.append(nsamples)
        ntraining_samples = self.ntraining_samples
        for nn in range(ntraining_samples, nsamples):
            obj_vals = np.inf*np.ones(self.candidate_samples.shape[1])
            if self.econ:
                self.L_up = np.zeros((self.L.shape[0]+1, self.L.shape[0]+1))
                self.L_up[:self.L.shape[0], :self.L.shape[0]] = self.L.copy()
            for mm in range(self.candidate_samples.shape[1]):
                if mm not in self.pivots:
                    obj_vals[mm] = self.objective(mm)

            pivot = np.argmin(obj_vals)

            if self.econ:
                self.L = update_cholesky_factorization(
                    self.L,
                    self.A[self.pivots, pivot:pivot+1],
                    np.atleast_2d(self.A[pivot, pivot]))
                self.prev_best_obj = obj_vals[pivot]
                self.y_1 = np.concatenate(
                    [self.y_1, [self.candidate_y_2[pivot]]])

            self.pivots.append(pivot)
            new_sample = self.candidate_samples[:, pivot:pivot+1]
            self.training_samples = np.hstack(
                [self.training_samples,
                 self.candidate_samples[:, pivot:pivot+1]])


            #print(f'Number of points generated {nn+1}')

        new_samples = self.training_samples[:,ntraining_samples:]
        self.ntraining_samples = self.training_samples.shape[1]

        return new_samples, 0
