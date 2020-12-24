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
from pyapprox.low_discrepancy_sequences import transformed_halton_sequence
from pyapprox.utilities import pivoted_cholesky_decomposition, \
    continue_pivoted_cholesky_decomposition
from scipy.special import kv, gamma


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

    def predict_random_realization(self, samples, nugget=0):
        mean, cov = self(samples, return_cov=True)
        # add small number to diagonal to ensure covariance matrix is
        # positive definite
        cov[np.arange(cov.shape[0]), np.arange(cov.shape[0])] += nugget
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
                lb, ub = v.interval(1-1e-6)
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
        Function with signature

        ``generate_random_samples(nsamples) -> np.ndarray (nvars, nsamples)``

        used to generate samples to enrich default candidate set.
        If this is not None then num_candidate_samples//2 will be created
        by this function and the other half of samples will be from a Halton
        sequence.

    init_pivots : np.ndarray (ninit_pivots)
        The array indices of the candidate_samples to keep

    econ : boolean
        True - pivot based upon diagonal of schur complement
        False - pivot to minimize trace norm of low-rank approximation
    """

    def __init__(self, num_vars, num_candidate_samples, variables=None,
                 generate_random_samples=None, init_pivots=None,
                 nugget=0, econ=True):
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
        self.nugget = nugget
        self.econ = econ

    def add_nugget(self):
        self.Kmatrix[np.arange(self.Kmatrix.shape[0]),
                     np.arange(self.Kmatrix.shape[1])] += self.nugget

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

        if num_samples < self.training_samples.shape[1]:
            msg = f'Requesting number of samples {num_samples} which is less '
            msg += 'than number of training samples already generated '
            msg += f'{self.training_samples.shape[1]}'
            raise Exception(msg)
        if self.kernel_theta is None:
            assert self.kernel_changed

        nprev_train_samples = self.ntraining_samples

        if (self.weight_function_changed or self.kernel_changed or
                self.init_pivots_changed):
            self.Kmatrix = self.kernel(self.candidate_samples.T)
            if self.econ is False and self.pivot_weights is not None:
                weights = np.sqrt(self.pivot_weights)
                assert np.allclose(np.diag(weights).dot(self.Kmatrix.dot(
                    np.diag(weights))),
                                   weights[:, np.newaxis]*self.Kmatrix*weights)
                self.Kmatrix = weights[:, np.newaxis]*self.Kmatrix*weights
                self.pivot_weights = None

            if self.nugget > 0:
                self.add_nugget()
            self.L, self.pivots, error, self.chol_flag, self.diag, \
                self.init_error, self.ntraining_samples = \
                pivoted_cholesky_decomposition(
                    self.Kmatrix, num_samples, init_pivots=self.init_pivots,
                    pivot_weights=self.pivot_weights,
                    error_on_small_tol=False, return_full=True, econ=self.econ)

            self.weight_function_changed = False
            self.kernel_changed = False
        else:
            self.L, self.pivots, self.diag, self.chol_flag, \
                self.ntraining_samples, error = \
                continue_pivoted_cholesky_decomposition(
                    self.Kmatrix, self.L, num_samples, self.init_pivots,
                    0., False, self.pivot_weights, self.pivots, self.diag,
                    self.ntraining_samples, self.init_error, econ=self.econ)

        if self.chol_flag == 0:
            assert self.ntraining_samples == num_samples
        self.init_pivots = self.pivots[:self.ntraining_samples].copy()

        # extract samples that were not already in sample set
        # pivots has already been reduced to have the size of the number of
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

    def get_current_chol_factor(self):
        nn = self.sampler.ntraining_samples
        if type(self.sampler) == CholeskySampler:
            chol_factor = self.sampler.L[self.sampler.pivots[:nn], :nn]
        elif type(self.sampler) == GreedyIntegratedVarianceSampler:
            chol_factor = self.sampler.L[:nn, :nn]
        else:
            raise Exception()
        return chol_factor

    def fit(self):
        chol_factor = self.get_current_chol_factor()
        self.coef = cholesky_solve_linear_system(
            chol_factor, self.train_values)

    def __call__(self, samples):
        return self.sampler.kernel(samples.T, self.train_samples.T).dot(
            self.coef)

    def num_training_samples(self):
        return self.train_samples.shape[1]

    def condition_number(self):
        chol_factor = self.get_current_chol_factor()
        return np.linalg.cond(chol_factor.dot(chol_factor.T))


def gaussian_process_pointwise_variance(kernel, pred_samples, train_samples,
                                        nugget=0):
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
    K_train[np.arange(ntrain_samples), np.arange(ntrain_samples)] += nugget
    k_pred = kernel(train_samples.T, pred_samples.T)
    L = np.linalg.cholesky(K_train)
    tmp = solve_triangular(L, k_pred, lower=True)
    variance = kernel.diag(pred_samples.T) - np.sum(tmp*tmp, axis=0)
    return variance


def RBF_gradient_wrt_samples(query_sample, other_samples, length_scale):
    r"""
    Gradient of the squared exponential kernel

    .. math::

       \frac{\partial}{\partial x}K(x, Y) = -K(x, Y)^T \circ D\Lambda^{-1}

    Here :math:`x=[x_1,\ldots,x_d]^T` is a sample, 
    :math:`Y=[y^{(1)},\ldots,y^{(N)}]`
    is a set of samples  and the kernel is given by

    .. math:: 

       K(x, y^{(i)}) =
       \exp\left(-\frac{1}{2}(x-y^{(i)})^T\Lambda^{-1}(x-y^{(i)})\right)

    where
    :math:`\Lambda^{-1}=\mathrm{diag}([l_1^2,\ldots,l_d^2])`,
    :math:`D=[\tilde{x}-\tilde{y}^{(1)},\ldots,\tilde{x}-\tilde{y}^{(N)}]` and

    .. math:: 

       \tilde{x} = \left[\frac{x_1}{l_1^2}, \ldots, \frac{x_d}{l_d^2}\right],
       \qquad  \tilde{y}^{(i)} = 
       \left[\frac{y_1^{(i)}}{l_1^2},\ldots, \frac{y_d^{(i)}}{l_d^2}\right]

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


def RBF_integrated_posterior_variance_gradient_wrt_samples(
        train_samples, quad_x, quad_w,
        kernel, new_samples_index=0, nugget=0):
    r"""
    """
    nvars, ntrain_samples = train_samples.shape
    length_scale = kernel.length_scale
    if np.isscalar(length_scale):
        length_scale = np.array([length_scale]*nvars)
    K_train = kernel(train_samples.T)
    # add small number to diagonal to ensure covariance matrix is
    # positive definite
    ntrain_samples = train_samples.shape[1]
    K_train[np.arange(ntrain_samples), np.arange(ntrain_samples)] += nugget
    A_inv = np.linalg.inv(K_train)
    grad_P, P = integrate_grad_P(
        quad_x, quad_w, train_samples, length_scale)
    AinvPAinv = (A_inv.dot(P).dot(A_inv))

    noptimized_train_samples = ntrain_samples-new_samples_index
    jac = np.zeros((nvars*noptimized_train_samples))
    cnt = 0
    for kk in range(new_samples_index, ntrain_samples):
        K_train_grad_all_train_points_kk = \
            RBF_gradient_wrt_samples(
                train_samples[:, kk:kk+1], train_samples, length_scale)
        # Use the follow properties for tmp3 and tmp4
        # Do sparse matrix element wise product
        # 0 a 0   D00 D01 D02
        # a b c x D10 D11 D12
        # 0 c 0   D20 D21 D22
        # =2*(a*D01 b*D11 + c*D21)-b*D11
        #
        # Trace [RCRP] = Trace[RPRC] for symmetric matrices
        tmp3 = -2*np.sum(K_train_grad_all_train_points_kk.T*AinvPAinv[:, kk],
                         axis=1)
        tmp3 -= -K_train_grad_all_train_points_kk[kk, :]*AinvPAinv[kk, kk]
        jac[cnt*nvars:(cnt+1)*nvars] = -tmp3
        tmp4 = 2*np.sum(grad_P[kk*nvars:(kk+1)*nvars]*A_inv[:, kk], axis=1)
        tmp4 -= grad_P[kk*nvars:(kk+1)*nvars, kk]*A_inv[kk, kk]
        jac[cnt*nvars:(cnt+1)*nvars] -= tmp4
        cnt += 1
    return jac


def RBF_posterior_variance_jacobian_wrt_samples(
        train_samples, pred_samples,
        kernel, new_samples_index=0, nugget=0):
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

    where :math:`R = K(x, x)^{-1}` and

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
            RBF_gradient_wrt_samples(
            train_samples[:, jj:jj+1], pred_samples, length_scale)
        ii += 1

    K_train = kernel(train_samples.T)
    # add small number to diagonal to ensure covariance matrix is
    # positive definite
    ntrain_samples = train_samples.shape[1]
    K_train[np.arange(ntrain_samples), np.arange(ntrain_samples)] += nugget

    K_inv = np.linalg.inv(K_train)
    k_pred = kernel(train_samples.T, pred_samples.T)
    jac = np.zeros((npred_samples, nvars*noptimized_train_samples))
    tau = k_pred.T.dot(K_inv)
    #K_train_grad = np.zeros((ntrain_samples, ntrain_samples))
    ii = 0
    for jj in range(new_samples_index, ntrain_samples):
        K_train_grad_all_train_points_jj = \
            RBF_gradient_wrt_samples(
                train_samples[:, jj:jj+1], train_samples, length_scale)
        jac[:, ii*nvars:(ii+1)*nvars] += \
            2*tau[:, jj:jj+1]*k_pred_grad_all_train_points[ii, :, :]
        tmp1 = K_train_grad_all_train_points_jj.T[:, np.newaxis, :] *\
            np.tile(tau[:, jj:jj+1], (2, 1, ntrain_samples))
        tmp1[:, :, jj] = K_train_grad_all_train_points_jj.T.dot(tau.T)
        tmp2 = np.sum(tau*tmp1, axis=(2))
        jac[:, ii*nvars:(ii+1)*nvars] -= tmp2.T  # check if -= is needed over =
        # leave the following for loop to show how sparsity is taken advantage
        # of above. Above is abstract and hard to see what is being done
        # for kk in range(nvars):
        #     # K_train_grad[jj, :] = K_train_grad_all_train_points_jj[:, kk]
        #     # K_train_grad[:, jj] = K_train_grad[jj, :]
        #     # The following takes advantage of sparsity of
        #     # tmp = tau.dot(K_train_grad)
        #     # Reset to zero
        #     # K_train_grad[jj, :] = 0
        #     # K_train_grad[:, jj] = 0
        #     tmp = K_train_grad_all_train_points_jj[:, kk:kk+1].T *\
        #         np.tile(tau[:, jj:jj+1], (1, ntrain_samples))
        #     tmp[:, jj] = tau.dot(K_train_grad_all_train_points_jj[:, kk])
        #     assert np.allclose(tmp[:,jj], tmp1[kk,:,jj])
        #     assert np.allclose(tmp,tmp1[kk,:,:])
        #     jac[:, ii*nvars+kk] -= np.sum(tmp*tau, axis=1)
        ii += 1
    jac *= -1
    return jac


def gaussian_grad_P_diag_term1(xtr_ii, lscale, mu, sigma):
    m, s, l, a = mu, sigma, lscale, xtr_ii
    term1 = (np.exp(-((a-m)**2/(l**2+2*s**2)))*l*(-a+m))/(l**2+2*s**2)**(3/2)
    return term1


def gaussian_grad_P_diag_term2(xtr_ii, lscale, mu, sigma):
    n, p, q, b = mu, sigma, lscale, xtr_ii
    term2 = np.exp(-((b-n)**2/(2*p**2+q**2)))/(p*np.sqrt(1/p**2+2/q**2))
    return term2


def gaussian_grad_P_offdiag_term1(xtr_ii, xtr_jj, lscale, mu, sigma):
    m, s, l, a, c = mu, sigma, lscale, xtr_ii, xtr_jj
    term1 = (
        np.exp(-((-2*c*l**2*m+2*l**2*m**2+a**2*(l**2+s**2)+c**2*(l**2+s**2) -
                  2*a*(l**2*m+c*s**2))/(
                      2*l**2*(l**2+2*s**2))))*(l**2*m+c*s**2-a*(l**2+s**2)))/(l*(l**2+2*s**2)**(3/2))
    return term1


def gaussian_grad_P_offdiag_term2(xtr_ii, xtr_jj, lscale, mu, sigma):
    b, d, q, n, p = xtr_ii, xtr_jj, lscale, mu, sigma
    term2 = np.exp(-((-2*d*n*q**2+2*n**2*q**2+b**2*(p**2+q**2)+d **
                      2*(p**2+q**2)-2*b*(d*p**2+n*q**2))/(2*q**2*(2*p**2+q**2))))
    term2 /= p*np.sqrt(1/p**2+2/q**2)
    return term2


def integrate_grad_P(xx, ww, xtr, lscale):
    nvars = len(lscale)
    assert len(xx) == len(ww) == nvars
    assert xtr.shape[0] == nvars
    dist_func = partial(cdist, metric='sqeuclidean')
    ntrain_samples = xtr.shape[1]
    grad_P = np.empty((nvars*ntrain_samples, ntrain_samples))
    K = []  # keep K as list to allow for different size quadrature rules
    diffs = []  # similarly for diffs
    P = np.empty((nvars, ntrain_samples, ntrain_samples))
    for nn in range(nvars):
        xx_1d, ww_1d = xx[nn], ww[nn]
        lscale_nn = lscale[nn]
        dists_1d_x1_xtr = dist_func(
            xx_1d[:, np.newaxis]/lscale_nn, xtr[nn:nn+1, :].T/lscale_nn)
        K.append(np.exp(-.5*dists_1d_x1_xtr))
        P[nn] = K[-1].T.dot(ww_1d[:, np.newaxis]*K[-1])
        diffs.append(-(xtr[nn:nn+1, :].T-xx_1d)/lscale_nn**2)

    # TODO replace loop over train samples with numpy operations
    for ii in range(ntrain_samples):
        for nn in range(nvars):
            diff = diffs[nn][ii]
            grad_P[nvars*ii+nn, :] = ww_1d.dot(
                (diff*K[nn][:, ii])[:, np.newaxis]*K[nn])
            grad_P[nvars*ii+nn, :] *= np.prod(P[:nn, ii, :], axis=0)
            grad_P[nvars*ii+nn, :] *= np.prod(P[nn+1:, ii, :], axis=0)
            grad_P[nvars*ii+nn, ii] *= 2
    return grad_P, np.prod(P, axis=0)


class IVARSampler(object):
    """
    Parameters
    ----------
    num_vars : integer
        The number of dimensions

    nquad_samples : integer
        The number of samples used to compute the sample based estimate
        of the integrated variance (IVAR). If use_quadrature is True
        then this should be 100-1000. Otherwise this value should be at 
        least 10,000.

    ncandidate_samples : integer
        The number of samples used by the greedy downselection procedure
        used to determine the initial guess (set of points) for the gradient 
        based optimization

    generate_random_samples : callable
        Function with signature

        ``generate_random_samples(nsamples) -> np.ndarray (nvars, nsamples)``

        used to generate samples needed to compute IVAR using Monte Carlo
        quadrature. Note even if use_gauss_quadrature is True, this function  
        will be used (if provided) to enrich the default candidate set of the 
        greedy method used to compute the initial guess for the gradient based 
        optimization.
        If this is not None then num_candidate_samples//2 will be created
        by this function and the other half of samples will be from a Halton
        sequence.

    variables : :class:`pyapprox.variable.IndependentMultivariateRandomVariable`
        A set of independent univariate random variables. The tensor-product
        of the 1D PDFs yields the joint density :math:`\rho`. The bounds and CDFs
        of these variables are used to transform the Halton sequence used as
        the candidate set for the greedy generation of the initial guess.

    greedy_method : string
        Name of the greedy strategy for computing the initial guess used
        for the gradient based optimization

    use_gauss_quadrature : boolean
        True - Assume the kernel is the tensor product of univariate kernels
               and compute integrated variance by computing a set of univariate
               integrals with Gaussian quadrature
        False - Use monte carlo quadrature to estimate integrated variance.
                Any kernel can be used.

    nugget : float
        A small value added to the diagonal of the kernel matrix to improve
        conditioning.
    """

    def __init__(self, num_vars, nquad_samples,
                 ncandidate_samples, generate_random_samples, variables=None,
                 greedy_method='ivar', use_gauss_quadrature=False,
                 nugget=0):
        self.nvars = num_vars
        self.nquad_samples = nquad_samples
        self.greedy_method = greedy_method
        self.use_gauss_quadrature = use_gauss_quadrature
        self.pred_samples = generate_random_samples(self.nquad_samples)
        self.ncandidate_samples = ncandidate_samples
        self.variables = variables
        self.generate_random_samples = generate_random_samples
        self.nugget = nugget
        self.ntraining_samples = 0
        self.training_samples = np.empty((num_vars, self.ntraining_samples))
        self.nsamples_requested = []
        self.set_optimization_options(
            {'gtol': 1e-8, 'ftol': 0, 'disp': False, 'iprint': 0})
        self.initialize_greedy_sampler()

        if use_gauss_quadrature:
            self.precompute_gauss_quadrature()
            self.objective = self.quadrature_objective
            self.objective_gradient = self.quadrature_objective_gradient
            assert self.greedy_sampler.variables is not None
        else:
            self.objective = self.monte_carlo_objective
            self.objective_gradient = self.monte_carlo_objective_gradient

    def initialize_greedy_sampler(self):
        if self.greedy_method == 'chol':
            self.greedy_sampler = CholeskySampler(
                self.nvars, self.ncandidate_samples, self.variables,
                generate_random_samples=self.generate_random_samples)
        elif self.greedy_method == 'ivar':
            self.greedy_sampler = GreedyIntegratedVarianceSampler(
                self.nvars, self.nquad_samples, self.ncandidate_samples,
                self.generate_random_samples, self.variables,
                use_gauss_quadrature=self.use_gauss_quadrature, econ=True,
                nugget=self.nugget)
        else:
            msg = f'Incorrect greedy_method {greedy_method}'
            raise Exception(msg)

    def precompute_gauss_quadrature(self):
        degrees = [min(100, self.nquad_samples)]*self.nvars
        self.univariate_quad_rules, self.pce = \
            get_univariate_quadrature_rules_from_variable(
                self.greedy_sampler.variables, degrees)
        self.quad_rules = []
        for ii in range(self.nvars):
            xx_1d, ww_1d = self.univariate_quad_rules[ii](degrees[ii]+1)
            jj = self.pce.basis_type_index_map[ii]
            loc, scale = self.pce.var_trans.scale_parameters[jj, :]
            xx_1d = xx_1d*scale+loc
            self.quad_rules.append([xx_1d, ww_1d])

    def get_univariate_quadrature_rule(self, ii):
        return self.quad_rules[ii]

    def compute_P(self, train_samples):
        self.degrees = [self.nquad_samples]*self.nvars
        length_scale = self.greedy_sampler.kernel.length_scale
        if np.isscalar(length_scale):
            length_scale = np.array([length_scale]*self.nvars)
        P = 1
        for ii in range(self.nvars):
            xx_1d, ww_1d = self.get_univariate_quadrature_rule(ii)
            xtr = train_samples[ii:ii+1, :]
            K = self.greedy_sampler.kernels_1d[ii](
                xx_1d[np.newaxis, :], xtr, length_scale[ii])
            P_ii = K.T.dot(ww_1d[:, np.newaxis]*K)
            P *= P_ii
        return P

    def quadrature_objective(self, new_train_samples_flat):
        train_samples = np.hstack(
            [self.training_samples,
             new_train_samples_flat.reshape(
                 (self.nvars, new_train_samples_flat.shape[0]//self.nvars),
                 order='F')])
        A = self.greedy_sampler.kernel(train_samples.T)
        A[np.arange(A.shape[0]), np.arange(A.shape[1])] += self.nugget

        A_inv = np.linalg.inv(A)
        P = self.compute_P(train_samples)
        return 1-np.trace(A_inv.dot(P))

    def quadrature_objective_gradient(self, new_train_samples_flat):
        train_samples = np.hstack(
            [self.training_samples,
             new_train_samples_flat.reshape(
                 (self.nvars, new_train_samples_flat.shape[0]//self.nvars),
                 order='F')])
        xx = [q[0] for q in self.quad_rules]
        ww = [q[1] for q in self.quad_rules]
        new_samples_index = self.training_samples.shape[1]
        return RBF_integrated_posterior_variance_gradient_wrt_samples(
            train_samples, xx, ww, self.greedy_sampler.kernel,
            new_samples_index, nugget=self.nugget)

    def monte_carlo_objective(self, new_train_samples_flat):
        train_samples = np.hstack(
            [self.training_samples,
             new_train_samples_flat.reshape(
                 (self.nvars, new_train_samples_flat.shape[0]//self.nvars),
                 order='F')])
        val = gaussian_process_pointwise_variance(
            self.greedy_sampler.kernel, self.pred_samples,
            train_samples, self.nugget).mean()
        # print('f',val)
        return val

    def monte_carlo_objective_gradient(self, new_train_samples_flat):
        train_samples = np.hstack(
            [self.training_samples,
             new_train_samples_flat.reshape(
                 (self.nvars, new_train_samples_flat.shape[0]//self.nvars),
                 order='F')])
        new_samples_index = self.training_samples.shape[1]
        return RBF_posterior_variance_jacobian_wrt_samples(
            train_samples, self.pred_samples, self.greedy_sampler.kernel,
            new_samples_index, self.nugget).mean(axis=0)

    def set_weight_function(self, weight_function):
        self.greedy_sampler.set_weight_function(weight_function)

    def set_kernel(self, kernel, kernels_1d=None):
        if ((self.use_gauss_quadrature is True) and (self.nvars != 1) and
                ((type(kernel) != Matern) or (np.isfinite(kernel.nu)))):
            # TODO: To deal with sum kernel with noise, need to ammend
            # gradient computation which currently assumes no noise
            msg = f'GP Kernel type: {type(kernel)} '
            msg += 'Only squared exponential kernel supported when '
            msg += 'use_gauss_quadrature is True and nvars > 1'
            # TODO add other tensor product kernels
            raise Exception(msg)
        self.greedy_sampler.set_kernel(copy.deepcopy(kernel), kernels_1d)

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
        self.bounds = Bounds(lbs, ubs)

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
        candidate_samples = np.hstack([
            self.training_samples.copy(), candidate_samples])

        # make sure greedy sampler recomputes all necessary information
        # but first extract necessary information
        pred_samples = self.greedy_sampler.pred_samples
        if hasattr(self.greedy_sampler, 'weight_function'):
            weight_function = self.greedy_sampler.weight_function
        else:
            weight_function = None
        kernel = self.greedy_sampler.kernel
        self.initialize_greedy_sampler()
        if weight_function is not None:
            self.set_weight_function(weight_function)
        # self.greedy_sampler.candidate_samples must be called before
        # set kernel to make sure self.A matrix is set correctly
        self.greedy_sampler.candidate_samples = candidate_samples
        # currently the following will no effect a different set
        # of prediction samples will be generated by greedy sampler when
        # set kernel is called
        self.greedy_sampler.pred_samples = pred_samples
        self.set_kernel(kernel)

        # Make sure greedy_sampler chooses self.training_samples
        # only used if greedy_sampler is a Choleskysampler.
        self.greedy_sampler.set_init_pivots(np.arange(self.ntraining_samples))

        # Get the initial guess for new samples to add.
        # Note the Greedy sampler will return only new samples not in
        # self.training_samples
        self.init_guess, chol_flag = self.greedy_sampler(nsamples)
        self.init_guess = self.init_guess[:, self.ntraining_samples:]
        # assert np.allclose(
        #    self.greedy_sampler.L[:self.ntraining_samples,
        #                          :self.ntraining_samples],
        #    np.linalg.cholesky(kernel(self.training_samples.T)))
        assert chol_flag == 0

        self.set_bounds(nsamples-self.ntraining_samples)

        init_guess = self.init_guess.flatten(order='F')
        # Optimize the locations of only the new training samples
        jac = self.objective_gradient
        res = minimize(self.objective, init_guess, jac=jac,
                       method='L-BFGS-B', options=self.optim_opts,
                       bounds=self.bounds)
        print(res)

        new_samples = res.x.reshape(
            (self.nvars, res.x.shape[0]//self.nvars), order='F')
        self.training_samples = np.hstack([self.training_samples, new_samples])
        self.ntraining_samples = self.training_samples.shape[1]

        return new_samples, 0


def matern_kernel_1d_inf(dists):
    return np.exp(-.5*dists**2)


def matern_kernel_1d_12(dists):
    return np.exp(-dists)


def matern_kernel_1d_32(dists):
    tmp = np.sqrt(3)*dists
    return (1+tmp)*np.exp(-tmp)


def matern_kernel_1d_52(dists):
    tmp = np.sqrt(5)*dists
    return (1+tmp+tmp**2/3)*np.exp(-tmp)


def matern_kernel_general(nu, dists):
    dists[dists == 0] += np.finfo(float).eps
    tmp = (np.sqrt(2*nu) * dists)
    return tmp**nu*(2**(1.-nu))/gamma(nu)*kv(nu, tmp)


def matern_kernel_1d(nu, x, y, lscale):
    explicit_funcs = {0.5: matern_kernel_1d_12, 1.5: matern_kernel_1d_32,
                      2.: matern_kernel_1d_52, np.inf: matern_kernel_1d_inf}
    dist_func = partial(cdist, metric='euclidean')
    dists = dist_func(x.T/lscale, y.T/lscale)
    if nu in explicit_funcs:
        return explicit_funcs[nu](dists)

    return matern_kernel_general(nu, dists)


class GreedyVarianceOfMeanSampler(object):
    """
    Parameters
    ----------
    num_vars : integer
        The number of dimensions

    nquad_samples : integer
        The number of samples used to compute the sample based estimate
        of the variance of mean criteria 

    ncandidate_samples : integer
        The number of samples used by the greedy downselection procedure
    """

    def __init__(self, num_vars, nquad_samples,
                 ncandidate_samples, generate_random_samples, variables=None,
                 use_gauss_quadrature=False, econ=True,
                 compute_cond_nums=False, nugget=0):
        self.nvars = num_vars
        self.nquad_samples = nquad_samples
        self.variables = variables
        self.ntraining_samples = 0
        self.training_samples = np.empty((num_vars, self.ntraining_samples))
        self.generate_random_samples = generate_random_samples
        self.use_gauss_quadrature = use_gauss_quadrature
        self.econ = econ

        self.candidate_samples = generate_candidate_samples(
            self.nvars, ncandidate_samples, generate_random_samples,
            self.variables)
        self.nsamples_requested = []
        self.pivots = []
        self.cond_nums = []
        self.compute_cond_nums = compute_cond_nums
        self.init_pivots = None
        self.nugget = nugget
        self.initialize()
        self.best_obj_vals = []
        self.pred_samples = None

    def initialize(self):
        self.L = np.zeros((0, 0))

        if self.econ is True:
            self.y_1 = np.zeros((0))
            self.candidate_y_2 = np.empty(self.candidate_samples.shape[1])

    # def monte_carlo_objective(self, new_sample_index):
    #     train_samples = np.hstack(
    #         [self.training_samples,
    #          self.candidate_samples[:, new_sample_index:new_sample_index+1]])
    #     return gaussian_process_pointwise_variance(
    #         self.kernel, self.pred_samples,
    #         train_samples).mean()

    def precompute_monte_carlo(self):
        self.pred_samples = self.generate_random_samples(
            self.nquad_samples)
        k = self.kernel(self.pred_samples.T, self.candidate_samples.T)
        self.tau = k.mean(axis=0)
        assert self.tau.shape[0] == self.candidate_samples.shape[1]

        # Note because tau is simplified down to one integral instead of their
        # double used for u, it is possible for self.u - tau.dot(A_inv.dot(tau)
        # to be negative if tau is comptued using an inaccurate quadrature rule.
        # This is not important if using gauss quadrature
        #pred_samples2 = self.generate_random_samples(self.pred_samples.shape[1])
        # self.u = np.diag(
        #    self.kernel(self.pred_samples.T, pred_samples2.T)).mean()

    def get_univariate_quadrature_rule(self, ii):
        xx_1d, ww_1d = self.univariate_quad_rules[ii](self.degrees[ii]+1)
        jj = self.pce.basis_type_index_map[ii]
        loc, scale = self.pce.var_trans.scale_parameters[jj, :]
        xx_1d = xx_1d*scale+loc
        return xx_1d, ww_1d

    def precompute_gauss_quadrature(self):
        nvars = self.variables.num_vars()
        length_scale = self.kernel.length_scale
        if np.isscalar(length_scale):
            length_scale = [length_scale]*nvars
        self.degrees = [self.nquad_samples]*nvars

        self.univariate_quad_rules, self.pce = \
            get_univariate_quadrature_rules_from_variable(
                self.variables, self.degrees)
        dist_func = partial(cdist, metric='sqeuclidean')
        self.tau = 1

        for ii in range(self.nvars):
            # Get 1D quadrature rule
            xx_1d, ww_1d = self.get_univariate_quadrature_rule(ii)

            # Training samples of ith variable
            xtr = self.candidate_samples[ii:ii+1, :]
            lscale_ii = length_scale[ii]
            # dists_1d_x1_xtr = dist_func(
            #    xx_1d[:, np.newaxis]/lscale_ii, xtr.T/lscale_ii)
            #K = np.exp(-.5*dists_1d_x1_xtr)
            K = self.kernels_1d[ii](xx_1d[np.newaxis, :], xtr, lscale_ii)
            self.tau *= ww_1d.dot(K)

    def objective(self, new_sample_index):
        indices = np.concatenate(
            [self.pivots, [new_sample_index]]).astype(int)
        A = self.A[np.ix_(indices, indices)]
        try:
            L = np.linalg.cholesky(A)
        except:
            return np.inf
        tau = self.tau[indices]
        return -tau.T.dot(cholesky_solve_linear_system(L, tau))

    def objective_vals(self):
        obj_vals = np.inf*np.ones(self.candidate_samples.shape[1])
        for mm in range(self.candidate_samples.shape[1]):
            if mm not in self.pivots:
                obj_vals[mm] = self.objective(mm)
        # assert np.allclose(self.candidate_samples[:,self.pivots],self.training_samples)
        # if len(self.pivots)>22:
        #     I = np.argsort(self.candidate_samples[0,:])
        #     plt.plot(self.candidate_samples[0,self.pivots],np.ones((len(self.pivots)))*obj_vals.min(),'ko')
        #     plt.plot(self.candidate_samples[0,I],obj_vals[I])
        #     J = np.argmin(obj_vals)
        #     plt.plot(self.candidate_samples[0,J],obj_vals[J], 'rs')
        #     plt.show()
        return obj_vals

    def refine_naive(self):
        if (self.init_pivots is not None and
                len(self.pivots) < len(self.init_pivots)):
            pivot = self.init_pivots[len(self.pivots)]
            obj_val = self.objective(pivot)
        else:
            ntraining_samples = self.ntraining_samples
            obj_vals = self.objective_vals()
            pivot = np.argmin(obj_vals)
            obj_val = obj_vals[pivot]
        return pivot, obj_val

    def refine_econ(self):
        if (self.init_pivots is not None and
                len(self.pivots) < len(self.init_pivots)):
            pivot = self.init_pivots[len(self.pivots)]
            obj_val = self.objective_econ(pivot)
        else:
            training_samples = self.ntraining_samples
            obj_vals = self.vectorized_objective_vals_econ()
            pivot = np.argmin(obj_vals)

        assert np.isfinite(obj_vals[pivot])

        if self.L.shape[0] == 0:
            self.L = np.atleast_2d(self.A[pivot, pivot])
        else:
            A_12 = self.A[self.pivots, pivot:pivot+1]
            L_12 = solve_triangular(self.L, A_12, lower=True)
            L_22_sq = self.A[pivot, pivot] - L_12.T.dot(L_12)
            if L_22_sq <= 0:
                # recompute Cholesky from scratch to make sure roundoff error
                # is not causing L_22_sq to be negative
                indices = np.concatenate([self.pivots, [pivot]]).astype(int)
                try:
                    self.L = np.linalg.cholesky(
                        self.A[np.ix_(indices, indices)])
                except:
                    return -1, np.inf

            L_22 = np.sqrt(L_22_sq)
            self.L = np.block(
                [[self.L, np.zeros(L_12.shape)],
                 [L_12.T, L_22]])

        assert np.isfinite(self.candidate_y_2[pivot])
        self.y_1 = np.concatenate([self.y_1, [self.candidate_y_2[pivot]]])

        return pivot, obj_vals[pivot]

    def objective_vals_econ(self):
        obj_vals = np.inf*np.ones(self.candidate_samples.shape[1])
        for mm in range(self.candidate_samples.shape[1]):
            if mm not in self.pivots:
                obj_vals[mm] = self.objective_econ(mm)
        return obj_vals

    def vectorized_objective_vals_econ(self):
        if self.L.shape[0] == 0:
            diag_A = np.diagonal(self.A)
            L = np.sqrt(diag_A)
            vals = self.tau**2/diag_A
            self.candidate_y_2 = self.tau/L
            return -vals

        A_12 = np.atleast_2d(self.A[self.pivots, :])
        L_12 = solve_triangular(self.L, A_12, lower=True)
        J = np.where((np.diagonal(self.A)-np.sum(L_12*L_12, axis=0)) <= 0)[0]
        self.temp = np.diagonal(self.A)-np.sum(L_12*L_12, axis=0)
        useful_candidates = np.ones(
            (self.candidate_samples.shape[1]), dtype=bool)
        useful_candidates[J] = False
        useful_candidates[self.pivots] = False
        L_12 = L_12[:, useful_candidates]
        L_22 = np.sqrt(np.diagonal(self.A)[useful_candidates] - np.sum(
            L_12*L_12, axis=0))
        y_2 = (self.tau[useful_candidates]-L_12.T.dot(self.y_1))/L_22
        self.candidate_y_2[useful_candidates] = y_2
        self.candidate_y_2[~useful_candidates] = np.inf
        z_2 = y_2/L_22
        vals = np.inf*np.ones((self.candidate_samples.shape[1]))

        vals[useful_candidates] = -(
            self.best_obj_vals[-1] + self.tau[useful_candidates]*z_2 -
            self.tau[self.pivots].dot(
                solve_triangular(self.L.T, L_12*z_2, lower=False)))
        return vals

    def objective_econ(self, new_sample_index):
        if self.L.shape[0] == 0:
            L = np.sqrt(self.A[new_sample_index, new_sample_index])
            self.candidate_y_2[new_sample_index] = self.tau[new_sample_index]/L
            val = self.tau[new_sample_index]**2/self.A[
                new_sample_index, new_sample_index]
            return -val

        A_12 = self.A[self.pivots, new_sample_index:new_sample_index+1]
        L_12 = solve_triangular(self.L, A_12, lower=True)
        L_22 = np.sqrt(
            self.A[new_sample_index, new_sample_index] - L_12.T.dot(L_12))
        y_2 = (self.tau[new_sample_index]-L_12.T.dot(self.y_1))/L_22[0, 0]
        self.candidate_y_2[new_sample_index] = y_2
        z_2 = y_2/L_22[0, 0]

        val = -(-self.best_obj_vals[-1] + self.tau[new_sample_index]*z_2 -
                self.tau[self.pivots].dot(
                    solve_triangular(self.L.T, L_12*z_2, lower=False)))
        return val[0, 0]

    def set_kernel(self, kernel, kernels_1d=None):
        self.kernel = kernel

        self.kernels_1d = kernels_1d
        if self.kernels_1d is None and self.use_gauss_quadrature:
            # TODO: remove kernels 1D and just create tensor product
            # kernel with this as a property.
            assert self.kernel.nu == np.inf
            self.kernels_1d = [partial(matern_kernel_1d, np.inf)]*self.nvars

        if ((self.use_gauss_quadrature is True) and (self.nvars != 1) and
                ((type(kernel) != Matern) or (np.isfinite(kernel.nu)))):
            # TODO: To deal with sum kernel with noise, need to ammend
            # gradient computation which currently assumes no noise
            msg = f'GP Kernel type: {type(kernel)} '
            msg += 'Only squared exponential kernel supported when '
            msg += 'use_gauss_quadrature is True and nvars > 1'
            # TODO add other tensor product kernels
            raise Exception(msg)

        self.active_candidates = np.ones(
            self.candidate_samples.shape[1], dtype=bool)
        if self.use_gauss_quadrature:
            self.precompute_gauss_quadrature()
        else:
            self.precompute_monte_carlo()
        self.A = self.kernel(self.candidate_samples.T,
                             self.candidate_samples.T)
        # designs are better if a small nugget is added to the diagonal
        self.add_nugget()

    def add_nugget(self):
        self.A[np.arange(self.A.shape[0]), np.arange(self.A.shape[1])] += \
            self.nugget
        print(self.nugget)

    def set_init_pivots(self, init_pivots):
        assert len(self.pivots) == 0
        self.init_pivots = list(init_pivots)

    def __call__(self, nsamples, verbosity=1):
        if not hasattr(self, 'kernel'):
            raise Exception('Must call set_kernel')
        if self.econ is True:
            self.refine = self.refine_econ
        else:
            self.refine = self.refine_naive
        flag = 0
        self.nsamples_requested.append(nsamples)
        ntraining_samples = self.ntraining_samples
        for nn in range(ntraining_samples, nsamples):
            pivot, obj_val = self.refine()

            if pivot < 0:
                flag = 1
                break
                # if self.econ is False:
                #     flag = 1
                #     break
                # else:
                #     self.econ = False
                #     # Switch of econ mode which struggles when condition number
                #     # is poor
                #     print('switching naive updating strategy on')
                #     self.refine = self.refine_naive
                #     pivot, obj_val = self.refine()
                #     if pivot < 0:
                #         flag = 1
                #         break
            if verbosity > 0:
                print(f'Iter: {nn}, Objective: {obj_val}')
            self.best_obj_vals.append(obj_val)

            self.pivots.append(pivot)
            new_sample = self.candidate_samples[:, pivot:pivot+1]
            self.training_samples = np.hstack(
                [self.training_samples,
                 self.candidate_samples[:, pivot:pivot+1]])
            #print(f'Number of points generated {nn+1}')
            self.active_candidates[pivot] = False
            if self.compute_cond_nums is True:
                if self.econ:
                    self.cond_nums.append(np.linalg.cond(self.L)**2)
                else:
                    self.cond_nums.append(
                        np.linalg.cond(
                            self.A[np.ix_(self.pivots, self.pivots)]))
            print(np.linalg.cond(
                self.A[np.ix_(self.pivots, self.pivots)]))

        new_samples = self.training_samples[:, ntraining_samples:]
        self.ntraining_samples = self.training_samples.shape[1]

        return new_samples, flag


def matern_gradient_wrt_samples(nu, query_sample, other_samples, length_scale):
    length_scale = np.asarray(length_scale)
    dists = cdist(query_sample.T/length_scale, other_samples.T/length_scale,
                  metric='euclidean')
    if nu == 3/2:
        tmp1 = np.sqrt(3)*dists
        tmp2 = (np.tile(
            query_sample.T, (other_samples.shape[1], 1))-other_samples.T)/(
                length_scale**2)
        K = np.exp(-tmp1)
        grad = -3*K.T*tmp2
    elif nu == 5/2:
        tmp1 = np.sqrt(5)*dists
        K = np.exp(-tmp1)
        tmp2 = (np.tile(
            query_sample.T, (other_samples.shape[1], 1))-other_samples.T)/(
                length_scale**2)
        grad = -5/3*K.T*tmp2*(np.sqrt(5)*dists+1)
    elif nu == np.inf:
        tmp2 = (np.tile(
            query_sample.T, (other_samples.shape[1], 1))-other_samples.T)/(
                length_scale**2)
        K = np.exp(-.5 * dists**2)
        grad = -K.T*tmp2
    else:
        raise Exception(f'Matern gradient with nu={nu} not supported')
    return grad


class GreedyIntegratedVarianceSampler(GreedyVarianceOfMeanSampler):
    """
    Parameters
    ----------
    num_vars : integer
        The number of dimensions

    nquad_samples : integer
        The number of samples used to compute the sample based estimate
        of the integrated variance (IVAR)

    ncandidate_samples : integer
        The number of samples used by the greedy downselection procedure
    """

    def initialize(self):
        self.L = np.zeros((0, 0))
        self.L_inv = np.zeros((0, 0))
        self.A_inv = np.zeros((0, 0))

    def precompute_monte_carlo(self):
        self.pred_samples = self.generate_random_samples(
            self.nquad_samples)
        #lscale = self.kernel.length_scale
        # if np.isscalar(lscale):
        #    lscale = np.array([lscale]*self.nvars)
        #dist_func = partial(cdist, metric='sqeuclidean')
        # dists_x1_xtr = dist_func(
        #    self.pred_samples.T/lscale, self.candidate_samples.T/lscale)
        #K = np.exp(-.5*dists_x1_xtr)
        K = self.kernel(self.pred_samples.T, self.candidate_samples.T)
        ww = np.ones(self.pred_samples.shape[1])/self.pred_samples.shape[1]
        self.P = K.T.dot(ww[:, np.newaxis]*K)

    def precompute_gauss_quadrature(self):
        self.degrees = [self.nquad_samples]*self.nvars
        length_scale = self.kernel.length_scale
        if np.isscalar(length_scale):
            length_scale = np.array([length_scale]*self.nvars)
        self.univariate_quad_rules, self.pce = \
            get_univariate_quadrature_rules_from_variable(
                self.variables, self.degrees)
        self.P = 1
        for ii in range(self.nvars):
            xx_1d, ww_1d = self.get_univariate_quadrature_rule(ii)
            xtr = self.candidate_samples[ii:ii+1, :]
            K = self.kernels_1d[ii](
                xx_1d[np.newaxis, :], xtr, length_scale[ii])
            P_ii = K.T.dot(ww_1d[:, np.newaxis]*K)
            self.P *= P_ii

    def objective(self, new_sample_index):
        indices = np.concatenate(
            [self.pivots, [new_sample_index]]).astype(int)
        A = self.A[np.ix_(indices, indices)]
        A_inv = np.linalg.inv(A)
        P = self.P[np.ix_(indices, indices)]
        # P1=1
        # length_scale = self.kernel.length_scale
        # if np.isscalar(length_scale):
        #     length_scale = np.array([length_scale]*self.nvars)
        # for ii in range(self.nvars):
        #     xx_1d, ww_1d = self.get_univariate_quadrature_rule(ii)
        #     xtr = self.candidate_samples[ii:ii+1, indices]
        #     K = self.kernels_1d[ii](xx_1d[np.newaxis, :], xtr, length_scale[ii])
        #     P_ii = K.T.dot(ww_1d[:, np.newaxis]*K)
        #     P1*=P_ii
        # assert np.allclose(P, P1)

        return -np.trace(A_inv.dot(P))

    def objective_econ(self, new_sample_index):
        if self.L_inv.shape[0] == 0:
            val = self.P[new_sample_index, new_sample_index]/self.A[
                new_sample_index, new_sample_index]
            return -val

        A_12 = self.A[self.pivots, new_sample_index:new_sample_index+1]
        L_12 = solve_triangular(self.L, A_12, lower=True)
        L_22 = np.sqrt(
            self.A[new_sample_index, new_sample_index] - L_12.T.dot(L_12))
        C = -np.dot(L_12.T/L_22, self.L_inv)

        # TODO set self.P_11 when pivot is chosen so do not constantly
        # have to reduce matrix
        P_11 = self.P[np.ix_(self.pivots, self.pivots)]
        P_12 = self.P[self.pivots, new_sample_index:new_sample_index+1]
        P_22 = self.P[new_sample_index, new_sample_index]

        val = -(-self.best_obj_vals[-1] + np.sum(C.T.dot(C)*P_11) +
                2*np.sum(C.T/L_22*P_12) + 1/L_22**2*P_22)
        return val[0, 0]

    def vectorized_objective_vals_econ(self):
        if self.L_inv.shape[0] == 0:
            vals = np.diagonal(self.P)/np.diagonal(self.A)
            return -vals

        A_12 = np.atleast_2d(self.A[self.pivots, :])
        L_12 = solve_triangular(self.L, A_12, lower=True)
        J = np.where((np.diagonal(self.A)-np.sum(L_12*L_12, axis=0)) <= 0)[0]
        self.temp = np.diagonal(self.A)-np.sum(L_12*L_12, axis=0)
        useful_candidates = np.ones(
            (self.candidate_samples.shape[1]), dtype=bool)
        useful_candidates[J] = False
        useful_candidates[self.pivots] = False
        L_12 = L_12[:, useful_candidates]
        L_22 = np.sqrt(np.diagonal(self.A)[useful_candidates] - np.sum(
            L_12*L_12, axis=0))

        P_11 = self.P[np.ix_(self.pivots, self.pivots)]
        P_12 = self.P[np.ix_(self.pivots, useful_candidates)]
        P_22 = np.diagonal(self.P)[useful_candidates]

        C = -np.dot((L_12/L_22).T, self.L_inv)
        vals = np.inf*np.ones((self.candidate_samples.shape[1]))

        vals[useful_candidates] = -(
            -self.best_obj_vals[-1] +
            np.sum(C.T*P_11.dot(C.T), axis=0) +
            2*np.sum(C.T/L_22*P_12, axis=0) + 1/L_22**2*P_22)

        return vals

    def refine_econ(self):
        if (self.init_pivots is not None and
                len(self.pivots) < len(self.init_pivots)):
            pivot = self.init_pivots[len(self.pivots)]
            obj_val = self.objective_econ(pivot)
        else:
            training_samples = self.ntraining_samples
            obj_vals = self.vectorized_objective_vals_econ()
            # obj_vals = self.objective_vals_econ()
            pivot = np.argmin(obj_vals)
            obj_val = obj_vals[pivot]

        if not np.isfinite(obj_val):  # or obj_val < -1:
            # ill conditioning causes obj_val to go below -1 which should not
            # be possible
            return -1, np.inf

        if self.L_inv.shape[0] == 0:
            self.L = np.atleast_2d(self.A[pivot, pivot])
            self.L_inv = np.atleast_2d(1/self.A[pivot, pivot])
            return pivot, obj_val

        A_12 = self.A[self.pivots, pivot:pivot+1]
        L_12 = solve_triangular(self.L, A_12, lower=True)
        L_22_sq = self.A[pivot, pivot] - L_12.T.dot(L_12)
        if L_22_sq <= 0:
            # recompute Cholesky from scratch to make sure roundoff error
            # is not causing L_22_sq to be negative
            indices = np.concatenate([self.pivots, [pivot]]).astype(int)
            try:
                self.L = np.linalg.cholesky(self.A[np.ix_(indices, indices)])
            except:
                return -1, np.inf
            self.L_inv = np.linalg.inv(self.L)
            return pivot, obj_val

        L_22 = np.sqrt(L_22_sq)

        self.L = np.block(
            [[self.L, np.zeros(L_12.shape)],
             [L_12.T, L_22]])
        indices = np.concatenate([self.pivots, [pivot]]).astype(int)
        L_22_inv = np.linalg.inv(L_22)
        self.L_inv = np.block(
            [[self.L_inv, np.zeros(L_12.shape)],
             [-np.dot(L_22_inv.dot(L_12.T), self.L_inv), L_22_inv]])

        return pivot, obj_val
