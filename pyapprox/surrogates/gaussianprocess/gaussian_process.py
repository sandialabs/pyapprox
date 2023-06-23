import numpy as np
import copy
from itertools import combinations
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import cdist
from functools import partial
from scipy.linalg import solve_triangular
from scipy.special import kv, gamma

from sklearn.gaussian_process import GaussianProcessRegressor

from pyapprox.surrogates.gaussianprocess.kernels import (
    Matern, ConstantKernel, WhiteKernel, RBF, MultilevelKernel,
    extract_covariance_kernel, MultifidelityPeerKernel
)
from pyapprox.util.utilities import (
    cartesian_product, outer_product
)
from pyapprox.util.linalg import (
    pivoted_cholesky_decomposition,
    continue_pivoted_cholesky_decomposition, cholesky_solve_linear_system
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.surrogates.interp.indexing import (
    argsort_indices_leixographically
)
from pyapprox.surrogates.polychaos.gpc import (
    get_univariate_quadrature_rules_from_variable
)
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)


class GaussianProcess(GaussianProcessRegressor):
    """
    A Gaussian process.
    """
    def set_variable_transformation(self, var_trans):
        self.var_trans = var_trans

    def map_to_canonical(self, samples):
        if hasattr(self, 'var_trans'):
            return self.var_trans.map_to_canonical(samples)
        return samples

    def map_from_canonical(self, canonical_samples):
        if hasattr(self, 'var_trans'):
            return self.var_trans.map_from_canonical(canonical_samples)
        return canonical_samples

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
        canonical_train_samples = self.map_to_canonical(train_samples)
        return super().fit(canonical_train_samples.T, train_values)

    def __call__(self, samples, return_std=False, return_cov=False,
                 return_grad=False):
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
        if return_grad and (return_std or return_cov):
            msg = "if return_grad is True then return_std and return_cov "
            msg += "must be False"
            raise ValueError(msg)

        canonical_samples = self.map_to_canonical(samples)
        result = self.predict(canonical_samples.T, return_std, return_cov)

        if return_grad:
            kernel = extract_covariance_kernel(self.kernel_, [Matern])
            if kernel is None:
                kernel = extract_covariance_kernel(self.kernel_, [RBF])
                if kernel is None:
                    msg = "return_grad only available when using the Matern kernel"
                    raise ValueError(msg)
                nu = np.inf
            else:
                nu = kernel.nu
            assert samples.shape[1] == 1
            gradK = matern_gradient_wrt_samples(
                nu, samples, self.X_train_.T, kernel.length_scale)
            grad = gradK.T.dot(self.alpha_)
            kernel = extract_covariance_kernel(self.kernel_, [ConstantKernel])
            if result.ndim == 1:
                # gpr in later versions of sklearn only return 1D array
                # while earlier versions return 2D array with one column
                result = result[:, None]
            if kernel is not None:
                grad *= kernel.constant_value
            return result, grad

        if type(result) == tuple:
            # when returning prior stdev covariance then must reshape vals
            if result[0].ndim == 1:
                result = [result[0][:, None]] + [r for r in result[1:]]
                result = tuple(result)
            return result

        if result.ndim == 1:
            # gpr in later versions of sklearn only return 1D array
            # while earlier versions return 2D array with one column
            result = result[:, None]
        return result

    def predict_random_realization(self, samples, rand_noise=1,
                                   truncated_svd=None, keep_normalized=False):
        """
        Predict values of a random realization of the Gaussian process

        Notes
        -----
        A different realization will be returned for two different samples
        Even if the same random noise i used. To see this for a 1D GP use:

        xx = np.linspace(0, 1, 101)
        rand_noise = np.random.normal(0, 1, (xx.shape[0], 1))
        yy = gp.predict_random_realization(xx[None, :], rand_noise)
        plt.plot(xx, yy)
        xx = np.linspace(0, 1, 97)
        rand_noise = np.random.normal(0, 1, (xx.shape[0], 1))
        yy = gp.predict_random_realization(xx[None, :], rand_noise)
        plt.plot(xx, yy)
        plt.show()

        Parameters
        ----------
        truncated_svd : dictionary
           Dictionary containing the following attribues needed to define
           a truncated singular values decomposition. If None then
           factor the entire matrix

        nsingular_vals : integer
            Only compute the first n singular values when
            factorizing the covariance matrix. n=truncated_svd

        tol : float
            The contribution to total variance from the truncated singular
            values must not exceed this value.
        Notes
        -----
        This function replaces
        gp.sample_y(samples.T, n_samples=rand_noise, random_state=0)
        which cannot be passed rand_noise vectors and cannot use truncated SVD
        """
        # mapping of samples is performed in __call__
        mean, cov = self(samples, return_cov=True)
        if keep_normalized is True:
            mean = (mean - self._y_train_mean) / self._y_train_std
            cov /= self._y_train_std**2
        # Use SVD because it is more robust than Cholesky
        # L = np.linalg.cholesky(cov)
        if truncated_svd is None:
            U, S, V = np.linalg.svd(cov)
        else:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(
                n_components=min(samples.shape[1]-1,
                                 truncated_svd['nsingular_vals']), n_iter=7)
            svd.fit(cov)
            U = svd.components_.T
            S = svd.singular_values_
            print('Explained variance', svd.explained_variance_ratio_.sum())
            assert svd.explained_variance_ratio_.sum() >= truncated_svd['tol']
            # print(S.shape, cov.shape)
        L = U*np.sqrt(S)
        # create nsamples x nvars then transpose so same samples
        # are produced if this function is called repeatedly with nsamples=1
        if np.isscalar(rand_noise):
            rand_noise = np.random.normal(0, 1, (rand_noise, mean.shape[0])).T
        else:
            assert rand_noise.shape[0] == mean.shape[0]
        if truncated_svd is not None:
            rand_noise = rand_noise[:S.shape[0], :]
        vals = mean + L.dot(rand_noise)
        return vals

    def num_training_samples(self):
        return self.X_train_.shape[0]

    def condition_number(self):
        return np.linalg.cond(self.L_.dot(self.L_.T))

    def get_training_samples(self):
        if hasattr(self, "var_trans") and self.var_trans is not None:
            return self.var_trans.map_from_canonical(self.X_train_.T)
        else:
            return self.X_train_.T

    def plot_1d(self, num_XX_test, bounds,
                ax=None, num_stdev=2, plt_kwargs={},
                fill_kwargs={"alpha": 0.3},
                prior_fill_kwargs=None, model_eval_id=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
        XX_test = np.linspace(bounds[0], bounds[1], num_XX_test)[None, :]
        # return_std=True does not work for gradient enhanced krigging
        # gp_mean, gp_std = predict(XX_test,return_std=True)
        gp_mean, gp_std = self(
            XX_test, return_std=True)
        gp_mean = gp_mean[:, 0]
        if prior_fill_kwargs is not None:
            if model_eval_id is not None:
                self.kernel_.model_eval_id = model_eval_id
            prior_std = np.sqrt(self.kernel_.diag(XX_test.T))
            ax.fill_between(
                XX_test[0, :], self._y_train_mean-num_stdev*prior_std,
                self._y_train_mean+num_stdev*prior_std, **prior_fill_kwargs)
        ax.fill_between(
           XX_test[0, :], gp_mean-num_stdev*gp_std, gp_mean+num_stdev*gp_std,
           **fill_kwargs)
        ax.plot(XX_test[0, :], gp_mean, **plt_kwargs)
        return ax


class RandomGaussianProcessRealizations:
    """
    Light weight wrapper that allows random realizations of a Gaussian process
    to be evaluated at an arbitrary set of samples.
    GaussianProcess.predict_random_realization can only evaluate the GP
    at a finite set of samples. This wrapper can only compute the mean
    interpolant as we assume that the number of training samples
    was sufficient to produce an approximation with accuracy (samll pointwise
    variance acceptable to the user. Unlike GaussianProcess predictions
    can return a np.ndarray (nsamples, nrandom_realizations)
    instead of size (nsamples, 1) where nrandom_realizations is the number
    of random realizations interpolated

    Parameters
    ----------
    nvalidation_samples : integer
        The number of samples of the random realization used to compute the
        accuracy of the interpolant.
    """
    def __init__(self, gp, use_cholesky=False, alpha=0):
        self.gp = gp
        kernel_types = [RBF, Matern]
        # ignore white noise kernel as we want to interpolate the data
        self.kernel = extract_covariance_kernel(gp.kernel_, kernel_types)
        constant_kernel = extract_covariance_kernel(
            gp.kernel_, [ConstantKernel])
        if constant_kernel is not None:
            self.kernel = constant_kernel*self.kernel
        self.use_cholesky = use_cholesky
        # it is useful to specify alpha different to the one use to invert
        # Kernel marix at training data of gp
        self.alpha = alpha

    def fit(self, candidate_samples, rand_noise=None,
            ninterpolation_samples=500, nvalidation_samples=100,
            verbosity=0):
        """
        Construct interpolants of random realizations evaluated at the
        training data and at a new set of additional points
        """
        assert (ninterpolation_samples <=
                candidate_samples.shape[1] + self.gp.X_train_.T.shape[1]), (
                    ninterpolation_samples,
                    candidate_samples.shape[1] + self.gp.X_train_.T.shape[1])

        canonical_candidate_samples = self.gp.map_to_canonical(
            candidate_samples)
        canonical_candidate_samples = np.hstack(
            (self.gp.X_train_.T, canonical_candidate_samples))
        if self.use_cholesky is True:
            Kmatrix = self.kernel(canonical_candidate_samples.T)
            Kmatrix[np.diag_indices_from(Kmatrix)] += self.alpha
            init_pivots = np.arange(self.gp.X_train_.T.shape[1])
            # init_pivots = None
            L, pivots, error, chol_flag = pivoted_cholesky_decomposition(
                Kmatrix, ninterpolation_samples,
                init_pivots=init_pivots, pivot_weights=None,
                error_on_small_tol=False, return_full=False, econ=True)
            if verbosity > 0:
                print("Realization log10 cond num",
                      np.log10(np.linalg.cond(L.T.dot(L))))
            if chol_flag > 0:
                pivots = pivots[:-1]
                msg = "Number of samples used for interpolation "
                msg += f"{pivots.shape[0]} "
                msg += f"was less than requested {ninterpolation_samples}"
                print(msg)
                # then not all points requested were selected
                # because L became illconditioned. This usually means that no
                # more candidate samples are useful and that error in
                # interpolant will be small. Note  chol_flag > 0 even when
                # pivots.shape[0] == ninterpolation_samples. This means last
                # step of cholesky factorization triggered the incomplete flag

            self.L = L[pivots, :pivots.shape[0]]
            # print('Condition Number', np.linalg.cond(L.dot(L.T)))
            self.selected_canonical_samples = \
                canonical_candidate_samples[:, pivots]

            mask = np.ones(canonical_candidate_samples.shape[1], dtype=bool)
            mask[pivots] = False
            canonical_validation_samples = canonical_candidate_samples[
                :, mask]
            self.canonical_validation_samples = \
                canonical_validation_samples[:, :nvalidation_samples]
        else:
            assert (ninterpolation_samples + nvalidation_samples <=
                    candidate_samples.shape[1])
            self.selected_canonical_samples = \
                canonical_candidate_samples[:, :ninterpolation_samples]
            self.canonical_validation_samples = \
                canonical_candidate_samples[:, ninterpolation_samples:ninterpolation_samples+nvalidation_samples]
            Kmatrix = self.kernel(self.selected_canonical_samples.T)
            Kmatrix[np.diag_indices_from(Kmatrix)] += self.alpha
            self.L = np.linalg.cholesky(Kmatrix)

        samples = np.hstack(
            (self.selected_canonical_samples,
             self.canonical_validation_samples))
        # make last sample mean of gaussian process
        rand_noise = rand_noise[:samples.shape[1], :]
        rand_noise[:, -1] = np.zeros((rand_noise.shape[0]))
        vals = self.gp.predict_random_realization(
            self.gp.map_from_canonical(samples),
            rand_noise=rand_noise, truncated_svd=None,
            keep_normalized=True)
        self.train_vals = vals[:self.selected_canonical_samples.shape[1]]
        self.validation_vals = vals[self.selected_canonical_samples.shape[1]:]
        # Entries of the following should be size of alpha when
        # rand_noise[:, -1] = np.zeros((rand_noise.shape[0]))
        # print(self.train_vals[:, -1]-self.gp.y_train_[:, 0])

        # L_inv = np.linalg.inv(L.T)
        # L_inv = solve_triangular(L.T, np.eye(L.shape[0]))
        # self.K_inv_ = L_inv.dot(L_inv.T)
        # self.alpha_ = self.K_inv_.dot(self.train_vals)
        tmp = solve_triangular(self.L, self.train_vals, lower=True)
        self.alpha_ = solve_triangular(self.L.T, tmp, lower=False)

        approx_validation_vals = self.kernel(
            self.canonical_validation_samples.T,
            self.selected_canonical_samples.T).dot(self.alpha_)
        error = np.linalg.norm(
            approx_validation_vals-self.validation_vals, axis=0)/(
                np.linalg.norm(self.validation_vals, axis=0))
        # Error in interpolation of gp mean when
        # rand_noise[:, -1] = np.zeros((rand_noise.shape[0]))
        # print(np.linalg.norm((approx_validation_vals[:, -1]*self.gp._y_train_std+self.gp._y_train_mean)-self.gp(self.canonical_validation_samples)[:, 0])/np.linalg.norm(self.gp(self.canonical_validation_samples)[:, 0]))
        if verbosity > 0:
            print('Worst case relative interpolation error', error.max())
            print('Median relative interpolation error', np.median(error))

    def __call__(self, samples):
        canonical_samples = self.gp.map_to_canonical(samples)
        K_pred = self.kernel(
            canonical_samples.T, self.selected_canonical_samples.T)
        vals = K_pred.dot(self.alpha_)
        vals = self.gp._y_train_std*vals + self.gp._y_train_mean
        return vals


class AdaptiveGaussianProcess(GaussianProcess):
    def setup(self, func, sampler):
        self.func = func
        self.sampler = sampler

    def refine(self, num_samples):
        # new samples must be in user domain
        new_samples, chol_flag = self.sampler(num_samples)
        new_values = self.func(new_samples)
        assert new_values.shape[1] == 1  # must be scalar values QoI
        if hasattr(self, 'X_train_'):
            # get_training_samples returns samples in user space
            train_samples = self.get_training_samples()
            train_samples = np.hstack([train_samples, new_samples])
            train_values = np.vstack([self.y_train_, new_values])
        else:
            train_samples, train_values = new_samples, new_values
            # if self.var_trans is not None then when fit is called
            # train_samples are mapped to cannonical domain
        self.fit(train_samples, train_values)
        return chol_flag


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
    # v_sq = 1-np.trace(A_inv.dot(P))
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


def extract_kernel_attributes_for_integration(kernel):
    if extract_covariance_kernel(kernel, [WhiteKernel]) is not None:
        raise Exception('kernels with noise not supported')

    kernel_types = [
        RBF, Matern, UnivariateMarginalizedSquaredExponentialKernel,
        MultilevelKernel, MultifidelityPeerKernel]
    base_kernel = extract_covariance_kernel(kernel, kernel_types)

    constant_kernel = extract_covariance_kernel(kernel, [ConstantKernel])
    if constant_kernel is not None:
        kernel_var = constant_kernel.constant_value
    else:
        kernel_var = 1

    if isinstance(base_kernel, (MultilevelKernel, MultifidelityPeerKernel)):
        _kernels = [extract_covariance_kernel(k, kernel_types)
                    for k in base_kernel.kernels]
    else:
        _kernels = [base_kernel]

    for _kernel in _kernels:
        if not isinstance(_kernel, tuple(kernel_types[:3])):
            msg = f'GP Kernel type: {type(_kernel)} '
            msg += 'Only squared exponential kernel supported'
            raise Exception(msg)

    return base_kernel.length_scale, kernel_var


def extract_gaussian_process_attributes_for_integration(gp):
    length_scale, kernel_var = extract_kernel_attributes_for_integration(
        gp.kernel_)

    if not hasattr(gp, '_K_inv') or gp._K_inv is None:
        # scikit-learn < 0.24.2 has _K_inv
        # scikit-learn >= 0.24.2 does not
        L_inv = solve_triangular(gp.L_.T, np.eye(gp.L_.shape[0]), lower=False)
        K_inv = L_inv.dot(L_inv.T)
    else:
        K_inv = gp._K_inv.copy()

    transform_quad_rules = (not hasattr(gp, 'var_trans'))
    # gp.X_train_ will already be in the canonical space if var_trans is used
    x_train = gp.X_train_.T

    # correct for normalization of gaussian process training data
    # gp.y_train_ is normalized such that
    # y_train = gp._y_train_std*gp.y_train_ + gp._y_train_mean
    # shift must be accounted for in integration so do not add here
    y_train = gp._y_train_std*gp.y_train_
    kernel_var *= float(gp._y_train_std**2)
    K_inv /= gp._y_train_std**2
    return (x_train, y_train, K_inv, length_scale, kernel_var,
            transform_quad_rules)


def integrate_gaussian_process(gp, variable, return_full=False,
                               nquad_samples=50):
    """
    The alpha regularization parameter used to construct the gp stored
    in gp.alpha can significantly impact condition number of A_inv
    and thus the accuracy that can be obtained in estimates of integrals
    particularly associated with variance. However setting alpha too large
    will also limit the accuracy that can be achieved
    """
    (x_train, y_train, K_inv, kernel_length_scale, kernel_var,
     transform_quad_rules) = (
         extract_gaussian_process_attributes_for_integration(gp))

    result = integrate_gaussian_process_squared_exponential_kernel(
        x_train, y_train, K_inv, kernel_length_scale,
        kernel_var, variable, return_full, transform_quad_rules,
        nquad_samples, gp._y_train_mean)
    expected_random_mean, variance_random_mean, expected_random_var, \
        variance_random_var = result[:4]
    if return_full is True:
        return expected_random_mean, variance_random_mean, \
            expected_random_var, variance_random_var, result[4]

    return expected_random_mean, variance_random_mean, \
        expected_random_var, variance_random_var


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

    dist_func = partial(cdist, metric='sqeuclidean')
    dists_2d_x1_x2 = (xx_2d[0:1, :].T/lscale_ii-xx_2d[1:2, :].T/lscale_ii)**2
    dists_2d_x2_xtr = dist_func(xx_2d[1:2, :].T/lscale_ii, xtr.T/lscale_ii)
    lamda = np.exp(-.5*dists_2d_x1_x2.T-.5*dists_2d_x2_xtr.T).dot(ww_2d)

    dists_2d_x1_xtr = dist_func(xx_2d[0:1, :].T/lscale_ii, xtr.T/lscale_ii)
    # ntrain_samples = xtr.shape[1]
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


def get_gaussian_process_squared_exponential_kernel_1d_integrals(
        X_train, length_scale, variable, transform_quad_rules,
        nquad_samples=50, skip_xi_1=False):
    nvars = variable.num_vars()
    degrees = [nquad_samples]*nvars
    univariate_quad_rules = get_univariate_quadrature_rules_from_variable(
        variable, np.asarray(degrees)+1, True)

    lscale = np.atleast_1d(length_scale)
    # tau, u = 1, 1
    # ntrain_samples = X_train.shape[1]
    # P = np.ones((ntrain_samples, ntrain_samples))
    # lamda = np.ones(ntrain_samples)
    # Pi = np.ones((ntrain_samples, ntrain_samples))
    # xi_1, nu = 1, 1

    var_trans = AffineTransform(variable)

    tau_list, P_list, u_list, lamda_list = [], [], [], []
    Pi_list, nu_list, xi_1_list = [], [], []
    for ii in range(nvars):
        # TODO only compute quadrature once for each unique quadrature rules
        # But all quantities must be computed for all dimensions because
        # distances depend on either of both dimension dependent length scale
        # and training sample values
        # But others like u only needed to be computed for each unique
        # Quadrature rule and raised to the power equal to the number of
        # instances of a unique rule

        # Define distance function
        # dist_func = partial(cdist, metric='sqeuclidean')

        # Training samples of ith variable
        xtr = X_train[ii:ii+1, :]

        # Get 1D quadrature rule
        xx_1d, ww_1d = univariate_quad_rules[ii](degrees[ii]+1)
        if transform_quad_rules:
            xx_1d = var_trans.map_from_canonical_1d(xx_1d, ii)

        # Evaluate 1D integrals
        tau_ii, P_ii = integrate_tau_P(xx_1d, ww_1d, xtr, lscale[ii])
        # tau *= tau_ii
        # P *= P_ii

        u_ii, lamda_ii, Pi_ii, nu_ii = integrate_u_lamda_Pi_nu(
            xx_1d, ww_1d, xtr, lscale[ii])
        # u *= u_ii
        # lamda *= lamda_ii
        # Pi *= Pi_ii
        # nu *= nu_ii
        if skip_xi_1 is False:
            xi_1_ii = integrate_xi_1(xx_1d, ww_1d, lscale[ii])
        else:
            xi_1_ii = None
        # xi_1 *= xi_1_ii

        tau_list.append(tau_ii)
        P_list.append(P_ii)
        u_list.append(u_ii)
        lamda_list.append(lamda_ii)
        Pi_list.append(Pi_ii)
        nu_list.append(nu_ii)
        xi_1_list.append(xi_1_ii)

    return tau_list, P_list, u_list, lamda_list, Pi_list, nu_list, xi_1_list


def integrate_gaussian_process_squared_exponential_kernel(
        X_train,
        Y_train,
        K_inv,
        length_scale,
        kernel_var,
        variable,
        return_full=False,
        transform_quad_rules=False,
        nquad_samples=50,
        y_train_mean=0):
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

    variable : :class:`pyapprox.variable.IndependentMarginalsVariable`
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
    tau_list, P_list, u_list, lamda_list, Pi_list, nu_list, xi_1_list = \
        get_gaussian_process_squared_exponential_kernel_1d_integrals(
            X_train, length_scale, variable, transform_quad_rules,
            nquad_samples)
    tau = np.prod(np.array(tau_list), axis=0)
    P = np.prod(np.array(P_list), axis=0)
    u = np.prod(u_list)
    lamda = np.prod(np.array(lamda_list), axis=0)
    Pi = np.prod(np.array(Pi_list), axis=0)
    nu = np.prod(nu_list)
    xi_1 = np.prod(xi_1_list)

    # K_inv is inv(kernel_var*A). Thus multiply by kernel_var to get
    # Haylock formula
    A_inv = K_inv*kernel_var
    # No kernel_var because it cancels out because it appears in K (1/s^2)
    # and t (s^2)
    A_inv_y = A_inv.dot(Y_train)
    expected_random_mean = tau.dot(A_inv_y)
    expected_random_mean += y_train_mean

    varpi = compute_varpi(tau, A_inv)
    varsigma_sq = compute_varsigma_sq(u, varpi)
    variance_random_mean = variance_of_mean(kernel_var, varsigma_sq)

    A_inv_P = A_inv.dot(P)
    A_inv_tau = A_inv.dot(tau)
    v_sq = compute_v_sq(A_inv, P)
    # zeta = compute_zeta(Y_train, A_inv, P)
    zeta = compute_zeta_econ(Y_train, A_inv_y, A_inv_P)
    zeta += 2*tau.dot(A_inv_y)*y_train_mean+y_train_mean**2

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
    # adjust phi with unadjusted varrho
    phi += 2*y_train_mean*varrho+y_train_mean**2*varsigma_sq
    # now adjust varrho
    varrho += y_train_mean*varsigma_sq
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


def generate_gp_candidate_samples(nvars, num_candidate_samples,
                                  generate_random_samples, variable):
    if generate_random_samples is not None:
        num_halton_candidates = num_candidate_samples//2
        num_random_candidates = num_candidate_samples//2
    else:
        num_halton_candidates = num_candidate_samples
        num_random_candidates = 0
    if num_candidate_samples % 2 == 1:
        num_halton_candidates += 1

    # if variable is None:
    #     marginal_icdfs = None
    # else:
    #     # marginal_icdfs = [v.ppf for v in self.variable]
    #     from scipy import stats
    #     marginal_icdfs = []
    #     # spread QMC samples over entire domain. Range of variable
    #     # is used but not its PDF
    #     for v in variable.marginals():
    #         lb, ub = v.interval(1)
    #         if not np.isfinite(lb) or not np.isfinite(ub):
    #             lb, ub = v.interval(1-1e-6)
    #         marginal_icdfs.append(stats.uniform(lb, ub-lb).ppf)

    # candidate_samples = transformed_halton_sequence(
    #     marginal_icdfs, nvars, num_halton_candidates)
    from pyapprox.expdesign.low_discrepancy_sequences import sobol_sequence
    candidate_samples = sobol_sequence(nvars, num_halton_candidates, 1,
                                       variable)

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

    variable : :class:`pyapprox.variable.IndependentMarginalsVariable`
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

    def __init__(self, num_vars, num_candidate_samples, variable=None,
                 generate_random_samples=None, init_pivots=None,
                 nugget=0, econ=True, gen_candidate_samples=None,
                 var_trans=None):
        self.nvars = num_vars
        self.kernel_theta = None
        self.chol_flag = None
        self.variable = variable
        self.generate_random_samples = generate_random_samples
        if gen_candidate_samples is None:
            gen_candidate_samples = partial(
                generate_gp_candidate_samples, self.nvars,
                generate_random_samples=self.generate_random_samples,
                variable=self.variable)
        self.var_trans = var_trans
        self.set_candidate_samples(
            gen_candidate_samples(num_candidate_samples))
        self.set_weight_function(None)
        self.ntraining_samples = 0
        self.set_init_pivots(init_pivots)
        self.nugget = nugget
        self.econ = econ

    def set_candidate_samples(self, candidate_samples):
        if self.var_trans is not None:
            self.candidate_samples = self.var_trans.map_to_canonical(
                candidate_samples)
        else:
            self.candidate_samples = candidate_samples

    def add_nugget(self):
        self.Kmatrix[np.arange(self.Kmatrix.shape[0]),
                     np.arange(self.Kmatrix.shape[1])] += self.nugget

    def set_weight_function(self, weight_function):
        self.pivot_weights = None
        if self.var_trans is None or weight_function is None:
            self.weight_function = weight_function
        else:
            # weight function is applied in canonical_space
            def wt_function(x):
                return weight_function(
                    self.var_trans.map_from_canonical(x))
            self.weight_function = wt_function
        if self.weight_function is not None:
            self.pivot_weights = self.weight_function(self.candidate_samples)
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
                # assert np.allclose(np.diag(weights).dot(self.Kmatrix.dot(
                #    np.diag(weights))),
                #    weights[:, np.newaxis]*self.Kmatrix*weights)
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

        if self.var_trans is None:
            return new_samples, self.chol_flag
        return self.var_trans.map_from_canonical(
            new_samples), self.chol_flag


class AdaptiveCholeskyGaussianProcessFixedKernel(object):
    """
    Efficient implementation when Gaussian process kernel has no tunable
    hyper-parameters. Cholesky factor computed to generate training samples
    is reused for fitting
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
    # K_train_grad = np.zeros((ntrain_samples, ntrain_samples))
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
                      2*l**2*(l**2+2*s**2))))*(l**2*m+c*s**2-a*(l**2+s**2)))/(
                          l*(l**2+2*s**2)**(3/2))
    return term1


def gaussian_grad_P_offdiag_term2(xtr_ii, xtr_jj, lscale, mu, sigma):
    b, d, q, n, p = xtr_ii, xtr_jj, lscale, mu, sigma
    term2 = np.exp(-((-2*d*n*q**2+2*n**2*q**2+b**2*(p**2+q**2)+d **
                      2*(p**2+q**2)-2*b*(d*p**2+n*q**2))/(
                          2*q**2*(2*p**2+q**2))))
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

    variable : :class:`pyapprox.variable.IndependentMarginalsVariable`
        A set of independent univariate random variables. The tensor-product
        of the 1D PDFs yields the joint density :math:`\rho`. The bounds and
        CDFs of these variables are used to transform the Halton sequence used
        as the candidate set for the greedy generation of the initial guess.

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
                 ncandidate_samples, generate_random_samples, variable=None,
                 greedy_method='ivar', use_gauss_quadrature=False,
                 nugget=0):
        self.nvars = num_vars
        self.nquad_samples = nquad_samples
        self.greedy_method = greedy_method
        self.use_gauss_quadrature = use_gauss_quadrature
        self.pred_samples = generate_random_samples(self.nquad_samples)
        self.ncandidate_samples = ncandidate_samples
        self.variable = variable
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
            assert self.greedy_sampler.variable is not None
        else:
            self.objective = self.monte_carlo_objective
            self.objective_gradient = self.monte_carlo_objective_gradient

    def initialize_greedy_sampler(self):
        if self.greedy_method == 'chol':
            self.greedy_sampler = CholeskySampler(
                self.nvars, self.ncandidate_samples, self.variable,
                generate_random_samples=self.generate_random_samples)
        elif self.greedy_method == 'ivar':
            self.greedy_sampler = GreedyIntegratedVarianceSampler(
                self.nvars, self.nquad_samples, self.ncandidate_samples,
                self.generate_random_samples, self.variable,
                use_gauss_quadrature=self.use_gauss_quadrature, econ=True,
                nugget=self.nugget)
        else:
            msg = f'Incorrect greedy_method {self.greedy_method}'
            raise Exception(msg)

    def precompute_gauss_quadrature(self):
        degrees = [min(100, self.nquad_samples)]*self.nvars
        self.univariate_quad_rules = \
            get_univariate_quadrature_rules_from_variable(
                self.greedy_sampler.variable, np.asarray(degrees)+1, False)
        self.quad_rules = []
        for ii in range(self.nvars):
            xx_1d, ww_1d = self.univariate_quad_rules[ii](degrees[ii]+1)
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
        if self.greedy_sampler.variable is None:
            lbs, ubs = np.zeros(self.nvars), np.ones(self.nvars)
        else:
            variables = self.greedy_sampler.variable.marginals()
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
        # print(res)

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
                 ncandidate_samples, generate_random_samples, variable=None,
                 use_gauss_quadrature=False, econ=True,
                 compute_cond_nums=False, nugget=0, candidate_samples=None,
                 quadrature_rule=None):
        self.nvars = num_vars
        self.nquad_samples = nquad_samples
        self.variable = variable
        self.ntraining_samples = 0
        self.training_samples = np.empty((num_vars, self.ntraining_samples))
        self.generate_random_samples = generate_random_samples
        self.use_gauss_quadrature = use_gauss_quadrature
        self.econ = econ
        if quadrature_rule is None:
            self._quadrature_rule = self._monte_carlo_quadrature
        else:
            self._quadrature_rule = quadrature_rule

        if candidate_samples is None:
            self.candidate_samples = self._generate_candidate_samples(
                ncandidate_samples)
        else:
            assert ncandidate_samples == candidate_samples.shape[1]
            self.candidate_samples = candidate_samples
        self.nsamples_requested = []
        self.pivots = []
        self.cond_nums = []
        self.compute_cond_nums = compute_cond_nums
        self.init_pivots = None
        self.nugget = nugget
        self.initialize()
        self.best_obj_vals = []
        self.pred_samples = None

    def _monte_carlo_quadrature(self):
        xx = self.generate_random_samples(self.nquad_samples)
        ww = np.ones(xx.shape[1])/xx.shape[1]
        return xx, ww

    def _generate_candidate_samples(self, ncandidate_samples):
        return generate_gp_candidate_samples(
            self.nvars, ncandidate_samples, self.generate_random_samples,
            self.variable)

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
        xx, ww = self._quadrature_rule()
        k = self.kernel(xx.T, self.candidate_samples.T)
        self.tau = (ww[:, None]*k).sum(axis=0)
        assert self.tau.shape[0] == self.candidate_samples.shape[1]

        # Note because tau is simplified down to one integral instead of their
        # double used for u, it is possible for self.u - tau.dot(A_inv.dot(tau)
        # to be negative if tau is comptued using an inaccurate quadrature
        # rule. This is not important if using gauss quadrature
        # pred_samples2 = self.generate_random_samples(self.pred_samples.shape[1])
        # self.u = np.diag(
        #    self.kernel(self.pred_samples.T, pred_samples2.T)).mean()

    def get_univariate_quadrature_rule(self, ii):
        xx_1d, ww_1d = self.univariate_quad_rules[ii](self.degrees[ii]+1)
        return xx_1d, ww_1d

    def precompute_gauss_quadrature(self):
        nvars = self.variable.num_vars()
        length_scale = self.kernel.length_scale
        if np.isscalar(length_scale):
            length_scale = [length_scale]*nvars
        self.degrees = [self.nquad_samples]*nvars

        self.univariate_quad_rules = \
            get_univariate_quadrature_rules_from_variable(
                self.variable, np.asarray(self.degrees)+1, False)
        # dist_func = partial(cdist, metric='sqeuclidean')
        self.tau = 1

        for ii in range(self.nvars):
            # Get 1D quadrature rule
            xx_1d, ww_1d = self.get_univariate_quadrature_rule(ii)

            # Training samples of ith variable
            xtr = self.candidate_samples[ii:ii+1, :]
            lscale_ii = length_scale[ii]
            # dists_1d_x1_xtr = dist_func(
            #    xx_1d[:, np.newaxis]/lscale_ii, xtr.T/lscale_ii)
            # K = np.exp(-.5*dists_1d_x1_xtr)
            K = self.kernels_1d[ii](xx_1d[np.newaxis, :], xtr, lscale_ii)
            self.tau *= ww_1d.dot(K)

    def objective(self, new_sample_index):
        indices = np.concatenate(
            [self.pivots, [new_sample_index]]).astype(int)
        A = self.A[np.ix_(indices, indices)]
        try:
            L = np.linalg.cholesky(A)
        except RuntimeError as e:
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
            # ntraining_samples = self.ntraining_samples
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
            # training_samples = self.ntraining_samples
            obj_vals = self.vectorized_objective_vals_econ()
            pivot = np.argmin(obj_vals)
            obj_val = obj_vals[pivot]

        assert np.isfinite(obj_val)

        if self.L.shape[0] == 0:
            # self.L = np.atleast_2d(self.A[pivot, pivot])
            self.L = np.atleast_2d(np.sqrt(self.A[pivot, pivot]))
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
                except RuntimeError:
                    return -1, np.inf

            L_22 = np.sqrt(L_22_sq)
            self.L = np.block(
                [[self.L, np.zeros(L_12.shape)],
                 [L_12.T, L_22]])

        assert np.isfinite(self.candidate_y_2[pivot])
        self.y_1 = np.concatenate([self.y_1, [self.candidate_y_2[pivot]]])

        return pivot, obj_val

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

    def compute_A(self):
        self.active_candidates = np.ones(
            self.candidate_samples.shape[1], dtype=bool)
        self.A = self.kernel(self.candidate_samples.T)

    def set_kernel(self, kernel, kernels_1d=None):
        self.kernel = kernel
        self.base_kernel = extract_covariance_kernel(
            self.kernel, [RBF, Matern], view=True)
        if not (isinstance(self.base_kernel, RBF) or
                self.base_kernel.nu == np.inf) and self.use_gauss_quadrature:
            msg = "kernel {0} not supported".format(kernel)
            raise ValueError(msg)

        self.kernels_1d = kernels_1d
        if self.kernels_1d is None and self.use_gauss_quadrature:
            # TODO: remove kernels 1D and just create tensor product
            # kernel with this as a property.
            self.kernels_1d = [partial(matern_kernel_1d, np.inf)]*self.nvars

        base_kernel = extract_covariance_kernel(kernel, [Matern, RBF])
        if ((self.use_gauss_quadrature is True) and (self.nvars != 1) and
                (not (isinstance(base_kernel, (Matern, RBF)) or
                      (np.isfinite(base_kernel.nu))))):
            # TODO: To deal with sum kernel with noise, need to ammend
            # gradient computation which currently assumes no noise
            msg = f'GP Kernel type: {type(kernel)} '
            msg += 'Only squared exponential kernel supported when '
            msg += 'use_gauss_quadrature is True and nvars > 1'
            # TODO add other tensor product kernels
            raise Exception(msg)

        if self.use_gauss_quadrature:
            self.precompute_gauss_quadrature()
        else:
            self.precompute_monte_carlo()
        self.compute_A()
        # designs are better if a small nugget is added to the diagonal
        self.add_nugget()

    def add_nugget(self):
        self.A[np.arange(self.A.shape[0]), np.arange(self.A.shape[1])] += \
            self.nugget

    def set_init_pivots(self, init_pivots):
        assert len(self.pivots) == 0
        self.init_pivots = list(init_pivots)

    def update_training_samples(self, pivot):
        self.pivots.append(pivot)
        # new_sample = self.candidate_samples[:, pivot:pivot+1]
        self.training_samples = np.hstack(
            [self.training_samples,
             self.candidate_samples[:, pivot:pivot+1]])

    def __call__(self, nsamples, verbosity=0):
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
                #     # Switch of econ mode which struggles when condition
                #     # number is poor
                #     print('switching naive updating strategy on')
                #     self.refine = self.refine_naive
                #     pivot, obj_val = self.refine()
                #     if pivot < 0:
                #         flag = 1
                #         break
            if verbosity > 0:
                print(f'Iter: {nn}, Objective: {obj_val}')
            self.best_obj_vals.append(obj_val)

            self.update_training_samples(pivot)
            # print(f'Number of points generated {nn+1}')
            self.active_candidates[pivot] = False
            if self.compute_cond_nums is True:
                if self.econ:
                    self.cond_nums.append(np.linalg.cond(self.L)**2)
                else:
                    self.cond_nums.append(
                        np.linalg.cond(
                            self.A[np.ix_(self.pivots, self.pivots)]))
            # print(np.linalg.cond(
            #    self.A[np.ix_(self.pivots, self.pivots)]))

        new_samples = self.training_samples[:, ntraining_samples:]
        self.ntraining_samples = self.training_samples.shape[1]

        return new_samples, flag


def matern_gradient_wrt_samples(nu, query_sample, other_samples, length_scale):
    """
    Parameters
    ----------
    query_sample : np.ndarray (nvars, 1)

    other_samples : np.ndarray (nvars, nother_samples)

    length_scale : np.ndarray (nvars)
    """
    if type(length_scale) == np.ndarray:
        assert length_scale.shape[0] == query_sample.shape[0]
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
        grad = -5/3*K.T*tmp2*(np.sqrt(5)*dists.T+1)
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

    def _precompute_quadrature(self, xx, ww):
        K = self.kernel(xx.T, self.candidate_samples.T)
        P = K.T.dot(ww[:, np.newaxis]*K)
        return P

    def precompute_monte_carlo(self):
        xx, ww = self._quadrature_rule()
        assert ww.ndim == 1
        self.pred_samples = xx
        self.P = self._precompute_quadrature(xx, ww)

    def precompute_gauss_quadrature(self):
        self.degrees = [self.nquad_samples]*self.nvars
        length_scale = self.base_kernel.length_scale
        if np.isscalar(length_scale):
            length_scale = np.array([length_scale]*self.nvars)
        self.univariate_quad_rules = \
            get_univariate_quadrature_rules_from_variable(
                self.variable, np.array(self.degrees)+1, False)
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
        #     K = self.kernels_1d[ii](
        #         xx_1d[np.newaxis, :], xtr, length_scale[ii])
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
        L_22_sq = self.A[new_sample_index, new_sample_index] - L_12.T.dot(L_12)
        if L_22_sq <= 0:
            # Ill conditioning causes this issue
            return np.inf
        L_22 = np.sqrt(L_22_sq)
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
            # training_samples = self.ntraining_samples
            obj_vals = self.vectorized_objective_vals_econ()
            # obj_vals = self.objective_vals_econ() # hack
            pivot = np.argmin(obj_vals)
            obj_val = obj_vals[pivot]

        if not np.isfinite(obj_val):  # or obj_val < -1:
            # ill conditioning causes obj_val to go below -1 which should not
            # be possible
            return -1, np.inf

        if self.L_inv.shape[0] == 0:
            # self.L = np.atleast_2d(self.A[pivot, pivot])
            self.L = np.atleast_2d(np.sqrt(self.A[pivot, pivot]))
            self.L_inv = 1/self.L
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
            except RuntimeError:
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


class UnivariateMarginalizedGaussianProcess:
    """
    Parameters
    ----------
    mean : float
        The expectation of the gaussian process with respect to the random
        variables. If provided then the marginalized gaussian process will
        the main effect used in sensitivity analysis.
    """
    def __init__(self, kernel, train_samples, L_factor, train_values,
                 y_train_mean=0, y_train_std=1, mean=0):
        # the names are chosen to match names of _gpr from sklearn
        # so functions can be applied to both these methods in the same way
        self.kernel_ = kernel
        self.L_ = L_factor
        self.L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        self.K_inv_y = self.L_inv.dot(self.L_inv.T.dot(train_values))
        self.X_train_ = train_samples.T
        self.y_train_ = train_values
        assert train_samples.shape[0] == 1
        self._y_train_mean = y_train_mean
        self._y_train_std = y_train_std
        self._K_inv = None
        self.var_trans = None
        self.mean = mean

    def map_to_canonical(self, samples):
        if self.var_trans is not None:
            return self.var_trans.map_to_canonical(samples)
        return samples

    def set_variable_transformation(self, var_trans):
        self.var_trans = var_trans

    def __call__(self, samples, return_std=False):
        assert samples.shape[0] == 1
        canonical_samples = self.map_to_canonical(samples)
        K_pred = self.kernel_(canonical_samples.T, self.X_train_)
        mean = K_pred.dot(self.K_inv_y)
        mean = self._y_train_std*mean + self._y_train_mean - self.mean

        if not return_std:
            return mean

        pointwise_cov = self.kernel_.diag(canonical_samples.T)-np.sum(
            K_pred.dot(self.L_inv)**2, axis=1)
        return mean, self._y_train_std*np.sqrt(pointwise_cov)


class UnivariateMarginalizedSquaredExponentialKernel(RBF):
    def __init__(self, tau, u, length_scale, X_train):
        super().__init__(length_scale, length_scale_bounds='fixed')
        self.tau = tau
        self.u = u
        self.X_train_ = X_train
        assert self.tau.shape[0] == X_train.shape[0]

    def __call__(self, X, Y):
        assert np.allclose(Y, self.X_train_)
        assert Y is not None  # only used for prediction
        K = super().__call__(X, Y)
        return K*self.tau

    def diag(self, X):
        return super().diag(X)*self.u


def marginalize_gaussian_process(gp, variable, center=True):
    """
    Return all 1D marginal Gaussian process obtained after excluding all
    but a single variable
    """
    kernel_types = [RBF, Matern]
    kernel = extract_covariance_kernel(gp.kernel_, kernel_types)

    constant_kernel = extract_covariance_kernel(gp.kernel_, [ConstantKernel])
    if constant_kernel is not None:
        kernel_var = constant_kernel.constant_value
    else:
        kernel_var = 1

    # Warning  extract_gaussian_process scales kernel_var by gp.y_train_std**2
    x_train, y_train, K_inv, kernel_length_scale, kernel_var, \
        transform_quad_rules = \
        extract_gaussian_process_attributes_for_integration(gp)

    # x_train = gp.X_train_.T
    # kernel_length_scale = kernel.length_scale
    # transform_quad_rules = (not hasattr(gp, 'var_trans'))
    L_factor = gp.L_.copy()

    tau_list, P_list, u_list, lamda_list, Pi_list, nu_list, __ = \
        get_gaussian_process_squared_exponential_kernel_1d_integrals(
            x_train, kernel_length_scale, variable, transform_quad_rules,
            skip_xi_1=True)

    if center is True:
        A_inv = K_inv*kernel_var
        tau = np.prod(np.array(tau_list), axis=0)
        A_inv_y = A_inv.dot(y_train)
        shift = tau.dot(A_inv_y)
        shift += gp._y_train_mean
    else:
        shift = 0

    kernel_var /= float(gp._y_train_std**2)

    length_scale = np.atleast_1d(kernel_length_scale)
    nvars = variable.num_vars()
    marginalized_gps = []
    for ii in range(nvars):
        tau = np.prod(np.array(tau_list)[:ii], axis=0)*np.prod(
            np.array(tau_list)[ii+1:], axis=0)
        u = np.prod(u_list[:ii])*np.prod(u_list[ii+1:])
        assert np.isscalar(kernel_var)
        kernel = kernel_var*UnivariateMarginalizedSquaredExponentialKernel(
            tau, u, length_scale[ii], gp.X_train_[:, ii:ii+1])
        # undo kernel_var *= gp._y_train_std**2 in extact_gaussian_process_attr
        gp_ii = UnivariateMarginalizedGaussianProcess(
            kernel, gp.X_train_[:, ii:ii+1].T, L_factor, gp.y_train_,
            gp._y_train_mean, gp._y_train_std, mean=shift)
        if hasattr(gp, 'var_trans'):
            variable_ii = IndependentMarginalsVariable(
                [gp.var_trans.variable.marginals()[ii]])
            var_trans_ii = AffineTransform(variable_ii)
            gp_ii.set_variable_transformation(var_trans_ii)
        marginalized_gps.append(gp_ii)
    return marginalized_gps


def compute_conditional_P(xx_1d, ww_1d, xtr, lscale_ii):
    # Get 2D tensor product quadrature rule
    xx_2d = cartesian_product([xx_1d]*2)
    ww_2d = outer_product([ww_1d]*2)

    # ntrain_samples = xtr.shape[1]
    dist_func = partial(cdist, metric='sqeuclidean')
    dists_2d_x2_xtr = dist_func(xx_2d[1:2, :].T/lscale_ii, xtr.T/lscale_ii)
    dists_2d_x1_xtr = dist_func(xx_2d[0:1, :].T/lscale_ii, xtr.T/lscale_ii)
    P = np.exp(-.5*dists_2d_x1_xtr).T.dot(ww_2d[:, np.newaxis]*np.exp(
        -.5*dists_2d_x2_xtr))
    return P


def compute_expected_sobol_indices(gp, variable, interaction_terms,
                                   nquad_samples=50):
    """
    The alpha regularization parameter used to construct the gp stored
    in gp.alpha can significantly impact condition number of A_inv
    and thus the accuracy that can be obtained in estimates of integrals
    particularly associated with variance. However setting alpha too large
    will also limit the accuracy that can be achieved
    """
    x_train, y_train, K_inv, lscale, kernel_var, transform_quad_rules = \
        extract_gaussian_process_attributes_for_integration(gp)
    result = _compute_expected_sobol_indices(
        gp, variable, interaction_terms, nquad_samples, x_train, y_train,
        K_inv, lscale, kernel_var, transform_quad_rules, gp._y_train_mean)
    return result


def _compute_expected_sobol_indices(
        gp, variable, interaction_terms, nquad_samples, x_train, y_train,
        K_inv, lscale, kernel_var, transform_quad_rules, y_train_mean=0):
    assert np.isscalar(y_train_mean) or y_train_mean.shape == (1,)
    tau_list, P_list, u_list, lamda_list, Pi_list, nu_list, _ = \
        get_gaussian_process_squared_exponential_kernel_1d_integrals(
            x_train, lscale, variable, transform_quad_rules,
            nquad_samples=nquad_samples, skip_xi_1=True)

    lscale = np.atleast_1d(lscale)  # for 1D gps
    nvars = variable.num_vars()
    degrees = [nquad_samples]*nvars
    univariate_quad_rules = get_univariate_quadrature_rules_from_variable(
        variable, np.asarray(degrees)+1, True)

    var_trans = AffineTransform(variable)

    P_mod_list = []
    for ii in range(nvars):
        # Training samples of ith variable
        xtr = x_train[ii:ii+1, :]
        xx_1d, ww_1d = univariate_quad_rules[ii](degrees[ii]+1)
        if transform_quad_rules:
            xx_1d = var_trans.map_from_canonical_1d(xx_1d, ii)
        P_mod_list.append(compute_conditional_P(xx_1d, ww_1d, xtr, lscale[ii]))

    cond_num = np.linalg.cond(K_inv)
    # print("Kernel log10 Cond Num", np.log10(cond_num))
    if cond_num > 1e11:
        msg = "Condition number of kernel matrix is to high."
        msg += f" Log10 condition number is {np.log10(cond_num)}. "
        msg += "Increase alpha"
        # raise RuntimeError(msg)

    A_inv = K_inv*kernel_var
    # print('cond num', np.linalg.cond(A_inv))
    tau = np.prod(np.array(tau_list), axis=0)
    u = np.prod(np.array(u_list), axis=0)
    varpi = compute_varpi(tau, A_inv)
    varsigma_sq = compute_varsigma_sq(u, varpi)
    P = np.prod(np.array(P_list), axis=0)
    A_inv_P = A_inv.dot(P)
    v_sq = compute_v_sq(A_inv, P)

    A_inv_y = A_inv.dot(y_train)
    expected_random_mean = tau.dot(A_inv_y)
    expected_random_mean += y_train_mean
    variance_random_mean = np.empty_like(expected_random_mean)
    expected_random_var = np.empty_like(expected_random_mean)
    for ii in range(y_train.shape[1]):
        variance_random_mean[ii] = variance_of_mean(kernel_var, varsigma_sq)
        zeta_ii = compute_zeta_econ(
            y_train[:, ii:ii+1], A_inv_y[:, ii:ii+1], A_inv_P)
        zeta_ii += 2*tau.dot(A_inv_y[:, ii:ii+1])*y_train_mean+y_train_mean**2
        expected_random_var[ii] = mean_of_variance(
            zeta_ii, v_sq, kernel_var, expected_random_mean[ii],
            variance_random_mean[ii])

    assert interaction_terms.max() == 1
    # add indices need to compute main effects. These may already be
    # in interaction terms but cost of recomputing them is negligible
    # and avoids extra book keeping
    total_effect_interaction_terms = np.ones((nvars, nvars))-np.eye(nvars)
    myinteraction_terms = np.hstack(
        (interaction_terms, total_effect_interaction_terms))
    unnormalized_interaction_values = np.empty(
        (myinteraction_terms.shape[1], y_train.shape[1]))
    for jj in range(myinteraction_terms.shape[1]):
        index = myinteraction_terms[:, jj]
        P_p, U_p = 1, 1
        for ii in range(nvars):
            if index[ii] == 1:
                P_p *= P_list[ii]
                U_p *= 1
            else:
                P_p *= P_mod_list[ii]
                U_p *= u_list[ii]
        trace_A_inv_Pp = np.sum(A_inv*P_p)  # U_p-np.trace(A_inv.dot(P_p))
        for ii in range(y_train.shape[1]):
            v_sq_ii = U_p-trace_A_inv_Pp
            zeta_ii = A_inv_y[:, ii:ii+1].T.dot(P_p.dot(A_inv_y[:, ii:ii+1]))
            zeta_ii += 2*tau.dot(A_inv_y[:, ii:ii+1])*y_train_mean +\
                y_train_mean**2
            unnormalized_interaction_values[jj, ii] = mean_of_variance(
                zeta_ii, v_sq_ii, kernel_var, expected_random_mean[ii],
                variance_random_mean[ii])
    unnormalized_total_effect_values = \
        unnormalized_interaction_values[interaction_terms.shape[1]:]
    unnormalized_interaction_values = \
        unnormalized_interaction_values[:interaction_terms.shape[1]]

    II = argsort_indices_leixographically(interaction_terms)
    unnormalized_sobol_indices = unnormalized_interaction_values.copy()
    sobol_indices_dict = dict()
    for ii in range(II.shape[0]):
        index = interaction_terms[:, II[ii]]
        active_vars = np.where(index > 0)[0]
        nactive_vars = index.sum()
        sobol_indices_dict[tuple(active_vars)] = II[ii]
        if nactive_vars > 1:
            for jj in range(nactive_vars-1):
                indices = combinations(active_vars, jj+1)
                for key in indices:
                    unnormalized_sobol_indices[II[ii]] -= \
                        unnormalized_sobol_indices[sobol_indices_dict[key]]

    # print(unnormalized_sobol_indices.shape)
    # print(np.sum(unnormalized_sobol_indices, axis=0))
    # print(expected_random_var)

    if np.any(unnormalized_sobol_indices < 0):
        msg = "Some Sobol indices were negative. "
        msg += "Likely due to ill conditioning "
        msg += "of GP kernel. Try increaseing alpha"
        raise RuntimeError(msg)

    if np.any(expected_random_var < 0):
        msg = "Some expected variances were negative. "
        msg += "Likely due to ill conditioning "
        msg += "of GP kernel. Samlpes used to generate GP realization "
        msg += "are likely ill conditioned. Try increasing alpha "
        msg += "(not used to fit gp) but used to generate realizations"
        msg += "or reduce ninterpolation_samples"
        raise RuntimeError(msg)

    if np.any(unnormalized_sobol_indices.max(axis=0) > expected_random_var+1e-6):
        print('relative max diff',
              ((unnormalized_sobol_indices.max(axis=0)-expected_random_var)/
               expected_random_var).max())
        msg = "Some Sobol indices were larger than the variance. "
        msg += "Likely due to ill conditioning "
        msg += "of GP kernel. Samlpes used to generate GP realization "
        msg += "are likely ill conditioned. Try increasing alpha "
        msg += "(not used to fit gp) but used to generate realizations"
        msg += "or reduce ninterpolation_samples"
        raise RuntimeError(msg)
        # import warnings
        # warnings.warn(msg)

    total_effect_indices = (
        1-unnormalized_total_effect_values/expected_random_var)
    if np.any(total_effect_indices < 0):
        print('relative max diff',
              ((unnormalized_total_effect_values.max(axis=0) -
                expected_random_var)/expected_random_var).max())
        print(total_effect_indices)
        msg = "Some total effect values were negative. "
        msg += "Likely due to ill conditioning "
        msg += "of GP kernel. Samlpes used to generate GP realization "
        msg += "are likely ill conditioned. Try increasing alpha "
        msg += "(not used to fit gp) but used to generate realizations"
        msg += "or reduce ninterpolation_samples"
        raise RuntimeError(msg)
        # import warnings
        # warnings.warn(msg)

    return unnormalized_sobol_indices/expected_random_var, \
        total_effect_indices, \
        expected_random_mean, expected_random_var


def generate_gp_realizations(gp, ngp_realizations, ninterpolation_samples,
                             nvalidation_samples, ncandidate_samples,
                             variable, use_cholesky=True, alpha=0,
                             verbosity=0):
    rand_noise = np.random.normal(
        0, 1, (ngp_realizations, ninterpolation_samples+nvalidation_samples)).T
    gp_realizations = RandomGaussianProcessRealizations(gp, use_cholesky,
                                                        alpha)
    if use_cholesky is True:
        generate_random_samples = partial(
            generate_independent_random_samples, variable)
    else:
        generate_random_samples = None
    from pyapprox.surrogates.gaussianprocess.gaussian_process import (
        generate_gp_candidate_samples)
    candidate_samples = generate_gp_candidate_samples(
        variable.num_vars(), ncandidate_samples, generate_random_samples,
        variable)
    gp_realizations.fit(
        candidate_samples, rand_noise, ninterpolation_samples,
        nvalidation_samples, verbosity)
    fun = gp_realizations
    return fun
