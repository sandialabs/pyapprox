import unittest
import numpy as np
from scipy import stats
from functools import partial

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.bayes.laplace import (
    laplace_posterior_approximation_for_linear_models, laplace_evidence)
from pyapprox.bayes.metropolis import (
    MetropolisMCMCVariable, compute_mvn_cholesky_based_data, mvn_log_pdf)
from pyapprox.surrogates.interp.monomial import (
    univariate_monomial_basis_matrix)
from pyapprox.util.utilities import (
    cartesian_product, outer_product, check_gradients)
from pyapprox.surrogates.orthopoly.quadrature import gauss_hermite_pts_wts_1D


def _loglike_fun_linear_model(Amatrix, obs, noise_stdev, x, return_grad=False):
    nobs = Amatrix.shape[0]
    if x.ndim == 1:
        x = x[:, None]
    residual = (Amatrix.dot(x)-obs)
    llike = -(nobs*np.log(np.pi)/2+nobs/2*np.log(noise_stdev**2) +
              1/(2*noise_stdev**2)*np.sum(residual**2))
    if not return_grad:
        return llike
    llike_grad = -1/(noise_stdev**2)*residual.T.dot(Amatrix)
    return llike, llike_grad


def _setup_gaussian_linear_inverse_problem(
        nobs, nvars, noise_stdev, prior_mean, prior_std):
    x = np.linspace(0., 9., nobs)
    Amatrix = univariate_monomial_basis_matrix(nvars-1, x)
    true_sample = np.random.normal(0, 1, (nvars, 1))
    obs = Amatrix.dot(true_sample) + noise_stdev*np.random.randn(nobs)[:, None]
    prior_variable = IndependentMarginalsVariable(
        [stats.norm(prior_mean, prior_std)]*nvars)
    loglike_fun = partial(_loglike_fun_linear_model, Amatrix, obs, noise_stdev)

    prior_mean = prior_variable.get_statistics("mean")
    prior_cov = np.diag(prior_variable.get_statistics("std")[:, 0]**2)
    prior_hessian = np.linalg.inv(prior_cov)
    noise_covariance = noise_stdev**2*np.eye(nobs)
    noise_covariance_inv = np.linalg.inv(noise_covariance)
    post_mean, post_covariance = \
        laplace_posterior_approximation_for_linear_models(
            Amatrix, prior_mean, prior_hessian,
            noise_covariance_inv, obs)
    return (Amatrix, obs, true_sample, prior_variable, loglike_fun, post_mean,
            post_covariance)


class ExponentialQuarticLogLikelihoodModel(object):
    def __init__(self):
        self.a = 3.0

    def loglikelihood_function(self, x):
        value = -(0.1*x[0]**4 + 0.5*(2.*x[1]-x[0]**2)**2)
        return value

    def gradient(self, x):
        assert x.ndim == 2
        assert x.shape[1] == 1
        grad = -np.array([12./5.*x[0]**3-4.*x[0]*x[1],
                          4.*x[1]-2.*x[0]**2])
        return grad

    def __call__(self, x, jac=False):
        vals = np.array([self.loglikelihood_function(x)]).T
        if not jac:
            return vals
        return vals, self.gradient(x)


class TestMetropolis(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_mvnpdf(self):
        nvars = 3
        m = np.random.normal(0, 1, (nvars, 1))
        C = np.random.normal(0, 1, (nvars, nvars))
        C = C.T.dot(C)

        L, L_inv, logdet = compute_mvn_cholesky_based_data(C)
        L = np.linalg.cholesky(C)
        assert np.allclose(L_inv, np.linalg.inv(L))
        logdet = 2*np.log(np.diag(L)).sum()
        assert np.allclose(logdet, np.linalg.slogdet(C)[1])

        xx = np.random.uniform(-3, 3, (nvars, 100))
        assert np.allclose(
            np.exp(mvn_log_pdf(xx, m, C)),
            stats.multivariate_normal(m.squeeze(), C).pdf(xx.T))

    def test_loglike_fun_linear_model(self):
        nvars = 2
        nobs = 3  # number of observations
        noise_stdev = 2  # standard deviation of noise
        prior_mean, prior_std = 0, 1
        (Amatrix, obs, true_sample, prior_variable, loglike_fun,
         exact_post_mean, exact_post_covariance) = (
             _setup_gaussian_linear_inverse_problem(
                nobs, nvars, noise_stdev, prior_mean, prior_std))

        evidence = laplace_evidence(
                lambda x: np.atleast_1d(np.exp(loglike_fun(x))),
                prior_variable.pdf, exact_post_covariance, exact_post_mean)
        # print(evidence, 'evidence')
        xx1d, ww1d = gauss_hermite_pts_wts_1D(200)
        xx = cartesian_product([xx1d]*2)
        ww = outer_product([ww1d]*2)
        lp_vals = np.array([np.exp(loglike_fun(x)) for x in xx.T])
        const = lp_vals.dot(ww)
        assert np.allclose(const, evidence, rtol=1e-5)

    def _check_logpost_gradients(self, prior_variable):
        nvars = prior_variable.num_vars()
        nobs = 3  # number of observations
        noise_stdev = 2  # standard deviation of noise
        prior_mean, prior_std = 0, 1
        (Amatrix, obs, true_sample, _, loglike_fun,
         exact_post_mean, exact_post_covariance) = (
             _setup_gaussian_linear_inverse_problem(
                nobs, nvars, noise_stdev, prior_mean, prior_std))

        fd_eps = np.logspace(-13, 0, 14)[::-1]*0.1

        test_sample = prior_variable.get_statistics("mean")
        errors = check_gradients(loglike_fun, True, test_sample, fd_eps=fd_eps)
        print(errors.min()/errors.max())
        assert errors.min()/errors.max() < 3e-6

        mcmc_variable = MetropolisMCMCVariable(prior_variable, loglike_fun)

        from functools import partial
        errors = check_gradients(
            partial(mcmc_variable._variable.pdf, log=True),
            mcmc_variable._logprior_grad, test_sample, rel=False,
            fd_eps=fd_eps)
        assert errors.min()/errors.max() < 2e-7 or errors.min() == 0
        errors = check_gradients(
            mcmc_variable._log_bayes_numerator, True, test_sample,
            fd_eps=fd_eps)
        print(errors.min()/errors.max())
        assert errors.min()/errors.max() < 4e-7

    def test_logpost_gradients(self):
        self._check_logpost_gradients(
            IndependentMarginalsVariable([stats.norm(1, 2)]*3))

        self._check_logpost_gradients(
            IndependentMarginalsVariable([stats.uniform(1, 2)]*3))

        self._check_logpost_gradients(
            IndependentMarginalsVariable([stats.beta(2, 3, 0, 1)]*3))

    def _check_mcmc_variable(self, nvars, algorithm, method_opts):
        np.random.seed(3)
        nsamples = 5000
        burn_fraction = 0.2
        nsamples_per_tuning = 20
        nobs = 3  # number of observations
        noise_stdev = 2  # standard deviation of noise
        # init_proposal_cov = np.eye(nvars)
        init_proposal_cov = None
        prior_mean, prior_std = 0, 1

        (Amatrix, obs, true_sample, prior_variable, loglike_fun,
         exact_post_mean, exact_post_covariance) = (
             _setup_gaussian_linear_inverse_problem(
                 nobs, nvars, noise_stdev, prior_mean, prior_std))

        mcmc_variable = MetropolisMCMCVariable(
            prior_variable, loglike_fun,
            nsamples_per_tuning=nsamples_per_tuning,
            algorithm=algorithm, burn_fraction=burn_fraction,
            method_opts=method_opts, init_proposal_cov=init_proposal_cov)
        map_sample = mcmc_variable.maximum_aposteriori_point(
            prior_variable.get_statistics("mean"))
        assert np.allclose(map_sample, exact_post_mean)

        mcmc_samples = mcmc_variable.rvs(nsamples, map_sample)
        print(mcmc_samples.shape)
        acceptance_ratio = mcmc_variable._acceptance_rate
        print('acceptance ratio', acceptance_ratio)
        # assert acceptance_ratio >= 0.2 and acceptance_ratio < 0.41
        print(exact_post_mean[:, 0], "EXACT Mean")
        print(mcmc_samples.mean(axis=1), "Samples Mean")
        print(exact_post_covariance, "EXACT COV")
        print(np.cov(mcmc_samples, ddof=1), "Samples COV")
        print("mean error", exact_post_mean[:, 0] - mcmc_samples.mean(axis=1))
        print("cov_error", exact_post_covariance-np.cov(mcmc_samples))
        # import matplotlib.pyplot as plt
        # from pyapprox.variables.density import NormalDensity
        # ax = plt.subplots(1, 1)[1]
        # pdf = NormalDensity(exact_post_mean, exact_post_covariance)
        # pdf.plot_contours(ax=ax)
        # ax.plot(*mcmc_samples, 'o')
        # plt.show()
        assert np.allclose(
            exact_post_mean[:, 0], mcmc_samples.mean(axis=1), atol=4.0e-2)
        assert np.allclose(
            exact_post_covariance, np.cov(mcmc_samples, ddof=1), atol=4.5e-2)

    def test_mcmc_variable(self):
        def dram_opts(nvars):
            cov_scaling = 1
            nugget = 1e-8
            return {"cov_scaling": cov_scaling, "nugget": nugget,
                    "sd": 2.4**2/nvars*2}
        hmc_opts = {"num_steps": 5, "epsilon": 3e-1}  # 2D distribution
        # hmc_opts = {"num_steps": 50, "epsilon": 0.001}  # 4D
        test_cases = [
            [2, "DRAM", dram_opts(2)],
            [2, "hmc", hmc_opts]
        ]
        for test_case in test_cases[-1:]:
            self._check_mcmc_variable(*test_case)


if __name__ == '__main__':
    metropolis_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMetropolis)
    unittest.TextTestRunner(verbosity=2).run(metropolis_test_suite)
