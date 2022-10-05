import unittest
import numpy as np
from scipy import stats

from pyapprox.bayes.laplace import (
    laplace_posterior_approximation_for_linear_models, laplace_evidence)
from pyapprox.bayes.metropolis import MetropolisMCMCVariable
from pyapprox.surrogates.interp.monomial import (
    univariate_monomial_basis_matrix)
from pyapprox.util.utilities import cartesian_product, outer_product
from pyapprox.surrogates.orthopoly.quadrature import gauss_hermite_pts_wts_1D


class TestMetropolis(unittest.TestCase):

    def test_dram(self):
        np.random.seed(3)
        nsamples = 20000
        burn_fraction = 0.2
        nsamples_per_tuning = 20
        # algorithm = "metropolis"
        # method_opts = {}
        algorithm = "DRAM"
        nugget = 1e-8
        cov_scaling = 0.1
        method_opts = {"cov_scaling": cov_scaling, "nugget": nugget}
        nvars = 4
        nobs = 3  # number of observations
        noise_stdev = 2  # standard deviation of noise

        # init_proposal_cov = np.eye(nvars)
        init_proposal_cov = None

        x = np.linspace(0., 9., nobs)
        Amatrix = univariate_monomial_basis_matrix(nvars-1, x)
        true_sample = np.random.normal(0, 1, (nvars, 1))

        def model(x):
            return np.dot(Amatrix, x)

        obs = model(true_sample) + noise_stdev*np.random.randn(nobs)[:, None]

        prior_mean = np.zeros((nvars, 1))
        prior_cov = np.eye(nvars)
        prior_hessian = np.linalg.inv(prior_cov)
        noise_covariance = noise_stdev**2*np.eye(nobs)
        noise_covariance_inv = np.linalg.inv(noise_covariance)
        exact_mean, exact_covariance = \
            laplace_posterior_approximation_for_linear_models(
                Amatrix, prior_mean, prior_hessian,
                noise_covariance_inv, obs)

        def loglike_fun(x):
            if x.ndim == 1:
                x = x[:, None]
            llike = -(nobs*np.log(np.pi)/2+nobs/2*np.log(noise_stdev**2) +
                      1/(2*noise_stdev**2)*np.sum((model(x)-obs)**2))
            return llike

        from pyapprox.variables.joint import IndependentMarginalsVariable
        prior_variable = IndependentMarginalsVariable([stats.norm(0, 1)]*nvars)

        if nvars == 2:
            evidence = laplace_evidence(
                lambda x: np.atleast_1d(np.exp(loglike_fun(x))),
                prior_variable.pdf, exact_covariance, exact_mean)
            # print(evidence, 'evidence')

            # x = np.random.normal(0, 1, (nvars, 1))
            # log_post_exact = mvn_log_pdf(x, exact_mean, exact_covariance)
            xx1d, ww1d = gauss_hermite_pts_wts_1D(200)
            xx = cartesian_product([xx1d]*2)
            ww = outer_product([ww1d]*2)
            lp_vals = np.array([np.exp(loglike_fun(x)) for x in xx.T])
            const = lp_vals.dot(ww)
            # print(const, "CONST")
            assert np.allclose(const, evidence, rtol=1e-5)

        mcmc_variable = MetropolisMCMCVariable(
            prior_variable, loglike_fun,
            nsamples_per_tuning=nsamples_per_tuning,
            algorithm=algorithm, burn_fraction=burn_fraction,
            method_opts=method_opts, init_proposal_cov=init_proposal_cov)
        mcmc_samples = mcmc_variable.rvs(nsamples)

        acceptance_ratio = mcmc_variable._acceptance_rate
        print('acceptance ratio', acceptance_ratio)
        assert acceptance_ratio >= 0.2 and acceptance_ratio < 0.41
        # print(exact_mean[:, 0])
        # print(exact_covariance, "EXACT COV")
        # print(mcmc_samples.mean(axis=1))
        # # print(np.cov(mcmc_samples))
        # print("mean error", exact_mean[:, 0] - mcmc_samples.mean(axis=1))
        # print("cov_error", exact_covariance-np.cov(mcmc_samples))

        assert np.allclose(
            exact_mean[:, 0], mcmc_samples.mean(axis=1), atol=2.5e-2)
        assert np.allclose(exact_covariance, np.cov(mcmc_samples), atol=4.5e-2)


if __name__ == '__main__':
    metropolis_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMetropolis)
    unittest.TextTestRunner(verbosity=2).run(metropolis_test_suite)
