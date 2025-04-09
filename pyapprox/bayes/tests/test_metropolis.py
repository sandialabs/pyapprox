import unittest
import numpy as np
from scipy import stats
from functools import partial

from pyapprox.variables.joint import IndependentMarginalsVariable

from pyapprox.surrogates.univariate.base import Monomial1D
from pyapprox.bayes.metropolis import (
    MetropolisMCMCVariable,
    compute_mvn_cholesky_based_data,
    mvn_log_pdf,
)
from pyapprox.bayes.laplace import DenseMatrixLaplacePosteriorApproximation
from pyapprox.bayes.likelihood import ModelBasedGaussianLogLikelihood
from pyapprox.bayes.tests.test_likelihood import Linear1DRegressionModel


def _setup_gaussian_linear_inverse_problem(
    nobs, nvars, noise_stdev, prior_mean, prior_std
):
    # design = np.linspace(0.0, 9.0, nobs)[None, :]
    design = np.linspace(-1.0, 1.0, nobs)[None, :]
    obs_model = Linear1DRegressionModel(design, nvars - 1)
    noise_cov = noise_stdev**2 * np.eye(nobs)
    loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)
    true_sample = np.full((nvars, 1), 0.4)
    obs = loglike.rvs(true_sample)
    loglike.set_observations(obs)
    prior = IndependentMarginalsVariable(
        [stats.norm(prior_mean, prior_std)] * nvars
    )
    from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian

    prior = DenseCholeskyMultivariateGaussian(
        np.full((nvars, 1), prior_mean), np.eye(nvars) * prior_std**2
    )
    laplace = DenseMatrixLaplacePosteriorApproximation(
        obs_model.matrix(), prior.mean(), prior.covariance(), noise_cov
    )
    laplace.compute(obs)
    return (true_sample, prior, loglike, laplace)


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
        logdet = 2 * np.log(np.diag(L)).sum()
        assert np.allclose(logdet, np.linalg.slogdet(C)[1])

        xx = np.random.uniform(-3, 3, (nvars, 100))
        assert np.allclose(
            np.exp(mvn_log_pdf(xx, m, C)),
            stats.multivariate_normal(m.squeeze(), C).pdf(xx.T),
        )

    def _check_mcmc_variable(self, nvars, algorithm, method_opts):
        np.random.seed(3)
        nsamples = 5000
        burn_fraction = 0.2
        nsamples_per_tuning = 20
        nobs = 4  # number of observations
        noise_stdev = np.sqrt(0.3)  # standard deviation of noise
        # init_proposal_cov = np.eye(nvars)
        init_proposal_cov = None
        prior_mean, prior_std = 0, 1

        (true_sample, prior, loglike, laplace) = (
            _setup_gaussian_linear_inverse_problem(
                nobs, nvars, noise_stdev, prior_mean, prior_std
            )
        )

        mcmc_variable = MetropolisMCMCVariable(
            prior,
            loglike,
            nsamples_per_tuning=nsamples_per_tuning,
            algorithm=algorithm,
            burn_fraction=burn_fraction,
            method_opts=method_opts,
            init_proposal_cov=init_proposal_cov,
        )
        map_sample = mcmc_variable.maximum_aposteriori_point(prior.mean())
        print(map_sample, laplace.posterior_mean())
        assert np.allclose(map_sample, laplace.posterior_mean())

        mcmc_samples = mcmc_variable.rvs(nsamples, map_sample)
        print(mcmc_samples.shape)
        acceptance_ratio = mcmc_variable._acceptance_rate
        print("acceptance ratio", acceptance_ratio)
        # assert acceptance_ratio >= 0.2 and acceptance_ratio < 0.41
        print(laplace.posterior_mean()[:, 0], "EXACT Mean")
        print(mcmc_samples.mean(axis=1), "Samples Mean")
        print(laplace.posterior_covariance(), "EXACT COV")
        print(np.cov(mcmc_samples, ddof=1), "Samples COV")
        print(
            "mean error",
            laplace.posterior_mean()[:, 0] - mcmc_samples.mean(axis=1),
        )
        print(
            "cov_error", laplace.posterior_covariance() - np.cov(mcmc_samples)
        )
        assert np.allclose(
            mcmc_samples.mean(axis=1),
            laplace.posterior_mean()[:, 0],
            atol=4.0e-2,
        )
        assert np.allclose(
            np.cov(mcmc_samples, ddof=1),
            laplace.posterior_covariance(),
            atol=4.5e-2,
        )

    def test_mcmc_variable(self):
        def dram_opts(nvars):
            cov_scaling = 1
            nugget = 1e-8
            return {
                "cov_scaling": cov_scaling,
                "nugget": nugget,
                "sd": 2.4**2 / nvars * 2,
            }

        hmc_opts = {"nsteps": 5, "epsilon": 3e-1}  # 2D distribution
        # hmc_opts = {"nsteps": 50, "epsilon": 0.001}  # 4D
        test_cases = [[2, "DRAM", dram_opts(2)], [2, "hmc", hmc_opts]]
        for test_case in test_cases[:1]:
            self._check_mcmc_variable(*test_case)


if __name__ == "__main__":
    metropolis_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMetropolis
    )
    unittest.TextTestRunner(verbosity=2).run(metropolis_test_suite)
