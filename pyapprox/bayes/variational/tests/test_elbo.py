import unittest
import numpy as np

from pyapprox.interface.model import DenseMatrixLinearModel
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.bayes.variational.elbo import (
    VariationalInverseProblem,
    CholeskyGaussianVariationalPosterior,
    CholeskyGaussianKLDivergenceForVariationalInference,
    IndependentGaussianVariationalPosterior,
    IndependentGaussianKLDivergenceForVariationalInference,
    IndependentBetaVariationalPosterior,
    CustomIndependentMarginalsVariableKLDivergenceForVariationalInference,
)
from pyapprox.bayes.likelihood import ModelBasedGaussianLogLikelihood
from pyapprox.bayes.laplace import DenseMatrixLaplacePosteriorApproximation
from pyapprox.variables.gaussian import (
    DenseCholeskyMultivariateGaussian,
    IndependentMultivariateGaussian,
)
from pyapprox.util.hyperparameter import (
    flattened_lower_diagonal_matrix_entries,
)
from pyapprox.variables.marginals import BetaMarginal
from pyapprox.variables.joint import CustomIndependentMarginalsVariable


class TestVariationalInference:
    def setUp(self):
        np.random.seed(1)

    def _setup_linear_model_gaussian_loglike(self, nvars, nobs, noise_cov):
        bkd = self.get_backend()
        obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))
        obs_model = DenseMatrixLinearModel(obs_mat, backend=bkd)
        loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)
        true_sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        obs = loglike.rvs(true_sample)
        loglike.set_observations(obs)
        return loglike, obs, obs_model

    def _check_gaussian_vi_linear_gaussian_model(
        self,
        nobs,
        noise_std,
        rtol,
        prior,
        variational_posterior,
        divergence_cls,
    ):
        bkd = self.get_backend()
        noise_cov = noise_std**2 * bkd.eye(nobs)
        loglike, obs, obs_model = self._setup_linear_model_gaussian_loglike(
            prior.nvars(), nobs, noise_cov
        )

        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_model.matrix(),
            prior.mean(),
            prior.covariance(),
            noise_cov,
            backend=bkd,
        )
        laplace.compute(obs)

        divergence = divergence_cls(prior, variational_posterior)
        vi = VariationalInverseProblem(prior, loglike, divergence)
        vi.fit()

        print(variational_posterior.mean())
        print(laplace.posterior_mean())
        print(variational_posterior.covariance())
        print(laplace.posterior_covariance())
        print(
            (
                variational_posterior.covariance()
                - laplace.posterior_covariance()
            )
            / laplace.posterior_covariance()
        )
        assert bkd.allclose(
            variational_posterior.covariance(),
            laplace.posterior_covariance(),
            rtol=rtol,
        )
        assert bkd.allclose(
            variational_posterior.mean(), laplace.posterior_mean(), rtol=rtol
        )

    def _check_cholesky_based_gaussian_vi_linear_gaussian_model(
        self, nvars, nobs, noise_std, prior_std, nlatent_samples, rtol
    ):
        bkd = self.get_backend()
        mean = bkd.ones((nvars, 1))
        covariance = prior_std**2 * bkd.eye(nvars)
        prior = DenseCholeskyMultivariateGaussian(
            mean, covariance, backend=bkd
        )
        variational_posterior = CholeskyGaussianVariationalPosterior(
            nvars,
            nlatent_samples,
            flattened_lower_diagonal_matrix_entries(
                bkd.cholesky(prior.covariance())
            ),
            backend=bkd,
        )
        self._check_gaussian_vi_linear_gaussian_model(
            nobs,
            noise_std,
            rtol,
            prior,
            variational_posterior,
            CholeskyGaussianKLDivergenceForVariationalInference,
        )

    def test_cholesky_based_gaussian_vi_linear_gaussian_model(self):
        test_cases = [
            (1, 2, 0.01, 1.0, 1000000, 1e-3),
            (2, 2, 0.01, 1.0, 100000, 7e-3),
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_cholesky_based_gaussian_vi_linear_gaussian_model(
                *test_case
            )

    def _check_independent_gaussian_vi_linear_gaussian_model(
        self, nvars, nobs, noise_std, prior_std, nlatent_samples, rtol
    ):
        bkd = self.get_backend()
        mean = bkd.ones((nvars, 1))
        std_diag = bkd.full((nvars,), prior_std)
        prior = IndependentMultivariateGaussian(mean, std_diag**2, backend=bkd)
        variational_posterior = IndependentGaussianVariationalPosterior(
            nvars, nlatent_samples, std_diag, backend=bkd
        )
        self._check_gaussian_vi_linear_gaussian_model(
            nobs,
            noise_std,
            rtol,
            prior,
            variational_posterior,
            IndependentGaussianKLDivergenceForVariationalInference,
        )

    def test_independent_gaussian_vi_linear_gaussian_model(self):
        # can only use check_gaussian_vi_linear_gaussian_model with nvars=1
        # because reference solution is exact laplace posterior which
        # will have correlations which cannot be captured by
        # an independent gaussian variational posterior
        test_cases = [
            (1, 2, 0.01, 1.0, 100000, 5e-3),
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_independent_gaussian_vi_linear_gaussian_model(
                *test_case
            )

    def test_independent_beta_vi(self):
        bkd = self.get_backend()
        noise_std = 0.01
        nobs = 2
        noise_cov = noise_std**2 * bkd.eye(nobs)
        a1, b1 = 2, 3
        a2, b2 = 3, 3
        bounds = [0, 1]
        marginals = [
            BetaMarginal(a1, b1, *bounds),
            BetaMarginal(a2, b2, *bounds),
        ]
        prior = CustomIndependentMarginalsVariable(marginals)
        loglike, obs, obs_model = self._setup_linear_model_gaussian_loglike(
            prior.nvars(), nobs, noise_cov
        )
        nlatent_samples = 100000
        ashapes = [marginal._a for marginal in prior._marginals]
        bshapes = [marginal._b for marginal in prior._marginals]
        variational_posterior = IndependentBetaVariationalPosterior(
            prior.nvars(),
            nlatent_samples,
            ashapes,
            bshapes,
            backend=bkd,
        )
        divergence = CustomIndependentMarginalsVariableKLDivergenceForVariationalInference(
            prior, variational_posterior
        )
        vi = VariationalInverseProblem(prior, loglike, divergence)
        vi.fit()
        # TODO instead of passing around divergence object
        # make divergence proprty of posterior
        print(vi)


# class TestNumpyVariationalInference(
#     TestVariationalInference, unittest.TestCase
# ):
#     def get_backend(self):
#         return NumpyMixin


class TestTorchVariationalInference(
    TestVariationalInference, unittest.TestCase
):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
