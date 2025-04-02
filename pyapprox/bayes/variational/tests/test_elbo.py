import unittest
import numpy as np

from pyapprox.interface.model import DenseMatrixLinearModel
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.bayes.variational.elbo import (
    VariationalInverseProblem,
    CholeskyGaussianVariationalPosterior,
    CholeskyGaussianKLDivergenceForVariationalInference,
    IndependentGaussianVariationalPosterior,
    IndependentGaussianKLDivergenceForVariationalInference,
)
from pyapprox.bayes.likelihood import ModelBasedGaussianLogLikelihood
from pyapprox.bayes.laplace import DenseMatrixLaplacePosteriorApproximation
from pyapprox.variables.gaussian import (
    DenseMatrixMultivariateGaussian,
    DenseMatrixIndependentMultivariateGaussian,
)
from pyapprox.util.hyperparameter import (
    flattened_lower_diagonal_matrix_entries,
)


class TestVariationalInference:
    def setUp(self):
        np.random.seed(1)

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
        nvars = prior.nvars()
        noise_cov = noise_std**2 * bkd.eye(nobs)
        obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))

        obs_model = DenseMatrixLinearModel(obs_mat, backend=bkd)
        loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)

        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_mat, prior.mean(), prior.covariance(), noise_cov, backend=bkd
        )
        true_sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        obs = loglike.rvs(true_sample)
        loglike.set_observations(obs)
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
        prior = DenseMatrixMultivariateGaussian(mean, covariance, backend=bkd)
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
        prior = DenseMatrixIndependentMultivariateGaussian(
            mean, std_diag**2, backend=bkd
        )
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


# class TestNumpyVariationalInference(
#     TestVariationalInference, unittest.TestCase
# ):
#     def get_backend(self):
#         return NumpyLinAlgMixin


class TestTorchVariationalInference(
    TestVariationalInference, unittest.TestCase
):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
