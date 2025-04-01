import unittest
import numpy as np

from pyapprox.interface.model import DenseMatrixLinearModel
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.bayes.variational.elbo import (
    VariationalInverseProblem,
    CholeskyGaussianVariationalPosterior,
    CholeskyGaussianKLDivergenceForVariationalInference,
)
from pyapprox.bayes.likelihood import ModelBasedGaussianLogLikelihood
from pyapprox.bayes.laplace import DenseMatrixLaplacePosteriorApproximation
from pyapprox.variables.gaussian import (
    DenseMatrixMultivariateGaussian,
    #  DenseMatrixIndependentMultivariateGaussian,
)
from pyapprox.util.hyperparameter import (
    flattened_lower_diagonal_matrix_entries,
)


class TestVariationalInference:
    def setUp(self):
        np.random.seed(1)

    def test_vi_linear_gaussian_model(self):
        bkd = self.get_backend()
        nvars = 2
        nobs = 3
        mean = bkd.ones((nvars, 1))
        covariance = 0.1 * bkd.eye(nvars)
        noise_std = 0.01
        noise_cov = noise_std**2 * bkd.eye(nobs)
        obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))
        prior = DenseMatrixMultivariateGaussian(mean, covariance, backend=bkd)
        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_mat, prior.mean(), prior.covariance(), noise_cov, backend=bkd
        )
        true_sample = bkd.asarray(np.random.uniform(-1, 1, (nvars, 1)))
        noise = bkd.asarray(np.random.normal(0, noise_std, (nobs, 1)))
        obs = obs_mat @ true_sample + noise
        laplace.compute(obs)

        obs_model = DenseMatrixLinearModel(obs_mat, backend=bkd)
        loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)
        loglike.set_observations(obs)

        variational_posterior = CholeskyGaussianVariationalPosterior(
            nvars,
            flattened_lower_diagonal_matrix_entries(
                bkd.cholesky(prior.covariance())
            ),
            backend=bkd,
        )
        print(variational_posterior)

        divergence = CholeskyGaussianKLDivergenceForVariationalInference(
            prior, variational_posterior
        )
        print(divergence)
        vi = VariationalInverseProblem(prior, loglike, divergence)
        vi.fit()

        print(variational_posterior.mean())
        print(laplace.posterior_mean())
        print(variational_posterior.covariance())
        print(laplace.posterior_covariance())
        assert bkd.allclose(
            variational_posterior.covariance(),
            laplace.posterior_covariance(),
        )
        assert bkd.allclose(
            variational_posterior.mean(),
            laplace.posterior_mean(),
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
