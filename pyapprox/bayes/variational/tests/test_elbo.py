import unittest
import numpy as np

from pyapprox.interface.model import DenseMatrixLinearModel
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.bayes.variational.elbo import (
    VariationalInverseProblem,
    CholeskyGaussianVariationalPosterior,
    IndependentGaussianVariationalPosterior,
    IndependentBetaVariationalPosterior,
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
from pyapprox.variables.joint import IndependentMarginalsVariable


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

        vi = VariationalInverseProblem(prior, loglike, variational_posterior)

        iterate = vi._neg_elbo.hyp_list().get_active_opt_params()[:, None]
        errors = vi._neg_elbo.check_apply_jacobian(iterate, disp=True)
        assert errors.min() / errors.max() < 1e-6

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
            prior,
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
            prior, nlatent_samples, std_diag, backend=bkd
        )
        self._check_gaussian_vi_linear_gaussian_model(
            nobs,
            noise_std,
            rtol,
            prior,
            variational_posterior,
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
        # TODO: create class that uses same independence divergence as used here
        # bust using gaussian posteriors to better check this code where
        # no closed form solution exists.

        bkd = self.get_backend()
        noise_std = 0.01
        nobs = 2
        noise_cov = noise_std**2 * bkd.eye(nobs)
        # values for testing variational inference with uniformative (uniform)
        # priors
        a1, b1 = 2, 2
        a2, b2 = 2, 2
        # values good for testing gradients
        # a1, b1 = 1.4444, 14.5701
        # a2, b2 = 3, 3
        bounds = [0, 1]
        marginals = [
            BetaMarginal(a1, b1, *bounds, backend=bkd),
            BetaMarginal(a2, b2, *bounds, backend=bkd),
        ]
        xx = bkd.linspace(0, 1, 101)
        assert bkd.all(marginals[0]._cdf_jacobian_diagonal(xx) >= 0)
        prior = IndependentMarginalsVariable(marginals)
        loglike, obs, obs_model = self._setup_linear_model_gaussian_loglike(
            prior.nvars(), nobs, noise_cov
        )
        nlatent_samples = 1000
        ashapes = [marginal._a for marginal in prior.marginals()]
        bshapes = [marginal._b for marginal in prior.marginals()]
        variational_posterior = IndependentBetaVariationalPosterior(
            prior,
            nlatent_samples,
            ashapes,
            bshapes,
            prior.interval(1),
            ashape_bounds=(1, 100),
            bshape_bounds=(1, 100),
            backend=bkd,
        )
        vi = VariationalInverseProblem(prior, loglike, variational_posterior)
        iterate = vi._neg_elbo.hyp_list().get_active_opt_params()[:, None]
        print(vi._neg_elbo.jacobian(iterate))
        errors = vi._neg_elbo.check_apply_jacobian(iterate, disp=True)
        print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 4e-6
        vi.fit()
        # TODO instead of passing around divergence object
        # make divergence proprty of posterior
        import matplotlib.pyplot as plt

        ax = plt.figure().gca()

        print(variational_posterior.hyp_list().get_values())
        # Currently pytorch code will set self._a and self._b
        # when computing gradients. These variables will retain
        # requires grad and thus if used for numpy functions
        # will throw must be detached error.
        # So for now reset shape parameters using values
        # that do not have requires_grad
        variational_posterior.hyp_list().set_values(
            variational_posterior.hyp_list().get_values()
        )
        variational_posterior.update()
        variational_posterior._variable.plot_pdf(
            ax, prior.interval(1.0).flatten(), levels=31
        )
        plt.show()
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
    unittest.main(verbosity=2)
