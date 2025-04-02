import unittest

import numpy as np
from scipy import stats

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.interface.model import (
    DenseMatrixLinearModel,
    QuadraticMatrixModel,
)
from pyapprox.bayes.likelihood import (
    ModelBasedGaussianLogLikelihood,
    ModelBasedIndependentGaussianLogLikelihood,
    ModelBasedIndependentExponentialLogLikelihood,
    LogUnNormalizedPosterior,
)
from pyapprox.bayes.laplace import DenseMatrixLaplacePosteriorApproximation
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.bases.basis import (
    setup_tensor_product_gauss_quadrature_rule,
)
from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian


class Linear1DRegressionModel(DenseMatrixLinearModel):
    def __init__(self, design, degree, min_degree=0, backend=NumpyLinAlgMixin):
        assert degree >= min_degree
        self._design = design
        self._degree = degree
        matrix = self._design.T ** (
            backend.arange(min_degree, self._degree + 1)[None, :]
        )
        super().__init__(matrix, None, backend=backend)


class TestLikelihood:

    def setUp(self):
        np.random.seed(1)

    def _check_model_based_gaussian_loglike_fun(self, loglike, prior_variable):
        bkd = self.get_backend()
        nvars = prior_variable.nvars()
        obs_model = loglike.model()
        true_sample = bkd.full((nvars, 1), 0.4)
        obs = loglike.rvs(true_sample)
        loglike.set_observations(obs)
        design_weights = bkd.asarray(
            np.random.uniform(0, 1, (loglike.nobs(), 1))
        )  # hack uncomment
        # design_weights = bkd.ones((loglike.nobs(), 1))
        loglike.set_design_weights(design_weights)

        prior_mean = prior_variable.get_statistics("mean")
        prior_cov = bkd.diag(prior_variable.get_statistics("std")[:, 0] ** 2)
        # noise_cov_inv = bkd.inv(loglike.noise_covariance())
        # sqrt_design_weights = bkd.diag(bkd.sqrt(design_weights[:, 0]))
        # weighted_noise_cov_inv = (
        #     sqrt_design_weights @ noise_cov_inv @ sqrt_design_weights
        # )
        # weighted_noise_cov = bkd.inv(weighted_noise_cov_inv)
        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_model.matrix(),
            prior_mean,
            prior_cov,
            loglike.noise_covariance(),  # weighted_noise_cov,
            obs_model.vector(),
            backend=bkd,
        )
        laplace.compute(obs)

        # will not give correct answer if loglike.obs().shape[1] > 1
        quad_rule = setup_tensor_product_gauss_quadrature_rule(prior_variable)
        xx_gauss, ww_gauss = quad_rule([400] * nvars)
        loglike_vals = loglike(xx_gauss)
        like_vals = bkd.exp(loglike_vals)
        post_cov = bkd.cov(
            xx_gauss,
            aweights=(ww_gauss[:, 0] * like_vals[:, 0]),
            ddof=0,
        )
        evidence = like_vals[:, 0] @ ww_gauss[:, 0]
        print(post_cov, laplace.posterior_covariance())
        # print(evidence, laplace.evidence())
        assert bkd.allclose(post_cov, laplace.posterior_covariance())
        assert bkd.allclose(evidence, laplace.evidence())

        n_xx = 100
        bounds = prior_variable.get_statistics("interval", confidence=0.99)
        xx = bkd.cartesian_product(
            [bkd.linspace(*bound, n_xx) for bound in bounds]
        )
        numerator = bkd.exp(loglike(xx)) * prior_variable.pdf(xx)
        assert numerator.shape == (xx.shape[1], 1)
        post_pdf_vals = numerator / evidence
        true_post_pdf_vals = laplace.posterior_variable().pdf(xx)
        # make sure xx captures some regions of non-trivial probability
        assert true_post_pdf_vals.max() > 0.1
        # accuracy depends on quadrature rule and size of noise
        # print(post_pdf_vals-true_post_pdf_vals)
        assert bkd.allclose(post_pdf_vals, true_post_pdf_vals)

        errors = loglike.check_apply_jacobian(true_sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

        errors = loglike.check_apply_hessian(true_sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

        vec = bkd.ones((obs_model.nvars(), 1))

    def test_linear_model_based_gaussian_likelihood(self):
        bkd = self.get_backend()
        degree = 1
        nvars = degree + 1
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)] * nvars, backend=bkd
        )

        nobs = 4
        design = bkd.linspace(-1, 1, nobs)[None, :]
        noise_cov = bkd.diag(bkd.full((nobs,), 0.3))
        obs_model = Linear1DRegressionModel(design, degree, backend=bkd)
        loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)

        self._check_model_based_gaussian_loglike_fun(loglike, prior_variable)

        loglike = ModelBasedIndependentGaussianLogLikelihood(
            obs_model, bkd.diag(noise_cov)[:, None]
        )
        self._check_model_based_gaussian_loglike_fun(loglike, prior_variable)

    def test_quadratic_model_based_gaussian_likelihood(self):
        # tests apply_hessian when model hessian is non-zero
        bkd = self.get_backend()
        degree = 1
        nvars = degree + 1
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)] * nvars, backend=bkd
        )
        nobs = 4
        noise_cov = bkd.diag(bkd.full((nobs,), 0.3))
        obs_model = QuadraticMatrixModel(
            bkd.asarray(np.random.uniform(-1, 1, (nobs, nvars))), backend=bkd
        )
        loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)
        true_sample = prior_variable.rvs(1)
        obs = loglike.rvs(true_sample)
        loglike.set_observations(obs)
        errors = loglike.check_apply_jacobian(true_sample, disp=True)
        assert errors.min() / errors.max() < 1e-6
        errors = loglike.check_apply_hessian(true_sample, disp=True)
        assert errors.min() / errors[0] < 1e-6

    def test_model_based_exponential_likelihood(self):
        bkd = self.get_backend()
        nobs = 2
        degree = 2
        nvars = degree + 1
        design = bkd.linspace(-1, 1, nobs)[None, :]
        obs_model = Linear1DRegressionModel(design, degree, backend=bkd)
        # noise_scale_diag = bkd.full((nobs, 1), .5)
        noise_scale_diag = bkd.asarray(np.random.uniform(0.5, 1, (nobs, 1)))
        design_weights = bkd.ones((nobs, 1))
        loglike = ModelBasedIndependentExponentialLogLikelihood(
            obs_model, noise_scale_diag, tile_obs=False
        )
        loglike.set_design_weights(design_weights)
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)] * nvars, backend=bkd
        )
        nsamples = int(1e5)
        true_sample = prior_variable.rvs(1)
        # true_pred_obs = obs_model(true_sample).T*0  # hack works
        true_pred_obs = obs_model(true_sample).T
        noise = loglike._sample_noise(nsamples)
        assert bkd.allclose(
            noise.mean(axis=1), 1 / noise_scale_diag[:, 0], rtol=1e-2
        )
        assert bkd.allclose(
            noise.std(axis=1), 1 / noise_scale_diag[:, 0], rtol=1e-2
        )
        many_pred_obs = bkd.repeat(true_pred_obs, nsamples, axis=1)
        obs = loglike._make_noisy(many_pred_obs, noise)
        loglike.set_observations(obs)
        like_vals = bkd.exp(loglike._loglike(many_pred_obs))
        assert bkd.allclose(
            bkd.prod(
                bkd.vstack(
                    [
                        bkd.asarray(
                            stats.expon(scale=1 / noise_scale_diag[ii, 0]).pdf(
                                obs[ii] - true_pred_obs[ii]
                            )
                        )
                        for ii in range(nobs)
                    ]
                ),
                axis=0,
            ),
            like_vals[:, 0],
        )
        assert like_vals.shape == (nsamples, 1)

    def test_unnormalized_log_posterior(self):
        bkd = self.get_backend()
        degree = 2
        nvars = degree + 1
        prior_variable = DenseCholeskyMultivariateGaussian(
            bkd.zeros((nvars, 1)), bkd.eye(nvars), backend=bkd
        )

        nobs = 4
        design = bkd.linspace(-1, 1, nobs)[None, :]
        noise_cov = bkd.diag(bkd.full((nobs,), 0.3))
        obs_model = Linear1DRegressionModel(design, degree, backend=bkd)
        loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)
        true_sample = bkd.full((nvars, 1), 0.4)
        obs = loglike.rvs(true_sample)
        loglike.set_observations(obs)
        unnormalized_posterior = LogUnNormalizedPosterior(
            loglike, prior_variable
        )
        sample = prior_variable.rvs(1)
        errors = unnormalized_posterior.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6
        errors = unnormalized_posterior.check_apply_hessian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6


class TestNumpyLikelihood(TestLikelihood, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchLikelihood(TestLikelihood, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
