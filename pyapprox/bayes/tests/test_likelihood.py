import unittest

import numpy as np
from scipy import stats

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.interface.model import (
    DenseMatrixLinearModel,
    QuadraticMatrixModel,
)
from pyapprox.bayes.likelihood import (
    ModelBasedGaussianLogLikelihood,
    ModelBasedIndependentGaussianLogLikelihood,
    IndependentExponentialLogLikelihood,
    LogUnNormalizedPosterior,
    BernoulliLogLikelihood,
    MultinomialLogLikelihood,
)
from pyapprox.bayes.laplace import DenseMatrixLaplacePosteriorApproximation
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    DirichletVariable,
)
from pyapprox.variables.marginals import BetaMarginal
from pyapprox.surrogates.affine.basis import (
    setup_tensor_product_gauss_quadrature_rule,
)
from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian


class Linear1DRegressionModel(DenseMatrixLinearModel):
    def __init__(self, design, degree, min_degree=0, backend=NumpyMixin):
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

    def _check_model_based_gaussian_loglike_fun(
        self, loglike, prior_variable, nexperiments
    ):
        bkd = self.get_backend()
        nvars = prior_variable.nvars()
        obs_model = loglike.model()
        true_sample = bkd.full((nvars, 1), 0.4)
        obs = loglike.rvs(bkd.hstack([true_sample] * nexperiments))
        loglike.set_observations(obs)
        design_weights = bkd.asarray(
            np.random.uniform(0, 1, (loglike.nobs(), 1))
        )  # hack uncomment
        design_weights = bkd.ones((loglike.nobs(), 1))
        loglike.set_design_weights(design_weights)

        prior_mean = prior_variable.mean()
        prior_cov = bkd.diag(prior_variable.var()[:, 0])
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
        # print(post_cov, laplace.posterior_covariance())
        # print(evidence, laplace.evidence())
        assert bkd.allclose(post_cov, laplace.posterior_covariance())
        assert bkd.allclose(evidence, laplace.evidence())

        n_xx = 100
        bounds = prior_variable.interval(0.99)
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

        errors = loglike.check_apply_jacobian(true_sample, disp=False)
        # check apply_jacobian function
        assert errors.min() / errors.max() < 1e-6
        # check jacobian function
        vec = bkd.ones((nvars, 1))
        assert bkd.allclose(
            loglike.jacobian(true_sample) @ vec,
            loglike.apply_jacobian(true_sample, vec),
        )

        errors = loglike.check_apply_hessian(true_sample, disp=False)
        assert errors.min() / errors.max() < 1e-6

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
        loglike_corr = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)

        # Test correlated noise single experiment
        self._check_model_based_gaussian_loglike_fun(
            loglike_corr, prior_variable, 1
        )

        # Test correlated noise multiple experiments
        self._check_model_based_gaussian_loglike_fun(
            loglike_corr, prior_variable, 2
        )

        # Test independent noise single experiment
        loglike_ind = ModelBasedIndependentGaussianLogLikelihood(
            obs_model, bkd.diag(noise_cov)[:, None]
        )
        self._check_model_based_gaussian_loglike_fun(
            loglike_ind, prior_variable, 1
        )

        # Test independent noise multiple experiment
        loglike_ind = ModelBasedIndependentGaussianLogLikelihood(
            obs_model, bkd.diag(noise_cov)[:, None]
        )
        self._check_model_based_gaussian_loglike_fun(
            loglike_ind, prior_variable, 2
        )

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
        loglike = IndependentExponentialLogLikelihood(backend=bkd)
        prior = IndependentMarginalsVariable(
            [stats.uniform(1, 2)], backend=bkd
        )

        # make sure observations are distributed according to shapes
        # by checking first two moments
        nsamples = int(1e6)
        base_shape = bkd.array(2.0)
        shapes = bkd.full((1, nsamples), base_shape)
        obs = loglike.rvs(shapes)
        assert bkd.allclose(obs.mean(axis=1), 1 / base_shape, rtol=1e-2)
        assert bkd.allclose(
            obs.var(axis=1),
            1 / base_shape**2,
            rtol=1e-2,
        )

        true_sample = prior.rvs(1)
        obs = loglike.rvs(true_sample)
        loglike.set_observations(obs)

        nsamples = 5
        shapes = prior.rvs(nsamples)
        like_vals = bkd.exp(loglike(shapes))
        print(obs.shape, shapes.shape)
        ref_like_vals = bkd.prod(
            bkd.vstack(
                [
                    bkd.asarray(
                        stats.expon(scale=1.0 / shapes[0]).pdf(obs[ii])
                    )
                    for ii in range(loglike.nobs())
                ]
            ),
            axis=0,
        )
        assert bkd.allclose(ref_like_vals, like_vals[:, 0])
        assert like_vals.shape == (nsamples, 1)

    def test_unnormalized_log_posterior(self):
        bkd = self.get_backend()
        degree = 2
        nvars = degree + 1
        prior = DenseCholeskyMultivariateGaussian(
            bkd.zeros((nvars, 1)), bkd.eye(nvars), backend=bkd
        )

        nobs = 4
        design = bkd.linspace(-1, 1, nobs)[None, :]
        noise_cov = bkd.diag(bkd.full((nobs,), 0.3))
        obs_model = Linear1DRegressionModel(design, degree, backend=bkd)
        loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)
        true_sample = bkd.full((nvars, 1), 0.4)
        obs = loglike.rvs(true_sample)
        print(obs.shape)
        loglike.set_observations(obs)
        unnormalized_posterior = LogUnNormalizedPosterior(loglike, prior)
        sample = prior.rvs(1)
        errors = unnormalized_posterior.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6
        errors = unnormalized_posterior.check_apply_hessian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_model.matrix(),
            prior.mean(),
            prior.covariance(),
            noise_cov,
            backend=bkd,
        )
        laplace.compute(obs)

        MAP = unnormalized_posterior.maximum_aposteriori_point()
        assert bkd.allclose(MAP, laplace.posterior_mean())

    def test_multinomial_likelihood(self):
        bkd = self.get_backend()
        nexperiments = 3
        ntrials = 10
        noptions = 4
        shape_args = bkd.array([2, 3, 4, 5])
        probs = np.random.uniform(0.5, 1, noptions)
        probs /= probs.sum()

        obs_np = stats.multinomial(ntrials, probs).rvs(nexperiments).T
        obs = bkd.asarray(obs_np)
        prior = DirichletVariable(shape_args, backend=bkd)
        loglike = MultinomialLogLikelihood(noptions, ntrials, backend=bkd)
        loglike.set_observations(obs)
        prior_samples = prior.rvs(1000000)
        # check value of loglike at one prior_sample

        assert bkd.allclose(
            bkd.exp(loglike(prior_samples[:, :1])),
            bkd.prod(
                bkd.array(
                    [
                        stats.multinomial(ntrials, prior_samples[:, 0]).pmf(
                            obs_np[:, nn]
                        )
                        for nn in range(nexperiments)
                    ]
                )
            ),
        )

    def test_bernoulli_likelihood(self):
        bkd = self.get_backend()
        shape_args = bkd.array([[2], [6]])
        obs = bkd.array([1, 0, 1])[None, :]
        loglike = BernoulliLogLikelihood(backend=bkd)
        loglike.set_observations(obs)
        prior = IndependentMarginalsVariable(
            [BetaMarginal(*shape_args[:, 0], 0.0, 1.0, backend=bkd)],
            backend=bkd,
        )
        prior_samples = prior.rvs(100000)
        assert bkd.allclose(
            bkd.exp(loglike(prior_samples[:, :1])),
            bkd.prod(
                bkd.array(stats.bernoulli(prior_samples[:, 0]).pmf(obs[0]))
            ),
        )


class TestNumpyLikelihood(TestLikelihood, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchLikelihood(TestLikelihood, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
