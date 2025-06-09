import unittest

import numpy as np
from scipy import stats

from pyapprox.interface.model import (
    DenseMatrixLinearModel,
    ModelFromSingleSampleCallable,
)
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.bayes.variational.elbo import (
    VariationalInverseProblem,
    CholeskyGaussianVariationalPosterior,
    IndependentGaussianVariationalPosterior,
    IndependentBetaVariationalPosterior,
    QuadratureRuleLatentVariableGenerator,
    DirichletVariationalPosterior,
)
from pyapprox.bayes.likelihood import (
    ModelBasedGaussianLogLikelihood,
    BernoulliLogLikelihood,
    MultinomialLogLikelihood,
)
from pyapprox.bayes.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
    BetaConjugatePriorPosterior,
    DirichletConjugatePriorPosterior,
)
from pyapprox.variables.gaussian import (
    DenseCholeskyMultivariateGaussian,
    IndependentMultivariateGaussian,
)
from pyapprox.util.hyperparameter import (
    flattened_lower_diagonal_matrix_entries,
)
from pyapprox.variables.marginals import BetaMarginal
from pyapprox.variables.joint import (
    IndependentMarginalsVariable,
    DirichletVariable,
)
from pyapprox.surrogates.affine.basis import (
    TensorProductQuadratureRule,
    TriangleLebesqueQuadratureRule,
)
from pyapprox.surrogates.univariate.orthopoly import GaussQuadratureRule


class TestVariationalInference:
    def setUp(self):
        np.random.seed(1)

    def _setup_linear_model_gaussian_loglike(self, nvars, nobs, noise_cov):
        bkd = self.get_backend()
        obs_mat = bkd.asarray(np.random.normal(0.0, 1.0, (nobs, nvars)))
        obs_model = DenseMatrixLinearModel(obs_mat, backend=bkd)
        loglike = ModelBasedGaussianLogLikelihood(obs_model, noise_cov)
        true_sample = bkd.asarray(np.random.uniform(0, 1, (nvars, 1)))
        print(true_sample, "true_sample")
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
        errors = vi._neg_elbo.check_apply_jacobian(iterate, disp=False)
        assert errors.min() / errors.max() < 1e-6

        vi.fit()

        print(variational_posterior.mean(), "mean")
        print(laplace.posterior_mean(), "true mean")
        print(variational_posterior.covariance(), "cov")
        print(laplace.posterior_covariance(), "true cov")
        print(prior)
        print(
            (
                variational_posterior.covariance()
                - laplace.posterior_covariance()
            )
            / laplace.posterior_covariance(),
            "rel cov error",
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
            (1, 2, 0.01, 1.0, 1000000, 2e-3),
            (2, 2, 0.01, 1.0, 1000000, 2e-3),
        ]
        for test_case in test_cases:
            np.random.seed(1)
            print(test_case)
            self._check_cholesky_based_gaussian_vi_linear_gaussian_model(
                *test_case
            )

    def _check_independent_gaussian_vi_linear_gaussian_model(
        self,
        nvars,
        nobs,
        noise_std,
        prior_std,
        nlatent_samples,
        rtol,
        latent_generator,
    ):
        bkd = self.get_backend()
        mean = bkd.ones((nvars, 1))
        std_diag = bkd.full((nvars,), prior_std)
        prior = IndependentMultivariateGaussian(mean, std_diag**2, backend=bkd)
        init_post_mean = mean
        init_post_std_diag = std_diag
        variational_posterior = IndependentGaussianVariationalPosterior(
            prior,
            nlatent_samples,
            init_post_std_diag,
            mean_values=init_post_mean[:, 0],
            latent_generator=latent_generator,
            backend=bkd,
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
        bkd = self.get_backend()
        marginal = stats.norm(0, 1)
        quad_rule = TensorProductQuadratureRule(
            1, [GaussQuadratureRule(marginal, backend=bkd)]
        )
        latent_gen_1d = QuadratureRuleLatentVariableGenerator(quad_rule)

        test_cases = [
            (1, 2, 0.01, 1.0, 1000000, 2e-3, None),
            (1, 2, 0.01, 1.0, 100, 1e-8, latent_gen_1d),
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_independent_gaussian_vi_linear_gaussian_model(
                *test_case
            )

    def test_beta_conjugate_prior(self):
        bkd = self.get_backend()
        shape_args = bkd.array([[2], [6]])
        nobs = 3
        post = BetaConjugatePriorPosterior(shape_args, nobs, backend=bkd)
        obs = bkd.array([1, 0, 1])[:, None][:nobs]
        post.compute(obs)

        prior = IndependentMarginalsVariable(
            [BetaMarginal(*shape_args[:, 0], 0.0, 1.0, backend=bkd)]
        )
        loglike = BernoulliLogLikelihood(backend=bkd)
        loglike.set_observations(obs)

        ashapes = [marginal._a for marginal in prior.marginals()]
        bshapes = [marginal._b for marginal in prior.marginals()]
        marginal = stats.uniform(0, 1)
        quad_rule = TensorProductQuadratureRule(
            1, [GaussQuadratureRule(marginal, backend=bkd)]
        )
        nlatent_samples = 1000
        latent_generator = QuadratureRuleLatentVariableGenerator(quad_rule)
        # nlatent_samples = 10000
        # latent_generator = None
        variational_posterior = IndependentBetaVariationalPosterior(
            prior,
            nlatent_samples,
            ashapes,
            bshapes,
            prior.interval(1),
            ashape_bounds=(1, 100),
            bshape_bounds=(1, 100),
            latent_generator=latent_generator,
            backend=bkd,
        )
        vi = VariationalInverseProblem(prior, loglike, variational_posterior)
        vi.set_optimizer(
            vi.default_optimizer(
                gtol=1e-10, verbosity=0, local_method="trust-constr"
            )
        )
        iterate = vi._neg_elbo.hyp_list().get_active_opt_params()[:, None]
        # print(vi._neg_elbo.jacobian(iterate))
        errors = vi._neg_elbo.check_apply_jacobian(iterate, disp=False)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 4e-6
        vi.fit()
        print(
            variational_posterior._ashapes.get_values(),
            post._posterior_shapes[0],
        )
        print(
            variational_posterior._ashapes.get_values()
            - post._posterior_shapes[0],
        )
        assert bkd.allclose(
            variational_posterior._ashapes.get_values(),
            post._posterior_shapes[0],
            rtol=5e-3,  # increasing nlatent_samples increases accuracy
        )
        assert bkd.allclose(
            variational_posterior._bshapes.get_values(),
            post._posterior_shapes[1],
            rtol=5e-3,  # increasing nlatent_samples increases accuracy
        )

    def test_dirichlet_conjugate_prior(self):
        bkd = self.get_backend()
        shape_args = bkd.array([2, 3, 4])
        nvars = len(shape_args)
        nobs, ntrials = bkd.array([3, 10], dtype=int)
        noptions = len(shape_args)
        probs = bkd.array(np.random.uniform(0.5, 1, noptions))
        probs /= probs.sum()
        post = DirichletConjugatePriorPosterior(
            shape_args, nobs, ntrials, noptions, backend=bkd
        )
        obs = bkd.asarray(stats.multinomial(ntrials, probs).rvs(nobs))
        post.compute(obs)
        prior = DirichletVariable(shape_args, backend=bkd)
        loglike = MultinomialLogLikelihood(noptions, ntrials, backend=bkd)
        loglike.set_observations(obs)

        # vertices = bkd.array([[0, 0.5, 1], [0, 1, 0]])
        # quad_rule = TriangleLebesqueQuadratureRule(vertices, backend=bkd)
        # marginal = stats.uniform(0, 1)
        # quad_rule = TensorProductQuadratureRule(
        #     nvars,
        #     [GaussQuadratureRule(marginal, backend=bkd)] * nvars,
        # )
        # nlatent_samples = 50
        # latent_generator = QuadratureRuleLatentVariableGenerator(quad_rule)
        nlatent_samples = 10000
        latent_generator = None
        variational_posterior = DirichletVariationalPosterior(
            prior,
            nlatent_samples,
            shape_args,
            ashape_bounds=(1, 30),
            latent_generator=latent_generator,
            backend=bkd,
        )
        vi = VariationalInverseProblem(prior, loglike, variational_posterior)
        vi.set_optimizer(
            vi.default_optimizer(
                gtol=1e-10, verbosity=0, local_method="trust-constr"
            )
        )
        iterate = vi._neg_elbo.hyp_list().get_active_opt_params()[:, None]
        errors = vi._neg_elbo.check_apply_jacobian(iterate, disp=False)
        assert errors.min() / errors.max() < 4e-6
        vi.fit()
        assert bkd.allclose(
            variational_posterior._ashapes.get_values(),
            post._posterior_shapes,
            rtol=1e-3,  # increasing nlatent_samples increases accuracy
        )

    def test_independent_beta_vi(self):
        # This tests is really just an example. As error from assuming
        # independence is nontrivial.

        bkd = self.get_backend()
        noise_std = 0.1
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

        axs = plt.subplots(1, 2)[1]

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
            axs[0], prior.interval(1.0).flatten(), levels=31
        )

        # plot true posterior
        # def posterior_numerator(x):
        #     return bkd.exp(loglike(x) + prior.logpdf(x))

        # quad_rule = TensorProductQuadratureRule(
        #     2,
        #     [
        #         GaussQuadratureRule(marginal, backend=bkd)
        #         for marginal in prior.marginals()
        #     ],
        # )
        # quadx, quadw = quad_rule([100, 100])
        # print(quadx.min(), prior.marginals()[0].interval(1))
        # evidence = posterior_numerator(quadx)[:, 0] @ quadw[:, 0]
        # posterior_pdf = ModelFromSingleSampleCallable(
        #     1, 2, lambda x: posterior_numerator(x) / evidence, backend=bkd
        # )
        # print(evidence, "evidence")
        # im = posterior_pdf.plot_contours(axs[1], [0, 1, 0, 1], levels=30)
        # plt.colorbar(im, ax=axs[1])
        # plt.show()
        print(vi)
        # regression test
        assert bkd.allclose(
            variational_posterior.hyp_list().get_values(),
            bkd.array([2.0000, 29.3894, 2.0000, 26.8369]),
        )


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
