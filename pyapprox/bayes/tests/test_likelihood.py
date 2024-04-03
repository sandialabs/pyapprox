import unittest

import numpy as np
from scipy import stats

from pyapprox.interface.model import Model
from pyapprox.bayes.likelihood import (
    GaussianLogLikelihood, ModelBasedGaussianLogLikelihood,
    IndependentGaussianLogLikelihood,
    ModelBasedIndependentGaussianLogLikelihood,
    OEDGaussianLogLikelihood, Evidence, LogEvidence, KLOEDObjective,
    SingleObsIndependentGaussianLogLikelihood)
from pyapprox.bayes.laplace import (
    laplace_posterior_approximation_for_linear_models, laplace_evidence)
from pyapprox.util.utilities import cartesian_product
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.integrate import integrate


class Linear1DRegressionModel(Model):
    def __init__(self, design, degree):
        super().__init__()
        self._design = design
        self._degree = degree
        self._jac_matrix = self._design.T**(
            np.arange(self._degree+1)[None, :])
        self._jacobian_implemented = True

    def __call__(self, samples):
        return (self._jac_matrix @ (samples)).T

    def _jacobian(self, sample):
        return self._jac_matrix


class TestLikelihood(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def _check_gaussian_loglike_fun(self, model_loglike, prior_variable):
        nvars = prior_variable.num_vars()
        obs_model = model_loglike._model
        loglike = model_loglike._loglike
        true_sample = np.full((nvars, 1), 0.4)
        obs = model_loglike.rvs(true_sample)
        loglike.set_observations(obs)
        weights = np.random.uniform(0, 1, (loglike._get_nobs(), 1))
        loglike.set_weights(weights)

        prior_mean = prior_variable.get_statistics("mean")
        prior_cov = np.diag(prior_variable.get_statistics("std")[:, 0]**2)
        noise_cov_inv = np.linalg.inv(loglike.noise_covariance())
        sqrt_weights = np.diag(np.sqrt(weights[:, 0]))
        noise_cov_inv = sqrt_weights@noise_cov_inv@sqrt_weights
        exact_post_mean, exact_post_cov = \
            laplace_posterior_approximation_for_linear_models(
                obs_model._jac_matrix, prior_mean, np.linalg.inv(prior_cov),
                noise_cov_inv, obs)

        xx_gauss, ww_gauss = integrate(
            "tensorproduct", prior_variable, levels=[400]*nvars)
        pred_obs = obs_model(xx_gauss).T
        loglike_vals = loglike(pred_obs)
        evidence = np.exp(loglike_vals[:, 0]).dot(ww_gauss)

        post_cov = np.cov(
            xx_gauss, aweights=(ww_gauss*np.exp(loglike_vals))[:, 0], ddof=0)
        # print((post_cov, exact_post_cov))
        assert np.allclose(post_cov, exact_post_cov)

        gauss_evidence = laplace_evidence(
            lambda x: np.exp(loglike(obs_model(x).T)[:, 0]),
            prior_variable.pdf, exact_post_cov, exact_post_mean)
        # print(evidence, gauss_evidence)
        assert np.allclose(evidence, gauss_evidence)

        n_xx = 100
        bounds = prior_variable.get_statistics("interval", confidence=0.99)
        xx = cartesian_product([np.linspace(*bound, n_xx) for bound in bounds])
        pred_obs = obs_model(xx).T
        numerator = np.exp(loglike(pred_obs))*prior_variable.pdf(xx)
        assert numerator.shape == (xx.shape[1], 1)
        post_pdf_vals = numerator/evidence
        true_post_pdf_vals = stats.multivariate_normal(
            exact_post_mean[:, 0], cov=exact_post_cov).pdf(xx.T)[:, None]
        # make sure xx captures some regions of non-trivial probability
        assert true_post_pdf_vals.max() > 0.1
        # accuracy depends on quadrature rule and size of noise
        # print(post_pdf_vals-true_post_pdf_vals)
        assert np.allclose(post_pdf_vals, true_post_pdf_vals)

        errors = model_loglike.check_apply_jacobian(true_sample, disp=True)
        assert errors.min()/errors.max() < 1e-6

    def test_independent_gaussian_likelihood(self):
        degree = 1
        nvars = degree+1
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*nvars)

        nobs = 4
        design = np.linspace(-1, 1, nobs)[None, :]
        noise_cov = np.diag(np.full((nobs,), 0.3))
        obs_model = Linear1DRegressionModel(design, degree)
        loglike = GaussianLogLikelihood(noise_cov)
        model_loglike = ModelBasedGaussianLogLikelihood(obs_model, loglike)

        self._check_gaussian_loglike_fun(model_loglike, prior_variable)

        loglike = IndependentGaussianLogLikelihood(np.diag(noise_cov)[:, None])
        model_loglike = ModelBasedIndependentGaussianLogLikelihood(
            obs_model, loglike)
        self._check_gaussian_loglike_fun(model_loglike, prior_variable)

        loglike = SingleObsIndependentGaussianLogLikelihood(
            np.diag(noise_cov)[:, None])
        model_loglike = ModelBasedIndependentGaussianLogLikelihood(
            obs_model, loglike)
        self._check_gaussian_loglike_fun(model_loglike, prior_variable)

    def test_many_obs_independent_gaussian_likelihood_many_obs(self):
        degree = 1
        nvars = degree+1
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*nvars)

        nobs = 4
        design = np.linspace(-1, 1, nobs)[None, :]
        noise_cov = np.full((nobs, 1), 0.3)
        obs_model = Linear1DRegressionModel(design, degree)
        loglike = IndependentGaussianLogLikelihood(noise_cov)

        ntrue_samples = 10
        true_samples = prior_variable.rvs(ntrue_samples)
        obs = loglike._make_noisy(
            obs_model(true_samples).T, loglike._sample_noise(ntrue_samples))
        loglike.set_observations(obs)

        samples, ww_gauss = integrate(
            "tensorproduct", prior_variable, levels=[40]*nvars)
        many_pred_obs = obs_model(samples).T
        vals = loglike(many_pred_obs)
        assert vals.shape[0] == many_pred_obs.shape[1]*obs.shape[1]

        single_loglike = SingleObsIndependentGaussianLogLikelihood(
            noise_cov)
        NN = many_pred_obs.shape[1]
        for ii in range(ntrue_samples):
            single_loglike.set_observations(obs[:, ii:ii+1])
            assert np.allclose(
                single_loglike(many_pred_obs), vals[ii*NN:(ii+1)*NN])

    def test_OED_gaussian_likelihood(self):
        degree = 1
        nvars = degree+1
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*nvars)

        nobs = 2
        design = np.linspace(-1, 1, nobs)[None, :]
        noise_cov_diag = np.full((nobs, 1), 0.3)
        obs_model = Linear1DRegressionModel(design, degree)
        loglike = IndependentGaussianLogLikelihood(noise_cov_diag)

        ntrue_samples = 3
        true_samples = prior_variable.rvs(ntrue_samples)
        outer_pred_obs = obs_model(true_samples).T
        noise_samples = loglike._sample_noise(ntrue_samples)
        obs = loglike._make_noisy(outer_pred_obs, noise_samples)
        loglike.set_observations(obs)

        samples, ww_gauss = integrate(
            "tensorproduct", prior_variable, levels=[1]*nvars)
        many_pred_obs = obs_model(samples).T
        oed_loglike = OEDGaussianLogLikelihood(loglike, many_pred_obs)

        weights = np.ones((nobs, 1))
        errors = oed_loglike.check_apply_jacobian(weights, disp=True)
        assert errors.min()/errors.max() < 1e-6

        # a = np.arange(6).reshape(3, 2)
        # b = np.repeat(a, 2, axis=1)
        # print(a)
        # print(b)

        evidence_model = Evidence(oed_loglike)
        evidence = evidence_model(weights)
        assert evidence.shape[0] == ntrue_samples
        evidence_model.check_apply_jacobian(weights, disp=True)

        log_evidence_model = LogEvidence(oed_loglike)
        log_evidence = log_evidence_model(weights)
        np.set_printoptions(precision=16, linewidth=1000)
        assert log_evidence.shape[0] == ntrue_samples
        log_evidence_model.check_apply_jacobian(weights, disp=True)

        inner_pred_obs = many_pred_obs
        oed_objective = KLOEDObjective(
            noise_cov_diag, outer_pred_obs, noise_samples, inner_pred_obs)
        errors = oed_objective.check_apply_jacobian(weights, disp=True)


if __name__ == '__main__':
    unittest.main()
