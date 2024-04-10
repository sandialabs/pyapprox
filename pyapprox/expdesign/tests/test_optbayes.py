import unittest

import numpy as np
from scipy import stats

from pyapprox.interface.model import Model
from pyapprox.bayes.likelihood import (
    IndependentGaussianLogLikelihood,
    SingleObsIndependentGaussianLogLikelihood)
from pyapprox.expdesign.optbayes import (
    OEDGaussianLogLikelihood, Evidence, LogEvidence, KLOEDObjective,
    WeightsConstraint)
from pyapprox.bayes.laplace import (
    laplace_posterior_approximation_for_linear_models, laplace_evidence)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.integrate import integrate
from pyapprox.expdesign.tests.test_bayesian_oed import (
    expected_kl_divergence_gaussian_inference,
    posterior_mean_data_stats, gaussian_kl_divergence)
from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer, Bounds)


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


class TestBayesOED(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_OED_gaussian_likelihood(self):
        degree = 0
        nvars = degree+1
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)]*nvars)

        nobs = 4
        design = np.linspace(-1, 1, nobs)[None, :]
        noise_cov_diag = np.full((nobs, 1), .3**2)
        obs_model = Linear1DRegressionModel(design, degree)
        loglike = IndependentGaussianLogLikelihood(noise_cov_diag)

        np.random.seed(1)
        ntrue_samples = 10000
        true_samples = prior_variable.rvs(ntrue_samples)
        outer_pred_weights = np.full((ntrue_samples, 1), 1/ntrue_samples)
        outer_pred_obs = obs_model(true_samples).T
        np.random.seed(1)
        noise_samples = loglike._sample_noise(ntrue_samples)
        obs = loglike._make_noisy(outer_pred_obs, noise_samples)
        loglike.set_observations(obs)

        samples, inner_pred_weights = integrate(
            "tensorproduct", prior_variable, levels=[300]*nvars)
        # samples = prior_variable.rvs(int(1e6))
        # inner_pred_weights = np.full(
        #     (samples.shape[1], 1), 1/samples.shape[1])
        many_pred_obs = obs_model(samples).T
        oed_loglike = OEDGaussianLogLikelihood(
            loglike, many_pred_obs, inner_pred_weights)

        design_weights = np.ones((nobs, 1))
        errors = oed_loglike.check_apply_jacobian(design_weights, disp=True)
        assert errors.min()/errors.max() < 1e-6

        evidence_model = Evidence(oed_loglike)
        evidence = evidence_model(design_weights)
        assert evidence.shape[0] == ntrue_samples
        errors = evidence_model.check_apply_jacobian(design_weights, disp=True)
        assert errors.min()/errors.max() < 1e-6

        log_evidence_model = LogEvidence(oed_loglike)
        log_evidence = log_evidence_model(design_weights)
        assert log_evidence.shape[0] == ntrue_samples
        errors = log_evidence_model.check_apply_jacobian(
            design_weights, disp=True)
        assert errors.min()/errors.max() < 1e-6

        inner_pred_obs = many_pred_obs
        oed_objective = KLOEDObjective(
            noise_cov_diag, outer_pred_obs, outer_pred_weights,
            noise_samples, inner_pred_obs, inner_pred_weights)
        errors = oed_objective.check_apply_jacobian(design_weights, disp=True)
        assert errors.min()/errors.max() < 1e-6

        prior_mean = prior_variable.get_statistics("mean")
        prior_cov = np.diag(prior_variable.get_statistics("std")[:, 0]**2)
        prior_cov_inv = np.linalg.inv(prior_cov)
        noise_cov = np.diag(noise_cov_diag[:, 0])
        noise_cov_inv = np.linalg.inv(noise_cov)
        kl_divs = []
        # todo write test that compares multiple evaluations of evidence
        # with single obs to one evaluation of evidence with many obs
        oed_evidences = np.exp(oed_objective._log_evidence(design_weights))
        for obs_idx in range(ntrue_samples):
            post_mean, post_cov = \
                laplace_posterior_approximation_for_linear_models(
                    obs_model.jacobian(true_samples[:, obs_idx:obs_idx+1]),
                    prior_mean, prior_cov_inv,
                    noise_cov_inv, obs[:, obs_idx:obs_idx+1])
            kl_div = gaussian_kl_divergence(
                post_mean, post_cov, prior_mean, prior_cov)
            kl_divs.append(kl_div)

            single_obs_loglike = SingleObsIndependentGaussianLogLikelihood(
                noise_cov_diag)
            single_obs_loglike.set_observations(obs[:, obs_idx:obs_idx+1])
            evidence = laplace_evidence(
                lambda x: np.exp(single_obs_loglike(obs_model(x).T)[:, 0]),
                prior_variable.pdf, post_cov, post_mean)
            assert np.allclose(evidence, oed_evidences[obs_idx])

        kl_divs = np.array(kl_divs)[:, None]
        numeric_expected_kl_div = np.sum(kl_divs*outer_pred_weights)
        nu_vec, Cmat = posterior_mean_data_stats(
            prior_mean, prior_cov, prior_cov_inv, post_cov,
            obs_model.jacobian(true_samples[:, obs_idx:obs_idx+1]),
            noise_cov, noise_cov_inv)
        expected_kl_div = expected_kl_divergence_gaussian_inference(
            prior_mean, prior_cov, prior_cov_inv, post_cov, Cmat, nu_vec)
        assert np.allclose(numeric_expected_kl_div, expected_kl_div, rtol=1e-2)

        # print(oed_objective(design_weights), 'oed objective')
        # print(expected_kl_div, 'expected kl div')
        # print(numeric_expected_kl_div, 'numeric expected kl div')
        assert np.allclose(
            expected_kl_div, -oed_objective(design_weights), rtol=1e-2)

        bounds = Bounds(np.zeros((nobs,)), np.ones((nobs,)))

        nfinal_obs = 1
        constraint = WeightsConstraint(nfinal_obs)
        x0 = np.full((nobs, 1), 0.5)
        constraint._model.check_apply_jacobian(x0)
        optimizer = ScipyConstrainedOptimizer(
            oed_objective, bounds=bounds, constraints=[constraint],
            opts={"gtol": 1e-6, "verbose": 3, "maxiter": 200})
        result = optimizer.minimize(x0)
        print(result.x, result.fun)


if __name__ == '__main__':
    unittest.main()
