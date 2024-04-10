import unittest

import numpy as np
from scipy import stats

from pyapprox.interface.model import Model
from pyapprox.bayes.likelihood import (
    IndependentGaussianLogLikelihood,
    SingleObsIndependentGaussianLogLikelihood)
from pyapprox.expdesign.optbayes import (
    OEDGaussianLogLikelihood, Evidence, LogEvidence, KLOEDObjective,
    WeightsConstraint, SparseOEDObjective, DOptimalLinearModelObjective)
from pyapprox.bayes.laplace import (
    laplace_posterior_approximation_for_linear_models, laplace_evidence)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.integrate import integrate
from pyapprox.expdesign.tests.test_bayesian_oed import (
    expected_kl_divergence_gaussian_inference,
    posterior_mean_data_stats, gaussian_kl_divergence)
from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer, Bounds, LinearConstraint)


class Linear1DRegressionModel(Model):
    def __init__(self, design, degree, min_degree=0):
        super().__init__()
        assert degree >= min_degree
        self._design = design
        self._degree = degree
        self._jac_matrix = self._design.T**(
            np.arange(min_degree, self._degree+1)[None, :])
        print(self._jac_matrix.shape)
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

    def test_OED_gaussian_optimization(self):
        nobs = 20
        # min_degree, degree, nout_samples, level1d = 1, 1, 10000, 300
        # min_degree, degree, nout_samples, level1d = 0, 1, 10000, 30
        min_degree, degree, nout_samples, level1d = 0, 3, 10000, None
        # min_degree, degree, nout_samples, level1d = 0, 2, 10000, None
        nvars = degree-min_degree+1
        # noise_std = 0.02
        noise_std = 0.1
        prior_std = 1
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, prior_std)]*nvars)
        design = np.linspace(-1, 1, nobs-2)[None, :]
        design = np.sort(np.hstack(
            (design[0], [-1/np.sqrt(5), 1/np.sqrt(5)])))[None, :]
        noise_cov_diag = np.full((nobs, 1), noise_std**2)
        obs_model = Linear1DRegressionModel(
            design, degree, min_degree=min_degree)
        loglike = IndependentGaussianLogLikelihood(noise_cov_diag)

        true_samples = prior_variable.rvs(nout_samples)
        outer_pred_weights = np.full((nout_samples, 1), 1/nout_samples)
        outer_pred_obs = obs_model(true_samples).T
        noise_samples = loglike._sample_noise(nout_samples)

        if level1d is not None:
            samples, inner_pred_weights = integrate(
                "tensorproduct", prior_variable, levels=[level1d]*nvars)
        else:
            samples = prior_variable.rvs(int(np.sqrt(nout_samples)))
            inner_pred_weights = np.full(
                (samples.shape[1], 1), 1/samples.shape[1])
        many_pred_obs = obs_model(samples).T

        inner_pred_obs = many_pred_obs
        oed_objective = KLOEDObjective(
            noise_cov_diag, outer_pred_obs, outer_pred_weights,
            noise_samples, inner_pred_obs, inner_pred_weights)

        bounds = Bounds(
            np.zeros((nobs,)), np.ones((nobs,)), keep_feasible=True)

        nfinal_obs = 1
        dopt_objective = DOptimalLinearModelObjective(
            obs_model, noise_cov_diag[0, 0], prior_std**2)
        constraint = LinearConstraint(
           np.ones((1, nobs)), nfinal_obs, nfinal_obs, keep_feasible=True)
        optimizer = ScipyConstrainedOptimizer(
            dopt_objective, bounds=bounds, constraints=[constraint],
            opts={"gtol": 1e-6, "verbose": 3, "maxiter": 200})
        x0 = np.full((nobs, 1), nfinal_obs/nobs)
        errors = dopt_objective.check_apply_jacobian(x0, disp=True)
        assert errors.min()/errors.max() < 1e-6
        result = optimizer.minimize(x0)
        print(result.x, result.fun, result.x.sum())
        # should choose 1.np.sqrt(5) when min_degree, degree=0,3
        print(design[0, result.x > 0.1], 1/np.sqrt(5))
        import matplotlib.pyplot as plt
        plt.plot(design[0], result.x, 'o')

        II = np.hstack(
            [[0], np.where(np.isclose(np.abs(design[0]), 1/np.sqrt(5))==True)[0],
             [nobs-1]])
        x0 = np.zeros((nobs, 1))
        x0[II] = 1.
        print(x0)
        print(dopt_objective(x0), oed_objective(x0))
        assert False
        plt.show()

        # constraint = WeightsConstraint(nfinal_obs, keep_feasible=True)
        constraint = LinearConstraint(
            np.ones((1, nobs)), nfinal_obs, nfinal_obs, keep_feasible=True)
        objective = oed_objective
        # objective = SparseOEDObjective(oed_objective, 1)
        x0 = np.full((nobs, 1), nfinal_obs/nobs)
        errors = objective.check_apply_jacobian(
            x0, disp=True, fd_eps=np.logspace(-13, np.log(0.5), 13)[::-1])
        assert errors.min()/errors.max() < 1e-6
        if isinstance(constraint, WeightsConstraint):
            errors = constraint._model.check_apply_jacobian(
                x0, disp=True, fd_eps=np.logspace(-13, -1, 13)[::-1])
            assert errors.min()/errors.max() < 1e-6
        optimizer = ScipyConstrainedOptimizer(
            objective, bounds=bounds, constraints=[constraint],
            opts={"gtol": 1e-6, "verbose": 3, "maxiter": 200})
        result = optimizer.minimize(x0)
        print(result.x, result.fun, result.x.sum())
        import matplotlib.pyplot as plt
        plt.plot(design[0], result.x, 'o')
        plt.show()


if __name__ == '__main__':
    unittest.main()
