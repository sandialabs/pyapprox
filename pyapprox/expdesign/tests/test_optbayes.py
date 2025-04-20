import unittest

import numpy as np
from scipy import stats

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.interface.model import Model
from pyapprox.bayes.likelihood import (
    IndependentGaussianLogLikelihood,
    ModelBasedIndependentGaussianLogLikelihood,
)
from pyapprox.expdesign.optbayes import (
    OEDGaussianLogLikelihood,
    Evidence,
    LogEvidence,
    KLOEDObjective,
    WeightsConstraint,
    SparseOEDObjective,
    DOptimalLinearModelObjective,
    PredictionOEDObjective,
    NoiseStatistic,
)
from pyapprox.bayes.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.optimization.minimize import (
    LinearConstraint,
    SampleAverageMean,
    SampleAverageMeanPlusStdev,
    SampleAverageEntropicRisk,
)
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.surrogates.affine.basis import (
    setup_tensor_product_gauss_quadrature_rule,
)


class Linear1DRegressionModel(Model):
    def __init__(self, design, degree, min_degree=0, backend=NumpyMixin):
        super().__init__(backend=backend)
        assert degree >= min_degree
        self._design = design
        self._degree = degree
        self._jac_matrix = self._design.T ** (
            self._bkd.arange(min_degree, self._degree + 1)[None, :]
        )

    def jacobian_implemented(self):
        return True

    def nqoi(self):
        return self._jac_matrix.shape[0]

    def nvars(self):
        return self._jac_matrix.shape[1]

    def _values(self, samples):
        return (self._jac_matrix @ (samples)).T

    def _jacobian(self, sample):
        return self._jac_matrix


class TestBayesOED:

    def setUp(self):
        np.random.seed(1)

    def test_OED_gaussian_likelihood(self):
        bkd = self.get_backend()
        degree = 0
        nvars = degree + 1
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, 1)] * nvars, backend=bkd
        )

        nobs = 4
        design = bkd.linspace(-1, 1, nobs)[None, :]
        noise_cov_diag = bkd.full((nobs, 1), 0.3**2)
        obs_model = Linear1DRegressionModel(design, degree, backend=bkd)
        loglike = IndependentGaussianLogLikelihood(noise_cov_diag, backend=bkd)
        # loglike = ModelBasedIndependentGaussianLogLikelihood(
        #     obs_model, noise_cov_diag
        # )

        np.random.seed(1)
        ntrue_samples = 10000
        true_samples = prior_variable.rvs(ntrue_samples)
        outer_pred_weights = bkd.full((ntrue_samples, 1), 1 / ntrue_samples)
        outer_pred_obs = obs_model(true_samples).T
        np.random.seed(1)
        noise_samples = loglike._sample_noise(ntrue_samples)
        obs = loglike._make_noisy(outer_pred_obs, noise_samples)
        loglike.set_observations(obs)

        quad_rule = setup_tensor_product_gauss_quadrature_rule(prior_variable)
        samples, inner_pred_weights = quad_rule([300] * nvars)
        # samples = prior_variable.rvs(int(1e6))
        # inner_pred_weights = bkd.full(
        #     (samples.shape[1], 1), 1/samples.shape[1])
        many_pred_obs = obs_model(samples).T
        print(many_pred_obs, "a")
        oed_loglike = OEDGaussianLogLikelihood(
            loglike, many_pred_obs, inner_pred_weights
        )

        design_weights = bkd.ones((nobs, 1))
        errors = oed_loglike.check_apply_jacobian(design_weights, disp=True)
        assert errors.min() / errors.max() < 1e-6

        evidence_model = Evidence(oed_loglike)
        evidence = evidence_model(design_weights)
        assert evidence.shape[0] == ntrue_samples
        errors = evidence_model.check_apply_jacobian(design_weights, disp=True)
        assert errors.min() / errors.max() < 1e-6

        log_evidence_model = LogEvidence(oed_loglike)
        log_evidence = log_evidence_model(design_weights)
        assert log_evidence.shape[0] == ntrue_samples
        errors = log_evidence_model.check_apply_jacobian(
            design_weights, disp=True
        )
        assert errors.min() / errors.max() < 1e-6

        inner_pred_obs = many_pred_obs
        oed_objective = KLOEDObjective(
            noise_cov_diag,
            outer_pred_obs,
            outer_pred_weights,
            noise_samples,
            inner_pred_obs,
            inner_pred_weights,
        )
        errors = oed_objective.check_apply_jacobian(design_weights, disp=True)
        assert errors.min() / errors.max() < 1e-6

        prior_mean = prior_variable.get_statistics("mean")
        prior_cov = bkd.diag(prior_variable.get_statistics("std")[:, 0] ** 2)
        prior_cov_inv = bkd.linalg.inv(prior_cov)
        noise_cov = bkd.diag(noise_cov_diag[:, 0])
        noise_cov_inv = bkd.linalg.inv(noise_cov)
        kl_divs = []
        # todo write test that compares multiple evaluations of evidence
        # with single obs to one evaluation of evidence with many obs
        oed_evidences = bkd.exp(oed_objective._log_evidence(design_weights))
        for obs_idx in range(ntrue_samples):
            post_mean, post_cov = (
                laplace_posterior_approximation_for_linear_models(
                    obs_model.jacobian(true_samples[:, obs_idx : obs_idx + 1]),
                    prior_mean,
                    prior_cov_inv,
                    noise_cov_inv,
                    obs[:, obs_idx : obs_idx + 1],
                )
            )
            kl_div = gaussian_kl_divergence(
                post_mean, post_cov, prior_mean, prior_cov
            )
            kl_divs.append(kl_div)

            single_obs_loglike = SingleObsIndependentGaussianLogLikelihood(
                noise_cov_diag
            )
            single_obs_loglike.set_observations(obs[:, obs_idx : obs_idx + 1])
            evidence = laplace_evidence(
                lambda x: bkd.exp(single_obs_loglike(obs_model(x).T)[:, 0]),
                prior_variable.pdf,
                post_cov,
                post_mean,
            )
            assert bkd.allclose(evidence, oed_evidences[obs_idx])

        kl_divs = bkd.array(kl_divs)[:, None]
        numeric_expected_kl_div = bkd.sum(kl_divs * outer_pred_weights)
        nu_vec, Cmat = posterior_mean_data_stats(
            prior_mean,
            prior_cov,
            prior_cov_inv,
            post_cov,
            obs_model.jacobian(true_samples[:, obs_idx : obs_idx + 1]),
            noise_cov,
            noise_cov_inv,
        )
        expected_kl_div = expected_kl_divergence_gaussian_inference(
            prior_mean, prior_cov, prior_cov_inv, post_cov, Cmat, nu_vec
        )
        assert bkd.allclose(
            numeric_expected_kl_div, expected_kl_div, rtol=1e-2
        )

        # print(oed_objective(design_weights), 'oed objective')
        # print(expected_kl_div, 'expected kl div')
        # print(numeric_expected_kl_div, 'numeric expected kl div')
        assert bkd.allclose(
            expected_kl_div, -oed_objective(design_weights), rtol=1e-2
        )

    def _check_classical_KL_OED_gaussian_optimization(
        self, nobs, min_degree, degree, nout_samples, level1d
    ):
        bkd = self.get_backend()
        nvars = degree - min_degree + 1
        # the smaller the noise the more number of nout_samples are needed
        noise_std = 0.125 * 4
        prior_std = 0.5
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, prior_std)] * nvars
        )
        design = bkd.linspace(-1, 1, nobs - 2)[None, :]
        design = bkd.sort(
            bkd.hstack(
                (design[0], bkd.asarray([-1 / np.sqrt(5), 1 / np.sqrt(5)]))
            )
        )[None, :]
        noise_cov_diag = bkd.full((nobs, 1), noise_std**2)
        obs_model = Linear1DRegressionModel(
            design, degree, min_degree=min_degree
        )
        loglike = IndependentGaussianLogLikelihood(noise_cov_diag)

        true_samples = prior_variable.rvs(nout_samples)
        outer_pred_weights = bkd.full((nout_samples, 1), 1 / nout_samples)
        outer_pred_obs = obs_model(true_samples).T
        noise_samples = loglike._sample_noise(nout_samples)

        if level1d is not None:
            samples, inner_pred_weights = integrate(
                "tensorproduct",
                prior_variable,
                levels=[level1d] * nvars,
                rule="quadratic",
            )
        else:
            samples = prior_variable.rvs(2 * int(bkd.sqrt(nout_samples)))
            inner_pred_weights = bkd.full(
                (samples.shape[1], 1), 1 / samples.shape[1]
            )
        many_pred_obs = obs_model(samples).T

        inner_pred_obs = many_pred_obs
        oed_objective = KLOEDObjective(
            noise_cov_diag,
            outer_pred_obs,
            outer_pred_weights,
            noise_samples,
            inner_pred_obs,
            inner_pred_weights,
        )

        bounds = bkd.stack((bkd.zeros((nobs,)), bkd.ones((nobs,))), axis=1)

        nfinal_obs = 1
        dopt_objective = DOptimalLinearModelObjective(
            obs_model, noise_cov_diag[0, 0], prior_std**2
        )
        constraint = LinearConstraint(
            bkd.ones((1, nobs)), nfinal_obs, nfinal_obs, keep_feasible=True
        )
        optimizer = ScipyConstrainedOptimizer(
            dopt_objective,
            bounds=bounds,
            constraints=[constraint],
            opts={"gtol": 1e-5, "verbose": 3, "maxiter": 200},
        )
        x0 = bkd.full((nobs, 1), nfinal_obs / nobs)
        errors = dopt_objective.check_apply_jacobian(x0, disp=True)
        assert errors.min() / errors.max() < 1e-6 and errors.max() < 10
        errors = dopt_objective.check_apply_hessian(x0, disp=True)
        assert errors.min() / errors.max() < 1e-6 and errors.max() < 10
        result = optimizer.minimize(x0)
        print(result.x, result.fun, result.x.sum(), result)

        II = bkd.hstack(
            [
                [0, nobs - 1],
                bkd.where(bkd.isclose(bkd.abs(design[0]), 1 / bkd.sqrt(5)))[0],
            ]
        )
        x0 = bkd.zeros((nobs, 1))
        x0[II] = 1.0
        print(dopt_objective(x0), oed_objective(x0))
        assert bkd.allclose(dopt_objective(x0), oed_objective(x0), rtol=1e-2)

        constraint = LinearConstraint(
            bkd.ones((1, nobs)), nfinal_obs, nfinal_obs, keep_feasible=True
        )
        objective = oed_objective
        x0 = bkd.full((nobs, 1), nfinal_obs / nobs)
        errors = objective.check_apply_jacobian(
            x0, disp=True, fd_eps=bkd.logspace(-13, bkd.log(0.2), 13)[::-1]
        )
        assert errors.min() / errors.max() < 7e-6, errors.min() / errors.max()
        # turn on hessian for testing hessian implementation, but
        # apply hessian is turned off because while it reduces
        # optimization iteration count but increases
        # run time because cost of each iteration increases
        objective._apply_hessian_implemented = True
        errors = objective.check_apply_hessian(
            x0, disp=True, fd_eps=bkd.logspace(-13, bkd.log(0.2), 13)[::-1]
        )
        assert errors.min() / errors.max() < 3e-6 and errors.max() < 10
        objective._apply_hessian_implemented = False

        if isinstance(constraint, WeightsConstraint):
            errors = constraint._model.check_apply_jacobian(
                x0, disp=True, fd_eps=bkd.logspace(-13, -1, 13)[::-1]
            )
            assert errors.min() / errors.max() < 1e-6 and errors.max() < 10
        optimizer = ScipyConstrainedOptimizer(
            objective,
            bounds=bounds,
            constraints=[constraint],
            opts={"gtol": 1e-5, "verbose": 3, "maxiter": 200},
        )
        result = optimizer.minimize(x0)

    def test_classical_KL_OED_gaussian_optimization(self):
        test_cases = [
            [3, 0, 1, 4000, 50],
            [3, 1, 1, 4000, 50],
            [3, 0, 3, 50000, None],
        ]
        for test_case in test_cases:
            self._check_classical_KL_OED_gaussian_optimization(*test_case)

    def _check_KL_OED_gaussian_optimization(
        self, nobs, min_degree, degree, nout_samples, level1d, noise_stat
    ):
        bkd = self.get_backend()
        nvars = degree - min_degree + 1
        # the smaller the noise the more number of nout_samples are needed
        noise_std = 0.125 * 4
        prior_std = 0.5
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, prior_std)] * nvars
        )
        design = bkd.linspace(-1, 1, nobs - 2)[None, :]
        design = bkd.sort(
            bkd.hstack((design[0], [-1 / bkd.sqrt(5), 1 / bkd.sqrt(5)]))
        )[None, :]
        noise_cov_diag = bkd.full((nobs, 1), noise_std**2)
        obs_model = Linear1DRegressionModel(
            design, degree, min_degree=min_degree
        )
        loglike = IndependentGaussianLogLikelihood(noise_cov_diag)

        true_samples = prior_variable.rvs(nout_samples)
        outer_pred_weights = bkd.full((nout_samples, 1), 1 / nout_samples)
        outer_pred_obs = obs_model(true_samples).T
        noise_samples = loglike._sample_noise(nout_samples)

        if level1d is not None:
            samples, inner_pred_weights = integrate(
                "tensorproduct",
                prior_variable,
                levels=[level1d] * nvars,
                rule="quadratic",
            )
        else:
            samples = prior_variable.rvs(2 * int(bkd.sqrt(nout_samples)))
            inner_pred_weights = bkd.full(
                (samples.shape[1], 1), 1 / samples.shape[1]
            )
        many_pred_obs = obs_model(samples).T

        inner_pred_obs = many_pred_obs
        oed_objective = KLOEDObjective(
            noise_cov_diag,
            outer_pred_obs,
            outer_pred_weights,
            noise_samples,
            inner_pred_obs,
            inner_pred_weights,
            noise_stat=noise_stat,
        )

        bounds = bkd.stack((bkd.zeros((nobs,)), bkd.ones((nobs,))), axis=1)

        nfinal_obs = 1
        constraint = LinearConstraint(
            bkd.ones((1, nobs)), nfinal_obs, nfinal_obs, keep_feasible=True
        )
        objective = oed_objective
        x0 = bkd.full((nobs, 1), nfinal_obs / nobs)
        errors = objective.check_apply_jacobian(
            x0, disp=True, fd_eps=bkd.logspace(-13, bkd.log(0.2), 13)[::-1]
        )
        assert errors.min() / errors.max() < 6e-6, errors.min() / errors.max()

        if isinstance(constraint, WeightsConstraint):
            errors = constraint._model.check_apply_jacobian(
                x0, disp=True, fd_eps=bkd.logspace(-13, -1, 13)[::-1]
            )
            assert errors.min() / errors.max() < 1e-6 and errors.max() < 10
        optimizer = ScipyConstrainedOptimizer(
            objective,
            bounds=bounds,
            constraints=[constraint],
            opts={"gtol": 1e-5, "verbose": 3, "maxiter": 200},
        )
        result = optimizer.minimize(x0)
        print(result.x)

    def test_KL_OED_gaussian_optimization(self):
        test_cases = [
            [3, 0, 1, 4000, 50, NoiseStatistic(SampleAverageMean())],
            [3, 0, 1, 4000, 50, NoiseStatistic(SampleAverageMeanPlusStdev(1))],
            [
                3,
                0,
                1,
                4000,
                50,
                NoiseStatistic(SampleAverageEntropicRisk(0.5)),
            ],
        ]
        for test_case in test_cases:
            self._check_KL_OED_gaussian_optimization(*test_case)

    def _check_prediction_gaussian_OED(
        self, nobs, min_degree, degree, nout_samples, level1d, noise_stat
    ):
        bkd = self.get_backend()
        nvars = degree - min_degree + 1
        # the smaller the noise the more number of nout_samples are needed
        noise_std = 0.125 * 4
        prior_std = 0.5
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, prior_std)] * nvars
        )
        design = bkd.linspace(-1, 1, nobs - 2)[None, :]
        design = bkd.sort(
            bkd.hstack((design[0], [-1 / bkd.sqrt(5), 1 / bkd.sqrt(5)]))
        )[None, :]
        noise_cov_diag = bkd.full((nobs, 1), noise_std**2)
        obs_model = Linear1DRegressionModel(
            design, degree, min_degree=min_degree
        )
        loglike = IndependentGaussianLogLikelihood(noise_cov_diag)

        true_samples = prior_variable.rvs(nout_samples)
        outer_pred_weights = bkd.full((nout_samples, 1), 1 / nout_samples)
        outer_pred_obs = obs_model(true_samples).T
        noise_samples = loglike._sample_noise(nout_samples)

        if level1d is not None:
            samples, inner_pred_weights = integrate(
                "tensorproduct",
                prior_variable,
                levels=[level1d] * nvars,
                rule="quadratic",
            )
        else:
            samples = prior_variable.rvs(2 * int(bkd.sqrt(nout_samples)))
            inner_pred_weights = bkd.full(
                (samples.shape[1], 1), 1 / samples.shape[1]
            )
        many_pred_obs = obs_model(samples).T

        inner_pred_obs = many_pred_obs
        oed_objective = PredictionOEDObjective(
            noise_cov_diag,
            outer_pred_obs,
            outer_pred_weights,
            noise_samples,
            inner_pred_obs,
            inner_pred_weights,
            noise_stat=noise_stat,
        )

        bounds = bkd.stack((bkd.zeros((nobs,)), bkd.ones((nobs,))), axis=1)

        nfinal_obs = 1
        constraint = LinearConstraint(
            bkd.ones((1, nobs)), nfinal_obs, nfinal_obs, keep_feasible=True
        )
        objective = oed_objective
        x0 = bkd.full((nobs, 1), nfinal_obs / nobs)
        errors = objective.check_apply_jacobian(
            x0, disp=True, fd_eps=bkd.logspace(-13, bkd.log(0.2), 13)[::-1]
        )
        assert errors.min() / errors.max() < 6e-6, errors.min() / errors.max()

        if isinstance(constraint, WeightsConstraint):
            errors = constraint._model.check_apply_jacobian(
                x0, disp=True, fd_eps=bkd.logspace(-13, -1, 13)[::-1]
            )
            assert errors.min() / errors.max() < 1e-6 and errors.max() < 10
        optimizer = ScipyConstrainedOptimizer(
            objective,
            bounds=bounds,
            constraints=[constraint],
            opts={"gtol": 1e-5, "verbose": 3, "maxiter": 200},
        )
        result = optimizer.minimize(x0)
        print(result.x)

    @unittest.skip("Implementation not finished")
    def test_prediction_gaussian_OED(self):
        test_cases = [
            [3, 0, 1, 4000, 50, NoiseStatistic(SampleAverageMean())],
            [3, 0, 1, 4000, 50, NoiseStatistic(SampleAverageMeanPlusStdev(1))],
            [
                3,
                0,
                1,
                4000,
                50,
                NoiseStatistic(SampleAverageEntropicRisk(0.5)),
            ],
        ]
        for test_case in test_cases:
            self._check_prediction_gaussian_OED(*test_case)


# class TestNumpyBayesOED(TestBayesOED, unittest.TestCase):
#     def get_backend(self):
#         return NumpyMixin


class TestTorchBayesOED(TestBayesOED, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
