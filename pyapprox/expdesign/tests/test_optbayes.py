import unittest

import numpy as np
from scipy import stats

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.bayes.likelihood import (
    ModelBasedIndependentGaussianLogLikelihood,
)
from pyapprox.expdesign.optbayes import (
    IndependentGaussianOEDOuterLoopLogLikelihood,
    IndependentGaussianOEDInnerLoopLogLikelihood,
    Evidence,
    LogEvidence,
    KLOEDObjective,
    DOptimalLinearModelObjective,
    PredictionOEDObjective,
    NoiseStatistic,
    BayesianOED,
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
from pyapprox.surrogates.affine.basis import (
    setup_tensor_product_gauss_quadrature_rule,
    setup_tensor_product_piecewise_poly_quadrature_rule,
)
from pyapprox.optimization.risk import multivariate_gaussian_kl_divergence
from pyapprox.interface.model import DenseMatrixLinearModel


class Linear1DRegressionModel(DenseMatrixLinearModel):
    def __init__(
        self, design, degree: int, min_degree: int = 0, backend=NumpyMixin
    ):
        assert degree >= min_degree
        self._design = design
        self._degree = degree
        super().__init__(
            self._design.T
            ** backend.arange(min_degree, self._degree + 1)[None, :],
            backend=backend,
        )


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

        # generate observations to compute expected information gain
        nouterloop_samples = 5  # 10000
        outerloop_samples = prior_variable.rvs(nouterloop_samples)
        loglike = ModelBasedIndependentGaussianLogLikelihood(
            obs_model, noise_cov_diag
        )
        outerloop_shapes = obs_model(outerloop_samples).T
        obs = loglike.rvs_from_shapes(outerloop_shapes)

        # setup OED log likelihood
        outerloop_loglike = IndependentGaussianOEDOuterLoopLogLikelihood(
            noise_cov_diag, backend=bkd
        )
        outerloop_loglike.set_observations_and_shapes(obs, outerloop_shapes)

        # compute the likelihood values with traditional loglikelihood
        # which is a function of the shapes
        design_weights = bkd.array(np.random.uniform(0.25, 1.0, (nobs, 1)))
        loglike.set_design_weights(design_weights)
        vals = []
        for ii in range(obs.shape[1]):
            loglike.set_observations(obs[:, ii : ii + 1])
            # unlike likelihoods from pyapprox.bayes.likelihood, which take
            # obs and shapes with different shapes, obs and shapes passed to
            # OED likelihoods require obs and shapes with the same shape.
            # Thus we only take ii entry for the ith observation
            vals.append(loglike._loglike_from_shapes(outerloop_shapes)[ii, 0])
        # predict the values with the OED likelihood.
        assert bkd.allclose(
            outerloop_loglike(design_weights), bkd.hstack(vals)
        )

        # check the gradients of the OED likelihood
        design_weights = bkd.ones((nobs, 1))
        errors = outerloop_loglike.check_apply_jacobian(
            design_weights, disp=False
        )
        assert errors.min() / errors.max() < 1e-6

        # check the simultaneous computation of evidences for different
        # realizations of the data
        ninnerloop_samples = 3
        innerloop_samples = prior_variable.rvs(ninnerloop_samples)
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            noise_cov_diag, backend=bkd
        )
        innerloop_shapes = obs_model(innerloop_samples).T
        innerloop_loglike.set_observations_and_shapes(obs, innerloop_shapes)
        evidence = Evidence(innerloop_loglike, innerloop_samples)
        evidence_vals = evidence(design_weights)
        assert evidence_vals.shape == (1, nouterloop_samples)
        errors = evidence.check_apply_jacobian(design_weights, disp=True)
        assert errors.min() / errors.max() < 1e-6

        log_evidence_model = LogEvidence(oed_loglike)
        log_evidence = log_evidence_model(design_weights)
        assert log_evidence.shape == (1, nouterloop_samples)
        errors = log_evidence_model.check_apply_jacobian(
            design_weights, disp=False
        )
        assert errors.min() / errors.max() < 1e-6

        inner_pred_obs = many_pred_obs
        oed_objective = KLOEDObjective(
            noise_cov_diag,
            outer_pred_obs,
            outer_pred_weights,
            inner_pred_obs,
            inner_pred_weights,
            backend=bkd,
        )
        errors = oed_objective.check_apply_jacobian(design_weights, disp=False)
        assert errors.min() / errors.max() < 1e-6
        # check hessian code works. But currently apply_hessian_implemented
        # is set to False because the code is slow. Temporarily
        # activate hessian computation
        oed_objective.apply_hessian_implemented = lambda: True
        errors = oed_objective.check_apply_hessian(design_weights, disp=False)
        assert errors.min() / errors.max() < 1e-6
        # disable hesian computation
        oed_objective.apply_hessian_implemented = lambda: False

        noise_cov = bkd.diag(noise_cov_diag[:, 0])
        kl_divs = []
        # todo write test that compares multiple evaluations of evidence
        # with single obs to one evaluation of evidence with many obs
        oed_evidences = bkd.exp(oed_objective._log_evidence(design_weights))[0]
        for obs_idx in range(nouterloop_samples):
            laplace = DenseMatrixLaplacePosteriorApproximation(
                obs_model.jacobian(true_samples[:, obs_idx : obs_idx + 1]),
                prior_variable.mean(),
                prior_variable.covariance(),
                noise_cov,
                backend=bkd,
            )
            laplace.compute(
                oed_objective._outer_oed_loglike._obs[:, obs_idx : obs_idx + 1]
            )
            kl_div = multivariate_gaussian_kl_divergence(
                laplace.posterior_mean(),
                laplace.posterior_covariance(),
                prior_variable.mean(),
                prior_variable.covariance(),
                bkd=bkd,
            )
            kl_divs.append(kl_div)
            assert bkd.allclose(laplace.evidence(), oed_evidences[obs_idx])

        kl_divs = bkd.array(kl_divs)[:, None]
        numeric_expected_kl_div = bkd.sum(kl_divs * outer_pred_weights)
        expected_kl_div = laplace.expected_kl_divergence()
        assert bkd.allclose(
            numeric_expected_kl_div, expected_kl_div, rtol=1e-2
        )
        assert bkd.allclose(
            expected_kl_div, -oed_objective(design_weights), rtol=1e-2
        )

    def _check_KL_OED(
        self, nobs, min_degree, degree, nout_samples, level1d, noise_stat
    ):
        bkd = self.get_backend()
        nvars = degree - min_degree + 1
        # the smaller the noise the more number of nout_samples are needed
        noise_std = 0.125 * 4
        prior_std = 0.5
        prior_variable = IndependentMarginalsVariable(
            [stats.norm(0, prior_std)] * nvars, backend=bkd
        )
        design = bkd.linspace(-1, 1, nobs - 2)[None, :]
        design = bkd.sort(
            bkd.hstack(
                (design[0], bkd.asarray([-1 / np.sqrt(5), 1 / np.sqrt(5)]))
            )
        )[None, :]
        noise_cov_diag = bkd.full((nobs, 1), noise_std**2)
        obs_model = Linear1DRegressionModel(
            design, degree, min_degree=min_degree, backend=bkd
        )

        true_samples = prior_variable.rvs(nout_samples)
        outer_pred_weights = bkd.full((nout_samples, 1), 1 / nout_samples)
        outer_pred_obs = obs_model(true_samples).T
        # todo need to create inside kle oed objective
        # noise_samples = loglike._sample_noise(nout_samples)

        if level1d is not None:
            quad_rule = setup_tensor_product_piecewise_poly_quadrature_rule(
                prior_variable, ["quadratic"] * nvars, weighted=True
            )
            samples, inner_pred_weights = quad_rule([level1d] * nvars)
        else:
            samples = prior_variable.rvs(2 * int(np.sqrt(nout_samples)))
            inner_pred_weights = bkd.full(
                (samples.shape[1], 1), 1 / samples.shape[1]
            )
        many_pred_obs = obs_model(samples).T

        inner_pred_obs = many_pred_obs
        kl_oed_objective = KLOEDObjective(
            noise_cov_diag,
            outer_pred_obs,
            outer_pred_weights,
            inner_pred_obs,
            inner_pred_weights,
            noise_stat=noise_stat,
            backend=bkd,
        )

        x0 = bkd.full((nobs, 1), 1 / nobs)
        errors = kl_oed_objective.check_apply_jacobian(
            x0, disp=False, fd_eps=bkd.flip(bkd.logspace(-13, np.log(0.2), 13))
        )
        assert errors.min() / errors.max() < 7e-6, errors.min() / errors.max()
        # turn on hessian for testing hessian implementation, but
        # apply hessian is turned off because while it reduces
        # optimization iteration count but increases
        # run time because cost of each iteration increases
        if isinstance(noise_stat._stat, SampleAverageMean):
            kl_oed_objective.apply_hessian_implemented = lambda: True
            errors = kl_oed_objective.check_apply_hessian(
                x0,
                disp=False,
                fd_eps=bkd.flip(bkd.logspace(-13, np.log(0.2), 13)),
            )
            assert errors.min() / errors.max() < 5e-6 and errors.max() < 10
            kl_oed_objective.apply_hessian_implemented = lambda: False

        # just test optimization runs
        kl_oed = BayesianOED(kl_oed_objective)
        kl_oed.compute()

        return design, noise_cov_diag, prior_std, kl_oed_objective, obs_model

    def _check_classical_KL_OED_gaussian_optimization(
        self, nobs, min_degree, degree, nout_samples, level1d
    ):
        design, noise_cov_diag, prior_std, kl_oed_objective, obs_model = (
            self._check_KL_OED(
                nobs, min_degree, degree, nout_samples, level1d, None
            )
        )
        bkd = self.get_backend()
        dopt_objective = DOptimalLinearModelObjective(
            obs_model, noise_cov_diag[0, 0], bkd.array(prior_std**2)
        )
        x0 = bkd.full((nobs, 1), 1 / nobs)
        errors = dopt_objective.check_apply_jacobian(x0, disp=False)
        assert errors.min() / errors.max() < 1e-6 and errors.max() < 10
        errors = dopt_objective.check_apply_hessian(x0, disp=False)
        assert errors.min() / errors.max() < 1e-6 and errors.max() < 10

        II = bkd.hstack(
            [
                bkd.array([0, nobs - 1], dtype=int),
                bkd.where(
                    bkd.isclose(bkd.abs(design[0]), 1 / bkd.sqrt(bkd.array(5)))
                )[0],
            ]
        )
        x0 = bkd.zeros((nobs, 1))
        x0[II] = 1.0
        # print(dopt_objective(x0), kl_oed_objective(x0))
        assert bkd.allclose(
            dopt_objective(x0), kl_oed_objective(x0), rtol=1e-2
        )

        # just test optimizations run because of noise in monte carlo
        # I am not sure if I can ensure that klopt and dopt designs
        # will be the same.Equivalence would also only occur if
        # noise_stat = SampleAverageMean
        # kl_oed = BayesianOED(kl_oed_objective)
        # klopt_design = kl_oed.compute()

        dopt_oed = BayesianOED(dopt_objective)
        dopt_design = dopt_oed.compute()

    def test_classical_KL_OED_gaussian_optimization(self):
        test_cases = [
            [3, 0, 1, 4000, 51],
            [3, 1, 1, 4000, 51],
            [3, 0, 3, 50000, None],
        ]
        for test_case in test_cases:
            self._check_classical_KL_OED_gaussian_optimization(*test_case)

    def test_KL_OED_gaussian_optimization(self):
        bkd = self.get_backend()
        test_cases = [
            [3, 0, 1, 4000, 51, NoiseStatistic(SampleAverageMean(bkd))],
            [
                3,
                0,
                1,
                4000,
                51,
                NoiseStatistic(SampleAverageMeanPlusStdev(1, bkd)),
            ],
            [
                3,
                0,
                1,
                4000,
                51,
                NoiseStatistic(SampleAverageEntropicRisk(0.5, bkd)),
            ],
        ]
        for test_case in test_cases:
            print(test_case)
            self._check_KL_OED(*test_case)

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
            bkd.hstack(
                (design[0], bkd.asarray([-1 / np.sqrt(5), 1 / np.sqrt(5)]))
            )
        )[None, :]
        noise_cov_diag = bkd.full((nobs, 1), noise_std**2)
        obs_model = Linear1DRegressionModel(
            design, degree, min_degree=min_degree
        )

        true_samples = prior_variable.rvs(nout_samples)
        outer_pred_weights = bkd.full((nout_samples, 1), 1 / nout_samples)
        outer_pred_obs = obs_model(true_samples).T

        if level1d is not None:
            quad_rule = setup_tensor_product_piecewise_poly_quadrature_rule(
                prior_variable, ["quadratic"] * nvars, weighted=True
            )
            samples, inner_pred_weights = quad_rule([level1d] * nvars)
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
            inner_pred_obs,
            inner_pred_weights,
            noise_stat=noise_stat,
        )

        x0 = bkd.full((nobs, 1), 1 / nobs)
        errors = oed_objective.check_apply_jacobian(
            x0, disp=True, fd_eps=bkd.flip(bkd.logspace(-13, np.log(0.2), 13))
        )
        assert errors.min() / errors.max() < 6e-6, errors.min() / errors.max()

        # test optimization runs
        oed = BayesianOED(oed_objective)
        design = oed.compute()

    def test_prediction_gaussian_OED(self):
        test_cases = [
            [3, 0, 1, 4000, 51, NoiseStatistic(SampleAverageMean())],
            [3, 0, 1, 4000, 51, NoiseStatistic(SampleAverageMeanPlusStdev(1))],
            [
                3,
                0,
                1,
                4000,
                51,
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
