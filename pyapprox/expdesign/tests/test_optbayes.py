import unittest
import itertools

import numpy as np
from scipy import stats

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.bayes.likelihood import (
    ModelBasedIndependentGaussianLogLikelihood,
)
from pyapprox.expdesign.optbayes import (
    IndependentGaussianOEDInnerLoopLogLikelihood,
    Evidence,
    LogEvidence,
    KLOEDObjective,
    DOptimalLinearModelObjective,
    PredictionOEDObjective,
    NoiseStatistic,
    KLBayesianOED,
    BayesianOEDForPrediction,
    OEDStandardDeviationMeasure,
    ConjugateGaussianPriorOEDForLinearPredictionKLDivergence,
    ConjugateGaussianPriorOEDForLinearPredictionStandardDeviation,
    ConjugateGaussianPriorOEDForLogNormalPredictionStandardDeviation,
)
from pyapprox.expdesign.sequences import HaltonSequence
from pyapprox.bayes.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
    GaussianPushForward,
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.optimization.minimize import (
    SampleAverageMean,
    SampleAverageMeanPlusStdev,
    SampleAverageEntropicRisk,
)
from pyapprox.surrogates.affine.basis import (
    setup_tensor_product_gauss_quadrature_rule,
    setup_tensor_product_piecewise_poly_quadrature_rule,
)
from pyapprox.interface.model import DenseMatrixLinearModel
from pyapprox.optimization.risk import LogNormalAnalyticalRiskMeasures


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

    def _check_expected_information_gain(
        self,
        prior_variable,
        oed_objective,
        outerloop_samples,
        design_weights,
        obs_model,
        noise_cov,
        tol,
    ):
        # Note old test used to also compare evidences computed by
        # OED objective. However, the accuracy of the evidences depends
        # only on the number of inner-loop evaluations, but we typically
        # make these much smaller than number of outerloop samples, which
        # dictates accuracy of the expected information gain much more.
        # So we no longer test evidences here

        bkd = self.get_backend()
        # compute laplace for any observation so we can compute
        # expected KL divergence which does not depend on the observation
        obs_idx = 0
        laplace = DenseMatrixLaplacePosteriorApproximation(
            obs_model.jacobian(outerloop_samples[:, obs_idx : obs_idx + 1]),
            prior_variable.mean(),
            prior_variable.covariance(),
            noise_cov,
            backend=bkd,
        )
        laplace.compute(
            oed_objective._outerloop_loglike._obs[:, obs_idx : obs_idx + 1]
        )
        print(
            "error in expected information gain",
            (-oed_objective(design_weights) - laplace.expected_kl_divergence())
            / laplace.expected_kl_divergence(),
        )
        assert bkd.allclose(
            laplace.expected_kl_divergence(),
            -oed_objective(design_weights),
            rtol=tol,
        )

    def _setup_quadrature_data(self, quadtype: str, prior, nsamples: int):
        bkd = self.get_backend()
        if quadtype == "MC":
            return prior.rvs(nsamples), bkd.full((nsamples, 1), 1.0 / nsamples)

        if quadtype == "Halton":
            sequence = HaltonSequence(
                prior.nvars(), 1, prior, prior._bkd, unbounded_eps=1e-3
            )
            return sequence.rvs(nsamples), bkd.full(
                (nsamples, 1), 1.0 / nsamples
            )

        if quadtype == "quadratic":
            quad_rule = setup_tensor_product_piecewise_poly_quadrature_rule(
                prior,
                ["quadratic"] * prior.nvars(),
                weighted=True,
                unbounded_alpha=1.0 - 1e-4,
            )
            level1d = int(nsamples ** (1 / prior.nvars()))
            if level1d % 2 == 0:
                level1d += 1
            print(level1d)
            return quad_rule([level1d] * prior.nvars())

        if quadtype == "gauss":
            quad_rule = setup_tensor_product_gauss_quadrature_rule(prior)
            level1d = int(nsamples ** (1 / prior.nvars()))
            return quad_rule([level1d] * prior.nvars())

        raise ValueError(f"{quadtype} not supported")

    def _setup_linear_gaussian_oed(
        self,
        outerloop_quadtype,
        nouterloop_samples,
        innerloop_quadtype,
        ninnerloop_samples,
        degree,
        nobs,
    ):
        bkd = self.get_backend()
        nvars = degree + 1
        prior = IndependentMarginalsVariable(
            [stats.norm(0, 1)] * nvars, backend=bkd
        )

        design = bkd.linspace(-1, 1, nobs)[None, :]
        noise_cov_diag = bkd.full((nobs, 1), 0.3**2)

        prior_data_variable = IndependentMarginalsVariable(
            prior.marginals()
            + [
                stats.norm(0, bkd.sqrt(variance))
                for variance in noise_cov_diag[:, 0]
            ],
            backend=bkd,
        )
        print(prior_data_variable)

        obs_model = Linear1DRegressionModel(design, degree, backend=bkd)

        # setup OED log likelihood
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            noise_cov_diag, backend=bkd
        )

        # generate observations to compute expected information gain
        outerloop_samples, outerloop_quad_weights = (
            self._setup_quadrature_data(
                outerloop_quadtype, prior_data_variable, nouterloop_samples
            )
        )
        outerloop_shapes_samples = outerloop_samples[: prior.nvars()]
        outerloop_shapes = obs_model(outerloop_shapes_samples).T
        outerloop_loglike = innerloop_loglike.outerloop_loglike()
        obs = outerloop_loglike.rvs_from_shapes(outerloop_shapes)
        outerloop_loglike.set_observations_and_shapes(obs, outerloop_shapes)
        nouterloop_samples = outerloop_loglike._shapes.shape[1]

        return (
            prior,
            obs_model,
            innerloop_loglike,
            noise_cov_diag,
            prior_data_variable,
            outerloop_samples,
            outerloop_quad_weights,
        )

    def test_OED_gaussian_likelihood_values(self):
        bkd = self.get_backend()

        outerloop_quadtype = "MC"
        innerloop_quadtype = "MC"
        nouterloop_samples = 100
        ninnerloop_samples = 100

        degree = 0
        nobs = 1
        (
            prior,
            obs_model,
            innerloop_loglike,
            noise_cov_diag,
            prior_data_variable,
            outerloop_samples,
            outerloop_quad_weights,
        ) = self._setup_linear_gaussian_oed(
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
            degree,
            nobs,
        )
        outerloop_loglike = innerloop_loglike.outerloop_loglike()

        # reset nouterloop_samples because tensor product quadrature rules
        # will not be able to match the number of requested samples exactly
        nouterloop_samples = outerloop_loglike._shapes.shape[1]

        # compute the likelihood values with traditional loglikelihood
        # which is a function of the shapes
        design_weights = bkd.array(np.random.uniform(0.5, 1.0, (nobs, 1)))
        loglike = ModelBasedIndependentGaussianLogLikelihood(
            obs_model, noise_cov_diag
        )
        loglike.set_design_weights(design_weights)
        vals = []
        obs = outerloop_loglike._obs
        outerloop_shapes = outerloop_loglike._shapes
        for ii in range(obs.shape[1]):
            loglike.set_observations(obs[:, ii : ii + 1])
            # unlike likelihoods from pyapprox.bayes.likelihood, which take
            # obs and shapes with different shapes, obs and shapes passed to
            # OED likelihoods require obs and shapes with the same shape.
            # Thus we only take ii entry for the ith observation
            vals.append(
                loglike._loglike_from_shapes(outerloop_shapes[:, ii : ii + 1])[
                    :, 0
                ]
            )
        # predict the values with the OED likelihood.
        assert bkd.allclose(
            outerloop_loglike(design_weights), bkd.hstack(vals)
        )

        # check the simultaneous computation of evidences for different
        # realizations of the data
        innerloop_samples, innerloop_quad_weights = (
            self._setup_quadrature_data(
                innerloop_quadtype, prior, ninnerloop_samples
            )
        )

        # reset ninnerloop_samples because tensor product quadrature rules
        # will not be able to match the number of requested samples exactly
        ninnerloop_samples = innerloop_samples.shape[1]
        innerloop_shapes = obs_model(innerloop_samples).T
        innerloop_loglike.set_observations_and_shapes(obs, innerloop_shapes)
        # check likelihood vals are computed correctly for each observation
        innerloop_loglike_vals = innerloop_loglike(design_weights)
        loglike = ModelBasedIndependentGaussianLogLikelihood(
            obs_model, noise_cov_diag
        )
        loglike.set_design_weights(design_weights)

        vals = []
        for ii in range(obs.shape[1]):
            loglike.set_observations(obs[:, ii : ii + 1])
            vals.append(loglike._loglike_from_shapes(innerloop_shapes)[:, 0])
        assert bkd.allclose(innerloop_loglike_vals, bkd.hstack(vals))

    def test_OED_gaussian_likelihood_gradients(self):
        bkd = self.get_backend()

        outerloop_quadtype = "MC"
        innerloop_quadtype = "MC"
        nouterloop_samples = 100
        ninnerloop_samples = 100

        degree = 0
        nobs = 3
        (
            prior,
            obs_model,
            innerloop_loglike,
            noise_cov_diag,
            prior_data_variable,
            outerloop_samples,
            outerloop_quad_weights,
        ) = self._setup_linear_gaussian_oed(
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
            degree,
            nobs,
        )

        outerloop_loglike = innerloop_loglike.outerloop_loglike()
        # reset nouterloop_samples because tensor product quadrature rules
        # will not be able to match the number of requested samples exactly
        nouterloop_samples = outerloop_loglike._shapes.shape[1]

        # check the gradients of the OED likelihood
        design_weights = bkd.array(np.random.uniform(0.5, 1.0, (nobs, 1)))
        errors = outerloop_loglike.check_apply_jacobian(
            design_weights,
            disp=False,
            fd_eps=bkd.flip(bkd.logspace(-13, -1, 13)),
        )
        assert errors.min() / errors.max() < 2e-6

        # check the simultaneous computation of evidences for different
        # realizations of the data
        innerloop_samples, innerloop_quad_weights = (
            self._setup_quadrature_data(
                innerloop_quadtype, prior, ninnerloop_samples
            )
        )
        # reset ninnerloop_samples because tensor product quadrature rules
        # will not be able to match the number of requested samples exactly
        ninnerloop_samples = innerloop_samples.shape[1]
        innerloop_shapes = obs_model(innerloop_samples).T
        innerloop_loglike.set_observations_and_shapes(
            outerloop_loglike._obs, innerloop_shapes
        )

        evidence = Evidence(innerloop_loglike, innerloop_quad_weights)
        evidence_vals = evidence(design_weights)
        assert evidence_vals.shape == (1, nouterloop_samples)
        errors = evidence.check_apply_jacobian(
            design_weights,
            disp=False,
            fd_eps=bkd.flip(bkd.logspace(-13, -1, 13)),
        )
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 2e-6
        log_evidence = LogEvidence(innerloop_loglike, innerloop_quad_weights)
        log_evidence_vals = log_evidence(design_weights)
        assert log_evidence_vals.shape == (1, nouterloop_samples)
        errors = log_evidence.check_apply_jacobian(
            design_weights,
            disp=False,
            fd_eps=bkd.flip(bkd.logspace(-13, -1, 13)),
        )
        print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 2e-6

        oed_objective = KLOEDObjective(
            innerloop_loglike,
            outerloop_loglike._shapes,
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_shapes,
            innerloop_quad_weights,
            backend=bkd,
        )
        design_weights_map = bkd.array([0, 1, 1])
        oed_objective.set_design_weights_map(design_weights_map)
        design_weights = bkd.array([0.5, 0.5])[:, None]
        errors = oed_objective.check_apply_jacobian(
            design_weights,
            disp=False,
            fd_eps=bkd.flip(bkd.logspace(-13, -1, 13)),
        )
        print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 2e-6
        # check hessian code works. But currently apply_hessian_implemented
        # is set to False because the code is slow. Temporarily
        # activate hessian computation
        oed_objective.apply_hessian_implemented = lambda: True
        errors = oed_objective.check_apply_hessian(
            design_weights,
            disp=True,
            fd_eps=bkd.flip(bkd.logspace(-13, -1, 13)),
        )
        assert errors.min() / errors.max() < 2e-6
        # disable hesian computation
        oed_objective.apply_hessian_implemented = lambda: False

    def _check_independent_gaussian_expected_information_gain(
        self,
        outerloop_quadtype,
        nouterloop_samples,
        innerloop_quadtype,
        ninnerloop_samples,
        degree,
        nobs,
        tol,
    ):
        bkd = self.get_backend()
        (
            prior,
            obs_model,
            innerloop_loglike,
            noise_cov_diag,
            prior_data_variable,
            outerloop_samples,
            outerloop_quad_weights,
        ) = self._setup_linear_gaussian_oed(
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
            degree,
            nobs,
        )

        outerloop_loglike = innerloop_loglike.outerloop_loglike()
        innerloop_samples, innerloop_quad_weights = (
            self._setup_quadrature_data(
                innerloop_quadtype, prior, ninnerloop_samples
            )
        )
        ninnerloop_samples = innerloop_samples.shape[1]
        innerloop_shapes = obs_model(innerloop_samples).T
        oed_objective = KLOEDObjective(
            innerloop_loglike,
            outerloop_loglike._shapes,
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_shapes,
            innerloop_quad_weights,
            backend=bkd,
        )

        design_weights = bkd.array(np.random.uniform(0.5, 1.0, (nobs, 1)))
        noise_cov = bkd.diag(noise_cov_diag[:, 0])
        self._check_expected_information_gain(
            prior,
            oed_objective,
            outerloop_samples[: prior.nvars()],
            design_weights,
            obs_model,
            noise_cov,
            tol,
        )

    def test_independent_gaussian_expected_information_gain(self):
        test_cases = [
            ["gauss", 5000, "gauss", 100, 1, 1, 3e-3],
            ["quadratic", 500000, "quadratic", 100, 0, 1, 2e-2],
            ["MC", 100000, "MC", 1000, 1, 2, 2e-2],
            ["MC", 100000, "Halton", 1000, 1, 2, 2e-2],
        ]

        for test_case in test_cases:
            self._check_independent_gaussian_expected_information_gain(
                *test_case
            )

    def _setup_bayesian_OED_data(
        self,
        nobs,
        min_degree,
        degree,
        noise_stat,
        outerloop_quadtype,
        nouterloop_samples,
        innerloop_quadtype,
        ninnerloop_samples,
    ):
        bkd = self.get_backend()
        nvars = degree - min_degree + 1
        # the smaller the noise the more number of nout_samples are needed
        noise_std = 0.125 * 4
        prior_std = 0.5
        prior = IndependentMarginalsVariable(
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

        prior_data_variable = IndependentMarginalsVariable(
            prior.marginals()
            + [
                stats.norm(0, bkd.sqrt(variance))
                for variance in noise_cov_diag[:, 0]
            ],
            backend=bkd,
        )
        print(prior_data_variable)

        # setup OED log likelihood
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            noise_cov_diag, backend=bkd
        )

        # generate observations to compute expected information gain
        outerloop_samples, outerloop_quad_weights = (
            self._setup_quadrature_data(
                outerloop_quadtype, prior_data_variable, nouterloop_samples
            )
        )
        outerloop_shapes_samples = outerloop_samples[: prior.nvars()]
        outerloop_shapes = obs_model(outerloop_shapes_samples).T
        outerloop_loglike = innerloop_loglike.outerloop_loglike()
        obs = outerloop_loglike.rvs_from_shapes(outerloop_shapes)
        outerloop_loglike.set_observations_and_shapes(obs, outerloop_shapes)
        nouterloop_samples = outerloop_loglike._shapes.shape[1]

        innerloop_samples, innerloop_quad_weights = (
            self._setup_quadrature_data(
                innerloop_quadtype, prior, ninnerloop_samples
            )
        )
        ninnerloop_samples = innerloop_samples.shape[1]
        innerloop_shapes = obs_model(innerloop_samples).T
        innerloop_loglike.set_observations_and_shapes(obs, innerloop_shapes)
        return (
            innerloop_loglike,
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_shapes,
            innerloop_quad_weights,
            obs_model,
            noise_cov_diag,
            prior_std,
            design,
            prior,
        )

    def _check_classical_KL_OED_gaussian_optimization(
        self,
        nobs,
        min_degree,
        degree,
        noise_stat,
        outerloop_quadtype,
        nouterloop_samples,
        innerloop_quadtype,
        ninnerloop_samples,
    ):
        bkd = self.get_backend()
        (
            innerloop_loglike,
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_shapes,
            innerloop_quad_weights,
            obs_model,
            noise_cov_diag,
            prior_std,
            design,
            prior,
        ) = self._setup_bayesian_OED_data(
            nobs,
            min_degree,
            degree,
            noise_stat,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )
        kl_oed = KLBayesianOED(innerloop_loglike)
        kl_oed.set_data(
            innerloop_loglike.outerloop_loglike().shapes(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_loglike.shapes(),
            innerloop_quad_weights,
        )
        x0 = bkd.full((nobs, 1), 1 / nobs)
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 13))
        errors = kl_oed.objective().check_apply_jacobian(
            x0, disp=True, fd_eps=fd_eps
        )
        assert errors.min() / errors.max() < 1e-6 and errors.max() < 10

        dopt_objective = DOptimalLinearModelObjective(
            obs_model, noise_cov_diag[0, 0], bkd.array(prior_std**2)
        )
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
        # print(dopt_objective(x0), oed.objective()(x0))
        assert bkd.allclose(
            dopt_objective(x0), kl_oed.objective()(x0), rtol=1e-2
        )

        # just test optimizations run because of noise in monte carlo
        # I am not sure if I can ensure that klopt and dopt designs
        # will be the same. Equivalence would also only occur if
        # noise_stat = SampleAverageMean
        # kl_oed = BayesianOED(kl_oed_objective)
        # klopt_design = kl_oed.compute()
        kl_oed.compute()

    def test_KL_OED_gaussian_optimization(self):
        bkd = self.get_backend()
        noise_stats = [
            NoiseStatistic(SampleAverageMean(bkd)),
            NoiseStatistic(SampleAverageMeanPlusStdev(1, bkd)),
            NoiseStatistic(SampleAverageEntropicRisk(0.5, bkd)),
        ]
        for noise_stat in noise_stats:
            test_case = [3, 0, 1, noise_stat, "Halton", 10000, "Halton", 100]
            self._check_classical_KL_OED_gaussian_optimization(*test_case)

    def _setup_OED_for_prediction(
        self,
        nqoi,
        nobs,
        min_degree,
        degree,
        noise_stat,
        risk_measure,
        outerloop_quadtype,
        nouterloop_samples,
        innerloop_quadtype,
        ninnerloop_samples,
    ):
        bkd = self.get_backend()
        (
            innerloop_loglike,
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_shapes,
            innerloop_quad_weights,
            obs_model,
            noise_cov_diag,
            prior_std,
            design,
            prior,
        ) = self._setup_bayesian_OED_data(
            nobs,
            min_degree,
            degree,
            noise_stat,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )

        qoi_mat = bkd.array(np.random.uniform(0, 1, (nqoi, prior.nvars())))

        def qoi_model(samples):
            return (qoi_mat @ samples).T

        qoi_vals = qoi_model(innerloop_samples)
        # assume MC quadrature for prediction space
        qoi_quad_weights = bkd.full((nqoi, 1), 1.0 / nqoi)

        deviation_measure = OEDStandardDeviationMeasure(nqoi, bkd)
        deviation_measure.set_loglikelihood(innerloop_loglike)
        oed = BayesianOEDForPrediction(
            innerloop_loglike,
            deviation_measure,
            risk_measure,
            noise_stat,
        )
        oed.set_data(
            innerloop_loglike.outerloop_loglike().shapes(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_loglike.shapes(),
            innerloop_quad_weights,
            qoi_vals,
            qoi_quad_weights,
        )
        return oed, prior, qoi_mat, obs_model

    def test_prediction_gaussian_OED_conjugate_gaussian_prior_std(self):
        """
        Check MC estimates used by oed objective are consistent
        with analytical formula when using Gaussian Confulgate prior and
        linear qoi model and using std as the inner deviation measure
        """
        noise_stat = NoiseStatistic(SampleAverageMean(self.get_backend()))
        test_case = [2, 0, 1, noise_stat, "MC", 10000, "MC", 10000]
        (
            nobs,
            min_degree,
            degree,
            noise_stat,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        ) = test_case
        bkd = self.get_backend()
        risk_measure = SampleAverageMean(bkd)
        oed, prior, qoi_mat, obs_model = self._setup_OED_for_prediction(
            1,
            nobs,
            min_degree,
            degree,
            noise_stat,
            risk_measure,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )
        x0 = bkd.full((nobs, 1), 1 / nobs)
        std_utility = (
            ConjugateGaussianPriorOEDForLinearPredictionStandardDeviation(
                prior, qoi_mat
            )
        )
        std_utility.set_observation_matrix(obs_model._matrix)
        noise_cov = bkd.diag(
            oed.objective()._outerloop_loglike._wnoise_std_diag[:, 0]
        )
        std_utility.set_noise_covariance(noise_cov)
        # print(oed.objective()(x0), std_utility.value())
        assert bkd.allclose(
            oed.objective()(x0), std_utility.value(), rtol=1e-2
        )

    def _check_prediction_gaussian_OED_gradients(
        self,
        nobs,
        min_degree,
        degree,
        noise_stat,
        risk_measure,
        outerloop_quadtype,
        nouterloop_samples,
        innerloop_quadtype,
        ninnerloop_samples,
    ):
        bkd = self.get_backend()
        oed, prior, qoi_mat, obs_model = self._setup_OED_for_prediction(
            2,
            nobs,
            min_degree,
            degree,
            noise_stat,
            risk_measure,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )

        # test gradient
        x0 = bkd.full((nobs, 1), 1 / nobs)
        errors = oed.objective().check_apply_jacobian(
            x0, disp=True, fd_eps=bkd.flip(bkd.logspace(-13, -1, 12))
        )
        assert errors.min() / errors.max() < 6e-6, errors.min() / errors.max()
        # test optimization runs
        oed.compute()

    def test_prediction_gaussian_OED_gradients(self):
        bkd = self.get_backend()
        noise_stats = [
            NoiseStatistic(SampleAverageMean(bkd)),
            NoiseStatistic(SampleAverageMeanPlusStdev(1, bkd)),
            NoiseStatistic(SampleAverageEntropicRisk(0.5, bkd)),
        ]
        risk_measures = [
            SampleAverageMeanPlusStdev(1, bkd),
            SampleAverageEntropicRisk(0.5, bkd),
        ]
        for noise_stat, risk_measure in itertools.product(
            noise_stats, risk_measures
        ):
            np.random.seed(1)
            test_case = [2, 0, 1, noise_stat, risk_measure, "MC", 4, "MC", 3]
            self._check_prediction_gaussian_OED_gradients(*test_case)

    def test_conjugate_gaussian_prior_OED_for_prediction_exact_formulas(
        self,
    ):
        bkd = self.get_backend()
        nobs = 3
        nvars = 2
        nqoi = 1
        prior_mean = bkd.zeros((nvars, 1))
        prior_cov = bkd.eye(nvars)
        obs_mat = bkd.array(np.random.uniform(1, 2, (nobs, nvars)))
        noise_cov = bkd.diag(bkd.full((nobs,), 0.1))
        qoi_mat = bkd.array(np.random.uniform(1, 2, (nqoi, nvars)))
        prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, backend=bkd
        )
        kl_utility = ConjugateGaussianPriorOEDForLinearPredictionKLDivergence(
            prior, qoi_mat
        )

        nsamples = int(1e4)
        samples = prior.rvs(nsamples)
        obs = obs_mat @ samples + bkd.cholesky(noise_cov) @ bkd.array(
            np.random.normal(0, 1, (nobs, nsamples))
        )
        posterior = DenseMatrixLaplacePosteriorApproximation(
            obs_mat, prior.mean(), prior.covariance(), noise_cov, backend=bkd
        )
        kl_divs = []
        stdevs = []
        lognormal_stdevs = []
        prior_push = kl_utility._prior_pushforward.pushforward_variable()
        for ii in range(nsamples):
            posterior.compute(obs[:, ii : ii + 1])
            post_push = GaussianPushForward(
                qoi_mat,
                posterior.posterior_mean(),
                posterior.posterior_covariance(),
                backend=bkd,
            )
            kl_divs.append(
                post_push.pushforward_variable().kl_divergence(prior_push)
            )
            stdevs.append(bkd.sqrt(post_push.covariance()[0, 0]))
            lognormal_risks = LogNormalAnalyticalRiskMeasures(
                post_push.mean()[0, 0], stdevs[-1]
            )
            # print(
            #     bkd.exp(post_push.pushforward_variable().rvs(100000)).mean(),
            #     lognormal_risks.mean(),
            # )
            lognormal_stdevs.append(lognormal_risks.std())

        print(lognormal_risks)

        kl_utility.set_observation_matrix(obs_mat)
        kl_utility.set_noise_covariance(noise_cov)
        assert bkd.allclose(
            kl_utility.value(), bkd.array(kl_divs).mean(), rtol=1e-2
        )

        # The standard deviation of the posterior and its pushforward
        # are independent of the data, so expected utility will just
        # be standard deviation of posterior pushforward
        std_utility = (
            ConjugateGaussianPriorOEDForLinearPredictionStandardDeviation(
                prior, qoi_mat
            )
        )
        std_utility.set_observation_matrix(obs_mat)
        std_utility.set_noise_covariance(noise_cov)
        assert bkd.allclose(std_utility.value(), bkd.array(stdevs).mean())

        std_lognormal_utility = (
            ConjugateGaussianPriorOEDForLogNormalPredictionStandardDeviation(
                prior, qoi_mat
            )
        )
        std_lognormal_utility.set_observation_matrix(obs_mat)
        std_lognormal_utility.set_noise_covariance(noise_cov)
        assert bkd.allclose(
            std_lognormal_utility.value(),
            bkd.array(lognormal_stdevs).mean(),
            rtol=1e-2,
        )


# class TestNumpyBayesOED(TestBayesOED, unittest.TestCase):
#     def get_backend(self):
#         return NumpyMixin


class TestTorchBayesOED(TestBayesOED, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
