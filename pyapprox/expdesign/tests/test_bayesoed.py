import unittest
import itertools
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.expdesign.optbayes_benchmarks import (
    LinearGaussianBayesianOEDBenchmark,
    LinearGaussianBayesianOEDForPredictionBenchmark,
    BayesianKLOEDDiagnostics,
    BayesianOEDForPredictionDiagnostics,
    ConjugateGaussianPriorOEDForLinearPredictionKLDivergence,
    ConjugateGaussianPriorOEDForLinearPredictionStandardDeviation,
    ConjugateGaussianPriorOEDForLinearPredictionEntropicDeviation,
    ConjugateGaussianPriorOEDForLinearPredictionAVaRDeviation,
    ConjugateGaussianPriorOEDForLogNormalPredictionStandardDeviation,
)
from pyapprox.expdesign.optbayes import (
    Evidence,
    LogEvidence,
    KLBayesianOED,
    DOptimalLinearModelObjective,
    IndependentGaussianOEDInnerLoopLogLikelihood,
    BayesianOEDForPrediction,
    NoiseStatistic,
    OEDStandardDeviationMeasure,
    OEDEntropicDeviationMeasure,
    OEDAVaRDeviationMeasure,
)
from pyapprox.bayes.likelihood import (
    ModelBasedIndependentGaussianLogLikelihood,
)
from pyapprox.interface.model import ModelFromSingleSampleCallable
from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.bayes.laplace import (
    DenseMatrixLaplacePosteriorApproximation,
    GaussianPushForward,
)
from pyapprox.optimization.risk import (
    LogNormalAnalyticalRiskMeasures,
    GaussianAnalyticalRiskMeasures,
)
from pyapprox.optimization.minimize import (
    SampleAverageMean,
    SampleAverageMeanPlusStdev,
    SampleAverageEntropicRisk,
)


class TestBayesOED:

    def setUp(self):
        np.random.seed(1)

    def _check_KL_OED_objective_values(
        self,
        outerloop_quadtype,
        innerloop_quadtype,
        min_convergence_rate,
        max_error,
    ):
        bkd = self.get_backend()
        nobs = 2
        min_degree = 0
        degree = 3
        noise_std = 0.5
        prior_std = 0.5

        # Initialize problem
        problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=bkd
        )

        # Initialize diagnostic
        oed_diagnostic = BayesianKLOEDDiagnostics(problem)

        # Compute MSE
        if outerloop_quadtype != "MC" or innerloop_quadtype != "MC":
            nrealizations = 1
        else:
            nrealizations = 100
        design_weights = bkd.ones((nobs, 1)) / nobs

        # Test expected convergence rate
        outerloop_sample_counts = [500, 1000, 2000, 5000, 10000]
        innerloop_sample_counts = [250]

        fig, axes = plt.subplots(
            1, 3, figsize=(3 * 8, 6), sharex=True, sharey=True
        )
        values = oed_diagnostic.plot_mse_for_sample_combinations(
            axes,
            outerloop_sample_counts,
            innerloop_sample_counts,
            nrealizations,
            design_weights,
            outerloop_quadtype,
            innerloop_quadtype,
        )[0]
        convergence_rate = oed_diagnostic.compute_convergence_rate(
            outerloop_sample_counts, values["mse"][0]
        )
        # gauss quadrature rules obtain such small error with
        # the number of samples requested that it will mess up estimation
        # of convergence_rate
        if min_convergence_rate is not None:
            assert (
                convergence_rate >= min_convergence_rate
            ), f"{convergence_rate=} must be >= {min_convergence_rate}"
        assert (
            values["mse"][0][-1] <= max_error
        ), f"best mse {values['mse'][0][-1]} must be >= {max_error}"

        # Test adding convergence rate to plot runs
        oed_diagnostic.add_convergence_rate_to_mse_plot(
            axes[2], outerloop_sample_counts, values["mse"][0]
        )

    def test_KL_OED_objective_values(self):
        test_cases = [
            ["MC", "MC", 0.94, 1e-4],
            ["Halton", "Halton", 2.0, 4e-7],
            ["gauss", "gauss", None, 2e-6],
        ]

        for test_case in test_cases:
            self._check_KL_OED_objective_values(*test_case)

    def _check_OED_likelihood_evidence_gradients(
        self,
        outerloop_quadtype,
        nouterloop_samples,
        innerloop_quadtype,
        ninnerloop_samples,
    ):
        """
        Test the gradients of the inner and outer likelihoods and log evidence

        The reference gradients DO NOT include the impact of the
        reparameterization trick to compute the observational data.
        Instead they assume the data is not a function of the weights
        """
        bkd = self.get_backend()
        nobs = 3
        min_degree = 0
        degree = 3
        noise_std = 0.5
        prior_std = 0.5

        # Initialize problem
        problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=bkd
        )

        # Initialize diagnostic
        oed_diagnostic = BayesianKLOEDDiagnostics(problem)
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            problem.get_noise_covariance_diag()[:, None],
            backend=bkd,
        )
        kl_oed = KLBayesianOED(innerloop_loglike)
        (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        ) = oed_diagnostic.prepare_simulation_inputs(
            kl_oed,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )
        kl_oed.set_data_from_model(
            problem.get_observation_model(),
            problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        )

        # Helper function to check gradients
        def check_likelihood_gradients(objective, design_weights):
            errors = objective.check_apply_jacobian(
                design_weights,
                disp=False,
                fd_eps=bkd.flip(bkd.logspace(-13, -1, 13)),
            )
            print(errors.min() / errors.max())
            assert errors.min() / errors.max() < 2e-6

        # Helper function to check evidences
        def check_evidence_gradients(
            evidence_object, design_weights, expected_shape
        ):
            evidence_vals = evidence_object(design_weights)
            print(evidence_vals.shape, expected_shape)
            assert evidence_vals.shape == expected_shape
            errors = evidence_object.check_apply_jacobian(
                design_weights,
                disp=False,
                fd_eps=bkd.flip(bkd.logspace(-13, -1, 13)),
            )
            # print(errors.min() / errors.max())
            assert errors.min() / errors.max() < 2e-6

        # Check gradients of the outer loop log-likelihood
        design_weights = bkd.array(np.random.uniform(0.5, 1.0, (nobs, 1)))
        outerloop_loglike = kl_oed.get_outerloop_loglike()
        outerloop_loglike.set_design_weights(design_weights)
        latent_likelihood_samples = kl_oed.objective()._outerloop_quad_samples[
            -kl_oed.objective()._outerloop_shapes.shape[0] :
        ]
        obs = outerloop_loglike._rvs_from_likelihood_samples(
            kl_oed.objective()._outerloop_shapes,
            latent_likelihood_samples,
        )
        outerloop_loglike.set_artificial_observations(obs)
        check_likelihood_gradients(outerloop_loglike, design_weights)

        # Check evidences for inner loop
        # udpate nouterloop_samples because some quad rules cannot guarantee
        # using exactly the number of samples requestd

        nouterloop_samples = kl_oed.objective()._outerloop_shapes.shape[1]
        innerloop_loglike.set_artificial_observations(obs)
        evidence = Evidence(innerloop_loglike, innerloop_quad_weights)
        check_evidence_gradients(
            evidence, design_weights, (1, nouterloop_samples)
        )

        # Check log-evidences for inner loop
        log_evidence = LogEvidence(innerloop_loglike, innerloop_quad_weights)
        check_evidence_gradients(
            log_evidence, design_weights, (1, nouterloop_samples)
        )

    def test_oed_likelihood_evidence_gradients(self):
        test_cases = [
            ["gauss", 5000, "gauss", 1000],
            ["MC", 5000, "MC", 1000],
            ["MC", 5000, "Halton", 1000],
        ]

        for test_case in test_cases:
            self._check_OED_likelihood_evidence_gradients(*test_case)

    def _check_KL_OED_gradients(
        self,
        outerloop_quadtype,
        nouterloop_samples,
        innerloop_quadtype,
        ninnerloop_samples,
    ):
        """
        Test the gradients needed to compute an KL-based OED.

        The reference gradients DO include the impact of the
        reparameterization trick to compute the observational data.
        They assume the data is a function of the weights
        """
        bkd = self.get_backend()
        nobs = 3
        min_degree = 0
        degree = 3
        noise_std = 0.5
        prior_std = 0.5

        # Initialize problem
        problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=bkd
        )

        # Initialize diagnostic
        oed_diagnostic = BayesianKLOEDDiagnostics(problem)
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            problem.get_noise_covariance_diag()[:, None],
            backend=bkd,
        )
        kl_oed = KLBayesianOED(innerloop_loglike)
        (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        ) = oed_diagnostic.prepare_simulation_inputs(
            kl_oed,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )
        kl_oed.set_data_from_model(
            problem.get_observation_model(),
            problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        )

        # Create function that has knowledge about the use of their
        # reparameterization trick. The OED objective takes care of
        # this, but need to replicate its functionality here
        # when using the liklihood outside the OED objective
        def likelihood_with_reparameterization_trick(return_values, weights):
            outerloop_loglike.set_design_weights(weights)
            latent_likelihood_samples = (
                kl_oed.objective()._outerloop_quad_samples[
                    -kl_oed.objective()._outerloop_shapes.shape[0] :
                ]
            )
            obs = outerloop_loglike._rvs_from_likelihood_samples(
                kl_oed.objective()._outerloop_shapes,
                latent_likelihood_samples,
            )

            outerloop_loglike.set_artificial_observations(obs)
            outerloop_loglike.set_latent_likelihood_samples(
                latent_likelihood_samples
            )
            if return_values:
                return outerloop_loglike(weights)
            else:
                return outerloop_loglike.jacobian(weights)

        design_weights = bkd.array(np.random.uniform(0.5, 1.0, (nobs, 1)))
        design_weights /= design_weights.sum()
        outerloop_loglike = kl_oed.get_outerloop_loglike()

        # Helper function to check gradients
        def check_likelihood_gradients(objective, design_weights):
            errors = objective.check_apply_jacobian(
                design_weights,
                disp=False,
                fd_eps=bkd.flip(bkd.logspace(-13, -1, 13)),
            )
            assert errors.min() / errors.max() < 2e-6

        design_weights = bkd.array([0.5, 0.5, 0.5])[:, None]

        # Check gradients of the outerloop likelihood
        likelihood_wrapper = ModelFromSingleSampleCallable(
            outerloop_loglike._shapes.shape[1],
            design_weights.shape[0],
            partial(likelihood_with_reparameterization_trick, True),
            partial(likelihood_with_reparameterization_trick, False),
            backend=bkd,
        )
        check_likelihood_gradients(likelihood_wrapper, design_weights)

        # Check gradients of the OED objective
        check_likelihood_gradients(kl_oed.objective(), design_weights)

        # Check gradients of the OED objective using design_weights_map
        design_weights_map = bkd.array([0, 1, 1])
        kl_oed.objective().set_design_weights_map(design_weights_map)
        design_weights = bkd.array([0.5, 0.5])[:, None]
        check_likelihood_gradients(kl_oed.objective(), design_weights)

        # check oed optimization runs
        kl_oed.compute()

    def test_KL_OED_gradients(self):
        test_cases = [
            ["gauss", 5000, "gauss", 1000],
            ["MC", 5000, "MC", 1000],
            ["MC", 5000, "Halton", 1000],
        ]

        for test_case in test_cases:
            self._check_KL_OED_gradients(*test_case)

    def test_OED_gaussian_likelihood_values(self):
        (
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        ) = ["gauss", 5000, "gauss", 1000]
        bkd = self.get_backend()
        nobs = 3
        min_degree = 0
        degree = 3
        noise_std = 0.5
        prior_std = 0.5

        # Initialize problem
        problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=bkd
        )

        # Initialize diagnostic
        oed_diagnostic = BayesianKLOEDDiagnostics(problem)
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            problem.get_noise_covariance_diag()[:, None],
            backend=bkd,
        )
        kl_oed = KLBayesianOED(innerloop_loglike)
        (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        ) = oed_diagnostic.prepare_simulation_inputs(
            kl_oed,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )
        kl_oed.set_data_from_model(
            problem.get_observation_model(),
            problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        )
        outerloop_loglike = innerloop_loglike.outerloop_loglike()
        # compute the likelihood values with traditional loglikelihood
        # which is a function of the shapes
        design_weights = bkd.array(np.random.uniform(0.5, 1.0, (nobs, 1)))
        design_weights /= design_weights.sum()
        # setting the design weights for the kl_oed objective will
        # populate likleihoods with correct data
        kl_oed.objective()._set_expanded_design_weights(design_weights)
        loglike = ModelBasedIndependentGaussianLogLikelihood(
            problem.get_observation_model(),
            problem.get_noise_covariance_diag()[:, None],
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

        # reset ninnerloop_samples because tensor product quadrature rules
        # will not be able to match the number of requested samples exactly

        # check likelihood vals are computed correctly for each observation
        innerloop_loglike_vals = innerloop_loglike(design_weights)
        loglike = ModelBasedIndependentGaussianLogLikelihood(
            problem.get_observation_model(),
            problem.get_noise_covariance_diag()[:, None],
        )
        loglike.set_design_weights(design_weights)

        vals = []
        for ii in range(obs.shape[1]):
            loglike.set_observations(obs[:, ii : ii + 1])
            vals.append(
                loglike._loglike_from_shapes(innerloop_loglike._shapes)[:, 0]
            )
        assert bkd.allclose(innerloop_loglike_vals, bkd.hstack(vals))

    def test_doptimal_oed_gradients(self):
        """
        Test that KL-based OED is equivalent to D-optimal design
        for linear Gaussian Models. Also check gradients of D-optimal design
        obejctive
        """
        bkd = self.get_backend()
        nobs = 2
        min_degree = 0
        degree = 2
        noise_std = 0.5
        prior_std = 0.5

        outerloop_quadtype = "gauss"
        nouterloop_samples = 100000
        innerloop_quadtype = "gauss"
        ninnerloop_samples = 1000

        # Initialize problem
        problem = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=bkd
        )

        # Initialize diagnostic
        oed_diagnostic = BayesianKLOEDDiagnostics(problem)
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            problem.get_noise_covariance_diag()[:, None],
            backend=bkd,
        )
        kl_oed = KLBayesianOED(innerloop_loglike)
        (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        ) = oed_diagnostic.prepare_simulation_inputs(
            kl_oed,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )
        kl_oed.set_data_from_model(
            problem.get_observation_model(),
            problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        )

        dopt_objective = DOptimalLinearModelObjective(
            problem.get_observation_model(),
            problem.get_noise_covariance_diag()[0],
            bkd.array(prior_std**2),
        )
        design_weights = bkd.ones((nobs, 1)) / nobs
        II = bkd.hstack(
            [
                bkd.array([0, nobs - 1], dtype=int),
                bkd.where(
                    bkd.isclose(
                        bkd.abs(problem.get_design_matrix()[0]),
                        1 / bkd.sqrt(bkd.array(5)),
                    )
                )[0],
            ]
        )
        x0 = bkd.zeros((nobs, 1))
        x0[II] = 1.0
        assert bkd.allclose(
            dopt_objective(x0), kl_oed.objective()(x0), rtol=1e-5
        )

        # Check gradients
        errors = dopt_objective.check_apply_jacobian(
            design_weights, disp=False
        )
        assert errors.min() / errors.max() < 1e-6 and errors.max() < 10
        errors = dopt_objective.check_apply_hessian(design_weights, disp=False)
        assert errors.min() / errors.max() < 1e-6 and errors.max() < 10

    def test_conjugate_gaussian_prior_OED_for_prediction_exact_formulas(self):
        bkd = self.get_backend()
        nobs = 2
        min_degree = 0
        degree = 3
        noise_std = 0.5
        prior_std = 0.5

        # Initialize problem
        problem = LinearGaussianBayesianOEDForPredictionBenchmark(
            nobs, min_degree, degree, noise_std, prior_std, backend=bkd, nqoi=1
        )
        prior = DenseCholeskyMultivariateGaussian(
            problem.get_prior().mean(),
            problem.get_prior().covariance(),
            backend=bkd,
        )
        prior_push = GaussianPushForward(
            problem.get_qoi_model().matrix(),
            prior.mean(),
            prior.covariance(),
            backend=bkd,
        )

        # Generate samples used to computed expectation over datas
        nsamples = int(1e4)
        obs_mat = problem.get_observation_model().matrix()
        noise_cov = bkd.diag(problem.get_noise_covariance_diag())
        qoi_mat = problem.get_qoi_model().matrix()
        samples = prior.rvs(nsamples)
        obs = obs_mat @ samples + bkd.cholesky(noise_cov) @ bkd.array(
            np.random.normal(0, 1, (nobs, nsamples))
        )

        # Compute the posterior pushforward for each observation realization
        # and compute exact metrics
        posterior = DenseMatrixLaplacePosteriorApproximation(
            obs_mat, prior.mean(), prior.covariance(), noise_cov, backend=bkd
        )
        kl_divs = []
        stdevs = []
        entropicdevs = []
        lamda = 2.0  # strength of entropic deviation
        lognormal_stdevs = []
        beta = 0.5  # AVaR quantile
        avardevs = []
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
            normal_risks = GaussianAnalyticalRiskMeasures(
                post_push.mean()[0, 0], stdevs[-1]
            )
            lognormal_stdevs.append(lognormal_risks.std())
            entropicdevs.append(
                normal_risks.entropic(lamda) - post_push.mean()[0, 0]
            )
            avardevs.append(normal_risks.AVaR(beta) - post_push.mean()[0, 0])

        # The all deviation measures, the posterior and its pushforward
        # through a linear model,
        # are independent of the data, so expected utility will just
        # be standard deviation of posterior pushforward
        std_utility = (
            ConjugateGaussianPriorOEDForLinearPredictionStandardDeviation(
                prior, qoi_mat
            )
        )
        std_utility.set_observation_matrix(obs_mat)
        std_utility.set_noise_covariance(noise_cov)
        assert bkd.allclose(bkd.array(stdevs).mean(), std_utility.value())

        entropicdev_utility = (
            ConjugateGaussianPriorOEDForLinearPredictionEntropicDeviation(
                prior, qoi_mat, lamda
            )
        )
        entropicdev_utility.set_observation_matrix(obs_mat)
        entropicdev_utility.set_noise_covariance(noise_cov)
        assert bkd.allclose(
            bkd.array(entropicdevs).mean(),
            entropicdev_utility.value(),
            rtol=1e-2,
        )

        avardev_utility = (
            ConjugateGaussianPriorOEDForLinearPredictionAVaRDeviation(
                prior, qoi_mat, beta
            )
        )
        avardev_utility.set_observation_matrix(obs_mat)
        avardev_utility.set_noise_covariance(noise_cov)
        assert bkd.allclose(
            bkd.array(avardevs).mean(),
            avardev_utility.value(),
            rtol=1e-2,
        )

        # The following combinations of deviations and linear or nonlinear
        # model, the utiltities depend on the realizations of the data

        # Compute exected KL divergence between the prior and posterior
        # pushforwards
        kl_utility = ConjugateGaussianPriorOEDForLinearPredictionKLDivergence(
            prior, qoi_mat
        )
        kl_utility.set_observation_matrix(obs_mat)
        kl_utility.set_noise_covariance(noise_cov)
        assert bkd.allclose(
            bkd.array(kl_divs).mean(), kl_utility.value(), rtol=1e-2
        )

        std_lognormal_utility = (
            ConjugateGaussianPriorOEDForLogNormalPredictionStandardDeviation(
                prior, qoi_mat
            )
        )
        std_lognormal_utility.set_observation_matrix(obs_mat)
        std_lognormal_utility.set_noise_covariance(noise_cov)
        assert bkd.allclose(
            bkd.array(lognormal_stdevs).mean(),
            std_lognormal_utility.value(),
            rtol=1e-2,
        )

    def _check_prediction_gaussian_OED_gradients(
        self,
        nobs,
        min_degree,
        degree,
        noise_stat,
        risk_measure,
        deviation_measure,
        outerloop_quadtype,
        nouterloop_samples,
        innerloop_quadtype,
        ninnerloop_samples,
    ):
        bkd = self.get_backend()
        nqoi = deviation_measure.npred()
        noise_std = noise_std = 0.125 * 4
        prior_std = 0.5
        qoi_quad_weights = bkd.full((nqoi, 1), 1.0 / nqoi)
        problem = LinearGaussianBayesianOEDForPredictionBenchmark(
            nobs,
            min_degree,
            degree,
            noise_std,
            prior_std,
            backend=bkd,
            nqoi=nqoi,
        )
        oed_diagnostic = BayesianKLOEDDiagnostics(problem)
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            problem.get_noise_covariance_diag()[:, None],
            backend=bkd,
        )
        oed = BayesianOEDForPrediction(innerloop_loglike)
        (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        ) = oed_diagnostic.prepare_simulation_inputs(
            oed,
            outerloop_quadtype,
            nouterloop_samples,
            innerloop_quadtype,
            ninnerloop_samples,
        )
        oed.set_data_from_model(
            problem.get_observation_model(),
            problem.get_qoi_model(),
            problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
            qoi_quad_weights,
            deviation_measure,
            risk_measure,
            noise_stat,
        )

        # test gradient
        design_weights = bkd.full((nobs, 1), 1 / nobs)
        errors = oed.objective().check_apply_jacobian(
            design_weights,
            disp=True,
            fd_eps=bkd.flip(bkd.logspace(-13, -1, 12)),
        )
        assert errors.min() / errors.max() < 6e-6, errors.min() / errors.max()

    def test_prediction_OED_gradients(self):
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
        nqoi = 2
        deviation_measures = [
            OEDStandardDeviationMeasure(nqoi, bkd),
            OEDEntropicDeviationMeasure(nqoi, 1.0, bkd),
            OEDAVaRDeviationMeasure(nqoi, 0.5, 100, bkd),
        ]
        for noise_stat, risk_measure, deviation_measure in itertools.product(
            noise_stats, risk_measures, deviation_measures
        ):
            print("###")
            np.random.seed(1)
            test_case = [
                2,
                0,
                1,
                noise_stat,
                risk_measure,
                deviation_measure,
                "MC",
                4,
                "MC",
                3,
            ]
            self._check_prediction_gaussian_OED_gradients(*test_case)

    def _check_prediction_gaussian_OED_values(
        self,
        noise_stat,
        risk_measure,
        deviation_measure,
        quadtype,
        min_convergence_rate,
        max_error,
        utility_cls,
    ):
        import torch

        torch.set_printoptions(precision=8)
        bkd = self.get_backend()
        nobs = 2
        min_degree = 0
        degree = 3
        nqoi = deviation_measure.npred()
        noise_std = 0.125 * 4
        prior_std = 0.5
        problem = LinearGaussianBayesianOEDForPredictionBenchmark(
            nobs,
            min_degree,
            degree,
            noise_std,
            prior_std,
            nqoi=nqoi,
            backend=bkd,
        )
        oed_diagnostic = BayesianOEDForPredictionDiagnostics(
            problem,
            utility_cls,
            deviation_measure,
            risk_measure,
            noise_stat,
        )

        # Test expected convergence_rate
        if quadtype != "MC":
            nrealizations = 1
        else:
            nrealizations = 1000
        design_weights = bkd.ones((nobs, 1)) / nobs
        outerloop_sample_counts = [2]
        innerloop_sample_counts = [500, 1000, 2000, 5000]

        fig, axes = plt.subplots(
            1, 3, figsize=(3 * 8, 6), sharex=True, sharey=True
        )
        values = oed_diagnostic.plot_mse_for_sample_combinations(
            axes,
            outerloop_sample_counts,
            innerloop_sample_counts,
            nrealizations,
            design_weights,
            quadtype,
            quadtype,
        )[0]
        # When models are linear and Gaussian nouterloop_samples does not
        # impact error as expected_deviations are independent of the
        # observation data so check convergence_rate with respect to
        # innerloop samples
        convergence_rate = oed_diagnostic.compute_convergence_rate(
            innerloop_sample_counts, bkd.vstack(values["mse"])[:, 0]
        )
        print(convergence_rate)
        # gauss quadrature rules obtain such small error with
        # the number of samples requested that it will mess up estimation
        # of convergence_rate
        if min_convergence_rate is not None:
            assert (
                convergence_rate >= min_convergence_rate
            ), f"{convergence_rate=} must be >= {min_convergence_rate}"
        assert (
            values["mse"][-1][0] <= max_error
        ), f"best mse {values['mse'][-1][0]} must be >= {max_error}"

    def test_prediction_OED_values_linear_problem(self):
        # Note: The number of outerloop samples does not effect
        # these tests because criteria are not dependent on the observations
        # so only need to test with NoiseStatistic(SampleAverageMean(bkd))
        bkd = self.get_backend()
        nqoi = 1
        test_cases = [
            # [
            #     NoiseStatistic(SampleAverageMean(bkd)),
            #     SampleAverageMean(bkd),
            #     OEDStandardDeviationMeasure(nqoi, bkd),
            #     "MC",
            #     0.95,
            #     1e-2,
            #     ConjugateGaussianPriorOEDForLinearPredictionStandardDeviation,
            # ],
            # [
            #     NoiseStatistic(SampleAverageMean(bkd)),
            #     SampleAverageMean(bkd),
            #     OEDEntropicDeviationMeasure(nqoi, 1.0, bkd),
            #     "MC",
            #     0.95,
            #     1e-5,
            #     ConjugateGaussianPriorOEDForLinearPredictionEntropicDeviation,
            # ],
            [
                NoiseStatistic(SampleAverageMean(bkd)),
                SampleAverageMean(bkd),
                OEDAVaRDeviationMeasure(nqoi, 0.5, 1000000, bkd),
                "MC",
                0.97,
                3e-5,
                ConjugateGaussianPriorOEDForLinearPredictionAVaRDeviation,
            ],
            # TOdo write gradient code and tests for AVAR deviation
            # todo: write tests for nonlinear benchmark. that tests different noise statistics because for nonlinar model criteria depend on observations
            # [
            #     NoiseStatistic(SampleAverageMean(bkd)),
            #     SampleAverageMeanPlusStdev(1, bkd),
            #     OEDStandardDeviationMeasure(nqoi, bkd),
            #     "MC",
            #     1.0,
            #     1e-2,
            #     ConjugateGaussianPriorOEDForLogNormalPredictionStandardDeviation,
            # ],
        ]

        for test_case in test_cases:
            np.random.seed(1)
            self._check_prediction_gaussian_OED_values(*test_case)


class TestTorchBayesOED(TestBayesOED, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


class TestNumpyBayesOED(TestBayesOED, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


if __name__ == "__main__":
    unittest.main()
