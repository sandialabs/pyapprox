"""
Legacy comparison tests for prediction OED objective values.

TODO: Delete after legacy removed.

Tests verify that typing prediction OED objective values match the legacy
implementation using the same problem setup and data.

Note: Gradient verification is done via DerivativeChecker in standalone tests.
This file only compares objective values between typing and legacy implementations.
"""

import unittest

import numpy as np
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.util.backends.torch import TorchMixin
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401

import torch
torch.set_default_dtype(torch.float64)


class TestPredictionOEDValuesLegacyComparison(ParametrizedTestCase):
    """Compare typing prediction OED objective values with legacy.

    These tests set up both legacy and typing implementations with identical
    problem setup and data, then verify the objective values match.
    """

    @parametrize(
        "deviation_type",
        [
            ("stdev",),
            ("entropic",),
        ],
    )
    @slow_test
    def test_full_prediction_oed_objective_matches_legacy(self, deviation_type):
        """Test that typing prediction OED full objective matches legacy.

        This test sets up the full prediction OED workflow with both
        legacy and typing implementations using identical problem setup
        and data, then compares the objective values.
        """
        from pyapprox.expdesign.bayesoed import (
            NoiseStatistic,
            OEDStandardDeviationMeasure,
            OEDEntropicDeviationMeasure,
            IndependentGaussianOEDInnerLoopLogLikelihood,
            BayesianOEDForPrediction,
        )
        from pyapprox.optimization.sampleaverage import SampleAverageMean
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianOEDForPredictionBenchmark,
            BayesianKLOEDDiagnostics,
        )

        bkd = TorchMixin
        np.random.seed(42)

        nobs = 3
        min_degree = 0
        degree = 2
        nqoi = 2
        noise_std = 0.5
        prior_std = 0.5
        nouterloop_samples = 10
        ninnerloop_samples = 8

        # Create deviation measure (legacy)
        if deviation_type == "stdev":
            legacy_deviation = OEDStandardDeviationMeasure(nqoi, bkd)
        else:
            legacy_deviation = OEDEntropicDeviationMeasure(nqoi, 1.0, bkd)

        # Create risk measure (legacy) - use Mean for comparable results
        legacy_risk = SampleAverageMean(bkd)
        legacy_noise_stat = NoiseStatistic(SampleAverageMean(bkd))
        qoi_quad_weights = bkd.full((nqoi, 1), 1.0 / nqoi)

        # ============ LEGACY SETUP ============
        legacy_problem = LinearGaussianBayesianOEDForPredictionBenchmark(
            nobs, min_degree, degree, noise_std, prior_std,
            backend=bkd, nqoi=nqoi
        )

        oed_diagnostic = BayesianKLOEDDiagnostics(legacy_problem)
        innerloop_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            legacy_problem.get_noise_covariance_diag()[:, None],
            backend=bkd,
        )
        legacy_oed = BayesianOEDForPrediction(innerloop_loglike)

        (
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
        ) = oed_diagnostic.data_generator().prepare_simulation_inputs(
            legacy_oed,
            legacy_problem.get_prior(),
            "MC",
            nouterloop_samples,
            "MC",
            ninnerloop_samples,
        )

        legacy_oed.set_data_from_model(
            legacy_problem.get_observation_model(),
            legacy_problem.get_qoi_model(),
            legacy_problem.get_prior(),
            outerloop_samples,
            outerloop_quad_weights,
            innerloop_samples,
            innerloop_quad_weights,
            qoi_quad_weights,
            legacy_deviation,
            legacy_risk,
            legacy_noise_stat,
        )

        # Compute legacy objective value
        design_weights = bkd.full((nobs, 1), 1 / nobs)
        legacy_value = legacy_oed.objective()(design_weights)
        legacy_value_np = bkd.to_numpy(legacy_value)[0, 0]

        # ============ TYPING SETUP ============
        # Extract the same data used by legacy to pass to typing
        from pyapprox.typing.util.backends.torch import TorchBkd
        from pyapprox.typing.expdesign import create_prediction_oed_objective

        typing_bkd = TorchBkd()

        # Get the data from the legacy OED objective
        legacy_obj = legacy_oed.objective()

        # Extract noise variances
        noise_variances = typing_bkd.asarray(
            bkd.to_numpy(legacy_problem.get_noise_covariance_diag())
        )

        # Extract outer/inner shapes and latent samples from the legacy objective
        outer_shapes = typing_bkd.asarray(
            bkd.to_numpy(legacy_obj._outloop_shapes)
        )
        inner_shapes = typing_bkd.asarray(
            bkd.to_numpy(legacy_obj._inloop_shapes)
        )
        latent_samples = typing_bkd.asarray(
            bkd.to_numpy(legacy_obj._outloop_quad_samples[-nobs:])
        )

        # Extract QoI values from the deviation measure (not the objective)
        qoi_vals = typing_bkd.asarray(
            bkd.to_numpy(legacy_deviation._qoi_vals)
        )

        # Extract quadrature weights
        outer_quad_weights_np = bkd.to_numpy(outerloop_quad_weights).flatten()
        inner_quad_weights_np = bkd.to_numpy(innerloop_quad_weights).flatten()

        # Map deviation type
        if deviation_type == "stdev":
            typing_deviation = "stdev"
            typing_extra = {}
        else:
            typing_deviation = "entropic"
            typing_extra = {"alpha": 1.0}

        typing_objective = create_prediction_oed_objective(
            noise_variances,
            outer_shapes,
            inner_shapes,
            latent_samples,
            qoi_vals,
            typing_bkd,
            deviation_type=typing_deviation,
            risk_type="mean",
            noise_stat_type="mean",
            outer_quad_weights=typing_bkd.asarray(outer_quad_weights_np),
            inner_quad_weights=typing_bkd.asarray(inner_quad_weights_np),
            qoi_quad_weights=typing_bkd.asarray(bkd.to_numpy(qoi_quad_weights)),
            **typing_extra,
        )

        typing_weights = typing_bkd.asarray(np.full((nobs, 1), 1 / nobs))
        typing_value = typing_objective(typing_weights)
        typing_value_np = typing_bkd.to_numpy(typing_value)[0, 0]

        # ============ COMPARISON ============
        # Both values should be finite
        self.assertTrue(np.isfinite(legacy_value_np))
        self.assertTrue(np.isfinite(typing_value_np))

        # Values should match
        np.testing.assert_allclose(
            typing_value_np, legacy_value_np, rtol=1e-10,
            err_msg=f"Typing ({typing_value_np}) != Legacy ({legacy_value_np}) "
                    f"for deviation_type={deviation_type}"
        )

    @slow_test
    def test_sample_statistics_match_legacy(self):
        """Test sample statistics (Mean, Variance) match between typing and legacy."""
        np.random.seed(42)
        nsamples = 20
        nqoi = 3

        values_np = np.random.randn(nsamples, nqoi)
        weights_np = np.random.dirichlet(np.ones(nsamples))[:, None]

        # Legacy
        from pyapprox.optimization.sampleaverage import (
            SampleAverageMean as LegacySampleAverageMean,
            SampleAverageVariance as LegacySampleAverageVariance,
        )

        legacy_values = TorchMixin.asarray(values_np)
        legacy_weights = TorchMixin.asarray(weights_np)

        legacy_mean = LegacySampleAverageMean(TorchMixin)
        legacy_var = LegacySampleAverageVariance(TorchMixin)

        legacy_mean_result = TorchMixin.to_numpy(legacy_mean(legacy_values, legacy_weights))
        legacy_var_result = TorchMixin.to_numpy(legacy_var(legacy_values, legacy_weights))

        # Typing
        from pyapprox.typing.util.backends.torch import TorchBkd
        from pyapprox.typing.expdesign import SampleAverageMean, SampleAverageVariance

        typing_bkd = TorchBkd()
        typing_values = typing_bkd.asarray(values_np)
        typing_weights = typing_bkd.asarray(weights_np)

        typing_mean = SampleAverageMean(typing_bkd)
        typing_var = SampleAverageVariance(typing_bkd)

        typing_mean_result = typing_bkd.to_numpy(typing_mean(typing_values, typing_weights))
        typing_var_result = typing_bkd.to_numpy(typing_var(typing_values, typing_weights))

        # Should match exactly
        np.testing.assert_allclose(typing_mean_result, legacy_mean_result, rtol=1e-12)
        np.testing.assert_allclose(typing_var_result, legacy_var_result, rtol=1e-12)

    @slow_test
    def test_sample_statistics_jacobian_match_legacy(self):
        """Test sample statistics Jacobians match between typing and legacy."""
        np.random.seed(42)
        nsamples = 20
        nqoi = 3
        nvars = 4

        values_np = np.random.randn(nsamples, nqoi)
        weights_np = np.random.dirichlet(np.ones(nsamples))[:, None]
        jac_values_np = np.random.randn(nsamples, nqoi, nvars)

        # Legacy
        from pyapprox.optimization.sampleaverage import (
            SampleAverageMean as LegacySampleAverageMean,
            SampleAverageVariance as LegacySampleAverageVariance,
        )

        legacy_values = TorchMixin.asarray(values_np)
        legacy_weights = TorchMixin.asarray(weights_np)
        legacy_jac_values = TorchMixin.asarray(jac_values_np)

        legacy_mean = LegacySampleAverageMean(TorchMixin)
        legacy_var = LegacySampleAverageVariance(TorchMixin)

        legacy_mean_jac = TorchMixin.to_numpy(
            legacy_mean.jacobian(legacy_values, legacy_jac_values, legacy_weights)
        )
        legacy_var_jac = TorchMixin.to_numpy(
            legacy_var.jacobian(legacy_values, legacy_jac_values, legacy_weights)
        )

        # Typing
        from pyapprox.typing.util.backends.torch import TorchBkd
        from pyapprox.typing.expdesign import SampleAverageMean, SampleAverageVariance

        typing_bkd = TorchBkd()
        typing_values = typing_bkd.asarray(values_np)
        typing_weights = typing_bkd.asarray(weights_np)
        typing_jac_values = typing_bkd.asarray(jac_values_np)

        typing_mean = SampleAverageMean(typing_bkd)
        typing_var = SampleAverageVariance(typing_bkd)

        typing_mean_jac = typing_bkd.to_numpy(
            typing_mean.jacobian(typing_values, typing_jac_values, typing_weights)
        )
        typing_var_jac = typing_bkd.to_numpy(
            typing_var.jacobian(typing_values, typing_jac_values, typing_weights)
        )

        # Should match exactly
        np.testing.assert_allclose(typing_mean_jac, legacy_mean_jac, rtol=1e-12)
        np.testing.assert_allclose(typing_var_jac, legacy_var_jac, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
