"""
Legacy comparison tests for prediction OED convergence.

TODO: Delete after legacy removed.

Tests verify that typing prediction OED utilities produce results consistent
with the legacy implementation using same problem setup and parameters.
"""

import unittest

import numpy as np
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401


class TestPredictionOEDConvergenceLegacyComparison(ParametrizedTestCase):
    """Compare typing prediction OED convergence with legacy."""

    def setUp(self):
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        self._bkd = NumpyBkd()

    @parametrize(
        "utility_cls_name",
        [
            ("ConjugateGaussianOEDForNormalExpectedStdDev",),
            ("ConjugateGaussianOEDForNormalExpectedEntropicDev",),
            ("ConjugateGaussianOEDForNormalExpectedAVaRDev",),
        ],
    )
    def test_conjugate_utility_matches_legacy(self, utility_cls_name):
        """Test conjugate utility values match legacy for linear problem."""
        np.random.seed(42)
        nobs = 3
        nvars = 4
        nqoi = 1

        # Setup shared data
        prior_mean = np.random.randn(nvars, 1)
        prior_cov = np.eye(nvars) * 0.5
        obs_mat = np.random.randn(nobs, nvars)
        noise_cov = np.diag(np.random.uniform(0.1, 0.5, nobs))
        qoi_mat = np.random.randn(nqoi, nvars)

        # Legacy
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        import pyapprox.expdesign.bayesoed_benchmarks as legacy_benchmarks

        legacy_prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, backend=NumpyMixin
        )

        # Get legacy utility class
        legacy_cls = getattr(legacy_benchmarks, utility_cls_name)

        # Handle different constructors
        if "Entropic" in utility_cls_name:
            legacy_utility = legacy_cls(legacy_prior, qoi_mat, 2.0)  # lamda=2.0
        elif "AVaR" in utility_cls_name:
            legacy_utility = legacy_cls(legacy_prior, qoi_mat, 0.75)  # beta=0.75
        else:
            legacy_utility = legacy_cls(legacy_prior, qoi_mat)

        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_result = legacy_utility.value()

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        import pyapprox.typing.expdesign.analytical as typing_analytical

        bkd = NumpyBkd()

        # Map legacy names to typing names
        name_map = {
            "ConjugateGaussianOEDForNormalExpectedStdDev": "ConjugateGaussianOEDExpectedStdDev",
            "ConjugateGaussianOEDForNormalExpectedEntropicDev": "ConjugateGaussianOEDExpectedEntropicDev",
            "ConjugateGaussianOEDForNormalExpectedAVaRDev": "ConjugateGaussianOEDExpectedAVaRDev",
        }
        typing_cls = getattr(typing_analytical, name_map[utility_cls_name])

        if "Entropic" in utility_cls_name:
            typing_utility = typing_cls(
                bkd.asarray(prior_mean),
                bkd.asarray(prior_cov),
                bkd.asarray(qoi_mat),
                2.0,  # lamda=2.0
                bkd,
            )
        elif "AVaR" in utility_cls_name:
            typing_utility = typing_cls(
                bkd.asarray(prior_mean),
                bkd.asarray(prior_cov),
                bkd.asarray(qoi_mat),
                0.75,  # beta=0.75
                bkd,
            )
        else:
            typing_utility = typing_cls(
                bkd.asarray(prior_mean),
                bkd.asarray(prior_cov),
                bkd.asarray(qoi_mat),
                bkd,
            )

        typing_utility.set_observation_matrix(bkd.asarray(obs_mat))
        typing_utility.set_noise_covariance(bkd.asarray(noise_cov))
        typing_result = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )

    @parametrize(
        "utility_cls_name",
        [
            ("ConjugateGaussianOEDForLogNormalExpectedStdDev",),
            ("ConjugateGaussianOEDForLogNormalAVaRStdDev",),
        ],
    )
    def test_lognormal_utility_matches_legacy(self, utility_cls_name):
        """Test lognormal utility values match legacy."""
        np.random.seed(42)
        nobs = 3
        nvars = 4
        nqoi = 1

        # Setup shared data
        prior_mean = np.random.randn(nvars, 1)
        prior_cov = np.eye(nvars) * 0.5
        obs_mat = np.random.randn(nobs, nvars)
        noise_cov = np.diag(np.random.uniform(0.1, 0.5, nobs))
        qoi_mat = np.random.randn(nqoi, nvars)

        # Legacy
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        import pyapprox.expdesign.bayesoed_benchmarks as legacy_benchmarks

        legacy_prior = DenseCholeskyMultivariateGaussian(
            prior_mean, prior_cov, backend=NumpyMixin
        )

        legacy_cls = getattr(legacy_benchmarks, utility_cls_name)

        if "AVaR" in utility_cls_name:
            legacy_utility = legacy_cls(legacy_prior, qoi_mat, 0.5)  # beta=0.5
        else:
            legacy_utility = legacy_cls(legacy_prior, qoi_mat)

        legacy_utility.set_observation_matrix(obs_mat)
        legacy_utility.set_noise_covariance(noise_cov)
        legacy_result = legacy_utility.value()

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        import pyapprox.typing.expdesign.analytical as typing_analytical

        bkd = NumpyBkd()
        typing_cls = getattr(typing_analytical, utility_cls_name)

        if "AVaR" in utility_cls_name:
            typing_utility = typing_cls(
                bkd.asarray(prior_mean),
                bkd.asarray(prior_cov),
                bkd.asarray(qoi_mat),
                0.5,  # beta=0.5
                bkd,
            )
        else:
            typing_utility = typing_cls(
                bkd.asarray(prior_mean),
                bkd.asarray(prior_cov),
                bkd.asarray(qoi_mat),
                bkd,
            )

        typing_utility.set_observation_matrix(bkd.asarray(obs_mat))
        typing_utility.set_noise_covariance(bkd.asarray(noise_cov))
        typing_result = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )

    @slow_test
    def test_linear_benchmark_setup_matches_legacy(self):
        """Test that linear benchmark produces same design matrix as legacy."""
        nobs = 5
        min_degree = 0
        degree = 3
        noise_std = 0.5
        prior_std = 0.5

        # Legacy
        from pyapprox.expdesign.bayesoed_benchmarks import (
            LinearGaussianBayesianOEDForPredictionBenchmark,
        )

        legacy_problem = LinearGaussianBayesianOEDForPredictionBenchmark(
            nobs, min_degree, degree, noise_std, prior_std,
            backend=NumpyMixin, nqoi=1
        )
        legacy_obs_mat = NumpyMixin.to_numpy(
            legacy_problem.get_observation_model().matrix()
        )

        # Typing
        from pyapprox.typing.util.backends.numpy import NumpyBkd
        from pyapprox.typing.expdesign.benchmarks import LinearGaussianOEDBenchmark

        bkd = NumpyBkd()
        typing_benchmark = LinearGaussianOEDBenchmark(
            nobs, degree, noise_std, prior_std, bkd, min_degree=min_degree
        )
        typing_obs_mat = bkd.to_numpy(typing_benchmark.design_matrix())

        # Design matrices should match
        self._bkd.assert_allclose(
            self._bkd.asarray(typing_obs_mat),
            self._bkd.asarray(legacy_obs_mat),
            rtol=1e-12,
        )

    @slow_test
    def test_sample_statistics_mean_matches_legacy(self):
        """Test SampleAverageMean matches legacy."""
        np.random.seed(42)
        nsamples = 20
        nqoi = 3

        values_np = np.random.randn(nsamples, nqoi)
        weights_np = np.random.dirichlet(np.ones(nsamples))[:, None]

        # Legacy
        from pyapprox.optimization.sampleaverage import (
            SampleAverageMean as LegacySampleAverageMean,
        )
        from pyapprox.util.backends.torch import TorchMixin

        import torch
        torch.set_default_dtype(torch.float64)

        legacy_values = TorchMixin.asarray(values_np)
        legacy_weights = TorchMixin.asarray(weights_np)
        legacy_mean = LegacySampleAverageMean(TorchMixin)
        legacy_result = TorchMixin.to_numpy(legacy_mean(legacy_values, legacy_weights))

        # Typing
        from pyapprox.typing.util.backends.torch import TorchBkd
        from pyapprox.typing.expdesign import SampleAverageMean

        bkd = TorchBkd()
        typing_values = bkd.asarray(values_np)
        typing_weights = bkd.asarray(weights_np)
        typing_mean = SampleAverageMean(bkd)
        typing_result = bkd.to_numpy(typing_mean(typing_values, typing_weights))

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )

    @slow_test
    def test_sample_statistics_variance_matches_legacy(self):
        """Test SampleAverageVariance matches legacy."""
        np.random.seed(42)
        nsamples = 20
        nqoi = 3

        values_np = np.random.randn(nsamples, nqoi)
        weights_np = np.random.dirichlet(np.ones(nsamples))[:, None]

        # Legacy
        from pyapprox.optimization.sampleaverage import (
            SampleAverageVariance as LegacySampleAverageVariance,
        )
        from pyapprox.util.backends.torch import TorchMixin

        import torch
        torch.set_default_dtype(torch.float64)

        legacy_values = TorchMixin.asarray(values_np)
        legacy_weights = TorchMixin.asarray(weights_np)
        legacy_var = LegacySampleAverageVariance(TorchMixin)
        legacy_result = TorchMixin.to_numpy(legacy_var(legacy_values, legacy_weights))

        # Typing
        from pyapprox.typing.util.backends.torch import TorchBkd
        from pyapprox.typing.expdesign import SampleAverageVariance

        bkd = TorchBkd()
        typing_values = bkd.asarray(values_np)
        typing_weights = bkd.asarray(weights_np)
        typing_var = SampleAverageVariance(bkd)
        typing_result = bkd.to_numpy(typing_var(typing_values, typing_weights))

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_result).reshape(-1),
            self._bkd.asarray(legacy_result).reshape(-1),
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
