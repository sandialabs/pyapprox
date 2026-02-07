"""
Legacy comparison tests for nonlinear prediction OED convergence.

TODO: Delete after legacy removed.

Tests verify that the typed prediction OED diagnostics produce results
consistent with the legacy implementation for the nonlinear lognormal
benchmark. This test exactly replicates the setup from:
    pyapprox/expdesign/tests/test_bayesoed.py::test_prediction_OED_values_nonlinear_problem

The key insight is that the noise statistic must be different for
different utility types:
- Expected StdDev: uses SampleAverageMean as noise_stat
- AVaR StdDev: uses SampleAverageSmoothedAVaR as noise_stat
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestNonlinearPredictionOEDLegacyComparison(
    Generic[Array], unittest.TestCase
):
    """
    Legacy comparison tests for nonlinear prediction OED.

    Compares typing and legacy implementations step-by-step.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _create_legacy_benchmark(self):
        """Create legacy benchmark matching test_prediction_OED_values_nonlinear_problem."""
        from pyapprox.util.backends.numpy import NumpyMixin
        from pyapprox.expdesign.bayesoed_benchmarks import (
            NonLinearGaussianBayesianOEDForPredictionBenchmark,
        )

        nobs = 2
        min_degree = 0
        degree = 3
        noise_std = 0.125 * 4  # 0.5
        prior_std = 0.5
        nqoi = 1

        return NonLinearGaussianBayesianOEDForPredictionBenchmark(
            nobs, min_degree, degree, noise_std, prior_std,
            nqoi=nqoi, backend=NumpyMixin
        )

    def _create_typing_benchmark(self):
        """Create typing benchmark matching legacy."""
        from pyapprox.typing.expdesign.benchmarks import NonLinearGaussianOEDBenchmark

        bkd = NumpyBkd()  # Always use NumPy for comparison
        return NonLinearGaussianOEDBenchmark(
            nobs=2,
            degree=3,
            noise_std=0.125 * 4,  # 0.5
            prior_std=0.5,
            bkd=bkd,
            npred=1,
            min_degree=0,
        )

    def test_benchmark_design_matrices_match(self) -> None:
        """Test that benchmark design matrices match legacy."""
        legacy = self._create_legacy_benchmark()
        typing = self._create_typing_benchmark()

        # Design matrix
        from pyapprox.util.backends.numpy import NumpyMixin
        legacy_design = NumpyMixin.to_numpy(legacy.get_observation_model().matrix())
        typing_design = NumpyBkd().to_numpy(typing.design_matrix())

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_design),
            self._bkd.asarray(legacy_design),
            rtol=1e-12,
        )

    def test_benchmark_qoi_matrices_match(self) -> None:
        """Test that benchmark QoI matrices match legacy."""
        legacy = self._create_legacy_benchmark()
        typing = self._create_typing_benchmark()

        from pyapprox.util.backends.numpy import NumpyMixin
        legacy_qoi = NumpyMixin.to_numpy(legacy.get_qoi_model().matrix())
        typing_qoi = NumpyBkd().to_numpy(typing.qoi_matrix())

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_qoi),
            self._bkd.asarray(legacy_qoi),
            rtol=1e-12,
        )

    def test_exact_stdev_utility_matches_legacy(self) -> None:
        """Test exact stdev utility matches legacy."""
        legacy = self._create_legacy_benchmark()
        typing = self._create_typing_benchmark()
        bkd = NumpyBkd()

        # Design weights
        nobs = 2
        design_weights = bkd.ones((nobs, 1)) / nobs

        # Legacy exact utility
        from pyapprox.util.backends.numpy import NumpyMixin
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        from pyapprox.expdesign.bayesoed_benchmarks import (
            ConjugateGaussianOEDForLogNormalExpectedStdDev,
        )

        legacy_prior = DenseCholeskyMultivariateGaussian(
            legacy.get_prior().mean(),
            legacy.get_prior().covariance(),
            backend=NumpyMixin,
        )
        legacy_utility = ConjugateGaussianOEDForLogNormalExpectedStdDev(
            legacy_prior, legacy.get_qoi_model().matrix()
        )
        legacy_utility.set_observation_matrix(legacy.get_observation_model().matrix())
        legacy_noise_cov_diag = legacy.get_noise_covariance_diag() / NumpyMixin.to_numpy(design_weights)[:, 0]
        legacy_utility.set_noise_covariance(NumpyMixin.diag(legacy_noise_cov_diag))
        legacy_exact = float(legacy_utility.value())

        # Typing exact utility
        from pyapprox.typing.expdesign.analytical import (
            ConjugateGaussianOEDForLogNormalExpectedStdDev as TypingStdDevUtility,
        )

        typing_utility = TypingStdDevUtility(
            typing.prior_mean(),
            typing.prior_covariance(),
            typing.qoi_matrix(),
            bkd,
        )
        typing_utility.set_observation_matrix(typing.design_matrix())
        typing_noise_cov_diag = typing.noise_variances() / bkd.reshape(design_weights, (nobs,))
        typing_utility.set_noise_covariance(bkd.diag(typing_noise_cov_diag))
        typing_exact = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_exact).reshape(-1),
            self._bkd.asarray(legacy_exact).reshape(-1),
            rtol=1e-12,
        )

    def test_exact_avar_stdev_utility_matches_legacy(self) -> None:
        """Test exact AVaR stdev utility matches legacy."""
        legacy = self._create_legacy_benchmark()
        typing = self._create_typing_benchmark()
        bkd = NumpyBkd()

        # Design weights
        nobs = 2
        design_weights = bkd.ones((nobs, 1)) / nobs
        beta = 0.5

        # Legacy exact utility
        from pyapprox.util.backends.numpy import NumpyMixin
        from pyapprox.variables.gaussian import DenseCholeskyMultivariateGaussian
        from pyapprox.expdesign.bayesoed_benchmarks import (
            ConjugateGaussianOEDForLogNormalAVaRStdDev,
        )

        legacy_prior = DenseCholeskyMultivariateGaussian(
            legacy.get_prior().mean(),
            legacy.get_prior().covariance(),
            backend=NumpyMixin,
        )
        legacy_utility = ConjugateGaussianOEDForLogNormalAVaRStdDev(
            legacy_prior, legacy.get_qoi_model().matrix(), beta
        )
        legacy_utility.set_observation_matrix(legacy.get_observation_model().matrix())
        legacy_noise_cov_diag = legacy.get_noise_covariance_diag() / NumpyMixin.to_numpy(design_weights)[:, 0]
        legacy_utility.set_noise_covariance(NumpyMixin.diag(legacy_noise_cov_diag))
        legacy_exact = float(legacy_utility.value())

        # Typing exact utility
        from pyapprox.typing.expdesign.analytical import (
            ConjugateGaussianOEDForLogNormalAVaRStdDev as TypingAVaRUtility,
        )

        typing_utility = TypingAVaRUtility(
            typing.prior_mean(),
            typing.prior_covariance(),
            typing.qoi_matrix(),
            beta,
            bkd,
        )
        typing_utility.set_observation_matrix(typing.design_matrix())
        typing_noise_cov_diag = typing.noise_variances() / bkd.reshape(design_weights, (nobs,))
        typing_utility.set_noise_covariance(bkd.diag(typing_noise_cov_diag))
        typing_exact = typing_utility.value()

        self._bkd.assert_allclose(
            self._bkd.asarray(typing_exact).reshape(-1),
            self._bkd.asarray(legacy_exact).reshape(-1),
            rtol=1e-12,
        )



class TestNonlinearPredictionOEDLegacyComparisonNumpy(
    TestNonlinearPredictionOEDLegacyComparison[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


if __name__ == "__main__":
    unittest.main()
