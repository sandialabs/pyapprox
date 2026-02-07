"""
Legacy comparison tests for OED likelihood implementations.

Compares typing module likelihoods against IndependentGaussianOEDOuterLoopLogLikelihood
and IndependentGaussianOEDInnerLoopLogLikelihood from legacy pyapprox.expdesign.bayesoed.

Replicates legacy test_OED_gaussian_likelihood_values from test_bayesoed.py:391-490.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.likelihood import (
    GaussianOEDOuterLoopLikelihood,
    GaussianOEDInnerLoopLikelihood,
)

# Import legacy modules for comparison
from pyapprox.expdesign.bayesoed_benchmarks import LinearGaussianBayesianOEDBenchmark
from pyapprox.expdesign.bayesoed import (
    IndependentGaussianOEDInnerLoopLogLikelihood,
    IndependentGaussianOEDOuterLoopLogLikelihood,
)
from pyapprox.util.backends.numpy import NumpyMixin


class TestOEDLikelihoodLegacyComparison(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Legacy comparison tests for OED likelihood implementations.

    Compares GaussianOEDOuterLoopLikelihood and GaussianOEDInnerLoopLikelihood
    against legacy OED likelihood classes (vectorized comparison).

    Replicates test_OED_gaussian_likelihood_values from test_bayesoed.py:391-490.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    @parametrize(
        "nobs,min_degree,degree,noise_std,prior_std",
        [
            (3, 0, 3, 0.5, 0.5),  # Original legacy test case
        ],
    )
    def test_outer_loop_likelihood_matches_legacy(
        self,
        nobs: int,
        min_degree: int,
        degree: int,
        noise_std: float,
        prior_std: float,
    ):
        """Compare GaussianOEDOuterLoopLikelihood against legacy OED likelihood.

        Both compute log p(obs_i | shape_i, weights) for all i in a single
        vectorized call.

        Replicates test_bayesoed.py:451-468.
        """
        np.random.seed(42)

        # Create legacy benchmark
        legacy_benchmark = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std,
            backend=NumpyMixin,
        )

        nouter = 100
        nparams = degree - min_degree + 1

        # Generate outer loop samples from prior
        theta_outer = np.random.randn(nparams, nouter) * prior_std

        # Compute shapes using legacy model
        outer_shapes = legacy_benchmark.get_observation_model()(theta_outer).T

        # Generate latent samples for observations
        latent_samples = np.random.randn(nobs, nouter)

        # Random design weights
        design_weights = np.random.uniform(0.5, 1.0, (nobs, 1))
        design_weights = design_weights / design_weights.sum()

        # Compute observations using reparameterization
        noise_cov_diag = legacy_benchmark.get_noise_covariance_diag()[:, None]
        effective_std = np.sqrt(noise_cov_diag / design_weights)
        observations = outer_shapes + effective_std * latent_samples

        # --- Legacy OED likelihood ---
        legacy_loglike = IndependentGaussianOEDOuterLoopLogLikelihood(
            noise_cov_diag, backend=NumpyMixin
        )
        legacy_loglike.set_shapes(outer_shapes)
        legacy_loglike.set_artificial_observations(observations)
        legacy_loglike.set_latent_likelihood_samples(latent_samples)
        legacy_vals = legacy_loglike(design_weights)  # shape (1, nouter)

        # --- Typing module likelihood ---
        noise_variances = noise_cov_diag[:, 0]
        typing_noise_var = self._bkd.asarray(noise_variances)
        typing_outer_shapes = self._bkd.asarray(outer_shapes)
        typing_observations = self._bkd.asarray(observations)
        typing_design_weights = self._bkd.asarray(design_weights)

        outer_likelihood = GaussianOEDOuterLoopLikelihood(typing_noise_var, self._bkd)
        outer_likelihood.set_shapes(typing_outer_shapes)
        outer_likelihood.set_observations(typing_observations)
        typing_vals = outer_likelihood(typing_design_weights)  # shape (1, nouter)

        # Compare
        self._bkd.assert_allclose(
            typing_vals,
            self._bkd.asarray(legacy_vals),
            rtol=1e-10,
        )

    @parametrize(
        "nobs,min_degree,degree,noise_std,prior_std",
        [
            (3, 0, 3, 0.5, 0.5),  # Original legacy test case
        ],
    )
    def test_inner_loop_likelihood_matches_legacy(
        self,
        nobs: int,
        min_degree: int,
        degree: int,
        noise_std: float,
        prior_std: float,
    ):
        """Compare GaussianOEDInnerLoopLikelihood against legacy OED likelihood.

        Both compute log p(obs_j | shape_i, weights) for all pairs (i,j).
        Legacy returns flattened (1, ninner*nouter), typing returns (ninner, nouter).

        Replicates test_bayesoed.py:476-490.
        """
        np.random.seed(42)

        # Create legacy benchmark
        legacy_benchmark = LinearGaussianBayesianOEDBenchmark(
            nobs, min_degree, degree, noise_std, prior_std,
            backend=NumpyMixin,
        )

        nouter = 50
        ninner = 30
        nparams = degree - min_degree + 1

        # Generate samples
        theta_outer = np.random.randn(nparams, nouter) * prior_std
        theta_inner = np.random.randn(nparams, ninner) * prior_std

        # Compute shapes
        outer_shapes = legacy_benchmark.get_observation_model()(theta_outer).T
        inner_shapes = legacy_benchmark.get_observation_model()(theta_inner).T

        # Generate latent samples
        latent_samples = np.random.randn(nobs, nouter)

        # Random design weights
        design_weights = np.random.uniform(0.5, 1.0, (nobs, 1))
        design_weights = design_weights / design_weights.sum()

        # Compute observations
        noise_cov_diag = legacy_benchmark.get_noise_covariance_diag()[:, None]
        effective_std = np.sqrt(noise_cov_diag / design_weights)
        observations = outer_shapes + effective_std * latent_samples

        # --- Legacy OED likelihood ---
        legacy_loglike = IndependentGaussianOEDInnerLoopLogLikelihood(
            noise_cov_diag, backend=NumpyMixin
        )
        legacy_loglike.set_shapes(inner_shapes)
        legacy_loglike.set_artificial_observations(observations)
        legacy_loglike.set_latent_likelihood_samples(latent_samples)
        # Legacy returns (1, ninner*nouter) flattened
        legacy_vals_flat = legacy_loglike(design_weights)
        # Reshape to (ninner, nouter) - legacy flattens inner loop first
        legacy_vals = legacy_vals_flat.reshape(ninner, nouter, order='F')

        # --- Typing module likelihood ---
        noise_variances = noise_cov_diag[:, 0]
        typing_noise_var = self._bkd.asarray(noise_variances)
        typing_inner_shapes = self._bkd.asarray(inner_shapes)
        typing_observations = self._bkd.asarray(observations)
        typing_design_weights = self._bkd.asarray(design_weights)

        inner_likelihood = GaussianOEDInnerLoopLikelihood(typing_noise_var, self._bkd)
        inner_likelihood.set_shapes(typing_inner_shapes)
        inner_likelihood.set_observations(typing_observations)
        typing_vals = inner_likelihood.logpdf_matrix(typing_design_weights)

        # Compare
        self._bkd.assert_allclose(
            typing_vals,
            self._bkd.asarray(legacy_vals),
            rtol=1e-10,
        )


class TestOEDLikelihoodLegacyComparisonNumpy(
    TestOEDLikelihoodLegacyComparison[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOEDLikelihoodLegacyComparisonTorch(
    TestOEDLikelihoodLegacyComparison[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
