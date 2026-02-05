"""
Standalone tests for Gaussian likelihood value computations.

PERMANENT - no legacy imports.

Tests verify correctness using:
1. Manual Gaussian log-pdf computation
2. Consistency checks between inner/outer likelihoods
3. Shape and variance scaling checks
"""

import math
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


class TestLikelihoodValuesStandalone(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Standalone tests for Gaussian likelihood values."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

        # Default test dimensions
        self._nobs = 3
        self._ninner = 10
        self._nouter = 5

    def _manual_gaussian_logpdf(
        self, residuals: Array, variances: Array
    ) -> Array:
        """Compute Gaussian log-pdf manually for verification.

        log N(r | 0, diag(v)) = -0.5 * (n*log(2*pi) + sum(log(v)) + sum(r^2/v))

        Parameters
        ----------
        residuals : Array
            Residuals (obs - mean). Shape: (nobs,) for single sample
        variances : Array
            Effective variances. Shape: (nobs,)

        Returns
        -------
        Array
            Scalar log-pdf value
        """
        bkd = self._bkd
        nobs = residuals.shape[0]
        log_norm = -0.5 * nobs * math.log(2 * math.pi)
        log_det = -0.5 * bkd.sum(bkd.log(variances))
        quad_term = -0.5 * bkd.sum(residuals**2 / variances)
        return log_norm + log_det + quad_term

    def _create_outer_likelihood(
        self, nobs: int, nouter: int
    ) -> tuple[GaussianOEDOuterLoopLikelihood[Array], Array, Array, Array]:
        """Create outer likelihood with test data."""
        np.random.seed(42)
        bkd = self._bkd

        noise_variances = bkd.asarray(np.random.uniform(0.1, 0.5, nobs))
        shapes = bkd.asarray(np.random.randn(nobs, nouter))
        observations = bkd.asarray(np.random.randn(nobs, nouter))

        likelihood = GaussianOEDOuterLoopLikelihood(noise_variances, bkd)
        likelihood.set_shapes(shapes)
        likelihood.set_observations(observations)

        return likelihood, noise_variances, shapes, observations

    def _create_inner_likelihood(
        self, nobs: int, ninner: int, nouter: int
    ) -> tuple[GaussianOEDInnerLoopLikelihood[Array], Array, Array, Array]:
        """Create inner likelihood with test data."""
        np.random.seed(42)
        bkd = self._bkd

        noise_variances = bkd.asarray(np.random.uniform(0.1, 0.5, nobs))
        shapes = bkd.asarray(np.random.randn(nobs, ninner))
        observations = bkd.asarray(np.random.randn(nobs, nouter))

        likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
        likelihood.set_shapes(shapes)
        likelihood.set_observations(observations)

        return likelihood, noise_variances, shapes, observations

    # ==========================================================================
    # Outer likelihood tests
    # ==========================================================================

    @parametrize(
        "nobs,nouter",
        [
            (2, 5),
            (3, 10),
            (5, 20),
        ],
    )
    def test_outer_likelihood_gaussian_formula(
        self, nobs: int, nouter: int
    ) -> None:
        """Test outer likelihood values match manual Gaussian log-pdf."""
        likelihood, noise_variances, shapes, observations = (
            self._create_outer_likelihood(nobs, nouter)
        )
        bkd = self._bkd

        # Create weights
        np.random.seed(123)
        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))

        # Compute via likelihood
        log_like = likelihood(weights)  # (1, nouter)

        # Compute manually for each outer sample
        for j in range(nouter):
            residuals = observations[:, j] - shapes[:, j]
            effective_variances = noise_variances / weights[:, 0]
            expected = self._manual_gaussian_logpdf(residuals, effective_variances)

            bkd.assert_allclose(
                bkd.asarray([log_like[0, j]]),
                bkd.asarray([expected]),
                rtol=1e-10,
            )

    def test_outer_likelihood_shape(self) -> None:
        """Test outer likelihood output shape is (1, nouter)."""
        likelihood, _, _, _ = self._create_outer_likelihood(
            self._nobs, self._nouter
        )
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        result = likelihood(weights)
        self.assertEqual(result.shape, (1, self._nouter))

    def test_outer_likelihood_jacobian_shape(self) -> None:
        """Test outer likelihood Jacobian shape is (nouter, nobs)."""
        likelihood, _, _, _ = self._create_outer_likelihood(
            self._nobs, self._nouter
        )
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        jac = likelihood.jacobian(weights)
        self.assertEqual(jac.shape, (self._nouter, self._nobs))

    # ==========================================================================
    # Inner likelihood tests
    # ==========================================================================

    def test_inner_likelihood_matrix_shape(self) -> None:
        """Test inner likelihood logpdf_matrix shape is (ninner, nouter)."""
        likelihood, _, _, _ = self._create_inner_likelihood(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        result = likelihood.logpdf_matrix(weights)
        self.assertEqual(result.shape, (self._ninner, self._nouter))

    def test_inner_jacobian_matrix_shape(self) -> None:
        """Test inner likelihood jacobian_matrix shape is (ninner, nouter, nobs)."""
        likelihood, _, _, _ = self._create_inner_likelihood(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        jac = likelihood.jacobian_matrix(weights)
        self.assertEqual(jac.shape, (self._ninner, self._nouter, self._nobs))

    @parametrize(
        "nobs,ninner,nouter",
        [
            (2, 5, 3),
            (3, 10, 5),
            (4, 8, 6),
        ],
    )
    def test_inner_likelihood_matrix_values(
        self, nobs: int, ninner: int, nouter: int
    ) -> None:
        """Test inner likelihood matrix values match manual computation."""
        likelihood, noise_variances, shapes, observations = (
            self._create_inner_likelihood(nobs, ninner, nouter)
        )
        bkd = self._bkd

        np.random.seed(123)
        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))

        log_like_matrix = likelihood.logpdf_matrix(weights)

        # Verify each entry manually
        for i in range(ninner):
            for j in range(nouter):
                residuals = observations[:, j] - shapes[:, i]
                effective_variances = noise_variances / weights[:, 0]
                expected = self._manual_gaussian_logpdf(residuals, effective_variances)

                bkd.assert_allclose(
                    bkd.asarray([log_like_matrix[i, j]]),
                    bkd.asarray([expected]),
                    rtol=1e-10,
                )

    # ==========================================================================
    # Variance scaling tests
    # ==========================================================================

    def test_likelihood_weights_scaling(self) -> None:
        """Test effective variance scales as base_var / weights."""
        likelihood, noise_variances, _, _ = self._create_outer_likelihood(
            self._nobs, self._nouter
        )
        bkd = self._bkd

        # With weights=1, effective variance = base variance
        weights1 = bkd.ones((self._nobs, 1))
        log_like1 = likelihood(weights1)

        # With weights=2, effective variance = base variance / 2
        # This increases precision, so likelihood should increase
        weights2 = bkd.ones((self._nobs, 1)) * 2.0
        log_like2 = likelihood(weights2)

        # Higher weights = smaller variance = likelihood values change
        # (not necessarily higher due to quadratic term)
        self.assertFalse(
            bkd.allclose(log_like1, log_like2, rtol=1e-6),
            "Likelihood should change with different weights",
        )

    def test_likelihood_heteroscedastic_variances(self) -> None:
        """Test likelihood handles different noise variance per observation."""
        bkd = self._bkd
        nobs, nouter = 3, 5

        # Create heteroscedastic variances
        noise_variances = bkd.asarray(np.array([0.1, 0.5, 1.0]))

        np.random.seed(42)
        shapes = bkd.asarray(np.random.randn(nobs, nouter))
        observations = bkd.asarray(np.random.randn(nobs, nouter))

        likelihood = GaussianOEDOuterLoopLikelihood(noise_variances, bkd)
        likelihood.set_shapes(shapes)
        likelihood.set_observations(observations)

        weights = bkd.ones((nobs, 1))
        log_like = likelihood(weights)

        # Should be finite
        log_like_np = bkd.to_numpy(log_like)
        self.assertTrue(np.all(np.isfinite(log_like_np)))

        # Verify against manual computation
        for j in range(nouter):
            residuals = observations[:, j] - shapes[:, j]
            expected = self._manual_gaussian_logpdf(residuals, noise_variances)
            bkd.assert_allclose(
                bkd.asarray([log_like[0, j]]),
                bkd.asarray([expected]),
                rtol=1e-10,
            )

    # ==========================================================================
    # Log-determinant and quadratic term tests
    # ==========================================================================

    def test_likelihood_logdet_term(self) -> None:
        """Test log-determinant term is computed correctly.

        For diagonal covariance with effective_var = base_var / weights:
        log|Cov| = sum(log(base_var / weights)) = sum(log(base_var)) - sum(log(weights))
        """
        bkd = self._bkd
        nobs, nouter = 3, 2

        noise_variances = bkd.asarray(np.array([0.1, 0.2, 0.3]))
        shapes = bkd.zeros((nobs, nouter))  # Zero shapes
        observations = bkd.zeros((nobs, nouter))  # Zero observations

        likelihood = GaussianOEDOuterLoopLikelihood(noise_variances, bkd)
        likelihood.set_shapes(shapes)
        likelihood.set_observations(observations)

        weights = bkd.asarray(np.array([[1.0], [2.0], [0.5]]))
        log_like = likelihood(weights)

        # With zero residuals, only log-determinant term matters
        # log p = -0.5 * n * log(2*pi) - 0.5 * log|Cov|
        # log|Cov| = sum(log(var_i / w_i))
        effective_vars = noise_variances / weights[:, 0]
        expected_log_det = bkd.sum(bkd.log(effective_vars))
        expected = -0.5 * nobs * math.log(2 * math.pi) - 0.5 * expected_log_det

        # All outer samples should have the same value
        for j in range(nouter):
            bkd.assert_allclose(
                bkd.asarray([log_like[0, j]]),
                bkd.asarray([expected]),
                rtol=1e-10,
            )

    def test_likelihood_quadratic_term(self) -> None:
        """Test quadratic term is computed correctly.

        quadratic = -0.5 * sum(r_i^2 / effective_var_i)
                  = -0.5 * sum(r_i^2 * w_i / base_var_i)
        """
        bkd = self._bkd
        nobs, nouter = 2, 1

        noise_variances = bkd.asarray(np.array([1.0, 1.0]))  # Unit variance
        shapes = bkd.zeros((nobs, nouter))

        # Set observations to have known residuals
        observations = bkd.asarray(np.array([[1.0], [2.0]]))

        likelihood = GaussianOEDOuterLoopLikelihood(noise_variances, bkd)
        likelihood.set_shapes(shapes)
        likelihood.set_observations(observations)

        weights = bkd.ones((nobs, 1))
        log_like = likelihood(weights)

        # With unit variance and unit weights:
        # quadratic = -0.5 * (1^2 + 2^2) = -0.5 * 5 = -2.5
        # log_det = 0 (log(1) = 0)
        # log_norm = -0.5 * 2 * log(2*pi)
        expected = -0.5 * nobs * math.log(2 * math.pi) - 2.5

        bkd.assert_allclose(
            bkd.asarray([log_like[0, 0]]),
            bkd.asarray([expected]),
            rtol=1e-10,
        )

    # ==========================================================================
    # Outer/Inner consistency tests
    # ==========================================================================

    def test_outer_inner_diagonal_consistency(self) -> None:
        """Test that inner matrix diagonal matches outer values when shapes align.

        When inner and outer shapes are the same, the diagonal of the inner
        likelihood matrix should match the outer likelihood values.
        """
        bkd = self._bkd
        nobs, nouter = 3, 5

        np.random.seed(42)
        noise_variances = bkd.asarray(np.random.uniform(0.1, 0.5, nobs))
        # Use same shapes for both
        shared_shapes = bkd.asarray(np.random.randn(nobs, nouter))
        observations = bkd.asarray(np.random.randn(nobs, nouter))

        # Create outer likelihood
        outer_likelihood = GaussianOEDOuterLoopLikelihood(noise_variances, bkd)
        outer_likelihood.set_shapes(shared_shapes)
        outer_likelihood.set_observations(observations)

        # Create inner likelihood with same shapes
        inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
        inner_likelihood.set_shapes(shared_shapes)  # ninner = nouter
        inner_likelihood.set_observations(observations)

        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))

        outer_log_like = outer_likelihood(weights)  # (1, nouter)
        inner_log_like_matrix = inner_likelihood.logpdf_matrix(weights)  # (nouter, nouter)

        # Diagonal of inner matrix should match outer values
        diagonal = bkd.diag(inner_log_like_matrix)
        bkd.assert_allclose(diagonal, outer_log_like[0], rtol=1e-10)

    def test_likelihood_values_finite(self) -> None:
        """Test all likelihood values are finite."""
        likelihood, _, _, _ = self._create_outer_likelihood(
            self._nobs, self._nouter
        )
        bkd = self._bkd

        np.random.seed(456)
        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))
        log_like = likelihood(weights)

        log_like_np = bkd.to_numpy(log_like)
        self.assertTrue(np.all(np.isfinite(log_like_np)))

    def test_inner_likelihood_values_finite(self) -> None:
        """Test all inner likelihood matrix values are finite."""
        likelihood, _, _, _ = self._create_inner_likelihood(
            self._nobs, self._ninner, self._nouter
        )
        bkd = self._bkd

        np.random.seed(456)
        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))
        log_like_matrix = likelihood.logpdf_matrix(weights)

        log_like_np = bkd.to_numpy(log_like_matrix)
        self.assertTrue(np.all(np.isfinite(log_like_np)))


class TestLikelihoodValuesStandaloneNumpy(
    TestLikelihoodValuesStandalone[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLikelihoodValuesStandaloneTorch(
    TestLikelihoodValuesStandalone[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
