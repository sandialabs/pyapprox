"""
Tests for OED likelihood classes.

Tests cover:
- logpdf_matrix shape and values
- Jacobian correctness via finite differences
- Reparameterization trick gradient
- Parallel vs sequential equivalence (when implemented)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.expdesign.likelihood import (
    GaussianOEDOuterLoopLikelihood,
    GaussianOEDInnerLoopLikelihood,
)


class TestGaussianOEDLikelihood(Generic[Array], unittest.TestCase):
    """Base test class for Gaussian OED likelihood."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        # Set up test data
        self._nobs = 3
        self._ninner = 5
        self._nouter = 4

        np.random.seed(42)
        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.2, 0.15])
        )
        self._shapes_inner = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._shapes_outer = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._latent_samples = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._design_weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

    def _generate_observations(self, shapes: Array, weights: Array) -> Array:
        """Generate observations using reparameterization trick."""
        # obs = shapes + sqrt(var / weights) * latent
        std = self._bkd.sqrt(
            self._noise_variances[:, None] / weights
        )
        return shapes + std * self._latent_samples

    # --- Outer Loop Likelihood Tests ---

    def test_outer_loop_shapes(self):
        """Test that outer loop likelihood returns correct shapes."""
        likelihood = GaussianOEDOuterLoopLikelihood(
            self._noise_variances, self._bkd
        )
        likelihood.set_shapes(self._shapes_outer)
        obs = self._generate_observations(
            self._shapes_outer, self._design_weights
        )
        likelihood.set_observations(obs)

        values = likelihood(self._design_weights)
        self.assertEqual(values.shape, (1, self._nouter))

        jac = likelihood.jacobian(self._design_weights)
        self.assertEqual(jac.shape, (self._nouter, self._nobs))

    def test_outer_loop_values(self):
        """Test outer loop likelihood values against manual computation."""
        likelihood = GaussianOEDOuterLoopLikelihood(
            self._noise_variances, self._bkd
        )
        likelihood.set_shapes(self._shapes_outer)

        # Generate observations without reparameterization for simpler test
        obs = self._shapes_outer + 0.1  # Fixed offset
        likelihood.set_observations(obs)

        values = likelihood(self._design_weights)

        # Manual computation
        residuals = obs - self._shapes_outer
        inv_var = self._design_weights[:, 0] / self._noise_variances
        squared_dist = self._bkd.sum(residuals**2 * inv_var[:, None], axis=0)

        log_det = float(
            self._bkd.sum(self._bkd.log(self._noise_variances))
            - self._bkd.sum(self._bkd.log(self._design_weights))
        )
        log_norm = -0.5 * self._nobs * np.log(2 * np.pi) - 0.5 * log_det
        expected = log_norm - 0.5 * squared_dist

        self.assertTrue(
            self._bkd.allclose(values[0], expected, rtol=1e-10)
        )

    def test_outer_loop_jacobian_finite_diff(self):
        """Test outer loop Jacobian against finite differences."""
        likelihood = GaussianOEDOuterLoopLikelihood(
            self._noise_variances, self._bkd
        )
        likelihood.set_shapes(self._shapes_outer)
        obs = self._shapes_outer + 0.05
        likelihood.set_observations(obs)

        # Compute analytical Jacobian (without reparameterization term)
        jac_analytical = likelihood.jacobian(self._design_weights)

        # Compute finite difference Jacobian
        eps = 1e-6
        jac_fd = self._bkd.zeros((self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = self._bkd.copy(self._design_weights)
            weights_minus = self._bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = likelihood(weights_plus)
            val_minus = likelihood(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)
        )

    def test_outer_loop_jacobian_with_reparam(self):
        """Test outer loop Jacobian with reparameterization trick."""
        likelihood = GaussianOEDOuterLoopLikelihood(
            self._noise_variances, self._bkd
        )
        likelihood.set_shapes(self._shapes_outer)
        likelihood.set_latent_samples(self._latent_samples)

        # Generate observations using reparameterization
        obs = self._generate_observations(
            self._shapes_outer, self._design_weights
        )
        likelihood.set_observations(obs)

        # Compute analytical Jacobian
        jac_analytical = likelihood.jacobian(self._design_weights)

        # Compute finite difference Jacobian
        # Need to recompute observations for each perturbed weight
        eps = 1e-6
        jac_fd = self._bkd.zeros((self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = self._bkd.copy(self._design_weights)
            weights_minus = self._bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            # Recompute observations with perturbed weights
            obs_plus = self._generate_observations(
                self._shapes_outer, weights_plus
            )
            obs_minus = self._generate_observations(
                self._shapes_outer, weights_minus
            )

            likelihood.set_observations(obs_plus)
            val_plus = likelihood(weights_plus)

            likelihood.set_observations(obs_minus)
            val_minus = likelihood(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-6)
        )

    # --- Inner Loop Likelihood Tests ---

    def test_inner_loop_shapes(self):
        """Test that inner loop likelihood returns correct shapes."""
        likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        likelihood.set_shapes(self._shapes_inner)

        obs = self._generate_observations(
            self._shapes_outer, self._design_weights
        )
        likelihood.set_observations(obs)

        matrix = likelihood.logpdf_matrix(self._design_weights)
        self.assertEqual(matrix.shape, (self._ninner, self._nouter))

        jac = likelihood.jacobian_matrix(self._design_weights)
        self.assertEqual(jac.shape, (self._ninner, self._nouter, self._nobs))

    def test_inner_loop_values(self):
        """Test inner loop likelihood matrix values."""
        likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        likelihood.set_shapes(self._shapes_inner)
        likelihood.set_observations(self._shapes_outer)  # Use shapes as obs

        matrix = likelihood.logpdf_matrix(self._design_weights)

        # Manual computation for one entry
        i, j = 1, 2  # inner sample 1, outer sample 2
        residuals = self._shapes_outer[:, j] - self._shapes_inner[:, i]
        inv_var = self._design_weights[:, 0] / self._noise_variances
        squared_dist = float(self._bkd.sum(residuals**2 * inv_var))

        log_det = float(
            self._bkd.sum(self._bkd.log(self._noise_variances))
            - self._bkd.sum(self._bkd.log(self._design_weights))
        )
        log_norm = -0.5 * self._nobs * np.log(2 * np.pi) - 0.5 * log_det
        expected = log_norm - 0.5 * squared_dist

        actual = float(self._bkd.to_numpy(matrix[i, j]))
        self.assertTrue(
            self._bkd.allclose(
                self._bkd.asarray([[actual]]),
                self._bkd.asarray([[expected]]),
                rtol=1e-10
            )
        )

    def test_inner_loop_jacobian_finite_diff(self):
        """Test inner loop Jacobian matrix against finite differences."""
        likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        likelihood.set_shapes(self._shapes_inner)
        likelihood.set_observations(self._shapes_outer)

        jac_analytical = likelihood.jacobian_matrix(self._design_weights)

        # Compute finite difference
        eps = 1e-6
        jac_fd = self._bkd.zeros((self._ninner, self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = self._bkd.copy(self._design_weights)
            weights_minus = self._bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            matrix_plus = likelihood.logpdf_matrix(weights_plus)
            matrix_minus = likelihood.logpdf_matrix(weights_minus)

            jac_fd[:, :, k] = (matrix_plus - matrix_minus) / (2 * eps)

        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)
        )

    def test_create_outer_loop_likelihood(self):
        """Test factory method for outer loop likelihood."""
        inner = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        outer = inner.create_outer_loop_likelihood()

        self.assertIsInstance(outer, GaussianOEDOuterLoopLikelihood)
        self.assertEqual(outer.nobs(), inner.nobs())

    def test_consistency_inner_outer(self):
        """Test that inner and outer loop agree for matching samples."""
        inner = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        outer = inner.create_outer_loop_likelihood()

        # Use same shapes for both (ninner = nouter = 4)
        shapes = self._shapes_outer
        obs = shapes + 0.1

        inner.set_shapes(shapes)
        inner.set_observations(obs)

        outer.set_shapes(shapes)
        outer.set_observations(obs)

        matrix = inner.logpdf_matrix(self._design_weights)
        values = outer(self._design_weights)

        # Diagonal of matrix should match outer values
        diag = self._bkd.asarray([matrix[i, i] for i in range(shapes.shape[1])])
        self.assertTrue(
            self._bkd.allclose(diag, values[0], rtol=1e-10)
        )


class TestGaussianOEDLikelihoodNumpy(TestGaussianOEDLikelihood[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianOEDLikelihoodTorch(TestGaussianOEDLikelihood[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
