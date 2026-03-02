"""
Tests for OED likelihood classes.

Tests cover:
- logpdf_matrix shape and values
- Jacobian correctness via finite differences
- Reparameterization trick gradient
- Parallel vs sequential equivalence (when implemented)
"""

import numpy as np

from pyapprox.expdesign.likelihood import (
    GaussianOEDInnerLoopLikelihood,
    GaussianOEDOuterLoopLikelihood,
)


class TestGaussianOEDLikelihood:
    """Base test class for Gaussian OED likelihood."""

    def _setup_data(self, bkd):
        """Set up test data."""
        self._nobs = 3
        self._ninner = 5
        self._nouter = 4

        np.random.seed(42)
        self._noise_variances = bkd.asarray(np.array([0.1, 0.2, 0.15]))
        self._shapes_inner = bkd.asarray(np.random.randn(self._nobs, self._ninner))
        self._shapes_outer = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        self._latent_samples = bkd.asarray(np.random.randn(self._nobs, self._nouter))
        self._design_weights = bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))

    def _generate_observations(self, bkd, shapes, weights):
        """Generate observations using reparameterization trick."""
        # obs = shapes + sqrt(var / weights) * latent
        std = bkd.sqrt(self._noise_variances[:, None] / weights)
        return shapes + std * self._latent_samples

    # --- Outer Loop Likelihood Tests ---

    def test_outer_loop_values(self, bkd):
        """Test outer loop likelihood values against manual computation."""
        self._setup_data(bkd)
        likelihood = GaussianOEDOuterLoopLikelihood(self._noise_variances, bkd)
        likelihood.set_shapes(self._shapes_outer)

        # Generate observations without reparameterization for simpler test
        obs = self._shapes_outer + 0.1  # Fixed offset
        likelihood.set_observations(obs)

        values = likelihood(self._design_weights)

        # Manual computation
        residuals = obs - self._shapes_outer
        inv_var = self._design_weights[:, 0] / self._noise_variances
        squared_dist = bkd.sum(residuals**2 * inv_var[:, None], axis=0)

        log_det = float(
            bkd.sum(bkd.log(self._noise_variances))
            - bkd.sum(bkd.log(self._design_weights))
        )
        log_norm = -0.5 * self._nobs * np.log(2 * np.pi) - 0.5 * log_det
        expected = log_norm - 0.5 * squared_dist

        assert bkd.allclose(values[0], expected, rtol=1e-10)

    def test_outer_loop_jacobian_finite_diff(self, bkd):
        """Test outer loop Jacobian against finite differences."""
        self._setup_data(bkd)
        likelihood = GaussianOEDOuterLoopLikelihood(self._noise_variances, bkd)
        likelihood.set_shapes(self._shapes_outer)
        obs = self._shapes_outer + 0.05
        likelihood.set_observations(obs)

        # Compute analytical Jacobian (without reparameterization term)
        jac_analytical = likelihood.jacobian(self._design_weights)

        # Compute finite difference Jacobian
        eps = 1e-6
        jac_fd = bkd.zeros((self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = bkd.copy(self._design_weights)
            weights_minus = bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = likelihood(weights_plus)
            val_minus = likelihood(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)

    def test_outer_loop_jacobian_with_reparam(self, bkd):
        """Test outer loop Jacobian with reparameterization trick."""
        self._setup_data(bkd)
        likelihood = GaussianOEDOuterLoopLikelihood(self._noise_variances, bkd)
        likelihood.set_shapes(self._shapes_outer)
        likelihood.set_latent_samples(self._latent_samples)

        # Generate observations using reparameterization
        obs = self._generate_observations(bkd, self._shapes_outer, self._design_weights)
        likelihood.set_observations(obs)

        # Compute analytical Jacobian
        jac_analytical = likelihood.jacobian(self._design_weights)

        # Compute finite difference Jacobian
        # Need to recompute observations for each perturbed weight
        eps = 1e-6
        jac_fd = bkd.zeros((self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = bkd.copy(self._design_weights)
            weights_minus = bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            # Recompute observations with perturbed weights
            obs_plus = self._generate_observations(
                bkd, self._shapes_outer, weights_plus
            )
            obs_minus = self._generate_observations(
                bkd, self._shapes_outer, weights_minus
            )

            likelihood.set_observations(obs_plus)
            val_plus = likelihood(weights_plus)

            likelihood.set_observations(obs_minus)
            val_minus = likelihood(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-6)

    # --- Inner Loop Likelihood Tests ---

    def test_inner_loop_values(self, bkd):
        """Test inner loop likelihood matrix values."""
        self._setup_data(bkd)
        likelihood = GaussianOEDInnerLoopLikelihood(self._noise_variances, bkd)
        likelihood.set_shapes(self._shapes_inner)
        likelihood.set_observations(self._shapes_outer)  # Use shapes as obs

        matrix = likelihood.logpdf_matrix(self._design_weights)

        # Manual computation for one entry
        i, j = 1, 2  # inner sample 1, outer sample 2
        residuals = self._shapes_outer[:, j] - self._shapes_inner[:, i]
        inv_var = self._design_weights[:, 0] / self._noise_variances
        squared_dist = float(bkd.sum(residuals**2 * inv_var))

        log_det = float(
            bkd.sum(bkd.log(self._noise_variances))
            - bkd.sum(bkd.log(self._design_weights))
        )
        log_norm = -0.5 * self._nobs * np.log(2 * np.pi) - 0.5 * log_det
        expected = log_norm - 0.5 * squared_dist

        actual = float(bkd.to_numpy(matrix[i, j]))
        bkd.assert_allclose(
            bkd.asarray([[actual]]),
            bkd.asarray([[expected]]),
            rtol=1e-10,
        )

    def test_inner_loop_jacobian_finite_diff(self, bkd):
        """Test inner loop Jacobian matrix against finite differences."""
        self._setup_data(bkd)
        likelihood = GaussianOEDInnerLoopLikelihood(self._noise_variances, bkd)
        likelihood.set_shapes(self._shapes_inner)
        likelihood.set_observations(self._shapes_outer)

        jac_analytical = likelihood.jacobian_matrix(self._design_weights)

        # Compute finite difference
        eps = 1e-6
        jac_fd = bkd.zeros((self._ninner, self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = bkd.copy(self._design_weights)
            weights_minus = bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            matrix_plus = likelihood.logpdf_matrix(weights_plus)
            matrix_minus = likelihood.logpdf_matrix(weights_minus)

            jac_fd[:, :, k] = (matrix_plus - matrix_minus) / (2 * eps)

        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-5, atol=1e-7)

    def test_create_outer_loop_likelihood(self, bkd):
        """Test factory method for outer loop likelihood."""
        self._setup_data(bkd)
        inner = GaussianOEDInnerLoopLikelihood(self._noise_variances, bkd)
        outer = inner.create_outer_loop_likelihood()

        assert isinstance(outer, GaussianOEDOuterLoopLikelihood)
        assert outer.nobs() == inner.nobs()

    def test_consistency_inner_outer(self, bkd):
        """Test that inner and outer loop agree for matching samples."""
        self._setup_data(bkd)
        inner = GaussianOEDInnerLoopLikelihood(self._noise_variances, bkd)
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
        diag = bkd.asarray([matrix[i, i] for i in range(shapes.shape[1])])
        assert bkd.allclose(diag, values[0], rtol=1e-10)
