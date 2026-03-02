"""
Tests for KL-OED objective function.

Tests cover:
- Objective value correctness
- Jacobian verification via finite differences
- Expected information gain computation
"""

import numpy as np
import pytest

from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import KLOEDObjective


class TestKLOEDObjective:
    """Base test class for KL-OED objective."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        # Set up test data
        self._nobs = 3
        self._ninner = 15
        self._nouter = 10

        np.random.seed(42)
        self._noise_variances = bkd.asarray(np.array([0.1, 0.2, 0.15]))

        # Model outputs (shapes)
        self._outer_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._inner_shapes = bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )

        # Latent samples for reparameterization
        self._latent_samples = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )

        # Design weights
        self._design_weights = bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

        # Create inner likelihood
        self._inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, bkd
        )

    def _create_objective(self, bkd, outer_weights=None, inner_weights=None):
        """Helper to create KLOEDObjective."""
        return KLOEDObjective(
            self._inner_likelihood,
            self._outer_shapes,
            self._latent_samples,
            self._inner_shapes,
            outer_weights,
            inner_weights,
            bkd,
        )

    def test_objective_scalar(self, bkd):
        """Test that nqoi is 1."""
        objective = self._create_objective(bkd)
        assert objective.nqoi() == 1
        assert objective.nvars() == self._nobs

    @pytest.mark.slow_on("TorchBkd")
    def test_jacobian_finite_diff(self, bkd):
        """Test Jacobian against finite differences."""
        objective = self._create_objective(bkd)

        jac_analytical = objective.jacobian(self._design_weights)

        # Finite difference
        eps = 1e-5
        jac_fd = bkd.zeros((1, self._nobs))
        for k in range(self._nobs):
            weights_plus = bkd.copy(self._design_weights)
            weights_minus = bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = objective(weights_plus)
            val_minus = objective(weights_minus)

            jac_fd[0, k] = (val_plus[0, 0] - val_minus[0, 0]) / (2 * eps)

        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-3, atol=1e-6)

    def test_expected_information_gain(self, bkd):
        """Test expected information gain is returned correctly."""
        objective = self._create_objective(bkd)

        eig = objective.expected_information_gain(self._design_weights)
        neg_eig = objective(self._design_weights)

        # EIG should be positive for meaningful designs
        # (though not guaranteed for all random data)
        expected_eig = -float(bkd.to_numpy(neg_eig)[0, 0])
        bkd.assert_allclose(
            bkd.asarray([eig]),
            bkd.asarray([expected_eig]),
            rtol=1e-10,
        )
