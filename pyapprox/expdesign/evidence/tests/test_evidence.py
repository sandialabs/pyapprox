"""
Tests for Evidence and LogEvidence classes.

Tests cover:
- Evidence computation correctness
- Log-evidence numerical stability
- Jacobian correctness via finite differences
- Effective sample size computation
"""

import numpy as np
import pytest

from pyapprox.expdesign.evidence import Evidence, LogEvidence
from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood


class TestEvidence:
    """Base test class for Evidence computation."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        # Set up test data
        self._nobs = 3
        self._ninner = 10
        self._nouter = 5

        self._noise_variances = bkd.asarray(np.array([0.1, 0.2, 0.15]))
        self._shapes_inner = bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._observations = bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._design_weights = bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

        # Create inner likelihood
        self._likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, bkd
        )
        self._likelihood.set_shapes(self._shapes_inner)
        self._likelihood.set_observations(self._observations)

    def _create_evidence(self, bkd, quad_weights=None):
        """Helper to create Evidence instance."""
        return Evidence(self._likelihood, quad_weights, bkd)

    def _create_log_evidence(self, bkd, quad_weights=None):
        """Helper to create LogEvidence instance."""
        return LogEvidence(self._likelihood, quad_weights, bkd)

    # --- Evidence Tests ---

    def test_evidence_manual_computation(self, bkd):
        """Test evidence against manual computation."""
        quad_weights = bkd.ones((self._ninner,)) / self._ninner
        evidence = self._create_evidence(bkd, quad_weights)

        # Compute evidence
        values = evidence(self._design_weights)

        # Manual computation
        loglike_matrix = self._likelihood.logpdf_matrix(self._design_weights)
        like_matrix = bkd.exp(loglike_matrix)
        expected = bkd.sum(quad_weights[:, None] * like_matrix, axis=0)

        assert bkd.allclose(values[0], expected, rtol=1e-10)

    def test_evidence_jacobian_shape(self, bkd):
        """Test that evidence Jacobian has correct shape."""
        evidence = self._create_evidence(bkd)
        jac = evidence.jacobian(self._design_weights)
        assert jac.shape == (self._nouter, self._nobs)

    def test_evidence_jacobian_finite_diff(self, bkd):
        """Test evidence Jacobian against finite differences."""
        evidence = self._create_evidence(bkd)

        jac_analytical = evidence.jacobian(self._design_weights)

        # Finite difference
        eps = 1e-6
        jac_fd = bkd.zeros((self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = bkd.copy(self._design_weights)
            weights_minus = bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = evidence(weights_plus)
            val_minus = evidence(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    def test_effective_sample_size(self, bkd):
        """Test ESS computation."""
        evidence = self._create_evidence(bkd)
        ess = evidence.effective_sample_size(self._design_weights)

        # ESS should be between 1 and ninner
        assert ess.shape == (self._nouter,)
        assert bkd.all_bool(ess >= 1.0)
        assert bkd.all_bool(ess <= self._ninner)

    # --- LogEvidence Tests ---

    def test_log_evidence_consistency(self, bkd):
        """Test that log(evidence) = log_evidence."""
        evidence = self._create_evidence(bkd)
        log_evidence = self._create_log_evidence(bkd)

        ev_values = evidence(self._design_weights)
        log_ev_values = log_evidence(self._design_weights)

        expected = bkd.log(ev_values)
        assert bkd.allclose(log_ev_values, expected, rtol=1e-10)

    def test_log_evidence_jacobian_shape(self, bkd):
        """Test that log-evidence Jacobian has correct shape."""
        log_evidence = self._create_log_evidence(bkd)
        jac = log_evidence.jacobian(self._design_weights)
        assert jac.shape == (self._nouter, self._nobs)

    def test_log_evidence_jacobian_finite_diff(self, bkd):
        """Test log-evidence Jacobian against finite differences."""
        log_evidence = self._create_log_evidence(bkd)

        jac_analytical = log_evidence.jacobian(self._design_weights)

        # Finite difference
        eps = 1e-6
        jac_fd = bkd.zeros((self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = bkd.copy(self._design_weights)
            weights_minus = bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = log_evidence(weights_plus)
            val_minus = log_evidence(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        assert bkd.allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)

    def test_log_evidence_jacobian_chain_rule(self, bkd):
        """Test that log-evidence Jacobian satisfies chain rule."""
        evidence = self._create_evidence(bkd)
        log_evidence = self._create_log_evidence(bkd)

        ev_values = evidence(self._design_weights)
        ev_jac = evidence.jacobian(self._design_weights)
        log_ev_jac = log_evidence.jacobian(self._design_weights)

        # d/dw log(ev) = (1/ev) * d/dw ev
        expected_jac = ev_jac / ev_values.T

        assert bkd.allclose(log_ev_jac, expected_jac, rtol=1e-10)

    def test_custom_quadrature_weights(self, bkd):
        """Test evidence with custom quadrature weights."""
        # Non-uniform weights
        weights = bkd.asarray(np.random.dirichlet(np.ones(self._ninner)))
        evidence = self._create_evidence(bkd, weights)
        log_evidence = self._create_log_evidence(bkd, weights)

        ev_values = evidence(self._design_weights)
        log_ev_values = log_evidence(self._design_weights)

        # Should still satisfy log relationship
        expected = bkd.log(ev_values)
        assert bkd.allclose(log_ev_values, expected, rtol=1e-10)

    def test_evidence_jacobian_fused_matches_separate(self, bkd):
        """Test fused evidence jacobian matches separate jacobian + einsum."""
        quad_weights = bkd.ones((self._ninner,)) / self._ninner
        evidence = Evidence(self._likelihood, quad_weights, bkd)
        jac_fused = evidence.jacobian(self._design_weights)

        # Reference: separate jacobian_matrix + einsum via vectorized compute
        from pyapprox.expdesign.likelihood.compute import (
            evidence_jacobian_vectorized,
            jacobian_matrix_vectorized,
        )

        self._likelihood.set_shapes(self._shapes_inner)
        self._likelihood.set_observations(self._observations)
        self._likelihood.set_latent_samples(bkd.zeros((self._nobs, self._nouter)))
        loglike = self._likelihood.logpdf_matrix(self._design_weights)
        like = bkd.exp(loglike)
        qwl = quad_weights[:, None] * like
        jac_3d = jacobian_matrix_vectorized(
            self._shapes_inner,
            self._observations,
            bkd.zeros((self._nobs, self._nouter)),
            self._noise_variances,
            self._design_weights,
            bkd,
        )
        jac_ref = evidence_jacobian_vectorized(jac_3d, qwl, bkd)

        bkd.assert_allclose(jac_fused, jac_ref, rtol=1e-10)

    def test_log_evidence_jacobian_fused_matches_separate(self, bkd):
        """Test fused log-evidence jacobian matches separate path."""
        quad_weights = bkd.ones((self._ninner,)) / self._ninner
        log_ev = LogEvidence(self._likelihood, quad_weights, bkd)
        jac_fused = log_ev.jacobian(self._design_weights)

        # Reference: create a second likelihood to get separate computation
        likelihood2 = GaussianOEDInnerLoopLikelihood(
            self._noise_variances,
            bkd,
        )
        likelihood2.set_shapes(self._shapes_inner)
        likelihood2.set_observations(self._observations)
        likelihood2.set_latent_samples(bkd.zeros((self._nobs, self._nouter)))
        log_ev2 = LogEvidence(likelihood2, quad_weights, bkd)
        jac_ref = log_ev2.jacobian(self._design_weights)

        bkd.assert_allclose(jac_fused, jac_ref, rtol=1e-10)

    def test_evidence_accessors(self, bkd):
        """Test bkd(), ninner(), nouter() accessors."""
        evidence = self._create_evidence(bkd)
        assert evidence.bkd() is not None
        assert evidence.ninner() == self._ninner
        assert evidence.nouter() == self._nouter

    def test_log_evidence_accessors(self, bkd):
        """Test bkd(), ninner(), nouter() accessors on LogEvidence."""
        log_evidence = self._create_log_evidence(bkd)
        assert log_evidence.bkd() is not None
        assert log_evidence.ninner() == self._ninner
        assert log_evidence.nouter() == self._nouter
