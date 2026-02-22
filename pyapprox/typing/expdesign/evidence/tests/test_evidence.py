"""
Tests for Evidence and LogEvidence classes.

Tests cover:
- Evidence computation correctness
- Log-evidence numerical stability
- Jacobian correctness via finite differences
- Effective sample size computation
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

from pyapprox.typing.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.typing.expdesign.evidence import Evidence, LogEvidence


class TestEvidence(Generic[Array], unittest.TestCase):
    """Base test class for Evidence computation."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        # Set up test data
        self._nobs = 3
        self._ninner = 10
        self._nouter = 5

        np.random.seed(42)
        self._noise_variances = self._bkd.asarray(
            np.array([0.1, 0.2, 0.15])
        )
        self._shapes_inner = self._bkd.asarray(
            np.random.randn(self._nobs, self._ninner)
        )
        self._observations = self._bkd.asarray(
            np.random.randn(self._nobs, self._nouter)
        )
        self._design_weights = self._bkd.asarray(
            np.random.uniform(0.5, 1.5, (self._nobs, 1))
        )

        # Create inner likelihood
        self._likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        self._likelihood.set_shapes(self._shapes_inner)
        self._likelihood.set_observations(self._observations)

    def _create_evidence(self, quad_weights=None):
        """Helper to create Evidence instance."""
        return Evidence(self._likelihood, quad_weights, self._bkd)

    def _create_log_evidence(self, quad_weights=None):
        """Helper to create LogEvidence instance."""
        return LogEvidence(self._likelihood, quad_weights, self._bkd)

    # --- Evidence Tests ---

    def test_evidence_manual_computation(self):
        """Test evidence against manual computation."""
        quad_weights = self._bkd.ones((self._ninner,)) / self._ninner
        evidence = self._create_evidence(quad_weights)

        # Compute evidence
        values = evidence(self._design_weights)

        # Manual computation
        loglike_matrix = self._likelihood.logpdf_matrix(self._design_weights)
        like_matrix = self._bkd.exp(loglike_matrix)
        expected = self._bkd.sum(
            quad_weights[:, None] * like_matrix, axis=0
        )

        self.assertTrue(
            self._bkd.allclose(values[0], expected, rtol=1e-10)
        )

    def test_evidence_jacobian_shape(self):
        """Test that evidence Jacobian has correct shape."""
        evidence = self._create_evidence()
        jac = evidence.jacobian(self._design_weights)
        self.assertEqual(jac.shape, (self._nouter, self._nobs))

    def test_evidence_jacobian_finite_diff(self):
        """Test evidence Jacobian against finite differences."""
        evidence = self._create_evidence()

        jac_analytical = evidence.jacobian(self._design_weights)

        # Finite difference
        eps = 1e-6
        jac_fd = self._bkd.zeros((self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = self._bkd.copy(self._design_weights)
            weights_minus = self._bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = evidence(weights_plus)
            val_minus = evidence(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)
        )

    def test_effective_sample_size(self):
        """Test ESS computation."""
        evidence = self._create_evidence()
        ess = evidence.effective_sample_size(self._design_weights)

        # ESS should be between 1 and ninner
        self.assertEqual(ess.shape, (self._nouter,))
        self.assertTrue(self._bkd.all_bool(ess >= 1.0))
        self.assertTrue(self._bkd.all_bool(ess <= self._ninner))

    # --- LogEvidence Tests ---

    def test_log_evidence_consistency(self):
        """Test that log(evidence) = log_evidence."""
        evidence = self._create_evidence()
        log_evidence = self._create_log_evidence()

        ev_values = evidence(self._design_weights)
        log_ev_values = log_evidence(self._design_weights)

        expected = self._bkd.log(ev_values)
        self.assertTrue(
            self._bkd.allclose(log_ev_values, expected, rtol=1e-10)
        )

    def test_log_evidence_jacobian_shape(self):
        """Test that log-evidence Jacobian has correct shape."""
        log_evidence = self._create_log_evidence()
        jac = log_evidence.jacobian(self._design_weights)
        self.assertEqual(jac.shape, (self._nouter, self._nobs))

    def test_log_evidence_jacobian_finite_diff(self):
        """Test log-evidence Jacobian against finite differences."""
        log_evidence = self._create_log_evidence()

        jac_analytical = log_evidence.jacobian(self._design_weights)

        # Finite difference
        eps = 1e-6
        jac_fd = self._bkd.zeros((self._nouter, self._nobs))
        for k in range(self._nobs):
            weights_plus = self._bkd.copy(self._design_weights)
            weights_minus = self._bkd.copy(self._design_weights)
            weights_plus[k, 0] = weights_plus[k, 0] + eps
            weights_minus[k, 0] = weights_minus[k, 0] - eps

            val_plus = log_evidence(weights_plus)
            val_minus = log_evidence(weights_minus)

            jac_fd[:, k] = (val_plus[0] - val_minus[0]) / (2 * eps)

        self.assertTrue(
            self._bkd.allclose(jac_analytical, jac_fd, rtol=1e-4, atol=1e-7)
        )

    def test_log_evidence_jacobian_chain_rule(self):
        """Test that log-evidence Jacobian satisfies chain rule."""
        evidence = self._create_evidence()
        log_evidence = self._create_log_evidence()

        ev_values = evidence(self._design_weights)
        ev_jac = evidence.jacobian(self._design_weights)
        log_ev_jac = log_evidence.jacobian(self._design_weights)

        # d/dw log(ev) = (1/ev) * d/dw ev
        expected_jac = ev_jac / ev_values.T

        self.assertTrue(
            self._bkd.allclose(log_ev_jac, expected_jac, rtol=1e-10)
        )

    def test_custom_quadrature_weights(self):
        """Test evidence with custom quadrature weights."""
        # Non-uniform weights
        weights = self._bkd.asarray(np.random.dirichlet(np.ones(self._ninner)))
        evidence = self._create_evidence(weights)
        log_evidence = self._create_log_evidence(weights)

        ev_values = evidence(self._design_weights)
        log_ev_values = log_evidence(self._design_weights)

        # Should still satisfy log relationship
        expected = self._bkd.log(ev_values)
        self.assertTrue(
            self._bkd.allclose(log_ev_values, expected, rtol=1e-10)
        )


    def test_evidence_jacobian_fused_matches_separate(self):
        """Test fused evidence jacobian matches separate jacobian + einsum."""
        quad_weights = self._bkd.ones((self._ninner,)) / self._ninner
        evidence = Evidence(self._likelihood, quad_weights, self._bkd)
        jac_fused = evidence.jacobian(self._design_weights)

        # Reference: separate jacobian_matrix + einsum via vectorized compute
        from pyapprox.typing.expdesign.likelihood.compute import (
            jacobian_matrix_vectorized,
            evidence_jacobian_vectorized,
        )
        self._likelihood.set_shapes(self._shapes_inner)
        self._likelihood.set_observations(self._observations)
        self._likelihood.set_latent_samples(
            self._bkd.zeros((self._nobs, self._nouter))
        )
        loglike = self._likelihood.logpdf_matrix(self._design_weights)
        like = self._bkd.exp(loglike)
        qwl = quad_weights[:, None] * like
        jac_3d = jacobian_matrix_vectorized(
            self._shapes_inner, self._observations,
            self._bkd.zeros((self._nobs, self._nouter)),
            self._noise_variances, self._design_weights, self._bkd,
        )
        jac_ref = evidence_jacobian_vectorized(jac_3d, qwl, self._bkd)

        self._bkd.assert_allclose(jac_fused, jac_ref, rtol=1e-10)

    def test_log_evidence_jacobian_fused_matches_separate(self):
        """Test fused log-evidence jacobian matches separate path."""
        quad_weights = self._bkd.ones((self._ninner,)) / self._ninner
        log_ev = LogEvidence(
            self._likelihood, quad_weights, self._bkd
        )
        jac_fused = log_ev.jacobian(self._design_weights)

        # Reference: create a second likelihood to get separate computation
        likelihood2 = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd,
        )
        likelihood2.set_shapes(self._shapes_inner)
        likelihood2.set_observations(self._observations)
        likelihood2.set_latent_samples(
            self._bkd.zeros((self._nobs, self._nouter))
        )
        log_ev2 = LogEvidence(likelihood2, quad_weights, self._bkd)
        jac_ref = log_ev2.jacobian(self._design_weights)

        self._bkd.assert_allclose(jac_fused, jac_ref, rtol=1e-10)

    def test_evidence_accessors(self):
        """Test bkd(), ninner(), nouter() accessors."""
        evidence = self._create_evidence()
        self.assertIsNotNone(evidence.bkd())
        self.assertEqual(evidence.ninner(), self._ninner)
        self.assertEqual(evidence.nouter(), self._nouter)

    def test_log_evidence_accessors(self):
        """Test bkd(), ninner(), nouter() accessors on LogEvidence."""
        log_evidence = self._create_log_evidence()
        self.assertIsNotNone(log_evidence.bkd())
        self.assertEqual(log_evidence.ninner(), self._ninner)
        self.assertEqual(log_evidence.nouter(), self._nouter)


class TestEvidenceNumpy(TestEvidence[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestEvidenceTorch(TestEvidence[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
