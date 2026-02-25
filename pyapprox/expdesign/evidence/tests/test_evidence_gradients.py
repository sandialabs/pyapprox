"""
Standalone tests for Evidence and LogEvidence gradients.

PERMANENT - no legacy imports.

Tests verify correctness using:
1. DerivativeChecker for Jacobian verification
2. Consistency between Evidence and LogEvidence
3. Shape and positivity checks
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.expdesign.evidence import Evidence, LogEvidence
from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestEvidenceGradientsStandalone(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Standalone tests for Evidence and LogEvidence gradients."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

        # Default test dimensions
        self._nobs = 3
        self._ninner = 30
        self._nouter = 20

    def _create_likelihood_and_evidence(
        self, nobs: int, ninner: int, nouter: int
    ) -> tuple:
        """Create likelihood, Evidence, and LogEvidence with test data.

        NOTE: We do NOT set latent_samples on the likelihood. This is intentional.
        When latent_samples is set, the Jacobian includes a reparameterization term
        that assumes observations change with weights (obs = shape + noise(weights)).
        For standalone Evidence tests, observations are FIXED, so the analytical
        Jacobian should NOT include the reparameterization term.

        This matches the legacy test approach (test_bayesoed.py lines 153-156):
        "The reference gradients DO NOT include the impact of the reparameterization
        trick to compute the observational data. Instead they assume the data is
        not a function of the weights."
        """
        np.random.seed(42)
        bkd = self._bkd

        # Create noise variances (heteroscedastic)
        noise_variances = bkd.asarray(np.random.uniform(0.1, 0.5, nobs))

        # Create inner loop likelihood
        inner_loglike = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)

        # Create shapes and observations (fixed, not dependent on weights)
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        observations = bkd.asarray(np.random.randn(nobs, nouter))

        # Set up likelihood state
        # NOTE: Do NOT set latent_samples - see docstring above
        inner_loglike.set_shapes(inner_shapes)
        inner_loglike.set_observations(observations)

        # Create Evidence and LogEvidence
        quad_weights = None  # Uniform MC weights
        evidence = Evidence(inner_loglike, quad_weights, bkd)
        log_evidence = LogEvidence(inner_loglike, quad_weights, bkd)

        return inner_loglike, evidence, log_evidence, nobs

    def _create_evidence_checker_function(self, evidence_obj: Evidence[Array]):
        """
        Wrap Evidence for DerivativeChecker.

        DerivativeChecker expects: f: (nvars, 1) -> (nqoi, 1)
        Evidence provides: f: (nobs, 1) -> (1, nouter)

        We create a scalar objective by summing over outer samples:
            scalar_f(w) = sum_j evidence(w)[0, j]

        The Jacobian is then:
            d/dw scalar_f = sum_j d/dw evidence[j] = sum over rows of jacobian
        """

        def value_fun(samples: Array) -> Array:
            # samples: (nvars, nsamples) where nvars = nobs
            nsamples = samples.shape[1]
            results = []
            for ii in range(nsamples):
                w = samples[:, ii : ii + 1]  # (nobs, 1)
                evidence_vals = evidence_obj(w)  # (1, nouter)
                scalar_sum = self._bkd.sum(evidence_vals)  # sum over outer samples
                results.append(scalar_sum)
            return self._bkd.reshape(self._bkd.stack(results), (1, nsamples))

        def jacobian_fun(sample: Array) -> Array:
            # sample: (nobs, 1)
            jac = evidence_obj.jacobian(sample)  # (nouter, nobs)
            # Sum over outer samples (axis 0) to get scalar objective gradient
            return self._bkd.sum(jac, axis=0, keepdims=True)  # (1, nobs)

        nobs = evidence_obj._loglike.nobs()
        return FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=nobs,
            fun=value_fun,
            jacobian=jacobian_fun,
            bkd=self._bkd,
        )

    # ==========================================================================
    # Shape tests
    # ==========================================================================

    def test_evidence_shape(self) -> None:
        """Test Evidence output shape is (1, nouter)."""
        _, evidence, _, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((nobs, 1)) / nobs
        result = evidence(weights)
        self.assertEqual(result.shape, (1, self._nouter))

    def test_log_evidence_shape(self) -> None:
        """Test LogEvidence output shape is (1, nouter)."""
        _, _, log_evidence, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((nobs, 1)) / nobs
        result = log_evidence(weights)
        self.assertEqual(result.shape, (1, self._nouter))

    def test_evidence_jacobian_shape(self) -> None:
        """Test Evidence Jacobian shape is (nouter, nobs)."""
        _, evidence, _, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((nobs, 1)) / nobs
        jac = evidence.jacobian(weights)
        self.assertEqual(jac.shape, (self._nouter, nobs))

    def test_log_evidence_jacobian_shape(self) -> None:
        """Test LogEvidence Jacobian shape is (nouter, nobs)."""
        _, _, log_evidence, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((nobs, 1)) / nobs
        jac = log_evidence.jacobian(weights)
        self.assertEqual(jac.shape, (self._nouter, nobs))

    # ==========================================================================
    # Gradient verification tests
    # ==========================================================================

    @parametrize(
        "nobs,ninner,nouter",
        [
            (3, 30, 20),  # Small case
            (5, 50, 40),  # Medium case
            (2, 100, 50),  # Few obs, many samples
        ],
    )
    def test_evidence_jacobian_derivative_checker(
        self, nobs: int, ninner: int, nouter: int
    ) -> None:
        """Test Evidence.jacobian using DerivativeChecker."""
        _, evidence, _, _ = self._create_likelihood_and_evidence(nobs, ninner, nouter)
        wrapped = self._create_evidence_checker_function(evidence)
        checker = DerivativeChecker(wrapped)

        np.random.seed(123)
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))

        # Use smaller step sizes to avoid numerical issues with large perturbations
        # Default is logspace(-13, 0, 14) which includes step=1.0 that's too large
        fd_eps = self._bkd.flip(self._bkd.logspace(-12, -1, 12))
        errors = checker.check_derivatives(weights, fd_eps=fd_eps)

        ratio = checker.error_ratio(errors[0])
        self.assertLessEqual(
            float(self._bkd.to_numpy(ratio)),
            1e-5,
            f"Evidence Jacobian check failed: ratio={ratio}",
        )

    @parametrize(
        "nobs,ninner,nouter",
        [
            (3, 30, 20),  # Small case
            (5, 50, 40),  # Medium case
            (2, 100, 50),  # Few obs, many samples
        ],
    )
    def test_log_evidence_jacobian_derivative_checker(
        self, nobs: int, ninner: int, nouter: int
    ) -> None:
        """Test LogEvidence.jacobian using DerivativeChecker."""
        _, _, log_evidence, _ = self._create_likelihood_and_evidence(
            nobs, ninner, nouter
        )
        wrapped = self._create_evidence_checker_function(log_evidence)
        checker = DerivativeChecker(wrapped)

        np.random.seed(123)
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))

        # Use smaller step sizes to avoid numerical issues with large perturbations
        fd_eps = self._bkd.flip(self._bkd.logspace(-12, -1, 12))
        errors = checker.check_derivatives(weights, fd_eps=fd_eps)

        ratio = checker.error_ratio(errors[0])
        self.assertLessEqual(
            float(self._bkd.to_numpy(ratio)),
            1e-5,
            f"LogEvidence Jacobian check failed: ratio={ratio}",
        )

    # ==========================================================================
    # Value property tests
    # ==========================================================================

    def test_evidence_positive(self) -> None:
        """Test all evidence values are positive."""
        _, evidence, _, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((nobs, 1)) / nobs
        result = evidence(weights)

        result_np = self._bkd.to_numpy(result)
        self.assertTrue(np.all(result_np > 0), "Evidence values must be positive")

    def test_log_evidence_finite(self) -> None:
        """Test all log-evidence values are finite."""
        _, _, log_evidence, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((nobs, 1)) / nobs
        result = log_evidence(weights)

        result_np = self._bkd.to_numpy(result)
        self.assertTrue(
            np.all(np.isfinite(result_np)), "Log-evidence values must be finite"
        )

    def test_log_evidence_equals_log_of_evidence(self) -> None:
        """Test log(Evidence()) approximately equals LogEvidence()."""
        _, evidence, log_evidence, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((nobs, 1)) / nobs

        evidence_vals = evidence(weights)
        log_evidence_vals = log_evidence(weights)

        # log(Evidence) should equal LogEvidence
        log_of_evidence = self._bkd.log(evidence_vals)

        self._bkd.assert_allclose(log_of_evidence, log_evidence_vals, rtol=1e-10)

    def test_log_evidence_numerical_stability(self) -> None:
        """Test LogEvidence handles very small likelihood values."""
        bkd = self._bkd
        nobs, ninner, nouter = 2, 20, 10

        # Create likelihood with large noise (small likelihoods)
        noise_variances = bkd.asarray(np.array([10.0, 10.0]))
        inner_loglike = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)

        # Create shapes and observations that are far apart (low likelihood)
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner) * 10)
        observations = bkd.asarray(np.random.randn(nobs, nouter) * 10)

        inner_loglike.set_shapes(inner_shapes)
        inner_loglike.set_observations(observations)
        # NOTE: Do NOT set latent_samples (see _create_likelihood_and_evidence
        # docstring)

        log_evidence = LogEvidence(inner_loglike, None, bkd)
        weights = bkd.ones((nobs, 1)) / nobs

        result = log_evidence(weights)
        result_np = bkd.to_numpy(result)

        # Should be finite (not -inf or nan)
        self.assertTrue(
            np.all(np.isfinite(result_np)),
            "LogEvidence must be numerically stable for small likelihoods",
        )

    # ==========================================================================
    # Consistency and auxiliary tests
    # ==========================================================================

    def test_evidence_jacobian_finite(self) -> None:
        """Test Evidence Jacobian values are finite."""
        _, evidence, _, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        np.random.seed(456)
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))
        jac = evidence.jacobian(weights)

        jac_np = self._bkd.to_numpy(jac)
        self.assertTrue(np.all(np.isfinite(jac_np)))

    def test_log_evidence_jacobian_finite(self) -> None:
        """Test LogEvidence Jacobian values are finite."""
        _, _, log_evidence, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        np.random.seed(456)
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))
        jac = log_evidence.jacobian(weights)

        jac_np = self._bkd.to_numpy(jac)
        self.assertTrue(np.all(np.isfinite(jac_np)))

    def test_evidence_jacobian_changes_with_weights(self) -> None:
        """Test Evidence Jacobian changes with different weights."""
        _, evidence, _, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )

        weights1 = self._bkd.ones((nobs, 1)) * 0.5
        weights2 = self._bkd.ones((nobs, 1)) * 2.0

        jac1 = evidence.jacobian(weights1)
        jac2 = evidence.jacobian(weights2)

        jac1_np = self._bkd.to_numpy(jac1)
        jac2_np = self._bkd.to_numpy(jac2)

        self.assertFalse(
            np.allclose(jac1_np, jac2_np),
            "Jacobians should differ at different weights",
        )

    def test_evidence_deterministic(self) -> None:
        """Test Evidence evaluation is deterministic."""
        _, evidence, _, nobs = self._create_likelihood_and_evidence(
            self._nobs, self._ninner, self._nouter
        )
        weights = self._bkd.ones((nobs, 1)) / nobs

        val1 = evidence(weights)
        val2 = evidence(weights)
        jac1 = evidence.jacobian(weights)
        jac2 = evidence.jacobian(weights)

        self._bkd.assert_allclose(val1, val2, rtol=1e-12)
        self._bkd.assert_allclose(jac1, jac2, rtol=1e-12)


class TestEvidenceGradientsStandaloneNumpy(
    TestEvidenceGradientsStandalone[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestEvidenceGradientsStandaloneTorch(
    TestEvidenceGradientsStandalone[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
