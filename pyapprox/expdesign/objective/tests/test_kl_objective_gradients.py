"""
Standalone tests for KLOEDObjective gradients.

PERMANENT - no legacy imports.

Tests verify correctness using:
1. DerivativeChecker for Jacobian verification
2. Shape and value property checks
3. Consistency tests (EIG positive, deterministic, etc.)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)

from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood


class TestKLOEDObjectiveGradientsStandalone(
    Generic[Array], ParametrizedTestCase, unittest.TestCase
):
    """Standalone tests for KLOEDObjective gradients."""

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

    def _create_kl_objective(
        self,
        nobs: int,
        ninner: int,
        nouter: int,
        outer_quad_weights: Array = None,
        inner_quad_weights: Array = None,
    ) -> KLOEDObjective[Array]:
        """Create KLOEDObjective with test data.

        NOTE: KLOEDObjective internally uses the reparameterization trick:
        observations are generated as obs = shapes + sqrt(var/weights) * latent.
        This means observations DO depend on weights, and the Jacobian must
        account for this dependency through the reparameterization term.
        """
        np.random.seed(42)
        bkd = self._bkd

        # Create noise variances (heteroscedastic)
        noise_variances = bkd.asarray(np.random.uniform(0.1, 0.5, nobs))

        # Create inner loop likelihood
        inner_loglike = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)

        # Create shapes for inner and outer loops
        inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
        outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))

        # Create latent samples for reparameterization
        latent_samples = bkd.asarray(np.random.randn(nobs, nouter))

        return KLOEDObjective(
            inner_loglike,
            outer_shapes,
            latent_samples,
            inner_shapes,
            outer_quad_weights,
            inner_quad_weights,
            bkd,
        )

    def _create_objective_checker_function(self, obj: KLOEDObjective[Array]):
        """Wrap KLOEDObjective for DerivativeChecker.

        KLOEDObjective already has the correct interface:
        - __call__(weights): (nobs, 1) -> (1, 1)
        - jacobian(weights): (nobs, 1) -> (1, nobs)

        DerivativeChecker expects:
        - __call__(samples): (nvars, nsamples) -> (nqoi, nsamples)
        - jacobian(sample): (nvars, 1) -> (nqoi, nvars)
        """

        def value_fun(samples: Array) -> Array:
            # samples: (nvars, nsamples) where nvars = nobs
            nsamples = samples.shape[1]
            results = []
            for ii in range(nsamples):
                w = samples[:, ii : ii + 1]  # (nobs, 1)
                val = obj(w)  # (1, 1)
                results.append(val[0, 0])
            return self._bkd.reshape(self._bkd.stack(results), (1, nsamples))

        def jacobian_fun(sample: Array) -> Array:
            # sample: (nobs, 1)
            jac = obj.jacobian(sample)  # (1, nobs)
            return jac  # Already (nqoi=1, nvars=nobs)

        return FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=obj.nvars(),
            fun=value_fun,
            jacobian=jacobian_fun,
            bkd=self._bkd,
        )

    # ==========================================================================
    # Shape tests
    # ==========================================================================

    def test_objective_shape(self) -> None:
        """Test KLOEDObjective output shape is (1, 1)."""
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        result = obj(weights)
        self.assertEqual(result.shape, (1, 1))

    def test_jacobian_shape(self) -> None:
        """Test KLOEDObjective Jacobian shape is (1, nobs)."""
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs
        jac = obj.jacobian(weights)
        self.assertEqual(jac.shape, (1, self._nobs))

    def test_nvars_nqoi(self) -> None:
        """Test nvars and nqoi accessors."""
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)
        self.assertEqual(obj.nvars(), self._nobs)
        self.assertEqual(obj.nqoi(), 1)
        self.assertEqual(obj.nobs(), self._nobs)
        self.assertEqual(obj.ninner(), self._ninner)
        self.assertEqual(obj.nouter(), self._nouter)

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
    def test_jacobian_derivative_checker(
        self, nobs: int, ninner: int, nouter: int
    ) -> None:
        """Test KLOEDObjective.jacobian using DerivativeChecker."""
        obj = self._create_kl_objective(nobs, ninner, nouter)
        wrapped = self._create_objective_checker_function(obj)
        checker = DerivativeChecker(wrapped)

        np.random.seed(123)
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))

        # Use smaller step sizes to avoid numerical issues
        fd_eps = self._bkd.flip(self._bkd.logspace(-12, -1, 12))
        errors = checker.check_derivatives(weights, fd_eps=fd_eps)

        ratio = checker.error_ratio(errors[0])
        self.assertLessEqual(
            float(self._bkd.to_numpy(ratio)),
            1e-5,
            f"KLOEDObjective Jacobian check failed: ratio={ratio}",
        )

    def test_jacobian_derivative_checker_with_quad_weights(self) -> None:
        """Test Jacobian when using non-uniform quadrature weights."""
        np.random.seed(42)
        nobs, ninner, nouter = 3, 30, 20

        # Create custom quadrature weights (not uniform)
        outer_weights = self._bkd.asarray(np.random.dirichlet(np.ones(nouter)))
        inner_weights = self._bkd.asarray(np.random.dirichlet(np.ones(ninner)))

        obj = self._create_kl_objective(
            nobs, ninner, nouter,
            outer_quad_weights=outer_weights,
            inner_quad_weights=inner_weights,
        )

        wrapped = self._create_objective_checker_function(obj)
        checker = DerivativeChecker(wrapped)

        np.random.seed(123)
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))

        # Use smaller step sizes to avoid numerical issues
        fd_eps = self._bkd.flip(self._bkd.logspace(-12, -1, 12))
        errors = checker.check_derivatives(weights, fd_eps=fd_eps)

        ratio = checker.error_ratio(errors[0])
        self.assertLessEqual(
            float(self._bkd.to_numpy(ratio)),
            1e-5,
            f"KLOEDObjective Jacobian with quad weights check failed: ratio={ratio}",
        )

    # ==========================================================================
    # Value property tests
    # ==========================================================================

    def test_objective_returns_negative_eig(self) -> None:
        """Test objective() == -expected_information_gain()."""
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        obj_val = obj(weights)
        eig = obj.expected_information_gain(weights)

        # objective returns -EIG for minimization
        self._bkd.assert_allclose(
            obj_val,
            self._bkd.asarray([[-eig]]),
            rtol=1e-12,
        )

    def test_eig_positive(self) -> None:
        """Test expected_information_gain() > 0.

        EIG should be positive (information gain is non-negative).
        """
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        eig = obj.expected_information_gain(weights)
        self.assertGreater(eig, 0, "Expected information gain must be positive")

    def test_jacobian_finite(self) -> None:
        """Test KLOEDObjective Jacobian values are finite."""
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)
        np.random.seed(456)
        weights = self._bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))
        jac = obj.jacobian(weights)

        jac_np = self._bkd.to_numpy(jac)
        self.assertTrue(np.all(np.isfinite(jac_np)))

    def test_evaluate_alias(self) -> None:
        """Test evaluate() is alias for __call__()."""
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        result_call = obj(weights)
        result_eval = obj.evaluate(weights)

        self._bkd.assert_allclose(result_call, result_eval, rtol=1e-12)

    def test_deterministic(self) -> None:
        """Test KLOEDObjective evaluation is deterministic."""
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)
        weights = self._bkd.ones((self._nobs, 1)) / self._nobs

        val1 = obj(weights)
        val2 = obj(weights)
        jac1 = obj.jacobian(weights)
        jac2 = obj.jacobian(weights)

        self._bkd.assert_allclose(val1, val2, rtol=1e-12)
        self._bkd.assert_allclose(jac1, jac2, rtol=1e-12)

    # ==========================================================================
    # Consistency and auxiliary tests
    # ==========================================================================

    def test_jacobian_changes_with_weights(self) -> None:
        """Test KLOEDObjective Jacobian changes with different weights."""
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)

        weights1 = self._bkd.ones((self._nobs, 1)) * 0.5
        weights2 = self._bkd.ones((self._nobs, 1)) * 2.0

        jac1 = obj.jacobian(weights1)
        jac2 = obj.jacobian(weights2)

        jac1_np = self._bkd.to_numpy(jac1)
        jac2_np = self._bkd.to_numpy(jac2)

        self.assertFalse(
            np.allclose(jac1_np, jac2_np),
            "Jacobians should differ at different weights",
        )

    def test_eig_increases_with_more_observations(self) -> None:
        """Test EIG generally increases when observation weights increase.

        This is a weak consistency check - more informative observations
        should generally yield higher expected information gain.
        """
        obj = self._create_kl_objective(self._nobs, self._ninner, self._nouter)

        # Smaller weights = less informative
        weights_low = self._bkd.ones((self._nobs, 1)) * 0.1
        # Larger weights = more informative
        weights_high = self._bkd.ones((self._nobs, 1)) * 2.0

        eig_low = obj.expected_information_gain(weights_low)
        eig_high = obj.expected_information_gain(weights_high)

        # Higher weights should generally give higher EIG
        # (more observations = more information)
        self.assertGreater(
            eig_high, eig_low,
            f"EIG with high weights ({eig_high}) should exceed EIG with "
            f"low weights ({eig_low})",
        )


class TestKLOEDObjectiveGradientsStandaloneNumpy(
    TestKLOEDObjectiveGradientsStandalone[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestKLOEDObjectiveGradientsStandaloneTorch(
    TestKLOEDObjectiveGradientsStandalone[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
