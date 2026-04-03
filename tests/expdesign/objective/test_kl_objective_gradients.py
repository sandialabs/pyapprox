"""
Standalone tests for KLOEDObjective gradients.

PERMANENT - no legacy imports.

Tests verify correctness using:
1. DerivativeChecker for Jacobian verification
2. Shape and value property checks
3. Consistency tests (EIG positive, deterministic, etc.)
"""

import numpy as np
import pytest

from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from tests._helpers.markers import slow_test


class TestKLOEDObjectiveGradientsStandalone:
    """Standalone tests for KLOEDObjective gradients."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        np.random.seed(42)

        # Default test dimensions
        self._nobs = 3
        self._ninner = 30
        self._nouter = 20

    def _create_kl_objective(
        self,
        bkd,
        nobs,
        ninner,
        nouter,
        outer_quad_weights=None,
        inner_quad_weights=None,
    ):
        """Create KLOEDObjective with test data.

        NOTE: KLOEDObjective internally uses the reparameterization trick:
        observations are generated as obs = shapes + sqrt(var/weights) * latent.
        This means observations DO depend on weights, and the Jacobian must
        account for this dependency through the reparameterization term.
        """
        np.random.seed(42)

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

    def _create_objective_checker_function(self, bkd, obj):
        """Wrap KLOEDObjective for DerivativeChecker.

        KLOEDObjective already has the correct interface:
        - __call__(weights): (nobs, 1) -> (1, 1)
        - jacobian(weights): (nobs, 1) -> (1, nobs)

        DerivativeChecker expects:
        - __call__(samples): (nvars, nsamples) -> (nqoi, nsamples)
        - jacobian(sample): (nvars, 1) -> (nqoi, nvars)
        """

        def value_fun(samples):
            # samples: (nvars, nsamples) where nvars = nobs
            nsamples = samples.shape[1]
            results = []
            for ii in range(nsamples):
                w = samples[:, ii : ii + 1]  # (nobs, 1)
                val = obj(w)  # (1, 1)
                results.append(val[0, 0])
            return bkd.reshape(bkd.stack(results), (1, nsamples))

        def jacobian_fun(sample):
            # sample: (nobs, 1)
            jac = obj.jacobian(sample)  # (1, nobs)
            return jac  # Already (nqoi=1, nvars=nobs)

        return FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=obj.nvars(),
            fun=value_fun,
            jacobian=jacobian_fun,
            bkd=bkd,
        )

    # ==========================================================================
    # Shape tests
    # ==========================================================================

    def test_objective_shape(self, bkd):
        """Test KLOEDObjective output shape is (1, 1)."""
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        result = obj(weights)
        assert result.shape == (1, 1)

    def test_jacobian_shape(self, bkd):
        """Test KLOEDObjective Jacobian shape is (1, nobs)."""
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)
        weights = bkd.ones((self._nobs, 1)) / self._nobs
        jac = obj.jacobian(weights)
        assert jac.shape == (1, self._nobs)

    def test_nvars_nqoi(self, bkd):
        """Test nvars and nqoi accessors."""
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)
        assert obj.nvars() == self._nobs
        assert obj.nqoi() == 1
        assert obj.nobs() == self._nobs
        assert obj.ninner() == self._ninner
        assert obj.nouter() == self._nouter

    # ==========================================================================
    # Gradient verification tests
    # ==========================================================================

    @pytest.mark.parametrize(
        "nobs,ninner,nouter",
        [
            (3, 30, 20),  # Small case
            (5, 50, 40),  # Medium case
            (2, 100, 50),  # Few obs, many samples
        ],
    )
    def test_jacobian_derivative_checker(
        self, bkd, nobs, ninner, nouter
    ):
        """Test KLOEDObjective.jacobian using DerivativeChecker."""
        obj = self._create_kl_objective(bkd, nobs, ninner, nouter)
        wrapped = self._create_objective_checker_function(bkd, obj)
        checker = DerivativeChecker(wrapped)

        np.random.seed(123)
        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))

        # Use smaller step sizes to avoid numerical issues
        fd_eps = bkd.flip(bkd.logspace(-12, -1, 12))
        errors = checker.check_derivatives(weights, fd_eps=fd_eps)

        ratio = checker.error_ratio(errors[0])
        assert float(bkd.to_numpy(ratio)) <= 1e-5, (
            f"KLOEDObjective Jacobian check failed: ratio={ratio}"
        )

    def test_jacobian_derivative_checker_with_quad_weights(self, bkd):
        """Test Jacobian when using non-uniform quadrature weights."""
        np.random.seed(42)
        nobs, ninner, nouter = 3, 30, 20

        # Create custom quadrature weights (not uniform)
        outer_weights = bkd.asarray(np.random.dirichlet(np.ones(nouter)))
        inner_weights = bkd.asarray(np.random.dirichlet(np.ones(ninner)))

        obj = self._create_kl_objective(
            bkd,
            nobs,
            ninner,
            nouter,
            outer_quad_weights=outer_weights,
            inner_quad_weights=inner_weights,
        )

        wrapped = self._create_objective_checker_function(bkd, obj)
        checker = DerivativeChecker(wrapped)

        np.random.seed(123)
        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (nobs, 1)))

        # Use smaller step sizes to avoid numerical issues
        fd_eps = bkd.flip(bkd.logspace(-12, -1, 12))
        errors = checker.check_derivatives(weights, fd_eps=fd_eps)

        ratio = checker.error_ratio(errors[0])
        assert float(bkd.to_numpy(ratio)) <= 1e-5, (
            f"KLOEDObjective Jacobian with quad weights check failed: ratio={ratio}"
        )

    # ==========================================================================
    # Value property tests
    # ==========================================================================

    def test_objective_returns_negative_eig(self, bkd):
        """Test objective() == -expected_information_gain()."""
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        obj_val = obj(weights)
        eig = obj.expected_information_gain(weights)

        # objective returns -EIG for minimization
        bkd.assert_allclose(
            obj_val,
            bkd.asarray([[-eig]]),
            rtol=1e-12,
        )

    def test_eig_positive(self, bkd):
        """Test expected_information_gain() > 0.

        EIG should be positive (information gain is non-negative).
        """
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        eig = obj.expected_information_gain(weights)
        assert eig > 0, "Expected information gain must be positive"

    def test_jacobian_finite(self, bkd):
        """Test KLOEDObjective Jacobian values are finite."""
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)
        np.random.seed(456)
        weights = bkd.asarray(np.random.uniform(0.5, 1.5, (self._nobs, 1)))
        jac = obj.jacobian(weights)

        jac_np = bkd.to_numpy(jac)
        assert np.all(np.isfinite(jac_np))

    def test_evaluate_alias(self, bkd):
        """Test evaluate() is alias for __call__()."""
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        result_call = obj(weights)
        result_eval = obj.evaluate(weights)

        bkd.assert_allclose(result_call, result_eval, rtol=1e-12)

    @slow_test
    def test_deterministic(self, bkd):
        """Test KLOEDObjective evaluation is deterministic."""
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)
        weights = bkd.ones((self._nobs, 1)) / self._nobs

        val1 = obj(weights)
        val2 = obj(weights)
        jac1 = obj.jacobian(weights)
        jac2 = obj.jacobian(weights)

        bkd.assert_allclose(val1, val2, rtol=1e-12)
        bkd.assert_allclose(jac1, jac2, rtol=1e-12)

    # ==========================================================================
    # Consistency and auxiliary tests
    # ==========================================================================

    def test_jacobian_changes_with_weights(self, bkd):
        """Test KLOEDObjective Jacobian changes with different weights."""
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)

        weights1 = bkd.ones((self._nobs, 1)) * 0.5
        weights2 = bkd.ones((self._nobs, 1)) * 2.0

        jac1 = obj.jacobian(weights1)
        jac2 = obj.jacobian(weights2)

        jac1_np = bkd.to_numpy(jac1)
        jac2_np = bkd.to_numpy(jac2)

        assert not np.allclose(jac1_np, jac2_np), (
            "Jacobians should differ at different weights"
        )

    @pytest.mark.slow_on("TorchBkd")
    def test_eig_increases_with_more_observations(self, bkd):
        """Test EIG generally increases when observation weights increase.

        This is a weak consistency check - more informative observations
        should generally yield higher expected information gain.
        """
        obj = self._create_kl_objective(bkd, self._nobs, self._ninner, self._nouter)

        # Smaller weights = less informative
        weights_low = bkd.ones((self._nobs, 1)) * 0.1
        # Larger weights = more informative
        weights_high = bkd.ones((self._nobs, 1)) * 2.0

        eig_low = obj.expected_information_gain(weights_low)
        eig_high = obj.expected_information_gain(weights_high)

        # Higher weights should generally give higher EIG
        # (more observations = more information)
        assert eig_high > eig_low, (
            f"EIG with high weights ({eig_high}) should exceed EIG with "
            f"low weights ({eig_low})"
        )
