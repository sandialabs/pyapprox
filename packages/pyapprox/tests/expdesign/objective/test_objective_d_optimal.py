"""
Standalone tests for DOptimalLinearModelObjective.

PERMANENT - no legacy imports.

These tests verify correctness using:
1. DerivativeChecker for Jacobian and Hessian
2. Self-consistent property checks (symmetry, shapes)
"""

import numpy as np

from pyapprox.expdesign.objective import DOptimalLinearModelObjective
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)


class TestDOptimalStandalone:
    """Standalone tests for DOptimalLinearModelObjective."""

    def _setup_data(self, bkd):
        np.random.seed(42)

        self._nobs = 5
        self._nparams = 3
        self._design_matrix = bkd.asarray(
            np.random.randn(self._nobs, self._nparams)
        )
        self._noise_cov = bkd.asarray(np.array(0.1))
        self._prior_cov = bkd.asarray(np.array(1.0))
        self._weights = bkd.ones((self._nobs, 1)) / self._nobs

    def _create_objective(self, bkd):
        return DOptimalLinearModelObjective(
            self._design_matrix,
            self._noise_cov,
            self._prior_cov,
            bkd,
        )

    def _create_derivative_checker_function(self, bkd, obj):
        """Wrap objective for DerivativeChecker compatibility.

        DerivativeChecker expects:
        - __call__(samples): (nvars, nsamples) -> (nqoi, nsamples)
        - jacobian(sample): (nvars, 1) -> (nqoi, nvars)
        - hvp(sample, vec): (nvars, 1), (nvars, 1) -> (nvars, 1)

        DOptimalLinearModelObjective has:
        - __call__(weights): (nobs, 1) -> (1, 1)
        - jacobian(weights): (nobs, 1) -> (1, nobs)
        - hvp(weights, vec): (nobs, 1), (nobs, 1) -> (1, nobs)
        """

        def value_fun(samples):
            # samples: (nvars, nsamples) = (nobs, nsamples)
            nsamples = samples.shape[1]
            results = []
            for ii in range(nsamples):
                w = samples[:, ii : ii + 1]  # (nobs, 1)
                val = obj(w)  # (1, 1)
                results.append(val[0, 0])
            return bkd.reshape(bkd.stack(results), (1, nsamples))

        def jacobian_fun(sample):
            # sample: (nvars, 1) = (nobs, 1)
            jac = obj.jacobian(sample)  # (1, nobs)
            return jac  # (nqoi=1, nvars=nobs)

        def hvp_fun(sample, vec):
            # sample: (nvars, 1), vec: (nvars, 1)
            hvp = obj.hvp(sample, vec)  # (1, nobs)
            return hvp.T  # (nvars=nobs, 1)

        return FunctionWithJacobianAndHVPFromCallable(
            nvars=obj.nvars(),
            fun=value_fun,
            jacobian=jacobian_fun,
            hvp=hvp_fun,
            bkd=bkd,
        )

    def test_objective_shape(self, bkd):
        """Test objective returns correct shape."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        result = obj(self._weights)
        assert result.shape == (1, 1)

    def test_jacobian_shape(self, bkd):
        """Test Jacobian returns correct shape."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        jac = obj.jacobian(self._weights)
        assert jac.shape == (1, self._nobs)

    def test_hessian_shape(self, bkd):
        """Test Hessian returns correct shape."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        hess = obj.hessian(self._weights)
        assert hess.shape == (1, self._nobs, self._nobs)

    def test_jacobian_derivative_checker(self, bkd):
        """Test Jacobian using DerivativeChecker."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        wrapped = self._create_derivative_checker_function(bkd, obj)
        checker = DerivativeChecker(wrapped)

        # Check at uniform weights
        errors = checker.check_derivatives(self._weights)

        # Jacobian errors should show second-order convergence
        # (error_ratio ~ 0.25 for correct implementation)
        ratio = checker.error_ratio(errors[0])
        assert float(bkd.to_numpy(ratio)) <= 1e-6

    def test_hvp_derivative_checker(self, bkd):
        """Test HVP (Hessian-vector product) using DerivativeChecker."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        wrapped = self._create_derivative_checker_function(bkd, obj)
        checker = DerivativeChecker(wrapped)

        # Check at uniform weights
        errors = checker.check_derivatives(self._weights)

        # HVP errors should show second-order convergence
        ratio = checker.error_ratio(errors[1])
        assert float(bkd.to_numpy(ratio)) <= 1e-6

    def test_hessian_symmetric(self, bkd):
        """Test Hessian is symmetric."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        hess = obj.hessian(self._weights)

        # Extract the 2D matrix
        hess_2d = hess[0]
        bkd.assert_allclose(hess_2d, hess_2d.T, rtol=1e-12)

    def test_different_weights(self, bkd):
        """Test with non-uniform weights."""
        self._setup_data(bkd)
        np.random.seed(123)
        weights = bkd.asarray(np.random.dirichlet(np.ones(self._nobs))[:, None])

        obj = self._create_objective(bkd)
        result = obj(weights)
        jac = obj.jacobian(weights)
        hess = obj.hessian(weights)

        # Check shapes
        assert result.shape == (1, 1)
        assert jac.shape == (1, self._nobs)
        assert hess.shape == (1, self._nobs, self._nobs)

        # Check finite
        assert np.isfinite(bkd.to_numpy(result)).all()
        assert np.isfinite(bkd.to_numpy(jac)).all()
        assert np.isfinite(bkd.to_numpy(hess)).all()

    def test_scalar_validation(self, bkd):
        """Test that non-scalar covariances raise error."""
        self._setup_data(bkd)
        import pytest

        with pytest.raises(TypeError):
            DOptimalLinearModelObjective(
                self._design_matrix,
                bkd.asarray(np.array([0.1, 0.2])),  # Not scalar
                self._prior_cov,
                bkd,
            )

        with pytest.raises(TypeError):
            DOptimalLinearModelObjective(
                self._design_matrix,
                self._noise_cov,
                bkd.asarray(np.array([1.0, 2.0])),  # Not scalar
                bkd,
            )

    def test_nobs_nparams(self, bkd):
        """Test nobs and nparams accessors."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        assert obj.nobs() == self._nobs
        assert obj.nparams() == self._nparams
        assert obj.nvars() == self._nobs
        assert obj.nqoi() == 1

    def test_jacobian_implemented(self, bkd):
        """Test jacobian_implemented returns True."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        assert obj.jacobian_implemented()

    def test_hessian_implemented(self, bkd):
        """Test hessian_implemented returns True."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        assert obj.hessian_implemented()

    def test_hvp_implemented(self, bkd):
        """Test hvp_implemented returns True."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        assert obj.hvp_implemented()

    def test_evaluate_alias(self, bkd):
        """Test evaluate is alias for __call__."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        result_call = obj(self._weights)
        result_eval = obj.evaluate(self._weights)
        bkd.assert_allclose(result_call, result_eval, rtol=1e-12)

    def test_hvp_matches_hessian_times_vec(self, bkd):
        """Test HVP equals Hessian @ vec."""
        self._setup_data(bkd)
        obj = self._create_objective(bkd)
        np.random.seed(456)
        vec = bkd.asarray(np.random.randn(self._nobs, 1))

        hvp_result = obj.hvp(self._weights, vec)
        hess = obj.hessian(self._weights)
        hess_2d = bkd.reshape(hess, (self._nobs, self._nobs))
        expected = bkd.dot(hess_2d, vec).T  # (1, nobs)

        bkd.assert_allclose(hvp_result, expected, rtol=1e-12)
