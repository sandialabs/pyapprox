"""Tests for Genz functions.

This module tests the four differentiable Genz integration test functions:
- OscillatoryFunction
- ProductPeakFunction
- CornerPeakFunction
- GaussianPeakFunction

Tests verify:
1. Protocol compliance (FunctionProtocol, FunctionWithJacobianAndHVPProtocol)
2. Correct shapes for evaluation, Jacobian, and HVP
3. Derivative correctness via DerivativeChecker
4. Monte Carlo integral convergence rate (MSE ~ O(1/n))
"""

import math
from abc import abstractmethod
from typing import Any

import numpy as np
import pytest

from pyapprox.benchmarks.functions.genz import (
    CornerPeakFunction,
    GaussianPeakFunction,
    OscillatoryFunction,
    ProductPeakFunction,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
)


class GenzFunctionTestBase:
    """Base test class for all Genz functions.

    Subclasses must implement:
    - _create_function_2d(bkd) -> function instance
    - _create_function_5d(bkd) -> function instance
    """

    @abstractmethod
    def _create_function_2d(self, bkd) -> Any:
        """Create a 2D version of the Genz function."""
        raise NotImplementedError

    @abstractmethod
    def _create_function_5d(self, bkd) -> Any:
        """Create a 5D version of the Genz function."""
        raise NotImplementedError

    # Protocol compliance tests
    def test_protocol_compliance_function(self, bkd) -> None:
        """Test that function satisfies FunctionProtocol."""
        func = self._create_function_2d(bkd)
        assert isinstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self, bkd) -> None:
        """Test protocol compliance with Jacobian and HVP."""
        func = self._create_function_2d(bkd)
        assert isinstance(func, FunctionWithJacobianAndHVPProtocol)

    # nvars and nqoi tests
    def test_nvars_2d(self, bkd) -> None:
        """Test nvars returns 2 for 2D function."""
        func = self._create_function_2d(bkd)
        assert func.nvars() == 2

    def test_nvars_5d(self, bkd) -> None:
        """Test nvars returns 5 for 5D function."""
        func = self._create_function_5d(bkd)
        assert func.nvars() == 5

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns 1."""
        func = self._create_function_2d(bkd)
        assert func.nqoi() == 1

    # Evaluation tests
    def test_evaluation_batch_2d(self, bkd) -> None:
        """Test evaluation at multiple samples (2D)."""
        func = self._create_function_2d(bkd)
        samples = bkd.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        result = func(samples)
        assert result.shape == (1, 3)

    def test_evaluation_batch_5d(self, bkd) -> None:
        """Test evaluation at multiple samples (5D)."""
        func = self._create_function_5d(bkd)
        samples = bkd.array(
            [
                [0.0, 0.5, 1.0],
                [0.1, 0.4, 0.9],
                [0.2, 0.3, 0.8],
                [0.3, 0.2, 0.7],
                [0.4, 0.1, 0.6],
            ]
        )
        result = func(samples)
        assert result.shape == (1, 3)

    # Jacobian tests
    def test_jacobian_shape_2d(self, bkd) -> None:
        """Test Jacobian has correct shape (2D)."""
        func = self._create_function_2d(bkd)
        sample = bkd.array([[0.5], [0.3]])
        jac = func.jacobian(sample)
        assert jac.shape == (1, 2)

    def test_jacobian_shape_5d(self, bkd) -> None:
        """Test Jacobian has correct shape (5D)."""
        func = self._create_function_5d(bkd)
        sample = bkd.array([[0.5], [0.3], [0.7], [0.2], [0.8]])
        jac = func.jacobian(sample)
        assert jac.shape == (1, 5)

    def test_jacobian_invalid_shape(self, bkd) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = self._create_function_2d(bkd)
        sample = bkd.array([[0.5, 0.1], [0.3, 0.2]])
        with pytest.raises(ValueError):
            func.jacobian(sample)

    # HVP tests
    def test_hvp_shape_2d(self, bkd) -> None:
        """Test HVP has correct shape (2D)."""
        func = self._create_function_2d(bkd)
        sample = bkd.array([[0.5], [0.3]])
        vec = bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        assert hvp.shape == (2, 1)

    def test_hvp_shape_5d(self, bkd) -> None:
        """Test HVP has correct shape (5D)."""
        func = self._create_function_5d(bkd)
        sample = bkd.array([[0.5], [0.3], [0.7], [0.2], [0.8]])
        vec = bkd.array([[1.0], [0.0], [0.5], [-0.5], [0.2]])
        hvp = func.hvp(sample, vec)
        assert hvp.shape == (5, 1)

    # Derivative checker tests
    def test_derivative_checker_jacobian_2d(self, bkd) -> None:
        """Test Jacobian passes derivative checker (2D)."""
        func = self._create_function_2d(bkd)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        assert error_ratio < 5e-6

    def test_derivative_checker_jacobian_5d(self, bkd) -> None:
        """Test Jacobian passes derivative checker (5D)."""
        func = self._create_function_5d(bkd)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.3], [0.7], [0.2], [0.8], [0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        assert error_ratio < 5e-6

    def test_derivative_checker_hvp_2d(self, bkd) -> None:
        """Test HVP passes derivative checker (2D)."""
        func = self._create_function_2d(bkd)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        assert error_ratio < 2e-6

    def test_derivative_checker_hvp_5d(self, bkd) -> None:
        """Test HVP passes derivative checker (5D)."""
        func = self._create_function_5d(bkd)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.3], [0.7], [0.2], [0.8], [0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        assert error_ratio < 5e-6

    # MC integral convergence tests
    def _run_mc_convergence_test(self, bkd, func: Any) -> None:
        """Run MC convergence test for a function.

        Verifies that Monte Carlo integration error decreases at the expected
        rate: MSE ~ O(1/n), which corresponds to a slope of -1 on a log-log
        plot.
        """
        exact_integral = func.integrate()

        # Set seed for reproducibility
        rng = np.random.default_rng(seed=42)

        sample_sizes = [100, 500, 1000, 5000, 10000]
        n_trials = 300
        mse_values = []

        for n in sample_sizes:
            squared_errors = []
            for _ in range(n_trials):
                # Use numpy for random sampling (acceptable in tests)
                samples_np = rng.uniform(0, 1, (func.nvars(), n))
                samples = bkd.asarray(samples_np)
                mc_estimate = float(bkd.mean(func(samples)))
                squared_errors.append((mc_estimate - exact_integral) ** 2)
            mse_values.append(sum(squared_errors) / n_trials)

        # Fit log-log line: log(MSE) = slope * log(n) + intercept
        # Expected slope ~ -1 (MSE ~ 1/n)
        log_n = [math.log(n) for n in sample_sizes]
        log_mse = [math.log(mse) for mse in mse_values]

        # Linear regression
        n_pts = len(sample_sizes)
        mean_log_n = sum(log_n) / n_pts
        mean_log_mse = sum(log_mse) / n_pts
        numerator = sum(
            (ln - mean_log_n) * (lm - mean_log_mse) for ln, lm in zip(log_n, log_mse)
        )
        denominator = sum((ln - mean_log_n) ** 2 for ln in log_n)
        slope = numerator / denominator

        # Slope should be approximately -1 (within +/- 0.05)
        assert slope > -1.05
        assert slope < -0.95

    def test_integrate_mc_convergence_2d(self, bkd) -> None:
        """Test integral via MC convergence rate (2D)."""
        self._run_mc_convergence_test(bkd, self._create_function_2d(bkd))

    def test_integrate_mc_convergence_5d(self, bkd) -> None:
        """Test integral via MC convergence rate (5D)."""
        self._run_mc_convergence_test(bkd, self._create_function_5d(bkd))


# =============================================================================
# Oscillatory Function Tests
# =============================================================================
class TestOscillatoryFunctionBase(GenzFunctionTestBase):
    """Base tests for OscillatoryFunction."""

    def _create_function_2d(self, bkd):
        """Create 2D oscillatory function."""
        return OscillatoryFunction(bkd, c=[1.0, 2.0], w=[0.25, 0.25])

    def _create_function_5d(self, bkd):
        """Create 5D oscillatory function."""
        return OscillatoryFunction(bkd, c=[1.0, 2.0, 1.5, 0.5, 1.0], w=[0.25] * 5)

    def test_empty_c_raises(self, bkd) -> None:
        """Test that empty c raises ValueError."""
        with pytest.raises(ValueError):
            OscillatoryFunction(bkd, c=[], w=[])


# =============================================================================
# Product Peak Function Tests
# =============================================================================
class TestProductPeakFunctionBase(GenzFunctionTestBase):
    """Base tests for ProductPeakFunction."""

    def _create_function_2d(self, bkd):
        """Create 2D product peak function."""
        return ProductPeakFunction(bkd, c=[1.0, 2.0], w=[0.5, 0.5])

    def _create_function_5d(self, bkd):
        """Create 5D product peak function."""
        return ProductPeakFunction(bkd, c=[1.0, 2.0, 1.5, 0.5, 1.0], w=[0.5] * 5)

    def test_evaluation_at_center(self, bkd) -> None:
        """Test evaluation at peak center."""
        func = ProductPeakFunction(bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = bkd.array([[0.5], [0.5]])
        result = func(sample)
        # At w, f = prod(c_i^2) = 1.0
        bkd.assert_allclose(result, bkd.array([[1.0]]), rtol=1e-10)

    def test_jacobian_at_center_is_zero(self, bkd) -> None:
        """Test Jacobian at center is zero (gradient is zero at peak)."""
        func = ProductPeakFunction(bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = bkd.array([[0.5], [0.5]])
        jac = func.jacobian(sample)
        bkd.assert_allclose(jac, bkd.zeros((1, 2)), atol=1e-10)

    def test_empty_c_raises(self, bkd) -> None:
        """Test that empty c raises ValueError."""
        with pytest.raises(ValueError):
            ProductPeakFunction(bkd, c=[], w=[])


# =============================================================================
# Corner Peak Function Tests
# =============================================================================
class TestCornerPeakFunctionBase(GenzFunctionTestBase):
    """Base tests for CornerPeakFunction."""

    def _create_function_2d(self, bkd):
        """Create 2D corner peak function."""
        return CornerPeakFunction(bkd, c=[1.0, 2.0])

    def _create_function_5d(self, bkd):
        """Create 5D corner peak function."""
        return CornerPeakFunction(bkd, c=[1.0, 2.0, 1.5, 0.5, 1.0])

    def test_evaluation_at_origin(self, bkd) -> None:
        """Test evaluation at origin (corner)."""
        func = CornerPeakFunction(bkd, c=[1.0, 1.0])
        sample = bkd.array([[0.0], [0.0]])
        result = func(sample)
        # At origin, f = 1^{-(D+1)} = 1
        bkd.assert_allclose(result, bkd.array([[1.0]]), rtol=1e-10)

    def test_empty_c_raises(self, bkd) -> None:
        """Test that empty c raises ValueError."""
        with pytest.raises(ValueError):
            CornerPeakFunction(bkd, c=[])


# =============================================================================
# Gaussian Peak Function Tests
# =============================================================================
class TestGaussianPeakFunctionBase(GenzFunctionTestBase):
    """Base tests for GaussianPeakFunction."""

    def _create_function_2d(self, bkd):
        """Create 2D Gaussian peak function."""
        return GaussianPeakFunction(bkd, c=[1.0, 2.0], w=[0.5, 0.5])

    def _create_function_5d(self, bkd):
        """Create 5D Gaussian peak function."""
        return GaussianPeakFunction(bkd, c=[1.0, 2.0, 1.5, 0.5, 1.0], w=[0.5] * 5)

    def test_evaluation_at_center(self, bkd) -> None:
        """Test evaluation at peak center."""
        func = GaussianPeakFunction(bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = bkd.array([[0.5], [0.5]])
        result = func(sample)
        # At w, f = exp(0) = 1
        bkd.assert_allclose(result, bkd.array([[1.0]]), rtol=1e-10)

    def test_jacobian_at_center_is_zero(self, bkd) -> None:
        """Test Jacobian at center is zero (gradient is zero at peak)."""
        func = GaussianPeakFunction(bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = bkd.array([[0.5], [0.5]])
        jac = func.jacobian(sample)
        bkd.assert_allclose(jac, bkd.zeros((1, 2)), atol=1e-10)

    def test_empty_c_raises(self, bkd) -> None:
        """Test that empty c raises ValueError."""
        with pytest.raises(ValueError):
            GaussianPeakFunction(bkd, c=[], w=[])
