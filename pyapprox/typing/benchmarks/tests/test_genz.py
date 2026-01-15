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
import unittest
from abc import abstractmethod
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.typing.benchmarks.functions.genz import (
    OscillatoryFunction,
    ProductPeakFunction,
    CornerPeakFunction,
    GaussianPeakFunction,
)


class GenzFunctionTestBase(Generic[Array], unittest.TestCase):
    """Base test class for all Genz functions.

    Subclasses must implement:
    - bkd() -> Backend[Array]
    - _create_function_2d() -> function instance
    - _create_function_5d() -> function instance
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    @abstractmethod
    def _create_function_2d(self) -> Any:
        """Create a 2D version of the Genz function."""
        raise NotImplementedError

    @abstractmethod
    def _create_function_5d(self) -> Any:
        """Create a 5D version of the Genz function."""
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    # Protocol compliance tests
    def test_protocol_compliance_function(self) -> None:
        """Test that function satisfies FunctionProtocol."""
        func = self._create_function_2d()
        self.assertIsInstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self) -> None:
        """Test protocol compliance with Jacobian and HVP."""
        func = self._create_function_2d()
        self.assertIsInstance(func, FunctionWithJacobianAndHVPProtocol)

    # nvars and nqoi tests
    def test_nvars_2d(self) -> None:
        """Test nvars returns 2 for 2D function."""
        func = self._create_function_2d()
        self.assertEqual(func.nvars(), 2)

    def test_nvars_5d(self) -> None:
        """Test nvars returns 5 for 5D function."""
        func = self._create_function_5d()
        self.assertEqual(func.nvars(), 5)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        func = self._create_function_2d()
        self.assertEqual(func.nqoi(), 1)

    # Evaluation tests
    def test_evaluation_batch_2d(self) -> None:
        """Test evaluation at multiple samples (2D)."""
        func = self._create_function_2d()
        samples = self._bkd.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_evaluation_batch_5d(self) -> None:
        """Test evaluation at multiple samples (5D)."""
        func = self._create_function_5d()
        samples = self._bkd.array([
            [0.0, 0.5, 1.0],
            [0.1, 0.4, 0.9],
            [0.2, 0.3, 0.8],
            [0.3, 0.2, 0.7],
            [0.4, 0.1, 0.6],
        ])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    # Jacobian tests
    def test_jacobian_shape_2d(self) -> None:
        """Test Jacobian has correct shape (2D)."""
        func = self._create_function_2d()
        sample = self._bkd.array([[0.5], [0.3]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 2))

    def test_jacobian_shape_5d(self) -> None:
        """Test Jacobian has correct shape (5D)."""
        func = self._create_function_5d()
        sample = self._bkd.array([[0.5], [0.3], [0.7], [0.2], [0.8]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 5))

    def test_jacobian_invalid_shape(self) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = self._create_function_2d()
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2]])
        with self.assertRaises(ValueError):
            func.jacobian(sample)

    # HVP tests
    def test_hvp_shape_2d(self) -> None:
        """Test HVP has correct shape (2D)."""
        func = self._create_function_2d()
        sample = self._bkd.array([[0.5], [0.3]])
        vec = self._bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (2, 1))

    def test_hvp_shape_5d(self) -> None:
        """Test HVP has correct shape (5D)."""
        func = self._create_function_5d()
        sample = self._bkd.array([[0.5], [0.3], [0.7], [0.2], [0.8]])
        vec = self._bkd.array([[1.0], [0.0], [0.5], [-0.5], [0.2]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (5, 1))

    # Derivative checker tests
    def test_derivative_checker_jacobian_2d(self) -> None:
        """Test Jacobian passes derivative checker (2D)."""
        func = self._create_function_2d()
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_jacobian_5d(self) -> None:
        """Test Jacobian passes derivative checker (5D)."""
        func = self._create_function_5d()
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7], [0.2], [0.8], [0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_hvp_2d(self) -> None:
        """Test HVP passes derivative checker (2D)."""
        func = self._create_function_2d()
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_hvp_5d(self) -> None:
        """Test HVP passes derivative checker (5D)."""
        func = self._create_function_5d()
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7], [0.2], [0.8], [0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 1e-6)

    # MC integral convergence tests
    def _run_mc_convergence_test(self, func: Any) -> None:
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
                samples = self._bkd.asarray(samples_np)
                mc_estimate = float(self._bkd.mean(func(samples)))
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
            (ln - mean_log_n) * (lm - mean_log_mse)
            for ln, lm in zip(log_n, log_mse)
        )
        denominator = sum((ln - mean_log_n) ** 2 for ln in log_n)
        slope = numerator / denominator

        # Slope should be approximately -1 (within +/- 0.05)
        self.assertGreater(slope, -1.05)
        self.assertLess(slope, -0.95)

    def test_integrate_mc_convergence_2d(self) -> None:
        """Test integral via MC convergence rate (2D)."""
        self._run_mc_convergence_test(self._create_function_2d())

    def test_integrate_mc_convergence_5d(self) -> None:
        """Test integral via MC convergence rate (5D)."""
        self._run_mc_convergence_test(self._create_function_5d())


# =============================================================================
# Oscillatory Function Tests
# =============================================================================
class TestOscillatoryFunctionBase(GenzFunctionTestBase[Array]):
    """Base tests for OscillatoryFunction."""

    __test__ = False

    def _create_function_2d(self) -> OscillatoryFunction[Array]:
        """Create 2D oscillatory function."""
        return OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.25, 0.25])

    def _create_function_5d(self) -> OscillatoryFunction[Array]:
        """Create 5D oscillatory function."""
        return OscillatoryFunction(
            self._bkd, c=[1.0, 2.0, 1.5, 0.5, 1.0], w=[0.25] * 5
        )

    def test_empty_c_raises(self) -> None:
        """Test that empty c raises ValueError."""
        with self.assertRaises(ValueError):
            OscillatoryFunction(self._bkd, c=[], w=[])


class TestOscillatoryFunctionNumpy(TestOscillatoryFunctionBase[NDArray[Any]]):
    """NumPy backend tests for OscillatoryFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOscillatoryFunctionTorch(TestOscillatoryFunctionBase[torch.Tensor]):
    """PyTorch backend tests for OscillatoryFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Product Peak Function Tests
# =============================================================================
class TestProductPeakFunctionBase(GenzFunctionTestBase[Array]):
    """Base tests for ProductPeakFunction."""

    __test__ = False

    def _create_function_2d(self) -> ProductPeakFunction[Array]:
        """Create 2D product peak function."""
        return ProductPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])

    def _create_function_5d(self) -> ProductPeakFunction[Array]:
        """Create 5D product peak function."""
        return ProductPeakFunction(
            self._bkd, c=[1.0, 2.0, 1.5, 0.5, 1.0], w=[0.5] * 5
        )

    def test_evaluation_at_center(self) -> None:
        """Test evaluation at peak center."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.5]])
        result = func(sample)
        # At w, f = prod(c_i^2) = 1.0
        self._bkd.assert_allclose(result, self._bkd.array([[1.0]]), rtol=1e-10)

    def test_jacobian_at_center_is_zero(self) -> None:
        """Test Jacobian at center is zero (gradient is zero at peak)."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.5]])
        jac = func.jacobian(sample)
        self._bkd.assert_allclose(jac, self._bkd.zeros((1, 2)), atol=1e-10)

    def test_empty_c_raises(self) -> None:
        """Test that empty c raises ValueError."""
        with self.assertRaises(ValueError):
            ProductPeakFunction(self._bkd, c=[], w=[])


class TestProductPeakFunctionNumpy(TestProductPeakFunctionBase[NDArray[Any]]):
    """NumPy backend tests for ProductPeakFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestProductPeakFunctionTorch(TestProductPeakFunctionBase[torch.Tensor]):
    """PyTorch backend tests for ProductPeakFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Corner Peak Function Tests
# =============================================================================
class TestCornerPeakFunctionBase(GenzFunctionTestBase[Array]):
    """Base tests for CornerPeakFunction."""

    __test__ = False

    def _create_function_2d(self) -> CornerPeakFunction[Array]:
        """Create 2D corner peak function."""
        return CornerPeakFunction(self._bkd, c=[1.0, 2.0])

    def _create_function_5d(self) -> CornerPeakFunction[Array]:
        """Create 5D corner peak function."""
        return CornerPeakFunction(self._bkd, c=[1.0, 2.0, 1.5, 0.5, 1.0])

    def test_evaluation_at_origin(self) -> None:
        """Test evaluation at origin (corner)."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 1.0])
        sample = self._bkd.array([[0.0], [0.0]])
        result = func(sample)
        # At origin, f = 1^{-(D+1)} = 1
        self._bkd.assert_allclose(result, self._bkd.array([[1.0]]), rtol=1e-10)

    def test_empty_c_raises(self) -> None:
        """Test that empty c raises ValueError."""
        with self.assertRaises(ValueError):
            CornerPeakFunction(self._bkd, c=[])


class TestCornerPeakFunctionNumpy(TestCornerPeakFunctionBase[NDArray[Any]]):
    """NumPy backend tests for CornerPeakFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCornerPeakFunctionTorch(TestCornerPeakFunctionBase[torch.Tensor]):
    """PyTorch backend tests for CornerPeakFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# =============================================================================
# Gaussian Peak Function Tests
# =============================================================================
class TestGaussianPeakFunctionBase(GenzFunctionTestBase[Array]):
    """Base tests for GaussianPeakFunction."""

    __test__ = False

    def _create_function_2d(self) -> GaussianPeakFunction[Array]:
        """Create 2D Gaussian peak function."""
        return GaussianPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])

    def _create_function_5d(self) -> GaussianPeakFunction[Array]:
        """Create 5D Gaussian peak function."""
        return GaussianPeakFunction(
            self._bkd, c=[1.0, 2.0, 1.5, 0.5, 1.0], w=[0.5] * 5
        )

    def test_evaluation_at_center(self) -> None:
        """Test evaluation at peak center."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.5]])
        result = func(sample)
        # At w, f = exp(0) = 1
        self._bkd.assert_allclose(result, self._bkd.array([[1.0]]), rtol=1e-10)

    def test_jacobian_at_center_is_zero(self) -> None:
        """Test Jacobian at center is zero (gradient is zero at peak)."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.5]])
        jac = func.jacobian(sample)
        self._bkd.assert_allclose(jac, self._bkd.zeros((1, 2)), atol=1e-10)

    def test_empty_c_raises(self) -> None:
        """Test that empty c raises ValueError."""
        with self.assertRaises(ValueError):
            GaussianPeakFunction(self._bkd, c=[], w=[])


class TestGaussianPeakFunctionNumpy(TestGaussianPeakFunctionBase[NDArray[Any]]):
    """NumPy backend tests for GaussianPeakFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianPeakFunctionTorch(TestGaussianPeakFunctionBase[torch.Tensor]):
    """PyTorch backend tests for GaussianPeakFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
