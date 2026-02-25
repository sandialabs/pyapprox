"""Tests for RosenbrockFunction."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.benchmarks.functions.algebraic.rosenbrock import (
    RosenbrockFunction,
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
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestRosenbrockFunction(Generic[Array], unittest.TestCase):
    """Base tests for RosenbrockFunction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance_function(self) -> None:
        """Test that RosenbrockFunction satisfies FunctionProtocol."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        self.assertIsInstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self) -> None:
        """Test that RosenbrockFunction satisfies FunctionWithJacobianAndHVPProtocol."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        self.assertIsInstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars_default(self) -> None:
        """Test default nvars is 2."""
        func = RosenbrockFunction(self._bkd)
        self.assertEqual(func.nvars(), 2)

    def test_nvars_custom(self) -> None:
        """Test custom nvars."""
        func = RosenbrockFunction(self._bkd, nvars=5)
        self.assertEqual(func.nvars(), 5)

    def test_nvars_minimum(self) -> None:
        """Test that nvars < 2 raises."""
        with self.assertRaises(ValueError):
            RosenbrockFunction(self._bkd, nvars=1)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        self.assertEqual(func.nqoi(), 1)

    def test_evaluation_at_minimum(self) -> None:
        """Test evaluation at global minimum."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        sample = self._bkd.array([[1.0], [1.0]])
        result = func(sample)
        self.assertEqual(result.shape, (1, 1))
        # f(1,1) = 0
        self._bkd.assert_allclose(result, self._bkd.zeros((1, 1)), atol=1e-14)

    def test_evaluation_at_origin(self) -> None:
        """Test evaluation at origin."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        sample = self._bkd.array([[0.0], [0.0]])
        result = func(sample)
        # f(0,0) = 100*(0-0)^2 + (1-0)^2 = 1
        expected = self._bkd.array([[1.0]])
        self._bkd.assert_allclose(result, expected, atol=1e-14)

    def test_evaluation_batch(self) -> None:
        """Test evaluation at multiple samples."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        samples = self._bkd.array(
            [
                [1.0, 0.0, -1.0],
                [1.0, 0.0, 2.0],
            ]
        )
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))
        # f(1,1) = 0
        # f(0,0) = 1
        # f(-1,2) = 100*(2-1)^2 + (1-(-1))^2 = 100 + 4 = 104
        expected = self._bkd.array([[0.0, 1.0, 104.0]])
        self._bkd.assert_allclose(result, expected, atol=1e-12)

    def test_evaluation_3d(self) -> None:
        """Test evaluation with 3 variables."""
        func = RosenbrockFunction(self._bkd, nvars=3)
        sample = self._bkd.array([[1.0], [1.0], [1.0]])
        result = func(sample)
        # f(1,1,1) = 0
        self._bkd.assert_allclose(result, self._bkd.zeros((1, 1)), atol=1e-14)

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        sample = self._bkd.array([[0.5], [0.3]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 2))

    def test_jacobian_at_minimum(self) -> None:
        """Test Jacobian at minimum is zero."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        sample = self._bkd.array([[1.0], [1.0]])
        jac = func.jacobian(sample)
        expected = self._bkd.zeros((1, 2))
        self._bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_jacobian_invalid_shape(self) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2]])
        with self.assertRaises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        sample = self._bkd.array([[0.5], [0.3]])
        vec = self._bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (2, 1))

    def test_hvp_invalid_sample_shape(self) -> None:
        """Test HVP raises for invalid sample shape."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2]])
        vec = self._bkd.array([[1.0], [0.0]])
        with self.assertRaises(ValueError):
            func.hvp(sample, vec)

    def test_hvp_invalid_vec_shape(self) -> None:
        """Test HVP raises for invalid vec shape."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        sample = self._bkd.array([[0.5], [0.3]])
        vec = self._bkd.array([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            func.hvp(sample, vec)

    def test_derivative_checker_jacobian_2d(self) -> None:
        """Test Jacobian passes derivative checker (2D)."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.7], [-0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_hvp_2d(self) -> None:
        """Test HVP passes derivative checker (2D)."""
        func = RosenbrockFunction(self._bkd, nvars=2)
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.7], [-0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_jacobian_5d(self) -> None:
        """Test Jacobian passes derivative checker (5D)."""
        func = RosenbrockFunction(self._bkd, nvars=5)
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.7], [-0.5], [0.3], [0.1], [-0.2]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_hvp_5d(self) -> None:
        """Test HVP passes derivative checker (5D)."""
        func = RosenbrockFunction(self._bkd, nvars=5)
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.7], [-0.5], [0.3], [0.1], [-0.2]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 1e-6)


class TestRosenbrockFunctionNumpy(TestRosenbrockFunction[NDArray[Any]]):
    """NumPy backend tests for RosenbrockFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRosenbrockFunctionTorch(TestRosenbrockFunction[torch.Tensor]):
    """PyTorch backend tests for RosenbrockFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
