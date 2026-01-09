"""Tests for BraninFunction."""

import math
import unittest
from typing import Any, Generic

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
from pyapprox.typing.benchmarks.functions.algebraic.branin import (
    BraninFunction,
    BRANIN_GLOBAL_MINIMUM,
    BRANIN_MINIMIZERS,
)


class TestBraninFunction(Generic[Array], unittest.TestCase):
    """Base tests for BraninFunction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance_function(self) -> None:
        """Test that BraninFunction satisfies FunctionProtocol."""
        func = BraninFunction(self._bkd)
        self.assertIsInstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self) -> None:
        """Test that BraninFunction satisfies FunctionWithJacobianAndHVPProtocol."""
        func = BraninFunction(self._bkd)
        self.assertIsInstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns 2."""
        func = BraninFunction(self._bkd)
        self.assertEqual(func.nvars(), 2)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        func = BraninFunction(self._bkd)
        self.assertEqual(func.nqoi(), 1)

    def test_evaluation_at_minimizer_1(self) -> None:
        """Test evaluation at first global minimizer."""
        func = BraninFunction(self._bkd)
        x1, x2 = BRANIN_MINIMIZERS[0]
        sample = self._bkd.array([[x1], [x2]])
        result = func(sample)
        expected = self._bkd.array([[BRANIN_GLOBAL_MINIMUM]])
        self._bkd.assert_allclose(result, expected, rtol=1e-4)

    def test_evaluation_at_minimizer_2(self) -> None:
        """Test evaluation at second global minimizer."""
        func = BraninFunction(self._bkd)
        x1, x2 = BRANIN_MINIMIZERS[1]
        sample = self._bkd.array([[x1], [x2]])
        result = func(sample)
        expected = self._bkd.array([[BRANIN_GLOBAL_MINIMUM]])
        self._bkd.assert_allclose(result, expected, rtol=1e-4)

    def test_evaluation_at_minimizer_3(self) -> None:
        """Test evaluation at third global minimizer."""
        func = BraninFunction(self._bkd)
        x1, x2 = BRANIN_MINIMIZERS[2]
        sample = self._bkd.array([[x1], [x2]])
        result = func(sample)
        expected = self._bkd.array([[BRANIN_GLOBAL_MINIMUM]])
        self._bkd.assert_allclose(result, expected, rtol=1e-4)

    def test_evaluation_batch(self) -> None:
        """Test evaluation at multiple samples."""
        func = BraninFunction(self._bkd)
        samples = self._bkd.array([
            [0.0, -math.pi, math.pi],
            [0.0, 12.275, 2.275],
        ])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        func = BraninFunction(self._bkd)
        sample = self._bkd.array([[0.5], [5.0]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 2))

    def test_jacobian_at_minimizer_near_zero(self) -> None:
        """Test Jacobian at minimizer is near zero."""
        func = BraninFunction(self._bkd)
        x1, x2 = BRANIN_MINIMIZERS[1]
        sample = self._bkd.array([[x1], [x2]])
        jac = func.jacobian(sample)
        # At minimum, gradient should be near zero
        self._bkd.assert_allclose(jac, self._bkd.zeros((1, 2)), atol=1e-3)

    def test_jacobian_invalid_shape(self) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = BraninFunction(self._bkd)
        sample = self._bkd.array([[0.5, 0.1], [5.0, 3.0]])
        with self.assertRaises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        func = BraninFunction(self._bkd)
        sample = self._bkd.array([[0.5], [5.0]])
        vec = self._bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (2, 1))

    def test_hvp_invalid_sample_shape(self) -> None:
        """Test HVP raises for invalid sample shape."""
        func = BraninFunction(self._bkd)
        sample = self._bkd.array([[0.5, 0.1], [5.0, 3.0]])
        vec = self._bkd.array([[1.0], [0.0]])
        with self.assertRaises(ValueError):
            func.hvp(sample, vec)

    def test_hvp_invalid_vec_shape(self) -> None:
        """Test HVP raises for invalid vec shape."""
        func = BraninFunction(self._bkd)
        sample = self._bkd.array([[0.5], [5.0]])
        vec = self._bkd.array([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            func.hvp(sample, vec)

    def test_derivative_checker_jacobian(self) -> None:
        """Test Jacobian passes derivative checker."""
        func = BraninFunction(self._bkd)
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.5], [5.0]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 2e-6)

    def test_derivative_checker_hvp(self) -> None:
        """Test HVP passes derivative checker."""
        func = BraninFunction(self._bkd)
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.5], [5.0]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 2e-6)


class TestBraninFunctionNumpy(TestBraninFunction[NDArray[Any]]):
    """NumPy backend tests for BraninFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBraninFunctionTorch(TestBraninFunction[torch.Tensor]):
    """PyTorch backend tests for BraninFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
