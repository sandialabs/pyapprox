"""Tests for IshigamiFunction."""

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
from pyapprox.typing.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
)


class TestIshigamiFunction(Generic[Array], unittest.TestCase):
    """Base tests for IshigamiFunction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance_function(self) -> None:
        """Test that IshigamiFunction satisfies FunctionProtocol."""
        func = IshigamiFunction(self._bkd)
        self.assertIsInstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self) -> None:
        """Test that IshigamiFunction satisfies FunctionWithJacobianAndHVPProtocol."""
        func = IshigamiFunction(self._bkd)
        self.assertIsInstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns 3."""
        func = IshigamiFunction(self._bkd)
        self.assertEqual(func.nvars(), 3)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        func = IshigamiFunction(self._bkd)
        self.assertEqual(func.nqoi(), 1)

    def test_evaluation_single(self) -> None:
        """Test evaluation at a single sample."""
        func = IshigamiFunction(self._bkd)
        sample = self._bkd.array([[0.0], [0.0], [0.0]])
        result = func(sample)
        self.assertEqual(result.shape, (1, 1))
        # f(0,0,0) = sin(0) + 7*sin^2(0) + 0.1*0^4*sin(0) = 0
        self._bkd.assert_allclose(result, self._bkd.zeros((1, 1)), atol=1e-14)

    def test_evaluation_batch(self) -> None:
        """Test evaluation at multiple samples."""
        func = IshigamiFunction(self._bkd)
        pi = math.pi
        samples = self._bkd.array([
            [0.0, pi/2, pi/2],
            [0.0, pi/2, 0.0],
            [0.0, 0.0, 0.0],
        ])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))
        # f(0,0,0) = 0
        # f(pi/2, pi/2, 0) = sin(pi/2) + 7*sin^2(pi/2) = 1 + 7 = 8
        # f(pi/2, 0, 0) = sin(pi/2) + 0 = 1
        expected = self._bkd.array([[0.0, 8.0, 1.0]])
        self._bkd.assert_allclose(result, expected, atol=1e-12)

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        func = IshigamiFunction(self._bkd)
        sample = self._bkd.array([[0.5], [0.3], [-0.2]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 3))

    def test_jacobian_at_origin(self) -> None:
        """Test Jacobian at origin."""
        func = IshigamiFunction(self._bkd)
        sample = self._bkd.array([[0.0], [0.0], [0.0]])
        jac = func.jacobian(sample)
        # At origin:
        # df/dx1 = cos(0)*(1 + 0.1*0) = 1
        # df/dx2 = 2*7*sin(0)*cos(0) = 0
        # df/dx3 = 4*0.1*0^3*sin(0) = 0
        expected = self._bkd.array([[1.0, 0.0, 0.0]])
        self._bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_jacobian_invalid_shape(self) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = IshigamiFunction(self._bkd)
        # Wrong number of columns
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2], [-0.2, 0.3]])
        with self.assertRaises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        func = IshigamiFunction(self._bkd)
        sample = self._bkd.array([[0.5], [0.3], [-0.2]])
        vec = self._bkd.array([[1.0], [0.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (3, 1))

    def test_hvp_invalid_sample_shape(self) -> None:
        """Test HVP raises for invalid sample shape."""
        func = IshigamiFunction(self._bkd)
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2], [-0.2, 0.3]])
        vec = self._bkd.array([[1.0], [0.0], [0.0]])
        with self.assertRaises(ValueError):
            func.hvp(sample, vec)

    def test_hvp_invalid_vec_shape(self) -> None:
        """Test HVP raises for invalid vec shape."""
        func = IshigamiFunction(self._bkd)
        sample = self._bkd.array([[0.5], [0.3], [-0.2]])
        vec = self._bkd.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        with self.assertRaises(ValueError):
            func.hvp(sample, vec)

    def test_derivative_checker_jacobian(self) -> None:
        """Test Jacobian passes derivative checker."""
        func = IshigamiFunction(self._bkd)
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.7], [-0.5], [0.3]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 2e-6)

    def test_derivative_checker_hvp(self) -> None:
        """Test HVP passes derivative checker."""
        func = IshigamiFunction(self._bkd)
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.7], [-0.5], [0.3]])
        errors = checker.check_derivatives(sample, verbosity=0)
        # errors[1] is the Hessian/HVP error
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 2e-6)

    def test_custom_parameters(self) -> None:
        """Test custom a and b parameters."""
        func = IshigamiFunction(self._bkd, a=5.0, b=0.2)
        sample = self._bkd.array([[0.0], [math.pi/2], [0.0]])
        result = func(sample)
        # f(0, pi/2, 0) = sin(0) + 5*sin^2(pi/2) + 0.2*0*sin(0) = 0 + 5 + 0 = 5
        expected = self._bkd.array([[5.0]])
        self._bkd.assert_allclose(result, expected, atol=1e-12)


class TestIshigamiFunctionNumpy(TestIshigamiFunction[NDArray[Any]]):
    """NumPy backend tests for IshigamiFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIshigamiFunctionTorch(TestIshigamiFunction[torch.Tensor]):
    """PyTorch backend tests for IshigamiFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
