"""Tests for Genz functions."""

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
from pyapprox.typing.benchmarks.functions.genz import (
    OscillatoryFunction,
    ProductPeakFunction,
    CornerPeakFunction,
    GaussianPeakFunction,
)


class TestOscillatoryFunction(Generic[Array], unittest.TestCase):
    """Base tests for OscillatoryFunction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance_function(self) -> None:
        """Test that OscillatoryFunction satisfies FunctionProtocol."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        self.assertIsInstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self) -> None:
        """Test protocol compliance with Jacobian and HVP."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        self.assertIsInstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0, 3.0], w=[0.5] * 3)
        self.assertEqual(func.nvars(), 3)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        self.assertEqual(func.nqoi(), 1)

    def test_evaluation_batch(self) -> None:
        """Test evaluation at multiple samples."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        samples = self._bkd.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0, 3.0], w=[0.5] * 3)
        sample = self._bkd.array([[0.5], [0.3], [0.7]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 3))

    def test_jacobian_invalid_shape(self) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2]])
        with self.assertRaises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.3]])
        vec = self._bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (2, 1))

    def test_derivative_checker_jacobian(self) -> None:
        """Test Jacobian passes derivative checker."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_hvp(self) -> None:
        """Test HVP passes derivative checker."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 1e-6)

    def test_integrate(self) -> None:
        """Test analytical integral computation."""
        func = OscillatoryFunction(self._bkd, c=[1.0, 2.0], w=[0.25, 0.25])
        integral = func.integrate()
        # Check that integral is a finite number
        self.assertTrue(-2.0 < integral < 2.0)

    def test_empty_c_raises(self) -> None:
        """Test that empty c raises ValueError."""
        with self.assertRaises(ValueError):
            OscillatoryFunction(self._bkd, c=[], w=[])


class TestOscillatoryFunctionNumpy(TestOscillatoryFunction[NDArray[Any]]):
    """NumPy backend tests for OscillatoryFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOscillatoryFunctionTorch(TestOscillatoryFunction[torch.Tensor]):
    """PyTorch backend tests for OscillatoryFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestProductPeakFunction(Generic[Array], unittest.TestCase):
    """Base tests for ProductPeakFunction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance_function(self) -> None:
        """Test that ProductPeakFunction satisfies FunctionProtocol."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        self.assertIsInstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self) -> None:
        """Test protocol compliance with Jacobian and HVP."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        self.assertIsInstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 2.0, 3.0], w=[0.5] * 3)
        self.assertEqual(func.nvars(), 3)

    def test_evaluation_at_center(self) -> None:
        """Test evaluation at peak center."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.5]])
        result = func(sample)
        # At w, f = prod(c_i^2) = 1.0
        self._bkd.assert_allclose(result, self._bkd.array([[1.0]]), rtol=1e-10)

    def test_evaluation_batch(self) -> None:
        """Test evaluation at multiple samples."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        samples = self._bkd.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 2.0, 3.0], w=[0.5] * 3)
        sample = self._bkd.array([[0.5], [0.3], [0.7]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 3))

    def test_jacobian_at_center_is_zero(self) -> None:
        """Test Jacobian at center is zero (gradient is zero at peak)."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.5]])
        jac = func.jacobian(sample)
        self._bkd.assert_allclose(jac, self._bkd.zeros((1, 2)), atol=1e-10)

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.3]])
        vec = self._bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (2, 1))

    def test_derivative_checker_jacobian(self) -> None:
        """Test Jacobian passes derivative checker."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_hvp(self) -> None:
        """Test HVP passes derivative checker."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 1e-6)

    def test_integrate(self) -> None:
        """Test analytical integral computation."""
        func = ProductPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        integral = func.integrate()
        self.assertTrue(integral > 0)


class TestProductPeakFunctionNumpy(TestProductPeakFunction[NDArray[Any]]):
    """NumPy backend tests for ProductPeakFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestProductPeakFunctionTorch(TestProductPeakFunction[torch.Tensor]):
    """PyTorch backend tests for ProductPeakFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestCornerPeakFunction(Generic[Array], unittest.TestCase):
    """Base tests for CornerPeakFunction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance_function(self) -> None:
        """Test that CornerPeakFunction satisfies FunctionProtocol."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 2.0])
        self.assertIsInstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self) -> None:
        """Test protocol compliance with Jacobian and HVP."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 2.0])
        self.assertIsInstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 2.0, 3.0])
        self.assertEqual(func.nvars(), 3)

    def test_evaluation_at_origin(self) -> None:
        """Test evaluation at origin (corner)."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 1.0])
        sample = self._bkd.array([[0.0], [0.0]])
        result = func(sample)
        # At origin, f = 1^{-(D+1)} = 1
        self._bkd.assert_allclose(result, self._bkd.array([[1.0]]), rtol=1e-10)

    def test_evaluation_batch(self) -> None:
        """Test evaluation at multiple samples."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 2.0])
        samples = self._bkd.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 2.0, 3.0])
        sample = self._bkd.array([[0.5], [0.3], [0.7]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 3))

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 2.0])
        sample = self._bkd.array([[0.5], [0.3]])
        vec = self._bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (2, 1))

    def test_derivative_checker_jacobian(self) -> None:
        """Test Jacobian passes derivative checker."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 2.0])
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_hvp(self) -> None:
        """Test HVP passes derivative checker."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 2.0])
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 1e-6)

    def test_integrate(self) -> None:
        """Test analytical integral computation."""
        func = CornerPeakFunction(self._bkd, c=[1.0, 2.0])
        integral = func.integrate()
        self.assertTrue(integral > 0)


class TestCornerPeakFunctionNumpy(TestCornerPeakFunction[NDArray[Any]]):
    """NumPy backend tests for CornerPeakFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCornerPeakFunctionTorch(TestCornerPeakFunction[torch.Tensor]):
    """PyTorch backend tests for CornerPeakFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestGaussianPeakFunction(Generic[Array], unittest.TestCase):
    """Base tests for GaussianPeakFunction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance_function(self) -> None:
        """Test that GaussianPeakFunction satisfies FunctionProtocol."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        self.assertIsInstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self) -> None:
        """Test protocol compliance with Jacobian and HVP."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        self.assertIsInstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns correct value."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 2.0, 3.0], w=[0.5] * 3)
        self.assertEqual(func.nvars(), 3)

    def test_evaluation_at_center(self) -> None:
        """Test evaluation at peak center."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.5]])
        result = func(sample)
        # At w, f = exp(0) = 1
        self._bkd.assert_allclose(result, self._bkd.array([[1.0]]), rtol=1e-10)

    def test_evaluation_batch(self) -> None:
        """Test evaluation at multiple samples."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        samples = self._bkd.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 2.0, 3.0], w=[0.5] * 3)
        sample = self._bkd.array([[0.5], [0.3], [0.7]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 3))

    def test_jacobian_at_center_is_zero(self) -> None:
        """Test Jacobian at center is zero (gradient is zero at peak)."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 1.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.5]])
        jac = func.jacobian(sample)
        self._bkd.assert_allclose(jac, self._bkd.zeros((1, 2)), atol=1e-10)

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        sample = self._bkd.array([[0.5], [0.3]])
        vec = self._bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (2, 1))

    def test_derivative_checker_jacobian(self) -> None:
        """Test Jacobian passes derivative checker."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_hvp(self) -> None:
        """Test HVP passes derivative checker."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        checker = DerivativeChecker(func)
        sample = self._bkd.array([[0.3], [0.7]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 1e-6)

    def test_integrate(self) -> None:
        """Test analytical integral computation."""
        func = GaussianPeakFunction(self._bkd, c=[1.0, 2.0], w=[0.5, 0.5])
        integral = func.integrate()
        self.assertTrue(integral > 0)


class TestGaussianPeakFunctionNumpy(TestGaussianPeakFunction[NDArray[Any]]):
    """NumPy backend tests for GaussianPeakFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGaussianPeakFunctionTorch(TestGaussianPeakFunction[torch.Tensor]):
    """PyTorch backend tests for GaussianPeakFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
