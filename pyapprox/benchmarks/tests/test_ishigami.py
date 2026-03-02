"""Tests for IshigamiFunction."""

import math

import pytest

from pyapprox.benchmarks.functions.algebraic.ishigami import (
    IshigamiFunction,
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


class TestIshigamiFunction:
    """Base tests for IshigamiFunction."""

    def test_protocol_compliance_function(self, bkd) -> None:
        """Test that IshigamiFunction satisfies FunctionProtocol."""
        func = IshigamiFunction(bkd)
        assert isinstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self, bkd) -> None:
        """Test that IshigamiFunction satisfies FunctionWithJacobianAndHVPProtocol."""
        func = IshigamiFunction(bkd)
        assert isinstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self, bkd) -> None:
        """Test nvars returns 3."""
        func = IshigamiFunction(bkd)
        assert func.nvars() == 3

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns 1."""
        func = IshigamiFunction(bkd)
        assert func.nqoi() == 1

    def test_evaluation_single(self, bkd) -> None:
        """Test evaluation at a single sample."""
        func = IshigamiFunction(bkd)
        sample = bkd.array([[0.0], [0.0], [0.0]])
        result = func(sample)
        assert result.shape == (1, 1)
        # f(0,0,0) = sin(0) + 7*sin^2(0) + 0.1*0^4*sin(0) = 0
        bkd.assert_allclose(result, bkd.zeros((1, 1)), atol=1e-14)

    def test_evaluation_batch(self, bkd) -> None:
        """Test evaluation at multiple samples."""
        func = IshigamiFunction(bkd)
        pi = math.pi
        samples = bkd.array(
            [
                [0.0, pi / 2, pi / 2],
                [0.0, pi / 2, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        result = func(samples)
        assert result.shape == (1, 3)
        # f(0,0,0) = 0
        # f(pi/2, pi/2, 0) = sin(pi/2) + 7*sin^2(pi/2) = 1 + 7 = 8
        # f(pi/2, 0, 0) = sin(pi/2) + 0 = 1
        expected = bkd.array([[0.0, 8.0, 1.0]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_jacobian_shape(self, bkd) -> None:
        """Test Jacobian has correct shape."""
        func = IshigamiFunction(bkd)
        sample = bkd.array([[0.5], [0.3], [-0.2]])
        jac = func.jacobian(sample)
        assert jac.shape == (1, 3)

    def test_jacobian_at_origin(self, bkd) -> None:
        """Test Jacobian at origin."""
        func = IshigamiFunction(bkd)
        sample = bkd.array([[0.0], [0.0], [0.0]])
        jac = func.jacobian(sample)
        # At origin:
        # df/dx1 = cos(0)*(1 + 0.1*0) = 1
        # df/dx2 = 2*7*sin(0)*cos(0) = 0
        # df/dx3 = 4*0.1*0^3*sin(0) = 0
        expected = bkd.array([[1.0, 0.0, 0.0]])
        bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_jacobian_invalid_shape(self, bkd) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = IshigamiFunction(bkd)
        # Wrong number of columns
        sample = bkd.array([[0.5, 0.1], [0.3, 0.2], [-0.2, 0.3]])
        with pytest.raises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self, bkd) -> None:
        """Test HVP has correct shape."""
        func = IshigamiFunction(bkd)
        sample = bkd.array([[0.5], [0.3], [-0.2]])
        vec = bkd.array([[1.0], [0.0], [0.0]])
        hvp = func.hvp(sample, vec)
        assert hvp.shape == (3, 1)

    def test_hvp_invalid_sample_shape(self, bkd) -> None:
        """Test HVP raises for invalid sample shape."""
        func = IshigamiFunction(bkd)
        sample = bkd.array([[0.5, 0.1], [0.3, 0.2], [-0.2, 0.3]])
        vec = bkd.array([[1.0], [0.0], [0.0]])
        with pytest.raises(ValueError):
            func.hvp(sample, vec)

    def test_hvp_invalid_vec_shape(self, bkd) -> None:
        """Test HVP raises for invalid vec shape."""
        func = IshigamiFunction(bkd)
        sample = bkd.array([[0.5], [0.3], [-0.2]])
        vec = bkd.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        with pytest.raises(ValueError):
            func.hvp(sample, vec)

    def test_derivative_checker_jacobian(self, bkd) -> None:
        """Test Jacobian passes derivative checker."""
        func = IshigamiFunction(bkd)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.7], [-0.5], [0.3]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        assert error_ratio < 2e-6

    def test_derivative_checker_hvp(self, bkd) -> None:
        """Test HVP passes derivative checker."""
        func = IshigamiFunction(bkd)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.7], [-0.5], [0.3]])
        errors = checker.check_derivatives(sample, verbosity=0)
        # errors[1] is the Hessian/HVP error
        error_ratio = checker.error_ratio(errors[1])
        assert error_ratio < 2e-6

    def test_custom_parameters(self, bkd) -> None:
        """Test custom a and b parameters."""
        func = IshigamiFunction(bkd, a=5.0, b=0.2)
        sample = bkd.array([[0.0], [math.pi / 2], [0.0]])
        result = func(sample)
        # f(0, pi/2, 0) = sin(0) + 5*sin^2(pi/2) + 0.2*0*sin(0) = 0 + 5 + 0 = 5
        expected = bkd.array([[5.0]])
        bkd.assert_allclose(result, expected, atol=1e-12)
