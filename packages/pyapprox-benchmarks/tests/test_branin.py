"""Tests for BraninFunction."""

import math

import pytest

from pyapprox_benchmarks.functions.algebraic.branin import (
    BRANIN_GLOBAL_MINIMUM,
    BRANIN_MINIMIZERS,
    BraninFunction,
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


# TODO: this test class should be where branin function
# is defined not at this level which is for integration
# tests
class TestBraninFunction:
    """Base tests for BraninFunction."""

    def test_protocol_compliance_function(self, bkd) -> None:
        """Test that BraninFunction satisfies FunctionProtocol."""
        func = BraninFunction(bkd)
        assert isinstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self, bkd) -> None:
        """Test that BraninFunction satisfies FunctionWithJacobianAndHVPProtocol."""
        func = BraninFunction(bkd)
        assert isinstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self, bkd) -> None:
        """Test nvars returns 2."""
        func = BraninFunction(bkd)
        assert func.nvars() == 2

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns 1."""
        func = BraninFunction(bkd)
        assert func.nqoi() == 1

    def test_evaluation_at_minimizer_1(self, bkd) -> None:
        """Test evaluation at first global minimizer."""
        func = BraninFunction(bkd)
        x1, x2 = BRANIN_MINIMIZERS[0]
        sample = bkd.array([[x1], [x2]])
        result = func(sample)
        expected = bkd.array([[BRANIN_GLOBAL_MINIMUM]])
        bkd.assert_allclose(result, expected, rtol=1e-4)

    def test_evaluation_at_minimizer_2(self, bkd) -> None:
        """Test evaluation at second global minimizer."""
        func = BraninFunction(bkd)
        x1, x2 = BRANIN_MINIMIZERS[1]
        sample = bkd.array([[x1], [x2]])
        result = func(sample)
        expected = bkd.array([[BRANIN_GLOBAL_MINIMUM]])
        bkd.assert_allclose(result, expected, rtol=1e-4)

    def test_evaluation_at_minimizer_3(self, bkd) -> None:
        """Test evaluation at third global minimizer."""
        func = BraninFunction(bkd)
        x1, x2 = BRANIN_MINIMIZERS[2]
        sample = bkd.array([[x1], [x2]])
        result = func(sample)
        expected = bkd.array([[BRANIN_GLOBAL_MINIMUM]])
        bkd.assert_allclose(result, expected, rtol=1e-4)

    def test_evaluation_batch(self, bkd) -> None:
        """Test evaluation at multiple samples."""
        func = BraninFunction(bkd)
        samples = bkd.array(
            [
                [0.0, -math.pi, math.pi],
                [0.0, 12.275, 2.275],
            ]
        )
        result = func(samples)
        assert result.shape == (1, 3)

    def test_jacobian_shape(self, bkd) -> None:
        """Test Jacobian has correct shape."""
        func = BraninFunction(bkd)
        sample = bkd.array([[0.5], [5.0]])
        jac = func.jacobian(sample)
        assert jac.shape == (1, 2)

    def test_jacobian_at_minimizer_near_zero(self, bkd) -> None:
        """Test Jacobian at minimizer is near zero."""
        func = BraninFunction(bkd)
        x1, x2 = BRANIN_MINIMIZERS[1]
        sample = bkd.array([[x1], [x2]])
        jac = func.jacobian(sample)
        # At minimum, gradient should be near zero
        bkd.assert_allclose(jac, bkd.zeros((1, 2)), atol=1e-3)

    def test_jacobian_invalid_shape(self, bkd) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = BraninFunction(bkd)
        sample = bkd.array([[0.5, 0.1], [5.0, 3.0]])
        with pytest.raises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self, bkd) -> None:
        """Test HVP has correct shape."""
        func = BraninFunction(bkd)
        sample = bkd.array([[0.5], [5.0]])
        vec = bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        assert hvp.shape == (2, 1)

    def test_hvp_invalid_sample_shape(self, bkd) -> None:
        """Test HVP raises for invalid sample shape."""
        func = BraninFunction(bkd)
        sample = bkd.array([[0.5, 0.1], [5.0, 3.0]])
        vec = bkd.array([[1.0], [0.0]])
        with pytest.raises(ValueError):
            func.hvp(sample, vec)

    def test_hvp_invalid_vec_shape(self, bkd) -> None:
        """Test HVP raises for invalid vec shape."""
        func = BraninFunction(bkd)
        sample = bkd.array([[0.5], [5.0]])
        vec = bkd.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            func.hvp(sample, vec)

    def test_derivative_checker_jacobian(self, bkd) -> None:
        """Test Jacobian passes derivative checker."""
        func = BraninFunction(bkd)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.5], [5.0]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        assert error_ratio < 5e-6

    def test_derivative_checker_hvp(self, bkd) -> None:
        """Test HVP passes derivative checker."""
        func = BraninFunction(bkd)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.5], [5.0]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        assert error_ratio < 5e-6
