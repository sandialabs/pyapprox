"""Tests for RosenbrockFunction."""

import numpy as np
import pytest

from pyapprox_benchmarks.functions.algebraic.rosenbrock import (
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

# TODO: this test class should be where function is defined
# not at this level which is for integration tests.


class TestRosenbrockFunction:
    """Base tests for RosenbrockFunction."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_protocol_compliance_function(self, bkd) -> None:
        """Test that RosenbrockFunction satisfies FunctionProtocol."""
        func = RosenbrockFunction(bkd, nvars=2)
        assert isinstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self, bkd) -> None:
        """Test that RosenbrockFunction satisfies FunctionWithJacobianAndHVPProtocol."""
        func = RosenbrockFunction(bkd, nvars=2)
        assert isinstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars_default(self, bkd) -> None:
        """Test default nvars is 2."""
        func = RosenbrockFunction(bkd)
        assert func.nvars() == 2

    def test_nvars_custom(self, bkd) -> None:
        """Test custom nvars."""
        func = RosenbrockFunction(bkd, nvars=5)
        assert func.nvars() == 5

    def test_nvars_minimum(self, bkd) -> None:
        """Test that nvars < 2 raises."""
        with pytest.raises(ValueError):
            RosenbrockFunction(bkd, nvars=1)

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns 1."""
        func = RosenbrockFunction(bkd, nvars=2)
        assert func.nqoi() == 1

    def test_evaluation_at_minimum(self, bkd) -> None:
        """Test evaluation at global minimum."""
        func = RosenbrockFunction(bkd, nvars=2)
        sample = bkd.array([[1.0], [1.0]])
        result = func(sample)
        assert result.shape == (1, 1)
        # f(1,1) = 0
        bkd.assert_allclose(result, bkd.zeros((1, 1)), atol=1e-14)

    def test_evaluation_at_origin(self, bkd) -> None:
        """Test evaluation at origin."""
        func = RosenbrockFunction(bkd, nvars=2)
        sample = bkd.array([[0.0], [0.0]])
        result = func(sample)
        # f(0,0) = 100*(0-0)^2 + (1-0)^2 = 1
        expected = bkd.array([[1.0]])
        bkd.assert_allclose(result, expected, atol=1e-14)

    def test_evaluation_batch(self, bkd) -> None:
        """Test evaluation at multiple samples."""
        func = RosenbrockFunction(bkd, nvars=2)
        samples = bkd.array(
            [
                [1.0, 0.0, -1.0],
                [1.0, 0.0, 2.0],
            ]
        )
        result = func(samples)
        assert result.shape == (1, 3)
        # f(1,1) = 0
        # f(0,0) = 1
        # f(-1,2) = 100*(2-1)^2 + (1-(-1))^2 = 100 + 4 = 104
        expected = bkd.array([[0.0, 1.0, 104.0]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_evaluation_3d(self, bkd) -> None:
        """Test evaluation with 3 variables."""
        func = RosenbrockFunction(bkd, nvars=3)
        sample = bkd.array([[1.0], [1.0], [1.0]])
        result = func(sample)
        # f(1,1,1) = 0
        bkd.assert_allclose(result, bkd.zeros((1, 1)), atol=1e-14)

    def test_jacobian_shape(self, bkd) -> None:
        """Test Jacobian has correct shape."""
        func = RosenbrockFunction(bkd, nvars=2)
        sample = bkd.array([[0.5], [0.3]])
        jac = func.jacobian(sample)
        assert jac.shape == (1, 2)

    def test_jacobian_at_minimum(self, bkd) -> None:
        """Test Jacobian at minimum is zero."""
        func = RosenbrockFunction(bkd, nvars=2)
        sample = bkd.array([[1.0], [1.0]])
        jac = func.jacobian(sample)
        expected = bkd.zeros((1, 2))
        bkd.assert_allclose(jac, expected, atol=1e-14)

    def test_jacobian_invalid_shape(self, bkd) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = RosenbrockFunction(bkd, nvars=2)
        sample = bkd.array([[0.5, 0.1], [0.3, 0.2]])
        with pytest.raises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self, bkd) -> None:
        """Test HVP has correct shape."""
        func = RosenbrockFunction(bkd, nvars=2)
        sample = bkd.array([[0.5], [0.3]])
        vec = bkd.array([[1.0], [0.0]])
        hvp = func.hvp(sample, vec)
        assert hvp.shape == (2, 1)

    def test_hvp_invalid_sample_shape(self, bkd) -> None:
        """Test HVP raises for invalid sample shape."""
        func = RosenbrockFunction(bkd, nvars=2)
        sample = bkd.array([[0.5, 0.1], [0.3, 0.2]])
        vec = bkd.array([[1.0], [0.0]])
        with pytest.raises(ValueError):
            func.hvp(sample, vec)

    def test_hvp_invalid_vec_shape(self, bkd) -> None:
        """Test HVP raises for invalid vec shape."""
        func = RosenbrockFunction(bkd, nvars=2)
        sample = bkd.array([[0.5], [0.3]])
        vec = bkd.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            func.hvp(sample, vec)

    def test_derivative_checker_jacobian_2d(self, bkd) -> None:
        """Test Jacobian passes derivative checker (2D)."""
        func = RosenbrockFunction(bkd, nvars=2)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.7], [-0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        assert error_ratio < 1e-6

    def test_derivative_checker_hvp_2d(self, bkd) -> None:
        """Test HVP passes derivative checker (2D)."""
        func = RosenbrockFunction(bkd, nvars=2)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.7], [-0.5]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        assert error_ratio < 1e-5

    def test_derivative_checker_jacobian_5d(self, bkd) -> None:
        """Test Jacobian passes derivative checker (5D)."""
        func = RosenbrockFunction(bkd, nvars=5)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.7], [-0.5], [0.3], [0.1], [-0.2]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        assert error_ratio < 1e-6

    def test_derivative_checker_hvp_5d(self, bkd) -> None:
        """Test HVP passes derivative checker (5D)."""
        func = RosenbrockFunction(bkd, nvars=5)
        checker = DerivativeChecker(func)
        sample = bkd.array([[0.7], [-0.5], [0.3], [0.1], [-0.2]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        assert error_ratio < 1e-5
