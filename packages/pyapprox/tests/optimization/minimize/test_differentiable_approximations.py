"""Tests for differentiable approximations to non-smooth functions.

Tests focus on:
- Convergence to non-smooth function as eps -> 0
- Derivative correctness via DerivativeChecker
- Numerical stability at extreme values
- Protocol compliance
"""

import numpy as np
import pytest

from pyapprox.optimization.minimize.differentiable_approximations import (
    DifferentiableApproximationProtocol,
    SmoothLogBasedLeftHeavisideFunction,
    SmoothLogBasedMaxFunction,
    SmoothLogBasedRightHeavisideFunction,
)


class TestDifferentiableApproximations:
    """Base test class - NOT run directly."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    # --- Protocol Tests ---

    def test_smooth_max_satisfies_protocol(self, bkd) -> None:
        """SmoothLogBasedMaxFunction satisfies protocol."""
        func = SmoothLogBasedMaxFunction(bkd, eps=0.1)
        assert isinstance(func, DifferentiableApproximationProtocol)

    def test_smooth_right_heaviside_satisfies_protocol(self, bkd) -> None:
        """SmoothLogBasedRightHeavisideFunction satisfies protocol."""
        func = SmoothLogBasedRightHeavisideFunction(bkd, eps=0.1)
        assert isinstance(func, DifferentiableApproximationProtocol)

    def test_smooth_left_heaviside_satisfies_protocol(self, bkd) -> None:
        """SmoothLogBasedLeftHeavisideFunction satisfies protocol."""
        func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.1)
        assert isinstance(func, DifferentiableApproximationProtocol)

    # --- Accessor Tests ---

    def test_accessors(self, bkd) -> None:
        """Accessors return correct values."""
        eps = 0.5
        threshold = 50.0
        func = SmoothLogBasedMaxFunction(bkd, eps=eps, threshold=threshold)

        bkd.assert_allclose(
            bkd.asarray([func.eps()]),
            bkd.asarray([eps]),
        )
        bkd.assert_allclose(
            bkd.asarray([func.threshold()]),
            bkd.asarray([threshold]),
        )

    def test_invalid_eps_raises(self, bkd) -> None:
        """Non-positive eps raises ValueError."""
        with pytest.raises(ValueError):
            SmoothLogBasedMaxFunction(bkd, eps=0.0)
        with pytest.raises(ValueError):
            SmoothLogBasedMaxFunction(bkd, eps=-0.1)

    # --- SmoothLogBasedMaxFunction Tests ---

    def test_smooth_max_convergence(self, bkd) -> None:
        """Smooth max converges to max(0, x) as eps -> 0."""
        x = bkd.asarray([[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]])
        true_max = bkd.maximum(bkd.zeros(x.shape), x)

        # Test convergence with decreasing eps
        for eps in [1.0, 0.1, 0.01]:
            func = SmoothLogBasedMaxFunction(bkd, eps=eps)
            approx = func(x)

            # Error should decrease with eps
            error = bkd.max(bkd.abs(approx - true_max))
            # For small eps, error should be O(eps)
            assert float(error) < 2 * eps

    def test_smooth_max_asymptotic_positive(self, bkd) -> None:
        """Smooth max equals x for large positive x."""
        eps = 0.1
        func = SmoothLogBasedMaxFunction(bkd, eps=eps)

        x_large = bkd.asarray([[100.0, 200.0, 500.0]])
        result = func(x_large)

        bkd.assert_allclose(result, x_large, rtol=1e-10)

    def test_smooth_max_asymptotic_negative(self, bkd) -> None:
        """Smooth max equals 0 for large negative x."""
        eps = 0.1
        func = SmoothLogBasedMaxFunction(bkd, eps=eps)

        x_large_neg = bkd.asarray([[-100.0, -200.0, -500.0]])
        result = func(x_large_neg)

        bkd.assert_allclose(result, bkd.zeros(x_large_neg.shape), atol=1e-10)

    def test_smooth_max_first_derivative_finite_diff(self, bkd) -> None:
        """Smooth max first derivative matches finite differences."""
        eps = 0.5
        func = SmoothLogBasedMaxFunction(bkd, eps=eps)

        x = bkd.asarray([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        h = 1e-7

        # Finite difference approximation
        fd_deriv = (func(x + h) - func(x - h)) / (2 * h)
        analytical_deriv = func.first_derivative(x)

        bkd.assert_allclose(analytical_deriv, fd_deriv, rtol=1e-5)

    def test_smooth_max_second_derivative_finite_diff(self, bkd) -> None:
        """Smooth max second derivative matches finite differences."""
        eps = 0.5
        func = SmoothLogBasedMaxFunction(bkd, eps=eps)

        x = bkd.asarray([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        h = 1e-5

        # Finite difference approximation of second derivative
        fd_deriv2 = (func.first_derivative(x + h) - func.first_derivative(x - h)) / (
            2 * h
        )
        analytical_deriv2 = func.second_derivative(x)

        bkd.assert_allclose(analytical_deriv2, fd_deriv2, rtol=1e-4)

    def test_smooth_max_derivative_bounds(self, bkd) -> None:
        """Smooth max first derivative is in [0, 1]."""
        eps = 0.1
        func = SmoothLogBasedMaxFunction(bkd, eps=eps)

        x = bkd.asarray(np.linspace(-5, 5, 100).reshape(1, -1))
        deriv = func.first_derivative(x)

        # Derivative should be in [0, 1] (sigmoid)
        assert float(bkd.min(deriv)) >= -1e-10
        assert float(bkd.max(deriv)) <= 1.0 + 1e-10

    # --- SmoothLogBasedRightHeavisideFunction Tests ---

    def test_smooth_right_heaviside_convergence(self, bkd) -> None:
        """Smooth right Heaviside converges to H(x) as eps -> 0."""
        # Avoid exact zero where H(x) is undefined
        x = bkd.asarray([[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]])
        true_heaviside = bkd.asarray([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])

        for eps in [0.1, 0.01, 0.001]:
            func = SmoothLogBasedRightHeavisideFunction(bkd, eps=eps)
            approx = func(x)

            # Should converge to step function
            bkd.assert_allclose(approx, true_heaviside, atol=0.1)

    def test_smooth_right_heaviside_at_zero(self, bkd) -> None:
        """Smooth right Heaviside equals 0.5 at x=0."""
        func = SmoothLogBasedRightHeavisideFunction(bkd, eps=0.1)

        x = bkd.asarray([[0.0]])
        result = func(x)

        bkd.assert_allclose(result, bkd.asarray([[0.5]]), rtol=1e-10)

    def test_smooth_right_heaviside_first_derivative_finite_diff(self, bkd) -> None:
        """Smooth right Heaviside first derivative matches finite differences."""
        eps = 0.5
        func = SmoothLogBasedRightHeavisideFunction(bkd, eps=eps)

        x = bkd.asarray([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        h = 1e-7

        fd_deriv = (func(x + h) - func(x - h)) / (2 * h)
        analytical_deriv = func.first_derivative(x)

        bkd.assert_allclose(analytical_deriv, fd_deriv, rtol=1e-5)

    def test_smooth_right_heaviside_second_derivative_finite_diff(self, bkd) -> None:
        """Smooth right Heaviside second derivative matches finite differences."""
        eps = 0.5
        func = SmoothLogBasedRightHeavisideFunction(bkd, eps=eps)

        x = bkd.asarray([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        h = 1e-5

        fd_deriv2 = (func.first_derivative(x + h) - func.first_derivative(x - h)) / (
            2 * h
        )
        analytical_deriv2 = func.second_derivative(x)

        bkd.assert_allclose(analytical_deriv2, fd_deriv2, rtol=1e-4, atol=1e-10)

    # --- SmoothLogBasedLeftHeavisideFunction Tests ---

    def test_smooth_left_heaviside_convergence(self, bkd) -> None:
        """Smooth left Heaviside converges to H(-x) as eps -> 0."""
        # Avoid exact zero
        x = bkd.asarray([[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]])
        # H(-x) = 1 if x <= 0, else 0
        true_heaviside = bkd.asarray([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])

        for eps in [0.1, 0.01, 0.001]:
            func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=eps)
            approx = func(x)

            bkd.assert_allclose(approx, true_heaviside, atol=0.1)

    def test_smooth_left_heaviside_at_zero(self, bkd) -> None:
        """Smooth left Heaviside equals 0.5 at x=0."""
        func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.1)

        x = bkd.asarray([[0.0]])
        result = func(x)

        bkd.assert_allclose(result, bkd.asarray([[0.5]]), rtol=1e-10)

    def test_smooth_left_heaviside_first_derivative_finite_diff(self, bkd) -> None:
        """Smooth left Heaviside first derivative matches finite differences."""
        eps = 0.5
        func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=eps)

        x = bkd.asarray([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        h = 1e-7

        fd_deriv = (func(x + h) - func(x - h)) / (2 * h)
        analytical_deriv = func.first_derivative(x)

        bkd.assert_allclose(analytical_deriv, fd_deriv, rtol=1e-5)

    def test_smooth_left_heaviside_second_derivative_finite_diff(self, bkd) -> None:
        """Smooth left Heaviside second derivative matches finite differences."""
        eps = 0.5
        func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=eps)

        x = bkd.asarray([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        h = 1e-5

        fd_deriv2 = (func.first_derivative(x + h) - func.first_derivative(x - h)) / (
            2 * h
        )
        analytical_deriv2 = func.second_derivative(x)

        bkd.assert_allclose(analytical_deriv2, fd_deriv2, rtol=1e-4, atol=1e-10)

    def test_left_right_heaviside_relationship(self, bkd) -> None:
        """Left and right Heaviside sum to 1."""
        eps = 0.3
        left = SmoothLogBasedLeftHeavisideFunction(bkd, eps=eps)
        right = SmoothLogBasedRightHeavisideFunction(bkd, eps=eps)

        x = bkd.asarray(np.linspace(-3, 3, 50).reshape(1, -1))

        # H(-x) + H(x) = 1 for smooth approximations
        total = left(x) + right(x)
        bkd.assert_allclose(total, bkd.ones(x.shape), rtol=1e-10)

    # --- Shape Tests ---

    def test_output_shape_preserved(self, bkd) -> None:
        """Output shape matches input shape."""
        func = SmoothLogBasedMaxFunction(bkd, eps=0.1)

        # Test various shapes
        for shape in [(1, 10), (3, 5), (1, 1)]:
            x = bkd.asarray(np.random.randn(*shape))
            assert func(x).shape == shape
            assert func.first_derivative(x).shape == shape
            assert func.second_derivative(x).shape == shape

    # --- Numerical Stability Tests ---

    def test_numerical_stability_extreme_values(self, bkd) -> None:
        """Functions handle extreme values without overflow."""
        eps = 0.1

        # Test all three functions
        funcs = [
            SmoothLogBasedMaxFunction(bkd, eps=eps),
            SmoothLogBasedRightHeavisideFunction(bkd, eps=eps),
            SmoothLogBasedLeftHeavisideFunction(bkd, eps=eps),
        ]

        x_extreme = bkd.asarray([[-1000.0, -100.0, 100.0, 1000.0]])

        for func in funcs:
            # Should not raise or return inf/nan
            result = func(x_extreme)
            deriv1 = func.first_derivative(x_extreme)
            deriv2 = func.second_derivative(x_extreme)

            # Check no NaN values (sum of NaN is NaN)
            assert not bool(bkd.isnan(bkd.sum(result)))
            assert not bool(bkd.isnan(bkd.sum(deriv1)))
            assert not bool(bkd.isnan(bkd.sum(deriv2)))

            # Check finite (max and min should be finite)
            assert float(bkd.max(bkd.abs(result))) < 1e10
            assert float(bkd.max(bkd.abs(deriv1))) < 1e10
            assert float(bkd.max(bkd.abs(deriv2))) < 1e10

    # --- Shift Parameter Tests ---

    def test_shift_accessor(self, bkd) -> None:
        """Shift accessor returns correct value."""
        shift = 0.5
        func = SmoothLogBasedMaxFunction(bkd, eps=0.1, shift=shift)
        bkd.assert_allclose(
            bkd.asarray([func.shift()]),
            bkd.asarray([shift]),
        )

    def test_smooth_max_with_shift(self, bkd) -> None:
        """Smooth max with shift shifts the transition point."""
        eps = 0.1
        shift = 0.5

        func_no_shift = SmoothLogBasedMaxFunction(bkd, eps=eps)
        func_with_shift = SmoothLogBasedMaxFunction(bkd, eps=eps, shift=shift)

        x = bkd.asarray([[-1.0, -0.5, 0.0, 0.5, 1.0]])

        # func(x) with shift should equal func(x + shift) without shift
        result_shifted = func_with_shift(x)
        result_manual = func_no_shift(x + shift)

        bkd.assert_allclose(result_shifted, result_manual, rtol=1e-10)

    def test_smooth_max_shift_derivative(self, bkd) -> None:
        """Smooth max derivatives with shift match finite differences."""
        eps = 0.5
        shift = 0.3
        func = SmoothLogBasedMaxFunction(bkd, eps=eps, shift=shift)

        x = bkd.asarray([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        h = 1e-7

        # Finite difference
        fd_deriv = (func(x + h) - func(x - h)) / (2 * h)
        analytical = func.first_derivative(x)

        bkd.assert_allclose(analytical, fd_deriv, rtol=1e-5)

    def test_smooth_heaviside_with_shift(self, bkd) -> None:
        """Smooth Heaviside with shift shifts the transition point."""
        eps = 0.1
        shift = 0.5

        # Right Heaviside: H(x+shift) instead of H(x)
        # Transition at x = -shift instead of x = 0
        func_right = SmoothLogBasedRightHeavisideFunction(bkd, eps=eps, shift=shift)

        # At x = -shift, value should be 0.5
        x_right_transition = bkd.asarray([[-shift]])
        bkd.assert_allclose(
            func_right(x_right_transition), bkd.asarray([[0.5]]), rtol=1e-6
        )

        # Left Heaviside: evaluates right Heaviside at -x
        # So H_left(x) = H_right(-x) with shift = H_right(-x + shift)
        # Transition occurs when -x + shift = 0, i.e., x = shift
        func_left = SmoothLogBasedLeftHeavisideFunction(bkd, eps=eps, shift=shift)

        # At x = shift, value should be 0.5
        x_left_transition = bkd.asarray([[shift]])
        bkd.assert_allclose(
            func_left(x_left_transition), bkd.asarray([[0.5]]), rtol=1e-6
        )

        # For x well below shift, left Heaviside should be ~1
        x_neg = bkd.asarray([[-1.0]])
        assert float(func_left(x_neg)[0, 0]) > 0.99

        # For x well above shift, left Heaviside should be ~0
        x_pos = bkd.asarray([[2.0]])
        assert float(func_left(x_pos)[0, 0]) < 0.01

    def test_shift_propagates_to_heaviside(self, bkd) -> None:
        """Shift parameter propagates from Heaviside to underlying max."""
        shift = 0.25
        func = SmoothLogBasedLeftHeavisideFunction(bkd, eps=0.1, shift=shift)

        bkd.assert_allclose(
            bkd.asarray([func.shift()]),
            bkd.asarray([shift]),
        )
