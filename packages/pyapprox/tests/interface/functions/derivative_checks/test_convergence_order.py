"""Tests for central differences and the convergence-order check.

``f(x) = x_0^3 + x_1^2 + sin(x_0 x_1)``, smooth, with exact Jacobian and
Hessian-vector product. A correct derivative's finite-difference error
falls like ``h`` (forward) or ``h**2`` (central); a wrong one stays flat.
"""

import math

import pytest
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestConvergenceOrder:
    def _checker(
        self, bkd: Backend[Array], jacobian_offset: float = 0.0
    ) -> DerivativeChecker[Array]:
        """``jacobian_offset`` adds a constant error to the Jacobian."""

        def value(x: Array) -> Array:
            return bkd.reshape(
                x[0] ** 3 + x[1] ** 2 + bkd.sin(x[0] * x[1]), (1, x.shape[1])
            )

        def jacobian(x: Array) -> Array:
            c = bkd.cos(x[0] * x[1])
            return bkd.stack(
                [3 * x[0] ** 2 + x[1] * c + jacobian_offset, 2 * x[1] + x[0] * c],
                axis=1,
            )

        def hvp(x: Array, v: Array) -> Array:
            s, c = bkd.sin(x[0] * x[1]), bkd.cos(x[0] * x[1])
            h00 = 6 * x[0] - x[1] ** 2 * s
            h01 = c - x[0] * x[1] * s
            h11 = 2 - x[0] ** 2 * s
            return bkd.stack([h00 * v[0] + h01 * v[1], h01 * v[0] + h11 * v[1]])

        return DerivativeChecker(
            FunctionWithJacobianAndHVPFromCallable(
                nvars=2, fun=value, jacobian=jacobian, hvp=hvp, bkd=bkd
            )
        )

    def _steps(self, bkd: Backend[Array]) -> Array:
        return bkd.flip(bkd.logspace(-12, -1, 23))

    def _sample(self, bkd: Backend[Array]) -> Array:
        return bkd.asarray([[0.7], [-0.4]])

    @pytest.mark.parametrize("central,order", [(False, 1.0), (True, 2.0)])
    def test_order_of_correct_derivatives(
        self, bkd: Backend[Array], central: bool, order: float
    ) -> None:
        """Both the Jacobian and the Hessian-vector product."""
        checker = self._checker(bkd)
        steps = self._steps(bkd)
        errors = checker.check_derivatives(
            self._sample(bkd), fd_eps=steps, central=central
        )
        for error in errors:
            fitted = bkd.to_float(checker.convergence_order(error, steps))
            assert abs(fitted - order) < 0.2, f"order {fitted:.2f}"

    def test_central_differences_go_deeper(self, bkd: Backend[Array]) -> None:
        checker = self._checker(bkd)
        steps = self._steps(bkd)
        forward = checker.check_derivatives(self._sample(bkd), fd_eps=steps)[0]
        central = checker.check_derivatives(
            self._sample(bkd), fd_eps=steps, central=True
        )[0]
        ratio_forward = bkd.to_float(checker.error_ratio(forward))
        ratio_central = bkd.to_float(checker.error_ratio(central))
        assert ratio_central < 1e-2 * ratio_forward

    def _direction(self, bkd: Backend[Array]) -> Array:
        """Fixed, so which error sign cancels the truncation error is fixed."""
        return bkd.asarray([[0.6], [0.8]])

    @pytest.mark.parametrize("central", [False, True])
    def test_v_shape_passes_correct_derivatives(
        self, bkd: Backend[Array], central: bool
    ) -> None:
        """Both the Jacobian and the Hessian-vector product."""
        checker = self._checker(bkd)
        steps = self._steps(bkd)
        errors = checker.check_derivatives(
            self._sample(bkd),
            fd_eps=steps,
            direction=self._direction(bkd),
            central=central,
        )
        for error in errors:
            report = checker.check_v_shape(error, steps, central=central)
            assert report.passed, report

    @pytest.mark.parametrize("central", [False, True])
    @pytest.mark.parametrize("offset", [1e-3, -1e-3])
    def test_v_shape_fails_wrong_jacobian(
        self, bkd: Backend[Array], central: bool, offset: float
    ) -> None:
        """Whichever sign the error has relative to the truncation error."""
        checker = self._checker(bkd, jacobian_offset=offset)
        steps = self._steps(bkd)
        error = checker.check_derivatives(
            self._sample(bkd),
            fd_eps=steps,
            direction=self._direction(bkd),
            central=central,
        )[0]
        assert not checker.check_v_shape(error, steps, central=central).passed

    def test_plateau_has_no_rounding_side_and_order_zero(
        self, bkd: Backend[Array]
    ) -> None:
        """An error with the truncation error's sign: the error flattens."""
        checker = self._checker(bkd, jacobian_offset=-1e-3)
        steps = self._steps(bkd)
        error = checker.check_derivatives(
            self._sample(bkd), fd_eps=steps, direction=self._direction(bkd)
        )[0]
        report = checker.check_v_shape(error, steps)
        assert not report.rounding_side_rises
        assert abs(report.order) < 0.3

    def test_cancellation_dip_fools_order_but_not_location(
        self, bkd: Backend[Array]
    ) -> None:
        """An error that cancels the truncation error at one step makes a
        dip whose side falls at order 1. Its bottom is far above where
        rounding puts a correct derivative's minimum."""
        checker = self._checker(bkd, jacobian_offset=1e-3)
        steps = self._steps(bkd)
        error = checker.check_derivatives(
            self._sample(bkd), fd_eps=steps, direction=self._direction(bkd)
        )[0]
        report = checker.check_v_shape(error, steps)
        assert report.order_matches
        assert not report.bottom_is_rounding_limited
        assert not report.passed
        assert report.bottom_step > 1e-5

    def test_v_shape_without_truncation_regime(self, bkd: Backend[Array]) -> None:
        """Only tiny steps: no fitting window, so the order is NaN and the
        check fails rather than raising."""
        checker = self._checker(bkd)
        steps = bkd.flip(bkd.logspace(-14, -11, 7))
        error = checker.check_derivatives(self._sample(bkd), fd_eps=steps)[0]
        report = checker.check_v_shape(error, steps)
        assert math.isnan(report.order)
        assert not report.passed

    def test_rejects_steps_without_truncation_regime(self, bkd: Backend[Array]) -> None:
        """Only tiny steps: the error rises throughout, minimum at the top."""
        checker = self._checker(bkd)
        steps = bkd.flip(bkd.logspace(-14, -11, 7))
        error = checker.check_derivatives(self._sample(bkd), fd_eps=steps)[0]
        with pytest.raises(ValueError, match="widen fd_eps"):
            checker.convergence_order(error, steps)

    def test_rejects_mismatched_shapes(self, bkd: Backend[Array]) -> None:
        checker = self._checker(bkd)
        steps = self._steps(bkd)
        error = checker.check_derivatives(self._sample(bkd), fd_eps=steps)[0]
        with pytest.raises(ValueError, match="same shape"):
            checker.convergence_order(error, steps[:-1])
