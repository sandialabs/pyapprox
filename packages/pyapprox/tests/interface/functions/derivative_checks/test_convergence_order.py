"""Tests for central differences and the convergence-order check.

``f(x) = x_0^3 + x_1^2 + sin(x_0 x_1)``, smooth, with exact Jacobian and
Hessian-vector product. A correct derivative's finite-difference error
falls like ``h`` (forward) or ``h**2`` (central); a wrong one stays flat.

A linear ``f`` has no truncation error, so no V: with the rounding
predicted at each step its curve passes as truncation free (the ratio of
error to prediction at most 10, against a default allowance of 100),
while a relative defect of 3e-9 or more fails, and a caller may widen
the allowance. A recorded curve with a broad bottom fails an order fit
by the minimum (order below 0.75) and passes one on the truncation side
the prediction marks (order 1). The curve's validations and the rounding
formula are checked directly.
"""

import math

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
    FDCurve,
    predicted_rounding,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
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


class TestTruncationFree:
    r"""``f(x) = 3 x_0 - 2 x_1`` at (0.7, -0.4) along (0.6, 0.8): linear, so
    forward differences are exact but for rounding, which rises as the
    step shrinks; there is no V. The rounding predicted there is about
    :math:`6.6 \times 10^{-15}/h` relative to :math:`J d = 0.2`."""

    _STEPS = [10.0**-k for k in range(1, 14)]

    def _curve(self, bkd: Backend[Array], jacobian_offset: float = 0.0) -> FDCurve:
        """``jacobian_offset`` adds to the Jacobian's first entry, a
        relative defect of 3 times itself along the direction."""

        def value(x: Array) -> Array:
            return bkd.reshape(3.0 * x[0] - 2.0 * x[1], (1, x.shape[1]))

        def jacobian(x: Array) -> Array:
            return bkd.asarray([[3.0 + jacobian_offset, -2.0]])

        checker = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=1, nvars=2, fun=value, jacobian=jacobian, bkd=bkd
            )
        )
        sample, direction = bkd.asarray([[0.7], [-0.4]]), bkd.asarray([[0.6], [0.8]])
        steps = bkd.asarray(self._STEPS)
        errors = checker.check_derivatives(
            sample, fd_eps=steps, direction=direction
        )[0]
        action = bkd.dot(jacobian(sample), direction)
        rounding = predicted_rounding(
            float(np.finfo(np.float64).eps),
            bkd.to_float(bkd.norm(value(sample))),
            bkd.to_float(bkd.norm(action)),
            bkd.to_float(bkd.norm(sample)),
            self._STEPS,
        )
        return checker.curve(errors, steps, rounding=rounding)

    def test_without_a_prediction_a_linear_quantity_has_no_v(
        self, bkd: Backend[Array]
    ) -> None:
        """The minimum is at the largest step, not where rounding puts a
        V's bottom, so the V test alone fails it."""
        curve = self._curve(bkd)
        report = FDCurve(curve.errors, curve.steps).v_shape()
        assert report.bottom_step == 1e-1
        assert not report.bottom_is_rounding_limited
        assert math.isnan(report.rounding_ratio)
        assert not report.passed

    def test_with_the_prediction_it_is_truncation_free(
        self, bkd: Backend[Array]
    ) -> None:
        report = self._curve(bkd).v_shape()
        assert report.rounding_ratio <= 10.0
        assert report.truncation_free and report.passed

    @pytest.mark.parametrize("offset", [1e-3, 1e-9, -1e-9])
    def test_a_defect_stands_out_against_rounding(
        self, bkd: Backend[Array], offset: float
    ) -> None:
        report = self._curve(bkd, jacobian_offset=offset).v_shape()
        assert not report.truncation_free
        assert not report.passed

    def test_the_rounding_factor_is_the_callers(self, bkd: Backend[Array]) -> None:
        """A relative defect of about 1e-10, some thousand times the
        rounding at the largest step: refused at the default factor,
        accepted by a caller allowing 1e5."""
        curve = self._curve(bkd, jacobian_offset=3e-11)
        assert not curve.v_shape().passed
        assert curve.v_shape(rounding_factor=1e5).passed


class TestBroadBottom:
    r"""A forward-difference curve recorded from a correct derivative
    (:math:`0.3\, p` of :math:`0.15\, p^\top p` under Torch), steps
    1e-1 down to 1e-13: the error falls a decade per decade down to
    1e-7, then lies flat near 1e-7 for two decades before rounding
    rises."""

    _ERRORS = (
        *(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
        *(1e-7, 9e-8, 2e-6, 5e-5, 4e-4, 9e-4),
    )
    _STEPS = tuple(10.0**-k for k in range(1, 14))
    _ROUNDING = tuple(1.1e-15 / h for h in _STEPS)

    def test_a_window_by_the_minimum_lands_on_the_flat_bottom(self) -> None:
        """Fitted 1 to 3 decades above the minimum (1e-9), the order is
        flattened by the transition to rounding and the curve fails."""
        report = FDCurve(self._ERRORS, self._STEPS).v_shape()
        assert report.order < 0.75 and not report.passed

    def test_a_window_by_the_prediction_fits_the_truncation_side(self) -> None:
        """The truncation side starts at 1e-6, the smallest step whose
        error exceeds 100 times its predicted rounding: fitted there, the
        order is 1."""
        report = FDCurve(self._ERRORS, self._STEPS, rounding=self._ROUNDING).v_shape()
        assert abs(report.order - 1.0) < 1e-12
        assert not report.truncation_free and report.passed


class TestFDCurve:
    def test_min_error(self) -> None:
        assert FDCurve((3.0, 1.0, 2.0), (1e-1, 1e-2, 1e-3)).min_error() == 1.0

    def test_rejects_errors_and_steps_of_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="same nonzero length"):
            FDCurve((1.0, 2.0), (1e-1,))

    def test_rejects_an_empty_curve(self) -> None:
        with pytest.raises(ValueError, match="same nonzero length"):
            FDCurve((), ())

    def test_rejects_a_prediction_of_another_length(self) -> None:
        with pytest.raises(ValueError, match="one entry per step"):
            FDCurve((1.0, 2.0), (1e-1, 1e-2), rounding=(1e-14,))

    def test_zero_errors_are_rounding(self) -> None:
        """Differences exact to the last bit: a ratio of zero, even against
        a zero prediction."""
        report = FDCurve((0.0, 0.0), (1e-1, 1e-2), rounding=(0.0, 0.0)).v_shape()
        assert report.rounding_ratio == 0.0 and report.passed

    def test_an_error_against_a_zero_prediction_is_not(self) -> None:
        report = FDCurve((1e-20, 0.0), (1e-1, 1e-2), rounding=(0.0, 0.0)).v_shape()
        assert math.isinf(report.rounding_ratio) and not report.truncation_free


class TestPredictedRounding:
    def test_relative_and_absolute(self, numpy_bkd: Backend[Array]) -> None:
        """Relative to an action of size 2: eps (2 |q| / 2 + |x|) / h;
        against a zero action, absolute: 2 eps |q| / h."""
        bkd, eps = numpy_bkd, 1e-16
        relative = predicted_rounding(eps, 4.0, 2.0, 1.0, [1e-1, 1e-2])
        bkd.assert_allclose(
            bkd.asarray(relative), bkd.asarray([5e-15, 5e-14]), rtol=1e-12
        )
        absolute = predicted_rounding(eps, 4.0, 0.0, 1.0, [1e-1])
        bkd.assert_allclose(bkd.asarray(absolute), bkd.asarray([8e-15]), rtol=1e-12)
