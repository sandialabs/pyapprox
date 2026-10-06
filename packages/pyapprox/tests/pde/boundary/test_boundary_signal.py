"""Tests for BoundarySignal: declared time dependence, derivatives by order."""

import pickle
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    TimeDependent,
    TimeIndependent,
)
from pyapprox.util.backends.numpy import NumpyBkd

_COORDS = np.array([[0.0, 0.5, 1.0], [1.0, 2.0, 3.0]])


def _steady(x: NDArray[Any]) -> NDArray[Any]:
    return x[0] + x[1]


def _steady_vector(x: NDArray[Any]) -> NDArray[Any]:
    return np.vstack([x[0], -x[1]])


def _cubic(x: NDArray[Any], t: float) -> NDArray[Any]:
    return x[0] * t**3


def _cubic_dot(x: NDArray[Any], t: float) -> NDArray[Any]:
    return 3.0 * x[0] * t**2


def _cubic_ddot(x: NDArray[Any], t: float) -> NDArray[Any]:
    return 6.0 * x[0] * t


class TestBoundarySignal:
    def test_constant_has_exact_zero_derivatives(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        signal = BoundarySignal(2.5)
        assert not signal.is_time_dependent()
        numpy_bkd.assert_allclose(
            signal.values()(_COORDS, 0.3), np.full(3, 2.5)
        )
        for order in (1, 2, 5):
            derivative = signal.time_derivative(order)
            assert derivative is not None
            numpy_bkd.assert_allclose(derivative(_COORDS, 0.3), np.zeros(3))

    def test_bare_one_argument_callable_is_time_independent(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        signal = BoundarySignal(_steady)
        assert not signal.is_time_dependent()
        derivative = signal.time_derivative(1)
        assert derivative is not None
        numpy_bkd.assert_allclose(derivative(_COORDS, 1.0), np.zeros(3))

    def test_vector_zero_derivative_matches_shape(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        derivative = BoundarySignal(
            TimeIndependent(_steady_vector)
        ).time_derivative(2)
        assert derivative is not None
        numpy_bkd.assert_allclose(derivative(_COORDS, 1.0), np.zeros((2, 3)))

    def test_bare_callable_that_could_take_time_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="ambiguous"):
            BoundarySignal(_cubic)

    def test_undeclared_derivative_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="ambiguous"):
            BoundarySignal(TimeDependent(_cubic), [_cubic_dot])

    def test_derivatives_of_time_independent_values_raise(self) -> None:
        with pytest.raises(ValueError, match="time-independent"):
            BoundarySignal(
                TimeIndependent(_steady), [TimeDependent(_cubic_dot)]
            )

    def test_orders_by_index_and_missing_order_is_none(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        signal = BoundarySignal(
            TimeDependent(_cubic),
            [TimeDependent(_cubic_dot), TimeDependent(_cubic_ddot)],
        )
        assert signal.is_time_dependent()
        assert signal.norders() == 2
        first = signal.time_derivative(1)
        second = signal.time_derivative(2)
        assert first is not None and second is not None
        numpy_bkd.assert_allclose(first(_COORDS, 2.0), 12.0 * _COORDS[0])
        numpy_bkd.assert_allclose(second(_COORDS, 2.0), 12.0 * _COORDS[0])
        assert signal.time_derivative(3) is None

    def test_time_dependent_without_derivatives_has_none(self) -> None:
        assert BoundarySignal(TimeDependent(_cubic)).time_derivative(1) is None

    def test_invalid_order_raises(self) -> None:
        with pytest.raises(ValueError, match="order"):
            BoundarySignal(1.0).time_derivative(0)

    @pytest.mark.parametrize(
        "signal",
        [
            BoundarySignal(1.0),
            BoundarySignal(TimeIndependent(_steady)),
            BoundarySignal(TimeDependent(_cubic), [TimeDependent(_cubic_dot)]),
        ],
    )
    def test_pickle_round_trip(
        self, numpy_bkd: NumpyBkd, signal: BoundarySignal
    ) -> None:
        restored = pickle.loads(pickle.dumps(signal))
        assert restored.is_time_dependent() == signal.is_time_dependent()
        numpy_bkd.assert_allclose(
            restored.values()(_COORDS, 0.4), signal.values()(_COORDS, 0.4)
        )
        derivative = signal.time_derivative(1)
        restored_derivative = restored.time_derivative(1)
        assert derivative is not None and restored_derivative is not None
        numpy_bkd.assert_allclose(
            restored_derivative(_COORDS, 0.4), derivative(_COORDS, 0.4)
        )
