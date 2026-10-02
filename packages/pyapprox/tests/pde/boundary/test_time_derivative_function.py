"""Tests for the time-derivative function view and its FD check.

Each check runs on a non-polynomial signal (FD error has a genuine
minimum near sqrt(machine eps)) and a polynomial one (FD exact at every
step), each with a wrong-derivative control that must fail.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pyapprox.interface.functions.protocols.objective import (
    ObjectiveProtocol,
)
from pyapprox.pde.boundary import (
    DirichletConstraintSet,
    DofSignal,
    TimeDerivativeFunction,
    time_derivative_functions,
)
from pyapprox.util.backends.protocols import Array, Backend

from tests._helpers.time_derivative_checks import (
    assert_time_derivatives_match,
)

_SCALE = np.array([1.0, -2.0, 0.5])
_OMEGA = 3.0


def _sine(t: float) -> NDArray[Any]:
    return _SCALE * np.sin(_OMEGA * t)


def _sine_dot(t: float) -> NDArray[Any]:
    return _SCALE * _OMEGA * np.cos(_OMEGA * t)


def _sine_ddot(t: float) -> NDArray[Any]:
    return -_SCALE * _OMEGA**2 * np.sin(_OMEGA * t)


def _sine_dot_missing_chain_rule(t: float) -> NDArray[Any]:
    return _SCALE * np.cos(_OMEGA * t)


def _cubic(t: float) -> NDArray[Any]:
    return _SCALE * t**3


def _cubic_dot(t: float) -> NDArray[Any]:
    return 3.0 * _SCALE * t**2


def _cubic_ddot(t: float) -> NDArray[Any]:
    return 6.0 * _SCALE * t


def _cubic_ddot_wrong(t: float) -> NDArray[Any]:
    return 3.0 * _SCALE * t


def _set(
    bkd: Backend[Array], *derivatives: Any, values: Any = _sine
) -> DirichletConstraintSet[Array]:
    from pyapprox.pde.galerkin.boundary import CallableDirichletBC

    bc = CallableDirichletBC([0, 2, 4], DofSignal(values, derivatives), bkd)
    return DirichletConstraintSet([bc], 6, bkd)


def _assert_set(cs: DirichletConstraintSet[Array], norders: int,
                bkd: Backend[Array]) -> None:
    assert_time_derivatives_match(
        cs.values, cs.values_derivative, norders, cs.ndofs(), bkd
    )


class TestTimeDerivativeFunction:
    def test_shapes_and_protocol(self, bkd: Backend[Array]) -> None:
        cs = _set(bkd, _sine_dot)
        derivative = cs.values_derivative(1)
        assert derivative is not None
        function = TimeDerivativeFunction(cs.values, derivative, 3, bkd)
        assert isinstance(function, ObjectiveProtocol)
        assert function.nvars() == 1 and function.nqoi() == 3
        assert function(bkd.asarray([[0.1, 0.2]])).shape == (3, 2)
        jac = function.jacobian(bkd.asarray([[0.1]]))
        bkd.assert_allclose(jac[:, 0], bkd.asarray(_sine_dot(0.1)))

    def test_wrong_ndofs_raises(self, bkd: Backend[Array]) -> None:
        cs = _set(bkd, _sine_dot)
        function = TimeDerivativeFunction(
            cs.values, cs.values, 2, bkd
        )
        with pytest.raises(ValueError, match="shape"):
            function.jacobian(bkd.asarray([[0.1]]))

    def test_absent_order_raises(self, bkd: Backend[Array]) -> None:
        cs = _set(bkd, _sine_dot)
        with pytest.raises(ValueError, match="order 2 is absent"):
            time_derivative_functions(
                cs.values, cs.values_derivative, 2, 3, bkd
            )


class TestTimeDerivativeCheck:
    def test_non_polynomial_orders_pass(self, bkd: Backend[Array]) -> None:
        _assert_set(_set(bkd, _sine_dot, _sine_ddot), 2, bkd)

    def test_non_polynomial_wrong_first_order_fails(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(AssertionError, match="order 1"):
            _assert_set(_set(bkd, _sine_dot_missing_chain_rule), 1, bkd)

    def test_polynomial_orders_pass(self, bkd: Backend[Array]) -> None:
        _assert_set(
            _set(bkd, _cubic_dot, _cubic_ddot, values=_cubic), 2, bkd
        )

    def test_polynomial_wrong_second_order_fails(
        self, bkd: Backend[Array]
    ) -> None:
        """Order 1 is right; only order 2 is wrong, and it is caught."""
        with pytest.raises(AssertionError, match="order 2"):
            _assert_set(
                _set(bkd, _cubic_dot, _cubic_ddot_wrong, values=_cubic),
                2,
                bkd,
            )
