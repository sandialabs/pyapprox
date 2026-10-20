"""SemiExplicitSystem and its builders, on Lotka-Volterra with a mass.

The shifted Jacobian is checked as the directional derivative of the
residual along :math:`(d, \\sigma d)`, which needs no derivation; the
dense and matrix-free systems against each other and against today's
Newton matrix :math:`M - cJ` (with :math:`\\sigma = 1/c`); and a
singular mass, the differential-algebraic case, by solving with it.
"""

from typing import TypeVar

import numpy as np
import pytest

from pyapprox.ode.dense_jacobian_operator import DenseJacobianOperator
from pyapprox.ode.matrix_free_jacobian_operator import MatrixFreeJacobianOperator
from pyapprox.ode.protocols import ImplicitSystemProtocol
from pyapprox.ode.semi_explicit_system import (
    SemiExplicitSystem,
    dense_system,
    matrix_free_system,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    DenseFormat,
)
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    LinearizationPoint,
    MatrixFree,
    SpaceDescriptor,
    SpaceMismatchError,
)
from tests._helpers.ode_models import LotkaVolterra

_POINT = LinearizationPoint(0, ("step", 1))
_TIME, _SHIFT = 0.3, 4.0
_STATE = SpaceDescriptor("state", 2)
F = TypeVar("F", bound=FormatProtocol)


def _model(bkd: Backend[Array], mass: list[list[float]]) -> LotkaVolterra[Array]:
    return LotkaVolterra(bkd, mass=bkd.asarray(mass))


_MASS = [[2.0, 0.5], [0.5, 1.0]]
_SINGULAR = [[1.0, 0.0], [0.0, 0.0]]


def _state(bkd: Backend[Array]) -> tuple[Array, Array]:
    return bkd.asarray([10.0, 5.0]), bkd.asarray([0.7, -1.3])


def _systems(
    bkd: Backend[Array], mass: list[list[float]]
) -> tuple[
    SemiExplicitSystem[Array, DenseFormat[Array]],
    SemiExplicitSystem[Array, MatrixFree],
]:
    model = _model(bkd, mass)
    return dense_system(model), matrix_free_system(model, nstates=2)


def _assert_solves(
    bkd: Backend[Array],
    system: SemiExplicitSystem[Array, F],
    y: Array,
    ydot: Array,
    b: Array,
    expected: Array,
) -> None:
    """The system's own solver solves its shifted Jacobian."""
    op = system.shifted_jacobian(y, ydot, _TIME, _SHIFT, _POINT)
    x, info = system.solver().setup(op).solve(b)
    assert info.converged
    bkd.assert_allclose(x, expected, rtol=1e-9)


class TestSemiExplicitSystem:
    def test_satisfies_the_protocol(self, bkd: Backend[Array]) -> None:
        dense, free = _systems(bkd, _MASS)
        stepped: ImplicitSystemProtocol[Array, Array, DenseFormat[Array]] = dense
        assert isinstance(stepped, ImplicitSystemProtocol)
        assert isinstance(free, ImplicitSystemProtocol)

    def test_residual_is_mass_rate_minus_rhs(self, bkd: Backend[Array]) -> None:
        y, ydot = _state(bkd)
        model = _model(bkd, _MASS)
        expected = bkd.dot(bkd.asarray(_MASS), ydot) - model(y, _TIME)
        for system in _systems(bkd, _MASS):
            bkd.assert_allclose(system.residual(y, ydot, _TIME), expected, rtol=1e-14)

    def test_shifted_jacobian_is_the_directional_derivative(
        self, bkd: Backend[Array]
    ) -> None:
        r""":math:`\frac{d}{d\epsilon} F(y + \epsilon d, \dot y + \sigma\epsilon d)`
        by central differences."""
        y, ydot = _state(bkd)
        d = bkd.asarray([0.4, -0.9])
        step = 1e-6
        for system in _systems(bkd, _MASS):
            difference = (
                system.residual(y + step * d, ydot + _SHIFT * step * d, _TIME)
                - system.residual(y - step * d, ydot - _SHIFT * step * d, _TIME)
            ) / (2 * step)
            op = system.shifted_jacobian(y, ydot, _TIME, _SHIFT, _POINT)
            assert op.linearization() == _POINT
            bkd.assert_allclose(op.apply(d[:, None])[:, 0], difference, rtol=1e-7)

    def test_scaled_shift_is_todays_newton_matrix(self, bkd: Backend[Array]) -> None:
        r""":math:`c(\sigma M - J) = M - cJ` with :math:`\sigma = 1/c`, for
        both systems, actions and transposes."""
        y, ydot = _state(bkd)
        coefficient = 0.25
        model = _model(bkd, _MASS)
        newton = bkd.asarray(_MASS) - coefficient * model.jacobian(y, _TIME)
        x = bkd.asarray(np.random.default_rng(0).normal(size=(2, 3)))
        for system in _systems(bkd, _MASS):
            op = system.shifted_jacobian(y, ydot, _TIME, 1.0 / coefficient, _POINT)
            bkd.assert_allclose(coefficient * op.apply(x), bkd.dot(newton, x))
            bkd.assert_allclose(
                coefficient * op.apply_transpose(x), bkd.dot(newton.T, x)
            )

    @pytest.mark.parametrize("mass", [_MASS, _SINGULAR], ids=["mass", "singular"])
    def test_default_solvers_solve_the_shifted_system(
        self, bkd: Backend[Array], mass: list[list[float]]
    ) -> None:
        """Dense LU and GMRES agree with a dense solve; the singular mass
        (a differential-algebraic model) is never inverted."""
        y, ydot = _state(bkd)
        model = _model(bkd, mass)
        reference = _SHIFT * bkd.asarray(mass) - model.jacobian(y, _TIME)
        b = bkd.asarray([[1.0], [-2.0]])
        expected = bkd.solve(reference, b)
        dense, free = _systems(bkd, mass)
        _assert_solves(bkd, dense, y, ydot, b, expected)
        _assert_solves(bkd, free, y, ydot, b, expected)

    def test_refuses_mismatches(self, bkd: Backend[Array]) -> None:
        model = _model(bkd, _MASS)
        dense, _ = _systems(bkd, _MASS)
        jacobian = DenseJacobianOperator(model, _STATE, bkd)
        other = BackendDenseOperator(
            bkd.eye(3, dtype=bkd.default_dtype()), SpaceDescriptor("w", 3), bkd
        )
        with pytest.raises(SpaceMismatchError):
            SemiExplicitSystem(
                model, other, jacobian, dense.space(), BackendDenseLU(bkd)
            )
        with pytest.raises(TypeError, match="solver"):
            SemiExplicitSystem(
                model,
                dense.mass(),
                jacobian,
                dense.space(),
                model,  # type: ignore[arg-type]
            )

    def test_mixed_formats_are_refused_by_a_direct_solver(
        self, bkd: Backend[Array]
    ) -> None:
        """A dense mass with a matrix-free Jacobian is a type error, and a
        dense solver refuses the sum at setup."""
        model = _model(bkd, _MASS)
        dense, _ = _systems(bkd, _MASS)
        matrix_free = MatrixFreeJacobianOperator(model, _STATE)
        mixed = SemiExplicitSystem(  # type: ignore[misc]
            model,
            dense.mass(),
            matrix_free,
            dense.space(),
            BackendDenseLU(bkd),
        )
        y, ydot = _state(bkd)
        op = mixed.shifted_jacobian(y, ydot, _TIME, _SHIFT, _POINT)
        with pytest.raises(TypeError, match="different formats"):
            mixed.solver().setup(op)
