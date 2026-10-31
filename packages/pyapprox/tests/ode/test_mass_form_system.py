"""The mass-form system takes its Jacobian from the residual alone.

On M y' = c K y (the shared 3-state LINEAR_MASS and LINEAR_K), both
backends:

- the system's Jacobian is the residual bundle's, c K;
- after the residual replaces its bundle (c from 1 to 2.5), a system built
  before the change gives the new Jacobian, so it holds no copy that could
  disagree with the residual;
- a residual whose bundle has no Jacobian is refused at construction.
"""

from typing import Generic

import pytest

from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.mass_matrix import ConstantDenseMassMatrix
from pyapprox.ode.residual_derivatives import (
    ODEResidualDerivatives,
    backend_state_space,
    dense_jacobian,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import LinearizationPoint
from tests._helpers.ode.runge_kutta_cases import LINEAR_K, LINEAR_MASS

_Bundle = ODEResidualDerivatives[Array, DenseFormat[Array]]


class _ScaledMatrix(Generic[Array]):
    """``(state, time) ->`` a fixed scale of ``LINEAR_K``."""

    def __init__(self, bkd: Backend[Array], scale: float) -> None:
        self._matrix = scale * bkd.asarray(LINEAR_K)

    def __call__(self, state: Array, time: float) -> Array:
        return self._matrix


class _Scaled(Generic[Array]):
    """:math:`f = c K y`, whose bundle is rebuilt when :math:`c` changes."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._mass = ConstantDenseMassMatrix(bkd.asarray(LINEAR_MASS), bkd)
        self._scale = 1.0
        self._bundle = self._build()

    def _build(self) -> _Bundle[Array]:
        states = backend_state_space(self._bkd, 3)
        matrix = _ScaledMatrix(self._bkd, self._scale)
        return ODEResidualDerivatives.state_only(
            dense_jacobian(matrix, states, self._bkd)
        )

    def set_scale(self, scale: float) -> None:
        self._scale = scale
        self._bundle = self._build()

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, state: Array, time: float) -> Array:
        return self._scale * self._bkd.dot(self._bkd.asarray(LINEAR_K), state)

    def mass_matrix(self) -> ConstantDenseMassMatrix[Array]:
        return self._mass

    def derivatives(self) -> _Bundle[Array]:
        return self._bundle


class _NoJacobian(_Scaled[Array]):
    """The same residual with a bundle declaring no derivatives."""

    def derivatives(self) -> _Bundle[Array]:
        return ODEResidualDerivatives.none()


def _matrix(
    bkd: Backend[Array],
    system: MassFormSystem[Array, Array, DenseFormat[Array]],
) -> Array:
    state = bkd.asarray([1.0, -0.5, 2.0])
    return system.jacobian(state, 0.2, LinearizationPoint(0)).format().matrix()


class TestTheJacobianIsTheResiduals:
    def test_it_is_the_bundles(self, bkd: Backend[Array]) -> None:
        system = dense_mass_form_system(_Scaled(bkd))
        bkd.assert_allclose(_matrix(bkd, system), bkd.asarray(LINEAR_K), rtol=0.0)

    def test_a_replaced_bundle_is_seen(self, bkd: Backend[Array]) -> None:
        residual: _Scaled[Array] = _Scaled(bkd)
        system = dense_mass_form_system(residual)
        residual.set_scale(2.5)
        bkd.assert_allclose(_matrix(bkd, system), 2.5 * bkd.asarray(LINEAR_K), rtol=0.0)

    def test_a_residual_without_a_jacobian_is_refused(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(TypeError, match="has no jacobian"):
            dense_mass_form_system(_NoJacobian(bkd))
