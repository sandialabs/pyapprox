"""Tests for composed Galerkin systems."""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.boundary import DirichletConstraintSet, NaturalBCOperator
from pyapprox.pde.galerkin.boundary import (
    DirectDirichletBC,
)
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinSteadySystemProtocol,
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.spatial_operator import ComposedSpatialOperator
from pyapprox.pde.galerkin.system import GalerkinSteadySystem, GalerkinSystem
from pyapprox.util.backends.protocols import Array, Backend


class _StubInterior(Generic[Array]):
    """``F_Omega(u) = -2 u``: an interior with no base class."""

    def __init__(self, nstates: int, bkd: Backend[Array]) -> None:
        self._nstates = nstates
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nstates(self) -> int:
        return self._nstates

    def interior_residual(self, state: Array, time: float) -> Array:
        return -2.0 * state

    def interior_jacobian(self, state: Array, time: float) -> Array:
        return -2.0 * self._bkd.eye(self._nstates)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        return StateDerivatives.linear(self._bkd)

    def interior_is_time_invariant(self) -> bool:
        return True


class _ScaledMass(Generic[Array]):
    """A mass provider whose matrix changes when its scale is reset."""

    def __init__(self, nstates: int, bkd: Backend[Array]) -> None:
        self._nstates = nstates
        self._bkd = bkd
        self._scale = 1.0

    def set_scale(self, scale: float) -> None:
        self._scale = scale

    def mass_matrix(self) -> Array:
        return self._scale * self._bkd.eye(self._nstates)


class _StubBasis:
    """Linear elements on a line of ``nstates`` nodes."""

    def __init__(self, nstates: int) -> None:
        self._nstates = nstates

    def element_dofs(self) -> NDArray[np.int64]:
        nodes = np.arange(self._nstates, dtype=np.int64)
        return np.stack([nodes[:-1], nodes[1:]], axis=1)


def _operator(
    bkd: Backend[Array], nstates: int = 4
) -> ComposedSpatialOperator[Array]:
    return ComposedSpatialOperator(
        _StubInterior(nstates, bkd), NaturalBCOperator([])
    )


def _constraints(
    bkd: Backend[Array], dofs: Any = (0,), nstates: int = 4
) -> DirichletConstraintSet[Array]:
    bc = DirectDirichletBC(list(dofs), [1.0] * len(dofs), bkd)
    return DirichletConstraintSet([bc], nstates, bkd)


class TestGalerkinSystem:
    def test_steady_system_has_no_mass(self, bkd: Backend[Array]) -> None:
        system = GalerkinSteadySystem(_operator(bkd), _constraints(bkd))
        assert isinstance(system, GalerkinSteadySystemProtocol)
        assert not isinstance(system, GalerkinTransientSystemProtocol)
        assert system.nstates() == 4

    def test_transient_system_satisfies_both(
        self, bkd: Backend[Array]
    ) -> None:
        basis = _StubBasis(4)
        system = GalerkinSystem(
            _operator(bkd), _constraints(bkd), _ScaledMass(4, bkd), basis
        )
        assert isinstance(system, GalerkinTransientSystemProtocol)
        assert isinstance(system, GalerkinSteadySystemProtocol)
        assert system.basis() is basis

    def test_mass_is_read_on_every_call(self, bkd: Backend[Array]) -> None:
        """A parameterized mass is never a stale copy."""
        mass = _ScaledMass(4, bkd)
        system = GalerkinSystem(
            _operator(bkd), _constraints(bkd), mass, _StubBasis(4)
        )
        mass.set_scale(3.0)
        bkd.assert_allclose(system.mass_matrix(), 3.0 * bkd.eye(4))

    def test_rejects_parts_of_the_wrong_kind(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(TypeError, match="spatial_operator"):
            GalerkinSteadySystem(object(), _constraints(bkd))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="constraint_set"):
            GalerkinSteadySystem(_operator(bkd), object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="mass"):
            GalerkinSystem(
                _operator(bkd),
                _constraints(bkd),
                object(),  # type: ignore[arg-type]
                _StubBasis(4),
            )
        with pytest.raises(TypeError, match="basis"):
            GalerkinSystem(
                _operator(bkd),
                _constraints(bkd),
                _ScaledMass(4, bkd),
                object(),  # type: ignore[arg-type]
            )

    def test_rejects_constraint_outside_operator(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(ValueError, match="constrains DOF 5"):
            GalerkinSteadySystem(
                _operator(bkd), _constraints(bkd, dofs=(5,), nstates=6)
            )

    def test_rejects_mass_of_wrong_shape(self, bkd: Backend[Array]) -> None:
        system = GalerkinSystem(
            _operator(bkd), _constraints(bkd), _ScaledMass(3, bkd), _StubBasis(4)
        )
        with pytest.raises(ValueError, match="mass matrix has shape"):
            system.mass_matrix()
