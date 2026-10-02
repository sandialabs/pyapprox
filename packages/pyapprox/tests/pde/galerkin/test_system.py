"""Tests for composed Galerkin systems and physics.system()."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.boundary import DirichletConstraintSet, NaturalBCOperator
from pyapprox.pde.constitutive.coefficient_functions import TimeIndependent
from pyapprox.pde.galerkin.basis import LagrangeBasis, VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary import (
    DirectDirichletBC,
    DirichletBC,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.physics.euler_bernoulli import (
    EulerBernoulliBeamFEM,
)
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
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
        system = GalerkinSystem(
            _operator(bkd), _constraints(bkd), _ScaledMass(4, bkd)
        )
        assert isinstance(system, GalerkinTransientSystemProtocol)
        assert isinstance(system, GalerkinSteadySystemProtocol)

    def test_mass_is_read_on_every_call(self, bkd: Backend[Array]) -> None:
        """A parameterized mass is never a stale copy."""
        mass = _ScaledMass(4, bkd)
        system = GalerkinSystem(_operator(bkd), _constraints(bkd), mass)
        mass.set_scale(3.0)
        bkd.assert_allclose(system.mass_matrix(), 3.0 * bkd.eye(4))

    def test_rejects_parts_of_the_wrong_kind(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(TypeError, match="spatial_operator"):
            GalerkinSteadySystem(object(), _constraints(bkd))
        with pytest.raises(TypeError, match="constraint_set"):
            GalerkinSteadySystem(_operator(bkd), object())
        with pytest.raises(TypeError, match="mass"):
            GalerkinSystem(_operator(bkd), _constraints(bkd), object())

    def test_rejects_constraint_outside_operator(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(ValueError, match="constrains DOF 5"):
            GalerkinSteadySystem(
                _operator(bkd), _constraints(bkd, dofs=(5,), nstates=6)
            )

    def test_rejects_mass_of_wrong_shape(self, bkd: Backend[Array]) -> None:
        system = GalerkinSystem(
            _operator(bkd), _constraints(bkd), _ScaledMass(3, bkd)
        )
        with pytest.raises(ValueError, match="mass matrix has shape"):
            system.mass_matrix()


def _zero(x: NDArray[Any]) -> NDArray[Any]:
    return np.zeros(x.shape[1])


def _one(x: NDArray[Any]) -> NDArray[Any]:
    return np.ones_like(x)


class TestPhysicsSystem:
    def _check(self, physics: Any, bkd: Backend[Array]) -> None:
        system = physics.system()
        assert isinstance(system, GalerkinTransientSystemProtocol)
        assert system.nstates() == physics.nstates()
        assert system.constraint_set() is physics.constraint_set()
        state = bkd.linspace(-1.0, 1.0, physics.nstates())
        bkd.assert_allclose(
            system.spatial_operator().spatial_residual(state, 0.0),
            physics.spatial_residual(state, 0.0),
        )
        assert system.mass_matrix() is physics.mass_matrix()

    def test_adr(self, bkd: Backend[Array]) -> None:
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=1.0,
            bkd=bkd,
            forcing=TimeIndependent(_zero),
            boundary_conditions=[
                DirichletBC(basis, "left", 1.0, bkd),
                RobinBC(basis, "right", alpha=2.0, value_func=1.0, bkd=bkd),
            ],
        )
        self._check(physics, bkd)
        assert physics.system() is physics.system()
        assert physics.system().spatial_operator() is physics.spatial_operator()

    def test_stokes(self, numpy_bkd: Backend[Array]) -> None:
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=numpy_bkd)
        physics = StokesPhysics(
            vel_basis=VectorLagrangeBasis(mesh, degree=2),
            pres_basis=LagrangeBasis(mesh, degree=1),
            bkd=numpy_bkd,
        )
        self._check(physics, numpy_bkd)

    def test_beam(self, numpy_bkd: Backend[Array]) -> None:
        physics = EulerBernoulliBeamFEM(
            nx=5, length=1.0, EI=1.0, load_func=_one, bkd=numpy_bkd
        )
        self._check(physics, numpy_bkd)
