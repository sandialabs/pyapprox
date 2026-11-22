r"""The mass-form system of a collocation physics, no row treated.

- the mass is the diagonal of the physics' nodal weights, declared the
  identity when every weight is one and not SPD when a row is algebraic;
- refusals: negative or misshapen weights, a physics declaring none, and a
  physics second order in time;
- the right-hand side's bundle passes ``ODEResidualBundleChecker``, with
  and without parameters;
- backward Euler with the essential rows differentiated
  (``CoupledRows``) and pinned at the step end (``RowPinning``) reproduces
  the old collocation path's trajectory for advection-diffusion-reaction
  with constant Dirichlet ends;
- a step solves with algebraic (zero-weight) rows, which then hold
  :math:`f = 0`.
"""

import math
from typing import Generic, List

import numpy as np
import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.differentiated_rows import CoupledRows, differentiated_rows
from pyapprox.ode.integration import Integrator
from pyapprox.ode.pinned_rows import RowPinning
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.tableau import backward_euler
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.boundary import constant_dirichlet_bc
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.collocation.time_integration import (
    CollocationModel,
    TimeIntegrationConfig,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.right_hand_side import CollocationRightHandSide
from pyapprox.pde.models.collocation.unconstrained_system import (
    collocation_unconstrained_system,
)
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton

_NPTS = 12
_TIME = 0.3
_LEFT, _RIGHT = 1.0, -0.5


def _basis(bkd: Backend[Array]) -> ChebyshevBasis1D[Array]:
    return ChebyshevBasis1D(TransformedMesh1D(_NPTS, bkd), bkd)


def _adr(bkd: Backend[Array]) -> AdvectionDiffusionReaction[Array]:
    return AdvectionDiffusionReaction(
        _basis(bkd),
        bkd,
        diffusion=0.2,
        velocity=[bkd.full((_NPTS,), 0.5)],
        reaction=-0.4,
    )


class _Weighted(AdvectionDiffusionReaction[Array]):
    """ADR with weights a test states."""

    def __init__(self, bkd: Backend[Array], weights: Array) -> None:
        super().__init__(
            _basis(bkd),
            bkd,
            diffusion=0.2,
            velocity=[bkd.full((_NPTS,), 0.5)],
            reaction=-0.4,
        )
        self._weights = weights

    def interior_mass_weights(self) -> Array:
        return self._weights


class _Unweighted(Generic[Array]):
    """An interior operator declaring no mass weights."""

    def __init__(self, physics: AdvectionDiffusionReaction[Array]) -> None:
        self._physics = physics

    def bkd(self) -> Backend[Array]:
        return self._physics.bkd()

    def nstates(self) -> int:
        return self._physics.nstates()

    def interior_residual(self, state: Array, time: float) -> Array:
        return self._physics.interior_residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        return self._physics.interior_jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        return self._physics.interior_state_derivatives()

    def interior_is_time_invariant(self) -> bool:
        return self._physics.interior_is_time_invariant()

    def interior_jacobian_is_time_invariant(self) -> bool:
        return self._physics.interior_jacobian_is_time_invariant()


def _ends(bkd: Backend[Array]) -> DirichletRows[Array]:
    mesh = TransformedMesh1D(_NPTS, bkd)
    return DirichletRows(
        bkd,
        StateLayout.uniform(_NPTS, 1),
        [
            DirichletPart(mesh.boundary_indices(0), _LEFT),
            DirichletPart(mesh.boundary_indices(1), _RIGHT),
        ],
    )


def _state(bkd: Backend[Array], seed: int = 3) -> Array:
    return bkd.asarray(np.random.default_rng(seed).normal(0.0, 0.5, _NPTS))


class TestMass:
    def test_identity(self, bkd: Backend[Array]) -> None:
        system = collocation_unconstrained_system(CollocationSystem(_adr(bkd)))
        properties = system.mass.properties()
        assert properties.identity and properties.diagonal
        vec = _state(bkd)[:, None]
        bkd.assert_allclose(system.mass.apply(vec), vec)

    def test_algebraic_rows(self, bkd: Backend[Array]) -> None:
        """Zero weights: diagonal, symmetric, neither SPD nor the
        identity."""
        weights = bkd.asarray([1.0] * 4 + [0.0, 0.0] + [1.0] * (_NPTS - 6))
        system = collocation_unconstrained_system(
            CollocationSystem(_Weighted(bkd, weights))
        )
        properties = system.mass.properties()
        assert properties.diagonal and properties.symmetric
        assert not properties.spd and not properties.identity
        vec = _state(bkd)
        bkd.assert_allclose(system.mass.apply(vec[:, None])[:, 0], weights * vec)


class TestRefusals:
    def test_negative_weight(self, bkd: Backend[Array]) -> None:
        weights = bkd.asarray([1.0] * (_NPTS - 1) + [-1.0])
        with pytest.raises(ValueError, match="nonnegative"):
            collocation_unconstrained_system(CollocationSystem(_Weighted(bkd, weights)))

    def test_misshapen_weights(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="shape"):
            collocation_unconstrained_system(
                CollocationSystem(_Weighted(bkd, bkd.ones((_NPTS - 1,))))
            )

    def test_no_weights(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="mass weights"):
            collocation_unconstrained_system(CollocationSystem(_Unweighted(_adr(bkd))))

    def test_second_order_in_time(self, bkd: Backend[Array]) -> None:
        basis = ChebyshevBasis2D(TransformedMesh2D(4, 4, bkd), bkd)
        physics = LinearElasticityPhysics(basis, bkd, lamda=1.0, mu=1.0)
        with pytest.raises(TypeError, match="second order in time"):
            collocation_unconstrained_system(CollocationSystem(physics))


class TestBundle:
    def test_without_parameters(self, bkd: Backend[Array]) -> None:
        """The right-hand side is the raw interior operator's."""
        system = collocation_unconstrained_system(CollocationSystem(_adr(bkd)))
        rhs = system.rhs
        assert isinstance(rhs, CollocationRightHandSide)
        report = ODEResidualBundleChecker(rhs).check(_state(bkd), _TIME)
        assert report.failures() == []

    def test_with_parameters(self, bkd: Backend[Array]) -> None:
        physics = _adr(bkd)
        rng = np.random.default_rng(5)
        field_map = MeshKLEFieldMap(
            bkd,
            bkd.full((_NPTS,), 2.0),
            bkd.asarray(rng.uniform(-0.3, 0.3, (_NPTS, 3))),
        )
        parameters = ParameterState(
            CollocationAdvectionDiffusionParameterization(
                physics, diffusion_map=field_map, bkd=bkd
            )
        )
        system = collocation_unconstrained_system(
            CollocationSystem(physics), parameters
        )
        rhs = system.rhs
        assert isinstance(rhs, CollocationRightHandSide)
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd),
            _TIME,
            adj=_state(bkd, seed=6),
            param=bkd.asarray(rng.normal(0.0, 0.3, 3)),
        )
        assert report.failures() == []


def _backward_euler(
    bkd: Backend[Array], physics: AdvectionDiffusionReaction[Array], times: List[float]
) -> List[Array]:
    """The stored states of backward Euler on the new path, the ends
    differentiated and pinned at the step end."""
    rows = _ends(bkd)
    system = differentiated_rows(
        collocation_unconstrained_system(CollocationSystem(physics, rows)),
        rows,
        CoupledRows(),
    )
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
    stepper = RungeKuttaStepper(
        backward_euler(),
        system,
        RecomputeAlways(system.solver),
        newton,
        RowPinning(rows),
    )
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    # The linear interpolant of the end values plus a bump vanishing at
    # the ends: compatible with the data
    mesh = TransformedMesh1D(_NPTS, bkd)
    nodes = mesh.points()[0]
    left = bkd.to_float(nodes[mesh.boundary_indices(0)][0])
    right = bkd.to_float(nodes[mesh.boundary_indices(1)][0])
    slope = (_RIGHT - _LEFT) / (right - left)
    init = _LEFT + slope * (nodes - left) + bkd.sin(math.pi * nodes)
    Integrator(stepper).solve(
        bkd.reshape(init, (-1, 1)), GivenSizes.from_times(times), store
    )
    return [store.state(n)[:, 0] for n in range(store.ntimes())]


def test_backward_euler_reproduces_the_old_path(bkd: Backend[Array]) -> None:
    """Ten steps of 0.02: every stored state equal to the old path's to
    1e-12."""
    deltat, nsteps = 0.02, 10
    times = [deltat * n for n in range(nsteps + 1)]
    physics = _adr(bkd)
    new = _backward_euler(bkd, physics, times)
    old_physics = _adr(bkd)
    mesh = TransformedMesh1D(_NPTS, bkd)
    old_physics.set_boundary_conditions(
        [
            constant_dirichlet_bc(bkd, mesh.boundary_indices(0), _LEFT),
            constant_dirichlet_bc(bkd, mesh.boundary_indices(1), _RIGHT),
        ]
    )
    config: TimeIntegrationConfig[Array] = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=deltat * nsteps,
        deltat=deltat,
        newton_tol=1e-13,
        newton_maxiter=20,
        verbosity=0,
    )
    solutions, _ = CollocationModel(old_physics, bkd).solve_transient(new[0], config)
    assert solutions.shape == (_NPTS, nsteps + 1)
    for n, state in enumerate(new):
        bkd.assert_allclose(state, solutions[:, n], rtol=1e-12, atol=1e-12)


def test_algebraic_rows_hold_the_interior_equation(bkd: Backend[Array]) -> None:
    """Two interior rows of weight zero: one backward Euler step solves,
    and on those rows the new state satisfies f = 0."""
    weights = bkd.asarray([1.0] * 4 + [0.0, 0.0] + [1.0] * (_NPTS - 6))
    physics = _Weighted(bkd, weights)
    states = _backward_euler(bkd, physics, [0.0, 0.02])
    residual = physics.interior_residual(states[1], 0.02)
    bkd.assert_allclose(residual[4:6], bkd.zeros((2,)), atol=1e-11)
