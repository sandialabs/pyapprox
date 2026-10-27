"""Galerkin state constraints: the essential pin and the mass-consistent
projection.

1D advection-diffusion, P1 on 8 cells, with the time-varying Dirichlet
value g(t) = (1 + x) t^2 on the left. The pin sets the essential entries
to g(t) and leaves the rest; the mass-consistent projection also moves
the interior so that M_II (y_I - z_I) = M_ID (z_D - g), checked against
blocks of the dense consistent mass, with either row treatment's mass
doing the solves. Both are projections, and since both are affine their
tangents are exact for any perturbation, which the checks use instead of
finite differences.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from pyapprox.ode.protocols import StateConstraintProtocol
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.basis.mixed import MixedBasis
from pyapprox.pde.galerkin.basis.vector_lagrange import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    NeutralizedRows,
    RowTreatmentFactory,
    SymmetricNeutralizedRows,
    skfem_ode_residual,
)
from pyapprox.pde.galerkin.time_integration.state_constraints import (
    EssentialPin,
    MassConsistentProjection,
)
from pyapprox.util.backends.numpy import NumpyBkd

_Np = NDArray[Any]
_NPARAMS = 2


def _ramp(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * t**2
    return value


def _ramp_dot(x: _Np, t: float) -> _Np:
    value: _Np = 2.0 * (1.0 + x[0]) * t
    return value


def _forcing(x: _Np, t: float) -> _Np:
    value: _Np = np.sin(np.pi * x[0]) * (1.0 + t)
    return value


def _system(bkd: NumpyBkd) -> GalerkinSystem[_Np]:
    mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    bc = DirichletBC(
        basis,
        "left",
        BoundarySignal(TimeDependent(_ramp), [TimeDependent(_ramp_dot)]),
        bkd,
    )
    physics = AdvectionDiffusionReaction(
        basis=basis, diffusivity=0.1, bkd=bkd, forcing=TimeDependent(_forcing)
    )
    return compose_galerkin_system(physics, [bc])


def _vectors(n: int, ncols: int, seed: int) -> _Np:
    vectors: _Np = np.random.default_rng(seed).normal(size=(n, ncols))
    return vectors


class TestEssentialPin:
    def test_sets_the_essential_entries_and_keeps_the_rest(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        system = _system(numpy_bkd)
        pin = EssentialPin(system, _NPARAMS)
        assert isinstance(pin, StateConstraintProtocol)
        dofs = np.asarray(system.constraint_set().dofs())
        interior = np.setdiff1d(np.arange(system.nstates()), dofs)
        state = _vectors(system.nstates(), 1, 0)[:, 0]
        for time in (0.3, 0.8):
            pinned = pin.constrain(state, time)
            numpy_bkd.assert_allclose(
                pinned[dofs], system.constraint_set().values(time), rtol=0.0
            )
            numpy_bkd.assert_allclose(pinned[interior], state[interior], rtol=0.0)
            numpy_bkd.assert_allclose(pin.constrain(pinned, time), pinned, rtol=0.0)

    def test_tangent_is_exact_and_its_transpose_adjoint(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Affine, so Pi(z + d) - Pi(z) = Pi' d for any block d; and
        <Pi' x, y> = <x, Pi'^T y>."""
        system = _system(numpy_bkd)
        pin = EssentialPin(system, _NPARAMS)
        tangent = pin.derivatives().tangent
        assert tangent is not None
        n = system.nstates()
        state, direction = _vectors(n, 1, 1)[:, 0], _vectors(n, 3, 2)
        op = tangent(state, 0.5)
        for jj in range(3):
            difference = pin.constrain(state + direction[:, jj], 0.5) - pin.constrain(
                state, 0.5
            )
            numpy_bkd.assert_allclose(
                op.apply(direction)[:, jj], difference, rtol=1e-14, atol=1e-14
            )
        x, y = _vectors(n, 1, 3), _vectors(n, 1, 4)
        numpy_bkd.assert_allclose(
            np.sum(op.apply(x) * y), np.sum(x * op.apply_transpose(y)), rtol=1e-13
        )

    def test_parameter_and_second_derivatives_are_stated_zeros(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        system = _system(numpy_bkd)
        derivs = EssentialPin(system, _NPARAMS).derivatives()
        state, adj = np.zeros(system.nstates()), np.ones(system.nstates())
        param, state_state = derivs.param, derivs.state_state
        state_param, param_param = derivs.state_param, derivs.param_param
        assert param is not None and state_state is not None
        assert state_param is not None and param_param is not None
        assert param(state, 0.0).properties().zero
        for second in (state_state, state_param, param_param):
            assert second(state, 0.0, adj).properties().zero


_TREATMENTS = pytest.mark.parametrize(
    "treatment",
    [NeutralizedRows, SymmetricNeutralizedRows],
    ids=["coupled_mass", "symmetric_mass"],
)


def _projection(
    system: GalerkinSystem[_Np],
    treatment: RowTreatmentFactory[_Np],
    lumped_mass: bool = False,
) -> MassConsistentProjection[_Np]:
    adapter = skfem_ode_residual(
        system, lumped_mass=lumped_mass, row_treatment=treatment
    )
    return MassConsistentProjection(adapter, _NPARAMS)


class TestMassConsistentProjection:
    @_TREATMENTS
    def test_interior_rows_hold_with_the_exact_increment(
        self, numpy_bkd: NumpyBkd, treatment: RowTreatmentFactory[_Np]
    ) -> None:
        """y_D = g and M_II (y_I - z_I) = M_ID (z_D - g); a projection."""
        system = _system(numpy_bkd)
        projection = _projection(system, treatment)
        assert isinstance(projection, StateConstraintProtocol)
        raw_mass = system.mass_matrix()
        # Typed as a backend array; skfem assembles SciPy sparse.
        mass = (
            np.asarray(raw_mass.toarray())
            if isinstance(raw_mass, spmatrix)
            else np.asarray(raw_mass)
        )
        dofs = np.asarray(system.constraint_set().dofs())
        interior = np.setdiff1d(np.arange(system.nstates()), dofs)
        state = _vectors(system.nstates(), 1, 5)[:, 0]
        time = 0.7
        gvals = np.asarray(system.constraint_set().values(time))
        projected = projection.constrain(state, time)
        numpy_bkd.assert_allclose(projected[dofs], gvals, rtol=0.0)
        numpy_bkd.assert_allclose(
            mass[np.ix_(interior, interior)] @ (projected - state)[interior],
            mass[np.ix_(interior, dofs)] @ (state[dofs] - gvals),
            rtol=1e-10,
            atol=1e-14,
        )
        numpy_bkd.assert_allclose(
            projection.constrain(projected, time), projected, rtol=1e-13, atol=1e-15
        )

    @_TREATMENTS
    def test_tangent_is_exact_and_its_transpose_adjoint(
        self, numpy_bkd: NumpyBkd, treatment: RowTreatmentFactory[_Np]
    ) -> None:
        system = _system(numpy_bkd)
        projection = _projection(system, treatment)
        tangent = projection.derivatives().tangent
        assert tangent is not None
        n = system.nstates()
        state, direction = _vectors(n, 1, 6)[:, 0], _vectors(n, 3, 7)
        op = tangent(state, 0.4)
        applied = op.apply(direction)
        for jj in range(3):
            difference = projection.constrain(
                state + direction[:, jj], 0.4
            ) - projection.constrain(state, 0.4)
            numpy_bkd.assert_allclose(
                applied[:, jj], difference, rtol=1e-12, atol=1e-14
            )
        x, y = _vectors(n, 2, 8), _vectors(n, 2, 9)
        numpy_bkd.assert_allclose(
            np.sum(op.apply(x) * y), np.sum(x * op.apply_transpose(y)), rtol=1e-12
        )

    def test_lumped_mass_is_the_pin(self, numpy_bkd: NumpyBkd) -> None:
        """A lumped mass has M_ID = 0: nothing to correct."""
        system = _system(numpy_bkd)
        projection = _projection(system, SymmetricNeutralizedRows, lumped_mass=True)
        pin = EssentialPin(system, _NPARAMS)
        state = _vectors(system.nstates(), 1, 10)[:, 0]
        numpy_bkd.assert_allclose(
            projection.constrain(state, 0.6), pin.constrain(state, 0.6), rtol=0.0
        )
        assert "corrects=False" in repr(projection)

    def test_differs_from_the_pin_with_a_consistent_mass(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Control: the interior correction is not negligible here."""
        system = _system(numpy_bkd)
        projection = _projection(system, SymmetricNeutralizedRows)
        pin = EssentialPin(system, _NPARAMS)
        state = _vectors(system.nstates(), 1, 11)[:, 0]
        gap = np.abs(projection.constrain(state, 0.6) - pin.constrain(state, 0.6))
        assert gap.max() > 1e-2


def _ramp_2d(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0] + 2.0 * x[1]) * t**2
    return value


def _ramp_2d_dot(x: _Np, t: float) -> _Np:
    value: _Np = 2.0 * (1.0 + x[0] + 2.0 * x[1]) * t
    return value


def _forcing_2d(x: _Np, t: float) -> _Np:
    value: _Np = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * (1.0 + t)
    return value


def _system_2d(bkd: NumpyBkd) -> GalerkinSystem[_Np]:
    """2D advection-diffusion on a 3x3 P1 grid with time-varying Dirichlet
    data on all four sides: each corner DOF belongs to two boundaries."""
    mesh = StructuredMesh2D(3, 3, [(0.0, 1.0), (0.0, 1.0)], bkd)
    basis = LagrangeBasis(mesh, degree=1)
    signal = BoundarySignal(TimeDependent(_ramp_2d), [TimeDependent(_ramp_2d_dot)])
    physics = AdvectionDiffusionReaction(
        basis=basis, diffusivity=0.1, bkd=bkd, forcing=TimeDependent(_forcing_2d)
    )
    return compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, name, signal, bkd)
            for name in ("left", "right", "bottom", "top")
        ],
    )


def _inlet(x: _Np, t: float) -> _Np:
    values = np.zeros((2, x.shape[1]))
    values[0] = x[1] * (1.0 - x[1]) * t**2
    return values


def _inlet_dot(x: _Np, t: float) -> _Np:
    values = np.zeros((2, x.shape[1]))
    values[0] = 2.0 * x[1] * (1.0 - x[1]) * t
    return values


def _stokes(bkd: NumpyBkd) -> GalerkinSystem[_Np]:
    """2D Stokes, P2 velocity and P1 pressure on a 3x3 grid, time-varying
    velocity data on every side: the pressure rows carry no mass."""
    mesh = StructuredMesh2D(3, 3, [(0.0, 1.0), (0.0, 1.0)], bkd)
    fields: List[Tuple[str, Any]] = [
        ("velocity", VectorLagrangeBasis(mesh, degree=2)),
        ("pressure", LagrangeBasis(mesh, degree=1)),
    ]
    basis = MixedBasis(fields)
    signal = BoundarySignal(TimeDependent(_inlet), [TimeDependent(_inlet_dot)])
    return compose_galerkin_system(
        StokesPhysics(basis, bkd, viscosity=1.0),
        [
            DirichletBC(basis.field("velocity"), name, signal, bkd)
            for name in ("left", "right", "bottom", "top")
        ],
    )


def _dense_mass(system: GalerkinSystem[_Np]) -> _Np:
    raw_mass = system.mass_matrix()
    return (
        np.asarray(raw_mass.toarray())
        if isinstance(raw_mass, spmatrix)
        else np.asarray(raw_mass)
    )


class TestTwoDimensions:
    def test_shared_corners_take_the_constraint_sets_values(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """Pin and projection set every essential entry, corners included,
        to the constraint set's resolved values; the projection's interior
        identity holds and it is a projection."""
        system = _system_2d(numpy_bkd)
        cs = system.constraint_set()
        dofs = np.asarray(cs.dofs())
        interior = np.setdiff1d(np.arange(system.nstates()), dofs)
        mass = _dense_mass(system)
        state = _vectors(system.nstates(), 1, 12)[:, 0]
        time = 0.6
        gvals = np.asarray(cs.values(time))
        projection = _projection(system, SymmetricNeutralizedRows)
        constraints: List[StateConstraintProtocol[_Np]] = [
            EssentialPin(system, _NPARAMS),
            projection,
        ]
        for constraint in constraints:
            numpy_bkd.assert_allclose(
                constraint.constrain(state, time)[dofs], gvals, rtol=0.0
            )
        projected = projection.constrain(state, time)
        numpy_bkd.assert_allclose(
            mass[np.ix_(interior, interior)] @ (projected - state)[interior],
            mass[np.ix_(interior, dofs)] @ (state[dofs] - gvals),
            rtol=1e-10,
            atol=1e-14,
        )
        numpy_bkd.assert_allclose(
            projection.constrain(projected, time), projected, rtol=1e-13, atol=1e-15
        )

    def test_stokes_corrects_only_the_velocity(self, numpy_bkd: NumpyBkd) -> None:
        """The pressure rows carry no mass, so the interior mass block is
        singular there: the projection solves on the velocity block, leaves
        the pressure as given, and the velocity rows hold with the exact
        boundary increment; the tangent stays exact."""
        system = _stokes(numpy_bkd)
        projection = _projection(system, SymmetricNeutralizedRows)
        mass = _dense_mass(system)
        dofs = np.asarray(system.constraint_set().dofs())
        massless = np.flatnonzero(np.abs(mass).sum(axis=1) == 0.0)
        assert massless.size > 0
        velocity = np.setdiff1d(
            np.setdiff1d(np.arange(system.nstates()), dofs), massless
        )
        state = _vectors(system.nstates(), 1, 13)[:, 0]
        time = 0.5
        gvals = np.asarray(system.constraint_set().values(time))
        projected = projection.constrain(state, time)
        numpy_bkd.assert_allclose(projected[massless], state[massless], rtol=0.0)
        numpy_bkd.assert_allclose(
            mass[np.ix_(velocity, velocity)] @ (projected - state)[velocity],
            mass[np.ix_(velocity, dofs)] @ (state[dofs] - gvals),
            rtol=1e-10,
            atol=1e-14,
        )
        tangent = projection.derivatives().tangent
        assert tangent is not None
        direction = _vectors(system.nstates(), 1, 14)
        numpy_bkd.assert_allclose(
            tangent(state, time).apply(direction)[:, 0],
            projection.constrain(state + direction[:, 0], time)
            - projection.constrain(state, time),
            rtol=1e-12,
            atol=1e-13,
        )
