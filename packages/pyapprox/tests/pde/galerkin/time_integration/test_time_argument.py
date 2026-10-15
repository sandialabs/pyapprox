"""Time reaches the ODE residual as an argument, never as stored state.

A residual that stored its time could answer a query at one time with
data left over from another. Steppers evaluate ``f`` at several times
inside one step (Crank-Nicolson at both ends, implicit midpoint at the
midpoint, Heun at each stage), so each check alternates times and
compares every evaluation against the spatial operator evaluated
directly at the time it was asked for. The forcing and the Dirichlet
signal both vary in time, so the interior and the essential rows each
expose a wrong time.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.explicit_steppers import HeunStepper
from pyapprox.ode.implicit_steppers import (
    CrankNicolsonStepper,
    ImplicitMidpointStepper,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.util.backends.protocols import Array, Backend


def _forcing(x: NDArray[Any], t: float) -> NDArray[Any]:
    return np.sin(np.pi * x[0]) * (1.0 + 3.0 * t)


def _ramp(x: NDArray[Any], t: float) -> NDArray[Any]:
    return (1.0 + x[0]) * t**2


def _ramp_dot(x: NDArray[Any], t: float) -> NDArray[Any]:
    return 2.0 * (1.0 + x[0]) * t


def _system(bkd: Backend[Array]) -> GalerkinSystem[Array]:
    mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    bc = DirichletBC(
        basis,
        "left",
        BoundarySignal(TimeDependent(_ramp), [TimeDependent(_ramp_dot)]),
        bkd,
    )
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=0.1,
        bkd=bkd,
        forcing=TimeDependent(_forcing),
    )
    return compose_galerkin_system(physics, [bc])


def _reference_rhs(system: GalerkinSystem[Array], state: Array, time: float) -> Array:
    """The BC-neutralized ``f(y, t)``, built from the system directly."""
    bkd = system.bkd()
    constraint_set = system.constraint_set()
    velocity = constraint_set.values_derivative(1)
    assert velocity is not None
    residual = system.spatial_operator().spatial_residual(state, time)
    return bkd.index_update(residual, constraint_set.dofs(), velocity(time))


def _state(system: GalerkinSystem[Array], scale: float) -> Array:
    bkd = system.bkd()
    nstates = system.spatial_operator().nstates()
    return bkd.asarray(scale * np.cos(np.linspace(0.0, 2.0, nstates)))


class TestTimeArgument:
    def test_adapter_answers_each_time_asked(self, bkd: Backend[Array]) -> None:
        system = _system(bkd)
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        state = _state(system, 1.0)
        for time in (0.2, 0.9, 0.2, 0.55):
            bkd.assert_allclose(
                adapter(state, time),
                _reference_rhs(system, state, time),
                rtol=1e-12,
                atol=1e-14,
            )
        # The two times give different residuals, so a stale time is
        # visible to the comparisons above.
        gap = bkd.norm(adapter(state, 0.2) - adapter(state, 0.9))
        assert bkd.to_float(gap) > 1e-2

    def test_crank_nicolson_evaluates_each_end_at_its_time(
        self, bkd: Backend[Array]
    ) -> None:
        system = _system(bkd)
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        y_prev, y_curr = _state(system, 1.0), _state(system, 1.3)
        t_prev, deltat = 0.2, 0.3
        stepper = CrankNicolsonStepper(adapter)
        stepper.bind(StepContext(t_prev=t_prev, deltat=deltat, y_prev=y_prev))
        expected = adapter.mass_matrix().apply(y_curr - y_prev) - 0.5 * deltat * (
            _reference_rhs(system, y_prev, t_prev)
            + _reference_rhs(system, y_curr, t_prev + deltat)
        )
        bkd.assert_allclose(stepper(y_curr), expected, rtol=1e-12, atol=1e-14)
        # Wrong-model control: both ends at t_curr, the error a stored
        # time set once per step would make.
        stale = adapter.mass_matrix().apply(y_curr - y_prev) - 0.5 * deltat * (
            _reference_rhs(system, y_prev, t_prev + deltat)
            + _reference_rhs(system, y_curr, t_prev + deltat)
        )
        assert bkd.to_float(bkd.norm(stale - expected)) > 1e-3

    def test_implicit_midpoint_evaluates_at_the_midpoint_time(
        self, bkd: Backend[Array]
    ) -> None:
        system = _system(bkd)
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        y_prev, y_curr = _state(system, 1.0), _state(system, 1.3)
        t_prev, deltat = 0.2, 0.3
        stepper = ImplicitMidpointStepper(adapter)
        stepper.bind(StepContext(t_prev=t_prev, deltat=deltat, y_prev=y_prev))
        expected = adapter.mass_matrix().apply(
            y_curr - y_prev
        ) - deltat * _reference_rhs(
            system, 0.5 * (y_prev + y_curr), t_prev + 0.5 * deltat
        )
        bkd.assert_allclose(stepper(y_curr), expected, rtol=1e-12, atol=1e-14)

    def test_heun_evaluates_each_stage_at_its_time(self, bkd: Backend[Array]) -> None:
        system = _system(bkd)
        adapter = GalerkinPhysicsToODEResidualAdapter(system)
        mass = adapter.mass_matrix()
        y_prev, y_curr = _state(system, 1.0), _state(system, 1.3)
        t_prev, deltat = 0.2, 0.3
        stepper = HeunStepper(adapter)
        stepper.bind(StepContext(t_prev=t_prev, deltat=deltat, y_prev=y_prev))
        k1 = _reference_rhs(system, y_prev, t_prev)
        k2 = _reference_rhs(system, y_prev + deltat * mass.solve(k1), t_prev + deltat)
        expected = mass.apply(y_curr - y_prev) - 0.5 * deltat * (k1 + k2)
        bkd.assert_allclose(stepper(y_curr), expected, rtol=1e-12, atol=1e-14)
