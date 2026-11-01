r"""Reuse of stage factorizations within one solve, on a Galerkin problem.

1D heat equation on [0, 1], P1 on 10 cells, forcing (1 + x)(1 + sin 5t),
u(0) = 0 and a moving right end 0.5 sin 4t (its rate supplied), from
0.3 sin(pi x), 20 steps of dt = 0.001 to T = 0.02 (``FixedStep``: the
sizes are given exactly, the last one landing on T), Newton tolerance
1e-13, the essential rows pinned in each step's last block. With the
diffusivity :math:`\kappa = 1 + x` declared time-independent the Jacobian
is constant though the forcing and the boundary move, so one solve
factors each distinct stage operator once per distinct step size (two:
0.001, and the last step's): backward Euler and Crank-Nicolson twice,
SDIRK3 four times (its unpinned stages share one operator, its pinned
last stage is another). With :math:`\kappa = (1 + t)(1 + x)` declared
time-dependent, every implicit stage is factored afresh. A second solve
opens a new session and factors again, and the reused trajectory equals
the one stepped without reuse to round-off.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, List

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.integration import Integrator
from pyapprox.ode.pinned_rows import RowPinning
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.tableau import (
    ButcherTableau,
    alexander_sdirk3,
    backward_euler,
    crank_nicolson,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    CoordinateDiffusion,
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import (
    NeutralizedRows,
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    create_galerkin_physics_ode_residual,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    SolverCapabilities,
    SolverHandleProtocol,
)
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]
_DELTAT, _NSTEPS = 0.001, 20
# Steps of exactly 0.001: the size is given, not recovered from times
# (0.001 n would give four different sizes in 20 steps).
_STEPS = FixedStep(_DELTAT, (0.0, _NSTEPS * _DELTAT))


def _forcing(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * (1.0 + np.sin(5.0 * t))
    return value


def _right_value(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 0.5 * np.sin(4.0 * t))
    return value


def _right_rate(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 2.0 * np.cos(4.0 * t))
    return value


def _kappa(x: _Np) -> _Np:
    value: _Np = 1.0 + x[0]
    return value


def _growing_kappa(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + t) * (1.0 + x[0])
    return value


class _CountingLU:
    """The sparse LU, counting its setups."""

    def __init__(self) -> None:
        self._lu = ScipySparseLU()
        self._setups = 0

    def setups(self) -> int:
        return self._setups

    def capabilities(self) -> SolverCapabilities:
        return self._lu.capabilities()

    def setup(
        self, op: ActionOperatorProtocol[_Np, CSRFormat]
    ) -> SolverHandleProtocol[_Np]:
        self._setups += 1
        return self._lu.setup(op)


class _Run:
    """The heat problem under one tableau and diffusivity, stepped with
    the step-end rows pinned; its solver counts setups."""

    def __init__(
        self,
        bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        diffusivity: CoordinateDiffusion,
    ) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=diffusivity,
            bkd=bkd,
            forcing=TimeDependent(_forcing),
        )
        right = BoundarySignal(
            TimeDependent(_right_value), [TimeDependent(_right_rate)]
        )
        system = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", right, bkd),
            ],
        )
        self.lu = _CountingLU()
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(system), self.lu, row_treatment=NeutralizedRows
        )
        newton: Newton[_Np, _Np] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
        self.stepper = RungeKuttaStepper(
            make(),
            adapter.mass_form_system(),
            newton,
            RowPinning(adapter.pinned_rows()),
        )
        x = np.asarray(basis.dof_coordinates())[0]
        self.init_state: _Np = (0.3 * np.sin(np.pi * x))[:, None]

    def solve(self) -> List[_Np]:
        """The stored states of one solve."""
        store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        Integrator(self.stepper).solve(self.init_state, _STEPS, store)
        return [store.state(n) for n in range(store.ntimes())]

    def step_without_reuse(self) -> List[_Np]:
        """The states stepped one bare step at a time, over the same spans:
        a session each, so nothing is reused between steps."""
        states = [self.init_state]
        sizes = _STEPS.sizes()
        span = StepSpan.first(_STEPS.origin(), sizes[0])
        for n in range(_NSTEPS):
            if n > 0:
                span = span.advanced(sizes[n], 1)
            unknown = self.stepper.step((states[-1],), span)
            states.append(self.stepper.readout().apply(unknown).blocks[0])
        return states


_CONSTANT = CoordinateDiffusion(TimeIndependent(_kappa))
_GROWING = CoordinateDiffusion(TimeDependent(_growing_kappa))


class TestFactorizationReuse:
    @pytest.mark.parametrize(
        "make, distinct",
        [(backward_euler, 1), (crank_nicolson, 1), (alexander_sdirk3, 2)],
        ids=["backward_euler", "crank_nicolson", "sdirk3"],
    )
    def test_a_constant_jacobian_is_factored_once_per_solve(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        distinct: int,
    ) -> None:
        # The last step lands exactly on 0.02, absorbing the drift of the
        # summed times, so it has a size of its own: each distinct stage
        # operator is factored once for 0.001 and once for the last step.
        assert len(set(_STEPS.sizes())) == 2
        per_solve = 2 * distinct
        run = _Run(numpy_bkd, make, _CONSTANT)
        reused = run.solve()
        assert run.lu.setups() == per_solve
        run.solve()
        assert run.lu.setups() == 2 * per_solve
        before = run.lu.setups()
        fresh = run.step_without_reuse()
        # A bare step reuses within itself only.
        assert run.lu.setups() - before == _NSTEPS * distinct
        for mine, theirs in zip(reused, fresh):
            numpy_bkd.assert_allclose(mine, theirs, rtol=1e-13, atol=1e-15)

    @pytest.mark.parametrize(
        "make, implicit_stages",
        [(backward_euler, 1), (crank_nicolson, 1), (alexander_sdirk3, 3)],
        ids=["backward_euler", "crank_nicolson", "sdirk3"],
    )
    def test_a_time_dependent_jacobian_is_factored_at_every_stage(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        implicit_stages: int,
    ) -> None:
        run = _Run(numpy_bkd, make, _GROWING)
        run.solve()
        assert run.lu.setups() == _NSTEPS * implicit_stages
