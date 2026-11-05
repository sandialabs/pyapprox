r"""The Galerkin heat problem on the new time integration, against the old.

1D heat equation on [0, 1], P1 on 10 cells, diffusivity exp(W p) from a
three-term log-KLE at p = (0.4, -0.3, 0.2), forcing (1 + x)(1 + sin 5t),
u(0) = 0 and a moving right end 0.5 sin 4t (its rate supplied), from
0.3 sin(pi x) (``tests._helpers.galerkin_heat``), dt = 0.001 to T =
0.02, Newton tolerance 1e-13. The
essential rows are handled by one of the strategies:

- ``step_end`` (A): the coupled row treatment, with the essential rows
  pinned inside each step's final solve. New: the adapter's
  ``pinned_rows()`` through ``RowPinning`` at the step end. Old: the BC
  wrapper around the tiered stepper.
- ``projection`` (B): the symmetric row treatment and the mass-consistent
  projection at stored states, on both paths.
- ``pin`` (C): the symmetric row treatment and the essential pin at stored
  states, on both paths.
- ``pinned_rows`` (C, generic): as ``pin``, the new path pinning with
  ``PinnedRowsConstraint`` over the adapter's ``pinned_rows()``, the old
  path with ``EssentialPin``.

For each strategy and each method both paths have (backward and forward
Euler, Crank-Nicolson, implicit midpoint, Heun), the new path (the
adapter's ``mass_form_system()`` under the tableau stepper,
``Integrator``, ``TrajectoryDerivatives``) must reproduce the old one
(``TimeIntegrator`` with the tiered stepper, its adjoint): the trajectory
to 1e-10 and the gradient and Hessian-vector product of the value at the
DOF next to the moving end to 1e-9, relative. Implicit midpoint pinned at
the step end is the exception: the old stepper pins the step's endpoint
and the tableau stepper its final block, different discretizations whose
gap (5e-8) is checked to fall like :math:`\Delta t^3`.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.integration import (
    DifferentiableStackConstraintProtocol,
    Integrator,
    ResidualFormConstraint,
)
from pyapprox.ode.observation_terms import EndpointTerm, StateEntries
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.pinned_rows import PinnedRowsConstraint, RowPinning
from pyapprox.ode.protocols import StateConstraintProtocol
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_derivatives import (
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    NeutralizedRows,
    SymmetricNeutralizedRows,
    create_galerkin_bc_enforcing_residual,
    skfem_discretization,
)
from pyapprox.pde.galerkin.time_integration.state_constraints import (
    EssentialPin,
    MassConsistentProjection,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.newton import NewtonSolver
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.galerkin_heat import HEAT_NPARAMS, HEAT_PARAM, MovingEndHeat

_Np = NDArray[Any]
_NPARAMS = HEAT_NPARAMS
_PARAM = HEAT_PARAM
_DELTAT, _FINAL_TIME = 0.001, 0.02
_Strategy = Literal["step_end", "projection", "pin", "pinned_rows"]
_STRATEGIES: List[_Strategy] = ["step_end", "projection", "pin", "pinned_rows"]

# The old path's stepper name for each tableau, for the same method.
_METHODS: List[tuple[Callable[[], ButcherTableau], str]] = [
    (backward_euler, "backward_euler"),
    (forward_euler, "forward_euler"),
    (crank_nicolson, "crank_nicolson"),
    (implicit_midpoint, "implicit_midpoint"),
    (heun, "heun"),
]


class _Heat:
    """The parameterized heat system, its initial state and the DOF next
    to the moving right end."""

    def __init__(self, bkd: NumpyBkd) -> None:
        heat = MovingEndHeat(bkd)
        self.system: GalerkinSystem[_Np] = heat.system
        self.parameterization = heat.parameterization
        self.init_state: _Np = heat.init_state
        self.near_right = heat.near_right

    def adapter(
        self, strategy: _Strategy
    ) -> GalerkinPhysicsToODEResidualWithSetParamAdapter[_Np, Any]:
        """The adapter at the parameters: the coupled row treatment for
        the step-end pin, the symmetric one for the others."""
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(self.system),
            ScipySparseLU(),
            self.parameterization,
            row_treatment=(
                NeutralizedRows if strategy == "step_end" else SymmetricNeutralizedRows
            ),
        )
        adapter.set_param(_PARAM[:, 0])
        return adapter

    def constraint(
        self,
        adapter: GalerkinPhysicsToODEResidualWithSetParamAdapter[_Np, Any],
        strategy: _Strategy,
    ) -> StateConstraintProtocol[_Np]:
        """The old path's stored-state constraint (the new path's too, for
        the projection and the pin)."""
        if strategy == "projection":
            return MassConsistentProjection(adapter, _NPARAMS)
        return EssentialPin(self.system, _NPARAMS)

    def endpoint(self) -> TrajectoryFunctional[_Np, _Np]:
        """:meth:`functional` as the new path's native endpoint term."""
        bkd = NumpyBkd()
        term = EndpointTerm(StateEntries(bkd, self.system.nstates(), [self.near_right]))
        return TrajectoryFunctional.from_terms(bkd, 1, state_terms=(term,))

    def functional(self) -> EndpointFunctional[_Np]:
        """The value at the DOF next to the moving end."""
        return EndpointFunctional(
            self.near_right, self.system.nstates(), _NPARAMS, NumpyBkd()
        )


class _Paths:
    """The heat problem on both paths under one strategy and method, to
    ``_FINAL_TIME`` in steps of ``deltat``."""

    def __init__(
        self,
        bkd: NumpyBkd,
        strategy: _Strategy,
        make: Callable[[], ButcherTableau],
        old_name: str,
        deltat: float = _DELTAT,
    ) -> None:
        heat = _Heat(bkd)
        self._heat, self._bkd = heat, bkd
        nsteps = round(_FINAL_TIME / deltat)
        times = [deltat * n for n in range(nsteps + 1)]
        # The new path: the step-end strategy pins inside the stepper, the
        # others constrain the stored states.
        adapter = heat.adapter(strategy)
        tableau = make()
        newton: Newton[_Np, _Np] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
        system = adapter.mass_form_system()
        self._stepper = RungeKuttaStepper(
            tableau,
            system,
            None if tableau.is_explicit() else newton,
            RowPinning(adapter.pinned_rows()) if strategy == "step_end" else None,
        )
        form = BackendColumnForm(bkd)
        constraint: Optional[DifferentiableStackConstraintProtocol[_Np]] = None
        if strategy == "pinned_rows":
            _, params = backend_spaces(bkd, len(heat.init_state), _NPARAMS)
            constraint = PinnedRowsConstraint(
                adapter.pinned_rows(), system.space, params
            )
        elif strategy != "step_end":
            constraint = ResidualFormConstraint(
                heat.constraint(adapter, strategy), form
            )
        self._store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        Integrator(self._stepper, constraint).solve(
            bkd.reshape(bkd.asarray(heat.init_state), (-1, 1)),
            GivenSizes.from_times(times),
            self._store,
        )
        self._sweeps: TrajectoryDerivatives[_Np, _Np] = TrajectoryDerivatives(
            self._stepper, constraint, ResidualInitialState(adapter, form)
        )
        # The old path, on its own adapter: the BC wrapper around the
        # stepper for the step-end strategy, a stored-state constraint
        # otherwise (the essential pin for both pins).
        old_adapter = heat.adapter(strategy)
        old_stepper = create_stepper(old_name, old_adapter)
        if strategy == "step_end":
            old_newton = NewtonSolver(
                create_galerkin_bc_enforcing_residual(
                    old_stepper, heat.system.constraint_set(), bkd
                )
            )
        else:
            old_newton = NewtonSolver(old_stepper)
        old_newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
        self._integrator = TimeIntegrator(
            0.0,
            _FINAL_TIME,
            deltat,
            old_newton,
            constraint=(
                None
                if strategy == "step_end"
                else heat.constraint(
                    old_adapter, "pin" if strategy == "pinned_rows" else strategy
                )
            ),
        )
        self._old_states, _ = self._integrator.solve(heat.init_state)

    def new_states(self) -> _Np:
        return np.hstack([self._store.state(n) for n in range(self._store.ntimes())])

    def old_states(self) -> _Np:
        return self._old_states

    def new_gradient(self) -> _Np:
        return self._sweeps.gradient(self._store, self._heat.endpoint())

    def old_gradient(self) -> _Np:
        operator = TimeAdjointOperatorWithHVP(self._integrator, self._heat.functional())
        return np.reshape(operator.jacobian(self._heat.init_state, _PARAM), (-1, 1))

    def new_hvp(self, direction: _Np) -> _Np:
        return self._sweeps.hvp(self._store, self._heat.endpoint(), direction)

    def old_hvp(self, direction: _Np) -> _Np:
        operator = TimeAdjointOperatorWithHVP(self._integrator, self._heat.functional())
        product = operator.hvp(self._heat.init_state, _PARAM, direction)
        return np.reshape(product, (-1, 1))


_DIRECTION = np.array([[0.7], [-1.1], [0.4]])


def _rtol(strategy: _Strategy, old_name: str, rtol: float) -> float:
    """The relative tolerance between the paths: ``rtol``, except for
    implicit midpoint pinned at the step end, where the paths are different
    discretizations.

    The old midpoint stepper solves for :math:`y_{n+1}` and pins it, so the
    midpoint state's boundary entry is :math:`(g_n + g_{n+1})/2`; the
    tableau stepper solves its stage unpinned, giving :math:`g_n + (\\Delta
    t/2)\\dot g(t_{n+1/2})`. These differ by :math:`O(\\Delta t^3)`, which
    reaches :math:`y_{n+1}` through :math:`\\Delta t f`, so the paths
    differ by :math:`O(\\Delta t^4)` a step (observed: 5e-9 trajectory,
    1.8e-8 gradient, 1.8e-9 HVP);
    :meth:`TestMidpointAtTheStepEnd.test_the_gap_is_the_formulation` checks
    the gap's order."""
    if strategy == "step_end" and old_name == "implicit_midpoint":
        return 5e-8
    return rtol


@pytest.mark.parametrize("strategy", _STRATEGIES)
@pytest.mark.parametrize("make, old_name", _METHODS, ids=[n for _, n in _METHODS])
class TestAgainstTheOldPath:
    def test_trajectory(
        self,
        numpy_bkd: NumpyBkd,
        strategy: _Strategy,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, strategy, make, old_name)
        numpy_bkd.assert_allclose(
            paths.new_states(),
            paths.old_states(),
            rtol=_rtol(strategy, old_name, 1e-10),
            atol=1e-13,
        )

    def test_gradient(
        self,
        numpy_bkd: NumpyBkd,
        strategy: _Strategy,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, strategy, make, old_name)
        numpy_bkd.assert_allclose(
            paths.new_gradient(),
            paths.old_gradient(),
            rtol=_rtol(strategy, old_name, 1e-9),
        )

    def test_hvp(
        self,
        numpy_bkd: NumpyBkd,
        strategy: _Strategy,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, strategy, make, old_name)
        numpy_bkd.assert_allclose(
            paths.new_hvp(_DIRECTION),
            paths.old_hvp(_DIRECTION),
            rtol=_rtol(strategy, old_name, 1e-9),
        )


class TestMidpointAtTheStepEnd:
    def test_the_gap_is_the_formulation(self, numpy_bkd: NumpyBkd) -> None:
        """The final-state gap between the paths falls like
        :math:`\\Delta t^3` (an :math:`O(\\Delta t^4)` difference a step,
        see :func:`_rtol`) when the step halves from 0.002 to 0.001
        (observed ratio 8.0); a defect in either path would not."""
        gaps = []
        for deltat in (2 * _DELTAT, _DELTAT):
            paths = _Paths(
                numpy_bkd, "step_end", implicit_midpoint, "implicit_midpoint", deltat
            )
            gaps.append(
                np.linalg.norm(paths.new_states()[:, -1] - paths.old_states()[:, -1])
            )
        assert 7.0 < gaps[0] / gaps[1] < 9.0
