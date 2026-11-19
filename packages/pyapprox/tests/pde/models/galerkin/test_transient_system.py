r"""The transient Galerkin builder and driver, against the old transient model.

The moving-end heat problem (``tests._helpers.galerkin_heat``: P1 on 10
cells, diffusivity exp(W p) from a three-term log-KLE at
p = (0.4, -0.3, 0.2), a moving right end with its rate), dt = 0.001 to
T = 0.02, Newton tolerance 1e-13. For each method both paths have, the
driver (:func:`solve_galerkin_transient` over
:func:`galerkin_transient_system`) and its sweeps
(:func:`galerkin_transient_derivatives`) must reproduce
``GalerkinTransientModel`` with its default row treatment and the old
adjoint: the trajectory to 1e-10 and the gradient and Hessian-vector
product of the value next to the moving end to 1e-9, relative. Implicit
midpoint is the exception, a different discretization of the step-end pin
(5e-8; its order is checked in ``test_new_time_integration.py``).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, List, Optional

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.observation_terms import StateEntries
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.galerkin.time_integration.transient_model import (
    GalerkinTransientModel,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.models.galerkin.transient_system import (
    GalerkinTransientSystem,
    galerkin_transient_derivatives,
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.galerkin_heat import HEAT_NPARAMS, HEAT_PARAM, MovingEndHeat

_Np = NDArray[Any]
_DELTAT, _FINAL_TIME = 0.001, 0.02
_DIRECTION = np.array([[0.7], [-1.1], [0.4]])
_METHODS: List[tuple[Callable[[], ButcherTableau], str]] = [
    (backward_euler, "backward_euler"),
    (forward_euler, "forward_euler"),
    (crank_nicolson, "crank_nicolson"),
    (implicit_midpoint, "implicit_midpoint"),
    (heun, "heun"),
]


def _rtol(old_name: str, rtol: float) -> float:
    return 5e-8 if old_name == "implicit_midpoint" else rtol


class _Paths:
    """The heat problem through the driver and through the old model."""

    def __init__(
        self, bkd: NumpyBkd, make: Callable[[], ButcherTableau], old_name: str
    ) -> None:
        heat = MovingEndHeat(bkd)
        self._heat = heat
        parameters = ParameterState(heat.parameterization)
        parameters.set_param(HEAT_PARAM[:, 0])
        self._system: GalerkinTransientSystem[_Np, Any] = galerkin_transient_system(
            skfem_discretization(heat.system), ScipySparseLU(), parameters
        )
        tableau = make()
        method: RungeKutta[_Np, _Np] = RungeKutta(
            tableau,
            None
            if tableau.is_explicit()
            else Newton(atol=1e-13, rtol=0.0, maxiters=20),
        )
        self._store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        stepper = solve_galerkin_transient(
            self._system,
            method,
            RecomputeAlways(self._system.solved.solver),
            FixedStep(_DELTAT, (0.0, _FINAL_TIME)),
            np.reshape(heat.init_state, (-1, 1)),
            self._store,
        )
        self._sweeps = galerkin_transient_derivatives(self._system, stepper)

        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(heat.system), ScipySparseLU(), heat.parameterization
        )
        adapter.set_param(HEAT_PARAM[:, 0])
        self._old = GalerkinTransientModel(adapter)
        self._old_states, _ = self._old.solve_transient(
            heat.init_state,
            TimeIntegrationConfig(
                method=old_name,
                init_time=0.0,
                final_time=_FINAL_TIME,
                deltat=_DELTAT,
                newton_tol=1e-13,
                newton_maxiter=20,
                verbosity=0,
            ),
        )

    def new_states(self) -> _Np:
        return np.hstack([self._store.state(n) for n in range(self._store.ntimes())])

    def old_states(self) -> _Np:
        return self._old_states

    def _endpoint(self) -> TrajectoryFunctional[_Np, _Np]:
        bkd = NumpyBkd()
        nstates = self._system.solved.space.descriptor().size
        term = EndpointTerm(StateEntries(bkd, nstates, [self._heat.near_right]))
        return TrajectoryFunctional.from_terms(bkd, 1, state_terms=(term,))

    def _old_operator(self) -> TimeAdjointOperatorWithHVP[_Np]:
        functional = EndpointFunctional(
            self._heat.near_right,
            len(self._heat.init_state),
            HEAT_NPARAMS,
            NumpyBkd(),
        )
        return TimeAdjointOperatorWithHVP(self._old.last_integrator(), functional)

    def new_gradient(self) -> _Np:
        return self._sweeps.gradient(self._store, self._endpoint())

    def old_gradient(self) -> _Np:
        jacobian = self._old_operator().jacobian(self._heat.init_state, HEAT_PARAM)
        return np.reshape(jacobian, (-1, 1))

    def new_hvp(self) -> _Np:
        return self._sweeps.hvp(self._store, self._endpoint(), _DIRECTION)

    def old_hvp(self) -> _Np:
        product = self._old_operator().hvp(
            self._heat.init_state, HEAT_PARAM, _DIRECTION
        )
        return np.reshape(product, (-1, 1))


@pytest.mark.parametrize("make, old_name", _METHODS, ids=[n for _, n in _METHODS])
class TestAgainstTheOldModel:
    def test_trajectory(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, make, old_name)
        numpy_bkd.assert_allclose(
            paths.new_states(),
            paths.old_states(),
            rtol=_rtol(old_name, 1e-10),
            atol=1e-13,
        )

    def test_gradient_and_hvp(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, make, old_name)
        numpy_bkd.assert_allclose(
            paths.new_gradient(), paths.old_gradient(), rtol=_rtol(old_name, 1e-9)
        )
        numpy_bkd.assert_allclose(
            paths.new_hvp(), paths.old_hvp(), rtol=_rtol(old_name, 1e-9)
        )


def _system(
    bkd: NumpyBkd, with_parameters: bool = False
) -> tuple[GalerkinTransientSystem[_Np, Any], MovingEndHeat]:
    heat = MovingEndHeat(bkd)
    parameters: Optional[ParameterState[_Np]] = None
    if with_parameters:
        parameters = ParameterState(heat.parameterization)
        parameters.set_param(HEAT_PARAM[:, 0])
    return (
        galerkin_transient_system(
            skfem_discretization(heat.system), ScipySparseLU(), parameters
        ),
        heat,
    )


class TestDriver:
    def test_injects_the_essential_data_at_the_origin(
        self, numpy_bkd: NumpyBkd
    ) -> None:
        """The initial state's essential entries become their data at the
        plan's origin, whatever was passed: here at t = 0.1, where the
        moving end is 0.5 sin 0.4."""
        system, heat = _system(numpy_bkd, with_parameters=True)
        store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        solve_galerkin_transient(
            system,
            RungeKutta(backward_euler(), Newton(atol=1e-13, rtol=0.0)),
            RecomputeAlways(system.solved.solver),
            FixedStep(_DELTAT, (0.1, 0.1 + 2 * _DELTAT)),
            np.full((len(heat.init_state), 1), 7.0),
            store,
        )
        expected = system.constraints.inject(np.full(len(heat.init_state), 7.0), 0.1)
        numpy_bkd.assert_allclose(store.state(0)[:, 0], expected, rtol=1e-15)
        assert not np.allclose(expected, 7.0)

    def test_refuses_a_one_dimensional_initial_state(self, numpy_bkd: NumpyBkd) -> None:
        system, heat = _system(numpy_bkd)
        method: RungeKutta[_Np, _Np] = RungeKutta(forward_euler())
        store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        with pytest.raises(ValueError, match="init_state must have shape"):
            solve_galerkin_transient(
                system,
                method,
                RecomputeAlways(system.solved.solver),
                FixedStep(_DELTAT, (0.0, _FINAL_TIME)),
                heat.init_state,
                store,
            )

    def test_refuses_a_method_that_is_not_one(self, numpy_bkd: NumpyBkd) -> None:
        system, heat = _system(numpy_bkd)
        store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        with pytest.raises(TypeError, match="SteppingMethodProtocol"):
            solve_galerkin_transient(
                system,
                backward_euler(),  # type: ignore[arg-type]
                RecomputeAlways(system.solved.solver),
                FixedStep(_DELTAT, (0.0, _FINAL_TIME)),
                np.reshape(heat.init_state, (-1, 1)),
                store,
            )
