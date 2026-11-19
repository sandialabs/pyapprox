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

``TransientImplicitMap`` over the same system reproduces those numbers
for the endpoint value, and the old weighted-L2 forward model for a time
integral as its functional (backward Euler, Crank-Nicolson). The quantity
factories (flux, probes, outflux, weighted L2) are each checked against
their old functionals.
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
from pyapprox.ode.functionals.protocols import TransientFunctionalProtocol
from pyapprox.ode.functionals.time_integrated_weighted_l2 import (
    TimeIntegratedWeightedL2Functional,
)
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.step_residual_terms import StepResidualTerm
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectorySinkProtocol,
)
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TrajectoryFunctionalSourceProtocol,
    TransientImplicitMap,
)
from pyapprox.pde.galerkin.observations import (
    nodal_outflux_functional,
    probe_observation_functional,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.galerkin.time_integration.transient_model import (
    GalerkinTransientModel,
)
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    TimeIntegratedEssentialBoundaryFluxFunctional,
    time_integrated_essential_boundary_flux,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.models.galerkin.transient import GalerkinTransientForwardModel
from pyapprox.pde.models.galerkin.transient_observations import (
    nodal_outflux_observations,
    probe_observations,
    time_integrated_weighted_l2,
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
from pyapprox.util.linalg.symmetric_weight import DenseMatrixWeight
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
        self._method = method

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

    def endpoint_functional(self) -> TrajectoryFunctional[_Np, _Np]:
        """The value next to the moving end at the last stored state."""
        return self._endpoint()

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

    def flux(self, weights: _Np) -> StepResidualTerm[_Np, _Np]:
        """The time-integrated flux through both ends under ``weights``."""
        return time_integrated_essential_boundary_flux(
            self._system,
            self._method,
            RecomputeAlways(self._system.unconstrained.solver),
            np.asarray(self._heat.ends),
            weights,
        )

    def heat(self) -> MovingEndHeat:
        return self._heat

    def system(self) -> GalerkinTransientSystem[_Np, Any]:
        return self._system

    def method(self) -> RungeKutta[_Np, _Np]:
        return self._method

    def store(self) -> InMemoryTrajectoryStore[_Np]:
        return self._store

    def sweeps(self) -> TrajectoryDerivatives[_Np, _Np]:
        return self._sweeps

    def resolve(self, sink: TrajectorySinkProtocol[_Np]) -> None:
        """Solve again, into ``sink``."""
        solve_galerkin_transient(
            self._system,
            self._method,
            RecomputeAlways(self._system.solved.solver),
            FixedStep(_DELTAT, (0.0, _FINAL_TIME)),
            np.reshape(self._heat.init_state, (-1, 1)),
            sink,
        )

    def flux_value_and_derivatives(
        self, weights: _Np, direction: _Np
    ) -> tuple[_Np, _Np, _Np]:
        """The flux summed over the steps, its gradient and HVP (first QoI
        weighted 1, the others 0)."""
        term = self.flux(weights)
        nqoi = weights.shape[0]
        value = sum(
            term.value(step, self._store.window(step))
            for step in range(self._store.nsteps())
        )
        functional = TrajectoryFunctional.from_terms(
            NumpyBkd(), nqoi, step_terms=(term,)
        )
        first = np.zeros((nqoi, 1))
        first[0, 0] = 1.0
        gradient = self._sweeps.gradient(self._store, functional, weights=first)
        hvp = self._sweeps.hvp(self._store, functional, direction, weights=first)
        return np.reshape(value, (-1, 1)), gradient, hvp

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


_FLUX_METHODS = [m for m in _METHODS if m[1] != "heun"]


def _old_flux(
    heat: MovingEndHeat, old_name: str, weights: _Np
) -> GalerkinTransientForwardModel[_Np]:
    """The old forward model of the time-integrated flux under ``weights``."""
    bkd = NumpyBkd()
    functional = TimeIntegratedEssentialBoundaryFluxFunctional(
        GalerkinParameterizedResidual(
            heat.system.spatial_operator(), heat.parameterization
        ),
        heat.system,
        bkd.asarray(heat.ends),
        weights,
    )
    return _old_model(heat, old_name, functional)


def _old_model(
    heat: MovingEndHeat,
    old_name: str,
    functional: TransientFunctionalProtocol[_Np],
) -> GalerkinTransientForwardModel[_Np]:
    """The old forward model of ``functional`` on the heat problem."""
    bkd = NumpyBkd()
    return GalerkinTransientForwardModel(
        skfem_discretization(heat.system),
        ScipySparseLU(),
        heat.parameterization,
        bkd.asarray(heat.init_state),
        TimeIntegrationConfig(
            method=old_name,
            init_time=0.0,
            final_time=_FINAL_TIME,
            deltat=_DELTAT,
            newton_tol=1e-13,
            newton_maxiter=20,
            verbosity=0,
        ),
        functional=functional,
    )


@pytest.mark.parametrize(
    "make, old_name", _FLUX_METHODS, ids=[n for _, n in _FLUX_METHODS]
)
class TestTimeIntegratedFlux:
    """The flux term against ``TimeIntegratedEssentialBoundaryFluxFunctional``
    (which refuses Heun): two QoIs under distinct weights on the two ends,
    the value of both and the first's gradient and HVP, to 1e-9. Under
    implicit midpoint the old functional offers no HVP (its rule couples
    steps), so there only the value and gradient are compared."""

    _WEIGHTS = np.array([[1.0, -0.5], [0.3, 2.0]])

    def test_value_gradient_and_hvp(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, make, old_name)
        value, gradient, hvp = paths.flux_value_and_derivatives(
            self._WEIGHTS, _DIRECTION
        )
        old_both = _old_flux(paths.heat(), old_name, self._WEIGHTS)
        rtol = _rtol(old_name, 1e-9)
        numpy_bkd.assert_allclose(value, old_both(HEAT_PARAM), rtol=rtol)
        old_first = _old_flux(paths.heat(), old_name, self._WEIGHTS[:1])
        derivatives = old_first.derivatives()
        assert derivatives.jacobian is not None
        numpy_bkd.assert_allclose(
            gradient, derivatives.jacobian(HEAT_PARAM).T, rtol=rtol
        )
        if old_name == "implicit_midpoint":
            assert derivatives.hvp is None
            return
        assert derivatives.hvp is not None
        numpy_bkd.assert_allclose(
            hvp, derivatives.hvp(HEAT_PARAM, _DIRECTION), rtol=rtol
        )


def _stored_value(
    functional: TrajectoryFunctional[_Np, _Np], store: InMemoryTrajectoryStore[_Np]
) -> _Np:
    """``functional`` on the stored trajectory, term by term."""
    total = functional.zero
    last = store.ntimes() - 1
    for term in functional.state_terms:
        for index in range(store.ntimes()):
            if term.reads(index, index == last):
                member = term.at(index, store.time(index))
                total = total + member.value(store.state(index))
    for step_term in functional.step_terms:
        for step in range(store.nsteps()):
            total = total + step_term.value(step, store.window(step))
    return total


# The initial state, a middle one and the last of the 21 stored.
_STORED = [0, 7, 20]
# Two interior points and the moving end.
_POINTS = np.array([[0.37, 0.81, 1.0]])


def _check_observations(
    bkd: NumpyBkd,
    paths: _Paths,
    functional: TrajectoryFunctional[_Np, _Np],
    old: GalerkinTransientForwardModel[_Np],
    rtol: float,
) -> None:
    """The value, the tangent along ``_DIRECTION`` and the gradient under
    random weights against the old model's value and Jacobian."""
    store, sweeps = paths.store(), paths.sweeps()
    bkd.assert_allclose(_stored_value(functional, store), old(HEAT_PARAM), rtol=rtol)
    derivatives = old.derivatives()
    assert derivatives.jacobian is not None
    jacobian = derivatives.jacobian(HEAT_PARAM)
    bkd.assert_allclose(
        sweeps.tangent(store, functional, _DIRECTION),
        jacobian @ _DIRECTION,
        rtol=rtol,
        atol=1e-14,
    )
    weights = np.random.default_rng(11).normal(size=(functional.nqoi(), 1))
    bkd.assert_allclose(
        sweeps.gradient(store, functional, weights=weights),
        jacobian.T @ weights,
        rtol=rtol,
    )


@pytest.mark.parametrize("make, old_name", _METHODS, ids=[n for _, n in _METHODS])
class TestProbes:
    """Probes at two interior points and the moving end, at stored states
    0, 7 and 20, against ``probe_observation_functional``: the value, the
    tangent and the weighted gradient, to 1e-9 (5e-8 under implicit
    midpoint). The moving end's value is data, so its derivatives are
    zero on both paths."""

    def test_value_tangent_and_gradient(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(numpy_bkd, make, old_name)
        heat = paths.heat()
        points = numpy_bkd.asarray(_POINTS)
        functional = probe_observations(heat.basis, points, _STORED)
        assert functional.nqoi() == 9
        old = _old_model(
            heat,
            old_name,
            probe_observation_functional(heat.basis, points, _STORED, HEAT_NPARAMS),
        )
        _check_observations(numpy_bkd, paths, functional, old, _rtol(old_name, 1e-9))


class TestNodalOutflux:
    """The outflux at both ends and the DOF next to the moving end, with
    normal velocities (1, -0.5, 2), at stored states 0, 7 and 20, against
    ``nodal_outflux_functional`` under backward Euler, to 1e-9."""

    def test_value_tangent_and_gradient(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        paths = _Paths(bkd, backward_euler, "backward_euler")
        heat = paths.heat()
        dofs = bkd.asarray(
            np.append(heat.ends, heat.near_right), dtype=bkd.int64_dtype()
        )
        velocity = bkd.asarray([1.0, -0.5, 2.0])
        nstates = len(heat.init_state)
        functional = nodal_outflux_observations(bkd, nstates, dofs, velocity, _STORED)
        old = _old_model(
            heat,
            "backward_euler",
            nodal_outflux_functional(
                dofs, velocity, _STORED, nstates, HEAT_NPARAMS, bkd
            ),
        )
        _check_observations(bkd, paths, functional, old, 1e-9)


def _weight(nstates: int) -> DenseMatrixWeight[_Np]:
    """``I + 0.3 (E_{+1} + E_{-1})``, symmetric positive definite."""
    matrix = np.eye(nstates) + 0.3 * (np.eye(nstates, k=1) + np.eye(nstates, k=-1))
    return DenseMatrixWeight(matrix, NumpyBkd())


def _weighted_l2(
    bkd: NumpyBkd, make: Callable[[], ButcherTableau], old_name: str
) -> tuple[_Paths, _Np, TrajectoryFunctional[_Np, _Np], _Np]:
    """The paths, the sink's value of the integral during a solve, the
    sweeps' functional and its value on the store."""
    paths = _Paths(bkd, make, old_name)
    system = paths.system()
    integral = time_integrated_weighted_l2(
        system,
        paths.method(),
        RecomputeAlways(system.solved.solver),
        _weight(len(paths.heat().init_state)),
    )
    sink = integral.sink()
    paths.resolve(sink)
    functional = integral.trajectory_functional(paths.store())
    return paths, sink.value(), functional, _stored_value(functional, paths.store())


_L2_METHODS = [m for m in _METHODS if m[1] != "heun"]


@pytest.mark.parametrize("make, old_name", _L2_METHODS, ids=[n for _, n in _L2_METHODS])
class TestTimeIntegratedWeightedL2:
    """The integral of ``y^T W y`` against ``TimeIntegratedWeightedL2Functional``
    with ``W = I + 0.3 (E_{+1} + E_{-1})``: the value from the sink during
    a solve and from the sweeps' functional on the store, the gradient and
    the HVP, to 1e-9 (5e-8 under implicit midpoint, where the old
    functional has no HVP; there a gradient entry of 4e-6 differs by 4e-13,
    hence the absolute tolerance).

    Heun is left out: the new integral takes Heun's predictor stage from
    the stored step unknown, while the old functional applies the
    trapezoid rule to the stored states (6e-5 apart, relative), so the old
    one is no oracle. Heun's integral is checked against finite
    differences in ``tests/ode/test_time_integral.py``."""

    def test_value_gradient_and_hvp(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        bkd = numpy_bkd
        paths, sink_value, functional, stored_value = _weighted_l2(bkd, make, old_name)
        store = paths.store()
        old = _old_model(
            paths.heat(),
            old_name,
            TimeIntegratedWeightedL2Functional(
                _weight(len(paths.heat().init_state)), HEAT_NPARAMS, bkd
            ),
        )
        rtol = _rtol(old_name, 1e-9)
        expected = old(HEAT_PARAM)
        bkd.assert_allclose(sink_value, expected, rtol=rtol)
        bkd.assert_allclose(stored_value, expected, rtol=rtol)
        derivatives = old.derivatives()
        assert derivatives.jacobian is not None
        bkd.assert_allclose(
            paths.sweeps().gradient(store, functional),
            derivatives.jacobian(HEAT_PARAM).T,
            rtol=rtol,
            atol=1e-12,
        )
        if derivatives.hvp is None:
            assert old_name == "implicit_midpoint"
            return
        bkd.assert_allclose(
            paths.sweeps().hvp(store, functional, _DIRECTION),
            derivatives.hvp(HEAT_PARAM, _DIRECTION),
            rtol=rtol,
        )


def _implicit_map(
    paths: _Paths, functional: TrajectoryFunctionalSourceProtocol[_Np, _Np]
) -> TransientImplicitMap[_Np, _Np, Any]:
    """The transient map of the heat problem, the system its own
    discretization and parameter setter."""
    system = paths.system()
    return TransientImplicitMap(
        system,
        system,
        paths.method(),
        RecomputeAlways(system.solved.solver),
        FixedStep(_DELTAT, (0.0, _FINAL_TIME)),
        np.reshape(paths.heat().init_state, (-1, 1)),
        functional,
    )


@pytest.mark.parametrize("make, old_name", _METHODS, ids=[n for _, n in _METHODS])
class TestTransientImplicitMapOnGalerkin:
    """``TransientImplicitMap`` over ``GalerkinTransientSystem`` (as both
    the discretization and the parameter setter): the value next to the
    moving end against the old model's trajectory, and its gradient and
    HVP against the old adjoint, as for the driver (1e-10 for the value,
    1e-9 for the derivatives; 5e-8 under implicit midpoint)."""

    def test_endpoint_value_gradient_and_hvp(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        bkd = numpy_bkd
        paths = _Paths(bkd, make, old_name)
        implicit = _implicit_map(
            paths, FixedTrajectoryFunctional(paths.endpoint_functional())
        )
        assert implicit.nparams() == HEAT_NPARAMS and implicit.nqoi() == 1
        param = bkd.asarray(HEAT_PARAM)
        point = implicit.point(param)
        expected = paths.old_states()[paths.heat().near_right, -1:]
        bkd.assert_allclose(point.value()[0], expected, rtol=_rtol(old_name, 1e-10))
        rtol = _rtol(old_name, 1e-9)
        bkd.assert_allclose(point.gradient(), paths.old_gradient(), rtol=rtol)
        bkd.assert_allclose(point.hvp(_DIRECTION), paths.old_hvp(), rtol=rtol)


@pytest.mark.parametrize(
    "make, old_name",
    [m for m in _METHODS if m[1] in ("backward_euler", "crank_nicolson")],
    ids=["backward_euler", "crank_nicolson"],
)
class TestTransientImplicitMapWithATimeIntegral:
    """The weighted L2 integral as the map's functional source, its value
    during the solve, gradient and HVP against
    ``TimeIntegratedWeightedL2Functional`` through the old model, to
    1e-9."""

    def test_value_gradient_and_hvp(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        bkd = numpy_bkd
        paths = _Paths(bkd, make, old_name)
        system = paths.system()
        weight = _weight(len(paths.heat().init_state))
        integral = time_integrated_weighted_l2(
            system, paths.method(), RecomputeAlways(system.solved.solver), weight
        )
        implicit = _implicit_map(paths, integral)
        old = _old_model(
            paths.heat(),
            old_name,
            TimeIntegratedWeightedL2Functional(weight, HEAT_NPARAMS, bkd),
        )
        param = bkd.asarray(HEAT_PARAM)
        bkd.assert_allclose(implicit.value(param), old(param), rtol=1e-9)
        point = implicit.point(param)
        derivatives = old.derivatives()
        assert derivatives.jacobian is not None and derivatives.hvp is not None
        bkd.assert_allclose(point.gradient(), derivatives.jacobian(param).T, rtol=1e-9)
        bkd.assert_allclose(
            point.hvp(_DIRECTION), derivatives.hvp(param, _DIRECTION), rtol=1e-9
        )


class TestObservationRefusals:
    def test_refuses_no_stored_state(self, numpy_bkd: NumpyBkd) -> None:
        heat = MovingEndHeat(numpy_bkd)
        with pytest.raises(ValueError, match="at least one stored state"):
            probe_observations(heat.basis, numpy_bkd.asarray(_POINTS), [])

    def test_refuses_a_negative_stored_state(self, numpy_bkd: NumpyBkd) -> None:
        heat = MovingEndHeat(numpy_bkd)
        with pytest.raises(ValueError, match="non-negative"):
            probe_observations(heat.basis, numpy_bkd.asarray(_POINTS), [-1])

    def test_refuses_mismatched_outflux_data(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError, match="nbdofs"):
            nodal_outflux_observations(
                bkd,
                11,
                bkd.asarray(np.array([1, 2]), dtype=bkd.int64_dtype()),
                bkd.ones((3,)),
                [1],
            )

    def test_refuses_a_weight_of_another_size(self, numpy_bkd: NumpyBkd) -> None:
        system, _ = _system(numpy_bkd)
        with pytest.raises(ValueError, match="weight acts on 4 states"):
            time_integrated_weighted_l2(
                system,
                RungeKutta(forward_euler()),
                RecomputeAlways(system.solved.solver),
                _weight(4),
            )


class TestTimeIntegratedFluxRefusals:
    def test_refuses_a_dof_that_is_not_essential(self, numpy_bkd: NumpyBkd) -> None:
        system, heat = _system(numpy_bkd)
        method: RungeKutta[_Np, _Np] = RungeKutta(forward_euler())
        with pytest.raises(ValueError, match="not essential"):
            time_integrated_essential_boundary_flux(
                system,
                method,
                RecomputeAlways(system.unconstrained.solver),
                np.array([heat.near_right]),
                np.ones((1, 1)),
            )
