r"""The ODE benchmarks on the new time integration, against the old.

For each benchmark residual and each method both paths have (backward
and forward Euler, Crank-Nicolson, implicit midpoint, Heun), the new
path (``Integrator`` with the tableau stepper, ``TrajectoryDerivatives``)
must reproduce the old one (``TimeIntegrator`` with the tiered stepper,
its adjoint): the same trajectory, the same gradient of a scalar
functional, and, for residuals with second derivatives, the same
Hessian-vector product.

Cases, at the builders' nominal parameters over short horizons (ten
steps or fewer, Newton tolerance 1e-13):

- Lotka-Volterra, 3 species, 12 parameters (second order), from
  (0.3, 0.4, 0.3), T = 2, dt 0.25;
- chemical reaction surface, 6 parameters, from 0, T = 1, dt 0.1;
- coupled springs, 12 parameters, the last 4 the initial state, T = 1,
  dt 0.1;
- Hastings ecology, 9 parameters, the last 3 the initial state, T = 5,
  dt 0.5;
- the nonlinear decoupled system, 3 states, time-dependent coefficient,
  p = (c, y_0) with the initial state every entry y_0 (second order),
  T = 1, dt 0.1;
- the time-modulated quadratic, 2 states, f carrying g(t) = 1 + t/2
  (second order), from (0.5, -0.3), T = 1, dt 0.1.

The functional is a mean squared error with noise 0.5 observing every
state at step 2 and at the last step. Trajectories agree to 1e-10,
gradients and Hessian-vector products to 1e-9, relative.
"""

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.accumulating_sink import MultiSink
from pyapprox.ode.functional_sink import FunctionalSink
from pyapprox.ode.functionals.mse import TransientMSEFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import DenseModelProtocol, dense_mass_form_system
from pyapprox.ode.observation_terms import MisfitTerm
from pyapprox.ode.operator.qoi_jacobian import adjoint_jacobian
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.protocols import ODEResidualProtocol
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
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectorySinkProtocol,
)
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.util.backends.protocols import Array, ArrayProtocol, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver
from pyapprox.util.rootfinding.nonlinear import Newton
from pyapprox_benchmarks.functions.ode.nonlinear_decoupled import (
    NonLinearDecoupledODE,
)
from pyapprox_benchmarks.functions.ode.time_modulated_quadratic import (
    TimeModulatedQuadraticODE,
)
from pyapprox_benchmarks.ode import (
    build_chemical_reaction_surface,
    build_coupled_springs_2mass,
    build_hastings_ecology_3species,
    build_lotka_volterra_3species,
)
from pyapprox_benchmarks.problems.ode import ODEForwardUQProblem

# The old path's stepper name for each tableau, for the same method.
_METHODS: List[tuple[Callable[[], ButcherTableau], str]] = [
    (backward_euler, "backward_euler"),
    (forward_euler, "forward_euler"),
    (crank_nicolson, "crank_nicolson"),
    (implicit_midpoint, "implicit_midpoint"),
    (heun, "heun"),
]


@runtime_checkable
class _Residual(DenseModelProtocol[Array], ODEResidualProtocol[Array], Protocol[Array]):
    """What both paths need of a benchmark residual: the new path's dense
    model, the old path's ODE residual, and its parameters."""

    def set_param(self, param: Array) -> None: ...


_A_co = TypeVar("_A_co", bound=ArrayProtocol, covariant=True)


@runtime_checkable
class _StartFromParameters(Protocol[_A_co]):
    """A residual whose initial state is among its parameters."""

    def get_initial_condition(self) -> _A_co: ...


@dataclass(frozen=True)
class _Case(Generic[Array]):
    """A benchmark residual at its nominal parameters, on one backend.

    Attributes
    ----------
    residual : callable
        ``() ->`` a fresh residual.
    param : Array
        The nominal parameters, 1-D.
    start : callable
        ``(residual, param) ->`` the initial state, shape (n, 1), after
        ``set_param``.
    final_time, deltat : float
        The horizon and step.
    second_order : bool
        Whether the residual has second derivatives (and an HVP).
    old_adjoint : bool
        Whether the old path can differentiate it: its adjoint needs the
        old method-style ``param_jacobian``. Without it the new gradient
        and HVP are checked against finite differences instead.
    """

    residual: Callable[[], _Residual[Array]]
    param: Array
    start: Callable[[_Residual[Array], Array], Array]
    final_time: float
    deltat: float
    second_order: bool
    old_adjoint: bool = True


class _CaseFactory(Protocol):
    """Builds a case on any backend."""

    def __call__(self, bkd: Backend[Array]) -> _Case[Array]: ...


def _checked(residual: object) -> _Residual[Array]:
    """``residual``, refused unless both paths can use it."""
    if not isinstance(residual, _Residual):
        raise TypeError(f"{type(residual).__name__} does not fit both paths")
    return residual


def _from_builder(
    bkd: Backend[Array],
    builder: Callable[[Backend[Array]], ODEForwardUQProblem[Array]],
    final_time: float,
    deltat: float,
    second_order: bool,
) -> _Case[Array]:
    """A case from a benchmark builder: the builder's start, or the
    residual's own when its initial state is among its parameters."""
    problem = builder(bkd)
    builder_start = problem.initial_condition()

    def start(residual: _Residual[Array], param: Array) -> Array:
        if isinstance(residual, _StartFromParameters):
            return bkd.reshape(residual.get_initial_condition(), (-1, 1))
        return bkd.copy(builder_start)

    return _Case(
        residual=lambda: _checked(builder(bkd).residual()),
        param=bkd.flatten(problem.nominal_parameters()),
        start=start,
        final_time=final_time,
        deltat=deltat,
        second_order=second_order,
    )


def _lotka_volterra(bkd: Backend[Array]) -> _Case[Array]:
    return _from_builder(bkd, build_lotka_volterra_3species, 2.0, 0.25, True)


def _chemical(bkd: Backend[Array]) -> _Case[Array]:
    # Stiff: dt 0.01 keeps the explicit methods and Crank-Nicolson's
    # Newton inside their working range.
    return _from_builder(bkd, build_chemical_reaction_surface, 0.1, 0.01, False)


def _springs(bkd: Backend[Array]) -> _Case[Array]:
    return _from_builder(bkd, build_coupled_springs_2mass, 1.0, 0.1, False)


def _hastings(bkd: Backend[Array]) -> _Case[Array]:
    return _from_builder(bkd, build_hastings_ecology_3species, 5.0, 0.5, False)


def _decoupled(bkd: Backend[Array]) -> _Case[Array]:
    return _Case(
        residual=lambda: _checked(NonLinearDecoupledODE(3, True, bkd)),
        param=bkd.asarray([0.5, 1.2]),
        start=lambda residual, param: param[1] * bkd.ones((3, 1)),
        final_time=1.0,
        deltat=0.1,
        second_order=True,
        old_adjoint=False,
    )


def _time_modulated(bkd: Backend[Array]) -> _Case[Array]:
    matrix = bkd.asarray([[-1.0, 0.2], [0.1, -0.8]])
    return _Case(
        residual=lambda: _checked(TimeModulatedQuadraticODE(matrix, bkd)),
        param=bkd.asarray([0.3, 0.1]),
        start=lambda residual, param: bkd.asarray([[0.5], [-0.3]]),
        final_time=1.0,
        deltat=0.1,
        second_order=True,
    )


_CASES: Dict[str, _CaseFactory] = {
    "lotka_volterra": _lotka_volterra,
    "chemical": _chemical,
    "springs": _springs,
    "hastings": _hastings,
    "decoupled": _decoupled,
    "time_modulated": _time_modulated,
}


def _functional(
    bkd: Backend[Array], nstates: int, nparams: int, last: int
) -> TransientMSEFunctional[Array]:
    """MSE with noise 0.5 observing every state at step 2 and the last."""
    steps = bkd.asarray([2, last], dtype=int)
    mse = TransientMSEFunctional(
        nstates, nparams, [(k, steps) for k in range(nstates)], bkd, noise_std=0.5
    )
    mse.set_observations(bkd.asarray(_observations(nstates)))
    return mse


def _observations(nstates: int) -> List[float]:
    return [0.3 * (k + 1) for k in range(2 * nstates)]


def _misfit(
    bkd: Backend[Array], nstates: int, last: int
) -> TrajectoryFunctional[Array, Array]:
    """:func:`_functional` as the new path's native misfit term."""
    misfit: MisfitTerm[Array, Array] = MisfitTerm.from_state_times(
        bkd,
        nstates,
        [(k, (2, last)) for k in range(nstates)],
        bkd.asarray(_observations(nstates)),
        noise_std=0.5,
    )
    return TrajectoryFunctional.from_terms(bkd, 1, state_terms=(misfit,))


class _Paths(Generic[Array]):
    """One case on both paths, at its nominal parameters or at
    ``param`` (1-D) when given."""

    def __init__(
        self,
        bkd: Backend[Array],
        case: _Case[Array],
        make: Callable[[], ButcherTableau],
        old_name: str,
        param: Optional[Array] = None,
    ) -> None:
        self._bkd, self._case = bkd, case
        self._param = case.param if param is None else param
        self._column = bkd.reshape(self._param, (-1, 1))
        nsteps = round(case.final_time / case.deltat)
        self._times = [case.deltat * n for n in range(nsteps + 1)]
        # The new path.
        self._residual = case.residual()
        self._residual.set_param(self._param)
        self._start = case.start(self._residual, self._param)
        tableau = make()
        newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        self._stepper = RungeKuttaStepper(
            tableau,
            dense_mass_form_system(self._residual),
            None if tableau.is_explicit() else newton,
        )
        self._store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        nstates = self._start.shape[0]
        self._misfit = _misfit(bkd, nstates, nsteps)
        value_sink: FunctionalSink[Array, Array] = FunctionalSink(
            self._misfit.zero, state_terms=self._misfit.state_terms
        )
        sinks: List[TrajectorySinkProtocol[Array]] = [value_sink, self._store]
        # The old path's grid, dt n, exactly.
        steps = GivenSizes.from_times(self._times)
        Integrator(self._stepper).solve(self._start, steps, MultiSink(sinks))
        self._new_value = value_sink.value()
        self._sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            self._stepper,
            None,
            ResidualInitialState(self._residual, BackendColumnForm(bkd)),
        )
        # The old path, on its own residual.
        self._old_residual = case.residual()
        self._old_residual.set_param(self._param)
        stepper = create_stepper(old_name, self._old_residual)
        old_newton = NewtonSolver(stepper)
        old_newton.set_options(maxiters=30, atol=1e-13, rtol=0.0)
        self._integrator = TimeIntegrator(0.0, case.final_time, case.deltat, old_newton)
        self._old_states, self._old_times = self._integrator.solve(
            bkd.flatten(self._start)
        )
        self._functional_args = (nstates, self._param.shape[0], nsteps)

    def functional(self) -> TransientMSEFunctional[Array]:
        """The old path's functional."""
        return _functional(self._bkd, *self._functional_args)

    def new_value(self) -> Array:
        """Q on the new path, the native misfit summed during the solve.
        Shape: (1, 1)."""
        return self._new_value

    def new_states(self) -> Array:
        return self._bkd.hstack(
            [self._store.state(n) for n in range(self._store.ntimes())]
        )

    def old_states(self) -> Array:
        return self._old_states

    def new_gradient(self) -> Array:
        return self._sweeps.gradient(self._store, self._misfit)

    def old_gradient(self) -> Array:
        row = adjoint_jacobian(
            self._integrator,
            self.functional(),
            self._old_states,
            self._old_times,
            self._column,
        )
        return self._bkd.reshape(row, (-1, 1))

    def new_hvp(self, direction: Array) -> Array:
        return self._sweeps.hvp(self._store, self._misfit, direction)

    def old_hvp(self, direction: Array) -> Array:
        operator = TimeAdjointOperatorWithHVP(self._integrator, self.functional())
        product = operator.hvp(self._bkd.flatten(self._start), self._column, direction)
        return self._bkd.reshape(product, (-1, 1))


def _direction(bkd: Backend[Array], n: int) -> Array:
    return bkd.asarray(np.random.default_rng(11).normal(size=(n, 1)))


def _fd_errors(
    bkd: Backend[Array],
    case: _Case[Array],
    make: Callable[[], ButcherTableau],
    old_name: str,
) -> tuple[float, float]:
    """The smallest central-difference errors of the new path's gradient
    (against its Q) and HVP (against its gradient), for a case the old
    path cannot differentiate."""

    def paths(sample: Array) -> _Paths[Array]:
        return _Paths(bkd, case, make, old_name, bkd.flatten(sample))

    def values(samples: Array) -> Array:
        return bkd.hstack(
            [paths(samples[:, k : k + 1]).new_value() for k in range(samples.shape[1])]
        )

    def gradients(samples: Array) -> Array:
        return bkd.hstack(
            [
                paths(samples[:, k : k + 1]).new_gradient()
                for k in range(samples.shape[1])
            ]
        )

    nominal = bkd.reshape(case.param, (-1, 1))
    n = nominal.shape[0]
    first = FunctionWithJacobianFromCallable(
        nqoi=1,
        nvars=n,
        fun=values,
        jacobian=lambda sample: paths(sample).new_gradient().T,
        bkd=bkd,
    )
    second = FunctionWithJVPFromCallable(
        nqoi=n,
        nvars=n,
        fun=gradients,
        jvp=lambda sample, vec: paths(sample).new_hvp(vec),
        bkd=bkd,
    )
    errors = [
        DerivativeChecker(wrapper).check_derivatives(
            nominal, relative=True, central=True
        )[0]
        for wrapper in (first, second)
    ]
    return bkd.to_float(bkd.min(errors[0])), bkd.to_float(bkd.min(errors[1]))


_METHOD_IDS = [name for _, name in _METHODS]


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("make, old_name", _METHODS, ids=_METHOD_IDS)
class TestAgainstTheOldPath:
    def test_trajectory(
        self,
        bkd: Backend[Array],
        name: str,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        paths = _Paths(bkd, _CASES[name](bkd), make, old_name)
        bkd.assert_allclose(
            paths.new_states(), paths.old_states(), rtol=1e-10, atol=1e-13
        )

    def test_gradient(
        self,
        bkd: Backend[Array],
        name: str,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        case = _CASES[name](bkd)
        if not case.old_adjoint:
            gradient_error, hvp_error = _fd_errors(bkd, case, make, old_name)
            assert gradient_error < 1e-7 and hvp_error < 1e-7
            return
        paths = _Paths(bkd, case, make, old_name)
        bkd.assert_allclose(paths.new_gradient(), paths.old_gradient(), rtol=1e-9)

    def test_hvp(
        self,
        bkd: Backend[Array],
        name: str,
        make: Callable[[], ButcherTableau],
        old_name: str,
    ) -> None:
        case = _CASES[name](bkd)
        if not case.second_order:
            pytest.skip("the residual has no second derivatives")
        if not case.old_adjoint:
            pytest.skip("checked against finite differences in test_gradient")
        paths = _Paths(bkd, case, make, old_name)
        direction = _direction(bkd, case.param.shape[0])
        bkd.assert_allclose(
            paths.new_hvp(direction), paths.old_hvp(direction), rtol=1e-9
        )
