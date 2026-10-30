"""ODE benchmarks whose parameters include the initial state.

The coupled-springs and Hastings residuals define ``y_0(p)`` from their last
parameters (``get_initial_condition``). The QoI function must use it, so
those parameters reach the solution, and the derivatives the residual states
for them (``initial_param_jacobian``) must match the forward map.

The derivative references need no derivation: finite differences of the
benchmark's own QoI function.
"""

from typing import Callable, Dict, Protocol, Tuple

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.marginalize import ActiveSetFunction
from pyapprox.ode.functionals.all_states_endpoint import (
    AllStatesEndpointFunctional,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.operator.forward_sensitivity import (
    forward_sensitivity_jacobian,
)
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import crank_nicolson
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver
from pyapprox_benchmarks.functions.ode import ODEBenchmarkResidualProtocol
from pyapprox_benchmarks.functions.ode.ode_qoi import (
    ODEResidualWithInitialStateProtocol,
)
from pyapprox_benchmarks.ode import (
    build_coupled_springs_2mass,
    build_hastings_ecology_3species,
)
from pyapprox_benchmarks.problems.ode import ODEForwardUQProblem


class _Builder(Protocol):
    """Builds an ODE benchmark problem on any backend."""

    def __call__(self, bkd: Backend[Array]) -> ODEForwardUQProblem[Array]: ...


# (builder, number of physical parameters before the initial conditions)
_CASES: Dict[str, Tuple[_Builder, int]] = {
    "coupled_springs": (build_coupled_springs_2mass, 8),
    "hastings": (build_hastings_ecology_3species, 6),
}
_STEPPER = "crank_nicolson"


def _problem(
    bkd: Backend[Array], name: str
) -> Tuple[ODEForwardUQProblem[Array], int]:
    builder, nphys = _CASES[name]
    return builder(bkd), nphys


def _initial_state(residual: ODEBenchmarkResidualProtocol[Array]) -> Array:
    """``y_0(p)`` at the residual's current parameters; these cases'
    residuals all define it."""
    if not isinstance(residual, ODEResidualWithInitialStateProtocol):
        raise TypeError(f"{type(residual).__name__} defines no initial state")
    # isinstance cannot check the array type, so bind it here.
    start: ODEResidualWithInitialStateProtocol[Array] = residual
    return start.get_initial_condition()


def _integrate(
    problem: ODEForwardUQProblem[Array], param_1d: Array, init_state: Array
) -> Tuple[TimeIntegrator[Array], Array, Array]:
    """Solve from an explicit initial state; return (stepper, sols, times)."""
    residual = problem.residual()
    residual.set_param(param_1d)
    stepper = create_stepper(_STEPPER, residual)
    newton = NewtonSolver(stepper)
    newton.set_options(maxiters=30, atol=1e-13, rtol=0.0)
    tc = problem.time_config()
    integrator = TimeIntegrator(tc.init_time, tc.final_time, tc.deltat, newton)
    sols, times = integrator.solve(init_state)
    return integrator, sols, times


def _final_states(
    problem: ODEForwardUQProblem[Array], bkd: Backend[Array]
) -> Callable[[Array], Array]:
    """Finite-difference reference: final states from y_0(p), solved tightly.

    Uses the residual's own initial-state map (as the QoI function does)
    with a tight Newton tolerance, so the reference is accurate to ~1e-8
    rather than to the QoI function's default tolerance.
    """
    residual = problem.residual()

    def fun(samples: Array) -> Array:
        cols = []
        for ii in range(samples.shape[1]):
            p = samples[:, ii]
            residual.set_param(p)
            _, sols, _ = _integrate(problem, p, _initial_state(residual))
            cols.append(sols[:, -1])
        return bkd.stack(cols, axis=1)

    return fun


@pytest.mark.parametrize("name", list(_CASES))
class TestInitialStateParameters:
    def test_initial_state_parameters_reach_the_qoi(
        self, bkd: Backend[Array], name: str
    ) -> None:
        problem, nphys = _problem(bkd, name)
        func = problem.function(method=RungeKutta(crank_nicolson()))
        base = bkd.copy(problem.nominal_parameters())
        moved = bkd.copy(base)
        moved[nphys, 0] = moved[nphys, 0] + 0.05
        diff = bkd.max(bkd.abs(func(moved) - func(base)))
        assert float(bkd.to_numpy(diff)) > 1e-4

    def test_active_set_recovers_the_fixed_initial_state(
        self, bkd: Backend[Array], name: str
    ) -> None:
        """Fixing the initial-condition parameters at their nominal values
        starts every solve from the problem's nominal initial condition,
        which is what the QoI function used for every sample before."""
        problem, nphys = _problem(bkd, name)
        func = problem.function(method=RungeKutta(crank_nicolson()))
        nominal = bkd.flatten(problem.nominal_parameters())
        residual = problem.residual()
        residual.set_param(nominal)
        bkd.assert_allclose(
            _initial_state(residual),
            bkd.flatten(problem.initial_condition()),
            rtol=0.0,
            atol=0.0,
        )
        reduced = ActiveSetFunction(func, nominal, list(range(nphys)), bkd)
        physical = bkd.stack(
            [nominal[:nphys], 1.02 * nominal[:nphys]], axis=1
        )
        full = bkd.vstack(
            [physical, bkd.stack([nominal[nphys:]] * 2, axis=1)]
        )
        bkd.assert_allclose(reduced(physical), func(full), rtol=0.0, atol=0.0)

    def test_adjoint_gradient_includes_initial_state(
        self, bkd: Backend[Array], name: str
    ) -> None:
        problem, _ = _problem(bkd, name)
        nparams = problem.nparams()
        nstates = problem.nstates()
        state_idx = 0
        final_states = _final_states(problem, bkd)

        def qoi(samples: Array) -> Array:
            return final_states(samples)[state_idx : state_idx + 1, :]

        residual = problem.residual()

        def gradient(sample: Array) -> Array:
            p = bkd.flatten(sample)
            residual.set_param(p)
            init = _initial_state(residual)
            integrator, sols, times = _integrate(problem, p, init)
            integrator.set_functional(
                EndpointFunctional(state_idx, nstates, nparams, bkd)
            )
            return integrator.gradient(sols, times, sample)

        _check(bkd, qoi, gradient, 1, nparams, problem.nominal_parameters())

    def test_forward_sensitivity_includes_initial_state(
        self, bkd: Backend[Array], name: str
    ) -> None:
        problem, _ = _problem(bkd, name)
        nparams = problem.nparams()
        qoi = _final_states(problem, bkd)
        residual = problem.residual()

        def sensitivity(sample: Array) -> Array:
            p = bkd.flatten(sample)
            residual.set_param(p)
            init = _initial_state(residual)
            integrator, sols, times = _integrate(problem, p, init)
            return forward_sensitivity_jacobian(
                integrator,
                AllStatesEndpointFunctional(problem.nstates(), nparams, bkd),
                sols,
                times,
                sample,
            )

        _check(
            bkd, qoi, sensitivity, problem.nstates(), nparams,
            problem.nominal_parameters(),
        )


def _check(
    bkd: Backend[Array],
    qoi: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    nqoi: int,
    nparams: int,
    sample: Array,
) -> None:
    wrapper = FunctionWithJacobianFromCallable(
        nqoi=nqoi, nvars=nparams, fun=qoi, jacobian=jacobian, bkd=bkd
    )
    checker = DerivativeChecker(wrapper)
    errors = checker.check_derivatives(sample, relative=True)[0]
    # One-sided differences of the tightly solved reference bottom out at
    # 1e-7 to 7e-7 here (the Hastings dynamics amplify roundoff over 40
    # steps); a wrong initial-state column plateaus at O(1) instead.
    assert float(bkd.to_numpy(bkd.min(errors))) <= 1e-6
    assert float(bkd.to_numpy(checker.error_ratio(errors))) <= 5e-6
