r"""ODE QoI functions on the new time integration, their QoIs streamed.

The 3-species Lotka-Volterra problem at its nominal parameters, T = 1.05,
dt = 0.1 (so the last step is shortened to 0.05) unless stated:

- **Composition:** the endpoint QoIs of species 0 and of species 2, fed
  from one solve through a :class:`MultiSink`, equal rows 0 and 2 of the
  all-states endpoint QoI from one sink; species 0 and species 2 at
  every time, likewise, equal the observation QoI (both together in one
  accumulator). Exact: the paths compose sinks differently over the same
  arithmetic.
- **Definitions:** every functional's streamed value equals its
  definition applied to the stored trajectory (last column, one entry of
  it, row maxima, the selected rows and steps). Exact.
- **Old path:** the default method (backward Euler, Newton at its
  defaults) and Crank-Nicolson give the old integrator's endpoint
  values to 1e-10, relative, on the builder's grid (T = 10, dt = 1) and
  on the shortened grid; BDF2 runs on the shortened grid.
- The time grid shortens the last step and counts it; refusals.
"""

from typing import List

import pytest

from pyapprox.ode.accumulating_sink import MultiSink, PerStateSink
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.multistep_methods import bdf2
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.ode.stepping_methods import (
    LinearMultistep,
    RungeKutta,
    SteppingMethodProtocol,
)
from pyapprox.ode.tableau import backward_euler, crank_nicolson
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.newton import NewtonSolver
from pyapprox_benchmarks.expdesign.lotka_volterra import (
    ObservationFunctional,
    PredictionFunctional,
)
from pyapprox_benchmarks.functions.ode import (
    AllStatesEndpointODEFunctional,
    MaxODEFunctional,
    ODEFunctionalProtocol,
    ODEQoIFunction,
    ODETimeConfig,
    SelectedStateEntries,
    SingleStateEndpointODEFunctional,
)
from pyapprox_benchmarks.ode.lotka_volterra import build_lotka_volterra_3species
from pyapprox_benchmarks.problems.ode import ODEForwardUQProblem

_FINAL, _DELTAT = 1.05, 0.1


def _problem(
    bkd: Backend[Array], final: float = _FINAL, deltat: float = _DELTAT
) -> ODEForwardUQProblem[Array]:
    return build_lotka_volterra_3species(bkd, final_time=final, deltat=deltat)


class TestComposition:
    def test_two_scalar_endpoints_match_the_vector_one(
        self, bkd: Backend[Array]
    ) -> None:
        problem = _problem(bkd)
        param = problem.nominal_parameters()
        vector = problem.function(AllStatesEndpointODEFunctional(3))(param)
        first = SingleStateEndpointODEFunctional[Array](0).sink()
        third = SingleStateEndpointODEFunctional[Array](2).sink()
        problem.function().solve_into(param, MultiSink([first, third]))
        bkd.assert_allclose(first.value(), vector[0:1, 0], rtol=0.0, atol=0.0)
        bkd.assert_allclose(third.value(), vector[2:3, 0], rtol=0.0, atol=0.0)

    def test_two_species_series_match_the_observation(
        self, bkd: Backend[Array]
    ) -> None:
        problem = _problem(bkd)
        param = problem.nominal_parameters()
        ntimes = problem.time_config().ntimes()
        observed = problem.function(ObservationFunctional(ntimes, bkd))(param)
        species0: PerStateSink[Array, Array] = PerStateSink(
            SelectedStateEntries((0,), bkd)
        )
        species2: PerStateSink[Array, Array] = PerStateSink(
            SelectedStateEntries((2,), bkd)
        )
        problem.function().solve_into(param, MultiSink([species0, species2]))
        bkd.assert_allclose(
            bkd.concatenate([species0.value(), species2.value()]),
            observed[:, 0],
            rtol=0.0,
            atol=0.0,
        )


class TestDefinitions:
    def test_streamed_values_are_their_definitions(self, bkd: Backend[Array]) -> None:
        problem = _problem(bkd)
        param = problem.nominal_parameters()
        sol, _ = problem.function().solve_trajectory(param)
        ntimes = problem.time_config().ntimes()
        assert sol.shape == (3, ntimes)
        odd = list(range(1, ntimes, 2))
        cases: List[tuple[ODEFunctionalProtocol[Array], Array]] = [
            (AllStatesEndpointODEFunctional(3), sol[:, -1]),
            (SingleStateEndpointODEFunctional(1), sol[1:2, -1]),
            (MaxODEFunctional(3, bkd), bkd.max(sol, axis=1)),
            (
                ObservationFunctional(ntimes, bkd),
                bkd.concatenate([sol[0, :], sol[2, :]]),
            ),
            (PredictionFunctional(ntimes, bkd), sol[1, odd]),
        ]
        for functional, expected in cases:
            value = problem.function(functional)(param)
            assert value.shape == (functional.nqoi(), 1)
            bkd.assert_allclose(value[:, 0], expected, rtol=0.0, atol=0.0)


def _old_endpoint(
    bkd: Backend[Array], problem: ODEForwardUQProblem[Array], name: str
) -> Array:
    """The old integrator's final state at the nominal parameters."""
    residual = problem.residual()
    residual.set_param(bkd.flatten(problem.nominal_parameters()))
    config = problem.time_config()
    integrator = TimeIntegrator(
        config.init_time,
        config.final_time,
        config.deltat,
        NewtonSolver(create_stepper(name, residual)),
    )
    sols, _ = integrator.solve(bkd.flatten(problem.initial_condition()))
    return sols[:, -1]


class TestAgainstTheOldPath:
    @pytest.mark.parametrize(
        "final, deltat", [(10.0, 1.0), (_FINAL, _DELTAT)], ids=["builder", "short"]
    )
    @pytest.mark.parametrize("name", ["backward_euler", "crank_nicolson"])
    def test_endpoint_values(
        self, bkd: Backend[Array], final: float, deltat: float, name: str
    ) -> None:
        problem = _problem(bkd, final, deltat)
        method: SteppingMethodProtocol[Array, Array] = (
            RungeKutta(crank_nicolson())
            if name == "crank_nicolson"
            else RungeKutta(backward_euler())
        )
        new = problem.function(method=method)(problem.nominal_parameters())
        bkd.assert_allclose(new[:, 0], _old_endpoint(bkd, problem, name), rtol=1e-10)

    def test_the_default_is_backward_euler(self, bkd: Backend[Array]) -> None:
        problem = _problem(bkd)
        param = problem.nominal_parameters()
        bkd.assert_allclose(
            problem.function()(param),
            problem.function(method=RungeKutta(backward_euler()))(param),
            rtol=0.0,
            atol=0.0,
        )

    def test_a_multistep_method_on_the_shortened_grid(
        self, bkd: Backend[Array]
    ) -> None:
        problem = _problem(bkd)
        method: SteppingMethodProtocol[Array, Array] = LinearMultistep(bdf2())
        value = problem.function(method=method)(problem.nominal_parameters())
        assert value.shape == (3, 1)


class TestTimeGrid:
    def test_the_last_step_is_shortened_and_counted(self) -> None:
        config = ODETimeConfig(0.0, _FINAL, _DELTAT)
        times = config.times()
        assert len(times) == config.ntimes() == 12
        assert times[-1] == _FINAL
        assert times[-1] - times[-2] == pytest.approx(0.05, abs=1e-12)
        assert ODETimeConfig(0.0, 1.0, 0.25).times() == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_rounding_adds_no_step(self) -> None:
        # Summing 0.1 a thousand times falls 1.4e-12 short of 100.
        config = ODETimeConfig(0.0, 100.0, 0.1)
        times = config.times()
        assert config.ntimes() == len(times) == 1001
        assert times[-1] == 100.0
        steps = [later - earlier for earlier, later in zip(times, times[1:])]
        assert min(steps) == pytest.approx(0.1, rel=1e-9)


class TestRefusals:
    def test_arguments_of_the_wrong_type(self, bkd: Backend[Array]) -> None:
        problem = _problem(bkd)
        residual, init = problem.residual(), problem.initial_condition()
        config = problem.time_config()
        functional: AllStatesEndpointODEFunctional[Array] = (
            AllStatesEndpointODEFunctional(3)
        )
        with pytest.raises(TypeError, match="ODEBenchmarkResidualProtocol"):
            ODEQoIFunction(object(), init, config, functional)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ODEFunctionalProtocol"):
            ODEQoIFunction(residual, init, config, object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="SteppingMethodProtocol"):
            ODEQoIFunction(
                residual,
                init,
                config,
                functional,
                object(),  # type: ignore[arg-type]
            )
