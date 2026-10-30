"""Stepping methods bind to a system and build the steppers directly built.

Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` from (10, 5),
parameters (1, 0.1, 0.075, 1.5), dt 0.05 to 0.5:

- a method with Newton (atol 1e-13) gives the trajectory of the stepper
  built by hand with the same solver, exactly, for every tableau and
  multistep method;
- an implicit method without a solver defaults to Newton at its defaults,
  agreeing with the tight one to Newton's tolerance;
- row pinning is forwarded (the pinned row holds its data);
- refusals: an explicit method given a solver, a solver or coefficients of
  the wrong type.
"""

from typing import Any, Callable, Generic, List

import pytest
from numpy.typing import NDArray

from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.multistep_methods import adams_bashforth2, bdf2
from pyapprox.ode.pinned_rows import RowPinning
from pyapprox.ode.protocols import StepperProtocol
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.ode.stepping_methods import (
    LinearMultistep,
    RungeKutta,
    SteppingMethodProtocol,
)
from pyapprox.ode.tableau import ButcherTableau, backward_euler, heun
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet, DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.ode.multistep_cases import METHOD_IDS, METHODS
from tests._helpers.ode.runge_kutta_cases import (
    LV_PARAM,
    TABLEAU_IDS,
    TABLEAUX,
    lotka_volterra_with_mass,
)
from tests._helpers.ode.trajectory_cases import TIMES, Method, make_stepper


def _method(method: Method, tight: bool) -> SteppingMethodProtocol[Array, Array]:
    """``method`` as a method object, with Newton at atol 1e-13 when
    ``tight`` and the method is implicit, its default otherwise."""
    newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
    solver = newton if tight and not method.is_explicit() else None
    if isinstance(method, ButcherTableau):
        return RungeKutta(method, solver)
    return LinearMultistep(method, solver)


def _trajectory(bkd: Backend[Array], stepper: StepperProtocol[Array, Array]) -> Array:
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    Integrator(stepper).solve(bkd.asarray([[10.0], [5.0]]), TIMES, store)
    return bkd.hstack([store.state(n) for n in range(store.ntimes())])


def _system(bkd: Backend[Array]) -> MassFormSystem[Array, Array, DenseFormat[Array]]:
    """Lotka-Volterra with a mass at the nominal parameters."""
    model = lotka_volterra_with_mass(bkd)
    model.set_param(bkd.asarray(LV_PARAM))
    return dense_mass_form_system(model)


_ALL: List[Callable[[], Method]] = [
    *(make for make, *_ in TABLEAUX),
    *(make for make, *_ in METHODS),
]
_ALL_IDS = [*TABLEAU_IDS, *METHOD_IDS]


class TestBinding:
    @pytest.mark.parametrize("make", _ALL, ids=_ALL_IDS)
    def test_matches_the_stepper_built_by_hand(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        system = _system(bkd)
        method: SteppingMethodProtocol[Array, Array] = _method(make(), tight=True)
        stepper = method.stepper(system)
        assert isinstance(stepper, StepperProtocol)
        bkd.assert_allclose(
            _trajectory(bkd, stepper),
            _trajectory(bkd, make_stepper(make(), system)),
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize(
        "make", [backward_euler, bdf2], ids=["backward_euler", "bdf2"]
    )
    def test_an_implicit_method_defaults_to_newton(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        system = _system(bkd)
        default_method: SteppingMethodProtocol[Array, Array] = _method(
            make(), tight=False
        )
        tight_method: SteppingMethodProtocol[Array, Array] = _method(make(), tight=True)
        default = _trajectory(bkd, default_method.stepper(system))
        tight = _trajectory(bkd, tight_method.stepper(system))
        bkd.assert_allclose(default, tight, rtol=1e-6)

    @pytest.mark.parametrize("make", [heun, bdf2], ids=["heun", "bdf2"])
    def test_row_pinning_is_forwarded(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        states, _ = backend_spaces(bkd, 2, 4)
        rows: BackendRowSet[Array] = BackendRowSet(states.descriptor(), [0], bkd)
        pinning: RowPinning[Array] = RowPinning(_Fixed(bkd, rows))
        method: SteppingMethodProtocol[Array, Array] = _method(make(), tight=True)
        stepper = method.stepper(_system(bkd), pinning)
        trajectory = _trajectory(bkd, stepper)
        bkd.assert_allclose(
            trajectory[0, 1:], bkd.full((trajectory.shape[1] - 1,), 7.0)
        )

    def test_names(self) -> None:
        assert RungeKutta(heun()).name() == heun().name
        assert LinearMultistep(bdf2()).name() == bdf2().name
        assert repr(RungeKutta(heun())) == f"RungeKutta({heun().name})"


class _Fixed(Generic[Array]):
    """Row 0 pinned to 7."""

    def __init__(self, bkd: Backend[Array], rows: BackendRowSet[Array]) -> None:
        self._bkd, self._rows = bkd, rows

    def rows(self) -> BackendRowSet[Array]:
        return self._rows

    def values(self, time: float) -> Array:
        return self._bkd.asarray([[7.0], [0.0]])


class TestRefusals:
    def test_an_explicit_method_takes_no_solver(self) -> None:
        newton: Newton[NDArray[Any], NDArray[Any]] = Newton()
        with pytest.raises(TypeError, match="explicit"):
            RungeKutta(heun(), newton)
        with pytest.raises(TypeError, match="explicit"):
            LinearMultistep(adams_bashforth2(), newton)

    def test_wrong_types(self) -> None:
        with pytest.raises(TypeError, match="NonlinearSolverProtocol"):
            RungeKutta(backward_euler(), object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ButcherTableau"):
            RungeKutta(bdf2())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="MultistepFormulaProtocol"):
            LinearMultistep(backward_euler())  # type: ignore[arg-type]
