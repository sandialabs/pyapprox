r"""The transient implicit map: a functional of a trajectory as a map of
the parameters, in the stack's types.

Lotka-Volterra with a mass from (10, 5), parameters (1, 0.1, 0.075, 1.5),
ten steps of 0.05, under backward Euler, Heun and BDF2 (Newton to
1e-13):

- Against the sweeps run directly on the same solve
  (``TrajectoryProblem``), for the Tikhonov-augmented MSE as a fixed
  functional: the value, the tangent along a random direction, the
  gradient and the HVP agree to 1e-12 (the same arithmetic, so this
  checks the wiring: parameters set before the solve, setups cleared,
  the initial state taken, the sweeps on the solve's stepper).
- For the integral of :math:`q = \cos(t)\, y^\top y / 100 + (s^\top
  p)(\mathbf 1^\top y)` by the method, a source whose functional depends
  on the recorded trajectory (BDF2's recursion weights): the point's
  value equals the value of a solve keeping no trajectory to 1e-14, the
  gradient passes central differences of the value and the HVP central
  differences of the gradient (DerivativeChecker, minimum ratio below
  1e-6).
- A value of two entries (two MSEs observing different stored states,
  stacked, under BDF2): each entry's value, and the gradient under unit
  weights each entry's gradient, to 1e-12.
- Refusals: arguments that do not satisfy their protocols, and an
  earlier point's derivatives after the next solve has moved the
  parameters.
"""

from typing import Callable, Generic, Optional, Union

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.multistep_methods import bdf2
from pyapprox.ode.pinned_rows import RowPinningProtocol
from pyapprox.ode.stepping_methods import (
    LinearMultistep,
    RungeKutta,
    SteppingMethodProtocol,
)
from pyapprox.ode.tableau import backward_euler, heun
from pyapprox.ode.time_integral import TimeIntegral
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.transient_implicit_map import (
    FixedTrajectoryFunctional,
    TrajectoryFunctionalSourceProtocol,
    TransientImplicitMap,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.ode.runge_kutta_cases import LV_PARAM, lotka_volterra_with_mass
from tests._helpers.ode.trajectory_cases import (
    START,
    STEPS,
    EnergyIntegrand,
    Method,
    TrajectoryProblem,
    native_tikhonov_mse,
    random_direction,
)
from tests._helpers.ode_models import LotkaVolterra

_Method = Union[RungeKutta[Array, Array], LinearMultistep[Array, Array]]


def _newton() -> Newton[Array, Array]:
    return Newton(atol=1e-13, rtol=0.0, maxiters=30)


def _backward_euler() -> _Method[Array]:
    return RungeKutta(backward_euler(), _newton())


def _heun() -> _Method[Array]:
    return RungeKutta(heun())


def _bdf2() -> _Method[Array]:
    return LinearMultistep(bdf2(), _newton())


_METHODS = [
    (_backward_euler, backward_euler),
    (_heun, heun),
    (_bdf2, bdf2),
]
_IDS = ["backward Euler", "Heun", "BDF2"]


class _Discretization(Generic[Array]):
    """Lotka-Volterra's dense mass-form system: nothing pinned, the
    initial state taken as given."""

    def __init__(self, bkd: Backend[Array], model: LotkaVolterra[Array]) -> None:
        self._bkd = bkd
        self._system = dense_mass_form_system(model)

    def system(self) -> MassFormSystem[Array, Array, DenseFormat[Array]]:
        return self._system

    def pinning(self) -> Optional[RowPinningProtocol[Array]]:
        return None

    def initial_state(self, state: Array, time: float) -> Array:
        return self._bkd.copy(state)


class _Parameters(Generic[Array]):
    """Sets the model's parameters and those of every term that reads
    them."""

    def __init__(self, *readers: Callable[[Array], None]) -> None:
        self._readers = readers

    def nparams(self) -> int:
        return 4

    def set_param(self, param: Array) -> None:
        for reader in self._readers:
            reader(param)


def _map(
    bkd: Backend[Array],
    method: SteppingMethodProtocol[Array, Array],
    model: LotkaVolterra[Array],
    functional: TrajectoryFunctionalSourceProtocol[Array, Array],
    *readers: Callable[[Array], None],
) -> TransientImplicitMap[Array, Array, DenseFormat[Array]]:
    discretization = _Discretization(bkd, model)
    return TransientImplicitMap(
        discretization,
        _Parameters(model.set_param, *readers),
        method,
        RecomputeAlways(discretization.system().solver),
        STEPS,
        bkd.asarray(START),
        functional,
    )


def _integral(
    bkd: Backend[Array],
    method: SteppingMethodProtocol[Array, Array],
    model: LotkaVolterra[Array],
    integrand: EnergyIntegrand[Array],
) -> TimeIntegral[Array, Array]:
    system = dense_mass_form_system(model)
    stepper = method.stepper(system, RecomputeAlways(system.solver))
    return TimeIntegral(integrand, stepper, bkd)


@pytest.mark.parametrize("make, tableau_or_formula", _METHODS, ids=_IDS)
class TestAgainstTheSweeps:
    def test_value_tangent_gradient_and_hvp(
        self,
        bkd: Backend[Array],
        make: Callable[[], _Method[Array]],
        tableau_or_formula: Callable[[], Method],
    ) -> None:
        oracle = TrajectoryProblem(bkd, tableau_or_formula())
        model = lotka_volterra_with_mass(bkd)
        functional, tikhonov = native_tikhonov_mse(bkd)
        implicit = _map(
            bkd,
            make(),
            model,
            FixedTrajectoryFunctional(functional),
            tikhonov.set_param,
        )
        param = bkd.asarray(LV_PARAM)
        direction = random_direction(bkd)
        bkd.assert_allclose(implicit.value(param), oracle.value(param), rtol=1e-12)
        point = implicit.point(param)
        bkd.assert_allclose(point.value(), oracle.value(param), rtol=1e-12)
        bkd.assert_allclose(
            point.tangent(direction), oracle.tangent(param, direction), rtol=1e-12
        )
        bkd.assert_allclose(point.gradient(), oracle.gradient(param).T, rtol=1e-12)
        bkd.assert_allclose(
            point.hvp(direction), oracle.hvp(param, direction), rtol=1e-12
        )


@pytest.mark.parametrize("make, tableau_or_formula", _METHODS, ids=_IDS)
class TestATimeIntegral:
    def test_derivatives_pass_finite_differences(
        self,
        bkd: Backend[Array],
        make: Callable[[], _Method[Array]],
        tableau_or_formula: Callable[[], Method],
    ) -> None:
        method = make()
        model = lotka_volterra_with_mass(bkd)
        integrand = EnergyIntegrand(bkd)
        implicit = _map(
            bkd,
            method,
            model,
            _integral(bkd, method, model, integrand),
            integrand.set_param,
        )
        param = bkd.asarray(LV_PARAM)
        bkd.assert_allclose(
            implicit.point(param).value(), implicit.value(param), rtol=1e-14
        )

        def values(samples: Array) -> Array:
            return bkd.hstack(
                [implicit.value(samples[:, k : k + 1]) for k in range(samples.shape[1])]
            )

        def gradient(sample: Array) -> Array:
            return implicit.point(sample).gradient()

        def gradients(samples: Array) -> Array:
            return bkd.hstack(
                [gradient(samples[:, k : k + 1]) for k in range(samples.shape[1])]
            )

        def hvp(sample: Array, direction: Array) -> Array:
            return implicit.point(sample).hvp(direction)

        first = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=1,
                nvars=4,
                fun=values,
                jacobian=lambda sample: gradient(sample).T,
                bkd=bkd,
            )
        ).check_derivatives(param, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(first)) < 1e-6
        second = DerivativeChecker(
            FunctionWithJVPFromCallable(
                nqoi=4, nvars=4, fun=gradients, jvp=hvp, bkd=bkd
            )
        ).check_derivatives(param, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(second)) < 1e-6


class TestTwoEntries:
    """Two MSEs observing different stored states, stacked: the value is
    both, and the gradient under unit weights each one's, to 1e-12."""

    def test_unit_weights_give_each_entrys_gradient(self, bkd: Backend[Array]) -> None:
        method: _Method[Array] = _bdf2()
        model = lotka_volterra_with_mass(bkd)
        first, first_tikhonov = native_tikhonov_mse(bkd)
        second, second_tikhonov = native_tikhonov_mse(
            bkd, prey=(1, 4, 9), predator=(6, 10)
        )
        readers = (first_tikhonov.set_param, second_tikhonov.set_param)
        param = bkd.asarray(LV_PARAM)
        stacked = FixedTrajectoryFunctional(
            TrajectoryFunctional.stack(bkd, [first, second])
        )
        point = _map(bkd, method, model, stacked, *readers).point(param)
        assert point.value().shape == (2, 1)
        gradients = []
        for entry in range(2):
            weights = bkd.copy(bkd.zeros((2, 1)))
            weights[entry, 0] = 1.0
            gradients.append(point.gradient(weights))
        for entry, alone in enumerate((first, second)):
            # A model of its own, so the stacked point stays valid.
            single = _map(
                bkd,
                method,
                lotka_volterra_with_mass(bkd),
                FixedTrajectoryFunctional(alone),
                *readers,
            ).point(param)
            bkd.assert_allclose(point.value()[entry], single.value()[0], rtol=1e-12)
            bkd.assert_allclose(gradients[entry], single.gradient(), rtol=1e-12)


class TestRefusals:
    def test_refuses_arguments_of_the_wrong_kind(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        model = lotka_volterra_with_mass(bkd)
        functional, _ = native_tikhonov_mse(bkd)
        discretization = _Discretization(bkd, model)
        arguments = [
            discretization,
            _Parameters(model.set_param),
            _backward_euler(),
            RecomputeAlways(discretization.system().solver),
            STEPS,
            bkd.asarray(START),
            FixedTrajectoryFunctional(functional),
        ]
        for position, name in (
            (0, "discretization"),
            (1, "parameters"),
            (2, "method"),
            (4, "steps"),
            (6, "functional"),
        ):
            wrong = list(arguments)
            wrong[position] = object()
            with pytest.raises(TypeError, match=name):
                TransientImplicitMap(*wrong)  # type: ignore[arg-type]

    def test_a_point_is_refused_after_the_next_solve(self, numpy_bkd: NumpyBkd) -> None:
        """The next solve moves the parameters the sweeps read, so an
        earlier point's derivatives are refused, not silently wrong."""
        bkd = numpy_bkd
        model = lotka_volterra_with_mass(bkd)
        functional, tikhonov = native_tikhonov_mse(bkd)
        implicit = _map(
            bkd,
            _backward_euler(),
            model,
            FixedTrajectoryFunctional(functional),
            tikhonov.set_param,
        )
        param = bkd.asarray(LV_PARAM)
        point = implicit.point(param)
        implicit.value(1.1 * param)
        with pytest.raises(ValueError, match="solve again"):
            point.gradient()

    def test_fixed_functional_refuses_another_kind(self) -> None:
        with pytest.raises(TypeError, match="TrajectoryFunctional"):
            FixedTrajectoryFunctional(object())  # type: ignore[arg-type]
