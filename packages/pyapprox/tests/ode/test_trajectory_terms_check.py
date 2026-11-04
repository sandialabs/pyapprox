r"""Finite-difference conformance of a functional's terms.

Lotka-Volterra with a mass from (10, 5), parameters (1, 0.1, 0.075, 1.5),
ten steps of 0.05. The terms: the time integral of :math:`q = \cos(t)\,
y^\top y / 100 + (s^\top p)(\mathbf 1^\top y)` by the stepper's own
method (a recursion-weighted quadrature step term); an observation state
term :math:`(c^\top p)\, y_m^\top y_m / 10` at stored states 2, 5 and the
last; a Tikhonov parameter term :math:`0.15\, p^\top p`. Every check is
judged by the report with its default tolerances.

- Every term passes everywhere it reads the trajectory, first and second
  derivatives, for every tableau and multistep method; so does a stack of
  the integral over the observation under weights (0.7, -1.3), each term
  read through its own rows.
- Planted faults are named: a state derivative scaled by 1.01 (its value
  check and the two second derivatives built on it), a parameter
  dependence the bundle leaves out (compared with zero), a mixed second
  derivative scaled by 1.01, a transpose scaled by 1.01 (the dot test,
  with the finite differences of its forward action passing), a step
  curvature scaled by 1.01, a dropped lag derivative.
- On an opaque stack (mock vectors, no indexing) the integral of
  :math:`y^\top y`, through the space's inner product, passes for forward
  and backward Euler, its unknown and lag derivatives checked.
- The adjoint sweep reads the trajectory only backward: a store refusing
  any state or window later than the last read serves the gradient of
  the integral plus the native terms (backward Euler, BDF2), equal to the
  gradient over an unrestricted store to 1e-12.
- A value-only term is checked for nothing. Refusals: a non-stepper, a
  non-term, curvatures without parameters, parameters without
  ``set_param``.
"""

from dataclasses import replace
from typing import Any, Callable, Generic, Optional

import pytest
from numpy.typing import NDArray

from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import bdf2
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import ButcherTableau, backward_euler, forward_euler, heun
from pyapprox.ode.time_integral import IntegrandDerivatives, TimeIntegral
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import (
    BackendScalarGradient,
    LinearizationProtocol,
    ParameterTermDerivatives,
    StateTermDerivatives,
    StateTermProtocol,
    StepTermDerivatives,
    StepTermProtocol,
    TrajectoryFunctional,
)
from pyapprox.ode.trajectory_terms_check import TrajectoryTermChecker
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators.protocols import InnerProductSpaceProtocol
from tests._helpers.linalg.mock_stack import MockVector
from tests._helpers.ode.runge_kutta_cases import (
    LV_PARAM,
    backend_sample,
    lotka_volterra_with_mass,
    mock_sample,
    mock_stepper,
)
from tests._helpers.ode.trajectory_cases import (
    ALL_IDS,
    ALL_METHODS,
    START,
    STEPS,
    EnergyIntegrand,
    Method,
    Stepper,
    make_stepper,
)


class _Observation(Generic[Array]):
    r""":math:`g_m = (c^\top p)\, y_m^\top y_m / 10` at stored states 2, 5
    and the last, :math:`c = (0.2, 0.4, -0.1, 0.3)`, with every second
    derivative (:math:`g_{pp} = 0`)."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._c = bkd.asarray([[0.2], [0.4], [-0.1], [0.3]])
        self._param = bkd.asarray(LV_PARAM)

    def set_param(self, param: Array) -> None:
        self._param = param

    def nqoi(self) -> int:
        return 1

    def reads(self, index: int, last: bool) -> bool:
        return last or index in (2, 5)

    def _weight(self) -> Array:
        return self._bkd.dot(self._c.T, self._param)

    def value(self, index: int, time: float, state: Array) -> Array:
        return self._weight() * self._bkd.dot(state.T, state) / 10.0

    def _wrt_state(
        self, index: int, time: float, state: Array
    ) -> BackendScalarGradient[Array]:
        return BackendScalarGradient(2.0 * self._weight() * state / 10.0, self._bkd)

    def _wrt_param(
        self, index: int, time: float, state: Array
    ) -> BackendScalarGradient[Array]:
        gradient = self._c * self._bkd.dot(state.T, state) / 10.0
        return BackendScalarGradient(gradient, self._bkd)

    def _state_state(
        self, index: int, time: float, state: Array, weights: Array, w: Array
    ) -> Array:
        return 2.0 * self._weight() * w / 10.0 * weights

    def _state_param(
        self, index: int, time: float, state: Array, weights: Array, v: Array
    ) -> Array:
        return 2.0 * state / 10.0 * self._bkd.dot(self._c.T, v) * weights

    def _param_state(
        self, index: int, time: float, state: Array, weights: Array, w: Array
    ) -> Array:
        return self._c * 2.0 * self._bkd.dot(state.T, w) / 10.0 * weights

    def _param_param(
        self, index: int, time: float, state: Array, weights: Array, v: Array
    ) -> Optional[Array]:
        return None

    def derivatives(self) -> StateTermDerivatives[Array, Array]:
        return StateTermDerivatives.second_order(
            self._wrt_state,
            self._wrt_param,
            self._state_state,
            self._state_param,
            self._param_state,
            self._param_param,
        )


class _Tikhonov(Generic[Array]):
    r""":math:`r(p) = 0.15\, p^\top p`."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._param = bkd.asarray(LV_PARAM)

    def set_param(self, param: Array) -> None:
        self._param = param

    def nqoi(self) -> int:
        return 1

    def value(self) -> Array:
        return 0.15 * self._bkd.dot(self._param.T, self._param)

    def _wrt_param(self) -> BackendScalarGradient[Array]:
        return BackendScalarGradient(0.3 * self._param, self._bkd)

    def _param_param(self, weights: Array, v: Array) -> Array:
        return 0.3 * v * weights

    def derivatives(self) -> ParameterTermDerivatives[Array, Array]:
        return ParameterTermDerivatives.second_order(
            self._wrt_param, self._param_param
        )


class _Case(Generic[Array]):
    """The stepper, the terms, the recorded trajectory at ``LV_PARAM`` and
    a checker setting the parameters of the model and every term."""

    def __init__(self, bkd: Backend[Array], method: Method) -> None:
        self.bkd = bkd
        self.model = lotka_volterra_with_mass(bkd)
        self.stepper: Stepper[Array] = make_stepper(
            method, dense_mass_form_system(self.model)
        )
        self.integrand = EnergyIntegrand(bkd)
        self.observation = _Observation(bkd)
        self.tikhonov = _Tikhonov(bkd)
        self.integral: TimeIntegral[Array, Array] = TimeIntegral(
            self.integrand, self.stepper, bkd
        )
        self.param = bkd.asarray(LV_PARAM)
        self.set_param(self.param)
        self.store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(self.stepper).solve(bkd.asarray(START), STEPS, self.store)
        self.checker: TrajectoryTermChecker[Array, Array] = TrajectoryTermChecker(
            self.stepper, bkd, backend_sample(bkd), set_param=self.set_param
        )

    def set_param(self, param: Array) -> None:
        self.model.set_param(param)
        self.integrand.set_param(param)
        self.observation.set_param(param)
        self.tikhonov.set_param(param)

    def native(self) -> TrajectoryFunctional[Array, Array]:
        """The observation and the Tikhonov term."""
        return TrajectoryFunctional.from_terms(
            self.bkd,
            1,
            parameter_terms=(self.tikhonov,),
            state_terms=(self.observation,),
        )

    def window(self, step: int) -> StepWindow[Array]:
        return self.store.window(step, min(self.stepper.depth(), step + 1))

    def ones(self) -> Array:
        return self.bkd.ones((1, 1))


class TestEveryTermPasses:
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_the_integral_and_native_terms(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        case = _Case(bkd, make())
        functional = case.integral.trajectory_functional(case.store) + case.native()
        report = case.checker.check(functional, case.store, param=case.param)
        assert report.failures() == []
        nsteps = case.store.nsteps()
        for step in range(nsteps):
            for name in ("wrt_unknown", "wrt_param", "curvature_wrt_param"):
                assert f"step_terms[0]@{step}.{name}" in report.fd_checks
        for index in (2, 5, nsteps):
            assert f"state_terms[0]@{index}.param_state" in report.fd_checks
        assert "state_terms[0]@3.wrt_state" not in report.fd_checks
        assert "parameter_terms[0].param_param" in report.fd_checks

    def test_a_stack_under_weights(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, bdf2())
        stacked = TrajectoryFunctional.stack(
            bkd, [case.integral.trajectory_functional(case.store), case.native()]
        )
        report = case.checker.check(
            stacked, case.store, bkd.asarray([[0.7], [-1.3]]), case.param
        )
        assert report.failures() == []

    def test_a_value_only_term_is_checked_for_nothing(
        self, bkd: Backend[Array]
    ) -> None:
        case = _Case(bkd, heun())
        report = case.checker.check_state_term(
            _ValueOnly(case.observation),
            2,
            case.store.time(2),
            case.store.state(2),
            case.ones(),
            case.param,
        )
        assert dict(report.fd_checks) == {} and report.failures() == []


class _ValueOnly(Generic[Array]):
    """A state term's value without derivatives."""

    def __init__(self, term: StateTermProtocol[Array, Array]) -> None:
        self._term = term

    def nqoi(self) -> int:
        return self._term.nqoi()

    def reads(self, index: int, last: bool) -> bool:
        return self._term.reads(index, last)

    def value(self, index: int, time: float, state: Array) -> Array:
        return self._term.value(index, time, state)

    def derivatives(self) -> StateTermDerivatives[Array, Array]:
        return StateTermDerivatives.none()


class _Scaled(Generic[Array]):
    """A linearization with its action and its transpose scaled."""

    def __init__(
        self,
        inner: LinearizationProtocol[Array, Array],
        forward: float,
        transposed: float,
    ) -> None:
        self._inner, self._forward, self._transposed = inner, forward, transposed

    def apply(self, x: Array) -> Array:
        return self._forward * self._inner.apply(x)

    def apply_transpose(self, weights: Array) -> Array:
        return self._transposed * self._inner.apply_transpose(weights)


_StateEdit = Callable[
    [StateTermDerivatives[Array, Array]], StateTermDerivatives[Array, Array]
]


class _EditedState(Generic[Array]):
    """A state term whose bundle is edited."""

    def __init__(
        self, term: StateTermProtocol[Array, Array], edit: _StateEdit[Array]
    ) -> None:
        self._term, self._edit = term, edit

    def nqoi(self) -> int:
        return self._term.nqoi()

    def reads(self, index: int, last: bool) -> bool:
        return self._term.reads(index, last)

    def value(self, index: int, time: float, state: Array) -> Array:
        return self._term.value(index, time, state)

    def derivatives(self) -> StateTermDerivatives[Array, Array]:
        return self._edit(self._term.derivatives())


def _scale_wrt_state(forward: float, transposed: float) -> _StateEdit[Array]:
    def edit(
        bundle: StateTermDerivatives[Array, Array],
    ) -> StateTermDerivatives[Array, Array]:
        field = bundle.wrt_state
        assert field is not None

        def wrt_state(
            index: int, time: float, state: Array
        ) -> LinearizationProtocol[Array, Array]:
            return _Scaled(field(index, time, state), forward, transposed)

        return replace(bundle, wrt_state=wrt_state)

    return edit


def _drop_wrt_param(
    bundle: StateTermDerivatives[Array, Array],
) -> StateTermDerivatives[Array, Array]:
    return replace(bundle, wrt_param=None)


def _scale_state_param(
    bundle: StateTermDerivatives[Array, Array],
) -> StateTermDerivatives[Array, Array]:
    field = bundle.state_param
    assert field is not None

    def state_param(
        index: int, time: float, state: Array, weights: Array, v: Array
    ) -> Optional[Array]:
        result = field(index, time, state, weights, v)
        return None if result is None else 1.01 * result

    return replace(bundle, state_param=state_param)


_StepEdit = Callable[
    [StepTermDerivatives[Array, Array]], StepTermDerivatives[Array, Array]
]


class _EditedStep(Generic[Array]):
    """A step term whose bundle is edited."""

    def __init__(
        self, term: StepTermProtocol[Array, Array], edit: _StepEdit[Array]
    ) -> None:
        self._term, self._edit = term, edit

    def nqoi(self) -> int:
        return self._term.nqoi()

    def value(self, step: int, window: StepWindow[Array]) -> Array:
        return self._term.value(step, window)

    def derivatives(self) -> StepTermDerivatives[Array, Array]:
        return self._edit(self._term.derivatives())


def _scale_curvature_wrt_param(
    bundle: StepTermDerivatives[Array, Array],
) -> StepTermDerivatives[Array, Array]:
    field = bundle.curvature_wrt_param
    assert field is not None

    def curvature(
        step: int,
        window: StepWindow[Array],
        tangents: StepWindow[Array],
        direction: Array,
        weights: Array,
    ) -> Optional[Array]:
        result = field(step, window, tangents, direction, weights)
        return None if result is None else 1.01 * result

    return replace(bundle, curvature_wrt_param=curvature)


def _drop_wrt_lag(
    bundle: StepTermDerivatives[Array, Array],
) -> StepTermDerivatives[Array, Array]:
    def wrt_lag(
        step: int, window: StepWindow[Array], lag: int
    ) -> Optional[LinearizationProtocol[Array, Array]]:
        return None

    return replace(bundle, wrt_lag=wrt_lag)


class TestPlantedFaults:
    def _state_failures(
        self, bkd: Backend[Array], edit: _StateEdit[Array]
    ) -> list[str]:
        case = _Case(bkd, backward_euler())
        report = case.checker.check_state_term(
            _EditedState(case.observation, edit),
            5,
            case.store.time(5),
            case.store.state(5),
            case.ones(),
            case.param,
        )
        return sorted(report.failures())

    def test_a_scaled_state_derivative(self, bkd: Backend[Array]) -> None:
        """Its value check fails, and so do the two second derivatives
        differencing its transpose."""
        failures = self._state_failures(bkd, _scale_wrt_state(1.01, 1.01))
        assert failures == ["state_param", "state_state", "wrt_state"]

    def test_an_omitted_parameter_dependence(self, bkd: Backend[Array]) -> None:
        """Left out, the parameter derivative is compared with zero; so is
        the transpose ``param_state`` differences."""
        failures = self._state_failures(bkd, _drop_wrt_param)
        assert failures == ["param_state", "wrt_param"]

    def test_a_scaled_mixed_second_derivative(self, bkd: Backend[Array]) -> None:
        assert self._state_failures(bkd, _scale_state_param) == ["state_param"]

    def test_a_wrong_transpose_reaches_the_dot_test(
        self, bkd: Backend[Array]
    ) -> None:
        """The forward action is right, so its finite differences pass;
        the dot test does not."""
        case = _Case(bkd, backward_euler())
        report = case.checker.check_state_term(
            _EditedState(case.observation, _scale_wrt_state(1.0, 1.01)),
            5,
            case.store.time(5),
            case.store.state(5),
            case.ones(),
            case.param,
        )
        assert report.transpose_errors["wrt_state"] > 1e-3
        curve = report.fd_checks["wrt_state"]
        assert curve.min_error() < 1e-6 and curve.v_shape().passed

    def _step_failures(
        self, bkd: Backend[Array], method: Method, edit: _StepEdit[Array]
    ) -> list[str]:
        case = _Case(bkd, method)
        term = _EditedStep(case.integral.step_term(), edit)
        report = case.checker.check_step_term(
            term, 3, case.window(3), case.ones(), case.param
        )
        return sorted(report.failures())

    def test_a_scaled_step_curvature(self, bkd: Backend[Array]) -> None:
        failures = self._step_failures(bkd, heun(), _scale_curvature_wrt_param)
        assert failures == ["curvature_wrt_param"]

    def test_a_dropped_lag_derivative(self, bkd: Backend[Array]) -> None:
        """Heun's first stage is the stored state, read as lag 1: dropping
        its derivative fails the value check and the curvature through
        it."""
        failures = self._step_failures(bkd, heun(), _drop_wrt_lag)
        assert failures == ["curvature_wrt_lag[1]", "wrt_lag[1]"]


_MockSpace = InnerProductSpaceProtocol[MockVector, NDArray[Any]]


class _MockEnergy:
    r""":math:`q = y^\top y` through the space's inner product: an
    integrand on an opaque stack, without parameters."""

    def __init__(self, space: _MockSpace) -> None:
        self._space = space

    def nqoi(self) -> int:
        return 1

    def value(self, state: MockVector, time: float) -> NDArray[Any]:
        return self._space.dot(state, state)

    def _state(self, state: MockVector, time: float) -> "_MockGradient":
        return _MockGradient(self._space, state)

    def derivatives(self) -> IntegrandDerivatives[MockVector, NDArray[Any]]:
        return IntegrandDerivatives.param_free(self._state)


class _MockGradient:
    r""":math:`2 y^\top x` forward, :math:`2 \omega y` transposed (scaled
    by a plain number, as library code must)."""

    def __init__(self, space: _MockSpace, state: MockVector) -> None:
        self._space, self._state = space, state

    def apply(self, x: MockVector) -> NDArray[Any]:
        return 2.0 * self._space.dot(self._state, x)

    def apply_transpose(self, weights: NDArray[Any]) -> MockVector:
        return self._space.scale(2.0 * float(weights[0, 0]), self._state)


class TestOnAnOpaqueStack:
    @pytest.mark.parametrize("make", [forward_euler, backward_euler])
    def test_the_checker_uses_only_the_spaces(
        self, make: Callable[[], ButcherTableau]
    ) -> None:
        bkd = NumpyBkd()
        stepper = mock_stepper(make())
        space = stepper.system().space
        integral: TimeIntegral[MockVector, NDArray[Any]] = TimeIntegral(
            _MockEnergy(space), stepper, bkd
        )
        store: InMemoryTrajectoryStore[MockVector] = InMemoryTrajectoryStore()
        Integrator(stepper).solve(
            mock_sample(space.descriptor(), 3), GivenSizes(0.0, [0.1] * 4), store
        )
        checker: TrajectoryTermChecker[MockVector, NDArray[Any]] = (
            TrajectoryTermChecker(stepper, bkd, mock_sample)
        )
        report = checker.check(integral.trajectory_functional(store), store)
        assert report.failures() == []
        for step in range(4):
            for name in ("wrt_unknown", "wrt_lag[1]"):
                assert f"step_terms[0]@{step}.{name}" in report.fd_checks


class _BackwardOnlyStore(InMemoryTrajectoryStore[Array]):
    """Once armed, refuses a state or a window later than the last of its
    kind read: a store readable only backward, as one replaying
    checkpointed segments in reverse would be."""

    def __init__(self) -> None:
        super().__init__()
        self._armed = False
        self._last: dict[str, int] = {}

    def arm(self) -> None:
        self._armed = True

    def _read(self, kind: str, index: int, count: int) -> None:
        if not self._armed:
            return
        index %= count
        last = self._last.get(kind)
        if last is not None and index > last:
            raise ValueError(f"forward read of {kind} {index} after {last}")
        self._last[kind] = index

    def state(self, index: int) -> Array:
        self._read("state", index, self.ntimes())
        return super().state(index)

    def window(self, step: int, depth: int = 1) -> StepWindow[Array]:
        self._read("window", step, self.nsteps())
        return super().window(step, depth)


class TestTheAdjointSweepReadsBackward:
    @pytest.mark.parametrize("make", [backward_euler, bdf2], ids=["BE", "BDF2"])
    def test_a_backward_only_store_serves_the_gradient(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        """The integral and the native terms: the gradient reads each step's
        window, and each observed state, last first, and binding the
        integral to the store reads none; it equals the gradient over an
        unrestricted store."""
        case = _Case(bkd, make())
        store: _BackwardOnlyStore[Array] = _BackwardOnlyStore()
        Integrator(case.stepper).solve(bkd.asarray(START), STEPS, store)
        store.arm()
        functional = case.integral.trajectory_functional(store) + case.native()
        sweeps = TrajectoryDerivatives(case.stepper)
        expected = sweeps.gradient(
            case.store, case.integral.trajectory_functional(case.store) + case.native()
        )
        bkd.assert_allclose(sweeps.gradient(store, functional), expected, rtol=1e-12)
        # The store does refuse: the sweep ended at window 0, so window 1
        # would be a forward read.
        with pytest.raises(ValueError, match="forward read of window"):
            store.window(1)


class TestRefusals:
    def test_a_non_stepper(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="DifferentiableStepperProtocol"):
            TrajectoryTermChecker(object(), bkd, backend_sample(bkd))  # type: ignore[arg-type]

    def test_a_non_term(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, backward_euler())
        with pytest.raises(TypeError, match="StepTermProtocol"):
            case.checker.check_step_term(
                object(),  # type: ignore[arg-type]
                0,
                case.window(0),
                case.ones(),
            )

    def test_curvatures_need_the_parameters(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, backward_euler())
        with pytest.raises(ValueError, match="pass param"):
            case.checker.check_step_term(
                case.integral.step_term(), 0, case.window(0), case.ones()
            )

    def test_parameters_need_set_param(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, backward_euler())
        checker = TrajectoryTermChecker(case.stepper, bkd, backend_sample(bkd))
        with pytest.raises(ValueError, match="set_param"):
            checker.check_parameter_term(case.tikhonov, case.ones(), case.param)
