r"""The tangent and adjoint sweeps over a recorded trajectory.

Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` from (10, 5),
parameters (1, 0.1, 0.075, 1.5), dt = 0.05 to T = 0.5, Newton tolerance
1e-13. The functional is a Tikhonov-augmented (alpha 0.3) mean squared
error with noise 0.5 on prey at steps 2, 5, 10 and predator at 3, 10,
so it reads several stored states and depends on the parameters
directly.

- For every tableau the adjoint gradient passes ``DerivativeChecker``
  (central differences of the whole solve), and the tangent sweep in a
  random direction equals the gradient's inner product with it to 1e-12.
- For the five methods the old path also has, the gradient equals the
  old adjoint's to 1e-10.
- A curved, time-dependent constraint coupling state and parameter (the
  predator pinned to :math:`3 + 0.1\, a \sin(x + 2t)`) and an initial
  state with a
  parameter cross term (:math:`\hat y_0 = (10 a,\ 5 e^{a b - 0.1})`) are
  differentiated too, each alone and both together: finite differences
  and the tangent agree with the gradient. Neither is additive, so a
  derivative evaluated at the wrong state or parameters is caught.
- With every parameter derivative refusing more than one column, both
  sweeps still run: no state-by-parameter matrix is formed.
- On the mock stack (:math:`\dot y = K y + B p`, a functional of the last
  state and the parameters, satisfying the sweeps' protocol directly)
  the gradient matches central differences, which are exact for this
  affine problem up to rounding.
- Refusals: a trajectory recorded at other parameters, a stepper without
  parameters.
"""

from typing import Any, Callable, Generic, List, Optional, Union

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.explicit_steppers import ForwardEulerHVP, HeunHVP
from pyapprox.ode.implicit_steppers import (
    BackwardEulerHVP,
    CrankNicolsonHVP,
    ImplicitMidpointHVP,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.integration import Integrator, ResidualFormConstraint
from pyapprox.ode.linear_multistep import LinearMultistepStepper
from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.protocols import TimeSteppingResidualProtocol
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    ODEResidualDerivatives,
    SecondDerivativeFn,
)
from pyapprox.ode.residual_form import BackendColumnForm, IdentityForm
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.state_constraint import StateConstraintDerivatives
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    classical_rk4,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_derivatives import (
    InitialStateDerivativeProtocol,
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import (
    ParameterTermDerivatives,
    StateTermDerivatives,
    StateTermProtocol,
    TrajectoryFunctional,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import DenseFormat
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    LinearizationPoint,
    LinearizedOperator,
    OperatorProperties,
    SpaceDescriptor,
)
from pyapprox.util.rootfinding.newton import NewtonSolver
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.linalg.mock_stack import (
    MockFormat,
    MockLU,
    MockOperator,
    MockVector,
    MockVectorSpace,
    peek,
    wrap,
)
from tests._helpers.ode.runge_kutta_cases import (
    LINEAR_K,
    LV_PARAM,
    MOCK_STATE,
    LinearMassModel,
    lotka_volterra_with_mass,
    rk_stepper,
)
from tests._helpers.ode.trajectory_cases import (
    ALL_IDS,
    ALL_METHODS,
    MULTISTEP,
    SOME_IDS,
    SOME_METHODS,
    STEPS,
    CurvedPin,
    Method,
    ScaledStart,
    TrajectoryProblem,
    hvp_symmetry_error,
    min_gradient_fd_error,
    min_hvp_fd_error,
    random_direction,
    tikhonov_mse,
)
from tests._helpers.ode_models import LotkaVolterra


class TestGradient:
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_matches_finite_differences(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        assert min_gradient_fd_error(bkd, TrajectoryProblem(bkd, make())) < 1e-7

    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_tangent_is_the_gradient_in_a_direction(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem = TrajectoryProblem(bkd, make())
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        expected = problem.gradient(param) @ direction
        bkd.assert_allclose(problem.tangent(param, direction), expected, rtol=1e-12)


# The old path's stepper class for each tableau, for the same method.
_OLD_STEPPERS: list[tuple[Callable[[], Method], Callable[..., object]]] = [
    (backward_euler, BackwardEulerHVP),
    (forward_euler, ForwardEulerHVP),
    (crank_nicolson, CrankNicolsonHVP),
    (implicit_midpoint, ImplicitMidpointHVP),
    (heun, HeunHVP),
]


class TestAgainstTheOldPath:
    @pytest.mark.parametrize(
        "make, old", _OLD_STEPPERS, ids=[make().name for make, _ in _OLD_STEPPERS]
    )
    def test_same_gradient(
        self,
        bkd: Backend[Array],
        make: Callable[[], Method],
        old: Callable[..., object],
    ) -> None:
        param = bkd.asarray(LV_PARAM)
        model = lotka_volterra_with_mass(bkd)
        old_stepper = old(model)
        assert isinstance(old_stepper, TimeSteppingResidualProtocol)
        newton: NewtonSolver[Array] = NewtonSolver(old_stepper)
        newton.set_options(atol=1e-13, rtol=0.0, maxiters=30)
        operator = TimeAdjointOperatorWithHVP(
            TimeIntegrator(0.0, 0.5, 0.05, newton), tikhonov_mse(bkd)
        )
        old_gradient = operator.jacobian(bkd.asarray([10.0, 5.0]), param)
        bkd.assert_allclose(
            TrajectoryProblem(bkd, make()).gradient(param), old_gradient, rtol=1e-10
        )


_CONSTRAINED: List[Callable[[], Method]] = [
    backward_euler,
    heun,
    classical_rk4,
    *MULTISTEP,
]
_CONSTRAINED_IDS = [make().name for make in _CONSTRAINED]
# Which parameter-dependent pieces a case has: (constraint, initial state).
# Each is checked alone, so a wrong term in one cannot hide behind the
# other, and then both together.
_PIECES = {"constraint": (True, False), "start": (False, True), "both": (True, True)}


def _pieces_problem(
    bkd: Backend[Array], make: Callable[[], Method], pieces: str
) -> TrajectoryProblem[Array]:
    pin, start = _PIECES[pieces]
    return TrajectoryProblem(
        bkd,
        make(),
        CurvedPin(bkd) if pin else None,
        ScaledStart(bkd) if start else None,
    )


class TestConstraintAndInitialState:
    @pytest.mark.parametrize("pieces", list(_PIECES))
    @pytest.mark.parametrize("make", _CONSTRAINED, ids=_CONSTRAINED_IDS)
    def test_matches_finite_differences(
        self, bkd: Backend[Array], make: Callable[[], Method], pieces: str
    ) -> None:
        assert min_gradient_fd_error(bkd, _pieces_problem(bkd, make, pieces)) < 1e-7

    @pytest.mark.parametrize("pieces", list(_PIECES))
    @pytest.mark.parametrize("make", _CONSTRAINED, ids=_CONSTRAINED_IDS)
    def test_tangent_is_the_gradient_in_a_direction(
        self, bkd: Backend[Array], make: Callable[[], Method], pieces: str
    ) -> None:
        problem = _pieces_problem(bkd, make, pieces)
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        expected = problem.gradient(param) @ direction
        bkd.assert_allclose(problem.tangent(param, direction), expected, rtol=1e-12)

    def test_store_records_where_the_constraint_was_applied(
        self, bkd: Backend[Array]
    ) -> None:
        problem = TrajectoryProblem(bkd, heun(), CurvedPin(bkd), ScaledStart(bkd))
        param = bkd.asarray(LV_PARAM)
        store = problem.solve(param)
        bkd.assert_allclose(store.unconstrained(0), ScaledStart(bkd).start(param))
        readout = problem.stepper().readout()
        for n in range(store.nsteps()):
            stepped = readout.apply(store.window(n).unknown).blocks[0]
            bkd.assert_allclose(store.unconstrained(n + 1), stepped, rtol=0.0)

    def test_the_residual_initial_state_is_its_stated_zero(
        self, bkd: Backend[Array]
    ) -> None:
        initial: ResidualInitialState[Array] = ResidualInitialState(
            lotka_volterra_with_mass(bkd), BackendColumnForm(bkd)
        )
        assert isinstance(initial, InitialStateDerivativeProtocol)
        assert initial.param_derivative().properties().zero
        curvature = initial.curvature(bkd.ones((2, 1)))
        assert curvature is not None and curvature.properties().zero


class TestHessianVectorProduct:
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_matches_finite_differences_of_the_gradient(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        assert min_hvp_fd_error(bkd, TrajectoryProblem(bkd, make())) < 1e-7

    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_is_symmetric(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        assert hvp_symmetry_error(bkd, TrajectoryProblem(bkd, make())) < 1e-10

    def test_the_tangent_store_holds_the_tangent_trajectory(
        self, bkd: Backend[Array]
    ) -> None:
        """The tangents recorded for the HVP give the tangent sweep's
        directional derivative: sum_m <Q_y_m, w_m> + <Q_p, v>, from the
        misfit's state derivatives and the Tikhonov term's parameter
        derivative."""
        problem = TrajectoryProblem(bkd, heun(), CurvedPin(bkd), ScaledStart(bkd))
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        store = problem.solve(param)
        functional = problem.coupling(store, param)
        tangents: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            problem.stepper(),
            ResidualFormConstraint(problem.pin(), BackendColumnForm(bkd)),
            problem.start(),
        )
        sweeps.hvp(store, functional, direction, tangents)
        assert tangents.ntimes() == store.ntimes()
        assert tangents.nsteps() == store.nsteps()
        assert tangents.times() == store.times()
        (tikhonov,), (misfit,) = functional.parameter_terms, functional.state_terms
        wrt_param, wrt_state = (
            tikhonov.derivatives().wrt_param,
            misfit.derivatives().wrt_state,
        )
        assert wrt_param is not None and wrt_state is not None
        total = wrt_param().apply(direction)
        ntimes = store.ntimes()
        for index in range(ntimes):
            if misfit.reads(index, index == ntimes - 1):
                linearization = wrt_state(index, store.time(index), store.state(index))
                total = total + linearization.apply(tangents.state(index))
        expected = sweeps.tangent(store, functional, direction)
        bkd.assert_allclose(total, expected, rtol=1e-12)
        with pytest.raises(ValueError, match="tangent store must be empty"):
            sweeps.hvp(store, functional, direction, tangents)


class TestHessianVectorProductContracts:
    def test_gradient_and_hvp_gives_the_gradient(self, bkd: Backend[Array]) -> None:
        problem = TrajectoryProblem(bkd, heun(), CurvedPin(bkd), ScaledStart(bkd))
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        store = problem.solve(param)
        functional = problem.coupling(store, param)
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            problem.stepper(),
            ResidualFormConstraint(problem.pin(), BackendColumnForm(bkd)),
            problem.start(),
        )
        gradient, hvp = sweeps.gradient_and_hvp(store, functional, direction)
        bkd.assert_allclose(gradient, sweeps.gradient(store, functional), rtol=1e-14)
        bkd.assert_allclose(hvp, sweeps.hvp(store, functional, direction), rtol=1e-14)

    def test_tangent_steps_replay_from_a_checkpoint(self, bkd: Backend[Array]) -> None:
        """From the tangent recorded at step 4, tangent_step reproduces
        the rest of the recorded tangent trajectory: what a checkpointing
        store does to recompute a segment."""
        problem = TrajectoryProblem(bkd, heun(), CurvedPin(bkd), ScaledStart(bkd))
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        store = problem.solve(param)
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            problem.stepper(),
            ResidualFormConstraint(problem.pin(), BackendColumnForm(bkd)),
            problem.start(),
        )
        tangents: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        functional = problem.coupling(store, param)
        sweeps.hvp(store, functional, direction, tangents)
        initial = sweeps.tangent_initial(store, direction)
        bkd.assert_allclose(initial.state, tangents.state(0), rtol=0.0)
        recent = tangents.state(4)
        for n in range(4, store.nsteps()):
            window = store.window(n)
            replayed = sweeps.tangent_step(store, n, window, (recent,), direction)
            bkd.assert_allclose(replayed.state, tangents.state(n + 1), rtol=0.0)
            bkd.assert_allclose(
                replayed.unconstrained, tangents.unconstrained(n + 1), rtol=0.0
            )
            recent = replayed.state
        with pytest.raises(ValueError, match="tangents were given"):
            sweeps.tangent_step(store, 4, store.window(4), (), direction)


class TestHessianVectorProductAgainstTheOldPath:
    @pytest.mark.parametrize(
        "make, old", _OLD_STEPPERS, ids=[make().name for make, _ in _OLD_STEPPERS]
    )
    def test_same_hvp(
        self,
        bkd: Backend[Array],
        make: Callable[[], Method],
        old: Callable[..., object],
    ) -> None:
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        old_stepper = old(lotka_volterra_with_mass(bkd))
        assert isinstance(old_stepper, TimeSteppingResidualProtocol)
        newton: NewtonSolver[Array] = NewtonSolver(old_stepper)
        newton.set_options(atol=1e-13, rtol=0.0, maxiters=30)
        operator = TimeAdjointOperatorWithHVP(
            TimeIntegrator(0.0, 0.5, 0.05, newton), tikhonov_mse(bkd)
        )
        old_hvp = operator.hvp(bkd.asarray([10.0, 5.0]), param, direction)
        bkd.assert_allclose(
            TrajectoryProblem(bkd, make()).hvp(param, direction),
            bkd.reshape(old_hvp, (-1, 1)),
            rtol=1e-10,
        )


class TestHessianVectorProductConstraintAndInitialState:
    @pytest.mark.parametrize("pieces", list(_PIECES))
    @pytest.mark.parametrize("make", _CONSTRAINED, ids=_CONSTRAINED_IDS)
    def test_matches_finite_differences_of_the_gradient(
        self, bkd: Backend[Array], make: Callable[[], Method], pieces: str
    ) -> None:
        assert min_hvp_fd_error(bkd, _pieces_problem(bkd, make, pieces)) < 1e-7

    @pytest.mark.parametrize("pieces", list(_PIECES))
    @pytest.mark.parametrize("make", _CONSTRAINED, ids=_CONSTRAINED_IDS)
    def test_is_symmetric(
        self, bkd: Backend[Array], make: Callable[[], Method], pieces: str
    ) -> None:
        assert hvp_symmetry_error(bkd, _pieces_problem(bkd, make, pieces)) < 1e-10


class _FlatStart(ScaledStart[Array]):
    """:class:`ScaledStart` without its curvature."""

    def curvature(self, weight: Array) -> Optional[DerivativeOperator[Array]]:
        return None


class _Truncated(Generic[Array]):
    """A state term keeping its value and only the derivatives up to
    ``order`` (0: none, 1: the first)."""

    def __init__(self, inner: StateTermProtocol[Array, Array], order: int) -> None:
        self._inner = inner
        bundle = inner.derivatives()
        self._derivatives: StateTermDerivatives[Array, Array] = (
            StateTermDerivatives.none()
            if order == 0
            else StateTermDerivatives(bundle.wrt_state, bundle.wrt_param)
        )

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def reads(self, index: int, last: bool) -> bool:
        return self._inner.reads(index, last)

    def value(self, index: int, time: float, state: Array) -> Array:
        return self._inner.value(index, time, state)

    def derivatives(self) -> StateTermDerivatives[Array, Array]:
        return self._derivatives


class TestHessianVectorProductRefusals:
    def test_a_functional_without_second_derivatives(self, bkd: Backend[Array]) -> None:
        problem = TrajectoryProblem(bkd, heun())
        param = bkd.asarray(LV_PARAM)
        store = problem.solve(param)
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            problem.stepper()
        )
        misfit = problem.coupling(store, param).state_terms[0]
        functional = TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(_Truncated(misfit, 1),)
        )
        with pytest.raises(TypeError, match="_Truncated.*no second derivatives"):
            sweeps.hvp(store, functional, random_direction(bkd))

    def test_a_term_only_evaluated(self, bkd: Backend[Array]) -> None:
        """A term giving only its value is refused by every sweep, before
        any solve, naming it."""
        problem = TrajectoryProblem(bkd, heun())
        param = bkd.asarray(LV_PARAM)
        store = problem.solve(param)
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            problem.stepper()
        )
        misfit = problem.coupling(store, param).state_terms[0]
        functional = TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(_Truncated(misfit, 0),)
        )
        direction = random_direction(bkd)
        with pytest.raises(TypeError, match="_Truncated.*only evaluated"):
            sweeps.gradient(store, functional)
        with pytest.raises(TypeError, match="_Truncated.*only evaluated"):
            sweeps.tangent(store, functional, direction)
        with pytest.raises(TypeError, match="_Truncated.*only evaluated"):
            sweeps.hvp(store, functional, direction)

    def test_an_initial_state_without_curvature(self, bkd: Backend[Array]) -> None:
        problem = TrajectoryProblem(bkd, heun(), start=_FlatStart(bkd))
        with pytest.raises(TypeError, match="no curvature"):
            problem.hvp(bkd.asarray(LV_PARAM), random_direction(bkd))


class _OneColumn(Generic[Array]):
    """An operator applying only to single columns."""

    def __init__(self, op: DerivativeOperator[Array]) -> None:
        self._op = op

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return self._op.properties()

    def format(self) -> FormatProtocol:
        return self._op.format()

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        if x.shape[1] != 1:
            raise AssertionError(f"applied to {x.shape[1]} columns")
        return self._op.apply(x)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        if y.shape[1] != 1:
            raise AssertionError(f"transpose applied to {y.shape[1]} columns")
        return self._op.apply_transpose(y)


class _OneColumnParam(Generic[Array]):
    """A ``(state, time)`` parameter derivative whose operators apply to
    one column."""

    def __init__(
        self, field: Callable[[Array, float], DerivativeOperator[Array]]
    ) -> None:
        self._field = field

    def __call__(self, state: Array, time: float) -> _OneColumn[Array]:
        return _OneColumn(self._field(state, time))


class _OneColumnSecond(Generic[Array]):
    """A ``(state, time, adj)`` second derivative whose operators apply to
    one column."""

    def __init__(self, field: SecondDerivativeFn[Array]) -> None:
        self._field = field

    def __call__(self, state: Array, time: float, adj: Array) -> _OneColumn[Array]:
        return _OneColumn(self._field(state, time, adj))


class _OneColumnRhs(Generic[Array]):
    """Lotka-Volterra whose parameter derivatives, first and second order,
    refuse more than one column."""

    def __init__(self, model: LotkaVolterra[Array]) -> None:
        self._model = model
        inner = model.derivatives()
        if (
            inner.param is None
            or inner.state_param is None
            or inner.param_param is None
        ):
            raise AssertionError("Lotka-Volterra declares its second order")
        param: _OneColumnParam[Array] = _OneColumnParam(inner.param)
        state_param: _OneColumnSecond[Array] = _OneColumnSecond(inner.state_param)
        param_param: _OneColumnSecond[Array] = _OneColumnSecond(inner.param_param)
        self._derivatives: ODEResidualDerivatives[Array, DenseFormat[Array]] = (
            ODEResidualDerivatives(
                jacobian=inner.jacobian,
                param=param,
                initial_state=inner.initial_state,
                initial_state_curvature=inner.initial_state_curvature,
                state_state=inner.state_state,
                state_param=state_param,
                param_param=param_param,
                param_space=inner.param_space,
            )
        )

    def __call__(self, state: Array, time: float) -> Array:
        value: Array = self._model(state, time)
        return value

    def derivatives(self) -> ODEResidualDerivatives[Array, DenseFormat[Array]]:
        return self._derivatives

    def parameters_version(self) -> int:
        version: int = self._model.parameters_version()
        return version


def _one_column_system(
    model: LotkaVolterra[Array],
) -> MassFormSystem[Array, Array, DenseFormat[Array]]:
    dense = dense_mass_form_system(model)
    return MassFormSystem(
        rhs=_OneColumnRhs(model),
        mass=dense.mass,
        space=dense.space,
        solver=dense.solver,
        form=dense.form,
    )


class TestNoStateByParameterMatrix:
    @pytest.mark.parametrize("make", SOME_METHODS, ids=SOME_IDS)
    def test_parameter_terms_are_applied_one_column_at_a_time(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        refusing = TrajectoryProblem(bkd, make(), system=_one_column_system)
        plain = TrajectoryProblem(bkd, make())
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        bkd.assert_allclose(refusing.gradient(param), plain.gradient(param), rtol=1e-14)
        bkd.assert_allclose(
            refusing.tangent(param, direction),
            plain.tangent(param, direction),
            rtol=1e-14,
        )
        bkd.assert_allclose(
            refusing.hvp(param, direction), plain.hvp(param, direction), rtol=1e-14
        )


_MOCK_PARAM = SpaceDescriptor("param", 2, "float64")
_B = [[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]]
_C = [[1.0], [-2.0], [0.5]]
_D = [[0.3], [-0.7]]


class _MockAffineRhs:
    r""":math:`\dot y = K y + B p` on the mock stack (``LINEAR_K``,
    ``_B``), with its Jacobian, parameter derivative and second
    derivatives, all stated zeros (the model is affine)."""

    def __init__(self) -> None:
        self._k = MockOperator.from_dense(np.array(LINEAR_K), MOCK_STATE)
        self._b = MockOperator.from_dense(np.array(_B), _MOCK_PARAM, MOCK_STATE)
        self._states = MockVectorSpace(MOCK_STATE)
        self._p = wrap(np.zeros((2, 1)), _MOCK_PARAM)
        self._version = 0
        self._derivatives: ODEResidualDerivatives[MockVector, MockFormat] = (
            ODEResidualDerivatives.second_order(
                self._states,
                MockVectorSpace(_MOCK_PARAM),
                self._jacobian,
                self._param,
                initial_state=STATED_ZERO,
                initial_state_curvature=STATED_ZERO,
                state_state=STATED_ZERO,
                state_param=STATED_ZERO,
                param_param=STATED_ZERO,
            )
        )

    def set_param(self, param: NDArray[Any]) -> None:
        self._p = wrap(param, _MOCK_PARAM)
        self._version += 1

    def parameters_version(self) -> int:
        return self._version

    def __call__(self, state: MockVector, time: float) -> MockVector:
        return self._states.axpby(
            1.0, self._k.apply(state), 1.0, self._b.apply(self._p)
        )

    def _jacobian(
        self, state: MockVector, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[MockVector, MockFormat]:
        return LinearizedOperator(self._k, point)

    def _param(self, state: MockVector, time: float) -> MockOperator:
        return self._b

    def derivatives(self) -> ODEResidualDerivatives[MockVector, MockFormat]:
        return self._derivatives


_E = [[0.5], [-1.5]]


class _MockColumn:
    r"""The linearization of a scalar by a gradient column :math:`g` on
    the mock stack: forward :math:`g^\top x`, transposed :math:`g\,
    \omega`."""

    def __init__(self, gradient: NDArray[Any], descriptor: SpaceDescriptor) -> None:
        self._gradient, self._descriptor = gradient, descriptor

    def apply(self, x: MockVector) -> NDArray[Any]:
        result: NDArray[Any] = self._gradient.T @ peek(x)
        return result

    def apply_transpose(self, weights: NDArray[Any]) -> MockVector:
        return wrap(self._gradient @ weights, self._descriptor)


class _MockLastTerm:
    r"""The state term :math:`g(y_N;\, p) = a\, c^\top y_N + \tfrac{b}{2}
    |y_N|^2 + x\, (c^\top y_N)(e^\top p)` at the last stored state of a mock
    trajectory, at the parameters it was made with; a coefficient of zero
    drops its part, its second derivatives stated zero (None)."""

    def __init__(
        self, param: NDArray[Any], linear: float, quadratic: float, cross: float
    ) -> None:
        self._param = param
        self._a, self._b, self._x = linear, quadratic, cross
        self._c, self._e = np.array(_C), np.array(_E)

    def nqoi(self) -> int:
        return 1

    def reads(self, index: int, last: bool) -> bool:
        return last

    def value(self, index: int, time: float, state: MockVector) -> NDArray[Any]:
        y, c, e = peek(state), self._c, self._e
        result: NDArray[Any] = (
            self._a * c.T @ y
            + 0.5 * self._b * y.T @ y
            + self._x * (c.T @ y) * (e.T @ self._param)
        )
        return result

    def wrt_state(self, index: int, time: float, state: MockVector) -> _MockColumn:
        y, c, e = peek(state), self._c, self._e
        gradient = self._a * c + self._b * y + self._x * c * (e.T @ self._param)
        return _MockColumn(gradient, MOCK_STATE)

    def wrt_param(
        self, index: int, time: float, state: MockVector
    ) -> Optional[_MockColumn]:
        if self._x == 0.0:
            return None
        return _MockColumn(self._x * self._e * (self._c.T @ peek(state)), _MOCK_PARAM)

    def derivatives(self) -> StateTermDerivatives[MockVector, NDArray[Any]]:
        return StateTermDerivatives.second_order(
            self.wrt_state,
            self.wrt_param,
            self.state_state,
            self.state_param,
            self.param_state,
            self.param_param,
        )

    def state_state(
        self,
        index: int,
        time: float,
        state: MockVector,
        weights: NDArray[Any],
        w: MockVector,
    ) -> Optional[MockVector]:
        if self._b == 0.0:
            return None
        return wrap(self._b * peek(w) * weights[0, 0], MOCK_STATE)

    def state_param(
        self,
        index: int,
        time: float,
        state: MockVector,
        weights: NDArray[Any],
        v: MockVector,
    ) -> Optional[MockVector]:
        if self._x == 0.0:
            return None
        value = self._x * self._c * (self._e.T @ peek(v)) * weights[0, 0]
        return wrap(value, MOCK_STATE)

    def param_state(
        self,
        index: int,
        time: float,
        state: MockVector,
        weights: NDArray[Any],
        w: MockVector,
    ) -> Optional[MockVector]:
        if self._x == 0.0:
            return None
        value = self._x * self._e * (self._c.T @ peek(w)) * weights[0, 0]
        return wrap(value, _MOCK_PARAM)

    def param_param(
        self,
        index: int,
        time: float,
        state: MockVector,
        weights: NDArray[Any],
        v: MockVector,
    ) -> Optional[MockVector]:
        return None


class _MockParamTerm:
    r"""The parameter term :math:`r(p) = d^\top p + \tfrac{s}{2} |p|^2` at
    the parameters it was made with."""

    def __init__(self, param: NDArray[Any], quadratic: float) -> None:
        self._param, self._s, self._d = param, quadratic, np.array(_D)

    def nqoi(self) -> int:
        return 1

    def value(self) -> NDArray[Any]:
        p = self._param
        result: NDArray[Any] = self._d.T @ p + 0.5 * self._s * p.T @ p
        return result

    def wrt_param(self) -> _MockColumn:
        return _MockColumn(self._d + self._s * self._param, _MOCK_PARAM)

    def derivatives(self) -> ParameterTermDerivatives[MockVector, NDArray[Any]]:
        return ParameterTermDerivatives.second_order(self.wrt_param, self.param_param)

    def param_param(self, weights: NDArray[Any], v: MockVector) -> Optional[MockVector]:
        if self._s == 0.0:
            return None
        return wrap(self._s * peek(v) * weights[0, 0], _MOCK_PARAM)


def _mock_functional(
    state_term: StateTermProtocol[MockVector, NDArray[Any]],
    parameter_term: _MockParamTerm,
) -> TrajectoryFunctional[MockVector, NDArray[Any]]:
    """A scalar mock functional of one state term and one parameter term."""
    return TrajectoryFunctional(
        np.zeros((1, 1)),
        parameter_terms=(parameter_term,),
        state_terms=(state_term,),
    )


class _Buffered:
    """A linearization whose transposed results are written into a
    term's one scratch vector, as a term reusing its storage may."""

    def __init__(
        self, inner: _MockColumn, into: Callable[[MockVector], MockVector]
    ) -> None:
        self._inner, self._into = inner, into

    def apply(self, x: MockVector) -> NDArray[Any]:
        return self._inner.apply(x)

    def apply_transpose(self, weights: NDArray[Any]) -> MockVector:
        return self._into(self._inner.apply_transpose(weights))


class _OneBuffer:
    """A :class:`_MockLastTerm` writing every vector result into one
    scratch vector per space and returning it, as a term reusing its
    storage may: a sweep keeping a returned vector as an accumulator would
    see it overwritten by the next call."""

    def __init__(self, inner: _MockLastTerm) -> None:
        self._inner = inner
        self._state = wrap(np.zeros((3, 1)), MOCK_STATE)
        self._param = wrap(np.zeros((2, 1)), _MOCK_PARAM)
        # What each buffer held when last handed out: the sweeps must not
        # write into a returned vector.
        self._handed: dict[int, NDArray[Any]] = {}

    def _into(self, buffer: MockVector, value: MockVector) -> MockVector:
        handed = self._handed.get(id(buffer))
        if handed is not None and not np.array_equal(peek(buffer), handed):
            raise AssertionError("a returned vector was written into")
        # Overwrite the buffer in place through the stack (peek copies).
        space = MockVectorSpace(MOCK_STATE if buffer is self._state else _MOCK_PARAM)
        result = space.copy(value, out=buffer)
        self._handed[id(buffer)] = peek(result)
        return result

    def check_untouched(self) -> None:
        """Raise if a vector last returned was written into since."""
        for buffer in (self._state, self._param):
            handed = self._handed.get(id(buffer))
            if handed is not None and not np.array_equal(peek(buffer), handed):
                raise AssertionError("a returned vector was written into")

    def _into_state(self, value: MockVector) -> MockVector:
        return self._into(self._state, value)

    def _into_param(self, value: MockVector) -> MockVector:
        return self._into(self._param, value)

    def nqoi(self) -> int:
        return 1

    def reads(self, index: int, last: bool) -> bool:
        return self._inner.reads(index, last)

    def value(self, index: int, time: float, state: MockVector) -> NDArray[Any]:
        return self._inner.value(index, time, state)

    def wrt_state(self, index: int, time: float, state: MockVector) -> _Buffered:
        return _Buffered(self._inner.wrt_state(index, time, state), self._into_state)

    def wrt_param(
        self, index: int, time: float, state: MockVector
    ) -> Optional[_Buffered]:
        inner = self._inner.wrt_param(index, time, state)
        return None if inner is None else _Buffered(inner, self._into_param)

    def derivatives(self) -> StateTermDerivatives[MockVector, NDArray[Any]]:
        return StateTermDerivatives.second_order(
            self.wrt_state,
            self.wrt_param,
            self.state_state,
            self.state_param,
            self.param_state,
            self.param_param,
        )

    def state_state(
        self,
        index: int,
        time: float,
        state: MockVector,
        weights: NDArray[Any],
        w: MockVector,
    ) -> Optional[MockVector]:
        value = self._inner.state_state(index, time, state, weights, w)
        return None if value is None else self._into_state(value)

    def state_param(
        self,
        index: int,
        time: float,
        state: MockVector,
        weights: NDArray[Any],
        v: MockVector,
    ) -> Optional[MockVector]:
        value = self._inner.state_param(index, time, state, weights, v)
        return None if value is None else self._into_state(value)

    def param_state(
        self,
        index: int,
        time: float,
        state: MockVector,
        weights: NDArray[Any],
        w: MockVector,
    ) -> Optional[MockVector]:
        value = self._inner.param_state(index, time, state, weights, w)
        return None if value is None else self._into_param(value)

    def param_param(
        self,
        index: int,
        time: float,
        state: MockVector,
        weights: NDArray[Any],
        v: MockVector,
    ) -> Optional[MockVector]:
        return None


_MockStepper = Union[
    RungeKuttaStepper[MockVector, NDArray[Any], MockFormat],
    LinearMultistepStepper[MockVector, NDArray[Any], MockFormat],
]


class _MockPin:
    r""":math:`\Pi_t(y;\, p)`: the third component pinned to
    :math:`0.3 + 0.1\, p_1 \sin(y_1 + 2t)`, on the mock stack's vectors
    directly, with its first and second derivatives. Its tangent depends
    on the parameters and time, its parameter derivative on the state and
    time, and :math:`\Pi_{yy}`, :math:`\Pi_{yp}` are nonzero."""

    def __init__(self) -> None:
        self._p = np.zeros((2, 1))
        states, params = MockVectorSpace(MOCK_STATE), MockVectorSpace(_MOCK_PARAM)
        self._derivs: StateConstraintDerivatives[MockVector] = (
            StateConstraintDerivatives.second_order(
                states,
                params,
                tangent=self._tangent,
                param=self._param,
                state_state=self._state_state,
                state_param=self._state_param,
                param_param=STATED_ZERO,
            )
        )

    def set_param(self, param: NDArray[Any]) -> None:
        self._p = param.copy()

    def _phase(self, state: MockVector, time: float) -> float:
        return float(peek(state)[0, 0]) + 2.0 * time

    def constrain(self, state: MockVector, time: float) -> MockVector:
        entries = peek(state)
        entries[2, 0] = 0.3 + 0.1 * self._p[0, 0] * np.sin(self._phase(state, time))
        return wrap(entries, MOCK_STATE)

    def _tangent(self, state: MockVector, time: float) -> MockOperator:
        matrix = np.diag([1.0, 1.0, 0.0])
        matrix[2, 0] = 0.1 * self._p[0, 0] * np.cos(self._phase(state, time))
        return MockOperator.from_dense(matrix, MOCK_STATE)

    def _param(self, state: MockVector, time: float) -> MockOperator:
        matrix = np.zeros((3, 2))
        matrix[2, 0] = 0.1 * np.sin(self._phase(state, time))
        return MockOperator.from_dense(matrix, _MOCK_PARAM, MOCK_STATE)

    def _state_state(
        self, state: MockVector, time: float, adj: MockVector
    ) -> MockOperator:
        # adj_3 d^2(0.1 p_1 sin(y_1 + 2t))/dy_1^2
        matrix = np.zeros((3, 3))
        weight = float(peek(adj)[2, 0])
        matrix[0, 0] = -0.1 * self._p[0, 0] * np.sin(self._phase(state, time)) * weight
        return MockOperator.from_dense(matrix, MOCK_STATE)

    def _state_param(
        self, state: MockVector, time: float, adj: MockVector
    ) -> MockOperator:
        # adj_3 d^2(0.1 p_1 sin(y_1 + 2t))/dy_1 dp_1
        matrix = np.zeros((3, 2))
        weight = float(peek(adj)[2, 0])
        matrix[0, 0] = 0.1 * np.cos(self._phase(state, time)) * weight
        return MockOperator.from_dense(matrix, _MOCK_PARAM, MOCK_STATE)

    def derivatives(self) -> StateConstraintDerivatives[MockVector]:
        return self._derivs


def _mock_value(
    store: InMemoryTrajectoryStore[MockVector], param: NDArray[Any]
) -> float:
    r""":func:`_cross`'s :math:`Q` on a recorded trajectory."""
    last = peek(store.state(-1))
    c, d, e = np.array(_C), np.array(_D), np.array(_E)
    value = 0.5 * last.T @ last + d.T @ param + 0.5 * param.T @ param
    return float((value + (c.T @ last) * (e.T @ param))[0, 0])


def _mock_problem(
    make: Callable[[], Method],
    pin: Optional[_MockPin] = None,
) -> tuple[_MockStepper, Callable[[NDArray[Any]], InMemoryTrajectoryStore[MockVector]]]:
    """The mock-stack stepper of :class:`_MockAffineRhs`, and a solve from
    (1, -0.5, 0.8) over the module's times at given parameters, with
    ``pin`` applied at stored states when given."""
    rhs = _MockAffineRhs()
    newton: Newton[MockVector, NDArray[Any]] = Newton(atol=1e-13, rtol=0.0)
    system: MassFormSystem[MockVector, NDArray[Any], MockFormat] = MassFormSystem(
        rhs=rhs,
        mass=MockOperator.identity(MOCK_STATE),
        space=MockVectorSpace(MOCK_STATE),
        solver=MockLU(),
        form=IdentityForm(),
    )
    method = make()
    solver = None if method.is_explicit() else newton
    stepper: _MockStepper = (
        RungeKuttaStepper(method, system, solver)
        if isinstance(method, ButcherTableau)
        else LinearMultistepStepper(method, system, solver)
    )
    start = np.array([[1.0], [-0.5], [0.8]])

    def solve(param: NDArray[Any]) -> InMemoryTrajectoryStore[MockVector]:
        rhs.set_param(param)
        if pin is not None:
            pin.set_param(param)
        store: InMemoryTrajectoryStore[MockVector] = InMemoryTrajectoryStore()
        Integrator(stepper, pin).solve(wrap(start.copy(), MOCK_STATE), STEPS, store)
        return store

    return stepper, solve


_MockFunctional = Callable[
    [NDArray[Any]], TrajectoryFunctional[MockVector, NDArray[Any]]
]
_MockValue = Callable[[InMemoryTrajectoryStore[MockVector], NDArray[Any]], float]


def _last_state(param: NDArray[Any]) -> TrajectoryFunctional[MockVector, NDArray[Any]]:
    r""":math:`Q = c^\top y_N + d^\top p`: linear, its second derivatives
    zero."""
    return _mock_functional(
        _MockLastTerm(param, 1.0, 0.0, 0.0), _MockParamTerm(param, 0.0)
    )


def _last_state_value(
    store: InMemoryTrajectoryStore[MockVector], param: NDArray[Any]
) -> float:
    last = peek(store.state(-1))
    return float((np.array(_C).T @ last + np.array(_D).T @ param)[0, 0])


def _quadratic_last(
    param: NDArray[Any],
) -> TrajectoryFunctional[MockVector, NDArray[Any]]:
    r""":math:`Q = \tfrac12 |y_N|^2 + d^\top p`: only :math:`Q_{y_N y_N} =
    I` is nonzero, so the Hessian :math:`(dy_N/dp)^\top (dy_N/dp)` comes
    from the sweeps alone."""
    return _mock_functional(
        _MockLastTerm(param, 0.0, 1.0, 0.0), _MockParamTerm(param, 0.0)
    )


def _cross(param: NDArray[Any]) -> TrajectoryFunctional[MockVector, NDArray[Any]]:
    r""":math:`Q = \tfrac12 |y_N|^2 + d^\top p + \tfrac12 |p|^2 + (c^\top
    y_N)(e^\top p)`: every second derivative is nonzero, the parameter ones
    included, so every term the sweeps accumulate into appears."""
    return _mock_functional(
        _MockLastTerm(param, 0.0, 1.0, 1.0), _MockParamTerm(param, 1.0)
    )


def _quadratic_last_value(
    store: InMemoryTrajectoryStore[MockVector], param: NDArray[Any]
) -> float:
    last = peek(store.state(-1))
    return float((0.5 * last.T @ last + np.array(_D).T @ param)[0, 0])


class _MockDerivatives:
    """A mock-stack problem :math:`p \\mapsto Q` as NumPy functions of the
    parameters, so ``DerivativeChecker`` drives it on the NumPy backend:
    the parameters and the sweeps' results are read out of the mock
    vectors, the sweeps themselves run on the mock stack.

    Parameters
    ----------
    make : callable
        The tableau.
    functional : callable
        ``param ->`` the functional at those parameters.
    value : callable
        ``(store, param) ->`` :math:`Q` on a recorded trajectory.
    pin : _MockPin, optional
        A constraint at stored states.
    """

    def __init__(
        self,
        make: Callable[[], Method],
        functional: _MockFunctional,
        value: _MockValue,
        pin: Optional[_MockPin] = None,
    ) -> None:
        stepper, self._solve = _mock_problem(make, pin)
        self._sweeps: TrajectoryDerivatives[MockVector, NDArray[Any]] = (
            TrajectoryDerivatives(stepper, pin)
        )
        self._make, self._value = functional, value

    def _functional(
        self, store: InMemoryTrajectoryStore[MockVector], param: NDArray[Any]
    ) -> TrajectoryFunctional[MockVector, NDArray[Any]]:
        return self._make(param)

    def values(self, samples: NDArray[Any]) -> NDArray[Any]:
        """Q at each column. Shape: (1, nsamples)."""
        return np.array(
            [
                [
                    self._value(
                        self._solve(samples[:, k : k + 1]), samples[:, k : k + 1]
                    )
                    for k in range(samples.shape[1])
                ]
            ]
        )

    def gradient(self, sample: NDArray[Any]) -> NDArray[Any]:
        """dQ/dp by the adjoint sweep. Shape: (1, 2)."""
        store = self._solve(sample)
        column = peek(self._sweeps.gradient(store, self._functional(store, sample)))
        row: NDArray[Any] = column.T
        return row

    def gradients(self, samples: NDArray[Any]) -> NDArray[Any]:
        """dQ/dp at each column. Shape: (2, nsamples)."""
        return np.hstack(
            [self.gradient(samples[:, k : k + 1]).T for k in range(samples.shape[1])]
        )

    def tangent(self, sample: NDArray[Any], vec: NDArray[Any]) -> NDArray[Any]:
        """dQ/dp v by the tangent sweep. Shape: (1, 1)."""
        store = self._solve(sample)
        direction = wrap(vec, _MOCK_PARAM)
        result: NDArray[Any] = self._sweeps.tangent(
            store, self._functional(store, sample), direction
        )
        return result

    def hvp(self, sample: NDArray[Any], vec: NDArray[Any]) -> NDArray[Any]:
        """(d^2Q/dp^2) v by the tangent and adjoint sweeps. Shape: (2, 1)."""
        store = self._solve(sample)
        direction = wrap(vec, _MOCK_PARAM)
        product = self._sweeps.hvp(store, self._functional(store, sample), direction)
        column: NDArray[Any] = peek(product)
        return column

    def min_errors(self, param: NDArray[Any]) -> tuple[float, float]:
        """The smallest central-difference errors of the gradient (against
        Q) and of the HVP (against the gradient), on the NumPy backend."""
        bkd = NumpyBkd()
        first = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=2, fun=self.values, jacobian=self.gradient, bkd=bkd
        )
        second = FunctionWithJVPFromCallable(
            nqoi=2, nvars=2, fun=self.gradients, jvp=self.hvp, bkd=bkd
        )
        errors = [
            DerivativeChecker(wrapper).check_derivatives(
                param, relative=True, central=True
            )[0]
            for wrapper in (first, second)
        ]
        return float(np.min(errors[0])), float(np.min(errors[1]))


_MOCK_POINT = np.array([[0.4], [-0.2]])
_MOCK_DIRECTION = np.array([[0.6], [1.3]])


class TestMockStack:
    @pytest.mark.parametrize("make", SOME_METHODS, ids=SOME_IDS)
    def test_linear_functional(self, make: Callable[[], Method]) -> None:
        r""":math:`Q = c^\top y_N + d^\top p` on an affine model: the gradient
        passes the checker, and the tangent is the gradient in a
        direction."""
        problem = _MockDerivatives(make, _last_state, _last_state_value)
        assert isinstance(_MockLastTerm(_MOCK_POINT, 1.0, 0.0, 0.0), StateTermProtocol)
        gradient_error, _ = problem.min_errors(_MOCK_POINT)
        assert gradient_error < 1e-8
        gradient = problem.gradient(_MOCK_POINT)
        np.testing.assert_allclose(
            problem.tangent(_MOCK_POINT, _MOCK_DIRECTION),
            gradient @ _MOCK_DIRECTION,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("make", SOME_METHODS, ids=SOME_IDS)
    def test_quadratic_in_the_last_state(self, make: Callable[[], Method]) -> None:
        r""":math:`Q = \tfrac12 |y_N|^2 + d^\top p`: the Hessian
        :math:`(dy_N/dp)^\top (dy_N/dp)` comes from the sweeps alone."""
        problem = _MockDerivatives(make, _quadratic_last, _quadratic_last_value)
        gradient_error, hvp_error = problem.min_errors(_MOCK_POINT)
        assert gradient_error < 1e-8 and hvp_error < 1e-8

    @pytest.mark.parametrize("make", SOME_METHODS, ids=SOME_IDS)
    def test_parameter_terms(self, make: Callable[[], Method]) -> None:
        """:func:`_cross`: every second derivative of Q nonzero."""
        problem = _MockDerivatives(make, _cross, _mock_value)
        gradient_error, hvp_error = problem.min_errors(_MOCK_POINT)
        assert gradient_error < 1e-8 and hvp_error < 1e-8

    @pytest.mark.parametrize("constrained", [False, True], ids=["free", "pinned"])
    def test_a_functional_reusing_its_storage(self, constrained: bool) -> None:
        """On a stack that writes ``out`` in place, a sweep keeping a
        functional's returned vector as an accumulator would be clobbered
        by the functional's next call; the sweeps copy first. The
        constraint adds the per-state terms summed with the functional's."""
        pin = _MockPin() if constrained else None
        stepper, solve = _mock_problem(heun, pin)
        sweeps: TrajectoryDerivatives[MockVector, NDArray[Any]] = TrajectoryDerivatives(
            stepper, pin
        )
        param, direction = np.array([[0.4], [-0.2]]), np.array([[0.6], [1.3]])
        store = solve(param)
        plain = _MockLastTerm(param, 0.0, 1.0, 1.0)
        reusing = _OneBuffer(_MockLastTerm(param, 0.0, 1.0, 1.0))
        v = wrap(direction, _MOCK_PARAM)
        zero = np.zeros((1, 1))
        gradient, hvp = sweeps.gradient_and_hvp(
            store, TrajectoryFunctional(zero, state_terms=(reusing,)), v
        )
        reusing.check_untouched()
        expected_gradient, expected_hvp = sweeps.gradient_and_hvp(
            store, TrajectoryFunctional(zero, state_terms=(plain,)), v
        )
        np.testing.assert_array_equal(peek(gradient), peek(expected_gradient))
        np.testing.assert_array_equal(peek(hvp), peek(expected_hvp))
        # The same reusing term twice: each result is added before the
        # term's next call overwrites it.
        twice = TrajectoryFunctional(zero, state_terms=(reusing, reusing))
        gradient, hvp = sweeps.gradient_and_hvp(store, twice, v)
        np.testing.assert_allclose(
            peek(gradient), 2.0 * peek(expected_gradient), rtol=1e-14
        )
        np.testing.assert_allclose(peek(hvp), 2.0 * peek(expected_hvp), rtol=1e-14)

    @pytest.mark.parametrize("make", SOME_METHODS, ids=SOME_IDS)
    def test_constraint(self, make: Callable[[], Method]) -> None:
        """The constraint's first and second derivatives on a stack that
        writes ``out`` in place: the gradient and HVP pass the checker,
        and the tangent is the gradient in a direction."""
        problem = _MockDerivatives(make, _cross, _mock_value, _MockPin())
        gradient_error, hvp_error = problem.min_errors(_MOCK_POINT)
        assert gradient_error < 1e-7 and hvp_error < 1e-7
        np.testing.assert_allclose(
            problem.tangent(_MOCK_POINT, _MOCK_DIRECTION),
            problem.gradient(_MOCK_POINT) @ _MOCK_DIRECTION,
            rtol=1e-12,
        )


class TestRefusals:
    def test_a_trajectory_recorded_at_other_parameters(
        self, bkd: Backend[Array]
    ) -> None:
        problem = TrajectoryProblem(bkd, heun())
        param = bkd.asarray(LV_PARAM)
        store = problem.solve(param)
        coupling = problem.coupling(store, param)
        problem.solve(2.0 * param)
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            problem.stepper()
        )
        with pytest.raises(ValueError, match="parameter version"):
            sweeps.gradient(store, coupling)

    def test_a_stepper_without_parameters(self, bkd: Backend[Array]) -> None:
        stepper = rk_stepper(heun(), dense_mass_form_system(LinearMassModel(bkd)))
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(stepper).solve(bkd.ones((3, 1)), STEPS, store)
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(stepper)
        functional: TrajectoryFunctional[Array, Array] = TrajectoryFunctional(
            bkd.zeros((1, 1))
        )
        with pytest.raises(TypeError, match="no parameter"):
            sweeps.gradient(store, functional)
        with pytest.raises(TypeError, match="DifferentiableStepperProtocol"):
            TrajectoryDerivatives(object())  # type: ignore[arg-type]
