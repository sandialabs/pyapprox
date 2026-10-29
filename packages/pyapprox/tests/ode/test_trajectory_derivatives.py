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
  parameters, and functionals the adapter cannot serve.
"""

from typing import Any, Callable, Generic, Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.explicit_steppers import ForwardEulerHVP, HeunHVP
from pyapprox.ode.functionals.mse import TransientMSEFunctional
from pyapprox.ode.functionals.tikhonov import TikhonovAugmentedFunctional
from pyapprox.ode.implicit_steppers import (
    BackwardEulerHVP,
    CrankNicolsonHVP,
    ImplicitMidpointHVP,
)
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.integration import Integrator, ResidualFormConstraint
from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.operator.time_adjoint_hvp import TimeAdjointOperatorWithHVP
from pyapprox.ode.protocols import TimeSteppingResidualProtocol
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    ODEResidualDerivatives,
    backend_spaces,
    dense_operator,
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
    TrajectoryFunctionalProtocol,
)
from pyapprox.ode.trajectory_functional import BackendTrajectoryFunctional
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore, trajectory_array
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
    TABLEAU_IDS,
    TABLEAUX,
    LinearMassModel,
    lotka_volterra_with_mass,
    rk_stepper,
)
from tests._helpers.ode_models import LotkaVolterra

_TIMES = [0.05 * n for n in range(11)]
_START = [[10.0], [5.0]]
_ALL_TABLEAUX = [make for make, *_ in TABLEAUX]
_Stepper = RungeKuttaStepper[Array, Array, DenseFormat[Array]]


def _functional(bkd: Backend[Array]) -> TikhonovAugmentedFunctional[Array]:
    """The Tikhonov-augmented MSE of the module docstring."""
    mse = TransientMSEFunctional(
        2,
        4,
        [
            (0, bkd.asarray([2, 5, 10], dtype=int)),
            (1, bkd.asarray([3, 10], dtype=int)),
        ],
        bkd,
        noise_std=0.5,
    )
    mse.set_observations(bkd.asarray([9.0, 8.0, 7.5, 6.0, 5.5]))
    return TikhonovAugmentedFunctional(mse, 0.3, bkd)


class _CurvedPin(Generic[Array]):
    r""":math:`\Pi_t(x, z;\, p) = (x,\ 3 + 0.1\, a \sin(x + 2t))` on 1-D
    states, with its first derivatives; its parameters are set beside the
    model's. The tangent depends on :math:`a` and :math:`t`, the
    parameter derivative on the state and :math:`t`, so either evaluated
    at the wrong state, parameters or time is caught."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._a = bkd.asarray([1.0])
        states, params = backend_spaces(bkd, 2, 4)
        self._derivs: StateConstraintDerivatives[Array] = (
            StateConstraintDerivatives.first_order(
                states,
                params,
                tangent=dense_operator(self._tangent, states, states, bkd),
                param=dense_operator(self._param, params, states, bkd),
            )
        )

    def set_param(self, param: Array) -> None:
        self._a = self._bkd.flatten(param)[0:1]

    def constrain(self, state: Array, time: float) -> Array:
        pinned = 3.0 + 0.1 * self._a * self._bkd.sin(state[0:1] + 2.0 * time)
        return self._bkd.concatenate([state[0:1], pinned])

    def _tangent(self, state: Array, time: float) -> Array:
        bkd = self._bkd
        slope = 0.1 * self._a * bkd.cos(state[0:1] + 2.0 * time)
        return bkd.stack(
            [
                bkd.concatenate([bkd.ones((1,)), bkd.zeros((1,))]),
                bkd.concatenate([slope, bkd.zeros((1,))]),
            ]
        )

    def _param(self, state: Array, time: float) -> Array:
        # d(3 + 0.1 a sin(x + 2t))/da in the second row; nothing else
        # depends on p.
        matrix = self._bkd.zeros((2, 4))
        matrix[1, 0] = 0.1 * self._bkd.sin(state[0] + 2.0 * time)
        return matrix

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return self._derivs


class _ScaledStart(Generic[Array]):
    r""":math:`\hat y_0 = (10 a,\ 5 e^{a b - 0.1})` and its derivative,
    (10, 5) at the nominal parameters. The second entry is a
    non-polynomial function of two parameters, so its derivative depends
    on both and has two nonzero columns, and every entry of its curvature
    (aa, ab, bb) is nonzero."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._param = bkd.asarray(LV_PARAM)
        states, params = backend_spaces(bkd, 2, 4)
        self._derivative = dense_operator(self._matrix, params, states, bkd)

    def set_param(self, param: Array) -> None:
        self._param = param

    def start(self, param: Array) -> Array:
        a, b = param[0:1], param[1:2]
        return self._bkd.concatenate([10.0 * a, 5.0 * self._bkd.exp(a * b - 0.1)])

    def _matrix(self) -> Array:
        # d(10 a, 5 exp(a b - 0.1))/d(a, b, c, d) at the parameters set.
        a, b = self._param[0, 0], self._param[1, 0]
        growth = 5.0 * self._bkd.exp(a * b - 0.1)
        matrix = self._bkd.zeros((2, 4))
        matrix[0, 0] = 10.0
        matrix[1, 0] = b * growth
        matrix[1, 1] = a * growth
        return matrix

    def param_derivative(self) -> DerivativeOperator[Array]:
        return self._derivative()


class _Problem(Generic[Array]):
    r""":math:`p \mapsto Q(y(p), p)` on Lotka-Volterra, with its gradient
    and tangent from the new sweeps.

    Parameters
    ----------
    bkd : Backend
        The backend.
    tableau : ButcherTableau
        The method.
    pin : _CurvedPin, optional
        A constraint at stored states.
    start : _ScaledStart, optional
        A parameter-dependent initial state; (10, 5) otherwise.
    system : callable, optional
        Builds the stepper's system from the model; the dense mass-form
        system by default.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        tableau: ButcherTableau,
        pin: Optional[_CurvedPin[Array]] = None,
        start: Optional[_ScaledStart[Array]] = None,
        system: Optional[
            Callable[
                [LotkaVolterra[Array]],
                MassFormSystem[Array, Array, DenseFormat[Array]],
            ]
        ] = None,
    ) -> None:
        self._bkd, self._pin, self._start = bkd, pin, start
        self._model = lotka_volterra_with_mass(bkd)
        build = dense_mass_form_system if system is None else system
        self._stepper: _Stepper[Array] = rk_stepper(tableau, build(self._model))
        constraint = (
            None if pin is None else ResidualFormConstraint(pin, BackendColumnForm(bkd))
        )
        self._integrator = Integrator(self._stepper, constraint)
        self._sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            self._stepper, constraint, start
        )
        self._functional = _functional(bkd)

    def stepper(self) -> _Stepper[Array]:
        return self._stepper

    def solve(self, param: Array) -> InMemoryTrajectoryStore[Array]:
        self._model.set_param(param)
        if self._pin is not None:
            self._pin.set_param(param)
        if self._start is not None:
            self._start.set_param(param)
        start = (
            self._bkd.asarray(_START)
            if self._start is None
            else self._start.start(param)
        )
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        self._integrator.solve(start, _TIMES, store)
        return store

    def value(self, samples: Array) -> Array:
        """Q at each column of ``samples``. Shape: (1, nsamples)."""
        values = []
        for k in range(samples.shape[1]):
            param = samples[:, k : k + 1]
            store = self.solve(param)
            values.append(self._functional(trajectory_array(store, self._bkd), param))
        return self._bkd.hstack(values)

    def coupling(
        self, store: InMemoryTrajectoryStore[Array], param: Array
    ) -> BackendTrajectoryFunctional[Array]:
        return BackendTrajectoryFunctional(self._functional, store, param)

    def gradient(self, sample: Array) -> Array:
        """dQ/dp by the adjoint sweep. Shape: (1, 4)."""
        store = self.solve(sample)
        return self._sweeps.gradient(store, self.coupling(store, sample)).T

    def tangent(self, sample: Array, direction: Array) -> Array:
        """dQ/dp v by the tangent sweep. Shape: (1, 1)."""
        store = self.solve(sample)
        return self._sweeps.tangent(store, self.coupling(store, sample), direction)


def _direction(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.random.default_rng(7).normal(size=(4, 1)))


def _min_fd_error(bkd: Backend[Array], problem: _Problem[Array]) -> float:
    """The smallest central-difference error of the gradient."""
    wrapper = FunctionWithJacobianFromCallable(
        nqoi=1, nvars=4, fun=problem.value, jacobian=problem.gradient, bkd=bkd
    )
    errors = DerivativeChecker(wrapper).check_derivatives(
        bkd.asarray(LV_PARAM), relative=True, central=True
    )[0]
    return bkd.to_float(bkd.min(errors))


class TestGradient:
    @pytest.mark.parametrize("make", _ALL_TABLEAUX, ids=TABLEAU_IDS)
    def test_matches_finite_differences(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        assert _min_fd_error(bkd, _Problem(bkd, make())) < 1e-7

    @pytest.mark.parametrize("make", _ALL_TABLEAUX, ids=TABLEAU_IDS)
    def test_tangent_is_the_gradient_in_a_direction(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        problem = _Problem(bkd, make())
        param, direction = bkd.asarray(LV_PARAM), _direction(bkd)
        expected = problem.gradient(param) @ direction
        bkd.assert_allclose(problem.tangent(param, direction), expected, rtol=1e-12)


# The old path's stepper class for each tableau, for the same method.
_OLD_STEPPERS: list[tuple[Callable[[], ButcherTableau], Callable[..., object]]] = [
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
        make: Callable[[], ButcherTableau],
        old: Callable[..., object],
    ) -> None:
        param = bkd.asarray(LV_PARAM)
        model = lotka_volterra_with_mass(bkd)
        old_stepper = old(model)
        assert isinstance(old_stepper, TimeSteppingResidualProtocol)
        newton: NewtonSolver[Array] = NewtonSolver(old_stepper)
        newton.set_options(atol=1e-13, rtol=0.0, maxiters=30)
        operator = TimeAdjointOperatorWithHVP(
            TimeIntegrator(0.0, 0.5, 0.05, newton), _functional(bkd)
        )
        old_gradient = operator.jacobian(bkd.asarray([10.0, 5.0]), param)
        bkd.assert_allclose(
            _Problem(bkd, make()).gradient(param), old_gradient, rtol=1e-10
        )


_CONSTRAINED = [backward_euler, heun, classical_rk4]
_CONSTRAINED_IDS = [make().name for make in _CONSTRAINED]
# Which parameter-dependent pieces a case has: (constraint, initial state).
# Each is checked alone, so a wrong term in one cannot hide behind the
# other, and then both together.
_PIECES = {"constraint": (True, False), "start": (False, True), "both": (True, True)}


def _pieces_problem(
    bkd: Backend[Array], make: Callable[[], ButcherTableau], pieces: str
) -> _Problem[Array]:
    pin, start = _PIECES[pieces]
    return _Problem(
        bkd,
        make(),
        _CurvedPin(bkd) if pin else None,
        _ScaledStart(bkd) if start else None,
    )


class TestConstraintAndInitialState:
    @pytest.mark.parametrize("pieces", list(_PIECES))
    @pytest.mark.parametrize("make", _CONSTRAINED, ids=_CONSTRAINED_IDS)
    def test_matches_finite_differences(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau], pieces: str
    ) -> None:
        assert _min_fd_error(bkd, _pieces_problem(bkd, make, pieces)) < 1e-7

    @pytest.mark.parametrize("pieces", list(_PIECES))
    @pytest.mark.parametrize("make", _CONSTRAINED, ids=_CONSTRAINED_IDS)
    def test_tangent_is_the_gradient_in_a_direction(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau], pieces: str
    ) -> None:
        problem = _pieces_problem(bkd, make, pieces)
        param, direction = bkd.asarray(LV_PARAM), _direction(bkd)
        expected = problem.gradient(param) @ direction
        bkd.assert_allclose(problem.tangent(param, direction), expected, rtol=1e-12)

    def test_store_records_where_the_constraint_was_applied(
        self, bkd: Backend[Array]
    ) -> None:
        problem = _Problem(bkd, heun(), _CurvedPin(bkd), _ScaledStart(bkd))
        param = bkd.asarray(LV_PARAM)
        store = problem.solve(param)
        bkd.assert_allclose(store.unconstrained(0), _ScaledStart(bkd).start(param))
        readout = problem.stepper().readout()
        for n in range(store.nsteps()):
            stepped = readout.apply(store.window(n).unknown).blocks[0]
            bkd.assert_allclose(store.unconstrained(n + 1), stepped, rtol=0.0)

    def test_the_residual_initial_state_is_its_stated_zero(
        self, bkd: Backend[Array]
    ) -> None:
        initial = ResidualInitialState(lotka_volterra_with_mass(bkd))
        assert isinstance(initial, InitialStateDerivativeProtocol)
        assert initial.param_derivative().properties().zero


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
    """A model's ``param`` field, its operators applying to one column."""

    def __init__(self, model: LotkaVolterra[Array]) -> None:
        self._model = model

    def __call__(self, state: Array, time: float) -> _OneColumn[Array]:
        field = self._model.derivatives().param
        if field is None:
            raise AssertionError("Lotka-Volterra declares param")
        return _OneColumn(field(state, time))


class _OneColumnRhs(Generic[Array]):
    """Lotka-Volterra whose parameter derivatives refuse more than one
    column."""

    def __init__(self, model: LotkaVolterra[Array]) -> None:
        self._model = model
        inner = model.derivatives()
        param: _OneColumnParam[Array] = _OneColumnParam(model)
        self._derivatives: ODEResidualDerivatives[Array, FormatProtocol] = (
            ODEResidualDerivatives(
                jacobian=inner.jacobian,
                param=param,
                initial_state=inner.initial_state,
                param_space=inner.param_space,
            )
        )

    def __call__(self, state: Array, time: float) -> Array:
        value: Array = self._model(state, time)
        return value

    def derivatives(self) -> ODEResidualDerivatives[Array, FormatProtocol]:
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
        jacobian=dense.jacobian,
        space=dense.space,
        solver=dense.solver,
        form=dense.form,
    )


class TestNoStateByParameterMatrix:
    @pytest.mark.parametrize("make", [backward_euler, heun], ids=["BE", "heun"])
    def test_parameter_terms_are_applied_one_column_at_a_time(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        refusing = _Problem(bkd, make(), system=_one_column_system)
        plain = _Problem(bkd, make())
        param, direction = bkd.asarray(LV_PARAM), _direction(bkd)
        bkd.assert_allclose(refusing.gradient(param), plain.gradient(param), rtol=1e-14)
        bkd.assert_allclose(
            refusing.tangent(param, direction),
            plain.tangent(param, direction),
            rtol=1e-14,
        )


_MOCK_PARAM = SpaceDescriptor("param", 2, "float64")
_B = [[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]]
_C = [[1.0], [-2.0], [0.5]]
_D = [[0.3], [-0.7]]


class _MockAffineRhs:
    r""":math:`\dot y = K y + B p` on the mock stack (``LINEAR_K``,
    ``_B``), with its Jacobian and parameter derivative."""

    def __init__(self) -> None:
        self._k = MockOperator.from_dense(np.array(LINEAR_K), MOCK_STATE)
        self._b = MockOperator.from_dense(np.array(_B), _MOCK_PARAM, MOCK_STATE)
        self._states = MockVectorSpace(MOCK_STATE)
        self._p = wrap(np.zeros((2, 1)), _MOCK_PARAM)
        self._version = 0
        self._derivatives: ODEResidualDerivatives[MockVector, MockFormat] = (
            ODEResidualDerivatives.first_order(
                self._states,
                MockVectorSpace(_MOCK_PARAM),
                self._jacobian,
                self._param,
                STATED_ZERO,
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


class _MockLastState:
    r""":math:`Q = c^\top y_N + d^\top p`, read one stored state at a time:
    a functional satisfying the sweeps' protocol directly."""

    def __init__(self, nstates: int) -> None:
        self._last = nstates - 1

    def state_cotangent(self, index: int) -> MockVector:
        weight = np.array(_C) if index == self._last else np.zeros((3, 1))
        return wrap(weight, MOCK_STATE)

    def param_cotangent(self) -> MockVector:
        return wrap(np.array(_D), _MOCK_PARAM)


class TestMockStack:
    @pytest.mark.parametrize("make", [backward_euler, heun], ids=["BE", "heun"])
    def test_gradient_and_tangent(self, make: Callable[[], ButcherTableau]) -> None:
        rhs = _MockAffineRhs()
        newton: Newton[MockVector, NDArray[Any]] = Newton(atol=1e-13, rtol=0.0)
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None
        system = MassFormSystem(
            rhs=rhs,
            mass=MockOperator.identity(MOCK_STATE),
            jacobian=jacobian,
            space=MockVectorSpace(MOCK_STATE),
            solver=MockLU(),
            form=IdentityForm(),
        )
        tableau = make()
        stepper = RungeKuttaStepper(
            tableau, system, None if tableau.is_explicit() else newton
        )
        start = np.array([[1.0], [-0.5], [0.8]])

        def solve(param: NDArray[Any]) -> InMemoryTrajectoryStore[MockVector]:
            rhs.set_param(param)
            store: InMemoryTrajectoryStore[MockVector] = InMemoryTrajectoryStore()
            Integrator(stepper).solve(wrap(start.copy(), MOCK_STATE), _TIMES, store)
            return store

        def value(param: NDArray[Any]) -> float:
            last = peek(solve(param).state(-1))
            return float((np.array(_C).T @ last + np.array(_D).T @ param)[0, 0])

        param = np.array([[0.4], [-0.2]])
        store = solve(param)
        functional = _MockLastState(store.ntimes())
        assert isinstance(functional, TrajectoryFunctionalProtocol)
        sweeps: TrajectoryDerivatives[MockVector, NDArray[Any]] = TrajectoryDerivatives(
            stepper
        )
        gradient = peek(sweeps.gradient(store, functional))
        step = 1e-5
        # Central differences along each unit direction, as a column.
        expected = np.array(
            [
                [(value(param + step * e) - value(param - step * e)) / (2 * step)]
                for e in np.eye(2)[:, :, None]
            ]
        )
        np.testing.assert_allclose(gradient, expected, rtol=1e-8)
        direction = np.array([[0.6], [1.3]])
        store = solve(param)
        tangent = sweeps.tangent(store, functional, wrap(direction, _MOCK_PARAM))
        np.testing.assert_allclose(tangent, gradient.T @ direction, rtol=1e-12)


class TestRefusals:
    def test_a_trajectory_recorded_at_other_parameters(
        self, bkd: Backend[Array]
    ) -> None:
        problem = _Problem(bkd, heun())
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
        Integrator(stepper).solve(bkd.ones((3, 1)), _TIMES, store)
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(stepper)
        functional = _ZeroFunctional(bkd)
        with pytest.raises(TypeError, match="no parameter"):
            sweeps.gradient(store, functional)
        with pytest.raises(TypeError, match="DifferentiableStepperProtocol"):
            TrajectoryDerivatives(object())  # type: ignore[arg-type]

    def test_functionals_the_adapter_cannot_serve(self, bkd: Backend[Array]) -> None:
        store = _Problem(bkd, heun()).solve(bkd.asarray(LV_PARAM))
        with_sigma = TransientMSEFunctional(
            2, 4, [(0, bkd.asarray([2], dtype=int))], bkd
        )
        with pytest.raises(ValueError, match="parameters of its own"):
            BackendTrajectoryFunctional(with_sigma, store, bkd.asarray(LV_PARAM))
        with pytest.raises(ValueError, match="shape"):
            BackendTrajectoryFunctional(
                _functional(bkd), store, bkd.flatten(bkd.asarray(LV_PARAM))
            )
        with pytest.raises(TypeError, match="TransientFunctionalProtocol"):
            BackendTrajectoryFunctional(object(), store, bkd.asarray(LV_PARAM))  # type: ignore[arg-type]


class _ZeroFunctional(Generic[Array]):
    """A zero functional on the backend stack, for the refusal tests."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def state_cotangent(self, index: int) -> Array:
        return self._bkd.zeros((3, 1))

    def param_cotangent(self) -> Array:
        return self._bkd.zeros((1, 1))
