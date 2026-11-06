"""Shared problems for testing the trajectory sweeps: Lotka-Volterra
with a mass under a Tikhonov-augmented MSE, a curved time-dependent
constraint, a parameter cross-term initial state, and the
finite-difference checks of the gradient and HVP."""

import math
from typing import Callable, Generic, List, Optional, Union

import numpy as np

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
from pyapprox.ode.functionals.tikhonov import TikhonovAugmentedFunctional
from pyapprox.ode.integration import (
    DifferentiableStackConstraintProtocol,
    Integrator,
    ResidualFormConstraint,
)
from pyapprox.ode.linear_multistep import LinearMultistepStepper
from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.multistep_methods import MultistepFormulaProtocol
from pyapprox.ode.observation_terms import MisfitTerm, TikhonovTerm
from pyapprox.ode.pinned_rows import RowPinningProtocol
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    backend_spaces,
    dense_operator,
)
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.state_constraint import StateConstraintDerivatives
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    heun,
)
from pyapprox.ode.time_integral import IntegrandDerivatives
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectorySinkProtocol,
)
from pyapprox.ode.trajectory_terms import BackendScalarGradient, TrajectoryFunctional
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet, DenseFormat
from tests._helpers.ode.multistep_cases import METHOD_IDS, METHODS, lm_stepper
from tests._helpers.ode.runge_kutta_cases import (
    LV_PARAM,
    TABLEAU_IDS,
    TABLEAUX,
    lotka_volterra_with_mass,
    rk_stepper,
)
from tests._helpers.ode_models import LotkaVolterra

TIMES = [0.05 * n for n in range(11)]
STEPS = GivenSizes.from_times(TIMES)
"""The steps between ``TIMES``, exactly: their sizes are the differences."""
START = [[10.0], [5.0]]
Method = Union[ButcherTableau, MultistepFormulaProtocol]
MULTISTEP = [make for make, *_ in METHODS]
# Every one-step and multistep method: the sweeps are generic in both.
ALL_METHODS: List[Callable[[], Method]] = [
    *(make for make, *_ in TABLEAUX),
    *MULTISTEP,
]
ALL_IDS = TABLEAU_IDS + METHOD_IDS
# A one-step pair (explicit and implicit) and every multistep method, for
# the costlier checks.
SOME_METHODS: List[Callable[[], Method]] = [backward_euler, heun, *MULTISTEP]
SOME_IDS = [make().name for make in SOME_METHODS]
Stepper = Union[
    RungeKuttaStepper[Array, Array, DenseFormat[Array]],
    LinearMultistepStepper[Array, Array, DenseFormat[Array]],
]


def make_stepper(
    method: Method,
    system: MassFormSystem[Array, Array, DenseFormat[Array]],
    row_pinning: Optional[RowPinningProtocol[Array]] = None,
) -> Stepper[Array]:
    """The stepper of a tableau or of a multistep method, with the given
    row pinning."""
    stepper: Stepper[Array] = (
        rk_stepper(method, system, row_pinning)
        if isinstance(method, ButcherTableau)
        else lm_stepper(method, system, row_pinning)
    )
    return stepper


class PinnedPrey(Generic[Array]):
    r"""Lotka-Volterra's prey pinned to :math:`g(t) = 10 + \sin 3t`, equal
    to the start at t = 0 and moving: a pinned-rows declaration."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        states, _ = backend_spaces(bkd, 2, 4)
        self._rows: BackendRowSet[Array] = BackendRowSet(states.descriptor(), [0], bkd)

    def rows(self) -> BackendRowSet[Array]:
        return self._rows

    def data(self, time: float) -> float:
        return 10.0 + float(np.sin(3.0 * time))

    def values(self, time: float) -> Array:
        return self._bkd.asarray([[self.data(time)], [0.0]])

    def rates(self) -> Optional[Callable[[float], Array]]:
        return self._rate

    def _rate(self, time: float) -> Array:
        return self._bkd.asarray([[3.0 * float(np.cos(3.0 * time))], [0.0]])


_S = [[0.3], [-0.2], [0.5], [0.1]]


class EnergyIntegrand(Generic[Array]):
    r""":math:`q = \cos(t)\, y^\top y / 100 + (s^\top p)(\mathbf 1^\top
    y)`, shape (1, 1), with :math:`s = (0.3, -0.2, 0.5, 0.1)`, at the
    parameters last set: it reads the state nonlinearly, the time and the
    parameters. With its second derivatives unless ``second_order`` is
    False."""

    def __init__(self, bkd: Backend[Array], second_order: bool = True) -> None:
        self._bkd = bkd
        self._s = bkd.asarray(_S)
        self._param = bkd.asarray(LV_PARAM)
        self._second_order = second_order

    def set_param(self, param: Array) -> None:
        self._param = param

    def nqoi(self) -> int:
        return 1

    def value(self, state: Array, time: float) -> Array:
        bkd = self._bkd
        return math.cos(time) * bkd.dot(state.T, state) / 100.0 + bkd.dot(
            self._s.T, self._param
        ) * bkd.sum(state, axis=0, keepdims=True)

    def _state_gradient(
        self, state: Array, time: float
    ) -> BackendScalarGradient[Array]:
        weight = self._bkd.dot(self._s.T, self._param)
        gradient = 2.0 * math.cos(time) * state / 100.0 + weight * self._bkd.ones(
            state.shape
        )
        return BackendScalarGradient(gradient, self._bkd)

    def _param_gradient(
        self, state: Array, time: float
    ) -> BackendScalarGradient[Array]:
        gradient = self._s * self._bkd.sum(state, axis=0, keepdims=True)
        return BackendScalarGradient(gradient, self._bkd)

    def _state_state(
        self, state: Array, time: float, weights: Array, w: Array
    ) -> Array:
        return 2.0 * math.cos(time) * w / 100.0 * weights

    def _state_param(
        self, state: Array, time: float, weights: Array, v: Array
    ) -> Array:
        return self._bkd.dot(self._s.T, v) * self._bkd.ones(state.shape) * weights

    def _param_state(
        self, state: Array, time: float, weights: Array, w: Array
    ) -> Array:
        return self._s * self._bkd.sum(w, axis=0, keepdims=True) * weights

    def _param_param(
        self, state: Array, time: float, weights: Array, v: Array
    ) -> Array:
        return 0.0 * v

    def derivatives(self) -> IntegrandDerivatives[Array, Array]:
        if not self._second_order:
            return IntegrandDerivatives.first_order(
                self._state_gradient, self._param_gradient
            )
        return IntegrandDerivatives.second_order(
            self._state_gradient,
            self._param_gradient,
            self._state_state,
            self._state_param,
            self._param_state,
            self._param_param,
        )


def tikhonov_mse(
    bkd: Backend[Array],
    prey: tuple[int, ...] = (2, 5, 10),
    predator: tuple[int, ...] = (3, 10),
) -> TikhonovAugmentedFunctional[Array]:
    """The Tikhonov-augmented MSE of the module docstring, observing prey
    and predator at the given stored states (five observations)."""
    mse = TransientMSEFunctional(
        2,
        4,
        [
            (0, bkd.asarray(list(prey), dtype=int)),
            (1, bkd.asarray(list(predator), dtype=int)),
        ],
        bkd,
        noise_std=0.5,
    )
    mse.set_observations(bkd.asarray([9.0, 8.0, 7.5, 6.0, 5.5]))
    return TikhonovAugmentedFunctional(mse, 0.3, bkd)


def native_tikhonov_mse(
    bkd: Backend[Array],
    prey: tuple[int, ...] = (2, 5, 10),
    predator: tuple[int, ...] = (3, 10),
) -> tuple[TrajectoryFunctional[Array, Array], TikhonovTerm[Array]]:
    """The functional of :func:`tikhonov_mse` as native terms (a misfit
    state term and a Tikhonov parameter term), and its Tikhonov term,
    whose parameters the caller sets."""
    misfit: MisfitTerm[Array, Array] = MisfitTerm.from_state_times(
        bkd,
        2,
        [(0, prey), (1, predator)],
        bkd.asarray([9.0, 8.0, 7.5, 6.0, 5.5]),
        noise_std=0.5,
    )
    tikhonov = TikhonovTerm(bkd, 0.3)
    functional = TrajectoryFunctional.from_terms(
        bkd, 1, parameter_terms=(tikhonov,), state_terms=(misfit,)
    )
    return functional, tikhonov


class CurvedPin(Generic[Array]):
    r""":math:`\Pi_t(x, z;\, p) = (x,\ 3 + 0.1\, a \sin(x + 2t))` on 1-D
    states, with its first and second derivatives; its parameters are set
    beside the model's. The tangent depends on :math:`a` and :math:`t`,
    the parameter derivative on the state and :math:`t`, so either
    evaluated at the wrong state, parameters or time is caught; the
    second derivatives :math:`\Pi_{xx}` and :math:`\Pi_{xa}` are nonzero
    (:math:`\Pi_{aa}` is a stated zero)."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._a = bkd.asarray([1.0])
        states, params = backend_spaces(bkd, 2, 4)
        self._derivs: StateConstraintDerivatives[Array] = (
            StateConstraintDerivatives.second_order(
                states,
                params,
                tangent=dense_operator(self._tangent, states, states, bkd),
                param=dense_operator(self._param, params, states, bkd),
                state_state=dense_operator(
                    self._state_state, states, states, bkd, symmetric=True
                ),
                state_param=dense_operator(self._state_param, params, states, bkd),
                param_param=STATED_ZERO,
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

    def _state_state(self, state: Array, time: float, adj: Array) -> Array:
        # adj_1 d^2(0.1 a sin(x + 2t))/dx^2 in the (x, x) entry.
        matrix = self._bkd.zeros((2, 2))
        matrix[0, 0] = -0.1 * self._a[0] * self._bkd.sin(state[0] + 2.0 * time) * adj[1]
        return matrix

    def _state_param(self, state: Array, time: float, adj: Array) -> Array:
        # adj_1 d^2(0.1 a sin(x + 2t))/dx da in the (x, a) entry.
        matrix = self._bkd.zeros((2, 4))
        matrix[0, 0] = 0.1 * self._bkd.cos(state[0] + 2.0 * time) * adj[1]
        return matrix

    def derivatives(self) -> StateConstraintDerivatives[Array]:
        return self._derivs


class ScaledStart(Generic[Array]):
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
        self._curvature = dense_operator(
            self._curvature_matrix, params, params, bkd, symmetric=True
        )

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

    def _curvature_matrix(self, weight: Array) -> Array:
        # weight_1 times the Hessian of G = 5 exp(a b - 0.1) in (a, b):
        # G_aa = b^2 G, G_ab = (1 + a b) G, G_bb = a^2 G.
        a, b = self._param[0, 0], self._param[1, 0]
        scaled = 5.0 * self._bkd.exp(a * b - 0.1) * weight[1, 0]
        matrix = self._bkd.zeros((4, 4))
        matrix[0, 0] = b * b * scaled
        matrix[0, 1] = (1.0 + a * b) * scaled
        matrix[1, 0] = (1.0 + a * b) * scaled
        matrix[1, 1] = a * a * scaled
        return matrix

    def param_derivative(self) -> DerivativeOperator[Array]:
        return self._derivative()

    def curvature(self, weight: Array) -> Optional[DerivativeOperator[Array]]:
        return self._curvature(weight)


class TrajectoryProblem(Generic[Array]):
    r""":math:`p \mapsto Q(y(p), p)` on Lotka-Volterra, with its gradient
    and tangent from the new sweeps; :math:`Q` is
    :func:`native_tikhonov_mse`'s functional, valued by the sink during
    the solve.

    Parameters
    ----------
    bkd : Backend
        The backend.
    method : ButcherTableau or MultistepFormulaProtocol
        The method.
    pin : CurvedPin, optional
        A constraint at stored states.
    start : ScaledStart, optional
        A parameter-dependent initial state; (10, 5) otherwise.
    system : callable, optional
        Builds the stepper's system from the model; the dense mass-form
        system by default.
    times : list of float, optional
        The step boundaries; 0 to 0.5 in steps of 0.05 by default.
    prey, predator : tuple of int
        The stored states at which the misfit observes each.
    row_pinning : RowPinningProtocol, optional
        Rows pinned inside the stepper's solves.
    stack_constraint : DifferentiableStackConstraintProtocol, optional
        A constraint on stack vectors at stored states, instead of ``pin``.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        method: Method,
        pin: Optional[CurvedPin[Array]] = None,
        start: Optional[ScaledStart[Array]] = None,
        system: Optional[
            Callable[
                [LotkaVolterra[Array]],
                MassFormSystem[Array, Array, DenseFormat[Array]],
            ]
        ] = None,
        times: Optional[List[float]] = None,
        prey: tuple[int, ...] = (2, 5, 10),
        predator: tuple[int, ...] = (3, 10),
        row_pinning: Optional[RowPinningProtocol[Array]] = None,
        stack_constraint: Optional[DifferentiableStackConstraintProtocol[Array]] = None,
    ) -> None:
        self._bkd, self._pin, self._start = bkd, pin, start
        self._times = TIMES if times is None else times
        self._model = lotka_volterra_with_mass(bkd)
        build = dense_mass_form_system if system is None else system
        self._stepper: Stepper[Array] = make_stepper(
            method, build(self._model), row_pinning
        )
        constraint: Optional[DifferentiableStackConstraintProtocol[Array]] = (
            stack_constraint
            if pin is None
            else ResidualFormConstraint(pin, BackendColumnForm(bkd))
        )
        self._integrator = Integrator(self._stepper, constraint)
        self._sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            self._stepper, constraint, start
        )
        self._functional, self._tikhonov = native_tikhonov_mse(bkd, prey, predator)

    def stepper(self) -> Stepper[Array]:
        return self._stepper

    def pin(self) -> CurvedPin[Array]:
        if self._pin is None:
            raise AssertionError("this problem has no constraint")
        return self._pin

    def start(self) -> ScaledStart[Array]:
        if self._start is None:
            raise AssertionError("this problem has a fixed initial state")
        return self._start

    def solve(
        self, param: Array, sink: Optional[FunctionalSink[Array, Array]] = None
    ) -> InMemoryTrajectoryStore[Array]:
        """The trajectory at ``param``, also fed to ``sink`` when given."""
        self._model.set_param(param)
        self._tikhonov.set_param(param)
        if self._pin is not None:
            self._pin.set_param(param)
        if self._start is not None:
            self._start.set_param(param)
        start = (
            self._bkd.asarray(START)
            if self._start is None
            else self._start.start(param)
        )
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        plan = GivenSizes.from_times(self._times)
        if sink is None:
            self._integrator.solve(start, plan, store)
            return store
        sinks: List[TrajectorySinkProtocol[Array]] = [sink, store]
        self._integrator.solve(start, plan, MultiSink(sinks))
        return store

    def value(self, samples: Array) -> Array:
        """Q at each column of ``samples``, by the value sink during the
        solve. Shape: (1, nsamples)."""
        values = []
        functional = self._functional
        for k in range(samples.shape[1]):
            sink: FunctionalSink[Array, Array] = FunctionalSink(
                functional.zero, functional.parameter_terms, functional.state_terms
            )
            self.solve(samples[:, k : k + 1], sink)
            values.append(sink.value())
        return self._bkd.hstack(values)

    def coupling(
        self, store: InMemoryTrajectoryStore[Array], param: Array
    ) -> TrajectoryFunctional[Array, Array]:
        """The functional the sweeps read: its terms hold no trajectory,
        so it is the same for every store, read at the parameters the last
        solve set."""
        return self._functional

    def gradient(self, sample: Array) -> Array:
        """dQ/dp by the adjoint sweep. Shape: (1, 4)."""
        store = self.solve(sample)
        return self._sweeps.gradient(store, self.coupling(store, sample)).T

    def tangent(self, sample: Array, direction: Array) -> Array:
        """dQ/dp v by the tangent sweep. Shape: (1, 1)."""
        store = self.solve(sample)
        return self._sweeps.tangent(store, self.coupling(store, sample), direction)

    def gradients(self, samples: Array) -> Array:
        """dQ/dp at each column of ``samples``. Shape: (4, nsamples)."""
        return self._bkd.hstack(
            [self.gradient(samples[:, k : k + 1]).T for k in range(samples.shape[1])]
        )

    def hvp(self, sample: Array, direction: Array) -> Array:
        """(d^2Q/dp^2) v by the tangent and adjoint sweeps. Shape: (4, 1)."""
        store = self.solve(sample)
        return self._sweeps.hvp(store, self.coupling(store, sample), direction)


def random_direction(bkd: Backend[Array], seed: int = 7) -> Array:
    return bkd.asarray(np.random.default_rng(seed).normal(size=(4, 1)))


def min_hvp_fd_error(bkd: Backend[Array], problem: TrajectoryProblem[Array]) -> float:
    """The smallest central-difference error of the HVP, differencing the
    adjoint gradient along a random direction."""
    wrapper = FunctionWithJVPFromCallable(
        nqoi=4, nvars=4, fun=problem.gradients, jvp=problem.hvp, bkd=bkd
    )
    errors = DerivativeChecker(wrapper).check_derivatives(
        bkd.asarray(LV_PARAM), relative=True, central=True
    )[0]
    return bkd.to_float(bkd.min(errors))


def hvp_symmetry_error(bkd: Backend[Array], problem: TrajectoryProblem[Array]) -> float:
    """|u^T H v - v^T H u| relative to their size, for two random
    directions: the Hessian is symmetric, so a one-sided fault shows."""
    param = bkd.asarray(LV_PARAM)
    u, v = random_direction(bkd, 3), random_direction(bkd, 5)
    uhv = bkd.to_float((u.T @ problem.hvp(param, v))[0, 0])
    vhu = bkd.to_float((v.T @ problem.hvp(param, u))[0, 0])
    return abs(uhv - vhu) / max(abs(uhv), abs(vhu))


def min_gradient_fd_error(
    bkd: Backend[Array], problem: TrajectoryProblem[Array]
) -> float:
    """The smallest central-difference error of the gradient."""
    wrapper = FunctionWithJacobianFromCallable(
        nqoi=1, nvars=4, fun=problem.value, jacobian=problem.gradient, bkd=bkd
    )
    errors = DerivativeChecker(wrapper).check_derivatives(
        bkd.asarray(LV_PARAM), relative=True, central=True
    )[0]
    return bkd.to_float(bkd.min(errors))
