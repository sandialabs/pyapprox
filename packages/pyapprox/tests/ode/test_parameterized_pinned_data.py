r"""Pinned rows whose data depend on the parameters.

Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` from (10, 5),
parameters :math:`p = (a, b, c, d) = (1, 0.1, 0.075, 1.5)`, dt 0.05 to
0.5, Newton tolerance 1e-13; the prey row pinned to :math:`g(t; p) = 10 +
a^2 \sin 3t` (rate :math:`3 a^2 \cos 3t`), equal to the start at t = 0 for
every :math:`p`. The quantity is the predator at the final time, which
the pinned prey drives.

- Pinned inside the solve, at the step end and at every block, for
  backward Euler, Heun, SDIRK3 and BDF2: stored states hold
  :math:`g(t; p)` (1e-12); the stepper checker passes at a step's window,
  the data's derivatives in the parameter block and curvature; the
  quantity's gradient and Hessian-vector product pass finite differences
  (smallest error below 1e-7).
- In differentiated form (the layer, both treatments) and pinned at the
  step end: the same, for Heun and SDIRK3.
- Pinned after the step (``PinnedRowsConstraint``): the gradient and
  Hessian-vector product pass, for backward Euler and Heun.
- The continuous extension reads the data at its own time: both states
  at 0.137 and 0.41 pass the term checker (``fd_tol`` 3e-5, see
  ``test_terms_at_times``), for backward Euler, Heun and BDF2.
- Wrong-model control: the same data declared independent of the
  parameters give a gradient whose finite-difference error stays above
  1e-2.
- Unavailable derivatives are refused: by the sweeps, by the post-step
  constraint, and by the layer.
"""

from typing import Callable, Generic, List, Optional

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.differentiated_rows import (
    CoupledRows,
    RowRatesTreatmentProtocol,
    SymmetricRows,
    differentiated_rows,
)
from pyapprox.ode.integration import (
    DifferentiableStackConstraintProtocol,
    Integrator,
)
from pyapprox.ode.mass_form_system import MassFormSystem, dense_mass_form_system
from pyapprox.ode.multistep_methods import bdf2
from pyapprox.ode.observation_terms import EndpointTerm, MisfitTerm, StateEntries
from pyapprox.ode.pinned_rows import (
    AtEveryBlock,
    AtStepEnd,
    PinnedRowsConstraint,
    PinnedRowsParamDerivatives,
    PinPlacementProtocol,
    RowPinning,
)
from pyapprox.ode.residual_derivatives import DerivativeOperator, backend_spaces
from pyapprox.ode.stepper_derivatives_check import StepperBundleChecker
from pyapprox.ode.tableau import alexander_sdirk3, backward_euler, heun
from pyapprox.ode.terms_at_times import TermAtTimes
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.trajectory_terms_check import TrajectoryTermChecker
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowSet,
    DenseFormat,
)
from pyapprox.util.linalg.operators import BlockVector
from tests._helpers.ode.runge_kutta_cases import (
    LV_PARAM,
    backend_sample,
    lotka_volterra_with_mass,
)
from tests._helpers.ode.trajectory_cases import STEPS, Method, Stepper, make_stepper

_System = MassFormSystem[Array, Array, DenseFormat[Array]]
_METHODS: List[Callable[[], Method]] = [backward_euler, heun, alexander_sdirk3, bdf2]
_IDS = ["BE", "Heun", "SDIRK3", "BDF2"]
_PLACEMENTS: List[PinPlacementProtocol] = [AtStepEnd(), AtEveryBlock()]
_PLACEMENT_IDS = ["step end", "every block"]
_TREATMENTS: List[RowRatesTreatmentProtocol] = [CoupledRows(), SymmetricRows()]
_TREATMENT_IDS = ["coupled", "symmetric"]
_DIRECTION = [[0.7], [-1.1], [0.4], [0.3]]


def _sin(time: float) -> float:
    return float(np.sin(3.0 * time))


def _three_cos(time: float) -> float:
    return 3.0 * float(np.cos(3.0 * time))


class _DataJacobian(Generic[Array]):
    r""":math:`t \mapsto` the prey row :math:`(2 a\, s(t), 0, 0, 0)`, the
    parameter Jacobian of :math:`a^2 s(t)`."""

    def __init__(
        self, prey: "_ParamPrey[Array]", factor: Callable[[float], float]
    ) -> None:
        self._prey, self._factor = prey, factor

    def __call__(self, time: float) -> DerivativeOperator[Array]:
        prey = self._prey
        entry = 2.0 * prey.a() * self._factor(time)
        matrix = prey.bkd.asarray([[entry, 0.0, 0.0, 0.0], [0.0] * 4])
        return BackendDenseOperator(
            matrix, prey.params, prey.bkd, range=prey.states
        )


class _DataCurvature(Generic[Array]):
    r""":math:`(t, w) \mapsto w_0\, \partial^2 (a^2 s(t))/\partial p^2`,
    :math:`2 s(t) w_0` in the :math:`(a, a)` entry."""

    def __init__(
        self, prey: "_ParamPrey[Array]", factor: Callable[[float], float]
    ) -> None:
        self._prey, self._factor = prey, factor

    def __call__(self, time: float, weights: Array) -> DerivativeOperator[Array]:
        prey = self._prey
        entry = 2.0 * self._factor(time) * prey.bkd.to_float(weights[0, 0])
        matrix = prey.bkd.asarray(
            [[entry if (i, j) == (0, 0) else 0.0 for j in range(4)] for i in range(4)]
        )
        return BackendDenseOperator(matrix, prey.params, prey.bkd)


class _ParamPrey(Generic[Array]):
    r"""The prey pinned to :math:`10 + a^2 \sin 3t`, its parameters set
    with the model's; ``declared`` chooses what its bundle says."""

    def __init__(self, bkd: Backend[Array], declared: str = "derivatives") -> None:
        self.bkd = bkd
        states, params = backend_spaces(bkd, 2, 4)
        self.states, self.params = states.descriptor(), params.descriptor()
        self._rows: BackendRowSet[Array] = BackendRowSet(self.states, [0], bkd)
        self._a = 1.0
        self._declared = declared

    def set_param(self, param: Array) -> None:
        self._a = self.bkd.to_float(self.bkd.flatten(param)[0])

    def a(self) -> float:
        return self._a

    def data(self, time: float) -> float:
        return 10.0 + self._a**2 * _sin(time)

    def rows(self) -> BackendRowSet[Array]:
        return self._rows

    def values(self, time: float) -> Array:
        return self.bkd.asarray([[self.data(time)], [0.0]])

    def rates(self) -> Optional[Callable[[float], Array]]:
        return self._rate

    def _rate(self, time: float) -> Array:
        return self.bkd.asarray([[self._a**2 * _three_cos(time)], [0.0]])

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[Array]:
        if self._declared == "independent":
            return PinnedRowsParamDerivatives.independent()
        if self._declared == "unavailable":
            return PinnedRowsParamDerivatives.unavailable()
        return PinnedRowsParamDerivatives.second_order(
            _DataJacobian(self, _sin),
            _DataJacobian(self, _three_cos),
            _DataCurvature(self, _sin),
            _DataCurvature(self, _three_cos),
        )


class _Problem(Generic[Array]):
    """The model, the declaration, a stepper and optionally a post-step
    constraint, all moved by one ``set_param``."""

    def __init__(
        self,
        bkd: Backend[Array],
        method: Method,
        build: Callable[[_System[Array], _ParamPrey[Array]], _System[Array]],
        placement: Optional[PinPlacementProtocol] = AtStepEnd(),
        after_step: bool = False,
        declared: str = "derivatives",
    ) -> None:
        self.bkd = bkd
        self.model = lotka_volterra_with_mass(bkd)
        self.prey: _ParamPrey[Array] = _ParamPrey(bkd, declared)
        self.param = bkd.asarray(LV_PARAM)
        self.set_param(self.param)
        system = build(dense_mass_form_system(self.model), self.prey)
        pinning = None if placement is None else RowPinning(self.prey, placement)
        self.stepper: Stepper[Array] = make_stepper(method, system, pinning)
        self.constraint: Optional[DifferentiableStackConstraintProtocol[Array]] = None
        if after_step:
            _, params = backend_spaces(bkd, 2, 4)
            self.constraint = PinnedRowsConstraint(self.prey, system.space, params)
        self.sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            self.stepper, self.constraint
        )
        self.functional = TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(EndpointTerm(StateEntries(bkd, 2, [1])),)
        )

    def set_param(self, param: Array) -> None:
        self.model.set_param(param)
        self.prey.set_param(param)

    def solve(self, param: Array) -> InMemoryTrajectoryStore[Array]:
        self.set_param(param)
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(self.stepper, self.constraint).solve(
            self.bkd.asarray([[10.0], [5.0]]), STEPS, store
        )
        return store

    def errors(self) -> tuple[float, float]:
        """The smallest finite-difference errors of the gradient and of the
        Hessian-vector product, relative and central."""
        bkd = self.bkd

        def value(samples: Array) -> Array:
            return bkd.hstack(
                [
                    self.solve(samples[:, k : k + 1]).state(-1)[1:2, :]
                    for k in range(samples.shape[1])
                ]
            )

        def gradient(sample: Array) -> Array:
            return self.sweeps.gradient(self.solve(sample), self.functional).T

        def gradients(samples: Array) -> Array:
            return bkd.hstack(
                [gradient(samples[:, k : k + 1]).T for k in range(samples.shape[1])]
            )

        def hvp(sample: Array, direction: Array) -> Array:
            return self.sweeps.hvp(self.solve(sample), self.functional, direction)

        first = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=1, nvars=4, fun=value, jacobian=gradient, bkd=bkd
            )
        ).check_derivatives(self.param, relative=True, central=True)[0]
        second = DerivativeChecker(
            FunctionWithJVPFromCallable(
                nqoi=4, nvars=4, fun=gradients, jvp=hvp, bkd=bkd
            )
        ).check_derivatives(self.param, relative=True, central=True)[0]
        return bkd.to_float(bkd.min(first)), bkd.to_float(bkd.min(second))


def _raw(system: _System[Array], prey: _ParamPrey[Array]) -> _System[Array]:
    return system


class _Layered:
    def __init__(self, treatment: RowRatesTreatmentProtocol) -> None:
        self._treatment = treatment

    def __call__(
        self, system: _System[Array], prey: _ParamPrey[Array]
    ) -> _System[Array]:
        return differentiated_rows(system, prey, self._treatment)


class TestPinnedInsideTheSolve:
    @pytest.mark.parametrize("placement", _PLACEMENTS, ids=_PLACEMENT_IDS)
    @pytest.mark.parametrize("make", _METHODS, ids=_IDS)
    def test_states_and_the_stepper_checker(
        self,
        bkd: Backend[Array],
        make: Callable[[], Method],
        placement: PinPlacementProtocol,
    ) -> None:
        problem: _Problem[Array] = _Problem(bkd, make(), _raw, placement)
        store = problem.solve(problem.param)
        for n in range(1, store.ntimes()):
            value = bkd.to_float(store.state(n)[0, 0])
            assert abs(value - problem.prey.data(store.time(n))) <= 1e-12
        stepper, sample = problem.stepper, backend_sample(bkd)
        space = stepper.residual_space()
        adj = BlockVector(
            tuple(
                sample(space.component(i).descriptor(), 99 + i)
                for i in range(space.nblocks())
            )
        )
        checker = StepperBundleChecker(stepper, sample, set_param=problem.set_param)
        window = store.window(3, min(stepper.depth(), 4))
        assert checker.check(window, adj, problem.param).failures() == []

    @pytest.mark.parametrize("placement", _PLACEMENTS, ids=_PLACEMENT_IDS)
    @pytest.mark.parametrize("make", _METHODS, ids=_IDS)
    def test_gradient_and_hvp(
        self,
        bkd: Backend[Array],
        make: Callable[[], Method],
        placement: PinPlacementProtocol,
    ) -> None:
        gradient, hvp = _Problem(bkd, make(), _raw, placement).errors()
        assert gradient < 1e-7
        assert hvp < 1e-7


class TestDifferentiatedRows:
    @pytest.mark.parametrize("treatment", _TREATMENTS, ids=_TREATMENT_IDS)
    @pytest.mark.parametrize(
        "make", [heun, alexander_sdirk3], ids=["Heun", "SDIRK3"]
    )
    def test_gradient_and_hvp(
        self,
        bkd: Backend[Array],
        make: Callable[[], Method],
        treatment: RowRatesTreatmentProtocol,
    ) -> None:
        problem: _Problem[Array] = _Problem(bkd, make(), _Layered(treatment))
        gradient, hvp = problem.errors()
        assert gradient < 1e-7
        assert hvp < 1e-7


class TestPinnedAfterTheStep:
    @pytest.mark.parametrize("make", [backward_euler, heun], ids=["BE", "Heun"])
    def test_gradient_and_hvp(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem: _Problem[Array] = _Problem(
            bkd, make(), _raw, placement=None, after_step=True
        )
        store = problem.solve(problem.param)
        assert abs(bkd.to_float(store.state(-1)[0, 0]) - problem.prey.data(0.5)) < 1e-12
        gradient, hvp = problem.errors()
        assert gradient < 1e-7
        assert hvp < 1e-7


class TestTheContinuousExtension:
    @pytest.mark.parametrize(
        "make", [backward_euler, heun, bdf2], ids=["BE", "Heun", "BDF2"]
    )
    def test_terms_read_between_steps(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem: _Problem[Array] = _Problem(bkd, make(), _raw)
        entries = StateEntries(bkd, 2, [0, 1])
        term: MisfitTerm[Array, Array] = MisfitTerm(
            bkd,
            {
                0: (entries, bkd.asarray([[9.0], [5.0]])),
                1: (entries, bkd.asarray([[8.0], [6.0]])),
            },
            0.5,
        )
        functional = TrajectoryFunctional.from_terms(
            bkd,
            1,
            step_terms=(
                TermAtTimes(
                    term, problem.stepper.continuous_extension(), (0.137, 0.41), bkd
                ),
            ),
        )
        checker: TrajectoryTermChecker[Array, Array] = TrajectoryTermChecker(
            problem.stepper, bkd, backend_sample(bkd), problem.set_param
        )
        report = checker.check(
            functional, problem.solve(problem.param), param=problem.param
        )
        assert report.failures(fd_tol=3e-5) == []


class TestControlsAndRefusals:
    def test_data_declared_independent_give_a_wrong_gradient(
        self, bkd: Backend[Array]
    ) -> None:
        problem: _Problem[Array] = _Problem(
            bkd, backward_euler(), _raw, declared="independent"
        )
        gradient, _ = problem.errors()
        assert gradient > 1e-2

    def test_unavailable_derivatives_are_refused(self, bkd: Backend[Array]) -> None:
        problem: _Problem[Array] = _Problem(
            bkd, backward_euler(), _raw, declared="unavailable"
        )
        store = problem.solve(problem.param)
        with pytest.raises(ValueError, match="does not declare"):
            problem.sweeps.gradient(store, problem.functional)
        with pytest.raises(ValueError, match="does not declare"):
            _Problem(
                bkd,
                backward_euler(),
                _raw,
                placement=None,
                after_step=True,
                declared="unavailable",
            )
        with pytest.raises(ValueError, match="does not declare"):
            _Problem(bkd, heun(), _Layered(CoupledRows()), declared="unavailable")
