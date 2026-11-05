r"""A functional of the step residual of the unconstrained system, read
at a solved trajectory: the flux through pinned rows.

Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` from (10, 5),
parameters (1, 0.1, 0.075, 1.5), steps of 0.05 to 0.5; the prey pinned
to :math:`g(t) = 10 + \sin 3t` inside each step's solve (at the step
end); the functional is the unconstrained residual's prey row of the
block producing :math:`y_{n+1}`, :math:`\Omega = ` ``LastBlock`` of the
prey entry.

- On a trajectory solved without pinning, every step's residual
  vanishes, so the functional is zero at every step (to 1e-10, Newton's
  tolerance), for every tableau and multistep method.
- On the pinned trajectory, for backward Euler, it equals the prey row
  of :math:`M (y_{n+1} - y_n) - h f(y_{n+1}, t_{n+1})` computed from the
  stored states and the model directly (to 1e-12), and is not zero.
- The term passes the term checker at every step of the pinned
  trajectory, first and second derivatives, for every method.
- The functional summed over steps has its gradient pass central
  differences and its Hessian-vector product central differences of the
  gradient (smallest error below 1e-7), for backward Euler, Heun and
  BDF2.
- On the mock stack (no parameters), the term over forward and backward
  Euler passes the checker, first derivatives, reaching the residual
  only through its observation.
- Refusals: a non-stepper, a non-observation.
"""

from typing import Any, Callable, Generic, Optional

import pytest
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import bdf2
from pyapprox.ode.observation_terms import StateEntries
from pyapprox.ode.pinned_rows import RowPinning
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.step_residual_terms import LastBlock, StepResidualTerm
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import backward_euler, forward_euler, heun
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
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
    Method,
    PinnedPrey,
    Stepper,
    make_stepper,
)


class _Case(Generic[Array]):
    """The pinned (or, with ``pinned`` False, unpinned) solve, and the
    unconstrained stepper of the same method on the same model."""

    def __init__(self, bkd: Backend[Array], method: Method, pinned: bool) -> None:
        self.bkd = bkd
        self.model = lotka_volterra_with_mass(bkd)
        self.param = bkd.asarray(LV_PARAM)
        self.model.set_param(self.param)
        prey: PinnedPrey[Array] = PinnedPrey(bkd)
        pinning: Optional[RowPinning[Array]] = RowPinning(prey) if pinned else None
        self.solver: Stepper[Array] = make_stepper(
            method, dense_mass_form_system(self.model), pinning
        )
        self.unconstrained: Stepper[Array] = make_stepper(
            method, dense_mass_form_system(self.model)
        )
        self.store = self.solve(self.param)

    def solve(self, param: Array) -> InMemoryTrajectoryStore[Array]:
        self.model.set_param(param)
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(self.solver).solve(self.bkd.asarray(START), STEPS, store)
        return store

    def window(
        self, store: InMemoryTrajectoryStore[Array], step: int
    ) -> StepWindow[Array]:
        return store.window(step, min(self.solver.depth(), step + 1))

    def flux(self) -> StepResidualTerm[Array, Array]:
        prey = LastBlock(
            StateEntries(self.bkd, 2, [0]), self.unconstrained.residual_space()
        )
        return StepResidualTerm(self.unconstrained, prey)


class TestTheFlux:
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_it_vanishes_on_an_unpinned_solve(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        case = _Case(bkd, make(), pinned=False)
        flux = case.flux()
        for step in range(case.store.nsteps()):
            value = flux.value(step, case.window(case.store, step))
            assert abs(bkd.to_float(value[0, 0])) < 1e-10

    def test_it_is_the_pinned_rows_residual(self, bkd: Backend[Array]) -> None:
        """Backward Euler: the prey row of M (y_{n+1} - y_n) - h f(y_{n+1}),
        computed from the stored states and the model."""
        case = _Case(bkd, backward_euler(), pinned=True)
        flux, store = case.flux(), case.store
        mass = bkd.asarray([[2.0, 0.5], [0.5, 1.0]])
        for step in range(store.nsteps()):
            old, new = store.state(step), store.state(step + 1)
            time, deltat = store.time(step + 1), store.time(step + 1) - store.time(step)
            slope = bkd.reshape(case.model(new[:, 0], time), (2, 1))
            expected = (bkd.dot(mass, new - old) - deltat * slope)[0, 0]
            value = flux.value(step, case.window(store, step))[0, 0]
            bkd.assert_allclose(
                bkd.asarray([bkd.to_float(value)]),
                bkd.asarray([bkd.to_float(expected)]),
                rtol=1e-12,
            )
            assert abs(bkd.to_float(expected)) > 1e-3

    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_conformance(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        case = _Case(bkd, make(), pinned=True)
        functional = TrajectoryFunctional.from_terms(bkd, 1, step_terms=(case.flux(),))
        checker: TrajectoryTermChecker[Array, Array] = TrajectoryTermChecker(
            case.unconstrained, bkd, backend_sample(bkd), case.model.set_param
        )
        report = checker.check(functional, case.store, param=case.param)
        assert report.failures() == []

    @pytest.mark.parametrize(
        "make", [backward_euler, heun, bdf2], ids=["backward Euler", "Heun", "BDF2"]
    )
    def test_gradient_and_hvp(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        case = _Case(bkd, make(), pinned=True)
        flux = case.flux()
        functional = TrajectoryFunctional.from_terms(bkd, 1, step_terms=(flux,))
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            case.solver
        )

        def value(samples: Array) -> Array:
            totals = []
            for k in range(samples.shape[1]):
                store = case.solve(samples[:, k : k + 1])
                total = bkd.zeros((1, 1))
                for step in range(store.nsteps()):
                    total = total + flux.value(step, case.window(store, step))
                totals.append(total)
            return bkd.hstack(totals)

        def gradient(sample: Array) -> Array:
            return sweeps.gradient(case.solve(sample), functional).T

        def gradients(samples: Array) -> Array:
            return bkd.hstack(
                [gradient(samples[:, k : k + 1]).T for k in range(samples.shape[1])]
            )

        def hvp(sample: Array, direction: Array) -> Array:
            return sweeps.hvp(case.solve(sample), functional, direction)

        errors = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=1, nvars=4, fun=value, jacobian=gradient, bkd=bkd
            )
        ).check_derivatives(case.param, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(errors)) < 1e-7
        errors = DerivativeChecker(
            FunctionWithJVPFromCallable(
                nqoi=4, nvars=4, fun=gradients, jvp=hvp, bkd=bkd
            )
        ).check_derivatives(case.param, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(errors)) < 1e-7

    def test_refusals(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, backward_euler(), pinned=True)
        prey = LastBlock(
            StateEntries(bkd, 2, [0]), case.unconstrained.residual_space()
        )
        with pytest.raises(TypeError, match="DifferentiableStepperProtocol"):
            StepResidualTerm(object(), prey)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ObservationOperatorProtocol"):
            StepResidualTerm(case.unconstrained, object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ObservationOperatorProtocol"):
            LastBlock(object(), case.unconstrained.residual_space())  # type: ignore[arg-type]


_MockSpace = InnerProductSpaceProtocol[MockVector, NDArray[Any]]


class _MockEntry:
    r"""The observation :math:`e^\top y` through the space's inner
    product, for a fixed vector :math:`e`."""

    def __init__(self, space: _MockSpace, direction: MockVector) -> None:
        self._space, self._direction = space, direction

    def nobs(self) -> int:
        return 1

    def apply(self, state: MockVector) -> NDArray[Any]:
        return self._space.dot(self._direction, state)

    def apply_transpose(self, values: NDArray[Any]) -> MockVector:
        return self._space.scale(float(values[0, 0]), self._direction)


class TestOnAnOpaqueStack:
    @pytest.mark.parametrize("make", [forward_euler, backward_euler])
    def test_the_term_uses_only_the_spaces(
        self, make: Callable[[], Any]
    ) -> None:
        bkd = NumpyBkd()
        stepper = mock_stepper(make())
        space = stepper.system().space
        store: InMemoryTrajectoryStore[MockVector] = InMemoryTrajectoryStore()
        Integrator(stepper).solve(
            mock_sample(space.descriptor(), 3), GivenSizes(0.0, [0.1] * 4), store
        )
        observation = LastBlock(
            _MockEntry(space, mock_sample(space.descriptor(), 5)),
            stepper.residual_space(),
        )
        term: StepResidualTerm[MockVector, NDArray[Any]] = StepResidualTerm(
            stepper, observation
        )
        functional = TrajectoryFunctional.from_terms(bkd, 1, step_terms=(term,))
        checker: TrajectoryTermChecker[MockVector, NDArray[Any]] = (
            TrajectoryTermChecker(stepper, bkd, mock_sample)
        )
        report = checker.check(functional, store)
        assert report.failures() == []
        assert "step_terms[0]@3.wrt_unknown" in report.fd_checks
