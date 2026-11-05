r"""The steppers' continuous extensions and state terms read at times
between steps.

Lotka-Volterra with a mass from (10, 5), parameters (1, 0.1, 0.075, 1.5).

- The extension ends where the step does: at :math:`t_n` it is the stored
  :math:`y_n` (to 1e-14) and at :math:`t_{n+1}` the stored
  :math:`y_{n+1}` (to 1e-11: an implicit method's new state is a Newton
  solve), for every tableau and multistep method, ten steps of 0.05.
- Its error at :math:`t^* = 0.135`, the middle of a step for steps of
  0.09, 0.03 and 0.01 (so the fraction of the step is the same, 1/2),
  against RK4 stepping to :math:`t^*` exactly in 1350 steps, falls at
  rate :math:`\min(p, p^* + 1)`, :math:`p^*` the extension's order:
  1 (Euler), 2 (Crank-Nicolson, Heun, implicit midpoint), 4 (RK4, whose
  extension is order 3), 2 for SDIRK3, whose default linear extension
  caps it, and for the multistep methods their own polynomials' (BDF2
  and Adams-Bashforth 2: 2; Adams-Moulton 3: 3). Measured for the last
  refinement: 1.03, 0.97, 2.00, 2.00, 2.00, 3.96, 2.00, 2.00, 2.01,
  3.00; asserted within 0.2.
- The multistep coefficients on the variable-step span (0, 0.1, 0.25):
  BDF's are its interpolant of y (the new state at theta = 1, y_n at 0,
  constants and linears reproduced), Adams' the partial integrals of
  its slopes' interpolant (the step's beta at theta = 1, zero at 0, a
  constant slope integrating to theta); a fixed table has none, and its
  stepper's extension is refused.
- A misfit of both states observed at 0.137, 0.263, 0.41 and 0.42 (two
  times in one step) and at the stored time 0.25, read at the extension
  through :class:`TermAtTimes`, passes the term checker at every step,
  first and second derivatives, for every tableau, ten steps of 0.05,
  with ``fd_tol`` 3e-5, twice the worst measured floor (the forward
  curves are correct V's whose bottoms reach at most 1.6e-5, the
  coupling :math:`h b(\theta)` being weak); a step with no observation
  is checked against zero exactly.
- Its gradient by the adjoint sweep passes central differences, and its
  Hessian-vector product central differences of the gradient (smallest
  error below 1e-7), for backward Euler, Heun, implicit midpoint, RK4,
  BDF2 and Adams-Moulton 3.
- Refusals: a time outside the step; a non-term, a non-extension, no
  times.
"""

import math
from typing import Callable, Generic

import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
    FunctionWithJVPFromCallable,
)
from pyapprox.ode.continuous_extension import ContinuousExtensionProtocol
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import (
    AdamsBashforth,
    AdamsMoulton,
    BackwardDifferentiation,
    MultistepCoefficients,
    MultistepMethod,
    adams_moulton3,
    bdf2,
)
from pyapprox.ode.observation_terms import MisfitTerm, StateEntries
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import (
    backward_euler,
    classical_rk4,
    heun,
    implicit_midpoint,
)
from pyapprox.ode.terms_at_times import TermAtTimes
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.trajectory_terms_check import TrajectoryTermChecker
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.ode.runge_kutta_cases import (
    LV_PARAM,
    backend_sample,
    lotka_volterra_with_mass,
)
from tests._helpers.ode.trajectory_cases import (
    ALL_IDS,
    ALL_METHODS,
    START,
    TIMES,
    Method,
    Stepper,
    make_stepper,
)

_RATE = {
    "forward Euler": 1,
    "backward Euler": 1,
    "Crank-Nicolson": 2,
    "implicit midpoint": 2,
    "Heun": 2,
    "classical RK4": 4,
    "Alexander SDIRK3": 2,
    "BDF2": 2,
    "Adams-Bashforth 2": 2,
    "Adams-Moulton 3": 3,
}
_OBSERVED = (0.137, 0.263, 0.41, 0.42, 0.25)


class _Case(Generic[Array]):
    """A stepper of a tableau or multistep method on Lotka-Volterra and
    its trajectory at ``LV_PARAM`` over ``times``."""

    def __init__(
        self, bkd: Backend[Array], method: Method, times: list[float]
    ) -> None:
        self.bkd = bkd
        self.model = lotka_volterra_with_mass(bkd)
        self.stepper: Stepper[Array] = make_stepper(
            method, dense_mass_form_system(self.model)
        )
        self.times = times
        self.param = bkd.asarray(LV_PARAM)
        self.store = self.solve(self.param)

    def solve(self, param: Array) -> InMemoryTrajectoryStore[Array]:
        self.model.set_param(param)
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(self.stepper).solve(
            self.bkd.asarray(START), GivenSizes.from_times(self.times), store
        )
        return store

    def extension(self) -> ContinuousExtensionProtocol[Array, Array]:
        return self.stepper.continuous_extension()

    def window(
        self, store: InMemoryTrajectoryStore[Array], step: int
    ) -> StepWindow[Array]:
        """Step ``step``'s window, reading the stepper's depth of states
        or all there are at a startup step."""
        return store.window(step, min(self.stepper.depth(), step + 1))

    def misfit(self) -> TermAtTimes[Array, Array]:
        """Both states observed at ``_OBSERVED``, noise 0.5."""
        bkd = self.bkd
        entries = StateEntries(bkd, 2, [0, 1])
        observations = {
            k: (entries, bkd.asarray([[9.0 - k], [5.0 + 0.5 * k]]))
            for k in range(len(_OBSERVED))
        }
        term: MisfitTerm[Array, Array] = MisfitTerm(bkd, observations, 0.5)
        return TermAtTimes(term, self.extension(), _OBSERVED, bkd)


class TestTheExtension:
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_it_ends_where_the_step_does(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        case = _Case(bkd, make(), TIMES)
        extension, store = case.extension(), case.store
        for step in range(store.nsteps()):
            window = case.window(store, step)
            bkd.assert_allclose(
                extension.value(window, store.time(step)),
                store.state(step),
                rtol=1e-14,
            )
            bkd.assert_allclose(
                extension.value(window, store.time(step + 1)),
                store.state(step + 1),
                rtol=1e-11,
            )

    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_its_error_falls_at_its_order(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        time = 0.135
        reference = _Case(
            bkd, classical_rk4(), [time * k / 1350 for k in range(1351)]
        ).store.state(-1)
        errors = []
        for deltat in (0.09, 0.03, 0.01):
            step = int(round(time / deltat - 0.5))
            case = _Case(bkd, make(), [deltat * k for k in range(step + 2)])
            value = case.extension().value(case.window(case.store, step), time)
            errors.append(bkd.to_float(bkd.norm(value - reference)))
        rate = math.log(errors[-2] / errors[-1], 3)
        assert abs(rate - _RATE[make().name]) < 0.2, f"rate {rate:.2f}"

    def test_a_time_outside_the_step_is_refused(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, heun(), TIMES)
        with pytest.raises(ValueError, match="outside the step"):
            case.extension().value(case.window(case.store, 0), 0.2)


class TestMultistepCoefficients:
    """On the variable-step span (0, 0.1, 0.25): its nodes in the step's
    variable are 1, 0 and -2/3."""

    _SPAN = StepSpan.from_times((0.0, 0.1, 0.25))
    _NODES = (1.0, 0.0, -0.1 / 0.15)

    def _close(self, left: tuple[float, ...], right: tuple[float, ...]) -> bool:
        return max(abs(a - b) for a, b in zip(left, right)) < 1e-14

    def test_bdf_is_the_interpolant_of_y(self) -> None:
        method = BackwardDifferentiation(2, "BDF2")
        at_end = method.continuous_coefficients_at(self._SPAN, 1.0)
        at_start = method.continuous_coefficients_at(self._SPAN, 0.0)
        assert self._close(at_end.states, (1.0, 0.0, 0.0))
        assert self._close(at_start.states, (0.0, 1.0, 0.0))
        for theta in (0.3, 0.7):
            c = method.continuous_coefficients_at(self._SPAN, theta)
            assert all(d == 0.0 for d in c.slopes)
            # It reproduces constants and linears in s exactly.
            assert abs(sum(c.states) - 1.0) < 1e-14
            linear = sum(w * s for w, s in zip(c.states, self._NODES))
            assert abs(linear - theta) < 1e-14

    @pytest.mark.parametrize(
        "method",
        [AdamsBashforth(2, "Adams-Bashforth 2"), AdamsMoulton(2, "Adams-Moulton 3")],
        ids=["Adams-Bashforth 2", "Adams-Moulton 3"],
    )
    def test_adams_integrates_the_slopes_interpolant(
        self, method: AdamsBashforth | AdamsMoulton
    ) -> None:
        at_end = method.continuous_coefficients_at(self._SPAN, 1.0)
        assert self._close(at_end.states, (0.0, 1.0, 0.0))
        assert self._close(at_end.slopes, method.coefficients_at(self._SPAN).beta)
        at_start = method.continuous_coefficients_at(self._SPAN, 0.0)
        assert self._close(at_start.slopes, (0.0, 0.0, 0.0))
        for theta in (0.3, 0.7):
            c = method.continuous_coefficients_at(self._SPAN, theta)
            # A constant slope integrates to theta.
            assert abs(sum(c.slopes) - theta) < 1e-14

    def test_a_fixed_table_has_none(self, bkd: Backend[Array]) -> None:
        fixed = MultistepMethod(
            MultistepCoefficients(
                alpha=(1.0, -4.0 / 3.0, 1.0 / 3.0), beta=(2.0 / 3.0, 0.0, 0.0)
            ),
            (MultistepCoefficients(alpha=(1.0, -1.0), beta=(1.0, 0.0)),),
            2,
            "fixed BDF2",
        )
        assert fixed.continuous_coefficients_at(self._SPAN, 0.5) is None
        # Steps of exactly 0.0625, which the fixed table needs.
        case = _Case(bkd, fixed, [0.0625 * k for k in range(9)])
        with pytest.raises(TypeError, match="no continuous extension"):
            case.extension().value(case.window(case.store, 2), 0.17)
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            bdf2().continuous_coefficients_at(self._SPAN, -0.5)


class TestTermAtTimes:
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_conformance(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        case = _Case(bkd, make(), TIMES)
        functional = TrajectoryFunctional.from_terms(
            bkd, 1, step_terms=(case.misfit(),)
        )
        checker: TrajectoryTermChecker[Array, Array] = TrajectoryTermChecker(
            case.stepper, bkd, backend_sample(bkd), case.model.set_param
        )
        report = checker.check(functional, case.store, param=case.param)
        # y(t) depends on the unknown only through h b(theta) f(Y), so the
        # action is small beside the value and rounding sets in early: the
        # forward curves are correct V's (order 1), every one passing the V
        # test, whose bottoms reach at most 1.6e-5 (Heun, both backends).
        # Finer or central differences do not help: their bottoms move to
        # h near 1e-2, failing the dip guard. So the floor is set at twice
        # the worst measured, the V test unchanged.
        assert report.failures(fd_tol=3e-5) == []
        # 0.41 and 0.42 lie in step 8, (0.4, 0.45].
        assert "step_terms[0]@8.curvature_wrt_unknown" in report.fd_checks
        # A step with no observation is checked against zero, exactly.
        assert report.fd_checks["step_terms[0]@0.wrt_unknown"].min_error() == 0.0

    @pytest.mark.parametrize(
        "make",
        [backward_euler, heun, implicit_midpoint, classical_rk4, bdf2, adams_moulton3],
        ids=[
            "backward Euler",
            "Heun",
            "implicit midpoint",
            "classical RK4",
            "BDF2",
            "Adams-Moulton 3",
        ],
    )
    def test_gradient_and_hvp(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        case = _Case(bkd, make(), TIMES)
        term = case.misfit()
        functional = TrajectoryFunctional.from_terms(bkd, 1, step_terms=(term,))
        sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            case.stepper
        )

        def value(samples: Array) -> Array:
            totals = []
            for k in range(samples.shape[1]):
                store = case.solve(samples[:, k : k + 1])
                total = bkd.zeros((1, 1))
                for step in range(store.nsteps()):
                    total = total + term.value(step, case.window(store, step))
                totals.append(total)
            return bkd.hstack(totals)

        def gradient(sample: Array) -> Array:
            store = case.solve(sample)
            return sweeps.gradient(store, functional).T

        def gradients(samples: Array) -> Array:
            return bkd.hstack(
                [gradient(samples[:, k : k + 1]).T for k in range(samples.shape[1])]
            )

        def hvp(sample: Array, direction: Array) -> Array:
            store = case.solve(sample)
            return sweeps.hvp(store, functional, direction)

        param = bkd.asarray(LV_PARAM)
        errors = DerivativeChecker(
            FunctionWithJacobianFromCallable(
                nqoi=1, nvars=4, fun=value, jacobian=gradient, bkd=bkd
            )
        ).check_derivatives(param, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(errors)) < 1e-7
        errors = DerivativeChecker(
            FunctionWithJVPFromCallable(
                nqoi=4, nvars=4, fun=gradients, jvp=hvp, bkd=bkd
            )
        ).check_derivatives(param, relative=True, central=True)[0]
        assert bkd.to_float(bkd.min(errors)) < 1e-7

    def test_refusals(self, bkd: Backend[Array]) -> None:
        case = _Case(bkd, heun(), TIMES)
        term = MisfitTerm(
            bkd, {0: (StateEntries(bkd, 2, [0]), bkd.asarray([[1.0]]))}
        )
        with pytest.raises(TypeError, match="StateTermProtocol"):
            TermAtTimes(object(), case.extension(), (0.1,), bkd)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ContinuousExtensionProtocol"):
            TermAtTimes(term, object(), (0.1,), bkd)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least one time"):
            TermAtTimes(term, case.extension(), (), bkd)
