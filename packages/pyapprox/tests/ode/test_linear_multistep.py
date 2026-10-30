r"""The linear multistep stepper: forward steps and their derivatives.

On :math:`M \dot y = K y` (3 states, a nontrivial mass), the startup
step and a full step of BDF2, Adams-Bashforth 2 and Adams-Moulton 3,
with equal and with unequal earlier steps, equal their closed forms (the
linear solve of the step equation) to 1e-11. From (1, -0.5, 2) to T = 1
the error against the exact solution :math:`\exp(T M^{-1} K) y_0` falls
at rates of 2 (BDF2), 2 (Adams-Bashforth 2) and 3 (Adams-Moulton 3):
within 0.1 at fixed steps 0.05, 0.025, 0.0125, and within 0.05 on uneven
grids of 28, 56 and 112 cycles of steps 0.6, 1.4, 1.0 times h ending in
a shortened step of 0.5 h (observed 1.964/1.982, 2.004/2.002,
2.990/2.995).

On Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` from (10, 5),
the step residual vanishes (below 1e-10) at every recorded window,
startup and full, and the stepper checker passes at the startup window,
a full window and a window spanning steps of 0.03 and 0.07, second
derivatives included; on the grid 0, 0.05, 0.08, 0.15, 0.2, 0.26, 0.3,
0.37, 0.4, 0.46, 0.5 the gradient and HVP of the Tikhonov-augmented MSE
pass finite differences. On the mock stack, whose vectors are opaque,
the steps run and the checker passes. A fixed table (classical BDF2)
refuses a shortened last step and a changed second step, and takes a
one-step grid of any size and steps equal up to rounding. Refusals: an
implicit method without a nonlinear solver, and a step given too many
states or times that do not match them.
"""

from typing import Any, Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm

from pyapprox.ode.integration import Integrator
from pyapprox.ode.linear_multistep import LinearMultistepStepper
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import (
    MultistepCoefficients,
    MultistepFormulaProtocol,
    MultistepMethod,
    bdf2,
)
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.stepper_derivatives_check import StepperBundleChecker
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.linalg.mock_stack import MockVector, wrap
from tests._helpers.ode.multistep_cases import METHOD_IDS, METHODS, lm_stepper
from tests._helpers.ode.runge_kutta_cases import (
    LINEAR_K,
    LINEAR_MASS,
    LV_PARAM,
    MOCK_STATE,
    LinearMassModel,
    backend_sample,
    lotka_volterra_with_mass,
    mock_linear_system,
    mock_sample,
)
from tests._helpers.ode.trajectory_cases import (
    TrajectoryProblem,
    min_gradient_fd_error,
    min_hvp_fd_error,
)


def _closed_form(
    method: MultistepFormulaProtocol,
    past: list[NDArray[Any]],
    times: tuple[float, ...],
) -> NDArray[Any]:
    r"""The step of :math:`M \dot y = K y` reading ``past`` (most recent
    first) at ``times``: :math:`(\alpha_0 M - \Delta t \beta_0 K) y_{n+1}
    = -\sum_{l \ge 1} (\alpha_l M - \Delta t \beta_l K) y_{n+1-l}`."""
    mass, k = np.array(LINEAR_MASS), np.array(LINEAR_K)
    formula = method.coefficients_at(times)
    deltat = times[0] - times[1]
    rhs = -sum(
        (formula.alpha[lag] * mass - deltat * formula.beta[lag] * k) @ y
        for lag, y in enumerate(past, start=1)
    )
    lhs = formula.alpha[0] * mass - deltat * formula.beta[0] * k
    new: NDArray[Any] = np.linalg.solve(lhs, rhs)
    return new


_Y0 = np.array([[1.0], [-0.5], [2.0]])
_Y1 = np.array([[0.9], [-0.4], [1.8]])


class TestLinearSteps:
    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_startup_and_full_steps_are_their_closed_forms(
        self,
        bkd: Backend[Array],
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
    ) -> None:
        method = make()
        stepper = lm_stepper(method, dense_mass_form_system(LinearMassModel(bkd)))
        # t_{n+1} = 0.4 from t_n = 0.3; the earlier state at 0.2 (equal
        # steps) or 0.15 (the earlier step longer).
        for past, times in (
            ([_Y0], (0.4, 0.3)),
            ([_Y1, _Y0], (0.4, 0.3, 0.2)),
            ([_Y1, _Y0], (0.4, 0.3, 0.15)),
        ):
            unknown = stepper.step(tuple(bkd.asarray(y) for y in past), times)
            bkd.assert_allclose(
                unknown.blocks[0],
                bkd.asarray(_closed_form(method, past, times)),
                rtol=1e-11,
            )


def _solve(
    bkd: Backend[Array], stepper: Any, deltat: float, final: float
) -> InMemoryTrajectoryStore[Array]:
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    times = [deltat * n for n in range(round(final / deltat) + 1)]
    Integrator(stepper).solve(bkd.asarray([[10.0], [5.0]]), times, store)
    return store


_FINAL = 1.0
_DELTATS = (0.05, 0.025, 0.0125)


class TestNonlinearSteps:
    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_convergence_order(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
    ) -> None:
        r"""Against the exact solution of :math:`M \dot y = K y`,
        :math:`y(T) = \exp(T M^{-1} K)\, y_0`, so the rate depends on no
        other stepper."""
        bkd = numpy_bkd
        system = dense_mass_form_system(LinearMassModel(bkd))
        exact = expm(
            _FINAL * np.linalg.solve(np.array(LINEAR_MASS), np.array(LINEAR_K))
        )
        exact_final = exact @ _Y0
        errors = []
        for deltat in _DELTATS:
            store: InMemoryTrajectoryStore[NDArray[Any]] = InMemoryTrajectoryStore()
            times = [deltat * n for n in range(round(_FINAL / deltat) + 1)]
            Integrator(lm_stepper(make(), system)).solve(bkd.asarray(_Y0), times, store)
            errors.append(
                float(np.linalg.norm(bkd.to_numpy(store.state(-1)) - exact_final))
            )
        rates = [np.log2(e0 / e1) for e0, e1 in zip(errors, errors[1:])]
        assert all(abs(rate - order) < 0.1 for rate in rates), rates

    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_the_step_residual_vanishes_at_every_window(
        self,
        bkd: Backend[Array],
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
    ) -> None:
        stepper = lm_stepper(
            make(), dense_mass_form_system(lotka_volterra_with_mass(bkd))
        )
        store = _solve(bkd, stepper, 0.05, 0.5)
        for n in range(store.nsteps()):
            window = store.window(n, min(stepper.depth(), n + 1))
            residual = stepper.residual_space().norm(stepper.step_residual(window))
            assert bkd.to_float(residual[0, 0]) < 1e-10


class TestDerivatives:
    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    @pytest.mark.parametrize("step", [0, 3], ids=["startup", "full"])
    def test_the_checker_passes(
        self,
        bkd: Backend[Array],
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
        step: int,
    ) -> None:
        model = lotka_volterra_with_mass(bkd)
        param = bkd.asarray(LV_PARAM)
        model.set_param(param)
        stepper = lm_stepper(make(), dense_mass_form_system(model))
        store = _solve(bkd, stepper, 0.05, 0.5)
        window = store.window(step, min(stepper.depth(), step + 1))
        sample = backend_sample(bkd)
        adj = BlockVector(
            (sample(stepper.residual_space().component(0).descriptor(), 99),)
        )
        checker = StepperBundleChecker(stepper, sample, set_param=model.set_param)
        report = checker.check(window, adj, param)
        assert "param_param" in report.fd_errors
        assert report.failures() == []


def _mock_stepper(
    method: MultistepFormulaProtocol,
) -> LinearMultistepStepper[MockVector, NDArray[Any], Any]:
    newton: Newton[MockVector, NDArray[Any]] = Newton(atol=1e-13, rtol=0.0)
    return LinearMultistepStepper(
        method, mock_linear_system(), None if method.is_explicit() else newton
    )


class TestOnAnOpaqueStack:
    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_steps_and_checker(
        self, make: Callable[[], MultistepFormulaProtocol], order: int, explicit: bool
    ) -> None:
        stepper = _mock_stepper(make())
        store: InMemoryTrajectoryStore[MockVector] = InMemoryTrajectoryStore()
        times = [0.1 * n for n in range(6)]
        Integrator(stepper).solve(wrap(_Y0, MOCK_STATE), times, store)
        for step in (0, 3):
            window = store.window(step, min(stepper.depth(), step + 1))
            report = StepperBundleChecker(stepper, mock_sample).check(window)
            assert report.failures() == []


class TestRefusals:
    def test_an_implicit_method_needs_a_nonlinear_solver(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(TypeError, match="needs a nonlinear solver"):
            LinearMultistepStepper(bdf2(), dense_mass_form_system(LinearMassModel(bkd)))

    def test_too_many_states(self, bkd: Backend[Array]) -> None:
        stepper = lm_stepper(bdf2(), dense_mass_form_system(LinearMassModel(bkd)))
        y = bkd.asarray(_Y0)
        with pytest.raises(ValueError, match="between 1 and 2"):
            stepper.step((y, y, y), (0.1, 0.0, -0.1, -0.2))
        window = StepWindow((0.1, 0.0), (y,), stepper.step((y,), (0.1, 0.0)))
        off_diag = stepper.derivatives().off_diag
        assert off_diag is not None
        with pytest.raises(ValueError, match="lag must be between 1 and 1"):
            off_diag(window, 2)

    def test_times_that_do_not_match_the_states(self, bkd: Backend[Array]) -> None:
        stepper = lm_stepper(bdf2(), dense_mass_form_system(LinearMassModel(bkd)))
        y = bkd.asarray(_Y0)
        with pytest.raises(ValueError, match="needs 3 times"):
            stepper.step((y, y), (0.1, 0.0))


def _fixed_bdf2() -> MultistepMethod:
    """The classical BDF2 table, fixed, with backward Euler to start."""
    return MultistepMethod(
        coefficients=MultistepCoefficients(
            alpha=(1.0, -4.0 / 3.0, 1.0 / 3.0), beta=(2.0 / 3.0, 0.0, 0.0)
        ),
        startup=(MultistepCoefficients(alpha=(1.0, -1.0), beta=(1.0, 0.0)),),
        order=2,
        name="fixed BDF2",
    )


_CYCLES = (28, 56, 112)


def _uneven(ncycles: int) -> list[float]:
    """0 to 1 in ``ncycles`` cycles of steps 0.6, 1.4 and 1.0 times
    :math:`h` (ratios 0.6/1.0, 1.4/0.6 and 1.0/1.4, within variable-step
    BDF2's stability bound 1 + sqrt 2), then a shortened last step of
    0.5 h, with :math:`h = 1/(3\\,\\mathrm{ncycles} + 0.5)`. Doubling the
    cycles halves every step and keeps the grid's shape, the last step
    included, so the error falls at the method's rate."""
    base = _FINAL / (3 * ncycles + 0.5)
    sizes = [base * size for size in (0.6, 1.4, 1.0)] * ncycles + [0.5 * base]
    times = [0.0]
    for size in sizes[:-1]:
        times.append(times[-1] + size)
    times.append(_FINAL)
    return times


class TestVariableSteps:
    """The library's methods take their formulas from the step times, so
    they keep their order on uneven grids, and their derivatives stay
    consistent there."""

    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_convergence_order_on_uneven_steps(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
    ) -> None:
        r"""Against the exact solution of :math:`M \dot y = K y` at T = 1
        on :func:`_uneven` grids of 28, 56 and 112 cycles."""
        bkd = numpy_bkd
        system = dense_mass_form_system(LinearMassModel(bkd))
        exact = expm(
            _FINAL * np.linalg.solve(np.array(LINEAR_MASS), np.array(LINEAR_K))
        )
        errors = []
        for ncycles in _CYCLES:
            store: InMemoryTrajectoryStore[NDArray[Any]] = InMemoryTrajectoryStore()
            times = _uneven(ncycles)
            Integrator(lm_stepper(make(), system)).solve(bkd.asarray(_Y0), times, store)
            errors.append(
                float(np.linalg.norm(bkd.to_numpy(store.state(-1)) - exact @ _Y0))
            )
        rates = [np.log2(e0 / e1) for e0, e1 in zip(errors, errors[1:])]
        assert all(abs(rate - order) < 0.05 for rate in rates), rates

    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_the_checker_passes_at_an_uneven_window(
        self,
        bkd: Backend[Array],
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
    ) -> None:
        model = lotka_volterra_with_mass(bkd)
        param = bkd.asarray(LV_PARAM)
        model.set_param(param)
        stepper = lm_stepper(make(), dense_mass_form_system(model))
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        times = [0.0, 0.05, 0.08, 0.15, 0.2]
        Integrator(stepper).solve(bkd.asarray([[10.0], [5.0]]), times, store)
        # Step 2 spans steps of 0.03 and 0.07.
        window = store.window(2, stepper.depth())
        sample = backend_sample(bkd)
        adj = BlockVector(
            (sample(stepper.residual_space().component(0).descriptor(), 99),)
        )
        checker = StepperBundleChecker(stepper, sample, set_param=model.set_param)
        assert checker.check(window, adj, param).failures() == []

    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_gradient_and_hvp_on_uneven_steps(
        self,
        bkd: Backend[Array],
        make: Callable[[], MultistepFormulaProtocol],
        order: int,
        explicit: bool,
    ) -> None:
        times = [0.0, 0.05, 0.08, 0.15, 0.2, 0.26, 0.3, 0.37, 0.4, 0.46, 0.5]
        problem: TrajectoryProblem[Array] = TrajectoryProblem(
            bkd, make(), times=times
        )
        assert min_gradient_fd_error(bkd, problem) < 1e-7
        assert min_hvp_fd_error(bkd, problem) < 1e-7


class TestFixedTable:
    """A fixed table, derived for equal steps, refuses a step whose earlier
    steps had another size. Fixed BDF2 on M y' = K y from (1, -0.5, 2)."""

    def _solve(self, bkd: Backend[Array], times: list[float]) -> None:
        stepper = lm_stepper(
            _fixed_bdf2(), dense_mass_form_system(LinearMassModel(bkd))
        )
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(stepper).solve(bkd.asarray(_Y0), times, store)

    def test_a_shortened_last_step_is_refused(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="fixed coefficients"):
            self._solve(bkd, [0.0, 0.1, 0.2, 0.3, 0.35])

    def test_the_startup_formula_takes_any_size(self, bkd: Backend[Array]) -> None:
        # The one-state startup formula spans one step only.
        self._solve(bkd, [0.0, 0.37])

    def test_a_changed_second_step_is_refused(self, bkd: Backend[Array]) -> None:
        # Step 1 already reads two states, spanning steps of 0.05 and 0.1.
        with pytest.raises(ValueError, match="fixed coefficients"):
            self._solve(bkd, [0.0, 0.05, 0.15])

    def test_steps_equal_up_to_rounding_pass(self, bkd: Backend[Array]) -> None:
        # 0.1 * n differs from n steps of 0.1 by rounding.
        self._solve(bkd, [0.1 * n for n in range(8)])
