r"""The linear multistep stepper: forward steps and their derivatives.

On :math:`M \dot y = K y` (3 states, a nontrivial mass), the startup
step and a full step of BDF2, Adams-Bashforth 2 and Adams-Moulton 3
equal their closed forms (the linear solve of the step equation) to
1e-11, and from (1, -0.5, 2) to T = 1 at fixed steps 0.05, 0.025,
0.0125 the error against the exact solution :math:`\exp(T M^{-1} K)
y_0` falls at rates within 0.1 of 2 (BDF2), 2 (Adams-Bashforth 2) and 3
(Adams-Moulton 3). On Lotka-Volterra with the mass ``[[2, 0.5], [0.5,
1]]`` from (10, 5), the step residual vanishes (below 1e-10) at every
recorded window, startup and full, and
the stepper checker passes at the startup window and at a full window,
second derivatives included. On the mock stack, whose vectors are
opaque, the steps run and the checker passes. Refusals: an implicit
method without a nonlinear solver, and a step given too many states.
"""

from typing import Any, Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm

from pyapprox.ode.integration import Integrator
from pyapprox.ode.linear_multistep import LinearMultistepStepper
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import MultistepMethod, bdf2
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


def _closed_form(
    method: MultistepMethod, past: list[NDArray[Any]], deltat: float
) -> NDArray[Any]:
    r"""The step of :math:`M \dot y = K y` reading ``past`` (most recent
    first): :math:`(\alpha_0 M - \Delta t \beta_0 K) y_{n+1} = -\sum_{l
    \ge 1} (\alpha_l M - \Delta t \beta_l K) y_{n+1-l}`."""
    mass, k = np.array(LINEAR_MASS), np.array(LINEAR_K)
    formula = method.at(len(past))
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
        make: Callable[[], MultistepMethod],
        order: int,
        explicit: bool,
    ) -> None:
        method = make()
        stepper = lm_stepper(method, dense_mass_form_system(LinearMassModel(bkd)))
        for past in ([_Y0], [_Y1, _Y0]):
            # t_{n+1} = 0.4 from t_n = 0.3, the earlier state at 0.2.
            times = (0.4, 0.3, 0.2)[: len(past) + 1]
            unknown = stepper.step(tuple(bkd.asarray(y) for y in past), times)
            bkd.assert_allclose(
                unknown.blocks[0],
                bkd.asarray(_closed_form(method, past, 0.1)),
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
        make: Callable[[], MultistepMethod],
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
        make: Callable[[], MultistepMethod],
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
        make: Callable[[], MultistepMethod],
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
    method: MultistepMethod,
) -> LinearMultistepStepper[MockVector, NDArray[Any], Any]:
    newton: Newton[MockVector, NDArray[Any]] = Newton(atol=1e-13, rtol=0.0)
    return LinearMultistepStepper(
        method, mock_linear_system(), None if method.is_explicit() else newton
    )


class TestOnAnOpaqueStack:
    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_steps_and_checker(
        self, make: Callable[[], MultistepMethod], order: int, explicit: bool
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


class TestStepSizes:
    """The library's methods hold fixed coefficients, derived for equal
    steps, so they refuse a step whose earlier steps had another size.
    BDF2 on M y' = K y from (1, -0.5, 2)."""

    def _solve(self, bkd: Backend[Array], times: list[float]) -> None:
        stepper = lm_stepper(bdf2(), dense_mass_form_system(LinearMassModel(bkd)))
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
