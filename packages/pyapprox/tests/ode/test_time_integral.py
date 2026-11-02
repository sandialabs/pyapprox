r"""The derivatives of a functional integrating over time by the
stepper's own method.

Lotka-Volterra with a mass from (10, 5), parameters (1, 0.1, 0.075, 1.5),
ten steps of 0.05, and :math:`q(y, t; p) = \cos(t)\, y^\top y / 100 +
(s^\top p)(\mathbf 1^\top y)`, which reads the state nonlinearly, the
time and the parameters.

- The adjoint gradient passes central finite differences, and the
  tangent sweep's :math:`dQ/dp \cdot v` equals the gradient's inner
  product with :math:`v`, for every tableau (stages read from the
  unknown, a trivial first stage from :math:`y_n`) and every multistep
  method (BDF's recursion weighs each step by the transposed recursion's
  response).
- The sensitivities :math:`\theta_n = \partial z_N/\partial b_n` (``b_n``
  step :math:`n`'s right-hand side) equal the last row of
  :math:`L^{-1}`, :math:`L` the lower-triangular matrix whose row
  :math:`n` holds step :math:`n`'s :math:`\alpha`, for BDF2 on unequal
  steps (startup included), and are 1 for Runge-Kutta and Adams methods.
- Against torch autograd through the forward solve, the gradient agrees
  to 1e-13, so the sink keeps the graph intact.
- Refusals: an integrand without gradients; a parameter gradient without
  the state gradient; a second-order sweep over a functional reading
  steps.
"""

import math
from typing import Callable, Generic, List

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.accumulating_sink import MultiSink
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import (
    adams_bashforth2,
    adams_moulton3,
    bdf2,
)
from pyapprox.ode.step_quadrature import MultistepQuadrature, TableauQuadrature
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.tableau import backward_euler, classical_rk4, heun
from pyapprox.ode.time_integral import (
    BoundTimeIntegral,
    IntegrandDerivatives,
    TimeIntegral,
    TimeIntegralStepTerms,
    recursion_sensitivities,
)
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.ode.runge_kutta_cases import LV_PARAM, lotka_volterra_with_mass
from tests._helpers.ode.trajectory_cases import (
    ALL_IDS,
    ALL_METHODS,
    START,
    STEPS,
    Method,
    Stepper,
    make_stepper,
    random_direction,
)

_S = [[0.3], [-0.2], [0.5], [0.1]]


class _Integrand(Generic[Array]):
    r""":math:`q = \cos(t)\, y^\top y / 100 + (s^\top p)(\mathbf 1^\top
    y)`, shape (1, 1), at the parameters last set."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._s = bkd.asarray(_S)
        self._param = bkd.asarray(LV_PARAM)

    def set_param(self, param: Array) -> None:
        self._param = param

    def value(self, state: Array, time: float) -> Array:
        bkd = self._bkd
        return math.cos(time) * bkd.dot(state.T, state) / 100.0 + bkd.dot(
            self._s.T, self._param
        ) * bkd.sum(state, axis=0, keepdims=True)

    def _state_gradient(self, state: Array, time: float) -> Array:
        weight = self._bkd.dot(self._s.T, self._param)
        return 2.0 * math.cos(time) * state / 100.0 + weight * self._bkd.ones(
            state.shape
        )

    def _param_gradient(self, state: Array, time: float) -> Array:
        return self._s * self._bkd.sum(state, axis=0, keepdims=True)

    def derivatives(self) -> IntegrandDerivatives[Array]:
        return IntegrandDerivatives.first_order(
            self._state_gradient, self._param_gradient
        )


class _Problem(Generic[Array]):
    r""":math:`p \mapsto Q(y(p), p)` with the sweeps' gradient and
    tangent."""

    def __init__(self, bkd: Backend[Array], method: Method) -> None:
        self._bkd = bkd
        self._model = lotka_volterra_with_mass(bkd)
        self._stepper: Stepper[Array] = make_stepper(
            method, dense_mass_form_system(self._model)
        )
        self._integrand = _Integrand(bkd)
        self._integral: TimeIntegral[Array, Array, Array] = TimeIntegral(
            self._integrand, self._stepper
        )
        self._sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            self._stepper
        )

    def solve(self, param: Array) -> tuple[Array, InMemoryTrajectoryStore[Array]]:
        self._model.set_param(param)
        self._integrand.set_param(param)
        sink = self._integral.sink()
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(self._stepper).solve(
            self._bkd.asarray(START), STEPS, MultiSink([sink, store])
        )
        return sink.value(), store

    def value(self, samples: Array) -> Array:
        return self._bkd.hstack(
            [self.solve(samples[:, k : k + 1])[0] for k in range(samples.shape[1])]
        )

    def gradient(self, sample: Array) -> Array:
        _, store = self.solve(sample)
        return self._sweeps.gradient(store, self._integral.on(store)).T

    def tangent(self, sample: Array, direction: Array) -> Array:
        _, store = self.solve(sample)
        return self._sweeps.tangent(store, self._integral.on(store), direction)

    def bound(
        self, store: InMemoryTrajectoryStore[Array]
    ) -> BoundTimeIntegral[Array, Array]:
        return self._integral.on(store)

    def sweeps(self) -> TrajectoryDerivatives[Array, Array]:
        return self._sweeps


class TestDerivatives:
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_gradient_passes_finite_differences(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem = _Problem(bkd, make())
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=4, fun=problem.value, jacobian=problem.gradient, bkd=bkd
        )
        errors = DerivativeChecker(wrapper).check_derivatives(
            bkd.asarray(LV_PARAM), relative=True, central=True
        )[0]
        assert bkd.to_float(bkd.min(errors)) < 1e-7

    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_tangent_is_the_gradient_along_the_direction(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem = _Problem(bkd, make())
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        bkd.assert_allclose(
            problem.tangent(param, direction),
            bkd.dot(problem.gradient(param), direction),
            rtol=1e-12,
        )


class TestSensitivities:
    def test_bdf2_is_the_last_row_of_the_recursion_matrix_inverse(self) -> None:
        sizes = (0.125, 0.25, 0.25, 0.0625, 0.125)
        spans: List[StepSpan] = [StepSpan.first(0.0, sizes[0])]
        for size in sizes[1:]:
            spans.append(spans[-1].advanced(size, 2))
        rules = [MultistepQuadrature(bdf2()).rule(span) for span in spans]
        # Row n of L: step n's recursion over z_1..z_N (z_0 = 0 drops).
        nsteps = len(rules)
        matrix = np.zeros((nsteps, nsteps))
        for n, rule in enumerate(rules):
            for lag, alpha in enumerate(rule.alpha):
                if n + 1 - lag >= 1:
                    matrix[n, n - lag] = alpha
        expected = np.linalg.inv(matrix)[-1]
        np.testing.assert_allclose(
            recursion_sensitivities(rules), expected, rtol=1e-14
        )

    @pytest.mark.parametrize(
        "quadrature",
        [
            TableauQuadrature(classical_rk4()),
            MultistepQuadrature(adams_bashforth2()),
            MultistepQuadrature(adams_moulton3()),
        ],
        ids=["RK4", "Adams-Bashforth 2", "Adams-Moulton 3"],
    )
    def test_runge_kutta_and_adams_weigh_every_step_once(
        self, quadrature: TableauQuadrature | MultistepQuadrature
    ) -> None:
        depth = 1 if isinstance(quadrature, TableauQuadrature) else 2
        spans = [StepSpan.first(0.0, 0.25)]
        for size in (0.125, 0.5, 0.25):
            spans.append(spans[-1].advanced(size, min(depth, spans[-1].nlags())))
        rules = [quadrature.rule(span) for span in spans]
        assert recursion_sensitivities(rules) == [1.0] * len(rules)


class TestAgainstAutograd:
    @pytest.mark.parametrize(
        "make",
        [backward_euler, heun, classical_rk4, bdf2, adams_bashforth2],
        ids=["backward Euler", "Heun", "RK4", "BDF2", "Adams-Bashforth 2"],
    )
    def test_gradient(
        self, torch_bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        torch = pytest.importorskip("torch")
        bkd = torch_bkd
        problem = _Problem(bkd, make())
        param = torch.tensor(LV_PARAM, requires_grad=True)
        value, _ = problem.solve(param)
        (auto_gradient,) = torch.autograd.grad(value[0, 0], param)
        bkd.assert_allclose(
            problem.gradient(param.detach()).T, auto_gradient, rtol=1e-13
        )


class TestRefusals:
    def test_an_integrand_without_gradients(self, bkd: Backend[Array]) -> None:
        problem = _Problem(bkd, backward_euler())
        _, store = problem.solve(bkd.asarray(LV_PARAM))
        stepper = make_stepper(
            backward_euler(), dense_mass_form_system(lotka_volterra_with_mass(bkd))
        )
        integral: TimeIntegral[Array, Array, Array] = TimeIntegral(
            _ValueOnly(bkd), stepper
        )
        with pytest.raises(TypeError, match="no state gradient"):
            integral.on(store)

    def test_a_parameter_gradient_needs_the_state_gradient(self) -> None:
        with pytest.raises(ValueError, match="needs the state gradient"):
            IntegrandDerivatives(None, _ValueOnly(NumpyBkd()).value)

    def test_a_second_order_sweep_refuses_step_terms(
        self, bkd: Backend[Array]
    ) -> None:
        problem = _Problem(bkd, backward_euler())
        _, store = problem.solve(bkd.asarray(LV_PARAM))
        functional = _SecondOrderReadingSteps(problem.bound(store))
        with pytest.raises(TypeError, match="reads steps"):
            problem.sweeps().gradient_and_hvp(
                store, functional, random_direction(bkd)
            )


class _ValueOnly(Generic[Array]):
    """An integrand with no derivatives."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def value(self, state: Array, time: float) -> Array:
        return self._bkd.dot(state.T, state)

    def derivatives(self) -> IntegrandDerivatives[Array]:
        return IntegrandDerivatives.none()


class _SecondOrderReadingSteps(Generic[Array]):
    """A time integral posing as a second-order functional (zero second
    derivatives of the stored states), to reach the refusal."""

    def __init__(self, bound: BoundTimeIntegral[Array, Array]) -> None:
        self._bound = bound

    def state_cotangent(self, index: int) -> Array:
        return self._bound.state_cotangent(index)

    def param_cotangent(self) -> Array:
        return self._bound.param_cotangent()

    def step_terms(self) -> TimeIntegralStepTerms[Array, Array]:
        return self._bound.step_terms()

    def state_state(self, index: int, w: Array) -> Array:
        return w * 0.0

    def state_param(self, index: int, v: Array) -> Array:
        return self._bound.state_cotangent(index)

    def param_state(self, index: int, w: Array) -> Array:
        return self._bound.param_cotangent()

    def param_param(self, v: Array) -> Array:
        return v * 0.0
