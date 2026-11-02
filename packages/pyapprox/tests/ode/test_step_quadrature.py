r"""A stepper's own method applied to :math:`\dot z = q(y, t)`.

- Consistency with the step, exact to the solves' round-off: with
  :math:`q = f` the integral is :math:`M (y_N - y_0)` for every tableau and
  every multistep method, on unequal steps (startup steps included), so a
  node's weight, time and place (unknown block or stored state), and the
  recursion's :math:`\alpha`, are the ones the step used.
- The old rules as oracle: backward Euler, Crank-Nicolson and implicit
  midpoint give the old right-rectangle, trapezoid and midpoint values
  for a nonlinear, time-dependent :math:`q` (implicit midpoint's stage is
  the mean of the step's ends, the old rule's sample, for a nonsingular
  mass).
- Accuracy: for that :math:`q` along :math:`M \dot y = K y`, whose exact
  integral is known, halving the step shows each method's own order.
- The transpose: :func:`recursion_sensitivities` against
  :meth:`StepRule.advance` by duality, :math:`z_N = \theta^\top b` to
  1e-13 for random three-lag recursions and BDF2 on unequal steps; the
  weights are 1 for RK4, Adams-Bashforth 2 and Adams-Moulton 3.
- The rules themselves: classical RK4's nodes and their places; a stage
  of zero weight is no node; a multistep rule is the method's formula
  for the span.
- Refusals: a span of the wrong number of sizes; a malformed recursion;
  a sink receiving states out of order or asked for a value before a
  step; an integrand or stepper not satisfying its protocol.
"""

import math
from typing import Any, Callable, Generic, List, Protocol, Sequence, TypeVar

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from pyapprox.ode.accumulating_sink import MultiSink
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import (
    MultistepFormulaProtocol,
    adams_bashforth2,
    adams_moulton3,
    bdf2,
)
from pyapprox.ode.step_plan import FixedStep, GivenSizes
from pyapprox.ode.step_quadrature import (
    MultistepQuadrature,
    QuadratureNode,
    StepRule,
    TableauQuadrature,
    recursion_sensitivities,
)
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    classical_rk4,
    crank_nicolson,
    implicit_midpoint,
)
from pyapprox.ode.time_integral import IntegrandDerivatives, TimeIntegral
from pyapprox.ode.time_quadrature import (
    midpoint_quadrature,
    right_rectangle_quadrature,
    trapezoidal_quadrature,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector
from tests._helpers.ode.multistep_cases import METHOD_IDS, METHODS, lm_stepper
from tests._helpers.ode.runge_kutta_cases import (
    LINEAR_K,
    LINEAR_MASS,
    TABLEAU_IDS,
    TABLEAUX,
    LinearMassModel,
    rk_stepper,
)

A = TypeVar("A")

_Y0 = [[1.0], [-0.5], [2.0]]
# Unequal dyadic sizes, so every time is exact.
_SIZES = (0.125, 0.25, 0.0625, 0.5, 0.125, 0.25)


class _Rhs(Protocol[A]):
    def rhs(self, state: A, time: float) -> A: ...


class _RhsIntegrand(Generic[Array]):
    """:math:`q = f`, the stepper's own right-hand side: a vector-valued
    integrand, one entry per state."""

    def __init__(self, stepper: _Rhs[Array]) -> None:
        self._stepper = stepper

    def nqoi(self) -> int:
        return len(_Y0)

    def value(self, state: Array, time: float) -> Array:
        return self._stepper.rhs(state, time)

    def derivatives(self) -> IntegrandDerivatives[Array, Array]:
        return IntegrandDerivatives.none()


class _Energy(Generic[Array]):
    r""":math:`q(y, t) = \cos(t)\, y^\top y`, shape (1, 1)."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def nqoi(self) -> int:
        return 1

    def value(self, state: Array, time: float) -> Array:
        return math.cos(time) * self._bkd.dot(state.T, state)

    def derivatives(self) -> IntegrandDerivatives[Array, Array]:
        return IntegrandDerivatives.none()


def _energy_exact(final: float) -> float:
    """The exact integral of :class:`_Energy` along :math:`M \\dot y = K y`
    from ``_Y0`` at 0 to ``final``."""
    z = np.linalg.solve(np.array(LINEAR_MASS), np.array(LINEAR_K))
    y0 = np.array(_Y0)

    def integrand(t: float) -> float:
        y = expm(t * z) @ y0
        return float(math.cos(t) * (y.T @ y)[0, 0])

    value, _ = quad(integrand, 0.0, final, epsabs=1e-13, epsrel=1e-13)
    return float(value)


def _solve(
    bkd: Backend[Array], stepper: Any, integrand: Any, steps: Any
) -> tuple[Array, InMemoryTrajectoryStore[Array]]:
    integral: TimeIntegral[Array, Array] = TimeIntegral(integrand, stepper, bkd)
    sink = integral.sink()
    store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
    Integrator(stepper).solve(bkd.asarray(_Y0), steps, MultiSink([sink, store]))
    return sink.value(), store


def _multistep(
    make: Callable[[], MultistepFormulaProtocol], bkd: Backend[Array]
) -> Any:
    return lm_stepper(make(), dense_mass_form_system(LinearMassModel(bkd)))


def _runge_kutta(make: Callable[[], ButcherTableau], bkd: Backend[Array]) -> Any:
    return rk_stepper(make(), dense_mass_form_system(LinearMassModel(bkd)))


class TestConsistencyWithTheStep:
    @pytest.mark.parametrize("make", [case[0] for case in TABLEAUX], ids=TABLEAU_IDS)
    def test_integrating_f_is_the_mass_increment_runge_kutta(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        self._check(bkd, _runge_kutta(make, bkd))

    @pytest.mark.parametrize("make", [case[0] for case in METHODS], ids=METHOD_IDS)
    def test_integrating_f_is_the_mass_increment_multistep(
        self, bkd: Backend[Array], make: Callable[[], MultistepFormulaProtocol]
    ) -> None:
        self._check(bkd, _multistep(make, bkd))

    def _check(self, bkd: Backend[Array], stepper: Any) -> None:
        value, store = _solve(
            bkd, stepper, _RhsIntegrand(stepper), GivenSizes(0.0, _SIZES)
        )
        mass = LinearMassModel(bkd).mass_matrix()
        increment = mass.apply(store.state(store.ntimes() - 1) - store.state(0))
        bkd.assert_allclose(value, increment, rtol=1e-11, atol=1e-13)


def _old_value(bkd: Backend[Array], rule: Any, store: Any, q: _Energy[Array]) -> float:
    trajectory = bkd.hstack([store.state(i) for i in range(store.ntimes())])
    sampled = rule.sample(trajectory)
    weights, times = rule.weights(), rule.sample_times()
    return sum(
        float(weights[j]) * float(q.value(sampled[:, j : j + 1], float(times[j]))[0, 0])
        for j in range(sampled.shape[1])
    )


class TestTheOldRules:
    @pytest.mark.parametrize(
        "make, old",
        [
            (backward_euler, right_rectangle_quadrature),
            (crank_nicolson, trapezoidal_quadrature),
            (implicit_midpoint, midpoint_quadrature),
        ],
        ids=["backward Euler", "Crank-Nicolson", "implicit midpoint"],
    )
    def test_the_value_is_the_old_rules(
        self,
        bkd: Backend[Array],
        make: Callable[[], ButcherTableau],
        old: Callable[..., Any],
    ) -> None:
        q = _Energy(bkd)
        value, store = _solve(bkd, _runge_kutta(make, bkd), q, GivenSizes(0.0, _SIZES))
        rule = old(bkd.asarray(store.times()), bkd)
        np.testing.assert_allclose(
            float(value[0, 0]), _old_value(bkd, rule, store, q), rtol=1e-12
        )


def _rate(errors: Sequence[float]) -> float:
    return math.log2(errors[-2] / errors[-1])


class TestOrder:
    _FINAL = 1.0
    # Heun's error changes sign between 1/8 and 1/16, so the rate is
    # measured in the asymptotic range only.
    _DELTATS = (1 / 32, 1 / 64, 1 / 128)

    @pytest.mark.parametrize(
        "make, order", [case[:2] for case in TABLEAUX], ids=TABLEAU_IDS
    )
    def test_runge_kutta_order(
        self, make: Callable[[], ButcherTableau], order: int
    ) -> None:
        self._check(lambda bkd: _runge_kutta(make, bkd), order)

    @pytest.mark.parametrize("make, order, explicit", METHODS, ids=METHOD_IDS)
    def test_multistep_order(
        self, make: Callable[[], MultistepFormulaProtocol], order: int, explicit: bool
    ) -> None:
        self._check(lambda bkd: _multistep(make, bkd), order)

    def _check(self, build: Callable[[NumpyBkd], Any], order: int) -> None:
        bkd = NumpyBkd()
        exact = _energy_exact(self._FINAL)
        errors: List[float] = []
        for deltat in self._DELTATS:
            value, _ = _solve(
                bkd, build(bkd), _Energy(bkd), FixedStep(deltat, (0.0, self._FINAL))
            )
            errors.append(abs(float(value[0, 0]) - exact))
        assert abs(_rate(errors) - order) < 0.3, errors


def _chain(rules: Sequence[StepRule], increments: Sequence[float]) -> float:
    """:math:`z_N` by :meth:`StepRule.advance` from :math:`z_0 = 0`."""
    bkd = NumpyBkd()
    history = [bkd.zeros((1,))]  # most recent first
    for rule, increment in zip(rules, increments):
        past = history[: len(rule.alpha) - 1]
        history.insert(0, rule.advance(past, bkd.asarray([increment])))
    return float(history[0][0])


def _bdf2_rules() -> List[StepRule]:
    """BDF2's recursions on unequal steps, the startup step included."""
    spans = [StepSpan.first(0.0, 0.125)]
    for size in (0.25, 0.25, 0.0625, 0.125):
        spans.append(spans[-1].advanced(size, 2))
    return [MultistepQuadrature(bdf2()).rule(span) for span in spans]


class TestSensitivities:
    r""":func:`recursion_sensitivities` is the transpose of
    :meth:`StepRule.advance`, checked by duality.

    Stacking every step's recursion :math:`\alpha^{(n)}_0 z_{n+1} +
    \sum_{l \ge 1} \alpha^{(n)}_l z_{n+1-l} = b_n` gives one lower
    triangular system :math:`L z = b`, so :math:`z_N = e_N^\top L^{-1} b`,
    a linear function of the increments :math:`b`. It can be computed two
    ways:

    - forward, :math:`e_N^\top (L^{-1} b)`: solve for :math:`z` step by
      step with ``advance`` and read off its last entry (what the value
      sink does);
    - by the transpose, :math:`(L^{-\top} e_N)^\top b = \theta^\top b`:
      solve :math:`L^\top \zeta = e_N` backward once for the weights
      :math:`\theta = L^{-\top} e_N` and take their inner product with
      :math:`b` (what the derivatives use).

    The two agree for every :math:`b` exactly when the backward solve is
    the transpose of the forward one, the identity :math:`\langle y, A
    x\rangle = \langle A^\top y, x\rangle` with :math:`A = L^{-1}`, the
    same check that pairs an adjoint with its forward model. So the
    forward recursion and its transpose are checked against each other,
    with no independent oracle: a wrong coefficient, lag or sign on
    either side breaks the equality for a random :math:`b`."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_duality_on_random_recursions(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        rules = []
        for n in range(8):
            nlags = min(3, n + 1)
            alpha = (1.0 + rng.uniform(), *rng.normal(size=nlags))
            rules.append(StepRule(tuple(float(a) for a in alpha), ()))
        increments = [float(b) for b in rng.normal(size=len(rules))]
        theta = recursion_sensitivities(rules)
        np.testing.assert_allclose(
            _chain(rules, increments), float(np.dot(theta, increments)), rtol=1e-13
        )

    def test_duality_on_bdf2(self) -> None:
        rules = _bdf2_rules()
        increments = [float(b) for b in np.random.default_rng(3).normal(size=5)]
        theta = recursion_sensitivities(rules)
        np.testing.assert_allclose(
            _chain(rules, increments), float(np.dot(theta, increments)), rtol=1e-13
        )
        assert theta != [1.0] * len(rules)

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


class TestRules:
    def test_rk4_nodes_and_their_places(self) -> None:
        rule = TableauQuadrature(classical_rk4()).rule(StepSpan.first(0.5, 0.25))
        assert rule.alpha == (1.0, -1.0)
        assert rule.nodes == (
            QuadratureNode(0.25 / 6, 0.5, 1, 0),
            QuadratureNode(0.25 / 3, 0.625, 0, 0),
            QuadratureNode(0.25 / 3, 0.625, 0, 1),
            QuadratureNode(0.25 / 6, 0.75, 0, 2),
        )

    def test_a_stage_of_zero_weight_is_no_node(self) -> None:
        explicit_midpoint = ButcherTableau(
            ((0.0, 0.0), (0.5, 0.0)), (0.0, 1.0), (0.0, 0.5), "explicit midpoint"
        )
        rule = TableauQuadrature(explicit_midpoint).rule(StepSpan.first(0.0, 0.5))
        assert rule.nodes == (QuadratureNode(0.5, 0.25, 0, 0),)

    def test_a_multistep_rule_is_the_formula(self) -> None:
        span = StepSpan(0.25, (0.375, 0.125))
        formula = bdf2().coefficients_at(span)
        rule = MultistepQuadrature(bdf2()).rule(span)
        assert rule.alpha == formula.alpha
        assert rule.nodes == (QuadratureNode(0.125 * formula.beta[0], 0.75, 0, 0),)

    def test_advance_is_the_recursion(self, bkd: Backend[Array]) -> None:
        rule = StepRule((2.0, -3.0, 1.0), ())
        past = [bkd.asarray([5.0]), bkd.asarray([7.0])]
        bkd.assert_allclose(
            rule.advance(past, bkd.asarray([4.0])), bkd.asarray([(4 + 15 - 7) / 2])
        )

    def test_refusals(self) -> None:
        with pytest.raises(ValueError, match="1 sizes, got 2"):
            TableauQuadrature(backward_euler()).rule(StepSpan(0.0, (0.5, 0.5)))
        with pytest.raises(ValueError, match="alpha_0 != 0"):
            StepRule((0.0, 1.0), ())
        with pytest.raises(ValueError, match="alpha_0 != 0"):
            StepRule((1.0,), ())
        with pytest.raises(ValueError, match="beyond the step's 1 lags"):
            StepRule((1.0, -1.0), (QuadratureNode(1.0, 0.0, 2, 0),))
        zero = NumpyBkd().zeros((1,))
        with pytest.raises(ValueError, match="reads 1 earlier values, got 2"):
            StepRule((1.0, -1.0), ()).advance([zero] * 2, zero)


class TestSinkRefusals:
    def _sink(self, bkd: Backend[Array]) -> Any:
        stepper = _runge_kutta(backward_euler, bkd)
        integral: TimeIntegral[Array, Array] = TimeIntegral(
            _Energy(bkd), stepper, bkd
        )
        return integral.sink()

    def test_out_of_order_and_early_value(self, bkd: Backend[Array]) -> None:
        sink = self._sink(bkd)
        y = bkd.asarray(_Y0)
        unknown = BlockVector((y,))
        with pytest.raises(ValueError, match="append one first"):
            sink.append_step(StepSpan.first(0.0, 0.5), y, unknown, 0)
        sink.append(0.0, y)
        with pytest.raises(ValueError, match="no step has been received"):
            sink.value()
        with pytest.raises(ValueError, match="only the initial state"):
            sink.append(0.0, y)
        with pytest.raises(ValueError, match="not where the last one ended"):
            sink.append_step(StepSpan.first(0.25, 0.5), y, unknown, 0)

    def test_protocols_are_checked(self, bkd: Backend[Array]) -> None:
        stepper = _runge_kutta(backward_euler, bkd)
        with pytest.raises(TypeError, match="IntegrandProtocol"):
            TimeIntegral(object(), stepper, bkd)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="StepperProtocol"):
            TimeIntegral(_Energy(bkd), object(), bkd)  # type: ignore[arg-type]
