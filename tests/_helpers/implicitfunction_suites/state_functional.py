r"""Stack-generic suites: a functional of one state, its value, sum and
stack.

The pinned quadratic and :math:`Q = g + r` of
``tests._helpers.implicitfunction_stack_cases`` at :math:`p = (0.7,
-0.4)`, solved by Newton (atol 1e-13) with the stack's direct solver;
:math:`Q_2 = r` alone.

- The value is the terms' sum; ``a + b`` adds values; the weights of a
  scalar functional default to one and a stacked one needs them given.
- ``stack([Q, Q_2])`` has the two values one above another, and with
  weights :math:`\omega = (\omega_0, \omega_1)` its adjoint gradient and
  HVP are :math:`\omega_0` times :math:`Q`'s plus :math:`\omega_1` times
  :math:`Q_2`'s, to 1e-12; its tangent along a direction is the two
  parts' tangents one above another.
- An empty stack is refused.
"""

import pytest

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_stack_cases import (
    STATES,
    QuadraticParameterTerm,
    QuadraticResidual,
    QuadraticStateTerm,
    pinned_quadratic,
    quadratic_functional,
)
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V


def _penalty(f: QuadraticResidual[V, F, A]) -> StateFunctional[V, A]:
    return StateFunctional.from_terms(
        f.bkd, 1, parameter_terms=(QuadraticParameterTerm(f),)
    )


class TheValueSuite:
    def test_sum_and_weights(self, case: TypedStackCase[V, F, A]) -> None:
        f, _ = pinned_quadratic(case)
        bkd = case.bkd
        state = case.vector(bkd.asarray([[0.3], [-0.8], [1.1]]), STATES)
        q = quadratic_functional(f)
        g = QuadraticStateTerm(f).value(state)
        r = QuadraticParameterTerm(f).value()
        bkd.assert_allclose(q.value(state), g + r, rtol=1e-14)
        both = q + q
        bkd.assert_allclose(both.value(state), 2.0 * (g + r), rtol=1e-14)
        bkd.assert_allclose(q.weights(), bkd.ones((1, 1)))
        stacked = StateFunctional.stack(bkd, [q, _penalty(f)])
        assert stacked.nqoi() == 2
        bkd.assert_allclose(stacked.value(state), bkd.vstack([g + r, r]), rtol=1e-14)
        with pytest.raises(ValueError, match="needs its weights given"):
            stacked.weights()
        with pytest.raises(ValueError, match="cannot add"):
            stacked + q


class TheStackSuite:
    def test_derivatives_are_the_weighted_parts(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f, pinned = pinned_quadratic(case)
        bkd = case.bkd
        system = SteadySystem(pinned, f.states, RecomputeAlways(case.direct_solver()))
        newton: Newton[V, A] = Newton(atol=1e-13, rtol=0.0)
        point = system.solve(newton, f.states.zeros(1))
        q, q2 = quadratic_functional(f), _penalty(f)
        stacked = SteadyDerivatives(
            pinned, f.states, StateFunctional.stack(bkd, [q, q2])
        )
        parts = [
            SteadyDerivatives(pinned, f.states, q),
            SteadyDerivatives(pinned, f.states, q2),
        ]
        omega = bkd.asarray([[0.6], [-1.3]])
        direction = f.parameters.vector(bkd.asarray([[0.5], [-0.2]]))

        adjoint = stacked.adjoint(point, omega)
        hessian = stacked.hessian()
        assert hessian is not None
        gradient = case.array(stacked.gradient(adjoint))
        hvp = case.array(hessian.hvp(adjoint, direction))
        expected_gradient = bkd.zeros((2, 1))
        expected_hvp = bkd.zeros((2, 1))
        for k, part in enumerate(parts):
            part_adjoint = part.adjoint(point)
            part_hessian = part.hessian()
            assert part_hessian is not None
            expected_gradient = expected_gradient + omega[k, 0] * case.array(
                part.gradient(part_adjoint)
            )
            expected_hvp = expected_hvp + omega[k, 0] * case.array(
                part_hessian.hvp(part_adjoint, direction)
            )
        bkd.assert_allclose(gradient, expected_gradient, rtol=1e-12)
        bkd.assert_allclose(hvp, expected_hvp, rtol=1e-12)
        # Forward: one solve gives both entries, each its part's.
        expected_tangent = bkd.vstack(
            [part.tangent(point, direction) for part in parts]
        )
        bkd.assert_allclose(
            stacked.tangent(point, direction), expected_tangent, rtol=1e-12
        )

    def test_an_empty_stack(self, case: TypedStackCase[V, F, A]) -> None:
        with pytest.raises(ValueError, match="at least one functional"):
            StateFunctional.stack(case.bkd, [])
