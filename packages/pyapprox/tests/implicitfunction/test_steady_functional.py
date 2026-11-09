r"""A steady functional's value, sum and stack.

The pinned quadratic and :math:`Q = g + r` of
``tests._helpers.implicitfunction_cases`` at :math:`p = (0.7, -0.4)`,
solved by Newton (atol 1e-13) with dense LU; :math:`Q_2 = r` alone.

- The value is the terms' sum; ``a + b`` adds values; the weights of a
  scalar functional default to one and a stacked one needs them given.
- ``stack([Q, Q_2])`` has the two values one above another, and with
  weights :math:`\omega = (\omega_0, \omega_1)` its adjoint gradient and
  HVP are :math:`\omega_0` times :math:`Q`'s plus :math:`\omega_1` times
  :math:`Q_2`'s, to 1e-12.
- An empty stack is refused.
"""

import pytest

from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_functional import SteadyFunctional
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_cases import (
    QuadraticParameterTerm,
    QuadraticResidual,
    QuadraticStateTerm,
    pinned_quadratic,
    quadratic_functional,
)


def _penalty(
    f: QuadraticResidual[Array], bkd: Backend[Array]
) -> SteadyFunctional[Array, Array]:
    return SteadyFunctional.from_terms(
        bkd, 1, parameter_terms=(QuadraticParameterTerm(f),)
    )


class TestTheValue:
    def test_sum_and_weights(self, bkd: Backend[Array]) -> None:
        f, _ = pinned_quadratic(bkd)
        state = bkd.asarray([[0.3], [-0.8], [1.1]])
        q = quadratic_functional(f)
        g = QuadraticStateTerm(f).value(state)
        r = QuadraticParameterTerm(f).value()
        bkd.assert_allclose(q.value(state), g + r, rtol=1e-14)
        both = q + q
        bkd.assert_allclose(both.value(state), 2.0 * (g + r), rtol=1e-14)
        bkd.assert_allclose(q.weights(), bkd.ones((1, 1)))
        stacked = SteadyFunctional.stack(bkd, [q, _penalty(f, bkd)])
        assert stacked.nqoi() == 2
        bkd.assert_allclose(stacked.value(state), bkd.vstack([g + r, r]), rtol=1e-14)
        with pytest.raises(ValueError, match="needs its weights given"):
            stacked.weights()
        with pytest.raises(ValueError, match="cannot add"):
            stacked + q


class TestTheStack:
    def test_derivatives_are_the_weighted_parts(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        system = SteadySystem(pinned, f.states, BackendDenseLU(bkd))
        point = system.solve(Newton(atol=1e-13, rtol=0.0), bkd.zeros((3, 1)))
        q, q2 = quadratic_functional(f), _penalty(f, bkd)
        stacked = SteadyDerivatives(system, SteadyFunctional.stack(bkd, [q, q2]))
        parts = [SteadyDerivatives(system, q), SteadyDerivatives(system, q2)]
        omega = bkd.asarray([[0.6], [-1.3]])
        direction = bkd.asarray([[0.5], [-0.2]])

        adjoint = stacked.adjoint(point, omega)
        hessian = stacked.hessian()
        assert hessian is not None
        gradient = stacked.gradient(adjoint)
        hvp = hessian.hvp(adjoint, direction)
        expected_gradient = bkd.zeros((2, 1))
        expected_hvp = bkd.zeros((2, 1))
        for k, part in enumerate(parts):
            part_adjoint = part.adjoint(point)
            part_hessian = part.hessian()
            assert part_hessian is not None
            expected_gradient = expected_gradient + omega[k, 0] * part.gradient(
                part_adjoint
            )
            expected_hvp = expected_hvp + omega[k, 0] * part_hessian.hvp(
                part_adjoint, direction
            )
        bkd.assert_allclose(gradient, expected_gradient, rtol=1e-12)
        bkd.assert_allclose(hvp, expected_hvp, rtol=1e-12)

    def test_an_empty_stack(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="at least one functional"):
            SteadyFunctional.stack(bkd, [])
