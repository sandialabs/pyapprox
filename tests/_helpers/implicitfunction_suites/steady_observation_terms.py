r"""Stack-generic suites on array stacks: the observation, misfit and
Tikhonov terms on a steady solution.

``StateEntries`` reads backend-array states by index and
``TikhonovTerm`` computes on array parameters, so these suites run on
stacks whose vectors are arrays (``array_case``: CSR, dense NumPy, dense
Torch). A stack-generic observation would let them take ``case``.

The pinned quadratic of ``tests._helpers.implicitfunction_stack_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned, :math:`p = (0.7, -0.4)`), solved by Newton (``atol`` 1e-13) with
the stack's direct solver; :math:`H` observes entries 1 and 2 of
:math:`u`.

- :math:`Q = \|H u - d\|^2 / (2 \cdot 0.5^2) + (0.3/2)\, p^\top W p`, with
  :math:`d = (0.5, -0.3)` and :math:`W = [[2, 0.5], [0.5, 1]]`: its value
  is the two formulas evaluated directly (1e-14), its adjoint gradient
  the derivative of the solved value and its HVP the derivative of the
  gradient, by central finite differences (minimum relative error below
  1e-6).
- :math:`Q = H u` (two entries): its forward tangent along each
  parameter is the derivative of the solved value, likewise.
- Refusals: empty or out-of-range entries; a misfit of data of the wrong
  shape, of a non-operator or with non-positive noise; an observed value
  of a non-operator; a non-square or asymmetric weight, an unset or
  non-column parameter.
"""

from typing import Generic

import pytest

from pyapprox.implicitfunction.observation_terms import (
    ObservationMisfit,
    ObservedValue,
    StateEntries,
    StateObservationProtocol,
)
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.implicitfunction.tikhonov_term import TikhonovTerm
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_stack_cases import (
    QUADRATIC_PARAM,
    pinned_quadratic,
)
from tests._helpers.implicitfunction_suites.checks import min_fd_error
from tests._helpers.linalg.stack_case import A, F, TypedStackCase

_ROWS = [1, 2]
_DATA = [[0.5], [-0.3]]
_WEIGHT = [[2.0, 0.5], [0.5, 1.0]]


class _Solved(Generic[A, F]):
    """The pinned quadratic, its system, and a Tikhonov term kept at the
    residual's parameters."""

    def __init__(self, case: TypedStackCase[A, F, A]) -> None:
        self.bkd = bkd = case.bkd
        self.f, self.pinned = pinned_quadratic(case)
        self.system = SteadySystem(
            self.pinned, self.f.states, RecomputeAlways(case.direct_solver())
        )
        self.observation = StateEntries(bkd, 3, _ROWS)
        self.tikhonov = TikhonovTerm(bkd, 0.3, bkd.asarray(_WEIGHT))

    def solve(self, p: A) -> SteadyPoint[A, A]:
        self.f.set_param(p[:, 0])
        self.tikhonov.set_param(p)
        newton: Newton[A, A] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        point = self.system.solve(newton, self.f.states.zeros(1))
        assert point.converged()
        return point

    def misfit_functional(self) -> StateFunctional[A, A]:
        misfit = ObservationMisfit(
            self.bkd, self.observation, self.bkd.asarray(_DATA), noise_std=0.5
        )
        return StateFunctional.from_terms(
            self.bkd, 1, parameter_terms=(self.tikhonov,), state_terms=(misfit,)
        )


def _param(case: TypedStackCase[A, F, A]) -> A:
    return case.bkd.reshape(case.bkd.asarray(QUADRATIC_PARAM), (2, 1))


class TheMisfitAndTikhonovTermsSuite:
    def test_the_value(self, array_case: TypedStackCase[A, F, A]) -> None:
        bkd = array_case.bkd
        solved = _Solved(array_case)
        p = _param(array_case)
        derivs = SteadyDerivatives(
            solved.pinned, solved.f.states, solved.misfit_functional()
        )
        point = solved.solve(p)
        u = solved.system.space().copy(point.solution())
        residual = u[1:3] - bkd.asarray(_DATA)
        expected = 2.0 * bkd.sum(residual * residual, keepdims=True) + 0.15 * (
            p.T @ bkd.asarray(_WEIGHT) @ p
        )
        bkd.assert_allclose(derivs.value(point), expected, rtol=1e-14)

    def test_gradient_and_hvp(self, array_case: TypedStackCase[A, F, A]) -> None:
        bkd = array_case.bkd
        solved = _Solved(array_case)
        derivs = SteadyDerivatives(
            solved.pinned, solved.f.states, solved.misfit_functional()
        )
        hessian = derivs.hessian()
        assert hessian is not None

        def value(p: A) -> A:
            return derivs.value(solved.solve(p))

        def gradient(p: A) -> A:
            return derivs.gradient(derivs.adjoint(solved.solve(p)))

        def gradient_row(p: A) -> A:
            return gradient(p).T

        def hessian_matrix(p: A) -> A:
            adjoint = derivs.adjoint(solved.solve(p))
            eye = bkd.eye(2)
            return bkd.hstack(
                [hessian.hvp(adjoint, eye[:, j : j + 1]) for j in range(2)]
            )

        p = _param(array_case)
        assert min_fd_error(array_case, 1, 2, value, gradient_row, p) < 1e-6
        assert min_fd_error(array_case, 2, 2, gradient, hessian_matrix, p) < 1e-6


class TheObservedValueSuite:
    def test_the_tangent(self, array_case: TypedStackCase[A, F, A]) -> None:
        bkd = array_case.bkd
        solved = _Solved(array_case)
        functional: StateFunctional[A, A] = StateFunctional.from_terms(
            bkd, 2, state_terms=(ObservedValue(solved.observation),)
        )
        derivs = SteadyDerivatives(solved.pinned, solved.f.states, functional)

        def value(p: A) -> A:
            return derivs.value(solved.solve(p))

        def jacobian(p: A) -> A:
            point = solved.solve(p)
            eye = bkd.eye(2)
            return bkd.hstack(
                [derivs.tangent(point, eye[:, j : j + 1]) for j in range(2)]
            )

        p = _param(array_case)
        assert min_fd_error(array_case, 2, 2, value, jacobian, p) < 1e-6


class RefusalsSuite:
    def test_state_entries(self, array_case: TypedStackCase[A, F, A]) -> None:
        bkd = array_case.bkd
        with pytest.raises(ValueError, match="at least one entry"):
            StateEntries(bkd, 2, [])
        with pytest.raises(ValueError, match="not all in"):
            StateEntries(bkd, 2, [0, 2])
        assert isinstance(StateEntries(bkd, 2, [0]), StateObservationProtocol)

    def test_misfit_and_observed_value(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        bkd = array_case.bkd
        entries = StateEntries(bkd, 2, [0])
        datum = bkd.asarray([[1.0]])
        with pytest.raises(ValueError, match="must have shape"):
            ObservationMisfit(bkd, entries, bkd.asarray([[1.0], [2.0]]))
        with pytest.raises(TypeError, match="StateObservationProtocol"):
            ObservationMisfit(bkd, object(), datum)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive"):
            ObservationMisfit(bkd, entries, datum, noise_std=0.0)
        with pytest.raises(TypeError, match="StateObservationProtocol"):
            ObservedValue(object())  # type: ignore[arg-type]

    def test_tikhonov(self, array_case: TypedStackCase[A, F, A]) -> None:
        bkd = array_case.bkd
        with pytest.raises(ValueError, match="square"):
            TikhonovTerm(bkd, 1.0, bkd.ones((2, 3)))
        with pytest.raises(ValueError, match="symmetric"):
            TikhonovTerm(bkd, 1.0, bkd.asarray([[1.0, 2.0], [0.0, 1.0]]))
        term = TikhonovTerm(bkd, 1.0)
        with pytest.raises(ValueError, match="set_param"):
            term.value()
        with pytest.raises(ValueError, match="column"):
            term.set_param(bkd.ones((2,)))
