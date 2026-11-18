r"""Stack-generic suites on array stacks: a linear observation of a
residual, :math:`Q = H F(u)`, on a steady solution.

``RowWeights`` and ``StateEntries`` read backend-array states, so these
suites run on stacks whose vectors are arrays (``array_case``: CSR,
dense NumPy, dense Torch). A stack-generic row-reading observation (weight
vectors in the stack, :math:`Q = W^\top R` by the space's dot) would let
them take ``case``.

The pinned quadratic of ``tests._helpers.implicitfunction_stack_cases``
(:math:`F_i = (A u)_i + p_0 u_i^2 + p_1^2 c_i`, rows :math:`\{0, 2\}`
pinned, :math:`p = (0.7, -0.4)`), solved by Newton (``atol`` 1e-13) with
the stack's direct solver. :math:`H = W^\top`, :math:`W = [e_0, e_2]`,
observes the unpinned :math:`F` on the pinned rows, where it does not
vanish at the solution: the analog of an essential boundary's flux. The
term is built from the solved pinned residual (``pinned_rows_residual``).
Weights :math:`\omega = (1.5, -0.5)` make it scalar for the adjoint.

- The value is :math:`F` evaluated directly on rows 0 and 2 (1e-14), and
  :math:`F` on the free row vanishes (1e-12).
- The adjoint gradient of :math:`\omega^\top Q` is the derivative of the
  solved value, its HVP the derivative of the gradient, and the forward
  tangent of :math:`Q` the derivative of both entries, by central finite
  differences (minimum relative error below 1e-6).
- The term's bundle follows the residual's: none without a parameter
  Jacobian, first order without second derivatives, and a state-state
  block skipped where the residual states it zero (the affine quadratic,
  checked directly: it is affine only at :math:`p_0 = 0`, so finite
  differences in :math:`p_0` would test a false declaration).
- It reads the solve's own :math:`F`: after ``set_param`` on the solved
  problem, the term's value moves with it.
- Refusals: a residual, observation or space not satisfying its
  protocol; an observation of another space; weights of the wrong shape
  or nonzero off the pinned rows (a free row, where :math:`F` vanishes);
  a residual that is not pinned.
"""

from typing import Generic

import pytest

from pyapprox.implicitfunction.observation_terms import RowWeights, StateEntries
from pyapprox.implicitfunction.observed_residual import (
    ObservedResidual,
    pinned_rows_residual,
)
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.implicitfunction_stack_cases import (
    QUADRATIC_PARAM,
    QuadraticResidual,
    pinned_quadratic,
)
from tests._helpers.implicitfunction_suites.checks import min_fd_error
from tests._helpers.linalg.stack_case import A, F, TypedStackCase

_PINNED = [0, 2]
_WEIGHTS = [[1.5], [-0.5]]


def _unit_columns(case: TypedStackCase[A, F, A]) -> A:
    """:math:`W = [e_0, e_2]`: each pinned row read on its own."""
    return case.bkd.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])


class _Solved(Generic[A, F]):
    """The pinned quadratic, its system, and the observed unpinned
    residual."""

    def __init__(self, case: TypedStackCase[A, F, A]) -> None:
        self.bkd = bkd = case.bkd
        self.f, self.pinned = pinned_quadratic(case)
        self.system = SteadySystem(
            self.pinned, self.f.states, RecomputeAlways(case.direct_solver())
        )
        self.term = pinned_rows_residual(bkd, self.pinned, _unit_columns(case))
        functional: StateFunctional[A, A] = StateFunctional.from_terms(
            bkd, 2, state_terms=(self.term,)
        )
        self.derivs = SteadyDerivatives(self.pinned, self.f.states, functional)

    def param(self) -> A:
        return self.bkd.reshape(self.bkd.asarray(QUADRATIC_PARAM), (2, 1))

    def solve(self, p: A) -> SteadyPoint[A, A]:
        self.f.set_param(p[:, 0])
        newton: Newton[A, A] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        point = self.system.solve(newton, self.f.states.zeros(1))
        assert point.converged()
        return point

    def weights(self) -> A:
        return self.bkd.asarray(_WEIGHTS)


class TheValueSuite:
    def test_reads_the_unpinned_residual(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        bkd = array_case.bkd
        solved = _Solved(array_case)
        point = solved.solve(solved.param())
        u = solved.system.space().copy(point.solution())
        raw = solved.f(u)
        bkd.assert_allclose(solved.term.value(u), raw[_PINNED], rtol=1e-14)
        bkd.assert_allclose(raw[1:2], bkd.zeros((1, 1)), atol=1e-12)
        assert solved.term.nqoi() == 2

    def test_reads_the_solves_parameters(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        bkd = array_case.bkd
        solved = _Solved(array_case)
        u = solved.system.space().copy(solved.solve(solved.param()).solution())
        before = solved.term.value(u)
        solved.f.set_param(bkd.asarray([0.2, 0.9]))
        bkd.assert_allclose(solved.term.value(u), solved.f(u)[_PINNED], rtol=1e-14)
        assert bkd.to_float(bkd.max(bkd.abs(solved.term.value(u) - before))) > 0.1


class TheDerivativesSuite:
    def test_gradient_and_hvp(self, array_case: TypedStackCase[A, F, A]) -> None:
        bkd = array_case.bkd
        solved = _Solved(array_case)
        derivs, weights = solved.derivs, solved.weights()
        hessian = derivs.hessian()
        assert hessian is not None

        def value(p: A) -> A:
            return weights.T @ derivs.value(solved.solve(p))

        def gradient(p: A) -> A:
            return derivs.gradient(derivs.adjoint(solved.solve(p), weights))

        def gradient_row(p: A) -> A:
            return gradient(p).T

        def hessian_matrix(p: A) -> A:
            adjoint = derivs.adjoint(solved.solve(p), weights)
            eye = bkd.eye(2)
            return bkd.hstack(
                [hessian.hvp(adjoint, eye[:, j : j + 1]) for j in range(2)]
            )

        p = solved.param()
        assert min_fd_error(array_case, 1, 2, value, gradient_row, p) < 1e-6
        assert min_fd_error(array_case, 2, 2, gradient, hessian_matrix, p) < 1e-6

    def test_tangent(self, array_case: TypedStackCase[A, F, A]) -> None:
        bkd = array_case.bkd
        solved = _Solved(array_case)
        derivs = solved.derivs

        def value(p: A) -> A:
            return derivs.value(solved.solve(p))

        def jacobian(p: A) -> A:
            point = solved.solve(p)
            eye = bkd.eye(2)
            return bkd.hstack(
                [derivs.tangent(point, eye[:, j : j + 1]) for j in range(2)]
            )

        assert min_fd_error(array_case, 2, 2, value, jacobian, solved.param()) < 1e-6


class TheBundleSuite:
    def test_follows_the_residual(self, array_case: TypedStackCase[A, F, A]) -> None:
        bkd = array_case.bkd
        entries = StateEntries(bkd, 3, _PINNED)
        f0: QuadraticResidual[A, F, A] = QuadraticResidual(array_case, order=0)
        f1: QuadraticResidual[A, F, A] = QuadraticResidual(array_case, order=1)
        f2: QuadraticResidual[A, F, A] = QuadraticResidual(array_case, order=2)
        none = ObservedResidual(f0, entries, f0.states)
        assert none.derivatives().wrt_state is None
        first = ObservedResidual(f1, entries, f1.states)
        assert first.derivatives().wrt_state is not None
        assert first.derivatives().wrt_param is not None
        assert first.derivatives().state_state is None
        second = ObservedResidual(f2, entries, f2.states)
        assert second.derivatives().state_state is not None

    def test_a_stated_zero_block_is_none(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        bkd = array_case.bkd
        # The affine quadratic declares its state curvature zero, so the
        # term's state-state block is skipped, not applied.
        f: QuadraticResidual[A, F, A] = QuadraticResidual(
            array_case, order=2, affine=True
        )
        term = ObservedResidual(f, StateEntries(bkd, 3, _PINNED), f.states)
        state_state = term.derivatives().state_state
        assert state_state is not None
        u, w = bkd.ones((3, 1)), bkd.ones((3, 1))
        assert state_state(u, bkd.asarray(_WEIGHTS), w) is None


class RefusalsSuite:
    def test_protocols(self, array_case: TypedStackCase[A, F, A]) -> None:
        f: QuadraticResidual[A, F, A] = QuadraticResidual(array_case)
        entries = StateEntries(array_case.bkd, 3, [0])
        with pytest.raises(TypeError, match="ResidualProtocol"):
            ObservedResidual(object(), entries, f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="StateObservationProtocol"):
            ObservedResidual(f, object(), f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="VectorSpaceProtocol"):
            ObservedResidual(f, entries, object())  # type: ignore[arg-type]

    def test_an_observation_of_another_space(
        self, array_case: TypedStackCase[A, F, A]
    ) -> None:
        f: QuadraticResidual[A, F, A] = QuadraticResidual(array_case)
        with pytest.raises(ValueError, match="observation"):
            ObservedResidual(f, StateEntries(array_case.bkd, 4, [0]), f.states)

    def test_weights(self, array_case: TypedStackCase[A, F, A]) -> None:
        bkd = array_case.bkd
        _, pinned = pinned_quadratic(array_case)
        rows = pinned.pin().rows()
        with pytest.raises(ValueError, match="shape"):
            RowWeights(bkd, rows, bkd.ones((3,)))
        with pytest.raises(ValueError, match="shape"):
            RowWeights(bkd, rows, bkd.ones((4, 1)))
        # Row 1 is free: F vanishes there, so reading it is an error.
        with pytest.raises(ValueError, match="vanish off the rows"):
            pinned_rows_residual(bkd, pinned, bkd.asarray([[1.0], [0.5], [0.0]]))
        with pytest.raises(TypeError, match="PinnedResidual"):
            pinned_rows_residual(
                bkd,
                QuadraticResidual(array_case),  # type: ignore[arg-type]
                _unit_columns(array_case),
            )
