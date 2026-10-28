r"""Newton's method on any stack.

On the backend stack (both backends): the elementwise cubic
:math:`x^3 + x = b`, :math:`b = (2, 10, 30)`, solved from zero to the
exact roots :math:`(1, 2, 3)` with the residual norm under the tolerance
and quadratic convergence (each residual norm at most a constant times
the square of the previous one once close); :math:`\arctan x = 0` from
:math:`x_0 = 3`, where full steps diverge and the backtracking line
search converges to 0; a system declaring itself linear solved in one
step. Failures are reported, never raised: a step budget exhausted, a
residual that stops being finite, a failed linear solve. On the mock
stack, whose vectors cannot be looked inside, a linear system solved as
nonlinear converges in one step, so the solver uses only the space and
the handles.
"""

from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendVectorSpace,
)
from pyapprox.util.linalg.operators import (
    InnerProductSpaceProtocol,
    SolveInfo,
    SolverHandleBase,
    SolverHandleProtocol,
    SpaceDescriptor,
)
from pyapprox.util.rootfinding.nonlinear import (
    Newton,
    NonlinearSolverProtocol,
    NonlinearSystemProtocol,
)
from tests._helpers.linalg.mock_stack import (
    MockLU,
    MockOperator,
    MockVector,
    MockVectorSpace,
    peek,
    wrap,
)

_X = SpaceDescriptor("x", 3)
_SCALAR = SpaceDescriptor("x", 1)


class _Elementwise(Generic[Array]):
    r""":math:`R(x) = g(x)` entry by entry, with a diagonal Jacobian.

    ``kind`` is ``"cubic"`` (:math:`x^3 + x - b`) or ``"arctan"``
    (:math:`\arctan x`).
    """

    def __init__(
        self, bkd: Backend[Array], descriptor: SpaceDescriptor, kind: str, rhs: Array
    ) -> None:
        self._bkd, self._kind, self._rhs = bkd, kind, rhs
        self._space = BackendVectorSpace(descriptor, bkd)
        self._descriptor = descriptor

    def space(self) -> BackendVectorSpace[Array]:
        return self._space

    def residual(self, x: Array) -> Array:
        if self._kind == "cubic":
            return x**3 + x - self._rhs
        return self._bkd.arctan(x)

    def _derivative(self, x: Array) -> Array:
        if self._kind == "cubic":
            return 3.0 * x**2 + 1.0
        return 1.0 / (1.0 + x**2)

    def jacobian_handle(self, x: Array) -> SolverHandleProtocol[Array]:
        matrix = self._bkd.diag(self._derivative(x)[:, 0])
        operator = BackendDenseOperator(matrix, self._descriptor, self._bkd)
        return BackendDenseLU(self._bkd).setup(operator)

    def is_linear(self) -> bool:
        return False


class _Linear(Generic[Array]):
    """:math:`R(x) = A x - b`, declared linear or not."""

    def __init__(self, bkd: Backend[Array], declared_linear: bool) -> None:
        self._bkd = bkd
        self._space = BackendVectorSpace(_X, bkd)
        self._matrix = bkd.asarray([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        self._rhs = bkd.asarray([[1.0], [2.0], [3.0]])
        self._declared = declared_linear

    def space(self) -> BackendVectorSpace[Array]:
        return self._space

    def residual(self, x: Array) -> Array:
        return self._bkd.dot(self._matrix, x) - self._rhs

    def jacobian_handle(self, x: Array) -> SolverHandleProtocol[Array]:
        return BackendDenseLU(self._bkd).setup(
            BackendDenseOperator(self._matrix, _X, self._bkd)
        )

    def is_linear(self) -> bool:
        return self._declared

    def solution(self) -> Array:
        return self._bkd.solve(self._matrix, self._rhs)


class _NotFinite(_Linear[Array]):
    """A residual that is NaN everywhere after the first evaluation."""

    def __init__(self, bkd: Backend[Array]) -> None:
        super().__init__(bkd, declared_linear=False)
        self._calls = 0

    def residual(self, x: Array) -> Array:
        self._calls += 1
        value = super().residual(x)
        return value if self._calls == 1 else value * float("nan")


class _FailingHandle(SolverHandleBase[Array]):
    def solve(
        self, rhs: Array, tol: float | None = None, x0: Array | None = None
    ) -> tuple[Array, SolveInfo]:
        return rhs, SolveInfo(False, message="planted failure")

    def transpose(self) -> None:
        return None


class _FailingLinearSolve(_Linear[Array]):
    """The Jacobian handle never converges."""

    def jacobian_handle(self, x: Array) -> SolverHandleProtocol[Array]:
        return _FailingHandle(BackendDenseOperator(self._matrix, _X, self._bkd))


def _norm(
    bkd: Backend[Array], space: InnerProductSpaceProtocol[Array, Array], x: Array
) -> float:
    return bkd.to_float(space.norm(x)[0, 0])


def _newton(
    bkd: Backend[Array],
    atol: float = 1e-7,
    rtol: float = 1e-7,
    maxiters: int = 10,
    line_search: bool = False,
) -> Newton[Array, Array]:
    """Newton on the backend stack: vectors and reductions are both
    ``bkd``'s arrays, which a bare ``Newton()`` cannot infer."""
    return Newton(atol, rtol, maxiters, line_search=line_search)


class TestNewton:
    def test_satisfies_the_protocols(self, bkd: Backend[Array]) -> None:
        assert isinstance(_newton(bkd), NonlinearSolverProtocol)
        assert isinstance(_Linear(bkd, False), NonlinearSystemProtocol)

    def test_cubic_converges_quadratically_to_the_roots(
        self, bkd: Backend[Array]
    ) -> None:
        system = _Elementwise(bkd, _X, "cubic", bkd.asarray([[2.0], [10.0], [30.0]]))
        solver = _newton(bkd, atol=1e-13, rtol=0.0, maxiters=30)
        solution, info = solver.solve(system, bkd.zeros((3, 1)))
        assert info.converged and info.residual_norm is not None
        bkd.assert_allclose(solution, bkd.asarray([[1.0], [2.0], [3.0]]), rtol=1e-12)
        # Quadratic convergence: rerun with growing budgets and check the
        # residual norms' ratio r_{k+1} / r_k^2 stays bounded near the end.
        norms = []
        for budget in range(1, info.iterations + 1):
            iterate, _ = _newton(bkd, atol=0.0, rtol=0.0, maxiters=budget).solve(
                system, bkd.zeros((3, 1))
            )
            norms.append(_norm(bkd, system.space(), system.residual(iterate)))
        tail = [n for n in norms if 1e-12 < n < 1e-1]
        ratios = [b / a**2 for a, b in zip(tail, tail[1:])]
        assert ratios and max(ratios) < 10.0

    def test_line_search_converges_where_full_steps_diverge(
        self, bkd: Backend[Array]
    ) -> None:
        system = _Elementwise(bkd, _SCALAR, "arctan", bkd.zeros((1, 1)))
        start = bkd.asarray([[3.0]])
        _, diverged = _newton(bkd, atol=1e-12, maxiters=10).solve(system, start)
        assert not diverged.converged
        solution, info = _newton(
            bkd, atol=1e-12, rtol=0.0, maxiters=50, line_search=True
        ).solve(system, start)
        assert info.converged
        # Converged means |arctan x| <= 1e-12, so |x| is about 1e-12 at
        # most; a decade of margin keeps the check off that boundary.
        assert abs(bkd.to_float(solution[0, 0])) < 1e-11

    def test_a_linear_system_takes_one_exact_step(self, bkd: Backend[Array]) -> None:
        system = _Linear(bkd, declared_linear=True)
        solution, info = _newton(bkd).solve(system, bkd.zeros((3, 1)))
        assert info.converged and info.iterations == 1
        assert info.residual_norm is None
        bkd.assert_allclose(solution, system.solution(), rtol=1e-12)

    def test_the_initial_guess_is_not_modified(self, bkd: Backend[Array]) -> None:
        guess = bkd.zeros((3, 1))
        _newton(bkd).solve(_Linear(bkd, declared_linear=False), guess)
        bkd.assert_allclose(guess, bkd.zeros((3, 1)), rtol=0.0)


class TestReportedFailures:
    def test_step_budget_exhausted(self, bkd: Backend[Array]) -> None:
        system = _Elementwise(bkd, _X, "cubic", bkd.asarray([[2.0], [10.0], [30.0]]))
        _, info = _newton(bkd, atol=1e-14, rtol=0.0, maxiters=2).solve(
            system, bkd.zeros((3, 1))
        )
        assert not info.converged and info.iterations == 2
        assert "2 steps reached" in info.message

    def test_a_residual_that_stops_being_finite(self, bkd: Backend[Array]) -> None:
        _, info = _newton(bkd).solve(_NotFinite(bkd), bkd.zeros((3, 1)))
        assert not info.converged
        assert info.message == "the residual is no longer finite"

    def test_a_failed_linear_solve(self, bkd: Backend[Array]) -> None:
        for declared in (False, True):
            _, info = _newton(bkd).solve(
                _FailingLinearSolve(bkd, declared), bkd.zeros((3, 1))
            )
            assert not info.converged
            assert info.message == "linear solve failed: planted failure"


class _MockLinear:
    """:math:`R(x) = A x - b` on the mock stack, solved as nonlinear."""

    def __init__(self) -> None:
        self._matrix: NDArray[np.float64] = np.array(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        )
        self._operator = MockOperator.from_dense(self._matrix, _X)
        self._space = MockVectorSpace(_X)
        self._rhs = wrap(np.array([[1.0], [2.0], [3.0]]), _X)

    def space(self) -> MockVectorSpace:
        return self._space

    def residual(self, x: MockVector) -> MockVector:
        return self._space.axpby(1.0, self._operator.apply(x), -1.0, self._rhs)

    def jacobian_handle(self, x: MockVector) -> SolverHandleProtocol[MockVector]:
        return MockLU().setup(self._operator)

    def is_linear(self) -> bool:
        return False

    def solution(self) -> NDArray[np.floating[Any]]:
        return np.linalg.solve(self._matrix, peek(self._rhs))


class TestOnAnOpaqueStack:
    def test_newton_uses_only_the_space_and_the_handles(self) -> None:
        system = _MockLinear()
        # The mock stack's reductions are NumPy arrays of any float dtype.
        newton: Newton[MockVector, NDArray[Any]] = Newton(atol=1e-12, rtol=0.0)
        solution, info = newton.solve(system, system.space().zeros())
        assert info.converged and info.iterations == 1
        np.testing.assert_allclose(peek(solution), system.solution(), rtol=1e-12)
