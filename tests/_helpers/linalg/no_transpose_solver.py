"""A dense solver whose handles offer no transposed solve.

For tests of consumers that must refuse a solver without transposes
when they need one (an adjoint's transposed solves).
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    SolveInfo,
    SolverCapabilities,
    SolverHandleBase,
    SolverHandleProtocol,
)


class NoTransposeHandle(SolverHandleBase[Array]):
    """Solves through another handle but offers no transposed solve."""

    def __init__(self, inner: SolverHandleProtocol[Array]) -> None:
        super().__init__(inner)
        self._inner = inner

    def solve(
        self, rhs: Array, tol: Optional[float] = None, x0: Optional[Array] = None
    ) -> tuple[Array, SolveInfo]:
        return self._inner.solve(rhs, tol, x0)

    def transpose(self) -> None:
        return None


class NoTransposeSolver(Generic[Array]):
    """A dense LU whose handles have no transposed solve."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._lu = BackendDenseLU(bkd)

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=False)

    def setup(
        self, op: ActionOperatorProtocol[Array, DenseFormat[Array]]
    ) -> SolverHandleProtocol[Array]:
        return NoTransposeHandle(self._lu.setup(op))
