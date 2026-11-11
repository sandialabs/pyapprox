"""A linear solver that counts its setups, for checking setup reuse."""

from typing import Generic, TypeVar

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    LinearSolverProtocol,
    SolverHandleProtocol,
)

V = TypeVar("V")
F = TypeVar("F")


class CountingSolver(Generic[V, F]):
    """``inner`` with a count of its setups (factorizations, say)."""

    def __init__(self, inner: LinearSolverProtocol[V, F]) -> None:
        self._inner = inner
        self.setups = 0

    def setup(self, op: ActionOperatorProtocol[V, F]) -> SolverHandleProtocol[V]:
        self.setups += 1
        return self._inner.setup(op)
