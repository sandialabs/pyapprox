"""An operator known only by its forward action.

For tests of the transpose capability (``OperatorProperties.transposable``):
a matrix-free operator whose library gives no transposed action, on any
stack.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.util.linalg.operators import (
    LinearMapProtocol,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
)

V = TypeVar("V")


class ForwardOnly(Generic[V]):
    """``op``'s forward action, with no transpose declared or offered."""

    def __init__(self, op: LinearMapProtocol[V]) -> None:
        self._op = op

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return OperatorProperties(transposable=False)

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._op.apply(x, out=out)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        raise TypeError("ForwardOnly has no transpose")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._op!r})"
