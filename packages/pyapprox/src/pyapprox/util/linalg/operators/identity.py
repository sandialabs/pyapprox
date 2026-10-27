"""The identity operator on a space of a stack.

``IdentityOperator`` is the map :math:`A = I` on one vector space. It
declares ``identity`` in its properties, so a consumer may skip it instead
of applying it, and it copies through the space's own ``copy``, so it
works on every stack without storing a matrix.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.formats import MatrixFree
from pyapprox.util.linalg.operators.protocols import (
    OperatorProperties,
    VectorSpaceProtocol,
)

V = TypeVar("V")


class IdentityOperator(Generic[V]):
    r""":math:`A = I` on ``space``.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.ActionOperatorProtocol` in the
    ``MatrixFree`` format. ``apply`` returns a copy of its input (into
    ``out`` when given), never the input itself, so a caller may modify the
    result.

    Parameters
    ----------
    space : VectorSpaceProtocol
        The space of both the input and the output.
    """

    def __init__(self, space: VectorSpaceProtocol[V]) -> None:
        self._space = space

    def domain(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._space.descriptor()

    def properties(self) -> OperatorProperties:
        return OperatorProperties(
            symmetric=True, spd=True, diagonal=True, identity=True
        )

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        """A copy of ``x``."""
        return self._space.copy(x, out=out)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        """A copy of ``y``."""
        return self._space.copy(y, out=out)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.domain()!r})"
