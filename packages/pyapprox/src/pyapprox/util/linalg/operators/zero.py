"""The zero operator between two spaces of a stack.

``ZeroOperator`` is the map :math:`A = 0` from one vector space to
another (they may differ in size and layout). It declares ``zero`` in its
properties, so a consumer may skip it instead of applying it, and it
returns fresh zero blocks from the stack's own spaces, so it works on
every stack.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.formats import MatrixFree
from pyapprox.util.linalg.operators.protocols import (
    OperatorProperties,
    VectorSpaceProtocol,
)

V = TypeVar("V")


class ZeroOperator(Generic[V]):
    r""":math:`A = 0` from ``domain`` to ``range``.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.ActionOperatorProtocol` in the
    ``MatrixFree`` format. ``out`` is ignored: a fresh zero block is
    returned.

    Parameters
    ----------
    domain : VectorSpaceProtocol
        The space of the input of ``apply``.
    range : VectorSpaceProtocol
        The space of its output. Defaults to ``domain``.
    """

    def __init__(
        self,
        domain: VectorSpaceProtocol[V],
        range: Optional[VectorSpaceProtocol[V]] = None,
    ) -> None:
        self._domain = domain
        self._range = domain if range is None else range

    def domain(self) -> SpaceDescriptor:
        return self._domain.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._range.descriptor()

    def properties(self) -> OperatorProperties:
        return OperatorProperties(zero=True)

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        """A zero block of the range with as many columns as ``x``."""
        return self._range.zeros(self._domain.ncols(x))

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        """A zero block of the domain with as many columns as ``y``."""
        return self._domain.zeros(self._range.ncols(y))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.domain()!r} -> {self.range()!r})"
