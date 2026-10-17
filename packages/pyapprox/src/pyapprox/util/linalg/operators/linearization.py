r"""Linearizations: operators that are derivatives at a recorded point.

The derivative :math:`J = \partial F/\partial u` of a nonlinear map
:math:`F(u; p)` is an operator that depends on the point :math:`u` and
the parameters :math:`p`. A *linearization* is that operator together
with a record of where it was taken, so that anything reusing work done
with it (a factorization, a preconditioner) can tell whether a later
derivative is the same operator.

The record, :class:`LinearizationPoint`, is an immutable value, and so
is the operator: a linearization never changes after it is made, and a
derivative at another point is a new linearization. Nothing here knows
what a point is; whoever computes the derivative identifies the point
with any hashable value of its own.
"""

from dataclasses import dataclass
from typing import Generic, Hashable, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    OperatorProperties,
)

V = TypeVar("V")
F_co = TypeVar("F_co", covariant=True)


@dataclass(frozen=True)
class LinearizationPoint:
    """Where a derivative was taken.

    Two linearizations of one map with equal points are the same
    operator. Hashable, so it can key a cache.

    Parameters
    ----------
    parameters_version : int
        A counter the map's owner increments whenever its parameters
        change.
    point : hashable or None
        An identity of the point, chosen by whoever computes the
        derivative. ``None`` declares that the derivative does not
        depend on the point (the map is linear in it), so every point
        gives the same operator.
    provisional : bool
        Whether the point is an intermediate iterate of an outer
        iterative solve rather than its result. Work done at a
        provisional point is valid only at that point, so it must never
        be reused for the result, even when the identities agree.

    Raises
    ------
    TypeError
        If ``point`` is not hashable.
    ValueError
        If ``parameters_version`` is negative, or ``provisional`` is set
        on a point-independent derivative.
    """

    parameters_version: int
    point: Optional[Hashable] = None
    provisional: bool = False

    def __post_init__(self) -> None:
        if self.parameters_version < 0:
            raise ValueError(
                f"parameters_version must be nonnegative, got {self.parameters_version}"
            )
        try:
            hash(self.point)
        except TypeError as error:
            raise TypeError(
                f"point must be hashable, got {type(self.point).__name__}"
            ) from error
        if self.point is None and self.provisional:
            raise ValueError(
                "a point-independent derivative (point=None) cannot be provisional"
            )


@runtime_checkable
class JacobianSnapshotProtocol(ActionOperatorProtocol[V, F_co], Protocol[V, F_co]):
    """An operator that is a derivative taken at a recorded point."""

    def linearization(self) -> LinearizationPoint:
        """Where the derivative was taken."""
        ...


class LinearizedOperator(Generic[V, F_co]):
    """An operator labeled with the point it was linearized at.

    Delegates everything to the wrapped operator, which must not change
    afterwards: a linearization is a value of one point.

    Parameters
    ----------
    op : ActionOperatorProtocol
        The derivative.
    point : LinearizationPoint
        Where it was taken.
    """

    def __init__(
        self, op: ActionOperatorProtocol[V, F_co], point: LinearizationPoint
    ) -> None:
        self._op = op
        self._point = point

    def linearization(self) -> LinearizationPoint:
        return self._point

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return self._op.properties()

    def format(self) -> F_co:
        return self._op.format()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._op.apply(x, out=out)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        return self._op.apply_transpose(y, out=out)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._op!r}, {self._point!r})"
