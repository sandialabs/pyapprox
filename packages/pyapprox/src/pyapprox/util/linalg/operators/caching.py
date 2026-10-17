r"""Reuse of solver setups across equal linearizations.

Setting up a solver (a factorization, a preconditioner) is often the
dominant cost of a solve, and repeated solves frequently use the same
operator: one derivative serves several right-hand sides, or a
derivative independent of the point serves every point. A
:class:`CachingSolver` reuses a handle exactly when the operator is
provably the same, as recorded by its :class:`~.linearization.LinearizationPoint`
and its spaces; anything it cannot prove equal is set up afresh.

Reuse rests on the producer's labels being truthful: two operators
labeled with equal points must be equal. A label that is wrong makes the
cache return a handle for a different operator, so a producer that
cannot tell whether its derivative changed must label it with a new
point.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.linearization import (
    JacobianSnapshotProtocol,
    LinearizationPoint,
)
from pyapprox.util.linalg.operators.protocols import ActionOperatorProtocol
from pyapprox.util.linalg.operators.solvers import (
    LinearSolverProtocol,
    SolverCapabilities,
    SolverHandleProtocol,
)

V = TypeVar("V")
F_contra = TypeVar("F_contra", contravariant=True)


@dataclass(frozen=True)
class ReuseKey:
    """What must agree for a solver setup to be reused.

    Parameters
    ----------
    point : LinearizationPoint
        Where the operator was linearized. Its ``provisional`` flag takes
        part in equality, so work done at an intermediate iterate is
        never reused for a final point, even under the same identity.
    domain, range : SpaceDescriptor
        The operator's spaces, versions included, so a rebuilt space
        never reuses a setup made for the old one.
    """

    point: LinearizationPoint
    domain: SpaceDescriptor
    range: SpaceDescriptor


def reuse_key(op: ActionOperatorProtocol[Any, Any]) -> Optional[ReuseKey]:
    """The key of ``op``, or ``None`` when ``op`` records no linearization
    point (it is then never reused)."""
    if not isinstance(op, JacobianSnapshotProtocol):
        return None
    return ReuseKey(op.linearization(), op.domain(), op.range())


class CachingSolver(Generic[V, F_contra]):
    """A solver that reuses handles of an inner solver for equal keys.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.solvers.LinearSolverProtocol`
    with the inner solver's format, so it can be injected wherever the
    inner solver can. Operators without a linearization point are set up
    afresh every time.

    Parameters
    ----------
    inner : LinearSolverProtocol
        The solver whose setups are cached.
    capacity : int
        How many handles are kept; the least recently used is evicted
        first. Each kept handle holds its setup (e.g. a factorization)
        in memory.

    Raises
    ------
    ValueError
        If ``capacity`` is less than 1.
    """

    def __init__(self, inner: LinearSolverProtocol[V, F_contra], capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._inner = inner
        self._capacity = capacity
        self._handles: OrderedDict[ReuseKey, SolverHandleProtocol[V]] = OrderedDict()
        self._setups = 0
        self._hits = 0

    def capabilities(self) -> SolverCapabilities:
        return self._inner.capabilities()

    def setup(self, op: ActionOperatorProtocol[V, F_contra]) -> SolverHandleProtocol[V]:
        key = reuse_key(op)
        if key is not None and key in self._handles:
            self._handles.move_to_end(key)
            self._hits += 1
            return self._handles[key]
        handle = self._inner.setup(op)
        self._setups += 1
        if key is not None:
            self._handles[key] = handle
            if len(self._handles) > self._capacity:
                self._handles.popitem(last=False)
        return handle

    def setups(self) -> int:
        """How many times the inner solver has been set up."""
        return self._setups

    def hits(self) -> int:
        """How many setups were answered from the cache."""
        return self._hits

    def clear(self) -> None:
        """Drop every kept handle (the counts are kept)."""
        self._handles.clear()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self._inner!r}, capacity={self._capacity}, "
            f"kept={len(self._handles)})"
        )
