"""An essential condition with some of its DOFs left free.

A DOF an essential condition would constrain is sometimes decided
elsewhere: a subdomain's port node on a Dirichlet edge, whose value an
outer problem supplies as data. :class:`ExceptDofs` wraps the condition
and drops those DOFs from everything it reports, so the condition stays
one object and its class gains no option.
"""

from typing import Callable, Generic, Optional, Sequence

from pyapprox.pde.boundary.protocols import EssentialBCProtocol
from pyapprox.util.backends.protocols import Array, Backend


class _Kept(Generic[Array]):
    """``fn``'s values at the kept positions; a class rather than a
    closure, so a condition holding it pickles."""

    def __init__(self, fn: Callable[[float], Array], kept: Array) -> None:
        self._fn, self._kept = fn, kept

    def __call__(self, time: float) -> Array:
        return self._fn(time)[self._kept]


class ExceptDofs(Generic[Array]):
    """``bc`` with the DOFs ``dofs`` no longer constrained.

    Satisfies :class:`~pyapprox.pde.boundary.protocols.EssentialBCProtocol`.

    Parameters
    ----------
    bc
        The essential condition.
    dofs
        Global DOF indices to leave free, each constrained by ``bc``.

    Raises
    ------
    TypeError
        If ``bc`` does not satisfy ``EssentialBCProtocol``.
    ValueError
        If a DOF in ``dofs`` is not constrained by ``bc`` or is repeated.
    """

    def __init__(self, bc: EssentialBCProtocol[Array], dofs: Sequence[int]) -> None:
        if not isinstance(bc, EssentialBCProtocol):
            raise TypeError(
                f"bc must satisfy EssentialBCProtocol, got {type(bc).__name__}"
            )
        bkd = bc.bkd()
        constrained = bc.constrained_dofs()
        # Fixed structure, read once: which positions stay constrained.
        position = {
            bkd.to_int(constrained[k]): k for k in range(constrained.shape[0])
        }
        exempt = set(dofs)
        if len(exempt) != len(dofs):
            raise ValueError(f"dofs has repeated entries: {list(dofs)}")
        missing = sorted(exempt - position.keys())
        if missing:
            raise ValueError(f"{bc!r} does not constrain DOFs {missing}")
        kept = [k for d, k in position.items() if d not in exempt]
        self._bc = bc
        self._kept = bkd.asarray(sorted(kept), dtype=bkd.int64_dtype())
        self._dofs = constrained[self._kept]
        self._nexempt = len(exempt)

    def bkd(self) -> Backend[Array]:
        return self._bc.bkd()

    def constrained_dofs(self) -> Array:
        return self._dofs

    def constrained_values(self, time: float) -> Array:
        return self._bc.constrained_values(time)[self._kept]

    def is_time_invariant(self) -> bool:
        return self._bc.is_time_invariant()

    def constrained_values_derivative(
        self, order: int
    ) -> Optional[Callable[[float], Array]]:
        derivative = self._bc.constrained_values_derivative(order)
        if derivative is None:
            return None
        return _Kept(derivative, self._kept)

    def __repr__(self) -> str:
        return f"ExceptDofs({self._bc!r}, nexempt={self._nexempt})"
