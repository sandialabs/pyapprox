"""Constraint protocol built on the Derivatives bundle.

The sibling of :class:`~pyapprox.interface.functions.protocols.objective.
ObjectiveProtocol` — see that module's docstring for the deliberate
objective/constraint differences (bounds, vector nqoi, ``resolved_whvp``
access with the OWNING object's ``nqoi()``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class NonlinearConstraintProtocol(FunctionProtocol[Array], Protocol):
    """A vector-valued constraint with bounds and a derivative bundle."""

    def lb(self) -> Array:
        """Lower bounds. Shape ``(nqoi,)`` (existing constraint convention)."""
        ...

    def ub(self) -> Array:
        """Upper bounds. Shape ``(nqoi,)`` (existing constraint convention)."""
        ...

    def derivatives(self) -> Derivatives[Array]:
        """Return this constraint's derivative capabilities."""
        ...
