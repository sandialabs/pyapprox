"""Moments of a fitted hierarchical sparse grid surrogate.

The hierarchical surrogate stores a surplus per grid point together with
that point's hierarchical quadrature weight, so the mean is their inner
product and needs no combination over subspaces. Kept here rather than
on the surrogate for the same reason as the combination moments: the
surrogate evaluates, and what is computed from it lives alongside the
other statistics.
"""

from typing import Generic, Optional

from pyapprox.surrogates.sparsegrids.hierarchical.hierarchical_surrogate import (
    HierarchicalSurrogate,
)
from pyapprox.util.backends.protocols import Array


class HierarchicalMoments(Generic[Array]):
    """Mean of a hierarchical surrogate under its own quadrature rule.

    mean = sum_{l,j} v_{l,j} w_{l,j}, over the surpluses and the
    hierarchical weights the surrogate already holds.

    No variance is offered. The hierarchical basis is not orthonormal,
    so a variance would need either the second moment of the expansion
    or a conversion, neither of which the surrogate carries.

    Parameters
    ----------
    surrogate : HierarchicalSurrogate[Array]
        Fitted surrogate.

    Raises
    ------
    TypeError
        If surrogate is not a HierarchicalSurrogate.
    """

    def __init__(self, surrogate: HierarchicalSurrogate[Array]) -> None:
        if not isinstance(surrogate, HierarchicalSurrogate):
            raise TypeError(
                "surrogate must be a HierarchicalSurrogate, got "
                f"{type(surrogate).__name__}"
            )
        self._surrogate = surrogate
        self._mean: Optional[Array] = None

    def surrogate(self) -> HierarchicalSurrogate[Array]:
        """Return the surrogate these moments describe."""
        return self._surrogate

    def mean(self) -> Array:
        """Return the hierarchical quadrature mean, shape (nqoi,)."""
        if self._mean is None:
            bkd = self._surrogate.bkd()
            # surpluses: (nqoi, npoints), quad_weights: (npoints,)
            self._mean = bkd.dot(
                self._surrogate.surpluses(),
                self._surrogate.quad_weights(),
            )
        return self._mean

    def __repr__(self) -> str:
        return f"HierarchicalMoments(nqoi={self._surrogate.nqoi()})"
