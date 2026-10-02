"""Protocol for the feasible set of relaxed design weights.

A design space owns everything a relaxed solver needs to know about the
weights it searches over: their bounds, any further constraints, the budget
they share, and a feasible starting point. Solvers read these from the
design space rather than building them, so the budget lives in one place.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class DesignSpaceProtocol(Protocol, Generic[Array]):
    """Feasible set of relaxed design weights.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of design weights.
    bounds()
        Lower and upper bound of each weight.
    constraints()
        Constraints beyond the bounds.
    budget()
        Total weight the design may spend.
    initial()
        A feasible starting point.
    """

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        ...

    def nvars(self) -> int:
        """Number of design weights."""
        ...

    def bounds(self) -> Array:
        """Lower and upper bound of each weight.

        Returns
        -------
        Array
            Each row is ``[lower, upper]``. Shape: (nvars, 2)
        """
        ...

    def constraints(self) -> SequenceOfConstraintProtocols[Array]:
        """Constraints on the weights beyond the bounds."""
        ...

    def budget(self) -> float:
        """Total weight the design may spend."""
        ...

    def initial(self) -> Array:
        """A feasible starting point.

        Returns
        -------
        Array
            Design weights satisfying the bounds and constraints.
            Shape: (nvars, 1)
        """
        ...
