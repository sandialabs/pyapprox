r"""Protocol for the mass an explicit time step solves with.

An explicit step of :math:`M \dot y = f(y, t)` evaluates :math:`f` at
known states and solves only with :math:`M`, so it needs the mass as an
operator and a solver for it. Every semi-explicit system has both.
"""

from typing import Protocol, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import ActionOperatorProtocol, LinearSolverProtocol

F = TypeVar("F")


@runtime_checkable
class MassSystemProtocol(Protocol[Array, F]):
    """A mass operator of format ``F`` and a solver paired with it.

    ``Array`` is the backend array type the operator acts on, as blocks
    of shape ``(nstates, ncols)``.
    """

    def mass(self) -> ActionOperatorProtocol[Array, F]:
        """:math:`M`."""
        ...

    def solver(self) -> LinearSolverProtocol[Array, F]:
        """A solver for :math:`M`'s format."""
        ...
