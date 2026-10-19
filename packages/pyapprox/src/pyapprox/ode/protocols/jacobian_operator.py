r"""Protocol for the state Jacobian of an ODE right-hand side as an operator.

For :math:`M \, dy/dt = f(y, t)`, a producer of this protocol returns
:math:`\partial f/\partial y` at ``(state, time)`` as an operator of
some linear-algebra stack, labeled with the linearization point the
caller minted. A time stepper that consumes it needs only the operator's
action, its transpose and its format, never a matrix, so any
discretization whose producer satisfies the protocol runs through the
stepper unchanged.
"""

from typing import Protocol, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array_contra
from pyapprox.util.linalg.operators import (
    JacobianSnapshotProtocol,
    LinearizationPoint,
)

V = TypeVar("V")
F_co = TypeVar("F_co", covariant=True)


@runtime_checkable
class JacobianOperatorProtocol(Protocol[Array_contra, V, F_co]):
    r"""Supplies :math:`\partial f/\partial y` as a labeled operator.

    ``Array_contra`` is the backend array type of states, ``V`` the
    stack's vector type and ``F_co`` its storage format.
    """

    def jacobian_operator(
        self, state: Array_contra, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[V, F_co]:
        """The Jacobian at ``(state, time)``, labeled with ``point`` and
        independent of later calls.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Time t.
        point : LinearizationPoint
            The identity of ``(state, time)``, minted by the caller.
        """
        ...
