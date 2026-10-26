r"""Protocol for the system an implicit time step solves.

An implicit one-step method writes each stage as an approximation of
:math:`\dot y` built from the unknown state with a known shift
:math:`\sigma`: Backward Euler uses :math:`\dot y = (y - y_n)/\Delta t`
with :math:`\sigma = 1/\Delta t`. The step then solves
:math:`F(t, y, \dot y) = 0` by Newton's method, whose linear systems are
the shifted Jacobian
:math:`\sigma\,\partial F/\partial \dot y + \partial F/\partial y`.

Stating the step in this residual form keeps the stepper independent of
how :math:`F` is built. A mass-matrix system :math:`M \dot y = f(y, t)`,
including a differential-algebraic one with a singular mass, has
:math:`F = M \dot y - f` and shifted Jacobian :math:`\sigma M - J`; a fully
implicit model (a state-dependent mass, or :math:`F` nonlinear in
:math:`\dot y`) supplies its two derivatives directly. Both satisfy this
protocol, so neither needs a stepper of its own.
"""

from typing import Protocol, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearSolverProtocol,
)

V = TypeVar("V")
F = TypeVar("F")


@runtime_checkable
class ResidualFormSystemProtocol(Protocol[Array, V, F]):
    r""":math:`F(t, y, \dot y) = 0`, its shifted Jacobian, and a solver for it.

    ``Array`` is the backend array type of states, ``V`` the vector type
    of the stack the shifted Jacobian acts in and ``F`` its format; the
    solver pairs with ``F``.
    """

    def residual(self, state: Array, state_rate: Array, time: float) -> Array:
        r""":math:`F(t, y, \dot y)`. Shape: (nstates,)

        Parameters
        ----------
        state : Array
            :math:`y`. Shape: (nstates,)
        state_rate : Array
            :math:`\dot y`, or the stepper's approximation of it.
            Shape: (nstates,)
        time : float
            :math:`t`.
        """
        ...

    def shifted_jacobian(
        self,
        state: Array,
        state_rate: Array,
        time: float,
        shift: float,
        point: LinearizationPoint,
    ) -> JacobianSnapshotProtocol[V, F]:
        r""":math:`\sigma\,\partial F/\partial \dot y + \partial F/\partial y`
        at :math:`(t, y, \dot y)`, labeled with ``point``.

        The result may refer to ``state`` and ``state_rate`` rather than
        copy them, so it is valid while the caller leaves them unmodified.

        Parameters
        ----------
        shift : float
            :math:`\sigma`, the derivative of the stepper's
            :math:`\dot y` approximation with respect to :math:`y`.
        point : LinearizationPoint
            The identity of the linearization, minted by the caller.
        """
        ...

    def solver(self) -> LinearSolverProtocol[V, F]:
        """A solver paired with the shifted Jacobian's format."""
        ...
