"""The Galerkin state Jacobian as an operator of an injected stack.

``GalerkinJacobianOperator`` turns an injected
:class:`~pyapprox.pde.assembly.protocols.JacobianAssemblerProtocol` into
the state Jacobian with its essential rows zeroed, labeled with a
linearization point. It works on any stack: how the entries reach the
stack is the assembler's concern, and each assembler states what it
needs of the stack in its own constructor's types.
"""

from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.assembly.protocols import (
    AssemblyStackProtocol,
    JacobianAssemblerProtocol,
)
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    LinearizationPoint,
    LinearizedOperator,
    RowSetProtocol,
    RowsZeroedOperator,
    SpaceDescriptor,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)

_Dofs = NDArray[np.int64]
_Values = NDArray[np.float64]


def check_state_space(
    system: GalerkinTransientSystemProtocol[Array], descriptor: SpaceDescriptor
) -> None:
    """Check that ``descriptor`` can be the state space of ``system``.

    Raises
    ------
    TypeError
        If ``system`` does not satisfy ``GalerkinTransientSystemProtocol``.
    ValueError
        If ``descriptor`` does not have the system's number of states.
    """
    if not isinstance(system, GalerkinTransientSystemProtocol):
        raise TypeError(
            "system must satisfy GalerkinTransientSystemProtocol, "
            f"got {type(system).__name__}"
        )
    if descriptor.size != system.nstates():
        raise ValueError(
            f"descriptor has size {descriptor.size}, the system "
            f"{system.nstates()} states"
        )


def essential_rows(
    system: GalerkinTransientSystemProtocol[Array],
    stack: AssemblyStackProtocol[V, F, _Dofs, _Values],
    descriptor: SpaceDescriptor,
) -> RowSetProtocol[V]:
    """The system's essential DOFs as a row set of ``stack``.

    The constraint set gives its DOFs as a backend array; they are
    converted to the stack's index type once, here, as integer data that
    no derivative flows through.

    Raises
    ------
    TypeError
        If ``system`` or ``stack`` does not satisfy its protocol.
    ValueError
        If ``descriptor`` does not have the system's number of states.
    """
    check_state_space(system, descriptor)
    if not isinstance(stack, AssemblyStackProtocol):
        raise TypeError(
            f"stack must satisfy AssemblyStackProtocol, got {type(stack).__name__}"
        )
    dofs: _Dofs = np.asarray(
        system.bkd().to_numpy(system.constraint_set().dofs()), dtype=np.int64
    )
    return stack.row_set(descriptor, dofs)


class GalerkinJacobianOperator(Generic[Array, V, F]):
    """The state Jacobian with essential rows zeroed, on any stack.

    The essential rows are zeroed by
    :class:`~pyapprox.util.linalg.operators.RowsZeroedOperator` rather
    than in storage, so the stored operator is the plain assembled one.

    Parameters
    ----------
    assembler : JacobianAssemblerProtocol
        Produces the assembled Jacobian.
    rows : RowSetProtocol
        The essential rows, of the assembler's stack (see
        :func:`essential_rows`).

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    """

    def __init__(
        self,
        assembler: JacobianAssemblerProtocol[Array, V, F],
        rows: RowSetProtocol[V],
    ) -> None:
        if not isinstance(assembler, JacobianAssemblerProtocol):
            raise TypeError(
                "assembler must satisfy JacobianAssemblerProtocol, "
                f"got {type(assembler).__name__}"
            )
        if not isinstance(rows, RowSetProtocol):
            raise TypeError(
                f"rows must satisfy RowSetProtocol, got {type(rows).__name__}"
            )
        self._assembler = assembler
        self._rows = rows

    def jacobian_operator(
        self, state: Array, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[V, F]:
        """The Jacobian at ``(state, time)``, labeled with ``point``.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Time t.
        point : LinearizationPoint
            The identity of ``(state, time)``, minted by the caller.

        Returns
        -------
        LinearizedOperator
            The operator, independent of later calls.
        """
        assembled = self._assembler.assemble_jacobian(state, time)
        return LinearizedOperator(RowsZeroedOperator(assembled, self._rows), point)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(assembler={self._assembler!r})"
