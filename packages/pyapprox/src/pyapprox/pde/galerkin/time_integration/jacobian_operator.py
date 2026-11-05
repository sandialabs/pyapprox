"""The Galerkin state Jacobian as an operator of an injected stack.

``GalerkinJacobianOperator`` turns an injected
:class:`~pyapprox.pde.assembly.protocols.JacobianAssemblerProtocol` into
the state Jacobian, its essential rows zeroed or as assembled, labeled
with a linearization point; :func:`classified_jacobian` declares it in a
derivatives bundle by what it depends on. It works on any stack: how the
entries reach the stack is the assembler's concern, and each assembler
states what it needs of the stack in its own constructor's types.
"""

from typing import Generic, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

from pyapprox.ode.residual_derivatives import Jacobian
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
    """The state Jacobian, with given rows zeroed or as assembled, on any
    stack.

    The rows are zeroed by
    :class:`~pyapprox.util.linalg.operators.RowsZeroedOperator` rather
    than in storage, so the stored operator is the plain assembled one.

    Parameters
    ----------
    assembler : JacobianAssemblerProtocol
        Produces the assembled Jacobian.
    rows : RowSetProtocol, optional
        The rows to zero, of the assembler's stack (the essential rows,
        see :func:`essential_rows`); none, for the Jacobian as assembled.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol.
    """

    def __init__(
        self,
        assembler: JacobianAssemblerProtocol[Array, V, F],
        rows: Optional[RowSetProtocol[V]],
    ) -> None:
        if not isinstance(assembler, JacobianAssemblerProtocol):
            raise TypeError(
                "assembler must satisfy JacobianAssemblerProtocol, "
                f"got {type(assembler).__name__}"
            )
        if rows is not None and not isinstance(rows, RowSetProtocol):
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
        if self._rows is None:
            return LinearizedOperator(assembled, point)
        return LinearizedOperator(RowsZeroedOperator(assembled, self._rows), point)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(assembler={self._assembler!r})"


class _StateFreeJacobian(Generic[Array, F]):
    """The Jacobian of an affine :math:`F` at a time, assembled at a fixed
    reference state: an affine physics ignores the state, so any state
    gives this same Jacobian."""

    def __init__(
        self, operator: GalerkinJacobianOperator[Array, Array, F], reference: Array
    ) -> None:
        self._operator, self._reference = operator, reference

    def __call__(
        self, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[Array, F]:
        return self._operator.jacobian_operator(self._reference, time, point)


class _ConstantJacobian(Generic[Array, F]):
    """The Jacobian of an affine :math:`F` declared time-invariant: the
    same at every state and time, so assembled at the reference state and
    time zero."""

    def __init__(self, at_time: _StateFreeJacobian[Array, F]) -> None:
        self._at_time = at_time

    def __call__(self, point: LinearizationPoint) -> LinearizedOperator[Array, F]:
        return self._at_time(0.0, point)


def classified_jacobian(
    system: GalerkinTransientSystemProtocol[Array],
    operator: GalerkinJacobianOperator[Array, Array, F],
) -> Jacobian[Array, F]:
    r"""``operator`` as a bundle's Jacobian, given only what it depends on.

    Of the time only when the physics declares :math:`F` affine in the
    state (its zero-curvature ``StateDerivatives``), so it is not given
    the state, and of neither when every part of :math:`F` also declares
    its Jacobian time-invariant. Zeroing a fixed set of rows keeps both:
    the zeroed rows see neither state nor time.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The system whose physics declares what :math:`F` depends on.
    operator : GalerkinJacobianOperator
        The Jacobian, with whatever rows it zeroes.
    """
    spatial_operator = system.spatial_operator()
    if not spatial_operator.state_derivatives().affine():
        return Jacobian.general(operator.jacobian_operator)
    # The state an affine physics' Jacobian is assembled at: it ignores
    # the state, so any one gives the same Jacobian; allocated once.
    at_time = _StateFreeJacobian(operator, system.bkd().zeros((system.nstates(),)))
    if spatial_operator.jacobian_is_time_invariant():
        return Jacobian.constant(_ConstantJacobian(at_time))
    return Jacobian.state_free(at_time)
