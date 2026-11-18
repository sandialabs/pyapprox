r"""Skeleton DOFs pinned to given values.

A skeleton DOF on the physical boundary where the problem fixes the
solution (a Dirichlet closure point) is not decided by flux balance: its
value is the boundary data :math:`g`. :class:`PinnedSkeletonResidual`
wraps any skeleton residual :math:`S` and replaces the pinned rows,

.. math::

    \tilde S(\lambda) = (I - P^\top P) S(\lambda)
        + P^\top (P \lambda - g),

with :math:`P` the 0/1 selection of the pinned DOFs. Its Jacobian is
:math:`(I - P^\top P) J + P^\top P`: free rows keep their dependence on
the pinned values, pinned rows are the identity, so Newton's first step
sets them to :math:`g` exactly. Subdomains receive :math:`g` through
:math:`\lambda` like any port value, so they must not also impose it as
an essential row at those port nodes.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    ResidualJacobian,
)
from pyapprox.pde.coupling.ports import PortRowsProtocol
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    LinearizationPoint,
    LinearizedOperator,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
)

V = TypeVar("V")


class _PinnedJacobian(Generic[V, Array]):
    """The inner Jacobian with the pinned rows replaced by the identity's,
    applied matrix-free."""

    def __init__(
        self,
        inner: ActionOperatorProtocol[V, MatrixFree],
        rows: PortRowsProtocol[V, Array],
    ) -> None:
        self._inner, self._rows = inner, rows

    def domain(self) -> SpaceDescriptor:
        return self._inner.domain()

    def range(self) -> SpaceDescriptor:
        return self._inner.range()

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        y = self._inner.apply(x)
        return self._rows.add_to(y, self._rows.take(x) - self._rows.take(y))

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        # (I - P^T P) J + P^T P transposed: J^T (I - P^T P) y + P^T P y.
        pinned = self._rows.take(y)
        free = self._rows.add_to(y, -pinned)
        return self._rows.add_to(self._inner.apply_transpose(free), pinned)


class PinnedSkeletonResidual(Generic[V, Array]):
    r"""A skeleton residual with the rows of pinned DOFs replaced by
    :math:`\lambda - g`.

    Satisfies :class:`~pyapprox.implicitfunction.residual.ResidualProtocol`
    with a matrix-free Jacobian: its rows are replaced as it is applied,
    so the inner residual's Jacobian must be matrix-free too.

    Parameters
    ----------
    residual
        The skeleton residual, with a state Jacobian (e.g.
        :class:`~pyapprox.pde.coupling.skeleton.SkeletonResidual` with
        :class:`~pyapprox.pde.coupling.skeleton.MatrixFreeSchur`).
    rows
        Selects the pinned DOFs of the skeleton.
    values
        Their values :math:`g`. Shape: (rows.nrows(), 1).

    Raises
    ------
    TypeError
        If ``residual`` or ``rows`` does not satisfy its protocol.
    ValueError
        If ``residual`` has no state Jacobian or ``values`` has the wrong
        shape.
    """

    def __init__(
        self,
        residual: ResidualProtocol[V, MatrixFree],
        rows: PortRowsProtocol[V, Array],
        values: Array,
    ) -> None:
        if not isinstance(residual, ResidualProtocol):
            raise TypeError(
                "residual must satisfy ResidualProtocol, got "
                f"{type(residual).__name__}"
            )
        if not isinstance(rows, PortRowsProtocol):
            raise TypeError(
                f"rows must satisfy PortRowsProtocol, got {type(rows).__name__}"
            )
        jacobian = residual.derivatives().jacobian
        if jacobian is None:
            raise ValueError("the residual has no state Jacobian to pin rows of")
        if tuple(values.shape) != (rows.nrows(), 1):
            raise ValueError(
                f"values must have shape ({rows.nrows()}, 1), got "
                f"{tuple(values.shape)}"
            )
        self._residual, self._rows, self._values = residual, rows, values
        self._inner_jacobian = jacobian
        self._derivs: ResidualDerivatives[V, MatrixFree] = (
            ResidualDerivatives.state_only(ResidualJacobian.general(self._jacobian))
        )

    def __call__(self, state: V) -> V:
        r = self._residual(state)
        rows = self._rows
        return rows.add_to(r, rows.take(state) - self._values - rows.take(r))

    def _jacobian(
        self, state: V, point: LinearizationPoint
    ) -> LinearizedOperator[V, MatrixFree]:
        inner = self._inner_jacobian(state, point)
        return LinearizedOperator(_PinnedJacobian(inner, self._rows), point)

    def derivatives(self) -> ResidualDerivatives[V, MatrixFree]:
        return self._derivs

    def parameters_version(self) -> int:
        return self._residual.parameters_version()
