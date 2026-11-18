r"""Skeleton DOFs pinned to given values.

A skeleton DOF on the physical boundary where the problem fixes the
solution (a Dirichlet closure point) is not decided by flux balance: its
value is the boundary data :math:`g`. :class:`PinnedSkeletonResidual`
wraps any skeleton residual :math:`S` and replaces the pinned rows,

.. math::

    \tilde S(\lambda) = (I - P) S(\lambda) + P (\lambda - g),

with :math:`P` the diagonal 0/1 selection of the pinned DOFs. Its
Jacobian is :math:`(I - P) J + P`
(:class:`~pyapprox.util.linalg.operators.RowReplacedOperator`), in the
inner Jacobian's format: free rows keep their dependence on the pinned
values, pinned rows are the identity, so Newton's first step sets them
to :math:`g` exactly. Subdomains receive :math:`g` through
:math:`\lambda` like any port value, so they must not also impose it as
an essential row at those port nodes.
"""

from typing import Generic, TypeVar

from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    ResidualJacobian,
)
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    LinearizationPoint,
    LinearizedOperator,
    RowReplacedOperator,
    RowSetProtocol,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)


class PinnedSkeletonResidual(Generic[V, F]):
    r"""A skeleton residual with the rows of pinned DOFs replaced by
    :math:`\lambda - g`.

    Satisfies :class:`~pyapprox.implicitfunction.residual.ResidualProtocol`
    with a Jacobian in the inner residual's format, so it pairs with the
    same solvers: Krylov for
    :class:`~pyapprox.pde.coupling.skeleton.MatrixFreeSchur`, a dense
    direct solver for :class:`~pyapprox.pde.coupling.skeleton.DenseSchur`.

    Parameters
    ----------
    residual
        The skeleton residual, with a state Jacobian.
    space
        The skeleton's space.
    rows
        The pinned DOFs, a row set of the skeleton's space.
    values
        A skeleton vector holding :math:`g` on the pinned rows; its other
        entries are not read.

    Raises
    ------
    TypeError
        If ``residual`` or ``rows`` does not satisfy its protocol.
    ValueError
        If ``residual`` has no state Jacobian, or ``rows`` or ``values``
        belongs to another space.
    """

    def __init__(
        self,
        residual: ResidualProtocol[V, F],
        space: VectorSpaceProtocol[V],
        rows: RowSetProtocol[V],
        values: V,
    ) -> None:
        if not isinstance(residual, ResidualProtocol):
            raise TypeError(
                "residual must satisfy ResidualProtocol, got "
                f"{type(residual).__name__}"
            )
        if not isinstance(rows, RowSetProtocol):
            raise TypeError(
                f"rows must satisfy RowSetProtocol, got {type(rows).__name__}"
            )
        jacobian = residual.derivatives().jacobian
        if jacobian is None:
            raise ValueError("the residual has no state Jacobian to pin rows of")
        space.descriptor().require_same(rows.descriptor(), "pinned rows")
        if space.ncols(values) != 1:
            raise ValueError("values must be a single skeleton vector")
        self._residual, self._space = residual, space
        self._rows, self._values = rows, space.copy(values)
        self._inner_jacobian = jacobian
        self._derivs: ResidualDerivatives[V, F] = ResidualDerivatives.state_only(
            ResidualJacobian.general(self._jacobian)
        )

    def __call__(self, state: V) -> V:
        offset = self._space.axpby(1.0, state, -1.0, self._values)
        return self._rows.replace_rows(self._residual(state), offset)

    def _jacobian(
        self, state: V, point: LinearizationPoint
    ) -> LinearizedOperator[V, F]:
        inner = self._inner_jacobian(state, point)
        return LinearizedOperator(RowReplacedOperator(inner, self._rows), point)

    def derivatives(self) -> ResidualDerivatives[V, F]:
        return self._derivs

    def parameters_version(self) -> int:
        return self._residual.parameters_version()
