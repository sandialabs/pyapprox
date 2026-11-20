r"""Nested substructuring: a group of subdomains as one response.

A group of subdomains with its own internal skeleton is itself a
Dirichlet-to-Neumann map. Its skeleton DOFs split into outer ones
:math:`o`, which subdomains outside the group also read, and inner ones
:math:`i`, read only by its members. Given the trace :math:`\lambda_o`
on the outer DOFs, the group solves its inner skeleton,
:math:`S_i(\lambda_o, \lambda_i) = 0`, and returns its members' summed
flux on the outer DOFs, :math:`S_o(\lambda_o, \lambda_i)`. Its trace
derivative is the inner Schur complement

.. math::

    S_{oo} - S_{oi} S_{ii}^{-1} S_{io},

the implicit-function derivative a mesh subdomain's response takes one
level down. So the group is built as that response:
:class:`~pyapprox.pde.coupling.pinned_response.SteadyPinnedResponse`
over the group's skeleton residual, its outer DOFs the port rows. The
response reads a residual in right-hand-side form and returns minus it
on the port rows, so the residual it is given is :math:`-S`. The outer
skeleton cannot tell a group from a subdomain, so nesting goes to any
depth, and each tangent product of a group costs one inner linear solve
with whatever solver its setup strategy pairs with the inner Jacobian's
format.
"""

from bisect import bisect_left
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import (
    PinnedValuesParamDerivatives,
    RowPin,
)
from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    ResidualJacobian,
)
from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import (
    PinnedReactionFlux,
    SteadyPinnedResponse,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    InnerProductSpaceProtocol,
    LinearizationPoint,
    LinearizedOperator,
    RowSetProtocol,
    VectorSpaceProtocol,
)
from pyapprox.util.linalg.operators.algebra import ScaledSum
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


@dataclass(frozen=True)
class GroupDofs:
    """A group's skeleton DOFs in its parent's numbering.

    Attributes
    ----------
    dofs
        The parent DOFs any member reads, increasing; the group's own
        skeleton numbers them by position.
    outer
        The positions in ``dofs`` of the DOFs a subdomain outside the
        group also reads, increasing: the group's ports.
    """

    dofs: tuple[int, ...]
    outer: tuple[int, ...]

    def size(self) -> int:
        """The number of the group's skeleton DOFs."""
        return len(self.dofs)

    def local(self, parent: Sequence[int]) -> tuple[int, ...]:
        """The group's numbers of the parent DOFs ``parent``.

        Raises
        ------
        ValueError
            If a DOF of ``parent`` is not the group's.
        """
        positions = []
        for dof in parent:
            k = bisect_left(self.dofs, dof)
            if k == len(self.dofs) or self.dofs[k] != dof:
                raise ValueError(f"parent DOF {dof} is not the group's")
            positions.append(k)
        return tuple(positions)

    def outer_parent(self) -> tuple[int, ...]:
        """The parent DOFs of the group's ports, in port order: its port
        DOFs one level up."""
        return tuple(self.dofs[k] for k in self.outer)

    def inner(self) -> tuple[int, ...]:
        """The positions of the DOFs only members read."""
        ports = set(self.outer)
        return tuple(k for k in range(len(self.dofs)) if k not in ports)


def group_dofs(
    members: Sequence[Sequence[int]], outside: Sequence[Sequence[int]]
) -> GroupDofs:
    """The skeleton DOFs of a group, in the parent's numbering.

    The group's DOFs are those its members read, one level down: a member
    that is itself a group reads only its ports, so the DOFs inside it
    are not the parent group's. Its outer DOFs are those also read
    outside it.

    Parameters
    ----------
    members
        Each member's port DOFs: a subdomain's, or a member group's
        :meth:`GroupDofs.outer_parent`.
    outside
        The port DOFs of whatever outside the group reads the skeleton:
        its siblings, or equally the subdomains outside it at any depth
        (a DOF a sibling group reads is read by one of its subdomains).

    Raises
    ------
    ValueError
        If there are no members, or they read no DOF.
    """
    if not members:
        raise ValueError("a group needs at least one member")
    inside = sorted({d for dofs in members for d in dofs})
    if not inside:
        raise ValueError("the members read no skeleton DOF")
    read_outside = {d for dofs in outside for d in dofs}
    outer = tuple(k for k, d in enumerate(inside) if d in read_outside)
    return GroupDofs(tuple(inside), outer)


class FixedValues(Generic[V]):
    """Rows of a state held at constant values: data that do not depend
    on any parameter.

    Satisfies :class:`~pyapprox.implicitfunction.pinned_rows.
    PinnedValuesProtocol`.

    Parameters
    ----------
    rows
        The rows, a row set of ``space``.
    values
        A single column of ``space`` holding the data on ``rows``; its
        other entries are not read. Copied.
    space
        The state space.

    Raises
    ------
    ValueError
        If ``rows`` belongs to another space or ``values`` is not a single
        column.
    """

    def __init__(
        self, rows: RowSetProtocol[V], values: V, space: VectorSpaceProtocol[V]
    ) -> None:
        space.descriptor().require_same(rows.descriptor(), "the fixed rows")
        if space.ncols(values) != 1:
            raise ValueError("values must be a single column")
        self._rows, self._values = rows, space.copy(values)
        self._bundle: PinnedValuesParamDerivatives[V] = (
            PinnedValuesParamDerivatives.independent()
        )

    def rows(self) -> RowSetProtocol[V]:
        return self._rows

    def values(self) -> V:
        return self._values

    def parameter_derivatives(self) -> PinnedValuesParamDerivatives[V]:
        return self._bundle


class _RightHandSide(Generic[V, F]):
    r""":math:`-S`: a skeleton residual in right-hand-side form, its
    Jacobian :math:`-J` in :math:`J`'s format."""

    def __init__(self, residual: ResidualProtocol[V, F], space: VectorSpaceProtocol[V]):
        jacobian = residual.derivatives().jacobian
        if jacobian is None:
            raise ValueError("the group's skeleton residual has no state Jacobian")
        self._residual, self._space, self._inner = residual, space, jacobian
        self._derivs: ResidualDerivatives[V, F] = ResidualDerivatives.state_only(
            ResidualJacobian.general(self._jacobian)
        )

    def __call__(self, state: V) -> V:
        return self._space.scale(-1.0, self._residual(state))

    def _jacobian(
        self, state: V, point: LinearizationPoint
    ) -> LinearizedOperator[V, F]:
        inner = self._inner(state, point)
        return LinearizedOperator(ScaledSum([(-1.0, inner)], self._space), point)

    def derivatives(self) -> ResidualDerivatives[V, F]:
        return self._derivs

    def parameters_version(self) -> int:
        return self._residual.parameters_version()


def skeleton_group_response(
    residual: ResidualProtocol[V, F],
    space: InnerProductSpaceProtocol[V, A],
    ports: PortPinnedData[V, A],
    essential: FixedValues[V],
    setups: SetupStrategyProtocol[SteadyRequest[V, F], V, F],
    nonlinear: NonlinearSolverProtocol[V, A],
    bkd: Backend[A],
) -> SteadyPinnedResponse[V, A, F]:
    r"""A group of subdomains as a Dirichlet-to-Neumann response (see the
    module docstring).

    Parameters
    ----------
    residual
        The group's skeleton residual :math:`S` on its own skeleton (e.g.
        a :class:`~pyapprox.pde.coupling.skeleton.SkeletonResidual` over
        its members), unpinned, with a state Jacobian.
    space
        The group's skeleton space.
    ports
        The group's outer DOFs as port rows of ``space``.
    essential
        The group's own pinned DOFs (Dirichlet closure points only members
        read) and their values; its rows may be empty. Pinned DOFs a
        subdomain outside the group also reads are outer: the parent pins
        them.
    setups
        Sets up the inner Jacobian's solves, paired with its format.
    nonlinear
        Solves the inner skeleton.
    bkd
        The backend of the port blocks.

    Raises
    ------
    TypeError
        If ``residual`` does not satisfy its protocol.
    ValueError
        If ``residual`` has no state Jacobian, or a port is also pinned.
    """
    if not isinstance(residual, ResidualProtocol):
        raise TypeError(
            f"residual must satisfy ResidualProtocol, got {type(residual).__name__}"
        )
    pinned = PinnedResidual(_RightHandSide(residual, space), RowPin(essential), space)
    return SteadyPinnedResponse(
        pinned, ports, PinnedReactionFlux(pinned, ports), space, setups, nonlinear, bkd
    )
