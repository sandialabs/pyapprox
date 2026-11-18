r"""Boundary conditions of scalar collocation problems, by side.

Every condition is a spec naming mesh sides by
:class:`~pyapprox.pde.collocation.mesh.BoundarySide` (``LEFT`` and
``RIGHT`` in 1D; also ``BOTTOM`` and ``TOP`` on a rectangle) and its
data. A builder makes the
mesh, basis and physics, then calls each spec's ``rows`` with them, so
every kind goes through one path; specs never take node indices, normals
or derivative matrices. A spec returns :class:`BoundaryRows`: essential
parts (pinned, :math:`u = g`) or closure rows (kept in the closed
right-hand side); :func:`assemble_boundaries` merges them into what
``CollocationSystem`` takes.

- :class:`Dirichlet`: :math:`u = g`;
- :class:`GradientRobin`: :math:`\alpha u + \beta\, \nabla u \cdot n = g`;
- :class:`FluxNeumann`: the total normal flux :math:`q \cdot n = g`, read
  from the physics (:math:`q = -D \nabla u` for diffusion);
- :class:`Periodic`: a pair of opposite sides, the value match on the
  primary side and the derivative match on the partner.

Scalar states only (one component).
"""

from dataclasses import dataclass
from typing import (
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsProtocol,
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic_rows import PeriodicPart, PeriodicRows
from pyapprox.pde.collocation.mesh import BoundarySide
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.util.backends.protocols import Array, Backend

Array_co = TypeVar("Array_co", covariant=True)


@runtime_checkable
class BoundaryMeshProtocol(Protocol, Generic[Array]):
    """What the specs read from a mesh."""

    def bkd(self) -> Backend[Array]: ...

    def npts(self) -> int: ...

    def nboundaries(self) -> int: ...

    def boundary_indices(self, boundary_id: int) -> Array: ...

    def boundary_normals(self, boundary_id: int) -> Array: ...

    def reference_points(self) -> Array: ...


@runtime_checkable
class BoundaryBasisProtocol(Protocol[Array_co]):
    """What the specs read from a basis."""

    def ndim(self) -> int: ...

    def derivative_matrix(self, order: int, dim: int) -> Array_co: ...


@dataclass(frozen=True)
class BoundaryContext(Generic[Array]):
    """What a spec builds its rows from.

    Attributes
    ----------
    mesh : BoundaryMeshProtocol
        The mesh, for the sides' nodes and normals.
    basis : BoundaryBasisProtocol
        The basis, for derivative matrices.
    physics : FluxProviderProtocol
        The physics, for flux rows.
    """

    mesh: BoundaryMeshProtocol[Array]
    basis: BoundaryBasisProtocol[Array]
    physics: FluxProviderProtocol[Array]


@dataclass(frozen=True)
class BoundaryRows(Generic[Array]):
    """A spec's rows: essential parts, closure rows, or both."""

    essential: Tuple[DirichletPart[Array], ...] = ()
    closure: Optional[ClosureRowsProtocol[Array]] = None


@runtime_checkable
class BoundarySpecProtocol(Protocol):
    """A boundary condition that builds its rows from the context."""

    def sides(self) -> Tuple[BoundarySide, ...]:
        """The sides it constrains."""
        ...

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        """The rows on ``context``."""
        ...


@dataclass(frozen=True)
class Dirichlet:
    """:math:`u = g` on ``side``."""

    side: BoundarySide
    value: float

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        indices = context.mesh.boundary_indices(self.side)
        return BoundaryRows(essential=(DirichletPart(indices, self.value),))


@dataclass(frozen=True)
class GradientRobin:
    r""":math:`\alpha u + \beta\, \nabla u \cdot n = g` on ``side``."""

    side: BoundarySide
    alpha: float
    beta: float
    value: float

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        mesh, basis, bkd = context.mesh, context.basis, context.mesh.bkd()
        indices = mesh.boundary_indices(self.side)
        normal_term = GradientNormalOperator(
            bkd,
            indices,
            mesh.boundary_normals(self.side),
            [basis.derivative_matrix(1, dim) for dim in range(basis.ndim())],
        )
        part = RobinPart(indices, normal_term, self.alpha, self.beta, self.value)
        return BoundaryRows(closure=RobinRows(bkd, mesh.npts(), 1, [part]))


@dataclass(frozen=True)
class FluxNeumann:
    """The total normal flux :math:`q \\cdot n = g` on ``side``, :math:`q`
    the physics' flux."""

    side: BoundarySide
    value: float

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        mesh, bkd = context.mesh, context.mesh.bkd()
        indices = mesh.boundary_indices(self.side)
        normal_term = FluxNormalOperator(
            bkd, indices, mesh.boundary_normals(self.side), context.physics
        )
        part = RobinPart(indices, normal_term, 0.0, 1.0, self.value)
        return BoundaryRows(closure=RobinRows(bkd, mesh.npts(), 1, [part]))


@dataclass(frozen=True)
class Periodic:
    """Periodicity in direction ``dim`` between the opposite sides
    ``primary`` and ``partner``: the value match on ``primary``'s nodes,
    the derivative match on ``partner``'s. Default the two ends of a 1D
    mesh."""

    primary: BoundarySide = BoundarySide.LEFT
    partner: BoundarySide = BoundarySide.RIGHT
    dim: int = 0

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.primary, self.partner)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        mesh = context.mesh
        part = PeriodicPart(
            mesh.boundary_indices(self.primary),
            mesh.boundary_indices(self.partner),
            context.basis.derivative_matrix(1, self.dim),
        )
        return BoundaryRows(
            closure=PeriodicRows(mesh.bkd(), mesh.reference_points(), 1, [part])
        )


def assemble_boundaries(
    context: BoundaryContext[Array], specs: Sequence[BoundarySpecProtocol]
) -> Tuple[Optional[DirichletRows[Array]], List[ClosureRowsProtocol[Array]]]:
    """The essential rows (``None`` when no spec has any) and the closure
    rows of ``specs``, in order.

    Raises
    ------
    TypeError
        If a spec does not satisfy ``BoundarySpecProtocol``.
    ValueError
        If ``specs`` is empty, a side is not one of the mesh's, or two
        specs name the same side.
    """
    if not specs:
        raise ValueError("a steady problem needs at least one boundary spec")
    mesh = context.mesh
    sides: List[BoundarySide] = []
    parts: List[DirichletPart[Array]] = []
    closures: List[ClosureRowsProtocol[Array]] = []
    for spec in specs:
        if not isinstance(spec, BoundarySpecProtocol):
            raise TypeError(
                "boundary specs must satisfy BoundarySpecProtocol, got "
                f"{type(spec).__name__}"
            )
        for side in spec.sides():
            if not 0 <= side < mesh.nboundaries():
                raise ValueError(
                    f"{spec!r} names side {side}; the mesh has sides 0 to "
                    f"{mesh.nboundaries() - 1}"
                )
        sides.extend(spec.sides())
        built = spec.rows(context)
        parts.extend(built.essential)
        if built.closure is not None:
            closures.append(built.closure)
    if len(set(sides)) != len(sides):
        raise ValueError(f"two boundary specs name the same side: {list(specs)}")
    essential = DirichletRows(mesh.bkd(), mesh.npts(), 1, parts) if parts else None
    return essential, closures
