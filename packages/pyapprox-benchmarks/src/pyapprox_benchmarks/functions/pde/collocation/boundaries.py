r"""Boundary conditions of collocation problems, by side and component.

Every condition is a spec naming mesh sides by
:class:`~pyapprox.pde.collocation.mesh.BoundarySide` (``LEFT`` and
``RIGHT`` in 1D; also ``BOTTOM`` and ``TOP`` on a rectangle) and its
data. A builder makes the
mesh, basis and physics, then calls each spec's ``rows`` with them, so
every kind goes through one path; specs never take node indices, normals
or derivative matrices. A spec returns :class:`BoundaryRows`: essential
parts (pinned, :math:`u = g`) or closure rows (kept in the closed
right-hand side); :func:`assemble_boundaries` merges them into what
``CollocationSystem`` takes. Where two specs share a node, essential
rows beat closure rows and, among closures, the later spec wins.

- :class:`Dirichlet`: :math:`u_c = g` for a state component :math:`c`;
- :class:`GradientRobin`: :math:`\alpha u + \beta\, \nabla u \cdot n = g`;
- :class:`FluxNeumann`: the total normal flux :math:`q \cdot n = g`, read
  from the physics (:math:`q = -D \nabla u` for diffusion);
- :class:`Periodic`: a pair of opposite sides, the value match on the
  primary side and the derivative match on the partner;
- :class:`CauchyTraction` and :class:`CauchyPressure`: a component of the
  traction :math:`\sigma n = t`, or :math:`t = -p\, n` on every component,
  for linear elasticity;
- :class:`PiolaTraction` and :class:`PiolaPressure`: the same with the
  first Piola-Kirchhoff stress, :math:`P n`, for hyperelasticity.

``GradientRobin``, ``FluxNeumann`` and ``Periodic`` are scalar
(component 0).
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
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    HyperelasticTractionNormalOperator,
    PK1StressSourceProtocol,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    CauchyStressSourceProtocol,
    FluxNormalOperator,
    GradientNormalOperator,
    TractionNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic_rows import PeriodicPart, PeriodicRows
from pyapprox.pde.collocation.mesh import BoundarySide
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.collocation.state_layout import StateLayout
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
    physics : CollocationInteriorOperatorProtocol
        The physics, for its number of states and for the flux and
        traction rows, which each check it for what they read.
    """

    mesh: BoundaryMeshProtocol[Array]
    basis: BoundaryBasisProtocol[Array]
    physics: CollocationInteriorOperatorProtocol[Array]

    def ncomponents(self) -> int:
        """The number of state components, ``nstates / npts``."""
        return self.physics.nstates() // self.mesh.npts()


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

    def components(self) -> Tuple[int, ...]:
        """The state components it constrains on each of its sides."""
        ...

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        """The rows on ``context``."""
        ...


@dataclass(frozen=True)
class Dirichlet:
    """:math:`u_c = g` on ``side``, :math:`c` the ``component``."""

    side: BoundarySide
    value: float
    component: int = 0

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def components(self) -> Tuple[int, ...]:
        return (self.component,)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        indices = context.mesh.boundary_indices(self.side)
        return BoundaryRows(
            essential=(DirichletPart(indices, self.value, self.component),)
        )


@dataclass(frozen=True)
class GradientRobin:
    r""":math:`\alpha u + \beta\, \nabla u \cdot n = g` on ``side``."""

    side: BoundarySide
    alpha: float
    beta: float
    value: float

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def components(self) -> Tuple[int, ...]:
        return (0,)

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
        return BoundaryRows(
            closure=RobinRows(
                bkd, StateLayout.uniform(mesh.npts(), context.ncomponents()), [part]
            )
        )


@dataclass(frozen=True)
class FluxNeumann:
    """The total normal flux :math:`q \\cdot n = g` on ``side``, :math:`q`
    the physics' flux."""

    side: BoundarySide
    value: float

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def components(self) -> Tuple[int, ...]:
        return (0,)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        mesh, bkd, physics = context.mesh, context.mesh.bkd(), context.physics
        if not isinstance(physics, FluxProviderProtocol):
            raise TypeError(
                "a flux condition needs a physics satisfying "
                f"FluxProviderProtocol, got {type(physics).__name__}"
            )
        indices = mesh.boundary_indices(self.side)
        normal_term = FluxNormalOperator(
            bkd, indices, mesh.boundary_normals(self.side), physics
        )
        part = RobinPart(indices, normal_term, 0.0, 1.0, self.value)
        return BoundaryRows(
            closure=RobinRows(
                bkd, StateLayout.uniform(mesh.npts(), context.ncomponents()), [part]
            )
        )


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

    def components(self) -> Tuple[int, ...]:
        return (0,)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        mesh = context.mesh
        part = PeriodicPart(
            mesh.boundary_indices(self.primary),
            mesh.boundary_indices(self.partner),
            context.basis.derivative_matrix(1, self.dim),
        )
        return BoundaryRows(
            closure=PeriodicRows(
                mesh.bkd(),
                StateLayout.uniform(
                    mesh.reference_points().shape[1], context.ncomponents()
                ),
                mesh.reference_points(),
                [part],
            )
        )


def _cauchy_check(
    physics: CollocationInteriorOperatorProtocol[Array],
) -> CauchyStressSourceProtocol[Array]:
    if not isinstance(physics, CauchyStressSourceProtocol):
        raise TypeError(
            "a Cauchy traction needs a physics satisfying "
            f"CauchyStressSourceProtocol, got {type(physics).__name__}"
        )
    return physics


def _piola_check(
    physics: CollocationInteriorOperatorProtocol[Array],
) -> PK1StressSourceProtocol[Array]:
    if not isinstance(physics, PK1StressSourceProtocol):
        raise TypeError(
            "a Piola traction needs a physics satisfying "
            f"PK1StressSourceProtocol, got {type(physics).__name__}"
        )
    return physics


def _traction_rows(
    context: BoundaryContext[Array], parts: List[RobinPart[Array]]
) -> BoundaryRows[Array]:
    mesh = context.mesh
    return BoundaryRows(
        closure=RobinRows(
            mesh.bkd(), StateLayout.uniform(mesh.npts(), context.ncomponents()), parts
        )
    )


def _pressure_data(
    context: BoundaryContext[Array], side: BoundarySide, pressure: float
) -> List[NodalField[Array]]:
    """:math:`t_c = -p\\, n_c` at the side's nodes, one per component of a
    2D displacement."""
    if context.ncomponents() != 2:
        raise ValueError(
            "a pressure needs a 2D displacement (2 components), got "
            f"{context.ncomponents()}"
        )
    normals = context.mesh.boundary_normals(side)
    return [
        NodalField(-pressure * normals[:, component])
        for component in range(context.ncomponents())
    ]


@dataclass(frozen=True)
class CauchyTraction:
    """Component ``component`` of the Cauchy traction,
    :math:`(\\sigma n)_c = t`, on ``side`` (linear elasticity)."""

    side: BoundarySide
    component: int
    value: float

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def components(self) -> Tuple[int, ...]:
        return (self.component,)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        mesh, bkd = context.mesh, context.mesh.bkd()
        physics = _cauchy_check(context.physics)
        indices, normals = (
            mesh.boundary_indices(self.side),
            mesh.boundary_normals(self.side),
        )
        operator = TractionNormalOperator(
            bkd, indices, normals, physics, self.component
        )
        part = RobinPart(indices, operator, 0.0, 1.0, self.value, self.component)
        return _traction_rows(context, [part])


@dataclass(frozen=True)
class CauchyPressure:
    """A pressure :math:`p` on ``side``: the Cauchy traction
    :math:`\\sigma n = -p\\, n` on every component (linear elasticity)."""

    side: BoundarySide
    pressure: float

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def components(self) -> Tuple[int, ...]:
        # Both components of a 2D displacement; rows refuses any other.
        return (0, 1)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        mesh, bkd = context.mesh, context.mesh.bkd()
        physics = _cauchy_check(context.physics)
        indices, normals = (
            mesh.boundary_indices(self.side),
            mesh.boundary_normals(self.side),
        )
        parts = [
            RobinPart(
                indices,
                TractionNormalOperator(bkd, indices, normals, physics, component),
                0.0,
                1.0,
                data,
                component,
            )
            for component, data in enumerate(
                _pressure_data(context, self.side, self.pressure)
            )
        ]
        return _traction_rows(context, parts)


@dataclass(frozen=True)
class PiolaTraction:
    """Component ``component`` of the first Piola-Kirchhoff traction,
    :math:`(P n)_c = t`, on ``side`` (hyperelasticity)."""

    side: BoundarySide
    component: int
    value: float

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def components(self) -> Tuple[int, ...]:
        return (self.component,)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        mesh, bkd = context.mesh, context.mesh.bkd()
        physics = _piola_check(context.physics)
        indices, normals = (
            mesh.boundary_indices(self.side),
            mesh.boundary_normals(self.side),
        )
        operator = HyperelasticTractionNormalOperator(
            bkd, indices, normals, physics, self.component
        )
        part = RobinPart(indices, operator, 0.0, 1.0, self.value, self.component)
        return _traction_rows(context, [part])


@dataclass(frozen=True)
class PiolaPressure:
    """A pressure :math:`p` on ``side``: the first Piola-Kirchhoff
    traction :math:`P n = -p\\, n` on every component, :math:`n` the
    reference normal (hyperelasticity)."""

    side: BoundarySide
    pressure: float

    def sides(self) -> Tuple[BoundarySide, ...]:
        return (self.side,)

    def components(self) -> Tuple[int, ...]:
        # Both components of a 2D displacement; rows refuses any other.
        return (0, 1)

    def rows(self, context: BoundaryContext[Array]) -> BoundaryRows[Array]:
        mesh, bkd = context.mesh, context.mesh.bkd()
        physics = _piola_check(context.physics)
        indices, normals = (
            mesh.boundary_indices(self.side),
            mesh.boundary_normals(self.side),
        )
        parts = [
            RobinPart(
                indices,
                HyperelasticTractionNormalOperator(
                    bkd, indices, normals, physics, component
                ),
                0.0,
                1.0,
                data,
                component,
            )
            for component, data in enumerate(
                _pressure_data(context, self.side, self.pressure)
            )
        ]
        return _traction_rows(context, parts)


def assemble_boundaries(
    context: BoundaryContext[Array], specs: Sequence[BoundarySpecProtocol]
) -> Tuple[Optional[DirichletRows[Array]], List[ClosureRowsProtocol[Array]]]:
    """The essential rows (``None`` when no spec has any) and the closure
    rows of ``specs``, in order.

    Raises
    ------
    TypeError
        If a spec does not satisfy ``BoundarySpecProtocol``, or its
        physics lacks what the spec reads.
    ValueError
        If ``specs`` is empty, a side is not one of the mesh's, a
        component is not one of the state's, or two specs name the same
        component on the same side.
    """
    if not specs:
        raise ValueError("a steady problem needs at least one boundary spec")
    mesh = context.mesh
    ncomponents = context.ncomponents()
    targets: List[Tuple[BoundarySide, int]] = []
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
        for component in spec.components():
            if not 0 <= component < ncomponents:
                raise ValueError(
                    f"{spec!r} names component {component}; the state has "
                    f"components 0 to {ncomponents - 1}"
                )
        targets.extend(
            (side, component)
            for side in spec.sides()
            for component in spec.components()
        )
        built = spec.rows(context)
        parts.extend(built.essential)
        if built.closure is not None:
            closures.append(built.closure)
    if len(set(targets)) != len(targets):
        raise ValueError(
            "two boundary specs name the same component on the same side: "
            f"{list(specs)}"
        )
    essential = (
        DirichletRows(mesh.bkd(), StateLayout.uniform(mesh.npts(), ncomponents), parts)
        if parts
        else None
    )
    return essential, closures
