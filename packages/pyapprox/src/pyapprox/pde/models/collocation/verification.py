r"""Verify collocation discretizations with :mod:`pyapprox.pde.verification`.

The adapter between the stack-independent verification core and this
repository's spectral collocation stack. Backend-generic: the levels,
coefficients and rows live on the family's backend, and only the errors
leave it, as floats.

- A level family gives a study's levels, each a :class:`ChebyshevLevel`:
  a Chebyshev-Gauss-Lobatto tensor mesh of the reference cube mapped onto
  the domain, its basis, and the mesh boundary of each boundary part.
  :class:`ChebyshevBox` meshes a :class:`~pyapprox.pde.verification.
  domains.Box` (affine maps), one level per point count; a side of the
  box is the mesh boundary at the same position in
  :meth:`~pyapprox.pde.verification.domains.Box.boundaries` (left,
  right, bottom, top, back, front), so named boxes work unchanged.
  :class:`ChebyshevAnnulus` and :class:`ChebyshevEllipticalAnnulus` mesh
  the two rings through the polar and elliptical maps over the full
  angle, the angle's two ends joined by a :class:`PeriodicSeam`.
- A system factory, supplied by the case, builds the physics under test
  on a level and closes it with :func:`collocation_system`. Nothing here
  knows any physics: a factory builds its physics class from the generic
  helpers :func:`field`, :func:`nodal_vector` and :func:`forcing`, which
  declare each coefficient's time dependence from its expression (it
  uses ``t`` or not).
- The state stacks one component per field of the problem, in the
  problem's field order; a scalar problem has one.
- :func:`boundary_conditions` builds the rows of each side's condition
  on each field, on that field's component: Dirichlet as
  :class:`~pyapprox.pde.collocation.boundary.essential_rows.
  DirichletRows`; Neumann and Robin as :class:`~pyapprox.pde.collocation.
  boundary.closure_rows.RobinRows` on the normal term supplied for that
  field: a flux provider's flux normal operator (:class:`FluxTerm`) or
  another :class:`NormalTermProtocol`, such as the Cauchy traction
  (:class:`TractionTerm`) or the nominal (first Piola) traction
  (:class:`PiolaTractionTerm`); periodic sides as
  :class:`~pyapprox.pde.collocation.
  boundary.periodic_rows.PeriodicRows` pairing the two sides of a
  coordinate. It adds the level's interior closures
  (:class:`InteriorClosureProtocol`), rows for mesh boundaries inside the
  domain such as a ring's seam, on every field. A node two sets of rows
  share follows the system's rule: essential rows win it, then the Robin
  rows, then the periodic rows, then the interior closures. A case that
  builds its rows by hand (a wrong-model control) takes a side's nodes
  and data from :func:`side_nodes` and :func:`side_values`.
- :class:`CollocationDiscretizer` reports each level's sizes (the
  largest node gap, the degree and the number of unknowns) without
  solving, and solves a level through ``collocation_steady_residual`` +
  ``SteadySystem`` with a dense LU, from the state an injected
  :class:`InitialGuessProtocol` gives (:class:`ConstantGuess` zero by
  default). It returns :class:`CollocationApproximation` objects, which
  the stack's measures read per field: :func:`collocation_l2_error` by
  Gauss quadrature mapped onto the domain, :func:`collocation_nodal_error`
  at the collocation nodes. Spectral studies are judged by the core's
  ``DecaysExponentially`` along the degree.

Natural data: the core states them with the outward flux :math:`q\cdot
n`, which is what this stack's flux normal operator computes, so Neumann
data pass unnegated, as rows :math:`q(u)\cdot n = g`. The core's Robin
condition :math:`q\cdot n - \alpha u = g` is the rows' :math:`\alpha' u +
\beta\,q(u)\cdot n = g` with :math:`\alpha' = -\alpha` and
:math:`\beta = 1`. A normal term whose operator is :math:`N = s\,q\cdot n`
takes :math:`\beta = s` instead, so the data still pass unnegated.
"""

import math
from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import sympy as sp

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.basis import ChebyshevBasis, ChebyshevGauss1D
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsProtocol,
    RobinNormalTermProtocol,
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
from pyapprox.pde.collocation.boundary.mean_value_rows import MeanValueRows
from pyapprox.pde.collocation.boundary.normal_operators import (
    CauchyStressSourceProtocol,
    FluxNormalOperator,
    TractionNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic_rows import (
    PeriodicPart,
    PeriodicRows,
)
from pyapprox.pde.collocation.mesh import TransformedMesh
from pyapprox.pde.collocation.mesh.transforms.affine import AffineTransform
from pyapprox.pde.collocation.mesh.transforms.elliptical import (
    EllipticalTransform,
)
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.physics.coefficients import NodalSupplier
from pyapprox.pde.collocation.protocols.basis import TensorProductBasisProtocol
from pyapprox.pde.collocation.protocols.basis_1d import CollocationBasis1DProtocol
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.pde.collocation.protocols.mesh import TransformProtocol
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.pde.expressions.evaluation import PointExpressions
from pyapprox.pde.expressions.symbols import coordinates, time
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.pde.verification.conditions import (
    Dirichlet,
    Neumann,
    Periodic,
    Robin,
    Unprescribed,
)
from pyapprox.pde.verification.domains import (
    Annulus,
    Box,
    DomainProtocol,
    EllipticalAnnulus,
)
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.protocols import (
    DEGREE,
    NDOFS,
    H,
    Sizes,
    SpaceLevel,
)
from pyapprox.pde.verification.study import Measure, Measurement, field_label
from pyapprox.polynomials.univariate.globalpoly.jacobi import LegendrePolynomial1D
from pyapprox.polynomials.univariate.globalpoly.quadrature import GaussQuadratureRule
from pyapprox.polynomials.univariate.lagrange import univariate_lagrange_polynomial
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.expressions.parsing import Value, parse
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton, NonlinearSolverProtocol


class LevelMapProtocol(Protocol[Array]):
    """What the adapter reads from a level mesh's map of the reference
    cube :math:`[-1, 1]^d` onto the domain. The adapter keeps the
    reference directions (``d``, the mesh's ``npts_per_dim``) apart from
    the physical coordinates (the rows of ``points``), so a map into a
    space of more coordinates than directions fits too."""

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Shape: (d, npts) to (ncoords, npts)"""
        ...

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """The map's volume element, the factor that turns a reference
        integral into a physical one. Shape: (d, npts) to (npts,)"""
        ...


class LevelMeshProtocol(Protocol[Array]):
    """What the adapter reads from a level's mesh: a tensor
    Chebyshev-Gauss-Lobatto mesh of the reference cube, first dimension
    fastest, mapped onto the domain. Boundaries ``2 d`` and ``2 d + 1``
    are the lower and upper ends of reference direction ``d``."""

    def ndim(self) -> int: ...

    def npts(self) -> int: ...

    def npts_per_dim(self) -> Tuple[int, ...]: ...

    def points(self) -> Array:
        """Physical coordinates. Shape: (ncoords, npts)"""
        ...

    def reference_points(self) -> Array:
        """Reference coordinates. Shape: (d, npts)"""
        ...

    def reference_nodes_1d(self, dim: int) -> Array:
        """Reference nodes of dimension ``dim``. Shape: (npts_dim,)"""
        ...

    def basis_1d(self, dim: int) -> CollocationBasis1DProtocol[Array]:
        """The 1D basis that made dimension ``dim``'s nodes."""
        ...

    def boundary_indices(self, boundary_id: int) -> Array:
        """Node indices of a boundary. Shape: (nboundary,)"""
        ...

    def boundary_normals(self, boundary_id: int) -> Array:
        """Outward unit normals of a boundary. Shape: (nboundary, ncoords)"""
        ...

    def jacobian_determinant(self) -> Array:
        """The map's volume element at the nodes. Shape: (npts,)"""
        ...

    def transform(self) -> LevelMapProtocol[Array]:
        """The map of the reference cube onto the domain."""
        ...


@runtime_checkable
class InteriorClosureProtocol(Protocol[Array]):
    """Rows that join a level's mesh to itself inside the domain, on every
    field: mesh boundaries that are not boundary of the domain, such as
    the two ends of a ring's angle (one physical line, a seam)."""

    def rows(
        self, level: "ChebyshevLevel[Array]", layout: StateLayout
    ) -> ClosureRowsProtocol[Array]:
        """The rows on ``level``'s mesh, for a state laid out as
        ``layout``."""
        ...


@dataclass(frozen=True)
class FieldMesh(Generic[Array]):
    """A field's own mesh and basis on a level, apart from the level's
    mesh: a staggered pressure on interior Gauss nodes. It shares the
    level mesh's map, so both describe one geometry.

    ``mesh`` is the field's mesh; ``basis`` the collocation basis on it.
    """

    mesh: LevelMeshProtocol[Array]
    basis: TensorProductBasisProtocol[Array]


class ChebyshevLevel(Generic[Array]):
    """One level of a study: a mapped Chebyshev mesh of a domain and its
    basis, on which every field lives unless the level gives it a mesh of
    its own (:class:`FieldMesh`). Boundary conditions, interior closures
    and :meth:`h` belong to the level's mesh.

    Parameters
    ----------
    domain : DomainProtocol
        The domain meshed.
    mesh : LevelMeshProtocol
        The mesh.
    basis : TensorProductBasisProtocol
        The collocation basis on ``mesh``.
    parts : Mapping[str, Sequence[int]]
        The mesh boundaries of each of the domain's boundary parts; a part
        may cover several (a disk's circle, all four sides of a square
        mapped onto it).
    closures : Sequence[InteriorClosureProtocol], optional
        Rows for the mesh boundaries no part covers, which lie inside the
        domain. Default none.
    field_meshes : Mapping[str, FieldMesh], optional
        By field name, the fields on a mesh of their own. Default none:
        every field on ``mesh``.

    Raises
    ------
    ValueError
        If ``parts`` does not name exactly the domain's boundary parts, or
        gives a part no mesh boundary, or a field mesh has another
        dimension or another map than ``mesh``.
    TypeError
        If a closure does not satisfy ``InteriorClosureProtocol``.
    """

    def __init__(
        self,
        domain: DomainProtocol,
        mesh: LevelMeshProtocol[Array],
        basis: TensorProductBasisProtocol[Array],
        parts: Mapping[str, Sequence[int]],
        closures: Sequence[InteriorClosureProtocol[Array]] = (),
        field_meshes: Optional[Mapping[str, FieldMesh[Array]]] = None,
    ) -> None:
        if set(parts) != set(domain.boundaries()):
            raise ValueError(
                f"parts must name the domain's boundary parts "
                f"{list(domain.boundaries())}, got {sorted(parts)}"
            )
        for part, boundaries in parts.items():
            if not boundaries:
                raise ValueError(f"part {part!r} covers no mesh boundary")
        for closure in closures:
            if not isinstance(closure, InteriorClosureProtocol):
                raise TypeError(
                    "closures must satisfy InteriorClosureProtocol, got "
                    f"{type(closure).__name__}"
                )
        own = {} if field_meshes is None else dict(field_meshes)
        for name, placed in own.items():
            if placed.mesh.ndim() != mesh.ndim():
                raise ValueError(
                    f"field {name!r}'s mesh is {placed.mesh.ndim()}D, the "
                    f"level's {mesh.ndim()}D"
                )
            if placed.mesh.transform() is not mesh.transform():
                raise ValueError(
                    f"field {name!r}'s mesh must share the level mesh's map, "
                    "so both describe one geometry"
                )
        self._domain, self._mesh, self._basis = domain, mesh, basis
        self._parts = {part: tuple(ids) for part, ids in parts.items()}
        self._closures = tuple(closures)
        self._field_meshes = own

    def bkd(self) -> Backend[Array]:
        return self._basis.bkd()

    def domain(self) -> DomainProtocol:
        return self._domain

    def mesh(self) -> LevelMeshProtocol[Array]:
        return self._mesh

    def basis(self) -> TensorProductBasisProtocol[Array]:
        return self._basis

    def npts(self) -> int:
        """The number of nodes of the level's mesh."""
        return self._mesh.npts()

    def field_mesh(self, name: str) -> LevelMeshProtocol[Array]:
        """The mesh field ``name`` lives on: its own, or the level's."""
        placed = self._field_meshes.get(name)
        return self._mesh if placed is None else placed.mesh

    def field_basis(self, name: str) -> TensorProductBasisProtocol[Array]:
        """The basis of field ``name``'s mesh."""
        placed = self._field_meshes.get(name)
        return self._basis if placed is None else placed.basis

    def is_uniform(self) -> bool:
        """Whether every field lives on the level's mesh."""
        return not self._field_meshes

    def layout(self, fields: Sequence[str]) -> StateLayout:
        """Where each of ``fields`` sits in the state, stacked in order,
        each with its mesh's node count.

        Raises
        ------
        ValueError
            If there are no fields, a name repeats, or the level gives a
            mesh of its own to a field not among ``fields``.
        """
        unknown = sorted(set(self._field_meshes) - set(fields))
        if unknown:
            raise ValueError(
                f"the level places {unknown} on their own meshes, but the "
                f"state's fields are {list(fields)}"
            )
        if self.is_uniform():
            return StateLayout.uniform(self.npts(), len(fields), names=fields)
        return StateLayout(
            [self.field_mesh(name).npts() for name in fields], names=fields
        )

    def closures(self) -> Tuple[InteriorClosureProtocol[Array], ...]:
        """Rows for the mesh boundaries inside the domain."""
        return self._closures

    def h(self) -> float:
        """The largest physical distance between neighboring nodes along
        a mesh line. On a box's Chebyshev-Gauss-Lobatto nodes it is the
        central gap, about :math:`\\pi L / (2 (N - 1))` on a side of length
        :math:`L` with :math:`N` nodes, so :math:`1/h` grows linearly in
        :math:`N`."""
        bkd, mesh = self.bkd(), self._mesh
        ndirections = len(mesh.npts_per_dim())
        points = mesh.points()
        # Nodes run first direction fastest, so direction d is array axis
        # ndirections - 1 - d of the C-ordered grid.
        shape = tuple(reversed(mesh.npts_per_dim()))
        grids = [bkd.reshape(points[k], shape) for k in range(points.shape[0])]
        gaps = []
        for dim in range(ndirections):
            axis = ndirections - 1 - dim
            squared = bkd.diff(grids[0], axis=axis) ** 2
            for grid in grids[1:]:
                squared = squared + bkd.diff(grid, axis=axis) ** 2
            gaps.append(bkd.to_float(bkd.max(bkd.sqrt(squared))))
        return max(gaps)

    def boundary_ids(self, part: str) -> Tuple[int, ...]:
        """The mesh boundaries of boundary part ``part``.

        Raises
        ------
        ValueError
            If the domain has no such part.
        """
        if part not in self._parts:
            raise ValueError(
                f"no boundary part {part!r}; the domain has "
                f"{list(self._domain.boundaries())}"
            )
        return self._parts[part]

    def __repr__(self) -> str:
        own = "".join(
            f", {name}={placed.mesh.npts_per_dim()}"
            for name, placed in self._field_meshes.items()
        )
        return f"{type(self).__name__}(npts_per_dim={self._mesh.npts_per_dim()}{own})"


class LevelFamilyProtocol(Protocol[Array]):
    """A study's levels on a domain, coarsest first."""

    def levels(self, domain: DomainProtocol) -> Sequence[ChebyshevLevel[Array]]:
        """The levels of ``domain``."""
        ...


class ChebyshevBox(Generic[Array]):
    """Chebyshev-Gauss-Lobatto meshes of a :class:`~pyapprox.pde.
    verification.domains.Box`, ``n`` nodes per direction at each level.

    Parameters
    ----------
    sizes : Sequence[int]
        Nodes per direction, one entry per level.
    bkd : Backend
        The backend of the meshes, coefficients and states.
    interior_fields : Sequence[str], optional
        Fields on ``n - 2`` Chebyshev-Gauss nodes per direction, which
        reach no boundary (a staggered pressure); default none.

    Raises
    ------
    ValueError
        If a level with interior fields has fewer than 3 nodes per
        direction.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        bkd: Backend[Array],
        interior_fields: Sequence[str] = (),
    ) -> None:
        self._sizes, self._bkd = tuple(sizes), bkd
        self._interior_fields = tuple(interior_fields)
        if self._interior_fields and min(self._sizes) < 3:
            raise ValueError(
                "interior fields need at least 3 nodes per direction (one "
                f"interior node), got sizes {list(self._sizes)}"
            )

    def levels(self, domain: DomainProtocol) -> List[ChebyshevLevel[Array]]:
        """One level per size.

        Raises
        ------
        TypeError
            If ``domain`` is not a ``Box``.
        """
        if not isinstance(domain, Box):
            raise TypeError(f"ChebyshevBox meshes a Box, got {type(domain).__name__}")
        bkd = self._bkd
        box = AffineTransform(domain.bounds(), bkd)
        # A side is the mesh boundary at its position in the box's side
        # order, lower side first in each direction.
        parts = {part: (i,) for i, part in enumerate(domain.boundaries())}
        levels: List[ChebyshevLevel[Array]] = []
        for n in self._sizes:
            mesh = TransformedMesh((n,) * domain.ndim(), bkd, box)
            own: Dict[str, FieldMesh[Array]] = {}
            if self._interior_fields:
                interior = TransformedMesh(
                    (n - 2,) * domain.ndim(),
                    bkd,
                    box,
                    bases=[ChebyshevGauss1D[Array]()] * domain.ndim(),
                )
                placed = FieldMesh(interior, ChebyshevBasis(interior, bkd))
                own = {name: placed for name in self._interior_fields}
            levels.append(
                ChebyshevLevel(
                    domain, mesh, ChebyshevBasis(mesh, bkd), parts, field_meshes=own
                )
            )
        return levels

    def __repr__(self) -> str:
        interior = (
            f", interior_fields={list(self._interior_fields)}"
            if self._interior_fields
            else ""
        )
        return f"{type(self).__name__}({list(self._sizes)}{interior})"


def _reference_derivative(level: ChebyshevLevel[Array], direction: int) -> Array:
    """:math:`\\partial / \\partial \\xi_d` on the level's tensor nodes,
    first direction fastest. Shape: (npts, npts)"""
    bkd, sizes = level.bkd(), level.mesh().npts_per_dim()
    factors = [bkd.eye(n) for n in sizes]
    factors[direction] = level.basis().derivative_matrix_1d(direction)
    # Node index i_0 + n_0 i_1 + ...: the last direction is the outer factor.
    matrix = factors[-1]
    for factor in factors[-2::-1]:
        matrix = bkd.kron(matrix, factor)
    return matrix


class PeriodicSeam(Generic[Array]):
    """The seam where a map closes reference direction ``d`` on itself
    (a ring's angle): mesh boundaries ``2 d`` and ``2 d + 1`` are one
    physical line inside the domain. Every field's value matches across
    it on the lower boundary's rows, and its derivative along the
    direction, :math:`\\partial u / \\partial \\xi_d`, on the upper's, so the
    solution and its gradient are continuous there. The reference
    derivative, not a physical one, because it is the same quantity on
    both sides for any map that closes on itself.

    Satisfies :class:`InteriorClosureProtocol`.

    Parameters
    ----------
    direction : int
        The reference direction the map closes.
    """

    def __init__(self, direction: int) -> None:
        self._direction = direction

    def rows(
        self, level: ChebyshevLevel[Array], layout: StateLayout
    ) -> PeriodicRows[Array]:
        """The seam's rows on every field of ``layout``.

        Raises
        ------
        ValueError
            If the direction is not one of the level's reference
            directions, or a field is not on the level's mesh.
        """
        mesh, direction = level.mesh(), self._direction
        if not 0 <= direction < len(mesh.npts_per_dim()):
            raise ValueError(
                f"the seam's direction {direction} is not a reference direction "
                f"of a mesh with {len(mesh.npts_per_dim())}"
            )
        elsewhere = [
            name for name in layout.names() if level.field_mesh(name) is not mesh
        ]
        if elsewhere:
            raise ValueError(
                f"a seam joins the level's mesh; {elsewhere} live on their own"
            )
        derivative = _reference_derivative(level, direction)
        lower = mesh.boundary_indices(2 * direction)
        upper = mesh.boundary_indices(2 * direction + 1)
        parts = [
            PeriodicPart(lower, upper, derivative, component)
            for component in range(layout.nfields())
        ]
        return PeriodicRows(level.bkd(), layout, mesh.reference_points(), parts)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._direction})"


def _ring_levels(
    domain: DomainProtocol,
    transform: TransformProtocol[Array],
    sizes: Sequence[int],
    bkd: Backend[Array],
) -> List[ChebyshevLevel[Array]]:
    """Levels of a ring meshed by ``transform`` from (radius, angle): the
    inner and outer curves are reference boundaries 0 and 1, the angle's
    two ends a seam."""
    parts = {"inner": (0,), "outer": (1,)}
    closures = (PeriodicSeam[Array](1),)
    levels: List[ChebyshevLevel[Array]] = []
    for n in sizes:
        mesh = TransformedMesh((n, n), bkd, transform)
        basis = ChebyshevBasis(mesh, bkd)
        levels.append(ChebyshevLevel(domain, mesh, basis, parts, closures))
    return levels


class ChebyshevAnnulus(Generic[Array]):
    """Chebyshev-Gauss-Lobatto meshes of an :class:`~pyapprox.pde.
    verification.domains.Annulus` through the polar map, ``n`` nodes in
    the radius and in the angle :math:`\\theta \\in [0, 2\\pi]` at each
    level; the angle's two ends, the segment :math:`y = 0`, :math:`x > 0`,
    are joined by a :class:`PeriodicSeam`.

    The nodes cluster at both ends of each direction: at the two circles,
    and in the angle at the seam, coarsest opposite it.

    Parameters
    ----------
    sizes : Sequence[int]
        Nodes per direction, one entry per level.
    bkd : Backend
        The backend of the meshes, coefficients and states.
    """

    def __init__(self, sizes: Sequence[int], bkd: Backend[Array]) -> None:
        self._sizes, self._bkd = tuple(sizes), bkd

    def levels(self, domain: DomainProtocol) -> List[ChebyshevLevel[Array]]:
        """One level per size.

        Raises
        ------
        TypeError
            If ``domain`` is not an ``Annulus``.
        ValueError
            If its center is not the origin (the polar map is about it).
        """
        if not isinstance(domain, Annulus):
            raise TypeError(
                f"ChebyshevAnnulus meshes an Annulus, got {type(domain).__name__}"
            )
        if domain.center() != (0.0, 0.0):
            raise ValueError(
                f"ChebyshevAnnulus meshes annuli about the origin, got center "
                f"{domain.center()}"
            )
        polar = PolarTransform(domain.radii(), (0.0, 2.0 * math.pi), self._bkd)
        return _ring_levels(domain, polar, self._sizes, self._bkd)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._sizes)})"


class ChebyshevEllipticalAnnulus(Generic[Array]):
    """Chebyshev-Gauss-Lobatto meshes of an :class:`~pyapprox.pde.
    verification.domains.EllipticalAnnulus` through the elliptical map
    :math:`x = a\\cosh u \\cos v`, :math:`y = a\\sinh u \\sin v`, ``n``
    nodes in :math:`u` and in :math:`v \\in [0, 2\\pi]` at each level; the
    two ends of :math:`v` are joined by a :class:`PeriodicSeam`.

    Parameters
    ----------
    sizes : Sequence[int]
        Nodes per direction, one entry per level.
    bkd : Backend
        The backend of the meshes, coefficients and states.
    """

    def __init__(self, sizes: Sequence[int], bkd: Backend[Array]) -> None:
        self._sizes, self._bkd = tuple(sizes), bkd

    def levels(self, domain: DomainProtocol) -> List[ChebyshevLevel[Array]]:
        """One level per size.

        Raises
        ------
        TypeError
            If ``domain`` is not an ``EllipticalAnnulus``.
        """
        if not isinstance(domain, EllipticalAnnulus):
            raise TypeError(
                "ChebyshevEllipticalAnnulus meshes an EllipticalAnnulus, got "
                f"{type(domain).__name__}"
            )
        elliptical = EllipticalTransform(
            domain.radii(), (0.0, 2.0 * math.pi), domain.focal(), self._bkd
        )
        return _ring_levels(domain, elliptical, self._sizes, self._bkd)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._sizes)})"


def _parsed(
    problem: ManufacturedProblem, expression: Value, allowed: Sequence[sp.Symbol]
) -> sp.Expr:
    """``expression`` parsed in the problem's names, constants substituted,
    refused if it uses anything but the coordinates and ``allowed``."""
    names = problem.names()
    parsed = names.substitute_constants(parse(expression, names))
    permitted = set(coordinates(problem.coordinates())) | set(allowed)
    extra = parsed.free_symbols - permitted
    if extra:
        raise ValueError(
            f"{expression!r} may depend only on "
            f"{sorted(map(str, permitted))}; got {sorted(map(str, extra))}"
        )
    return parsed


class _Field(Generic[Array]):
    """``f(coords, time)`` of one or more expressions, stacked: shape
    ``(nexpressions * npts,)``; time dependent exactly when an expression
    uses ``t``. Satisfies ``TimeAwareCallableProtocol``."""

    def __init__(
        self,
        expressions: Sequence[sp.Expr],
        problem: ManufacturedProblem,
        bkd: Backend[Array],
    ) -> None:
        self._points = PointExpressions(
            list(expressions), problem.names(), problem.coordinates(), bkd
        )
        self._bkd = bkd
        self._timed = any(time() in e.free_symbols for e in expressions)

    def __call__(self, coords: Array, time: float) -> Array:
        return self._bkd.reshape(self._points(coords, t=time), (-1,))

    def is_time_dependent(self) -> bool:
        return self._timed


class _BoundaryValues(Generic[Array]):
    """An expression at a side's nodes, with their outward normals, as a
    nodal coefficient of the side."""

    def __init__(
        self,
        expression: sp.Expr,
        problem: ManufacturedProblem,
        coords: Array,
        normals: Array,
        bkd: Backend[Array],
    ) -> None:
        self._points = PointExpressions(
            [expression], problem.names(), problem.coordinates(), bkd
        )
        self._coords, self._normals = coords, normals
        self._timed = time() in expression.free_symbols

    def values(self, time: float) -> Array:
        """The values at the side's nodes. Shape: (nboundary,)"""
        return self._points(self._coords, normals=self._normals, t=time)[0]

    def is_time_dependent(self) -> bool:
        return self._timed

    def version(self) -> int:
        """Zero: the expression is not rewritten."""
        return 0


def side_nodes(level: ChebyshevLevel[Array], part: str) -> Tuple[Array, Array]:
    """The node indices of boundary part ``part``, shape ``(nboundary,)``,
    and their outward unit normals, shape ``(nboundary, ncoords)``. A node
    two of the part's mesh boundaries share is taken once, with the
    normal of the first.

    Raises
    ------
    ValueError
        If the domain has no such part.
    """
    bkd, mesh = level.bkd(), level.mesh()
    boundaries = level.boundary_ids(part)
    if len(boundaries) == 1:
        return (
            mesh.boundary_indices(boundaries[0]),
            mesh.boundary_normals(boundaries[0]),
        )
    seen: set[int] = set()
    kept_indices: List[Array] = []
    kept_normals: List[Array] = []
    for boundary in boundaries:
        indices = mesh.boundary_indices(boundary)
        fresh = []
        for position in range(indices.shape[0]):
            node = bkd.to_int(indices[position])
            if node not in seen:
                seen.add(node)
                fresh.append(position)
        keep = bkd.asarray(fresh, dtype=bkd.int64_dtype())
        kept_indices.append(indices[keep])
        kept_normals.append(mesh.boundary_normals(boundary)[keep])
    return bkd.concatenate(kept_indices), bkd.concatenate(kept_normals, axis=0)


def side_values(
    problem: ManufacturedProblem,
    level: ChebyshevLevel[Array],
    part: str,
    expression: Value,
) -> Union[float, _BoundaryValues[Array]]:
    """An expression in the coordinates, the outward normal, ``t`` and the
    problem's constants at side ``part``'s nodes, as a coefficient of the
    side's rows; a constant as a float. Declared time-dependent exactly
    when the expression uses ``t``.

    Raises
    ------
    ValueError
        If the box has no such side.
    """
    names = problem.names()
    parsed = names.substitute_constants(parse(expression, names))
    if parsed.is_number:
        return float(parsed)
    indices, normals = side_nodes(level, part)
    coords = level.mesh().points()[:, indices]
    return _BoundaryValues(parsed, problem, coords, normals.T, level.bkd())


def field(
    problem: ManufacturedProblem, expression: Value, level: ChebyshevLevel[Array]
) -> Union[float, NodalSupplier[Array]]:
    """A scalar coefficient at the level's nodes, from an expression in the
    coordinates, ``t`` and the problem's constants; a constant as a float.
    Declared time-dependent exactly when the expression uses ``t``.

    Raises
    ------
    ValueError
        If the expression uses anything else.
    """
    return _nodal_coefficient(problem, expression, level, level.mesh())


def _nodal_coefficient(
    problem: ManufacturedProblem,
    expression: Value,
    level: ChebyshevLevel[Array],
    mesh: LevelMeshProtocol[Array],
) -> Union[float, NodalSupplier[Array]]:
    """:func:`field` at ``mesh``'s nodes."""
    parsed = _parsed(problem, expression, (time(),))
    if parsed.is_number:
        return float(parsed)
    # _Field declares its own time dependence, from the expression.
    return NodalSupplier(_Field([parsed], problem, level.bkd()), mesh.points())


def nodal_vector(
    problem: ManufacturedProblem,
    expressions: Sequence[Value],
    level: ChebyshevLevel[Array],
) -> List[Array]:
    """A vector coefficient fixed in time, one array of nodal values per
    coordinate, as a physics taking a fixed velocity reads it.

    Raises
    ------
    ValueError
        If there is not one expression per coordinate, or one uses
        anything but the coordinates and constants.
    """
    coords = problem.coordinates()
    if len(expressions) != len(coords):
        raise ValueError(
            f"one expression per coordinate {list(coords)}, got {len(expressions)}"
        )
    bkd, points = level.bkd(), level.mesh().points()
    values = []
    for expression in expressions:
        parsed = _parsed(problem, expression, ())
        if parsed.is_number:
            values.append(bkd.full((level.npts(),), float(parsed)))
        else:
            values.append(_Field([parsed], problem, bkd)(points, 0.0))
    return values


def forcing(
    problem: ManufacturedProblem, row: str, level: ChebyshevLevel[Array]
) -> Union[float, NodalSupplier[Array]]:
    """The forcing of ``row`` at the nodes of the mesh its field lives on."""
    return _nodal_coefficient(
        problem, problem.forcing(row), level, level.field_mesh(row)
    )


def stacked_forcing(
    problem: ManufacturedProblem, level: ChebyshevLevel[Array]
) -> NodalSupplier[Array]:
    """The forcing of every field at the level's nodes, stacked in the
    problem's field order, as a vector physics takes it: shape
    ``(nfields * npts,)``. Declared time-dependent exactly when a forcing
    uses ``t``.

    Raises
    ------
    ValueError
        If a field lives on a mesh of its own; such fields take their
        forcings one at a time (:func:`forcing`).
    """
    elsewhere = [
        row for row in problem.fields() if level.field_mesh(row) is not level.mesh()
    ]
    if elsewhere:
        raise ValueError(
            f"{elsewhere} live on their own meshes; take each field's forcing "
            "with forcing()"
        )
    expressions = [
        _parsed(problem, problem.forcing(row), (time(),)) for row in problem.fields()
    ]
    supplier = _Field(expressions, problem, level.bkd())
    return NodalSupplier(supplier, level.mesh().points())


@runtime_checkable
class NormalTermProtocol(Protocol[Array]):
    r"""How a field's Neumann and Robin sides measure the outward flux
    :math:`q(u)\cdot n` the problem prescribes: an operator :math:`N` on a
    side's nodes, and the sign :math:`s` with :math:`N = s\,q\cdot n`."""

    def operator(
        self, bkd: Backend[Array], indices: Array, normals: Array
    ) -> RobinNormalTermProtocol[Array]:
        """:math:`N` on the side's mesh nodes ``indices``, shape
        ``(nboundary,)``, with outward unit ``normals``, shape
        ``(nboundary, ndim)``."""
        ...

    def sign(self) -> float:
        """:math:`s`, 1 or -1."""
        ...


class FluxTerm(Generic[Array]):
    r"""A flux provider's outward normal flux, :math:`N = q\cdot n`
    (:math:`s = 1`), through
    :class:`~pyapprox.pde.collocation.boundary.normal_operators.
    FluxNormalOperator`.

    Parameters
    ----------
    flux : FluxProviderProtocol
        Maps the whole stacked state to the field's flux.

    Raises
    ------
    TypeError
        If ``flux`` does not satisfy ``FluxProviderProtocol``.
    """

    def __init__(self, flux: FluxProviderProtocol[Array]) -> None:
        if not isinstance(flux, FluxProviderProtocol):
            raise TypeError(
                f"flux must satisfy FluxProviderProtocol, got {type(flux).__name__}"
            )
        self._flux = flux

    def operator(
        self, bkd: Backend[Array], indices: Array, normals: Array
    ) -> FluxNormalOperator[Array]:
        return FluxNormalOperator(bkd, indices, normals, self._flux)

    def sign(self) -> float:
        return 1.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._flux!r})"


class TractionTerm(Generic[Array]):
    r"""One component of the Cauchy traction :math:`t_i = \sigma_{ij}
    n_j`, through :class:`~pyapprox.pde.collocation.boundary.
    normal_operators.TractionNormalOperator`. A problem stating
    :math:`-\nabla\cdot\sigma = f` row by row as the divergence of
    :math:`q_i = -\sigma_{i\cdot}` prescribes :math:`q_i\cdot n = -t_i`, so
    :math:`s = -1`.

    Parameters
    ----------
    stress : CauchyStressSourceProtocol
        The Cauchy stress, usually the physics.
    component : int
        The traction component :math:`i`.

    Raises
    ------
    TypeError
        If ``stress`` does not satisfy ``CauchyStressSourceProtocol``.
    """

    def __init__(
        self, stress: CauchyStressSourceProtocol[Array], component: int
    ) -> None:
        if not isinstance(stress, CauchyStressSourceProtocol):
            raise TypeError(
                "stress must satisfy CauchyStressSourceProtocol, got "
                f"{type(stress).__name__}"
            )
        self._stress, self._component = stress, component

    def operator(
        self, bkd: Backend[Array], indices: Array, normals: Array
    ) -> TractionNormalOperator[Array]:
        return TractionNormalOperator(
            bkd, indices, normals, self._stress, self._component
        )

    def sign(self) -> float:
        return -1.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(component={self._component})"


class PiolaTractionTerm(Generic[Array]):
    r"""One component of the nominal traction :math:`t_i = P_{iJ} N_J`,
    :math:`P` the first Piola-Kirchhoff stress, through
    :class:`~pyapprox.pde.collocation.boundary.hyperelastic_traction.
    HyperelasticTractionNormalOperator`. A problem stating
    :math:`-\nabla\cdot P = f` row by row as the divergence of
    :math:`q_i = -P_{i\cdot}` prescribes :math:`q_i\cdot N = -t_i`, so
    :math:`s = -1`.

    Parameters
    ----------
    stress : PK1StressSourceProtocol
        The first Piola-Kirchhoff stress, usually the physics.
    component : int
        The traction component :math:`i`.

    Raises
    ------
    TypeError
        If ``stress`` does not satisfy ``PK1StressSourceProtocol``.
    """

    def __init__(self, stress: PK1StressSourceProtocol[Array], component: int) -> None:
        if not isinstance(stress, PK1StressSourceProtocol):
            raise TypeError(
                "stress must satisfy PK1StressSourceProtocol, got "
                f"{type(stress).__name__}"
            )
        self._stress, self._component = stress, component

    def operator(
        self, bkd: Backend[Array], indices: Array, normals: Array
    ) -> HyperelasticTractionNormalOperator[Array]:
        return HyperelasticTractionNormalOperator(
            bkd, indices, normals, self._stress, self._component
        )

    def sign(self) -> float:
        return -1.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(component={self._component})"


FieldFlux = Union[FluxProviderProtocol[Array], NormalTermProtocol[Array]]


_BuiltPart = Union[DirichletPart[Array], RobinPart[Array], PeriodicPart[Array], None]


class _CollocationCondition(Generic[Array]):
    """Builds the rows of one row's condition on one side, on the row's
    state component; ``None`` for the upper side of a periodic pair, whose
    rows the lower side builds, and for an unprescribed side, whose nodes
    keep the row's own equation."""

    def __init__(
        self,
        problem: ManufacturedProblem,
        level: ChebyshevLevel[Array],
        part: str,
        row: str,
        component: int,
        term: Optional[NormalTermProtocol[Array]],
    ) -> None:
        self._problem, self._level = problem, level
        self._part, self._row, self._term = part, row, term
        self._component = component
        self._indices, self._normals = side_nodes(level, part)

    def _values(self, expression: sp.Expr) -> Union[float, _BoundaryValues[Array]]:
        return side_values(self._problem, self._level, self._part, expression)

    def _data(self) -> Union[float, _BoundaryValues[Array]]:
        return self._values(self._problem.boundary_data(self._part)[self._row])

    def _normal_term(self) -> NormalTermProtocol[Array]:
        if self._term is None:
            raise ValueError(
                f"the condition on {self._part!r} prescribes a flux of "
                f"{self._row!r}; pass the flux provider or normal term whose "
                "normal flux it prescribes"
            )
        return self._term

    def _robin_part(self, alpha: Union[float, sp.Expr]) -> RobinPart[Array]:
        """Rows :math:`\\alpha' u + s N(u) = g` for the core's
        :math:`q\\cdot n - \\alpha u = g`: with :math:`N = s\\,q\\cdot n`,
        :math:`s N = q\\cdot n`, and :math:`\\alpha' = -\\alpha`."""
        term = self._normal_term()
        operator = term.operator(self._level.bkd(), self._indices, self._normals)
        spring = 0.0 if alpha == 0 else self._values(-alpha)
        return RobinPart(
            self._indices,
            operator,
            spring,
            term.sign(),
            self._data(),
            self._component,
        )

    def dirichlet(self, condition: Dirichlet) -> _BuiltPart[Array]:
        return DirichletPart(self._indices, self._data(), self._component)

    def neumann(self, condition: Neumann) -> _BuiltPart[Array]:
        return self._robin_part(0.0)

    def robin(self, condition: Robin) -> _BuiltPart[Array]:
        names = self._problem.names()
        return self._robin_part(names.substitute_constants(condition.alpha(names)))

    def unprescribed(self, condition: Unprescribed) -> _BuiltPart[Array]:
        return None

    def periodic(self, condition: Periodic) -> _BuiltPart[Array]:
        domain = self._level.domain()
        if not isinstance(domain, Box):
            raise TypeError(
                f"periodic conditions pair opposite sides of a Box; "
                f"{self._part!r} is a boundary part of {domain!r}"
            )
        coordinate, sign = domain.side(self._part)
        sides = domain.boundaries()
        index = sides.index(self._part)
        lower, upper = (index, index + 1) if sign < 0 else (index - 1, index)
        other = sides[upper] if sign < 0 else sides[lower]
        if not isinstance(self._problem.conditions(other)[self._row], Periodic):
            raise ValueError(
                f"{self._part!r} is periodic but {other!r}, across "
                f"{coordinate}, is not; a periodic pair needs both sides"
            )
        if sign > 0:
            return None
        mesh = self._level.mesh()
        dim = domain.coordinates().index(coordinate)
        return PeriodicPart(
            mesh.boundary_indices(lower),
            mesh.boundary_indices(upper),
            self._level.basis().derivative_matrix(1, dim),
            self._component,
        )


@dataclass(frozen=True)
class BoundaryDeclarations(Generic[Array]):
    """The rows of a problem's conditions, as a ``CollocationSystem``
    takes them: the essential rows, and the closures in the order that
    settles shared nodes (later wins)."""

    essential: Optional[DirichletRows[Array]]
    closures: Tuple[ClosureRowsProtocol[Array], ...]


def _normal_terms(
    fields: Sequence[str], fluxes: Optional[Mapping[str, FieldFlux[Array]]]
) -> Dict[str, NormalTermProtocol[Array]]:
    """Each field's normal term, a flux provider wrapped as a
    :class:`FluxTerm`."""
    given = {} if fluxes is None else dict(fluxes)
    unknown = set(given) - set(fields)
    if unknown:
        raise ValueError(
            f"fluxes name fields {sorted(unknown)} the problem does not have; "
            f"its fields are {list(fields)}"
        )
    terms: Dict[str, NormalTermProtocol[Array]] = {}
    for row, flux in given.items():
        if isinstance(flux, NormalTermProtocol):
            terms[row] = flux
        elif isinstance(flux, FluxProviderProtocol):
            terms[row] = FluxTerm(flux)
        else:
            raise TypeError(
                f"the flux of {row!r} must satisfy FluxProviderProtocol or "
                f"NormalTermProtocol, got {type(flux).__name__}"
            )
    return terms


def boundary_conditions(
    problem: ManufacturedProblem,
    level: ChebyshevLevel[Array],
    fluxes: Optional[Mapping[str, FieldFlux[Array]]] = None,
) -> BoundaryDeclarations[Array]:
    """The rows of every side's condition on every field, with data from
    the problem's expressions.

    The state stacks one component per field of the problem, in the
    problem's field order (a scalar problem has one); a field's rows act
    on its component. Fields that take no boundary conditions get none.

    Parameters
    ----------
    problem : ManufacturedProblem
        Gives the fields, the conditions and their data.
    level : ChebyshevLevel
        The mesh the rows live on.
    fluxes : Mapping[str, FluxProviderProtocol or NormalTermProtocol], optional
        By field, how that field's Neumann and Robin sides measure the
        outward flux they prescribe. A flux provider (the physics' own, or
        another such as :class:`~pyapprox.pde.collocation.physics.
        advection_diffusion.AdvectionDiffusionFlux`) maps the whole
        stacked state to that field's flux and is used through
        :class:`FluxTerm`; a normal term, such as :class:`TractionTerm`,
        supplies the side's operator and its sign. Needed only for a field
        with such a side.

    Raises
    ------
    ValueError
        If ``fluxes`` names a field the problem does not have, a side
        prescribes a flux its field was not given, only one side of a
        coordinate is periodic, or a field taking boundary conditions lives
        on a mesh of its own (its nodes need not reach the boundary).
    TypeError
        If a flux satisfies neither ``FluxProviderProtocol`` nor
        ``NormalTermProtocol``.
    """
    fields = problem.fields()
    elsewhere = [
        row
        for row in problem.boundary_rows()
        if level.field_mesh(row) is not level.mesh()
    ]
    if elsewhere:
        raise ValueError(
            f"boundary rows live on the level's mesh; {elsewhere} take boundary "
            "conditions but live on their own meshes"
        )
    terms = _normal_terms(fields, fluxes)
    dirichlet: List[DirichletPart[Array]] = []
    robin: List[RobinPart[Array]] = []
    periodic: List[PeriodicPart[Array]] = []
    for component, row in enumerate(fields):
        if row not in problem.boundary_rows():
            continue
        for part in level.domain().boundaries():
            visitor = _CollocationCondition(
                problem, level, part, row, component, terms.get(row)
            )
            built = problem.conditions(part)[row].accept(visitor)
            if isinstance(built, DirichletPart):
                dirichlet.append(built)
            elif isinstance(built, RobinPart):
                robin.append(built)
            elif isinstance(built, PeriodicPart):
                periodic.append(built)
    bkd, layout = level.bkd(), level.layout(fields)
    # The level's interior closures first, so a boundary part's own rows
    # win the nodes it shares with them (a ring's seam meets both circles).
    closures: List[ClosureRowsProtocol[Array]] = [
        closure.rows(level, layout) for closure in level.closures()
    ]
    if periodic:
        closures.append(
            PeriodicRows(bkd, layout, level.mesh().reference_points(), periodic)
        )
    if robin:
        # After the periodic rows, so a side's own condition wins the
        # nodes it shares with a periodic side.
        closures.append(RobinRows(bkd, layout, robin))
    essential = DirichletRows(bkd, layout, dirichlet) if dirichlet else None
    return BoundaryDeclarations(essential, tuple(closures))


def collocation_system(
    physics: CollocationInteriorOperatorProtocol[Array],
    problem: ManufacturedProblem,
    level: ChebyshevLevel[Array],
    fluxes: Optional[Mapping[str, FieldFlux[Array]]] = None,
) -> CollocationSystem[Array]:
    """``physics`` closed by the rows of :func:`boundary_conditions`; its
    state stacks the problem's fields in the problem's order."""
    declarations = boundary_conditions(problem, level, fluxes)
    return CollocationSystem(physics, declarations.essential, declarations.closures)


class SystemFactoryProtocol(Protocol):
    """Builds the physics under test, with its conditions, on a level."""

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        """The system whose solution approximates ``problem``'s."""
        ...


@runtime_checkable
class InitialGuessProtocol(Protocol):
    """Where a level's nonlinear solve starts."""

    def guess(
        self,
        problem: ManufacturedProblem,
        level: ChebyshevLevel[Array],
        system: CollocationSystem[Array],
    ) -> Array:
        """The state the solve of ``system`` on ``level`` starts from, for
        example the solution of a linearization of its physics. Shape:
        (nstates, 1)"""
        ...


class ConstantGuess:
    """Every state entry starts from one value.

    Parameters
    ----------
    value : float
        The value.
    """

    def __init__(self, value: float) -> None:
        self._value = value

    def guess(
        self,
        problem: ManufacturedProblem,
        level: ChebyshevLevel[Array],
        system: CollocationSystem[Array],
    ) -> Array:
        """``value`` at every state entry. Shape: (nstates, 1)"""
        return level.bkd().full((system.nstates(), 1), self._value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._value!r})"


@dataclass(frozen=True)
class _ErrorGrid(Generic[Array]):
    """A tensor Gauss-Legendre grid of the reference cube with twice the
    level's nodes per direction, first dimension fastest, mapped onto the
    domain by the level mesh's map.

    ``points`` are physical, shape (ndim, nquad); ``weights`` integrate
    over the domain (the reference weights times the map's
    :math:`|\\det J|`), shape (nquad,); ``abscissas`` are the reference
    points of each direction, shape (nquad_d,) each, from which
    :meth:`interpolation` builds a mesh's map from nodal values to their
    Lagrange interpolant at the points, shape (nquad, npts). Every field
    is measured on this one grid, whichever mesh it lives on. With
    :math:`2N` points per direction the rule is exact
    to degree :math:`4N - 1` in the reference coordinates, so on a box
    (an affine map) the squared interpolant (degree :math:`2N - 2`) is
    integrated exactly and :math:`u^*` far more accurately than the
    error being measured; a curved map adds its own smooth factors.
    """

    points: Array
    weights: Array
    abscissas: Tuple[Array, ...]

    def interpolation(
        self, mesh: LevelMeshProtocol[Array], bkd: Backend[Array]
    ) -> Array:
        """From ``mesh``'s nodal values to their interpolant at the points,
        first dimension fastest. Shape: (nquad, mesh.npts())"""
        matrix = univariate_lagrange_polynomial(
            mesh.reference_nodes_1d(0), self.abscissas[0], bkd
        )
        for dim in range(1, len(self.abscissas)):
            lagrange = univariate_lagrange_polynomial(
                mesh.reference_nodes_1d(dim), self.abscissas[dim], bkd
            )
            matrix = bkd.kron(lagrange, matrix)
        return matrix


def _error_grid(level: ChebyshevLevel[Array]) -> _ErrorGrid[Array]:
    bkd, mesh = level.bkd(), level.mesh()
    coords: List[Array] = []
    abscissas: List[Array] = []
    weights: Optional[Array] = None
    for dim in range(len(mesh.npts_per_dim())):
        nquad = 2 * mesh.reference_nodes_1d(dim).shape[0]
        legendre = LegendrePolynomial1D(bkd)
        legendre.set_nterms(nquad)
        reference, probability = GaussQuadratureRule(legendre)(nquad)
        xi = reference[0, :]
        abscissas.append(xi)
        # Probability weights sum to 1; the reference interval has length 2.
        w = 2.0 * probability[:, 0]
        if weights is None:
            coords, weights = [xi], w
            continue
        previous = weights.shape[0]
        coords = [bkd.kron(bkd.ones((nquad,)), c) for c in coords]
        coords.append(bkd.kron(xi, bkd.ones((previous,))))
        weights = bkd.kron(w, weights)
    if weights is None:
        raise ValueError("a level has at least one dimension")
    reference_grid = bkd.stack(coords, axis=0)
    transform = mesh.transform()
    volume = bkd.abs(transform.jacobian_determinant(reference_grid))
    return _ErrorGrid(
        transform.map_to_physical(reference_grid), weights * volume, tuple(abscissas)
    )


def mean_value_rows(
    problem: ManufacturedProblem, level: ChebyshevLevel[Array], row: str
) -> MeanValueRows[Array]:
    """A closure row fixing field ``row``'s integral over the domain to the
    exact solution's, replacing the row of the field's first node: for a
    field the equations determine only up to a constant (the pressure
    with the velocity prescribed on every side).

    Both integrals use the level's error grid: the weights integrate the
    interpolant on the field's own nodes, and the value is the grid's
    quadrature of the exact field, so the condition is consistent with
    the exact solution to the grid's accuracy.

    Raises
    ------
    ValueError
        If ``row`` is not one of the problem's fields.
    """
    fields = problem.fields()
    if row not in fields:
        raise ValueError(f"{row!r} is not one of the problem's fields {list(fields)}")
    bkd = level.bkd()
    grid = _error_grid(level)
    weights = grid.weights @ grid.interpolation(level.field_mesh(row), bkd)
    exact = PointExpressions(
        [problem.solution(row)], problem.names(), problem.coordinates(), bkd
    )(grid.points)[0]
    value = bkd.to_float(bkd.sum(grid.weights * exact))
    return MeanValueRows(bkd, level.layout(fields), fields.index(row), weights, value)


class CollocationApproximation(Generic[Array]):
    """One level's collocation solution of a problem's fields, with errors
    against the exact solution, each beside the exact solution's norm in
    the same norm (a :class:`~pyapprox.pde.verification.study.
    Measurement`).

    The state stacks one component per field in the problem's field
    order, each at the nodes of the mesh it lives on
    (:meth:`ChebyshevLevel.layout`). Errors combine over the fields asked
    for (default all): the L2 error is the root of the sum of their
    squared L2 errors, the nodal error their largest. The measures
    (:func:`collocation_l2_error`, :func:`collocation_nodal_error`) read
    it.

    Parameters
    ----------
    problem : ManufacturedProblem
        The problem solved.
    level : ChebyshevLevel
        The level it was solved on.
    state : Array
        The nodal values, field by field. Shape: (nstates,)

    Raises
    ------
    ValueError
        If ``state`` is not one value per node per field.
    """

    def __init__(
        self,
        problem: ManufacturedProblem,
        level: ChebyshevLevel[Array],
        state: Array,
    ) -> None:
        bkd = level.bkd()
        fields = problem.fields()
        layout = level.layout(fields)
        expected = layout.nstates()
        if state.shape != (expected,):
            raise ValueError(
                f"the state must have shape ({expected},), one value per node "
                f"for each of {list(fields)}; got {tuple(state.shape)}"
            )
        solution = PointExpressions(
            [problem.solution(row) for row in fields],
            problem.names(),
            problem.coordinates(),
            bkd,
        )
        grid = _error_grid(level)
        meshes = [level.field_mesh(row) for row in fields]
        if level.is_uniform():
            exact = bkd.reshape(solution(level.mesh().points()), (-1,))
            interpolations = [grid.interpolation(level.mesh(), bkd)] * len(fields)
        else:
            # Each field's exact values at its own mesh's nodes.
            exact = bkd.hstack(
                [solution(mesh.points())[k] for k, mesh in enumerate(meshes)]
            )
            interpolations = [grid.interpolation(mesh, bkd) for mesh in meshes]
        pieces = [
            slice(layout.offset(k), layout.offset(k) + layout.npts(k))
            for k in range(layout.nfields())
        ]
        # Per field: the exact nodal values, and the exact values and the
        # interpolant's on the error grid.
        self._nodal = [(exact[piece], state[piece]) for piece in pieces]
        on_grid = solution(grid.points)
        self._on_grid = [
            (on_grid[k], matrix @ state[piece])
            for k, (matrix, piece) in enumerate(zip(interpolations, pieces))
        ]
        self._weights = grid.weights
        self._fields = tuple(fields)
        self._bkd, self._state, self._h = bkd, state, level.h()
        self._nodes = max(level.mesh().npts_per_dim())

    def h(self) -> float:
        """The largest gap between neighboring nodes (:meth:`ChebyshevLevel.h`)."""
        return self._h

    def nodes(self) -> int:
        """The most nodes along any direction, :math:`N`."""
        return self._nodes

    def state(self) -> Array:
        """The nodal values, field by field. Shape: (nstates,)"""
        return self._state

    def _chosen(self, fields: Optional[Sequence[str]]) -> List[int]:
        """The indices of ``fields`` (default all).

        Raises
        ------
        ValueError
            If ``fields`` is empty or names an unknown field.
        """
        if fields is None:
            return list(range(len(self._fields)))
        unknown = set(fields) - set(self._fields)
        if not fields or unknown:
            raise ValueError(
                f"measure some of the fields {list(self._fields)}, got {list(fields)}"
            )
        return [k for k, name in enumerate(self._fields) if name in fields]

    def l2(self, fields: Optional[Sequence[str]] = None) -> Measurement:
        r""":math:`\|u^* - u_h\|_{L^2}` and :math:`\|u^*\|_{L^2}` over
        ``fields`` (default all), :math:`u_h` the Lagrange interpolant of
        the nodal values, by Gauss-Legendre quadrature with twice the
        level's nodes per direction against :math:`u^*` itself. It sees
        the error between the nodes, which a quadrature at the nodes would
        not.

        Raises
        ------
        ValueError
            If ``fields`` is empty or names an unknown field.
        """
        bkd = self._bkd
        error, norm = 0.0, 0.0
        for k in self._chosen(fields):
            exact, interpolant = self._on_grid[k]
            error += bkd.to_float(bkd.sum(self._weights * (exact - interpolant) ** 2))
            norm += bkd.to_float(bkd.sum(self._weights * exact**2))
        return Measurement(math.sqrt(error), math.sqrt(norm))

    def l2_error(self, fields: Optional[Sequence[str]] = None) -> float:
        r""":math:`\|u^* - u_h\|_{L^2}` over ``fields`` (default all); see
        :meth:`l2`."""
        return self.l2(fields).value

    def nodal(self, fields: Optional[Sequence[str]] = None) -> Measurement:
        r""":math:`\max_i |u^*(x_i) - u_h(x_i)|` and :math:`\max_i
        |u^*(x_i)|` over the collocation nodes of ``fields`` (default all).

        Raises
        ------
        ValueError
            If ``fields`` is empty or names an unknown field.
        """
        bkd = self._bkd
        error, norm = 0.0, 0.0
        for k in self._chosen(fields):
            exact, values = self._nodal[k]
            error = max(error, bkd.to_float(bkd.max(bkd.abs(exact - values))))
            norm = max(norm, bkd.to_float(bkd.max(bkd.abs(exact))))
        return Measurement(error, norm)

    def nodal_error(self, fields: Optional[Sequence[str]] = None) -> float:
        r""":math:`\max_i |u^*(x_i) - u_h(x_i)|` over ``fields`` (default
        all); see :meth:`nodal`."""
        return self.nodal(fields).value


def collocation_l2_error(
    fields: Optional[Sequence[str]] = None, label: Optional[str] = None
) -> Measure[CollocationApproximation[Array]]:
    """The L2 error of ``fields`` (default all) beside the exact
    solution's L2 norm; see :meth:`CollocationApproximation.l2`. Label
    ``"l2"`` or ``"l2(<fields>)"`` by default.

    Raises
    ------
    ValueError
        If ``fields`` is given empty.
    """
    chosen = None if fields is None else tuple(fields)
    return Measure(
        label or field_label("l2", chosen),
        lambda approximation: approximation.l2(chosen),
    )


def collocation_nodal_error(
    fields: Optional[Sequence[str]] = None, label: Optional[str] = None
) -> Measure[CollocationApproximation[Array]]:
    """The largest error at the collocation nodes of ``fields`` (default
    all); see :meth:`CollocationApproximation.nodal`. Label ``"nodal"``
    or ``"nodal(<fields>)"`` by default.

    Raises
    ------
    ValueError
        If ``fields`` is given empty.
    """
    chosen = None if fields is None else tuple(fields)
    return Measure(
        label or field_label("nodal", chosen),
        lambda approximation: approximation.nodal(chosen),
    )


class CollocationDiscretizer(Generic[Array]):
    """Solves a steady problem with this stack at the levels of a family.
    The factory's system stacks the problem's fields in the problem's
    order. A level is a
    :class:`~pyapprox.pde.verification.protocols.SpaceLevel`, the index
    of a level in the family.

    Each solve: the factory's system, the steady residual
    (``collocation_steady_residual``), and ``SteadySystem`` with a dense
    LU recomputed every step, from the initial guess's state.

    Parameters
    ----------
    levels : LevelFamilyProtocol
        The levels.
    factory : SystemFactoryProtocol
        The system of a problem on a level.
    nonlinear : NonlinearSolverProtocol, optional
        Default: Newton, ``atol = rtol = 1e-12``, 30 steps (a linear
        system is solved in one step).
    initial_guess : InitialGuessProtocol, optional
        Where each level's solve starts; default ``ConstantGuess(0.0)``.
        A physics whose Jacobian degenerates at zero (shallow ice, at
        zero thickness) starts elsewhere.

    Raises
    ------
    TypeError
        If ``initial_guess`` does not satisfy ``InitialGuessProtocol``.
    """

    def __init__(
        self,
        levels: LevelFamilyProtocol[Array],
        factory: SystemFactoryProtocol,
        nonlinear: Optional[NonlinearSolverProtocol[Array, Array]] = None,
        initial_guess: Optional[InitialGuessProtocol] = None,
    ) -> None:
        if initial_guess is not None and not isinstance(
            initial_guess, InitialGuessProtocol
        ):
            raise TypeError(
                "initial_guess must satisfy InitialGuessProtocol, got "
                f"{type(initial_guess).__name__}"
            )
        self._levels, self._factory = levels, factory
        self._initial_guess: InitialGuessProtocol = (
            ConstantGuess(0.0) if initial_guess is None else initial_guess
        )
        default: Newton[Array, Array] = Newton(atol=1e-12, rtol=1e-12, maxiters=30)
        self._nonlinear: NonlinearSolverProtocol[Array, Array] = (
            default if nonlinear is None else nonlinear
        )
        self._family: Optional[
            Tuple[DomainProtocol, Sequence[ChebyshevLevel[Array]]]
        ] = None

    def _levels_of(
        self, problem: ManufacturedProblem
    ) -> Sequence[ChebyshevLevel[Array]]:
        """The family's levels of the problem's domain, built once per
        domain.

        Raises
        ------
        ValueError
            If ``problem`` is transient or has no domain.
        """
        if problem.is_transient():
            raise ValueError(
                "CollocationDiscretizer solves steady problems; this one depends on t"
            )
        domain = problem.domain()
        if domain is None:
            raise ValueError("CollocationDiscretizer needs a domain; this is an ODE")
        if self._family is None or self._family[0] is not domain:
            self._family = (domain, self._levels.levels(domain))
        return self._family[1]

    def _level(
        self, problem: ManufacturedProblem, request: SpaceLevel
    ) -> ChebyshevLevel[Array]:
        """The level ``request`` names.

        Raises
        ------
        ValueError
            If the family has no such level (and as :meth:`_levels_of`).
        """
        levels = self._levels_of(problem)
        if request.index >= len(levels):
            raise ValueError(
                f"the family has {len(levels)} levels, no level {request.index}"
            )
        return levels[request.index]

    def requests(self, problem: ManufacturedProblem) -> List[SpaceLevel]:
        """One request per level of the family, coarsest first.

        Raises
        ------
        ValueError
            If ``problem`` is transient or has no domain.
        """
        return [SpaceLevel(index) for index in range(len(self._levels_of(problem)))]

    def sizes(self, problem: ManufacturedProblem, request: SpaceLevel) -> Sizes:
        """The level's largest gap between neighboring nodes (``H``), its
        degree (``DEGREE``, one less than the most nodes along any
        direction) and its number of unknowns (``NDOFS``), without
        solving.

        Raises
        ------
        ValueError
            As :meth:`requests`, or if the family has no such level.
        """
        level = self._level(problem, request)
        return Sizes(
            {
                H: level.h(),
                DEGREE: max(level.mesh().npts_per_dim()) - 1,
                NDOFS: level.layout(problem.fields()).nstates(),
            }
        )

    def solve(
        self, problem: ManufacturedProblem, request: SpaceLevel
    ) -> CollocationApproximation[Array]:
        """The solution at the level ``request`` names.

        Raises
        ------
        ValueError
            If ``problem`` is transient or has no domain, the family has
            no such level, the system's state is not one value per node
            per field, or the initial guess does not have shape
            (nstates, 1).
        RuntimeError
            If the solve does not converge.
        """
        level = self._level(problem, request)
        fields = problem.fields()
        system = self._factory.system(problem, level)
        bkd, nstates = level.bkd(), system.nstates()
        layout = level.layout(fields)
        if nstates != layout.nstates():
            raise ValueError(
                f"the system has {nstates} states, not one per node "
                f"({layout.nstates()} over {list(fields)})"
            )
        steady = SteadySystem(
            collocation_steady_residual(system),
            backend_state_space(bkd, nstates),
            RecomputeAlways(BackendDenseLU(bkd)),
        )
        initial = self._initial_guess.guess(problem, level, system)
        if tuple(initial.shape) != (nstates, 1):
            raise ValueError(
                f"the initial guess on {level!r} must have shape "
                f"({nstates}, 1); got {tuple(initial.shape)}"
            )
        point = steady.solve(self._nonlinear, initial)
        if not point.converged():
            raise RuntimeError(
                f"the solve on {level!r} did not converge: {point.info().message}"
            )
        return CollocationApproximation(problem, level, point.state()[:, 0])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._levels!r}, {self._factory!r})"
