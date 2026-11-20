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
  coordinate. A node two sides share follows the system's rule: essential
  rows win it, then the Robin rows, then the periodic rows. A case that
  builds its rows by hand (a wrong-model control) takes a side's nodes
  and data from :func:`side_nodes` and :func:`side_values`.
- :class:`CollocationDiscretizer` solves each level through
  ``collocation_steady_residual`` + ``SteadySystem`` with a dense LU,
  from the state an injected :class:`InitialGuessProtocol` gives
  (:class:`ConstantGuess` zero by default), and returns
  :class:`CollocationApproximation` objects, whose L2 errors are measured
  by Gauss quadrature mapped onto the domain and nodal errors at the
  collocation nodes.
- :class:`DecaysExponentially` judges a spectral study, a
  :class:`NodeStudy` of errors against node counts from
  :func:`run_node_study`: the error falls exponentially in the number of
  nodes, which no algebraic rate states. The core's study records mesh
  sizes only, so this pair lives here until the core has its own.

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
from pyapprox.pde.collocation.basis import (
    ChebyshevBasis1D,
    ChebyshevBasis2D,
    ChebyshevBasis3D,
)
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
from pyapprox.pde.collocation.boundary.normal_operators import (
    CauchyStressSourceProtocol,
    FluxNormalOperator,
    TractionNormalOperator,
)
from pyapprox.pde.collocation.boundary.periodic_rows import (
    PeriodicPart,
    PeriodicRows,
)
from pyapprox.pde.collocation.mesh import (
    TransformedMesh1D,
    TransformedMesh2D,
    TransformedMesh3D,
)
from pyapprox.pde.collocation.mesh.transforms.affine import (
    AffineTransform1D,
    AffineTransform2D,
    AffineTransform3D,
)
from pyapprox.pde.collocation.physics.coefficients import NodalSupplier
from pyapprox.pde.collocation.protocols.basis import TensorProductBasisProtocol
from pyapprox.pde.collocation.protocols.boundary import FluxProviderProtocol
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
)
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
)
from pyapprox.pde.verification.domains import Box, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.study import CheckResult, ErrorMeasureProtocol
from pyapprox.polynomials.univariate.globalpoly.jacobi import LegendrePolynomial1D
from pyapprox.polynomials.univariate.globalpoly.quadrature import GaussQuadratureRule
from pyapprox.polynomials.univariate.lagrange import univariate_lagrange_polynomial
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.expressions.parsing import Value, parse
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton, NonlinearSolverProtocol


class LevelMapProtocol(Protocol[Array]):
    """What the adapter reads from a level mesh's map of the reference
    cube :math:`[-1, 1]^d` onto the domain."""

    def map_to_physical(self, reference_pts: Array) -> Array:
        """Shape: (ndim, npts) to (ndim, npts)"""
        ...

    def jacobian_determinant(self, reference_pts: Array) -> Array:
        """Shape: (ndim, npts) to (npts,)"""
        ...


class LevelMeshProtocol(Protocol[Array]):
    """What the adapter reads from a level's mesh: a tensor
    Chebyshev-Gauss-Lobatto mesh of the reference cube, first dimension
    fastest, mapped onto the domain."""

    def ndim(self) -> int: ...

    def npts(self) -> int: ...

    def npts_per_dim(self) -> Tuple[int, ...]: ...

    def points(self) -> Array:
        """Physical coordinates. Shape: (ndim, npts)"""
        ...

    def reference_points(self) -> Array:
        """Reference coordinates. Shape: (ndim, npts)"""
        ...

    def reference_nodes_1d(self, dim: int) -> Array:
        """Reference nodes of dimension ``dim``. Shape: (npts_dim,)"""
        ...

    def boundary_indices(self, boundary_id: int) -> Array:
        """Node indices of a boundary. Shape: (nboundary,)"""
        ...

    def boundary_normals(self, boundary_id: int) -> Array:
        """Outward unit normals of a boundary. Shape: (nboundary, ndim)"""
        ...

    def jacobian_determinant(self) -> Array:
        """The map's Jacobian determinant at the nodes. Shape: (npts,)"""
        ...

    def transform(self) -> LevelMapProtocol[Array]:
        """The map of the reference cube onto the domain."""
        ...


class ChebyshevLevel(Generic[Array]):
    """One level of a study: a mapped Chebyshev mesh of a domain and its
    basis.

    Parameters
    ----------
    domain : DomainProtocol
        The domain meshed.
    mesh : LevelMeshProtocol
        The mesh.
    basis : TensorProductBasisProtocol
        The collocation basis on ``mesh``.
    parts : Mapping[str, int]
        The mesh boundary of each of the domain's boundary parts.

    Raises
    ------
    ValueError
        If ``parts`` does not name exactly the domain's boundary parts.
    """

    def __init__(
        self,
        domain: DomainProtocol,
        mesh: LevelMeshProtocol[Array],
        basis: TensorProductBasisProtocol[Array],
        parts: Mapping[str, int],
    ) -> None:
        if set(parts) != set(domain.boundaries()):
            raise ValueError(
                f"parts must name the domain's boundary parts "
                f"{list(domain.boundaries())}, got {sorted(parts)}"
            )
        self._domain, self._mesh, self._basis = domain, mesh, basis
        self._parts = dict(parts)

    def bkd(self) -> Backend[Array]:
        return self._basis.bkd()

    def domain(self) -> DomainProtocol:
        return self._domain

    def mesh(self) -> LevelMeshProtocol[Array]:
        return self._mesh

    def basis(self) -> TensorProductBasisProtocol[Array]:
        return self._basis

    def npts(self) -> int:
        return self._mesh.npts()

    def h(self) -> float:
        """The largest physical distance between neighboring nodes along
        a mesh line. On a box's Chebyshev-Gauss-Lobatto nodes it is the
        central gap, about :math:`\\pi L / (2 (N - 1))` on a side of length
        :math:`L` with :math:`N` nodes, so :math:`1/h` grows linearly in
        :math:`N`."""
        bkd, mesh = self.bkd(), self._mesh
        ndim = mesh.ndim()
        # Nodes run first dimension fastest, so dimension d is array axis
        # ndim - 1 - d of the C-ordered grid.
        shape = tuple(reversed(mesh.npts_per_dim()))
        grids = [bkd.reshape(mesh.points()[k], shape) for k in range(ndim)]
        gaps = []
        for dim in range(ndim):
            axis = ndim - 1 - dim
            squared = bkd.diff(grids[0], axis=axis) ** 2
            for grid in grids[1:]:
                squared = squared + bkd.diff(grid, axis=axis) ** 2
            gaps.append(bkd.to_float(bkd.max(bkd.sqrt(squared))))
        return max(gaps)

    def boundary_id(self, part: str) -> int:
        """The mesh boundary of boundary part ``part``.

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
        return f"{type(self).__name__}(npts_per_dim={self._mesh.npts_per_dim()})"


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
    """

    def __init__(self, sizes: Sequence[int], bkd: Backend[Array]) -> None:
        self._sizes, self._bkd = tuple(sizes), bkd

    def levels(self, domain: DomainProtocol) -> List[ChebyshevLevel[Array]]:
        """One level per size.

        Raises
        ------
        TypeError
            If ``domain`` is not a ``Box``.
        """
        if not isinstance(domain, Box):
            raise TypeError(f"ChebyshevBox meshes a Box, got {type(domain).__name__}")
        bkd, bounds = self._bkd, domain.bounds()
        flat = tuple(value for pair in bounds for value in pair)
        # A side is the mesh boundary at its position in the box's side
        # order, lower side first in each direction.
        parts = {part: i for i, part in enumerate(domain.boundaries())}
        levels: List[ChebyshevLevel[Array]] = []
        for n in self._sizes:
            if domain.ndim() == 1:
                mesh1 = TransformedMesh1D(n, bkd, AffineTransform1D(bounds[0], bkd))
                basis1 = ChebyshevBasis1D(mesh1, bkd)
                levels.append(ChebyshevLevel(domain, mesh1, basis1, parts))
            elif domain.ndim() == 2:
                square = AffineTransform2D((flat[0], flat[1], flat[2], flat[3]), bkd)
                mesh2 = TransformedMesh2D(n, n, bkd, square)
                basis2 = ChebyshevBasis2D(mesh2, bkd)
                levels.append(ChebyshevLevel(domain, mesh2, basis2, parts))
            else:
                cube = AffineTransform3D(
                    (flat[0], flat[1], flat[2], flat[3], flat[4], flat[5]), bkd
                )
                mesh3 = TransformedMesh3D(n, n, n, bkd, cube)
                basis3 = ChebyshevBasis3D(mesh3, bkd)
                levels.append(ChebyshevLevel(domain, mesh3, basis3, parts))
        return levels

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
    """The node indices of side ``part``, shape ``(nboundary,)``, and their
    outward unit normals, shape ``(nboundary, ndim)``.

    Raises
    ------
    ValueError
        If the box has no such side.
    """
    boundary = level.boundary_id(part)
    mesh = level.mesh()
    return mesh.boundary_indices(boundary), mesh.boundary_normals(boundary)


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
    parsed = _parsed(problem, expression, (time(),))
    if parsed.is_number:
        return float(parsed)
    # _Field declares its own time dependence, from the expression.
    return NodalSupplier(_Field([parsed], problem, level.bkd()), level.mesh().points())


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
    """The forcing of ``row`` at the level's nodes."""
    return field(problem, problem.forcing(row), level)


def stacked_forcing(
    problem: ManufacturedProblem, level: ChebyshevLevel[Array]
) -> NodalSupplier[Array]:
    """The forcing of every field at the level's nodes, stacked in the
    problem's field order, as a vector physics takes it: shape
    ``(nfields * npts,)``. Declared time-dependent exactly when a forcing
    uses ``t``."""
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
    rows the lower side builds."""

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
        prescribes a flux its field was not given, or only one side of a
        coordinate is periodic.
    TypeError
        If a flux satisfies neither ``FluxProviderProtocol`` nor
        ``NormalTermProtocol``.
    """
    fields = problem.fields()
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
    bkd, npts, ncomponents = level.bkd(), level.npts(), len(fields)
    closures: List[ClosureRowsProtocol[Array]] = []
    if periodic:
        closures.append(
            PeriodicRows(bkd, level.mesh().reference_points(), ncomponents, periodic)
        )
    if robin:
        # After the periodic rows, so a side's own condition wins the
        # nodes it shares with a periodic side.
        closures.append(RobinRows(bkd, npts, ncomponents, robin))
    essential = DirichletRows(bkd, npts, ncomponents, dirichlet) if dirichlet else None
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
    :math:`|\\det J|`), shape (nquad,); ``interpolation`` maps nodal
    values to the nodes' Lagrange interpolant at the points, shape
    (nquad, npts). With :math:`2N` points per direction the rule is exact
    to degree :math:`4N - 1` in the reference coordinates, so on a box
    (an affine map) the squared interpolant (degree :math:`2N - 2`) is
    integrated exactly and :math:`u^*` far more accurately than the
    error being measured; a curved map adds its own smooth factors.
    """

    points: Array
    weights: Array
    interpolation: Array


def _error_grid(level: ChebyshevLevel[Array]) -> _ErrorGrid[Array]:
    bkd, mesh = level.bkd(), level.mesh()
    coords: List[Array] = []
    weights: Optional[Array] = None
    interpolation: Optional[Array] = None
    for dim in range(mesh.ndim()):
        nodes = mesh.reference_nodes_1d(dim)
        nquad = 2 * nodes.shape[0]
        legendre = LegendrePolynomial1D(bkd)
        legendre.set_nterms(nquad)
        reference, probability = GaussQuadratureRule(legendre)(nquad)
        xi = reference[0, :]
        # Probability weights sum to 1; the reference interval has length 2.
        w = 2.0 * probability[:, 0]
        lagrange = univariate_lagrange_polynomial(nodes, xi, bkd)
        if weights is None or interpolation is None:
            coords, weights, interpolation = [xi], w, lagrange
            continue
        previous = weights.shape[0]
        coords = [bkd.kron(bkd.ones((nquad,)), c) for c in coords]
        coords.append(bkd.kron(xi, bkd.ones((previous,))))
        weights = bkd.kron(w, weights)
        interpolation = bkd.kron(lagrange, interpolation)
    if weights is None or interpolation is None:
        raise ValueError("a level has at least one dimension")
    reference_grid = bkd.stack(coords, axis=0)
    transform = mesh.transform()
    volume = bkd.abs(transform.jacobian_determinant(reference_grid))
    return _ErrorGrid(
        transform.map_to_physical(reference_grid), weights * volume, interpolation
    )


class CollocationApproximation(Generic[Array]):
    """One level's collocation solution of a problem's fields, with errors
    against the exact solution at the collocation nodes.

    The state stacks one component per field in the problem's field
    order. Errors combine over the fields: the L2 error is the root of the
    sum of their squared L2 errors, the nodal error their largest.

    Parameters
    ----------
    problem : ManufacturedProblem
        The problem solved.
    level : ChebyshevLevel
        The level it was solved on.
    state : Array
        The nodal values, field by field. Shape: (nfields * npts,)

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
        expected = len(fields) * level.npts()
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
        exact = solution(level.mesh().points())
        self._error = bkd.reshape(exact, (-1,)) - state
        grid = _error_grid(level)
        npts = level.npts()
        interpolated = bkd.hstack(
            [
                grid.interpolation @ state[k * npts : (k + 1) * npts]
                for k in range(len(fields))
            ]
        )
        self._grid_error = bkd.reshape(solution(grid.points), (-1,)) - interpolated
        self._weights = bkd.tile(grid.weights, (len(fields),))
        self._bkd, self._state, self._h = bkd, state, level.h()
        self._nodes = max(level.mesh().npts_per_dim())

    def h(self) -> float:
        """The largest gap between neighboring nodes (:meth:`ChebyshevLevel.h`)."""
        return self._h

    def nodes(self) -> int:
        """The most nodes along any direction, :math:`N`."""
        return self._nodes

    def state(self) -> Array:
        """The nodal values, field by field. Shape: (nfields * npts,)"""
        return self._state

    def l2_error(self) -> float:
        r""":math:`\|u^* - u_h\|_{L^2}`, :math:`u_h` the Lagrange
        interpolant of the nodal values, by Gauss-Legendre quadrature with
        twice the level's nodes per direction against :math:`u^*` itself,
        summed over the fields. It sees the error between the nodes, which
        a quadrature at the nodes would not."""
        bkd = self._bkd
        squared = bkd.sum(self._weights * self._grid_error**2)
        return math.sqrt(bkd.to_float(squared))

    def nodal_error(self) -> float:
        r""":math:`\max_i |u^*(x_i) - u_h(x_i)|` over the collocation nodes
        and the fields."""
        bkd = self._bkd
        return bkd.to_float(bkd.max(bkd.abs(self._error)))


class CollocationDiscretizer(Generic[Array]):
    """Solves a steady problem with this stack at every level of a family.
    The factory's system stacks the problem's fields in the problem's
    order.

    Each level: the factory's system, the steady residual
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

    def approximations(
        self, problem: ManufacturedProblem
    ) -> List[CollocationApproximation[Array]]:
        """One approximation per level, coarsest first.

        Raises
        ------
        ValueError
            If ``problem`` is transient or has no domain, a system's
            state is not one value per node per field, or an initial guess
            does not have shape (nstates, 1).
        RuntimeError
            If a solve does not converge.
        """
        if problem.is_transient():
            raise ValueError(
                "CollocationDiscretizer solves steady problems; this one depends on t"
            )
        domain = problem.domain()
        if domain is None:
            raise ValueError("CollocationDiscretizer needs a domain; this is an ODE")
        fields = problem.fields()
        approximations = []
        for level in self._levels.levels(domain):
            system = self._factory.system(problem, level)
            bkd, nstates = level.bkd(), system.nstates()
            if nstates != len(fields) * level.npts():
                raise ValueError(
                    f"the system has {nstates} states, not one per node "
                    f"({level.npts()}) for each of {list(fields)}"
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
            approximations.append(
                CollocationApproximation(problem, level, point.state()[:, 0])
            )
        return approximations

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._levels!r}, {self._factory!r})"


@dataclass(frozen=True)
class NodeStudy:
    """Node counts and errors, coarsest first: a spectral study, which
    the core's :class:`~pyapprox.pde.verification.study.Study` (mesh
    sizes only) cannot hold.

    Raises
    ------
    ValueError
        If there are no levels, the lengths differ, a node count is below
        two, or an error is negative or not finite.
    """

    nodes: Tuple[int, ...]
    errors: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.nodes or len(self.nodes) != len(self.errors):
            raise ValueError(
                f"a study needs one error per node count, at least one; got "
                f"{len(self.nodes)} counts and {len(self.errors)} errors"
            )
        if any(n < 2 for n in self.nodes):
            raise ValueError(f"node counts must be at least two, got {self.nodes}")
        if any(not (math.isfinite(e) and e >= 0.0) for e in self.errors):
            raise ValueError(f"errors must be finite and non-negative: {self.errors}")


def run_node_study(
    problem: ManufacturedProblem,
    discretizer: CollocationDiscretizer[Array],
    measure: ErrorMeasureProtocol,
) -> NodeStudy:
    """Solve ``problem`` at every level of ``discretizer`` and record
    ``measure`` against each level's node count.

    Raises
    ------
    TypeError
        If ``measure`` does not satisfy ``ErrorMeasureProtocol``.
    """
    if not isinstance(measure, ErrorMeasureProtocol):
        raise TypeError(
            f"measure must satisfy ErrorMeasureProtocol, got {type(measure).__name__}"
        )
    approximations = discretizer.approximations(problem)
    return NodeStudy(
        tuple(a.nodes() for a in approximations),
        tuple(measure.measure(a) for a in approximations),
    )


class DecaysExponentially:
    r"""The error falls at least exponentially in the number of nodes: the
    least-squares slope of :math:`\log e` against :math:`N` is at most
    ``-rate``.

    Spectral methods on smooth solutions converge as :math:`e \sim
    C\exp(-\rho N)`, which no algebraic rate in :math:`h` states: the
    slope of :math:`\log e` against :math:`\log h` grows without bound,
    while against :math:`N` it is the constant :math:`-\rho`, with
    :math:`\rho = \ln r` for a solution analytic inside the Bernstein
    ellipse of parameter :math:`r`. For an entire solution the
    convergence is faster than any exponential and the slope steepens
    with :math:`N`; ``rate`` is a floor, which that only clears more
    easily. Fit levels above round-off, where the curve flattens.

    Parameters
    ----------
    rate : float
        The decay :math:`\rho` a correct model reaches, per node.
    last : int, optional
        Fit only the finest ``last`` levels.
    """

    def __init__(self, rate: float, last: Optional[int] = None) -> None:
        self._rate, self._last = rate, last

    def verify(self, study: NodeStudy) -> CheckResult:
        """Pass if the fitted decay reaches ``rate``; fail on a zero error,
        whose logarithm the fit cannot take.

        Raises
        ------
        ValueError
            If fewer than two levels are fitted or the fitted node counts
            are all equal.
        """
        nodes, errors = study.nodes, study.errors
        if self._last is not None:
            nodes, errors = nodes[-self._last :], errors[-self._last :]
        if len(nodes) < 2:
            raise ValueError(f"a decay needs at least two levels, got {len(nodes)}")
        if any(e == 0.0 for e in errors):
            return CheckResult(False, 0.0, f"an error is zero: {errors}")
        xs = [float(n) for n in nodes]
        ys = [math.log(e) for e in errors]
        xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - xmean) ** 2 for x in xs)
        if sxx == 0.0:
            raise ValueError(f"a decay needs distinct node counts, got {nodes}")
        slope = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys)) / sxx
        observed = -slope
        return CheckResult(
            observed >= self._rate,
            observed,
            f"fitted decay {observed:.3f} per node, expected at least "
            f"{self._rate:.3f}; nodes {nodes}, errors {errors}",
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rate={self._rate}, last={self._last})"
