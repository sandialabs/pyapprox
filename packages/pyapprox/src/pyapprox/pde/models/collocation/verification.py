r"""Verify collocation discretizations with :mod:`pyapprox.pde.verification`.

The adapter between the stack-independent verification core and this
repository's spectral collocation stack. Backend-generic: the levels,
coefficients and rows live on the family's backend, and only the errors
leave it, as floats.

- :class:`ChebyshevBox` gives a study's levels: Chebyshev-Gauss-Lobatto
  tensor meshes of a :class:`~pyapprox.pde.verification.domains.Box`
  (affine maps of the reference cube), one per point count, each a
  :class:`ChebyshevLevel` holding its mesh and basis. A side of the box
  is the mesh boundary at the same position in
  :meth:`~pyapprox.pde.verification.domains.Box.boundaries` (left,
  right, bottom, top, back, front), so named boxes work unchanged.
- A system factory, supplied by the case, builds the physics under test
  on a level and closes it with :func:`collocation_system`. Nothing here
  knows any physics: a factory builds its physics class from the generic
  helpers :func:`field`, :func:`nodal_vector` and :func:`forcing`, which
  declare each coefficient's time dependence from its expression (it
  uses ``t`` or not).
- :func:`boundary_conditions` builds the rows of each side's condition:
  Dirichlet as :class:`~pyapprox.pde.collocation.boundary.essential_rows.
  DirichletRows`; Neumann and Robin as :class:`~pyapprox.pde.collocation.
  boundary.closure_rows.RobinRows` on the flux normal operator of a
  supplied flux; periodic sides as :class:`~pyapprox.pde.collocation.
  boundary.periodic_rows.PeriodicRows` pairing the two sides of a
  coordinate. A node two sides share follows the system's rule: essential
  rows win it, then the Robin rows, then the periodic rows. A case that
  builds its rows by hand (a wrong-model control) takes a side's nodes
  and data from :func:`side_nodes` and :func:`side_values`.
- :class:`CollocationDiscretizer` solves each level through
  ``collocation_steady_residual`` + ``SteadySystem`` with a dense LU and
  returns :class:`CollocationApproximation` objects, whose errors are
  measured at the collocation nodes.
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
:math:`\beta = 1`.
"""

import math
from dataclasses import dataclass
from typing import (
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
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
    RobinPart,
    RobinRows,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import FluxNormalOperator
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
from pyapprox.pde.collocation.quadrature.collocation_quadrature import (
    CollocationQuadrature1D,
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
from pyapprox.util.backends.protocols import Array, Array_co, Backend
from pyapprox.util.expressions.parsing import Value, parse
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton, NonlinearSolverProtocol


class BoxMeshProtocol(Protocol[Array_co]):
    """What the adapter reads from a level's mesh."""

    def ndim(self) -> int: ...

    def npts(self) -> int: ...

    def npts_per_dim(self) -> Tuple[int, ...]: ...

    def points(self) -> Array_co:
        """Physical coordinates. Shape: (ndim, npts)"""
        ...

    def reference_points(self) -> Array_co:
        """Reference coordinates. Shape: (ndim, npts)"""
        ...

    def reference_nodes_1d(self, dim: int) -> Array_co:
        """Reference nodes of dimension ``dim``. Shape: (npts_dim,)"""
        ...

    def boundary_indices(self, boundary_id: int) -> Array_co:
        """Node indices of a boundary. Shape: (nboundary,)"""
        ...

    def boundary_normals(self, boundary_id: int) -> Array_co:
        """Outward unit normals of a boundary. Shape: (nboundary, ndim)"""
        ...

    def jacobian_determinant(self) -> Array_co:
        """The map's Jacobian determinant at the nodes. Shape: (npts,)"""
        ...


class ChebyshevLevel(Generic[Array]):
    """One level of a study: a Chebyshev mesh of a box and its basis.

    Parameters
    ----------
    domain : Box
        The box meshed.
    mesh : BoxMeshProtocol
        The mesh, its boundaries in the box's side order.
    basis : TensorProductBasisProtocol
        The collocation basis on ``mesh``.
    """

    def __init__(
        self,
        domain: Box,
        mesh: BoxMeshProtocol[Array],
        basis: TensorProductBasisProtocol[Array],
    ) -> None:
        self._domain, self._mesh, self._basis = domain, mesh, basis

    def bkd(self) -> Backend[Array]:
        return self._basis.bkd()

    def domain(self) -> Box:
        return self._domain

    def mesh(self) -> BoxMeshProtocol[Array]:
        return self._mesh

    def basis(self) -> TensorProductBasisProtocol[Array]:
        return self._basis

    def npts(self) -> int:
        return self._mesh.npts()

    def h(self) -> float:
        """The largest gap between neighboring nodes along any coordinate.
        On Chebyshev-Gauss-Lobatto nodes it is the central gap, about
        :math:`\\pi L / (2 (N - 1))` on a side of length :math:`L` with
        :math:`N` nodes, so :math:`1/h` grows linearly in :math:`N`."""
        bkd = self.bkd()
        gaps = []
        for dim, (lower, upper) in enumerate(self._domain.bounds()):
            nodes = self._mesh.reference_nodes_1d(dim)
            widest = bkd.to_float(bkd.max(bkd.abs(nodes[1:] - nodes[:-1])))
            gaps.append(widest * (upper - lower) / 2.0)
        return max(gaps)

    def boundary_id(self, part: str) -> int:
        """The mesh boundary of side ``part``.

        Raises
        ------
        ValueError
            If the box has no such side.
        """
        sides = self._domain.boundaries()
        if part not in sides:
            raise ValueError(f"no side {part!r}; the box has {list(sides)}")
        return sides.index(part)

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
        levels: List[ChebyshevLevel[Array]] = []
        for n in self._sizes:
            if domain.ndim() == 1:
                mesh1 = TransformedMesh1D(n, bkd, AffineTransform1D(bounds[0], bkd))
                basis1 = ChebyshevBasis1D(mesh1, bkd)
                levels.append(ChebyshevLevel(domain, mesh1, basis1))
            elif domain.ndim() == 2:
                square = AffineTransform2D((flat[0], flat[1], flat[2], flat[3]), bkd)
                mesh2 = TransformedMesh2D(n, n, bkd, square)
                basis2 = ChebyshevBasis2D(mesh2, bkd)
                levels.append(ChebyshevLevel(domain, mesh2, basis2))
            else:
                cube = AffineTransform3D(
                    (flat[0], flat[1], flat[2], flat[3], flat[4], flat[5]), bkd
                )
                mesh3 = TransformedMesh3D(n, n, n, bkd, cube)
                basis3 = ChebyshevBasis3D(mesh3, bkd)
                levels.append(ChebyshevLevel(domain, mesh3, basis3))
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
    """``f(coords, time)``, one expression, shape ``(npts,)``; time dependent
    exactly when the expression uses ``t``. Satisfies
    ``TimeAwareCallableProtocol``."""

    def __init__(
        self, expression: sp.Expr, problem: ManufacturedProblem, bkd: Backend[Array]
    ) -> None:
        self._points = PointExpressions(
            [expression], problem.names(), problem.coordinates(), bkd
        )
        self._timed = time() in expression.free_symbols

    def __call__(self, coords: Array, time: float) -> Array:
        return self._points(coords, t=time)[0]

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
    return NodalSupplier(_Field(parsed, problem, level.bkd()), level.mesh().points())


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
            values.append(_Field(parsed, problem, bkd)(points, 0.0))
    return values


def forcing(
    problem: ManufacturedProblem, row: str, level: ChebyshevLevel[Array]
) -> Union[float, NodalSupplier[Array]]:
    """The forcing of ``row`` at the level's nodes."""
    return field(problem, problem.forcing(row), level)


_BuiltPart = Union[
    DirichletPart[Array], RobinPart[Array], PeriodicPart[Array], None
]


class _CollocationCondition(Generic[Array]):
    """Builds the rows of one row's condition on one side; ``None`` for the
    upper side of a periodic pair, whose rows the lower side builds."""

    def __init__(
        self,
        problem: ManufacturedProblem,
        level: ChebyshevLevel[Array],
        part: str,
        row: str,
        flux: Optional[FluxProviderProtocol[Array]],
    ) -> None:
        self._problem, self._level = problem, level
        self._part, self._row, self._flux = part, row, flux
        self._indices, self._normals = side_nodes(level, part)

    def _values(self, expression: sp.Expr) -> Union[float, _BoundaryValues[Array]]:
        return side_values(self._problem, self._level, self._part, expression)

    def _data(self) -> Union[float, _BoundaryValues[Array]]:
        return self._values(self._problem.boundary_data(self._part)[self._row])

    def _flux_term(self) -> FluxNormalOperator[Array]:
        if self._flux is None:
            raise ValueError(
                f"the condition on {self._part!r} prescribes a flux; pass the "
                "flux provider whose normal flux it prescribes"
            )
        return FluxNormalOperator(
            self._level.bkd(), self._indices, self._normals, self._flux
        )

    def dirichlet(self, condition: Dirichlet) -> _BuiltPart[Array]:
        return DirichletPart(self._indices, self._data())

    def neumann(self, condition: Neumann) -> _BuiltPart[Array]:
        return RobinPart(self._indices, self._flux_term(), 0.0, 1.0, self._data())

    def robin(self, condition: Robin) -> _BuiltPart[Array]:
        names = self._problem.names()
        alpha = names.substitute_constants(condition.alpha(names))
        return RobinPart(
            self._indices, self._flux_term(), self._values(-alpha), 1.0, self._data()
        )

    def periodic(self, condition: Periodic) -> _BuiltPart[Array]:
        domain = self._level.domain()
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
        )


@dataclass(frozen=True)
class BoundaryDeclarations(Generic[Array]):
    """The rows of a problem's conditions, as a ``CollocationSystem``
    takes them: the essential rows, and the closures in the order that
    settles shared nodes (later wins)."""

    essential: Optional[DirichletRows[Array]]
    closures: Tuple[ClosureRowsProtocol[Array], ...]


def boundary_conditions(
    problem: ManufacturedProblem,
    level: ChebyshevLevel[Array],
    row: str,
    flux: Optional[FluxProviderProtocol[Array]] = None,
) -> BoundaryDeclarations[Array]:
    """The rows of every side's condition on scalar field ``row``, with data
    from the problem's expressions.

    Parameters
    ----------
    problem : ManufacturedProblem
        Gives the conditions and their data.
    level : ChebyshevLevel
        The mesh the rows live on.
    row : str
        The field.
    flux : FluxProviderProtocol, optional
        The flux whose outward normal component Neumann and Robin sides
        prescribe: the physics' own, or another such as
        :class:`~pyapprox.pde.collocation.physics.advection_diffusion.
        AdvectionDiffusionFlux`. Needed only when a side has one.

    Raises
    ------
    ValueError
        If ``row`` takes no boundary conditions, a side prescribes a flux
        and none is given, or only one side of a coordinate is periodic.
    TypeError
        If ``flux`` does not satisfy ``FluxProviderProtocol``.
    """
    if row not in problem.boundary_rows():
        raise ValueError(
            f"row {row!r} takes no boundary conditions; the boundary rows are "
            f"{list(problem.boundary_rows())}"
        )
    if flux is not None and not isinstance(flux, FluxProviderProtocol):
        raise TypeError(
            f"flux must satisfy FluxProviderProtocol, got {type(flux).__name__}"
        )
    dirichlet: List[DirichletPart[Array]] = []
    robin: List[RobinPart[Array]] = []
    periodic: List[PeriodicPart[Array]] = []
    for part in level.domain().boundaries():
        visitor = _CollocationCondition(problem, level, part, row, flux)
        built = problem.conditions(part)[row].accept(visitor)
        if isinstance(built, DirichletPart):
            dirichlet.append(built)
        elif isinstance(built, RobinPart):
            robin.append(built)
        elif isinstance(built, PeriodicPart):
            periodic.append(built)
    bkd, npts = level.bkd(), level.npts()
    closures: List[ClosureRowsProtocol[Array]] = []
    if periodic:
        closures.append(PeriodicRows(bkd, level.mesh().reference_points(), 1, periodic))
    if robin:
        # After the periodic rows, so a side's own condition wins the
        # nodes it shares with a periodic side.
        closures.append(RobinRows(bkd, npts, 1, robin))
    essential = DirichletRows(bkd, npts, 1, dirichlet) if dirichlet else None
    return BoundaryDeclarations(essential, tuple(closures))


def collocation_system(
    physics: CollocationInteriorOperatorProtocol[Array],
    problem: ManufacturedProblem,
    level: ChebyshevLevel[Array],
    row: str,
    flux: Optional[FluxProviderProtocol[Array]] = None,
) -> CollocationSystem[Array]:
    """``physics`` closed by the rows of :func:`boundary_conditions`."""
    declarations = boundary_conditions(problem, level, row, flux)
    return CollocationSystem(physics, declarations.essential, declarations.closures)


class SystemFactoryProtocol(Protocol):
    """Builds the physics under test, with its conditions, on a level."""

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        """The system whose solution approximates ``problem``'s."""
        ...


def _quadrature_weights(level: ChebyshevLevel[Array]) -> Array:
    """Weights at the level's nodes integrating over the box: tensor
    products of 1D Clenshaw-Curtis-type weights on the reference interval
    (first dimension fastest), times the map's Jacobian determinant.
    Shape: (npts,)"""
    bkd = level.bkd()
    weights: Optional[Array] = None
    for n in level.mesh().npts_per_dim():
        reference = ChebyshevBasis1D(TransformedMesh1D(n, bkd), bkd)
        w = CollocationQuadrature1D(reference, bkd).full_domain_weights()
        weights = w if weights is None else bkd.kron(w, weights)
    if weights is None:
        raise ValueError("a level has at least one dimension")
    return weights * level.mesh().jacobian_determinant()


class CollocationApproximation(Generic[Array]):
    """One level's collocation solution of a scalar field, with errors
    against the exact solution at the collocation nodes.

    Parameters
    ----------
    problem : ManufacturedProblem
        The problem solved.
    row : str
        The field the solution approximates.
    level : ChebyshevLevel
        The level it was solved on.
    state : Array
        The nodal values. Shape: (npts,)
    """

    def __init__(
        self,
        problem: ManufacturedProblem,
        row: str,
        level: ChebyshevLevel[Array],
        state: Array,
    ) -> None:
        bkd = level.bkd()
        exact = PointExpressions(
            [problem.solution(row)], problem.names(), problem.coordinates(), bkd
        )(level.mesh().points())[0]
        self._error = exact - state
        self._weights = _quadrature_weights(level)
        self._bkd, self._state, self._h = bkd, state, level.h()
        self._nodes = max(level.mesh().npts_per_dim())

    def h(self) -> float:
        """The largest gap between neighboring nodes (:meth:`ChebyshevLevel.h`)."""
        return self._h

    def nodes(self) -> int:
        """The most nodes along any direction, :math:`N`."""
        return self._nodes

    def state(self) -> Array:
        """The nodal values. Shape: (npts,)"""
        return self._state

    def l2_error(self) -> float:
        r""":math:`\|u^* - u_h\|_{L^2}` by the collocation quadrature of the
        squared nodal error: exact for the interpolant of the error when
        its square is in the quadrature's polynomial space."""
        bkd = self._bkd
        return math.sqrt(bkd.to_float(bkd.sum(self._weights * self._error**2)))

    def nodal_error(self) -> float:
        r""":math:`\max_i |u^*(x_i) - u_h(x_i)|` over the collocation nodes."""
        bkd = self._bkd
        return bkd.to_float(bkd.max(bkd.abs(self._error)))


class CollocationDiscretizer(Generic[Array]):
    """Solves a steady scalar problem with this stack at every level of a
    family.

    Each level: the factory's system, the steady residual
    (``collocation_steady_residual``), and ``SteadySystem`` with a dense
    LU recomputed every step, from a zero initial guess.

    Parameters
    ----------
    levels : LevelFamilyProtocol
        The levels.
    factory : SystemFactoryProtocol
        The system of a problem on a level.
    nonlinear : NonlinearSolverProtocol, optional
        Default: Newton, ``atol = rtol = 1e-12``, 30 steps (a linear
        system is solved in one step).
    """

    def __init__(
        self,
        levels: LevelFamilyProtocol[Array],
        factory: SystemFactoryProtocol,
        nonlinear: Optional[NonlinearSolverProtocol[Array, Array]] = None,
    ) -> None:
        self._levels, self._factory = levels, factory
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
            If ``problem`` is transient, has no domain, or has more than
            one field, or a system's state is not one value per node.
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
        if len(fields) != 1:
            raise ValueError(
                f"CollocationDiscretizer solves one field, got {list(fields)}"
            )
        approximations = []
        for level in self._levels.levels(domain):
            system = self._factory.system(problem, level)
            bkd, nstates = level.bkd(), system.nstates()
            if nstates != level.npts():
                raise ValueError(
                    f"the system has {nstates} states, not one per node "
                    f"({level.npts()})"
                )
            steady = SteadySystem(
                collocation_steady_residual(system),
                backend_state_space(bkd, nstates),
                RecomputeAlways(BackendDenseLU(bkd)),
            )
            point = steady.solve(self._nonlinear, bkd.zeros((nstates, 1)))
            if not point.converged():
                raise RuntimeError(
                    f"the solve on {level!r} did not converge: {point.info().message}"
                )
            approximations.append(
                CollocationApproximation(problem, fields[0], level, point.state()[:, 0])
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
