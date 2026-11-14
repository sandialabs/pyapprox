r"""Verify Galerkin discretizations with :mod:`pyapprox.pde.verification`.

The adapter between the stack-independent verification core and this
repository's skfem Galerkin stack. Bound to skfem and NumPy.

- Mesh families (:class:`StructuredBox`, :class:`SkfemDisk`) refine a
  domain, naming mesh boundaries as the domain names its parts.
- Basis factories (:class:`Lagrange`) build a basis on each mesh.
- A system factory, supplied by the case, builds the physics under test
  on a basis. Nothing here knows any physics: a factory builds its
  physics class from the generic helpers :func:`field`,
  :func:`vector_field`, :func:`state_law`, :func:`forcing` and
  :func:`boundary_conditions`, so each test exercises the class it names.
  The helpers declare each supplier's time dependence from its
  expression (it uses ``t`` or not), so they serve transient problems
  unchanged. :func:`field`, :func:`vector_field` and :func:`state_law`
  need only a scope of names and coordinates, not a manufactured
  problem, so text coefficients of a production problem pass through
  them too.
- :class:`GalerkinDiscretizer` solves each level through
  ``galerkin_steady_residual`` + ``SteadySystem`` and returns
  :class:`GalerkinApproximation` objects, whose errors are integrated by
  skfem quadrature against the exact solution itself.

Natural data: the core states them with the outward flux :math:`q\,n`;
this stack's natural terms take :math:`D\,\partial_n u - (v\cdot n) u =
-q\,n`, so the adapter negates Neumann and Robin data. Robin's
:math:`\alpha` is the same in both.

Shapes: the core evaluator takes points ``(ndim, npts)`` only. skfem
forms pass quadrature points ``(ndim, nelem, nquad)`` and states
``(nelem, nquad)``; :class:`SkfemPoints` is the one place that fits them.
"""

from typing import Any, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from skfem import Basis, Functional, MeshTri

from pyapprox.implicitfunction.setups import KeepAll
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.boundary.protocols import BoundaryConditionRole
from pyapprox.pde.constitutive.coefficient_functions import CallableReaction
from pyapprox.pde.expressions.evaluation import PointExpressions
from pyapprox.pde.expressions.symbols import coordinates, time
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.pde.galerkin.protocols.mesh import GalerkinMeshProtocol
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import skfem_discretization
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Periodic,
    Robin,
)
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.expressions.names import Names
from pyapprox.util.expressions.parsing import Value, parse
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton, NonlinearSolverProtocol

_Np = NDArray[Any]
_BKD = NumpyBkd()
_SIDES = ("left", "right", "bottom", "top", "back", "front")


class ExpressionScopeProtocol(Protocol):
    """What expressions may use, and the coordinates of the points they
    are evaluated at. A :class:`~pyapprox.pde.verification.problem.
    ManufacturedProblem` is one."""

    def names(self) -> Names:
        """What the expressions may use, from
        :func:`~pyapprox.pde.expressions.symbols.pde_names`."""
        ...

    def coordinates(self) -> Tuple[str, ...]:
        """The coordinate names, in the order of the points' rows."""
        ...


class SkfemPoints:
    """Problem expressions on skfem's point layouts.

    Points arrive as ``(ndim, ...)`` with any trailing layout (``(npts,)``
    at DOFs, ``(nelem, nquad)`` at quadrature points); normals as the
    points; state jets as the trailing layout. They are flattened for
    :class:`~pyapprox.pde.expressions.evaluation.PointExpressions` and its
    values returned in the same layout, ``(nexpressions, ...)``.

    Parameters
    ----------
    expressions : Sequence
        The expressions, as text or sympy, in the scope's names.
    scope : ExpressionScopeProtocol
        Gives the names, constants and coordinates.
    """

    def __init__(
        self, expressions: Sequence[Value], scope: ExpressionScopeProtocol
    ) -> None:
        self._evaluator: PointExpressions[_Np] = PointExpressions(
            expressions, scope.names(), scope.coordinates(), _BKD
        )
        self._ndim = len(scope.coordinates())

    def __call__(
        self,
        coords: _Np,
        normals: Optional[_Np] = None,
        t: float = 0.0,
        state: Optional[Mapping[str, _Np]] = None,
    ) -> _Np:
        """The expressions at ``coords``. Shape: ``(nexpressions,) +
        coords.shape[1:]``.

        Raises
        ------
        ValueError
            If ``coords`` does not lead with ``ndim``, or ``normals`` or a
            state jet does not match its layout.
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim < 2 or coords.shape[0] != self._ndim:
            raise ValueError(
                f"coords must have shape ({self._ndim}, ...), got {coords.shape}"
            )
        layout = coords.shape[1:]
        flat_normals = None
        if normals is not None:
            normals = np.asarray(normals, dtype=float)
            if normals.shape != coords.shape:
                raise ValueError(
                    f"normals must have shape {coords.shape}, got {normals.shape}"
                )
            flat_normals = normals.reshape(self._ndim, -1)
        flat_state = {}
        for name, value in ({} if state is None else state).items():
            value = np.asarray(value, dtype=float)
            if value.shape != layout:
                raise ValueError(f"{name} must have shape {layout}, got {value.shape}")
            flat_state[name] = value.reshape(-1)
        values = self._evaluator(
            coords.reshape(self._ndim, -1),
            normals=flat_normals,
            t=t,
            state=flat_state,
        )
        return values.reshape((self._evaluator.nexpressions(),) + layout)


def _parsed(
    scope: ExpressionScopeProtocol,
    expression: Value,
    allowed: Sequence[sp.Symbol],
) -> sp.Expr:
    """``expression`` parsed in ``scope``, constants substituted, refused
    if it uses anything but the coordinates and ``allowed``."""
    names = scope.names()
    parsed = names.substitute_constants(parse(expression, names))
    permitted = set(coordinates(scope.coordinates())) | set(allowed)
    extra = parsed.free_symbols - permitted
    if extra:
        raise ValueError(
            f"{expression!r} may depend only on "
            f"{sorted(map(str, permitted))}; got {sorted(map(str, extra))}"
        )
    return parsed


class _Field:
    """``f(coords, time)``, one expression, in the layout of ``coords``;
    time dependent exactly when the expression uses ``t``."""

    def __init__(self, expression: sp.Expr, scope: ExpressionScopeProtocol) -> None:
        self._points = SkfemPoints([expression], scope)
        self._timed = time() in expression.free_symbols

    def __call__(self, coords: _Np, time: float) -> _Np:
        values: _Np = self._points(coords, t=time)[0]
        return values

    def is_time_dependent(self) -> bool:
        return self._timed


class _VectorField:
    """``v(coords, time)``, one expression per direction, shape
    ``(ndim, ...)``; time dependent exactly when one uses ``t``."""

    def __init__(
        self, expressions: Sequence[sp.Expr], scope: ExpressionScopeProtocol
    ) -> None:
        self._points = SkfemPoints(expressions, scope)
        self._timed = any(time() in e.free_symbols for e in expressions)

    def __call__(self, coords: _Np, time: float) -> _Np:
        return self._points(coords, t=time)

    def is_time_dependent(self) -> bool:
        return self._timed


class _StateLaw:
    """``R(coords, u)``, in the layout of ``u``."""

    def __init__(
        self, expression: sp.Expr, scope: ExpressionScopeProtocol, name: str
    ) -> None:
        self._points = SkfemPoints([expression], scope)
        self._name = name

    def __call__(self, coords: _Np, u: _Np) -> _Np:
        values: _Np = self._points(coords, state={self._name: u})[0]
        return values


class _NaturalData:
    """This stack's natural data, :math:`-g`, from the core's outward-flux
    data :math:`g`: ``evaluate(coords, normal, time)``."""

    def __init__(self, data: sp.Expr, scope: ExpressionScopeProtocol) -> None:
        self._points = SkfemPoints([data], scope)
        self._timed = time() in data.free_symbols

    def evaluate(self, coords: _Np, normal: _Np, time: float) -> _Np:
        values: _Np = -self._points(coords, normals=normal, t=time)[0]
        return values

    def is_time_dependent(self) -> bool:
        return self._timed


def field(scope: ExpressionScopeProtocol, expression: Value) -> Union[float, _Field]:
    """A scalar coefficient ``f(coords, time)`` from an expression in the
    coordinates, ``t`` and the scope's constants; a constant as a float.

    Raises
    ------
    ValueError
        If the expression uses anything else.
    """
    parsed = _parsed(scope, expression, (time(),))
    if parsed.is_number:
        return float(parsed)
    return _Field(parsed, scope)


def vector_field(
    scope: ExpressionScopeProtocol, expressions: Sequence[Value]
) -> Union[_Np, _VectorField]:
    """A vector coefficient ``v(coords, time)``, shape ``(ndim, ...)``,
    one expression per coordinate; constants as an array ``(ndim,)``.

    Raises
    ------
    ValueError
        If there is not one expression per coordinate, or one uses
        anything but the coordinates, ``t`` and constants.
    """
    coords = scope.coordinates()
    if len(expressions) != len(coords):
        raise ValueError(
            f"one expression per coordinate {list(coords)}, got {len(expressions)}"
        )
    parsed = [_parsed(scope, e, (time(),)) for e in expressions]
    if all(p.is_number for p in parsed):
        return np.array([float(p) for p in parsed])
    return _VectorField(parsed, scope)


def state_law(
    scope: ExpressionScopeProtocol, expression: Value, name: str
) -> CallableReaction:
    """A pointwise law ``R(coords, u)`` of field ``name``, with its first
    and second ``u``-derivatives by sympy, so the physics and the forcing
    cannot disagree.

    Raises
    ------
    ValueError
        If ``name`` is not a declared name, or the expression uses
        anything but the coordinates, constants and ``name`` (this
        stack's reaction laws do not take time).
    """
    u = sp.Symbol(name)
    if u not in scope.names().variables():
        raise ValueError(f"{name!r} is not a field of the scope")
    law = _parsed(scope, expression, (u,))
    slope = sp.diff(law, u)
    return CallableReaction(
        _StateLaw(law, scope, name),
        _StateLaw(slope, scope, name),
        _StateLaw(sp.diff(slope, u), scope, name),
    )


def forcing(problem: ManufacturedProblem, row: str) -> _Field:
    """The forcing of ``row`` as ``f(coords, time)``."""
    return _Field(problem.forcing(row), problem)


class _GalerkinCondition:
    """Builds the Galerkin boundary condition of one row on one part."""

    def __init__(
        self,
        problem: ManufacturedProblem,
        basis: LagrangeBasis[_Np],
        part: str,
        row: str,
    ) -> None:
        self._problem, self._basis = problem, basis
        self._part, self._row = part, row

    def _data(self) -> sp.Expr:
        return self._problem.boundary_data(self._part)[self._row]

    def dirichlet(self, condition: Dirichlet) -> BoundaryConditionRole[_Np]:
        return DirichletBC(
            self._basis,
            self._part,
            _Field(self._data(), self._problem),
            _BKD,
        )

    def neumann(self, condition: Neumann) -> BoundaryConditionRole[_Np]:
        return NeumannBC(
            self._basis, self._part, _NaturalData(self._data(), self._problem), _BKD
        )

    def robin(self, condition: Robin) -> BoundaryConditionRole[_Np]:
        alpha = self._problem.names().substitute_constants(
            condition.alpha(self._problem.names())
        )
        return RobinBC(
            self._basis,
            self._part,
            field(self._problem, alpha),
            _NaturalData(self._data(), self._problem),
            _BKD,
        )

    def periodic(self, condition: Periodic) -> BoundaryConditionRole[_Np]:
        raise ValueError(
            f"the Galerkin stack has no periodic conditions; {self._part!r} is periodic"
        )


def boundary_conditions(
    problem: ManufacturedProblem, basis: LagrangeBasis[_Np], row: str
) -> List[BoundaryConditionRole[_Np]]:
    """One Galerkin boundary condition per boundary part of ``problem``
    for ``row``, with data from the problem's expressions.

    Raises
    ------
    ValueError
        If ``row`` takes no boundary conditions, or a condition is
        periodic.
    """
    if row not in problem.boundary_rows():
        raise ValueError(
            f"row {row!r} takes no boundary conditions; the boundary rows are "
            f"{list(problem.boundary_rows())}"
        )
    domain = problem.domain()
    parts = () if domain is None else domain.boundaries()
    conditions: List[BoundaryConditionRole[_Np]] = []
    for part in parts:
        condition: ConditionProtocol = problem.conditions(part)[row]
        conditions.append(
            condition.accept(_GalerkinCondition(problem, basis, part, row))
        )
    return conditions


class SystemFactoryProtocol(Protocol):
    """Builds the physics under test, with its conditions, on a basis."""

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        """The system whose solution approximates ``problem``'s."""
        ...


class MeshFamilyProtocol(Protocol):
    """Successively refined meshes of a domain, coarsest first, their
    boundaries named as the domain names its parts."""

    def meshes(self, domain: DomainProtocol) -> Sequence[GalerkinMeshProtocol[_Np]]:
        """The meshes of ``domain``."""
        ...


class StructuredBox:
    """Structured meshes of a :class:`~pyapprox.pde.verification.domains.
    Box`, ``n`` cells per direction at each level.

    Parameters
    ----------
    sizes : Sequence[int]
        Cells per direction, one entry per level.
    simplex : bool
        Triangles or tetrahedra instead of quadrilaterals or hexahedra (no
        effect in 1D).
    """

    def __init__(self, sizes: Sequence[int], simplex: bool = False) -> None:
        self._sizes, self._simplex = tuple(sizes), simplex

    def meshes(self, domain: DomainProtocol) -> List[GalerkinMeshProtocol[_Np]]:
        """One structured mesh per size.

        Raises
        ------
        TypeError
            If ``domain`` is not a ``Box``.
        ValueError
            If its sides are not the standard ``left``, ``right``,
            ``bottom``, ``top``, ``back``, ``front``, which name this
            stack's mesh boundaries.
        """
        if not isinstance(domain, Box):
            raise TypeError(f"StructuredBox meshes a Box, got {type(domain).__name__}")
        ndim = domain.ndim()
        if domain.boundaries() != _SIDES[: 2 * ndim]:
            raise ValueError(
                f"StructuredBox names the sides {list(_SIDES[: 2 * ndim])}; the "
                f"box has {list(domain.boundaries())}"
            )
        bounds = list(domain.bounds())
        meshes: List[GalerkinMeshProtocol[_Np]] = []
        for n in self._sizes:
            if ndim == 1:
                meshes.append(StructuredMesh1D(nx=n, bounds=bounds[0], bkd=_BKD))
            elif ndim == 2:
                element = "tri" if self._simplex else "quad"
                meshes.append(
                    StructuredMesh2D(n, n, bounds, _BKD, element_type=element)
                )
            else:
                element = "tet" if self._simplex else "hex"
                meshes.append(
                    StructuredMesh3D(n, n, n, bounds, _BKD, element_type=element)
                )
        return meshes

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._sizes)}, simplex={self._simplex})"


class _SkfemTriangles:
    """A triangle mesh held as an skfem mesh with named boundaries."""

    def __init__(self, mesh: MeshTri) -> None:
        self._mesh = mesh

    def bkd(self) -> NumpyBkd:
        return _BKD

    def ndim(self) -> int:
        return 2

    def nelements(self) -> int:
        return int(self._mesh.nelements)

    def nnodes(self) -> int:
        return int(self._mesh.nvertices)

    def nodes(self) -> _Np:
        return np.asarray(self._mesh.p, dtype=float)

    def elements(self) -> _Np:
        return np.asarray(self._mesh.t, dtype=np.int64)

    def skfem_mesh(self) -> MeshTri:
        return self._mesh

    def boundary_nodes(self, boundary_id: str) -> _Np:
        boundaries = self._mesh.boundaries or {}
        if boundary_id not in boundaries:
            raise ValueError(
                f"Unknown boundary_id '{boundary_id}'. Available: {list(boundaries)}"
            )
        facets = boundaries[boundary_id]
        return np.unique(self._mesh.facets[:, facets]).astype(np.int64)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nelements={self.nelements()})"


class SkfemDisk:
    """Polygonal triangle meshes of a :class:`~pyapprox.pde.verification.
    domains.Disk` (skfem ``MeshTri.init_circle``), boundary ``circle``.

    The polygon's vertices lie on the circle, so the domain differs from
    the disk by :math:`O(h^2)`.

    Parameters
    ----------
    refinements : Sequence[int]
        Refinements of skfem's initial disk, one entry per level.
    """

    def __init__(self, refinements: Sequence[int]) -> None:
        self._refinements = tuple(refinements)

    def meshes(self, domain: DomainProtocol) -> List[GalerkinMeshProtocol[_Np]]:
        """One mesh per refinement.

        Raises
        ------
        TypeError
            If ``domain`` is not a ``Disk``.
        """
        if not isinstance(domain, Disk):
            raise TypeError(f"SkfemDisk meshes a Disk, got {type(domain).__name__}")
        center = np.array(domain.center())[:, None]
        meshes: List[GalerkinMeshProtocol[_Np]] = []
        for nrefs in self._refinements:
            unit = MeshTri.init_circle(nrefs)
            mesh = MeshTri(
                domain.radius() * unit.p + center,
                unit.t,
                _boundaries={"circle": unit.boundary_facets()},
            )
            meshes.append(_SkfemTriangles(mesh))
        return meshes

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self._refinements)})"


class BasisFactoryProtocol(Protocol):
    """Builds a basis on a mesh."""

    def build(self, mesh: GalerkinMeshProtocol[_Np]) -> LagrangeBasis[_Np]:
        """The basis on ``mesh``."""
        ...


class Lagrange:
    """Continuous Lagrange elements of one degree.

    Parameters
    ----------
    degree : int
        The polynomial degree (1 or 2).
    """

    def __init__(self, degree: int) -> None:
        self._degree = degree

    def degree(self) -> int:
        """The polynomial degree."""
        return self._degree

    def build(self, mesh: GalerkinMeshProtocol[_Np]) -> LagrangeBasis[_Np]:
        """The degree-``degree`` Lagrange basis on ``mesh``."""
        return LagrangeBasis(mesh, degree=self._degree)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._degree})"


def _mesh_size(basis: LagrangeBasis[_Np]) -> float:
    """The largest element diameter."""
    mesh = basis.skfem_basis().mesh
    vertices = np.asarray(mesh.p)[:, np.asarray(mesh.t)]
    differences = vertices[:, :, None, :] - vertices[:, None, :, :]
    return float(np.sqrt((differences**2).sum(axis=0)).max())


class GalerkinApproximation:
    """One level's Galerkin solution of a field, with errors against its
    exact solution.

    Parameters
    ----------
    problem : ManufacturedProblem
        The problem solved.
    row : str
        The field the solution approximates.
    basis : LagrangeBasis
        The basis of the solution.
    state : NDArray
        The solution's DOFs. Shape: ``(ndofs,)``.
    intorder : int
        The quadrature order of the L2 error.
    """

    def __init__(
        self,
        problem: ManufacturedProblem,
        row: str,
        basis: LagrangeBasis[_Np],
        state: _Np,
        intorder: int,
    ) -> None:
        self._basis, self._state, self._intorder = basis, state, intorder
        self._exact = SkfemPoints([problem.solution(row)], problem)
        self._h = _mesh_size(basis)

    def h(self) -> float:
        """The largest element diameter."""
        return self._h

    def state(self) -> _Np:
        """The solution's DOFs. Shape: ``(ndofs,)``."""
        return self._state

    def l2_error(self) -> float:
        r""":math:`\|u^* - u_h\|_{L^2}`, integrated by skfem quadrature of
        order ``intorder`` against :math:`u^*` itself."""
        skfem_basis = self._basis.skfem_basis()
        fine = Basis(skfem_basis.mesh, skfem_basis.elem, intorder=self._intorder)
        exact = self._exact

        def squared_error(w: Any) -> _Np:
            return np.asarray((w["uh"] - exact(np.asarray(w.x))[0]) ** 2)

        value = Functional(squared_error).assemble(
            fine, uh=fine.interpolate(self._state)
        )
        return float(np.sqrt(value))

    def nodal_error(self) -> float:
        r""":math:`\max_i |u^*(x_i) - u_h(x_i)|` over the DOF locations."""
        coords = np.asarray(self._basis.dof_coordinates())
        return float(np.abs(self._state - self._exact(coords)[0]).max())


class GalerkinDiscretizer:
    """Solves a single-field steady problem with this stack on every mesh
    of a family.

    Each level: the factory's system on ``basis.build(mesh)``, the steady
    residual (``galerkin_steady_residual``), and ``SteadySystem`` with
    sparse LU, from a zero initial guess.

    Parameters
    ----------
    meshes : MeshFamilyProtocol
        The meshes.
    basis : BasisFactoryProtocol
        The basis on each mesh.
    factory : SystemFactoryProtocol
        The system of a problem on a basis.
    nonlinear : NonlinearSolverProtocol, optional
        Default: Newton with line search, ``atol = rtol = 1e-12``, 30
        steps (a linear system is solved in one step).
    intorder : int, optional
        The quadrature order of L2 errors. Default: three more than twice
        the basis degree.
    """

    def __init__(
        self,
        meshes: MeshFamilyProtocol,
        basis: BasisFactoryProtocol,
        factory: SystemFactoryProtocol,
        nonlinear: Optional[NonlinearSolverProtocol[_Np, _Np]] = None,
        intorder: Optional[int] = None,
    ) -> None:
        self._meshes, self._basis, self._factory = meshes, basis, factory
        self._nonlinear: NonlinearSolverProtocol[_Np, _Np] = (
            Newton(atol=1e-12, rtol=1e-12, maxiters=30, line_search=True)
            if nonlinear is None
            else nonlinear
        )
        self._intorder = intorder

    def approximations(
        self, problem: ManufacturedProblem
    ) -> List[GalerkinApproximation]:
        """One approximation per mesh, coarsest first.

        Raises
        ------
        ValueError
            If ``problem`` is transient, has no domain, or has more than
            one field.
        RuntimeError
            If a solve does not converge.
        """
        # Only the solve below is steady; the helpers and approximations
        # serve a transient discretizer unchanged.
        if problem.is_transient():
            raise ValueError(
                "GalerkinDiscretizer solves steady problems; this one depends on t"
            )
        domain = problem.domain()
        if domain is None:
            raise ValueError("GalerkinDiscretizer needs a domain; this is an ODE")
        if len(problem.fields()) != 1:
            raise ValueError(
                f"GalerkinDiscretizer solves single-field problems, got "
                f"{list(problem.fields())}"
            )
        (row,) = problem.fields()
        approximations = []
        for mesh in self._meshes.meshes(domain):
            basis = self._basis.build(mesh)
            system = self._factory.system(problem, basis)
            discretization = skfem_discretization(system)
            space = discretization.space()
            if not isinstance(space, InnerProductSpaceProtocol):
                raise TypeError(
                    f"the steady solve needs an inner-product state space, got "
                    f"{type(space).__name__}"
                )
            steady = SteadySystem(
                galerkin_steady_residual(discretization),
                space,
                KeepAll(ScipySparseLU()),
            )
            point = steady.solve(self._nonlinear, np.zeros((system.nstates(), 1)))
            if not point.converged():
                raise RuntimeError(
                    f"the solve on {mesh!r} did not converge: {point.info().message}"
                )
            intorder = (
                self._intorder if self._intorder is not None else 2 * basis.degree() + 3
            )
            approximations.append(
                GalerkinApproximation(
                    problem, row, basis, point.state()[:, 0], intorder
                )
            )
        return approximations

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._meshes!r}, {self._basis!r}, "
            f"{self._factory!r})"
        )
