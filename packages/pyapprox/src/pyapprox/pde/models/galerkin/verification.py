r"""Verify Galerkin discretizations with :mod:`pyapprox.pde.verification`.

The adapter between the stack-independent verification core and this
repository's skfem Galerkin stack. Bound to skfem and NumPy.

- Mesh families (:class:`StructuredBox`, :class:`SkfemDisk`) refine a
  domain, naming mesh boundaries as the domain names its parts.
- Basis factories (:class:`Lagrange`, :class:`VectorLagrange`,
  :class:`TaylorHood`) build a basis on each mesh. A vector basis
  carries one field per component, in the problem's field order; a
  mixed basis takes the fields in order, as many per basis field as it
  has components (Taylor-Hood: the velocity components, then pressure).
- A system factory, supplied by the case, builds the physics under test
  on a basis of the kind it names; pairing it with a factory of another
  kind is a type error. Nothing here knows any physics: a factory builds
  its physics class from the generic helpers :func:`field`,
  :func:`vector_field`, :func:`state_law`, :func:`quadrature_values`,
  :func:`forcing`, :func:`vector_forcing`, :func:`boundary_conditions`
  and :func:`vector_boundary_conditions`, so each test exercises the
  class it names.
  The helpers declare each supplier's time dependence from its
  expression (it uses ``t`` or not), so they serve transient problems
  unchanged. :func:`field`, :func:`vector_field` and :func:`state_law`
  need only a scope of names and coordinates, not a manufactured
  problem, so text coefficients of a production problem pass through
  them too. :func:`exact_dof_values` gives the exact solution at a
  basis's DOFs, for a case that compares one solve directly.
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

from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

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
from pyapprox.pde.galerkin.basis import (
    LagrangeBasis,
    MixedBasis,
    VectorLagrangeBasis,
)
from pyapprox.pde.galerkin.basis.mixed import MixedFieldBasisProtocol
from pyapprox.pde.galerkin.boundary.implementations import (
    DirectDirichletBC,
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.mesh import (
    PeriodicStructuredMesh1D,
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.pde.galerkin.protocols.basis import (
    BoundaryDofsProtocol,
    ComponentDofsBasisProtocol,
    GalerkinBasisProtocol,
    NaturalBoundaryBasisProtocol,
)
from pyapprox.pde.galerkin.protocols.mesh import GalerkinMeshProtocol
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    ConditionVisitorProtocol,
    Dirichlet,
    Neumann,
    Periodic,
    Robin,
    Unprescribed,
)
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.expressions.names import Names
from pyapprox.util.expressions.parsing import Value, parse
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton, NonlinearSolverProtocol

_Np = NDArray[Any]
_BKD = NumpyBkd()
_SIDES = ("left", "right", "bottom", "top", "back", "front")
# The basis a physics is built on: a factory producing one kind and a
# system needing another do not type-check together. A mixed basis holds
# several fields in one state.
_AnyBasis = Union[GalerkinBasisProtocol[_Np], MixedBasis[_Np]]
_B = TypeVar("_B", bound=_AnyBasis)
_B_co = TypeVar("_B_co", bound=_AnyBasis, covariant=True)
_B_contra = TypeVar("_B_contra", bound=_AnyBasis, contravariant=True)


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


class _NaturalVectorData:
    """:class:`_NaturalData` with one expression per component, shape
    ``(ncomponents, npts)``."""

    def __init__(self, data: Sequence[sp.Expr], scope: ExpressionScopeProtocol) -> None:
        self._points = SkfemPoints(data, scope)
        self._timed = any(time() in d.free_symbols for d in data)

    def evaluate(self, coords: _Np, normal: _Np, time: float) -> _Np:
        return -self._points(coords, normals=normal, t=time)

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


def vector_forcing(problem: ManufacturedProblem, rows: Sequence[str]) -> _VectorField:
    """The forcings of ``rows`` as one vector ``f(coords, time)``, shape
    ``(len(rows), ...)``: a vector physics' body force."""
    return _VectorField([problem.forcing(row) for row in rows], problem)


def quadrature_values(
    scope: ExpressionScopeProtocol,
    expression: Value,
    basis: GalerkinBasisProtocol[_Np],
) -> _Np:
    """A steady coefficient at the quadrature points of ``basis``, for a
    physics that takes coefficients as values there. Shape: ``(nelements,
    nquad)``.

    Raises
    ------
    ValueError
        If the expression uses anything but the coordinates and constants.
    """
    parsed = _parsed(scope, expression, ())
    points = np.asarray(basis.skfem_basis().global_coordinates().value)
    if parsed.is_number:
        return np.full(points.shape[1:], float(parsed))
    values: _Np = SkfemPoints([parsed], scope)(points)[0]
    return values


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

    def periodic(self, condition: Periodic) -> None:
        # Periodicity lives in the mesh topology, which identifies the
        # wrapped sides; there is no condition to build.
        return None

    def unprescribed(self, condition: Unprescribed) -> None:
        # A part with no condition is this stack's do-nothing condition:
        # no natural data, so zero diffusive flux, with the advective
        # boundary term the physics keeps.
        return None


def _boundary_parts(problem: ManufacturedProblem) -> Tuple[str, ...]:
    """The boundary parts of the problem's domain."""
    domain = problem.domain()
    return () if domain is None else domain.boundaries()


def _check_periodic(
    part: str, periodic: bool, basis: GalerkinBasisProtocol[_Np]
) -> None:
    """Refuse a periodic part the mesh still has as a boundary, or a
    non-periodic part it does not."""
    if periodic == (part in set(basis.skfem_basis().mesh.boundaries or {})):
        raise ValueError(
            f"{part!r} is {'periodic' if periodic else 'not periodic'} but "
            f"the mesh {'has' if periodic else 'does not have'} it as a "
            "boundary; wrap periodic sides in the mesh family "
            "(StructuredBox(..., periodic=...)) and only those"
        )


def boundary_conditions(
    problem: ManufacturedProblem, basis: LagrangeBasis[_Np], row: str
) -> List[BoundaryConditionRole[_Np]]:
    """One Galerkin boundary condition per non-periodic boundary part of
    ``problem`` for ``row``, with data from the problem's expressions.

    Periodic parts get none: this stack makes a side periodic by
    identifying it in the mesh (``StructuredBox(..., periodic=...)``), so
    a periodic part must be one the mesh has wrapped away, and every
    other part must be a boundary of the mesh.

    Unprescribed parts get none either: that is this stack's do-nothing
    condition, zero diffusive flux with the physics' advective boundary
    term kept, exact for a row without diffusion. On a row with diffusion
    the exact solution must have zero diffusive flux there; that is not
    checked.

    Raises
    ------
    ValueError
        If ``row`` takes no boundary conditions, a periodic part is a
        boundary of the mesh, or another part is not.
    """
    if row not in problem.boundary_rows():
        raise ValueError(
            f"row {row!r} takes no boundary conditions; the boundary rows are "
            f"{list(problem.boundary_rows())}"
        )
    conditions: List[BoundaryConditionRole[_Np]] = []
    for part in _boundary_parts(problem):
        condition: ConditionProtocol = problem.conditions(part)[row]
        _check_periodic(part, isinstance(condition, Periodic), basis)
        # Every kind builds a condition except periodic, which builds none.
        visitor: ConditionVisitorProtocol[Optional[BoundaryConditionRole[_Np]]] = (
            _GalerkinCondition(problem, basis, part, row)
        )
        built = condition.accept(visitor)
        if built is not None:
            conditions.append(built)
    return conditions


@dataclass(frozen=True)
class _Essential:
    """A row whose values a part prescribes."""


@dataclass(frozen=True)
class _Natural:
    """A row whose natural data a part prescribes: Robin with coefficient
    ``alpha``, or Neumann when ``alpha`` is ``None``."""

    alpha: Optional[sp.Expr]


@dataclass(frozen=True)
class _Free:
    """A row a part prescribes nothing for."""


_Kind = Union[_Essential, _Natural, _Free]


class _RowKind:
    """What one row's condition on a part prescribes; ``None`` for a
    periodic row, whose part the mesh wraps away."""

    def __init__(self, names: Names) -> None:
        self._names = names

    def dirichlet(self, condition: Dirichlet) -> _Kind:
        return _Essential()

    def neumann(self, condition: Neumann) -> _Kind:
        return _Natural(None)

    def robin(self, condition: Robin) -> _Kind:
        return _Natural(self._names.substitute_constants(condition.alpha(self._names)))

    def periodic(self, condition: Periodic) -> None:
        return None

    def unprescribed(self, condition: Unprescribed) -> _Kind:
        return _Free()


def vector_boundary_conditions(
    problem: ManufacturedProblem,
    basis: Union[VectorLagrangeBasis[_Np], MixedBasis[_Np]],
    rows: Sequence[str],
    field_name: Optional[str] = None,
) -> List[BoundaryConditionRole[_Np]]:
    """The Galerkin boundary conditions of a vector field whose components
    are ``rows``, in order, with data from the problem's expressions: the
    field of a vector basis, or the vector field ``field_name`` of a mixed
    basis.

    On each part the rows' conditions combine into at most two of this
    stack's: the Dirichlet rows into one ``DirichletBC`` on their
    components, and the others into one ``NeumannBC`` (all Neumann) or
    ``RobinBC`` with vector data, zero in the Dirichlet components (whose
    equations the constraint replaces). The Robin coefficient is shared
    when every row is Robin with one coefficient, and otherwise one per
    component, zero on Neumann rows, so a part may mix all three kinds.
    Unprescribed rows take zero natural data and no coefficient, which in
    this stack prescribes nothing (see :func:`boundary_conditions`).
    Periodic parts are handled as in :func:`boundary_conditions`.

    Raises
    ------
    ValueError
        If ``field_name`` is not given exactly for a mixed basis or names no
        vector field of it, ``rows`` are not one per component, a row
        takes no boundary conditions, or a part is periodic for some rows
        only or does not match the mesh's periodicity.
    """
    rows = tuple(rows)
    target: BoundaryDofsProtocol[_Np]
    natural_basis: NaturalBoundaryBasisProtocol[_Np]
    if isinstance(basis, MixedBasis):
        if field_name is None:
            raise ValueError("name the field of the mixed basis that rows make up")
        view = basis.field(field_name)
        if not isinstance(view, ComponentDofsBasisProtocol):
            raise ValueError(f"field {field_name!r} of the mixed basis is not a vector")
        target, natural_basis = view, view
        ncomponents = view.ncomponents()
        field_basis = basis.field_basis(field_name)
    else:
        if field_name is not None:
            raise ValueError(f"field {field_name!r} given, but the basis is not mixed")
        target, natural_basis, field_basis = basis, basis, basis
        ncomponents = basis.ncomponents()
    if len(rows) != ncomponents:
        raise ValueError(
            f"one row per component of the basis ({ncomponents}), got {list(rows)}"
        )
    missing = [row for row in rows if row not in problem.boundary_rows()]
    if missing:
        raise ValueError(
            f"rows {missing} take no boundary conditions; the boundary rows "
            f"are {list(problem.boundary_rows())}"
        )
    visitor: ConditionVisitorProtocol[Optional[_Kind]] = _RowKind(problem.names())
    zero = sp.Integer(0)
    conditions: List[BoundaryConditionRole[_Np]] = []
    for part in _boundary_parts(problem):
        by_row = problem.conditions(part)
        kinds = [by_row[row].accept(visitor) for row in rows]
        periodic = [kind is None for kind in kinds]
        if any(periodic) and not all(periodic):
            raise ValueError(f"{part!r} is periodic for some rows only")
        _check_periodic(part, all(periodic), field_basis)
        if all(periodic):
            continue
        data = problem.boundary_data(part)
        essential = tuple(
            i for i, kind in enumerate(kinds) if isinstance(kind, _Essential)
        )
        if essential:
            values = [
                data[row] if i in essential else zero for i, row in enumerate(rows)
            ]
            # A one-component basis takes scalar values.
            supplier: Union[_Field, _VectorField] = (
                _Field(values[0], problem)
                if len(rows) == 1
                else _VectorField(values, problem)
            )
            components = None if len(essential) == len(rows) else essential
            conditions.append(
                DirichletBC(target, part, supplier, _BKD, components=components)
            )
        natural = [kind for kind in kinds if isinstance(kind, _Natural)]
        if not natural:
            continue
        # Zero natural data on the Dirichlet rows, whose equations the
        # constraint replaces, and on the unprescribed ones.
        flux = _NaturalVectorData(
            [
                data[row] if isinstance(kind, _Natural) else zero
                for kind, row in zip(kinds, rows)
            ],
            problem,
        )
        alphas = {kind.alpha for kind in natural}
        if alphas == {None}:
            conditions.append(NeumannBC(natural_basis, part, flux, _BKD))
            continue
        shared = next(iter(alphas))
        coefficient: Union[float, _Field, _VectorField]
        if len(alphas) == 1 and shared is not None and len(natural) == len(rows):
            coefficient = field(problem, shared)
        else:
            # One per component: zero on Neumann rows, on Dirichlet
            # rows, whose equations the constraint replaces, and on
            # unprescribed rows.
            coefficient = _VectorField(
                [
                    kind.alpha
                    if isinstance(kind, _Natural) and kind.alpha is not None
                    else zero
                    for kind in kinds
                ],
                problem,
            )
        conditions.append(RobinBC(natural_basis, part, coefficient, flux, _BKD))
    return conditions


def pinned_value(
    problem: ManufacturedProblem, basis: MixedBasis[_Np], field_name: str, row: str
) -> DirectDirichletBC[_Np]:
    """The first DOF of scalar field ``field_name`` of a mixed basis fixed at
    the exact solution of ``row`` there.

    Removes the constant a field is determined only up to, such as the
    pressure when the velocity is prescribed on the whole boundary; any
    one DOF does, and fixing it at the exact value leaves the discrete
    solution's error unchanged in order.

    Raises
    ------
    ValueError
        If ``field_name`` is a vector field.
    """
    if isinstance(basis.field(field_name), ComponentDofsBasisProtocol):
        raise ValueError(f"field {field_name!r} is a vector; pin a scalar field")
    dof = basis.offset(field_name)
    point = np.asarray(basis.dof_coordinates())[:, dof : dof + 1]
    value = SkfemPoints([problem.solution(row)], problem)(point)[0]
    return DirectDirichletBC(np.array([dof]), value, _BKD)


def exact_dof_values(
    problem: ManufacturedProblem,
    basis: Union[LagrangeBasis[_Np], VectorLagrangeBasis[_Np]],
    rows: Sequence[str],
    time: float = 0.0,
) -> _Np:
    """The exact solution of ``rows`` (one per component of ``basis``) at
    its DOF locations at ``time``, each DOF its own component's value.
    Shape: ``(ndofs,)``.

    Raises
    ------
    ValueError
        If ``rows`` does not have one entry per component of ``basis``.
    """
    dofs = np.arange(basis.ndofs())
    if isinstance(basis, VectorLagrangeBasis):
        ncomponents = basis.ncomponents()
        components = np.asarray(basis.dof_components(dofs))
    else:
        ncomponents, components = 1, np.zeros_like(dofs)
    if len(rows) != ncomponents:
        raise ValueError(
            f"one row per component of the basis ({ncomponents}), got {list(rows)}"
        )
    coords = np.asarray(basis.dof_coordinates())
    exact = SkfemPoints([problem.solution(row) for row in rows], problem)
    values: _Np = exact(coords, t=time)[components, dofs]
    return values


class SystemFactoryProtocol(Protocol[_B_contra]):
    """Builds the physics under test, with its conditions, on a basis."""

    def system(
        self, problem: ManufacturedProblem, basis: _B_contra
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
    periodic : Sequence[str]
        Coordinates whose two sides are identified, so the mesh has no
        boundary there; the problem must declare those sides
        :class:`~pyapprox.pde.verification.conditions.Periodic`. This
        stack has periodic meshes in 1D only.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        simplex: bool = False,
        periodic: Sequence[str] = (),
    ) -> None:
        self._sizes, self._simplex = tuple(sizes), simplex
        self._periodic = tuple(periodic)

    def meshes(self, domain: DomainProtocol) -> List[GalerkinMeshProtocol[_Np]]:
        """One structured mesh per size.

        Raises
        ------
        TypeError
            If ``domain`` is not a ``Box``.
        ValueError
            If its sides are not the standard ``left``, ``right``,
            ``bottom``, ``top``, ``back``, ``front``, which name this
            stack's mesh boundaries; or a periodic coordinate is not one
            of the box's, or the box is not 1D.
        """
        if not isinstance(domain, Box):
            raise TypeError(f"StructuredBox meshes a Box, got {type(domain).__name__}")
        ndim = domain.ndim()
        if domain.boundaries() != _SIDES[: 2 * ndim]:
            raise ValueError(
                f"StructuredBox names the sides {list(_SIDES[: 2 * ndim])}; the "
                f"box has {list(domain.boundaries())}"
            )
        unknown = set(self._periodic) - set(domain.coordinates())
        if unknown:
            raise ValueError(
                f"periodic coordinates {sorted(unknown)} are not the box's "
                f"{list(domain.coordinates())}"
            )
        if self._periodic and ndim != 1:
            raise ValueError(
                f"this stack has periodic meshes in 1D only; the box is {ndim}D"
            )
        bounds = list(domain.bounds())
        meshes: List[GalerkinMeshProtocol[_Np]] = []
        for n in self._sizes:
            if self._periodic:
                meshes.append(PeriodicStructuredMesh1D(n, bounds[0], _BKD))
            elif ndim == 1:
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
        return (
            f"{type(self).__name__}({list(self._sizes)}, simplex={self._simplex}, "
            f"periodic={list(self._periodic)})"
        )


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


class BasisFactoryProtocol(Protocol[_B_co]):
    """Builds a basis on a mesh."""

    def build(self, mesh: GalerkinMeshProtocol[_Np]) -> _B_co:
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


class VectorLagrange:
    """Continuous vector Lagrange elements of one degree, one component
    per coordinate, DOFs interleaved per node.

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

    def build(self, mesh: GalerkinMeshProtocol[_Np]) -> VectorLagrangeBasis[_Np]:
        """The degree-``degree`` vector Lagrange basis on ``mesh``."""
        return VectorLagrangeBasis(mesh, degree=self._degree)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._degree})"


class TaylorHood:
    """Taylor-Hood elements: field ``velocity``, vector Lagrange of degree
    k, then field ``pressure``, scalar Lagrange of degree k - 1.

    Parameters
    ----------
    velocity_degree : int
        The velocity degree k. Default 2.

    Raises
    ------
    ValueError
        If ``velocity_degree`` is below 2: P1-P0 is not a continuous
        pressure space.
    """

    def __init__(self, velocity_degree: int = 2) -> None:
        if velocity_degree < 2:
            raise ValueError(
                f"Taylor-Hood needs velocity degree 2 or more, got {velocity_degree}"
            )
        self._degree = velocity_degree

    def build(self, mesh: GalerkinMeshProtocol[_Np]) -> MixedBasis[_Np]:
        """The ``[velocity | pressure]`` mixed basis on ``mesh``."""
        fields: List[Tuple[str, MixedFieldBasisProtocol[_Np]]] = [
            ("velocity", VectorLagrangeBasis(mesh, degree=self._degree)),
            ("pressure", LagrangeBasis(mesh, degree=self._degree - 1)),
        ]
        return MixedBasis(fields)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._degree})"


@dataclass(frozen=True)
class _Block:
    """A field of the state: its rows, its basis, and its first DOF."""

    rows: Tuple[str, ...]
    basis: GalerkinBasisProtocol[_Np]
    offset: int


def _blocks(basis: _AnyBasis, rows: Sequence[str]) -> List[_Block]:
    """The state's fields, taking ``rows`` in order: one block for a scalar
    or vector basis, one per field of a mixed basis, each with as many
    rows as its basis has components.

    Raises
    ------
    ValueError
        If ``rows`` are not one per component of the basis.
    """
    fields: List[Tuple[GalerkinBasisProtocol[_Np], int]] = (
        [(basis.field_basis(name), basis.offset(name)) for name in basis.field_names()]
        if isinstance(basis, MixedBasis)
        else [(basis, 0)]
    )
    counts = [
        field_basis.ncomponents()
        if isinstance(field_basis, ComponentDofsBasisProtocol)
        else 1
        for field_basis, _ in fields
    ]
    if len(rows) != sum(counts):
        raise ValueError(
            f"one field per component of the basis ({sum(counts)}), got {list(rows)}"
        )
    blocks, start = [], 0
    for (field_basis, offset), count in zip(fields, counts):
        blocks.append(_Block(tuple(rows[start : start + count]), field_basis, offset))
        start += count
    return blocks


def _degree(basis: _AnyBasis) -> int:
    """The polynomial degree; of a mixed basis, its fields' highest."""
    if isinstance(basis, MixedBasis):
        return max(basis.field_basis(name).degree() for name in basis.field_names())
    return basis.degree()


def _mesh_size(basis: GalerkinBasisProtocol[_Np]) -> float:
    """The largest element diameter.

    Vertices come from each element's own mapping of the reference
    element, not from ``mesh.p[:, mesh.t]``: a periodic mesh stores
    coordinates per element while its connectivity indexes the
    identified nodes, so the wrapping element would span the domain."""
    mesh = basis.skfem_basis().mesh
    # Shape (ndim, nelements, nvertices).
    vertices = np.asarray(mesh.mapping().F(mesh.init_refdom().p))
    differences = vertices[:, :, :, None] - vertices[:, :, None, :]
    return float(np.sqrt((differences**2).sum(axis=0)).max())


class GalerkinApproximation:
    """One level's Galerkin solution, with errors against the exact
    solution: of a scalar or vector field, or of the fields of a mixed
    basis, optionally only some of them.

    Errors combine over the measured fields: the L2 error is the root of
    the sum of their squared L2 errors, the nodal error their largest.
    Fields converging at different rates (Taylor-Hood velocity and
    pressure) are measured apart.

    Parameters
    ----------
    problem : ManufacturedProblem
        The problem solved.
    rows : Sequence[str]
        The fields the solution approximates, one per component of
        ``basis`` in order (one for a scalar basis; for a mixed basis,
        each basis field's components in field order).
    basis : GalerkinBasisProtocol or MixedBasis
        The basis of the solution.
    state : NDArray
        The solution's DOFs. Shape: ``(ndofs,)``.
    intorder : int
        The quadrature order of the L2 error.
    measured : Sequence[str], optional
        The rows whose error is measured, whole basis fields only.
        Default: all.

    Raises
    ------
    ValueError
        If ``rows`` are not one per component, or ``measured`` is empty,
        names an unknown row, or splits a basis field.
    """

    def __init__(
        self,
        problem: ManufacturedProblem,
        rows: Sequence[str],
        basis: _AnyBasis,
        state: _Np,
        intorder: int,
        measured: Optional[Sequence[str]] = None,
    ) -> None:
        self._state, self._intorder = state, intorder
        blocks = _blocks(basis, rows)
        if measured is not None:
            chosen = set(measured)
            unknown = chosen - set(rows)
            if not chosen or unknown:
                raise ValueError(
                    f"measure some of the rows {list(rows)}, got {list(measured)}"
                )
            split = [
                block.rows
                for block in blocks
                if chosen & set(block.rows) and not set(block.rows) <= chosen
            ]
            if split:
                raise ValueError(
                    f"measure whole basis fields; {list(measured)} splits {split}"
                )
            blocks = [block for block in blocks if set(block.rows) <= chosen]
        self._blocks = blocks
        self._exact = [
            SkfemPoints([problem.solution(row) for row in block.rows], problem)
            for block in blocks
        ]
        self._h = _mesh_size(blocks[0].basis)

    def h(self) -> float:
        """The largest element diameter."""
        return self._h

    def state(self) -> _Np:
        """The solution's DOFs. Shape: ``(ndofs,)``."""
        return self._state

    def l2_error(self) -> float:
        r""":math:`\|u^* - u_h\|_{L^2}`, integrated by skfem quadrature of
        order ``intorder`` against :math:`u^*` itself; for a vector field,
        of the Euclidean norm of the pointwise error; over the measured
        fields."""
        return float(
            np.sqrt(
                sum(
                    self._squared_l2_error(block, exact)
                    for block, exact in zip(self._blocks, self._exact)
                )
            )
        )

    def _field_state(self, block: _Block) -> _Np:
        return self._state[block.offset : block.offset + block.basis.ndofs()]

    def _squared_l2_error(self, block: _Block, exact: SkfemPoints) -> float:
        skfem_basis = block.basis.skfem_basis()
        fine = Basis(skfem_basis.mesh, skfem_basis.elem, intorder=self._intorder)
        nrows = len(block.rows)

        def squared_error(w: Any) -> _Np:
            x = np.asarray(w.x)
            uh = np.asarray(w["uh"].value).reshape((nrows,) + x.shape[1:])
            return np.asarray(((uh - exact(x)) ** 2).sum(axis=0))

        value = Functional(squared_error).assemble(
            fine, uh=fine.interpolate(self._field_state(block))
        )
        return float(value)

    def nodal_error(self) -> float:
        r""":math:`\max_i |u^*(x_i) - u_h(x_i)|` over the DOF locations of
        the measured fields, each DOF against its own component."""
        return max(
            self._nodal_error(block, exact)
            for block, exact in zip(self._blocks, self._exact)
        )

    def _nodal_error(self, block: _Block, exact: SkfemPoints) -> float:
        coords = np.asarray(block.basis.dof_coordinates())
        dofs = np.arange(block.basis.ndofs())
        components = (
            np.asarray(block.basis.dof_components(dofs))
            if isinstance(block.basis, ComponentDofsBasisProtocol)
            else np.zeros_like(dofs)
        )
        values = exact(coords)[components, dofs]
        return float(np.abs(self._field_state(block) - values).max())


class GalerkinDiscretizer(Generic[_B]):
    """Solves a steady problem with this stack on every mesh of a family:
    one field on a scalar basis, one field per component on a vector
    basis, or each basis field's components in turn on a mixed basis, in
    the problem's field order.

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
        the basis degree (a mixed basis's highest).
    measured : Sequence[str], optional
        The fields whose errors the approximations report, whole basis
        fields only (see :class:`GalerkinApproximation`). Default: all.
    """

    def __init__(
        self,
        meshes: MeshFamilyProtocol,
        basis: BasisFactoryProtocol[_B],
        factory: SystemFactoryProtocol[_B],
        nonlinear: Optional[NonlinearSolverProtocol[_Np, _Np]] = None,
        intorder: Optional[int] = None,
        measured: Optional[Sequence[str]] = None,
    ) -> None:
        self._meshes, self._basis, self._factory = meshes, basis, factory
        self._measured = None if measured is None else tuple(measured)
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
            If ``problem`` is transient, has no domain, or does not have
            one field per component of the basis.
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
        rows = problem.fields()
        approximations = []
        for mesh in self._meshes.meshes(domain):
            basis = self._basis.build(mesh)
            try:
                _blocks(basis, rows)
            except ValueError as error:
                raise ValueError(f"GalerkinDiscretizer needs {error}") from None
            system = self._factory.system(problem, basis)
            discretization = skfem_discretization(system)
            steady = SteadySystem(
                galerkin_steady_residual(discretization),
                discretization.space(),
                KeepAll(ScipySparseLU()),
            )
            point = steady.solve(self._nonlinear, np.zeros((system.nstates(), 1)))
            if not point.converged():
                raise RuntimeError(
                    f"the solve on {mesh!r} did not converge: {point.info().message}"
                )
            intorder = (
                self._intorder if self._intorder is not None else 2 * _degree(basis) + 3
            )
            approximations.append(
                GalerkinApproximation(
                    problem,
                    rows,
                    basis,
                    point.state()[:, 0],
                    intorder,
                    measured=self._measured,
                )
            )
        return approximations

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._meshes!r}, {self._basis!r}, "
            f"{self._factory!r})"
        )
