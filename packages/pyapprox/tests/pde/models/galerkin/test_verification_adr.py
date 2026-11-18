r"""Galerkin advection-diffusion-reaction verified by manufactured solutions.

The problem is stated as terms, its strong form
:math:`\nabla\cdot(-D\nabla u) + v\cdot\nabla u + [(\nabla\cdot v)\, u]
- R(u) = f` (the bracket present for the conservative form, whose
advection is :math:`\nabla\cdot(v u)`). Natural data prescribe the
diffusive flux in both forms, and the physics under test,
:class:`~pyapprox.pde.galerkin.physics.AdvectionDiffusionReaction`, is
built from the same coefficient strings by a system factory. Errors are
true :math:`L^2` errors, integrated by skfem quadrature against
:math:`u^*`.

Cases (fitted rate over the last three levels, tolerance 0.1):

- P1 in 1D (cells 8-64), 2D quads (4-32) and 3D hexes (2-8), and P2 in
  1D and 2D (one order higher): Dirichlet, Neumann and Robin sides
  together, variable :math:`D`, variable :math:`v`, cubic reaction,
  both advection forms;
- the unit disk, P1 triangles (refinements 2-5), Robin everywhere: rate
  2 despite the :math:`O(h^2)` polygonal boundary;
- a variable Robin coefficient :math:`1 + x` on a square side;
- P2 exact recovery (L2 error below 1e-10, stricter than the old
  suite's relative 1e-8 at the DOFs) of quadratics in 1D and 2D, quads
  and triangles, with natural conditions; the old suite's cases
  (1D: 24 plus 4 conservative; 2D: 9, and a Robin alpha varying along
  the sides) and the 2D cases with advection and reaction switched on,
  in both forms;
- periodic ends in 1D (the mesh identifies them), P1 and P2 rates in
  both advection forms;
- a do-nothing outflow, a side with no condition object, in 1D and 2D
  and both advection forms: P1 rate 2 for solutions with zero diffusive
  flux and nonzero advective flux leaving there;
- wrong-model controls, fitted rate below 0.5: natural data with the
  normal reversed; Robin without its spring term; a non-periodic
  solution on periodic ends; the conservative form without its
  advective boundary term on the do-nothing outflow.

Refusals: periodic sides the mesh does not wrap and wrapped sides with
ordinary conditions, periodic meshes beyond 1D, transient and
multi-field problems, a box with named coordinates.
"""

from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    TypeVar,
)

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from pyapprox.pde.boundary.protocols import BoundaryConditionRole
from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    FieldExpression,
    NonConservativeAdvection,
)
from pyapprox.pde.expressions.symbols import coordinates
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import (
    DirichletBC,
    NeumannBC,
    RobinBC,
)
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.verification import (
    GalerkinDiscretizer,
    Lagrange,
    SkfemDisk,
    SkfemPoints,
    StructuredBox,
    boundary_conditions,
    field,
    forcing,
    state_law,
    vector_field,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Periodic,
    Robin,
)
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.galerkin_verification import coded, converges, recovers, stalls

_Np = NDArray[Any]
_F = TypeVar("_F", bound=FieldExpression)


class _ADR:
    """The physics under test and its strong form, from the same strings.

    Parameters
    ----------
    diffusivity : str
        :math:`D(x)`.
    velocity : Sequence[str], optional
        :math:`v(x)`, one entry per coordinate.
    reaction : str, optional
        :math:`R(x, u)`, positive a source.
    conservative : bool
        The advection form.
    """

    def __init__(
        self,
        diffusivity: str,
        velocity: Optional[Sequence[str]] = None,
        reaction: Optional[str] = None,
        conservative: bool = False,
    ) -> None:
        self._diffusivity, self._velocity = diffusivity, velocity
        self._reaction, self._conservative = reaction, conservative

    def terms(self, coords: Sequence[str]) -> List[TermProtocol]:
        """The strong form as terms on row ``u``."""
        # Natural data are the diffusive flux in both forms (the
        # conservative physics keeps its advective boundary term), so
        # only -D grad(u) goes inside the divergence; the conservative
        # div(v u) is a source, expanded to v.grad(u) + div(v) u.
        fluxes = [f"-({self._diffusivity})*u_{c}" for c in coords]
        terms: List[TermProtocol] = []
        if self._velocity is not None:
            transport = [f"({v})*u_{c}" for v, c in zip(self._velocity, coords)]
            if self._conservative:
                symbols = dict(zip(coords, coordinates(coords)))
                divergence = sum(
                    sp.diff(sp.sympify(v, locals=symbols), symbols[c])
                    for v, c in zip(self._velocity, coords)
                )
                transport.append(f"({divergence})*u")
            terms.append(Source("u", " + ".join(transport)))
        terms.append(Divergence("u", fluxes))
        if self._reaction is not None:
            terms.append(Source("u", f"-({self._reaction})"))
        return terms

    def physics(
        self,
        problem: ManufacturedProblem,
        basis: LagrangeBasis[_Np],
        advection_form: Optional[AdvectionFormProtocol] = None,
    ) -> AdvectionDiffusionReaction[_Np]:
        """The physics, with ``advection_form`` in place of the one
        ``conservative`` selects when given (for wrong-model controls)."""
        form: AdvectionFormProtocol = (
            advection_form
            if advection_form is not None
            else ConservativeAdvection()
            if self._conservative
            else NonConservativeAdvection()
        )
        return AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=field(problem, self._diffusivity),
            bkd=NumpyBkd(),
            velocity=(
                None
                if self._velocity is None
                else vector_field(problem, self._velocity)
            ),
            reaction=(
                None
                if self._reaction is None
                else state_law(problem, self._reaction, "u")
            ),
            forcing=forcing(problem, "u"),
            advection_form=form,
        )

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        return compose_galerkin_system(
            self.physics(problem, basis), boundary_conditions(problem, basis, "u")
        )

    def __repr__(self) -> str:
        return (
            f"_ADR({self._diffusivity!r}, {self._velocity!r}, {self._reaction!r}, "
            f"conservative={self._conservative})"
        )


class _ConservativeDroppingBoundary:
    """Wrong model: the conservative weak integrand, but a flux of None,
    so the physics adds no advective boundary term and its natural data
    become the total flux."""

    def __init__(self) -> None:
        self._form = ConservativeAdvection()

    def strong_term(
        self, velocity: Sequence[_F], u: _F, derivative: Callable[[_F, int], _F]
    ) -> _F:
        return self._form.strong_term(velocity, u, derivative)

    def weak_integrand(
        self,
        velocity: Sequence[_F],
        u: _F,
        grad_u: Sequence[_F],
        test: _F,
        grad_test: Sequence[_F],
    ) -> _F:
        return self._form.weak_integrand(velocity, u, grad_u, test, grad_test)

    def flux(self, velocity: Sequence[_F], u: _F) -> Optional[List[_F]]:
        return None


class _DoNothing:
    """``adr`` with the Neumann conditions on ``sides`` left out of the
    system: those sides get no condition object at all.

    The problem still names a Neumann condition there, so its exact
    solution must have zero diffusive flux on them.
    """

    def __init__(
        self,
        adr: _ADR,
        sides: Sequence[str],
        advection_form: Optional[AdvectionFormProtocol] = None,
    ) -> None:
        self._adr = adr
        self._sides: FrozenSet[str] = frozenset(sides)
        self._form = advection_form

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        physics = self._adr.physics(problem, basis, self._form)
        conditions = boundary_conditions(problem, basis, "u")
        kept = [
            bc
            for bc in conditions
            if not (isinstance(bc, NeumannBC) and bc.boundary_name() in self._sides)
        ]
        if len(conditions) - len(kept) != len(self._sides):
            raise ValueError(f"no Neumann condition on every side of {self._sides}")
        return compose_galerkin_system(physics, kept)

    def __repr__(self) -> str:
        return (
            f"_DoNothing({self._adr!r}, {sorted(self._sides)}, "
            f"advection_form={self._form!r})"
        )


def _problem(
    solution: str,
    adr: _ADR,
    domain: DomainProtocol,
    conditions: Dict[str, ConditionProtocol],
) -> ManufacturedProblem:
    return ManufacturedProblem(
        {"u": solution}, adr.terms(domain.coordinates()), domain, conditions
    )


_MIXED_1D: Dict[str, ConditionProtocol] = {"left": Dirichlet(), "right": Neumann()}
_MIXED_2D: Dict[str, ConditionProtocol] = {
    "left": Dirichlet(),
    "right": Neumann(),
    "bottom": Robin(1.5),
    "top": Dirichlet(),
}
_MIXED_3D: Dict[str, ConditionProtocol] = {
    "left": Dirichlet(),
    "right": Neumann(),
    "bottom": Robin(2.0),
    "top": Dirichlet(),
    "back": Dirichlet(),
    "front": Neumann(),
}


@pytest.mark.parametrize("conservative", [False, True])
class TestRates:
    def test_1d(self, conservative: bool) -> None:
        """1D, Dirichlet left and Neumann right, variable D and v, cubic
        reaction: L2 rate 2 for P1 and 3 for P2."""
        adr = _ADR("1 + x/2", ["1 + x"], "-u**3", conservative)
        box = Box([(0.0, 1.0)])
        problem = _problem("sin(pi*x) + x", adr, box, _MIXED_1D)
        converges(problem, StructuredBox([8, 16, 32, 64]), 1, adr, 2.0)
        converges(problem, StructuredBox([4, 8, 16, 32]), 2, adr, 3.0)

    def test_2d(self, conservative: bool) -> None:
        """Unit square, quads, Dirichlet, Neumann and Robin sides,
        variable D and rotating v, nonlinear reaction: L2 rate 2 for P1
        and 3 for P2."""
        adr = _ADR("1 + x*y/4", ["1 - y", "x/2"], "-u - u**3/4", conservative)
        box = Box([(0.0, 1.0), (0.0, 1.0)])
        problem = _problem("exp(x/2)*sin(pi*y) + x*y", adr, box, _MIXED_2D)
        converges(problem, StructuredBox([4, 8, 16, 32]), 1, adr, 2.0)
        converges(problem, StructuredBox([2, 4, 8, 16]), 2, adr, 3.0)

    def test_3d(self, conservative: bool) -> None:
        """Unit cube, hexes, all three condition kinds: P1 L2 rate 2."""
        adr = _ADR("1 + z/2", ["1", "0", "x/2"], "-u", conservative)
        box = Box([(0.0, 1.0)] * 3)
        problem = _problem("sin(x)*cos(y)*exp(z)", adr, box, _MIXED_3D)
        converges(problem, StructuredBox([2, 4, 8]), 1, adr, 2.0)

    def test_disk(self, conservative: bool) -> None:
        """Unit disk, Robin on the curved boundary with the mesh's own
        facet normals: P1 L2 rate 2, so the O(h^2) polygon error does not
        spoil it."""
        adr = _ADR("1 + x**2/2", ["y", "-x"], "-u", conservative)
        problem = _problem("exp(x)*cos(y)", adr, Disk(), {"circle": Robin(2.0)})
        converges(problem, SkfemDisk([2, 3, 4, 5]), 1, adr, 2.0)


@pytest.mark.parametrize("conservative", [False, True])
class TestDoNothingOutflow:
    """An outflow side with no condition object is a free outflow, zero
    diffusive flux, in both advection forms. Solutions have zero normal
    derivative and a nonzero value there, so the advective flux leaving
    is nonzero and a zero-total-flux wall would be wrong."""

    def test_1d(self, conservative: bool) -> None:
        """Dirichlet left, nothing right, v = 1 + x leaving: P1 rate 2."""
        adr = _ADR("1 + x/2", ["1 + x"], "-u", conservative)
        problem = _problem(
            "2 + cos(pi*x) + x*(2 - x)", adr, Box([(0.0, 1.0)]), _MIXED_1D
        )
        converges(
            problem, StructuredBox([8, 16, 32, 64]), 1, _DoNothing(adr, ["right"]), 2.0
        )

    def test_2d(self, conservative: bool) -> None:
        """Unit square, nothing on the right where v leaves, Dirichlet and
        Robin elsewhere, variable D and diverging v: Q1 rate 2."""
        adr = _ADR("1 + x*y/4", ["1 + x/2", "y/2"], "-u", conservative)
        box = Box([(0.0, 1.0), (0.0, 1.0)])
        problem = _problem("2 + cos(pi*x)*sin(pi*y) + y", adr, box, _MIXED_2D)
        converges(
            problem, StructuredBox([4, 8, 16, 32]), 1, _DoNothing(adr, ["right"]), 2.0
        )


def test_conservative_without_boundary_term_stalls() -> None:
    """Wrong-model control: the conservative physics without its
    advective boundary term makes the do-nothing side a zero-total-flux
    wall, which the 1D outflow solution violates by (v u)(1) = 4."""
    adr = _ADR("1 + x/2", ["1 + x"], "-u", conservative=True)
    problem = _problem("2 + cos(pi*x) + x*(2 - x)", adr, Box([(0.0, 1.0)]), _MIXED_1D)
    stalls(
        problem,
        StructuredBox([8, 16, 32, 64]),
        1,
        _DoNothing(adr, ["right"], _ConservativeDroppingBoundary()),
    )


def test_variable_robin() -> None:
    """A Robin coefficient that varies along the side (1 + x on the top)
    reaches the stack as a function: P1 L2 rate 2."""
    adr = _ADR("1", ["1", "0"], "-u")
    box = Box([(0.0, 1.0), (0.0, 1.0)])
    conditions: Dict[str, ConditionProtocol] = {
        **_MIXED_2D,
        "top": Robin("1 + x"),
    }
    problem = _problem("cos(x)*exp(y)", adr, box, conditions)
    converges(problem, StructuredBox([4, 8, 16, 32]), 1, adr, 2.0)


@pytest.mark.parametrize(
    "ndim, simplex, solution",
    [
        (1, False, "1 + x - x**2/2"),
        (2, False, "1 + x*y - y**2/2"),
        (2, True, "1 + x*y - y**2/2"),
    ],
)
def test_p2_recovers_quadratics(ndim: int, simplex: bool, solution: str) -> None:
    """A quadratic lies in the P2 space, so with constant coefficients
    and natural conditions P2 reproduces it to round-off on every mesh:
    any wrong sign or missing boundary term shows as an O(1) error."""
    adr = _ADR("2", ["1"] * ndim, "-u")
    box = Box([(0.0, 1.0)] * ndim)
    conditions = _MIXED_1D if ndim == 1 else _MIXED_2D
    recovers(_problem(solution, adr, box, conditions), adr, 3, simplex)


@pytest.mark.parametrize("upper", [1.0, 1.1])
@pytest.mark.parametrize("codes", ["DN", "RD", "RR"])
@pytest.mark.parametrize("velocity", [None, "(1 + x)/10"])
@pytest.mark.parametrize("reaction", [None, "2*u"])
def test_p2_recovers_linear_1d(
    upper: float, codes: str, velocity: Optional[str], reaction: Optional[str]
) -> None:
    """u = x on [0, 1] or [0, 1.1], D = 4, every pairing of Dirichlet,
    Neumann and Robin ends, with and without advection and a linear
    source: P2 on 10 cells reproduces it to 1e-10 (L2)."""
    adr = _ADR("4", None if velocity is None else [velocity], reaction)
    box = Box([(0.0, upper)])
    recovers(_problem("x", adr, box, coded(codes, box)), adr, 10)


@pytest.mark.parametrize(
    "codes, reaction",
    [("DN", None), ("RD", "2*u"), ("RR", None), ("RR", "2*u")],
)
def test_p2_recovers_linear_1d_conservative(
    codes: str, reaction: Optional[str]
) -> None:
    """Conservative advection v = (1 + x)/10, where the natural data
    carry the advective flux: P2 reproduces u = x to 1e-10."""
    adr = _ADR("4", ["(1 + x)/10"], reaction, conservative=True)
    box = Box([(0.0, 1.0)])
    recovers(_problem("x", adr, box, coded(codes, box)), adr, 10)


@pytest.mark.parametrize(
    "solution, codes, diffusivity",
    [
        ("x + 2*y", "DDDD", "4"),
        ("x + 2*y", "DRDN", "4"),
        ("x + 2*y", "NRDR", "4"),
        ("x*y", "DDDD", "4"),
        ("x*y", "DRDD", "4"),
        ("x**2 + y**2", "DDDD", "4"),
        ("x**2 + y**2", "DRDN", "4"),
        ("x**2*y**2", "DDDD", "1"),
        ("x**2*y**2", "DRDD", "1"),
    ],
)
def test_q2_recovers_2d(solution: str, codes: str, diffusivity: str) -> None:
    """Pure diffusion on the unit square, sides in the order left, right,
    bottom, top: Q2 on 5x5 quads reproduces solutions in its space
    (x^2 y^2 included) to 1e-10."""
    adr = _ADR(diffusivity)
    box = Box([(0.0, 1.0)] * 2)
    recovers(_problem(solution, adr, box, coded(codes, box)), adr, 5)


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize(
    "solution, codes",
    [("x + 2*y", "DRDN"), ("x**2 + y**2", "DRDN"), ("x*y", "NRDR")],
)
def test_q2_recovers_2d_advected_reacting(
    solution: str, codes: str, conservative: bool
) -> None:
    """The 2D exact cases with advection v = (0.1, 0.1) and a linear
    source 2u switched on (the old suite's advection branch never ran):
    Q2 still reproduces them to 1e-10, in both advection forms."""
    adr = _ADR("4", ["0.1", "0.1"], "2*u", conservative)
    box = Box([(0.0, 1.0)] * 2)
    recovers(_problem(solution, adr, box, coded(codes, box)), adr, 5)


def test_q2_recovers_with_varying_robin_alpha() -> None:
    """Robin with alpha = 1 + x + y, which varies along the left and
    right sides (the layout a Danckwerts inflow needs), Dirichlet top
    and bottom: Q2 reproduces x y + x + 2 y to 1e-10."""
    adr = _ADR("4")
    box = Box([(0.0, 1.0)] * 2)
    conditions: Dict[str, ConditionProtocol] = {
        "left": Robin("1 + x + y"),
        "right": Robin("1 + x + y"),
        "bottom": Dirichlet(),
        "top": Dirichlet(),
    }
    recovers(_problem("x*y + x + 2*y", adr, box, conditions), adr, 5)


class _NoSpring(_ADR):
    """Robin conditions built with the spring term alpha u dropped (alpha
    = 0) but the problem's data kept."""

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        bkd = NumpyBkd()
        conditions: List[BoundaryConditionRole[_Np]] = []
        for part in ("left", "right"):
            data = problem.boundary_data(part)["u"]
            conditions.append(RobinBC(basis, part, 0.0, _Negated(data, problem), bkd))
        return compose_galerkin_system(self.physics(problem, basis), conditions)


class _Negated:
    """This stack's natural data from the core's, as the adapter does."""

    def __init__(self, data: sp.Expr, problem: ManufacturedProblem) -> None:
        self._points = SkfemPoints([data], problem)

    def evaluate(self, coords: _Np, normal: _Np, time: float) -> _Np:
        values: _Np = -self._points(coords, normals=normal)[0]
        return values

    def is_time_dependent(self) -> bool:
        return False


def test_robin_without_spring_stalls() -> None:
    """Wrong-model control: Robin ends with alpha u dropped stall, so the
    Robin cases above can see the spring term."""
    adr = _NoSpring("1", ["1"], "-u")
    box = Box([(0.0, 1.0)])
    problem = _problem("sin(pi*x) + x + 1", adr, box, coded("RR", box))
    stalls(problem, StructuredBox([8, 16, 32, 64]), 1, adr)


class _ReversedNormal(_ADR):
    """Natural data with the outward normal reversed: the core's flux
    data :math:`g` passed as this stack's :math:`-q\\,n` without the
    sign change."""

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        bkd = NumpyBkd()
        conditions: List[BoundaryConditionRole[_Np]] = []
        for part in ("left", "right"):
            data = problem.boundary_data(part)["u"]
            if isinstance(problem.conditions(part)["u"], Neumann):
                conditions.append(
                    NeumannBC(basis, part, _Unnegated(data, problem), bkd)
                )
            else:
                conditions.append(DirichletBC(basis, part, field(problem, data), bkd))
        return compose_galerkin_system(self.physics(problem, basis), conditions)


class _Unnegated:
    def __init__(self, data: sp.Expr, problem: ManufacturedProblem) -> None:
        self._points = SkfemPoints([data], problem)

    def evaluate(self, coords: _Np, normal: _Np, time: float) -> _Np:
        values: _Np = self._points(coords, normals=normal)[0]
        return values

    def is_time_dependent(self) -> bool:
        return False


def test_reversed_normal_stalls() -> None:
    """Wrong-model control: Neumann data with the normal reversed make
    the error stop falling, so the rate tests above can detect a sign
    error in natural data."""
    adr = _ReversedNormal("1", ["1"], "-u")
    problem = _problem("sin(pi*x) + x", adr, Box([(0.0, 1.0)]), _MIXED_1D)
    stalls(problem, StructuredBox([8, 16, 32, 64]), 1, adr)


def test_time_dependence_is_declared_per_supplier() -> None:
    """A steady forcing and time-varying boundary data can be mixed: each
    supplier declares time dependence exactly when its own expression
    uses t, so the stack caches the steady ones and re-evaluates the
    others; values at t = 0.5 match the expression."""
    problem = _problem("x", _ADR("1"), Box([(0.0, 1.0)]), _MIXED_1D)
    steady, varying = field(problem, "1 + x"), field(problem, "x*exp(-t)")
    assert not isinstance(steady, float) and not isinstance(varying, float)
    assert not steady.is_time_dependent() and varying.is_time_dependent()
    pts = np.array([[0.25, 0.75]])
    np.testing.assert_allclose(varying(pts, 0.5), pts[0] * np.exp(-0.5), rtol=1e-14)
    assert isinstance(field(problem, "2*pi"), float)


_PERIODIC: Dict[str, ConditionProtocol] = {"left": Periodic(), "right": Periodic()}


def _periodic_problem(solution: str, adr: _ADR) -> ManufacturedProblem:
    return _problem(solution, adr, Box([(0.0, 1.0)]), _PERIODIC)


@pytest.mark.parametrize("conservative", [False, True])
def test_periodic_rates(conservative: bool) -> None:
    """Both ends periodic, the mesh wrapping x (endpoints identified),
    periodic D and v, absorption -u so the problem has a unique
    solution: L2 rate 2 for P1 (cells 8-64) and 3 for P2 (4-32)."""
    adr = _ADR("1 + sin(2*pi*x)/2", ["1 + cos(2*pi*x)/4"], "-u - u**3/4", conservative)
    problem = _periodic_problem("sin(2*pi*x) + cos(4*pi*x)/2 + 1", adr)
    wrapped = StructuredBox([8, 16, 32, 64], periodic=("x",))
    converges(problem, wrapped, 1, adr, 2.0)
    converges(problem, StructuredBox([4, 8, 16, 32], periodic=("x",)), 2, adr, 3.0)


def test_non_periodic_solution_stalls() -> None:
    """Wrong-model control: the ends declared periodic and wrapped, but
    u = x + sin(2 pi x) jumps by 1 across the identified point, so the
    error does not fall."""
    adr = _ADR("1", None, "-u")
    problem = _periodic_problem("x + sin(2*pi*x)", adr)
    stalls(problem, StructuredBox([8, 16, 32, 64], periodic=("x",)), 1, adr)


class TestRefusals:
    def test_periodic_mismatch(self) -> None:
        """Periodicity lives in the mesh here, so the problem's periodic
        sides must be exactly the ones the mesh wraps: periodic sides on
        an unwrapped mesh, or ordinary conditions on a wrapped one, are
        refused rather than silently dropped."""
        adr = _ADR("1", None, "-u")
        periodic = _periodic_problem("sin(2*pi*x)", adr)
        plain = GalerkinDiscretizer(StructuredBox([4]), Lagrange(1), adr)
        with pytest.raises(ValueError, match="'left' is periodic"):
            plain.approximations(periodic)
        dirichlet = _problem("sin(2*pi*x)", adr, Box([(0.0, 1.0)]), _MIXED_1D)
        wrapped = GalerkinDiscretizer(
            StructuredBox([4], periodic=("x",)), Lagrange(1), adr
        )
        with pytest.raises(ValueError, match="'left' is not periodic"):
            wrapped.approximations(dirichlet)

    def test_periodic_mesh_limits(self) -> None:
        """Only 1D periodic meshes exist in this stack, and only a
        coordinate of the box can wrap."""
        square = Box([(0.0, 1.0)] * 2)
        with pytest.raises(ValueError, match="1D only"):
            StructuredBox([4], periodic=("x",)).meshes(square)
        with pytest.raises(ValueError, match="not the box's"):
            StructuredBox([4], periodic=("y",)).meshes(Box([(0.0, 1.0)]))

    def test_transient(self) -> None:
        """This discretizer solves only the steady equation F(u) = 0; on a
        transient problem it would drop du/dt and report a meaningless
        error, so it refuses. Transient problems will get their own
        discretizer, reusing the same helpers."""
        adr = _ADR("1")
        discretizer = GalerkinDiscretizer(StructuredBox([4]), Lagrange(1), adr)
        transient = ManufacturedProblem(
            {"u": "exp(-t)*x"},
            [TimeDerivative("u"), *adr.terms(("x",))],
            Box([(0.0, 1.0)]),
            _MIXED_1D,
        )
        with pytest.raises(ValueError, match="steady"):
            discretizer.approximations(transient)

    def test_multi_field(self) -> None:
        """A scalar basis carries one field, so a two-field problem on it
        is refused: its second field would have no DOFs."""
        adr = _ADR("1")
        box = Box([(0.0, 1.0)])
        discretizer = GalerkinDiscretizer(StructuredBox([4]), Lagrange(1), adr)
        two = ManufacturedProblem(
            {"u": "x", "w": "x"},
            [Divergence("u", ["-u_x"]), Divergence("w", ["-w_x"])],
            box,
            {"left": Dirichlet(), "right": Dirichlet()},
        )
        with pytest.raises(ValueError, match="one field per component"):
            discretizer.approximations(two)

    def test_named_box(self) -> None:
        """Structured meshes name their sides left, right, ...; a box with
        named coordinates (sides xmin, vmax, ...) would not match them, so
        it is refused."""
        with pytest.raises(ValueError, match="names the sides"):
            StructuredBox([4]).meshes(Box([(0.0, 1.0), (-1.0, 1.0)], ("x", "v")))

    def test_helpers(self) -> None:
        """Coefficients may use only coordinates, t and constants (not
        the state); a vector needs one entry per coordinate; a state law
        needs a declared field and may not use t, which this stack's
        reaction laws do not take."""
        box = Box([(0.0, 1.0)])
        problem = _problem("x", _ADR("1"), box, _MIXED_1D)
        with pytest.raises(ValueError, match="may depend only on"):
            field(problem, "u_x")
        with pytest.raises(ValueError, match="one expression per coordinate"):
            vector_field(problem, ["1", "2"])
        with pytest.raises(ValueError, match="not a field"):
            state_law(problem, "q**2", "q")
        with pytest.raises(ValueError, match="may depend only on"):
            state_law(problem, "u*t", "u")
