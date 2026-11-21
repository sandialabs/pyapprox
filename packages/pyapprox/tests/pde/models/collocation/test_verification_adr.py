r"""Collocation advection-diffusion-reaction verified by manufactured
solutions.

The problem is stated as terms, its strong form
:math:`\nabla\cdot(-D\nabla u + [v u]) + [v\cdot\nabla u] - r u = f`
(the bracket matching the advection form), and the physics under test,
:class:`~pyapprox.pde.collocation.physics.advection_diffusion.
AdvectionDiffusionReaction`, is built from the same coefficient strings
by a system factory. Its reaction is linear, :math:`r(x) u`. Errors are
the L2 norm of the collocation interpolant's error, by Gauss-Legendre
quadrature.

Cases:

- recovery (L2 error below 1e-10) of polynomial solutions of degree
  below the node count, where collocation is exact: 1D cubics on [0, 1]
  (8 nodes) under every pairing of Dirichlet, Neumann and Robin ends,
  2D cubics on [0, 1] x [-1, 1] (6 x 6 nodes) and 3D cubics on
  [0, 1] x [-1, 1] x [0, 2] (6 nodes per direction) with all three kinds
  on their sides; variable D, v and r, both advection forms, so the
  Neumann and Robin data carry the advective flux exactly when the form
  is conservative; and Neumann and Robin sides that measure the normal
  gradient :math:`\nabla u\cdot n` rather than the physics' flux, in 1D
  and 2D;
- spectral decay (fitted decay of log error per node) of analytic
  solutions: 1D and 2D under mixed conditions, and periodic ends in 1D,
  a 2D box periodic in x, and one periodic in both directions, in both
  advection forms;
- wrong-model controls, fitted algebraic rate below 0.5: Neumann data
  with the normal reversed; Robin ends without their spring term;
- the adapter's own pieces: the decay check on synthetic studies, and
  the quadrature L2 error of :math:`u^* = x` in 1D, 2D and 3D boxes
  against the closed form, and an injected initial guess asked once per
  level.

Refusals: transient and multi-field problems, a domain that is not a
box, one periodic side of a pair, a flux condition without a flux, a
row that takes no conditions, and an initial guess that is not an
``InitialGuessProtocol`` or not a column of the states.
"""

import math
from typing import Dict, Generic, List, Optional, Sequence

import pytest
import sympy as sp

from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.closure_rows import RobinPart, RobinRows
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    FluxNormalOperator,
    GradientNormalOperator,
)
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms.polar import PolarTransform
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.constitutive.advection import (
    AdvectionFormProtocol,
    ConservativeAdvection,
    NonConservativeAdvection,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevAnnulus,
    ChebyshevBox,
    ChebyshevEllipticalAnnulus,
    ChebyshevLevel,
    CollocationApproximation,
    CollocationDiscretizer,
    ConstantGuess,
    DecaysExponentially,
    NodeStudy,
    PeriodicSeam,
    boundary_conditions,
    collocation_system,
    field,
    forcing,
    nodal_vector,
    side_nodes,
    side_values,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Periodic,
    Robin,
)
from pyapprox.pde.verification.domains import (
    Annulus,
    Box,
    Disk,
    DomainProtocol,
    EllipticalAnnulus,
)
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.collocation_verification import (
    coded,
    decays,
    recovers,
    ring_newton,
    stalls,
)


class _ADR:
    """The physics under test and its strong form, from the same strings.

    Parameters
    ----------
    diffusivity : str
        :math:`D(x)`.
    velocity : Sequence[str], optional
        :math:`v(x)`, one entry per coordinate, fixed in time.
    reaction : str, optional
        :math:`r(x)`, the coefficient of the linear term :math:`r u`.
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
        fluxes = [f"-({self._diffusivity})*u_{c}" for c in coords]
        terms: List[TermProtocol] = []
        if self._velocity is not None:
            if self._conservative:
                fluxes = [f"{q} + ({v})*u" for q, v in zip(fluxes, self._velocity)]
            else:
                transport = [f"({v})*u_{c}" for v, c in zip(self._velocity, coords)]
                terms.append(Source("u", " + ".join(transport)))
        terms.append(Divergence("u", fluxes))
        if self._reaction is not None:
            terms.append(Source("u", f"-({self._reaction})*u"))
        return terms

    def physics(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> AdvectionDiffusionReaction[Array]:
        form: AdvectionFormProtocol = (
            ConservativeAdvection()
            if self._conservative
            else NonConservativeAdvection()
        )
        return AdvectionDiffusionReaction(
            level.basis(),
            level.bkd(),
            diffusion=field(problem, self._diffusivity, level),
            velocity=(
                None
                if self._velocity is None
                else nodal_vector(problem, self._velocity, level)
            ),
            reaction=(
                None
                if self._reaction is None
                else field(problem, self._reaction, level)
            ),
            forcing=forcing(problem, "u", level),
            advection_form=form,
        )

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = self.physics(problem, level)
        return collocation_system(physics, problem, level, {"u": physics})

    def __repr__(self) -> str:
        return (
            f"_ADR({self._diffusivity!r}, {self._velocity!r}, {self._reaction!r}, "
            f"conservative={self._conservative})"
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


_UNIT = Box([(0.0, 1.0)])
_RECTANGLE = Box([(0.0, 1.0), (-1.0, 1.0)])


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize("codes", ["DD", "DN", "ND", "RD", "DR", "RR", "NR"])
def test_recovers_cubic_1d(
    numpy_bkd: Backend[Array], codes: str, conservative: bool
) -> None:
    """u = 1 + x - x^2/2 + x^3 on [0, 1], D = 1 + x/2, v = 1 + x,
    r = -1: 8 nodes reproduce it to 1e-10 under every pairing of ends."""
    adr = _ADR("1 + x/2", ["1 + x"], "-1", conservative)
    problem = _problem("1 + x - x**2/2 + x**3", adr, _UNIT, coded(codes, _UNIT))
    recovers(problem, adr, 8, numpy_bkd)


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize("codes", ["DDDD", "DRDN", "NRDR", "RRRR"])
def test_recovers_cubic_2d(bkd: Backend[Array], codes: str, conservative: bool) -> None:
    """u = x y + y^2 - x^2 y on [0, 1] x [-1, 1] (sides left, right,
    bottom, top), D = 2 + x y, v = (1 + x, y/2) (divergence 3/2, so the
    two forms differ), r = -1: 6 x 6 nodes reproduce it to 1e-10, on
    both backends."""
    adr = _ADR("2 + x*y", ["1 + x", "y/2"], "-1", conservative)
    problem = _problem("x*y + y**2 - x**2*y", adr, _RECTANGLE, coded(codes, _RECTANGLE))
    recovers(problem, adr, 6, bkd)


_BRICK = Box([(0.0, 1.0), (-1.0, 1.0), (0.0, 2.0)])


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize("codes", ["DDDDDD", "DNRDRN"])
def test_recovers_cubic_3d(bkd: Backend[Array], codes: str, conservative: bool) -> None:
    """u = x y z + y^2 - x^2 z on [0, 1] x [-1, 1] x [0, 2] (sides left,
    right, bottom, top, back, front), D = 1 + x z/2, v = (1 + x, z/2,
    y/2 + z/2) (divergence 3/2), r = -1: 6 nodes per direction reproduce
    it to 1e-10, on both backends."""
    adr = _ADR("1 + x*z/2", ["1 + x", "z/2", "y/2 + z/2"], "-1", conservative)
    problem = _problem("x*y*z + y**2 - x**2*z", adr, _BRICK, coded(codes, _BRICK))
    recovers(problem, adr, 6, bkd)


_TESSERACT = Box([(0.0, 1.0), (-1.0, 1.0), (0.0, 2.0), (-1.0, 0.0)], "xyzw")


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize("codes", ["DDDDDDDD", "DNRDRNDR"])
def test_recovers_cubic_4d(bkd: Backend[Array], codes: str, conservative: bool) -> None:
    """u = x y w + z^2 - x^2 w on [0, 1] x [-1, 1] x [0, 2] x [-1, 0]
    (sides xmin, xmax, ..., wmin, wmax), D = 1 + x w/2, v = (1 + x, w/2,
    y/2, z/2 + w) (divergence 2), r = -1: 4 nodes per direction (256)
    reproduce it to 1e-10, on both backends; the box family and ADR hold
    no dimension cap."""
    adr = _ADR("1 + x*w/2", ["1 + x", "w/2", "y/2", "z/2 + w"], "-1", conservative)
    problem = _problem(
        "x*y*w + z**2 - x**2*w", adr, _TESSERACT, coded(codes, _TESSERACT)
    )
    recovers(problem, adr, 4, bkd)


class _GradientTerm(Generic[Array]):
    r"""Neumann and Robin sides that measure :math:`N = \nabla u\cdot n`,
    through :class:`~pyapprox.pde.collocation.boundary.normal_operators.
    GradientNormalOperator`. Against the problem's flux
    :math:`q = -\nabla u` of a unit diffusivity, :math:`N = -q\cdot n`
    (:math:`s = -1`).

    Parameters
    ----------
    level : ChebyshevLevel
        The level whose derivative matrices measure the gradient.
    """

    def __init__(self, level: ChebyshevLevel[Array]) -> None:
        basis = level.basis()
        self._derivatives = [
            basis.derivative_matrix(1, dim) for dim in range(basis.ndim())
        ]

    def operator(
        self, bkd: Backend[Array], indices: Array, normals: Array
    ) -> GradientNormalOperator[Array]:
        return GradientNormalOperator(bkd, indices, normals, self._derivatives)

    def sign(self) -> float:
        return -1.0


class _GradientADR(_ADR):
    r"""Unit diffusivity and non-conservative advection, so the problem's
    flux is :math:`-\nabla u`; Neumann and Robin sides measure the normal
    gradient instead of the physics' flux."""

    def __init__(self, velocity: Sequence[str], reaction: str) -> None:
        super().__init__("1", velocity, reaction, conservative=False)

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = self.physics(problem, level)
        return collocation_system(physics, problem, level, {"u": _GradientTerm(level)})


@pytest.mark.parametrize("codes", ["ND", "DN", "RD", "NR"])
def test_recovers_cubic_gradient_1d(numpy_bkd: Backend[Array], codes: str) -> None:
    """Gradient-measured ends: u = 1 + x - x^2/2 + x^3 on [0, 1], D = 1,
    v = 1 + x, r = -1; 8 nodes reproduce it to 1e-10."""
    adr = _GradientADR(["1 + x"], "-1")
    problem = _problem("1 + x - x**2/2 + x**3", adr, _UNIT, coded(codes, _UNIT))
    recovers(problem, adr, 8, numpy_bkd)


@pytest.mark.parametrize("codes", ["NDDD", "DRDN", "RRRR"])
def test_recovers_cubic_gradient_2d(bkd: Backend[Array], codes: str) -> None:
    """Gradient-measured sides: u = x y + y^2 - x^2 y on [0, 1] x [-1, 1],
    D = 1, v = (1 - y, x/2), r = -1; 6 x 6 nodes reproduce it to 1e-10, on
    both backends."""
    adr = _GradientADR(["1 - y", "x/2"], "-1")
    problem = _problem("x*y + y**2 - x**2*y", adr, _RECTANGLE, coded(codes, _RECTANGLE))
    recovers(problem, adr, 6, bkd)


@pytest.mark.parametrize("conservative", [False, True])
def test_decays_1d(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """u = exp(x) sin(3 x) on [0, 1], Dirichlet left and Neumann right,
    variable D and v: the L2 error falls exponentially over 4-12 nodes
    (0.2 to 4e-9 non-conservative, 0.7 to 1e-8 conservative; observed
    decay 2.10 and 2.13 per node, required 1)."""
    adr = _ADR("1 + x/2", ["1 + x"], "-1", conservative)
    problem = _problem("exp(x)*sin(3*x)", adr, _UNIT, coded("DN", _UNIT))
    decays(problem, adr, [4, 6, 8, 10, 12], 1.0, numpy_bkd)


@pytest.mark.parametrize("conservative", [False, True])
def test_decays_2d(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """u = exp(x/2) sin(2 y) + x y on [0, 1] x [-1, 1], Dirichlet,
    Robin and Neumann sides, v = (1 + x/2, y/2) with divergence 1:
    exponential decay over 4-12 nodes per direction (0.15 to 4e-9 L2
    non-conservative, 0.3 to 1e-8 conservative; observed 2.18 and 2.11
    per node, required 1)."""
    adr = _ADR("1 + x*y/4", ["1 + x/2", "y/2"], "-1", conservative)
    problem = _problem(
        "exp(x/2)*sin(2*y) + x*y", adr, _RECTANGLE, coded("DRDN", _RECTANGLE)
    )
    decays(problem, adr, [4, 6, 8, 10, 12], 1.0, numpy_bkd)


_PERIODIC_1D: Dict[str, ConditionProtocol] = {"left": Periodic(), "right": Periodic()}


@pytest.mark.parametrize("conservative", [False, True])
def test_decays_periodic_1d(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """Both ends periodic, periodic D and v, absorption r = -1: the
    value and derivative match close the system, exponential decay over
    8-20 nodes (2.1 to 3e-7 L2 non-conservative, 2.3 to 7e-7
    conservative; observed 1.31 and 1.25 per node, required 0.6)."""
    adr = _ADR("1 + sin(2*pi*x)/2", ["1 + cos(2*pi*x)/4"], "-1", conservative)
    problem = _problem("sin(2*pi*x) + cos(4*pi*x)/2 + 1", adr, _UNIT, _PERIODIC_1D)
    decays(problem, adr, [8, 12, 16, 20], 0.6, numpy_bkd)


@pytest.mark.parametrize("conservative", [False, True])
def test_decays_periodic_2d(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """[0, 1] x [-1, 1] periodic in x, Dirichlet bottom and Robin top:
    v = (1 + sin(2 pi x)/4, x/4 + 1/2), not divergence-free: exponential
    decay over 8-16 nodes per direction (1e-3 to 4e-11 L2
    non-conservative, 3e-3 to 5e-8 conservative; observed 2.16 and 1.39
    per node, required 0.7: the conservative form differentiates the
    interpolant of v u, whose frequencies reach twice those of u); the
    bottom Dirichlet values agree on the paired corners."""
    adr = _ADR("1", ["1 + sin(2*pi*x)/4", "x/4 + 1/2"], "-1", conservative)
    conditions: Dict[str, ConditionProtocol] = {
        "left": Periodic(),
        "right": Periodic(),
        "bottom": Dirichlet(),
        "top": Robin(1.0),
    }
    problem = _problem("sin(2*pi*x)*exp(y/2) + y", adr, _RECTANGLE, conditions)
    decays(problem, adr, [8, 10, 12, 14, 16], 0.7, numpy_bkd)


@pytest.mark.parametrize("conservative", [False, True])
def test_decays_periodic_both_2d(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """[0, 1] x [-1, 1] periodic in x and y, periodic D and v, absorption
    r = -1, v not divergence-free: both pairs share the corners, and the
    error falls exponentially over 8-16 nodes per direction (7e-3 to 8e-10
    L2 non-conservative, 5e-2 to 6e-7 conservative; observed 1.99 and
    1.42 per node, required 0.7)."""
    adr = _ADR(
        "1 + sin(2*pi*x)*cos(pi*y)/4",
        ["1 + sin(2*pi*x)/4", "cos(pi*y)/2"],
        "-1",
        conservative,
    )
    conditions: Dict[str, ConditionProtocol] = {
        side: Periodic() for side in _RECTANGLE.boundaries()
    }
    problem = _problem(
        "sin(2*pi*x)*cos(pi*y) + sin(pi*y) + 1", adr, _RECTANGLE, conditions
    )
    decays(problem, adr, [8, 10, 12, 14, 16], 0.7, numpy_bkd)


_RING = Annulus((0.0, 0.0), 1.0, 2.0)
_ELLIPTICAL_RING = EllipticalAnnulus(1.0, 0.5, 1.0)
_RING_SIZES = [8, 12, 16, 20, 24]


@pytest.mark.parametrize(
    "adr, codes",
    [
        (_ADR("1 + x**2/4"), "DD"),
        (_ADR("1 + x**2/4"), "DR"),
        (_ADR("1", ["y", "x"]), "DN"),
    ],
    ids=["variable-D-DD", "variable-D-DR", "advection-DN"],
)
def test_decays_annulus(numpy_bkd: Backend[Array], adr: _ADR, codes: str) -> None:
    """u* = exp(x/2) sin(y) on the annulus 1 <= r <= 2, meshed over the
    full angle with its seam closed, inner circle first: errors 2.6e-1 to
    9.5e-6 (DD), 3.5e-1 to 1.1e-5 (DR), 1.0 to 1.4e-5 (advection by
    (y, x), DN) over 8-24 nodes per direction (observed 0.64, 0.65, 0.70
    per node, required 0.45). Slower than on a box: Chebyshev nodes in the
    angle cluster at the seam and are coarsest opposite it."""
    problem = _problem("exp(x/2)*sin(y)", adr, _RING, coded(codes, _RING))
    decays(
        problem,
        adr,
        _RING_SIZES,
        0.45,
        numpy_bkd,
        nonlinear=ring_newton(),
        family=ChebyshevAnnulus,
    )


def test_decays_annulus_old_periodic_case(bkd: Backend[Array]) -> None:
    """The old periodic-theta case: u* = x on 1 <= r <= 2, D = 1,
    absorption 1, both circles Dirichlet. Errors 9.7e-3, 5.5e-6, 8.5e-10
    over 8, 12, 16 nodes per direction (observed 2.03 per node, required
    1.5; round-off near 2e-11 from 20 nodes); the old test needed 1e-4 at
    20 x 30 nodes. Both backends."""
    adr = _ADR("1", reaction="-1")
    problem = _problem("x", adr, _RING, coded("DD", _RING))
    decays(
        problem,
        adr,
        [8, 12, 16],
        1.5,
        bkd,
        nonlinear=ring_newton(),
        family=ChebyshevAnnulus,
    )


@pytest.mark.parametrize("codes", ["DD", "RD"])
def test_decays_elliptical_annulus(numpy_bkd: Backend[Array], codes: str) -> None:
    """u* = sin(x) cos(y) between the confocal ellipses u = 0.5 and
    u = 1 (foci (+-1, 0)), D = 1: errors 8.0e-2 to 7.2e-6 (DD) and
    8.0e-2 to 7.6e-6 (Robin inner, Dirichlet outer) over 8-24 nodes per
    direction (observed 0.545 and 0.544 per node, required 0.4)."""
    adr = _ADR("1")
    problem = _problem(
        "sin(x)*cos(y)", adr, _ELLIPTICAL_RING, coded(codes, _ELLIPTICAL_RING)
    )
    decays(
        problem,
        adr,
        _RING_SIZES,
        0.4,
        numpy_bkd,
        nonlinear=ring_newton(),
        family=ChebyshevEllipticalAnnulus,
    )


class _Seamless(Generic[Array]):
    """Annulus levels without their seam closure (a wrong-model control):
    the angle's two ends stay two independent copies of one line."""

    def __init__(self, sizes: Sequence[int], bkd: Backend[Array]) -> None:
        self._rings = ChebyshevAnnulus(sizes, bkd)

    def levels(self, domain: DomainProtocol) -> List[ChebyshevLevel[Array]]:
        return [
            ChebyshevLevel(
                domain,
                level.mesh(),
                level.basis(),
                {part: level.boundary_ids(part) for part in domain.boundaries()},
            )
            for level in self._rings.levels(domain)
        ]


def test_annulus_without_seam_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: the first annulus case without the seam rows
    diverges, errors 3.5e-1, 5.6e-1, 2.8, 1.0e3 over 8-20 nodes, where
    the seam gives 2.6e-1 to 1.2e-4."""
    adr = _ADR("1 + x**2/4")
    problem = _problem("exp(x/2)*sin(y)", adr, _RING, coded("DD", _RING))
    stalls(
        problem,
        adr,
        [8, 12, 16, 20],
        numpy_bkd,
        nonlinear=ring_newton(),
        family=_Seamless,
    )


class _NoSpring(_ADR):
    """Robin ends built with the spring term alpha u dropped (alpha = 0)
    but the problem's data kept."""

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = self.physics(problem, level)
        bkd = level.bkd()
        parts = []
        for part in ("left", "right"):
            indices, normals = side_nodes(level, part)
            parts.append(
                RobinPart(
                    indices,
                    FluxNormalOperator(bkd, indices, normals, physics),
                    0.0,
                    1.0,
                    side_values(problem, level, part, problem.boundary_data(part)["u"]),
                )
            )
        rows = RobinRows(bkd, level.npts(), 1, parts)
        return CollocationSystem(physics, None, [rows])


def test_robin_without_spring_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: Robin ends with alpha u dropped stall, so the
    Robin cases above can see the spring term."""
    adr = _NoSpring("1", ["1"], "-1")
    problem = _problem("sin(pi*x) + x + 1", adr, _UNIT, coded("RR", _UNIT))
    stalls(problem, adr, [6, 8, 10, 12], numpy_bkd)


class _ReversedNormal(_ADR):
    """Neumann data with the outward normal reversed: the core's flux
    data :math:`g` passed as :math:`-g`."""

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = self.physics(problem, level)
        bkd = level.bkd()
        left, _ = side_nodes(level, "left")
        right, normals = side_nodes(level, "right")
        data = problem.boundary_data("right")["u"]
        left_values = side_values(
            problem, level, "left", problem.boundary_data("left")["u"]
        )
        reversed_rows = RobinRows(
            bkd,
            level.npts(),
            1,
            [
                RobinPart(
                    right,
                    FluxNormalOperator(bkd, right, normals, physics),
                    0.0,
                    1.0,
                    side_values(problem, level, "right", -data),
                )
            ],
        )
        essential = DirichletRows(
            bkd, level.npts(), 1, [DirichletPart(left, left_values)]
        )
        return CollocationSystem(physics, essential, [reversed_rows])


def test_reversed_normal_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: Neumann data with the normal reversed make
    the error stop falling, so the cases above can detect a sign error
    in flux data."""
    adr = _ReversedNormal("1", ["1"], "-1")
    problem = _problem("sin(pi*x) + x", adr, _UNIT, coded("DN", _UNIT))
    stalls(problem, adr, [6, 8, 10, 12], numpy_bkd)


def test_time_dependence_is_declared_per_coefficient(
    numpy_bkd: Backend[Array],
) -> None:
    """A steady coefficient and a time-varying one can be mixed: each
    declares time dependence exactly when its own expression uses t;
    values at t = 0.5 match the expression, and a constant is a float."""
    bkd = numpy_bkd
    problem = _problem("x", _ADR("1"), _UNIT, coded("DN", _UNIT))
    (level,) = ChebyshevBox([5], bkd).levels(_UNIT)
    steady, varying = field(problem, "1 + x", level), field(problem, "x*exp(-t)", level)
    assert not isinstance(steady, float) and not isinstance(varying, float)
    assert not steady.is_time_dependent() and varying.is_time_dependent()
    x = level.mesh().points()[0]
    bkd.assert_allclose(varying.values(0.5), x * bkd.exp(bkd.asarray(-0.5)), rtol=1e-14)
    assert isinstance(field(problem, "2*pi", level), float)


class TestAdapterPieces:
    def test_two_fields_take_their_components(self, numpy_bkd: Backend[Array]) -> None:
        """Field w, second in the problem, has its rows on the second
        component (offset npts); its Neumann side gets w's flux; the
        approximation measures both fields: u* = x, w* = 2x against a
        zero state on [0, 1] give L2 error sqrt(1/3 + 4/3) and nodal
        error 2."""
        bkd = numpy_bkd
        two = ManufacturedProblem(
            {"u": "x", "w": "2*x"},
            [Divergence("u", ["-u_x"]), Divergence("w", ["-w_x"])],
            _UNIT,
            {
                "left": {"u": Dirichlet(), "w": Dirichlet()},
                "right": {"u": Dirichlet(), "w": Neumann()},
            },
        )
        (level,) = ChebyshevBox([5], bkd).levels(_UNIT)
        npts = level.npts()
        left, _ = side_nodes(level, "left")
        right, _ = side_nodes(level, "right")
        scalar = _problem("x", _ADR("1"), _UNIT, coded("DD", _UNIT))
        physics = _ADR("1").physics(scalar, level)
        declarations = boundary_conditions(two, level, {"w": physics})
        assert declarations.essential is not None
        pinned = list(declarations.essential.rows().indices())
        assert pinned == sorted([int(left[0]), int(right[0]), int(left[0]) + npts])
        (robin,) = declarations.closures
        assert list(robin.rows().indices()) == [int(right[0]) + npts]
        approximation = CollocationApproximation(two, level, bkd.zeros((2 * npts,)))
        assert abs(approximation.l2_error() - math.sqrt(5.0 / 3.0)) < 1e-13
        assert abs(approximation.nodal_error() - 2.0) < 1e-13

    def test_decay_check(self) -> None:
        """Errors exp(-2N) decay at 2 per node: required 1.5 passes, 2.5
        fails; a zero error fails, and a single level or a node count
        below two is refused."""
        nodes = (4, 8, 12)
        study = NodeStudy(nodes, tuple(math.exp(-2.0 * n) for n in nodes))
        result = DecaysExponentially(1.5).verify(study)
        assert result.passed and abs(result.observed - 2.0) < 1e-12
        assert not DecaysExponentially(2.5).verify(study).passed
        zero = NodeStudy(nodes, (1.0, 0.1, 0.0))
        assert not DecaysExponentially(1.0).verify(zero).passed
        with pytest.raises(ValueError, match="at least two levels"):
            DecaysExponentially(1.0, last=1).verify(study)
        with pytest.raises(ValueError, match="at least two"):
            NodeStudy((1, 4), (1.0, 0.1))

    @pytest.mark.parametrize(
        "box, expected",
        [
            (Box([(0.0, 1.0)]), 1.0 / 3.0),
            (Box([(0.0, 1.0), (-1.0, 1.0)]), 2.0 / 3.0),
            (Box([(0.0, 1.0), (0.0, 2.0), (0.0, 1.0)]), 2.0 / 3.0),
        ],
        ids=["1d", "2d", "3d"],
    )
    def test_l2_error_by_quadrature(
        self, bkd: Backend[Array], box: Box, expected: float
    ) -> None:
        """u* = x against a zero state: the L2 error is the box integral
        of x^2, exact by the tensor quadrature and the affine map's
        Jacobian; the nodal error is max x = 1."""
        conditions: Dict[str, ConditionProtocol] = {
            side: Dirichlet() for side in box.boundaries()
        }
        problem = _problem("x", _ADR("1"), box, conditions)
        (level,) = ChebyshevBox([4], bkd).levels(box)
        approximation = CollocationApproximation(
            problem, level, bkd.zeros((level.npts(),))
        )
        assert abs(approximation.l2_error() - math.sqrt(expected)) < 1e-13
        assert abs(approximation.nodal_error() - 1.0) < 1e-13

    def test_l2_error_sees_between_nodes(self, bkd: Backend[Array]) -> None:
        """u* = x^4 on [0, 1] with the state equal to u* at the 3 nodes
        (0, 1/2, 1): the nodal error is zero, while the L2 error is that
        of the interpolating quadratic, integrated exactly here by sympy
        (degree 8, within the 6-point Gauss rule's degree 11)."""
        box = Box([(0.0, 1.0)])
        conditions: Dict[str, ConditionProtocol] = {
            side: Dirichlet() for side in box.boundaries()
        }
        problem = _problem("x**4", _ADR("1"), box, conditions)
        (level,) = ChebyshevBox([3], bkd).levels(box)
        nodes = level.mesh().points()[0]
        approximation = CollocationApproximation(problem, level, nodes**4)
        x = sp.Symbol("x")
        quadratic = sp.interpolate(
            [(0, 0), (sp.Rational(1, 2), sp.Rational(1, 16)), (1, 1)], x
        )
        expected = float(sp.integrate((x**4 - quadratic) ** 2, (x, 0, 1)))
        assert approximation.nodal_error() < 1e-14
        assert abs(approximation.l2_error() - math.sqrt(expected)) < 1e-13

    def test_h_on_a_box(self, bkd: Backend[Array]) -> None:
        """On [0, 1] x [-1, 1] with 5 Chebyshev-Gauss-Lobatto nodes per
        direction, the widest physical gap is the longer side's central
        one: the reference gap cos(pi/4) - cos(pi/2) times half the side's
        length 2, sqrt(2)/2."""
        (level,) = ChebyshevBox([5], bkd).levels(_RECTANGLE)
        assert abs(level.h() - math.sqrt(2.0) / 2.0) < 1e-14

    def test_mapped_level(self, bkd: Backend[Array]) -> None:
        """An annulus 1 <= r <= 2 meshed by a polar map over theta in
        [0, 2 pi], 8 nodes per direction: the L2 error of u* = 1 and of
        u* = x against a zero state is the root of the area, 3 pi, and of
        the integral of x^2, 15 pi / 4, so the quadrature carries |det J|
        = r; h is the widest chord, between the outer circle's central
        theta nodes."""
        annulus = Annulus((0.0, 0.0), 1.0, 2.0)
        polar = PolarTransform((1.0, 2.0), (0.0, 2.0 * math.pi), bkd)
        mesh = TransformedMesh2D(8, 8, bkd, polar)
        level = ChebyshevLevel(
            annulus, mesh, ChebyshevBasis2D(mesh, bkd), {"inner": (0,), "outer": (1,)}
        )
        conditions: Dict[str, ConditionProtocol] = {
            "inner": Dirichlet(),
            "outer": Dirichlet(),
        }
        zero = bkd.zeros((level.npts(),))
        for solution, integral in (("1", 3.0 * math.pi), ("x", 3.75 * math.pi)):
            problem = _problem(solution, _ADR("1"), annulus, conditions)
            approximation = CollocationApproximation(problem, level, zero)
            assert abs(approximation.l2_error() - math.sqrt(integral)) < 1e-12
        # Central gap of 8 nodes on [-1, 1]: cos(3 pi/7) - cos(4 pi/7).
        dtheta = math.pi * (math.cos(3 * math.pi / 7) - math.cos(4 * math.pi / 7))
        assert abs(level.h() - 4.0 * math.sin(dtheta / 2.0)) < 1e-13
        assert level.boundary_ids("outer") == (1,)

    def test_seam_rows(self, bkd: Backend[Array]) -> None:
        """An annulus level's seam joins the angle's two ends on every
        field: with 5 nodes per direction and two fields, 2 x 5 value
        rows and 2 x 5 derivative rows, and its rows are among those
        boundary_conditions builds."""
        (level,) = ChebyshevAnnulus([5], bkd).levels(_RING)
        assert [repr(c) for c in level.closures()] == ["PeriodicSeam(1)"]
        seam = level.closures()[0].rows(level, 2)
        assert seam.rows().count() == 20
        two = ManufacturedProblem(
            {"u": "x", "w": "y"},
            [Divergence("u", ["-u_x", "-u_y"]), Divergence("w", ["-w_x", "-w_y"])],
            _RING,
            {"inner": Dirichlet(), "outer": Dirichlet()},
        )
        declarations = boundary_conditions(two, level)
        assert len(declarations.closures) == 1

    def test_part_over_several_boundaries(self, bkd: Backend[Array]) -> None:
        """A part may cover several mesh boundaries (a disk's circle, all
        four sides of a square mapped onto it): on a 4 x 4 reference
        square read as one part of a disk, the left and bottom sides have
        7 nodes, the shared corner once, with the left side's normal."""
        mesh = TransformedMesh2D(4, 4, bkd)
        level = ChebyshevLevel(
            Disk(),
            mesh,
            ChebyshevBasis2D(mesh, bkd),
            {"circle": (0, 2)},
        )
        indices, normals = side_nodes(level, "circle")
        expected = bkd.asarray([0, 4, 8, 12, 1, 2, 3], dtype=bkd.int64_dtype())
        bkd.assert_allclose(indices, expected)
        bkd.assert_allclose(normals[0], bkd.asarray([-1.0, 0.0]))
        bkd.assert_allclose(normals[-1], bkd.asarray([0.0, -1.0]))


class TestRefusals:
    def test_transient(self, numpy_bkd: Backend[Array]) -> None:
        """This discretizer solves only the steady equation; on a
        transient problem it would drop du/dt, so it refuses."""
        adr = _ADR("1")
        discretizer = CollocationDiscretizer(ChebyshevBox([4], numpy_bkd), adr)
        transient = ManufacturedProblem(
            {"u": "exp(-t)*x"},
            [TimeDerivative("u"), *adr.terms(("x",))],
            _UNIT,
            coded("DN", _UNIT),
        )
        with pytest.raises(ValueError, match="steady"):
            discretizer.approximations(transient)

    def test_scalar_system_for_two_fields(self, numpy_bkd: Backend[Array]) -> None:
        """A scalar physics given a two-field problem is refused rather
        than measured on one field: the two-component rows do not fit
        its states."""
        adr = _ADR("1")
        discretizer = CollocationDiscretizer(ChebyshevBox([4], numpy_bkd), adr)
        two = ManufacturedProblem(
            {"u": "x", "w": "x"},
            [Divergence("u", ["-u_x"]), Divergence("w", ["-w_x"])],
            _UNIT,
            {"left": Dirichlet(), "right": Dirichlet()},
        )
        with pytest.raises(ValueError, match="size 8, not the physics' 4 states"):
            discretizer.approximations(two)

    def test_not_a_box(self, numpy_bkd: Backend[Array]) -> None:
        """Chebyshev boxes mesh boxes only; a disk waits on a polar
        family."""
        with pytest.raises(TypeError, match="meshes a Box"):
            ChebyshevBox([4], numpy_bkd).levels(Disk())

    def test_initial_guess_not_a_guess(self, numpy_bkd: Backend[Array]) -> None:
        """A bare value is refused at construction: the discretizer takes
        an object that builds each level's starting state."""
        with pytest.raises(TypeError, match="InitialGuessProtocol"):
            CollocationDiscretizer(
                ChebyshevBox([4], numpy_bkd),
                _ADR("1"),
                initial_guess=2.0,  # type: ignore[arg-type]
            )

    def test_initial_guess_shape(self, numpy_bkd: Backend[Array]) -> None:
        """A guess that is not a column of the system's states is refused
        before the solve."""
        discretizer = CollocationDiscretizer(
            ChebyshevBox([4], numpy_bkd), _ADR("1"), initial_guess=_FlatGuess()
        )
        problem = _problem("x", _ADR("1"), _UNIT, coded("DD", _UNIT))
        with pytest.raises(ValueError, match=r"shape \(4, 1\); got \(4,\)"):
            discretizer.approximations(problem)

    def test_half_periodic(self, numpy_bkd: Backend[Array]) -> None:
        """A periodic side needs its partner across the same coordinate
        periodic too: one alone has nothing to match."""
        adr = _ADR("1", None, "-1")
        conditions: Dict[str, ConditionProtocol] = {
            "left": Periodic(),
            "right": Dirichlet(),
        }
        problem = _problem("sin(2*pi*x)", adr, _UNIT, conditions)
        (level,) = ChebyshevBox([5], numpy_bkd).levels(_UNIT)
        with pytest.raises(ValueError, match="'right', across x, is not"):
            boundary_conditions(problem, level)

    def test_level_parts(self, numpy_bkd: Backend[Array]) -> None:
        """A level names the mesh boundary of every boundary part of its
        domain, and of no other."""
        mesh = TransformedMesh2D(4, 4, numpy_bkd)
        basis = ChebyshevBasis2D(mesh, numpy_bkd)
        with pytest.raises(ValueError, match="must name the domain's boundary"):
            ChebyshevLevel(Annulus(), mesh, basis, {"inner": (0,)})
        with pytest.raises(ValueError, match="'outer' covers no mesh boundary"):
            ChebyshevLevel(Annulus(), mesh, basis, {"inner": (0,), "outer": ()})
        with pytest.raises(TypeError, match="InteriorClosureProtocol"):
            ChebyshevLevel(
                Annulus(),
                mesh,
                basis,
                {"inner": (0,), "outer": (1,)},
                closures=[2],  # type: ignore[list-item]
            )
        level = ChebyshevLevel(Annulus(), mesh, basis, {"inner": (0,), "outer": (1,)})
        with pytest.raises(ValueError, match="no boundary part 'left'"):
            level.boundary_ids("left")

    def test_ring_families(self, numpy_bkd: Backend[Array]) -> None:
        """Each ring family meshes its own domain only, the annulus about
        the origin; a seam names one of the mesh's reference
        directions."""
        with pytest.raises(TypeError, match="meshes an Annulus"):
            ChebyshevAnnulus([4], numpy_bkd).levels(_UNIT)
        with pytest.raises(ValueError, match="about the origin"):
            ChebyshevAnnulus([4], numpy_bkd).levels(Annulus((1.0, 0.0), 1.0, 2.0))
        with pytest.raises(TypeError, match="meshes an EllipticalAnnulus"):
            ChebyshevEllipticalAnnulus([4], numpy_bkd).levels(_RING)
        (level,) = ChebyshevAnnulus([4], numpy_bkd).levels(_RING)
        with pytest.raises(ValueError, match="direction 2 is not"):
            PeriodicSeam[Array](2).rows(level, 1)

    def test_periodic_off_a_box(self, numpy_bkd: Backend[Array]) -> None:
        """Periodic conditions pair a box's opposite sides; an annulus'
        circles have no partner."""
        annulus = Annulus()
        polar = PolarTransform((1.0, 2.0), (0.0, 2.0 * math.pi), numpy_bkd)
        mesh = TransformedMesh2D(4, 4, numpy_bkd, polar)
        level = ChebyshevLevel(
            annulus,
            mesh,
            ChebyshevBasis2D(mesh, numpy_bkd),
            {"inner": (0,), "outer": (1,)},
        )
        conditions: Dict[str, ConditionProtocol] = {
            "inner": Periodic(),
            "outer": Dirichlet(),
        }
        problem = _problem("x", _ADR("1"), annulus, conditions)
        with pytest.raises(TypeError, match="opposite sides of a Box"):
            boundary_conditions(problem, level)

    def test_flux_condition_needs_a_flux(self, numpy_bkd: Backend[Array]) -> None:
        """Neumann and Robin rows prescribe a normal flux, so they need
        the flux whose normal component they prescribe."""
        adr = _ADR("1")
        problem = _problem("x", adr, _UNIT, coded("DN", _UNIT))
        (level,) = ChebyshevBox([5], numpy_bkd).levels(_UNIT)
        with pytest.raises(ValueError, match="prescribes a flux of 'u'"):
            boundary_conditions(problem, level)

    def test_flux_of_an_unknown_field(self, numpy_bkd: Backend[Array]) -> None:
        """A flux for a field the problem lacks is a misspelling, not a
        flux to ignore."""
        adr = _ADR("1")
        problem = _problem("x", adr, _UNIT, coded("DN", _UNIT))
        (level,) = ChebyshevBox([5], numpy_bkd).levels(_UNIT)
        physics = adr.physics(problem, level)
        with pytest.raises(ValueError, match=r"name fields \['w'\]"):
            boundary_conditions(problem, level, {"w": physics})

    def test_flux_that_is_not_one(self, numpy_bkd: Backend[Array]) -> None:
        problem = _problem("x", _ADR("1"), _UNIT, coded("DN", _UNIT))
        (level,) = ChebyshevBox([5], numpy_bkd).levels(_UNIT)
        with pytest.raises(TypeError, match="flux of 'u' must satisfy"):
            boundary_conditions(problem, level, {"u": object()})  # type: ignore[dict-item]

    def test_state_of_the_wrong_size(self, numpy_bkd: Backend[Array]) -> None:
        """A two-field problem's approximation needs both fields' values."""
        two = ManufacturedProblem(
            {"u": "x", "w": "x"},
            [Divergence("u", ["-u_x"]), Divergence("w", ["-w_x"])],
            _UNIT,
            {"left": Dirichlet(), "right": Dirichlet()},
        )
        (level,) = ChebyshevBox([4], numpy_bkd).levels(_UNIT)
        with pytest.raises(ValueError, match=r"shape \(8,\)"):
            CollocationApproximation(two, level, numpy_bkd.zeros((4,)))

    def test_helpers(self, numpy_bkd: Backend[Array]) -> None:
        """Coefficients may use only coordinates, t and constants (not
        the state); a fixed vector may not use t and needs one entry per
        coordinate."""
        problem = _problem("x", _ADR("1"), _UNIT, coded("DN", _UNIT))
        (level,) = ChebyshevBox([5], numpy_bkd).levels(_UNIT)
        with pytest.raises(ValueError, match="may depend only on"):
            field(problem, "u_x", level)
        with pytest.raises(ValueError, match="may depend only on"):
            nodal_vector(problem, ["x*t"], level)
        with pytest.raises(ValueError, match="one expression per coordinate"):
            nodal_vector(problem, ["1", "2"], level)


class _FlatGuess:
    """A guess of the wrong shape, (nstates,)."""

    def guess(
        self,
        problem: ManufacturedProblem,
        level: ChebyshevLevel[Array],
        system: CollocationSystem[Array],
    ) -> Array:
        return level.bkd().zeros((system.nstates(),))


class _RecordingGuess:
    """Starts every level from a constant and records the node counts of
    the levels it was asked for."""

    def __init__(self) -> None:
        self._npts: List[int] = []

    def npts(self) -> List[int]:
        return self._npts

    def guess(
        self,
        problem: ManufacturedProblem,
        level: ChebyshevLevel[Array],
        system: CollocationSystem[Array],
    ) -> Array:
        self._npts.append(level.npts())
        return ConstantGuess(0.5).guess(problem, level, system)


def test_initial_guess_reaches_each_solve(bkd: Backend[Array]) -> None:
    """An injected guess is asked once per level, coarsest first, with
    that level's system, and the solves from it recover the solution."""
    adr = _ADR("1 + x/2")
    problem = _problem("1 + x - x**2/2 + x**3", adr, _UNIT, coded("DN", _UNIT))
    recording = _RecordingGuess()
    discretizer = CollocationDiscretizer(
        ChebyshevBox([6, 8], bkd), adr, initial_guess=recording
    )
    errors = [a.l2_error() for a in discretizer.approximations(problem)]
    assert recording.npts() == [6, 8]
    assert max(errors) < 1e-10
