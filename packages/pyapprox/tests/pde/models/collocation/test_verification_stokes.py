r"""Collocation Stokes flow verified by manufactured solutions.

:class:`~pyapprox.pde.collocation.physics.stokes.StokesPhysics` solves
:math:`-\nabla\cdot(2\mu\varepsilon(u)) + \nabla p = f`,
:math:`\nabla\cdot u = g` with the velocity on ``n`` Gauss-Lobatto nodes
per direction and the pressure on ``n - 2`` interior Gauss nodes. The
problem states the momentum rows as the divergence of
:math:`q_i = -\sigma_{i\cdot}`, :math:`\sigma = 2\mu\varepsilon(u) - pI`,
so a traction side closes on the physics' Cauchy stress through
:class:`~pyapprox.pde.models.collocation.verification.TractionTerm`. With
the velocity prescribed on every side the pressure is fixed by its
integral (:func:`~pyapprox.pde.models.collocation.verification.
mean_value_rows`).

Cases:

- recovery (L2 error over all fields below 1e-10) of a polynomial pair,
  velocity of degree at most ``n - 1`` and pressure at most ``n - 3`` per
  direction: Dirichlet sides with constant and variable viscosity, a
  traction side, and in 3D;
- spectral decay of an analytic pair with Dirichlet sides, and with a
  traction side;
- wrong-model control: the pressure on the velocity's own nodes;
- Navier-Stokes (:class:`~pyapprox.pde.collocation.physics.stokes.
  NavierStokesPhysics`): recovery of the polynomial pair with the
  convective term, and spectral decay of Kovasznay flow, with Dirichlet
  sides and with a traction side.
"""

import math
from functools import partial
from typing import Dict, List, Mapping, Sequence, Union

import pytest

from pyapprox.pde.collocation.basis.transfer import grid_transfer_matrix
from pyapprox.pde.collocation.physics.stokes import NavierStokesPhysics, StokesPhysics
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevBox,
    ChebyshevLevel,
    CollocationDiscretizer,
    TractionTerm,
    boundary_conditions,
    field,
    forcing,
    mean_value_rows,
)
from pyapprox.pde.verification.conditions import ConditionProtocol
from pyapprox.pde.verification.domains import Box, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.protocols import SpaceLevel
from pyapprox.pde.verification.terms import Divergence, Source, TermProtocol
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.collocation_verification import coded, decays, recovers

_VELOCITY = ("ux", "uy", "uz")
_COORDS = ("x", "y", "z")
_Conditions = Mapping[str, Union[ConditionProtocol, Mapping[str, ConditionProtocol]]]
_STAGGERED = partial(ChebyshevBox, interior_fields=["p"])


class _Stokes:
    """The physics under test and its strong form, from one viscosity.

    Parameters
    ----------
    mu : str
        :math:`\\mu(x)`.
    mean : bool, optional
        Fix the pressure by its integral; default True. A traction side
        determines the pressure itself.
    convective : bool, optional
        Navier-Stokes: add :math:`(u\\cdot\\nabla)u` to the momentum rows
        and solve with :class:`NavierStokesPhysics`; default False.
    """

    def __init__(self, mu: str, mean: bool = True, convective: bool = False) -> None:
        self._mu, self._mean, self._convective = mu, mean, convective

    def terms(self, ndim: int) -> List[TermProtocol]:
        """The strong form as terms on the velocity and pressure rows."""
        velocity, coords = _VELOCITY[:ndim], _COORDS[:ndim]
        terms: List[TermProtocol] = []
        for i in range(ndim):
            fluxes = []
            for j in range(ndim):
                shear = (
                    f"-({self._mu})*({velocity[i]}_{coords[j]} + "
                    f"{velocity[j]}_{coords[i]})"
                )
                fluxes.append(f"{shear} + p" if i == j else shear)
            terms.append(Divergence(velocity[i], fluxes))
            if self._convective:
                advection = " + ".join(
                    f"{u}*{velocity[i]}_{c}" for u, c in zip(velocity, coords)
                )
                terms.append(Source(velocity[i], advection))
        divergence = " + ".join(f"{u}_{c}" for u, c in zip(velocity, coords))
        terms.append(Source("p", divergence))
        return terms

    def problem(
        self,
        velocity: Sequence[str],
        pressure: str,
        domain: DomainProtocol,
        conditions: _Conditions,
    ) -> ManufacturedProblem:
        ndim = domain.ndim()
        solution: Dict[str, str] = dict(zip(_VELOCITY[:ndim], velocity))
        solution["p"] = pressure
        return ManufacturedProblem(solution, self.terms(ndim), domain, conditions)

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        bkd = level.bkd()
        rows = problem.fields()[:-1]
        velocity, pressure = level.field_mesh(rows[0]), level.field_mesh("p")
        operator = NavierStokesPhysics if self._convective else StokesPhysics
        physics = operator(
            level.field_basis(rows[0]),
            grid_transfer_matrix(pressure, velocity, bkd),
            grid_transfer_matrix(velocity, pressure, bkd),
            viscosity=field(problem, self._mu, level),
            forcing=[forcing(problem, row, level) for row in rows],
            divergence=forcing(problem, "p", level),
        )
        fluxes = {row: TractionTerm(physics, i) for i, row in enumerate(rows)}
        declarations = boundary_conditions(problem, level, fluxes)
        closures = list(declarations.closures)
        if self._mean:
            closures.append(mean_value_rows(problem, level, "p"))
        return CollocationSystem(physics, declarations.essential, closures)

    def __repr__(self) -> str:
        return (
            f"_Stokes({self._mu!r}, mean={self._mean}, convective={self._convective})"
        )


_UNIT_SQUARE = Box([(0.0, 1.0)] * 2)
_CUBE = Box([(-1.0, 1.0)] * 3)
# Velocity of degree 4 per direction, divergence nonzero; pressure of
# degree 2 per direction.
_PAIR = (("x**2*y**3 + y", "x**4 - x*y**2"), "x**2*y - x*y**2 + 1")


@pytest.mark.parametrize("mu", ["1", "1 + x*y/2"], ids=["constant", "variable"])
def test_recovers_dirichlet(bkd: Backend[Array], mu: str) -> None:
    """The pair on 5 x 5 velocity and 3 x 3 pressure nodes, every side
    Dirichlet, the pressure fixed by its integral."""
    stokes = _Stokes(mu)
    problem = stokes.problem(*_PAIR, _UNIT_SQUARE, coded("DDDD", _UNIT_SQUARE))
    recovers(problem, stokes, 5, bkd, family=_STAGGERED)


def test_recovers_traction_side(bkd: Backend[Array]) -> None:
    """The right side carries the traction, which fixes the pressure."""
    stokes = _Stokes("1 + x*y/2", mean=False)
    problem = stokes.problem(*_PAIR, _UNIT_SQUARE, coded("DNDD", _UNIT_SQUARE))
    recovers(problem, stokes, 5, bkd, family=_STAGGERED)


def test_recovers_3d(numpy_bkd: Backend[Array]) -> None:
    """A degree-2 velocity and a degree-1 pressure on [-1, 1]^3: 4 x 4 x 4
    velocity and 2 x 2 x 2 pressure nodes, every face Dirichlet."""
    stokes = _Stokes("2")
    problem = stokes.problem(
        ("y*z + x**2", "x*z - y**2", "x*y + z**2"),
        "x + y*z",
        _CUBE,
        coded("DDDDDD", _CUBE),
    )
    recovers(problem, stokes, 4, numpy_bkd, family=_STAGGERED)


_SMOOTH = (("sin(x + y)/2", "cos(x - 2*y)/4"), "exp(x*y) - 1")


def test_decays_dirichlet(numpy_bkd: Backend[Array]) -> None:
    """An analytic pair, viscosity 1 + x y/2, every side Dirichlet:
    exponential decay over 5-13 velocity nodes per direction (1.1e-2 to
    3.4e-11; observed 2.52 per node, required 1)."""
    stokes = _Stokes("1 + x*y/2")
    problem = stokes.problem(*_SMOOTH, _UNIT_SQUARE, coded("DDDD", _UNIT_SQUARE))
    decays(problem, stokes, [5, 7, 9, 11, 13], 1.0, numpy_bkd, family=_STAGGERED)


def test_decays_traction_side(numpy_bkd: Backend[Array]) -> None:
    """The same pair with a traction side: exponential decay (4.6e-3 to
    1.6e-12; observed 2.76 per node, required 1)."""
    stokes = _Stokes("1 + x*y/2", mean=False)
    problem = stokes.problem(*_SMOOTH, _UNIT_SQUARE, coded("DNDD", _UNIT_SQUARE))
    decays(problem, stokes, [5, 7, 9, 11, 13], 1.0, numpy_bkd, family=_STAGGERED)


@pytest.mark.parametrize("n", [5, 7])
def test_pressure_on_the_velocity_nodes_is_undetermined(
    numpy_bkd: Backend[Array], n: int
) -> None:
    """Wrong-model control: the pressure on the velocity's own nodes
    (P_N-P_N) has spurious modes besides the constant, so even with its
    integral fixed the linear solve is singular and fails."""
    stokes = _Stokes("1 + x*y/2")
    problem = stokes.problem(*_SMOOTH, _UNIT_SQUARE, coded("DDDD", _UNIT_SQUARE))
    discretizer = CollocationDiscretizer(ChebyshevBox([n], numpy_bkd), stokes)
    with pytest.raises(RuntimeError, match="non-finite solution"):
        discretizer.solve(problem, SpaceLevel(0))


@pytest.mark.parametrize(
    "codes, mean", [("DDDD", True), ("DNDD", False)], ids=["dirichlet", "traction"]
)
def test_navier_stokes_recovers_polynomials(
    bkd: Backend[Array], codes: str, mean: bool
) -> None:
    """The polynomial pair with the convective term, viscosity 1 + x y/2,
    on 5 x 5 velocity nodes. The convective product is evaluated at the
    nodes from exact nodal values and never differentiated, so its
    aliasing costs nothing here and the pair is reproduced."""
    stokes = _Stokes("1 + x*y/2", mean=mean, convective=True)
    problem = stokes.problem(*_PAIR, _UNIT_SQUARE, coded(codes, _UNIT_SQUARE))
    recovers(problem, stokes, 5, bkd, family=_STAGGERED)


_RE = 20.0
_LAMBDA = _RE / 2.0 - math.sqrt(_RE**2 / 4.0 + 4.0 * math.pi**2)
_KOVASZNAY = (
    (
        f"1 - exp({_LAMBDA!r}*x)*cos(2*pi*y)",
        f"{_LAMBDA / (2.0 * math.pi)!r}*exp({_LAMBDA!r}*x)*sin(2*pi*y)",
    ),
    f"(1 - exp({2.0 * _LAMBDA!r}*x))/2",
)
_KOVASZNAY_BOX = Box([(-0.5, 1.0), (-0.5, 1.5)])


@pytest.mark.parametrize(
    "codes, mean", [("DDDD", True), ("DNDD", False)], ids=["dirichlet", "traction"]
)
def test_navier_stokes_decays_kovasznay(
    numpy_bkd: Backend[Array], codes: str, mean: bool
) -> None:
    """Kovasznay flow at Re = 20 (viscosity 1/20, no forcing) on
    [-0.5, 1] x [-0.5, 1.5], Newton from zero: exponential decay over
    8-24 velocity nodes per direction (Dirichlet 5.9e-1 to 5.1e-12,
    observed 1.60 per node; traction side 7.1e-1 to 4.9e-12, 1.62;
    required 1)."""
    stokes = _Stokes(repr(1.0 / _RE), mean=mean, convective=True)
    problem = stokes.problem(*_KOVASZNAY, _KOVASZNAY_BOX, coded(codes, _KOVASZNAY_BOX))
    decays(problem, stokes, [8, 12, 16, 20, 24], 1.0, numpy_bkd, family=_STAGGERED)
