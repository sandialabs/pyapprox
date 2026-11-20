r"""Collocation two-species reaction-diffusion verified by manufactured
solutions.

The steady system the physics solves is :math:`\nabla\cdot(D_k\nabla
u_k) + R_k(u, v) + f_k = 0` for the species :math:`u` (first) and
:math:`v`, stated as terms
:math:`\nabla\cdot(-D_k\nabla u_k) - R_k = f_k`, with the kinetics
written out here rather than taken from the reaction objects under test:

- linear, :math:`R_u = a_{00} u + a_{01} v`, :math:`R_v = a_{10} u +
  a_{11} v`;
- FitzHugh-Nagumo, :math:`R_u = u(1 - u)(u - \alpha) - v`,
  :math:`R_v = \epsilon(\beta u - \gamma v)`.

A species' Neumann and Robin sides prescribe its own diffusive flux,
:class:`~pyapprox.pde.collocation.physics.reaction_diffusion.
SpeciesDiffusiveFlux`. A species with zero diffusivity has no divergence
term, so it takes no boundary conditions and its equation is algebraic
at every node.

Cases:

- recovery (L2 error over both species below 1e-10) of polynomials
  vanishing at both ends of [-1, 1] under zero Dirichlet ends, with
  linear and FitzHugh-Nagumo kinetics, the latter with a non-diffusing
  second species;
- recovery of cubics on [0, 1] with variable diffusivities, coupled
  linear kinetics, and different conditions on each species; and on
  [0, 1] x [-1, 1] on both backends;
- spectral decay, fitted per node, of analytic solutions in 1D and 2D;
- wrong-model control: each species' flux sides given the other
  species' flux stall.
"""

from typing import Dict, Mapping, Tuple

import pytest

from pyapprox.pde.collocation.physics.reaction_diffusion import (
    FitzHughNagumoReaction,
    LinearReaction,
    ReactionProtocol,
    SpeciesDiffusiveFlux,
    TwoSpeciesReactionDiffusionPhysics,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevLevel,
    collocation_system,
    field,
    forcing,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Robin,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, Source, TermProtocol
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.collocation_verification import decays, recovers, stalls

_KINDS: Dict[str, ConditionProtocol] = {
    "D": Dirichlet(),
    "N": Neumann(),
    "R": Robin(1.0),
}


class _Linear:
    """Linear kinetics with coefficients a00, a01, a10, a11."""

    def __init__(self, a00: float, a01: float, a10: float, a11: float) -> None:
        self._a = (a00, a01, a10, a11)

    def expressions(self) -> Tuple[str, str]:
        a00, a01, a10, a11 = self._a
        return f"({a00})*u + ({a01})*v", f"({a10})*u + ({a11})*v"

    def reaction(self, bkd: Backend[Array]) -> ReactionProtocol[Array]:
        return LinearReaction(*self._a, bkd)

    def __repr__(self) -> str:
        return f"_Linear{self._a}"


class _FitzHughNagumo:
    """FitzHugh-Nagumo kinetics with alpha, eps, beta, gamma."""

    def __init__(self, alpha: float, eps: float, beta: float, gamma: float) -> None:
        self._p = (alpha, eps, beta, gamma)

    def expressions(self) -> Tuple[str, str]:
        alpha, eps, beta, gamma = self._p
        return (
            f"u*(1 - u)*(u - ({alpha})) - v",
            f"({eps})*(({beta})*u - ({gamma})*v)",
        )

    def reaction(self, bkd: Backend[Array]) -> ReactionProtocol[Array]:
        return FitzHughNagumoReaction(*self._p, bkd)

    def __repr__(self) -> str:
        return f"_FitzHughNagumo{self._p}"


_Kinetics = _Linear | _FitzHughNagumo


class _ReactionDiffusion:
    """The physics under test and its strong form.

    Parameters
    ----------
    diffusivities : tuple of str
        :math:`D_u` and :math:`D_v`; ``"0"`` for a species that does not
        diffuse.
    kinetics : _Linear or _FitzHughNagumo
        The reaction, as expressions and as the object under test.
    swap_fluxes : bool, optional
        Give each species' flux sides the other species' flux (the
        wrong-model control).
    """

    def __init__(
        self,
        diffusivities: Tuple[str, str],
        kinetics: _Kinetics,
        swap_fluxes: bool = False,
    ) -> None:
        self._diffusivities, self._kinetics = diffusivities, kinetics
        self._swap = swap_fluxes

    def terms(self, coords: Tuple[str, ...]) -> list[TermProtocol]:
        """The strong form as terms on rows ``u`` and ``v``."""
        reactions = self._kinetics.expressions()
        terms: list[TermProtocol] = []
        for row, diffusivity, reaction in zip(
            ("u", "v"), self._diffusivities, reactions
        ):
            if diffusivity != "0":
                terms.append(
                    Divergence(row, [f"-({diffusivity})*{row}_{c}" for c in coords])
                )
            terms.append(Source(row, f"-({reaction})"))
        return terms

    def problem(
        self,
        solutions: Tuple[str, str],
        box: Box,
        codes: Tuple[str, str],
    ) -> ManufacturedProblem:
        """``codes`` gives each species' conditions, one letter per side
        in the box's side order; a non-diffusing species' code is
        ignored."""
        conditions: Dict[str, Mapping[str, ConditionProtocol]] = {}
        for index, part in enumerate(box.boundaries()):
            conditions[part] = {
                row: _KINDS[code[index]]
                for row, code, diffusivity in zip(
                    ("u", "v"), codes, self._diffusivities
                )
                if diffusivity != "0"
            }
        return ManufacturedProblem(
            {"u": solutions[0], "v": solutions[1]},
            self.terms(box.coordinates()),
            box,
            conditions,
        )

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        bkd = level.bkd()
        physics = TwoSpeciesReactionDiffusionPhysics(
            level.basis(),
            bkd,
            diffusion0=field(problem, self._diffusivities[0], level),
            diffusion1=field(problem, self._diffusivities[1], level),
            reaction=self._kinetics.reaction(bkd),
            forcing0=forcing(problem, "u", level),
            forcing1=forcing(problem, "v", level),
        )
        own = (0, 1) if not self._swap else (1, 0)
        fluxes = {
            row: SpeciesDiffusiveFlux(physics, species)
            for row, species, diffusivity in zip(("u", "v"), own, self._diffusivities)
            if diffusivity != "0"
        }
        return collocation_system(physics, problem, level, fluxes)

    def __repr__(self) -> str:
        return (
            f"_ReactionDiffusion({self._diffusivities!r}, {self._kinetics!r}, "
            f"swap_fluxes={self._swap})"
        )


_REFERENCE = Box([(-1.0, 1.0)])
_UNIT = Box([(0.0, 1.0)])
_RECTANGLE = Box([(0.0, 1.0), (-1.0, 1.0)])
_COUPLED = _Linear(1.0, -0.5, 0.5, -1.0)


@pytest.mark.parametrize(
    "solutions, diffusivities, kinetics, npts",
    [
        (("1 - x**2", "1 - x**2"), ("0.1", "0.05"), _Linear(0.0, 0.0, 0.0, 0.0), 15),
        (("1 - x**2", "(1 - x**2)/2"), ("1", "0.5"), _COUPLED, 15),
        (
            ("(1 - x**2)/2", "3*(1 - x**2)/10"),
            ("0.01", "0"),
            _FitzHughNagumo(0.1, 0.01, 0.5, 1.0),
            15,
        ),
    ],
    ids=["uncoupled", "linear", "fitzhugh-nagumo"],
)
def test_recovers_dirichlet_polynomials(
    numpy_bkd: Backend[Array],
    solutions: Tuple[str, str],
    diffusivities: Tuple[str, str],
    kinetics: _Kinetics,
    npts: int,
) -> None:
    """Polynomials vanishing at both ends of [-1, 1], zero Dirichlet ends
    on each diffusing species: reproduced to 1e-10."""
    rd = _ReactionDiffusion(diffusivities, kinetics)
    problem = rd.problem(solutions, _REFERENCE, ("DD", "DD"))
    recovers(problem, rd, npts, numpy_bkd)


@pytest.mark.parametrize(
    "codes", [("DN", "ND"), ("RD", "DR"), ("NR", "RN"), ("DD", "NN")]
)
@pytest.mark.parametrize(
    "kinetics",
    [_COUPLED, _FitzHughNagumo(0.1, 0.5, 0.5, 1.0)],
    ids=["linear", "fitzhugh-nagumo"],
)
def test_recovers_cubics_1d(
    numpy_bkd: Backend[Array], codes: Tuple[str, str], kinetics: _Kinetics
) -> None:
    """u = 1 + x - x^2/2 + x^3, v = x^2 - x^3/2 on [0, 1], D_u = 1 + x/2,
    D_v = 2 - x, each species under its own conditions: 8 nodes
    reproduce both to 1e-10."""
    rd = _ReactionDiffusion(("1 + x/2", "2 - x"), kinetics)
    problem = rd.problem(("1 + x - x**2/2 + x**3", "x**2 - x**3/2"), _UNIT, codes)
    recovers(problem, rd, 8, numpy_bkd)


@pytest.mark.parametrize("codes", [("DRDN", "NDRD"), ("RRRR", "DDDD")])
def test_recovers_cubics_2d(bkd: Backend[Array], codes: Tuple[str, str]) -> None:
    """u = x y + y^2 - x^2 y, v = x^2 + x y^2 on [0, 1] x [-1, 1],
    D_u = 1 + x y/4, D_v = 1/2, coupled linear kinetics: 6 x 6 nodes
    reproduce both to 1e-10, on both backends."""
    rd = _ReactionDiffusion(("1 + x*y/4", "1/2"), _COUPLED)
    problem = rd.problem(("x*y + y**2 - x**2*y", "x**2 + x*y**2"), _RECTANGLE, codes)
    recovers(problem, rd, 6, bkd)


@pytest.mark.parametrize(
    "kinetics",
    [_COUPLED, _FitzHughNagumo(0.1, 0.5, 0.5, 1.0)],
    ids=["linear", "fitzhugh-nagumo"],
)
def test_decays_1d(numpy_bkd: Backend[Array], kinetics: _Kinetics) -> None:
    """u = exp(x) sin(3 x), v = cos(2 x) on [0, 1], D_u = 1 + x/2,
    D_v = 2 - x, u Dirichlet-Neumann and v Robin-Dirichlet: exponential
    decay over 4-12 nodes (observed 2.09 and 2.07 per node for the two
    kinetics, required 1)."""
    rd = _ReactionDiffusion(("1 + x/2", "2 - x"), kinetics)
    problem = rd.problem(("exp(x)*sin(3*x)", "cos(2*x)"), _UNIT, ("DN", "RD"))
    decays(problem, rd, [4, 6, 8, 10, 12], 1.0, numpy_bkd)


def test_decays_2d(numpy_bkd: Backend[Array]) -> None:
    """u = exp(x/2) sin(2 y) + x y, v = cos(x + y) on [0, 1] x [-1, 1],
    D_u = 1 + x y/4, D_v = 1/2, FitzHugh-Nagumo kinetics, mixed sides:
    exponential decay over 4-12 nodes per direction (0.14 to 3e-9;
    observed 2.20 per node, required 1)."""
    rd = _ReactionDiffusion(("1 + x*y/4", "1/2"), _FitzHughNagumo(0.1, 0.5, 0.5, 1.0))
    problem = rd.problem(
        ("exp(x/2)*sin(2*y) + x*y", "cos(x + y)"), _RECTANGLE, ("DRDN", "NDRD")
    )
    decays(problem, rd, [4, 6, 8, 10, 12], 1.0, numpy_bkd)


def test_swapped_fluxes_stall(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: with D_u = 1 + x/2 and D_v = 2 - x, each
    species' Neumann and Robin ends given the other species' flux
    prescribe the wrong quantity, so the error stops falling and the
    cases above check that each side reads its own species."""
    rd = _ReactionDiffusion(("1 + x/2", "2 - x"), _COUPLED, swap_fluxes=True)
    problem = rd.problem(("exp(x)*sin(3*x)", "cos(2*x)"), _UNIT, ("DN", "RD"))
    stalls(problem, rd, [6, 8, 10, 12], numpy_bkd)
