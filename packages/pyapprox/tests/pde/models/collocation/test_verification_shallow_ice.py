r"""Collocation shallow ice verified by manufactured solutions.

The steady equation is :math:`\nabla\cdot q = f` with the ice flux
:math:`q = -D \nabla s`, surface :math:`s = H + b`, and

.. math:: D = \gamma H^5 (|\nabla s|^2 + \epsilon) + \frac{\rho g}{C} H^2,
   \qquad \gamma = \frac{2 A (\rho g)^3}{5},

Glen's exponent 3 and :math:`\epsilon = 10^{-12}`, written out here from
the bed :math:`b`, friction :math:`C` and constants, not taken from the
physics. Neumann and Robin sides prescribe :math:`q\cdot n` through the
injected :class:`ShallowIceFlux`. :math:`A = 2.5` and :math:`\rho = 1/g`
make :math:`\gamma \approx 1` and the sliding coefficient :math:`1/C`.
Solves start from :math:`H = 2`: the Jacobian vanishes at zero
thickness.

Cases:

- recovery (L2 error below 1e-10) of the old residual tests' profiles
  on [-1, 1] under Dirichlet ends: flat, sloped and curved beds, a higher
  friction, and a quartic profile;
- recovery on [0, 1] with variable friction and a flux end, Neumann or
  Robin, on both backends; and in 2D on [0, 1] x [-1, 1];
- spectral decay, fitted per node, of an analytic profile with a flux
  end;
- wrong-model control: a physics that takes the surface to be the
  thickness (no bed) stalls.
"""

from typing import List, Optional, Sequence

import pytest
import sympy as sp

from pyapprox.pde.collocation.physics.shallow_ice import (
    ShallowIceFlux,
    ShallowIcePhysics,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevLevel,
    ConstantGuess,
    collocation_system,
    field,
    forcing,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, TermProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.collocation_verification import coded, decays, recovers, stalls

_GRAVITY = 9.81
_A = 2.5
_RHO = 1.0 / _GRAVITY
_START = ConstantGuess(2.0)


class _ShallowIce:
    """The physics under test and its strong form.

    Parameters
    ----------
    bed : str
        :math:`b(x)`.
    friction : str
        :math:`C(x)`.
    physics_bed : str, optional
        The bed the physics is given; default ``bed``. A control passes
        another.
    """

    def __init__(
        self, bed: str, friction: str, physics_bed: Optional[str] = None
    ) -> None:
        self._bed, self._friction = bed, friction
        self._physics_bed = bed if physics_bed is None else physics_bed

    def terms(self, coords: Sequence[str]) -> List[TermProtocol]:
        """The strong form as one divergence on row ``u``."""
        bed = sp.sympify(self._bed)
        slopes = [f"(u_{c} + ({sp.diff(bed, sp.Symbol(c))}))" for c in coords]
        squared = " + ".join(f"{slope}**2" for slope in slopes)
        rho_g = _RHO * _GRAVITY
        gamma = 2 * _A * rho_g**3.0 / 5.0
        diffusion = (
            f"{gamma!r}*u**5*({squared} + 1e-12) + {rho_g!r}/({self._friction})*u**2"
        )
        return [Divergence("u", [f"-({diffusion})*{slope}" for slope in slopes])]

    def problem(self, solution: str, box: Box, codes: str) -> ManufacturedProblem:
        return ManufacturedProblem(
            {"u": solution}, self.terms(box.coordinates()), box, coded(codes, box)
        )

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = ShallowIcePhysics(
            level.basis(),
            level.bkd(),
            bed=field(problem, self._physics_bed, level),
            friction=field(problem, self._friction, level),
            A=_A,
            rho=_RHO,
            forcing=forcing(problem, "u", level),
        )
        return collocation_system(
            physics, problem, level, {"u": ShallowIceFlux(physics)}
        )

    def __repr__(self) -> str:
        return (
            f"_ShallowIce({self._bed!r}, {self._friction!r}, "
            f"physics_bed={self._physics_bed!r})"
        )


_REFERENCE = Box([(-1.0, 1.0)])
_UNIT = Box([(0.0, 1.0)])
_RECTANGLE = Box([(0.0, 1.0), (-1.0, 1.0)])


@pytest.mark.parametrize(
    "solution, bed, friction, npts",
    [
        ("2 + 0.5*(1 - x**2)", "0", "1", 16),
        ("2 + 0.5*(1 - x**2)", "0.1*x", "1", 16),
        ("2 + 0.5*(1 - x**2)", "0.05*x**2", "1", 16),
        ("2 + 0.5*(1 - x**2)", "0.1*x", "10", 16),
        ("2 + 0.25*(1 - x**2)**2", "0.1*x", "1", 30),
    ],
    ids=["flat", "sloped", "curved", "higher_friction", "quartic"],
)
def test_recovers_dirichlet_profiles(
    numpy_bkd: Backend[Array], solution: str, bed: str, friction: str, npts: int
) -> None:
    """The old residual tests' profiles on [-1, 1], Dirichlet ends: the
    flux is a polynomial (degree 13, or 29 for the quartic), so npts
    nodes above its degree reproduce the profile to 1e-10."""
    ice = _ShallowIce(bed, friction)
    problem = ice.problem(solution, _REFERENCE, "DD")
    recovers(problem, ice, npts, numpy_bkd, initial_guess=_START)


@pytest.mark.parametrize("codes", ["DN", "ND", "RD", "NR"])
def test_recovers_flux_end(bkd: Backend[Array], codes: str) -> None:
    """H = 1 + x - x^2/2 on [0, 1], bed 0.1 x, friction 1 + x, a Neumann
    or Robin end: the sliding term 1/(1 + x) is not polynomial, but 16
    nodes resolve it, reproducing H to 1e-10 on both backends."""
    ice = _ShallowIce("0.1*x", "1 + x")
    problem = ice.problem("1 + x - x**2/2", _UNIT, codes)
    recovers(problem, ice, 16, bkd, initial_guess=_START)


def test_recovers_2d(numpy_bkd: Backend[Array]) -> None:
    """H = 2 + x/4 - y/4 + x y/8 on [0, 1] x [-1, 1], bed 0.1 x + 0.05 y,
    friction 2, Dirichlet, Robin, Dirichlet, Neumann sides: the flux has
    degree 7 per direction, so 10 nodes per direction reproduce H to
    1e-10. (Friction 1 + x leaves 2.3e-9 at 10 nodes: its sliding term is
    not polynomial.) Newton stops at a residual of 1e-10: the default
    1e-12 sits at the round-off floor, which on some BLAS builds is
    4.6e-11 here."""
    ice = _ShallowIce("0.1*x + 0.05*y", "2")
    problem = ice.problem("2 + x/4 - y/4 + x*y/8", _RECTANGLE, "DRDN")
    newton: Newton[Array, Array] = Newton(atol=1e-10, rtol=1e-12, maxiters=30)
    recovers(problem, ice, 10, numpy_bkd, initial_guess=_START, nonlinear=newton)


def test_decays(numpy_bkd: Backend[Array]) -> None:
    """H = 2 + 0.3 sin(pi x) on [0, 1], bed 0.1 x, friction 1 + x,
    Dirichlet left, Neumann right: exponential decay over 6-14 nodes
    (observed 0.835 per node, errors 3.7e-2 to 5.7e-5, required 0.4)."""
    ice = _ShallowIce("0.1*x", "1 + x")
    problem = ice.problem("2 + 0.3*sin(pi*x)", _UNIT, "DN")
    decays(problem, ice, [6, 8, 10, 12, 14], 0.4, numpy_bkd, initial_guess=_START)


def test_bedless_surface_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: the physics is given a flat bed, so it takes
    the surface to be the thickness, while the problem's bed slopes; the
    error stops falling (fitted algebraic rate -0.69 over 6-12 nodes,
    errors 3e-2 to 6e-2), so the cases above check the bed's part in the
    surface gradient."""
    ice = _ShallowIce("0.1*x", "1 + x", physics_bed="0")
    problem = ice.problem("2 + 0.3*sin(pi*x)", _UNIT, "DN")
    stalls(problem, ice, [6, 8, 10, 12], numpy_bkd, initial_guess=_START)
