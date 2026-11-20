r"""Collocation shallow shelf velocity verified by manufactured solutions.

The steady momentum equations, for the velocity :math:`(u, v)` with the
thickness :math:`H` given, are, row by row,

.. math:: \nabla\cdot q_u + C u + H \rho g\, s_x = f_u, \qquad
   \nabla\cdot q_v + C v + H \rho g\, s_y = f_v,

with :math:`q_u = -(\tau_{xx}, \tau_{xy})`, :math:`q_v = -(\tau_{xy},
\tau_{yy})`, the membrane stress :math:`\tau = 2\mu H a`, :math:`a_{xx} =
2u_x + v_y`, :math:`a_{yy} = u_x + 2v_y`, :math:`a_{xy} = (u_y + v_x)/2`,
the viscosity :math:`\mu = \tfrac12 A^{-1/3} (Q + \epsilon)^{-1/3}`,
:math:`Q = u_x^2 + v_y^2 + u_x v_y + (u_y + v_x)^2/4`, :math:`\epsilon =
10^{-12}`, and the surface :math:`s = H + b`; written out here from
:math:`H`, :math:`b`, :math:`C` and the constants (:math:`A = \rho = g =
1`), not taken from the physics. Neumann and Robin sides prescribe
:math:`q\cdot n` through the injected :class:`ShallowShelfStressFlux`.

At zero velocity the strain vanishes and :math:`\mu` is
:math:`\tfrac12\epsilon^{-1/3}`, so solves start from the linear shear
:math:`u = x + y/2`, :math:`v = x/2 + y` (:math:`Q = 13/4`), an injected
initial guess.

Cases:

- recovery (L2 error below 1e-10) of linear velocities, whose strain
  and so viscosity are constant, on [0, 1] x [-1, 1] with a variable
  thickness, flat and sloped beds and two frictions, under Dirichlet,
  Neumann and Robin sides, on both backends;
- spectral decay, fitted per node, of the old residual tests' quadratic
  velocity on [-1, 1]^2;
- wrong-model control: each component's flux sides given the other
  component's flux stall.
"""

from typing import List, Sequence

import pytest
import sympy as sp

from pyapprox.pde.collocation.physics.shallow_shelf import (
    ShallowShelfStressFlux,
    ShallowShelfVelocityPhysics,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevLevel,
    collocation_system,
    field,
    stacked_forcing,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, Source, TermProtocol
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.collocation_verification import coded, decays, recovers, stalls

_A = 1.0
_RHO = 1.0
_G = 1.0


class _LinearShear:
    """Starts every level from u = x + y/2, v = x/2 + y."""

    def guess(
        self,
        problem: ManufacturedProblem,
        level: ChebyshevLevel[Array],
        system: CollocationSystem[Array],
    ) -> Array:
        bkd = level.bkd()
        x, y = level.mesh().points()[0], level.mesh().points()[1]
        return bkd.reshape(bkd.hstack([x + 0.5 * y, 0.5 * x + y]), (-1, 1))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


_START = _LinearShear()


class _ShallowShelf:
    """The physics under test and its strong form.

    Parameters
    ----------
    depth, bed, friction : str
        :math:`H(x, y)`, :math:`b(x, y)`, :math:`C(x, y)`.
    swapped : bool
        Close each component's flux sides with the other component's
        flux (a control).
    """

    def __init__(
        self, depth: str, bed: str, friction: str, swapped: bool = False
    ) -> None:
        self._depth, self._bed, self._friction = depth, bed, friction
        self._swapped = swapped

    def terms(self) -> List[TermProtocol]:
        """The strong form as terms on rows ``u`` and ``v``."""
        surface = sp.sympify(f"({self._depth}) + ({self._bed})")
        s_x = sp.diff(surface, sp.Symbol("x"))
        s_y = sp.diff(surface, sp.Symbol("y"))
        visc = 0.5 * _A ** (-1.0 / 3)
        strain = "u_x**2 + v_y**2 + u_x*v_y + (u_y + v_x)**2/4"
        coef = f"2*{visc!r}*({strain} + 1e-12)**(-1/3)*({self._depth})"
        tau_xx = f"{coef}*(2*u_x + v_y)"
        tau_yy = f"{coef}*(u_x + 2*v_y)"
        tau_xy = f"{coef}*(u_y + v_x)/2"
        drive = f"({self._depth})*{_RHO * _G!r}"
        return [
            Divergence("u", [f"-({tau_xx})", f"-({tau_xy})"]),
            Source("u", f"({self._friction})*u + {drive}*({s_x})"),
            Divergence("v", [f"-({tau_xy})", f"-({tau_yy})"]),
            Source("v", f"({self._friction})*v + {drive}*({s_y})"),
        ]

    def problem(
        self, solution: Sequence[str], box: Box, codes: str
    ) -> ManufacturedProblem:
        conditions = coded(codes, box)
        return ManufacturedProblem(
            {"u": solution[0], "v": solution[1]},
            self.terms(),
            box,
            {part: {"u": c, "v": c} for part, c in conditions.items()},
        )

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = ShallowShelfVelocityPhysics(
            level.basis(),
            level.bkd(),
            depth=field(problem, self._depth, level),
            bed=field(problem, self._bed, level),
            friction=field(problem, self._friction, level),
            A=_A,
            rho=_RHO,
            g=_G,
            forcing=stacked_forcing(problem, level),
        )
        u_flux = ShallowShelfStressFlux(physics, 0)
        v_flux = ShallowShelfStressFlux(physics, 1)
        if self._swapped:
            u_flux, v_flux = v_flux, u_flux
        return collocation_system(physics, problem, level, {"u": u_flux, "v": v_flux})

    def __repr__(self) -> str:
        return (
            f"_ShallowShelf({self._depth!r}, {self._bed!r}, {self._friction!r}, "
            f"swapped={self._swapped})"
        )


_RECTANGLE = Box([(0.0, 1.0), (-1.0, 1.0)])
_SQUARE = Box([(-1.0, 1.0), (-1.0, 1.0)])


@pytest.mark.parametrize(
    "bed, friction",
    [("0", "1"), ("0.1*x - 0.05*y", "1"), ("0.1*x - 0.05*y", "10")],
    ids=["flat", "sloped", "higher_friction"],
)
@pytest.mark.parametrize("codes", ["DDDD", "DNDN", "RDNR", "NNNN"])
def test_recovers_linear_velocity(
    bkd: Backend[Array], codes: str, bed: str, friction: str
) -> None:
    """u = 1 + x - y/2, v = x/2 + y - 1 on [0, 1] x [-1, 1], H = 1.5 +
    x/4 - y/8: the strain is constant, so the stress is linear and the
    forcing polynomial, and 6 x 6 nodes reproduce the velocity to 1e-10
    under every pairing of sides (pure Neumann is closed by the
    friction)."""
    shelf = _ShallowShelf("1.5 + x/4 - y/8", bed, friction)
    problem = shelf.problem(["1 + x - y/2", "x/2 + y - 1"], _RECTANGLE, codes)
    recovers(problem, shelf, 6, bkd, initial_guess=_START)


@pytest.mark.parametrize("codes", ["DDDD", "DNDR"])
def test_decays(numpy_bkd: Backend[Array], codes: str) -> None:
    """The old residual tests' u = x^2 + y, v = x + y^2 on [-1, 1]^2,
    H = 1, bed 0.1 x, friction 1 (Q = 4(x^2 + x y + y^2) + 1 >= 1, so the
    viscosity is analytic): exponential decay over 6-14 nodes per
    direction (observed 0.746 per node Dirichlet, errors 4.0e-2 to
    1.0e-4; 0.735 mixed, 4.2e-2 to 1.2e-4; required 0.35). The old tests
    needed 40 nodes for a residual below 1e-6."""
    shelf = _ShallowShelf("1", "0.1*x", "1")
    problem = shelf.problem(["x**2 + y", "x + y**2"], _SQUARE, codes)
    decays(problem, shelf, [6, 8, 10, 12, 14], 0.35, numpy_bkd, initial_guess=_START)


def test_swapped_fluxes_stall(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: Dirichlet left and bottom, Neumann right and
    top, and each component's Neumann sides prescribe the other
    component's flux; the error stops falling (errors 2.2 to 1.7 over
    6-12 nodes), so the flux cases above check which stress row each side
    carries. With a Robin side instead (DNDR) the wrong model's Newton
    does not converge at most sizes."""
    shelf = _ShallowShelf("1", "0.1*x", "1", swapped=True)
    problem = shelf.problem(["x**2 + y", "x + y**2"], _SQUARE, "DNDN")
    stalls(problem, shelf, [6, 8, 10, 12], numpy_bkd, initial_guess=_START)
