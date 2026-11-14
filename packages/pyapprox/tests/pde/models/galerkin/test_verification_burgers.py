r"""Galerkin viscous Burgers verified by manufactured solutions.

:class:`~pyapprox.pde.galerkin.physics.BurgersPhysics` (1D) assembles
the advective form :math:`u u_x v` without integrating it by parts, so
its natural data carry only the viscous flux :math:`-\nu u_x`. The
problem states that weak form's split as terms:
:math:`(-\nu u_x)_x + u u_x - f = 0`. Errors are true :math:`L^2` errors;
solves start from zero.

Cases:

- P2 exact recovery (L2 below 1e-10) of the old suite's two cases,
  :math:`u = x(1 - x)`, :math:`\nu = 10`, 64 cells, Dirichlet ends and
  Neumann-Robin ends;
- rates (fitted over the last three levels, within 0.1) with variable
  :math:`\nu` and Dirichlet-Neumann ends: P1 2, P2 3;
- the classical viscous shock :math:`u = -a\tanh(a x/(2\nu))`, a steady
  solution with zero forcing, :math:`a = 1`, :math:`\nu = 0.1` on
  :math:`[-1, 1]`: P1 2, P2 3;
- wrong-model control: the problem stated in conservative form, so its
  Neumann data include the advective flux :math:`u^2/2` that this weak
  form never integrates by parts; the error stalls (fitted rate below
  0.5).
"""

from typing import Any, List

import pytest
from numpy.typing import NDArray

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import BurgersPhysics
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.verification import (
    StructuredBox,
    boundary_conditions,
    field,
    forcing,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, Source, TermProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.galerkin_verification import coded, converges, recovers, stalls

_Np = NDArray[Any]


class _Burgers:
    """The physics under test and its strong form, from one viscosity.

    Parameters
    ----------
    viscosity : str
        :math:`\\nu(x)`.
    conservative : bool
        State the problem as :math:`(u^2/2 - \\nu u_x)_x`, whose natural
        data include the advective flux (the wrong-model control).
    """

    def __init__(self, viscosity: str, conservative: bool = False) -> None:
        self._viscosity, self._conservative = viscosity, conservative

    def terms(self) -> List[TermProtocol]:
        if self._conservative:
            return [Divergence("u", [f"u**2/2 - ({self._viscosity})*u_x"])]
        return [
            Divergence("u", [f"-({self._viscosity})*u_x"]),
            Source("u", "u*u_x"),
        ]

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        physics = BurgersPhysics(
            basis,
            field(problem, self._viscosity),
            NumpyBkd(),
            forcing=forcing(problem, "u"),
        )
        return compose_galerkin_system(
            physics, boundary_conditions(problem, basis, "u")
        )

    def __repr__(self) -> str:
        return f"_Burgers({self._viscosity!r}, conservative={self._conservative})"


def _problem(
    solution: str,
    burgers: _Burgers,
    codes: str,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> ManufacturedProblem:
    box = Box([bounds])
    return ManufacturedProblem({"u": solution}, burgers.terms(), box, coded(codes, box))


@pytest.mark.parametrize("codes", ["DD", "NR"])
def test_p2_recovers(codes: str) -> None:
    """The old suite's two cases: u = x(1 - x) is in the P2 space, so P2
    on 64 cells reproduces it to 1e-10 with Dirichlet ends, and with a
    Neumann and a Robin end."""
    burgers = _Burgers("10")
    recovers(_problem("x*(1 - x)", burgers, codes), burgers, 64)


def test_rates() -> None:
    """u = sin(pi x) + 2x, nu = 1 + x/2, Dirichlet left, Neumann right:
    L2 rate 2 for P1 (cells 8-64) and 3 for P2 (4-32)."""
    burgers = _Burgers("1 + x/2")
    problem = _problem("sin(pi*x) + 2*x", burgers, "DN")
    converges(problem, StructuredBox([8, 16, 32, 64]), 1, burgers, 2.0)
    converges(problem, StructuredBox([4, 8, 16, 32]), 2, burgers, 3.0)


def test_viscous_shock() -> None:
    """The classical steady shock -tanh(x/(2 nu)), nu = 0.1, on [-1, 1]
    with Dirichlet ends solves Burgers with zero forcing; from a zero
    initial guess Newton finds it, at L2 rate 2 for P1 (cells 16-128)
    and 3 for P2 (8-64)."""
    burgers = _Burgers("0.1")
    problem = _problem("-tanh(x/(2*0.1))", burgers, "DD", (-1.0, 1.0))
    converges(problem, StructuredBox([16, 32, 64, 128]), 1, burgers, 2.0)
    converges(problem, StructuredBox([8, 16, 32, 64]), 2, burgers, 3.0)


def test_conservative_natural_data_stall() -> None:
    """Wrong-model control: the same strong form written conservatively
    puts u^2/2 into the Neumann data, but this weak form integrates only
    the viscous flux by parts; the error stops falling, so the rate tests
    check which flux the natural data carry."""
    burgers = _Burgers("1", conservative=True)
    problem = _problem("sin(pi*x) + 2*x", burgers, "DN")
    stalls(problem, StructuredBox([8, 16, 32, 64]), 1, burgers)
