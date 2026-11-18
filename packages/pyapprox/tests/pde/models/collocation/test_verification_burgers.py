r"""Collocation viscous Burgers verified by manufactured solutions.

The steady equation is :math:`(u^2/2 - \nu u_x)_x = f`. Which flux a
Neumann or Robin end prescribes is a choice of the problem, not of the
equation: the problem states either the total flux
:math:`u^2/2 - \nu u_x` as one divergence, or only the viscous flux
:math:`-\nu u_x` with :math:`u u_x` as a source. The system injects the
matching flux, :class:`BurgersTotalFlux` or :class:`BurgersViscousFlux`,
into the boundary rows; the interior uses either discrete form of
:class:`~pyapprox.pde.collocation.physics.burgers.BurgersPhysics1D`.
Solves start from zero.

Cases:

- recovery (L2 error below 1e-10) of polynomials vanishing at both ends
  of [-1, 1] under zero Dirichlet ends, for viscosities 0.1 to 2, in
  both discrete forms;
- recovery of a cubic on [0, 1] with variable viscosity under Neumann
  and Robin ends, for both boundary fluxes and both forms;
- spectral decay, fitted per node, of an analytic solution under mixed
  ends;
- the steady viscous shock :math:`u = -\tanh(x/(2\nu))`, :math:`\nu =
  0.1`, zero forcing, on [-1, 1]: its poles at :math:`x = \pm i\pi\nu`
  set the decay :math:`\ln(\pi\nu + \sqrt{1 + (\pi\nu)^2}) \approx 0.31`
  per node, and the fit must reach 0.9 of it;
- wrong-model control: a problem stating the total flux, closed with the
  viscous flux at a Neumann end, stalls.
"""

import math
from typing import List

import pytest

from pyapprox.pde.collocation.physics.burgers import (
    BurgersPhysics1D,
    BurgersTotalFlux,
    BurgersViscousFlux,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevLevel,
    collocation_system,
    field,
    forcing,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, Source, TermProtocol
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.collocation_verification import coded, decays, recovers, stalls


class _Burgers:
    """The physics under test and its strong form.

    Parameters
    ----------
    viscosity : str
        :math:`\\nu(x)`.
    total : bool
        Whether the problem states the total flux (else the viscous
        flux); its Neumann and Robin data follow.
    conservative : bool
        The interior's discrete form.
    total_at_boundary : bool, optional
        Whether the boundary rows use the total flux; default ``total``.
        A control passes the other.
    """

    def __init__(
        self,
        viscosity: str,
        total: bool,
        conservative: bool,
        total_at_boundary: bool | None = None,
    ) -> None:
        self._viscosity, self._total = viscosity, total
        self._conservative = conservative
        self._total_at_boundary = (
            total if total_at_boundary is None else total_at_boundary
        )

    def terms(self) -> List[TermProtocol]:
        """The strong form as terms on row ``u``."""
        if self._total:
            return [Divergence("u", [f"u**2/2 - ({self._viscosity})*u_x"])]
        return [
            Divergence("u", [f"-({self._viscosity})*u_x"]),
            Source("u", "u*u_x"),
        ]

    def problem(self, solution: str, box: Box, codes: str) -> ManufacturedProblem:
        conditions = coded(codes, box)
        return ManufacturedProblem({"u": solution}, self.terms(), box, conditions)

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = BurgersPhysics1D(
            level.basis(),
            level.bkd(),
            viscosity=field(problem, self._viscosity, level),
            forcing=forcing(problem, "u", level),
            conservative=self._conservative,
        )
        flux = (
            BurgersTotalFlux(physics)
            if self._total_at_boundary
            else BurgersViscousFlux(physics)
        )
        return collocation_system(physics, problem, level, "u", flux=flux)

    def __repr__(self) -> str:
        return (
            f"_Burgers({self._viscosity!r}, total={self._total}, "
            f"conservative={self._conservative}, "
            f"total_at_boundary={self._total_at_boundary})"
        )


_REFERENCE = Box([(-1.0, 1.0)])
_UNIT = Box([(0.0, 1.0)])
_FORMS = pytest.mark.parametrize(
    "conservative", [True, False], ids=["conservative", "advective"]
)


@_FORMS
@pytest.mark.parametrize(
    "solution, viscosity, npts",
    [
        ("1 - x**2", "0.1", 15),
        ("1 - x**2", "0.5", 15),
        ("1 - x**2", "2", 15),
        ("(1 - x**2)**2", "0.1", 18),
        ("x*(1 - x**2)", "0.1", 15),
        ("(1 - x**2)**3", "0.1", 20),
    ],
)
def test_recovers_dirichlet_polynomials(
    numpy_bkd: Backend[Array],
    solution: str,
    viscosity: str,
    npts: int,
    conservative: bool,
) -> None:
    """Polynomials vanishing at both ends of [-1, 1], zero Dirichlet
    ends: reproduced to 1e-10."""
    burgers = _Burgers(viscosity, total=True, conservative=conservative)
    recovers(burgers.problem(solution, _REFERENCE, "DD"), burgers, npts, numpy_bkd)


@_FORMS
@pytest.mark.parametrize("total", [True, False], ids=["total", "viscous"])
@pytest.mark.parametrize("codes", ["DN", "ND", "RD", "NR"])
def test_recovers_cubic(
    bkd: Backend[Array], codes: str, total: bool, conservative: bool
) -> None:
    """u = 1 + x - x^2/2 + x^3 on [0, 1], nu = 1 + x/2, an end with a
    flux condition: 8 nodes reproduce it to 1e-10 whichever flux the
    problem states, in both discrete forms, on both backends."""
    burgers = _Burgers("1 + x/2", total=total, conservative=conservative)
    problem = burgers.problem("1 + x - x**2/2 + x**3", _UNIT, codes)
    recovers(problem, burgers, 8, bkd)


@_FORMS
@pytest.mark.parametrize("total", [True, False], ids=["total", "viscous"])
def test_decays(numpy_bkd: Backend[Array], total: bool, conservative: bool) -> None:
    """u = sin(pi x) + 2 x on [0, 1], nu = 1 + x/2, Dirichlet left,
    Neumann right: exponential decay over 4-12 nodes (observed 1.35 to
    2.07 per node across the four combinations, required 0.6)."""
    burgers = _Burgers("1 + x/2", total=total, conservative=conservative)
    problem = burgers.problem("sin(pi*x) + 2*x", _UNIT, "DN")
    decays(problem, burgers, [4, 6, 8, 10, 12], 0.6, numpy_bkd)


@_FORMS
def test_viscous_shock(numpy_bkd: Backend[Array], conservative: bool) -> None:
    """The steady shock -tanh(x/(2 nu)), nu = 0.1, zero forcing, Dirichlet
    ends on [-1, 1]; Newton finds it from zero. Its poles at
    +-i pi nu set the decay ln r, r = pi nu + sqrt(1 + (pi nu)^2), 0.309
    per node; the fit over 20-50 nodes (2.8e-3 to 7.4e-8) must reach 0.9
    of the prediction. At 60 nodes the conservative form's Newton
    residual stalls at 5e-12, round-off against the 1e-12 tolerance."""
    nu = 0.1
    predicted = math.log(math.pi * nu + math.sqrt(1.0 + (math.pi * nu) ** 2))
    burgers = _Burgers(str(nu), total=True, conservative=conservative)
    problem = burgers.problem(f"-tanh(x/(2*{nu}))", _REFERENCE, "DD")
    decays(problem, burgers, [20, 30, 40, 50], 0.9 * predicted, numpy_bkd)


def test_viscous_flux_on_total_problem_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: the problem states the total flux, so its
    Neumann data include u^2/2, but the boundary rows prescribe only the
    viscous flux; the error stops falling, so the flux cases above check
    which flux the ends carry."""
    burgers = _Burgers("1", total=True, conservative=True, total_at_boundary=False)
    problem = burgers.problem("sin(pi*x) + 2*x", _UNIT, "DN")
    stalls(problem, burgers, [6, 8, 10, 12], numpy_bkd)
