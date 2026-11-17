r"""Galerkin Helmholtz verified by manufactured solutions.

:class:`~pyapprox.pde.galerkin.physics.Helmholtz` solves the screened
(modified) equation :math:`-\nabla^2 u + k^2 u = f`, positive definite,
not the wave equation :math:`\nabla^2 u + k^2 u = 0`. The problem states
that strong form as terms, :math:`\nabla\cdot(-\nabla u) + k^2 u`, so no
sign workaround is needed; a control with the wave-equation sign shows
the convention is checked. Errors are true :math:`L^2` errors.

Cases:

- P2 exact recovery (L2 below 1e-10) of the old suite's three cases:
  :math:`u = x`, :math:`k^2 = 4`, Dirichlet ends; :math:`u = x^2`,
  :math:`k^2 = x`, Neumann and Robin ends; :math:`u = x^2 y^2` on
  :math:`[0, 1/2]\times[0, 1]`, :math:`k^2 = 1`, sides D, D, R, N;
- rates (fitted over the last three levels, within 0.1) with variable
  :math:`k^2` and Dirichlet, Neumann and Robin parts: P1 and P2 in 1D,
  on quads and on triangles, P1 on hexes and on the disk;
- wrong-model control: the problem posed with the wave-equation sign
  stalls (fitted rate below 0.5).
"""

import math
from typing import Any, Callable, List, Sequence, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from numpy.typing import NDArray

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import Helmholtz
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.verification import (
    SkfemDisk,
    StructuredBox,
    boundary_conditions,
    field,
    forcing,
)
from pyapprox.pde.verification.conditions import Robin
from pyapprox.pde.verification.domains import Box, Disk, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, Source, TermProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.galerkin_verification import coded, converges, recovers, stalls

_Np = NDArray[Any]


class _SpatialOnly:
    """:math:`k^2(x)` as ``Helmholtz`` calls it, with coordinates alone:
    its wavenumber is not time threaded, unlike its forcing."""

    def __init__(self, ksq: Callable[[_Np, float], _Np]) -> None:
        self._ksq = ksq

    def __call__(self, coords: _Np) -> _Np:
        values: _Np = self._ksq(coords, 0.0)
        return values


class _Helmholtz:
    """The physics under test and its strong form, from one :math:`k^2`.

    Parameters
    ----------
    ksq : str
        :math:`k^2(x)`, non-negative.
    wave_sign : bool
        State the problem with the wave-equation sign
        :math:`-\\nabla^2 u - k^2 u` while the physics stays screened (the
        wrong-model control).
    """

    def __init__(self, ksq: str, wave_sign: bool = False) -> None:
        self._ksq, self._wave_sign = ksq, wave_sign

    def terms(self, coords: Sequence[str]) -> List[TermProtocol]:
        sign = "-" if self._wave_sign else ""
        return [
            Divergence("u", [f"-u_{c}" for c in coords]),
            Source("u", f"{sign}({self._ksq})*u"),
        ]

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        ksq = field(problem, self._ksq)
        # A float is the wavenumber k; a callable is k^2(x).
        wavenumber = math.sqrt(ksq) if isinstance(ksq, float) else _SpatialOnly(ksq)
        physics = Helmholtz(
            basis, wavenumber, NumpyBkd(), forcing=forcing(problem, "u")
        )
        return compose_galerkin_system(
            physics, boundary_conditions(problem, basis, "u")
        )

    def __repr__(self) -> str:
        return f"_Helmholtz({self._ksq!r}, wave_sign={self._wave_sign})"


def _problem(
    solution: str, helmholtz: _Helmholtz, domain: DomainProtocol, codes: str
) -> ManufacturedProblem:
    return ManufacturedProblem(
        {"u": solution},
        helmholtz.terms(domain.coordinates()),
        domain,
        coded(codes, domain),
    )


@pytest.mark.parametrize(
    "solution, ksq, bounds, codes",
    [
        ("x", "4", [(0.0, 1.0)], "DD"),
        ("x**2", "x", [(0.0, 1.0)], "NR"),
        ("x**2*y**2", "1", [(0.0, 0.5), (0.0, 1.0)], "DDRN"),
    ],
)
def test_p2_recovers(
    solution: str, ksq: str, bounds: List[Tuple[float, float]], codes: str
) -> None:
    """The old suite's three cases, solutions in the P2 (Q2) space: P2 on
    10 cells (1D) or 5x5 quads reproduces them to 1e-10, including a
    k^2 varying in x with Neumann and Robin ends."""
    helmholtz = _Helmholtz(ksq)
    box = Box(bounds)
    recovers(
        _problem(solution, helmholtz, box, codes),
        helmholtz,
        10 if len(bounds) == 1 else 5,
    )


class TestRates:
    def test_1d(self) -> None:
        """k^2 = 4 + x, Dirichlet left, Neumann right: L2 rate 2 for P1
        (cells 8-64) and 3 for P2 (4-32)."""
        helmholtz = _Helmholtz("4 + x")
        problem = _problem("sin(3*x) + x", helmholtz, Box([(0.0, 1.0)]), "DN")
        converges(problem, StructuredBox([8, 16, 32, 64]), 1, helmholtz, 2.0)
        converges(problem, StructuredBox([4, 8, 16, 32]), 2, helmholtz, 3.0)

    @pytest.mark.parametrize("simplex", [False, True])
    def test_2d(self, simplex: bool) -> None:
        """Unit square, k^2 = 1 + x y, sides D, N, R, D: L2 rate 2 for P1
        (4-32 cells per side) and 3 for P2 (2-16), on quads and on
        triangles."""
        helmholtz = _Helmholtz("1 + x*y")
        box = Box([(0.0, 1.0)] * 2)
        problem = _problem("exp(x)*cos(2*y)", helmholtz, box, "DNRD")
        converges(problem, StructuredBox([4, 8, 16, 32], simplex), 1, helmholtz, 2.0)
        converges(problem, StructuredBox([2, 4, 8, 16], simplex), 2, helmholtz, 3.0)

    def test_3d(self) -> None:
        """Unit cube, hexes 2-8, k^2 = 2 + z, all three kinds: P1 rate 2."""
        helmholtz = _Helmholtz("2 + z")
        box = Box([(0.0, 1.0)] * 3)
        problem = _problem("sin(x + y)*exp(z)", helmholtz, box, "DNRDDN")
        converges(problem, StructuredBox([2, 4, 8]), 1, helmholtz, 2.0)

    def test_disk(self) -> None:
        """Unit disk, Robin on the curved boundary, k^2 = 1 + x^2: P1
        rate 2 (refinements 2-5)."""
        helmholtz = _Helmholtz("1 + x**2")
        problem = ManufacturedProblem(
            {"u": "cos(x)*exp(y)"},
            helmholtz.terms(("x", "y")),
            Disk(),
            {"circle": Robin(2.0)},
        )
        converges(problem, SkfemDisk([2, 3, 4, 5]), 1, helmholtz, 2.0)


def test_wave_sign_stalls() -> None:
    """Wrong-model control: the problem's forcing built for
    -lap u - k^2 u (the wave-equation sign) while the physics solves the
    screened form; the error stops falling, so the rate tests check the
    sign of the k^2 term."""
    helmholtz = _Helmholtz("4", wave_sign=True)
    problem = _problem("sin(3*x) + x", helmholtz, Box([(0.0, 1.0)]), "DN")
    stalls(problem, StructuredBox([8, 16, 32, 64]), 1, helmholtz)
