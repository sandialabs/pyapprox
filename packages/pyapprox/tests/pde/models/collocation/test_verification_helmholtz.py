r"""Collocation Helmholtz verified by manufactured solutions.

The problem is stated as terms in the screened sign the physics solves,
:math:`\nabla\cdot(-\nabla u) + k^2 u = f`, and the physics under test,
:class:`~pyapprox.pde.collocation.physics.helmholtz.HelmholtzPhysics`, is
built from the same :math:`k^2`. A negative :math:`k^2` is the standard
(unscreened) Helmholtz equation :math:`-\Delta u - |k^2| u = f`.

Cases:

- recovery (L2 error below 1e-10) of polynomials vanishing on the
  boundary, with zero Dirichlet sides: on [-1, 1], (1 - x^2) at
  k^2 = 1, -1, -2 and -5 (-5 lies above the first Dirichlet eigenvalue
  pi^2/4, an indefinite operator), (1 - x^2)^2 and x (1 - x^2); on
  [-1, 1]^2, their products with (1 - y^2);
- recovery of cubics under Neumann and Robin sides, 1D and 2D, both
  backends in 2D;
- spectral decay, fitted per node, of analytic solutions in 1D and 2D
  under mixed sides, and on an annulus;
- wrong-model control: the physics built with -k^2, the standard sign,
  for a screened problem stalls.
"""

from typing import List, Sequence

import pytest

from pyapprox.pde.collocation.physics.helmholtz import HelmholtzPhysics
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevAnnulus,
    ChebyshevLevel,
    collocation_system,
    forcing,
)
from pyapprox.pde.verification.domains import Annulus, Box, DomainProtocol
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, Source, TermProtocol
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.collocation_verification import (
    coded,
    decays,
    recovers,
    ring_newton,
    stalls,
)


class _Helmholtz:
    """The physics under test and its strong form.

    Parameters
    ----------
    wave_number_sq : float
        :math:`k^2` of the strong form.
    physics_wave_number_sq : float, optional
        :math:`k^2` given to the physics; default the same. A control
        passes another value.
    """

    def __init__(
        self, wave_number_sq: float, physics_wave_number_sq: float | None = None
    ) -> None:
        self._k2 = wave_number_sq
        self._physics_k2 = (
            wave_number_sq if physics_wave_number_sq is None else physics_wave_number_sq
        )

    def terms(self, coords: Sequence[str]) -> List[TermProtocol]:
        """The screened strong form as terms on row ``u``."""
        return [
            Divergence("u", [f"-u_{c}" for c in coords]),
            Source("u", f"({self._k2})*u"),
        ]

    def problem(
        self, solution: str, domain: DomainProtocol, codes: str
    ) -> ManufacturedProblem:
        return ManufacturedProblem(
            {"u": solution},
            self.terms(domain.coordinates()),
            domain,
            coded(codes, domain),
        )

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = HelmholtzPhysics(
            level.basis(),
            level.bkd(),
            wave_number_sq=self._physics_k2,
            forcing=forcing(problem, "u", level),
        )
        return collocation_system(physics, problem, level, {"u": physics})

    def __repr__(self) -> str:
        return f"_Helmholtz({self._k2}, physics={self._physics_k2})"


_REFERENCE_1D = Box([(-1.0, 1.0)])
_REFERENCE_2D = Box([(-1.0, 1.0)] * 2)
_UNIT = Box([(0.0, 1.0)])
_RECTANGLE = Box([(0.0, 1.0), (-1.0, 1.0)])
_RING = Annulus((0.0, 0.0), 1.0, 2.0)


@pytest.mark.parametrize(
    "solution, k2, npts",
    [
        ("1 - x**2", 1.0, 20),
        ("1 - x**2", -1.0, 10),
        ("1 - x**2", -2.0, 12),
        ("1 - x**2", -5.0, 12),
        ("(1 - x**2)**2", -2.0, 12),
        ("x*(1 - x**2)", -0.5, 10),
    ],
)
def test_recovers_dirichlet_polynomials_1d(
    numpy_bkd: Backend[Array], solution: str, k2: float, npts: int
) -> None:
    """Polynomials vanishing at both ends of [-1, 1], screened and
    standard sign, zero Dirichlet ends: reproduced to 1e-10."""
    helmholtz = _Helmholtz(k2)
    problem = helmholtz.problem(solution, _REFERENCE_1D, "DD")
    recovers(problem, helmholtz, npts, numpy_bkd)


@pytest.mark.parametrize(
    "solution, k2, npts",
    [
        ("(1 - x**2)*(1 - y**2)", -1.0, 10),
        ("(1 - x**2)**2*(1 - y**2)", -2.0, 10),
        ("(1 - x**2)*(1 - y**2)*x", -1.5, 10),
    ],
)
def test_recovers_dirichlet_polynomials_2d(
    numpy_bkd: Backend[Array], solution: str, k2: float, npts: int
) -> None:
    """Polynomials vanishing on the boundary of [-1, 1]^2, standard sign,
    zero Dirichlet sides: reproduced to 1e-10."""
    helmholtz = _Helmholtz(k2)
    problem = helmholtz.problem(solution, _REFERENCE_2D, "DDDD")
    recovers(problem, helmholtz, npts, numpy_bkd)


@pytest.mark.parametrize("k2", [2.0, -1.0])
@pytest.mark.parametrize("codes", ["DN", "ND", "RD", "RR", "NR"])
def test_recovers_cubic_1d(numpy_bkd: Backend[Array], codes: str, k2: float) -> None:
    """u = 1 + x - x^2/2 + x^3 on [0, 1], screened and unscreened, every
    pairing with a flux end: 8 nodes reproduce it to 1e-10."""
    helmholtz = _Helmholtz(k2)
    problem = helmholtz.problem("1 + x - x**2/2 + x**3", _UNIT, codes)
    recovers(problem, helmholtz, 8, numpy_bkd)


@pytest.mark.parametrize("codes", ["DRDN", "NRDR", "RRRR"])
def test_recovers_cubic_2d(bkd: Backend[Array], codes: str) -> None:
    """u = x y + y^2 - x^2 y on [0, 1] x [-1, 1], k^2 = 2: 6 x 6 nodes
    reproduce it to 1e-10, on both backends."""
    helmholtz = _Helmholtz(2.0)
    problem = helmholtz.problem("x*y + y**2 - x**2*y", _RECTANGLE, codes)
    recovers(problem, helmholtz, 6, bkd)


def test_decays_1d(numpy_bkd: Backend[Array]) -> None:
    """u = exp(x) sin(3 x) on [0, 1], Dirichlet left and Neumann right,
    k^2 = -4: the standard sign, between the first two eigenvalues
    (pi/2)^2 and (3 pi/2)^2 of -u'' with these ends, so indefinite but
    nonsingular. Exponential decay over 4-12 nodes (0.73 to 1.5e-8;
    observed 2.07 per node, required 1)."""
    helmholtz = _Helmholtz(-4.0)
    problem = helmholtz.problem("exp(x)*sin(3*x)", _UNIT, "DN")
    decays(problem, helmholtz, [4, 6, 8, 10, 12], 1.0, numpy_bkd)


def test_decays_2d(numpy_bkd: Backend[Array]) -> None:
    """u = exp(x/2) sin(2 y) + x y on [0, 1] x [-1, 1], k^2 = 3,
    Dirichlet, Robin and Neumann sides: exponential decay over 4-12
    nodes per direction (0.14 to 3.5e-9; observed 2.18 per node,
    required 1)."""
    helmholtz = _Helmholtz(3.0)
    problem = helmholtz.problem("exp(x/2)*sin(2*y) + x*y", _RECTANGLE, "DRDN")
    decays(problem, helmholtz, [4, 6, 8, 10, 12], 1.0, numpy_bkd)


@pytest.mark.parametrize("k2, codes", [(3.0, "DD"), (-2.0, "DR")])
def test_decays_annulus(numpy_bkd: Backend[Array], k2: float, codes: str) -> None:
    """u = exp(x/2) sin(y) on the annulus 1 <= r <= 2 meshed over the
    full angle, screened (k^2 = 3, both circles Dirichlet) and standard
    sign (k^2 = -2, Robin outer): errors 2.5e-1 to 9.6e-6 and 3.5e-1 to
    1.1e-5 over 8-24 nodes per direction (observed 0.64 and 0.65 per
    node, required 0.45)."""
    helmholtz = _Helmholtz(k2)
    problem = helmholtz.problem("exp(x/2)*sin(y)", _RING, codes)
    decays(
        problem,
        helmholtz,
        [8, 12, 16, 20, 24],
        0.45,
        numpy_bkd,
        nonlinear=ring_newton(),
        family=ChebyshevAnnulus,
    )


def test_wrong_sign_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: the physics given -k^2 (the standard sign)
    for a screened problem with k^2 = 2 solves another equation, so the
    error stops falling and the cases above can detect a sign error."""
    helmholtz = _Helmholtz(2.0, physics_wave_number_sq=-2.0)
    problem = helmholtz.problem("exp(x)*sin(3*x)", _UNIT, "DD")
    stalls(problem, helmholtz, [6, 8, 10, 12], numpy_bkd)
