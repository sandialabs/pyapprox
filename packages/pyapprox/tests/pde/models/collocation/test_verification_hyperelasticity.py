r"""Collocation neo-Hookean hyperelasticity verified by manufactured solutions.

:class:`~pyapprox.pde.collocation.physics.hyperelasticity.
HyperelasticityPhysics`, given a :class:`~pyapprox.pde.collocation.physics.
neo_hookean_law.NeoHookeanLaw`, solves :math:`-\nabla\cdot P(F) = f`, total
Lagrangian, with the first Piola stress

.. math:: P = \mu F + (\lambda \ln J - \mu) F^{-T}, \quad F = I + \nabla u,
   \quad J = \det F.

The problem states it row by row (fields :math:`u, v, w`):
:math:`\nabla\cdot(-P_{i\cdot}) = f_i`, with :math:`P` written here from the
formula above, not taken from the law, so the two are independent. Its
natural data are :math:`-P N`; traction and Robin sides close on the
physics' nominal traction through :class:`~pyapprox.pde.models.collocation.
verification.PiolaTractionTerm`, which takes that sign. Every solve starts
from zero.

Cases:

- recovery (L2 error below 1e-10) of affine displacements, whose
  deformation gradient and so stress are constant, under Dirichlet,
  traction and Robin sides in 1D, 2D and 3D, on both backends, and of
  the old 3D residual case's polynomial bubble;
- the interior residual at clamped 1D and 2D bubbles for Lamé pairs from
  soft to stiff;
- spectral decay, fitted per node, of the comparison suite's 1D and 2D
  displacements (clamped, all sides Dirichlet), and of the Galerkin
  catalog's 1D, 2D and 3D solutions with traction and Robin sides;
- wrong-model controls, which stall: the problem stated with the
  small-strain law, the physics given :math:`\lambda` and :math:`\mu`
  swapped, and traction sides given the traction with the wrong sign.
"""

from typing import List, Sequence

import pytest
import sympy as sp

from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevBox,
    ChebyshevLevel,
    PiolaTractionTerm,
    collocation_system,
    field,
    side_nodes,
    stacked_forcing,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, TermProtocol
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.collocation_verification import coded, decays, recovers, stalls
from tests._helpers.markers import slow_test

_FIELDS = ("u", "v", "w")
_COORDS = ("x", "y", "z")


def _gradient(ndim: int) -> sp.Matrix:
    """:math:`\\nabla u` in the problem's names: entry (i, j) is the
    derivative of field i in coordinate j."""
    return sp.Matrix(ndim, ndim, lambda i, j: sp.Symbol(f"{_FIELDS[i]}_{_COORDS[j]}"))


def _neo_hookean(ndim: int, lam: str, mu: str) -> sp.Matrix:
    """:math:`P = \\mu F + (\\lambda \\ln J - \\mu) F^{-T}`, with
    :math:`F^{-T} = \\mathrm{adj}(F)^T / J`."""
    deformation = sp.eye(ndim) + _gradient(ndim)
    jacobian = deformation.det()
    lam_value, mu_value = sp.sympify(lam), sp.sympify(mu)
    stress: sp.Matrix = (
        mu_value * deformation
        + (lam_value * sp.log(jacobian) - mu_value)
        * deformation.adjugate().T
        / jacobian
    )
    return stress


def _linear(ndim: int, lam: str, mu: str) -> sp.Matrix:
    """:math:`\\sigma = \\lambda\\,\\mathrm{tr}(\\varepsilon) I + 2\\mu
    \\varepsilon`, the small-strain law (a wrong-model control)."""
    gradient = _gradient(ndim)
    strain = (gradient + gradient.T) / 2
    stress: sp.Matrix = (
        sp.sympify(lam) * strain.trace() * sp.eye(ndim) + 2 * sp.sympify(mu) * strain
    )
    return stress


class _ReversedTraction(PiolaTractionTerm[Array]):
    """The traction with the wrong sign: :math:`t_i` read as the outward
    flux :math:`q_i\\cdot N` (a wrong-model control)."""

    def sign(self) -> float:
        return 1.0


class _Hyperelasticity:
    """The physics under test and its strong form, from one pair of Lamé
    values.

    Parameters
    ----------
    lam, mu : str
        :math:`\\lambda(x)` and :math:`\\mu(x)`.
    linear_problem : bool
        State the problem with the small-strain law while the physics
        stays neo-Hookean (a wrong-model control).
    swapped : bool
        Give the physics :math:`\\mu` for :math:`\\lambda` and the reverse
        (a wrong-model control).
    reversed_traction : bool
        Close traction sides with the traction's sign reversed (a
        wrong-model control).
    """

    def __init__(
        self,
        lam: str,
        mu: str,
        linear_problem: bool = False,
        swapped: bool = False,
        reversed_traction: bool = False,
    ) -> None:
        self._lam, self._mu = lam, mu
        self._linear_problem, self._swapped = linear_problem, swapped
        self._reversed = reversed_traction

    def terms(self, ndim: int) -> List[TermProtocol]:
        """The strong form as terms on the displacement rows."""
        law = _linear if self._linear_problem else _neo_hookean
        stress = law(ndim, self._lam, self._mu)
        return [
            Divergence(_FIELDS[i], [str(-stress[i, j]) for j in range(ndim)])
            for i in range(ndim)
        ]

    def problem(
        self, solution: Sequence[str], box: Box, codes: str
    ) -> ManufacturedProblem:
        """The problem on ``box``, one code per side for every component."""
        ndim = box.ndim()
        return ManufacturedProblem(
            dict(zip(_FIELDS[:ndim], solution)),
            self.terms(ndim),
            box,
            coded(codes, box),
        )

    def physics(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> HyperelasticityPhysics[Array]:
        lam, mu = (self._mu, self._lam) if self._swapped else (self._lam, self._mu)
        bkd = level.bkd()
        return HyperelasticityPhysics(
            level.basis(),
            bkd,
            NeoHookeanLaw(bkd),
            mu=field(problem, mu, level),
            lamda=field(problem, lam, level),
            forcing=stacked_forcing(problem, level),
        )

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        physics = self.physics(problem, level)
        term = _ReversedTraction if self._reversed else PiolaTractionTerm
        fluxes = {row: term(physics, i) for i, row in enumerate(problem.fields())}
        return collocation_system(physics, problem, level, fluxes)

    def __repr__(self) -> str:
        return (
            f"_Hyperelasticity({self._lam!r}, {self._mu!r}, "
            f"linear_problem={self._linear_problem}, swapped={self._swapped}, "
            f"reversed_traction={self._reversed})"
        )


_LINE = Box([(0.0, 1.0)])
_SQUARE = Box([(0.0, 1.0)] * 2)
_CUBE = Box([(0.0, 1.0)] * 3)
_SOLUTION_1D = ("0.3*x**2*(1 - x) + 0.15*x + 0.1*sin(3*x)",)
_SOLUTION_2D = ("0.1*sin(x + 2*y) + 0.1*x*y", "0.15*cos(2*x - y) - 0.1*y")


@pytest.mark.parametrize(
    "solution, box, codes, npts",
    [
        (("0.2*x + 0.1",), _LINE, "DN", 4),
        (("0.2*x + 0.1",), _LINE, "RD", 4),
        (("0.1*x + 0.05*y", "-0.05*x + 0.15*y + 0.1"), _SQUARE, "DNDR", 4),
        (("0.1*x + 0.05*y", "-0.05*x + 0.15*y + 0.1"), _SQUARE, "NRDN", 4),
        (
            ("0.1*x + 0.02*z", "0.05*y - 0.03*x", "0.04*z + 0.01*y"),
            _CUBE,
            "DNRDND",
            3,
        ),
    ],
    ids=["1d-DN", "1d-RD", "2d-DNDR", "2d-NRDN", "3d"],
)
def test_recovers_affine(
    bkd: Backend[Array], solution: Sequence[str], box: Box, codes: str, npts: int
) -> None:
    """An affine displacement has a constant F, so a constant stress and
    zero forcing, with traction and Robin data polynomial on each side:
    a few nodes per direction reproduce it to 1e-10 with lambda = 1.5,
    mu = 0.8."""
    physics = _Hyperelasticity("1.5", "0.8")
    recovers(physics.problem(solution, box, codes), physics, npts, bkd)


_BUBBLE = "x**2*(1 - x)**2*y**2*(1 - y)**2"


@pytest.mark.parametrize(
    "solution, lam, mu, npts",
    [
        (("0.1*x**2*(1 - x)**2",), "1", "1", 16),
        (("0.2*x**2*(1 - x)**2",), "0.5", "0.5", 16),
        (("0.05*x**2*(1 - x)**2",), "5", "10", 16),
        ((f"0.1*{_BUBBLE}", f"0.05*{_BUBBLE}"), "1", "1", 10),
        ((f"0.05*{_BUBBLE}", f"0.03*{_BUBBLE}"), "10", "1", 10),
    ],
    ids=["1d", "1d-soft", "1d-stiff", "2d", "2d-high-lambda"],
)
def test_residual_vanishes(
    bkd: Backend[Array], solution: Sequence[str], lam: str, mu: str, npts: int
) -> None:
    """Clamped bubbles on the unit line and square, Lamé pairs from soft
    (0.5, 0.5) to stiff (lambda 5, mu 10; lambda 10): the interior
    residual at the exact nodal displacement is below 1e-8, on both
    backends."""
    box = _LINE if len(solution) == 1 else _SQUARE
    elasticity = _Hyperelasticity(lam, mu)
    problem = elasticity.problem(solution, box, "D" * len(box.boundaries()))
    (level,) = ChebyshevBox([npts], bkd).levels(box)
    exact = []
    for component in solution:
        values = field(problem, component, level)
        assert not isinstance(values, float)
        exact.append(values.values(0.0))
    residual = elasticity.physics(problem, level).residual(bkd.concatenate(exact), 0.0)
    boundary = {int(i) for part in box.boundaries() for i in side_nodes(level, part)[0]}
    interior = [
        node + offset
        for offset in range(0, len(solution) * level.npts(), level.npts())
        for node in range(level.npts())
        if node not in boundary
    ]
    interior_residual = residual[bkd.asarray(interior, dtype=bkd.int64_dtype())]
    bkd.assert_allclose(
        interior_residual, bkd.zeros(interior_residual.shape), atol=1e-8
    )


def test_decays_clamped_1d(numpy_bkd: Backend[Array]) -> None:
    """The comparison suite's bar, u = 0.1 x^2 (1 - x)^2 on [0, 1],
    lambda = mu = 1, both ends Dirichlet: errors 1.4e-5 to 1.3e-12 over
    6-14 nodes (observed 1.92 per node, required 0.9); the old test
    needed a relative error below 1e-6 at 20 nodes."""
    physics = _Hyperelasticity("1", "1")
    problem = physics.problem(("0.1*x**2*(1 - x)**2",), _LINE, "DD")
    decays(problem, physics, [6, 8, 10, 12, 14], 0.9, numpy_bkd)


def test_decays_clamped_2d(numpy_bkd: Backend[Array]) -> None:
    """The comparison suite's 2D case, u = 0.1 b, v = 0.05 b with
    b = x^2 (1 - x)^2 y^2 (1 - y)^2 on the unit square, lambda = mu = 1,
    all sides Dirichlet: errors 3.0e-8 to 3.4e-14 over 6-12 nodes per
    direction (observed 2.38 per node, required 1.1); the old test needed
    a relative error below 1e-6 at 10 x 10 nodes."""
    physics = _Hyperelasticity("1", "1")
    bubble = "x**2*(1 - x)**2*y**2*(1 - y)**2"
    problem = physics.problem((f"0.1*{bubble}", f"0.05*{bubble}"), _SQUARE, "DDDD")
    decays(problem, physics, [6, 8, 10, 12], 1.1, numpy_bkd)


@pytest.mark.parametrize("codes", ["DN", "DR", "RD"])
def test_decays_1d(numpy_bkd: Backend[Array], codes: str) -> None:
    """The Galerkin catalog's bar, strains up to 0.45, lambda = mu = 1,
    with its end pairs: over 6-14 nodes, DN 5.5e-3 to 1.6e-7 (observed
    1.30 per node), DR 3.9e-3 to 1.1e-7 (1.30), RD 1.7e-4 to 2.0e-8
    (1.16); required 0.55."""
    physics = _Hyperelasticity("1", "1")
    problem = physics.problem(_SOLUTION_1D, _LINE, codes)
    decays(problem, physics, [6, 8, 10, 12, 14], 0.55, numpy_bkd)


@pytest.mark.parametrize("codes", ["DNDN", "DRDR", "DNDR"])
def test_decays_2d(numpy_bkd: Backend[Array], codes: str) -> None:
    """The Galerkin catalog's 2D solution on the unit square,
    lambda = 1.5, mu = 0.8, with its side sets: over 6-12 nodes per
    direction, DNDN 6.0e-5 to 9.1e-10 (observed 1.86 per node), DRDR
    3.0e-5 to 6.1e-10 (1.81), DNDR 4.0e-5 to 6.4e-10 (1.85); required
    0.9."""
    physics = _Hyperelasticity("1.5", "0.8")
    problem = physics.problem(_SOLUTION_2D, _SQUARE, codes)
    decays(problem, physics, [6, 8, 10, 12], 0.9, numpy_bkd)


def test_decays_varying_lame(numpy_bkd: Backend[Array]) -> None:
    """Lambda = 1 + x, mu = 1 + y/2, sides D, N, D, R: errors 3.8e-5 to
    6.4e-10 over 6-12 nodes per direction (observed 1.84 per node,
    required 0.9)."""
    physics = _Hyperelasticity("1 + x", "1 + y/2")
    problem = physics.problem(_SOLUTION_2D, _SQUARE, "DNDR")
    decays(problem, physics, [6, 8, 10, 12], 0.9, numpy_bkd)


@slow_test
def test_recovers_3d_bubble(numpy_bkd: Backend[Array]) -> None:
    """The old 3D residual case: u, v, w = 0.1 b, 0.05 b, 0.03 b with
    b = x^2 (1 - x)^2 y^2 (1 - y)^2 z^2 (1 - z)^2 on the unit cube,
    lambda = mu = 1, all faces Dirichlet. b has degree 4 per direction,
    so it lies in the space from 5 nodes; 7 nodes reproduce it to 1e-10
    (observed 9.4e-12; 6.6e-11 at 6, the Newton floor). Its forcing is
    slow to generate (about 5 s), not its solve."""
    physics = _Hyperelasticity("1", "1")
    bubble = "x**2*(1 - x)**2*y**2*(1 - y)**2*z**2*(1 - z)**2"
    solution = (f"0.1*{bubble}", f"0.05*{bubble}", f"0.03*{bubble}")
    problem = physics.problem(solution, _CUBE, "DDDDDD")
    recovers(problem, physics, 7, numpy_bkd)


@pytest.mark.parametrize("codes", ["DNDDDD", "DNDDRD"])
def test_decays_3d(numpy_bkd: Backend[Array], codes: str) -> None:
    """The Galerkin catalog's 3D solution on the unit cube, displacement
    gradients up to about 0.15, lambda = mu = 1, a traction right face
    and then also a Robin back face (faces left, right, bottom, top,
    back, front): over 4-7 nodes per direction, DNDDDD 9.5e-5 to 5.9e-9,
    DNDDRD 9.9e-5 to 6.3e-9 (observed about 3.2 per node, required
    1.5). It checks what the affine 3D recovery cannot: the divergence of
    a varying stress and Newton through a varying deformation in 3D."""
    physics = _Hyperelasticity("1", "1")
    solution = (
        "0.1*sin(x + y)*z + 0.05*x",
        "0.1*cos(y - z) - 0.05*x*y",
        "0.1*x*z + 0.05*sin(y)",
    )
    problem = physics.problem(solution, _CUBE, codes)
    decays(problem, physics, [4, 5, 6, 7], 1.5, numpy_bkd)


@pytest.mark.parametrize(
    "physics",
    [
        _Hyperelasticity("1.5", "0.8", linear_problem=True),
        _Hyperelasticity("1.5", "0.8", swapped=True),
    ],
    ids=["linear-law", "swapped-lame"],
)
def test_wrong_models_stall(
    numpy_bkd: Backend[Array], physics: _Hyperelasticity
) -> None:
    """Wrong-model controls on the 2D solution with sides D, N, D, R: the
    error stops falling (small-strain law about 8e-3, swapped Lamé values
    about 1.4e-2, over 6-12 nodes per direction), so the decay cases check
    the nonlinear law, not a linearization of it."""
    problem = physics.problem(_SOLUTION_2D, _SQUARE, "DNDR")
    stalls(problem, physics, [6, 8, 10, 12], numpy_bkd)


def test_reversed_traction_stalls_1d(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: the bar's traction end reads the traction as
    the outward flux, the sign reversed; the error stays at 2.2 over 6-14
    nodes, so the traction cases check the sign."""
    physics = _Hyperelasticity("1", "1", reversed_traction=True)
    problem = physics.problem(_SOLUTION_1D, _LINE, "DN")
    stalls(problem, physics, [6, 8, 10, 12, 14], numpy_bkd)


def test_reversed_traction_stalls_2d(numpy_bkd: Backend[Array]) -> None:
    """The same control in 2D, sides D, N, D, R, on the 2D solution scaled
    by 0.1: the error stays near 3.5e-2 over 6-12 nodes per direction,
    while the correctly signed model reaches 2e-13. At full scale the
    reversed load is large enough to drive J below zero, and the solves
    fail rather than stall (the correctly signed model still decays at
    1.85 per node there)."""
    physics = _Hyperelasticity("1.5", "0.8", reversed_traction=True)
    solution = tuple(f"0.1*({s})" for s in _SOLUTION_2D)
    problem = physics.problem(solution, _SQUARE, "DNDR")
    stalls(problem, physics, [6, 8, 10, 12], numpy_bkd)


def test_piola_traction_term_refuses_non_stress() -> None:
    """A stress that does not satisfy PK1StressSourceProtocol is refused
    at construction."""
    with pytest.raises(TypeError, match="PK1StressSourceProtocol"):
        PiolaTractionTerm(object(), 0)  # type: ignore[arg-type]
