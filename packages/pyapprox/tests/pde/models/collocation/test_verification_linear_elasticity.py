r"""Collocation linear elasticity verified by manufactured solutions.

:class:`~pyapprox.pde.collocation.physics.linear_elasticity.
LinearElasticityPhysics` solves :math:`-\nabla\cdot\sigma(u) = f` with
:math:`\sigma = \lambda (\nabla\cdot u) I + \mu (\nabla u + \nabla u^T)`.
The problem states it row by row, the displacement components
:math:`u, v, w` as fields: :math:`\nabla\cdot(-\sigma_{i\cdot}) = f_i`,
so its natural data are :math:`-\sigma n`. Traction and Robin sides close
on the physics' Cauchy traction through
:class:`~pyapprox.pde.models.collocation.verification.TractionTerm`,
which takes that sign. The Lamé values and the forcing are written out
here, not taken from the physics.

Cases:

- recovery (L2 error over all components below 1e-10) of the old
  suite's polynomial solutions vanishing on the boundary of
  :math:`[-1, 1]^2` under zero Dirichlet sides, for several Lamé pairs
  and at their node counts; with variable Lamé values; with traction
  Neumann and traction Robin on one side and the other sides Dirichlet;
  in 3D; and with conditions mixing kinds by component, on both backends;
- spectral decay, fitted per node, of analytic solutions in 2D and 3D;
- wrong-model controls, which stall: the physics given :math:`\lambda`
  and :math:`\mu` swapped, and traction sides given the traction with
  the wrong sign.
"""

from typing import Dict, List, Mapping, Sequence, Union

import pytest

from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.verification import (
    ChebyshevLevel,
    TractionTerm,
    collocation_system,
    field,
    stacked_forcing,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Robin,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import Divergence, TermProtocol
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.collocation_verification import decays, recovers, stalls

_FIELDS = ("u", "v", "w")
_COORDS = ("x", "y", "z")
_Conditions = Mapping[str, Union[ConditionProtocol, Mapping[str, ConditionProtocol]]]


class _ReversedTraction(TractionTerm[Array]):
    """The traction with the wrong sign: :math:`t_i` read as the outward
    flux :math:`q_i\\cdot n` (the wrong-model control)."""

    def sign(self) -> float:
        return 1.0


class _Elasticity:
    """The physics under test and its strong form, from one pair of Lamé
    values.

    Parameters
    ----------
    lam, mu : str
        :math:`\\lambda(x)` and :math:`\\mu(x)`.
    swapped : bool, optional
        Give the physics :math:`\\mu` for :math:`\\lambda` and the reverse
        (a wrong-model control).
    reversed_traction : bool, optional
        Close traction sides with the traction's sign reversed (a
        wrong-model control).
    """

    def __init__(
        self,
        lam: str,
        mu: str,
        swapped: bool = False,
        reversed_traction: bool = False,
    ) -> None:
        self._lam, self._mu = lam, mu
        self._swapped, self._reversed = swapped, reversed_traction

    def terms(self, ndim: int) -> List[TermProtocol]:
        """The strong form as terms on the displacement rows."""
        fields, coords = _FIELDS[:ndim], _COORDS[:ndim]
        divergence = " + ".join(f"{f}_{c}" for f, c in zip(fields, coords))
        terms: List[TermProtocol] = []
        for i in range(ndim):
            stresses = []
            for j in range(ndim):
                shear = (
                    f"({self._mu})*({fields[i]}_{coords[j]} + {fields[j]}_{coords[i]})"
                )
                normal = f"({self._lam})*({divergence}) + " if i == j else ""
                stresses.append(f"-({normal}{shear})")
            terms.append(Divergence(fields[i], stresses))
        return terms

    def problem(
        self, solution: Sequence[str], box: Box, conditions: _Conditions
    ) -> ManufacturedProblem:
        ndim = box.ndim()
        return ManufacturedProblem(
            dict(zip(_FIELDS[:ndim], solution)),
            self.terms(ndim),
            box,
            conditions,
        )

    def system(
        self, problem: ManufacturedProblem, level: ChebyshevLevel[Array]
    ) -> CollocationSystem[Array]:
        lam, mu = (self._mu, self._lam) if self._swapped else (self._lam, self._mu)
        physics = LinearElasticityPhysics(
            level.basis(),
            level.bkd(),
            lamda=field(problem, lam, level),
            mu=field(problem, mu, level),
            forcing=stacked_forcing(problem, level),
        )
        term = _ReversedTraction if self._reversed else TractionTerm
        fluxes = {row: term(physics, i) for i, row in enumerate(problem.fields())}
        return collocation_system(physics, problem, level, fluxes)

    def __repr__(self) -> str:
        return (
            f"_Elasticity({self._lam!r}, {self._mu!r}, swapped={self._swapped}, "
            f"reversed_traction={self._reversed})"
        )


def _sides(box: Box, codes: str) -> Dict[str, ConditionProtocol]:
    """One condition per side, for every component: D, N or R (alpha 2)."""
    kinds: Dict[str, ConditionProtocol] = {
        "D": Dirichlet(),
        "N": Neumann(),
        "R": Robin(2.0),
    }
    return {side: kinds[code] for code, side in zip(codes, box.boundaries())}


_SQUARE = Box([(-1.0, 1.0)] * 2)
_UNIT_SQUARE = Box([(0.0, 1.0)] * 2)
_CUBE = Box([(-1.0, 1.0)] * 3)
_BUBBLE = ("(1 - x**2)*(1 - y**2)", "(1 - x**2)*(1 - y**2)*x")


@pytest.mark.parametrize(
    "solution, lam, mu, npts",
    [
        (_BUBBLE, "1", "1", 10),
        (("(1 - x**2)*(1 - y**2)*x*y",) * 2, "1", "1", 12),
        (("(1 - x**2)*(1 - y**2)*x**2", "(1 - x**2)*(1 - y**2)*y**2"), "1", "1", 14),
        (_BUBBLE, "10", "1", 10),
        (_BUBBLE, "1", "10", 10),
        (_BUBBLE, "100", "1", 10),
        (_BUBBLE, "2.5", "0.5", 10),
        (_BUBBLE, "2 + x + y", "1 + x*y/2", 10),
    ],
    ids=[
        "basic",
        "symmetric",
        "higher-degree",
        "high-lambda",
        "high-mu",
        "nearly-incompressible",
        "lambda-2.5-mu-0.5",
        "variable-lame",
    ],
)
def test_recovers_dirichlet_polynomials(
    numpy_bkd: Backend[Array], solution: Sequence[str], lam: str, mu: str, npts: int
) -> None:
    """The old suite's polynomials vanishing on the boundary of
    [-1, 1]^2, zero Dirichlet sides: reproduced to 1e-10 at the old node
    counts, including lambda = 100 and variable Lamé values."""
    elasticity = _Elasticity(lam, mu)
    problem = elasticity.problem(solution, _SQUARE, _sides(_SQUARE, "DDDD"))
    recovers(problem, elasticity, npts, numpy_bkd)


@pytest.mark.parametrize("codes", ["NDDD", "RDDD"], ids=["traction", "robin"])
def test_recovers_traction_side(numpy_bkd: Backend[Array], codes: str) -> None:
    """The old traction cases: u = x^2 (1 - y^2), v = x y (1 - y^2),
    lambda = mu = 1, nonzero on the left side, which carries a traction
    or a Robin condition, the other sides Dirichlet: 10 nodes reproduce
    it to 1e-10."""
    elasticity = _Elasticity("1", "1")
    problem = elasticity.problem(
        ("x**2*(1 - y**2)", "x*y*(1 - y**2)"), _SQUARE, _sides(_SQUARE, codes)
    )
    recovers(problem, elasticity, 10, numpy_bkd)


@pytest.mark.parametrize("codes", ["DDDDDD", "DNRDDN"])
def test_recovers_3d(numpy_bkd: Backend[Array], codes: str) -> None:
    """The old 3D cubic displacement on [-1, 1]^3, lambda = 2.5,
    mu = 0.5, all sides Dirichlet as in the old suite and with traction
    and Robin faces: 5 nodes per direction reproduce it to 1e-10."""
    elasticity = _Elasticity("2.5", "0.5")
    solution = (
        "(1 - x**2)*(1 - y**2)*(1 - z**2)",
        "(1 - x**2)*(1 - y**2)*(1 - z**2)*x",
        "(1 - x**2)*(1 - y**2)*(1 - z**2)*y",
    )
    problem = elasticity.problem(solution, _CUBE, _sides(_CUBE, codes))
    recovers(problem, elasticity, 5, numpy_bkd)


def test_recovers_mixed_components(bkd: Backend[Array]) -> None:
    """Parts mixing kinds by component on the unit square: the left side
    holds u and springs v (alpha 3), the right side springs u (alpha 2)
    and loads v, the top springs u and v with different stiffnesses, the
    bottom is clamped. A cubic displacement with lambda = 1.5,
    mu = 1 + x y/2 is reproduced to 1e-10 on 6 x 6 nodes, on both
    backends, so each component gets its own condition."""
    elasticity = _Elasticity("1.5", "1 + x*y/2")
    conditions: _Conditions = {
        "left": {"u": Dirichlet(), "v": Robin(3.0)},
        "right": {"u": Robin(2.0), "v": Neumann()},
        "bottom": Dirichlet(),
        "top": {"u": Robin(1.0), "v": Robin("1 + x")},
    }
    problem = elasticity.problem(
        ("x*y + y**2 - x**2*y", "x**2 + x*y**2 - y**3/3"), _UNIT_SQUARE, conditions
    )
    recovers(problem, elasticity, 6, bkd)


def test_decays_2d(numpy_bkd: Backend[Array]) -> None:
    """u = sin(x + y)/2, v = cos(x - 2 y)/4 on the unit square,
    lambda = 2, mu = 1 + x y/2, sides Dirichlet, traction, Robin,
    Dirichlet: exponential decay over 4-12 nodes per direction (8e-3 to
    4e-12; observed 2.69 per node, required 1)."""
    elasticity = _Elasticity("2", "1 + x*y/2")
    problem = elasticity.problem(
        ("sin(x + y)/2", "cos(x - 2*y)/4"), _UNIT_SQUARE, _sides(_UNIT_SQUARE, "DNRD")
    )
    decays(problem, elasticity, [4, 6, 8, 10, 12], 1.0, numpy_bkd)


def test_decays_3d(numpy_bkd: Backend[Array]) -> None:
    """The old 3D spectral case: sin(pi x) sin(pi y) sin(pi z) in every
    component on [-1, 1]^3, lambda = mu = 1, zero Dirichlet faces:
    exponential decay over 4-8 nodes per direction (0.9 to 3e-3;
    observed 1.45 per node, required 1)."""
    elasticity = _Elasticity("1", "1")
    problem = elasticity.problem(
        ("sin(pi*x)*sin(pi*y)*sin(pi*z)",) * 3, _CUBE, _sides(_CUBE, "DDDDDD")
    )
    decays(problem, elasticity, [4, 6, 8], 1.0, numpy_bkd)


def test_swapped_lame_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: the physics given lambda and mu swapped while
    the forcing uses them as stated; the error stops falling (fitted rate
    0.016, L2 about 0.068), so the cases above check the constitutive
    law."""
    elasticity = _Elasticity("1.5", "0.8", swapped=True)
    problem = elasticity.problem(
        ("sin(x + y)/2", "cos(x - 2*y)/4"), _UNIT_SQUARE, _sides(_UNIT_SQUARE, "DNND")
    )
    stalls(problem, elasticity, [6, 8, 10, 12], numpy_bkd)


def test_reversed_traction_stalls(numpy_bkd: Backend[Array]) -> None:
    """Wrong-model control: traction sides that read the traction as the
    outward flux, the sign reversed, prescribe the wrong load; the error
    stops falling (fitted rate 0.017, L2 about 0.53), so the traction
    cases check the sign."""
    elasticity = _Elasticity("1.5", "0.8", reversed_traction=True)
    problem = elasticity.problem(
        ("sin(x + y)/2", "cos(x - 2*y)/4"), _UNIT_SQUARE, _sides(_UNIT_SQUARE, "DNND")
    )
    stalls(problem, elasticity, [6, 8, 10, 12], numpy_bkd)
