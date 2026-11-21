r"""Galerkin neo-Hookean hyperelasticity verified by manufactured solutions.

Both :class:`~pyapprox.pde.galerkin.physics.HyperelasticityPhysics`
(given a :class:`~pyapprox.pde.constitutive.neo_hookean.NeoHookeanStress`)
and :class:`~pyapprox.pde.galerkin.physics.composite_hyperelasticity.
CompositeHyperelasticityPhysics` solve :math:`-\nabla\cdot P(F) = f`, total
Lagrangian, with the first Piola stress

.. math:: P = \mu F + (\lambda \ln J - \mu) F^{-T}, \quad F = I + \nabla u,
   \quad J = \det F.

The problem states it row by row (fields :math:`u, v, w`):
:math:`\nabla\cdot(-P_{i\cdot}) - f_i = 0`, with :math:`P` written from
the formula above by :mod:`pyapprox.pde.verification.solid_mechanics`,
not from the stress model's own symbolic form, so the two are
independent. Its natural data are :math:`-P N`, negated by the
adapter into this stack's nominal traction :math:`P N`. Solutions have
displacement gradients up to about 0.45, large enough that the stress is far
from linear (the linearized control below stalls), small enough that
:math:`J > 0`. Every solve starts from zero.

Cases, each for both classes unless stated:

- exact recovery (L2 below 1e-10) of the old suite's 3D case: a small
  quadratic displacement, all Dirichlet, P2 on two hexes or tetrahedra
  per side (limited only by the quadrature of the :math:`\ln J` terms);
- rates (fitted over the last three levels, within 0.1): 1D P1 2 and P2
  3 with the old suite's end pairs D-N, D-R, R-D; 2D quads P1 2 and P2 3
  with its side sets D-N, D-R, D-N-R; triangles; varying Lamé values set
  at the quadrature points; 3D hexes P1 2 with its face sets;
- wrong-model controls, which stall (fitted rate below 0.5): the problem
  stated with the linear stress law, and the physics given :math:`\lambda`
  and :math:`\mu` swapped.
"""

from typing import Any, Dict, List, Sequence, Union

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import sympy as sp
from numpy.typing import NDArray

from pyapprox.pde.constitutive.neo_hookean import NeoHookeanStress
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import HyperelasticityPhysics
from pyapprox.pde.galerkin.physics.composite_hyperelasticity import (
    CompositeHyperelasticityPhysics,
)
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.verification import (
    StructuredBox,
    VectorLagrange,
    quadrature_values,
    vector_boundary_conditions,
    vector_forcing,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Robin,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.solid_mechanics import (
    DISPLACEMENT_FIELDS,
    equilibrium_terms,
    linear_stress,
    neo_hookean_stress,
)
from pyapprox.pde.verification.terms import TermProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.galerkin_verification import (
    converges_on,
    recovers_on,
    stalls_on,
)

_Np = NDArray[Any]
_FIELDS = DISPLACEMENT_FIELDS


class _Hyperelasticity:
    """One of the two physics classes under test and its strong form,
    from one pair of Lamé values.

    Parameters
    ----------
    lam, mu : str
        :math:`\\lambda(x)` and :math:`\\mu(x)`.
    composite : bool
        Build ``CompositeHyperelasticityPhysics`` (constant values as
        :math:`(E, \\nu)` through ``from_uniform``) instead of
        ``HyperelasticityPhysics`` with a ``NeoHookeanStress``.
    linear_problem : bool
        State the problem with the linear stress law while the physics
        stays neo-Hookean (a wrong-model control).
    swapped : bool
        Give the physics :math:`\\mu` for :math:`\\lambda` and the reverse
        (a wrong-model control).
    """

    def __init__(
        self,
        lam: str,
        mu: str,
        composite: bool = False,
        linear_problem: bool = False,
        swapped: bool = False,
    ) -> None:
        self._lam, self._mu = lam, mu
        self._composite = composite
        self._linear_problem, self._swapped = linear_problem, swapped

    def terms(self, ndim: int) -> List[TermProtocol]:
        law = linear_stress if self._linear_problem else neo_hookean_stress
        return equilibrium_terms(law(ndim, self._lam, self._mu))

    def _lame(
        self, problem: ManufacturedProblem, basis: VectorLagrangeBasis[_Np]
    ) -> tuple[Union[float, _Np], Union[float, _Np]]:
        """The physics' Lamé values: floats, or values at the quadrature
        points."""
        lam, mu = (self._mu, self._lam) if self._swapped else (self._lam, self._mu)
        values: List[Union[float, _Np]] = []
        for expression in (lam, mu):
            parsed = sp.sympify(expression)
            values.append(
                float(parsed)
                if parsed.is_number
                else quadrature_values(problem, expression, basis)
            )
        return values[0], values[1]

    def system(
        self, problem: ManufacturedProblem, basis: VectorLagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        rows = problem.fields()
        bkd = NumpyBkd()
        force = vector_forcing(problem, rows)
        lam, mu = self._lame(problem, basis)
        physics: Union[
            HyperelasticityPhysics[_Np], CompositeHyperelasticityPhysics[_Np]
        ]
        if not self._composite:
            physics = HyperelasticityPhysics(
                basis, NeoHookeanStress(lam, mu), bkd, body_force=force
            )
        elif isinstance(lam, float) and isinstance(mu, float):
            # E and nu of the Lame values, so from_uniform converts back.
            youngs = mu * (3 * lam + 2 * mu) / (lam + mu)
            poisson = lam / (2 * (lam + mu))
            physics = CompositeHyperelasticityPhysics.from_uniform(
                basis, youngs, poisson, bkd, body_force=force
            )
        else:
            physics = CompositeHyperelasticityPhysics.from_uniform(
                basis, 1.0, 0.25, bkd, body_force=force
            )
            physics.set_lame_parameters(
                quadrature_values(problem, self._lam, basis),
                quadrature_values(problem, self._mu, basis),
            )
        return compose_galerkin_system(
            physics, vector_boundary_conditions(problem, basis, rows)
        )

    def __repr__(self) -> str:
        return (
            f"_Hyperelasticity({self._lam!r}, {self._mu!r}, "
            f"composite={self._composite}, linear_problem={self._linear_problem}, "
            f"swapped={self._swapped})"
        )


def _problem(
    solution: Sequence[str],
    physics: _Hyperelasticity,
    box: Box,
    codes: str,
) -> ManufacturedProblem:
    """The problem on ``box``, one code per side for every component: D,
    N or R (alpha 1)."""
    kinds: Dict[str, ConditionProtocol] = {
        "D": Dirichlet(),
        "N": Neumann(),
        "R": Robin(1.0),
    }
    ndim = box.ndim()
    return ManufacturedProblem(
        dict(zip(_FIELDS[:ndim], solution)),
        physics.terms(ndim),
        box,
        {side: kinds[code] for code, side in zip(codes, box.boundaries())},
    )


_COMPOSITE = pytest.mark.parametrize(
    "composite", [False, True], ids=["plain", "composite"]
)
_LINE = Box([(0.0, 1.0)])
_SQUARE = Box([(0.0, 1.0)] * 2)
_CUBE = Box([(0.0, 1.0)] * 3)
_SOLUTION_1D = ("0.3*x**2*(1 - x) + 0.15*x + 0.1*sin(3*x)",)
_SOLUTION_2D = ("0.1*sin(x + 2*y) + 0.1*x*y", "0.15*cos(2*x - y) - 0.1*y")


@_COMPOSITE
@pytest.mark.parametrize("simplex", [False, True], ids=["hex", "tet"])
def test_3d_quadratic_recovers(composite: bool, simplex: bool) -> None:
    """The old suite's 3D case, lambda = mu = 1: a small quadratic
    displacement, nonzero on the boundary, all faces Dirichlet, is in the
    P2 space, so P2 on two cells per side reproduces it to 1e-10."""
    physics = _Hyperelasticity("1", "1", composite)
    solution = (
        "0.02*x*y + 0.01*z**2",
        "0.01*x**2 - 0.02*y*z",
        "0.005*x*z + 0.015*y**2",
    )
    problem = _problem(solution, physics, _CUBE, "DDDDDD")
    recovers_on(problem, VectorLagrange(2), physics, 2, simplex)


class TestRates:
    @_COMPOSITE
    @pytest.mark.parametrize("codes", ["DN", "DR", "RD"])
    def test_1d(self, composite: bool, codes: str) -> None:
        """A bar, lambda = 1, mu = 1, with the old suite's end pairs
        (Dirichlet-Neumann, Dirichlet-Robin, Robin-Dirichlet) and strains
        up to 0.45: L2 rate 2 for P1 (cells 8-64) and 3 for P2 (4-32)."""
        physics = _Hyperelasticity("1", "1", composite)
        problem = _problem(_SOLUTION_1D, physics, _LINE, codes)
        converges_on(
            problem, StructuredBox([8, 16, 32, 64]), VectorLagrange(1), physics, 2.0
        )
        converges_on(
            problem, StructuredBox([4, 8, 16, 32]), VectorLagrange(2), physics, 3.0
        )

    @_COMPOSITE
    @pytest.mark.parametrize("codes", ["DNDN", "DRDR", "DNDR"])
    def test_2d(self, composite: bool, codes: str) -> None:
        """Unit square, lambda = 1.5, mu = 0.8, with the old suite's side
        sets (sides in the order left, right, bottom, top): L2 rate 2 for
        P1 (4-32 cells per side) and 3 for P2 (2-16)."""
        physics = _Hyperelasticity("1.5", "0.8", composite)
        problem = _problem(_SOLUTION_2D, physics, _SQUARE, codes)
        converges_on(
            problem, StructuredBox([4, 8, 16, 32]), VectorLagrange(1), physics, 2.0
        )
        converges_on(
            problem, StructuredBox([2, 4, 8, 16]), VectorLagrange(2), physics, 3.0
        )

    def test_triangles(self) -> None:
        """The 2D case with D, N, D, R sides on triangles: P1 rate 2
        (4-32 cells per side), P2 rate 3 (2-16)."""
        physics = _Hyperelasticity("1.5", "0.8")
        problem = _problem(_SOLUTION_2D, physics, _SQUARE, "DNDR")
        meshes = [
            StructuredBox([4, 8, 16, 32], True),
            StructuredBox([2, 4, 8, 16], True),
        ]
        converges_on(problem, meshes[0], VectorLagrange(1), physics, 2.0)
        converges_on(problem, meshes[1], VectorLagrange(2), physics, 3.0)

    @_COMPOSITE
    def test_varying_lame(self, composite: bool) -> None:
        """Lambda = 1 + x, mu = 1 + y/2 set at the quadrature points, sides
        D, N, D, R: P1 rate 2 (4-32 cells per side)."""
        physics = _Hyperelasticity("1 + x", "1 + y/2", composite)
        problem = _problem(_SOLUTION_2D, physics, _SQUARE, "DNDR")
        converges_on(
            problem, StructuredBox([4, 8, 16, 32]), VectorLagrange(1), physics, 2.0
        )

    @_COMPOSITE
    @pytest.mark.parametrize("codes", ["DNDDDD", "DNDDRD"])
    def test_3d(self, composite: bool, codes: str) -> None:
        """Unit cube, lambda = mu = 1, the old suite's face sets: traction
        on the right face, then also a Robin back face (faces in the order
        left, right, bottom, top, back, front): P1 rate 2 (hexes 2-8)."""
        physics = _Hyperelasticity("1", "1", composite)
        solution = (
            "0.1*sin(x + y)*z + 0.05*x",
            "0.1*cos(y - z) - 0.05*x*y",
            "0.1*x*z + 0.05*sin(y)",
        )
        problem = _problem(solution, physics, _CUBE, codes)
        converges_on(problem, StructuredBox([2, 4, 8]), VectorLagrange(1), physics, 2.0)


def test_linear_stress_law_stalls() -> None:
    """Wrong-model control: the forcing and tractions made from the
    small-strain law while the physics is neo-Hookean. At these strains
    the two differ well above the P2 discretization error, so the error
    stops falling: the rates check the nonlinear constitutive law, not a
    linearization of it."""
    physics = _Hyperelasticity("1.5", "0.8", linear_problem=True)
    problem = _problem(_SOLUTION_2D, physics, _SQUARE, "DNDR")
    stalls_on(problem, StructuredBox([2, 4, 8, 16]), VectorLagrange(2), physics)


def test_swapped_lame_stalls() -> None:
    """Wrong-model control: the physics given lambda and mu swapped while
    the forcing uses them as stated; the error stops falling."""
    physics = _Hyperelasticity("1.5", "0.8", swapped=True)
    problem = _problem(_SOLUTION_2D, physics, _SQUARE, "DNDR")
    stalls_on(problem, StructuredBox([4, 8, 16, 32]), VectorLagrange(1), physics)
