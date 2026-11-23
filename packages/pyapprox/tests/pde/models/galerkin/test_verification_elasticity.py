r"""Galerkin linear elasticity verified by manufactured solutions.

:class:`~pyapprox.pde.galerkin.physics.composite_linear_elasticity.
CompositeLinearElasticity` solves :math:`-\nabla\cdot\sigma(u) = f` with
:math:`\sigma = \lambda (\nabla\cdot u) I + \mu (\nabla u + \nabla u^T)`
on a vector Lagrange basis. The problem states it row by row, the
displacement components :math:`u, v, w` as fields, through
:mod:`pyapprox.pde.verification.solid_mechanics`:
:math:`\nabla\cdot(-\sigma_{i\cdot}) - f_i = 0`, so its natural data are
:math:`-\sigma n`, which the adapter negates into this stack's traction
:math:`\sigma n`. Constant Lamé values reach the physics as
:math:`(E, \nu)` through ``from_uniform``, so the conversion is under test
too; varying ones are set at the quadrature points. Errors are true
:math:`L^2` errors of the displacement vector.

Cases:

- exact recovery (L2 below 1e-10) of the old suite's cases: the 3D patch
  test (linear :math:`u`, all Dirichlet, hexes and tetrahedra, P1 and
  P2, two cells per side), the 3D quadratic case (P2, :math:`\lambda =
  1.5`, :math:`\mu = 0.8`), the varying Lamé case (:math:`\lambda = 1 +
  x/2`, :math:`\mu = 1 + y/4`, P1, three hexes per side), the roller
  case, generalized from :math:`\nu = 0` to symmetry planes holding one
  component each with the others traction free (also with the free
  components ``Unprescribed()``, one part mixing them with a Robin one),
  and the 1D P2 bar with
  nonzero clamped ends;
- rates (fitted over the last three levels, within 0.1): 1D P1 2 and P2
  3; 2D on quads and triangles, P1 2 and P2 3, with Dirichlet, Neumann
  and Robin sides; the old suite's Robin case (P1 2); parts mixing a
  Dirichlet, a Neumann and a Robin component; 3D hexes with varying
  Lamé values, P1 2;
- wrong-model controls, which stall (fitted rate below 0.5): Robin
  built without its spring term, and the physics given :math:`\lambda`
  and :math:`\mu` swapped;
- refusals: a basis with more components than the problem has fields.
"""

from typing import Any, Dict, List, Sequence, Union

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import sympy as sp
from numpy.typing import NDArray

from pyapprox.pde.boundary.protocols import BoundaryConditionRole
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import RobinBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.verification import (
    GalerkinDiscretizer,
    SkfemPoints,
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
    Unprescribed,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.protocols import SpaceLevel
from pyapprox.pde.verification.solid_mechanics import (
    DISPLACEMENT_FIELDS,
    equilibrium_terms,
    linear_stress,
)
from pyapprox.pde.verification.terms import Divergence, TermProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.galerkin_verification import (
    converges_on,
    recovers_on,
    stalls_on,
)

_Np = NDArray[Any]
_FIELDS = DISPLACEMENT_FIELDS
_Conditions = Dict[str, Union[ConditionProtocol, Dict[str, ConditionProtocol]]]


class _Elasticity:
    """The physics under test and its strong form, from one pair of Lamé
    values.

    Parameters
    ----------
    lam, mu : str
        :math:`\\lambda(x)` and :math:`\\mu(x)`.
    swapped : bool
        Give the physics :math:`\\mu` for :math:`\\lambda` and the reverse
        (the wrong-model control).
    """

    def __init__(self, lam: str, mu: str, swapped: bool = False) -> None:
        self._lam, self._mu, self._swapped = lam, mu, swapped

    def terms(self, ndim: int) -> List[TermProtocol]:
        return equilibrium_terms(linear_stress(ndim, self._lam, self._mu))

    def physics(
        self, problem: ManufacturedProblem, basis: VectorLagrangeBasis[_Np]
    ) -> CompositeLinearElasticity[_Np]:
        rows = problem.fields()
        lam, mu = (self._mu, self._lam) if self._swapped else (self._lam, self._mu)
        lam_value, mu_value = sp.sympify(lam), sp.sympify(mu)
        constant = lam_value.is_number and mu_value.is_number
        # E and nu of the Lame values, so from_uniform converts them back.
        youngs, poisson = (
            (
                float(
                    mu_value * (3 * lam_value + 2 * mu_value) / (lam_value + mu_value)
                ),
                float(lam_value / (2 * (lam_value + mu_value))),
            )
            if constant
            else (1.0, 0.25)
        )
        physics = CompositeLinearElasticity.from_uniform(
            basis, youngs, poisson, NumpyBkd(), body_force=vector_forcing(problem, rows)
        )
        if not constant:
            physics.set_lame_parameters(
                quadrature_values(problem, lam, basis),
                quadrature_values(problem, mu, basis),
            )
        return physics

    def system(
        self, problem: ManufacturedProblem, basis: VectorLagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        return compose_galerkin_system(
            self.physics(problem, basis),
            vector_boundary_conditions(problem, basis, problem.fields()),
        )

    def __repr__(self) -> str:
        return f"_Elasticity({self._lam!r}, {self._mu!r}, swapped={self._swapped})"


def _problem(
    solution: Sequence[str],
    elasticity: _Elasticity,
    box: Box,
    conditions: _Conditions,
) -> ManufacturedProblem:
    ndim = box.ndim()
    return ManufacturedProblem(
        dict(zip(_FIELDS[:ndim], solution)),
        elasticity.terms(ndim),
        box,
        conditions,
    )


def _sides(box: Box, kinds: str) -> _Conditions:
    """One condition per side, for every component: D, N or R (alpha 2)."""
    kind: Dict[str, ConditionProtocol] = {
        "D": Dirichlet(),
        "N": Neumann(),
        "R": Robin(2.0),
    }
    return {side: kind[k] for k, side in zip(kinds, box.boundaries())}


_UNIT_SQUARE = Box([(0.0, 1.0)] * 2)
_UNIT_CUBE = Box([(0.0, 1.0)] * 3)
_LINEAR_3D = (
    "0.1 + 0.2*x - 0.1*y + 0.05*z",
    "-0.05 + 0.1*x + 0.15*y - 0.2*z",
    "0.02 - 0.1*x + 0.05*y + 0.1*z",
)


class TestRecovery:
    @pytest.mark.parametrize("degree", [1, 2])
    @pytest.mark.parametrize("simplex", [False, True])
    def test_3d_patch(self, degree: int, simplex: bool) -> None:
        """The old suite's patch test: a linear displacement, all sides
        Dirichlet, lambda = 1.5, mu = 0.8, is reproduced to 1e-10 on two
        cells per side, hexes and tetrahedra, P1 and P2."""
        elasticity = _Elasticity("1.5", "0.8")
        problem = _problem(
            _LINEAR_3D, elasticity, _UNIT_CUBE, _sides(_UNIT_CUBE, "DDDDDD")
        )
        recovers_on(problem, VectorLagrange(degree), elasticity, 2, simplex)

    @pytest.mark.parametrize("simplex", [False, True])
    def test_3d_quadratic(self, simplex: bool) -> None:
        """The old suite's quadratic case, now with traction and Robin
        sides: a quadratic displacement is in the P2 (Q2) space, so P2 on
        two cells per side reproduces it."""
        elasticity = _Elasticity("1.5", "0.8")
        solution = ("x**2 + 0.5*y*z", "0.3*x*y - z**2", "0.2*y**2 + x*z")
        problem = _problem(
            solution, elasticity, _UNIT_CUBE, _sides(_UNIT_CUBE, "DNRDND")
        )
        recovers_on(problem, VectorLagrange(2), elasticity, 2, simplex)

    def test_varying_lame(self) -> None:
        """The old suite's case: lambda = 1 + x/2, mu = 1 + y/4 set at the
        quadrature points, with a linear displacement (a constant body
        force results), is reproduced by P1 on three hexes per side."""
        elasticity = _Elasticity("1 + x/2", "1 + y/4")
        problem = _problem(
            _LINEAR_3D, elasticity, _UNIT_CUBE, _sides(_UNIT_CUBE, "DDDDDD")
        )
        recovers_on(problem, VectorLagrange(1), elasticity, 3)

    def test_symmetry_planes(self) -> None:
        """The old roller case, generalized: each of the planes x = 0,
        y = 0, z = 0 holds its normal component (a roller) and leaves the
        others traction free, the opposite faces carry tractions, and no
        rigid motion is left. A linear displacement vanishing on those
        planes is reproduced by P1 on two hexes per side, so a Dirichlet
        subset of components combines with Neumann on one part."""
        elasticity = _Elasticity("1.5", "0.8")
        neumann, dirichlet = Neumann(), Dirichlet()

        def roller(held: str) -> Dict[str, ConditionProtocol]:
            return {f: dirichlet if f == held else neumann for f in _FIELDS}

        conditions: _Conditions = {
            "left": roller("u"),
            "right": neumann,
            "bottom": roller("v"),
            "top": neumann,
            "back": roller("w"),
            "front": neumann,
        }
        problem = _problem(
            ("0.1*x", "-0.05*y", "0.2*z"),
            elasticity,
            _UNIT_CUBE,
            conditions,
        )
        recovers_on(problem, VectorLagrange(1), elasticity, 2)

    def test_unprescribed_components(self) -> None:
        """The symmetry planes with the free components stated as
        ``Unprescribed()``, and a right face springing u (alpha 2) with v
        and w unprescribed, so a part mixes an essential or a natural
        component with free ones. The shear tractions vanish there, so P1
        on two hexes per side reproduces the displacement."""
        elasticity = _Elasticity("1.5", "0.8")
        free, dirichlet, neumann = Unprescribed(), Dirichlet(), Neumann()

        def roller(held: str) -> Dict[str, ConditionProtocol]:
            return {f: dirichlet if f == held else free for f in _FIELDS}

        conditions: _Conditions = {
            "left": roller("u"),
            "right": {"u": Robin(2.0), "v": free, "w": free},
            "bottom": roller("v"),
            "top": neumann,
            "back": roller("w"),
            "front": neumann,
        }
        problem = _problem(
            ("0.1*x", "-0.05*y", "0.2*z"),
            elasticity,
            _UNIT_CUBE,
            conditions,
        )
        recovers_on(problem, VectorLagrange(1), elasticity, 2)

    def test_1d_bar(self) -> None:
        """The old suite's 1D case: u = x(1 - x)/10 + 1/5, nonzero at both
        clamped ends, is reproduced by P2 on 10 cells. It guards the 1D
        vector basis's boundary DOFs, once returned empty."""
        elasticity = _Elasticity("0.6", "0.4")
        box = Box([(0.0, 1.0)])
        problem = _problem(("x*(1 - x)/10 + 1/5",), elasticity, box, _sides(box, "DD"))
        recovers_on(problem, VectorLagrange(2), elasticity, 10)


class TestRates:
    def test_1d(self) -> None:
        """A bar, lambda = 1 + x, mu = 0.5, clamped left and loaded right:
        L2 rate 2 for P1 (cells 8-64) and 3 for P2 (4-32)."""
        elasticity = _Elasticity("1 + x", "0.5")
        box = Box([(0.0, 1.0)])
        problem = _problem(("sin(2*x) + x",), elasticity, box, _sides(box, "DN"))
        converges_on(
            problem, StructuredBox([8, 16, 32, 64]), VectorLagrange(1), elasticity, 2.0
        )
        converges_on(
            problem, StructuredBox([4, 8, 16, 32]), VectorLagrange(2), elasticity, 3.0
        )

    @pytest.mark.parametrize("simplex", [False, True])
    def test_2d(self, simplex: bool) -> None:
        """Unit square, lambda = 2, mu = 1 + x y / 2, sides D, N, R, D:
        L2 rate 2 for P1 (4-32 cells per side) and 3 for P2 (2-16), on
        quads and on triangles."""
        elasticity = _Elasticity("2", "1 + x*y/2")
        problem = _problem(
            ("sin(x + y)/2", "cos(x - 2*y)/4"),
            elasticity,
            _UNIT_SQUARE,
            _sides(_UNIT_SQUARE, "DNRD"),
        )
        converges_on(
            problem,
            StructuredBox([4, 8, 16, 32], simplex),
            VectorLagrange(1),
            elasticity,
            2.0,
        )
        converges_on(
            problem,
            StructuredBox([2, 4, 8, 16], simplex),
            VectorLagrange(2),
            elasticity,
            3.0,
        )

    def test_robin(self) -> None:
        """The old suite's Robin case: lambda = 1.5, mu = 0.8, a spring of
        stiffness 2 on x = 1, the other sides clamped, a coupled
        non-polynomial displacement nonzero on the spring edge: P1 rate 2
        (4-32 cells per side)."""
        elasticity = _Elasticity("1.5", "0.8")
        problem = _problem(
            ("0.1*sin(x + 0.5*y) + 0.05", "0.05*cos(0.3*x + y) + 0.02"),
            elasticity,
            _UNIT_SQUARE,
            _sides(_UNIT_SQUARE, "DRDD"),
        )
        converges_on(
            problem, StructuredBox([4, 8, 16, 32]), VectorLagrange(1), elasticity, 2.0
        )

    def test_mixed_components(self) -> None:
        """Parts mixing kinds by component: the left side holds u and
        springs v (alpha 3), the right side springs u (alpha 2) and loads
        v, the top springs u and v with different stiffnesses, the bottom
        is clamped. P1 rate 2 and P2 rate 3 show each component gets its
        own condition."""
        elasticity = _Elasticity("1.5", "0.8")
        conditions: _Conditions = {
            "left": {"u": Dirichlet(), "v": Robin(3.0)},
            "right": {"u": Robin(2.0), "v": Neumann()},
            "bottom": Dirichlet(),
            "top": {"u": Robin(1.0), "v": Robin("1 + x")},
        }
        problem = _problem(
            ("sin(x + y)/2 + x", "cos(x - 2*y)/4"), elasticity, _UNIT_SQUARE, conditions
        )
        converges_on(
            problem, StructuredBox([4, 8, 16, 32]), VectorLagrange(1), elasticity, 2.0
        )
        converges_on(
            problem, StructuredBox([2, 4, 8, 16]), VectorLagrange(2), elasticity, 3.0
        )

    def test_3d(self) -> None:
        """Unit cube, hexes 2-8, varying lambda = 1 + z, mu = 1 + x/2, all
        three kinds: P1 rate 2."""
        elasticity = _Elasticity("1 + z", "1 + x/2")
        problem = _problem(
            ("sin(x + y)*z/2", "cos(y - z)/4", "exp(x)*y/5"),
            elasticity,
            _UNIT_CUBE,
            _sides(_UNIT_CUBE, "DNRDDN"),
        )
        converges_on(
            problem, StructuredBox([2, 4, 8]), VectorLagrange(1), elasticity, 2.0
        )


class _NoSpring(_Elasticity):
    """The adapter's conditions with the Robin side ``right`` rebuilt
    without its spring term alpha u (alpha = 0) but with the problem's
    data kept."""

    def system(
        self, problem: ManufacturedProblem, basis: VectorLagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        rows = problem.fields()
        conditions: List[BoundaryConditionRole[_Np]] = [
            bc
            for bc in vector_boundary_conditions(problem, basis, rows)
            if not isinstance(bc, RobinBC)
        ]
        data = problem.boundary_data("right")
        flux = _Negated([data[row] for row in rows], problem)
        conditions.append(RobinBC(basis, "right", 0.0, flux, NumpyBkd()))
        return compose_galerkin_system(self.physics(problem, basis), conditions)


class _Negated:
    """This stack's natural data from the core's, as the adapter does."""

    def __init__(self, data: Sequence[sp.Expr], problem: ManufacturedProblem) -> None:
        self._points = SkfemPoints(data, problem)

    def evaluate(self, coords: _Np, normal: _Np, time: float) -> _Np:
        return -self._points(coords, normals=normal)

    def is_time_dependent(self) -> bool:
        return False


def test_robin_without_spring_stalls() -> None:
    """Wrong-model control: the Robin case with the spring term dropped
    stalls, so the Robin rates can see it."""
    elasticity = _NoSpring("1.5", "0.8")
    problem = _problem(
        ("0.1*sin(x + 0.5*y) + 0.05", "0.05*cos(0.3*x + y) + 0.02"),
        elasticity,
        _UNIT_SQUARE,
        _sides(_UNIT_SQUARE, "DRDD"),
    )
    stalls_on(problem, StructuredBox([4, 8, 16, 32]), VectorLagrange(1), elasticity)


def test_swapped_lame_stalls() -> None:
    """Wrong-model control: the physics given lambda and mu swapped while
    the forcing uses them as stated; the error stops falling, so the
    rates check the constitutive law, not only the discretization."""
    elasticity = _Elasticity("1.5", "0.8", swapped=True)
    problem = _problem(
        ("sin(x + y)/2", "cos(x - 2*y)/4"),
        elasticity,
        _UNIT_SQUARE,
        _sides(_UNIT_SQUARE, "DNND"),
    )
    stalls_on(problem, StructuredBox([4, 8, 16, 32]), VectorLagrange(1), elasticity)


def test_basis_with_more_components_than_fields_is_refused() -> None:
    """A 2D vector basis carries two fields, so a one-field problem on
    the square is refused before any solve."""
    elasticity = _Elasticity("1", "1")
    problem = ManufacturedProblem(
        {"u": "x"},
        [Divergence("u", ["-u_x", "-u_y"])],
        _UNIT_SQUARE,
        _sides(_UNIT_SQUARE, "DDDD"),
    )
    discretizer = GalerkinDiscretizer(StructuredBox([2]), VectorLagrange(1), elasticity)
    with pytest.raises(ValueError, match="one field per component"):
        discretizer.solve(problem, SpaceLevel(0))
