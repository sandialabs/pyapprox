r"""Galerkin Euler-Bernoulli beams verified by manufactured solutions.

:class:`~pyapprox.pde.galerkin.physics.euler_bernoulli.EulerBernoulliBeamFEM`
solves :math:`(EI\, w'')'' = f` on cubic Hermite elements, whose DOFs at
each node are the deflection and the slope. The problem states the row as
:math:`(q)' = f` with :math:`q = EI\, w'''`, so the core gives the forcing
and, at an end with a Neumann condition, the shear data :math:`q\, n`; the
weak form :math:`\int EI\, w'' v'' = \int f v - \sum (q n) v + \sum
(EI\, w'' n) v'` takes it negated, as this stack's other natural data.

The core states one condition per row and end; a beam end needs two. The
second is supplied here, from the exact solution:

- ``C`` clamped: deflection and slope prescribed (Dirichlet, plus the
  slope DOF at :math:`w^{*\prime}`);
- ``P`` pinned: deflection prescribed, zero moment;
- ``F`` free: shear prescribed (Neumann), zero moment;
- ``G`` guided: shear prescribed, slope prescribed.

A zero-moment end needs no data, so solutions are chosen with
:math:`w^{*\prime\prime} = 0` there. Applied end moments are not tested:
this stack has no condition loading the slope test function. The
harness's discretizer reads a basis's DOF components as fields, which
does not fit a Hermite basis, so these tests solve as it does
(``galerkin_steady_residual`` + ``SteadySystem``, sparse LU, Newton),
integrate errors themselves and judge them with the harness's checks.

Cases:

- exact recovery (L2 below 1e-10): cubic deflections, clamped-clamped
  and clamped-free;
- nodal exactness of the old suite's analytic beams (deflection and
  slope at the nodes to 1e-10): the cantilever under uniform and linear
  loads and the simply supported beam under a uniform load, with the
  forcing and end data from the core rather than the formulas;
- rates: L2 4 for clamped-free, pinned-pinned, clamped-pinned and
  guided-pinned ends;
- wrong-model controls, which stall (fitted rate below 0.5): EI doubled
  in the physics, and clamped ends built without their slope condition.
"""

from typing import Any, Dict, List, Sequence, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from skfem import Basis, Functional

from pyapprox.implicitfunction.setups import KeepAll
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.pde.boundary.protocols import BoundaryConditionRole
from pyapprox.pde.galerkin.basis import CubicHermiteBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC, NeumannBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics.euler_bernoulli import EulerBernoulliBeamFEM
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.steady_residual import galerkin_steady_residual
from pyapprox.pde.models.galerkin.verification import SkfemPoints
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.protocols import H, Sizes
from pyapprox.pde.verification.study import (
    CheckResult,
    Converges,
    Recovers,
    Stalls,
    Study,
)
from pyapprox.pde.verification.terms import Divergence
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import InnerProductSpaceProtocol
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]
_BKD = NumpyBkd()
_X = sp.Symbol("x")
# The core's condition on the deflection row at an end of each kind.
_ROW_CONDITION: Dict[str, ConditionProtocol] = {
    "C": Dirichlet(),
    "P": Dirichlet(),
    "F": Neumann(),
    "G": Neumann(),
}
_SLOPE_PRESCRIBED = {"C", "G"}


class _Load:
    """The core's forcing as the beam's load ``q(x)``, ``x`` in skfem's
    quadrature layout."""

    def __init__(self, problem: ManufacturedProblem) -> None:
        self._points = SkfemPoints([problem.forcing("w")], problem)

    def __call__(self, x: _Np) -> _Np:
        values: _Np = self._points(np.asarray(x)[None])[0]
        return values


class _EndValues:
    """The deflection and slope of the exact solution, ``(2, npts)``: one
    row per DOF component of the Hermite basis."""

    def __init__(self, problem: ManufacturedProblem) -> None:
        w = problem.solution("w")
        self._points = SkfemPoints([w, sp.diff(w, _X)], problem)

    def __call__(self, coords: _Np, time: float) -> _Np:
        values: _Np = self._points(coords, t=time)
        return values

    def is_time_dependent(self) -> bool:
        return False


class _Shear:
    """The core's shear data :math:`q\\, n`, negated into this stack's
    natural load."""

    def __init__(self, data: sp.Expr, problem: ManufacturedProblem) -> None:
        self._points = SkfemPoints([data], problem)

    def evaluate(self, coords: _Np, normal: _Np, time: float) -> _Np:
        values: _Np = -self._points(coords, normals=normal, t=time)[0]
        return values

    def is_time_dependent(self) -> bool:
        return False


def _problem(
    solution: str, ends: str, ei: float, length: float = 1.0
) -> ManufacturedProblem:
    """:math:`(EI\\, w''')' = f` on ``[0, length]`` with the core's
    condition for each end kind."""
    box = Box([(0.0, length)])
    return ManufacturedProblem(
        {"w": solution},
        [Divergence("w", [f"({ei})*w_xxx"])],
        box,
        {side: _ROW_CONDITION[end] for side, end in zip(box.boundaries(), ends)},
    )


def _solve(
    problem: ManufacturedProblem,
    ends: str,
    nx: int,
    ei: float,
    slopes: bool = True,
) -> Tuple[CubicHermiteBasis[_Np], _Np]:
    """The beam with stiffness ``ei`` on ``nx`` elements, its ends as
    ``ends``; ``slopes=False`` leaves out the slope conditions."""
    domain = problem.domain()
    assert isinstance(domain, Box)
    basis = CubicHermiteBasis(StructuredMesh1D(nx, domain.bounds()[0], _BKD))
    beam = EulerBernoulliBeamFEM(basis, ei, _Load(problem), _BKD)
    conditions: List[BoundaryConditionRole[_Np]] = []
    for side, end in zip(domain.boundaries(), ends):
        if end in ("C", "P"):
            conditions.append(
                DirichletBC(basis, side, _EndValues(problem), _BKD, components=(0,))
            )
        else:
            data = problem.boundary_data(side)["w"]
            conditions.append(NeumannBC(basis, side, _Shear(data, problem), _BKD))
        if slopes and end in _SLOPE_PRESCRIBED:
            conditions.append(
                DirichletBC(basis, side, _EndValues(problem), _BKD, components=(1,))
            )
    system = compose_galerkin_system(beam, conditions)
    # The steady path every verification suite solves through.
    discretization = skfem_discretization(system)
    space = discretization.space()
    assert isinstance(space, InnerProductSpaceProtocol)
    steady = SteadySystem(
        galerkin_steady_residual(discretization), space, KeepAll(ScipySparseLU())
    )
    point = steady.solve(
        Newton(atol=1e-12, rtol=1e-12, maxiters=30, line_search=True),
        np.zeros((system.nstates(), 1)),
    )
    assert point.converged(), point.info().message
    return basis, np.asarray(point.state()[:, 0])


def _l2_error(
    problem: ManufacturedProblem, basis: CubicHermiteBasis[_Np], state: _Np
) -> float:
    r""":math:`\|w^* - w_h\|_{L^2}`, by skfem quadrature of order 9."""
    skfem_basis = basis.skfem_basis()
    fine = Basis(skfem_basis.mesh, skfem_basis.elem, intorder=9)
    exact = SkfemPoints([problem.solution("w")], problem)

    def squared_error(w: Any) -> _Np:
        return np.asarray((w["uh"].value - exact(np.asarray(w.x))[0]) ** 2)

    value = Functional(squared_error).assemble(fine, uh=fine.interpolate(state))
    return float(np.sqrt(value))


def _nodal_error(
    problem: ManufacturedProblem, basis: CubicHermiteBasis[_Np], state: _Np
) -> float:
    """The largest error of the nodal deflections and slopes."""
    w = problem.solution("w")
    nodes = np.asarray(basis.mesh().nodes())
    exact = SkfemPoints([w, sp.diff(w, _X)], problem)(nodes)
    values = np.asarray(basis.values(_BKD.asarray(state)))
    slopes = np.asarray(basis.slopes(_BKD.asarray(state)))
    return float(max(np.abs(values - exact[0]).max(), np.abs(slopes - exact[1]).max()))


def _study(
    problem: ManufacturedProblem,
    ends: str,
    sizes: Sequence[int],
    ei: float,
    slopes: bool = True,
) -> Study:
    """L2 errors over ``nx`` elements per level, against h."""
    domain = problem.domain()
    assert isinstance(domain, Box)
    lower, upper = domain.bounds()[0]
    errors = []
    for nx in sizes:
        basis, state = _solve(problem, ends, nx, ei, slopes)
        errors.append(_l2_error(problem, basis, state))
    return Study.from_records(
        [
            (Sizes({H: (upper - lower) / nx}), {"l2": error})
            for nx, error in zip(sizes, errors)
        ]
    )


def _check(result: CheckResult) -> None:
    assert result.passed, result.message


# w'' vanishes at both ends, so either end may carry a zero moment.
_SMOOTH = "sin(pi*x) + x/2 + 1/5"
_LEVELS = (4, 8, 16, 32)


class TestRecovery:
    @pytest.mark.parametrize(
        "solution,ends",
        [
            ("1 + x - x**2 + x**3/2", "CC"),
            ("1/5 + x - 3*x**2 + x**3", "CF"),
        ],
        ids=["clamped-clamped", "clamped-free"],
    )
    def test_cubic(self, solution: str, ends: str) -> None:
        """A cubic deflection is in the Hermite space and has no load:
        four elements reproduce it to 1e-10, clamped at both ends with
        nonzero deflection and slope, and clamped-free with a nonzero
        shear at the free end (w'' = 0 there, so its moment is zero)."""
        problem = _problem(solution, ends, 2.0)
        _check(Recovers(1e-10).verify(_study(problem, ends, (4,), 2.0)))

    @pytest.mark.parametrize(
        "solution",
        [
            # q0 / (24 EI) x^2 (6 L^2 - 4 L x + x^2), q0 = 1.5, EI = 3, L = 2
            "1.5/(24*3)*x**2*(6*2**2 - 4*2*x + x**2)",
            # q0 / (120 L EI) x^2 (20 L^3 - 10 L^2 x + x^3)
            "1.5/(120*2*3)*x**2*(20*2**3 - 10*2**2*x + x**3)",
        ],
        ids=["uniform", "linear"],
    )
    def test_analytic_cantilever(self, solution: str) -> None:
        """The old suite's analytic cantilevers (L = 2, EI = 3, q0 = 1.5,
        uniform and linear loads), clamped at x = 0 and free at x = L:
        cubic Hermite elements are nodally exact for beams, so the nodal
        deflections and slopes on eight elements match to 1e-10. The
        load and the free end's zero shear come from the core."""
        problem = _problem(solution, "CF", 3.0, length=2.0)
        basis, state = _solve(problem, "CF", 8, 3.0)
        assert _nodal_error(problem, basis, state) < 1e-10

    def test_analytic_simply_supported(self) -> None:
        """The old suite's simply supported beam, w = q x (L^3 - 2 L x^2
        + x^3) / (24 EI) with q = 1, EI = 1.5, L = 2, pinned at both
        ends: nodally exact on eight elements to 1e-10."""
        problem = _problem("x*(2**3 - 2*2*x**2 + x**3)/(24*1.5)", "PP", 1.5, length=2.0)
        basis, state = _solve(problem, "PP", 8, 1.5)
        assert _nodal_error(problem, basis, state) < 1e-10


class TestRates:
    @pytest.mark.parametrize("ends", ["CF", "PP", "CP", "GP"])
    def test_smooth(self, ends: str) -> None:
        """A smooth deflection with w'' = 0 at both ends, EI = 2: the L2
        error converges at 4 over 4 to 32 elements, with clamped, pinned,
        free and guided ends."""
        problem = _problem(_SMOOTH, ends, 2.0)
        _check(Converges(4.0, last=3).verify(_study(problem, ends, _LEVELS, 2.0)))


class TestControls:
    def test_stiffness_doubled(self) -> None:
        """The physics given 2 EI for the problem's EI solves another
        beam: the error stalls."""
        problem = _problem(_SMOOTH, "CF", 2.0)
        _check(Stalls(below=0.5).verify(_study(problem, "CF", _LEVELS, 4.0)))

    def test_clamps_without_slopes(self) -> None:
        """Clamped ends built without their slope condition become pinned
        with zero moment; the solution has w'' nonzero at both ends, so
        the error stalls."""
        problem = _problem("sin(2*x) + x", "CC", 2.0)
        study = _study(problem, "CC", _LEVELS, 2.0, slopes=False)
        _check(Stalls(below=0.5).verify(study))
