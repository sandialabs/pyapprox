r"""Transient Galerkin discretizations verified by manufactured solutions.

The problem is heat flow :math:`u_t + \nabla\cdot(-D\nabla u) = f` on
:math:`[0, 1]`, :math:`D = 0.1`, Dirichlet data varying in time at both
ends, stepped by
:class:`~pyapprox.pde.models.galerkin.verification.
GalerkinTransientDiscretizer` from the interpolant of :math:`u^*` and
measured by the largest L2 error over every step
(:func:`~pyapprox.pde.verification.transient.max_over_steps`).

The solutions are quadratic in :math:`x`, so P2 on four cells holds them
exactly with a consistent mass, and every error is the time stepper's:

- exact in time, up to the method's stage order, recovered to 1e-10
  relative to :math:`\max_t \|u^*\|`: :math:`u^* = (x + x(1-x))(1+t)`
  for backward Euler, forward Euler, implicit midpoint and Heun,
  :math:`(x + x(1-x))(1+t+t^2)` for Crank-Nicolson, on
  :math:`[0, 0.2]` (implicit steps 0.05, explicit 0.01);
- the control: implicit midpoint (stage order 1) leaves an error above
  1e-6 on the degree-2 solution at step 0.05;
- temporal order with :math:`u^* = (x + x(1-x))(1+t^3)` on
  :math:`[0, 1]`, the fitted rate along ``DT`` within 0.15 from both
  sides: backward Euler 1, forward Euler 1, Crank-Nicolson 2, implicit
  midpoint 2, Heun 2 (implicit steps 0.1 to 0.025, explicit 0.01 to
  0.0025), with the essential rows pinned at each step's end and at
  every block.

Also the sizes (``H``, ``DEGREE``, ``NDOFS`` and ``DT``), the series'
times, and the refusals: a problem without a time derivative, a level
beyond the family, a method or grid of the wrong type.
"""

from typing import Any, Callable, Dict, List, Optional

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from numpy.typing import NDArray

from pyapprox.ode.pinned_rows import AtEveryBlock, AtStepEnd, PinPlacementProtocol
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    forward_euler,
    heun,
    implicit_midpoint,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.verification import (
    GalerkinTransientDiscretizer,
    Lagrange,
    StructuredBox,
    boundary_conditions,
    field,
    forcing,
    galerkin_l2_error,
)
from pyapprox.pde.verification.conditions import ConditionProtocol, Dirichlet
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.protocols import DEGREE, DT, NDOFS, H, SpaceLevel
from pyapprox.pde.verification.study import Converges, Recovers, run_study
from pyapprox.pde.verification.terms import Divergence, TermProtocol, TimeDerivative
from pyapprox.pde.verification.transient import (
    SpaceSizesProtocol,
    TimeGrid,
    TransientLevel,
    max_over_steps,
    refine_time,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]
_DIFFUSIVITY = "0.1"
_SHAPE = "(x + x*(1 - x))"


class _Method:
    """A Runge-Kutta method by name, its order and stage order."""

    def __init__(
        self,
        name: str,
        tableau: Callable[[], ButcherTableau],
        order: int,
        stage_order: int,
    ) -> None:
        self.name, self._tableau = name, tableau
        self.order, self.stage_order = order, stage_order

    def explicit(self) -> bool:
        return self._tableau().is_explicit()

    def stepping(self) -> RungeKutta[_Np, _Np]:
        tableau = self._tableau()
        return RungeKutta(
            tableau,
            None
            if tableau.is_explicit()
            else Newton(atol=1e-12, rtol=0.0, maxiters=20),
        )

    def __repr__(self) -> str:
        return self.name


_METHODS = [
    _Method("backward_euler", backward_euler, 1, 1),
    _Method("forward_euler", forward_euler, 1, 1),
    _Method("crank_nicolson", crank_nicolson, 2, 2),
    _Method("implicit_midpoint", implicit_midpoint, 2, 1),
    _Method("heun", heun, 2, 1),
]
_IMPLICIT_MIDPOINT = _METHODS[3]
_PLACEMENTS: Dict[str, PinPlacementProtocol] = {
    "at_step_end": AtStepEnd(),
    "at_every_block": AtEveryBlock(),
}


class _Heat:
    """Heat flow with constant diffusivity, the physics from the same
    string as the strong form."""

    def system(
        self, problem: ManufacturedProblem, basis: LagrangeBasis[_Np]
    ) -> GalerkinSystem[_Np]:
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=field(problem, _DIFFUSIVITY),
            bkd=NumpyBkd(),
            forcing=forcing(problem, "u"),
        )
        return compose_galerkin_system(
            physics, boundary_conditions(problem, basis, "u")
        )

    def __repr__(self) -> str:
        return "_Heat()"


def _problem(solution: str, transient: bool = True) -> ManufacturedProblem:
    terms: List[TermProtocol] = [Divergence("u", [f"-({_DIFFUSIVITY})*u_x"])]
    if transient:
        terms.insert(0, TimeDerivative("u"))
    conditions: Dict[str, ConditionProtocol] = {
        "left": Dirichlet(),
        "right": Dirichlet(),
    }
    return ManufacturedProblem({"u": solution}, terms, Box([(0.0, 1.0)]), conditions)


def _discretizer(
    method: _Method,
    grid: TimeGrid,
    placement: Optional[PinPlacementProtocol] = None,
) -> GalerkinTransientDiscretizer[LagrangeBasis[_Np]]:
    return GalerkinTransientDiscretizer(
        StructuredBox([4]), Lagrange(2), _Heat(), method.stepping(), grid, placement
    )


@pytest.mark.parametrize("method", _METHODS, ids=repr)
def test_exact_in_time_up_to_the_stage_order(method: _Method) -> None:
    solution = (
        f"{_SHAPE}*(1 + t)" if method.stage_order == 1 else (f"{_SHAPE}*(1 + t + t**2)")
    )
    grid = TimeGrid(0.0, 0.2, 0.01 if method.explicit() else 0.05)
    study = run_study(
        _problem(solution),
        _discretizer(method, grid),
        [max_over_steps(galerkin_l2_error())],
    )
    result = Recovers(rtol=1e-10).verify(study)
    assert result.passed, result.message


def test_implicit_midpoint_does_not_reproduce_degree_two() -> None:
    """The control: stage order 1 leaves an error on a quadratic in t."""
    grid = TimeGrid(0.0, 0.2, 0.05)
    study = run_study(
        _problem(f"{_SHAPE}*(1 + t + t**2)"),
        _discretizer(_IMPLICIT_MIDPOINT, grid),
        [max_over_steps(galerkin_l2_error())],
    )
    assert not Recovers(rtol=1e-10).verify(study).passed
    assert min(study.errors()) > 1e-6


@pytest.mark.parametrize("placement", list(_PLACEMENTS), ids=str)
@pytest.mark.parametrize("method", _METHODS, ids=repr)
def test_temporal_order(method: _Method, placement: str) -> None:
    grid = TimeGrid(0.0, 1.0, 0.01 if method.explicit() else 0.1)
    discretizer = _discretizer(method, grid, _PLACEMENTS[placement])
    study = run_study(
        _problem(f"{_SHAPE}*(1 + t**3)"),
        discretizer,
        [max_over_steps(galerkin_l2_error())],
        refine_time(SpaceLevel(0), grid, 3),
    )
    result = Converges(float(method.order), axis=DT, atol=0.15).verify(study)
    assert result.passed, result.message


def test_sizes_and_times() -> None:
    grid = TimeGrid(0.0, 0.2, 0.05)
    discretizer = _discretizer(_METHODS[0], grid)
    problem = _problem(f"{_SHAPE}*(1 + t)")
    assert isinstance(discretizer, SpaceSizesProtocol)
    (request,) = discretizer.requests(problem)
    sizes = discretizer.sizes(problem, request)
    assert (sizes.of(H), sizes.of(DEGREE), sizes.of(NDOFS), sizes.of(DT)) == (
        0.25,
        2,
        9,
        0.05,
    )
    assert DT not in discretizer.space_sizes(problem, request.space).values
    series = discretizer.solve(problem, request)
    assert series.ntimes() == 5
    assert series.times()[0] == 0.0 and series.times()[-1] == pytest.approx(0.2)
    assert series.at(4).time() == series.times()[-1]
    with pytest.raises(IndexError, match="no 5"):
        series.at(5)


def test_refusals() -> None:
    grid = TimeGrid(0.0, 0.2, 0.05)
    discretizer = _discretizer(_METHODS[0], grid)
    steady = _problem(f"{_SHAPE}*(1 + t)", transient=False)
    with pytest.raises(ValueError, match="has none"):
        discretizer.requests(steady)
    with pytest.raises(ValueError, match="no level 3"):
        discretizer.solve(
            _problem(f"{_SHAPE}*(1 + t)"), TransientLevel(SpaceLevel(3), grid)
        )
    with pytest.raises(TypeError, match="SteppingMethodProtocol"):
        GalerkinTransientDiscretizer(
            StructuredBox([4]),
            Lagrange(2),
            _Heat(),
            3.0,  # type: ignore[arg-type]
            grid,
        )
    with pytest.raises(TypeError, match="TimeGrid"):
        GalerkinTransientDiscretizer(
            StructuredBox([4]),
            Lagrange(2),
            _Heat(),
            _METHODS[0].stepping(),
            0.1,  # type: ignore[arg-type]
        )
