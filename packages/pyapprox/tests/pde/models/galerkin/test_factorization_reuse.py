r"""Stage factorizations kept by each setup strategy, on a Galerkin problem.

1D heat equation on [0, 1], P1 on 10 cells, forcing (1 + x)(1 + sin 5t),
u(0) = 0 and a moving right end 0.5 sin 4t (its rate supplied), from
0.3 sin(pi x), 20 steps of dt = 0.001 to T = 0.02 (``FixedStep``: the
sizes are given exactly, the last one landing on T), Newton tolerance
1e-13, the essential rows in differentiated form (the unconstrained
system under ``differentiated_rows``) and pinned in each step's last
block. The Jacobian does not see the state, so ``KeepStateFree`` and
``KeepAll`` keep every stage setup; ``RecomputeAlways`` keeps none.
Setups, cumulative over two forward solves:

- :math:`\kappa = 1 + x` declared time-independent: the stage operator
  depends only on the step size (two: 0.001, and the last step's) and the
  pinning. Kept, backward Euler and Crank-Nicolson factor twice, SDIRK3
  four times (its unpinned stages share one operator, its pinned last
  stage is another), and the second solve nothing; recomputed, every
  implicit stage of both solves (20, 40 and 60, 120).
- :math:`\kappa = (1 + t)(1 + x)` declared time-dependent: every stage
  time is its own operator, so the first solve factors at every implicit
  stage under any strategy (20, SDIRK3 60); kept, the second solve, at
  the same times, factors nothing.
- ``clear()`` drops what was kept: the next solve factors as the first.

Every strategy's trajectory equals the recomputed one to round-off.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import KeepAll, KeepStateFree, RecomputeAlways
from pyapprox.ode.differentiated_rows import differentiated_rows
from pyapprox.ode.integration import Integrator
from pyapprox.ode.pinned_rows import RowPinning
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.setups import MassShiftStrategy
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.tableau import (
    ButcherTableau,
    alexander_sdirk3,
    backward_euler,
    crank_nicolson,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.boundary.signal import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import (
    CoordinateDiffusion,
    TimeDependent,
    TimeIndependent,
)
from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh.structured import StructuredMesh1D
from pyapprox.pde.galerkin.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.galerkin.time_integration.essential_pinned_rows import (
    EssentialPinnedRows,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.unconstrained_system import (
    galerkin_unconstrained_system,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    LinearSolverProtocol,
    SolverHandleProtocol,
)
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton

_Np = NDArray[Any]
_DELTAT, _NSTEPS = 0.001, 20
# Steps of exactly 0.001: the size is given, not recovered from times
# (0.001 n would give four different sizes in 20 steps).
_STEPS = FixedStep(_DELTAT, (0.0, _NSTEPS * _DELTAT))


def _forcing(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * (1.0 + np.sin(5.0 * t))
    return value


def _right_value(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 0.5 * np.sin(4.0 * t))
    return value


def _right_rate(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 2.0 * np.cos(4.0 * t))
    return value


def _kappa(x: _Np) -> _Np:
    value: _Np = 1.0 + x[0]
    return value


def _growing_kappa(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + t) * (1.0 + x[0])
    return value


class _CountingLU:
    """The sparse LU, counting its setups."""

    def __init__(self) -> None:
        self._lu = ScipySparseLU()
        self._setups = 0

    def setups(self) -> int:
        return self._setups

    def setup(
        self, op: ActionOperatorProtocol[_Np, CSRFormat]
    ) -> SolverHandleProtocol[_Np]:
        self._setups += 1
        return self._lu.setup(op)


_Strategy = MassShiftStrategy[_Np, _Np, CSRFormat]
_StrategyOf = Callable[[LinearSolverProtocol[_Np, CSRFormat]], _Strategy]


class _Run:
    """The heat problem under one tableau, diffusivity and strategy,
    stepped with the step-end rows pinned; its solver counts setups."""

    def __init__(
        self,
        bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        diffusivity: CoordinateDiffusion,
        strategy_of: _StrategyOf,
    ) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=diffusivity,
            bkd=bkd,
            forcing=TimeDependent(_forcing),
        )
        right = BoundarySignal(
            TimeDependent(_right_value), [TimeDependent(_right_rate)]
        )
        system = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", right, bkd),
            ],
        )
        self.lu = _CountingLU()
        discretization = skfem_discretization(system)
        rows = EssentialPinnedRows(discretization)
        newton: Newton[_Np, _Np] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
        self.strategy = strategy_of(self.lu)
        self.stepper = RungeKuttaStepper(
            make(),
            differentiated_rows(
                galerkin_unconstrained_system(discretization, self.lu), rows
            ),
            self.strategy,
            newton,
            RowPinning(rows),
        )
        x = np.asarray(basis.dof_coordinates())[0]
        self.init_state: _Np = (0.3 * np.sin(np.pi * x))[:, None]

    def solve(self) -> List[_Np]:
        """The stored states of one solve."""
        store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        Integrator(self.stepper).solve(self.init_state, _STEPS, store)
        return [store.state(n) for n in range(store.ntimes())]

    def two_solves(self) -> Tuple[List[_Np], Tuple[int, int]]:
        """The first solve's states and the setups after each solve."""
        states = self.solve()
        first = self.lu.setups()
        self.solve()
        return states, (first, self.lu.setups())


_CONSTANT = CoordinateDiffusion(TimeIndependent(_kappa))
_GROWING = CoordinateDiffusion(TimeDependent(_growing_kappa))
_STRATEGIES: List[_StrategyOf] = [RecomputeAlways, KeepStateFree, KeepAll]
_STRATEGY_IDS = ["recompute_always", "keep_state_free", "keep_all"]
_TABLEAU_IDS = ["backward_euler", "crank_nicolson", "sdirk3"]
# Each tableau with its implicit stages per step and its distinct stage
# operators per step size.
_TABLEAUS = [(backward_euler, 1, 1), (crank_nicolson, 1, 1), (alexander_sdirk3, 3, 2)]


def _recomputed(
    bkd: NumpyBkd, make: Callable[[], ButcherTableau], diffusivity: CoordinateDiffusion
) -> List[_Np]:
    return _Run(bkd, make, diffusivity, RecomputeAlways).solve()


class TestFactorizationReuse:
    @pytest.mark.parametrize("strategy_of", _STRATEGIES, ids=_STRATEGY_IDS)
    @pytest.mark.parametrize(
        "make, implicit_stages, distinct", _TABLEAUS, ids=_TABLEAU_IDS
    )
    def test_a_constant_jacobian(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        implicit_stages: int,
        distinct: int,
        strategy_of: _StrategyOf,
    ) -> None:
        # The last step lands exactly on 0.02, absorbing the drift of the
        # summed times, so it has a size of its own: a kept stage operator
        # is factored once for 0.001 and once for the last step.
        assert len(set(_STEPS.sizes())) == 2
        run = _Run(numpy_bkd, make, _CONSTANT, strategy_of)
        states, counts = run.two_solves()
        if strategy_of is RecomputeAlways:
            per_solve = _NSTEPS * implicit_stages
            assert counts == (per_solve, 2 * per_solve)
        else:
            assert counts == (2 * distinct, 2 * distinct)
        for mine, theirs in zip(states, _recomputed(numpy_bkd, make, _CONSTANT)):
            numpy_bkd.assert_allclose(mine, theirs, rtol=1e-13, atol=1e-15)

    @pytest.mark.parametrize("strategy_of", _STRATEGIES, ids=_STRATEGY_IDS)
    @pytest.mark.parametrize(
        "make, implicit_stages, distinct", _TABLEAUS, ids=_TABLEAU_IDS
    )
    def test_a_time_dependent_jacobian(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        implicit_stages: int,
        distinct: int,
        strategy_of: _StrategyOf,
    ) -> None:
        run = _Run(numpy_bkd, make, _GROWING, strategy_of)
        states, counts = run.two_solves()
        per_solve = _NSTEPS * implicit_stages
        second = 2 * per_solve if strategy_of is RecomputeAlways else per_solve
        assert counts == (per_solve, second)
        for mine, theirs in zip(states, _recomputed(numpy_bkd, make, _GROWING)):
            numpy_bkd.assert_allclose(mine, theirs, rtol=1e-13, atol=1e-15)

    @pytest.mark.parametrize(
        "strategy_of", [KeepStateFree, KeepAll], ids=_STRATEGY_IDS[1:]
    )
    def test_clear_drops_what_was_kept(
        self, numpy_bkd: NumpyBkd, strategy_of: _StrategyOf
    ) -> None:
        run = _Run(numpy_bkd, alexander_sdirk3, _CONSTANT, strategy_of)
        run.solve()
        assert run.lu.setups() == 4
        run.strategy.clear()
        run.solve()
        assert run.lu.setups() == 8
