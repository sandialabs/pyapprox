r"""Stage factorizations shared by a solve and its derivative sweeps, per
setup strategy.

The linear ODE :math:`\dot y = A y + B p` (3 states, 2 parameters, its
Jacobian :math:`A` declared constant), from (1, -0.5, 2), 10 steps of
0.1 to T = 1 (``FixedStep``: the last step lands on 1 with a size of its
own, so a step forms two different stage operators over the run), an MSE
with noise 0.5 observing every state at step 2 and the last. The
stepper's solver counts its setups; cumulative after the forward solve,
the gradient and the HVP:

- ``KeepStateFree`` and ``KeepAll``: backward Euler, Crank-Nicolson and
  SDIRK3 factor their stage operator once per step size in the forward
  solve (2), and the sweeps, asking for the same operators, factor
  nothing: 2, 2, 2. Heun, explicit, factors the mass once: 1, 1, 1.
- ``RecomputeAlways``: every solve is factored in every pass (steps
  times solves per step for the forward solve and for the gradient's
  sweep, twice that for the HVP: the tangent sweep, and the adjoints
  sharing one handle per step): 10, 20, 40 (SDIRK3 30, 60, 120; Heun,
  an explicit stage and a final block per step, 20, 40, 80).
- The gradient and HVP of every strategy equal the recomputed ones to
  1e-13.
"""

from typing import Any, Callable, List, Tuple

import numpy as np
import pytest

from pyapprox.implicitfunction.setups import KeepAll, KeepStateFree, RecomputeAlways
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.observation_terms import MisfitTerm
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.setups import MassShiftStrategy
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.tableau import (
    ButcherTableau,
    alexander_sdirk3,
    backward_euler,
    crank_nicolson,
    heun,
)
from pyapprox.ode.trajectory_derivatives import (
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    LinearSolverProtocol,
    SolverCapabilities,
    SolverHandleProtocol,
)
from pyapprox.util.rootfinding.nonlinear import Newton
from pyapprox_benchmarks.functions.ode.linear_ode import LinearODEResidual

_Np = Any
_STEPS = FixedStep(0.1, (0.0, 1.0))
_NSTEPS = len(_STEPS.sizes())


class _CountingLU:
    """The dense LU, counting its setups."""

    def __init__(self, bkd: NumpyBkd) -> None:
        self._lu = BackendDenseLU(bkd)
        self.setups = 0

    def capabilities(self) -> SolverCapabilities:
        return self._lu.capabilities()

    def setup(
        self, op: ActionOperatorProtocol[_Np, DenseFormat[_Np]]
    ) -> SolverHandleProtocol[_Np]:
        self.setups += 1
        return self._lu.setup(op)


_StrategyOf = Callable[
    [LinearSolverProtocol[_Np, DenseFormat[_Np]]],
    MassShiftStrategy[_Np, _Np, DenseFormat[_Np]],
]


class _Run:
    """The linear ODE stepped by one tableau under one setup strategy."""

    def __init__(
        self,
        bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        strategy_of: _StrategyOf,
    ) -> None:
        amat = bkd.asarray([[-1.0, 0.3, 0.0], [0.2, -0.8, 0.1], [0.0, 0.4, -1.2]])
        bmat = bkd.asarray([[1.0, 0.0], [0.5, -0.2], [0.0, 1.0]])
        self.bkd = bkd
        self.residual = LinearODEResidual(amat, bmat, bkd)
        self.param = bkd.asarray([0.3, -0.7])
        self.residual.set_param(self.param)
        self.lu = _CountingLU(bkd)
        newton: Newton[_Np, _Np] = Newton(atol=1e-13, rtol=0.0, maxiters=20)
        self.stepper = RungeKuttaStepper(
            make(),
            dense_mass_form_system(self.residual, self.lu),
            strategy_of(self.lu),
            newton,
        )
        self.store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
        Integrator(self.stepper).solve(
            bkd.asarray([[1.0], [-0.5], [2.0]]), _STEPS, self.store
        )
        self.sweeps: TrajectoryDerivatives[_Np, _Np] = TrajectoryDerivatives(
            self.stepper,
            None,
            ResidualInitialState(self.residual, BackendColumnForm(bkd)),
        )
        observed = [(k, (2, _NSTEPS)) for k in range(3)]
        misfit: MisfitTerm[_Np, _Np] = MisfitTerm.from_state_times(
            bkd,
            3,
            observed,
            bkd.asarray([0.3 * (k + 1) for k in range(6)]),
            noise_std=0.5,
        )
        self.misfit: TrajectoryFunctional[_Np, _Np] = TrajectoryFunctional.from_terms(
            bkd, 1, state_terms=(misfit,)
        )

    def gradient(self) -> _Np:
        return self.sweeps.gradient(self.store, self.misfit)

    def hvp(self) -> _Np:
        direction = self.bkd.asarray([[0.4], [-1.1]])
        return self.sweeps.hvp(self.store, self.misfit, direction)

    def counted(self) -> Tuple[Tuple[int, int, int], _Np, _Np]:
        """The cumulative setups after the solve, the gradient and the
        HVP, and the gradient and HVP."""
        solved = self.lu.setups
        gradient = self.gradient()
        after_gradient = self.lu.setups
        hvp = self.hvp()
        return (solved, after_gradient, self.lu.setups), gradient, hvp


# Each tableau with its solves per step (implicit stages; for Heun its
# explicit stage and final block, each a mass solve) and the distinct
# operators a keeping strategy sets up over the run (one stage operator
# per step size; Heun's mass alone).
_TABLEAUX = [
    (backward_euler, 1, 2),
    (crank_nicolson, 1, 2),
    (alexander_sdirk3, 3, 2),
    (heun, 2, 1),
]
_IDS = ["backward_euler", "crank_nicolson", "sdirk3", "heun"]
_KEEPING: List[_StrategyOf] = [KeepStateFree, KeepAll]


class TestSweepReuse:
    @pytest.mark.parametrize(
        "strategy_of", _KEEPING, ids=["keep_state_free", "keep_all"]
    )
    @pytest.mark.parametrize("make, solves, distinct", _TABLEAUX, ids=_IDS)
    def test_the_sweeps_reuse_the_forward_setups(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        solves: int,
        distinct: int,
        strategy_of: _StrategyOf,
    ) -> None:
        assert len(set(_STEPS.sizes())) == 2
        counts, gradient, hvp = _Run(numpy_bkd, make, strategy_of).counted()
        assert counts == (distinct, distinct, distinct)
        _, expected_gradient, expected_hvp = _Run(
            numpy_bkd, make, RecomputeAlways
        ).counted()
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(hvp, expected_hvp, rtol=1e-13, atol=1e-15)

    @pytest.mark.parametrize("make, solves, distinct", _TABLEAUX, ids=_IDS)
    def test_recompute_always_factors_every_solve(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        solves: int,
        distinct: int,
    ) -> None:
        # One setup per solve per step, in each pass: the sweeps' diagonal
        # blocks are the forward solves' operators.
        per_pass = _NSTEPS * solves
        counts, _, _ = _Run(numpy_bkd, make, RecomputeAlways).counted()
        assert counts == (per_pass, 2 * per_pass, 4 * per_pass)
