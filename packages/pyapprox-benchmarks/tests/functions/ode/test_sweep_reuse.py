r"""Factorization reuse across a solve and its derivative sweeps.

The linear ODE :math:`\dot y = A y + B p` (3 states, 2 parameters, its
Jacobian :math:`A` declared constant), from (1, -0.5, 2), 10 steps of
0.1 to T = 1 (``FixedStep``: the last step lands on 1 with a size of its
own, so a step forms two different stage operators over the run), an MSE
with noise 0.5 observing every state at step 2 and the last. The stepper's solver
counts its setups.

- Backward Euler, Crank-Nicolson and SDIRK3 factor their stage operator
  once per step size: 2 setups for the forward solve, 2 for the gradient's
  adjoint sweep, and 2 for the HVP's tangent and two adjoint sweeps
  together, which share one session.
- With ``NoReuse`` every implicit stage is factored afresh in every pass
  (steps times implicit stages for the forward solve and for the
  gradient's sweep, twice that for the HVP: the tangent sweep, and the
  adjoints sharing one handle per step), and the gradient and HVP equal
  the reused ones to 1e-13.

The default keeps two step sizes' operators: the tangent sweep ends on
the last step's size and the adjoints start there, so with one the 0.1
operator would be factored a third time.
"""

from typing import Any, Callable, Optional

import numpy as np
import pytest

from pyapprox.ode.functionals.mse import TransientMSEFunctional
from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.tableau import (
    ButcherTableau,
    alexander_sdirk3,
    backward_euler,
    crank_nicolson,
)
from pyapprox.ode.trajectory_derivatives import (
    ResidualInitialState,
    TrajectoryDerivatives,
)
from pyapprox.ode.trajectory_functional import (
    BackendSecondOrderTrajectoryFunctional,
    BackendTrajectoryFunctional,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    NoReuse,
    ReusePolicyProtocol,
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


class _Run:
    """The linear ODE stepped by one tableau under one reuse policy."""

    def __init__(
        self,
        bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        reuse: Optional[ReusePolicyProtocol] = None,
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
            make(), dense_mass_form_system(self.residual, self.lu), newton, None, reuse
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
        observed = [(k, bkd.asarray([2, _NSTEPS], dtype=int)) for k in range(3)]
        mse = TransientMSEFunctional(3, 2, observed, bkd, noise_std=0.5)
        mse.set_observations(bkd.asarray([0.3 * (k + 1) for k in range(6)]))
        self.mse = mse

    def gradient(self) -> _Np:
        column = self.bkd.reshape(self.param, (-1, 1))
        coupling = BackendTrajectoryFunctional(self.mse, self.store, column)
        return self.sweeps.gradient(self.store, coupling)

    def hvp(self) -> _Np:
        column = self.bkd.reshape(self.param, (-1, 1))
        coupling = BackendSecondOrderTrajectoryFunctional(self.mse, self.store, column)
        direction = self.bkd.asarray([[0.4], [-1.1]])
        return self.sweeps.hvp(self.store, coupling, direction)


_TABLEAUX = [(backward_euler, 1), (crank_nicolson, 1), (alexander_sdirk3, 3)]
_IDS = ["backward_euler", "crank_nicolson", "sdirk3"]


class TestSweepReuse:
    @pytest.mark.parametrize("make, implicit_stages", _TABLEAUX, ids=_IDS)
    def test_a_constant_jacobian_is_factored_once_per_size_per_session(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        implicit_stages: int,
    ) -> None:
        assert len(set(_STEPS.sizes())) == 2
        run = _Run(numpy_bkd, make)
        assert run.lu.setups == 2
        run.gradient()
        assert run.lu.setups == 4
        run.hvp()
        assert run.lu.setups == 6

    @pytest.mark.parametrize("make, implicit_stages", _TABLEAUX, ids=_IDS)
    def test_no_reuse_factors_every_solve_and_changes_nothing(
        self,
        numpy_bkd: NumpyBkd,
        make: Callable[[], ButcherTableau],
        implicit_stages: int,
    ) -> None:
        kept = _Run(numpy_bkd, make)
        run = _Run(numpy_bkd, make, NoReuse())
        # One setup per implicit stage per step, in each pass: the sweeps'
        # diagonal blocks are the stages'.
        per_pass = _NSTEPS * implicit_stages
        assert run.lu.setups == per_pass
        gradient = run.gradient()
        assert run.lu.setups == 2 * per_pass
        hvp = run.hvp()
        assert run.lu.setups == 4 * per_pass
        np.testing.assert_allclose(gradient, kept.gradient(), rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(hvp, kept.hvp(), rtol=1e-13, atol=1e-15)
