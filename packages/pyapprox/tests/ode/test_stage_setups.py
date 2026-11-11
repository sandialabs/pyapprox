r"""Stage setups shared by the forward solve and the sweeps, by position.

Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` at ``LV_PARAM``
from (10, 5), ten steps of 0.05, under the Tikhonov-augmented MSE of
``native_tikhonov_mse``; backward Euler (one implicit stage), SDIRK3
(three), both stiffly accurate, and BDF2 and Adams-Moulton 3 (one
implicit solve per step, startup steps included), so every setup is an
implicit solve's. The Jacobian sees the state, so a forward Newton
iterate's setup is never kept; a sweep's diagonal block is named by its
window's origin and block. Setups added by a gradient, then two HVPs
(each a tangent and an adjoint sweep), with ``s`` implicit solves per
step:

- ``KeepAll``: the gradient 10 s, each HVP none (every sweep reads the
  same positions); the same with a store copying every state it gives
  out; with windows carrying no origin, as ``RecomputeAlways``.
- ``RecomputeAlways`` (and ``KeepStateFree``, which keeps nothing here):
  the gradient 10 s, each HVP 20 s.
- Gradients and HVPs equal across strategies and stores to 1e-13.
- Kept setups name the trajectory, not the store: a dropped store is
  freed while its setups stay kept.
- After a solve and its sweeps, an SDIRK3 stepper pickles under every
  strategy, and the copy steps the same trajectory (1e-14).
"""

import gc
import pickle
import warnings
import weakref
from dataclasses import dataclass
from typing import Callable, Generic, List, Tuple, Union

import pytest

from pyapprox.implicitfunction.setups import KeepAll, KeepStateFree, RecomputeAlways
from pyapprox.ode.integration import Integrator
from pyapprox.ode.linear_multistep import LinearMultistepStepper
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.multistep_methods import (
    MultistepFormulaProtocol,
    adams_moulton3,
    bdf2,
)
from pyapprox.ode.runge_kutta import RungeKuttaStepper
from pyapprox.ode.setups import MassShiftStrategy
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import ButcherTableau, alexander_sdirk3, backward_euler
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU, DenseFormat
from pyapprox.util.linalg.operators import BlockVector, LinearSolverProtocol
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.linalg.counting_solver import CountingSolver
from tests._helpers.ode.runge_kutta_cases import LV_PARAM, lotka_volterra_with_mass
from tests._helpers.ode.trajectory_cases import (
    START,
    STEPS,
    Stepper,
    native_tikhonov_mse,
)

_Strategy = MassShiftStrategy[Array, Array, DenseFormat[Array]]
_StrategyOf = Callable[
    [LinearSolverProtocol[Array, DenseFormat[Array]]], _Strategy[Array]
]
_NSTEPS = len(STEPS.sizes())
_Method = Union[ButcherTableau, MultistepFormulaProtocol]
# Each method with its implicit solves per step: a tableau's implicit
# stages, an implicit multistep formula's one (its startup steps too).
_TABLEAUS: List[Tuple[Callable[[], _Method], int]] = [
    (backward_euler, 1),
    (alexander_sdirk3, 3),
    (bdf2, 1),
    (adams_moulton3, 1),
]
_TABLEAU_IDS = ["backward_euler", "sdirk3", "bdf2", "adams_moulton3"]


class _CopyingStore(InMemoryTrajectoryStore[Array]):
    """Gives out a new copy of every state and unknown on each read, as a
    store recomputing them from checkpoints would; the origin is kept."""

    def __init__(self, bkd: Backend[Array]) -> None:
        super().__init__()
        self._bkd = bkd

    def state(self, index: int) -> Array:
        return self._bkd.copy(super().state(index))

    def window(self, step: int, depth: int = 1) -> StepWindow[Array]:
        window = super().window(step, depth)
        copy = self._bkd.copy
        return StepWindow(
            window.span,
            tuple(copy(x) for x in window.past_states),
            BlockVector(tuple(copy(b) for b in window.unknown.blocks)),
            window.parameters_version,
            window.origin,
        )


class _UnnamedStore(InMemoryTrajectoryStore[Array]):
    """Gives out windows naming no origin, as a window built by hand."""

    def window(self, step: int, depth: int = 1) -> StepWindow[Array]:
        window = super().window(step, depth)
        return StepWindow(
            window.span, window.past_states, window.unknown, window.parameters_version
        )


class _Run(Generic[Array]):
    """One stepper under one strategy; its solver counts setups."""

    def __init__(
        self,
        bkd: Backend[Array],
        make: Callable[[], _Method],
        strategy_of: _StrategyOf[Array],
    ) -> None:
        self.bkd = bkd
        model = lotka_volterra_with_mass(bkd)
        model.set_param(bkd.asarray(LV_PARAM))
        self.lu: CountingSolver[Array, DenseFormat[Array]] = CountingSolver(
            BackendDenseLU(bkd)
        )
        self.strategy = strategy_of(self.lu)
        newton: Newton[Array, Array] = Newton(atol=1e-13, rtol=0.0, maxiters=30)
        method, system = make(), dense_mass_form_system(model, self.lu)
        self.stepper: Stepper[Array] = (
            RungeKuttaStepper(method, system, self.strategy, newton)
            if isinstance(method, ButcherTableau)
            else LinearMultistepStepper(method, system, self.strategy, newton)
        )
        self.sweeps: TrajectoryDerivatives[Array, Array] = TrajectoryDerivatives(
            self.stepper, None, None
        )
        self.functional, tikhonov = native_tikhonov_mse(bkd)
        tikhonov.set_param(bkd.asarray(LV_PARAM))

    def sequence(
        self, store: InMemoryTrajectoryStore[Array], direction: Array
    ) -> "_Result[Array]":
        """Forward, gradient, two HVPs: the setups each sweep added, the
        gradient and the first HVP."""
        Integrator(self.stepper).solve(self.bkd.asarray(START), STEPS, store)
        before = self.lu.setups
        gradient = self.sweeps.gradient(store, self.functional)
        counts = [self.lu.setups - before]
        hvps: List[Array] = []
        for _ in range(2):
            before = self.lu.setups
            hvps.append(self.sweeps.hvp(store, self.functional, direction))
            counts.append(self.lu.setups - before)
        return _Result(tuple(counts), gradient, hvps[0])


@dataclass(frozen=True)
class _Result(Generic[Array]):
    sweep_setups: Tuple[int, ...]
    gradient: Array
    hvp: Array


def _direction(bkd: Backend[Array]) -> Array:
    return bkd.asarray([[0.3], [-0.2], [0.1], [0.4]])


class TestStageSetups:
    @pytest.mark.parametrize("make, stages", _TABLEAUS, ids=_TABLEAU_IDS)
    def test_kept_by_position_across_sweeps(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau], stages: int
    ) -> None:
        direction = _direction(bkd)
        per_sweep = _NSTEPS * stages
        recomputed = _Run(bkd, make, RecomputeAlways).sequence(
            InMemoryTrajectoryStore(), direction
        )
        assert recomputed.sweep_setups == (per_sweep, 2 * per_sweep, 2 * per_sweep)
        cases: List[
            Tuple[str, _StrategyOf[Array], InMemoryTrajectoryStore[Array], bool]
        ] = [
            ("KeepAll", KeepAll, InMemoryTrajectoryStore(), True),
            ("KeepAll, copying store", KeepAll, _CopyingStore(bkd), True),
            ("KeepAll, no origin", KeepAll, _UnnamedStore(), False),
            ("KeepStateFree", KeepStateFree, InMemoryTrajectoryStore(), False),
        ]
        for label, strategy_of, store, kept in cases:
            result = _Run(bkd, make, strategy_of).sequence(store, direction)
            expected = (per_sweep, 0, 0) if kept else recomputed.sweep_setups
            assert result.sweep_setups == expected, label
            bkd.assert_allclose(result.gradient, recomputed.gradient, rtol=1e-13)
            bkd.assert_allclose(result.hvp, recomputed.hvp, rtol=1e-13)

    @pytest.mark.parametrize(
        "strategy_of",
        [RecomputeAlways, KeepStateFree, KeepAll],
        ids=["recompute_always", "keep_state_free", "keep_all"],
    )
    def test_a_stepper_pickles_after_a_solve(
        self, bkd: Backend[Array], strategy_of: _StrategyOf[Array]
    ) -> None:
        """The stepper holds no setup of its own, so after a solve and
        its sweeps it pickles with its strategy, and the copy steps the
        same trajectory."""
        run = _Run(bkd, alexander_sdirk3, strategy_of)
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        run.sequence(store, _direction(bkd))
        # Deprecated pickling (an itertools counter, say) fails here
        # rather than on a later Python.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            copy = pickle.loads(pickle.dumps(run.stepper))
        again: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(copy).solve(bkd.asarray(START), STEPS, again)
        for n in range(store.ntimes()):
            bkd.assert_allclose(again.state(n), store.state(n), rtol=1e-14)

    def test_kept_setups_do_not_hold_the_store(self, bkd: Backend[Array]) -> None:
        strategy: KeepAll[Array, DenseFormat[Array]] = KeepAll(BackendDenseLU(bkd))
        run = _Run(bkd, backward_euler, lambda solver: strategy)
        store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        run.sequence(store, _direction(bkd))
        assert strategy.nkept() == _NSTEPS
        reference = weakref.ref(store)
        del store
        gc.collect()
        assert reference() is None
        assert strategy.nkept() == _NSTEPS
