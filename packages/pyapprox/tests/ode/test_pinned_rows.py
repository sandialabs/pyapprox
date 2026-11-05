r"""Rows pinned to boundary data, inside the step's solves or after it.

Lotka-Volterra with the mass ``[[2, 0.5], [0.5, 1]]`` from (10, 5),
parameters (1, 0.1, 0.075, 1.5), dt 0.05 to 0.5, Newton tolerance 1e-13,
with the prey pinned to :math:`g(t) = 10 + \sin 3t` (equal to the start
at t = 0, and moving).

- Pinned inside the solve at the step end (:class:`RowPinning` with
  :class:`AtStepEnd`), for every Runge-Kutta tableau and multistep method:
  each stored state holds the data to 1e-12; the step residual vanishes
  at every window; the stepper checker passes at a step's window, second
  derivatives included; and the trajectory's gradient and Hessian-vector
  product of the Tikhonov-augmented MSE pass ``DerivativeChecker``, the
  tangent equal to the gradient in a direction.
- A placement written here, pinning every block (every stage and the
  step end), is injected with no stepper change: each pinned stage holds
  the data at its stage time, and the checker and finite differences
  pass.
- The continuous extension is pinned as the block producing the new
  state is, at its own time: under either placement, for backward Euler,
  Heun, SDIRK3 and every multistep method, its prey row holds the data
  at 0.3 and 0.7 of every step (to 1e-12); a misfit of both states at
  0.137 and 0.41 read at the extension passes the term checker with
  ``fd_tol`` 3e-5 (see ``test_terms_at_times``).
- Pinned after the step (:class:`PinnedRowsConstraint`): stored states
  hold the data, and the gradient and HVP pass finite differences.
- Refusals.
"""

from typing import Callable, Generic, List, Optional

import numpy as np
import pytest

from pyapprox.ode.integration import Integrator
from pyapprox.ode.mass_form_system import dense_mass_form_system
from pyapprox.ode.observation_terms import MisfitTerm, StateEntries
from pyapprox.ode.pinned_rows import (
    AtStepEnd,
    BlockPinProtocol,
    PinnedRowsConstraint,
    PinnedRowsProtocol,
    PinPlacementProtocol,
    RowPinning,
    RowPinningProtocol,
)
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.stepper_derivatives_check import StepperBundleChecker
from pyapprox.ode.tableau import ButcherTableau, alexander_sdirk3, backward_euler, heun
from pyapprox.ode.terms_at_times import TermAtTimes
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.ode.trajectory_terms import TrajectoryFunctional
from pyapprox.ode.trajectory_terms_check import TrajectoryTermChecker
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet
from pyapprox.util.linalg.operators import BlockVector
from tests._helpers.ode.runge_kutta_cases import (
    LV_PARAM,
    backend_sample,
    lotka_volterra_with_mass,
)
from tests._helpers.ode.trajectory_cases import (
    ALL_IDS,
    ALL_METHODS,
    MULTISTEP,
    STEPS,
    Method,
    Stepper,
    TrajectoryProblem,
    make_stepper,
    min_gradient_fd_error,
    min_hvp_fd_error,
    random_direction,
)


class _Prey(Generic[Array]):
    r"""The prey pinned to :math:`g(t) = 10 + \sin 3t`."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        states, _ = backend_spaces(bkd, 2, 4)
        self._rows: BackendRowSet[Array] = BackendRowSet(states.descriptor(), [0], bkd)

    def rows(self) -> BackendRowSet[Array]:
        return self._rows

    def data(self, time: float) -> float:
        return 10.0 + float(np.sin(3.0 * time))

    def values(self, time: float) -> Array:
        return self._bkd.asarray([[self.data(time)], [0.0]])


class _EveryBlock:
    """Pins every block: each stage and the step end."""

    def pins(self, block: int, nblocks: int) -> bool:
        return True


_SOME: List[Callable[[], ButcherTableau]] = [backward_euler, heun, alexander_sdirk3]
_SOME_IDS = [make().name for make in _SOME]
# The tableaux above and every multistep method, for the continuous
# extension.
_EXTENDED: List[Callable[[], Method]] = [*_SOME, *MULTISTEP]
_EXTENDED_IDS = [make().name for make in _EXTENDED]


class _Pinned(Generic[Array]):
    """A pinned stepper on Lotka-Volterra at the nominal parameters, and
    its trajectory.

    Parameters
    ----------
    bkd : Backend
        The backend.
    method : ButcherTableau or MultistepFormulaProtocol
        The method.
    pinning : RowPinningProtocol
        The pinning.
    """

    def __init__(
        self, bkd: Backend[Array], method: Method, pinning: RowPinningProtocol[Array]
    ) -> None:
        self.model = lotka_volterra_with_mass(bkd)
        self.param = bkd.asarray(LV_PARAM)
        self.model.set_param(self.param)
        self.stepper: Stepper[Array] = make_stepper(
            method, dense_mass_form_system(self.model), pinning
        )
        self.store: InMemoryTrajectoryStore[Array] = InMemoryTrajectoryStore()
        Integrator(self.stepper).solve(bkd.asarray([[10.0], [5.0]]), STEPS, self.store)

    def window(self, step: int) -> StepWindow[Array]:
        return self.store.window(step, min(self.stepper.depth(), step + 1))


class TestPinnedInsideTheSolve:
    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_stored_states_hold_the_data(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        prey = _Prey(bkd)
        store = _Pinned(bkd, make(), RowPinning(prey)).store
        for n in range(1, store.ntimes()):
            value = bkd.to_float(store.state(n)[0, 0])
            assert abs(value - prey.data(store.time(n))) <= 1e-12
        # The other row is solved, not pinned.
        assert abs(bkd.to_float(store.state(-1)[1, 0]) - 5.0) > 1e-3

    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_the_step_residual_vanishes_and_the_checker_passes(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        pinned: _Pinned[Array] = _Pinned(bkd, make(), RowPinning(_Prey(bkd)))
        stepper = pinned.stepper
        for n in range(pinned.store.nsteps()):
            residual = stepper.step_residual(pinned.window(n))
            norm = stepper.residual_space().norm(residual)
            assert bkd.to_float(norm[0, 0]) < 1e-10
        sample = backend_sample(bkd)
        space = stepper.residual_space()
        adj = BlockVector(
            tuple(
                sample(space.component(i).descriptor(), 99 + i)
                for i in range(space.nblocks())
            )
        )
        checker = StepperBundleChecker(
            stepper, sample, set_param=pinned.model.set_param
        )
        report = checker.check(pinned.window(3), adj, pinned.param)
        assert report.failures() == []

    @pytest.mark.parametrize("make", ALL_METHODS, ids=ALL_IDS)
    def test_gradient_and_hvp(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        problem: TrajectoryProblem[Array] = TrajectoryProblem(
            bkd, make(), row_pinning=RowPinning(_Prey(bkd))
        )
        assert min_gradient_fd_error(bkd, problem) < 1e-7
        assert min_hvp_fd_error(bkd, problem) < 1e-7
        param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
        bkd.assert_allclose(
            problem.tangent(param, direction),
            problem.gradient(param) @ direction,
            rtol=1e-12,
        )


class TestAnotherPlacement:
    """A placement this module defines, pinning every block, needs no
    stepper change."""

    @pytest.mark.parametrize("make", _SOME, ids=_SOME_IDS)
    def test_every_pinned_block_holds_the_data(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        prey = _Prey(bkd)
        tableau = make()
        store = _Pinned(bkd, tableau, RowPinning(prey, _EveryBlock())).store
        # Block r is unknown stage r + offset, then the final block.
        offset = 1 if tableau.has_trivial_first_stage() else 0
        stages = list(range(offset, tableau.nstages()))
        for n in range(store.nsteps()):
            window = store.window(n)
            for r, block in enumerate(window.unknown.blocks):
                time = (
                    window.time() + tableau.c[stages[r]] * window.deltat()
                    if r < len(stages)
                    else window.time() + window.deltat()
                )
                assert abs(bkd.to_float(block[0, 0]) - prey.data(time)) <= 1e-12

    @pytest.mark.parametrize("make", _SOME, ids=_SOME_IDS)
    def test_derivatives(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        pinning: RowPinning[Array] = RowPinning(_Prey(bkd), _EveryBlock())
        problem: TrajectoryProblem[Array] = TrajectoryProblem(
            bkd, make(), row_pinning=pinning
        )
        assert min_gradient_fd_error(bkd, problem) < 1e-7
        assert min_hvp_fd_error(bkd, problem) < 1e-7


class TestTheContinuousExtension:
    """The extension is pinned as the block producing the new state is,
    at its own time."""

    @pytest.mark.parametrize(
        "placement", [AtStepEnd(), _EveryBlock()], ids=["step end", "every block"]
    )
    @pytest.mark.parametrize("make", _EXTENDED, ids=_EXTENDED_IDS)
    def test_it_holds_the_data_between_steps(
        self,
        bkd: Backend[Array],
        make: Callable[[], Method],
        placement: PinPlacementProtocol,
    ) -> None:
        prey = _Prey(bkd)
        pinned = _Pinned(bkd, make(), RowPinning(prey, placement))
        extension = pinned.stepper.continuous_extension()
        store = pinned.store
        for n in range(store.nsteps()):
            window = pinned.window(n)
            for theta in (0.3, 0.7):
                time = window.time() + theta * window.deltat()
                value = bkd.to_float(extension.value(window, time)[0, 0])
                assert abs(value - prey.data(time)) <= 1e-12

    @pytest.mark.parametrize("make", _EXTENDED, ids=_EXTENDED_IDS)
    def test_a_misfit_between_steps_passes_the_checker(
        self, bkd: Backend[Array], make: Callable[[], Method]
    ) -> None:
        """Both states observed at 0.137 and 0.41: the pinned prey row
        reads the data, so its derivatives are zero there."""
        pinned: _Pinned[Array] = _Pinned(bkd, make(), RowPinning(_Prey(bkd)))
        stepper = pinned.stepper
        entries = StateEntries(bkd, 2, [0, 1])
        term: MisfitTerm[Array, Array] = MisfitTerm(
            bkd,
            {
                0: (entries, bkd.asarray([[9.0], [5.0]])),
                1: (entries, bkd.asarray([[8.0], [6.0]])),
            },
            0.5,
        )
        functional = TrajectoryFunctional.from_terms(
            bkd,
            1,
            step_terms=(
                TermAtTimes(term, stepper.continuous_extension(), (0.137, 0.41), bkd),
            ),
        )
        checker: TrajectoryTermChecker[Array, Array] = TrajectoryTermChecker(
            stepper, bkd, backend_sample(bkd), pinned.model.set_param
        )
        report = checker.check(functional, pinned.store, param=pinned.param)
        # As in test_terms_at_times: correct forward V's whose bottoms the
        # weak coupling h b(theta) raises; every V test still holds.
        assert report.failures(fd_tol=3e-5) == []


class TestPinnedAfterTheStep:
    @pytest.mark.parametrize("make", _SOME, ids=_SOME_IDS)
    def test_stored_states_and_derivatives(
        self, bkd: Backend[Array], make: Callable[[], ButcherTableau]
    ) -> None:
        prey = _Prey(bkd)
        states, params = backend_spaces(bkd, 2, 4)
        constraint = PinnedRowsConstraint(prey, states, params)
        problem = TrajectoryProblem(bkd, make(), stack_constraint=constraint)
        store = problem.solve(bkd.asarray(LV_PARAM))
        for n in range(store.ntimes()):
            value = bkd.to_float(store.state(n)[0, 0])
            assert value == pytest.approx(prey.data(store.time(n)), abs=1e-14)
        assert min_gradient_fd_error(bkd, problem) < 1e-7
        assert min_hvp_fd_error(bkd, problem) < 1e-7


class TestProtocols:
    def test_the_pieces_satisfy_them(self, bkd: Backend[Array]) -> None:
        prey = _Prey(bkd)
        assert isinstance(prey, PinnedRowsProtocol)
        pinning = RowPinning(prey)
        assert isinstance(pinning, RowPinningProtocol)
        assert pinning.at(0, 2) is None
        pin: Optional[BlockPinProtocol[Array]] = pinning.at(1, 2)
        assert isinstance(pin, BlockPinProtocol)
        assert isinstance(AtStepEnd(), type(AtStepEnd()))

    def test_refusals(self, bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="PinnedRowsProtocol"):
            RowPinning(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="PinPlacementProtocol"):
            RowPinning(_Prey(bkd), object())  # type: ignore[arg-type]
        model = lotka_volterra_with_mass(bkd)
        with pytest.raises(TypeError, match="RowPinningProtocol"):
            make_stepper(backward_euler(), dense_mass_form_system(model), object())  # type: ignore[arg-type]
        other, _ = backend_spaces(bkd, 3, 4)
        with pytest.raises(ValueError):
            PinnedRowsConstraint[Array](_Prey(bkd), other)
