r"""A trajectory kept in ``.npz`` files against the same one in memory.

Lotka-Volterra with mass (``tests._helpers.ode.trajectory_cases``), Heun
from 0 to 0.5 in steps of 0.05, with a curved time-dependent constraint
at the stored states and a parameter-dependent initial state, so the
store keeps unconstrained values and two-block unknowns. One run writes
both stores through a multi-sink; they must agree window for window
(states, unconstrained values, spans, unknowns, versions, past states),
and the adjoint gradient and the Hessian-vector product read from the
file store, its tangent trajectory also kept in files, must equal the
in-memory ones to 1e-14.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pyapprox.ode.integration import ResidualFormConstraint
from pyapprox.ode.npz_trajectory_store import NpzTrajectoryStore
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import heun
from pyapprox.ode.trajectory_derivatives import TrajectoryDerivatives
from pyapprox.ode.trajectory_store import (
    TrajectoryStoreProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import BlockVector
from tests._helpers.ode.runge_kutta_cases import LV_PARAM
from tests._helpers.ode.trajectory_cases import (
    CurvedPin,
    ScaledStart,
    TrajectoryProblem,
    random_direction,
)


def _problem(bkd: Backend[Array]) -> TrajectoryProblem[Array]:
    return TrajectoryProblem(bkd, heun(), CurvedPin(bkd), ScaledStart(bkd))


def _sweeps(
    bkd: Backend[Array], problem: TrajectoryProblem[Array]
) -> TrajectoryDerivatives[Array, Array]:
    return TrajectoryDerivatives(
        problem.stepper(),
        ResidualFormConstraint(problem.pin(), BackendColumnForm(bkd)),
        problem.start(),
    )


def test_files_hold_the_same_trajectory(bkd: Backend[Array], tmp_path: Path) -> None:
    problem = _problem(bkd)
    files: NpzTrajectoryStore[Array] = NpzTrajectoryStore(str(tmp_path / "run"), bkd)
    assert isinstance(files, TrajectoryStoreProtocol)
    memory = problem.solve(bkd.asarray(LV_PARAM), files)
    assert files.ntimes() == memory.ntimes() and files.nsteps() == memory.nsteps()
    assert files.times() == memory.times()
    for index in range(memory.ntimes()):
        bkd.assert_allclose(files.state(index), memory.state(index), atol=0.0)
        bkd.assert_allclose(
            files.unconstrained(index), memory.unconstrained(index), atol=0.0
        )
        assert files.time(index) == memory.time(index)
    for step in range(memory.nsteps()):
        got, want = files.window(step), memory.window(step)
        assert got.span == want.span and files.span(step) == memory.span(step)
        assert got.parameters_version == want.parameters_version
        assert len(got.unknown.blocks) == len(want.unknown.blocks) == 2
        for a, b in zip(got.unknown.blocks, want.unknown.blocks):
            bkd.assert_allclose(a, b, atol=0.0)
        assert len(got.past_states) == len(want.past_states)
        for a, b in zip(got.past_states, want.past_states):
            bkd.assert_allclose(a, b, atol=0.0)
        # A window names its step of this store, the same on every read.
        assert got.origin == files.window(step).origin
        assert got.origin is not None and got.origin.step() == step
    # One self-describing file per state.
    names = sorted(os.listdir(files.directory()))
    assert len(names) == memory.ntimes()
    with np.load(os.path.join(files.directory(), names[-1])) as data:
        assert {"state", "unconstrained", "unknown_0", "unknown_1"} <= set(data)
        assert float(data["size"]) == memory.span(-1).deltat()
        assert int(data["nread"]) == 1


def test_sweeps_read_the_files(bkd: Backend[Array], tmp_path: Path) -> None:
    problem = _problem(bkd)
    param, direction = bkd.asarray(LV_PARAM), random_direction(bkd)
    files: NpzTrajectoryStore[Array] = NpzTrajectoryStore(str(tmp_path / "run"), bkd)
    memory = problem.solve(param, files)
    sweeps = _sweeps(bkd, problem)
    functional = problem.coupling(memory, param)
    bkd.assert_allclose(
        sweeps.gradient(files, functional),
        sweeps.gradient(memory, functional),
        rtol=1e-14,
    )
    tangents: NpzTrajectoryStore[Array] = NpzTrajectoryStore(
        str(tmp_path / "tangents"), bkd
    )
    bkd.assert_allclose(
        sweeps.hvp(files, functional, direction, tangent_store=tangents),
        sweeps.hvp(memory, functional, direction),
        rtol=1e-14,
    )
    assert tangents.ntimes() == memory.ntimes()


def test_invalid_use(bkd: Backend[Array], tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "other.txt").write_text("kept")
    with pytest.raises(ValueError, match="must hold no files"):
        NpzTrajectoryStore(str(occupied), bkd)
    store: NpzTrajectoryStore[Array] = NpzTrajectoryStore(str(tmp_path / "s"), bkd)
    state = bkd.asarray([[1.0], [2.0]])
    window = StepWindow(StepSpan(0.0, (0.1,)), (state,), BlockVector((state,)))
    with pytest.raises(ValueError, match="append one first"):
        store.append_step(window, state)
    store.append(0.0, state)
    with pytest.raises(ValueError, match="only the initial state"):
        store.append(0.0, state)
    late = StepWindow(StepSpan(0.5, (0.1,)), (state,), BlockVector((state,)))
    with pytest.raises(ValueError, match="last stored time"):
        store.append_step(late, state)
    deep = StepWindow(StepSpan(-0.1, (0.1, 0.1)), (state, state), BlockVector((state,)))
    with pytest.raises(ValueError, match="reads 2 states"):
        store.append_step(deep, state)
    store.append_step(window, 2.0 * state)
    for read in (store.state, store.unconstrained):
        with pytest.raises(IndexError):
            read(2)
    for step_read in (store.span, store.window):
        with pytest.raises(IndexError):
            step_read(1)
    bkd.assert_allclose(store.unconstrained(1), 2.0 * state, atol=0.0)
    bkd.assert_allclose(store.state(-1), 2.0 * state, atol=0.0)
