r"""Full-order snapshots and the parameters and times they came from.

A projection reduced-order model is fitted to states of the full-order
model. Each state is only meaningful together with the parameter that
produced it and, for a transient solve, the time it belongs to: the
boundary lift is evaluated at that time, and hyper-reduction trains on
residuals evaluated at that parameter and time. :class:`SnapshotSet`
keeps the three together, one column per snapshot, so they cannot be
paired up wrongly downstream.

Snapshots are collected through protocols every steady state equation
and every transient forward model already satisfies, so nothing here
depends on the discretization that produced them.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, Sequence, Tuple, runtime_checkable

from pyapprox.optimization.implicitfunction.state_equations.protocols import (
    SteadyStateEquationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class SnapshotSet(Generic[Array]):
    """Snapshots with the parameter and time of each column.

    Parameters
    ----------
    states : Array
        Full-order states. Shape: (nstates, nsnapshots).
    params : Array
        The parameter each state was solved at. Shape:
        (nparams, nsnapshots).
    times : Array, optional
        The time of each state. Shape: (nsnapshots,). ``None`` for
        steady snapshots, which belong to no time.

    Raises
    ------
    ValueError
        If the arrays are not 2D (``times`` 1D) or disagree on the
        number of snapshots.
    """

    states: Array
    params: Array
    times: Optional[Array] = None

    def __post_init__(self) -> None:
        if self.states.ndim != 2:
            raise ValueError(
                f"states must be 2D (nstates, nsnapshots), got ndim={self.states.ndim}"
            )
        if self.params.ndim != 2:
            raise ValueError(
                f"params must be 2D (nparams, nsnapshots), got ndim={self.params.ndim}"
            )
        nsnapshots = self.states.shape[1]
        if self.params.shape[1] != nsnapshots:
            raise ValueError(
                f"params has {self.params.shape[1]} columns but states has {nsnapshots}"
            )
        if self.times is not None:
            if self.times.ndim != 1:
                raise ValueError(
                    f"times must be 1D (nsnapshots,), got ndim={self.times.ndim}"
                )
            if self.times.shape[0] != nsnapshots:
                raise ValueError(
                    f"times has {self.times.shape[0]} entries but states "
                    f"has {nsnapshots} columns"
                )

    def nstates(self) -> int:
        """Return the number of states per snapshot."""
        return int(self.states.shape[0])

    def nparams(self) -> int:
        """Return the number of parameters."""
        return int(self.params.shape[0])

    def nsnapshots(self) -> int:
        """Return the number of snapshots."""
        return int(self.states.shape[1])

    def is_steady(self) -> bool:
        """Whether the snapshots belong to no time."""
        return self.times is None


def concatenate_snapshot_sets(
    snapshot_sets: Sequence[SnapshotSet[Array]], bkd: Backend[Array]
) -> SnapshotSet[Array]:
    """Join snapshot sets column-wise.

    Raises
    ------
    ValueError
        If ``snapshot_sets`` is empty, the sets disagree on the number
        of states or parameters, or steady and transient sets are mixed:
        a steady state has no time to evaluate a lift at.
    """
    if not snapshot_sets:
        raise ValueError("at least one snapshot set is required")
    first = snapshot_sets[0]
    for other in snapshot_sets[1:]:
        if (other.nstates(), other.nparams()) != (
            first.nstates(),
            first.nparams(),
        ):
            raise ValueError(
                "snapshot sets disagree on (nstates, nparams): "
                f"{(first.nstates(), first.nparams())} vs "
                f"{(other.nstates(), other.nparams())}"
            )
        if other.is_steady() != first.is_steady():
            raise ValueError("cannot join steady and transient snapshots")
    states = bkd.hstack([s.states for s in snapshot_sets])
    params = bkd.hstack([s.params for s in snapshot_sets])
    if first.is_steady():
        return SnapshotSet(states, params)
    times = bkd.concatenate([s.times for s in snapshot_sets if s.times is not None])
    return SnapshotSet(states, params, times)


def collect_steady_snapshots(
    state_equation: SteadyStateEquationProtocol[Array],
    samples: Array,
    init_state: Array,
) -> SnapshotSet[Array]:
    """Solve a steady state equation at every parameter sample.

    Parameters
    ----------
    state_equation : SteadyStateEquationProtocol
        ``R(u, p) = 0`` with its solve, e.g. the state equation of a
        steady forward model.
    samples : Array
        Parameter samples. Shape: (nparams, nsamples).
    init_state : Array
        Initial Newton guess, used for every sample so the result does
        not depend on the sample order. Shape: (nstates, 1).

    Returns
    -------
    SnapshotSet
        One steady snapshot per sample.

    Raises
    ------
    TypeError
        If ``state_equation`` does not satisfy the protocol.
    ValueError
        If ``samples`` or ``init_state`` has the wrong shape.
    """
    if not isinstance(state_equation, SteadyStateEquationProtocol):
        raise TypeError(
            "state_equation must satisfy SteadyStateEquationProtocol, got "
            f"{type(state_equation).__name__}"
        )
    bkd = state_equation.bkd()
    nparams, nstates = state_equation.nparams(), state_equation.nstates()
    if samples.ndim != 2 or samples.shape[0] != nparams:
        raise ValueError(
            f"samples must have shape ({nparams}, nsamples), got {tuple(samples.shape)}"
        )
    if tuple(init_state.shape) != (nstates, 1):
        raise ValueError(
            f"init_state must have shape ({nstates}, 1), got {tuple(init_state.shape)}"
        )
    columns = [
        state_equation.solve(init_state, samples[:, ii : ii + 1])
        for ii in range(samples.shape[1])
    ]
    return SnapshotSet(bkd.hstack(columns), bkd.copy(samples))


@runtime_checkable
class TrajectorySolverProtocol(Protocol, Generic[Array]):
    """A transient model that returns the trajectory it solves for.

    Satisfied by the Galerkin and collocation transient forward models.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of parameters."""
        ...

    def forward_solve(self, sample: Array) -> Tuple[Array, Array]:
        """Solve at one parameter sample (nvars, 1).

        Returns the states (nstates, ntimes) and times (ntimes,).
        """
        ...


def collect_transient_snapshots(
    model: TrajectorySolverProtocol[Array], samples: Array
) -> SnapshotSet[Array]:
    """Solve a transient model at every parameter sample.

    Every time step of every trajectory becomes a snapshot, with its
    sample repeated as the parameter and its step time as the time.

    Parameters
    ----------
    model : TrajectorySolverProtocol
        The transient model.
    samples : Array
        Parameter samples. Shape: (nvars, nsamples).

    Raises
    ------
    TypeError
        If ``model`` does not satisfy the protocol.
    ValueError
        If ``samples`` has the wrong shape.
    """
    if not isinstance(model, TrajectorySolverProtocol):
        raise TypeError(
            f"model must satisfy TrajectorySolverProtocol, got {type(model).__name__}"
        )
    bkd = model.bkd()
    nvars = model.nvars()
    if samples.ndim != 2 or samples.shape[0] != nvars:
        raise ValueError(
            f"samples must have shape ({nvars}, nsamples), got {tuple(samples.shape)}"
        )
    sets = []
    for ii in range(samples.shape[1]):
        sample = samples[:, ii : ii + 1]
        states, times = model.forward_solve(sample)
        params = bkd.dot(sample, bkd.ones((1, int(times.shape[0]))))
        sets.append(SnapshotSet(states, params, times))
    return concatenate_snapshot_sets(sets, bkd)
