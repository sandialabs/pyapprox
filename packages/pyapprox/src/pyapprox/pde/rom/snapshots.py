r"""Full-order snapshots and the parameters and times they came from.

A projection reduced-order model is fitted to states of the full-order
model. Each state is only meaningful together with the parameter that
produced it and, for a transient solve, the time it belongs to: the
boundary lift is evaluated at that time, and hyper-reduction trains on
residuals evaluated at that parameter and time. :class:`SnapshotSet`
keeps the three together, one column per snapshot, so they cannot be
paired up wrongly downstream.

How the states are computed is the caller's choice: a reduced model reads
only the data, so nothing here depends on the solver or the
discretization that produced them.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Sequence

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

    @staticmethod
    def from_trajectory(
        states: Array, param: Array, times: Array, bkd: Backend[Array]
    ) -> "SnapshotSet[Array]":
        """One trajectory's states, each paired with the parameter that
        produced the whole trajectory and with its own time.

        Parameters
        ----------
        states : Array
            The trajectory's states. Shape: (nstates, ntimes).
        param : Array
            The parameter it was solved at. Shape: (nparams, 1).
        times : Array
            The time of each state. Shape: (ntimes,).
        bkd : Backend
            The arrays' backend.

        Raises
        ------
        ValueError
            If ``param`` is not a single column, or the arrays disagree on
            the number of snapshots.
        """
        if param.ndim != 2 or param.shape[1] != 1:
            raise ValueError(
                f"param must have shape (nparams, 1), got {tuple(param.shape)}"
            )
        if times.ndim != 1:
            raise ValueError(f"times must be 1D (ntimes,), got ndim={times.ndim}")
        params = bkd.dot(param, bkd.ones((1, int(times.shape[0]))))
        return SnapshotSet(states, params, times)

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
