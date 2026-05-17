"""Snapshot dataset for dynamical systems learning.

Stores state snapshots and time derivatives for derivative matching,
with factory methods for constructing from trajectories.
"""

from __future__ import annotations

from typing import Generic, List, Optional

from pyapprox.surrogates.dynamical_systems.protocols import EncoderProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend


class SnapshotDataset(Generic[Array]):
    """Holds state snapshots and their time derivatives.

    Parameters
    ----------
    states : Array
        State snapshots (may include auxiliary rows like parameters).
        Shape: (nstates_input, nsamples)
    derivatives : Array
        Time derivatives dx/dt at each snapshot.
        Shape: (nstates_output, nsamples). May have fewer rows than states
        when states includes auxiliary (non-dynamic) rows.
    bkd : Backend[Array]
        Computational backend.
    times : Array, optional
        Time stamps for each snapshot. Shape: (nsamples,)
    """

    def __init__(
        self,
        states: Array,
        derivatives: Array,
        bkd: Backend[Array],
        times: Optional[Array] = None,
    ):
        validate_backend(bkd)
        if states.ndim != 2:
            raise ValueError(f"states must be 2D, got {states.ndim}D")
        if derivatives.ndim != 2:
            raise ValueError(f"derivatives must be 2D, got {derivatives.ndim}D")
        if states.shape[1] != derivatives.shape[1]:
            raise ValueError(
                f"states nsamples {states.shape[1]} != derivatives nsamples "
                f"{derivatives.shape[1]}"
            )
        if times is not None and times.shape[0] != states.shape[1]:
            raise ValueError(
                f"times length {times.shape[0]} != nsamples {states.shape[1]}"
            )
        self._states = states
        self._derivatives = derivatives
        self._bkd = bkd
        self._times = times

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def states(self) -> Array:
        """Return state snapshots. Shape: (nstates, nsamples)."""
        return self._states

    def derivatives(self) -> Array:
        """Return time derivatives. Shape: (nstates, nsamples)."""
        return self._derivatives

    def times(self) -> Optional[Array]:
        """Return time stamps, or None. Shape: (nsamples,) if present."""
        return self._times

    def nstates(self) -> int:
        return self._states.shape[0]

    def nstates_input(self) -> int:
        """Number of input state rows (may include auxiliaries)."""
        return self._states.shape[0]

    def nstates_output(self) -> int:
        """Number of output derivative rows (dynamic states only)."""
        return self._derivatives.shape[0]

    def nsamples(self) -> int:
        return self._states.shape[1]

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Array,
        times: Array,
        bkd: Backend[Array],
        fd_method: str = "central",
    ) -> SnapshotDataset[Array]:
        """Construct from a single trajectory using finite differences.

        Parameters
        ----------
        trajectory : Array
            Solution at each time. Shape: (nstates, ntimes)
        times : Array
            Time points. Shape: (ntimes,)
        bkd : Backend[Array]
            Computational backend.
        fd_method : str
            Finite difference method: 'central', 'forward', or 'backward'.
            Should match the time-stepping scheme order: 'backward' for
            backward Euler, 'forward' for forward Euler, 'central' for
            Crank-Nicolson or other second-order methods.

        Returns
        -------
        SnapshotDataset[Array]
            Dataset with FD-approximated derivatives. Interior points only
            for 'central'; all but last for 'forward'; all but first for
            'backward'.
        """
        validate_backend(bkd)
        ntimes = trajectory.shape[1]
        if times.shape[0] != ntimes:
            raise ValueError(
                f"times length {times.shape[0]} != trajectory columns {ntimes}"
            )

        if fd_method == "central":
            dt_fwd = times[2:] - times[:-2]
            derivs = (trajectory[:, 2:] - trajectory[:, :-2]) / dt_fwd[None, :]
            states = trajectory[:, 1:-1]
            t_out = times[1:-1]
        elif fd_method == "forward":
            dt = times[1:] - times[:-1]
            derivs = (trajectory[:, 1:] - trajectory[:, :-1]) / dt[None, :]
            states = trajectory[:, :-1]
            t_out = times[:-1]
        elif fd_method == "backward":
            dt = times[1:] - times[:-1]
            derivs = (trajectory[:, 1:] - trajectory[:, :-1]) / dt[None, :]
            states = trajectory[:, 1:]
            t_out = times[1:]
        else:
            raise ValueError(
                f"fd_method must be 'central', 'forward', or 'backward', "
                f"got '{fd_method}'"
            )

        return cls(states=states, derivatives=derivs, bkd=bkd, times=t_out)

    @classmethod
    def from_trajectories(
        cls,
        trajectories: List[Array],
        times_list: List[Array],
        bkd: Backend[Array],
        fd_method: str = "central",
    ) -> SnapshotDataset[Array]:
        """Construct from multiple trajectories, concatenating samples.

        Parameters
        ----------
        trajectories : list of Array
            Each trajectory has shape (nstates, ntimes_k).
        times_list : list of Array
            Each has shape (ntimes_k,).
        bkd : Backend[Array]
            Computational backend.
        fd_method : str
            Finite difference method for derivative approximation.

        Returns
        -------
        SnapshotDataset[Array]
            Concatenated dataset from all trajectories.
        """
        datasets = [
            cls.from_trajectory(traj, t, bkd, fd_method)
            for traj, t in zip(trajectories, times_list)
        ]
        all_states = bkd.hstack([d.states() for d in datasets])
        all_derivs = bkd.hstack([d.derivatives() for d in datasets])
        times_arrays: List[Array] = []
        for d in datasets:
            t = d.times()
            assert t is not None
            times_arrays.append(t)
        all_times = bkd.concatenate(times_arrays)
        return cls(
            states=all_states, derivatives=all_derivs, bkd=bkd, times=all_times
        )

    def project(self, encoder: EncoderProtocol[Array]) -> SnapshotDataset[Array]:
        """Project dataset into latent space via encoder.

        States are encoded, derivatives are projected via the encoder
        Jacobian: dz/dt = (d_encode/dx) @ dx/dt.

        Parameters
        ----------
        encoder : EncoderProtocol[Array]
            Encoder mapping full -> latent space.

        Returns
        -------
        SnapshotDataset[Array]
            New dataset in latent coordinates.
        """
        latent_states = encoder.encode(self._states)
        enc_jac = encoder.encode_jacobian()
        latent_derivs = enc_jac @ self._derivatives
        return SnapshotDataset(
            states=latent_states,
            derivatives=latent_derivs,
            bkd=self._bkd,
            times=self._times,
        )
