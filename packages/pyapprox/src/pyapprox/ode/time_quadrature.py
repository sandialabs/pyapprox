"""
Trajectory quadrature: a stepper's scheme-implied rule applied to
stored trajectories.

Time-integrated functionals approximate :math:`\\int_0^T q(y(t))\\, dt`
as :math:`\\sum_j w_j\\, q(\\hat{y}_j)` where the sample states
:math:`\\hat{y} = S y` are affine combinations of the stored trajectory
columns. The pair :math:`(w, S)` is implied by the time-integration
scheme, so each stepper constructs its own rule object via
``trajectory_quadrature(times)`` — consumers obtain instances from the
stepper of an actual solve and never build weights by hand, which
would silently break the order-consistency between scheme and
quadrature. A new stepper with a new sample layout implements
:class:`TrajectoryQuadratureProtocol` (or reuses a constructor below)
and returns it; nothing else changes.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class TrajectoryQuadratureProtocol(Protocol, Generic[Array]):
    """Quadrature over a stored trajectory: Q = sum_j w_j q((S y)_j)."""

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def ntimes(self) -> int:
        """Return the number of stored trajectory columns."""
        ...

    def nsamples(self) -> int:
        """Return the number of quadrature samples."""
        ...

    def weights(self) -> Array:
        """Return quadrature weights. Shape: ``(nsamples,)``."""
        ...

    def sample(self, traj: Array) -> Array:
        """Map a trajectory to sample states :math:`S y`.

        Parameters
        ----------
        traj : Array
            Trajectory. Shape: ``(nstates, ntimes)``.

        Returns
        -------
        Array
            Sample states. Shape: ``(nstates, nsamples)``.
        """
        ...

    def accumulate(self, sampled: Array) -> Array:
        """Map sample-space values back by :math:`S^\\top`.

        The adjoint of :meth:`sample`; chain rule for trajectory
        derivatives: :math:`dQ/dy = S^\\top (w \\odot q'(\\hat{y}))`.

        Parameters
        ----------
        sampled : Array
            Sample-space values. Shape: ``(nstates, nsamples)``.

        Returns
        -------
        Array
            Trajectory-space values. Shape: ``(nstates, ntimes)``.
        """
        ...

    def is_time_diagonal(self) -> bool:
        """Return True when every sample involves exactly one column.

        Time-diagonal rules make a time-integrated functional's second
        derivative block-diagonal in time — the structure the per-step
        HVP machinery assumes.
        """
        ...

    def nodal_weights(self) -> Array:
        """Return dense per-column weights :math:`S^\\top w`.

        Shape: ``(ntimes,)``. Raises for time-coupled rules, where a
        single per-column weight cannot represent the quadrature.
        """
        ...


class NodalTrajectoryQuadrature(Generic[Array]):
    """Quadrature whose samples are stored trajectory columns.

    Covers the rules of backward Euler (right endpoints), forward
    Euler (left endpoints), and Crank-Nicolson/Heun (all nodes,
    trapezoidal weights).

    Parameters
    ----------
    node_indices : Array
        Column index of each sample, all distinct. Shape:
        ``(nsamples,)``, integer dtype.
    weights : Array
        Quadrature weights. Shape: ``(nsamples,)``.
    ntimes : int
        Number of stored trajectory columns.
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self,
        node_indices: Array,
        weights: Array,
        ntimes: int,
        bkd: Backend[Array],
    ) -> None:
        if node_indices.ndim != 1 or weights.ndim != 1:
            raise ValueError("node_indices and weights must be 1D")
        if node_indices.shape[0] != weights.shape[0]:
            raise ValueError(
                f"node_indices has {node_indices.shape[0]} entries but "
                f"weights has {weights.shape[0]}"
            )
        self._node_indices = node_indices
        self._weights = weights
        self._ntimes = ntimes
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ntimes(self) -> int:
        return self._ntimes

    def nsamples(self) -> int:
        return int(self._weights.shape[0])

    def weights(self) -> Array:
        return self._weights

    def sample(self, traj: Array) -> Array:
        return traj[:, self._node_indices]

    def accumulate(self, sampled: Array) -> Array:
        out = self._bkd.copy(
            self._bkd.zeros((sampled.shape[0], self._ntimes))
        )
        out[:, self._node_indices] = sampled
        return out

    def is_time_diagonal(self) -> bool:
        return True

    def nodal_weights(self) -> Array:
        out = self._bkd.copy(self._bkd.zeros((self._ntimes,)))
        out[self._node_indices] = self._weights
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nsamples={self.nsamples()}, ntimes={self._ntimes})"
        )


class MidpointTrajectoryQuadrature(Generic[Array]):
    """Quadrature sampling interval midpoints (implicit midpoint rule).

    Sample states are averages of adjacent columns,
    :math:`\\hat{y}_j = (y_j + y_{j+1})/2` — the states the implicit
    midpoint scheme itself steps through. Not time-diagonal: a
    functional's second derivative couples adjacent columns, so
    per-step HVP support requires the block-tridiagonal extension of
    the adjoint machinery (value and gradient are exact).

    Parameters
    ----------
    weights : Array
        Interval weights. Shape: ``(ntimes - 1,)``.
    ntimes : int
        Number of stored trajectory columns.
    bkd : Backend
        Backend for array operations.
    """

    def __init__(
        self, weights: Array, ntimes: int, bkd: Backend[Array]
    ) -> None:
        if weights.ndim != 1:
            raise ValueError("weights must be 1D")
        if weights.shape[0] != ntimes - 1:
            raise ValueError(
                f"midpoint rule needs ntimes - 1 = {ntimes - 1} weights, "
                f"got {weights.shape[0]}"
            )
        self._weights = weights
        self._ntimes = ntimes
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ntimes(self) -> int:
        return self._ntimes

    def nsamples(self) -> int:
        return self._ntimes - 1

    def weights(self) -> Array:
        return self._weights

    def sample(self, traj: Array) -> Array:
        return 0.5 * (traj[:, :-1] + traj[:, 1:])

    def accumulate(self, sampled: Array) -> Array:
        pad = self._bkd.zeros((sampled.shape[0], 1))
        return 0.5 * (
            self._bkd.hstack([sampled, pad])
            + self._bkd.hstack([pad, sampled])
        )

    def is_time_diagonal(self) -> bool:
        return False

    def nodal_weights(self) -> Array:
        raise ValueError(
            "the midpoint rule couples adjacent trajectory columns; no "
            "per-column weight vector represents it. Use sample()/"
            "accumulate(), or a nodal-rule scheme for per-step HVPs."
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nsamples={self.nsamples()}, ntimes={self._ntimes})"
        )


def _validate_times(times: Array) -> int:
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError(
            f"times must be 1D with at least 2 entries, got shape "
            f"{times.shape}"
        )
    return int(times.shape[0])


def _indices(start: int, stop: int, bkd: Backend[Array]) -> Array:
    return bkd.asarray(list(range(start, stop)), dtype=int)


def right_rectangle_quadrature(
    times: Array, bkd: Backend[Array]
) -> NodalTrajectoryQuadrature[Array]:
    """Right endpoints, interval widths — backward Euler's rule."""
    ntimes = _validate_times(times)
    return NodalTrajectoryQuadrature(
        _indices(1, ntimes, bkd), bkd.diff(times), ntimes, bkd
    )


def left_rectangle_quadrature(
    times: Array, bkd: Backend[Array]
) -> NodalTrajectoryQuadrature[Array]:
    """Left endpoints, interval widths — forward Euler's rule."""
    ntimes = _validate_times(times)
    return NodalTrajectoryQuadrature(
        _indices(0, ntimes - 1, bkd), bkd.diff(times), ntimes, bkd
    )


def trapezoidal_quadrature(
    times: Array, bkd: Backend[Array]
) -> NodalTrajectoryQuadrature[Array]:
    """All nodes, trapezoidal weights — Crank-Nicolson/Heun's rule."""
    ntimes = _validate_times(times)
    deltas = bkd.diff(times)
    pad = bkd.zeros((1,))
    weights = 0.5 * (
        bkd.hstack([deltas, pad]) + bkd.hstack([pad, deltas])
    )
    return NodalTrajectoryQuadrature(
        _indices(0, ntimes, bkd), weights, ntimes, bkd
    )


def midpoint_quadrature(
    times: Array, bkd: Backend[Array]
) -> MidpointTrajectoryQuadrature[Array]:
    """Interval midpoints, interval widths — implicit midpoint's rule."""
    ntimes = _validate_times(times)
    return MidpointTrajectoryQuadrature(bkd.diff(times), ntimes, bkd)
