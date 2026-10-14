r"""The offset that carries the essential boundary data of a reduced model.

Essential conditions make the full-order solutions an affine set,
:math:`u = u_{\mathrm{lift}}(t) + u_0` with :math:`u_0` zero on the
constrained DOFs. A reduced model approximates only :math:`u_0`, in a
basis built from homogenized snapshots :math:`s - u_{\mathrm{lift}}`,
so its basis vanishes on the constrained DOFs. Projecting the raw
residual (before row replacement) onto such a basis never reads a
replaced row, which is why the reduced model works with the system's
raw spatial operator rather than its constrained one.

Which field carries the data is a choice: any field that matches the
boundary data on the constrained DOFs is a valid lift, and smoother
ones give faster POD decay when the data depends on the parameter. The
choice is injected through :class:`LiftProtocol`; :class:`AlgebraicLift`
is the simplest, zero away from the constrained DOFs.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.rom.snapshots import SnapshotSet
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class LiftProtocol(Protocol, Generic[Array]):
    """A field carrying the essential boundary data at each time."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the number of states."""
        ...

    def is_time_invariant(self) -> bool:
        """Whether the lift is declared independent of time."""
        ...

    def lift(self, time: float) -> Array:
        """Return the lift at ``time``. Shape: (nstates,)."""
        ...


class AlgebraicLift(Generic[Array]):
    """The boundary data on the constrained DOFs and zero elsewhere.

    Exact on the constrained DOFs by construction, and the cheapest lift.
    When the boundary data varies with the parameter it leaves boundary
    structure in the homogenized snapshots and slows POD decay; a lift
    that extends the data smoothly into the interior avoids that.

    Parameters
    ----------
    constraint_set : ConstraintSetProtocol
        The essential constraints of the full-order system.
    nstates : int
        Number of full-order states.

    Raises
    ------
    TypeError
        If ``constraint_set`` does not satisfy its protocol.
    """

    def __init__(
        self, constraint_set: ConstraintSetProtocol[Array], nstates: int
    ) -> None:
        if not isinstance(constraint_set, ConstraintSetProtocol):
            raise TypeError(
                "constraint_set must satisfy ConstraintSetProtocol, got "
                f"{type(constraint_set).__name__}"
            )
        self._constraint_set = constraint_set
        self._nstates = nstates
        self._bkd = constraint_set.bkd()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the number of states."""
        return self._nstates

    def is_time_invariant(self) -> bool:
        """Whether the boundary data is declared time-invariant."""
        return self._constraint_set.is_time_invariant()

    def lift(self, time: float) -> Array:
        """Return the boundary data at ``time``, zero elsewhere."""
        return self._constraint_set.inject(self._bkd.zeros((self._nstates,)), time)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nstates={self._nstates}, "
            f"nconstrained={self._constraint_set.ndofs()})"
        )


def homogenize(
    snapshots: SnapshotSet[Array],
    lift: LiftProtocol[Array],
    constraint_set: ConstraintSetProtocol[Array],
    atol: float = 1e-10,
) -> SnapshotSet[Array]:
    """Subtract the lift from every snapshot at the snapshot's time.

    Checks that each homogenized snapshot vanishes on the constrained
    DOFs, which holds only when the snapshot carries the boundary data
    of ``constraint_set`` at its time and the lift does too. Snapshots
    from a different system, or a lift built for different data, would
    otherwise give a basis that does not vanish on the constrained DOFs
    and a reduced model that silently violates the boundary conditions.

    Parameters
    ----------
    snapshots : SnapshotSet
        Full-order snapshots.
    lift : LiftProtocol
        The lift to subtract.
    constraint_set : ConstraintSetProtocol
        The essential constraints the snapshots satisfy.
    atol : float
        Largest accepted absolute value of a homogenized snapshot on a
        constrained DOF.

    Returns
    -------
    SnapshotSet
        The homogenized states with the same parameters and times.

    Raises
    ------
    TypeError
        If ``lift`` or ``constraint_set`` does not satisfy its protocol.
    ValueError
        If the sizes disagree, steady snapshots meet time-dependent
        data, or a homogenized snapshot is nonzero on a constrained DOF.
    """
    if not isinstance(lift, LiftProtocol):
        raise TypeError(f"lift must satisfy LiftProtocol, got {type(lift).__name__}")
    if not isinstance(constraint_set, ConstraintSetProtocol):
        raise TypeError(
            "constraint_set must satisfy ConstraintSetProtocol, got "
            f"{type(constraint_set).__name__}"
        )
    if lift.nstates() != snapshots.nstates():
        raise ValueError(
            f"lift has {lift.nstates()} states but the snapshots have "
            f"{snapshots.nstates()}"
        )
    bkd = lift.bkd()
    times = snapshots.times
    if times is None and not (
        lift.is_time_invariant() and constraint_set.is_time_invariant()
    ):
        raise ValueError(
            "steady snapshots belong to no time, so the lift and the "
            "constraints must be declared time-invariant"
        )
    nsnapshots = snapshots.nsnapshots()
    if times is None:
        lifts = [lift.lift(0.0)] * nsnapshots
    else:
        lifts = [lift.lift(bkd.to_float(times[ii])) for ii in range(nsnapshots)]
    columns = []
    for ii, current in enumerate(lifts):
        homogenized = snapshots.states[:, ii] - current
        on_constrained = homogenized - constraint_set.zero_entries(homogenized)
        mismatch = bkd.to_float(bkd.max(bkd.abs(on_constrained)))
        if mismatch > atol:
            raise ValueError(
                f"snapshot {ii} differs from the lift by {mismatch:.2e} on "
                f"a constrained DOF (atol={atol:.1e}): it was produced with "
                "different boundary data, or at a different time, than the "
                "constraint set and lift describe"
            )
        columns.append(homogenized[:, None])
    return SnapshotSet(bkd.hstack(columns), snapshots.params, times)
