r"""Subdomains advancing through time windows.

A windowed response covers a window :math:`[t_0, t_1]` of a transient
problem. Its port data are the trace at the window's end,
:math:`\lambda(t_1)`; within the window the trace is the linear
interpolant from the subdomain's own trace at :math:`t_0` (its committed
state's port rows) to :math:`\lambda(t_1)`, so the rate of the port data
is known and a Runge-Kutta stage reads the data at its own time.

**A window is a trial, then a commit.** :meth:`WindowedResponseProtocol.
linearize` steps the subdomain over the window from its committed state
and changes nothing it holds: the point keeps the trial trajectory, and
a skeleton Newton may linearize any number of times.
:meth:`~WindowedResponseProtocol.accept` commits one point: the
subdomain's state becomes the trial's last state and the trial's steps
go into the subdomain's sink. Not accepting is the rollback, so a
rejected window, or one solved again, needs nothing more. The response
owns its state: the skeleton never holds a subdomain state.

**Storage is injected.** Each response writes its committed steps into a
:class:`~pyapprox.ode.trajectory_store.TrajectorySinkProtocol` given at
construction: a store when sweeps will read the trajectory back, a sink
keeping nothing when none will, a file store, or several through a
multi-sink. The trial trajectory is an
:class:`~pyapprox.ode.trajectory_store.InMemoryTrajectoryStore` of its
own, replayed into the sink step by step on commit, so the sink sees the
same windows (unknowns, sizes, parameter versions) a single-domain
integration would write.

:class:`WindowedPinnedResponse` is the Dirichlet-to-Neumann response of
a stepped subdomain: its port rows pinned beside its physical essential
rows (:class:`WindowPortRows`), its output the step residual of the
unconstrained stepper on the port rows, summed over the window's steps.
For a Runge-Kutta step the last block of that residual is
:math:`M(y_{n+1} - y_n) - \Delta t \sum_i b_i f(Y_i)`: on a port row,
the outward flux integrated over the step plus the row's inertia, the
time-discrete reaction. Summed over the subdomains at an interface it is
the single-domain step residual of the interface rows, so a skeleton
solve per window reproduces the single-domain transient solution.
"""

from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.integration import Integrator
from pyapprox.ode.pinned_rows import PinnedRowsParamDerivatives, PinnedRowsProtocol
from pyapprox.ode.protocols.differentiable_stepper import (
    DifferentiableStepperProtocol,
    StepperProtocol,
)
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.trajectory_store import (
    InMemoryTrajectoryStore,
    TrajectorySinkProtocol,
)
from pyapprox.pde.coupling.ports import PortRowsProtocol
from pyapprox.pde.coupling.responses import (
    Flux,
    LocalResponseProtocol,
    ResponseDerivatives,
    ResponsePointProtocol,
    TimeWindow,
    Trace,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import (
    BlockVector,
    RowSetProtocol,
    VectorSpaceProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)


@runtime_checkable
class WindowedResponseProtocol(LocalResponseProtocol[A], Protocol[A]):
    """A subdomain that advances through time windows, as a transient
    skeleton solve sees it (see the module docstring)."""

    def time(self) -> float:
        """The end of the last accepted window; the initial time before
        any."""
        ...

    def start(self) -> None:
        """Write the initial state into the sink. Called once, before any
        window is accepted."""
        ...

    def accept(self, point: ResponsePointProtocol[A]) -> None:
        """Commit ``point``, one of this response's points over a window
        starting at :meth:`time`."""
        ...


class WindowPortRows(Generic[V, A]):
    r"""A subdomain's essential rows and its port rows, with their data
    over one window: the essential data :math:`g(t)` and, on the port
    rows, the linear interpolant from the window's start trace to its
    end trace.

    Satisfies :class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol`.
    Holds the window, set by :meth:`set_window`; the rows, the rates'
    presence and the parameter derivatives are fixed for its life.

    Parameters
    ----------
    essential : PinnedRowsProtocol
        The physical essential rows and their data; independent of the
        parameters.
    rows : RowSetProtocol
        The union of the essential rows and the port rows, as the stack's
        row set. The two must be disjoint.
    ports : PortRowsProtocol
        The port rows of the state space, in port order.
    states : VectorSpaceProtocol
        The state space.

    Raises
    ------
    ValueError
        If ``rows`` does not count the essential and port rows once each.
    """

    def __init__(
        self,
        essential: PinnedRowsProtocol[V],
        rows: RowSetProtocol[V],
        ports: PortRowsProtocol[V, A],
        states: VectorSpaceProtocol[V],
    ) -> None:
        expected = essential.rows().count() + ports.nrows()
        if rows.count() != expected:
            raise ValueError(
                f"rows must hold the {essential.rows().count()} essential and "
                f"{ports.nrows()} port rows once each; it has {rows.count()}"
            )
        self._essential, self._rows = essential, rows
        self._ports, self._states = ports, states
        # Read once: a declaration's rates are fixed for its life.
        self._essential_rates = essential.rates()
        self._rates: Optional[Callable[[float], V]] = (
            None if self._essential_rates is None else self._rate
        )
        self._window = TimeWindow(0.0, 1.0)
        self._start: Optional[A] = None
        self._end: Optional[A] = None

    def set_window(self, window: TimeWindow, start: A, end: A) -> None:
        """Hold ``window`` with the port trace ``start`` at its start and
        ``end`` at its end, both of shape (nport, 1).

        Raises
        ------
        ValueError
            If the window is steady.
        """
        if window.is_steady():
            raise ValueError("port data are interpolated over a nonempty window")
        self._window, self._start, self._end = window, start, end

    def _traces(self) -> tuple[A, A]:
        if self._start is None or self._end is None:
            raise ValueError("no window is set")
        return self._start, self._end

    def _placed(self, block: A) -> V:
        return self._ports.add_to(self._states.zeros(1), block)

    def rows(self) -> RowSetProtocol[V]:
        return self._rows

    def values(self, time: float) -> V:
        start, end = self._traces()
        window = self._window
        s = (time - window.start) / (window.end - window.start)
        trace = start + s * (end - start)
        return self._states.axpby(
            1.0, self._essential.values(time), 1.0, self._placed(trace)
        )

    def _rate(self, time: float) -> V:
        start, end = self._traces()
        window = self._window
        if self._essential_rates is None:
            raise ValueError("the essential data supply no rate")
        slope = (end - start) / (window.end - window.start)
        return self._states.axpby(
            1.0, self._essential_rates(time), 1.0, self._placed(slope)
        )

    def rates(self) -> Optional[Callable[[float], V]]:
        """The rate of the essential data plus the interpolant's slope on
        the port rows; None when the essential data supply no rate."""
        return self._rates

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[V]:
        """Independent of the parameters: the port data enter as data."""
        return PinnedRowsParamDerivatives.independent()


class _Owner:
    """Identifies the response a point came from."""


class WindowedPinnedPoint(Generic[V, A]):
    r"""A :class:`WindowedPinnedResponse` stepped over one window at one
    end trace. Its output and derivatives refer to its own trial, so they
    stay valid after the response is linearized again.
    """

    def __init__(
        self,
        owner: _Owner,
        window: TimeWindow,
        trial: InMemoryTrajectoryStore[V],
        output: A,
        bkd: Backend[A],
        tangent: Optional["_EndTraceTangent[V, A]"],
    ) -> None:
        self._owner, self._window, self._trial = owner, window, trial
        self._output, self._bkd = output, bkd
        self._derivs: ResponseDerivatives[A] = (
            ResponseDerivatives.none()
            if tangent is None
            else ResponseDerivatives.tangent(tangent.bind(trial))
        )

    def output(self) -> A:
        """The time-discrete reaction on the ports over the window.
        Shape: (nport, 1)."""
        return self._output

    def qoi(self) -> A:
        return self._bkd.zeros((0, 1))

    def derivatives(self) -> ResponseDerivatives[A]:
        return self._derivs

    def window(self) -> TimeWindow:
        """The window stepped."""
        return self._window

    def comes_from(self, owner: _Owner) -> bool:
        """Whether the response identified by ``owner`` made this point."""
        return self._owner is owner

    def trial(self) -> InMemoryTrajectoryStore[V]:
        """The trial trajectory, from the committed state to the window's
        end. Not to be modified."""
        return self._trial

    def state(self) -> V:
        """The subdomain state at the window's end."""
        return self._trial.state(-1)


class _EndTraceTangent(Generic[V, A]):
    r""":math:`\partial r/\partial \lambda(t_1)` for a one-step window of
    a one-block stepper (backward Euler): the step's pinned port rows read
    :math:`Y_R - \lambda(t_1)`, so the unknown's change solves
    :math:`(\partial R/\partial u)\, \delta u = [\delta\lambda]_R`, and the
    output changes by the unconstrained step residual's derivative at
    :math:`\delta u` on the port rows."""

    def __init__(
        self,
        stepper: StepperProtocol[V, A],
        reaction: DifferentiableStepperProtocol[V, A],
        ports: PortRowsProtocol[V, A],
        states: VectorSpaceProtocol[V],
    ) -> None:
        self._stepper, self._reaction = stepper, reaction
        self._ports, self._states = ports, states

    def bind(self, trial: InMemoryTrajectoryStore[V]) -> Callable[[A], A]:
        def jvp(directions: A) -> A:
            return self._apply(trial, directions)

        return jvp

    def _apply(self, trial: InMemoryTrajectoryStore[V], directions: A) -> A:
        nport = self._ports.nrows()
        if directions.ndim != 2 or directions.shape[0] != nport:
            raise ValueError(
                f"directions must have shape ({nport}, ncols), got "
                f"{tuple(directions.shape)}"
            )
        ncols = int(directions.shape[1])
        window = trial.window(0)
        rhs = BlockVector((self._ports.add_to(self._states.zeros(ncols), directions),))
        delta, info = self._stepper.diag_handle(window).solve(rhs)
        if not info.converged:
            raise ValueError("the tangent solve did not converge")
        change = self._reaction.diag(window).apply(delta)
        return self._ports.take(change.blocks[-1])


class WindowedPinnedResponse(Generic[V, A]):
    r"""A stepped subdomain as a Dirichlet-to-Neumann response over time
    windows (see the module docstring). Receives the trace at a window's
    end, returns the time-discrete reaction on the ports over the window.
    Without parameters or quantities of interest.

    Satisfies :class:`WindowedResponseProtocol`. The trace derivative
    (``jvp_trace``) is supplied for one step per window of a one-block
    stepper (backward Euler); otherwise the bundle is empty.

    Parameters
    ----------
    stepper : StepperProtocol
        Steps the subdomain with ``rows`` pinned: built over the system
        whose essential rows are ``rows``, with a pin of them.
    reaction : DifferentiableStepperProtocol
        The same method over the unconstrained system, whose step
        residual is read on the port rows.
    rows : WindowPortRows
        The rows ``stepper`` pins; the response sets its window.
    ports : PortRowsProtocol
        The port rows of the state space, in port order: ``rows``' own.
    states : VectorSpaceProtocol
        The state space.
    init_state : V
        The state at ``origin``, a single column; its port rows are the
        initial trace.
    origin : float
        The initial time.
    sink : TrajectorySinkProtocol
        Empty; receives the initial state on :meth:`start` and every
        accepted step.
    bkd : Backend
        The backend of the port blocks.
    nsubsteps : int, default 1
        Equal steps per window.

    Raises
    ------
    TypeError
        If ``stepper``, ``reaction`` or ``sink`` does not satisfy its
        protocol.
    ValueError
        If ``nsubsteps`` is less than 1 or ``sink`` is not empty.
    """

    def __init__(
        self,
        stepper: StepperProtocol[V, A],
        reaction: DifferentiableStepperProtocol[V, A],
        rows: WindowPortRows[V, A],
        ports: PortRowsProtocol[V, A],
        states: VectorSpaceProtocol[V],
        init_state: V,
        origin: float,
        sink: TrajectorySinkProtocol[V],
        bkd: Backend[A],
        nsubsteps: int = 1,
    ) -> None:
        for name, value, protocol in (
            ("stepper", stepper, StepperProtocol),
            ("reaction", reaction, DifferentiableStepperProtocol),
            ("sink", sink, TrajectorySinkProtocol),
        ):
            if not isinstance(value, protocol):
                raise TypeError(
                    f"{name} must satisfy {protocol.__name__}, got "
                    f"{type(value).__name__}"
                )
        if nsubsteps < 1:
            raise ValueError(f"nsubsteps must be at least 1, got {nsubsteps}")
        if sink.ntimes() != 0:
            raise ValueError("the sink must be empty")
        self._stepper, self._reaction = stepper, reaction
        self._rows, self._ports, self._states = rows, ports, states
        self._state, self._time = init_state, float(origin)
        self._sink, self._bkd = sink, bkd
        self._nsubsteps = nsubsteps
        self._integrator = Integrator(stepper)
        self._owner = _Owner()
        self._started = False
        one_block = stepper.unknown_space().nblocks() == 1
        self._tangent: Optional[_EndTraceTangent[V, A]] = (
            _EndTraceTangent(stepper, reaction, ports, states)
            if one_block and nsubsteps == 1
            else None
        )

    def bkd(self) -> Backend[A]:
        return self._bkd

    def nport(self) -> int:
        return self._ports.nrows()

    def receives(self) -> Trace:
        return Trace()

    def returns(self) -> Flux:
        return Flux()

    def nparams(self) -> int:
        return 0

    def nqoi(self) -> int:
        return 0

    def time(self) -> float:
        return self._time

    def state(self) -> V:
        """The committed state, at :meth:`time`. Not to be modified."""
        return self._state

    def start(self) -> None:
        """Write the initial state into the sink.

        Raises
        ------
        ValueError
            If the response has already started.
        """
        if self._started:
            raise ValueError("the response has already started")
        self._sink.append(self._time, self._state)
        self._started = True

    def linearize(
        self, port_data: A, params: A, window: TimeWindow
    ) -> WindowedPinnedPoint[V, A]:
        """Step over ``window`` from the committed state, the port trace
        ending at ``port_data`` (nport, 1); ``params`` is (0, 1).

        Raises
        ------
        ValueError
            If a shape is wrong, the window is steady or does not start at
            :meth:`time`.
        SolveFailedError
            If a step's solve fails.
        """
        for name, block, nrows in (
            ("port_data", port_data, self.nport()),
            ("params", params, 0),
        ):
            if tuple(block.shape) != (nrows, 1):
                raise ValueError(
                    f"{name} must have shape ({nrows}, 1), got {tuple(block.shape)}"
                )
        if window.is_steady():
            raise ValueError("a windowed response needs a nonempty window")
        if window.start != self._time:
            raise ValueError(
                f"the window starts at {window.start}, not at the committed time "
                f"{self._time}"
            )
        self._rows.set_window(window, self._ports.take(self._state), port_data)
        size = (window.end - window.start) / self._nsubsteps
        trial: InMemoryTrajectoryStore[V] = InMemoryTrajectoryStore()
        self._integrator.solve(
            self._state, GivenSizes(window.start, [size] * self._nsubsteps), trial
        )
        output = self._ports.take(
            self._reaction.step_residual(trial.window(0)).blocks[-1]
        )
        for step in range(1, trial.nsteps()):
            residual = self._reaction.step_residual(trial.window(step))
            output = output + self._ports.take(residual.blocks[-1])
        return WindowedPinnedPoint(
            self._owner, window, trial, output, self._bkd, self._tangent
        )

    def accept(self, point: ResponsePointProtocol[A]) -> None:
        """Commit ``point``: its last state becomes the committed state,
        its window's end the committed time, and its steps go into the
        sink.

        Raises
        ------
        ValueError
            If the response has not started, ``point`` is not one of its
            points, or its window does not start at :meth:`time`.
        """
        if not self._started:
            raise ValueError("start the response before accepting a window")
        if not isinstance(point, WindowedPinnedPoint) or not point.comes_from(
            self._owner
        ):
            raise ValueError("the point is not one of this response's")
        window = point.window()
        if window.start != self._time:
            raise ValueError(
                f"the point's window starts at {window.start}, not at the "
                f"committed time {self._time}"
            )
        trial = point.trial()
        for step in range(trial.nsteps()):
            self._sink.append_step(trial.window(step), trial.state(step + 1))
        self._state, self._time = trial.state(-1), window.end

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nport={self.nport()}, time={self._time})"
