r"""Subdomains advancing through time windows.

A windowed response covers a window :math:`[t_0, t_1]` of a transient
problem: given its port data for the window it steps the subdomain from
its committed state at :math:`t_0` to :math:`t_1` and returns an output
on its ports.

**A window is a trial, then a commit.** :meth:`WindowedResponseProtocol.
linearize` steps the subdomain over the window and changes nothing it
holds: the point keeps the trial trajectory, and a skeleton Newton may
linearize any number of times. :meth:`~WindowedResponseProtocol.accept`
commits one point: the subdomain's state becomes the trial's last state
and the trial's steps go into the subdomain's sink. Not accepting is the
rollback, so a rejected window, or one solved again, needs nothing more.
The response owns its state: the skeleton never holds a subdomain state.

**Storage is injected.** Each response writes its committed steps into a
:class:`~pyapprox.ode.trajectory_store.TrajectorySinkProtocol` given at
construction: a store when sweeps will read the trajectory back, a sink
keeping nothing when none will, a file store, or several through a
multi-sink. The trial trajectory is an
:class:`~pyapprox.ode.trajectory_store.InMemoryTrajectoryStore` of its
own, replayed into the sink step by step on commit, so the sink sees the
same windows (unknowns, sizes, parameter versions) a single-domain
integration would write.

**The transmission is injected.** :class:`WindowedResponse` holds only
the lifecycle. How the port data enter the subdomain is a
:class:`WindowPortInputProtocol`, set before each trial; what the
response returns is a :class:`WindowPortOutputProtocol`, read from the
trial. A new transmission (port data as a natural load, a Robin datum;
a trace or pointwise-flux output) is a new input or output, not a new
response.

The Dirichlet-to-Neumann pair: :class:`WindowPortRows` pins the port
rows beside the physical essential rows, to the linear interpolant from
the subdomain's own trace at :math:`t_0` (its committed state's port
rows) to the port data :math:`\lambda(t_1)`, so the data's rate is known
and a Runge-Kutta stage reads the data at its own time.
:class:`StepResidualReaction` returns the step residual of the
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
    PortQuantity,
    ResponseDerivatives,
    ResponsePointProtocol,
    TimeWindow,
    Trace,
    TraceJVP,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import (
    BlockVector,
    RowSetProtocol,
    VectorSpaceProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
V_contra = TypeVar("V_contra", contravariant=True)
A_contra = TypeVar("A_contra", bound=ArrayProtocol, contravariant=True)
A_co = TypeVar("A_co", bound=ArrayProtocol, covariant=True)

UnknownTangent = Callable[[BlockVector[V]], A]
"""An output's change for changes of a step's unknown, a block of
columns."""


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


@runtime_checkable
class WindowPortInputProtocol(Protocol[V_contra, A_contra]):
    """How a :class:`WindowedResponse`'s port data enter its subdomain:
    set before each trial, read by the stepper while it steps."""

    def nport(self) -> int:
        """The number of rows of the port data."""
        ...

    def receives(self) -> PortQuantity:
        """What the port data are."""
        ...

    def set_window(
        self, window: TimeWindow, state: V_contra, port_data: A_contra
    ) -> None:
        """Hold ``port_data`` (nport, 1) for ``window``, stepped from the
        committed ``state`` at the window's start."""
        ...


@runtime_checkable
class WindowPortOutputProtocol(Protocol[V, A_co]):
    """What a :class:`WindowedResponse` returns, read from a trial."""

    def returns(self) -> PortQuantity:
        """What the output is."""
        ...

    def output(self, trial: InMemoryTrajectoryStore[V]) -> A_co:
        """The output over the window ``trial`` stepped. Shape:
        (nport, 1)."""
        ...

    def unknown_tangent(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[UnknownTangent[V, A_co]]:
        """For a one-step ``trial``, the output's change for changes of the
        step's unknown (a block of columns), shape (nport, ncols); None
        when not supplied. Whether it is supplied is fixed for the
        output's life."""
        ...


class WindowPortRows(Generic[V, A]):
    r"""A subdomain's essential rows and its port rows, with their data
    over one window: the essential data :math:`g(t)` and, on the port
    rows, the linear interpolant from the committed state's trace at the
    window's start to the port data at its end.

    Satisfies :class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol` (for
    the stepper's pin) and :class:`WindowPortInputProtocol`, receiving a
    :class:`~pyapprox.pde.coupling.responses.Trace`. The rows, the rates'
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

    def nport(self) -> int:
        return self._ports.nrows()

    def receives(self) -> Trace:
        return Trace()

    def set_window(self, window: TimeWindow, state: V, port_data: A) -> None:
        """Hold ``window``, the trace interpolated from ``state``'s port
        rows at its start to ``port_data`` at its end.

        Raises
        ------
        ValueError
            If the window is steady.
        """
        if window.is_steady():
            raise ValueError("port data are interpolated over a nonempty window")
        self._window = window
        self._start, self._end = self._ports.take(state), port_data

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


class StepResidualReaction(Generic[V, A]):
    r"""The time-discrete reaction on the port rows: the last block of the
    unconstrained stepper's step residual there, summed over the window's
    steps (see the module docstring).

    Satisfies :class:`WindowPortOutputProtocol`, returning a
    :class:`~pyapprox.pde.coupling.responses.Flux`, for a trial stepped
    with the port rows pinned (:class:`WindowPortRows`). For a one-step
    trial its change with the step's unknown is the unconstrained step
    Jacobian's action, read on the port rows.

    Parameters
    ----------
    reaction : DifferentiableStepperProtocol
        The trials' method over the unconstrained system.
    ports : PortRowsProtocol
        The port rows of the state space, in port order.
    """

    def __init__(
        self,
        reaction: DifferentiableStepperProtocol[V, A],
        ports: PortRowsProtocol[V, A],
    ) -> None:
        self._reaction, self._ports = reaction, ports

    def returns(self) -> Flux:
        return Flux()

    def output(self, trial: InMemoryTrajectoryStore[V]) -> A:
        take, residual = self._ports.take, self._reaction.step_residual
        output = take(residual(trial.window(0)).blocks[-1])
        for step in range(1, trial.nsteps()):
            output = output + take(residual(trial.window(step)).blocks[-1])
        return output

    def unknown_tangent(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[UnknownTangent[V, A]]:
        """Supplied for a one-step trial."""
        if trial.nsteps() != 1:
            return None
        jacobian = self._reaction.diag(trial.window(0))

        def change(delta: BlockVector[V]) -> A:
            return self._ports.take(jacobian.apply(delta).blocks[-1])

        return change


class _TraceTangent(Generic[V, A]):
    r""":math:`(\partial r/\partial \lambda) D` at a one-step trial, for
    any method, pin placement and port input.

    The step residual :math:`R(u; \lambda)` is affine in the port data, so
    its change for the columns of :math:`D` is exactly
    :math:`R(u; \lambda + d_j) - R(u; \lambda)`, both evaluated by the
    stepper with the input holding those data. The unknown's change solves
    :math:`(\partial R/\partial u)\, \delta u = -\Delta R` with the step's
    block-triangular Jacobian, and the output's change is the output's
    tangent at :math:`\delta u`. Leaves the input holding the trial's
    own port data.
    """

    def __init__(
        self,
        stepper: StepperProtocol[V, A],
        port_input: WindowPortInputProtocol[V, A],
        window: TimeWindow,
        start: V,
        port_data: A,
        trial: InMemoryTrajectoryStore[V],
        change: UnknownTangent[V, A],
    ) -> None:
        self._stepper, self._input = stepper, port_input
        self._window, self._start, self._port_data = window, start, port_data
        self._step, self._change = trial.window(0), change

    def __call__(self, directions: A) -> A:
        nport = self._input.nport()
        if directions.ndim != 2 or directions.shape[0] != nport:
            raise ValueError(
                f"directions must have shape ({nport}, ncols), got "
                f"{tuple(directions.shape)}"
            )
        stepper, port_input = self._stepper, self._input
        space = stepper.residual_space()
        port_input.set_window(self._window, self._start, self._port_data)
        base = stepper.step_residual(self._step)
        changes = []
        for j in range(int(directions.shape[1])):
            shifted = self._port_data + directions[:, j : j + 1]
            port_input.set_window(self._window, self._start, shifted)
            # -(R(u; lambda + d_j) - R(u; lambda)), the tangent solve's rhs.
            changes.append(
                space.axpby(1.0, base, -1.0, stepper.step_residual(self._step))
            )
        port_input.set_window(self._window, self._start, self._port_data)
        delta, info = stepper.diag_handle(self._step).solve(space.concatenate(changes))
        if not info.converged:
            raise ValueError("the tangent solve did not converge")
        return self._change(delta)


class _Owner:
    """Identifies the response a point came from."""


class WindowedPoint(Generic[V, A]):
    """A :class:`WindowedResponse` stepped over one window at one set of
    port data. Its output and derivatives refer to its own trial, so they
    stay valid after the response is linearized again.
    """

    def __init__(
        self,
        owner: _Owner,
        window: TimeWindow,
        trial: InMemoryTrajectoryStore[V],
        output: A,
        bkd: Backend[A],
        tangent: Optional[TraceJVP[A]],
    ) -> None:
        self._owner, self._window, self._trial = owner, window, trial
        self._output, self._bkd = output, bkd
        self._derivs: ResponseDerivatives[A] = (
            ResponseDerivatives.none()
            if tangent is None
            else ResponseDerivatives.tangent(tangent)
        )

    def output(self) -> A:
        """The output over the window. Shape: (nport, 1)."""
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


class WindowedResponse(Generic[V, A]):
    r"""A stepped subdomain as a response over time windows (see the
    module docstring): the port data enter through ``port_input``, the
    output is read by ``port_output``. Without parameters or quantities
    of interest.

    Satisfies :class:`WindowedResponseProtocol`. The trace derivative
    (``jvp_trace``) is supplied for one step per window when
    ``port_output`` gives its change with the step's unknown: the step
    residual's change with the port data, solved with the step's
    Jacobian, for any method.

    Parameters
    ----------
    stepper : StepperProtocol
        Steps the subdomain, reading the port data ``port_input`` holds.
    port_input : WindowPortInputProtocol
        Sets the port data before each trial.
    port_output : WindowPortOutputProtocol
        Reads the output from each trial.
    init_state : V
        The state at ``origin``, a single column.
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
        If an argument does not satisfy its protocol.
    ValueError
        If ``nsubsteps`` is less than 1 or ``sink`` is not empty.
    """

    def __init__(
        self,
        stepper: StepperProtocol[V, A],
        port_input: WindowPortInputProtocol[V, A],
        port_output: WindowPortOutputProtocol[V, A],
        init_state: V,
        origin: float,
        sink: TrajectorySinkProtocol[V],
        bkd: Backend[A],
        nsubsteps: int = 1,
    ) -> None:
        for name, value, protocol in (
            ("stepper", stepper, StepperProtocol),
            ("port_input", port_input, WindowPortInputProtocol),
            ("port_output", port_output, WindowPortOutputProtocol),
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
        self._input, self._output = port_input, port_output
        self._state, self._time = init_state, float(origin)
        self._sink, self._bkd = sink, bkd
        self._nsubsteps = nsubsteps
        self._stepper = stepper
        self._integrator = Integrator(stepper)
        self._owner = _Owner()
        self._started = False

    def bkd(self) -> Backend[A]:
        return self._bkd

    def nport(self) -> int:
        return self._input.nport()

    def receives(self) -> PortQuantity:
        return self._input.receives()

    def returns(self) -> PortQuantity:
        return self._output.returns()

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
    ) -> WindowedPoint[V, A]:
        """Step over ``window`` from the committed state with
        ``port_data`` (nport, 1); ``params`` is (0, 1).

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
        self._input.set_window(window, self._state, port_data)
        size = (window.end - window.start) / self._nsubsteps
        trial: InMemoryTrajectoryStore[V] = InMemoryTrajectoryStore()
        self._integrator.solve(
            self._state, GivenSizes(window.start, [size] * self._nsubsteps), trial
        )
        change = self._output.unknown_tangent(trial)
        tangent: Optional[TraceJVP[A]] = (
            None
            if change is None
            else _TraceTangent(
                self._stepper,
                self._input,
                window,
                self._state,
                port_data,
                trial,
                change,
            )
        )
        return WindowedPoint(
            self._owner,
            window,
            trial,
            self._output.output(trial),
            self._bkd,
            tangent,
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
        if not isinstance(point, WindowedPoint) or not point.comes_from(self._owner):
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
