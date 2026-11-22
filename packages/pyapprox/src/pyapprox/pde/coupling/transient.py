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

from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from pyapprox.ode.integration import Integrator
from pyapprox.ode.pinned_rows import PinnedRowsParamDerivatives, PinnedRowsProtocol
from pyapprox.ode.protocols.differentiable_stepper import (
    DifferentiableStepperProtocol,
    StepperProtocol,
)
from pyapprox.ode.step_plan import GivenSizes
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.tableau import ButcherTableau
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

TrajectoryTangent = Callable[[Sequence[BlockVector[V]], Sequence[V]], A]
"""An output's change for changes of a trial's step unknowns and stored
states, each a block of columns: ``unknowns[m]`` step ``m``'s and
``states[k]`` the state after step ``k``. The trial's first state, the
committed one, does not change."""


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

    def trajectory_tangent(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryTangent[V, A_co]]:
        """The output's change for changes of ``trial``'s step unknowns
        and stored states, shape (nport, ncols); None when not supplied
        for this trial."""
        ...


def _read_changes(
    window: StepWindow[V], step: int, states: Sequence[V]
) -> list[tuple[int, V]]:
    """The changes of the stored states step ``step`` reads, by lag: state
    ``step + 1 - lag`` for each past state of ``window``, the committed
    first state (index 0) left out since it does not change; ``states[k
    - 1]`` is state ``k``'s change."""
    read = []
    for lag in range(1, len(window.past_states) + 1):
        index = step + 1 - lag
        if index >= 1:
            read.append((lag, states[index - 1]))
    return read


def _lagrange(nodes: Sequence[float], s: float) -> list[float]:
    """The Lagrange basis on ``nodes`` at ``s``."""
    weights = []
    for k, node in enumerate(nodes):
        weight = 1.0
        for m, other in enumerate(nodes):
            if m != k:
                weight *= (s - other) / (node - other)
        weights.append(weight)
    return weights


def _lagrange_derivative(nodes: Sequence[float], s: float) -> list[float]:
    """The derivative in ``s`` of the Lagrange basis on ``nodes``."""
    weights = []
    for k, node in enumerate(nodes):
        total = 0.0
        for m, other in enumerate(nodes):
            if m == k:
                continue
            term = 1.0 / (node - other)
            for n, third in enumerate(nodes):
                if n not in (k, m):
                    term *= (s - third) / (node - third)
            total += term
        weights.append(total)
    return weights


def runge_kutta_block_fractions(tableau: ButcherTableau) -> tuple[float, ...]:
    r"""The times of a Runge-Kutta step's blocks as fractions of the step:
    :math:`c_i` for each stage that is an unknown (a trivial first stage
    is the stored state), then 1 for the final block unless the method is
    stiffly accurate. With these as :class:`WindowPortRows`' fractions and
    every block pinned, each block reads its own port data.

    Raises
    ------
    ValueError
        If two blocks share a time or a block is at the step's start: one
        trace per time cannot tell them apart (Heun's last stage and final
        block are both at the step's end).
    """
    first = 1 if tableau.has_trivial_first_stage() else 0
    fractions = [float(tableau.c[i]) for i in range(first, tableau.nstages())]
    if not tableau.is_stiffly_accurate():
        fractions.append(1.0)
    if fractions[0] <= 0.0 or any(
        a >= b for a, b in zip(fractions[:-1], fractions[1:])
    ):
        raise ValueError(
            f"{tableau.name}'s blocks are at fractions {tuple(fractions)}: "
            "stage-wise port data need distinct times after the step's start"
        )
    return tuple(fractions)


class WindowPortRows(Generic[V, A]):
    r"""A subdomain's essential rows and its port rows, with their data
    over one window: the essential data :math:`g(t)` and, on the port
    rows, the Lagrange interpolant in time through the committed state's
    trace at the window's start and the port data at the window's
    ``fractions``, its rate the interpolant's derivative.

    The port data hold one trace per fraction, interleaved with the
    fraction as the fast index: entry ``q * p + j`` is port row ``p`` at
    fraction ``j`` of ``q``, as a :class:`~pyapprox.pde.coupling.spaces.
    ComponentSkeletonSpace` with one component per fraction lays out a
    skeleton vector. With the one fraction 1 the interpolant is linear from
    the start trace to the end trace. With a Runge-Kutta method's block
    times (:func:`runge_kutta_block_fractions`) and every block pinned,
    each block reads its own port data: stage-wise coupling.

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
    fractions : sequence of float, default (1.0,)
        Where in the window the port data are, as fractions of its
        length: strictly increasing in (0, 1], the last 1.

    Raises
    ------
    ValueError
        If ``rows`` does not count the essential and port rows once each,
        or ``fractions`` is not as described.
    """

    def __init__(
        self,
        essential: PinnedRowsProtocol[V],
        rows: RowSetProtocol[V],
        ports: PortRowsProtocol[V, A],
        states: VectorSpaceProtocol[V],
        fractions: Sequence[float] = (1.0,),
    ) -> None:
        expected = essential.rows().count() + ports.nrows()
        if rows.count() != expected:
            raise ValueError(
                f"rows must hold the {essential.rows().count()} essential and "
                f"{ports.nrows()} port rows once each; it has {rows.count()}"
            )
        nodes = (0.0,) + tuple(float(f) for f in fractions)
        increasing = all(a < b for a, b in zip(nodes[:-1], nodes[1:]))
        if len(nodes) < 2 or not increasing or nodes[-1] != 1.0:
            raise ValueError(
                "fractions must increase strictly in (0, 1] and end at 1, got "
                f"{tuple(fractions)}"
            )
        # The start trace at 0, then the port data's fractions.
        self._nodes = nodes
        self._essential, self._rows = essential, rows
        self._ports, self._states = ports, states
        # Read once: a declaration's rates are fixed for its life.
        self._essential_rates = essential.rates()
        self._rates: Optional[Callable[[float], V]] = (
            None if self._essential_rates is None else self._rate
        )
        self._window = TimeWindow(0.0, 1.0)
        self._traces_held: Optional[list[A]] = None

    def nport(self) -> int:
        return self._ports.nrows() * (len(self._nodes) - 1)

    def fractions(self) -> tuple[float, ...]:
        """Where in the window the port data are."""
        return self._nodes[1:]

    def receives(self) -> Trace:
        return Trace()

    def set_window(self, window: TimeWindow, state: V, port_data: A) -> None:
        """Hold ``window``, the trace interpolated through ``state``'s
        port rows at its start and ``port_data`` (:meth:`nport`, 1) at its
        fractions.

        Raises
        ------
        ValueError
            If the window is steady.
        """
        if window.is_steady():
            raise ValueError("port data are interpolated over a nonempty window")
        self._window = window
        nfractions = len(self._nodes) - 1
        self._traces_held = [self._ports.take(state)] + [
            port_data[j::nfractions] for j in range(nfractions)
        ]

    def _traces(self) -> list[A]:
        if self._traces_held is None:
            raise ValueError("no window is set")
        return self._traces_held

    def _combined(self, weights: Sequence[float]) -> A:
        traces = self._traces()
        total = weights[0] * traces[0]
        for weight, trace in zip(weights[1:], traces[1:]):
            total = total + weight * trace
        return total

    def _fraction(self, time: float) -> float:
        window = self._window
        return (time - window.start) / (window.end - window.start)

    def _placed(self, block: A) -> V:
        return self._ports.add_to(self._states.zeros(1), block)

    def rows(self) -> RowSetProtocol[V]:
        return self._rows

    def values(self, time: float) -> V:
        trace = self._combined(_lagrange(self._nodes, self._fraction(time)))
        return self._states.axpby(
            1.0, self._essential.values(time), 1.0, self._placed(trace)
        )

    def _rate(self, time: float) -> V:
        if self._essential_rates is None:
            raise ValueError("the essential data supply no rate")
        window = self._window
        scale = 1.0 / (window.end - window.start)
        weights = _lagrange_derivative(self._nodes, self._fraction(time))
        slope = self._combined([scale * w for w in weights])
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
    with the port rows pinned (:class:`WindowPortRows`). Its change is the
    unconstrained step Jacobians' action on each step's unknown and stored
    states, read on the port rows; supplied when the reaction stepper
    gives its past-state derivatives or the trial is one step.

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

    def trajectory_tangent(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryTangent[V, A]]:
        reaction = self._reaction
        windows = [trial.window(m) for m in range(trial.nsteps())]
        off_diag = reaction.derivatives().off_diag
        if off_diag is None and len(windows) > 1:
            return None
        space = reaction.residual_space()

        def change(unknowns: Sequence[BlockVector[V]], states: Sequence[V]) -> A:
            total: Optional[A] = None
            for m, window in enumerate(windows):
                block = reaction.diag(window).apply(unknowns[m])
                for lag, state in _read_changes(window, m, states):
                    # Unreachable without it: a one-step trial reads only
                    # the committed state, which does not change.
                    if off_diag is None:
                        raise ValueError("no past-state derivative")
                    coupled = off_diag(window, lag).apply(BlockVector((state,)))
                    block = space.axpby(1.0, coupled, 1.0, block, out=block)
                taken = self._ports.take(block.blocks[-1])
                total = taken if total is None else total + taken
            if total is None:
                raise ValueError("the trial has no steps")
            return total

        return change


class StageReactions(Generic[V, A]):
    r"""Each block's time-discrete reaction on the port rows: block
    :math:`j` of the unconstrained stepper's step residual there, for a
    one-step trial whose every block was pinned to its own port data
    (:class:`WindowPortRows` with :func:`runge_kutta_block_fractions`).
    Interleaved as the port data are: entry ``q * p + j`` is port row
    ``p`` of block ``j``. Summed over the subdomains at an interface, each
    block's entries are the single-domain residual of that block's
    interface rows, so the coupled step is the single-domain step.

    Satisfies :class:`WindowPortOutputProtocol`, returning a
    :class:`~pyapprox.pde.coupling.responses.Flux`.

    Parameters
    ----------
    reaction : DifferentiableStepperProtocol
        The trials' method over the unconstrained system.
    ports : PortRowsProtocol
        The port rows of the state space, in port order.
    bkd : Backend
        The backend of the port blocks.
    """

    def __init__(
        self,
        reaction: DifferentiableStepperProtocol[V, A],
        ports: PortRowsProtocol[V, A],
        bkd: Backend[A],
    ) -> None:
        self._reaction, self._ports, self._bkd = reaction, ports, bkd

    def returns(self) -> Flux:
        return Flux()

    def _interleaved(self, residual: BlockVector[V]) -> A:
        bkd = self._bkd
        blocks = [self._ports.take(block) for block in residual.blocks]
        stacked = bkd.stack(blocks, axis=1)
        nport, nblocks, ncols = stacked.shape
        return bkd.reshape(stacked, (nport * nblocks, ncols))

    def output(self, trial: InMemoryTrajectoryStore[V]) -> A:
        """Raises ValueError unless ``trial`` is one step."""
        if trial.nsteps() != 1:
            raise ValueError(
                f"stage-wise reactions read a one-step window, got {trial.nsteps()}"
            )
        return self._interleaved(self._reaction.step_residual(trial.window(0)))

    def trajectory_tangent(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryTangent[V, A]]:
        """Supplied for a one-step trial, which reads only the committed
        state: the change is the step Jacobian's on the unknown."""
        if trial.nsteps() != 1:
            return None
        jacobian = self._reaction.diag(trial.window(0))

        def change(unknowns: Sequence[BlockVector[V]], states: Sequence[V]) -> A:
            return self._interleaved(jacobian.apply(unknowns[0]))

        return change


class _TraceTangent(Generic[V, A]):
    r""":math:`(\partial r/\partial \lambda) D` at a trial of any number
    of steps, for any method, pin placement and port input.

    Each step residual :math:`R_m(u_m; y_m, \dots; \lambda)` is affine in
    the port data, so its change for the columns of :math:`D` at fixed
    unknown and states is exactly :math:`R_m(\lambda + d_j) -
    R_m(\lambda)`, both evaluated by the stepper with the input holding
    those data. A forward sweep then gives the trajectory's change: step
    :math:`m`'s unknown solves :math:`(\partial R_m/\partial u_m)\, \delta
    u_m = -\Delta R_m - \sum_l (\partial R_m/\partial y_{m+1-l})\, \delta
    y_{m+1-l}` with the step's block-triangular Jacobian, the new state
    is the readout of :math:`\delta u_m`, and the committed start does
    not change. The output's change is the output's tangent at those
    changes. Leaves the input holding the trial's own port data.
    """

    def __init__(
        self,
        stepper: StepperProtocol[V, A],
        port_input: WindowPortInputProtocol[V, A],
        window: TimeWindow,
        start: V,
        port_data: A,
        trial: InMemoryTrajectoryStore[V],
        change: TrajectoryTangent[V, A],
    ) -> None:
        self._stepper, self._input = stepper, port_input
        self._window, self._start, self._port_data = window, start, port_data
        self._steps = [trial.window(m) for m in range(trial.nsteps())]
        self._change = change

    def _data_changes(self, directions: A) -> list[BlockVector[V]]:
        """Per step, :math:`-\\Delta R_m` for every column of
        ``directions``, concatenated."""
        stepper, port_input = self._stepper, self._input
        space = stepper.residual_space()
        port_input.set_window(self._window, self._start, self._port_data)
        bases = [stepper.step_residual(step) for step in self._steps]
        columns: list[list[BlockVector[V]]] = [[] for _ in self._steps]
        for j in range(int(directions.shape[1])):
            shifted = self._port_data + directions[:, j : j + 1]
            port_input.set_window(self._window, self._start, shifted)
            for m, step in enumerate(self._steps):
                # -(R_m(lambda + d_j) - R_m(lambda)).
                columns[m].append(
                    space.axpby(1.0, bases[m], -1.0, stepper.step_residual(step))
                )
        port_input.set_window(self._window, self._start, self._port_data)
        return [space.concatenate(per_step) for per_step in columns]

    def __call__(self, directions: A) -> A:
        nport = self._input.nport()
        if directions.ndim != 2 or directions.shape[0] != nport:
            raise ValueError(
                f"directions must have shape ({nport}, ncols), got "
                f"{tuple(directions.shape)}"
            )
        stepper = self._stepper
        space = stepper.residual_space()
        off_diag = stepper.derivatives().off_diag
        unknowns: list[BlockVector[V]] = []
        states: list[V] = []
        for m, (step, rhs) in enumerate(
            zip(self._steps, self._data_changes(directions))
        ):
            for lag, state in _read_changes(step, m, states):
                if off_diag is None:
                    raise ValueError(
                        "a trial of several steps needs the stepper's past-state "
                        "derivatives"
                    )
                coupled = off_diag(step, lag).apply(BlockVector((state,)))
                rhs = space.axpby(-1.0, coupled, 1.0, rhs, out=rhs)
            delta, info = stepper.diag_handle(step).solve(rhs)
            if not info.converged:
                raise ValueError(f"the tangent solve of step {m} did not converge")
            unknowns.append(delta)
            states.append(stepper.readout().apply(delta).blocks[0])
        return self._change(unknowns, states)


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
    (``jvp_trace``) is supplied whenever ``port_output`` gives its change
    with the trial's unknowns and states: the step residuals' changes
    with the port data, carried through the window's steps by a forward
    tangent sweep, for any method (several steps need the stepper's
    past-state derivatives).

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
        change = self._output.trajectory_tangent(trial)
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
