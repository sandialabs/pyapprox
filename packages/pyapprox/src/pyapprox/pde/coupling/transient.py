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

from dataclasses import replace
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
from pyapprox.ode.mass_form_system import (
    MassFormSystem,
    RightHandSideProtocol,
    parameters_version_reader,
)
from pyapprox.ode.pinned_rows import PinnedRowsParamDerivatives, PinnedRowsProtocol
from pyapprox.ode.protocols.differentiable_stepper import (
    DifferentiableStepperProtocol,
    StepperProtocol,
)
from pyapprox.ode.residual_derivatives import ODEResidualDerivatives
from pyapprox.ode.residual_form import ResidualFormProtocol
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
    TraceVJP,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import (
    BlockVector,
    FormatProtocol,
    RowSetProtocol,
    SolverHandleProtocol,
    VectorSpaceProtocol,
)
from pyapprox.util.linalg.operators.linearization import LinearizationPoint

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
V_contra = TypeVar("V_contra", contravariant=True)
A_contra = TypeVar("A_contra", bound=ArrayProtocol, contravariant=True)
V_co = TypeVar("V_co", covariant=True)
F = TypeVar("F", bound=FormatProtocol)

TrajectoryTangent = Callable[[Sequence[BlockVector[V]], Sequence[V]], A]
"""An output's change for changes of a trial's step unknowns and stored
states, each a block of columns: ``unknowns[m]`` step ``m``'s and
``states[k]`` the state after step ``k``. The trial's first state, the
committed one, does not change."""

TrajectoryAdjoint = Callable[[A], tuple[list[BlockVector[V]], list[V]]]
"""The transpose of a :data:`TrajectoryTangent`: for an output cotangent
of one column, each step unknown's cotangent and each changing stored
state's, laid out as the tangent's arguments (``states[k]`` the state
after step ``k``)."""


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

    def accept(self, port_data: A_contra) -> None:
        """The data of the window just committed: the start of the next
        one for an input whose start is not read from the state."""
        ...


@runtime_checkable
class WindowPortOutputProtocol(Protocol[V, A]):
    """What a :class:`WindowedResponse` returns, read from a trial."""

    def returns(self) -> PortQuantity:
        """What the output is."""
        ...

    def output(self, trial: InMemoryTrajectoryStore[V]) -> A:
        """The output over the window ``trial`` stepped. Shape:
        (nport, 1)."""
        ...

    def reads_port_data(self) -> bool:
        """Whether :meth:`output` reads the port data the input holds
        beside the trial, affinely; its change with them is then taken
        by differences at the trial. Fixed for the output's life."""
        ...

    def trajectory_tangent(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryTangent[V, A]]:
        """The output's change for changes of ``trial``'s step unknowns
        and stored states, shape (nport, ncols); None when not supplied
        for this trial."""
        ...

    def trajectory_adjoint(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryAdjoint[A, V]]:
        """The transpose of :meth:`trajectory_tangent` at ``trial``; None
        when not supplied for this trial."""
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


class _LinearizedResiduals(Generic[V, A]):
    """Each step residual's change of a trial for changes of its step
    unknowns and stored states: the step Jacobian on the unknown plus the
    past-state couplings."""

    def __init__(
        self,
        stepper: DifferentiableStepperProtocol[V, A],
        windows: list[StepWindow[V]],
    ) -> None:
        self._stepper, self._windows = stepper, windows

    @staticmethod
    def of(
        stepper: DifferentiableStepperProtocol[V, A],
        trial: InMemoryTrajectoryStore[V],
    ) -> "Optional[_LinearizedResiduals[V, A]]":
        """For ``trial``; None for a trial of several steps when the
        stepper gives no past-state derivatives, or for an empty one."""
        windows = [trial.window(m) for m in range(trial.nsteps())]
        if not windows:
            return None
        if stepper.derivatives().off_diag is None and len(windows) > 1:
            return None
        return _LinearizedResiduals(stepper, windows)

    def __call__(
        self, unknowns: Sequence[BlockVector[V]], states: Sequence[V]
    ) -> list[BlockVector[V]]:
        stepper = self._stepper
        off_diag = stepper.derivatives().off_diag
        space = stepper.residual_space()
        changes = []
        for m, window in enumerate(self._windows):
            block = stepper.diag(window).apply(unknowns[m])
            for lag, state in _read_changes(window, m, states):
                # Unreachable without it: a one-step trial reads only the
                # committed state, which does not change.
                if off_diag is None:
                    raise ValueError("no past-state derivative")
                coupled = off_diag(window, lag).apply(BlockVector((state,)))
                block = space.axpby(1.0, coupled, 1.0, block, out=block)
            changes.append(block)
        return changes

    def transpose(
        self, cotangents: Sequence[BlockVector[V]]
    ) -> tuple[list[BlockVector[V]], list[V]]:
        """The transpose of :meth:`__call__`: for each step residual's
        cotangent, each step unknown's cotangent and each changing stored
        state's (``states[k - 1]`` state ``k``'s)."""
        stepper = self._stepper
        off_diag = stepper.derivatives().off_diag
        space = stepper.state_space().component(0)
        unknowns = []
        states: list[Optional[V]] = [None] * len(self._windows)
        for m, window in enumerate(self._windows):
            unknowns.append(stepper.diag(window).apply_transpose(cotangents[m]))
            for lag in range(1, len(window.past_states) + 1):
                # The committed first state does not change.
                index = m + 1 - lag
                if index < 1:
                    continue
                if off_diag is None:
                    raise ValueError("no past-state derivative")
                pulled = off_diag(window, lag).apply_transpose(cotangents[m]).blocks[0]
                held = states[index - 1]
                states[index - 1] = (
                    pulled if held is None else space.axpby(1.0, pulled, 1.0, held)
                )
        return unknowns, [space.zeros(1) if s is None else s for s in states]


def _on_blocks(
    stepper: DifferentiableStepperProtocol[V, A],
    ports: PortRowsProtocol[V, A],
    blocks: dict[int, A],
) -> BlockVector[V]:
    r"""A residual cotangent that is :math:`R^\top` ``blocks[i]`` on block
    ``i``, zero elsewhere: the transpose of reading the port rows of
    residual blocks."""
    parts = list(stepper.residual_space().zeros(1).blocks)
    for i, block in blocks.items():
        parts[i] = ports.add_to(parts[i], block)
    return BlockVector(tuple(parts))


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


@runtime_checkable
class TemporalTrialSpaceProtocol(Protocol):
    r"""The port data's space in time over one window, as
    :class:`WindowPortRows` reads it: at fraction :math:`s` of the window,
    :math:`\lambda(s) = c_0(s)\, \lambda_{\mathrm{start}} + \sum_{k \ge 1}
    c_k(s)\, \lambda_k`, with :math:`\lambda_{\mathrm{start}}` the
    committed state's trace and :math:`\lambda_k` the window's unknown
    traces. A space that does not read the start (a jump at the window's
    start) has :math:`c_0 = 0`. Fixed for its life."""

    def nunknowns(self) -> int:
        """The number of unknown traces per port row."""
        ...

    def coefficients(self, fraction: float) -> Sequence[float]:
        """:math:`(c_0, \\dots, c_n)` at ``fraction``, length
        :meth:`nunknowns` + 1, the start's first."""
        ...

    def rate_coefficients(self, fraction: float) -> Sequence[float]:
        """Their derivatives in the fraction, laid out as
        :meth:`coefficients`."""
        ...


class StartContinuousLagrange:
    """The Lagrange interpolant in time through the committed trace at
    the window's start and one unknown trace at each of ``fractions``:
    polynomials of degree ``len(fractions)``, continuous across windows.
    Satisfies :class:`TemporalTrialSpaceProtocol`.

    Parameters
    ----------
    fractions : sequence of float, default (1.0,)
        Where in the window the unknown traces are, as fractions of its
        length: strictly increasing in (0, 1], the last 1, so the
        window's end trace is an unknown and the next window starts from
        it.

    Raises
    ------
    ValueError
        If ``fractions`` is not as described.
    """

    def __init__(self, fractions: Sequence[float] = (1.0,)) -> None:
        nodes = (0.0,) + tuple(float(f) for f in fractions)
        increasing = all(a < b for a, b in zip(nodes[:-1], nodes[1:]))
        if len(nodes) < 2 or not increasing or nodes[-1] != 1.0:
            raise ValueError(
                "fractions must increase strictly in (0, 1] and end at 1, got "
                f"{tuple(fractions)}"
            )
        self._nodes = nodes

    def fractions(self) -> tuple[float, ...]:
        """Where in the window the unknown traces are."""
        return self._nodes[1:]

    def nunknowns(self) -> int:
        return len(self._nodes) - 1

    def coefficients(self, fraction: float) -> list[float]:
        return _lagrange(self._nodes, fraction)

    def rate_coefficients(self, fraction: float) -> list[float]:
        return _lagrange_derivative(self._nodes, fraction)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.fractions()})"


def runge_kutta_block_fractions(tableau: ButcherTableau) -> tuple[float, ...]:
    r"""The times of a Runge-Kutta step's blocks as fractions of the step:
    :math:`c_i` for each stage that is an unknown (a trivial first stage
    is the stored state), then 1 for the final block unless the method is
    stiffly accurate. With these as a :class:`StartContinuousLagrange`'s
    fractions and every block pinned, each block reads its own port data.

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
    rows, the port data's function of time in ``trial``, built from the
    committed state's trace at the window's start and the window's unknown
    traces, its rate that function's derivative.

    The port data hold one trace per unknown of ``trial``, interleaved
    with the unknown as the fast index: entry ``q * p + j`` is port row
    ``p``'s unknown ``j`` of ``q``, as a :class:`~pyapprox.pde.coupling.
    spaces.ComponentSkeletonSpace` with one component per unknown lays
    out a skeleton vector. With :class:`StartContinuousLagrange` at the
    one fraction 1 the trace is linear from the start trace to the end
    trace. With a Runge-Kutta method's block times
    (:func:`runge_kutta_block_fractions`) and every block pinned, each
    block reads its own port data: stage-wise coupling.

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
    trial : TemporalTrialSpaceProtocol, optional
        The port data's space in time;
        :class:`StartContinuousLagrange` at the fraction 1 when omitted.

    Raises
    ------
    TypeError
        If ``trial`` does not satisfy its protocol.
    ValueError
        If ``rows`` does not count the essential and port rows once each.
    """

    def __init__(
        self,
        essential: PinnedRowsProtocol[V],
        rows: RowSetProtocol[V],
        ports: PortRowsProtocol[V, A],
        states: VectorSpaceProtocol[V],
        trial: Optional[TemporalTrialSpaceProtocol] = None,
    ) -> None:
        expected = essential.rows().count() + ports.nrows()
        if rows.count() != expected:
            raise ValueError(
                f"rows must hold the {essential.rows().count()} essential and "
                f"{ports.nrows()} port rows once each; it has {rows.count()}"
            )
        chosen: TemporalTrialSpaceProtocol = (
            StartContinuousLagrange() if trial is None else trial
        )
        if not isinstance(chosen, TemporalTrialSpaceProtocol):
            raise TypeError(
                "trial must satisfy TemporalTrialSpaceProtocol, got "
                f"{type(chosen).__name__}"
            )
        self._trial, self._nunknowns = chosen, chosen.nunknowns()
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
        return self._ports.nrows() * self._nunknowns

    def trial(self) -> TemporalTrialSpaceProtocol:
        """The port data's space in time."""
        return self._trial

    def receives(self) -> Trace:
        return Trace()

    def set_window(self, window: TimeWindow, state: V, port_data: A) -> None:
        """Hold ``window``, ``state``'s port rows as the start trace and
        ``port_data`` (:meth:`nport`, 1) as the unknown traces.

        Raises
        ------
        ValueError
            If the window is steady.
        """
        if window.is_steady():
            raise ValueError("port data are interpolated over a nonempty window")
        self._window = window
        n = self._nunknowns
        self._traces_held = [self._ports.take(state)] + [
            port_data[j::n] for j in range(n)
        ]

    def accept(self, port_data: A) -> None:
        """Nothing to keep: the next window's start trace is the committed
        state's, which the pinned end rows hold."""

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
        trace = self._combined(self._trial.coefficients(self._fraction(time)))
        return self._states.axpby(
            1.0, self._essential.values(time), 1.0, self._placed(trace)
        )

    def _rate(self, time: float) -> V:
        if self._essential_rates is None:
            raise ValueError("the essential data supply no rate")
        window = self._window
        scale = 1.0 / (window.end - window.start)
        weights = self._trial.rate_coefficients(self._fraction(time))
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


class WindowPortLoad(Generic[V, A]):
    r"""Port data entering as a natural load on the port rows over one
    window: :math:`+B \lambda_h(t)` on the right-hand side, with
    :math:`\lambda_h` the port data's function of time in ``trial``. In
    right-hand-side form :math:`F = f - K u` a tested outward flux in
    Neumann form adds with sign +, as for :class:`~pyapprox.pde.coupling.
    loaded_response.SteadyLoadedResponse`.

    A load's start is not read from the state, so the input keeps it:
    ``initial`` at first, then on each commit the accepted window's data
    at its end (:meth:`accept`), so the load is continuous across windows
    for a trial space continuous through its start.

    Satisfies :class:`WindowPortInputProtocol`; the port data are laid
    out as :class:`WindowPortRows`' are.

    Parameters
    ----------
    ports : PortRowsProtocol
        The port rows of the state space, in port order.
    states : VectorSpaceProtocol
        The state space.
    trial : TemporalTrialSpaceProtocol
        The port data's space in time.
    initial : A
        The data at the first window's start. Shape: (nrows, 1).
    receives : PortQuantity, default Flux()
        What the port data are.

    Raises
    ------
    TypeError
        If ``trial`` does not satisfy its protocol.
    ValueError
        If ``initial`` is not one value per port row.
    """

    def __init__(
        self,
        ports: PortRowsProtocol[V, A],
        states: VectorSpaceProtocol[V],
        trial: TemporalTrialSpaceProtocol,
        initial: A,
        receives: PortQuantity = Flux(),
    ) -> None:
        if not isinstance(trial, TemporalTrialSpaceProtocol):
            raise TypeError(
                "trial must satisfy TemporalTrialSpaceProtocol, got "
                f"{type(trial).__name__}"
            )
        if tuple(initial.shape) != (ports.nrows(), 1):
            raise ValueError(
                f"initial must have shape ({ports.nrows()}, 1), got "
                f"{tuple(initial.shape)}"
            )
        self._ports, self._states, self._trial = ports, states, trial
        self._n, self._receives = trial.nunknowns(), receives
        self._start = initial
        self._window = TimeWindow(0.0, 1.0)
        self._traces_held: Optional[list[A]] = None

    def nport(self) -> int:
        return self._ports.nrows() * self._n

    def receives(self) -> PortQuantity:
        return self._receives

    def trial(self) -> TemporalTrialSpaceProtocol:
        """The port data's space in time."""
        return self._trial

    def _split(self, port_data: A) -> list[A]:
        n = self._n
        return [self._start] + [port_data[j::n] for j in range(n)]

    def set_window(self, window: TimeWindow, state: V, port_data: A) -> None:
        """Hold ``window`` and ``port_data`` (:meth:`nport`, 1); the start
        is the kept data, not ``state``.

        Raises
        ------
        ValueError
            If the window is steady.
        """
        if window.is_steady():
            raise ValueError("a load is interpolated over a nonempty window")
        self._window, self._traces_held = window, self._split(port_data)

    def accept(self, port_data: A) -> None:
        """Keep the accepted data's value at the window's end as the next
        window's start."""
        self._start = _combined(self._split(port_data), self._trial.coefficients(1.0))

    def load(self, time: float) -> V:
        """:math:`B \\lambda_h(t)`, a single column of the state space.

        Raises
        ------
        ValueError
            If no window is set.
        """
        if self._traces_held is None:
            raise ValueError("no window is set")
        window = self._window
        fraction = (time - window.start) / (window.end - window.start)
        value = _combined(self._traces_held, self._trial.coefficients(fraction))
        return self._ports.add_to(self._states.zeros(1), value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nport={self.nport()}, trial={self._trial!r})"


def _combined(traces: Sequence[A], weights: Sequence[float]) -> A:
    """:math:`\\sum_k w_k\\, t_k`."""
    total = weights[0] * traces[0]
    for weight, trace in zip(weights[1:], traces[1:]):
        total = total + weight * trace
    return total


@runtime_checkable
class TimeLoadProtocol(Protocol[V_co]):
    """A load on the state's rows at a time, as
    :class:`LoadedRightHandSide` reads it."""

    def load(self, time: float) -> V_co:
        """A single column of the state space."""
        ...


class LoadedRightHandSide(Generic[V, F]):
    r""":math:`f(y, t) + L(t)`: a right-hand side plus a load that does
    not depend on the state or the parameters, so every derivative is
    ``rhs``'s. Satisfies :class:`~pyapprox.ode.mass_form_system.
    RightHandSideProtocol`, states in the residual's form.

    Parameters
    ----------
    rhs : RightHandSideProtocol
        :math:`f`.
    load : TimeLoadProtocol
        :math:`L`, read at each call.
    form : ResidualFormProtocol
        Converts between the state space's columns and the residual's
        form.
    space : VectorSpaceProtocol
        The state space, where the load is added.
    """

    def __init__(
        self,
        rhs: RightHandSideProtocol[V, F],
        load: TimeLoadProtocol[V],
        form: ResidualFormProtocol[V, V],
        space: VectorSpaceProtocol[V],
    ) -> None:
        self._rhs, self._load, self._form, self._space = rhs, load, form, space
        self._version = parameters_version_reader(rhs)

    def __call__(self, state: V, time: float) -> V:
        form = self._form
        value = form.from_residual(self._rhs(state, time))
        loaded = self._space.axpby(1.0, value, 1.0, self._load.load(time))
        return form.to_residual(loaded)

    def derivatives(self) -> ODEResidualDerivatives[V, F]:
        return self._rhs.derivatives()

    def parameters_version(self) -> int:
        return self._version()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._rhs!r})"


def loaded_system(
    system: MassFormSystem[V, A, F], load: TimeLoadProtocol[V]
) -> MassFormSystem[V, A, F]:
    """``system`` with ``load`` added to its right-hand side."""
    rhs = LoadedRightHandSide(system.rhs, load, system.form, system.space)
    return replace(system, rhs=rhs)


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

    def reads_port_data(self) -> bool:
        """The unconstrained residual does not read the port data."""
        return False

    def output(self, trial: InMemoryTrajectoryStore[V]) -> A:
        take, residual = self._ports.take, self._reaction.step_residual
        output = take(residual(trial.window(0)).blocks[-1])
        for step in range(1, trial.nsteps()):
            output = output + take(residual(trial.window(step)).blocks[-1])
        return output

    def trajectory_tangent(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryTangent[V, A]]:
        linearized = _LinearizedResiduals.of(self._reaction, trial)
        if linearized is None:
            return None
        take = self._ports.take

        def change(unknowns: Sequence[BlockVector[V]], states: Sequence[V]) -> A:
            blocks = linearized(unknowns, states)
            total = take(blocks[0].blocks[-1])
            for block in blocks[1:]:
                total = total + take(block.blocks[-1])
            return total

        return change

    def trajectory_adjoint(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryAdjoint[A, V]]:
        """Supplied when :meth:`trajectory_tangent` is: every step's last
        residual block reads the cotangent on the port rows."""
        linearized = _LinearizedResiduals.of(self._reaction, trial)
        if linearized is None:
            return None
        last = self._reaction.residual_space().nblocks() - 1
        nsteps = trial.nsteps()

        def adjoint(cotangent: A) -> tuple[list[BlockVector[V]], list[V]]:
            on_last = _on_blocks(self._reaction, self._ports, {last: cotangent})
            return linearized.transpose([on_last] * nsteps)

        return adjoint


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

    def reads_port_data(self) -> bool:
        """The unconstrained residual does not read the port data."""
        return False

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

    def trajectory_adjoint(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryAdjoint[A, V]]:
        """Supplied for a one-step trial, as the tangent is: block ``j``
        reads entries ``q * p + j`` of the cotangent."""
        if trial.nsteps() != 1:
            return None
        jacobian = self._reaction.diag(trial.window(0))
        nblocks = self._reaction.residual_space().nblocks()

        def adjoint(cotangent: A) -> tuple[list[BlockVector[V]], list[V]]:
            per_block = self._bkd.reshape(cotangent, (-1, nblocks))
            blocks = {j: per_block[:, j : j + 1] for j in range(nblocks)}
            residual = _on_blocks(self._reaction, self._ports, blocks)
            # The step's end state enters only through its unknown.
            end = self._reaction.state_space().component(0).zeros(1)
            return [jacobian.apply_transpose(residual)], [end]

        return adjoint


@runtime_checkable
class TemporalTestFunctionsProtocol(Protocol):
    r"""The functions :math:`\psi_k` of the fraction :math:`s` of a window
    a flux is tested against in time, as :class:`StageQuadratureFlux`
    reads them. Fixed for their life."""

    def count(self) -> int:
        """The number of test functions."""
        ...

    def values(self, fraction: float) -> Sequence[float]:
        r""":math:`(\psi_1, \dots, \psi_n)` at ``fraction``, length
        :meth:`count`."""
        ...


class LagrangeTestFunctions:
    r"""The Lagrange basis on ``nodes`` of the polynomials of degree
    ``len(nodes) - 1`` in the window's fraction. With :math:`q` nodes and
    a trial space of degree :math:`q` continuous through the start
    (:class:`StartContinuousLagrange` on :math:`q` fractions) this is
    continuous Galerkin in time, cG(q): the test functions sum to one, so
    the flux tested against their sum is the window's whole flux and the
    coupled solve conserves it exactly per window. Satisfies
    :class:`TemporalTestFunctionsProtocol`.

    Parameters
    ----------
    nodes : sequence of float
        Distinct fractions in [0, 1]; any choice spans the same
        polynomials, so the coupled solution does not depend on it.

    Raises
    ------
    ValueError
        If ``nodes`` is empty, has repeats or leaves [0, 1].
    """

    def __init__(self, nodes: Sequence[float]) -> None:
        chosen = tuple(float(n) for n in nodes)
        if (
            not chosen
            or len(set(chosen)) != len(chosen)
            or any(not 0.0 <= n <= 1.0 for n in chosen)
        ):
            raise ValueError(f"nodes must be distinct fractions in [0, 1], got {nodes}")
        self._nodes = chosen

    def count(self) -> int:
        return len(self._nodes)

    def values(self, fraction: float) -> list[float]:
        return _lagrange(self._nodes, fraction)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._nodes})"


def _inverse(matrix: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    """The inverse of a small square ``matrix`` of method coefficients,
    by Gauss-Jordan elimination with partial pivoting.

    Raises
    ------
    ValueError
        If ``matrix`` is singular.
    """
    n = len(matrix)
    work = [
        list(map(float, row)) + [float(i == j) for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < 1e-14:
            raise ValueError(
                f"{name}'s stage reactions are not determined by its step blocks"
            )
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [value / scale for value in work[col]]
        for row in range(n):
            if row != col and work[row][col] != 0.0:
                factor = work[row][col]
                work[row] = [v - factor * w for v, w in zip(work[row], work[col])]
    return [row[n:] for row in work]


@runtime_checkable
class StepStartReactionProtocol(Protocol[V]):
    r"""The port reaction at a step's start, :math:`\rho = M k - f(y)`
    with :math:`k` the stepped (differentiated) system's slope at the
    stored state, as :class:`StageQuadratureFlux` reads it for a method
    whose first stage is that state and whose blocks do not determine
    its reaction. Read with the port data the input holds."""

    def reaction(self, state: V, time: float) -> V:
        """:math:`M k - f(y)` at ``state``, a single column of the state
        space."""
        ...

    def tangent(self, state: V, time: float, delta: V, version: int) -> V:
        """Its change for changes ``delta`` (a block of columns) of the
        state, at fixed port data, parameters at ``version``."""
        ...

    def adjoint(self, state: V, time: float, cotangent: V, version: int) -> V:
        """The transpose of :meth:`tangent` at the same point: the state's
        cotangent for a cotangent of the reaction (a block of columns)."""
        ...


class MassFormStartReaction(Generic[V, A, F]):
    r""":math:`\rho = M k - f(y)` with :math:`M_s k = f_s(y)`: the slope
    of the stepped system (its port and essential rows in differentiated
    form, so :math:`k` there is the data's rate) put into the
    unconstrained system's equation. Satisfies
    :class:`StepStartReactionProtocol`.

    The stepped system's mass is set up once and kept: a mass-form
    system's mass does not change. A singular stepped mass (a saddle-point
    system such as Stokes) cannot be solved with; such systems need a
    method without an explicit first stage.

    Parameters
    ----------
    solved : MassFormSystem
        The stepped system.
    unconstrained : MassFormSystem
        The system without pinned or differentiated rows, on the same
        space.
    """

    def __init__(
        self,
        solved: MassFormSystem[V, A, F],
        unconstrained: MassFormSystem[V, A, F],
    ) -> None:
        self._solved, self._unconstrained = solved, unconstrained
        self._mass = solved.solver.setup(solved.mass)
        # Made on the first adjoint: a forward-only run never needs it.
        self._mass_transpose: Optional[SolverHandleProtocol[V]] = None

    @staticmethod
    def _rhs(system: MassFormSystem[V, A, F], state: V, time: float) -> V:
        form = system.form
        return form.from_residual(system.rhs(form.to_residual(state), time))

    def _slope(self, value: V) -> V:
        slope, info = self._mass.solve(value)
        if not info.converged:
            raise ValueError("the step-start slope's mass solve did not converge")
        return slope

    def reaction(self, state: V, time: float) -> V:
        free = self._unconstrained
        slope = self._slope(self._rhs(self._solved, state, time))
        return free.space.axpby(
            1.0, free.mass.apply(slope), -1.0, self._rhs(free, state, time)
        )

    def tangent(self, state: V, time: float, delta: V, version: int) -> V:
        def jacobian(system: MassFormSystem[V, A, F]) -> V:
            point = LinearizationPoint(version, object())
            at = system.form.to_residual(state)
            return system.jacobian(at, time, point).apply(delta)

        free = self._unconstrained
        slope = self._slope(jacobian(self._solved))
        return free.space.axpby(1.0, free.mass.apply(slope), -1.0, jacobian(free))

    def adjoint(self, state: V, time: float, cotangent: V, version: int) -> V:
        r""":math:`J_s^\top M_s^{-\top} M^\top \bar\rho - J^\top \bar\rho`.

        Raises
        ------
        TypeError
            If the stepped mass's solver cannot solve transposed.
        """

        def transposed(system: MassFormSystem[V, A, F], vec: V) -> V:
            point = LinearizationPoint(version, object())
            at = system.form.to_residual(state)
            return system.jacobian(at, time, point).apply_transpose(vec)

        if self._mass_transpose is None:
            transpose = self._mass.transpose()
            if transpose is None:
                raise TypeError(
                    "the stepped mass's solver cannot solve transposed, so the "
                    "step-start reaction has no adjoint"
                )
            self._mass_transpose = transpose
        free = self._unconstrained
        weighted, info = self._mass_transpose.solve(
            free.mass.apply_transpose(cotangent)
        )
        if not info.converged:
            raise ValueError("the step-start adjoint's mass solve did not converge")
        return free.space.axpby(
            1.0, transposed(self._solved, weighted), -1.0, transposed(free, cotangent)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class StageQuadratureFlux(Generic[V, A]):
    r"""The port flux tested in time, :math:`\int \psi_k(s)\, \rho\, dt`
    over the window for each test function, by the Runge-Kutta method's
    own quadrature: :math:`\sum_m \Delta t_m \sum_j b_j\, \psi_k(s_{m,j})
    \rho_{m,j}`, :math:`s_{m,j}` stage :math:`j` of step :math:`m` as a
    fraction of the window.

    :math:`\rho_j = M k_j - f(Y_j)` is stage :math:`j`'s reaction on the
    port rows. The unconstrained stepper's stage block :math:`i` is
    :math:`M(Y_i - y_n) - \Delta t \sum_j a_{ij} f(Y_j) = \Delta t \sum_j
    a_{ij} \rho_j`, and its final block, when the method has one,
    :math:`\Delta t \sum_j b_j \rho_j`: the blocks are :math:`R = \Delta
    t\, W \rho`. The stage blocks are used, and the final block too when a
    trivial first stage (the stored state) leaves them one short, as for
    explicit methods; :math:`W` is then square and solved once from the
    tableau. A stiffly accurate method with a trivial first stage
    (Crank-Nicolson) has no final block, so its first reaction comes from
    ``start`` and the others from the remaining stage blocks. For
    :math:`\psi \equiv 1` the sum is the step's final-block residual (for
    a stiffly accurate method :math:`b` is :math:`A`'s last row, whatever
    the first reaction), the time-discrete reaction, so a constant among
    the test functions' span gives exact conservation whatever each
    side's method and step.

    Interleaved as the port data are: entry ``q * p + k`` is port row
    ``p`` tested against function ``k`` of ``q``. Its count must equal the
    trial space's number of unknowns for the skeleton residual to be
    square; the builder checks it. Satisfies
    :class:`WindowPortOutputProtocol`, returning a
    :class:`~pyapprox.pde.coupling.responses.Flux`, with its change for
    any number of steps when the reaction stepper gives its past-state
    derivatives.

    Parameters
    ----------
    reaction : DifferentiableStepperProtocol
        The trials' Runge-Kutta method over the unconstrained system.
    tableau : ButcherTableau
        Its tableau.
    ports : PortRowsProtocol
        The port rows of the state space, in port order.
    test : TemporalTestFunctionsProtocol
        The test functions.
    bkd : Backend
        The backend of the port blocks.
    start : StepStartReactionProtocol, optional
        The reaction at each step's start; read only for a stiffly
        accurate method with a trivial first stage, which needs it.

    Raises
    ------
    TypeError
        If ``test`` or ``start`` does not satisfy its protocol.
    ValueError
        If the method needs ``start`` and it is omitted, or its blocks do
        not determine its stage reactions.
    """

    def __init__(
        self,
        reaction: DifferentiableStepperProtocol[V, A],
        tableau: ButcherTableau,
        ports: PortRowsProtocol[V, A],
        test: TemporalTestFunctionsProtocol,
        bkd: Backend[A],
        start: Optional[StepStartReactionProtocol[V]] = None,
    ) -> None:
        if not isinstance(test, TemporalTestFunctionsProtocol):
            raise TypeError(
                "test must satisfy TemporalTestFunctionsProtocol, got "
                f"{type(test).__name__}"
            )
        if start is not None and not isinstance(start, StepStartReactionProtocol):
            raise TypeError(
                "start must satisfy StepStartReactionProtocol, got "
                f"{type(start).__name__}"
            )
        nstages = tableau.nstages()
        first = 1 if tableau.has_trivial_first_stage() else 0
        # The step's residual blocks: the unknown stages, then the final
        # block unless the method is stiffly accurate.
        rows: list[tuple[int, tuple[float, ...]]] = [
            (i - first, tuple(tableau.a[i])) for i in range(first, nstages)
        ]
        if len(rows) < nstages and not tableau.is_stiffly_accurate():
            rows.append((nstages - first, tuple(tableau.b)))
        self._needs_start = len(rows) < nstages
        if self._needs_start and start is None:
            raise ValueError(
                f"{tableau.name} is stiffly accurate with a trivial first stage: "
                "its first stage reaction needs the step-start reaction"
            )
        # The stages whose reactions the blocks give.
        self._solved = list(range(1 if self._needs_start else 0, nstages))
        inverse = _inverse(
            [[row[j] for j in self._solved] for _, row in rows], tableau.name
        )
        self._reaction, self._ports, self._test, self._bkd = (
            reaction,
            ports,
            test,
            bkd,
        )
        self._start = start
        self._c, self._b = tableau.c, tableau.b
        self._blocks = [block for block, _ in rows]
        self._inverse = inverse
        # Each equation's coefficient of the first reaction, moved to the
        # known side when it comes from the start.
        self._first = [row[0] for _, row in rows]

    def returns(self) -> Flux:
        return Flux()

    def nfunctions(self) -> int:
        """The number of test functions."""
        return self._test.count()

    def reads_port_data(self) -> bool:
        """Whether the output reads the port data beside the trial: the
        step-start reaction does, through the trial's rate."""
        return self._needs_start

    def _weights(
        self, windows: Sequence[StepWindow[V]]
    ) -> list[tuple[list[list[float]], list[float]]]:
        """Per step: per test function the weight of each used block's
        port rows, and of the step-start reaction's."""
        origin = windows[0].time()
        last = windows[-1]
        length = last.time() + last.deltat() - origin
        ntests, nblocks = self._test.count(), len(self._blocks)
        weights = []
        for window in windows:
            deltat = window.deltat()

            def psi(j: int) -> Sequence[float]:
                time = window.time() + self._c[j] * deltat
                return self._test.values((time - origin) / length)

            blocks = [[0.0] * nblocks for _ in range(ntests)]
            start = [0.0] * ntests
            for jj, j in enumerate(self._solved):
                for k, value in enumerate(psi(j)):
                    for r in range(nblocks):
                        weight = self._b[j] * value * self._inverse[jj][r]
                        blocks[k][r] += weight
                        start[k] -= deltat * weight * self._first[r]
            if self._needs_start:
                for k, value in enumerate(psi(0)):
                    start[k] += deltat * self._b[0] * value
            weights.append((blocks, start))
        return weights

    def _tested(
        self,
        windows: Sequence[StepWindow[V]],
        residuals: Sequence[BlockVector[V]],
        starts: Sequence[Optional[V]],
    ) -> A:
        """The output from each step's residual blocks and, when the
        method needs it, its step-start reaction (None where it does not
        change)."""
        bkd, take = self._bkd, self._ports.take
        tested = []
        weights = self._weights(windows)
        for k in range(self._test.count()):
            total: Optional[A] = None
            for (blocks, start), residual, reaction in zip(weights, residuals, starts):
                terms = [
                    (weight, residual.blocks[block])
                    for weight, block in zip(blocks[k], self._blocks)
                ]
                if self._needs_start and reaction is not None:
                    terms.append((start[k], reaction))
                for weight, vector in terms:
                    if weight != 0.0:
                        term = weight * take(vector)
                        total = term if total is None else total + term
            if total is None:
                raise ValueError("the trial has no steps")
            tested.append(total)
        stacked = bkd.stack(tested, axis=1)
        nport, ntests, ncols = stacked.shape
        return bkd.reshape(stacked, (nport * ntests, ncols))

    def _start_reactions(self, windows: Sequence[StepWindow[V]]) -> list[Optional[V]]:
        start = self._start
        if not self._needs_start or start is None:
            return [None] * len(windows)
        return [start.reaction(w.past_state(1), w.time()) for w in windows]

    def output(self, trial: InMemoryTrajectoryStore[V]) -> A:
        """Raises ValueError if ``trial`` has no steps."""
        windows = [trial.window(m) for m in range(trial.nsteps())]
        if not windows:
            raise ValueError("the trial has no steps")
        residual = self._reaction.step_residual
        return self._tested(
            windows,
            [residual(window) for window in windows],
            self._start_reactions(windows),
        )

    def trajectory_tangent(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryTangent[V, A]]:
        linearized = _LinearizedResiduals.of(self._reaction, trial)
        if linearized is None:
            return None
        windows = [trial.window(m) for m in range(trial.nsteps())]
        start = self._start

        def change(unknowns: Sequence[BlockVector[V]], states: Sequence[V]) -> A:
            starts: list[Optional[V]] = [None] * len(windows)
            if self._needs_start and start is not None:
                # Step m starts from state m, which changes for m >= 1.
                for m in range(1, len(windows)):
                    window = windows[m]
                    starts[m] = start.tangent(
                        window.past_state(1),
                        window.time(),
                        states[m - 1],
                        window.parameters_version,
                    )
            return self._tested(windows, linearized(unknowns, states), starts)

        return change

    def trajectory_adjoint(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryAdjoint[A, V]]:
        """Supplied when :meth:`trajectory_tangent` is: each step's used
        blocks read the cotangent's test-function entries, weighted as
        :meth:`output` weights them; for a method that reads the step-start
        reaction, its adjoint carries step ``m``'s weighted entries to the
        state step ``m`` starts from (the committed one, for the first
        step, does not change)."""
        linearized = _LinearizedResiduals.of(self._reaction, trial)
        if linearized is None:
            return None
        windows = [trial.window(m) for m in range(trial.nsteps())]
        weights = self._weights(windows)
        ntests, bkd, start = self._test.count(), self._bkd, self._start
        state_space = self._reaction.state_space().component(0)

        def weighted(per_test: A, row: Sequence[float]) -> Optional[A]:
            total: Optional[A] = None
            for k, weight in enumerate(row):
                if weight != 0.0:
                    term = weight * per_test[:, k : k + 1]
                    total = term if total is None else total + term
            return total

        def adjoint(cotangent: A) -> tuple[list[BlockVector[V]], list[V]]:
            per_test = bkd.reshape(cotangent, (-1, ntests))
            residuals = []
            for blocks, _ in weights:
                parts: dict[int, A] = {}
                for r, block in enumerate(self._blocks):
                    part = weighted(per_test, [blocks[k][r] for k in range(ntests)])
                    if part is not None:
                        held = parts.get(block)
                        parts[block] = part if held is None else held + part
                residuals.append(_on_blocks(self._reaction, self._ports, parts))
            unknowns, states = linearized.transpose(residuals)
            if not self._needs_start or start is None:
                return unknowns, states
            for m in range(1, len(windows)):
                part = weighted(per_test, weights[m][1])
                if part is None:
                    continue
                window = windows[m]
                pulled = start.adjoint(
                    window.past_state(1),
                    window.time(),
                    self._ports.add_to(state_space.zeros(1), part),
                    window.parameters_version,
                )
                states[m - 1] = state_space.axpby(1.0, pulled, 1.0, states[m - 1])
            return unknowns, states

        return adjoint

    def __repr__(self) -> str:
        return f"{type(self).__name__}(test={self._test!r})"


@runtime_checkable
class StageValuesProtocol(Protocol[V]):
    """A Runge-Kutta stepper's stage values at a step's window, as
    :class:`StageQuadratureTrace` reads them; linear in the window's
    unknown and stored state."""

    def stage_values(self, window: StepWindow[V]) -> Sequence[V]:
        """Every stage value: the stored state for a trivial first stage,
        then the unknown's stage blocks."""
        ...

    def stage_values_transpose(
        self, window: StepWindow[V], cotangents: Sequence[V]
    ) -> tuple[BlockVector[V], V]:
        """The transpose of :meth:`stage_values`: for one cotangent per
        stage, the unknown's cotangent and the stored state's."""
        ...


class StageQuadratureTrace(Generic[V, A]):
    r"""The state on the port rows tested in time, :math:`\int \psi_k(s)\,
    u\, dt` over the window for each test function, by the Runge-Kutta
    method's own quadrature: :math:`\sum_m \Delta t_m \sum_j b_j\,
    \psi_k(s_{m,j})\, Y_{m,j}` on the port rows. The output of a
    port-loaded (Neumann-to-Dirichlet) windowed response, tested in time
    as :class:`StageQuadratureFlux` tests a flux, laid out as it is.

    Satisfies :class:`WindowPortOutputProtocol`, with its change for any
    number of steps: the stage values are a selection of the unknown's
    blocks and the stored state.

    Parameters
    ----------
    stepper : StageValuesProtocol
        The trials' stepper.
    tableau : ButcherTableau
        Its tableau.
    ports : PortRowsProtocol
        The port rows of the state space, in port order.
    test : TemporalTestFunctionsProtocol
        The test functions.
    space : VectorSpaceProtocol
        The state space.
    bkd : Backend
        The backend of the port blocks.
    returns : PortQuantity, default Trace()
        What the output is.

    Raises
    ------
    TypeError
        If ``stepper`` or ``test`` does not satisfy its protocol.
    """

    def __init__(
        self,
        stepper: StageValuesProtocol[V],
        tableau: ButcherTableau,
        ports: PortRowsProtocol[V, A],
        test: TemporalTestFunctionsProtocol,
        space: VectorSpaceProtocol[V],
        bkd: Backend[A],
        returns: PortQuantity = Trace(),
    ) -> None:
        for name, value, protocol in (
            ("stepper", stepper, StageValuesProtocol),
            ("test", test, TemporalTestFunctionsProtocol),
        ):
            if not isinstance(value, protocol):
                raise TypeError(
                    f"{name} must satisfy {protocol.__name__}, got "
                    f"{type(value).__name__}"
                )
        self._stepper, self._tableau, self._ports = stepper, tableau, ports
        self._test, self._space, self._bkd = test, space, bkd
        self._returns = returns

    def returns(self) -> PortQuantity:
        return self._returns

    def reads_port_data(self) -> bool:
        """The stage values do not read the port data beside the trial."""
        return False

    def nfunctions(self) -> int:
        """The number of test functions."""
        return self._test.count()

    def _tested(
        self, windows: Sequence[StepWindow[V]], stages: Sequence[Sequence[V]]
    ) -> A:
        bkd, take, tableau = self._bkd, self._ports.take, self._tableau
        origin = windows[0].time()
        last = windows[-1]
        length = last.time() + last.deltat() - origin
        tested: list[Optional[A]] = [None] * self._test.count()
        for window, values in zip(windows, stages):
            deltat = window.deltat()
            for j, value in enumerate(values):
                weight = deltat * tableau.b[j]
                if weight == 0.0:
                    continue
                time = window.time() + tableau.c[j] * deltat
                trace = take(value)
                for k, psi in enumerate(self._test.values((time - origin) / length)):
                    term = weight * psi * trace
                    current = tested[k]
                    tested[k] = term if current is None else current + term
        parts = []
        for part in tested:
            if part is None:
                raise ValueError("the trial has no steps")
            parts.append(part)
        stacked = bkd.stack(parts, axis=1)
        nport, ntests, ncols = stacked.shape
        return bkd.reshape(stacked, (nport * ntests, ncols))

    def output(self, trial: InMemoryTrajectoryStore[V]) -> A:
        """Raises ValueError if ``trial`` has no steps."""
        windows = [trial.window(m) for m in range(trial.nsteps())]
        if not windows:
            raise ValueError("the trial has no steps")
        stepper = self._stepper
        return self._tested(windows, [stepper.stage_values(w) for w in windows])

    def trajectory_tangent(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryTangent[V, A]]:
        windows = [trial.window(m) for m in range(trial.nsteps())]
        if not windows:
            return None
        stepper, space = self._stepper, self._space

        def change(unknowns: Sequence[BlockVector[V]], states: Sequence[V]) -> A:
            stages = []
            for m, window in enumerate(windows):
                # Step m starts from state m: the committed one for m = 0.
                start = (
                    space.scale(0.0, unknowns[m].blocks[0]) if m == 0 else states[m - 1]
                )
                changed = replace(window, past_states=(start,), unknown=unknowns[m])
                stages.append(stepper.stage_values(changed))
            return self._tested(windows, stages)

        return change

    def trajectory_adjoint(
        self, trial: InMemoryTrajectoryStore[V]
    ) -> Optional[TrajectoryAdjoint[A, V]]:
        """Supplied for any trial with steps: each stage's cotangent is the
        cotangent's test-function entries weighted as :meth:`output`
        weights them, put on the port rows, and the stepper's
        ``stage_values_transpose`` carries them to the step's unknown and
        to the state it starts from (the committed one, for the first
        step, does not change)."""
        windows = [trial.window(m) for m in range(trial.nsteps())]
        if not windows:
            return None
        bkd, tableau, take_t = self._bkd, self._tableau, self._ports.add_to
        stepper, space, test = self._stepper, self._space, self._test
        origin = windows[0].time()
        length = windows[-1].time() + windows[-1].deltat() - origin
        ntests = test.count()

        def adjoint(cotangent: A) -> tuple[list[BlockVector[V]], list[V]]:
            per_test = bkd.reshape(cotangent, (-1, ntests))
            unknowns: list[BlockVector[V]] = []
            states = [space.zeros(1) for _ in windows]
            for m, window in enumerate(windows):
                deltat = window.deltat()
                stages = []
                for j in range(tableau.nstages()):
                    time = window.time() + tableau.c[j] * deltat
                    weight = deltat * tableau.b[j]
                    values = test.values((time - origin) / length)
                    total = bkd.zeros_like(per_test[:, 0:1])
                    for k, psi in enumerate(values):
                        total = total + (weight * psi) * per_test[:, k : k + 1]
                    stages.append(take_t(space.zeros(1), total))
                unknown, start = stepper.stage_values_transpose(window, stages)
                unknowns.append(unknown)
                if m >= 1:
                    states[m - 1] = space.axpby(1.0, start, 1.0, states[m - 1])
            return unknowns, states

        return adjoint

    def __repr__(self) -> str:
        return f"{type(self).__name__}(test={self._test!r})"


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
    changes, plus, for an output reading the port data itself, its change
    with them at the fixed trial, by differences again (it reads them
    affinely). Leaves the input holding the trial's own port data.
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
        port_output: WindowPortOutputProtocol[V, A],
        bkd: Backend[A],
    ) -> None:
        self._stepper, self._input = stepper, port_input
        self._window, self._start, self._port_data = window, start, port_data
        self._trial = trial
        self._steps = [trial.window(m) for m in range(trial.nsteps())]
        self._change, self._output, self._bkd = change, port_output, bkd

    def _data_changes(self, directions: A) -> tuple[list[BlockVector[V]], Optional[A]]:
        """Per step, :math:`-\\Delta R_m` for every column of
        ``directions``, concatenated; and the output's direct change with
        the port data, None when it reads none."""
        stepper, port_input = self._stepper, self._input
        space = stepper.residual_space()
        reads = self._output.reads_port_data()
        port_input.set_window(self._window, self._start, self._port_data)
        bases = [stepper.step_residual(step) for step in self._steps]
        base_output = self._output.output(self._trial) if reads else None
        columns: list[list[BlockVector[V]]] = [[] for _ in self._steps]
        outputs: list[A] = []
        for j in range(int(directions.shape[1])):
            shifted = self._port_data + directions[:, j : j + 1]
            port_input.set_window(self._window, self._start, shifted)
            for m, step in enumerate(self._steps):
                # -(R_m(lambda + d_j) - R_m(lambda)).
                columns[m].append(
                    space.axpby(1.0, bases[m], -1.0, stepper.step_residual(step))
                )
            if base_output is not None:
                outputs.append(self._output.output(self._trial) - base_output)
        port_input.set_window(self._window, self._start, self._port_data)
        direct = self._bkd.hstack(outputs) if outputs else None
        return [space.concatenate(per_step) for per_step in columns], direct

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
        changes, direct = self._data_changes(directions)
        for m, (step, rhs) in enumerate(zip(self._steps, changes)):
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
        through = self._change(unknowns, states)
        return through if direct is None else through + direct

    def data_columns(self) -> tuple[list[BlockVector[V]], Optional[A]]:
        r"""Per step, :math:`-\partial R_m/\partial \lambda`, one column per
        port datum; and the output's direct derivative in them, None when
        it reads none. Exact by differences of unit data, since both are
        affine in them; residual evaluations only."""
        return self._data_changes(self._bkd.eye(self._input.nport()))

    def steps(self) -> list[StepWindow[V]]:
        """The trial's step windows."""
        return self._steps

    def stepper(self) -> StepperProtocol[V, A]:
        return self._stepper

    def nport(self) -> int:
        """The number of port data."""
        return self._input.nport()


class _TraceAdjoint(Generic[V, A]):
    r""":math:`(\partial r/\partial \lambda)^\top w`, the transpose of a
    :class:`_TraceTangent`, by a backward sweep over the trial's steps.

    The output's cotangents of the step unknowns and stored states come
    from the output's :data:`TrajectoryAdjoint`. From the last step back,
    step :math:`m`'s adjoint solves :math:`(\partial R_m/\partial
    u_m)^\top \mu_m = \bar u_m + E^\top \bar y_{m+1}`, adds
    :math:`(-\partial R_m/\partial \lambda)^\top \mu_m` to the result and
    :math:`-(\partial R_m/\partial y_{m+1-l})^\top \mu_m` to the
    cotangent of each changing state it read; the output's direct
    derivative in the port data adds its transpose. The data columns are
    formed on the first call, the transposed step handles once per step.
    """

    def __init__(
        self,
        tangent: _TraceTangent[V, A],
        adjoint: TrajectoryAdjoint[A, V],
        nreturn: int,
        bkd: Backend[A],
    ) -> None:
        self._tangent, self._adjoint = tangent, adjoint
        self._nreturn, self._bkd = nreturn, bkd
        self._columns: Optional[tuple[list[BlockVector[V]], Optional[A]]] = None
        self._handles: dict[int, SolverHandleProtocol[BlockVector[V]]] = {}

    def _handle(self, m: int) -> SolverHandleProtocol[BlockVector[V]]:
        handle = self._handles.get(m)
        if handle is None:
            stepper = self._tangent.stepper()
            transposed = stepper.diag_handle(self._tangent.steps()[m]).transpose()
            if transposed is None:
                raise TypeError(
                    f"{type(stepper).__name__}'s diag_handle cannot solve "
                    "transposed, so the trace has no adjoint"
                )
            handle = self._handles[m] = transposed
        return handle

    def _column(self, cotangent: A) -> A:
        if self._columns is None:
            self._columns = self._tangent.data_columns()
        columns, direct = self._columns
        stepper = self._tangent.stepper()
        unknown_space = stepper.unknown_space()
        residual_space = stepper.residual_space()
        state_space = stepper.state_space().component(0)
        off_diag = stepper.derivatives().off_diag
        unknowns, states = self._adjoint(cotangent)
        states = list(states)
        total = (
            self._bkd.zeros((self._tangent.nport(), 1))
            if direct is None
            else direct.T @ cotangent
        )
        steps = self._tangent.steps()
        for m in reversed(range(len(steps))):
            step = steps[m]
            rhs = unknown_space.axpby(
                1.0,
                stepper.readout().apply_transpose(BlockVector((states[m],))),
                1.0,
                unknowns[m],
            )
            mu, info = self._handle(m).solve(rhs)
            if not info.converged:
                raise ValueError(f"the adjoint solve of step {m} did not converge")
            total = total + residual_space.dot(columns[m], mu)
            for lag in range(1, len(step.past_states) + 1):
                # The committed first state does not change.
                index = m + 1 - lag
                if index < 1:
                    continue
                if off_diag is None:
                    raise ValueError(
                        "a trial of several steps needs the stepper's past-state "
                        "derivatives"
                    )
                pulled = off_diag(step, lag).apply_transpose(mu).blocks[0]
                states[index - 1] = state_space.axpby(
                    -1.0, pulled, 1.0, states[index - 1]
                )
        return total

    def __call__(self, cotangents: A) -> A:
        if cotangents.ndim != 2 or cotangents.shape[0] != self._nreturn:
            raise ValueError(
                f"cotangents must have shape ({self._nreturn}, ncols), got "
                f"{tuple(cotangents.shape)}"
            )
        return self._bkd.hstack(
            [
                self._column(cotangents[:, j : j + 1])
                for j in range(int(cotangents.shape[1]))
            ]
        )


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
        port_data: A,
        transpose: Optional[TraceVJP[A]] = None,
    ) -> None:
        self._owner, self._window, self._trial = owner, window, trial
        self._output, self._bkd, self._port_data = output, bkd, port_data
        self._derivs: ResponseDerivatives[A]
        if tangent is None:
            self._derivs = ResponseDerivatives.none()
        elif transpose is None:
            self._derivs = ResponseDerivatives.tangent(tangent)
        else:
            self._derivs = ResponseDerivatives.adjoint(tangent, transpose)

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

    def port_data(self) -> A:
        """The port data the window was stepped with. Not to be
        modified."""
        return self._port_data


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
    past-state derivatives). Its transpose (``vjp_trace``) is supplied
    beside it whenever ``port_output`` gives
    :meth:`~WindowPortOutputProtocol.trajectory_adjoint`, by a backward
    sweep over the same steps; calling it needs the stepper's step
    handles to solve transposed.

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

    def _substeps(self, window: TimeWindow) -> GivenSizes:
        """Equal steps over ``window`` whose times add up to its end
        exactly, so the sink's last time is the next window's start. Equal
        sizes added in floating point can miss it (three steps of 0.01/3
        from 0.02 end at 0.030000000000000002)."""
        n, start, length = self._nsubsteps, window.start, window.end - window.start
        times = [start + length * k / n for k in range(n)] + [window.end]
        return GivenSizes.from_times(times)

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
        trial: InMemoryTrajectoryStore[V] = InMemoryTrajectoryStore()
        self._integrator.solve(self._state, self._substeps(window), trial)
        change = self._output.trajectory_tangent(trial)
        tangent: Optional[_TraceTangent[V, A]] = (
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
                self._output,
                self._bkd,
            )
        )
        output = self._output.output(trial)
        adjoint = None if tangent is None else self._output.trajectory_adjoint(trial)
        transpose: Optional[TraceVJP[A]] = (
            None
            if tangent is None or adjoint is None
            else _TraceAdjoint(tangent, adjoint, int(output.shape[0]), self._bkd)
        )
        return WindowedPoint(
            self._owner,
            window,
            trial,
            output,
            self._bkd,
            tangent,
            port_data,
            transpose,
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
        # An input whose start is not read from the state keeps the data.
        self._input.accept(point.port_data())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nport={self.nport()}, time={self._time})"
