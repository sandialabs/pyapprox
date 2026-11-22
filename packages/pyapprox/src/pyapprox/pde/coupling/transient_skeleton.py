r"""A transient skeleton solve, one window at a time.

Each window :math:`[t_0, t_1]` is a steady skeleton problem: find the
skeleton state :math:`\lambda(t_1)` with
:math:`S(\lambda) = \sum_i E_i^\top r_i(E_i \lambda) + C\lambda = 0`,
every response :math:`r_i` stepping its subdomain over the window
(:mod:`~pyapprox.pde.coupling.transient`). :class:`TransientSkeletonSolver`
builds the window's :class:`~pyapprox.pde.coupling.skeleton.
SkeletonResidual`, solves it from the previous window's state, and has
every response accept its point at the solution: the subdomains commit
their states and write their steps into their own sinks.

The skeleton states go into an injected interface sink as a trajectory
of skeleton vectors: the initial state, then one step per window, its
unknown the window's solution. Any
:class:`~pyapprox.ode.trajectory_store.TrajectorySinkProtocol` serves,
so the interface trajectory is kept, accumulated or written by the same
stores as a subdomain's.
"""

from typing import Generic, Optional, Sequence, TypeVar

from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.implicitfunction.steady_system import SteadySystem
from pyapprox.ode.step_span import StepSpan
from pyapprox.ode.stepper_derivatives import StepWindow
from pyapprox.ode.trajectory_store import TrajectorySinkProtocol
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.skeleton import (
    CoupledSubdomain,
    CouplingProtocol,
    KeepCurrentPoints,
    SchurBuilderProtocol,
    SerialExecutor,
    SkeletonResidual,
    SubdomainExecutorProtocol,
)
from pyapprox.pde.coupling.transient import WindowedResponseProtocol
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    BlockVector,
    FormatProtocol,
    InnerProductSpaceProtocol,
)
from pyapprox.util.linalg.operators.solvers import SolveFailedError
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
Array = TypeVar("Array", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


class TransientSkeletonSolver(Generic[V, A, Array, F]):
    r"""March a coupled transient problem through windows, one skeleton
    solve per window (see the module docstring).

    Parameters
    ----------
    space
        The skeleton's space; states are its single-column vectors.
    subdomains
        Each subdomain's windowed response and port transfer.
    schur
        Forms each window's Jacobian, as for
        :class:`~pyapprox.pde.coupling.skeleton.SkeletonResidual`.
    setups
        Sets up each window's Jacobian; cleared before every window, since
        each window's residual is a new one.
    nonlinear
        Solves each window's skeleton problem.
    executor
        Runs the per-subdomain work. Default:
        :class:`~pyapprox.pde.coupling.skeleton.SerialExecutor`.
    coupling
        The fixed coupling :math:`C`, or ``None``, as for
        :class:`~pyapprox.pde.coupling.skeleton.SkeletonResidual`.

    Raises
    ------
    TypeError
        If a response does not satisfy
        :class:`~pyapprox.pde.coupling.transient.WindowedResponseProtocol`.
    """

    def __init__(
        self,
        space: InnerProductSpaceProtocol[V, A],
        subdomains: Sequence[CoupledSubdomain[V, Array]],
        schur: SchurBuilderProtocol[V, Array, F],
        setups: SetupStrategyProtocol[SteadyRequest[V, F], V, F],
        nonlinear: NonlinearSolverProtocol[V, A],
        executor: Optional[SubdomainExecutorProtocol] = None,
        coupling: Optional[CouplingProtocol[V]] = None,
    ) -> None:
        responses: list[WindowedResponseProtocol[Array]] = []
        for i, sd in enumerate(subdomains):
            if not isinstance(sd.response, WindowedResponseProtocol):
                raise TypeError(
                    f"subdomain {i}'s response must satisfy WindowedResponseProtocol"
                )
            responses.append(sd.response)
        self._space, self._subdomains = space, tuple(subdomains)
        self._responses = tuple(responses)
        self._schur, self._setups, self._nonlinear = schur, setups, nonlinear
        self._executor: SubdomainExecutorProtocol = (
            SerialExecutor() if executor is None else executor
        )
        self._coupling = coupling

    def solve(
        self,
        lam0: V,
        windows: Sequence[TimeWindow],
        interface_sink: TrajectorySinkProtocol[V],
    ) -> V:
        r"""Start every response, then solve and accept each window in
        turn, writing the skeleton states into ``interface_sink``.

        Parameters
        ----------
        lam0
            The skeleton state at the first window's start: the trace of
            the subdomains' initial states, which the responses hold;
            the first window's initial guess.
        windows
            Nonempty and contiguous (each starts where the last ended),
            the first starting at every response's time.
        interface_sink
            Empty; receives :math:`\lambda` at the start and at each
            window's end.

        Returns
        -------
        V
            The skeleton state at the last window's end.

        Raises
        ------
        ValueError
            If ``windows`` is empty, not contiguous, or does not start at
            every response's time, a window is steady, or
            ``interface_sink`` is not empty.
        SolveFailedError
            If a window's skeleton solve does not converge; the windows
            before it stay accepted.
        """
        self._check(windows, interface_sink)
        for response in self._responses:
            response.start()
        interface_sink.append(windows[0].start, lam0)
        lam = lam0
        for window in windows:
            lam = self._window(lam, window, interface_sink)
        return lam

    def _check(
        self, windows: Sequence[TimeWindow], sink: TrajectorySinkProtocol[V]
    ) -> None:
        if not windows:
            raise ValueError("there must be at least one window")
        if sink.ntimes() != 0:
            raise ValueError("the interface sink must be empty")
        for before, after in zip(windows[:-1], windows[1:]):
            if after.start != before.end:
                raise ValueError(
                    f"windows must be contiguous: one ends at {before.end}, the "
                    f"next starts at {after.start}"
                )
        for window in windows:
            if window.is_steady():
                raise ValueError("a transient window must be nonempty")
        times = {response.time() for response in self._responses}
        if times != {windows[0].start}:
            raise ValueError(
                f"the first window starts at {windows[0].start}, the responses "
                f"at {sorted(times)}"
            )

    def _window(self, lam: V, window: TimeWindow, sink: TrajectorySinkProtocol[V]) -> V:
        points: KeepCurrentPoints[V, Array] = KeepCurrentPoints()
        residual = SkeletonResidual(
            self._space,
            self._subdomains,
            self._schur,
            points,
            window,
            self._executor,
            self._coupling,
        )
        self._setups.clear()
        solved = SteadySystem(residual, self._space, self._setups).solve(
            self._nonlinear, lam
        )
        if not solved.converged():
            raise SolveFailedError(
                f"the skeleton solve of window [{window.start}, {window.end}] did "
                f"not converge ({solved.info().message})"
            )
        new = solved.solution()
        # Kept from the solve's last evaluation when it was at the solution.
        accepted = points.points(new, residual)

        def accept(i: int) -> None:
            self._responses[i].accept(accepted[i])

        self._executor.map(accept, range(len(self._responses)))
        span = StepSpan(window.start, (window.end - window.start,))
        sink.append_step(StepWindow(span, (lam,), BlockVector((new,))), new)
        return new

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nsubdomains={len(self._subdomains)})"
