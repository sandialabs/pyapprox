r"""Check that a setup strategy never returns a handle for another operator.

A strategy may keep any setup and drop it at any time, so what it keeps
cannot be checked, only whether every handle it returns solves with the
operator the request describes. :func:`check_strategy` drives a sequence
of requests through the strategy in three passes: from a cleared
strategy; again, where a strategy that keeps answers from what it kept;
and after :meth:`~pyapprox.implicitfunction.setups.SetupStrategyProtocol.
clear`. Each returned handle's solve, and its transposed solve where the
handle has one, is compared with a fresh setup of the same request by a
reference solver.

Order the requests so that keys repeat with different states (a
recomputed state at one position) and differ in one part at a time: a
strategy keying on less than the request's key then returns a wrong
handle, and the check reports it. Meant for a strategy's own tests.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple, TypeVar

from pyapprox.implicitfunction.setups import (
    OperatorBuilderProtocol,
    SetupRequestProtocol,
    SetupStrategyProtocol,
)
from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    InnerProductSpaceProtocol,
    LinearSolverProtocol,
    SolverHandleProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)
Q = TypeVar("Q", bound=SetupRequestProtocol)


@dataclass(frozen=True)
class StrategyMismatch:
    """A handle whose solve differs from a fresh setup's.

    Attributes
    ----------
    sweep : int
        The pass: 0 from a cleared strategy, 1 repeated, 2 after
        ``clear``.
    request : int
        The request's index in the sequence.
    transposed : bool
        Whether the transposed solve differs.
    error : float
        The relative difference of the solutions.
    """

    sweep: int
    request: int
    transposed: bool
    error: float


@dataclass(frozen=True)
class StrategyCheckReport:
    """The outcome of :func:`check_strategy`.

    Attributes
    ----------
    nchecked : int
        Solves compared (forward and transposed count once each).
    mismatches : tuple of StrategyMismatch
        Every solve that differs from a fresh setup's.
    """

    nchecked: int
    mismatches: Tuple[StrategyMismatch, ...]

    def passed(self) -> bool:
        """Whether every handle solved as a fresh setup does."""
        return not self.mismatches


def _relative_error(
    space: InnerProductSpaceProtocol[V, A], got: V, expected: V
) -> float:
    r""":math:`\|g - e\| / \|e\|` of single columns, as a float: a
    check's verdict, never fed into a computation."""
    difference = space.norm(space.axpby(1.0, got, -1.0, expected))
    return float((difference / space.norm(expected)).item())


def check_strategy(
    strategy: SetupStrategyProtocol[Q, V, F],
    reference: LinearSolverProtocol[V, F],
    requests: Sequence[Tuple[Q, OperatorBuilderProtocol[Q, V, F]]],
    space: InnerProductSpaceProtocol[V, A],
    rhs: V,
    rtol: float = 1e-12,
) -> StrategyCheckReport:
    """Drive ``requests`` through ``strategy`` in three passes and
    compare every handle with a fresh setup.

    Parameters
    ----------
    strategy : SetupStrategyProtocol
        The strategy under test; cleared before the first and third
        passes, and at the end.
    reference : LinearSolverProtocol
        Sets up each request afresh; the strategy's own solver is fine.
    requests : sequence of (request, builder)
        In the order a consumer would make them.
    space : InnerProductSpaceProtocol
        The space of ``rhs`` and of the solutions.
    rhs : V
        A single column to solve with.
    rtol : float, default 1e-12
        Allowed relative difference of a solve from the fresh one.

    Returns
    -------
    StrategyCheckReport
    """
    nchecked = 0
    mismatches: List[StrategyMismatch] = []

    def compare(
        sweep: int,
        index: int,
        transposed: bool,
        handle: SolverHandleProtocol[V],
        fresh: SolverHandleProtocol[V],
    ) -> None:
        nonlocal nchecked
        nchecked += 1
        error = _relative_error(space, handle.solve(rhs)[0], fresh.solve(rhs)[0])
        if not error <= rtol:
            mismatches.append(StrategyMismatch(sweep, index, transposed, error))

    for sweep in range(3):
        if sweep != 1:
            strategy.clear()
        for index, (request, builder) in enumerate(requests):
            handle = strategy.handle(request, builder)
            fresh = reference.setup(builder.build(request))
            compare(sweep, index, False, handle, fresh)
            handle_t, fresh_t = handle.transpose(), fresh.transpose()
            if handle_t is not None and fresh_t is not None:
                compare(sweep, index, True, handle_t, fresh_t)
    strategy.clear()
    return StrategyCheckReport(nchecked, tuple(mismatches))
