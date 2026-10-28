r"""Nonlinear solves on any linear-algebra stack.

A *nonlinear system* :math:`R(x) = 0` states its residual, a solver
handle for its Jacobian at a point, and the space its vectors belong to;
a *nonlinear solver* is configured once and solves any such system, so
one solver serves many systems (each stage of a time step, say) and
nothing is bound at construction. Everything goes through the space's
operations and the handles, so the vectors may belong to any stack.

:class:`Newton` is Newton's method with an optional backtracking line
search. It converges when
:math:`\|R(x_k)\| \le \mathrm{atol} + \mathrm{rtol}\,\|R(x_0)\|`; a
system declaring itself linear is solved by one exact step with no
tolerance test.
"""

from dataclasses import dataclass
from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.util.backends.protocols import ArrayProtocol
from pyapprox.util.linalg.operators import (
    InnerProductSpaceProtocol,
    SolverHandleProtocol,
)

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
"""The reduction type of the system's space: small dense arrays."""
A_co = TypeVar("A_co", bound=ArrayProtocol, covariant=True)


def _holds(comparison: A) -> bool:
    """A comparison of single-column norms, shape ``(1, 1)``, as a bool.

    The one place a reduction leaves the array world: it decides whether
    to stop or to backtrack, and no value derived from it flows into a
    computation, so reading it with ``item()`` detaches nothing that
    matters.
    """
    return bool(comparison.item())


@runtime_checkable
class NonlinearSystemProtocol(Protocol[V, A_co]):
    """A system :math:`R(x) = 0` of vectors of one space."""

    def space(self) -> InnerProductSpaceProtocol[V, A_co]:
        """The space of :math:`x` and of :math:`R(x)`."""
        ...

    def residual(self, x: V) -> V:
        """:math:`R(x)`, for a single-column ``x``."""
        ...

    def jacobian_handle(self, x: V) -> SolverHandleProtocol[V]:
        r"""A solver handle for :math:`\partial R/\partial x` at ``x``."""
        ...

    def is_linear(self) -> bool:
        """Whether :math:`R` is affine in :math:`x`, so one exact Newton
        step solves it."""
        ...


@dataclass(frozen=True)
class NonlinearSolveInfo(Generic[A]):
    """The outcome of a nonlinear solve.

    Attributes
    ----------
    converged : bool
        Whether the solution is valid.
    iterations : int
        Newton steps taken.
    residual_norm : A or None
        The final residual norm, shape ``(1, 1)``; ``None`` after the
        one-step path for a linear system, which does not evaluate the
        residual again.
    message : str
        How the solve ended.
    """

    converged: bool
    iterations: int
    residual_norm: Optional[A]
    message: str


@runtime_checkable
class NonlinearSolverProtocol(Protocol[V, A]):
    """A configured solver for any nonlinear system."""

    def solve(
        self, system: NonlinearSystemProtocol[V, A], initial_guess: V
    ) -> tuple[V, NonlinearSolveInfo[A]]:
        """Solve ``system`` from ``initial_guess``. Never raises on
        failure: check ``info.converged``."""
        ...


class Newton(Generic[V, A]):
    r"""Newton's method on any stack.

    Each step solves :math:`J(x_k)\,\delta = R(x_k)` with the system's
    Jacobian handle and sets :math:`x_{k+1} = x_k - s\,\delta`.

    Parameters
    ----------
    atol, rtol : float, default 1e-7
        Converged when :math:`\|R(x_k)\| \le \mathrm{atol} +
        \mathrm{rtol}\,\|R(x_0)\|`.
    maxiters : int, default 10
        Steps allowed before reporting failure.
    step_size : float, default 1.0
        The scaling :math:`s` of each step.
    line_search : bool, default False
        Backtrack along each step until the residual norm decreases
        sufficiently (Armijo, :math:`c = 10^{-4}`), halving at most
        ``linesearch_maxiters`` times; the last trial is kept.
    linesearch_maxiters : int, default 5
        Halvings allowed per step.
    """

    def __init__(
        self,
        atol: float = 1e-7,
        rtol: float = 1e-7,
        maxiters: int = 10,
        step_size: float = 1.0,
        line_search: bool = False,
        linesearch_maxiters: int = 5,
    ) -> None:
        if maxiters < 1:
            raise ValueError(f"maxiters must be at least 1, got {maxiters}")
        self._atol, self._rtol = atol, rtol
        self._maxiters = maxiters
        self._step_size = step_size
        self._line_search = line_search
        self._linesearch_maxiters = linesearch_maxiters

    def solve(
        self, system: NonlinearSystemProtocol[V, A], initial_guess: V
    ) -> tuple[V, NonlinearSolveInfo[A]]:
        """Solve ``system`` from ``initial_guess``.

        Parameters
        ----------
        system : NonlinearSystemProtocol
            The system.
        initial_guess : V
            A single-column vector of the system's space.

        Returns
        -------
        tuple
            The last iterate and how the solve ended. Never raises on
            failure: a failed linear solve, a residual that stops being
            finite, or ``maxiters`` steps without convergence are reported
            with ``converged=False``.
        """
        space = system.space()
        iterate = space.copy(initial_guess)
        residual = system.residual(iterate)
        if system.is_linear():
            delta, info = system.jacobian_handle(iterate).solve(residual)
            if not info.converged:
                return iterate, NonlinearSolveInfo(
                    False, 0, None, f"linear solve failed: {info.message}"
                )
            iterate = space.axpby(-self._step_size, delta, 1.0, iterate, out=iterate)
            return iterate, NonlinearSolveInfo(
                True, 1, None, "one step: the system is linear"
            )
        norm = space.norm(residual)
        tolerance = self._atol + self._rtol * norm
        for iteration in range(1, self._maxiters + 1):
            delta, info = system.jacobian_handle(iterate).solve(residual)
            if not info.converged:
                return iterate, NonlinearSolveInfo(
                    False, iteration - 1, norm, f"linear solve failed: {info.message}"
                )
            iterate = self._step(system, iterate, delta, norm)
            residual = system.residual(iterate)
            if not space.all_finite(residual):
                return iterate, NonlinearSolveInfo(
                    False, iteration, None, "the residual is no longer finite"
                )
            norm = space.norm(residual)
            if _holds(norm <= tolerance):
                return iterate, NonlinearSolveInfo(
                    True, iteration, norm, f"tolerance {tolerance} reached"
                )
        return iterate, NonlinearSolveInfo(
            False,
            self._maxiters,
            norm,
            f"{self._maxiters} steps reached; the residual norm {norm} "
            f"needs to be {tolerance}",
        )

    def _step(
        self,
        system: NonlinearSystemProtocol[V, A],
        iterate: V,
        delta: V,
        norm: A,
    ) -> V:
        """The next iterate: the full (scaled) step, or the line search's."""
        space = system.space()
        if not self._line_search:
            return space.axpby(-self._step_size, delta, 1.0, iterate)
        alpha = 1.0
        for _ in range(self._linesearch_maxiters):
            trial = space.axpby(-alpha * self._step_size, delta, 1.0, iterate)
            if _holds(space.norm(system.residual(trial)) < (1.0 - 1e-4 * alpha) * norm):
                return trial
            alpha *= 0.5
        return space.axpby(-alpha * self._step_size, delta, 1.0, iterate)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(atol={self._atol}, rtol={self._rtol}, "
            f"maxiters={self._maxiters}, line_search={self._line_search})"
        )
