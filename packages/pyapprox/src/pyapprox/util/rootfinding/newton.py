from dataclasses import dataclass
from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class NewtonSolverResidualProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    # By defining arguments as positional-only in the Protocol definition,
    # you indicate that the names do not matter in the implementation.
    # Use the / character in the function signature:
    def __call__(self, iterate: Array, /) -> Array: ...

    def linsolve(self, state: Array, prev_residual: Array) -> Array: ...

    def is_one_step_solvable(self) -> bool: ...


class NewtonSolverOptions:
    """
    Encapsulates options for configuring the Newton solver.

    Useful for passing options through a sequence of function calls.
    PyApprox tries to avoid this but it is not always possible

    Parameters
    ----------
    maxiters : int, optional
        Maximum number of iterations for the Newton solver. Default is 10.
    verbosity : int, optional
        Verbosity level for the solver. Default is 0.
    step_size : float, optional
        Step size for the Newton solver. Default is 1.
    atol : float, optional
        Absolute tolerance for convergence. Default is 1e-7.
    rtol : float, optional
        Relative tolerance for convergence. Default is 1e-7.
    linesearch_maxiters : int, optional
        Maximum number of step halvings in the line search. Default is 5.
    line_search : bool, optional
        Whether to backtrack along each Newton step. Default is False.
    """

    def __init__(
        self,
        maxiters: int = 10,
        verbosity: int = 0,
        step_size: float = 1.0,
        atol: float = 1e-7,
        rtol: float = 1e-7,
        linesearch_maxiters: int = 5,
        line_search: bool = False,
    ) -> None:
        self.maxiters = maxiters
        self.verbosity = verbosity
        self.step_size = step_size
        self.atol = atol
        self.rtol = rtol
        self.linesearch_maxiters = linesearch_maxiters
        self.line_search = line_search

    def __repr__(self) -> str:
        """
        Return a string representation of the Newton solver options.

        Returns
        -------
        str
            String representation of the options.
        """
        return (
            f"NewtonSolverOptions("
            f"maxiters={self.maxiters}, "
            f"verbosity={self.verbosity}, "
            f"step_size={self.step_size}, "
            f"atol={self.atol}, "
            f"rtol={self.rtol}, "
            f"linesearch_maxiters={self.linesearch_maxiters}, "
            f"line_search={self.line_search})"
        )


@dataclass(frozen=True)
class NewtonResult(Generic[Array]):
    """Outcome of ``NewtonSolver.solve_with_info``.

    ``residual_norm`` is ``None`` after the one-step path, which stops
    without evaluating the residual again.
    """

    solution: Array
    converged: bool
    iterations: int
    residual_norm: Optional[Array]
    message: str


class NewtonSolver(Generic[Array]):
    """Newton's method for ``R(x) = 0``.

    Converges when ``||R(x_k)|| <= atol + rtol * ||R(x_0)||``. A residual
    that reports ``is_one_step_solvable()`` (linear in the iterate) is
    solved by one exact step with no tolerance test: the residual left
    is round-off, which grows with the conditioning. An optional
    backtracking line search (``line_search=True``) shortens steps that
    do not reduce the residual norm.
    """

    def __init__(self, residual: NewtonSolverResidualProtocol[Array]) -> None:
        if not isinstance(residual, NewtonSolverResidualProtocol):
            raise ValueError("residual must satisfy NewtonSolverResidualProtocol")
        self._residual = residual
        self._bkd = residual.bkd()
        self.set_options()

    def set_options(
        self,
        maxiters: int = 10,
        verbosity: int = 0,
        step_size: float = 1,
        atol: float = 1e-7,
        rtol: float = 1e-7,
        linesearch_maxiters: int = 5,
        line_search: bool = False,
    ) -> None:
        self._maxiters = maxiters
        self._verbosity = verbosity
        self._step_size = step_size
        self._atol = atol
        self._rtol = rtol
        self._linesearch_maxiters = linesearch_maxiters
        self._line_search = line_search

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def _update_iterate(self, prev_iterate: Array, delta: Array) -> Array:
        return prev_iterate - self._step_size * delta

    def _line_search_step(
        self, iterate: Array, delta: Array, residual_norm: Array
    ) -> Array:
        """Backtrack along the Newton step until the residual norm
        decreases sufficiently (Armijo, ``c = 1e-4``), halving at most
        ``linesearch_maxiters`` times; the last trial is kept."""
        alpha = 1.0
        for _ in range(self._linesearch_maxiters):
            trial = iterate - alpha * self._step_size * delta
            if self._bkd.norm(self._residual(trial)) < (
                1.0 - 1e-4 * alpha
            ) * residual_norm:
                return trial
            alpha *= 0.5
        return iterate - alpha * self._step_size * delta

    def solve_with_info(self, init_iterate: Array) -> NewtonResult[Array]:
        """Solve ``R(x) = 0`` and report how, without raising on failure.

        Parameters
        ----------
        init_iterate : Array
            Initial iterate. Shape: (n,)

        Returns
        -------
        NewtonResult
            The final iterate (``solution``), whether it converged, the
            number of steps and the final residual norm.
        """
        if init_iterate.ndim != 1:
            raise ValueError("init_iterate must be 1D array")
        iterate = init_iterate
        residual = self._residual(iterate)

        # Fast path: one Newton step is exact for linear residuals
        if self._residual.is_one_step_solvable():
            delta = self._residual.linsolve(iterate, residual)
            return NewtonResult(
                solution=self._update_iterate(iterate, delta),
                converged=True,
                iterations=1,
                residual_norm=None,
                message="One step: the residual is linear",
            )

        residual_norms = [self._bkd.norm(residual)]
        tolerance = self._atol + self._rtol * residual_norms[0]
        it = 0
        if self._verbosity > 1:
            print("Iter", it, "rnorm", residual_norms[0])
        while True:
            delta = self._residual.linsolve(iterate, residual)
            if self._line_search:
                iterate = self._line_search_step(
                    iterate, delta, residual_norms[-1]
                )
            else:
                iterate = self._update_iterate(iterate, delta)
            residual = self._residual(iterate)
            residual_norm = self._bkd.norm(residual)
            residual_norms.append(residual_norm)

            it += 1
            if self._verbosity > 1:
                print("Iter", it, "rnorm", residual_norm)
            if not self._bkd.isfinite(residual_norm):
                return NewtonResult(
                    iterate, False, it, residual_norm,
                    "Residual is no longer finite",
                )
            if residual_norm <= tolerance:
                message = f"Tolerance {tolerance} reached"
                if self._verbosity > 0:
                    print(message)
                return NewtonResult(iterate, True, it, residual_norm, message)
            if it >= self._maxiters:
                return NewtonResult(
                    iterate,
                    False,
                    it,
                    residual_norm,
                    f"Max iterations {self._maxiters} reached.\n"
                    f"Rel residual norm is {residual_norm} "
                    f"Needs to be {tolerance}",
                )

    def solve(self, init_iterate: Array) -> Array:
        """Solve ``R(x) = 0`` and return the iterate.

        Raises
        ------
        RuntimeError
            If the residual stops being finite, or the tolerance is not
            reached in ``maxiters`` steps.
        """
        result = self.solve_with_info(init_iterate)
        if not result.converged:
            raise RuntimeError(result.message)
        return result.solution

    def __repr__(self) -> str:
        return (
            "{0}(maxiters={1}, verbosity={2}, step_size={3}, atol={4}, rol={5})"
        ).format(
            self.__class__.__name__,
            self._maxiters,
            self._verbosity,
            self._step_size,
            self._atol,
            self._rtol,
        )

    def residual(self) -> NewtonSolverResidualProtocol[Array]:
        return self._residual
