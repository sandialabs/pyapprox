from typing import Protocol, Generic, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class NewtonSolverResidualProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    # By defining arguments as positional-only in the Protocol definition,
    # you indicate that the names do not matter in the implementation.
    # Use the / character in the function signature:
    def __call__(self, iterate: Array, /) -> Array: ...

    def linsolve(self, state: Array, prev_residual: Array) -> Array: ...


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
        Maximum number of iterations for line search. Default is 5.
    """

    def __init__(
        self,
        maxiters: int = 10,
        verbosity: int = 0,
        step_size: float = 1.0,
        atol: float = 1e-7,
        rtol: float = 1e-7,
        linesearch_maxiters: int = 5,
    ) -> None:
        self.maxiters = maxiters
        self.verbosity = verbosity
        self.step_size = step_size
        self.atol = atol
        self.rtol = rtol
        self.linesearch_maxiters = linesearch_maxiters

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
            f"linesearch_maxiters={self.linesearch_maxiters})"
        )


class NewtonSolver(Generic[Array]):
    def __init__(self, residual: NewtonSolverResidualProtocol[Array]) -> None:
        if not isinstance(residual, NewtonSolverResidualProtocol):
            raise ValueError(
                "residual must satisfy NewtonSolverResidualProtocol"
            )
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
        linesearch_maxiters: float = 5,
    ) -> None:
        self._maxiters = maxiters
        self._verbosity = verbosity
        self._step_size = step_size
        self._atol = atol
        self._rtol = rtol
        self._linesearch_maxiters = linesearch_maxiters

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def _update_iterate(self, prev_iterate: Array, delta: Array) -> Array:
        return prev_iterate - self._step_size * delta

    def solve(self, init_iterate: Array) -> Array:
        if init_iterate.ndim != 1:
            raise ValueError("init_iterate must be 1D array")
        if not hasattr(self, "_residual"):
            raise ValueError("must call set_residual")
        iterate = init_iterate
        residual = self._residual(iterate)
        residual_norms = [self._bkd.norm(residual)]
        it = 0
        if self._verbosity > 1:
            print("Iter", it, "rnorm", residual_norms[0])
        while True:
            prev_iterate = iterate
            prev_residual = residual
            iterate = self._update_iterate(
                prev_iterate, self._residual.linsolve(iterate, prev_residual)
            )
            residual = self._residual(iterate)
            residual_norm = self._bkd.norm(residual)
            residual_norms.append(residual_norm)

            it += 1
            if self._verbosity > 1:
                print("Iter", it, "rnorm", residual_norm)
            if not self._bkd.isfinite(residual_norm):
                raise RuntimeError("Residual is no longer finite")
            if residual_norm <= self._atol + self._rtol * residual_norms[0]:
                exit_msg = (
                    f"Tolerance {self._atol+self._rtol*residual_norms[0]} "
                    "reached"
                )
                break
            if it >= self._maxiters:
                exit_msg = (
                    f"Max iterations {self._maxiters} reached.\n"
                    f"Rel residual norm is {residual_norm} "
                    f"Needs to be {self._atol+self._rtol*residual_norms[0]}"
                )
                raise RuntimeError(exit_msg)
        if self._verbosity > 0:
            print(exit_msg)
        return iterate

    def __repr__(self) -> str:
        return (
            "{0}(maxiters={1}, verbosity={2}, step_size={3}, atol={4}, "
            "rol={5})"
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
