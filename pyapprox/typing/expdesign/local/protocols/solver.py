"""
Protocols for local OED solvers and optimizers.

Solvers find optimal design weights by minimizing a criterion subject to
simplex constraints (sum = 1, weights >= 0).
"""

from typing import Protocol, Generic, runtime_checkable, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class OptimizerResultProtocol(Protocol, Generic[Array]):
    """
    Protocol for optimization results.

    Methods
    -------
    optima()
        Get optimal solution.
    value()
        Get optimal objective value.
    success()
        Whether optimization succeeded.
    """

    def optima(self) -> Array:
        """
        Get optimal solution.

        Returns
        -------
        Array
            Optimal design weights. Shape: (nvars, 1)
        """
        ...

    def value(self) -> float:
        """
        Get optimal objective value.

        Returns
        -------
        float
            Objective value at optimum.
        """
        ...

    def success(self) -> bool:
        """
        Whether optimization succeeded.

        Returns
        -------
        bool
            True if optimization converged successfully.
        """
        ...


@runtime_checkable
class OptimizerProtocol(Protocol, Generic[Array]):
    """
    Protocol for optimizers.

    Optimizers minimize objective functions subject to constraints.

    Methods
    -------
    minimize(init_guess)
        Run optimization from initial guess.
    """

    def minimize(self, init_guess: Array) -> OptimizerResultProtocol[Array]:
        """
        Run optimization.

        Parameters
        ----------
        init_guess : Array
            Initial guess. Shape: (nvars, 1)

        Returns
        -------
        OptimizerResultProtocol[Array]
            Optimization result.
        """
        ...


@runtime_checkable
class LocalOEDSolverProtocol(Protocol, Generic[Array]):
    """
    Protocol for local OED solvers.

    Solvers combine a criterion with an optimizer to find optimal designs.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of design variables.
    construct(init_weights)
        Find optimal design weights.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Number of design variables.

        Returns
        -------
        int
            Number of candidate design points.
        """
        ...

    def construct(self, init_weights: Optional[Array] = None) -> Array:
        """
        Find optimal design weights.

        Parameters
        ----------
        init_weights : Array, optional
            Initial design weights. Shape: (nvars, 1)
            If None, uses uniform weights.

        Returns
        -------
        Array
            Optimal design weights. Shape: (nvars, 1)
        """
        ...
