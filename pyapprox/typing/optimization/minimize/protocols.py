"""Protocols for bindable optimizers supporting deferred binding."""

from typing import Protocol, Generic, Optional, Self, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.optimization.minimize.objective.protocols import (
    ObjectiveProtocol,
)
from pyapprox.typing.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)


@runtime_checkable
class BindableOptimizerProtocol(Protocol, Generic[Array]):
    """Protocol for optimizers that support deferred binding.

    This protocol defines the interface for optimizers that can be configured
    with options first, then bound to an objective and bounds later. This
    enables patterns like:

    ```python
    # Create optimizer with options only (no objective/bounds)
    optimizer = ScipyTrustConstrOptimizer(verbosity=1, maxiter=500)

    # Later, bind to objective and bounds
    optimizer.bind(objective, bounds)
    result = optimizer.minimize(init_guess)
    ```

    The deferred binding pattern is useful for:
    - Configuring a "template" optimizer to share across multiple problems
    - Integrating with GPs where the optimizer is set before fit() is called
    - Cloning optimizers to avoid shared state across concurrent uses
    """

    def bind(
        self,
        objective: ObjectiveProtocol[Array],
        bounds: Array,
        constraints: Optional[object] = None,
    ) -> Self:
        """Bind the optimizer to an objective, bounds, and optional constraints.

        Parameters
        ----------
        objective : ObjectiveProtocol
            The objective function to minimize.
        bounds : Array
            Bounds for the optimization variables, shape (nvars, 2).
        constraints : Optional[object], optional
            Constraints for the optimization problem. Defaults to None.

        Returns
        -------
        Self
            Returns self to enable method chaining.

        Notes
        -----
        Calling bind() on an already-bound optimizer replaces the previous
        binding. This allows re-using a single optimizer instance for
        multiple problems.
        """
        ...

    def is_bound(self) -> bool:
        """Return True if the optimizer has been bound to an objective.

        Returns
        -------
        bool
            True if bind() has been called, False otherwise.
        """
        ...

    def minimize(self, init_guess: Array) -> ScipyOptimizerResultWrapper[Array]:
        """Run optimization starting from the initial guess.

        Parameters
        ----------
        init_guess : Array
            Initial guess for the optimization variables, shape (nvars, 1).

        Returns
        -------
        ScipyOptimizerResultWrapper[Array]
            Wrapped optimization result.

        Raises
        ------
        RuntimeError
            If the optimizer has not been bound (is_bound() returns False).
        """
        ...

    def copy(self) -> Self:
        """Return an unbound copy of this optimizer with the same options.

        The copy preserves all optimizer options (maxiter, gtol, etc.) but
        is in an unbound state, regardless of whether the original is bound.

        Returns
        -------
        Self
            A new optimizer instance with the same options, unbound.
        """
        ...

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            The backend used for array computations.

        Raises
        ------
        RuntimeError
            If the optimizer has not been bound.
        """
        ...
