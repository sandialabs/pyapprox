from typing import Generic, Optional, Self, cast

from pyapprox.optimization.minimize.constraints.protocols import (
    SequenceOfConstraintProtocols,
)
from pyapprox.optimization.minimize.objective.protocols import (
    ObjectiveProtocol,
)
from pyapprox.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.util.backends.protocols import Array, Backend


class ChainedOptimizer(Generic[Array]):
    """A chained optimizer that uses a global optimizer followed by a local.

    This class is useful for combining the strengths of a global optimizer
    (e.g., differential evolution) with a local optimizer (e.g., trust-constr)
    for better convergence and accuracy.

    Supports two usage patterns:

    1. Pre-bound optimizers (original API):
    ```python
    global_opt = ScipyDifferentialEvolutionOptimizer(obj, bounds)
    local_opt = ScipyTrustConstrOptimizer(obj, bounds)
    chained = ChainedOptimizer(global_opt, local_opt)
    result = chained.minimize(init_guess)
    ```

    2. Unbound optimizers with deferred binding (new API):
    ```python
    global_opt = ScipyDifferentialEvolutionOptimizer(maxiter=100)
    local_opt = ScipyTrustConstrOptimizer(maxiter=500)
    chained = ChainedOptimizer(global_opt, local_opt)
    chained.bind(objective, bounds)
    result = chained.minimize(init_guess)
    ```

    Parameters
    ----------
    global_optimizer : BindableOptimizerProtocol[Array]
        The global optimizer to use for the initial optimization.
    local_optimizer : BindableOptimizerProtocol[Array]
        The local optimizer to refine the solution obtained by the global
        optimizer.
    """

    def __init__(
        self,
        global_optimizer: BindableOptimizerProtocol[Array],
        local_optimizer: BindableOptimizerProtocol[Array],
    ):
        if not isinstance(global_optimizer, BindableOptimizerProtocol):
            raise TypeError(
                f"global_optimizer must satisfy BindableOptimizerProtocol, "
                f"got {type(global_optimizer).__name__}"
            )
        if not isinstance(local_optimizer, BindableOptimizerProtocol):
            raise TypeError(
                f"local_optimizer must satisfy BindableOptimizerProtocol, "
                f"got {type(local_optimizer).__name__}"
            )
        self._global_optimizer = global_optimizer
        self._local_optimizer = local_optimizer

    def bind(
        self,
        objective: ObjectiveProtocol[Array],
        bounds: Array,
        constraints: Optional[SequenceOfConstraintProtocols[Array]] = None,
    ) -> Self:
        """Bind objective, bounds, and constraints to both optimizers.

        Parameters
        ----------
        objective : ObjectiveProtocol[Array]
            Objective function for the optimization problem.
        bounds : Array
            Bounds for the optimization variables, shape (nvars, 2).
        constraints : Optional[SequenceOfConstraintProtocols[Array]], optional
            Constraints for the optimization problem. Defaults to None.

        Returns
        -------
        Self
            Returns self to enable method chaining.
        """
        self._global_optimizer.bind(objective, bounds, constraints)
        self._local_optimizer.bind(objective, bounds, constraints)
        return self

    def is_bound(self) -> bool:
        """Return True if both optimizers are bound.

        Returns
        -------
        bool
            True if both optimizers are bound, False otherwise.
        """
        return self._global_optimizer.is_bound() and self._local_optimizer.is_bound()

    def copy(self) -> Self:
        """Return an unbound copy with same options.

        Returns
        -------
        Self
            A new ChainedOptimizer with copies of the underlying optimizers.
        """
        return cast(
            Self,
            ChainedOptimizer(
                global_optimizer=self._global_optimizer.copy(),
                local_optimizer=self._local_optimizer.copy(),
            ),
        )

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations.

        Raises
        ------
        RuntimeError
            If the optimizers have not been bound.
        """
        if not self.is_bound():
            raise RuntimeError("Optimizer not bound. Call bind() first.")
        return self._global_optimizer.bkd()

    def minimize(self, init_guess: Array) -> ScipyOptimizerResultWrapper[Array]:
        """Perform optimization using the chained approach.

        Parameters
        ----------
        init_guess : Array
            Initial guess for the optimization variables.

        Returns
        -------
        ScipyOptimizerResultWrapper[Array]
            The final optimization result after applying both optimizers.

        Raises
        ------
        RuntimeError
            If the optimizers have not been bound.
        """
        if not self.is_bound():
            raise RuntimeError("Optimizer not bound. Call bind() first.")

        # Step 1: Perform global optimization
        global_result = self._global_optimizer.minimize(init_guess)

        # Step 2: Use the result of the global optimizer as the initial guess
        # for the local optimizer
        local_result = self._local_optimizer.minimize(global_result.optima())

        return local_result

    def local_optimizer_verbosity(self) -> int:
        """Return verbosity level from local optimizer.

        Returns
        -------
        int
            Verbosity level from the local optimizer, or 0 if not available.
        """
        if hasattr(self._local_optimizer, "_verbosity"):
            return int(self._local_optimizer._verbosity)
        return 0
