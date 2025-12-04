from typing import Generic

from pyapprox.typing.util.backends.protocols import Array
from pyapprox.typing.optimization.minimize.chained.protocols import (
    OptimizerProtocol,
)
from pyapprox.typing.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)


class ChainedOptimizer(Generic[Array]):
    """
    A chained optimizer that uses a global optimizer followed by a local
    optimizer.

    This class is useful for combining the strengths of a global optimizer
    (e.g., differential evolution) with a local optimizer (e.g., trust-constr)
    for better convergence and accuracy.

    Parameters
    ----------
    global_optimizer : ScipyDifferentialEvolutionOptimizer
        The global optimizer to use for the initial optimization.
    local_optimizer : ScipyTrustConstrOptimizer
        The local optimizer to refine the solution obtained by the global
    optimizer.
    """

    def __init__(
        self,
        global_optimizer: OptimizerProtocol[Array],
        local_optimizer: OptimizerProtocol[Array],
    ):
        self._global_optimizer = global_optimizer
        self._local_optimizer = local_optimizer

    def minimize(
        self, init_guess: Array
    ) -> ScipyOptimizerResultWrapper[Array]:
        """
        Perform optimization using the chained approach.

        Returns
        -------
        ScipyOptimizerResultWrapper
            The final optimization result after applying both optimizers.
        """
        # Step 1: Perform global optimization
        global_result = self._global_optimizer.minimize(init_guess)

        # Step 2: Use the result of the global optimizer as the initial guess
        # for the local optimizer
        local_result = self._local_optimizer.minimize(global_result.optima())

        return local_result
