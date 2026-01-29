"""Result classes for MSE fitting operations."""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.functiontrain.functiontrain import FunctionTrain
from pyapprox.typing.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)


class MSEFitterResult(Generic[Array]):
    """Result from MSE-based gradient optimization.

    All attributes are accessed via methods per CLAUDE.md convention.

    Only exposes attributes guaranteed by the optimizer result protocol.
    Use optimizer_result().get_raw_result() for optimizer-specific attributes
    like number of iterations.

    Parameters
    ----------
    surrogate : FunctionTrain[Array]
        The fitted FunctionTrain surrogate.
    optimizer_result : ScipyOptimizerResultWrapper[Array]
        The underlying optimizer result with detailed diagnostics.
    final_loss : float
        Final MSE loss value.
    """

    def __init__(
        self,
        surrogate: FunctionTrain[Array],
        optimizer_result: ScipyOptimizerResultWrapper[Array],
        final_loss: float,
    ):
        self._surrogate = surrogate
        self._optimizer_result = optimizer_result
        self._final_loss = final_loss

    def surrogate(self) -> FunctionTrain[Array]:
        """Return the fitted FunctionTrain."""
        return self._surrogate

    def params(self) -> Array:
        """Return fitted parameters."""
        return self._surrogate._flatten_params()

    def optimizer_result(self) -> ScipyOptimizerResultWrapper[Array]:
        """Return the underlying optimizer result.

        Use get_raw_result() for optimizer-specific attributes like nit.
        """
        return self._optimizer_result

    def final_loss(self) -> float:
        """Return the final MSE loss value."""
        return self._final_loss

    def converged(self) -> bool:
        """Return whether the optimization converged successfully."""
        return self._optimizer_result.success()

    def bkd(self) -> Backend[Array]:
        """Return backend from surrogate."""
        return self._surrogate.bkd()

    def __call__(self, samples: Array) -> Array:
        """Evaluate fitted surrogate at samples.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Values at samples. Shape: (nqoi, nsamples)
        """
        return self._surrogate(samples)

    def __repr__(self) -> str:
        return (
            f"MSEFitterResult(converged={self.converged()}, "
            f"final_loss={self._final_loss:.6e})"
        )
