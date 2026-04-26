"""Result classes for SUPN fitting operations."""

from typing import Generic

from pyapprox.optimization.minimize.scipy.scipy_result import (
    ScipyOptimizerResultWrapper,
)
from pyapprox.surrogates.supn.supn import SUPN
from pyapprox.util.backends.protocols import Array, Backend


class SUPNFitterResult(Generic[Array]):
    """Result from SUPN MSE fitting.

    Parameters
    ----------
    surrogate : SUPN[Array]
        The fitted SUPN surrogate.
    optimizer_result : ScipyOptimizerResultWrapper[Array]
        The underlying optimizer result.
    final_loss : float
        Final MSE loss value.
    """

    def __init__(
        self,
        surrogate: SUPN[Array],
        optimizer_result: ScipyOptimizerResultWrapper[Array],
        final_loss: float,
    ) -> None:
        self._surrogate = surrogate
        self._optimizer_result = optimizer_result
        self._final_loss = final_loss

    def surrogate(self) -> SUPN[Array]:
        """Return the fitted SUPN."""
        return self._surrogate

    def params(self) -> Array:
        """Return fitted parameters."""
        return self._surrogate._flatten_params()

    def optimizer_result(self) -> ScipyOptimizerResultWrapper[Array]:
        """Return the underlying optimizer result."""
        return self._optimizer_result

    def final_loss(self) -> float:
        """Return the final MSE loss value."""
        return self._final_loss

    def converged(self) -> bool:
        """Return whether the optimization converged."""
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
            Values. Shape: (nqoi, nsamples)
        """
        return self._surrogate(samples)

    def __repr__(self) -> str:
        return (
            f"SUPNFitterResult(converged={self.converged()}, "
            f"final_loss={self._final_loss:.6e})"
        )
