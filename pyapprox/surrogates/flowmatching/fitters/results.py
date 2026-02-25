"""Result class for flow matching fitting operations."""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class FlowMatchingFitResult(Generic[Array]):
    """Result from flow matching fitting.

    Contains only universally available outputs: the fitted surrogate
    and the final training loss. Fitter-specific metadata (e.g.
    optimizer convergence info, iteration history) should be accessed
    from the fitter or optimizer directly.

    Parameters
    ----------
    surrogate : object
        The fitted vector field (e.g. BasisExpansion).
    training_loss : float
        Final training loss value.
    """

    def __init__(
        self,
        surrogate: object,
        training_loss: float,
    ) -> None:
        self._surrogate = surrogate
        self._training_loss = training_loss

    def surrogate(self) -> object:
        """Return the fitted vector field."""
        return self._surrogate

    def training_loss(self) -> float:
        """Return the final training loss."""
        return self._training_loss

    def bkd(self) -> Backend[Array]:
        """Return backend from surrogate."""
        return self._surrogate.bkd()  # type: ignore[union-attr]

    def __call__(self, samples: Array) -> Array:
        """Evaluate fitted VF at samples (delegates to surrogate)."""
        return self._surrogate(samples)  # type: ignore[operator]

    def __repr__(self) -> str:
        return f"FlowMatchingFitResult(loss={self._training_loss:.6e})"
