"""Result classes for MFNet fitters."""

from typing import Generic, List, Optional

from pyapprox.util.backends.protocols import Array
from pyapprox.surrogates.mfnets.network import MFNet


class MFNetGradientFitResult(Generic[Array]):
    """Result of gradient-based MFNet fitting.

    Parameters
    ----------
    surrogate : MFNet[Array]
        The fitted network.
    loss_value : float
        Final loss value.
    optimizer_result : object
        Raw optimizer result.
    """

    def __init__(
        self,
        surrogate: MFNet[Array],
        loss_value: float,
        optimizer_result: Optional[object] = None,
    ) -> None:
        self._surrogate = surrogate
        self._loss_value = loss_value
        self._optimizer_result = optimizer_result

    def surrogate(self) -> MFNet[Array]:
        return self._surrogate

    def loss_value(self) -> float:
        return self._loss_value

    def optimizer_result(self) -> Optional[object]:
        return self._optimizer_result

    def __call__(self, samples: Array) -> Array:
        return self._surrogate(samples)


class MFNetALSFitResult(Generic[Array]):
    """Result of alternating least-squares MFNet fitting.

    Parameters
    ----------
    surrogate : MFNet[Array]
        The fitted network.
    loss_history : list of float
        Loss at each sweep.
    n_sweeps : int
        Number of sweeps performed.
    converged : bool
        Whether the algorithm converged.
    """

    def __init__(
        self,
        surrogate: MFNet[Array],
        loss_history: List[float],
        n_sweeps: int,
        converged: bool,
    ) -> None:
        self._surrogate = surrogate
        self._loss_history = loss_history
        self._n_sweeps = n_sweeps
        self._converged = converged

    def surrogate(self) -> MFNet[Array]:
        return self._surrogate

    def loss_history(self) -> List[float]:
        return self._loss_history

    def n_sweeps(self) -> int:
        return self._n_sweeps

    def converged(self) -> bool:
        return self._converged

    def __call__(self, samples: Array) -> Array:
        return self._surrogate(samples)
