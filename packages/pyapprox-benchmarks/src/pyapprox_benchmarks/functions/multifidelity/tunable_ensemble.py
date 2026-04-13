"""Tunable model function for multifidelity benchmarks."""

from typing import Callable, Generic

from pyapprox.util.backends.protocols import Array, Backend


class TunableModelFunction(Generic[Array]):
    """Single model from the tunable ensemble.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    func : Callable
        Function that evaluates the model.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        func: Callable[[Array], Array],
    ) -> None:
        self._bkd = bkd
        self._func = func

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return 2

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return 1

    def __call__(self, samples: Array) -> Array:
        """Evaluate the model.

        Parameters
        ----------
        samples : Array
            Input samples of shape (2, nsamples).

        Returns
        -------
        Array
            Values of shape (nqoi, nsamples) = (1, nsamples).
        """
        return self._func(samples)


