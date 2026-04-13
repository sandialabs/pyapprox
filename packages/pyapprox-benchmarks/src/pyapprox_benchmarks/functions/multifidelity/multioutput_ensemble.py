"""Multi-output model function for multifidelity benchmarks.

Reference: Dixon et al. (2024), SIAM/ASA JUQ
"""
#TODO: add http link to Dixon paper https://doi.org/10.1137/23M1607994

from typing import Callable, Generic

from pyapprox.util.backends.protocols import Array, Backend


class MultiOutputModelFunction(Generic[Array]):
    """Single multi-output model.

    Implements FunctionProtocol with multiple QoI.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    func : Callable
        Function that evaluates the model.
    nqoi : int
        Number of quantities of interest.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        func: Callable[[Array], Array],
        nqoi: int,
    ) -> None:
        self._bkd = bkd
        self._func = func
        self._nqoi = nqoi

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables."""
        return 1

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        """Evaluate the model.

        Parameters
        ----------
        samples : Array
            Input samples of shape (1, nsamples).

        Returns
        -------
        Array
            Values of shape (nqoi, nsamples).
        """
        return self._func(samples)


