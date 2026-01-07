"""Utilities for tensor/array transfer in parallel execution.

This module provides TensorTransfer for handling NumPy/PyTorch array
conversion, which is necessary for multiprocessing since PyTorch
tensors cannot be pickled directly in some contexts.
"""

from typing import Callable, Generic, TypeVar

from pyapprox.typing.util.backends.protocols import Array, Backend

T = TypeVar("T")


class TensorTransfer(Generic[Array]):
    """Handle array/tensor conversion for multiprocessing.

    When using multiprocessing with PyTorch tensors, we need to convert
    to NumPy arrays for pickling, then convert back. This class provides
    utilities for wrapping functions to handle this conversion.

    Parameters
    ----------
    bkd : Backend[Array]
        The array backend (NumPy or PyTorch).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.torch import TorchBkd
    >>> bkd = TorchBkd()
    >>> transfer = TensorTransfer(bkd)
    >>> def my_func(x):
    ...     return x * 2
    >>> wrapped = transfer.wrap_function(my_func)
    >>> # wrapped accepts numpy, converts to tensor, calls func, returns numpy
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the array backend."""
        return self._bkd

    def to_numpy(self, arr: Array) -> "Array":
        """Convert array to NumPy for serialization.

        Parameters
        ----------
        arr : Array
            Array to convert (NumPy or PyTorch).

        Returns
        -------
        ndarray
            NumPy array representation.
        """
        return self._bkd.to_numpy(arr)

    def from_numpy(self, arr: "Array") -> Array:
        """Convert NumPy array back to backend type.

        Parameters
        ----------
        arr : ndarray
            NumPy array to convert.

        Returns
        -------
        Array
            Array in backend's native format.
        """
        return self._bkd.asarray(arr)

    def wrap_function(
        self,
        func: Callable[[Array], Array],
    ) -> Callable[["Array"], "Array"]:
        """Wrap function to handle numpy conversion for multiprocessing.

        The wrapped function:
        1. Receives NumPy arrays (serialized from workers)
        2. Converts to backend format (e.g., PyTorch tensor)
        3. Calls the original function
        4. Converts result back to NumPy for return

        Parameters
        ----------
        func : Callable[[Array], Array]
            Function that operates on backend arrays.

        Returns
        -------
        Callable
            Wrapped function that accepts/returns NumPy arrays.
        """

        def wrapped(arr_np: "Array") -> "Array":
            arr = self.from_numpy(arr_np)
            result = func(arr)
            return self.to_numpy(result)

        return wrapped

    def wrap_starmap_function(
        self,
        func: Callable[..., Array],
    ) -> Callable[..., "Array"]:
        """Wrap function for starmap with numpy conversion.

        Similar to wrap_function but for functions with multiple arguments.

        Parameters
        ----------
        func : Callable[..., Array]
            Function that operates on backend arrays.

        Returns
        -------
        Callable
            Wrapped function that accepts/returns NumPy arrays.
        """

        def wrapped(*args_np: "Array") -> "Array":
            args = tuple(self.from_numpy(arg) for arg in args_np)
            result = func(*args)
            return self.to_numpy(result)

        return wrapped
