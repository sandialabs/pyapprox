from abc import ABC, abstractmethod
from typing import Protocol, Generic, Callable, runtime_checkable, Any

from pyapprox.typing.util.backend import Array, Backend


@runtime_checkable
class FunctionProtocol(Protocol, Generic[Array]):
    """
    A protocol defining the required interface for a Function.
    """

    _bkd: Backend[Array]

    def nvars(self) -> int:
        """
        Return the number of variables in the function.
        """
        ...

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest in the function.
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the function with the given samples.
        """
        ...


def validate_samples(nvars: int, samples: Array) -> None:
    expected_rows = nvars
    actual_rows, actual_cols = samples.shape
    if actual_rows != expected_rows:
        raise ValueError(
            f"Invalid samples shape: expected {expected_rows} rows, "
            f"got {actual_rows} rows."
        )


def validate_values(nqoi: int, samples: Array, values: Array) -> None:
    expected_shape = (nqoi, samples.shape[1])
    if values.shape != expected_shape:
        raise ValueError(
            f"Invalid values shape: expected {expected_shape}, "
            f"got {values.shape}."
        )


def validate_sample(nvars: int, samples: Array) -> None:
    """
    Validate that the sample has shape (nvars, 1).
    Some member functions may only use 1 sample.
    """
    expected_shape = (nvars, 1)
    actual_shape = samples.shape
    if actual_shape != expected_shape:
        raise ValueError(
            f"Invalid sample shape: expected {expected_shape}, "
            f"got {actual_shape}."
        )


class Function(ABC, Generic[Array]):
    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def nqoi(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, samples: Array) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the object for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"nvars={self.nvars()}, "
            f"nqoi={self.nqoi()}, "
            f"bkd={type(self._bkd).__name__})"
        )


class FunctionFromCallable(Function[Array]):
    def __init__(
        self,
        nqoi: int,
        nvars: int,
        fun: Callable[[Array], Array],
        bkd: Backend[Array],
    ):
        self._nqoi = nqoi
        self._nvars = nvars
        if not callable(fun):
            raise ValueError(
                "The provided 'fun' object must be callable. "
                "Expected a callable object that takes an input of type "
                "'Array' and returns an output of type 'Array'. "
                f"Got an object of type {type(fun).__name__}. "
                f"Object details: {str(fun)}"
            )
        super().__init__(bkd)
        self._fun: Callable[[Array], Array] = fun

    def nvars(self) -> int:
        return self._nvars

    def nqoi(self) -> int:
        return self._nqoi

    def __call__(self, samples: Array) -> Array:
        validate_samples(self.nvars(), samples)
        values = self._fun(samples)
        validate_values(self.nqoi(), samples, values)
        return values


def validate_function(function: Any) -> None:
    if not isinstance(function, FunctionProtocol):
        raise TypeError(
            f"Invalid function type: expected an object implementing "
            f"FunctionProtocol, got {type(function).__name__}. "
            f"Object details: {function}"
        )
