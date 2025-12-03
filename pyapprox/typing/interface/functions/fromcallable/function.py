from abc import ABC, abstractmethod
from typing import Generic, Callable, Any

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.interface.functions.protocols.validation import (
    validate_samples,
    validate_values,
)


class FunctionFromCallable(Generic[Array]):
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
        self._bkd = bkd
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

    def bkd(self) -> Backend[Array]:
        return self._bkd

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
