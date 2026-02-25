from typing import Protocol, Generic, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class FunctionProtocol(Protocol, Generic[Array]):
    """
    A protocol defining the required interface for a Function.
    """

    def bkd(self) -> Backend[Array]: ...

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
