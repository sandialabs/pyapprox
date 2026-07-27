from typing import Generic, Protocol, runtime_checkable

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

    def __call__(self, samples: Array, /) -> Array:
        """
        Evaluate the function with the given samples.

        The parameter is positional-only: the name is not part of the
        contract, so domain protocols may rename it (design_weights,
        params_1d, ...) and remain structural subtypes.
        """
        ...
