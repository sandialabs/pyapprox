from typing import Protocol, Generic, runtime_checkable, Union, Any

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class FunctionWithJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for functions with single-sample Jacobian computation.

    The jacobian method takes a single sample of shape (nvars, 1) and returns
    the Jacobian matrix of shape (nqoi, nvars).
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample. Shape: (nvars, 1). Must be 2D.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nqoi, nvars)

        Raises
        ------
        ValueError
            If sample is not 2D with shape (nvars, 1).
        """
        ...


@runtime_checkable
class FunctionWithJacobianBatchProtocol(Protocol, Generic[Array]):
    """Protocol for functions with batch Jacobian computation.

    The jacobian_batch method takes multiple samples of shape (nvars, nsamples)
    and returns Jacobians of shape (nsamples, nqoi, nvars).
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobian_batch(self, samples: Array) -> Array:
        """Compute Jacobians at multiple samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Jacobians for each sample. Shape: (nsamples, nqoi, nvars)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (nvars, nsamples).
        """
        ...


@runtime_checkable
class FunctionWithJVPProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jvp(self, sample: Array, vec: Array) -> Array: ...


FunctionWithJacobianOrJVPProtocol = Union[
    FunctionWithJacobianProtocol[Array],
    FunctionWithJVPProtocol[Array],
]


def function_has_jacobian_or_jvp(function: Any) -> bool:
    if not isinstance(
        function, FunctionWithJacobianProtocol
    ) and not isinstance(function, FunctionWithJVPProtocol):
        return False
    return True
