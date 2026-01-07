from typing import Protocol, runtime_checkable, Generic, Union, Any

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.typing.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
)


@runtime_checkable
class FunctionWithJacobianAndHVPProtocol(Protocol, Generic[Array]):
    """Protocol for functions with single-sample Jacobian and HVP.

    The hvp method computes the Hessian-vector product at a single sample.
    For scalar-valued functions (nqoi=1), this is the product of the Hessian
    matrix with a direction vector.
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
        """
        ...

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample. Shape: (nvars, 1). Must be 2D.
        vec : Array
            Direction vector. Shape: (nvars, 1). Must be 2D.

        Returns
        -------
        Array
            Hessian-vector product. Shape: (nvars, 1)

        Raises
        ------
        ValueError
            If sample or vec is not 2D with shape (nvars, 1).
        """
        ...


@runtime_checkable
class FunctionWithJVPAndHVPProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jvp(self, sample: Array, vec: Array) -> Array: ...

    def hvp(self, sample: Array, vec: Array) -> Array: ...


@runtime_checkable
class FunctionWithJacobianAndWHVPProtocol(Protocol, Generic[Array]):
    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def jacobian(self, sample: Array) -> Array: ...

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array: ...


FunctionWithHVPAndJacobianOrJVPProtocol = Union[
    FunctionWithJacobianAndHVPProtocol[Array],
    FunctionWithJVPAndHVPProtocol[Array],
]


def function_has_hvp_and_jacobian_or_jvp(function: Any) -> bool:
    if function.nqoi() == 1:
        if isinstance(function, FunctionWithJacobianAndHVPProtocol):
            return True
        if isinstance(function, FunctionWithJVPAndHVPProtocol):
            return True
    if isinstance(function, FunctionWithJacobianAndWHVPProtocol):
        return True
    return False


@runtime_checkable
class FunctionWithHessianBatchProtocol(Protocol, Generic[Array]):
    """Protocol for functions with batch Hessian computation.

    The hessian_batch method takes multiple samples of shape (nvars, nsamples)
    and returns Hessians of shape (nsamples, nqoi, nvars, nvars).
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def hessian_batch(self, samples: Array) -> Array:
        """Compute Hessians at multiple samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Hessians for each sample. Shape: (nsamples, nqoi, nvars, nvars)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (nvars, nsamples).
        """
        ...


@runtime_checkable
class FunctionWithHVPBatchProtocol(Protocol, Generic[Array]):
    """Protocol for functions with batch HVP computation.

    The hvp_batch method takes multiple samples and vectors, and returns
    Hessian-vector products for each pair.
    """

    def bkd(self) -> Backend[Array]: ...

    def nvars(self) -> int: ...

    def nqoi(self) -> int: ...

    def __call__(self, samples: Array) -> Array: ...

    def hvp_batch(self, samples: Array, vecs: Array) -> Array:
        """Compute Hessian-vector products at multiple samples.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples). Must be 2D.
        vecs : Array
            Direction vectors. Shape: (nvars, nsamples). Must be 2D.

        Returns
        -------
        Array
            Hessian-vector products. Shape: (nsamples, nvars)

        Raises
        ------
        ValueError
            If samples or vecs is not 2D with shape (nvars, nsamples).
        """
        ...
