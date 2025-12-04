from abc import ABC, abstractmethod
from typing import Generic

from pyapprox.typing.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.validation import validate_backend


from typing import Protocol
from pyapprox.typing.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)
from pyapprox.typing.util.backends.protocols import Array, Backend


class KernelProtocol(Protocol, Generic[Array]):
    """
    Protocol for kernel implementations.

    Defines the interface for kernel classes, including methods for evaluating
    the kernel, computing Jacobians, and handling hyperparameters.
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
            Backend for numerical computations.
        """
        ...

    def hyp_list(self) -> HyperParameterList:
        """
        Return the list of hyperparameters associated with the kernel.

        Returns
        -------
        hyp_list : HyperParameterList
            List of hyperparameters.
        """
        ...

    def diag(self, X1: Array) -> Array:
        """
        Return the diagonal of the kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data.

        Returns
        -------
        diag : Array
            Diagonal of the kernel matrix.
        """
        ...

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Compute the kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data.
        X2 : Array, optional
            Input data. If None, the kernel matrix is computed for X1 only.

        Returns
        -------
        kernel_matrix : Array
            Kernel matrix.
        """
        ...


class KernelWithJacobianProtocol(Protocol):
    """
    Protocol for kernel implementations.

    Defines the interface for kernel classes, including methods for evaluating
    the kernel, computing Jacobians, and handling hyperparameters.
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
            Backend for numerical computations.
        """
        ...

    def hyp_list(self) -> HyperParameterList:
        """
        Return the list of hyperparameters associated with the kernel.

        Returns
        -------
        hyp_list : HyperParameterList
            List of hyperparameters.
        """
        ...

    def diag(self, X1: Array) -> Array:
        """
        Return the diagonal of the kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data.

        Returns
        -------
        diag : Array
            Diagonal of the kernel matrix.
        """
        ...

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Compute the kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data.
        X2 : Array, optional
            Input data. If None, the kernel matrix is computed for X1 only.

        Returns
        -------
        kernel_matrix : Array
            Kernel matrix.
        """
        ...


class KernelHasJacobianProtocol(Protocol, Generic[Array]):
    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Compute the Jacobian of the kernel with respect to input data.

        Parameters
        ----------
        X1 : Array
            Input data.
        X2 : Array
            Input data.

        Returns
        -------
        jacobian : Array
            Jacobian of the kernel with respect to input data.
        """
        ...


class KernelHasParameterJacobianProtocol(Protocol, Generic[Array]):
    def jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the kernel with respect to hyperparameters.

        Parameters
        ----------
        samples : Array
            Input data.

        Returns
        -------
        jacobian_wrt_params : Array
            Jacobian of the kernel with respect to hyperparameters.
        """
        ...


class KernelWithJacobianProtocol(
    KernelProtocol[Array], KernelHasJacobianProtocol[Array], Protocol
):
    pass


class KernelWithJacobianAndParameterJacobianProtocol(
    KernelWithJacobianProtocol[Array],
    KernelHasParameterJacobianProtocol[Array],
    Protocol,
):
    pass


class Kernel(ABC, Generic[Array]):
    """
    The base class for any kernel.

    Parameters
    ----------
    backend : Backend
        Backend for numerical computations.

    Attributes
    ----------
    _bkd : Backend
        Backend for numerical computations.
    _hyp_list : HyperParameterList
        List of hyperparameters associated with the kernel.
    """

    def __init__(self, backend: Backend):
        """
        Initialize the Kernel.

        Parameters
        ----------
        backend : Backend
            Backend for numerical computations.
        """
        validate_backend(backend)
        self._bkd = backend

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
            Backend for numerical computations.
        """
        return self._bkd

    def diag(self, X1: Array) -> Array:
        """
        Return the diagonal of the kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data.

        Returns
        -------
        diag : Array
            Diagonal of the kernel matrix.
        """
        return self._bkd.get_diagonal(self(X1))

    @abstractmethod
    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Compute the kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data.
        X2 : Array, optional
            Input data. If None, the kernel matrix is computed for X1 only.

        Returns
        -------
        kernel_matrix : Array
            Kernel matrix.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """
        Return a string representation of the Kernel.

        Returns
        -------
        repr : str
            String representation of the Kernel.
        """
        return "{0}({1}, bkd={2})".format(
            self.__class__.__name__,
            str(self._hyp_list),
            self._bkd.__class__.__name__,
        )
