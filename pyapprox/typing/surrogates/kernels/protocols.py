from abc import ABC, abstractmethod
from typing import Generic, Protocol, runtime_checkable

from pyapprox.typing.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.validation import validate_backend


@runtime_checkable
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


@runtime_checkable
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


@runtime_checkable
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


@runtime_checkable
class KernelHasHVPProtocol(Protocol, Generic[Array]):
    def hvp_wrt_x1(
        self, X1: Array, X2: Array, direction: Array
    ) -> Array:
        """
        Compute Hessian-vector product of kernel w.r.t. first argument.

        For kernel k(X1, X2), computes:
            H[k(X1, X2)] · V
        where H is the Hessian matrix ∂²k/∂X1² and V is the direction vector.

        This is more efficient than computing the full Hessian tensor and
        then contracting, especially for high-dimensional problems.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of input points
        X2 : Array, shape (nvars, n2)
            Second set of input points
        direction : Array, shape (nvars,)
            Direction vector for Hessian-vector product

        Returns
        -------
        hvp : Array, shape (n1, n2, nvars)
            H[k(X1[:, i], X2[:, j])]·V for each pair (i, j)

            This shape is consistent with jacobian() which returns (n1, n2, nvars).

        Notes
        -----
        This computes the HVP without materializing the (nvars, nvars, n1, n2)
        Hessian tensor, making it memory-efficient for high dimensions.
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

    def __mul__(self, other: "Kernel") -> "ProductKernel":
        """
        Multiply two kernels element-wise.

        Parameters
        ----------
        other : Kernel
            Another kernel to multiply with.

        Returns
        -------
        product_kernel : ProductKernel
            Product of the two kernels.
        """
        from pyapprox.typing.surrogates.kernels.composition import ProductKernel

        return ProductKernel(self, other)

    def __add__(self, other: "Kernel") -> "SumKernel":
        """
        Add two kernels element-wise.

        Parameters
        ----------
        other : Kernel
            Another kernel to add.

        Returns
        -------
        sum_kernel : SumKernel
            Sum of the two kernels.
        """
        from pyapprox.typing.surrogates.kernels.composition import SumKernel

        return SumKernel(self, other)

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
