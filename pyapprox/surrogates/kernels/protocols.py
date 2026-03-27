from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend
from pyapprox.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)

if TYPE_CHECKING:
    from pyapprox.surrogates.kernels.composition import (
        ProductKernel,
        SumKernel,
    )


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

    def nvars(self) -> int:
        """
        Return the number of input variables (dimensionality).

        Returns
        -------
        int
            Number of input dimensions.
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
class KernelHasHVPWrtX1Protocol(Protocol, Generic[Array]):
    """
    Protocol for kernels with HVP w.r.t. input X1.

    This is the 'Has' protocol - just checks for the method existence.
    For full protocol including jacobian, use KernelWithJacobianAndHVPWrtX1Protocol.
    """

    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array:
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


@runtime_checkable
class KernelHasHVPWrtParamsProtocol(Protocol, Generic[Array]):
    """
    Protocol for kernels with HVP w.r.t. hyperparameters.

    This is the 'Has' protocol - just checks for the method existence.
    For full protocol including jacobian_wrt_params, use
    KernelWithParameterJacobianAndHVPProtocol.
    """

    def hvp_wrt_params(self, samples: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. hyperparameters.

        Computes HVP = Σ_j (∂²K/∂θ_i∂θ_j) * v[j] for each i,
        without forming the full Hessian tensor.

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).
        direction : Array
            Direction vector, shape (nparams,).

        Returns
        -------
        hvp : Array
            Hessian-vector product, shape (n, n, nparams).
            hvp[:, :, i] = Σ_j (∂²K/∂θ_i∂θ_j) * v[j]
        """
        ...


class KernelWithJacobianProtocol(
    KernelProtocol[Array], KernelHasJacobianProtocol[Array], Protocol
):
    """Kernel with jacobian w.r.t. input X1."""

    pass


class KernelWithJacobianAndHVPWrtX1Protocol(
    KernelWithJacobianProtocol[Array],
    KernelHasHVPWrtX1Protocol[Array],
    Protocol,
):
    """Kernel with both jacobian and hvp_wrt_x1 (second derivative w.r.t. X1)."""

    pass


class KernelWithParameterJacobianProtocol(
    KernelProtocol[Array],
    KernelHasParameterJacobianProtocol[Array],
    Protocol,
):
    """Kernel with jacobian_wrt_params."""

    pass


class KernelWithParameterJacobianAndHVPProtocol(
    KernelWithParameterJacobianProtocol[Array],
    KernelHasHVPWrtParamsProtocol[Array],
    Protocol,
):
    """Kernel with both jacobian_wrt_params and hvp_wrt_params."""

    pass


class KernelWithJacobianAndParameterJacobianProtocol(
    KernelWithJacobianProtocol[Array],
    KernelHasParameterJacobianProtocol[Array],
    Protocol,
):
    """Kernel with jacobian (w.r.t. X1) and jacobian_wrt_params."""

    pass


class KernelWithFullDerivativesProtocol(
    KernelWithJacobianAndHVPWrtX1Protocol[Array],
    KernelWithParameterJacobianAndHVPProtocol[Array],
    Protocol,
):
    """Kernel with all derivative methods: jacobian, hvp_wrt_x1, jacobian_wrt_params,
    hvp_wrt_params."""

    pass


@runtime_checkable
class SeparableKernelProtocol(Protocol, Generic[Array]):
    """
    Protocol for separable (product) kernels.

    A separable kernel has the form:
        k(x, y) = prod_d k_d(x_d, y_d)

    where k_d is a 1D kernel operating on dimension d. This structure
    enables efficient computation of multidimensional integrals as
    products of 1D integrals.

    Only kernels where the dimensions truly factor satisfy this protocol:
    - SeparableProductKernel: Explicitly constructed from 1D kernels
    - SquaredExponentialKernel: exp(-0.5 * sum_d ...) = prod_d exp(-0.5 * ...)

    Note: Matern 3/2 and 5/2 kernels are NOT separable because they use
    the combined Euclidean distance inside nonlinear polynomial terms.
    """

    def nvars(self) -> int:
        """Return the number of input dimensions."""
        ...

    def get_kernel_1d(self, dim: int) -> "KernelProtocol[Array]":
        """
        Get the 1D kernel for a specific dimension.

        Parameters
        ----------
        dim : int
            Dimension index (0 to nvars-1).

        Returns
        -------
        kernel_1d : KernelProtocol[Array]
            The 1D kernel for dimension dim.
            For SeparableProductKernel, returns the stored 1D kernel.
            For SquaredExponentialKernel, returns a new 1D SE kernel
            with that dimension's length scale.
        """
        ...


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

    def __init__(self, backend: Backend[Array]):
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

    def __mul__(self, other: "Kernel[Array]") -> "ProductKernel":
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
        from pyapprox.surrogates.kernels.composition import ProductKernel

        return ProductKernel(self, other)

    def __add__(self, other: "Kernel[Array]") -> "SumKernel":
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
        from pyapprox.surrogates.kernels.composition import SumKernel

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
