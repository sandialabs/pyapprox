"""
Composition kernels for building complex kernels from simple ones.

This module provides kernel composition operations (product and sum) that enable
building sophisticated kernels from simpler building blocks.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.surrogates.kernels.protocols import Kernel, KernelHasHVPProtocol


class CompositionKernel(Kernel, Generic[Array]):
    """
    Base class for kernel compositions.

    Combines two kernels and manages their combined hyperparameters.

    Parameters
    ----------
    kernel1 : Kernel
        First kernel in the composition.
    kernel2 : Kernel
        Second kernel in the composition.

    Attributes
    ----------
    _kernel1 : Kernel
        First kernel.
    _kernel2 : Kernel
        Second kernel.
    _hyp_list : HyperParameterList
        Combined hyperparameter list from both kernels.
    """

    def __init__(self, kernel1: Kernel, kernel2: Kernel):
        """
        Initialize the CompositionKernel.

        Parameters
        ----------
        kernel1 : Kernel
            First kernel in the composition.
        kernel2 : Kernel
            Second kernel in the composition.

        Raises
        ------
        ValueError
            If the kernels have different backends.
        """
        # Validate backends match
        if kernel1.bkd().__class__ != kernel2.bkd().__class__:
            raise ValueError(
                f"Kernels must have the same backend type. "
                f"Got {kernel1.bkd().__class__.__name__} and "
                f"{kernel2.bkd().__class__.__name__}"
            )

        # Initialize with kernel1's backend
        super().__init__(kernel1.bkd())

        self._kernel1 = kernel1
        self._kernel2 = kernel2

        # Combine hyperparameter lists
        self._hyp_list = kernel1.hyp_list() + kernel2.hyp_list()

    def hyp_list(self) -> HyperParameterList:
        """
        Return the combined hyperparameter list.

        Returns
        -------
        hyp_list : HyperParameterList
            Combined hyperparameters from both kernels.
        """
        return self._hyp_list

    def nvars(self) -> int:
        """
        Return the number of input variables.

        Infers from the first kernel.

        Returns
        -------
        nvars : int
            Number of input dimensions.
        """
        return self._kernel1.nvars()


class ProductKernel(CompositionKernel):
    """
    Product of two kernels (element-wise multiplication).

    K(X1, X2) = K1(X1, X2) * K2(X1, X2)

    Parameters
    ----------
    kernel1 : Kernel
        First kernel.
    kernel2 : Kernel
        Second kernel.

    Examples
    --------
    >>> from pyapprox.typing.surrogates.kernels import MaternKernel, ConstantKernel
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> matern = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> constant = ConstantKernel(1.0, (0.1, 10.0), bkd)
    >>> product = matern * constant  # Uses operator overloading
    """

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Compute product kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n1).
        X2 : Array, optional
            Input data, shape (nvars, n2). If None, uses X1.

        Returns
        -------
        K : Array
            Product kernel matrix, shape (n1, n2) or (n1, n1).
        """
        K1 = self._kernel1(X1, X2)
        K2 = self._kernel2(X1, X2)
        return K1 * K2

    def diag(self, X1: Array) -> Array:
        """
        Compute diagonal of product kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n).

        Returns
        -------
        diag : Array
            Diagonal elements, shape (n,).
        """
        diag1 = self._kernel1.diag(X1)
        diag2 = self._kernel2.diag(X1)
        return diag1 * diag2

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Compute Jacobian of product kernel w.r.t. inputs.

        Uses product rule: d(K1 * K2)/dx = dK1/dx * K2 + K1 * dK2/dx

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n1).
        X2 : Array
            Input data, shape (nvars, n2).

        Returns
        -------
        jac : Array
            Jacobian, shape (n1, n2, nvars).

        Raises
        ------
        NotImplementedError
            If either kernel doesn't support Jacobians.
        """
        if not (hasattr(self._kernel1, "jacobian") and hasattr(self._kernel2, "jacobian")):
            raise NotImplementedError(
                "Both kernels must implement jacobian() for ProductKernel Jacobian"
            )

        K1 = self._kernel1(X1, X2)
        K2 = self._kernel2(X1, X2)
        dK1 = self._kernel1.jacobian(X1, X2)
        dK2 = self._kernel2.jacobian(X1, X2)

        # Product rule: dK1 * K2 + K1 * dK2
        # K1, K2 have shape (n1, n2)
        # dK1, dK2 have shape (n1, n2, nvars)
        # Need to broadcast K to match dK shape
        return dK1 * K2[..., None] + K1[..., None] * dK2

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Jacobian of product kernel w.r.t. hyperparameters.

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).

        Returns
        -------
        jac : Array
            Jacobian, shape (n, n, nparams1 + nparams2).

        Raises
        ------
        NotImplementedError
            If either kernel doesn't support parameter Jacobians.
        """
        if not (
            hasattr(self._kernel1, "jacobian_wrt_params")
            and hasattr(self._kernel2, "jacobian_wrt_params")
        ):
            raise NotImplementedError(
                "Both kernels must implement jacobian_wrt_params() "
                "for ProductKernel parameter Jacobian"
            )

        K1 = self._kernel1(samples, samples)
        K2 = self._kernel2(samples, samples)
        dK1 = self._kernel1.jacobian_wrt_params(samples)
        dK2 = self._kernel2.jacobian_wrt_params(samples)

        # Product rule: [dK1 * K2, K1 * dK2]
        # K1, K2 have shape (n, n)
        # dK1 has shape (n, n, nparams1)
        # dK2 has shape (n, n, nparams2)

        nparams1 = dK1.shape[2]
        nparams2 = dK2.shape[2]

        # First part: dK1 * K2
        jac1 = dK1 * K2[..., None]

        # Second part: K1 * dK2
        jac2 = K1[..., None] * dK2

        # Concatenate along parameter dimension
        return self._bkd.concatenate([jac1, jac2], axis=2)

    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array:
        """
        Compute HVP of product kernel w.r.t. first argument.

        Uses product rule for Hessian:
        H[K1 * K2]·V = H[K1]·V * K2 + 2·(∇K1 ⊗ ∇K2)·V + K1 * H[K2]·V

        Where (∇K1 ⊗ ∇K2)·V is the mixed term from the product rule.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, shape (nvars, n2)
            Second set of points
        direction : Array, shape (nvars,)
            Direction vector for HVP

        Returns
        -------
        hvp : Array, shape (n1, n2, nvars)
            H[K(X1, X2)]·V for product kernel

        Raises
        ------
        NotImplementedError
            If either kernel doesn't support hvp_wrt_x1
        """
        if not (isinstance(self._kernel1, KernelHasHVPProtocol) and
                isinstance(self._kernel2, KernelHasHVPProtocol)):
            raise NotImplementedError(
                "Both kernels must implement hvp_wrt_x1() for ProductKernel HVP"
            )

        # Evaluate kernels
        K1 = self._kernel1(X1, X2)  # (n1, n2)
        K2 = self._kernel2(X1, X2)  # (n1, n2)

        # Get Jacobians (gradients)
        if not (hasattr(self._kernel1, "jacobian") and
                hasattr(self._kernel2, "jacobian")):
            raise NotImplementedError(
                "Both kernels must implement jacobian() for ProductKernel HVP"
            )

        dK1 = self._kernel1.jacobian(X1, X2)  # (n1, n2, nvars)
        dK2 = self._kernel2.jacobian(X1, X2)  # (n1, n2, nvars)

        # Get HVPs from both kernels
        HK1_V = self._kernel1.hvp_wrt_x1(X1, X2, direction)  # (n1, n2, nvars)
        HK2_V = self._kernel2.hvp_wrt_x1(X1, X2, direction)  # (n1, n2, nvars)

        # Product rule for Hessian:
        # H[K1·K2]·V = H[K1]·V · K2 + K1 · H[K2]·V + 2·(∇K1·V)·∇K2 + 2·∇K1·(∇K2·V)
        # Simplified: H[K1·K2]·V = (H[K1]·V)·K2 + K1·(H[K2]·V) + 2·(∇K1^T·V)·∇K2

        # Term 1: H[K1]·V * K2
        term1 = HK1_V * K2[:, :, None]  # (n1, n2, nvars)

        # Term 2: K1 * H[K2]·V
        term2 = K1[:, :, None] * HK2_V  # (n1, n2, nvars)

        # Term 3: Mixed derivative term
        # (∇K1)^T · V gives scalar per point pair: (n1, n2, nvars) · (nvars,) -> (n1, n2)
        dK1_dot_V = self._bkd.einsum('ijk,k->ij', dK1, direction)  # (n1, n2)
        # Then multiply by ∇K2: (n1, n2) * (n1, n2, nvars) -> (n1, n2, nvars)
        term3 = dK1_dot_V[:, :, None] * dK2  # (n1, n2, nvars)

        # Term 4: Symmetric mixed term
        dK2_dot_V = self._bkd.einsum('ijk,k->ij', dK2, direction)  # (n1, n2)
        term4 = dK2_dot_V[:, :, None] * dK1  # (n1, n2, nvars)

        return term1 + term2 + term3 + term4


class SumKernel(CompositionKernel):
    """
    Sum of two kernels (element-wise addition).

    K(X1, X2) = K1(X1, X2) + K2(X1, X2)

    Parameters
    ----------
    kernel1 : Kernel
        First kernel.
    kernel2 : Kernel
        Second kernel.

    Examples
    --------
    >>> from pyapprox.typing.surrogates.kernels import MaternKernel, WhiteKernel
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> matern = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> white = WhiteKernel(0.1, (0.01, 1.0), bkd)
    >>> sum_kernel = matern + white  # Uses operator overloading
    """

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Compute sum kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n1).
        X2 : Array, optional
            Input data, shape (nvars, n2). If None, uses X1.

        Returns
        -------
        K : Array
            Sum kernel matrix, shape (n1, n2) or (n1, n1).
        """
        K1 = self._kernel1(X1, X2)
        K2 = self._kernel2(X1, X2)
        return K1 + K2

    def diag(self, X1: Array) -> Array:
        """
        Compute diagonal of sum kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n).

        Returns
        -------
        diag : Array
            Diagonal elements, shape (n,).
        """
        diag1 = self._kernel1.diag(X1)
        diag2 = self._kernel2.diag(X1)
        return diag1 + diag2

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Compute Jacobian of sum kernel w.r.t. inputs.

        Uses sum rule: d(K1 + K2)/dx = dK1/dx + dK2/dx

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n1).
        X2 : Array
            Input data, shape (nvars, n2).

        Returns
        -------
        jac : Array
            Jacobian, shape (n1, n2, nvars).

        Raises
        ------
        NotImplementedError
            If either kernel doesn't support Jacobians.
        """
        if not (hasattr(self._kernel1, "jacobian") and hasattr(self._kernel2, "jacobian")):
            raise NotImplementedError(
                "Both kernels must implement jacobian() for SumKernel Jacobian"
            )

        dK1 = self._kernel1.jacobian(X1, X2)
        dK2 = self._kernel2.jacobian(X1, X2)

        # Sum rule: dK1 + dK2
        return dK1 + dK2

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Jacobian of sum kernel w.r.t. hyperparameters.

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).

        Returns
        -------
        jac : Array
            Jacobian, shape (n, n, nparams1 + nparams2).

        Raises
        ------
        NotImplementedError
            If either kernel doesn't support parameter Jacobians.
        """
        if not (
            hasattr(self._kernel1, "jacobian_wrt_params")
            and hasattr(self._kernel2, "jacobian_wrt_params")
        ):
            raise NotImplementedError(
                "Both kernels must implement jacobian_wrt_params() "
                "for SumKernel parameter Jacobian"
            )

        dK1 = self._kernel1.jacobian_wrt_params(samples)
        dK2 = self._kernel2.jacobian_wrt_params(samples)

        # Sum rule: [dK1, dK2]
        # Concatenate along parameter dimension
        return self._bkd.concatenate([dK1, dK2], axis=2)

    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array:
        """
        Compute HVP of sum kernel w.r.t. first argument.

        Uses sum rule for Hessian:
        H[K1 + K2]·V = H[K1]·V + H[K2]·V

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, shape (nvars, n2)
            Second set of points
        direction : Array, shape (nvars,)
            Direction vector for HVP

        Returns
        -------
        hvp : Array, shape (n1, n2, nvars)
            H[K(X1, X2)]·V for sum kernel

        Raises
        ------
        NotImplementedError
            If either kernel doesn't support hvp_wrt_x1
        """
        if not (isinstance(self._kernel1, KernelHasHVPProtocol) and
                isinstance(self._kernel2, KernelHasHVPProtocol)):
            raise NotImplementedError(
                "Both kernels must implement hvp_wrt_x1() for SumKernel HVP"
            )

        # Sum rule: just add the HVPs
        HK1_V = self._kernel1.hvp_wrt_x1(X1, X2, direction)
        HK2_V = self._kernel2.hvp_wrt_x1(X1, X2, direction)

        return HK1_V + HK2_V
