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

    Optional Methods
    ----------------
    This class uses dynamic method binding with AND logic for composition:

    - ``jacobian_wrt_params(samples)``: Available if BOTH kernels have it.
    - ``hvp_wrt_params(samples, direction)``: Available if BOTH kernels have it.

    Check availability with ``hasattr(kernel, 'jacobian_wrt_params')`` or
    ``hasattr(kernel, 'hvp_wrt_params')``.

    Notes
    -----
    **AND Logic Convention**: Composed objects only have a capability if ALL
    component objects have it. This ensures derivative methods are only available
    when they can be correctly computed for the entire composition.

    This pattern is implemented via ``_setup_derivative_methods()`` which checks
    each component kernel's capabilities and conditionally assigns public methods.
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

        # Conditionally add derivative methods based on component kernel support
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        """
        Conditionally add jacobian_wrt_params and hvp_wrt_params methods.

        Composition kernels only support these methods if BOTH component
        kernels support them. This ensures that derivative methods are only
        available when they can be correctly computed.
        """
        # Check if both kernels support jacobian_wrt_params
        has_jac_1 = hasattr(self._kernel1, 'jacobian_wrt_params')
        has_jac_2 = hasattr(self._kernel2, 'jacobian_wrt_params')

        if has_jac_1 and has_jac_2:
            # Both kernels support jacobian, add the method
            self.jacobian_wrt_params = self._jacobian_wrt_params
        # Otherwise, jacobian_wrt_params will not exist on this instance

        # Check if both kernels support hvp_wrt_params
        has_hvp_1 = hasattr(self._kernel1, 'hvp_wrt_params')
        has_hvp_2 = hasattr(self._kernel2, 'hvp_wrt_params')

        if has_hvp_1 and has_hvp_2:
            # Both kernels support HVP, add the method
            self.hvp_wrt_params = self._hvp_wrt_params
        # Otherwise, hvp_wrt_params will not exist on this instance

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

    def _jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Jacobian of product kernel w.r.t. hyperparameters.

        This is a private method. The public jacobian_wrt_params() method is
        dynamically added during __init__ if both component kernels support it.

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).

        Returns
        -------
        jac : Array
            Jacobian, shape (n, n, nparams1 + nparams2).
        """
        # This method is only callable if both kernels support jacobian_wrt_params
        # (checked in _setup_derivative_methods)

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

    def hessian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Hessian of product kernel w.r.t. hyperparameters.

        For K = K1 * K2, using product rule:
        - ∂²K/∂θ_i∂θ_j = (∂²K1/∂θ_i∂θ_j) * K2  (both K1 params)
        - ∂²K/∂θ_i∂θ_j = (∂K1/∂θ_i) * (∂K2/∂θ_j)  (mixed: K1 and K2 params)
        - ∂²K/∂θ_i∂θ_j = K1 * (∂²K2/∂θ_i∂θ_j)  (both K2 params)

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).

        Returns
        -------
        hess : Array
            Hessian, shape (n, n, nparams, nparams).

        Raises
        ------
        NotImplementedError
            If either kernel doesn't support parameter Hessians.
        """
        if not (
            hasattr(self._kernel1, "hessian_wrt_params")
            and hasattr(self._kernel2, "hessian_wrt_params")
        ):
            raise NotImplementedError(
                "Both kernels must implement hessian_wrt_params() "
                "for ProductKernel parameter Hessian"
            )

        # Get kernel evaluations
        K1 = self._kernel1(samples, samples)  # (n, n)
        K2 = self._kernel2(samples, samples)  # (n, n)

        # Get Jacobians
        dK1 = self._kernel1.jacobian_wrt_params(samples)  # (n, n, p1)
        dK2 = self._kernel2.jacobian_wrt_params(samples)  # (n, n, p2)

        # Get Hessians
        H1 = self._kernel1.hessian_wrt_params(samples)  # (n, n, p1, p1)
        H2 = self._kernel2.hessian_wrt_params(samples)  # (n, n, p2, p2)

        n = samples.shape[1]
        p1 = dK1.shape[2]
        p2 = dK2.shape[2]
        p_total = p1 + p2

        # Initialize full Hessian
        hess = self._bkd.zeros((n, n, p_total, p_total))

        # Block 1: K1 params × K1 params -> H1 * K2
        hess[:, :, :p1, :p1] = H1 * K2[:, :, None, None]

        # Block 2: K2 params × K2 params -> K1 * H2
        hess[:, :, p1:, p1:] = K1[:, :, None, None] * H2

        # Block 3: K1 params × K2 params (mixed derivatives)
        # ∂²K/∂θ_i∂θ_j = (∂K1/∂θ_i) * (∂K2/∂θ_j)
        # dK1[:, :, i] * dK2[:, :, j]
        # Use einsum: 'ijk,ijl->ijkl' broadcasts properly
        cross_term = self._bkd.einsum('ijk,ijl->ijkl', dK1, dK2)
        hess[:, :, :p1, p1:] = cross_term

        # Block 4: K2 params × K1 params (symmetric)
        hess[:, :, p1:, :p1] = self._bkd.transpose(cross_term, (0, 1, 3, 2))

        return hess

    def _hvp_wrt_params(self, samples: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. hyperparameters for product kernel.

        For K = K1 * K2, using product rule:
        ∂²K/∂θ_i∂θ_j depends on whether i, j belong to K1 or K2:
        - If both in K1: (∂²K1/∂θ_i∂θ_j) * K2
        - If both in K2: K1 * (∂²K2/∂θ_i∂θ_j)
        - If i in K1, j in K2: (∂K1/∂θ_i) * (∂K2/∂θ_j)
        - If i in K2, j in K1: (∂K2/∂θ_i) * (∂K1/∂θ_j)

        HVP computes: Σ_j (∂²K/∂θ_i∂θ_j) * v[j] for each i.

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).
        direction : Array
            Direction vector, shape (p1 + p2,).

        Returns
        -------
        hvp : Array
            Hessian-vector product, shape (n, n, p1+p2).
        """
        # This method is only callable if both kernels support hvp_wrt_params
        # (checked in _setup_derivative_methods)

        # Get dimensions
        p1 = self._kernel1.hyp_list().nactive_params()
        p2 = self._kernel2.hyp_list().nactive_params()
        n = samples.shape[1]

        # Split direction vector
        v1 = direction[:p1]  # Direction for K1 params
        v2 = direction[p1:]  # Direction for K2 params

        # Evaluate kernels and their derivatives
        K1 = self._kernel1(samples, samples)  # (n, n)
        K2 = self._kernel2(samples, samples)  # (n, n)

        dK1 = self._kernel1.jacobian_wrt_params(samples)  # (n, n, p1)
        dK2 = self._kernel2.jacobian_wrt_params(samples)  # (n, n, p2)

        HK1_v = self._kernel1.hvp_wrt_params(samples, v1)  # (n, n, p1)
        HK2_v = self._kernel2.hvp_wrt_params(samples, v2)  # (n, n, p2)

        hvp = self._bkd.zeros((n, n, p1 + p2))

        # Block 1: K1 parameters (i in K1)
        # hvp[:, :, i] = Σ_{j in K1} (∂²K1/∂θ_i∂θ_j * K2) * v1[j]
        #              + Σ_{j in K2} (∂K1/∂θ_i * ∂K2/∂θ_j) * v2[j]
        # = (Σ_{j in K1} ∂²K1/∂θ_i∂θ_j * v1[j]) * K2 + (∂K1/∂θ_i) * (Σ_{j in K2} ∂K2/∂θ_j * v2[j])
        # = HK1_v[:, :, i] * K2 + dK1[:, :, i] * (dK2 · v2)

        # Compute dK2 · v2: shape (n, n)
        dK2_dot_v2 = self._bkd.einsum('ijk,k->ij', dK2, v2)

        for i in range(p1):
            hvp[:, :, i] = HK1_v[:, :, i] * K2 + dK1[:, :, i] * dK2_dot_v2

        # Block 2: K2 parameters (i in K2)
        # hvp[:, :, p1+i] = Σ_{j in K1} (∂K1/∂θ_j * ∂K2/∂θ_i) * v1[j]
        #                 + Σ_{j in K2} (K1 * ∂²K2/∂θ_i∂θ_j) * v2[j]
        # = (Σ_{j in K1} ∂K1/∂θ_j * v1[j]) * ∂K2/∂θ_i + K1 * (Σ_{j in K2} ∂²K2/∂θ_i∂θ_j * v2[j])
        # = (dK1 · v1) * dK2[:, :, i] + K1 * HK2_v[:, :, i]

        # Compute dK1 · v1: shape (n, n)
        dK1_dot_v1 = self._bkd.einsum('ijk,k->ij', dK1, v1)

        for i in range(p2):
            hvp[:, :, p1 + i] = dK1_dot_v1 * dK2[:, :, i] + K1 * HK2_v[:, :, i]

        return hvp

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

    def _jacobian_wrt_params(self, samples: Array) -> Array:
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

        """
        # This method is only callable if both kernels support jacobian_wrt_params
        # (checked in _setup_derivative_methods)

        dK1 = self._kernel1.jacobian_wrt_params(samples)
        dK2 = self._kernel2.jacobian_wrt_params(samples)

        # Sum rule: [dK1, dK2]
        # Concatenate along parameter dimension
        return self._bkd.concatenate([dK1, dK2], axis=2)

    def hessian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Hessian of sum kernel w.r.t. hyperparameters.

        For K = K1 + K2, using sum rule:
        - ∂²K/∂θ_i∂θ_j = ∂²K1/∂θ_i∂θ_j  (both K1 params)
        - ∂²K/∂θ_i∂θ_j = 0  (mixed: K1 and K2 params)
        - ∂²K/∂θ_i∂θ_j = ∂²K2/∂θ_i∂θ_j  (both K2 params)

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).

        Returns
        -------
        hess : Array
            Hessian, shape (n, n, nparams, nparams).

        Raises
        ------
        NotImplementedError
            If either kernel doesn't support parameter Hessians.
        """
        if not (
            hasattr(self._kernel1, "hessian_wrt_params")
            and hasattr(self._kernel2, "hessian_wrt_params")
        ):
            raise NotImplementedError(
                "Both kernels must implement hessian_wrt_params() "
                "for SumKernel parameter Hessian"
            )

        # Get Hessians from both kernels
        H1 = self._kernel1.hessian_wrt_params(samples)  # (n, n, p1, p1)
        H2 = self._kernel2.hessian_wrt_params(samples)  # (n, n, p2, p2)

        n = samples.shape[1]
        p1 = H1.shape[2]
        p2 = H2.shape[2]
        p_total = p1 + p2

        # Initialize full Hessian
        hess = self._bkd.zeros((n, n, p_total, p_total))

        # Block 1: K1 params × K1 params -> H1
        hess[:, :, :p1, :p1] = H1

        # Block 2: K2 params × K2 params -> H2
        hess[:, :, p1:, p1:] = H2

        # Blocks 3 & 4: Mixed derivatives are zero (no coupling between K1 and K2 params)

        return hess

    def _hvp_wrt_params(self, samples: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. hyperparameters for sum kernel.

        For K = K1 + K2, using sum rule:
        ∂²K/∂θ_i∂θ_j = ∂²K1/∂θ_i∂θ_j if both i,j belong to K1
        ∂²K/∂θ_i∂θ_j = ∂²K2/∂θ_i∂θ_j if both i,j belong to K2
        ∂²K/∂θ_i∂θ_j = 0 if i,j belong to different kernels

        HVP computes: Σ_j (∂²K/∂θ_i∂θ_j) * v[j] for each i.

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).
        direction : Array
            Direction vector, shape (p1 + p2,).

        Returns
        -------
        hvp : Array
            Hessian-vector product, shape (n, n, p1+p2).
        """
        # This method is only callable if both kernels support hvp_wrt_params
        # (checked in _setup_derivative_methods)

        # Get dimensions
        p1 = self._kernel1.hyp_list().nactive_params()
        p2 = self._kernel2.hyp_list().nactive_params()
        n = samples.shape[1]

        # Split direction vector
        v1 = direction[:p1]  # Direction for K1 params
        v2 = direction[p1:]  # Direction for K2 params

        # Get HVPs from both kernels
        HK1_v = self._kernel1.hvp_wrt_params(samples, v1)  # (n, n, p1)
        HK2_v = self._kernel2.hvp_wrt_params(samples, v2)  # (n, n, p2)

        # Sum rule: Just concatenate the HVPs
        # For i in K1: hvp[:, :, i] = HK1_v[:, :, i] (only K1 params contribute)
        # For i in K2: hvp[:, :, p1+i] = HK2_v[:, :, i] (only K2 params contribute)
        return self._bkd.concatenate([HK1_v, HK2_v], axis=2)

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
