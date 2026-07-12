"""Kernel base class and composition kernels.

Defines the abstract Kernel base class and composition operations (product,
sum, separable product) for building complex kernels from simple ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.validation import validate_backend
from pyapprox.util.hyperparameter import HyperParameterList


class _HasInputJacobianMethod(Protocol, Generic[Array]):
    """Structural type for the curried input-jacobian callable (typing
    only; never used for isinstance capability checks)."""

    def jacobian(self, X1: Array, X2: Array) -> Array: ...


class _HasInputHVPMethod(Protocol, Generic[Array]):
    """Structural type for the curried input-hvp callable (typing only)."""

    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array: ...


@dataclass(frozen=True)
class KernelInputJacobian(Generic[Array]):
    """Picklable curried form of ``kernel.jacobian(X1, X2)`` with X2
    fixed, for use as a Derivatives bundle field."""

    kernel: _HasInputJacobianMethod[Array]
    X2: Array

    def __call__(self, X1: Array) -> Array:
        return self.kernel.jacobian(X1, self.X2)


@dataclass(frozen=True)
class KernelInputHVP(Generic[Array]):
    """Picklable curried form of ``kernel.hvp_wrt_x1(X1, X2, direction)``
    with X2 fixed, for use as a Derivatives bundle field."""

    kernel: _HasInputHVPMethod[Array]
    X2: Array

    def __call__(self, X1: Array, direction: Array) -> Array:
        return self.kernel.hvp_wrt_x1(X1, self.X2, direction)


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
    _hyp_list : HyperParameterList[Array]
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

    @abstractmethod
    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return the list of hyperparameters associated with the kernel.

        Returns
        -------
        hyp_list : HyperParameterList[Array]
            List of hyperparameters.
        """
        raise NotImplementedError()

    @abstractmethod
    def nvars(self) -> int:
        """
        Return the number of input variables (dimensionality).

        Returns
        -------
        int
            Number of input dimensions.
        """
        raise NotImplementedError()

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
    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
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

    def param_derivatives(self) -> Derivatives[Array]:
        """Derivatives of ``theta -> K(samples; theta)`` forms.

        Shape conventions are documented on
        ``KernelProtocol.param_derivatives``. The base class has no
        analytic parameter derivatives; capable kernels override this.
        """
        empty: Derivatives[Array] = Derivatives.none()
        return empty

    def input_derivatives(self, X2: Array) -> Derivatives[Array]:
        """Derivatives of ``x1 -> k(x1, X2)``; fields close over ``X2``.

        Shape conventions are documented on
        ``KernelProtocol.input_derivatives``. The base class has no
        analytic input derivatives; capable kernels override this.
        """
        empty: Derivatives[Array] = Derivatives.none()
        return empty

    def __mul__(self, other: "Kernel[Array]") -> "Kernel[Array]":
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
        return ProductKernel(self, other)

    def __add__(self, other: "Kernel[Array]") -> "Kernel[Array]":
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
            str(self.hyp_list()),
            self._bkd.__class__.__name__,
        )


class CompositionKernel(Kernel[Array], Generic[Array]):
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
    _hyp_list : HyperParameterList[Array]
        Combined hyperparameter list from both kernels.

    Notes
    -----
    **AND Logic Convention**: a composition only declares a derivative
    capability in its bundles if ALL component kernels declare it. The
    component capabilities are captured once at construction from the
    children's ``param_derivatives()`` bundles; ``param_derivatives()`` on
    the composition branches on those captured fields.
    """

    def __init__(self, kernel1: Kernel[Array], kernel2: Kernel[Array]):
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

        # capture component capabilities ONCE (construction-time branching)
        d1 = kernel1.param_derivatives()
        d2 = kernel2.param_derivatives()
        self._k1_param_jac = d1.jacobian
        self._k2_param_jac = d2.jacobian
        self._k1_param_hvp = d1.hvp
        self._k2_param_hvp = d2.hvp

    def param_derivatives(self) -> Derivatives[Array]:
        """AND logic: capability only when BOTH components declare it."""
        if self._k1_param_jac is None or self._k2_param_jac is None:
            empty: Derivatives[Array] = Derivatives.none()
            return empty
        if self._k1_param_hvp is not None and self._k2_param_hvp is not None:
            return Derivatives(
                jacobian=self._jacobian_wrt_params,
                hvp=self._hvp_wrt_params,
            )
        return Derivatives.first_order(jacobian=self._jacobian_wrt_params)

    def input_derivatives(self, X2: Array) -> Derivatives[Array]:
        """AND logic over the components' input capabilities."""
        d1 = self._kernel1.input_derivatives(X2)
        d2 = self._kernel2.input_derivatives(X2)
        if d1.jacobian is None or d2.jacobian is None:
            empty: Derivatives[Array] = Derivatives.none()
            return empty
        jacobian = KernelInputJacobian(self, X2)
        if d1.hvp is None or d2.hvp is None:
            return Derivatives.first_order(jacobian=jacobian)
        return Derivatives(jacobian=jacobian, hvp=KernelInputHVP(self, X2))

    @abstractmethod
    def _jacobian_wrt_params(self, samples: Array) -> Array:
        """Composition rule for the parameter jacobian; only reachable
        when ``param_derivatives()`` declared the capability."""
        raise NotImplementedError()

    @abstractmethod
    def _hvp_wrt_params(self, samples: Array, direction: Array) -> Array:
        """Composition rule for the parameter hvp; only reachable when
        ``param_derivatives()`` declared the capability."""
        raise NotImplementedError()

    @abstractmethod
    def jacobian(self, X1: Array, X2: Array) -> Array:
        """Composition rule for the input jacobian."""
        raise NotImplementedError()

    @abstractmethod
    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array:
        """Composition rule for the input hvp."""
        raise NotImplementedError()

    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return the combined hyperparameter list.

        Returns
        -------
        hyp_list : HyperParameterList[Array]
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


class ProductKernel(CompositionKernel[Array]):
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
    >>> from pyapprox.surrogates.kernels import MaternKernel, ConstantKernel
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> matern = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> constant = ConstantKernel(1.0, (0.1, 10.0), bkd)
    >>> product = matern * constant  # Uses operator overloading
    """

    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
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
        k1_jac = self._kernel1.input_derivatives(X2).jacobian
        k2_jac = self._kernel2.input_derivatives(X2).jacobian
        if k1_jac is None or k2_jac is None:
            raise NotImplementedError(
                "Both kernels must provide an input jacobian for "
                "ProductKernel Jacobian"
            )

        K1 = self._kernel1(X1, X2)
        K2 = self._kernel2(X1, X2)
        dK1 = k1_jac(X1)
        dK2 = k2_jac(X1)

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
        k1_jac = self._k1_param_jac
        k2_jac = self._k2_param_jac
        if k1_jac is None or k2_jac is None:
            raise RuntimeError(
                "_jacobian_wrt_params requires both component kernels to "
                "declare a parameter jacobian"
            )

        K1 = self._kernel1(samples, samples)
        K2 = self._kernel2(samples, samples)
        dK1 = k1_jac(samples)
        dK2 = k2_jac(samples)

        # Product rule: [dK1 * K2, K1 * dK2]
        # K1, K2 have shape (n, n)
        # dK1 has shape (n, n, nparams1)
        # dK2 has shape (n, n, nparams2)

        # First part: dK1 * K2
        jac1 = dK1 * K2[..., None]

        # Second part: K1 * dK2
        jac2 = K1[..., None] * dK2

        # Concatenate along parameter dimension
        return self._bkd.concatenate([jac1, jac2], axis=2)

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
        k1_jac = self._k1_param_jac
        k2_jac = self._k2_param_jac
        k1_hvp = self._k1_param_hvp
        k2_hvp = self._k2_param_hvp
        if k1_jac is None or k2_jac is None or k1_hvp is None or k2_hvp is None:
            raise RuntimeError(
                "_hvp_wrt_params requires both component kernels to "
                "declare parameter jacobian and hvp"
            )

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

        dK1 = k1_jac(samples)  # (n, n, p1)
        dK2 = k2_jac(samples)  # (n, n, p2)

        HK1_v = k1_hvp(samples, v1)  # (n, n, p1)
        HK2_v = k2_hvp(samples, v2)  # (n, n, p2)

        hvp = self._bkd.zeros((n, n, p1 + p2))

        # Block 1: K1 parameters (i in K1)
        # hvp[:, :, i] = Σ_{j in K1} (∂²K1/∂θ_i∂θ_j * K2) * v1[j]
        #              + Σ_{j in K2} (∂K1/∂θ_i * ∂K2/∂θ_j) * v2[j]
        # = (Σ_{j in K1} ∂²K1/∂θ_i∂θ_j * v1[j]) * K2 + (∂K1/∂θ_i) * (Σ_{j in K2}
        # ∂K2/∂θ_j * v2[j])
        # = HK1_v[:, :, i] * K2 + dK1[:, :, i] * (dK2 · v2)

        # Compute dK2 · v2: shape (n, n)
        dK2_dot_v2 = self._bkd.einsum("ijk,k->ij", dK2, v2)

        for i in range(p1):
            hvp[:, :, i] = HK1_v[:, :, i] * K2 + dK1[:, :, i] * dK2_dot_v2

        # Block 2: K2 parameters (i in K2)
        # hvp[:, :, p1+i] = Σ_{j in K1} (∂K1/∂θ_j * ∂K2/∂θ_i) * v1[j]
        #                 + Σ_{j in K2} (K1 * ∂²K2/∂θ_i∂θ_j) * v2[j]
        # = (Σ_{j in K1} ∂K1/∂θ_j * v1[j]) * ∂K2/∂θ_i + K1 * (Σ_{j in K2} ∂²K2/∂θ_i∂θ_j
        # * v2[j])
        # = (dK1 · v1) * dK2[:, :, i] + K1 * HK2_v[:, :, i]

        # Compute dK1 · v1: shape (n, n)
        dK1_dot_v1 = self._bkd.einsum("ijk,k->ij", dK1, v1)

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
        d1 = self._kernel1.input_derivatives(X2)
        d2 = self._kernel2.input_derivatives(X2)
        k1_jac, k1_hvp = d1.jacobian, d1.hvp
        k2_jac, k2_hvp = d2.jacobian, d2.hvp
        if k1_hvp is None or k2_hvp is None:
            raise NotImplementedError(
                "Both kernels must provide an input hvp for ProductKernel HVP"
            )
        if k1_jac is None or k2_jac is None:
            raise NotImplementedError(
                "Both kernels must provide an input jacobian for "
                "ProductKernel HVP"
            )

        # Evaluate kernels
        K1 = self._kernel1(X1, X2)  # (n1, n2)
        K2 = self._kernel2(X1, X2)  # (n1, n2)

        dK1 = k1_jac(X1)  # (n1, n2, nvars)
        dK2 = k2_jac(X1)  # (n1, n2, nvars)

        # Get HVPs from both kernels
        HK1_V = k1_hvp(X1, direction)  # (n1, n2, nvars)
        HK2_V = k2_hvp(X1, direction)  # (n1, n2, nvars)

        # Product rule for Hessian:
        # H[K1·K2]·V = H[K1]·V · K2 + K1 · H[K2]·V + 2·(∇K1·V)·∇K2 + 2·∇K1·(∇K2·V)
        # Simplified: H[K1·K2]·V = (H[K1]·V)·K2 + K1·(H[K2]·V) + 2·(∇K1^T·V)·∇K2

        # Term 1: H[K1]·V * K2
        term1 = HK1_V * K2[:, :, None]  # (n1, n2, nvars)

        # Term 2: K1 * H[K2]·V
        term2 = K1[:, :, None] * HK2_V  # (n1, n2, nvars)

        # Term 3: Mixed derivative term
        # (∇K1)^T · V gives scalar per point pair: (n1, n2, nvars) · (nvars,) -> (n1,
        # n2)
        dK1_dot_V = self._bkd.einsum("ijk,k->ij", dK1, direction)  # (n1, n2)
        # Then multiply by ∇K2: (n1, n2) * (n1, n2, nvars) -> (n1, n2, nvars)
        term3 = dK1_dot_V[:, :, None] * dK2  # (n1, n2, nvars)

        # Term 4: Symmetric mixed term
        dK2_dot_V = self._bkd.einsum("ijk,k->ij", dK2, direction)  # (n1, n2)
        term4 = dK2_dot_V[:, :, None] * dK1  # (n1, n2, nvars)

        return term1 + term2 + term3 + term4


class SumKernel(CompositionKernel[Array]):
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
    >>> from pyapprox.surrogates.kernels import MaternKernel, WhiteKernel
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> matern = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> white = WhiteKernel(0.1, (0.01, 1.0), bkd)
    >>> sum_kernel = matern + white  # Uses operator overloading
    """

    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
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
        k1_jac = self._kernel1.input_derivatives(X2).jacobian
        k2_jac = self._kernel2.input_derivatives(X2).jacobian
        if k1_jac is None or k2_jac is None:
            raise NotImplementedError(
                "Both kernels must provide an input jacobian for "
                "SumKernel Jacobian"
            )

        dK1 = k1_jac(X1)
        dK2 = k2_jac(X1)

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
        k1_jac = self._k1_param_jac
        k2_jac = self._k2_param_jac
        if k1_jac is None or k2_jac is None:
            raise RuntimeError(
                "_jacobian_wrt_params requires both component kernels to "
                "declare a parameter jacobian"
            )

        dK1 = k1_jac(samples)
        dK2 = k2_jac(samples)

        # Sum rule: [dK1, dK2]
        # Concatenate along parameter dimension
        return self._bkd.concatenate([dK1, dK2], axis=2)

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
        k1_hvp = self._k1_param_hvp
        k2_hvp = self._k2_param_hvp
        if k1_hvp is None or k2_hvp is None:
            raise RuntimeError(
                "_hvp_wrt_params requires both component kernels to "
                "declare a parameter hvp"
            )

        # Get dimensions
        p1 = self._kernel1.hyp_list().nactive_params()

        # Split direction vector
        v1 = direction[:p1]  # Direction for K1 params
        v2 = direction[p1:]  # Direction for K2 params

        # Get HVPs from both kernels
        HK1_v = k1_hvp(samples, v1)  # (n, n, p1)
        HK2_v = k2_hvp(samples, v2)  # (n, n, p2)

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
        k1_hvp = self._kernel1.input_derivatives(X2).hvp
        k2_hvp = self._kernel2.input_derivatives(X2).hvp
        if k1_hvp is None or k2_hvp is None:
            raise NotImplementedError(
                "Both kernels must provide an input hvp for SumKernel HVP"
            )

        # Sum rule: just add the HVPs
        HK1_V = k1_hvp(X1, direction)
        HK2_V = k2_hvp(X1, direction)

        return HK1_V + HK2_V


class SeparableProductKernel(Kernel[Array], Generic[Array]):
    """
    Product kernel constructed from 1D kernels, one per dimension.

    For d-dimensional inputs, given 1D kernels k₁, k₂, ..., k_d:

        k(x, y) = ∏ᵢ kᵢ(xᵢ, yᵢ)

    where kᵢ operates only on dimension i.

    This is the correct separable kernel structure required for
    GP sensitivity analysis with the integral formulas in
    SeparableKernelIntegralCalculator.

    Parameters
    ----------
    kernels_1d : List[Kernel]
        List of 1D kernels, one per dimension.
        kernels_1d[i] is applied to dimension i.
        Each kernel must have nvars=1.
    bkd : Backend[Array]
        Backend for array operations.

    Examples
    --------
    >>> from pyapprox.surrogates.kernels.matern import SquaredExponentialKernel
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> k1 = SquaredExponentialKernel([1.0], (0.01, 100.0), nvars=1, bkd=bkd)
    >>> k2 = SquaredExponentialKernel([2.0], (0.01, 100.0), nvars=1, bkd=bkd)
    >>> kernel = SeparableProductKernel([k1, k2], bkd=bkd)
    >>> # kernel(x, y) = k1(x[0], y[0]) * k2(x[1], y[1])
    """

    def __init__(
        self,
        kernels_1d: list[Kernel[Array]],
        bkd: Backend[Array],
    ):
        super().__init__(bkd)
        self._kernels_1d = kernels_1d
        self._nvars = len(kernels_1d)

        # Validate each kernel is 1D
        for i, k in enumerate(kernels_1d):
            if k.nvars() != 1:
                raise ValueError(f"Kernel {i} must have nvars=1, got {k.nvars()}")

        # Combine hyperparameter lists from all kernels
        hyp_lists = [k.hyp_list() for k in kernels_1d]
        combined = hyp_lists[0]
        for hl in hyp_lists[1:]:
            combined = combined + hl
        self._hyp_list = combined

    def hyp_list(self) -> HyperParameterList[Array]:
        """Return the combined hyperparameter list."""
        return self._hyp_list

    def nvars(self) -> int:
        """Return the number of input dimensions."""
        return self._nvars

    def get_kernel_1d(self, dim: int) -> Kernel[Array]:
        """
        Get the 1D kernel for a specific dimension.

        Parameters
        ----------
        dim : int
            Dimension index.

        Returns
        -------
        Kernel
            The 1D kernel for dimension dim.
        """
        return self._kernels_1d[dim]

    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
        """
        Evaluate separable product kernel.

        Parameters
        ----------
        X1 : Array
            First set of points, shape (nvars, n1).
        X2 : Array
            Second set of points, shape (nvars, n2).
            If None, uses X1.

        Returns
        -------
        Array
            Kernel matrix, shape (n1, n2).
        """
        if X2 is None:
            X2 = X1

        n1 = X1.shape[1]
        n2 = X2.shape[1]

        # Start with ones
        K = self._bkd.ones((n1, n2))

        # Multiply by each 1D kernel
        for dim, kernel_1d in enumerate(self._kernels_1d):
            # Extract dimension dim from each point set
            X1_dim = self._bkd.reshape(X1[dim, :], (1, -1))  # (1, n1)
            X2_dim = self._bkd.reshape(X2[dim, :], (1, -1))  # (1, n2)

            # Evaluate 1D kernel
            K_dim = kernel_1d(X1_dim, X2_dim)  # (n1, n2)

            # Multiply into product
            K = K * K_dim

        return K

    def diag(self, X1: Array) -> Array:
        """
        Compute diagonal of separable product kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n).

        Returns
        -------
        diag : Array
            Diagonal elements, shape (n,).
        """
        n = X1.shape[1]
        diag = self._bkd.ones((n,))

        for dim, kernel_1d in enumerate(self._kernels_1d):
            X1_dim = self._bkd.reshape(X1[dim, :], (1, -1))
            diag_dim = kernel_1d.diag(X1_dim)
            diag = diag * diag_dim

        return diag

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Compute Jacobian of separable product kernel w.r.t. first input.

        Uses product rule across dimensions.

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
        """
        n1 = X1.shape[1]
        n2 = X2.shape[1]
        nvars = self._nvars

        # Precompute all 1D kernel values and jacobians
        K_1d = []
        dK_1d = []
        for dim, kernel_1d in enumerate(self._kernels_1d):
            X1_dim = self._bkd.reshape(X1[dim, :], (1, -1))
            X2_dim = self._bkd.reshape(X2[dim, :], (1, -1))
            K_1d.append(kernel_1d(X1_dim, X2_dim))
            jac_1d = kernel_1d.input_derivatives(X2_dim).jacobian
            if jac_1d is None:
                raise NotImplementedError(
                    f"Kernel for dimension {dim} must provide an input "
                    "jacobian"
                )
            # jacobian returns shape (n1, n2, 1) for 1D kernel
            dK_1d.append(jac_1d(X1_dim)[:, :, 0])  # (n1, n2)

        # Build jacobian: for each dimension d, derivative is
        # dK/dx_d = (∏_{i≠d} K_i) * dK_d/dx_d
        jac = self._bkd.zeros((n1, n2, nvars))
        for d in range(nvars):
            # Product of all K_i except dimension d
            prod = self._bkd.ones((n1, n2))
            for i in range(nvars):
                if i != d:
                    prod = prod * K_1d[i]
            jac[:, :, d] = prod * dK_1d[d]

        return jac

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Jacobian of separable product kernel w.r.t. hyperparameters.

        For K = ∏ᵢ Kᵢ, the derivative with respect to parameters of kernel j is:
        ∂K/∂θⱼ = (∏ᵢ≠ⱼ Kᵢ) * ∂Kⱼ/∂θⱼ

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).

        Returns
        -------
        jac : Array
            Jacobian, shape (n, n, nactive_params).
        """
        n = samples.shape[1]
        nvars = self._nvars

        # Precompute all 1D kernel matrices
        K_1d = []
        for dim, kernel_1d in enumerate(self._kernels_1d):
            X_dim = self._bkd.reshape(samples[dim, :], (1, -1))
            K_1d.append(kernel_1d(X_dim, X_dim))

        # Compute product of all kernels (needed for efficiency)
        K_all = self._bkd.ones((n, n))
        for K_dim in K_1d:
            K_all = K_all * K_dim

        # Build list of jacobians for each 1D kernel
        jac_parts = []
        for dim, kernel_1d in enumerate(self._kernels_1d):
            param_jac_1d = kernel_1d.param_derivatives().jacobian
            if param_jac_1d is None:
                raise NotImplementedError(
                    f"Kernel for dimension {dim} must provide a parameter "
                    "jacobian"
                )
            X_dim = self._bkd.reshape(samples[dim, :], (1, -1))
            dK_dim = param_jac_1d(X_dim)  # (n, n, nparams_dim)

            # Product of all K_i except dimension dim
            # K_all / K_dim avoids recomputing product, but must handle zeros
            # Use explicit product instead for numerical stability
            prod_except_dim = self._bkd.ones((n, n))
            for i in range(nvars):
                if i != dim:
                    prod_except_dim = prod_except_dim * K_1d[i]

            # Scale each parameter derivative by the product of other kernels
            nparams_dim = dK_dim.shape[2]
            for p in range(nparams_dim):
                jac_parts.append(prod_except_dim * dK_dim[:, :, p])

        # Stack all jacobians along the parameter dimension
        # Each element in jac_parts is (n, n), stack to get (n, n, total_nparams)
        jac = self._bkd.stack(jac_parts, axis=2)

        return jac

    def param_derivatives(self) -> Derivatives[Array]:
        """Capability requires every 1D kernel to declare a parameter
        jacobian (AND logic)."""
        for kernel_1d in self._kernels_1d:
            if kernel_1d.param_derivatives().jacobian is None:
                empty: Derivatives[Array] = Derivatives.none()
                return empty
        return Derivatives.first_order(jacobian=self.jacobian_wrt_params)

    def input_derivatives(self, X2: Array) -> Derivatives[Array]:
        """Capability requires every 1D kernel to declare an input
        jacobian (AND logic)."""
        for dim, kernel_1d in enumerate(self._kernels_1d):
            X2_dim = self._bkd.reshape(X2[dim, :], (1, -1))
            if kernel_1d.input_derivatives(X2_dim).jacobian is None:
                empty: Derivatives[Array] = Derivatives.none()
                return empty
        return Derivatives.first_order(
            jacobian=KernelInputJacobian(self, X2)
        )
