"""
Scaling functions for spatially varying kernels.

This module provides polynomial scaling functions that can be used for:
- Spatially varying prior covariance (e.g., spatial_scaling * matern_kernel)
- Multi-level/autoregressive kernels with varying correlations between levels
- Non-stationary kernel construction
"""

from typing import (
    TYPE_CHECKING,
    Generic,
    List,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
)

if TYPE_CHECKING:
    from pyapprox.surrogates.kernels.composition import (
        ProductKernel,
        SumKernel,
    )


@runtime_checkable
class ScalingFunctionProtocol(Protocol, Generic[Array]):
    """
    Protocol for scaling functions used in spatially varying kernels.

    Scaling functions ρ(x) map input locations to scalar multipliers,
    enabling spatially varying kernel behavior.

    Common Use Cases
    ----------------
    1. Spatially varying prior covariance:
       k(x, x') = ρ(x) * k_base(x, x') * ρ(x')

    2. Multi-level autoregressive GPs:
       f_l(x) = ρ_{l-1}(x) * f_{l-1}(x) + δ_l(x)

    3. Non-stationary kernel construction via composition

    Methods
    -------
    __call__(X) : Evaluate scaling at points X
    jacobian(X) : Compute spatial derivative ∂ρ/∂x
    hyp_list() : Return hyperparameters
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        ...

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list."""
        ...

    def nvars(self) -> int:
        """Return the number of input variables."""
        ...

    def __call__(self, X: Array) -> Array:
        """
        Evaluate the scaling function.

        Parameters
        ----------
        X : Array
            Input points, shape (nvars, nsamples).

        Returns
        -------
        rho : Array
            Scaling values, shape (nsamples, 1).
        """
        ...

    def jacobian(self, X: Array) -> Array:
        """
        Compute spatial Jacobian ∂ρ/∂x.

        Parameters
        ----------
        X : Array
            Input points, shape (nvars, nsamples).

        Returns
        -------
        jac : Array
            Jacobian, shape (nsamples, nvars).
        """
        ...


class PolynomialScaling(Generic[Array]):
    """
    Polynomial scaling that can be used both as a function and as a kernel.

    As a scaling function: ρ(x) = c0 + c1*x1 + c2*x2 + ... + cd*xd
    As a kernel: K(x, x') = ρ(x) * ρ(x')

    This generalizes constant and linear scaling:
    - Degree 0 (1 coefficient): ρ(x) = c0 (replaces ConstantKernel)
    - Degree 1 (nvars+1 coefficients): ρ(x) = c0 + c1*x1 + ... + cd*xd

    Parameters
    ----------
    coefficients : List[float]
        Initial coefficient values in order: [c0, c1, ..., cd].
        Length determines degree:
        - 1 coefficient: constant (degree 0)
        - nvars+1 coefficients: linear (degree 1)
    bounds : Tuple[float, float]
        Bounds for all coefficients.
    bkd : Backend[Array]
        Backend for numerical computations.
    nvars : int
        Number of input variables (only needed for degree 0).
    fixed : bool, optional
        If True, coefficients are fixed. Default: False.

    Examples
    --------
    As a scaling function:
    >>> from pyapprox.surrogates.kernels.scalings import PolynomialScaling
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> scaling = PolynomialScaling([0.8], (0.1, 2.0), bkd, nvars=1)
    >>> X = bkd.array([[-1.0, 0.0, 1.0]])
    >>> rho = scaling.eval_scaling(X)  # [[0.8], [0.8], [0.8]]

    As a kernel (for composition):
    >>> kernel = scaling * MaternKernel(...)  # Works with operator overloading
    """

    def __init__(
        self,
        coefficients: List[float],
        bounds: Tuple[float, float],
        bkd: Backend[Array],
        nvars: int = None,
        fixed: bool = False,
    ):
        """Initialize the PolynomialScaling."""
        if not coefficients:
            raise ValueError("coefficients cannot be empty")

        self._bkd = bkd
        self._ncoeffs = len(coefficients)
        self._degree = self._ncoeffs - 1

        # Determine nvars
        if self._degree == 0:
            if nvars is None:
                raise ValueError("nvars required for constant scaling (degree 0)")
            self._nvars = nvars
        else:
            self._nvars = self._degree
            if nvars is not None and nvars != self._nvars:
                raise ValueError(
                    f"For degree {self._degree}, expected nvars={self._nvars}, "
                    f"got {nvars}"
                )

        # Create hyperparameters
        hyperparams = []
        for i, val in enumerate(coefficients):
            if i == 0:
                name = "intercept"
            else:
                name = f"coeff_{i}"
            hyp = HyperParameter(
                name=name,
                nparams=1,
                values=val,
                bounds=bounds,
                bkd=bkd,
                fixed=fixed,
            )
            hyperparams.append(hyp)

        self._hyp_list = HyperParameterList(hyperparams, bkd=bkd)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list."""
        return self._hyp_list

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def degree(self) -> int:
        """Return the polynomial degree."""
        return self._degree

    def eval_scaling(self, X: Array) -> Array:
        """
        Evaluate polynomial scaling function ρ(x).

        Parameters
        ----------
        X : Array
            Input points, shape (nvars, nsamples).

        Returns
        -------
        rho : Array
            Scaling values, shape (nsamples, 1).
        """
        hyps = self._hyp_list.hyperparameters()
        coeffs = [hyps[i].get_values()[0] for i in range(self._ncoeffs)]

        if self._degree == 0:
            n_samples = X.shape[1]
            # Use ones * value instead of full to preserve autograd graph
            # when coeffs[0] is a tensor scalar
            return self._bkd.ones((n_samples, 1)) * coeffs[0]
        else:
            intercept = coeffs[0]
            slopes = self._bkd.stack(coeffs[1:])
            result = intercept + X.T @ slopes
            return self._bkd.reshape(result, (-1, 1))

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Evaluate as a kernel: K(X1, X2) = ρ(X1) * ρ(X2)^T.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, optional, shape (nvars, n2)
            Second set of points. If None, uses X1.

        Returns
        -------
        K : Array, shape (n1, n2)
            Kernel matrix
        """
        if X2 is None:
            X2 = X1

        rho1 = self.eval_scaling(X1)  # (n1, 1)
        rho2 = self.eval_scaling(X2)  # (n2, 1)

        return rho1 @ rho2.T

    def diag(self, X1: Array) -> Array:
        """
        Compute diagonal of kernel matrix.

        Parameters
        ----------
        X1 : Array, shape (nvars, n)
            Input points

        Returns
        -------
        diag : Array, shape (n,)
            Diagonal elements ρ(X)²
        """
        rho = self.eval_scaling(X1)  # (n, 1)
        return self._bkd.reshape(rho**2, (-1,))

    def jacobian_scaling(self, X: Array) -> Array:
        """
        Compute spatial Jacobian of scaling function ∂ρ/∂x.

        For degree 0: ∂ρ/∂x = 0
        For degree 1: ∂ρ/∂xi = ci

        Parameters
        ----------
        X : Array
            Input points, shape (nvars, nsamples).

        Returns
        -------
        jac : Array
            Jacobian, shape (nsamples, nvars).
            jac[i, j] = ∂ρ/∂xj evaluated at X[:, i].
        """
        n_samples = X.shape[1]

        if self._degree == 0:
            return self._bkd.zeros((n_samples, self._nvars))
        else:
            hyps = self._hyp_list.hyperparameters()
            slopes = self._bkd.stack(
                [hyps[i + 1].get_values()[0] for i in range(self._nvars)]
            )
            jac = self._bkd.tile(slopes[None, :], (n_samples, 1))
            return jac

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Compute Jacobian of kernel K(X1, X2) = ρ(X1)·ρ(X2)^T w.r.t. X1.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, shape (nvars, n2)
            Second set of points

        Returns
        -------
        jac : Array, shape (n1, n2, nvars)
            Jacobian tensor (matches Matern convention)
        """
        rho2 = self.eval_scaling(X2)  # (n2, 1)
        drho1 = self.jacobian_scaling(X1)  # (n1, nvars)

        # jac[i, k, j] = drho1[i, j] * rho2[k, 0]
        # drho1 is (n1, nvars), rho2 is (n2, 1)
        # Broadcast: drho1[:, None, :] is (n1, 1, nvars)
        #            rho2[:, 0] is (n2,), need (1, n2, 1)
        jac = drho1[:, None, :] * rho2[:, 0][None, :, None]  # (n1, n2, nvars)
        return jac

    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array:
        """
        Compute HVP for PolynomialScaling kernel.

        For K(x, x') = ρ(x)·ρ(x'), the Hessian is zero for polynomial scaling
        (degree 0 or 1) since the second derivative is zero.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, shape (nvars, n2)
            Second set of points
        direction : Array, shape (nvars,)
            Direction vector

        Returns
        -------
        hvp : Array, shape (n1, n2, nvars)
            HVP (all zeros for polynomial kernels)
        """
        nvars = X1.shape[0]
        n1 = X1.shape[1]
        n2 = X2.shape[1]
        return self._bkd.zeros((n1, n2, nvars))

    def __mul__(self, other: "PolynomialScaling[Array]") -> "ProductKernel":
        """Multiply two kernels."""
        from pyapprox.surrogates.kernels.composition import ProductKernel

        return ProductKernel(self, other)

    def __add__(self, other: "PolynomialScaling[Array]") -> "SumKernel":
        """Add two kernels."""
        from pyapprox.surrogates.kernels.composition import SumKernel

        return SumKernel(self, other)

    def jacobian_wrt_params(self, X: Array) -> Array:
        """
        Compute Jacobian of scaling function ρ(X) w.r.t. hyperparameters.

        Parameters
        ----------
        X : Array
            Input points, shape (nvars, nsamples).

        Returns
        -------
        jac : Array
            Jacobian w.r.t. parameters, shape (nsamples, ncoeffs).
            jac[i, j] = ∂ρ(X[:, i])/∂θ_j
        """
        n_samples = X.shape[1]

        # Compute ∂ρ/∂θ
        if self._degree == 0:
            # Constant: ∂ρ/∂c0 = 1
            return self._bkd.ones((n_samples, 1))  # (n, 1)
        else:
            # Linear: ∂ρ/∂c0 = 1, ∂ρ/∂ci = xi
            ones_col = self._bkd.ones((n_samples, 1))
            return self._bkd.hstack([ones_col, X.T])  # (n, ncoeffs)

    def kernel_jacobian_wrt_params(self, X: Array) -> Array:
        """
        Compute Jacobian of kernel K(X, X) w.r.t. hyperparameters.

        For K(x, x') = ρ(x) * ρ(x'), the Jacobian is:
        ∂K/∂θ_i = (∂ρ(x)/∂θ_i) * ρ(x')^T + ρ(x) * (∂ρ(x')/∂θ_i)^T

        For symmetric evaluation K(X, X):
        ∂K/∂θ_i = (∂ρ(X)/∂θ_i) @ ρ(X)^T + ρ(X) @ (∂ρ(X)/∂θ_i)^T

        Parameters
        ----------
        X : Array
            Input points, shape (nvars, nsamples).

        Returns
        -------
        jac : Array
            Jacobian w.r.t. parameters, shape (n, n, ncoeffs).
        """
        n_samples = X.shape[1]
        rho = self.eval_scaling(X)  # (n, 1)
        drho_dtheta = self.jacobian_wrt_params(X)  # (n, ncoeffs)

        ncoeffs = drho_dtheta.shape[1]
        jac = self._bkd.zeros((n_samples, n_samples, ncoeffs))

        # For each parameter i: ∂K/∂θ_i = (∂ρ/∂θ_i) @ ρ^T + ρ @ (∂ρ/∂θ_i)^T
        for i in range(ncoeffs):
            drho_i = self._bkd.reshape(drho_dtheta[:, i], (-1, 1))  # (n, 1)
            jac[:, :, i] = drho_i @ rho.T + rho @ drho_i.T  # (n, n)

        return jac

    def hvp_wrt_params(self, X: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. hyperparameters.

        For polynomial scaling K(x, x') = ρ(x) * ρ(x'):
        ∂²K/∂θ_i∂θ_j = (∂ρ/∂θ_i) @ (∂ρ/∂θ_j)^T + (∂ρ/∂θ_j) @ (∂ρ/∂θ_i)^T

        HVP computes: Σ_j (∂²K/∂θ_i∂θ_j) * v[j] for each i.

        Parameters
        ----------
        X : Array
            Input points, shape (nvars, nsamples).
        direction : Array
            Direction vector, shape (ncoeffs,).

        Returns
        -------
        hvp : Array
            Hessian-vector product, shape (n, n, ncoeffs).
            hvp[:, :, i] = Σ_j (∂²K/∂θ_i∂θ_j) * v[j]
        """
        drho_dtheta = self.jacobian_wrt_params(X)  # (n, ncoeffs)

        # Vectorized HVP computation
        # For each i: hvp[:, :, i] = Σ_j H[:, :, i, j] * v[j]
        # where H[:, :, i, j] = (∂ρ/∂θ_i) @ (∂ρ/∂θ_j)^T + (∂ρ/∂θ_j) @ (∂ρ/∂θ_i)^T
        #
        # Expanding: hvp[:, :, i] = Σ_j [(∂ρ/∂θ_i) @ (∂ρ/∂θ_j)^T * v[j] + (∂ρ/∂θ_j) @
        # (∂ρ/∂θ_i)^T * v[j]]
        # = (∂ρ/∂θ_i) @ (Σ_j ∂ρ/∂θ_j * v[j])^T + (Σ_j ∂ρ/∂θ_j * v[j]) @ (∂ρ/∂θ_i)^T
        #                         = (∂ρ/∂θ_i) @ D^T + D @ (∂ρ/∂θ_i)^T
        # where D = drho_dtheta @ direction (shape: n_samples, 1)

        # Compute D = drho_dtheta @ direction
        D = drho_dtheta @ direction  # (n_samples,)

        # For each i: hvp[:, :, i] = (∂ρ/∂θ_i) @ D^T + D @ (∂ρ/∂θ_i)^T
        # drho_dtheta[:, i] is (n_samples,), D is (n_samples,)
        # Outer product: drho_dtheta[:, i:i+1] @ D^T is (n_samples, n_samples)

        # Vectorize using einsum:
        # hvp[:, :, i] = drho_dtheta[:, i] ⊗ D + D ⊗ drho_dtheta[:, i]
        hvp = self._bkd.einsum("ni,m->nmi", drho_dtheta, D) + self._bkd.einsum(
            "n,mi->nmi", D, drho_dtheta
        )
        # Shape: (n_samples, n_samples, ncoeffs)

        return hvp

    def __repr__(self) -> str:
        """String representation."""
        hyps = self._hyp_list.hyperparameters()
        coeffs = [hyps[i].get_values()[0] for i in range(self._ncoeffs)]

        if self._degree == 0:
            return f"PolynomialScaling(degree=0, value={coeffs[0]:.3f})"
        else:
            return (
                f"PolynomialScaling(degree={self._degree}, "
                f"coeffs={[f'{c:.3f}' for c in coeffs]})"
            )


class ScalingKernel(Generic[Array]):
    """
    Kernel wrapper for polynomial scaling functions.

    This wraps a PolynomialScaling to create a kernel:
    K(x, x') = ρ(x) * ρ(x')

    Commonly used for spatially varying priors, e.g.:
    kernel = ScalingKernel(scaling) * MaternKernel(...)

    Parameters
    ----------
    scaling : PolynomialScaling
        The scaling function to wrap.

    Examples
    --------
    >>> from pyapprox.surrogates.kernels.scalings import (
    ...     PolynomialScaling, ScalingKernel
    ... )
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Constant variance kernel (like ConstantKernel)
    >>> scaling = PolynomialScaling([0.8], (0.1, 2.0), bkd, nvars=2)
    >>> kernel = ScalingKernel(scaling)
    """

    def __init__(self, scaling: PolynomialScaling[Array]):
        """Initialize ScalingKernel."""
        self._scaling = scaling
        self._bkd = scaling.bkd()
        self._hyp_list = scaling.hyp_list()

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList:
        """Return hyperparameter list."""
        return self._hyp_list

    def nvars(self) -> int:
        """Return number of input variables."""
        return self._scaling.nvars()

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Compute kernel matrix K(X1, X2) = ρ(X1) * ρ(X2)^T.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, optional, shape (nvars, n2)
            Second set of points. If None, uses X1.

        Returns
        -------
        K : Array, shape (n1, n2)
            Kernel matrix
        """
        if X2 is None:
            X2 = X1

        rho1 = self._scaling(X1)  # (n1, 1)
        rho2 = self._scaling(X2)  # (n2, 1)

        # Outer product: (n1, 1) @ (1, n2) = (n1, n2)
        return rho1 @ rho2.T

    def diag(self, X1: Array) -> Array:
        """
        Compute diagonal K(X, X) = ρ(X)².

        Parameters
        ----------
        X1 : Array, shape (nvars, n)
            Input points

        Returns
        -------
        diag : Array, shape (n,)
            Diagonal elements
        """
        rho = self._scaling(X1)  # (n, 1)
        return self._bkd.reshape(rho**2, (-1,))

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Compute Jacobian ∂K/∂X1 where K(X1, X2) = ρ(X1) * ρ(X2)^T.

        Using product rule:
        ∂K/∂X1_j = (∂ρ(X1)/∂X1_j) * ρ(X2)^T

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, shape (nvars, n2)
            Second set of points

        Returns
        -------
        jac : Array, shape (n1, n2, nvars)
            Jacobian tensor (matches Matern convention)
        """
        rho2 = self._scaling(X2)  # (n2, 1)
        drho1 = self._scaling.jacobian(X1)  # (n1, nvars)

        # drho1[i, j] = ∂ρ(X1[:, i])/∂X1_j
        # Result: jac[i, k, j] = drho1[i, j] * rho2[k]
        # = ∂K(X1[:, i], X2[:, k])/∂X1_j

        X1.shape[0]
        X1.shape[1]
        X2.shape[1]

        # drho1 is (n1, nvars), rho2 is (n2, 1)
        # Want jac[i, k, j] = drho1[i, j] * rho2[k, 0]
        # = drho1[:, :, None] * rho2.T[None, :, None]
        jac = drho1[:, None, :] * rho2.T[None, :, None]  # (n1, n2, nvars)

        return jac

    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array:
        """
        Compute HVP for ScalingKernel.

        For K(x, x') = ρ(x)·ρ(x'), the Hessian is:
        H_jk[K(x, x')] = ∂²ρ(x)/∂x_j∂x_k · ρ(x')

        For PolynomialScaling:
        - Degree 0 (constant): H = 0
        - Degree 1 (linear): H = 0 (second derivative of linear is zero)

        Therefore, HVP is zero for polynomial scaling kernels.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, shape (nvars, n2)
            Second set of points
        direction : Array, shape (nvars,)
            Direction vector

        Returns
        -------
        hvp : Array, shape (nvars, n1, n2)
            HVP (all zeros for polynomial kernels)
        """
        nvars = X1.shape[0]
        n1 = X1.shape[1]
        n2 = X2.shape[1]

        # For polynomial scaling (degree 0 or 1), second derivative is zero
        return self._bkd.zeros((nvars, n1, n2))

    def __mul__(self, other: "ScalingKernel[Array]") -> "ProductKernel":
        """Multiply two kernels."""
        from pyapprox.surrogates.kernels.composition import ProductKernel

        return ProductKernel(self, other)

    def __add__(self, other: "ScalingKernel[Array]") -> "SumKernel":
        """Add two kernels."""
        from pyapprox.surrogates.kernels.composition import SumKernel

        return SumKernel(self, other)
