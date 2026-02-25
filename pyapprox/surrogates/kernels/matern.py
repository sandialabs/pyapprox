"""
Matern kernels for varying levels of smoothness.

This module provides Matern kernel implementations:
- SquaredExponentialKernel (nu=inf): Infinitely differentiable, smoothest kernel
- Matern52Kernel (nu=5/2): Twice differentiable
- Matern32Kernel (nu=3/2): Once differentiable

All kernels support analytical Jacobians and Hessian-vector products (HVP)
w.r.t. hyperparameters for efficient optimization.
"""

from typing import Tuple
import math
from abc import abstractmethod

import numpy as np

from pyapprox.util.hyperparameter import LogHyperParameter
from pyapprox.util.hyperparameter import HyperParameterList
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.kernels.protocols import Kernel


class MaternKernel(Kernel):
    """
    Abstract base class for Matern kernel family.

    Provides common functionality for all Matern kernels including:
    - Length scale hyperparameter management
    - Distance computation
    - Operator overloading for composition

    Concrete implementations:
    - SquaredExponentialKernel (nu=∞): Infinitely differentiable
    - Matern52Kernel (nu=5/2): Twice differentiable
    - Matern32Kernel (nu=3/2): Once differentiable

    Subclasses must implement:
    - _eval_distance_form(distances): kernel value from distances
    - jacobian(X1, X2): Jacobian w.r.t. first input
    - jacobian_wrt_params(X1): Jacobian w.r.t. hyperparameters
    - hvp_wrt_params(X1, direction): HVP w.r.t. hyperparameters
    - radial_derivatives(r): φ'(r) and φ''(r)
    """

    def __init__(
        self,
        lenscale: Array,
        lenscale_bounds: Tuple[float, float],
        nvars: int,
        bkd: Backend[Array],
        fixed: bool = False,
    ):
        super().__init__(bkd)
        self._nvars = nvars

        self._log_lenscale = LogHyperParameter(
            "lenscale",
            nvars,
            lenscale,
            lenscale_bounds,
            bkd=self._bkd,
            fixed=fixed,
        )
        self._hyp_list = HyperParameterList([self._log_lenscale])

    @property
    @abstractmethod
    def nu(self) -> float:
        """Return the smoothness parameter nu."""
        ...

    def hyp_list(self) -> HyperParameterList:
        """Return the list of hyperparameters."""
        return self._hyp_list

    def nvars(self) -> int:
        """Return the number of input dimensions."""
        return self._nvars

    def diag(self, X1: Array) -> Array:
        """Return the diagonal of the kernel matrix (all ones for Matern)."""
        return self._bkd.full((X1.shape[1],), 1.0)

    @abstractmethod
    def _eval_distance_form(self, distances: Array) -> Array:
        """Evaluate kernel from scaled distances."""
        ...

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """Compute the kernel matrix."""
        lenscale = self._log_lenscale.exp_values()
        if X2 is None:
            X2 = X1
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)
        return self._eval_distance_form(distances)

    @abstractmethod
    def jacobian(self, X1: Array, X2: Array) -> Array:
        """Compute Jacobian w.r.t. first input."""
        ...

    @abstractmethod
    def jacobian_wrt_params(self, X1: Array) -> Array:
        """Compute Jacobian w.r.t. hyperparameters."""
        ...

    @abstractmethod
    def hvp_wrt_params(self, X1: Array, direction: Array) -> Array:
        """Compute Hessian-vector product w.r.t. hyperparameters."""
        ...

    @abstractmethod
    def radial_derivatives(self, r: Array) -> Tuple[Array, Array]:
        """Compute φ'(r) and φ''(r) for the radial kernel."""
        ...

    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product of kernel w.r.t. first argument.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points (n1 must be 1)
        X2 : Array, shape (nvars, n2)
            Second set of points
        direction : Array, shape (nvars,)
            Direction vector for HVP

        Returns
        -------
        hvp : Array, shape (n1, n2, nvars)
            Hessian-vector product
        """
        nvars = X1.shape[0]
        n1 = X1.shape[1]
        n2 = X2.shape[1]

        lenscale = self._log_lenscale.exp_values()

        if n1 != 1:
            raise NotImplementedError(
                "hvp_wrt_x1 currently only supports n1=1 (single query point)"
            )

        x_star = X1[:, 0]
        V = direction

        diffs = x_star[:, None] - X2
        diffs_scaled = diffs / lenscale[:, None]

        r_squared = self._bkd.sum(diffs_scaled**2, axis=0)
        r = self._bkd.sqrt(r_squared + 1e-12)

        phi_prime, phi_double_prime = self.radial_derivatives(r)

        ell_squared = lenscale[:, None]**2
        r_inv = 1.0 / r
        dr_dx = diffs / (r[None, :] * ell_squared)

        dr_dot_V = self._bkd.einsum('ki,k->i', dr_dx, V)

        term1 = dr_dx * (phi_double_prime[None, :] * dr_dot_V[None, :])

        diffs_over_ell2 = diffs / ell_squared
        diffs_scaled_dot_V = self._bkd.einsum('ki,k->i', diffs_over_ell2, V)

        r_inv3 = r_inv**3
        d2r_dot_V = (-diffs_over_ell2 * (r_inv3[None, :] * diffs_scaled_dot_V[None, :]) +
                     V[:, None] * r_inv[None, :] / ell_squared)

        term2 = d2r_dot_V * phi_prime[None, :]

        hvp_2d = term1 + term2

        return self._bkd.transpose(hvp_2d[:, None, :], (1, 2, 0))


class SquaredExponentialKernel(MaternKernel):
    """
    Squared Exponential kernel, also known as RBF (Radial Basis Function).

    This is the Matern kernel with nu=∞, giving an infinitely differentiable
    covariance function.

    K(x, x') = exp(-0.5 * Σ_d (x_d - x'_d)² / ℓ_d²)

    Parameters
    ----------
    lenscale : Array
        Length scale parameters, one per dimension.
    lenscale_bounds : Tuple[float, float]
        Bounds for length scale parameters.
    nvars : int
        Number of input dimensions.
    bkd : Backend
        Backend for computations.
    fixed : bool, optional
        Whether hyperparameters are fixed.
    """

    @property
    def nu(self) -> float:
        return np.inf

    def _eval_distance_form(self, distances: Array) -> Array:
        return self._bkd.exp(-(distances**2) / 2.0)

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """Compute Jacobian w.r.t. first input."""
        lenscale = self._log_lenscale.exp_values()
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)
        K = self._bkd.exp(-0.5 * distances**2)

        tmp2 = (X1.T[:, None, :] - X2.T[None, :, :]) / lenscale**2
        return -K[..., None] * tmp2

    def jacobian_wrt_params(self, X1: Array) -> Array:
        """Compute Jacobian w.r.t. log-length scale parameters."""
        lenscale = self._log_lenscale.exp_values()
        Kmat = self(X1, X1)

        tmp2 = ((X1.T[:, None, :] - X1.T[None, :, :]) / lenscale**2) ** 2
        jac = tmp2 * Kmat[..., None]
        jac *= lenscale**2
        return jac

    def hvp_wrt_params(self, X1: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. log-length scale parameters.

        HVP[i] = Σ_j H[i,j] * v[j]
               = K * r_i² * (Σ_j r_j² * v[j] - 2*v[i])
        """
        lenscale = self._log_lenscale.exp_values()
        Kmat = self(X1, X1)
        nvars = X1.shape[0]

        diffs = X1[:, :, None] - X1[:, None, :]
        r_sq = (diffs / lenscale[:, None, None])**2

        r_sq_dot_v = self._bkd.einsum('ijk,i->jk', r_sq, direction)

        r_sq_transposed = self._bkd.transpose(r_sq, (1, 2, 0))
        hvp = Kmat[:, :, None] * r_sq_transposed * (
            r_sq_dot_v[:, :, None] - 2.0 * direction[None, None, :]
        )

        return hvp

    def radial_derivatives(self, r: Array) -> Tuple[Array, Array]:
        """Compute φ'(r) and φ''(r) for RBF kernel."""
        exp_term = self._bkd.exp(-r**2 / 2.0)
        phi_prime = -r * exp_term
        phi_double_prime = (r**2 - 1) * exp_term
        return phi_prime, phi_double_prime

    def get_kernel_1d(self, dim: int) -> "SquaredExponentialKernel":
        """
        Get a 1D SE kernel for a specific dimension.

        The Squared Exponential kernel is separable:
            K(x, y) = exp(-0.5 * sum_d (x_d - y_d)^2 / l_d^2)
                    = prod_d exp(-0.5 * (x_d - y_d)^2 / l_d^2)
                    = prod_d K_d(x_d, y_d)

        This method returns the 1D kernel K_d for dimension d.

        Parameters
        ----------
        dim : int
            Dimension index (0 to nvars-1).

        Returns
        -------
        kernel_1d : SquaredExponentialKernel
            A new 1D SE kernel with the length scale for dimension `dim`.

        Raises
        ------
        IndexError
            If dim is out of range [0, nvars-1].
        """
        if dim < 0 or dim >= self._nvars:
            raise IndexError(
                f"dim must be in range [0, {self._nvars - 1}], got {dim}"
            )

        # Extract length scale for this dimension
        ls_dim = self._log_lenscale.exp_values()[dim:dim+1]

        # Get bounds in user space (exp of log bounds)
        bounds = self._bkd.exp(self._log_lenscale.get_bounds())
        bounds_tuple = (
            float(self._bkd.to_numpy(bounds[0, 0])),
            float(self._bkd.to_numpy(bounds[0, 1]))
        )

        return SquaredExponentialKernel(
            lenscale=ls_dim,
            lenscale_bounds=bounds_tuple,
            nvars=1,
            bkd=self._bkd,
            fixed=self._log_lenscale.nactive_params() == 0
        )

    def __repr__(self) -> str:
        return f"SquaredExponentialKernel({self._hyp_list}, bkd={self._bkd.__class__.__name__})"


class Matern52Kernel(MaternKernel):
    """
    Matern kernel with nu=5/2, giving twice-differentiable sample paths.

    K(x, x') = (1 + √5·r + 5r²/3) * exp(-√5·r)

    where r = ||x - x'||_ℓ (scaled Euclidean distance).

    Parameters
    ----------
    lenscale : Array
        Length scale parameters.
    lenscale_bounds : Tuple[float, float]
        Bounds for length scale parameters.
    nvars : int
        Number of input dimensions.
    bkd : Backend
        Backend for computations.
    fixed : bool, optional
        Whether hyperparameters are fixed.
    """

    @property
    def nu(self) -> float:
        return 2.5

    def _eval_distance_form(self, distances: Array) -> Array:
        tmp = math.sqrt(5) * distances
        return (1.0 + tmp + tmp**2 / 3.0) * self._bkd.exp(-tmp)

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """Compute Jacobian w.r.t. first input."""
        lenscale = self._log_lenscale.exp_values()
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)
        tmp1 = math.sqrt(5) * distances
        K = self._bkd.exp(-tmp1)

        tmp2 = (X1.T[:, None, :] - X2.T[None, :, :]) / lenscale**2
        return -5/3 * K[..., None] * tmp2 * (math.sqrt(5) * distances[..., None] + 1)

    def jacobian_wrt_params(self, X1: Array) -> Array:
        """Compute Jacobian w.r.t. log-length scale parameters."""
        lenscale = self._log_lenscale.exp_values()
        distances = self._bkd.cdist(X1.T / lenscale, X1.T / lenscale)
        Kmat = self._eval_distance_form(distances)

        tmp = math.sqrt(5) * distances
        tmp2 = ((X1.T[:, None, :] - X1.T[None, :, :]) / lenscale**2) ** 2
        jac = tmp2 * Kmat[..., None]
        jac *= (5.0 * (tmp + 1.0) / (tmp**2 + 3.0 * tmp + 3.0))[..., None]
        jac *= lenscale**2
        return jac

    def hvp_wrt_params(self, X1: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. log-length scale parameters.

        Vectorized computation without forming full Hessian:
        HVP[i] = Σ_j H[i,j] * v[j]
               = K*g²*s_i²*(Σ_j s_j²*v_j) - K*g'/r*s_i²*(Σ_j s_j²*v_j) - 2*K*g*s_i²*v_i
               = K*s_i²*(g² - g'/r)*(s²·v) - 2*K*g*s_i²*v_i
        """
        lenscale = self._log_lenscale.exp_values()
        nvars = X1.shape[0]

        # Compute scaled differences
        diffs = X1[:, :, None] - X1[:, None, :]  # (nvars, n, n)
        s_sq = (diffs / lenscale[:, None, None])**2  # (nvars, n, n)

        # Compute r
        r_sq = self._bkd.sum(s_sq, axis=0)  # (n, n)
        r = self._bkd.sqrt(r_sq + 1e-12)

        # Compute kernel and g(r), g'(r)
        sqrt5 = math.sqrt(5)
        sqrt5_r = sqrt5 * r
        Kmat = self._eval_distance_form(r)

        denom = 5*r_sq + 3*sqrt5_r + 3.0
        g = 5.0 * (sqrt5_r + 1.0) / denom

        num = 5.0 * (sqrt5_r + 1.0)
        dnum_dr = 5.0 * sqrt5
        ddenom_dr = 10*r + 3*sqrt5
        g_prime = (dnum_dr * denom - num * ddenom_dr) / (denom**2)

        # s_sq transposed: (n, n, nvars)
        s_sq_T = self._bkd.transpose(s_sq, (1, 2, 0))

        # Compute s²·v = Σ_j s_j² * v_j
        s_sq_dot_v = self._bkd.einsum('ijd,d->ij', s_sq_T, direction)  # (n, n)

        # Coefficient for off-diagonal terms: g² - g'/r
        r_safe = r + 1e-12
        coeff = g**2 - g_prime / r_safe  # (n, n)

        # HVP = K*s_i²*coeff*(s²·v) - 2*K*g*s_i²*v_i
        # First term: K*coeff*(s²·v) broadcast with s_i²
        first_factor = Kmat * coeff * s_sq_dot_v  # (n, n)
        term1 = first_factor[:, :, None] * s_sq_T  # (n, n, nvars)

        # Second term: -2*K*g*s_i²*v_i
        Kg = Kmat * g  # (n, n)
        term2 = -2.0 * Kg[:, :, None] * s_sq_T * direction[None, None, :]  # (n, n, nvars)

        return term1 + term2

    def radial_derivatives(self, r: Array) -> Tuple[Array, Array]:
        """Compute φ'(r) and φ''(r) for Matern 5/2 kernel."""
        sqrt5 = math.sqrt(5)
        sqrt5_r = sqrt5 * r
        exp_term = self._bkd.exp(-sqrt5_r)

        phi_prime = (5 * r / 3) * (-sqrt5_r - 1) * exp_term
        phi_double_prime = (5.0 / 3.0) * (5 * r**2 - sqrt5_r - 1) * exp_term

        return phi_prime, phi_double_prime

    def __repr__(self) -> str:
        return f"Matern52Kernel({self._hyp_list}, bkd={self._bkd.__class__.__name__})"


class Matern32Kernel(MaternKernel):
    """
    Matern kernel with nu=3/2, giving once-differentiable sample paths.

    K(x, x') = (1 + √3·r) * exp(-√3·r)

    where r = ||x - x'||_ℓ (scaled Euclidean distance).

    Parameters
    ----------
    lenscale : Array
        Length scale parameters.
    lenscale_bounds : Tuple[float, float]
        Bounds for length scale parameters.
    nvars : int
        Number of input dimensions.
    bkd : Backend
        Backend for computations.
    fixed : bool, optional
        Whether hyperparameters are fixed.
    """

    @property
    def nu(self) -> float:
        return 1.5

    def _eval_distance_form(self, distances: Array) -> Array:
        tmp = math.sqrt(3) * distances
        return (1.0 + tmp) * self._bkd.exp(-tmp)

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """Compute Jacobian w.r.t. first input."""
        lenscale = self._log_lenscale.exp_values()
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)
        tmp1 = math.sqrt(3) * distances
        K = self._bkd.exp(-tmp1)

        tmp2 = (X1.T[:, None, :] - X2.T[None, :, :]) / lenscale**2
        return -3 * K[..., None] * tmp2

    def jacobian_wrt_params(self, X1: Array) -> Array:
        """Compute Jacobian w.r.t. log-length scale parameters."""
        lenscale = self._log_lenscale.exp_values()
        distances = self._bkd.cdist(X1.T / lenscale, X1.T / lenscale)
        Kmat = self._eval_distance_form(distances)

        tmp = math.sqrt(3) * distances
        tmp2 = ((X1.T[:, None, :] - X1.T[None, :, :]) / lenscale**2) ** 2
        jac = tmp2 * Kmat[..., None]
        jac *= (3.0 / (tmp + 1.0))[..., None]
        jac *= lenscale**2
        return jac

    def hvp_wrt_params(self, X1: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. log-length scale parameters.

        Vectorized computation without forming full Hessian:
        HVP[i] = Σ_j H[i,j] * v[j]
               = K*s_i²*(g² - g'/r)*(s²·v) - 2*K*g*s_i²*v_i
        """
        lenscale = self._log_lenscale.exp_values()

        # Compute scaled differences
        diffs = X1[:, :, None] - X1[:, None, :]  # (nvars, n, n)
        s_sq = (diffs / lenscale[:, None, None])**2  # (nvars, n, n)

        # Compute r
        r_sq = self._bkd.sum(s_sq, axis=0)  # (n, n)
        r = self._bkd.sqrt(r_sq + 1e-12)

        sqrt3 = math.sqrt(3)
        sqrt3_r = sqrt3 * r
        Kmat = self._eval_distance_form(r)

        # g(r) = 3/(√3r+1)
        g = 3.0 / (sqrt3_r + 1.0)

        # g'(r) = -3√3/(√3r+1)²
        g_prime = -3.0 * sqrt3 / ((sqrt3_r + 1.0)**2)

        # s_sq transposed: (n, n, nvars)
        s_sq_T = self._bkd.transpose(s_sq, (1, 2, 0))

        # Compute s²·v = Σ_j s_j² * v_j
        s_sq_dot_v = self._bkd.einsum('ijd,d->ij', s_sq_T, direction)  # (n, n)

        # Coefficient: g² - g'/r
        r_safe = r + 1e-12
        coeff = g**2 - g_prime / r_safe  # (n, n)

        # HVP = K*s_i²*coeff*(s²·v) - 2*K*g*s_i²*v_i
        first_factor = Kmat * coeff * s_sq_dot_v  # (n, n)
        term1 = first_factor[:, :, None] * s_sq_T  # (n, n, nvars)

        Kg = Kmat * g  # (n, n)
        term2 = -2.0 * Kg[:, :, None] * s_sq_T * direction[None, None, :]  # (n, n, nvars)

        return term1 + term2

    def radial_derivatives(self, r: Array) -> Tuple[Array, Array]:
        """Compute φ'(r) and φ''(r) for Matern 3/2 kernel."""
        sqrt3 = math.sqrt(3)
        sqrt3_r = sqrt3 * r
        exp_term = self._bkd.exp(-sqrt3_r)

        phi_prime = -3 * r * exp_term
        phi_double_prime = 3 * (sqrt3_r - 1) * exp_term

        return phi_prime, phi_double_prime

    def __repr__(self) -> str:
        return f"Matern32Kernel({self._hyp_list}, bkd={self._bkd.__class__.__name__})"


class ExponentialKernel(MaternKernel):
    """
    Exponential kernel (Matern with nu=1/2, Ornstein-Uhlenbeck).

    K(x, x') = exp(-r)

    where r = ||x - x'||_ℓ (scaled Euclidean distance).

    This kernel produces continuous but non-differentiable sample paths.

    Parameters
    ----------
    lenscale : Array
        Length scale parameters.
    lenscale_bounds : Tuple[float, float]
        Bounds for length scale parameters.
    nvars : int
        Number of input dimensions.
    bkd : Backend
        Backend for computations.
    fixed : bool, optional
        Whether hyperparameters are fixed.
    """

    @property
    def nu(self) -> float:
        return 0.5

    def _eval_distance_form(self, distances: Array) -> Array:
        return self._bkd.exp(-distances)

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """Compute Jacobian w.r.t. first input."""
        lenscale = self._log_lenscale.exp_values()
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)
        K = self._bkd.exp(-distances)
        r_safe = distances + 1e-12

        tmp2 = (X1.T[:, None, :] - X2.T[None, :, :]) / lenscale**2
        return -K[..., None] * tmp2 / r_safe[..., None]

    def jacobian_wrt_params(self, X1: Array) -> Array:
        """Compute Jacobian w.r.t. log-length scale parameters."""
        lenscale = self._log_lenscale.exp_values()
        distances = self._bkd.cdist(X1.T / lenscale, X1.T / lenscale)
        K = self._bkd.exp(-distances)
        r_safe = distances + 1e-12

        tmp2 = ((X1.T[:, None, :] - X1.T[None, :, :]) / lenscale**2) ** 2
        jac = tmp2 * K[..., None] / r_safe[..., None]
        jac *= lenscale**2
        return jac

    def hvp_wrt_params(self, X1: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. log-length scale parameters.

        HVP[i] = Σ_j H[i,j] * v[j]
               = K*s_i²*(g² - g'/r)*(s²·v) - 2*K*g*s_i²*v_i

        where g(r) = 1/r and K = exp(-r).
        """
        lenscale = self._log_lenscale.exp_values()

        # Compute scaled differences
        diffs = X1[:, :, None] - X1[:, None, :]  # (nvars, n, n)
        s_sq = (diffs / lenscale[:, None, None])**2  # (nvars, n, n)

        # Compute r
        r_sq = self._bkd.sum(s_sq, axis=0)  # (n, n)
        r = self._bkd.sqrt(r_sq + 1e-12)

        K = self._bkd.exp(-r)

        # g(r) = 1/r for exponential kernel
        r_safe = r + 1e-12
        g = 1.0 / r_safe

        # g'(r) = -1/r²
        g_prime = -1.0 / (r_safe**2)

        # s_sq transposed: (n, n, nvars)
        s_sq_T = self._bkd.transpose(s_sq, (1, 2, 0))

        # Compute s²·v = Σ_j s_j² * v_j
        s_sq_dot_v = self._bkd.einsum(
            'ijd,d->ij', s_sq_T, direction
        )  # (n, n)

        # Coefficient: g² - g'/r
        coeff = g**2 - g_prime / r_safe  # (n, n)

        # HVP = K*s_i²*coeff*(s²·v) - 2*K*g*s_i²*v_i
        first_factor = K * coeff * s_sq_dot_v  # (n, n)
        term1 = first_factor[:, :, None] * s_sq_T  # (n, n, nvars)

        Kg = K * g  # (n, n)
        term2 = (
            -2.0 * Kg[:, :, None] * s_sq_T * direction[None, None, :]
        )  # (n, n, nvars)

        return term1 + term2

    def radial_derivatives(self, r: Array) -> Tuple[Array, Array]:
        """Compute φ'(r) and φ''(r) for exponential kernel."""
        exp_term = self._bkd.exp(-r)
        phi_prime = -exp_term
        phi_double_prime = exp_term
        return phi_prime, phi_double_prime

    def __repr__(self) -> str:
        return f"ExponentialKernel({self._hyp_list}, bkd={self._bkd.__class__.__name__})"
