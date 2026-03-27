"""
General Matern kernel supporting arbitrary smoothness parameter nu.

This kernel extends the backend-agnostic Matern implementation to support
any positive nu value, not just the special cases (1/2, 3/2, 5/2, inf).

For half-integer nu (n + 0.5), closed-form expressions are used.
For general nu, asymptotic expansions are used.

This kernel does not implement analytical Jacobian or HVP methods.
Use with TorchExactGaussianProcess which computes derivatives via autograd.
"""

import math
from typing import Any, Tuple

from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.util.backends.protocols import Array
from pyapprox.util.hyperparameter import HyperParameterList, LogHyperParameter


class GeneralMaternKernel(Kernel[Array]):
    """
    General Matern kernel with arbitrary smoothness parameter nu.

    The Matern kernel is defined as:

        K(r) = (2^(1-nu) / Gamma(nu)) * (sqrt(2*nu) * r)^nu * K_nu(sqrt(2*nu) * r)

    where:
        - r = ||x - x'||_ell (scaled Euclidean distance)
        - K_nu is the modified Bessel function of the second kind
        - nu controls smoothness (higher = smoother)

    Special cases:
        - nu = 1/2: Exponential kernel (Ornstein-Uhlenbeck)
        - nu = 3/2: Once differentiable
        - nu = 5/2: Twice differentiable
        - nu -> inf: Squared Exponential (RBF)

    For half-integer nu (n + 0.5), closed-form polynomial expressions are used.
    For general nu, asymptotic expansions approximate the Bessel function.

    This kernel does NOT implement jacobian, jacobian_wrt_params, or hvp methods.
    Use with TorchExactGaussianProcess which computes derivatives via autograd.

    Note: This kernel always uses TorchBkd internally since it requires
    PyTorch autograd for derivative computation.

    Warning: For non-half-integer nu values (not 0.5, 1.5, 2.5, 3.5, ...),
    the implementation uses approximations that may be less accurate.
    For best results, use half-integer nu values.

    Parameters
    ----------
    nu : float
        Smoothness parameter. Must be positive.
    lenscale : list or Array
        Length scale parameters, one per dimension.
    lenscale_bounds : Tuple[float, float]
        Bounds for length scale parameters (used in optimization).
    nvars : int
        Number of input dimensions.
    fixed : bool, optional
        Whether hyperparameters are fixed during optimization. Default False.

    Examples
    --------
    >>> kernel = GeneralMaternKernel(
    ...     nu=2.3, lenscale=[1.0, 1.0],
    ...     lenscale_bounds=(0.1, 10.0), nvars=2
    ... )
    >>> X = kernel._bkd.array([[0.0, 1.0], [0.0, 0.0]])
    >>> K = kernel(X, X)  # Shape: (2, 2)
    """

    def __init__(
        self,
        nu: float,
        lenscale: list[Any],
        lenscale_bounds: Tuple[float, float],
        nvars: int,
        fixed: bool = False,
    ):
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")

        # Always use TorchBkd since this kernel requires PyTorch autograd
        from pyapprox.util.backends.torch import TorchBkd

        bkd = TorchBkd()

        super().__init__(bkd)

        self._nu = nu
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

        # Check if nu is half-integer (n + 0.5) for closed-form computation
        # Half-integers: 0.5, 1.5, 2.5, 3.5, ... => 2*nu is odd integer
        two_nu = 2.0 * nu
        self._is_half_integer = (
            abs(two_nu - round(two_nu)) < 1e-10 and int(round(two_nu)) % 2 == 1
        )
        if self._is_half_integer:
            self._half_int_n = int(round(nu - 0.5))

    @property
    def nu(self) -> float:
        """Return the smoothness parameter nu."""
        return self._nu

    def hyp_list(self) -> HyperParameterList:
        """Return the list of hyperparameters."""
        return self._hyp_list

    def nvars(self) -> int:
        """Return the number of input dimensions."""
        return self._nvars

    def diag(self, X1: Array) -> Array:
        """Return the diagonal of the kernel matrix (all ones for Matern)."""
        return self._bkd.full((X1.shape[1],), 1.0)

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Compute the kernel matrix K(X1, X2).

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points.
        X2 : Array, shape (nvars, n2), optional
            Second set of points. If None, uses X1.

        Returns
        -------
        K : Array, shape (n1, n2)
            Kernel matrix.
        """
        if X2 is None:
            X2 = X1

        lenscale = self._log_lenscale.exp_values()
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)

        return self._eval_distance_form(distances)

    def _eval_distance_form(self, r: Array) -> Array:
        """
        Evaluate kernel from scaled distances.

        Parameters
        ----------
        r : Array
            Scaled distances.

        Returns
        -------
        K : Array
            Kernel values.
        """
        nu = self._nu

        # Handle nu -> inf (RBF kernel)
        if nu > 100:
            return self._bkd.exp(-0.5 * r**2)

        # Add small epsilon for numerical stability at r=0
        eps = 1e-10
        r_safe = r + eps

        # Compute sqrt(2*nu) * r
        sqrt_2nu = math.sqrt(2.0 * nu)
        z = sqrt_2nu * r_safe

        # Use closed-form for half-integer nu, otherwise asymptotic
        if self._is_half_integer:
            K = self._matern_half_integer(z, r_safe)
        else:
            K = self._matern_general(z, r_safe)

        return K

    def _matern_half_integer(self, z: Array, r: Array) -> Array:
        """
        Compute Matern kernel for half-integer nu using closed form.

        For nu = n + 1/2, the Matern kernel simplifies to:
            K(r) = exp(-z) * P_n(z)
        where z = sqrt(2*nu)*r and P_n is a polynomial with P_n(0) = 1.
        """
        n = self._half_int_n

        if n == 0:  # nu = 0.5 (Exponential)
            return self._bkd.exp(-z)

        elif n == 1:  # nu = 1.5 (Matern 3/2)
            return (1.0 + z) * self._bkd.exp(-z)

        elif n == 2:  # nu = 2.5 (Matern 5/2)
            return (1.0 + z + z**2 / 3.0) * self._bkd.exp(-z)

        elif n == 3:  # nu = 3.5
            return (1.0 + z + 2.0 * z**2 / 5.0 + z**3 / 15.0) * self._bkd.exp(-z)

        elif n == 4:  # nu = 4.5
            return (
                1.0 + z + 3.0 * z**2 / 7.0 + 2.0 * z**3 / 21.0 + z**4 / 105.0
            ) * self._bkd.exp(-z)

        else:
            # General half-integer formula using recurrence
            # The polynomial P_n(z) satisfies P_n(0) = 1
            # P_n(z) = sum_{k=0}^{n} c_k * z^k
            # where c_0 = 1 and c_k = c_{k-1} * (n-k+1) / (k * (2n-k+1))
            coeffs = [1.0]
            for k in range(1, n + 1):
                c_prev = coeffs[-1]
                c_k = c_prev * (n - k + 1) / (k * (2 * n - k + 1))
                coeffs.append(c_k)

            result = self._bkd.full(z.shape, coeffs[0])
            z_pow = z
            for k in range(1, n + 1):
                result = result + coeffs[k] * z_pow
                z_pow = z_pow * z

            return result * self._bkd.exp(-z)

    def _matern_general(self, z: Array, r: Array) -> Array:
        """
        Compute Matern kernel for general nu.

        For general nu, we use a combination of:
        - Small z: Series expansion
        - Large z: Asymptotic expansion
        """
        nu = self._nu

        # For r ≈ 0 (z ≈ 0), the kernel value should be 1
        # Use a threshold to switch between methods
        z_threshold = 0.1

        # Create masks for small and large z
        small_z_mask = z < z_threshold
        large_z_mask = ~small_z_mask

        # Initialize result
        K = self._bkd.zeros(z.shape)

        # For small z, use the limiting form
        # As z → 0, K(r) → 1
        # Use Taylor expansion: K(r) ≈ 1 - c * z^2 + O(z^4)
        # where c depends on nu
        if self._bkd.any_bool(small_z_mask):
            # For small z: K(r) ≈ 1 - (z^2)/(2*nu) + higher order terms
            z_small = z * small_z_mask
            K_small = 1.0 - (z_small**2) / (4.0 * nu)
            K = K + K_small * small_z_mask

        # For large z, use asymptotic expansion
        if self._bkd.any_bool(large_z_mask):
            z_large = z * large_z_mask + (~large_z_mask) * 1.0  # avoid division by zero

            # Normalization: 2^(1-nu) / Gamma(nu)
            log_norm = (1.0 - nu) * math.log(2.0) - math.lgamma(nu)
            norm = math.exp(log_norm)

            # z^nu
            z_pow_nu = z_large**nu

            # K_nu(z) using asymptotic expansion
            kve_exp_neg_z = self._kve_times_exp_neg_z(nu, z_large)

            K_large = norm * z_pow_nu * kve_exp_neg_z
            K = K + K_large * large_z_mask

        return K

    def _kve_times_exp_neg_z(self, nu: float, z: Array) -> Array:
        """
        Compute kve(nu, z) * exp(-z) = K_nu(z) using asymptotic expansion.

        K_nu(z) ~ sqrt(pi/(2z)) * exp(-z) * sum_{k=0}^{inf} a_k(nu) / z^k
        """
        mu = 4.0 * nu * nu

        # Start with leading term
        sqrt_pi_2z = math.sqrt(math.pi / 2.0) * self._bkd.sqrt(1.0 / z)
        exp_neg_z = self._bkd.exp(-z)

        # Asymptotic series
        result = self._bkd.full(z.shape, 1.0)
        term = self._bkd.full(z.shape, 1.0)

        # Compute terms of asymptotic expansion
        for k in range(1, 15):
            factor = (mu - (2 * k - 1) ** 2) / (8.0 * k)
            term = term * factor / z
            result = result + term

        return sqrt_pi_2z * exp_neg_z * result

    def __repr__(self) -> str:
        lenscale = self._log_lenscale.exp_values()
        return (
            f"GeneralMaternKernel(nu={self._nu}, "
            f"lenscale={self._bkd.to_numpy(lenscale).tolist()}, "
            f"nvars={self._nvars}, bkd={self._bkd.__class__.__name__})"
        )


# Alias for backwards compatibility
TorchMaternKernel = GeneralMaternKernel
