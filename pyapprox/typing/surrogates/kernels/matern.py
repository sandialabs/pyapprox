from typing import Tuple
import math

import numpy as np

from pyapprox.typing.util.hyperparameter import LogHyperParameter
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.kernels.protocols import Kernel


class MaternKernel(Kernel):
    """
    The Matern kernel for varying levels of smoothness.

    Parameters
    ----------
    nu : float
        Smoothness parameter of the Matern kernel.
    lenscale : float
        Length scale parameter of the Matern kernel.
    lenscale_bounds : Tuple[float, float]
        Bounds for the length scale parameter.
    nvars : int
        Number of variables (dimensions) in the input data.
    fixed : bool, optional
        Whether the hyperparameter is fixed (default is False).
    bkd : Backend
        Backend for numerical computations.
    """

    def __init__(
        self,
        nu: float,
        lenscale: Array,
        lenscale_bounds: Tuple[float, float],
        nvars: int,
        bkd: Backend[Array],
        fixed: bool = False,
    ):
        super().__init__(bkd)
        self._nvars = nvars
        self._nu = nu

        self._log_lenscale = LogHyperParameter(
            "lenscale",
            nvars,
            lenscale,
            lenscale_bounds,
            bkd=self._bkd,
            fixed=fixed,
        )
        self._hyp_list = HyperParameterList([self._log_lenscale])

    def hyp_list(self) -> HyperParameterList:
        """
        Return the list of hyperparameters associated with the kernel.

        Returns
        -------
        hyp_list : HyperParameterList
            List of hyperparameters.
        """
        return self._hyp_list

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
        return self._bkd.full((X1.shape[1],), 1.0)

    def _eval_distance_form(self, distances: Array) -> Array:
        """
        Evaluate the kernel based on the distance form.

        Parameters
        ----------
        distances : Array
            Pairwise distances between input data points.

        Returns
        -------
        kernel_values : Array
            Kernel values based on the distance form.
        """
        if self._nu == np.inf:
            return self._bkd.exp(-(distances**2) / 2.0)
        if self._nu == 5 / 2:
            tmp = math.sqrt(5) * distances
            return (1.0 + tmp + tmp**2 / 3.0) * self._bkd.exp(-tmp)
        if self._nu == 3 / 2:
            tmp = math.sqrt(3) * distances
            return (1.0 + tmp) * self._bkd.exp(-tmp)
        raise ValueError(
            "Matern kernel with nu={0} not supported".format(self._nu)
        )

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
        lenscale = self._log_lenscale.exp_values()
        if X2 is None:
            X2 = X1
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)
        return self._eval_distance_form(distances)

    def nvars(self) -> int:
        """
        Return the number of variables (dimensions) in the input data.

        Returns
        -------
        nvars : int
            Number of variables.
        """
        return self._nvars

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
        lenscale = self._log_lenscale.exp_values()
        distances = self._bkd.cdist(X1.T / lenscale, X2.T / lenscale)
        if self._nu == np.inf:
            K = self._bkd.exp(-0.5 * distances**2)
            # Compute the pairwise difference normalized by the length scale
            tmp2 = (
                X1.T[:, None, :] - X2.T[None, :, :]
            ) / lenscale**2  # Shape: (n1, n2, d)

            # Compute the Jacobian
            return -K[..., None] * tmp2  # Shape: (n1, n2, d)
        if self._nu == 3 / 2:
            # For nu = 3/2
            tmp1 = math.sqrt(3) * distances
            K = self._bkd.exp(-tmp1)

            # Compute the pairwise difference normalized by the length scale
            tmp2 = (
                X1.T[:, None, :] - X2.T[None, :, :]
            ) / lenscale**2  # Shape: (n1, n2, d)

            # Compute the Jacobian
            return -3 * K[..., None] * tmp2  # Shape: (n1, n2, d)

        if self._nu == 5 / 2:
            # For nu = 5/2
            tmp1 = math.sqrt(5) * distances
            K = self._bkd.exp(-tmp1)

            # Compute the pairwise difference normalized by the length scale
            tmp2 = (
                X1.T[:, None, :] - X2.T[None, :, :]
            ) / lenscale**2  # Shape: (n1, n2, d)

            # Compute the Jacobian
            return (
                -5
                / 3
                * K[..., None]
                * tmp2
                * (math.sqrt(5) * distances[..., None] + 1)
            )  # Shape: (n1, n2, d)

        raise NotImplementedError(f"jacobian not implemented for {self._nu=}")

    def jacobian_wrt_params(self, X1: Array) -> Array:
        """
        Compute the Jacobian of the kernel with respect to hyperparameters
         (log of the lenscale).

        Parameters
        ----------
        X1 : Array
            Input data.

        Returns
        -------
        jacobian_wrt_params : Array
            Jacobian of the kernel with respect to hyperparameters
            (log of the lenscale).
        """
        lenscale = (
            self._log_lenscale.exp_values()
        )  # Vector-valued length scale
        distances = self._bkd.cdist(
            X1.T / lenscale, X1.T / lenscale
        )  # Pairwise distances
        Kmat = self._eval_distance_form(distances)  # Kernel matrix

        # Compute the Jacobian for each supported case
        if self._nu == np.inf:
            # For nu = infinity (Squared Exponential Kernel)
            tmp2 = (
                (X1.T[:, None, :] - X1.T[None, :, :]) / lenscale**2
            ) ** 2  # Shape: (n, n, d)
            jac = tmp2 * Kmat[..., None]  # Shape: (n, n, d)
            jac *= lenscale**2
            return jac

        if self._nu == 5 / 2:
            # For nu = 5/2
            tmp = math.sqrt(5) * distances
            tmp2 = (
                (X1.T[:, None, :] - X1.T[None, :, :]) / lenscale**2
            ) ** 2  # Shape: (n, n, d)
            jac = tmp2 * Kmat[..., None]  # Shape: (n, n, d)
            # Correct factor from symbolic differentiation:
            # dK/d(log ℓ) = K * (Δ²/ℓ²) * 5*(√5*r + 1)/(5*r² + 3*√5*r + 3)
            # where tmp = √5*r, so denominator = tmp² + 3*tmp + 3
            jac *= (5.0 * (tmp + 1.0) / (tmp**2 + 3.0 * tmp + 3.0))[..., None]

            # Adjust for log(lenscale) using chain rule
            jac *= lenscale**2  # Multiply by lenscale^2
            return jac

        if self._nu == 3 / 2:
            # For nu = 3/2
            tmp = math.sqrt(3) * distances
            tmp2 = (
                (X1.T[:, None, :] - X1.T[None, :, :]) / lenscale**2
            ) ** 2  # Shape: (n, n, d)
            jac = tmp2 * Kmat[..., None]  # Shape: (n, n, d)
            # Correct factor from symbolic differentiation:
            # dK/d(log ℓ) = K * (Δ²/ℓ²) * 3/(√3*r + 1)
            # where tmp = √3*r
            jac *= (3.0 / (tmp + 1.0))[..., None]

            # Adjust for log(lenscale) using chain rule
            jac *= lenscale**2  # Multiply by lenscale^2
            return jac

        raise ValueError(
            "Matern kernel with nu={0} not supported".format(self._nu)
        )

    def hessian_wrt_params(self, X1: Array) -> Array:
        """
        Compute Hessian of kernel w.r.t. log-length scale parameters.

        Computes all second derivatives ∂²K/∂(log ℓ_i)∂(log ℓ_j).

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n).

        Returns
        -------
        hessian : Array
            Hessian tensor, shape (n, n, nparams, nparams).
            hessian[:, :, i, j] = ∂²K/∂(log ℓ_i)∂(log ℓ_j)
        """
        lenscale = self._log_lenscale.exp_values()
        Kmat = self(X1, X1)  # Kernel matrix
        n = X1.shape[1]
        nvars = X1.shape[0]

        hess = self._bkd.zeros((n, n, nvars, nvars))

        if self._nu == np.inf:
            # RBF/Squared Exponential kernel
            # K = exp(-0.5 * Σ_d (x_d - x'_d)²/ℓ_d²)
            #
            # ∂K/∂(log ℓ_d) = K * r_d² where r_d = (x_d - x'_d)/ℓ_d
            # ∂²K/∂(log ℓ_i)∂(log ℓ_j) = K * r_i² * r_j² for i≠j
            # ∂²K/∂(log ℓ_d)² = K * r_d² * (r_d² - 2)

            # Compute all scaled squared differences
            r_sq = []  # List of r_d² for each dimension
            for d in range(nvars):
                diff_d = X1[d:d+1, :].T - X1[d:d+1, :]  # Shape: (n, n)
                r_sq_d = (diff_d / lenscale[d])**2  # (Δ_d/ℓ_d)²
                r_sq.append(r_sq_d)

            # Compute all Hessian elements
            for i in range(nvars):
                for j in range(nvars):
                    if i == j:
                        # Diagonal: ∂²K/∂(log ℓ_d)²
                        hess[:, :, i, j] = Kmat * r_sq[i] * (r_sq[i] - 2.0)
                    else:
                        # Off-diagonal: ∂²K/∂(log ℓ_i)∂(log ℓ_j)
                        hess[:, :, i, j] = Kmat * r_sq[i] * r_sq[j]

            return hess

        else:
            raise NotImplementedError(
                f"hessian_wrt_params not implemented for nu={self._nu}"
            )

    def radial_derivatives(self, r: Array) -> Tuple[Array, Array]:
        """
        Compute first and second derivatives of the radial kernel φ(r).

        For Matern kernels: k(x, x') = φ(r) where r = ||x - x'|| / lengthscale

        Parameters
        ----------
        r : Array
            Scaled distances, shape (n_points,)

        Returns
        -------
        phi_prime : Array
            First derivative φ'(r), shape (n_points,)
        phi_double_prime : Array
            Second derivative φ''(r), shape (n_points,)

        Notes
        -----
        For Matern kernels with different smoothness:
        - nu = ∞ (RBF): φ(r) = exp(-r²/2)
                        φ'(r) = -r·exp(-r²/2)
                        φ''(r) = (r² - 1)·exp(-r²/2)

        - nu = 3/2: φ(r) = (1 + √3·r)·exp(-√3·r)
                    φ'(r) = -3r·exp(-√3·r)
                    φ''(r) = 3(√3·r - 1)·exp(-√3·r)

        - nu = 5/2: φ(r) = (1 + √5·r + 5r²/3)·exp(-√5·r)
                    φ'(r) = (5r/3)·(-√5·r - 1)·exp(-√5·r)
                    φ''(r) = (5/3)·(5r² - √5·r - 1)·exp(-√5·r)
        """
        if self._nu == np.inf:
            # RBF kernel: φ(r) = exp(-r²/2)
            exp_term = self._bkd.exp(-r**2 / 2.0)
            phi_prime = -r * exp_term
            phi_double_prime = (r**2 - 1) * exp_term
            return phi_prime, phi_double_prime

        elif self._nu == 3 / 2:
            # Matern 3/2: φ(r) = (1 + √3·r)·exp(-√3·r)
            sqrt3 = math.sqrt(3)
            sqrt3_r = sqrt3 * r
            exp_term = self._bkd.exp(-sqrt3_r)

            # φ'(r) = -3r·exp(-√3·r)
            phi_prime = -3 * r * exp_term

            # φ''(r) = 3(√3·r - 1)·exp(-√3·r)
            phi_double_prime = 3 * (sqrt3_r - 1) * exp_term

            return phi_prime, phi_double_prime

        elif self._nu == 5 / 2:
            # Matern 5/2: φ(r) = (1 + √5·r + 5r²/3)·exp(-√5·r)
            sqrt5 = math.sqrt(5)
            sqrt5_r = sqrt5 * r
            exp_term = self._bkd.exp(-sqrt5_r)

            # φ'(r) = (5r/3)·(-√5·r - 1)·exp(-√5·r)
            phi_prime = (5 * r / 3) * (-sqrt5_r - 1) * exp_term

            # φ''(r) = (5/3)·(5r² - √5·r - 1)·exp(-√5·r)
            phi_double_prime = (5.0 / 3.0) * (5 * r**2 - sqrt5_r - 1) * exp_term

            return phi_prime, phi_double_prime

        else:
            raise ValueError(
                f"radial_derivatives not implemented for nu={self._nu}"
            )

    def hvp_wrt_params(self, X1: Array, direction: Array) -> Array:
        """
        Compute Hessian-vector product w.r.t. hyperparameters.

        Computes HVP = Σ_j (∂²K/∂θ_i∂θ_j) * v[j] for each i,
        without forming the full Hessian tensor.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n).
        direction : Array
            Direction vector, shape (nparams,).

        Returns
        -------
        hvp : Array
            Hessian-vector product, shape (n, n, nparams).
            hvp[:, :, i] = Σ_j (∂²K/∂θ_i∂θ_j) * v[j]
        """
        lenscale = self._log_lenscale.exp_values()
        Kmat = self(X1, X1)  # Kernel matrix
        n = X1.shape[1]
        nvars = X1.shape[0]

        hvp = self._bkd.zeros((n, n, nvars))

        if self._nu == np.inf:
            # RBF/Squared Exponential kernel
            # ∂²K/∂(log ℓ_i)∂(log ℓ_j) = K * r_i² * r_j² for i≠j
            # ∂²K/∂(log ℓ_d)² = K * r_d² * (r_d² - 2)

            # Vectorized computation of all scaled squared differences
            # Shape: (nvars, n, n)
            diffs = X1[:, :, None] - X1[:, None, :]  # (nvars, n, n)
            r_sq = (diffs / lenscale[:, None, None])**2  # (nvars, n, n)

            # Vectorized HVP computation
            # For each i: hvp[:,:,i] = Σ_j H[:,:,i,j] * v[j]
            #           = K * r_i² * Σ_j r_j² * v[j] + K * r_i² * (r_i² - 2) * v[i] - K * r_i² * r_i² * v[i]
            #           = K * r_i² * (Σ_j r_j² * v[j] - 2*v[i])

            # Compute Σ_j r_j² * v[j]: shape (n, n)
            r_sq_dot_v = self._bkd.einsum('ijk,i->jk', r_sq, direction)  # (n, n)

            # For each i: hvp[:,:,i] = K * r_i² * (r_sq_dot_v - 2*v[i])
            # Expand: (n, n, nvars)
            hvp = Kmat[:, :, None] * r_sq.transpose(1, 2, 0) * (
                r_sq_dot_v[:, :, None] - 2.0 * direction[None, None, :]
            )

            return hvp

        else:
            raise NotImplementedError(
                f"hvp_wrt_params not implemented for nu={self._nu}"
            )

    def hvp_wrt_x1(
        self, X1: Array, X2: Array, direction: Array
    ) -> Array:
        """
        Compute Hessian-vector product of kernel w.r.t. first argument.

        For Matern kernels with spatial scaling (anisotropic length scales),
        this computes H[k(X1, X2)]·V where H = ∂²k/∂X1².

        This is the kernel-specific implementation that was originally hardcoded
        in the GP's hvp() method, now modularized into the kernel class.

        Parameters
        ----------
        X1 : Array, shape (nvars, n1)
            First set of points
        X2 : Array, shape (nvars, n2)
            Second set of points
        direction : Array, shape (nvars,)
            Direction vector for Hessian-vector product

        Returns
        -------
        hvp : Array, shape (n1, n2, nvars)
            H[k(X1[:, i], X2[:, j])]·V for each pair (i, j)

            This shape is consistent with jacobian() which returns (n1, n2, nvars).

        Notes
        -----
        Uses chain rule: H_jk[k] = φ''·(∂r/∂x_j)·(∂r/∂x_k) + φ'·(∂²r/∂x_j∂x_k)

        So H·V = Σ_k H_jk·V_k
               = φ''·(∂r/∂x_j)·(∂r/∂x·V) + φ'·(Σ_k ∂²r/∂x_j∂x_k·V_k)

        For GP case where n1=1 (single query point):
        - X1[:, 0] = x_star (query point)
        - X2 = X_train (all training points)
        - Result shape: (1, n_train, nvars)
        """
        nvars = X1.shape[0]
        n1 = X1.shape[1]
        n2 = X2.shape[1]

        lenscale = self._log_lenscale.exp_values()  # (nvars,)

        # For simplicity, handle n1=1 case (typical for GP HVP)
        # Can be generalized later if needed
        if n1 != 1:
            raise NotImplementedError(
                "hvp_wrt_x1 currently only supports n1=1 (single query point)"
            )

        # X1 is (nvars, 1), X2 is (nvars, n2), direction is (nvars,)
        x_star = X1[:, 0]  # (nvars,)
        V = direction  # (nvars,)

        # Compute differences: (nvars, n2)
        diffs = x_star[:, None] - X2  # (nvars, n2)

        # Apply length scale scaling
        diffs_scaled = diffs / lenscale[:, None]  # (nvars, n2)

        # Compute distances
        r_squared = self._bkd.sum(diffs_scaled**2, axis=0)  # (n2,)
        r = self._bkd.sqrt(r_squared + 1e-12)

        # Get radial derivatives
        phi_prime, phi_double_prime = self.radial_derivatives(r)  # (n2,)

        # Compute ∂r/∂x_j = Δx_j / (r · ℓ_j²)
        # Shape: (nvars, n2)
        ell_squared = lenscale[:, None]**2
        r_inv = 1.0 / r  # (n2,)
        dr_dx = diffs / (r[None, :] * ell_squared)  # (nvars, n2)

        # Compute ∂r/∂x · V: (nvars, n2) · (nvars,) -> (n2,)
        dr_dot_V = self._bkd.einsum('ki,k->i', dr_dx, V)  # (n2,)

        # TERM 1: φ''·(∂r/∂x_j)·(∂r/∂x·V)
        #  Shape: (nvars, n2) * (n2,) -> (nvars, n2)
        term1 = dr_dx * (phi_double_prime[None, :] * dr_dot_V[None, :])

        # TERM 2: φ'·(∂²r/∂x_j∂x_k)·V_k
        # ∂²r/∂x_j∂x_k = -Δx_j·Δx_k/(r³·ℓ_j²·ℓ_k²) + δ_jk/(r·ℓ_j²)
        diffs_over_ell2 = diffs / ell_squared  # (nvars, n2)

        # Σ_k Δx_k/ℓ_k²·V_k for each training point: (n2,)
        diffs_scaled_dot_V = self._bkd.einsum('ki,k->i', diffs_over_ell2, V)

        r_inv3 = r_inv**3  # (n2,)

        # Σ_k ∂²r/∂x_j∂x_k·V_k, shape: (nvars, n2)
        d2r_dot_V = (-diffs_over_ell2 * (r_inv3[None, :] * diffs_scaled_dot_V[None, :]) +
                     V[:, None] * r_inv[None, :] / ell_squared)

        term2 = d2r_dot_V * phi_prime[None, :]  # (nvars, n2)

        hvp_2d = term1 + term2  # (nvars, n2)

        # Transpose to (1, n2, nvars) to match expected output shape (n1, n2, nvars)
        # Add n1 dimension, then transpose: (nvars, 1, n2) -> (1, n2, nvars)
        return self._bkd.transpose(hvp_2d[:, None, :], (1, 2, 0))

    def __repr__(self) -> str:
        """
        Return a string representation of the MaternKernel.

        Returns
        -------
        repr : str
            String representation of the MaternKernel.
        """
        return "{0}(nu={1}, {2}, bkd={3})".format(
            self.__class__.__name__,
            self._nu,
            str(self._hyp_list),
            self._bkd.__class__.__name__,
        )
