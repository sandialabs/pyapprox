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
