"""Gaussian (normal) likelihood for GP regression."""

import math
from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList, LogHyperParameter


class GaussianLikelihood(Generic[Array]):
    """Gaussian likelihood p(y|f) = N(y; f, sigma^2).

    Parameters
    ----------
    noise_std : float
        Initial noise standard deviation.
    noise_std_bounds : Tuple[float, float]
        Bounds for noise std (in original space).
    bkd : Backend[Array]
        Backend for numerical operations.
    fixed : bool
        Whether noise_std is fixed during optimization.
    """

    def __init__(
        self,
        noise_std: float,
        noise_std_bounds: Tuple[float, float],
        bkd: Backend[Array],
        fixed: bool = False,
    ) -> None:
        self._bkd = bkd
        self._noise = LogHyperParameter(
            "noise_std",
            1,
            noise_std,
            noise_std_bounds,
            bkd=bkd,
            fixed=fixed,
        )
        self._hyp_list = HyperParameterList([self._noise])

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def noise_std(self) -> Array:
        """Return noise standard deviation, shape (1,)."""
        return self._noise.exp_values()

    def noise_var(self) -> Array:
        """Return noise variance, shape (1,)."""
        s = self.noise_std()
        return s * s

    def log_prob(self, y: Array, f: Array) -> Array:
        """Pointwise log N(y; f, sigma^2), shape (1, n_points)."""
        bkd = self._bkd
        sigma2 = self.noise_var()[0]
        residual = y - f
        return bkd.reshape(
            -0.5 * (math.log(2.0 * math.pi) + bkd.log(sigma2)
                     + residual * residual / sigma2),
            (1, y.shape[-1]),
        )

    def expected_log_prob(
        self, y: Array, f_mean: Array, f_var: Array
    ) -> Array:
        """E_{N(f_mean, f_var)}[log N(y; f, sigma^2)].

        Closed-form for Gaussian likelihood:
        = -0.5 * (log(2*pi) + log(sigma^2) + ((y - mu)^2 + sigma_f^2) / sigma^2)

        Returns shape (1, n_points).
        """
        bkd = self._bkd
        sigma2 = self.noise_var()[0]
        residual = y - f_mean
        return bkd.reshape(
            -0.5 * (math.log(2.0 * math.pi) + bkd.log(sigma2)
                     + (residual * residual + f_var) / sigma2),
            (1, y.shape[-1]),
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __repr__(self) -> str:
        return f"GaussianLikelihood(noise_std={self._noise.exp_values()})"
