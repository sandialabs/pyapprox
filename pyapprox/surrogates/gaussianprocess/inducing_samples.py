"""
Inducing samples management for Variational Gaussian Processes.

This module provides the InducingSamples class which manages inducing
point locations and noise hyperparameters for variational GP inference.
"""

from typing import Generic, Tuple, Union

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
    LogHyperParameter,
)


class InducingSamples(Generic[Array]):
    """
    Manages inducing points and noise hyperparameter for variational GPs.

    The noise standard deviation is stored in log-space (via LogHyperParameter)
    to ensure positivity during optimization. Inducing point coordinates
    are stored directly (IdentityTransform).

    Parameters
    ----------
    nvars : int
        Number of input variables.
    ninducing_samples : int
        Number of inducing points.
    bkd : Backend[Array]
        Backend for numerical operations.
    inducing_samples : Array
        Initial inducing point locations, shape (nvars, ninducing_samples).
    noise_std : float
        Initial noise standard deviation.
    noise_std_bounds : Tuple[float, float]
        Bounds for noise standard deviation (in original space).
    inducing_sample_bounds : Array
        Bounds for inducing sample coordinates, shape
        (nvars * ninducing_samples, 2) or (2,) to broadcast.
    """

    def __init__(
        self,
        nvars: int,
        ninducing_samples: int,
        bkd: Backend[Array],
        inducing_samples: Array,
        noise_std: float,
        noise_std_bounds: Tuple[float, float],
        inducing_sample_bounds: Union[Array, Tuple[float, float]],
    ) -> None:
        self._bkd = bkd
        self._nvars = nvars
        self._ninducing_samples = ninducing_samples

        # Validate inducing_samples shape
        if inducing_samples.shape != (nvars, ninducing_samples):
            raise ValueError(
                f"inducing_samples must have shape ({nvars}, {ninducing_samples}), "
                f"got {inducing_samples.shape}"
            )

        # Noise hyperparameter (log-space for positivity)
        self._noise = LogHyperParameter(
            "noise",
            1,
            noise_std,
            noise_std_bounds,
            bkd=bkd,
        )

        # Parse inducing sample bounds
        n_flat = nvars * ninducing_samples
        inducing_sample_bounds_arr = bkd.atleast_1d(bkd.asarray(inducing_sample_bounds))
        if inducing_sample_bounds_arr.ndim == 1:
            if inducing_sample_bounds_arr.shape[0] != 2:
                raise ValueError(
                    "inducing_sample_bounds must have shape (2,) or "
                    f"({n_flat}, 2), got {inducing_sample_bounds_arr.shape}"
                )
            inducing_sample_bounds_arr = bkd.tile(
                bkd.reshape(inducing_sample_bounds_arr, (1, 2)),
                (n_flat, 1),
            )

        # Inducing point coordinates hyperparameter
        self._inducing_samples = HyperParameter(
            "inducing_samples",
            n_flat,
            bkd.flatten(inducing_samples),
            inducing_sample_bounds_arr,
            bkd=bkd,
        )

        self._hyp_list = HyperParameterList([self._noise, self._inducing_samples])

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list."""
        return self._hyp_list

    def get_samples(self) -> Array:
        """
        Get inducing point locations.

        Returns
        -------
        Array
            Inducing samples, shape (nvars, ninducing_samples).
        """
        return self._bkd.reshape(
            self._inducing_samples.get_values(),
            (self._nvars, self._ninducing_samples),
        )

    def get_noise(self) -> Array:
        """
        Get noise standard deviation as an array (preserves autograd graph).

        Returns
        -------
        Array
            Noise standard deviation, shape (1,).
        """
        return self._noise.exp_values()

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def ninducing_samples(self) -> int:
        """Return the number of inducing samples."""
        return self._ninducing_samples

    def __repr__(self) -> str:
        return (
            f"InducingSamples(ninducing={self._ninducing_samples}, nvars={self._nvars})"
        )
