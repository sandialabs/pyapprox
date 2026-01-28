"""Sobol adaptive sampler for adaptive GP."""

from typing import Generic

from pyapprox.typing.surrogates.kernels.protocols import KernelProtocol
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.quadrature import SobolSampler


class SobolAdaptiveSampler(Generic[Array]):
    """Adaptive sampler using Sobol sequences.

    Ignores kernel information — simply returns quasi-random points.
    Operates in scaled space.

    Parameters
    ----------
    nvars : int
        Number of input variables.
    bkd : Backend[Array]
        Backend for numerical computations.
    scaled_bounds : Array | None
        Bounds of shape (nvars, 2) in scaled space. Default: [0, 1]^d.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        scaled_bounds: Array | None = None,
        seed: int = 42,
    ) -> None:
        self._nvars = nvars
        self._bkd = bkd
        self._sobol = SobolSampler(
            nvars, bkd, scramble=True, seed=seed,
        )
        if scaled_bounds is not None:
            if scaled_bounds.shape != (nvars, 2):
                raise ValueError(
                    f"scaled_bounds must have shape ({nvars}, 2), "
                    f"got {scaled_bounds.shape}"
                )
            self._lower = scaled_bounds[:, 0]
            self._upper = scaled_bounds[:, 1]
        else:
            self._lower = bkd.zeros((nvars,))
            self._upper = bkd.ones((nvars,))

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def select_samples(self, nsamples: int) -> Array:
        """Select new sample locations using Sobol sequence.

        Parameters
        ----------
        nsamples : int
            Number of samples to select.

        Returns
        -------
        samples : Array
            Selected samples of shape (nvars, nsamples) in scaled space.
        """
        pts, _ = self._sobol.sample(nsamples)
        return (
            self._lower[:, None]
            + pts * (self._upper - self._lower)[:, None]
        )

    def set_kernel(self, kernel: KernelProtocol[Array]) -> None:
        """No-op: Sobol sampler does not use kernel information."""

    def add_additional_training_samples(
        self, new_samples: Array
    ) -> None:
        """No-op: Sobol sampler does not track training samples."""
