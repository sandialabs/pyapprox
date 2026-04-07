"""Monte Carlo SAA strategy with growing sample subsets.

Implements sample average approximation (SAA) using the reparameterization
trick: base samples are fixed upfront and growing prefixes are returned
as ``tol`` shrinks. Sample count follows ``N = O(1/tol²)``, consistent
with the MC gradient error rate ``O(1/√N)``.
"""

import math
from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class MonteCarloSAAStrategy(Generic[Array]):
    """SAA strategy using random base samples.

    Fixes ``N_max`` base samples upfront and returns the first ``N_tol``
    as a function of the ROL tolerance:

        ``N_tol = min(N_max, ceil(scale_factor / tol²))``

    This ensures that the same base samples are always reused
    (reparameterization trick) and that smaller tolerances produce
    more samples.

    Parameters
    ----------
    base_samples : Array
        Pre-generated random samples. Shape ``(n_random_vars, N_max)``.
    bkd : Backend[Array]
        Computational backend.
    scale_factor : float, optional
        Controls the mapping from ``tol`` to sample count. Corresponds
        to an estimate of the gradient variance σ². Default ``1.0``.
    """

    def __init__(
        self,
        base_samples: Array,
        bkd: Backend[Array],
        scale_factor: float = 1.0,
    ) -> None:
        if scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be positive, got {scale_factor}"
            )
        self._base_samples = base_samples
        self._bkd = bkd
        self._scale_factor = scale_factor
        self._n_max: int = int(base_samples.shape[1])

    @classmethod
    def from_variance_bound(
        cls,
        base_samples: Array,
        bkd: Backend[Array],
        gradient_variance_bound: float,
    ) -> "MonteCarloSAAStrategy[Array]":
        """Construct with scale_factor derived from a gradient variance bound.

        For MC gradients, ``‖∇J_N − ∇J‖ ≈ σ/√N`` where σ is the gradient
        standard deviation. To achieve accuracy ``tol``, we need
        ``N ≈ σ²/tol²``. So ``scale_factor = gradient_variance_bound``
        (an estimate of σ²).

        Parameters
        ----------
        base_samples : Array
            Pre-generated random samples. Shape ``(n_random_vars, N_max)``.
        bkd : Backend[Array]
            Computational backend.
        gradient_variance_bound : float
            Estimate of σ² (gradient variance).
        """
        return cls(base_samples, bkd, scale_factor=gradient_variance_bound)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of random variables."""
        return int(self._base_samples.shape[0])

    def _nsamples_for_tol(self, tol: float) -> int:
        """Compute number of samples for a given tolerance."""
        if tol <= 0:
            return self._n_max
        n = math.ceil(self._scale_factor / (tol * tol))
        return max(1, min(n, self._n_max))

    def samples_and_weights(self, tol: float) -> Tuple[Array, Array]:
        """Return a prefix of base samples sized for the given tolerance.

        Parameters
        ----------
        tol : float
            Accuracy tolerance from ROL. Smaller means more samples.

        Returns
        -------
        Tuple[Array, Array]
            ``(samples, weights)`` with shapes
            ``(n_random_vars, N_tol)`` and ``(N_tol,)``.
            Weights are uniform ``1/N_tol``.
        """
        bkd = self._bkd
        n = self._nsamples_for_tol(tol)
        samples = self._base_samples[:, :n]
        weights = bkd.ones((n,)) / n
        return samples, weights
