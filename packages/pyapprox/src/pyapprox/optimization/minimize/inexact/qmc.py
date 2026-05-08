"""Quasi-Monte Carlo SAA strategy with growing sample subsets.

Implements sample average approximation (SAA) using Sobol or Halton
base samples. QMC achieves better convergence rates than MC (up to
``O(1/N)`` vs ``O(1/√N)``), so the tol-to-N mapping uses a configurable
exponent: ``N = ceil(scale_factor / tol^exponent)``.
"""

import math
from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class QMCSAAStrategy(Generic[Array]):
    """SAA strategy using quasi-Monte Carlo base samples.

    Fixes ``N_max`` base samples (e.g., Sobol or Halton) upfront and
    returns the first ``N_tol`` as a function of the ROL tolerance:

        ``N_tol = min(N_max, ceil(scale_factor / tol^tol_exponent))``

    The configurable ``tol_exponent`` reflects QMC's superior convergence
    rate compared to MC.

    Parameters
    ----------
    base_samples : Array
        Pre-generated QMC samples. Shape ``(n_random_vars, N_max)``.
    bkd : Backend[Array]
        Computational backend.
    scale_factor : float, optional
        Controls the mapping from ``tol`` to sample count. Default ``1.0``.
    tol_exponent : float, optional
        Exponent in the ``tol``-to-N mapping. Default ``1.0`` reflects
        QMC's ``O(1/N)`` convergence. Set to ``2.0`` to match MC behavior.
    """

    def __init__(
        self,
        base_samples: Array,
        bkd: Backend[Array],
        scale_factor: float = 1.0,
        tol_exponent: float = 1.0,
    ) -> None:
        if scale_factor <= 0:
            raise ValueError(
                f"scale_factor must be positive, got {scale_factor}"
            )
        if tol_exponent <= 0:
            raise ValueError(
                f"tol_exponent must be positive, got {tol_exponent}"
            )
        self._base_samples = base_samples
        self._bkd = bkd
        self._scale_factor = scale_factor
        self._tol_exponent = tol_exponent
        self._n_max: int = int(base_samples.shape[1])

    @classmethod
    def from_variance_bound(
        cls,
        base_samples: Array,
        bkd: Backend[Array],
        gradient_variance_bound: float,
        tol_exponent: float = 1.0,
    ) -> "QMCSAAStrategy[Array]":
        """Construct with scale_factor derived from a gradient variance bound.

        Parameters
        ----------
        base_samples : Array
            Pre-generated QMC samples. Shape ``(n_random_vars, N_max)``.
        bkd : Backend[Array]
            Computational backend.
        gradient_variance_bound : float
            Estimate of σ² (gradient variance).
        tol_exponent : float, optional
            Exponent in the ``tol``-to-N mapping. Default ``1.0``.
        """
        return cls(
            base_samples,
            bkd,
            scale_factor=gradient_variance_bound,
            tol_exponent=tol_exponent,
        )

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
        n = math.ceil(self._scale_factor / (tol ** self._tol_exponent))
        return int(max(1, min(n, self._n_max)))

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
