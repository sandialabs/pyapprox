r"""The blended relaxation of an observation.

Datum :math:`i` is observed as

.. math::

    z_i = \sqrt{w_i}\,(g_i + e_i) + \sqrt{\nu_i}\,\epsilon_i, \qquad
    \nu_i = (1 - w_i)\, s_i^2,

with :math:`\epsilon` independent of everything and reference variances
:math:`s_i^2 > 0`, by default the noise variances :math:`\sigma_i^2`. Its
noise variance is :math:`w_i \sigma_i^2 + (1 - w_i) s_i^2`, which is
:math:`\sigma_i^2` for every :math:`w` when :math:`s = \sigma`: only the
signal is devalued. At :math:`w_i = 1` the datum is observed as it is; at
:math:`w_i = 0` it is independent noise, so the sensor is truly removed,
even when the noise is correlated across sensors. For diagonal noise and
:math:`s = \sigma` this is the same distribution as precision weighting,
noise variance :math:`\sigma_i^2 / w_i`, written without its singularity at
:math:`w_i = 0`.
"""

from typing import Generic

from pyapprox.probability.protocols import CovarianceOperatorProtocol
from pyapprox.util.backends.protocols import Array, Backend


class BlendedObservation(Generic[Array]):
    """``nu(w) = (1 - w) s^2``.

    Parameters
    ----------
    ref_variances : Array
        Reference variances ``s^2``, positive. Shape: (nobs, 1)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, ref_variances: Array, bkd: Backend[Array]) -> None:
        if ref_variances.ndim != 2 or ref_variances.shape[1] != 1:
            raise ValueError(
                "ref_variances must have shape (nobs, 1), got "
                f"{tuple(ref_variances.shape)}"
            )
        if bkd.any_bool(ref_variances <= 0.0):
            raise ValueError(
                "ref_variances must be positive, so a zero weight leaves "
                "independent noise and the observation stays nonsingular"
            )
        self._s2 = ref_variances
        self._bkd = bkd

    @classmethod
    def from_noise(
        cls, noise: CovarianceOperatorProtocol[Array]
    ) -> "BlendedObservation[Array]":
        """The default relaxation, ``s^2`` the diagonal of the noise covariance."""
        bkd = noise.bkd()
        nobs = noise.nvars()
        return cls(bkd.reshape(bkd.diag(noise.covariance()), (nobs, 1)), bkd)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observations."""
        return int(self._s2.shape[0])

    def ref_variances(self) -> Array:
        """Reference variances ``s^2``. Shape: (nobs, 1)"""
        return self._s2

    def variances(self, weights: Array) -> Array:
        """``nu = (1 - w) s^2``. Shape: (nobs, 1)"""
        self._check_weights(weights)
        return (1.0 - weights) * self._s2

    def variances_jacobian_diagonal(self, weights: Array) -> Array:
        """``d nu_i / d w_i = -s_i^2``. Shape: (nobs, 1)"""
        self._check_weights(weights)
        return -self._s2

    def _check_weights(self, weights: Array) -> None:
        nobs = self.nobs()
        if tuple(weights.shape) != (nobs, 1):
            raise ValueError(
                f"weights must have shape ({nobs}, 1), got {tuple(weights.shape)}"
            )
        if self._bkd.any_bool(weights < 0.0) or self._bkd.any_bool(weights > 1.0):
            raise ValueError("weights must lie in [0, 1]")
