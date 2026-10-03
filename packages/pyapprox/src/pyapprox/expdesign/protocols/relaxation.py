"""Protocol for how relaxed design weights enter an observation.

A relaxed design observes ``z_i = sqrt(w_i) y_i + sqrt(nu_i) eps_i`` with
``eps`` independent of everything. The relaxation chooses ``nu(w)``, the
variances of that independent part; the weights themselves scale the
datum. Objectives are exact at binary weights when ``nu_i = 0`` at
``w_i = 1`` (the datum is observed as it is) and ``nu_i > 0`` at
``w_i = 0`` (the datum is replaced by independent noise).
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ObservationRelaxationProtocol(Protocol, Generic[Array]):
    """Maps design weights to independent-noise variances.

    Methods
    -------
    bkd()
        Get the computational backend.
    nobs()
        Number of observations.
    variances(weights)
        ``nu(w)``.
    variances_jacobian_diagonal(weights)
        ``d nu_i / d w_i``; ``nu_i`` depends on ``w_i`` only.
    """

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        ...

    def nobs(self) -> int:
        """Number of observations."""
        ...

    def variances(self, weights: Array) -> Array:
        """Independent-noise variances ``nu(w)``.

        Parameters
        ----------
        weights : Array
            Design weights in [0, 1]. Shape: (nobs, 1)

        Returns
        -------
        Array
            Non-negative, and positive wherever ``w_i = 0``. Shape: (nobs, 1)
        """
        ...

    def variances_jacobian_diagonal(self, weights: Array) -> Array:
        """``d nu_i / d w_i``, the diagonal of the Jacobian. Shape: (nobs, 1)"""
        ...
