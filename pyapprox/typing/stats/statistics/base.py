"""Base class for statistics implementations.

Provides common functionality for all statistic types.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend


class AbstractStatistic(ABC, Generic[Array]):
    """Abstract base class for statistics.

    Provides common functionality for computing statistics from model samples
    and their covariance structure for multifidelity estimation.

    Parameters
    ----------
    nqoi : int
        Number of quantities of interest.
    bkd : Backend[Array]
        Computational backend.

    Attributes
    ----------
    _nqoi : int
        Number of quantities of interest.
    _bkd : Backend[Array]
        Computational backend.
    _cov : Optional[Array]
        Pilot covariance matrix (set via set_pilot_quantities).
    """

    def __init__(self, nqoi: int, bkd: Backend[Array]):
        self._nqoi = nqoi
        self._bkd = bkd
        self._cov: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nqoi(self) -> int:
        """Return number of quantities of interest."""
        return self._nqoi

    @abstractmethod
    def nstats(self) -> int:
        """Return total number of scalar statistics."""
        ...

    @abstractmethod
    def sample_estimate(self, values: Array) -> Array:
        """Compute sample estimate of the statistic.

        Parameters
        ----------
        values : Array
            Model output samples. Shape: (nsamples, nqoi)

        Returns
        -------
        Array
            Estimated statistic values. Shape: (nstats,)
        """
        ...

    @abstractmethod
    def min_nsamples(self) -> int:
        """Return minimum number of samples needed."""
        ...

    def cov(self) -> Array:
        """Return the pilot covariance matrix.

        Returns
        -------
        Array
            Covariance matrix. Shape depends on statistic type.

        Raises
        ------
        ValueError
            If pilot quantities have not been set.
        """
        if self._cov is None:
            raise ValueError(
                "Pilot covariance not set. Call set_pilot_quantities() first."
            )
        return self._cov

    @abstractmethod
    def compute_pilot_quantities(
        self, pilot_values: List[Array]
    ) -> Tuple[Array, ...]:
        """Compute pilot quantities from pilot samples.

        Parameters
        ----------
        pilot_values : List[Array]
            List of model outputs from pilot samples.
            pilot_values[m] has shape (npilot, nqoi) for model m.

        Returns
        -------
        Tuple[Array, ...]
            Pilot quantities (e.g., covariance matrix, means).
        """
        ...

    @abstractmethod
    def set_pilot_quantities(self, *args) -> None:
        """Set pilot quantities directly.

        Parameters
        ----------
        *args
            Pilot quantities (statistic-specific).
        """
        ...

    @abstractmethod
    def high_fidelity_estimator_covariance(self, nhf_samples: int) -> Array:
        """Compute covariance of the high-fidelity sample estimator.

        Parameters
        ----------
        nhf_samples : int
            Number of high-fidelity samples.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nstats, nstats)
        """
        ...

    def _validate_values(self, values: Array, min_samples: int = 1) -> None:
        """Validate input values array.

        Parameters
        ----------
        values : Array
            Values to validate. Shape: (nsamples, nqoi)
        min_samples : int
            Minimum required samples.

        Raises
        ------
        ValueError
            If values has wrong shape or insufficient samples.
        """
        if values.ndim != 2:
            raise ValueError(
                f"Values must be 2D, got shape {values.shape}"
            )
        if values.shape[1] != self._nqoi:
            raise ValueError(
                f"Values must have {self._nqoi} columns, got {values.shape[1]}"
            )
        if values.shape[0] < min_samples:
            raise ValueError(
                f"Need at least {min_samples} samples, got {values.shape[0]}"
            )
