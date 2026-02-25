"""Protocols for sensitivity analysis.

This module defines protocols for sensitivity analysis classes, enabling
duck typing with runtime type checking.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SensitivityAnalysisProtocol(Protocol, Generic[Array]):
    """Protocol for sensitivity analysis results.

    All sensitivity analysis classes should implement this protocol
    to provide a consistent interface for accessing sensitivity indices.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend used for array operations."""
        ...

    def nvars(self) -> int:
        """Return the number of input variables."""
        ...

    def main_effects(self) -> Array:
        """Return first-order (main effect) Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - main effect index for each variable and QoI.
        """
        ...

    def total_effects(self) -> Array:
        """Return total-order Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - total effect index for each variable and QoI.
        """
        ...


@runtime_checkable
class SensitivityAnalysisWithSobolIndicesProtocol(
    SensitivityAnalysisProtocol[Array], Protocol, Generic[Array]
):
    """Extended protocol that includes higher-order Sobol indices."""

    def sobol_indices(self) -> Array:
        """Return Sobol indices for interaction terms.

        Returns
        -------
        Array
            Shape (nterms, nqoi) - Sobol index for each interaction term.
        """
        ...

    def interaction_terms(self) -> Array:
        """Return the interaction terms for which Sobol indices are computed.

        Returns
        -------
        Array
            Shape (nvars, nterms) - binary indicator of which variables
            are active in each interaction term.
        """
        ...


@runtime_checkable
class SensitivityAnalysisWithMomentsProtocol(
    SensitivityAnalysisProtocol[Array], Protocol, Generic[Array]
):
    """Extended protocol that includes mean and variance."""

    def mean(self) -> Array:
        """Return the mean of the output.

        Returns
        -------
        Array
            Shape (nqoi,) - mean for each QoI.
        """
        ...

    def variance(self) -> Array:
        """Return the variance of the output.

        Returns
        -------
        Array
            Shape (nqoi,) - variance for each QoI.
        """
        ...
