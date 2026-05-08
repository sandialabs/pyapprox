"""PCE-based sensitivity analysis.

This module provides sensitivity analysis using Polynomial Chaos Expansions.
It leverages the orthonormality of the PCE basis to compute Sobol indices
analytically from the expansion coefficients.
"""

from typing import Generic, List, Tuple

from pyapprox.sensitivity.variance_based.base import (
    VarianceBasedSensitivityAnalysis,
)
from pyapprox.surrogates.affine.expansions.pce_statistics import (
    interaction_sobol_indices,
    main_effect_sobol_indices,
    total_sobol_indices,
)
from pyapprox.surrogates.affine.expansions.pce_statistics import (
    mean as pce_mean,
)
from pyapprox.surrogates.affine.expansions.pce_statistics import (
    variance as pce_variance,
)
from pyapprox.surrogates.affine.protocols import PCEStatisticsProtocol
from pyapprox.util.backends.protocols import Array


class PolynomialChaosSensitivityAnalysis(
    VarianceBasedSensitivityAnalysis[Array], Generic[Array]
):
    """Sensitivity analysis using Polynomial Chaos Expansion.

    Computes Sobol indices analytically from PCE coefficients by exploiting
    the orthonormality of the polynomial basis. This is computationally
    efficient as it requires only the fitted PCE, not additional sampling.

    The PCE must use an orthonormal basis with respect to the input
    distribution for the computed indices to be valid.

    Examples
    --------
    >>> from pyapprox.sensitivity.variance_based import (
    ...     PolynomialChaosSensitivityAnalysis,
    ... )
    >>> # Assuming pce is a fitted PolynomialChaosExpansion
    >>> sa = PolynomialChaosSensitivityAnalysis(pce)
    >>> main = sa.main_effects()
    >>> total = sa.total_effects()
    """

    def __init__(self, pce: PCEStatisticsProtocol[Array]) -> None:
        """Initialize PCE sensitivity analysis.

        Parameters
        ----------
        pce : PCEStatisticsProtocol[Array]
            A fitted PCE object with orthonormal basis.
        """
        if not isinstance(pce, PCEStatisticsProtocol):
            raise TypeError(
                f"pce must satisfy PCEStatisticsProtocol, got {type(pce).__name__}"
            )
        super().__init__(pce.nvars(), pce.bkd())
        self._pce = pce
        self._main_effects: Array | None = None
        self._total_effects: Array | None = None

    def compute(self) -> None:
        """Compute sensitivity indices from the PCE.

        This method is called automatically when accessing results,
        but can be called explicitly to precompute all indices.
        """
        self._main_effects = main_effect_sobol_indices(self._pce)
        self._total_effects = total_sobol_indices(self._pce)

    def mean(self) -> Array:
        """Return the mean of the PCE output.

        Returns
        -------
        Array
            Shape (nqoi,) - mean for each QoI.
        """
        return pce_mean(self._pce)

    def variance(self) -> Array:
        """Return the variance of the PCE output.

        Returns
        -------
        Array
            Shape (nqoi,) - variance for each QoI.
        """
        return pce_variance(self._pce)

    def main_effects(self) -> Array:
        """Return first-order (main effect) Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - main effect index for each variable and QoI.
        """
        if self._main_effects is None:
            self._main_effects = main_effect_sobol_indices(self._pce)
        return self._main_effects

    def total_effects(self) -> Array:
        """Return total-order Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - total effect index for each variable and QoI.
        """
        if self._total_effects is None:
            self._total_effects = total_sobol_indices(self._pce)
        return self._total_effects

    def sobol_indices(
        self, variable_sets: List[Tuple[int, ...]] | None = None
    ) -> Array:
        """Return Sobol indices for specified interaction terms.

        Parameters
        ----------
        variable_sets : List[Tuple[int, ...]] | None
            Variable index sets to compute interactions for. If None,
            uses the interaction terms set via set_interaction_terms_of_interest().

        Returns
        -------
        Array
            Shape (nterms, nqoi) - Sobol index for each interaction term.
        """
        if variable_sets is None:
            # Convert interaction_terms matrix to list of tuples
            interaction_terms = self.interaction_terms()
            variable_sets = []
            for jj in range(interaction_terms.shape[1]):
                active = self._bkd.where(interaction_terms[:, jj] > 0)[0]
                variable_sets.append(tuple(int(v) for v in self._bkd.to_numpy(active)))

        return interaction_sobol_indices(self._pce, variable_sets)

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        return self._pce.nqoi()
