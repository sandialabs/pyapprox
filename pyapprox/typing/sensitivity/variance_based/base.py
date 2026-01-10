"""Base class for variance-based sensitivity analysis.

This module provides the abstract base class for all variance-based
(Sobol) sensitivity analysis methods.
"""

from abc import ABC, abstractmethod
from itertools import combinations
from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.indices import (
    HyperbolicIndexGenerator,
    hash_index,
    argsort_indices_lexiographically,
)


class VarianceBasedSensitivityAnalysis(ABC, Generic[Array]):
    """Abstract base class for variance-based sensitivity analysis.

    This class provides common functionality for computing Sobol indices
    from different surrogate models (PCE, GP, sparse grids) or samples.

    Parameters
    ----------
    nvars : int
        Number of input variables.
    bkd : Backend[Array]
        Backend for array operations.
    """

    def __init__(self, nvars: int, bkd: Backend[Array]) -> None:
        self._nvars = nvars
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the backend used for array operations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def isotropic_interaction_terms(self, order: int) -> Array:
        """Generate interaction terms up to a given order.

        Creates binary indicator arrays for all variable interactions
        up to the specified order (e.g., order=2 gives main effects
        and pairwise interactions).

        Parameters
        ----------
        order : int
            Maximum interaction order (1 = main effects only,
            2 = main + pairwise, etc.).

        Returns
        -------
        Array
            Shape (nvars, nterms) - binary indicator of which variables
            are active in each interaction term.
        """
        gen = HyperbolicIndexGenerator(
            self.nvars(), order, 1.0, bkd=self._bkd
        )
        interaction_terms = gen.get_indices()
        # Keep only terms where max degree is 1 (binary indicators)
        mask = self._bkd.where(
            self._bkd.max(interaction_terms, axis=0) == 1
        )[0]
        interaction_terms = interaction_terms[:, mask]
        return interaction_terms

    def _default_interaction_terms(self) -> Array:
        """Return default interaction terms (main + pairwise)."""
        return self.isotropic_interaction_terms(2)

    def set_interaction_terms_of_interest(
        self, interaction_terms: Array
    ) -> None:
        """Set the interaction terms for which to compute Sobol indices.

        Parameters
        ----------
        interaction_terms : Array
            Shape (nvars, nterms) - binary indicator of which variables
            are active in each interaction term. Must include all main
            effects (all single-variable terms).

        Raises
        ------
        ValueError
            If interaction_terms does not include all main effect indices.
        """
        # Check that all main effects are included
        main_effect_indices = interaction_terms[
            :, interaction_terms.sum(axis=0) == 1
        ]
        if main_effect_indices.shape[1] != self.nvars():
            raise ValueError(
                "interaction_terms must contain all main effect indices"
            )
        self._interaction_terms = interaction_terms

    def interaction_terms(self) -> Array:
        """Return the interaction terms for which Sobol indices are computed.

        Returns
        -------
        Array
            Shape (nvars, nterms) - binary indicator of which variables
            are active in each interaction term.
        """
        if not hasattr(self, "_interaction_terms"):
            self._interaction_terms = self._default_interaction_terms()
        return self._interaction_terms

    def _correct_interaction_variance_ratios(
        self, interaction_variances: Array
    ) -> Array:
        """Correct variance ratios to get true Sobol indices.

        Subtracts contributions from lower-dimensional terms from each
        interaction value. For example, if R_ij is the interaction variance,
        the Sobol index S_ij satisfies: R_ij = S_i + S_j + S_ij

        Parameters
        ----------
        interaction_variances : Array
            Shape (nterms, nqoi) - uncorrected variance ratios.

        Returns
        -------
        Array
            Shape (nterms, nqoi) - corrected Sobol indices.
        """
        idx = argsort_indices_lexiographically(
            self._interaction_terms, self._bkd
        )
        sobol_indices = self._bkd.copy(interaction_variances)
        sobol_indices_dict: dict[str, int] = {}

        for ii in range(idx.shape[0]):
            index = self._interaction_terms[:, idx[ii]]
            active_vars = self._bkd.where(index > 0)[0]
            nactive_vars = int(index.sum())

            # Store mapping from active variables to index
            key = hash_index(active_vars, self._bkd)
            sobol_indices_dict[key] = int(idx[ii])

            # Subtract lower-order contributions
            if nactive_vars > 1:
                for jj in range(nactive_vars - 1):
                    for subset in combinations(active_vars, jj + 1):
                        subset_key = hash_index(
                            self._bkd.asarray(subset), self._bkd
                        )
                        sobol_indices[idx[ii]] -= sobol_indices[
                            sobol_indices_dict[subset_key]
                        ]

        return sobol_indices

    @abstractmethod
    def main_effects(self) -> Array:
        """Return first-order (main effect) Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - main effect index for each variable and QoI.
        """
        ...

    @abstractmethod
    def total_effects(self) -> Array:
        """Return total-order Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - total effect index for each variable and QoI.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nvars={self._nvars})"
