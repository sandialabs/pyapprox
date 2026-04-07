"""
Brute-force solver for discrete OED problems.

Enumerates all k-subsets of n candidate observations and evaluates
EIG for each subset, returning the design with maximum EIG.
"""

from itertools import combinations
from typing import Generic, List, Optional, Tuple

import numpy as np

from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.util.backends.protocols import Array, Backend


# TODO: this still uses weight based formulation of likelihood
# We need a version that works with  weights (0,1,2,3) and removes
# design location from liklihood computation if weight is zero (so
# covariance matrix is not singular) and adds multiple repititions
# at the same location. That new class and this should be consistent
# when no weights are zero. Should they be consistent if weight is closed
# to zero numerically, ie.. when we use small positive weight for
# unselected to avoid divide by zero (below)
class BruteForceKLOEDSolver(Generic[Array]):
    """Brute-force solver for discrete OED.

    Enumerates all k-subsets of n candidate observation locations
    and returns the subset with maximum expected information gain.

    Parameters
    ----------
    objective : KLOEDObjective[Array]
        The KL-OED objective function.
    """

    def __init__(self, objective: KLOEDObjective[Array]) -> None:
        self._objective = objective
        self._bkd = objective.bkd()
        self._nobs = objective.nobs()
        self._all_indices: List[Tuple[int, ...]] = []
        self._all_eigs: List[float] = []

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of candidate observation locations."""
        return self._nobs

    def _indices_to_weights(
        self, indices: Tuple[int, ...], eps: float = 1e-10
    ) -> Array:
        """Convert subset indices to design weights.

        Parameters
        ----------
        indices : Tuple[int, ...]
            Indices of selected observations.
        eps : float
            Small positive weight for unselected observations to avoid
            numerical issues with zero weights.

        Returns
        -------
        Array
            Design weights. Shape: (nobs, 1)
            Selected indices have weight 1.0, others have weight eps.
        """
        # Use small positive weight for unselected to avoid divide by zero
        weights = self._bkd.ones((self._nobs, 1)) * eps
        for idx in indices:
            weights[idx, 0] = 1.0
        return weights

    def solve(self, k: int, store_all: bool = False) -> Tuple[Array, float, List[int]]:
        """Find the optimal k-subset design.

        Parameters
        ----------
        k : int
            Number of observations to select.
        store_all : bool, optional
            If True, store all evaluated indices and EIG values for later
            inspection via all_indices() and all_eigs(). Default False.

        Returns
        -------
        optimal_weights : Array
            Optimal design weights. Shape: (nobs, 1)
        optimal_eig : float
            Expected information gain for optimal design.
        optimal_indices : List[int]
            Indices of selected observations.

        Raises
        ------
        ValueError
            If k <= 0 or k > nobs.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > self._nobs:
            raise ValueError(f"k={k} exceeds number of observations {self._nobs}")

        best_weights = None
        best_eig = -np.inf
        best_indices: List[int] = []

        if store_all:
            self._all_indices = []
            self._all_eigs = []

        # Enumerate all k-subsets
        for indices in combinations(range(self._nobs), k):
            weights = self._indices_to_weights(indices)
            eig = self._objective.expected_information_gain(weights)

            if store_all:
                self._all_indices.append(indices)
                self._all_eigs.append(eig)

            if eig > best_eig:
                best_eig = eig
                best_weights = weights
                best_indices = list(indices)

        if best_weights is None:
            # Should not happen, but handle gracefully
            best_weights = self._bkd.ones((self._nobs, 1)) / self._nobs
            best_eig = self._objective.expected_information_gain(best_weights)

        return best_weights, best_eig, best_indices

    def solve_all_k(
        self, k_min: int = 1, k_max: Optional[int] = None
    ) -> List[Tuple[int, Array, float, List[int]]]:
        """Find optimal designs for all subset sizes.

        Parameters
        ----------
        k_min : int
            Minimum subset size. Default 1.
        k_max : int, optional
            Maximum subset size. Default nobs.

        Returns
        -------
        List[Tuple[int, Array, float, List[int]]]
            List of (k, weights, eig, indices) for each k.
        """
        if k_max is None:
            k_max = self._nobs

        results = []
        for k in range(k_min, k_max + 1):
            weights, eig, indices = self.solve(k)
            results.append((k, weights, eig, indices))

        return results

    def n_combinations(self, k: int) -> int:
        """Compute number of k-subsets.

        Parameters
        ----------
        k : int
            Subset size.

        Returns
        -------
        int
            Number of k-subsets of n elements.
        """
        from math import comb

        return comb(self._nobs, k)

    def all_indices(self) -> List[Tuple[int, ...]]:
        """Get all evaluated design indices.

        Only populated after solve() is called with store_all=True.

        Returns
        -------
        List[Tuple[int, ...]]
            List of index tuples for all evaluated designs.
        """
        return self._all_indices

    def all_eigs(self) -> List[float]:
        """Get all evaluated EIG values.

        Only populated after solve() is called with store_all=True.

        Returns
        -------
        List[float]
            List of EIG values for all evaluated designs, in same order
            as all_indices().
        """
        return self._all_eigs
