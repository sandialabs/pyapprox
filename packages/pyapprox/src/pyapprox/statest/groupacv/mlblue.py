"""MLBLUE estimator implementation.

This module provides the MLBLUEEstimator class, a specialized GroupACV
estimator for Multi-Level Best Linear Unbiased Estimation.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from pyapprox.statest.groupacv.optimization import MLBLUEObjective
from pyapprox.statest.groupacv.variants import GroupACVEstimatorIS
from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    from pyapprox.statest.statistics import MultiOutputStatistic


class MLBLUEEstimator(GroupACVEstimatorIS[Array]):
    """Multi-Level Best Linear Unbiased Estimator.

    A specialized GroupACV estimator that uses independent sampling
    and precomputes psi blocks for efficient optimization.

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic object containing covariance information

    costs : Array
        The computational costs of each model

    reg_blue : float, optional
        Regularization parameter for BLUE. Default is 0.

    model_subsets : List[Array], optional
        List of model subsets. If None, all subsets are generated.

    asketch : Array, optional
        Sketch matrix for extracting statistics. If None, identity-like
        matrix extracting high-fidelity model statistics.

    use_pseudo_inv : bool, optional
        Whether to use pseudo-inverse. Default is True.
    """

    def __init__(
        self,
        stat: "MultiOutputStatistic[Array]",
        costs: Array,
        reg_blue: float = 0,
        model_subsets: List[Array] = None,
        asketch: Array = None,
        use_pseudo_inv: bool = True,
        known_quantities: Optional[Dict[Tuple[int, str], Array]] = None,
    ):
        from pyapprox.statest.statistics import (
            MultiOutputMeanAndVariance,
            MultiOutputVariance,
        )

        if isinstance(stat, (MultiOutputVariance, MultiOutputMeanAndVariance)):
            raise NotImplementedError(
                "MLBLUEEstimator precomputes per-group psi blocks assuming "
                "the covariance of per-group estimators is (1/m) * "
                "Cov(Q_l, Q_l'). This identity holds for mean estimation "
                "but not for variance or mean+variance estimation (which "
                "has additional 1/(m-1) terms). Use GroupACVEstimatorIS "
                "with known_quantities for IS-style variance estimation; "
                "it computes psi correctly through the stat class's "
                "_group_acv_sigma_block method."
            )
        super().__init__(
            stat,
            costs,
            reg_blue,
            model_subsets,
            asketch=asketch,
            use_pseudo_inv=use_pseudo_inv,
            known_quantities=known_quantities,
        )
        self._best_model_indices = self._bkd.arange(len(costs))

        # compute psi blocks once and store because they are independent
        # of the number of samples per partition/subset
        self._psi_blocks = self._compute_psi_blocks()
        self._psi_blocks_flat = self._bkd.hstack(
            [self._bkd.flatten(b)[:, None] for b in self._psi_blocks]
        )

        self._obj_jac = True

    def _compute_psi_blocks(self) -> List[Array]:
        # For mean-only estimation, covariance blocks are independent of which
        # means are known. For future variance estimation with known means,
        # C^k depends on whether mu_l is used for centering — this function
        # would need the known-value mask passed to the stat class.
        submats = []
        for ii, subset in enumerate(self._subsets):
            Rk = self._restriction_matrices[ii]
            submat = self._bkd.multidot(
                (
                    Rk,
                    self._inv(self._stat._cov[np.ix_(subset, subset)]),
                    Rk.T,
                )
            )
            submats.append(submat)
        return submats

    def _psi_matrix(self, npartition_samples: Array) -> Array:
        n = self._nT_stats
        psi = self._bkd.eye(n) * self._reg_blue
        psi += self._bkd.reshape(
            self._psi_blocks_flat @ npartition_samples, (n, n)
        )
        return psi

    def estimate_all_means(self, values_per_subset: List[Array]) -> Array:
        """Estimate means for all models.

        Parameters
        ----------
        values_per_subset : List[Array]
            Values for each subset

        Returns
        -------
        Array
            Estimated means for each model
        """
        asketch = self._bkd.copy(self._asketch)
        means = self._bkd.empty(self.nmodels())
        if self._stat.nstats() > 1:
            raise NotImplementedError(
                "Must adjust this function to work for multiple outputs"
            )
        for ii in range(self.nmodels()):
            self._asketch = self._bkd.full((self.nmodels()), 0.0)
            self._asketch[ii] = 1.0
            means[ii] = self._estimate(values_per_subset)
        self._asketch = asketch
        return means

    def default_objective(self) -> MLBLUEObjective[Array]:
        """Return the default MLBLUE objective with analytical derivatives."""
        return MLBLUEObjective(self._bkd)
