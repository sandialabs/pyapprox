"""MLBLUE estimator implementation.

This module provides the MLBLUEEstimator class, a specialized GroupACV
estimator for Multi-Level Best Linear Unbiased Estimation.
"""

from typing import TYPE_CHECKING, List

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
    ):
        super().__init__(
            stat,
            costs,
            reg_blue,
            model_subsets,
            asketch=asketch,
            use_pseudo_inv=use_pseudo_inv,
        )
        self._best_model_indices = self._bkd.arange(len(costs))

        # compute psi blocks once and store because they are independent
        # of the number of samples per partition/subset
        self._psi_blocks = self._compute_psi_blocks()
        self._psi_blocks_flat = self._bkd.hstack(
            [b.flatten()[:, None] for b in self._psi_blocks]
        )

        self._obj_jac = True

    def _compute_psi_blocks(self):
        submats = []
        for ii, subset in enumerate(self._subsets):
            R = self._restriction_matrix(subset)
            submat = self._bkd.multidot(
                (
                    R.T,
                    self._inv(self._stat._cov[np.ix_(subset, subset)]),
                    R,
                )
            )
            submats.append(submat)
        return submats

    def _psi_matrix(self, npartition_samples):
        psi = self._bkd.eye(self.nmodels() * self._stat.nstats()) * self._reg_blue
        psi += (self._psi_blocks_flat @ npartition_samples).reshape(
            (
                self.nmodels() * self._stat.nstats(),
                self.nmodels() * self._stat.nstats(),
            )
        )
        return psi

    def estimate_all_means(self, values_per_subset):
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

    def default_objective(self) -> MLBLUEObjective:
        """Return the default MLBLUE objective with analytical derivatives."""
        return MLBLUEObjective(self._bkd)
