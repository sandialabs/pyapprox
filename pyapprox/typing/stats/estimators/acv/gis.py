"""Generalized Independent Samples (GIS) estimator.

GIS (Generalized Independent Samples) uses union semantics for
sample allocation. Unlike GMF (hierarchical inclusion) or GRD
(disjoint), GIS merges shared and model-specific samples via
the maximum operator in allocation matrix construction.

The allocation matrix uses:
    mat[:, 2*ii+1] = maximum(mat[:, 2*ii], mat[:, 2*ii+1])

This creates a union of sample sets where each model evaluates
on both shared samples (with its recursion parent) AND its own
samples.
"""

from typing import Generic, List, Callable, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.acv.base import ACVEstimator
from pyapprox.typing.stats.allocation.matrices import get_allocation_matrix_gis


class GISEstimator(ACVEstimator[Array], Generic[Array]):
    """Generalized Independent Samples (GIS) estimator.

    GIS uses the ACV framework with a specific allocation matrix structure
    that merges shared and model-specific samples via the maximum operator.
    This creates union semantics for sample allocation.

    The estimator is:
        Q_GIS = Q_0 + sum_m eta_m * (mu_m - Q_m)

    where:
    - Q_0 is the HF sample mean
    - Q_m are LF sample means on shared samples
    - mu_m are LF sample means on all LF samples
    - eta_m are optimal weights

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.
    recursion_index : Array, optional
        Recursion index. Shape: (nmodels-1,)
        If None, uses MFMC structure (all coupled with HF).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([10.0, 1.0, 0.1])
    >>> gis = GISEstimator(stat, costs, bkd)
    >>> gis.allocate_samples(target_cost=100.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
        recursion_index: Optional[Array] = None,
    ):
        # Initialize base class with recursion index
        super().__init__(stat, costs, bkd, recursion_index)

    def get_allocation_matrix(self) -> Array:
        """Return the GIS allocation matrix.

        GIS uses a specific allocation matrix structure with:
        - Shape: (nmodels, 2*nmodels)
        - Column 2*i: shared samples for model i (from recursion parent)
        - Column 2*i+1: all samples for model i (after maximum merge)

        The maximum merge in step 3 creates union semantics:
            mat[:, 2*ii+1] = maximum(mat[:, 2*ii], mat[:, 2*ii+1])

        Returns
        -------
        Array
            Allocation matrix. Shape: (nmodels, 2*nmodels)
        """
        if self._allocation_mat is None:
            self._allocation_mat = get_allocation_matrix_gis(
                self.nmodels(), self._recursion_index, self._bkd
            )
        return self._allocation_mat

    def npartitions(self) -> int:
        """Return number of sample partitions.

        GIS uses 2*nmodels partitions (different from standard ACV).
        """
        return 2 * self.nmodels()

    def _solve_partition_allocation(
        self, alloc_mat: Array, nsamples: Array
    ) -> Array:
        """Solve for partition samples from model samples.

        For GIS, the partition structure is:
        - Even columns (2*i): shared samples
        - Odd columns (2*i+1): all samples for model i

        Parameters
        ----------
        alloc_mat : Array
            Allocation matrix. Shape: (nmodels, 2*nmodels)
        nsamples : Array
            Samples per model. Shape: (nmodels,)

        Returns
        -------
        Array
            Samples per partition. Shape: (2*nmodels,)
        """
        bkd = self._bkd
        nmodels = self.nmodels()

        # Build partition samples
        # Column 0: unused (HF shared with itself = 0)
        # Column 1: HF all samples = n_0
        # Column 2*i: shared samples between model i and its recursion parent
        # Column 2*i+1: all samples for model i
        parts = []

        # Columns 0-1 for HF
        parts.append(bkd.asarray([0.0]))  # Column 0: unused
        parts.append(bkd.reshape(nsamples[0], (1,)))  # Column 1: HF samples

        # Columns for each LF model
        for m in range(1, nmodels):
            ridx = int(self._recursion_index[m - 1])
            # Shared samples: min of model m and its parent
            n_shared = bkd.minimum(nsamples[m], nsamples[ridx])
            parts.append(bkd.reshape(n_shared, (1,)))  # Even column

            # All samples for model m
            parts.append(bkd.reshape(nsamples[m], (1,)))  # Odd column

        return bkd.concatenate(parts)

    def __call__(self, values: List[Array]) -> Array:
        """Compute GIS estimate.

        Q_GIS = Q_0 + sum_m eta_m * (mu_m - Q_m)

        where:
        - Q_0 is the HF sample mean
        - mu_m is the LF mean on all samples
        - Q_m is the LF mean on shared samples with recursion parent

        Parameters
        ----------
        values : List[Array]
            Model outputs. values[m] has shape (nsamples_m, nqoi)

        Returns
        -------
        Array
            Estimated statistic. Shape: (nstats,)
        """
        if len(values) != self.nmodels():
            raise ValueError(
                f"Expected {self.nmodels()} model outputs, got {len(values)}"
            )

        bkd = self._bkd
        nmodels = self.nmodels()
        nsamples = self.nsamples_per_model()
        weights = self.weights()

        nhf = int(nsamples[0].item())

        # Q_0: HF mean
        Q0 = bkd.sum(values[0], axis=0) / nhf

        # Control variate correction
        correction = bkd.zeros((self._stat.nstats(),))

        for m in range(1, nmodels):
            nlf = int(nsamples[m].item())
            ridx = int(self._recursion_index[m - 1])
            n_parent = int(nsamples[ridx].item())

            # Number of shared samples between model m and its parent
            n_shared = min(nlf, n_parent)

            # Q_m: LF mean on shared samples
            Q_m = bkd.sum(values[m][:n_shared], axis=0) / n_shared

            # mu_m: LF mean on all samples
            mu_m = bkd.sum(values[m], axis=0) / nlf

            # Weight for this control
            eta_m = weights[m - 1] if weights.ndim == 1 else weights[m - 1, :]

            correction = correction + eta_m * (mu_m - Q_m)

        return Q0 + correction

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"GISEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
