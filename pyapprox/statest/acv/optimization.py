"""ACV optimization infrastructure.

This module provides:
- Allocation matrix functions for different ACV variants
- Sample/value combination utilities
- ACVObjective and ACVPartitionConstraint for sample allocation optimization
"""

from abc import abstractmethod, ABC
from typing import Generic, List, TYPE_CHECKING

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.statest.acv.base import ACVEstimator


def _combine_acv_values(
    reorder_allocation_mat: Array,
    npartition_samples: Array,
    acv_values: List,
    bkd: Backend[Array],
) -> List[Array]:
    r"""
    Extract the unique values from the sets
    :math:`f_\alpha(\mathcal{Z}_\alpha), `f_\alpha(\mathcal{Z}_\alpha^*)`
    for each model :math:`\alpha=0,\ldots,M`
    """
    nmodels = len(acv_values)
    values_per_model = [None for ii in range(nmodels)]
    values_per_model[0] = acv_values[0][1]
    for ii in range(1, nmodels):
        lb, ub = 0, 0
        lb2, ub2 = 0, 0
        values_per_model[ii] = []
        for jj in range(nmodels):
            found = False
            if reorder_allocation_mat[jj, 2 * ii] == 1:
                ub = lb + int(npartition_samples[jj])
                values_per_model[ii] += [acv_values[ii][0][lb:ub]]
                lb = ub
                found = True
            if reorder_allocation_mat[jj, 2 * ii + 1] == 1:
                # there is no need to enter here is samle set has already
                # been added by acv_values[ii][0], hence the use of elseif here
                ub2 = lb2 + int(npartition_samples[jj])
                if not found:
                    values_per_model[ii] += [acv_values[ii][1][lb2:ub2]]
                lb2 = ub2
        values_per_model[ii] = bkd.vstack(values_per_model[ii])
    return values_per_model


def _combine_acv_samples(
    reorder_allocation_mat: Array,
    npartition_samples: Array,
    acv_samples: List,
    bkd: Backend[Array],
) -> List[Array]:
    r"""
    Extract the unique amples from the sets
    :math:`\mathcal{Z}_\alpha, `\mathcal{Z}_\alpha^*` for each model
    :math:`\alpha=0,\ldots,M`
    """
    nmodels = len(acv_samples)
    samples_per_model = [None for ii in range(nmodels)]
    samples_per_model[0] = acv_samples[0][1]
    for ii in range(1, nmodels):
        lb, ub = 0, 0
        lb2, ub2 = 0, 0
        samples_per_model[ii] = []
        for jj in range(nmodels):
            found = False
            if reorder_allocation_mat[jj, 2 * ii] == 1:
                ub = lb + int(npartition_samples[jj])
                samples_per_model[ii] += [acv_samples[ii][0][:, lb:ub]]
                lb = ub
                found = True
            if reorder_allocation_mat[jj, 2 * ii + 1] == 1:
                ub2 = lb2 + int(npartition_samples[jj])
                if not found:
                    # Only add samples if they were not in Z_m^*
                    samples_per_model[ii] += [acv_samples[ii][1][:, lb2:ub2]]
                    lb2 = ub2
                    samples_per_model[ii] = bkd.hstack(samples_per_model[ii])
    return samples_per_model


def _get_allocation_matrix_gmf(
    recursion_index: Array, bkd: Backend[Array]
) -> Array:
    nmodels = len(recursion_index) + 1
    mat = bkd.zeros((nmodels, 2 * nmodels))
    for ii in range(nmodels):
        mat[ii, 2 * ii + 1] = 1.0
    for ii in range(1, nmodels):
        mat[:, 2 * ii] = mat[:, recursion_index[ii - 1] * 2 + 1]
    for ii in range(2, 2 * nmodels):
        II = bkd.where(mat[:, ii] == 1)[0][-1]
        mat[:II, ii] = 1.0
    return mat


def _get_allocation_matrix_acvis(
    recursion_index: Array, bkd: Backend[Array]
) -> Array:
    nmodels = len(recursion_index) + 1
    mat = bkd.zeros((nmodels, 2 * nmodels))
    for ii in range(nmodels):
        mat[ii, 2 * ii + 1] = 1
    for ii in range(1, nmodels):
        mat[:, 2 * ii] = mat[:, recursion_index[ii - 1] * 2 + 1]
    for ii in range(1, nmodels):
        mat[:, 2 * ii + 1] = bkd.maximum(mat[:, 2 * ii], mat[:, 2 * ii + 1])
    return mat


def _get_allocation_matrix_acvrd(
    recursion_index: Array, bkd: Backend[Array]
) -> Array:
    nmodels = len(recursion_index) + 1
    allocation_mat = bkd.zeros((nmodels, 2 * nmodels))
    for ii in range(nmodels):
        allocation_mat[ii, 2 * ii + 1] = 1
    for ii in range(1, nmodels):
        allocation_mat[:, 2 * ii] = allocation_mat[
            :, recursion_index[ii - 1] * 2 + 1
        ]
    return allocation_mat


class ACVObjective(ABC, Generic[Array]):
    """Abstract base for ACV optimization objectives.

    Satisfies ObjectiveProtocol for use with ScipyTrustConstrOptimizer.
    """

    def __init__(
        self,
        scaling: float = 1,
        bkd: Backend[Array] = None,
    ):
        self._scaling = scaling
        self._bkd = bkd
        self._est = None
        self._target_cost = None

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_target_cost(self, target_cost: float):
        self._target_cost = target_cost

    def nvars(self) -> int:
        return self._est._nmodels - 1

    def nqoi(self) -> int:
        return 1

    def set_estimator(self, est: "ACVEstimator"):
        from pyapprox.statest.acv.base import ACVEstimator
        if not isinstance(est, ACVEstimator):
            raise ValueError("est must be an instance of ACVEstimator")
        self._est = est
        self._bkd = est._bkd

    @abstractmethod
    def _optimization_criteria(self, est_covariance: Array) -> Array:
        raise NotImplementedError

    def _objective_value(self, partition_ratios: Array) -> Array:
        if partition_ratios.shape[1] != 1:
            raise ValueError(
                "partition_ratios must be a 2D array with one column"
            )
        est_covariance = self._est._covariance_from_partition_ratios(
            self._target_cost, partition_ratios[:, 0]
        )
        return self._optimization_criteria(est_covariance) * self._scaling

    def __call__(self, partition_ratios: Array) -> Array:
        return self._bkd.atleast_2d(self._objective_value(partition_ratios))

    def jacobian(self, partition_ratios: Array) -> Array:
        return self._bkd.jacobian(self._objective_value, partition_ratios).T


class ACVLogDeterminantObjective(ACVObjective[Array]):
    """Log-determinant objective for ACV optimization."""

    def _optimization_criteria(self, est_covariance: Array) -> Array:
        # Only compute large eigvalues as the variance will
        # be singular when estimating variance or mean+variance
        # because of the duplicate entries in
        # the covariance matrix
        eigvals = self._bkd.eigh(est_covariance)[0]
        return self._bkd.log(eigvals[eigvals > 1e-14]).sum()


class ACVPartitionConstraint(Generic[Array]):
    """Constraint ensuring valid partition sample counts.

    Satisfies NonlinearConstraintProtocolWithJacobianAndWHVP for use with
    ScipyTrustConstrOptimizer.
    """

    def __init__(
        self,
        est: "ACVEstimator",
        target_cost: float,
    ):
        from pyapprox.statest.acv.base import ACVEstimator
        if not isinstance(est, ACVEstimator):
            raise ValueError("est must be an instance of ACVEstimator")
        self._est = est
        self._target_cost = target_cost
        self._bkd = est._bkd
        self._lb = self._bkd.zeros((self._est._npartitions,))
        self._ub = self._bkd.full((self._est._npartitions,), np.inf)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._est._nmodels - 1

    def nqoi(self) -> int:
        return self._est._npartitions

    def lb(self) -> Array:
        return self._lb

    def ub(self) -> Array:
        return self._ub

    def _eval_constraint(self, partition_ratios: Array) -> Array:
        if partition_ratios.ndim != 1:
            raise ValueError("partition_ratios.ndim != 1")
        nsamples = self._est._npartition_samples_from_partition_ratios(
            self._target_cost, partition_ratios
        )
        vals = nsamples - self._est._stat.min_nsamples()
        return vals

    def __call__(self, partition_ratios: Array) -> Array:
        return self._eval_constraint(partition_ratios[:, 0])[:, None]

    def jacobian(self, partition_ratios: Array) -> Array:
        return self._bkd.jacobian(
            self._eval_constraint, partition_ratios[:, 0]
        )
    # Note: whvp is intentionally not implemented.
    # Legacy code sets apply_hessian_implemented() -> False
    # to let scipy use finite difference for constraint hessians.
