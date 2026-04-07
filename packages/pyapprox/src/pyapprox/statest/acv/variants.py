"""ACV estimator variants: GMF, GIS, GRD, MFMC, MLMC.

This module provides specialized ACV estimator variants with different
sample allocation strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, List, Tuple, Union

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

from pyapprox.statest.acv.base import ACVEstimator
from pyapprox.statest.acv.optimization import (
    _get_allocation_matrix_acvis,
    _get_allocation_matrix_acvrd,
    _get_allocation_matrix_gmf,
)
from pyapprox.statest.statistics import MultiOutputStatistic
from pyapprox.util.backends.protocols import Array, Backend


def _covariance_to_correlation(cov: Array, bkd: Backend[Array]) -> Array:
    """Compute the correlation matrix from a covariance matrix."""
    stdev_inv = 1.0 / bkd.sqrt(bkd.get_diagonal(cov))
    cor = stdev_inv[None, :] * cov * stdev_inv[:, None]
    return cor


def _variance_reduction(
    get_rsquared: "Callable[[Array, Array], Array]",
    cov: Array,
    nsample_ratios: Array,
) -> Array:
    r"""
    Compute the variance reduction:

    .. math:: \gamma = 1-r^2

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Returns
    -------
    gamma : float
        The variance reduction
    """
    return 1 - get_rsquared(cov, nsample_ratios)


def _check_mfmc_model_costs_and_correlations(costs: Array, corr: Array) -> bool:
    """
    Check that the model costs and correlations satisfy equation 3.12
    in MFMC paper.
    """
    nmodels = len(costs)
    for ii in range(1, nmodels):
        if ii < nmodels - 1:
            denom = corr[0, ii] ** 2 - corr[0, ii + 1] ** 2
        else:
            denom = corr[0, ii] ** 2
        if denom <= np.finfo(float).eps:
            return False
        corr_ratio = (corr[0, ii - 1] ** 2 - corr[0, ii] ** 2) / denom
        cost_ratio = costs[ii - 1] / costs[ii]
        if corr_ratio >= cost_ratio:
            return False
    return True


def _get_rsquared_mfmc(cov: Array, nsample_ratios: Array) -> Array:
    r"""
    Compute r^2 used to compute the variance reduction  of
    Multifidelity Monte Carlo (MFMC)

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples, i=1,...,nmodels-1

    Returns
    -------
    rsquared : float
        The value r^2
    """
    nmodels = cov.shape[0]
    assert len(nsample_ratios) == nmodels - 1
    rsquared = (
        (nsample_ratios[0] - 1)
        / (nsample_ratios[0])
        * cov[0, 1]
        / (cov[0, 0] * cov[1, 1])
        * cov[0, 1]
    )
    for ii in range(1, nmodels - 1):
        p1 = (nsample_ratios[ii] - nsample_ratios[ii - 1]) / (
            nsample_ratios[ii] * nsample_ratios[ii - 1]
        )
        p1 *= cov[0, ii + 1] / (cov[0, 0] * cov[ii + 1, ii + 1]) * cov[0, ii + 1]
        rsquared += p1
    return rsquared


def _allocate_samples_mfmc(
    cov: Array,
    costs: Array,
    target_cost: float,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    r"""
    Determine the samples to be allocated to each model when using MFMC

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    target_cost : float
        The total cost budget

    Returns
    -------
    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i=r_i*nhf_samples, i=1,...,nmodels-1

    log_variance : float
        The logarithm of the variance of the estimator
    """

    nmodels = len(costs)
    corr = _covariance_to_correlation(cov, bkd)
    II = bkd.flip(bkd.argsort(bkd.abs(corr[0, 1:])))
    if II.shape[0] != nmodels - 1:
        msg = "Correlation shape {0} inconsistent with len(costs) {1}.".format(
            corr.shape, len(costs)
        )
        raise RuntimeError(msg)
    if not bkd.allclose(II, bkd.arange(nmodels - 1, dtype=int)):
        msg = "Models must be ordered with decreasing correlation with "
        msg += "high-fidelity model"
        raise RuntimeError(msg)

    r = []
    for ii in range(nmodels - 1):
        # Step 3 in Algorithm 2 in Peherstorfer et al 2016
        num = costs[0] * (corr[0, ii] ** 2 - corr[0, ii + 1] ** 2)
        den = costs[ii] * (1 - corr[0, 1] ** 2)
        r.append(bkd.sqrt(num / den))

    num = costs[0] * corr[0, -1] ** 2
    den = costs[-1] * (1 - corr[0, 1] ** 2)
    r.append(bkd.sqrt(num / den))
    r = bkd.array(r)

    # Step 4 in Algorithm 2 in Peherstorfer et al 2016
    nhf_samples = target_cost / bkd.dot(costs, r)
    nsample_ratios = r[1:]

    gamma = _variance_reduction(_get_rsquared_mfmc, cov, nsample_ratios)
    log_variance = bkd.log(gamma) + bkd.log(cov[0, 0]) - bkd.log(nhf_samples)
    return bkd.atleast_1d(nsample_ratios), log_variance


def _get_sample_allocation_matrix_mfmc(nmodels: int, bkd: Backend[Array]) -> Array:
    mat = bkd.zeros((nmodels, 2 * nmodels))
    mat[0, 1:] = 1
    for ii in range(1, nmodels):
        mat[ii, 2 * ii + 1 :] = 1
    return mat


def _get_rsquared_mlmc(cov: Array, nsample_ratios: Array) -> Array:
    r"""
    Compute r^2 used to compute the variance reduction of
    Multilevel Monte Carlo (MLMC)

    See Equation 2.24 in ARXIV paper where alpha_i=-1 for all i

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples,
        i=1,...,nmodels-1.
        The values r_i correspond to eta_i in Equation 2.24

    Returns
    -------
    gamma : float
        The variance reduction
    """
    nmodels = cov.shape[0]
    assert len(nsample_ratios) == nmodels - 1
    gamma = 0.0
    rhat = np.ones((nmodels), dtype=float)
    for ii in range(1, nmodels):
        rhat[ii] = nsample_ratios[ii - 1] - rhat[ii - 1]

    for ii in range(nmodels - 1):
        vardelta = cov[ii, ii] + cov[ii + 1, ii + 1] - 2 * cov[ii, ii + 1]
        gamma += vardelta / (rhat[ii])

    v = cov[nmodels - 1, nmodels - 1]
    gamma += v / (rhat[-1])

    gamma /= cov[0, 0]
    return 1 - gamma


def _allocate_samples_mlmc(
    cov: Array,
    costs: Array,
    target_cost: float,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    r"""
    Determine the samples to be allocated to each model when using MLMC

    Parameters
    ----------
    cov : np.ndarray (nmodels,nmodels)
        The covariance C between each of the models. The highest fidelity
        model is the first model, i.e its variance is cov[0,0]

    costs : np.ndarray (nmodels)
        The relative costs of evaluating each model

    target_cost : float
        The total cost budget

    Returns
    -------
    nhf_samples : integer
        The number of samples of the high fidelity model

    nsample_ratios : np.ndarray (nmodels-1)
        The sample ratios r used to specify the number of samples of the
        lower fidelity models, e.g. N_i = r_i*nhf_samples,
        i=1,...,nmodels-1. For model i>0 nsample_ratio*nhf_samples equals
        the number of samples in the two different discrepancies involving
        the ith model.

    log_variance : float
        The logarithm of the variance of the estimator
    """
    nmodels = cov.shape[0]
    costs = bkd.asarray(costs)

    II = bkd.flip(bkd.argsort(costs))
    if not bkd.allclose(II, bkd.arange(nmodels)):
        # print(costs)
        raise ValueError("Models cost do not decrease monotonically")

    # compute the variance of the discrepancy
    var_deltas = bkd.empty((nmodels,))
    for ii in range(nmodels - 1):
        var_deltas[ii] = cov[ii, ii] + cov[ii + 1, ii + 1] - 2 * cov[ii, ii + 1]
    var_deltas[nmodels - 1] = cov[nmodels - 1, nmodels - 1]

    # compute the cost of one sample of the discrepancy
    cost_deltas = bkd.empty((nmodels,))
    cost_deltas[: nmodels - 1] = costs[: nmodels - 1] + costs[1:nmodels]
    cost_deltas[nmodels - 1] = costs[nmodels - 1]

    # compute variance * cost
    var_cost_prods = var_deltas * cost_deltas

    # compute variance / cost
    var_cost_ratios = var_deltas / cost_deltas

    # compute the lagrange multiplier
    lagrange_multiplier = target_cost / bkd.sum(bkd.sqrt(var_cost_prods))

    # compute the number of samples needed for each discrepancy
    nsamples_per_delta = lagrange_multiplier * bkd.sqrt(var_cost_ratios)

    # compute the number of samples allocated to each model. For
    # all but the highest fidelity model we need to collect samples
    # from two discrepancies.
    nhf_samples = nsamples_per_delta[0]
    nsample_ratios = bkd.empty((nmodels - 1,))
    for ii in range(nmodels - 1):
        nsample_ratios[ii] = (
            nsamples_per_delta[ii] + nsamples_per_delta[ii + 1]
        ) / nhf_samples

    assert bkd.allclose(
        nhf_samples * costs[0] + bkd.dot(nsample_ratios * nhf_samples, costs[1:]),
        bkd.dot(cost_deltas, nsamples_per_delta),
    )

    gamma = _variance_reduction(_get_rsquared_mlmc, cov, nsample_ratios)
    log_variance = bkd.log(gamma) + bkd.log(cov[0, 0]) - bkd.log(nhf_samples)
    # print(log_variance)
    if bkd.isnan(log_variance):
        raise RuntimeError("MLMC variance is NAN")
    return bkd.atleast_1d(nsample_ratios), log_variance


def _get_sample_allocation_matrix_mlmc(nmodels: int, bkd: Backend[Array]) -> Array:
    r"""
    Get the sample allocation matrix

    Parameters
    ----------
    nmodel : integer
        The number of models :math:`M`

    Returns
    -------
    mat : np.ndarray (nmodels, 2*nmodels)
        For columns :math:`2j, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i^\star\subseteq z_j^\star`
        For columns :math:`2j+1, j=0,\ldots,M-1` the ith row contains a
        flag specifiying if :math:`z_i\subseteq z_j`
    """
    mat = bkd.zeros((nmodels, 2 * nmodels))
    for ii in range(nmodels - 1):
        mat[ii, 2 * ii + 1 : 2 * ii + 3] = 1
    mat[-1, -1] = 1
    return mat


class GMFEstimator(ACVEstimator[Array], Generic[Array]):
    """Generalized Multifidelity (GMF) estimator."""

    def _create_allocation_matrix(self, recursion_index: Array) -> Array:
        self._allocation_mat = _get_allocation_matrix_gmf(recursion_index, self._bkd)


class GISEstimator(ACVEstimator[Array], Generic[Array]):
    """
    The GIS estimator from Gorodetsky et al. and Bomorito et al
    """

    def _create_allocation_matrix(self, recursion_index: Array) -> Array:
        self._allocation_mat = _get_allocation_matrix_acvis(recursion_index, self._bkd)


class GRDEstimator(ACVEstimator[Array], Generic[Array]):
    """
    The GRD estimator.
    """

    def _create_allocation_matrix(self, recursion_index: Array) -> Array:
        self._allocation_mat = _get_allocation_matrix_acvrd(recursion_index, self._bkd)


class MFMCEstimator(GMFEstimator[Array], Generic[Array]):
    """Multifidelity Monte Carlo (MFMC) estimator."""

    def __init__(
        self,
        stat: MultiOutputStatistic[Array],
        costs: Union[List[float], Array],
        opt_criteria: None = None,
        opt_qoi: int = 0,
    ) -> None:
        # Use the sample analytical sample allocation for estimating a scalar
        # mean when estimating any statistic
        nmodels = len(costs)
        super().__init__(
            stat,
            costs,
            recursion_index=stat._bkd.arange(nmodels - 1, dtype=int),
            opt_criteria=None,
        )
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    def _allocate_samples(
        self, target_cost: float,
    ) -> Tuple[Array, Array]:
        # nsample_ratios returned will be listed in according to
        # self.model_order which is what self.get_rsquared requires
        if not _check_mfmc_model_costs_and_correlations(
            self._costs,
            _covariance_to_correlation(self._stat._cov, self._bkd),
        ):
            raise ValueError("models do not admit a hierarchy")
        nsample_ratios, val = _allocate_samples_mfmc(
            self._stat._cov[
                self._opt_qoi :: self._stat._nqoi,
                self._opt_qoi :: self._stat._nqoi,
            ],
            self._costs,
            target_cost,
            self._bkd,
        )
        nsample_ratios = self._native_ratios_to_npartition_ratios(nsample_ratios)
        return nsample_ratios, val

    def _allocate_samples_analytical(
        self, target_cost: float,
    ) -> Tuple[Array, Array]:
        """Analytical allocation for MFMC.

        Returns partition_ratios and objective_value as Arrays to support
        the AnalyticalAllocator interface. This is a wrapper around
        _allocate_samples() that ensures the objective_value has shape (1,).

        Parameters
        ----------
        target_cost : float
            The total computational budget.

        Returns
        -------
        partition_ratios : Array, shape (nmodels-1,)
            The partition ratios for sample allocation.
        objective_value : Array, shape (1,)
            The log-variance of the estimator.
        """
        partition_ratios, log_variance = self._allocate_samples(target_cost)
        # Ensure objective_value is Array with shape (1,)
        objective_value = self._bkd.atleast_1d(log_variance)
        return partition_ratios, objective_value

    def _native_ratios_to_npartition_ratios(
        self, ratios: Array,
    ) -> Array:
        partition_ratios = self._bkd.hstack((ratios[0] - 1, self._bkd.diff(ratios)))
        return partition_ratios

    def _get_allocation_matrix(self) -> Array:
        return _get_sample_allocation_matrix_mfmc(self._nmodels, self._bkd)


class MLMCEstimator(GRDEstimator[Array], Generic[Array]):
    """Multilevel Monte Carlo (MLMC) estimator."""

    def __init__(
        self,
        stat: MultiOutputStatistic[Array],
        costs: Union[List[float], Array],
        opt_criteria: None = None,
        opt_qoi: int = 0,
    ) -> None:
        """
        Use the sample analytical sample allocation for estimating a scalar
        mean when estimating any statistic

        Use optimal ACV weights instead of all weights=-1 used by
        classical MLMC.
        """
        nmodels = len(costs)
        super().__init__(
            stat,
            costs,
            recursion_index=stat._bkd.arange(nmodels - 1),
            opt_criteria=None,
        )
        # The qoi index used to generate the sample allocation
        self._opt_qoi = opt_qoi

    def _weights(self, CF: Array, cf: Array) -> Array:
        # raise NotImplementedError("check weights size is correct")
        return -self._bkd.ones(cf.shape)

    def _covariance_from_npartition_samples(self, npartition_samples: Array) -> Array:
        CF, cf = self._get_discrepancy_covariances(npartition_samples)
        weights = self._weights(CF, cf)
        # cannot use formulation of variance that uses optimal weights
        # must use the more general expression below, e.g. Equation 8
        # from Dixon 2024.
        return self._covariance_non_optimal_weights(
            self._stat.high_fidelity_estimator_covariance(npartition_samples[0]),
            weights,
            CF,
            cf,
        )

    def _allocate_samples(
        self, target_cost: float,
    ) -> Tuple[Array, Array]:
        nsample_ratios, val = _allocate_samples_mlmc(
            self._stat._cov[
                self._opt_qoi :: self._stat._nqoi,
                self._opt_qoi :: self._stat._nqoi,
            ],
            self._costs,
            target_cost,
            self._bkd,
        )
        partition_ratios = self._native_ratios_to_npartition_ratios(nsample_ratios)
        return partition_ratios, val

    def _allocate_samples_analytical(
        self, target_cost: float,
    ) -> Tuple[Array, Array]:
        """Analytical allocation for MLMC.

        Returns partition_ratios and objective_value as Arrays to support
        the AnalyticalAllocator interface. This is a wrapper around
        _allocate_samples() that ensures the objective_value has shape (1,).

        Parameters
        ----------
        target_cost : float
            The total computational budget.

        Returns
        -------
        partition_ratios : Array, shape (nmodels-1,)
            The partition ratios for sample allocation.
        objective_value : Array, shape (1,)
            The log-variance of the estimator.
        """
        partition_ratios, log_variance = self._allocate_samples(target_cost)
        # Ensure objective_value is Array with shape (1,)
        objective_value = self._bkd.atleast_1d(log_variance)
        return partition_ratios, objective_value

    def _create_allocation_matrix(self, dummy: Array) -> Array:
        self._allocation_mat = _get_sample_allocation_matrix_mlmc(
            self._nmodels, self._bkd
        )

    def _native_ratios_to_npartition_ratios(self, ratios: Array) -> Array:
        partition_ratios = [ratios[0] - 1]
        for ii in range(1, len(ratios)):
            partition_ratios.append(ratios[ii] - partition_ratios[ii - 1])
        return self._bkd.hstack(partition_ratios)
