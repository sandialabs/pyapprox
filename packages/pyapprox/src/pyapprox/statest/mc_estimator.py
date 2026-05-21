"""Monte Carlo estimator for single-fidelity estimation.

This module provides the base MCEstimator class for computing statistics
using standard Monte Carlo sampling, and the FittedMCEstimator which
holds a frozen allocation.
"""

from typing import Any, Callable, Generic, List, Tuple, Union

import numpy as np

from pyapprox.statest.statistics import (
    MultiOutputStatistic,
    log_determinant_variance,
)
from pyapprox.util.backends.protocols import Array, Backend


class MCEstimator(Generic[Array]):
    """Monte Carlo estimator template (immutable after construction)."""

    def __init__(
        self,
        stat: MultiOutputStatistic[Array],
        costs: Union[List[Any], Array],
    ):
        r"""
        Parameters
        ----------
        stat : MultiOutputStatistic
            Object defining what statistic will be calculated

        costs : Array (nmodels)
            The relative costs of evaluating each model
        """
        self._bkd = stat._bkd

        self._stat, self._costs = self._check_inputs(stat, costs)
        self._optimization_criteria: Callable[[Array], Array] = (
            lambda var: log_determinant_variance(self._bkd, var)
        )
        self._npartitions = 1

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def _check_inputs(
        self, stat: MultiOutputStatistic[Array], costs: Union[List[Any], Array]
    ) -> Tuple[MultiOutputStatistic[Array], Array]:
        if not isinstance(stat, MultiOutputStatistic):
            raise ValueError("stat must be an instance of MultiOutputStatistic")

        costs = self._bkd.atleast_1d(self._bkd.asarray(costs))
        if costs.ndim != 1:
            raise ValueError("costs is not a 1D iterable")
        self._nmodels = stat._nmodels
        return stat, costs

    def _covariance_from_npartition_samples(self, npartition_samples: Array) -> Array:
        """
        Get the variance of the Monte Carlo estimator from costs and cov
        and npartition_samples
        """
        return self._stat.high_fidelity_estimator_covariance(
            npartition_samples[0]
        )

    def __repr__(self) -> str:
        return "{0}(stat={1}, nqoi={2})".format(
            self.__class__.__name__, self._stat, self._stat._nqoi
        )


class FittedMCEstimator(Generic[Array]):
    """Frozen MC estimator with a fixed sample allocation.

    Composes a template MCEstimator with allocation data. All evaluation
    methods read from the frozen allocation — no mutable state.
    """

    def __init__(
        self,
        template: MCEstimator[Array],
        nsamples_per_model: Array,
        actual_cost: float,
    ) -> None:
        if not template._bkd.is_integer_dtype(nsamples_per_model):
            raise TypeError(
                f"nsamples_per_model must be integer-typed, "
                f"got dtype={nsamples_per_model.dtype}"
            )
        self._template = template
        self._bkd = template._bkd
        self._stat = template._stat
        self._nsamples_per_model = nsamples_per_model
        self._actual_cost = actual_cost
        self._covariance_val = template._covariance_from_npartition_samples(
            nsamples_per_model
        )
        self._criteria_val = template._optimization_criteria(self._covariance_val)

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def covariance(self) -> Array:
        """Return the estimator covariance at the allocated sample count."""
        return self._covariance_val

    def nsamples_per_model(self) -> Array:
        """Return the number of samples allocated to each model."""
        return self._nsamples_per_model

    def actual_cost(self) -> float:
        """Return the actual cost of the allocation."""
        return self._actual_cost

    def generate_samples_per_model(self, rvs: Callable[..., Any]) -> List[Array]:
        """
        Returns the samples needed to the model

        Parameters
        ----------
        rvs : callable
            Function with signature

            `rvs(nsamples)->Array (nvars, nsamples)`

        Returns
        -------
        samples_per_model : list[Array] (1)
            List with one entry Array (nvars, nsamples_per_model[0])
        """
        return [rvs(self._nsamples_per_model[0])]

    def __call__(self, values: Array) -> Array:
        """
        Return the value of the estimator using a set of model evaluations.

        Parameters
        ----------
        values: Array
            The values of each model output at the optimal number of samples.
            Shape: (nqoi, nsamples)

        Return
        ------
        stat_value: Array
            The value of the estimate statistic. Shape: (nstats,)
        """
        nhf = self._bkd.to_int(self._nsamples_per_model[0])
        if (values.ndim != 2) or (values.shape[1] != nhf):
            msg = "values has the incorrect shape {0} expected {1}".format(
                values.shape,
                (self._stat._nqoi, nhf),
            )
            raise ValueError(msg)
        return self._stat.sample_estimate(values)

    def bootstrap(
        self, values: List[Array], nbootstraps: int = 1000
    ) -> Tuple[Array, Array]:
        r"""
        Approximate the variance of the estimator using
        bootstraping. The accuracy of bootstapping depends on the number
        of values per model. As it gets large the boostrapped statistics
        will approach the theoretical values.

        Parameters
        ----------
        values : [Array (nqoi, nsamples)]
            A single entry list containing the unique values of each model.
            The list is required to allow consistent interface with
            multi-fidelity estimators

        nbootstraps : integer
            The number of boostraps used to compute estimator variance

        Returns
        -------
        bootstrap_stats : Array
            The bootstrap estimate of the estimator

        bootstrap_covar : Array
            The bootstrap estimate of the estimator covariance
        """
        nbootstraps = int(nbootstraps)
        estimator_vals = self._bkd.empty((nbootstraps, self._stat._nqoi))
        nsamples = values[0].shape[1]
        for kk in range(nbootstraps):
            bootstrapped_indices = self._bkd.array(
                np.random.choice(nsamples, size=nsamples, replace=True),
                dtype=int,
            )
            estimator_vals[kk] = self._stat.sample_estimate(
                values[0][:, bootstrapped_indices]
            )
            bootstrap_mean = self._bkd.mean(estimator_vals, axis=0)
            bootstrap_covar = self._bkd.cov(estimator_vals, rowvar=False, ddof=1)
        return bootstrap_mean, bootstrap_covar

    def __repr__(self) -> str:
        return "{0}(criteria={1:.3g}, target_cost={2:.5g}, nsamples={3})".format(
            self.__class__.__name__,
            self._bkd.to_float(self._criteria_val),
            self._actual_cost,
            self._bkd.to_int(self._nsamples_per_model[0]),
        )
