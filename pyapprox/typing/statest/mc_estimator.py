"""Monte Carlo estimator for single-fidelity estimation.

This module provides the base MCEstimator class for computing statistics
using standard Monte Carlo sampling.
"""

from typing import Callable, Generic, List, Tuple, Union

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend

from pyapprox.typing.statest.statistics import MultiOutputStatistic


class MCEstimator(Generic[Array]):
    """Monte Carlo estimator for single-fidelity statistics."""

    def __init__(
        self,
        stat: MultiOutputStatistic[Array],
        costs: Union[List, Array],
        opt_criteria: Callable = None,
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
        self._optimization_criteria = self._log_determinant_variance

        self._rounded_nsamples_per_model = None
        self._rounded_npartition_samples = None
        self._rounded_target_cost = None
        self._optimized_criteria = None
        self._optimized_covariance = None
        self._model_labels = None
        self._npartitions = 1

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def _check_inputs(
        self, stat: MultiOutputStatistic[Array], costs: Union[List, Array]
    ) -> Tuple[MultiOutputStatistic[Array], Array]:
        if not isinstance(stat, MultiOutputStatistic):
            raise ValueError(
                "stat must be an instance of MultiOutputStatistic"
            )

        costs = self._bkd.atleast_1d(self._bkd.asarray(costs))
        if costs.ndim != 1:
            raise ValueError("costs is not a 1D iterable")
        self._nmodels = stat._nmodels
        return stat, costs

    def _log_determinant_variance(self, variance: Array) -> Array:
        # Only compute large eigvalues as the variance will
        # be singular when estimating variance or mean+variance
        # because of the duplicate entries in
        # the covariance matrix
        eigvals = self._bkd.eigh(variance)[0]
        val = self._bkd.log(eigvals[eigvals > 1e-14]).sum()
        return val

    def _covariance_from_npartition_samples(
        self, npartition_samples: Array
    ) -> Array:
        """
        Get the variance of the Monte Carlo estimator from costs and cov
        and npartition_samples
        """
        return self._stat.high_fidelity_estimator_covariance(
            npartition_samples[0]
        )

    def optimized_covariance(self) -> Array:
        """
        Return the estimator covariance at the optimal sample allocation
        computed using self.allocate_samples()
        """
        return self._optimized_covariance

    def allocate_samples(self, target_cost: float) -> None:
        """
        Find the optimal number of samples that minimize the metric of the
        estimator covariance for the specified target cost.

        Parameters
        ----------
        target_cost : float
            The total computational budget that can be used to compute the
            estimator
        """
        self._rounded_nsamples_per_model = self._bkd.asarray(
            [int(self._bkd.floor(target_cost / self._costs[0]))]
        )
        self._rounded_npartition_samples = self._rounded_nsamples_per_model
        est_covariance = self._covariance_from_npartition_samples(
            self._rounded_npartition_samples
        )
        self._optimized_covariance = est_covariance
        optimized_criteria = self._optimization_criteria(est_covariance)
        self._rounded_target_cost = (
            self._costs[0] * self._rounded_nsamples_per_model[0]
        )
        self._optimized_criteria = optimized_criteria

    def generate_samples_per_model(self, rvs: Callable) -> List[Array]:
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
        return [rvs(self._rounded_nsamples_per_model[0])]

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
        if not isinstance(values, self._bkd.array_type()):
            raise ValueError(
                "values must be an {0} but type={1}".format(
                    self._bkd.array_type(), type(values)
                )
            )
        if (values.ndim != 2) or (
            values.shape[1] != self._rounded_nsamples_per_model[0]
        ):
            msg = "values has the incorrect shape {0} expected {1}".format(
                values.shape,
                (self._stat._nqoi, self._rounded_nsamples_per_model[0]),
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
        indices = self._bkd.arange(nsamples, dtype=int)
        for kk in range(nbootstraps):
            bootstrapped_indices = self._bkd.array(
                np.random.choice(indices, size=nsamples, replace=True),
                dtype=int,
            )
            estimator_vals[kk] = self._stat.sample_estimate(
                values[0][:, bootstrapped_indices]
            )
            bootstrap_mean = estimator_vals.mean(axis=0)
            bootstrap_covar = self._bkd.cov(
                estimator_vals, rowvar=False, ddof=1
            )
        return bootstrap_mean, bootstrap_covar

    def __repr__(self) -> str:
        if self._optimized_criteria is None:
            return "{0}(stat={1}, nqoi={2})".format(
                self.__class__.__name__, self._stat, self._stat._nqoi
            )
        rep = "{0}(stat={1}, criteria={2:.3g}".format(
            self.__class__.__name__, self._stat, self._optimized_criteria
        )
        rep += " target_cost={0:.5g}, nsamples={1})".format(
            self._rounded_target_cost, self._rounded_nsamples_per_model[0]
        )
        return rep
