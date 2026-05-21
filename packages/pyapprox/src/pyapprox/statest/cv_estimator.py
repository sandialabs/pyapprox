"""Control variate estimator for multi-fidelity estimation.

This module provides the CVEstimator template class and FittedCVEstimator
which holds a frozen allocation with precomputed weights.
"""

import copy
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from pyapprox.statest.mc_estimator import MCEstimator
from pyapprox.statest.statistics import MultiOutputStatistic
from pyapprox.util.backends.protocols import Array, Backend


class CVEstimator(MCEstimator[Array], Generic[Array]):
    """Control variate estimator template (immutable after construction)."""

    def __init__(
        self,
        stat: MultiOutputStatistic[Array],
        costs: Union[List[Any], Array],
        lowfi_stats: Optional[Array] = None,
    ):
        super().__init__(stat, costs)
        if lowfi_stats is not None:
            if lowfi_stats.shape != (self._nmodels - 1, self._stat.nstats()):
                raise ValueError(
                    "lowfi_stats must be a 2D Array with shape {0} "
                    "but has shape {1}".format(
                        (self._nmodels - 1, self._stat.nstats()),
                        lowfi_stats.shape,
                    )
                )
        self._lowfi_stats = lowfi_stats
        self._best_model_indices = self._bkd.arange(len(costs), dtype=int)

    def _get_discrepancy_covariances(
        self, npartition_samples: Array
    ) -> Tuple[Array, Array]:
        return self._stat._get_cv_discrepancy_covariances(npartition_samples)

    def _covariance_from_nsamples_per_model(self, nsamples_per_model: Array) -> Array:
        """Compute covariance from nsamples_per_model for CV estimator."""
        CF, cf = self._get_discrepancy_covariances(nsamples_per_model)
        weights = self._optimal_weights(CF, cf)
        return self._stat.high_fidelity_estimator_covariance(
            nsamples_per_model[0]
        ) + self._bkd.multidot([weights, cf.T])

    def _covariance_from_npartition_samples(self, npartition_samples: Array) -> Array:
        """Compute covariance from npartition_samples.

        For CV estimator, npartition_samples = nsamples_per_model.
        """
        return self._covariance_from_nsamples_per_model(npartition_samples)

    def _estimator_cost(self, npartition_samples: Array) -> Array:
        return self._bkd.sum(npartition_samples[0] * self._costs)

    def _optimal_weights(self, CF: Array, cf: Array) -> Array:
        # print(self._bkd.cond(CF), "ACV COND")
        # return -self._bkd.multidot((self._bkd.pinv(CF), cf.T)).T
        return -self._bkd.solve(CF, cf.T).T

    def _covariance_non_optimal_weights(
        self, hf_est_covar: Array, weights: Array, CF: Array, cf: Array
    ) -> Array:
        # The expression below, e.g. Equation 8
        # from Dixon 2024, can be used for non optimal control variate weights
        # Warning: Even though this function is general,
        # it should only ever be used for MLMC, because
        # expression for optimal weights is more efficient
        return (
            hf_est_covar
            + self._bkd.multidot([weights, CF, weights.T])
            + self._bkd.multidot([cf, weights.T])
            + self._bkd.multidot([weights, cf.T])
        )

    def _estimate(
        self,
        values_per_model: List[Array],
        weights: Array,
        bootstrap: bool = False,
    ) -> Array:
        """Compute CV estimate.

        Parameters
        ----------
        values_per_model : List[Array]
            Model values. Each array has shape (nqoi, nsamples).
        weights : Array
            Control variate weights.
        bootstrap : bool
            Whether to use bootstrap resampling.

        Returns
        -------
        Array
            Estimate. Shape: (nstats,)
        """
        if len(values_per_model) != self._nmodels:
            msg = "Must provide the values for each model."
            msg += " {0} != {1}".format(len(values_per_model), self._nmodels)
            raise ValueError(msg)
        if self._lowfi_stats is None:
            raise ValueError("lowfi_stats must be set before calling _estimate")
        nsamples = values_per_model[0].shape[1]
        for values in values_per_model[1:]:
            if values.shape[1] != nsamples:
                msg = "Must provide the same number of samples for each model"
                raise ValueError(msg)
        indices = self._bkd.arange(nsamples, dtype=int)
        if bootstrap:
            indices = self._bkd.array(
                np.random.choice(
                    self._bkd.to_numpy(indices),
                    size=nsamples,
                    replace=True,
                ),
                dtype=int,
            )

        lowfi_stats = self._lowfi_stats
        deltas = self._bkd.hstack(
            [
                self._stat.sample_estimate(values_per_model[ii][:, indices])
                - lowfi_stats[ii - 1]
                for ii in range(1, self._nmodels)
            ]
        )
        est = (
            self._stat.sample_estimate(values_per_model[0][:, indices])
            + weights @ deltas
        )
        return est

    def insert_pilot_values(
        self, pilot_values: List[Array], values_per_model: List[Array]
    ) -> List[Array]:
        """
        Only add pilot values to the first independent partition and thus
        only to models that use that partition.

        Parameters
        ----------
        pilot_values : List[Array]
            Pilot values for each model. Each array has shape (nqoi, npilot).
        values_per_model : List[Array]
            Model values. Each array has shape (nqoi, nsamples).

        Returns
        -------
        List[Array]
            Combined values. Each array has shape (nqoi, npilot + nsamples).
        """
        # CV estimator: all models share samples from the same partition,
        # so pilot values are always prepended to every model.
        return [
            self._bkd.hstack((pilot_values[ii], values_per_model[ii]))
            for ii in range(self._nmodels)
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stat={self._stat})"


class FittedCVEstimator(Generic[Array]):
    """Frozen CV estimator with a fixed sample allocation and precomputed weights."""

    def __init__(
        self,
        template: CVEstimator[Array],
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
        self._npartition_samples = nsamples_per_model
        self._actual_cost = actual_cost

        CF, cf = template._get_discrepancy_covariances(nsamples_per_model)
        self._weights_val = template._optimal_weights(CF, cf)
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

    def weights(self) -> Array:
        """Return the optimal control variate weights."""
        return self._weights_val

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
        samples_per_model : list[Array] (nmodels)
            List with nmodels entries, each Array (nvars, nsamples_per_model[0])
        """
        samples = rvs(self._nsamples_per_model[0])
        return [self._bkd.copy(samples) for ii in range(self._template._nmodels)]

    def __call__(self, values_per_model: List[Array]) -> Array:
        r"""
        Return the value of the Monte Carlo like estimator

        Parameters
        ----------
        values_per_model : list (nmodels)
            The unique values of each model. Each array has shape (nqoi, nsamples).

        Returns
        -------
        est : Array (nstats,)
            The estimator value.
        """
        for vals in values_per_model:
            if not isinstance(vals, self._bkd.array_type()):
                raise ValueError(
                    "vals must be an instance of {0}".format(self._bkd.array_type())
                )
        return self._template._estimate(values_per_model, self._weights_val)

    def insert_pilot_values(
        self, pilot_values: List[Array], values_per_model: List[Array]
    ) -> List[Array]:
        """Delegate to template's insert_pilot_values."""
        return self._template.insert_pilot_values(pilot_values, values_per_model)

    def bootstrap(
        self,
        values_per_model: List[Array],
        nbootstraps: int = 1000,
        mode: str = "values",
        pilot_values: Optional[List[Array]] = None,
    ) -> Union[
        Tuple[Array, Array],
        Tuple[Array, Array, Array, Array],
    ]:
        """Bootstrap variance estimation.

        Parameters
        ----------
        values_per_model : List[Array]
            Model values. Each array has shape (nqoi, nsamples).
        nbootstraps : int
            Number of bootstrap iterations.
        mode : str
            Bootstrap mode.
        pilot_values : List[Array], optional
            Pilot values. Each array has shape (nqoi, npilot).
        """
        modes = ["values", "values_weights", "weights"]
        if mode not in modes:
            raise ValueError("mode must be in {0}".format(modes))
        if pilot_values is not None and mode not in modes[1:]:
            raise ValueError("pilot_values given by mode not in {0}".format(modes[1:]))
        bootstrap_vals = mode in modes[:2]
        bootstrap_weights = mode in modes[1:]
        nbootstraps = int(nbootstraps)
        estimator_vals_list: List[Array] = []
        weights_acc: List[Array] = []
        self_stat = copy.deepcopy(self._stat)
        if bootstrap_weights:
            if pilot_values is None:
                raise ValueError(
                    "pilot_values must be provided for bootstrap_weights"
                )
            npilot_samples = pilot_values[0].shape[1]
        for kk in range(nbootstraps):
            if bootstrap_weights:
                if pilot_values is None:
                    raise RuntimeError("pilot_values is None")
                indices = self._bkd.array(
                    np.random.choice(
                        npilot_samples,
                        size=npilot_samples,
                        replace=True,
                    ),
                    dtype=int,
                )
                boostrap_pilot_values = [vals[:, indices] for vals in pilot_values]
                self._stat.set_pilot_quantities(
                    *self._stat.compute_pilot_quantities(boostrap_pilot_values)
                )
                CF, cf = self._template._get_discrepancy_covariances(
                    self._npartition_samples
                )
                weights = self._template._optimal_weights(CF, cf)
                weights_acc.append(self._bkd.flatten(weights))
            else:
                weights = self._weights_val
            estimator_vals_list.append(
                self._bkd.flatten(self._template._estimate(
                    values_per_model, weights, bootstrap=bootstrap_vals
                ))
            )
        estimator_vals_arr = self._bkd.stack(estimator_vals_list)
        bootstrap_values_mean = self._bkd.mean(estimator_vals_arr, axis=0)
        bootstrap_values_covar = self._bkd.cov(
            estimator_vals_arr, rowvar=False, ddof=1
        )
        if bootstrap_weights:
            self._template._stat = self_stat
            weights_arr = self._bkd.stack(weights_acc)
            bootstrap_weights_mean = self._bkd.mean(weights_arr, axis=0)
            bootstrap_weights_covar = self._bkd.cov(
                weights_arr, rowvar=False, ddof=1
            )
            return (
                bootstrap_values_mean,
                bootstrap_values_covar,
                bootstrap_weights_mean,
                bootstrap_weights_covar,
            )
        return (bootstrap_values_mean, bootstrap_values_covar)

    def __repr__(self) -> str:
        crit = self._bkd.flatten(self._criteria_val)[0]
        criteria_val = self._bkd.to_float(crit)
        return (
            f"{self.__class__.__name__}("
            f"criteria={criteria_val:.3g}, "
            f"target_cost={self._actual_cost:.5g})"
        )
