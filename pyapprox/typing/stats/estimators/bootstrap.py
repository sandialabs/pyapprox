"""Bootstrap mixin for estimator variance estimation.

Provides bootstrap resampling functionality for all estimators.
"""

from typing import Generic, List, Tuple, Optional, Any
import copy

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


class BootstrapMixin(Generic[Array]):
    """Mixin providing bootstrap variance estimation for estimators.

    This mixin can be added to any estimator class to enable bootstrap
    variance estimation. The estimator must have:
    - self._bkd: Backend[Array] attribute
    - self._stat: Statistic object with nstats() method
    - self.__call__(values): Method to compute estimate from values

    Three bootstrap modes are supported:
    - "values": Resample model outputs only (works for all estimators)
    - "values_weights": Resample both outputs and pilot values (requires CV/ACV)
    - "weights": Resample pilot values only (requires CV/ACV)

    For estimators without control variates (e.g., MCEstimator), only
    "values" mode is supported.
    """

    _bkd: Backend[Array]

    def bootstrap(
        self,
        values_per_model: List[Array],
        nbootstraps: int = 1000,
        mode: str = "values",
        pilot_values: Optional[List[Array]] = None,
    ) -> Tuple[Array, ...]:
        """Estimate variance using bootstrap resampling.

        Parameters
        ----------
        values_per_model : List[Array]
            Model outputs. values_per_model[m] has shape (nsamples_m, nqoi)
        nbootstraps : int
            Number of bootstrap iterations.
        mode : str
            Bootstrap mode:
            - "values": Resample model outputs only
            - "values_weights": Resample both outputs and pilot values
            - "weights": Resample pilot values only (keep outputs fixed)
        pilot_values : List[Array], optional
            Pilot sample values. Required if mode is "values_weights" or "weights".

        Returns
        -------
        Tuple[Array, ...]
            If mode == "values":
                (bootstrap_mean, bootstrap_covariance)
            If mode in ("values_weights", "weights"):
                (bootstrap_mean, bootstrap_covariance,
                 bootstrap_weights_mean, bootstrap_weights_covariance)
        """
        modes = ["values", "values_weights", "weights"]
        if mode not in modes:
            raise ValueError(f"mode must be in {modes}, got {mode}")

        if pilot_values is not None and mode not in modes[1:]:
            raise ValueError(
                f"pilot_values given but mode not in {modes[1:]}"
            )
        if pilot_values is None and mode in modes[1:]:
            raise ValueError(
                f"pilot_values required when mode in {modes[1:]}"
            )

        bootstrap_vals = mode in modes[:2]  # values or values_weights
        bootstrap_weights = mode in modes[1:]  # values_weights or weights

        # Check if this estimator supports weight bootstrapping
        has_weights = hasattr(self, "_weights") and hasattr(
            self, "_compute_optimal_weights"
        )
        if bootstrap_weights and not has_weights:
            raise ValueError(
                f"mode '{mode}' requires control variate estimator with weights"
            )

        bkd = self._bkd
        nbootstraps = int(nbootstraps)
        estimator_vals_list: List[Array] = []

        # Save original stat if we'll be modifying it
        original_stat: Any = None
        weights_list: List[Array] = []
        npilot_samples = 0
        if bootstrap_weights:
            npilot_samples = pilot_values[0].shape[0]  # type: ignore
            original_stat = copy.deepcopy(self._stat)  # type: ignore

        for _ in range(nbootstraps):
            weights: Any = None
            if bootstrap_weights:
                # Resample pilot values and recompute weights
                indices = np.random.choice(
                    npilot_samples, size=npilot_samples, replace=True
                )
                bootstrap_pilot = [vals[indices] for vals in pilot_values]  # type: ignore
                # Recompute pilot quantities from bootstrapped pilot values
                pilot_quantities = self._stat.compute_pilot_quantities(  # type: ignore
                    bootstrap_pilot
                )
                self._stat.set_pilot_quantities(*pilot_quantities)  # type: ignore
                # Recompute optimal weights
                weights = self._compute_optimal_weights()  # type: ignore
                weights_list.append(bkd.flatten(weights))
            elif has_weights:
                weights = self._weights  # type: ignore

            # Compute estimate (with or without value resampling)
            if bootstrap_vals:
                # Resample values for each model
                resampled_values = self._resample_values(values_per_model)
                est = self._estimate_with_weights(resampled_values, weights)
            else:
                est = self._estimate_with_weights(values_per_model, weights)

            estimator_vals_list.append(bkd.flatten(est))

        # Stack results and compute statistics
        estimator_vals = bkd.stack(estimator_vals_list, axis=0)
        bootstrap_mean = bkd.mean(estimator_vals, axis=0)
        bootstrap_covar = self._compute_covariance(estimator_vals)

        # Restore original stat
        if bootstrap_weights:
            self._stat = original_stat  # type: ignore
            weights_arr = bkd.stack(weights_list, axis=0)
            weights_mean = bkd.mean(weights_arr, axis=0)
            weights_covar = self._compute_covariance(weights_arr)
            return (bootstrap_mean, bootstrap_covar, weights_mean, weights_covar)

        return (bootstrap_mean, bootstrap_covar)

    def _compute_covariance(self, data: Array) -> Array:
        """Compute sample covariance matrix.

        Parameters
        ----------
        data : Array
            Data matrix. Shape: (nsamples, nvars)

        Returns
        -------
        Array
            Covariance matrix. Shape: (nvars, nvars) or scalar if nvars=1
        """
        bkd = self._bkd
        nsamples = data.shape[0]
        mean = bkd.mean(data, axis=0, keepdims=True)
        centered = data - mean
        cov = centered.T @ centered / (nsamples - 1)
        # Handle scalar case (single QoI)
        if cov.shape == (1, 1):
            return bkd.reshape(cov, (1,))
        return cov

    def _resample_values(self, values_per_model: List[Array]) -> List[Array]:
        """Resample model values with replacement.

        Parameters
        ----------
        values_per_model : List[Array]
            Original model values.

        Returns
        -------
        List[Array]
            Resampled model values.
        """
        resampled = []
        for values in values_per_model:
            nsamples = values.shape[0]
            indices = np.random.choice(nsamples, size=nsamples, replace=True)
            resampled.append(values[indices])
        return resampled

    def _estimate_with_weights(
        self, values_per_model: List[Array], weights: Array
    ) -> Array:
        """Compute estimate using specified weights.

        Default implementation calls __call__. Subclasses may override
        for more efficient computation when weights are provided directly.

        Parameters
        ----------
        values_per_model : List[Array]
            Model values.
        weights : Array
            Control variate weights.

        Returns
        -------
        Array
            Estimated statistic.
        """
        # Default: just use __call__ (ignores weights, assumes optimized)
        return self(values_per_model)  # type: ignore
