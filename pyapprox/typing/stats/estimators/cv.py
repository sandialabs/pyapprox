"""Control Variate estimator.

Uses low-fidelity models with known statistics as control variates
to reduce variance of the high-fidelity estimator.
"""

from typing import Generic, List, Callable, Optional, Any

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.base import AbstractEstimator
from pyapprox.typing.stats.estimators.bootstrap import BootstrapMixin


class CVEstimator(BootstrapMixin[Array], AbstractEstimator[Array], Generic[Array]):
    """Control Variate estimator using low-fidelity models with known statistics.

    The CV estimator uses low-fidelity models as control variates:
        Q_CV = Q_0 + sum_{m=1}^{M-1} eta_m * (Q_m - mu_m)

    where:
    - Q_0 is the HF sample estimate (on shared samples)
    - Q_m is the LF sample estimate (on shared samples)
    - mu_m is the known LF statistic (e.g., known mean)
    - eta_m are the optimal weights

    All models use the same samples (fully shared).

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.
    lowfi_stats : Array, optional
        Known statistics for low-fidelity models. Shape: (nmodels-1, nstats)
        If None, defaults to zeros.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.95], [0.8, 0.95, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([10.0, 1.0, 0.1])
    >>> # Known means for LF models
    >>> lowfi_stats = bkd.asarray([[0.5], [0.6]])
    >>> cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
    >>> cv.allocate_samples(target_cost=100.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
        lowfi_stats: Optional[Array] = None,
    ):
        super().__init__(stat, costs, bkd)

        nmodels_costs = costs.shape[0]
        nstats = stat.nstats()

        # Validate costs length matches number of models in covariance
        nmodels_stat = stat.nmodels()
        if nmodels_costs != nmodels_stat:
            raise ValueError(
                f"Number of costs ({nmodels_costs}) must match number of models "
                f"in pilot covariance ({nmodels_stat})"
            )

        nmodels = nmodels_costs

        # Validate and store lowfi_stats
        if lowfi_stats is not None:
            expected_shape = (nmodels - 1, nstats)
            if lowfi_stats.shape != expected_shape:
                raise ValueError(
                    f"lowfi_stats must have shape {expected_shape} "
                    f"({nmodels - 1} LF models, {nstats} stats), "
                    f"got {lowfi_stats.shape}"
                )
            self._lowfi_stats = lowfi_stats
        else:
            # Default to zeros if not provided
            self._lowfi_stats = bkd.zeros((nmodels - 1, nstats))

        self._weights: Optional[Array] = None
        self._npartition_samples: Optional[Array] = None

    def lowfi_stats(self) -> Array:
        """Return the known low-fidelity statistics."""
        return self._lowfi_stats

    def allocate_samples(self, target_cost: float) -> None:
        """Allocate samples across models.

        For CV, all models use the same samples (fully shared).
        The number of samples is determined by the total cost.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        bkd = self._bkd

        # Total cost per sample = sum of all model costs
        total_cost_per_sample = bkd.sum(self._costs)

        # Number of shared samples
        nsamples_float = bkd.asarray(target_cost) / total_cost_per_sample
        nsamples = bkd.maximum(
            bkd.floor(nsamples_float),
            bkd.asarray(float(self._stat.min_nsamples()))
        )

        # All models get the same number of samples
        nmodels = self.nmodels()
        self._nsamples = bkd.full((nmodels,), nsamples)

        # Single partition with all samples
        self._npartition_samples = bkd.reshape(nsamples, (1,))

        # Compute optimal weights
        self._compute_weights()

    def _compute_weights(self) -> None:
        """Compute optimal control variate weights.

        The optimal weights are:
            eta = -CF^{-1} @ cf^T

        where CF is the covariance of discrepancies and cf is the
        covariance between HF estimate and discrepancies.
        """
        bkd = self._bkd
        npart = self.npartition_samples()

        # Get discrepancy covariances from statistic
        CF, cf = self._stat.get_cv_discrepancy_covariances(npart)

        # Solve for weights: eta = -CF^{-1} @ cf^T
        # Note: negative sign because discrepancy is (Q_m - mu_m)
        self._weights = -bkd.solve(CF, cf.T).T

    def weights(self) -> Array:
        """Return optimal control variate weights.

        Returns
        -------
        Array
            Weights. Shape: (nstats, nstats*(nmodels-1)) or flattened.
        """
        if self._weights is None:
            raise ValueError("Weights not computed. Call allocate_samples() first.")
        return self._weights

    def npartition_samples(self) -> Array:
        """Return samples per partition.

        For CV, there is only one partition (all shared).

        Returns
        -------
        Array
            Partition samples. Shape: (1,)
        """
        if self._npartition_samples is None:
            raise ValueError("Samples not allocated.")
        return self._npartition_samples

    def generate_samples_per_model(
        self, rvs: Callable[[int], Array]
    ) -> List[Array]:
        """Generate samples for all models.

        All models use the same samples (fully shared).

        Parameters
        ----------
        rvs : Callable[[int], Array]
            Random variable sampler.

        Returns
        -------
        List[Array]
            Same samples for each model.
        """
        bkd = self._bkd
        nsamples = self.nsamples_per_model()
        n = int(nsamples[0].item())

        samples = rvs(n)

        # All models get the same samples
        return [bkd.copy(samples) for _ in range(self.nmodels())]

    def __call__(self, values: List[Array]) -> Array:
        """Compute control variate estimate.

        Q_CV = Q_0 + sum_{m=1}^{M-1} eta_m * (Q_m - mu_m)

        Parameters
        ----------
        values : List[Array]
            Model outputs. values[m] has shape (nsamples, nqoi)

        Returns
        -------
        Array
            Estimated statistic. Shape: (nstats,)
        """
        nmodels = self.nmodels()
        if len(values) != nmodels:
            raise ValueError(
                f"CVEstimator expects {nmodels} model outputs, got {len(values)}"
            )

        bkd = self._bkd
        weights = self.weights()

        # Q_0: HF sample estimate
        Q0 = self._stat.sample_estimate(values[0])

        # Compute discrepancies and apply weights
        # delta_m = Q_m - mu_m
        deltas = bkd.hstack([
            self._stat.sample_estimate(values[ii]) - self._lowfi_stats[ii - 1]
            for ii in range(1, nmodels)
        ])

        # Q_CV = Q_0 + weights @ deltas
        result = Q0 + weights @ deltas

        return result

    def optimized_covariance(self) -> Array:
        """Return covariance of the CV estimator.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nstats, nstats)
        """
        return self._covariance_from_npartition_samples(self.npartition_samples())

    def _covariance_from_npartition_samples(
        self, npartition_samples: Array
    ) -> Array:
        """Compute estimator covariance from partition samples.

        Parameters
        ----------
        npartition_samples : Array
            Samples per partition. Shape: (1,)

        Returns
        -------
        Array
            Estimator covariance matrix.
        """
        bkd = self._bkd

        CF, cf = self._stat.get_cv_discrepancy_covariances(npartition_samples)
        weights = -bkd.solve(CF, cf.T).T

        # Covariance: Var(Q_0) + weights @ cf^T
        hf_cov = self._stat.high_fidelity_estimator_covariance(
            npartition_samples[0]
        )

        return hf_cov + weights @ cf.T

    def variance_reduction(self) -> float:
        """Return variance reduction factor compared to MC.

        Returns
        -------
        float
            Ratio Var(Q_CV) / Var(Q_MC).
        """
        bkd = self._bkd
        cov = self._stat.cov()
        nqoi = self._stat.nqoi()

        # Compute total correlation from all LF models
        total_rho_sq = bkd.asarray(0.0)
        for m in range(1, self.nmodels()):
            if nqoi == 1:
                rho = cov[0, m] / bkd.sqrt(cov[0, 0] * cov[m, m])
            else:
                # Use trace-based correlation
                cov_0m = cov[:nqoi, m * nqoi : (m + 1) * nqoi]
                var_0 = bkd.trace(cov[:nqoi, :nqoi]) / nqoi
                var_m = bkd.trace(cov[m * nqoi : (m + 1) * nqoi, m * nqoi : (m + 1) * nqoi]) / nqoi
                rho = bkd.trace(cov_0m) / nqoi / bkd.sqrt(var_0 * var_m)
            total_rho_sq = total_rho_sq + rho ** 2

        var_red = bkd.maximum(1 - total_rho_sq, bkd.asarray(0.01))
        return float(bkd.to_numpy(var_red))

    def _compute_optimal_weights(self) -> Array:
        """Compute optimal control variate weights.

        This method is called by BootstrapMixin during weight resampling.

        Returns
        -------
        Array
            Optimal weights.
        """
        self._compute_weights()
        return self._weights  # type: ignore

    def _estimate_with_weights(
        self, values_per_model: List[Array], weights: Any
    ) -> Array:
        """Compute estimate using specified weights.

        Parameters
        ----------
        values_per_model : List[Array]
            Model values.
        weights : Any
            Control variate weights. If None, uses stored weights.

        Returns
        -------
        Array
            Estimated statistic.
        """
        if weights is None:
            return self(values_per_model)

        bkd = self._bkd
        nmodels = self.nmodels()

        # Q_0: HF sample estimate
        Q0 = self._stat.sample_estimate(values_per_model[0])

        # Compute discrepancies
        deltas = bkd.hstack([
            self._stat.sample_estimate(values_per_model[ii]) - self._lowfi_stats[ii - 1]
            for ii in range(1, nmodels)
        ])

        # Q_CV = Q_0 + weights @ deltas
        result = Q0 + weights @ deltas

        return result

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str([int(n) for n in ns])
        return f"CVEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
