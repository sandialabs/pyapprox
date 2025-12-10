"""Adaptive Ensemble Target Cost (AETC) base estimator.

AETC adaptively allocates samples to achieve a target variance while
minimizing computational cost.
"""

from typing import Generic, List, Callable, Optional, Dict, Any

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import (
    StatisticWithDiscrepancyProtocol,
    EstimatorProtocol,
)
from pyapprox.typing.stats.factory import get_estimator


class AETCEstimator(Generic[Array]):
    """Adaptive Ensemble Target Cost estimator.

    AETC adaptively adjusts sample allocation to achieve a target
    variance while minimizing computational cost. It uses an iterative
    approach with pilot samples and refinement.

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.
    base_estimator : str
        Name of the base estimator to use. Default: "mfmc".
    pilot_fraction : float
        Fraction of budget to use for pilot samples. Default: 0.1.
    max_iterations : int
        Maximum number of adaptation iterations. Default: 10.
    variance_tol : float
        Tolerance for variance convergence. Default: 0.1 (10%).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> costs = bkd.asarray([10.0, 1.0, 0.1])
    >>> aetc = AETCEstimator(stat, costs, bkd)
    >>> aetc.allocate_samples(target_variance=0.01)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
        base_estimator: str = "mfmc",
        pilot_fraction: float = 0.1,
        max_iterations: int = 10,
        variance_tol: float = 0.1,
    ):
        self._stat = stat
        self._costs = costs
        self._bkd = bkd
        self._base_estimator_name = base_estimator
        self._pilot_fraction = pilot_fraction
        self._max_iterations = max_iterations
        self._variance_tol = variance_tol

        self._nsamples: Optional[Array] = None
        self._total_cost: Optional[float] = None
        self._achieved_variance: Optional[float] = None
        self._estimator: Optional[EstimatorProtocol[Array]] = None
        self._iteration_history: List[Dict[str, Any]] = []

    def nmodels(self) -> int:
        """Return number of models."""
        return self._costs.shape[0]

    def allocate_samples(self, target_variance: float) -> None:
        """Allocate samples to achieve target variance.

        Parameters
        ----------
        target_variance : float
            Target variance for the estimator.
        """
        bkd = self._bkd
        costs_np = bkd.to_numpy(self._costs)
        hf_cost = costs_np[0]

        # Estimate initial cost using HF-only variance
        try:
            cov = self._stat.cov()
            cov_np = bkd.to_numpy(cov)
            if cov_np.ndim == 2:
                hf_var = cov_np[0, 0] if cov_np.shape[0] == self.nmodels() else np.trace(cov_np) / cov_np.shape[0]
            else:
                hf_var = float(cov_np)

            # Rough estimate: n_hf = var_hf / target_variance
            n_hf_estimate = max(hf_var / target_variance, self._stat.min_nsamples())
            initial_cost = n_hf_estimate * hf_cost * 2  # Factor 2 for safety margin
        except Exception:
            initial_cost = 1000 * hf_cost  # Default

        self._iteration_history = []
        current_cost = initial_cost

        for iteration in range(self._max_iterations):
            # Create and allocate estimator
            estimator = get_estimator(
                self._base_estimator_name, self._stat, self._costs, bkd
            )
            estimator.allocate_samples(target_cost=current_cost)

            # Compute achieved variance
            opt_cov = estimator.optimized_covariance()
            opt_cov_np = bkd.to_numpy(opt_cov)

            if opt_cov_np.ndim == 2:
                achieved_variance = float(np.trace(opt_cov_np))
            else:
                achieved_variance = float(opt_cov_np)

            # Compute actual cost
            nsamples = estimator.nsamples_per_model()
            nsamples_np = bkd.to_numpy(nsamples)
            actual_cost = float(np.sum(nsamples_np * costs_np))

            self._iteration_history.append({
                "iteration": iteration,
                "target_cost": current_cost,
                "actual_cost": actual_cost,
                "achieved_variance": achieved_variance,
                "nsamples": nsamples_np.copy(),
            })

            # Check convergence
            variance_ratio = achieved_variance / target_variance
            if abs(variance_ratio - 1.0) < self._variance_tol:
                # Converged
                break

            # Adjust cost for next iteration
            # If variance too high, increase cost
            # If variance too low, decrease cost
            adjustment = np.sqrt(achieved_variance / target_variance)
            current_cost = actual_cost * adjustment

        # Store final results
        self._estimator = estimator
        self._nsamples = nsamples
        self._total_cost = actual_cost
        self._achieved_variance = achieved_variance

    def allocate_for_target_cost(self, target_cost: float) -> None:
        """Allocate samples for a target cost (compatibility method).

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        # Create estimator and allocate
        self._estimator = get_estimator(
            self._base_estimator_name, self._stat, self._costs, self._bkd
        )
        self._estimator.allocate_samples(target_cost)

        costs_np = self._bkd.to_numpy(self._costs)
        nsamples_np = self._bkd.to_numpy(self._estimator.nsamples_per_model())

        self._nsamples = self._estimator.nsamples_per_model()
        self._total_cost = float(np.sum(nsamples_np * costs_np))

        opt_cov = self._estimator.optimized_covariance()
        opt_cov_np = self._bkd.to_numpy(opt_cov)

        if opt_cov_np.ndim == 2:
            self._achieved_variance = float(np.trace(opt_cov_np))
        else:
            self._achieved_variance = float(opt_cov_np)

    def nsamples_per_model(self) -> Array:
        """Return samples per model."""
        if self._nsamples is None:
            raise ValueError("Samples not allocated.")
        return self._nsamples

    def total_cost(self) -> float:
        """Return total computational cost."""
        if self._total_cost is None:
            raise ValueError("Samples not allocated.")
        return self._total_cost

    def achieved_variance(self) -> float:
        """Return achieved variance."""
        if self._achieved_variance is None:
            raise ValueError("Samples not allocated.")
        return self._achieved_variance

    def optimized_covariance(self) -> Array:
        """Return covariance of the estimator."""
        if self._estimator is None:
            raise ValueError("Samples not allocated.")
        return self._estimator.optimized_covariance()

    def iteration_history(self) -> List[Dict[str, Any]]:
        """Return history of allocation iterations."""
        return self._iteration_history

    def generate_samples_per_model(
        self, rvs: Callable[[int], Array]
    ) -> List[Array]:
        """Generate samples for each model.

        Parameters
        ----------
        rvs : Callable[[int], Array]
            Random variable sampler.

        Returns
        -------
        List[Array]
            Samples for each model.
        """
        if self._estimator is None:
            raise ValueError("Samples not allocated.")
        return self._estimator.generate_samples_per_model(rvs)

    def __call__(self, values: List[Array]) -> Array:
        """Compute estimate.

        Parameters
        ----------
        values : List[Array]
            Model outputs.

        Returns
        -------
        Array
            Estimated statistic.
        """
        if self._estimator is None:
            raise ValueError("Samples not allocated.")
        return self._estimator(values)

    def __repr__(self) -> str:
        if self._nsamples is None:
            return f"AETCEstimator(base={self._base_estimator_name!r}, not allocated)"
        return (
            f"AETCEstimator(base={self._base_estimator_name!r}, "
            f"cost={self._total_cost:.1f}, variance={self._achieved_variance:.2e})"
        )
