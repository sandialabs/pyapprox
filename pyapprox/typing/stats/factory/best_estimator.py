"""Best estimator selection via optimization.

Finds the optimal estimator configuration for a given budget.
"""

from typing import Generic, List, Optional, Dict, Any, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import (
    StatisticWithDiscrepancyProtocol,
    EstimatorProtocol,
)
from pyapprox.typing.stats.factory.estimator_factory import get_estimator


class BestEstimator(Generic[Array]):
    """Find the best estimator configuration.

    Compares multiple estimator types and model subsets to find the
    configuration that minimizes variance for a given cost budget.

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate. Must have pilot quantities set.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.
    estimator_types : List[str], optional
        Estimator types to compare. Default: ["mfmc", "mlmc", "acv", "gmf"].
    max_nmodels : int, optional
        Maximum number of models to use. If None, uses all models.
    require_hf : bool, optional
        If True, always include the high-fidelity model (index 0).
        Default: True.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([10.0, 1.0, 0.1])
    >>> best = BestEstimator(stat, costs, bkd)
    >>> best.allocate_samples(target_cost=100.0)
    >>> print(best.best_estimator_type())
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
        estimator_types: Optional[List[str]] = None,
        max_nmodels: Optional[int] = None,
        require_hf: bool = True,
    ):
        self._stat = stat
        self._costs = costs
        self._bkd = bkd
        self._require_hf = require_hf

        nmodels = costs.shape[0]
        self._nmodels = nmodels

        if max_nmodels is None:
            max_nmodels = nmodels
        self._max_nmodels = min(max_nmodels, nmodels)

        if estimator_types is None:
            estimator_types = ["mfmc", "mlmc", "gmf", "grd"]
        self._estimator_types = estimator_types

        self._best_estimator: Optional[EstimatorProtocol[Array]] = None
        self._best_type: Optional[str] = None
        self._best_models: Optional[List[int]] = None
        self._comparison_results: Dict[str, Any] = {}

    def nmodels(self) -> int:
        """Return total number of available models."""
        return self._nmodels

    def allocate_samples(self, target_cost: float) -> None:
        """Find optimal estimator and allocate samples.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        bkd = self._bkd
        costs_np = bkd.to_numpy(self._costs)

        best_variance = np.inf
        best_estimator = None
        best_type = None
        best_models = None

        # Try different model subsets
        model_subsets = self._generate_model_subsets()

        for models in model_subsets:
            # Create subset stat and costs
            subset_costs = bkd.asarray(costs_np[models])
            subset_cov = self._extract_covariance_subset(models)

            # Create subset stat
            subset_stat = self._create_subset_stat(subset_cov)

            for est_type in self._estimator_types:
                try:
                    estimator = get_estimator(
                        est_type, subset_stat, subset_costs, bkd
                    )
                    estimator.allocate_samples(target_cost)

                    # Get variance
                    cov = estimator.optimized_covariance()
                    cov_np = bkd.to_numpy(cov)

                    # Use trace for multi-QoI
                    if cov_np.ndim == 2:
                        variance = np.trace(cov_np)
                    else:
                        variance = float(cov_np)

                    # Store result
                    key = f"{est_type}_{len(models)}models"
                    self._comparison_results[key] = {
                        "estimator_type": est_type,
                        "models": models,
                        "variance": variance,
                        "nsamples": bkd.to_numpy(estimator.nsamples_per_model()),
                    }

                    if variance < best_variance:
                        best_variance = variance
                        best_estimator = estimator
                        best_type = est_type
                        best_models = models

                except Exception as e:
                    # Skip configurations that fail
                    continue

        if best_estimator is None:
            # Fallback to MC with HF only
            hf_cost = bkd.asarray(costs_np[:1])
            hf_cov = self._extract_covariance_subset([0])
            hf_stat = self._create_subset_stat(hf_cov)
            best_estimator = get_estimator("mc", hf_stat, hf_cost, bkd)
            best_estimator.allocate_samples(target_cost)
            best_type = "mc"
            best_models = [0]

        self._best_estimator = best_estimator
        self._best_type = best_type
        self._best_models = best_models

    def _generate_model_subsets(self) -> List[List[int]]:
        """Generate model subsets to try."""
        subsets = []
        nmodels = self._nmodels
        max_n = self._max_nmodels

        # Always try using all models up to max
        all_models = list(range(min(nmodels, max_n)))
        subsets.append(all_models)

        # Try subsets of increasing size
        for n in range(2, max_n + 1):
            if n <= nmodels:
                subset = list(range(n))
                if subset not in subsets:
                    subsets.append(subset)

        # Single model (MC)
        if self._require_hf:
            if [0] not in subsets:
                subsets.append([0])

        return subsets

    def _extract_covariance_subset(self, models: List[int]) -> Array:
        """Extract covariance matrix for model subset."""
        bkd = self._bkd
        cov = self._stat.cov()
        cov_np = bkd.to_numpy(cov)
        nqoi = self._stat.nqoi()

        if nqoi == 1:
            # Simple case: cov is nmodels x nmodels
            subset_cov = cov_np[np.ix_(models, models)]
        else:
            # Multi-QoI: cov is (nmodels*nqoi) x (nmodels*nqoi)
            indices = []
            for m in models:
                indices.extend(range(m * nqoi, (m + 1) * nqoi))
            subset_cov = cov_np[np.ix_(indices, indices)]

        return bkd.asarray(subset_cov)

    def _create_subset_stat(self, cov: Array) -> StatisticWithDiscrepancyProtocol:
        """Create a new stat instance with subset covariance."""
        from pyapprox.typing.stats.statistics.mean import MultiOutputMean
        from pyapprox.typing.stats.statistics.variance import MultiOutputVariance
        from pyapprox.typing.stats.statistics.mean_variance import (
            MultiOutputMeanAndVariance,
        )

        bkd = self._bkd
        nqoi = self._stat.nqoi()

        # Create same type as original stat
        stat_type = type(self._stat)

        if stat_type.__name__ == "MultiOutputMean":
            new_stat = MultiOutputMean(nqoi=nqoi, bkd=bkd)
        elif stat_type.__name__ == "MultiOutputVariance":
            new_stat = MultiOutputVariance(nqoi=nqoi, bkd=bkd)
        elif stat_type.__name__ == "MultiOutputMeanAndVariance":
            new_stat = MultiOutputMeanAndVariance(nqoi=nqoi, bkd=bkd)
        else:
            # Default to mean
            new_stat = MultiOutputMean(nqoi=nqoi, bkd=bkd)

        new_stat.set_pilot_quantities(cov)
        return new_stat

    def best_estimator(self) -> EstimatorProtocol[Array]:
        """Return the best estimator.

        Raises
        ------
        ValueError
            If allocate_samples has not been called.
        """
        if self._best_estimator is None:
            raise ValueError("Call allocate_samples first.")
        return self._best_estimator

    def best_estimator_type(self) -> str:
        """Return the type of the best estimator."""
        if self._best_type is None:
            raise ValueError("Call allocate_samples first.")
        return self._best_type

    def best_models(self) -> List[int]:
        """Return the model indices used by the best estimator."""
        if self._best_models is None:
            raise ValueError("Call allocate_samples first.")
        return self._best_models

    def comparison_results(self) -> Dict[str, Any]:
        """Return comparison results for all tried configurations."""
        return self._comparison_results

    def nsamples_per_model(self) -> Array:
        """Return samples per model for best estimator."""
        return self.best_estimator().nsamples_per_model()

    def optimized_covariance(self) -> Array:
        """Return covariance of the best estimator."""
        return self.best_estimator().optimized_covariance()

    def __call__(self, values: List[Array]) -> Array:
        """Compute estimate using best estimator.

        Parameters
        ----------
        values : List[Array]
            Model outputs for the models in best_models().

        Returns
        -------
        Array
            Estimated statistic.
        """
        return self.best_estimator()(values)

    def __repr__(self) -> str:
        if self._best_type is None:
            return "BestEstimator(not allocated)"
        return (
            f"BestEstimator(type={self._best_type!r}, "
            f"models={self._best_models})"
        )
