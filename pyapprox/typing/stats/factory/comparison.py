"""Estimator comparison utilities.

Functions for comparing different estimator configurations.
"""

from typing import Generic, List, Dict, Any, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import (
    StatisticWithDiscrepancyProtocol,
    EstimatorProtocol,
)
from pyapprox.typing.stats.factory.estimator_factory import get_estimator


def compare_estimators(
    stat: StatisticWithDiscrepancyProtocol[Array],
    costs: Array,
    bkd: Backend[Array],
    target_cost: float,
    estimator_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple estimator types.

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate. Must have pilot quantities set.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.
    target_cost : float
        Total computational budget.
    estimator_types : List[str], optional
        Estimator types to compare. Default: all available for the model count.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Comparison results keyed by estimator type.
        Each entry contains:
        - "estimator": The estimator instance
        - "variance": Estimator variance (scalar or trace for multi-QoI)
        - "nsamples": Samples per model
        - "total_cost": Actual total cost

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([10.0, 1.0, 0.1])
    >>> results = compare_estimators(stat, costs, bkd, target_cost=100.0)
    >>> for name, info in results.items():
    ...     print(f"{name}: variance={info['variance']:.4f}")
    """
    nmodels = costs.shape[0]

    if estimator_types is None:
        estimator_types = _get_available_types(nmodels)

    results = {}

    for est_type in estimator_types:
        try:
            estimator = get_estimator(est_type, stat, costs, bkd)
            estimator.allocate_samples(target_cost)

            # Compute variance
            cov = estimator.optimized_covariance()
            cov_np = bkd.to_numpy(cov)

            if cov_np.ndim == 2:
                variance = float(np.trace(cov_np))
            else:
                variance = float(cov_np)

            # Compute actual cost
            nsamples = estimator.nsamples_per_model()
            nsamples_np = bkd.to_numpy(nsamples)
            costs_np = bkd.to_numpy(costs)
            total_cost = float(np.sum(nsamples_np * costs_np))

            results[est_type] = {
                "estimator": estimator,
                "variance": variance,
                "nsamples": nsamples_np.copy(),
                "total_cost": total_cost,
            }

        except Exception as e:
            results[est_type] = {
                "error": str(e),
            }

    return results


def _get_available_types(nmodels: int) -> List[str]:
    """Return available estimator types for given model count."""
    types = []

    # MC always available
    types.append("mc")

    if nmodels >= 2:
        types.extend(["cv", "mfmc", "mlmc", "acv", "gmf", "grd", "gis"])

    if nmodels >= 3:
        types.extend(["groupacv", "mlblue"])

    return types


def rank_estimators(
    results: Dict[str, Dict[str, Any]],
    metric: str = "variance",
) -> List[str]:
    """Rank estimators by a metric.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results from compare_estimators.
    metric : str
        Metric to rank by. Options: "variance" (default).

    Returns
    -------
    List[str]
        Estimator names ranked from best to worst.
    """
    valid_results = {
        name: info
        for name, info in results.items()
        if "error" not in info and metric in info
    }

    ranked = sorted(valid_results.keys(), key=lambda x: valid_results[x][metric])

    return ranked


def variance_reduction(
    results: Dict[str, Dict[str, Any]],
    baseline: str = "mc",
) -> Dict[str, float]:
    """Compute variance reduction relative to baseline.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results from compare_estimators.
    baseline : str
        Baseline estimator type. Default: "mc".

    Returns
    -------
    Dict[str, float]
        Variance reduction factor for each estimator.
        Values > 1 indicate improvement over baseline.
    """
    if baseline not in results or "variance" not in results[baseline]:
        return {}

    baseline_var = results[baseline]["variance"]

    reductions = {}
    for name, info in results.items():
        if "variance" in info and info["variance"] > 0:
            reductions[name] = baseline_var / info["variance"]

    return reductions
