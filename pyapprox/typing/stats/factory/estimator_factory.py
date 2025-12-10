"""Estimator factory for creating multifidelity estimators.

Provides a unified interface for creating any estimator type.
"""

from typing import Generic, List, Optional, Dict, Any, TypeVar

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import (
    StatisticWithDiscrepancyProtocol,
    EstimatorProtocol,
)
from pyapprox.typing.stats.estimators import (
    MCEstimator,
    CVEstimator,
    ACVEstimator,
    GMFEstimator,
    GRDEstimator,
    GISEstimator,
    MFMCEstimator,
    MLMCEstimator,
    GroupACVEstimator,
    MLBLUEEstimator,
)


# Mapping of estimator names to classes
_ESTIMATOR_REGISTRY: Dict[str, type] = {
    "mc": MCEstimator,
    "cv": CVEstimator,
    "acv": ACVEstimator,
    "gmf": GMFEstimator,
    "grd": GRDEstimator,
    "gis": GISEstimator,
    "mfmc": MFMCEstimator,
    "mlmc": MLMCEstimator,
    "groupacv": GroupACVEstimator,
    "group_acv": GroupACVEstimator,
    "mlblue": MLBLUEEstimator,
}


def get_estimator(
    name: str,
    stat: StatisticWithDiscrepancyProtocol[Array],
    costs: Array,
    bkd: Optional[Backend[Array]] = None,
    **kwargs: Any,
) -> EstimatorProtocol[Array]:
    """Create an estimator by name.

    Factory function for creating multifidelity estimators. Supports all
    estimator types: MC, CV, ACV (GMF, GRD, GIS), MFMC, MLMC, GroupACV, MLBLUE.

    Parameters
    ----------
    name : str
        Name of the estimator. Case-insensitive. Options:
        - "mc": Monte Carlo estimator (single model)
        - "cv": Control variate estimator (two models)
        - "acv": Approximate control variate estimator
        - "gmf": Generalized multifidelity estimator
        - "grd": Generalized recursive difference estimator
        - "gis": Generalized independent samples estimator
        - "mfmc": Multifidelity Monte Carlo estimator
        - "mlmc": Multilevel Monte Carlo estimator
        - "groupacv" or "group_acv": Group ACV estimator
        - "mlblue": Multilevel Best Linear Unbiased Estimator
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate. Must have pilot quantities set.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array], optional
        Computational backend. If None, uses stat.bkd().
    **kwargs
        Additional arguments passed to the estimator constructor.

    Returns
    -------
    EstimatorProtocol[Array]
        The created estimator instance.

    Raises
    ------
    ValueError
        If the estimator name is not recognized.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([10.0, 1.0])
    >>> estimator = get_estimator("mfmc", stat, costs)
    >>> estimator.allocate_samples(target_cost=100.0)
    """
    name_lower = name.lower()

    if name_lower not in _ESTIMATOR_REGISTRY:
        available = ", ".join(sorted(_ESTIMATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown estimator: {name!r}. Available estimators: {available}"
        )

    estimator_cls = _ESTIMATOR_REGISTRY[name_lower]

    if bkd is None:
        bkd = stat.bkd()

    return estimator_cls(stat, costs, bkd, **kwargs)


def list_estimators() -> List[str]:
    """Return list of available estimator names.

    Returns
    -------
    List[str]
        Sorted list of estimator names.
    """
    return sorted(_ESTIMATOR_REGISTRY.keys())


def register_estimator(name: str, estimator_cls: type) -> None:
    """Register a custom estimator.

    Parameters
    ----------
    name : str
        Name for the estimator (case-insensitive).
    estimator_cls : type
        Estimator class to register.

    Examples
    --------
    >>> class CustomEstimator:
    ...     pass
    >>> register_estimator("custom", CustomEstimator)
    """
    _ESTIMATOR_REGISTRY[name.lower()] = estimator_cls
