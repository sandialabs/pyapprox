"""
Factory functions for prediction OED.

Provides convenience functions for creating prediction OED objectives
with different deviation measures (StdDev, Entropic, AVaR).
"""

from typing import Any, Dict, Optional

from pyapprox.risk import (
    SampleAverageEntropicRisk,
    SampleAverageMean,
    SampleAverageMeanPlusStdev,
    SampleAverageSmoothedAVaR,
    SampleAverageStdev,
    SampleAverageVariance,
    SampleStatistic,
)
from pyapprox.util.backends.protocols import Array, Backend

from ..deviation import (
    AVaRDeviationMeasure,
    DeviationMeasure,
    EntropicDeviationMeasure,
    StandardDeviationMeasure,
)
from ..likelihood import GaussianOEDInnerLoopLikelihood
from .prediction_objective import PredictionOEDObjective


def create_deviation_measure(
    deviation_type: str,
    npred: int,
    bkd: Backend[Array],
    alpha: float = 0.5,
    delta: float = 100.0,
) -> DeviationMeasure[Array]:
    """Create a deviation measure.

    Parameters
    ----------
    deviation_type : {"stdev", "entropic", "avar"}
        Type of deviation measure.
    npred : int
        Number of prediction QoIs.
    bkd : Backend[Array]
        Computational backend.
    alpha : float, optional
        Risk parameter for entropic or AVaR measures. Default: 0.5
        For entropic: controls exponential sensitivity.
        For AVaR: risk level (e.g., 0.8 means focus on top 20% tail).
    delta : float, optional
        Smoothing parameter for AVaR. Default: 100.0

    Returns
    -------
    DeviationMeasure[Array]
        The configured deviation measure.

    Notes
    -----
    - "stdev": Standard deviation, sqrt(Var[qoi | obs])
    - "entropic": (1/alpha)*log(E[exp(alpha*qoi)|obs]) - E[qoi|obs]
    - "avar": AVaR_alpha[qoi | obs] - E[qoi | obs]
      (uses smoothed approximation, requires PyTorch for gradients)
    """
    if deviation_type == "stdev":
        return StandardDeviationMeasure(npred, bkd)
    elif deviation_type == "entropic":
        return EntropicDeviationMeasure(npred, alpha, bkd)
    elif deviation_type == "avar":
        return AVaRDeviationMeasure(npred, alpha, bkd, delta)
    else:
        raise ValueError(
            f"Unknown deviation type: {deviation_type}. "
            f"Expected one of: stdev, entropic, avar"
        )


def create_risk_measure(
    risk_type: str,
    bkd: Backend[Array],
    **kwargs: Any,
) -> SampleStatistic[Array]:
    """Create a risk measure / sample statistic.

    Parameters
    ----------
    risk_type : str
        Type of risk measure. Supported values and corresponding classes:

        - ``"mean"``: `SampleAverageMean` — expected value (risk-neutral).
        - ``"variance"``: `SampleAverageVariance` — sample variance.
        - ``"stdev"``: `SampleAverageStdev` — sample standard deviation.
        - ``"entropic"``: `SampleAverageEntropicRisk` — entropic risk measure.
          Extra keyword: ``alpha`` (float, default 0.5).
        - ``"avar"``: `SampleAverageSmoothedAVaR` — smoothed Average Value at
          Risk. Extra keywords: ``alpha`` (float, default 0.5),
          ``delta`` (float, default 100000).
        - ``"mean_stdev"``: `SampleAverageMeanPlusStdev` — mean plus scaled
          standard deviation. Extra keyword: ``safety_factor`` (float,
          default 1.0).

    bkd : Backend[Array]
        Computational backend.
    **kwargs
        Extra parameters forwarded to the statistic constructor (see above).

    Returns
    -------
    SampleStatistic[Array]
        The configured risk measure.
    """
    if risk_type == "mean":
        return SampleAverageMean(bkd)
    elif risk_type == "variance":
        return SampleAverageVariance(bkd)
    elif risk_type == "stdev":
        return SampleAverageStdev(bkd)
    elif risk_type == "entropic":
        alpha = kwargs.get("alpha", 0.5)
        return SampleAverageEntropicRisk(alpha, bkd)
    elif risk_type == "avar":
        alpha = kwargs.get("alpha", 0.5)
        delta = kwargs.get("delta", 100000)
        return SampleAverageSmoothedAVaR(alpha, bkd, delta=delta)
    elif risk_type == "mean_stdev":
        safety_factor = kwargs.get("safety_factor", 1.0)
        return SampleAverageMeanPlusStdev(safety_factor, bkd)
    else:
        raise ValueError(
            f"Unknown risk type: {risk_type}. "
            f"Expected one of: mean, variance, stdev, entropic, avar, "
            f"mean_stdev"
        )


def create_prediction_oed_objective(
    noise_variances: Array,
    outer_shapes: Array,
    inner_shapes: Array,
    latent_samples: Array,
    qoi_vals: Array,
    bkd: Backend[Array],
    deviation_type: str = "stdev",
    risk_type: str = "mean",
    noise_stat_type: str = "mean",
    alpha: float = 0.5,
    delta: float = 100.0,
    deviation_measure: Optional[DeviationMeasure[Array]] = None,
    risk_measure: Optional[SampleStatistic[Array]] = None,
    noise_stat: Optional[SampleStatistic[Array]] = None,
    outer_quad_weights: Optional[Array] = None,
    inner_quad_weights: Optional[Array] = None,
    qoi_quad_weights: Optional[Array] = None,
    risk_kwargs: Optional[Dict[str, Any]] = None,
    noise_stat_kwargs: Optional[Dict[str, Any]] = None,
) -> PredictionOEDObjective[Array]:
    """Create a prediction OED objective from data arrays.

    Convenience factory function that creates all components and the
    objective in one step.

    Components can be specified either as string types (which are mapped to
    classes via ``create_deviation_measure`` and ``create_risk_measure``) or
    as pre-configured objects. Object overrides take priority over strings.

    Parameters
    ----------
    noise_variances : Array
        Base noise variances. Shape: (nobs,)
    outer_shapes : Array
        Model outputs for outer samples. Shape: (nobs, nouter)
    inner_shapes : Array
        Model outputs for inner samples. Shape: (nobs, ninner)
    latent_samples : Array
        Latent noise samples. Shape: (nobs, nouter)
    qoi_vals : Array
        QoI values at inner samples. Shape: (ninner, npred)
    bkd : Backend[Array]
        Computational backend.
    deviation_type : str, optional
        Type of deviation measure (see ``create_deviation_measure``).
        Ignored if ``deviation_measure`` is provided. Default: ``"stdev"``
    risk_type : str, optional
        Risk measure over predictions (see ``create_risk_measure``).
        Ignored if ``risk_measure`` is provided. Default: ``"mean"``
    noise_stat_type : str, optional
        Statistic over data realizations (see ``create_risk_measure``).
        Ignored if ``noise_stat`` is provided. Default: ``"mean"``
    alpha : float, optional
        Risk parameter for ``create_deviation_measure`` only (entropic/AVaR
        deviation). Not used for risk or noise stat — use ``risk_kwargs``
        or ``noise_stat_kwargs`` for those. Default: 0.5
    delta : float, optional
        Smoothing parameter for ``create_deviation_measure`` only (AVaR
        deviation). Default: 100.0
    deviation_measure : DeviationMeasure, optional
        Pre-configured deviation measure. Overrides ``deviation_type``.
    risk_measure : SampleStatistic, optional
        Pre-configured risk measure. Overrides ``risk_type``.
    noise_stat : SampleStatistic, optional
        Pre-configured noise statistic. Overrides ``noise_stat_type``.
    outer_quad_weights : Array, optional
        Quadrature weights for outer expectation. Shape: (nouter,)
    inner_quad_weights : Array, optional
        Quadrature weights for evidence integration. Shape: (ninner,)
    qoi_quad_weights : Array, optional
        Quadrature weights for prediction aggregation. Shape: (1, npred)
    risk_kwargs : dict, optional
        Extra keyword arguments forwarded to ``create_risk_measure`` when
        constructing the risk measure from ``risk_type``.
    noise_stat_kwargs : dict, optional
        Extra keyword arguments forwarded to ``create_risk_measure`` when
        constructing the noise statistic from ``noise_stat_type``.

    Returns
    -------
    PredictionOEDObjective[Array]
        The configured prediction OED objective.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.expdesign import create_prediction_oed_objective
    >>>
    >>> bkd = NumpyBkd()
    >>> nobs, ninner, nouter, npred = 3, 20, 10, 2
    >>>
    >>> noise_variances = bkd.asarray([0.1, 0.15, 0.2])
    >>> outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
    >>> inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
    >>> latent_samples = bkd.asarray(np.random.randn(nobs, nouter))
    >>> qoi_vals = bkd.asarray(np.random.randn(ninner, npred))
    >>>
    >>> objective = create_prediction_oed_objective(
    ...     noise_variances, outer_shapes, inner_shapes,
    ...     latent_samples, qoi_vals, bkd,
    ...     deviation_type="stdev", risk_type="mean"
    ... )
    >>>
    >>> weights = bkd.ones((nobs, 1))
    >>> value = objective(weights)  # Shape: (1, 1)
    """
    npred = qoi_vals.shape[1]

    # Create components — object overrides take priority
    inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)

    if deviation_measure is None:
        deviation_measure = create_deviation_measure(
            deviation_type, npred, bkd, alpha, delta
        )

    if risk_measure is None:
        risk_measure = create_risk_measure(
            risk_type, bkd, **(risk_kwargs or {})
        )

    if noise_stat is None:
        noise_stat = create_risk_measure(
            noise_stat_type, bkd, **(noise_stat_kwargs or {})
        )

    return PredictionOEDObjective(
        inner_likelihood,
        outer_shapes,
        latent_samples,
        inner_shapes,
        qoi_vals,
        deviation_measure,
        risk_measure,
        noise_stat,
        outer_quad_weights,
        inner_quad_weights,
        qoi_quad_weights,
        bkd,
    )


__all__ = [
    "create_deviation_measure",
    "create_risk_measure",
    "create_prediction_oed_objective",
]
