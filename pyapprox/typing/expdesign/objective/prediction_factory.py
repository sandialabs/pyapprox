"""
Factory functions for prediction OED.

Provides convenience functions for creating prediction OED objectives
with different deviation measures (StdDev, Entropic, AVaR).
"""

from typing import Literal, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend

from ..likelihood import GaussianOEDInnerLoopLikelihood
from .prediction_objective import PredictionOEDObjective
from ..deviation import (
    DeviationMeasure,
    StandardDeviationMeasure,
    EntropicDeviationMeasure,
    AVaRDeviationMeasure,
)
from ..statistics import (
    SampleStatistic,
    SampleAverageMean,
    SampleAverageVariance,
)


DeviationType = Literal["stdev", "entropic", "avar"]
RiskType = Literal["mean", "variance"]


def create_deviation_measure(
    deviation_type: DeviationType,
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
    risk_type: RiskType,
    bkd: Backend[Array],
) -> SampleStatistic[Array]:
    """Create a risk measure statistic.

    Parameters
    ----------
    risk_type : {"mean", "variance"}
        Type of risk measure.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    SampleStatistic[Array]
        The configured risk measure.

    Notes
    -----
    - "mean": Expected value (risk-neutral)
    - "variance": Sample variance (risk-averse)
    """
    if risk_type == "mean":
        return SampleAverageMean(bkd)
    elif risk_type == "variance":
        return SampleAverageVariance(bkd)
    else:
        raise ValueError(
            f"Unknown risk type: {risk_type}. "
            f"Expected one of: mean, variance"
        )


def create_prediction_oed_objective(
    noise_variances: Array,
    outer_shapes: Array,
    inner_shapes: Array,
    latent_samples: Array,
    qoi_vals: Array,
    bkd: Backend[Array],
    deviation_type: DeviationType = "stdev",
    risk_type: RiskType = "mean",
    noise_stat_type: RiskType = "mean",
    alpha: float = 0.5,
    delta: float = 100.0,
    outer_quad_weights: Optional[Array] = None,
    inner_quad_weights: Optional[Array] = None,
    qoi_quad_weights: Optional[Array] = None,
) -> PredictionOEDObjective[Array]:
    """Create a prediction OED objective from data arrays.

    Convenience factory function that creates all components and the
    objective in one step.

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
    deviation_type : {"stdev", "entropic", "avar"}, optional
        Type of deviation measure. Default: "stdev"
    risk_type : {"mean", "variance"}, optional
        Risk measure over predictions. Default: "mean"
    noise_stat_type : {"mean", "variance"}, optional
        Statistic over data realizations. Default: "mean"
    alpha : float, optional
        Risk parameter for entropic/AVaR. Default: 0.5
    delta : float, optional
        Smoothing parameter for AVaR. Default: 100.0
    outer_quad_weights : Array, optional
        Quadrature weights for outer expectation. Shape: (nouter,)
    inner_quad_weights : Array, optional
        Quadrature weights for evidence integration. Shape: (ninner,)
    qoi_quad_weights : Array, optional
        Quadrature weights for prediction aggregation. Shape: (1, npred)

    Returns
    -------
    PredictionOEDObjective[Array]
        The configured prediction OED objective.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.expdesign import create_prediction_oed_objective
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

    # Create components
    inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
    deviation_measure = create_deviation_measure(
        deviation_type, npred, bkd, alpha, delta
    )
    risk_measure = create_risk_measure(risk_type, bkd)
    noise_stat = create_risk_measure(noise_stat_type, bkd)

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
