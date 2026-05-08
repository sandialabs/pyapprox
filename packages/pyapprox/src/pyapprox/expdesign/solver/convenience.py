"""Convenience functions for solving OED problems."""

from typing import Any, Dict, Optional, Tuple

from pyapprox.expdesign.deviation import DeviationMeasure
from pyapprox.expdesign.objective.factory import (
    create_kl_oed_objective_from_data,
)
from pyapprox.expdesign.objective.prediction_factory import (
    create_prediction_oed_objective,
)
from pyapprox.risk.base import SampleStatistic
from pyapprox.util.backends.protocols import Array, Backend

from .relaxed import RelaxedKLOEDSolver, RelaxedOEDConfig, RelaxedOEDSolver


def solve_kl_oed(
    noise_variances: Array,
    outer_shapes: Array,
    inner_shapes: Array,
    latent_samples: Array,
    bkd: Backend[Array],
    config: Optional[RelaxedOEDConfig] = None,
) -> Tuple[Array, float]:
    """Solve KL-OED problem in one function call.

    Convenience function that creates objective and solves in one step.

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
    bkd : Backend[Array]
        Computational backend.
    config : RelaxedOEDConfig, optional
        Solver configuration.

    Returns
    -------
    optimal_weights : Array
        Optimal design weights. Shape: (nobs, 1)
    optimal_eig : float
        Expected information gain at optimal design.
    """
    objective = create_kl_oed_objective_from_data(
        noise_variances, outer_shapes, inner_shapes, latent_samples, bkd
    )
    solver = RelaxedKLOEDSolver(objective, config)
    return solver.solve()


def solve_prediction_oed(
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
    config: Optional[RelaxedOEDConfig] = None,
    deviation_measure: Optional[DeviationMeasure[Array]] = None,
    risk_measure: Optional[SampleStatistic[Array]] = None,
    noise_stat: Optional[SampleStatistic[Array]] = None,
    risk_kwargs: Optional[Dict[str, Any]] = None,
    noise_stat_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Array, float]:
    """Solve prediction OED problem in one function call.

    Convenience function that creates a prediction OED objective and solves
    in one step using RelaxedOEDSolver. All parameters related to deviation,
    risk, and noise statistic are forwarded to
    ``create_prediction_oed_objective``.

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
        Type of deviation measure. Default: "stdev"
    risk_type : str, optional
        Risk measure over predictions. Default: "mean"
    noise_stat_type : str, optional
        Statistic over data realizations. Default: "mean"
    alpha : float, optional
        Risk parameter for deviation measure only. Default: 0.5
    delta : float, optional
        Smoothing parameter for deviation measure only. Default: 100.0
    config : RelaxedOEDConfig, optional
        Solver configuration.
    deviation_measure : DeviationMeasure, optional
        Pre-configured deviation measure. Overrides ``deviation_type``.
    risk_measure : SampleStatistic, optional
        Pre-configured risk measure. Overrides ``risk_type``.
    noise_stat : SampleStatistic, optional
        Pre-configured noise statistic. Overrides ``noise_stat_type``.
    risk_kwargs : dict, optional
        Extra kwargs for ``create_risk_measure`` (risk measure).
    noise_stat_kwargs : dict, optional
        Extra kwargs for ``create_risk_measure`` (noise statistic).

    Returns
    -------
    optimal_weights : Array
        Optimal design weights. Shape: (nobs, 1)
    optimal_value : float
        Objective value at optimal design.
    """
    objective = create_prediction_oed_objective(
        noise_variances,
        outer_shapes,
        inner_shapes,
        latent_samples,
        qoi_vals,
        bkd,
        deviation_type=deviation_type,
        risk_type=risk_type,
        noise_stat_type=noise_stat_type,
        alpha=alpha,
        delta=delta,
        deviation_measure=deviation_measure,
        risk_measure=risk_measure,
        noise_stat=noise_stat,
        risk_kwargs=risk_kwargs,
        noise_stat_kwargs=noise_stat_kwargs,
    )
    solver = RelaxedOEDSolver(objective, config)
    return solver.solve()
