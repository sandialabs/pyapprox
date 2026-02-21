"""Convenience functions for solving OED problems."""

from typing import Literal, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.objective.factory import (
    create_kl_oed_objective_from_data,
)
from pyapprox.typing.expdesign.objective.prediction_factory import (
    DeviationType,
    RiskType,
    create_prediction_oed_objective,
)
from .relaxed import RelaxedKLOEDSolver, RelaxedOEDSolver, RelaxedOEDConfig


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
    deviation_type: DeviationType = "stdev",
    risk_type: RiskType = "mean",
    noise_stat_type: RiskType = "mean",
    alpha: float = 0.5,
    delta: float = 100.0,
    config: Optional[RelaxedOEDConfig] = None,
) -> Tuple[Array, float]:
    """Solve prediction OED problem in one function call.

    Convenience function that creates a prediction OED objective and solves
    in one step using RelaxedOEDSolver.

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
    config : RelaxedOEDConfig, optional
        Solver configuration.

    Returns
    -------
    optimal_weights : Array
        Optimal design weights. Shape: (nobs, 1)
    optimal_value : float
        Objective value at optimal design.
    """
    objective = create_prediction_oed_objective(
        noise_variances, outer_shapes, inner_shapes,
        latent_samples, qoi_vals, bkd,
        deviation_type=deviation_type,
        risk_type=risk_type,
        noise_stat_type=noise_stat_type,
        alpha=alpha,
        delta=delta,
    )
    solver = RelaxedOEDSolver(objective, config)
    return solver.solve()
