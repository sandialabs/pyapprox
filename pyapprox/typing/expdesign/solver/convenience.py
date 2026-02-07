"""Convenience functions for solving OED problems."""

from typing import Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.objective.factory import (
    create_kl_oed_objective_from_data,
)
from .relaxed import RelaxedKLOEDSolver, RelaxedOEDConfig


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
