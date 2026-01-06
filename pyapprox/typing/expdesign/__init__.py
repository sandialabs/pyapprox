"""
Experimental Design module for pyapprox.typing.

This module provides optimal experimental design (OED) functionality,
with a focus on Bayesian OED using expected information gain (KL divergence).

Example
-------
Basic KL-OED workflow:

    >>> from pyapprox.typing.expdesign import (
    ...     GaussianOEDInnerLoopLikelihood,
    ...     KLOEDObjective,
    ...     RelaxedKLOEDSolver,
    ...     RelaxedOEDConfig,
    ... )
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>>
    >>> bkd = NumpyBkd()
    >>> nobs, ninner, nouter = 4, 50, 50
    >>>
    >>> # Define noise model
    >>> noise_variances = bkd.asarray([0.1, 0.15, 0.2, 0.12])
    >>>
    >>> # Model outputs (from forward model evaluations)
    >>> outer_shapes = bkd.asarray(np.random.randn(nobs, nouter))
    >>> inner_shapes = bkd.asarray(np.random.randn(nobs, ninner))
    >>> latent_samples = bkd.asarray(np.random.randn(nobs, nouter))
    >>>
    >>> # Create objective
    >>> likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
    >>> objective = KLOEDObjective(
    ...     likelihood, outer_shapes, latent_samples, inner_shapes,
    ...     None, None, bkd
    ... )
    >>>
    >>> # Optimize
    >>> solver = RelaxedKLOEDSolver(objective, RelaxedOEDConfig(maxiter=100))
    >>> optimal_weights, eig = solver.solve()

Submodules
----------
protocols
    Protocol definitions for OED components.
likelihood
    OED-specific likelihood wrappers.
evidence
    Evidence computation for Bayesian OED.
objective
    OED objective functions (KL-OED, etc.).
quadrature
    Quadrature samplers for expectation computation.
solver
    OED optimization solvers.
"""

from typing import Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend

from .likelihood import (
    GaussianOEDOuterLoopLikelihood,
    GaussianOEDInnerLoopLikelihood,
)
from .evidence import Evidence, LogEvidence
from .objective import KLOEDObjective
from .quadrature import (
    QuadratureSampler,
    MonteCarloSampler,
    HaltonSampler,
    GaussianQuadratureSampler,
    OEDQuadratureSampler,
)
from .solver import (
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    BruteForceKLOEDSolver,
)


def create_kl_oed_objective(
    noise_variances: Array,
    outer_shapes: Array,
    inner_shapes: Array,
    latent_samples: Array,
    bkd: Backend[Array],
    outer_quad_weights: Optional[Array] = None,
    inner_quad_weights: Optional[Array] = None,
) -> KLOEDObjective[Array]:
    """Create a KL-OED objective from data arrays.

    Convenience factory function that creates the likelihood and objective
    in one step.

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
    outer_quad_weights : Array, optional
        Quadrature weights for outer expectation. Shape: (nouter,)
    inner_quad_weights : Array, optional
        Quadrature weights for evidence integration. Shape: (ninner,)

    Returns
    -------
    KLOEDObjective[Array]
        The configured objective function.
    """
    inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)
    return KLOEDObjective(
        inner_likelihood,
        outer_shapes,
        latent_samples,
        inner_shapes,
        outer_quad_weights,
        inner_quad_weights,
        bkd,
    )


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
    objective = create_kl_oed_objective(
        noise_variances, outer_shapes, inner_shapes, latent_samples, bkd
    )
    solver = RelaxedKLOEDSolver(objective, config)
    return solver.solve()


__all__ = [
    # Likelihood
    "GaussianOEDOuterLoopLikelihood",
    "GaussianOEDInnerLoopLikelihood",
    # Evidence
    "Evidence",
    "LogEvidence",
    # Objective
    "KLOEDObjective",
    # Quadrature samplers
    "QuadratureSampler",
    "MonteCarloSampler",
    "HaltonSampler",
    "GaussianQuadratureSampler",
    "OEDQuadratureSampler",
    # Solvers
    "RelaxedKLOEDSolver",
    "RelaxedOEDConfig",
    "BruteForceKLOEDSolver",
    # Convenience functions
    "create_kl_oed_objective",
    "solve_kl_oed",
]
