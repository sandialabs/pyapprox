"""
Factory functions for creating OED objectives.

This module provides factory functions that simplify the creation of
KLOEDObjective instances with proper quadrature setup. The factory
supports any forward model satisfying FunctionProtocol, enabling
both linear and nonlinear models.
"""

#TODO: This factory is only for KL based OED modify file doc string and rename file name

from typing import Optional

from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective.kl_objective import KLOEDObjective
from pyapprox.expdesign.quadrature.strategies import get_sampler
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend


def create_kl_oed_objective(
    prior: IndependentJoint[Array],
    forward_model: FunctionProtocol[Array],
    noise_variances: Array,
    bkd: Backend[Array],
    outer_sampler_type: str = "gauss",
    inner_sampler_type: str = "gauss",
    nouter_approx: int = 100000,
    ninner_approx: int = 1000,
    outer_seed: Optional[int] = None,
    inner_seed: Optional[int] = None,
) -> KLOEDObjective[Array]:
    """Create a KLOEDObjective with specified quadrature.

    This factory function handles the joint sampling required for correct
    OED computation. The outer loop requires joint samples from both the
    prior (nparams dimensions) and observation noise (nobs dimensions).
    The latent noise samples are always standard normal N(0,1) for the
    reparameterization trick.

    The forward model can be any function satisfying FunctionProtocol,
    enabling both linear and nonlinear models.

    Parameters
    ----------
    prior : IndependentJoint[Array]
        Prior distribution over parameters.
    forward_model : FunctionProtocol[Array]
        Forward model mapping parameters to observations.
        Must satisfy: nvars() == prior.nvars()
        The number of observations is determined by nqoi().
        Input shape: (nparams, nsamples), Output shape: (nobs, nsamples)
    noise_variances : Array
        Observation noise variances. Shape: (nobs,)
    bkd : Backend[Array]
        Computational backend.
    outer_sampler_type : str, optional
        Sampler type for outer loop. Options: "gauss", "mc", "halton", "sobol".
        Default "gauss".
    inner_sampler_type : str, optional
        Sampler type for inner loop. Options: "gauss", "mc", "halton", "sobol".
        Default "gauss".
    nouter_approx : int, optional
        Approximate number of outer loop samples. For Gauss quadrature,
        actual count is npoints_1d^(nparams+nobs). Default 100000.
    ninner_approx : int, optional
        Approximate number of inner loop samples. For Gauss quadrature,
        actual count is npoints_1d^nparams. Default 1000.
    outer_seed : int, optional
        Random seed for outer loop sampling. Ignored for Gauss quadrature.
    inner_seed : int, optional
        Random seed for inner loop sampling. Ignored for Gauss quadrature.

    Returns
    -------
    KLOEDObjective[Array]
        The configured KL-OED objective.

    Raises
    ------
    ValueError
        If forward_model.nvars() doesn't match prior.nvars().
    TypeError
        If Gauss quadrature is requested but marginals are not Gaussian.
    """
    nobs = forward_model.nqoi()
    nparams = prior.nvars()

    # Validate dimensions
    if forward_model.nvars() != nparams:
        raise ValueError(
            f"forward_model.nvars() ({forward_model.nvars()}) must match "
            f"prior.nvars() ({nparams})"
        )

    # Build joint distribution for outer loop: [prior, latent]
    # Latent is always standard normal N(0,1) for reparameterization trick
    latent_marginals = [GaussianMarginal(0.0, 1.0, bkd) for _ in range(nobs)]
    outer_joint = IndependentJoint(list(prior.marginals()) + latent_marginals, bkd)

    # Get sampler strategies
    outer_strategy = get_sampler(outer_sampler_type)()
    inner_strategy = get_sampler(inner_sampler_type)()

    # Sample from outer joint distribution
    outer_samples, outer_weights = outer_strategy.sample(
        outer_joint, nouter_approx, bkd, outer_seed
    )

    # Sample from prior for inner loop
    inner_samples, inner_weights = inner_strategy.sample(
        prior, ninner_approx, bkd, inner_seed
    )

    # Split outer samples into prior parameters and latent noise
    theta_outer = outer_samples[:nparams, :]
    latent_samples = outer_samples[nparams:, :]

    # Compute forward model outputs (shapes) - works for ANY forward model
    outer_shapes = forward_model(theta_outer)  # (nobs, nouter)
    inner_shapes = forward_model(inner_samples)  # (nobs, ninner)

    # Create likelihood and objective
    inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, bkd)

    return KLOEDObjective(
        inner_likelihood,
        outer_shapes,
        latent_samples,
        inner_shapes,
        outer_weights,
        inner_weights,
        bkd,
    )


def create_kl_oed_objective_from_data(
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
    in one step from pre-computed model outputs.

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
