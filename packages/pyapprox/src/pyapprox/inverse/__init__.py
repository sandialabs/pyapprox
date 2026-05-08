"""
Inverse problems module for pyapprox.

This module provides tools for Bayesian inverse problems including:
- Conjugate prior solvers for exact posterior computation
- Laplace approximation for nonlinear models
- Gaussian pushforward for uncertainty propagation
- Log unnormalized posterior for MAP estimation and MCMC
- MCMC samplers (Metropolis-Hastings, HMC)

Subpackages
-----------
protocols
    Core interfaces for inverse problem components.
conjugate
    Conjugate prior solvers (Gaussian, Beta, Dirichlet).
laplace
    True Laplace approximation for nonlinear models.
pushforward
    Gaussian pushforward through linear models.
posterior
    Log unnormalized posterior utilities.
sampling
    MCMC samplers.

Examples
--------
>>> import numpy as np
>>> from pyapprox.util.backends.numpy import NumpyBkd
>>> from pyapprox.inverse import DenseGaussianConjugatePosterior

Solve a linear inverse problem with Gaussian conjugate prior:

>>> bkd = NumpyBkd()
>>> # Define linear observation model y = A @ x + noise
>>> A = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]])
>>> prior_mean = np.zeros((2, 1))
>>> prior_cov = np.eye(2)
>>> noise_cov = 0.1 * np.eye(3)
>>> solver = DenseGaussianConjugatePosterior(
...     A, prior_mean, prior_cov, noise_cov, bkd
... )
>>> observations = np.array([[1.0], [1.5], [2.0]])
>>> solver.compute(observations)
>>> posterior_mean = solver.posterior_mean()
"""

# Protocols
# Bayesian networks
from .bayesnet import (
    GaussianFactor,
    GaussianNetwork,
    cond_prob_variable_elimination,
)

# Conjugate prior solvers
from .conjugate import (
    BetaConjugatePosterior,
    DenseGaussianConjugatePosterior,
    DirichletConjugatePosterior,
)

# Laplace approximation
from .laplace import (
    ApplyNegLogLikelihoodHessian,
    DenseLaplacePosterior,
    LowRankLaplacePosterior,
    PriorConditionedHessianMatVec,
)

# Posterior utilities
from .posterior import (
    LogUnNormalizedPosterior,
)
from .protocols import (
    ConjugatePosteriorProtocol,
    GaussianConjugatePosteriorProtocol,
    GaussianPushforwardProtocol,
    HessianMatVecOperatorProtocol,
    LaplacePosteriorProtocol,
    LogUnNormalizedPosteriorProtocol,
    ObservationOperatorProtocol,
    ObservationOperatorWithHessianProtocol,
    ObservationOperatorWithJacobianProtocol,
)

# Pushforward
from .pushforward import (
    DenseGaussianPrediction,
    GaussianPushforward,
)

# Sampling
from .sampling import (
    AdaptiveMetropolisSampler,
    DelayedRejectionAdaptiveMetropolis,
    HamiltonianMonteCarlo,
    MCMCDiagnostics,
    MCMCResult,
    MetropolisHastingsSampler,
    autocorrelation,
    compute_diagnostics,
    effective_sample_size,
    integrated_autocorrelation_time,
    rhat,
)

__all__ = [
    # Protocols
    "ConjugatePosteriorProtocol",
    "GaussianConjugatePosteriorProtocol",
    "ObservationOperatorProtocol",
    "ObservationOperatorWithJacobianProtocol",
    "ObservationOperatorWithHessianProtocol",
    "LaplacePosteriorProtocol",
    "HessianMatVecOperatorProtocol",
    "LogUnNormalizedPosteriorProtocol",
    "GaussianPushforwardProtocol",
    # Conjugate prior solvers
    "DenseGaussianConjugatePosterior",
    "BetaConjugatePosterior",
    "DirichletConjugatePosterior",
    # Pushforward
    "GaussianPushforward",
    "DenseGaussianPrediction",
    # Laplace approximation
    "ApplyNegLogLikelihoodHessian",
    "PriorConditionedHessianMatVec",
    "DenseLaplacePosterior",
    "LowRankLaplacePosterior",
    # Posterior utilities
    "LogUnNormalizedPosterior",
    # Sampling
    "MetropolisHastingsSampler",
    "AdaptiveMetropolisSampler",
    "HamiltonianMonteCarlo",
    "DelayedRejectionAdaptiveMetropolis",
    "MCMCResult",
    "autocorrelation",
    "effective_sample_size",
    "integrated_autocorrelation_time",
    "rhat",
    "MCMCDiagnostics",
    "compute_diagnostics",
    # Bayesian networks
    "GaussianFactor",
    "GaussianNetwork",
    "cond_prob_variable_elimination",
]
