"""
MCMC sampling methods for Bayesian inference.

This module provides Markov Chain Monte Carlo samplers:
- MetropolisHastingsSampler: Standard Metropolis-Hastings
- AdaptiveMetropolisSampler: Adaptive Metropolis with covariance estimation
- HamiltonianMonteCarlo: HMC with leapfrog integration
- DelayedRejectionAdaptiveMetropolis: DRAM combining DR and AM

And MCMC diagnostics:
- autocorrelation: Compute autocorrelation function
- effective_sample_size: Compute ESS = n / IAT
- integrated_autocorrelation_time: Estimate IAT using Sokal's windowing
- rhat: Compute Gelman-Rubin R-hat for multiple chains
- MCMCDiagnostics: Dataclass holding all diagnostic measures
- compute_diagnostics: Convenience function to compute all diagnostics
"""

from .diagnostics import (
    MCMCDiagnostics,
    autocorrelation,
    compute_diagnostics,
    effective_sample_size,
    integrated_autocorrelation_time,
    rhat,
)
from .dram import DelayedRejectionAdaptiveMetropolis
from .hmc import HamiltonianMonteCarlo
from .metropolis import (
    AdaptiveMetropolisSampler,
    MCMCResult,
    MetropolisHastingsSampler,
)

__all__ = [
    # Samplers
    "MetropolisHastingsSampler",
    "AdaptiveMetropolisSampler",
    "HamiltonianMonteCarlo",
    "DelayedRejectionAdaptiveMetropolis",
    "MCMCResult",
    # Diagnostics
    "autocorrelation",
    "effective_sample_size",
    "integrated_autocorrelation_time",
    "rhat",
    "MCMCDiagnostics",
    "compute_diagnostics",
]
