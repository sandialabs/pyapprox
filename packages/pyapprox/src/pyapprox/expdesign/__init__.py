# TODO: Should we provide examples here. If we are going to
# provide examples here we should include example using local
# OED. Should we split module into two submodules, local
# (frequentist oed and Bayesian OED or even further, local,
# Bayes oed for params (currently we only support KL but we will
# add additional classes in the future), and bayes oed for
# prediction?

"""
Experimental Design module for pyapprox.

This module provides optimal experimental design (OED) functionality,
including:
- KL-OED: Expected information gain (KL divergence) based designs
- Prediction OED: Designs that minimize expected deviation in predictions
- Local OED: Classical optimal designs for linear regression (D, A, C, I, G, R-optimal)

Example
-------
Basic KL-OED workflow:

    >>> from pyapprox.expdesign import (
    ...     GaussianOEDInnerLoopLikelihood,
    ...     KLOEDObjective,
    ...     RelaxedKLOEDSolver,
    ...     RelaxedOEDConfig,
    ... )
    >>> from pyapprox.util.backends.numpy import NumpyBkd
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

Basic Prediction OED workflow:

    >>> from pyapprox.expdesign import create_prediction_oed_objective
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
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

Submodules
----------
protocols
    Protocol definitions for OED components.
likelihood
    OED-specific likelihood wrappers.
evidence
    Evidence computation for Bayesian OED.
objective
    OED objective functions (KL-OED, Prediction OED).
deviation
    Deviation measures for prediction OED (StdDev, Entropic, AVaR).
statistics
    Sample statistics for prediction OED (Mean, Variance, etc.).
quadrature
    Quadrature samplers for expectation computation.
solver
    OED optimization solvers.
local
    Local OED for linear regression (D, A, C, I, G, R-optimal criteria).
"""

# Analytical utilities (conjugate priors)
from pyapprox.risk import (
    SampleAverageEntropicRisk,
    SampleAverageMean,
    SampleAverageSmoothedAVaR,
    SampleAverageStdev,
    SampleAverageVariance,
    SampleStatistic,
)

from .analytical import (
    ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev,
    ConjugateGaussianOEDDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDDataMeanQoIMeanEntropicDev,
    ConjugateGaussianOEDDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDExpectedInformationGain,
    ConjugateGaussianOEDExpectedPushforwardKLDivergence,
    ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev,
    LogNormalDataMeanQoIAVaRStdDevObjective,
)

# Data generation and management
from .data import OEDDataGenerator, OEDDataManager
from .deviation import (
    AVaRDeviationMeasure,
    DeviationMeasure,
    EntropicDeviationMeasure,
    StandardDeviationMeasure,
)
from .evidence import Evidence, LogEvidence
from .likelihood import (
    GaussianOEDInnerLoopLikelihood,
    GaussianOEDOuterLoopLikelihood,
)

# Local OED (linear regression design)
from .local import (
    AOptimalCriterion,
    AVaRLocalOEDSolver,
    COptimalCriterion,
    # Criteria
    DOptimalCriterion,
    GOptimalCriterion,
    IOptimalCriterion,
    # Design matrices
    LeastSquaresDesignMatrices,
    MinimaxLocalOEDSolver,
    QuantileDesignMatrices,
    ROptimalCriterion,
    # Solvers
    ScipyLocalOEDSolver,
    create_criterion,
    # Factory
    create_design_matrices,
    create_solver,
)
from .objective import (
    DOptimalLinearModelObjective,
    KLOEDObjective,
    PredictionOEDObjective,
    create_deviation_measure,
    create_kl_oed_objective,
    create_kl_oed_objective_from_data,
    create_prediction_oed_objective,
    create_risk_measure,
)
from .quadrature import (
    GaussianQuadratureSampler,
    HaltonSampler,
    MonteCarloSampler,
    OEDQuadratureSampler,
    SobolSampler,
)
from .solver import (
    BruteForceKLOEDSolver,
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    RelaxedOEDSolver,
    solve_kl_oed,
    solve_prediction_oed,
)

__all__ = [
    # Likelihood
    "GaussianOEDOuterLoopLikelihood",
    "GaussianOEDInnerLoopLikelihood",
    # Evidence
    "Evidence",
    "LogEvidence",
    # Objectives
    "KLOEDObjective",
    "PredictionOEDObjective",
    "DOptimalLinearModelObjective",
    # Analytical utilities (conjugate priors)
    "ConjugateGaussianOEDDataMeanQoIMeanStdDev",
    "ConjugateGaussianOEDDataMeanQoIMeanEntropicDev",
    "ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev",
    "ConjugateGaussianOEDDataMeanQoIAVaRStdDev",
    "ConjugateGaussianOEDExpectedInformationGain",
    "ConjugateGaussianOEDExpectedPushforwardKLDivergence",
    "ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev",
    "ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev",
    "ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev",
    "ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev",
    "LogNormalDataMeanQoIAVaRStdDevObjective",
    # Deviation measures
    "DeviationMeasure",
    "StandardDeviationMeasure",
    "EntropicDeviationMeasure",
    "AVaRDeviationMeasure",
    # Sample statistics
    "SampleStatistic",
    "SampleAverageMean",
    "SampleAverageVariance",
    "SampleAverageStdev",
    "SampleAverageEntropicRisk",
    "SampleAverageSmoothedAVaR",
    # Quadrature samplers
    "MonteCarloSampler",
    "HaltonSampler",
    "SobolSampler",
    "GaussianQuadratureSampler",
    "OEDQuadratureSampler",
    # Solvers
    "RelaxedOEDSolver",
    "RelaxedKLOEDSolver",
    "RelaxedOEDConfig",
    "BruteForceKLOEDSolver",
    # KL-OED convenience functions
    "create_kl_oed_objective",
    "create_kl_oed_objective_from_data",
    "solve_kl_oed",
    # Prediction OED convenience functions (solver)
    "solve_prediction_oed",
    # Prediction OED convenience functions
    "create_deviation_measure",
    "create_risk_measure",
    "create_prediction_oed_objective",
    # Data generation and management
    "OEDDataGenerator",
    "OEDDataManager",
    # Local OED - Design matrices
    "LeastSquaresDesignMatrices",
    "QuantileDesignMatrices",
    # Local OED - Criteria
    "DOptimalCriterion",
    "AOptimalCriterion",
    "COptimalCriterion",
    "IOptimalCriterion",
    "GOptimalCriterion",
    "ROptimalCriterion",
    # Local OED - Solvers
    "ScipyLocalOEDSolver",
    "MinimaxLocalOEDSolver",
    "AVaRLocalOEDSolver",
    # Local OED - Factory
    "create_design_matrices",
    "create_criterion",
    "create_solver",
]
