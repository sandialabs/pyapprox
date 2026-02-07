"""
Experimental Design module for pyapprox.typing.

This module provides optimal experimental design (OED) functionality,
including:
- KL-OED: Expected information gain (KL divergence) based designs
- Prediction OED: Designs that minimize expected deviation in predictions
- Local OED: Classical optimal designs for linear regression (D, A, C, I, G, R-optimal)

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

Basic Prediction OED workflow:

    >>> from pyapprox.typing.expdesign import create_prediction_oed_objective
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
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

from .likelihood import (
    GaussianOEDOuterLoopLikelihood,
    GaussianOEDInnerLoopLikelihood,
)
from .evidence import Evidence, LogEvidence
from .objective import (
    KLOEDObjective,
    PredictionOEDObjective,
    DOptimalLinearModelObjective,
    create_kl_oed_objective,
    create_kl_oed_objective_from_data,
    create_deviation_measure,
    create_risk_measure,
    create_prediction_oed_objective,
)
from .deviation import (
    DeviationMeasure,
    StandardDeviationMeasure,
    EntropicDeviationMeasure,
    AVaRDeviationMeasure,
)
from .statistics import (
    SampleStatistic,
    SampleAverageMean,
    SampleAverageVariance,
    SampleAverageStdev,
    SampleAverageEntropicRisk,
    SampleAverageSmoothedAVaR,
)
from .quadrature import (
    MonteCarloSampler,
    HaltonSampler,
    SobolSampler,
    GaussianQuadratureSampler,
    OEDQuadratureSampler,
)
from .protocols import QuadratureSamplerProtocol
from .solver import (
    RelaxedKLOEDSolver,
    RelaxedOEDConfig,
    BruteForceKLOEDSolver,
    solve_kl_oed,
)
# Benchmarks
from .benchmarks import LinearGaussianOEDBenchmark, NonLinearGaussianOEDBenchmark
# Diagnostics
from .diagnostics import (
    KLOEDDiagnostics,
    PredictionOEDDiagnostics,
    create_prediction_oed_diagnostics,
    register_utility,
    get_registered_utility_types,
)
# Analytical utilities (conjugate priors)
from .analytical import (
    ConjugateGaussianOEDExpectedStdDev,
    ConjugateGaussianOEDExpectedEntropicDev,
    ConjugateGaussianOEDExpectedAVaRDev,
    ConjugateGaussianOEDExpectedKLDivergence,
    ConjugateGaussianOEDForLogNormalExpectedStdDev,
    ConjugateGaussianOEDForLogNormalAVaRStdDev,
)
# Local OED (linear regression design)
from .local import (
    # Design matrices
    LeastSquaresDesignMatrices,
    QuantileDesignMatrices,
    # Criteria
    DOptimalCriterion,
    AOptimalCriterion,
    COptimalCriterion,
    IOptimalCriterion,
    GOptimalCriterion,
    ROptimalCriterion,
    # Solvers
    ScipyLocalOEDSolver,
    MinimaxLocalOEDSolver,
    AVaRLocalOEDSolver,
    # Factory
    create_design_matrices,
    create_criterion,
    create_solver,
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
    # Benchmarks
    "LinearGaussianOEDBenchmark",
    "NonLinearGaussianOEDBenchmark",
    # Diagnostics
    "KLOEDDiagnostics",
    "PredictionOEDDiagnostics",
    "create_prediction_oed_diagnostics",
    "register_utility",
    "get_registered_utility_types",
    # Analytical utilities (conjugate priors)
    "ConjugateGaussianOEDExpectedStdDev",
    "ConjugateGaussianOEDExpectedEntropicDev",
    "ConjugateGaussianOEDExpectedAVaRDev",
    "ConjugateGaussianOEDExpectedKLDivergence",
    "ConjugateGaussianOEDForLogNormalExpectedStdDev",
    "ConjugateGaussianOEDForLogNormalAVaRStdDev",
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
    "QuadratureSamplerProtocol",
    "MonteCarloSampler",
    "HaltonSampler",
    "SobolSampler",
    "GaussianQuadratureSampler",
    "OEDQuadratureSampler",
    # Solvers
    "RelaxedKLOEDSolver",
    "RelaxedOEDConfig",
    "BruteForceKLOEDSolver",
    # KL-OED convenience functions
    "create_kl_oed_objective",
    "create_kl_oed_objective_from_data",
    "solve_kl_oed",
    # Prediction OED convenience functions
    "create_deviation_measure",
    "create_risk_measure",
    "create_prediction_oed_objective",
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
